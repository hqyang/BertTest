#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 09:28 2019-02-21 
@author: haiqinyang

Feature: 

Scenario: 
"""

import os

from tqdm import tqdm, trange
from src.config import args
from src.preprocess import CWS_POS, get_dataset_stored_with_dict_and_dataloader, get_eval_stored_with_dict_dataloaders
import time
from src.BERT.optimization import BertAdam
from src.metrics import outputFscoreUsedBIO, outputPOSFscoreUsedBIO

import numpy as np
import torch
from glob import glob
import pdb
import re

from src.BERT.modeling import BertConfig
# here generate exeception of Parameter config in `BertVariant(config)` should be an instance of class
# `BertConfig`. To create a models from a Google pretrained models use
# `models = BertVariant.from_pretrained(PRETRAINED_MODEL_NAME)`

from src.customize_modeling import BertMLVariantCWSPOS_with_Dict
from tensorboardX import SummaryWriter

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

TS_WRITER = SummaryWriter()

def load_CWS_POS_model(CWS_label_list, POS_label_list, args):
    if args.visible_device is not None:
        if isinstance(args.visible_device, int):
            args.visible_device = str(args.visible_device)
        elif isinstance(args.visible_device, (tuple, list)):
            args.visible_device = ','.join([str(_) for _ in args.visible_device])
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_device

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.bert_model_dir is not None:
        config_file = os.path.join(args.bert_model_dir, CONFIG_NAME)
        bert_config = BertConfig.from_json_file(config_file)
    else:
        bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.num_hidden_layers>0 and args.num_hidden_layers<bert_config.num_hidden_layers:
        bert_config.num_hidden_layers = args.num_hidden_layers

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT models was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if not args.override_output:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        else:
            #os.system("rm %s" % os.path.join(args.output_dir, '*'))
            os.system('mkdir %s' % args.output_dir+'/old')
            os.system('chmod 777 %s' % args.output_dir+'/old')
            os.system('mv %s %s' % (os.path.join(args.output_dir, '*'), args.output_dir+'/old'))


#    models = {
#        'CRF': lambda: BertCRFVariant(bert_config, len(label_list), method=args.method),
#        'Softmax': lambda: BertSoftmaxVariant(bert_config, len(label_list), method=args.method),
#    }
#    models = models[args.fclassifier]()

    model = BertMLVariantCWSPOS_with_Dict(bert_config, len(CWS_label_list), len(POS_label_list), method=args.method, \
                fclassifier=args.fclassifier, pclassifier=args.pclassifier, do_mask_as_whole=args.do_mask_as_whole, \
                dict_file=args.dict_file)

    if args.bert_model_dir is None:
        raise RuntimeError('Evaluating a random initialized models is not supported...!')
    else:
        weights_path = os.path.join(args.bert_model_dir, WEIGHTS_NAME)

        # main code copy from modeling.py line after 506
        state_dict = torch.load(weights_path)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained models: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained models not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))

    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)

    return model, device


def do_train(model, train_dataloader, optimizer, param_optimizer, device, args, eval_dataloaders):
    model.train()

    global_step = 0
    tr_times = []

    old_cws_F1 = 0.
    old_cws_Acc = 0.

    for ep in trange(int(args.num_train_epochs), desc="Epoch"):
        st = time.time()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        step = 1
        if args.dict_file is not None: # contains dictionary file
            if args.do_mask_as_whole:
                for step, (batch, batch2, input_via_dict) in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)
                    batch2 = tuple(t.to(device) for t in batch2) # for t in cand_indexes
                    input_via_dict = input_via_dict.to(device)

                    input_ids, segment_ids, input_mask = batch[:3]

                    cand_indexes, token_ids = batch2[:2]

                    label_ids, pos_label_ids = batch[3:]

                    loss = model(input_ids, segment_ids, input_mask, cand_indexes, token_ids, input_via_dict,
                                 label_ids, pos_label_ids)

                    n_gpu = torch.cuda.device_count()
                    if n_gpu > 1: # or loss.shape[0] > 1:
                        loss = loss.mean() # mean() to average on multi-gpu or multitask.
                    if args.fp16 and args.loss_scale != 1.0:
                        # rescale loss for fp16 training
                        # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                        loss = loss * args.loss_scale
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fclassifier == 'Softmax':
                        logger.info("Training loss: {:d}: {:+.2f}".format(ep, loss*1e5))
                    else:
                        logger.info("Training loss: {:d}: {:+.2f}".format(ep, loss))

                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16 or args.optimize_on_cpu:
                            if args.fp16 and args.loss_scale != 1.0:
                                # scale down gradients for fp16 training
                                for param in model.parameters():
                                    param.grad.data = param.grad.data / args.loss_scale
                            is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                            if is_nan:
                                logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                                args.loss_scale = args.loss_scale / 2
                                model.zero_grad()
                                continue
                            optimizer.step()
                            copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                        else:
                            optimizer.step()
                        model.zero_grad()
                        global_step += 1
            else: # no cand_info
                for step, (batch, input_via_dict) in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)
                    input_via_dict = input_via_dict.to(device)

                    input_ids, segment_ids, input_mask = batch[:3]

                    label_ids, pos_label_ids = batch[3:] #if len(batch[3:])>2 else batch[3]
                    loss = model(input_ids, segment_ids, input_mask, input_via_dict, label_ids, pos_label_ids)

                    n_gpu = torch.cuda.device_count()
                    if n_gpu > 1: # or loss.shape[0] > 1:
                        loss = loss.mean() # mean() to average on multi-gpu or multitask.
                    if args.fp16 and args.loss_scale != 1.0:
                        # rescale loss for fp16 training
                        # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                        loss = loss * args.loss_scale
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fclassifier == 'Softmax':
                        logger.info("Training loss: {:d}: {:+.2f}".format(ep, loss*1e5))
                    else:
                        logger.info("Training loss: {:d}: {:+.2f}".format(ep, loss))

                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16 or args.optimize_on_cpu:
                            if args.fp16 and args.loss_scale != 1.0:
                                # scale down gradients for fp16 training
                                for param in model.parameters():
                                    param.grad.data = param.grad.data / args.loss_scale
                            is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                            if is_nan:
                                logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                                args.loss_scale = args.loss_scale / 2
                                model.zero_grad()
                                continue
                            optimizer.step()
                            copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                        else:
                            optimizer.step()
                        model.zero_grad()
                        global_step += 1
        else: # no dictionary file
            if args.do_mask_as_whole:
                for step, (batch, batch2) in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)
                    batch2 = tuple(t.to(device) for t in batch2) # for t in cand_indexes

                    input_ids, segment_ids, input_mask = batch[:3]
                    cand_indexes, token_ids = batch2[:2]

                    label_ids, pos_label_ids = batch[3:]

                    loss = model(input_ids, segment_ids, input_mask, cand_indexes, token_ids, label_ids, pos_label_ids)

                    n_gpu = torch.cuda.device_count()
                    if n_gpu > 1: # or loss.shape[0] > 1:
                        loss = loss.mean() # mean() to average on multi-gpu or multitask.
                    if args.fp16 and args.loss_scale != 1.0:
                        # rescale loss for fp16 training
                        # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                        loss = loss * args.loss_scale
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fclassifier == 'Softmax':
                        logger.info("Training loss: {:d}: {:+.2f}".format(ep, loss*1e5))
                    else:
                        logger.info("Training loss: {:d}: {:+.2f}".format(ep, loss))

                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16 or args.optimize_on_cpu:
                            if args.fp16 and args.loss_scale != 1.0:
                                # scale down gradients for fp16 training
                                for param in model.parameters():
                                    param.grad.data = param.grad.data / args.loss_scale
                            is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                            if is_nan:
                                logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                                args.loss_scale = args.loss_scale / 2
                                model.zero_grad()
                                continue
                            optimizer.step()
                            copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                        else:
                            optimizer.step()
                        model.zero_grad()
                        global_step += 1
            else: # no cand_info
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, segment_ids, input_mask = batch[:3]

                    label_ids, pos_label_ids = batch[3:] #if len(batch[3:])>2 else batch[3]
                    loss = model(input_ids, segment_ids, input_mask, label_ids, pos_label_ids)

                    n_gpu = torch.cuda.device_count()
                    if n_gpu > 1: # or loss.shape[0] > 1:
                        loss = loss.mean() # mean() to average on multi-gpu or multitask.
                    if args.fp16 and args.loss_scale != 1.0:
                        # rescale loss for fp16 training
                        # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                        loss = loss * args.loss_scale
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fclassifier == 'Softmax':
                        logger.info("Training loss: {:d}: {:+.2f}".format(ep, loss*1e5))
                    else:
                        logger.info("Training loss: {:d}: {:+.2f}".format(ep, loss))

                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16 or args.optimize_on_cpu:
                            if args.fp16 and args.loss_scale != 1.0:
                                # scale down gradients for fp16 training
                                for param in model.parameters():
                                    param.grad.data = param.grad.data / args.loss_scale
                            is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                            if is_nan:
                                logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                                args.loss_scale = args.loss_scale / 2
                                model.zero_grad()
                                continue
                            optimizer.step()
                            copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                        else:
                            optimizer.step()
                        model.zero_grad()
                        global_step += 1

        tr_time = time.time()-st
        tr_times.append(tr_time)
        logger.info('Training time is {:.3f} seconds.'.format(tr_time))

        # logger.info(tr_loss/step)
        tr_loss = tr_loss / step

        TS_WRITER.add_text('Text', 'text logged at step:' + str(ep), ep)
        if args.fclassifier == 'Softmax':
            TS_WRITER.add_scalar('data/tr_loss', tr_loss*1e5)
        else:
            TS_WRITER.add_scalar('data/tr_loss', tr_loss)

        if eval_dataloaders:
            rs = {}
            if 'ontonotes' in args.task_name.lower():
                parts = ['test', 'dev', 'train']
            else:
                parts = ['test', 'train']

            for part in parts:
                rs[part] = do_eval(model, eval_dataloaders[part], device, args, times=tr_time, type=part, ep=ep)

        if len(rs) != 0:
            for part in parts:
                if part=='train':
                    TS_WRITER.add_scalar('data/train_time', rs['train'][0])

                TS_WRITER.add_scalar('data/'+part+'_eval_time', rs[part][1])
                if args.fclassifier == 'Softmax':
                    TS_WRITER.add_scalar('data/'+part+'_eval_cws_loss', rs[part][2]*1e5)
                    TS_WRITER.add_scalar('data/'+part+'_eval_pos_loss', rs[part][3]*1e5)
                else:
                    TS_WRITER.add_scalar('data/'+part+'_eval_cws_loss', rs[part][2])
                    TS_WRITER.add_scalar('data/'+part+'_eval_pos_loss', rs[part][3])

                TS_WRITER.add_scalar('data/'+part+'_cws_F1', rs[part][4])
                TS_WRITER.add_scalar('data/'+part+'_cws_P', rs[part][5])
                TS_WRITER.add_scalar('data/'+part+'_cws_R', rs[part][6])
                TS_WRITER.add_scalar('data/'+part+'_cws_Acc', rs[part][7])
                TS_WRITER.add_scalar('data/'+part+'_cws_tags', rs[part][8])

                TS_WRITER.add_scalar('data/'+part+'_pos_F1', rs[part][9])
                TS_WRITER.add_scalar('data/'+part+'_pos_P', rs[part][10])
                TS_WRITER.add_scalar('data/'+part+'_pos_R', rs[part][11])
                TS_WRITER.add_scalar('data/'+part+'_pos_Acc', rs[part][12])
                TS_WRITER.add_scalar('data/'+part+'_pos_tags', rs[part][13])
                # results = [avg_times, eval_time, cws_avg_loss, pos_avg_loss, cws_score[0], cws_score[1], cws_score[2], \
                # cws_score[3], cws_sInfo[-1], pos_score[0], pos_score[1], pos_score[2], pos_score[3], pos_sInfo[-1]]

        ts_cws_F1 = rs['test'][4]
        ts_cws_Acc = rs['test'][7]

        ts_pos_F1 = rs['test'][9]
        ts_pos_Acc = rs['test'][12]

        if ts_cws_F1 > old_cws_F1: # only save the best models
            old_cws_F1 = ts_cws_F1

            ckpt_files = sorted(glob(os.path.join(args.output_dir, 'cws_F1_*.pt')))
            for ckpt_file in ckpt_files:
                logger.info('rm %s' % ckpt_file)
                os.system("rm %s" % ckpt_file)

            output_weight_file = os.path.join(args.output_dir, 'cws_F1_weights_epoch%02d.pt'%ep)

            state_dict = model.state_dict()
            if isinstance(model, torch.nn.DataParallel):
                #The models is in a DataParallel container.
                #Its state dict keys are all start with a "module."
                state_dict = OrderedDict({k[len('module.'):]:v for k,v in state_dict.items()})
            torch.save(state_dict, output_weight_file)

        if ts_cws_Acc > old_cws_Acc: # only save the best models
            old_cws_Acc = ts_pos_Acc

            ckpt_files = sorted(glob(os.path.join(args.output_dir, 'cws_Acc_*.pt')))
            for ckpt_file in ckpt_files:
                logger.info('rm %s' % ckpt_file)
                os.system("rm %s" % ckpt_file)

            output_weight_file = os.path.join(args.output_dir, 'cws_Acc_weights_epoch%02d.pt'%ep)

            state_dict = model.state_dict()
            if isinstance(model, torch.nn.DataParallel):
                #The models is in a DataParallel container.
                #Its state dict keys are all start with a "module."
                state_dict = OrderedDict({k[len('module.'):]:v for k,v in state_dict.items()})
            torch.save(state_dict, output_weight_file)


def do_eval(model, eval_dataloader, device, args, times=None, type='test', ep=0):
    model.eval()

    all_label_ids = []
    pos_all_label_ids = []
    all_mask_tokens = []
    cws_all_labels = []
    pos_all_labels = []
    cws_all_losses = []
    pos_all_losses = []
    results = []

    st = time.time()
    if args.dict_file is not None: # contains dictionary file
        if args.do_mask_as_whole:
            for step, (batch, batch2, input_via_dict) in enumerate(tqdm(eval_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                batch2 = tuple(t.to(device) for t in batch2) # for t in cand_indexes
                input_via_dict = input_via_dict.to(device)

                input_ids, segment_ids, input_mask = batch[:3]

                cand_indexes, token_ids = batch2[:2]
                label_ids, pos_label_ids = batch[3:]

                with torch.no_grad():
                    n_gpu = torch.cuda.device_count()

                    if n_gpu > 1: # multiple gpus
                        # models.module.decode to replace original models() since forward cannot output multiple outputs in multiple gpus
                        loss_cws, loss_pos, best_cws_tags_list, best_pos_tags_list \
                            = model.decode(input_ids, segment_ids, input_mask, cand_indexes, token_ids, input_via_dict,
                                           label_ids, pos_label_ids)
                        loss_cws = loss_cws.mean()
                        loss_pos = loss_pos.mean()
                    else:
                        loss_cws, loss_pos, best_cws_tags_list, best_pos_tags_list \
                            = model.decode(input_ids, segment_ids, input_mask, cand_indexes, token_ids, input_via_dict,
                                           label_ids, pos_label_ids)

                if args.no_cuda: # fix bug for can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
                    label_array = label_ids.data
                    pos_label_array = pos_label_ids.data
                    mask_array = input_mask.data
                    tmp_el_cws = loss_cws
                    tmp_el_pos = loss_pos
                else:
                    label_array = label_ids.data.cpu()
                    pos_label_array = pos_label_ids.cpu()
                    mask_array = input_mask.data.cpu()
                    tmp_el_cws = loss_cws.cpu()
                    tmp_el_pos = loss_pos.cpu()

                all_label_ids.extend(label_array.tolist())
                pos_all_label_ids.extend(pos_label_array.tolist())
                all_mask_tokens.extend(mask_array.tolist())
                cws_all_labels.extend(best_cws_tags_list)
                pos_all_labels.extend(best_pos_tags_list)
                cws_all_losses.append(tmp_el_cws.tolist())
                pos_all_losses.append(tmp_el_pos.tolist())
        else:
            for step, (batch, input_via_dict) in enumerate(tqdm(eval_dataloader, desc="TestIter")):
                batch = tuple(t.to(device) for t in batch)
                input_via_dict = input_via_dict.to(device)

                input_ids, segment_ids, input_mask = batch[:3]

                label_ids, pos_label_ids, cand_indexes = batch[3:]

                with torch.no_grad():
                    n_gpu = torch.cuda.device_count()

                    if n_gpu > 1: # multiple gpus
                        # models.module.decode to replace original models() since forward cannot output multiple outputs in multiple gpus
                        loss_cws, loss_pos, best_cws_tags_list, best_pos_tags_list \
                            = model.decode(input_ids, segment_ids, input_mask, input_via_dict,
                                           label_ids, pos_label_ids)
                        loss_cws = loss_cws.mean()
                        loss_pos = loss_pos.mean()
                    else:
                        loss_cws, loss_pos, best_cws_tags_list, best_pos_tags_list \
                            = model.decode(input_ids, segment_ids, input_mask, input_via_dict,
                                           label_ids, pos_label_ids)

                if args.no_cuda: # fix bug for can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
                    label_array = label_ids.data
                    pos_label_array = pos_label_ids.data
                    mask_array = input_mask.data
                    tmp_el_cws = loss_cws
                    tmp_el_pos = loss_pos
                else:
                    label_array = label_ids.data.cpu()
                    pos_label_array = pos_label_ids.cpu()
                    mask_array = input_mask.data.cpu()
                    tmp_el_cws = loss_cws.cpu()
                    tmp_el_pos = loss_pos.cpu()

                all_label_ids.extend(label_array.tolist())
                pos_all_label_ids.extend(pos_label_array.tolist())
                all_mask_tokens.extend(mask_array.tolist())
                cws_all_labels.extend(best_cws_tags_list)
                pos_all_labels.extend(best_pos_tags_list)
                cws_all_losses.append(tmp_el_cws.tolist())
                pos_all_losses.append(tmp_el_pos.tolist())
    else: # no dictionary
        if args.do_mask_as_whole:
            for step, (batch, batch2) in enumerate(tqdm(eval_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                batch2 = tuple(t.to(device) for t in batch2) # for t in cand_indexes

                input_ids, segment_ids, input_mask = batch[:3]
                cand_indexes, token_ids = batch2[:2]
                label_ids, pos_label_ids = batch[3:]

                with torch.no_grad():
                    n_gpu = torch.cuda.device_count()

                    if n_gpu > 1: # multiple gpus
                        # models.module.decode to replace original models() since forward cannot output multiple outputs in multiple gpus
                        loss_cws, loss_pos, best_cws_tags_list, best_pos_tags_list \
                            = model.decode(input_ids, segment_ids, input_mask, label_ids, pos_label_ids, cand_indexes, token_ids)
                        loss_cws = loss_cws.mean()
                        loss_pos = loss_pos.mean()
                    else:
                        loss_cws, loss_pos, best_cws_tags_list, best_pos_tags_list \
                            = model.decode(input_ids, segment_ids, input_mask, label_ids, pos_label_ids, cand_indexes, token_ids)

                if args.no_cuda: # fix bug for can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
                    label_array = label_ids.data
                    pos_label_array = pos_label_ids.data
                    mask_array = input_mask.data
                    tmp_el_cws = loss_cws
                    tmp_el_pos = loss_pos
                else:
                    label_array = label_ids.data.cpu()
                    pos_label_array = pos_label_ids.cpu()
                    mask_array = input_mask.data.cpu()
                    tmp_el_cws = loss_cws.cpu()
                    tmp_el_pos = loss_pos.cpu()

                all_label_ids.extend(label_array.tolist())
                pos_all_label_ids.extend(pos_label_array.tolist())
                all_mask_tokens.extend(mask_array.tolist())
                cws_all_labels.extend(best_cws_tags_list)
                pos_all_labels.extend(best_pos_tags_list)
                cws_all_losses.append(tmp_el_cws.tolist())
                pos_all_losses.append(tmp_el_pos.tolist())
        else:
            for batch in tqdm(eval_dataloader, desc="TestIter"):
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, input_mask = batch[:3]

                label_ids, pos_label_ids, cand_indexes = batch[3:]

                with torch.no_grad():
                    n_gpu = torch.cuda.device_count()

                    if n_gpu > 1: # multiple gpus
                        # models.module.decode to replace original models() since forward cannot output multiple outputs in multiple gpus
                        loss_cws, loss_pos, best_cws_tags_list, best_pos_tags_list \
                            = model.decode(input_ids, segment_ids, input_mask, label_ids, pos_label_ids)
                        loss_cws = loss_cws.mean()
                        loss_pos = loss_pos.mean()
                    else:
                        loss_cws, loss_pos, best_cws_tags_list, best_pos_tags_list \
                            = model.decode(input_ids, segment_ids, input_mask, label_ids, pos_label_ids)

                if args.no_cuda: # fix bug for can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
                    label_array = label_ids.data
                    pos_label_array = pos_label_ids.data
                    mask_array = input_mask.data
                    tmp_el_cws = loss_cws
                    tmp_el_pos = loss_pos
                else:
                    label_array = label_ids.data.cpu()
                    pos_label_array = pos_label_ids.cpu()
                    mask_array = input_mask.data.cpu()
                    tmp_el_cws = loss_cws.cpu()
                    tmp_el_pos = loss_pos.cpu()

                all_label_ids.extend(label_array.tolist())
                pos_all_label_ids.extend(pos_label_array.tolist())
                all_mask_tokens.extend(mask_array.tolist())
                cws_all_labels.extend(best_cws_tags_list)
                pos_all_labels.extend(best_pos_tags_list)
                cws_all_losses.append(tmp_el_cws.tolist())
                pos_all_losses.append(tmp_el_pos.tolist())

    cws_score, cws_sInfo = outputFscoreUsedBIO(all_label_ids, cws_all_labels, all_mask_tokens)

    pos_score, pos_sInfo = outputPOSFscoreUsedBIO(pos_all_label_ids, pos_all_labels, all_mask_tokens)

    eval_time = (time.time() - st) / 60.
    model.train()

    logger.info('Eval time: %.2fmin' % eval_time)
    output_eval_file = os.path.join(args.output_dir, type + "_eval_results.txt")

    if times is not None:
        np_times = np.array(times)
        avg_times = np.mean(np_times)

    cws_np_loss = np.array(cws_all_losses)
    cws_avg_loss = np.mean(cws_np_loss)

    pos_np_loss = np.array(pos_all_losses)
    pos_avg_loss = np.mean(pos_np_loss)

    #if args.fclassifier == 'Softmax':
    #    avg_loss *= 1e5

    with open(output_eval_file, "a+") as writer:
        logger.info("***** Eval results *****")
        if times is not None:
            logger.info('{:s}: ep: {:d}, train time: {:.3f}, test time: {:.3f}, cws_loss: {:.3f}, pos_loss: {:.3f}'.format( \
                type, ep, avg_times, eval_time, cws_avg_loss, pos_avg_loss))
            logger.info(type + ': cws_F1: {:.3f}, cws_P: {:.3f}, cws_R: {:.3f}, cws_Acc: {:.3f}, cws_Tags: {:d}'.format( \
                cws_score[0], cws_score[1], cws_score[2], cws_score[3], cws_sInfo[-1]))
            logger.info(type + ': pos_F1: {:.3f}, pos_P: {:.3f}, pos_R: {:.3f}, pos_Acc: {:.3f}, pos_Tags: {:d}'.format( \
                pos_score[0], pos_score[1], pos_score[2], pos_score[3], pos_sInfo[-1]))

            writer.write('{:s}: ep: {:d}, train time: {:.3f}, test time: {:.3f}, cws_loss: {:.3f}, pos_loss: {:.3f}\n'.format( \
                type, ep, avg_times, eval_time, cws_avg_loss, pos_avg_loss))
            writer.write(type + ': cws_F1: {:.3f}, cws_P: {:.3f}, cws_R: {:.3f}, cws_Acc: {:.3f}, cws_Tags: {:d}\n'.format( \
                cws_score[0], cws_score[1], cws_score[2], cws_score[3], cws_sInfo[-1]))
            writer.write(type + ': pos_F1: {:.3f}, pos_P: {:.3f}, pos_R: {:.3f}, pos_Acc: {:.3f}, pos_Tags: {:d}\n'.format( \
                pos_score[0], pos_score[1], pos_score[2], pos_score[3], pos_sInfo[-1]))

            results = [avg_times, eval_time, cws_avg_loss, pos_avg_loss, cws_score[0], cws_score[1], cws_score[2], \
                       cws_score[3], cws_sInfo[-1], pos_score[0], pos_score[1], pos_score[2], pos_score[3], pos_sInfo[-1]]
        else:
            logger.info('{:s}: ep: {:d}, test time: {:.3f}, cws_loss: {:.3f}, pos_loss: {:.3f}'.format( \
                type, ep, eval_time, cws_avg_loss, pos_avg_loss))
            logger.info(type + ': cws_F1: {:.3f}, cws_P: {:.3f}, cws_R: {:.3f}, cws_Acc: {:.3f}, cws_Tags: {:d}'.format( \
                cws_score[0], cws_score[1], cws_score[2], cws_score[3], cws_sInfo[-1]))
            logger.info(type + ': pos_F1: {:.3f}, pos_P: {:.3f}, pos_R: {:.3f}, pos_Acc: {:.3f}, pos_Tags: {:d}'.format( \
                pos_score[0], pos_score[1], pos_score[2], pos_score[3], pos_sInfo[-1]))

            writer.write('{:s}: ep: {:d}, test time: {:.3f}, cws_loss: {:.3f}, pos_loss: {:.3f}\n'.format( \
                type, ep, eval_time, cws_avg_loss, pos_avg_loss))
            writer.write(type + ': cws_F1: {:.3f}, cws_P: {:.3f}, cws_R: {:.3f}, cws_Acc: {:.3f}, cws_Tags: {:d}\n'.format( \
                cws_score[0], cws_score[1], cws_score[2], cws_score[3], cws_sInfo[-1]))
            writer.write(type + ': pos_F1: {:.3f}, pos_P: {:.3f}, pos_R: {:.3f}, pos_Acc: {:.3f}, pos_Tags: {:d}\n'.format( \
                pos_score[0], pos_score[1], pos_score[2], pos_score[3], pos_sInfo[-1]))

            results = [eval_time, cws_avg_loss, pos_avg_loss, cws_score[0], cws_score[1], cws_score[2], \
                       cws_score[3], cws_sInfo[-1], pos_score[0], pos_score[1], pos_score[2], pos_score[3], pos_sInfo[-1]]

    return results


def train_CWS_POS(args):
    processors = {
        **dict.fromkeys(['ontonotes_cws_pos', 'ontonotes_cws_pos2.0'], lambda: \
            CWS_POS(nopunc=args.nopunc, drop_columns=['full_pos', 'bert_ner', 'src_ner', 'src_seg', 'text_seg'],
                    pos_tags_file='./resource/pos_tags.txt')),
        **dict.fromkeys(['msr', 'pku', 'as', 'cityu'], lambda: \
            CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']))
    }
        #'4cws_cws': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']),

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # Prepare models
    processor = processors[task_name]()
    CWS_label_list = processor.get_labels() # get_CWS_labels
    POS_label_list = processor.get_POS_labels() # get_POS_labels

    if task_name in ['msr', 'pku', 'as', 'cityu']:
        args.data_dir += args.task_name.upper()

    print('data_dir: ' + args.data_dir)

    args.output_dir = args.output_dir + '/l' + str(args.num_hidden_layers)

    print('output_dir: ' + args.output_dir)
    os.system('mkdir %s' %args.output_dir)
    os.system('chmod 777 %s' %args.output_dir)

    train_dataset, train_dataloader = get_dataset_stored_with_dict_and_dataloader(processor, args, training=True, type_name='train')

    eval_dataloaders = get_eval_stored_with_dict_and_dataloader(processor, args)

    num_train_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    no_decay = ['bias', 'gamma', 'beta']

    model, device = load_CWS_POS_model(CWS_label_list, POS_label_list, args)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())

        if args.method != 'fine_tune':
            for param in model.bert.parameters():
                param.requires_grad = False

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
        ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    do_train(model, train_dataloader, optimizer, param_optimizer,
             device, args, eval_dataloaders)


def set_local_Ontonotes_param():
    return {'task_name': 'ontonotes_cws_pos2.0',
            'model_type': 'sequencelabeling',
            'data_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4nerpos_update/valid',
            'vocab_file': './src/BERT/models/multi_cased_L-12_H-768_A-12/vocab.txt',
            'bert_config_file': './src/BERT/models/multi_cased_L-12_H-768_A-12/bert_config.json',
            'output_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/ontonotes/CWSPOS2/dict/',
            'do_lower_case': True,
            'train_batch_size': 5,
            'max_seq_length': 128,
            'num_hidden_layers': 1,
            'init_checkpoint': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/multi_cased_L-12_H-768_A-12/',
            'bert_model_dir': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/multi_cased_L-12_H-768_A-12/',
            'no_cuda': True,
            'num_train_epochs': 20,
            'method': 'fine_tune',
            'do_mask_as_whole': True,
            'learning_rate': 1e-5,
            'override_output': True,
            }


def set_server_Ontonotes_param():
    return {'task_name': 'ontonotes_cws_pos2.0',
            'model_type': 'sequencelabeling',
            'data_dir': '../data/ontonotes5/4nerpos_update/valid/',
            'vocab_file': './src/BERT/models/multi_cased_L-12_H-768_A-12/vocab.txt',
            'bert_config_file': './src/BERT/models/multi_cased_L-12_H-768_A-12/bert_config.json',
            'output_dir': './tmp/ontonotes/CWSPOS2/dict/',
            'do_lower_case': False,
            'train_batch_size': 4,
            'max_seq_length': 128,
            'num_hidden_layers': 1,
            'init_checkpoint': '../models/multi_cased_L-12_H-768_A-12/',
            'bert_model_dir': '../models/multi_cased_L-12_H-768_A-12/',
            'visible_device': 3,
            'num_train_epochs': 20,
            'method': 'fine_tune',
            'do_mask_as_whole': True,
            'learning_rate': 1e-5,
            'override_output': True,
            }


TEST_FLAG = False
TEST_FLAG = True
isServer = True
isServer = False

def main(**kwargs):
    if TEST_FLAG:
        if isServer:
            kwargs = set_server_Ontonotes_param()
        else:
            kwargs = set_local_Ontonotes_param()
    else:
        print('load parameters from .sh')

    args._parse(kwargs)
    train_CWS_POS(args)

    fn = os.path.join(args.output_dir, args.method + '_l' + str(args.num_hidden_layers) + '_rs.json')

    TS_WRITER.export_scalars_to_json(fn)
    TS_WRITER.close()


if __name__=='__main__':
    import fire
    fire.Fire(main)



