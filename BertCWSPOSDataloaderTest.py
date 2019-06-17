#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 09:28 2019-02-21 
@author: haiqinyang

Feature: 

Scenario: 
"""
import sys
sys.path.append('./src')

import os

from src.pkuseg.metrics import getFscoreFromBIOTagList
from tqdm import tqdm, trange
from src.utilis import get_Ontonotes, convertList2BIOwithComma, BMES2BIO, space2Comma, load_4CWS
import pandas as pd
from src.config import args
from src.preprocess import CWS_POS # dataset_to_dataloader, randomly_mask_input, OntoNotesDataset
import time
from src.utilis import get_dataset_and_dataloader, get_eval_dataloaders
from src.BERT.optimization import BertAdam
from src.metrics import outputFscoreUsedBIO

import numpy as np
import torch
from glob import glob
import pdb
import re

from src.BERT.modeling import BertConfig
# here generate exeception of Parameter config in `BertVariant(config)` should be an instance of class
# `BertConfig`. To create a models from a Google pretrained models use
# `models = BertVariant.from_pretrained(PRETRAINED_MODEL_NAME)`

from src.customize_modeling import BertVariantCWSPOS
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

    model = BertVariantCWSPOS(bert_config, len(CWS_label_list), len(POS_label_list), method=args.method, fclassifier=args.fclassifier)

    if args.bert_model_dir is None:
        raise RuntimeError('Evaluating a random initialized models is not supported...!')
    #elif os.path.isdir(args.init_checkpoint):
    #    raise ValueError("init_checkpoint is not a file")
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

    old_F1 = 0.
    old_Acc = 0.
    for ep in trange(int(args.num_train_epochs), desc="Epoch"):
        st = time.time()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids, input_mask = batch[:3]
            if args.pretraining:
                input_ids, label_ids = randomly_mask_input(input_ids, train_dataloader.dataset.tokenizer)
                input_ids = input_ids.to(device)
                label_ids = label_ids.to(device)
            else:
                label_ids, pos_label_ids = batch[3:] #if len(batch[3:])>2 else batch[3]
                #pos_label_ids = batch[4:] if len(batch[4:])>1 else batch[4]

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
            TS_WRITER.add_scalar('data/tr_loss', tr_loss*1e5)

        if eval_dataloaders:
            rs = {}
            if args.task_name.lower() == 'ontonotes_cws':
                parts = ['test', 'dev', 'train']
            else:
                parts = ['test', 'train']

            for part in parts:
                rs[part] = do_eval(model, eval_dataloaders[part], device, args, times=tr_time, type=part)

        if len(rs) != 0:
            for part in parts:
                if part=='train':
                    TS_WRITER.add_scalar('data/train_time', rs['train'][0])

                TS_WRITER.add_scalar('data/'+part+'_eval_time', rs[part][1])
                if args.fclassifier == 'Softmax':
                    TS_WRITER.add_scalar('data/'+part+'_eval_loss', rs[part][2]*1e5)
                else:
                    TS_WRITER.add_scalar('data/'+part+'_eval_loss', rs[part][2])
                TS_WRITER.add_scalar('data/'+part+'_F1', rs[part][3])
                TS_WRITER.add_scalar('data/'+part+'_P', rs[part][4])
                TS_WRITER.add_scalar('data/'+part+'_R', rs[part][5])
                TS_WRITER.add_scalar('data/'+part+'_Acc', rs[part][6])

        ts_F1 = rs['test'][3]
        ts_Acc = rs['test'][6]

        if ts_F1 > old_F1: # only save the best models
            old_F1 = ts_F1

            ckpt_files = sorted(glob(os.path.join(args.output_dir, 'F1_*.pt')))
            for ckpt_file in ckpt_files:
                logger.info('rm %s' % ckpt_file)
                os.system("rm %s" % ckpt_file)

            output_weight_file = os.path.join(args.output_dir, 'F1_weights_epoch%02d.pt'%ep)

            state_dict = model.state_dict()
            if isinstance(model, torch.nn.DataParallel):
                #The models is in a DataParallel container.
                #Its state dict keys are all start with a "module."
                state_dict = OrderedDict({k[len('module.'):]:v for k,v in state_dict.items()})
            torch.save(state_dict, output_weight_file)

        if ts_Acc > old_Acc: # only save the best models
            old_Acc = ts_Acc

            ckpt_files = sorted(glob(os.path.join(args.output_dir, 'Acc_*.pt')))
            for ckpt_file in ckpt_files:
                logger.info('rm %s' % ckpt_file)
                os.system("rm %s" % ckpt_file)

            output_weight_file = os.path.join(args.output_dir, 'Acc_weights_epoch%02d.pt'%ep)

            state_dict = model.state_dict()
            if isinstance(model, torch.nn.DataParallel):
                #The models is in a DataParallel container.
                #Its state dict keys are all start with a "module."
                state_dict = OrderedDict({k[len('module.'):]:v for k,v in state_dict.items()})
            torch.save(state_dict, output_weight_file)


def do_eval(model, eval_dataloader, device, args, times=None, type='test'):
    model.eval()

    all_label_ids = []
    all_pos_label_ids = []
    all_mask_tokens = []
    all_cws_labels = []
    all_pos_labels = []
    results = []
    st = time.time()
    for batch in tqdm(eval_dataloader, desc="TestIter"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, segment_ids, input_mask = batch[:3]
        label_ids = batch[3:] if len(batch[3:])>1 else batch[3]
        pos_label_ids = batch[4:] if len(batch[4:])>1 else batch[4]
        with torch.no_grad():
            n_gpu = torch.cuda.device_count()

            if n_gpu > 1: # multiple gpus
            	# models.module.decode to replace original models() since forward cannot output multiple outputs in multiple gpus
                loss_cws, loss_POS, best_cws_tags_list, best_pos_tags_list \
                    = model.module.decode(input_ids, segment_ids, input_mask, label_ids, pos_label_ids)
                loss_cws = loss_cws.mean()
                loss_pos = loss_pos.mean()
            else:
                loss_cws, loss_pos, best_cws_tags_list, best_pos_tags_list \
                    = model.module.decode(input_ids, segment_ids, input_mask, label_ids, pos_label_ids)

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
                tmp_el_cws = loss_CWS.cpu()
                tmp_el_pos = loss_POS.cpu()

        all_label_ids.extend(label_array.tolist())
        pos_all_label_ids.extend(pos_label_array.tolist())
        all_mask_tokens.extend(mask_array.tolist())
        cws_all_labels.extend(best_cws_tags_list)
        pos_all_labels.extend(best_pos_tags_list)
        cws_all_losses.append(tmp_el_cws.tolist())
        pos_all_losses.append(tmp_el_pos.tolist())

    cws_score, cws_sInfo = outputFscoreUsedBIO(all_label_ids, cws_all_labels, all_mask_tokens)

    pdb.set_trace()
    pos_score, pos_sInfo = outputFscoreUsedBIO(pos_all_label_ids, pos_all_labels, all_mask_tokens)

    eval_time = (time.time() - st) / 60.
    model.train()

    logger.info('Eval time: %.2fmin' % eval_time)
    output_eval_file = os.path.join(args.output_dir, type+"_eval_results.txt")

    if times is not None:
        np_times = np.array(times)
        avg_times = np.mean(np_times)

    np_loss = np.array(all_cws_losses)
    cws_avg_loss = np.mean(np_loss)

    if args.fclassifier == 'Softmax':
        avg_loss *= 1e5

    with open(output_eval_file, "a+") as writer:
        logger.info("***** Eval results *****")
        if times is not None:
            logger.info(type + ': train time: {:.3f}, test time: {:.3f}, loss: {:.3f}, F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, Tags: {:d}'.format( \
                               avg_times, eval_time, avg_loss, score[0], score[1], score[2], score[3], sInfo[-1]))
            writer.write(type + ': train time: {:.3f}, test time: {:.3f}, loss: {:.3f}, F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, Tags: {:d}\n'.format( \
                               avg_times, eval_time, avg_loss, score[0], score[1], score[2], score[3], sInfo[-1]))
            results = [avg_times, eval_time, avg_loss, score[0], score[1], score[2], score[3], sInfo[-1]]
        else:
            logger.info(type + ': test time: {:.3f}, loss: {:.3f}, F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, Tags: {:d}'.format( \
                               eval_time, avg_loss, score[0], score[1], score[2], score[3], sInfo[-1]))
            writer.write(type + ': test time: {:.3f}, loss: {:.3f}, F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, Tags: {:d}\n'.format( \
                               eval_time, avg_loss, score[0], score[1], score[2], score[3], sInfo[-1]))
            results = [eval_time, avg_loss, score[0], score[1], score[2], score[3], sInfo[-1]]

    return results


def train_CWS_POS(args):
    processors = {
        'ontonotes': lambda: CWS_POS(nopunc=args.nopunc, drop_columns=['full_pos', 'bert_ner', 'src_ner', 'src_seg', 'text_seg'], \
                                     pos_tags_file='./resource/pos_tags.txt'),
        #'4cws_cws': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']),
        #'msr': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']),
        #'pku': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']),
        #'as': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']),
        #'cityu': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg'])
    }

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # Prepare models
    processor = processors[task_name]()
    CWS_label_list = processor.get_labels() # get_CWS_labels
    POS_label_list = processor.get_POS_labels() # get_POS_labels

    #args.data_dir += args.task_name
    #print('data_dir: ' + args.data_dir)

    #if args.method == 'last_layer':
    #    args.output_dir = args.output_dir + args.task_name + '/' + args.fclassifier \
    #                      + '/l' + str(args.num_hidden_layers)
    #else:
    #    args.output_dir = args.output_dir + args.task_name + '/' + args.fclassifier \
    #                      + '/' + args.method + '/l' + str(args.num_hidden_layers)

    print('output_dir: ' + args.output_dir)
    os.system('mkdir %s' %args.output_dir)
    os.system('chmod 777 %s' %args.output_dir)

    train_dataset, train_dataloader = get_dataset_and_dataloader(processor, args, training=True, type = 'train')

    eval_dataloaders = get_eval_dataloaders(processor, args)

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
        #pdb.set_trace()
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
    return {'task_name': 'ontonotes',
            'model_type': 'sequencelabeling',
            'data_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4nerpos_data/valid/',
            'vocab_file': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/vocab.txt',
            'bert_config_file': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/bert_config.json',
            'output_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/ontonotes/CWSPOS',
            'do_lower_case': True,
            'train_batch_size': 32,
            'max_seq_length': 128,
            'num_hidden_layers': 12,
            'init_checkpoint': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/',
            'bert_model_dir': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/',
            'no_cuda': True,
            'num_train_epochs': 10,
            'method': 'fine_tune',
            'learning_rate': 2e-5,
            'override_output': True,
            }


TEST_FLAG = False
#TEST_FLAG = True

def main(**kwargs):
    if TEST_FLAG:
        kwargs = set_local_Ontonotes_param()
    else:
        print('load parameters from .sh')

    args._parse(kwargs)
    train_CWS_POS(args)

    if args.method == 'last_layer':
        fn = os.path.join(args.output_dir, args.fclassifier + '_l' + str(args.num_hidden_layers) + '_rs.json')
    else:
        fn = os.path.join(args.output_dir, args.fclassifier + '_' + args.method + '_rs.json')

    TS_WRITER.export_scalars_to_json(fn)
    TS_WRITER.close()


if __name__=='__main__':
    import fire
    fire.Fire(main)



