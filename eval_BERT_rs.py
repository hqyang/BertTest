#!/anaconda3/envs/haiqin370/bin/python3
# -*- coding: utf-8 -*-
"""
Created on at 2:17 PM 23/1/2019 
@author: haiqinyang

Feature: 

Scenario: 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import random
from collections import OrderedDict
from tqdm import tqdm, trange
from src.metrics import outputFscoreUsedBIO

import numpy as np
import torch

import time
from glob import glob
import pdb

#import sys
#import importlib
#importlib.reload(sys)
'''
import tokenization
import time
import argparse
import itertools
import pandas as pd
from torchnet.meter import APMeter
import .src.customize_modeling
'''

from src.BERT.modeling import BertConfig, BertForMaskedLM
from src.customize_modeling import BertCRF
# ModuleNotFoundError: No module named '__main__.src'; '__main__' is not a package
#, BertForSequenceClassification
#from .src.BERT.modeling import BertForSequenceMultilabelClassification
#from .src.BERT.modeling import BertForSequenceMultiTaskClassification
from src.BERT.optimization import BertAdam

from src.preprocess import dataset_to_dataloader, randomly_mask_input, OntoNotesDataset, CWS_BMEO
from src.config import args
from src.tokenization import FullTokenizer

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
WEIGHTS_NAME = 'pytorch_model.bin'

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def set_server_param():
    return {'task_name': 'ontonotes_CWS',
            'model_type': 'sequencelabeling',
            'data_dir': '../data/ontonotes5/',
            'bert_model_dir': '../models/bert-base-chinese/',
            'vocab_file': '../models/bert-base-chinese/vocab.txt',
            'output_dir': './tmp/ontonotes',
            'do_train': True,
            'init_checkpoint': '../models/bert-base-chinese/pytorch_model.bin',
            'do_eval': True,
            'do_lower_case': True,
            'train_batch_size': 128,
            'override_output': True,
            'tensorboardWriter': False,
            'visible_device': (0),
            #'visible_device': 0,
            'num_train_epochs': 1,
            'max_seq_length': 128,
	        'num_hidden_layers': 3
            }


def get_dataset_and_dataloader(processor, args, training=True, type='train'):
    #if type=='train': # loading data for training
    #    training=True
    #else:
    #    training=False
 
    #    if type=='train_with_eval': #eval on training data, other type is 'dev', 'test'
    #        type='train'
    
    # training for training mode or test mode
    # type: which dataset 
    dataset = OntoNotesDataset(processor, args.data_dir, args.vocab_file,
                                 args.max_seq_length, training, type=type)

    dataloader = dataset_to_dataloader(dataset, args.train_batch_size,
                                       args.local_rank, training=training)
    return dataset, dataloader


def load_model(label_list, tokenizer, args):
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

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.bert_model_dir is not None:
        config_file = os.path.join(args.bert_model_dir, CONFIG_NAME)
        bert_config = BertConfig.from_json_file(config_file)
    else:
        bert_config = BertConfig.from_json_file(args.bert_config_file)
    
    if args.num_hidden_layers>0 and args.num_hidden_layers<bert_config.num_hidden_layers:
        bert_config.num_hidden_layers = args.num_hidden_layers 

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if not args.override_output:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        else:
            os.system("rm %s" % os.path.join(args.output_dir, '*'))

    model = None
    model_type = {
        "pre_training": lambda: BertForMaskedLM(bert_config),
        #"multilabel": lambda: BertForSequenceMultilabelClassification(bert_config, len(label_list)),
        #"multitask": lambda: BertForSequenceMultiTaskClassification(bert_config, [len(_) for _ in label_list]),
        #"sequenceclassification": lambda: BertForSequenceClassification(bert_config, len(label_list))
        "sequencelabeling": lambda: BertCRF(bert_config, len(label_list))
    }

    model = model_type[args.model_type]()

    if args.init_checkpoint is not None:
        if os.path.isdir(args.init_checkpoint):
            assert (not args.do_train and args.do_eval)
        else:
            # retrained_model_dir is not needed
            #if args.retrained_model_dir is not None:
            #    weights_path = os.path.join(args.retrained_model_dir, WEIGHTS_NAME)
            #else:
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
                logger.info("Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
  
            #pdb.set_trace()
 
    if args.fp16:
        model.half()
    if args.modification:
        modify_func = getattr(customize_modeling, args.modification)
        model = modify_func(model)
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model, device


def do_train(model, train_dataloader, optimizer, param_optimizer, device, args, eval_dataloader=None):
    global_step = 0

    if args.tensorboardWriter:
        loss_all = []
    
    tr_times = []
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
                label_ids = batch[3:] if len(batch[3:])>1 else batch[3]
            #loss, decode_rs = model(input_ids, segment_ids, input_mask, label_ids)
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            #s1 = outputFscoreUsedBIO(list(label_ids.data.numpy()), decode_rs, list(input_mask.data.numpy()))
           
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1: # or loss.shape[0] > 1:
                loss = loss.mean() # mean() to average on multi-gpu or multitask.
            if args.fp16 and args.loss_scale != 1.0:
                # rescale loss for fp16 training
                # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            logger.info("Training loss: {:d}: {:+.2f}".format(ep, loss))
            #l #logger.info("Training F1, Precision, Recall: {:d}: {:+.2f}, {:+.2f}, {:+.2f}".format(ep, s1))

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
            
        if args.tensorboardWriter:
            loss_all.append(loss)

        output_weight_file = os.path.join(args.output_dir, 'weights_epoch%02d.pt'%ep)

        state_dict = model.state_dict()
        if isinstance(model, torch.nn.DataParallel):
            #The model is in a DataParallel container.
            #Its state dict keys are all start with a "module."
            state_dict = OrderedDict({k[len('module.'):]:v for k,v in state_dict.items()})
        torch.save(state_dict, output_weight_file)
        # logger.info(tr_loss/step)
        tr_loss = tr_loss / step
        if (args.do_eval) and (not args.pretraining):
            #do_eval(model, train_dataloader, device, args, tr_times, type='train')
            do_eval(model, eval_dataloader, device, args, tr_times, type='train')
    
    if args.tensorboardWriter:
        len_loss_all = len(loss_all)
        labels = ["%d" % (i) for i in range(len_loss_all)]

        with SummaryWriter() as writer:
            for i in range():
                writer.add_embedding(
                    mat=loss_all[i],
                    metadata = labels,
                    tag = "train_loss",
                    global_step=i)

def do_eval(model, eval_dataloader, device, args, times=[], type='test'):
    model.eval()
    all_label_ids = []
    all_losses = []
   
    results = []

    label_list = eval_dataloader.dataset.label_list
    st = time.time()
    for batch in tqdm(eval_dataloader, desc="TestIter"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, segment_ids, input_mask = batch[:3]
        label_ids = batch[3:] if len(batch[3:])>1 else batch[3]
        with torch.no_grad():
            n_gpu = torch.cuda.device_count()
            
            if n_gpu > 1: # multiple gpus 
            	# model.module.decode to replace original model() since forward cannot output multiple outputs in multiple gpus
                tmp_eval_loss, tmp_decode_rs = model.module.decode(input_ids, segment_ids, input_mask, label_ids)
                tmp_eval_loss = tmp_eval_loss.mean()
            else:
                tmp_eval_loss, tmp_decode_rs = model.decode(input_ids, segment_ids, input_mask, label_ids)

            if args.no_cuda: # fix bug for can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
                label_array = label_ids.data
                mask_array = input_mask.data
            else:
                label_array = label_ids.data.cpu()
                mask_array = input_mask.data.cpu()
            score, _ = outputFscoreUsedBIO(list(label_array.numpy()), tmp_decode_rs, list(mask_array.numpy()))

            logger.info(type+' F1, Precision, Recall: {:+.2f}, {:+.2f}, {:+.2f}'.format(score[0], score[1], score[2]))
            #score = output_Fscore(eval_dataloader.dataset.idx_to_label_map, label_list, input_mask, tmp_decode_rs)
        if args.no_cuda:
            tmp_el = tmp_eval_loss
        else: 
            tmp_el = tmp_eval_loss.cpu() 
   
        result = [tmp_el.numpy().tolist()]
        result.extend(score)

    #    all_label_ids.append(label_ids)
    #    all_losses.append(tmp_eval_loss)
    #results = eval_by_metrics(all_label_ids, all_losses, all_logits, label_list,
    #                          tr_loss, global_step, args.multilabel)
        results.append(result) # loss, F1, P, R

    model.train()
    eval_time = (time.time() - st) / 60
    logger.info('Eval time: %.2fmin' % eval_time)
    output_eval_file = os.path.join(args.output_dir, type+'_eval_rs.txt')
    print(output_eval_file)
   
    if times!=[]:
       np_times = np.array(times)
       avg_times = np.mean(np_times)     
  
    np_res = np.array(results)
    avg_res = np.mean(np_res, axis=0)

    with open(output_eval_file, "a+") as writer:
        logger.info("***** Eval results *****")
        
        if times!=[]:
            logger.info("time: {:.3f}, loss: {:.3f}, F1: {:.3f}, P: {:.3f}, R: {:.3f}".format(avg_times, avg_res[0], avg_res[1], avg_res[2], avg_res[3]))
            writer.write("time: {:.3f}, loss: {:.3f}, F1: {:.3f}, P: {:.3f}, R: {:.3f}\n".format(avg_times, avg_res[0], avg_res[1], avg_res[2], avg_res[3]))
        else:         
            logger.info("loss: {:.3f}, F1: {:.3f}, P: {:.3f}, R: {:.3f}".format(avg_res[0], avg_res[1], avg_res[2], avg_res[3]))
            writer.write("loss: {:.3f}, F1: {:.3f}, P: {:.3f}, R: {:.3f}\n".format(avg_res[0], avg_res[1], avg_res[2], avg_res[3]))
         
    '''
    for result in results:
        with open(output_eval_file, "a+") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write("\n")
    '''
    return results

def eval_by_metrics(labels, losses, logits, label_lists, train_loss, global_step, multilabel=False):
    if not isinstance(logits, list):
        infos = [labels], [losses], [logits], [label_lists]
    else:
        infos = [labels, losses, logits, label_lists]
    results = []
    for task_id, info in enumerate(zip(*infos)):
        label, loss, logit, label_list = info
        if multilabel:
            acc_count = accuracy_multilabel(logit, label)
            output = predict_at_least_one(logit)
            acc = acc_count / logit.shape[0]
        else:
            acc = accuracy(logit, label, ignore_index=0, reduce=True)
            score = np.exp(logit) / np.exp(logit).sum(axis=1, keepdims=True)
            output, label = map_score_to_multilabel(label_list, score, label)

        apmeter_by_class = APMeter()
        apmeter_by_sample = APMeter()
        apmeter_by_class.add(output, label)
        apmeter_by_sample.add(output.T, label.T)
        eval_loss = loss.mean()
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': acc,
                  'mAP_class': apmeter_by_class.value().mean().item(),
                  'mAP_sample': apmeter_by_sample.value().mean().item(),
                  'global_step': global_step,
                  'loss': train_loss,
                  'task': task_id}
        results.append(result)
    return results

def set_test_param():
    return {'task_name': 'ontonotes_CWS',
            'model_type': 'sequencelabeling',
            'data_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data',
            'bert_model_dir': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/',
            'vocab_file': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/vocab.txt',
            'output_dir': '/Users/haiqinyang/Downloads/tmp/ontonotes',
            'do_train': True,
            'do_eval': True,
            'do_lower_case': True,
            'train_batch_size': 32,
            'init_checkpoint': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/pytorch_model.bin',
            'override_output': True,
            'tensorboardWriter': True
            }

def set_eval_param():
    return {'task_name': 'ontonotes_CWS',
            'model_type': 'sequencelabeling',
            'data_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data',
            'bert_model_dir': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/',
            'vocab_file': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/vocab.txt',
            'output_dir': '/Users/haiqinyang/Downloads/tmp/ontonotes/out',
            'do_train': False,
            'init_checkpoint': '/Users/haiqinyang/Downloads/tmp/ontonotes/',
            'do_eval': True,
            'do_eval_train': True,
            'do_lower_case': True,
            'train_batch_size': 64,
            'override_output': True,
            'tensorboardWriter': False,
            'visible_device': 3,
            'num_train_epochs': 15,
            'max_seq_length': 128,
	        'num_hidden_layers': 3
            }

def set_server_eval_param():
    return {'task_name': 'ontonotes_CWS',
            'model_type': 'sequencelabeling',
            'data_dir': '../data/ontonotes5/',
            'bert_model_dir': '../models/bert-base-chinese/',
            'vocab_file': '../models/bert-base-chinese/vocab.txt',
            'output_dir': './tmp/ontonotes',
            'do_train': False,
            'init_checkpoint': './tmp/ontonotes/',
            'do_eval': True,
            'do_eval_train': True,
            'do_lower_case': True,
            'train_batch_size': 64,
            'override_output': False,
            'tensorboardWriter': False,
            'visible_device': 0, # device 0
            'num_train_epochs': 15,
            'max_seq_length': 128,
	    'num_hidden_layers': 3
            }

def main(**kwargs):
    #kwargs = set_test_param()

    #kwargs = set_eval_param()
    args._parse(kwargs)
    
    processors = {
        "ontonotes_cws": lambda: CWS_BMEO(nopunc=args.nopunc),
    }

    #pdb.set_trace()
    if args.do_train:
        args.output_dir = args.output_dir + '/nhl' \
                +str(args.num_hidden_layers)+'_nte'+str(args.num_train_epochs) \
                +'_nbs'+str(args.train_batch_size) 
        print(args.output_dir)
    
    if args.do_eval:
        args.init_checkpoint = args.init_checkpoint + '/nhl' \
                +str(args.num_hidden_layers)+'_nte'+str(args.num_train_epochs) \
                +'_nbs'+str(args.train_batch_size) 
        args.output_dir = args.init_checkpoint + '/out'

        print(args.init_checkpoint)
        print(args.output_dir)
    
    if not args.do_train: 
        os.makedirs(args.output_dir, exist_ok=True)
    #pdb.set_trace() 
	
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # Prepare model
    processor = processors[task_name]()
    tokenizer = FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    label_list = processor.get_labels() # get_labels

    model, device = load_model(label_list, tokenizer, args)

    if args.do_train:
        processor.save_labelidmap(args.output_dir)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
        ]

    #do train
    train_dataset = None
    num_train_steps = None
    if args.do_train:
        train_dataset, train_dataloader = get_dataset_and_dataloader(processor, args, True, type='train') # type can be 'train', 'dev', 'test'
        #train_dataset, train_dataloader = get_dataset_and_dataloader(processor, args, True, type='dev') # type can be 'train', 'dev', 'test'
        train_dataset._tokenize()
        num_train_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)
        
        #pdb.set_trace()
        eval_dataloader = None
        if args.do_eval:
            #eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, False, type='dev') # train and eval on training set
            #if args.train_batch_size==128:
            #    eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, False, type='dev') # train and eval on training set
            #else:
            eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, False, type='train') # train and eval on training set
            eval_dataset._tokenize()
     
            #pdb.set_trace()
        do_train(model, train_dataloader, optimizer, param_optimizer,
                 device, args, eval_dataloader=eval_dataloader)

    #pdb.set_trace()
    if (args.do_eval) and not (args.do_train):
        eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, False, type='test') # eval on test data
        global_step = 0
        eval_fc = do_eval_pretraining if args.pretraining else do_eval
        if args.init_checkpoint is None:
            raise RuntimeError('Evaluating a random initialized model is not supported...!')
        elif os.path.isdir(args.init_checkpoint):
            ckpt_files = sorted(glob(os.path.join(args.init_checkpoint, '*.pt')))
            for ckpt_file in ckpt_files:
                weights = torch.load(ckpt_file, map_location='cpu')
                try:
                    model.load_state_dict(weights)
                except RuntimeError:
                    model.module.load_state_dict(weights)
                eval_fc(model, eval_dataloader, device, args)

                type='train'
                eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, False, type) # eval on training data
                eval_fc(model, eval_dataloader, device, args, [], type)

                type='dev'
                eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, False, type) # eval on training data
                eval_fc(model, eval_dataloader, device, args, [], type)
        else:
            eval_fc(model, eval_dataloader, device, args, 'test')

if __name__ == "__main__":
    import fire
    fire.Fire(main)
