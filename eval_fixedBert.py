#!/anaconda3/envs/haiqin370/bin/ python3
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

from src.BERT.modeling import BertConfig, BertForMaskedLM
from src.customize_modeling import BertCRFWAMCWS
from src.BERT.optimization import BertAdam
from src.pkuseg.metrics import getFscoreFromBIOTagList

from src.preprocess import dataset_to_dataloader, randomly_mask_input, OntoNotesDataset, CWS_BMEO
from src.config import args
from src.tokenization import FullTokenizer
from src.utilis import save_model, get_Ontonotes, convertList2BIOwithComma, BMES2BIO, space2Comma


CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def set_server_param():
    return {'task_name': 'ontonotes_CWS',
            'model_type': 'sequencelabeling',
            'data_dir': '../data/ontonotes5/4ner_data/',
            'bert_model_dir': '../models/bert-base-chinese/',
            'vocab_file': '../models/bert-base-chinese/vocab.txt',
            'output_dir': './tmp/ontonotes/',
            'do_train': True,
            'init_checkpoint': '../models/bert-base-chinese/pytorch_model.bin',
            'do_eval': True,
            'do_lower_case': True,
            'train_batch_size': 32,
            'append_dir': True, 
            'override_output': True,
            'tensorboardWriter': False,
            #'visible_device': (0,1,2),
            'visible_device': 0,
            'num_train_epochs': 15,
            'max_seq_length': 128,
	    'num_hidden_layers': 12
            }

def set_test_param():
    return {'task_name': 'ontonotes_CWS',
            'model_type': 'sequencelabeling',
            'data_dir': './tmp/ontonotes/final_data',
            'bert_model_dir': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/',
            'vocab_file': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/vocab.txt',
            'output_dir': './tmp/ontonotes',
            'do_train': True,
            'do_eval': True,
            'do_lower_case': True,
            'train_batch_size': 2,
            'init_checkpoint': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/pytorch_model.bin',
            'override_output': True,
            'tensorboardWriter': False
            }

def get_dataset_and_dataloader(processor, args, training=True, type='train'):
    dataset = OntoNotesDataset(processor, args.data_dir, args.vocab_file,
                                 args.max_seq_length, training=training, type=type)
    dataloader = dataset_to_dataloader(dataset, args.train_batch_size,
                                       args.local_rank, training=training)
    return dataset, dataloader


def load_model(label_list, args):
    #pdb.set_trace()
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

    if not args.do_train and not args.do_eval_df:
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

    model = BertCRFWAMCWS(device, bert_config, args.vocab_file, args.max_seq_length, args.projected_size, \
                          len(label_list), 32)

    if args.init_checkpoint is not None:
        if os.path.isdir(args.init_checkpoint):
            assert (not args.do_train and args.do_eval_df)
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
                logger.info("Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys))
            '''
            weights = torch.load(args.init_checkpoint, map_location='cpu')
            try:
                model.bert.load_state_dict(weights, strict=True)
            except RuntimeError:
                logger.info("Try loading self-pretrained weights(strict=True) instead of the google's one")
                weights = OrderedDict({k:v for k,v in weights.items()})
                try:
                    model.load_state_dict(weights, strict=True)
                except RuntimeError:
                    logger.info('Load self-pretrained weights(strict=True) failed...')
                    logger.info('Loading self-pretrained weights(strict=False)')
                    model.load_state_dict(weights, strict=False)
            '''
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


def do_eval_df_with_model(model, df, output_eval_file, type):
    bertCRFList = []
    trueLabelList = []

    sent_list = []
    truelabelstr = ''


    for i, data in tqdm(enumerate(df.itertuples())):
        sentence = data.text

        sent_list.append(sentence)

        tl = BMES2BIO(data.label)
        tl = space2Comma(tl)
        trueLabelList.append(tl)
        truelabelstr += tl

    rs_precision_all = model.cutlist_noUNK(sent_list)

    for idx in tqdm(range(len(rs_precision_all))):
        rs_precision = rs_precision_all[idx]
        bertCRF_rs = ' '.join(rs_precision)

        str_BIO = convertList2BIOwithComma(rs_precision)
        bertCRFList.append(str_BIO)

        tl = trueLabelList[idx]

        sentence = df.text[idx]
        text_seg = df.text_seg[idx]
        if str_BIO != tl:
            print('{:d}: '.format(idx))
            print(sentence)
            print(text_seg)
            print(bertCRF_rs)
            print(tl)
            print(str_BIO)
            print('\n')

    score, scoreInfo = getFscoreFromBIOTagList(trueLabelList, bertCRFList)

    print('Eval ' + type + ' results:')
    print('Test F1, Precision, Recall, Acc, No. Tags: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:d}'.format(score[0], \
                                                  score[1], score[2], score[3], scoreInfo[-1]))

    with open(output_eval_file, "a+") as writer:
        writer.write('Eval ' + type + ' results: ')
        writer.write("F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, No. Tags: {:d}\n\n".format(score[0], \
                                                score[1], score[2], score[3], scoreInfo[-1]))

    model.train()
    return score, scoreInfo



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

TEST_FLAG = False

def main(**kwargs):
    if TEST_FLAG:
        kwargs = set_test_param()
    else:
        #kwargs = set_server_param()
        print('load parameters from .sh')

    #kwargs = set_eval_param()
    args._parse(kwargs)

    processors = {
        "ontonotes_cws": lambda: CWS_BMEO(nopunc=args.nopunc),
    }

    if args.do_eval_df and not args.do_train:
        args.init_checkpoint = args.init_checkpoint + '/nhl' \
                + str(args.num_hidden_layers) + '_nte' + str(args.num_train_epochs) \
                + '_nbs' + str(args.train_batch_size) \
                + '_pjs' + str(args.projected_size)

        args.output_dir = args.init_checkpoint + '/out'
        os.makedirs(args.output_dir, exist_ok=True)

        print('init_checkpoint:')
        print(args.init_checkpoint)
        print(args.output_dir)
   
    task_name = args.task_name.lower() 
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # Prepare model
    processor = processors[task_name]()
    label_list = processor.get_labels() # get_labels

    model, device = load_model(label_list, args)

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
        train_dataset, train_dataloader = get_dataset_and_dataloader(processor, args, training=True, type = 'train')
        train_dataset._tokenize()
        num_train_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)
        eval_dataloader = None
        if args.do_eval:
            eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, training=False, type='train')
            eval_dataset._tokenize()
        do_train(model, train_dataloader, optimizer, param_optimizer,
                 device, args, eval_dataloader=eval_dataloader)

    if (args.do_eval) and not (args.do_train):
        eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, training=False, type='test')
        global_step = 0
        eval_fc = do_eval_pretraining if args.pretraining else do_eval
        if args.init_checkpoint is None:
            raise RuntimeError('Evaluating a random initialized model is not supported...!')
        elif os.path.isdir(args.init_checkpoint):
            ckpt_files = sorted(glob(os.path.join(args.init_checkpoint, '*.pt')))
            for ckpt_file in ckpt_files:
                print('Predicting via ' + ckpt_file)
                weights = torch.load(ckpt_file, map_location='cpu')
                try:
                    model.load_state_dict(weights)
                except RuntimeError:
                    model.module.load_state_dict(weights)
                eval_fc(model, eval_dataloader, device, args) # test

                #eval_fc(model, eval_dataloader, device, 0., global_step, args)

                type = 'train'
                eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, False, type)
                eval_fc(model, eval_dataloader, device, args, [], type)

                type='dev'
                eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, False, type) # eval on training data
                eval_fc(model, eval_dataloader, device, args, [], type)

        else:
            eval_fc(model, eval_dataloader, device, args, 'test')


    if (args.do_eval_df) and not (args.do_train):
        output_dir = args.output_dir
        data_dir = args.data_dir
        if args.init_checkpoint is None:
            raise RuntimeError('Evaluating a random initialized model is not supported...!')
        elif os.path.isdir(args.init_checkpoint):
            ckpt_files = sorted(glob(os.path.join(args.init_checkpoint, '*.pt')))
            for ckpt_file in ckpt_files:
                print('Predicting via ' + ckpt_file)
                weights = torch.load(ckpt_file, map_location='cpu')
                try:
                    model.load_state_dict(weights)
                except RuntimeError:
                    model.module.load_state_dict(weights)

                type = 'test'
                df = get_Ontonotes(data_dir, type)
                output_eval_file = os.path.join(output_dir, type+"_eval_results.txt")
                do_eval_df_with_model(model, df, output_eval_file, type)

                type = 'dev'
                df = get_Ontonotes(data_dir, type)
                output_eval_file = os.path.join(output_dir, type+"_eval_results.txt")
                do_eval_df_with_model(model, df, output_eval_file, type)

                type = 'train'
                df = get_Ontonotes(data_dir, type)
                output_eval_file = os.path.join(output_dir, type+"_eval_results.txt")
                do_eval_df_with_model(model, df, output_eval_file, type)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
