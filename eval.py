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

import sys
sys.path.append('./src')

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
from src.customize_modeling import BertCWS
from src.BERT.optimization import BertAdam
from src.pkuseg.metrics import getFscoreFromBIOTagList

from src.preprocess import randomly_mask_input, OntoNotesDataset, CWS_BMEO
from src.config import args
from src.tokenization import FullTokenizer
from src.utilis import get_dataset_and_dataloader, load_4CWS, convertList2BIOwithComma, BMES2BIO, space2Comma


CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


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
        print('num_hidden_layers: ' + str(bert_config.num_hidden_layers))

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if not args.override_output:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        else:
            # os.system("rm %s" % os.path.join(args.output_dir, '*'))
            # os.system('mkdir %s' % args.output_dir+'/old')
            # os.system('chmod 777 %s' % args.output_dir+'/old')
            # os.system('mv %s %s' % (os.path.join(args.output_dir, '*'), args.output_dir+'/old'))
            print(args.output_dir)

    model = BertCWS(device, bert_config, args.vocab_file, args.max_seq_length, len(label_list),
                    args.train_batch_size, args.fclassifier)

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


def do_eval_df_with_model(model, df, part, output_eval_file, output_diff_file):
    model.eval() # important

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
        bert_rs = ' '.join(rs_precision)

        str_BIO = convertList2BIOwithComma(rs_precision)
        bertCRFList.append(str_BIO)

        tl = trueLabelList[idx]

        sentence = df.text[idx]
        text_seg = df.text_seg[idx]
        if str_BIO != tl:
            print('{:d}: '.format(idx))
            print(sentence)
            print(text_seg)
            print(bert_rs)
            print(tl)
            print(str_BIO)
            print('\n')
            with open(output_diff_file, 'a+') as writer:
                writer.write('{0}: {1} \n {2}\n'.format(idx, sentence, text_seg))
                writer.write(bert_rs + '\n')
                writer.write(tl + '\n')
                writer.write(str_BIO + '\n\n')

    score, scoreInfo = getFscoreFromBIOTagList(trueLabelList, bertCRFList)

    print('Eval ' + part + ' results:')
    print('Test F1, Precision, Recall, Acc, No. Tags: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:d}'.format(score[0], \
                                                  score[1], score[2], score[3], scoreInfo[-1]))

    with open(output_eval_file, "a+") as writer:
        writer.write('Eval ' + part + ' results: ')
        writer.write("F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, No. Tags: {:d}\n\n".format(score[0], \
                                                score[1], score[2], score[3], scoreInfo[-1]))

    model.train()
    return score, scoreInfo


def set_server_eval_param():
    return {'model_type': 'sequencelabeling',
            'data_dir': '../data/',
            'output_dir': './tmp/',
            'bert_model_dir': '../models/bert-base-chinese/',
            'vocab_file': '../models/bert-base-chinese/vocab.txt',
            'init_checkpoint': './tmp/4CWS/ModelSize/MSR/Softmax/fine_tune/l12/',
            'max_seq_length': 128,
            'do_lower_case': True,
            'do_eval_df': True,
            'train_batch_size': 32,
            'method': 'fine_tune',
            'fclassifier': 'Softmax',
            'override_output': True,
            'visible_device': 0,
            'num_hidden_layers': 12
            }


def eval_layers(kwargs):
    args._parse(kwargs)

    #datasets = ['PKU', 'MSR']
    #fclassifiers = ['CRF', 'Softmax']
    parts = ['train', 'test']

    #--num_hidden_layers 12 \
    #--train_batch_size 32 \
    #--num_train_epochs 30

    processors = {
        'ontonotes': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['full_pos', 'bert_ner', 'src_ner', 'src_seg', 'text_seg']),
        '4cws_cws': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']),
        'msr': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']),
        'pku': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']),
        'as': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']),
        'cityu': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg'])
    }


    #for dataset in datasets:
    args.data_dir += args.task_name

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # Prepare model
    processor = processors[task_name]()
    label_list = processor.get_labels() # get_labels

    output_dir = os.path.join(args.output_dir, args.task_name + '/rs/')
    os.system('mkdir ' + output_dir)
    os.system('chmod 777 ' + output_dir)

    # tmp/4CWS/PKU/rs
    args.init_checkpoint = args.output_dir + args.task_name + '/' + args.fclassifier \
                            + '/' + args.method + '/l' + str(args.num_hidden_layers)

    print(args.init_checkpoint)

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

    #eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, training=False, part='test')

    global_step = 0
    if args.init_checkpoint is None:
        raise RuntimeError('Evaluating a random initialized model is not supported...!')
    elif os.path.isdir(args.init_checkpoint):
        ckpt_files = sorted(glob(os.path.join(args.init_checkpoint, '*.pt')))

        for ckpt_file in ckpt_files:
            print('Predicting via ' + ckpt_file)
            wfn, ext = os.path.splitext(ckpt_file)
            wfn_used = wfn.split('/')[-1]

            weights = torch.load(ckpt_file, map_location='cpu')
            try:
                model.load_state_dict(weights)
            except RuntimeError:
                model.module.load_state_dict(weights)

            for part in parts:
                df = load_4CWS(os.path.join(args.data_dir, part+".tsv"))#get_Ontonotes(args.data_dir, part)

                sfn = part + '_ft_l' + str(args.num_hidden_layers) + '_' + args.fclassifier + '_' + wfn_used + '.txt'
                dfn = part + '_ft_l' + str(args.num_hidden_layers) + '_' + args.fclassifier + '_' + wfn_used + '_diff.txt'

                output_eval_file = os.path.join(output_dir, sfn)
                output_diff_file = os.path.join(output_dir, dfn)
                do_eval_df_with_model(model, df, part, output_eval_file, output_diff_file)


def eval_dataset(args):
    parts = ['train', 'test']

    processors = {
        'ontonotes': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['full_pos', 'bert_ner', 'src_ner', 'src_seg', 'text_seg']),
        '4cws_cws': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']),
        'msr': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']),
        'pku': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']),
        'as': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg']),
        'cityu': lambda: CWS_BMEO(nopunc=args.nopunc, drop_columns=['src_seg', 'text_seg'])
    }

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

    #eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, training=False, part='test')

    global_step = 0
    if args.init_checkpoint is None:
        raise RuntimeError('Evaluating a random initialized model is not supported...!')
    elif os.path.isdir(args.init_checkpoint):
        ckpt_files = sorted(glob(os.path.join(args.init_checkpoint, '*.pt')))

        for ckpt_file in ckpt_files:
            print('Predicting via ' + ckpt_file)
            wfn, ext = os.path.splitext(ckpt_file)
            wfn_used = wfn.split('/')[-1]

            weights = torch.load(ckpt_file, map_location='cpu')
            try:
                model.load_state_dict(weights)
            except RuntimeError:
                model.module.load_state_dict(weights)

            for part in parts:
                df = load_4CWS(os.path.join(args.data_dir, part+".tsv"))#get_Ontonotes(args.data_dir, part)

                sfn = args.task_name + '_' + part + '_ft_l' + str(args.num_hidden_layers) + '_' + args.fclassifier + '_' + wfn_used + '.txt'
                dfn = args.task_name + '_' + part + '_ft_l' + str(args.num_hidden_layers) + '_' + args.fclassifier + '_' + wfn_used + '_diff.txt'

                output_eval_file = os.path.join(output_dir, sfn)
                output_diff_file = os.path.join(output_dir, dfn)
                do_eval_df_with_model(model, df, part, output_eval_file, output_diff_file)


def main(**kwargs):
    #print('load initialized parameter from server')
    #
    # eval_layers(kwargs)

    kwargs = set_server_eval_param()
    args._parse(kwargs)

    datasets = ['AS', 'CITYU', 'MSR', 'PKU', 'ONTONOTES']
    trained_datasets = ['MSR', 'PKU']

    fclassifiers = ['CRF', 'Softmax']
    #init_dirs = [
    #    #'./tmp/4CWS/ModelSize/MSR/Softmax/fine_tune/l12/',
    #    './tmp/4CWS/ModelSize/MSR/CRF/fine_tune/l12/',
    #    './tmp/4CWS/ModelSize/PKU/Softmax/fine_tune/l12/',
    #    './tmp/4CWS/ModelSize/PKU/CRF/fine_tune/l12/'
    #]
    data_dir_init = args.data_dir
    output_dir_init = args.output_dir

    for dataset in datasets:
        args.task_name = dataset
        args.data_dir = data_dir_init
        #args.output_dir = output_dir_init

        if dataset == 'ONTONOTES':
            args.data_dir += 'ontonotes5/4ner_data/'
        else:
            args.data_dir += 'CWS/BMES/' + dataset

        for trained_dataset in trained_datasets:
            for fclassifier in fclassifiers:
                args.init_checkpoint = output_dir_init + '4CWS/ModelSize/' + trained_dataset + \
                                       '/' + fclassifier + '/fine_tun/l12'

                if dataset == 'ONTONOTES':
                    args.output_dir = output_dir_init + '/ontonotes/' + trained_dataset + '/eval_'+ dataset + '_' \
                            + fclassifier + '_' + 'ft_l12'
                    os.system('mkdir ' + args.output_dir)
                    os.system('chmod 777 ' + args.output_dir)
                else:
                    args.output_dir = output_dir_init + '4CWS/ModelSize/' + trained_dataset + '/eval_' + dataset + '_' \
                            + fclassifier + '_' + 'ft_l12'
                    os.system('mkdir ' + args.output_dir)
                    os.system('chmod 777 ' + args.output_dir)

                print('init_checkpoint: ' + args.init_checkpoint)
                print('data_dir: ' + args.data_dir)
                print('output_dir: ' + args.output_dir)

                eval_dataset(args)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
