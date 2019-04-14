#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 09:28 2019-02-21 
@author: haiqinyang

Feature: 

Scenario: 
"""
import os

from src.pkuseg.metrics import getFscoreFromBIOTagList
from tqdm import tqdm
from src.utilis import get_Ontonotes, convertList2BIOwithComma, BMES2BIO, space2Comma, load_4CWS
import pandas as pd
from src.config import args
from src.preprocess import CWS_BMEO # dataset_to_dataloader, randomly_mask_input, OntoNotesDataset
import time
from src.utilis import get_dataset_and_dataloader

import numpy as np
import torch
import pdb
import re

from src.BERT.modeling import BertConfig
from src.customize_modeling import BertCRFCWS
from src.utilis import save_model

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def do_eval_with_model(model, data_dir, type, output_dir, mode=False):
    df = get_Ontonotes(data_dir, type)

    bertCRFList = []
    trueLabelList = []

    output_diff_file = os.path.join(output_dir, type+"_diff.txt")

    for i, data in tqdm(enumerate(df.itertuples())):
        sentence = data.text
        #sentence = re.sub('“|”', '"', sentence)
        #rs_full = jieba.lcut(sentence, cut_all=True) # Full mode, all possible cuts
        #rs_ser = jieba.lcut_for_search(sentence) # search engine mode, similar to Full mode

        # sentence = '台湾的公视今天主办的台北市长候选人辩论会，'
        # rs_precision = jieba.lcut(sentence, cut_all=False)
        #   rs_precision = ['台湾', '的', '公视', '今天', '主办', '的', '台北', '市长', '候选人', '辩论会', '，']
        # jieba_rs = ' '.join(rs_precision)
        #   jieba_rs = '台湾 的 公视 今天 主办 的 台北 市长 候选人 辩论会 ，'

        #rs_precision = model.cut(sentence, mode)
        rs_precision = model.cutlist(sentence)
        bertCRF_rs = ' '.join(rs_precision)

        #str_precision = convertList2BMES(rs_precision)
        str_BIO = convertList2BIOwithComma(rs_precision, model.tokenizer)
        bertCRFList.append(str_BIO)

        tl = BMES2BIO(data.label)
        tl = space2Comma(tl)
        trueLabelList.append(tl)

        if str_BIO != tl:
            print('{:d}: '.format(i))
            print(sentence)
            print(data.text_seg)
            print(bertCRF_rs)
            print(tl)
            print(str_BIO)
            print('\n')

        with open(output_diff_file, "a+") as writer:
            writer.write('{:d}: '.format(i))
            writer.write(sentence+'\n')
            writer.write(data.text_seg+'\n')
            writer.write(bertCRF_rs+'\n')
            writer.write(tl+'\n')
            writer.write(str_BIO+'\n\n')

    score, scoreInfo = getFscoreFromBIOTagList(trueLabelList, bertCRFList)

    print('Eval ' + type + ' results:')
    print('Test F1, Precision, Recall, Acc, No. Tags: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:d}'.format(score[0], \
                                                  score[1], score[2], score[3], scoreInfo[-1]))

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        writer.write('Eval ' + type + ' results: ')
        writer.write("F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, No. Tags: {:d}\n\n".format(score[0], \
                                                score[1], score[2], score[3], scoreInfo[-1]))

    return score, scoreInfo


def do_eval_df_with_model(model, df, output_diff_file, output_eval_file, type):
    bertCRFList = []
    trueLabelList = []

    sent_list = []
    truelabelstr = ''


    for i, data in tqdm(enumerate(df.itertuples())):
        sentence = data.text
        #sentence = re.sub('“|”', '"', sentence)

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

        if output_diff_file is not None:
            with open(output_diff_file, "a+") as writer:
                writer.write('{:d}: '.format(i))
                writer.write(sentence+'\n')
                writer.write(text_seg+'\n')
                writer.write(bertCRF_rs+'\n')
                writer.write(tl+'\n')
                writer.write(str_BIO+'\n\n')

    score, scoreInfo = getFscoreFromBIOTagList(trueLabelList, bertCRFList)

    print('Eval ' + type + ' results:')
    print('Test F1, Precision, Recall, Acc, No. Tags: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:d}'.format(score[0], \
                                                  score[1], score[2], score[3], scoreInfo[-1]))

    with open(output_eval_file, "a+") as writer:
        writer.write('Eval ' + type + ' results: ')
        writer.write("F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, No. Tags: {:d}\n\n".format(score[0], \
                                                score[1], score[2], score[3], scoreInfo[-1]))

    return score, scoreInfo


def load_BertCRF_model(label_list, args):
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
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if not args.override_output:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        else:
            os.system("rm %s" % os.path.join(args.output_dir, '*'))

    model = BertCRFCWS(device, bert_config, args.vocab_file, args.max_seq_length, len(label_list))

    if args.init_checkpoint is None:
        raise RuntimeError('Evaluating a random initialized model is not supported...!')
    #elif os.path.isdir(args.init_checkpoint):
    #    raise ValueError("init_checkpoint is not a file")
    else:
        weights_path = os.path.join(args.init_checkpoint, WEIGHTS_NAME)

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

    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1 and not args.no_cuda:
        model = torch.nn.DataParallel(model)

    return model, device


def do_train(model, train_dataloader, optimizer, param_optimizer, device, args):
    types = ['test', 'dev', 'train']

    global_step = 0
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
            loss = model(input_ids, segment_ids, input_mask, label_ids)

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

        output_weight_file = os.path.join(args.output_dir, 'weights_epoch%02d.pt'%ep)

        state_dict = model.state_dict()
        if isinstance(model, torch.nn.DataParallel):
            #The model is in a DataParallel container.
            #Its state dict keys are all start with a "module."
            state_dict = OrderedDict({k[len('module.'):]:v for k,v in state_dict.items()})
        torch.save(state_dict, output_weight_file)

        output_model_file = os.path.join(args.output_dir, 'weights_epoch%02d_nhl%d.tsv'%(ep, args.num_hidden_layers))
        save_model(model, output_model_file)

        for ttype in types:
            df = get_Ontonotes(args.data_dir, type)
            output_diff_file = os.path.join(output_dir, type + "_diff.txt")
            output_eval_file = os.path.join(output_dir, type + "_eval_results.txt")

            do_eval_df_with_model(model, df, output_diff_file, output_eval_file, type)


        # logger.info(tr_loss/step)
        tr_loss = tr_loss / step


def eval_eachlayer_ontonotes(args):
    processors = {
        "ontonotes_cws": lambda: CWS_BMEO(nopunc=args.nopunc),
    }

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # Prepare model
    processor = processors[task_name]()

    train_dataset, train_dataloader = get_dataset_and_dataloader(processor, args, training=True, type = 'train')
    train_dataset._tokenize()

    label_list = processor.get_labels() # get_labels

    num_train_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    no_decay = ['bias', 'gamma', 'beta']

    output_dir = args.output_dir
    for nhl in range(12):
        args.num_hidden_layers = nhl+1
        args.output_dir = output_dir + '/nhl' + str(args.num_hidden_layers)
        os.makedirs(output_dir, exist_ok=True)

        model, device = load_BertCRF_model(label_list, args)

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
                 device, args)

        #output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        #with open(output_eval_file, "a+") as writer:
        #    writer.write(args.bert_model + '\n')
        #    writer.write(str(args.num_hidden_layers) + '\n')

        if 0:
            type = 'tmp_test'
            df = get_Ontonotes(data_dir, type)
            output_diff_file = os.path.join(output_dir, type+"_diff.txt")
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            do_eval_df_with_model(model, df, output_diff_file, output_eval_file, type)




def eval_CWS(args):
    fnames = ['as', 'cityu', 'msr', 'pku']
    modes = ['train', 'test']
    tagMode = 'BIO'
    #data_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/cws/'
    data_dir = args.data_dir + tagMode + '/'

    #output_dir='./tmp/cws/jieba/'
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model = preload(args)

    for wt in fnames:
        for md in modes:
            infile = data_dir + wt + '_' + md + '.tsv'
            otag = wt + '_' + md

            df = load_4CWS(infile) #pd.read_csv(infile, sep='\t')        
            output_diff_file = os.path.join(output_dir, otag+"_diff.txt")
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            do_eval_df_with_model(model, df, output_diff_file, output_eval_file, otag)
            #do_eval_with_file_model(model, infile, output_dir, otag, tagMode)


def set_local_eval_ontonotes_param():
    return {'task_name': 'ontonotes_CWS',
            'model_type': 'sequencelabeling',
            'data_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4ner_data/',
            'vocab_file': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/vocab.txt',
            'bert_config_file': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/bert_config.json',
            'output_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/2019_3_23/rs/nhl3/',
            'do_lower_case': True,
            'train_batch_size': 128,
            'max_seq_length': 128,
            'num_hidden_layers': 3,
            'init_checkpoint': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/',
            'bert_model': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/2019_3_23/models/nhl3/weights_epoch03.pt',
            'override_output': True,
            'tensorboardWriter': False
            }

def set_local_eval_4CWS_param():
    return {'task_name': 'ontonotes_CWS',
            'data_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/cws/',
            'vocab_file': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/vocab.txt',
            'bert_config_file': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/bert_config.json',
            'output_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/ontonotes_2019_3_23_nhl3/',
            'do_lower_case': True,
            'train_batch_size': 128,
            'max_seq_length': 128,
            'num_hidden_layers': 3,
            'init_checkpoint': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/',
            'bert_model': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/2019_3_23/models/nhl3/weights_epoch03.pt',
            'override_output': True,
            'tensorboardWriter': False
            #             'model_type': 'sequencelabeling',
            }

def set_server_eval_ontonotes_param():
    return {'task_name': 'ontonotes_CWS',
            'data_dir': '../data/ontonotes5/4ner_data/',
            'vocab_file': '../models/bert-base-chinese/vocab.txt',
            'bert_config_file': '../models/bert-base-chinese/bert_config.json',
            'output_dir': './tmp/ontonotes/',
            'do_lower_case': True,
            'train_batch_size': 128,
            'visible_device': 0,
            'max_seq_length': 128,
            'init_checkpoint': '../models/bert-base-chinese/',
            }

def set_server_eval_4CWS_param():
    return {'task_name': 'ontonotes_CWS',
            'model_type': 'sequencelabeling',
            'data_dir': '../data/CWS/',
            'vocab_file': '../models/bert-base-chinese/vocab.txt',
            'bert_config_file': '../models/bert-base-chinese/bert_config.json',
            'output_dir': './tmp_2019_3_22/out/4CWS/',
            'do_lower_case': True,
            'train_batch_size': 128,
            'max_seq_length': 128,
            'num_hidden_layers': 3,
            'init_checkpoint': '../models/bert-base-chinese/',
            'bert_model': './tmp_2019_3_23/ontonotes/nhl3_nte15_nbs64/weights_epoch03.pt',
            'visible_device': 0,
            'override_output': True,
            'tensorboardWriter': False
            }


LOCAL_FLAG = False
#LOCAL_FLAG = True
#
TEST_ONTONOTES = True
#TEST_ONTONOTES = False
#TEST_CWS = True
TEST_CWS = False

if __name__=='__main__':
    if TEST_ONTONOTES:
        if LOCAL_FLAG:
            kwargs = set_local_eval_ontonotes_param()
        else:
            kwargs = set_server_eval_ontonotes_param()

        args._parse(kwargs)
        eval_eachlayer_ontonotes(args)


    if TEST_CWS:
        if LOCAL_FLAG:
            kwargs = set_local_eval_4CWS_param()
        else:
            kwargs = set_server_eval_4CWS_param()

        args._parse(kwargs)
        eval_CWS(args)

