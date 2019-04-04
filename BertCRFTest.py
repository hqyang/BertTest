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


def load_eval_model(label_list, args):
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

    model = BertCRFCWS(device, bert_config, args.vocab_file, args.max_seq_length, len(label_list), args.train_batch_size)

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

    if args.bert_model is not None:
        weights = torch.load(args.bert_model, map_location='cpu')

        try:
            model.load_state_dict(weights)
        except RuntimeError:
            model.module.load_state_dict(weights)

    return model, device


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

        with open(output_diff_file, "a+") as writer:
            writer.write('{:d}: '.format(i))
            writer.write(sentence+'\n')
            writer.write(text_seg+'\n')
            writer.write(bertCRF_rs+'\n')
            writer.write(tl+'\n')
            writer.write(str_BIO+'\n\n')

    #score, scoreInfo = getFscoreFromBIOTagList([truelabelstr], [str_BIO])
    score, scoreInfo = getFscoreFromBIOTagList(trueLabelList, bertCRFList)

    print('Eval ' + type + ' results:')
    print('Test F1, Precision, Recall, Acc, No. Tags: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:d}'.format(score[0], \
                                                  score[1], score[2], score[3], scoreInfo[-1]))

    with open(output_eval_file, "a+") as writer:
        writer.write('Eval ' + type + ' results: ')
        writer.write("F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, No. Tags: {:d}\n\n".format(score[0], \
                                                score[1], score[2], score[3], scoreInfo[-1]))

    return score, scoreInfo


def do_eval_with_file_model(model, infile, output_dir, otag, tagMode, mode=False):
    # model: BertCRF model
    # infile: input file in tsv format
    # output_dir: the directory to store evaluation file
    # otag: to denote what type of file should be stored
    # tagMode: to indicate the label coding is 'BIO' or 'BMES'

    df = pd.read_csv(infile, sep='\t')

    bertCRFList = []
    trueLabelList = []

    output_diff_file = os.path.join(output_dir, otag+"_diff.txt")

    with open(output_diff_file, "a+") as writer:
        writer.write('order: source, true, prediction\n')

    for i, data in tqdm(enumerate(df.itertuples())):
        sentence = data.text
        #rs_full = jieba.lcut(sentence, cut_all=True) # Full mode, all possible cuts
        #rs_ser = jieba.lcut_for_search(sentence) # search engine mode, similar to Full mode

        # sentence = '台湾的公视今天主办的台北市长候选人辩论会，'
        # rs_precision = jieba.lcut(sentence, cut_all=False)
        #   rs_precision = ['台湾', '的', '公视', '今天', '主办', '的', '台北', '市长', '候选人', '辩论会', '，']
        # jieba_rs = ' '.join(rs_precision)
        #   jieba_rs = '台湾 的 公视 今天 主办 的 台北 市长 候选人 辩论会 ，'
        if tagMode=='BIO':
            tl = data.src_seg
        elif tagMode=='BMES':
            tl = BMES2BIO(data.src_seg)
            tl = space2Comma(tl)

        rs_precision = model.cutlist(sentence)
        bertCRF_rs = ' '.join(rs_precision)

        #str_precision = convertList2BMES(rs_precision)
        str_BIO = convertList2BIOwithComma(rs_precision, model.tokenizer)

        bertCRFList.append(str_BIO)
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

    score, sInfo = getFscoreFromBIOTagList(trueLabelList, bertCRFList)

    print('Eval ' + otag + ' results:')
    print("F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, Token: {:d}\n\n".format(score[0], \
                                              score[1], score[2], score[3], sInfo[-1]))

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        writer.write('Eval ' + otag + ' results: ')
        writer.write("F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, Token: {:d}\n\n".format(score[0], \
                                             score[1], score[2], score[3], sInfo[-1]))

    return score

def do_eval_list_with_file_model(model, infile, output_dir, otag, tagMode, mode=False):
    # model: BertCRF model
    # infile: input file in tsv format
    # output_dir: the directory to store evaluation file
    # otag: to denote what type of file should be stored
    # tagMode: to indicate the label coding is 'BIO' or 'BMES'

    df = pd.read_csv(infile, sep='\t')

    bertCRFList = []
    trueLabelList = []

    output_diff_file = os.path.join(output_dir, otag+"_diff.txt")

    with open(output_diff_file, "a+") as writer:
        writer.write('order: source, true, prediction\n')

    for i, data in tqdm(enumerate(df.itertuples())):
        sentence = data.text
        #rs_full = jieba.lcut(sentence, cut_all=True) # Full mode, all possible cuts
        #rs_ser = jieba.lcut_for_search(sentence) # search engine mode, similar to Full mode

        # sentence = '台湾的公视今天主办的台北市长候选人辩论会，'
        # rs_precision = jieba.lcut(sentence, cut_all=False)
        #   rs_precision = ['台湾', '的', '公视', '今天', '主办', '的', '台北', '市长', '候选人', '辩论会', '，']
        # jieba_rs = ' '.join(rs_precision)
        #   jieba_rs = '台湾 的 公视 今天 主办 的 台北 市长 候选人 辩论会 ，'
        if tagMode=='BIO':
            tl = data.src_seg
        elif tagMode=='BMES':
            tl = BMES2BIO(data.src_seg)
            tl = space2Comma(tl)

        rs_precision = model.cutlist(sentence)
        bertCRF_rs = ' '.join(rs_precision)

        #str_precision = convertList2BMES(rs_precision)
        str_BIO = convertList2BIOwithComma(rs_precision, model.tokenizer)

        bertCRFList.append(str_BIO)
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

    score, sInfo = getFscoreFromBIOTagList(trueLabelList, bertCRFList)

    print('Eval ' + otag + ' results:')
    print("F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, Token: {:d}\n\n".format(score[0], \
                                              score[1], score[2], score[3], sInfo[-1]))

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        writer.write('Eval ' + otag + ' results: ')
        writer.write("F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, Token: {:d}\n\n".format(score[0], \
                                             score[1], score[2], score[3], sInfo[-1]))

    return score

def load_model(label_list, args):
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

def preload(args):
    processors = {
        "ontonotes_cws": lambda: CWS_BMEO(nopunc=args.nopunc),
    }

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # Prepare model
    processor = processors[task_name]()

    label_list = processor.get_labels()
    model, device = load_model(label_list, args)

    if args.bert_model is not None:
        if torch.cuda.is_available():
            weights = torch.load(args.bert_model)
        else:
            weights = torch.load(args.bert_model, map_location='cpu')

        try:
            model.load_state_dict(weights)
        except RuntimeError:
            model.module.load_state_dict(weights)

    model.to(device)
    model.eval()
    save_model(model, args.output_dir + 'model_eval.tsv')

    return model


def eval_ontonotes(args):
    #data_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data'

    #output_dir='./tmp/ontonotes/BerTCRF/'
    localtime = time.localtime(time.time())
    data_dir = args.data_dir
    output_dir = args.output_dir + 'nhl' + str(args.num_hidden_layers) + '_' + str(localtime.tm_year) \
                 + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) \
                 + '_' + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec)
    os.makedirs(output_dir, exist_ok=True)

    model = preload(args)

    text1 = '''
        目前由２３２位院士（Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）
        ２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ）及２位通信協院士
        （Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者）
        # of students is 256.
    '''
    outputT1 = model.cut2(text1)
    print(text1)
    print(' '.join(outputT1)+'\n')

    text2 = '款款好看的美甲，简直能搞疯“选择综合症”诶！。这是一组超级温柔又带点设计感的美甲💅。春天来了🌺。美甲也从深色系转变为淡淡的浅色系了💐。今天给大家推荐最适合春天的美甲💅。希望你们会喜欢~😍@MT小美酱 @MT情报局 @美图秀秀 #春季美甲##显白美甲##清新美甲##ins美甲#'
    outputT2 = model.cut2(text2)
    print(text2)
    print(' '.join(outputT2)+'\n')


    text3 = '''
    在朝野各界为核四事件吵嚷不休之际，发生在一月中旬的垦丁龙坑生态油污事件，直到二月底才受到初步控制，加上近来台湾山区森林火灾屡扑屡起，显现台湾生态的危机，已不容人们将焦点放在单一的开发事件上，全面性的大地破坏与自然反扑更值得关注。
    '''
    outputT3 = model.cutlist([text3])
    output3 = [' '.join(lst) for lst in outputT3]
    print(text3)
    print(output3[0]+'\n')

    text4 = '''
      创维又一波黑科技。  尸体卡通风管。  这次撞上海底了。  欢迎新老师生前来就餐。  工信处女干事每月经过下属科室都亲口交代24口交换机等技术性器件的安装工程。  
      结婚和尚未结婚的的确在干扰分词哈。  商品和服务。  买水果然后来世博会最后去世博会。  中国的首都是北京。  
      随着页游兴起到现在页游繁盛，依赖于存档进行逻辑判断的设计减少了，但这块都不能完全忽略掉。  
    '''
    outputT4 = model.cutlist([text1, text2, text3, text4])
    output4 = [' '.join(lst) for lst in outputT4]
    print(text1+'\t'+text2+'\t'+text3+'\t'+text4)
    o4 = ''
    for x in output4: o4 += x + '\t'
    print(o4+'\n')

    t5 = '''
        单枪匹马逛英国——伦敦篇。伦敦就是这个样子初次来到这个“老牌资本主义”的“雾都“，
        就像回到了上海，一幢幢不高的小楼，显得异常陈旧，很多楼房被数百年烟尘熏的就像被刷了一层黑色的油漆，
        油光锃亮，如果不是旁边的楼房正在清洗，很难让人相信如今的伦敦是饱经污染沧桑后及时刹车的高手，
        因为一座现代化的国际大都市也是有不少楼房是黑色的呢，黑色显得凝重、高雅，但是绝对不能靠油烟去熏……堵车，
        是所有大都市的通病，虽然不足为怪，但是，1988年的北京还没有那么多的车，也没有全城大堵车的现象，
        有的是刚刚开始的“靠油烟和汽车的尾气烟熏火燎美丽的古城”，有谁能够想到，短短的十年，北京就气喘吁吁的追赶上了伦敦，
        没有一条洁净的河流，没有清新的空气，有的是让人窒息的空气污染…….以及，让人始料未及的全城大堵车。
        如果，我们那些负责城市建设规划的先生们，在国外，不只只是游山玩水的话，带回别人的教训、总结别人的经验的话，
        我们这个被穷祖先毁的“一塌糊涂”的脆弱的生态环境也不会再经受20世纪90年代的现代化的大污染了。但是，
        伦敦是一座改过自新的城市，人家痛定思痛，紧急刹车，及时的治理了污染，我们在泰吾士河里可以看到鱼儿在自由的翻滚，
        天空湛蓝，翠绿的草地与兰天辉映着，一片“污染大战”后的和平景象    
    '''
    outputT5 = model.cutlist_noUNK([t5])
    output5 = [' '.join(lst) for lst in outputT5]
    print(t5)
    print(output5[0]+'\n')

    t6 = '中国队以１１：１战胜伊朗队。中国队在当晚的另一场对阵巴勒斯坦队的比赛中同样获胜，比分是９９：２０。'
    outputT6 = model.cutlist_noUNK([t6])
    output6 = [' '.join(lst) for lst in outputT6]
    print(t6)
    print(output6[0]+'\n')

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        writer.write(args.bert_model + '\n')
        writer.write(str(args.num_hidden_layers) + '\n')

    mode = False
    #mode = True
    type = 'tmp_test'
    df = get_Ontonotes(data_dir, type)
    output_diff_file = os.path.join(output_dir, type+"_diff.txt")
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    do_eval_df_with_model(model, df, output_diff_file, output_eval_file, type)

    type = 'test'
    df = get_Ontonotes(data_dir, type)
    output_diff_file = os.path.join(output_dir, type+"_diff.txt")
    output_eval_file = os.path.join(output_dir, "eval_results.txt")

    do_eval_df_with_model(model, df, output_diff_file, output_eval_file, type)

    #do_eval_with_model(model, data_dir, type, output_dir, mode)

    type = 'dev'
    df = get_Ontonotes(data_dir, type)
    output_diff_file = os.path.join(output_dir, type+"_diff.txt")
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    do_eval_df_with_model(model, df, output_diff_file, output_eval_file, type)
    #do_eval_with_model(model, data_dir, type, output_dir, mode)

    type = 'train'
    df = get_Ontonotes(data_dir, type)
    output_diff_file = os.path.join(output_dir, type+"_diff.txt")
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    do_eval_df_with_model(model, df, output_diff_file, output_eval_file, type)
    #do_eval_with_model(model, data_dir, type, output_dir, mode)


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
            'model_type': 'sequencelabeling',
            'data_dir': '../data/ontonotes5/4ner_data/',
            'vocab_file': '../models/bert-base-chinese/vocab.txt',
            'bert_config_file': '../models/bert-base-chinese/bert_config.json',
            'output_dir': './tmp_2019_3_22/out/ontonotes/',
            'do_lower_case': True,
            'train_batch_size': 128,
            'visible_device': 0,
            'max_seq_length': 128,
            'num_hidden_layers': 3,
            'init_checkpoint': '../models/bert-base-chinese/',
            'bert_model': './tmp_2019_3_23/ontonotes/nhl3_nte15_nbs64/weights_epoch03.pt',
            'override_output': True,
            'tensorboardWriter': False
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
LOCAL_FLAG = True
#TEST_CWS = False
TEST_ONTONOTES = True
#TEST_ONTONOTES = False
TEST_CWS = True

if __name__=='__main__':
    if TEST_ONTONOTES:
        if LOCAL_FLAG:
            kwargs = set_local_eval_ontonotes_param()
        else:
            kwargs = set_server_eval_ontonotes_param()

        args._parse(kwargs)
        eval_ontonotes(args)


    if TEST_CWS:
        if LOCAL_FLAG:
            kwargs = set_local_eval_4CWS_param()
        else:
            kwargs = set_server_eval_4CWS_param()

        args._parse(kwargs)
        eval_CWS(args)

