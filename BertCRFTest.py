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
from src.utilis import get_Ontonotes, convertList2BIOwithComma, BMES2BIO, space2Comma
import pandas as pd
from src.config import args
from src.preprocess import CWS_BMEO # dataset_to_dataloader, randomly_mask_input, OntoNotesDataset
import time

import numpy as np
import torch
import pdb

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
        #rs_full = jieba.lcut(sentence, cut_all=True) # Full mode, all possible cuts
        #rs_ser = jieba.lcut_for_search(sentence) # search engine mode, similar to Full mode

        # sentence = '台湾的公视今天主办的台北市长候选人辩论会，'
        # rs_precision = jieba.lcut(sentence, cut_all=False)
        #   rs_precision = ['台湾', '的', '公视', '今天', '主办', '的', '台北', '市长', '候选人', '辩论会', '，']
        # jieba_rs = ' '.join(rs_precision)
        #   jieba_rs = '台湾 的 公视 今天 主办 的 台北 市长 候选人 辩论会 ，'

        rs_precision = model.cut(sentence, mode)
        bertCRF_rs = ' '.join(rs_precision)

        #str_precision = convertList2BMES(rs_precision)
        str_BIO = convertList2BIOwithComma(rs_precision)
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
            #pdb.set_trace()

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

    return score


def do_eval_with_file_model(model, infile, output_dir, otag, tagMode, mode=False):
    # model: BertCRF model
    # infile: input file in tsv format
    # output_dir: the directory to store evaluation file
    # otag: to denote what type of file should be stored
    # tagMode: to indicate the label coding is 'BIO' or 'BMES'

    df = pd.read_csv(infile, sep='\t')

    jiebaList = []
    trueLabelList = []

    output_diff_file = os.path.join(output_dir, otag+"_diff.txt")

    with open(output_diff_file, "a+") as writer:
        writer.write('order: source, true, jieba\n')

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

        rs_precision = model.cut(sentence, mode)
        bertCRF_rs = ' '.join(rs_precision)

        #str_precision = convertList2BMES(rs_precision)
        str_BIO = convertList2BIOwithComma(rs_precision)

        bertCRFList.append(str_BIO)
        trueLabelList.append(tl)

        if i % 20000 == 0:
            print('{:d}: '.format(i))
            print(sentence)
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

    model = BertCRFCWS(bert_config, args.vocab_file, args.max_seq_length, len(label_list))

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
        weights = torch.load(args.bert_model, map_location='cpu')

        try:
            model.load_state_dict(weights)
        except RuntimeError:
            model.module.load_state_dict(weights)

    model.eval()
    save_model(model, args.output_dir + 'model_eval.tsv')
    #pdb.set_trace()

    return model


def test_ontonotes(args):
    #data_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data'

    #output_dir='./tmp/ontonotes/BerTCRF/'
    localtime = time.localtime(time.time())
    data_dir = args.data_dir
    output_dir = args.output_dir + 'nhl' + str(args.num_hidden_layers) + '_' + str(localtime.tm_year) \
                 + '_' + str(localtime.tm_mon) + '_' + str(localtime.tm_mday) \
                 + '_' + str(localtime.tm_hour) + str(localtime.tm_min) + str(localtime.tm_sec)
    os.makedirs(output_dir, exist_ok=True)

    model = preload(args)

    text = '''
        目前由２３２位院士（Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）
        ２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ）及２位通信協院士
        （Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者）
        # of students is 256.
    '''

    outputF = model.cut(text, False)

    outTextF = ' '.join(outputF)
    print(outTextF)

    outputT = model.cut(text)

    outTextT = ' '.join(outputT)
    print(outTextT)

    t2 = '''
    在朝野各界为核四事件吵嚷不休之际，发生在一月中旬的垦丁龙坑生态油污事件，直到二月底才受到初步控制，加上近来台湾山区森林火灾屡扑屡起，显现台湾生态的危机，已不容人们将焦点放在单一的开发事件上，全面性的大地破坏与自然反扑更值得关注。
    '''
    outputT2 = model.cut(t2)
    outTextT2 = ' '.join(outputT2)
    print(outTextT2)

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        writer.write(args.bert_model + '\n')  
        writer.write(str(args.num_hidden_layers) + '\n')
     
    mode = False
    #mode = True
    type = 'test'
    do_eval_with_model(model, data_dir, type, output_dir, mode)

    type = 'dev'
    do_eval_with_model(model, data_dir, type, output_dir, mode)

    type = 'train'
    do_eval_with_model(model, data_dir, type, output_dir, mode)

def test_CWS(args):
    fnames = ['as', 'cityu', 'msr', 'pku']
    modes = ['train', 'test']
    tagMode = 'BIO'
    data_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/cws/'
    data_dir += tagMode + '/'

    output_dir='./tmp/cws/jieba/'
    os.makedirs(output_dir, exist_ok=True)

    model = preload(args)

    for wt in fnames:
        for md in modes:
            infile = data_dir + wt + '_' + md + '.tsv'
            otag = wt + '_' + md
            do_eval_with_file_model(model, infile, output_dir, otag, tagMode)


def set_local_eval_param():
    return {'task_name': 'ontonotes_CWS',
            'model_type': 'sequencelabeling',
            'data_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4ner_data/',
            #'bert_model_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data/eval/2019_3_12/models/',
            'vocab_file': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/vocab.txt',
            'bert_config_file': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/bert_config.json',
            'output_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/2019_3_12/rs/nhl3/',
            'do_lower_case': True,
            'train_batch_size': 128,
            'max_seq_length': 128,
            'num_hidden_layers': 3,
            'init_checkpoint': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/',
            'bert_model': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/2019_3_12/models/nhl3/weights_epoch03.pt',
            'override_output': True,
            'tensorboardWriter': False
            }

def set_server_eval_param():
    return {'task_name': 'ontonotes_CWS',
            'model_type': 'sequencelabeling',
            'data_dir': '../data/ontonotes5/',
            'vocab_file': '../models/bert-base-chinese/vocab.txt',
            'bert_config_file': '../models/bert-base-chinese/bert_config.json',
            'output_dir': './tmp_2019_3_20/out/',
            'do_lower_case': True,
            'train_batch_size': 128,
            'max_seq_length': 128,
            'num_hidden_layers': 3,
            'init_checkpoint': '../models/bert-base-chinese/',
            'bert_model': './tmp_2019_3_20/ontonotes/nhl3_nte15_nbs64/weights_epoch03.pt',
            'no_cuda': True,
            'override_output': True,
            'tensorboardWriter': False
            }

LOCAL_FLAG = False

if __name__=='__main__':
    if LOCAL_FLAG:
        kwargs = set_local_eval_param()
    else:
        kwargs = set_server_eval_param()

    args._parse(kwargs)
    test_ontonotes(args)
    #test_CWS()
    #do_eval_with_file('tmp/cws/tmp.txt', 'tmp', '', 'BIO')
    #test_CWS()
