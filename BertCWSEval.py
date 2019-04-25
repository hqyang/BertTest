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

from src.config import args
from src.preprocess import CWS_BMEO # dataset_to_dataloader, randomly_mask_input, OntoNotesDataset
import torch
import time
from tqdm import tqdm

from src.BERT.modeling import BertConfig
from src.customize_modeling import BertCRFCWS
from src.utilis import save_model
import pdb

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


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
        weights = torch.load(args.bert_model, map_location='cpu')

        try:
            model.load_state_dict(weights)
        except RuntimeError:
            model.module.load_state_dict(weights)

    model.eval()
    save_model(model, args.output_dir + 'model_eval.tsv')

    return model

def set_local_eval_param():
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

def set_server_eval_param():
    return {'task_name': 'ontonotes_CWS',
            'data_dir': '../data/ontonotes5/4ner_data/',
            'vocab_file': '../models/bert-base-chinese/vocab.txt',
            'bert_config_file': '../models/bert-base-chinese/bert_config.json',
            'output_dir': './tmp_2019_3_22/out/ontonotes_eval/',
            'do_lower_case': True,
            'train_batch_size': 128,
            'max_seq_length': 128,
            'num_hidden_layers': 3,
            'init_checkpoint': '../models/bert-base-chinese/',
            'bert_model': './tmp_2019_3_23/ontonotes/nhl3_nte15_nbs64/weights_epoch03.pt',
            'visible_device': 0,
            }


def test_cases(model):
    tt00 = '''
        ✨今日份牛仔外套穿搭打卡|初春一定要有一件万能牛仔外套鸭💯。-我今天又双叒叕没化妆出门逛街了、懒癌晚期间歇性发作哈哈哈哈、。
        -落肩袖、不会显肩宽/后背有涂鸦和蕾丝拼接、见图六/。-Look1:搭配了衬衫和黑灰色牛仔裤/。-Look2：搭配了白色短T和牛仔裤/。
        牛仔裤我尝试了两种颜色、浅色系蓝色牛仔裤整体就偏复古风一点、配深色系就更日常活力一些、。#春天花会开##每日穿搭##日常穿搭#
    '''
    print(tt00)
    t0 = time.time()
    outputT0 = model.cutlist_noUNK([tt00])
    output0 = [' '.join(lst)+' ' for lst in outputT0]
    o0 = '\t'
    for x in output0: o0 += x + '\t'
    print(o0+'\n')
    print('Processing time: ' + str(time.time()-t0))

    tt00 = '''
        #大鱼海棠# #大鱼海棠壁纸# 很感人的一部电影《大鱼海棠》，椿为了救鲲，不惜牺牲自己的一半寿命，湫喜欢椿，
        却把自己的一半寿命给了椿……人一但死后都会化成一条大鱼，椿听我的，数到3，2，1，我们一起跳下去，3.2.1跳，
        我会化成******陪着你。。椿，我喜欢你！！。“北冥有鱼，其名为鲲。”。“鲲之大。”。“一锅炖不下。。“化而为鸟。”。
        “其名为鹏。”。“鹏之大。”。“需要两个烧烤架。”。“一个秘制。”。“一个麻辣。”。“来瓶雪花！！！”。“带你勇闯天涯
    '''
    print(tt00)
    t0 = time.time()
    outputT0 = model.cutlist_noUNK([tt00])
    output0 = [' '.join(lst)+' ' for lst in outputT0]
    o0 = '\t'
    for x in output0: o0 += x + '\t'
    print(o0+'\n')
    print('Processing time: ' + str(time.time()-t0))

    '''
    	# 大 鱼 海棠 # # 大 鱼 海棠 壁纸 # 很 感人 的 一 部 电影 《 大 鱼 海棠 》 ， 椿 为了 救 鲲 ， 不惜 牺牲 
    	自己 的 一半 寿命 ， 湫 喜欢椿 ， 却 把 自己 的 一半 寿命 给 了 椿 …… 人 一但 死 后 都 会 化成 一 条 大 鱼 ， 
    	椿 听 我 的 ， 数 到 3 ， 2 ， 1 ， 我们 一起 跳 下去 ， 3.2.1 跳 ， 我 会 化 成 ****** 陪 着 你 。 。 椿 ， 
    	我 喜欢 你 ！ ！ 。 “ 北冥 有 鱼 ， 其 名 为 鲲 。 ”。“ 鲲 之 大 。 ”。“ 一 锅 炖 不 下 。 。“ 化 而 为 鸟 。 ”。
    	“ 其 名 为 鹏 。 ”。“ 鹏 之 大 。 ”。“ 需要 两 个 烧烤 架 。 ”。“ 一 个 秘制 。 ”。“ 一 个 麻辣 。 ”。“ 来 瓶 
    	雪花 ！ ！！” 。 “ 带 你 勇闯 天涯 	
    '''


def test_ontonotes_file(model, args):
    parts = ['dev', 'train', 'test']

    for part in tqdm(parts):
        text_file = os.path.join(args.data_dir, 'eval_data/ontonotes_'+part+'.txt') #  data in text
        output_file = os.path.join(args.output_dir, 'ontonotes_'+part+'.txt') #  data in text
        st_read = time.time()
        with open(text_file, 'r') as f:
            sents = f.readlines()
        end_read = time.time()
        print('reading time: {:.3f} seconds'.format(end_read-st_read))

        outputT = model.cutlist_noUNK(sents)
        end_decode = time.time()
        print('decode time: {:.3f} seconds'.format(end_decode-end_read))

        with open(output_file, 'a+') as f:
            for lst in outputT:
                outstr = ' '.join(lst)
                f.writelines(outstr + '\n')
        end_write = time.time()
        print('write time: {:.3f} seconds'.format(end_write-end_decode))

        print(part + ' done!')

LOCAL_FLAG = False
#LOCAL_FLAG = True

if __name__=='__main__':
    if LOCAL_FLAG:
        kwargs = set_local_eval_param()
    else:
        kwargs = set_server_eval_param()

    args._parse(kwargs)
    print(args.data_dir+' '+args.output_dir)

    model = preload(args)

    test_cases(model)
    test_ontonotes_file(model, args)

