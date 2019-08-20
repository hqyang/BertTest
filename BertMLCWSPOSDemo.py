#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 09:28 2019-02-21 
@author: haiqinyang

Feature: 

Scenario: 
"""
import os

from src.config import args
from src.preprocess import CWS_BMEO, CWS_POS # dataset_to_dataloader, randomly_mask_input, OntoNotesDataset
import torch
import time

from src.BERT.modeling import BertConfig
from src.customize_modeling import BertMLCWSPOS
from src.utilis import save_model
from tqdm import tqdm

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def load_CWS_POS_model(label_list, pos_label_list, args):
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
            os.system("rm %s" % os.path.join(args.output_dir, '*'))

    model = BertMLCWSPOS(device, bert_config, args.vocab_file, args.max_seq_length, len(label_list),
                       len(pos_label_list), batch_size=args.train_batch_size,
                         do_lower_case=args.do_lower_case, do_mask_as_whole=args.do_mask_as_whole)

    if args.init_checkpoint is None:
        raise RuntimeError('Evaluating a random initialized models is not supported...!')
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

def preload(args):
    processors = {
        "ontonotes_cws": lambda: CWS_BMEO(nopunc=args.nopunc),
        'ontonotes_cws_pos': lambda: CWS_POS(nopunc=args.nopunc, drop_columns=None, \
                                     pos_tags_file='./resource/pos_tags.txt'),
    }

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    # Prepare models
    processor = processors[task_name]()

    label_list = processor.get_labels()
    pos_label_list = processor.get_POS_labels()

    model, device = load_CWS_POS_model(label_list, pos_label_list, args)

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
    return {'task_name': 'ontonotes_cws_pos',
            'model_type': 'sequencelabeling',
            'data_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data'
                        '4nerpos_data/valid',
            'vocab_file': './src/BERT/models/multi_cased_L-12_H-768_A-12/vocab.txt',
            'bert_config_file': './src/BERT/models/multi_cased_L-12_H-768_A-12/bert_config.json',
            'output_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/ontonotes/CWSPOS2/L6/',
            'do_lower_case': False,
            'do_mask_as_whole': True,
            'train_batch_size': 64,
            'max_seq_length': 128,
            'num_hidden_layers': 6,
            'bert_model': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/' \
                          'ontonotes/CWSPOS2/uncased_l6_cws_F1_weights_epoch16.pt',
            'init_checkpoint': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/multi_cased_L-12_H-768_A-12/',
            'override_output': True,
            }
#            'bert_model': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/' \
#                          'ontonotes/CWSPOS2/uncased_l6_cws_F1_weights_epoch16.pt',
#            'bert_model': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/' \
#                          'ontonotes/CWSPOS2/cased_cws_F1_weights_epoch17.pt',


def set_server_eval_param():
    return {'task_name': 'ontonotes_cws_pos',
            'model_type': 'sequencelabeling',
            'data_dir': '../data/ontonotes5/4ner_data/',
            'vocab_file': './src/BERT/models//multi_cased_L-12_H-768_A-12/vocab.txt',
            'bert_config_file': './src/BERT/models/multi_cased_L-12_H-768_A-12/bert_config.json',
            'output_dir': './tmp/ontonotes/out/',
            'do_lower_case': False,
            'train_batch_size': 128,
            'max_seq_length': 128,
            'num_hidden_layers': 3,
            'init_checkpoint': '../models/multi_cased_L-12_H-768_A-12/',
            'bert_model': './tmp/ontonotes/l3/cws_F1_weights_epoch16.pt',
            'no_cuda': True,
            'override_output': True
            }

def extract_CWSPOS(model, t1):
    outputT0 = model.cutlist_noUNK([t1])
    output0 = ['    '.join(lst)+' ' for lst in outputT0]
    o0 = '\t'
    for x in output0: o0 += x + '\t'
    print(t1+'\n')
    print(o0+'\n')


def test_cases(model):
    tt00 = '''
    【上新】💰145     。斯图西牛油果绿🥑人像印花短袖T恤，胸前logo刺绣，男女同款，采用纯棉面料，柔软舒适。设计简单大方，配上今年夏季最流行的牛油果绿🥑，衬托肤色，清新一夏！休闲百搭、潮流有范。。颜色：牛油果绿   。尺码：M-XXL。尺码表图9⃣️
    '''
    extract_CWSPOS(model, tt00)

    '''
	【 / PU    上新 / VV    】 / PU    💰145      / PU    。 / PU    斯图西 / NR    牛油果绿 / NN    🥑 / PU    人 / NN    
	像 / NN    印花 / NN    短袖 / NN    T恤 / NN    ， / PU    胸 / NN    前 / LC    logo / NN    刺绣 / NN    ， / PU    
	男 / NN    女 / NN    同 / NN    款 / NN    ， / PU    采用 / PU    纯棉 / JJ    面料 / NN    ， / PU    柔软 / VA    
	舒适 / VA    。 / PU    设计 / NN    简单 / VA    大方 / VA    ， / PU    配上 / VV    今年 / NT    夏季 / NT    
	最 / AD    流行 / VV    的 / DEC    牛油果绿 / NN    🥑 / PU    ， / PU    衬托 / VV    肤色 / NN    ， / PU    
	清新 / NN    一 / CD    夏 / NN    ！ / PU    休闲 / NN    百搭 / NN    、 / PU    潮流 / NN    有范 / VA    。 / PU    
	。 / PU    颜色 / NN    ： / PU    牛油果 / NN    绿 / NR       。 / PU    尺码 / NN    ： / PU    M / PU    - / NN    
	XXL / X    。 / X    尺码 / NN    表图 / NN 	
    '''

    # 2019-7-29 v1
    '''
    【 / PU    上新 / VV    】 / PU    💰145 / NN    。 / PU    斯图西牛 / NR    油果 / NN    绿 / NN    🥑 / NN    人 / NN    
    像 / P    印花 / NN    短袖 / NN    T恤 / NN    ， / PU    胸前 / NN    logo / NN    刺绣 / NN    ， / PU    男女 / NN    
    同款 / NN    ， / PU    采用 / VV    纯 / NN    棉 / NN    面料 / NN    ， / PU    柔软 / JJ    舒适 / VA    。 / PU    
    设计 / NN    简单 / VA    大方 / NN    ， / PU    配上 / VV    今年 / NT    夏季 / NT    最 / AD    流行 / VV    的 / DEC    
    牛 / NN    油果 / NN    绿 / NN    🥑 / NN    ， / PU    衬托 / VV    肤色 / NN    ， / PU    清新 / VA    一夏 / NT    
    ！ / PU    休闲 / VV    百搭 / NN    、 / PU    潮流 / NN    有范 / VA    。 / PU    。 / PU    颜色 / NN    ： / PU    
    牛 / NN    油果 / NN    绿 / NN    。 / PU    尺码 / NN    ： / PU    M-XX / NN    L / PU    。尺 / NN    码表 / NN    
    图 / PU 	
    '''
    # 2019-7-29 v2
    '''
    【 / PU    上新 / VV    】 / PU    💰145 / NN    。 / PU    斯图西牛 / NR    油果 / NN    绿 / NN    🥑 / NN    人 / NN
    像 / P    印花 / NN    短袖 / NN    T恤 / NN    ， / PU    胸前 / NN    logo / NN    刺绣 / NN    ， / PU    男女 / NN    
    同款 / NN    ， / PU    采用 / VV    纯 / NN    棉 / NN    面料 / NN    ， / PU    柔软 / JJ    舒适 / VA    。 / PU    
    设计 / NN    简单 / VA    大方 / NN    ， / PU    配上 / VV    今年 / NT    夏季 / NT    最 / AD    流行 / VV    的 / DEC    
    牛 / NN    油果 / NN    绿 / NN    🥑 / NN    ， / PU    衬托 / VV    肤色 / NN    ， / PU    清新 / VA    一夏 / NT    
    ！ / PU    休闲 / VV    百搭 / NN    、 / PU    潮流 / NN    有范 / VA    。 / PU    。 / PU    颜色 / NN    ： / PU    
    牛 / NN    油果 / NN    绿 / NN    。 / PU    尺码 / NN    ： / PU    M-XXL / NN    。 / PU    尺码 / NN    表图 / NN 9⃣️ / PU 	
    '''

    tt00 = '''
    #仙女屋♡# #樱雪头像库-# #eve女孩星痕# #eve女孩茜茜# #eve女孩樱雪#。处关系啦吼吼吼。大爱@.近我者软♡ @.近我者仙♡ 。闺闺@-笑里有盈盈秋波. 。妹妹@두눈을감다_沈熙妍 。哥哥@顾沐辰心动氿氿💨 。好像就没了八🌝
    '''
    extract_CWSPOS(model, tt00)

    # 2019-7-29 v1
    '''
    # / PU    仙女屋 / NN    ♡ / PU    # / PU    # / PU    樱雪头 / NR    像 / NN    库 / NN    - / PU    # / PU    # / PU    
    ev / NR    e女 / NN    孩星 / NN    痕 / PU    # / PU    # / NR    eve / NN    女孩 / NR    茜 / PU    茜 / PU    # / NR    
    #ev / NN    e女 / NR    孩 / PU    樱 / PU    雪 / NN    #。 / NN    处 / SP    关系啦 / VV    吼 / PU    吼吼 / NN    
    。大爱 / PU    @. / PN    近 / NR    我 / NR    者软♡ / PU    @. / PN    近我 / NR    者 / PU    仙♡ / NN    。闺 / PU    
    闺 / NN    @ / LC    -笑里 / VE    有盈 / NR    盈 / PU    秋 / PU    波. / NN    。 / PU    妹妹 / NR    @두눈 / NR    
    을 / PU    감다 / NN    _ / PU    沈熙妍。哥 / NR    哥 / PU    @顾 / VV    沐 / PU    辰心 / X    动 / AD    氿氿💨 / AS    
    。 / CD 	
    '''
    # 2019-7-29 v2
    '''
    # / PU    仙女屋 / NN    ♡ / PU    # / PU    # / PU    樱雪头 / NR    像 / NN    库 / NN    - / PU    # / PU    # / PU    
    eve / NR    女孩 / NN    星痕 / NN    # / PU    # / PU    eve / NR    女孩 / NN    茜茜 / NR    # / PU    # / PU    
    eve / NR    女孩 / NN    樱雪 / NR    # / PU    。 / PU    处 / NN    关系 / NN    啦 / SP    吼吼吼 / VV    。 / PU    
    大爱 / NN    @.近 / PU    我者 / PN    软 / NR    ♡ / NR    @.近 / PU    我者 / PN    仙♡ / NR    。 / PU    闺闺 / NN    
    @- / PU    笑 / NN    里 / LC    有盈盈 / VE    秋波 / NR    . / PU    。 / PU    妹妹 / NN    @ / PU    두눈을감다_ / NR    
    沈熙妍 / NR    。 / PU    哥哥 / NN    @ / PU    顾沐辰心动 / NR    氿氿💨 / VV    。 / PU    好像 / X    就 / AD   
     没 / VE    了 / AS    八🌝 / CD 	
    '''

    tt00 = '''
     大家加油(ง •̀_•́)ง
    '''
    extract_CWSPOS(model, tt00)
    '''
    大家 / PN    加油 / VV    ( / PU    ง / IJ     •̀ / PU    _•́) / PU    ง / X 
    '''
    # 2019-7-29 v2
    '''
    大家 / PN    加油 / VV    ( / PU    ง• / X    ̀_•́ / NR    ) / PU    ง / PU 
    '''

    tt00 = '''
    女人保养：不仅要外养，还要内调，内外双管齐下，才能调养出好气色，主内调，副外养！。藏红花——斑的克星，妇科病的救星！。每天早晨泡3---6根，坚持服用三个月，会有你意想不到的效果！
    '''
    extract_CWSPOS(model, tt00)

    '''
    女人 / NN    保养 / VV    ： / PU    不仅 / X    要 / VV    外养 / VV    ， / PU    还 / AD    要 / VV    内调 / VV    
    ， / PU    内外 / NN    双管齐下 / VV    ， / PU    才 / AD    能 / VV    调养 / VV    出 / VV    好 / JJ    气色 / NN  
    ， / PU    主 / AD    内调 / VV    ， / PU    副 / AD    外养 / VV    ！ / PU    。 / PU    藏红花 / NR    ——斑 / NN    
    的 / DEG    克星 / NN    ， / PU    妇科病 / NN    的 / DEG    救星 / NN    ！ / PU    。 / PU    每 / X    天 / M    
    早晨 / NT    泡 / VV    3--- / CD    6 / CD    根 / M    ， / PU    坚持 / X    服用 / VV    三 / CD    个 / M    
    月 / NN    ， / PU    会 / VV    有 / VE    你 / PN    意想不到 / VV    的 / DEC    效果 / NN    ！ / PU 	
    '''
    # 2019-7-29 v2
    '''
    女人 / NN    保养 / VV    ： / PU    不仅 / X    要 / VV    外养 / VV    ， / PU    还 / AD    要 / VV    内调 / NN    
    ， / PU    内外 / NN    双 / CD    管齐 / VV    下 / VV    ， / PU    才 / AD    能 / VV    调养 / VV    出 / VV    
    好 / NN    气色 / NN    ， / PU    主 / NN    内调 / NN    ， / PU    副 / NN    外养 / NN    ！ / PU    。 / PU    
    藏红花 / VV    —— / PU    斑 / PU    的 / DEC    克星 / NN    ， / PU    妇科病 / NN    的 / DEG    救星 / NN    
    ！ / PU    。 / PU    每 / DT    天 / M    早晨 / NT    泡 / VV    3 / CD    - / PU    -- / PU    6 / CD    根 / M    
    ， / PU    坚持 / VV    服用 / VV    三 / CD    个 / M    月 / NN    ， / PU    会 / VV    有 / VE    你 / PN    
    意想 / VV    不 / AD    到 / VV    的 / DEC    效果 / NN    ！ / PU 	
    '''

    tt00 = '''
    ““        希望以后喜欢的人，。         不要让我哭，让我受委屈，。         不要不理我，更不要放弃我。。         要陪我长大，给我回应，。         更懂得要保护我，也要喜欢我。 ​​​ ​​​​  ​​​​<200b>
    '''
    extract_CWSPOS(model, tt00)

    '''
    	““         / PU    希 / PU    望 / PU    以后 / AD    喜 / VV    欢 / VV    的 / VV    人 / NN    ， / PU    
    	。 / PU             不 / PU    要 / VV    让 / VV    我 / PN    哭 / VV    ， / PU    让 / PU    我 / PN    
    	受 / AD    委 / PU    屈 / PU    ， / PU    。 / PU             不 / PU    要 / PU    不 / AD    理 / VV    
    	我 / PN    ， / PU    更 / PU    不 / AD    要 / VV    放 / VV    弃 / VV    我 / VV    。 / PU    。 / PU  
       要 / PU    陪 / PU    我 / PN    长 / VV    大 / VV    ， / PU    给 / PU    我 / PN    回 / PU    
       应 / PU    ， / PU    。 / PU             更 / PU    懂 / PU    得 / PU    要 / VV    保 / PU    护 / X    
       我 / PN    ， / PU    也 / PU    要 / VV    喜 / PU    欢 / X    我 / PN    。 / X    ​​​ ​​​​  ​​​​ / PU    < / PU    
       200 / PU    b / PU    > / X 	
   '''
    # 2019-7-29 v2
    '''
    ““ / PU    希望 / VV    以后 / X    喜欢 / VV    的 / DEC    人 / NN    ， / PU    。 / PU    不 / AD    要 / VV    
    让 / VV    我 / PN    哭 / VV    ， / PU    让 / VV    我 / PN    受 / VV    委屈 / NN    ， / PU    。 / PU    
    不 / AD    要 / VV    不 / AD    理 / VV    我 / PN    ， / PU    更 / AD    不 / AD    要 / VV    放弃 / VV    
    我 / PN    。 / PU    。 / PU    要 / VV    陪 / VV    我 / PN    长大 / VV    ， / PU    给 / P    我 / PN    
    回应 / VV    ， / PU    。 / PU    更 / AD    懂得 / VV    要 / VV    保护 / VV    我 / PN    ， / PU    也 / AD    
    要 / VV    喜欢 / VV    我 / PN    。 / PU    ​ / PU    ​​​​ / NR    ​ / PU 	
    '''
    tt00 = '''
    夏日冰淇淋调色盘🎨你值得拥有。能打的花西子-金陵栖霞 01。了解我的应该都知道我喜欢橘调吧 嘻嘻🤭。古风包装 这大地➕橘 一股东方美韵。没有欧美盘那么显色 但也不至于不上色（我在说啥？）。总之 我觉得值👍。#这不是化妆是魔##试色##眼妆##眼妆教程##眼妆分享#
    '''
    extract_CWSPOS(model, tt00)

    '''
        夏日 / NT    冰淇淋 / NN    调色盘 / NN    🎨 / PU    你 / PN    值得 / VV    拥有 / VV    。 / PU    能 / VV    
        打 / VV    的 / DEC    花西子 / NN    - / PU    金陵 / NR    栖霞 / NR     01 / PU    。 / PU    了解 / VV    
        我 / PN    的 / DEC    应该 / VV    都 / AD    知道 / VV    我 / PN    喜欢 / VV    橘调 / NN    吧 / SP     
        嘻嘻 / IJ    🤭 / PU    。 / PU    古 / JJ    风 / NN    包装 / NN     这 / DT    大 / NN    地 / NN    ➕ / PU    
        橘 / NN     一 / CD    股 / M    东 / NN    方 / NN    美韵 / NN    。 / PU    没有 / VE    欧 / NR    美盘 / NN    
        那么 / X    显色 / VV     但 / AD    也 / AD    不 / X    至于 / AD    不 / AD    上色 / VV    （ / PU    我 / PN    
        在 / AD    说 / VV    啥 / PN    ？ / PU    ） / PU    。 / PU    总之 / X     我 / PN    觉得 / VV    值 / VV    
        👍 / SP    。 / PU    # / PU    这 / PN    不 / AD    是 / VC    化妆 / NN    是 / VC    魔 / NN    # / PU    
        # / PU    试 / NN    色 / NN    # / NN    # / PU    眼妆 / NN    # / NN    # / PU    眼 / NN    妆教 / NN    
        程 / NN    # / NN    # / PU    眼妆 / NN    分享 / NN    # / PU 	
    '''

    # 2019-7-29 v2
    '''
    夏日 / NT    冰淇淋 / NN    调色盘 / NN    🎨 / PU    你 / PN    值得 / VV    拥有 / VV    。 / PU    能 / VV    
    打 / VV    的 / DEC    花西子 / NN    - / PU    金陵 / NR    栖霞01 / NR    。 / PU    了解 / VV    我 / PN    
    的 / DEG    应该 / NN    都 / AD    知道 / VV    我 / PN    喜欢 / VV    橘调 / NN    吧 / SP    嘻嘻🤭 / VV    
    。 / PU    古风 / NN    包装 / VV    这 / DT    大地 / NN    ➕ / PU    橘 / NN    一 / CD    股 / M    东方 / NN    
    美韵 / NN    。 / PU    没有 / VE    欧 / NR    美盘 / NN    那么 / X    显色 / VA    但 / AD    也 / AD    
    不至于 / X    不 / AD    上色 / VV    （ / PU    我 / PN    在 / AD    说 / VV    啥 / PU    ？ / PU    ） / PU   
     。 / PU    总之 / X    我 / PN    觉得 / VV    值👍 / VV    。 / PU    # / PU    这 / PN    不 / AD    是 / VC    
     化妆 / NN    是 / VC    魔 / NN    # / PU    # / PU    试色 / NN    # / PU    # / PU    眼妆 / NN    # / PU    
     # / PU    眼妆 / NN    教程 / NN    # / PU    # / PU    眼妆 / NN    分享 / VV    # / PU 	
    '''

    tt00 = '''
    酱酱，仙女们～。今天是睡睡推少女的第一天，元气满满！。不过，今天的主题是:招特价【8.8软妹币】呆梨！。
    有网购经历的女孩们应该大多数都听说过我们的品牌，质量有保障，作用效果好……哎呀，反正有点很多嘛。做我呆梨的好处:。①不用自己囤货，
    一件代发。②活动结束后会有我亲自叫你们写文案，引流什么的。③最基础的当然是自用巨无敌划算！比售价会便宜很多哟～。➕我q1349178766  
    做pong友叭！。爱你们！mua～#逆袭小仙女# #逆袭小仙女#
    '''
    extract_CWSPOS(model, tt00)

    '''
    	酱酱 / IJ    ， / PU    仙女们 / NN    ～ / PU    。 / PU    今天 / NT    是 / VC    睡睡 / VV    推 / VV    
    	少女 / NN    的 / DEC    第一 / OD    天 / M    ， / PU    元气 / NN    满满 / VA    ！ / PU    。 / PU    
    	不过 / X    ， / PU    今天 / NT    的 / DEG    主题 / NN    是 / VC    : / PU    招 / VV    特价 / NN    
    	【 / PU    8.8 / CD    软妹币 / NN    】 / PU    呆梨 / NN    ！ / PU    。 / PU    有 / VE    网购 / NN    
    	经历 / NN    的 / DEC    女孩们 / NN    应该 / VV    大多数 / CD    都 / AD    听说 / VV    过 / AS    
    	我们 / PN    的 / DEG    品牌 / NN    ， / PU    质量 / NN    有 / VE    保障 / NN    ， / PU    作用 / NN    
    	效果 / NN    好 / VA    …… / PU     / IJ    哎呀， / PU    反正 / X    有点 / X    很 / AD    多 / CD    
    	嘛 / SP    。 / PU    做 / VV    我 / PN    呆梨 / NN    的 / DEC    好处 / NN    : / PU    。 / PU    
    	① / PU    不用 / AD    自己 / PN    囤货 / VV    ， / PU    一 / CD    件 / M    代发 / VV    。 / PU    
    	② / PU    活动 / NN    结束 / VV    后 / LC    会 / VV    有 / VE    我 / PN    亲自 / X    叫 / VV    
    	你们 / PN    写 / VV    文案 / NN    ， / PU    引流 / VV    什么 / PN    的 / PN    。 / PU    ③ / PU    
    	最 / AD    基础 / JJ    的 / DEC    当然 / X    是 / VC    自用 / VV    巨无敌 / NN    划算 / VA    ！ / PU    
    	比 / P    售价 / NN    会 / VV    便宜 / VA    很 / AD    多 / AD    哟 / SP    ～ / PU    。 / PU    ➕ / PU    
    	 / PN    我q134917876 / PU    6 / NN    做 / NN    pong / VV    友 / PU    叭 / PU    ！ / PU    。 / PU    
    	 爱 / PU    你 / PN    们 / PN    ！ / PU    mu / PU    a / PU    ～ / PU    # / PU    逆袭 / VV    小 / JJ    
    	 仙女 / NN    # / PU     # / PU    逆袭 / VV    小 / NN    仙女 / NN    # / PU 	
    '''

    '''
    酱酱 / NN    ， / PU    仙女们 / NN    ～ / PU    。 / PU    今天 / NT    是 / VC    睡 / VV    睡 / VV    推 / VV    少女 / NN    的 / DEC    第一 / OD    天 / M    ， / PU    元气 / NN    满满 / VA    ！ / PU    。 / PU    不过 / X    ， / PU    今天 / NT    的 / DEG    主题 / NN    是 / VC    : / PU    招 / NN    特价 / NN    【 / PU    8.8 / CD    软妹币 / JJ    】 / PU    呆梨 / VV    ！ / PU    。 / PU    有 / VE    网购 / NN    经历 / NN    的 / DEC    女孩们 / NN    应该 / VV    大多数 / CD    都 / AD    听说 / VV    过 / AS    我们 / PN    的 / DEG    品牌 / NN    ， / PU    质量 / NN    有 / VE    保障 / NN    ， / PU    作用 / NN    效果 / NN    好 / VA    ……哎 / PU    呀 / PU    ， / PU    反正 / X    有点 / X    很多 / AD    嘛 / SP    。 / PU    做 / VV    我 / PN    呆梨 / VV    的 / DEC    好处 / NN    : / PU    。 / PU    ① / PU    不用 / AD    自己 / PN    囤货 / VV    ， / PU    一 / CD    件 / M    代发 / NN    。 / PU    ② / PU    活动 / NN    结 / NN    束后 / X    会 / VV    有 / VE    我 / PN    亲自 / X    叫 / VV    你们 / PN    写 / VV    文案 / NN    ， / PU    引流 / VV    什么 / PN    的 / DEG    。 / PU    ③ / PU    最 / AD    基础 / JJ    的 / DEC    当然 / X    是 / VC    自用 / VV    巨无敌 / JJ    划算 / VV    ！ / PU    比 / P    售价 / NN    会 / VV    便宜 / VA    很多 / CD    哟 / PU    ～ / PU    。 / PU    ➕ / PU    我 / PN    q1349178766 / CD    做 / VV    pong / NR    友叭 / NN    ！ / PU    。 / PU    爱 / VV    你们 / PN    ！ / PU    mua～ / PU    # / PU    逆袭 / NN    小 / JJ    仙女 / NN    # / PU    # / PU    逆袭 / NN    小 / JJ    仙女 / NN    # / PU 	
    '''

    tt00 = '''
        花生是个美姑娘
        此条下的愿望都会实现
        连云都那么可爱
        入住广州丽思卡尔顿
        雨中最美的崂山
        在田园风光的小店里约上2.3姐们
        图一二三都是P图后
    '''
    extract_CWSPOS(model, tt00)

    '''
    花生 / NN    是 / VC    个 / M    美 / JJ    姑娘 / NN    此 / DT    条 / M    下 / LC    的 / DEG    愿望 / NN    都 / AD    会 / VV    实现 / VV    连 / AD    云 / NR    都 / AD    那么 / X    可爱 / VA    入住 / VV    广州 / NR    丽思卡尔顿 / NR    雨 / NN    中 / LC    最 / AD    美 / VA    的 / DEC    崂山 / NR    在 / P    田园 / NN    风光 / NN    的 / DEC    小 / JJ    店 / NN    里约 / LC    上 / VV    2.3 / CD    姐们 / NN    图 / NN    一二三 / CD    都 / AD    是 / VC    p图 / NN    后 / LC 	
    '''

    tt00 = '''
        ６６位协院士（Ａｓｓｏｃｉａｔｅ Ｆｅｌｌｏｗ）２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ Ｆｅｌｌｏｗ）及２位通信协院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ Ａｓｓｏｃｉａｔｅ Ｆｅｌｌｏｗ）组成（不包括一九九四年当选者），
    '''
    extract_CWSPOS(model, tt00)


    # ６６ / CD    位 / M    协 / NN    院士 / NN    （ / PU    Ａｓｓｏｃｉａｔｅ / NR    Ｆｅｌｌｏ / NN    ｗ） / NN
    # ２４位 / NN    通 / CD    信院 / M    士 / NN    （Ｃｏｒｒｅｓｐｏｎｄｉｎｇ / NN
    # Ｆｅ / NN    ｌ / NN    ｌ / NN    ｏ / NN    ｗ） / NN    及 / NN    ２ / NN    位 / PU
    # 通 / CD    信 / PU    协院 / NN    士 / VV    （Ｃｏｒｒｅｓｐｏｎｄｉｎｇ / NN    Ａｓｓｏｃｉａｔｅ / NN
    # Ｆ / NN    ｅ / NN    ｌ / NN    ｌ / NN    ｏ / NN    ｗ） / NN    组 / PU    成 / PU
    # （不 / NN    包 / PU    括一 / VV    九九四年当 / NT    选 / X    者） / NN    ， / X

    tt00 = '''
        ✨今日份牛仔外套穿搭打卡|初春一定要有一件万能牛仔外套鸭💯。-我今天又双叒叕没化妆出门逛街了、懒癌晚期间歇性发作哈哈哈哈、。
        -落肩袖、不会显肩宽/后背有涂鸦和蕾丝拼接、见图六/。-Look1:搭配了衬衫和黑灰色牛仔裤/。-Look2：搭配了白色短T和牛仔裤/。
        牛仔裤我尝试了两种颜色、浅色系蓝色牛仔裤整体就偏复古风一点、配深色系就更日常活力一些、。#春天花会开##每日穿搭##日常穿搭#
    '''
    extract_CWSPOS(model, tt00)


    tt00 = '''
        #大鱼海棠# #大鱼海棠壁纸# 很感人的一部电影《大鱼海棠》，椿为了救鲲，不惜牺牲自己的一半寿命，湫喜欢椿，
        却把自己的一半寿命给了椿……人一但死后都会化成一条大鱼，椿听我的，数到3，2，1，我们一起跳下去，3.2.1跳，
        我会化成******陪着你。。椿，我喜欢你！！。“北冥有鱼，其名为鲲。”。“鲲之大。”。“一锅炖不下。。“化而为鸟。”。
        “其名为鹏。”。“鹏之大。”。“需要两个烧烤架。”。“一个秘制。”。“一个麻辣。”。“来瓶雪花！！！”。“带你勇闯天涯
    '''
    extract_CWSPOS(model, tt00)

    '''
    # / PU    大鱼 / NN    海棠 / NN    # / PU    # / PU    大鱼 / NN    海棠 / NN    壁纸 / NN    # / PU    
    很 / AD    感人 / VA    的 / DEC    一 / CD    部 / M    电影 / NN    《 / PU    大 / NN    鱼 / NN    
    海棠 / NN    》 / PU    ， / PU    椿 / NR    为了 / P    救 / VV    鲲 / NN    ， / PU    不惜 / X    
    牺牲 / VV    自己 / PN    的 / DEG    一半 / CD    寿命 / NN    ， / PU    湫 / NN    喜欢 / VV    
    椿 / NR    ， / PU    却 / AD    把 / BA    自己 / PN    的 / DEG    一半 / CD    寿命 / NN    
    给 / VV    了 / AS    椿 / NN    …… / PU    人 / NN    一但 / X    死 / VV    后 / LC    都 / AD    
    会 / VV    化成 / VV    一 / CD    条 / M    大 / JJ    鱼 / NN    ， / PU    椿 / NN    听 / VV    
    我 / PN    的 / DEG    ， / PU    数 / VV    到 / VV    3 / CD    ， / PU    2 / CD    ， / PU    
    1 / CD    ， / PU    我们 / PN    一起 / X    跳 / VV    下去 / VV    ， / PU    3.2.1 / CD    跳 / VV    
    ， / PU    我 / PN    会 / VV    化 / PU    成 / VV    ****** / PU    陪 / PU    着 / VV    你 / PN    
    。 / PU    。 / PU    椿 / PU    ， / PU    我 / PN    喜 / PU    欢 / VV    你 / PN    ！ / PU    ！ / PU    
    。 / PU    “ / PU    北冥 / NR    有 / VE    鱼 / NN    ， / PU    其 / PN    名 / PU    为 / VC    鲲 / NN   
     。 / PU    ” / PU    。 / PU    “ / PU    鲲 / NN    之 / DEG    大 / NN    。 / PU    ” / PU    。 / PU   
      “ / PU    一 / CD    锅 / NN    炖 / VV    不 / AD    下 / VV    。 / PU    。 / PU    “ / PU    化而为 / VV    
      鸟 / NN    。 / PU    ” / PU    。 / PU    “ / PU    其 / PN    名 / NN    为 / VC    鹏 / NN    。 / PU    
      ” / PU    。 / PU    “ / PU    鹏 / NN    之 / DEG    大 / NN    。 / PU    ” / PU    。 / PU    “ / PU    
      需要 / VV    两 / CD    个 / M    烧烤架 / NN    。 / PU    ” / PU    。 / PU    “ / PU    一 / CD    个 / M    
      秘制 / NN    。 / PU    ” / PU    。 / PU    “ / PU    一 / CD    个 / M    麻辣 / NN    。 / PU    ” / PU   
       。 / PU    “ / PU    来 / VV    瓶 / M    雪花 / NN    ！ / PU    ！ / PU    ！ / PU    ” / PU    。 / PU    
       “ / PU    带 / VV    你 / PN    勇闯 / VV    天涯 / NN 	
    '''

    tt0 = '''
          兰心餐厅\n
          咳咳￣ ￣)σ第一次穿汉服出门🎀💞开心ing
    '''
    extract_CWSPOS(model, tt00)

    '''
	兰心 / NR    餐厅 / NN    咳咳 / IJ    ￣ / IJ    ￣ / PU    ) / PU    σ / PU    第 / X    一次 / OD    穿 / VV    
	汉服 / NN    出门 / VV    🎀💞 / PU    开心 / VA    ing / X 	
    '''

    tt0 = '安一波情侣壁纸ヾ(･ω･｀＝´･ω･)ﾉ♪ 单身的我要起身去学校了#晒晒我的手账# #我好棒求表扬# #今日份滤镜推荐# #仗着好看为所欲为#'
    '''
        安一波 / NR    情侣 / NN    壁纸 / NN    ヾ / PU    (･ω･｀＝´･ω･) / PU    ﾉ / PU    ♪ / PU     单身 / NN    
        的 / DEC    我 / PN    要 / VV    起身 / VV    去 / VV    学校 / NN    了 / SP    # / PU    晒晒 / VV    
        我 / PN    的 / DEG    手账 / NN    # / PU     # / PU    我 / PN    好 / AD    棒 / VA    求 / VV    
        表扬 / NN    # / PU     # / PU    今日 / NT    份 / CD    滤镜 / NN    推荐 / NN    # / PU     # / PU    
        仗着 / P    好看 / VV    为所欲为 / VV    # / PU 	
    '''
    extract_CWSPOS(model, tt00)

    tt00 = '''
    #显瘦搭配##小个子显高穿搭##每日穿搭[话题]##晒腿battle##仙女裙##度假这样穿##仙女必备##春的气息#👧🏻。
    春装穿搭 做一个又酷又仙的少女啊👧🏻。今天小脸去踏春啦🌿这个裙子直接给我暴击！。小个子对于裙子长度是要求非常严格的，
    这件简直满足了所有要求好吗！！！长度刚好可以遮住大腿的肉肉，又温柔又仙？！简直就是仙女本仙了❗️❗️。
    第一眼是看中吊带上的珠珠小设计，还有裙子上的小流苏，让整件裙子简单又不失温柔。真的是太仙了吧……。颜色是那种纯白色，
    但不是那种死白的！！！！！！！度假❗️踏春❗️逛街❗️简直你就是温柔小姐姐啊🎀上面我搭配的是微毛绒设计感的透明带拖鞋  
    必须要综合一下 这样才能又酷又仙哈哈哈。鞋子也是百搭。@MT小美酱 @MT情报局
    '''
    extract_CWSPOS(model, tt00)

    tt00 = '''
        单枪匹马逛英国——伦敦篇。伦敦就是这个样子初次来到这个“老牌资本主义”的“雾都“，就像回
        到了上海，一幢幢不高的小楼，显得异常陈旧，很多楼房被数百年烟尘熏的就像被刷了一层黑色的油漆，
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
    extract_CWSPOS(model, tt00)

    tt00 = '''
        兰心餐厅\n作为一个无辣不欢的妹子，对上海菜的偏清淡偏甜真的是各种吃不惯。
        每次出门和闺蜜越饭局都是避开本帮菜。后来听很多朋友说上海有几家特别正宗味道做
        的很好的餐厅于是这周末和闺蜜们准备一起去尝一尝正宗的本帮菜。\n进贤路是我在上
        海比较喜欢的一条街啦，这家餐厅就开在这条路上。已经开了三十多年的老餐厅了，地
        方很小，就五六张桌子。但是翻桌率比较快。二楼之前的居民间也改成了餐厅，但是在
        上海的名气却非常大。烧的就是家常菜，普通到和家里烧的一样，生意非常好，外面排
        队的比里面吃的人还要多。
    '''
    extract_CWSPOS(model, tt00)

    '''
        兰心 餐厅 作为 一 个 无辣不欢 的 妹子 ， 对 上海 菜 的 偏 清淡 偏甜 真的 是 各 种 吃 不惯 。 
        每次 出门 和 闺蜜 越 饭局 都 是 避开 本帮 菜 。 后来 听 很多 朋友 说 上海 有 几 家 特别 
        正宗 味道 做 的 很 好 的 餐厅 于是 这 周末 和 闺蜜们 准备 一起 去 尝一尝 正宗 的 本帮菜 。 
        进贤路 是 我 在 上海 比较 喜欢 的 一 条 街 啦 ， 这 家 餐厅 就 开 在 这 条 路 上 。 已经 
        开 了 三十多 年 的 老 餐厅 了 ， 地方 很 小 ， 就 五六 张 桌子 。 但是 翻桌率 比较 快 。 
        二 楼 之前 的 居民间 也 改成 了 餐厅 ， 但是 在 上海 的 名气 却 非常 大 。 烧 的 就 是 家常菜 ， 
        普通 到 和 家里 烧 的 一样 ， 生意 非常 好 ， 外面 排队 的 比 里面 吃 的 人 还 要 多 。	
    '''

    tt00 = '''
        款款好看的美甲，简直能搞疯“选择综合症”诶！。这是一组超级温柔又带点设计感的美甲💅。
        春天来了🌺。美甲也从深色系转变为淡淡的浅色系了💐。今天给大家推荐最适合春天的美甲💅。
        希望你们会喜欢~😍@MT小美酱 @MT情报局 @美图秀秀 #春季美甲##显白美甲##清新美甲##ins美甲#
        '''
    extract_CWSPOS(model, tt00)

    tt00 = '''
        #我超甜的##口红安利##最热口红色号#今天给大家安利一款平价口红，卡拉泡泡唇膏笔，我比较喜欢的色号是Who   run   this。
        这个色号是一款非常正的土橘色，不论是你是白皮、黄皮、黄黑皮和黑皮都可以很安心的闭眼入。
        白皮涂简直就像仙女下凡了一样，黄皮和黄黑皮涂上特别提气色，而且还超级显白，超级安利这一款，这款唇膏笔我已经入了好几只了。
        这款土橘色已经快被我用完了，超级好看。就酱紫啦！拜拜！#口红安利#
    '''
    extract_CWSPOS(model, tt00)

    # # / PU    我 / PN    超 / AD    甜 / VA    的 / SP    # / PU    # / PU    口红 / NN    安利 / NR    # / PU
    # # / PU    最 / AD    热 / VA    口红 / NN    色号 / NN    # / PU    今天 / NT    给 / P    大家 / PN
    # 安利 / VV    一 / CD    款 / M    平价 / JJ    口红 / NN    ， / PU    卡拉泡泡 / NN    唇膏笔 / NN    ， / PU
    # 我 / PN    比较 / AD    喜欢 / VV    的 / DEC    色号 / NN    是 / VC    Who / PN    run / NN    this / NN
    # 。 / PU    这 / PU    个 / DT    色号 / NN    是 / VC    一 / CD    款 / M    非 / AD    常 / AD    正 / VA
    # 的 / DEG    土 / NN    橘 / PU    色 / NN    ， / PU    不论 / CS    是 / VC    你 / PN    是 / VC    白 / NN
    # 皮 / NN    、 / PU    黄 / PU    皮 / NN    、 / PU    黄 / PU    黑 / PU    皮 / NN    和 / CC    黑 / PU
    # 皮 / NN    都 / AD    可 / AD    以 / VV    很 / AD    安心 / VA    的 / DEV    闭眼 / VV    入 / VV
    # 。 / PU    白 / NN    皮 / NN    涂 / VV    简直 / AD    就 / AD    像 / P    仙女 / NN    下凡 / NN
    # 了 / AS    一样 / VA    ， / PU    黄 / NN    皮 / NN    和 / CC    黄 / JJ    黑 / JJ    皮 / NN
    # 涂上 / VV    特别 / AD    提 / VV    气色 / NN    ， / PU    而且 / AD    还 / AD    超级 / AD    显白 / VV
    # ， / PU    超级 / AD    安利 / VA    这 / DT    一 / CD    款 / M    ， / PU    这 / DT    款 / M
    # 唇膏笔 / NN    我 / PN    已经 / AD    入 / VV    了 / AS    好几 / CD    只 / M    了 / SP    。 / PU
    # 这 / DT    款 / M    土橘色 / NN    已经 / AD    快 / AD    被 / LB    我 / PN    用完 / VV    了 / AS
    # ， / PU    超级 / AD    好看 / VA    。 / PU    就 / AD    酱紫 / VV    啦 / SP    ！ / PU    拜拜 / VV
    # ！ / PU    # / PU    口红 / NN    安利 / NR    # / PU


def test_from_file(model, infile, outfile): # line 77
    with open(infile, 'r', encoding='utf8') as f:
        raw_data = f.readlines()

    #text_list = [s.strip() for s in raw_data]

    t0 = time.time()
    #outputT0 = model.cutlist_noUNK(text_list) #
    #output0 = ['    '.join(lst)+' ' for lst in outputT0]
    #o0 = '\t'
    #for x in output0: o0 += x + '\t'
    #print(o0+'\n')

    with open(outfile, 'w+') as fo:
        for x in tqdm(raw_data):
            print(x)

            outputT0 = model.cutlist_noUNK([x])
            output0 = ['    '.join(lst)+' ' for lst in outputT0]

            print(output0[0])
            fo.write(output0[0]+'\n')

    if 0:
        tx = [x for x in raw_data if x.strip()]

        outputT0 = model.cutlist_noUNK(tx)
        output0 = ['    '.join(lst)+' ' for lst in outputT0]
        with open(outfile+'2', 'wt') as fo:
            for sent in output0:
                print(sent)
                fo.write(sent+'\n')

    print('Processing time: ' + str(time.time()-t0))


LOCAL_FLAG = False
LOCAL_FLAG = True

TEST_FLAG = False
#TEST_FLAG = True


if __name__=='__main__':
    if LOCAL_FLAG:
        kwargs = set_local_eval_param()
    else:
        kwargs = set_server_eval_param()

    args._parse(kwargs)

    if TEST_FLAG:
        test_cases(model)
    else:
        if not LOCAL_FLAG:
            infile = '../data/xiuxiu/fenci_all.txt'

            types = {'cased': 'cws_F1_weights_epoch17.pt', 'uncased': 'cws_F1_weights_epoch16.pt'}

            nhl = 6
            for x in types.keys():
                args.bert_model = './tmp/ontonotes/CWSPOS2/'+x+'/l'+str(nhl)+'/'+types[x]
                args.num_hidden_layers = nhl

                outfile = './tmp/ontonotes/CWSPOS2/rs/fenci_all_'+x+'_'+str(nhl)+'_'+types[x]

                model = preload(args)
                test_from_file(model, infile, outfile)

        else:
            #test_from_file(model, './Test/except.txt', './Test/except_rs.txt')
            #test_from_file(model, './Test/fenci.txt', './Test/fenci_rs.txt')
            infile = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/test_data/fenci_multilingual.txt'
            outfile = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/test_data/fenci_multilingual_rs.txt'

            infile = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/test_data/fenci_all.txt'
            outfile = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/test_data/fenci_all_rs.txt'

            infile = './Test_cases/test_cases'
            outfile = './Test_cases/test_cases_rs'

            infile = './Test_cases/bad_cases.txt'
            outfile = './Test_cases/bad_cases_rs.txt'

            model = preload(args)
            test_from_file(model, infile, outfile)


