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
from src.preprocess import CWS_BMEO # dataset_to_dataloader, randomly_mask_input, OntoNotesDataset
import torch
import time

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
            'model_type': 'sequencelabeling',
            'data_dir': '../data/ontonotes5/4ner_data/',
            'vocab_file': '../models/bert-base-chinese/vocab.txt',
            'bert_config_file': '../models/bert-base-chinese/bert_config.json',
            'output_dir': './tmp_2019_3_22/out/',
            'do_lower_case': True,
            'train_batch_size': 128,
            'max_seq_length': 128,
            'num_hidden_layers': 3,
            'init_checkpoint': '../models/bert-base-chinese/',
            'bert_model': './tmp_2019_3_23/ontonotes/nhl3_nte15_nbs64/weights_epoch03.pt',
            'no_cuda': True,
            'override_output': True,
            'tensorboardWriter': False
            }

LOCAL_FLAG = False
LOCAL_FLAG = True

if __name__=='__main__':
    if LOCAL_FLAG:
        kwargs = set_local_eval_param()
    else:
        kwargs = set_server_eval_param()

    args._parse(kwargs)
    model = preload(args)

    tt0 = '''
    #显瘦搭配##小个子显高穿搭##每日穿搭[话题]##晒腿battle##仙女裙##度假这样穿##仙女必备##春的气息#👧🏻。
    春装穿搭 做一个又酷又仙的少女啊👧🏻。今天小脸去踏春啦🌿这个裙子直接给我暴击！。小个子对于裙子长度是要求非常严格的，
    这件简直满足了所有要求好吗！！！长度刚好可以遮住大腿的肉肉，又温柔又仙？！简直就是仙女本仙了❗️❗️。
    第一眼是看中吊带上的珠珠小设计，还有裙子上的小流苏，让整件裙子简单又不失温柔。真的是太仙了吧……。颜色是那种纯白色，
    但不是那种死白的！！！！！！！度假❗️踏春❗️逛街❗️简直你就是温柔小姐姐啊🎀上面我搭配的是微毛绒设计感的透明带拖鞋  
    必须要综合一下 这样才能又酷又仙哈哈哈。鞋子也是百搭。@MT小美酱 @MT情报局
    '''

    text0 = '''
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
    t0 = time.time()
    outputT0 = model.cutlist_noUNK([tt0, text0])
    output0 = [' '.join(lst) for lst in outputT0]
    o0 = ''
    for x in output0: o0 += x + '\t'
    print(o0+'\n')
    print('Processing time: ' + str(time.time()-t0))

    '''
        # 显瘦 搭配 # # 小 个子 显高 穿搭 # # 每日 穿搭 [ 话题 ] # # 晒腿 battle # # 仙女裙 # # 度假 这样 
        穿 # # 仙女 必备 # # 春 的 气息 # 👧🏻 。 春装 穿搭 做 一 个 又 酷 又 仙 的 少女 啊 👧🏻 。 今天 小脸 
        去 踏春 啦 🌿 这个 裙子 直接 给 我 暴击 ！ 。 小 个子 对于 裙子 长度 是 要求 非常 严格 的 ， 这 件 简直 
        满足 了 所有 要求 好 吗 ！ ！！ 长度 刚好 可以 遮住 大腿 的 肉肉 ， 又 温柔 又 仙 ？ ！ 简直 就 是 仙女 
        本仙 了 ❗️❗️ 。 第一 眼 是 看 中 吊带 上 的 珠珠 小 设计 ， 还 有 裙子 上 的 小 流苏 ， 让 整 件 裙子 
        简单 又 不 失 温柔 。 真的 是 太 仙 了 吧 …… 。 颜色 是 那 种 纯 白色 ， 但 不 是 那 种 死白 的 ！ ！！！！！！ 
        度假 ❗️ 踏春 ❗️ 逛街 ❗️ 简直 你 就 是 温柔 小 姐姐 啊 🎀 上面 我 搭配 的 是 微 毛绒 设计感 的 透明带 拖鞋 必须 
        要 综合 一下 这样 才 能 又 酷 又 仙哈哈哈 。 鞋子 也 是 百搭 。 @ MT 小美酱 @ MT 情报局	
        
        单枪 匹马 逛 英国 —— 伦敦 篇 。 伦敦 就 是 这个 样子 初次 来到 这个 “ 老牌 资本主义 ” 的 “ 雾都 “ ， 
        就 像 回到 了 上海 ， 一 幢 幢 不 高 的 小 楼 ， 显得 异常 陈旧 ， 很多 楼房 被 数百 年 烟尘熏 的 就 
        像 被 刷 了 一 层 黑色 的 油漆 ， 油光 锃亮 ， 如果 不 是 旁边 的 楼房 正在 清洗 ， 很 难 让 人 相信 
        如今 的 伦敦 是 饱经 污染 沧桑 后 及时 刹车 的 高手 ， 因为 一 座 现代化 的 国际 大都市 也 是 有 不少 
        楼房 是 黑色 的 呢 ， 黑色 显得 凝重 、 高雅 ， 但是 绝对 不 能 靠 油烟 去 熏 …… 堵车 ， 是 所有 大都市 
        的 通病 ， 虽然 不足为怪 ， 但是 ， 1988年 的 北京 还 没有 那么 多 的 车 ， 也 没有 全 城 大 堵车 的 现象 ， 
        有的 是 刚刚 开始 的 “ 靠 油烟 和 汽车 的 尾气 烟熏火燎 美丽 的 古城 ” ， 有 谁 能够 想到 ， 短短 的 十 年 ， 
        北京 就 气喘吁吁 的 追赶 上 了 伦敦 ， 没有 一 条 洁净 的 河流 ， 没有 清新 的 空气 ， 有的 是 让 人 窒息 
        的 空气 污 染 ……. 以及 ， 让 人 始料 未及 的 全 城 大 堵车 。 如果 ， 我们 那些 负责 城市 建设 规划 的 先生们 ， 
        在 国外 ， 不 只 只 是 游山玩水 的话 ， 带回 别人 的 教训 、 总结 别人 的 经验 的话 ， 我们 这个 被 穷祖先 毁 的 
        “ 一塌糊涂 ” 的 脆弱 的 生态 环境 也 不 会 再 经受 20世纪 90年代 的 现代化 的 大 污染 了 。 但是 ， 伦敦 是 一 
        座 改 过 自 新 的 城市 ， 人家 痛定思痛 ， 紧急 刹车 ， 及时 的 治理 了 污染 ， 我们 在 泰吾士河 里 可以 看到 鱼儿 
        在 自由 的 翻滚 ， 天空 湛蓝 ， 翠绿 的 草地 与 兰天 辉映 着 ， 一 片 “ 污染 大战 ” 后 的 和平 景象	
    '''

    text1 = '''
        兰心餐厅\n作为一个无辣不欢的妹子，对上海菜的偏清淡偏甜真的是各种吃不惯。
        每次出门和闺蜜越饭局都是避开本帮菜。后来听很多朋友说上海有几家特别正宗味道做
        的很好的餐厅于是这周末和闺蜜们准备一起去尝一尝正宗的本帮菜。\n进贤路是我在上
        海比较喜欢的一条街啦，这家餐厅就开在这条路上。已经开了三十多年的老餐厅了，地
        方很小，就五六张桌子。但是翻桌率比较快。二楼之前的居民间也改成了餐厅，但是在
        上海的名气却非常大。烧的就是家常菜，普通到和家里烧的一样，生意非常好，外面排
        队的比里面吃的人还要多。
    '''
    t0 = time.time()
    outputT1 = model.cutlist_noUNK([text1])
    output1 = [' '.join(lst) for lst in outputT1]
    o1 = ''
    for x in output1: o1 += x + '\t'
    print(text1)
    print(o1+'\n')
    print('Processing time: ' + str(time.time()-t0))
    '''
        兰心 餐厅 作为 一 个 无辣不欢 的 妹子 ， 对 上海 菜 的 偏 清淡 偏甜 真的 是 各 种 吃 不惯 。 
        每次 出门 和 闺蜜 越 饭局 都 是 避开 本帮 菜 。 后来 听 很多 朋友 说 上海 有 几 家 特别 
        正宗 味道 做 的 很 好 的 餐厅 于是 这 周末 和 闺蜜们 准备 一起 去 尝一尝 正宗 的 本帮菜 。 
        进贤路 是 我 在 上海 比较 喜欢 的 一 条 街 啦 ， 这 家 餐厅 就 开 在 这 条 路 上 。 已经 
        开 了 三十多 年 的 老 餐厅 了 ， 地方 很 小 ， 就 五六 张 桌子 。 但是 翻桌率 比较 快 。 
        二 楼 之前 的 居民间 也 改成 了 餐厅 ， 但是 在 上海 的 名气 却 非常 大 。 烧 的 就 是 家常菜 ， 
        普通 到 和 家里 烧 的 一样 ， 生意 非常 好 ， 外面 排队 的 比 里面 吃 的 人 还 要 多 。	
    '''

    text2 = '''
        款款好看的美甲，简直能搞疯“选择综合症”诶！。这是一组超级温柔又带点设计感的美甲💅。
        春天来了🌺。美甲也从深色系转变为淡淡的浅色系了💐。今天给大家推荐最适合春天的美甲💅。
        希望你们会喜欢~😍@MT小美酱 @MT情报局 @美图秀秀 #春季美甲##显白美甲##清新美甲##ins美甲#
        '''
    t0 = time.time()
    outputT2 = model.cutlist_noUNK([text2])
    output2 = [' '.join(lst) for lst in outputT2]
    o2 = ''
    for x in output2: o2 += x + '\t'
    print(text2)
    print(o2+'\n')
    print('Processing time: ' + str(time.time()-t0))
    '''
        款款 好看 的 美甲 ， 简直 能 搞疯 “ 选择 综合症 ” 诶 ！ 。 这 是 一 组 超级 温柔 又 带 点 
        设计感 的 美甲 💅 。 春天 来 了 🌺 。 美甲 也 从 深 色系 转变 为 淡淡 的 浅 色系 了 💐 。 
        今天 给 大家 推荐 最 适合 春天 的 美甲 💅 。 希望 你们 会 喜欢 ~ 😍 @ MT 小美酱 @ MT 情报局 
        @ 美图 秀秀 # 春季 美甲 # # 显白 美甲 # # 清新 美甲 # # ins 美甲 #	
    '''

