#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 11:21 2019-01-30 
@author: haiqinyang

Feature: 

Scenario: 
"""
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('../src')
from src.utilis import save_model
from src.config import args, segType
from src.utilis import get_dataset_and_dataloader, get_eval_dataloaders
from src.preprocess import CWS_BMEO
from tqdm import tqdm
import time
import torch


def test_BertCRF_constructor():
    from src.BERT.modeling import BertCRF
    from collections import namedtuple

    test_input_args = namedtuple("test_input_args", "bert_model cache_dir")
    test_input_args.bert_model = '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese.tar.gz'
    test_input_args.cache_dir = '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/'

    model = BertCRF(test_input_args, 4)


def test_BasicTokenizer():
    from src.tokenization import BasicTokenizer
    # prove processing English and Chinese characters
    basic_tokenizer = BasicTokenizer(do_lower_case=True)
    text = 'beauty一百分\n Beauty 一百分!!'
    print(basic_tokenizer.tokenize(text))


def test_FullTokenizer():
    from src.tokenization import FullTokenizer, BasicTokenizer

    vocab_file = '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/models.txt'
    full_tokenizer = FullTokenizer(vocab_file, do_lower_case=True)

    text = '任天堂游戏商店的加入被业界视为android的革命。'
    print(full_tokenizer.tokenize(text))

    text = '台湾!!. 比赛。今天，开始吗？  ？？！咳咳￣ ￣)σ第一次穿汉服出门🎀💞开心Laughing'
    print(full_tokenizer.tokenize(text))

    text = '台湾的公视今天主办的台北市长 Candidate  Defence  ，'
    print(full_tokenizer.tokenize(text))
    #['台', '湾', '的', '公', '视', '今', '天', '主', '办', '的', '台', '北', '市', '长', 'can', '##di', '##da', '##te', 'de', '##fe', '##nce', '，']

    text = 'Candidate'
    print(full_tokenizer.tokenize(text))
    # ['can', '##di', '##da', '##te']

    text = '  Defence  ，'
    print(full_tokenizer.tokenize(text))

    text1 = '''植物研究所所長周昌弘先生當選第三世界科學院（Ｔｈｅ　Ｔｈｉｒｄ　Ｗｏｒｌｄ　Ａｃａｄｅｍｙ　ｏｆ　Ｓｃｉｅｎｃｅｓ，簡稱ＴＷＡＳ）
    院士。ＴＷＡＳ係一九八三年由Ｐｒｏｆ　Ａｄｂｕｓ　Ｓａｌａｍ（巴基斯坦籍，曾獲諾貝爾獎）發起成立，會員遍佈６３個國家，目前由２３２位院士
    （Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ）
    　及２位通信協院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者），李政道、楊振寧、丁肇中、
    李遠哲、陳省身、吳健雄、朱經武、蔡南海等院士均為該院Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ。本院數理組院士、哈佛大學數學系教授丘成桐，經瑞典皇家科學院評定為
    一九九四年克列佛（Ｃｒａｆｏｏ　Ｐｒｉｚｅ）獎得主，藉以表彰其在微分幾何領域影響深遠之貢獻。'''
    print(full_tokenizer.tokenize(text1))

    text = 'Ｐｒｏｆ　Ａｄｂｕｓ　Ｓａｌａｍ'
    print(full_tokenizer.tokenize(text))

    text = ' Ｐｒｏｆ　Ａｄｂｕｓ　Ｓａｌａｍ '
    print(full_tokenizer.tokenize(text))

    text = '( Ｐｒｏｆ　 Ａｄｂｕｓ　 Ｓａｌａｍ  ) '
    print(full_tokenizer.tokenize(text))

    text = '６３個國家'
    print(full_tokenizer.tokenize(text))

    text = '６３ 個國家'
    print(full_tokenizer.tokenize(text))

    text = '２４２４位通信院士'
    print(full_tokenizer.tokenize(text))

    text = '２４２４ 位通信院士'
    print(full_tokenizer.tokenize(text))

    text = '2424 位通信院士'
    print(full_tokenizer.tokenize(text))

    text = '2424位通信院士'
    print(full_tokenizer.tokenize(text))

    text = '下一波DVD及用于可携式小型资讯用品的微型光碟（Minidisk），也已迫不及待地等着敲开消费者的荷包。'
    print(full_tokenizer.tokenize(text))

    test_text = '第三世界科學院（Ｔｈｅ　Ｔｈｉｒｄ　Ｗｏｒｌｄ　Ａｃａｄｅｍｙ　ｏｆ　Ｓｃｉｅｎｃｅｓ，簡稱ＴＷＡＳ）'
    basic_tokenizer = BasicTokenizer(do_lower_case=True)
    sep_tokens = basic_tokenizer.tokenize(test_text)
    print('Basic:')
    for tt in sep_tokens:
        if basic_tokenizer._is_chinese_char(ord(tt[0])):
            wt = 'C'
        elif basic_tokenizer._is_english_char(ord(tt[0])):
            wt = 'E'
        else:
            wt = 'O'

        print(tt+' '+wt)

def check_english(w):
    import re

    english_check = re.compile(r'[a-z]')

    if english_check.match(w):
        print("english", w)
    else:
        print("other:", w)

def test_pandas_drop():
    import pandas as pd
    import os

    data_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data'
    df = pd.read_csv(os.path.join(data_dir, "test_code.tsv"), sep='\t')

    # full_pos (chunk), ner, seg, text
    df.drop(['full_pos', 'ner'], axis=1)

    # change name to tag for consistently processing
    df.rename(columns={'seg': 'label'}, inplace=True)

    print(len(df.full_pos))

def test_pandas_drop_syn():
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(np.arange(12).reshape(3,4), columns=['A', 'B', 'C', 'D'])
    print(df)

    # need parameter inplace=True
    df.drop(['B', 'C'], axis=1, inplace=True)
    print(df)

def test_metrics():
    from src.metrics import get_ner_BMES, get_ner_BIO, get_ner_fmeasure, reverse_style
    label_list_BMES = "O O O O O O O O O O O O B-PER E-PER O O O O O O O O O O O O O O O O O O O O O O O"
    label_list_BIO = "O B-PER I-PER O O O B-PER O O B-ORG I-ORG I-ORG"

    label_list_BMES = label_list_BMES.split()
    #get_ner_BMES(label_list_BMES)

    #label_list_BIO = label_list_BIO.split()
    ll_BIO1 = ['O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG']
    ll_BIO2 = ['O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O']

    print(get_ner_BIO(ll_BIO1))
    print(get_ner_BIO(ll_BIO2))

def test_CWS_Dict():
    from src.utilis import CWS_Dict
    cws_dict = CWS_Dict()

    sent = '迈向  充满  希望  的  新  世纪  ——  一九九八年  新年  讲话  （  附  图片  １  张  ）\n  ' \
           '中共中央  总书记  、  国家  主席  江  泽民\n' \
           '（  一九九七年  十二月  三十一日  ） \n ' \
           '１２月  ３１日  ，  中共中央  总书记  、  国家  主席  江  泽民  发表  １９９８年  新年  讲话  ' \
           '《  迈向  充满  希望  的  新  世纪  》  。  （  新华社  记者  兰  红光  摄  ）\n' \
           '同胞  们  、  朋友  们  、  女士  们  、  先生  们  ：\n ' \
           '在  １９９８年  来临  之际  ，  我  十分  高兴  地  通过  中央  人民  广播  电台  、' \
           '  中国  国际  广播  电台  和  中央  电视台  ，  向  全国  各族  人民  ，  向  香港  特别  行政区  ' \
           '同胞  、  澳门  和  台湾  同胞  、  海外  侨胞  ，  向  世界  各国  的  朋友  们  ，  致以  诚挚  的  ' \
           '问候  和  良好  的  祝愿  ！  '

    q_num = cws_dict._findNum(sent)
    print(list(q_num.queue))

    sent = '人悬挂在半空中孤立无援，而他的脚下就是万丈深渊。\n' \
           '但由于担心卷起的气浪会把马修连人带伞吹落悬崖，直升机无法直接实施空中救援。\n' \
           '救援人员只能借助直升机登上悬崖顶端，将绳索扔给马修进行营救。\n' \
           '在经历了一番周折之后，马修终于被救援人员拉上了悬崖，' \
           '幸运的是由于营救及时，马修本人并无大碍。中央台编译报道。\n' \
           '好，这次中国新闻节目就到这，我是徐俐，谢谢大家。\n' \
           '接下来是由王世林主持的今日关注节目。\n' \
           '各位观众，再见。\n' \
           ' ＥＭＰＴＹ'
    q_eng = cws_dict._findEng(sent)
    print(list(q_eng.queue))

    # some problems are here
    sent = 'ＥＭＰＴＹ'
    q_eng = cws_dict._findEng(sent)
    print(list(q_eng.queue))


def test_pkuseg():
    from src.metrics import getChunks, getFscore
    tag_list = ['BBIBBIIBIIIB', 'BBBBIBBBIIIB']
    #tag_list = ['S,BI,S,BII,BIII,S,S,S,S,BI,S']

    tmp_list = [','.join(tag_list[i]) for i in range(len(tag_list))]
    print(getChunks(tmp_list)) # ['B*1*0,B*2*1,B*1*3,B*3*4,B*4*7,B*1*11,', 'B*1*0,B*1*1,B*1*2,B*2*3,B*1*5,B*1*6,B*4*7,B*1*11,']


    tag_to_idx = {'B': 0, 'I': 1, 'O': 2}
    idx_to_chunk_tag = {}

    '''
    for tag, idx in tag_to_idx.items():
        if tag.startswith("I"):
            tag = "I"
        if tag.startswith("O"):
            tag = "O"
        idx_to_chunk_tag[idx] = tag
    '''

    idx_to_chunk_tag = {}
    tag_to_idx = {'B': 0, 'M': 1, 'E': 2, 'S': 3, '[START]': 4, '[END]': 5}
    BIO_tag_to_idx = {'B': 0, 'I': 1, 'O': 2, '[START]': 3, '[END]': 4}
    token_list = [''.join(str(BIO_tag_to_idx[item])+',') for i in range(len(tag_list)) for item in tag_list[i] ]

    idx_to_chunk_tag = idx_to_tag(BIO_tag_to_idx)

    token_list = []
    for i in range(len(tag_list)):
        t = ''
        for item in tag_list[i]:
            t += ''.join(str(BIO_tag_to_idx[item])+',')
        token_list.append(t)


    goldTagList = [token_list[0]]
    resTagList = [token_list[1]]

    scoreList, infoList = getFscore(goldTagList, resTagList, idx_to_chunk_tag)
    # ValueError: invalid literal for int() with base 10: 'B'
    print(scoreList)
    print(infoList)


def calSize(H, vs, mpe, L):
    for l in L:
        # embedding: (vs+mpe)*H; # Query, Key, value: 3*H*H; Intermediate: 4*H *H; Pooler: H*H
        sz = (vs+mpe)*H + ((3+4)*H*H)*l + H*H
        print('# layer: '+str(l)+', #para: '+str(sz))


def verifyModelSize():
    H = 768
    vs = 21128
    mpe = 512
    L = [3, 6, 12]

    num_model_para = calSize(H, vs, mpe, L)

'''
def test_parse_one2BERTformat():
    from OntoNotes.f6_generate_training_data import parse_one2BERT2Dict
    s = '(NP (CP (IP (NP (DNP (NER-GPE (NR Taiwan)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN candidate) (NN defence)) (PU ，))))'
    out_dict = parse_one2BERT2Dict(s)
    print('src_seg:'+out_dict['src_seg'])
    print('src_ner:'+out_dict['src_ner'])
    print('full_pos:'+out_dict['full_pos'])
    print('text:'+out_dict['text'])
    print('text_seg:'+out_dict['text_seg'])
    print('bert_seg:'+out_dict['bert_seg'])
    print('bert_ner:'+out_dict['bert_ner'])
'''

def set_local_eval_param():
    return {'task_name': 'ontonotes_CWS',
            'model_type': 'sequencelabeling',
            'data_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4ner_data/',
            #'bert_model_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data/eval/2019_3_12/models/',
            'vocab_file': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/models.txt',
            'bert_config_file': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/bert_config.json',
            'output_dir': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/2019_3_12/rs/nhl3/',
            'do_lower_case': True,
            'train_batch_size': 128,
            'max_seq_length': 64,
            'num_hidden_layers': 3,
            'init_checkpoint': '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/',
            'bert_model': '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/2019_3_12/models/nhl3/weights_epoch03.pt',
            'override_output': True,
            'tensorboardWriter': False
            }


def test_load_model():
    kwargs = set_local_eval_param()
    args._parse(kwargs)

    label_list = ['B', 'M', 'E', 'S', '[START]', '[END]']
    model, device = load_model(label_list, args)

    save_model(model, args.output_dir + 'tmp.tsv')


def test_dataloader():
    kwargs = set_local_eval_param()
    args._parse(kwargs)

    processors = {
        "ontonotes_cws": lambda: CWS_BMEO(nopunc=args.nopunc),
    }

    task_name = args.task_name.lower()

    # Prepare models
    processor = processors[task_name]()
    train_dataset, train_dataloader = get_dataset_and_dataloader(processor, args, training=False, type = 'tmp_test')

    eval_dataloaders = get_ontonotes_eval_dataloaders(processor, args)

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        input_ids, segment_ids, input_mask = batch[:3]
        label_ids = batch[3:] if len(batch[3:])>1 else batch[3]


def decode_iter(logits, attention_mask):
    mask = attention_mask.byte()
    batch_size, seq_length = mask.shape

    best_tags_list = []
    for idx in range(batch_size):
        # Find the tag which maximizes the score at the last timestep; this is our best tag
        # for the last timestep
        best_tags = []
        for iseq in range(seq_length):
            if mask[idx, iseq]:
                _, best_selected_tag = logits[idx, iseq].max(dim=0)
                best_tags.append(best_selected_tag.item())

        best_tags_list.append(best_tags)
    return best_tags_list


def decode_batch(logits, attention_mask):
    mask = attention_mask.byte()
    batch_size, seq_length = mask.shape

    _, best_selected_tag = logits.max(dim=2)

    best_tags_list = []
    for n in range(batch_size):
        selected_tag = torch.masked_select(best_selected_tag[n, :], mask[n, :])
        best_tags_list.append(selected_tag.tolist())

    return best_tags_list


def test_decode():
    n_sample = 32
    n_len = 8
    n_tag = 6

    logits = torch.rand((n_sample, n_len, n_tag))

    mask = [1]*(n_len//2) + [0]*(n_len//2)
    attention_mask = torch.ByteTensor([mask]*n_sample)

    tm = time.time()
    lit = decode_iter(logits, attention_mask)
    print('time: ' + str(time.time()-tm))
    print(lit)

    tm = time.time()
    lbt = decode_batch(logits, attention_mask)
    print('time: ' + str(time.time()-tm))
    print(lbt)


def test_split():
    import re

    text = '但是，规模大不等于规模经济。这可以从三个方面考察：（１）生产能力的限度。投入增加超过一定点，产出的增量或边际产出将会减少，出现规模报酬递减现象。（２）交易成本的限度，主要是企业内部交易成本———通常称为管理成本———限制。企业之所以替代市场存在，是因为通过市场交易是需要成本的，如搜寻合适产品、谈判、签约、监督执行等，都需要花费成本，在一些情况下，企业将一些经济活动内部化，通过行政权威加以组织，能够节约市场上的交易成本。企业内部协调一般通过层级制结构进行，也需要一定的费用，这种费用乃是企业内部发生的交易成本。如果管理幅度过大，或者层次太多，从基层到中心决策者的信息传递速度就会变慢，甚至信号失真，致使企业效率降低，出现规模不经济。组织管理形式的变动，如实行事业部制等，能够改变信息传递的速度和信息质量，改善决策水平，从而拉长规模经济存在的时间跨度。但这不是没有限度的。（３）对技术进步的限制，这在出现垄断情形时尤其如此。随着企业规模扩大，在市场中的垄断力量的增强，市场将偏离充分竞争时的均衡，垄断者将通过垄断定价和进入壁垒限制竞争者，赚取垄断利润。此时企业追求创造、追求技术进步的压力和动力将会减弱。这在一个行业只有一个企业的完全垄断（独占）情形中最为明显。也正因为这一点，主要市场经济国家的反垄断法都极力限制垄断程度，不允许一个行业只有一个企业（厂商）。特别是，在新的科技革命面前，小企业也因其能够灵活地面对市场、富有创造力而显示出生命力，大企业反而可能对市场变化反应迟缓而处于竞争劣势。总之，规模经济包含的是一个适度规模、有效规模，既不是越大越经济，也不是越小越经济。'
    at = re.split('(。|，|：|\n|#)', text)
    print(at)


def test_write():
    with open('tmp.txt', 'w+') as f:
        f.write('abc\n')
        f.write('dd')
        f.write(' ee')
        f.write('\n abc')

    with open('tmp.txt', 'r') as f:
        line = f.read()
        while line != '':
            print(line)
            line = f.read()


def test_construct_pos_tags():
    from src.preprocess import construct_pos_tags

    pos_tags_file = '../resource/pos_tags.txt'
    pos_label_list, pos_label_map, pos_idx_to_label_map = \
        construct_pos_tags(pos_tags_file, mode = 'BIO')

    print(pos_label_list)
    print(pos_label_map)
    print(pos_idx_to_label_map)


def test_outputPOSFscoreUsedBIO():
    goldTagList = [[0, 62, 63, 62, 63, 104, 105, 85, 85, 104, 105, 65, 66, 85, 104, 105, 105, 105, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 85, 65, 66, 66, 85, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 79, 62, 63, 62, 63, 79, 62, 63, 62, 63, 104, 105, 104, 105, 53, 54, 85, 104, 105,
                    79, 68, 69, 62, 63, 22, 65, 66, 65, 66, 62, 63, 62, 63, 62, 63, 85, 4, 79, 68, 69, 55,
                    4, 104, 105, 47, 48, 62, 63, 85, 2, 3, 2, 3, 65, 66, 62, 63, 62, 63, 62, 63, 4, 106,
                    4, 106, 85, 104, 105, 65, 66, 62, 63, 25, 62, 63, 85, 4, 4, 106, 62, 63, 10, 62, 63,
                    106, 79, 47, 48, 25, 62, 63, 62, 63, 55, 85, 47, 48, 48, 25, 62, 63, 62, 63, 13, 62,
                    63, 62, 63, 4, 104, 105, 62, 63, 85, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 65, 66, 65, 66, 62, 63, 63, 62, 63, 62, 63, 71, 72, 62, 63, 62, 63, 62, 63, 85, 79,
                    68, 69, 68, 69, 69, 104, 105, 105, 105, 85, 79, 62, 63, 104, 105, 14, 15, 15, 56, 57,
                    104, 105, 55, 85, 104, 105, 7, 14, 15, 15, 58, 62, 63, 85, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0]]

    input_mask = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    preTagList = [[0, 62, 63, 62, 63, 63, 63, 85, 85, 62, 63, 65, 66, 85, 105, 4, 105, 104, 1],
                  [0, 85, 65, 66, 63, 85, 1],
                  [0, 79, 66, 66, 34, 63, 79, 63, 16, 62, 63, 104, 105, 4, 63, 62, 63, 85, 104, 105, 79,
                   16, 69, 65, 69, 25, 62, 66, 66, 63, 62, 63, 62, 63, 62, 63, 85, 104, 105, 68, 69, 63,
                   4, 104, 105, 104, 105, 62, 105, 85, 104, 55, 104, 105, 65, 66, 62, 63, 66, 66, 62, 63,
                   104, 105, 104, 105, 85, 104, 105, 65, 66, 62, 63, 22, 62, 63, 85, 4, 4, 63, 62, 81, 4,
                   62, 63, 105, 79, 2, 16, 22, 104, 104, 62, 63, 55, 85, 62, 63, 63, 25, 62, 63, 104, 105,
                   13, 62, 105, 104, 105, 4, 104, 105, 104, 105, 85, 1],
                  [0, 65, 66, 66, 63, 62, 63, 63, 62, 63, 62, 63, 71, 16, 62, 63, 104, 105, 62, 63, 85, 79,
                   68, 69, 69, 15, 69, 105, 16, 63, 105, 85, 79, 66, 63, 104, 105, 15, 15, 15, 62, 58, 62,
                   63, 105, 85, 104, 105, 7, 14, 15, 15, 63, 63, 63, 85, 1]]

    from src.metrics import outputPOSFscoreUsedBIO
    scoreList, infoList = outputPOSFscoreUsedBIO(goldTagList, preTagList, input_mask)
    print(scoreList)
    print(infoList)


def compare_string_directly(text, tag):
    result_str = ''
    for idx in range(len(tag)):
        tt = text[idx]
        tt = tt.replace('##', '')
        ti = tag[idx]

        if int(ti) == 2:  # 'B'
            result_str += ' ' + tt
        elif int(ti) > 3:  # and (cur_word_is_english)
            # int(ti)>1: tokens of 'E' and 'S'
            # current word is english
            result_str += tt + ' '
        else:
            result_str += tt

    return result_str


def compare_string_reference(text, tag):
    result_str = ''
    for idx in range(len(tag)):
        tt = text[idx]
        tt = tt.replace('##', '')
        ti = tag[idx]

        if int(ti) == segType.BMES_label_map['B']:  # 'B'
            result_str += ' ' + tt
        elif int(ti) > segType.BMES_label_map['M']:  # and (cur_word_is_english)
            # int(ti)>1: tokens of 'E' and 'S'
            # current word is english
            result_str += tt + ' '
        else:
            result_str += tt

    return result_str

def check_time():
    import time

    count = 100000
    text_ele = ['电影首发。']*count
    text = ''.join(text_ele)

    tag_ele = ['24245']*count
    tag = ''.join(tag_ele)

    st = time.time()
    print(compare_string_directly(text, tag))
    print(time.time()-st)

    st = time.time()
    print(compare_string_reference(text, tag))
    print(time.time()-st)


if __name__ == '__main__':
    #test_BertCRF_constructor()
    #test_BasicTokenizer()
    #test_pandas_drop()
    #test_pandas_drop_syn()
    #test_metrics()
    #test_CWS_Dict()

    #test_pkuseg()
    #test_FullTokenizer()
    #check_english('candidate defence')
    #check_english('台北candidate defence')

    #test_parse_one2BERTformat()

    #test_load_model()

    #test_dataloader()

    #test_decode()
    #test_split()

    #test_write()

    #test_construct_pos_tags()

    #test_outputPOSFscoreUsedBIO()
    check_time()


