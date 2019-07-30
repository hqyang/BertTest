#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 11:21 2019-01-30 
@author: haiqinyang

Feature: 

Scenario: 
"""
from sklearn.preprocessing import LabelEncoder
from src.utilis import save_model
from src.config import args, segType
from src.utilis import get_dataset_and_dataloader, restore_unknown_tokens_without_unused_with_pos
from src.preprocess import CWS_BMEO, tokenize_list_with_cand_indexes
from src.BERT import BertTokenizer
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


def test_restore_unknown():
    original_str = '不只爱翡翠  一切珠宝 文玩都是我的最爱   一个女生戴着大金刚啥的确实很霸气#用珠宝讲故事# 了😂😂  改天拿出来晒晒太阳  😝#我要上墙# #精品翡翠#'
    result_str = '不 只 爱  翡翠  一切  珠宝 文玩 都 是 我 的 最 爱 一 个  女生 戴 着  大金刚 [UNK] 的  确实 很  霸气 # 用  珠宝 讲  故事 # 了 [UNK]  改天 拿  出来  晒晒  太阳 [UNK] # 我 要 上 墙 # #  精品  翡翠 # '
    result_pos = 'AD AD VV NR DT NN NN AD VC PN DEG AD NN CD M NN VV AS NN PU X X AD VA PU P NN VV NN PU AS PU VV VV VV VV NN PU PU PN VV VV NN PU PU NN NN PU '

    seg_ls, pos_ls = restore_unknown_tokens_without_unused_with_pos(original_str, result_str, result_pos)
    print(seg_ls)
    print(pos_ls)


def test_tokenize_list_with_cand_indexes():
    max_length = 128

    vocab_file = '../src/BERT/models/multi_cased_L-12_H-768_A-12/vocab.txt'
    do_lower_case = True
    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    lword = [['不', '只', '爱', '翡', '翠', '一', '切', '珠', '宝', '文', '玩', '都', '是', '我', '的', '最', '爱', '一', '个', '女', '生', '戴', '着', '大', '金', '刚', '[UNK]', '的', '确', '实', '很', '霸', '气', '#', '用', '珠', '宝', '讲', '故', '事', '#', '了', '[UNK]', '改', '天', '拿', '出', '来', '晒', '晒', '太', '阳', '[UNK]', '#', '我', '要', '上', '墙', '#', '#', '精', '品', '翡', '翠', '#'], ['#', '绘', '画', '机', '器', '人', '#', '⊙', '##∀', '##⊙', '！', '哦', '原', '来', '自', '己', '长', '成', '这', '样', '子', '的', '啦', '！'], ['#', '美', '食', '#', '之', '来', '自', '科', '华', '路', '的', '中', '东', '料', '理', '，', '迪', '拜', '料', '理', '味', '道', '真', '心', '很', '让', '人', '失', '望', '哈', '[UNK]', '味', '道', '太', '一', '般', '，', '果', '然', '只', '是', '美', '而', '已', '。', '还', '是', '土', '耳', '其', '料', '理', '好', '吃'], ['适', '合', '夏', '天', '的', '少', '女', '感', '香', '氛', '[UNK]', '超', '平', '价', '[UNK]', '超', '日', '常', '[UNK]', '。', '[UNK]', '很', '多', '宝', '宝', '平', '时', '呢', '和', '重', '要', '的', '人', '约', '会', '都', '会', '有', '种', '喷', '香', '水', '的', '习', '惯', '，', '但', '是', '呢', '，', '对', '于', '学', '生', '党', '来', '说', '，', '一', '瓶', '香', '水', '都', '很', '贵', '，', '很', '难', '随', '着', '心', '情', '换', '个', '味', '道', '，', '而', '且', '有', '些', '香', '水', '味', '道', '特', '别', '浓', '，', '很', '多', '直', '男', '癌', '们', '觉', '得', '你', '身', '上', '的', '味', '道', '特', '别', '刺', '鼻', '[UNK]', '。', '[UNK]', '所', '以', '我', '现', '在', '喜', '欢', '用', '体', '香', '原', '液', '，', '淡', '淡', '的'], ['清', '香', '，', '连', '出', '汗', '都', '是', '非', '常', '自', '然', '的', '体', '香', '，', '根', '本', '闻', '不', '到', '狐', '臭', '汗', '臭', '啊', '，', '每', '次', '出', '门', '都', '携', '带', '在', '身', '上', '可', '以', '随', '时', '涂', '抹', '，', '每', '次', '涂', '抹', '都', '被', '问', '我', '抹', '了', '什', '么', '那', '么', '香', '，', '哈', '哈', '哈', '哈', '一', '直', '被', '夸', '香', '香', '的', '[UNK]', '#', '狐', '臭', '#', '#', '体', '香', '#'], ['每', '个', '女', '孩', '子', '都', '可', '以', '是', '波', '妞', '，', '但', '遇', '到', '的', '不', '一', '定', '是', '宗', '介'], ['泡', '一', '泡', '，', '换', '新', '车', '啦', '#', '我', '爱', '官', '方', '我', '爱', '热', '门', '#', '#', '逆', '袭', '小', '仙', '女', '#', '[UNK]'], ['秀', '恩', '爱', '必', '备', '超', '可', '爱', '的', '情', '侣', '壁', '纸', '头', '像', '[UNK]', '图', '素', '材', '。', '给', '大', '家', '分', '享', '一', '些', '自', '己', '很', '喜', '欢', '的', '情', '侣', '壁', '纸', '原', '图', '可', '以', '思', '我', '[UNK]', '。', '[UNK]', '[UNK]', '图', '修', '图', '[UNK]', '。', '打', '开', 'pi', '##cs', '##art', '就', '可', '以', '从', '相', '册', '照', '片', '里', '[UNK]', '出', '一', '张', '大', '头', '贴', '，', '然', '后', '在', '这', '些', '卡', '通', '壁', '纸', '的', '空', '白', '处', '再', '调', '整', '一', '下', '亮', '度', '对', '比', '度', '边', '框', '什', '么', '的', '一', '张', '超', '甜', '超', '少', '女', '心', '的', '情', '侣', '头', '像', '壁', '纸', '背', '景'], ['图', '就', '[UNK]', '炉', '啦', '~', '。', '赶', '紧', '和', '男', '朋', '友', '一', '起', '用', '起', '来', '吧', '~', '[UNK]', '#', '有', '个', '恋', '爱', '想', '和', '你', '谈', '#', '#', '壁', '纸', '#', '#', '情', '侣', '[UNK]', '图', '素', '材', '#', '#', '情', '侣', '头', '像', '#'], ['#', '魔', '法', '照', '片', '#', '#', '魔', '法', '星', '云', '照', '#', '。', '小', '时', '候', '，', '总', '是', '满', '心', '幻', '想', '着', '拥', '有', '像', '哈', '利', '波', '特', '一', '样', '的', '魔', '法', '，', '小', '棒', '一', '挥', '，', '让', '自', '己', '拥', '有', '很', '多', '东', '西', '；', '长', '大', '后', '，', '才', '发', '现', '原', '来', '自', '己', '才', '是', '最', '值', '得', '拥', '有', '的', '[UNK]'], ['你', '还', '记', '得', '我', '么', '～', '我', '可', '是', '深', '深', '的', '记', '得', '你', '[UNK]', '[UNK]', '圈', '#', '90', '后', '养', '生', '日', '常', '#', '#', '童', '年', '儿', '时', '的', '回', '忆', '#', '圈', '出', '你', '的', '儿', '时', '小', '零', '食', '、', '回', '忆', '杀', '[UNK]'], ['Iris', '网', '拍', '模', '特', '反', '馈', '[', '嘿', '哈', ']', '。', '网', '拍', '就', '是', '送', '福', '利', '的', '地', '方', '呀', '！', '它', '不', '是', '微', '商', '要', '你', '卖', '货', '，', '而', '是', '简', '单', '拍', '照', '送', '你', '一', '堆', '货', '和', '[UNK]', '。', '它', '稳', '定', '安', '逸', '，', '用', '闲', '时', '就', '可', '以', '赚', '#', 'l', '##ris', '网', '拍', '#'], ['#', 'running', '##man'], ['夏', '日', '显', '白', '风', '。', '红', '色', '显', '白', '，', '搭', '配', '上', '最', '近', '流', '行', '的', '晕', '染', '。', '好', '看', '又', '有', '夏', '日', '气', '息'], ['闲', '的', '没', '事', '，', '放', '松', '下', '心', '情', '吧', '！'], ['木', '糠', '布', '丁', '[UNK]', '上', '次', '在', '澳', '门', '吃', '了', '念', '念', '不', '忘', '，', '今', '天', '自', '己', '动', '手', '做', '了', '。', '請', '給', '个', '赞', '。', '#', '自', '制', '美', '食', '#', '#', '烘', '焙', '食', '谱', '#', '#', '烘', '焙', '#', '#', '自', '己', '动', '手', '美', '食', '#', '#', '自', '己', '动', '手', '丰', '衣', '足', '食', '#', '#', '夏', '日', '甜', '點', '自', '己', '動', '手', '作', '#'], ['旗', '袍', '_', '216', '旗', '袍', '的', '发', '展', '史', '-', '古', '典', '旗', '袍', '。', '旗', '袍', ':', '相', '约', '温', '婉', '荣', '华', '。', '让', '时', '光', '把', '粗', '[UNK]', '褪', '去', '。', '#', '旗', '袍', '#', '#', '古', '风', '#', '#', '写', '真', '#', '#', '后', '期', '制', '作', '#', '#', '摄', '影', '#', '。', '让', '岁', '月', '把', '沧', '桑', '融', '化', '。', '穿', '过', '一', '个', '世', '纪', '的', '遥', '远', '距', '离', '。', '饱', '含', '天', '生', '的', '[UNK]', '媚', '端', '庄', '。', '和', '你', '相', '约', '温', '婉', '荣', '华', '。', '纤', '尘', '抖', '落', '在', '历', '史', '角', '落', '中', '。', '寂', '静', '地', '让', '出', '光', '彩', '的'], ['道', '路', '。', '让', '我', '脉', '脉', '含', '情', '而', '来', '。', '在', '百', '花', '开', '放', '的', '春', '天', '。', '在', '热', '情', '似', '火', '的', '夏', '日', '。', '在', '流', '水', '[UNK]', '[UNK]', '的', '金', '秋', '。', '在', '白', '雪', '飘', '飞', '的', '冬', '季', '。', '一', '如', '既', '往', '。', '与', '你', '相', '约', '温', '婉', '荣', '华', '。', '#', '旗', '袍', '#', '#', '古', '风', '#', '#', '写', '真', '#', '#', '后', '期', '制', '作', '#', '#', '摄', '影', '#'], ['星', '星', '灯', '＃', '当', '我', '看', '着', '你', '的', '时', '候', '眼', '里', '也', '有', '星', '星', '[UNK]'], ['没', '有', '那', '把', '剑', '，', '我', '照', '样', '可', '以', '歼', '灭', '敌', '军'], ['拍', '出', '普', '通', '人', '最', '美', '的', '一', '面', '～'], ['小', '姐', '姐', '又', '在', '拍', '拍', '拍', '啦', '[UNK]', '#', '逆', '袭', '小', '仙', '女', '#'], ['第', '一', '次', '做', '的', '图', '案', '勉', '强', '能', '看', '#', '520', '陪', '你', '撒', '糖', '#'], ['#', '地', '狱', '少', '女', '#', '#', '阎', '魔', '爱', '#', '[UNK]'], ['本', '人', '因', '工', '作', '原', '因', '，', '经', '常', '出', '差', '，', '家', '里', '有', '只', '狗', '狗', '不', '能', '照', '顾', '，', '恳', '请', '有', '爱', '心', '的', '女', '性', '山', '东', '菏', '泽', '本', '地', '朋', '友', '免', '费', '领', '养', ':', 'V', '##X', ':', 'j', '##sam', '##ine', '##12', '##12'], ['发', '一', '些', '艺', '术', '氛', '围', '的', '创', '作', '，', '美', '图', '上', '99', '%', '是', '千', '篇', '一', '律', '的', '糖', '水', '片', '，', '不', '耐', '看', '。'], ['黑', '龙', '江', '旅', '游', '攻', '略', '之', '五', '大', '连', '池', '火', '山', '探', '索', '之', '旅', '。', '[UNK]', '行', '程', '计', '划', '。', 'Day', '##1', ':', '。', '入', '住', '五', '大', '连', '池', '风', '景', '区', '万', '豪', '名', '苑', '商', '务', '酒', '店', '[UNK]', '[UNK]', '晚', '餐', '矿', '泉', '冷', '水', '鱼', '鱼', '宴', '。', 'Day', '##2', ':', '。', '新', '期', '火', '山', '观', '光', '区', '老', '黑', '山', '、', '火', '烧', '山', '[UNK]', '[UNK]', '午', '餐', '矿', '泉', '豆', '腐', '宴', '[UNK]', '[UNK]', '温', '泊', '[UNK]', '[UNK]', '水', '晶', '宫', '景', '区', '[UNK]', '[UNK]', '龙', '门', '石', '寨', '景', '区', '。', 'Day', '##3', '：', '。', '圣', '水', '祭', '典', '[UNK]', '[UNK]', '花'], ['车', '巡', '游', '[UNK]', '[UNK]', '泥', '浆', '大', '战', '[UNK]', '[UNK]', '火', '山', '温', '泉', '体', '验', '[UNK]', '[UNK]', '开', '幕', '式', '及', '歌', '舞', '晚', '会', '[UNK]', '[UNK]', '电', '音', '之', '夜', '。', 'Day', '##4', ':', '。', '返', '程', '。', '#', '长', '风', '计', '划', '#', '@', 'MT', '情', '报', '局', '@', 'MT', '小', '助', '手', '@', 'MT', '居', '委', '会', '@', 'MT', '玩', '图', '君'], ['#', '天', '津', '探', '店', '#', '#', '天', '津', '美', '食', '#', '。', '店', '名', ':', '羽', '深', '。', '图', '一', '焦', '糖', '冰', '淇', '淋', '我', '爱', '了', '[UNK]', '。', '餐', '厅', '环', '境', '也', '挺', '好', '的', '。', '推', '荐', '点', '牛', '排', '[UNK]', '，', '还', '有', '番', '茄', '牛', '[UNK]', '。', '需', '要', '提', '前', '订', '一', '下'], ['[UNK]', '超', '美', '背', '景', '墙', '送', '给', '你', '[UNK]'], ['鼓', '楼', '区', '老', '年', '大', '学', '汇', '报', '演', '出', '圆', '满', '结', '束'], ['[', '奸', '笑', ']', '[', '奸', '笑', ']', '腮', '红', '继', '续', '。', '以', '前', '我', '总', '觉', '得', '腮', '红', '用', '不', '用', '无', '所', '谓', '。', '但', '是', '后', '来', '我', '发', '现', '，', '眼', '影', '没', '时', '间', '细', '描', '慢', '化', '。', '直', '接', '腮', '红', '[UNK]', '口', '红', '。', '气', '色', '直', '接', '翻', '倍', '，', '年', '轻', '3', '-', '5', '岁', '[', '机', '智', ']'], ['[UNK]', '[', 'Mi', '##cist', '##y', '密', '汐', '[UNK]', '迪', '束', '腰', '带', ']', '。', '越', '美', '的', '人', '对', '自', '己', '的', '要', '求', '总', '是', '越', '高', '。', '不', '仅', '可', '以', '辅', '助', '燃', '脂', '減', '小', '腰', '围', '。', '运', '动', '刷', '脂', '还', '可', '以', '保', '護', '腰', '椎', '。', '塑', '形', '的', '同', '時', '帮', '你', '管', '控', '坐', '姿', '。', '缓', '解', '不', '良', '习', '惯', '导', '致', '的', '腰', '部', '疲', '劳', '。', '有', '效', '的', '矫', '正', '驼', '背', '以', '及', '脊', '椎', '弯', '曲', '。', '不', '仅', '仅', '是', '一', '件', '束', '腰', '哦', '。', '久', '坐', '人', '群', '都', '不', '会', '难', '受', '的', '一', '款', '束', '腰'], [], ['你', '归', '来', '是', '诗', '，', '离', '去', '成', '词', '，', '且', '笑', '风', '尘', '不', '敢', '造', '次', '。'], ['这', '波', '剧', '也', '太', '太', '太', '太', '好', '看', '了', '吧', '。', '有', '你', '们', '喜', '欢', '的', '吗', '。', '像', '看', '什', '么', '壁', '纸', '或', '电', '视', '剧', '评', '论', '区', '告', '诉', '我', '[UNK]', '#', '官', '方', '大', '大', '我', '要', '上', '热', '#'], ['那', '是', '雨', '后', '的', '傍', '晚', '[UNK]', '我', '有', '幸', '见', '到', '了', '这', '天', '象', '：', '云', '梅', '像', '美', '丽', '的', '梅', '花', '在', '天', '上', '飞', '来', '飞', '去', '，', '瞬', '间', '变', '化', '无', '穷', '，', '美', '轮', '美', '[UNK]', '！'], ['#', '沈', '阳', '二', '手', '车', '#', '[UNK]', '新', '到', '12', '年', '现', '代', '索', '纳', '塔', '八', '，', '自', '动', '2', '.', '0', '最', '高', '配', '，', '真', '皮', '座', '椅', '大', '天', '窗', '，', '导', '航', '倒', '车', '影', '像', '，', '巡', '航', '定', '速', '。', '一', '手', '车', '，', '全', '车', '原', '版', '。', '8', '万', '公', '里', '，', '费', '用', '全', '年', '，', '首', '付', '8000', '车', '款', '[UNK]'], ['清', '理', '空', '置', '房', '院', '子', '里', '的', '杂', '草'], ['好', '朋', '友', '的', '女', '儿', '确', '诊', '白', '血', '病', '，', '才', '8', '岁', '，', '为', '了', '鼓', '励', '她', '，', '我', '和', '她', '爸', '爸', '都', '剃', '了', '光', '头', '，', '陪', '她', '一', '起', '加', '油', '！'], ['今', '日', '份', '背', '景', '图', '：', '。', '[UNK]', '枕', '头', '地', '下', '一', '颗', '糖', '，', '做', '一', '个', '甜', '甜', '的', '梦', '.', '[UNK]', '。', '#', '有', '个', '恋', '爱', '想', '和', '你', '谈', '#', '#', '朋', '友', '圈', '背', '景', '图', '#', '#', 'ins', '风', '#', '#', '无', '水', '印', '#', '#', '少', '女', '心', '#'], ['这', '个', '画', '风', '我', '爱', '了', '(', '[UNK]', ')', '-', '♡', '。', 't', '##wi', ':', 'sa', '##kus', '##ya', '##2', '##hon', '##da', '。', '@', 'MT', '情', '报', '局', '#', '动', '漫', '图', '#', '#', '好', '看', '的', '动', '漫', '图', '#', '#', '壁', '纸', '#'], ['呵', '呵', '[UNK]', '消', '磨', '时', '间', '玩', '一', '玩'], ['[UNK]', '你', '必', '买', '的', '优', '梵', '溶', '脂', '霜', '[UNK]', '[UNK]', '。', '很', '难', '过', '要', '让', '你', '和', '你', '多', '年', '的', '小', '粗', '腿', '说', '白', '白', '了', '[UNK]', '随', '时', '随', '地', '滚', '一', '滚', '[UNK]', '一', '[UNK]', '和', '大', '象', '腿', '水', '桶', '腰', '不', '复', '再', '相', '见', '让', '情', '敌', '看', '见', '被', '气', '死', '让', '前', '任', '看', '见', '会', '后', '悔', '的', '瘦', '身', '神', '器', '全', '身', '都', '能', '瘦', '紧', '致', '溶', '脂', '去', '肌', '肉', '腿', '还', '能', '美', '白', '有', '一', '点', '辣', '辣', '的', '但', '是', '不', '刺', '激', '的', '敏', '感', '肌', '可', '用'], ['炎', '炎', '夏', '日', '来', '一', '杯', '加', '冰', '的', '苹', '果', '汁', '叭', '[UNK]'], ['没', '有', '冰', '奶', '茶', '的', '夏', '天', '不', '完', '美', '.', '.', '.', '.', '.'], ['#', '书', '籍', '安', '利', '#', '《', '与', '老', '妈', '的', '日', '常', '》', '。', '里', '面', '是', '一', '偏', '偏', '的', '小', '漫', '画', '，', '偏', '偏', '入', '心', '[UNK]', '。', '随', '手', '摘', '抄', '[UNK]', '分', '享', '中', '[UNK]', '[UNK]', '[UNK]'], ['#', '手', '绘', '#', '一', '天', '一', '画', '，', '贵', '在', '坚', '持', '[UNK]', '望', '友', '友', '们', '点', '赞', '评', '论', '[UNK]'], ['青', '藏', '特', '产', '红', '[UNK]', '杞', '，', '黑', '[UNK]', '杞', '，', '藏', '红', '花', '，', '纯', '天', '然', '产', '地', '正', '宗', '，', '抢', '购', '热', '线', ':', '1383', '##47', '##0', '##86', '##11'], ['美', '衣', '魅', '力', '，', '谁', '穿', '谁', '美', '丽', '[UNK]'], ['给', '大', '家', '推', '荐', '一', '部', '网', '剧', '。', '叫', '灵', '魂', '摆', '渡', '人', '。', '这', '部', '网', '络', '剧', '总', '共', '有', '3', '部', '。', '这', '个', '是', '1', '-', '2', '集', '一', '个', '故', '事', '。', '跟', '鬼', '片', '是', '一', '样', '的', '。', '这', '部', '网', '络', '剧', '反', '应', '了', '现', '实', '中', '很', '多', '存', '在', '的', '东', '西', '。', '该', '剧', '讲', '述', '了', '夏', '冬', '青', '和', '鬼', '差', '赵', '吏', '、', '九', '天', '玄', '女', '王', '小', '亚', '三', '人', '在', '处', '理', '一', '件', '件', '有', '关', '鬼', '魂', '和', '灵', '魂', '的', '故', '事', '中', '感', '悟', '人', '生', '的', '故', '事', '。', '真', '的', '超', '级', '好', '看', '。', '我', '都', '看', '了'], ['好', '多', '遍', '了', '。', '特', '别', '喜', '欢', '里', '面', '摆', '渡', '人', '赵', '吏', '。', '还', '有', '里', '面', '的', '冥', '王', '茶', '茶', '。', '真', '的', '超', '级', '推', '荐', '。'], ['包', '装', '好', '看', '颜', '色', '也', '好', '看', '。', '显', '色', '不', '飞', '粉', '，', '哑', '光', '珠', '光', '共', '12', '色', '，', '打', '造', '各', '种', '妆', '容', '谁', '用', '谁', '爱', '#', '欧', '束', '#'], ['#', '你', '好', '夏', '天', '#', '。', '上', '班', '高', '峰', '。', '胡', '子', '都', '没', '刮', '[UNK]'], ['余', '光', '中', '先', '生', '说', ':', '[UNK]', '月', '色', '与', '雪', '色', '之', '间', '，', '你', '是', '第', '三', '种', '绝', '色', '.', '[UNK]', '#', '背', '影', '杀', '#'], ['新', '款', '点', '点', '装', '～', '清', '新', '的', '像', '美', '少', '女', '～'], ['我', '的', '哈', '密', '洞', '！', '[UNK]'], ['请', '忽', '略', '粗', '略', '的', '背', '景', '～', '还', '有', '胖', '胖', '的', '腿', '腿', '(', '〃', '[UNK]', ')', '。', '这', '个', '防', '晒', '喷', '雾', '真', '的', '超', '级', '好', '用', '～', '一', '点', '不', '油', '～', '超', '级', '舒', '服', '，', '味', '道', '也', '很', '好', '闻', '～'], ['#', '如', '果', '有', '一', '天', '我', '消', '失', '了', '#', '你', '会', '怎', '样'], ['侄', '孙', '女', '儿', '具', '然', '和', '我', '一', '天', '生', '日', '！', '可', '喜', '可', '贺', '！', '同', '喜', '同', '乐', '啊', '！', '[UNK]'], ['同', '事', '相', '聚', '，', '美', '食', '少', '不', '了', '。'], ['[UNK]'], ['超', '级', '热', '的', '天', '气', '喝', '杯', '冰', '茶', '解', '暑', '啦', '啦', '啦', '[UNK]', '^', 'V', '^'], ['美', '术', '期', '末', '考', '试', 'over', '##～', '。', '感', '觉', '自', '己', '只', '有', 'b', '+', '[UNK]', '#', '美', '术', '#', '#', '期', '末', '#', '#', '考', '试', '#', '#', '高', '一', '#', '#', '长', '风', '计', '划', '#']]
    tuple1, tuple2 = zip(
            *[tokenize_list_with_cand_indexes(w, max_length, tokenizer) for w in lword])



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
    #check_time()

    #test_restore_unknown()
    test_tokenize_list_with_cand_indexes()


