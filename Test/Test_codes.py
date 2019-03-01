#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 11:21 2019-01-30 
@author: haiqinyang

Feature: 

Scenario: 
"""
from sklearn.preprocessing import LabelEncoder
#import sys
#sys.path.append('..')

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

    vocab_file = '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/vocab.txt'
    full_tokenizer = FullTokenizer(vocab_file, do_lower_case=True)
    text = '台湾的公视今天主办的台北市长 Candidate  Defence  ，'
    print(full_tokenizer.tokenize(text))

    text = 'Candidate'
    print(full_tokenizer.tokenize(text))

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
