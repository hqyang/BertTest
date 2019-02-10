#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 11:21 2019-01-30 
@author: haiqinyang

Feature: 

Scenario: 
"""
from sklearn.preprocessing import LabelEncoder


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


if __name__ == '__main__':
    #test_BertCRF_constructor()
    #test_BasicTokenizer()
    #test_pandas_drop()
    #test_pandas_drop_syn()
    #test_metrics()
    #test_CWS_Dict()

    test_pkuseg()
