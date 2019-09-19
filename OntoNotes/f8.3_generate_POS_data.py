#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 2:31 PM 25/12/2018 
@author: haiqinyang

Feature: 

Scenario: 
"""
import os
from src.BERT.tokenization import BertTokenizer
import numpy as np
from src.preprocess import read_dict
from common import genDataWithBERTSeg_with_dict, parse_one2BERTformat, parse_one2BERT2Dict_with_dict


def gen_4ner_type_with_dict(full_tokenizer, in_dir, out_dir, parts, dict):
    pos_all = set()

    fs = os.walk(in_dir)
    (p, d, f) = zip(*fs)

    files = {}
    for part in parts:
        files[part] = ''

    for fi in f[0]:
        for part in parts:
            if part in fi: files[part] = fi

    for part in parts:
        if len(files[part]) == 0: continue

        infile = os.path.join(p[0], files[part])

        pos = set()
        genDataWithBERTSeg_with_dict(full_tokenizer, infile, out_dir, part, pos, dict)
        pos_all = pos_all.union(pos)

    with open(out_dir +'all_pos_set.txt', 'w+') as f:
        for pos in pos_all:
            f.write(pos + '\n')


def show_parse_one2BERT(s, full_tokenizer, pos_set, dict):
    bert_seg, bert_ner, bert_pos, src_seg, src_ner, src_pos, full_pos, text_str, text_seg \
        = parse_one2BERTformat(s, full_tokenizer, pos_set)

    print(s)
    print(src_seg)
    print(bert_seg)

    print(src_ner)
    print(bert_ner)

    print(full_pos)
    print(text_str)
    print(text_seg)

    print(src_pos)
    print(bert_pos)


if __name__ == '__main__':
    lm = 'bert-base-chinese'
    lm = 'multi_cased_L-12_H-768_A-12'
    dict_file = '../resource/dict.txt'
    dict = list(read_dict(dict_file).keys())

    vocab_file = '../src/BERT/models/' + lm + '/vocab.txt'
    full_tokenizer = BertTokenizer(vocab_file, do_lower_case=True)

    cases = ['(NP (CP (IP (NP (DNP (NER-GPE (NR 台湾)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN 候选人) (NN 辩论会)) (PU ，))))',
        '(VA ＥＭＰＴＹ)', '(NP (CP (IP (NP (DNP (NER-GPE (NR #students)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN candidate) (NN defence)) (PU ，))))',
        '(NP (CP (IP (NP (DNP (NER-GPE (NR Taiwan)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN candidate) (NN defence) (VV 全力以赴)) (PU ，))))',
        '(IP (NP (NP (NER-ORG (NR 新华社)) (NP-m (NER-GPE (NR 开罗)) (NP-m (NP (NT １１月) (NT ２４日)) (NN 电)))) (PRN (PRN-m (PU （) (NP (NN 记者) (NP (NER-PER (NR 郭春菊)) (NER-PER (NR 林建杨))))) (PU ）))) (IP-m (NP (NP (NP (NER-ORG (NR 阿拉伯) (NN 国家) (NN 联盟)) (PRN (PRN-m (PU （) (NER-ORG (NR 阿盟))) (PU ）))) (NN 秘书长)) (NER-PER (NR 穆萨))) (IP-m (VP (NT ２４日) (VP (VP-m (VP (VV 发表) (NN 声明)) (PU ，)) (VP (VV 谴责) (NP (CP (IP (NP (NP (NP (NER-GPE (NR 伊拉克)) (NN 首都)) (NER-GPE (NR 巴格达))) (NP-m (NN 东部) (NER-GPE (NR 萨德尔城)))) (VP (NT ２３日) (VV 发生))) (DEC 的)) (NP-m (JJ 连环) (NP-m (NP (NN 汽车) (NP-m (NN 炸弹) (NN 袭击))) (NN 事件))))))) (PU 。))))',
        '(IP (IP (NP (NP (NER-ORG (NR 新华社)) (NP-m (NER-GPE (NR 吉隆坡)) (NP-m (NP (NT １１月) (NT ２４日)) (NN 电)))) (PRN (PRN-m (PU （) (NP (NN 记者) (NP (NER-PER (NR 公兵)) (NP-m (PU 、) (NER-PER (NR 熊平)))))) (PU ）))) (IP-m (CP (CS 尽管) (IP (NP (QP (OD 第九) (M 届)) (NP-m (NR 远南) (NN 运动会))) (VP (NT ２５日) (VP-m (AD 才) (VP-m (AD 正式) (VV 开幕)))))) (IP-m (PU ，) (IP-m (AD 但) (IP-m (NN 比赛) (VP (PP (P 在) (NT ２４日)) (VP-m (AD 已经) (VV 打响)))))))) (IP-m (PU ，) (IP-m (IP (NP (NER-GPE (NR 中国)) (NN 代表团)) (VP (PP (P 在) (LCP (NP (DNP (NT 当日) (DEG 的)) (NP-m (QP (CD 三) (M 场)) (NN 比赛))) (LC 中))) (VP-m (AD 均) (VP (VV 取得) (NN 胜利))))) (PU 。))))'
         ]

    pos_set = set()

    for s in cases:
        show_parse_one2BERT(s, full_tokenizer, pos_set, dict)

        out_dict = parse_one2BERT2Dict_with_dict(s, full_tokenizer, pos_set, dict)
        print('src_seg:'+out_dict['src_seg'])
        print('src_ner:'+out_dict['src_ner'])
        print('full_pos:'+out_dict['full_pos'])
        print('text:'+out_dict['text'])
        print('text_seg:'+out_dict['text_seg'])
        print('bert_seg:'+out_dict['bert_seg'])
        print('bert_ner:'+out_dict['bert_ner'])
        print('bert_pos:'+out_dict['bert_pos'])
        print('word_in_dict_tuple:'+str(out_dict['word_in_dict_tuple']))

    TEST_FLAG = False
    inServer = False
    inServer = True
    #TEST_FLAG = True
    if TEST_FLAG:
        in_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4nerpos_update/valid/src/'
        out_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4nerpos_update/valid/feat_with_dict/'

        parts = ['train', 'test', 'dev']
    else:
        if inServer:
            in_dir = '../../data/ontonotes5/4nerpos_ori/'
            out_dir = '../../data/ontonotes5/4nerpos_update/feat_with_dict/'
        else:
            in_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/5.fuse-tree2/'
            out_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4nerpos_update/feat_with_dict/'

        parts = ['train', 'test', 'dev']
        os.makedirs(out_dir, exist_ok=True)

    gen_4ner_type_with_dict(full_tokenizer, in_dir, out_dir, parts, dict)

    print()
    print(pos_set)
    print('Finish processing files in ' + in_dir)

