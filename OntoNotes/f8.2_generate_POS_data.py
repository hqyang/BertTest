#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 2:31 PM 25/12/2018 
@author: haiqinyang

Feature: 

Scenario: 
"""
import os
import re
from src.BERT.tokenization import BertTokenizer
import numpy as np
from src.utilis import check_english_words, check_chinese_words, define_tokens_set
from src.config import langtype, bmes2bio
from tqdm import tqdm

def output_seg_tokens(word, full_tokenizer, ner_type=''):
    # only one word, e.g., director, 日子， 在
    # ner_type
    # cws format: BMES
    # ner format: BMEWO
    #bChinese = check_chinese_words(word)

    bChinese = check_chinese_words(word)

    if bChinese: # chinese
        len_w = len(word)

        if len_w == 1: # only one character
            src_seg_gt = ['S']

            if len(ner_type)>0:
                src_ner_gt = ['W' + ner_type]
            else:
                src_ner_gt = ['O']
            bert_seg_gt = src_seg_gt # the same when there is no English
            bert_ner_gt = src_ner_gt # the same when there is no English
        else: #multiple chars
            wl = full_tokenizer.tokenize(word)
            cand_ind = define_tokens_set(wl)
            len_wl = len(cand_ind)

            src_seg_gt = ['B'] + ['M'] * (len_w - 2) + ['E']
            bert_seg_gt = ['B'] + ['M'] * (len_wl - 2) + ['E'] # different if len_w != len_wl

            if len(ner_type)>0:
                src_ner_gt = ['B' + ner_type] + ['M' + ner_type] * (len_w - 2) + ['E' + ner_type]
                bert_ner_gt = ['B' + ner_type] + ['M' + ner_type] * (len_wl - 2) + ['E' + ner_type]
            else:
                src_ner_gt = ['O'] * len_w
                bert_ner_gt = ['O'] * len_wl
    else: # English, numeric, or other special tokens
        src_seg_gt = ['S']

        if len(ner_type)>0:
            src_ner_gt = ['W' + ner_type]
        else:
            src_ner_gt = ['O']

        wl = full_tokenizer.tokenize(word)
        cand_ind = define_tokens_set(wl)
        len_wl = len(cand_ind)

        if len_wl == 1:
            bert_seg_gt = src_seg_gt
            bert_ner_gt = src_ner_gt
        else:
            bert_seg_gt = ['B'] + ['M'] * (len_wl - 2) + ['E']

            if len(ner_type)>0:
                bert_ner_gt = ['B' + ner_type] + ['M' + ner_type] * (len_wl - 2) + ['E' + ner_type]
            else:
                bert_ner_gt = ['O'] * len_wl

    return bert_seg_gt, bert_ner_gt, src_ner_gt, src_seg_gt, bChinese


def parse_one2BERTformat(s, full_tokenizer, pos_set): # store into lists
    # Adopt BMEWO for NER tagging,
    #   see https://lingpipe-blog.com/2009/10/14/coding-chunkers-as-taggers-io-bio-bmewo-and-bmewo/
    #
    #  An example
    #   s = '(NP (CP (IP (NP (DNP (NER-GPE (NR Taiwan)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN candidate) (NN defence)) (PU ，))))'
    #
    #   src_seg, src_ner, full_pos, text_str, text_seg, bert_ner
    #       = parse_one2BERTformat(s)
    #
    # output:
    #   # Chinese word segmentation
    #   src_seg = ['S', 'S', 'B', 'E', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'B', 'E', 'S', 'S', 'S']
    #   # NER: current BIO, consider to change to BMEWO+
    #   src_ner = ['W-GPE', 'O', 'B-ORG', 'E-ORG', 'B', 'E', 'B', 'E', 'O', 'B', 'E', 'B', 'E', 'O', 'O', 'O']
    #   # full_pos: chunk information
    #   full_pos = ['(NP', '(CP', '(IP', '(NP', '(DNP', '(NR', ')NR', ')DNP', '(DEG', ')DEG', ')NP', '(NR', ')NR', ')IP', ')CP', '(VP', '(NT', ')NT', '(VV', ')VV', ')VP', ')NP', '(DEC', ')DEC', ')DEC', '(NP-m', '(NP', '(NR', ')NR', '(NN', ')NN', ')NP', '(NP-m', '(NP', '(NN', ')NN', '(NN', ')NN', ')NP', '(PU', ')PU', ')NP-m', ')NP-m', ')NP-m']
    #   # text
    #   text_str = 'Taiwan的公视今天主办的台北市长candidate defence，'
    #   text_seg = 'Taiwan 的 公视 今天 主办 的 台北 市长 candidate defence ，'

    s = re.sub('\)', ') ', s)
    s = re.sub(' +', ' ', s).strip()
    pos = s.split(' ')
    buffer = []
    innermost = True
    full_pos = []
    bert_seg = [] # bert_seg for storing the segmentation of bert format, additional processing for words and English
    bert_pos = []
    bert_ner = [] # bert_seg for storing the ner segmentation of bert format, additional processing for words and English

    src_seg = []  # src_seg for storing the segmentation of resource words
    src_pos = []
    src_ner = []  # src_ner for storing the ner segmentation of resource words
    ner_types = []
    text = []  # store the resource words, English words and numbers are separated by space
    lang_status_list = [] # store the language type: 'C' (Chinese); 'NE' (Number and English)
    #num_ner = 0 # number of NER
    #num_words = 0 # number of words
    #len_chars = 0 # number of chars

    for p in pos:
        if '(' in p:
            innermost = True
            if 'NER' in p:
                ner_types.append(re.sub('NER', '', p[1:]))
                continue
            else:
                buffer.append(p)
        elif ')' in p:
            if buffer != []:
                suffix = buffer.pop()[1:]
                if innermost:
                    assert len(p) > 1
                    word = p[:-1]
                    text.append(word)
                    word = re.sub('“|”', '"', word)
                    innermost = False
                    p = p[-1]

                    if ner_types != []:
                        ner_type = ner_types.pop()
                        #num_ner += 1
                    else:
                        ner_type = ''

                    # Process multi-lingual
                    bert_seg_gt, bert_ner_gt, src_ner_gt, src_seg_gt, bChinese = output_seg_tokens(word, full_tokenizer, ner_type)

                    bert_pos_gt = [bmes2bio.map[x]+'-'+suffix for x in bert_seg_gt]
                    src_pos_gt = [bmes2bio.map[x]+'-'+suffix for x in src_seg_gt]

                    if bChinese:
                        lang_status_list.append(langtype.CHINESE)
                    else:
                        lang_status_list.append(langtype.OTHERS)

                    src_seg.extend(src_seg_gt)
                    src_ner.extend(src_ner_gt)
                    src_pos.extend(src_pos_gt)
                    bert_seg.extend(bert_seg_gt)
                    bert_ner.extend(bert_ner_gt)
                    bert_pos.extend(bert_pos_gt)

                    pos_set.add(suffix)
            p += suffix
        full_pos.append(p)

    text_seg = ' '.join(text)

    # additionally add space for other characters
    text_str = ''
    for idx in range(len(lang_status_list)):
        if idx>0 and lang_status_list[idx-1]==langtype.OTHERS and lang_status_list[idx]==langtype.OTHERS:
            text_str += ' ' + text[idx]
        else:
            text_str += text[idx]

    return bert_seg, bert_ner, bert_pos, src_seg, src_ner, src_pos, full_pos, text_str, text_seg


def parse_one2BERT2Dict(s, full_tokenizer, pos_set):
    bert_seg, bert_ner, bert_pos, src_seg, src_ner, src_pos, full_pos, text, text_seg = parse_one2BERTformat(s, full_tokenizer, pos_set)

    bert_seg = ' '.join(bert_seg)
    bert_ner = ' '.join(bert_ner)
    bert_pos = ' '.join(bert_pos)

    src_seg = ','.join(src_seg) + ','
    src_ner = ','.join(src_ner) + ','
    src_pos = ','.join(src_pos) + ','
    full_pos = ' '.join(full_pos)

    return {'src_seg': src_seg,  'src_ner': src_ner, 'full_pos': full_pos, 'text': text,
            'text_seg': text_seg, 'bert_seg': bert_seg, 'bert_ner': bert_ner,
            'src_pos': src_pos, 'bert_pos': bert_pos}


def gen_data(in_file, out_dir, mode):
    with open(in_file, 'r', encoding='utf8') as f:
        raw_data = f.readlines()

    #seg_all, ner_all, chunk_all, text_all = zip(*[parse_one_2str_list(s) for s in raw_data])
    #data_all = zip(*[parse_one_2dict(s) for s in raw_data])
    data_all = [parse_one_2dict(s) for s in raw_data]

    import pandas as pd
    df = pd.DataFrame(data_all)
    # separate with \t
    df.to_csv(out_dir+mode+'.tsv', sep='\t', encoding='utf-8', index=False)

    print('Finish writing generated ' + mode + ' data!')


def genDataWithBERTSeg(full_tokenizer, in_file, out_dir, part, pos_set):
    print('Running genDataWithBERTSeg...')
    #vocab_file = '../src/BERT/models/bert-base-chinese/vocab.txt'
    #full_tokenizer = BertTokenizer(vocab_file, do_lower_case=True)

    with open(in_file, 'r', encoding='utf8') as f:
        raw_data = f.readlines()

    data_all = [parse_one2BERT2Dict(s, full_tokenizer, pos_set) for s in tqdm(raw_data)]

    import pandas as pd
    df = pd.DataFrame(data_all)
    # separate with \t
    df.to_csv(out_dir+part+'.tsv', sep='\t', encoding='utf-8', index=False)

    with open(out_dir + part + '_pos_set.txt', 'w+') as f:
        for pos in pos_set:
            f.write(pos + '\n')

    print('Finish writing generated ' + part + ' data!')


def gen_4ner_type(full_tokenizer, in_dir, out_dir, parts):
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
        genDataWithBERTSeg(full_tokenizer, infile, out_dir, part, pos)
        pos_all = pos_all.union(pos)

    with open(out_dir +'all_pos_set.txt', 'w+') as f:
        for pos in pos_all:
            f.write(pos + '\n')


def show_parse_one2BERT(s, full_tokenizer, pos_set):
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

    #print(pos_set)


if __name__ == '__main__':
    lm = 'bert-base-chinese'
    lm = 'multi_cased_L-12_H-768_A-12'

    vocab_file = '../src/BERT/models/' + lm + '/vocab.txt'
    full_tokenizer = BertTokenizer(vocab_file, do_lower_case=True)

    cases = ['(NP (CP (IP (NP (DNP (NER-GPE (NR 台湾)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN 候选人) (NN 辩论会)) (PU ，))))',
        '(VA ＥＭＰＴＹ)', '(NP (CP (IP (NP (DNP (NER-GPE (NR #students)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN candidate) (NN defence)) (PU ，))))',
        '(NP (CP (IP (NP (DNP (NER-GPE (NR Taiwan)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN candidate) (NN defence)) (PU ，))))',
        '(IP (NP (NP (NER-ORG (NR 新华社)) (NP-m (NER-GPE (NR 开罗)) (NP-m (NP (NT １１月) (NT ２４日)) (NN 电)))) (PRN (PRN-m (PU （) (NP (NN 记者) (NP (NER-PER (NR 郭春菊)) (NER-PER (NR 林建杨))))) (PU ）))) (IP-m (NP (NP (NP (NER-ORG (NR 阿拉伯) (NN 国家) (NN 联盟)) (PRN (PRN-m (PU （) (NER-ORG (NR 阿盟))) (PU ）))) (NN 秘书长)) (NER-PER (NR 穆萨))) (IP-m (VP (NT ２４日) (VP (VP-m (VP (VV 发表) (NN 声明)) (PU ，)) (VP (VV 谴责) (NP (CP (IP (NP (NP (NP (NER-GPE (NR 伊拉克)) (NN 首都)) (NER-GPE (NR 巴格达))) (NP-m (NN 东部) (NER-GPE (NR 萨德尔城)))) (VP (NT ２３日) (VV 发生))) (DEC 的)) (NP-m (JJ 连环) (NP-m (NP (NN 汽车) (NP-m (NN 炸弹) (NN 袭击))) (NN 事件))))))) (PU 。))))',
        '(IP (IP (NP (NP (NER-ORG (NR 新华社)) (NP-m (NER-GPE (NR 吉隆坡)) (NP-m (NP (NT １１月) (NT ２４日)) (NN 电)))) (PRN (PRN-m (PU （) (NP (NN 记者) (NP (NER-PER (NR 公兵)) (NP-m (PU 、) (NER-PER (NR 熊平)))))) (PU ）))) (IP-m (CP (CS 尽管) (IP (NP (QP (OD 第九) (M 届)) (NP-m (NR 远南) (NN 运动会))) (VP (NT ２５日) (VP-m (AD 才) (VP-m (AD 正式) (VV 开幕)))))) (IP-m (PU ，) (IP-m (AD 但) (IP-m (NN 比赛) (VP (PP (P 在) (NT ２４日)) (VP-m (AD 已经) (VV 打响)))))))) (IP-m (PU ，) (IP-m (IP (NP (NER-GPE (NR 中国)) (NN 代表团)) (VP (PP (P 在) (LCP (NP (DNP (NT 当日) (DEG 的)) (NP-m (QP (CD 三) (M 场)) (NN 比赛))) (LC 中))) (VP-m (AD 均) (VP (VV 取得) (NN 胜利))))) (PU 。))))'
         ]

    pos_set = set()

    for s in cases:
        show_parse_one2BERT(s, full_tokenizer, pos_set)

        out_dict = parse_one2BERT2Dict(s, full_tokenizer, pos_set)
        print('src_seg:'+out_dict['src_seg'])
        print('src_ner:'+out_dict['src_ner'])
        print('full_pos:'+out_dict['full_pos'])
        print('text:'+out_dict['text'])
        print('text_seg:'+out_dict['text_seg'])
        print('bert_seg:'+out_dict['bert_seg'])
        print('bert_ner:'+out_dict['bert_ner'])
        print('bert_pos:' +out_dict['bert_pos'])

    TEST_FLAG = False
    inServer = False
    #TEST_FLAG = True
    if TEST_FLAG:
        out_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4nerpos_data/valid/'
        in_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4nerpos_data/valid/src/'

        parts = ['train', 'test', 'dev']

        gen_4ner_type(full_tokenizer, in_dir, out_dir, parts)

        print()
        print(pos_set)
        print('Finish processing files in ' + in_dir)
    else:
        if inServer:
            in_dir = '../../data/'
            out_dir = '../../data/ontonotes5/4nerpos_update'
        else:
            in_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/5.fuse-tree2/'
            out_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4nerpos_update/'

        parts = ['train', 'test', 'dev']
        os.makedirs(out_dir, exist_ok=True)

        gen_4ner_type(full_tokenizer, in_dir, out_dir, parts)

