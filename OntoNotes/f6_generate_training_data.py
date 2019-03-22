#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 2:31 PM 25/12/2018 
@author: haiqinyang

Feature: 

Scenario: 
"""
import os
import sys
sys.path.append('..')
import re
import csv
from src.tokenization import BasicTokenizer, FullTokenizer
import numpy as np


def parse_one(s): # store into lists
    # input: e.g.,
    #   s = '(NP (CP (IP (NP (DNP (NER-GPE (NR 台湾)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN 候选人) (NN 辩论会)) (PU ，))))'
    # output:
    #   # Chinese word segmentation
    #   seg = ['B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'M', 'E', 'B', 'M', 'E', 'S']
    #   # NER: current BIO, consider to change to BMEWO+
    #   neg = ['B-GPE', 'I-GPE', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    #   # full_pos: chunk information
    #   full_pos = ['(NP', '(CP', '(IP', '(NP', '(DNP', '(NR', ')NR', ')DNP', '(DEG', ')DEG', ')NP', '(NR', ')NR', ')IP', ')CP', '(VP', '(NT', ')NT', '(VV', ')VV', ')VP', ')NP', '(DEC', ')DEC', ')DEC', '(NP-m', '(NP', '(NR', ')NR', '(NN', ')NN', ')NP', '(NP-m', '(NP', '(NN', ')NN', '(NN', ')NN', ')NP', '(PU', ')PU', ')NP-m', ')NP-m', ')NP-m']
    #   # text
    #   text = '台湾的公视今天主办的台北市长候选人辩论会，'
    #   text_seg = '台湾 的 公视 今天 主办 的 台北 市长 候选人 辩论会 ，'
    #   # is_chinese_char
    #   False # may be determined by ，

    basic_tokenizer = BasicTokenizer(do_lower_case=True)
    s = re.sub('\)', ') ', s)
    s = re.sub(' +', ' ', s).strip()
    pos = s.split(' ')
    buffer = []
    innermost = True
    full_pos = []
    seg = []
    ner = []
    ner_types = []
    text = []

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
                    innermost = False
                    p = p[-1]

                    # Process multi-lingual
                    # if it is English word, it can be only one
                    # consider using BasicTokenizer(do_lower_case=do_lower_case) later
                    is_chinese_char = basic_tokenizer._is_chinese_char(ord(word[0]))

                    # cws format: BMES
                    # ner format: BIO
                    if is_chinese_char: # Chinese
                        if len(word) == 1:
                            seg_gt = ['S']

                            if ner_types != []:
                                ner_gt = ['B']
                            else:
                                ner_gt = ['O']
                        else:
                            seg_gt = ['B'] + ['M'] * (len(word) - 2) + ['E']
                            ner_gt = ['B'] + ['I'] * (len(word) - 1)
                    else: # non-chinese
                        seg_gt = ['S']
                        ner_gt = ['O']

                    if ner_types != []:
                        ner_type = ner_types.pop()
                        ner_gts = [_ + ner_type for _ in ner_gt]
                    else:
                        ner_gts = ['O' for _ in ner_gt]

                    seg.extend(seg_gt)
                    ner.extend(ner_gts)

            p += suffix
        full_pos.append(p)

    text_str = ''.join(text)
    text_seg = ' '.join(text)

    return seg, ner, full_pos, text_str, text_seg, is_chinese_char

def check_English_words(word, basic_tokenizer):
    for idx in range(len(word)):
        if not basic_tokenizer._is_english_char(ord(word[idx])):
            return False # one char is not English, it is not an English word
    return True

def output_seg_tokens(word, full_tokenizer, basic_tokenizer, ner_type=''):
    # only one word, e.g., director, 日子， 在
    # ner_type
    # cws format: BMES
    # ner format: BMEWO
    isEnglish = check_English_words(word, basic_tokenizer)

    if isEnglish:
        src_seg_gt = ['S']

        if len(ner_type)>0:
            src_ner_gt = ['W' + ner_type]
        else:
            src_ner_gt = ['O']

        wl = full_tokenizer.tokenize(word)
        len_wl = len(wl)

        if len_wl == 1:
            bert_seg_gt = src_seg_gt
            bert_ner_gt = src_ner_gt
        else:
            bert_seg_gt = ['B'] + ['M'] * (len_wl - 2) + ['E']

            if len(ner_type)>0:
                bert_ner_gt = ['B' + ner_type] + ['M' + ner_type] * (len_wl - 2) + ['E' + ner_type]
            else:
                bert_ner_gt = ['O'] * len_wl
    else:
        wl = full_tokenizer.tokenize(word)
        len_wl = len(wl)

        if len_wl == 1:
            src_seg_gt = ['S']

            if len(ner_type)>0:
                src_ner_gt = ['W' + ner_type]
            else:
                src_ner_gt = ['O']
        else:
            src_seg_gt = ['B'] + ['M'] * (len_wl - 2) + ['E']

            if len(ner_type)>0:
                src_ner_gt = ['B' + ner_type] + ['M' + ner_type] * (len_wl - 2) + ['E' + ner_type]
            else:
                src_ner_gt = ['O'] * len_wl

        bert_seg_gt = src_seg_gt # the same when there is no English
        bert_ner_gt = src_ner_gt # the same when there is no English

    return bert_ner_gt, bert_seg_gt, src_ner_gt, src_seg_gt, isEnglish

def parse_one2BERTformat(s, full_tokenizer, basic_tokenizer): # store into lists
    # Adopt BMEWO for NER tagging,
    #   see https://lingpipe-blog.com/2009/10/14/coding-chunkers-as-taggers-io-bio-bmewo-and-bmewo/
    #
    #  An example
    #   s = '(NP (CP (IP (NP (DNP (NER-GPE (NR Taiwan)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN candidate) (NN defence)) (PU ，))))'
    #
    #   src_seg, src_ner, full_pos, text_str, text_seg, bert_seg, bert_ner
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
    #   # bert_seg: special treatment for English word
    #   bert_seg = ['S', 'S', 'B', 'E', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'B', 'E', 'B', 'M', 'M', 'E', 'B', 'M', 'E', 'S']
    #   bert_ner = ['W-GPE', 'O', 'B-ORG', 'E-ORG', 'B', 'E', 'B', 'E', 'O', 'B', 'E', 'B', 'E', 'B', 'M', 'M', 'E', 'B', 'M', 'E', 'O']

    s = re.sub('\)', ') ', s)
    s = re.sub(' +', ' ', s).strip()
    pos = s.split(' ')
    buffer = []
    innermost = True
    full_pos = []
    src_seg = []  # src_seg for storing the segmentation of source words
    bert_seg = [] # bert_seg for storing the segmentation of bert format, additional processing for words and English
    src_ner = []  # src_ner for storing the ner segmentation of source words
    bert_ner = [] # bert_seg for storing the ner segmentation of bert format, additional processing for words and English
    ner_types = []
    text = []  # store the source words, English words and numbers are separated by space
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
                    innermost = False
                    p = p[-1]

                    #num_words += 1
                    #len_chars += len(word)
                    if ner_types != []:
                        ner_type = ner_types.pop()
                        #num_ner += 1
                    else:
                        ner_type = ''

                    # Process multi-lingual

                    #wls = word.split() this part is no needed
                    #if len(wls)==1: # no multiple English words
                    # English words are separated
                    bert_ner_gt, bert_seg_gt, src_ner_gt, src_seg_gt, isEnglish = output_seg_tokens(word, full_tokenizer, basic_tokenizer, ner_type)

                    if isEnglish:
                        lang_status_list.append('E')
                    else:
                        lang_status_list.append('O')

                    src_seg.extend(src_seg_gt)
                    src_ner.extend(src_ner_gt)
                    bert_seg.extend(bert_seg_gt)
                    bert_ner.extend(bert_ner_gt)

                    '''
                    else: # there are multiple English words
                        for ws in wls:
                            bert_ner_gt, bert_seg_gt, src_ner_gt, src_seg_gt = output_seg_tokens(ws, ner_type)

                            src_seg.extend(src_seg_gt)
                            src_ner.extend(src_ner_gt)
                            bert_seg.extend(bert_seg_gt)
                            bert_ner.extend(bert_ner_gt)
                    '''

            p += suffix
        full_pos.append(p)

    text_seg = ' '.join(text)
    # additional process the English issue
    text_str = ''
    for idx in range(len(lang_status_list)):
        if idx>0 and lang_status_list[idx-1]=='E' and lang_status_list[idx]=='E':
            text_str += ' ' + text[idx]
        else:
            text_str += text[idx]


    return src_seg, src_ner, full_pos, text_str, text_seg, bert_seg, bert_ner

'''
    The rest code has bug
                    # Process multi-lingual
                    # if it is English word, it can be only one
                    # consider using BasicTokenizer(do_lower_case=do_lower_case) later
                    is_chinese_char = basic_tokenizer._is_chinese_char(ord(word[0]))

                    if ner_types != []:
                        ner_type = ner_types.pop()
                        bNER = True
                        num_ner += 1
                    else:
                        bNER = False

                    # cws format: BMES
                    # ner format: BMEWO
                    if is_chinese_char: # Chinese
                        lang_status_list.append('C')
                        if len(word) == 1:
                            src_seg_gt = ['S']

                            if bNER:
                                src_ner_gt = ['W' + ner_type]
                            else:
                                src_ner_gt = ['O']
                        else:
                            src_seg_gt = ['B'] + ['M'] * (len(word) - 2) + ['E']

                            if bNER: #  ner_type exists
                                src_ner_gt = ['B' + ner_type] + ['M' + ner_type] * (len(word) - 2) + ['E' + ner_type]
                            else:
                                src_ner_gt = ['B'] + ['M'] * (len(word) - 2) + ['E']

                        bert_seg_gt = src_seg_gt # the same when the word is Chinese
                        bert_ner_gt = src_ner_gt # the same when the word is Chinese
                    else: # non-chinese
                        if basic_tokenizer._is_english_char(ord(word[0])):
                            lang_status_list.append('E')
                        else:
                            lang_status_list.append('O')

                        src_seg_gt = ['S'] # consider only one words
                        if bNER:
                            src_ner_gt = ['W' + ner_type]
                        else:
                            src_ner_gt = ['O']

                        wl = full_tokenizer.tokenize(word)
                        len_wl = len(wl)
                        if len_wl == 1:
                            bert_seg_gt = src_seg_gt
                            bert_ner_gt = src_ner_gt
                        else:
                            bert_seg_gt = ['B'] + ['M'] * (len_wl - 2) + ['E']

                            if bNER:
                                bert_ner_gt = ['B' + ner_type] + ['M' + ner_type] * (len_wl - 2) + ['E' + ner_type]
                            else:
                                bert_ner_gt = ['B'] + ['M'] * (len_wl - 2) + ['E']

                    src_seg.extend(src_seg_gt)
                    src_ner.extend(src_ner_gt)
                    bert_seg.extend(bert_seg_gt)
                    bert_ner.extend(bert_ner_gt)

            p += suffix
        full_pos.append(p)

    text_str = ''
    for idx in range(len(lang_status_list)):
        if idx>0 and lang_status_list[idx-1]=='E' and lang_status_list[idx]=='E':
            text_str += ' ' + text[idx]
        else:
            text_str += text[idx]

    text_seg = ' '.join(text)

    return src_seg, src_ner, full_pos, text_str, text_seg, bert_seg, bert_ner, num_ner
'''
def parse_one_2strs(s):
    #s = strQ2B(s) # no use
    seg, ner, full_pos, text_str, text_seg, is_chinese_char = parse_one(s)

    #if is_chinese_char:
    #    text = ' '.join(text)
    #else:
    #text = ''.join(text)

    seg = ' '.join(seg)
    ner = ' '.join(ner)
    full_pos = ' '.join(full_pos)
    return seg, ner, full_pos, text_str, text_seg


def parse_one_2dict(s):
    seg, ner, full_pos, text_str, text_seg, is_chinese_char = parse_one(s)

    #if is_chinese_char:
    #    text = ' '.join(text)
    #else:
    #text = ''.join(text)

    seg = ' '.join(seg)
    ner = ' '.join(ner)
    full_pos = ' '.join(full_pos)
    return {'seg': seg,  'ner': ner, 'full_pos': full_pos, 'text': text_str, 'text_seg': text_seg}

def parse_one2BERT2Dict(s, full_tokenizer, basic_tokenizer):
    src_seg, src_ner, full_pos, text, text_seg, bert_seg, bert_ner = parse_one2BERTformat(s, full_tokenizer, basic_tokenizer)

    src_seg = ','.join(src_seg) + ','
    src_ner = ','.join(src_ner) + ','
    full_pos = ' '.join(full_pos)
    bert_seg = ' '.join(bert_seg)
    bert_ner = ' '.join(bert_ner)
    return {'src_seg': src_seg,  'src_ner': src_ner, 'full_pos': full_pos, 'text': text, 'text_seg': text_seg,
            'bert_seg': bert_seg, 'bert_ner': bert_ner}

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

    print('Finish writing generated data!')

def genDataWithBERTSeg(in_file, out_dir, mode):
    print('Running genDataWithBERTSeg...')
    vocab_file = '../vocab/bert-base-chinese.txt'
    full_tokenizer = FullTokenizer(vocab_file, do_lower_case=True)
    basic_tokenizer = BasicTokenizer(do_lower_case=True)

    with open(in_file, 'r', encoding='utf8') as f:
        raw_data = f.readlines()

    #seg_all, ner_all, chunk_all, text_all = zip(*[parse_one_2str_list(s) for s in raw_data])
    #data_all = zip(*[parse_one_2dict(s) for s in raw_data])
    data_all = [parse_one2BERT2Dict(s, full_tokenizer, basic_tokenizer) for s in raw_data]

    import pandas as pd
    df = pd.DataFrame(data_all)
    # separate with \t
    df.to_csv(out_dir+mode+'.tsv', sep='\t', encoding='utf-8', index=False)

    print('Finish writing generated data!')

def parse_Ner(s, NerSet, wordLenList):
    # s: a tag list followed Ontonotes'format
    # NerSet: a dictionary to store the number of words in the corresponding type
    # wordLenList: a list to store the length of each word

    num_ner = 0
    s = re.sub('\)', ') ', s)
    s = re.sub(' +', ' ', s).strip()
    pos = s.split(' ')
    buffer = []
    innermost = True
    ner_types = []
    text = []

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
                    wordLenList.append(len(word))

                    text.append(word)
                    innermost = False
                    p = p[-1]

                    if ner_types != []:
                        ner_type = ner_types.pop()
                        ner_type = ner_type[1:]
                        if ner_type not in NerSet:
                            NerSet[ner_type] = 1
                        else:
                            NerSet[ner_type] += 1
                        num_ner += 1
    return num_ner

def countNer_Word(infile):
    with open(in_file, 'r', encoding='utf8') as f:
        raw_data = f.readlines()

    NerSet = {}

    wordLenList = []
    num_ners = [parseNerWord(s, NerSet, wordLenList) for s in raw_data]
    for v in NerSet:
        print(v+':' + str(NerSet[v]))

    count_ners = np.array(num_ners)

    return np.sum(count_ners).item(), NerSet, wordLenList


def gen_4ner_type():
    #out_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data/'
    out_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4ner_data/'
    os.makedirs(out_dir, exist_ok=True)

    in_file = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/5.fuse-tree2/test.fuse.parse'
    genDataWithBERTSeg(in_file, out_dir, 'test')

    in_file = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/5.fuse-tree2/train.fuse.parse'
    genDataWithBERTSeg(in_file, out_dir, 'train')

    in_file = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/5.fuse-tree2/dev.fuse.parse'
    genDataWithBERTSeg(in_file, out_dir, 'dev')


if __name__ == '__main__':
    ''' 
    s = '(NP (CP (IP (NP (DNP (NER-GPE (NR 台湾)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN 候选人) (NN 辩论会)) (PU ，))))'
    seg, ner, full_pos, text, text_seg, _ = parse_one(s)
    print(seg)
    print(ner)
    print(full_pos)
    print(text)
    print(text_seg)

    seg_str, ner_str, full_pos_str, text_str, text_seg = parse_one_2strs(s)
    print(seg_str)
    print(ner_str)
    print(full_pos_str)
    print(text_str)
    print(text_seg)

    out_dict = parse_one_2dict(s)
    print('seg:'+out_dict['seg'])
    print('ner:'+out_dict['ner'])
    print('full_pos:'+out_dict['full_pos'])
    print('text:'+out_dict['text'])
    print('text seg:'+out_dict['text_seg'])

    s2 = '(VA ＥＭＰＴＹ)'
    seg, ner, full_pos, text, text_seg, _ = parse_one(s2)
    print(seg)
    print(ner)
    print(full_pos)
    print(text)
    print(text_seg)

    seg_str, ner_str, full_pos_str, text_str, text_seg = parse_one_2strs(s2)
    print(seg_str)
    print(ner_str)
    print(full_pos_str)
    print(text_str)
    print(text_seg)
    '''
    vocab_file = '../vocab/bert-base-chinese.txt'
    full_tokenizer = FullTokenizer(vocab_file, do_lower_case=True)
    basic_tokenizer = BasicTokenizer(do_lower_case=True)

    s = '(NP (CP (IP (NP (DNP (NER-GPE (NR #students)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN candidate) (NN defence)) (PU ，))))'
    src_seg, src_ner, full_pos, text_str, text_seg, bert_seg, bert_ner = parse_one2BERTformat(s, full_tokenizer, basic_tokenizer)


    s = '(NP (CP (IP (NP (DNP (NER-GPE (NR Taiwan)) (DEG 的)) (NER-ORG (NR 公视))) (VP (NT 今天) (VV 主办))) (DEC 的)) (NP-m (NP (NR 台北) (NN 市长)) (NP-m (NP (NN candidate) (NN defence)) (PU ，))))'
    s = '(IP (NP (NP (NER-ORG (NR 新华社)) (NP-m (NER-GPE (NR 开罗)) (NP-m (NP (NT １１月) (NT ２４日)) (NN 电)))) (PRN (PRN-m (PU （) (NP (NN 记者) (NP (NER-PER (NR 郭春菊)) (NER-PER (NR 林建杨))))) (PU ）))) (IP-m (NP (NP (NP (NER-ORG (NR 阿拉伯) (NN 国家) (NN 联盟)) (PRN (PRN-m (PU （) (NER-ORG (NR 阿盟))) (PU ）))) (NN 秘书长)) (NER-PER (NR 穆萨))) (IP-m (VP (NT ２４日) (VP (VP-m (VP (VV 发表) (NN 声明)) (PU ，)) (VP (VV 谴责) (NP (CP (IP (NP (NP (NP (NER-GPE (NR 伊拉克)) (NN 首都)) (NER-GPE (NR 巴格达))) (NP-m (NN 东部) (NER-GPE (NR 萨德尔城)))) (VP (NT ２３日) (VV 发生))) (DEC 的)) (NP-m (JJ 连环) (NP-m (NP (NN 汽车) (NP-m (NN 炸弹) (NN 袭击))) (NN 事件))))))) (PU 。))))'
    src_seg, src_ner, full_pos, text_str, text_seg, bert_seg, bert_ner = parse_one2BERTformat(s, full_tokenizer, basic_tokenizer)
    print(s)
    print(src_seg)
    print(src_ner)
    print(full_pos)
    print(text_str)
    print(text_seg)
    print(bert_seg)
    print(bert_ner)

    out_dict = parse_one2BERT2Dict(s, full_tokenizer, basic_tokenizer)
    print('src_seg:'+out_dict['src_seg'])
    print('src_ner:'+out_dict['src_ner'])
    print('full_pos:'+out_dict['full_pos'])
    print('text:'+out_dict['text'])
    print('text_seg:'+out_dict['text_seg'])
    print('bert_seg:'+out_dict['bert_seg'])
    print('bert_ner:'+out_dict['bert_ner'])

    TEST_FLAG = True
    if TEST_FLAG:
        out_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4ner_data/valid/'
        in_file = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4ner_data/valid/test_data_ori.txt'
        genDataWithBERTSeg(in_file, out_dir, 'data_proc')


    else:
        gen_4ner_type()

    COUNT_FLAG = False
    if COUNT_FLAG:
        print('test:')
        in_file = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/5.fuse-tree2/test.fuse.parse'
        num_ner, NerSet = countNer_Word(in_file)
        print('test:' + str(num_ner))

        print('train:')
        in_file = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/5.fuse-tree2/train.fuse.parse'
        num_ner, NerSet = countNer_Word(in_file)
        print('train:' + str(num_ner))

        print('dev:')
        in_file = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/5.fuse-tree2/dev.fuse.parse'
        num_ner, NerSet = countNer_Word(in_file)
        print('dev:' + str(num_ner))
