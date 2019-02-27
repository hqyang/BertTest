#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 09:28 2019-02-21 
@author: haiqinyang

Feature: 

Scenario: 
"""
import os
import pandas as pd
from pyhanlp import *
from src.basics import _is_chinese_char
from src.pkuseg.metrics import getFscoreFromBIOTagList
import re

import pdb

def get_examples(data_dir, type='train'):
    """See base class."""
    df = pd.read_csv(os.path.join(data_dir, type+".tsv"), sep='\t')

    # full_pos (chunk), ner, seg, text
    # need parameter inplace=True
    df.drop(columns=['full_pos', 'ner'], inplace=True)

    # change name to tag for consistently processing
    df.rename(columns={'seg': 'label'}, inplace=True)

    return df

def convertList2BMES(rs):
    # rs: a list
    outStr = ''
    for i, word in enumerate(rs.__iter__()):
        if not _is_chinese_char(ord(word[0])) or len(word)==1:
            seg_gt = 'S '
        else: # Chinese char and multiple words
            seg_gt = 'B ' + 'M ' * (len(word) - 2) + 'E '

        outStr += seg_gt

        if i==len(rs)-1: # remove the additional space
            outStr = outStr[:-1]

    return outStr

def convertList2BIO(rs):
    # rs: a list
    outStr = ''
    for i, word in enumerate(rs.__iter__()):
        if not _is_chinese_char(ord(word[0])) or len(word)==1:
            seg_gt = 'O '
        else: # Chinese char and multiple words
            seg_gt = 'B ' + 'I ' * (len(word) - 1)

        outStr += seg_gt

        if i==len(rs)-1: # remove the additional space
            outStr = outStr[:-1]

    return outStr

def convertList2BIOwithComma(rs):
    # rs: a list
    outStr = ''
    for i, word in enumerate(rs.__iter__()):
        if not _is_chinese_char(ord(word[0])) or len(word)==1:
            seg_gt = 'O,'
        else: # Chinese char and multiple words
            seg_gt = 'B,' + 'I,' * (len(word) - 1)

        outStr += seg_gt

        if i==len(rs)-1: # remove the additional space
            outStr = outStr[:-1]

    return outStr

def BMES2BIO(text):
    # text = 'B E B M M E S'
    # sOut = BMES2BIO(text) # 'B I B I I I O'
    sOut = text
    sOut = sOut.replace('M', 'I')
    sOut = sOut.replace('E', 'I')
    sOut = sOut.replace('S', 'O')

    return sOut

def space2Comma(text):
    sOut = text
    sOut = sOut.replace(' ', ',')
    if sOut[-1]!=',':
        sOut += ','

    return sOut

def proc_HanLP_rs(text):
    # text = [台湾/ns, 的/ude1, 公/ng, 视/vg, 今天/t, 主办/v, 的/ude1, 台北/ns, 市长/nnt, 候选人/nnt, 辩论会/n, ，/w]
    # Output
    #   sText = ['台湾', '的', '公', '视', '今天', '主办', '的', '台北', '市长', '候选人', '辩论会', '，']
    #   sText_seg = '台湾 的 公 视 今天 主办 的 台北 市长 候选人 辩论会 ， '
    #   sText_pos =

    sText = []
    sText_seg = ''
    sText_pos = ''

    for i, word in enumerate(text.__iter__()):
        # need additional process of '/'
        ss = str(word)
        fa_idx = re.findall('/', ss)
        max_f = len(fa_idx)
        word = ss.replace('/', '[SLASH]', max_f-1)

        tt = word.split('/')

        if '[SLASH]' in tt[0]: # restore [SLASH] to /
            tt[0] = tt[0].replace('[SLASH]', '/')

        sText.append(tt[0])

        sText_seg += tt[0]
        sText_pos += tt[1]
        if i<len(text)-1:
            sText_seg += ' '
            sText_pos += ' '

    return sText, sText_seg, sText_pos


def do_eval(data_dir, type, output_dir):
    df = get_examples(data_dir, type)

    hanlpList = []
    trueLabelList = []

    output_diff_file = os.path.join(output_dir, type+"_diff.txt")

    for i, data in enumerate(df.itertuples()):
        sentence = data.text
        #rs_full = jieba.lcut(sentence, cut_all=True) # Full mode, all possible cuts
        #rs_ser = jieba.lcut_for_search(sentence) # search engine mode, similar to Full mode

        # sentence = '台湾的公视今天主办的台北市长候选人辩论会，'
        # hanlp_rs = HanLP.segment(sentence)
        #   rs_precision = ['台湾', '的', '公视', '今天', '主办', '的', '台北', '市长', '候选人', '辩论会', '，']
        # hanlp_rs = ' '.join(rs_precision)
        #   hanlp_rs = '台湾 的 公视 今天 主办 的 台北 市长 候选人 辩论会 ，'
        #if i==684:
        #    pdb.set_trace()

        hanlp_rs = HanLP.segment(sentence)
        hanlp_text, hanlp_seg, hanlp_pos = proc_HanLP_rs(hanlp_rs)

        #str_precision = convertList2BMES(rs_precision)
        hanlp_BIO = convertList2BIOwithComma(hanlp_text)
        hanlpList.append(hanlp_BIO)

        tl = BMES2BIO(data.label)
        tl = space2Comma(tl)
        trueLabelList.append(tl)

        print('{:d}: '.format(i))
        print(sentence)
        #print(tl)
        #print(str_BIO)
        #print('\n')

        with open(output_diff_file, "a+") as writer:
            writer.write('{:d}: '.format(i))
            writer.write(sentence+'\n')
            writer.write(data.text_seg+'\n')
            writer.write(hanlp_seg+'\n')
            writer.write(tl+'\n')
            writer.write(hanlp_BIO+'\n\n')

    score, _ = getFscoreFromBIOTagList(trueLabelList, hanlpList)

    print('Eval ' + type + ' results:')
    print('Test F1, Precision, Recall: {:+.2f}, {:+.2f}, {:+.2f}'.format(score[0], score[1], score[2]))

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        writer.write('Eval ' + type + ' results: ')
        writer.write("F1: {:.3f}, P: {:.3f}, R: {:.3f}\n\n".format(score[0], score[1], score[2]))

    return score

def main():
    data_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data'

    output_dir='./tmp/ontonotes/hanlp/'
    os.makedirs(output_dir, exist_ok=True)

    type = 'train'
    do_eval(data_dir, type, output_dir)

    type = 'test'
    do_eval(data_dir, type, output_dir)

    type = 'dev'
    do_eval(data_dir, type, output_dir)

if __name__=='__main__':
    main()
