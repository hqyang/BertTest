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
import jieba
from src.basics import _is_chinese_char
from src.pkuseg.metrics import getFscoreFromBIOTagList

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

def BMES2BIO(text):
    sOut = text
    sOut = sOut.replace('M', 'I')
    sOut = sOut.replace('E', 'I')
    sOut = sOut.replace('S', 'O')

    return sOut

def do_eval(data_dir, type, output_dir):
    df = get_examples(data_dir, type)

    jiebaList = []
    trueLabelList = []

    output_diff_file = os.path.join(output_dir, type+"_diff.txt")

    for i, data in enumerate(df.itertuples()):
        sentence = data.text
        #rs_full = jieba.lcut(sentence, cut_all=True) # Full mode, all possible cuts
        #rs_ser = jieba.lcut_for_search(sentence) # search engine mode, similar to Full mode

        rs_precision = jieba.lcut(sentence, cut_all=False)
        #str_precision = convertList2BMES(rs_precision)
        str_BIO = convertList2BIO(rs_precision)
        jiebaList.append(str_BIO)

        tl = BMES2BIO(data.label)
        trueLabelList.append(tl)

        print('{:d}: '.format(i))
        print(sentence)
        print(tl)
        print(str_BIO)
        print('\n')

        with open(output_diff_file, "a+") as writer:
            writer.write('{:d}: '.format(i))
            writer.write(sentence)
            writer.write(tl)
            writer.write(str_BIO)
            writer.write('\n')

    score, _ = getFscoreFromBIOTagList(trueLabelList, jiebaList)
    print('Test F1, Precision, Recall: {:+.2f}, {:+.2f}, {:+.2f}'.format(score[0], score[1], score[2]))

    output_eval_file = os.path.join(output_dir, type+"_eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        writer.write("F1: {:.3f}, P: {:.3f}, R: {:.3f}\n".format(score[0], score[1], score[2]))

    return score

def main():
    data_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data'
    type = 'test'

    output_dir='./tmp/ontonotes/jieba/'
    os.makedirs(output_dir, exist_ok=True)
    score = do_eval(data_dir, type, output_dir)


if __name__=='__main__':
    main()
