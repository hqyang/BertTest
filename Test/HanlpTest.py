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
from src.pkuseg.metrics import getFscoreFromBIOTagList
import re
from src.utilis import get_Ontonotes, convertList2BIOwithComma, BMES2BIO, space2Comma
import pdb
from tqdm import tqdm


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
    df = get_Ontonotes(data_dir, type)

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

        if i % 20000 == 0:
            print('{:d}: '.format(i))
            print(sentence)
            print(data.text_seg)
            print(hanlp_seg)
            print(tl)
            print(hanlp_BIO)
            print('\n')

        with open(output_diff_file, "a+") as writer:
            writer.write('{:d}: '.format(i))
            writer.write(sentence+'\n')
            writer.write(data.text_seg+'\n')
            writer.write(hanlp_seg+'\n')
            writer.write(tl+'\n')
            writer.write(hanlp_BIO+'\n\n')

    score, scoreInfo = getFscoreFromBIOTagList(trueLabelList, hanlpList)

    print('Eval ' + type + ' results:')
    print('Test F1, Precision, Recall, Acc, No. Tags: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:d}'.format(score[0], \
                                                  score[1], score[2], score[3], scoreInfo[-1]))

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        writer.write('Eval ' + type + ' results: ')
        writer.write("F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, No. Tags: {:d}\n\n".format(score[0], \
                                                score[1], score[2], score[3], scoreInfo[-1]))

    return score

def do_eval_with_file(infile, output_dir, otag, tagMode='BMES'):
    # infile: input file in tsv format
    # output_dir: the directory to store evaluation file
    # otag: to denote what type of file should be stored
    # tagMode: to indicate the label coding is 'BIO' or 'BMES'

    df = pd.read_csv(infile, sep='\t')

    hanlpList = []
    trueLabelList = []

    output_diff_file = os.path.join(output_dir, otag+"_diff.txt")

    with open(output_diff_file, "a+") as writer:
        writer.write('order: resource, true, jieba\n')

    for i, data in tqdm(enumerate(df.itertuples())):
        sentence = data.text
        #rs_full = jieba.lcut(sentence, cut_all=True) # Full mode, all possible cuts
        #rs_ser = jieba.lcut_for_search(sentence) # search engine mode, similar to Full mode

        # sentence = '台湾的公视今天主办的台北市长候选人辩论会，'
        # rs_precision = jieba.lcut(sentence, cut_all=False)
        #   rs_precision = ['台湾', '的', '公视', '今天', '主办', '的', '台北', '市长', '候选人', '辩论会', '，']
        # jieba_rs = ' '.join(rs_precision)
        #   jieba_rs = '台湾 的 公视 今天 主办 的 台北 市长 候选人 辩论会 ，'
        if tagMode=='BIO':
            tl = data.src_seg
        elif tagMode=='BMES':
            tl = BMES2BIO(data.src_seg)
            tl = space2Comma(tl)

        trueLabelList.append(tl)
        hanlp_rs = HanLP.segment(sentence)
        hanlp_text, hanlp_seg, hanlp_pos = proc_HanLP_rs(hanlp_rs)

        #str_precision = convertList2BMES(rs_precision)
        hanlp_BIO = convertList2BIOwithComma(hanlp_text)
        hanlpList.append(hanlp_BIO)

        if i % 20000 == 0:
            print('{:d}: '.format(i))
            print(sentence)
            print(tl)
            print(hanlp_BIO)
            print('\n')

        with open(output_diff_file, "a+") as writer:
            writer.write('{:d}: '.format(i))
            writer.write(sentence+'\n')
            writer.write(data.text_seg+'\n')
            writer.write(hanlp_seg+'\n')
            writer.write(tl+'\n')
            writer.write(hanlp_BIO+'\n\n')

    score, sInfo = getFscoreFromBIOTagList(trueLabelList, hanlpList)

    print('Eval ' + otag + ' results:')
    print("F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, Token: {:d}\n\n".format(score[0], \
                                              score[1], score[2], score[3], sInfo[-1]))

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "a+") as writer:
        writer.write('Eval ' + otag + ' results: ')
        writer.write("F1: {:.3f}, P: {:.3f}, R: {:.3f}, Acc: {:.3f}, Token: {:d}\n\n".format(score[0], \
                                             score[1], score[2], score[3], sInfo[-1]))

    return score

def test_ontonotes():
    data_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data'

    output_dir='./tmp/ontonotes/hanlp/'
    os.makedirs(output_dir, exist_ok=True)

    type = 'test'
    do_eval(data_dir, type, output_dir)

    type = 'dev'
    do_eval(data_dir, type, output_dir)

    type = 'train'
    do_eval(data_dir, type, output_dir)

def test_CWS():
    fnames = ['as', 'cityu', 'msr', 'pku']
    modes = ['train', 'test']
    tagMode = 'BIO'
    data_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/cws/'
    data_dir += tagMode + '/'

    output_dir='./tmp/cws/hanlp/'
    os.makedirs(output_dir, exist_ok=True)

    for wt in fnames:
        for md in modes:
            infile = data_dir + wt + '_' + md + '.tsv'
            otag = wt + '_' + md
            do_eval_with_file(infile, output_dir, otag, tagMode)

if __name__=='__main__':
    #test_ontonotes()
    test_CWS()
