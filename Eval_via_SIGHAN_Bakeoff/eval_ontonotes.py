#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 11:45 2019-04-25 
@author: haiqinyang

Feature: 

Scenario: 
"""
import sys
sys.path.append('../src')

import os
from utilis import get_Ontonotes
import pandas as pd
from collections import Counter

def save_proc_data(df, output_file, output_seg_file, output_word_file=None):
    df_text = pd.DataFrame(df.text)
    # separate with \t
    df_text.to_csv(output_file, sep='\t', encoding='utf-8', index=False, header=False)

    df_seg_text = pd.DataFrame(df.text_seg)
    df_seg_text.to_csv(output_seg_file, sep='\t', encoding='utf-8', index=False, header=False)

    if output_word_file is not None:
        aa = ''
        for x in df.text_seg:
            aa += x + ' '

        dc = dict(Counter(aa.split()))

        with open(output_word_file, 'w', encoding='utf8') as f:
            for word in dc:
                f.writelines(word + '\n')


def proc_ontonotes_for_prediction(data_dir, output_dir):
    parts = ['dev', 'train', 'test']

    for part in parts:
        output_seg_text_file = os.path.join(output_dir, 'ontonotes_'+part+'_seg.txt') # seg data
        output_text_file = os.path.join(output_dir, 'ontonotes_'+part+'.txt') #  data in text

        output_word_file = None

        if part == 'train':
            output_word_file = os.path.join(output_dir, 'ontonotes_'+part+'_words.txt')

        df = get_Ontonotes(data_dir, type=part)

        save_proc_data(df, output_text_file, output_seg_text_file, output_word_file)

        print('Finish processing ' + part + ' file!')


def output_sighan_bakeoff_rs(data_dir, pred_dir):
    parts = ['dev', 'train', 'test']

    dict_file = os.path.join(data_dir, 'ontonotes_train_words.txt')

    for part in parts:
        true_seg_text = os.path.join(data_dir, 'ontonotes_'+part+'_seg.txt')
        pred_seg_text = os.path.join(pred_dir, 'ontonotes_'+part+'.txt') # seg data
        outscore_text = os.path.join(pred_dir, 'ontonotes_'+part+'_score.txt')

        print('call ./scripts/score '+ part)
        os.system('./scripts/score ' + dict_file + ' ' + true_seg_text + ' ' + pred_seg_text + ' > ' + outscore_text)



bLocal = True
bLocal = False

SAVE_DATA = True
SAVE_DATA = False

SIGHAN_BAKEOFF = True

if __name__=='__main__':
    if SAVE_DATA:
        if bLocal:
            data_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4ner_data/'
        else:
            data_dir = '../../data/ontonotes5/4ner_data/'

            output_dir = data_dir + 'eval_data/'

        os.makedirs(output_dir, exist_ok=True)
        proc_ontonotes_for_prediction(data_dir, output_dir)

    if SIGHAN_BAKEOFF:
        if bLocal:
            data_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4ner_data/eval_data'
            pred_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/eval/2019_3_22/pred/'
        else:
            data_dir = '../../data/ontonotes5/4ner_data/eval_data/'
            pred_dir = '../tmp_2019_3_22/out/ontonotes_eval/'

        output_sighan_bakeoff_rs(data_dir, pred_dir)
