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


SAVE_DATA = True
bLocal = True
bLocal = False

if __name__=='__main__':
    if SAVE_DATA:
        if bLocal:
            data_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4ner_data/'
        else:
            data_dir = '../../data/ontonotes5/4ner_data/'

        output_dir = data_dir + 'eval_data/'
        os.makedirs(output_dir, exist_ok=True)
        proc_ontonotes_for_prediction(data_dir, output_dir)
