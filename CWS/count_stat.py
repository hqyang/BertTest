#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 10:09 2019-04-24 
@author: haiqinyang

Feature: 

Scenario: 
"""

import os
import numpy as np

import sys
sys.path.append('../src')

from src.utilis import count_words_in_part, count_data_stat_in_part, savestat2file, savewords
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def count_stat_data(info_all):
    # input: sentence list

    parts = ['train', 'test']

    data_count = {('test', 'num_words_in_sent'): [], ('train', 'num_words_in_sent'): [],
                   ('test', 'num_chars_in_word'): [], ('train', 'num_chars_in_word'): []}

    data_stat = {
                 ('test', 'total_eng_words'): 0, ('train', 'total_eng_words'): 0,
                 ('test', 'total_chi_words'): 0, ('train', 'total_chi_words'): 0,
                 ('test', 'total_dig_words'): 0, ('train', 'total_dig_words'): 0,
                 ('test', 'total_eng_chars'): 0, ('train', 'total_eng_chars'): 0,
                 ('test', 'total_chi_chars'): 0, ('train', 'total_chi_chars'): 0,
                 ('test', 'total_dig_chars'): 0, ('train', 'total_dig_chars'): 0,
                 ('test', 'great126'): 0, ('train', 'great126'): 0,
                 ('test', 'great62'): 0, ('train', 'great62'): 0,
                 ('test', 'great30'): 0, ('train', 'great30'): 0,
                 ('test', 'less30'): 0, ('train', 'less30'): 0,
        # The following stat is computed by len(store_dicts(x, x))
        # ('test', 'unique_eng_words'): 0, ('train', 'unique_eng_words'): 0,
        # ('test', 'unique_chi_words'): 0, ('train', 'unique_chi_words'): 0,
        # ('test', 'unique_dig_words'): 0, ('train', 'unique_dig_words'): 0,

        # This stat is computed via len(store_chi_chars(x))
        # ('test', 'unique_chi_chars'): 0, ('train', 'unique_chi_chars'): 0,
    }

    # store_dict: store the unique English words, Chinese words, and numeric
    store_dicts = {('test', 'eng'): set(), ('train', 'eng'): set(),
                   ('test', 'chi'): set(), ('train', 'chi'): set(),
                   ('test', 'dig'): set(), ('train', 'dig'): set(),
                   }

    # store_chars: store the unique English words, Chinese chars, and numeric chars
    store_chi_chars = {'test': set(), 'train': set()}

    for part in parts:
        data_stat[part, 'line'] = len(info_all[part, 'sent'])

        for line in tqdm(info_all[part, 'sent']):
            words = line.split()

            count_words_in_part(words, part, data_stat, data_count, store_dicts, store_chi_chars)

        count_data_stat_in_part(data_count, part, store_dicts, store_chi_chars, data_stat)

    categories = ['chi', 'dig', 'eng']

    set_diff_test = set()

    for cat_gor in categories:
        set_diff_test = set_diff_test.union(store_dicts['test', cat_gor] - store_dicts['train', cat_gor])

    data_stat['oov_test'] = len(set_diff_test)
    data_stat['ratio_oov_test'] = len(set_diff_test) * 1. / data_stat['train', 'unique_words']

    #set_diff_char_test = store_chars['test'] - store_chars['train']
    #data_stat['oov_char_test'] = len(set_diff_char_test)

    return data_stat, store_chi_chars, store_dicts


def list_and_stat_docs(data_dir, out_dir):
    datasets = ['as', 'cityu', 'msr', 'pku']
    parts = ['train', 'test']

    for dataset in datasets:
        out_file = os.path.join(out_dir, dataset+'_data_stat.txt')

        info_all = {('test', 'sent'): [], ('train', 'sent'): []}

        for part in parts:
            in_file = os.path.join(data_dir, dataset+'_'+part+'.tsv')

            with open(in_file, 'r', encoding='utf8') as f:
                info_all[part, 'sent'] = [_.strip() for _ in f.readlines()]

        #with Pool(cpu_count()) as p:
        #data_stat = p.start_new_thread(count_stat_data, info_all)

        data_stat, store_chi_chars, store_dicts = count_stat_data(info_all)
        print(data_stat)

        savestat2file(data_stat, out_file, parts)
        savewords(store_dicts, out_dir, parts, dataset)
# end listdocs


if __name__ == '__main__':
    PRE_DIR = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/'
    DATA_DIR = PRE_DIR + 'cws/'
    OUTPUT_DIR = PRE_DIR + 'data_stat/cws/'

    list_and_stat_docs(DATA_DIR, OUTPUT_DIR)
