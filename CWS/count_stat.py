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
from src.utilis import check_english_words
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def savestat2file(data_stat, out_file):
    parts = ['train', 'test']

    with open(out_file, 'a', encoding='utf-8') as f:
        f.writelines('Total: num_lines: {:d}\n'.format(data_stat['train', 'line']+data_stat['test', 'line']))

    for part in parts:
        with open(out_file, 'a', encoding='utf-8') as f:
            f.writelines('Type: {:s}, num_lines: {:d}\n'.format(part, data_stat[part, 'line']))
            f.writelines('Unique tokens: {:d}, Total tokens: {:d}, num_chi: {:d}, num_eng: {:d}\n'.format(
                data_stat[part, 'unique_tokens'], data_stat[part, 'total_tokens'], data_stat[part, 'num_chi'],
                data_stat[part, 'num_eng']))
            f.writelines('Per sent.: max. tokens: {:d}, min. tokens: {:d}, mean tokens: {:.2f}\n'.format(
                data_stat[part, 'max_tokens_per_sent'], data_stat[part, 'min_tokens_per_sent'],
                data_stat[part, 'mean_tokens_per_sent']))

            f.writelines('Per token.: max. len: {:d}, min. len: {:d}, mean len: {:.3f}\n'.format(
                data_stat[part, 'max_len_per_token'], data_stat[part, 'min_len_per_token'],
                data_stat[part, 'mean_len_per_token']))

            f.writelines('num (>126): {:d}, num (126<>62): {:d}, num (62<>30): {:d}, num (<30): {:d}\n'.format(
                data_stat[part, 'great126'], data_stat[part, 'great62'],
                data_stat[part, 'great30'], data_stat[part, 'less30']))

            f.writelines('\n')

    with open(out_file, 'a', encoding='utf-8') as f:
        f.writelines('num_oov_test: {:d}, ratio_oov_test: {:.3f}'.format(
            data_stat['oov_test'], data_stat['ratio_oov_test']))


def count_stat_data(info_all):
    # input: sentence list

    parts = ['train', 'test']

    data_count = {('test', 'num_words'): [], ('train', 'num_words'): [],
                   ('test', 'len_words'): [], ('train', 'len_words'): []}

    data_stat = {('test', 'num_eng'): 0, ('train', 'num_eng'): 0,
                 ('test', 'num_chi'): 0, ('train', 'num_chi'): 0,
                 ('test', 'great126'): 0, ('train', 'great126'): 0,
                 ('test', 'great62'): 0, ('train', 'great62'): 0,
                 ('test', 'great30'): 0, ('train', 'great30'): 0,
                 ('test', 'less30'): 0, ('train', 'less30'): 0
                 }

    store_dicts = {'test': set(), 'train': set()}

    for part in parts:
        data_stat[part, 'line'] = len(info_all[part, 'sent'])

        for line in tqdm(info_all[part, 'sent']):
            words = line.split()

            data_count[part, 'num_words'].append(len(words))

            if len(words)>=126:
                data_stat[part, 'great126'] += 1
            elif len(words)>=62:
                data_stat[part, 'great62'] += 1
            elif len(words)>=30:
                data_stat[part, 'great30'] += 1
            else:
                data_stat[part, 'less30'] += 1

            for word in words:
                store_dicts[part].add(word)

                if check_english_words(word):
                    data_count[part, 'len_words'].append(1)
                    data_stat[part, 'num_eng'] += 1
                else: # non-English word
                    data_count[part, 'len_words'].append(len(word))
                    data_stat[part, 'num_chi'] += 1

        np_sents = np.array(data_count[part, 'num_words'])
        data_stat[part, 'total_tokens'] = np_sents.sum()
        data_stat[part, 'max_tokens_per_sent'] = np_sents.max()
        data_stat[part, 'min_tokens_per_sent'] = np_sents.min()
        data_stat[part, 'mean_tokens_per_sent'] = np_sents.mean()

        np_words = np.array(data_count[part, 'len_words'])
        data_stat[part, 'max_len_per_token'] = np_words.max()
        data_stat[part, 'min_len_per_token'] = np_words.min()
        data_stat[part, 'mean_len_per_token'] = np_words.mean()

        data_stat[part, 'unique_tokens'] = len(store_dicts[part])

    set_diff_test = store_dicts['test'] - store_dicts['train']
    data_stat['oov_test'] = len(set_diff_test)
    data_stat['ratio_oov_test'] = len(set_diff_test) * 1. / len(store_dicts['train'])

    return data_stat


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

        data_stat = count_stat_data(info_all)
        print(data_stat)

        savestat2file(data_stat, out_file)
# end listdocs


if __name__ == '__main__':
    PRE_DIR = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/'
    DATA_DIR = PRE_DIR + 'cws/'
    OUTPUT_DIR = PRE_DIR + 'data_stat/cws/'

    list_and_stat_docs(DATA_DIR, OUTPUT_DIR)
