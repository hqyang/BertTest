#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 11:48 2019-02-11 
@author: haiqinyang

Feature: 

Scenario: 
"""
import pandas as pd
from tokenization import FullTokenizer
import numpy as np
from tqdm import tqdm

def count_stat(infile, vocab_file):
    tokenizer = FullTokenizer(
                vocab_file=vocab_file, do_lower_case=True)

    df = pd.read_csv(infile, sep='\t')

    len_words = []
    #words_pos = [0]
    for i, data in tqdm(enumerate(df.itertuples())):
        texts = data.text.split('，')
        #words_pos.append(np.sum(words_pos)+len(texts))

        for text in texts:
            word = tokenizer.tokenize(text)
            len_words.append(len(word))

    print('min:', np.min(len_words), '; max:', np.max(len_words), '; avg: ', np.mean(len_words), '; median:', np.median(len_words))

    '''
    argmin = np.argmin(len_words)
    argmax = np.argmax(len_words)
    print('argmin:', argmin, 'argmax:', argmax)

    print('min:', df.loc[argmin, ['text']])
    print('max:', df.loc[argmax, ['text']])
    '''

    return len_words


if __name__=='__main__':
    infile = "/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data/train.tsv"
    vocab_file = '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/vocab.txt'

    print('\ntraining set')
    len_train_words = count_stat(infile, vocab_file)
    # min: 0 ; max: 205 ; avg:  13.002189806802528 ; median: 11.0

    print('\ndev set')
    infile = "/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data/dev.tsv"
    len_train_words = count_stat(infile, vocab_file)
    # min: 1 ; max: 264 ; avg:  14.431172559685706 ; median: 13.0

    print('\ntest set')
    infile = "/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data/test.tsv"
    len_train_words = count_stat(infile, vocab_file)
    # min: 1 ; max: 148 ; avg:  14.359896618565582 ; median: 13.0
