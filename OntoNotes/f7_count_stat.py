#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 11:48 2019-02-11 
@author: haiqinyang

Feature: 

Scenario: 
"""
import pandas as pd
from src.tokenization import FullTokenizer
import numpy as np
from tqdm import tqdm

#import matplotlib.pyplot as plt
import numpy as np

from itertools import chain
from collections import Counter, OrderedDict, namedtuple

#from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
import random

Sentence = namedtuple("Sentence", "words tags")


def read_data(filename):
    """Read tagged sentence data"""
    with open(filename, 'r') as f:
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
    return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                        for l in s[1:]]))) for s in sentence_lines if s[0]))


def read_tags(filename):
    """Read a list of word tag classes"""
    with open(filename, 'r') as f:
        tags = f.read().split("\n")
    return frozenset(tags)


class Subset(namedtuple("BaseSet", "sentences keys vocab X tagset Y N stream")):
    def __new__(cls, sentences, keys):
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        wordset = frozenset(chain(*word_sequences))
        tagset = frozenset(chain(*tag_sequences))
        N = sum(1 for _ in chain(*(sentences[k].words for k in keys)))
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))

        return super().__new__(cls, {k: sentences[k] for k in keys}, keys, wordset, word_sequences,
                               tagset, tag_sequences, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)


    def __iter__(self):
        return iter(self.sentences.items())


class Dataset(namedtuple("_Dataset", "sentences keys vocab X tagset Y training_set testing_set N stream")):
    def __new__(cls, tagfile, datafile, train_test_split=0.8, seed=112890):
        tagset = read_tags(tagfile)
        sentences = read_data(datafile)
        keys = tuple(sentences.keys())
        wordset = frozenset(chain(*[s.words for s in sentences.values()]))
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        N = sum(1 for _ in chain(*(s.words for s in sentences.values())))

        # split data into train/test sets
        _keys = list(keys)
        if seed is not None: random.seed(seed)
        random.shuffle(_keys)
        split = int(train_test_split * len(_keys))
        training_data = Subset(sentences, _keys[:split])
        testing_data = Subset(sentences, _keys[split:])
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, dict(sentences), keys, wordset, word_sequences, tagset,
                               tag_sequences, training_data, testing_data, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())


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
    if 0:
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

    infile = "/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/4ner_data/train.tsv"
    vocab_file = '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/vocab.txt'
    data = Dataset(vocab_file, infile, train_test_split=1)

    print('Train set of Ontonotes')
    print("There are {} sentences in the corpus.".format(len(data)))
    print("There are {} sentences in the training set.".format(len(data.training_set)))
    print("There are {} sentences in the testing set.".format(len(data.testing_set)))
