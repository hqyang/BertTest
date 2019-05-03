#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 11:48 2019-02-11 
@author: haiqinyang

Feature: 

Scenario: 
"""
from tqdm import tqdm
from itertools import chain
from collections import Counter, OrderedDict, namedtuple, UserString
import random
import re
import os
import numpy as np

import sys
sys.path.append('../src')
from src.utilis import check_english_words

Sentence = namedtuple("Sentence", "words tags")

def split_filename(filename):
    #print(filename)
    name, ext = os.path.splitext(filename)
    num = name.split('_')[-1]
    word = '_'.join(name.split('_')[:-1])
    return word, int(num), ext


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


def read_name_lines(file):
    lines = open(file, encoding='utf-8').readlines()
    return lines[1:-1]


def read_parse_lines(file):
    content = open(file, encoding='utf-8').read()
    trees = [t for t in content.split('\n\n') if len(t) > 0]
    trees = [t.replace('\n', ' ') for t in trees]
    trees = [re.sub(' {2,}', ' ', t) + '\n' for t in trees]
    return trees


def which_part(domain, source, filename):
    word, num, ext = split_filename(filename)
    if domain in ['mz', 'nw'] and source in ['sinorama', 'xinhua']:
        if word == 'chtb' and (1 <= num <= 325 or 1001 <= num <= 1078):
            if num % 2 == 0:
                return 'test'
            else:
                return 'dev'
        else:
            assert False
    else:
        return 'train'


def savestat2file(stat, data_stat, out_file):
    parts = ['train', 'dev', 'test']

    with open(out_file, 'w', encoding='utf-8') as f:
        f.writelines('Total: num_docs: {:d}, num_lines: {:d}\n\n'.format(
            stat['train', 'file']+stat['dev', 'file']+stat['test', 'file'],
            stat['train', 'line']+stat['dev', 'line']+stat['test', 'line']))

    for part in parts:
        with open(out_file, 'a', encoding='utf-8') as f:
            f.writelines('Type: {:s}\nnum_docs: {:d}, num_lines: {:d}, avg. lines per doc: {:.3f}\n'.format(part,
                                stat[part, 'file'], stat[part, 'line'], stat[part, 'line']*1./stat[part, 'file']))
            f.writelines('unique words: {:d}, Total words: {:d}, num_chi: {:d}, num_eng: {:d}\n'.format(
                data_stat[part, 'unique_words'], data_stat[part, 'total_words'], data_stat[part, 'num_chi'],
                data_stat[part, 'num_eng']))
            f.writelines('per sent.: max. words: {:d}, min. words: {:d}, mean words: {:.3f}\n'.format(
                data_stat[part, 'max_words_per_sent'], data_stat[part, 'min_words_per_sent'], data_stat[part, 'mean_words_per_sent']))

            f.writelines('per word: max. len: {:d}, min. len: {:d}, mean len: {:.3f}\n'.format(
                data_stat[part, 'max_len_per_word'], data_stat[part, 'min_len_per_word'], data_stat[part, 'mean_len_per_word']))

            f.writelines('unique character: {:d}\n'.format(data_stat[part, 'unique_chars']))

            f.writelines('\n')

    with open(out_file, 'a', encoding='utf-8') as f:
        f.writelines('num_oov_dev: {:d}, num_oov_test: {:d}, ratio_oov_dev: {:.3f}, ratio_oov_test: {:.3f}\n'.format(
            data_stat['oov_dev'], data_stat['oov_test'], data_stat['ratio_oov_dev'], data_stat['ratio_oov_test']))

        f.writelines('num_oov_char_dev: {:d}, num_oov_char_test: {:d}'.format(
            data_stat['oov_char_dev'], data_stat['oov_char_test']))


def savefilenamelist(info_all, out_dir):
    parts = ['train', 'dev', 'test']

    for part in parts:
        out_file = os.path.join(out_dir, part+'_fn.txt')

        with open(out_file, 'w', encoding='utf-8') as f:
            for fn in info_all[part, 'filename']:
                f.writelines(fn+'\n')


def count_stat_data(info_all):
    # input: sentence list

    parts = ['train', 'dev', 'test']

    data_count = {('dev', 'num_words'): [], ('test', 'num_words'): [], ('train', 'num_words'): [],
                  ('dev', 'len_words'): [], ('test', 'len_words'): [], ('train', 'len_words'): []}

    data_stat = {('dev', 'num_eng'): 0, ('test', 'num_eng'): 0, ('train', 'num_eng'): 0,
                  ('dev', 'num_chi'): 0, ('test', 'num_chi'): 0, ('train', 'num_chi'): 0}

    store_words = {'dev': set(), 'test': set(), 'train': set()}
    store_chars = {'dev': set(), 'test': set(), 'train': set()}

    for part in parts:
        for line in info_all[part, 'sent']:
            words = line.split()

            data_count[part, 'num_words'].append(len(words))

            for word in words:
                store_words[part].add(word)

                if check_english_words(word):
                    data_count[part, 'len_words'].append(1)
                    data_stat[part, 'num_eng'] += 1
                    store_chars[part].add(word)
                else: # non-English word
                    data_count[part, 'len_words'].append(len(word))
                    data_stat[part, 'num_chi'] += 1

                    for w in word:
                        store_chars[part].add(w)

        np_sents = np.array(data_count[part, 'num_words'])
        data_stat[part, 'total_words'] = np_sents.sum()
        data_stat[part, 'max_words_per_sent'] = np_sents.max()
        data_stat[part, 'min_words_per_sent'] = np_sents.min()
        data_stat[part, 'mean_words_per_sent'] = np_sents.mean()

        np_words = np.array(data_count[part, 'len_words'])
        data_stat[part, 'max_len_per_word'] = np_words.max()
        data_stat[part, 'min_len_per_word'] = np_words.min()
        data_stat[part, 'mean_len_per_word'] = np_words.mean()

        data_stat[part, 'unique_words'] = len(store_words[part])
        data_stat[part, 'unique_chars'] = len(store_chars[part])

    set_diff_dev = store_words['dev'] - store_words['train']
    set_diff_test = store_words['test'] - store_words['train']
    data_stat['oov_dev'] = len(set_diff_dev)
    data_stat['oov_test'] = len(set_diff_test)
    data_stat['ratio_oov_dev'] = len(set_diff_dev) * 1. / len(store_words['train'])
    data_stat['ratio_oov_test'] = len(set_diff_test) * 1. / len(store_words['train'])

    set_diff_char_dev = store_chars['dev'] - store_chars['train']
    set_diff_char_test = store_chars['test'] - store_chars['train']
    data_stat['oov_char_dev'] = len(set_diff_char_dev)
    data_stat['oov_char_test'] = len(set_diff_char_test)

    return data_stat, store_chars, store_words


def savechars(store_chars, out_dir):
    parts = ['train', 'dev', 'test']

    for part in parts:
        out_file = os.path.join(out_dir, part+'_chars.txt')

        with open(out_file, 'w', encoding='utf-8') as f:
            for ch in store_chars[part]:
                f.writelines(ch+'\n')


def savewords(store_words, out_dir):
    parts = ['train', 'dev', 'test']

    for part in parts:
        out_file = os.path.join(out_dir, part+'_words.txt')

        with open(out_file, 'w', encoding='utf-8') as f:
            for ch in store_words[part]:
                f.writelines(ch+'\n')


DEV_DROP = [
    '（ 完 ）\n',
    '（ <ENAMEX TYPE="PERSON">杨桂林</ENAMEX> <ENAMEX TYPE="PERSON">常新华</ENAMEX> （ 完 ）\n',
    '<ENAMEX TYPE="ORG">新华社</ENAMEX> 记者 <ENAMEX TYPE="PERSON">郭庆华</ENAMEX> （ 完 ）\n']
TEST_DROP = ['（ 完 ）\n']


def list_and_stat_docs(anno_dir, out_dir):
    out_file = os.path.join(out_dir, 'data_stat.txt')
    drop = {'dev': set(DEV_DROP), 'test': set(TEST_DROP), 'train': set()}

    stat = Counter()
    info_all = {('dev', 'sent'): [], ('test', 'sent'): [], ('train', 'sent'): [],
                ('dev', 'filename'): [], ('test', 'filename'): [], ('train', 'filename'): []}

    for domain in os.listdir(anno_dir):
        domain_dir = os.path.join(anno_dir, domain)
        if os.path.isfile(domain_dir): continue

        for source in os.listdir(domain_dir):
            source_dir = os.path.join(domain_dir, source)
            if os.path.isfile(source_dir): continue

            for group in os.listdir(source_dir):
                group_dir = os.path.join(source_dir, group)
                if os.path.isfile(group_dir): continue

                for filename in tqdm(os.listdir(group_dir)):
                    if filename == '.DS_Store': continue # add by haiqin for ignoring .DS_Store
                    part = which_part(domain, source, filename)
                    root, ext = os.path.splitext(filename)
                    if ext == '.name':
                        info_all[part, 'filename'].append(root)

                        name_lines = read_name_lines(os.path.join(group_dir, filename))
                        parse_lines = read_parse_lines(os.path.join(group_dir, root + '.parse'))
                        assert len(name_lines) == len(parse_lines)

                        for name_line, parse_line in zip(name_lines, parse_lines):
                            if name_line in drop[part]: continue

                            sent = re.sub('<.*?>', ' ', name_line).strip()
                            info_all[part, 'sent'].append(sent)

                            stat[part, 'line'] += 1
                        stat[part, 'file'] += 1

                print('finish g', group)
            print('finish s', source)
        print('finish d', domain)

    print(stat)
    data_stat, store_chars, store_words = count_stat_data(info_all)
    print(data_stat)

    savestat2file(stat, data_stat, out_file)
    savefilenamelist(info_all, out_dir)
    savechars(store_chars, out_dir)
    savewords(store_words, out_dir)

# end listdocs


if __name__=='__main__':
    sent_line = '<a>iPhoneXs 在 澳大利亚 发售 。</a>'
    sent = re.sub('<.*?>', ' ', sent_line).strip()

    PRE_DIR = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/'
    ANNOTATION_DIR = PRE_DIR+'raw_data/annotations'
    OUTPUT_DIR = PRE_DIR+'proc_data/data_stat/Ontonotes/'
    vocab_file = '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese/vocab.txt'
    list_and_stat_docs(anno_dir=ANNOTATION_DIR, out_dir=OUTPUT_DIR)
