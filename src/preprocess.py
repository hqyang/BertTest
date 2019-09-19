import os
import pandas as pd
import logging
import itertools
import re
import torch
from src.BERT.tokenization import BertTokenizer
import copy
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from .config import MAX_SUBWORDS, MAX_GRAM_LEN, NUM_HIDDEN_SIZE


def set1_from_tuple(pp, mat, wd_tuple, shift0=0, shift1=0):
    # pp: re.compile(r'[(](.*?)[)]', re.S)
    # mat: matrix
    # wd_tuple: word_tuple stored in string format
    # shift0: add the shifted index in the first axis to store 1
    # shift1: add the shifted index in the second axis to store 1

    if wd_tuple != '[]':
        wd_list = re.findall(pp, wd_tuple)

        for wdl in wd_list:
            idx_l = wdl.split(',')
            i0 = int(idx_l[0])
            i1 = int(idx_l[1])
            mat[i0+shift0][i1+shift1] = 1


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True
    return False


def check_chinese_words(word):
    for idx in range(len(word)):
        if not is_chinese_char(ord(word[idx])):
            return False # one char is not English, it is not an English word
    return True


def is_english_char(cp):
    """Checks whether CP is an English character."""
    # https://zh.wikipedia.org/wiki/%E5%85%A8%E5%BD%A2%E5%92%8C%E5%8D%8A%E5%BD%A2
    if ((cp >= 0x0041 and cp <= 0x005A) or
        (cp >= 0x0061 and cp <= 0x007A) or
        (cp >= 0xFF21 and cp <= 0xFF3A) or
        (cp >= 0xFF41 and cp <= 0xFF5A)):
        return True

    return False


def check_english_words(word):
    word = word.lower()
    if '[unk]' in word or '[unused' in word: # detecting unknown token
        return True

    for idx in range(len(word)):
        if not is_english_char(ord(word[idx])):
            return False # one char is not English, it is not an English word
    return True


def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def define_words_set(words, do_whole_word_mask=True):
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.

    cand_indexes = []

    for (i, word) in enumerate(words):
        if word == "[CLS]" or word == "[SEP]":
            continue

        if (do_whole_word_mask and len(cand_indexes) >= 1 and
            word.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    return cand_indexes


# prepare the length of words does not exceed max_length while considering the situation of do_whole as _mask
def set_words_boundary(words, cand_indexes, max_length):
    i = 0

    for i, cand_index in enumerate(cand_indexes):
        last_index = cand_index[-1]
        if last_index > max_length - 2:
            i = i - 1
            break

    if i > 0 or len(cand_indexes[i]) < max_length - 1:
        last_index = cand_indexes[i][-1]+1
    else:  # i = 0, max_length - 2, consider two specific tokens, 'CLS' and 'SEP'
        last_index = cand_indexes[i][max_length-2]+1

    return words[:last_index], i+3 # include 'CLS' and 'SEP' and the current index


def define_tokens_set(words, tokens, do_whole_word_mask=True):
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.

    cand_indexes = []
    token_ids = []
    #last_index = 0

    for (i, word) in enumerate(words):
        if word == "[CLS]" or word == "[SEP]":
            token_ids.append([tokens[i]])
            continue

        if (do_whole_word_mask and len(cand_indexes) >= 1 and
            word.startswith("##")):
            cand_indexes[-1].append(i)
            token_ids[-1].append(tokens[i])
        else:
            cand_indexes.append([i])
            token_ids.append([tokens[i]])

        # len(token_ids) < max_length
        # if len(token_ids) > max_length - 2: # keep one more token for ['SEP']
        #    words = words[:last_index]
        #    tokens = tokens[:last_index]
        #    break

    return cand_indexes, token_ids, words, tokens


def define_tokens_set_lang_status(words, tokens, do_whole_word_mask=True):
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.

    cand_indexes = []
    token_ids = []
    lang_status = []

    for (i, word) in enumerate(words):
        if word == "[CLS]" or word == "[SEP]":
            token_ids.append([tokens[i]])
            continue

        if (do_whole_word_mask and len(cand_indexes) >= 1 and
            word.startswith("##")):
            cand_indexes[-1].append(i)
            token_ids[-1].append(tokens[i])
        else:
            cand_indexes.append([i])
            token_ids.append([tokens[i]])

            lsd = 1 if check_english_words(word) else 0

            lang_status.append(lsd)

        # len(token_ids) < max_length
        # if len(token_ids) > max_length - 2: # keep one more token for ['SEP']
        #    words = words[:last_index]
        #    tokens = tokens[:last_index]
        #    break

    return cand_indexes, token_ids, words, tokens, lang_status


def indexes2nparray(max_length, cand_indexes, token_ids):
    '''
     Inputs:
       max_length: e.g., 128 (<=512)
       cand_indexes: e.g., [[1], [2, 3], [4], ...]
       token_ids: e.g., [[101], [1738, 4501], [5110], ...]
    # Output:
       o_cand_indexes: array([[1, 0, 0, 0, 0], [2, 3, 0, 0, 0], [4, 0, 0, 0, 0], ...])
       token_ids: array([[101, 0, 0, 0, 0], [1738, 4501, 0, 0, 0], [5110, 0, 0, 0, 0], ...]
       since the index is at least 1, we set -1 to indicate unused indexes if MAX_SUBWORDS=5
    '''

    # 1. get max length
    max_subwords = 0
    for cand_index in cand_indexes:
        len_cand = len(cand_index)
        if max_subwords < len_cand:
            max_subwords = len_cand

    if max_subwords > MAX_SUBWORDS/2:
        print('The length of maximum sub-words is '+str(max_subwords))

    # 2. convert into np array with the same size
    o_cand_indexes = copy.deepcopy(cand_indexes)
    o_token_ids = copy.deepcopy(token_ids)

    for cand_index in o_cand_indexes:
        cand_index.extend([0]*(MAX_SUBWORDS-len(cand_index)))

    for o_token_id in o_token_ids:
        o_token_id.extend([0]*(MAX_SUBWORDS-len(o_token_id)))

    len_cand_indexes = len(o_cand_indexes)
    if len_cand_indexes < max_length:
        o_cand_indexes.extend([[0]*MAX_SUBWORDS]*(max_length-len_cand_indexes))

    len_token_ids = len(o_token_ids)
    if len_token_ids < max_length:
        o_token_ids.extend([[0]*MAX_SUBWORDS]*(max_length-len_token_ids))

    o_cand_indexes = np.array(o_cand_indexes)
    o_token_ids = np.array(o_token_ids)

    #print(o_cand_indexes.shape)
    #print(o_token_ids.shape)
    #if o_cand_indexes.shape!=o_token_ids.shape:
    #    pdb.set_trace()
    #assert(o_cand_indexes.shape==o_token_ids.shape)

    return o_cand_indexes, o_token_ids


def indexes2nparray_b(max_length, cand_indexes, token_ids):
    '''
     Inputs:
       max_length: e.g., 128 (<=512)
       cand_indexes: e.g., [[1], [2, 3], [4], ...]
       token_ids: e.g., [[101], [1738, 4501], [5110], ...]
    # Output:
       o_cand_indexes: array([[1, 0, 0, 0, 0], [2, 3, 0, 0, 0], [4, 0, 0, 0, 0], ...])
       token_ids: array([[101, 0, 0, 0, 0], [1738, 4501, 0, 0, 0], [5110, 0, 0, 0, 0], ...]
       since the index is at least 1, we set -1 to indicate unused indexes if MAX_SUBWORDS=5
    '''

    # 1. get max length
    max_subwords = 0
    for cand_index in cand_indexes:
        len_cand = len(cand_index)
        if max_subwords < len_cand:
            max_subwords = len_cand

    if max_subwords > MAX_SUBWORDS/2:
        print('The length of maximum sub-words is '+str(max_subwords))

    # 2. convert into np array with the same size
    o_cand_indexes = []
    o_token_ids = []

    for cand_index in cand_indexes:
        if len(cand_index)>MAX_SUBWORDS:
            o_cand_indexes.append(cand_index[:MAX_SUBWORDS])
        else:
            o_cand_indexes.append(cand_index.extend([0]*(MAX_SUBWORDS-len(cand_index))))

    for token_id in token_ids:
        if len(token_id)>MAX_SUBWORDS:
            o_token_ids.append(token_id[:MAX_SUBWORDS])
        else:
            o_token_ids.append(token_id.extend([0]*(MAX_SUBWORDS-len(o_token_id))))

    len_cand_indexes = len(o_cand_indexes)
    if len_cand_indexes < max_length:
        o_cand_indexes.extend([[0]*MAX_SUBWORDS]*(max_length-len_cand_indexes))

    len_token_ids = len(o_token_ids)
    if len_token_ids < max_length:
        o_token_ids.extend([[0]*MAX_SUBWORDS]*(max_length-len_token_ids))

    o_cand_indexes = np.array(o_cand_indexes)
    o_token_ids = np.array(o_token_ids)

    #print(o_cand_indexes.shape)
    #print(o_token_ids.shape)
    #if o_cand_indexes.shape!=o_token_ids.shape:
    #    pdb.set_trace()
    #assert(o_cand_indexes.shape==o_token_ids.shape)

    return o_cand_indexes, o_token_ids


def cand2nparray(cand_indexes):
    # 1. get max length
    max_subwords = 0
    for cand_index in cand_indexes:
        len_cand = len(cand_index)
        if max_subwords < len_cand:
            max_subwords = len_cand
    print('The length of maximum sub-words is '+str(max_subwords))

    # 2. convert into np array with the same size
    o_cand_indexes = copy.deepcopy(cand_indexes)
    o_cand_mask = []
    for cand_index in o_cand_indexes:
        len_cand = len(cand_index)
        cand_index.extend([0]*(MAX_SUBWORDS-len_cand))
        o_cand_mask.append([1]*len_cand + [0]*(MAX_SUBWORDS-len_cand))

    o_cand_indexes = np.array(o_cand_indexes)
    o_cand_mask = np.array(o_cand_mask)

    return o_cand_indexes, o_cand_mask


def cand_max2nparray(cand_indexes, max_length):
    # 1. get max length
    max_subwords = 0
    for cand_index in cand_indexes:
        len_cand = len(cand_index)
        if max_subwords < len_cand:
            max_subwords = len_cand

    print('The length of maximum sub-words is '+str(max_subwords))

    # 2. convert into np array with the same size
    o_cand_indexes = copy.deepcopy(cand_indexes)
    o_cand_mask = []
    for cand_index in o_cand_indexes:
        len_cand = len(cand_index)
        cand_index.extend([0]*(MAX_SUBWORDS-len_cand))
        o_cand_mask.append([1]*len_cand + [0]*(MAX_SUBWORDS-len_cand))

    len_cand_indexes = len(o_cand_indexes)
    if len_cand_indexes < max_length:
        o_cand_indexes.extend([[0]*MAX_SUBWORDS]*(max_length-len_cand_indexes))

    o_cand_indexes = np.array(o_cand_indexes)
    o_cand_mask = np.array(o_cand_mask)

    return o_cand_indexes, o_cand_mask


def construct_pos_tags(pos_tags_file, mode = 'BIO'):
    pos_label_list  = ['[START]', '[END]']

    with open(pos_tags_file) as f:
        raw_POS_list = f.readlines()

    pos_label_list.extend([m+'-'+x.strip() for x in raw_POS_list for m in mode])

    pos_label_map = {}
    for i, label in enumerate(pos_label_list):
        pos_label_map[label] = i

    pos_idx_to_label_map = {}
    for i, lmap in enumerate(pos_label_map):
        pos_idx_to_label_map[i] = lmap

    return pos_label_list, pos_label_map, pos_idx_to_label_map


def get_dataset_and_dataloader(processor, args, training=True, type_name='train'):
    dataset = OntoNotesDataset(processor, args.data_dir, args.vocab_file,
                             args.max_seq_length, training=training, type_name=type_name,
                               do_lower_case=args.do_lower_case,
                               do_mask_as_whole=args.do_mask_as_whole)
    dataloader = dataset_to_dataloader(dataset, args.train_batch_size,
                                       args.local_rank, training=training)
    return dataset, dataloader


def get_eval_dataloaders(processor, args):
    if 'ontonotes' in args.task_name.lower():
        parts = ['test', 'dev', 'train']
    else:
        parts = ['test', 'train']

    eval_dataloaders = {}
    for part in parts:
        eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, training=False, type_name=part)
        eval_dataloaders[part] = eval_dataloader

    return eval_dataloaders


def get_dataset_with_dict_and_dataloader(processor, args, training=True, type_name='train'):
    dataset = OntoNotesDataset_With_Dict(processor, args.data_dir, args.vocab_file,
                             args.max_seq_length, training=training, type_name=type_name,
                                do_lower_case=args.do_lower_case,
                                do_mask_as_whole=args.do_mask_as_whole)

    dataloader = dataset_to_dataloader(dataset, args.train_batch_size,
                                       args.local_rank, training=training)
    return dataset, dataloader


def get_eval_with_dict_dataloaders(processor, args):
    if 'ontonotes' in args.task_name.lower():
        parts = ['test', 'dev', 'train']
    else:
        parts = ['test', 'train']

    eval_dataloaders = {}
    for part in parts:
        eval_dataset, eval_dataloader = get_dataset_with_dict_and_dataloader(processor, args, training=False, type_name=part)
        eval_dataloaders[part] = eval_dataloader

    return eval_dataloaders


def get_dataset_stored_with_dict_and_dataloader(processor, args, training=True, type_name='train'):
    dataset = OntoNotesDataset_Stored_With_Dict(processor, args.data_dir, args.vocab_file,
                             args.max_seq_length, training=training, type_name=type_name,
                                do_lower_case=args.do_lower_case,
                                do_mask_as_whole=args.do_mask_as_whole,
                                dict_file=args.dict_file)

    dataloader = dataset_to_dataloader(dataset, args.train_batch_size,
                                       args.local_rank, training=training)
    return dataset, dataloader


def get_eval_stored_with_dict_dataloaders(processor, args):
    if 'ontonotes' in args.task_name.lower():
        parts = ['test', 'dev', 'train']
    else:
        parts = ['test', 'train']

    eval_dataloaders = {}
    for part in parts:
        eval_dataset, eval_dataloader = get_dataset_stored_with_dict_and_dataloader(processor, args, training=False, type_name=part)
        eval_dataloaders[part] = eval_dataloader

    return eval_dataloaders


class MeituProcessor:
    def __init__(self, level, multilabel=False, multitask=False, nopunc=False):
        self.label_list = None
        self.label_map = None
        assert level in (1, 2)
        self.level = level
        self.multilabel = multilabel
        self.nopunc = nopunc
        self.multitask = multitask
        self.text_col = 3
        self.train_df = None
        if not self.multitask:
            self.label_col = 1 if self.level == 1 else 2
        else:
            self.label_col = [1, 2]

    def get_train_examples(self, data_dir):
        """See base class."""
        logging.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        if self.train_df is None:
            df = pd.read_csv(os.path.join(data_dir, "train_bert.tsv"),
                         sep='\t', low_memory=False)
            df = df.iloc[df.iloc[:, self.label_col].dropna(how='all').index]
        else:
            df = self.train_df
        if self.label_list is None:
            self.get_labels(data_dir=data_dir, df=None)
        train_examples = self._create_examples(df.values, "train")
        return train_examples

    def get_dev_examples(self, data_dir):
        if self.label_list is None:
            self.get_labels(data_dir=data_dir, df=None)
        """See base class."""
        df = pd.read_csv(os.path.join(data_dir, "dev_bert.tsv"),
                         sep='\t', low_memory=False)
        df = df.iloc[df.iloc[:, self.label_col].dropna(how='all').index]
        return self._create_examples(df.values, "dev")

    def _get_label_by_sample(self, line):
        l = line[self.label_col]
        if not self.multitask:
            if not self.multilabel:
                label = l if l in self.label_list else 'UNK'
            else:
                label = [_ for _ in l.split(',') if _ in self.label_list]
                if label == []:
                    label = None
        else:
            label = []
            if not self.multilabel:
                for (label_list_by_task, label_by_task) in zip(self.label_list, l):
                    label_tmp = label_by_task if label_by_task in label_list_by_task else 'UNK'
                    label.append(label_tmp)
            else:
                for (label_list_by_task, label_by_task) in zip(self.label_list, l):
                    label_tmp = [_ for _ in label_by_task.split(',') if _ in label_list_by_task]
                    if label_tmp == []:
                        label = None
                        break
                    label.append(label_tmp)
        return label

    def _get_label_list_from_df(self, df):
        all_labels = df.iloc[:, self.label_col].dropna(how='all')
        label_list = []
        if not self.multitask:
            if not self.multilabel:
                label_list = ['UNK'] + sorted(list(set([_ for _ in all_labels.unique()])))
            else:
                label_list = sorted(list(set(itertools.chain(
                    *[_.split(',') for _ in all_labels.unique()]))))
        else:
            if not self.multilabel:
                for name, series in all_labels.iteritems():
                    l = ['UNK'] + sorted(list(set([_ for _ in series.dropna().unique()])))
                    label_list.append(l)
            else:
                 for name, series in all_labels.iteritems():
                    l = sorted(list(set(itertools.chain(
                        *[_.split(',') for _ in series.dropna().unique()]))))
                    label_list.append(l)
        return label_list

    def get_labels(self, data_dir=None, df=None):
        """See base class."""
        if self.label_list is None:
            #hardcode here
            if df is None:
                df = pd.read_csv(os.path.join(data_dir, "train_bert.tsv"),
                             sep='\t', low_memory=False)
                df = df.iloc[df.iloc[:, self.label_col].dropna(how='all').index]
                self.train_df = df
            self.label_list = self._get_label_list_from_df(df)
        label_list = self.label_list

        if not self.multitask:
            self.label_map = dict()
            for i, label in enumerate(self.label_list):
                self.label_map[label] = i
        else:
            self.label_map = []
            for label_list_by_task in self.label_list:
                tmp_map = {}
                for i, label in enumerate(label_list_by_task):
                    tmp_map[label] = i
                self.label_map.append(tmp_map)
        return label_list

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            text_a = line[self.text_col]
            if self.nopunc:
                text_a = re.sub('[^a-zA-Z0-9\u4e00-\u9fa5 ]+', '', text_a)
                text_a = re.sub(' +', ' ', text_a)
            text_a = tokenization.convert_to_unicode(text_a)
            label = self._get_label_by_sample(line)
            if (label is None) or (label == []):
                continue
            examples.append([text_a, label])
        examples = pd.DataFrame(examples, columns=['text', 'label'])
        return examples

    def save_labelidmap(self, output_dir):
        label_map = self.label_map
        if not isinstance(label_map, list):
            label_map = [label_map]
        for i, lmap in enumerate(label_map):
            lidmap = {y:x for x,y in lmap.items()}
            output_dict_file = os.path.join(output_dir, 'tag%d_labelidmap.dict'%(i+1))
            torch.save(lidmap, output_dict_file)


class CWS_BMEO(MeituProcessor):
    def __init__(self, nopunc=False, drop_columns=None):
        self.nopunc = nopunc
        self.drop_columns = drop_columns
        self.label_list = ['[START]', '[END]', 'B', 'M', 'E', 'S']
        self.label_map = {'[START]': 0, '[END]': 1, 'B': 2, 'M': 3, 'E': 4, 'S': 5}
        self.idx_to_label_map = {0: '[START]', 1: '[END]', 2: 'B', 3: 'M', 4: 'E', 5: 'S'}

    def get_examples(self, fn):
        """See base class."""
        df = pd.read_csv(fn, sep='\t')

        # full_pos (chunk), ner, seg, text
        # need parameter inplace=True
        df.drop(columns=self.drop_columns, inplace=True)

        # change name to tag for consistently processing
        df.rename(columns={'bert_seg': 'label'}, inplace=True)

        return df

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "train.tsv"))

    def get_dev_examples(self, data_dir):
        return self.get_examples(os.path.join(data_dir, "dev.tsv"))

    def get_test_examples(self, data_dir):
        return self.get_examples(os.path.join(data_dir, "test.tsv"))

    def get_other_examples(self, data_dir, fn):
        return self.get_examples(os.path.join(data_dir, fn))

    def get_labels(self):
        return self.label_list


class CWS_POS(MeituProcessor):
    def __init__(self, nopunc=False, drop_columns=None, pos_tags_file='pos_tags.txt'):
        self.nopunc = nopunc
        self.drop_columns = drop_columns
        self.label_list = ['[START]', '[END]', 'B', 'M', 'E', 'S']
        self.label_map = {'[START]': 0, '[END]': 1, 'B': 2, 'M': 3, 'E': 4, 'S': 5}
        self.idx_to_label_map = {0: '[START]', 1: '[END]', 2: 'B', 3: 'M', 4: 'E', 5: 'S'}
        self.pos_label_list, self.pos_label_map, self.pos_idx_to_label_map \
            = construct_pos_tags(pos_tags_file, mode = 'BIO')

    def get_examples(self, fn):
        """See base class."""
        df = pd.read_csv(fn, sep='\t')

        # full_pos (chunk), ner, seg, text
        # need parameter inplace=True
        df.drop(columns=self.drop_columns, inplace=True)

        # change name to tag for consistently processing
        df.rename(columns={'bert_seg': 'label'}, inplace=True)

       # change name to tag for consistently processing
        df.rename(columns={'bert_pos': 'label_pos'}, inplace=True)

        return df

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.get_examples(os.path.join(data_dir, "train.tsv"))

    def get_dev_examples(self, data_dir):
        return self.get_examples(os.path.join(data_dir, "dev.tsv"))

    def get_test_examples(self, data_dir):
        return self.get_examples(os.path.join(data_dir, "test.tsv"))

    def get_other_examples(self, data_dir, fn):
        return self.get_examples(os.path.join(data_dir, fn))

    def get_labels(self):
        return self.label_list

    def get_POS_labels(self):
        return self.pos_label_list


class OntoNotesDataset(Dataset):
    def __init__(self, processor, data_dir, vocab_file, max_length, training=True, type_name='train', do_lower_case=True, \
                 do_mask_as_whole=False):
        self.tokenizer = BertTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_length = max_length
        self.processor = processor
        self.data_dir = data_dir
        self.training = training
        self.train_df = None
        self.dev_df = None
        self.test_df = None
        self.df = None
        self.train(training=training, type_name=type_name)
        self.label_list = processor.get_labels()
        self.label_map = processor.label_map
        self.do_mask_as_whole = do_mask_as_whole

        pos_label_map = getattr(processor, 'pos_label_map', None)
        if pos_label_map is not None:
            self.pos_label_map = pos_label_map
        else:
            self.pos_label_map = None

    def train(self, training=True, type_name='train'):
        self.training = training
        if type_name=='train':
            if self.train_df is None:
                self.train_df = self.processor.get_train_examples(self.data_dir)
            self.df = self.train_df
        elif type_name=='dev':
            if self.dev_df is None:
                self.dev_df = self.processor.get_dev_examples(self.data_dir)
            self.df = self.dev_df
        elif type_name=='test':
            if self.test_df is None:
                self.test_df = self.processor.get_test_examples(self.data_dir)
            self.df = self.test_df
        else:
            self.df = self.processor.get_other_examples(self.data_dir, ty+".tsv")

        return self

    def dev(self):
        return self.train(training=False)

    def _tokenize(self):
        logging.info('Tokenizing...')
        st = time.time()
        tokens = []
        labelids = []
        pos_label_ids = []
        cand_indexes = []

        for i, data in enumerate(self.df.itertuples()):
            if self.do_mask_as_whole:
                token, cand_index = tokenize_text_with_cand_indexes(data.text, self.max_length, self.tokenizer)
                # cand_index_len = list(cand_index.size())[0]
                # labelid = tokenize_label_list_restriction(data.label, self.max_length, self.label_map, cand_index_len)
                labelid = tokenize_label_list(data.label, self.max_length, self.label_map)

                if self.pos_label_map:
                    # pos_label_id = tokenize_label_list_restriction(data.label_pos, self.max_length, self.pos_label_map, cand_index_len)
                    pos_label_id = tokenize_label_list(data.label_pos, self.max_length, self.pos_label_map)
                    pos_label_ids.append(pos_label_id)
            else: # no cand_index
                token = tokenize_text(data.text, self.max_length, self.tokenizer)
                labelid = tokenize_label_list(data.label, self.max_length, self.label_map)

                if self.pos_label_map:
                    pos_label_id = tokenize_label_list(data.label_pos, self.max_length, self.pos_label_map)
                    pos_label_ids.append(pos_label_id)

            tokens.append(token)
            labelids.append(labelid)

            if self.do_mask_as_whole:
                cand_indexes.append(cand_index)

            if i % 100000 == 0:
                logging.info("Writing example %d of %d" % (i, self.df.shape[0]))
        self.df['token'] = tokens
        self.df['labelid'] = labelids

        # The rest two components may be empty
        self.df['pos_label_id'] = pos_label_ids
        self.df['cand_index'] = cand_indexes

        logging.info('Loading time: %.2fmin' % ((time.time()-st)/60))
        logging.info('Loading time: %.2fmin' % ((time.time()-st)/60))

    def __getitem__(self, i):
        if 'token' not in self.df.columns: # encode is here
            self._tokenize()
        data = self.df.iloc[i]

        if self.pos_label_map is None:
            if not self.do_mask_as_whole: # no cand_index
                token, labelid = data.token, data.labelid
                if not isinstance(labelid, list):
                    labelid = [labelid]

                return tuple(token + labelid)
            else: # contain cand_index
                token, labelid, cand_index = data.token, data.labelid, data.cand_index

                if not isinstance(labelid, list):
                    labelid = [labelid]

                if not isinstance(cand_index, list):
                    cand_index = [cand_index]

                return tuple(token + labelid), cand_index # three tuples
        else:
            if not self.do_mask_as_whole: # no cand_index, cand_mask
                token, labelid, pos_label_id = data.token, data.labelid, data.pos_label_id

                if not isinstance(labelid, list):
                    labelid = [labelid]

                if not isinstance(pos_label_id, list):
                    pos_label_id = [pos_label_id]

                return tuple(token + labelid + pos_label_id) # three tuples
            else: # contain cand_index, cand_mask
                token, labelid, pos_label_id, cand_index \
                    = data.token, data.labelid, data.pos_label_id, data.cand_index

                if not isinstance(labelid, list):
                    labelid = [labelid]

                if not isinstance(pos_label_id, list):
                    pos_label_id = [pos_label_id]

                if not isinstance(cand_index, list):
                    cand_index = [cand_index]

                return tuple(token + labelid + pos_label_id), cand_index # four tuples

    def __call__(self, i):
        data = self.df.iloc[i]

        if self.pos_label_map is None:
            if self.do_mask_as_whole: # no cand_index
                text, label, cand_index = data.text, data.label, data.cand_index#, data.cand_mask
                return text, label, cand_index#, cand_mask
            else: # contain cand_index
                text, label = data.text, data.label
                return text, label
        else: # no pos_label_map
            if self.do_mask_as_whole: # no cand_index
                text, label, pos_label, cand_index \
                    = data.text, data.label, data.pos_label,data.cand_index
                return text, label, pos_label, cand_index#, cand_mask
            else: # contain cand_index
                text, label, pos_label = data.text, data.label, data.pos_label
                return text, label, pos_label

    def __len__(self):
        return self.df.shape[0]


class OntoNotesDataset_With_Dict(OntoNotesDataset):
    def __init__(self, processor, data_dir, vocab_file, max_length, training=True, type_name='train', do_lower_case=True, \
                 do_mask_as_whole=False, dict_file='./resource/dict.txt'):
        self.tokenizer = BertTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_length = max_length
        self.processor = processor
        self.data_dir = data_dir
        self.training = training
        self.train_df = None
        self.dev_df = None
        self.test_df = None
        self.df = None
        self.train(training=training, type_name=type_name)
        self.label_list = processor.get_labels()
        self.label_map = processor.label_map
        self.do_mask_as_whole = do_mask_as_whole
        self.dict_file = dict_file

        if dict_file is not None:
            self.dict = list(read_dict(dict_file).keys()) # set dict as a list
            self.max_gram = MAX_GRAM_LEN # default is 16
        else:
            self.max_gram = 1 # if no dictionary

        pos_label_map = getattr(processor, 'pos_label_map', None)
        if pos_label_map is not None:
            self.pos_label_map = pos_label_map
        else:
            self.pos_label_map = None

    def train(self, training=True, type_name='train'):
        self.training = training
        if type_name=='train':
            if self.train_df is None:
                self.train_df = self.processor.get_train_examples(self.data_dir)
            self.df = self.train_df
        elif type_name=='dev':
            if self.dev_df is None:
                self.dev_df = self.processor.get_dev_examples(self.data_dir)
            self.df = self.dev_df
        elif type_name=='test':
            if self.test_df is None:
                self.test_df = self.processor.get_test_examples(self.data_dir)
            self.df = self.test_df
        else:
            self.df = self.processor.get_other_examples(self.data_dir, ty+".tsv")

        return self

    def dev(self):
        return self.train(training=False)

    def _tokenize(self):
        logging.info('Tokenizing...')
        st = time.time()
        tokens = []
        labelids = []
        pos_label_ids = []
        cand_indexes = []
        wd_fvs = []

        for i, data in enumerate(self.df.itertuples()):
            if self.do_mask_as_whole:
                token, cand_index, words = tokenize_text_with_cand_indexes_to_words(data.text, self.max_length, self.tokenizer)
                labelid = tokenize_label_list(data.label, self.max_length, self.label_map)

                if self.pos_label_map:
                    pos_label_id = tokenize_label_list(data.label_pos, self.max_length, self.pos_label_map)
                    pos_label_ids.append(pos_label_id)
            else: # no cand_index
                token, words = tokenize_text_to_words(data.text, self.max_length, self.tokenizer)
                labelid = tokenize_label_list(data.label, self.max_length, self.label_map)

                if self.pos_label_map:
                    pos_label_id = tokenize_label_list(data.label_pos, self.max_length, self.pos_label_map)
                    pos_label_ids.append(pos_label_id)

            tokens.append(token)
            labelids.append(labelid)

            if self.do_mask_as_whole:
                cand_indexes.append(cand_index)

            if self.max_gram > 1:
                wd_fv = words2dict_feature_vec(words, self.dict, self.max_gram, self.max_length)
                wd_fvs.append(np.array(wd_fv))

            if i % 100000 == 0:
                logging.info("Writing example %d of %d" % (i, self.df.shape[0]))

        # generate t_mask
        sz_eo = token.shape
        sz_ivd = wd_fv.shape

        t_mask0 = np.zeros((sz_eo[0], sz_eo[1], sz_eo[2]-sz_ivd[2]))
        t_mask1 = np.ones((sz_eo[0], sz_eo[1], sz_ivd[2]))
        t_mask = np.concatenate((t_mask0, t_mask1), axis=2)

        self.df['token'] = tokens
        self.df['labelid'] = labelids

        # The rest three components may be empty
        self.df['pos_label_id'] = pos_label_ids
        self.df['cand_index'] = cand_indexes
        self.df['word_feature_vector'] = wd_fvs
        self.df['t_mask'] = t_mask

        logging.info('Loading time: %.2fmin' % ((time.time()-st)/60))
        logging.info('Loading time: %.2fmin' % ((time.time()-st)/60))

    def __getitem__(self, i):
        if 'token' not in self.df.columns: # encode is here
            self._tokenize()
        data = self.df.iloc[i]

        if self.max_gram > 1: # non-empty
            wd_fvs = data.word_feature_vector
        else:
            wd_fvs = []

        if self.pos_label_map is None:
            if not self.do_mask_as_whole: # no cand_index
                token, labelid = data.token, data.labelid
                if not isinstance(labelid, list):
                    labelid = [labelid]

                return tuple(token + labelid), wd_fvs
            else: # contain cand_index
                token, labelid, cand_index = data.token, data.labelid, data.cand_index

                if not isinstance(labelid, list):
                    labelid = [labelid]

                if not isinstance(cand_index, list):
                    cand_index = [cand_index]

                return tuple(token + labelid), cand_index, wd_fvs # three tuples
        else:
            if not self.do_mask_as_whole: # no cand_index, cand_mask
                token, labelid, pos_label_id = data.token, data.labelid, data.pos_label_id

                if not isinstance(labelid, list):
                    labelid = [labelid]

                if not isinstance(pos_label_id, list):
                    pos_label_id = [pos_label_id]

                return tuple(token + labelid + pos_label_id), wd_fvs # three tuples
            else: # contain cand_index, cand_mask
                token, labelid, pos_label_id, cand_index \
                    = data.token, data.labelid, data.pos_label_id, data.cand_index

                if not isinstance(labelid, list):
                    labelid = [labelid]

                if not isinstance(pos_label_id, list):
                    pos_label_id = [pos_label_id]

                if not isinstance(cand_index, list):
                    cand_index = [cand_index]

                return tuple(token + labelid + pos_label_id), cand_index, wd_fvs # three tuples

    def __call__(self, i):
        data = self.df.iloc[i]

        if self.pos_label_map is None:
            if self.do_mask_as_whole: # no cand_index
                text, label, cand_index, wd_fv = data.text, data.label, data.cand_inde, data.word_feature_vector#, data.cand_mask
                return text, label, cand_index, wd_fv#, cand_mask
            else: # contain cand_index
                text, label, wd_fv = data.text, data.label, data.word_feature_vector
                return text, label, wd_fv
        else: # no pos_label_map
            if self.do_mask_as_whole: # no cand_index
                text, label, pos_label, cand_index, wd_fv \
                    = data.text, data.label, data.pos_label, data.cand_index, data.word_feature_vector
                return text, label, pos_label, cand_index, wd_fv #, cand_mask
            else: # contain cand_index
                text, label, pos_label, wd_fv = data.text, data.label, data.pos_label, data.word_feature_vector
                return text, label, pos_label, wd_fv

    def __len__(self):
        return self.df.shape[0]


class OntoNotesDataset_Stored_With_Dict(OntoNotesDataset):
    def __init__(self, processor, data_dir, vocab_file, max_length, training=True, type_name='train', do_lower_case=True, \
                 do_mask_as_whole=False, dict_file='./resource/dict.txt'):
        self.tokenizer = BertTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_length = max_length
        self.processor = processor
        self.data_dir = data_dir
        self.training = training
        self.train_df = None
        self.dev_df = None
        self.test_df = None
        self.df = None
        self.train(training=training, type_name=type_name)
        self.label_list = processor.get_labels()
        self.label_map = processor.label_map
        self.do_mask_as_whole = do_mask_as_whole
        self.dict_file = dict_file
        self.zeros_mat = np.zeros((max_length, NUM_HIDDEN_SIZE+2*(MAX_GRAM_LEN-1)), dtype=np.uint8)
        self.pattern = re.compile(r'[(](.*?)[)]', re.S) # minimum matching ()

        if dict_file is not None:
            self.dict = list(read_dict(dict_file).keys()) # set dict as a list
            self.max_gram = MAX_GRAM_LEN # default is 16
        else:
            self.max_gram = 1 # if no dictionary

        pos_label_map = getattr(processor, 'pos_label_map', None)
        if pos_label_map is not None:
            self.pos_label_map = pos_label_map
        else:
            self.pos_label_map = None

    def _generate_wd_vec(self, wd_tuple):
        # generate t_mask
        dict2feat_vec = self.zeros_mat.copy()
        # the second axis should be shifted NUM_HIDDEN_SIZE
        set1_from_tuple(self.pattern, dict2feat_vec, wd_tuple, 0, NUM_HIDDEN_SIZE)

        return dict2feat_vec

    def _tokenize(self):
        logging.info('Tokenizing...')
        st = time.time()
        tokens = []
        labelids = []
        pos_label_ids = []
        cand_indexes = []
        wd_fvs = []

        for i, data in enumerate(self.df.itertuples()):
            if self.do_mask_as_whole:
                token, cand_index, words = tokenize_text_with_cand_indexes_to_words(data.text, self.max_length, self.tokenizer)
                labelid = tokenize_label_list(data.label, self.max_length, self.label_map)

                if self.pos_label_map:
                    pos_label_id = tokenize_label_list(data.label_pos, self.max_length, self.pos_label_map)
                    pos_label_ids.append(pos_label_id)
            else: # no cand_index
                token, words = tokenize_text_to_words(data.text, self.max_length, self.tokenizer)
                labelid = tokenize_label_list(data.label, self.max_length, self.label_map)

                if self.pos_label_map:
                    pos_label_id = tokenize_label_list(data.label_pos, self.max_length, self.pos_label_map)
                    pos_label_ids.append(pos_label_id)

            tokens.append(token)
            labelids.append(labelid)

            if self.do_mask_as_whole:
                cand_indexes.append(cand_index)

            if self.max_gram > 1:
                wd_fv = self._generate_wd_vec(data.word_in_dict_tuple)
                wd_fvs.append(wd_fv)

            if i % 100000 == 0:
                logging.info("Writing example %d of %d" % (i, self.df.shape[0]))


        self.df['token'] = tokens
        self.df['labelid'] = labelids

        # The rest three components may be empty
        self.df['pos_label_id'] = pos_label_ids
        self.df['cand_index'] = cand_indexes
        self.df['word_feature_vector'] = wd_fvs
        self.df['t_mask'] = t_mask

        logging.info('Loading time: %.2fmin' % ((time.time()-st)/60))
        logging.info('Loading time: %.2fmin' % ((time.time()-st)/60))

    def __getitem__(self, i):
        if 'token' not in self.df.columns: # encode is here
            self._tokenize()
        data = self.df.iloc[i]

        if hasattr(data, 'token'):
            token = data.token
        else:
            token = []

        if hasattr(data, 'labelid'):
            labelid = data.labelid

            if not isinstance(labelid, list):
                labelid = [labelid]
        else:
            labelid = []

        if hasattr(data, 'pos_label_id'):
            pos_label_id = data.pos_label_id

            if not isinstance(pos_label_id, list):
                pos_label_id = [pos_label_id]
        else:
            pos_label_id = ''

        if hasattr(data, 'cand_index'):
            cand_index = data.cand_index

            if not isinstance(cand_index, list):
                cand_index = [cand_index]
        else:
            cand_index = []

        if hasattr(data, 'word_feature_vector'):
            wd_fvs = data.word_feature_vector
        else:
            wd_fvs = []

        return tuple(token + labelid + pos_label_id), cand_index, wd_fvs # three tuples

    def __call__(self, i):
        data = self.df.iloc[i]

        if hasattr(data, 'text'):
            text = data.text
        else:
            text = ''

        if hasattr(data, 'label'):
            label = data.label
        else:
            label = ''

        if hasattr(data, 'pos_label'):
            pos_label = data.pos_label
        else:
            pos_label = ''

        if hasattr(data, 'cand_index'):
            cand_index = data.cand_index
        else:
            cand_index = []

        if hasattr(data, 'word_feature_vector'):
            wd_fv = data.word_feature_vector
        else:
            wd_fv = []

        return text, label, pos_label, cand_index, wd_fv

    ''' Inherited from OntoNotesDataset
    def train(self, training=True, type_name='train'):
        self.training = training
        if type_name=='train':
            if self.train_df is None:
                self.train_df = self.processor.get_train_examples(self.data_dir)
            self.df = self.train_df
        elif type_name=='dev':
            if self.dev_df is None:
                self.dev_df = self.processor.get_dev_examples(self.data_dir)
            self.df = self.dev_df
        elif type_name=='test':
            if self.test_df is None:
                self.test_df = self.processor.get_test_examples(self.data_dir)
            self.df = self.test_df
        else:
            self.df = self.processor.get_other_examples(self.data_dir, ty+".tsv")

        return self
    '''

    '''Inherited from OntoNotesDataset

    def dev(self):
        return self.train(training=False)
    '''

    ''' Inherited from OntoNotesDataset
    def __len__(self):
        return self.df.shape[0]
    '''


def text2tokens(text, max_length, tokenizer):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length - 2:
        tokens = tokens[:max_length - 2]
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    return tokens


def tokens2ids(tokens, max_length, tokenizer):
    # words = re.findall('[^0-9a-zA-Z]|[0-9a-zA-Z]+', text.lower())
    # words = list(filter(lambda x: x!=' ', words))
    # words = list(itertools.chain(*[tokenizer.tokenize(x) for x in words]))

    # models = tokenizer.models
    # tokens = [models[_] if _ in models.keys() else models['[UNK]'] for _ in words]
    # tokens = [models['[CLS]']] + tokens + [models['[SEP]']]
    tokens = tokenizer.convert_tokens_to_ids(tokens)
    len_tokens = len(tokens)
    if len_tokens < max_length:
        tokens.extend([0] * (max_length - len_tokens))
    tokens = np.array(tokens)
    mask = np.array([1] * len_tokens + [0] * (max_length - len_tokens))
    segment = np.array([0] * max_length)
    return [tokens, segment, mask]


def tokenize_text_with_cand_indexes(text, max_length, tokenizer):
    # words = re.findall('[^0-9a-zA-Z]|[0-9a-zA-Z]+', text.lower())
    # words = list(filter(lambda x: x!=' ', words))
    # words = list(itertools.chain(*[tokenizer.tokenize(x) for x in words]))
    words = tokenizer.tokenize(text)
    words = ['[CLS]'] + words

    cand_indexes = define_words_set(words)

    # prepare the length of words does not exceed max_length while considering the situation of do_whole as _mask
    words, can_index_len = set_words_boundary(words, cand_indexes, max_length)
    words += ['[SEP]']

    tokens = tokenizer.convert_tokens_to_ids(words)

    # suppose the length of words and tokens is less than max_length
    cand_indexes, token_ids, words, tokens = define_tokens_set(words, tokens)

    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))

    tokens = np.array(tokens)
    mask = np.array([1] * can_index_len + [0] * (max_length - can_index_len))
    segment = np.array([0] * max_length)

    cand_indexes, token_ids = indexes2nparray(max_length, cand_indexes, token_ids)

    return [tokens, segment, mask], [cand_indexes, token_ids] #, can_index_len # include ['SEP']


def tokenize_text_with_cand_indexes_to_words(text, max_length, tokenizer):
    # words = re.findall('[^0-9a-zA-Z]|[0-9a-zA-Z]+', text.lower())
    # words = list(filter(lambda x: x!=' ', words))
    # words = list(itertools.chain(*[tokenizer.tokenize(x) for x in words]))
    words = tokenizer.tokenize(text)
    words = ['[CLS]'] + words

    cand_indexes = define_words_set(words)

    # prepare the length of words does not exceed max_length while considering the situation of do_whole as _mask
    words, can_index_len = set_words_boundary(words, cand_indexes, max_length)
    words += ['[SEP]']

    tokens = tokenizer.convert_tokens_to_ids(words)

    # suppose the length of words and tokens is less than max_length
    cand_indexes, token_ids, words, tokens = define_tokens_set(words, tokens)

    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))
        words.extend(['[PAD]'] * (max_length - len(tokens)))

    tokens = np.array(tokens)
    mask = np.array([1] * can_index_len + [0] * (max_length - can_index_len))
    segment = np.array([0] * max_length)

    cand_indexes, token_ids = indexes2nparray(max_length, cand_indexes, token_ids)

    return [tokens, segment, mask], [cand_indexes, token_ids], words#, can_index_len # include ['SEP']


def tokenize_list_with_cand_indexes(words, max_length, tokenizer):
    # words = re.findall('[^0-9a-zA-Z]|[0-9a-zA-Z]+', text.lower())
    # words = list(filter(lambda x: x!=' ', words))
    # words = list(itertools.chain(*[tokenizer.tokenize(x) for x in words]))
    words = ['[CLS]'] + words

    cand_indexes = define_words_set(words)

    # prepare the length of words does not exceed max_length while considering the situation of do_whole as _mask
    if len(cand_indexes)!=0:
        words, can_index_len = set_words_boundary(words, cand_indexes, max_length)
    words += ['[SEP]']

    tokens = tokenizer.convert_tokens_to_ids(words)

    # suppose the length of words and tokens is less than max_length
    cand_indexes, token_ids, words, tokens = define_tokens_set(words, tokens)

    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))
    tokens = np.array(tokens)
    mask = np.array([1] * can_index_len + [0] * (max_length - can_index_len))
    segment = np.array([0] * max_length)

    cand_indexes, token_ids = indexes2nparray(max_length, cand_indexes, token_ids)
    return [tokens, segment, mask], [cand_indexes, token_ids]#, can_index_len # include ['SEP']


def tokenize_list_with_cand_indexes_lang_status(words, max_length, tokenizer):
    # words = re.findall('[^0-9a-zA-Z]|[0-9a-zA-Z]+', text.lower())
    # words = list(filter(lambda x: x!=' ', words))
    # words = list(itertools.chain(*[tokenizer.tokenize(x) for x in words]))
    words = ['[CLS]'] + words

    cand_indexes = define_words_set(words)

    # prepare the length of words does not exceed max_length while considering the situation of do_whole as _mask
    if len(cand_indexes)!=0:
        words, can_index_len = set_words_boundary(words, cand_indexes, max_length)
    words += ['[SEP]']

    tokens = tokenizer.convert_tokens_to_ids(words)

    # suppose the length of words and tokens is less than max_length
    cand_indexes, token_ids, words, tokens, lang_status = define_tokens_set_lang_status(words, tokens)

    len_tokens = len(tokens)
    if len_tokens < max_length:
        tokens.extend([0] * (max_length - len_tokens))
        lang_status.extend([0] * (max_length - len(token_ids)))  # no two tokens: [CLS] and [SEP]

    tokens = np.array(tokens)
    lang_status = np.array(lang_status)
    mask = np.array([1] * can_index_len + [0] * (max_length - can_index_len))
    segment = np.array([0] * max_length)

    cand_indexes, token_ids = indexes2nparray(max_length, cand_indexes, token_ids)
    return [tokens, segment, mask], [cand_indexes, token_ids], [lang_status]#, can_index_len # include ['SEP']


def tokenize_text(text, max_length, tokenizer):
    # words = re.findall('[^0-9a-zA-Z]|[0-9a-zA-Z]+', text.lower())
    # words = list(filter(lambda x: x!=' ', words))
    # words = list(itertools.chain(*[tokenizer.tokenize(x) for x in words]))
    words = tokenizer.tokenize(text)
    if len(words) > max_length - 2:
        words = words[:max_length - 2]
    words = ['[CLS]'] + words + ['[SEP]']
    # models = tokenizer.models
    # tokens = [models[_] if _ in models.keys() else models['[UNK]'] for _ in words]
    # tokens = [models['[CLS]']] + tokens + [models['[SEP]']]
    tokens = tokenizer.convert_tokens_to_ids(words)
    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))
    tokens = np.array(tokens)
    mask = np.array([1] * (len(words)) + [0] * (max_length - len(words)))
    segment = np.array([0] * max_length)
    return [tokens, segment, mask]


def tokenize_text_to_words(text, max_length, tokenizer):
    # words = re.findall('[^0-9a-zA-Z]|[0-9a-zA-Z]+', text.lower())
    # words = list(filter(lambda x: x!=' ', words))
    # words = list(itertools.chain(*[tokenizer.tokenize(x) for x in words]))
    words = tokenizer.tokenize(text)
    if len(words) > max_length - 2:
        words = words[:max_length - 2]
    words = ['[CLS]'] + words + ['[SEP]']
    # models = tokenizer.models
    # tokens = [models[_] if _ in models.keys() else models['[UNK]'] for _ in words]
    # tokens = [models['[CLS]']] + tokens + [models['[SEP]']]
    tokens = tokenizer.convert_tokens_to_ids(words)
    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))
        words.extend(['[PAD]'] * (max_length - len(tokens)))

    tokens = np.array(tokens)
    mask = np.array([1] * (len(words)) + [0] * (max_length - len(words)))
    segment = np.array([0] * max_length)
    return [tokens, segment, mask], words


def tokenize_list(words, max_length, tokenizer):
    # words = re.findall('[^0-9a-zA-Z]|[0-9a-zA-Z]+', text.lower())
    # words = list(filter(lambda x: x!=' ', words))
    # words = list(itertools.chain(*[tokenizer.tokenize(x) for x in words]))
    #words = tokenizer.tokenize(text)
    if len(words) > max_length - 2:
        words = words[:max_length - 2]
    words = ['[CLS]'] + words + ['[SEP]']
    # models = tokenizer.models
    # tokens = [models[_] if _ in models.keys() else models['[UNK]'] for _ in words]
    # tokens = [models['[CLS]']] + tokens + [models['[SEP]']]
    tokens = tokenizer.convert_tokens_to_ids(words)
    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))
    tokens = np.array(tokens)
    mask = np.array([1] * (len(words)) + [0] * (max_length - len(words)))
    segment = np.array([0] * max_length)
    return [tokens, segment, mask]


def tokenize_list_no_seg(words, max_length, tokenizer):
    # words = re.findall('[^0-9a-zA-Z]|[0-9a-zA-Z]+', text.lower())
    # words = list(filter(lambda x: x!=' ', words))
    # words = list(itertools.chain(*[tokenizer.tokenize(x) for x in words]))
    #words = tokenizer.tokenize(text)
    #if len(words) > max_length - 2:
    #    words = words[:max_length - 2]
    #words = ['[CLS]'] + words + ['[SEP]']
    # models = tokenizer.models
    # tokens = [models[_] if _ in models.keys() else models['[UNK]'] for _ in words]
    # tokens = [models['[CLS]']] + tokens + [models['[SEP]']]
    tokens = tokenizer.convert_tokens_to_ids(words)
    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))
    tokens = np.array(tokens)
    mask = np.array([1] * (len(words)) + [0] * (max_length - len(words)))
    segment = np.array([0] * max_length)
    return [tokens, segment, mask]


def tokenize_text_tag(text, tag, max_length, tokenizer):
    text_words = tokenizer.tokenize(text)
    tag_words = tokenizer.tokenize(tag)
    if len(text_words) + len(tag_words) > max_length - 3:
        text_words = text_words[:max_length - 3 - len(tag_words)]
    words = ['[CLS]'] + text_words + ['[SEP]'] + tag_words + ['[SEP]']
    tokens = tokenizer.convert_tokens_to_ids(words)
    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))

    tokens = np.array(tokens)
    mask = np.array([1] * (len(words)) + [0] * (max_length - len(words)))
    '''这个segment要看一下是怎么的结构'''
    segment = np.zeros(max_length)
    segment[2+len(text_words): 3+len(text_words)+len(tag_words)] = 1
    return [tokens, segment, mask]


def tokenize_label(label, label_map):
    assert isinstance(label_map, (dict, list))
    if isinstance(label_map, list):
        if isinstance(label[0], list):
            #multitask & multilabel
            label_id = [[task_map[_] for _ in task_label] for task_label, task_map in zip(label, label_map)]
            label_id = [np.array([1 if _ in task_labelid else 0 for _ in range(len(task_map))]) for \
                            task_labelid, task_map in zip(label_id, label_map)]
        else:
            #multitask & multiclass
            label_id = [task_map[task_label] for task_label, task_map in zip(label, label_map)]
    else:
        if isinstance(label, list):
            #multilabel
            label_id = [label_map[_] for _ in label]
            label_id = np.array([1 if _ in label_id else 0 for _ in range(len(label_map))])
        else:
            #multiclass
            label_id = label_map[label]
    return label_id


def tokenize_label_list_restriction(label, max_length, label_map, cand_index_len):
    # final_index is to restrict the size of the list
    assert isinstance(label_map, (dict, list))

    label_list = label.split()

    if len(label_list) > max_length - 2:
        label_list = label_list[:cand_index_len - 2]
    label_list = ['[START]'] + label_list + ['[END]']

    label_id = [label_map[_] for _ in label_list]
    #label_id = np.array([1 if _ in label_id else 0 for _ in range(len(label_map))])
    # add dump tokens to make them consistent with text tokens
    if len(label_id) < max_length:
        label_id.extend([0] * (max_length - len(label_id)))

    label_id = np.array(label_id)
    return label_id


def tokenize_label_list(label, max_length, label_map):
    assert isinstance(label_map, (dict, list))

    label_list = label.split()

    if len(label_list) > max_length - 2:
        label_list = label_list[:max_length - 2]
    label_list = ['[START]'] + label_list + ['[END]']

    label_id = [label_map[_] for _ in label_list]
    #label_id = np.array([1 if _ in label_id else 0 for _ in range(len(label_map))])
    # add dump tokens to make them consistent with text tokens
    if len(label_id) < max_length:
        label_id.extend([0] * (max_length - len(label_id)))

    label_id = np.array(label_id)
    return label_id


def dataset_to_dataloader(dataset, batch_size, local_rank=-1, training=True):
    if local_rank == -1:
        if training:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
    else:
        sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def randomly_mask_input(input_ids, tokenizer, mask_token_rate=0.15,
                        replace_mask=0.8, replace_random=0.1):
    assert replace_mask + replace_random < 1.
    vocab = tokenizer.vocab
    is_token = (input_ids != vocab['[PAD]']) & (input_ids != vocab['[CLS]']) & (input_ids !=vocab['[SEP]'])
    mask_token_count = (is_token.float().sum(1) * mask_token_rate + 1).long()
    mask_rand = torch.rand_like(is_token, dtype=torch.float32) * is_token.float()
    mask_rand_sort_index = mask_rand.sort(descending=True)[1].sort()[1]
    is_mask = (mask_rand_sort_index < mask_token_count.unsqueeze(1))
    mask_type_rand = torch.rand_like(is_token, dtype=torch.float32) * is_mask.float()
    is_replace_mask = (mask_type_rand < replace_mask) & (0. < mask_type_rand)
    is_replace_random = (mask_type_rand < replace_random + replace_mask) & \
                        (replace_mask <= mask_type_rand)

    masked_input_ids = input_ids * (1 - is_replace_random - is_replace_mask).long() + \
                    vocab['[MASK]'] * is_replace_mask.long() + \
                    torch.randint_like(
                            is_replace_random.long(), vocab['[MASK]']+1, 
                            len(vocab)) * is_replace_random.long()
    label_ids = input_ids * is_mask.long() + (-1) * (1 - is_mask.long())
    return masked_input_ids, label_ids


def read_dict(dict_file):
    with open(dict_file, 'r', encoding='utf8') as f:
        raw_data = f.readlines()

    dict = {}
    for rd in raw_data:
        wdl = rd.strip().split()

        if len(wdl)==1:
            dict[wdl[0]] = 'Nil'
        else:
            dict[wdl[0]] = wdl[1]

    return dict


def make_dict_feature_vec(sentence: str, word_dict: list, max_gram: int): #-> list
    if max_gram <= 1:
        raise ValueError('max gram should be greater than 1')

    if not sentence:
        return []

    res = []

    # for each char,
    for char_ind, char in enumerate(sentence):
        char_res = [0]*(2*(max_gram - 1))
        for rel_ind in range(1, max_gram):

            if char_ind - rel_ind >= 0:
                if sentence[char_ind - rel_ind:char_ind + 1] in word_dict:
                    char_res[2*(rel_ind - 1)] = 1
            if char_ind + rel_ind < len(sentence):
                if sentence[char_ind:char_ind + rel_ind + 1] in word_dict:
                    char_res[2*(rel_ind) - 1] = 1
        res.append(char_res)
    return res


def words2dict_feature_vec(words: list, word_dict: list, max_gram: int, max_length: int): #-> list
    if max_gram <= 1:
        raise ValueError('max gram should be greater than 1')

    if not words:
        return []

    res = []

    # for each char,
    for char_ind, char in enumerate(words):
        char_res = [0]*(2*(max_gram - 1))

        for rel_ind in range(1, max_gram):
            if char_ind - rel_ind >= 0:
                # construct words
                sent = ''.join(words[char_ind - rel_ind:char_ind + 1])
                if sent in word_dict:
                    char_res[2*(rel_ind - 1)] = 1
            if char_ind + rel_ind < len(words):
                sent = ''.join(words[char_ind:char_ind + rel_ind + 1])
                if sent in word_dict:
                    char_res[2*rel_ind - 1] = 1
        res.append(char_res)

    if len(words) < max_length:
        res.extend([[0]*(2*(max_gram - 1))] * (max_length - len(words)))
    return res


def words2dict_tuple(words: list, word_dict: list, max_gram: int): #-> tuple
    # the output tuple stores the indexes to indicate the words appear in the dict

    if max_gram <= 1:
        raise ValueError('max gram should be greater than 1')

    if not words:
        return []

    res = []

    # for each char,
    for char_ind, char in enumerate(words):
        #char_res = [0]*(2*(max_gram - 1))
        for rel_ind in range(1, max_gram):
            if char_ind - rel_ind >= 0:
                # construct words
                sent = ''.join(words[char_ind - rel_ind:char_ind + 1])
                if sent in word_dict:
                    # char_res[2*(rel_ind - 1)] = 1
                    idx_res = (char_ind, 2*(rel_ind - 1))
                    res.append(idx_res)

            if char_ind + rel_ind < len(words):
                sent = ''.join(words[char_ind:char_ind + rel_ind + 1])
                if sent in word_dict:
                    #char_res[2*rel_ind - 1] = 1
                    idx_res = (char_ind, 2*rel_ind - 1)
                    res.append(idx_res)


        #res.append(char_res)

    #if len(words) < max_length:
    #    res.extend([[0]*(2*(max_gram - 1))] * (max_length - len(words)))
    return res


def make_dict_feature_vec(sentence: str, word_dict: list, max_gram: int): #-> list
    if max_gram <= 1:
        raise ValueError('max gram should be greater than 1')

    if not sentence:
        return []

    res = []

    # for each char,
    for char_ind, char in enumerate(sentence):
        char_res = [0]*(2*(max_gram - 1))
        for rel_ind in range(1, max_gram):

            if char_ind - rel_ind >= 0:
                if sentence[char_ind - rel_ind:char_ind + 1] in word_dict:
                    char_res[2*(rel_ind - 1)] = 1
            if char_ind + rel_ind < len(sentence):
                if sentence[char_ind:char_ind + rel_ind + 1] in word_dict:
                    char_res[2*(rel_ind) - 1] = 1
        res.append(char_res)
    return res


def make_dict_feature_np_vec(sentence: str, word_dict: list, max_gram: int): #-> list of np
    if max_gram <= 1:
        raise ValueError('max gram should be greater than 1')

    if not sentence:
        return []

    res = []

    # for each char,
    for char_ind, char in enumerate(sentence):
        char_res = [0]*(2*(max_gram - 1))
        for rel_ind in range(1, max_gram):

            if char_ind - rel_ind >= 0:
                if sentence[char_ind - rel_ind:char_ind + 1] in word_dict:
                    char_res[2*(rel_ind - 1)] = 1
            if char_ind + rel_ind < len(sentence):
                if sentence[char_ind:char_ind + rel_ind + 1] in word_dict:
                    char_res[2*(rel_ind) - 1] = 1
        res.append(np.array(char_res))
    return res

# need more checkings
def preprocess_ner_2dict(json_path, tokenizer):
    s = replace_nrt(open(json_path, 'r', encoding='utf8').read())
    raw_data_list = s.split('{"text": "')
    raw_data_list = [raw_string.replace('"}, ', '')
                     for raw_string in raw_data_list]
    raw_data_list = raw_data_list[1:]
    project_table = {
        '人名': 'PER',
        '组织机构': 'ORG',
        '地址': 'LOC',
        '产品': 'PRD',
        '时间': 'TME',
        '地缘政治实体': 'GPE'
    }
    input_list = []
    target_list = []
    for sentence in raw_data_list:
        input_list.append([])
        target_list.append([])
        doc_chunk_list = sentence.split('{{')
        for chunk in doc_chunk_list:
            if '}}' not in chunk or ':' not in chunk:
                tokenized_chunk = tokenizer.tokenize(chunk)
                target_list[-1] += ['O']*len(tokenized_chunk)
                input_list[-1] += tokenized_chunk
            else:
                ent_chunk, text_chunk = chunk.split('}}')
                punc_ind = ent_chunk.index(':')
                ent_type = ent_chunk[:punc_ind]
                ent = ent_chunk[punc_ind+1:]
                if ent_type in project_table:
                    ent = tokenizer.tokenize(ent)
                    for char_ind, ent_char in enumerate(ent):
                        if char_ind == 0:
                            loc_char = 'B'
                        else:
                            loc_char = 'I'
                        target_list[-1].append(loc_char +
                                               '-'+project_table[ent_type])
                        input_list[-1].append(ent_char)
                else:
                    ent = tokenizer.tokenize(ent)
                    target_list[-1] += ['O']*len(ent)
                    input_list[-1] += ent

                text_chunk = tokenizer.tokenize(text_chunk)
                target_list[-1] += ['O']*len(text_chunk)
                input_list[-1] += text_chunk

    return input_list, target_list


