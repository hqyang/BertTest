#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 18:12 2019-02-27 
@author: haiqinyang

Feature: 

Scenario: 
"""
import os
import sys
sys.path.append('..')
import re
import pandas as pd
from src.tokenization import BasicTokenizer, FullTokenizer
from tqdm import tqdm
from src.utilis import check_english_words

'''
def check_English_words(word, basic_tokenizer):
    for idx in range(len(word)):
        if not basic_tokenizer._is_english_char(ord(word[idx])):
            return False # one char is not English, it is not an English word
    return True
'''

def preprocess_file(origin_path, output_path):
    with open(origin_path, 'r') as fr:
        with open(output_path, 'w') as fw:
            for line in fr:
                output_line = preprocess(line.strip())
                fw.write(output_line + '\n')
'''
def preprocess(seg):
    text = re.sub(' ', '', seg)
    seg_list = seg.split()
    bieo_list = [text2bio(text) for text in seg_list]
    bieo = ','.join(''.join(bieo_list))+ ','
    output = bieo + '\t' + text + '\t' + seg
    return output
'''

def preprocess2dict(sent, tagType, full_tokenizer, basic_tokenizer):
    # Input:
    #   sent: sentence (suppose the sentence is not empty and the words are segmented by space)
    #   tagType: the generate BIO or BMES
    #
    # Output:
    #  src_seg, text_str, text_seg, bert_seg

    seg_list = sent.split()
    text_seg = ' '.join(seg_list)

    model_type = {
            'BIO': list2BIOList,
            'BMES': list2BMESList
        }
    assert(tagType in model_type)

    func = model_type[tagType]
    text, bert_seg_list, src_seg_list = func(seg_list, full_tokenizer, basic_tokenizer) #[text2bio(text) for text in seg_list]
    bert_seg = ' '.join(''.join(bert_seg_list))
    src_seg = ','.join(''.join(src_seg_list)) + ','

    return {'bert_seg': bert_seg, 'src_seg': src_seg, 'text': text, 'text_seg': text_seg, \
            'full_pos': '', 'bert_ner': '', 'src_ner': ''} # full_pos, bert_ner, src_ner are redundant columns for consistence of Ontonotes datasets


def len_to_bio(length):
    assert (length > 0) and (isinstance(length, int))
    if length == 1:
        return 'O'
    else:
        return 'B' + 'I' * (length - 1)

def list2BIOList(text_list, full_tokenizer, basic_tokenizer):
    mode_status_list = [check_english_words(text) for text in text_list]

    src_seg_List = []
    bert_seg_list = []

    outText = ''
    for idx in range(len(text_list)):
        text = text_list[idx]
        src_seg = ''
        bert_seg = ''
        if mode_status_list[idx]: # English
            src_seg = ['O']

            wl = full_tokenizer.tokenize(text)
            len_wl = len(wl)
            if len_wl==1:
                bert_seg = src_seg
            else:
                bert_seg = ['B'] + ['I'] * (len_wl - 1)
        else: # Chinese or numerical
            len_text = len(text)
            wl = full_tokenizer.tokenize(text) # a list
            len_wl = len(wl)

            if len_text==1:
                src_seg = ['O']
            else:
                src_seg = ['B'] + ['I'] * (len_text - 1)

            if len_text == len_wl or len_wl==1: # len_wl may be a string of numbers
                bert_seg = src_seg
            else:
                bert_seg = ['B'] + ['I'] * (len_wl - 1)

        if idx>1 and mode_status_list[idx-1] and mode_status_list[idx]:
            outText += ' ' + text
        else:
            outText += text

        src_seg_List.extend(src_seg)
        bert_seg_list.extend(bert_seg)

    return  outText, bert_seg_list, src_seg_List

def list2BMESList(text_list, full_tokenizer, basic_tokenizer):
    # An example
    #  text_list = ['目前', '由', '２３２', '位', '院士', '（', 'Ｆｅｌｌｏｗ', '及', 'Ｆｏｕｎｄｉｎｇ', 'Ｆｅｌｌｏｗ', '）', '，', '６６', '位', '協院士', '（', 'Ａｓｓｏｃｉａｔｅ', 'Ｆｅｌｌｏｗ', '）', '２４', '位', '通信', '院士', '（', 'Ｃｏｒｒｅｓｐｏｎｄｉｎｇ', 'Ｆｅｌｌｏｗ', '）', '及', '２', '位', '通信', '協院士', '（', 'Ｃｏｒｒｅｓｐｏｎｄｉｎｇ', 'Ａｓｓｏｃｉａｔｅ', 'Ｆｅｌｌｏｗ', '）', '組成', '（', '不', '包括', '一九九四年', '當選', '者', '）', '，']
    #
    #  outText, bert_seg_list, src_seg_List = list2BMESList(text_list)
    #    outText = '目前由２３２位院士（Ｆｅｌｌｏｗ及Ｆｏｕｎｄｉｎｇ Ｆｅｌｌｏｗ），６６位協院士（Ａｓｓｏｃｉａｔｅ Ｆｅｌｌｏｗ）２４位通信院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ Ｆｅｌｌｏｗ）及２位通信協院士（Ｃｏｒｒｅｓｐｏｎｄｉｎｇ Ａｓｓｏｃｉａｔｅ Ｆｅｌｌｏｗ）組成（不包括一九九四年當選者），'
    #    bert_seg_list = ['B', 'E', 'S', 'B', 'M', 'E', 'S', 'B', 'E', 'S', 'B', 'M', 'M', 'M', 'M', 'E', 'S', 'B', 'M', 'M', 'M', 'M', 'M', 'M', 'E', 'B', 'M', 'M', 'M', 'M', 'E', 'S', 'S', 'B', 'E', 'S', 'B', 'M', 'E', 'S', 'B', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'E', 'B', 'M', 'M', 'M', 'M', 'E', 'S', 'S', 'S', 'B', 'E', 'B', 'E', 'S', 'B', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'E', 'B', 'M', 'M', 'M', 'M', 'E', 'S', 'S', 'S', 'S', 'B', 'E', 'B', 'M', 'E', 'S', 'B', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'E', 'B', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'E', 'B', 'M', 'M', 'M', 'M', 'E', 'S', 'B', 'E', 'S', 'S', 'B', 'E', 'B', 'M', 'M', 'M', 'E', 'B', 'E', 'S', 'S', 'S']
    #    src_seg_List =  ['B', 'E', 'S', 'S', 'S', 'B', 'E', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'B', 'M', 'E', 'S', 'S', 'S', 'S', 'S', 'S', 'B', 'E', 'B', 'E', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'B', 'E', 'B', 'M', 'E', 'S', 'S', 'S', 'S', 'S', 'B', 'E', 'S', 'S', 'B', 'E', 'B', 'M', 'M', 'M', 'E', 'B', 'E', 'S', 'S', 'S']
    #vocab_file = '../vocab/bert-base-chinese.txt'
    #full_tokenizer = FullTokenizer(vocab_file, do_lower_case=True)
    #basic_tokenizer = BasicTokenizer(do_lower_case=True)

    mode_status_list = [check_english_words(text) for text in text_list]

    src_seg_List = []
    bert_seg_list = []

    outText = ''
    for idx in range(len(text_list)):
        text = text_list[idx]

        if mode_status_list[idx]: # English
            src_seg = ['S']

            wl = full_tokenizer.tokenize(text)
            len_wl = len(wl)
            if len_wl==1:
                bert_seg = src_seg
            else:
                bert_seg = ['B'] + ['M'] * (len_wl - 2) + ['E']
        else: # Chinese or numerical
            len_text = len(text)
            wl = full_tokenizer.tokenize(text)
            len_wl = len(wl)

            if len_text==1:
                src_seg = ['S']
            else:
                src_seg = ['B'] + ['M'] * (len_text - 2) + ['E']

            if len_text == len_wl or len_wl==1: # len_wl may be a string of numbers
                bert_seg = src_seg
            else:
                bert_seg = ['B'] + ['M'] * (len_wl - 2) + ['E']

        if idx>1 and mode_status_list[idx-1] and mode_status_list[idx]:
            outText += ' ' + text
        else:
            outText += text

        src_seg_List.extend(src_seg)
        bert_seg_list.extend(bert_seg)

    return  outText, bert_seg_list, src_seg_List

def gen_data(in_file, out_file, tagType):
    with open(in_file, 'r', encoding='utf8') as f:
        raw_data = [_.strip() for _ in f.readlines()]

    vocab_file = '../vocab/bert-base-chinese.txt'
    full_tokenizer = FullTokenizer(vocab_file, do_lower_case=True)
    basic_tokenizer = BasicTokenizer(do_lower_case=True)

    data_all = [preprocess2dict(s, tagType, full_tokenizer, basic_tokenizer) for s in tqdm(raw_data)]

    df = pd.DataFrame(data_all)
    # separate with \t
    df.to_csv(out_file, sep='\t', encoding='utf-8', index=False)

    print('Finish writing generated '+tagType+' data!')

def remove_u3000(infile, outfile):
    with open(infile, 'r', encoding='utf8') as f:
        raw_data = [_.strip() for _ in f.readlines()] #  if _.strip()!='

    #data2 = [s for s in raw_data if s.strip()!='']
    with open(outfile, 'w', encoding='utf8') as fo:
        for text in raw_data:
            if text.strip()!='':
                fo.write(text+'\n')

def batch_remove_u3000():
    infile_dir = '/Users/haiqinyang/Downloads/codes/bert-multitask-learning/data/cws/'
    types = ['as', 'cityu', 'msr', 'pku']

    outfile_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/cws/'

    for dt in types:
        infile = infile_dir + 'training/' + dt + '_training.utf8'
        outfile = outfile_dir + dt + '_train.tsv'
        remove_u3000(infile, outfile)

        infile = infile_dir + 'gold/' + dt + '_test_gold.utf8'
        outfile = outfile_dir + dt + '_test.tsv'
        remove_u3000(infile, outfile)

def batch_gendata():
    infile_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/cws/'
    #infile_dir = '../../data/CWS/'
    types = ['as', 'cityu', 'msr', 'pku']

    tagType = 'BIO'
    #tagType = 'BMES'
    outfile_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/cws/'

    #outfile_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/cws/'
    #outfile_dir = '../../data/CWS/'

    outfile_dir += tagType + '/'
    os.makedirs(outfile_dir, exist_ok=True)

    for dt in types:
        infile = infile_dir + dt + '_train.tsv'
        outfile = outfile_dir + dt + '_train.tsv'
        gen_data(infile, outfile, tagType)

        infile = infile_dir + dt + '_test.tsv'
        outfile = outfile_dir + dt + '_test.tsv'
        gen_data(infile, outfile, tagType)

TESTFLAG = False
TESTFLAG = True

if __name__ == '__main__':
    #remove_u3000('tmp_input.txt', 'tmp_output.txt')
    #batch_remove_u3000()
    if TESTFLAG:
        vocab_file = '../vocab/bert-base-chinese.txt'
        full_tokenizer = FullTokenizer(vocab_file, do_lower_case=True)
        basic_tokenizer = BasicTokenizer(do_lower_case=True)
        sent = """
            目前　由　２３２　位　院士　（　Ｆｅｌｌｏｗ　及　Ｆｏｕｎｄｉｎｇ　Ｆｅｌｌｏｗ　）　，
            ６６　位　協院士　（　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ　）　２４　位　通信　院士　
            （　Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ｆｅｌｌｏｗ　）　及　２　位　通信　協院士　
            （　Ｃｏｒｒｅｓｐｏｎｄｉｎｇ　Ａｓｓｏｃｉａｔｅ　Ｆｅｌｌｏｗ　）　組成　
            （　不　包括　一九九四年　當選　者　）　，
            """
        print(preprocess2dict(sent, 'BIO', full_tokenizer, basic_tokenizer))
        print(preprocess2dict(sent, 'BMES', full_tokenizer, basic_tokenizer))
        print(preprocess2dict(sent, 'BMES', full_tokenizer, basic_tokenizer))
        sent = '目前 由 ２３２ 位 院士 （ Ｆｅｌｌｏｗ 及 Ｆｏｕｎｄｉｎｇ Ｆｅｌｌｏｗ ） ，'
        print(preprocess2dict(sent, 'BIO', full_tokenizer, basic_tokenizer))
        print(preprocess2dict(sent, 'BMES', full_tokenizer, basic_tokenizer))

        infile = 'tmp_output.txt'
        outfile = 'tmp_outBMES.txt'
        #gen_data(infile, outfile, 'BMES')

        outfile = 'tmp_outBIO.txt'
        gen_data(infile, outfile, 'BIO')

        infile = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/cws/cityu_test.tsv'
        outfile = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/cws/BIO/cityu_test.tsv'
        gen_data(infile, outfile, 'BIO')
    else:
        batch_gendata()
