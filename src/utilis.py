#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 19:10 2019-02-01 
@author: haiqinyang

Feature: 

Scenario: 
"""
from  __future__ import unicode_literals
import os
import re
import pandas as pd
import torch
import numpy as np
from .config import UNK_TOKEN, PUNC_TOKENS
from .preprocess import dataset_to_dataloader, OntoNotesDataset

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def get_dataset_and_dataloader(processor, args, training=True, type='train'):
    dataset = OntoNotesDataset(processor, args.data_dir, args.vocab_file,
                                 args.max_seq_length, training=training, type=type)
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
        eval_dataset, eval_dataloader = get_dataset_and_dataloader(processor, args, training=False, type=part)
        eval_dataloaders[part] = eval_dataloader

    return eval_dataloaders


# copy from https://github.com/supercoderhawk/DNN_CWS/blob/master/utils.py
def strQ2B(ustring):
    '''全角转半角'''
    rstring = ''

    for uchar in ustring:
        inside_code = ord(uchar)

        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

    rstring += chr(inside_code)

    return rstring


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


def count_words_in_part(words, part, data_stat, data_count, store_dicts, store_chi_chars):
    len_words = len(words)
    data_count[part, 'num_words_in_sent'].append(len_words)

    if len_words >= 126:
        data_stat[part, 'great126'] += 1
    elif len_words >= 62:
        data_stat[part, 'great62'] += 1
    elif len_words >= 30:
        data_stat[part, 'great30'] += 1
    else:
        data_stat[part, 'less30'] += 1

    for word in words:
        len_word = len(word)
        data_count[part, 'num_chars_in_word'].append(len_word)

        if check_english_words(word):
            data_stat[part, 'total_eng_words'] += 1
            store_dicts[part, 'eng'].add(word) # store unique english words
            data_stat[part, 'total_eng_chars'] += len_word
        elif is_numeric(word):
            data_stat[part, 'total_dig_words'] += 1
            store_dicts[part, 'dig'].add(word)
            data_stat[part, 'total_dig_chars'] += len_word
        else:  # non-English word/non-digit
            data_stat[part, 'total_chi_words'] += 1
            store_dicts[part, 'chi'].add(word)
            data_stat[part, 'total_chi_chars'] += len_word

            for w in word:
                store_chi_chars[part].add(w)


def count_data_stat_in_part(data_count, part, store_dicts, store_chi_chars, data_stat):
    np_sents = np.array(data_count[part, 'num_words_in_sent'])
    data_stat[part, 'total_words'] = np_sents.sum()
    data_stat[part, 'max_words_per_sent'] = np_sents.max()
    data_stat[part, 'min_words_per_sent'] = np_sents.min()
    data_stat[part, 'mean_words_per_sent'] = np_sents.mean()

    np_words = np.array(data_count[part, 'num_chars_in_word'])
    data_stat[part, 'max_chars_per_word'] = np_words.max()
    data_stat[part, 'min_chars_per_word'] = np_words.min()
    data_stat[part, 'mean_chars_per_word'] = np_words.mean()

    data_stat[part, 'unique_eng_words'] = len(store_dicts[part, 'eng'])
    data_stat[part, 'unique_chi_words'] = len(store_dicts[part, 'chi'])
    data_stat[part, 'unique_dig_words'] = len(store_dicts[part, 'dig'])
    data_stat[part, 'unique_words'] = len(store_dicts[part, 'eng']) + len(store_dicts[part, 'chi'])\
                                      + len(store_dicts[part, 'dig'])

    data_stat[part, 'unique_chi_chars'] = len(store_chi_chars[part])
    data_stat[part, 'total_chars'] = data_stat[part, 'total_chi_chars'] + data_stat[part, 'total_eng_chars'] \
                                    + data_stat[part, 'total_dig_chars']


def savestat2file(data_stat, out_file, parts):
    if 0:
        with open(out_file, 'a', encoding='utf-8') as f:
            f.write('\n')
            for part in parts:
                f.write('{:s}: num_lines: {:d}\t'.format(part, data_stat[part, 'line']))
            f.write('\n')

    for part in parts:
        with open(out_file, 'a', encoding='utf-8') as f:
            f.write('Type: {:s}, num_lines: {:d}\n'.format(part, data_stat[part, 'line']))

            f.write('Unique words: {:d}, num_chi: {:d}, num_eng: {:d}, num_digit: {:d}\n'.format(
                data_stat[part, 'unique_words'], data_stat[part, 'unique_chi_words'],
                data_stat[part, 'unique_eng_words'], data_stat[part, 'unique_dig_words']))

            f.write('Unique Chinese chars: {:d}\n'.format(
                data_stat[part, 'unique_chi_chars']))

            f.write('Total words: {:d}, total Chi. words: {:d}, total Eng. words: {:d}, total digit: {:d}\n'.format(
                data_stat[part, 'total_words'], data_stat[part, 'total_chi_words'],
                data_stat[part, 'total_eng_words'], data_stat[part, 'total_dig_words']))

            f.write('Total chars: {:d}, total Chi. chars: {:d}, total Eng. chars: {:d}, total digit chars: {:d}\n'.format(
                data_stat[part, 'total_chars'], data_stat[part, 'total_chi_chars'],
                data_stat[part, 'total_eng_chars'], data_stat[part, 'total_dig_chars']))

            f.write('Per sent.: max. words: {:d}, min. words: {:d}, mean words: {:.2f}\n'.format(
                data_stat[part, 'max_words_per_sent'], data_stat[part, 'min_words_per_sent'],
                data_stat[part, 'mean_words_per_sent']))

            f.write('Per word.: max. chars: {:d}, min. chars: {:d}, mean chars: {:.3f}\n'.format(
                data_stat[part, 'max_chars_per_word'], data_stat[part, 'min_chars_per_word'],
                data_stat[part, 'mean_chars_per_word']))

            f.write('num (>126): {:d}, num (126<>62): {:d}, num (62<>30): {:d}, num (<30): {:d}\n'.format(
                data_stat[part, 'great126'], data_stat[part, 'great62'],
                data_stat[part, 'great30'], data_stat[part, 'less30']))

            f.write('\n')

    with open(out_file, 'a', encoding='utf-8') as f:
        if 'dev' in parts:
            f.write(
                'num_oov_dev: {:d}, num_oov_test: {:d}, ratio_oov_dev: {:.3f}, ratio_oov_test: {:.3f}\n\n'.format(
                    data_stat['oov_dev'], data_stat['oov_test'], data_stat['ratio_oov_dev'],
                    data_stat['ratio_oov_test']))

            #f.write('num_oov_char_dev: {:d}, num_oov_char_test: {:d}'.format(
            #    data_stat['oov_char_dev'], data_stat['oov_char_test']))
        else:
            f.write('num_oov_test: {:d}, ratio_oov_test: {:.3f}\n\n'.format(
                data_stat['oov_test'], data_stat['ratio_oov_test']))

            #f.write('num_oov_char_test: {:d}'.format(data_stat['oov_char_test']))


def savewords(store_words, out_dir, parts, pre):
    for part in parts:
        out_file = os.path.join(out_dir, pre + '_' + part+'_words.txt')

        categories = ['chi', 'dig', 'eng']

        for cat_gor in categories:
            with open(out_file, 'w', encoding='utf-8') as f:
                for ch in store_words[part, cat_gor]:
                    f.writelines(ch+'\n')


def savediffwords(store_words, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        for word in store_words:
            f.write(word+'\n')


def escape(text):
    '''html转义'''
    text = (text.replace("&quot;", "\"").replace("&ldquo;", "“").replace("&rdquo;", "”")
          .replace("&middot;", "·").replace("&#8217;", "’").replace("&#8220;", "“")
          .replace("&#8221;", "\”").replace("&#8212;", "——").replace("&hellip;", "…")
          .replace("&#8226;", "·").replace("&#40;", "(").replace("&#41;", ")")
          .replace("&#183;", "·").replace("&amp;", "&").replace("&bull;", "·")
          .replace("&lt;", "<").replace("&#60;", "<").replace("&gt;", ">")
          .replace("&#62;", ">").replace("&nbsp;", " ").replace("&#160;", " ")
          .replace("&tilde;", "~").replace("&mdash;", "—").replace("&copy;", "@")
          .replace("&#169;", "@").replace("♂", "").replace("\r\n|\r", "\n").replace('&nbsp', ' '))
    return text


def read_sogou_report():
    base = 'Reduced/'
    types = os.listdir(base)
    sentences = []
    count = 0
    index = 0
    for type in types:
    # type = 'C000008'
        docs = os.listdir(base + type)

        for doc in docs:
            file = None
            try:
                file = open(base + type + '/' + doc, 'r', encoding='gbk')
                content = escape(strQ2B(file.read())).replace(r'\s', '').replace(r'\n\d+\n', '')
                lines = re.split(r'\n', re.sub(r'[ \t\f]+', r'', content))

                for line in lines:
                    sentences.extend(line.split('。'))
                # break
                file.close()
            except UnicodeDecodeError as e:
                count += 1
                file.close()
                # sentences.append(content)

    return sentences


def idx_to_tag(tag_to_idx, mode='BMES'):
    idx_to_chunk_tag = {}
    for key, idx in tag_to_idx.items():
        idx_to_chunk_tag[idx] = key

    return idx_to_chunk_tag

def get_Ontonotes(data_dir, type='train'):
    """See base class."""
    df = pd.read_csv(os.path.join(data_dir, type+".tsv"), sep='\t')

    # full_pos (chunk), ner, seg, text
    # need parameter inplace=True
    df.drop(columns=['bert_ner', 'bert_seg', 'full_pos', 'src_ner'], inplace=True)

    # change name to tag for consistently processing
    df.rename(columns={'src_seg': 'label'}, inplace=True)

    return df

def load_4CWS(infile):
    """See base class."""
    df = pd.read_csv(infile, sep='\t')

    # full_pos (chunk), ner, seg, text
    # need parameter inplace=True
    #df.drop(columns=['bert_ner', 'bert_seg', 'full_pos', 'src_ner'], inplace=True)
    df.drop(columns=['bert_seg'], inplace=True)

    # change name to tag for consistently processing
    df.rename(columns={'src_seg': 'label'}, inplace=True)

    return df


def convertList2BMES(rs):
    # rs: a list
    outStr = ''
    for i, word in enumerate(rs.__iter__()):
        len_word = len(word)
        if len_word==1 or check_english_words(word):
            seg_gt = 'S '
        else: # Chinese word with multiple chars or numerical values
            seg_gt = 'B ' + 'M ' * (len_word - 2) + 'E '
        outStr += seg_gt

        if i==len(rs)-1: # remove the additional space
            outStr = outStr[:-1]

    return outStr

def convertList2BIO(rs): # full_tokenizer
    # rs: a list
    outStr = ''
    for i, word in enumerate(rs.__iter__()):
        len_word = len(word)
        if len_word==1 or check_english_words(word):
            seg_gt = 'O,'
        else: # Chinese word with multiple chars or numerical values
            seg_gt = 'B,' + 'I,' * (len_word - 1)

        outStr += seg_gt

        #if i==len(rs)-1: # remove the additional space
        #    outStr = outStr[:-1]

    return outStr

def convertList2BIOwithComma(rs):
    # rs: a list
    outStr = ''
    for i, word in enumerate(rs.__iter__()):
        word = word.replace(UNK_TOKEN, ' ') # tackle unknown tokens
        len_word = len(word)
        if len_word==1 or check_english_words(word):
            seg_gt = 'O,'
        else: # Chinese word with multiple chars or numerical values
            seg_gt = 'B,' + 'I,' * (len_word - 1)

        outStr += seg_gt
    return outStr

def BMES2BIO(text):
    sOut = text
    sOut = sOut.replace('M', 'I')
    sOut = sOut.replace('E', 'I')
    sOut = sOut.replace('S', 'O')

    return sOut

def space2Comma(text):
    sOut = text
    sOut = sOut.replace(' ', ',')
    if sOut[-1]!=',':
        sOut += ','

    return sOut


def export_stat_list(n, param):
    a = param.view(-1)

    return {'name': n, 'max': '{:.6f}'.format(torch.max(a).item()), 'min': '{:.6f}'.format(torch.min(a).item()), \
            'mean': '{:.6f}'.format(torch.mean(a).item()), 'std': '{:.6f}'.format(torch.std(a).item()), \
            'median': '{:.6f}'.format(torch.median(a).item()) }


def save_model(model, fo='tmp.tsv'):
    data_all = [export_stat_list(n, param) for n, param in model.named_parameters()]

    import pandas as pd
    df = pd.DataFrame(data_all)
    # separate with \t
    df.to_csv(fo, sep='\t', encoding='utf-8', index=False,  header=True, \
              columns=['name', 'max', 'min', 'mean', 'median', 'std'])

    print('Finish writing models data to ' + fo + '!')


RE_SPECIAL_TOKENS = ['.', '^', '$', '*', '+', '?', '{', '}', '\\', '[', ']', '|', '(', ')']


def findretokens(text):
    # first output indicates whether there is special tokens
    # second output indicates whether it is only special token

    for st in RE_SPECIAL_TOKENS:
        if st in text:
            return True, len(st)==len(text)
    return False, False


def setspacefortext(strOut, used_idx, text):
    # strOut: xxxabcxxx
    # text: abc
    # strOut: xxx abc xxx
    idx_obj = findtext(text, strOut[used_idx:].lower())

    if idx_obj:
        start_idx, end_idx = idx_obj.span()
        start_idx += used_idx
        end_idx += used_idx
        strOut = strOut[:start_idx] + ' ' + strOut[start_idx:end_idx] + ' ' + strOut[end_idx:]
        used_idx = end_idx + 2

    return strOut, used_idx


def findtext(strIn, text):
    ftoken, onlytoken = findretokens(text)
    if not ftoken:
        idx_obj = re.search(text, strIn.lower())
    else: # find special token
        if onlytoken: # only one token
            text = '\\' + text
        else: # text consists of chars with special tokens, e.g., [, (, or ), ...
            text = handle_special_tokens(text)
        idx_obj = re.search(text, strIn.lower())
    return idx_obj


def setnospacefortext(strOut, used_idx, text):
    ftoken, onlytoken = findretokens(text)
    if not ftoken:
        idx_obj = re.search(text, strOut[used_idx:].lower())
    else: # find special token
        if onlytoken: # only one token
            text = '\\' + text
        else: # text consists of chars with special tokens, e.g., [, (, or ), ...
            text = handle_special_tokens(text)
        idx_obj = re.search(text, strOut[used_idx:].lower())

    if idx_obj:
        start_idx, end_idx = idx_obj.span()
        start_idx += used_idx
        end_idx += used_idx
        strOut = strOut[:start_idx] + strOut[start_idx:end_idx] + strOut[end_idx:]
        used_idx = end_idx

    return strOut, used_idx


def handle_special_tokens(text):
    for token in RE_SPECIAL_TOKENS:
        if token in text:
            text = text.replace(token, '\\'+token)

    return text

# original_str = '单枪匹马逛英国——伦敦篇。伦敦就是这个样子初次来到这个“老牌资本主义”的“雾都“，就像回到了上海，一幢幢不高的小楼，显得异常陈旧，很多楼房被数百年烟尘熏的就像被刷了一层黑色的油漆，油光锃亮，如果不是旁边的楼房正在清洗，很难让人相信如今的伦敦是饱经污染沧桑后及时刹车的高手，因为一座现代化的国际大都市也是有不少楼房是黑色的呢，黑色显得凝重、高雅，但是绝对不能靠油烟去熏……堵车，是所有大都市的通病，虽然不足为怪，但是，1988年的北京还没有那么多的车，也没有全城大堵车的现象，有的是刚刚开始的“靠油烟和汽车的尾气烟熏火燎美丽的古城”，有谁能够想到，短短的十年，北京就气喘吁吁的追赶上了伦敦，没有一条洁净的河流，没有清新的空气，有的是让人窒息的空气污染…….以及，让人始料未及的全城大堵车。如果，我们那些负责城市建设规划的先生们，在国外，不只只是游山玩水的话，带回别人的教训、总结别人的经验的话，我们这个被穷祖先毁的“一塌糊涂”的脆弱的生态环境也不会再经受20世纪90年代的现代化的大污染了。但是，伦敦是一座改过自新的城市，人家痛定思痛，紧急刹车，及时的治理了污染，我们在泰吾士河里可以看到鱼儿在自由的翻滚，天空湛蓝，翠绿的草地与兰天辉映着，一片“污染大战”后的和平景象'
# str_with_uknown_tokens = '单枪匹马逛英国[UNK][UNK]伦敦篇。伦敦就是这个样子初次来到这个[UNK]老牌资本主义[UNK]的[UNK]雾都[UNK]，就像回到了上海，一幢幢不高的小楼，显得异常陈旧，很多楼房被数百年烟尘熏的就像被刷了一层黑色的油漆，油光[UNK]亮，如果不是旁边的楼房正在清洗，很难让人相信如今的伦敦是饱经污染沧桑后及时刹车的高手，因为一座现代化的国际大都市也是有不少楼房是黑色的呢，黑色显得凝重、高雅，但是绝对不能靠油烟去熏[UNK][UNK]堵车，是所有大都市的通病，虽然不足为怪，但是，1988年的北京还没有那么多的车，也没有全城大堵车的现象，有的是刚刚开始的[UNK]靠油烟和汽车的尾气烟熏火燎美丽的古城[UNK]，有谁能够想到，短短的十年，北京就气喘吁吁的追赶上了伦敦，没有一条洁净的河流，没有清新的空气，有的是让人窒息的空气污染[UNK][UNK].以及，让人始料未及的全城大堵车。如果，我们那些负责城市建设规划的先生们，在国外，不只只是游山玩水的话，带回别人的教训、总结别人的经验的话，我们这个被穷祖先毁的[UNK]一塌糊涂[UNK]的脆弱的生态环境也不会再经受20世纪90年代的现代化的大污染了。但是，伦敦是一座改过自新的城市，人家痛定思痛，紧急刹车，及时的治理了污染，我们在泰吾士河里可以看到鱼儿在自由的翻滚，天空湛蓝，翠绿的草地与兰天辉映着，一片[UNK]污染大战[UNK]后的和平景象'
def restore_unknown_tokens(original_str, str_with_unknown_tokens):
    text_ls = str_with_unknown_tokens.split()

    strOut = original_str
    used_idx = 0

    for text in text_ls:
        if UNK_TOKEN not in text:
            if '[unused1]' in text:
                if len(text)>len('[unused1]'):
                    idx_ts = re.search('unused1', text)
                    s_idx, e_idx = idx_ts.span()

                    prestr = text[:s_idx-1]
                    poststr = text[e_idx+1:]

                    strOut, used_idx = setnospacefortext(strOut, used_idx, prestr)
                    strOut, used_idx = setnospacefortext(strOut, used_idx, poststr)
            else:
                strOut, used_idx = setspacefortext(strOut, used_idx, text)

    return strOut


def findtextdirect(strIn, start_idx, len_original_str, text, shift=0):
    text = text.lower()
    s_idx = strIn[start_idx:start_idx+shift+len(text)].find(text)

    if s_idx==-1:
        s_idx = strIn[start_idx:start_idx+2*shift+len(text)].find(text)

    if s_idx==-1: # different tokens after processing
        while len(strIn[start_idx]) == 0 and start_idx < len_original_str:
            start_idx += 1
        s_idx = 0

    return s_idx

def restore_unknown_tokens_with_pos(original_str, str_with_unknown_tokens, pos_str):
    s_str = original_str.lower()
    len_original_str = len(original_str)

    text_ls = str_with_unknown_tokens.split()
    pos_ls = pos_str.split()
    assert(len(text_ls) == len(pos_ls))

    pos_outstr = ''

    strOut = original_str
    used_idx = 0

    text_list = []
    pos_list = []
    ori_used_idx = 0

    unk_status = False
    shift = 0
    for i, text in enumerate(text_ls):
        #if i == 67:
        #    print(text)
        pos = pos_ls[i]

        if UNK_TOKEN not in text:
            if '[unused1]' in text:
                shift += text.count('[unused1]')
                tmp_text_list = text.split('[unused1]')
                tmp_text_list = [v for v in tmp_text_list if v]

                if len(tmp_text_list) > 0:
                    if unk_status:
                        unk_status = False
                        s_idx = findtextdirect(s_str, ori_used_idx, len_original_str, tmp_text_list[0], shift)

                        # append unknown token
                        text_list.append(original_str[ori_used_idx:ori_used_idx+s_idx])
                        pos_list.append(unk_pos)
                        ori_used_idx += s_idx

                    for v in tmp_text_list:
                        s_idx = findtextdirect(s_str, ori_used_idx, len_original_str, v, shift)

                        ori_used_idx += s_idx
                        e_idx = len(v)
                        text_list.append(original_str[ori_used_idx:ori_used_idx+e_idx])
                        pos_list.append(pos)

                        ori_used_idx += e_idx
            else: # normal text
                if unk_status: # previous is an unknown token
                    unk_status = False

                    s_idx = findtextdirect(s_str, ori_used_idx, len_original_str, text, shift)

                    pos_list.append(unk_pos)
                    e_idx = len(text)
                    text_list.append(original_str[ori_used_idx:ori_used_idx+s_idx])
                    ori_used_idx += s_idx
                    e_idx = ori_used_idx + e_idx
                else: # previous is a normal token
                    s_idx = findtextdirect(s_str, ori_used_idx, len_original_str, text, shift)
                    e_idx = ori_used_idx + s_idx + len(text)

                text_list.append(original_str[ori_used_idx:e_idx])
                ori_used_idx = e_idx

                pos_list.append(pos)
                shift = 0
        else: # unknown tokens exist, need to update shift
            shift += len(text)
            tmp_text_list = text.split(UNK_TOKEN)
            tmp_text_list = [v for v in tmp_text_list if v.replace('[unused1]', '')]

            # process tmp_text_list
            for v in tmp_text_list: # unknown tokens with word and
                s_idx = findtextdirect(s_str, ori_used_idx, len_original_str, v, shift)

                if s_idx > 0: # append unknown token
                    text_list.append(original_str[ori_used_idx:ori_used_idx+s_idx])

                    ori_used_idx += s_idx
                    # keep previous unknown tokens
                    if unk_status:
                        pos_list.append(unk_pos)
                        unk_status = False
                    else:
                        pos_list.append(pos)

                    text_list.append(original_str[ori_used_idx:ori_used_idx+len(v)])
                    pos_list.append(pos)

                    ori_used_idx += len(v)

            if text[-5:]==UNK_TOKEN: # [UNK] is in the end
                unk_status = True
                unk_pos = pos
            else:
                unk_status = False

    if unk_status:
        text_list.append(original_str[ori_used_idx:])
        pos_list.append(pos)

    return text_list, pos_list


def append_to_buff(processed_text_list, buff, append_text, len_max, merge_index):
    if len(buff) + len(append_text) > len_max:
        processed_text_list.append(buff)
        merge_index += 1
        buff = append_text
    else:
        buff += append_text
    return buff, merge_index


def split_text_by_punc(text):
    text = text.strip('\r\n')
    text = text.strip()
    text = text.replace('\u3000', ' ')
    text = "".join(text.split())

    #text_chunk_list = re.split('(。|，|：|\n|#)', text)
    text_chunk_list = re.split(PUNC_TOKENS, text)

    return text_chunk_list


def extract_pos(pos_list):
    result_pos_str = ''

    for pos in pos_list:
        if len(pos)<=2:
            pos_used = pos[0]
        else:
            pos_set = {}

            max_count = 0
            for pos_i in pos:
                if pos_i in pos_set:
                    pos_set[pos_i] += 1
                else:
                    pos_set[pos_i] = 1

                if pos_set[pos_i] > max_count:
                    max_count = pos_set[pos_i]
                    pos_used = pos_i

        result_pos_str += pos_used + ' '

    return result_pos_str
