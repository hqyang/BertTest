#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 11:21 2019-01-30 
@author: haiqinyang

Feature: 

Scenario: 
"""


def test_BertCRF_constructor():
    from src.BERT.modeling import BertCRF
    from collections import namedtuple

    test_input_args = namedtuple("test_input_args", "bert_model cache_dir")
    test_input_args.bert_model = '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/bert-base-chinese.tar.gz'
    test_input_args.cache_dir = '/Users/haiqinyang/Downloads/codes/pytorch-pretrained-BERT-master/models/'

    model = BertCRF(test_input_args, 4)


def test_BasicTokenizer():
    from src.tokenization import BasicTokenizer
    # prove processing English and Chinese characters
    basic_tokenizer = BasicTokenizer(do_lower_case=True)
    text = 'beauty一百分\n Beauty 一百分!!'
    print(basic_tokenizer.tokenize(text))

def test_pandas_drop():
    import pandas as pd
    import os

    data_dir = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/proc_data/final_data'
    df = pd.read_csv(os.path.join(data_dir, "test_code.tsv"), sep='\t')

    # full_pos (chunk), ner, seg, text
    df.drop(['full_pos', 'ner'], axis=1)

    # change name to tag for consistently processing
    df.rename(columns={'seg': 'label'}, inplace=True)

    print(len(df.full_pos))

def test_pandas_drop_syn():
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(np.arange(12).reshape(3,4), columns=['A', 'B', 'C', 'D'])
    print(df)

    # need parameter inplace=True
    df.drop(['B', 'C'], axis=1, inplace=True)
    print(df)


if __name__ == '__main__':
    #test_BertCRF_constructor()
    #test_BasicTokenizer()
    test_pandas_drop()

    test_pandas_drop_syn()
