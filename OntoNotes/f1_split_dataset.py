#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 6:26 PM 10/12/2018 
@author: haiqinyang

Feature: 

Scenario: 
"""

"""
指定chinese数据的annotation文件夹。
从中提取出train、dev、test的指定类型文件。
"""
import sys
import argparse
import os
import shutil
from collections import Counter, defaultdict
import re


def split_filename(filename):
    #print(filename)
    name, ext = os.path.splitext(filename)
    num = name.split('_')[-1]
    word = '_'.join(name.split('_')[:-1])
    return word, int(num), ext


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


def read_name_lines(file):
    lines = open(file, encoding='utf-8').readlines()
    return lines[1:-1]


def read_parse_lines(file):
    content = open(file, encoding='utf-8').read()
    trees = [t for t in content.split('\n\n') if len(t) > 0]
    trees = [t.replace('\n', ' ') for t in trees]
    trees = [re.sub(' {2,}', ' ', t) + '\n' for t in trees]
    return trees


def copy_onto_files(src_files, dst_dir):
    if not os.path.exists(dst_dir): os.mkdir(dst_dir)
    for file in src_files:
        folder, filename = os.path.split(file)
        domain, source = folder.split('/')[-1].split('\\')[-3:-1]
        dst = os.path.join(dst_dir, domain + '.' + source + '.' + filename)
        if os.path.exists(dst):
            print(dst)
        shutil.copyfile(file, dst)


DEV_DROP = [
    '（ 完 ）\n',
    '（ <ENAMEX TYPE="PERSON">杨桂林</ENAMEX> <ENAMEX TYPE="PERSON">常新华</ENAMEX> （ 完 ）\n',
    '<ENAMEX TYPE="ORG">新华社</ENAMEX> 记者 <ENAMEX TYPE="PERSON">郭庆华</ENAMEX> （ 完 ）\n']
TEST_DROP = ['（ 完 ）\n']


def main(anno_dir, out_dir, correct):
    """
    annatations/{domain}/{source}/{group}/{word}.{num}.{ext}
    :param anno_dir:Annotations目录
    :param out_dir:train.parse, train.name, ...
    :return:
    """
    drop = {'dev': set(DEV_DROP), 'test': set(TEST_DROP),'train':set()}
    stat = Counter()
    filename2lines = defaultdict(list)

    for domain in os.listdir(anno_dir):
        domain_dir = os.path.join(anno_dir, domain)
        if os.path.isfile(domain_dir): continue

        for source in os.listdir(domain_dir):
            source_dir = os.path.join(domain_dir, source)
            if os.path.isfile(source_dir): continue

            for group in os.listdir(source_dir):
                group_dir = os.path.join(source_dir, group)
                if os.path.isfile(group_dir): continue

                for filename in os.listdir(group_dir):
                    if filename == '.DS_Store': continue # add by haiqin for ignoring .DS_Store
                    part = which_part(domain, source, filename)
                    root, ext = os.path.splitext(filename)
                    if ext == '.name':
                        name_lines = read_name_lines(os.path.join(group_dir, filename))
                        parse_lines = read_parse_lines(os.path.join(group_dir, root + '.parse'))
                        assert len(name_lines) == len(parse_lines)
                        for name_line, parse_line in zip(name_lines, parse_lines):
                            if correct and name_line in drop[part]:continue
                            filename2lines[f'{part}.name'].append(name_line)
                            filename2lines[f'{part}.parse'].append(parse_line)
                            stat[part, 'line'] += 1
                        stat[part, 'file'] += 1

                print('finish g', group)
            print('finish s', source)
        print('finish d', domain)


    print(stat)

    if not os.path.exists(out_dir): os.mkdir(out_dir)
    for filename, lines in filename2lines.items():
        with open(os.path.join(out_dir, filename), 'w', encoding='utf-8')as f:
            f.writelines(lines)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        PRE_DIR = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/'
        ANNOTATION_DIR = PRE_DIR+'raw_data/annotations'
        OUTPUT_DIR = PRE_DIR+'proc_data/1.name-parse'
        CORRECT=True
    else:
        parser = argparse.ArgumentParser(description='从OntoNotes4中，根据Che2013的方案，分割数据集的文件')
        parser.add_argument('-i', '--input', help='annotations文件夹')
        parser.add_argument('-o', '--output', help='输出的文件夹')
        parser.add_argument('-c','--correct',default='true')
        args = parser.parse_args()
        ANNOTATION_DIR = args.input
        OUTPUT_DIR = args.output
        CORRECT={'true':True,'false':False}[args.correct]
    main(anno_dir=ANNOTATION_DIR, out_dir=OUTPUT_DIR,correct=CORRECT)
