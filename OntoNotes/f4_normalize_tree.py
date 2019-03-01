from collections import Counter
from nltk import Tree
import sys
import argparse
import os
import time
from tqdm import tqdm

def oneline(t):
    return t._pformat_flat(nodesep='', parens='()', quotes=False)


def normalize_trees(tree_file, interpreter, u=True):
    tree_file=os.path.realpath(tree_file)
    norm_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'normalize_tree')
    script_file = os.path.join(norm_dir, 'binarize.py')
    rule_file = os.path.join(norm_dir, 'rule.txt')
    tmp_file = os.path.join(norm_dir, 'tmp.tree.txt')

    cmd = f'{interpreter} "{script_file}" {"-u" if u else ""} "{rule_file}" "{tree_file}" > "{tmp_file}"'

    os.system(cmd)
    with open(tmp_file, encoding='utf-8')as f:
        lines = f.readlines()
    os.remove(tmp_file)

    pro_map = {
        ' t ': ' ',
        ' s ': ' ',
        ' r ': ' ',
        ' l ': ' ',
        ' r* ': '-m ',
        ' l* ': '-m ',
    }
    ret_lines = []
    for line in lines:
        for r, s in pro_map.items():
            line = line.replace(r, s)
        line = oneline(Tree.fromstring(line)) + '\n'
        ret_lines.append(line)
    return ret_lines


def compare_change(ori_file, norm_file):
    stat = Counter()
    ori_lines = open(ori_file, encoding='utf-8').readlines()
    norm_lines = open(norm_file, encoding='utf-8').readlines()
    assert len(ori_lines) == len(norm_lines),[len(ori_lines),len(norm_lines)]
    for ori_line, norm_line in zip(ori_lines, norm_lines):
        if ori_line == norm_line:
            stat['unchanged'] += 1
        else:
            stat['changed'] += 1
    print(stat)


def main_normalize_tree(interpreter, in_dir, out_dir):
    for part in tqdm(['train', 'dev', 'test']):
        file = os.path.join(in_dir, f'{part}.corrected.parse')
        out_file = os.path.join(out_dir, f'{part}.norm-tree.parse')

        ret_lines = normalize_trees(file, interpreter, u=True)

        with open(out_file, 'w', encoding='utf-8')as f:
            f.writelines(ret_lines)
        print(part)
        compare_change(file, out_file)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        PYTHON = r'/anaconda3/envs/haiqin370/bin/python'
        PRE_DIR = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/'
        IN_DIR = PRE_DIR+'proc_data/3.correct-span'
        OUT_DIR = PRE_DIR+'proc_data/4.norm-tree'
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--python')
        parser.add_argument('-i', '--input-dir')
        parser.add_argument('-o', '--output-dir')
        args = parser.parse_args()
        PYTHON = args.python
        IN_DIR = args.input_dir
        OUT_DIR = args.output_dir

    if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)

    main_normalize_tree(
        interpreter=PYTHON,
        in_dir=IN_DIR,
        out_dir=OUT_DIR)
