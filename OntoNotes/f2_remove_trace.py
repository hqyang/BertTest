import os
import sys
import argparse

from nltk import Tree
def oneline(t):
    return t._pformat_flat(nodesep='', parens='()', quotes=False)


def remove_trace(t):
    """Remove trace constituents and coarsen the corresponding labels.

#    >>> t = Tree.fromstring('(S (S-TPC-2 (NP-SBJ (PRP (-LBR- -LBR-) a (-RBR- -RBR-)))) (VP (VBD b) (S (-NONE- *T*-2))) (. .))')
#    >>> print(remove_trace(t))
    (S (S (NP (PRP (-LBR- -LBR-) a (-RBR- -RBR-)))) (VP (VBD b)) (. .))

    """
    if is_terminal(t):
        assert isinstance(t, str), t
        return t
    label = t.label()
    if label == '-NONE-':  # trace
        return None
    if not label.startswith('-'):    # skip special labels like -LRB-
        label = label.split('-')[0]
        label = label.split('=')[0]
    new = filter(None, [remove_trace(c) for c in t])
    if not new:        # if no more children (and not originally a leaf)
        return None    # remove this item
    return Tree(label, new)


def is_terminal(d):
    return not isinstance(d, Tree)

def main(data_dir,out_dir):
    for file in os.listdir(data_dir):
        root,ext=os.path.splitext(file)
        if ext!='.parse':continue
        lines=[]
        for line in open(os.path.join(data_dir,file),encoding='utf-8'):
            t=Tree.fromstring(line)
            t=remove_trace(t)
            lines.append(oneline(t)+'\n')
        with open(os.path.join(out_dir,f'{root}.rm-trace.parse'),'w',encoding='utf-8')as f:
            f.writelines(lines)

if __name__ == '__main__':
    if len(sys.argv)==1:
        PRE_DIR = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/'
        DATA_DIR = PRE_DIR+'proc_data/1.name-parse'
        OUT_DIR = PRE_DIR+'proc_data/2.remove-trace'
    else:
        parser = argparse.ArgumentParser(description='对.parse文件中的树执行remove trace')
        parser.add_argument('-i', '--input-dir', help='输入数据所在文件夹，处理其中*.parse文件')
        parser.add_argument('-o', '--output-dir', help='输出文件夹，输出*.rm-trace.parse')
        args=parser.parse_args()
        DATA_DIR=args.input_dir
        OUT_DIR=args.ouput_dir

    if not os.path.exists(OUT_DIR):os.mkdir(OUT_DIR)

    main(data_dir=DATA_DIR, out_dir=OUT_DIR)
