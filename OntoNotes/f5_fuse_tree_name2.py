import sys
import argparse
import os
from nltk import Tree
import re
from tqdm import tqdm


def zip_lines(f1, f2):
    ls1 = open(f1, encoding='utf-8').readlines()
    ls2 = open(f2, encoding='utf-8').readlines()
    assert len(ls1) == len(ls2)
    for l1, l2 in tqdm(list(zip(ls1, ls2))):
        yield l1, l2


def get_name_spans(line):
    line = line.strip('\n')
    ner_chunks = re.findall('<.*?>\S+|\S+', line)
    sent = re.sub('<.*?>', '', line).replace(' ', '')

    span2name = {}
    cur = 1
    in_name = False
    for chunk in ner_chunks:
        next_cur = cur + len(re.sub('</?ENAME.*?>', '', chunk))
        if chunk.startswith('<ENAMEX'):
            etype = re.findall('<ENAMEX TYPE="(.*?)"', chunk)[0]
            if etype in ['PERSON', 'LOC', 'GPE', 'ORG']:
                S = cur
                in_name = True
        if in_name and chunk.endswith('</ENAMEX>'):
            E = next_cur - 1
            span = (S, E)
            word = sent[S - 1:E]
            span2name[span] = (etype, word)
            in_name = False
        cur = next_cur
    return span2name


def add_span(t, i=1):
    if isinstance(t, Tree):
        S = i
        for c in t:
            add_span(c, i=i)
            if isinstance(c, Tree):
                _, j = c.span
            else:
                j = i + len(c) - 1
            i = j + 1
        t.span = (S, j)
        return t
    else:
        return t


def post_order(t, terminal=False):
    for c in t:
        if isinstance(c, Tree):
            for _ in post_order(c):
                yield _
        else:
            if terminal:
                yield c
    yield t


def reset_list(old, new):
    # 将old指向的list内容重置为new
    for _ in range(len(old)):
        del old[0]
    old.extend(new)


def oneline(t):
    return t._pformat_flat(nodesep='', parens='()', quotes=False)


def check_no_overlap_span(sub, span2sub):
    if sub.label().startswith('NER'):
        # NER内的单叉不用管
        assert sub.height() == 3, sub
        return True
    else:
        return sub.span not in span2sub


def fuse_parse_and_name2(parse_line, name_line):
    """
    将name信息融合入parse树里。name对应span合为一个词，词性为NER-TYPE。
    1. 找出所有实体span
    2. 找出span对应结点（完整对应）
    3. 修改结点label
    4. 提取结点下的字，合成
    :param parse_line:
    :param name_line:
    :return:
    """
    span2name = get_name_spans(name_line)

    ret_line = parse_line
    for span, (etype, word) in span2name.items():
        t = Tree.fromstring(ret_line)
        t = add_span(t)
        span2sub = {}
        for sub in post_order(t):
            # assert sub.span not in span2sub, '存在重合，可能有单叉'
            assert check_no_overlap_span(sub, span2sub), '存在重合，可能有单叉'
            span2sub[sub.span] = sub
        sub = span2sub[span]
        etype = 'PER' if etype == 'PERSON' else etype

        if not isinstance(sub[0], str):
            # NER对应成分
            assert sub.height() > 2
            sub.set_label(f'NER-{etype}')

            sub_words = []
            for w, pos in sub.pos():
                sub_words.append(Tree(pos, [w]))
            reset_list(sub, sub_words)
            assert ''.join(sub.leaves()) == word

            # sub_sent = ''.join(sub.leaves())
            # assert sub_sent == word
            # reset_list(sub, [sub_sent])
        else:
            # NER对应POS
            assert sub.height() == 2
            assert sub[0] == word
            pos = sub.label()
            sub.set_label(f'NER-{etype}')
            reset_list(sub, [Tree(pos, [word])])

        ret_line = oneline(t) + '\n'
    return ret_line


def main_fuse(parse_dir, name_dir, out_dir):
    for part in ['train', 'dev', 'test']:
        parse_file = os.path.join(parse_dir, f'{part}.norm-tree.parse')
        name_file = os.path.join(name_dir, f'{part}.name')
        out_file = os.path.join(out_dir, f'{part}.fuse.parse')

        lines = []
        for parse_line, name_line in zip_lines(parse_file, name_file):
            fuse_line = fuse_parse_and_name2(parse_line, name_line)
            lines.append(fuse_line)
        with open(out_file, 'w', encoding='utf-8')as f:
            f.writelines(lines)
        print(part)


if __name__ == '__main__':
    # 实体作为结点，内部保留POS

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--parse-dir')
        parser.add_argument('-n', '--name-dir')
        parser.add_argument('-o', '--output-dir')
        args = parser.parse_args()
        PARSE_DIR = args.input_dir
        NAME_DIR = args.name_dir
        OUT_DIR = args.output_dir
    else:
        PRE_DIR = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/'
        PARSE_DIR = PRE_DIR+'proc_data/4.norm-tree'
        NAME_DIR = PRE_DIR+'proc_data/1.name-parse'
        OUT_DIR = PRE_DIR+'proc_data/5.fuse-tree2'

    if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)

    main_fuse(
        parse_dir=PARSE_DIR,
        name_dir=NAME_DIR,
        out_dir=OUT_DIR)
