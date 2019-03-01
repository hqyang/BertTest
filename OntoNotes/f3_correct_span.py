"""

修正lattice：
1. 对某NER的span，发现parse中无span但有lattice。
2. 对树中每个结点标记lattice集合。找到NER span的父节点。
3. 对子结点遍历，将span内的子结点归入一个结点

修正mismatch:
1. 对某NER的span，发现parse中无span也无lattice
2. 树中每个结点标记span
3. 找到开头词，置为整个name
4. 找到其他词，置为空格

"""
import sys
import argparse
import os
from nltk import Tree
import re
from collections import Counter

# =======================
# NER Utils
# =======================

def get_name_spans(line):
    line = line.strip('\n')
    ner_chunks = re.findall('<.*?>\S+|\S+', line)
    sent = re.sub('<.*?>', '', line).replace(' ', '')

    span2word = {}
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
            span2word[span] = word
            in_name = False
        cur = next_cur
    return span2word


# =======================
# Tree Utils
# =======================
def oneline(t):
    return t._pformat_flat(nodesep='', parens='()', quotes=False)


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


def gen_lattices(start, lens):
    lattices = []
    s = 0
    for ix, ilen in enumerate(lens):
        e = s
        for jlen in lens[ix:]:
            e += jlen
            lattices.append((start + s, start + e - 1))
        s += ilen
    return sorted(lattices)


def add_lattice(t):
    if isinstance(t, Tree):
        S, _ = t.span
        lens = []
        for c in t:
            if isinstance(c, Tree):
                lens.append(c.span[1] - c.span[0] + 1)
                add_lattice(c)
            else:
                lens.append(len(c))
        t.lattices = gen_lattices(S, lens)
    return t


def attr_into_label(t, names):
    if isinstance(t, Tree):
        cs = [attr_into_label(c, names) for c in t]
        values = [f'{name}:{getattr(t,name)}' for name in names]
        label = t.label() + '\t' + ','.join(values)
        return Tree(label, cs)
    else:
        return t


def lattices_in_tree(t):
    lattices = list(t.lattices)
    for c in t:
        if isinstance(c, Tree):
            lattices += lattices_in_tree(c)
    return lattices


def spans_in_tree(t):
    spans = [t.span]
    for c in t:
        if isinstance(c, Tree):
            spans += spans_in_tree(c)
    return spans


def post_order(t, terminal=False):
    for sub in t:
        if isinstance(sub, Tree):
            for _ in post_order(sub):
                yield _
        else:
            if terminal:
                yield sub
    yield t


def reset_list(old, new):
    for _ in range(len(old)):
        del old[0]
    old.extend(new)


def span_in(span, other):
    return span[0] >= other[0] and span[1] <= other[1]


def span_inter(span, other):
    ret = (other[0] <= span[0] <= other[1]) \
          or (other[0] <= span[1] <= other[1]) \
          or (span[0] <= other[0] <= span[1]) \
          or (span[0] <= other[1] <= span[1])
    return ret


def remove_span_in_tree(t, span):
    """
    去除所有在span内的结点
    :param t:
    :param span:
    :return:
    """
    cs = []
    for sub in t:
        if span_in(sub.span, span):
            continue
        elif span_inter(sub.span, span):
            sub = remove_span_in_tree(sub, span)
            cs.append(sub)
        else:
            cs.append(sub)
    reset_list(t, cs)
    return t


def find_min_include(t, span):
    assert span_in(span, t.span)
    for sub in t:
        if span_in(span, sub.span):
            return find_min_include(sub, span)
    return t


def get_span_lex(t, span):
    lexs = []
    for sub in t:
        if span_inter(sub.span, span):
            if isinstance(sub[0], str):
                lexs.append(sub)
            else:
                lexs.extend(get_span_lex(sub, span))
    return lexs


#
# line = '(TOP (IP (NP (PU 《) (NP (NR 华视)) (NP (NT 午间) (NN 新闻)) (PU 》)) (PU 。)))'
# t = Tree.fromstring(line)
# t = add_span(t)
# t = add_lattice(t)
# #t = remove_span_in_tree(t, [6, 9])
# print(get_span_lex(t,(2,6)))
# t.draw()
# exit()
#

# =======================
# Process Alg
# =======================


def aggregate_span(parse_line, name_span):
    """
    聚合多叉内的实体
    1. 解析出Tree，提取span和lattice
    2. 根据lattice找到实体对应的成分结点
    3. 重新构造该成分的子节点
    :param parse_line:
    :param name_span:
    :return:
    """
    name_span = tuple(name_span)
    t = Tree.fromstring(parse_line)
    t = add_span(t)
    t = add_lattice(t)

    mul_tree = None
    for tt in post_order(t):
        if name_span in tt.lattices:
            mul_tree = tt

    new_cs = []
    in_name = False
    name_cs = []
    for sub in mul_tree:
        if span_inter(name_span, sub.span):
            in_name = True
            name_cs.append(sub)
        else:
            if in_name:
                new_cs.append(Tree('NP', name_cs))
                in_name = False
            new_cs.append(sub)
    if in_name:
        new_cs.append(Tree('NP', name_cs))
    reset_list(mul_tree, new_cs)
    return oneline(t)


def aggregate_cross_span(parse_line, name_span):
    """
    聚合不在同一子树下的实体
    1. 找到包含实体的最小的成分
    2. 删除该成分下所有包含实体的子树
    3. 将实体作为整体填充到该成分下
    :param parse_line:
    :param name_span:
    :return:
    """
    name_span = tuple(name_span)
    t = Tree.fromstring(parse_line)
    t = add_span(t)
    t = add_lattice(t)
    min_t = find_min_include(t, name_span)
    lexs = get_span_lex(min_t, name_span)
    lex_t = Tree('NP', lexs)
    remove_span_in_tree(min_t, name_span)

    # name_span不应正好在min_t中，必然存在不属于name_span的子树
    cs = []
    no_lex = True
    for sub in min_t:
        if sub.span[1] >= name_span[1] and no_lex:
            cs.append(lex_t)
            no_lex = False
        cs.append(sub)
    if no_lex:
        cs.append(lex_t)
    reset_list(min_t, cs)
    ret = t
    return oneline(ret)


#line='(TOP (IP (NP (PN 这)) (VP (VP (VC 是) (NP (CP (CP (IP (VP (VV 位于) (NP (NR 山西) (NR 阳泉)))) (DEC 的))) (NP (NR 狮脑山)))) (PU ，) (VP (VC 是) (NP (NP (NT 当年)) (DNP (NP (NR 正太) (NN 铁路)) (DEG 的)) (NP (NN 咽喉) (NN 要地))))) (PU 。)))'
#
# # line = '(TOP (IP-Q (NP-SBJ (NN 台商) (NN 眷属)) (VP (ADVP-WH (AD 为何)) (ADVP (AD 不)) (VP (VV 愿) (VP (VV 来) (NP-OBJ (NN 大陆))))) (PU ？)))'
#line='(TOP (IP (NP (CP (CP (IP (VP (PP (P 由) (NP (NP (NR 联合国)) (NP (NP (NN 采购司)) (PU 、) (NP (NN 人口) (NN 资金会)) (PU 、) (NP (NN 儿童) (NN 资金会)) (PU 、) (NP (NN 项目) (NN 资金) (NN 办公室)) (ETC 等)) (QP (CD 四) (CLP (M 个))) (NP (NN 部门)))) (VP (VV 组成)))) (DEC 的))) (NP (NR 联合国)) (NP (NN 采购团))) (VP (PP (P 向) (NP (NP (NR 中国)) (ADJP (JJ 参展)) (NP (NN 企业)))) (ADVP (AD 详细)) (VP (VV 介绍) (AS 了) (NP (DNP (NP (NR 联合国)) (DEG 的)) (NP (NP (NN 采购) (NN 特点)) (PU 、) (NP (NN 渠道)) (PU 、) (NP (NN 项目)) (CC 和) (NP (NN 交易) (NN 形式)))))) (PU 。)))'
#  # line = aggregate_span(line, (13, 16))
#line = aggregate_cross_span(line, (2,7))
# #
# # # # t=Tree.fromstring(line)
# # # # t=add_lattice(add_span(t))
# # # # find_min_include(t,(11,12)).draw()
# # #
# Tree.fromstring(line).draw()
# exit()


def gen_lines(name_file, parse_file):
    for name_line, parse_line in zip(open(name_file, encoding='utf-8'), open(parse_file, encoding='utf-8')):
        yield name_line, parse_line


def correct(parse_line, name_line):
    """

    :param parse_line:
    :param name_line:
    :return:
    """
    t = Tree.fromstring(parse_line)
    t = add_span(t)
    t = add_lattice(t)
    spans = set(spans_in_tree(t))
    lattices = set(lattices_in_tree(t))

    tree_line = parse_line
    c=False
    for span, word in get_name_spans(name_line).items():
        has_span = span in spans
        has_lat = span in lattices
        if has_span:
            continue
        elif not has_span and has_lat:
            tree_line = aggregate_span(tree_line, span)
            c=True
        else:
            tree_line = aggregate_cross_span(tree_line, span)
            c=True
    if c:
        return tree_line
    else:
        return None


def main_correct_span(parse_dir, name_dir, out_dir):
    """
    根据NER修正parse的span。
    parse是remove-trace的
    NER是name-parse的

    有span：不处理
    无span，有lattice：lattice合并
    无span，无lattice：字符归入第一词，其他置空
    :param parse_dir:
    :return:
    """
    count=Counter()

    for part in ['train', 'dev', 'test']:
        parse_file = os.path.join(parse_dir, f'{part}.rm-trace.parse')
        name_file = os.path.join(name_dir, f'{part}.name')
        out_file = os.path.join(out_dir, f'{part}.corrected.parse')

        lines = []
        for name_line, parse_line in gen_lines(name_file, parse_file):
            print(name_line)
            correct_line = correct(parse_line, name_line)
            if correct_line is None:
                count[part,'no_co']+=1
                correct_line=parse_line
            else:
                count[part,'co']+=1
            lines.append(correct_line.strip('\n')+'\n')
        with open(out_file,'w',encoding='utf-8')as f:
            f.writelines(lines)
    print(count)



if __name__ == '__main__':
    # line = '(TOP (IP-Q (NP-SBJ (NN 台商) (NN 眷属)) (VP (ADVP-WH (AD 为何)) (ADVP (AD 不)) (VP (VV 愿) (VP (VV 来) (NP-OBJ (NN 大陆))))) (PU ？)))'
    # t = Tree.fromstring(line)
    # t = add_span(t)
    # t=add_lattice(t)
    # print(spans_in_tree(t))
    # s = attr_into_label(t, ['span'])  # ,'lattices'])
    # s.draw()
    # exit()

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input-dir')
        parser.add_argument('-n', '--name-dir')
        parser.add_argument('-o', '--output-dir')
        args = parser.parse_args()
        IN_DIR = args.input_dir
        NAME_DIR = args.name_dir
        OUT_DIR = args.output_dir
    else:
        PRE_DIR = '/Users/haiqinyang/Downloads/datasets/ontonotes-release-5.0/ontonote_data/'
        IN_DIR = PRE_DIR+'proc_data/2.remove-trace'
        NAME_DIR = PRE_DIR+'proc_data/1.name-parse'
        OUT_DIR = PRE_DIR+'proc_data/3.correct-span'

    if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)

    main_correct_span(
        parse_dir=IN_DIR,
        name_dir=NAME_DIR,
        out_dir=OUT_DIR)
