#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 5:53 PM 7/12/2018 
@author: haiqinyang

Feature: Process OntoNotes

Scenario: 
"""

from nltk import Tree
import re

# line='(TOP (IP (NP (NP (NR 澳门)) (NP (NN 资产))) (VP (PP (P 在) (NP (NN 知识))))))'
# line='(TOP (IP (NP (PU 《) (NP (NR 华视)) (NP (NT 午间) (NN 新闻)) (PU 》)) (PU 。)))'

# line='(TOP (IP (ADVP (AD 以至)) (PP (P 在) (NP (DNP (LCP (IP (NP (QP (OD 第二) (CLP (M 次))) (NP (NN 世界) (NN 大战))) (VP (VV 结束) (QP (ADVP (AD 已经)) (QP (CD 五十多) (CLP (M 年)))))) (LC 后)) (DEG 的)) (NP (NT 今天)))) (PU ，) (NP (QP (CD 两)) (NP (NN 国))) (VP (ADVP (AD 仍)) (ADVP (AD 未)) (VP (VP (VV 签订) (NP (NN 和平) (NN 条约))) (PU ，) (VP (VV 实现) (IP (NP (ADJP (JJ 双边)) (NP (NN 关系))) (VP (ADVP (AD 完全)) (VP (VV 正常化))))))) (PU ，)))'
# t=Tree.fromstring(line)

# t=_item_tree_char(t,terminal=True)
# print(all_lattices(t))d
# t.draw()

#line='「 <ENAMEX TYPE="GPE">台湾</ENAMEX> 与 <ENAMEX TYPE="GPE">大陆</ENAMEX> 那 个 近 ？ 」 <ENAMEX TYPE="PERSON">谭志强</ENAMEX> 反 问 。'
#spans=get_name_spans(line)
#print(spans)


# t = Tree.fromstring('(TOP (IP (NP (PU 《) (NP (NR 华视)) (NP (NT 午间) (NN 新闻)) (PU 》)) (PU 。)))')
# t = add_span(t)
# t = add_lattice(t)
# t =attr_into_label(t,['span','lattices'])
# t.draw()

def _item_tree_char(t, i=1, terminal=False):
    """Convert to item tree.
    # >>> t = Tree.fromstring('(A aa (B bbb) cc)')
    # >>> _item_tree_char(t)
    Tree(('A', 1, 7), ['aa', Tree(('B', 3, 5), ['bbb']), 'cc'])
    1: start index, 7: total length of the rest tree
    3: start index, 5: total length of the rest tree
    """
    # if not isinstance(t, Tree):
    #     if terminals:
    #         return t, i, i+1
    #     else:
    #         return t
    # else:
    if isinstance(t, Tree):
        cs = []
        I = i
        for c in t:
            x = _item_tree_char(c, i=i, terminal=terminal)
            if isinstance(x, Tree):
                (_, _, j) = x.label()
            else:
                j = i + len(c) - 1
            i = j + 1
            cs.append(x)
        return Tree((t.label(), I, j), cs)
    else:
        if terminal:
            j = i + len(t) - 1
            return t, i, j
        else:
            return t

def get_ij(t):
    if isinstance(t, Tree):
        return t.label()[1], t.label()[2]
    else:
        return t[1], t[2] # some problems here: index out of range

def all_lattices(t):
    if isinstance(t, Tree):
        (_, i, j) = t.label()
        lattices = [(i, j)]
        for ix, sub in enumerate(t): # problem come from the definition of the structure
            S, _ = get_ij(sub)
            for jx, ssub in enumerate(t[ix:]):
                _, E = get_ij(ssub)
                lattices.append((S, E))
        for sub in t:
            lattices += all_lattices(sub)
        return lattices
    else:
        return [(t[1], t[2])]

def _extract_spans(line):
    ner_chunks = re.findall('<.*?>\S+|\S+', line)
    spans = []
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
            spans.append((S, E))
            in_name = False
        cur = next_cur
    return spans

def print_span(t):
    if isinstance(t, Tree):
        cs = []
        for c in t:
            cs.append(print_span(c))
        return Tree((t.label(), t.span, t.lattices), cs)
        # return Tree((t.label(), t.span), cs)
    else:
        return t

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

def add_span(t, i=1):
    """Convert to item tree.
    # >>> t = Tree.fromstring('(A aa (B bb) cc)')
    # >>> _item_tree_char(t)
    Tree(('A', 1, 6), ['aa', Tree(('B', 3, 4), ['bb']), 'cc'])
    """
    # if not isinstance(t, Tree):
    #     if terminals:
    #         return t, i, i+1
    #     else:
    #         return t
    # else:
    if isinstance(t, Tree):
        S = i
        for c in t:
            c = add_span(c, i=i)
            #add_span(c, i=i)
            if isinstance(c, Tree):
                _, j = c.span
            else:
                j = i + len(c) - 1
            i = j + 1
        t.span = (S, j)
        return t
    else:
        return t
    #
    # if isinstance(t, Tree):
    #     (_, i, j) = t.label()
    #     lattices = []
    #     for ix, sub in enumerate(t):
    #         S, _ = get_ij(sub)
    #         for jx, ssub in enumerate(t[ix:]):
    #             _, E = get_ij(ssub)
    #             lattices.append((S, E))
    #
    #     for sub in t:
    #         lattices += all_lattices(sub)
    #     return lattices
    # else:
    #     return [(t[1], t[2])]


def attr_into_label(t, names):
    if isinstance(t, Tree):
        cs = [attr_into_label(c, names) for c in t]
        values=[f'{name}:{getattr(t,name)}' for name in names]
        label=t.label()+'\t'+','.join(values)
        return Tree(label,cs)
    else:
        return t

NER_TYPES = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
             'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
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
            # https://spacy.io/api/annotation#named-entities NER types supported in OntoNotes 5.0
            # 'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
            # 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'

            # PERSON: People, including fictional
            # NORP: Nationalities or religious or political groups.
            # FAC: Buildings, airports, highways, bridges, etc.
            # ORG: Companies, agencies, institutions, etc.
            # GPE: Countries, cities, states.
            # LOC: Non-GPE locations, mountain ranges, bodies of water.
            # PRODUCT: Objects, vehicles, foods, etc. (Not services.)
            # EVENT: Named hurricanes, battles, wars, sports events, etc.
            # WORK_OF_ART: Titles of books, songs, etc.
            # LAW: Named documents made into laws.
            # LANGUAGE: Any named language.
            # DATE: Absolute or relative dates or periods.
            # TIME: Times smaller than a day.
            # PERCENT: Percentage, including "%".
            # MONEY: Monetary values, including unit.
            # QUANTITY: Measurements, as of weight or distance.
            # ORDINAL: "first", "second", etc.
            # CARDINAL: Numerals that do not fall under another type.

            if etype in NER_TYPES: #['PERSON', 'LOC', 'GPE', 'ORG']:
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


def oneline(t):
    return t._pformat_flat(nodesep='', parens='()', quotes=False)

def post_order(t, terminal=False):
    for c in t:
        if isinstance(c, Tree):
            for _ in post_order(c):
                yield _
        else:
            if terminal:
                yield c
    yield t
    

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
