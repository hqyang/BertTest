# logic of processing raw annotations data

1. 1.split_dataset.py
# need to remove .DS_Store if there is
# need to justify the splitting of datasets

2. 2.remove_trace.py
# Remove trace constituents and coarsen the corresponding labels.
#  But the information is useful for coreference, e.g., '*pro*'

#    >>> t = Tree.fromstring('(S (S-TPC-2 (NP-SBJ (PRP (-LBR- -LBR-) a (-RBR- -RBR-)))) (VP (VBD b) (S (-NONE- *T*-2))) (. .))')
#    >>> print(remove_trace(t))
    (S (S (NP (PRP (-LBR- -LBR-) a (-RBR- -RBR-)))) (VP (VBD b)) (. .))

3. 3.correct_span.py
修正lattice：
1. 对某NER的span，发现parse中无span但有lattice。
2. 对树中每个结点标记lattice集合。找到NER span的父节点。
3. 对子结点遍历，将span内的子结点归入一个结点

修正mismatch:
1. 对某NER的span，发现parse中无span也无lattice
2. 树中每个结点标记span
3. 找到开头词，置为整个name
4. 找到其他词，置为空格

4. 4.normalize_tree.py
#

5. 5.fuse_tree_name2.py
将name信息融合入parse树里。name对应span合为一个词，词性为NER-TYPE。
1. 找出所有实体span
2. 找出span对应结点（完整对应）
3. 修改结点label
4. 提取结点下的字，合成
