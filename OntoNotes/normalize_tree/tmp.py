rmt='(TOP (IP (NP (PN 这)) (VP (VP (VC 是) (NP (CP (CP (IP (VP (VV 位于) (NP (NR 山西) (NR 阳泉)))) (DEC 的))) (NP (NR 狮脑山)))) (PU ，) (VP (VC 是) (NP (NP (NT 当年)) (DNP (NP (NR 正太) (NN 铁路)) (DEG 的)) (NP (NN 咽喉) (NN 要地))))) (PU 。)))'
cor='(TOP (IP (NP (PN 这)) (VP (VP (VC 是) (NP (CP (CP (IP (VP (VV 位于))))) (NP (NR 山西) (NR 阳泉) (DEC 的) (NR 狮脑山)))) (PU ，) (VP (VC 是) (NP (NP (NT 当年)) (DNP (NP (NR 正太) (NN 铁路)) (DEG 的)) (NP (NN 咽喉) (NN 要地))))) (PU 。)))'
mul='(IP (PN 这) (IP-m (VP (VP-m (VP (VC 是) (NP (VV 位于) (NP (NR 山西) (NP-m (NR 阳泉) (NP-m (DEC 的) (NR 狮脑山)))))) (PU ，)) (VP (VC 是) (NP (NT 当年) (NP-m (DNP (NP (NR 正太) (NN 铁路)) (DEG 的)) (NP (NN 咽喉) (NN 要地)))))) (PU 。)))'
from nltk import Tree

def draw(line):
    Tree.fromstring(line).draw()

#draw(rmt)
#draw(cor)
draw(mul)