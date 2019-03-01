import os
import re



interpreter=r'C:\Python27\python.exe'
#os.system(interpreter+' binarize.py rule.txt test.txt')
os.system(interpreter+' binarize.py -u rule.txt test.txt > tmp_input.txt')

line=open('tmp_input.txt',encoding='utf-8').readlines()[0]
os.remove('tmp_input.txt')


reserve_all={
    ' t ':'-t ',
    ' s ':'-s ',
    ' r ':'-r ',
    ' l ':'-l ',
    ' r* ':'-r* ',
    ' l* ':'-l* ',
}

only_new={
    ' t ':' ',
    ' s ':' ',
    ' r ':' ',
    ' l ':' ',
    ' r*':'-r',
    ' l*':'-l',
}

pro=dict(only_new)
pro[' r*']='-m'
pro[' l*']='-m'


res={}
res=pro
for r,s in res.items():
    line=line.replace(r,s)
    #line=re.sub(r,s,line)
print(line)
from nltk import Tree
Tree.fromstring(line).draw()
