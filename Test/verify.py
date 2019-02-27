#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 11:51 2019-02-22 
@author: haiqinyang

Feature: 

Scenario: 
"""

def calSize(H, vs, mpe, L):
    for l in L:
        # embedding: (vs+mpe)*H; # Query, Key, value: 3*H*H; Intermediate: 4*H *H; Pooler: H*H
        sz = (vs+mpe)*H + ((3+4)*H*H)*l + H*H
        print('# layer: '+str(l)+', #para: '+str(sz))


if __name__=='__main__':
    H = 768
    vs = 21128
    mpe = 512
    L = [3, 6, 12]

    num_model_para = calSize(H, vs, mpe, L)
