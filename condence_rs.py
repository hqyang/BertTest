#!/anaconda3/envs/haiqin370/bin/python3 
# -*- coding: utf-8 -*- 
#import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import pdb

def getScore(infile):
    outScore = []
    with open(infile) as fi: 
        for line in fi:
            outVal = re.split(':|,', line)
            idx_range = range(1, len(outVal), 2)
            score = [float(outVal[i]) for i in idx_range]
            outScore.append(score)

    return outScore    

if __name__=='__main__':
    num_hidden_layers = [3, 6]
    #num_epoches = [1, 5, 10, 15]
    num_epoches = [15]
    num_train_size = [64]
    types = ['train', 'dev', 'test']

    pre_out_dir = './tmp_2019_2_22/ontonotes/'
    
    idx = 0
    for nhl in num_hidden_layers:
        idx += 1
        for ne in num_epoches:
            xl = [v+1 for v in range(ne)]
            for nts in num_train_size:
                out_dir = pre_out_dir + 'nhl' + str(nhl) + '_nte' + str(ne) + '_nbs' + str(nts) + '/out/'
                for type in types:
                   infile = out_dir + type + '_eval_rs.txt'
                   #pdb.set_trace()
                   score = getScore(infile)
                   nscore = np.array(score)

                   outdir = pre_out_dir+'out/'
                   os.makedirs(outdir, exist_ok=True)     
                   with open(outdir+'condense_rs.txt', 'a+') as fout:
                       fout.write('\n'+type+' results at No. of hidden layers: {:d}, No. of training epoches: {:d}, No. of training size: {:d}\n'.format(nhl, ne, nts))

                       with open(infile) as fin:
                           for line in fin:
                              fout.write(line)

                   plt.plot(xl, nscore[:,1], 'bo')
                   plt.axis()
                   #nps[idx,:] = np.array(score)
