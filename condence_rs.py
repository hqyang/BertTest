#!/anaconda3/envs/haiqin370/bin/python3 
# -*- coding: utf-8 -*- 
#import pandas as pd
import numpy as np
import re
import os
import matplotlib
matplotlib.use("TkAgg")
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

def saveScore2File():
    num_hidden_layers = [3, 6]
    #num_epoches = [1, 5, 10, 15]
    num_epochs = [15]
    num_train_size = [64]
    types = ['train', 'dev', 'test']

    pre_out_dir = './tmp_2019_2_22/ontonotes/'

    for nhl in num_hidden_layers:
        for ne in num_epochs:
            for nts in num_train_size:
                out_dir = pre_out_dir + 'nhl' + str(nhl) + '_nte' + str(ne) + '_nbs' + str(nts) + '/out/'
                for type in types:
                    infile = out_dir + type + '_eval_rs.txt'
                    #pdb.set_trace()
                    score = getScore(infile)
                    nscore = np.array(score)

                    outdir = pre_out_dir+'out/'
                    os.makedirs(outdir, exist_ok=True)
                    with open(outdir+'condense_rs_ep'+str(ne)+'.txt', 'a+') as fout:
                       fout.write('\n'+type+' results at No. of hidden layers: {:d}, No. of training epochs: {:d}, No. of training size: {:d}\n'.format(nhl, ne, nts))

                       with open(infile) as fin:
                           for line in fin:
                              fout.write(line)

                    with open(outdir+type+'_prs_ep'+str(ne)+'_nhl'+str(nhl)+'.txt', 'a+') as fout2:
                        for row in range(nscore.shape[0]):
                            fout2.write(nscore[row, :]+'\n')

'''
def plotResults():
    types = ['train', 'dev', 'test']

    scoreFile = './tmp_2019_2_22/ontonotes/out/condense_rs_ep15.txt'

    score_rs = []
    nl = False
    with open(scoreFile, 'r') as fin:
        for line in fin:
            for type in types:
                if type in line:
                    nl = True
                    break

            if nl==True:
                if len(score_rs)!=0: # plot results
                    xl = [v+1 for v in range(len(score_rs))]
                    nscore = np.array(score_rs)

                    fig = plt.figure()
                    plt.plot(xl, nscore[:,0], 'ob')
                    plt.xlabel('No. of epochs')
                    plt.ylabel('Loss')
                    plt.grid()
                    plt.title('Results on the '+ori_type+' dataset')
                    plt.show()

                score_rs = []
                nl = False
                ori_type = type
            else: # add results
                outVal = re.split(':|,', line)

                if len(outVal)>1: # the size of stored data should be larger than 1
                    idx_range = range(1, len(outVal), 2)
                    score = [float(outVal[i]) for i in idx_range]
                    score_rs.append(score)

def plotDetResults():
    types = ['train', 'dev', 'test']

    scoreFile = './tmp_2019_2_22/ontonotes/out/condense_rs_ep15.txt'

    score_rs = []
    nl = False
    idx = 0
    with open(scoreFile, 'r') as fin:
        for line in fin:
            idx += 1
            if idx % 51 == 2: 
                type = 'train'
            if idx % 51 == 2+17: type = 'dev'
            if idx % 51 == 2+34: type = 'dev'


            if nl==True:
                if len(score_rs)!=0: # plot results
                    xl = [v+1 for v in range(len(score_rs))]
                    nscore = np.array(score_rs)

                    fig = plt.figure()
                    plt.plot(xl, nscore[:,0], 'ob')
                    plt.xlabel('No. of epochs')
                    plt.ylabel('Loss')
                    plt.grid()
                    plt.title('Results on the '+ori_type+' dataset')
                    plt.show()

                score_rs = []
                nl = False
                ori_type = type
            else: # add results
                outVal = re.split(':|,', line)

                if len(outVal)>1: # the size of stored data should be larger than 1
                    idx_range = range(1, len(outVal), 2)
                    score = [float(outVal[i]) for i in idx_range]
                    score_rs.append(score)
'''

if __name__=='__main__':
    saveScore2File()

    #plotDetResults()
