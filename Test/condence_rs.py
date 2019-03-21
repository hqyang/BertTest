#!/anaconda3/envs/haiqin370/bin/python3 
# -*- coding: utf-8 -*- 
#import pandas as pd
import numpy as np
import re
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.patches as mpatches
import pdb

def getScore(infile):
    outScore = []
    with open(infile) as fi: 
        for line in fi:
            outVal = re.split(':|,', line)
            outVal = outVal[3:] # change due to modify the input the file
            idx_range = range(1, len(outVal), 2)
            score = [float(outVal[i]) for i in idx_range]
            outScore.append(score)

    return outScore    

def getScoreFromPureFile(infile):
    outScore = []
    with open(infile) as fi:
        for line in fi:
            #outVal = re.split(line)
            outVal = line.split()
            if outVal!='':
                score = [float(val) for val in outVal]
                outScore.append(score)

    return outScore


def saveScore2File(pre_out_dir, num_epochs = [15]):
    num_hidden_layers = [3, 6]
    #num_epochs = [1, 5, 10, 15]
    #num_epochs = [15]
    num_train_size = [64]
    types = ['train', 'dev', 'test']

    outdir = pre_out_dir+'out/'

    if os.path.exists(outdir) and os.listdir(outdir):
        os.system("rm %s" % os.path.join(outdir, '*'))
    os.makedirs(outdir, exist_ok=True)

    for nhl in num_hidden_layers:
        for ne in num_epochs:
            for nts in num_train_size:
                out_dir = pre_out_dir + 'nhl' + str(nhl) + '_nte' + str(ne) + '_nbs' + str(nts) + '/out/'
                for type in types:
                    infile = out_dir + type + '_eval_results.txt'
                    #pdb.set_trace()
                    score = getScore(infile)
                    nscore = np.array(score)

                    with open(outdir+'condense_rs_ep'+str(ne)+'.txt', 'a+') as fout:
                       fout.write('\n'+type+' results at No. of hidden layers: {:d}, No. of training epochs: {:d}, No. of training size: {:d}\n'.format(nhl, ne, nts))

                       with open(infile) as fin:
                           for line in fin:
                              fout.write(line)

                    #pdb.set_trace()
                    nrow, ncol = nscore.shape
                    with open(outdir+type+'_prs_ep'+str(ne)+'_nhl'+str(nhl)+'.txt', 'a+') as fout2:
                        for row in range(nrow):
                            ostr = ''
                            val = list(nscore[row, :])
                            #pdb.set_trace()
                            for col in range(ncol):
                               ostr += str(val[col]) + ' '
                            #pdb.set_trace()
                            fout2.write(ostr+'\n')

def plotResults(pre_dir, num_epochs = [15]):
    types = ['train', 'dev', 'test']
    num_hidden_layers = [3, 6]

    for nhl in num_hidden_layers:
        for ne in num_epochs:
            for type in types:
                outFile = pre_dir+type+'_prs_ep'+str(ne)+'_nhl'+str(nhl)+'.txt'
                score = getScoreFromPureFile(outFile)

                if type=='train':
                    tr_score = np.array(score)
                elif type=='dev':
                    dev_score = np.array(score)
                elif type=='test':
                    ts_score = np.array(score)

            xl = [v+1 for v in range(len(score))]
            fig1 = plt.figure()
            p1, = plt.plot(xl, tr_score[:,0], marker='v', color='b', label='Train')
            p2, = plt.plot(xl, dev_score[:,0], marker='o', color='r', label='Dev')
            p3, = plt.plot(xl, ts_score[:,0], marker='p', color='c', label='Test')
            plt.xlabel('No. of epochs')
            plt.ylabel('Loss')
            plt.grid()

            # get minimum value of dev
            am_dev_loss = np.argmin(dev_score[:,0])
            min_dev_loss = dev_score[am_dev_loss, 0]
            plt.text(am_dev_loss+0.5, min_dev_loss-45, str(am_dev_loss+1)+': '+str(min_dev_loss), color='r')
            #plt.arrow(am_dev_loss+1.8, min_dev_loss-45, -.5, 40, color='r',
            #          head_width=0.5, head_length=.5) # \ #length_includes_head=True, , shape='full'
            #mpatches.FancyArrow(am_dev_loss+1.8, min_dev_loss-45, -.5, 40, color='r', length_includes_head=True,
            #          head_width=0.2, head_length=0.2)

            #plt.legend(handles=[p1, p2, p3], loc=4)
            plt.legend(handler_map={p1: HandlerLine2D(numpoints=1)})
            #plt.legend(handles=[p1, p2, p3], labels=types)
            plt.title('Loss on Ontonotes (No. hidden layers: {:d})'.format(nhl))
            plt.show()

            fig2 = plt.figure()
            p1, = plt.plot(xl, tr_score[:,1], marker='v', color='b', label='Train')
            p2, = plt.plot(xl, dev_score[:,1], marker='o', color='r', label='Dev')
            p3, = plt.plot(xl, ts_score[:,1],  marker='p', color='c', label='Test')
            plt.xlabel('No. of epochs')
            plt.ylabel('F1')
            plt.grid()
            get_test_F1 = ts_score[am_dev_loss,1]
            plt.text(am_dev_loss+.5, get_test_F1+1, str(am_dev_loss+1)+': '+str(get_test_F1), color='c')
            plt.arrow(am_dev_loss+1.4, get_test_F1+0.9, -.4, -.8, color='c', length_includes_head=True,
                      head_width=0.2, head_length=0.2)

            amax_test_F1 = np.argmax(ts_score[:,1])
            max_test_F1 = ts_score[amax_test_F1, 1]
            plt.text(amax_test_F1+.5, max_test_F1+1, str(amax_test_F1+1)+': '+str(max_test_F1), color='c')
            plt.arrow(amax_test_F1+1.4, max_test_F1+.9, -.4, -.8, color='c', length_includes_head=True,
                      head_width=0.2, head_length=0.2)

            #plt.legend(handles=[p1, p2, p3], loc=4)
            plt.legend(handler_map={p1: HandlerLine2D(numpoints=1)})
            #plt.legend(handles=[p1, p2, p3], labels=types) # , types
            plt.title('F1 on Ontonotes (No. hidden layers: {:d})'.format(nhl))
            plt.show()

            fig3 = plt.figure()
            p1, = plt.plot(xl, tr_score[:,2], marker='v', color='b', label='Train')
            p2, = plt.plot(xl, dev_score[:,2], marker='o', color='r', label='Dev')
            p3, = plt.plot(xl, ts_score[:,2],  marker='p', color='c', label='Test')
            plt.xlabel('No. of epochs')
            plt.ylabel('Precision')
            plt.grid()
            plt.legend(handler_map={p1: HandlerLine2D(numpoints=1)})
            plt.title('Precision on Ontonotes (No. hidden layers: {:d})'.format(nhl))
            plt.show()

            fig4 = plt.figure()
            p1, = plt.plot(xl, tr_score[:,3], marker='v', color='b', label='Train')
            p2, = plt.plot(xl, dev_score[:,3], marker='o', color='r', label='Dev')
            p3, = plt.plot(xl, ts_score[:,3],  marker='p', color='c', label='Test')
            plt.xlabel('No. of epochs')
            plt.ylabel('Recall')
            plt.grid()
            plt.legend(handler_map={p1: HandlerLine2D(numpoints=1)})
            plt.title('Recall on Ontonotes (No. hidden layers: {:d})'.format(nhl))
            plt.show()

            fig5 = plt.figure()
            p1, = plt.plot(xl, tr_score[:,4], marker='v', color='b', label='Train')
            p2, = plt.plot(xl, dev_score[:,4], marker='o', color='r', label='Dev')
            p3, = plt.plot(xl, ts_score[:,4],  marker='p', color='c', label='Test')
            plt.xlabel('No. of epochs')
            plt.ylabel('Accuracy')
            plt.grid()

            get_test_Acc = ts_score[am_dev_loss,4]
            plt.text(am_dev_loss-1, get_test_Acc+1, str(am_dev_loss+1)+': '+str(get_test_Acc), color='c')
            plt.arrow(am_dev_loss-0.1, get_test_Acc+0.9, +1., -.8, color='c', length_includes_head=True,
                      head_width=0.2, head_length=0.2)

            # get minimum value of dev
            am_dev_acc = np.argmax(dev_score[:,4])
            max_dev_acc = dev_score[am_dev_acc, 4]

            #get_test_F1 = ts_score[am_dev_loss,1]
            plt.text(am_dev_acc+1.5, max_dev_acc+1, str(am_dev_acc+1)+': '+str(max_dev_acc), color='c')
            plt.arrow(am_dev_acc+1.9, max_dev_acc+0.9, -.9, -.8, color='c', length_includes_head=True,
                      head_width=0.2, head_length=0.2)

            #amax_test_F1 = np.argmax(ts_score[:,1])
            #max_test_F1 = ts_score[amax_test_F1, 1]
            #plt.text(amax_test_F1-1.5, max_test_F1-1, str(max_test_F1), color='c')
            #plt.arrow(amax_test_F1+0.1, max_test_F1-.9, .8, +.8, color='c', length_includes_head=True,
            #          head_width=0.2, head_length=0.2)

            plt.legend(handler_map={p1: HandlerLine2D(numpoints=1)})
            plt.title('Accuracy on Ontonotes (No. hidden layers: {:d})'.format(nhl))
            plt.show()


if __name__=='__main__':
    pre_out_dir = '../tmp_2019_3_11/ontonotes/'
    pre_out_dir = '../tmp_2019_3_12/ontonotes/'
    pre_out_dir = '../tmp_2019_3_20/ontonotes/'
    saveScore2File(pre_out_dir, num_epochs=[15])

    pre_dir = pre_out_dir + 'out/'
    plotResults(pre_dir, num_epochs = [15])
