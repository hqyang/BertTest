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
from src.pkuseg.metrics import getChunks
import matplotlib.patches as mpatches
import pdb


def splitScore(infile):
    tr_score = []
    ts_score = []
    dev_score = []

    with open(infile) as fi:
        for line in fi:
            outVal = re.split(':|,', line)
            idx_range = range(2, len(outVal), 2)
            score = [float(outVal[i]) for i in idx_range]

            if 'train' in line:
                tr_score.append(score)

            if 'test' in line:
                ts_score.append(score)

            if 'dev' in line:
                dev_score.append(score)

    return tr_score, dev_score, ts_score


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


def plotResults(tr_score, dev_score, ts_score, nhl, am_dev_loss):
    # F1, P, R, Acc
    tr_score = np.array(tr_score)
    dev_score = np.array(dev_score)
    ts_score = np.array(ts_score)

    xl = [v+1 for v in range(len(tr_score))]

    fig2 = plt.figure()
    p1, = plt.plot(xl, tr_score[:, 0], '-', marker='v', color='b', label='Train')
    p2, = plt.plot(xl, dev_score[:, 0], '--', marker='o', color='c', label='Dev')
    p3, = plt.plot(xl, ts_score[:, 0], ':', marker='p', color='r', label='Test')
    plt.xlabel('No. of epochs')
    plt.ylabel('F1')
    plt.grid()

    # get minimum value of dev
    #am_dev_loss = np.argmin(dev_score[:, 0])
    min_ts_loss = ts_score[am_dev_loss, 0]
    plt.text(am_dev_loss+0.5, min_ts_loss-45, str(am_dev_loss+1)+': '+str(min_ts_loss), color='r')

    get_test_F1 = ts_score[am_dev_loss, 0]
    plt.text(am_dev_loss+.5, get_test_F1+1, str(am_dev_loss+1)+': '+str(get_test_F1), color='r')
    plt.arrow(am_dev_loss+1.4, get_test_F1+0.9, -.4, -.8, color='r', length_includes_head=True,
              head_width=0.2, head_length=0.2)

    amax_test_F1 = np.argmax(ts_score[:, 0])
    max_test_F1 = ts_score[amax_test_F1, 0]
    plt.text(amax_test_F1+.5, max_test_F1+1, str(amax_test_F1+1)+': '+str(max_test_F1), color='r')
    plt.arrow(amax_test_F1+1.4, max_test_F1+.9, -.4, -.8, color='r', length_includes_head=True,
              head_width=0.2, head_length=0.2)

    #plt.legend(handles=[p1, p2, p3], loc=4)
    plt.legend(handler_map={p1: HandlerLine2D(numpoints=1)})
    #plt.legend(handles=[p1, p2, p3], labels=types) # , types
    plt.title('F1 on Ontonotes (No. hidden layers: {:d})'.format(nhl))
    plt.show()

    fig3 = plt.figure()
    p1, = plt.plot(xl, tr_score[:, 1], '-', marker='v', color='b', label='Train')
    p2, = plt.plot(xl, dev_score[:, 1], '--', marker='o', color='c', label='Dev')
    p3, = plt.plot(xl, ts_score[:, 1], ':',  marker='p', color='r', label='Test')
    plt.xlabel('No. of epochs')
    plt.ylabel('Precision')
    plt.grid()
    plt.legend(handler_map={p1: HandlerLine2D(numpoints=1)})
    plt.title('Precision on Ontonotes (No. hidden layers: {:d})'.format(nhl))
    plt.show()

    fig4 = plt.figure()
    p1, = plt.plot(xl, tr_score[:, 2], '-', marker='v', color='b', label='Train')
    p2, = plt.plot(xl, dev_score[:, 2], '--', marker='o', color='c', label='Dev')
    p3, = plt.plot(xl, ts_score[:, 2], ':',  marker='p', color='r', label='Test')
    plt.xlabel('No. of epochs')
    plt.ylabel('Recall')
    plt.grid()
    plt.legend(handler_map={p1: HandlerLine2D(numpoints=1)})
    plt.title('Recall on Ontonotes (No. hidden layers: {:d})'.format(nhl))
    plt.show()

    fig5 = plt.figure()
    p1, = plt.plot(xl, tr_score[:, 3], '-', marker='v', color='b', label='Train')
    p2, = plt.plot(xl, dev_score[:, 3], '--', marker='o', color='c', label='Dev')
    p3, = plt.plot(xl, ts_score[:, 3], ':',  marker='p', color='r', label='Test')
    plt.xlabel('No. of epochs')
    plt.ylabel('Accuracy')
    plt.grid()

    get_test_Acc = ts_score[am_dev_loss, 3]
    plt.text(am_dev_loss-1, get_test_Acc+0.5, str(am_dev_loss+1)+': '+str(get_test_Acc), color='r')
    plt.arrow(am_dev_loss-0.1, get_test_Acc+0.4, +1., -.4, color='r', length_includes_head=True,
              head_width=0.2, head_length=0.2)

    # get maximum value of test accuracy
    am_ts_acc = np.argmax(ts_score[:, 3])
    max_ts_acc = ts_score[am_ts_acc, 3]

    #get_test_F1 = ts_score[am_dev_loss,1]
    plt.text(am_ts_acc+1.5, max_ts_acc+.5, str(am_ts_acc+1)+': '+str(max_ts_acc), color='r')
    plt.arrow(am_ts_acc+1.9, max_ts_acc+0.4, -.9, -.4, color='r', length_includes_head=True,
              head_width=0.2, head_length=0.2)

    plt.legend(handler_map={p1: HandlerLine2D(numpoints=1)})
    plt.title('Accuracy on Ontonotes (No. hidden layers: {:d})'.format(nhl))
    plt.show()


def output1():
    pre_out_dir = '../tmp_2019_3_11/ontonotes/'
    pre_out_dir = '../tmp_2019_3_12/ontonotes/'
    pre_out_dir = '../tmp_2019_3_20/ontonotes/'
    pre_out_dir = '../tmp_2019_3_22/ontonotes/'
    pre_out_dir = '../tmp_2019_3_23/ontonotes/'
    saveScore2File(pre_out_dir, num_epochs=[15])

    pre_dir = pre_out_dir + 'out/'
    plotResults(pre_dir, num_epochs = [15])


def output2():
    infile = '../tmp_2019_3_23/ontonotes/out/eval_all_nhl3_nte15_nbs64.tsv'
    tr_score, dev_score, ts_score = splitScore(infile)
    plotResults(tr_score, dev_score, ts_score, nhl=3, am_dev_loss=3)

    infile = '../tmp_2019_3_23/ontonotes/out/eval_all_nhl6_nte15_nbs64.tsv'
    tr_score, dev_score, ts_score = splitScore(infile)
    plotResults(tr_score, dev_score, ts_score, nhl=6, am_dev_loss=1)

    infile = '../tmp_2019_3_23/ontonotes/out/eval_all_nhl12_nte15_nbs32.tsv'
    tr_score, dev_score, ts_score = splitScore(infile)
    plotResults(tr_score, dev_score, ts_score, nhl=12, am_dev_loss=0)


def plotResults(s1, s2, strlabels, strtitle, strYlabels, posY, methods):
    # F1, P, R, Acc
    s1 = np.array(s1)
    s2 = np.array(s2)
    print('{:s} training time: {:.2f}\pm {:.2f}, test time: {:.4f}\pm {:.2f}'.format( \
        strlabels[0], s1[:, 0].mean(), s1[:, 0].std(), s1[:, 1].mean(), s1[:, 1].std()))
    print('{:s} training time: {:.2f}\pm {:.2f}, test time: {:.4f}\pm {:.2f}'.format( \
        strlabels[2], s2[:, 0].mean(), s2[:, 0].std(), s2[:, 1].mean(), s2[:, 1].std()))

    xl = [v+1 for v in range(len(s1))]

    fig2 = plt.figure()
    p1, = plt.plot(xl, s1[:, 3], '-', marker='v', color='b', label=strlabels[0])
    p2, = plt.plot(xl, s2[:, 3], '--', marker='o', color='b', label=strlabels[1])
    p3, = plt.plot(xl, s1[:, 6], '-.', marker='^', color='r', label=strlabels[2])
    p4, = plt.plot(xl, s2[:, 6], ':', marker='x', color='r', label=strlabels[3])
    plt.xlabel('No. of epochs')
    #plt.ylabel('')
    plt.legend(strlabels)
    plt.title(strtitle)
    plt.text(7, posY-2, '{:s}: Training time/epoch: {:.1f}'.format(methods[0], \
                        s1[:, 0].mean())+r'$\pm$ ' + '{:.1f}'.format(s1[:, 0].std()),
             bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
             )
    plt.text(7, posY-4.5, '{:s}: Training time/epoch: {:.1f}'.format(methods[1], \
                        s2[:, 0].mean())+r'$\pm$ ' + '{:.1f}'.format(s2[:, 0].std()),
             bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
             )
    #plt.text(10, posY-10, r'time/ep.: {:.2f}$\pm$ {:.2f}'.format(s2[:, 0].mean(), s2[:, 0].std()), 'color'='r')

    # left Y-axis label
    plt.ylabel(strYlabels[0], {'color': 'b'}) # , 'fontsize': 12
    #plt.yticks((0, 0.5, 1), (r'\bf{0}', r'\bf{.5}', r'\bf{1}'), color='k', size=20)

    # right Y-axis label
    plt.text(32, posY, strYlabels[1], {'color': 'r'}, # , 'fontsize': 12
         horizontalalignment='left',
         verticalalignment='center',
         rotation=90,
#         clip_on=False,
#         transform=plt.gca().transAxes
             )

    plt.grid()


def compare_CRF_Softmax():
    infile_dir = './results/'

    ts_scores = {'CRF': [], 'Softmax': []}

    datasets = ['MSR', 'PKU']
    methods = ['CRF', 'Softmax']

    strlabels = ['F1_CRF', 'F1_Softmax', 'Acc._CRF', 'Acc._Softmax']
    posYs = {'MSR': 87, 'PKU': 72}

    strYlabels = ['F1', 'Accuracy']
    for dataset in datasets:
        strtitle = 'Performance on ' + dataset

        for method in methods:
            infile = infile_dir + dataset + '_' + method + '_l1.txt'

            _, _, ts_scores[method] = splitScore(infile)
        plotResults(ts_scores['CRF'], ts_scores['Softmax'], strlabels, strtitle, strYlabels, posYs[dataset], methods)


def read_finetune_layers_results():
    infile_dir = './results/'

    datasets = ['MSR', 'PKU']
    methods = ['CRF', 'Softmax']
    layers = [1, 3, 6, 12]

    for dataset in datasets:
        for layer in layers:
            for method in methods:
                infile = infile_dir + dataset + '_' + method + '_l' + str(layer) + '.txt'
                _, _, ts_score = splitScore(infile)
                s = np.array(ts_score)

                print('{:s}_{:s}_{:d}: \nmax F1: {:.2f}, max Acc.: {:.2f}' \
                      'tr_time: {:.1f}\pm {:.1f}, ts_time: {:.3f}\pm {:.3f}, ' \
                      'loss: {:.2f}\pm {:.2f}, F1: {:.2f}\pm {:.2f}, Acc.: {:.2f}\pm {:.2f}' \
                      ''.format(dataset, method, layer, s[:, 3].max(), s[:, 6].max(),
                    s[:, 0].mean(), s[:, 0].std(), s[:, 1].mean(), s[:, 1].std(),
                    s[:, 2].mean(), s[:, 2].std(), s[:, 3].mean(), s[:, 3].std(),
                    s[:, 6].mean(), s[:, 6].std()))


def read_BiLSTM_activations_results():
    infile_dir = './results/'

    datasets = ['MSR', 'PKU']
    methods = ['CRF', 'Softmax']
    activations = ['l1', 'l11', 'l12', 'SumL4', 'CatL4', 'SumAll']

    for dataset in datasets:
        for activation in activations:
            for method in methods:
                infile = infile_dir + dataset + '_' + method + '_BiLSTM_' + activation + '.txt'
                _, _, ts_score = splitScore(infile)
                s = np.array(ts_score)
                #s = s[0:10, :]

                print('{:s}_{:s}_{:s}: \nmax F1: {:.2f}, max Acc.: {:.2f} ' \
                      'tr_time: {:.1f}\pm {:.1f}, ts_time: {:.3f}\pm {:.3f}, ' \
                      'loss: {:.2f}\pm {:.2f}, F1: {:.2f}\pm {:.2f}, Acc.: {:.2f}\pm {:.2f}'\
                      ''.format(dataset, method, activation, s[:, 3].max(), s[:, 6].max(),
                    s[:, 0].mean(), s[:, 0].std(), s[:, 1].mean(), s[:, 1].std(),
                    s[:, 2].mean(), s[:, 2].std(), s[:, 3].mean(), s[:, 3].std(),
                    s[:, 6].mean(), s[:, 6].std()))


def extract_diff(infile, outfile):
    with open(outfile, 'w+') as fo:
        with open(infile, 'r') as fi:
            l_sent = fi.readline()
            while l_sent != '': # not EOF
                l_sent_old = l_sent
                l_t_seg = fi.readline()
                l_l_seg = fi.readline()
                l_tl_bio = fi.readline()
                l_ll_bio = fi.readline()
                l_sp = fi.readline()
                l_sent = fi.readline()

                # process difference
                l_t_seg_l = l_t_seg.split()
                l_l_seg_l = l_l_seg.split()
                diff_gold_list = []
                diff_res_list = []

                l_t_seg_l_used = l_t_seg_l
                for im in l_l_seg_l:
                    if im not in l_t_seg_l_used:
                        diff_res_list.append(im)
                    else:
                        l_t_seg_l_used.remove(im)

                l_l_seg_l_used = l_l_seg_l
                for im in l_t_seg_l:
                    if im not in l_l_seg_l:
                        diff_gold_list.append(im)
                    else:
                        l_l_seg_l_used.remove(im)

                fo.write(l_sent_old)
                fo.write('gold: ')
                for im in diff_gold_list:
                    fo.write(im + ' ')
                fo.write('\npredict: ')

                for im in diff_res_list:
                    fo.write(im + ' ')
                fo.write('\n\n')

                print(l_sent_old)
                print(diff_gold_list)
                print(diff_res_list)



if __name__=='__main__':
    #output1()
    #output2()
    #compare_CRF_Softmax()

    #read_finetune_layers_results()
    #print('\n\n')
    #read_BiLSTM_activations_results()

    infile = './results/MSR_test_ft_l12_Softmax_F1_weights_epoch22_diff.txt'
    outfile = './results/MSR_diff_auto.txt'
    extract_diff(infile, outfile)

    infile = './results/PKU_test_ft_l12_Softmax_Acc_weights_epoch10_diff.txt'
    outfile = './results/PKU_diff_auto.txt'
    extract_diff(infile, outfile)
