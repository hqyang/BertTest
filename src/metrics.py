import itertools
import numpy as np


#from .input_fn import predict_input_fn
#from .utils import get_text_and_label, LabelEncoder, get_or_make_label_encoder
from src.pkuseg.metrics import getFscore

def accuracy(out, labels, ignore_index=-100, reduce=False):
    outputs = np.argmax(out, axis=1)
    accs = np.sum((outputs==labels) & (labels!=ignore_index))
    if reduce:
        accs = accs / np.sum(labels!=ignore_index)
    return accs

def map_score_to_multilabel(label_list, out, labels):
    labelid_map = {}
    for (i, label) in enumerate(label_list):
        labelid_map[i] = label
    split_label_list = [_.split(',') for _ in label_list]
    multilabel_list = sorted(list(set(itertools.chain(*split_label_list))))
    label_lists_mapping = [[y in x for x in split_label_list] for y in multilabel_list]

    spilt_label_lists = [labelid_map[_].split(',') for _ in labels]
    multilabels = np.array([[1 if x in y else 0 for x in multilabel_list] for y in spilt_label_lists])
    outputs = []
    for cols in label_lists_mapping:
        outputs.append(out[:, cols].sum(axis=1))
    outputs = np.stack(outputs).T
    return outputs, multilabels

def predict_at_least_one(out, threshold=0.):
    #drop 'UNK' column
    outputs = (out > threshold).astype(float)
    argmax_onehot = (out == out.max(1).reshape(-1, 1)).astype(float)
    is_all_zero = (outputs==0.).prod(axis=1).reshape(-1, 1)
    outputs = (1 - is_all_zero) * outputs + is_all_zero * argmax_onehot
    return outputs

def accuracy_multilabel(out, labels):
    outputs = predict_at_least_one(out)
    # logging.debug(outputs)
    return (outputs==labels).prod(axis=1).sum()

'''
This part is revised from pkuseg.getFscore
#from src.pkuseg.metrics import getChunks, getNewTagList
#from src.pkuseg.metrics import Config
def getFscore(goldTagList, resTagList, idx_to_chunk_tag=None):
    scoreList = []
    assert len(resTagList) == len(goldTagList)
    getNewTagList(idx_to_chunk_tag, goldTagList)
    getNewTagList(idx_to_chunk_tag, resTagList)
    goldChunkList = getChunks(goldTagList)
    resChunkList = getChunks(resTagList)
    gold_chunk = 0
    res_chunk = 0
    correct_chunk = 0
    for i in range(len(goldChunkList)):
        res = resChunkList[i]
        gold = goldChunkList[i]
        resChunkAry = res.split(Config.comma)
        tmp = []
        for t in resChunkAry:
            if len(t) > 0:
                tmp.append(t)
        resChunkAry = tmp
        goldChunkAry = gold.split(Config.comma)
        tmp = []
        for t in goldChunkAry:
            if len(t) > 0:
                tmp.append(t)
        goldChunkAry = tmp
        gold_chunk += len(goldChunkAry)
        res_chunk += len(resChunkAry)
        goldChunkSet = set()
        for im in goldChunkAry:
            goldChunkSet.add(im)
        for im in resChunkAry:
            if im in goldChunkSet:
                correct_chunk += 1
    pre = correct_chunk / res_chunk * 100
    rec = correct_chunk / gold_chunk * 100
    f1 = 0 if correct_chunk == 0 else 2 * pre * rec / (pre + rec)
    scoreList.append(f1)
    scoreList.append(pre)
    scoreList.append(rec)
    infoList = []
    infoList.append(gold_chunk)
    infoList.append(res_chunk)
    infoList.append(correct_chunk)
    return scoreList, infoList
'''

''' 
# copy from https://github.com/supercoderhawk/DNN_CWS/blob/master/utils.py
def estimate_cws(current_labels,correct_labels):
    cor_dict = {}
    curt_dict = {}
    curt_start = 0
    cor_start = 0

    for label_index, (curt_label, cor_label) in enumerate(zip(current_labels, correct_labels)):
        if cor_label == 0:
            cor_dict[label_index] = label_index + 1
        elif cor_label == 1:
            cor_start = label_index
        elif cor_label == 3:
            cor_dict[cor_start] = label_index + 1

        if curt_label == 0:
            curt_dict[label_index] = label_index + 1
        elif curt_label == 1:
            curt_start = label_index
        elif curt_label == 3:
            curt_dict[curt_start] = label_index + 1

    cor_count = 0
    recall_length = len(curt_dict)
    prec_length = len(cor_dict)

    for curt_start in curt_dict.keys():
        if curt_start in cor_dict and curt_dict[curt_start] == cor_dict[curt_start]:
            cor_count += 1

    return  cor_count,prec_length,recall_length
'''
'''
This part is copied from https://github.com/jiesutd/LatticeLSTM
'''

def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    return accuracy, precision, recall, f_measure


def reverse_style(input_string):
    # Example:
    #   istr = "PER[12,13]"
    #   ostr = "[12,13]PER"

    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + \
        input_string[0:target_position]
    return output_string


def get_ner_BIO(label_list):
    # Example:
    #   label_list1 = ['O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG']
    #   out_list1 = get_ner_BIO(label_list1) # ['[1,2]PER', '[6,6]PER', '[9]ORG']

    #   label_list2 = ['O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O']
    #   out_list2 = get_ner_BIO(label_list2) # ['[1,2]PER', '[6,6]PER', '[9,11]ORG']


    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []

    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(
                    begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(
                    begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)

        elif inside_label in current_label:
            if current_label.replace(inside_label, "", 1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '') & (index_tag != ''):
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def get_ner_BMES(label_list):
    # need to give an example
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(
                begin_label, "", 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, "", 1)
        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(
                single_label, "", 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix

'''
def ner_evaluate(problem, estimator, params):
    estimator_problem = copy(params.problem_str)
    base_dir = os.path.split(params.ckpt_dir)[0]
    params.assign_problem(problem, base_dir=base_dir)
    text, label_data = get_text_and_label(params, problem, 'eval')

    def pred_input_fn(): return predict_input_fn(text, params, mode='predict')

    params.assign_problem(estimator_problem, base_dir=base_dir)

    label_encoder = get_or_make_label_encoder(params, problem, mode='eval')

    pred_list = estimator.predict(pred_input_fn)

    decode_pred_list = []
    decode_label_list = []

    for p, label, t in zip(pred_list, label_data, text):
        true_seq_length = len(t) - 1

        pred_prob = p[problem]

        pred_prob = pred_prob[1:true_seq_length]

        # crf returns tags
        predict = pred_prob
        label = label[1:true_seq_length]

        decode_pred = label_encoder.inverse_transform(predict)
        decode_label = label_encoder.inverse_transform(label)

        decode_pred_list.append(decode_pred)
        decode_label_list.append(decode_label)

    result_dict = {}

    for metric_name, result in zip(['Acc', 'Precision', 'Recall', 'F1'],
                                   get_ner_fmeasure(decode_label_list,
                                                    decode_pred_list,
                                                    label_type='BIO')):
        print('%s Score: %f' % (metric_name,  result))
        result_dict[metric_name] = result
    return result_dict
'''

from src.config import segType
def outBIOTagList(inputTagList, input_mask=None, mode='BMES'):
    if mode == 'BMES':
        idx_to_label_map = segType.BMES_idx_to_label_map
    else:
        idx_to_label_map = segType.BIO_idx_to_label_map

    outTagList = []
    for i in range(len(inputTagList)):
        tmp = inputTagList[i]
        if input_mask == None:
            tmp_mask = [1]*len(tmp)
        else:
            tmp_mask = input_mask[i]

        t = ''
        for ii in range(len(tmp)):
            if tmp_mask[ii] == 1:
                t += ''.join(str(idx_to_label_map[tmp[ii]])+',')

        t = t.replace('[START],', '')
        t = t.replace('[END],', '')
        if mode == 'BMES':
            t = t.replace('M', 'I')
            t = t.replace('E', 'I')

        outTagList.append([t])

    return outTagList

def process2BIOTagList(inputTagList, input_mask=None, mode='BMES'):
    outTagList = []
    for i in range(len(inputTagList)):
        tmp = inputTagList[i]
        if input_mask == None:
            tmp_mask = [1]*len(tmp)
        else:
            tmp_mask = input_mask[i]

        t = ''
        for ii in range(len(tmp)):
            if tmp_mask[ii] == 1:
                t += ''.join(str(tmp[ii])+',')

        t = t.replace('4,', '')
        t = t.replace('5,', '')
        if mode == 'BMES': # B: 0, M: 1, E: 2, S: 3, [STXX]: 4, [END]:5
            t = t.replace('2', '1')

        outTagList.append(t)

    return outTagList


def outputFscoreUsedBIO(goldTagList, preTagList, input_mask=None, mode='BMES'):
    # input format is list
    # need additional process before calling getFscore
    # 1. idx to tag for goldTagList, need consider input_mask
    goldTagList = process2BIOTagList(goldTagList, input_mask)

    # 2. idx to tag for preTagList, no need for input_mask
    preTagList = process2BIOTagList(preTagList)

    scoreList, infoList = getFscore(goldTagList, preTagList, segType.BIO_idx_to_label_map)

    return scoreList, infoList
