import numpy as np
import itertools

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