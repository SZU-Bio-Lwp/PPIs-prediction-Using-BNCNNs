import numpy as np
from numpy.random import RandomState
from sklearn.metrics import matthews_corrcoef

Input_dim = 300
Output_dim = 2


def compute_mcc(preds, labels):
    preds = np.array(preds, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc


def compute_performance_in_fixed_spec(preds, labels):
    for t in range(1, 1001):
        threshold = t / 1000.0
        predictions = (preds > threshold).astype(np.int32)

        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        tn = len(labels) - tp - fp - fn

        if tp == 0 and fp == 0 and fn == 0:
            continue
        if tn + fp == 0: continue

        sp = tn / (1.0 * (tn + fp))
        if sp == 0.950:  # fixed-specificity=0.95
            acc = (tp + tn) / (1.0 * (tp + fp + tn + fn))
            recall = tp / (1.0 * (tp + fn))
            precision = tp / (1.0 * (tp + fp))
            f1 = 2 * precision * recall / (precision + recall)
            mcc = compute_mcc(predictions, labels)
            return [acc, recall, precision, f1, sp, mcc]
    return [0, 0, 0, 0, 0, 0, 0]


def td_to_2d(index, train_data):
    x_ = np.zeros((len(index), Input_dim))
    y_ = np.zeros((len(index), Output_dim))
    j = 0
    for i in index:
        k = 0
        for l in train_data[i][0]:
            x_[j][k] = l
            k += 1
        k = 0
        for l in train_data[i][1]:
            y_[j][k] = l
            k += 1
        j += 1
    return x_, y_


def td_Allto_2d(data):
    x_ = np.zeros((len(data), Input_dim))
    y_ = np.zeros((len(data), Output_dim))
    j = 0
    for single_d in data:
        k = 0
        for l in single_d[0]:
            x_[j][k] = l
            k += 1
        k = 0
        for l in single_d[1]:
            y_[j][k] = l
            k += 1
        j += 1
    return x_, y_


def random_batch(x_train, y_train, batch_size):
    while True:
        rnd_indices = np.random.randint(0, len(x_train), batch_size)
        num_pos, num_neg = 0, 0
        for i in rnd_indices:
            if y_train[i][0] == 0:
                num_pos += 1
            else:
                num_neg += 1
        if num_pos == 0 or num_neg == 0: continue
        break

    x_batch = np.zeros((batch_size, Input_dim))
    y_batch = np.zeros((batch_size, Output_dim))
    j = 0
    for i in rnd_indices:
        k = 0
        for l in x_train[i]:
            x_batch[j][k] = l
            k += 1
        k = 0
        for l in y_train[i]:
            y_batch[j][k] = l
            k += 1
        j += 1
    return x_batch, y_batch


def one_hot_to_label(label):
    y = []
    for data in label:
        if data[0] == 0:
            l = 1
        else:
            l = 0
        y.append(l)
    return y
