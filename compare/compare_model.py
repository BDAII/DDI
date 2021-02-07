# -*- coding: utf-8 -*-
"""
该程序是将 MLARAM, MLkNN, BRkNNaClassifier, BRkNNbClassifier四种模型进行对比的代码
output: xxx
"""
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLARAM, MLkNN, BRkNNaClassifier, BRkNNbClassifier, MLTSVM

from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import gc

hident_size = 512 * 2


def get_data(file_path, item):
    L = set(range(1, 6))
    item_set = set([item])
    L = list(L - item_set)

    train_data = []
    for i in L:
        print(i)
        train_data.append(np.load(file_path + "data_set_" + str(i) + ".npy", allow_pickle=True))

    train_data = np.vstack(train_data)

    test_data = np.load(file_path + "data_set_" + str(item) + ".npy", allow_pickle=True)

    return train_data[:, :hident_size], train_data[:, hident_size:].astype(int), test_data[:, :hident_size], test_data[
                                                                                                             :,
                                                                                                             hident_size:].astype(
        int)


def ML_model_predict(train_x, train_y, test_x):
    # classifier = MLARAM(threshold=0.2)
    # classifier = MLkNN()
    # classifier = BRkNNaClassifier()
    # classifier = BRkNNbClassifier()
    classifier = MLTSVM(c_k=2 ** -1)
    classifier.fit(train_x, train_y)
    prediction = classifier.predict(test_x)
    return prediction


def ACC_RECALL_PERC_BA(label_ture, predict_label):
    F1_score, Recall, Percision, BA = [], [], [], []

    for i in range(predict_label.shape[1]):
        y_pred = predict_label[:, i]
        y_true = label_ture[:, i]
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        F1_score.append(f1)
        Recall.append(recall_score(y_true, y_pred))
        Percision.append(precision_score(y_true, y_pred))
        BA_sep = 0
        if tp == 0:
            BA_sep += 0
        else:
            BA_sep += tp / (tp + fp)
        if tn == 0:
            BA_sep += 0
        else:
            BA_sep += tn / (fp + tn)

        BA.append(BA_sep / 2)

    return np.mean(F1_score), np.mean(Recall), np.mean(Percision), np.mean(BA)


def metric(y_true, predict_label):
    f1, recall, percision, BA = ACC_RECALL_PERC_BA(y_true, predict_label)
    print("  thredhold in 0.2 ==F1-score:%.4f  Recall:%.4f  Percision:%.4f  BA:%.4f;" % (f1, recall, percision, BA))


import sys

loss_train = []
best_metric = [0, 0, 0, 0]
if __name__ == "__main__":
    open_file = "/home/moshenglong/drug_9/step_7_5fold/GCN_AE_40_edge/"
    # save_path = "/home/moshenglong/drug_9/step_7_5fold/GCN_clustring/model/model_"
    part = int(sys.argv[1])
    ##############
    train_X, train_Y, test_X, test_Y = get_data(open_file, part)
    print(type(train_X))
    print(train_X.shape, train_Y.shape)
    print(test_X.shape, test_Y.shape)

    predict = ML_model_predict(train_X, train_Y, test_X)
    predict = predict.toarray()
    print(type(predict))
    print(predict.shape)
    print(predict[:10, :10])
    metric(test_Y, predict)
