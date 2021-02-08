# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLARAM, MLkNN, BRkNNaClassifier, BRkNNbClassifier, MLTSVM
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import gc

hident_size = 512 * 2


def get_data(train_data_path, test_data_path):

    train_data = []
    train_data.append(np.load(train_data_path, allow_pickle=True))

    train_data = np.vstack(train_data)

    test_data = np.load(test_data_path, allow_pickle=True)

    return train_data[:, :hident_size], train_data[:, hident_size:].astype(int), test_data[:, :hident_size], test_data[
                                                                                                             :,
                                                                                                             hident_size:].astype(
        int)


def ML_model_predict(train_x, train_y, test_x, model_name):
    print(f"--------train {model_name} model----------")
    classifier = None
    if model_name == "MLARAM":
        classifier = MLARAM(threshold=0.2)
    elif model_name == "MLkNN":
        classifier = MLkNN()
    elif model_name == "BRkNNa":
        classifier = BRkNNaClassifier()
    elif model_name == "BRkNNb":
        classifier = BRkNNbClassifier()
    elif model_name == "RF":
        classifier = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    elif model_name == "MLTSVM":
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
    train_data_path = str(sys.argv[1])
    test_data_path = str(sys.argv[1])

    train_X, train_Y, test_X, test_Y = get_data(train_data_path, test_data_path)
    print(type(train_X))
    print(train_X.shape, train_Y.shape)
    print(test_X.shape, test_Y.shape)
    model_list = ["MLARAM", "MLkNN", "BRkNNa", "BRkNNb", "RF", "MLTSVM"]
    for model in model_list:
        predict = ML_model_predict(train_X, train_Y, test_X, model)
        print(f"-------{model} model predict---------")
        predict = predict.toarray()
        print(type(predict))
        print(predict.shape)
        print(predict[:10, :10])
        metric(test_Y, predict)
        print("--------------------------------------")
