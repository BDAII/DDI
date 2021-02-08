# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import gc

hident_size = 512


def get_data(train_data_path, test_data_path):
    train_data = []
    train_data.append(np.load(train_data_path, allow_pickle=True))

    train_data = np.vstack(train_data)

    test_data = np.load(test_data_path, allow_pickle=True)

    train_feature = train_data[:, :hident_size * 2].reshape((-1, 2, hident_size))
    train_feature = np.stack((train_feature[:, 1, :], train_feature[:, 0, :]), axis=1)
    train_label = train_data[:, hident_size * 2:].astype(int)

    test_feature = test_data[:, :hident_size * 2].reshape((-1, 2, hident_size))
    test_feature = np.stack((test_feature[:, 1, :], test_feature[:, 0, :]), axis=1)
    test_label = test_data[:, hident_size * 2:].astype(int)

    return train_feature, train_label, test_feature, test_label


BATCH_SIZE = 256


class data_set():
    def __init__(self, train_feature, train_label, batch_size):
        self.train_feature = train_feature
        self.train_label = train_label
        self.batch_size = batch_size

    def get_next(self, index):
        star = index * self.batch_size
        end = (index + 1) * self.batch_size
        if end > self.train_feature.shape[0]:
            end = self.train_feature.shape[0]

        feature = self.train_feature[star:end]

        label = self.train_label[star:end]
        return feature, label


class MLNN():
    def __init__(self, feature_layer, feature_dim, rnn_unit, rnn_number, label_dim):


        self.feature = tf.placeholder(dtype=tf.float32, shape=[None, 2, feature_dim])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, label_dim])
        self.dropout = tf.placeholder_with_default(0., shape=())
        self.l2_norm = 0
        self.feature_dim = feature_dim
        self.feature_weight, self.feature_bias = self.weights_bias_init(feature_layer, "feature")
        self.build_rnn(rnn_unit, rnn_number)

    def build_rnn(self, unit, rnn_number):
        cells = [tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(num_units=unit, forget_bias=0.7, activation=tf.nn.softsign),
            input_keep_prob=1 - self.dropout) for _ in range(rnn_number)]
        cells = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs, state = tf.nn.dynamic_rnn(cells, self.feature, dtype=tf.float32, time_major=False)
        outputs = tf.transpose(outputs, [1, 0, 2])
        self.outputs = tf.concat([outputs[0], outputs[1]], axis=1)


    def weights_bias_init(self, encorder_layers, name_pref):
        feature_weights = []
        feature_bias = []
        for i in range(1, len(encorder_layers)):
            weight_layer = tf.Variable(tf.truncated_normal([encorder_layers[i - 1], encorder_layers[i]], stddev=0.1),
                                       name=name_pref + "_weight_" + str(i - 1))
            feature_weights.append(weight_layer)
            bias_layer = tf.Variable(tf.zeros([encorder_layers[i]]), name=name_pref + "_bias_" + str(i - 1))
            feature_bias.append(bias_layer)

        return feature_weights, feature_bias

    def forward(self, feature, weights, bias):
        for i in range(len(weights) - 1):
            feature = tf.nn.selu(tf.add(tf.matmul(feature, weights[i]), bias[i]))
            feature = tf.nn.dropout(feature, keep_prob=1 - self.dropout)
        feature = tf.nn.sigmoid(tf.add(tf.matmul(feature, weights[-1]), bias[-1]))
        # feature = tf.nn.softmax(feature)
        return feature

    def predict_forward(self, feature, weights, bias):
        for i in range(len(weights) - 1):
            feature = tf.nn.selu(tf.add(tf.matmul(feature, weights[i]), bias[i]))
        feature = tf.nn.sigmoid(tf.add(tf.matmul(feature, weights[-1]), bias[-1]))
        # feature = tf.nn.softmax(feature)
        return feature

    def predict(self):

        label = self.predict_forward(self.outputs, self.feature_weight, self.feature_bias)
        return label

    def build_FL_Classifier_loss(self, label, predict, alpha=0.5):

        Pt_true = tf.multiply((1 - predict) ** 2, -tf.log(tf.clip_by_value(predict, 1e-4, 3.0)))
        Pt_true = tf.multiply(Pt_true, label)

        Pt_false = tf.multiply(predict ** 2, -tf.log(tf.clip_by_value(1 - predict, 1e-4, 3.0)))
        Pt_false = tf.multiply(Pt_false, 1.0 - label)

        loss = alpha * Pt_true + (1 - alpha) * Pt_false
        loss = tf.reduce_sum(loss)
        return loss


    def get_MSE(self, predict):
        predict_loss = tf.reduce_sum((predict - self.label) ** 2) / 2
        return predict_loss

    def build_DNN(self, learning_rate=0.0001, L2_rate=0.00001):
        temp_feature = tf.reshape(self.feature, [-1, self.feature_dim * 2])
        temp_feature = tf.concat([self.outputs, temp_feature], axis=1)
        self.predict = self.forward(self.outputs, self.feature_weight, self.feature_bias)

        loss = self.build_FL_Classifier_loss(self.label, self.predict)

        trainable_variable = tf.trainable_variables()
        regularization_cost = L2_rate * tf.reduce_sum([tf.nn.l2_loss(v) for v in trainable_variable])
        loss += regularization_cost

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return optimizer, loss


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


def metric(y_true, predict_score):
    # thredhold = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    predict_sum = np.sum(predict_score, axis=1).reshape((-1, 1))
    thredhold = [0.1, 0.2]
    predict_score = predict_score / predict_sum

    best_f1 = 0
    m = []
    for i in thredhold:
        predict_label = (predict_score > i).astype(int)
        f1, recall, percision, BA = ACC_RECALL_PERC_BA(y_true, predict_label)
        print("  thredhold in %.2f ==F1-score:%.3f  Recall:%.3f  Percision:%.3f  BA:%.3f;" % (
        i, f1, recall, percision, BA))
        if f1 > best_f1:
            best_f1 = f1
            m = [f1, recall, percision, BA]
    return m


def sigle_label(y_true, y_predict):
    f1 = f1_score(y_true, y_predict, average="macro")
    recall = recall_score(y_true, y_predict, average="macro")
    percision = precision_score(y_true, y_predict, average="macro")
    acc = accuracy_score(y_true, y_predict)
    print("  ==F1-score:%.3f  Recall:%.3f  Percision:%.3f  ACC:%.3f" % (f1, recall, percision, acc))


import sys

loss_train = []
best_metric = [0, 0, 0, 0]
if __name__ == "__main__":

    train_data_path = str(sys.argv[1])
    test_data_path = str(sys.argv[2])
    save_path = "../../result/model/model_"
    ##############
    train_X, train_Y, test_X, test_Y = get_data(train_data_path, test_data_path)
    N = train_X.shape[0]
    dataset = data_set(train_X, train_Y, BATCH_SIZE)
    del train_X, train_Y
    gc.collect()

    ##############

    tf.reset_default_graph()
    RNN_unit = 400
    rnn_number = 2
    feature_layer = [RNN_unit * 2, 80]
    label_dim = 80
    epochs = 5001

    net = MLNN(feature_layer, hident_size, RNN_unit, rnn_number, label_dim)
    op, loss = net.build_DNN()

    num_batch = int(np.ceil(N / BATCH_SIZE))
    config = tf.ConfigProto(device_count={"CPU": 20})

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for e in range(epochs):
            train_loss = 0
            for i in range(num_batch):
                batch_x, batch_y = dataset.get_next(i)
                _, temp_loss = sess.run([op, loss],
                                        feed_dict={net.feature: batch_x, net.label: batch_y, net.dropout: 0.5})
                train_loss += temp_loss
            if e % 20 == 0:
                predict, test_loss = sess.run([net.predict, loss],
                                              feed_dict={net.feature: test_X, net.label: test_Y, net.dropout: 0.})
                print("====iteration  %04d======:train_loss:%0.4f========test_loss:%0.4f" % (e, train_loss, test_loss))
                predict = np.array(predict)
                temp_f1 = metric(test_Y, predict)

                # loss_train.append(train_loss)

                if temp_f1[0] > best_metric[0]:
                    best_metric = temp_f1
                    saver.save(sess, save_path + "/model_cv")

        print("====iteration  last======:strain_loss:%0.3f;" % (train_loss))
        print("best_metric:", best_metric)


