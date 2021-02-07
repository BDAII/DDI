# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:25:14 2019

@author: msl
"""

import tensorflow as tf
from Layer import GraphConvolution, Dense

flags = tf.app.flags
FLAGS = flags.FLAGS


class GCN(object):
    def __init__(self, placeholders, input_dim, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.train_loss = 0

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

        self.layers = []

        self.activations = []

        self.outputs = None

        self.loss = 0
        self.accuracy = 0

        self.opt_op = None

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions

        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.tanh,
                                            dropout=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.embedding_size,
                                            placeholders=self.placeholders,
                                            act=tf.nn.tanh,
                                            dropout=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.embedding_size,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=tf.nn.tanh,
                                            dropout=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.class_number,
                                            placeholders=self.placeholders,
                                            act=tf.nn.sigmoid,
                                            dropout=False,
                                            logging=self.logging))
        '''
        self.layers.append(Dense(input_dim=FLAGS.embedding_size,
                                            output_dim=FLAGS.class_number,
                                            placeholders=self.placeholders,
                                            act=tf.nn.sigmoid,
                                            dropout=True,
                                            logging=self.logging))
        '''

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        self.embedding = self.activations[-3]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        return tf.nn.softmax(self.outputs)

    def build_FL_Classifier_loss(self, label, predict, alpha=0.5):

        Pt_true = tf.multiply((1 - predict) ** 2, -tf.log(tf.clip_by_value(predict, 1e-4, 3.0)))
        Pt_true = tf.multiply(Pt_true, label)

        Pt_false = tf.multiply(predict ** 2, -tf.log(tf.clip_by_value(1 - predict, 1e-4, 3.0)))
        Pt_false = tf.multiply(Pt_false, 1.0 - label)

        loss = alpha * Pt_true + (1 - alpha) * Pt_false
        loss = tf.reduce_sum(loss)
        return loss

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        # self.train_loss = tf.reduce_sum((self.embedding-self.placeholders['labels'])**2)

        self.train_loss = self.build_FL_Classifier_loss(self.placeholders['labels'], self.outputs)

        self.loss += self.train_loss

    def _accuracy(self):

        correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.placeholders['labels'], 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(accuracy_all)

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
