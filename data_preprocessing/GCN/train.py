# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import numpy as np
import pandas as pd
from Model import GCN


def get_spare_connect_matrix(W, K):
    N = W.shape[0]
    S = np.zeros(W.shape)

    for i in range(N):
        W[i, i] = 0

    k_nn_node = np.argsort(W, axis=1)[:, -K:]

    k_nn_node_sum = np.sum(np.sort(W, axis=1)[:, -K:], axis=1)

    for i in range(N):
        for k in range(N):
            if k in k_nn_node[i, :]:
                S[i, k] = W[i, k] / k_nn_node_sum[i]
    for i in range(N):
        S[i, i] = 1
    return S


def load_data(K=40):
    label = pd.read_csv("../../data/morgn_struct_Dice_sp_consen_label.csv", index_col=0)

    label_index = list(label.index)
    label = label.values
    morgn_fp = \
    pd.read_csv("../../data/morgn_FP.csv", index_col=0).loc[label_index].values
    feature = \
    pd.read_csv("../../data/PC3_gen_data.csv", index_col=0).loc[label_index].values
    adj_path = "../../data/morgn_struct_Dice.csv"
    sim = get_spare_connect_matrix(pd.read_csv(adj_path, index_col=0).loc[label_index, label_index].values, K)
    # sim = pd.read_csv(adj_path,index_col=0).loc[label_index,label_index].values
    sim_comb = [sim]

    print(adj_path)
    return sim_comb, feature, morgn_fp, label_index


def normarl_similarity(W, diag_set=True):
    if diag_set:
        for i in range(W.shape[0]):
            W[i, i] = 1
    W_sum = np.sum(W, axis=1).reshape((-1, 1))

    W = W / (2 * W_sum)
    for i in range(W.shape[0]):
        W[i, i] = 0.5

    return W


def construct_feed_dict(features, support, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 8001, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 640, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 640, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('embedding_size', 512, 'Number of units in embedding layer 2.')
flags.DEFINE_integer('class_number', 1024, 'Number of class.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-2, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, label, index = load_data()

# Some preprocessing
# features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = adj
    num_supports = len(adj)
    model_func = GCN

# Define placeholders
placeholders = {
    'support': [tf.placeholder(tf.float32, shape=(support[0].shape)) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
    'labels': tf.placeholder(tf.float32, shape=(None, label.shape[1])),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features.shape[1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, placeholders)
    outs_val = sess.run([model.loss], feed_dict=feed_dict_val)
    return outs_val, (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = 1
ACC = 0
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, label, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Print results
    if epoch % 20 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "ACC=", "{:.5f}".format(outs[2]), "time=", "{:.5f}".format(time.time() - t))

    if ACC < outs[2]:
        ACC = outs[2]
    if (cost_val == 1 and outs[1] > 0) or cost_val > outs[1]:
        cost_val = outs[1]
        # feed_dict.update({placeholders['dropout']: 0})
        # outputs = sess.run([model.embedding], feed_dict=feed_dict)
        # outputs = pd.DataFrame(np.array(outputs)[0],index=index)
        # outputs.to_csv("morgn_GCN_AE_bymorgnSim_10_edge_embedding.csv")

print("Optimization Finished!")
print(cost_val)
print(ACC)



