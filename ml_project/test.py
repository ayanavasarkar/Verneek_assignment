import pandas as pd
import os, tqdm
from typing import List, Tuple
import numpy as np
import numpy as np
import pickle, time
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#init random seed
import tqdm
np.random.seed(5)

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--train_data', type=str, default='data/train_data.dat', help='path to training data')
parser.add_argument('--test_data', type=str, default='data/test_data.dat',help='path to test data')
parser.add_argument('--test', type=bool, default=False, help='Test or not')
parser.add_argument('--n_users', type=int, default=5551,help='#Users')
parser.add_argument('--a_train', type=int, default=13584,help='#Articles train')
parser.add_argument('--a_test', type=int, default=1584,help='#Articles test/Eval')


args = parser.parse_args()

# find vocabulary_size = 8000
with open(r"data/vocabulary.dat") as vocabulary_file:
    vocabulary_size = len(vocabulary_file.readlines())

# find item_size = 16980
with open(r"data/mult.dat") as item_info_file:
    item_size = len(item_info_file.readlines())

# initialize item_infomation_matrix (16980 , 8000)
item_infomation_matrix = np.zeros((item_size, vocabulary_size))

# build item_infomation_matrix
with open(r"data/mult.dat") as f:
    s = f.readlines()

    for ind, s in enumerate(s):
        words = s.strip().split(" ")[1:]
        for word in words:
            vocab_ind, n = word.split(":")
            item_infomation_matrix[ind][int(vocab_ind)] = n

## train data user->article matching
def load_user_article_likes(path,  num_users, num_articles):
    """
    Load the article likes for each user

    Keyword arguments:
        data_folder -- the path to the folder that contains the data
        num_users -- the number of users in the dataset - default is 5551
        num_articles -- the number of articles in the dataset - default is 13584

    Returns:
        2d numpy matrix with describing user like information.
        This matrix is num_users X num_articles.
        There is a 1 in locations in which a user likes an article u -> a
        citations[u][a] and 0 otherwise.
    """

    user_items = np.zeros([num_users, num_articles])
    with open(os.path.join(path)) as users_file:
        for i, line in enumerate(users_file.readlines()):
            for article in line.strip().split()[1:]:
                user_items[i][int(article)] = 1
    return user_items


if args.test == False:
    rating_matrix = load_user_article_likes(args.train_data, args.n_users, args.a_train)
    rating_matrix_test = load_user_article_likes(args.train_data, args.n_users, args.a_train)
else:
    rating_matrix = load_user_article_likes(args.train_data, args.n_users, args.a_train)
    rating_matrix_test = load_user_article_likes(args.test_data, args.n_users, args.a_test)



def mask(c ,size):
    return np.random.binomial(1, 1 - c, [size[0],size[1]])

def add_noise(x, c):
    x *= mask(c, x.shape)
    return x


def recall_at_m(test_data_mat, prediction_mat, M=50) -> float:
    """
    Calculate recall at M metric. This metric calculates the top M article
    like predictions for each user and counts the number of actual user
    predictions in that list. The metric per user is:
         # counts in top M / number of total likes
    we then average over all user metrics
    We skip over all users that don't have any likes.

    Keyword arguments:
        test_data_mat -- The test data. A user x article length matrix of 0s and 1s
        prediction_mat -- The prediction data. A user x article length matrix of user like scores
            a larger score indicates higher likelihood of user liking an articleself.
        M -- The number of top predictions to evaluate from.

    Returns:
        float -- the recall_at_m metric
    """
    assert (
        test_data_mat.shape == prediction_mat.shape
    ), "test matrix and prediction matrix need to have the same shape"
    user_likes = [0 for _ in range(test_data_mat.shape[0])]
    user_corrects = [0 for _ in range(test_data_mat.shape[0])]
    user_recall = [0 for _ in range(test_data_mat.shape[0])]
    non_zero_users = 0
    for user in range(0, test_data_mat.shape[0]):
        user_likes[user] = sum(test_data_mat[user])
        ranked = sorted(
            enumerate(prediction_mat[user]), key=lambda x: x[1], reverse=True
        )[0:M]
        ranked_index = [i[0] for i in ranked]
        corrects = [
            i[0]
            for i in enumerate(test_data_mat[user])
            if i[0] in ranked_index and i[1] == 1
        ]
        user_corrects[user] = len(corrects)
        user_recall[user] = (
            user_corrects[user] / user_likes[user] if user_likes[user] > 0 else 0.0
        )
        if user_likes[user] > 0:
            non_zero_users += 1
    total_recall = sum(user_recall) / non_zero_users
    return total_recall


class CDL():
    def __init__(self, rating_matrix, rating_matrix_test, item_infomation_matrix):

        self.n_input = item_infomation_matrix.shape[1]
        self.n_hidden1 = 200
        self.n_hidden2 = 50
        self.k = 50

        self.lambda_w = 0.1
        self.lambda_n = 10
        self.lambda_u = 1
        self.lambda_v = 10

        self.drop_ratio = 0.1
        self.learning_rate = 0.05
        self.epochs = 1
        self.batch_size = 256

        self.a = 1
        self.b = 0.01
        self.P = 1

        self.num_u = rating_matrix.shape[0]
        self.num_v = rating_matrix.shape[1]
        self.num_v_ = rating_matrix_test.shape[1]

        self.Weights = {
            'w1': tf.Variable(
                tf.truncated_normal([self.n_input, self.n_hidden1], mean=0.0, stddev=tf.truediv(1.0, self.lambda_w))),
            'w2': tf.Variable(
                tf.truncated_normal([self.n_hidden1, self.n_hidden2], mean=0.0, stddev=tf.truediv(1.0, self.lambda_w))),
            'w3': tf.Variable(
                tf.truncated_normal([self.n_hidden2, self.n_hidden1], mean=0.0, stddev=tf.truediv(1.0, self.lambda_w))),
            'w4': tf.Variable(
                tf.truncated_normal([self.n_hidden1, self.n_input], mean=0.0, stddev=tf.truediv(1.0, self.lambda_w)))
        }
        self.Biases = {
            'b1': tf.Variable(tf.zeros(shape=self.n_hidden1)),
            'b2': tf.Variable(tf.zeros(shape=self.n_hidden2)),
            'b3': tf.Variable(tf.zeros(shape=self.n_hidden1)),
            'b4': tf.Variable(tf.zeros(shape=self.n_input)),
        }

        self.item_infomation_matrix = item_infomation_matrix

        self.rating_matrix = rating_matrix
        self.rating_matrix_test = rating_matrix_test

        for i in range(self.num_u):
            try:
                x = np.random.choice(np.where(self.rating_matrix[i, :] > 0)[0], self.P)
                x_ = np.random.choice(np.where(self.rating_matrix_test[i, :] > 0)[0], self.P)
            except:
                x = 1
                x_ = 1
            self.rating_matrix[i, :].fill(0)
            self.rating_matrix[i, x] = 1
            self.rating_matrix_test[i, :].fill(0)
            self.rating_matrix_test[i, x_] = 1

        self.confidence = np.mat(np.ones(self.rating_matrix.shape)) * self.b
        self.confidence[np.where(self.rating_matrix > 0)] = self.a
        self.confidence_test = np.mat(np.ones(self.rating_matrix_test.shape)) * self.b
        self.confidence_test[np.where(self.rating_matrix_test > 0)] = self.a

    def encoder(self, x, drop_ratio):
        w1 = self.Weights['w1']
        b1 = self.Biases['b1']
        L1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
        L1 = tf.nn.dropout(L1, keep_prob=1 - drop_ratio)

        w2 = self.Weights['w2']
        b2 = self.Biases['b2']
        L2 = tf.nn.sigmoid(tf.matmul(L1, w2) + b2)
        L2 = tf.nn.dropout(L2, keep_prob=1 - drop_ratio)

        return L2

    def decoder(self, x, drop_ratio):
        w3 = self.Weights['w3']
        b3 = self.Biases['b3']
        L3 = tf.nn.sigmoid(tf.matmul(x, w3) + b3)
        L3 = tf.nn.dropout(L3, keep_prob=1 - drop_ratio)

        w4 = self.Weights['w4']
        b4 = self.Biases['b4']
        L4 = tf.nn.sigmoid(tf.matmul(L3, w4) + b4)
        L4 = tf.nn.dropout(L4, keep_prob=1 - drop_ratio)

        return L4

    def build_model(self):

        self.X_0 = tf.placeholder(tf.float32, shape=(None, self.n_input))
        self.X_c = tf.placeholder(tf.float32, shape=(None, self.n_input))
        self.C = tf.placeholder(tf.float32, shape=(self.num_u, None))
        self.R = tf.placeholder(tf.float32, shape=(self.num_u, None))
        self.drop_ratio = tf.placeholder(tf.float32)
        self.model_batch_data_idx = tf.placeholder(tf.int32, shape=None)

        # SDAE item factor
        V_sdae = self.encoder(self.X_0, self.drop_ratio)

        # SDAE output
        self.sdae_output = self.decoder(V_sdae, self.drop_ratio)

        batch_size = tf.cast(tf.shape(self.X_0)[0], tf.int32)

        self.V = tf.Variable(tf.zeros(shape=[self.num_v, self.k], dtype=tf.float32))
        self.V_ = tf.Variable(tf.zeros(shape=[self.num_v_, self.k], dtype=tf.float32))
        self.U = tf.Variable(tf.zeros(shape=[self.num_u, self.k], dtype=tf.float32))

        batch_V = tf.reshape(tf.gather(self.V, self.model_batch_data_idx), shape=[batch_size, self.k])
        batch_V_ = tf.reshape(tf.gather(self.V_, self.model_batch_data_idx), shape=[batch_size, self.k])


        loss_1 = self.lambda_u * tf.nn.l2_loss(self.U)
        loss_2 = self.lambda_w * 1 / 2 * tf.reduce_sum(
            [tf.nn.l2_loss(w) + tf.nn.l2_loss(b) for w, b in zip(self.Weights.values(), self.Biases.values())])
        loss_3 = self.lambda_v * tf.nn.l2_loss(batch_V - V_sdae)
        loss_4 = self.lambda_n * tf.nn.l2_loss(self.sdae_output - self.X_c)

        loss_5 = tf.reduce_sum(tf.multiply(self.C,
                                           tf.square(self.R - tf.matmul(self.U, batch_V, transpose_b=True)))
                               )

        self.loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train_model(self, itr):
        start_time = time.time()
        total_batch = int(self.num_v / float(self.batch_size)) + 1
        ## Randomize the order of datapoints
        random_idx = np.random.permutation(self.num_v)

        ## Add NOISE
        self.item_infomation_matrix_noise = add_noise(self.item_infomation_matrix, 0.3)

        print("ENTERING THE TRAINING LOOP...................................................")

        batch_cost = 0
        for i in range(total_batch):
            if i == total_batch - 1:
                batch_idx = random_idx[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_idx = random_idx[i * self.batch_size : (i+1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.X_0: self.item_infomation_matrix_noise[batch_idx, :],
                                               self.X_c: self.item_infomation_matrix[batch_idx, :],
                                               self.R: self.rating_matrix[:, batch_idx],
                                               self.C: self.confidence[:, batch_idx],
                                               self.drop_ratio: 0.1,
                                               self.model_batch_data_idx: batch_idx})
            batch_cost = batch_cost + loss

        print ("Training //", "Epoch %d //" % (itr + 1), " Total cost = {:.2f}".format(batch_cost),
               "Elapsed time : %d sec" % (time.time() - start_time))


    def test_model(self):

        total_batch = int(self.num_v_ / float(self.batch_size)) + 1

        ## Randomize the order of datapoints
        # random_idx = np.random.permutation(self.num_v_)
        random_idx = list(range(0, self.num_v_))

        ## Add NOISE
        self.item_infomation_matrix_noise = add_noise(self.item_infomation_matrix, 0.3)


        batch_cost = 0
        for i in range(total_batch):
            if i == total_batch - 1:
                batch_idx = random_idx[i * self.batch_size:]
            elif i < total_batch - 1:
                batch_idx = random_idx[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss],
                                    feed_dict={self.X_0: self.item_infomation_matrix_noise[batch_idx, :],
                                               self.X_c: self.item_infomation_matrix[batch_idx, :],
                                               self.R: self.rating_matrix_test[:, batch_idx],
                                               self.C: self.confidence_test[:, batch_idx],
                                               self.drop_ratio: 0.1,
                                               self.model_batch_data_idx: batch_idx})
            batch_cost = batch_cost + loss

            Estimated_R = (tf.matmul(self.U, self.V_, transpose_b=True)).eval(session= self.sess)
            self.Estimated_R = Estimated_R.clip(min=0, max=1)
            return self.Estimated_R


    def run(self):
        self.sess = tf.Session()
        self.build_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(self.epochs):
            self.train_model(epoch_itr)


        return self.test_model()



R_train = rating_matrix.copy()
cdl = CDL(R_train, rating_matrix_test, item_infomation_matrix)
# cdl.build_model()
R = cdl.run()
# r_ = cdl.test_model()
# res = [idx for idx, val in enumerate(R[0]) if val > 0.0]
# res1 = [idx for idx, val in enumerate(rating_matrix[0]) if val > 0.0]
#
# print(R.shape, rating_matrix.shape, res)
# print(res1)
print(recall_at_m(rating_matrix, R))