# coding: utf-8

import tensorflow as tf
import numpy as np
import random
import math
import env

# using pmf to get item embedding
# params: rating_file
def mf_with_bias(config, lr=1e-2, l2_factor=1e-2, max_step=1000, train_rate=0.95, max_stop_count=30):
    rating_file = config['ENV']['RATING_FILE']
    rating_file_path = '../data/rating/' + rating_file
    rating = np.loadtxt(fname=rating_file_path, delimiter='\t')

    user_set = set()
    item_set = set()
    for i, j, k in rating:
        user_set.add(int(i))
        item_set.add(int(j))

    user_num = len(user_set)
    item_num = len(item_set)
    boundry_user_id = int(user_num * 0.8)
    emb_size = int(config['META']['ACTION_DIM'])

    print('training pmf...')
    print('user number: %d' % user_num)
    print('item number: %d' % item_num)

    data = np.array(list(filter(lambda x: x[0] < boundry_user_id, rating)))
    np.random.shuffle(data)

    t = int(len(data)*train_rate)
    dtrain = data[:t]
    dtest = data[t:]

    user_embeddings = tf.Variable(tf.truncated_normal([user_num, emb_size], mean=0, stddev=0.01))
    item_embeddings = tf.Variable(tf.truncated_normal([item_num, emb_size], mean=0, stddev=0.01))
    item_bias = tf.Variable(tf.zeros([item_num, 1], tf.float32))

    user_ids = tf.placeholder(tf.int32, shape=[None])
    item_ids = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    user_embs = tf.nn.embedding_lookup(user_embeddings, user_ids)
    item_embs = tf.nn.embedding_lookup(item_embeddings, item_ids)
    ibias_embs = tf.nn.embedding_lookup(item_bias, item_ids)
    dot_e = user_embs * item_embs

    ys_pre = tf.reduce_sum(dot_e, 1)+tf.squeeze(ibias_embs)

    target_loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre))
    loss = target_loss + l2_factor * (tf.reduce_mean(tf.square(user_embs) + tf.square(item_embs)))

    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.random.shuffle(dtrain)
        rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss], feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        rmse_test = sess.run(rmse, feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (0, rmse_train, rmse_test, loss_v, target_loss_v))
        pre_rmse_test = 100.0
        stop_count = 0
        stop_count_flag = False
        for i in range(max_step):
            feed_dict = {user_ids: dtrain[:, 0],
                         item_ids: dtrain[:, 1],
                         ys: np.float32(dtrain[:, 2])}
            sess.run(train_step, feed_dict)
            np.random.shuffle(dtrain)
            rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss], feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
            rmse_test = sess.run(rmse, feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1], ys: np.float32(dtest[:, 2])})
            print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (i + 1, rmse_train, rmse_test, loss_v, target_loss_v))
            if rmse_test>pre_rmse_test:
                stop_count += 1
                if stop_count==max_stop_count:
                    stop_count_flag = True
                    break
            pre_rmse_test = rmse_test

        item_embeddings_value = sess.run(item_embeddings)
        np.savetxt('../data/run_time/'+rating_file+'_item_embedding_dim%d'%emb_size, delimiter='\t', X=item_embeddings_value)
        print('done with full stop count' if stop_count_flag else 'done with full training step')

# to get clustering vector, rating/vae/mf
# params: rating_file
def clustering_vector_constructor(config, sess, max_user_num=30000):
    cur_env = env.Env(config)
    result_file_path = '../data/run_time/%s_%s_vector_v%d' % (config['ENV']['RATING_FILE'], config['TPGR']['CLUSTERING_VECTOR_TYPE'].lower(), int(config['TPGR']['CLUSTERING_VECTOR_VERSION']))

    if config['TPGR']['CLUSTERING_VECTOR_TYPE']=='RATING':
        rating_matrix = cur_env.get_init_data()[2].toarray()[:max_user_num]
        np.savetxt(X=rating_matrix.transpose(), fname=result_file_path, delimiter='\t')
    elif config['TPGR']['CLUSTERING_VECTOR_TYPE']=='MF':
        np.savetxt(X=np.loadtxt(fname='../data/run_time/%s_item_embedding_dim%s'%(config['ENV']['RATING_FILE'], config['META']['ACTION_DIM']), delimiter='\t'), fname=result_file_path, delimiter='\t')
    elif config['TPGR']['CLUSTERING_VECTOR_TYPE']=='VAE':
        rating_matrix = cur_env.get_init_data()[2].toarray()[:max_user_num]
        vae = VAE(rating_matrix.transpose(), sess, h2_size=int(config['META']['ACTION_DIM']))
        result = vae.run()
        np.savetxt(X=result, fname=result_file_path, delimiter='\t')
    else:
        print('not supported clustering vector type')
        exit(0)

class VAE():
    def __init__(self, data, sess, kl_factor = 5e-4, learning_rate = 2e-3, h1_size = 128, h2_size = 8, max_step = 1000):
        self.data = data
        self.sess = sess
        self.data_dim = len(self.data[0])
        self.kl_factor = kl_factor
        self.learning_rate = learning_rate
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.max_step = max_step

        self.input = tf.placeholder(shape=[None, self.data_dim], dtype=tf.float32)
        self.epsilon = tf.placeholder(shape=[None, self.h2_size], dtype=tf.float32)

    def construct_encoder(self):
        W1 = tf.Variable(tf.random_normal(shape=[self.data_dim, self.h1_size], stddev=0.01))
        W2 = tf.Variable(tf.random_normal(shape=[self.h1_size, self.h2_size*2], stddev=0.01))
        b1 = tf.Variable(tf.zeros(shape=[self.h1_size]))
        b2 = tf.Variable(tf.zeros(shape=[self.h2_size*2]))

        h1 = tf.nn.relu(tf.add(tf.matmul(self.input, W1), b1))
        self.mean = tf.nn.relu(tf.add(tf.matmul(h1, W2[:, :self.h2_size]), b2[:self.h2_size]))
        self.stddev = tf.nn.relu(tf.add(tf.matmul(h1, W2[:, self.h2_size:]), b2[self.h2_size:]))

    def construct_decoder(self):
        W1 = tf.Variable(tf.random_normal(shape=[self.h2_size, self.h1_size], stddev=0.01))
        W2 = tf.Variable(tf.random_normal(shape=[self.h1_size, self.data_dim], stddev=0.01))
        b1 = tf.Variable(tf.zeros(shape=[self.h1_size]))
        b2 = tf.Variable(tf.zeros(shape=[self.data_dim]))

        v = tf.add(self.mean, tf.multiply(self.stddev, self.epsilon))
        h1 = tf.nn.relu(tf.add(tf.matmul(v, W1), b1))
        self.output = tf.nn.relu(tf.add(tf.matmul(h1, W2), b2))

    def make_graph(self):
        self.construct_encoder()
        self.construct_decoder()

        self.KL_loss = -tf.reduce_mean(1.0 + 2.0 * tf.log(self.stddev+1e-10) - tf.square(self.stddev) - tf.square(self.mean), 1)
        self.reconstruction_loss = 0.5 * tf.reduce_mean(tf.square(self.input-self.output), 1)

        self.loss = self.reconstruction_loss + self.kl_factor * self.KL_loss

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def run(self, max_stop_count=100):
        self.make_graph()
        self.sess.run(tf.global_variables_initializer())
        stop_count = 0
        pre_loss = 1e10
        index = list(range(len(self.data)))
        for step in range(self.max_step):
            random.shuffle(index)
            _, loss, kl_loss, rec_loss = self.sess.run([self.train_op, tf.reduce_mean(self.loss), tf.reduce_mean(self.KL_loss), tf.reduce_mean(self.reconstruction_loss)], feed_dict={self.input: self.data[index[:1000]], self.epsilon: np.random.normal(size=[len(self.data[index[:1000]]), self.h2_size])})
            rmse = self.sess.run(tf.reduce_mean(self.reconstruction_loss), feed_dict={self.input: self.data[index[:1000]], self.epsilon: np.zeros(shape=[len(self.data[index[:1000]]), self.h2_size])})
            print('step %3d, loss %f, rec %.5f, rmse %.5f, kl %.5f'%(step, loss, rec_loss, rmse, kl_loss))
            if pre_loss<loss:
                stop_count += 1
                if stop_count == max_stop_count:
                    break
            pre_loss = loss

        result = [self.sess.run(self.mean, feed_dict={self.input: self.data[i*1000:(i+1)*1000]}) for i in range(math.ceil(len(self.data)/1000))]
        return np.concatenate(result, axis=0)
