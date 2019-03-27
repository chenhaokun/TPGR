#coding: utf-8

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
import run_time_tools
from env import Env
import numpy as np
import random
import utils
import math
import time
import gc
import os

class PRE_TRAIN():
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.log = utils.Log()

        self.action_dim = int(self.config['META']['ACTION_DIM'])
        self.statistic_dim = int(self.config['META']['STATISTIC_DIM'])
        self.reward_dim = int(self.config['META']['REWARD_DIM'])
        self.batch_size = int(self.config['TPGR']['PRE_TRAINING_BATCH_SIZE'])
        self.log_step = int(self.config['TPGR']['PRE_TRAINING_LOG_STEP'])
        self.learning_rate = float(self.config['TPGR']['PRE_TRAINING_LEARNING_RATE'])
        self.l2_factor = float(self.config['TPGR']['PRE_TRAINING_L2_FACTOR'])
        self.pre_train_truncated_length = int(self.config['TPGR']['PRE_TRAINING_RNN_TRUNCATED_LENGTH'])
        self.max_item_num = int(self.config['TPGR']['PRE_TRAINING_MAX_ITEM_NUM'])
        self.pre_train_seq_length = min(int(self.config['TPGR']['PRE_TRAINING_SEQ_LENGTH']), self.max_item_num)
        self.pre_train_mask_length = min(int(self.config['TPGR']['PRE_TRAINING_MASK_LENGTH']), self.pre_train_seq_length)
        self.rnn_file_path = '../data/run_time/%s_rnn_model_%s' % (self.config['ENV']['RATING_FILE'], self.config['TPGR']['RNN_MODEL_VS'].split('s')[0])

        self.rnn_input_dim = self.action_dim + self.reward_dim + self.statistic_dim
        self.rnn_output_dim = self.rnn_input_dim

        self.forward_env = Env(self.config)
        self.boundry_user_id = self.forward_env.boundry_user_id
        self.user_num, self.item_num, self.r_matrix, self.user_to_rele_num = self.forward_env.get_init_data()
        self.env = [Env(self.config, self.user_num, self.item_num, self.r_matrix, self.user_to_rele_num) for i in range(max(self.user_num, self.batch_size))]

        self.pre_training_steps = 0
        self.make_graph()
        self.sess.run(tf.global_variables_initializer())

        self.log.log('graph constructed', True)

    def make_graph(self):
        # placeholders
        self.pre_actions = [tf.placeholder(dtype=tf.int32, shape=[None]) for i in range(self.pre_train_truncated_length)]
        self.pre_rewards = [tf.placeholder(dtype=tf.float32, shape=[None]) for i in range(self.pre_train_truncated_length)]
        self.pre_statistic = [tf.placeholder(dtype=tf.float32, shape=[None, self.statistic_dim]) for i in range(self.pre_train_truncated_length)]
        self.pre_rnn_state = tf.placeholder(dtype=tf.float32, shape=[2, None, self.rnn_output_dim], name='pre_rnn_state')

        # action embeddings
        self.action_embeddings = tf.constant(dtype=tf.float32, value=self.forward_env.item_embedding)

        # rnn
        self.initial_states = tf.stack([tf.zeros([self.batch_size, self.rnn_output_dim]), tf.zeros([self.batch_size, self.rnn_output_dim])])
        self.rnn, self.rnn_variables = self.create_sru(self.rnn_input_dim, self.rnn_output_dim)

        # rnn input
        self.pre_a_embs = [tf.nn.embedding_lookup(self.action_embeddings, self.pre_actions[i]) for i in range(self.pre_train_truncated_length)]
        one_hot_rewards = [tf.one_hot(tf.cast(self.reward_dim * (1.0 - self.pre_rewards[i]) / 2, dtype=tf.int32), depth=self.reward_dim) for i in range(self.pre_train_truncated_length)]
        self.pre_ars = [tf.concat([self.pre_a_embs[i], one_hot_rewards[i], self.pre_statistic[i]], axis=1) for i in range(self.pre_train_truncated_length)]

        # rnn output
        self.cur_rnn_states_list = []
        tmp_state = self.pre_rnn_state
        for i in range(self.pre_train_truncated_length):
            tmp_state = self.rnn(self.pre_ars[i], tmp_state)
            self.cur_rnn_states_list.append(tmp_state)

        # pre-train operation
        self.W = tf.Variable(self.init_matrix(shape=[self.rnn_output_dim + self.statistic_dim, self.max_item_num]))
        self.b = tf.Variable(self.init_matrix(shape=[self.max_item_num]))
        self.l2_norm = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)
        self.pn_outputs = [tf.matmul(tf.concat([self.cur_rnn_states_list[i][0], self.pre_statistic[i]], axis=1), self.W) + self.b for i in range(self.pre_train_truncated_length)]
        self.expected_pn_outputs = tf.placeholder(dtype=tf.float32, shape=[None, self.max_item_num])
        self.mask = tf.placeholder(dtype=tf.float32, shape=[None, self.max_item_num])
        self.all_zero_loss = tf.pow((1.0 / tf.reduce_sum(self.mask)) * tf.reduce_sum(tf.square(self.mask * (tf.cast(tf.constant(np.zeros(shape=[self.batch_size, self.max_item_num])), dtype=tf.float32) - self.expected_pn_outputs))), 0.5)
        self.loss = [tf.pow((1.0 / tf.reduce_sum(self.mask)) * tf.reduce_sum(tf.square(self.mask * (self.pn_outputs[i] - self.expected_pn_outputs))), 0.5) for i in range(self.pre_train_truncated_length)]
        self.pre_train_op = [tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss[i] + self.l2_norm * self.l2_factor) for i in range(self.pre_train_truncated_length)]

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def create_sru(self, rnn_input_dim, rnn_output_dim):
        print('pre-train would cover the trained rnn model\n')

        Wf = tf.Variable(self.init_matrix([rnn_input_dim, rnn_output_dim]))
        bf = tf.Variable(self.init_matrix([rnn_output_dim]))
        Wr = tf.Variable(self.init_matrix([rnn_input_dim, rnn_output_dim]))
        br = tf.Variable(self.init_matrix([rnn_output_dim]))

        U = tf.Variable(self.init_matrix([rnn_input_dim, rnn_output_dim]))
        sru_variables = [Wf, Wr, U, bf, br]

        def unit(x, h_c):
            pre_h, pre_c = tf.unstack(h_c)

            # forget gate
            f = tf.sigmoid(tf.matmul(x, Wf) + bf)
            # reset gate
            r = tf.sigmoid(tf.matmul(x, Wr) + br)
            # memory cell
            c = f * pre_c + (1 - f) * tf.matmul(x, U)
            # hidden state
            h = r * tf.nn.tanh(c) + (1 - r) * x

            return tf.stack([h, c])

        return unit, sru_variables

    def _get_initial_ars(self, seq_num=-1):
        result = [[[]], [[]], [[]]]
        if seq_num==-1:
            seq_num = self.batch_size
        for i in range(seq_num):
            item_id = random.randint(0, self.item_num - 1)
            reward = self.env[i].get_reward(item_id)
            result[0][0].append(item_id)
            result[1][0].append([reward[0]])
            result[2][0].append((self.env[i].get_statistic()))
        return result

    def train(self):
        # sample random users
        for i in range(self.batch_size):
            user_id = random.randint(0, self.boundry_user_id - 1)
            self.env[i].reset(user_id)
        ars = [[],[],[]]

        # only consider first max_item_num items
        action_value_list = np.array([range(self.max_item_num) for i in range(self.batch_size)])
        [random.shuffle(action_value_list[i]) for i in range(self.batch_size)]
        action_value_list = action_value_list[:, :self.pre_train_seq_length]

        for i in range(self.pre_train_seq_length):
            sampled_action = action_value_list[:, i]
            ars[0].append([])
            ars[1].append([])
            ars[2].append([])
            for j in range(self.batch_size):
                reward = self.env[j].get_reward(sampled_action[j])
                ars[0][-1].append(sampled_action[j])
                ars[1][-1].append(reward[0])
                ars[2][-1].append(self.env[j].get_statistic())

        if self.pre_training_steps == 0:
            self.evaluate()

        ground_truth = np.zeros(dtype=float, shape=[self.batch_size, self.max_item_num])
        mask_value = np.zeros(dtype=float, shape=[self.batch_size, self.max_item_num])
        pre_rnn_state_list = [self.sess.run(self.initial_states)]

        for i in range(self.pre_train_seq_length):
            actions = ars[0][i]
            rewards = np.array(ars[1][i])

            # calculate reward ground truth
            for index in range(len(ground_truth)):
                ground_truth[index][actions[index]] = rewards[index]

            # mask
            for index in range(len(mask_value)):
                mask_value[index][actions[index]] = 1.0
            if i >= self.pre_train_mask_length:
                pre_actions = ars[0][i - self.pre_train_mask_length]
                for index in range(len(mask_value)):
                    mask_value[index][pre_actions[index]] = 0.0

            # train rnn
            if i < self.pre_train_truncated_length:
                feed_dict = {self.pre_rnn_state: pre_rnn_state_list[0],
                             self.expected_pn_outputs: ground_truth,
                             self.mask: mask_value}
                for j in range(i + 1):
                    feed_dict[self.pre_actions[j]] = ars[0][j]
                    feed_dict[self.pre_rewards[j]] = ars[1][j]
                    feed_dict[self.pre_statistic[j]] = ars[2][j]
                _, rnn_state = self.sess.run([self.pre_train_op[i], self.cur_rnn_states_list[i]], feed_dict=feed_dict)
            else:
                feed_dict = {self.pre_rnn_state: pre_rnn_state_list[i + 1 - self.pre_train_truncated_length],
                             self.expected_pn_outputs: ground_truth,
                             self.mask: mask_value}
                for j in range(self.pre_train_truncated_length):
                    feed_dict[self.pre_actions[j]] = ars[0][i + 1 - (self.pre_train_truncated_length - j)]
                    feed_dict[self.pre_rewards[j]] = ars[1][i + 1 - (self.pre_train_truncated_length - j)]
                    feed_dict[self.pre_statistic[j]] = ars[2][i + 1 - (self.pre_train_truncated_length - j)]
                _, rnn_state = self.sess.run([self.pre_train_op[self.pre_train_truncated_length - 1], self.cur_rnn_states_list[self.pre_train_truncated_length - 1]], feed_dict=feed_dict)
            pre_rnn_state_list.append(rnn_state)

        self.pre_training_steps += 1
        print('train step%3d over' % self.pre_training_steps)
        if self.pre_training_steps % self.log_step == 0:
            self.evaluate()
            utils.pickle_save(self.sess.run(self.rnn_variables), self.rnn_file_path + 's%d' % self.pre_training_steps)

    def evaluate(self):
        # sample random users
        for i in range(self.batch_size):
            user_id = random.randint(0, self.boundry_user_id - 1)
            self.env[i].reset(user_id)
        ars = [[], [], []]

        action_value_list = np.array([range(self.max_item_num) for i in range(self.batch_size)])
        [random.shuffle(action_value_list[i]) for i in range(self.batch_size)]
        action_value_list = action_value_list[:, :self.pre_train_seq_length]

        for i in range(self.pre_train_seq_length):
            sampled_action = action_value_list[:, i]
            ars[0].append([])
            ars[1].append([])
            ars[2].append([])
            for j in range(self.batch_size):
                reward = self.env[j].get_reward(sampled_action[j])
                ars[0][-1].append(sampled_action[j])
                ars[1][-1].append(reward[0])
                ars[2][-1].append(self.env[j].get_statistic())

        ground_truth = np.zeros(dtype=float, shape=[self.batch_size, self.max_item_num])
        mask_value = np.zeros(dtype=float, shape=[self.batch_size, self.max_item_num])
        rnn_state = self.sess.run(self.initial_states)

        # evaluate
        for i in range(self.pre_train_seq_length):
            actions = ars[0][i]
            rewards = np.array(ars[1][i])

            for index in range(len(ground_truth)):
                ground_truth[index][actions[index]] = rewards[index]

            for index in range(len(mask_value)):
                mask_value[index][actions[index]] = 1.0
            if i >= self.pre_train_mask_length:
                pre_actions = ars[0][i - self.pre_train_mask_length]
                for index in range(len(mask_value)):
                    mask_value[index][pre_actions[index]] = 0.0

            feed_dict = {self.pre_rnn_state: rnn_state,
                         self.expected_pn_outputs: ground_truth,
                         self.mask: mask_value,
                         self.pre_actions[0]: ars[0][i],
                         self.pre_rewards[0]: ars[1][i],
                         self.pre_statistic[0]: ars[2][i]}
            rnn_state = self.sess.run(self.cur_rnn_states_list[0], feed_dict=feed_dict)
            print('all zero rmse:%.3f, rnn rmse:%.3f, l2:%.3f' % tuple(self.sess.run([self.all_zero_loss, self.loss[0], self.l2_norm], feed_dict=feed_dict)))

class Tree():
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.log = utils.Log()

        self.child_num = int(self.config['TPGR']['CHILD_NUM'])
        self.clustering_type = self.config['TPGR']['CLUSTERING_TYPE']
        self.clustering_vector_file_path = '../data/run_time/%s_%s_vector_v%s' % (config['ENV']['RATING_FILE'], self.config['TPGR']['CLUSTERING_VECTOR_TYPE'].lower(), self.config['TPGR']['CLUSTERING_VECTOR_VERSION'])
        self.tree_file_path = '../data/run_time/%s_tree_model_%s_%s_c%d_%s' % (self.config['ENV']['RATING_FILE'], self.config['TPGR']['CLUSTERING_VECTOR_TYPE'].lower(), self.config['TPGR']['CLUSTERING_TYPE'].lower(), self.child_num, self.config['TPGR']['TREE_VS'])

        self.env = Env(self.config)
        self.bc_dim = int(math.ceil(math.log(self.env.item_num, self.child_num)))

    def construct_tree(self):
        id_to_code, code_to_id = self.build_mapping()
        obj = {'id_to_code': id_to_code, 'code_to_id': code_to_id, 'dataset': self.config['ENV']['RATING_FILE'], 'child_num': int(self.child_num), 'clustering_type': self.config['TPGR']['CLUSTERING_TYPE']}
        utils.pickle_save(obj, self.tree_file_path)

    def build_mapping(self):
        id_to_code = np.zeros(dtype=float, shape=[self.env.item_num, self.bc_dim])
        id_to_vector = None
        if self.clustering_type!='RANDOM':
            if not os.path.exists(self.clustering_vector_file_path):
                run_time_tools.clustering_vector_constructor(self.config, self.sess)
            id_to_vector = np.loadtxt(self.clustering_vector_file_path, delimiter='\t')
        self.hierarchical_code(list(range(self.env.item_num)), (0, int(int(math.pow(self.child_num, self.bc_dim)))), id_to_code, id_to_vector)
        code_to_id = self.get_code_to_id(id_to_code)
        return (id_to_code, code_to_id)

    def get_code(self, id):
        code = np.zeros(dtype=int, shape=[self.bc_dim])
        for i in range(self.bc_dim):
            c = id % self.child_num
            code[self.bc_dim-i-1] = c
            id = int(id / self.child_num)
            if id == 0:
                break
        return code

    def kmeans_clustering(self, item_list, id_to_vector):
        if len(item_list) <= self.child_num:
            return [[item] for item in item_list] + [[] for i in range(self.child_num - len(item_list))]

        random.shuffle(item_list)
        vectors = [id_to_vector[item] for item in item_list]
        vi_to_id = {}
        id_to_vi = {}
        for i, item in zip(range(len(item_list)), item_list):
            vi_to_id[i] = item
            id_to_vi[item] = i
        kmeans = KMeans(n_clusters=self.child_num)
        kmeans.fit(vectors)
        cs = kmeans.cluster_centers_
        labels = kmeans.labels_
        ds = [[] for i in range(self.child_num)]
        for i, l in zip(range(len(labels)), labels):
            ds[l].append(vi_to_id[i])

        index2len = [(i, len(ds[i])) for i in range(self.child_num)]
        reordered_index = [item[0] for item in sorted(index2len, key = lambda x: x[1], reverse=True)]
        tmp_cs = [cs[index] for index in reordered_index]
        tmp_ds = [ds[index] for index in reordered_index]
        result_cs = list(tmp_cs)
        result_ds = []

        list_len = int(math.ceil(len(item_list) * 1.0 / self.child_num))
        non_decrease_num = self.child_num - (self.child_num * list_len - len(item_list))
        target_len = [list_len for i in range(non_decrease_num)] + [list_len - 1 for i in range(self.child_num - non_decrease_num)]

        spare_ps = []
        for i in range(self.child_num):
            tmp_d = tmp_ds[i]
            if len(tmp_d) > target_len[i]:
                result_ds.append(list(tmp_d[0: target_len[i]]))
                for j in range(target_len[i], len(tmp_d)):
                    spare_ps.append(tmp_d[j])
            else:
                result_ds.append(tmp_d)

        for i in range(self.child_num):
            num = target_len[i] - len(result_ds[i])
            if num > 0:
                p_dis_pairs = []
                for p in spare_ps:
                    p_dis_pairs.append((p, self.dis(p, result_cs[i])))
                top_n_p_dis_pairs = sorted(p_dis_pairs, key=lambda x: x[1])[:num]
                for pair in top_n_p_dis_pairs:
                    result_ds[i].append(pair[0])
                    spare_ps.remove(pair[0])

        return result_ds

    def random_clustering(self, item_list, id_to_vector):
        if len(item_list) <= self.child_num:
            return [[item] for item in item_list] + [[] for i in range(self.child_num - len(item_list))]

        random.shuffle(item_list)
        list_len = int(math.ceil(len(item_list) * 1.0 / self.child_num))
        non_decrease_num = self.child_num - (self.child_num * list_len - len(item_list))
        target_len = [list_len for i in range(non_decrease_num)] + [list_len - 1 for i in range(self.child_num - non_decrease_num)]

        result_ds = [[] for i in range(self.child_num)]
        count = 0
        for i in range(self.child_num):
            for j in range(target_len[i]):
                result_ds[i].append(item_list[count+j])
            count += target_len[i]

        return result_ds

    def pca_clustering(self, item_list, id_to_vector):
        if len(item_list) <= self.child_num:
            return [[item] for item in item_list] + [[] for i in range(self.child_num - len(item_list))]

        data = id_to_vector[item_list]
        pca = PCA(n_components=1)
        pca.fit(data)
        w = pca.components_[0]
        item_to_projection = [(item, np.dot(id_to_vector[item], w)) for item in item_list]
        result = sorted(item_to_projection, key=lambda x: x[1])

        item_list_assign = []
        list_len = int(math.ceil(len(result) * 1.0 / self.child_num))
        non_decrease_num = self.child_num - (self.child_num * list_len - len(result))
        start = 0
        end = list_len
        for i in range(self.child_num):
            item_list_assign.append([result[j][0] for j in range(start, end)])
            start = end
            if i >= non_decrease_num - 1:
                end = end + list_len - 1
            else:
                end = end + list_len
        return item_list_assign

    def hierarchical_code(self, item_list, code_range, id_to_code, id_to_vector):
        if len(item_list) == 0:
            return
        if len(item_list) == 1:
            id_to_code[item_list[0]] = self.get_code(code_range[0])
            return

        if self.clustering_type=='PCA':
            item_list_assign = self.pca_clustering(item_list, id_to_vector)
        if self.clustering_type=='KMEANS':
            item_list_assign = self.kmeans_clustering(item_list, id_to_vector)
        if self.clustering_type=='RANDOM':
            item_list_assign = self.random_clustering(item_list, id_to_vector)
        range_len = int((code_range[1]-code_range[0])/self.child_num)
        for i in range(self.child_num):
            self.hierarchical_code(item_list_assign[i], (code_range[0]+i*range_len, code_range[0]+(i+1)*range_len), id_to_code, id_to_vector)

    def dis(self, a, b):
        return np.power(np.sum(np.square(a-b)), 0.5)

    def get_code_to_id(self, id_to_code):
        result = -np.ones(shape=[int(int(math.pow(self.child_num, float(self.bc_dim))))], dtype=int)
        for i in range(len(id_to_code)):
            code = id_to_code[i]
            result[self.get_index(code)] = i
        print('leaf num count: %d\nitem num count: %d'%(len(result), int(np.sum([int(item>=0) for item in result]))))

        return result

    def get_index(self, code):
        result = 0
        for c in code:
            result = self.child_num * result + int(c)
        return result

class TPGR():
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.log = utils.Log()

        self.episode_length = int(self.config['META']['EPISODE_LENGTH'])
        self.action_dim = int(self.config['META']['ACTION_DIM'])
        self.statistic_dim = int(self.config['META']['STATISTIC_DIM'])
        self.reward_dim = int(self.config['META']['REWARD_DIM'])
        self.discount_factor = float(self.config['META']['DISCOUNT_FACTOR'])
        self.log_step = int(self.config['META']['LOG_STEP'])
        self.sample_episodes_per_batch = int(self.config['TPGR']['SAMPLE_EPISODES_PER_BATCH'])
        self.sample_users_per_batch = int(self.config['TPGR']['SAMPLE_USERS_PER_BATCH'])
        self.learning_rate = float(self.config['TPGR']['LEARNING_RATE'])
        self.l2_factor = float(self.config['TPGR']['L2_FACTOR'])
        self.entropy_factor = float(self.config['TPGR']['ENTROPY_FACTOR'])
        self.child_num = int(self.config['TPGR']['CHILD_NUM'])
        self.boundary_rating = float(self.config['ENV']['BOUNDARY_RATING'])
        self.eval_batch_size = int(self.config['TPGR']['EVAL_BATCH_SIZE'])
        self.train_batch_size = self.sample_episodes_per_batch * self.sample_users_per_batch

        self.result_file_path = '../data/result/result_log/' + time.strftime('%Y%m%d%H%M%S') + '_' + self.config['ENV']['ALPHA'] + '_' + self.config['ENV']['RATING_FILE']
        self.rnn_file_path = '../data/run_time/%s_rnn_model_%s' % (self.config['ENV']['RATING_FILE'], self.config['TPGR']['RNN_MODEL_VS'])
        self.load_model = self.config['TPGR']['LOAD_MODEL'] == 'T'
        self.load_model_path = '../data/model/%s_tpgr_model_%s' % (self.config['ENV']['RATING_FILE'], self.config['TPGR']['MODEL_LOAD_VS'])
        self.save_model_path = '../data/model/%s_tpgr_model_%s' % (self.config['ENV']['RATING_FILE'], self.config['TPGR']['MODEL_SAVE_VS'].split('s')[0])
        self.tree_file_path = '../data/run_time/%s_tree_model_%s_%s_c%d_%s' % (self.config['ENV']['RATING_FILE'], self.config['TPGR']['CLUSTERING_VECTOR_TYPE'].lower(), self.config['TPGR']['CLUSTERING_TYPE'].lower(), self.child_num, self.config['TPGR']['TREE_VS'])
        self.hidden_units = [int(item) for item in self.config['TPGR']['HIDDEN_UNITS'].split(',')] if self.config['TPGR']['HIDDEN_UNITS'].lower() != 'none' else []

        self.forward_env = Env(self.config)
        self.user_num, self.item_num, self.r_matrix, self.user_to_rele_num = self.forward_env.get_init_data()

        self.boundry_user_id = self.forward_env.boundry_user_id
        self.test_user_num = int(self.user_num/self.eval_batch_size)*self.eval_batch_size-self.boundry_user_id
        self.bc_dim = int(math.ceil(math.log(self.item_num, self.child_num)))

        self.env = [Env(self.config, self.user_num, self.item_num, self.r_matrix, self.user_to_rele_num) for i in range(max(self.train_batch_size, self.eval_batch_size * int(math.ceil(self.user_num / self.eval_batch_size))))]

        ###
        self.rnn_input_dim = self.action_dim + self.reward_dim + self.statistic_dim
        self.rnn_output_dim = self.rnn_input_dim
        self.layer_units = [self.statistic_dim + self.rnn_output_dim] + self.hidden_units + [self.child_num]

        self.is_eval = False
        self.qs_mean_list = []
        self.storage = []

        self.training_steps = 0
        if self.load_model:
            self.training_steps = int(self.config['TPGR']['MODEL_LOAD_VS'].split('s')[-1])

        tree_model = utils.pickle_load(self.tree_file_path)
        self.id_to_code, self.code_to_id = (tree_model['id_to_code'], tree_model['code_to_id'])
        self.aval_val = self.get_aval()
        self.log.log('making graph')
        self.make_graph()
        self.sess.run(tf.global_variables_initializer())
        self.log.log('graph made')

    def make_graph(self):
        # placeholders
        self.forward_action = tf.placeholder(dtype=tf.int32, shape=[None], name='forward_action')
        self.forward_reward = tf.placeholder(dtype=tf.float32, shape=[None], name='forward_reward')
        self.forward_statistic = tf.placeholder(dtype=tf.float32, shape=[None, self.statistic_dim], name='forward_statistic')
        self.forward_rnn_state = tf.placeholder(dtype=tf.float32, shape=[2, None, self.rnn_output_dim], name='forward_rnn_state')
        self.cur_q = tf.placeholder(dtype=tf.float32, shape=[None], name='cur_qs')
        self.cur_action = tf.placeholder(dtype=tf.int32, shape=[None], name='cur_actions')

        self.action_embeddings = tf.constant(dtype=tf.float32, value=self.forward_env.item_embedding)
        self.bc_embeddings = tf.constant(dtype=tf.float32, value=self.id_to_code)

        # RNN input
        self.forward_a_emb = tf.nn.embedding_lookup(self.action_embeddings, self.forward_action)
        one_hot_reward = tf.one_hot(tf.cast((self.reward_dim*(1.0-self.forward_reward)/2), dtype=tf.int32), depth=self.reward_dim)
        self.forward_ars = tf.concat([self.forward_a_emb, one_hot_reward, self.forward_statistic], axis=1)

        # RNN initial state
        self.initial_states = tf.stack([tf.zeros([self.train_batch_size, self.rnn_output_dim]), tf.zeros([self.train_batch_size, self.rnn_output_dim])])

        l = utils.pickle_load(self.rnn_file_path)
        self.rnn, self.rnn_variables = self.create_sru(l)

        # RNN state
        self.rnn_state = self.rnn(self.forward_ars, self.forward_rnn_state)
        self.user_state = tf.concat([self.rnn_state[0], self.forward_statistic], axis=1)

        if self.load_model:
            model = utils.pickle_load(self.load_model_path)
            self.W_list = [tf.Variable(model['W_list'][i], dtype=tf.float32) for i in range(len(model['W_list']))]
            self.b_list = [tf.Variable(model['b_list'][i], dtype=tf.float32) for i in range(len(model['b_list']))]
            self.result_file_path = model['result_file_path']
            self.storage = utils.pickle_load(self.result_file_path)
        else:
            self.W_list = [tf.Variable(self.init_matrix(shape=[self.node_num_before_depth_i(self.bc_dim), self.layer_units[i], self.layer_units[i + 1]]))
                           for i in range(len(self.layer_units) - 1)]
            self.b_list = [tf.Variable(self.init_matrix(shape=[self.node_num_before_depth_i(self.bc_dim), self.layer_units[i + 1]]))
                           for i in range(len(self.layer_units) - 1)]

        # map hidden state to action
        ## variables
        self.code2id = tf.constant(value=self.code_to_id, dtype=tf.int32)
        self.aval_list = tf.Variable(np.tile(np.expand_dims(self.aval_val, 1), [1, self.train_batch_size, 1]), dtype=tf.float32)
        self.aval_eval_list = tf.Variable(np.tile(np.expand_dims(self.aval_val, 1), [1, self.eval_batch_size, 1]), dtype=tf.float32)
        self.eval_probs = []

        ## constant
        self.pre_shift = tf.constant(value=np.zeros(shape=[self.train_batch_size]), dtype=tf.int32)
        self.pre_mul_choice = tf.constant(value=np.zeros(shape=[self.train_batch_size]), dtype=tf.int32)
        self.action_index = tf.constant(value=np.zeros(shape=[self.train_batch_size]), dtype=tf.int32)
        self.pre_shift_eval = tf.constant(value=np.zeros(shape=[self.eval_batch_size]), dtype=tf.int32)
        self.pre_max_choice_eval = tf.constant(value=np.zeros(shape=[self.eval_batch_size]), dtype=tf.int32)
        self.action_index_eval = tf.constant(value=np.zeros(shape=[self.eval_batch_size]), dtype=tf.int32)

        self.aval_list_t = self.aval_list
        self.aval_eval_list_t = self.aval_eval_list

        ## get action index
        ### for sampling, using multinomial
        for i in range(self.bc_dim):
            self.forward_index = self.node_num_before_depth_i(i) + self.child_num * self.pre_shift + tf.cast(self.pre_mul_choice, tf.int32)
            if i == 0:
                h = self.user_state
            else:
                h = tf.expand_dims(self.user_state, axis=1)
            for k in range(len(self.W_list)):
                if k == (len(self.W_list) - 1):
                    # for speeding up, do not use embedding_lookup when i==0.
                    if i == 0:
                        self.forward_prob = tf.nn.softmax(tf.matmul(h, self.W_list[k][0]) + self.b_list[k][0], axis=1)
                    else:
                        self.forward_prob = tf.nn.softmax(tf.squeeze(tf.matmul(h, tf.nn.embedding_lookup(self.W_list[k], self.forward_index)) +
                                                                 tf.expand_dims(tf.nn.embedding_lookup(self.b_list[k], self.forward_index), axis=1)), axis=1)
                else:
                    if i == 0:
                        h = tf.nn.relu(tf.matmul(h, self.W_list[k][0]) + self.b_list[k][0])
                    else:
                        h = tf.nn.relu(tf.matmul(h, tf.nn.embedding_lookup(self.W_list[k], self.forward_index)) +
                                   tf.expand_dims(tf.nn.embedding_lookup(self.b_list[k], self.forward_index), axis=1))

            self.pre_shift = self.child_num * self.pre_shift + tf.cast(self.pre_mul_choice, tf.int32)
            self.aval_item_num_sum = 0
            self.gather_index = tf.transpose(tf.reshape(tf.concat([tf.reshape(tf.tile(tf.expand_dims(tf.range(self.child_num), 1), [1, self.train_batch_size]), [-1, 1]),
                                                                   tf.tile(tf.concat([tf.expand_dims(tf.range(self.train_batch_size), axis=1),
                                                                                      tf.expand_dims(self.forward_index, axis=1)], axis=1), [self.child_num, 1])], axis=1),
                                                        [self.child_num, self.train_batch_size, 3]), [1, 0, 2])
            self.aval_item_num_list = tf.gather_nd(self.aval_list, self.gather_index)
            self.aval_prob = self.aval_item_num_list / tf.reduce_sum(self.aval_item_num_list, axis=1, keep_dims=True)
            self.mix_prob = tf.clip_by_value(self.forward_prob, clip_value_min=1e-30, clip_value_max=1.0) * self.aval_prob
            # self.mix_prob = self.forward_prob * (1.0 - tf.cast(tf.equal(self.aval_prob, 0.0), tf.float32))
            self.real_prob_logit = tf.log(self.mix_prob / tf.reduce_sum(self.mix_prob, axis=1, keep_dims=True))
            self.pre_mul_choice = tf.cast(tf.squeeze(tf.multinomial(logits=self.real_prob_logit, num_samples=1)), tf.float32)
            self.aval_list = self.aval_list - tf.concat([tf.expand_dims(tf.one_hot(indices=self.forward_index, depth=self.node_num_before_depth_i(self.bc_dim)) * tf.expand_dims(tf.cast(tf.equal(self.pre_mul_choice, tf.constant(np.ones(shape=[self.train_batch_size]) * j, dtype=tf.float32)), tf.float32), axis=1), axis=0) for j in range(self.child_num)], axis=0)
            self.action_index = self.action_index * self.child_num + tf.cast(self.pre_mul_choice, tf.int32)

        ### for evaluation, using maximum
        for i in range(self.bc_dim):
            self.forward_index_eval = self.node_num_before_depth_i(i) + self.child_num * self.pre_shift_eval + tf.cast(self.pre_max_choice_eval, tf.int32)
            if i == 0:
                h = self.user_state
            else:
                h = tf.expand_dims(self.user_state, axis=1)
            for k in range(len(self.W_list)):
                if k == (len(self.W_list) - 1):
                    if i == 0:
                        self.forward_prob_eval = tf.nn.softmax(tf.matmul(h, self.W_list[k][0]) + self.b_list[k][0], axis=1)
                    else:
                        self.forward_prob_eval = tf.nn.softmax(tf.squeeze(tf.matmul(h, tf.nn.embedding_lookup(self.W_list[k], self.forward_index_eval)) +
                                                                      tf.expand_dims(tf.nn.embedding_lookup(self.b_list[k], self.forward_index_eval), axis=1)), axis=1)
                else:
                    if i == 0:
                        h = tf.nn.relu(tf.matmul(h, self.W_list[k][0]) + self.b_list[k][0])
                    else:
                        h = tf.nn.relu(tf.matmul(h, tf.nn.embedding_lookup(self.W_list[k], self.forward_index_eval)) +
                                   tf.expand_dims(tf.nn.embedding_lookup(self.b_list[k], self.forward_index_eval), axis=1))

            self.eval_probs.append(self.forward_prob_eval)
            self.pre_shift_eval = self.child_num * self.pre_shift_eval + tf.cast(self.pre_max_choice_eval, tf.int32)
            self.gather_index_eval = tf.transpose(tf.reshape(tf.concat([tf.reshape(tf.tile(tf.expand_dims(tf.range(self.child_num), 1), [1, self.eval_batch_size]), [-1, 1]), tf.tile(tf.concat([tf.expand_dims(tf.range(self.eval_batch_size), axis=1), tf.expand_dims(self.forward_index_eval, axis=1)], axis=1), [self.child_num, 1])], axis=1), [self.child_num, self.eval_batch_size, 3]), [1, 0, 2])
            self.aval_item_num_eval_list = tf.gather_nd(self.aval_eval_list, self.gather_index_eval)
            self.aval_prob_eval = self.aval_item_num_eval_list / tf.reduce_sum(self.aval_item_num_eval_list, axis=1, keep_dims=True)
            self.mix_prob_eval = tf.clip_by_value(self.forward_prob_eval, clip_value_min=1e-30, clip_value_max=1.0) * self.aval_prob_eval
            self.real_prob_logit_eval = self.mix_prob_eval / tf.reduce_sum(self.mix_prob_eval, axis=1, keep_dims=True)
            self.pre_max_choice_eval = tf.cast(tf.squeeze(tf.argmax(input=self.real_prob_logit_eval, axis=1)), tf.float32)
            self.aval_eval_list = self.aval_eval_list - tf.concat([tf.expand_dims(tf.one_hot(indices=self.forward_index_eval, depth=self.node_num_before_depth_i(self.bc_dim)) * tf.expand_dims(tf.cast(tf.equal(self.pre_max_choice_eval, tf.constant(np.ones(shape=[self.eval_batch_size]) * j, dtype=tf.float32)), tf.float32), axis=1), axis=0) for j in range(self.child_num)], axis=0)
            self.action_index_eval = self.action_index_eval * self.child_num + tf.cast(self.pre_max_choice_eval, tf.int32)

        self.eval_probs = tf.concat(self.eval_probs, axis=1)

        ## update avalable children items at each node
        self.update_aval_list = tf.assign(self.aval_list_t, self.aval_list)
        self.update_aval_eval_list = tf.assign(self.aval_eval_list_t, self.aval_eval_list)

        # assign avalable children items at each node
        self.aval_eval_list_v = tf.placeholder(dtype=tf.float32, shape=self.aval_eval_list_t.get_shape())
        self.assign_aval_eval_list = tf.assign(self.aval_eval_list_t, self.aval_eval_list_v)
        self.aval_list_v = tf.placeholder(dtype=tf.float32, shape=self.aval_list_t.get_shape())
        self.assign_aval_list = tf.assign(self.aval_list_t, self.aval_list_v)

        ## get action
        self.forward_sampled_action = tf.nn.embedding_lookup(self.code_to_id, self.action_index)
        self.forward_sampled_action_eval = tf.nn.embedding_lookup(self.code_to_id, self.action_index_eval)

        ## get policy network outputs
        self.pre_c = tf.cast(tf.concat([tf.zeros(shape=[self.train_batch_size, 1]), tf.nn.embedding_lookup(self.bc_embeddings, self.cur_action)[:, 0:-1]], axis=1), dtype=tf.int32)
        self.pre_con = tf.Variable(tf.zeros(shape=[self.train_batch_size, 1], dtype=tf.int32))
        self.index = []
        for i in range(self.bc_dim):
            self.index.append(self.node_num_before_depth_i(i) + self.child_num * self.pre_con + self.pre_c[:, i:i+1])
            self.pre_con = self.pre_con * self.child_num + self.pre_c[:, i:i+1]
        self.index = tf.concat(self.index, axis=1)
        self.pn_outputs = []
        for i in range(self.bc_dim):
            h = tf.expand_dims(self.user_state, axis=1)
            for k in range(len(self.W_list)):
                if k == (len(self.W_list) - 1):
                    self.pn_outputs.append(tf.nn.softmax(tf.squeeze(tf.matmul(h, tf.nn.embedding_lookup(self.W_list[k], self.index[:, i])) +
                                                                    tf.expand_dims(tf.nn.embedding_lookup(self.b_list[k], self.index[:, i]), axis=1))))
                else:
                    h = tf.nn.relu(tf.matmul(h, tf.nn.embedding_lookup(self.W_list[k], self.index[:, i])) +
                                   tf.expand_dims(tf.nn.embedding_lookup(self.b_list[k], self.index[:, i]), axis=1))

        self.bias_variables = self.b_list + self.rnn_variables[3:]
        self.weight_variables = self.W_list + self.rnn_variables[:3]

        self.train_mse = tf.reduce_mean(tf.square(tf.concat(self.pn_outputs, axis=1) - 1.0/self.child_num), axis=1)
        self.a_code = tf.nn.embedding_lookup(self.bc_embeddings, self.cur_action)
        self.log_pi = tf.reduce_sum(
            tf.concat(
                [tf.expand_dims(
                    tf.log(tf.clip_by_value(
                        tf.gather_nd(self.pn_outputs[i], tf.concat([tf.expand_dims(tf.range(self.train_batch_size), axis=1), tf.cast(self.a_code[:, i:i+1], tf.int32)], axis=1)), clip_value_min=1e-30, clip_value_max=1.0)
                    ), axis=1)
                 for i in range(self.bc_dim)], axis=1), axis=1)
        self.negative_likelyhood = -self.log_pi
        self.l2_norm = tf.add_n([tf.nn.l2_loss(item) for item in (self.weight_variables + self.bias_variables)])
        self.weighted_negative_likelyhood_with_l2_norm = self.negative_likelyhood * self.cur_q + self.entropy_factor * self.train_mse + self.l2_factor * self.l2_norm
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.weighted_negative_likelyhood_with_l2_norm)

    # record how many items avalable in each child tree of each non-leaf node
    def get_aval(self):
        aval_list = np.zeros(shape=[self.child_num, self.node_num_before_depth_i(self.bc_dim)], dtype=int)
        self.rec_get_aval(aval_list, self.node_num_before_depth_i(self.bc_dim-1), list(map(lambda x: int(x >= 0), self.code_to_id)))
        self.log.log('get_aval completed')
        return aval_list

    def rec_get_aval(self, aval_list, start_index, l):
        if len(l) == 1:
            return
        new_l = []
        for i in range(int(len(l)/self.child_num)):
            index = start_index + i
            for j in range(self.child_num):
                aval_list[j][index] = l[self.child_num*i+j]
            new_l.append(np.sum(aval_list[: ,index]))
        self.rec_get_aval(aval_list, int(start_index/self.child_num), new_l)

    def node_num_before_depth_i(self, i):
        return int((math.pow(self.child_num, i) - 1) / (self.child_num - 1))

    def dis(self, a, b):
        return np.power(np.sum(np.square(a-b)), 0.5)

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def create_sru(self, l):
        Wf = tf.constant(l[0])
        bf = tf.constant(l[3])

        Wr = tf.constant(l[1])
        br = tf.constant(l[4])

        U = tf.constant(l[2])

        sru_variables = [Wf, Wr, U, bf, br]

        def unit(x, h_c):
            pre_h, pre_c = tf.unstack(h_c)

            # forget gate
            f = tf.sigmoid(tf.matmul(x, Wf) + bf)
            # reset gate
            r = tf.sigmoid(tf.matmul(x, Wr) + br)
            # memory cell
            c = f * pre_c + (1 - f) * tf.matmul(x, U)
            # hidden state
            h = r * tf.nn.tanh(c) + (1 - r) * x

            return tf.stack([h, c])

        return unit, sru_variables

    def standardization(self, q_matrix):
        q_matrix -= np.mean(q_matrix)
        std = np.std(q_matrix)
        if std == 0.0:
            return q_matrix
        q_matrix /= std
        return q_matrix

    def update_avalable_items(self, sampled_items):
        sampled_codes = self.id_to_code[sampled_items]
        if self.is_eval:
            aval_val_tmp = np.tile(np.expand_dims(self.aval_val, axis=1), [1, self.eval_batch_size, 1])
        else:
            aval_val_tmp = np.tile(np.expand_dims(self.aval_val, axis=1), [1, self.train_batch_size, 1])
        for i in range(len(sampled_codes)):
            code = sampled_codes[i]
            index = 0
            for c in code:
                c = int(c)
                aval_val_tmp[c][i][index] -= 1
                index = self.child_num * index + 1 + c
        if self.is_eval:
            self.sess.run(self.assign_aval_eval_list, feed_dict={self.aval_eval_list_v: aval_val_tmp})
        else:
            self.sess.run(self.assign_aval_list, feed_dict={self.aval_list_v: aval_val_tmp})
        del aval_val_tmp
        gc.collect()

    # to start with, the hidden state is all zero, for a cold-start user, here the first action is not given by the policy but random or popularity
    def _get_initial_ars(self, batch_size=-1):
        result = [[[]], [[]], [[]]]
        if batch_size==-1:
            batch_size = self.train_batch_size
        for i in range(batch_size):
            item_id = random.randint(0, self.item_num - 1)
            reward = self.env[i].get_reward(item_id)
            result[0][0].append(item_id)
            result[1][0].append(reward[0])
            result[2][0].append((self.env[i].get_statistic()))
        return result

    def train(self):
        # initialize
        for i in range(self.sample_users_per_batch):
            user_id = random.randint(0, self.boundry_user_id - 1)
            for j in range(self.sample_episodes_per_batch):
                self.env[i*self.sample_episodes_per_batch+j].reset(user_id)
        ars = self._get_initial_ars()
        self.update_avalable_items(ars[0][0])
        rnn_state = self.sess.run(self.initial_states)

        step_count = 0
        stop_flag = False
        action_list = []
        # sample actions according to the current policy
        while True:
            feed_dict = {self.forward_action: ars[0][step_count],
                         self.forward_statistic:ars[2][step_count],
                         self.forward_reward: ars[1][step_count],
                         self.forward_rnn_state: rnn_state}

            # update avalable items and sample actions in a run since different multinomial sampling would lead to different result if splitted
            run_list = [self.forward_sampled_action, self.rnn_state, self.update_aval_list]
            sampled_action, rnn_state, _ = self.sess.run(run_list, feed_dict)
            action_list.append(sampled_action)

            ars[0].append([])
            ars[1].append([])
            ars[2].append([])
            step_count += 1
            for j in range(self.train_batch_size):
                reward = self.env[j].get_reward(sampled_action[j])
                ars[0][-1].append(sampled_action[j])
                ars[1][-1].append(reward[0])
                ars[2][-1].append(self.env[j].get_statistic())
                if reward[1]:
                    stop_flag = True
            if stop_flag:
                break

        # standardize the q-values user-wisely
        qs = np.array(ars[1])[1:]
        c_reward = np.zeros([len(qs[0])])
        for i in reversed(range(len(qs))):
            c_reward = self.discount_factor * c_reward + qs[i]
            qs[i] = c_reward
        self.qs_mean_list.append(np.mean(qs))
        for i in range(self.sample_users_per_batch):
            qs[:, i * self.sample_episodes_per_batch: (i + 1) * self.sample_episodes_per_batch] = self.standardization(qs[:, i * self.sample_episodes_per_batch: (i + 1) * self.sample_episodes_per_batch])

        rnn_state = self.sess.run(self.initial_states)
        # update the policy utilizing the REINFORCE algorithm
        for i in range(step_count):
            feed_dict = {self.forward_action: ars[0][i],
                         self.forward_reward: ars[1][i],
                         self.forward_statistic: ars[2][i],
                         self.forward_rnn_state: rnn_state,
                         self.cur_action: ars[0][i + 1],
                         self.cur_q: qs[i]}
            _, rnn_state = self.sess.run([self.train_op, self.rnn_state], feed_dict=feed_dict)

        del ars
        gc.collect()

        self.training_steps += 1

        if self.training_steps % self.log_step == 0:
            print('qs means: %.5f'%np.mean(self.qs_mean_list))
            self.qs_mean_list = []

    def evaluate(self):
        # initialize
        self.is_eval = True
        eval_step_num = int(math.ceil(self.user_num / self.eval_batch_size))
        for i in range(0, self.eval_batch_size * eval_step_num):
            self.env[i].reset(i % self.user_num)
        ars = self._get_initial_ars(self.eval_batch_size * eval_step_num)
        entropy_list = []

        # sample an episode for each user
        for s in range(eval_step_num):
            start = s * self.eval_batch_size
            end = (s + 1) * self.eval_batch_size
            self.update_avalable_items(ars[0][0][start:end])
            rnn_state = np.zeros([2, self.eval_batch_size, self.rnn_output_dim])
            step_count = 0
            stop_flag = False
            entropy_list.append([])
            while True:
                feed_dict = {self.forward_action: ars[0][step_count][start:end],
                             self.forward_reward: ars[1][step_count][start:end],
                             self.forward_statistic: ars[2][step_count][start:end],
                             self.forward_rnn_state: rnn_state}
                run_list = [self.forward_sampled_action_eval, self.rnn_state, self.eval_probs, self.update_aval_eval_list]
                result_list = self.sess.run(run_list, feed_dict)
                sampled_action, rnn_state, probs = result_list[0:3]
                rmse = np.power(np.mean(np.square(probs - 1.0 / self.child_num)), 0.5)
                entropy_list[-1].append(rmse)

                step_count += 1
                if len(ars[0]) == step_count:
                    ars[0].append([])
                    ars[1].append([])
                    ars[2].append([])
                for j in range(self.eval_batch_size):
                    reward = self.env[start + j].get_reward(sampled_action[j])
                    ars[0][step_count].append(sampled_action[j])
                    ars[1][step_count].append(reward[0])
                    ars[2][step_count].append(self.env[start + j].get_statistic())
                    if reward[1]:
                        stop_flag = True
                if stop_flag:
                    break

        reward_list = np.transpose(np.array(ars[1]))[:self.user_num]
        train_ave_reward = np.mean(reward_list[:self.boundry_user_id])
        test_ave_reward = np.mean(reward_list[self.boundry_user_id:self.user_num])
        ave_rmse = np.mean(np.array(entropy_list))

        tp_list = []
        rele_list = []
        for j in range(self.user_num):
            self.forward_env.reset(j)
            ratings = [self.forward_env.get_rating(ars[0][k][j]) for k in range(0, len(ars[0]))]
            tp = len(list(filter(lambda x: x>=self.boundary_rating, ratings)))
            tp_list.append(tp)
            rele_item_num = self.forward_env.get_relevant_item_num()
            rele_list.append(rele_item_num)

        precision = np.array(tp_list) / self.episode_length
        recall = np.array(tp_list) / (np.array(rele_list) + 1e-20)
        f1 = (2 * precision * recall) / (precision + recall + 1e-20)

        train_ave_precision = np.mean(precision[:self.boundry_user_id])
        train_ave_recall = np.mean(recall[:self.boundry_user_id])
        train_ave_f1 = np.mean(f1[:self.boundry_user_id])
        test_ave_precision = np.mean(precision[self.boundry_user_id:self.user_num])
        test_ave_recall = np.mean(recall[self.boundry_user_id:self.user_num])
        test_ave_f1 = np.mean(f1[self.boundry_user_id:self.user_num])

        # save the model
        params = self.sess.run(self.W_list + self.b_list)
        model = {'W_list': params[:len(self.W_list)], 'b_list': params[len(self.W_list):], 'result_file_path': self.result_file_path}
        utils.pickle_save(model, self.save_model_path + 's%d' % self.training_steps)

        # save the result
        self.storage.append([train_ave_reward, train_ave_precision, train_ave_recall, train_ave_f1, test_ave_reward, test_ave_precision, test_ave_recall, test_ave_f1, ave_rmse])
        utils.pickle_save(self.storage, self.result_file_path)

        print('training step: %d' % (self.training_steps))
        print('\ttrain average reward over step: %2.4f, precision@%d: %.4f, recall@%d: %.4f, f1@%d: %.4f' % (train_ave_reward, self.episode_length, train_ave_precision, self.episode_length, train_ave_recall, self.episode_length, train_ave_f1))
        print('\ttest  average reward over step: %2.4f, precision@%d: %.4f, recall@%d: %.4f, f1@%d: %.4f' % (test_ave_reward, self.episode_length, test_ave_precision, self.episode_length, test_ave_recall, self.episode_length, test_ave_f1))
        print('\taverage rmse over train and test: %3.6f' % (ave_rmse))

        del ars
        gc.collect()

        self.is_eval = False
