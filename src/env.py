#coding: utf-8

from scipy.sparse import coo_matrix
from tqdm import tqdm
import run_time_tools
import numpy as np
import utils
import os

class Env():
    def __init__(self, config, user_num=None, item_num=None, r_matrix=None, user_to_rele_num=None):
        self.config = config
        self.action_dim = int(self.config['META']['ACTION_DIM'])
        self.episode_length = int(self.config['META']['EPISODE_LENGTH'])
        self.alpha = float(self.config['ENV']['ALPHA'])
        self.boundary_rating = float(self.config['ENV']['BOUNDARY_RATING'])
        self.log = utils.Log()
        # to normalize the reward into (-1, 1], reward = self.a * rating + self.b
        self.a = 2.0 / (float(self.config['ENV']['MAX_RATING']) - float(self.config['ENV']['MIN_RATING']))
        self.b = - (float(self.config['ENV']['MAX_RATING']) + float(self.config['ENV']['MIN_RATING'])) / (float(self.config['ENV']['MAX_RATING']) - float(self.config['ENV']['MIN_RATING']))

        if not user_num is None:
            self.user_num = user_num
            self.item_num = item_num
            self.r_matrix = r_matrix
            self.user_to_rele_num = user_to_rele_num
            self.boundry_user_id = int(self.user_num * 0.8)
            self.test_user_num = self.user_num - self.boundry_user_id
        else:
            rating_file_path = '../data/rating/' + self.config['ENV']['RATING_FILE']
            rating = np.loadtxt(fname=rating_file_path, delimiter='\t')

            self.user = set()
            self.item = set()
            for i,j,k in rating:
                self.user.add(int(i))
                self.item.add(int(j))

            self.user_num = len(self.user)
            self.item_num = len(self.item)
            self.boundry_user_id = int(self.user_num * 0.8)
            self.test_user_num = self.user_num - self.boundry_user_id

            # if you replace the rating file without renaming, you should delete the old env object file before you run the code
            env_object_path = '../data/run_time/%s_env_objects'%self.config['ENV']['RATING_FILE']
            if os.path.exists(env_object_path):
                objects = utils.pickle_load(env_object_path)
                self.r_matrix = objects['r_matrix']
                self.user_to_rele_num = objects['user_to_rele_num']
            else:
                self.r_matrix = coo_matrix((rating[:, 2], (rating[:, 0].astype(int), rating[:, 1].astype(int)))).todok()

                self.log.log('construct relevant item number for each user')
                self.user_to_rele_num = {}
                for u in tqdm(range(self.user_num)):
                    rele_num = 0
                    for i in range(self.item_num):
                        if self.r_matrix[u, i] >= self.boundary_rating:
                            rele_num += 1
                    self.user_to_rele_num[u] = rele_num

                # dump the env object
                utils.pickle_save({'r_matrix': self.r_matrix, 'user_to_rele_num': self.user_to_rele_num}, env_object_path)

            item_embedding_file_path = '../data/run_time/' + self.config['ENV']['RATING_FILE'] + '_item_embedding_dim%d'%self.action_dim
            if not os.path.exists(item_embedding_file_path):
                run_time_tools.mf_with_bias(self.config)
            self.item_embedding = np.loadtxt(item_embedding_file_path, dtype=float, delimiter='\t')

    def get_init_data(self):
        return self.user_num, self.item_num, self.r_matrix, self.user_to_rele_num

    def reset(self, user_id):
        self.user_id = user_id
        self.step_count = 0
        self.con_neg_count = 0
        self.con_pos_count = 0
        self.con_zero_count = 0
        self.con_not_neg_count = 0
        self.con_not_pos_count = 0
        self.all_neg_count = 0
        self.all_pos_count = 0
        self.history_items = set()

    def get_relevant_item_num(self):
        return self.user_to_rele_num[self.user_id]

    def get_reward(self, item_id):
        reward = [0.0, False]
        if item_id in self.history_items:
            print('recommending repeated item')
            exit(0)
        else:
            r = self.get_rating(item_id)
            if r == 0:
                pass
            else:
                # normalize the reward value
                reward[0] = self.a * r + self.b

        self.step_count += 1
        sr = self.con_pos_count - self.con_neg_count
        if reward[0] < 0:
            self.con_neg_count += 1
            self.all_neg_count += 1
            self.con_not_pos_count += 1
            self.con_pos_count = 0
            self.con_not_neg_count = 0
            self.con_zero_count = 0
        elif reward[0] > 0:
            self.con_pos_count += 1
            self.all_pos_count += 1
            self.con_not_neg_count += 1
            self.con_neg_count = 0
            self.con_not_pos_count = 0
            self.con_zero_count = 0
        else:
            self.con_not_neg_count += 1
            self.con_not_pos_count += 1
            self.con_zero_count += 1
            self.con_pos_count = 0
            self.con_neg_count = 0

        self.history_items.add(item_id)

        if self.step_count == self.episode_length or len(self.history_items) == self.item_num:
            reward[1] = True

        reward[0] += self.alpha * sr
        return (reward[0], reward[1])

    def get_statistic(self):
        all_neg_count = self.all_neg_count
        all_pos_count = self.all_pos_count
        all_count = len(self.history_items)
        zero_count = all_count - all_neg_count - all_pos_count
        step_count = self.step_count
        con_neg_count = self.con_neg_count
        con_pos_count = self.con_pos_count
        con_zero_count = self.con_zero_count
        con_not_neg_count = self.con_not_neg_count
        con_not_pos_count = self.con_not_pos_count
        result = [all_neg_count, all_pos_count, zero_count, step_count, con_neg_count, con_pos_count, con_zero_count, con_not_neg_count, con_not_pos_count]
        return [item / float(self.episode_length) for item in result]

    def get_rating(self, item_id):
        return self.r_matrix[self.user_id, item_id]
