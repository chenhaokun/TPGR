#coding: utf-8

from tpgr import *
import utils
import os

# one can enable GPU by changing the following environment parameter
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class Recommender():
    def __init__(self, config):
        self.config = config
        self.sess = tf.InteractiveSession()

    def run(self):
        pre_training_step = int(self.config['TPGR']['PRE_TRAINING_STEP'])
        max_training_step = int(self.config['META']['MAX_TRAINING_STEP'])
        log_step = int(self.config['META']['LOG_STEP'])
        log = utils.Log()

        # pre-training
        if self.config['TPGR']['PRE_TRAINING'] == 'T':
            log.log('start pre-training', True)
            pre_train = PRE_TRAIN(self.config, self.sess)
            for i in range(pre_training_step):
                pre_train.train()
            log.log('end pre-training', True)

        # constructing tree
        if self.config['TPGR']['CONSTRUCT_TREE'] == 'T':
            log.log('start constructing tree', True)
            tree = Tree(self.config, self.sess)
            tree.construct_tree()
            log.log('end constructing tree', True)

        # training model
        log.log('start training tpgr', True)
        tpgr = TPGR(self.config, self.sess)
        for i in range(0, max_training_step):
            if i % log_step == 0:
                tpgr.evaluate()
                log.log('evaluated\n', True)
            tpgr.train()
        log.log('end training tpgr')
