#coding: utf-8

import pickle
import time

class Log():
    def log(self, text, log_time=False):
        print('log: %s' % text)
        if log_time:
            print('time: %s' % time.asctime(time.localtime(time.time())))

def pickle_save(object, file_path):
    f = open(file_path, 'wb')
    pickle.dump(object, f)

def pickle_load(file_path):
    f = open(file_path, 'rb')
    return pickle.load(f)
