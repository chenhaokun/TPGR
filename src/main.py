# coding: utf-8

from recommender import Recommender
import configparser

def main():
    config = configparser.ConfigParser()
    config.read_file(open('../config'))

    show_info(config)

    rec = Recommender(config)
    rec.run()

def show_info(config):
    print('rating file: %s' % config['ENV']['RATING_FILE'])
    print('alpha: %.1f' % float(config['ENV']['ALPHA']))

if __name__ == '__main__':
    main()
