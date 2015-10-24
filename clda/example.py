#!/usr/bin/env python
# -*- coding: utf-8 -*-

from clda import CLDA
import logging

logging.basicConfig(format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)


if __name__ == '__main__':
    clda = CLDA('../data/jstor.clda', 'jstor',
                 delta=[1, 1], num_topics_0=10, num_topics_c=[0, 10], alpha=1,
                 n_iter=100, burn_in=20, n_worker=-1, eval_interval=-1, save_interval=-1)
    clda.run(load=False)
    clda.save()
