#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chdp import CHDP
import logging

logging.basicConfig(format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)

if __name__ == '__main__':
    chdp = CHDP('../data/jstor.clda', 'chdp',
            delta=[1, 1], num_topics_0=10, num_topics_c=[5, 5],
            n_iter=50, burn_in=10, eval_interval=-1, save_interval=-1)
    chdp.run(load=False)
    chdp.save()
