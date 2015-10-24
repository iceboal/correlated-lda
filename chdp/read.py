#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import cPickle as pickle

'''
Parameters to set
'''
vocab_file = '../data/jstor.vocab'
corpus_file = '../data/jstor.clda'
result_file = 'chdp.pkl'
title = ['Science', 'Humanities']



id2word = dict(enumerate([x.strip() for x in open(vocab_file, 'r')]))

Y, U, Z, b, topics, n_iter, V, C, D, word2id = pickle.load(open(result_file, 'rb'))
T = len(topics)
topic2id = {}
for i, k in enumerate(topics):
    topic2id[k] = i
z0 = np.zeros((T, V), dtype=np.uint32)
z = np.zeros((T, V, C), dtype=np.uint32)
zc = np.zeros((T, V, C), dtype=np.uint32)

with open(corpus_file, 'r') as f:
    k = 0
    for line in f:
        line = line.strip().split()
        c = int(line[0])
        dy, dz = Y[k], Z[k]
        line = line[1:]
        assert len(line) == len(dy)
        for w, y, iz, in zip(line, dy, dz):
            w = word2id[w]
            iz = topic2id[iz]
            z[iz, w, c] += 1
            if y == 0:
                z0[iz, w] += 1
            else:
                zc[iz, w, c] += 1
        k += 1

id2id = dict([(idx, int(word)) for word, idx in word2id.items()])

N = z0.shape[1]

mapping = []
for idx in xrange(N):
    mapping.append(id2word[id2id[idx]])
mapping = np.asarray(mapping)

z0 = np.array(z0, dtype=float)
zc = np.array(zc, dtype=float)
z = np.array(z, dtype=float)

n0 = z0.sum(1)
n0c = z.sum(1)

n = n0.sum() + n0c.sum()

n0 /= n
n0c /= n

z0 /= z0.sum(1)[:,np.newaxis]
t0 = z0.argsort(1)[:,::-1][:,:10]
t0c = np.zeros((T, 10, C))
for c in xrange(C):
    z[:,:,c] /= z[:,:,c].sum(1)[:,np.newaxis]
    zc[:,:,c] /= zc[:,:,c].sum(1)[:,np.newaxis]
    t0c[:,:,c] = z[:,:,c].argsort(1)[:,::-1][:,:10]
dist = np.zeros(T)
for k in xrange(T):
    dist[k] = np.sum(np.abs(zc[k,:,0] - zc[k,:,0]))
order = dist.argsort()
t0 = t0[order]
t0c = t0c[order]

print """
<style>
table { border: none; border-collapse: collapse; }
table td { border-left: 1px solid #000; }
table td:first-child { border-left: none; }
table, th { border: 1px solid black; }
th { text-align: left; }
</style>
"""

for i, a in enumerate(zip(t0, t0c, n0, n0c)):
    a0, a0c, an0, an0c = a
    print "<h4>Topic %d (%.2g%%)</h4>" % (i+1, (an0+an0c.sum()) * 100)
    print '<table>'
    print '<tr>'
    print '\n'.join(['<th>' + x + '</th>' for x in ['shared'] + title])
    print '</tr>'
    print '<tr>'
    print '\n'.join(['<th>%.2g%%</th>' % (x*100) for x in [an0] + an0c.tolist()])
    print '</tr>'
    for a in zip(a0, *zip(*a0c)):
        print '<tr>'
        print '\n'.join(['<td>'+mapping[x]+'</td>' for x in a])
        print '</tr>'
    print '</table>'
