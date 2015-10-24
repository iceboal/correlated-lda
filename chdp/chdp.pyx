#!/usr/bin/env cython
# -*- coding: utf-8 -*-
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from cython_gsl cimport *
from cpython cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc
from collections import defaultdict
import logging
import datetime
from cymem.cymem cimport Pool
from cpython.mem cimport PyMem_Malloc, PyMem_Free
import numpy as np
cimport numpy as np
from scipy.special import polygamma, psi
from scipy.stats.mstats import gmean
import random
import cPickle as pickle
cimport openmp
from cython.parallel import parallel, prange
from libc.stdio cimport fopen, fclose, fprintf, fflush
from libc.stdint cimport *
from libc.math cimport exp, abs
from gsl_matrix cimport *

logger = logging.getLogger("CHDP")

cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int)

cdef extern from "sampler.h":
    cdef cppclass Sampler:
        Sampler(size_t burn_in, size_t &Tsize, vector[size_t] topics, uint32_t C, uint32_t D, uint32_t V,
            uint32_t *dW, uint8_t *dC, uint16_t **docs, uint8_t **dY, uint16_t **dZ,
            double *nZ, double *n0Z, double **nYZ,
            double **n0WZ, double ***n1CWZ, double **n1CZ,
            uint *t0k, uint *n0k, uint **tck, uint **nck, uint &N0_, uint *NC_, uint *ND_,
            double **U, double **X, size_t maxN, size_t maxK,
            map[size_t, uint] *tdk, map[size_t, uint] *ndk,
            double *b, double deltaNorm, double betaNorm, double *delta, double beta) except +
        vector[size_t] sampling(size_t iter, size_t *Tsize)
        uint *t0k
        uint *n0k
        double *n0Z
        double *nZ

cdef extern from "evaluate.h":
    double evaluate(size_t n_particle, int resampling,
            double **U, double **X, double *b, size_t T, size_t *topics, uint *n0k, uint **tck, uint **nck, uint N0_, uint *Nc_,
            size_t Tsize, uint t_maxWC, uint C, uint t_D, size_t V, uint8_t *t_dC, uint *t_dW, uint16_t **t_docs,
            double *nZ, double *n0Z, double **nYZ, double **n0WZ, double ***n1CWZ, double **n1CZ)

cdef np.ndarray gsl2numpy(gsl_vector *vector):
    cdef np.ndarray res = np.zeros(vector.size)
    cdef size_t i
    for i in range(vector.size):
        res[i] = gsl_vector_get(vector, i)
    return res

cdef double* alloc_1d_matrix(Pool mem, size_t A, double val):
    cdef double *m = <double*>mem.alloc(A, sizeof(double))
    cdef size_t i
    for i in range(A):
        m[i] = val
    return m

cdef double** alloc_2d_matrix(Pool mem, size_t A, size_t B, double val):
    cdef double **m = <double**>mem.alloc(A, sizeof(double*))
    cdef size_t i
    for i in range(A):
        m[i] = alloc_1d_matrix(mem, B, val)
    return m

cdef double*** alloc_3d_matrix(Pool mem, size_t A, size_t B, size_t C, double val):
    cdef double ***m = <double***>mem.alloc(A, sizeof(double**))
    cdef size_t i
    for i in range(A):
        m[i] = alloc_2d_matrix(mem, B, C, val)
    return m

cdef inline double l1(double *A, double *B, size_t n):
    cdef double sum = 0
    cdef size_t i
    for i in range(n):
        sum += abs(A[i] - B[i])

cdef class CHDP:
    cdef Pool mem
    cdef uint C, D, T0, V # D is all the docs
    cdef list Dc
    cdef uint *Tc
    cdef dict word2id
    cdef uint nW
    cdef double *n0Z
    cdef double *nZ
    cdef double **nYZ
    cdef double **n1CZ
    cdef double **n0WZ
    cdef double ***n1CWZ

    cdef size_t maxWC

    cdef double psum
    cdef double pnew

    # swap topics
    #cdef double ***z0
    #cdef double ***zc
    cdef np.ndarray z0
    cdef list zc

    cdef list pydC
    cdef list pydocs
    cdef np.uint16_t **docs
    cdef np.uint8_t *dC
    cdef uint *dW
    cdef np.uint8_t **dY
    cdef np.uint16_t **dZ
    cdef np.int8_t **dU

    # testing
    cdef bool test
    cdef list t_pydC
    cdef list t_pydocs
    cdef uint t_D
    cdef uint t_W
    cdef size_t t_maxWC
    cdef np.uint16_t **t_docs
    cdef np.uint8_t *t_dC
    cdef uint *t_dW
    cdef FILE *test_out

    cdef size_t Tsize
    cdef uint T
    cdef list topics
    cdef vector[size_t] topics_
    cdef uint *t0k
    cdef uint *n0k
    cdef uint **tck
    cdef uint **nck
    cdef uint N0_
    cdef uint *Nc_
    cdef uint *Nd_
    cdef double **U
    cdef double **X
    cdef size_t maxN
    cdef size_t maxK
    cdef double *p0
    cdef double *p1
    cdef double *p2
    cdef double *p3
    cdef map[size_t,uint] *tdk
    cdef map[size_t,uint] *ndk
    cdef double b[3]

    cdef double shape[3]
    cdef double scale

    cdef double *pZ
    cdef gsl_rng * rng

    cdef str prefix
    cdef uint nX # level of X
    cdef double delta[2]
    cdef double beta
    cdef uint n_iter
    cdef uint iter
    cdef double deltaNorm
    cdef double betaNorm
    cdef int save_interval
    cdef int eval_interval
    cdef uint burn_in

    def __init__(self, corpus_path, prefix, test_path=None, num_topics_0=10, num_topics_c=10, dic_path=None,
            delta=[1, 1], beta=.01, b=[1, 1, 1], shape=[5, 5, 0.1], scale=0.01, n_worker=-1,
                 n_iter=20, save_interval=50, eval_interval=10, burn_in=500):
        self.mem = Pool()
        # store user-supplied parameters
        self.word2id = {}
        self.V = 0
        self.C = 0
        self.pydC = []
        self.pydocs = []
        logger.info("Loading data...")
        self.nW = 0
        Dc = defaultdict(int)
        Wc = defaultdict(int)
        cdef int MIN_DOC_LENGTH = 10
        self.test = False
        cdef double ratio = 1.0
        cdef size_t i, j

        with open(corpus_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                if len(line) < MIN_DOC_LENGTH + 1:# or len(line) > 1000:
                    continue
                collection_id = int(line[0])
                line = line[1:]

                temp = []
                for word in line:
                    if word not in self.word2id:
                        self.word2id[word] = self.V
                        self.V += 1
                    temp.append(self.word2id[word])
                self.pydocs.append(temp)

                self.pydC.append(collection_id)
                Dc[collection_id] += 1
                self.nW += len(line)
                Wc[collection_id] += len(line)

        self.C = len(Dc)
        self.Dc = []
        for c in xrange(self.C):
            self.Dc.append(Dc[c])
        self.D = len(self.pydC)

        self.Tsize = 0
        self.T0 = num_topics_0
        cdef uint T = self.T0
        self.Tc = <uint*>self.mem.alloc(self.C, sizeof(uint))
        if isinstance(num_topics_c, int):
            num_topics_c = [num_topics_c] * self.C
        for i, tc in enumerate(num_topics_c):
            T += tc
            self.Tc[i] = tc
        self.Tsize = T * 2
        self.topics = range(T)

        for i in range(3):
            self.b[i] = b[i]

        if test_path:
            self.test = True
            if eval_interval > 0:
                self.test_out = fopen(prefix + '.evaluate', 'w')
            self.t_pydC = []
            self.t_pydocs = []
            with open(test_path, 'r') as f:
                for line in f:
                    line = line.strip().split()
                    collection_id = int(line[0])
                    line = line[1:]
                    self.t_pydC.append(collection_id)
                    self.t_pydocs.append([self.word2id[w] for w in line if w in self.word2id])
                    # assume at least one doc of each collection will show up in training data
            self.t_D = len(self.t_pydC)

        logger.info("Total docs: %d" % (self.D))
        logger.info("# of collections: %d" % (self.C))
        logger.info("# of vocab words: %d" % (self.V))
        logger.info("# of terms: %d" % (self.nW))
        logger.info("# of shared topics: %d" % (self.T0))
        logger.info("# of isolated topics: %s" % (', '.join([str(x) for x in num_topics_c])))
        for i in xrange(self.C):
            logger.info("Collection %d: %d" % (i, Dc[i]))
        logger.info("Running with %d threads" % n_worker)
        if self.test:
            logger.info("Test docs: %d" % (self.t_D))

        self.prefix = prefix
        self.deltaNorm = 0
        for i in range(2):
            self.delta[i] = delta[i]
            self.deltaNorm += delta[i]
        self.beta = beta
        for i in range(3):
            self.shape[i] = shape[i]
        self.scale = scale
        self.n_iter = n_iter
        self.betaNorm = beta * self.V
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.burn_in = burn_in

        logger.info("Allocating memory...")

        U = np.load('U.npy')
        self.maxN, self.maxK = U.shape
        self.U = alloc_2d_matrix(self.mem, self.maxN, self.maxK, 0)
        for i in range(self.maxN):
            for j in range(self.maxK):
                self.U[i][j] = U[i, j]
        del U
        X = np.load('X.npy')
        self.X = alloc_2d_matrix(self.mem, self.maxN, self.maxK, 0)
        for i in range(self.maxN):
            for j in range(self.maxK):
                self.X[i][j] = X[i, j]
        del X

        self.n0Z = alloc_1d_matrix(self.mem, self.Tsize, self.betaNorm)
        self.nZ = alloc_1d_matrix(self.mem, self.Tsize, self.deltaNorm)
        self.nYZ = alloc_2d_matrix(self.mem, 2, self.Tsize, 0)
        for i in range(self.Tsize):
            self.nYZ[0][i] = self.delta[0]
            self.nYZ[1][i] = self.delta[1]
        self.n1CZ = alloc_2d_matrix(self.mem, self.C, self.Tsize, self.betaNorm)
        self.n0WZ = alloc_2d_matrix(self.mem, self.V, self.Tsize, self.beta)
        self.n1CWZ = alloc_3d_matrix(self.mem, self.C, self.V, self.Tsize, self.beta)

        self.t0k = <uint*>self.mem.alloc(self.Tsize, sizeof(uint))
        self.n0k = <uint*>self.mem.alloc(self.Tsize, sizeof(uint))
        self.tck = <uint**>self.mem.alloc(self.C, sizeof(uint*))
        self.nck = <uint**>self.mem.alloc(self.C, sizeof(uint*))
        for i in range(self.C):
            self.tck[i] = <uint*>self.mem.alloc(self.Tsize, sizeof(uint))
            self.nck[i] = <uint*>self.mem.alloc(self.Tsize, sizeof(uint))
        for i in range(self.Tsize):
            self.t0k[i] = 0
            self.n0k[i] = 0
            for j in range(self.C):
                self.tck[j][i] = 0
                self.nck[j][i] = 0

        self.Nc_ = <uint*>self.mem.alloc(self.C, sizeof(uint))
        self.Nd_ = <uint*>self.mem.alloc(self.D, sizeof(uint))

        self.N0_ = 0
        for i in range(self.C):
            self.Nc_[i] = 0
        for i in range(self.D):
            self.Nd_[i] = 0

        self.p0 = alloc_1d_matrix(self.mem, self.Tsize, 0)
        self.p1 = alloc_1d_matrix(self.mem, self.Tsize, 0)
        self.p2 = alloc_1d_matrix(self.mem, self.Tsize, 0)
        self.p3 = alloc_1d_matrix(self.mem, self.Tsize, 0)
        self.pZ = alloc_1d_matrix(self.mem, self.Tsize, 0)

        #self.z0 = alloc_3d_matrix(self.mem, self.C, self.T0, self.V)
        #self.zc = <double***>self.mem.alloc(self.C, sizeof(double**))
        #for i in range(self.C):
        #    self.zc[i] = alloc_2d_matrix(self.mem, self.Tc[i], self.V)
        #self.z0 = np.zeros((self.C, self.T0, self.V))
        #self.zc = []
        #for i in range(self.C):
        #    self.zc.append(np.zeros(self.Tc[i], self.V))

        gsl_rng_env_setup()
        self.rng = gsl_rng_alloc(gsl_rng_default)
        gsl_rng_set(self.rng, 0)
        srand48(0)

    def __dealloc__(self):
        gsl_rng_free(self.rng)
        if self.test_out:
            fclose(self.test_out)

    def run(self, load=False):
        self.initialize(load)

        self.topics_.clear()
        for ii in self.topics:
            self.topics_.push_back(ii)

        cdef Sampler *sampler

        sampler = new Sampler(self.burn_in, self.Tsize, self.topics_, self.C, self.D, self.V,
            self.dW, self.dC, self.docs, self.dY, self.dZ,
            self.nZ, self.n0Z, self.nYZ,
            self.n0WZ, self.n1CWZ, self.n1CZ,
            self.t0k, self.n0k, self.tck, self.nck, self.N0_, self.Nc_, self.Nd_,
            self.U, self.X, self.maxN, self.maxK,
            self.tdk, self.ndk,
            self.b, self.deltaNorm, self.betaNorm, self.delta, self.beta)

        cdef double loglik, perplexity, cur_time, eval_time = 0
        cdef double count0
        cdef double *count = <double*>self.mem.alloc(self.C, sizeof(double))
        cdef size_t d, n, k, x, common
        cdef bool flag
        cdef int maxTCK = 0
        cdef size_t *topics
        timediff = []
        begin_time = datetime.datetime.now()
        for k in range(self.iter, self.n_iter):

            start_time = datetime.datetime.now()
            if k > self.burn_in and k % 10 == 0:
                self.update_b()
                #with open('b.txt', 'a') as f:
                #    f.write('%f, %f, %f\n' % (self.b[0], self.b[1], self.b[2]))

            count0 = 0
            ctopics = []
            for n in range(self.C):
                count[n] = 0
                x = 0
                for d in self.topics:
                    if self.tck[n][d] != 0:
                        x += 1
                ctopics.append(str(x))
            common = 0
            for d in self.topics:
                count0 += self.n0Z[d] - self.betaNorm
                flag = True
                for n in range(self.C):
                    count[n] += self.n1CZ[n][d] - self.betaNorm
                    if self.tck[n][d] == 0:
                        flag = False
                if flag:
                    common += 1
            logger.info('shared: %.1f\tindividual: %s' % (count0, ', '.join(['%.1f' % count[n] for n in range(self.C)])))
            logger.info('topic: root %d, common %d, collection: %s' % (len(self.topics), common, ', '.join(ctopics)))
            logger.info('b: %f, %f, %f' % (self.b[0], self.b[1], self.b[2]))
            for d in self.topics:
                for n in range(self.C):
                    if self.tck[n][d] > maxTCK:
                        maxTCK = self.tck[n][d]



            logger.info('Iteration %d...' % (k))
            if False: # not using alias method
                self.doSampling()
            else:
                self.topics_ = sampler.sampling(k, &self.Tsize)
                self.topics = []
                for i in xrange(self.topics_.size()):
                    self.topics.append(self.topics_[i])
                self.t0k = sampler.t0k
                self.n0k = sampler.n0k
                self.n0Z = sampler.n0Z
                self.nZ = sampler.nZ

            timediff.append(datetime.datetime.now() - start_time)
            with open('chdp.time', 'w') as f:
                f.write('\n'.join([str(t.total_seconds()) for t in timediff]))

            '''
            if k > self.burn_in and k % 30 == 0:
                logger.info('Optimizing alpha...')
                for c in xrange(self.C):
                    self.update_alpha(c)
            '''

            if self.eval_interval > 0 and self.test and k % self.eval_interval == 0:
                logger.info('Evaluating...')
                start_time = datetime.datetime.now()

                topics = <size_t*>self.mem.alloc(len(self.topics), sizeof(size_t))
                for n, d in enumerate(self.topics):
                    topics[n] = d
                loglik = evaluate(20, 0,
                    self.U, self.X, self.b, len(self.topics), topics, self.n0k, self.tck, self.nck, self.N0_, self.Nc_,
                    self.Tsize, self.t_maxWC, self.C, self.t_D, self.V, self.t_dC, self.t_dW, self.t_docs,
                    self.nZ, self.n0Z, self.nYZ, self.n0WZ, self.n1CWZ, self.n1CZ)
                perplexity = exp(- loglik / self.t_W)
                logger.info('log likelihood: %2f, perplexity: %2f' % (loglik, perplexity))
                eval_time += (datetime.datetime.now() - start_time).total_seconds()
                cur_time = (datetime.datetime.now() - begin_time).total_seconds() - eval_time
                fprintf(self.test_out, '%d\t%f\t%f\t%f\n', k, cur_time, loglik, perplexity)
                fflush(self.test_out)

            if self.save_interval > 0 and k >= self.burn_in and k % self.save_interval == 0:
                start_time = datetime.datetime.now()
                self.save(False, k)
                eval_time += (datetime.datetime.now() - start_time).total_seconds()
        del sampler

        self.save()
        logger.info('...done')
        avgdiff = sum(timediff, datetime.timedelta(0)) / len(timediff)
        logger.info('average running time of each iteration: %s' % (avgdiff))

    cdef inline double rgamma(self, double a, double b):
        return gsl_ran_gamma(self.rng, a, b)

    cdef inline double rbeta(self, double a, double b):
        return gsl_ran_beta(self.rng, a, b)

    cdef inline uint rbernoulli(self, double p):
        return gsl_ran_bernoulli(self.rng, p)

    cdef inline bool can_remove(self, int d, int c, int w, int u, uint z):
        if u < 0:
            return True
        if u == 2:
            if self.ndk[d][z] == 1:
                return True
            if self.tdk[d][z] > 1:
                return True
        if u == 1:
            if self.ndk[d][z] == 1 and self.nck[c][z] == 1:
                return True
            if self.tdk[d][z] > 1 and self.tck[c][z] > 1:
                return True
        if u == 0:
            if self.ndk[d][z] == 1 and self.nck[c][z] == 1 and self.n0k[z] == 1:
                return True
            if self.tdk[d][z] > 1 and self.tck[c][z] > 1 and self.t0k[z] > 1:
                return True
        return False

    cdef inline stat_incr(self, int d, int c, int w, int y, int u, uint z, int val):
        if u >= 0 and val < 0 and self.nck[c][z] == 0:
            raise ValueError('negative alpha, %d, %d' % (u, z))

        if u == 0:
            self.t0k[z] += val
            self.n0k[z] += val
            self.tck[c][z] += val
            self.nck[c][z] += val
            self.N0_ += val
            self.Nc_[c] += val
            if self.t0k[z] == 0:
                logger.debug('Removing root topic %d' % z)
                self.topics.remove(z)
        elif u == 1:
            self.n0k[z] += val
            self.tck[c][z] += val
            self.nck[c][z] += val
            self.N0_ += val
            self.Nc_[c] += val
        elif u == 2:
            self.nck[c][z] += val
            self.Nc_[c] += val

        self.Nd_[d] += val
        if self.tdk[d].count(z) == 0:
            if val > 0:
                self.tdk[d][z] = val
                self.ndk[d][z] = val
            else:
                raise ValueError('negative tdk, ndk')
        else:
            self.ndk[d][z] += val
            if u >= 0:
                self.tdk[d][z] += val
                if val < 0 and self.tdk[d][z] == 0:
                    self.tdk[d].erase(z)
                    self.ndk[d].erase(z)

        self.nYZ[y][z] += val
        self.nZ[z] += val
        if y == 0:
            self.n0WZ[w][z] += val
            self.n0Z[z] += val
        else:
            self.n1CWZ[c][w][z] += val
            self.n1CZ[c][z] += val

    cdef inline stirling_n(self, size_t n, size_t k):
        if n >= self.maxN or k >= self.maxK:
            raise ValueError('need pre-compute more stirling numbers: %d, %d' % (n, k))
        return self.U[n][k]

    cdef inline stirling_n_k(self, size_t n, size_t k):
        if n >= self.maxN or k >= self.maxK:
            raise ValueError('need pre-compute more stirling numbers: %d, %d' % (n, k))
        return self.X[n][k]

    def initialize(self, load=False):
        logger.info("Initializing...")
        np.random.seed(0)
        cdef size_t d
        cdef size_t n, i
        cdef list doc
        cdef size_t w
        cdef size_t y
        cdef int u
        cdef size_t z
        cdef size_t c
        if load:
            logger.info("Loading states...")
            _Y, _U, _Z, b, self.topics, self.iter, _, _, _, _ = pickle.load(open(self.prefix + '.pkl', 'rb'))
            for i in range(3):
                self.b[i] = b[i]
        else:
            self.iter = 0
        self.n_iter += self.iter

        self.dW = <uint*>self.mem.alloc(self.D, sizeof(uint)) # doc length
        self.dY = <np.uint8_t**>self.mem.alloc(self.D, sizeof(np.uint8_t*))
        self.dZ = <np.uint16_t**>self.mem.alloc(self.D, sizeof(np.uint16_t*))
        self.dU = <np.int8_t**>self.mem.alloc(self.D, sizeof(np.int8_t*))
        self.tdk = <map[size_t,uint]*>self.mem.alloc(self.D, sizeof(map[size_t,uint]))
        self.ndk = <map[size_t,uint]*>self.mem.alloc(self.D, sizeof(map[size_t,uint]))

        self.docs = <np.uint16_t**>self.mem.alloc(self.D, sizeof(np.uint16_t*))
        self.dC = <np.uint8_t*>self.mem.alloc(self.D, sizeof(np.uint8_t))

        self.docs = <np.uint16_t**>self.mem.alloc(self.D, sizeof(np.uint16_t*))
        self.dC = <np.uint8_t*>self.mem.alloc(self.D, sizeof(np.uint8_t))

        self.maxWC = 0
        for d, doc in enumerate(self.pydocs):
            N = len(doc)
            c = self.pydC[d]
            if N > self.maxWC:
                self.maxWC = N
            self.dW[d] = N
            self.dC[d] = c
            self.dY[d] = <np.uint8_t*>self.mem.alloc(N, sizeof(np.uint8_t))
            self.dZ[d] = <np.uint16_t*>self.mem.alloc(N, sizeof(np.uint16_t))
            self.dU[d] = <np.int8_t*>self.mem.alloc(N, sizeof(np.int8_t))
            self.docs[d] = <np.uint16_t*>self.mem.alloc(N, sizeof(np.uint16_t))
            self.tdk[d] = map[size_t,uint]()
            self.ndk[d] = map[size_t,uint]()

            if load:
                Y = _Y[d]
                U = _U[d]
                Z = _Z[d]
            else:
                Y = np.random.randint(2, size=N)
                Z = np.random.randint(self.Tc[c] + self.T0, size=N)

            for n, val in enumerate(zip(doc, Y, Z)):
                w, y, z = val

                if load:
                    u = U[n]
                else:
                    if z >= self.T0:
                        for i in range(c):
                            z += self.Tc[i]
                    if self.t0k[z] == 0:
                        u = 0
                    elif self.tck[c][z] == 0:
                        u = 1
                    elif self.tdk[d].count(z) == 0:
                        u = 2
                    else:
                        u = -1

                self.docs[d][n] = w
                self.dY[d][n] = y
                self.dU[d][n] = u
                self.dZ[d][n] = z

                self.stat_incr(d, c, w, y, u, z, 1)
                #if load: # only for evaluation purpose
                #    if z < self.T0 and y == 0:
                #        self.z0[c,z,w] += 1
                #    else:
                #        self.zc[c,z,w] += 1

        self.pydC =None
        self.pydocs = None

        # testing
        if self.t_pydC:
            self.t_W = 0
            self.t_maxWC = 0
            self.t_docs = <np.uint16_t**>self.mem.alloc(self.D, sizeof(np.uint16_t*))
            self.t_dC = <np.uint8_t*>self.mem.alloc(self.D, sizeof(np.uint8_t))
            self.t_dW = <uint*>self.mem.alloc(self.D, sizeof(uint)) # doc length
            for d, doc in enumerate(self.t_pydocs):
                N = len(doc)
                self.t_dW[d] = N
                self.t_W += N
                if N > self.t_maxWC:
                    self.t_maxWC = N
                self.t_dC[d] = self.t_pydC[d]
                self.t_docs[d] = <np.uint16_t*>self.mem.alloc(N, sizeof(np.uint16_t))
                for n, w in enumerate(doc):
                    self.t_docs[d][n] = w
            self.t_pydC = None
            self.t_pydocs = None

    cdef size_t addTopic(self) except +:
        logger.debug('Add topic...')
        cdef size_t i, j
        cdef size_t k
        cdef set topicSet = set(self.topics)
        for k in range(len(self.topics)+1):
            if k not in topicSet:
                break
        self.topics.append(k)

        if k >= self.Tsize: # double size
            self.Tsize *= 2
            self.t0k = <uint*>self.mem.realloc(self.t0k, self.Tsize * sizeof(uint))
            self.n0k = <uint*>self.mem.realloc(self.n0k, self.Tsize * sizeof(uint))
            for i in range(self.C):
                self.tck[i] = <uint*>self.mem.realloc(self.tck[i], self.Tsize * sizeof(uint))
                self.nck[i] = <uint*>self.mem.realloc(self.nck[i], self.Tsize * sizeof(uint))
            self.p0 = <double*>self.mem.realloc(self.p0, self.Tsize * sizeof(double))
            self.p1 = <double*>self.mem.realloc(self.p1, self.Tsize * sizeof(double))
            self.p2 = <double*>self.mem.realloc(self.p2, self.Tsize * sizeof(double))
            self.p3 = <double*>self.mem.realloc(self.p3, self.Tsize * sizeof(double))
            self.pZ = <double*>self.mem.realloc(self.pZ, self.Tsize * sizeof(double))

            self.n0Z = <double*>self.mem.realloc(self.n0Z, self.Tsize * sizeof(double))
            self.nZ = <double*>self.mem.realloc(self.nZ, self.Tsize * sizeof(double))
            self.nYZ[0] = <double*>self.mem.realloc(self.nYZ[0], self.Tsize * sizeof(double))
            self.nYZ[1] = <double*>self.mem.realloc(self.nYZ[1], self.Tsize * sizeof(double))
            for i in range(self.C):
                self.n1CZ[i] = <double*>self.mem.realloc(self.n1CZ[i], self.Tsize * sizeof(double))
                for j in range(self.V):
                    self.n1CWZ[i][j] = <double*>self.mem.realloc(self.n1CWZ[i][j], self.Tsize * sizeof(double))
            for j in range(self.V):
                self.n0WZ[j] = <double*>self.mem.realloc(self.n0WZ[j], self.Tsize * sizeof(double))

        # this should match already if from empty topics
        self.t0k[k] = 0
        self.n0k[k] = 0
        for i in range(self.C):
            self.tck[i][k] = 0
            self.nck[i][k] = 0
            self.n1CZ[i][k] = self.betaNorm
            for j in range(self.V):
                self.n1CWZ[i][j][k] = self.beta
        for j in range(self.V):
            self.n0WZ[j][k] = self.beta
        self.n0Z[k] = self.betaNorm
        self.nZ[k] = self.deltaNorm
        self.nYZ[0][k] = self.delta[0]
        self.nYZ[1][k] = self.delta[1]

        return k

    cdef void doSampling(self) except +:
        cdef size_t d, n
        self.psum = 0.0
        self.pnew = 0.0

        for d in range(self.D):
            for n in range(self.dW[d]):
                self.sample(d, n)

    cdef void sample(self, size_t d, size_t n) except +:
        cdef size_t w = self.docs[d][n]
        cdef size_t y = self.dY[d][n]
        cdef size_t z = self.dZ[d][n]
        cdef size_t c = self.dC[d]
        cdef uint k
        cdef int u = -1
        cdef double pu = float(self.tdk[d][z]) / self.ndk[d][z]
        if drand48() < pu:
            u = 2
            pu = float(self.tck[c][z]) / self.nck[c][z]
            if drand48() < pu:
                u = 1
                pu = 1.0 / self.n0k[z]
                if drand48() < pu:
                    u = 0

        if not self.can_remove(d, c, w, u, z):
            return

        # decrement counts
        self.stat_incr(d, c, w, y, u, z, -1)

        # sample z
        # word
        cdef double *pZ = self.pZ
        for k in self.topics:
            pZ[k] = (self.nYZ[0][k] * self.n0WZ[w][k] / self.n0Z[k] + \
                self.nYZ[1][k] * self.n1CWZ[c][w][k] / self.n1CZ[c][k]) / \
                self.nZ[k]

        cdef double p0sum = 0
        cdef double p1sum = 0
        cdef double p2sum = 0
        cdef double p3sum = 0
        cdef double psum = 0

        # doc
        cdef uint ndk
        cdef uint tdk
        cdef uint *nck = self.nck[c]
        cdef uint *tck = self.tck[c]
        cdef double *p0 = self.p0
        cdef double *p1 = self.p1
        cdef double *p2 = self.p2
        cdef double *p3 = self.p3
        cdef map[size_t,uint].iterator it = self.tdk[d].begin()
        while it != self.tdk[d].end():
            k = deref(it).first
            tdk = deref(it).second
            ndk = self.ndk[d][k]
            if nck[k] >= self.maxN:
                raise ValueError('large nck, %d' % nck[k])
            if tck[k] >= self.maxK:
                raise ValueError('large tck, %d' % tck[k])
            if ndk >= self.maxN:
                raise ValueError('large ndk, %d' % ndk)
            if tdk >= self.maxK:
                raise ValueError('large tdk, %d' % tdk)
            # new dish
            p1[k] = self.U[ndk][tdk] * (ndk+1-tdk)/(ndk+1) * (self.b[1]+self.Nc_[c])/self.b[2] * \
                pZ[k]
            p1sum += p1[k]
            # new doc table
            p2[k] = self.U[nck[k]][tck[k]] * self.X[ndk][tdk] * (tdk+1)/(nck[k]+1) * (nck[k]+1-tck[k])/(ndk+1) * \
                pZ[k]
            p2sum += p2[k]
            inc(it)
        psum += p1sum + p2sum

        for k in self.topics:
            #assert nck[k] < self.maxN, nck[k]
            #assert tck[k] < self.maxK, tck[k]
            if nck[k] >= self.maxN:
                raise ValueError('large nck, %d' % nck[k])
            if tck[k] >= self.maxK:
                raise ValueError('large tck, %d' % tck[k])
            if ndk >= self.maxN:
                raise ValueError('large ndk, %d' % ndk)
            if tdk >= self.maxK:
                raise ValueError('large tdk, %d' % tdk)
            if self.tdk[d].count(k) != 0:
                tdk = self.tdk[d][k]
                ndk = self.ndk[d][k]
                # new collect table
                p0[k] = self.b[1] / (self.b[0]+self.N0_) * (self.n0k[k]**2) * self.X[nck[k]][tck[k]] * \
                    self.X[ndk][tdk] * (tck[k]+1)/(self.n0k[k]+1) * (tdk+1)/(nck[k]+1) / (ndk+1) * \
                    pZ[k]
            elif self.nck[c][k] != 0:
                # new doc topic
                p0[k] = self.U[nck[k]][tck[k]] * (nck[k]+1-tck[k]) / (nck[k]+1) * \
                    pZ[k]
                # new collect table
                p3[k] = self.b[1]/(self.b[0]+self.N0_) * (self.n0k[k]**2) * \
                    self.X[nck[k]][tck[k]] * (tck[k]+1)/(self.n0k[k]+1)/(nck[k]+1) * \
                    pZ[k]
                p3sum += p3[k]
            else:
                # new collect topic
                p0[k] = self.b[1] / (self.b[0]+self.N0_) * (self.n0k[k]**2) / (self.n0k[k]+1) * \
                    pZ[k]
            p0sum += p0[k]
        psum += p0sum + p3sum

        # new root topic
        cdef double pnew = self.b[0]/(self.b[0]+self.N0_) * self.b[1] / self.V
        psum += pnew
        self.pnew += pnew
        self.psum += psum
        cdef double N = 0
        cdef double pold = psum

        psum *= drand48()
        if psum < p1sum: # new doc dish
            u = -1
            it = self.tdk[d].begin()
            z = deref(it).first
            while psum >= p1[z]:
                psum -= p1[z]
                inc(it)
                z = deref(it).first
        else:
            psum -= p1sum
            if psum < p2sum: # new doc table
                u = 2
                it = self.tdk[d].begin()
                z = deref(it).first
                while psum >= p2[z]:
                    psum -= p2[z]
                    inc(it)
                    z = deref(it).first
            else:
                psum -= p2sum
                if psum < p0sum:
                    k = 0
                    z = self.topics[k]
                    while psum >= p0[z]:
                        psum -= p0[z]
                        k += 1
                        z = self.topics[k]
                    if self.tdk[d].count(z) != 0: # new collect table
                        if self.tck[c][z] > 5000:
                            ndk = self.ndk[d][z]
                            tdk = self.tdk[d][z]
                            N = 0
                            it = self.ndk[d].begin()
                            while it != self.ndk[d].end():
                                N += deref(it).second
                                inc(it)
                        u = 1
                    elif self.nck[c][z] != 0: # new doc topic
                        u = 2
                    else: # new collect topic
                        u = 1
                else:
                    psum -= p0sum
                    if psum < p3sum: # new collect table
                        u = 1
                        for z in self.topics:
                            if self.tdk[d].count(z) != 0 or self.nck[c][z] == 0:
                                continue
                            if psum < p3[z]:
                                break
                            psum -= p3[z]
                    else: # new root topic
                        u = 0
                        z = self.addTopic()

        # sample y
        cdef double y0, y1
        y0 = self.nYZ[0][z] * self.n0WZ[w][z] / self.n0Z[z]
        y1 = self.nYZ[1][z] * self.n1CWZ[c][w][z] / self.n1CZ[c][z]
        y0 = y0 / (y0 + y1)
        if y0 > drand48():
            y = 0
        else:
            y = 1

        self.stat_incr(d, c, w, y, u, z, 1)

        # update values
        self.dY[d][n] = y
        self.dZ[d][n] = z
        self.dU[d][n] = u

    cdef update_b(self):
        cdef double n = 0
        cdef size_t k, c, step, d
        cdef uint T = len(self.topics)
        for k in self.topics:
            n += self.n0k[k]

        '''
        cdef double eta = self.rbeta(self.b[0] + 1, n)
        cdef double pi = self.shape[0] + T - 1
        cdef double rate = 1.0 / self.scale - log(eta)
        pi = pi / (pi + rate * n)

        cdef uint cc = self.rbernoulli(pi)
        if cc == 1:
            self.b[0] = self.rgamma(self.shape[0] + T, 1.0 / rate)
        else:
            self.b[0] = self.rgamma(self.shape[0] + T - 1, 1.0 / rate)
        '''

        '''
        cdef double nn
        for c in range(self.C):
            nn = 0
            T = 0
            for k in self.topics:
                nn += self.nck[c][k]
                if self.tck[c][k] != 0:
                    T += 1
            eta = self.rbeta(self.bc[c] + 1, nn)
            pi = self.shape[0] + T - 1
            rate = 1.0 / self.scale - log(eta)
            pi = pi / (pi + rate * nn)
            cc = self.rbernoulli(pi)
            if cc == 1:
                self.bc[c] = self.rgamma(self.shape[0] + T, 1.0 / rate)
            else:
                self.bc[c] = self.rgamma(self.shape[0] + T - 1, 1.0 / rate)
        '''

        cdef double sum_log_w, sum_s

        for step in range(20):
            sum_log_w = log(self.rbeta(self.b[0] + 1, n))
            sum_s = self.rbernoulli(n / (n + self.b[0]))
            rate = 1.0 / self.scale - sum_log_w
            self.b[0] = self.rgamma(self.shape[0] + T - sum_s, 1.0 / rate)

        cdef double *nn = <double*>self.mem.alloc(self.C, sizeof(double))
        for c in range(self.C):
            nn[c] = 0
            for k in self.topics:
                nn[c] += self.nck[c][k]

        for step in range(20):
            sum_log_w = 0
            sum_s = 0
            for c in range(self.C):
                sum_log_w += log(self.rbeta(self.b[1] + 1, nn[c]))
                sum_s += self.rbernoulli(nn[c] / (nn[c] + self.b[1]))
            rate = 1.0 / self.scale - sum_log_w
            self.b[1] = self.rgamma(self.shape[1] + n - sum_s, 1.0 / rate)

        n = 0
        for c in range(self.C):
            for k in self.topics:
                n += self.nck[c][k]

        for step in range(20):
            sum_log_w = 0
            sum_s = 0
            for d in range(self.D):
                sum_log_w += log(self.rbeta(self.b[2] + 1, self.dW[d]))
                sum_s += self.rbernoulli(self.dW[d] / (self.dW[d] + self.b[2]))
            rate = 1.0 / self.scale - sum_log_w
            self.b[2] = self.rgamma(self.shape[2] + n - sum_s, 1.0 / rate)


    def evaluate(self, resample=False, single=False, T0=None):

        cdef double scale = 0
        cdef size_t w, y, z, c, d, k
        '''
        if not single:
            assert self.save_interval > 0
            gsl_2d_matrix_set_all(self.alpha, self.C, 0)
            gsl_2d_matrix_set_all(self.nDZ, self.D, 0)
            gsl_2d_matrix_set_all(self.nYZ, 2, 0)
            gsl_vector_set_all(self.nZ, 0)
            gsl_2d_matrix_set_all(self.n0WZ, self.V, 0)
            gsl_vector_set_all(self.n0Z, 0)
            gsl_3d_matrix_set_all(self.n1CWZ, self.C, self.V, 0)
            gsl_2d_matrix_set_all(self.n1CZ, self.C, 0)
            for i in range(self.C):
                if self.Tc[i] - self.T0 <= 0:
                    continue
                gsl_vector_set_all(self.nCZ[i], 0)
                gsl_2d_matrix_set_all(self.nCWZ[i], self.V, 0)

            for k in range(self.n_iter):
                if k >= self.burn_in and k % self.save_interval == 0:
                    scale += 1
                    Y, U, Z, b, _, _, _, _, _, _, _ = pickle.load(open('%s-%04d.pkl' % (self.prefix, k), 'rb'))
                    for c in range(3):
                        self.b[c] = b[c]
                    for d in range(self.D):
                        c = self.dC[d]
                        for n in range(self.dW[d]):
                            w = self.docs[d][n]
                            y = Y[d][n]
                            u = U[d][n]
                            z = Z[d][n]
                            self.stat_incr(d, c, w, y, u, z, 1)
            scale = 1.0 / scale

            gsl_2d_matrix_scale(self.alpha, self.C, scale)
            gsl_2d_matrix_scale(self.nDZ, self.D, scale)
            gsl_2d_matrix_scale(self.nYZ, 2, scale)
            gsl_vector_scale(self.nZ, scale)
            gsl_2d_matrix_scale(self.n0WZ, self.V, scale)
            gsl_vector_scale(self.n0Z, scale)
            gsl_3d_matrix_scale(self.n1CWZ, self.C, self.V, scale)
            gsl_2d_matrix_scale(self.n1CZ, self.C, scale)
            for i in range(self.C):
                if self.Tc[i] - self.T0 <= 0:
                    continue
                gsl_vector_scale(self.nCZ[i], scale)
                gsl_2d_matrix_scale(self.nCWZ[i], self.V, scale)

            gsl_vector_add_constant(self.nYZ[0], self.delta[0])
            gsl_vector_add_constant(self.nYZ[1], self.delta[1])
            gsl_vector_add_constant(self.nZ, self.deltaNorm)
            gsl_2d_matrix_add_constant(self.n0WZ, self.V, self.beta)
            gsl_vector_add_constant(self.n0Z, self.betaNorm)
            gsl_3d_matrix_add_constant(self.n1CWZ, self.C, self.V, self.beta)
            gsl_2d_matrix_add_constant(self.n1CZ, self.C, self.betaNorm)
            for i in range(self.C):
                if self.Tc[i] - self.T0 <= 0:
                    continue
                gsl_vector_add_constant(self.nCZ[i], self.betaNorm)
                gsl_2d_matrix_add_constant(self.nCWZ[i], self.V, self.beta)
        else:
        '''
        if False: # experiment in the paper
            #self.initialize(load=True)
            ys = []
            for i in self.topics:
                if (self.nZ[i] - self.deltaNorm) <= 1e-5:
                    ys.append(0)
                else:
                    ys.append((self.nYZ[0][i] - self.delta[0]) / (self.nZ[i] - self.deltaNorm))
            ys = sorted(ys)
            T = len(ys)
            a = 0
            for i in range(T - T0, T):
                a += ys[i]
            b = 0
            for i in range(0, T - T0):
                b += ys[i]
            if T0 != 0:
                a /= T0
            if T0 != T:
                b /= (T - T0)
            return a, b, a - b

        cdef double loglik, perplexity
        cdef int f
        f = 1 if resample else 0
        cdef size_t *topics = <size_t*>self.mem.alloc(len(self.topics), sizeof(size_t))
        for i, k in enumerate(self.topics):
            topics[i] = k

        loglik = evaluate(20, f,
            self.U, self.X, self.b, len(self.topics), topics, self.n0k, self.tck, self.nck, self.N0_, self.Nc_,
            self.Tsize, self.t_maxWC, self.C, self.t_D, self.V, self.t_dC, self.t_dW, self.t_docs,
            self.nZ, self.n0Z, self.nYZ, self.n0WZ, self.n1CWZ, self.n1CZ)
        perplexity = exp(- loglik / self.t_W)
        logger.info('log likelihood: %2f, perplexity: %2f' % (loglik, perplexity))
        return perplexity

    def save(self, stats=False, k=None):
        cdef size_t d
        cdef size_t n
        Y = []
        U = []
        Z = []
        Tc = []
        for i in range(self.C):
            Tc.append(self.Tc[i])
        b = []
        for i in range(3):
            b.append(self.b[i])

        for d in range(self.D):
            c = self.dC[d]
            dw = [self.docs[d][n] for n in range(self.dW[d])]
            dy = [self.dY[d][n] for n in range(self.dW[d])]
            du = [self.dU[d][n] for n in range(self.dW[d])]
            dz = [self.dZ[d][n] for n in range(self.dW[d])]
            Y.append(dy)
            U.append(du)
            Z.append(dz)
        prefix = self.prefix
        n_iter = self.n_iter
        if k:
            prefix = prefix + ('-%04d' % (k))
            n_iter = k
        pickle.dump((Y, U, Z, b, self.topics, n_iter, self.V, self.C, self.D, self.word2id),
                open(prefix + '.pkl', 'wb'))
