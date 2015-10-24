#!/usr/bin/env cython
# -*- coding: utf-8 -*-
# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from cython_gsl cimport *
from cpython cimport bool
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

logger = logging.getLogger("CLDA")

cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int)

cdef extern from "utils.h":
    void count_hist(double **histZW, uint *nonZeroLimits, size_t c, double uint,
            size_t maxW, size_t Tc, size_t D, uint8_t *dC, uint *dW, uint16_t **dZ)

cdef extern from "sampler.h":
    cdef cppclass Sampler:
        Sampler(size_t burn_in, uint32_t T0, uint32_t *Tc, uint32_t C, uint32_t D, uint32_t V, double *unit,
            uint32_t *dW, uint8_t *dC, uint16_t **docs, uint8_t **dY, uint16_t **dZ, gsl_vector **alpha,
            gsl_vector **nDZ, gsl_vector *nZ, gsl_vector *n0Z, gsl_vector **nYZ,
            gsl_vector **n0WZ, gsl_vector ***n1CWZ, gsl_vector **n1CZ, gsl_vector ***nCWZ,
            gsl_vector **nCZ)  except +
        void sampling(size_t iter)

cdef extern from "evaluate.h":
    double evaluate(size_t n_particle, int resampling,
            uint *Tc, uint T0, uint t_maxWC, uint C, uint t_D, uint8_t *t_dC, uint *t_dW, uint16_t **t_docs,
            gsl_vector **alpha, double *unit,
            gsl_vector *nZ, gsl_vector *n0Z, gsl_vector **nYZ, gsl_vector **n0WZ, gsl_vector ***n1CWZ, gsl_vector **n1CZ, gsl_vector ***nCWZ, gsl_vector **nCZ)

cdef np.ndarray gsl2numpy(gsl_vector *vector):
    cdef np.ndarray res = np.zeros(vector.size)
    cdef size_t i
    for i in range(vector.size):
        res[i] = gsl_vector_get(vector, i)
    return res

cdef double* alloc_1d_matrix(Pool mem, size_t A):
    cdef double *m = <double*>mem.alloc(A, sizeof(double))
    cdef size_t i
    for i in range(A):
        m[i] = 0
    return m

cdef double** alloc_2d_matrix(Pool mem, size_t A, size_t B):
    cdef double **m = <double**>mem.alloc(A, sizeof(double*))
    cdef size_t i
    for i in range(A):
        m[i] = alloc_1d_matrix(mem, B)
    return m

cdef double*** alloc_3d_matrix(Pool mem, size_t A, size_t B, size_t C):
    cdef double ***m = <double***>mem.alloc(A, sizeof(double**))
    cdef size_t i
    for i in range(A):
        m[i] = alloc_2d_matrix(mem, B, C)
    return m

cdef inline double l1(double *A, double *B, size_t n):
    cdef double sum = 0
    cdef size_t i
    for i in range(n):
        sum += abs(A[i] - B[i])

cdef class CLDA:
    cdef Pool mem
    cdef uint C, D, T0, V # D is all the docs
    cdef list Dc
    cdef uint *Tc
    cdef dict word2id
    cdef uint nW
    cdef gsl_vector *n0Z
    cdef gsl_vector *nZ
    cdef gsl_vector **nYZ
    cdef gsl_vector **nCZ
    cdef gsl_vector **n1CZ
    cdef gsl_vector **n0WZ
    cdef gsl_vector ***n1CWZ
    cdef gsl_vector ***nCWZ
    cdef gsl_vector **nDZ

    # optimize hyperparameters
    cdef size_t *maxWC
    cdef double **histCW
    cdef double ***histCZW

    # swap topics
    #cdef double ***z0
    #cdef double ***zc
    cdef np.ndarray z0
    cdef list zc

    cdef gsl_rng * rng
    cdef gsl_ran_discrete_t *disc
    cdef gsl_vector **pZC
    cdef gsl_vector *p0
    cdef gsl_vector *p00

    cdef list pydC
    cdef list pydocs
    cdef np.uint16_t **docs
    cdef np.uint8_t *dC
    cdef uint *dW
    cdef np.uint8_t **dY
    cdef np.uint16_t **dZ

    # testing
    cdef bool test
    cdef list t_pydC
    cdef list t_pydocs
    cdef uint t_D
    cdef uint t_W
    cdef uint t_maxWC
    cdef np.uint16_t **t_docs
    cdef np.uint8_t *t_dC
    cdef uint *t_dW
    cdef FILE *test_out

    cdef double *unit # to balance collections

    cdef str prefix
    cdef uint nX # level of X
    cdef double delta[2]
    cdef gsl_vector **alpha
    cdef double beta
    cdef uint n_iter
    cdef uint iter
    cdef double deltaNorm
    cdef double betaNorm
    cdef int save_interval
    cdef int eval_interval
    cdef uint burn_in

    def __init__(self, corpus_path, prefix, test_path=None, num_topics_0=10, num_topics_c=10,
            alpha=50.0, delta=[1, 1], beta=.01, n_worker=-1,
                 n_iter=20, save_interval=50, eval_interval=10, burn_in=500):
        self.mem = Pool()
        n_thread = openmp.omp_get_max_threads()
        if n_worker < 0:
            if n_thread + n_worker > 0:
                n_worker += n_thread
            else:
                n_worker = 1
        if n_worker > 0 and n_worker < n_thread:
            openmp.omp_set_num_threads(n_worker)
        else:
            n_worker = n_thread
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
                if len(line) < MIN_DOC_LENGTH + 1:
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

        self.T0 = num_topics_0
        self.Tc = <uint*>self.mem.alloc(self.C, sizeof(uint))
        if isinstance(num_topics_c, int):
            num_topics_c = [num_topics_c] * self.C
        for i, tc in enumerate(num_topics_c):
            tc += self.T0
            self.Tc[i] = tc
        unit = np.array([1.0*Wc[i]/self.Tc[i] for i in range(self.C)])
        unit = gmean(unit) / unit
        #unit = np.ones(self.C)
        self.unit = <double*>self.mem.alloc(self.C, sizeof(double))
        for i in range(self.C):
            self.unit[i] = unit[i]

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
        self.alpha = <gsl_vector**>PyMem_Malloc(self.C * sizeof(gsl_vector*))
        for c in xrange(self.C):
            self.alpha[c] = gsl_vector_alloc(self.Tc[c])
            gsl_vector_set_all(self.alpha[c], 1.0 * alpha / self.Tc[c])
        self.beta = beta
        self.n_iter = n_iter
        self.betaNorm = beta * self.V
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.burn_in = burn_in

        logger.info("Allocating memory...")

        self.n0Z = gsl_vector_alloc(self.T0)
        gsl_vector_set_all(self.n0Z, self.betaNorm)
        self.nZ = gsl_vector_alloc(self.T0)
        gsl_vector_set_all(self.nZ, self.deltaNorm)
        self.nYZ = alloc_2d_gsl_matrix(2, self.T0, 0)
        gsl_vector_set_all(self.nYZ[0], self.delta[0])
        gsl_vector_set_all(self.nYZ[1], self.delta[1])
        self.n1CZ = alloc_2d_gsl_matrix(self.C, self.T0, self.betaNorm)
        self.n0WZ = alloc_2d_gsl_matrix(self.V, self.T0, self.beta)
        self.n1CWZ = alloc_3d_gsl_matrix(self.C, self.V, self.T0, self.beta)

        self.nCZ = <gsl_vector**>PyMem_Malloc(self.C * sizeof(gsl_vector*))
        for i in range(self.C):
            if self.Tc[i] - self.T0 <= 0:
                continue
            self.nCZ[i] = gsl_vector_alloc(self.Tc[i] - self.T0)
            gsl_vector_set_all(self.nCZ[i], self.betaNorm)

        self.nCWZ = <gsl_vector***>PyMem_Malloc(self.C * sizeof(gsl_vector**))
        for i in range(self.C):
            self.nCWZ[i] = <gsl_vector**>PyMem_Malloc(self.V * sizeof(gsl_vector*))
            if self.Tc[i] - self.T0 <= 0:
                continue
            for j in range(self.V):
                self.nCWZ[i][j] = gsl_vector_alloc(self.Tc[i] - self.T0)
                gsl_vector_set_all(self.nCWZ[i][j], self.beta)

        self.nDZ = <gsl_vector**>PyMem_Malloc(self.D * sizeof(gsl_vector*))
        for i in range(self.D):
            self.nDZ[i] = gsl_vector_alloc(self.Tc[self.pydC[i]])
            gsl_vector_set_all(self.nDZ[i], 0)

        gsl_rng_env_setup()
        self.rng = gsl_rng_alloc(gsl_rng_default)

        self.p0 = gsl_vector_alloc(self.T0)
        self.p00 = gsl_vector_alloc(self.T0)
        self.pZC = <gsl_vector**>PyMem_Malloc(self.C * sizeof(gsl_vector*))
        for i in range(self.C):
            self.pZC[i] = gsl_vector_alloc(self.Tc[i])

        gsl_rng_set(self.rng, 0)
        srand48(0)


    def __dealloc__(self):

        if self.test_out:
            fclose(self.test_out)

        gsl_vector_free(self.n0Z)
        gsl_vector_free(self.nZ)
        free_2d_gsl_matrix(self.nYZ, 2)
        free_2d_gsl_matrix(self.n1CZ, self.C)
        free_2d_gsl_matrix(self.n0WZ, self.V)
        free_3d_gsl_matrix(self.n1CWZ, self.C, self.V)
        free_2d_gsl_matrix(self.nDZ, self.D)

        for c in xrange(self.C):
            if self.Tc[c] - self.T0 > 0:
                gsl_vector_free(self.nCZ[c])
                free_2d_gsl_matrix(self.nCWZ[c], self.V)
        PyMem_Free(self.nCZ)
        PyMem_Free(self.nCWZ)

        gsl_rng_free(self.rng)
        gsl_vector_free(self.p0)
        gsl_vector_free(self.p00)
        free_2d_gsl_matrix(self.pZC, self.C)
        free_2d_gsl_matrix(self.alpha, self.C)

    def run(self, load=False):
        self.initialize(load)

        cdef Sampler *sampler
        sampler = new Sampler(self.burn_in, self.T0, self.Tc, self.C, self.D, self.V, self.unit,
                                  self.dW, self.dC, self.docs, self.dY, self.dZ, self.alpha,
                                  self.nDZ, self.nZ, self.n0Z, self.nYZ, self.n0WZ, self.n1CWZ, self.n1CZ, self.nCWZ, self.nCZ)
        cdef double loglik, perplexity, cur_time, eval_time = 0
        cdef double count0
        cdef double *count = <double*>self.mem.alloc(self.C, sizeof(double))
        cdef size_t d, n, k
        timediff = []
        timeout = open(self.prefix + '.time', 'w')
        begin_time = datetime.datetime.now()
        for k in range(self.iter, self.n_iter):

            logger.info('Iteration %d...' % (k))
            start_time = datetime.datetime.now()
            if False:
                self.doSampling()
            else:
                sampler.sampling(k)


            if k > self.burn_in and k % 10 == 0:
                logger.info('Optimizing alpha...')
                for c in xrange(self.C):
                    self.update_alpha(c)

            timed = datetime.datetime.now() - start_time
            timediff.append(timed)
            timeout.write(str(timed.total_seconds()) + '\n')

            if self.eval_interval > 0 and self.test and k % self.eval_interval == 0:
                logger.info('Evaluating...')
                start_time = datetime.datetime.now()
                loglik = evaluate(20, 0,
                    self.Tc, self.T0, self.t_maxWC, self.C, self.t_D, self.t_dC, self.t_dW, self.t_docs,
                    self.alpha, self.unit,
                    self.nZ, self.n0Z, self.nYZ, self.n0WZ, self.n1CWZ, self.n1CZ, self.nCWZ, self.nCZ)
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
        timeout.close()

    cdef inline stat_incr(self, int d, int c, int w, int y, uint z, double val):
        gsl_2d_matrix_incr(self.nDZ, d, z, val)
        if z < self.T0:
            gsl_2d_matrix_incr(self.nYZ, y, z, val)
            gsl_1d_matrix_incr(self.nZ, z, val)
            if y == 0:
                gsl_2d_matrix_incr(self.n0WZ, w, z, val)
                gsl_1d_matrix_incr(self.n0Z, z, val)
            else:
                gsl_3d_matrix_incr(self.n1CWZ, c, w, z, val)
                gsl_2d_matrix_incr(self.n1CZ, c, z, val)
        else:
            gsl_3d_matrix_incr(self.nCWZ, c, w, z - self.T0, val)
            gsl_2d_matrix_incr(self.nCZ, c, z - self.T0, val)

    def initialize(self, load=False):
        logger.info("Initializing...")
        cdef size_t d
        cdef size_t n
        cdef list doc
        cdef size_t w
        cdef size_t y
        cdef size_t z
        cdef size_t c
        if load:
            logger.info("Loading states...")
            _Y, _Z, alpha, self.iter, _, _, _, _, _, _ = pickle.load(open(self.prefix + '.pkl', 'rb'))
            for c in range(self.C):
                for z in range(self.Tc[c]):
                    gsl_vector_set(self.alpha[c], z, alpha[c][z])
        else:
            self.iter = 0
        self.n_iter += self.iter

        self.dW = <uint*>self.mem.alloc(self.D, sizeof(uint)) # doc length
        self.dY = <np.uint8_t**>self.mem.alloc(self.D, sizeof(np.uint8_t*))
        self.dZ = <np.uint16_t**>self.mem.alloc(self.D, sizeof(np.uint16_t*))

        self.docs = <np.uint16_t**>self.mem.alloc(self.D, sizeof(np.uint16_t*))
        self.dC = <np.uint8_t*>self.mem.alloc(self.D, sizeof(np.uint8_t))

        self.maxWC = <size_t*>self.mem.alloc(self.C, sizeof(size_t))
        for c in range(self.C):
            self.maxWC[c] = 0
        for d, doc in enumerate(self.pydocs):
            N = len(doc)
            c = self.pydC[d]
            if N > self.maxWC[c]:
                self.maxWC[c] = N
            self.dW[d] = N
            self.dC[d] = c
            self.dY[d] = <np.uint8_t*>self.mem.alloc(N, sizeof(np.uint8_t))
            self.dZ[d] = <np.uint16_t*>self.mem.alloc(N, sizeof(np.uint16_t))
            self.docs[d] = <np.uint16_t*>self.mem.alloc(N, sizeof(np.uint16_t))

            if load:
                Y = _Y[d]
                Z = _Z[d]
            else:
                Y = np.random.randint(2, size=N)
                Z = np.random.randint(self.Tc[c], size=N)

            for n, val in enumerate(zip(doc, Y, Z)):
                w, y, z = val

                self.docs[d][n] = w
                self.dY[d][n] = y
                self.dZ[d][n] = z

                self.stat_incr(d, c, w, y, z, self.unit[c])

        self.histCW = <double**>self.mem.alloc(self.C, sizeof(double*))
        self.histCZW = <double***>self.mem.alloc(self.C, sizeof(double**))
        for c in range(self.C):
            self.histCW[c] = alloc_1d_matrix(self.mem, self.maxWC[c] + 1)
            self.histCZW[c] = alloc_2d_matrix(self.mem, self.Tc[c], self.maxWC[c] + 1)
        for d in range(self.D):
            c = self.dC[d]
            self.histCW[c][self.dW[d]] += self.unit[c]

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

    cdef update_alpha(self, size_t c):
        cdef size_t n_iter = 5
        cdef double shape = 1.001
        cdef double scale = 1.0
        cdef double unit = self.unit[c]

        cdef double *alpha = self.alpha[c].data
        cdef size_t i, j, k
        cdef size_t Tc = self.Tc[c]

        cdef double oldAlphaK
        cdef double denominator
        cdef double currentDigamma

        cdef double alphaSum = 0
        for i in range(Tc):
            alphaSum += alpha[i]

        cdef double **histZW = self.histCZW[c]
        cdef double *histW = self.histCW[c]
        cdef size_t maxW = self.maxWC[c]
        cdef uint *count = <uint*>self.mem.alloc(Tc, sizeof(uint))
        cdef uint nonZeroLimit
        cdef uint *nonZeroLimits = <uint*>self.mem.alloc(Tc, sizeof(uint))
        cdef double *histogram

        for i in range(Tc):
            nonZeroLimits[i] = 0

        count_hist(histZW, nonZeroLimits, c, unit, maxW, Tc, self.D, self.dC, self.dW, self.dZ)

        for k in range(n_iter):
            denominator = 0
            currentDigamma = 0

            for i in range(1, maxW + 1):
                currentDigamma += 1 / (alphaSum + i - 1)
                denominator += histW[i] * currentDigamma

            denominator -= 1/scale

            alphaSum = 0

            for k in range(Tc):
                nonZeroLimit = nonZeroLimits[k]

                oldAlphaK = alpha[k]
                alpha[k] = 0
                currentDigamma = 0

                histogram = histZW[k]

                for i in range(1, nonZeroLimit + 1):
                    currentDigamma += 1 / (oldAlphaK + i - 1)
                    alpha[k] += histogram[i] * currentDigamma

                alpha[k] = oldAlphaK * (alpha[k] + shape) / denominator
                alphaSum += alpha[k]
        logger.info("%d alpha:\t%s" % (c, ', '.join(['%.5f' % alpha[i] for i in range(5)])))

        if alphaSum < 0 or alphaSum / Tc > 1000:
            raise ValueError('alpha became unstable, terminating..')

    def evaluate(self, resample=False, single=True, T0=None, eval_sigma=False):

        cdef double scale = 0
        cdef size_t w, y, z, c, d, k
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
                    Y, Z, alpha, _, _, _, _, _, _, _ = pickle.load(open('%s-%04d.pkl' % (self.prefix, k), 'rb'))
                    for c in range(self.C):
                        for z in range(self.Tc[c]):
                            gsl_1d_matrix_incr(self.alpha[c], z, alpha[c][z])
                    for d in range(self.D):
                        c = self.dC[d]
                        for n in range(self.dW[d]):
                            w = self.docs[d][n]
                            y = Y[d][n]
                            z = Z[d][n]
                            self.stat_incr(d, c, w, y, z, self.unit[c])
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
        if eval_sigma:
            ys = []
            for i in range(self.T0):
                print gsl_vector_get(self.nYZ[0], i) - self.delta[0], gsl_vector_get(self.nZ, i) - self.deltaNorm
                if (gsl_vector_get(self.nZ, i) - self.deltaNorm) < 1e-5:
                    ys.append(0)
                else:
                    ys.append((gsl_vector_get(self.nYZ[0], i) - self.delta[0]) / (gsl_vector_get(self.nZ, i) - self.deltaNorm))
            ys = sorted(ys)
            print ys
            a = 0
            for i in range(self.T0 - T0, self.T0):
                a += ys[i]
            b = 0
            for i in range(0, self.T0 - T0):
                b += ys[i]
            if T0 != 0:
                a /= T0
            if T0 != self.T0:
                b /= (self.T0 - T0)
            return a, b, a - b

        cdef double loglik, perplexity
        cdef int f
        f = 1 if resample else 0

        loglik = evaluate(20, f,
            self.Tc, self.T0, self.t_maxWC, self.C, self.t_D, self.t_dC, self.t_dW, self.t_docs,
            self.alpha, self.unit,
            self.nZ, self.n0Z, self.nYZ, self.n0WZ, self.n1CWZ, self.n1CZ, self.nCWZ, self.nCZ)
        perplexity = exp(- loglik / self.t_W)
        logger.info('log likelihood: %2f, perplexity: %2f' % (loglik, perplexity))
        return perplexity

    def save(self, stats=False, k=None):
        cdef size_t d
        cdef size_t n
        Y = []
        Z = []
        Tc = []
        for i in range(self.C):
            Tc.append(self.Tc[i])

        for d in range(self.D):
            c = self.dC[d]
            dw = [self.docs[d][n] for n in range(self.dW[d])]
            dy = [self.dY[d][n] for n in range(self.dW[d])]
            dz = [self.dZ[d][n] for n in range(self.dW[d])]
            Y.append(dy)
            Z.append(dz)
        prefix = self.prefix
        n_iter = self.n_iter
        if k:
            prefix = prefix + ('-%04d' % (k))
            n_iter = k
        alpha = []
        for c in xrange(self.C):
            alpha.append(gsl2numpy(self.alpha[c]))
        pickle.dump((Y, Z, alpha, n_iter, self.T0, Tc, self.V, self.C, self.D, self.word2id),
                open(prefix + '.pkl', 'wb'))
