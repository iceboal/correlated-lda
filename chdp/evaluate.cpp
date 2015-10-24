#include <cstdlib>
#include <cstdio>
#include <map>
#include <set>
#include <math.h>
#include <stdint.h>
#include <gsl_vector.h>
#include <gsl_rng.h>
#include <gsl_randist.h>
#include "evaluate.h"
#include <omp.h>

#if 0
    #define dbg(fmt, ...) printf("DEBUG: " fmt, ## __VA_ARGS__)
#else
    #define dbg(fmt, ...)
#endif

typedef unsigned int uint;

/* add value to prob */
static void left_to_right(double *prob, int resampling,
        double **U, double **X, double *b, double bc, size_t T, size_t *topics, uint *n0k, uint *tck, uint *nck, uint N0_, uint N1_,
        size_t V, size_t W, uint16_t *dZ, uint16_t *doc,
        double *p0, double *p1, double *p2, double *p3, double *pZ,
        double *nZ, double *n0Z, double **nYZ, double **n0WZ, double **n1CWZ, double *n1CZ) {
    std::set<size_t> tdk;
    std::map<size_t, uint> ndk;

    for (size_t n = 0; n < W; n++) {
        if (resampling) {
            for (size_t j = 0; j < n; j++) {
                uint z = dZ[j];
                uint w = doc[j];

                /* decrement counts */
                ndk[z]--;
                if (ndk[z] == 0) {
                    ndk.erase(z);
                    tdk.erase(z);
                }

                double p0sum = 0, p1sum = 0, p2sum = 0, p3sum = 0, psum = 0;
                for (size_t i = 0; i < T; i++) {
                    size_t k = topics[i];
                    pZ[k] = (nYZ[0][k] * n0WZ[w][k] / n0Z[k] + \
                        nYZ[1][k] * n1CWZ[w][k] / n1CZ[k]) / \
                        nZ[k];
                }

                for (std::set<size_t>::iterator it=tdk.begin(); it!=tdk.end(); it++) {
                    size_t k = *it;
                    uint ndk_ = ndk[k];
                    p1[k] = U[ndk_][1] * ndk_/(ndk_+1) * (bc+N1_)/b[2] * \
                        pZ[k];
                    p1sum += p1[k];
                    p2[k] = U[nck[k]][tck[k]] * X[ndk_][1] * 2/(nck[k]+1) * (nck[k]+1-tck[k])/(ndk_+1) * \
                        pZ[k];
                    p2sum += p2[k];
                }
                psum += p1sum + p2sum;

                for (size_t i = 0; i < T; i++) {
                    size_t k = topics[i];
                    if (tdk.count(k) != 0) {
                        uint ndk_ = ndk[k];
                        // new collect table
                        p0[k] = bc / (b[0]+N0_) * (n0k[k]^2) * X[nck[k]][tck[k]] * \
                            X[ndk_][1] * (tck[k]+1)/(n0k[k]+1) * 2/(nck[k]+1) / (ndk_+1) * \
                            pZ[k];
                    } else if (nck[k] != 0) {
                        // new doc topic
                        p0[k] = U[nck[k]][tck[k]] * (nck[k]+1-tck[k]) / (nck[k]+1) * \
                            pZ[k];
                        // new collect table
                        p3[k] = bc/(b[0]+N0_) * (n0k[k]^2) * \
                                X[nck[k]][tck[k]] * (tck[k]+1)/(n0k[k]+1)/(nck[k]+1) * \
                                pZ[k];
                        p3sum += p3[k];
                    } else {
                        // new collect topic
                        p0[k] = bc / (b[0]+N0_) * (n0k[k]^2) / (n0k[k]+1) * \
                            pZ[k];
                    }
                    p0sum += p0[k];
                }
                psum += p0sum + p3sum;

                psum *= drand48();
                if (psum < p1sum) { // new doc dish
                    std::set<size_t>::iterator it = tdk.begin();
                    z = *it;
                    while (psum >= p1[z]) {
                        psum -= p1[z];
                        it++;
                        z = *it;
                    }
                } else {
                    psum -= p1sum;
                    if (psum < p2sum) {// new doc table
                        std::set<size_t>::iterator it = tdk.begin();
                        z = *it;
                        while (psum >= p2[z]) {
                            psum -= p2[z];
                            it++;
                            z = *it;
                        }
                    } else {
                        psum -= p2sum;
                        if (psum < p0sum) {
                            size_t k = 0;
                            z = topics[k];
                            while (psum >= p0[z]) {
                                psum -= p0[z];
                                k += 1;
                                z = topics[k];
                            }
                        } else {
                            psum -= p0sum;
                            for (size_t i = 0; i < T; i++) {
                                size_t z = topics[i];
                                if (tdk.count(z) != 0 || nck[z] == 0)
                                    continue;
                                if (psum < p3[z])
                                    break;
                                psum -= p3[z];
                            }
                        }
                    }
                }

                /* increment counts */
                dZ[j] = z;
                if (tdk.count(z) == 0) {
                    tdk.insert(z);
                    ndk[z] = 1;
                } else {
                    ndk[z]++;
                }
            }
        }

        uint z, w = doc[n];
        /* sample z use the true marginal probability */

        double p0sum = 0, p1sum = 0, p2sum = 0, p3sum = 0, psum = 0;
        for (size_t i = 0; i < T; i++) {
            size_t k = topics[i];
            pZ[k] = (nYZ[0][k] * n0WZ[w][k] / n0Z[k] + \
                nYZ[1][k] * n1CWZ[w][k] / n1CZ[k]) / \
                nZ[k];
        }

        double topicSum = 0;
        for (std::set<size_t>::iterator it=tdk.begin(); it!=tdk.end(); it++) {
            size_t k = *it;
            uint ndk_ = ndk[k];
            p1[k] = U[ndk_][1] * ndk_/(ndk_+1) * (bc+N1_)/b[2];
            topicSum += p1[k];
            p1[k] *= pZ[k];
            p1sum += p1[k];
            p2[k] = U[nck[k]][tck[k]] * X[ndk_][1] * 2/(nck[k]+1) * (nck[k]+1-tck[k])/(ndk_+1);
            topicSum += p2[k];
            p2[k] *= pZ[k];
            p2sum += p2[k];
        }
        psum += p1sum + p2sum;

        for (size_t i = 0; i < T; i++) {
            size_t k = topics[i];
            if (tdk.count(k) != 0) {
                uint ndk_ = ndk[k];
                // new collect table
                p0[k] = bc / (b[0]+N0_) * (n0k[k]^2) * X[nck[k]][tck[k]] * \
                    X[ndk_][1] * (tck[k]+1)/(n0k[k]+1) * 2/(nck[k]+1) / (ndk_+1);
                topicSum += p0[k];
                p0[k] *= pZ[k];
            } else if (nck[k] != 0) {
                // new doc topic
                p0[k] = U[nck[k]][tck[k]] * (nck[k]+1-tck[k]) / (nck[k]+1);
                topicSum += p0[k];
                p0[k] *= pZ[k];
                // new collect table
                p3[k] = bc/(b[0]+N0_) * (n0k[k]^2) * \
                        X[nck[k]][tck[k]] * (tck[k]+1)/(n0k[k]+1)/(nck[k]+1);
                topicSum += p3[k];
                p3[k] *= pZ[k];
                p3sum += p3[k];
            } else {
                // new collect topic
                p0[k] = bc / (b[0]+N0_) * (n0k[k]^2) / (n0k[k]+1);
                topicSum += p0[k];
                p0[k] *= pZ[k];
            }
            p0sum += p0[k];
        }
        psum += p0sum + p3sum;

        double pold = psum;

        psum *= drand48();
        if (psum < p1sum) { // new doc dish
            std::set<size_t>::iterator it = tdk.begin();
            z = *it;
            while (psum >= p1[z]) {
                psum -= p1[z];
                it++;
                z = *it;
            }
        } else {
            psum -= p1sum;
            if (psum < p2sum) {// new doc table
                std::set<size_t>::iterator it = tdk.begin();
                z = *it;
                while (psum >= p2[z]) {
                    psum -= p2[z];
                    it++;
                    z = *it;
                }
            } else {
                psum -= p2sum;
                if (psum < p0sum) {
                    size_t k = 0;
                    z = topics[k];
                    while (psum >= p0[z]) {
                        psum -= p0[z];
                        k += 1;
                        z = topics[k];
                    }
                } else {
                    psum -= p0sum;
                    for (size_t i = 0; i < T; i++) {
                        z = topics[i];
                        if (tdk.count(z) != 0 || nck[z] == 0)
                            continue;
                        if (psum < p3[z])
                            break;
                        psum -= p3[z];
                    }
                }
            }
        }

        prob[n] += pold / topicSum;

        if (tdk.count(z) == 0) {
            tdk.insert(z);
            ndk[z] = 1;
        } else {
            ndk[z]++;
        }
        dZ[n] = z;
    }
}


double evaluate(size_t n_particle, int resampling,
        double **U, double **X, double *b, size_t T, size_t *topics, uint *n0k, uint **tck, uint **nck, uint N0_, uint *Nc_,
        size_t Tsize, uint t_maxWC, uint C, uint t_D, size_t V, uint8_t *t_dC, uint *t_dW, uint16_t **t_docs,
        double *nZ, double *n0Z, double **nYZ, double **n0WZ, double ***n1CWZ, double **n1CZ) {
    size_t d, i, c;
    double log_n_particle = log(n_particle);
    double loglik = 0;

    #pragma omp parallel private(i, c)
    {
        double doc_loglik;
        double *prob = (double*)malloc(sizeof(double) * t_maxWC);
        double *p0 = (double*)malloc(sizeof(double) * Tsize);
        double *p1 = (double*)malloc(sizeof(double) * Tsize);
        double *p2 = (double*)malloc(sizeof(double) * Tsize);
        double *p3 = (double*)malloc(sizeof(double) * Tsize);
        double *pZ = (double*)malloc(sizeof(double) * Tsize);
        uint16_t *dZ = (uint16_t*)malloc(sizeof(uint16_t) * t_maxWC);

        #pragma omp for
        for (d = 0; d < t_D; d++) {
            c = t_dC[d];
            doc_loglik = 0;
            for (i = 0; i < t_dW[d]; i++)
                prob[i] = 0;

            for (i = 0; i < n_particle; i++)
                left_to_right(prob, resampling,
                        U, X, b, b[1], T, topics, n0k, tck[c], nck[c], N0_, Nc_[c],
                        V, t_dW[d], dZ, t_docs[d],
                        p0, p1, p2, p3, pZ,
                        nZ, n0Z, nYZ, n0WZ, n1CWZ[c], n1CZ[c]);

            for (i = 0; i < t_dW[d]; i++) {
                doc_loglik += log(prob[i]);
            }
            doc_loglik -= log_n_particle * t_dW[d];
            #pragma omp atomic
            loglik += doc_loglik;
        }
        free(prob);
        free(p0);
        free(p1);
        free(p2);
        free(p3);
        free(pZ);
        free(dZ);
    }
    return loglik;
}

