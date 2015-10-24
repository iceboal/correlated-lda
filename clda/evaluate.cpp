#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <stdint.h>
#include <gsl_vector.h>
#include <gsl_rng.h>
#include <gsl_randist.h>
#include "evaluate.h"

#if 0
    #define dbg(fmt, ...) printf("DEBUG: " fmt, ## __VA_ARGS__)
#else
    #define dbg(fmt, ...)
#endif

typedef unsigned int uint;

/* add value to prob */
static void left_to_right(double *prob, int resampling,
        size_t W, uint16_t *dZ, uint16_t *doc, uint Tc, uint T0, gsl_vector *alpha, double alphaSum, double unit,
        gsl_vector *pZ, gsl_vector *p00, gsl_vector *p01, gsl_rng *rng,
        gsl_vector *nDZ, gsl_vector *nZ, gsl_vector *n0Z, gsl_vector **nYZ, gsl_vector **n0WZ, gsl_vector **n1CWZ, gsl_vector *n1CZ, gsl_vector **nCWZ, gsl_vector *nCZ) {
    double *p;
    gsl_ran_discrete_t *disc;
    size_t n, i;
    uint z, w;
    double tokens = 0;

    gsl_vector_memcpy(nDZ, alpha);

    for (n = 0; n < W; n++) {
        if (resampling) {
            for (i = 0; i < n; i++) {
                z = dZ[i];
                w = doc[i];

                /* decrement counts */
                nDZ->data[z] -= unit;

                gsl_vector_memcpy(pZ, nDZ);
                gsl_vector_view p0 = gsl_vector_subvector(pZ, 0, T0);

                gsl_vector_memcpy(p00, nYZ[0]);
                gsl_vector_mul(p00, n0WZ[w]);
                gsl_vector_div(p00, n0Z);

                gsl_vector_memcpy(p01, nYZ[1]);
                gsl_vector_mul(p01, n1CWZ[w]);
                gsl_vector_div(p01, n1CZ);

                gsl_vector_add(p00, p01);

                gsl_vector_mul(&p0.vector, p00);

                if (Tc - T0 > 0) {
                    gsl_vector_div(&p0.vector, nZ);
                    gsl_vector_view pc = gsl_vector_subvector(pZ, T0, Tc - T0);
                    gsl_vector_mul(&pc.vector, nCWZ[w]);
                    gsl_vector_div(&pc.vector, nCZ);
                }

                disc = gsl_ran_discrete_preproc(Tc, pZ->data);
                z = gsl_ran_discrete(rng, disc);
                gsl_ran_discrete_free(disc);

                /* increment counts */
                nDZ->data[z] += unit;
                dZ[i] = z;
            }
        }

        w = doc[n];
        /* sample z use the true marginal probability */
        gsl_vector_memcpy(pZ, nDZ);
        gsl_vector_scale(pZ, 1.0 / (alphaSum + tokens));
        gsl_vector_view p0 = gsl_vector_subvector(pZ, 0, T0);

        gsl_vector_memcpy(p00, nYZ[0]);
        gsl_vector_mul(p00, n0WZ[w]);
        gsl_vector_div(p00, n0Z);

        gsl_vector_memcpy(p01, nYZ[1]);
        gsl_vector_mul(p01, n1CWZ[w]);
        gsl_vector_div(p01, n1CZ);

        gsl_vector_add(p00, p01);
        gsl_vector_div(p00, nZ);

        gsl_vector_mul(&p0.vector, p00);

        if (Tc - T0 > 0) {
            gsl_vector_view pc = gsl_vector_subvector(pZ, T0, Tc - T0);
            gsl_vector_mul(&pc.vector, nCWZ[w]);
            gsl_vector_div(&pc.vector, nCZ);
        }

        p = prob + n;
        for (i = 0; i < Tc; i++) {
            *p += pZ->data[i];
        }

        disc = gsl_ran_discrete_preproc(Tc, pZ->data);
        z = gsl_ran_discrete(rng, disc);
        gsl_ran_discrete_free(disc);

        tokens += unit;
        nDZ->data[z] += unit;
        dZ[n] = z;
    }
}

double evaluate(size_t n_particle, int resampling,
        uint *Tc, uint T0, uint t_maxWC, uint C, uint t_D, uint8_t *t_dC, uint *t_dW, uint16_t **t_docs,
        gsl_vector **alpha, double *unit,
        gsl_vector *nZ, gsl_vector *n0Z, gsl_vector **nYZ, gsl_vector **n0WZ, gsl_vector ***n1CWZ, gsl_vector **n1CZ, gsl_vector ***nCWZ, gsl_vector **nCZ) {
    size_t d, i, c;
    double log_n_particle = log(n_particle);
    double loglik = 0;
    double *alphaSum = (double*)malloc(sizeof(double) * C);
    for (c = 0; c < C; c++) {
        alphaSum[c] = 0;
        for (i = 0; i < Tc[c]; i++)
            alphaSum[c] += gsl_vector_get(alpha[c], i);
    }

    #pragma omp parallel private(i, c)
    {
        double doc_loglik;
        double *prob = (double*)malloc(sizeof(double) * t_maxWC);
        uint16_t *dZ = (uint16_t*)malloc(sizeof(uint16_t) * t_maxWC);
        gsl_vector **pZ = (gsl_vector**)malloc(sizeof(gsl_vector*) * C);
        gsl_vector **nDZ = (gsl_vector**)malloc(sizeof(gsl_vector*) * C);
        for (c = 0; c < C; c++) {
            pZ[c] = gsl_vector_alloc(Tc[c]);
            nDZ[c] = gsl_vector_alloc(Tc[c]);
        }
        gsl_vector *p00, *p01;
        p00 = gsl_vector_alloc(T0);
        p01 = gsl_vector_alloc(T0);
        gsl_rng_env_setup();
        gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
        gsl_rng_set(rng, 0);

        #pragma omp for
        for (d = 0; d < t_D; d++) {
            c = t_dC[d];
            doc_loglik = 0;
            for (i = 0; i < t_dW[d]; i++)
                prob[i] = 0;

            for (i = 0; i < n_particle; i++)
                left_to_right(prob, resampling,
                        t_dW[d], dZ, t_docs[d], Tc[c], T0, alpha[c], alphaSum[c], unit[c],
                        pZ[c], p00, p01, rng,
                        nDZ[c], nZ, n0Z, nYZ, n0WZ, n1CWZ[c], n1CZ[c], nCWZ[c], nCZ[c]);

            for (i = 0; i < t_dW[d]; i++)
                doc_loglik += log(prob[i]);
            doc_loglik -= log_n_particle * t_dW[d];
            #pragma omp atomic
            loglik += doc_loglik;
        }
        free(prob);
        free(dZ);
        for (c = 0; c < C; c++) {
            gsl_vector_free(pZ[c]);
            gsl_vector_free(nDZ[c]);
        }
        free(pZ);
        free(nDZ);
        gsl_vector_free(p00);
        gsl_vector_free(p01);
    }
    free(alphaSum);
    return loglik;
}

