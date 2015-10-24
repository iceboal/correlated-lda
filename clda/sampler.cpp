#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <stdint.h>
#include <omp.h>
#include <gsl_vector.h>
#include "alias.h"
#include "sampler.h"

#if 0
    #define dbg(fmt, ...) printf("DEBUG: " fmt, ## __VA_ARGS__)
#else
    #define dbg(fmt, ...)
#endif

static inline double p(size_t z, size_t oldZ, size_t oldY, double unit, uint32_t T0, gsl_vector *alpha, gsl_vector *nDZ, gsl_vector **nYZ, gsl_vector *nZ,
        gsl_vector *n0WZ, gsl_vector *n0Z, gsl_vector *n1CWZ, gsl_vector *n1CZ,
        gsl_vector *nCWZ, gsl_vector *nCZ) {
    double x = nDZ->data[z] + alpha->data[z];
    if (z != oldZ) {
        if (z < T0)
            x *= (nYZ[0]->data[z] * n0WZ->data[z] / n0Z->data[z] + nYZ[1]->data[z] * n1CWZ->data[z] / n1CZ->data[z])
                / nZ->data[z];
        else
            x *= nCWZ->data[z - T0] / nCZ->data[z - T0];
    } else {
        double unit0 = oldY == 0 ? unit : 0;
        double unit1 = oldY == 1 ? unit : 0;
        if (z < T0)
            x *= ((nYZ[0]->data[z] - unit0) * (n0WZ->data[z] -unit0) / (n0Z->data[z] - unit0) + \
                    (nYZ[1]->data[z] - unit1) * (n1CWZ->data[z] - unit1) / (n1CZ->data[z] - unit1))
                 / (nZ->data[z] - unit);
        else
            x *= (nCWZ->data[z - T0] - unit) / (nCZ->data[z - T0] - unit);
    }
    return x;
}

Sampler::Sampler(size_t burn_in, uint32_t T0, uint32_t *Tc, uint32_t C, uint32_t D, uint32_t V, double *unit,
        uint32_t *dW, uint8_t *dC, uint16_t **docs, uint8_t **dY, uint16_t **dZ, gsl_vector **alpha,
        gsl_vector **nDZ, gsl_vector *nZ, gsl_vector *n0Z, gsl_vector **nYZ,
        gsl_vector **n0WZ, gsl_vector ***n1CWZ, gsl_vector **n1CZ, gsl_vector ***nCWZ,
        gsl_vector **nCZ) : burn_in(burn_in), T0(T0), Tc(Tc), C(C), D(D), V(V), unit(unit),
    dW(dW), dC(dC), docs(docs), dY(dY), dZ(dZ), alpha(alpha),
    nDZ(nDZ), nZ(nZ), n0Z(n0Z), nYZ(nYZ),
    n0WZ(n0WZ), n1CWZ(n1CWZ), n1CZ(n1CZ), nCWZ(nCWZ),
    nCZ(nCZ) {

    n_threads = omp_get_max_threads();

    _alias_word = new Alias**[n_threads];
    alias_doc = new Alias[D];
    dbg("Initializing\n");
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        _alias_word[tid] = new Alias*[C];
        for (size_t i = 0; i < C; i++)
            _alias_word[tid][i] = new Alias[V];

        gsl_vector *p00 = gsl_vector_alloc(T0);
        gsl_vector *p01 = gsl_vector_alloc(T0);
        gsl_vector **pZC = (gsl_vector**)malloc(C * sizeof(gsl_vector*));
        for (size_t i = 0; i < C; i++)
            pZC[i] = gsl_vector_alloc(Tc[i]);
        #pragma omp barrier

        dbg("Allocate alias doc\n");
        #pragma omp for
        for (size_t d = 0; d < D; d++) {
            size_t c = dC[d];
            gsl_vector_memcpy(pZC[c], nDZ[d]);
            gsl_vector_add(pZC[c], alpha[c]);
            alias_doc[d].init(pZC[c]->data, Tc[c], Tc[c]);
        }

        dbg("Allocate alias word\n");
        #pragma omp for collapse(2)
        for (size_t c = 0; c < C; c++) {
            for (size_t w = 0; w < V; w++) {
                gsl_vector_set_all(pZC[c], 1);
                gsl_vector_view p0 = gsl_vector_subvector(pZC[c], 0, T0);

                gsl_vector_memcpy(p00, nYZ[0]);
                gsl_vector_mul(p00, n0WZ[w]);
                gsl_vector_div(p00, n0Z);

                gsl_vector_memcpy(p01, nYZ[1]);
                gsl_vector_mul(p01, n1CWZ[c][w]);
                gsl_vector_div(p01, n1CZ[c]);

                gsl_vector_add(p00, p01);
                gsl_vector_div(p00, nZ);

                gsl_vector_mul(&p0.vector, p00);

                if (Tc[c] - T0 > 0) {
                    gsl_vector_view pc = gsl_vector_subvector(pZC[c], T0, Tc[c] - T0);
                    gsl_vector_mul(&pc.vector, nCWZ[c][w]);
                    gsl_vector_div(&pc.vector, nCZ[c]);
                }

                _alias_word[0][c][w].init(pZC[c]->data, Tc[c], Tc[c]);
            }
        }

        if (tid != 0)
            for (size_t c = 0; c < C; c++)
                std::copy(_alias_word[0][c], _alias_word[0][c] + V, _alias_word[tid][c]);

        gsl_vector_free(p00);
        gsl_vector_free(p01);
        for (size_t i = 0; i < C; i++)
            gsl_vector_free(pZC[i]);
        free(pZC);
    }
}

void Sampler::sampling(size_t iter)
{
    if (iter < burn_in)
        N = 4;
    else
        N = 4;

    #pragma omp parallel
    {
        dbg("initialize\n");
        gsl_vector *p00, *p01;
        gsl_vector **pZC;

        int tid = omp_get_thread_num();
        Alias **alias_word = _alias_word[tid];

        p01 = gsl_vector_alloc(T0);
        pZC = (gsl_vector**)malloc(C * sizeof(gsl_vector*));
        for (size_t i = 0; i < C; i++)
            pZC[i] = gsl_vector_alloc(Tc[i]);

        //#pragma omp for
        //for (size_t d = 0; d < D; d++) {
        //    size_t c = dC[d];
        //    gsl_vector_memcpy(pZC[c], nDZ[d]);
        //    gsl_vector_add(pZC[c], alpha[c]);
        //    _alias_doc[0][d].generate_alias(pZC[c]->data);
        //}
        //if (tid != 0)
        //    std::copy(_alias_doc[0], _alias_doc[0] + D, alias_doc);

        #pragma omp for collapse(2)
        for (size_t c = 0; c < C; c++) {
            for (size_t w = 0; w < V; w++) {
                gsl_vector_view p0 = gsl_vector_subvector(pZC[c], 0, T0);
                p00 = &p0.vector;

                gsl_vector_memcpy(p00, nYZ[0]);
                gsl_vector_mul(p00, n0WZ[w]);
                gsl_vector_div(p00, n0Z);

                gsl_vector_memcpy(p01, nYZ[1]);
                gsl_vector_mul(p01, n1CWZ[c][w]);
                gsl_vector_div(p01, n1CZ[c]);

                gsl_vector_add(p00, p01);
                gsl_vector_div(p00, nZ);

                if (Tc[c] - T0 > 0) {
                    gsl_vector_view pc = gsl_vector_subvector(pZC[c], T0, Tc[c] - T0);
                    gsl_vector_memcpy(&pc.vector, nCWZ[c][w]);
                    gsl_vector_div(&pc.vector, nCZ[c]);
                }

                _alias_word[0][c][w].generate_alias(pZC[c]->data);
            }
        }

        if (tid != 0)
            for (size_t c = 0; c < C; c++)
                std::copy(_alias_word[0][c], _alias_word[0][c] + V, alias_word[c]);

        dbg("begin sampling\n");
        #pragma omp for
        for (size_t d = 0; d < D; d++) {
            size_t c = dC[d];
            for (size_t n = 0; n < dW[d]; n++) {
                size_t w = docs[d][n];
                size_t y = dY[d][n];
                size_t z = dZ[d][n];
                size_t oldY = y;
                size_t oldZ = z;
                double _unit = unit[c];

                /* decrement counts */
                nDZ[d]->data[z] -= _unit;

                /* sample z */
                size_t j;
                for (size_t k = 0; k < N; k++) {
                    double pi = 1;

                    if (k % 2 == 1) {
                        if (!alias_word[c][w].ttl) {
                            //for (size_t i = 0; i < T0; i++) {
                            //    pZC[c]->data[i] = (nYZ[0]->data[i] * n0WZ[w]->data[i] / n0Z->data[i] + \
                            //                      nYZ[1]->data[i] * n1CWZ[c][w]->data[i] / n1CZ[c]->data[i]) / \
                            //                      nZ->data[i];
                            //}
                            //for (size_t i = 0; i < Tc[c] - T0; i++) {
                            //    pZC[c]->data[i + T0] = nCWZ[c][w]->data[i] / nCZ[c]->data[i];
                            //}
                            gsl_vector_view p0 = gsl_vector_subvector(pZC[c], 0, T0);
                            p00 = &p0.vector;

                            gsl_vector_memcpy(p00, nYZ[0]);
                            gsl_vector_mul(p00, n0WZ[w]);
                            gsl_vector_div(p00, n0Z);

                            gsl_vector_memcpy(p01, nYZ[1]);
                            gsl_vector_mul(p01, n1CWZ[c][w]);
                            gsl_vector_div(p01, n1CZ[c]);

                            gsl_vector_add(p00, p01);
                            gsl_vector_div(p00, nZ);

                            if (Tc[c] - T0 > 0) {
                                gsl_vector_view pc = gsl_vector_subvector(pZC[c], T0, Tc[c] - T0);
                                gsl_vector_memcpy(&pc.vector, nCWZ[c][w]);
                                gsl_vector_div(&pc.vector, nCZ[c]);
                            }

                            alias_word[c][w].generate_alias(pZC[c]->data);
                        }
                        j = alias_word[c][w].sample();
                        pi *= alias_word[c][w].p[z];
                        pi /= alias_word[c][w].p[j];
                    }
                    else {
                        if (!alias_doc[d].ttl) {
                            gsl_vector_memcpy(pZC[c], nDZ[d]);
                            pZC[c]->data[oldZ] += _unit;
                            gsl_vector_add(pZC[c], alpha[c]);
                            alias_doc[d].generate_alias(pZC[c]->data);
                        }
                        j = alias_doc[d].sample();
                        pi *= alias_doc[d].p[z];
                        pi /= alias_doc[d].p[j];
                    }

                    pi *= p(j, oldZ, oldY, _unit, T0, alpha[c], nDZ[d], nYZ, nZ, n0WZ[w], n0Z, n1CWZ[c][w], n1CZ[c], nCWZ[c][w], nCZ[c]);
                    pi /= p(z, oldZ, oldY, _unit, T0, alpha[c], nDZ[d], nYZ, nZ, n0WZ[w], n0Z, n1CWZ[c][w], n1CZ[c], nCWZ[c][w], nCZ[c]);

                    if (pi >= 1 || drand48() < pi)
                        z = j;
                }

                /* sample y */
                if (z < T0) {
                    double y0, y1;
                    if (z != oldZ) {
                        y0 = nYZ[0]->data[z] * n0WZ[w]->data[z] / n0Z->data[z];
                        y1 = nYZ[1]->data[z] * n1CWZ[c][w]->data[z] / n1CZ[c]->data[z];
                    } else {
                        double unit0 = oldY == 0 ? _unit : 0;
                        double unit1 = oldY == 1 ? _unit : 0;
                        y0 = (nYZ[0]->data[z] - unit0) * (n0WZ[w]->data[z] - unit0) / (n0Z->data[z] - unit0);
                        y1 = (nYZ[1]->data[z] - unit1) * (n1CWZ[c][w]->data[z] - unit1) / (n1CZ[c]->data[z] - unit1);
                    }
                    y0 = y0 / (y0 + y1);
                    if (y0 > drand48())
                        y = 0;
                    else
                        y = 1;
                }

                /* increment counts */
                nDZ[d]->data[z] += _unit;
                if (oldZ != z) {
                    if (oldZ < T0) {
                        #pragma omp atomic
                        nYZ[oldY]->data[oldZ] -= _unit;
                        #pragma omp atomic
                        nZ->data[oldZ] -= _unit;
                        if (oldY == 0) {
                            #pragma omp atomic
                            n0WZ[w]->data[oldZ] -= _unit;
                            #pragma omp atomic
                            n0Z->data[oldZ] -= _unit;
                        } else {
                            #pragma omp atomic
                            n1CWZ[c][w]->data[oldZ] -= _unit;
                            #pragma omp atomic
                            n1CZ[c]->data[oldZ] -= _unit;
                        }
                    } else {
                        #pragma omp atomic
                        nCWZ[c][w]->data[oldZ - T0] -= _unit;
                        #pragma omp atomic
                        nCZ[c]->data[oldZ - T0] -= _unit;
                    }

                    if (z < T0) {
                        #pragma omp atomic
                        nYZ[y]->data[z] += _unit;
                        #pragma omp atomic
                        nZ->data[z] += _unit;
                        if (y == 0) {
                            #pragma omp atomic
                            n0WZ[w]->data[z] += _unit;
                            #pragma omp atomic
                            n0Z->data[z] += _unit;
                        } else {
                            #pragma omp atomic
                            n1CWZ[c][w]->data[z] += _unit;
                            #pragma omp atomic
                            n1CZ[c]->data[z] += _unit;
                        }
                    } else {
                        #pragma omp atomic
                        nCWZ[c][w]->data[z - T0] += _unit;
                        #pragma omp atomic
                        nCZ[c]->data[z - T0] += _unit;
                    }
                } else if (oldY != y) { /* y shouldn't change if z >= T0 */
                    #pragma omp atomic
                    nYZ[y]->data[z] += _unit;
                    #pragma omp atomic
                    nYZ[oldY]->data[z] -= _unit;
                    if (y == 0) {
                        #pragma omp atomic
                        n0WZ[w]->data[z] += _unit;
                        #pragma omp atomic
                        n0Z->data[z] += _unit;
                        #pragma omp atomic
                        n1CWZ[c][w]->data[z] -= _unit;
                        #pragma omp atomic
                        n1CZ[c]->data[z] -= _unit;
                    } else {
                        #pragma omp atomic
                        n0WZ[w]->data[z] -= _unit;
                        #pragma omp atomic
                        n0Z->data[z] -= _unit;
                        #pragma omp atomic
                        n1CWZ[c][w]->data[z] += _unit;
                        #pragma omp atomic
                        n1CZ[c]->data[z] += _unit;
                    }
                }

                /* update values */
                dY[d][n] = y;
                dZ[d][n] = z;
            }
        }

        dbg("free\n");
        gsl_vector_free(p01);
        for (size_t i = 0; i < C; i++)
            gsl_vector_free(pZC[i]);
        free(pZC);

        dbg("end\n");
    }
}

Sampler::~Sampler() {
    for (size_t n = 0; n < n_threads; n++) {
        for (size_t i = 0; i < C; i++)
            delete []_alias_word[n][i];
        delete []_alias_word[n];
    }
    delete []alias_doc;
    delete []_alias_word;
}
