#include <cstdlib>
#include <stdint.h>
#include "utils.h"

static inline void normalize(double *A, size_t n) {
    double sum = 0;
    for(size_t i = 0; i < n; i++)
        sum += A[i];
    for(size_t i = 0; i < n; i++)
        A[i] /= sum;
}

void count_z(double ***zc, double ***z0, uint T0, uint *Tc,
        uint C, uint D, uint V, uint *dW, uint8_t *dC, uint16_t **dZ, uint16_t **docs) {
    for (size_t c = 0; c < C; c++) {
        for (size_t z = 0; z < T0; z++)
            for (size_t w = 0; w < V; w++)
                z0[c][z][w] = 0;
        for (size_t z = 0; z < Tc[c] - T0; z++)
            for (size_t w = 0; w < V; w++)
                zc[c][z][w] = 0;
    }
    #pragma omp parallel
    {
        #pragma omp for
        for (size_t d = 0; d < D; d++) {
            for (size_t n = 0; n < dW[d]; n++) {
                size_t c = dC[d];
                size_t w = docs[d][n];
                size_t z = dZ[d][n];
                if (z < T0)
                    #pragma omp atomic
                    z0[c][z][w]++;
                else
                    #pragma omp atomic
                    zc[c][z - T0][w]++;
            }
        }

        #pragma omp for collapse(2)
        for (size_t c = 0; c < C; c++)
            for (size_t z = 0; z < T0; z++)
                normalize(z0[c][z], V);
        #pragma omp for collapse(2)
        for (size_t c = 0; c < C; c++)
            for (size_t z = 0; z < Tc[c] - T0; z++)
                normalize(z0[c][z], V);
    }
}

void count_hist(double **histZW, uint *nonZeroLimits, size_t c, double unit,
        size_t maxW, size_t Tc, size_t D, uint8_t *dC, uint *dW, uint16_t **dZ) {
    #pragma omp parallel
    {
        uint *count = new uint[Tc];
        #pragma omp for collapse(2)
        for (size_t i = 0; i < Tc; i++)
            for (size_t j = 0; j < maxW + 1; j++)
                histZW[i][j] = 0;

        #pragma omp for
        for (size_t i = 0; i < D; i++) {
            if (dC[i] != c)
                continue;
            for (size_t j = 0; j < Tc; j++)
                count[j] = 0;
            for (size_t j = 0; j < dW[i]; j++)
                count[dZ[i][j]] += 1;
            for (size_t j = 0; j < Tc; j++)
                #pragma omp atomic
                histZW[j][count[j]] += unit;
        }

        #pragma omp for
        for (size_t i = 0; i < Tc; i++)
            for (size_t j = 0; j < maxW + 1; j++)
                if (histZW[i][j] > 0)
                    nonZeroLimits[i] = j;
    }
}
