#ifndef UTILS_H
#define UTILS_H

typedef uint32_t uint;

void count_z(double ***zc, double ***z0, uint T0, uint *Tc,
        uint C, uint D, uint V, uint *dW, uint8_t *dC, uint16_t **dZ, uint16_t **docs);

void count_hist(double **histZW, uint *nonZeroLimits, size_t c, double unit,
        size_t maxW, size_t Tc, size_t D, uint8_t *dC, uint *dW, uint16_t **dZ);

#endif
