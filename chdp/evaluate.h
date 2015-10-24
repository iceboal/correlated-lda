#ifndef EVALUATE_H
#define EVALUATE_H

typedef unsigned int uint;

double evaluate(size_t n_particle, int resampling,
        double **U, double **X, double *b, size_t T, size_t *topics, uint *n0k, uint **tck, uint **nck, uint N0_, uint *Nc_,
        size_t Tsize, uint t_maxWC, uint C, uint t_D, size_t V, uint8_t *t_dC, uint *t_dW, uint16_t **t_docs,
        double *nZ, double *n0Z, double **nYZ, double **n0WZ, double ***n1CWZ, double **n1CZ);

#endif
