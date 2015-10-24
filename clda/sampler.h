#ifndef SAMPLER_H
#define SAMPLER_H
#include "alias.h"

class Sampler {
    size_t burn_in;
    uint32_t T0;
    uint32_t *Tc;
    uint32_t C;
    uint32_t D;
    uint32_t V;
    double *unit;

    uint32_t *dW;
    uint8_t *dC;
    uint16_t **docs;
    uint8_t **dY;
    uint16_t **dZ;
    gsl_vector **alpha;

    gsl_vector **nDZ;
    gsl_vector *nZ;
    gsl_vector *n0Z;
    gsl_vector **nYZ;
    gsl_vector **n0WZ;
    gsl_vector ***n1CWZ;
    gsl_vector **n1CZ;
    gsl_vector ***nCWZ;
    gsl_vector **nCZ;

    size_t N;
    size_t n_threads;
    Alias ***_alias_word;
    Alias *alias_doc;

public:

    Sampler(size_t burn_in, uint32_t T0, uint32_t *Tc, uint32_t C, uint32_t D, uint32_t V, double *unit,
        uint32_t *dW, uint8_t *dC, uint16_t **docs, uint8_t **dY, uint16_t **dZ, gsl_vector **alpha,
        gsl_vector **nDZ, gsl_vector *nZ, gsl_vector *n0Z, gsl_vector **nYZ,
        gsl_vector **n0WZ, gsl_vector ***n1CWZ, gsl_vector **n1CZ, gsl_vector ***nCWZ,
        gsl_vector **nCZ);
    ~Sampler();
    void sampling(size_t iter);
};

#endif
