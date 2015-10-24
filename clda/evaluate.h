#ifndef EVALUATE_H
#define EVALUATE_H

typedef unsigned int uint;

double evaluate(size_t n_particle, int resampling,
        uint *Tc, uint T0, uint t_maxWC, uint C, uint t_D, uint8_t *t_dC, uint *t_dW, uint16_t **t_docs,
        gsl_vector **alpha, double *unit,
        gsl_vector *nZ, gsl_vector *n0Z, gsl_vector **nYZ, gsl_vector **n0WZ, gsl_vector ***n1CWZ, gsl_vector **n1CZ, gsl_vector ***nCWZ, gsl_vector **nCZ);

#endif
