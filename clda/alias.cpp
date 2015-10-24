#include <cstdlib>
#include <algorithm>
#include "alias.h"

#ifdef NDEBUG
    #define NDEBUG_DISABLED
    #undef NDEBUG
#endif
#include <cassert>
#ifdef NDEBUG_DISABLED
    #define NDEBUG        // re-enable NDEBUG if it was originally enabled
#endif

void Alias::init(double *p, uint16_t n, uint16_t ttl) {
    this->n = n;
    _ttl = ttl;
    this->ttl = 0;
    this->p = (double*)malloc(n*sizeof(double));
    table = (uint16_t*)malloc(n*sizeof(uint16_t));
    std::copy(p, p + n, this->p);
    generate_alias(p);
}

Alias::~Alias() {
    free(p);
    free(table);
}

Alias& Alias::operator= (const Alias &a) {
    assert(n == 0 || n == a.n);
    if (n == 0) {
        n = a.n;
        table = (uint16_t*)malloc(n*sizeof(uint16_t));
        p = (double*)malloc(n*sizeof(double));
    }
    _ttl = a._ttl;
    ttl = a.ttl;
    sum = a.sum;
    std::copy(a.table, a.table + n, table);
    std::copy(a.p, a.p + n, p);
    return *this;
}

/* p will be modified */
void Alias::generate_alias(double *p) {
    double *prob = (double*)malloc(n*sizeof(double));
    uint16_t *alias = (uint16_t*)malloc(n*sizeof(uint16_t));
    uint16_t *S = (uint16_t*)malloc(n*sizeof(uint16_t));
    uint16_t *L = (uint16_t*)malloc(n*sizeof(uint16_t));

    sum = 0;
    uint16_t i;
    std::copy(p, p + n, this->p);

    for (i = 0; i < n; i++)
        sum += p[i];
    for (i = 0; i < n; i++)
        p[i] = p[i] * n / sum;

    uint16_t nS = 0, nL = 0;
    for (i = 0; i < n; i++) {
        if (p[i] < 1)
            S[nS++] = i;
        else
            L[nL++] = i;
    }

    uint16_t a, g;
    while (nS && nL) {
        a = S[--nS];
        g = L[--nL];
        prob[a] = p[a];
        alias[a] = g;
        p[g] = p[g] + p[a] -1;
        if (p[g] < 1)
            S[nS++] = g;
        else
            L[nL++] = g;
    }

    while (nL)
        prob[L[--nL]] = 1;
    while (nS)
        prob[S[--nS]] = 1;
    ttl = _ttl;

    for (i = 0; i < n; i++) {
        uint16_t j = (uint16_t) (n * drand48());
        table[i] =  drand48() < prob[j] ? j : alias[j];
    }

    free(S);
    free(L);
    free(prob);
    free(alias);
}

uint16_t Alias::sample() {
    assert(ttl);
    return table[--ttl];
}
