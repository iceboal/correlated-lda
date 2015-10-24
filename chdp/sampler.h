#ifndef SAMPLER_H
#define SAMPLER_H
#include "alias.h"
#include <vector>
#include <map>

typedef uint32_t uint;

class Sampler {
public:
    size_t burn_in;
    size_t Tsize;
    uint32_t C;
    uint32_t D;
    uint32_t V;

    std::vector<size_t> topics;
    uint32_t *dW;
    uint8_t *dC;
    uint16_t **docs;
    uint8_t **dY;
    uint16_t **dZ;

    double *nZ;
    double *n0Z;
    double **nYZ;
    double **n0WZ;
    double ***n1CWZ;
    double **n1CZ;

    double *p;
    double *p0;

    uint *t0k;
    uint *n0k;
    uint **tck;
    uint **nck;
    uint N0_;
    uint *Nc_;
    uint *Nd_;
    double **U;
    double **X;
    size_t maxN;
    size_t maxK;
    std::map<size_t, uint> *tdk;
    std::map<size_t, uint> *ndk;
    double *b;

    double deltaNorm;
    double betaNorm;
    double *delta;
    double beta;

    size_t N;
    Alias **alias_word;
    Alias *alias_doc;
    int f;


    Sampler(size_t burn_in, size_t &Tsize, std::vector<size_t> &topics, uint32_t C, uint32_t D, uint32_t V,
        uint32_t *dW, uint8_t *dC, uint16_t **docs, uint8_t **dY, uint16_t **dZ,
        double *nZ, double *n0Z, double **nYZ,
        double **n0WZ, double ***n1CWZ, double **n1CZ,
        uint *t0k, uint *n0k, uint **tck, uint **nck, uint &N0_, uint *NC_, uint *ND_,
        double **U, double **X, size_t maxN, size_t maxK,
        std::map<size_t, uint> *tdk, std::map<size_t, uint> *ndk,
        double *b, double deltaNorm, double betaNorm, double *delta, double beta);
    ~Sampler();
    uint preAddTopic();
    int sampleU(size_t d, size_t z, size_t c);
    int sampleNewU(size_t d, size_t z, size_t c);
    int can_remove(int u, size_t z, size_t d, size_t c);
    void stat_incr(int u, size_t c, size_t z, size_t y, size_t d, size_t w, int val);
    double p_(size_t z, size_t w, size_t c, size_t d, int u);
    std::vector<size_t> sampling(size_t iter, size_t *Tsize);
    void reset();
};

#endif
