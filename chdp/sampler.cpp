#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <map>
#include <utility>

#ifdef NDEBUG
    #define NDEBUG_DISABLED
    #undef NDEBUG
#endif
#include <cassert>
#ifdef NDEBUG_DISABLED
    #define NDEBUG        // re-enable NDEBUG if it was originally enabled
#endif

#include <stdint.h>
#include "alias.h"
#include "sampler.h"

#if 0
#define dbg(fmt, ...) printf("DEBUG: " fmt, ## __VA_ARGS__)
#else
#define dbg(fmt, ...)
#endif

double Sampler::p_(size_t z, size_t w, size_t c, size_t d, int u) {
    double x = 1;

    if (u == 0) {
        return b[0] / (b[0] + N0_) * b[1] / V;
    }

    x *= (nYZ[0][z] * n0WZ[w][z] / n0Z[z] + nYZ[1][z] * n1CWZ[c][w][z] / n1CZ[c][z])
        / nZ[z];

    uint *tck_ = tck[c];
    uint *nck_ = nck[c];
    assert(nck_[z] < maxN);
    assert(tck_[z] < maxK);
    double y = 1.0;
    // new dish or table

    if (tdk[d].find(z) != tdk[d].end()) {
        uint tdk_ = tdk[d][z];
        uint ndk_ = ndk[d][z];
        assert(ndk_ < maxN);
        assert(tdk_ < maxK);
        switch(u) {
            case -1: // new dish
                y = U[ndk_][tdk_] * (ndk_+1-tdk_)/(ndk_+1) * (b[1] + Nc_[c]) / b[2];
                break;
            case 2: // new doc table
                y = U[nck_[z]][tck_[z]] * X[ndk_][tdk_] * (tdk_+1)/(nck_[z]+1) * (nck_[z]+1-tck_[z])/(ndk_+1);
                break;
            case 1: // new collect table
                y = b[1] / (b[0]+N0_) * (n0k[z] * n0k[z]) * X[nck_[z]][tck_[z]] * X[ndk_][tdk_] * (tck_[z]+1)/(n0k[z]+1) * (tdk_+1)/(nck_[z]+1) / (ndk_+1);
                break;
            default:
                throw std::runtime_error("wrong u");
        }
    } else if (nck[c][z] > 0) { // new collect table or new doc topic
        switch(u) {
            case 2: // new doc topic
                y = U[nck_[z]][tck_[z]] * (nck_[z]+1-tck_[z]) / (nck_[z]+1);
                break;
            case 1:// new collect table
                y = b[1] / (b[0]+N0_) * (n0k[z] * n0k[z]) * X[nck_[z]][tck_[z]] * (tck_[z]+1)/(n0k[z]+1)/(nck_[z]+1);
                break;
            default:
                throw std::runtime_error("wrong u");
        }
    } else {
        assert(std::find(topics.begin(), topics.end(), z) != topics.end());
        // new collect topic
        y = b[1] / (b[0]+N0_) * (n0k[z] * n0k[z]) / (n0k[z]+1);
        assert(u == 1);
    }
    return x * y;
}

uint Sampler::preAddTopic() {
    uint k;
    for(k = 0; k < topics.size()+1; k++)
        if (std::find(topics.begin(), topics.end(), k) == topics.end())
            break;

    if (k >= Tsize) {
        Tsize *= 2;
        p = (double*)realloc(p, 2 * Tsize * sizeof(double));
        p0 = (double*)realloc(p0, 2 * Tsize * sizeof(double));

        t0k = (uint*)realloc(t0k, Tsize * sizeof(uint));
        n0k = (uint*)realloc(n0k, Tsize * sizeof(uint));
        for (uint i = 0; i < C; i++) {
            tck[i] = (uint*)realloc(tck[i], Tsize * sizeof(uint));
            nck[i] = (uint*)realloc(nck[i], Tsize * sizeof(uint));
            n1CZ[i] = (double*)realloc(n1CZ[i], Tsize * sizeof(double));
            for (uint j = 0; j < V; j++)
                n1CWZ[i][j] = (double*)realloc(n1CWZ[i][j], Tsize * sizeof(double));
        }
        for (uint j = 0; j < V; j++)
            n0WZ[j] = (double*)realloc(n0WZ[j], Tsize * sizeof(double));

        n0Z = (double*)realloc(n0Z, Tsize * sizeof(double));
        nZ = (double*)realloc(nZ, Tsize * sizeof(double));
        nYZ[0] = (double*)realloc(nYZ[0], Tsize * sizeof(double));
        nYZ[1] = (double*)realloc(nYZ[1], Tsize * sizeof(double));
    }

    t0k[k] = 0;
    n0k[k] = 0;
    for (uint i = 0; i < C; i++) {
        tck[i][k] = 0;
        nck[i][k] = 0;
        n1CZ[i][k] = betaNorm;
        for (uint j = 0; j < V; j++)
            n1CWZ[i][j][k] = beta;
    }
    for (uint j = 0; j < V; j++)
        n0WZ[j][k] = beta;
    n0Z[k] = betaNorm;
    nZ[k] = deltaNorm;
    nYZ[0][k] = delta[0];
    nYZ[1][k] = delta[1];

    return k;
}

int Sampler::sampleU(size_t d, size_t z, size_t c) {
    int u = -1;
    double pu = 1.0 * tdk[d][z] / ndk[d][z];
    if (drand48() < pu) {
        u = 2;
        pu = 1.0 * tck[c][z] / nck[c][z];
        if (drand48() < pu) {
            u = 1;
            pu = 1.0 / n0k[z];
            if (drand48() < pu)
                u = 0;
        }
    }
    return u;
}

// TODO: does this make sense?
int Sampler::sampleNewU(size_t d, size_t z, size_t c) {
    if (ndk[d].count(z) > 0) {
        int u = sampleU(d, z, c);
        return u == 0? 1 : u;
    }

    if (nck[c][z] > 0) {
        int u = 2;
        double pu = 1.0 * tck[c][z] / nck[c][z];
        if (drand48() < pu) {
            u = 1;
        }
        return u;
    } else {
        return 1;
    }
}

int Sampler::can_remove(int u, size_t z, size_t d, size_t c) {
    if (u < 0)
        return 1;
    if (u == 2) {
        if (ndk[d][z] == 1)
            return 1;
        if (tdk[d][z] > 1)
            return 1;
    }
    if (u == 1) {
        if (ndk[d][z] == 1 && nck[c][z] == 1)
            return 1;
        if (tdk[d][z] > 1 && tck[c][z] > 1)
            return 1;
    }
    if (u == 0) {
        if (ndk[d][z] == 1 && nck[c][z] == 1 && n0k[z] == 1)
            return 1;
        if (tdk[d][z] > 1 && tck[c][z] > 1 && t0k[z] > 1)
            return 1;
    }
    return 0;
}

void Sampler::stat_incr(int u, size_t c, size_t z, size_t y, size_t d, size_t w, int val) {
    assert(!(u >= 0 && val < 0 && nck[c][z] == 0));
    switch(u) {
        case 0:
            if (t0k[z] == 0) {
                topics.push_back(z);
                dbg("new root topic %d\n", z);
                reset();
            }

            t0k[z] += val;
            n0k[z] += val;
            tck[c][z] += val;
            nck[c][z] += val;
            N0_ += val;
            Nc_[c] += val;
            if (t0k[z] == 0) {
                topics.erase(std::remove(topics.begin(), topics.end(), z), topics.end());
                dbg("Removing root topic %d\n", z);
                reset();
            }
            break;
        case 1:
            n0k[z] += val;
            tck[c][z] += val;
            nck[c][z] += val;
            N0_ += val;
            Nc_[c] += val;
            if (tck[c][z] == 0 || (val > 0 && tck[c][z] == val)) {
                for (size_t d = 0; d < D; d++)
                    if (dC[d] == c)
                        alias_doc[d].ttl = 0;
            }
            break;
        case 2:
            nck[c][z] += val;
            Nc_[c] += val;
    }

    Nd_[d] += val;
    if (tdk[d].count(z) == 0) {
        assert(val > 0);
        tdk[d][z] = val;
        ndk[d][z] = val;
        alias_doc[d].ttl = 0;
    } else {
        ndk[d][z] += val;
        if (u >= 0) {
            tdk[d][z] += val;
            if (tdk[d][z] == 0) {
                tdk[d].erase(z);
                ndk[d].erase(z);
                alias_doc[d].ttl = 0;
            }
        }
    }

    nYZ[y][z] += val;
    nZ[z] += val;
    if (y == 0) {
        n0WZ[w][z] += val;
        n0Z[z] += val;
    } else {
        n1CWZ[c][w][z] += val;
        n1CZ[c][z] += val;
    }
}

Sampler::Sampler(size_t burn_in, size_t& Tsize, std::vector<size_t> &topics, uint32_t C, uint32_t D, uint32_t V,
        uint32_t *dW, uint8_t *dC, uint16_t **docs, uint8_t **dY, uint16_t **dZ,
        double *nZ, double *n0Z, double **nYZ,
        double **n0WZ, double ***n1CWZ, double **n1CZ,
        uint *t0k, uint *n0k, uint **tck, uint **nck, uint& N0_, uint *Nc_, uint *Nd_,
        double **U, double **X, size_t maxN, size_t maxK,
        std::map<size_t, uint> *tdk, std::map<size_t, uint> *ndk,
        double *b, double deltaNorm, double betaNorm, double *delta, double beta
        ) : burn_in(burn_in), Tsize(Tsize), topics(topics), C(C), D(D), V(V),
    dW(dW), dC(dC), docs(docs), dY(dY), dZ(dZ),
    nZ(nZ), n0Z(n0Z), nYZ(nYZ),
    n0WZ(n0WZ), n1CWZ(n1CWZ), n1CZ(n1CZ),
    t0k(t0k), n0k(n0k), tck(tck), nck(nck), N0_(N0_), Nc_(Nc_), Nd_(Nd_),
    U(U), X(X), maxN(maxN), maxK(maxK),
    tdk(tdk), ndk(ndk),
    b(b), deltaNorm(deltaNorm), betaNorm(betaNorm), delta(delta), beta(beta)
{

    srand48(1);
    p = (double*)malloc(2 * Tsize * sizeof(double));
    p0 = (double*)malloc(2 * Tsize * sizeof(double));

    alias_word = new Alias*[C];
    for (size_t i = 0; i < C; i++)
        alias_word[i] = new Alias[V];
    alias_doc = new Alias[D];
}

void Sampler::reset() {
    for (size_t d = 0; d < D; d++)
        alias_doc[d].ttl = 0;
    for (size_t c = 0; c < C; c++) {
        for (size_t w = 0; w < V; w++) {
            alias_word[c][w].ttl = 0;
        }
    }
}

std::vector<size_t> Sampler::sampling(size_t iter, size_t *Tsize_)
{
    if (iter < burn_in)
        N = 4;
    else
        N = 4;

    reset();

    std::vector<double> Ps;
    std::vector<int> Us;
    std::vector<size_t> Zs;

    for (size_t d = 0; d < D; d++) {
        size_t c = dC[d];
        for (size_t n = 0; n < dW[d]; n++) {
            size_t w = docs[d][n];
            size_t y = dY[d][n];
            size_t z = dZ[d][n];

            /* decrement counts */
            int u = sampleU(d, z, c);
            if (!can_remove(u, z, d, c)) {
                continue;
            }
            stat_incr(u, c, z, y, d, w, -1);

            /* sample z */
            size_t j;
            for (size_t kk = 0; kk < N; kk++) {
                double pi = 1;
                int uj;

                if (kk % 2 == 1) {
                    if (!alias_word[c][w].ttl) {
                        dbg("new alias word\n");
                        Ps.clear();
                        Zs.clear();
                        for (size_t i = 0; i < topics.size(); i++) {
                            size_t k = topics[i];
                            Ps.push_back((nYZ[0][k] * n0WZ[w][k] / n0Z[k] + nYZ[1][k] * n1CWZ[c][w][k] / n1CZ[c][k]) / nZ[k]);
                            Zs.push_back(k);
                        }
                        alias_word[c][w].generate_alias(Ps, Zs);
                    }
                    j = alias_word[c][w].sample();
                    pi /= alias_word[c][w].p[j];
                    j = alias_word[c][w].Zs[j];
                    pi *= alias_word[c][w].topicToP[z];
                    uj = sampleNewU(d, j, c);
                    dbg("sampled word: %d, %d\n", uj, j);
                } else {
                    if (!alias_doc[d].ttl) {

                        uint *tck_ = tck[c];
                        uint *nck_ = nck[c];

                        Ps.clear();
                        Us.clear();
                        Zs.clear();


                        for (size_t i = 0; i < topics.size(); i++) {
                            size_t k = topics[i];
                            assert(nck_[k] < maxN);
                            assert(tck_[k] < maxK);
                            if (tdk[d].find(k) != tdk[d].end()) {
                                uint tdk_ = tdk[d][k];
                                uint ndk_ = ndk[d][k];
                                assert(ndk_ < maxN && ndk > 0);
                                assert(tdk_ < maxK && tdk > 0);

                                // new dish
                                Ps.push_back(U[ndk_][tdk_] * (ndk_+1-tdk_)/(ndk_+1) * (b[1] + Nc_[c]) / b[2]);
                                Us.push_back(-1);
                                Zs.push_back(k);
                                // new doc table
                                Ps.push_back(U[nck_[k]][tck_[k]] * X[ndk_][tdk_] * (tdk_+1)/(nck_[k]+1) * (nck_[k]+1-tck_[k])/(ndk_+1));
                                Us.push_back(2);
                                Zs.push_back(k);

                                // new collect table
                                Ps.push_back(b[1] / (b[0]+N0_) * (n0k[k] * n0k[k]) * X[nck_[k]][tck_[k]] * X[ndk_][tdk_] * (tck_[k]+1)/(n0k[k]+1) * (tdk_+1)/(nck_[k]+1) / (ndk_+1));
                                Us.push_back(1);
                                Zs.push_back(k);
                            } else if (nck_[k] != 0) {
                                // new doc topic
                                Ps.push_back(U[nck_[k]][tck_[k]] * (nck_[k]+1-tck_[k]) / (nck_[k]+1));
                                Us.push_back(2);
                                Zs.push_back(k);
                                // new collect table
                                Ps.push_back(b[1] / (b[0]+N0_) * (n0k[k] * n0k[k]) * X[nck_[k]][tck_[k]] * (tck_[k]+1)/(n0k[k]+1)/(nck_[k]+1));
                                Us.push_back(1);
                                Zs.push_back(k);
                            } else {
                                // new collect topic
                                Ps.push_back(b[1] / (b[0]+N0_) * (n0k[k] * n0k[k]) / (n0k[k]+1));
                                Us.push_back(1);
                                Zs.push_back(k);
                            }
                        }
                        // new root topics
                        Ps.push_back(b[0] / (b[0] + N0_) * b[1]);
                        Us.push_back(0);

                        alias_doc[d].generate_alias(Ps, Zs, Us);
                    }
                    j = alias_doc[d].sample();
                    pi /= alias_doc[d].p[j];

                    uj = alias_doc[d].Us[j];
                    if (uj != 0) {
                        j = alias_doc[d].Zs[j];
                    } else { // new root topics
                        j = preAddTopic();
                        dbg("preAddTopic %d, %d\n", j, uj);
                    }

                    pi *= alias_doc[d].pairToP[std::make_pair(z, u)];
                }


                pi *= p_(j, w, c, d, uj);
                pi /= p_(z, w, c, d, u);

                if (drand48() < pi) {
                    z = j;
                    u = uj;
                    if (u == 0) {
                        alias_doc[d].ttl = 0;
                        for (size_t c = 0; c < C; c++) {
                            for (size_t w = 0; w < V; w++) {
                                alias_word[c][w].ttl = 0;
                            }
                        }
                    }
                }
            }

            /* sample y */
            double y0, y1;
            y0 = nYZ[0][z] * n0WZ[w][z] / n0Z[z];
            y1 = nYZ[1][z] * n1CWZ[c][w][z] / n1CZ[c][z];

            y0 = y0 / (y0 + y1);
            if (y0 > drand48())
                y = 0;
            else
                y = 1;

            /* increment counts */
            stat_incr(u, c, z, y, d, w, 1);

            /* update values */
            dY[d][n] = y;
            dZ[d][n] = z;
        }
    }

    *Tsize_ = Tsize;
    return topics;
}

Sampler::~Sampler() {
    for (size_t i = 0; i < C; i++)
        delete []alias_word[i];
    delete []alias_word;
    delete []alias_doc;
}
