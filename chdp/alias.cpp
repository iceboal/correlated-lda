#include <cstdlib>
#include <algorithm>
#include <vector>
#include <map>
#include <utility>
#include "alias.h"

#ifdef NDEBUG
    #define NDEBUG_DISABLED
    #undef NDEBUG
#endif
#include <cassert>
#ifdef NDEBUG_DISABLED
    #define NDEBUG        // re-enable NDEBUG if it was originally enabled
#endif

void Alias::init(std::vector<double> p) {
    generate_alias(p);
}

void Alias::init(std::vector<double> p, std::vector<size_t> Zs, std::vector<int> Us) {
    generate_alias(p, Zs, Us);
}

void Alias::init(std::vector<double> p, std::vector<size_t> Zs) {
    generate_alias(p, Zs);
}

Alias& Alias::operator= (const Alias &a) {
    table = a.table;
    p = a.p;
    sum = a.sum;
    ttl = a.ttl;
    return *this;
}

void Alias::generate_alias(std::vector<double> p, std::vector<size_t> Zs) {
    generate_alias(p);
    this->Zs = Zs;
    topicToP.clear();
    for (int i = 0; i < Zs.size(); i++)
        topicToP[Zs[i]] = p[i];
}


void Alias::generate_alias(std::vector<double> p, std::vector<size_t> Zs, std::vector<int> Us) {
    generate_alias(p);
    this->Zs = Zs;
    this->Us = Us;
    pairToP.clear();
    for (int i = 0; i < Zs.size(); i++)
        pairToP[std::make_pair(Zs[i], Us[i])] = p[i];
}

/* p will be modified */
void Alias::generate_alias(std::vector<double> p) {
    size_t n = p.size();
    double *prob = (double*)malloc(n*sizeof(double));
    uint16_t *alias = (uint16_t*)malloc(n*sizeof(uint16_t));
    uint16_t *S = (uint16_t*)malloc(n*sizeof(uint16_t));
    uint16_t *L = (uint16_t*)malloc(n*sizeof(uint16_t));

    sum = 0;
    uint16_t i;
    this->p = p;

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
    ttl = n; // TODO: allow custom value if needed

    table.clear();
    for (i = 0; i < n; i++) {
        uint16_t j = (uint16_t) (n * drand48());
        table.push_back(drand48() < prob[j] ? j : alias[j]);
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
