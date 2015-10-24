#ifndef ALIAS_H
#define ALIAS_H

#include <stdint.h>

class Alias {
        uint16_t *table;
        uint16_t _ttl;
    public:
        double sum;
        double *p;
        uint16_t n;
        uint16_t ttl;
        void init(double*, uint16_t, uint16_t);
        Alias() : n(0) {};
        ~Alias();
        Alias& operator= (const Alias &a);
        uint16_t sample();
        void generate_alias(double*);
};

#endif
