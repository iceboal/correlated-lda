#ifndef ALIAS_H
#define ALIAS_H

#include <stdint.h>
#include <vector>
#include <map>
#include <utility>

class Alias {
        std::vector<uint16_t> table;
    public:
        std::map<std::pair<size_t,int>, double> pairToP;
        std::map<size_t, double> topicToP;
        std::vector<size_t> Zs;
        std::vector<int> Us;
        double sum;
        std::vector<double> p;
        uint16_t ttl;
        void init(std::vector<double> p);
        void init(std::vector<double> p, std::vector<size_t> Zs);
        void init(std::vector<double> p, std::vector<size_t> Zs, std::vector<int> Us);
        Alias() : ttl(0) {};
        Alias& operator= (const Alias &a);
        uint16_t sample();
        void generate_alias(std::vector<double> p);
        void generate_alias(std::vector<double> p, std::vector<size_t> Zs);
        void generate_alias(std::vector<double> p, std::vector<size_t> Zs, std::vector<int> Us);
};

#endif
