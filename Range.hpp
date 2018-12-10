#pragma once
#include <cassert>
#include <vector>

struct Range {
    int begin, end;

    int size() { 
        assert(end >= begin);
        return end - begin; 
    }
};

// Partition the range [begin, end) into 'n' equally sized parts.
//
std::vector<Range> partition(int begin, int end, int n) {
    std::vector<Range> subranges(n);

    int step = (end - begin) / n;
    int rest = (end - begin) % n;

    for (int i = 0; i < n; i++) {
        subranges[i].begin = begin;
        begin += step;

        // Account for uneven division of the range by n
        if (i < rest)
            begin++;

        subranges[i].end = begin;
    }
    return subranges;
}

std::vector<Range> partition_tiles(int begin, int end, int tilesize) {
    std::vector<Range> subranges;

    while (begin < end) {
        Range r;
        r.begin = begin;
        r.end = std::min(begin + tilesize, end);
        begin = r.end;
        subranges.push_back(r);
    }

    return subranges;
}

std::vector<Range> partition_min(int begin, int end, int n, int min_size) {
    std::vector<Range> subranges;
    subranges.reserve(n);

    int step = std::max((end - begin) / n, std::min(end-begin, min_size));

    Range r;
    r.begin = begin;
    for (int i = 0; i < n; i++) {
        r.end = std::min(r.begin + step, end);

        subranges.push_back(r);

        r.begin = r.end;
    }

    r.end = end;
    subranges.push_back(r);

    return subranges;
}
