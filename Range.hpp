#pragma once
#include <vector>

struct Range {
    int begin, end;

    int size() { return end - begin; }
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

std::vector<Range> partition_min(int begin, int end, int n, int min_size) {
    std::vector<Range> subranges;
    subranges.reserve(n);

    int step = std::max((end - begin) / n, std::min(end-begin, min_size));

    Range r;
    r.begin = begin;
    while (r.begin + step < end) {
        r.end = r.begin + step;

        subranges.push_back(r);

        r.begin = r.end;
    }

    r.end = end;
    subranges.push_back(r);

    return subranges;
}
