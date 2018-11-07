#pragma once
#include <vector>

struct Range {
    int begin, end;
};

// Partition the range [begin, end) into 'n' equally sized parts.
//
std::vector<Range> partition(int begin, int end, int n) {
    std::vector<Range> subranges(n);

    int step = (end - begin) / n;
    int rest = (end - begin) % n;

    for (int i = 0; i < n; i++) {
        subrange[i].begin = begin;
        begin += step;

        // Account for uneven division of the range by n
        if (i < rest)
            begin++;

        subrange[i].end = begin;
    }
    return subranges;
}
