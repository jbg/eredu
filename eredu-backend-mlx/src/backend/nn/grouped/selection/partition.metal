// Value-only selection adapted from libc++ __algorithm/nth_element.h and sort.h.
// Copyright (c) the LLVM Project contributors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt
// Indices, comparison and address spaces are specialized for routing scores.
inline bool route_less(device const float* scores, uint a, uint b) {
    float x = scores[a], y = scores[b];
    return isnan(x) ? false : (isnan(y) || x < y);
}
inline void route_swap(thread uint* ids, int a, int b) {
    uint tmp = ids[a]; ids[a] = ids[b]; ids[b] = tmp;
}
inline bool route_sort3(device const float* scores, thread uint* ids, int x, int y, int z) {
    if (!route_less(scores, ids[y], ids[x])) {
        if (!route_less(scores, ids[z], ids[y])) return false;
        route_swap(ids, y, z);
        if (route_less(scores, ids[y], ids[x])) route_swap(ids, x, y);
        return true;
    }
    if (route_less(scores, ids[z], ids[y])) {
        route_swap(ids, x, z); return true;
    }
    route_swap(ids, x, y);
    if (route_less(scores, ids[z], ids[y])) route_swap(ids, y, z);
    return true;
}
inline void route_partition(device const float* scores, thread uint* ids, int nth, int last) {
    int first = 0;
    while (true) {
        int len = last - first;
        if (nth == last || len < 2) return;
        if (len == 2) {
            if (route_less(scores, ids[last-1], ids[first])) route_swap(ids, first, last-1);
            return;
        }
        if (len == 3) { route_sort3(scores, ids, first, first+1, last-1); return; }
        if (len <= 7) {
            for (int a = first; a < last-1; ++a) {
                int best = a;
                for (int b = a+1; b < last; ++b)
                    if (route_less(scores, ids[b], ids[best])) best = b;
                if (best != a) route_swap(ids, a, best);
            }
            return;
        }
        int m = first + len/2, lm1 = last-1;
        bool swapped = route_sort3(scores, ids, first, m, lm1);
        int i = first, j = lm1;
        if (!route_less(scores, ids[i], ids[m])) {
            bool guard = false;
            while (i != --j) {
                if (route_less(scores, ids[j], ids[m])) { guard = true; break; }
            }
            if (guard) { route_swap(ids, i, j); swapped = true; }
            else {
                ++i; j = last;
                if (!route_less(scores, ids[first], ids[--j])) {
                    while (true) {
                        if (i == j) return;
                        if (route_less(scores, ids[first], ids[i])) {
                            route_swap(ids, i, j); swapped = true; ++i; break;
                        }
                        ++i;
                    }
                }
                if (i == j) return;
                while (true) {
                    while (!route_less(scores, ids[first], ids[i])) ++i;
                    do { --j; } while (route_less(scores, ids[first], ids[j]));
                    if (i >= j) break;
                    route_swap(ids, i, j); swapped = true; ++i;
                }
                if (nth < i) return;
                first = i; continue;
            }
        }
        ++i;
        if (i < j) {
            while (true) {
                while (route_less(scores, ids[i], ids[m])) ++i;
                do { --j; } while (!route_less(scores, ids[j], ids[m]));
                if (i >= j) break;
                route_swap(ids, i, j); swapped = true;
                if (m == i) m = j;
                ++i;
            }
        }
        if (i != m && route_less(scores, ids[m], ids[i])) { route_swap(ids, i, m); swapped = true; }
        if (nth == i) return;
        if (!swapped) {
            int lo = nth < i ? first : i, hi = nth < i ? i : last;
            bool sorted = true;
            for (int p = lo+1; p < hi; ++p) {
                if (route_less(scores, ids[p], ids[p-1])) { sorted = false; break; }
            }
            if (sorted) return;
        }
        if (nth < i) last = i; else first = i+1;
    }
}
