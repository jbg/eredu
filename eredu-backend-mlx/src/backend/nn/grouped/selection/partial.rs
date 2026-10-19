//! Value-only median partition, retaining the selected prefix's internal order.
// Adapted from libc++ __algorithm/nth_element.h and sort.h, as in partition.metal.
// Copyright (c) the LLVM Project contributors.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See LICENSE-LLVM.txt in this directory.

fn less(scores: &[f32], a: usize, b: usize) -> bool {
    let (x, y) = (scores[a], scores[b]);
    !x.is_nan() && (y.is_nan() || x < y)
}

fn sort3(scores: &[f32], ids: &mut [usize], x: usize, y: usize, z: usize) -> bool {
    if !less(scores, ids[y], ids[x]) {
        if !less(scores, ids[z], ids[y]) {
            return false;
        }
        ids.swap(y, z);
        if less(scores, ids[y], ids[x]) {
            ids.swap(x, y);
        }
        return true;
    }
    if less(scores, ids[z], ids[y]) {
        ids.swap(x, z);
        return true;
    }
    ids.swap(x, y);
    if less(scores, ids[z], ids[y]) {
        ids.swap(y, z);
    }
    true
}

/// The caller supplies ascending source IDs and a valid nth position. Scores
/// use ascending comparison, so largest-value selection passes negated scores.
pub(super) fn partition_ids(scores: &[f32], ids: &mut [usize], nth: usize) {
    let (mut first, mut last) = (0, ids.len());
    loop {
        let length = last - first;
        if nth == last || length < 2 {
            return;
        }
        if length == 2 {
            if less(scores, ids[last - 1], ids[first]) {
                ids.swap(first, last - 1);
            }
            return;
        }
        if length == 3 {
            sort3(scores, ids, first, first + 1, last - 1);
            return;
        }
        if length <= 7 {
            for a in first..last - 1 {
                let mut best = a;
                for b in a + 1..last {
                    if less(scores, ids[b], ids[best]) {
                        best = b;
                    }
                }
                if best != a {
                    ids.swap(a, best);
                }
            }
            return;
        }
        let mut median = first + length / 2;
        let mut swapped = sort3(scores, ids, first, median, last - 1);
        let (mut i, mut j) = (first, last - 1);
        if !less(scores, ids[i], ids[median]) {
            let mut guard = false;
            loop {
                j -= 1;
                if i == j {
                    break;
                }
                if less(scores, ids[j], ids[median]) {
                    guard = true;
                    break;
                }
            }
            if guard {
                ids.swap(i, j);
                swapped = true;
            } else {
                i += 1;
                j = last - 1;
                if !less(scores, ids[first], ids[j]) {
                    loop {
                        if i == j {
                            return;
                        }
                        if less(scores, ids[first], ids[i]) {
                            ids.swap(i, j);
                            swapped = true;
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                }
                if i == j {
                    return;
                }
                loop {
                    while !less(scores, ids[first], ids[i]) {
                        i += 1;
                    }
                    loop {
                        j -= 1;
                        if !less(scores, ids[first], ids[j]) {
                            break;
                        }
                    }
                    if i >= j {
                        break;
                    }
                    ids.swap(i, j);
                    swapped = true;
                    i += 1;
                }
                if nth < i {
                    return;
                }
                first = i;
                continue;
            }
        }
        i += 1;
        if i < j {
            loop {
                while less(scores, ids[i], ids[median]) {
                    i += 1;
                }
                loop {
                    j -= 1;
                    if less(scores, ids[j], ids[median]) {
                        break;
                    }
                }
                if i >= j {
                    break;
                }
                ids.swap(i, j);
                swapped = true;
                if median == i {
                    median = j;
                }
                i += 1;
            }
        }
        if i != median && less(scores, ids[median], ids[i]) {
            ids.swap(i, median);
            swapped = true;
        }
        if nth == i {
            return;
        }
        if !swapped {
            let (lo, hi) = if nth < i { (first, i) } else { (i, last) };
            if (lo + 1..hi).all(|p| !less(scores, ids[p], ids[p - 1])) {
                return;
            }
        }
        if nth < i {
            last = i;
        } else {
            first = i + 1;
        }
    }
}
