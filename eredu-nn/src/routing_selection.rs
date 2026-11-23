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

/// Partitions ascending source IDs by score using value-only median partition.
/// Largest-value selection passes negated scores. The caller supplies matching
/// nonempty rows and an in-range pivot.
pub fn partition_ids(scores: &[f32], ids: &mut [usize], nth: usize) {
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

/// Evaluates a scalar logistic score without an overflowing exponential.
pub fn sigmoid(value: f32) -> f32 {
    if value >= 0. {
        1. / (1. + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1. + exp)
    }
}

/// Evaluates scalar softplus without an overflowing exponential.
pub fn softplus(value: f32) -> f32 {
    value.max(0.) + (-value.abs()).exp().ln_1p()
}

/// Selects one host score row, optionally restricting eligibility to the
/// partitions with the largest two-score sums. Native backends own score
/// formation and the rounding of those partition sums.
pub fn select_row(
    scores: &[f32],
    spec: crate::TopKGroupSelectionSpec,
    round_sum: impl Fn(f32) -> f32,
) -> Result<Vec<i32>, crate::Error> {
    use crate::{Error, GroupSelectionOrder};
    let groups = spec.group_count() as usize;
    let count = spec.top_k() as usize;
    if scores.len() != groups || scores.iter().any(|x| x.is_nan() || *x == f32::INFINITY) {
        return Err(Error::backend("invalid group ranking scores"));
    }
    let partitions = spec.selection_partitions() as usize;
    let selected = spec.selected_groups() as usize;
    let width = groups / partitions;
    let mut eligible = scores.to_vec();
    if selected != partitions {
        let mut rank: Vec<_> = scores
            .chunks_exact(width)
            .enumerate()
            .map(|(id, row)| {
                let mut top = [f32::NEG_INFINITY; 2];
                for &value in row {
                    if value > top[0] {
                        top = [value, top[0]];
                    } else if value > top[1] {
                        top[1] = value;
                    }
                }
                let score = if width == 1 {
                    top[0]
                } else {
                    round_sum(top[0] + top[1])
                };
                (id, score)
            })
            .collect();
        rank.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for &(id, _) in &rank[selected..] {
            eligible[id * width..][..width].fill(f32::NEG_INFINITY);
        }
    }
    let mut ids: Vec<_> = (0..groups).collect();
    match spec.order() {
        GroupSelectionOrder::Partition => {
            let descending: Vec<_> = eligible.iter().map(|&x| -x).collect();
            partition_ids(&descending, &mut ids, count - 1);
        }
        GroupSelectionOrder::Ranked => {
            ids.sort_unstable_by(|&a, &b| eligible[b].total_cmp(&eligible[a]).then(a.cmp(&b)));
        }
    }
    Ok(ids[..count].iter().map(|&id| id as i32).collect())
}
