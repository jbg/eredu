//! Scalar reference for ordered binary32 attention and activation arithmetic.

// The ARM vector exponential computes 2^n * (1 + p(remainder)).
// Range reduction and polynomial coefficients are from Arm Optimized Routines
// via ggml. See THIRD-PARTY-NOTICES.
/// Exponential for the four-lane FP32 arithmetic.
pub fn exp(x: f32) -> f32 {
    const SHIFT: f32 = 12_582_912.;
    const LN_2_HIGH: f32 = f32::from_bits(0x3f31_7200);
    const LN_2_LOW: f32 = f32::from_bits(0x35bf_be8e);
    let biased = x.mul_add(std::f32::consts::LOG2_E, SHIFT);
    let n = biased - SHIFT;
    if !x.is_finite() || n.abs() > 126. {
        return x.exp();
    }
    let b = (-n).mul_add(LN_2_LOW, (-n).mul_add(LN_2_HIGH, x));
    let scale = f32::from_bits((biased.to_bits() << 23).wrapping_add(1f32.to_bits()));
    let b2 = b * b;
    let low = b.mul_add(f32::from_bits(0x3e2a_af33), f32::from_bits(0x3eff_fedb));
    let high = b.mul_add(f32::from_bits(0x3c07_2010), f32::from_bits(0x3d2b_9f17));
    let adjustment = high
        .mul_add(b2, low)
        .mul_add(b2, f32::from_bits(0x3f7f_fff6) * b);
    scale.mul_add(adjustment, scale)
}

/// Applies SiLU with the vector exponential on complete four-value groups.
pub fn silu(row: &mut [f32]) {
    let (groups, tail) = row.as_chunks_mut::<4>();
    for x in groups.iter_mut().flatten() {
        *x /= 1. + exp(-*x);
    }
    for x in tail {
        *x /= 1. + (-*x).exp();
    }
}

/// Normalizes a nonempty row with the blocked FP32 exponential and FP64 mass.
/// The caller supplies visible scores or negative infinity as the mask.
pub fn softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (groups, tail) = row.as_chunks_mut::<4>();
    let mut sum = 0.0f64;
    for group in groups {
        for x in group.iter_mut() {
            *x = exp(*x - max);
        }
        sum += f64::from(sum4(*group));
    }
    // A partial vector contributes zero in its unused probability lanes.
    let mut last = [0.; 4];
    for (x, lane) in tail.iter_mut().zip(&mut last) {
        *x = exp(*x - max);
        *lane = *x;
    }
    sum += f64::from(sum4(last));
    let inverse = sum.recip() as f32;
    for x in row {
        *x *= inverse;
    }
}

/// Pairwise sum of four FP32 lanes.
pub fn sum4([a, b, c, d]: [f32; 4]) -> f32 {
    (a + b) + (c + d)
}

/// Lane count for the value product of an unpartitioned query batch.
pub const fn attention_value_lanes(query_rows: u64) -> usize {
    if query_rows >= 4 {
        4
    } else {
        16
    }
}

/// Fused dot with four or sixteen lanes and a fixed pairwise merge.
/// The caller selects whether the tail occupies its original lanes.
pub fn dot<const N: usize>(a: &[f32], b: &[f32], pad: bool) -> f32 {
    let mut lanes = [0.; N];
    let aligned = a.len() / N * N;
    let end = if pad { a.len() } else { aligned };
    for (a, b) in a[..end].chunks(N).zip(b[..end].chunks(N)) {
        for ((sum, &a), &b) in lanes.iter_mut().zip(a).zip(b) {
            *sum = a.mul_add(b, *sum);
        }
    }
    let mut sum = reduce_dot_lanes(&mut lanes);
    for (&a, &b) in a[end..].iter().zip(&b[end..]) {
        sum = a.mul_add(b, sum);
    }
    sum
}

/// Merges four or sixteen FP32 lanes.
pub fn reduce_dot_lanes(lanes: &mut [f32]) -> f32 {
    for step in [8, 4] {
        if lanes.len() > step {
            for lane in 0..step {
                lanes[lane] += lanes[lane + step];
            }
        }
    }
    sum4(lanes[..4].try_into().expect("four dot product lanes"))
}

/// A sixteen-lane dot with scalar accumulation after the last complete group.
pub fn unpacked_dot(a: &[f32], b: &[f32]) -> f32 {
    dot::<16>(a, b, false)
}
