//! Scalar realization of the explicit RMS reduction policies.
//! Expected numerical values come from the independently pinned Torch fixtures.

pub(super) fn cascaded_squares(values: &[f32]) -> f32 {
    let vectors = values.len() / 4;
    let steps = vectors / 4;
    let log = usize::BITS as usize - steps.saturating_sub(1).leading_zeros() as usize;
    let power = 4.max(log / 4);
    let chunk = 1usize << power;
    let mut sums = [[[0.0_f32; 4]; 4]; 4];
    let mut index = 0;
    while index + chunk <= steps {
        let end = index + chunk;
        while index < end {
            for p in 0..4 {
                for lane in 0..4 {
                    let value = values[(index * 4 + p) * 4 + lane];
                    sums[0][p][lane] += value * value;
                }
            }
            index += 1;
        }
        for level in 1..4 {
            for p in 0..4 {
                for lane in 0..4 {
                    sums[level][p][lane] += sums[level - 1][p][lane];
                    sums[level - 1][p][lane] = 0.0;
                }
            }
            if index & ((chunk - 1) << (level * power)) != 0 {
                break;
            }
        }
    }
    while index < steps {
        for p in 0..4 {
            for lane in 0..4 {
                let value = values[(index * 4 + p) * 4 + lane];
                sums[0][p][lane] += value * value;
            }
        }
        index += 1;
    }
    for level in 1..4 {
        for p in 0..4 {
            for lane in 0..4 {
                sums[0][p][lane] += sums[level][p][lane];
            }
        }
    }
    for vector in steps * 4..vectors {
        for lane in 0..4 {
            let value = values[vector * 4 + lane];
            sums[0][0][lane] += value * value;
        }
    }
    for p in 1..4 {
        for lane in 0..4 {
            sums[0][0][lane] += sums[0][p][lane];
        }
    }
    let mut sum = 0.0_f32;
    for &value in &values[vectors * 4..] {
        sum += value * value;
    }
    for lane in 0..4 {
        sum += sums[0][0][lane];
    }
    sum
}

pub(super) fn channel_squares(values: &[f32], parallel: usize) -> f32 {
    let steps = values.len() / parallel;
    let log = usize::BITS as usize - steps.saturating_sub(1).leading_zeros() as usize;
    let power = 4.max(log / 4);
    let chunk = 1usize << power;
    let mut sums = [[0.0_f32; 4]; 4];
    let mut index = 0;
    while index + chunk <= steps {
        let end = index + chunk;
        while index < end {
            for p in 0..parallel {
                let value = values[index * parallel + p];
                sums[0][p] += value * value;
            }
            index += 1;
        }
        for level in 1..4 {
            for p in 0..parallel {
                sums[level][p] += sums[level - 1][p];
                sums[level - 1][p] = 0.0;
            }
            if index & ((chunk - 1) << (level * power)) != 0 {
                break;
            }
        }
    }
    while index < steps {
        for p in 0..parallel {
            let value = values[index * parallel + p];
            sums[0][p] += value * value;
        }
        index += 1;
    }
    for level in 1..4 {
        for p in 0..parallel {
            sums[0][p] += sums[level][p];
        }
    }
    for value in &values[steps * parallel..] {
        sums[0][0] += value * value;
    }
    for p in 1..parallel {
        sums[0][0] += sums[0][p];
    }
    sums[0][0]
}
