//! Scalar arithmetic and tensor composition for exact neural operator profiles.

pub mod ordered_f32;

use crate::{Error, Index, PadMode, Tensor, TensorElementType};

/// Portable scalar exponential source for C-like device compilers. Uses
/// FP32 fused multiply-add, nearest-even integer rounding, and power-of-two
/// scaling. The backend supplies the surrounding kernel and math intrinsics.
pub const STABLE_EXP_SOURCE: &str = include_str!("arithmetic/stable_exp.h");

/// FP32 exponential used before rounding input-precision attention probabilities.
///
/// The range reduction and polynomial are adapted from SLEEF xexpf. See the
/// source above for the revision, copyright, and license.
pub fn stable_exp_f32(value: f32) -> f32 {
    if value.is_nan() {
        return value;
    }
    if value > 100.0 {
        return f32::INFINITY;
    }
    if value < -104.0 {
        return 0.0;
    }
    let exponent = (value * std::f32::consts::LOG2_E).round_ties_even();
    let remainder = exponent.mul_add(-0.693_145_75, value);
    let remainder = exponent.mul_add(-1.428_606_8e-6, remainder);
    let poly = 0.000_198_527_62_f32
        .mul_add(remainder, 0.001_393_043_6)
        .mul_add(remainder, 0.008_333_361)
        .mul_add(remainder, 0.041_666_485)
        .mul_add(remainder, 0.166_666_67)
        .mul_add(remainder, 0.5);
    let mantissa = 1.0 + (remainder * remainder).mul_add(poly, remainder);
    let exponent = exponent as i32;
    let half = exponent >> 1;
    libm::scalbnf(libm::scalbnf(mantissa, half), exponent - half)
}

const BF16_VALUE_LANES: usize = 4;

/// Logical BF16 batched dot dimensions, excluding batches and heads.
#[derive(Clone, Copy)]
pub struct Bf16DotGeometry {
    /// Output rows per batch and head.
    pub rows: u64,
    /// Number of terms in each dot.
    pub width: u64,
    /// Output columns per batch and head.
    pub columns: u64,
}

impl Bf16DotGeometry {
    /// Selects the accumulation order using the full product before tiling.
    pub const fn uses_sequential(self) -> bool {
        self.rows != 0
            && self.width != 0
            && self.columns != 0
            && self
                .rows
                .saturating_mul(self.width)
                .saturating_mul(self.columns)
                < 400
    }

    /// Plans an ordered value dot over the full logical reduction width.
    ///
    /// The four-lane profile folds the lanes from left to right after all
    /// contributions. A native value GEMM can use its own large-dot kernel.
    pub const fn value_reduction(self) -> Bf16ValueReduction {
        if self.uses_sequential() {
            Bf16ValueReduction {
                lanes: 1,
                aligned_end: 0,
            }
        } else {
            Bf16ValueReduction {
                lanes: BF16_VALUE_LANES,
                aligned_end: self.width / BF16_VALUE_LANES as u64 * BF16_VALUE_LANES as u64,
            }
        }
    }
}

/// End positions of the complete 32-element and 8-element groups of a
/// nonsequential BF16 score dot. The remaining positions accumulate serially.
pub const fn bf16_row_block_ends(width: usize) -> [usize; 2] {
    [width / 32 * 32, width / 8 * 8]
}

/// Ordered BF16 value geometry shared by contiguous and partitioned products.
#[derive(Clone, Copy)]
pub struct Bf16ValueReduction {
    lanes: usize,
    aligned_end: u64,
}

impl Bf16ValueReduction {
    /// Number of FP32 lanes per output element.
    pub const fn lanes(self) -> usize {
        self.lanes
    }

    /// Maps part of the reduction axis onto the original accumulator lanes.
    pub fn page(self, start: u64, width: usize) -> Bf16ValuePage {
        Bf16ValuePage {
            first_lane: (start % self.lanes as u64) as usize,
            interleaved: self.aligned_end.saturating_sub(start).min(width as u64) as usize,
        }
    }
}

/// Lane assignment for a contiguous span of an ordered BF16 value dot.
#[derive(Clone, Copy)]
pub struct Bf16ValuePage {
    /// Lane of the first interleaved position, relative to the full dot.
    pub first_lane: usize,
    /// Number of positions in complete groups. The tail goes to lane zero.
    pub interleaved: usize,
}

impl Bf16ValuePage {
    /// Lane for a local position, including the serial tail.
    pub const fn lane(self, offset: usize) -> usize {
        if offset < self.interleaved {
            (self.first_lane + offset % BF16_VALUE_LANES) % BF16_VALUE_LANES
        } else {
            0
        }
    }
}

/// FP32 four-lane column projection with a serial tail in the first lane.
pub fn bf16_channel_dot(width: usize, product: impl Fn(usize) -> f32) -> f32 {
    let mut lanes = [0.0_f32; 4];
    let aligned = width / 4 * 4;
    for start in (0..aligned).step_by(4) {
        for (lane, total) in lanes.iter_mut().enumerate() {
            *total += product(start + lane);
        }
    }
    for index in aligned..width {
        lanes[0] += product(index);
    }
    ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3]
}

/// Channel-major RMS sum for one time row of a nondegenerate tensor.
pub fn channel_major_rms_sum(values: &[f32], time: usize, row: usize) -> f32 {
    let lanes = if row % time < (time / 16) * 16 { 1 } else { 4 };
    channel_major_sum_squares(values, lanes)
}

fn channel_major_sum_squares(values: &[f32], parallel: usize) -> f32 {
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

/// Contiguous cascaded FP32 sum of squares using the specified vector width.
pub fn cascaded_sum_squares<const LANES: usize>(values: &[f32]) -> f32 {
    let vectors = values.len() / LANES;
    let steps = vectors / 4;
    let log = usize::BITS as usize - steps.saturating_sub(1).leading_zeros() as usize;
    let power = 4.max(log / 4);
    let chunk = 1usize << power;
    let mut sums = [[[0.0_f32; LANES]; 4]; 4];
    let mut index = 0;
    while index + chunk <= steps {
        let end = index + chunk;
        while index < end {
            for p in 0..4 {
                for lane in 0..LANES {
                    let value = values[(index * 4 + p) * LANES + lane];
                    sums[0][p][lane] += value * value;
                }
            }
            index += 1;
        }
        for level in 1..4 {
            for p in 0..4 {
                for lane in 0..LANES {
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
            for lane in 0..LANES {
                let value = values[(index * 4 + p) * LANES + lane];
                sums[0][p][lane] += value * value;
            }
        }
        index += 1;
    }
    for level in 1..4 {
        for p in 0..4 {
            for lane in 0..LANES {
                sums[0][p][lane] += sums[level][p][lane];
            }
        }
    }
    for vector in steps * 4..vectors {
        for lane in 0..LANES {
            let value = values[vector * LANES + lane];
            sums[0][0][lane] += value * value;
        }
    }
    for p in 1..4 {
        for lane in 0..LANES {
            sums[0][0][lane] += sums[0][p][lane];
        }
    }
    let mut sum = 0.0_f32;
    for &value in &values[vectors * LANES..] {
        sum += value * value;
    }
    for lane in 0..LANES {
        sum += sums[0][0][lane];
    }
    sum
}

#[derive(Clone, Copy, Default)]
struct Moments {
    count: usize,
    mean: [f32; 4],
    square: [f32; 4],
}
impl Moments {
    fn add(&mut self, other: Self) {
        let count = self.count + other.count;
        let coefficient = if count == 0 {
            0.0
        } else {
            other.count as f32 / count as f32
        };
        for lane in 0..4 {
            let delta = other.mean[lane] - self.mean[lane];
            self.mean[lane] += coefficient * delta;
            self.square[lane] +=
                other.square[lane] + delta * delta * coefficient * self.count as f32;
        }
        self.count = count;
    }
}

fn rowwise_moments(values: &[f32], reduced: bool) -> (f32, f32) {
    let vector = if reduced { 8 } else { 4 };
    let vectors = values.len() / vector;
    let chunks = vectors.div_ceil(16);
    let depth = usize::BITS as usize - chunks.saturating_sub(1).leading_zeros() as usize;
    let mut stack = vec![Moments::default(); depth.max(1)];
    for chunk in 0..chunks {
        let count = 16.min(vectors - chunk * 16);
        for half in 0..(vector / 4) {
            let mut local = Moments {
                count,
                ..Moments::default()
            };
            for index in 0..count {
                let coefficient = 1.0_f32 / (index + 1) as f32;
                for lane in 0..4 {
                    let value = values[(chunk * 16 + index) * vector + half * 4 + lane];
                    let delta = value - local.mean[lane];
                    local.mean[lane] += delta * coefficient;
                    local.square[lane] += delta * (value - local.mean[lane]);
                }
            }
            stack[0].add(local);
        }
        let mut mask = chunk + 1;
        for level in 1..depth {
            if mask & 1 != 0 {
                break;
            }
            let prior = stack[level - 1];
            stack[level].add(prior);
            stack[level - 1] = Moments::default();
            mask >>= 1;
        }
    }
    for level in 1..depth {
        let prior = stack[level];
        stack[0].add(prior);
    }
    let mut count = 0usize;
    let mut mean = 0.0_f32;
    let mut square = 0.0_f32;
    for &value in &values[vectors * vector..] {
        let delta = value - mean;
        count += 1;
        mean += delta / count as f32;
        square += delta * (value - mean);
    }
    let additional = vectors * vector / 4;
    for lane in 0..4 {
        let next = count + additional;
        let coefficient = if next == 0 {
            0.0
        } else {
            additional as f32 / next as f32
        };
        let delta = stack[0].mean[lane] - mean;
        mean = coefficient.mul_add(delta, mean);
        square += (delta * delta * coefficient).mul_add(count as f32, stack[0].square[lane]);
        count = next;
    }
    (mean, square / values.len() as f32)
}

/// Cascaded FP32 layer normalization, including the selected affine order.
pub fn normalize_cascaded_f32(
    values: &mut [f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    epsilon: f32,
    reduced: bool,
) {
    let (mean, variance) = rowwise_moments(values, reduced);
    let inverse = 1.0_f32 / (variance + epsilon).sqrt();
    let shift = -inverse * mean;
    for (column, value) in values.iter_mut().enumerate() {
        let normalized = if reduced {
            *value * inverse + shift
        } else {
            (*value - mean) * inverse
        };
        *value = normalized * weight.map_or(1.0, |w| w[column]) + bias.map_or(0.0, |b| b[column]);
    }
}

/// Fused update of operands already rounded to binary16, returned as FP32.
pub fn binary16_mul_add(x: f32, y: f32, accumulator: f32) -> f32 {
    let exact = f64::from(x).mul_add(f64::from(y), f64::from(accumulator));
    let mut rounded = exact as f32;
    // A binary16 midpoint has an even FP32 low bit. Jam any lost bits into
    // an odd FP32 result so the final cast cannot double-round at a midpoint.
    if rounded.is_finite() && rounded.to_bits() & 1 == 0 && f64::from(rounded) != exact {
        let bits = rounded.abs().to_bits();
        let odd = if f64::from(rounded.abs()) < exact.abs() {
            bits + 1
        } else {
            bits - 1
        };
        rounded = f32::from_bits(odd).copysign(rounded);
    }
    half::f16::from_f32(rounded).to_f32()
}

/// Composes eight binary16 lanes using the supplied native lane update.
pub fn linear_binary16_lanes<T: Tensor>(
    input: &T,
    weight: &T,
    context: &T::Context,
    update: impl Fn(&T, &T, &T, &T::Context) -> Result<T, Error>,
) -> Result<T, Error> {
    let &[rows, width] = input.shape() else {
        return Err(Error::backend("FP16 projection requires a matrix input"));
    };
    let &[outputs, weight_width] = weight.shape() else {
        return Err(Error::backend("FP16 projection requires a matrix weight"));
    };
    if width <= 0 || width != weight_width {
        return Err(Error::backend("invalid FP16 projection feature width"));
    }
    let padding = (8 - width % 8) % 8;
    let padded_width = width
        .checked_add(padding)
        .ok_or_else(|| Error::backend("FP16 projection padded width overflows"))?;
    let prepare = |value: &T| {
        let value = value
            .cast_float(TensorElementType::F16, context)?
            .cast_float(TensorElementType::F32, context)?;
        if padding == 0 {
            Ok(value)
        } else {
            T::pad(&value, &[(0, 0), (0, padding)], PadMode::Constant, context)
        }
    };
    let input = prepare(input)?;
    let weight = prepare(weight)?;
    let mut lanes = T::full_f32(0., &[rows, outputs, 8], context)?;
    for start in (0..padded_width).step_by(8) {
        let slice = [Index::Full, Index::Range(start, start + 8)];
        let x = input
            .index(&slice, context)?
            .reshape(&[rows, 1, 8], context)?;
        let y = weight
            .index(&slice, context)?
            .reshape(&[1, outputs, 8], context)?;
        lanes = update(&x, &y, &lanes, context)?;
    }
    let combined = lanes
        .index(&[Index::Full, Index::Full, Index::Range(0, 4)], context)?
        .add(
            &lanes.index(&[Index::Full, Index::Full, Index::Range(4, 8)], context)?,
            context,
        )?;
    let lane = |i| combined.index(&[Index::Full, Index::Full, Index::At(i)], context);
    lane(0)?
        .add(&lane(1)?, context)?
        .add(&lane(2)?.add(&lane(3)?, context)?, context)
}

fn pairwise_sum4([a, b, c, d]: [f32; 4]) -> f32 {
    (a + b) + (c + d)
}

// The macOS ARM reduction merges the partial sums in a different order
// for a single block and for a sequence of blocks.
fn layer_norm_sum(row: &[f32], squared: bool) -> f32 {
    let (blocks, _) = row.as_chunks::<32>();
    if let [block] = blocks {
        let lanes = std::array::from_fn(|i| {
            let term = |group: usize| {
                let x = block[4 * group + i];
                let y = block[4 * (group + 4) + i];
                if squared {
                    y.mul_add(y, x * x)
                } else {
                    x
                }
            };
            let mut sum = (term(0) + term(2)) + (term(1) + term(3));
            if !squared {
                for group in 4..8 {
                    sum += block[4 * group + i];
                }
            }
            sum
        });
        return pairwise_sum4(lanes);
    }
    let [first, middle @ .., last] = blocks else {
        unreachable!()
    };
    let mut head = [0.0f32; 16];
    let mut tail = [0.0f32; 16];
    for i in 0..16 {
        if squared {
            head[i] = first[i] * first[i];
            tail[i] = first[16 + i] * first[16 + i];
        } else {
            head[i] = first[i] + first[16 + i];
        }
    }
    for block in middle {
        for i in 0..16 {
            if squared {
                head[i] = block[i].mul_add(block[i], head[i]);
                tail[i] = block[16 + i].mul_add(block[16 + i], tail[i]);
            } else {
                head[i] += block[i];
                tail[i] += block[16 + i];
            }
        }
    }
    for i in 0..16 {
        if squared {
            head[i] = last[i].mul_add(last[i], head[i]) + tail[i];
            head[i] = last[16 + i].mul_add(last[16 + i], head[i]);
        } else {
            head[i] += tail[i];
            head[i] += last[i];
            head[i] += last[16 + i];
        }
    }
    let lanes = std::array::from_fn(|i| (head[i] + head[i + 8]) + (head[i + 4] + head[i + 12]));
    pairwise_sum4(lanes)
}

/// Centers and normalizes an FP32 row in the blocked-moments profile.
/// The row must have a positive width divisible by 32.
pub fn normalize_blocked_f32(row: &mut [f32], epsilon: f32) {
    let width = row.len() as f32;
    let mean = layer_norm_sum(row, false) / width;
    for value in &mut *row {
        *value -= mean;
    }
    let variance = layer_norm_sum(row, true) / width;
    let inv = (variance + epsilon).sqrt().recip();
    for value in row {
        *value *= inv;
    }
}

/// Row-major sine/cosine pairs for the FP32 recurrence profile.
pub fn recurrent_rotary_tables(
    sequence: usize,
    width: usize,
    base: f32,
    scale: f32,
    offset: i32,
) -> Vec<(f32, f32)> {
    let step = base.powf(-2. / width as f32);
    (0..sequence)
        .flat_map(|position| {
            let mut angle = (i64::from(offset) + position as i64) as f32;
            (0..width / 2).map(move |_| {
                // Keep paired binary32 rounding independent of whether LLVM
                // combines platform sine and cosine calls in this build.
                let pair = libm::sincosf(angle * scale);
                angle *= step;
                pair
            })
        })
        .collect()
}

/// Composes the per-operation rounding boundaries of capped, masked attention.
pub fn capped_attention_probabilities<T: crate::Tensor>(
    logits: &T,
    mask: &T,
    softcap: f32,
    context: &T::Context,
) -> Result<T, crate::Error> {
    softcap_attention_scores(logits, softcap, context)?
        .add(mask, context)?
        .softmax_axis(-1, false, context)
}

/// Applies a positive tanh cap to scaled scores before masking.
pub fn softcap_attention_scores<T: crate::Tensor>(
    scores: &T,
    softcap: f32,
    context: &T::Context,
) -> Result<T, crate::Error> {
    if !softcap.is_finite() || softcap <= 0. {
        return Err(crate::Error::backend(
            "attention softcap must be finite and positive",
        ));
    }
    scores
        .multiply_scalar(softcap.recip(), context)?
        .tanh(context)?
        .multiply_scalar(softcap, context)
}
