use eredu_nn::arithmetic::{Bf16DotGeometry, Bf16ValueReduction};

use super::{invalid, BackendFailure, CandleContext, CandleTensor, DType, Error, NativeTensor};

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum OrderedCpuMode {
    #[default]
    ArmFloat,
    Bfloat16,
}

impl OrderedCpuMode {
    fn exp(self, score: f32) -> f32 {
        match self {
            Self::ArmFloat => eredu_nn::arithmetic::ordered_f32::exp(score),
            Self::Bfloat16 => eredu_nn::arithmetic::stable_exp_f32(score),
        }
    }
}

/// Retains the native reduction lanes across page boundaries. The live catalog
/// supplies the same positions in order for each sweep.
#[derive(Default)]
pub(super) struct OrderedCpuAttention {
    mode: OrderedCpuMode,
    end: usize,
    pass: usize,
    frontier: i64,
    shape: [usize; 4],
    maxima: Vec<f32>,
    mass: Vec<f64>,
    tail: Vec<[f32; 4]>,
    value_lanes: Vec<f32>,
}

impl OrderedCpuAttention {
    pub(super) fn new(mode: OrderedCpuMode, end: usize) -> Self {
        Self {
            mode,
            end,
            ..Default::default()
        }
    }

    pub(super) fn begin_pass(&mut self, pass: usize, end: i64) -> Result<(), BackendFailure> {
        if pass != self.pass + 1 || self.frontier != end {
            return Err(invalid("paged attention sweep is incomplete"));
        }
        self.pass = pass;
        self.frontier = 0;
        Ok(())
    }

    fn value_reduction(&self) -> Bf16ValueReduction {
        Bf16DotGeometry {
            rows: self.shape[2] as u64,
            width: self.end as u64,
            columns: self.shape[3] as u64,
        }
        .value_reduction()
    }

    fn lanes(&self) -> usize {
        match self.mode {
            OrderedCpuMode::ArmFloat if self.shape[2] < 4 => 16,
            OrderedCpuMode::ArmFloat => 4,
            OrderedCpuMode::Bfloat16 => self.value_reduction().lanes(),
        }
    }

    pub(super) fn accumulate(
        &mut self,
        start: i64,
        scores: &NativeTensor,
        values: &NativeTensor,
    ) -> Result<(), Error> {
        use rayon::prelude::*;

        let (batch, heads, queries, width) = scores.dims4().map_err(Error::backend_source)?;
        let (value_batch, value_heads, value_tokens, features) =
            values.dims4().map_err(Error::backend_source)?;
        let shape = [batch, heads, queries, features];
        if width == 0
            || queries == 0
            || features == 0
            || (batch, heads, width) != (value_batch, value_heads, value_tokens)
            || start != self.frontier
            || (!self.maxima.is_empty() && shape != self.shape)
        {
            return Err(Error::backend(
                "paged attention sweep has noncontiguous or incompatible pages",
            ));
        }
        self.shape = shape;
        self.frontier += width as i64;
        let rows = scores.elem_count() / width;
        let scores = scores
            .flatten_all()
            .and_then(|array| array.to_vec1::<f32>())
            .map_err(Error::backend_source)?;
        let start = start as usize;

        let mode = self.mode;
        let end = self.end;
        match self.pass {
            0 => {
                if self.maxima.is_empty() {
                    self.maxima.resize(rows, f32::MIN);
                }
                self.maxima
                    .par_iter_mut()
                    .zip(scores.par_chunks_exact(width))
                    .for_each(|(max, row)| {
                        *max = row.iter().copied().fold(*max, f32::max);
                    });
            }
            1 => {
                if self.mass.is_empty() {
                    self.mass.resize(rows, 0.);
                    self.tail.resize(rows, [0.; 4]);
                }
                self.mass
                    .par_iter_mut()
                    .zip(&mut self.tail)
                    .zip(&self.maxima)
                    .zip(scores.par_chunks_exact(width))
                    .for_each(|(((mass, tail), &max), row)| {
                        if max == f32::MIN {
                            return;
                        }
                        for (column, &score) in row.iter().enumerate() {
                            let lane = (start + column) % 4;
                            let exp = mode.exp(score - max);
                            if mode == OrderedCpuMode::Bfloat16 {
                                tail[lane] += exp;
                            } else {
                                tail[lane] = exp;
                                if lane == 3 {
                                    *mass +=
                                        f64::from(eredu_nn::arithmetic::ordered_f32::sum4(*tail));
                                    *tail = [0.; 4];
                                }
                            }
                        }
                    });
            }
            2 => {
                let lanes = self.lanes();
                let value_page = self.value_reduction().page(start as u64, width);
                let stride = features
                    .checked_mul(lanes)
                    .ok_or_else(|| Error::backend("paged value scratch size overflow"))?;
                if self.value_lanes.is_empty() {
                    let length = rows
                        .checked_mul(stride)
                        .ok_or_else(|| Error::backend("paged value scratch size overflow"))?;
                    self.value_lanes
                        .try_reserve_exact(length)
                        .map_err(Error::backend_source)?;
                    self.value_lanes.resize(length, 0.);
                }
                let values = values
                    .flatten_all()
                    .and_then(|array| array.to_vec1::<f32>())
                    .map_err(Error::backend_source)?;
                self.value_lanes
                    .par_chunks_exact_mut(stride)
                    .zip(&self.maxima)
                    .zip(&self.mass)
                    .zip(&self.tail)
                    .zip(scores.par_chunks_exact(width))
                    .enumerate()
                    .for_each(|(row, ((((output, &max), &mass), &tail), scores))| {
                        let inverse = match mode {
                            OrderedCpuMode::ArmFloat => (mass
                                + f64::from(eredu_nn::arithmetic::ordered_f32::sum4(tail)))
                            .recip() as f32,
                            OrderedCpuMode::Bfloat16 => {
                                crate::operators::bf16_softmax_inverse(tail, end)
                            }
                        };
                        if !inverse.is_finite() {
                            return;
                        }
                        let values = &values[row / queries * width * features..];
                        for (column, (&score, values)) in
                            scores.iter().zip(values.chunks_exact(features)).enumerate()
                        {
                            let probability = mode.exp(score - max) * inverse;
                            let position = start + column;
                            match mode {
                                OrderedCpuMode::ArmFloat => {
                                    let lane = position % lanes;
                                    let accumulators = &mut output[lane * features..][..features];
                                    for (accumulator, &value) in accumulators.iter_mut().zip(values)
                                    {
                                        *accumulator = probability.mul_add(value, *accumulator);
                                    }
                                }
                                OrderedCpuMode::Bfloat16 => {
                                    let probability = half::bf16::from_f32(probability).to_f32();
                                    let lane = value_page.lane(column);
                                    let accumulators = &mut output[lane * features..][..features];
                                    for (accumulator, &value) in accumulators.iter_mut().zip(values)
                                    {
                                        *accumulator += probability * value;
                                    }
                                }
                            }
                        }
                    });
            }
            _ => return Err(Error::backend("unknown paged attention sweep")),
        }
        Ok(())
    }

    pub(super) fn scratch_bytes(&self) -> u64 {
        (self.maxima.len() * 4
            + self.mass.len() * 8
            + self.tail.len() * 16
            + self.value_lanes.len() * 4) as u64
    }

    pub(super) fn finish(
        self,
        end: i64,
        queue: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        if self.pass != 2 || self.frontier != end || self.value_lanes.is_empty() {
            return Err(invalid("paged attention value sweep is incomplete"));
        }
        let lanes = self.lanes();
        let features = self.shape[3];
        let mut output = Vec::with_capacity(self.value_lanes.len() / lanes);
        for row in self.value_lanes.chunks_exact(features * lanes) {
            for feature in 0..features {
                let mut partial = [0.; 16];
                for (slot, values) in partial.iter_mut().zip(row.chunks_exact(features)) {
                    *slot = values[feature];
                }
                let sum = match self.mode {
                    OrderedCpuMode::ArmFloat => {
                        eredu_nn::arithmetic::ordered_f32::reduce_dot_lanes(&mut partial[..lanes])
                    }
                    OrderedCpuMode::Bfloat16 => {
                        crate::matmul::reduce_bf16_value_lanes(partial[..4].try_into().unwrap())
                    }
                };
                output.push(sum);
            }
        }
        let dtype = if self.mode == OrderedCpuMode::Bfloat16 {
            DType::BF16
        } else {
            DType::F32
        };
        CandleTensor::wrap(
            NativeTensor::from_vec(output, self.shape.as_slice(), &queue.device)
                .and_then(|value| value.to_dtype(dtype)),
        )
        .map_err(BackendFailure::from_error)
    }
}
