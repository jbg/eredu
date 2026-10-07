//! Explicit-key dense random mechanisms and supplied-uniform Gumbel selection.

use eredu_runtime::{
    dense_sampling::{validate_dense_draw_shape, validate_gumbel_epsilon},
    DenseSamplingBackend, GumbelTruncation,
};
use safemlx::{argmax_axis, error::Exception, Array, Dtype, Stream};

use super::MlxSamplingBackend;
use crate::{backend::random::RandomState, MlxTensor};

impl DenseSamplingBackend for MlxSamplingBackend {
    fn uniform_f32(
        shape: &[i32],
        random: &mut RandomState,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        validate_dense_draw_shape(shape).map_err(|e| Exception::custom(e.to_string()))?;
        let mut candidate = random.clone();
        let key = candidate.next_key(stream)?;
        let output = safemlx::random::uniform::<_, f32>(0.0, 1.0, shape, &key, stream)?;
        *random = candidate;
        Ok(MlxTensor::from_array(output))
    }

    fn standard_normal_f32(
        shape: &[i32],
        random: &mut RandomState,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        validate_dense_draw_shape(shape).map_err(|e| Exception::custom(e.to_string()))?;
        let mut candidate = random.clone();
        let key = candidate.next_key(stream)?;
        let output = safemlx::random::normal::<f32>(shape, None, None, &key, stream)?;
        *random = candidate;
        Ok(MlxTensor::from_array(output))
    }

    fn dense_shape(value: &MlxTensor) -> &[i32] {
        value.as_array().shape()
    }

    fn gumbel_max(
        logits: &MlxTensor,
        uniform: &MlxTensor,
        truncation: GumbelTruncation,
        epsilon: f32,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        truncation
            .validate()
            .map_err(|e| Exception::custom(e.to_string()))?;
        validate_gumbel_epsilon(epsilon).map_err(|e| Exception::custom(e.to_string()))?;
        let logits = logits.as_array();
        let uniform = uniform.as_array();
        validate_dense_draw_shape(logits.shape()).map_err(|e| Exception::custom(e.to_string()))?;
        if logits.shape().is_empty()
            || uniform.shape() != logits.shape()
            || logits.dtype() != Dtype::Float32
            || uniform.dtype() != Dtype::Float32
        {
            return Err(Exception::custom(
                "Gumbel selection requires matching non-scalar F32 logits and uniforms",
            ));
        }
        let width = logits.dim(-1);
        let filtered = match truncation {
            GumbelTruncation::None => logits.clone(),
            GumbelTruncation::TopK(k) if k >= width as usize => logits.clone(),
            GumbelTruncation::TopK(k) => {
                let top = safemlx::ops::indexing::topk_axis(logits, k as i32, -1, stream)?;
                let threshold = top.min_axis(-1, true, stream)?;
                masked(logits.lt(threshold, stream)?, logits.clone(), stream)?
            }
            GumbelTruncation::TopP(p) => {
                let indices = safemlx::ops::argsort_axis(logits, -1, stream)?;
                let sorted = safemlx::ops::indexing::take_along_axis(logits, &indices, -1, stream)?;
                let probabilities = safemlx::ops::softmax_axis(&sorted, -1, true, stream)?;
                let cumulative = probabilities.cumsum(-1, None, None, stream)?;
                let below = cumulative.le(Array::from_f32((1.0 - p) as f32), stream)?;
                let not_last = Array::arange::<i32, i32>(None, width, None, stream)?
                    .lt(Array::from_int(width - 1), stream)?;
                let sorted = masked(below.logical_and(not_last, stream)?, sorted, stream)?;
                safemlx::ops::indexing::put_along_axis(logits, &indices, &sorted, -1, stream)?
            }
        };
        let log_probs = filtered.subtract(filtered.logsumexp_axis(-1, true, stream)?, stream)?;
        let noise = uniform
            .add(Array::from_f32(epsilon), stream)?
            .log(stream)?
            .negative(stream)?
            .add(Array::from_f32(epsilon), stream)?
            .log(stream)?
            .negative(stream)?;
        Ok(MlxTensor::from_array(
            argmax_axis!(&log_probs.add(noise, stream)?, -1, stream = stream)?
                .as_dtype(Dtype::Int32, stream)?,
        ))
    }
}

fn masked(mask: Array, logits: Array, stream: &Stream) -> Result<Array, Exception> {
    safemlx::ops::r#where(mask, Array::from_f32(f32::NEG_INFINITY), logits, stream)
}
