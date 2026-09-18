//! Weighted pooled-position selection with a fixed contraction and closed errors.
use crate::MlxTensor;
use eredu_nn::{Error, PooledPositionInput, Tensor};
use safemlx::{Array, Dtype, OriginalScopeObserver, Stream, error::Exception, ops};
use std::mem::{size_of, size_of_val};

#[derive(Clone, Copy, Debug)]
pub(in crate::backend::nn) struct Geometry {
    pub batch: i32,
    pub heads: i32,
    pub queries: i32,
    pub dimensions: i32,
    pub pooled: i32,
    pub selected: i32,
    pub flattened_queries: i32,
}
#[derive(Clone, Copy, Debug, thiserror::Error)]
pub(in crate::backend::nn) enum GeometryError {
    #[error("invalid pooled-position geometry")]
    Invalid,
    #[error("pooled-position flattened query extent overflowed")]
    Overflow,
}
impl Geometry {
    pub(in crate::backend::nn) fn new(
        query: &[i32],
        pooled: &[i32],
        weights: &[i32],
        top_k: i32,
        scale: f32,
        head_scale: f32,
    ) -> Result<Self, GeometryError> {
        if query.len() != 4
            || pooled.len() != 3
            || weights.len() != 3
            || query[0] != pooled[0]
            || weights != [query[0], query[2], query[1]]
            || query[3] != pooled[2]
            || top_k <= 0
            || !scale.is_finite()
            || scale <= 0.0
            || !head_scale.is_finite()
            || head_scale <= 0.0
            || query.iter().any(|n| *n <= 0)
            || pooled[1] < 0
        {
            return Err(GeometryError::Invalid);
        }
        Ok(Self {
            batch: query[0],
            heads: query[1],
            queries: query[2],
            dimensions: query[3],
            pooled: pooled[1],
            selected: top_k.min(pooled[1]),
            // The empty branch never enters the contraction or flattens heads.
            flattened_queries: if pooled[1] == 0 {
                0
            } else {
                query[1]
                    .checked_mul(query[2])
                    .ok_or(GeometryError::Overflow)?
            },
        })
    }
    pub(in crate::backend::nn) fn output(self) -> [i32; 3] {
        [self.batch, self.queries, self.selected]
    }
    pub(in crate::backend::nn) fn returned_handles(self, masked: bool) -> usize {
        if self.pooled == 0 {
            1
        } else {
            24 + 2 * usize::from(masked)
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct OriginalFailure {
    #[source]
    cause: GeometryError,
    _custody: Exception,
}
fn error(cause: Exception) -> Error {
    match OriginalScopeObserver::try_current() {
        Ok(None) => Error::backend_retained_source(cause),
        _ => Error::backend_retained_source(cause),
    }
}
fn geometry_error(cause: GeometryError, input: &PooledPositionInput<'_, MlxTensor>) -> Error {
    let custody = match OriginalScopeObserver::try_current() {
        Ok(Some(scope)) => scope.invalid_input_error(),
        Err(cause) => cause,
        Ok(None) => {
            return match cause {
                GeometryError::Invalid => Error::backend(format!(
                    "invalid pooled-position geometry: queries={:?} pooled={:?} weights={:?} top_k={}",
                    input.queries.shape(),
                    input.pooled_keys.shape(),
                    input.head_weights.shape(),
                    input.top_k,
                )),
                GeometryError::Overflow => Error::backend(cause),
            };
        }
    };
    Error::backend_retained_source(Exception::from_retained_source(OriginalFailure {
        cause,
        _custody: custody,
    }))
}

// Exact two-operand einsum path for bhld,bpd->bhlp. Native path construction
// keeps operand order [0,1]; its batch/concat/contract axes are [0]/[1,2]/[3]
// and [0]/[1]/[2]. Keep even identity views, preserving source layout decisions.
fn scores(query: &Array, pooled: &Array, g: Geometry, stream: &Stream) -> Result<Array, Exception> {
    let query = ops::broadcast_to(query, query.shape(), stream)?;
    let pooled = ops::broadcast_to(pooled, pooled.shape(), stream)?;
    let query = query
        .transpose_axes(&[0, 1, 2, 3], stream)?
        .reshape(&[g.batch, g.flattened_queries, g.dimensions], stream)?;
    let pooled = pooled
        .transpose_axes(&[0, 2, 1], stream)?
        .reshape(&[g.batch, g.dimensions, g.pooled], stream)?;
    ops::matmul(&query, &pooled, stream)?
        .reshape(&[g.batch, g.heads, g.queries, g.pooled], stream)?
        .transpose_axes(&[0, 1, 2, 3], stream)
}

pub(in crate::backend::nn) fn run(
    input: PooledPositionInput<'_, MlxTensor>,
    stream: &Stream,
) -> Result<MlxTensor, Error> {
    let g = Geometry::new(
        input.queries.shape(),
        input.pooled_keys.shape(),
        input.head_weights.shape(),
        input.top_k,
        input.scale,
        input.head_scale,
    )
    .map_err(|cause| geometry_error(cause, &input))?;
    if g.pooled == 0 {
        return Array::zeros::<u32>(&g.output(), stream)
            .map(MlxTensor::from_array)
            .map_err(error);
    }
    let query = input
        .queries
        .as_array()
        .as_dtype(Dtype::Float32, stream)
        .map_err(error)?;
    let pooled = input
        .pooled_keys
        .as_array()
        .as_dtype(Dtype::Float32, stream)
        .map_err(error)?;
    let scores = scores(&query, &pooled, g, stream).map_err(error)?;
    let scores =
        ops::maximum(scores, Array::try_from_f32(0.0).map_err(error)?, stream).map_err(error)?;
    let scores = scores
        .multiply(Array::try_from_f32(input.scale).map_err(error)?, stream)
        .map_err(error)?;
    let weights = input
        .head_weights
        .as_array()
        .as_dtype(Dtype::Float32, stream)
        .map_err(error)?;
    let weights = weights
        .multiply(
            Array::try_from_f32(input.head_scale).map_err(error)?,
            stream,
        )
        .map_err(error)?;
    let weights = weights.transpose_axes(&[0, 2, 1], stream).map_err(error)?;
    let weights = weights.expand_dims(-1, stream).map_err(error)?;
    let scores = scores.multiply(weights, stream).map_err(error)?;
    let mut scores = scores.sum_axis(1, false, stream).map_err(error)?;
    if let Some(mask) = input.mask {
        scores = ops::r#where(
            mask.as_array(),
            scores,
            Array::try_from_f32(f32::NEG_INFINITY).map_err(error)?,
            stream,
        )
        .map_err(error)?;
    }
    let indices = ops::argpartition_axis(&scores, -g.selected, -1, stream).map_err(error)?;
    // Same all-range tuple slice as before; fixed arrays remove only the host
    // indexing plan. The view retains the complete [batch,query,pooled] backing.
    indices
        .try_slice(
            &[0, 0, g.pooled - g.selected],
            &[g.batch, g.queries, g.pooled],
            &[1; 3],
            stream,
        )
        .map(MlxTensor::from_array)
        .map_err(error)
}

pub(in crate::backend::nn) fn control_bytes(g: Geometry, masked: bool) -> Option<usize> {
    let calls = g.returned_handles(masked);
    let sizes = [
        size_of::<PooledPositionInput<'_, MlxTensor>>() * 2,
        size_of::<(&Array, &Array, Geometry, &Stream)>(), // contraction frame
        size_of::<Geometry>() * 2,
        size_of::<Result<Geometry, GeometryError>>(),
        // Every actual safe returned handle and native/neural result transport.
        calls.checked_mul(size_of::<Array>())?,
        calls.checked_mul(size_of::<Result<Array, Exception>>())?,
        calls.checked_mul(size_of::<Result<Array, Error>>())?,
        size_of::<[Result<MlxTensor, Error>; 2]>(),
        size_of::<Result<MlxTensor, Exception>>(),
        size_of::<[&Stream; 2]>(),
        size_of::<[i32; 4]>() * 3, // query/output transposes and score reshape
        size_of::<[i32; 3]>() * 6, // pooled transpose, input reshapes, tail bounds
        size_of::<[i32; 2]>(),     // partition kth and axis
        size_of::<[f32; 4]>(),     // actual eager scalar constructor arguments
        size_of::<Dtype>() * 3,
        size_of::<[&Array; 8]>(),
        size_of::<Option<&MlxTensor>>(),
        size_of::<GeometryError>(),
        size_of::<(GeometryError, &PooledPositionInput<'_, MlxTensor>)>(),
        size_of::<Result<Option<OriginalScopeObserver>, Exception>>(),
        OriginalScopeObserver::control_bytes()?,
        Exception::retained_source_control_bytes::<OriginalFailure>()?,
        Error::retained_source_construction_bytes::<Exception>()?,
    ];
    sizes
        .into_iter()
        .try_fold(size_of_val(&sizes), usize::checked_add)
}
