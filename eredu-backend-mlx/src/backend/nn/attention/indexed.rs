//! Fixed contractions for the existing indexed local/pooled attention worker.
use crate::MlxTensor;
use eredu_nn::{
    Error, IndexedAttentionGeometry, IndexedAttentionInput, IndexedAttentionValidationError, Tensor,
};
use safemlx::{
    Array, Dtype, OriginalScopeObserver, Stream,
    error::{Exception, InexactDtypeError},
    ops,
};
use std::mem::{size_of, size_of_val};

#[derive(Clone, Copy, Debug)]
pub(in crate::backend::nn) struct Geometry {
    pub shape: IndexedAttentionGeometry,
    pub rows: i32,
    pub joined: i32,
}
#[derive(Debug, thiserror::Error)]
pub(in crate::backend::nn) enum Failure {
    #[error(transparent)]
    Semantic(#[from] IndexedAttentionValidationError),
    #[error("indexed-attention flattened or joined extent overflowed")]
    Overflow,
    #[error(transparent)]
    ScoreDtype(#[from] InexactDtypeError),
}
impl Geometry {
    pub(in crate::backend::nn) fn new(
        shapes: [&[i32]; 6],
        scale: f32,
        sinks: Option<&[i32]>,
    ) -> Result<Self, Failure> {
        let shape = IndexedAttentionGeometry::new(shapes, scale, sinks)?;
        Ok(Self {
            rows: shape
                .heads
                .checked_mul(shape.queries)
                .ok_or(Failure::Overflow)?,
            joined: shape
                .local
                .checked_add(shape.selected)
                .and_then(|n| n.checked_add(i32::from(sinks.is_some())))
                .ok_or(Failure::Overflow)?,
            shape,
        })
    }
    pub(in crate::backend::nn) fn output(self) -> [i32; 4] {
        let g = self.shape;
        [g.batch, g.heads, g.queries, g.value_dimensions]
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct OriginalFailure {
    #[source]
    cause: Failure,
    _custody: Exception,
}
fn failure(cause: Failure) -> Exception {
    match OriginalScopeObserver::try_current() {
        Ok(None) => match cause {
            Failure::ScoreDtype(cause) => cause.into(),
            cause => Exception::custom(cause.to_string()),
        },
        state => {
            let custody = match state {
                Ok(Some(scope)) => scope.invalid_input_error(),
                Err(cause) => cause,
                Ok(None) => unreachable!(),
            };
            Exception::from_retained_source(OriginalFailure {
                cause,
                _custody: custody,
            })
        }
    }
}
fn error(cause: Exception) -> Error {
    match OriginalScopeObserver::try_current() {
        Ok(None) => Error::backend_retained_source(cause),
        _ => Error::backend_retained_source(cause),
    }
}

pub(super) struct Input<'a> {
    pub values: [&'a Array; 6],
    pub scale: f32,
    pub local_mask: Option<&'a Array>,
    pub pooled_mask: Option<&'a Array>,
    pub sinks: Option<&'a Array>,
}
impl Input<'_> {
    fn geometry(&self) -> Result<Geometry, Failure> {
        Geometry::new(
            self.values.map(Array::shape),
            self.scale,
            self.sinks.map(Array::shape),
        )
    }
}
pub(in crate::backend::nn) fn execute(
    input: IndexedAttentionInput<'_, MlxTensor>,
    stream: &Stream,
) -> Result<MlxTensor, Error> {
    // The ordinary formatted diagnostic and the original fixed error consume
    // the same neutral validation result. No shape contract is reimplemented.
    if let Err(cause) = input.geometry() {
        return match OriginalScopeObserver::try_current() {
            Ok(None) => input.validate().map(|()| unreachable!()),
            Ok(Some(scope)) => Err(Error::backend_retained_source(
                Exception::from_retained_source(OriginalFailure {
                    cause: Failure::Semantic(cause),
                    _custody: scope.invalid_input_error(),
                }),
            )),
            Err(cause) => Err(Error::backend_retained_source(cause)),
        };
    }
    run(
        Input {
            values: [
                input.queries.as_array(),
                input.local_keys.as_array(),
                input.local_values.as_array(),
                input.pooled_keys.as_array(),
                input.pooled_values.as_array(),
                input.selected_positions.as_array(),
            ],
            scale: input.scale,
            local_mask: input.local_mask.map(MlxTensor::as_array),
            pooled_mask: input.pooled_mask.map(MlxTensor::as_array),
            sinks: input.sinks.map(MlxTensor::as_array),
        },
        stream,
    )
    .map(MlxTensor::from_array)
    .map_err(error)
}

// Fixed bhld,btd->bhlt / bhlt,btv->bhlv contractions. The two-operand
// einsum keeps [0,1], batches axis b and flattens h,l in their original order.
fn local(
    left: &Array,
    right: &Array,
    g: Geometry,
    contract: i32,
    columns: i32,
    transpose_right: bool,
    stream: &Stream,
) -> Result<Array, Exception> {
    let left = ops::broadcast_to(left, left.shape(), stream)?;
    let right = ops::broadcast_to(right, right.shape(), stream)?;
    let left = left
        .transpose_axes(&[0, 1, 2, 3], stream)?
        .reshape(&[g.shape.batch, g.rows, contract], stream)?;
    let right = right
        .transpose_axes(
            if transpose_right {
                &[0, 2, 1]
            } else {
                &[0, 1, 2]
            },
            stream,
        )?
        .reshape(&[g.shape.batch, contract, columns], stream)?;
    ops::matmul(&left, &right, stream)?
        .reshape(
            &[g.shape.batch, g.shape.heads, g.shape.queries, columns],
            stream,
        )?
        .transpose_axes(&[0, 1, 2, 3], stream)
}
// Fixed bhld,blkd->bhlk / bhlk,blkv->bhlv. Shared b,l axes precede
// h,contract; the result's final transpose restores b,h,l,value order.
fn selected(
    left: &Array,
    right: &Array,
    g: Geometry,
    contract: i32,
    columns: i32,
    transpose_right: bool,
    stream: &Stream,
) -> Result<Array, Exception> {
    let left = ops::broadcast_to(left, left.shape(), stream)?;
    let right = ops::broadcast_to(right, right.shape(), stream)?;
    let left = left.transpose_axes(&[0, 2, 1, 3], stream)?.reshape(
        &[g.shape.batch, g.shape.queries, g.shape.heads, contract],
        stream,
    )?;
    let right = right
        .transpose_axes(
            if transpose_right {
                &[0, 1, 3, 2]
            } else {
                &[0, 1, 2, 3]
            },
            stream,
        )?
        .reshape(&[g.shape.batch, g.shape.queries, contract, columns], stream)?;
    ops::matmul(&left, &right, stream)?
        .reshape(
            &[g.shape.batch, g.shape.queries, g.shape.heads, columns],
            stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)
}
fn gather(
    value: &Array,
    indices: &Array,
    g: Geometry,
    width: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let value = value.expand_dims(1, stream)?;
    let value = ops::broadcast_to(
        &value,
        &[g.shape.batch, g.shape.queries, g.shape.pooled, width],
        stream,
    )?;
    let indices = indices.expand_dims(-1, stream)?;
    let indices = ops::broadcast_to(
        &indices,
        &[g.shape.batch, g.shape.queries, g.shape.selected, width],
        stream,
    )?;
    ops::indexing::take_along_axis(value, &indices, 2, stream)
}
fn mask(scores: &mut Array, mask: Option<&Array>, stream: &Stream) -> Result<(), Exception> {
    let Some(mask) = mask else {
        return Ok(());
    };
    *scores = if mask.dtype() == Dtype::Bool {
        let minimum = scores
            .dtype()
            .finfo_min()
            .map_err(|cause| failure(Failure::ScoreDtype(cause)))?;
        ops::r#where(mask, &*scores, Array::try_from_f32(minimum as f32)?, stream)?
    } else {
        scores.add(mask.as_dtype(scores.dtype(), stream)?, stream)?
    };
    Ok(())
}
pub(super) fn run(input: Input<'_>, stream: &Stream) -> Result<Array, Exception> {
    let g = input.geometry().map_err(failure)?;
    let d = g.shape;
    let [
        queries,
        local_keys,
        local_values,
        pooled_keys,
        pooled_values,
        indices,
    ] = input.values;
    let pooled_keys = gather(pooled_keys, indices, g, d.key_dimensions, stream)?;
    let pooled_values = gather(pooled_values, indices, g, d.value_dimensions, stream)?;
    let queries = queries.multiply(Array::try_from_f32(input.scale)?, stream)?;
    let mut local_scores = local(
        &queries,
        local_keys,
        g,
        d.key_dimensions,
        d.local,
        true,
        stream,
    )?;
    let mut pooled_scores = selected(
        &queries,
        &pooled_keys,
        g,
        d.key_dimensions,
        d.selected,
        true,
        stream,
    )?;
    mask(&mut local_scores, input.local_mask, stream)?;
    mask(&mut pooled_scores, input.pooled_mask, stream)?;
    let scores = if let Some(sinks) = input.sinks {
        let sinks = sinks
            .as_dtype(local_scores.dtype(), stream)?
            .reshape(&[1, d.heads, 1, 1], stream)?;
        let sinks = ops::broadcast_to(&sinks, &[d.batch, d.heads, d.queries, 1], stream)?;
        ops::concatenate_axis(&[local_scores, pooled_scores, sinks], -1, stream)?
    } else {
        ops::concatenate_axis(&[local_scores, pooled_scores], -1, stream)?
    };
    let weights = ops::softmax_axis(scores, -1, true, stream)?;
    let local_weights = weights.try_slice(
        &[0; 4],
        &[d.batch, d.heads, d.queries, d.local],
        &[1; 4],
        stream,
    )?;
    let pooled_weights = weights.try_slice(
        &[0, 0, 0, d.local],
        &[d.batch, d.heads, d.queries, d.local + d.selected],
        &[1; 4],
        stream,
    )?;
    let local_context = local(
        &local_weights,
        local_values,
        g,
        d.local,
        d.value_dimensions,
        false,
        stream,
    )?;
    let pooled_context = selected(
        &pooled_weights,
        &pooled_values,
        g,
        d.selected,
        d.value_dimensions,
        false,
        stream,
    )?;
    local_context.add(pooled_context, stream)
}

pub(in crate::backend::nn) fn returned_handles(masks: usize, sinks: bool) -> Option<usize> {
    // Two gather helpers (5 each), query scalar/product, four contractions
    // (9 each), concat, softmax, two slices and final Add. Each mask returns
    // either scalar/Where or cast/Add; a sink returns cast/reshape/broadcast.
    53usize
        .checked_add(masks.checked_mul(2)?)?
        .checked_add(usize::from(sinks).checked_mul(3)?)
}
pub(in crate::backend::nn) fn control_bytes(masks: usize, sinks: bool) -> Option<usize> {
    let calls = returned_handles(masks, sinks)?;
    let sizes = [
        size_of::<IndexedAttentionInput<'_, MlxTensor>>() * 2,
        size_of::<Input<'_>>() * 2,
        size_of::<Geometry>() * 3,
        size_of::<IndexedAttentionGeometry>() * 2,
        size_of::<Result<Geometry, Failure>>(),
        size_of::<Result<IndexedAttentionGeometry, IndexedAttentionValidationError>>() * 2,
        size_of::<[&[i32]; 6]>() * 2,
        size_of::<Option<&[i32]>>(),
        size_of::<(&Array, &Array, Geometry, i32, i32, bool, &Stream)>() * 2,
        size_of::<(&Array, &Array, Geometry, i32, &Stream)>(),
        size_of::<(&mut Array, Option<&Array>, &Stream)>(),
        calls.checked_mul(size_of::<Array>())?,
        calls.checked_mul(size_of::<Result<Array, Exception>>())?,
        size_of::<[Array; 3]>(),
        size_of::<[Array; 2]>(),
        size_of::<[Result<MlxTensor, Error>; 2]>(),
        size_of::<Result<(), Error>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<Result<f64, InexactDtypeError>>(),
        // Actual fixed destinations across the shared helpers: local has
        // three rank-four arrays; selected has seven (both right-axis
        // alternatives); gather has two; run has sink two and slice six.
        size_of::<[i32; 4]>() * (3 + 7 + 2 + 8),
        size_of::<[i32; 4]>(), // geometry output return
        size_of::<f64>(),      // finite mask minimum
        size_of::<[i32; 3]>() * 4,
        size_of::<[i32; 8]>(),
        size_of::<[bool; 2]>(),
        size_of::<[f32; 3]>(),
        size_of::<Dtype>() * 2,
        size_of::<[&Array; 12]>(),
        size_of::<[&Stream; 5]>(),
        size_of::<Failure>(),
        size_of::<(Failure,)>(),
        size_of::<Result<Option<OriginalScopeObserver>, Exception>>(),
        OriginalScopeObserver::control_bytes()?,
        Exception::retained_source_control_bytes::<OriginalFailure>()?,
        Error::retained_source_construction_bytes::<Exception>()?,
        ops::concatenate_axis_control_bytes()?,
    ];
    sizes
        .into_iter()
        .try_fold(size_of_val(&sizes), usize::checked_add)
}
