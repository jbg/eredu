//! Native profile gathering shared by contiguous and blockwise attention.
use crate::MlxTensor;
use eredu_nn::RelativeAttentionInput;
use safemlx::{
    error::Exception,
    ops::{
        arange, broadcast_to, clip,
        indexing::{take_along_axis, TryIndexOp},
        maximum, r#where,
    },
    Array, Dtype, Stream,
};

pub(crate) struct RelativeAttentionKernel {
    pub queries: Array,
    profiles: Array,
    query_start: i32,
    scaling: Option<Array>,
}

impl RelativeAttentionKernel {
    pub fn new(
        input: &RelativeAttentionInput<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        input
            .validate()
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mut queries = input.queries.as_array().clone();
        let query_len = queries.dim(2);
        let end = input
            .query_offset
            .checked_add(query_len)
            .ok_or_else(|| Exception::custom("relative query position overflow"))?;
        if input.query_offset < 0 {
            return Err(Exception::custom("negative relative query position"));
        }
        let scaling = if input.window.is_none() {
            input
                .log_scaling_floor
                .map(|floor| {
                    // The largest admitted position plus one still fits I32;
                    // its exclusive arange endpoint need not fit after shifting.
                    let positions = arange::<i32, i32>(input.query_offset, end, 1, stream)?
                        .add(Array::try_from_int(1)?, stream)?;
                    let ratio = positions
                        .as_dtype(Dtype::Float32, stream)?
                        .divide(Array::try_from_f32(floor as f32)?, stream)?;
                    let ratio = maximum(ratio, Array::try_from_f32(1.0)?, stream)?;
                    let tau = ratio
                        .log(stream)?
                        .multiply(Array::try_from_f32(input.log_scaling_alpha)?, stream)?;
                    tau.add(Array::try_from_f32(1.0)?, stream)?
                        .reshape(&[1, 1, query_len, 1], stream)
                })
                .transpose()?
        } else {
            None
        };
        if let Some(scale) = &scaling {
            queries = queries.multiply(scale, stream)?;
        }
        Ok(Self {
            queries,
            profiles: input.profiles.as_array().clone(),
            query_start: input.query_offset,
            scaling,
        })
    }

    /// Materializes only the current block's score bias. Historical keys remain
    /// cache-owned and no complete-context profile tensor is constructed.
    pub fn bias(&self, start: i64, end: i64, stream: &Stream) -> Result<Array, Exception> {
        use safemlx::ops::indexing::NewAxis;
        let start = i32::try_from(start)
            .map_err(|_| Exception::custom("relative block position overflow"))?;
        let end = i32::try_from(end)
            .map_err(|_| Exception::custom("relative block position overflow"))?;
        if start < 0 || end <= start {
            return Err(Exception::custom("invalid relative block range"));
        }
        let query_len = self.queries.dim(2);
        let query = arange::<i32, i32>(self.query_start, self.query_start + query_len, 1, stream)?
            .try_index_device((.., NewAxis), stream)?;
        let key =
            arange::<i32, i32>(start, end, 1, stream)?.try_index_device((NewAxis, ..), stream)?;
        let distances = query.subtract(key, stream)?;
        let extent = self.profiles.dim(3);
        let gather = clip(&distances, (0, extent - 1), stream)?
            .as_dtype(Dtype::Int32, stream)?
            .try_index_device((NewAxis, NewAxis, .., ..), stream)?;
        let gather = broadcast_to(
            &gather,
            &[
                self.queries.dim(0),
                self.queries.dim(1),
                query_len,
                end - start,
            ],
            stream,
        )?;
        let bias = take_along_axis(&self.profiles, &gather, -1, stream)?;
        let valid = distances
            .ge(Array::try_from_int(0)?, stream)?
            .logical_and(&distances.lt(Array::try_from_int(extent)?, stream)?, stream)?;
        let bias = r#where(&valid, bias, Array::try_from_f32(0.0)?, stream)?;
        match &self.scaling {
            Some(scale) => bias.multiply(scale, stream),
            None => Ok(bias),
        }
    }

    pub fn prepared_bytes(&self) -> u64 {
        // Only newly scaled queries/tau are additional to the caller's inputs.
        self.scaling
            .as_ref()
            .map_or(0, |scale| (scale.nbytes() + self.queries.nbytes()) as u64)
    }
}

/// Owning safe-wrapper transports in the existing constructor and bias worker.
/// Each count follows an actual call or immutable clone; nested primitive
/// descriptors are independently counted by the source recipe.
pub(crate) fn returned_handles(repeated: bool, scaled: bool, windowed: bool) -> Option<usize> {
    // Outer coordinates6 + scalar1, kernel query/profile clones2,
    // bias calls15 + scalars5, final attention calls7 + scalars2.
    38usize.checked_add(if repeated { 6 } else { 2 })?
        .checked_add(16 * usize::from(scaled))?
        .checked_add(3 * usize::from(windowed))
}

pub(crate) fn control_bytes(repeated: bool, scaled: bool, windowed: bool) -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    use eredu_nn::Error;
    let calls = returned_handles(repeated, scaled, windowed)?;
    let frames = [
        size_of::<RelativeAttentionInput<'_, MlxTensor>>() * 2,
        size_of::<RelativeAttentionKernel>() * 2,
        size_of::<Result<RelativeAttentionKernel, Exception>>(),
        size_of::<(&RelativeAttentionInput<'_, MlxTensor>, &Stream)>(),
        size_of::<(&RelativeAttentionKernel, i64, i64, &Stream)>(),
        size_of::<(&Array, &Stream, i32, i32, i32, i32, i32)>(),
        size_of::<(&RelativeAttentionInput<'_, MlxTensor>, &Stream, i32, i32)>(),
        size_of::<[&[i32]; 4]>(),
        size_of::<[i32; 5]>() * 2,
        size_of::<[i32; 4]>() * 4,
        size_of::<[i32; 2]>() * 4,
        size_of::<[i32; 10]>(),
        size_of::<[&Array; 8]>(),
        size_of::<[&Stream; 4]>(),
        size_of::<Option<Array>>() * 2,
        size_of::<Option<Result<Array, Exception>>>(),
        size_of::<Result<Option<Array>, Exception>>(),
        size_of::<Result<MlxTensor, Error>>(),
        size_of::<Result<(), Error>>() * 2,
        size_of::<Result<(), Exception>>(),
        calls.checked_mul(size_of::<Array>())?,
        calls.checked_mul(size_of::<Result<Array, Exception>>())?,
        calls.checked_mul(size_of::<Result<Array, Error>>())?,
        safemlx::Array::as_dtype_control_bytes()?,
        crate::tensor::clip::control_bytes()?,
        safemlx::OriginalScopeObserver::control_bytes()?,
    ];
    frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
}
