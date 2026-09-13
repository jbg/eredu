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
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Exception::custom("relative query position overflow"))?;
        if input.query_offset < 0 {
            return Err(Exception::custom("negative relative query position"));
        }
        let scaling = if input.window.is_none() {
            input
                .log_scaling_floor
                .map(|floor| {
                    let positions = arange::<i32, i32>(input.query_offset + 1, end, 1, stream)?;
                    let ratio = positions
                        .as_dtype(Dtype::Float32, stream)?
                        .divide(Array::from_f32(floor as f32), stream)?;
                    let ratio = maximum(ratio, Array::from_f32(1.0), stream)?;
                    let tau = ratio
                        .log(stream)?
                        .multiply(Array::from_f32(input.log_scaling_alpha), stream)?;
                    tau.add(Array::from_f32(1.0), stream)?
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
            .ge(Array::from_int(0), stream)?
            .logical_and(&distances.lt(Array::from_int(extent), stream)?, stream)?;
        let bias = r#where(&valid, bias, Array::from_f32(0.0), stream)?;
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
