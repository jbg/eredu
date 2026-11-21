//! MLX comparison kernels over position ranges admitted by the cache owner.

use std::ops::Range;

use eredu_core::{cache::mechanisms::PoolingCacheStorage, BackendFailure};
use safemlx::{error::Exception, Array, Stream};

use super::MlxCacheBackend;
use crate::MlxTensor;

impl PoolingCacheStorage for MlxCacheBackend {
    fn local_visibility(
        queries: Range<i32>,
        keys: Range<i32>,
        window: i32,
        queue: &Stream,
    ) -> Result<MlxTensor, BackendFailure> {
        (|| {
            let queries = Array::arange::<i32, i32>(Some(queries.start), queries.end, None, queue)?
                .reshape(&[-1, 1], queue)?;
            let keys = Array::arange::<i32, i32>(Some(keys.start), keys.end, None, queue)?
                .reshape(&[1, -1], queue)?;
            let causal = queries.ge(&keys, queue)?;
            let recent = keys.gt(queries.subtract(Array::from_int(window), queue)?, queue)?;
            Ok::<_, Exception>(MlxTensor::from_array(causal.logical_and(&recent, queue)?))
        })()
        .map_err(BackendFailure::from_error)
    }

    fn pooled_visibility(
        query_ends: Range<i32>,
        pooled_tokens: i32,
        ratio: i32,
        queue: &Stream,
    ) -> Result<Array, BackendFailure> {
        (|| {
            let pooled = Array::arange::<i32, i32>(Some(0), pooled_tokens, None, queue)?
                .reshape(&[1, pooled_tokens], queue)?;
            let visible =
                Array::arange::<i32, i32>(Some(query_ends.start), query_ends.end, None, queue)?
                    .floor_divide(Array::from_int(ratio), queue)?
                    .reshape(&[-1, 1], queue)?;
            pooled.lt(visible, queue)
        })()
        .map_err(BackendFailure::from_error)
    }
}
