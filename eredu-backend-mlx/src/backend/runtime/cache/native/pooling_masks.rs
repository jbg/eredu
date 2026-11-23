//! MLX comparison kernels over admitted pooling visibility.

use eredu_core::{cache::mechanisms::PoolingCacheStorage, BackendFailure};
use eredu_nn::attention_mask::{LocalWindowVisibility, PooledWindowVisibility};
use safemlx::{error::Exception, Array, Stream};

use super::MlxCacheBackend;
use crate::MlxTensor;

impl PoolingCacheStorage for MlxCacheBackend {
    fn local_visibility(
        geometry: LocalWindowVisibility,
        queue: &Stream,
    ) -> Result<MlxTensor, BackendFailure> {
        (|| {
            let queries = geometry.queries();
            let keys = geometry.keys();
            let [rows, columns] = geometry.shape();
            let queries = Array::arange::<i32, i32>(Some(queries.start), queries.end, None, queue)?
                .reshape(&[rows, 1], queue)?;
            let keys = Array::arange::<i32, i32>(Some(keys.start), keys.end, None, queue)?
                .reshape(&[1, columns], queue)?;
            let causal = queries.ge(&keys, queue)?;
            let earliest = queries.subtract(Array::from_int(geometry.max_past()), queue)?;
            let recent = keys.ge(earliest, queue)?;
            Ok::<_, Exception>(MlxTensor::from_array(causal.logical_and(&recent, queue)?))
        })()
        .map_err(BackendFailure::from_error)
    }

    fn pooled_visibility(
        geometry: PooledWindowVisibility,
        queue: &Stream,
    ) -> Result<Array, BackendFailure> {
        (|| {
            let queries = geometry.queries();
            let [rows, columns] = geometry.shape();
            let pooled = Array::arange::<i32, i32>(Some(0), columns, None, queue)?
                .reshape(&[1, columns], queue)?;
            let visible = Array::arange::<i32, i32>(Some(queries.start), queries.end, None, queue)?
                .add(Array::from_int(1), queue)?
                .floor_divide(Array::from_int(geometry.ratio()), queue)?
                .reshape(&[rows, 1], queue)?;
            pooled.lt(visible, queue)
        })()
        .map_err(BackendFailure::from_error)
    }
}
