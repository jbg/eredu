//! Eager mask materialization for admitted pooling visibility.

use super::*;
use eredu_core::cache::mechanisms::PoolingCacheStorage;
use eredu_nn::attention_mask::{LocalWindowVisibility, PooledWindowVisibility};

impl PoolingCacheStorage for CandleCacheStorage {
    fn local_visibility(
        geometry: LocalWindowVisibility,
        queue: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        visibility(geometry.shape(), queue, |row, column| {
            geometry.allows(row, column)
        })
    }

    fn pooled_visibility(
        geometry: PooledWindowVisibility,
        queue: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        visibility(geometry.shape(), queue, |row, column| {
            geometry.allows(row, column)
        })
    }
}

fn visibility(
    [rows, columns]: [i32; 2],
    queue: &CandleContext,
    visible: impl Fn(i32, i32) -> bool,
) -> Result<CandleTensor, BackendFailure> {
    let shape = (rows as usize, columns as usize);
    let mut mask = Vec::new();
    mask.try_reserve_exact(shape.0 * shape.1)
        .map_err(|error| BackendFailure::new(BackendFailureKind::ResourceExhausted, error))?;
    for row in 0..rows {
        mask.extend((0..columns).map(|column| u8::from(visible(row, column))));
    }
    let mut result = CandleTensor::wrap(NativeTensor::from_vec(mask, shape, &queue.device))
        .map_err(BackendFailure::from_error)?;
    result.boolean = true;
    Ok(result)
}
