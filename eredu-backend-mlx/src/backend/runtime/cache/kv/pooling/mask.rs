//! The same ordinary integer-coordinate mask, with original error custody.
use super::*;
use eredu_nn::operation_geometry::{PoolingMaskGeometry, PoolingMaskGeometryError};
use safemlx::OriginalScopeObserver;
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
#[error("pooling attention cache has no stream {0}")]
struct MissingStream(u32);
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct OriginalFailure<E: std::error::Error + 'static> {
    #[source]
    cause: E,
    _custody: Exception,
}
fn original<E: std::error::Error + Send + Sync + 'static>(
    cause: E,
    custody: Exception,
) -> Exception {
    Exception::from_retained_source(OriginalFailure {
        cause,
        _custody: custody,
    })
}
fn geometry_error(cause: PoolingMaskGeometryError) -> Exception {
    match OriginalScopeObserver::try_current() {
        Ok(Some(scope)) => original(cause, scope.invalid_input_error()),
        Err(custody) => original(cause, custody),
        // Geometry::new used this exact ordinary neural diagnostic adapter.
        Ok(None) => Exception::custom(ComputeError::backend(cause).to_string()),
    }
}
pub(super) fn neural_error(cause: Exception) -> ComputeError {
    match OriginalScopeObserver::try_current() {
        Ok(None) => ComputeError::backend_retained_source(cause),
        _ => ComputeError::backend_retained_source(cause),
    }
}
pub(super) fn missing_stream(stream: u32) -> ComputeError {
    match OriginalScopeObserver::try_current() {
        Ok(Some(scope)) => ComputeError::backend_retained_source(original(
            MissingStream(stream),
            scope.invalid_input_error(),
        )),
        Err(custody) => {
            ComputeError::backend_retained_source(original(MissingStream(stream), custody))
        }
        Ok(None) => {
            ComputeError::backend(format!("pooling attention cache has no stream {stream}"))
        }
    }
}
pub(super) fn make(
    queries: i32,
    pooled: i32,
    offset: i32,
    ratio: i32,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    let geometry =
        PoolingMaskGeometry::new_fixed(queries, pooled, offset, ratio).map_err(geometry_error)?;
    if geometry.pooled() == 0 || geometry.queries() == 1 {
        return Ok(None);
    }
    let pooled = Array::arange::<i32, i32>(Some(0), geometry.pooled(), None, stream)?;
    let visible = Array::arange::<i32, i32>(
        Some(geometry.offset() + 1),
        geometry.offset() + geometry.queries() + 1,
        None,
        stream,
    )?
    .floor_divide(Array::try_from_int(geometry.ratio())?, stream)?
    .reshape(&[geometry.queries(), 1], stream)?;
    Ok(Some(
        pooled
            .reshape(&[1, geometry.pooled()], stream)?
            .lt(visible, stream)?,
    ))
}
pub(super) fn control_bytes() -> Option<usize> {
    let sizes = [
        size_of::<PoolingMaskGeometry>(),
        size_of::<[i32; 6]>(),
        size_of::<[i32; 2]>() * 2,
        // Actual arange generic input and converted native scalar arguments.
        size_of::<[Option<i32>; 2]>(),
        size_of::<[f64; 3]>(),
        size_of::<safemlx::Dtype>(),
        // Two aranges, the eager I32 ratio, Divide, two reshapes and Less.
        size_of::<[Array; 7]>(),
        size_of::<[Result<Array, Exception>; 7]>(),
        size_of::<Option<Array>>(),
        size_of::<Option<MlxTensor>>(),
        // Worker return and the cache forwarder's actual returned transport.
        size_of::<Result<Option<Array>, Exception>>(),
        size_of::<Result<Option<Array>, Exception>>(),
        size_of::<Result<Option<MlxTensor>, ComputeError>>(),
        size_of::<Result<&PoolingCache, ComputeError>>(),
        size_of::<Option<OriginalScopeObserver>>(),
        size_of::<Result<Option<OriginalScopeObserver>, Exception>>(),
        size_of::<Result<PoolingMaskGeometry, PoolingMaskGeometryError>>(),
        size_of::<&PoolingCache>(),
        size_of::<&Stream>(),
        size_of::<MissingStream>(),
        OriginalScopeObserver::control_bytes()?,
        ComputeError::retained_source_construction_bytes::<Exception>()?,
        Exception::retained_source_control_bytes::<OriginalFailure<PoolingMaskGeometryError>>()?,
        Exception::retained_source_control_bytes::<OriginalFailure<MissingStream>>()?,
    ];
    sizes
        .into_iter()
        .try_fold(size_of_val(&sizes), usize::checked_add)
}
