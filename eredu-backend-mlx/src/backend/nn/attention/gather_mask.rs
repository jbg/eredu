//! The existing query-specific mask gather and its fixed original controls.
use crate::MlxTensor;
use eredu_nn::Error;
use safemlx::{Array, OriginalScopeObserver, Stream, error::Exception};
use std::mem::{size_of, size_of_val};

#[derive(Clone, Copy, Debug)]
pub(in crate::backend::nn) struct Geometry {
    pub expanded: [i32; 3],
    pub output: [i32; 4],
}
#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("pooled mask gathering expects rank 2 and rank 3, got {mask_rank} and {positions_rank}")]
pub(in crate::backend::nn) struct GeometryError {
    mask_rank: usize,
    positions_rank: usize,
}
impl Geometry {
    pub(in crate::backend::nn) fn new(
        mask: &[i32],
        positions: &[i32],
    ) -> Result<Self, GeometryError> {
        if mask.len() != 2 || positions.len() != 3 {
            return Err(GeometryError {
                mask_rank: mask.len(),
                positions_rank: positions.len(),
            });
        }
        Ok(Self {
            expanded: [positions[0], positions[1], mask[1]],
            output: [positions[0], 1, positions[1], positions[2]],
        })
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct OriginalFailure {
    #[source]
    cause: GeometryError,
    _custody: Exception,
}
fn native_error(cause: Exception) -> Error {
    match OriginalScopeObserver::try_current() {
        Ok(None) => Error::backend_retained_source(cause),
        _ => Error::backend_retained_source(cause),
    }
}
fn geometry_error(cause: GeometryError, mask: &Array, positions: &Array) -> Error {
    let custody = match OriginalScopeObserver::try_current() {
        Ok(Some(scope)) => scope.invalid_input_error(),
        Err(cause) => cause,
        Ok(None) => {
            return Error::backend(format!(
                "pooled mask gathering expects [query, pool] and [batch, query, selected], got {:?} and {:?}",
                mask.shape(),
                positions.shape()
            ));
        }
    };
    Error::backend_retained_source(Exception::from_retained_source(OriginalFailure {
        cause,
        _custody: custody,
    }))
}
pub(in crate::backend::nn) fn run(
    mask: &MlxTensor,
    positions: &MlxTensor,
    stream: &Stream,
) -> Result<MlxTensor, Error> {
    let mask = mask.as_array();
    let positions = positions.as_array();
    let geometry = Geometry::new(mask.shape(), positions.shape())
        .map_err(|cause| geometry_error(cause, mask, positions))?;
    // Same native worker and validation order as the ordinary backend method:
    // ExpandDims, Broadcast, GatherAxis, ExpandDims. Explicit axis keeps the
    // safe wrapper's input as a borrowed Cow, without a flattening owner.
    let expanded = mask.expand_dims(0, stream).map_err(native_error)?;
    let expanded =
        safemlx::ops::broadcast_to(&expanded, &geometry.expanded, stream).map_err(native_error)?;
    let selected = safemlx::ops::indexing::take_along_axis(&expanded, positions, 2, stream)
        .map_err(native_error)?;
    selected
        .expand_dims(1, stream)
        .map(MlxTensor::from_array)
        .map_err(native_error)
}
pub(in crate::backend::nn) fn control_bytes() -> Option<usize> {
    let sizes = [
        size_of::<(&MlxTensor, &MlxTensor, &Stream)>(), // adapter arguments
        size_of::<(&MlxTensor, &MlxTensor, &Stream)>(), // worker arguments
        size_of::<(&Array, &Array)>(),
        size_of::<Geometry>(),
        size_of::<Result<Geometry, GeometryError>>(),
        size_of::<[Array; 4]>(),
        size_of::<[Result<Array, Exception>; 4]>(),
        size_of::<[Result<Array, Error>; 3]>(),
        size_of::<[Result<MlxTensor, Error>; 2]>(),
        size_of::<std::borrow::Cow<'_, Array>>(),
        size_of::<Option<i32>>(),
        size_of::<[i32; 3]>(),                  // expand/gather axes
        size_of::<(&Array, &Array, &Stream)>(), // gather wrapper loans
        size_of::<(&Array, &[i32], &Stream)>(), // broadcast wrapper loans
        size_of::<(&Array, &Stream)>(),         // expand wrapper loans
        size_of::<GeometryError>(),
        size_of::<(GeometryError, &Array, &Array)>(),
        size_of::<Result<Option<OriginalScopeObserver>, Exception>>(),
        OriginalScopeObserver::control_bytes()?,
        Exception::retained_source_control_bytes::<OriginalFailure>()?,
        Error::retained_source_construction_bytes::<Exception>()?,
    ];
    sizes
        .into_iter()
        .try_fold(size_of_val(&sizes), usize::checked_add)
}
