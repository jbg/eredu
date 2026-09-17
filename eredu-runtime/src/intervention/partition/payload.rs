//! One row-major payload projection, shared by ordinary and prepaid windows.
use super::*;
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use std::mem::{size_of, size_of_val};
mod columns;

/// Fixed geometry refusal before indexing an immutable admitted payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum WindowInterventionPayloadError {
    /// The normalized destination or source payload extent differs.
    #[error("partition intervention payload destination is outside admission")]
    Destination,
    /// Exact element or offset arithmetic overflowed.
    #[error("partition intervention payload arithmetic overflowed")]
    Overflow,
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Geometry(#[from] WindowInterventionPayloadError),
    #[error(transparent)]
    Funding(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)]
    Metadata(#[from] eredu_nn::Error),
    #[error(transparent)]
    Columns(#[from] eredu_core::component::ComponentIndexProjectionError),
}
/// Any partial payload and metadata failure retire before the actual H account.
#[derive(Debug, thiserror::Error)]
#[error("prepared intervention window payload failed: {cause}")]
pub struct PreparedWindowInterventionPayloadError {
    #[source]
    cause: Cause,
    _funding: WorkspaceMetadataFunding,
}
/// Immutable local payload storage; this carries no source, phase or native grant.
/// Nonpayload actions continue to be borrowed from the enclosing immutable C.
#[derive(Debug)]
pub struct PreparedWindowInterventionPayload {
    action: Option<InterventionAction>,
    _funding: WorkspaceMetadataFunding,
}
impl PreparedWindowInterventionPayload {
    /// Copy only the selected payload values using the ordinary projection worker.
    /// Every vector is charged to this account before construction. The caller
    /// separately binds C, logical destination, physical slice and operation.
    pub fn prepare(
        action: &InterventionAction,
        destination: &ResolvedCaptureSlice,
        funding: WorkspaceMetadataFunding,
    ) -> Result<Self, PreparedWindowInterventionPayloadError> {
        let result: Result<Option<InterventionAction>, Cause> = (|| {
            funding.reserve_metadata(
                Self::control_bytes().ok_or(WindowInterventionPayloadError::Overflow)?,
            )?;
            project(action, destination, &mut Paid(&funding))
        })();
        match result {
            Ok(action) => Ok(Self {
                action,
                _funding: funding,
            }),
            Err(cause) => Err(PreparedWindowInterventionPayloadError {
                cause,
                _funding: funding,
            }),
        }
    }
    /// The selected action when it owns per-element payload. Otherwise use the
    /// original immutable action; component-ID inventories are never copied here.
    pub fn projected_action(&self) -> Option<&InterventionAction> {
        self.action.as_ref()
    }
    /// Fixed wrapper, shared projection and row-major traversal controls. Vector
    /// backing and allocation/refusal transports use the actual metadata_vec query.
    pub fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<PreparedWindowInterventionPayloadError>(),
            size_of::<Cause>(),
            size_of::<WindowInterventionPayloadError>(),
            size_of::<WorkspaceMetadataFundingError>(),
            size_of::<WorkspaceMetadataFunding>(),
            size_of::<Paid<'_>>(),
            size_of::<Result<Self, PreparedWindowInterventionPayloadError>>(),
            size_of::<Result<Option<InterventionAction>, Cause>>(),
            size_of::<(
                &InterventionAction,
                &ResolvedCaptureSlice,
                &WorkspaceMetadataFunding,
            )>(),
            size_of::<(&InterventionAction, &ResolvedCaptureSlice, &mut Paid<'_>)>(),
            size_of::<Option<InterventionAction>>(),
            size_of::<InterventionTensor>(),
            size_of::<InterventionValues>(),
            size_of::<Vec<u64>>(),
            size_of::<Result<Vec<u64>, Cause>>(),
            size_of::<(
                &InterventionAction,
                &ResolvedCaptureSlice,
                &WorkspaceMetadataFunding,
            )>(),
            size_of::<Result<(), WorkspaceMetadataFundingError>>(),
            size_of::<Option<usize>>(),
            size_of::<usize>(),
            copy_controls::<f32>(),
            copy_controls::<u16>(),
            copy_controls::<bool>(),
            size_of::<(&[u64], &ResolvedCaptureSlice, usize)>(),
            size_of::<[u64; 5]>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<Result<usize, WindowInterventionPayloadError>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
fn copy_controls<T>() -> usize {
    size_of::<(&[T], &[u64], &ResolvedCaptureSlice, &mut Paid<'_>)>()
        + size_of::<(&[u64], &mut Paid<'_>)>()
        + size_of::<Vec<T>>()
        + size_of::<Result<Vec<T>, Cause>>()
        + size_of::<[u64; 6]>()
        + size_of::<usize>()
        + size_of::<std::ops::Range<u64>>()
        + size_of::<std::iter::Rev<std::ops::Range<usize>>>()
        + size_of::<Option<&T>>()
        + size_of::<T>()
}
pub(super) trait Allocation {
    type Error: From<WindowInterventionPayloadError>;
    fn vector<T>(&mut self, capacity: usize) -> Result<Vec<T>, Self::Error>;
}
pub(super) struct Ordinary;
impl Allocation for Ordinary {
    type Error = CaptureError;
    fn vector<T>(&mut self, capacity: usize) -> Result<Vec<T>, CaptureError> {
        Ok(Vec::with_capacity(capacity))
    }
}
impl From<WindowInterventionPayloadError> for CaptureError {
    fn from(cause: WindowInterventionPayloadError) -> Self {
        match cause {
            WindowInterventionPayloadError::Destination => {
                invalid("partition intervention payload destination is outside admission")
            }
            WindowInterventionPayloadError::Overflow => Self::Overflow,
        }
    }
}
struct Paid<'a>(&'a WorkspaceMetadataFunding);
impl Allocation for Paid<'_> {
    type Error = Cause;
    fn vector<T>(&mut self, capacity: usize) -> Result<Vec<T>, Cause> {
        self.0.metadata_vec(capacity).map_err(Cause::Metadata)
    }
}
pub(super) fn project<A: Allocation>(
    action: &InterventionAction,
    destination: &ResolvedCaptureSlice,
    allocation: &mut A,
) -> Result<Option<InterventionAction>, A::Error> {
    Ok(Some(match action {
        InterventionAction::Mask { dtype, shape, keep } => {
            validate(shape, destination, keep.len())?;
            InterventionAction::Mask {
                dtype: *dtype,
                shape: copy_shape(&destination.shape, allocation)?,
                keep: select_payload(keep, shape, destination, allocation)?,
            }
        }
        InterventionAction::Replace { tensor } | InterventionAction::Add { tensor } => {
            let values =
                match &tensor.values {
                    InterventionValues::Float32(values) => InterventionValues::Float32(
                        select_payload(values, &tensor.shape, destination, allocation)?,
                    ),
                    InterventionValues::Float16(values) => InterventionValues::Float16(
                        select_payload(values, &tensor.shape, destination, allocation)?,
                    ),
                    InterventionValues::Bfloat16(values) => InterventionValues::Bfloat16(
                        select_payload(values, &tensor.shape, destination, allocation)?,
                    ),
                };
            let tensor = InterventionTensor {
                shape: copy_shape(&destination.shape, allocation)?,
                values,
            };
            if matches!(action, InterventionAction::Replace { .. }) {
                InterventionAction::Replace { tensor }
            } else {
                InterventionAction::Add { tensor }
            }
        }
        _ => return Ok(None),
    }))
}
fn copy_shape<A: Allocation>(shape: &[u64], allocation: &mut A) -> Result<Vec<u64>, A::Error> {
    let mut output = allocation.vector(shape.len())?;
    output.extend_from_slice(shape);
    Ok(output)
}
fn validate(
    shape: &[u64],
    slice: &ResolvedCaptureSlice,
    length: usize,
) -> Result<usize, WindowInterventionPayloadError> {
    use WindowInterventionPayloadError::{Destination, Overflow};
    let rank = shape.len();
    if [
        slice.starts.len(),
        slice.ends.len(),
        slice.strides.len(),
        slice.shape.len(),
    ]
    .into_iter()
    .any(|len| len != rank)
    {
        return Err(Destination);
    }
    let mut source = 1u64;
    let mut selected = 1u64;
    for axis in 0..rank {
        let (start, end, stride, count) = (
            slice.starts[axis],
            slice.ends[axis],
            slice.strides[axis],
            slice.shape[axis],
        );
        if start > end
            || end > shape[axis]
            || stride == 0
            || count != (end - start).div_ceil(stride)
        {
            return Err(Destination);
        }
        source = source.checked_mul(shape[axis]).ok_or(Overflow)?;
        selected = selected.checked_mul(count).ok_or(Overflow)?;
    }
    if usize::try_from(source).map_err(|_| Overflow)? != length {
        return Err(Destination);
    }
    usize::try_from(selected).map_err(|_| Overflow)
}
// Iterate selected elements only; no global index vector or half-float conversion.
fn select_payload<T: Copy, A: Allocation>(
    values: &[T],
    shape: &[u64],
    slice: &ResolvedCaptureSlice,
    allocation: &mut A,
) -> Result<Vec<T>, A::Error> {
    use WindowInterventionPayloadError::{Destination, Overflow};
    let count = validate(shape, slice, values.len())?;
    let mut output = allocation.vector(count)?;
    for ordinal in 0..u64::try_from(count).map_err(|_| Overflow)? {
        let mut remaining = ordinal;
        let mut source = 0u64;
        let mut stride = 1u64;
        for axis in (0..shape.len()).rev() {
            let coordinate = slice.starts[axis]
                .checked_add(
                    (remaining % slice.shape[axis])
                        .checked_mul(slice.strides[axis])
                        .ok_or(Overflow)?,
                )
                .ok_or(Overflow)?;
            remaining /= slice.shape[axis];
            source = source
                .checked_add(coordinate.checked_mul(stride).ok_or(Overflow)?)
                .ok_or(Overflow)?;
            stride = stride.checked_mul(shape[axis]).ok_or(Overflow)?;
        }
        output.push(
            *values
                .get(usize::try_from(source).map_err(|_| Overflow)?)
                .ok_or(Destination)?,
        );
    }
    Ok(output)
}
#[cfg(test)]
mod tests;
