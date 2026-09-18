//! Paid local column identities retain the original compact mask semantics.
use super::*;
use eredu_core::component::{ComponentCoordinateMap, ComponentIndexProjectionPlan};
impl PreparedWindowInterventionPayload {
    /// Localize an admitted complete-column action through the actual map.
    /// This is a host source constructor; callers still validate global/local
    /// slices, original plan identity and the real native invocation.
    pub fn prepare_columns(action: &InterventionAction, coordinates: &ComponentCoordinateMap,
        funding: HostMetadataFunding) -> Result<Self, PreparedWindowInterventionPayloadError> {
        let result: Result<InterventionAction, Cause> = (|| {
            funding.reserve_metadata(Self::column_control_bytes().ok_or(WindowInterventionPayloadError::Overflow)?)?;
            let indices = match action {
                InterventionAction::MaskComponents { indices, .. } => indices.as_slice(),
                InterventionAction::MaskLogits { token_ids, .. } => token_ids.as_slice(),
                _ => return Err(WindowInterventionPayloadError::Destination.into()),
            };
            let mut scratch = funding.metadata_vec(indices.len())?;
            scratch.resize(indices.len(), (0u32, 0usize));
            let source = ComponentIndexProjectionPlan::prepare(coordinates, indices, &mut scratch)?;
            let mut local = funding.metadata_vec(source.local_count())?;
            local.resize(source.local_count(), 0u32);
            source.write(&mut local)?;
            Ok(match action {
                InterventionAction::MaskComponents { dtype, keep_selected, .. } => InterventionAction::MaskComponents {
                    dtype: *dtype, keep_selected: *keep_selected, indices: local },
                InterventionAction::MaskLogits { dtype, .. } => InterventionAction::MaskLogits { dtype: *dtype, token_ids: local },
                _ => unreachable!("validated compact mask source"),
            })
        })();
        match result {
            Ok(action) => Ok(Self { action: Some(action), _funding: funding }),
            Err(cause) => Err(PreparedWindowInterventionPayloadError { cause, _funding: funding }),
        }
    }
    /// Fixed source worker controls; scratch and output vectors use actual
    /// metadata_vec layouts from the retained source lengths before allocation.
    pub fn column_control_bytes() -> Option<usize> {
        let frames = [Self::control_bytes()?, ComponentIndexProjectionPlan::control_bytes()?,
            size_of::<(&InterventionAction, &ComponentCoordinateMap, HostMetadataFunding)>(),
            size_of::<Result<InterventionAction, Cause>>(), size_of::<Vec<(u32, usize)>>(), size_of::<Vec<u32>>(),
            size_of::<Result<Vec<(u32, usize)>, eredu_nn::Error>>(), size_of::<Result<Vec<u32>, eredu_nn::Error>>()];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
