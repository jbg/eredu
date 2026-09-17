//! Selected neutral placement checked against actual projected mechanisms.
use super::*;
use crate::working_memory::WorkspacePagedGeometry;
use crate::{
    CacheResidencyPolicy, RuntimeState, SelectedStateRealization, StateComponentPlacement,
};
use std::mem::size_of;

#[derive(Debug, thiserror::Error)]
enum ProjectionSelectionError {
    #[error("workspace projection differs from the selected state layout")]
    Layout,
    #[error("workspace state mechanism differs from selected placement at layer {0}")]
    Placement(usize),
    #[error("workspace paged geometry differs from selected policy at layer {0}")]
    Geometry(usize),
}

/// Revalidates an actual symbolic state against retained neutral selection.
/// It allocates no state or discovery graph and supplies no native source,
/// materialization, promotion, copy or execution authority. Those owners stay
/// with the backend's complete projection and original source bank.
pub fn validate_workspace_state_realization(
    state: &DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
    selected: &SelectedStateRealization,
    context: &WorkspaceContext,
) -> Result<(), Error> {
    let frames = [
        size_of::<(
            &DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
            &SelectedStateRealization,
            &WorkspaceContext,
        )>(),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_, WorkspaceResidentLayerState>>>(),
        size_of::<std::slice::Iter<'_, crate::SelectedStateComponentRealization>>(),
        size_of::<(usize, bool, Option<WorkspacePagedGeometry>)>(),
        size_of::<ProjectionSelectionError>(),
        size_of::<Result<(), Error>>(),
    ];
    context.charge_metadata(
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
    )?;
    let fail = |cause| context.metadata_source(cause);
    if state.layout() != selected.layout() || state.as_ref().len() != selected.layout().len() {
        return Err(fail(ProjectionSelectionError::Layout));
    }
    for (index, layer) in state.as_ref().iter().enumerate() {
        layer.validate_workspace_context(context)?;
        let paged = match layer {
            WorkspaceResidentLayerState::Paged(state) => Some(state.geometry()),
            WorkspaceResidentLayerState::Pooling(state) => state.paged_geometry(),
            WorkspaceResidentLayerState::Ordinary(_)
            | WorkspaceResidentLayerState::Compressed(_) => None,
        };
        let selected_paged = selected.components().iter().any(|component| {
            component.layer() == index && component.placement() == StateComponentPlacement::Paged
        });
        if paged.is_some() != selected_paged {
            return Err(fail(ProjectionSelectionError::Placement(index)));
        }
        if let Some(geometry) = paged {
            let CacheResidencyPolicy::Paged(options) = selected.policy() else {
                return Err(fail(ProjectionSelectionError::Placement(index)));
            };
            if geometry.block_size != options.block_size_tokens()
                || geometry.retain_discarded != options.retains_discarded_for_persistence()
            {
                return Err(fail(ProjectionSelectionError::Geometry(index)));
            }
        }
    }
    Ok(())
}
