//! Actual complete paged source and independent copied-state metadata.
use super::super::workspace::ProjectedDenseResidentKvCopy;
use super::*;
use eredu_nn::workspace::{WorkspaceBackend, WorkspaceTensor};
use eredu_runtime::{DeviceState, RuntimeLayerState, working_memory::WorkspaceResidentLayerState};
use std::num::NonZeroU32;

impl PreparedPagedKvCopy<'_> {
    pub(crate) fn project_dense_workspace(
        &self,
        batch: NonZeroU32,
        context: &WorkspaceContext,
    ) -> Result<ProjectedDenseResidentKvCopy, Error> {
        context
            .charge_metadata(std::mem::size_of::<(
                &Self,
                NonZeroU32,
                &WorkspaceContext,
                ProjectedDenseResidentKvCopy,
                Result<ProjectedDenseResidentKvCopy, Error>,
                Vec<WorkspaceTensor>,
                &[WorkspaceResidentLayerState],
                usize,
                Option<usize>,
            )>())
            .map_err(|cause| Error::Neural(cause.into()))?;
        let source = self
            .source
            .project_complete_workspace_with_storage(batch, context)
            .map_err(|failure| Error::Neural(failure.retire_unsubmitted(context)))?;
        let layers: &[WorkspaceResidentLayerState] = source.state.as_ref();
        let mut count = Some(0usize);
        for layer in layers {
            for _ in layer.retained_values() {
                count = count.and_then(|value| value.checked_add(1));
            }
        }
        let count = count
            .and_then(|value| value.checked_mul(2))
            .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?;
        let mut retained = context.metadata_vec(count).map_err(Error::Neural)?;
        for layer in layers {
            retained.extend(layer.retained_values().cloned());
        }
        context.begin_state_span(&retained).map_err(Error::Neural)?;
        let state = DeviceState::<WorkspaceBackend, WorkspaceResidentLayerState>
            ::create_workspace_with_shared_layout_result(self.shared_layout().clone(), context,
                |index, _policy| -> Result<_, eredu_nn::Error> {
                    let copied = match (layers.get(index), self.source.layer(index)) {
                        (Some(WorkspaceResidentLayerState::Paged(paged)), Some(MlxKeyValueLayerState::Paged(_))) =>
                            WorkspaceResidentLayerState::Paged(paged.copy_for_resume(context)?),
                        (Some(WorkspaceResidentLayerState::Ordinary(ordinary)), Some(MlxKeyValueLayerState::Stateless)) =>
                            WorkspaceResidentLayerState::Ordinary(ordinary.clone()),
                        _ => return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch)),
                    };
                    for value in copied.retained_values() {
                        if retained.len() == count {
                            return Err(context.metadata_source(WorkingMemoryError::IdentityMismatch));
                        }
                        retained.push(value.clone());
                    }
                    Ok(copied)
                }).map_err(Error::Neural)?;
        if retained.len() != count {
            return Err(Error::Neural(
                context.metadata_source(WorkingMemoryError::IdentityMismatch),
            ));
        }
        let copy = context.finish_report(&retained).map_err(Error::Neural)?;
        Ok(ProjectedDenseResidentKvCopy {
            state,
            source_storage: source.storage,
            copy,
        })
    }
}
