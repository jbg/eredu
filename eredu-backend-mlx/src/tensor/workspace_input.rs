//! Projects only the actual original native tensor handles borrowed by the
//! architecture-owned source factory. No native allocation/evaluation occurs.
use super::*;
use crate::backend::nn::workspace::ExistingArrayProjection;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceTensor};
impl eredu_runtime::input::PreparedMediaWorkspaceTensor for MlxTensor {
    type Projection<'a> = ExistingArrayProjection<'a>;
    fn workspace_projection<'a>(context: &'a WorkspaceContext) -> Self::Projection<'a> {
        ExistingArrayProjection::new(context)
    }
    fn workspace_projection_with_count<'a>(
        context: &'a WorkspaceContext,
        slots: usize,
    ) -> Result<Self::Projection<'a>, eredu_nn::Error> {
        ExistingArrayProjection::with_source_count(context, slots).map_err(|cause| match cause {
            crate::backend::nn::workspace::ProjectionInventoryError::Metadata(cause) => cause,
            cause => context.metadata_source(cause),
        })
    }
    fn finish_workspace_projection<'a>(
        projection: Self::Projection<'a>,
        context: &'a WorkspaceContext,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceBorrowedStorage>, eredu_nn::Error> {
        if !projection.is_complete() {
            return Ok(None);
        }
        let slots = projection.storage_roots().count();
        let bytes = eredu_nn::workspace::WorkspaceBorrowedStorage::construction_bytes(slots)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(bytes)?;
        eredu_nn::workspace::WorkspaceBorrowedStorage::new_finite(
            context,
            projection.storage_roots().map(|(_, _, root)| root),
            slots,
        )
        .map(Some)
        .map_err(|cause| context.metadata_source(cause))
    }
    fn project_workspace_slot<'a>(
        &'a self,
        projection: &mut Self::Projection<'a>,
    ) -> Result<WorkspaceTensor, eredu_nn::Error> {
        #[cfg(test)]
        SLOT_PROJECTIONS.set(SLOT_PROJECTIONS.get() + 1);
        projection.project(self.as_array())
    }
}

#[cfg(test)]
thread_local! { static SLOT_PROJECTIONS:std::cell::Cell<usize>=const {std::cell::Cell::new(0)}; }
#[cfg(test)]
pub(crate) fn reset_workspace_slot_projections() {
    SLOT_PROJECTIONS.set(0);
}
#[cfg(test)]
pub(crate) fn workspace_slot_projections() -> usize {
    SLOT_PROJECTIONS.get()
}
