//! One exact native communication source for token, media and saved-state quotes.
use super::*;
use crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelSource;
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceMetadataAllocation};
use eredu_runtime::working_memory::WorkingMemoryError;
impl MlxModelSession {
    pub(in crate::composition::mlx) fn original_workspace_parallel_source(
        &self,
        funding: &HostMetadataFunding,
    ) -> Result<Option<OriginalParallelSource>, Error> {
        funding
            .reserve_metadata(std::mem::size_of::<(
                &Self,
                &HostMetadataFunding,
                Option<OriginalParallelSource>,
                Result<Option<OriginalParallelSource>, Error>,
            )>())
            .map_err(Error::WorkspacePlanning)?;
        let result = (|| {
            let Some(distributed) = &self.payload.distributed else {
                return Ok(None);
            };
            let missing = || Error::PrefillControl(WorkingMemoryError::UnknownBound);
            let selected = self
                .payload
                .model
                .inference_blueprint()
                .ok_or_else(missing)?
                .selected();
            let manifest = selected.communication_manifest().ok_or_else(missing)?;
            distributed
                .original_communication_owner(manifest, distributed.native_world(), funding)?
                .prepare_model_parallel_source(
                    selected.execution().partitioned_tensor_group(),
                    selected.partitioned_session_group().ok_or_else(missing)?,
                    selected
                        .execution()
                        .partitioned_output_publication()
                        .ok_or_else(missing)?,
                    &self.payload.memory_ledger,
                )
                .map(Some)
        })();
        result.map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
        })
    }
}
