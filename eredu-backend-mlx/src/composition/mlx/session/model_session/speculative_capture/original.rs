//! Actual loaded publication and communication table for independent capture.
use super::*;
use crate::backend::runtime::distributed::topology::original_source::{
    control::{CaptureTransportBinding, speculative::PreparedSpeculativeControl},
    parallel::OriginalParallelSource,
};
use crate::composition::mlx::session::bounded_capture::partition::{
    PartitionCaptureInvocation, PartitionCaptureInvocationSource,
};
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceMetadataAllocation};
use eredu_runtime::{
    replicated_session::ReplicatedTextControlOrigin,
    working_memory::{
        MemoryLedger, OriginalSpeculativeRequest, OwnedPartitionFragmentHostPlan,
        WorkingMemoryError,
    },
};
use std::mem::{size_of, size_of_val};

/// Move-only cold source. It contains no model role or native submission budget.
pub(in crate::composition::mlx) struct OriginalModelPartitionPreparation {
    loaded: super::super::partition_capture::LoadedPartitionCapture,
    parallel: OriginalParallelSource,
    transport: CaptureTransportBinding,
    origin: ReplicatedTextControlOrigin,
    overlay: Option<String>,
    pool: MemoryLedger,
    funding: HostMetadataFunding,
}
pub(in crate::composition::mlx) struct OriginalModelPartitionSource {
    loaded: super::super::partition_capture::LoadedPartitionCapture,
    control: PreparedSpeculativeControl,
    transport: CaptureTransportBinding,
    origin: ReplicatedTextControlOrigin,
    overlay: Option<String>,
    funding: HostMetadataFunding,
}
impl MlxModelSession {
    pub(in crate::composition::mlx) fn prepare_original_model_partition_source(
        &self,
        funding: &HostMetadataFunding,
    ) -> Result<Option<OriginalModelPartitionPreparation>, Error> {
        let frames = [
            size_of::<OriginalModelPartitionPreparation>(),
            size_of::<Option<OriginalModelPartitionPreparation>>(),
            size_of::<Result<Option<OriginalModelPartitionPreparation>, Error>>(),
            size_of::<(&Self, &HostMetadataFunding)>(),
            size_of::<ReplicatedTextControlOrigin>(),
            size_of::<Option<String>>(),
        ];
        funding.reserve_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?,
        )?;
        let Some(base) = self.payload.distributed.as_ref() else {
            return Ok(None);
        };
        let loaded = self
            .original_partition_capture(funding)?
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let parallel = self
            .original_workspace_parallel_source(funding)?
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        let actual = parallel.communication_source()?;
        if loaded.source_labels().2 != base.session_identity() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let transport = CaptureTransportBinding::prepare(base, &actual, funding)?;
        drop(actual);
        let origin = self
            .payload
            .model
            .erased()
            .resident_control_origin_fixed()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?
            .map_err(|cause| {
                crate::composition::mlx::model::retain_planning_error(cause, funding.clone())
            })?;
        let overlay = self
            .payload
            .parameter_state
            .active
            .as_deref()
            .map(|text| funding.metadata_string(format_args!("{text}")))
            .transpose()?;
        Ok(Some(OriginalModelPartitionPreparation {
            loaded,
            parallel,
            transport,
            origin,
            overlay,
            pool: self.payload.memory_ledger.clone(),
            funding: funding.clone(),
        }))
    }
}
impl OriginalModelPartitionPreparation {
    pub(in crate::composition::mlx) fn origin(&self) -> &ReplicatedTextControlOrigin {
        &self.origin
    }
    pub(in crate::composition::mlx) fn bind(
        self,
        request: &OriginalSpeculativeRequest,
    ) -> Result<OriginalModelPartitionSource, Error> {
        self.funding.reserve_metadata(size_of::<(
            Self,
            OriginalModelPartitionSource,
            Result<OriginalModelPartitionSource, Error>,
            &OriginalSpeculativeRequest,
        )>())?;
        request
            .validate_pool(&self.pool)
            .map_err(Error::PrefillControl)?;
        let control = PreparedSpeculativeControl::new(
            self.parallel,
            request.execution_identity(),
            &self.pool,
        )?;
        Ok(OriginalModelPartitionSource {
            loaded: self.loaded,
            control,
            transport: self.transport,
            origin: self.origin,
            overlay: self.overlay,
            funding: self.funding,
        })
    }
}
impl OriginalModelPartitionSource {
    pub(in crate::composition::mlx) fn origin(&self) -> &ReplicatedTextControlOrigin {
        &self.origin
    }
    pub(in crate::composition::mlx) fn control(&self) -> &PreparedSpeculativeControl {
        &self.control
    }
    pub(in crate::composition::mlx) fn transport(&self) -> &CaptureTransportBinding {
        &self.transport
    }
    pub(in crate::composition::mlx) fn rank(&self) -> usize {
        self.control.capture_rank()
    }
    pub(in crate::composition::mlx) fn layouts(
        &self,
    ) -> &eredu_architectures::component_partition::ComponentPartitionLayouts {
        self.loaded.layouts()
    }
    pub(in crate::composition::mlx) fn prepare_evidence(
        &self,
        parent: &SharedCapturePlan,
        model: &crate::composition::mlx::session::intervention::PreparedModelInterventions,
        funding: &HostMetadataFunding,
    ) -> Result<
        (
            crate::composition::mlx::session::bounded_capture::partition::ModelEvidenceHostSource,
            Vec<(usize, [OwnedPartitionFragmentHostPlan; 2])>,
        ),
        Error,
    > {
        crate::composition::mlx::session::bounded_capture::partition::ModelEvidenceHostSource::prepare(
            &self.loaded, self.rank(), self.overlay.as_deref(), parent, model, funding)
    }
    pub(in crate::composition::mlx) fn prepare_invocation(
        &self,
        source: &SharedCapturePlan,
        invocation: PartitionCaptureInvocation,
        funding: &HostMetadataFunding,
    ) -> Result<
        (
            PartitionCaptureInvocationSource,
            Vec<OwnedPartitionFragmentHostPlan>,
        ),
        Error,
    > {
        PartitionCaptureInvocationSource::prepare(
            &self.loaded,
            self.rank(),
            self.overlay.as_deref(),
            source,
            invocation,
            funding,
        )
    }
}
