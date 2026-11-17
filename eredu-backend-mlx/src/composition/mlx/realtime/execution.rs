//! Native ownership and family-neutral erasure of prepared realtime executors.

use std::sync::Arc;

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_nn::RetainedTensorRoots;
use eredu_runtime::{
    DenseDiskStreamReport, GenerationSampler, LayerwiseModelMetadata,
    PreparedRealtimeFrameExecutor, RealtimeResidencyTelemetry, ResidencyReport,
    ResidentLayerGroupReport, SequentialDecisionDriver,
};
use safemlx::Stream;

use super::{submission::SharedRealtimeExecution, MlxRealtimeExecutionResources};
use crate::{
    backend::{
        error::Error,
        runtime::{
            cache::state::{MlxKeyValueState, MlxKeyValueTransactionBranch},
            distributed::Group,
            generation::MlxSamplingBackend,
        },
    },
    MlxTensor,
};

type RetainedRoots = Box<dyn RetainedTensorRoots<MlxTensor>>;

trait ErasedRealtimeExecutionContract {
    fn telemetry(&self) -> &dyn RealtimeResidencyTelemetry<Error = Error>;

    fn execute_decisions(
        &mut self,
        state: &mut MlxKeyValueState,
        temporal: &[MlxTensor],
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, GenerationSampler>,
        stream: &Stream,
    ) -> Result<RetainedRoots, Error>;
}

impl<E, ExecError> ErasedRealtimeExecutionContract for E
where
    E: PreparedRealtimeFrameExecutor<
            MlxSamplingBackend,
            GenerationSampler,
            MlxKeyValueState,
            Error = ExecError,
        > + RealtimeResidencyTelemetry<Error = Error>,
    ExecError: std::fmt::Display,
    E::Retained: RetainedTensorRoots<MlxTensor> + 'static,
{
    fn telemetry(&self) -> &dyn RealtimeResidencyTelemetry<Error = Error> {
        self
    }

    fn execute_decisions(
        &mut self,
        state: &mut MlxKeyValueState,
        temporal: &[MlxTensor],
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, GenerationSampler>,
        stream: &Stream,
    ) -> Result<RetainedRoots, Error> {
        let retained = self
            .execute(state, temporal, driver, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(Box::new(retained))
    }
}

/// MLX mechanisms bound to one architecture-constructed realtime execution.
///
/// Architecture-owned configuration, realization, state geometry, and frame
/// semantics remain on the surrounding neutral execution handle. This value
/// erases the typed executor and retains the native resources needed by
/// that handle.
pub struct MlxRealtimeExecution {
    metadata: LayerwiseModelMetadata,
    native: SharedRealtimeExecution<RealtimeExecutionPayload>,
}

struct RealtimeExecutionPayload {
    execution: Box<dyn ErasedRealtimeExecutionContract>,
    _store: SharedCheckpointSource,
    world: Option<Arc<Group>>,
    _weights_stream: Stream,
}

impl MlxRealtimeExecution {
    pub(crate) fn bind<E, ExecError>(
        execution: E,
        metadata: LayerwiseModelMetadata,
        store: SharedCheckpointSource,
        world: Option<Arc<Group>>,
        stream: &Stream,
        weights_stream: Stream,
    ) -> Self
    where
        E: PreparedRealtimeFrameExecutor<
                MlxSamplingBackend,
                GenerationSampler,
                MlxKeyValueState,
                Error = ExecError,
            > + RealtimeResidencyTelemetry<Error = Error>
            + 'static,
        ExecError: std::fmt::Display,
        E::Retained: RetainedTensorRoots<MlxTensor> + 'static,
    {
        Self {
            metadata,
            native: SharedRealtimeExecution::new(
                RealtimeExecutionPayload {
                    execution: Box::new(execution),
                    _store: store,
                    world,
                    _weights_stream: weights_stream,
                },
                stream,
            ),
        }
    }

    fn ensure_healthy(&self) -> Result<(), Error> {
        self.native
            .require_healthy()
            .map_err(|message| Error::ArchitectureModel(message.into()))
    }

    pub(crate) fn retain_frame(&self) -> Result<Arc<dyn MlxRealtimeExecutionResources>, Error> {
        self.native
            .require_available()
            .map_err(|message| Error::ArchitectureModel(message.into()))?;
        Ok(self.native.retain(()))
    }

    /// Parameter topology and residency metadata.
    pub fn metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }

    /// Logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.ensure_healthy()?;
        self.native
            .components()
            .borrow()
            .execution
            .telemetry()
            .residency_report()
    }

    /// Disk-stream telemetry when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.ensure_healthy()?;
        self.native
            .components()
            .borrow()
            .execution
            .telemetry()
            .dense_stream_report()
    }

    /// Per-execution-group residency reports.
    pub fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        self.ensure_healthy()?;
        self.native
            .components()
            .borrow()
            .execution
            .telemetry()
            .execution_group_reports()
    }

    pub(crate) fn validate_context(
        &self,
        stream: &Stream,
        world: Option<&Group>,
    ) -> Result<(), Error> {
        self.ensure_healthy()?;
        crate::backend::validate_native_execution_target(
            &self.native.stream().get_device()?,
            self.native
                .components()
                .borrow()
                .world
                .as_deref()
                .map(|group| group.native_group()),
            stream,
            world.map(|group| group.native_group()),
        )
    }
}

impl
    PreparedRealtimeFrameExecutor<
        MlxSamplingBackend,
        GenerationSampler,
        MlxKeyValueTransactionBranch,
    > for &MlxRealtimeExecution
{
    type Error = Error;
    type Retained = Box<dyn RetainedTensorRoots<MlxTensor>>;

    fn execute(
        &mut self,
        model_state: &mut MlxKeyValueTransactionBranch,
        temporal: &[MlxTensor],
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, GenerationSampler>,
        context: &Stream,
    ) -> Result<Self::Retained, Error> {
        self.native
            .components()
            .borrow_mut()
            .execution
            .execute_decisions(model_state, temporal, driver, context)
    }
}
