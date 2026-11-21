//! Portable binding of a retained speculative partition to bounded capture.

use super::{
    CaptureBackendProvider, PartitionCaptureBackendProvider, SpeculativeCaptureObserver,
    SpeculativePartitionCaptureDiscovery, SpeculativePartitionLayout, SpeculativePartitionLayouts,
};
use crate::{
    capture::{partition::*, CaptureExecutionError},
    inspection::SpeculativeActivationObserver,
    run_preparation::TextPreparationCoordinator,
    speculative::SpeculativeActivationExecution,
};
use eredu_core::{
    capture::*,
    communication::{
        CommunicationGroupRequirements, CommunicationManifest, CommunicationSessionIdentity,
    },
    intervention::{InterventionEstimator, InterventionMechanisms},
    run_preparation::TextPreparationTransport,
    speculative::*,
    BackendFailure, CollectiveGroupId, Completion,
};
use std::sync::Arc;

/// Retained portable identity, layout, and coordination for partitioned speculative capture.
#[derive(Clone)]
pub struct SpeculativePartitionBinding<T> {
    prepared: Option<Arc<dyn SpeculativePartitionCaptureDiscovery>>,
    transport: Arc<T>,
    setup: CommunicationSessionIdentity,
    preparation: Option<Arc<TextPreparationCoordinator>>,
    execution: Option<SpeculativeActivationExecution>,
    session_identity: String,
    overlay: Option<String>,
    layouts: SpeculativePartitionLayouts,
}

impl<T> SpeculativePartitionBinding<T> {
    /// Retains selected resources. Layout compilation and collector admission are deferred.
    pub(crate) fn new(
        prepared: Option<Arc<dyn SpeculativePartitionCaptureDiscovery>>,
        transport: T,
        setup: CommunicationSessionIdentity,
        preparation: Option<Arc<TextPreparationCoordinator>>,
        execution: Option<SpeculativeActivationExecution>,
        session_identity: String,
        overlay: Option<String>,
        layouts: SpeculativePartitionLayouts,
    ) -> Self {
        Self {
            prepared,
            transport: Arc::new(transport),
            setup,
            preparation,
            execution,
            session_identity,
            overlay,
            layouts,
        }
    }

    /// Borrows transport retained by this installation.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Validates the requested operations against the selected tensor group.
    pub fn assistant_group(
        &self,
        requirements: &CommunicationGroupRequirements,
    ) -> Result<Option<(CollectiveGroupId, &CommunicationManifest)>, BackendFailure> {
        let Some((id, manifest)) = self
            .prepared()
            .map_err(BackendFailure::from_error)?
            .tensor_parallel_communication()
        else {
            return Ok(None);
        };
        let selected = manifest
            .groups()
            .iter()
            .find(|group| group.id() == id)
            .ok_or_else(|| {
                BackendFailure::from_error(CaptureError::Invalid(
                    "retained target tensor communicator is absent from its manifest".into(),
                ))
            })?;
        for requirement in requirements.operations() {
            selected
                .requirements()
                .validate_requirement(requirement)
                .map_err(BackendFailure::from_error)?;
        }
        Ok(Some((id, manifest)))
    }

    fn prepared(&self) -> Result<&dyn SpeculativePartitionCaptureDiscovery, CaptureError> {
        self.prepared.as_deref().ok_or_else(|| {
            CaptureError::Unsupported("session has no retained activation catalog".into())
        })
    }

    fn layouts(&self) -> Result<Arc<SpeculativePartitionLayout>, CaptureError> {
        self.layouts.resolve(
            self.prepared.clone(),
            self.execution.as_ref(),
            self.setup.clone(),
        )
    }
}

impl<T: TextPreparationTransport> SpeculativePartitionBinding<T>
where
    T::Error: Send + Sync + 'static,
{
    /// Agrees the decoding handoff with the selected participants.
    pub fn coordinate_decoding_control(
        &self,
        local: eredu_core::decoding::DecodingCoordination,
    ) -> Result<eredu_core::decoding::DecodingCoordination, eredu_core::BackendFailure> {
        TextPreparationCoordinator::require(self.preparation.as_deref())
            .and_then(|coordinator| {
                coordinator.coordinate_decoding_control(&*self.transport, local)
            })
            .map_err(eredu_core::BackendFailure::from_error)
    }
    /// Agrees the next scheduled speculative operations.
    pub fn coordinate_speculative_step(
        &self,
        local: Vec<eredu_core::SpeculativeScheduleState>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, eredu_core::BackendFailure> {
        self.require_preparation()?
            .coordinate_speculative_step(&*self.transport, local)
            .map_err(eredu_core::BackendFailure::from_error)
    }

    /// Agrees local request preparation at the retained stage.
    pub fn agree_text_preparation(
        &self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        self.require_preparation()?
            .agree(&*self.transport, stage, status)
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn require_preparation(
        &self,
    ) -> Result<&TextPreparationCoordinator, eredu_core::BackendFailure> {
        TextPreparationCoordinator::require(self.preparation.as_deref()).map_err(|error| {
            eredu_core::BackendFailure::new(eredu_core::BackendFailureKind::Unsupported, error)
        })
    }
}

impl<T: PartitionCaptureHookTransport> SpeculativePartitionBinding<T> {
    /// Combines retained model placement with facts about the installed collector.
    pub fn discovery(
        &self,
        mechanisms: &InterventionMechanisms,
    ) -> Result<SpeculativeActivationDiscovery, CaptureError> {
        self.layouts()?.discover(
            &*self.transport,
            mechanisms,
            &self.session_identity,
            self.overlay.as_deref(),
        )
    }

    /// Admits and installs an observer over native capture primitives and the retained placement.
    #[allow(clippy::too_many_arguments)]
    pub fn observer<P, E, N, F>(
        &self,
        plan: &AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        mechanisms: &InterventionMechanisms,
        provider: P,
        estimate: N,
        map_partition_error: F,
        map_error: impl Fn(&CaptureExecutionError<P::Error>) -> E + 'static,
        intervention: Arc<dyn InterventionEstimator>,
    ) -> Result<Option<Box<dyn SpeculativeActivationObserver<P::Tensor, E>>>, SpeculativeControlError>
    where
        P: CaptureBackendProvider + 'static,
        T: 'static,
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
        E: std::error::Error + Send + Sync + 'static,
        N: FnMut(
                &[u64],
                &CaptureSelection,
                &ResolvedCaptureSlice,
            ) -> Result<PartitionCaptureNativeEstimate, CaptureError>
            + 'static,
        F: Fn(PartitionCaptureObserverError<P::Error>) -> CaptureExecutionError<P::Error> + 'static,
    {
        if plan.is_empty() {
            return Ok(None);
        }
        plan.validate(&self.discovery(mechanisms)?)?;
        let provider = PartitionCaptureBackendProvider::new(
            provider,
            Arc::clone(&self.transport),
            self.layouts()?.layout(),
            PartitionCaptureIdentity::for_session(
                Some(plan.artifact_identity().into()),
                self.prepared()?.execution_identity().into(),
                self.setup.clone(),
                self.overlay.clone(),
            )?,
            PartitionCaptureReceiptLimits {
                max_producers: self.setup.participant_count(),
                max_fragments: 65_536,
                max_record_bytes: plan.captures().plan().limits.per_step.encoded_bytes,
            },
            estimate,
            map_partition_error,
        );
        Ok(SpeculativeCaptureObserver::from_admitted(
            plan,
            provider,
            map_error,
            request,
            intervention,
        )?
        .map(|observer| Box::new(observer) as Box<dyn SpeculativeActivationObserver<P::Tensor, E>>))
    }
}
