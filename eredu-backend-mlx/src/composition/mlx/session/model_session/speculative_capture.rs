//! Native resources bound to architecture-selected internal capture placement.
use super::*;
use eredu_architectures::{
    component_partition::ComponentPartitionLayouts, prepared_sources::PreparedModelDiscovery,
    speculative_execution::SpeculativeActivationExecution,
};
use eredu_core::{capture::*, speculative::*};
use eredu_runtime::capture::{
    CaptureExecutionError, PartitionCaptureBackendProvider, SpeculativeCaptureObserver,
    partition::*,
};
use std::sync::{Arc, OnceLock};

pub(super) type LayoutsCell = Arc<OnceLock<Result<Arc<ComponentPartitionLayouts>, CaptureError>>>;

pub(in crate::composition::mlx) struct SpeculativePartitionBinding {
    prepared: Option<Arc<PreparedModelDiscovery>>,
    transport: Arc<MlxDistributedSession>,
    execution: Option<SpeculativeActivationExecution>,
    session_identity: String,
    overlay: Option<String>,
    layouts: LayoutsCell,
}
impl std::fmt::Debug for SpeculativePartitionBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SpeculativePartitionBinding")
            .finish_non_exhaustive()
    }
}
impl MlxModelSession {
    /// Retains cheap loaded resources only. Layout compilation, content identity
    /// and collector checks remain deferred until discovery or actual admission.
    pub(in crate::composition::mlx) fn speculative_partition_binding(
        &self,
    ) -> Option<SpeculativePartitionBinding> {
        Some(SpeculativePartitionBinding {
            prepared: self.capture_discovery.clone(),
            transport: Arc::new(self.payload.distributed.as_ref()?.clone()),
            execution: self
                .payload
                .model
                .erased()
                .speculative_activation_execution(),
            session_identity: self.intervention_session_identity.clone(),
            overlay: self.payload.parameter_state.active.clone(),
            layouts: Arc::clone(&self.speculative_capture_layouts),
        })
    }
}
impl SpeculativePartitionBinding {
    pub(in crate::composition::mlx) fn coordinate_speculative_step<
        B: AsRef<[eredu_core::SpeculativeScheduleState]>
            + AsMut<[eredu_core::SpeculativeScheduleState]>,
    >(
        &self,
        local: B,
    ) -> Result<B, eredu_core::BackendFailure> {
        self.transport.coordinate_speculative_step(local)
    }

    pub(in crate::composition::mlx) fn agree_text_preparation(
        &self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, eredu_core::BackendFailure>
    {
        self.transport.agree_text_preparation(stage, status)
    }

    pub(in crate::composition::mlx) fn finish_preparation<T, E>(
        &self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        local: Result<T, E>,
        map_backend: impl FnOnce(eredu_core::BackendFailure) -> E,
    ) -> Result<T, E> {
        eredu_core::run_preparation::finish_preparation(
            stage,
            local,
            |status| self.agree_text_preparation(stage, status),
            map_backend,
        )
    }

    fn prepared(&self) -> Result<&PreparedModelDiscovery, CaptureError> {
        self.prepared.as_deref().ok_or_else(|| {
            CaptureError::Unsupported("session has no retained activation catalog".into())
        })
    }
    fn execution(&self) -> Result<&SpeculativeActivationExecution, CaptureError> {
        self.execution.as_ref().ok_or_else(|| {
            CaptureError::Unsupported(
                "selected prediction executor has no complete internal activation path".into(),
            )
        })
    }
    fn layouts(&self) -> Result<Arc<ComponentPartitionLayouts>, CaptureError> {
        self.layouts
            .get_or_init(|| {
                let layouts = self
                    .prepared()?
                    .speculative_component_partition_layouts(self.execution()?, 1024)
                    .map_err(|error| CaptureError::Unsupported(error.to_string()))?
                    .ok_or_else(|| {
                        CaptureError::Invalid(
                            "distributed prediction has no retained placement".into(),
                        )
                    })?;
                if layouts.topology().world_size()
                    != self.transport.session_identity().participant_count()
                {
                    return Err(CaptureError::Invalid(
                        "prediction placement differs from native communication setup".into(),
                    ));
                }
                Ok(Arc::new(layouts))
            })
            .clone()
    }
    pub(in crate::composition::mlx) fn discovery(
        &self,
    ) -> Result<SpeculativeActivationDiscovery, CaptureError> {
        let layouts = self.layouts()?;
        self.transport.capture_wait()?;
        self.transport.estimate_capture_gather(16)?;
        let mut discovery = self
            .prepared()?
            .speculative_activations_with_partition_support(
                self.execution()?,
                &super::super::intervention::mechanisms(),
                &self.session_identity,
                self.overlay.as_deref(),
                &layouts,
                |point| {
                    let Some(members) = layouts.capture_hook_members(&point.path) else {
                        return eredu_core::ObservationSupportStatus::Unverified(
                            "prediction point has no retained producer ownership".into(),
                        );
                    };
                    match self.transport.estimate_capture_hook(&members) {
                        Ok(_) => eredu_core::ObservationSupportStatus::Supported,
                        Err(error) => {
                            eredu_core::ObservationSupportStatus::Unverified(error.to_string())
                        }
                    }
                },
            )?;
        discovery.captures.support.capture.conditions.push("Every model rank participates in the same invocation plan; committed receipts retain independent prediction and physical sequence coordinates".into());
        Ok(discovery)
    }
    pub(in crate::composition::mlx) fn observer(
        &self,
        plan: &AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        stream: &Stream,
    ) -> Result<
        Option<
            Box<dyn eredu_runtime::inspection::SpeculativeActivationObserver<MlxTensor, Exception>>,
        >,
        SpeculativeControlError,
    > {
        self.observer_with_error(
            plan,
            request,
            stream,
            |error: &CaptureExecutionError<Error>| Exception::custom(error.to_string()),
        )
    }
    pub(in crate::composition::mlx) fn observer_with_error<E: 'static, F>(
        &self,
        plan: &AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        stream: &Stream,
        map_error: F,
    ) -> Result<
        Option<Box<dyn eredu_runtime::inspection::SpeculativeActivationObserver<MlxTensor, E>>>,
        SpeculativeControlError,
    >
    where
        F: eredu_runtime::capture::SpeculativeCaptureErrorTransport<Error, E> + 'static,
    {
        if plan.is_empty() {
            return Ok(None);
        }
        plan.validate(&self.discovery()?)?;
        let provider = PartitionCaptureBackendProvider::new(
            super::super::bounded_capture::SpeculativeCaptureProvider::new(
                stream.clone(),
                Some((*self.transport).clone()),
            ),
            Arc::clone(&self.transport),
            self.layouts()?,
            PartitionCaptureIdentity::for_session(
                plan.artifact_identity().into(),
                self.prepared()?.execution_identity().into(),
                self.transport.session_identity(),
                self.overlay.clone(),
            )?,
            PartitionCaptureReceiptLimits {
                max_producers: self.transport.session_identity().participant_count(),
                max_fragments: 65_536,
                max_record_bytes: plan.captures().plan().limits.per_step.encoded_bytes,
            },
            super::partition_capture::estimate,
            |error: PartitionCaptureObserverError<Error>| match error {
                PartitionCaptureObserverError::Capture(error) => error,
                PartitionCaptureObserverError::Exchange(
                    PartitionCaptureExchangeError::Capture(error),
                ) => CaptureExecutionError::Admission(error),
                error => CaptureExecutionError::Backend(Error::observation(error)),
            },
        );
        Ok(SpeculativeCaptureObserver::from_admitted(
            plan,
            provider,
            map_error,
            request,
            Arc::new(super::super::intervention::NativeInterventionEstimator),
        )?
        .map(|observer| {
            Box::new(observer)
                as Box<dyn eredu_runtime::inspection::SpeculativeActivationObserver<MlxTensor, E>>
        }))
    }
}

mod original;
pub(in crate::composition::mlx) use original::{
    OriginalModelPartitionPreparation, OriginalModelPartitionSource,
};
