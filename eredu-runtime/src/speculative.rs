//! Backend-neutral speculative request lifecycle and fair scheduling.

use eredu_core::{
    BoundedCompletion, CompletedSpeculativeSchedule, GenerationError, GenerationRequestTable,
    GenerationTiming, PreparedSpeculativeLane, SpeculativeConstraint, SpeculativeDriverError,
    SpeculativeExecutor, SpeculativeGenerationBatchOutput, SpeculativeGenerationOutput,
    SpeculativeGenerationVisitor, SpeculativePublisher, SpeculativeSampling,
};

mod batch;
pub use batch::{
    prepare_speculative_batch_lanes, run_speculative_batch, validate_lane_proposal_capacity,
    SpeculativeBatchLane,
};
mod capture;
mod host_sampling;
pub use host_sampling::HostSpeculativeSampling;
mod sampling;
pub use sampling::{InstalledSpeculativeSampling, SamplingContext};
mod control;
mod external_capture;
mod external_state;
pub use capture::{
    speculative_capture_scope, EmbeddedPredictionObservers, SpeculativeActivationExecution,
    EMBEDDED_PREDICTION_OUTPUT_PATH, EMBEDDED_PROPOSAL_LOGITS_PATH, EMBEDDED_TARGET_CAPTURE_PATH,
    EMBEDDED_VERIFICATION_LOGITS_PATH,
};
pub use control::*;
pub use external_capture::{
    ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
    ExternalPredictionTargetOperation, ExternalPredictionTargetOperationKind,
};
pub use external_state::{
    ExternalAssistantCache, ExternalAssistantCacheCheckpoint, ExternalAssistantObservers,
};
mod causal_verification;
pub use causal_verification::CausalVerification;
mod execution_context;
pub use execution_context::SpeculativeExecutionContext;
pub mod autoregressive;
pub mod shared_backbone;

/// Neutral owner of speculative request registration and fair scheduling.
pub struct SpeculativeScheduler<'a, E, S, C, P>
where
    E: SpeculativeExecutor,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error>,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    executor: &'a mut E,
    context: E::Context<'a>,
    optimistic_execution_available: bool,
    component_timings_collected: bool,
    requests: GenerationRequestTable<'a, E, S, C, P>,
}

impl<'a, E, S, C, P> SpeculativeScheduler<'a, E, S, C, P>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    /// Creates a scheduler for one prepared executor and placement.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        executor: &'a mut E,
        options: eredu_core::generation::SpeculativeSchedulerOptions,
        topology: eredu_core::SpeculativeExecutionTopology,
        optimistic_execution_available: bool,
        component_timings_collected: bool,
        context: E::Context<'a>,
    ) -> Result<Self, SpeculativeDriverError<E::Error>> {
        let prepared = (|| {
            let completion_wait = options
                .completion_wait()
                .map_err(SpeculativeDriverError::Generation)?;
            if !E::Completion::supports_cancellation(completion_wait.cancellation()) {
                return Err(SpeculativeDriverError::UnsupportedCompletionCancellation {
                    cancellation: completion_wait.cancellation(),
                });
            }
            GenerationRequestTable::new(options, topology)
                .map_err(SpeculativeDriverError::Generation)
        })();
        let stage = eredu_core::run_preparation::TextPreparationStage::Request;
        let requests = eredu_core::run_preparation::finish_preparation(
            stage,
            prepared,
            |status| executor.agree_text_preparation(stage, status, context),
            SpeculativeDriverError::Preparation,
        )?;
        executor.set_telemetry_enabled(component_timings_collected);
        Ok(Self {
            executor,
            context,
            optimistic_execution_available,
            component_timings_collected,
            requests,
        })
    }

    /// Constrains all lanes and prospective mode commands to the retained plan.
    pub fn with_speculation_limits(
        mut self,
        limits: Option<eredu_core::decoding::SpeculationLimits>,
    ) -> Self {
        self.requests = self.requests.with_speculation_limits(limits);
        self
    }

    /// Registers and prefills one independently progressing lane.
    pub fn submit(
        &mut self,
        mut lane: PreparedSpeculativeLane<'a, E, S, C, P>,
    ) -> Result<eredu_core::generation::SpeculativeRequestId, SpeculativeDriverError<E::Error>>
    {
        self.requests.submit(
            self.executor,
            lane.take_cache(),
            lane.take_input(),
            lane.take_config(),
            lane.take_runtime(),
            lane.take_randomness(),
            self.component_timings_collected,
            self.context,
        )
    }

    /// Performs one fairly selected lifecycle action.
    pub fn step(&mut self) -> Result<bool, SpeculativeDriverError<E::Error>> {
        self.requests.step(
            self.executor,
            self.optimistic_execution_available,
            self.context,
        )
    }

    /// Drives all registered lanes to terminal states.
    pub fn run(&mut self) -> Result<(), SpeculativeDriverError<E::Error>> {
        while self.step()? {}
        Ok(())
    }

    /// Returns the current portable status for one lane.
    pub fn status(
        &self,
        id: eredu_core::generation::SpeculativeRequestId,
    ) -> Option<eredu_core::generation::SpeculativeRequestStatus> {
        self.requests.status(id)
    }

    /// Requests cancellation at the next exact safe boundary.
    pub fn cancel(
        &mut self,
        id: eredu_core::generation::SpeculativeRequestId,
    ) -> Result<(), SpeculativeDriverError<E::Error>> {
        self.requests.signal_cancellation(id)
    }

    /// Whether all registered lanes are terminal.
    pub fn is_finished(&self) -> bool {
        self.requests.is_finished()
    }

    /// Consumes a terminal scheduler into stable ordered results.
    pub fn finish(
        self,
    ) -> Result<CompletedSpeculativeSchedule<S>, SpeculativeDriverError<E::Error>> {
        self.requests.finish()
    }
}

/// Facade-selected speculative generation driver.
///
/// Backends lend prepared native resources through
/// [`SpeculativeGenerationVisitor`]. This driver alone registers lanes, runs
/// the fair schedule, observes exact completions, and constructs public
/// terminal outputs.
#[derive(Debug, Clone, Copy)]
pub struct RunSpeculativeGeneration {
    limits: Option<eredu_core::decoding::SpeculationLimits>,
    options: eredu_core::generation::SpeculativeSchedulerOptions,
    started: std::time::Instant,
}

impl Default for RunSpeculativeGeneration {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl RunSpeculativeGeneration {
    /// Creates a driver with facade-selected scheduling and lookahead controls.
    /// Starts the shared request clock; construct before preparing backend inputs
    /// and resources so terminal TTFT includes that preparation.
    pub fn new(options: eredu_core::generation::SpeculativeSchedulerOptions) -> Self {
        Self::new_at(options, std::time::Instant::now())
    }

    /// Uses the facade call's origin so all active request preparation contributes to TTFT.
    pub fn new_at(
        options: eredu_core::generation::SpeculativeSchedulerOptions,
        started: std::time::Instant,
    ) -> Self {
        Self {
            options,
            limits: None,
            started,
        }
    }
    /// Retains the source's selected resource ceilings for mode admission.
    pub fn with_speculation_limits(
        mut self,
        limits: Option<eredu_core::decoding::SpeculationLimits>,
    ) -> Self {
        self.limits = limits;
        self
    }
}

impl SpeculativeGenerationVisitor for RunSpeculativeGeneration {
    fn run<'a, E, S, C, P>(
        self,
        executor: &'a mut E,
        lanes: Vec<PreparedSpeculativeLane<'a, E, S, C, P>>,
        topology: eredu_core::SpeculativeExecutionTopology,
        optimistic_execution_available: bool,
        component_timings_collected: bool,
        context: E::Context<'a>,
    ) -> Result<SpeculativeGenerationBatchOutput, SpeculativeDriverError<E::Error>>
    where
        E: SpeculativeExecutor + 'a,
        S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>>
            + 'a,
        C: SpeculativeConstraint,
        P: SpeculativePublisher<C>,
    {
        let mut scheduler = SpeculativeScheduler::new(
            executor,
            self.options,
            topology,
            optimistic_execution_available,
            component_timings_collected,
            context,
        )?
        .with_speculation_limits(self.limits);
        let mut preparation_elapsed = Vec::with_capacity(lanes.len());
        for lane in lanes {
            preparation_elapsed.push(self.started.elapsed());
            scheduler.submit(lane)?;
        }
        scheduler.run()?;
        let completed = scheduler.finish()?;
        completed_output(completed, |request| {
            GenerationTiming::new(
                request
                    .stats()
                    .submission_to_first_token()
                    .map(|elapsed| preparation_elapsed[request.id().index()] + elapsed),
            )
        })
    }
}

fn completed_output<S, E: std::error::Error + 'static>(
    mut completed: CompletedSpeculativeSchedule<S>,
    timing: impl Fn(&eredu_core::CompletedSpeculativeRequest<S>) -> GenerationTiming,
) -> Result<SpeculativeGenerationBatchOutput, SpeculativeDriverError<E>> {
    let requests = completed
        .take_requests()
        .into_iter()
        .map(|request| {
            let reason = request.finish_reason().ok_or_else(|| {
                SpeculativeDriverError::Generation(
                    GenerationError::MissingSpeculativeFinishReason {
                        index: request.id().index(),
                    },
                )
            })?;
            Ok(SpeculativeGenerationOutput::new(
                request.token_ids().to_vec(),
                reason,
                timing(&request),
                request.stats().clone(),
            ))
        })
        .collect::<Result<Vec<_>, SpeculativeDriverError<E>>>()?;
    Ok(SpeculativeGenerationBatchOutput::new(
        requests,
        completed.take_scheduler(),
    ))
}

mod assistant_policy;
pub use assistant_policy::AssistantParameterPolicy;
mod assistant_mechanisms;
pub use assistant_mechanisms::{
    bind_external_assistant, external_assistant_conversion_retention,
    external_assistant_input_identity, trim_external_assistant_conversions, AssistantExecutorTypes,
    BoundAssistantRefVisitor, BoundAssistantVisitor, ExternalAssistantConstruction,
    ExternalAssistantExecutionMechanisms, ExternalAssistantExecutorVisitor, ExternalAssistantModel,
    ExternalAssistantPrompt, ExternalAssistantTarget, ExternalAssistantTensorPlacement,
    ExternalAssistantTransfer, InstalledAssistantFor, InstalledExternalAssistant,
    PreparedAssistantBinding, SelectedAssistantExecutionVisitor, SelectedAssistantExecutor,
};
mod embedded_execution;
pub use embedded_execution::{
    validate_prediction_output, EmbeddedPredictionCache, EmbeddedPredictionCacheAccessError,
    EmbeddedPredictionCacheError, EmbeddedPredictionCacheFactory, EmbeddedPredictionCheckpoint,
    EmbeddedPredictionDraftCache, EmbeddedPredictionDraftState, EmbeddedPredictionExecutor,
    EmbeddedPredictionStrategy, EmbeddedPredictionTargetState,
};
pub(crate) mod embedded_mechanisms;
mod target_state;
pub use embedded_mechanisms::{
    EmbeddedPredictionOutput, ReplicatedPredictionInput, ReplicatedPredictionNative,
    SpeculativeTensorMechanisms,
};
pub use target_state::{PortablePredictionTargetState, PredictionTargetLifecycle};
mod prediction_binding;
pub(crate) use prediction_binding::with_prediction_module;
pub use prediction_binding::{
    materialize_prediction_module, materialize_prediction_units, prepare_prediction_residency,
    PredictionExtensionMaterializer, PredictionInvocation, PredictionModule,
    PredictionModuleResidency, PredictionModuleRole, PredictionModuleVisitor,
    PredictionOperationInvoker, PredictionPoolingState, PredictionResidency,
    PredictionResidencyDriver, PreparedPredictionUnit, SequentialPredictionCache,
};
mod prediction_executor;
pub use prediction_executor::{
    validate_selected_capture_shapes, MaterializedPredictionExecutor, ModelPredictionStateProfile,
    PoolingPredictionStateProfile, PredictionKernel, PredictionStateLifecycle, PredictionTarget,
    SequentialPredictionStateProfile,
};
mod replicated_prediction;
pub use replicated_prediction::ReplicatedMaterializedPredictionStrategy;
mod routed_preparation;
pub use routed_preparation::{PredictionConstructionError, PreparedRoutedPrediction};
mod installed_draft;
pub use installed_draft::{DraftSessionInstallation, InstalledDraftSession};
mod prepared_external;
pub use prepared_external::{
    AssistantDispatch, BoundAssistant, BoundAssistantDispatch, ExternalDraftExecution,
    IndependentDraftModel, InstalledExternalDraft, PreparedAssistantExecution,
    PreparedAutoregressiveDraft, PreparedExternalDraft, TypedAssistant,
};

mod target_operation;
pub use target_operation::{ExternalTargetPlacements, PreparedExternalTargetOperation};
