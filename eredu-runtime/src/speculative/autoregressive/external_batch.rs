//! Portable transactions for independently loaded draft models.

use eredu_core::speculative::assistant::ExternalAssistantBinder;
use eredu_core::{
    run_preparation::{finish_preparation, TextPreparationStage},
    session_authority::{DraftNativeResources, RetainedExecutionPayload, RetainedPayloadOperation},
    speculative::SpeculativeCacheFactory,
    BackendError, GenerationError, ModelRuntime, SpeculativeDriverError,
    SpeculativeExecutionTopology, SpeculativeGenerationBatchOutput, SpeculativeGenerationLane,
    SpeculativeGenerationVisitor, SpeculativeSampling, SpeculativeTokenFilterController,
};

use super::{
    batch::{prepare_independent_batch, BatchBinding, BatchQueue},
    ResidentBatchMechanisms,
};
use crate::{
    speculative::{
        prepare_speculative_batch_lanes, run_speculative_batch, BoundAssistantDispatch,
        ExternalAssistantConstruction, ExternalAssistantExecutionMechanisms,
        ExternalAssistantExecutorVisitor, ExternalDraftExecution, ExternalPredictionCaptureRequest,
        InstalledDraftSession, InstalledExternalDraft, SelectedAssistantExecutionVisitor,
        SelectedAssistantExecutor, SpeculativeBatchLane,
    },
    SelectedSpeculativeRealization,
};

/// Agrees ownership before exposing the draft payload to execution.
pub fn admit_external_draft_scope<E, S: RetainedPayloadOperation<E>>(
    acquired: Result<S, E>,
    agree: impl FnOnce(Result<(), E>) -> Result<(), E>,
) -> Result<S, E>
where
    E: From<BackendError>,
{
    match acquired {
        Ok(scope) => match agree(Ok(())) {
            Ok(()) => Ok(scope),
            Err(error) => Err(scope.settle(Ok(error))?),
        },
        Err(error) => {
            agree(Err(error))?;
            Err(BackendError::Execution {
                session: "external-draft".into(),
                operation: "acquire submission authority".into(),
                message: "draft preparation accepted absent mutation authority".into(),
            }
            .into())
        }
    }
}

/// Native resources for an external speculative batch.
pub trait ExternalBatchMechanisms<B>: ResidentBatchMechanisms<B>
where
    B: eredu_core::TextGenerationBackend,
{
    /// Native retirement resources for this draft's opaque execution payload.
    type DraftResources: RetainedExecutionPayload<
        Payload = InstalledExternalDraft<Self::AssistantDispatch, Self::ExecutableTypes>,
        Queue = BatchQueue<Self>,
        Error = B::Error,
    >;
    /// Binds the native materialized assistant types.
    type AssistantBinder: ExternalAssistantBinder;
    /// Retained portable dispatch for the selected conditioned assistant.
    type AssistantDispatch: BoundAssistantDispatch<Self::AssistantBinder> + 'static;
    /// Native operators used by the selected conditioned assistant.
    type AssistantMechanisms: for<'a> ExternalAssistantExecutionMechanisms<
        TargetTypes = Self::ExecutableTypes,
        Binder = Self::AssistantBinder,
        Logits = Self::Logits,
        Context<'a> = Self::Context<'a>,
        Error = Self::Error,
    >;
}

impl<B, M> ExternalBatchMechanisms<B> for M
where
    B: eredu_core::TextGenerationBackend + DraftNativeResources<DraftQueue = BatchQueue<M>>,
    M: ResidentBatchMechanisms<B>,
    B::AssistantBinder: ExternalAssistantBinder,
    B::AssistantDispatch: BoundAssistantDispatch<B::AssistantBinder>,
    B::AssistantOperators: for<'a> ExternalAssistantExecutionMechanisms<
        TargetTypes = M::ExecutableTypes,
        Binder = B::AssistantBinder,
        Logits = M::Logits,
        Context<'a> = M::Context<'a>,
        Error = M::Error,
    >,
{
    type DraftResources =
        B::DraftStorage<InstalledExternalDraft<B::AssistantDispatch, M::ExecutableTypes>>;
    type AssistantBinder = B::AssistantBinder;
    type AssistantDispatch = B::AssistantDispatch;
    type AssistantMechanisms = B::AssistantOperators;
}

/// A portable draft paired with the selected backend's retirement resources.
pub type BatchDraft<B, M> = InstalledDraftSession<
    <M as ExternalBatchMechanisms<B>>::AssistantBinder,
    <M as ExternalBatchMechanisms<B>>::DraftResources,
>;

/// Acquires the target and draft, runs the selected external route, and settles both.
pub fn execute_external_batch<B, M, C, V>(
    runtime: &mut ModelRuntime<B, super::BatchSession<B, M>>,
    drafter: &mut BatchDraft<B, M>,
    lanes: Vec<SpeculativeGenerationLane<'_, crate::text_session_driver::TextPromptFor<B>, C>>,
    visitor: V,
) -> Result<SpeculativeGenerationBatchOutput, B::Error>
where
    B: eredu_core::TextGenerationBackend,
    B::Error: From<BackendError>
        + From<GenerationError>
        + From<eredu_core::BackendFailure>
        + From<SpeculativeDriverError<M::Error>>,
    C: SpeculativeTokenFilterController + 'static,
    V: SpeculativeGenerationVisitor,
    M: ExternalBatchMechanisms<B>,
{
    let installed = drafter.installed();
    let selected = installed.selected().clone();
    let topology = selected.placement().topology();
    let capacity = selected.requirements().strategy().proposal_capacity();
    let capture = installed.conditioned_capture().cloned();
    let binding = BatchBinding::<B, M>::new(runtime);
    let map_agreement = |error| SpeculativeDriverError::<M::Error>::Preparation(error).into();
    let target_context = binding.context();
    let agree_target =
        |status| M::agree_text_preparation(TextPreparationStage::Request, status, target_context);
    let target = runtime.session();
    let pair_ready = target.ensure_no_submission_in_flight().and_then(|()| {
        if installed.target_identity() == target.intervention_identity() {
            return Ok(());
        }
        Err(BackendError::Execution {
            session: "external-draft".into(),
            operation: "validate target identity".into(),
            message: "external drafter belongs to a different target session".into(),
        }
        .into())
    });
    finish_preparation(
        TextPreparationStage::Request,
        pair_ready,
        agree_target,
        map_agreement,
    )?;
    use crate::capture::SpeculativeCaptureMechanisms;
    let draft_queue = drafter.retain_queue();
    let context = finish_preparation(
        TextPreparationStage::Request,
        M::Capture::bind_context(&binding.target, &draft_queue, topology)
            .map(|context| M::Capture::attach_partition(context, binding.capture.as_ref()))
            .map_err(|error| SpeculativeDriverError::Backend(error).into()),
        agree_target,
        map_agreement,
    )?;
    let agree_ownership = |local| {
        finish_preparation(
            TextPreparationStage::Request,
            local,
            |status| M::agree_text_preparation(TextPreparationStage::Request, status, context),
            map_agreement,
        )
    };
    runtime.session_mut().with_prepared_model_transaction(
        |local| {
            agree_ownership(local)?;
            let prepared = finish_preparation(
                TextPreparationStage::Sampling,
                prepare_speculative_batch_lanes(lanes, capacity.get(), M::prepare_sampling),
                |status| M::agree_text_preparation(TextPreparationStage::Sampling, status, context),
                map_agreement,
            )?;
            let draft = admit_external_draft_scope(drafter.begin_operation(), agree_ownership)?;
            Ok((draft, prepared))
        },
        |model, (mut draft, lanes)| {
            let result = (|| match draft.payload_mut().execution_mut() {
                ExternalDraftExecution::Autoregressive { model: draft, .. } => {
                    let (mut executor, prepared) = prepare_independent_batch::<M, _, _>(
                        model.as_mut(),
                        draft.executable_mut(),
                        capacity,
                        lanes,
                        context,
                    )?;
                    Ok(run_speculative_batch(
                        &mut executor,
                        prepared,
                        &(),
                        M::reborrow_context(context),
                        topology,
                        eredu_core::speculative::timing::component_timing_enabled(),
                        visitor,
                    )?)
                }
                ExternalDraftExecution::Assistant(assistant) => {
                    let capture = capture.ok_or_else(|| BackendError::Execution {
                        session: "external-draft".into(),
                        operation: "bind conditioned draft".into(),
                        message: "conditioned draft has no retained target capture request".into(),
                    })?;
                    let target = model.as_mut().external_prediction_mut().ok_or_else(|| {
                        BackendError::Execution {
                            session: "external-draft".into(),
                            operation: "bind conditioned target".into(),
                            message:
                                "selected target has no external-assistant prediction capability"
                                    .into(),
                        }
                    })?;
                    run_conditioned_batch::<M::AssistantMechanisms, _, _, _, B::Error>(
                        target,
                        assistant.dispatch_mut(),
                        capture,
                        &selected,
                        lanes,
                        M::reborrow_context(context),
                        topology,
                        eredu_core::speculative::timing::component_timing_enabled(),
                        visitor,
                    )
                }
            })();
            draft.settle(result)
        },
    )
}

/// Runs a bound conditioned assistant through the portable batch scheduler.
pub fn run_conditioned_batch<M, D, S, V, E>(
    target: &mut crate::speculative::ExternalAssistantTarget<M>,
    assistant: &mut D,
    capture: ExternalPredictionCaptureRequest,
    selected: &SelectedSpeculativeRealization,
    lanes: Vec<AssistantLane<'_, M, S>>,
    context: M::Context<'_>,
    topology: SpeculativeExecutionTopology,
    collect_timings: bool,
    visitor: V,
) -> Result<SpeculativeGenerationBatchOutput, E>
where
    M: ExternalAssistantExecutionMechanisms,
    D: BoundAssistantDispatch<M::Binder>,
    S: for<'a> SpeculativeSampling<
            Logits = M::Logits,
            Error = M::Error,
            Context<'a> = M::Context<'a>,
        > + 'static,
    V: SpeculativeGenerationVisitor,
    E: From<SpeculativeDriverError<M::Error>>,
{
    assistant.visit_executor(ConditionedBatch::<M, S, V, E> {
        target,
        capture,
        runner: ConditionedExecutorRunner {
            lanes,
            selected,
            context: M::reborrow_context(context),
            topology,
            collect_timings,
            visitor,
            error: std::marker::PhantomData,
        },
    })
}

type AssistantLane<'a, M, S> = SpeculativeBatchLane<
    'a,
    crate::speculative::ExternalAssistantPrompt<M>,
    S,
    <S as SpeculativeSampling>::Seed,
>;

struct ConditionedBatch<'a, 'lane, M, S, V, E>
where
    M: ExternalAssistantExecutionMechanisms,
    S: SpeculativeSampling,
{
    target: &'a mut crate::speculative::ExternalAssistantTarget<M>,
    capture: ExternalPredictionCaptureRequest,
    runner: ConditionedExecutorRunner<'a, 'lane, M, S, V, E>,
}

impl<M, S, V, E> SelectedAssistantExecutionVisitor<M::Binder>
    for ConditionedBatch<'_, '_, M, S, V, E>
where
    M: ExternalAssistantExecutionMechanisms,
    S: for<'a> SpeculativeSampling<
            Logits = M::Logits,
            Error = M::Error,
            Context<'a> = M::Context<'a>,
        > + 'static,
    V: SpeculativeGenerationVisitor,
    E: From<SpeculativeDriverError<M::Error>>,
{
    type Output = Result<SpeculativeGenerationBatchOutput, E>;

    fn visit<A, Dispatch>(
        self,
        assistant: &mut crate::speculative::InstalledExternalAssistant<M::Binder, A>,
        dispatch: Dispatch,
    ) -> Self::Output
    where
        A: ExternalAssistantConstruction,
        Dispatch: SelectedAssistantExecutor<A>,
    {
        dispatch.execute::<M, _>(self.target, assistant, self.capture, self.runner)
    }
}

struct ConditionedExecutorRunner<'a, 'lane, M, S, V, E>
where
    M: ExternalAssistantExecutionMechanisms,
    S: SpeculativeSampling,
{
    lanes: Vec<AssistantLane<'lane, M, S>>,
    selected: &'a SelectedSpeculativeRealization,
    context: M::Context<'a>,
    topology: SpeculativeExecutionTopology,
    collect_timings: bool,
    visitor: V,
    error: std::marker::PhantomData<fn() -> E>,
}

impl<A, M, S, V, F> ExternalAssistantExecutorVisitor<A, M>
    for ConditionedExecutorRunner<'_, '_, M, S, V, F>
where
    A: ExternalAssistantConstruction,
    M: ExternalAssistantExecutionMechanisms,
    S: for<'a> SpeculativeSampling<
            Logits = M::Logits,
            Error = M::Error,
            Context<'a> = M::Context<'a>,
        > + 'static,
    V: SpeculativeGenerationVisitor,
    F: From<SpeculativeDriverError<M::Error>>,
{
    type Output = Result<SpeculativeGenerationBatchOutput, F>;

    fn execute<'run, E>(self, executor: &'run mut E) -> Self::Output
    where
        Self: 'run,
        E: SpeculativeCacheFactory<
                SelectedSpeculativeRealization,
                Input = crate::speculative::ExternalAssistantPrompt<M>,
                Cache = crate::speculative::ExternalAssistantCache<M::NativeCache>,
                Logits = M::Logits,
                Context<'run> = M::Context<'run>,
                Completion = M::Completion,
                Telemetry = (),
                Error = M::Error,
            > + eredu_core::speculative::GenerationExecutorCacheFactory<
                crate::speculative::AssistantExecutorTypes<M>,
            > + 'run,
        E::Cache: 'static,
        E::TargetState: 'static,
        E::DraftState: 'static,
        E::CacheCheckpoint: 'static,
        E::Verification: 'static,
    {
        Ok(run_speculative_batch(
            executor,
            self.lanes,
            self.selected,
            M::reborrow_context(self.context),
            self.topology,
            self.collect_timings,
            self.visitor,
        )?)
    }
}
