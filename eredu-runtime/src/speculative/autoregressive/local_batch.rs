//! Portable batch assembly for causal targets with eager host score readout.

use std::{convert::Infallible, num::NonZeroUsize};

use eredu_core::{
    backend::{
        local_text::HostCausalVerification, CausalTextGeneration, NativeTextValues,
        TextSamplingNative, TextSessionNative, TextTransactionMechanisms,
    },
    eager_completion::EagerRetainedPayload,
    input::PlainTextTokens,
    session_authority::{RetainedExecutionPayload, RetainedPayloadOperation},
    speculative::{
        assistant::ExternalAssistantOperators,
        erasure::{DynGenerationExecutor, ErasedGenerationExecutor},
        timing::component_timing_enabled,
        GenerationExecutorCacheFactory, SpeculativeCacheFactory,
    },
    BackendFailure, BackendFailureKind, ModelRuntime, SpeculativeExecutionTopology,
    SpeculativeGenerationBatchOutput, SpeculativeGenerationLane, SpeculativeGenerationVisitor,
    SpeculativeTokenFilterController, TextGenerationBackend,
};

use crate::{
    generation::TextSampler,
    speculative::{
        batch::{prepare_batch_lane, validate_lane_proposal_capacity},
        run_speculative_batch, AssistantExecutorTypes, BoundAssistant, BoundAssistantDispatch,
        DraftSessionInstallation, ExternalAssistantCache, ExternalAssistantConstruction,
        ExternalAssistantExecutionMechanisms, ExternalAssistantExecutorVisitor,
        ExternalAssistantPrompt, ExternalAssistantTarget, ExternalDraftExecution,
        ExternalPredictionCaptureRequest, HostSpeculativeSampling, IndependentDraftModel,
        InstalledDraftSession, InstalledExternalDraft, SelectedAssistantExecutionVisitor,
        SelectedAssistantExecutor,
    },
    text_session_driver::{TextPromptFor, TextSessionDriver, TextSessionMechanisms},
    ConstrainedSampler,
};

use super::{
    local_causal::{LocalCausalCapture, LocalCausalMechanisms},
    AutoregressiveExecutor, AutoregressiveInput, CausalExecutor, NoTargetRefinement,
};

/// Native mechanisms required by the eager causal path.
pub trait LocalCausalBackend:
    TextGenerationBackend<Error = BackendFailure>
    + HostCausalVerification<Values = <Self as TextSessionNative<Self>>::TextValues>
    + TextSessionNative<
        Self,
        Generation = CausalTextGeneration,
        Retained: 'static,
        TextValues: NativeTextValues<
            InputTensor = <Self as TextGenerationBackend>::InputTensor,
            Embedded = <Self as HostCausalVerification>::Embedded,
            Context: 'static,
            Scores: Clone,
            Group = Infallible,
        >,
    > + PlainTextTokens<<Self as TextGenerationBackend>::InputTensor>
    + TextSamplingNative<
        Self,
        SamplingCompletion = <Self as TextGenerationBackend>::TextCompletion,
        ScoreRow = Vec<f32>,
        Sample = u32,
        Token = <Self as TextGenerationBackend>::Token,
    > + TextTransactionMechanisms<Error = BackendFailure>
    + 'static
{
}

impl<B> LocalCausalBackend for B where
    B: TextGenerationBackend<Error = BackendFailure>
        + HostCausalVerification<Values = <B as TextSessionNative<B>>::TextValues>
        + TextSessionNative<
            B,
            Generation = CausalTextGeneration,
            Retained: 'static,
            TextValues: NativeTextValues<
                InputTensor = <B as TextGenerationBackend>::InputTensor,
                Embedded = <B as HostCausalVerification>::Embedded,
                Context: 'static,
                Scores: Clone,
                Group = Infallible,
            >,
        > + PlainTextTokens<B::InputTensor>
        + TextSamplingNative<
            B,
            SamplingCompletion = <B as TextGenerationBackend>::TextCompletion,
            ScoreRow = Vec<f32>,
            Sample = u32,
            Token = <B as TextGenerationBackend>::Token,
        > + TextTransactionMechanisms<Error = BackendFailure>
        + 'static
{
}

/// Loaded causal session with eager local state and scores.
pub type LocalCausalSession<B> =
    TextSessionDriver<<B as TextSessionMechanisms<B>>::Resources, B, CausalTextGeneration>;
type Runtime<B> = ModelRuntime<B, LocalCausalSession<B>>;
type Queue<B> = <<B as HostCausalVerification>::Values as NativeTextValues>::Context;

type Context<'a, B> = crate::capture::SpeculativeContextFor<'a, LocalCausalCapture<B>>;

type Assistant<B> = <B as HostCausalVerification>::Assistant;
type Binder<B> = <Assistant<B> as ExternalAssistantOperators>::Binder;
type ExecutorFor<B> = AssistantExecutorTypes<Assistant<B>>;

enum DraftPayload<B: LocalCausalBackend> {
    Independent(
        Box<dyn IndependentDraftModel<crate::executable_binding::PortableTextTypes<B::Values>>>,
    ),
    Conditioned {
        assistant: Box<dyn ErasedLocalAssistant<B>>,
        selected: crate::SelectedSpeculativeRealization,
    },
}

trait ErasedLocalAssistant<B: LocalCausalBackend> {
    fn memory_profile(
        &self,
        memory: eredu_core::capability::ParameterMemoryFacts,
    ) -> Result<Option<crate::memory_forecast::ConditionedAssistantMemoryProfile>, BackendFailure>;

    fn with_executor(
        &mut self,
        target: &mut ExternalAssistantTarget<Assistant<B>>,
        run: &mut dyn FnMut(&mut dyn ErasedGenerationExecutor<ExecutorFor<B>, TextPromptFor<B>>),
    );
}

impl<B, D> ErasedLocalAssistant<B> for BoundAssistant<D>
where
    B: LocalCausalBackend,
    D: BoundAssistantDispatch<Binder<B>>,
{
    fn memory_profile(
        &self,
        memory: eredu_core::capability::ParameterMemoryFacts,
    ) -> Result<Option<crate::memory_forecast::ConditionedAssistantMemoryProfile>, BackendFailure>
    {
        self.memory_profile::<Binder<B>>(
            memory,
            crate::memory_estimation::MemoryBytes::unknown(
                "native conditioned scratch allowance is unavailable",
            ),
        )
    }

    fn with_executor(
        &mut self,
        target: &mut ExternalAssistantTarget<Assistant<B>>,
        run: &mut dyn FnMut(&mut dyn ErasedGenerationExecutor<ExecutorFor<B>, TextPromptFor<B>>),
    ) {
        let capture = self.capture().clone();
        self.dispatch_mut()
            .visit_executor(LocalAssistantVisitor::<B> {
                target,
                capture,
                run,
            });
    }
}

struct LocalAssistantVisitor<'a, B: LocalCausalBackend> {
    target: &'a mut ExternalAssistantTarget<Assistant<B>>,
    capture: ExternalPredictionCaptureRequest,
    run: &'a mut dyn FnMut(&mut dyn ErasedGenerationExecutor<ExecutorFor<B>, TextPromptFor<B>>),
}

impl<B: LocalCausalBackend> SelectedAssistantExecutionVisitor<Binder<B>>
    for LocalAssistantVisitor<'_, B>
{
    type Output = ();

    fn visit<A, D>(
        self,
        assistant: &mut crate::speculative::InstalledExternalAssistant<Binder<B>, A>,
        dispatch: D,
    ) where
        A: ExternalAssistantConstruction,
        D: SelectedAssistantExecutor<A>,
    {
        dispatch.execute::<Assistant<B>, _>(
            self.target,
            assistant,
            self.capture,
            LocalAssistantRunner::<B> { run: self.run },
        );
    }
}

struct LocalAssistantRunner<'a, B: LocalCausalBackend> {
    run: &'a mut dyn FnMut(&mut dyn ErasedGenerationExecutor<ExecutorFor<B>, TextPromptFor<B>>),
}

impl<A: ExternalAssistantConstruction, B: LocalCausalBackend>
    ExternalAssistantExecutorVisitor<A, Assistant<B>> for LocalAssistantRunner<'_, B>
{
    type Output = ();

    fn execute<'run, E>(self, executor: &'run mut E)
    where
        Self: 'run,
        E: SpeculativeCacheFactory<
                crate::SelectedSpeculativeRealization,
                Input = ExternalAssistantPrompt<Assistant<B>>,
                Cache = ExternalAssistantCache<
                    <Assistant<B> as ExternalAssistantExecutionMechanisms>::NativeCache,
                >,
                Logits = <Assistant<B> as ExternalAssistantOperators>::Logits,
                Context<'run> = Context<'run, B>,
                Completion = <Assistant<B> as ExternalAssistantOperators>::Completion,
                Telemetry = (),
                Error = BackendFailure,
            > + GenerationExecutorCacheFactory<ExecutorFor<B>>
            + 'run,
        E::Cache: 'static,
        E::TargetState: 'static,
        E::DraftState: 'static,
        E::CacheCheckpoint: 'static,
        E::Verification: 'static,
    {
        (self.run)(executor);
    }
}

/// Tokenizer-compatible independent draft retaining its exact native context.
pub struct LocalCausalDraft<B: LocalCausalBackend> {
    retained: EagerRetainedPayload<DraftPayload<B>, Queue<B>>,
    target_identity: String,
    capacity: NonZeroUsize,
    topology: SpeculativeExecutionTopology,
}

impl<B, D>
    DraftSessionInstallation<
        B,
        Binder<B>,
        EagerRetainedPayload<
            InstalledExternalDraft<D, crate::executable_binding::PortableTextTypes<B::Values>>,
            Queue<B>,
        >,
    > for CausalTextGeneration
where
    B: LocalCausalBackend,
    D: BoundAssistantDispatch<Binder<B>> + 'static,
{
    type Draft = LocalCausalDraft<B>;

    fn install(
        draft: InstalledDraftSession<
            Binder<B>,
            EagerRetainedPayload<
                InstalledExternalDraft<D, crate::executable_binding::PortableTextTypes<B::Values>>,
                Queue<B>,
            >,
        >,
    ) -> Result<Self::Draft, eredu_core::AutomaticPlanningError> {
        let capacity = draft
            .selected()
            .requirements()
            .strategy()
            .proposal_capacity();
        let target_identity = draft.installed().target_identity().to_owned();
        let topology = draft.selected().placement().topology();
        let retained = draft
            .into_resources()
            .map(|draft| match draft.into_parts().0 {
                ExternalDraftExecution::Autoregressive { model, .. } => {
                    DraftPayload::Independent(model)
                }
                ExternalDraftExecution::Assistant(assistant) => DraftPayload::Conditioned {
                    selected: assistant.selected().clone(),
                    assistant: Box::new(assistant),
                },
            });
        Ok(LocalCausalDraft {
            retained,
            target_identity,
            capacity,
            topology,
        })
    }
}

/// Executes an admitted target-only or independently drafted batch with one scheduler.
pub fn execute_local_causal_batch<B, C, V>(
    target: &mut Runtime<B>,
    draft: Option<&mut LocalCausalDraft<B>>,
    lanes: Vec<SpeculativeGenerationLane<'_, TextPromptFor<B>, C>>,
    visitor: V,
) -> Result<SpeculativeGenerationBatchOutput, BackendFailure>
where
    B: LocalCausalBackend,
    C: SpeculativeTokenFilterController + 'static,
    V: SpeculativeGenerationVisitor,
{
    let capacity = draft.as_ref().map_or(0, |draft| draft.capacity.get());
    for lane in &lanes {
        validate_lane_proposal_capacity(lane.config(), capacity)
            .map_err(BackendFailure::from_error)?;
    }
    let timings = component_timing_enabled();
    let queue = B::context(target.backend()).clone();
    let identity = target.session().intervention_identity();
    if let Some(draft) = draft {
        if draft.target_identity != identity {
            return Err(failure(
                BackendFailureKind::InvalidSession,
                "independent draft belongs to a different target",
            ));
        }
        let topology = draft.topology;
        let draft_queue = draft.retained.retain_queue();
        let context = Context::<B>::bind(&queue, &draft_queue, topology)?;
        let mut operation = draft.retained.begin_operation()?;
        let result = (|| match operation.payload_mut() {
            DraftPayload::Independent(draft_model) => {
                let lanes = prepare_independent::<B, C>(lanes)?;
                target.session_mut().with_model_transaction(|target| {
                    let mut executor = AutoregressiveExecutor::<LocalCausalMechanisms<B>, _>::new(
                        target.as_mut(),
                        draft_model.executable_mut(),
                        draft.capacity,
                        NoTargetRefinement,
                    );
                    run_speculative_batch(
                        &mut executor,
                        lanes,
                        &(),
                        context,
                        topology,
                        timings,
                        visitor,
                    )
                    .map_err(BackendFailure::from_error)
                })
            }
            DraftPayload::Conditioned {
                assistant,
                selected,
                ..
            } => execute_conditioned::<B, C, V>(
                target,
                assistant.as_mut(),
                selected,
                lanes,
                context,
                timings,
                visitor,
            ),
        })();
        operation.settle(result)
    } else {
        let topology = SpeculativeExecutionTopology::Single;
        let lanes = prepare_independent::<B, C>(lanes)?;
        let context = Context::<B>::single(&queue);
        target.session_mut().with_model_transaction(|target| {
            let mut executor = CausalExecutor::<LocalCausalMechanisms<B>, _>::new(
                target.as_mut(),
                NoTargetRefinement,
            );
            run_speculative_batch(
                &mut executor,
                lanes,
                &(),
                context,
                topology,
                timings,
                visitor,
            )
            .map_err(BackendFailure::from_error)
        })
    }
}

/// Executes masked proposals on the target's installed backbone using the shared batch scheduler.
pub fn execute_local_shared_batch<B, C, V>(
    runtime: &mut Runtime<B>,
    lanes: Vec<SpeculativeGenerationLane<'_, TextPromptFor<B>, C>>,
    visitor: V,
) -> Result<SpeculativeGenerationBatchOutput, BackendFailure>
where
    B: LocalCausalBackend,
    C: SpeculativeTokenFilterController + 'static,
    V: SpeculativeGenerationVisitor,
{
    use eredu_core::backend::TextOperationDriver;
    if !runtime.session().decoding_capabilities().speculation {
        return Err(failure(
            BackendFailureKind::Unsupported,
            "shared-backbone drafting is unavailable for the selected execution",
        ));
    }
    let queue = B::context(runtime.backend()).clone();
    let context = Context::<B>::single(&queue);
    runtime.session_mut().with_model_transaction(|model| {
        let (mut executor, lanes) = super::batch::prepare_shared_batch::<
            LocalCausalMechanisms<B>,
            B,
            _,
            _,
            _,
            BackendFailure,
        >(
            model.as_mut(),
            lanes,
            context,
            prepare_host_sampling::<B, C>,
        )?;
        run_speculative_batch(
            &mut executor,
            lanes,
            &(),
            context,
            SpeculativeExecutionTopology::Single,
            component_timing_enabled(),
            visitor,
        )
        .map_err(BackendFailure::from_error)
    })
}

type LocalSampler<B, C, R = crate::speculative::host_sampling::OwnedHostScores> =
    HostSpeculativeSampling<ConstrainedSampler<TextSampler, C>, LocalCausalCapture<B>, R>;

fn prepare_host_sampling<B: LocalCausalBackend, C: SpeculativeTokenFilterController + 'static>(
    config: eredu_core::TextGenerationConfig,
    constraint: C,
) -> Result<(Option<u64>, LocalSampler<B, C>), BackendFailure> {
    let seed = (config.sampling().temperature != 0.0).then(|| config.seed());
    let sampler = TextSampler::from_config(&config).map_err(BackendFailure::from_error)?;
    Ok((
        seed,
        HostSpeculativeSampling::new(ConstrainedSampler::new(sampler, constraint)),
    ))
}

fn prepare_independent<'a, B: LocalCausalBackend, C: SpeculativeTokenFilterController + 'static>(
    lanes: Vec<SpeculativeGenerationLane<'a, TextPromptFor<B>, C>>,
) -> Result<
    Vec<
        crate::speculative::SpeculativeBatchLane<
            'a,
            AutoregressiveInput<TextPromptFor<B>, ()>,
            LocalSampler<B, C>,
            u64,
        >,
    >,
    BackendFailure,
> {
    lanes
        .into_iter()
        .map(|lane| {
            prepare_batch_lane(
                lane,
                |input, _, _| Ok(AutoregressiveInput::new(input, ())),
                prepare_host_sampling::<B, C>,
            )
        })
        .collect()
}

/// Executes the predictor retained by the shared prepared text binder.
pub fn execute_local_embedded_batch<B, C, V>(
    runtime: &mut Runtime<B>,
    lanes: Vec<SpeculativeGenerationLane<'_, TextPromptFor<B>, C>>,
    visitor: V,
) -> Result<SpeculativeGenerationBatchOutput, BackendFailure>
where
    B: LocalCausalBackend,
    C: SpeculativeTokenFilterController + 'static,
    V: SpeculativeGenerationVisitor,
{
    let queue = B::context(runtime.backend()).clone();
    let context = Context::<B>::single(&queue);
    runtime.session_mut().with_model_transaction(|model| {
        let model: &mut dyn crate::executable_binding::ErasedTextExecutable<
            crate::executable_binding::PortableTextTypes<<B as HostCausalVerification>::Values>,
        > = model.as_mut();
        let mut executor = model.bind_embedded_prediction().ok_or_else(|| {
            failure(
                BackendFailureKind::Unsupported,
                "installed target has no embedded predictor",
            )
        })?;
        for lane in &lanes {
            validate_lane_proposal_capacity(lane.config(), executor.max_proposals())
                .map_err(BackendFailure::from_error)?;
        }
        let lanes = prepare_native_lanes::<B, C>(lanes)?;
        run_speculative_batch(
            &mut DynGenerationExecutor::new(executor.as_mut()),
            lanes,
            &(),
            context,
            SpeculativeExecutionTopology::Single,
            component_timing_enabled(),
            visitor,
        )
        .map_err(BackendFailure::from_error)
    })
}

type NativeSampler<B, C> = LocalSampler<B, C, crate::speculative::host_sampling::NativeHostScores>;

fn prepare_native_lanes<'a, B, C>(
    lanes: Vec<SpeculativeGenerationLane<'a, TextPromptFor<B>, C>>,
) -> Result<
    Vec<crate::speculative::SpeculativeBatchLane<'a, TextPromptFor<B>, NativeSampler<B, C>, u64>>,
    BackendFailure,
>
where
    B: LocalCausalBackend,
    C: SpeculativeTokenFilterController + 'static,
{
    lanes
        .into_iter()
        .map(|lane| {
            prepare_batch_lane(
                lane,
                |input, _, _| Ok(input),
                |config, constraint| {
                    let seed = (config.sampling().temperature != 0.0).then(|| config.seed());
                    let sampler =
                        TextSampler::from_config(&config).map_err(BackendFailure::from_error)?;
                    Ok((
                        seed,
                        NativeSampler::<B, C>::new(ConstrainedSampler::new(sampler, constraint)),
                    ))
                },
            )
        })
        .collect()
}

fn execute_conditioned<B, C, V>(
    runtime: &mut Runtime<B>,
    assistant: &mut dyn ErasedLocalAssistant<B>,
    selected: &crate::SelectedSpeculativeRealization,
    lanes: Vec<SpeculativeGenerationLane<'_, TextPromptFor<B>, C>>,
    context: Context<'_, B>,
    timings: bool,
    visitor: V,
) -> Result<SpeculativeGenerationBatchOutput, BackendFailure>
where
    B: LocalCausalBackend,
    C: SpeculativeTokenFilterController + 'static,
    V: SpeculativeGenerationVisitor,
{
    let lanes = prepare_native_lanes::<B, C>(lanes)?;
    let mut execution = Some((lanes, visitor));
    let mut result = None;
    runtime.session_mut().with_model_transaction(|model| {
        let target = model.as_mut().external_prediction_mut().ok_or_else(|| {
            failure(
                BackendFailureKind::Unsupported,
                "installed target has no conditioned capture binding",
            )
        })?;
        assistant.with_executor(target, &mut |executor| {
            let (lanes, visitor) = execution
                .take()
                .expect("conditioned dispatch invokes one executor");
            result = Some(
                run_speculative_batch(
                    &mut DynGenerationExecutor::new(executor),
                    lanes,
                    selected,
                    context,
                    selected.placement().topology(),
                    timings,
                    visitor,
                )
                .map_err(BackendFailure::from_error),
            );
        });
        result.take().ok_or_else(|| {
            failure(
                BackendFailureKind::InvalidSession,
                "conditioned dispatch supplied no executor",
            )
        })?
    })
}

fn failure(kind: BackendFailureKind, message: &'static str) -> BackendFailure {
    BackendFailure::new(kind, std::io::Error::other(message))
}

impl<B> super::SpeculativeGenerationBackend<LocalCausalSession<B>> for B
where
    B: LocalCausalBackend,
{
    type Drafter = LocalCausalDraft<B>;

    #[cfg(feature = "tensor-capture")]
    fn speculative_activation_discovery(
        runtime: &Runtime<B>,
    ) -> Result<
        eredu_core::speculative::SpeculativeActivationDiscovery,
        eredu_core::capture::CaptureError,
    > {
        use eredu_core::capture::{CaptureError, SpeculativeCapturePrimitives};
        let session = runtime.session();
        let discovery = session.model_discovery().ok_or_else(|| {
            CaptureError::Unsupported("local session has no retained activation catalog".into())
        })?;
        let execution = session
            .resources()
            .model
            .as_ref()
            .partition_activation_execution()
            .ok_or_else(|| {
                CaptureError::Unsupported(
                    "selected local executor has no internal activation path".into(),
                )
            })?;
        discovery.speculative_activations(
            &execution,
            &LocalCausalCapture::<B>::intervention_mechanisms(),
            session.intervention_identity(),
            session.parameter_overlay_identity(),
        )
    }

    #[cfg(feature = "tensor-capture")]
    fn validate_speculative_activations(
        runtime: &Runtime<B>,
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        use eredu_core::capture::SpeculativeCapturePrimitives;
        plan.validate(&Self::speculative_activation_discovery(runtime)?)?;
        crate::intervention::preflight(
            plan.captures(),
            plan.interventions(),
            &*LocalCausalCapture::<B>::intervention_estimator(),
        )
    }

    #[cfg(feature = "tensor-capture")]
    fn validate_speculative_capture(
        runtime: &Runtime<B>,
        plan: &eredu_core::capture::AdmittedCapturePlan,
    ) -> Result<(), eredu_core::capture::CaptureError> {
        super::validate_control_capture(plan)?;
        runtime.session().validate_local_capture::<B>(plan)
    }
}

impl<B> crate::text_session_driver::forecast::DraftForecastFacts for LocalCausalDraft<B>
where
    B: LocalCausalBackend + eredu_core::capability::NativeExecutionMemory<B>,
{
    fn forecast_proposal_capacity(&self) -> u64 {
        self.capacity.get() as u64
    }

    fn conditioned_forecast(
        &self,
    ) -> Result<
        Option<crate::memory_forecast::ConditionedAssistantMemoryProfile>,
        crate::memory_forecast::GenerationForecastError,
    > {
        match self.retained.payload() {
            DraftPayload::Independent(_) => Ok(None),
            DraftPayload::Conditioned { assistant, .. } => {
                Ok(assistant
                    .memory_profile(B::verification_memory(&self.retained.retain_queue())?)?)
            }
        }
    }

    fn independent_forecast(
        &self,
    ) -> Result<
        Option<crate::memory_forecast::LoadedMemoryProfile>,
        crate::memory_forecast::GenerationForecastError,
    > {
        let DraftPayload::Independent(model) = self.retained.payload() else {
            return Ok(None);
        };
        crate::text_session_driver::forecast::independent_draft_profile(
            model.as_ref(),
            B::verification_memory(&self.retained.retain_queue())?,
        )
        .map(Some)
    }
}
