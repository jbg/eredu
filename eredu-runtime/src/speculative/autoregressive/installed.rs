//! Portable invocation of installed text executables for independent and shared drafting.
use std::rc::Rc;

use eredu_core::refinement::RefinementBackend;
use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};

use crate::{
    capture::SpeculativeCaptureMechanisms,
    executable_binding::ErasedTextExecutable,
    generation::refinement::{architecture::BlockRefinementGeometry, RefinementValue},
    text_session_driver::TextPrefillInput,
    ActivationObserver,
};
use eredu_core::backend::TextExecutableTypes;
use eredu_core::speculative::installed::InstalledSpeculationOperators;

use super::{
    causal_invocation::InstalledCausalState, AutoregressiveMechanisms, AutoregressivePass,
};
use crate::speculative::shared_backbone::{SharedBackboneMechanisms, SharedRefinementState};

type Types<M> =
    crate::executable_binding::PortableTextTypes<<M as InstalledSpeculationOperators>::Text>;
type Scores<M> = <Types<M> as TextExecutableTypes>::Scores;
type Queue<M> = <Types<M> as TextExecutableTypes>::Context;
type Tensor<M> = <Types<M> as TextExecutableTypes>::Tensor;
type InvocationTensor<M> = <Types<M> as TextExecutableTypes>::InvocationTensor;
type InvocationError<M> = <Types<M> as TextExecutableTypes>::Error;
type Context<'a, M> =
    crate::capture::SpeculativeContextFor<'a, <M as InstalledSpeculationOperators>::Capture>;
type Refinement<M> = <M as InstalledSpeculationOperators>::Refinement;
type Refined<M> = <Refinement<M> as RefinementBackend>::Logits;
type Token<M> = <Refinement<M> as RefinementBackend>::Token;
type Key<M> = <Refinement<M> as RefinementBackend>::RandomState;

fn refinement_root<M: InstalledSpeculationPrimitives>(
    value: RefinementValue<'_, Refinement<M>>,
) -> &M::RetainedRoot {
    match value {
        RefinementValue::Tokens(value) => M::refinement_token_root(value),
        RefinementValue::Scores(value) => M::refinement_score_root(value),
        RefinementValue::Mask(value) => M::refinement_mask_root(value),
        RefinementValue::Random(value) => M::refinement_key_root(value),
    }
}

/// Scoped observer mapping supplied by portable composition for one target invocation.
pub trait SpeculativeTargetObserver<T, E, O> {
    /// Lends the already mapped observation hooks for this operation.
    fn with_observer(
        &self,
        run: &mut dyn FnMut(&mut dyn ActivationObserver<T, E>) -> Result<O, E>,
    ) -> Result<O, E>;
}

/// Portable borrow of target observation with a neutral tensor mapping.
pub struct MappedSpeculativeTargetObserver<'a, M: eredu_core::observation::ObserverTensorMapping> {
    observer: std::cell::RefCell<
        crate::inspection::MappedActivationObserver<
            'a,
            dyn ActivationObserver<M::ObserverTensor, M::ObserverError> + 'a,
            M,
        >,
    >,
}

impl<'a, M: eredu_core::observation::ObserverTensorMapping> MappedSpeculativeTargetObserver<'a, M> {
    /// Binds observation to the already selected tensor mapping.
    pub fn new(
        observer: &'a mut dyn ActivationObserver<M::ObserverTensor, M::ObserverError>,
    ) -> Self {
        Self {
            observer: std::cell::RefCell::new(crate::inspection::MappedActivationObserver::new(
                observer,
            )),
        }
    }
}

impl<M: eredu_core::observation::ObserverTensorMapping, O>
    SpeculativeTargetObserver<M::InvocationTensor, M::InvocationError, O>
    for MappedSpeculativeTargetObserver<'_, M>
{
    fn with_observer(
        &self,
        run: &mut dyn FnMut(
            &mut dyn ActivationObserver<M::InvocationTensor, M::InvocationError>,
        ) -> Result<O, M::InvocationError>,
    ) -> Result<O, M::InvocationError> {
        run(&mut *self.observer.borrow_mut())
    }
}

/// Native operators with the transport required by the portable speculative driver.
pub trait InstalledSpeculationPrimitives:
    InstalledSpeculationOperators<
    Capture: for<'a> SpeculativeCaptureMechanisms<
        Context<'a> = crate::capture::SpeculativeContextFor<'a, Self::Capture>,
        Queue = Queue<Self>,
        Provider: crate::capture::CaptureBackendProvider<Tensor = Tensor<Self>>,
        Error = Self::Error,
    >,
>
{
}

impl<M> InstalledSpeculationPrimitives for M
where
    M: InstalledSpeculationOperators,
    M::Capture: for<'a> SpeculativeCaptureMechanisms<
        Context<'a> = crate::capture::SpeculativeContextFor<'a, M::Capture>,
        Queue = Queue<M>,
        Provider: crate::capture::CaptureBackendProvider<Tensor = Tensor<M>>,
        Error = M::Error,
    >,
{
}

/// Retained verification scores paired with their selected queue.
pub struct InstalledCausalScores<M: InstalledSpeculationPrimitives> {
    scores: Scores<M>,
    queue: Queue<M>,
}

impl<M: InstalledSpeculationPrimitives> Clone for InstalledCausalScores<M> {
    fn clone(&self) -> Self {
        Self {
            scores: self.scores.clone(),
            queue: self.queue.clone(),
        }
    }
}

fn observe<M: InstalledSpeculationPrimitives, T>(
    observer: Option<&mut dyn ActivationObserver<Tensor<M>, M::Error>>,
    execute: impl FnOnce(
        &mut dyn ActivationObserver<InvocationTensor<M>, InvocationError<M>>,
    ) -> Result<T, InvocationError<M>>,
) -> Result<T, M::Error> {
    let Some(observer) = observer else {
        return execute(&mut crate::NoopObserver).map_err(M::execution_error);
    };
    let mut bridge = crate::inspection::ObserverErrorBridge::new(
        observer,
        M::execution_error,
        M::observer_error,
    );
    let mut mapped = crate::inspection::MappedActivationObserver::<
        _,
        <Types<M> as TextExecutableTypes>::ObserverMapping,
    >::new(&mut bridge);
    let result = execute(&mut mapped).map_err(M::execution_error);
    bridge.resolve(result)
}

impl<M: InstalledSpeculationPrimitives> AutoregressiveMechanisms for M {
    type Model = dyn ErasedTextExecutable<Types<M>>;
    type Input = crate::text_session_driver::PreparedTextPrompt<
        <Types<M> as TextExecutableTypes>::InputTensor,
    >;
    type State = InstalledCausalState<M::Text>;
    type Checkpoint = Rc<Self::State>;
    type Output = InstalledCausalScores<M>;
    type Logits = Scores<M>;
    type Context<'a> = Context<'a, M>;
    type Completion = M::Completion;
    type Error = M::Error;

    fn coordinate_speculative_step(
        local: Vec<eredu_core::SpeculativeScheduleState>,
        context: Self::Context<'_>,
    ) -> Result<Vec<eredu_core::SpeculativeScheduleState>, BackendFailure> {
        M::Capture::coordinate_speculative_step(&context, local)
    }
    fn agree_text_preparation(
        stage: eredu_core::run_preparation::TextPreparationStage,
        status: eredu_core::run_preparation::TextPreparationStatus,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::run_preparation::TextPreparationOutcome, BackendFailure> {
        M::Capture::agree_text_preparation(&context, stage, status)
    }
    fn coordinate_decoding_control(
        local: eredu_core::decoding::DecodingCoordination,
        context: Self::Context<'_>,
    ) -> Result<eredu_core::decoding::DecodingCoordination, BackendFailure> {
        M::Capture::coordinate_decoding_control(&context, local)
    }
    fn verification_output_bytes(output: &Self::Output) -> Option<u64> {
        M::verification_bytes(&output.scores)
    }

    fn logits_snapshot_estimate(logits: &Self::Logits) -> Option<SnapshotEstimate> {
        M::logits_estimate(logits)
    }
    fn empty(
        model: &mut Self::Model,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<Self::State, Self::Error> {
        let queue = match pass {
            AutoregressivePass::DraftPrefill => context.draft(),
            _ => context.target(),
        };
        InstalledCausalState::allocate(model, queue).map_err(M::execution_error)
    }
    fn prefill(
        model: &mut Self::Model,
        input: &Self::Input,
        state: &mut Self::State,
        _: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<(Self::Output, usize), Self::Error> {
        let mut positions = 0;
        let mut execute =
            |observer: &mut dyn ActivationObserver<InvocationTensor<M>, InvocationError<M>>| {
                let (scores, count) = state.prefill(model, input.borrowed(), observer)?;
                positions = count;
                Ok(scores)
            };
        let scores = match context.target_observer() {
            Some(observer) => observer.with_observer(&mut execute),
            None => execute(&mut crate::NoopObserver),
        }
        .map_err(M::execution_error)?;
        Ok((
            InstalledCausalScores {
                scores,
                queue: state.queue.clone(),
            },
            positions,
        ))
    }
    fn decode(
        model: &mut Self::Model,
        tokens: &[u32],
        state: &mut Self::State,
        _: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<Self::Output, Self::Error> {
        let tokens = M::causal_tokens(tokens)?;
        let mut execute = |observer: &mut dyn ActivationObserver<
            InvocationTensor<M>,
            InvocationError<M>,
        >| { state.append(model, &tokens, observer) };
        let scores = match context.target_observer() {
            Some(observer) => observer.with_observer(&mut execute),
            None => execute(&mut crate::NoopObserver),
        }
        .map_err(M::execution_error)?;
        Ok(InstalledCausalScores {
            scores,
            queue: state.queue.clone(),
        })
    }
    fn checkpoint(state: &Self::State) -> Result<Self::Checkpoint, Self::Error> {
        state.checkpoint().map_err(M::execution_error)
    }
    fn restore(saved: &Self::Checkpoint, _: Self::Context<'_>) -> Result<Self::State, Self::Error> {
        saved.copy_boundary().map_err(M::execution_error)
    }
    fn retain_verified_prefix(
        state: &mut Self::State,
        saved: &Self::Checkpoint,
        accepted: usize,
        context: Self::Context<'_>,
    ) -> Result<bool, Self::Error> {
        state
            .retain_verified_prefix(saved, accepted, context.target())
            .map_err(M::execution_error)
    }
    fn estimate(saved: &Self::Checkpoint) -> Option<SnapshotEstimate> {
        saved.estimate()
    }
    fn checkpoint_retained_bytes(saved: &Self::Checkpoint) -> Option<u64> {
        saved.retained_bytes()
    }
    fn estimate_state(state: &Self::State) -> Option<SnapshotEstimate> {
        state.estimate()
    }
    fn trim_parameter_conversions(
        model: &mut Self::Model,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        model.trim_parameter_conversions()
    }
    fn parameter_conversion_retention(
        model: &Self::Model,
    ) -> Result<
        eredu_core::Observed<Vec<eredu_core::residency::ParameterConversionRetentionReport>>,
        BackendFailure,
    > {
        model.parameter_conversion_retention()
    }
    fn memory_observation(
        model: &Self::Model,
        state: &Self::State,
        additional: u64,
    ) -> Result<Option<eredu_core::speculative::SpeculativeModelMemoryObservation>, BackendFailure>
    {
        super::memory_observation::model(
            state.cache.memory_bounds(additional)?,
            model
                .residency_report()
                .map_err(BackendFailure::from_error)?,
            model
                .parameter_bank_report()
                .map_err(BackendFailure::from_error)?,
            M::placement_memory(&state.queue)?,
        )
        .map(Some)
    }
    fn logits(
        output: &Self::Output,
        position: usize,
        _: Self::Context<'_>,
    ) -> Result<Self::Logits, Self::Error> {
        M::score_row(&output.scores, position, &output.queue)
    }
    fn completion(
        output: &Self::Output,
        _: Self::Context<'_>,
    ) -> Result<Self::Completion, Self::Error> {
        M::submit(&output.scores)
    }
    fn invalid(message: &'static str) -> Self::Error {
        M::invalid(message)
    }
}

impl<M: InstalledSpeculationPrimitives> SharedBackboneMechanisms for M {
    type ActivationTensor = Tensor<M>;
    type Capture = M::Capture;
    type ProposalRandomness = Key<M>;
    type Refinement = Refinement<M>;
    type Queue = Queue<M>;

    fn reborrow_context<'a: 'b, 'b>(context: Self::Context<'a>) -> Self::Context<'b> {
        context
    }
    fn publish_committed(
        model: &mut Self::Model,
        state: &mut Self::State,
    ) -> Result<(), Self::Error> {
        model
            .exchange_autoregressive_cache(&mut state.cache, &state.queue)
            .map_err(M::execution_error)
    }
    fn activation_prefill_plans(
        model: &Self::Model,
        input: &Self::Input,
    ) -> Result<Vec<crate::executable_binding::retained_session::PreparedInputPartPlan>, Self::Error>
    {
        model
            .prepared_input_plans(input.borrowed())
            .map_err(M::execution_error)
    }
    fn prefill_with_activation(
        model: &mut Self::Model,
        input: &Self::Input,
        state: &mut Self::State,
        observer: &mut dyn ActivationObserver<Tensor<M>, Self::Error>,
    ) -> Result<(Self::Output, usize), Self::Error> {
        let (scores, count) = observe::<M, _>(Some(observer), |observer| {
            model.prepared_context_forward(
                input.borrowed(),
                &mut state.cache,
                &state.queue,
                observer,
            )
        })?;
        Ok((
            InstalledCausalScores {
                scores,
                queue: state.queue.clone(),
            },
            count,
        ))
    }
    fn decode_with_activation(
        model: &mut Self::Model,
        tokens: &[u32],
        state: &mut Self::State,
        observer: &mut dyn ActivationObserver<Tensor<M>, Self::Error>,
    ) -> Result<Self::Output, Self::Error> {
        let tokens = M::causal_tokens(tokens)?;
        let scores = observe::<M, _>(Some(observer), |observer| {
            model.attention_forward_with_observer(
                &tokens,
                None,
                eredu_core::decoding::AttentionInvocation::CAUSAL,
                Some(&mut state.cache),
                false,
                &state.queue,
                observer,
            )
        })?;
        Ok(InstalledCausalScores {
            scores,
            queue: state.queue.clone(),
        })
    }
    fn ordinary_refinement_policy(
        model: &Self::Model,
    ) -> Option<crate::generation::refinement::architecture::RefinementExecutionPolicy> {
        model.capability_estimate().ordinary_refinement()
    }
    fn shared_backbone_policy(
        model: &Self::Model,
    ) -> Option<crate::speculative::shared_backbone::SharedBackboneExecutionPolicy> {
        model.capability_estimate().shared_backbone_draft()
    }
    fn block_geometry(model: &Self::Model) -> Option<BlockRefinementGeometry> {
        model.block_refinement_geometry()
    }
    fn target_refinement_context(context: Self::Context<'_>) -> &Self::Queue {
        context.target()
    }
    fn draft_refinement_context(context: Self::Context<'_>) -> Option<&Self::Queue> {
        (!context.crosses_devices()).then(|| context.draft())
    }
    fn execution_error(error: impl std::error::Error + Send + Sync + 'static) -> Self::Error {
        M::execution_error(error)
    }
    fn score_format(
        logits: &Self::Logits,
    ) -> <Self::Refinement as RefinementBackend>::ConditioningFormat {
        M::score_format(logits)
    }
    fn draft_score_format() -> <Self::Refinement as RefinementBackend>::ConditioningFormat {
        M::draft_score_format()
    }
    fn refinement_scores(logits: &Self::Logits) -> Refined<M> {
        M::refinement_scores(logits)
    }
    fn evaluate_refinement(
        model: &mut Self::Model,
        state: &mut Self::State,
        canvas: &Token<M>,
        feedback: Option<&Refined<M>>,
        geometry: BlockRefinementGeometry,
        invocation: eredu_core::decoding::AttentionInvocation,
        queue: &Self::Queue,
        observer: Option<&mut dyn ActivationObserver<Tensor<M>, Self::Error>>,
    ) -> Result<Refined<M>, Self::Error> {
        let feedback_shape = geometry.feedback_shape();
        let (canvas, feedback) = M::refinement_inputs(
            canvas,
            &geometry.canvas_shape(),
            feedback.map(|value| (value, feedback_shape.as_slice())),
            queue,
        )?;
        let scores = observe::<M, _>(observer, |observer| {
            model.attention_forward_with_observer(
                &canvas,
                feedback.as_ref(),
                invocation,
                Some(&mut state.cache),
                false,
                queue,
                observer,
            )
        })?;
        M::refinement_output(scores, &geometry.position_score_shape(), queue)
    }
    fn read_refinement_tokens(
        tokens: &Token<M>,
        range: std::ops::Range<usize>,
        queue: &Self::Queue,
    ) -> Result<Vec<u32>, Self::Error> {
        M::read_refinement_tokens(tokens, range, queue)
    }
    fn refinement_score_row(
        scores: &Refined<M>,
        position: usize,
        queue: &Self::Queue,
    ) -> Result<Self::Logits, Self::Error> {
        M::refinement_score_row(scores, position, queue)
    }
    fn settle_refinement(state: &SharedRefinementState<Self>) -> Result<(), Self::Error> {
        let mut roots = Vec::new();
        state.visit_retained_values(|value| roots.push(refinement_root::<M>(value).clone()));
        M::settle_refinement_roots(roots)
    }
    fn refinement_state_estimate(state: &SharedRefinementState<Self>) -> Option<SnapshotEstimate> {
        let host = state.snapshot_host_bytes()?;
        let mut estimate = Some(SnapshotEstimate {
            retained_bytes: host,
            copy_bytes: host,
        });
        state.visit_retained_values(|value| {
            estimate = estimate.and_then(|sum| {
                let value = M::refinement_root_estimate(refinement_root::<M>(value))?;
                Some(SnapshotEstimate {
                    retained_bytes: sum.retained_bytes.checked_add(value.retained_bytes)?,
                    copy_bytes: sum.copy_bytes.checked_add(value.copy_bytes)?,
                })
            });
        });
        estimate
    }
    fn model_randomness_root(seed: u64) -> Result<Self::ProposalRandomness, Self::Error> {
        M::randomness_root(seed)
    }
    fn split_model_randomness(
        root: &Self::ProposalRandomness,
        substream: usize,
        context: Self::Context<'_>,
    ) -> Result<Self::ProposalRandomness, Self::Error> {
        M::split_randomness(root, substream, context.target())
    }
    fn settle_model_randomness(
        randomness: &[&Self::ProposalRandomness],
    ) -> Result<(), Self::Error> {
        M::settle_randomness(randomness)
    }
    fn proposal_randomness_bytes(randomness: &Self::ProposalRandomness) -> Option<u64> {
        M::randomness_bytes(randomness)
    }
}
