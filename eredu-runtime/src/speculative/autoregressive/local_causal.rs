//! Shared installed causal execution using eager native score rows.

use std::{marker::PhantomData, rc::Rc};

use eredu_core::{
    backend::{local_text::HostCausalVerification, NativeTextValues},
    eager_completion::EagerCompletion,
    execution_control::SnapshotEstimate,
    sampling::{host::HostScores, host_random::HostRandomState},
    BackendFailure, BackendFailureKind,
};

use crate::{
    executable_binding::{ErasedTextExecutable, PortableTextTypes},
    generation::refinement::{
        architecture::{BlockRefinementGeometry, RefinementExecutionPolicy},
        host::{add_retained_value, HostRefinement, ScoreMatrix},
    },
    speculative::shared_backbone::{
        SharedBackboneExecutionPolicy, SharedBackboneMechanisms, SharedRefinementState,
    },
    text_session_driver::{PreparedTextPrompt, TextPrefillInput},
    NoopObserver,
};

use super::{
    causal_invocation::InstalledCausalState, local_batch::LocalCausalBackend,
    AutoregressiveMechanisms, AutoregressivePass,
};

/// Native queue and observer types shared by the installed prediction mechanisms.
pub(super) type LocalCausalCapture<B> = <B as HostCausalVerification>::ContextBinding;

/// Causal and refinement invocations using installed local state.
pub(crate) struct LocalCausalMechanisms<B>(PhantomData<fn() -> B>);

/// Settled native score rows and the context that owns them.
pub struct LocalVerificationOutput<V: NativeTextValues> {
    scores: V::Scores,
    queue: V::Context,
}

impl<V: NativeTextValues<Scores: Clone>> Clone for LocalVerificationOutput<V> {
    fn clone(&self) -> Self {
        Self {
            scores: self.scores.clone(),
            queue: self.queue.clone(),
        }
    }
}

impl<B, V> AutoregressiveMechanisms for LocalCausalMechanisms<B>
where
    B: HostCausalVerification<Values = V>,
    V: NativeTextValues<Error = BackendFailure, Context: 'static, Scores: Clone>,
{
    type Model = dyn ErasedTextExecutable<PortableTextTypes<V>>;
    type Input = PreparedTextPrompt<V::InputTensor>;
    type State = InstalledCausalState<V>;
    type Checkpoint = Rc<Self::State>;
    type Output = LocalVerificationOutput<V>;
    type Logits = HostScores;
    type Context<'a> = crate::capture::SpeculativeContextFor<'a, LocalCausalCapture<B>>;
    type Completion = EagerCompletion<BackendFailure>;
    type Error = BackendFailure;

    fn eager_component_timing() -> bool {
        true
    }

    fn verification_output_bytes(output: &Self::Output) -> Option<u64> {
        B::verification_storage(&output.scores)
    }

    fn logits_snapshot_estimate(scores: &HostScores) -> Option<SnapshotEstimate> {
        Some(SnapshotEstimate {
            retained_bytes: scores.retained_bytes()?,
            copy_bytes: size_of::<HostScores>() as u64,
        })
    }

    fn empty(
        model: &mut Self::Model,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<Self::State, BackendFailure> {
        let queue = if pass == AutoregressivePass::DraftPrefill {
            context.draft()
        } else {
            context.target()
        };
        InstalledCausalState::allocate(model, queue)
    }

    fn prefill(
        model: &mut Self::Model,
        input: &Self::Input,
        state: &mut Self::State,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<(Self::Output, usize), BackendFailure> {
        let draft_input = if pass == AutoregressivePass::DraftPrefill && context.is_split() {
            Some(input.transfer(
                &B::verification_input_transfer(context.target()),
                context.draft(),
            )?)
        } else {
            None
        };
        let mut execute =
            |observer: &mut dyn crate::ActivationObserver<V::InvocationTensor, BackendFailure>| {
                state.prefill(
                    model,
                    draft_input.as_ref().unwrap_or(input).borrowed(),
                    observer,
                )
            };
        let (scores, count) = match context
            .target_observer()
            .filter(|_| pass == AutoregressivePass::TargetPrefill)
        {
            Some(observer) => {
                let mut count = 0;
                let scores = observer.with_observer(&mut |observer| {
                    let (scores, positions) = execute(observer)?;
                    count = positions;
                    Ok(scores)
                })?;
                (scores, count)
            }
            None => execute(&mut NoopObserver)?,
        };
        Ok((
            LocalVerificationOutput {
                scores,
                queue: state.queue.clone(),
            },
            count,
        ))
    }

    fn decode(
        model: &mut Self::Model,
        tokens: &[u32],
        state: &mut Self::State,
        pass: AutoregressivePass,
        context: Self::Context<'_>,
    ) -> Result<Self::Output, BackendFailure> {
        let tokens = B::verification_tokens(tokens, &state.queue)?;
        let mut execute = |observer: &mut dyn crate::ActivationObserver<
            V::InvocationTensor,
            BackendFailure,
        >| { state.append(model, &tokens, observer) };
        let scores = match context
            .target_observer()
            .filter(|_| pass == AutoregressivePass::Verification)
        {
            Some(observer) => observer.with_observer(&mut execute)?,
            None => execute(&mut NoopObserver)?,
        };
        Ok(LocalVerificationOutput {
            scores,
            queue: state.queue.clone(),
        })
    }

    fn checkpoint(state: &Self::State) -> Result<Self::Checkpoint, BackendFailure> {
        state.checkpoint()
    }

    fn restore(
        saved: &Self::Checkpoint,
        _: Self::Context<'_>,
    ) -> Result<Self::State, BackendFailure> {
        saved.copy_boundary()
    }

    fn retain_verified_prefix(
        state: &mut Self::State,
        saved: &Self::Checkpoint,
        accepted: usize,
        _: Self::Context<'_>,
    ) -> Result<bool, BackendFailure> {
        let queue = state.queue.clone();
        state.retain_verified_prefix(saved, accepted, &queue)
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
            model.residency_report()?,
            model.parameter_bank_report()?,
            B::verification_memory(&state.queue)?,
        )
        .map(Some)
    }

    fn logits(
        output: &Self::Output,
        position: usize,
        _: Self::Context<'_>,
    ) -> Result<HostScores, BackendFailure> {
        HostScores::from_row(B::verification_score(
            &output.scores,
            position,
            &output.queue,
        )?)
    }

    fn completion(
        _: &Self::Output,
        _: Self::Context<'_>,
    ) -> Result<Self::Completion, BackendFailure> {
        Ok(EagerCompletion::completed())
    }

    fn invalid(message: &'static str) -> BackendFailure {
        BackendFailure::new(
            BackendFailureKind::InvalidSession,
            std::io::Error::other(message),
        )
    }
}

impl<B: LocalCausalBackend> SharedBackboneMechanisms for LocalCausalMechanisms<B> {
    type ActivationTensor = <B::Values as NativeTextValues>::Tensor;
    type Capture = LocalCausalCapture<B>;
    type ProposalRandomness = HostRandomState;
    type Queue = ();
    type Refinement = HostRefinement;

    fn reborrow_context<'a: 'b, 'b>(context: Self::Context<'a>) -> Self::Context<'b> {
        context
    }

    fn publish_committed(
        model: &mut Self::Model,
        state: &mut Self::State,
    ) -> Result<(), BackendFailure> {
        model.exchange_autoregressive_cache(&mut state.cache, &state.queue)
    }

    fn activation_prefill_plans(
        model: &Self::Model,
        input: &Self::Input,
    ) -> Result<
        Vec<crate::executable_binding::retained_session::PreparedInputPartPlan>,
        BackendFailure,
    > {
        model
            .prepared_input_plans(input.borrowed())
            .map_err(BackendFailure::from_error)
    }

    fn prefill_with_activation(
        model: &mut Self::Model,
        input: &Self::Input,
        state: &mut Self::State,
        observer: &mut dyn crate::ActivationObserver<Self::ActivationTensor, BackendFailure>,
    ) -> Result<(Self::Output, usize), BackendFailure> {
        let mut observer = crate::inspection::MappedActivationObserver::<
            _,
            <B::Values as NativeTextValues>::ObserverMapping,
        >::new(observer);
        let (scores, count) = state.prefill(model, input.borrowed(), &mut observer)?;
        Ok((
            LocalVerificationOutput {
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
        observer: &mut dyn crate::ActivationObserver<Self::ActivationTensor, BackendFailure>,
    ) -> Result<Self::Output, BackendFailure> {
        let tokens = B::verification_tokens(tokens, &state.queue)?;
        let mut observer = crate::inspection::MappedActivationObserver::<
            _,
            <B::Values as NativeTextValues>::ObserverMapping,
        >::new(observer);
        let scores = state.append(model, &tokens, &mut observer)?;
        Ok(LocalVerificationOutput {
            scores,
            queue: state.queue.clone(),
        })
    }

    fn ordinary_refinement_policy(model: &Self::Model) -> Option<RefinementExecutionPolicy> {
        model.capability_estimate().ordinary_refinement()
    }

    fn shared_backbone_policy(model: &Self::Model) -> Option<SharedBackboneExecutionPolicy> {
        model.capability_estimate().shared_backbone_draft()
    }

    fn model_randomness_root(seed: u64) -> Result<HostRandomState, BackendFailure> {
        Ok(HostRandomState::with_seed(seed))
    }

    fn split_model_randomness(
        root: &HostRandomState,
        substream: usize,
        _: Self::Context<'_>,
    ) -> Result<HostRandomState, BackendFailure> {
        Ok(root.split(substream as u64))
    }

    fn settle_model_randomness(_: &[&HostRandomState]) -> Result<(), BackendFailure> {
        Ok(())
    }

    fn target_refinement_context<'a>(_: Self::Context<'a>) -> &'a () {
        &()
    }

    fn draft_refinement_context<'a>(context: Self::Context<'a>) -> Option<&'a ()> {
        (!context.crosses_devices()).then_some(&())
    }

    fn execution_error(error: impl std::error::Error + Send + Sync + 'static) -> BackendFailure {
        BackendFailure::from_error(error)
    }

    fn block_geometry(model: &Self::Model) -> Option<BlockRefinementGeometry> {
        model.block_refinement_geometry()
    }

    fn score_format(_: &HostScores) {}
    fn draft_score_format() {}

    fn refinement_scores(logits: &HostScores) -> ScoreMatrix {
        ScoreMatrix::from_row(logits)
    }

    fn evaluate_refinement(
        model: &mut Self::Model,
        prefix: &mut Self::State,
        canvas: &std::sync::Arc<[u32]>,
        feedback: Option<&ScoreMatrix>,
        geometry: BlockRefinementGeometry,
        invocation: eredu_core::decoding::AttentionInvocation,
        _: &(),
        observer: Option<
            &mut dyn crate::ActivationObserver<Self::ActivationTensor, BackendFailure>,
        >,
    ) -> Result<ScoreMatrix, BackendFailure> {
        let queue = &prefix.queue;
        use eredu_core::observation::ObserverTensorMapping;
        type Map<V> = <V as NativeTextValues>::TensorObserver;
        let canvas = B::host_refinement_canvas(canvas, queue)?;
        let feedback = feedback
            .map(|scores| {
                let element = scores
                    .element
                    .ok_or_else(|| Self::invalid("missing native feedback precision"))?;
                B::host_refinement_feedback(
                    scores.values(),
                    &geometry.feedback_shape(),
                    element,
                    queue,
                )
            })
            .transpose()?;
        let mut noop = NoopObserver;
        let mut observer = crate::inspection::MappedActivationObserver::<
            _,
            <B::Values as NativeTextValues>::ObserverMapping,
        >::new(observer.unwrap_or(&mut noop));
        let scores = model.attention_forward_with_observer(
            Map::<B::Values>::borrow_tensor(&canvas),
            feedback.as_ref().map(Map::<B::Values>::borrow_tensor),
            invocation,
            Some(&mut prefix.cache),
            false,
            queue,
            &mut observer,
        )?;
        let shape = [
            geometry.canvas_length.get() as usize,
            geometry.vocabulary_size.get() as usize,
        ];
        let mut host = ScoreMatrix::new(B::host_refinement_scores(&scores, shape, queue)?, shape)?;
        host.element = B::host_refinement_format(&scores);
        Ok(host)
    }

    fn read_refinement_tokens(
        tokens: &std::sync::Arc<[u32]>,
        range: std::ops::Range<usize>,
        _: &(),
    ) -> Result<Vec<u32>, BackendFailure> {
        tokens
            .get(range)
            .map(<[u32]>::to_vec)
            .ok_or_else(|| Self::invalid("invalid refinement token span"))
    }

    fn refinement_score_row(
        scores: &ScoreMatrix,
        position: usize,
        _: &(),
    ) -> Result<HostScores, BackendFailure> {
        scores.row(position)
    }

    fn settle_refinement(_: &SharedRefinementState<Self>) -> Result<(), BackendFailure> {
        Ok(())
    }

    fn refinement_state_estimate(state: &SharedRefinementState<Self>) -> Option<SnapshotEstimate> {
        let copy_bytes = state.snapshot_host_bytes()?;
        let mut retained = Some(copy_bytes);
        state.visit_retained_values(|value| retained = add_retained_value(retained, value));
        Some(SnapshotEstimate {
            retained_bytes: retained?,
            copy_bytes,
        })
    }

    fn proposal_randomness_bytes(_: &HostRandomState) -> Option<u64> {
        Some(size_of::<HostRandomState>() as u64)
    }
}
