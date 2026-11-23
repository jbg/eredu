//! Portable submission and observation for ordinary text generation.

use crate::text_session_driver::TextSessionResourcesFor;

use eredu_core::{backend::TextTransactionMechanisms, sampling::SamplingBackend};
use std::ops::DerefMut;

use eredu_core::{
    backend::{CausalTextSession, NativeTextValues, TextSamplingNative},
    capture::{
        CaptureError, CapturePhase, CapturePrefillSpan, CaptureTensorOperations, CaptureTokenDomain,
    },
    BackendError, BackendProvider, ModelRuntime, Submission, TextOperationError,
    TextOperationPolicyError,
};

use super::{
    capture::{prepared_prompt_capture_shape, retained_prefill_span, TextCaptureMechanisms},
    PlainTextPrefill, TextDecodeInput, TextExecutable, TextPrefillInput, TextSessionDriver,
    TextSessionMechanisms,
};
use crate::{
    capture::{partition::PartitionCaptureHookTransport, CaptureSession},
    prefill::PrefillStep,
    NoopObserver,
};

/// Native input and collector bindings for the portable causal text driver.
pub trait CausalTextMechanisms<B: super::specialized::TextModeBinding>:
    TextSessionMechanisms<B>
    + eredu_core::backend::DeviceTextSampling<
        B,
        Sampling: SamplingBackend<
            RandomState = <B::Refinement as eredu_core::refinement::RefinementBackend>::RandomState,
        >,
    > + TextCaptureMechanisms<
        Self::Retained,
        Collector: CaptureTensorOperations<
            Self::Transport,
            Context = <Self::Executable as TextExecutable<
                crate::text_session_driver::SessionTextTypes<B, Self>,
            >>::Context,
            Tensor = <Self::Executable as TextExecutable<
                crate::text_session_driver::SessionTextTypes<B, Self>,
            >>::ObservationTensor,
            Error = B::Error,
        >,
    > + super::operations::TextResidencyTelemetry<Self::Resources>
{
}

impl<B, N> CausalTextMechanisms<B> for N
where
    B: super::specialized::TextModeBinding,
    N: TextSessionMechanisms<B>
        + eredu_core::backend::DeviceTextSampling<
            B,
            Sampling: SamplingBackend<
                RandomState = <B::Refinement as eredu_core::refinement::RefinementBackend>::RandomState,
            >,
        >
        + TextCaptureMechanisms<
            N::Retained,
            Collector: CaptureTensorOperations<
                N::Transport,
                Context = <N::Executable as TextExecutable<crate::text_session_driver::SessionTextTypes<B, N>>>::Context,
                Tensor = <N::Executable as TextExecutable<crate::text_session_driver::SessionTextTypes<B, N>>>::ObservationTensor,
                Error = B::Error,
            >,
        >
        + super::operations::TextResidencyTelemetry<N::Resources>,
{
}

/// Samples the exact final score row with the portable token filter and history policy.
pub fn sample_text_row<S: eredu_core::sampling::SamplingBackend>(
    sampler: &mut crate::generation::TextSampler,
    scores: &S::Logits,
    filter: &eredu_core::TokenFilter,
    temperature: f32,
    random: Option<&mut S::RandomState>,
    context: &S::Context,
) -> Result<S::Token, S::Error> {
    let scores = S::apply_token_filter(scores, filter, context)?;
    sampler.sample::<S>(&scores, temperature, random, context)
}

pub(super) fn sample_causal_scores<B, N>(
    model: &N::Model,
    backend: &B,
    submission: Submission<N::Output, N::Completion>,
    filter: &eredu_core::TokenFilter,
    sampler: &mut crate::generation::TextSampler,
    random: Option<&mut <N::Sampling as eredu_core::sampling::SamplingBackend>::RandomState>,
    temperature: f32,
) -> Result<Submission<N::Token, N::SamplingCompletion>, B::Error>
where
    B: eredu_core::TextGenerationBackend,
    N: TextSessionMechanisms<B> + eredu_core::backend::DeviceTextSampling<B>,
{
    sample_causal_with::<B, N>(model, backend, submission, |scores, context| {
        sample_text_row::<N::Sampling>(sampler, &scores, filter, temperature, random, context)
            .map_err(|error| N::preparation_error(eredu_core::BackendFailure::from_error(error)))
    })
}

pub(super) fn sample_causal_with<B, N>(
    model: &N::Model,
    backend: &B,
    submission: Submission<N::Output, N::Completion>,
    sample_row: impl FnOnce(
        N::ScoreRow,
        &<N::TextValues as NativeTextValues>::Context,
    ) -> Result<N::Sample, B::Error>,
) -> Result<Submission<N::Token, N::SamplingCompletion>, B::Error>
where
    B: eredu_core::TextGenerationBackend,
    N: TextSessionMechanisms<B> + TextSamplingNative<B>,
{
    use crate::executable_binding::ErasedTextExecutable as _;

    let context = N::context(backend);
    let scope = N::begin_sampling(&submission.completion)?;
    let sample = || {
        let scores = N::last_score_row(&submission.output, context)?.ok_or_else(|| {
            N::preparation_error(eredu_core::BackendFailure::from_error(
                std::io::Error::other("sampling rank requires its submitted text scores"),
            ))
        })?;
        sample_row(scores, context)
    };
    let token = match model.as_ref().partition_sampling_context() {
        Some((group, authority, exchange_context, rank)) => (|| {
            authority.ensure_active().map_err(|error| {
                N::preparation_error(eredu_core::BackendFailure::from_error(error))
            })?;
            let (local, count) = N::sampling_group_coordinates(group);
            if rank >= count {
                return Err(N::preparation_error(
                    eredu_core::BackendFailure::from_error(std::io::Error::other(
                        "sampling rank is outside its realized collective",
                    )),
                ));
            }
            let token = (local == rank).then(sample).transpose()?;
            N::exchange_sampled_token(token, rank, group, authority, exchange_context)
        })(),
        None => sample(),
    };
    N::finish_sampling(scope, submission.completion, token, context)
}

/// Failure updating the request cursor after a successful native submission.
#[derive(Debug, thiserror::Error)]
pub enum SamplePublicationError {
    /// The absolute prediction cannot advance.
    #[error("text prediction overflow")]
    PositionOverflow,
    /// Capture rejected the new boundary.
    #[error("capture rejected the sampled boundary: {0}")]
    Capture(#[from] CaptureError),
}

pub(super) fn advance_sample(
    prediction: &mut u64,
    capture: Option<&mut CaptureSession>,
) -> Result<(), SamplePublicationError> {
    *prediction = prediction
        .checked_add(1)
        .ok_or(SamplePublicationError::PositionOverflow)?;
    if let Some(capture) = capture {
        capture.set_commitment_boundary(*prediction)?;
    }
    Ok(())
}

/// The request state that participates in submitting ordinary causal work.
pub(super) struct CausalTextState<'a, Scores> {
    /// Parameter version admitted for this request.
    pub parameter_epoch: &'a mut Option<u64>,
    /// Current optional observation session.
    pub capture: Option<&'a mut CaptureSession>,
    /// Absolute position used to attribute one-token decode.
    pub prediction: &'a mut u64,
    /// Scores retained after a settled handoff.
    pub next_logits: &'a mut Option<Scores>,
    /// Whether the committed token is already present in the executable cache.
    pub committed_token_cached: &'a mut bool,
}

pub(super) fn causal_state<B: super::specialized::TextModeBinding>(
    state: &mut super::TextRequestFor<B>,
) -> CausalTextState<
    '_,
    <B::Shared as crate::speculative::autoregressive::AutoregressiveMechanisms>::Logits,
> {
    CausalTextState {
        parameter_epoch: &mut state.sampling.parameter_epoch,
        capture: state.capture.as_mut(),
        prediction: &mut state.sampling.next_prediction,
        next_logits: &mut state.sampling.decoding.next_logits,
        committed_token_cached: &mut state.sampling.decoding.committed_token_cached,
    }
}

type Prefill<B, N> = <N as TextSessionMechanisms<B>>::Prefill;
type CausalSubmission<B, N> = Result<
    Submission<
        <N as eredu_core::backend::TextSessionNative<B>>::Output,
        <N as eredu_core::backend::TextSessionNative<B>>::Completion,
    >,
    TextOperationError<<B as BackendProvider>::Error>,
>;

impl<B, R, N> CausalTextSession<B, Prefill<B, N>, N::Token, super::TextRequestFor<B>> for TextSessionDriver<R, N>
where
    B: super::specialized::TextModeBinding<Error: From<BackendError>>,
    N: CausalTextMechanisms<B, Resources = R,
        TextValues: eredu_core::backend::NativeTextValues<Scores = <B::Shared as crate::speculative::autoregressive::AutoregressiveMechanisms>::Logits>,
        Executable: TextExecutable<crate::text_session_driver::SessionTextTypes<B, N>, Scores = <B::Shared as crate::speculative::autoregressive::AutoregressiveMechanisms>::Logits>,
    >
        + TextCaptureMechanisms<
            N::Retained,
            Transport: PartitionCaptureHookTransport<
                Error: Send + Sync + 'static,
                Completion: eredu_core::Completion<Error: Send + Sync + 'static>,
            >,
        > + TextTransactionMechanisms<Error = B::Error>,
    R: DerefMut<Target = TextSessionResourcesFor<B, N>>
        + 'static,
{
    fn submit_causal_prefill(
        &mut self,
        backend: &B,
        prompt: Prefill<B, N>,
        domain: Option<CaptureTokenDomain<'_>>,
        state: &mut super::TextRequestFor<B>,
    ) -> CausalSubmission<B, N> {
        let state = causal_state::<B>(state);
        let prepared = self
            .validate_parameter_epoch(state.parameter_epoch)
            .map(|()| prompt);
        let Some(capture) = state.capture else {
            return self
                .advance_prefill(backend, prepared)
                .map_err(TextOperationError::Backend);
        };

        let request = capture.plan().request();
        if let Some(cursor) = prepared
            .as_ref()
            .ok()
            .and_then(TextPrefillInput::prefill_cursor)
        {
            if let Some(span) =
                retained_prefill_span(cursor)
                    .map_err(|error| TextOperationError::Backend(N::preparation_error(eredu_core::BackendFailure::from(error))))?
            {
                capture
                    .set_prefill_span(span)
                    .map_err(|error| N::preparation_error(
                                eredu_core::BackendFailure::from_error(error).with_model_state_preserved(true),
                            ))
                    .map_err(TextOperationError::Backend)?;
            }
        }
        let geometry = prepared
            .as_ref()
            .map_err(|error| CaptureError::Invalid(error.to_string()))
            .and_then(|prompt| self.payload.model.as_ref().prepared_input_plans(prompt.borrowed())
                .map_err(|error| CaptureError::Invalid(error.to_string())))
            .and_then(|parts| prepared_prompt_capture_shape(&parts));
        let prepared_capture =
            self.prepare_text_capture_observer(capture, 0, CapturePhase::Prefill, geometry);
        let mut observer =
            prepared_capture.open::<N::Collector>(capture, N::context(backend), domain, 0);
        let (submission, publish) = self
            .run_prefill(backend, prepared, Some(request), Some(&mut *observer))
            .map_err(TextOperationError::Backend)?;
        Ok(Submission {
            output: N::publish_scores(submission.output, publish),
            completion: submission.completion,
        })
    }

    fn submit_causal_decode(
        &mut self,
        backend: &B,
        token: N::Token,
        domain: Option<CaptureTokenDomain<'_>>,
        state: &mut super::TextRequestFor<B>,
    ) -> CausalSubmission<B, N> {
        let state = causal_state::<B>(state);
        let input = TextDecodeInput::Sampled(
            self.validate_parameter_epoch(state.parameter_epoch)
                .map(|()| N::sampled_token_value(token)),
        );
        let Some(capture) = state.capture else {
            return self
                .advance_decode(backend, input)
                .map_err(TextOperationError::Backend);
        };

        let prediction = *state.prediction;
        let prepared_capture = self.prepare_text_capture_observer(
            capture,
            prediction,
            CapturePhase::Decode,
            Ok((1, None)),
        );
        let mut observer = prepared_capture.open::<N::Collector>(
            capture,
            N::context(backend),
            domain,
            prediction,
        );
        let (submission, publish) = self
            .run_decode(backend, input, Some(&mut *observer))
            .map_err(TextOperationError::Backend)?;
        Ok(Submission {
            output: N::publish_scores(submission.output, publish),
            completion: submission.completion,
        })
    }

    fn submit_causal_continuation(
        &mut self,
        backend: &B,
        state: &mut super::TextRequestFor<B>,
    ) -> CausalSubmission<B, N> {
        let scores = causal_state::<B>(state)
            .next_logits
            .take()
            .ok_or(TextOperationPolicyError::UnsupportedContinuation)?;
        let submission = self
            .retain_scores(backend, scores)
            .map_err(TextOperationError::Backend)?;
        let publish = self.payload.model.as_ref().partition_public_output();
        Ok(Submission {
            output: N::publish_scores(submission.output, publish),
            completion: submission.completion,
        })
    }
}

/// Installed-session execution of non-sampling prefill.
pub(super) trait CausalPrefillDriver<B: super::specialized::TextModeBinding>:
    eredu_core::BackendSession<B> + Sized
{
    fn prefill_prefix(
        runtime: &mut ModelRuntime<B, Self>,
        prompt: &mut crate::text_session_driver::TextPromptFor<B>,
        max_tokens: std::num::NonZeroUsize,
        state: &mut super::TextRequestFor<B>,
    ) -> Result<bool, B::Error>;
}

impl<B, R, N> CausalPrefillDriver<B> for TextSessionDriver<R, N>
where
    B: super::specialized::TextModeBinding<Error: From<BackendError>>,
    N: CausalTextMechanisms<B, Resources = R, Prefill = super::TextPromptFor<B>>
        + TextCaptureMechanisms<
            N::Retained,
            Transport: PartitionCaptureHookTransport<
                Error: Send + Sync + 'static,
                Completion: eredu_core::Completion<Error: Send + Sync + 'static>,
            >,
        > + TextTransactionMechanisms<Error = B::Error>,
    Self: eredu_core::BackendSession<B, PrefillInput = Prefill<B, N>>,
    Prefill<B, N>: PlainTextPrefill<B>,
    R: DerefMut<Target = TextSessionResourcesFor<B, N>> + 'static,
{
    fn prefill_prefix(
        runtime: &mut ModelRuntime<B, Self>,
        prompt: &mut crate::text_session_driver::TextPromptFor<B>,
        max_tokens: std::num::NonZeroUsize,
        state: &mut super::TextRequestFor<B>,
    ) -> Result<bool, B::Error> {
        if runtime
            .session()
            .resources()
            .model
            .as_ref()
            .block_refinement_geometry()
            .is_some()
        {
            return Ok(false);
        }

        let mut state = causal_state::<B>(state);
        if runtime
            .session()
            .payload
            .model
            .as_ref()
            .retains_prefill_cursor()
        {
            runtime
                .session()
                .validate_parameter_epoch(state.parameter_epoch)?;
            let request = state
                .capture
                .as_ref()
                .map(|capture| capture.plan().request());
            let (backend, session) = runtime.parts_mut();
            let step = session.prepare_prefill_prefix(backend, prompt, max_tokens, request)?;
            if let Some(cursor) = prompt.prefill_cursor_mut() {
                let Some(step) = step else { return Ok(false) };
                if let Some(capture) = state.capture.as_deref_mut() {
                    if let PrefillStep::Decoder(range) = &step {
                        capture
                            .set_prefill_span(CapturePrefillSpan {
                                start: range.start as u64,
                                end: range.end as u64,
                                total: cursor.token_count() as u64,
                            })
                            .map_err(|error| {
                                N::preparation_error(
                                    eredu_core::BackendFailure::from_error(error)
                                        .with_model_state_preserved(true),
                                )
                            })?;
                    }
                    let prepared = session.prepare_capture_observer(capture);
                    let mut observer =
                        prepared.open::<N::Collector>(capture, N::context(backend), None, 0);
                    session.advance_prefill_prefix(backend, cursor, &step, &mut *observer)?;
                } else {
                    session.advance_prefill_prefix(backend, cursor, &step, &mut NoopObserver)?;
                }
                return Ok(true);
            }
        }

        if state.capture.is_some() || runtime.session().prefill_chunking_support::<B>().is_err() {
            return Ok(false);
        }
        Self::advance_plain_prefill_prefix::<B, Prefill<B, N>>(
            runtime,
            prompt,
            max_tokens,
            state.parameter_epoch,
        )
    }
}
