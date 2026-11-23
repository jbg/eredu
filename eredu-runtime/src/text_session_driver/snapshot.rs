//! Snapshot orchestration over the installed executable and native copy primitives.

use crate::text_session_driver::TextSessionResourcesFor;
use eredu_core::refinement::RefinementBackend;

use super::{
    InstalledTextState, TextExecutable, TextPrefillInput, TextSamplingState, TextSessionDriver,
    TextSessionMechanisms,
};
#[cfg(feature = "tensor-capture")]
use crate::capture::CaptureSession;
use crate::{
    execution_control::TextSnapshotBackend,
    generation::refinement::{architecture::BlockRefinementGeometry, RefinementValue},
};
use eredu_core::{
    backend::TextTransactionMechanisms,
    decoding::DecodingMode,
    execution_control::{
        NativeTextStateSession, SnapshotEstimate, TextModeSnapshotNative, TextSnapshotNative,
    },
    ModelRuntime, PendingTextInput,
};
#[cfg(feature = "tensor-capture")]
use eredu_core::{
    capture::{CaptureError, CaptureSelection, CaptureUsage, ResolvedCaptureSlice},
    intervention::InterventionEstimator,
};
#[cfg(feature = "tensor-capture")]
use std::sync::Arc;

/// Prepared input geometry observable without evaluating native work.
pub enum SnapshotPendingGeometry {
    /// One committed decoder token.
    Decode,
    /// Retained work with no newly supplied input.
    Continue,
    /// Remaining decoder positions in an existing cursor.
    PrefillCursor(u64),
    /// Verified count for a plain text prompt.
    TextPrompt(u64),
    /// Executable admission supplies position geometry.
    PreparedPrompt,
}

/// Position facts retained by an iterative decoder.
pub struct SnapshotRefinementFacts {
    /// Positions that can still commit before another context ingestion.
    pub pending_positions: u64,
    /// Context tokens waiting to be ingested, if in the context phase.
    pub context_tokens: Option<u64>,
}

/// Combines the portable text modes with exact snapshot storage mechanisms.
pub trait TextSnapshotMechanisms:
    super::specialized::TextModeBinding<
        Shared: crate::speculative::shared_backbone::SharedBackboneMechanisms<Output: Clone>,
        Error: From<eredu_core::BackendFailure>,
    > + TextModeSnapshotNative<
        CausalScores = ScoresFor<Self>,
        RefinementTokens = <<Self as super::specialized::TextModeBinding>::Refinement as RefinementBackend>::Token,
        RefinementScores = <<Self as super::specialized::TextModeBinding>::Refinement as RefinementBackend>::Logits,
        RefinementMask = <<Self as super::specialized::TextModeBinding>::Refinement as RefinementBackend>::Mask,
    >
{
}

impl<B> TextSnapshotMechanisms for B
where
    B: super::specialized::TextModeBinding<
        Shared: crate::speculative::shared_backbone::SharedBackboneMechanisms<Output: Clone>,
        Error: From<eredu_core::BackendFailure>,
    > + TextModeSnapshotNative<
        CausalScores = ScoresFor<B>,
        RefinementTokens = <<B as super::specialized::TextModeBinding>::Refinement as RefinementBackend>::Token,
        RefinementScores = <<B as super::specialized::TextModeBinding>::Refinement as RefinementBackend>::Logits,
        RefinementMask = <<B as super::specialized::TextModeBinding>::Refinement as RefinementBackend>::Mask,
    >,
{
}

type RandomFor<B> = <<B as super::specialized::TextModeBinding>::Refinement as eredu_core::refinement::RefinementBackend>::RandomState;
type ScoresFor<B> = <<B as super::specialized::TextModeBinding>::Shared as crate::speculative::autoregressive::AutoregressiveMechanisms>::Logits;
type SnapshotSamplingFor<B> = super::TextSamplingFor<B>;

fn refinement_facts<B: TextSnapshotMechanisms>(
    sampling: &SnapshotSamplingFor<B>,
) -> Option<SnapshotRefinementFacts> {
    let block = sampling.decoding.refinement.as_ref()?;
    Some(SnapshotRefinementFacts {
        pending_positions: block.pending_context_output()?,
        context_tokens: (block.phase() == eredu_core::refinement::GenerationPhase::Context)
            .then(|| block.authoritative_tokens().len() as u64),
    })
}

fn estimate_sampling_state<B: TextSnapshotMechanisms>(
    sampling: &SnapshotSamplingFor<B>,
) -> Option<SnapshotEstimate> {
    let mut bytes = u64::try_from(std::mem::size_of_val(sampling))
        .ok()?
        .checked_add(sampling.sampler.history_bytes()?)?;
    for random in [
        &sampling.prng,
        &sampling.decoding.refinement_random,
        &sampling.proposal_random,
    ]
    .into_iter()
    .flatten()
    {
        bytes = bytes.checked_add(B::random_bytes(random)?)?;
    }
    if let Some(scores) = &sampling.decoding.next_logits {
        bytes = bytes.checked_add(B::score_bytes(scores)?)?;
    }
    if let Some(transaction) = &sampling.decoding.speculative {
        bytes = bytes.checked_add(transaction.retained_bytes()?)?;
    }
    bytes = bytes.checked_add(
        sampling
            .canvas
            .as_ref()
            .map_or(0, |canvas| canvas.host_bytes()),
    )?;
    let mut copy_bytes = bytes;
    if let Some(refinement) = &sampling.decoding.refinement {
        let host = refinement.snapshot_host_bytes()?;
        copy_bytes = copy_bytes.checked_add(host)?;
        let mut retained = Some(copy_bytes);
        refinement.visit_retained_values(|value| {
            let backing_bytes = match value {
                RefinementValue::Tokens(value) => B::refinement_token_bytes(value),
                RefinementValue::Scores(value) => B::refinement_score_bytes(value),
                RefinementValue::Mask(value) => B::refinement_mask_bytes(value),
                RefinementValue::Random(value) => B::random_bytes(value),
            };
            retained = retained.and_then(|bytes| bytes.checked_add(backing_bytes?));
        });
        bytes = retained?;
    }
    Some(SnapshotEstimate {
        retained_bytes: bytes,
        copy_bytes,
    })
}

fn copy_sampling_state<B: TextSnapshotMechanisms>(
    backend: &B,
    sampling: &SnapshotSamplingFor<B>,
) -> Result<SnapshotSamplingFor<B>, B::Error> {
    let copy_random = |random: &Option<RandomFor<B>>| {
        random
            .as_ref()
            .map(|random| B::copy_random(backend, random))
            .transpose()
    };
    Ok(TextSamplingState {
        temperature: sampling.temperature,
        prng: copy_random(&sampling.prng)?,
        sampler: sampling.sampler.clone(),
        next_prediction: sampling.next_prediction,
        parameter_epoch: sampling.parameter_epoch,
        refinement_seed: sampling.refinement_seed,
        proposal_random: copy_random(&sampling.proposal_random)?,
        draft_events: sampling.draft_events,
        canvas: sampling.canvas.clone(),
        decoding: super::TextModeState {
            mode: sampling.decoding.mode,
            speculative: sampling.decoding.speculative.clone(),
            committed_token_cached: sampling.decoding.committed_token_cached,
            next_logits: sampling
                .decoding
                .next_logits
                .as_ref()
                .map(|scores| B::copy_scores(backend, scores))
                .transpose()?,
            refinement_random: copy_random(&sampling.decoding.refinement_random)?,
            refinement_policy: sampling.decoding.refinement_policy,
            refinement: sampling.decoding.refinement.clone(),
        },
    })
}

pub(super) fn shared_verification_growth<C: ?Sized>(
    state: Option<&dyn super::ErasedTextStateAccess<C>>,
    installed: Option<SnapshotEstimate>,
    inputs: u64,
) -> Option<u64> {
    // Baseline, verification, accepted restoration, and rollback can coexist.
    installed?
        .retained_bytes
        .checked_add(state?.estimate_installed_growth(inputs)?)?
        .checked_mul(4)
}

fn estimate_sampling_growth<B: TextSnapshotMechanisms>(
    sampling: &SnapshotSamplingFor<B>,
    predictions: u64,
    geometry: Option<BlockRefinementGeometry>,
    shared_state: Option<u64>,
) -> Option<u64> {
    let facts = B::sampling_growth_costs();
    let causal = predictions
        .checked_mul(4)?
        .checked_add(facts.random_key_bytes.checked_mul(2)?)?;
    let Some(geometry) = geometry else {
        return Some(causal);
    };
    let canvas = u64::from(geometry.canvas_length.get());
    let vocabulary = u64::from(geometry.vocabulary_size.get());
    let score_bytes = canvas
        .checked_mul(vocabulary)?
        .checked_mul(16)?
        .checked_add(facts.array_overhead.checked_mul(2)?)?;
    let history = u64::from(
        sampling
            .decoding
            .refinement_policy
            .stability_history_length(),
    )
    .checked_add(if sampling.canvas.is_some() { 5 } else { 3 })?;
    let tokens = canvas.checked_mul(8)?.checked_mul(history)?;
    const REFINEMENT_HOST_RESERVE: u64 = 4096;
    causal
        .checked_add(score_bytes)?
        .checked_add(tokens)?
        .checked_add(canvas.checked_mul(facts.array_overhead)?)?
        .checked_add(history.checked_mul(facts.array_overhead)?)?
        .checked_add(shared_state?)?
        .checked_add(REFINEMENT_HOST_RESERVE)
}

pub(super) fn plain_tokens<B: TextSnapshotNative>(
    prompt: &crate::text_session_driver::TextPromptFor<B>,
) -> Option<(&B::InputTensor, u64)> {
    use eredu_core::checkpoint::TensorDtype;
    let tensor = prompt.plain_token_tensor()?;
    let identity = B::prompt_tensor_identity(tensor)?;
    let [1, count] = identity.shape() else {
        return None;
    };
    if !matches!(identity.dtype(), TensorDtype::I32 | TensorDtype::U32) {
        return None;
    }
    Some((tensor, u64::try_from(*count).ok()?))
}

pub(super) fn pending_geometry<B: TextSnapshotNative>(
    input: PendingTextInput<&crate::text_session_driver::TextPromptFor<B>, &B::Token>,
) -> Option<SnapshotPendingGeometry> {
    Some(match input {
        PendingTextInput::Decode(token) => {
            B::token_bytes(token)?;
            SnapshotPendingGeometry::Decode
        }
        PendingTextInput::Continue => SnapshotPendingGeometry::Continue,
        PendingTextInput::Prefill(prompt) => match prompt.prefill_cursor() {
            Some(cursor) => SnapshotPendingGeometry::PrefillCursor(
                u64::try_from(
                    cursor
                        .token_count()
                        .checked_sub(cursor.decoder_position())?,
                )
                .ok()?,
            ),
            None => plain_tokens::<B>(prompt)
                .map_or(SnapshotPendingGeometry::PreparedPrompt, |(_, count)| {
                    SnapshotPendingGeometry::TextPrompt(count)
                }),
        },
    })
}

pub(super) fn estimate_pending_input<B: TextSnapshotNative>(
    input: Option<PendingTextInput<&crate::text_session_driver::TextPromptFor<B>, &B::Token>>,
) -> Result<Option<SnapshotEstimate>, B::Error>
where
    B::Error: From<eredu_core::BackendFailure>,
{
    let cursor_bytes = match input.as_ref() {
        Some(PendingTextInput::Prefill(prompt)) => prompt
            .prefill_cursor()
            .map(|cursor| {
                cursor
                    .retained_bytes()
                    .map_err(eredu_core::BackendFailure::from_error)
            })
            .transpose()?,
        _ => None,
    };
    Ok((|| {
        let bytes = match input {
            None | Some(PendingTextInput::Continue) => 0,
            Some(PendingTextInput::Decode(token)) => B::token_bytes(token)?,
            Some(PendingTextInput::Prefill(prompt)) => {
                let retained = match prompt.prefill_cursor() {
                    Some(_) => u64::try_from(cursor_bytes?).ok()?,
                    None => {
                        let input = prompt.borrowed();
                        input.parts.iter().try_fold(0u64, |bytes, part| {
                            let bytes = bytes
                                .checked_add(std::mem::size_of_val(part) as u64)?
                                .checked_add(B::prompt_tensor_bytes(part.payload().value())?)?;
                            part.metadata().values().try_fold(bytes, |bytes, tensor| {
                                bytes.checked_add(B::prompt_tensor_bytes(tensor)?)
                            })
                        })?
                    }
                };
                let metadata = match prompt.cache_identity() {
                    Some(identity) => identity.logical_metadata_bytes()?,
                    None => 0,
                };
                retained
                    .checked_add(
                        std::mem::size_of::<crate::text_session_driver::TextPromptFor<B>>() as u64,
                    )?
                    .checked_add(metadata)?
            }
        };
        Some(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    })())
}

pub(super) fn copy_pending_input<B: TextSnapshotNative>(
    backend: &B,
    input: Option<PendingTextInput<&crate::text_session_driver::TextPromptFor<B>, &B::Token>>,
) -> Result<
    Option<PendingTextInput<crate::text_session_driver::TextPromptFor<B>, B::Token>>,
    B::Error,
>
where
    B::Error: From<eredu_core::BackendFailure>,
{
    use eredu_core::{BackendFailure, InputModality};
    if estimate_pending_input::<B>(input.as_ref().map(|input| match input {
        PendingTextInput::Prefill(prompt) => PendingTextInput::Prefill(*prompt),
        PendingTextInput::Decode(token) => PendingTextInput::Decode(*token),
        PendingTextInput::Continue => PendingTextInput::Continue,
    }))?
    .is_none()
    {
        return Err(BackendFailure::from_error(std::io::Error::other(
            "pending input cannot be represented by the snapshot contract",
        ))
        .into());
    }
    Ok(match input {
        None => None,
        Some(PendingTextInput::Continue) => Some(PendingTextInput::Continue),
        Some(PendingTextInput::Decode(token)) => {
            Some(PendingTextInput::Decode(B::copy_token(backend, token)?))
        }
        Some(PendingTextInput::Prefill(prompt)) => {
            let copied = if prompt.prefill_cursor().is_some() {
                prompt.clone()
            } else if let Some((tokens, _)) = plain_tokens::<B>(prompt) {
                let parts = [eredu_core::PreparedInputPart::new(
                    InputModality::Text,
                    eredu_core::PreparedInputPayload::TokenIds(B::copy_prompt_tensor(
                        backend, tokens,
                    )?),
                    [],
                )
                .map_err(BackendFailure::from_error)?];
                let input = match prompt.cache_identity() {
                    Some(identity) => super::TextPromptView::with_cache_identity(&parts, identity),
                    None => super::TextPromptView::new(&parts),
                };
                super::PreparedTextPrompt::from(input)
            } else {
                B::complete_prompt_tensors(prompt.borrowed().parts.iter().flat_map(|part| {
                    std::iter::once(part.payload().value()).chain(part.metadata().values())
                }))?;
                prompt.clone()
            };
            Some(PendingTextInput::Prefill(copied))
        }
    })
}

impl<B, R, N> TextSnapshotBackend<TextSessionDriver<R, N>> for B
where
    B: TextSnapshotMechanisms,
    crate::text_session_driver::TextPromptFor<B>: TextPrefillInput,
    N: TextSessionMechanisms<
            B,
            Resources = R,
            Prefill = crate::text_session_driver::TextPromptFor<B>,
            TextValues: eredu_core::backend::NativeTextValues<InputTensor = B::InputTensor>,
        > + TextTransactionMechanisms<Error = B::Error>,
    R: std::ops::DerefMut<Target = TextSessionResourcesFor<B, N>> + 'static,
    <N as TextSessionMechanisms<B>>::Model: InstalledTextState,
    TextSessionDriver<R, N>: NativeTextStateSession<B>
        + eredu_core::backend::TextOperationDriver<
            B,
            Prompt = crate::text_session_driver::TextPromptFor<B>,
            TextGenerationState = super::TextRequestFor<B>,
        >,
{
    type SamplingState = SnapshotSamplingFor<B>;

    fn continuation_input_tokens(
        runtime: &ModelRuntime<Self, TextSessionDriver<R, N>>,
        sampling: &SnapshotSamplingFor<B>,
        pending: Option<
            PendingTextInput<&crate::text_session_driver::TextPromptFor<B>, &Self::Token>,
        >,
        predictions: u64,
    ) -> Option<u64> {
        use SnapshotPendingGeometry::*;
        if predictions == 0 {
            return Some(0);
        }
        let executable = runtime.session().resources().model.as_ref();
        let mode = sampling.decoding.mode;
        if let Some(geometry) = executable
            .block_refinement_geometry()
            .filter(|_| matches!(mode, DecodingMode::Diffusion { .. }))
        {
            let canvas = u64::from(geometry.canvas_length.get());
            if let Some(block) = refinement_facts::<B>(sampling) {
                return crate::generation::refinement::block::continuation_context_tokens(
                    canvas,
                    block.pending_positions,
                    predictions,
                );
            }
            let TextPrompt(prompt) = pending_geometry::<B>(pending?)? else {
                return None;
            };
            return prompt.checked_add((predictions - 1).checked_div(canvas)?.checked_mul(canvas)?);
        }
        let pending = pending?;
        if let DecodingMode::SelfSpeculation(options) = mode {
            if sampling.decoding.speculative.is_some()
                && matches!(
                    pending,
                    PendingTextInput::Decode(_) | PendingTextInput::Continue
                )
            {
                return predictions.checked_add(options.request().max_draft_tokens.get() as u64);
            }
        }
        let geometry = pending_geometry::<B>(match &pending {
            PendingTextInput::Prefill(prompt) => PendingTextInput::Prefill(*prompt),
            PendingTextInput::Decode(token) => PendingTextInput::Decode(*token),
            PendingTextInput::Continue => PendingTextInput::Continue,
        })?;
        let inputs = match geometry {
            Decode => Some(predictions),
            PrefillCursor(positions) | TextPrompt(positions) => {
                positions.checked_add(predictions - 1)
            }
            PreparedPrompt => {
                let PendingTextInput::Prefill(prompt) = pending else {
                    return None;
                };
                prompt
                    .with_borrowed(|view| executable.prepared_input_positions(view))?
                    .checked_add(predictions - 1)
            }
            Continue if sampling.decoding.next_logits.is_some() => predictions.checked_sub(1),
            Continue => None,
        };
        match mode {
            DecodingMode::SelfSpeculation(options) => {
                inputs?.checked_add(options.request().max_draft_tokens.get() as u64)
            }
            _ => inputs,
        }
    }

    fn decoding_settlement_input_tokens(
        runtime: &ModelRuntime<Self, TextSessionDriver<R, N>>,
        sampling: &SnapshotSamplingFor<B>,
        pending: Option<
            PendingTextInput<&crate::text_session_driver::TextPromptFor<B>, &Self::Token>,
        >,
    ) -> Option<u64> {
        match sampling.decoding.mode {
            DecodingMode::Diffusion { .. } => {
                let block = refinement_facts::<B>(sampling)?;
                block.context_tokens.or_else(|| {
                    runtime
                        .session()
                        .resources()
                        .model
                        .as_ref()
                        .block_refinement_geometry()
                        .map(|geometry| u64::from(geometry.canvas_length.get()))
                })
            }
            DecodingMode::SelfSpeculation(_) if sampling.decoding.speculative.is_some() => Some(1),
            DecodingMode::SelfSpeculation(_) => Some(u64::from(
                matches!(pending, Some(PendingTextInput::Decode(_)))
                    && !sampling.decoding.committed_token_cached,
            )),
            _ => Some(0),
        }
    }

    fn estimate_sampling_growth(
        runtime: &ModelRuntime<Self, TextSessionDriver<R, N>>,
        sampling: &SnapshotSamplingFor<B>,
        predictions: u64,
    ) -> Result<Option<u64>, Self::Error> {
        let session = runtime.session();
        let model = &session.resources().model;
        let geometry = model.as_ref().block_refinement_geometry();
        let shared = match (geometry.as_ref(), sampling.decoding.mode) {
            (Some(_), DecodingMode::SelfSpeculation(options)) => {
                let native = TextSessionDriver::<R, N>::estimate_native_text_state(runtime, None)?;
                let additional =
                    predictions.checked_add(options.request().max_draft_tokens.get() as u64);
                match (native, additional) {
                    (Some(native), Some(additional)) => {
                        session.ensure_no_submission_in_flight()?;
                        shared_verification_growth(model.control_state(), Some(native), additional)
                    }
                    _ => None,
                }
            }
            _ => Some(0),
        };
        Ok(estimate_sampling_growth::<B>(
            sampling,
            predictions,
            geometry,
            shared,
        ))
    }

    fn copy_sampling_state(
        runtime: &mut ModelRuntime<Self, TextSessionDriver<R, N>>,
        sampling: &SnapshotSamplingFor<B>,
    ) -> Result<SnapshotSamplingFor<B>, Self::Error> {
        let (backend, session) = runtime.parts_mut();
        // A rejected independent copy leaves the model unchanged. Settle native
        // work before propagating the copy error.
        session.with_model_transaction(|_| Ok(copy_sampling_state::<B>(backend, sampling)))?
    }

    fn copy_pending_input(
        runtime: &mut ModelRuntime<Self, TextSessionDriver<R, N>>,
        input: Option<
            PendingTextInput<&crate::text_session_driver::TextPromptFor<B>, &Self::Token>,
        >,
    ) -> Result<
        Option<PendingTextInput<crate::text_session_driver::TextPromptFor<B>, Self::Token>>,
        Self::Error,
    > {
        let (backend, session) = runtime.parts_mut();
        session.with_model_transaction(|_| Ok(copy_pending_input::<B>(backend, input)))?
    }

    fn sampling_state(state: &super::TextRequestFor<B>) -> &SnapshotSamplingFor<B> {
        &state.sampling
    }
    fn install_sampling_state(
        state: &mut super::TextRequestFor<B>,
        sampling: SnapshotSamplingFor<B>,
    ) {
        state.sampling = sampling;
    }
    fn assemble_generation_state(
        sampling: SnapshotSamplingFor<B>,
        #[cfg(feature = "tensor-capture")] capture: Option<CaptureSession>,
    ) -> super::TextRequestFor<B> {
        super::TextRequestState {
            sampling,
            #[cfg(feature = "tensor-capture")]
            capture,
            #[cfg(not(feature = "tensor-capture"))]
            capture: None,
            readout_selection: None,
            readout_rows: None,
        }
    }
    fn sampling_prediction(sampling: &SnapshotSamplingFor<B>) -> u64 {
        sampling.next_prediction
    }
    #[cfg(feature = "tensor-capture")]
    fn capture_run(state: &super::TextRequestFor<B>) -> Option<&CaptureSession> {
        state.capture.as_ref()
    }
    #[cfg(feature = "tensor-capture")]
    fn capture_run_mut(state: &mut super::TextRequestFor<B>) -> Option<&mut CaptureSession> {
        state.capture.as_mut()
    }
    fn estimate_sampling_state(
        _: &ModelRuntime<Self, TextSessionDriver<R, N>>,
        sampling: &SnapshotSamplingFor<B>,
    ) -> Result<Option<SnapshotEstimate>, Self::Error> {
        Ok(estimate_sampling_state::<B>(sampling))
    }
    fn estimate_pending_input(
        _: &ModelRuntime<Self, TextSessionDriver<R, N>>,
        input: Option<
            PendingTextInput<&crate::text_session_driver::TextPromptFor<B>, &Self::Token>,
        >,
    ) -> Result<Option<SnapshotEstimate>, Self::Error> {
        estimate_pending_input::<B>(input)
    }
    #[cfg(feature = "tensor-capture")]
    fn estimate_child_capture(
        _: &ModelRuntime<Self, TextSessionDriver<R, N>>,
        shape: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        <B as TextModeSnapshotNative>::child_capture_workspace().estimate(shape, selection, slice)
    }
    #[cfg(feature = "tensor-capture")]
    fn child_intervention_estimator(
        _: &ModelRuntime<Self, TextSessionDriver<R, N>>,
    ) -> Result<Arc<dyn InterventionEstimator>, CaptureError> {
        <B as TextModeSnapshotNative>::child_intervention_estimator()
    }
}

impl<B, R, N> crate::execution_control::TextSamplingControlBackend<TextSessionDriver<R, N>> for B
where
    B: TextSnapshotMechanisms,
    N: TextSessionMechanisms<B, Resources = R> + TextTransactionMechanisms<Error = B::Error>,
    R: std::ops::DerefMut<Target = TextSessionResourcesFor<B, N>> + 'static,
    TextSessionDriver<R, N>: eredu_core::backend::TextOperationDriver<
        B,
        Prompt = crate::text_session_driver::TextPromptFor<B>,
        TextGenerationState = super::TextRequestFor<B>,
    >,
{
    fn sampling_score_kind(
        _: &ModelRuntime<B, TextSessionDriver<R, N>>,
        state: &super::TextRequestFor<B>,
    ) -> eredu_core::refinement::PredictionScoreKind {
        use eredu_core::refinement::PredictionScoreKind;
        if matches!(state.sampling.decoding.mode, DecodingMode::Diffusion { .. }) {
            PredictionScoreKind::ConstrainedFinalization
        } else {
            PredictionScoreKind::CausalConditional
        }
    }

    fn sampling_control_facts(
        state: &super::TextRequestFor<B>,
    ) -> crate::execution_control::SamplingStateFacts {
        let sampling = &state.sampling;
        crate::execution_control::SamplingStateFacts {
            temperature: sampling.temperature,
            requires_positive_temperature: sampling.sampler.requires_positive_temperature(),
            has_rng: sampling.prng.is_some()
                || (matches!(sampling.decoding.mode, DecodingMode::Diffusion { .. })
                    && sampling.decoding.refinement.is_some()),
        }
    }

    fn install_sampling_override(
        runtime: &mut ModelRuntime<B, TextSessionDriver<R, N>>,
        state: &mut super::TextRequestFor<B>,
        request: crate::execution_control::ValidatedSamplingOverride,
    ) -> Result<(), B::Error> {
        let (backend, session) = runtime.parts_mut();
        let replacement = session.with_model_transaction(|model| {
            Ok(request
                .reseed()
                .map(|seed| {
                    let shared = model
                        .as_ref()
                        .capability_estimate()
                        .shared_backbone_draft()
                        .is_some();
                    let random = B::seeded_random(seed)?;
                    let split = shared
                        .then(|| B::split_random(backend, &random, [1, 2]))
                        .transpose()?;
                    Ok::<_, B::Error>((seed, random, split))
                })
                .transpose())
        })??;
        let sampling = &mut state.sampling;
        if let Some((seed, random, split)) = replacement {
            let block_random = split.as_ref().map_or(&random, |[refinement, _]| refinement);
            if let Some(refinement) = &mut sampling.decoding.refinement {
                refinement.replace_random(block_random.clone());
            }
            if let Some([refinement, proposal]) = split {
                sampling.decoding.refinement_random = Some(refinement);
                sampling.proposal_random = Some(proposal);
            }
            sampling.refinement_seed = seed;
            sampling.prng = Some(random);
        }
        sampling.temperature = request.temperature();
        Ok(())
    }
}
