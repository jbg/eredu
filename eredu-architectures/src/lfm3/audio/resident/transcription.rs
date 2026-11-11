//! Publisher LiveASR protocol over the selected speech encoder and language traversal.

use super::*;
use crate::lfm3::audio::encoder::STRIDE;
use crate::lfm3::audio::AudioSnapshotError;
use eredu_core::{
    execution_control::{
        ExecutionControlError, SnapshotLimits, SnapshotResourceKind, SnapshotUsage,
    },
    TokenFilter,
};
use eredu_runtime::execution_control::{
    IsolatedStateError, IsolatedStateSnapshot, SnapshotBudget, SnapshotRetainedGrowth,
};
use eredu_text::tokenizer::Tokenizer;
use std::{cell::RefCell, rc::Rc};

#[cfg(feature = "audio")]
mod pcm;
#[cfg(feature = "audio")]
pub use pcm::{AudioPcmTranscription, AudioPcmTranscriptionSnapshot};

const PROMPT: &str =
    "<|startoftext|><|im_start|>system\nPerform live ASR.<|im_end|>\n<|im_start|>assistant\n";

/// Output of a settled LiveASR block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioTranscriptionBlock {
    /// Generated token IDs, excluding the closing stop token.
    pub tokens: Vec<i32>,
    /// New projected audio positions appended by this block.
    pub audio_tokens: usize,
    /// Whether generation reached a closing token within the block budget.
    pub closed: bool,
}

/// Failure to prepare or advance a LiveASR stream.
#[derive(Debug, thiserror::Error)]
pub enum AudioTranscriptionError {
    /// Another stream is already advancing on the shared selected graph.
    #[error("the shared LFM3 Audio LiveASR graph is busy")]
    Busy,
    /// A child cannot retain the proposed state under its shared budget.
    #[error(transparent)]
    Retention(#[from] ExecutionControlError),
    /// Host PCM processing rejected the input.
    #[cfg(feature = "audio")]
    #[error(transparent)]
    Media(#[from] eredu_media::MediaError),
    /// The caller can drain a ready block before retrying the chunk.
    #[cfg(feature = "audio")]
    #[error("LiveASR input has {incoming} samples but only {available} fit before advancing")]
    Backpressure {
        /// Submitted samples.
        incoming: usize,
        /// Available sample slots.
        available: usize,
    },
    /// The PCM input was already finished.
    #[cfg(feature = "audio")]
    #[error("LFM3 Audio LiveASR PCM input is closed")]
    InputFinished,
    /// The retained tokenizer cannot construct the LiveASR prompt or policy.
    #[error("invalid LFM3 Audio LiveASR tokenizer or controls")]
    Policy,
    /// The submitted input differs from a finite, unpadded mel block.
    #[error("invalid LFM3 Audio LiveASR mel block")]
    Features,
    /// The block and its text reservation exceed the selected language context.
    #[error("LFM3 Audio LiveASR block exceeds the language context")]
    Context,
    /// The stream has finalized or was cancelled.
    #[error("LFM3 Audio LiveASR stream is closed")]
    Finished,
    /// Native execution failed before the block was published.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
}

struct Policy {
    prompt: Vec<i32>,
    text_end: i32,
    final_stops: [i32; 3],
    intermediate_filter: TokenFilter,
    final_filter: TokenFilter,
    frames_per_block: usize,
}

impl Policy {
    fn new(artifact: &PreparedAudio) -> Result<Self, AudioTranscriptionError> {
        use AudioTranscriptionError::Policy as Invalid;
        let tokenizer =
            Tokenizer::from_bytes(&artifact.text_assets.tokenizer).map_err(|_| Invalid)?;
        let vocab = artifact.language.args().vocab_size;
        let valid_id = |id: u32| i32::try_from(id).ok().filter(|id| (0..vocab).contains(id));
        let id = |token| {
            tokenizer
                .token_to_id(token)
                .and_then(valid_id)
                .ok_or(Invalid)
        };
        let prompt = tokenizer
            .encode(PROMPT, false)
            .map_err(|_| Invalid)?
            .get_ids()
            .iter()
            .map(|&id| valid_id(id).ok_or(Invalid))
            .collect::<Result<Vec<_>, _>>()?;
        let AudioTokens {
            input,
            output,
            audio_start,
            mixed_start,
            text_end,
        } = artifact.tokens;
        if prompt.is_empty() || prompt.iter().any(|&id| id == input || id == output) {
            return Err(Invalid);
        }
        let im_end = id("<|im_end|>")?;
        let eos = id("<|endoftext|>")?;
        let mut allowed = vec![true; vocab as usize];
        allowed[input as usize] = false;
        allowed[output as usize] = false;
        let final_filter = TokenFilter::allowed(allowed.clone()).map_err(|_| Invalid)?;
        for token in [audio_start, mixed_start, im_end, eos, id("<|mixed_end|>")?] {
            if token != text_end {
                allowed[token as usize] = false;
            }
        }
        let frames_per_block = artifact
            .interleaved_speech_slots()
            .checked_mul(STRIDE as usize)
            .ok_or(Invalid)?;
        Ok(Self {
            prompt,
            text_end,
            final_stops: [im_end, eos, id("<|im_start|>")?],
            intermediate_filter: TokenFilter::allowed(allowed).map_err(|_| Invalid)?,
            final_filter,
            frames_per_block,
        })
    }
}

/// Controlled, batch-one LiveASR stream. The encoder follows the audio across
/// language-context restarts. Each advance publishes one complete transcript block.
pub struct AudioTranscription<B, H, I = ()>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>,
{
    model: Rc<RefCell<ResidentAudio<B, H>>>,
    state: TranscriptionState<H::State, B::Tensor, I>,
    policy: Rc<Policy>,
    snapshot_budget: Option<SnapshotBudget>,
    cancelled: bool,
    fork: Option<ForkRetention<H::State, B::Tensor>>,
}

type EstimateModel<S, T> = fn(&EncoderState<S, T>, &S) -> Result<SnapshotEstimate, BackendFailure>;

struct ForkRetention<S, T: Tensor> {
    model: SnapshotRetainedGrowth,
    estimate_model: EstimateModel<S, T>,
    #[cfg(feature = "audio")]
    input: SnapshotRetainedGrowth,
}

struct TranscriptionState<S, T: Tensor, I> {
    encoder: EncoderState<S, T>,
    language: S,
    input: I,
    language_positions: usize,
    finished: bool,
    origin: Arc<()>,
}

fn estimate_model<S: SnapshotValue, T: Tensor + SnapshotValue>(
    encoder: &EncoderState<S, T>,
    language: &S,
) -> Result<SnapshotEstimate, BackendFailure> {
    snapshot_value_estimate::<()>([
        encoder.estimate_snapshot_value(),
        language.estimate_snapshot_value(),
    ])
}

impl<S: SnapshotValue, T: Tensor + SnapshotValue, I: SnapshotValue> IsolatedStateCopy
    for TranscriptionState<S, T, I>
{
    type Error = BackendFailure;

    fn estimate_isolated_copy(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            estimate_model(&self.encoder, &self.language),
            self.input.estimate_snapshot_value(),
        ])
    }

    fn copy_isolated(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            encoder: self.encoder.copy_snapshot_value(retention)?,
            language: self.language.copy_snapshot_value(retention)?,
            input: self.input.copy_snapshot_value(retention)?,
            origin: self.origin.clone(),
            ..*self
        })
    }

    fn validate_isolated_restore(&self, saved: &Self) -> Result<(), BackendFailure> {
        if Arc::ptr_eq(&self.origin, &saved.origin) {
            Ok(())
        } else {
            Err(failure(AudioSnapshotError::ForeignSession))
        }
    }

    fn restore_isolated(&mut self, saved: Self) -> Result<(), BackendFailure> {
        self.validate_isolated_restore(&saved)?;
        *self = saved;
        Ok(())
    }
}

/// Reusable saved LiveASR continuation with independently copied mutable state.
pub struct AudioTranscriptionSnapshot<B, H, I = ()>(
    IsolatedStateSnapshot<TranscriptionState<H::State, B::Tensor, I>>,
)
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>;

impl<B, H, I> AudioTranscription<B, H, I>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend<Tensor: SnapshotValue>,
    H: ResidentAudioBinding<B, State: SnapshotValue>,
    I: SnapshotValue,
{
    /// Configures snapshot limits once for this stream.
    pub fn enable_snapshots(&mut self, limits: SnapshotLimits) -> Result<(), AudioSnapshotError> {
        self.check_snapshot_boundary()?;
        if self.snapshot_budget.is_some() {
            return Err(AudioSnapshotError::AlreadyConfigured);
        }
        self.snapshot_budget = Some(SnapshotBudget::new(limits));
        Ok(())
    }

    /// Reports retained snapshots and cumulative copy charges.
    pub fn snapshot_usage(&self) -> Option<SnapshotUsage> {
        self.snapshot_budget.as_ref().map(SnapshotBudget::usage)
    }

    /// Captures the encoder, language and input state at the settled boundary.
    pub fn snapshot(&self) -> Result<AudioTranscriptionSnapshot<B, H, I>, AudioSnapshotError> {
        self.check_snapshot_boundary()?;
        let budget = self
            .snapshot_budget
            .as_ref()
            .ok_or(AudioSnapshotError::NotConfigured)?;
        Ok(AudioTranscriptionSnapshot(IsolatedStateSnapshot::capture(
            &self.state,
            budget,
        )?))
    }

    /// Restores a snapshot from this stream. Previously returned text belongs to the caller.
    pub fn restore(
        &mut self,
        saved: &AudioTranscriptionSnapshot<B, H, I>,
    ) -> Result<(), AudioSnapshotError> {
        self.check_snapshot_boundary()?;
        saved.0.restore(&mut self.state)?;
        Ok(())
    }

    /// Forks the current settled stream on the same selected graph.
    /// The child owns its continuation and shares the original copy budget.
    pub fn fork(&self) -> Result<Self, AudioSnapshotError> {
        self.check_snapshot_boundary()?;
        let budget = self
            .snapshot_budget
            .as_ref()
            .ok_or(AudioSnapshotError::NotConfigured)?;
        let model = estimate_model(&self.state.encoder, &self.state.language)
            .map_err(IsolatedStateError::Backend)?;
        let input = self
            .state
            .input
            .estimate_snapshot_value()
            .map_err(IsolatedStateError::Backend)?;
        // The child owns an origin and heap-backed reservation/high-water counters.
        let estimate = snapshot_value_estimate::<Self>([
            Ok(model),
            Ok(input),
            Ok(SnapshotEstimate {
                retained_bytes: 256,
                copy_bytes: 256,
            }),
        ])
        .map_err(IsolatedStateError::Backend)?;
        let retention = budget
            .reserve(SnapshotResourceKind::Branch, Some(estimate))
            .map_err(IsolatedStateError::Admission)?;
        let mut state = self
            .state
            .copy_isolated(&retention)
            .map_err(IsolatedStateError::Backend)?;
        state.origin = Arc::new(());
        Ok(Self {
            model: self.model.clone(),
            state,
            policy: self.policy.clone(),
            snapshot_budget: Some(budget.clone()),
            cancelled: false,
            fork: Some(ForkRetention {
                model: SnapshotRetainedGrowth::new(&retention, model.retained_bytes),
                estimate_model: estimate_model::<H::State, B::Tensor>,
                #[cfg(feature = "audio")]
                input: SnapshotRetainedGrowth::new(&retention, input.retained_bytes),
            }),
        })
    }

    fn check_snapshot_boundary(&self) -> Result<(), AudioSnapshotError> {
        if self.cancelled {
            return Err(AudioSnapshotError::Cancelled);
        }
        Ok(())
    }
}

impl<B, H> ResidentAudio<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>,
{
    /// Prepares the checkpoint's English LiveASR prompt without executing prefill.
    pub fn prepare_live_transcription(
        self,
    ) -> Result<AudioTranscription<B, H>, AudioTranscriptionError> {
        let policy = Policy::new(&self.artifact)?;
        self.prepare_transcription(policy, ())
    }

    fn prepare_transcription<I>(
        self,
        policy: Policy,
        input: I,
    ) -> Result<AudioTranscription<B, H, I>, AudioTranscriptionError> {
        let encoder = self
            .encoder
            .stream(self.binding.state(self.artifact.encoder.state_layout())?, 1)
            .map_err(failure)?;
        let language = self.binding.state(self.language.state_layout())?;
        Ok(AudioTranscription {
            model: Rc::new(RefCell::new(self)),
            state: TranscriptionState {
                encoder,
                language,
                input,
                language_positions: 0,
                finished: false,
                origin: Arc::new(()),
            },
            policy: Rc::new(policy),
            snapshot_budget: None,
            cancelled: false,
            fork: None,
        })
    }
}

impl<B, H, I> AudioTranscription<B, H, I>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend + SubmissionBackend<Executor = Context<B>>,
    H: ResidentAudioBinding<B>,
{
    /// The checkpoint's normal mel frame cadence, including the maximum final block.
    pub fn frames_per_block(&self) -> usize {
        self.policy.frames_per_block
    }

    /// Advances one live mel block, optionally restarting the text context.
    /// Blocks contain no batch padding. An empty final block closes the stream.
    pub fn advance(
        &mut self,
        features: &LogMelFeatures,
        final_chunk: bool,
        restart: bool,
    ) -> Result<AudioTranscriptionBlock, AudioTranscriptionError> {
        if let Some(fork) = &self.fork {
            H::with_fork_retention(fork.model.clone(), || {
                self.advance_block(features, final_chunk, restart)
            })?
        } else {
            self.advance_block(features, final_chunk, restart)
        }
    }

    fn advance_block(
        &mut self,
        features: &LogMelFeatures,
        final_chunk: bool,
        restart: bool,
    ) -> Result<AudioTranscriptionBlock, AudioTranscriptionError> {
        if self.is_finished() {
            return Err(AudioTranscriptionError::Finished);
        }
        if features.frames > self.frames_per_block()
            || features.mel_bins != 128
            || features.frames.checked_mul(features.mel_bins) != Some(features.values.len())
            || features.mask.len() != features.frames
            || features.mask.iter().any(|valid| !valid)
            || features.values.iter().any(|value| !value.is_finite())
        {
            return Err(AudioTranscriptionError::Features);
        }
        let mut position = if restart {
            0
        } else {
            self.state.language_positions
        };
        let pending = if position == 0 {
            self.policy.prompt.as_slice()
        } else {
            std::slice::from_ref(&self.policy.text_end)
        };
        let frames =
            i32::try_from(features.frames).map_err(|_| AudioTranscriptionError::Features)?;
        let count = self
            .state
            .encoder
            .projected_positions(frames, final_chunk)
            .ok_or(AudioTranscriptionError::Features)?;
        let budget = if final_chunk { 96 } else { 48 };
        let generate = count != 0 || final_chunk;
        let mut model = self
            .model
            .try_borrow_mut()
            .map_err(|_| AudioTranscriptionError::Busy)?;
        if generate
            && position
                .checked_add(pending.len())
                .and_then(|n| n.checked_add(count))
                .and_then(|n| n.checked_add(budget))
                .is_none_or(|n| n > model.artifact.language.args().max_position_embeddings as usize)
        {
            return Err(AudioTranscriptionError::Context);
        }
        let mut encoder = self.state.encoder.try_branch(H::branch_state)?;
        let projected = model.project_stream_chunk(features, &mut encoder, final_chunk)?;
        let mut language = if restart {
            Some(model.binding.state(model.language.state_layout())?)
        } else if generate {
            Some(H::branch_state(&self.state.language)?)
        } else {
            None
        };
        let mut block = AudioTranscriptionBlock {
            tokens: Vec::new(),
            audio_tokens: count,
            closed: false,
        };
        let mut hidden = None;
        if let Some(language) = language.as_mut().filter(|_| generate) {
            let mut tokens = pending.to_vec();
            tokens.extend(std::iter::repeat_n(model.artifact.tokens.input, count));
            let mut state = model.forward_prompt(language, &tokens, projected.as_ref(), None)?;
            let filter = if final_chunk {
                &self.policy.final_filter
            } else {
                &self.policy.intermediate_filter
            };
            for _ in 0..budget {
                let logits = model.language_logits(&state)?;
                let context = model.binding.context();
                let logits =
                    H::Sampling::apply_token_filter(&logits, filter, context).map_err(failure)?;
                let token =
                    H::Sampling::sample_raw(&logits, 0.0, None, context).map_err(failure)?;
                let token = i32::try_from(H::Sampling::token_id(&token, context).map_err(failure)?)
                    .map_err(failure)?;
                if token == self.policy.text_end
                    || final_chunk && self.policy.final_stops.contains(&token)
                {
                    block.closed = true;
                    break;
                }
                state = model.forward_prompt(language, &[token], None, None)?;
                block.tokens.push(token);
            }
            position += tokens.len() + block.tokens.len();
            hidden = Some(state);
        }
        let mut roots = H::retained_values(encoder.backbone());
        roots.extend(encoder.feature_carry());
        if let Some(language) = &language {
            roots.extend(H::retained_values(language));
        }
        roots.extend(hidden.as_ref());
        B::submit(model.binding.context(), roots)
            .and_then(|completion| completion.wait())
            .map_err(failure)?;
        if let Some(fork) = &self.fork {
            let estimate =
                (fork.estimate_model)(&encoder, language.as_ref().unwrap_or(&self.state.language))?;
            fork.model.admit(estimate.retained_bytes)?;
        }
        self.state.encoder = encoder;
        if let Some(language) = language {
            self.state.language = language;
        }
        self.state.language_positions = position;
        self.state.finished = final_chunk;
        Ok(block)
    }

    /// Whether the final block has been emitted or the stream was cancelled.
    pub fn is_finished(&self) -> bool {
        self.state.finished || self.cancelled
    }

    /// Stops between settled blocks.
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }
}
