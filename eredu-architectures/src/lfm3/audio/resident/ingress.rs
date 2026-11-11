//! Streaming input for one utterance in a resident Audio prompt.

use super::*;
use crate::lfm3::audio::{
    encoder::STRIDE, realtime::Transition, AudioSchedule, AudioSnapshotError,
};
use eredu_core::execution_control::{
    ExecutionControlError, SnapshotLimits, SnapshotResourceKind, SnapshotUsage,
};
use eredu_media::{
    audio::power_mel::{PowerMel, PowerMelState},
    MediaError,
};
use eredu_runtime::execution_control::{
    IsolatedStateError, IsolatedStateSnapshot, SnapshotBudget, SnapshotRetainedGrowth,
};
use std::{cell::RefCell, rc::Rc};

/// Text spans surrounding a single live utterance in checkpoint template order.
pub struct AudioIngressRequest<'a> {
    /// Prompt tokens preceding the audio positions, including any opening delimiter.
    pub prefix: &'a [i32],
    /// Prompt tokens following the audio positions, including the assistant prefix.
    pub suffix: &'a [i32],
    /// Initial response mode.
    pub mode: AudioMode,
    /// Maximum committed output slots reserved in the language context.
    pub max_slots: usize,
}

/// PCM and projected positions committed by successful input advances.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct AudioIngressProgress {
    /// Accepted mono samples, excluding right padding.
    pub samples: usize,
    /// Projected speech positions already appended to the language cache.
    pub audio_tokens: usize,
}

/// Failure while preparing or advancing streamed input.
#[derive(Debug, thiserror::Error)]
pub enum AudioIngressError {
    /// Another continuation is advancing on the same resident graph.
    #[error("the shared LFM3 Audio graph is busy")]
    Busy,
    /// A child cannot retain the proposed state under the shared budget.
    #[error(transparent)]
    Retention(#[from] ExecutionControlError),
    /// The child could not transfer its retained state to response generation.
    #[error(transparent)]
    Snapshot(#[from] AudioSnapshotError),
    /// The text spans or output reservation do not fit the streamed prompt.
    #[error("invalid LFM3 Audio streamed prompt")]
    Prompt,
    /// The chunk would exceed the encoder or language context after flush.
    #[error("audio chunk has {incoming} samples but only {remaining} fit in the prompt")]
    Context {
        /// Number of submitted samples.
        incoming: usize,
        /// Remaining sample capacity.
        remaining: usize,
    },
    /// Host audio processing rejected the input.
    #[error(transparent)]
    Media(#[from] MediaError),
    /// The response schedule or generation session could not be constructed.
    #[error(transparent)]
    Session(#[from] AudioSessionError),
    /// The selected backend could not complete this input step.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
}

/// Controlled preparation of a single streamed PCM utterance.
/// Completed input positions enter the language cache before the utterance ends.
pub struct AudioIngress<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend + SubmissionBackend<Executor = Context<B>>,
    H: ResidentAudioBinding<B>,
{
    transition: Rc<RefCell<Transition<ResidentAudio<B, H>>>>,
    transform: PowerMel,
    state: IngressState<H::State, B::Tensor>,
    snapshot_budget: Option<SnapshotBudget>,
    prefix: Arc<[i32]>,
    suffix: Arc<[i32]>,
    schedule: Rc<AudioSchedule>,
    random: <H::Sampling as SamplingBackend>::RandomState,
    max_samples: usize,
    fork: Option<IngressFork<B, H>>,
}

type EstimateIngress<S, T> = fn(
    &PowerMelState,
    &EncoderState<S, T>,
    &S,
    Option<&T>,
) -> Result<SnapshotEstimate, BackendFailure>;
type TransferFork<K> =
    fn(&mut AudioSession<K>, SnapshotRetainedGrowth) -> Result<(), AudioSnapshotError>;

struct IngressFork<B: GroupedNeuralBackend + DistributedNeuralBackend, H: ResidentAudioBinding<B>>
where
    B: SubmissionBackend<Executor = Context<B>>,
{
    growth: SnapshotRetainedGrowth,
    estimate: EstimateIngress<H::State, B::Tensor>,
    random_bytes: u64,
    transfer: TransferFork<ResidentAudio<B, H>>,
}

struct IngressState<S, T: Tensor> {
    mel: PowerMelState,
    encoder: EncoderState<S, T>,
    language: S,
    hidden: Option<T>,
    progress: AudioIngressProgress,
    origin: Arc<()>,
}

impl<S: SnapshotValue, T: Tensor + SnapshotValue> IsolatedStateCopy for IngressState<S, T> {
    type Error = BackendFailure;

    fn estimate_isolated_copy(&self) -> Result<SnapshotEstimate, BackendFailure> {
        estimate_ingress(
            &self.mel,
            &self.encoder,
            &self.language,
            self.hidden.as_ref(),
        )
    }

    fn copy_isolated(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            mel: self.mel.clone(),
            encoder: self.encoder.copy_snapshot_value(retention)?,
            language: self.language.copy_snapshot_value(retention)?,
            hidden: self.hidden.copy_snapshot_value(retention)?,
            origin: self.origin.clone(),
            progress: self.progress,
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

fn estimate_ingress<S: SnapshotValue, T: Tensor + SnapshotValue>(
    mel: &PowerMelState,
    encoder: &EncoderState<S, T>,
    language: &S,
    hidden: Option<&T>,
) -> Result<SnapshotEstimate, BackendFailure> {
    let bytes = mel
        .snapshot_storage_bytes()
        .ok_or_else(|| failure(ExecutionControlError::Overflow))?;
    snapshot_value_estimate::<IngressState<S, T>>(
        [
            Ok(SnapshotEstimate {
                retained_bytes: bytes,
                copy_bytes: bytes,
            }),
            encoder.estimate_snapshot_value(),
            language.estimate_snapshot_value(),
        ]
        .into_iter()
        .chain(hidden.map(T::estimate_snapshot_value)),
    )
}

/// Reusable snapshot of settled PCM, encoder and language prompt ingress.
pub struct AudioIngressSnapshot<B, H>(IsolatedStateSnapshot<IngressState<H::State, B::Tensor>>)
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>;

impl<B, H> AudioIngress<B, H>
where
    B: GroupedNeuralBackend
        + DistributedNeuralBackend<Tensor: SnapshotValue>
        + SubmissionBackend<Executor = Context<B>>,
    H: ResidentAudioBinding<B, State: SnapshotValue>,
{
    /// Configures the limits shared by prompt ingress and its eventual response.
    pub fn enable_snapshots(&mut self, limits: SnapshotLimits) -> Result<(), AudioSnapshotError> {
        if self.snapshot_budget.is_some() {
            return Err(AudioSnapshotError::AlreadyConfigured);
        }
        self.snapshot_budget = Some(SnapshotBudget::new(limits));
        Ok(())
    }

    /// Reports retention and cumulative copy work for the prompt.
    pub fn snapshot_usage(&self) -> Option<SnapshotUsage> {
        self.snapshot_budget.as_ref().map(SnapshotBudget::usage)
    }

    /// Saves settled input progress and its exact continuation.
    pub fn snapshot(&self) -> Result<AudioIngressSnapshot<B, H>, AudioSnapshotError> {
        let budget = self
            .snapshot_budget
            .as_ref()
            .ok_or(AudioSnapshotError::NotConfigured)?;
        Ok(AudioIngressSnapshot(IsolatedStateSnapshot::capture(
            &self.state,
            budget,
        )?))
    }

    /// Restores prompt ingress state before further samples are submitted.
    pub fn restore(
        &mut self,
        saved: &AudioIngressSnapshot<B, H>,
    ) -> Result<(), AudioSnapshotError> {
        saved.0.restore(&mut self.state)?;
        Ok(())
    }
}

impl<B, H> AudioIngress<B, H>
where
    B: GroupedNeuralBackend
        + DistributedNeuralBackend<Tensor: SnapshotValue>
        + SubmissionBackend<Executor = Context<B>>,
    H: ResidentAudioBinding<B, State: SnapshotValue>,
    <H::Sampling as SamplingBackend>::RandomState: Clone + SnapshotValue,
{
    /// Forks a settled prompt with an independent PCM and model continuation.
    pub fn fork(&self) -> Result<Self, AudioSnapshotError> {
        let budget = self
            .snapshot_budget
            .as_ref()
            .ok_or(AudioSnapshotError::NotConfigured)?;
        let random = self
            .random
            .estimate_snapshot_value()
            .map_err(IsolatedStateError::Backend)?;
        let state = self
            .state
            .estimate_isolated_copy()
            .map_err(IsolatedStateError::Backend)?;
        let continuation = snapshot_value_estimate::<()>([Ok(random), Ok(state)])
            .map_err(IsolatedStateError::Backend)?;
        // Includes the fresh restore origin and reservation metadata.
        let estimate = snapshot_value_estimate::<Self>([
            Ok(continuation),
            Ok(SnapshotEstimate {
                retained_bytes: 256,
                copy_bytes: 256,
            }),
        ])
        .map_err(IsolatedStateError::Backend)?;
        let retention = budget
            .reserve(SnapshotResourceKind::Branch, Some(estimate))
            .map_err(IsolatedStateError::Admission)?;
        let random_state = self
            .random
            .copy_snapshot_value(&retention)
            .map_err(IsolatedStateError::Backend)?;
        let mut state = self
            .state
            .copy_isolated(&retention)
            .map_err(IsolatedStateError::Backend)?;
        state.origin = Arc::new(());
        Ok(Self {
            transition: self.transition.clone(),
            transform: self.transform.clone(),
            state,
            snapshot_budget: Some(budget.clone()),
            prefix: self.prefix.clone(),
            suffix: self.suffix.clone(),
            schedule: self.schedule.clone(),
            random: random_state,
            max_samples: self.max_samples,
            fork: Some(IngressFork {
                growth: SnapshotRetainedGrowth::new(&retention, continuation.retained_bytes),
                estimate: estimate_ingress::<H::State, B::Tensor>,
                random_bytes: random.retained_bytes,
                transfer: AudioSession::inherit_fork_growth,
            }),
        })
    }
}

impl<B, H> ResidentAudio<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend + SubmissionBackend<Executor = Context<B>>,
    H: ResidentAudioBinding<B>,
    <H::Sampling as SamplingBackend>::RandomState: Clone,
{
    /// Admits token spans and reserves context for the output and final input window.
    pub fn prepare_ingress(
        self,
        request: AudioIngressRequest<'_>,
        random: <H::Sampling as SamplingBackend>::RandomState,
    ) -> Result<AudioIngress<B, H>, AudioIngressError> {
        let args = self.artifact.language.args();
        let AudioTokens { input, output, .. } = self.artifact.tokens;
        if request.prefix.iter().chain(request.suffix).any(|&token| {
            token == input || token == output || !(0..args.vocab_size).contains(&token)
        }) {
            return Err(AudioIngressError::Prompt);
        }
        let policy =
            AudioSchedulePolicy::from_artifact(&self.artifact, request.mode, request.max_slots)
                .map_err(AudioSessionError::from)?;
        let last = request.suffix.last().copied().unwrap_or(input);
        let marker = policy.prompt_marker(last);
        let schedule = policy
            .start(marker.unwrap_or(last), false)
            .map_err(AudioSessionError::from)?;
        let text_and_output = request
            .prefix
            .len()
            .checked_add(request.suffix.len())
            .and_then(|count| count.checked_add(usize::from(marker.is_some())))
            .and_then(|count| count.checked_add(request.max_slots))
            .ok_or(AudioIngressError::Prompt)?;
        let audio_capacity = (args.max_position_embeddings as usize)
            .checked_sub(text_and_output)
            .ok_or(AudioIngressError::Prompt)?
            .min(self.artifact.encoder.context_length());
        let transform = self.artifact.processor.transform()?;
        let max_samples = audio_capacity
            .checked_mul(STRIDE as usize)
            .and_then(|frames| transform.max_samples_for_frames(frames))
            .ok_or(AudioIngressError::Prompt)?;
        let encoder = self
            .encoder
            .stream(self.binding.state(self.artifact.encoder.state_layout())?, 1)
            .map_err(failure)?;
        let language = self
            .binding
            .state(self.artifact.language.definition().state_layout())?;
        let mut suffix = request.suffix.to_vec();
        suffix.extend(marker);
        Ok(AudioIngress {
            transition: Rc::new(RefCell::new(Transition(self))),
            transform,
            state: IngressState {
                mel: PowerMelState::default(),
                encoder,
                language,
                hidden: None,
                progress: AudioIngressProgress::default(),
                origin: Arc::new(()),
            },
            snapshot_budget: None,
            prefix: request.prefix.into(),
            suffix: suffix.into(),
            schedule: Rc::new(schedule),
            random,
            max_samples,
            fork: None,
        })
    }
}

impl<B, H> AudioIngress<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend + SubmissionBackend<Executor = Context<B>>,
    H: ResidentAudioBinding<B>,
    <H::Sampling as SamplingBackend>::RandomState: Clone,
{
    /// Reports committed input progress.
    pub fn progress(&self) -> AudioIngressProgress {
        self.state.progress
    }

    /// Submits arbitrary boundaries of mono 16-kHz PCM.
    /// Rejected chunks leave the input and both model caches at their previous positions.
    pub fn submit(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<AudioIngressProgress, AudioIngressError> {
        self.advance(samples, sample_rate, false)?;
        Ok(self.progress())
    }

    /// Flushes the trailing input and appends the text suffix, then enters the ordinary response driver.
    pub fn finish(mut self) -> Result<AudioSession<ResidentAudio<B, H>>, AudioIngressError> {
        self.advance(&[], 16_000, true)?;
        let state = ResidentAudioState {
            language: self.state.language,
            hidden: self.state.hidden.ok_or(AudioIngressError::Prompt)?,
            codec: None,
            origin: Arc::new(()),
            fork_retention: None,
            binding: PhantomData,
        };
        let sampling = self
            .transition
            .try_borrow()
            .map_err(|_| AudioIngressError::Busy)?
            .0
            .sampling
            .clone();
        let mut session = AudioSession::with_transition(
            self.transition,
            state,
            Rc::unwrap_or_clone(self.schedule),
            sampling,
            self.random,
        )?;
        session.inherit_snapshot_budget(self.snapshot_budget);
        if let Some(fork) = self.fork {
            (fork.transfer)(&mut session, fork.growth)?;
        }
        Ok(session)
    }

    /// Drops the incomplete input without starting a response.
    pub fn cancel(self) {}

    fn advance(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        final_chunk: bool,
    ) -> Result<(), AudioIngressError> {
        if let Some(fork) = &self.fork {
            H::with_fork_retention(fork.growth.clone(), || {
                self.advance_input(samples, sample_rate, final_chunk)
            })?
        } else {
            self.advance_input(samples, sample_rate, final_chunk)
        }
    }

    fn advance_input(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        final_chunk: bool,
    ) -> Result<(), AudioIngressError> {
        let remaining = self.max_samples - self.state.progress.samples;
        if samples.len() > remaining {
            return Err(AudioIngressError::Context {
                incoming: samples.len(),
                remaining,
            });
        }
        let mut mel = self.state.mel.clone();
        let features = self
            .transform
            .advance(&mut mel, samples, sample_rate, final_chunk)?;
        if features.frames == 0 && !final_chunk {
            self.admit_growth(
                &mel,
                &self.state.encoder,
                &self.state.language,
                self.state.hidden.as_ref(),
            )?;
            self.state.mel = mel;
            self.state.progress.samples += samples.len();
            return Ok(());
        }
        let mut transition = self
            .transition
            .try_borrow_mut()
            .map_err(|_| AudioIngressError::Busy)?;
        let model = &mut transition.0;
        let mut encoder = self.state.encoder.try_branch(H::branch_state)?;
        let projected = model.project_stream_chunk(&features, &mut encoder, final_chunk)?;
        let count = projected.as_ref().map_or(0, |rows| rows.dim(0) as usize);
        let mut tokens = Vec::new();
        if count != 0 && self.state.progress.audio_tokens == 0 {
            tokens.extend_from_slice(&self.prefix);
        }
        tokens.extend(std::iter::repeat_n(model.artifact.tokens.input, count));
        if final_chunk {
            tokens.extend_from_slice(&self.suffix);
        }
        let language = if tokens.is_empty() {
            None
        } else {
            let mut candidate = H::branch_state(&self.state.language)?;
            let hidden = model.forward_prompt(&mut candidate, &tokens, projected.as_ref(), None)?;
            Some((candidate, hidden))
        };
        let mut roots = H::retained_values(encoder.backbone());
        roots.extend(encoder.feature_carry());
        if let Some((state, hidden)) = &language {
            roots.extend(H::retained_values(state));
            roots.push(hidden);
        }
        B::submit(model.binding.context(), roots)
            .and_then(|completion| completion.wait())
            .map_err(failure)?;
        let (candidate, hidden) = language.as_ref().map_or(
            (&self.state.language, self.state.hidden.as_ref()),
            |(candidate, hidden)| (candidate, Some(hidden)),
        );
        self.admit_growth(&mel, &encoder, candidate, hidden)?;
        if let Some((state, hidden)) = language {
            self.state.language = state;
            self.state.hidden = Some(hidden);
        }
        self.state.encoder = encoder;
        self.state.mel = mel;
        self.state.progress.samples += samples.len();
        self.state.progress.audio_tokens += count;
        Ok(())
    }

    fn admit_growth(
        &self,
        mel: &PowerMelState,
        encoder: &EncoderState<H::State, B::Tensor>,
        language: &H::State,
        hidden: Option<&B::Tensor>,
    ) -> Result<(), AudioIngressError> {
        if let Some(fork) = &self.fork {
            let estimate = (fork.estimate)(mel, encoder, language, hidden)?;
            let bytes = estimate
                .retained_bytes
                .checked_add(fork.random_bytes)
                .ok_or(ExecutionControlError::Overflow)?;
            fork.growth.admit(bytes)?;
        }
        Ok(())
    }
}
