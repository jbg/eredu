//! Streaming input for one utterance in a resident Audio prompt.

use super::*;
use crate::lfm3::audio::{encoder::STRIDE, AudioSchedule};
use eredu_media::{audio::power_mel::PowerMelStream, MediaError};

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
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>,
{
    model: ResidentAudio<B, H>,
    mel: PowerMelStream<'static>,
    encoder: EncoderState<H::State, B::Tensor>,
    language: H::State,
    hidden: Option<B::Tensor>,
    prefix: Vec<i32>,
    suffix: Vec<i32>,
    schedule: AudioSchedule,
    random: <H::Sampling as SamplingBackend>::RandomState,
    max_samples: usize,
    progress: AudioIngressProgress,
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
            model: self,
            mel: transform.into_stream(),
            encoder,
            language,
            hidden: None,
            prefix: request.prefix.to_vec(),
            suffix,
            schedule,
            random,
            max_samples,
            progress: AudioIngressProgress::default(),
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
        self.progress
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
            language: self.language,
            hidden: self.hidden.ok_or(AudioIngressError::Prompt)?,
            codec: None,
            origin: Arc::new(()),
            fork_retention: None,
            binding: PhantomData,
        };
        let sampling = self.model.sampling.clone();
        Ok(AudioSession::new(
            self.model,
            state,
            self.schedule,
            sampling,
            self.random,
        )?)
    }

    /// Drops the incomplete input without starting a response.
    pub fn cancel(self) {}

    fn advance(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        final_chunk: bool,
    ) -> Result<(), AudioIngressError> {
        let remaining = self.max_samples - self.progress.samples;
        if samples.len() > remaining {
            return Err(AudioIngressError::Context {
                incoming: samples.len(),
                remaining,
            });
        }
        let mut mel = self.mel.clone();
        let features = mel.advance(samples, sample_rate, final_chunk)?;
        if features.frames == 0 && !final_chunk {
            self.mel = mel;
            self.progress.samples += samples.len();
            return Ok(());
        }
        let mut encoder = self.encoder.try_branch(H::branch_state)?;
        let projected = self
            .model
            .project_stream_chunk(&features, &mut encoder, final_chunk)?;
        let count = projected.as_ref().map_or(0, |rows| rows.dim(0) as usize);
        let mut tokens = Vec::new();
        if count != 0 && self.progress.audio_tokens == 0 {
            tokens.extend_from_slice(&self.prefix);
        }
        tokens.extend(std::iter::repeat_n(self.model.artifact.tokens.input, count));
        if final_chunk {
            tokens.extend_from_slice(&self.suffix);
        }
        let language = if tokens.is_empty() {
            None
        } else {
            let mut candidate = H::branch_state(&self.language)?;
            let hidden =
                self.model
                    .forward_prompt(&mut candidate, &tokens, projected.as_ref(), None)?;
            Some((candidate, hidden))
        };
        let mut roots = H::retained_values(encoder.backbone());
        roots.extend(encoder.feature_carry());
        if let Some((state, hidden)) = &language {
            roots.extend(H::retained_values(state));
            roots.push(hidden);
        }
        B::submit(self.model.binding.context(), roots)
            .and_then(|completion| completion.wait())
            .map_err(failure)?;
        if let Some((state, hidden)) = language {
            self.language = state;
            self.hidden = Some(hidden);
        }
        self.encoder = encoder;
        self.mel = mel;
        self.progress.samples += samples.len();
        self.progress.audio_tokens += count;
        Ok(())
    }
}
