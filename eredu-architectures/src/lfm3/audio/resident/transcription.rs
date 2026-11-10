//! Publisher LiveASR protocol over the resident speech encoder and language traversal.

use super::*;
use crate::lfm3::audio::encoder::STRIDE;
use eredu_core::TokenFilter;
use eredu_text::tokenizer::Tokenizer;

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
pub struct AudioTranscription<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>,
{
    model: ResidentAudio<B, H>,
    encoder: EncoderState<H::State, B::Tensor>,
    language: H::State,
    policy: Policy,
    language_positions: usize,
    finished: bool,
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
        let encoder = self
            .encoder
            .stream(self.binding.state(self.artifact.encoder.state_layout())?, 1)
            .map_err(failure)?;
        let language = self
            .binding
            .state(self.artifact.language.definition().state_layout())?;
        Ok(AudioTranscription {
            model: self,
            encoder,
            language,
            policy,
            language_positions: 0,
            finished: false,
        })
    }
}

impl<B, H> AudioTranscription<B, H>
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
        if self.finished {
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
        let mut position = if restart { 0 } else { self.language_positions };
        let pending = if position == 0 {
            self.policy.prompt.as_slice()
        } else {
            std::slice::from_ref(&self.policy.text_end)
        };
        let frames =
            i32::try_from(features.frames).map_err(|_| AudioTranscriptionError::Features)?;
        let count = self
            .encoder
            .projected_positions(frames, final_chunk)
            .ok_or(AudioTranscriptionError::Features)?;
        let budget = if final_chunk { 96 } else { 48 };
        let generate = count != 0 || final_chunk;
        if generate
            && position
                .checked_add(pending.len())
                .and_then(|n| n.checked_add(count))
                .and_then(|n| n.checked_add(budget))
                .is_none_or(|n| {
                    n > self.model.artifact.language.args().max_position_embeddings as usize
                })
        {
            return Err(AudioTranscriptionError::Context);
        }
        let mut encoder = self.encoder.try_branch(H::branch_state)?;
        let projected = self
            .model
            .project_stream_chunk(features, &mut encoder, final_chunk)?;
        let mut language = if restart {
            Some(
                self.model
                    .binding
                    .state(self.model.artifact.language.definition().state_layout())?,
            )
        } else if generate {
            Some(H::branch_state(&self.language)?)
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
            tokens.extend(std::iter::repeat_n(self.model.artifact.tokens.input, count));
            let mut state =
                self.model
                    .forward_prompt(language, &tokens, projected.as_ref(), None)?;
            let filter = if final_chunk {
                &self.policy.final_filter
            } else {
                &self.policy.intermediate_filter
            };
            for _ in 0..budget {
                let logits = self.model.language_logits(&state)?;
                let context = self.model.binding.context();
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
                state = self.model.forward_prompt(language, &[token], None, None)?;
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
        B::submit(self.model.binding.context(), roots)
            .and_then(|completion| completion.wait())
            .map_err(failure)?;
        self.encoder = encoder;
        if let Some(language) = language {
            self.language = language;
        }
        self.language_positions = position;
        self.finished = final_chunk;
        Ok(block)
    }

    /// Stops between settled blocks.
    pub fn cancel(&mut self) {
        self.finished = true;
    }
}
