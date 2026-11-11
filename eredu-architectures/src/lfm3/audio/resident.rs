//! Audio composition with selected language and speech input traversal.

#[cfg(feature = "audio")]
mod ingress;
mod transcription;
#[cfg(feature = "audio")]
pub use ingress::{
    AudioIngress, AudioIngressError, AudioIngressProgress, AudioIngressRequest,
    AudioIngressSnapshot,
};
#[cfg(feature = "audio")]
pub use transcription::{AudioPcmTranscription, AudioPcmTranscriptionSnapshot};
pub use transcription::{
    AudioTranscription, AudioTranscriptionBlock, AudioTranscriptionError,
    AudioTranscriptionSnapshot,
};

use super::{
    AudioDepth, AudioEncoder, AudioExecutionKernel, AudioLanguage, AudioMode, AudioPrompt,
    AudioSampling, AudioSchedulePolicy, AudioSession, AudioSessionError, AudioTokens,
    DepthDeclaration, DepthFrame, EncoderState, PreparedAudio, PreparedAudioCodec,
    PreparedAudioDepth, PreparedAudioEncoder, PreparedAudioLanguage,
};
use eredu_codec::liquid::{DecoderExecution, DecoderState};
use eredu_core::{
    execution_control::SnapshotEstimate, scheduler::SemanticStateTransaction, BackendFailure,
    Completion,
};
use eredu_media::audio::LogMelFeatures;
use eredu_nn::{AttentionCache, DistributedNeuralBackend, GroupedNeuralBackend, Index, Tensor};
use eredu_runtime::{
    execution_control::{
        snapshot_value_estimate, ForkStateEstimate, IsolatedStateCopy, IsolatedStateFork,
        SnapshotReservation, SnapshotRetainedGrowth, SnapshotValue,
    },
    DenseSamplingBackend, LayerRuntimeState, RuntimeStateComponents, SamplingBackend, StateLayout,
    SubmissionBackend,
};
use std::{marker::PhantomData, sync::Arc};

type Context<B> = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context;

/// Concrete binding and state mechanisms for the Audio graph.
pub trait ResidentAudioBinding<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
{
    /// State used for every declared neural layout.
    type State: LayerRuntimeState<
        B,
        LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    >;
    /// Language execution over the selected parameter policy.
    type Language: AudioLanguage<B, Self::State>;
    /// Speech encoder and projector over the selected parameter policy.
    type Encoder: AudioEncoder<B, Self::State>;
    /// Selected depth refinement and speech feedback execution.
    type Depth: AudioDepth<B>;
    /// Selected LiquidCodec synthesis.
    type Codec: DecoderExecution<B, Self::State>;

    /// Binds the exact LiquidCodec synthesis sources.
    fn bind_codec(&self, prepared: PreparedAudioCodec) -> Result<Self::Codec, BackendFailure>;

    /// Binds the retained speech output weights.
    fn bind_depth(&self, prepared: PreparedAudioDepth) -> Result<Self::Depth, BackendFailure>;

    /// Binds the retained speech encoder and projector.
    fn bind_encoder(&self, prepared: PreparedAudioEncoder)
        -> Result<Self::Encoder, BackendFailure>;

    /// Binds the exact language component with the selected parameter policy.
    fn bind_language(
        &self,
        prepared: PreparedAudioLanguage,
    ) -> Result<Self::Language, BackendFailure>;

    /// Native sampling failure retained as a source.
    type SamplingError: std::error::Error + Send + Sync + 'static;
    /// Existing sampling mechanism on the same context.
    type Sampling: DenseSamplingBackend<
        Logits = B::Tensor,
        Token = B::Tensor,
        Context = Context<B>,
        Error = Self::SamplingError,
    >;

    /// Selected execution context.
    fn context(&self) -> &Context<B>;
    /// Creates empty state with exact declared geometry.
    fn state(&self, layout: &StateLayout) -> Result<Self::State, BackendFailure>;
    /// Branches sequential state, isolating all mutable storage.
    fn branch_state(state: &Self::State) -> Result<Self::State, BackendFailure>;
    /// Collects state roots that must complete before canonical publication.
    fn retained_values(state: &Self::State) -> Vec<&B::Tensor>;

    /// Holds retained branch accounting while native work can outlive an advance.
    fn with_fork_retention<T>(
        retention: SnapshotRetainedGrowth,
        operation: impl FnOnce() -> T,
    ) -> Result<T, BackendFailure>;
}

/// Host input for one batch-one Audio session. Token IDs already contain the
/// placeholder repetitions corresponding to each processed clip or replay frame.
pub struct AudioRequest<'a> {
    /// Prompt IDs in checkpoint template order.
    pub tokens: &'a [i32],
    /// Finite processed speech clips in placeholder order.
    pub features: &'a [LogMelFeatures],
    /// Replayed speech frames in output-placeholder order.
    pub replay: &'a [[i32; 12]],
    /// Initial text/speech mode.
    pub mode: AudioMode,
    /// Finite bound on committed language slots.
    pub max_slots: usize,
}

/// Failure to bind an Audio graph or admit its initial prompt.
#[derive(Debug, thiserror::Error)]
pub enum ResidentAudioError {
    /// Prompt features or token budget disagree with the admitted geometry.
    #[error("invalid LFM3 Audio prompt geometry or context budget")]
    Prompt,
    /// Retained checkpoint metadata could not describe the selected graph.
    #[error(transparent)]
    Artifact(#[from] super::ArtifactError),
    /// The shared realtime driver could not start.
    #[error(transparent)]
    Session(#[from] AudioSessionError),
    /// A native mechanism or neural operation failed.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
}

/// Logical F32 parameter bytes for the resident Audio graph.
/// Cache state, native temporary arrays, and allocator overhead are separate.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct AudioResidentParameterBytes {
    /// Language weights, with the tied vocabulary table counted once.
    pub language: u64,
    /// Speech input encoder and projector weights.
    pub encoder: u64,
    /// Depth, tied codebook heads and feedback weights.
    pub depth: u64,
    /// LiquidCodec decoder weights and retained synthesis window.
    pub codec: u64,
}

/// Cold F32 Audio parameter and sampling selection.
pub struct PreparedResidentAudio {
    artifact: PreparedAudio,
    sampling: AudioSampling,
    language: PreparedAudioLanguage,
    encoder: PreparedAudioEncoder,
    depth: PreparedAudioDepth,
    codec: PreparedAudioCodec,
    report: AudioResidentParameterBytes,
}

impl PreparedResidentAudio {
    /// Retains exact binding recipes and checkpoint sampling settings.
    pub fn select(mut artifact: PreparedAudio) -> Result<Self, ResidentAudioError> {
        let sampling = AudioSampling::from_artifact(&artifact)?;
        let [language_scope, encoder_scope, depth_scope, codec_scope] =
            artifact.f32_weight_scopes()?;
        let (language_weights, language) = language_scope.finish("LFM3 Audio language")?;
        let (encoder_weights, encoder) = encoder_scope.finish("LFM3 Audio encoder")?;
        let (depth_weights, depth) = depth_scope.finish("LFM3 Audio depth")?;
        let (codec_weights, codec) = codec_scope.finish("LiquidCodec decoder")?;
        let language_parameters =
            PreparedAudioLanguage::new(artifact.language.clone(), &language_weights)?;
        artifact.language_weights = language_weights;
        let encoder_parameters =
            PreparedAudioEncoder::new(artifact.encoder.clone(), &encoder_weights)?;
        artifact.encoder_weights = encoder_weights;
        let depth_parameters = PreparedAudioDepth::new(artifact.depth.clone(), &depth_weights)?;
        artifact.depth_weights = depth_weights;
        let codec_parameters = PreparedAudioCodec::new(artifact.codec.clone(), &codec_weights)?;
        artifact.codec_weights = codec_weights;
        Ok(Self {
            artifact,
            sampling,
            language: language_parameters,
            encoder: encoder_parameters,
            depth: depth_parameters,
            codec: codec_parameters,
            report: AudioResidentParameterBytes {
                language,
                encoder,
                depth,
                codec,
            },
        })
    }

    /// Prepares one language pipeline stage without materializing checkpoint payloads.
    pub fn language_pipeline_partition(
        &self,
        rank: eredu_core::ParallelRankTopology,
    ) -> Result<super::PreparedAudioLanguagePartition, ResidentAudioError> {
        Ok(PreparedAudioLanguage::new(
            self.artifact.language.clone(),
            &self.artifact.language_weights,
        )?
        .select_pipeline_rank(rank)?)
    }

    /// Reports the selected logical F32 parameter bytes before device allocation.
    pub fn parameter_bytes(&self) -> AudioResidentParameterBytes {
        self.report
    }
}

/// Audio graph over selected language, speech input, depth, feedback and codec policies.
pub struct ResidentAudio<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>,
{
    language: H::Language,
    encoder: H::Encoder,
    depth: H::Depth,
    codec: H::Codec,
    binding: H,
    artifact: PreparedAudio,
    sampling: AudioSampling,
}

/// State retained on one sequential speech branch.
pub struct ResidentAudioState<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>,
{
    language: H::State,
    hidden: B::Tensor,
    codec: Option<DecoderState<H::State, B::Tensor>>,
    origin: Arc<()>,
    fork_retention: Option<SnapshotReservation>,
    binding: PhantomData<H>,
}

impl<B, H> SemanticStateTransaction for ResidentAudioState<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>,
{
    type Branch = Self;
    type Error = BackendFailure;

    fn branch(&self) -> Result<Self, Self::Error> {
        Ok(Self {
            language: H::branch_state(&self.language)?,
            hidden: self.hidden.clone(),
            codec: self
                .codec
                .as_ref()
                .map(|state| state.try_branch(H::branch_state))
                .transpose()?,
            origin: self.origin.clone(),
            fork_retention: self.fork_retention.clone(),
            binding: PhantomData,
        })
    }

    fn commit_branch(&mut self, branch: Self) -> Result<(), Self::Error> {
        *self = branch;
        Ok(())
    }
}

impl<B, H> IsolatedStateCopy for ResidentAudioState<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend<Tensor: SnapshotValue>,
    H: ResidentAudioBinding<B, State: SnapshotValue>,
{
    type Error = BackendFailure;

    fn estimate_isolated_copy(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            self.language.estimate_snapshot_value(),
            self.hidden.estimate_snapshot_value(),
            self.codec.estimate_snapshot_value(),
        ])
    }

    fn copy_isolated(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            language: self.language.copy_snapshot_value(retention)?,
            hidden: self.hidden.copy_snapshot_value(retention)?,
            codec: self.codec.copy_snapshot_value(retention)?,
            origin: self.origin.clone(),
            fork_retention: None,
            binding: PhantomData,
        })
    }

    fn validate_isolated_restore(&self, saved: &Self) -> Result<(), BackendFailure> {
        if !Arc::ptr_eq(&self.origin, &saved.origin) {
            return Err(failure(super::AudioSnapshotError::ForeignSession));
        }
        Ok(())
    }

    fn restore_isolated(&mut self, mut saved: Self) -> Result<(), BackendFailure> {
        self.validate_isolated_restore(&saved)?;
        saved.fork_retention = self.fork_retention.take();
        *self = saved;
        Ok(())
    }
}

impl<B, H> IsolatedStateFork for ResidentAudioState<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend<Tensor: SnapshotValue>,
    H: ResidentAudioBinding<B, State: SnapshotValue>,
{
    fn fork_isolated(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        let mut child = self.copy_isolated(retention)?;
        child.origin = Arc::new(());
        child.fork_retention = Some(retention.clone());
        Ok(child)
    }
}

impl<B, H> ForkStateEstimate for ResidentAudioState<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend<Tensor: SnapshotValue>,
    H: ResidentAudioBinding<B, State: SnapshotValue>,
{
    fn estimate_fork_branch(branch: &Self) -> Result<u64, BackendFailure> {
        Ok(branch.estimate_isolated_copy()?.retained_bytes)
    }
}

fn failure(error: impl std::error::Error + Send + Sync + 'static) -> BackendFailure {
    BackendFailure::from_error(error)
}

impl<B, H> ResidentAudio<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend + SubmissionBackend<Executor = Context<B>>,
    H: ResidentAudioBinding<B>,
    <H::Sampling as SamplingBackend>::RandomState: Clone,
{
    /// Prepares and binds the selected modules from retained sources.
    pub fn new(artifact: PreparedAudio, binding: H) -> Result<Self, ResidentAudioError> {
        Self::from_prepared(PreparedResidentAudio::select(artifact)?, binding)
    }

    /// Binds the exact cold selection without recomputing precision or sampling policy.
    pub fn from_prepared(
        prepared: PreparedResidentAudio,
        binding: H,
    ) -> Result<Self, ResidentAudioError> {
        let PreparedResidentAudio {
            artifact,
            sampling,
            language,
            encoder,
            depth,
            codec,
            ..
        } = prepared;
        let language = binding.bind_language(language)?;
        let encoder = binding.bind_encoder(encoder)?;
        let depth = binding.bind_depth(depth)?;
        let codec = binding.bind_codec(codec)?;
        Ok(Self {
            language,
            encoder,
            depth,
            codec,
            binding,
            artifact,
            sampling,
        })
    }

    /// Prefills a single prompt before returning the shared controlled driver.
    pub fn start(
        mut self,
        request: AudioRequest<'_>,
        random: <H::Sampling as SamplingBackend>::RandomState,
    ) -> Result<AudioSession<Self>, ResidentAudioError> {
        let preparation = (|| -> Result<_, ResidentAudioError> {
            let policy =
                AudioSchedulePolicy::from_artifact(&self.artifact, request.mode, request.max_slots)
                    .map_err(AudioSessionError::from)?;
            let sampling = self.sampling.clone();
            let mut tokens = request.tokens.to_vec();
            let original_last = *tokens.last().ok_or(ResidentAudioError::Prompt)?;
            if let Some(marker) = policy.prompt_marker(original_last) {
                tokens.push(marker);
            }
            let last = *tokens.last().ok_or(ResidentAudioError::Prompt)?;
            let replayed_eos = last == self.artifact.tokens.output
                && request
                    .replay
                    .last()
                    .is_some_and(|codes| codes.contains(&self.artifact.depth.eos_code()));
            let schedule = policy
                .start(last, replayed_eos)
                .map_err(AudioSessionError::from)?;
            let args = self.artifact.language.args();
            if tokens
                .len()
                .checked_add(request.max_slots)
                .is_none_or(|n| n > args.max_position_embeddings as usize)
            {
                return Err(ResidentAudioError::Prompt);
            }
            let ctx = self.binding.context();
            let mut clips = Vec::with_capacity(request.features.len());
            for features in request.features {
                if features.frames == 0
                    || features.mel_bins != 128
                    || features.frames.checked_mul(features.mel_bins) != Some(features.values.len())
                    || features.mask.len() != features.frames
                    || features.mask.iter().any(|&valid| !valid)
                    || features.values.iter().any(|&value| !value.is_finite())
                {
                    return Err(ResidentAudioError::Prompt);
                }
                let frames = i32::try_from(features.frames).map_err(failure)?;
                let input = B::Tensor::from_f32_slice(&features.values, &[1, frames, 128], ctx)
                    .and_then(|features| features.transpose_axes(&[0, 2, 1], ctx))
                    .map_err(failure)?;
                let mut state = self.binding.state(self.artifact.encoder.state_layout())?;
                clips.push(
                    self.encoder
                        .project(&input, &mut state, ctx)
                        .and_then(|rows| rows.squeeze_axes(&[0], ctx))
                        .map_err(failure)?,
                );
            }
            let speech_input = if clips.is_empty() {
                None
            } else {
                Some(B::Tensor::concatenate(&clips, 0, ctx).map_err(failure)?)
            };
            let speech_output = if request.replay.is_empty() {
                None
            } else {
                let count = i32::try_from(request.replay.len()).map_err(failure)?;
                let codes =
                    B::Tensor::from_i32_slice(request.replay.as_flattened(), &[count, 12], ctx)
                        .map_err(failure)?;
                Some(self.depth.embed(&codes, ctx).map_err(failure)?)
            };
            let language = self.binding.state(self.language.state_layout())?;
            Ok((
                schedule,
                sampling,
                tokens,
                speech_input,
                speech_output,
                language,
            ))
        })();
        let agreement = self.language.agree_audio(
            [
                0x41554453544152,
                request.max_slots as u64,
                request.mode as u64,
                0,
                0,
                0,
                0,
                0,
            ],
            preparation.is_ok(),
        );
        let (schedule, sampling, tokens, speech_input, speech_output, mut language) = preparation?;
        agreement?;
        let hidden = self.forward_prompt(
            &mut language,
            &tokens,
            speech_input.as_ref(),
            speech_output.as_ref(),
        )?;
        let state = ResidentAudioState {
            language,
            hidden,
            codec: None,
            origin: Arc::new(()),
            fork_retention: None,
            binding: PhantomData,
        };
        let completion = self.complete(&state);
        let agreement = self
            .language
            .agree_audio([0x41554453544143, 0, 0, 0, 0, 0, 0, 0], completion.is_ok());
        completion?;
        agreement?;
        self.language.commit_audio()?;
        Ok(AudioSession::new(self, state, schedule, sampling, random)?)
    }
}

fn last_position<T: Tensor>(hidden: &T, ctx: &T::Context) -> Result<T, eredu_nn::Error> {
    hidden.index(
        &[Index::Full, Index::At(hidden.dim(1) - 1), Index::Full],
        ctx,
    )
}

impl<B, H> ResidentAudio<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>,
{
    fn forward_prompt(
        &mut self,
        language: &mut H::State,
        tokens: &[i32],
        input: Option<&B::Tensor>,
        output: Option<&B::Tensor>,
    ) -> Result<B::Tensor, BackendFailure> {
        let ctx = self.binding.context();
        let prompt = AudioPrompt::new(
            tokens,
            self.artifact.tokens,
            self.artifact.language.args().hidden_size,
            input,
            output,
            ctx,
        )
        .map_err(failure);
        let agreement = self
            .language
            .agree_audio([0x41554450524f4d, 0, 0, 0, 0, 0, 0, 0], prompt.is_ok());
        let prompt = prompt?;
        agreement?;
        let position = self
            .language
            .forward_prompt(&prompt, language, ctx)
            .and_then(|hidden| last_position(&hidden, ctx).map_err(failure));
        let agreement = self
            .language
            .agree_audio([0x41554448494444, 0, 0, 0, 0, 0, 0, 0], position.is_ok());
        let position = position?;
        agreement?;
        Ok(position)
    }

    fn language_logits(&mut self, hidden: &B::Tensor) -> Result<B::Tensor, BackendFailure> {
        self.language
            .project_normalized(hidden, self.binding.context())
    }

    fn project_stream_chunk(
        &mut self,
        features: &LogMelFeatures,
        state: &mut EncoderState<H::State, B::Tensor>,
        final_chunk: bool,
    ) -> Result<Option<B::Tensor>, BackendFailure> {
        let context = self.binding.context();
        let frames = i32::try_from(features.frames).map_err(failure)?;
        let features = B::Tensor::from_f32_slice(&features.values, &[1, frames, 128], context)
            .and_then(|value| value.transpose_axes(&[0, 2, 1], context))
            .map_err(failure)?;
        let encoded = self
            .encoder
            .advance(&features, state, final_chunk, context)
            .map_err(failure)?;
        if encoded.dim(1) == 0 {
            return Ok(None);
        }
        encoded
            .squeeze_axes(&[0], context)
            .map(Some)
            .map_err(failure)
    }

    fn append(
        &mut self,
        state: &mut ResidentAudioState<B, H>,
        token: i32,
        output: Option<&B::Tensor>,
    ) -> Result<(), BackendFailure> {
        state.hidden = self.forward_prompt(&mut state.language, &[token], None, output)?;
        Ok(())
    }
}

impl<B, H> AudioExecutionKernel for ResidentAudio<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend + SubmissionBackend<Executor = Context<B>>,
    H: ResidentAudioBinding<B>,
{
    type Tensor = B::Tensor;
    type SamplingError = H::SamplingError;
    type SamplerBackend = H::Sampling;
    type State = ResidentAudioState<B, H>;
    type Completion = B::Completion;

    fn context(&self) -> &Context<B> {
        self.binding.context()
    }
    fn depth(&self) -> &DepthDeclaration {
        &self.artifact.depth
    }
    fn text_logits(&mut self, state: &Self::State) -> Result<B::Tensor, BackendFailure> {
        self.language_logits(&state.hidden)
    }
    fn depth_logits(
        &mut self,
        state: &Self::State,
        frame: &DepthFrame,
    ) -> Result<B::Tensor, BackendFailure> {
        self.depth
            .score_frame(&state.hidden, frame, self.binding.context())
            .map_err(failure)
    }
    fn append_text(&mut self, state: &mut Self::State, token: i32) -> Result<(), BackendFailure> {
        self.append(state, token, None)
    }
    fn append_speech(
        &mut self,
        state: &mut Self::State,
        codes: &[i32],
        eos: bool,
    ) -> Result<Vec<f32>, BackendFailure> {
        let ctx = self.binding.context();
        let local = B::Tensor::from_i32_slice(codes, &[1, 12], ctx)
            .and_then(|codes| {
                self.depth
                    .embed(&codes, ctx)
                    .map(|feedback| (codes, feedback))
            })
            .map_err(failure);
        let agreement = self.language.agree_audio(
            [0x41554446454544, u64::from(eos), 0, 0, 0, 0, 0, 0],
            local.is_ok(),
        );
        let (codes, feedback) = local?;
        agreement?;
        let AudioTokens { output, .. } = self.artifact.tokens;
        self.append(state, output, Some(&feedback))?;
        if eos {
            return Ok(Vec::new());
        }
        let ctx = self.binding.context();
        if state.codec.is_none() {
            state.codec = Some(
                self.codec
                    .decoder_state(
                        self.binding
                            .state(self.artifact.codec.decoder_state_layout())?,
                        1,
                    )
                    .map_err(failure)?,
            );
        }
        let codec = state
            .codec
            .as_mut()
            .ok_or_else(|| failure(ResidentAudioError::Prompt))?;
        self.codec
            .decode_chunk(
                Some(&codes.expand_dims(-1, ctx).map_err(failure)?),
                codec,
                false,
                ctx,
            )
            .map_err(failure)?
            .to_f32_vec(ctx)
            .map_err(failure)
    }
    fn flush_speech(&mut self, state: &mut Self::State) -> Result<Vec<f32>, BackendFailure> {
        let Some(mut codec) = state.codec.take() else {
            return Ok(Vec::new());
        };
        let ctx = self.binding.context();
        self.codec
            .decode_chunk(None, &mut codec, true, ctx)
            .map_err(failure)?
            .to_f32_vec(ctx)
            .map_err(failure)
    }
    fn complete(&mut self, state: &Self::State) -> Result<Self::Completion, BackendFailure> {
        let mut roots = H::retained_values(&state.language);
        roots.push(&state.hidden);
        if let Some(codec) = &state.codec {
            roots.extend(H::retained_values(codec.neural()));
            roots.extend(codec.overlap_values());
        }
        let completion = B::submit(self.binding.context(), roots).map_err(failure)?;
        completion.wait().map_err(failure)?;
        Ok(completion)
    }

    fn uses_collective_commit(&self) -> bool {
        self.language.uses_collective_commit()
    }

    fn agree_audio(&mut self, descriptor: [u64; 8], success: bool) -> Result<(), BackendFailure> {
        self.language.agree_audio(descriptor, success)
    }

    fn commit_audio(&mut self) -> Result<(), BackendFailure> {
        self.language.commit_audio()
    }

    fn with_fork_retention<T>(
        retention: SnapshotRetainedGrowth,
        operation: impl FnOnce() -> T,
    ) -> Result<T, BackendFailure> {
        H::with_fork_retention(retention, operation)
    }
}
