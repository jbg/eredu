//! Coupled EAR-TTS feedback and waveform execution over candidate component state.
//!
//! Attention caches and sampling remain owned by the caller's unpublished
//! transaction. Any failure after a cache or random operation requires that
//! transaction's completion handling; an error does not make its resources safe
//! to reuse. These equations neither commit output nor introduce a scheduler.

use super::tts_conditioning_layered::{ConditioningExecution, ResidentConditioning};
use super::tts_mog::MogHead;
use super::tts_mog_layered::MogExecution;
use super::tts_rvq::RvqRefinement;
use super::tts_rvq_layered::{
    RefinementPhase, RefinementSampling, RefinementStage, RvqExecution, agree_result,
};
use eredu_codec::ear::AudioDecoderExecution;
use eredu_codec::ear_decoder::DecoderState;
use eredu_core::{BackendFailure, execution_control::SnapshotEstimate};
use eredu_nn::RetainedTensorRoots;
use eredu_runtime::execution_control::{
    SnapshotReservation, SnapshotValue, snapshot_value_estimate,
};

use std::sync::Arc;

use super::tts_transformer::TtsTransformer;
use super::tts_transformer_layered::{CausalTransformerExecution, TransformerExecution};
use eredu_codec::ear::{AudioDecoder, AudioDecoderConfig, AudioDecoderState};
use eredu_nn::{Index, NeuralBackend, Tensor, TensorElementType};

use super::{
    config::TtsConfig,
    tts_conditioning::CharacterMapping,
    tts_feedback::{FeedbackControls, OutputPolicy, WaveformSilence},
    tts_generator::{TtsFrame, TtsGenerator},
    tts_ingress::{PromptLayout, SpeakerPrompt},
    tts_mog::TtsError,
};

#[derive(Debug, Clone, PartialEq)]
struct SpeechPolicy {
    tts: TtsConfig,
    controls: FeedbackControls,
    codec: AudioDecoderConfig,
}

/// Retained feedback and codec carry; attention caches live in the same owner's
/// transaction, and must have the recorded causal position.
#[derive(Debug, Clone)]
pub struct SpeechState<T: Tensor, K = DecoderState<T>> {
    policy: Arc<SpeechPolicy>,
    feedback: T,
    codec: AudioDecoderState<T, K>,
    model_position: i32,
}

impl<T: Tensor + SnapshotValue, K: SnapshotValue> SnapshotValue for SpeechState<T, K> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        // Immutable policy is shared with the execution components; snapshot
        // allocation consists of the handle, copied feedback and codec carry.
        snapshot_value_estimate::<Self>([
            self.feedback.estimate_snapshot_value(),
            self.codec.estimate_snapshot_value(),
        ])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            policy: Arc::clone(&self.policy),
            feedback: self.feedback.copy_snapshot_value(retention)?,
            codec: self.codec.copy_snapshot_value(retention)?,
            model_position: self.model_position,
        })
    }
}

impl<T: Tensor, K: RetainedTensorRoots<T>> RetainedTensorRoots<T> for SpeechState<T, K> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        visitor(&self.feedback);
        self.codec.visit_retained_tensors(visitor);
    }
}
impl<T: Tensor, K> SpeechState<T, K> {
    /// Retained geometry used by sibling component validation.
    pub(super) fn tts_config(&self) -> &TtsConfig {
        &self.policy.tts
    }
    /// Autoregressive codes after the explicit EOS/PAD feedback policy.
    pub fn feedback(&self) -> &T {
        &self.feedback
    }
    /// Neural/spectral codec carry and observed frame/sample clocks.
    pub fn codec(&self) -> &AudioDecoderState<T, K> {
        &self.codec
    }
    /// Exclusive causal position shared by the conditional and guidance rows.
    pub const fn model_position(&self) -> i32 {
        self.model_position
    }
}

/// A retained partial refinement, bound to its predecessor and rendering policy.
/// Only the owner of the corresponding cache/RNG branch may render and commit it.
#[derive(Debug, Clone)]
pub struct SpeechFrame<T: Tensor, K = DecoderState<T>> {
    tts: TtsFrame<T>,
    predecessor: SpeechState<T, K>,
    subword: Vec<i32>,
    force_silence_on_eos: bool,
    silence_on_pad: Vec<bool>,
    waveform: WaveformSilence,
}
impl<T: Tensor, K> SpeechFrame<T, K> {
    /// Observed causal position, guidance hidden states and refinement progress.
    pub fn tts(&self) -> &TtsFrame<T> {
        &self.tts
    }
}

/// Unpublished waveform and all matching recurrent feedback/carry.
#[derive(Debug, Clone)]
pub struct SpeechOutput<T: Tensor, K = DecoderState<T>> {
    /// Streaming PCM `[batch, 1, samples]`, clamped to the released range.
    pub pcm: T,
    /// Generated codes before feedback or waveform substitutions.
    pub generated: T,
    /// Codec-safe codes actually rendered into `pcm`.
    pub waveform_codes: T,
    /// Candidate feedback/carry; adoption belongs to the shared transaction.
    pub state: SpeechState<T, K>,
}

impl<T: Tensor, K: RetainedTensorRoots<T>> RetainedTensorRoots<T> for SpeechOutput<T, K> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        visitor(&self.pcm);
        visitor(&self.generated);
        visitor(&self.waveform_codes);
        self.state.visit_retained_tensors(visitor);
    }
}

/// Borrowed neural components, independently materializable by their owner.
pub struct SpeechComponents<
    'a,
    B: NeuralBackend,
    X = TtsTransformer<B>,
    Y = MogHead<B>,
    Z = RvqRefinement<B>,
    W = ResidentConditioning<B>,
    A = AudioDecoder<B>,
    E = X,
> {
    policy: Arc<SpeechPolicy>,
    pub(super) generator: &'a mut TtsGenerator<B, X, Y, Z, W, E>,
    controls: &'a FeedbackControls,
    codec: &'a mut A,
}
impl<
    'a,
    B: NeuralBackend,
    X: TransformerExecution<B>,
    Y: MogExecution<B>,
    Z: RvqExecution<B>,
    W: ConditioningExecution<B>,
    A: AudioDecoderExecution<B>,
    E: TransformerExecution<B>,
> SpeechComponents<'a, B, X, Y, Z, W, A, E>
{
    /// Validates cross-component codebook geometry before any causal invocation.
    pub fn new(
        generator: &'a mut TtsGenerator<B, X, Y, Z, W, E>,
        controls: &'a FeedbackControls,
        codec: &'a mut A,
    ) -> Result<Self, TtsError> {
        let config = generator.config();
        let geometry = codec.config().quantizer;
        if geometry.depth != 31
            || controls.codebooks() != geometry.depth
            || controls.cardinality() != geometry.cardinality
            || config.codebook_size != geometry.cardinality as usize
            || config.latent_size != geometry.channels as usize
        {
            return Err(TtsError::Configuration(
                "speech component codebook geometry differs",
            ));
        }
        let policy = Arc::new(SpeechPolicy {
            tts: config.clone(),
            controls: controls.clone(),
            codec: codec.config().clone(),
        });
        Ok(Self {
            policy,
            generator,
            controls,
            codec,
        })
    }

    /// Prefills both guidance lanes and retains the last supplied code as initial
    /// feedback. No generated warmup code or random draw replaces this input.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill<S: ?Sized>(
        &mut self,
        codes: &B::Tensor,
        layout: &PromptLayout,
        speaker: SpeakerPrompt,
        conditioning: Option<&B::Tensor>,
        caches: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SpeechState<B::Tensor, A::NeuralState>, TtsError>
    where
        X: CausalTransformerExecution<B, S>,
    {
        if codes.shape() != [layout.batch(), layout.frames(), 31]
            || layout.frames() <= 0
            || codes.element_type() != Some(TensorElementType::I32)
        {
            return Err(TtsError::Shape(
                "speech prefill requires complete I32 prompt codes",
            ));
        }
        let codec = self.codec.initial_state(layout.batch())?;
        let feedback = codes
            .index(
                &[
                    Index::Full,
                    Index::Range(layout.frames() - 1, layout.frames()),
                    Index::Full,
                ],
                context,
            )?
            .compact(context)?;
        self.generator
            .prefill(codes, layout, speaker, conditioning, caches, context)?;
        Ok(SpeechState {
            policy: self.policy.clone(),
            feedback,
            codec,
            model_position: self.generator.transformer.causal_position(caches)?,
        })
    }

    /// Runs one causal invocation from spoken text and retained code feedback.
    /// Supplied conditioning must represent these current spoken text IDs.
    /// Main/tool/reminder policy is explicit; reminders preserve their PAD tail.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_frame<S: ?Sized>(
        &mut self,
        state: &SpeechState<B::Tensor, A::NeuralState>,
        current_subword: &[i32],
        conditioning: Option<&B::Tensor>,
        policy: OutputPolicy<'_>,
        caches: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SpeechFrame<B::Tensor, A::NeuralState>, TtsError>
    where
        X: CausalTransformerExecution<B, S>,
    {
        self.validate_frame(state, current_subword, policy, caches)?;
        let input = self
            .controls
            .prepare_input(&state.feedback, current_subword, context)?;
        let tts = self
            .generator
            .begin_frame(&input, conditioning, caches, context)?;
        Ok(SpeechFrame {
            tts,
            predecessor: state.clone(),
            subword: current_subword.to_vec(),
            force_silence_on_eos: policy.force_silence_on_eos,
            silence_on_pad: policy.silence_on_pad.to_vec(),
            waveform: policy.waveform,
        })
    }

    /// Conditions one spoken token per stream through the retained character
    /// mapping and actual T5Gemma encoder before advancing the causal speech
    /// model. Live BOS/EOS/PAD tokens keep true character masks. Spoken tool
    /// acknowledgments may differ from effective agent feedback to Nemotron.
    /// Character capacity and retained-state checks precede neural execution.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_text_frame<S: ?Sized>(
        &mut self,
        state: &SpeechState<B::Tensor, A::NeuralState>,
        spoken_tokens: &[i32],
        mapping: &CharacterMapping,
        max_character_cells: usize,
        policy: OutputPolicy<'_>,
        caches: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SpeechFrame<B::Tensor, A::NeuralState>, TtsError>
    where
        X: CausalTransformerExecution<B, S>,
    {
        self.validate_frame(state, spoken_tokens, policy, caches)?;
        let characters = mapping.prepare(
            spoken_tokens,
            None,
            state.feedback.dim(0),
            1,
            max_character_cells,
        )?;
        let conditioning = self.generator.condition_text(&characters, context)?;
        self.begin_frame(
            state,
            spoken_tokens,
            conditioning.as_ref(),
            policy,
            caches,
            context,
        )
    }

    fn validate_frame<S: ?Sized>(
        &self,
        state: &SpeechState<B::Tensor, A::NeuralState>,
        current_subword: &[i32],
        policy: OutputPolicy<'_>,
        caches: &S,
    ) -> Result<(), TtsError>
    where
        X: CausalTransformerExecution<B, S>,
    {
        self.validate_state(state)?;
        if state.codec.is_finished() {
            return Err(eredu_codec::Error::StreamFinished.into());
        }
        self.controls
            .validate_step(&state.feedback, current_subword)?;
        if self.generator.transformer.causal_position(caches)? != state.model_position
            || policy.silence_on_pad.len() != current_subword.len()
        {
            return Err(TtsError::Shape(
                "speech caches or output policy differ from retained state",
            ));
        }
        Ok(())
    }

    /// Advances the next active RVQ slot using the owner's candidate random draws.
    /// This operation never invokes the causal transformer again.
    pub fn advance_frame<F>(
        &mut self,
        frame: &SpeechFrame<B::Tensor, A::NeuralState>,
        mut sample: F,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SpeechFrame<B::Tensor, A::NeuralState>, TtsError>
    where
        F: RefinementSampling<B>,
    {
        self.agree_refinement_frame(frame, &mut sample)?;
        Ok(SpeechFrame {
            tts: self.generator.advance_frame(&frame.tts, sample, context)?,
            ..frame.clone()
        })
    }

    /// Uninterrupted refinement repeats the same retained transition.
    pub fn generate_frame<F>(
        &mut self,
        frame: &SpeechFrame<B::Tensor, A::NeuralState>,
        mut sample: F,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SpeechFrame<B::Tensor, A::NeuralState>, TtsError>
    where
        F: RefinementSampling<B>,
    {
        self.agree_refinement_frame(frame, &mut sample)?;
        Ok(SpeechFrame {
            tts: self.generator.generate_frame(&frame.tts, sample, context)?,
            ..frame.clone()
        })
    }

    fn agree_refinement_frame<F: RefinementSampling<B> + ?Sized>(
        &self,
        frame: &SpeechFrame<B::Tensor, A::NeuralState>,
        sample: &mut F,
    ) -> Result<(), TtsError> {
        let mut phase = RefinementPhase::at_slot(
            RefinementStage::SpeechValidation,
            frame.tts.refinement().next_slot(),
        );
        phase.words.push(frame.predecessor.model_position as u64);
        agree_result::<B, _, _>(sample, phase, self.validate_state(&frame.predecessor))
    }

    /// Applies output substitutions in reference order, then proposes actual
    /// codec carry with the generated waveform. Replaying this method creates
    /// another candidate; only the shared owner may publish one successor.
    pub fn render(
        &mut self,
        frame: &SpeechFrame<B::Tensor, A::NeuralState>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SpeechOutput<B::Tensor, A::NeuralState>, TtsError> {
        self.render_with_sampling(
            frame,
            |_: usize, _: &B::Tensor, _: f64| {
                Err(TtsError::Configuration(
                    "waveform rendering does not sample",
                ))
            },
            context,
        )
    }

    /// Uses the shared refinement authority to gate local output preparation
    /// before any codec collective and candidate publication after it.
    pub(super) fn render_with_sampling<F: RefinementSampling<B>>(
        &mut self,
        frame: &SpeechFrame<B::Tensor, A::NeuralState>,
        mut sample: F,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SpeechOutput<B::Tensor, A::NeuralState>, TtsError> {
        let prepared = (|| {
            self.validate_state(&frame.predecessor)?;
            let generated = frame
                .tts
                .codes()
                .ok_or(TtsError::Shape("speech refinement is incomplete"))?;
            let output = self.controls.finish_output(
                generated,
                &frame.subword,
                OutputPolicy {
                    force_silence_on_eos: frame.force_silence_on_eos,
                    silence_on_pad: &frame.silence_on_pad,
                    waveform: frame.waveform,
                },
                context,
            )?;
            Ok((generated, output))
        })();
        let mut phase = RefinementPhase::at_slot(
            RefinementStage::RenderPreparation,
            frame.tts.refinement().next_slot(),
        );
        phase.words.extend([
            frame.predecessor.model_position as u64,
            frame.tts.model_position() as u64,
            frame.force_silence_on_eos as u64,
            match frame.waveform {
                WaveformSilence::PreserveGenerated => 0,
                WaveformSilence::ApplySilence => 1,
            },
            frame.subword.len() as u64,
        ]);
        phase.words.extend(frame.subword.iter().map(|&n| n as u64));
        phase.words.push(frame.silence_on_pad.len() as u64);
        phase
            .words
            .extend(frame.silence_on_pad.iter().map(|&n| n as u64));
        let (generated, output) = agree_result::<B, _, _>(&mut sample, phase, prepared)?;
        let rendered = self
            .codec
            .step(&output.waveform, &frame.predecessor.codec, true, context)
            .map_err(TtsError::from);
        let (pcm, codec) = agree_result::<B, _, _>(
            &mut sample,
            RefinementPhase::at_slot(RefinementStage::Render, None),
            rendered,
        )?;
        Ok(SpeechOutput {
            pcm,
            generated: generated.clone(),
            waveform_codes: output.waveform,
            state: SpeechState {
                policy: self.policy.clone(),
                feedback: output.next_feedback,
                codec,
                model_position: frame.tts.model_position(),
            },
        })
    }

    /// Drains codec overlap without another text, TTS or code frame invocation.
    /// Empty and already drained streams have no output.
    pub fn finish(
        &mut self,
        state: &SpeechState<B::Tensor, A::NeuralState>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Option<B::Tensor>, SpeechState<B::Tensor, A::NeuralState>), TtsError> {
        self.validate_state(state)?;
        let (pcm, codec) = self.codec.finish(&state.codec, true, context)?;
        Ok((
            pcm,
            SpeechState {
                codec,
                ..state.clone()
            },
        ))
    }
    fn validate_state(
        &self,
        state: &SpeechState<B::Tensor, A::NeuralState>,
    ) -> Result<(), TtsError> {
        if state.policy != self.policy {
            return Err(TtsError::Shape(
                "speech state belongs to different components or policy",
            ));
        }
        Ok(())
    }
}
