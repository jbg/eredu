//! Composed EAR-TTS inference over shared character, decoder and sampling paths.
//!
//! Causal caches are supplied as unpublished branches by the execution owner.
//! A frame retains both hidden guidance lanes and partial RVQ output; advancing
//! its refinement never reruns the causal transformer or advances its caches.

use super::tts_conditioning_layered::{
    ConditioningDimensions, ConditioningExecution, ResidentConditioning,
};
use super::tts_mog_layered::MogExecution;
use super::tts_rvq_layered::{
    BorrowedRefinementSampling, RefinementPhase, RefinementSampling, RefinementStage, RvqExecution,
    agree_result,
};
use super::tts_transformer_layered::{CausalTransformerExecution, TransformerExecution};
use eredu_nn::{Index, NeuralBackend, Parameterized, Tensor};

use super::{
    config::TtsConfig,
    tts_conditioning::CharacterBatch,
    tts_ingress::{PromptLayout, SpeakerPrompt},
    tts_mog::{MogDimensions, MogHead, TtsError},
    tts_rvq::{RefinementPolicy, RefinementState, RvqDimensions, RvqRefinement},
    tts_transformer::{TransformerConfig, TtsTransformer},
};

#[derive(Debug, Clone, Copy, PartialEq)]
struct FrameGeometry {
    hidden: i32,
    latent: i32,
    cardinality: i32,
    guidance: bool,
    policy: RefinementPolicy,
}

impl FrameGeometry {
    fn new(config: &TtsConfig, characters: i32, speaker_frames: i32) -> Result<Self, TtsError> {
        if config.context_hidden_size.is_some()
            || !config.use_gated_fusion_for_text_audio
            || !config.use_audio_prompt_frozen_projection
            || !config.backbone.causal
            || config.character_encoder.causal
            || characters <= 0
            || speaker_frames <= 0
        {
            return Err(TtsError::Configuration(
                "TTS generator requires the released conditioning equations",
            ));
        }
        let narrow =
            |n| i32::try_from(n).map_err(|_| TtsError::Configuration("TTS dimension exceeds I32"));
        Ok(Self {
            hidden: narrow(config.backbone.hidden_size)?,
            latent: narrow(config.latent_size)?,
            cardinality: narrow(config.codebook_size)?,
            guidance: config.guidance_enabled,
            policy: RefinementPolicy::from_config(config)?,
        })
    }
}

/// An unpublished speech frame, independent of learned parameter residency.
#[derive(Debug, Clone)]
pub struct TtsFrame<T: Tensor> {
    geometry: FrameGeometry,
    conditional: T,
    unconditional: Option<T>,
    refinement: RefinementState<T>,
    model_position: i32,
}
impl<T: Tensor> TtsFrame<T> {
    /// Conditional normalized Gemma output for this frame.
    pub fn conditional(&self) -> &T {
        &self.conditional
    }
    /// Unconditional normalized Gemma output, when guidance lanes are enabled.
    pub fn unconditional(&self) -> Option<&T> {
        self.unconditional.as_ref()
    }
    /// Partial code assignments and next active iteration.
    pub fn refinement(&self) -> &RefinementState<T> {
        &self.refinement
    }
    /// Complete raw generated codes; feedback substitutions are a separate policy.
    pub fn codes(&self) -> Option<&T> {
        self.refinement
            .next_slot()
            .is_none()
            .then(|| self.refinement.codes())
    }
    /// Exclusive Gemma position after the single causal invocation producing this frame.
    /// This is not the live audio frontier or a tool-channel substep count.
    pub const fn model_position(&self) -> i32 {
        self.model_position
    }
}

/// Independently materializable TTS components with one composed inference path.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TtsGenerator<
    B: NeuralBackend,
    X = TtsTransformer<B>,
    Y = MogHead<B>,
    Z = RvqRefinement<B>,
    W = ResidentConditioning<B>,
    E = X,
> {
    /// Independently leased character, ingress and fusion parameters.
    pub conditioning: W,
    /// Bidirectional character encoder blocks.
    pub character_transformer: E,
    /// Causal Gemma blocks and final normalization.
    pub transformer: X,
    /// Shared codebook embedding and iterative quantization.
    pub refinement: Z,
    /// Independently visitable Gaussian mixture stack and readout.
    pub mog: Y,
    #[parameter(skip)]
    geometry: FrameGeometry,
    #[parameter(skip)]
    config: TtsConfig,
    #[parameter(skip)]
    conditioning_dimensions: ConditioningDimensions,
    #[parameter(skip)]
    backend: std::marker::PhantomData<fn() -> B>,
}

/// Consuming assembly of units selected by the same architecture construction plan.
pub struct TtsGeneratorParts<
    B: NeuralBackend,
    X = TtsTransformer<B>,
    Y = MogHead<B>,
    Z = RvqRefinement<B>,
    W = ResidentConditioning<B>,
    E = X,
> {
    /// Independently leased character, ingress and fusion parameters.
    pub conditioning: W,
    /// Bidirectional character transformer and its selected parameter provider.
    pub character_transformer: E,
    /// Causal speech transformer and its selected parameter provider.
    pub transformer: X,
    /// Shared codebooks and iterative residual quantization.
    pub refinement: Z,
    /// Gaussian-mixture stack and readout.
    pub mog: Y,
    backend: std::marker::PhantomData<fn() -> B>,
}

impl<
    B: NeuralBackend,
    X: TransformerExecution<B>,
    Y: MogExecution<B>,
    Z: RvqExecution<B>,
    W: ConditioningExecution<B>,
    E: TransformerExecution<B>,
> TtsGeneratorParts<B, X, Y, Z, W, E>
{
    /// Collects selected providers for consuming generator assembly without new parameters.
    pub fn new(
        conditioning: W,
        character_transformer: E,
        transformer: X,
        refinement: Z,
        mog: Y,
    ) -> Self {
        Self {
            conditioning,
            character_transformer,
            transformer,
            refinement,
            mog,
            backend: std::marker::PhantomData,
        }
    }
}

impl<B: NeuralBackend> TtsGenerator<B> {
    /// Constructs the released inference equations from normalized configuration.
    /// Prompt codes and retained Aria latents are supplied/bound separately;
    /// construction never encodes a carrier waveform or reopens an artifact.
    pub fn new(
        config: &TtsConfig,
        characters: i32,
        speaker_frames: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, TtsError> {
        let geometry = FrameGeometry::new(config, characters, speaker_frames)?;
        let causal = TransformerConfig::new(&config.backbone)?;
        let character_config = TransformerConfig::new(&config.character_encoder)?;
        let mog = MogDimensions::from_config(config)?;
        let parts = TtsGeneratorParts::new(
            ResidentConditioning::new(
                ConditioningDimensions::from_config(config, characters, speaker_frames)?,
                context,
            )?,
            TtsTransformer::new(character_config, context)?,
            TtsTransformer::new(causal, context)?,
            RvqRefinement::new(
                geometry.cardinality,
                geometry.latent,
                geometry.hidden,
                context,
            )?,
            MogHead::new(mog, context)?,
        );
        Self::from_parts(config, characters, speaker_frames, parts)
    }
}

impl<
    B: NeuralBackend,
    X: TransformerExecution<B>,
    Y: MogExecution<B>,
    Z: RvqExecution<B>,
    W: ConditioningExecution<B>,
    E: TransformerExecution<B>,
> TtsGenerator<B, X, Y, Z, W, E>
{
    /// Assembles bound components without redeclaring learned parameter slots.
    pub fn from_parts(
        config: &TtsConfig,
        characters: i32,
        speaker_frames: i32,
        parts: TtsGeneratorParts<B, X, Y, Z, W, E>,
    ) -> Result<Self, TtsError> {
        let geometry = FrameGeometry::new(config, characters, speaker_frames)?;
        let generator = Self {
            conditioning: parts.conditioning,
            character_transformer: parts.character_transformer,
            transformer: parts.transformer,
            refinement: parts.refinement,
            mog: parts.mog,
            geometry,
            config: config.clone(),
            conditioning_dimensions: ConditioningDimensions::from_config(
                config,
                characters,
                speaker_frames,
            )?,
            backend: std::marker::PhantomData,
        };
        generator.validate_components()?;
        Ok(generator)
    }

    /// Consumes the owned modules for reassembly with independently prepared providers.
    /// This does not copy cache state or allocate replacement parameters.
    pub fn into_parts(self) -> TtsGeneratorParts<B, X, Y, Z, W, E> {
        TtsGeneratorParts {
            conditioning: self.conditioning,
            character_transformer: self.character_transformer,
            transformer: self.transformer,
            refinement: self.refinement,
            mog: self.mog,
            backend: std::marker::PhantomData,
        }
    }

    /// Exact normalized construction and sampling policy.
    pub fn config(&self) -> &TtsConfig {
        &self.config
    }

    /// Validates the selected causal state and observes its retained frontier.
    pub fn causal_position<S: ?Sized>(&self, state: &S) -> Result<i32, TtsError>
    where
        X: CausalTransformerExecution<B, S>,
    {
        Ok(self.transformer.causal_position(state)?)
    }

    /// Encodes a prepared character batch through the shared bidirectional path.
    /// An all-false token mask skips the entire character module, including flags.
    pub fn condition_text(
        &mut self,
        batch: &CharacterBatch,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, TtsError> {
        let transformer = &mut self.character_transformer;
        self.conditioning.character_forward(
            batch,
            |embedded, mask| Ok(transformer.forward_encoder(embedded, mask, context)?),
            context,
        )
    }

    /// Prefills speaker/text context without generating a code or drawing randomness.
    /// Cache branches must be empty and remain owned by the ordinary completion
    /// transaction. The released wrapper uses the final supplied prompt code as
    /// first recurrent feedback; it does not sample a code from this output.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill<S: ?Sized>(
        &mut self,
        codes: &B::Tensor,
        layout: &PromptLayout,
        speaker: SpeakerPrompt,
        conditioning: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, TtsError>
    where
        X: CausalTransformerExecution<B, S>,
    {
        if self.transformer.causal_position(state)? != 0 {
            return Err(TtsError::Shape(
                "TTS prompt prefill requires empty cache branches",
            ));
        }
        self.validate_components()?;
        let lanes = self.conditioning.prefill(
            &mut self.refinement,
            codes,
            layout,
            speaker,
            conditioning,
            self.geometry.guidance,
            context,
        )?;
        let embedded = self.conditioning.fuse(&lanes.audio, &lanes.text, context)?;
        Ok(self.transformer.forward_causal(&embedded, state, context)?)
    }

    /// Runs exactly one cached causal frame, then retains hidden lanes for
    /// explicit refinement. Feedback EOS substitutions happen before this call.
    /// The caller supplies unpublished caches; an error after submission does
    /// not establish native completion or permit reuse of a failed branch.
    pub fn begin_frame<S: ?Sized>(
        &mut self,
        codes: &B::Tensor,
        conditioning: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TtsFrame<B::Tensor>, TtsError>
    where
        X: CausalTransformerExecution<B, S>,
    {
        self.validate_components()?;
        if !matches!(codes.shape(), [batch, 1, 31] if *batch > 0)
            || self.transformer.causal_position(state)? <= 0
        {
            return Err(TtsError::Shape(
                "TTS frame requires one code row per batch and prefilled caches",
            ));
        }
        let batch = codes.dim(0);
        let refinement = self.refinement.start(batch, 1, context)?;
        let lanes = self.conditioning.decode(
            &mut self.refinement,
            codes,
            conditioning,
            self.geometry.guidance,
            context,
        )?;
        let embedded = self.conditioning.fuse(&lanes.audio, &lanes.text, context)?;
        let hidden = self.transformer.forward_causal(&embedded, state, context)?;
        let conditional = hidden
            .index(&[Index::Range(0, batch)], context)?
            .compact(context)?;
        let unconditional = if self.geometry.guidance {
            Some(
                hidden
                    .index(&[Index::Range(batch, batch * 2)], context)?
                    .compact(context)?,
            )
        } else {
            None
        };
        Ok(TtsFrame {
            geometry: self.geometry,
            conditional,
            unconditional,
            refinement,
            model_position: self.transformer.causal_position(state)?,
        })
    }

    /// Proposes one active refinement slot without touching causal caches.
    /// The sampler uses the session's candidate RNG; errors leave the parent
    /// frame unchanged but do not roll back that externally owned RNG.
    pub fn advance_frame<F>(
        &mut self,
        frame: &TtsFrame<B::Tensor>,
        mut sample: F,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TtsFrame<B::Tensor>, TtsError>
    where
        F: RefinementSampling<B>,
    {
        self.agree_frame(frame, &mut sample)?;
        let refinement = self.refinement.advance_sampled(
            &frame.refinement,
            &frame.conditional,
            frame.unconditional.as_ref(),
            self.geometry.policy,
            &mut self.mog,
            sample,
            context,
        )?;
        Ok(TtsFrame {
            refinement,
            ..frame.clone()
        })
    }

    /// Finishes a new or partial frame by repeating the controlled transition.
    /// No second autoregressive invocation or sampling engine is introduced.
    pub fn generate_frame<F>(
        &mut self,
        frame: &TtsFrame<B::Tensor>,
        mut sample: F,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TtsFrame<B::Tensor>, TtsError>
    where
        F: RefinementSampling<B>,
    {
        self.agree_frame(frame, &mut sample)?;
        let mut next = frame.clone();
        loop {
            sample.agree(
                RefinementPhase::at_slot(RefinementStage::Loop, next.refinement.next_slot()),
                true,
            )?;
            if next.refinement.next_slot().is_none() {
                break;
            }
            next = self.advance_frame(&next, BorrowedRefinementSampling(&mut sample), context)?;
        }
        Ok(next)
    }

    fn agree_frame<F: RefinementSampling<B> + ?Sized>(
        &self,
        frame: &TtsFrame<B::Tensor>,
        sample: &mut F,
    ) -> Result<(), TtsError> {
        let result = if frame.geometry == self.geometry {
            Ok(())
        } else {
            Err(TtsError::Shape(
                "TTS frame belongs to different generation geometry or policy",
            ))
        };
        let mut phase = RefinementPhase::at_slot(
            RefinementStage::FrameValidation,
            frame.refinement.next_slot(),
        );
        phase.words.extend([
            frame.model_position as u64,
            self.geometry.hidden as u64,
            self.geometry.latent as u64,
            self.geometry.cardinality as u64,
            self.geometry.guidance as u64,
            self.geometry.policy.guidance_scale.to_bits() as u64,
            self.geometry.policy.noise_scale.to_bits() as u64,
            self.geometry.policy.top_p.to_bits(),
        ]);
        agree_result::<B, _, _>(sample, phase, result)
    }

    fn validate_components(&self) -> Result<(), TtsError> {
        if self.conditioning.dimensions() != self.conditioning_dimensions
            || self.transformer.config().geometry() != &self.config.backbone
            || self.character_transformer.config().geometry() != &self.config.character_encoder
            || self.mog.dimensions() != MogDimensions::from_config(&self.config)?
            || self.refinement.dimensions() != RvqDimensions::from_config(&self.config)?
        {
            return Err(TtsError::Shape(
                "TTS components disagree with retained geometry",
            ));
        }
        Ok(())
    }
}
