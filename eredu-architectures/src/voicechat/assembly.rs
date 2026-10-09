//! Consuming speech assembly over independently constructed and bound units.
//!
//! The aggregate connects to the existing coupled executor. It owns learned
//! modules, not session caches, scheduling, or a full-graph residency policy.

use super::function_head_layered::{FunctionHeadDimensions, FunctionHeadExecution};
use super::tts_conditioning_layered::{
    ConditioningDimensions, ConditioningExecution, ResidentConditioning,
};
use super::tts_mog_layered::MogExecution;
use super::tts_rvq::{RvqDimensions, RvqRefinement};
use super::tts_rvq_layered::RvqExecution;
use super::tts_transformer_layered::TransformerExecution;
use eredu_codec::ear_decoder_layered::DecoderExecution;
use eredu_codec::prvq_layered::QuantizerExecution;
use std::collections::BTreeMap;

use eredu_codec::{
    ear::{AudioDecoder, AudioEncoder},
    ear_decoder::NeuralDecoder,
    ear_encoder::NeuralEncoder,
    ear_encoder_layered::EncoderExecution,
    prvq::ResidualQuantizer,
};
use eredu_core::BackendFailure;
use eredu_nn::{
    DistributedNeuralBackend, GroupedNeuralBackend, NeuralBackend, Parameterized, Tensor,
};
use eredu_runtime::TokenDomain;

use super::{
    channels::{ChannelFusion, FunctionHead},
    codec::{waveform_decoder_config, waveform_encoder_config},
    construction::{ConstructionError, SpeechConstructionPlan, SpeechUnit, SpeechUnitId},
    execution::Components,
    host::PreparedHostConditioning,
    perception::{Perception, PerceptionDimensions},
    perception_layered::PerceptionExecution,
    perception_rnnt::{PerceptionRnnt, PerceptionRnntError, PerceptionRnntLimits},
    rnnt::{Rnnt, RnntDimensions, RnntExecution},
    speech::SpeechComponents,
    tts_generator::{TtsGenerator, TtsGeneratorParts},
    tts_mog::{MogDimensions, MogHead, TtsError},
    tts_prompt::{AriaPromptPlan, PreparedTtsPrompt},
    tts_transformer::{TransformerConfig, TtsTransformer},
};

/// Failed construction or binding publishes no assembled aggregate.
#[derive(Debug, thiserror::Error)]
pub enum AssemblyError {
    /// A typed architecture construction or source declaration failed.
    #[error(transparent)]
    Construction(#[from] ConstructionError),
    /// The binder failed; native and application errors retain their source.
    #[error("VoiceChat binding failed for {unit:?}: {source}")]
    Binding {
        /// Exact unit whose binding failed.
        unit: SpeechUnitId,
        /// Original backend-independent failure wrapper.
        #[source]
        source: BackendFailure,
    },
    /// A binder replaced a unit with a different semantic kind.
    #[error("VoiceChat assembly received the wrong unit for {0:?}")]
    Unit(SpeechUnitId),
    /// Connected perception and RNNT geometry disagrees.
    #[error(transparent)]
    Input(#[from] PerceptionRnntError),
    /// Speech component geometry or prompt preparation failed.
    #[error(transparent)]
    Speech(#[from] TtsError),
    /// Codec unit order or geometry disagrees.
    #[error(transparent)]
    Codec(#[from] eredu_codec::Error),
    /// A selected provider does not implement the admitted component geometry.
    #[error("VoiceChat provider {0:?} disagrees with the admitted configuration")]
    ProviderGeometry(SpeechProviderKind),
}

/// Semantic component whose selected execution provider fails assembly validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechProviderKind {
    /// Shared perception feature encoder and projection.
    Perception,
    /// Recurrent transcription prediction and joint network.
    Rnnt,
    /// Function-channel vocabulary projection.
    FunctionHead,
    /// Character conditioning, causal TTS and iterative sampling policy.
    Generator,
    /// Startup waveform encoder.
    CodecEncoder,
    /// Streaming waveform decoder.
    CodecDecoder,
    /// The shared waveform codec codebooks.
    CodecQuantizer,
}

/// Independently prepared speech providers consumed by one shared coupled executor.
///
/// Providers retain their own residency mechanisms. Assembly validates their
/// admitted geometry and allocates only derived spectral constants; it does not
/// acquire learned parameters, execute providers, or create inference caches.
pub struct SpeechProviderParts<
    B: NeuralBackend,
    P = Perception<B>,
    X = TtsTransformer<B>,
    Y = MogHead<B>,
    Z = RvqRefinement<B>,
    W = ResidentConditioning<B>,
    V = NeuralDecoder<B>,
    Q = ResidualQuantizer<B>,
    H = NeuralEncoder<B>,
    F = FunctionHead<B>,
    J = Rnnt<B>,
> {
    /// Shared perception encoder and projection.
    pub perception: P,
    /// Retained recurrent transcription parameters.
    pub rnnt: J,
    /// Function-channel projection from the normalized backbone output.
    pub function: F,
    /// Character, transformer, mixture and RVQ execution providers.
    pub generator: TtsGenerator<B, X, Y, Z, W>,
    /// Finite startup encoder provider.
    pub encoder: H,
    /// Streaming waveform decoder provider.
    pub decoder: V,
    /// Single shared set of codec codebooks.
    pub quantizer: Q,
}

/// Owned learned components used by the existing coupled transition.
///
/// Construction consumes each unit once. The encoder borrows the decoder's
/// quantizer for startup; it never constructs a second set of codec codebooks.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct LearnedSpeechComponents<
    B: NeuralBackend,
    P = Perception<B>,
    X = TtsTransformer<B>,
    Y = MogHead<B>,
    Z = RvqRefinement<B>,
    W = ResidentConditioning<B>,
    V = NeuralDecoder<B>,
    Q = ResidualQuantizer<B>,
    H = NeuralEncoder<B>,
    F = FunctionHead<B>,
    J = Rnnt<B>,
> {
    /// Shared learned perception and recurrent transcription.
    pub input: PerceptionRnnt<B, P, J>,
    /// Function-channel readout of the normalized Nemotron result.
    pub function: F,
    /// Character, causal TTS, mixture and RVQ execution.
    pub generator: TtsGenerator<B, X, Y, Z, W>,
    /// Startup silent-carrier encoder, separate from live waveform decoding.
    pub encoder: AudioEncoder<B, H>,
    /// Waveform decoder and the single retained set of codec codebooks.
    pub decoder: AudioDecoder<B, V, Q>,
    #[parameter(skip)]
    fusion: ChannelFusion,
    #[parameter(skip)]
    prompt: AriaPromptPlan,
}

impl<
        B: NeuralBackend,
        P: PerceptionExecution<B>,
        X: TransformerExecution<B>,
        Y: MogExecution<B>,
        Z: RvqExecution<B>,
        W: ConditioningExecution<B>,
        V: DecoderExecution<B>,
        Q: QuantizerExecution<B>,
        H: EncoderExecution<B>,
        F: FunctionHeadExecution<B>,
        J: RnntExecution<B>,
    > LearnedSpeechComponents<B, P, X, Y, Z, W, V, Q, H, F, J>
{
    /// Prepares fixed tool-channel silence independently of live input and RNNT state.
    pub fn prepare_silence(
        &mut self,
        prepared: &super::silence::PreparedSilenceConditioning,
        output_dtype: eredu_nn::TensorElementType,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, super::silence::SilenceConditioningError> {
        self.input.prepare_silence(prepared, output_dtype, context)
    }

    /// Prepares fixed-Aria startup with the same codebooks used for decoding.
    /// Cache ownership and prefill execution remain with the enclosing session.
    pub fn prepare_prompt(
        &mut self,
        batch: i32,
        max_token_positions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PreparedTtsPrompt<B::Tensor>, TtsError> {
        self.prompt.encode(
            &mut self.encoder,
            &mut self.decoder.quantizer,
            batch,
            max_token_positions,
            context,
        )
    }
}

impl<
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        P: PerceptionExecution<B>,
        X: TransformerExecution<B>,
        Y: MogExecution<B>,
        Z: RvqExecution<B>,
        W: ConditioningExecution<B>,
        V: DecoderExecution<B>,
        Q: QuantizerExecution<B>,
        H: EncoderExecution<B>,
        F: FunctionHeadExecution<B>,
        J: RnntExecution<B>,
    > LearnedSpeechComponents<B, P, X, Y, Z, W, V, Q, H, F, J>
{
    /// Borrows the coupled executor around the supplied backbone provider.
    /// Neither an execution engine nor an independent transaction is created.
    pub fn execution<'a, R>(
        &'a mut self,
        backbone: R,
        host: &'a PreparedHostConditioning,
        input_limits: PerceptionRnntLimits,
        max_character_cells: usize,
    ) -> Result<
        Components<'a, B, R, PerceptionRnnt<B, P, J>, X, Y, Z, W, AudioDecoder<B, V, Q>, F>,
        TtsError,
    > {
        Ok(Components {
            input: &mut self.input,
            backbone,
            function: &mut self.function,
            fusion: &self.fusion,
            speech: SpeechComponents::new(&mut self.generator, &host.feedback, &mut self.decoder)?,
            characters: &host.characters,
            max_character_cells,
            input_limits,
            vocabulary: TokenDomain::new(131072),
        })
    }
}

impl SpeechConstructionPlan {
    /// Checks the mechanisms used to construct derived waveform constants.
    /// This reads backend capability facts without allocating native resources,
    /// so construction drivers can reject missing mechanisms before binding.
    pub fn validate_spectral_capabilities<B: NeuralBackend>(
        &self,
    ) -> Result<(), ConstructionError> {
        let decoder = waveform_decoder_config(&self.config().codec)?;
        B::require_operator_capabilities(
            "VoiceChat speech assembly",
            decoder.spectral.required_capabilities().union(
                eredu_codec::spectral::PeriodicHannStft::<B::Tensor>::required_capabilities(),
            ),
        )?;
        Ok(())
    }

    /// Consumes independently selected providers under the admitted architecture policy.
    /// All component geometry is checked before allocating spectral constants.
    /// This uses the ordinary coupled executor and does not choose residency budgets.
    pub fn assemble_providers<
        B: NeuralBackend,
        P: PerceptionExecution<B>,
        X: TransformerExecution<B>,
        Y: MogExecution<B>,
        Z: RvqExecution<B>,
        W: ConditioningExecution<B>,
        V: DecoderExecution<B>,
        Q: QuantizerExecution<B>,
        H: EncoderExecution<B>,
        F: FunctionHeadExecution<B>,
        J: RnntExecution<B>,
    >(
        &self,
        parts: SpeechProviderParts<B, P, X, Y, Z, W, V, Q, H, F, J>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LearnedSpeechComponents<B, P, X, Y, Z, W, V, Q, H, F, J>, AssemblyError> {
        let config = self.config();
        let perception = PerceptionDimensions::from_config(&config.perception)
            .map_err(ConstructionError::from)?;
        let rnnt = RnntDimensions::from_config(&config.rnnt).map_err(ConstructionError::from)?;
        let encoder = waveform_encoder_config(&config.codec)?;
        let decoder = waveform_decoder_config(&config.codec)?;
        let prompt = AriaPromptPlan::from_config(config, usize::MAX)?;
        let function = FunctionHeadDimensions {
            width: perception.output_width,
            vocabulary: 131072,
        };
        for (matches, kind) in [
            (
                parts.perception.dimensions() == perception,
                SpeechProviderKind::Perception,
            ),
            (parts.rnnt.dimensions() == rnnt, SpeechProviderKind::Rnnt),
            (
                parts.function.dimensions() == function,
                SpeechProviderKind::FunctionHead,
            ),
            (
                parts.generator.config() == &config.tts,
                SpeechProviderKind::Generator,
            ),
            (
                parts.encoder.config() == &encoder.neural,
                SpeechProviderKind::CodecEncoder,
            ),
            (
                parts.decoder.config() == &decoder.neural,
                SpeechProviderKind::CodecDecoder,
            ),
            (
                parts.quantizer.geometry() == decoder.quantizer,
                SpeechProviderKind::CodecQuantizer,
            ),
        ] {
            if !matches {
                return Err(AssemblyError::ProviderGeometry(kind));
            }
        }
        // Reassembly validates every public TTS subprovider, including conditioning
        // prompt geometry, against the same policy before spectral allocation.
        let generator = TtsGenerator::from_parts(
            &config.tts,
            256,
            prompt.frames(),
            parts.generator.into_parts(),
        )?;
        let input = PerceptionRnnt::from_parts(parts.perception, parts.rnnt)?;
        let fusion =
            ChannelFusion::from_config(&config.channels).map_err(ConstructionError::from)?;
        self.validate_spectral_capabilities::<B>()?;
        Ok(LearnedSpeechComponents {
            input,
            function: parts.function,
            generator,
            encoder: AudioEncoder::from_parts(encoder, parts.encoder, context)?,
            decoder: AudioDecoder::from_parts(decoder, parts.quantizer, parts.decoder, context)?,
            fusion,
            prompt,
        })
    }
    /// Constructs and offers each exact unit to a caller's binding mechanism once.
    /// Binding must establish safe materialization completion before returning.
    /// An error stops construction; completed units are dropped without publication.
    /// This does not choose a residency schedule or initialize inference caches.
    pub fn assemble_with<B: NeuralBackend, E: std::error::Error + Send + Sync + 'static>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
        mut bind: impl FnMut(SpeechUnitId, &mut SpeechUnit<B>) -> Result<(), E>,
    ) -> Result<LearnedSpeechComponents<B>, AssemblyError> {
        self.assemble_using(context, |id, unit| {
            bind(id, unit).map_err(|source| AssemblyError::Binding {
                unit: id,
                source: BackendFailure::from_error(source),
            })
        })
    }

    pub(super) fn assemble_using<B: NeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
        mut bind: impl FnMut(SpeechUnitId, &mut SpeechUnit<B>) -> Result<(), AssemblyError>,
    ) -> Result<LearnedSpeechComponents<B>, AssemblyError> {
        // Spectral constants are assembled after learned units; reject a missing
        // mechanism now, before any caller starts expensive parameter binding.
        self.validate_spectral_capabilities::<B>()?;
        let mut units = BTreeMap::new();
        for &id in self.units() {
            let mut unit = self.build_unit::<B>(id, context)?;
            bind(id, &mut unit)?;
            units.insert(id, unit);
        }
        let config = self.config();
        // The map is private and filled solely from the validated construction
        // order. Consuming variants preserves the exact bound parameter objects.
        macro_rules! take {
            ($id:expr, $variant:ident) => {{
                let id = $id;
                match units.remove(&id) {
                    Some(SpeechUnit::$variant(value)) => value,
                    _ => return Err(AssemblyError::Unit(id)),
                }
            }};
        }
        use SpeechUnitId as Id;
        let perception = Perception::from_parts(
            PerceptionDimensions::from_config(&config.perception)
                .map_err(ConstructionError::from)?,
            take!(Id::PerceptionSubsampling, PerceptionSubsampling),
            (0..config.perception.encoder.n_layers)
                .map(|i| Ok(take!(Id::PerceptionBlock(i), PerceptionBlock)))
                .collect::<Result<_, AssemblyError>>()?,
            take!(Id::PerceptionProjection, PerceptionProjection),
        )
        .map_err(ConstructionError::from)?;
        let rnnt = take!(Id::Rnnt, Rnnt);
        let function = take!(Id::FunctionHead, FunctionHead);
        let character_transformer = TtsTransformer::from_parts(
            TransformerConfig::new(&config.tts.character_encoder)
                .map_err(ConstructionError::from)?,
            (0..config.tts.character_encoder.num_hidden_layers)
                .map(|i| Ok(take!(Id::CharacterBlock(i), CharacterBlock)))
                .collect::<Result<_, AssemblyError>>()?,
            take!(Id::CharacterNorm, CharacterNorm),
        )
        .map_err(TtsError::from)?;
        let transformer = TtsTransformer::from_parts(
            TransformerConfig::new(&config.tts.backbone).map_err(ConstructionError::from)?,
            (0..config.tts.backbone.num_hidden_layers)
                .map(|i| Ok(take!(Id::TtsBlock(i), TtsBlock)))
                .collect::<Result<_, AssemblyError>>()?,
            take!(Id::TtsNorm, TtsNorm),
        )
        .map_err(TtsError::from)?;
        let mog = MogHead::from_parts(
            MogDimensions::from_config(&config.tts)?,
            (0..config.tts.mog.num_layers)
                .map(|i| Ok(take!(Id::MogBlock(i), MogBlock)))
                .collect::<Result<_, AssemblyError>>()?,
            take!(Id::MogPredictionHead, MogPredictionHead),
            take!(Id::MogComponentBank, MogComponentBank),
        )?;
        let prompt = AriaPromptPlan::from_config(config, usize::MAX)?;
        let generator = TtsGenerator::from_parts(
            &config.tts,
            256,
            prompt.frames(),
            TtsGeneratorParts::new(
                ResidentConditioning::from_parts(
                    ConditioningDimensions::from_config(&config.tts, 256, prompt.frames())?,
                    take!(Id::CharacterEmbedding, CharacterEmbedding),
                    take!(Id::TtsIngress, TtsIngress),
                    take!(Id::TtsFusion, TtsFusion),
                )?,
                character_transformer,
                transformer,
                RvqRefinement::from_parts(
                    RvqDimensions::from_config(&config.tts)?,
                    take!(Id::TtsRvqBank, TtsRvqBank),
                    take!(Id::TtsRvqProjection, TtsRvqProjection),
                )?,
                mog,
            ),
        )?;
        let encoder_config = waveform_encoder_config(&config.codec)?;
        let decoder_config = waveform_decoder_config(&config.codec)?;
        let encoder_neural = NeuralEncoder::<B>::from_units(
            encoder_config.neural.clone(),
            (0..encoder_config.neural.unit_specs()?.len())
                .map(|i| Ok(take!(Id::CodecEncoder(i), CodecEncoder)))
                .collect::<Result<_, AssemblyError>>()?,
        )?;
        let decoder_neural = NeuralDecoder::<B>::from_units(
            decoder_config.neural.clone(),
            (0..decoder_config.neural.unit_specs()?.len())
                .map(|i| Ok(take!(Id::CodecDecoder(i), CodecDecoder)))
                .collect::<Result<_, AssemblyError>>()?,
        )?;
        let quantizer = ResidualQuantizer::<B>::from_codebooks(
            decoder_config.quantizer,
            (0..config.codec.num_quantizers)
                .map(|i| Ok(take!(Id::CodecCodebook(i), CodecCodebook)))
                .collect::<Result<_, AssemblyError>>()?,
        )?;
        debug_assert!(units.is_empty());
        self.assemble_providers(
            SpeechProviderParts {
                perception,
                rnnt,
                function,
                generator,
                encoder: encoder_neural,
                decoder: decoder_neural,
                quantizer,
            },
            context,
        )
    }
}
