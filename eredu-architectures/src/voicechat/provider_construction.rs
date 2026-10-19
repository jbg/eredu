//! Cold preparation and consuming assembly of independently resident speech providers.
//!
//! All retained component recipes are admitted before the first binding callback.
//! The architecture owns composition; native visitors supply parameter mechanisms.

use std::collections::BTreeMap;

use eredu_codec::{
    ear_decoder_layered::DecoderExecution, ear_decoder_preparation::PreparedDecoderBindings,
    ear_encoder_layered::EncoderExecution, ear_encoder_preparation::PreparedEncoderBindings,
    prvq_layered::QuantizerExecution, prvq_preparation::PreparedQuantizerBindings,
};
use eredu_core::BackendFailure;
use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::LayerWeightResidency;

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    assembly::{AssemblyError, LearnedSpeechComponents, SpeechProviderParts},
    checkpoint::Component,
    codec_preparation::{
        prepare_neural_decoder, prepare_neural_encoder, prepare_quantizer,
        CodecDecoderPreparationError, CodecEncoderPreparationError, CodecQuantizerPreparationError,
    },
    construction::{ConstructionError, SpeechConstructionPlan},
    function_head_layered::FunctionHeadExecution,
    function_head_preparation::{FunctionHeadPreparationError, PreparedFunctionHeadBindings},
    parameters::SpeechParameterError,
    perception_layered::PerceptionExecution,
    perception_preparation::{PerceptionPreparationError, PreparedPerceptionBindings},
    provider_residency::{
        SpeechProvider, SpeechProviderResources, SpeechResidencyError, SpeechResidencyPolicy,
        SpeechResidencyReport,
    },
    rnnt::RnntExecution,
    rnnt_preparation::{PreparedRnntBindings, RnntPreparationError},
    tts_conditioning_layered::ConditioningExecution,
    tts_conditioning_preparation::{ConditioningPreparationError, PreparedConditioningBindings},
    tts_generator::{TtsGenerator, TtsGeneratorParts},
    tts_mog::TtsError,
    tts_mog_layered::MogExecution,
    tts_mog_preparation::{MogPreparationError, PreparedMogBindings},
    tts_rvq_layered::RvqExecution,
    tts_rvq_preparation::{PreparedRvqBindings, RvqPreparationError},
    tts_transformer_layered::TransformerExecution,
    tts_transformer_preparation::{
        PreparedTtsTransformerBindings, TtsTransformerKind, TtsTransformerPreparationError,
    },
};

/// Cold source/geometry failure or an unpublished provider binding failure.
#[derive(Debug, thiserror::Error)]
pub enum ProviderConstructionError {
    /// Aggregate speech parameter allowances are invalid or exceed their explicit bounds.
    #[error(transparent)]
    Residency(#[from] SpeechResidencyError),
    /// A required retained source role is absent.
    #[error(transparent)]
    Parameters(#[from] SpeechParameterError),
    /// Shared speech construction policy is invalid.
    #[error(transparent)]
    Construction(#[from] ConstructionError),
    /// Perception recipes are invalid.
    #[error(transparent)]
    Perception(#[from] PerceptionPreparationError),
    /// Function readout recipes are invalid.
    #[error(transparent)]
    Function(#[from] FunctionHeadPreparationError),
    /// Transformer recipes are invalid.
    #[error(transparent)]
    Transformer(#[from] TtsTransformerPreparationError),
    /// Conditioning recipes are invalid.
    #[error(transparent)]
    Conditioning(#[from] ConditioningPreparationError),
    /// Mixture recipes are invalid.
    #[error(transparent)]
    Mog(#[from] MogPreparationError),
    /// Refinement recipes are invalid.
    #[error(transparent)]
    Rvq(#[from] RvqPreparationError),
    /// Codec encoder recipes are invalid.
    #[error(transparent)]
    Encoder(#[from] CodecEncoderPreparationError),
    /// Codec decoder recipes are invalid.
    #[error(transparent)]
    Decoder(#[from] CodecDecoderPreparationError),
    /// Codec quantizer recipes are invalid.
    #[error(transparent)]
    Quantizer(#[from] CodecQuantizerPreparationError),
    /// RNNT recipes or geometry are invalid.
    #[error(transparent)]
    Rnnt(#[from] RnntPreparationError),
    /// The selected binding mechanism failed without publishing an aggregate.
    #[error("VoiceChat speech provider binding failed: {0}")]
    Backend(#[source] BackendFailure),
    /// Connected provider geometry disagrees.
    #[error(transparent)]
    Assembly(#[from] AssemblyError),
    /// Connected generator geometry disagrees.
    #[error(transparent)]
    Speech(#[from] TtsError),
}

/// Native binding mechanisms supplied to the architecture-owned construction order.
///
/// Every method consumes a cold exact source plan. Successful callbacks establish
/// safe materialization completion; errors preserve native failures through the
/// neutral wrapper. No callback resolves artifacts or chooses family equations.
pub trait SpeechProviderBinder<B: NeuralBackend> {
    /// Recurrent transcription realization.
    type Rnnt: RnntExecution<B>;
    /// Streaming perception realization.
    type Perception: PerceptionExecution<B>;
    /// Shared provider type for the causal and character transformers.
    type Transformer: TransformerExecution<B>;
    /// Gaussian-mixture realization.
    type Mog: MogExecution<B>;
    /// Iterative residual quantization realization.
    type Rvq: RvqExecution<B>;
    /// Character, ingress and fusion realization.
    type Conditioning: ConditioningExecution<B>;
    /// Finite startup encoder realization.
    type Encoder: EncoderExecution<B>;
    /// Streaming decoder realization.
    type Decoder: DecoderExecution<B>;
    /// Single shared codec quantizer realization.
    type Quantizer: QuantizerExecution<B>;
    /// Function-channel readout realization.
    type Function: FunctionHeadExecution<B>;

    /// Binds the shared perception invocation.
    fn perception(
        &mut self,
        prepared: PreparedPerceptionBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Perception, BackendFailure>;
    /// Binds the function-channel readout.
    fn function(
        &mut self,
        prepared: PreparedFunctionHeadBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Function, BackendFailure>;
    /// Binds one explicitly selected transformer role.
    fn transformer(
        &mut self,
        kind: TtsTransformerKind,
        prepared: PreparedTtsTransformerBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Transformer, BackendFailure>;
    /// Binds the iterative mixture stack.
    fn mog(
        &mut self,
        prepared: PreparedMogBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Mog, BackendFailure>;
    /// Binds the refinement bank and projection.
    fn rvq(
        &mut self,
        prepared: PreparedRvqBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Rvq, BackendFailure>;
    /// Binds character/ingress/fusion parameters.
    fn conditioning(
        &mut self,
        prepared: PreparedConditioningBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Conditioning, BackendFailure>;
    /// Binds the finite carrier encoder.
    fn encoder(
        &mut self,
        prepared: PreparedEncoderBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Encoder, BackendFailure>;
    /// Binds the streaming waveform decoder.
    fn decoder(
        &mut self,
        prepared: PreparedDecoderBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Decoder, BackendFailure>;
    /// Binds the one shared codec quantizer.
    fn quantizer(
        &mut self,
        prepared: PreparedQuantizerBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Quantizer, BackendFailure>;
    /// Binds independently leased recurrent transcription parameters.
    fn rnnt(
        &mut self,
        prepared: PreparedRnntBindings,
        residency: LayerWeightResidency,
    ) -> Result<Self::Rnnt, BackendFailure>;
}

/// Bound provider types assembled by the architecture driver.
pub type BoundSpeechProviders<B, V> = LearnedSpeechComponents<
    B,
    <V as SpeechProviderBinder<B>>::Perception,
    <V as SpeechProviderBinder<B>>::Transformer,
    <V as SpeechProviderBinder<B>>::Mog,
    <V as SpeechProviderBinder<B>>::Rvq,
    <V as SpeechProviderBinder<B>>::Conditioning,
    <V as SpeechProviderBinder<B>>::Decoder,
    <V as SpeechProviderBinder<B>>::Quantizer,
    <V as SpeechProviderBinder<B>>::Encoder,
    <V as SpeechProviderBinder<B>>::Function,
    <V as SpeechProviderBinder<B>>::Rnnt,
>;

/// Complete cold authority for speech providers; the Nemotron target remains separate.
///
/// This owns only exact retained source views and geometry. It performs no payload
/// reads, creates no native tensor, and allocates no conversation state.
pub struct PreparedSpeechProviders {
    plan: SpeechConstructionPlan,
    perception: PreparedPerceptionBindings,
    function: PreparedFunctionHeadBindings,
    rnnt: PreparedRnntBindings,
    character: PreparedTtsTransformerBindings,
    transformer: PreparedTtsTransformerBindings,
    mog: PreparedMogBindings,
    rvq: PreparedRvqBindings,
    conditioning: PreparedConditioningBindings,
    encoder: PreparedEncoderBindings,
    decoder: PreparedDecoderBindings,
    quantizer: PreparedQuantizerBindings,
    resources: BTreeMap<SpeechProvider, SpeechProviderResources>,
}

impl PreparedSpeechProviders {
    /// Selects eligible packed matrices while preserving all other speech policies.
    pub fn with_quantization(
        mut self,
        policy: super::quantization::SpeechQuantizationPolicy,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, ProviderConstructionError> {
        use eredu_checkpoint::recipe::RecipeDtype;
        let mut selected = parameters.clone();
        if let Some(quantization) = policy.function_head {
            let eligible =
                std::collections::BTreeSet::from(["stt_model.function_head.weight".to_owned()]);
            let (source, parameters) = super::quantization::project_parameters(
                self.function.source().clone(),
                &selected,
                quantization,
                &eligible,
                RecipeDtype::F32,
            )
            .map_err(FunctionHeadPreparationError::from)?;
            self.function = PreparedFunctionHeadBindings::from_parameters(
                self.function.dimensions(),
                source,
                &parameters,
            )?;
            selected = parameters;
        }
        if let Some(quantization) = policy.tts {
            let config = self.transformer.config().clone();
            let eligible = self
                .transformer
                .unit_bindings()
                .iter()
                .flatten()
                .filter(|binding| {
                    let name = binding.name();
                    config
                        .precision()
                        .narrows(name, selected[name].output.shape.len())
                })
                .map(|binding| binding.name().to_owned())
                .collect();
            let (source, parameters) = super::quantization::project_parameters(
                self.transformer.source().clone(),
                &selected,
                quantization,
                &eligible,
                RecipeDtype::F32,
            )
            .map_err(TtsTransformerPreparationError::from)?;
            let precision = super::precision::SpeechPrecision {
                weights: config.precision().weights,
                compute: super::precision::SpeechFloat::F32,
            };
            self.transformer = PreparedTtsTransformerBindings::from_parameters(
                config.with_precision(Default::default()),
                source,
                &parameters,
            )?
            .with_precision(precision)?;
            selected = parameters;
        }
        self.prepare_resources(&selected)?;
        Ok(self)
    }

    /// Admits every selected speech role before allowing a native binding callback.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, ProviderConstructionError> {
        for component in [
            Component::Perception,
            Component::FunctionHead,
            Component::Rnnt,
            Component::Tts,
            Component::CodecEncoder,
            Component::CodecDecoder,
            Component::CodecQuantizer,
        ] {
            if artifacts.component_source(component).is_none() {
                return Err(SpeechParameterError::MissingComponent(component).into());
            }
        }
        let mut prepared = Self {
            plan: SpeechConstructionPlan::from_config(artifacts.plan().config())?,
            perception: PreparedPerceptionBindings::from_retained(artifacts)?,
            function: PreparedFunctionHeadBindings::from_retained(artifacts)?,
            rnnt: PreparedRnntBindings::from_retained(artifacts)?,
            character: PreparedTtsTransformerBindings::from_retained(
                artifacts,
                TtsTransformerKind::CharacterEncoder,
            )?,
            transformer: PreparedTtsTransformerBindings::from_retained(
                artifacts,
                TtsTransformerKind::Causal,
            )?,
            mog: PreparedMogBindings::from_retained(artifacts)?,
            rvq: PreparedRvqBindings::from_retained(artifacts)?,
            conditioning: PreparedConditioningBindings::from_retained(artifacts)?,
            encoder: prepare_neural_encoder(artifacts)?,
            decoder: prepare_neural_decoder(artifacts)?,
            quantizer: prepare_quantizer(artifacts)?,
            resources: BTreeMap::new(),
        };
        prepared.prepare_resources(artifacts.plan().parameters())?;
        Ok(prepared)
    }

    /// Reprojects narrow weights before aggregate resource admission or native binding.
    pub fn with_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, ProviderConstructionError> {
        self.perception = self.perception.with_precision(precision)?;
        self.character = self.character.with_precision(precision)?;
        self.transformer = self.transformer.with_precision(precision)?;
        self.mog = self.mog.with_precision(precision)?;
        self.prepare_resources(parameters)?;
        Ok(self)
    }

    fn prepare_resources(
        &mut self,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<(), ProviderConstructionError> {
        macro_rules! resources {
            ($role:ident, $field:ident) => {
                self.resources.insert(
                    SpeechProvider::$role,
                    SpeechProviderResources::from_bindings(
                        self.$field.source(),
                        self.$field.unit_bindings(),
                        parameters,
                    )?,
                );
            };
        }
        resources!(Perception, perception);
        resources!(FunctionHead, function);
        resources!(CharacterEncoder, character);
        resources!(TtsTransformer, transformer);
        resources!(Mog, mog);
        resources!(Rvq, rvq);
        resources!(Conditioning, conditioning);
        resources!(CodecEncoder, encoder);
        resources!(CodecDecoder, decoder);
        resources!(CodecQuantizer, quantizer);
        resources!(Rnnt, rnnt);
        Ok(())
    }

    /// Exact converted parameter and materialization requirements by independent provider.
    pub fn resources(&self) -> &BTreeMap<SpeechProvider, SpeechProviderResources> {
        &self.resources
    }

    /// Admits parameter pools and bounded immutable speech execution caches.
    pub fn validate_residency(
        &self,
        policy: &SpeechResidencyPolicy,
    ) -> Result<SpeechResidencyReport, super::provider_residency::SpeechResidencyError> {
        use super::provider_residency::SpeechResidencyError;
        let norms = 31_u64
            .checked_mul(self.rvq.dimensions().cardinality as u64)
            .and_then(|n| n.checked_mul(4))
            .ok_or(SpeechResidencyError::ArithmeticOverflow)?;
        let characters = if policy
            .provider(SpeechProvider::Conditioning)
            .is_fully_resident()
            && policy
                .provider(SpeechProvider::CharacterEncoder)
                .is_fully_resident()
        {
            (super::tts_generator::CHARACTER_CACHE_ENTRIES as u64)
                .checked_mul(self.rvq.dimensions().hidden as u64)
                .and_then(|n| n.checked_mul(4))
                .ok_or(SpeechResidencyError::ArithmeticOverflow)?
        } else {
            0
        };
        let perception = self.perception.dimensions();
        // One single-frame position table and its projection per layer. Bound
        // rebuilt units by all layers as well; parameter pools do not pay for
        // derived tensors, and local heads never exceed this replicated bound.
        let positions = (perception.block.left_context as u64)
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(perception.block.width as u64))
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_mul(perception.layers as u64))
            .ok_or(SpeechResidencyError::ArithmeticOverflow)?;
        policy.validate_with_execution_cache(
            &self.resources,
            norms
                .checked_add(positions)
                .and_then(|n| n.checked_add(characters))
                .ok_or(SpeechResidencyError::ArithmeticOverflow)?,
        )
    }

    /// Binds selected providers and consumes them into the existing coupled executor.
    /// No backbone or session is constructed here.
    pub fn materialize<B: NeuralBackend, V: SpeechProviderBinder<B>>(
        self,
        policy: &SpeechResidencyPolicy,
        visitor: &mut V,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(BoundSpeechProviders<B, V>, SpeechResidencyReport), ProviderConstructionError>
    {
        let report = self.validate_residency(policy)?;
        self.plan.validate_spectral_capabilities::<B>()?;
        let dimensions = self.conditioning.dimensions();
        let perception = visitor
            .perception(self.perception, policy.provider(SpeechProvider::Perception))
            .map_err(ProviderConstructionError::Backend)?;
        let rnnt = visitor
            .rnnt(self.rnnt, policy.provider(SpeechProvider::Rnnt))
            .map_err(ProviderConstructionError::Backend)?;
        let function = visitor
            .function(self.function, policy.provider(SpeechProvider::FunctionHead))
            .map_err(ProviderConstructionError::Backend)?;
        let conditioning = visitor
            .conditioning(
                self.conditioning,
                policy.provider(SpeechProvider::Conditioning),
            )
            .map_err(ProviderConstructionError::Backend)?;
        let character = visitor
            .transformer(
                TtsTransformerKind::CharacterEncoder,
                self.character,
                policy.provider(SpeechProvider::CharacterEncoder),
            )
            .map_err(ProviderConstructionError::Backend)?;
        let transformer = visitor
            .transformer(
                TtsTransformerKind::Causal,
                self.transformer,
                policy.provider(SpeechProvider::TtsTransformer),
            )
            .map_err(ProviderConstructionError::Backend)?;
        let mog = visitor
            .mog(self.mog, policy.provider(SpeechProvider::Mog))
            .map_err(ProviderConstructionError::Backend)?;
        let rvq = visitor
            .rvq(self.rvq, policy.provider(SpeechProvider::Rvq))
            .map_err(ProviderConstructionError::Backend)?;
        let generator = TtsGenerator::from_parts(
            &self.plan.config().tts,
            dimensions.character.characters,
            dimensions.speaker_frames,
            TtsGeneratorParts::new(conditioning, character, transformer, rvq, mog),
        )?;
        let encoder = visitor
            .encoder(self.encoder, policy.provider(SpeechProvider::CodecEncoder))
            .map_err(ProviderConstructionError::Backend)?;
        let decoder = visitor
            .decoder(self.decoder, policy.provider(SpeechProvider::CodecDecoder))
            .map_err(ProviderConstructionError::Backend)?;
        let quantizer = visitor
            .quantizer(
                self.quantizer,
                policy.provider(SpeechProvider::CodecQuantizer),
            )
            .map_err(ProviderConstructionError::Backend)?;
        Ok((
            self.plan.assemble_providers(
                SpeechProviderParts {
                    perception,
                    rnnt,
                    function,
                    generator,
                    encoder,
                    decoder,
                    quantizer,
                },
                context,
            )?,
            report,
        ))
    }
}
