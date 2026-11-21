//! Cold preparation and consuming assembly of independently resident speech providers.
//!
//! All retained component recipes are admitted before the first binding callback.
//! The architecture owns composition; native visitors supply parameter mechanisms.

use eredu_core::SubmissionBackend;

use std::collections::BTreeMap;

use eredu_codec::{
    ear_decoder::DecoderUnit,
    ear_decoder_layered::BoundEarDecoder,
    ear_decoder_preparation::{DecoderPreparationError, PreparedDecoderBindings},
    ear_encoder::EncoderUnit,
    ear_encoder_layered::BoundEarEncoder,
    ear_encoder_preparation::{EncoderPreparationError, PreparedEncoderBindings},
    prvq::Codebook,
    prvq_layered::BoundQuantizer,
    prvq_preparation::{PreparedQuantizerBindings, QuantizerPreparationError},
};
use eredu_core::BackendFailure;
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{LayeredTensorCohort, ParameterUnitMechanisms};

use super::{
    artifact::{ParameterAdmission, RetainedVoiceChatArtifacts},
    assembly::{AssemblyError, LearnedSpeechComponents, SpeechProviderParts},
    channels::FunctionHead,
    checkpoint::Component,
    codec_preparation::{
        prepare_neural_decoder, prepare_neural_encoder, prepare_quantizer, CodecPreparationError,
    },
    construction::{ConstructionError, SpeechConstructionPlan},
    function_head_layered::BoundFunctionHead,
    function_head_preparation::{FunctionHeadPreparationError, PreparedFunctionHeadBindings},
    parameters::SpeechParameterError,
    perception_layered::{BoundPerception, PerceptionUnit},
    perception_preparation::{PerceptionPreparationError, PreparedPerceptionBindings},
    provider_residency::{
        SpeechProvider, SpeechProviderResources, SpeechResidencyError, SpeechResidencyPolicy,
        SpeechResidencyReport,
    },
    rnnt_layered::RnntUnit,
    rnnt_preparation::{PreparedRnntBindings, RnntPreparationError},
    rnnt_projection_layered::BoundRnnt,
    tts_conditioning_layered::{BoundConditioning, ConditioningUnit},
    tts_conditioning_preparation::{ConditioningPreparationError, PreparedConditioningBindings},
    tts_generator::{TtsGenerator, TtsGeneratorParts},
    tts_mog::TtsError,
    tts_mog_layered::{BoundMog, MogUnit},
    tts_mog_preparation::{MogPreparationError, PreparedMogBindings},
    tts_rvq_layered::{BoundRvq, RvqUnit},
    tts_rvq_preparation::{PreparedRvqBindings, RvqPreparationError},
    tts_transformer_layered::{BoundTransformer, TransformerUnit},
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
    Encoder(#[from] CodecPreparationError<EncoderPreparationError>),
    /// Codec decoder recipes are invalid.
    #[error(transparent)]
    Decoder(#[from] CodecPreparationError<DecoderPreparationError>),
    /// Codec quantizer recipes are invalid.
    #[error(transparent)]
    Quantizer(#[from] CodecPreparationError<QuantizerPreparationError>),
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

/// Function head bound by the shared unit mechanisms.
pub type SpeechFunctionHead<B, H, C> =
    BoundFunctionHead<B, <H as ParameterUnitMechanisms<B>>::UnitPolicy<FunctionHead<B>>, C>;

/// RNNT bound by the shared unit mechanisms.
pub type SpeechRnnt<B, H, C> =
    BoundRnnt<B, <H as ParameterUnitMechanisms<B>>::UnitPolicy<RnntUnit<B>>, C>;

/// Mog bound by the shared unit mechanisms.
pub type SpeechMog<B, H, C> =
    BoundMog<B, <H as ParameterUnitMechanisms<B>>::UnitPolicy<MogUnit<B>>, C>;

/// Rvq bound by the shared unit mechanisms.
pub type SpeechRvq<B, H, C> =
    BoundRvq<B, <H as ParameterUnitMechanisms<B>>::UnitPolicy<RvqUnit<B>>, C>;

/// Conditioning bound by the shared unit mechanisms.
pub type SpeechConditioning<B, H, C> =
    BoundConditioning<B, <H as ParameterUnitMechanisms<B>>::UnitPolicy<ConditioningUnit<B>>, C>;

/// Quantizer bound by the shared unit mechanisms.
pub type SpeechQuantizer<B, H, C> = BoundQuantizer<
    B,
    <H as ParameterUnitMechanisms<B>>::UnitPolicy<Codebook<<B as NeuralBackend>::Tensor>>,
    C,
>;

/// Encoder bound by the shared unit mechanisms.
pub type SpeechEncoder<B, H, C> = BoundEarEncoder<
    B,
    <H as ParameterUnitMechanisms<B>>::UnitPolicy<EncoderUnit<<B as NeuralBackend>::Tensor>>,
    C,
>;

/// Transformer bound by the shared unit mechanisms.
pub type SpeechTransformer<B, H, C> =
    BoundTransformer<B, <H as ParameterUnitMechanisms<B>>::UnitPolicy<TransformerUnit<B>>, C>;

/// Perception bound by the shared parameter unit mechanisms.
pub type SpeechPerception<B, H, C> =
    BoundPerception<B, <H as ParameterUnitMechanisms<B>>::UnitPolicy<PerceptionUnit<B>>, C>;

/// EAR decoder bound by the shared parameter unit mechanisms.
pub type SpeechDecoder<B, H, C> = BoundEarDecoder<
    B,
    <H as ParameterUnitMechanisms<B>>::UnitPolicy<DecoderUnit<<B as NeuralBackend>::Tensor>>,
    C,
>;

/// Bound provider types assembled by the architecture driver.
pub type BoundSpeechProviders<B, H, C> = LearnedSpeechComponents<
    B,
    SpeechPerception<B, H, C>,
    SpeechTransformer<B, H, C>,
    SpeechMog<B, H, C>,
    SpeechRvq<B, H, C>,
    SpeechConditioning<B, H, C>,
    SpeechDecoder<B, H, C>,
    SpeechQuantizer<B, H, C>,
    SpeechEncoder<B, H, C>,
    SpeechFunctionHead<B, H, C>,
    SpeechRnnt<B, H, C>,
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
    /// Causal transformer scope shared by state admission and materialization.
    pub(crate) fn transformer_declaration(
        &self,
    ) -> &super::tts_transformer_specs::TransformerDeclaration {
        self.transformer.declaration()
    }

    /// Selects eligible packed matrices while preserving all other speech policies.
    pub fn with_quantization(
        mut self,
        policy: super::quantization::SpeechQuantizationPolicy,
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, ProviderConstructionError> {
        use eredu_checkpoint::recipe::RecipeDtype;
        let mut selected = parameters.clone();
        if let Some(quantization) = policy.function_head {
            let eligible = std::collections::BTreeSet::from([self
                .function
                .declaration()
                .spec()
                .projection
                .weight
                .id
                .as_str()
                .to_owned()]);
            let (source, parameters) = super::quantization::project_parameters(
                self.function.source().clone(),
                &selected,
                quantization,
                &eligible,
                RecipeDtype::F32,
            )
            .map_err(FunctionHeadPreparationError::from)?;
            self.function = PreparedFunctionHeadBindings::from_parameters(
                self.function.declaration().clone(),
                source,
                &parameters,
            )?;
            selected = parameters;
        }
        if let Some(quantization) = policy.tts {
            let config = self.transformer.config().clone();
            let declaration = super::precision::SpeechPrecisionDeclaration::from_groups(
                self.transformer.definition().parameters().groups(),
            );
            let eligible = self
                .transformer
                .unit_bindings()
                .iter()
                .flatten()
                .filter(|binding| declaration.narrows(config.precision(), binding.name()))
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
                self.transformer
                    .declaration()
                    .clone()
                    .with_precision(Default::default())
                    .map_err(TtsTransformerPreparationError::from)?,
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
    pub fn materialize<B, H, C>(
        self,
        policy: &SpeechResidencyPolicy,
        mechanisms: &H,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(BoundSpeechProviders<B, H, C>, SpeechResidencyReport), ProviderConstructionError>
    where
        B: DistributedNeuralBackend
            + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
            + 'static,
        H: ParameterUnitMechanisms<B>,
        C: LayeredTensorCohort<B>,
    {
        let report = self.validate_residency(policy)?;
        self.plan.validate_spectral_capabilities::<B>()?;
        let dimensions = self.conditioning.dimensions();
        let (perception, _) = self
            .perception
            .bind::<B, _, C>(
                policy.provider(SpeechProvider::Perception),
                context,
                mechanisms,
                None,
            )
            .map_err(ProviderConstructionError::Backend)?;
        let (rnnt, _) = self
            .rnnt
            .bind::<B, _, C>(
                policy.provider(SpeechProvider::Rnnt),
                context,
                mechanisms,
                None,
            )
            .map_err(ProviderConstructionError::Backend)?;
        let (function, _) = self
            .function
            .bind::<B, _, C>(
                policy.provider(SpeechProvider::FunctionHead),
                context,
                mechanisms,
                None,
            )
            .map_err(ProviderConstructionError::Backend)?;
        let (conditioning, _) = self
            .conditioning
            .bind::<B, _, C>(
                policy.provider(SpeechProvider::Conditioning),
                context,
                mechanisms,
                None,
            )
            .map_err(ProviderConstructionError::Backend)?;
        let (character, _) = self
            .character
            .bind::<B, _, C>(
                policy.provider(SpeechProvider::CharacterEncoder),
                context,
                mechanisms,
                None,
            )
            .map_err(ProviderConstructionError::Backend)?;
        let (transformer, _) = self
            .transformer
            .bind::<B, _, C>(
                policy.provider(SpeechProvider::TtsTransformer),
                context,
                mechanisms,
                None,
            )
            .map_err(ProviderConstructionError::Backend)?;
        let (mog, _) = self
            .mog
            .bind::<B, _, C>(
                policy.provider(SpeechProvider::Mog),
                context,
                mechanisms,
                None,
            )
            .map_err(ProviderConstructionError::Backend)?;
        let (rvq, _) = self
            .rvq
            .bind::<B, _, C>(
                policy.provider(SpeechProvider::Rvq),
                context,
                mechanisms,
                None,
            )
            .map_err(ProviderConstructionError::Backend)?;
        let generator = TtsGenerator::from_parts(
            &self.plan.config().tts,
            dimensions.character.characters,
            dimensions.speaker_frames,
            TtsGeneratorParts::new(conditioning, character, transformer, rvq, mog),
        )?;
        let (encoder, _) = self
            .encoder
            .bind::<B, _, C>(
                policy.provider(SpeechProvider::CodecEncoder),
                context,
                mechanisms,
                None,
            )
            .map_err(ProviderConstructionError::Backend)?;
        let (decoder, _) = self
            .decoder
            .bind::<B, _, C>(
                policy.provider(SpeechProvider::CodecDecoder),
                context,
                mechanisms,
                None,
            )
            .map_err(ProviderConstructionError::Backend)?;
        let (quantizer, _) = self
            .quantizer
            .bind::<B, _, C>(
                policy.provider(SpeechProvider::CodecQuantizer),
                context,
                mechanisms,
                None,
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
