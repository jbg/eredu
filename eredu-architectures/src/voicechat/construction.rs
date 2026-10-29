//! Architecture-owned construction of independently bindable speech units.
//!
//! Configuration selects equations and parameter identities. Retained artifacts
//! supply exact recipes; a backend only constructs ordinary neural mechanisms.
//! A unit has no session state, residency schedule, or completion authority.

use crate::decoder::Config;

use eredu_codec::{
    ear::{AudioDecoderConfig, AudioEncoderConfig},
    ear_decoder::DecoderUnit,
    ear_encoder::EncoderUnit,
    prvq::Codebook,
};
use eredu_nn::{LinearFormatSpec, NeuralBackend, Parameterized, Tensor};

use super::{
    artifact::RetainedVoiceChatArtifacts,
    channels::{ChannelError, FunctionHead},
    checkpoint::Component,
    codec::{waveform_decoder_config, waveform_encoder_config},
    config::{ConfigError, VoiceChatConfig},
    conformer::{ConformerBlock, ConformerError},
    parameters::{PreparedSpeechParameterUnit, SpeechParameterError},
    perception::{PerceptionDimensions, PerceptionError},
    rnnt::{Rnnt, RnntDimensions, RnntError},
    subsampling::{CausalSubsampling, SubsamplingError},
    tts_conditioning::{CharacterDimensions, CharacterEmbedding, GatedFusion},
    tts_ingress::TtsIngress,
    tts_mog::{MogBlock, MogComponentBank, MogDimensions, MogPredictionHead, TtsError},
    tts_prompt::AriaPromptPlan,
    tts_rvq::{RvqBank, RvqDimensions, RvqProjection},
    tts_transformer::TransformerConfig,
};

/// One independently constructed and bound learned unit in the speech graph.
/// Indices follow the admitted architecture's order, never physical file order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum SpeechUnitId {
    PerceptionSubsampling,
    PerceptionBlock(usize),
    PerceptionProjection,
    Rnnt,
    FunctionHead,
    CharacterEmbedding,
    CharacterBlock(usize),
    CharacterNorm,
    TtsIngress,
    TtsFusion,
    TtsBlock(usize),
    TtsNorm,
    TtsRvqBank,
    TtsRvqProjection,
    MogBlock(usize),
    MogPredictionHead,
    MogComponentBank,
    CodecEncoder(usize),
    CodecDecoder(usize),
    CodecCodebook(usize),
}

impl SpeechUnitId {
    /// Source role whose canonical recipes supply this unit.
    pub const fn component(self) -> Component {
        match self {
            Self::PerceptionSubsampling | Self::PerceptionBlock(_) | Self::PerceptionProjection => {
                Component::Perception
            }
            Self::Rnnt => Component::Rnnt,
            Self::FunctionHead => Component::FunctionHead,
            Self::CodecEncoder(_) => Component::CodecEncoder,
            Self::CodecDecoder(_) => Component::CodecDecoder,
            Self::CodecCodebook(_) => Component::CodecQuantizer,
            _ => Component::Tts,
        }
    }
}

/// Concrete portable neural mechanisms selected by a speech unit declaration.
/// Large stacks use individual blocks; codec codebook depths and the stacked TTS
/// feedback bank have distinct canonical parameter ownership.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
#[allow(missing_docs)]
pub enum SpeechUnit<B: NeuralBackend> {
    PerceptionSubsampling(CausalSubsampling<B>),
    PerceptionBlock(ConformerBlock<B>),
    PerceptionProjection(B::Linear),
    Rnnt(Rnnt<B>),
    FunctionHead(FunctionHead<B>),
    CharacterEmbedding(CharacterEmbedding<B>),
    CharacterBlock(crate::decoder::TransformerBlock<B>),
    CharacterNorm(B::Normalization),
    TtsIngress(TtsIngress<B>),
    TtsFusion(GatedFusion<B>),
    TtsBlock(crate::decoder::TransformerBlock<B>),
    TtsNorm(B::Normalization),
    TtsRvqBank(RvqBank<B>),
    TtsRvqProjection(RvqProjection<B>),
    MogBlock(MogBlock<B>),
    MogPredictionHead(MogPredictionHead<B>),
    MogComponentBank(MogComponentBank<B>),
    CodecEncoder(EncoderUnit<B::Tensor>),
    CodecDecoder(DecoderUnit<B::Tensor>),
    CodecCodebook(Codebook<B::Tensor>),
}

/// Invalid construction, missing retained role, or an ordinary mechanism failure.
#[derive(Debug, thiserror::Error)]
pub enum ConstructionError {
    /// The unit's index is outside the admitted configuration.
    #[error("VoiceChat unit {0:?} is outside the selected construction")]
    Unit(SpeechUnitId),
    /// The task omitted the requested source role.
    #[error("VoiceChat construction requires retained component {0:?}")]
    MissingComponent(Component),
    /// The normalized checkpoint configuration is invalid.
    #[error(transparent)]
    Configuration(#[from] ConfigError),
    /// Perception geometry or mechanism failure.
    #[error(transparent)]
    Perception(#[from] PerceptionError),
    /// Perception block construction failed.
    #[error(transparent)]
    Conformer(#[from] ConformerError),
    /// Causal subsampling construction failed.
    #[error(transparent)]
    Subsampling(#[from] SubsamplingError),
    /// Recurrent transcription construction failed.
    #[error(transparent)]
    Rnnt(#[from] RnntError),
    /// TTS geometry or mechanism failure.
    #[error(transparent)]
    Tts(#[from] TtsError),
    /// Function projection construction failed.
    #[error(transparent)]
    Channel(#[from] ChannelError),
    /// Codec geometry or mechanism failure.
    #[error(transparent)]
    Codec(#[from] eredu_codec::Error),
    /// Ordinary neural mechanism failure.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
    /// Constructed parameter identities disagree with retained artifact admission.
    #[error(transparent)]
    Parameters(#[from] SpeechParameterError),
    /// A semantic parameter ownership projection is inconsistent.
    #[error(transparent)]
    ParameterDeclaration(#[from] eredu_runtime::ArchitectureParameterError),
    /// Shared embedded transformer declaration failed.
    #[error(transparent)]
    Transformer(#[from] super::tts_transformer_layered::TransformerError),
}

/// Cold, validated construction geometry without artifacts or native resources.
pub struct SpeechConstructionPlan {
    config: VoiceChatConfig,
    perception: PerceptionDimensions,
    rnnt: RnntDimensions,
    character: TransformerConfig,
    tts: TransformerConfig,
    mog: MogDimensions,
    encoder: AudioEncoderConfig,
    decoder: AudioDecoderConfig,
    speaker_frames: i32,
    units: Vec<SpeechUnitId>,
}

impl SpeechConstructionPlan {
    /// Normalizes every component's geometry before creating any backend module.
    pub fn from_config(config: &VoiceChatConfig) -> Result<Self, ConstructionError> {
        config.validate()?;
        let perception = PerceptionDimensions::from_config(&config.perception)?;
        let rnnt = RnntDimensions::from_config(&config.rnnt)?;
        let character = TransformerConfig::new(&config.tts.character_encoder)?;
        let tts = TransformerConfig::new(&config.tts.backbone)?;
        let mog = MogDimensions::from_config(&config.tts)?;
        let encoder = waveform_encoder_config(&config.codec)?;
        let decoder = waveform_decoder_config(&config.codec)?;
        let speaker_frames = AriaPromptPlan::from_config(config, usize::MAX)?.frames();
        use SpeechUnitId::*;
        let mut units = vec![PerceptionSubsampling];
        units.extend((0..perception.layers).map(PerceptionBlock));
        units.extend([PerceptionProjection, Rnnt, FunctionHead, CharacterEmbedding]);
        units.extend((0..config.tts.character_encoder.num_hidden_layers).map(CharacterBlock));
        units.extend([CharacterNorm, TtsIngress, TtsFusion]);
        units.extend((0..config.tts.backbone.num_hidden_layers).map(TtsBlock));
        units.extend([TtsNorm, TtsRvqBank, TtsRvqProjection]);
        units.extend((0..mog.layers).map(MogBlock));
        units.push(MogPredictionHead);
        units.push(MogComponentBank);
        units.extend((0..encoder.neural.unit_specs()?.len()).map(CodecEncoder));
        units.extend((0..decoder.neural.unit_specs()?.len()).map(CodecDecoder));
        units.extend((0..decoder.quantizer.depth).map(CodecCodebook));
        Ok(Self {
            config: config.clone(),
            perception,
            rnnt,
            character,
            tts,
            mog,
            encoder,
            decoder,
            speaker_frames,
            units,
        })
    }

    /// Architecture order of all independently bindable learned speech units.
    pub fn units(&self) -> &[SpeechUnitId] {
        &self.units
    }

    /// Exact normalized policy used by unit construction and consuming assembly.
    pub fn config(&self) -> &VoiceChatConfig {
        &self.config
    }

    /// Canonical parameters for one semantic unit, without native construction.
    /// Ownership remains in the component's validated family declaration.
    pub fn parameter_description(
        &self,
        id: SpeechUnitId,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, ConstructionError> {
        use super::{
            perception_layered as perception, tts_conditioning_layered as conditioning,
            tts_mog_layered as mog, tts_rvq_layered as rvq,
        };
        use eredu_codec::{
            ear_decoder_layered as decoder, ear_encoder_layered as encoder,
            prvq_layered as quantizer,
        };
        use eredu_runtime::{ArchitectureParameterDescription, ParameterGroupOwner};
        use SpeechUnitId::*;
        if !self.units.contains(&id) {
            return Err(ConstructionError::Unit(id));
        }
        let (description, selected) = match id {
            PerceptionSubsampling | PerceptionBlock(_) | PerceptionProjection => {
                let unit = match id {
                    PerceptionSubsampling => (perception::SUBSAMPLING.into(), 0),
                    PerceptionBlock(index) => (perception::ENCODER.into(), index),
                    _ => (perception::PROJECTION.into(), 0),
                };
                (
                    perception::perception_parameter_description(self.perception)?,
                    Some(unit),
                )
            }
            Rnnt => (
                super::rnnt_layered::rnnt_parameter_description(self.rnnt)?,
                None,
            ),
            FunctionHead => (
                super::function_head_layered::function_head_parameter_description(
                    super::function_head_layered::FunctionHeadDimensions::new(
                        self.perception.output_width,
                        131072,
                    )?,
                    &LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?,
                )?,
                None,
            ),
            CharacterEmbedding | TtsIngress | TtsFusion => {
                let group = match id {
                    CharacterEmbedding => conditioning::CHARACTER,
                    TtsIngress => conditioning::INGRESS,
                    _ => conditioning::FUSION,
                };
                (
                    conditioning::conditioning_parameter_description(
                        conditioning::ConditioningDimensions::from_config(
                            &self.config.tts,
                            256,
                            self.speaker_frames,
                        )?,
                    )?,
                    Some((group.into(), 0)),
                )
            }
            CharacterBlock(_) | CharacterNorm | TtsBlock(_) | TtsNorm => {
                let config = match id {
                    CharacterBlock(_) | CharacterNorm => &self.character,
                    _ => &self.tts,
                };
                let unit = match id {
                    CharacterBlock(index) | TtsBlock(index) => {
                        (config.parameter_root().into(), index)
                    }
                    _ => (format!("{}.normalization", config.parameter_root()), 0),
                };
                (
                    super::tts_transformer_layered::tts_transformer_parameter_description(config)?,
                    Some(unit),
                )
            }
            TtsRvqBank | TtsRvqProjection => {
                let group = if id == TtsRvqBank {
                    rvq::BANK
                } else {
                    rvq::PROJECTION
                };
                (
                    rvq::rvq_parameter_description(RvqDimensions::from_config(&self.config.tts)?)?,
                    Some((group.into(), 0)),
                )
            }
            MogBlock(_) | MogPredictionHead | MogComponentBank => {
                let unit = match id {
                    MogBlock(index) => (mog::BLOCKS.into(), index),
                    MogPredictionHead => (mog::PREDICTION.into(), 0),
                    _ => (mog::COMPONENTS.into(), 0),
                };
                (mog::mog_parameter_description(self.mog)?, Some(unit))
            }
            CodecEncoder(index) => (
                encoder::encoder_parameter_description(
                    &self.encoder.neural,
                    "tts_model.audio_codec.encoder",
                )?,
                Some((encoder::ENCODER_GROUP.into(), index)),
            ),
            CodecDecoder(index) => (
                decoder::decoder_parameter_description(
                    &self.decoder.neural,
                    "tts_model.audio_codec.decoder",
                )?,
                Some((decoder::DECODER_GROUP.into(), index)),
            ),
            CodecCodebook(depth) => (
                quantizer::quantizer_parameter_description(
                    self.decoder.quantizer,
                    "tts_model.audio_codec.prvq",
                )?,
                Some((quantizer::codebook_group_name(depth), 0)),
            ),
        };
        let Some((selected_group, selected_index)) = selected else {
            return Ok(description);
        };
        let groups = description
            .groups()
            .iter()
            .filter(|owned| {
                matches!(
                    owned.owner(), ParameterGroupOwner::ExecutionUnit { group, global_unit, .. }
                        if group.as_str() == selected_group && *global_unit == selected_index
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        if groups.is_empty() {
            return Err(ConstructionError::Unit(id));
        }
        Ok(ArchitectureParameterDescription::from_owned(
            description.graph(),
            description.unit_layout(),
            groups,
        )?)
    }

    /// Constructs only one unit's unloaded slots. No weight source is accessed.
    pub fn build_unit<B: NeuralBackend>(
        &self,
        id: SpeechUnitId,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SpeechUnit<B>, ConstructionError> {
        if !self.units.contains(&id) {
            return Err(ConstructionError::Unit(id));
        }
        let config = &self.config.tts;
        let hidden = config.backbone.hidden_size as i32;
        Ok(match id {
            SpeechUnitId::PerceptionSubsampling => SpeechUnit::PerceptionSubsampling(
                CausalSubsampling::new(self.perception.subsampling, context)?,
            ),
            SpeechUnitId::PerceptionBlock(index) => SpeechUnit::PerceptionBlock(
                ConformerBlock::new(self.perception.block, index, context)?,
            ),
            SpeechUnitId::PerceptionProjection => {
                SpeechUnit::PerceptionProjection(self.perception.build_projection::<B>(context)?)
            }
            SpeechUnitId::Rnnt => SpeechUnit::Rnnt(Rnnt::new(self.rnnt, context)?),
            SpeechUnitId::FunctionHead => SpeechUnit::FunctionHead(FunctionHead::new(
                self.perception.output_width,
                131072,
                LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?,
                context,
            )?),
            SpeechUnitId::CharacterEmbedding => {
                SpeechUnit::CharacterEmbedding(CharacterEmbedding::new(
                    CharacterDimensions {
                        characters: 256,
                        width: config.character_encoder.hidden_size as i32,
                        output_width: hidden,
                        use_continuation: config.use_subword_flag_emb,
                        use_special: config.use_bos_eos_emb,
                    },
                    context,
                )?)
            }
            SpeechUnitId::CharacterBlock(index) => {
                SpeechUnit::CharacterBlock(self.character.build_block::<B>(index, context)?)
            }
            SpeechUnitId::CharacterNorm => {
                SpeechUnit::CharacterNorm(self.character.build_final_norm::<B>(context)?)
            }
            SpeechUnitId::TtsIngress => {
                SpeechUnit::TtsIngress(TtsIngress::new(hidden, self.speaker_frames, context)?)
            }
            SpeechUnitId::TtsFusion => SpeechUnit::TtsFusion(GatedFusion::new(hidden, context)?),
            SpeechUnitId::TtsBlock(index) => {
                SpeechUnit::TtsBlock(self.tts.build_block::<B>(index, context)?)
            }
            SpeechUnitId::TtsNorm => SpeechUnit::TtsNorm(self.tts.build_final_norm::<B>(context)?),
            SpeechUnitId::TtsRvqBank => {
                SpeechUnit::TtsRvqBank(RvqBank::new(RvqDimensions::from_config(config)?, context)?)
            }
            SpeechUnitId::TtsRvqProjection => SpeechUnit::TtsRvqProjection(RvqProjection::new(
                RvqDimensions::from_config(config)?,
                context,
            )?),
            SpeechUnitId::MogBlock(index) => {
                SpeechUnit::MogBlock(MogBlock::new(self.mog, index, context)?)
            }
            SpeechUnitId::MogPredictionHead => {
                SpeechUnit::MogPredictionHead(MogPredictionHead::new(self.mog, context)?)
            }
            SpeechUnitId::MogComponentBank => {
                SpeechUnit::MogComponentBank(MogComponentBank::new(self.mog, context)?)
            }
            SpeechUnitId::CodecEncoder(index) => SpeechUnit::CodecEncoder(EncoderUnit::new(
                self.encoder.neural.unit_specs()?[index],
                "tts_model.audio_codec.encoder",
                context,
            )?),
            SpeechUnitId::CodecDecoder(index) => SpeechUnit::CodecDecoder(DecoderUnit::new(
                self.decoder.neural.unit_specs()?[index],
                "tts_model.audio_codec.decoder",
                context,
            )?),
            SpeechUnitId::CodecCodebook(index) => SpeechUnit::CodecCodebook(Codebook::unloaded(
                self.decoder.quantizer,
                "tts_model.audio_codec.prvq",
                index,
                context,
            )?),
        })
    }
}

/// Construction authority restricted to components retained for the selected task.
pub struct PreparedSpeechConstruction<'a> {
    artifacts: &'a RetainedVoiceChatArtifacts,
    plan: SpeechConstructionPlan,
    units: Vec<SpeechUnitId>,
}

impl<'a> PreparedSpeechConstruction<'a> {
    /// Materializes independent units through shared completion-safe binding and
    /// consumes them into the ordinary coupled execution components. This low-level
    /// operation retains the resulting learned weights; it does not register a
    /// family loader or select a full-graph residency policy.
    pub fn materialize<B, P>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
        materialization: &P::MaterializationContext,
    ) -> Result<super::assembly::LearnedSpeechComponents<B>, super::assembly::AssemblyError>
    where
        B: NeuralBackend,
        P: eredu_runtime::ParameterBackend<Parameter = B::Tensor>,
    {
        use super::assembly::AssemblyError;
        // Missing companion roles fail before construction or any payload read.
        for id in self.plan.units() {
            if self.artifacts.component_source(id.component()).is_none() {
                return Err(ConstructionError::MissingComponent(id.component()).into());
            }
        }
        let mut prepared = self
            .plan
            .units()
            .iter()
            .copied()
            .map(|id| Ok((id, self.prepare_unit(id)?)))
            .collect::<Result<std::collections::BTreeMap<_, _>, ConstructionError>>()?;
        self.plan.assemble_using(context, |id, module| {
            let prepared = prepared.remove(&id).expect("declared speech unit");
            prepared
                .select::<P>()
                .and_then(|selected| selected.bind(module, materialization))
                .map_err(|error| AssemblyError::Binding {
                    unit: id,
                    source: eredu_core::BackendFailure::from_error(error),
                })
        })
    }

    /// Uses only admitted configuration and retained sources; no artifact is reopened.
    pub fn from_retained(
        artifacts: &'a RetainedVoiceChatArtifacts,
    ) -> Result<Self, ConstructionError> {
        let plan = SpeechConstructionPlan::from_config(artifacts.plan().config())?;
        let units = plan
            .units()
            .iter()
            .copied()
            .filter(|id| artifacts.component_source(id.component()).is_some())
            .collect();
        Ok(Self {
            artifacts,
            plan,
            units,
        })
    }

    /// Only units whose complete component source is retained for this task.
    pub fn units(&self) -> &[SpeechUnitId] {
        &self.units
    }

    /// Resolves one semantic unit's exact recipes before native construction.
    /// Neighboring modules and host buffers grant no source authority.
    pub fn prepare_unit(
        &self,
        id: SpeechUnitId,
    ) -> Result<PreparedSpeechParameterUnit, ConstructionError> {
        if self.artifacts.component_source(id.component()).is_none() {
            return Err(ConstructionError::MissingComponent(id.component()));
        }
        Ok(PreparedSpeechParameterUnit::from_description(
            self.artifacts,
            id.component(),
            self.plan.parameter_description(id)?,
        )?)
    }

    /// Constructs an unloaded unit and its exact independently selectable bindings.
    /// Missing roles and invalid indices are rejected before native construction.
    pub fn build_unit<B: NeuralBackend>(
        &self,
        id: SpeechUnitId,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(SpeechUnit<B>, PreparedSpeechParameterUnit), ConstructionError> {
        let parameters = self.prepare_unit(id)?;
        let module = self.plan.build_unit::<B>(id, context)?;
        Ok((module, parameters))
    }
}
