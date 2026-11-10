//! Architecture-owned construction of independently bindable speech units.
//!
//! Configuration selects equations and parameter identities. Retained artifacts
//! supply exact recipes; a backend only constructs ordinary neural mechanisms.
//! A unit has no session state, residency schedule, or completion authority.

use crate::decoder::Config;

use eredu_codec::{ear_decoder::DecoderUnit, ear_encoder::EncoderUnit, prvq::Codebook};
use eredu_nn::{NeuralBackend, Parameterized, Tensor};

use super::{
    artifact::RetainedVoiceChatArtifacts,
    channels::{ChannelError, FunctionHead},
    checkpoint::Component,
    config::{ConfigError, VoiceChatConfig},
    conformer::{ConformerBlock, ConformerError},
    parameters::{PreparedSpeechParameterUnit, SpeechParameterError},
    perception::PerceptionError,
    rnnt::{Rnnt, RnntError},
    subsampling::{CausalSubsampling, SubsamplingError},
    tts_conditioning::{CharacterEmbedding, GatedFusion},
    tts_ingress::TtsIngress,
    tts_mog::{MogBlock, MogComponentBank, MogPredictionHead, TtsError},
    tts_rvq::{RvqBank, RvqProjection},
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
    pub(super) perception: &'static super::perception_specs::PerceptionDeclaration,
    rnnt: &'static super::rnnt_specs::RnntDeclaration,
    function: &'static super::function_head_layered::FunctionHeadDeclaration,
    pub(super) character: &'static super::tts_transformer_specs::TransformerDeclaration,
    pub(super) tts: &'static super::tts_transformer_specs::TransformerDeclaration,
    pub(super) mog: &'static super::tts_mog_specs::MogDeclaration,
    pub(super) conditioning: &'static super::tts_conditioning_specs::ConditioningDeclaration,
    pub(super) rvq: &'static super::tts_rvq_specs::RvqDeclaration,
    pub(super) codec: &'static super::codec::CodecDeclarations,
    units: Vec<SpeechUnitId>,
}

impl SpeechConstructionPlan {
    /// Normalizes every component's geometry before creating any backend module.
    pub fn from_config(config: &VoiceChatConfig) -> Result<Self, ConstructionError> {
        config.validate()?;
        let perception = super::perception_specs::released_declaration()?;
        let rnnt = super::rnnt_specs::released_declaration()?;
        let function = super::function_head_layered::released_declaration()?;
        let transformers = super::tts_transformer_specs::released_declarations()?;
        let character = &transformers.character;
        let tts = &transformers.causal;
        let mog = super::tts_mog_specs::released_declaration()?;
        let codec = super::codec::released_declarations()?;
        let conditioning = super::tts_conditioning_specs::released_declaration()?;
        let rvq = super::tts_rvq_specs::released_declaration()?;
        use SpeechUnitId::*;
        let mut units = vec![PerceptionSubsampling];
        units.extend((0..perception.spec().blocks.len()).map(PerceptionBlock));
        units.extend([PerceptionProjection, Rnnt, FunctionHead, CharacterEmbedding]);
        units.extend((0..character.spec().parameters.blocks.len()).map(CharacterBlock));
        units.extend([CharacterNorm, TtsIngress, TtsFusion]);
        units.extend((0..tts.spec().parameters.blocks.len()).map(TtsBlock));
        units.extend([TtsNorm, TtsRvqBank, TtsRvqProjection]);
        units.extend((0..mog.spec().blocks.len()).map(MogBlock));
        units.push(MogPredictionHead);
        units.push(MogComponentBank);
        units.extend((0..codec.encoder.spec().units().len()).map(CodecEncoder));
        units.extend((0..codec.decoder.spec().units().len()).map(CodecDecoder));
        units.extend((0..codec.quantizer.spec().codebooks().len()).map(CodecCodebook));
        Ok(Self {
            config: config.clone(),
            perception,
            rnnt,
            function,
            character,
            tts,
            mog,
            conditioning,
            rvq,
            codec,
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
                    self.perception.definition().parameters().clone(),
                    Some(unit),
                )
            }
            Rnnt => (self.rnnt.definition().parameters().clone(), None),
            FunctionHead => (self.function.definition().parameters().clone(), None),
            CharacterEmbedding | TtsIngress | TtsFusion => {
                let group = match id {
                    CharacterEmbedding => conditioning::CHARACTER,
                    TtsIngress => conditioning::INGRESS,
                    _ => conditioning::FUSION,
                };
                (
                    self.conditioning.definition().parameters().clone(),
                    Some((group.into(), 0)),
                )
            }
            CharacterBlock(_) | CharacterNorm | TtsBlock(_) | TtsNorm => {
                let declaration = match id {
                    CharacterBlock(_) | CharacterNorm => self.character,
                    _ => self.tts,
                };
                let config = declaration.config();
                let unit = match id {
                    CharacterBlock(index) | TtsBlock(index) => {
                        (config.parameter_root().into(), index)
                    }
                    _ => (format!("{}.normalization", config.parameter_root()), 0),
                };
                (declaration.definition().parameters().clone(), Some(unit))
            }
            TtsRvqBank | TtsRvqProjection => {
                let group = if id == TtsRvqBank {
                    rvq::BANK
                } else {
                    rvq::PROJECTION
                };
                (
                    self.rvq.definition().parameters().clone(),
                    Some((group.into(), 0)),
                )
            }
            MogBlock(_) | MogPredictionHead | MogComponentBank => {
                let unit = match id {
                    MogBlock(index) => (mog::BLOCKS.into(), index),
                    MogPredictionHead => (mog::PREDICTION.into(), 0),
                    _ => (mog::COMPONENTS.into(), 0),
                };
                (self.mog.definition().parameters().clone(), Some(unit))
            }
            CodecEncoder(index) => (
                self.codec.encoder.spec().parameter_description()?,
                Some((encoder::ENCODER_GROUP.into(), index)),
            ),
            CodecDecoder(index) => (
                self.codec.decoder.spec().parameter_description()?,
                Some((decoder::DECODER_GROUP.into(), index)),
            ),
            CodecCodebook(depth) => (
                self.codec.quantizer.spec().parameter_description()?,
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
        Ok(match id {
            SpeechUnitId::PerceptionSubsampling => SpeechUnit::PerceptionSubsampling(
                CausalSubsampling::new(self.perception.spec().pinned.subsampling.clone(), context)?,
            ),
            SpeechUnitId::PerceptionBlock(index) => SpeechUnit::PerceptionBlock(
                ConformerBlock::new(self.perception.spec().blocks[index].clone(), context)?,
            ),
            SpeechUnitId::PerceptionProjection => SpeechUnit::PerceptionProjection(B::linear(
                self.perception.spec().pinned.projection.clone(),
                context,
            )?),
            SpeechUnitId::Rnnt => SpeechUnit::Rnnt(Rnnt::new(self.rnnt.clone(), context)?),
            SpeechUnitId::FunctionHead => {
                SpeechUnit::FunctionHead(FunctionHead::new(self.function.spec().clone(), context)?)
            }
            SpeechUnitId::CharacterEmbedding => SpeechUnit::CharacterEmbedding(
                CharacterEmbedding::new(self.conditioning.spec().character.clone(), context)?,
            ),
            SpeechUnitId::CharacterBlock(index) => {
                SpeechUnit::CharacterBlock(crate::decoder::TransformerBlock::from_parameters(
                    self.character.config(),
                    index,
                    &self.character.spec().parameters.blocks[index],
                    context,
                )?)
            }
            SpeechUnitId::CharacterNorm => SpeechUnit::CharacterNorm(B::normalization(
                self.character.spec().parameters.normalization.clone(),
                context,
            )?),
            SpeechUnitId::TtsIngress => SpeechUnit::TtsIngress(TtsIngress::new(
                self.conditioning.spec().ingress.clone(),
                context,
            )?),
            SpeechUnitId::TtsFusion => SpeechUnit::TtsFusion(GatedFusion::new(
                self.conditioning.spec().fusion.clone(),
                context,
            )?),
            SpeechUnitId::TtsBlock(index) => {
                SpeechUnit::TtsBlock(crate::decoder::TransformerBlock::from_parameters(
                    self.tts.config(),
                    index,
                    &self.tts.spec().parameters.blocks[index],
                    context,
                )?)
            }
            SpeechUnitId::TtsNorm => SpeechUnit::TtsNorm(B::normalization(
                self.tts.spec().parameters.normalization.clone(),
                context,
            )?),
            SpeechUnitId::TtsRvqBank => {
                SpeechUnit::TtsRvqBank(RvqBank::new(self.rvq.spec().bank.clone(), context)?)
            }
            SpeechUnitId::TtsRvqProjection => SpeechUnit::TtsRvqProjection(RvqProjection::new(
                self.rvq.spec().projection.clone(),
                context,
            )?),
            SpeechUnitId::MogBlock(index) => SpeechUnit::MogBlock(MogBlock::new(
                self.mog.spec().blocks[index].clone(),
                context,
            )?),
            SpeechUnitId::MogPredictionHead => SpeechUnit::MogPredictionHead(
                MogPredictionHead::new(self.mog.spec().prediction.clone(), context)?,
            ),
            SpeechUnitId::MogComponentBank => SpeechUnit::MogComponentBank(MogComponentBank::new(
                self.mog.spec().components.clone(),
                context,
            )?),
            SpeechUnitId::CodecEncoder(index) => SpeechUnit::CodecEncoder(EncoderUnit::new(
                self.codec.encoder.spec().units()[index].clone(),
                context,
            )?),
            SpeechUnitId::CodecDecoder(index) => SpeechUnit::CodecDecoder(DecoderUnit::new(
                self.codec.decoder.spec().units()[index].clone(),
                context,
            )?),
            SpeechUnitId::CodecCodebook(index) => SpeechUnit::CodecCodebook(Codebook::new(
                self.codec.quantizer.spec().codebooks()[index].clone(),
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
