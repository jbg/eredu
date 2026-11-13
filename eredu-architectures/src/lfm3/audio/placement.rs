//! Cold parameter ownership for LFM3 Audio.

use super::{artifact::checkpoint, ArtifactError, BindingSource, DepthUnit, PreparedAudio, Unit};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::TensorSelection,
};
use eredu_codec::liquid::DecoderUnit;
use eredu_nn::{DeclareLogicalParameters, LogicalParameterDeclaration};
use std::collections::BTreeMap;

/// Independent parameter owners in the Audio graph. Tied tables have one owner
/// even when the equations consume them at multiple points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioWeightUnit {
    /// Language token embedding, also used for the tied vocabulary head.
    LanguageEmbedding,
    /// One causal language block.
    LanguageBlock(usize),
    /// Language output normalization and any untied vocabulary head.
    LanguageOutput,
    /// Speech downsampler, encoder blocks, normalization and projector.
    Encoder(Unit),
    /// Speech depth, tied codebook tables and feedback.
    Depth(DepthUnit),
    /// LiquidCodec synthesis component.
    Codec(DecoderUnit),
}

/// One parameter unit prepared from its retained checkpoint source.
pub struct PreparedAudioWeightUnit {
    /// Semantic parameter owner.
    pub unit: AudioWeightUnit,
    /// Restricted source and exact canonical binding recipes.
    pub weights: BindingSource,
    /// Materialized parameter bytes, excluding state and scratch storage.
    pub parameter_bytes: u64,
}

pub(super) struct WeightScope<'a> {
    pub source: &'a BindingSource,
    pub units: Vec<PreparedAudioWeightUnit>,
}

impl WeightScope<'_> {
    pub fn finish(self, scope: &str) -> Result<(BindingSource, u64), ArtifactError> {
        let mut recipes = BTreeMap::new();
        let mut bytes = 0u64;
        for unit in self.units {
            for (name, recipe) in unit.weights.recipes() {
                if recipes.insert(name.clone(), recipe.clone()).is_some() {
                    return Err(checkpoint(format!(
                        "Audio parameter {name} has multiple owners"
                    )));
                }
            }
            bytes = add_bytes(bytes, unit.parameter_bytes)?;
        }
        Ok((self.source.project(scope, recipes)?, bytes))
    }
}

fn add_bytes(total: u64, bytes: u64) -> Result<u64, ArtifactError> {
    total
        .checked_add(bytes)
        .ok_or_else(|| checkpoint("Audio parameter size overflowed"))
}

fn prepare_unit(
    source: &BindingSource,
    unit: AudioWeightUnit,
    parameters: Vec<LogicalParameterDeclaration<'_>>,
    dtype: RecipeDtype,
) -> Result<PreparedAudioWeightUnit, ArtifactError> {
    let mut recipes = BTreeMap::new();
    let mut bytes = 0;
    for parameter in parameters {
        let member = eredu_runtime::ParameterMemberSpec::from_logical_declaration(
            &parameter,
            eredu_runtime::MemberSharding::Replicated,
        );
        let members = if let Some(format) = parameter.linear_format {
            eredu_runtime::expand_linear_format_parameter(&member, format).map_err(checkpoint)?
        } else {
            vec![member]
        };
        for member in members {
            let name = member.target();
            let input = source
                .recipes()
                .get(name)
                .cloned()
                .unwrap_or_else(|| DerivedWeightRecipe::source(name, TensorSelection::Full));
            let recipe =
                crate::lfm3::checkpoint::language_storage_recipe(&member, input, dtype.clone());
            let metadata = recipe.infer(source.source().as_ref()).map_err(checkpoint)?;
            if metadata.shape() != member.global_shape() {
                return Err(checkpoint(format!(
                    "Audio parameter {name} has shape {:?}, expected {:?}",
                    metadata.shape(),
                    member.global_shape(),
                )));
            }
            if recipes.insert(name.to_owned(), recipe).is_some() {
                return Err(checkpoint(format!(
                    "Audio parameter {name} is duplicated in {unit:?}"
                )));
            }
            bytes = add_bytes(bytes, metadata.byte_len())?;
        }
    }
    Ok(PreparedAudioWeightUnit {
        unit,
        weights: source.project(&format!("LFM3 Audio {unit:?}"), recipes)?,
        parameter_bytes: bytes,
    })
}

impl PreparedAudio {
    /// Prepares exact independent parameter sources without reading tensor payloads.
    pub fn prepare_weight_units(&self) -> Result<Vec<PreparedAudioWeightUnit>, ArtifactError> {
        Ok(self
            .weight_scopes()?
            .into_iter()
            .flat_map(|scope| scope.units)
            .collect())
    }

    pub(super) fn weight_scopes(&self) -> Result<[WeightScope<'_>; 4], ArtifactError> {
        let statics = &self.language.statics;
        let language = &self.language_weights;
        let dtype = &self.neural_weight_dtype;
        let mut language_units = vec![prepare_unit(
            language,
            AudioWeightUnit::LanguageEmbedding,
            statics.embedding_spec().logical_parameters()?,
            dtype.clone(),
        )?];
        for (layer, block) in self.language.blocks.iter().enumerate() {
            language_units.push(prepare_unit(
                language,
                AudioWeightUnit::LanguageBlock(layer),
                block.logical_parameters()?,
                dtype.clone(),
            )?);
        }
        let mut output = statics.normalization_spec().logical_parameters()?;
        if let Some(head) = statics.head_spec() {
            output.extend(head.logical_parameters()?);
        }
        language_units.push(prepare_unit(
            language,
            AudioWeightUnit::LanguageOutput,
            output,
            dtype.clone(),
        )?);
        Ok([
            WeightScope {
                source: language,
                units: language_units,
            },
            WeightScope {
                source: &self.encoder_weights,
                units: self
                    .encoder
                    .units()
                    .map(|unit| {
                        prepare_unit(
                            &self.encoder_weights,
                            AudioWeightUnit::Encoder(unit),
                            self.encoder.unit_parameters(unit)?,
                            dtype.clone(),
                        )
                    })
                    .collect::<Result<_, _>>()?,
            },
            WeightScope {
                source: &self.depth_weights,
                units: self
                    .depth
                    .units()
                    .map(|unit| {
                        prepare_unit(
                            &self.depth_weights,
                            AudioWeightUnit::Depth(unit),
                            self.depth.unit_parameters(unit)?,
                            dtype.clone(),
                        )
                    })
                    .collect::<Result<_, _>>()?,
            },
            WeightScope {
                source: &self.codec_weights,
                units: self
                    .codec
                    .decoder_units()
                    .map(|unit| {
                        prepare_unit(
                            &self.codec_weights,
                            AudioWeightUnit::Codec(unit),
                            self.codec.decoder_unit_parameters(unit)?,
                            RecipeDtype::F32,
                        )
                    })
                    .collect::<Result<_, _>>()?,
            },
        ])
    }
}

/// Independent parameter limits for the Audio graph.
#[derive(Debug, Clone, Copy, Default)]
pub struct AudioParameterResidency {
    /// Language blocks and their pinned vocabulary and final normalization.
    pub language: eredu_runtime::LayerWeightResidency,
    /// Speech downsampler, encoder blocks, normalization and projector.
    pub encoder: eredu_runtime::LayerWeightResidency,
    /// Speech depth and feedback.
    pub depth: eredu_runtime::LayerWeightResidency,
    /// LiquidCodec synthesis.
    pub codec: eredu_runtime::LayerWeightResidency,
}

/// Exact cold LiquidCodec synthesis binding plan retained by Audio.
pub struct PreparedAudioCodec {
    declaration: eredu_codec::liquid::Declaration,
    source: eredu_checkpoint::store::SharedCheckpointSource,
    invocations: eredu_runtime::ExecutionUnitAddressMap,
    units: Vec<Vec<eredu_runtime::WeightBinding>>,
}

impl PreparedAudioCodec {
    pub(super) fn new(
        declaration: eredu_codec::liquid::Declaration,
        weights: &BindingSource,
    ) -> Result<Self, ArtifactError> {
        let invocations = declaration.decoder_invocations()?;
        let (source, units) = prepare_audio_units(
            weights,
            "lfm3/audio/codec",
            declaration.decoder_units().map(|unit| {
                declaration
                    .decoder_unit_parameters(unit)
                    .map_err(ArtifactError::from)
            }),
        )?;
        Ok(Self {
            declaration,
            source,
            invocations,
            units,
        })
    }

    /// Transfers the selected decoder bindings and geometry.
    pub fn into_parts(
        self,
    ) -> (
        eredu_codec::liquid::Declaration,
        eredu_checkpoint::store::SharedCheckpointSource,
        eredu_runtime::ExecutionUnitAddressMap,
        Vec<Vec<eredu_runtime::WeightBinding>>,
    ) {
        (self.declaration, self.source, self.invocations, self.units)
    }
}

pub(super) fn prepare_audio_units<'a>(
    weights: &BindingSource,
    scope: &str,
    parameters: impl IntoIterator<Item = Result<Vec<LogicalParameterDeclaration<'a>>, ArtifactError>>,
) -> Result<
    (
        eredu_checkpoint::store::SharedCheckpointSource,
        Vec<Vec<eredu_runtime::WeightBinding>>,
    ),
    ArtifactError,
> {
    let recipes = parameters
        .into_iter()
        .enumerate()
        .map(|(ordinal, parameters)| {
            let outputs = parameters?
                .into_iter()
                .flat_map(|parameter| {
                    std::iter::once(parameter.parameter).chain(
                        parameter.linear_format.into_iter().flat_map(|format| {
                            [format.scale(), format.affine_bias(), format.global_scale()]
                                .into_iter()
                                .flatten()
                        }),
                    )
                })
                .map(|parameter| {
                    let name = parameter.id.to_string();
                    let recipe = weights.recipes().get(&name).cloned().ok_or_else(|| {
                        checkpoint(format!("missing prepared {scope} parameter {name}"))
                    })?;
                    Ok((name, recipe))
                })
                .collect::<Result<_, ArtifactError>>()?;
            Ok((
                ordinal,
                eredu_runtime::ParameterResourceRecipes {
                    outputs,
                    aliases: Vec::new(),
                },
            ))
        })
        .collect::<Result<_, ArtifactError>>()?;
    let (source, units) =
        eredu_runtime::prepare_resource_bindings(weights.source().clone(), scope, recipes)
            .map_err(checkpoint)?;
    Ok((source, units.into_values().collect()))
}
