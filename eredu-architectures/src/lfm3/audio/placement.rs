//! Cold parameter ownership for LFM3 Audio.

use super::{artifact::checkpoint, ArtifactError, BindingSource};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::TensorSelection,
};
use eredu_nn::LogicalParameterDeclaration;
use eredu_runtime::{MemberSharding, ParameterMemberSpec};
use std::collections::{BTreeMap, BTreeSet};

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

/// Cold language topology selected for an Audio or Omni speech graph.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AudioPlacement {
    /// One complete language model.
    Local,
    /// One tensor rank without stage transport.
    Tensor(eredu_core::ParallelRankTopology),
    /// One stage rank, optionally also tensor sharded.
    Pipeline(eredu_core::ParallelRankTopology),
}

impl AudioPlacement {
    /// Returns the retained native-binding rank when distributed.
    pub const fn rank(self) -> Option<eredu_core::ParallelRankTopology> {
        match self {
            Self::Local => None,
            Self::Tensor(rank) | Self::Pipeline(rank) => Some(rank),
        }
    }
}

/// Exact cold LiquidCodec synthesis binding plan retained by Audio.
pub(super) struct PreparedAudioCodec {
    declaration: eredu_codec::liquid::Declaration,
    source: eredu_checkpoint::store::SharedCheckpointSource,
    invocations: eredu_runtime::ExecutionUnitAddressMap,
    units: Vec<Vec<eredu_runtime::WeightBinding>>,
}

impl PreparedAudioCodec {
    pub(super) fn parameter_bytes(
        &self,
        residency: eredu_runtime::LayerWeightResidency,
    ) -> Result<u64, super::ResidentAudioError> {
        super::placement::admit_audio_weights("codec", &self.invocations, &self.units, 0, residency)
    }

    pub(super) fn new(
        declaration: eredu_codec::liquid::Declaration,
        weights: &BindingSource,
    ) -> Result<Self, ArtifactError> {
        let invocations = declaration.decoder_invocations()?;
        let (source, units) = prepare_audio_units(
            weights,
            "lfm3/audio/codec",
            &RecipeDtype::F32,
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

    pub(super) fn bind<B: eredu_nn::NeuralBackend, M: eredu_runtime::ParameterUnitMechanisms<B>>(
        self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mechanisms: &M,
        residency: eredu_runtime::LayerWeightResidency,
    ) -> Result<
        eredu_codec::liquid::PolicyDecoder<B, M::UnitPolicy<eredu_codec::liquid::DecoderModule<B>>>,
        eredu_core::BackendFailure,
    > {
        self.declaration.bind_decoder(
            self.source,
            self.invocations,
            self.units,
            context,
            |source, addresses, units, build| {
                mechanisms
                    .bind_units(source, addresses, units, residency, build)
                    .map(|(policy, _)| policy)
            },
        )
    }
}

pub(super) fn audio_storage_recipe(
    weights: &BindingSource,
    member: &ParameterMemberSpec,
    dtype: &RecipeDtype,
) -> DerivedWeightRecipe {
    let name = member.target();
    let input = weights
        .recipes()
        .get(name)
        .cloned()
        .unwrap_or_else(|| DerivedWeightRecipe::source(name, TensorSelection::Full));
    crate::lfm3::checkpoint::language_storage_recipe(member, input, dtype.clone())
}

pub(super) fn prepare_audio_units<'a>(
    weights: &BindingSource,
    scope: &str,
    dtype: &RecipeDtype,
    parameters: impl IntoIterator<Item = Result<Vec<LogicalParameterDeclaration<'a>>, ArtifactError>>,
) -> Result<
    (
        eredu_checkpoint::store::SharedCheckpointSource,
        Vec<Vec<eredu_runtime::WeightBinding>>,
    ),
    ArtifactError,
> {
    let mut owned = BTreeSet::new();
    let recipes = parameters
        .into_iter()
        .enumerate()
        .map(|(ordinal, parameters)| {
            let mut outputs = BTreeMap::new();
            for parameter in parameters? {
                let member = ParameterMemberSpec::from_logical_declaration(
                    &parameter,
                    MemberSharding::Replicated,
                );
                let members = match parameter.linear_format {
                    Some(format) => eredu_runtime::expand_linear_format_parameter(&member, format)
                        .map_err(checkpoint)?,
                    None => vec![member],
                };
                for member in members {
                    let name = member.target();
                    if !owned.insert(name.to_owned()) {
                        return Err(checkpoint(format!(
                            "{scope} parameter {name} has multiple owners"
                        )));
                    }
                    let recipe = audio_storage_recipe(weights, &member, dtype);
                    let metadata = recipe
                        .infer(weights.source().as_ref())
                        .map_err(checkpoint)?;
                    if metadata.shape() != member.global_shape() {
                        return Err(checkpoint(format!(
                            "{scope} parameter {name} has shape {:?}, expected {:?}",
                            metadata.shape(),
                            member.global_shape(),
                        )));
                    }
                    outputs.insert(name.to_owned(), recipe);
                }
            }
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

pub(super) fn admit_audio_weights(
    component: &str,
    invocations: &eredu_runtime::ExecutionUnitAddressMap,
    units: &[Vec<eredu_runtime::WeightBinding>],
    pinned: u64,
    residency: eredu_runtime::LayerWeightResidency,
) -> Result<u64, super::ResidentAudioError> {
    use super::artifact::checkpoint;
    use eredu_runtime::{execution_unit_parameter_window_bytes, physical_binding_bytes};
    let units = units
        .iter()
        .map(|unit| physical_binding_bytes(unit))
        .collect::<Result<Vec<_>, _>>()
        .map_err(checkpoint)?;
    let window = execution_unit_parameter_window_bytes(invocations, &units, residency)
        .map_err(checkpoint)?;
    let overflow = || checkpoint("Audio parameter byte count overflow");
    let required = pinned.checked_add(window).ok_or_else(overflow)?;
    if let Some(limit) = residency.offload()?.device_budget_bytes() {
        if required > limit {
            return Err(super::ResidentAudioError::Selection(format!(
                "{component} device parameter window requires {required} bytes, budget {limit}"
            )));
        }
    }
    units.into_iter().try_fold(pinned, |sum, bytes| {
        sum.checked_add(bytes)
            .ok_or_else(overflow)
            .map_err(Into::into)
    })
}
