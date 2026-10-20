//! Exact retained binding plans for the ordinary Nemotron runtime.

use std::{collections::BTreeSet, sync::Arc};

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, RecipeAlias, RecipeError},
    store::{RestrictedCheckpointSource, StoreError},
};
use eredu_core::BackendFailure;
use eredu_nn::{
    validate_parameter_topology, AttentionCache, DistributedNeuralBackend, GroupedNeuralBackend,
    ParameterTopologyError, Parameterized, Tensor,
};
use eredu_runtime::{
    bind_materialized_unit, bindings_from_recipe_set, materialize_selected_bindings,
    select_bindings, ExecutionUnitLayout, LayerRuntimeState, LayeredArchitecture, ParameterBackend,
    RecipeBindingError, ResidentRuntime, RuntimeStateComponents, SelectedBindingPlan,
    WeightBinding,
};

use super::*;

/// Independently bound portion of the ordinary Nemotron target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackboneUnit {
    /// Embedding, final normalization, and agent text head.
    Static,
    /// One target block in architecture order.
    Layer(usize),
}

/// A construction failure returns no partially loaded runtime.
#[derive(Debug, thiserror::Error)]
pub enum BackboneConstructionError {
    /// Retained preparation does not describe the requested target.
    #[error(transparent)]
    Preparation(#[from] BackbonePreparationError),
    /// Ordinary Nemotron module or runtime construction failed.
    #[error(transparent)]
    Architecture(#[from] eredu_nn::Error),
    /// The prepared target unexpectedly includes prediction groups.
    #[error("VoiceChat backbone construction requires one target group")]
    Schedule,
    /// Constructed slots differ from the exact retained recipe identities.
    #[error(
        "VoiceChat {unit:?} parameter identities differ: missing {missing:?}, unexpected {unexpected:?}"
    )]
    Parameters {
        /// Unit whose declaration disagrees.
        unit: BackboneUnit,
        /// Constructed slots lacking a retained recipe.
        missing: Vec<String>,
        /// Retained recipes without a matching constructed slot.
        unexpected: Vec<String>,
    },
    /// A declared shared slot has a different retained recipe from its owner.
    #[error("VoiceChat backbone alias {alias} differs from {owner}")]
    Alias {
        /// Shared slot identity.
        alias: String,
        /// Storage owner identity.
        owner: String,
    },
    /// Ordinary module parameter identities are inconsistent.
    #[error(transparent)]
    Topology(#[from] ParameterTopologyError),
    /// Exact source restriction failed.
    #[error(transparent)]
    Source(#[from] StoreError),
    /// A retained recipe failed metadata admission.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Shared binding declarations are inconsistent.
    #[error(transparent)]
    Declaration(#[from] RecipeBindingError),
    /// Backend selection, completion, or binding failed with its source retained.
    #[error("VoiceChat backbone binding failed for {unit:?}: {source}")]
    Binding {
        /// Exact portion being selected or bound.
        unit: BackboneUnit,
        /// Original mechanism error, without exposing a native error type.
        #[source]
        source: BackendFailure,
    },
}

/// Exact static and independently materializable target-block bindings.
///
/// Construction validates the ordinary Nemotron slots against retained recipes
/// without reading weight payloads. The source exposes only physical inputs used
/// by the selected target; static vocabulary weights retain their own bindings.
pub struct PreparedBackboneBindings {
    source: SharedCheckpointSource,
    layout: ExecutionUnitLayout,
    static_bindings: Vec<WeightBinding>,
    units: Vec<Vec<WeightBinding>>,
}

impl PreparedBackboneBindings {
    /// Exact retained physical inputs, including source guards and provenance.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
    }

    /// Existing Nemotron target traversal in canonical block order.
    pub fn layout(&self) -> &ExecutionUnitLayout {
        &self.layout
    }

    /// Embedding, final normalization and agent readout bindings.
    pub fn static_bindings(&self) -> &[WeightBinding] {
        &self.static_bindings
    }

    /// Independently loadable target block bindings.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        &self.units
    }

    /// Transfers exact source authority and binding order to a shared provider.
    pub fn into_parts(
        self,
    ) -> (
        SharedCheckpointSource,
        ExecutionUnitLayout,
        Vec<WeightBinding>,
        Vec<Vec<WeightBinding>>,
    ) {
        (self.source, self.layout, self.static_bindings, self.units)
    }
}

impl PreparedBackboneParameters {
    /// Builds unloaded ordinary modules and validates all exact provider bindings.
    ///
    /// No weight payload is read. The returned architecture and plan share the
    /// retained configuration, so native binding cannot substitute another model
    /// geometry or rediscover checkpoint naming and numerical transformations.
    pub fn prepare_layerwise<B, S>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(nemotron_h::LayeredModel<B>, PreparedBackboneBindings), BackboneConstructionError>
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let model = nemotron_h::CompleteNemotronKernel::<B>::new(self.args.clone(), context)?;
        let graph =
            <nemotron_h::LayeredModel<B> as LayeredArchitecture<B, S>>::execution_graph(&model)?;
        let count = self.args.num_hidden_layers as usize;
        if graph.groups().len() != 1
            || <nemotron_h::LayeredModel<B> as LayeredArchitecture<B, S>>::group_unit_count(
                &model, 0,
            )? != count
        {
            return Err(BackboneConstructionError::Schedule);
        }
        let layout = ExecutionUnitLayout::new(&graph, [count]).map_err(eredu_nn::Error::backend)?;
        let (_, static_bindings) = prepare_bindings(
            self.source.clone(),
            self.static_recipes(),
            model.kernel().static_modules(),
            BackboneUnit::Static,
        )?;
        let mut units = Vec::with_capacity(count);
        for index in 0..count {
            let unit = <nemotron_h::LayeredModel<B> as LayeredArchitecture<B, S>>::build_unit(
                &model, 0, index, context,
            )?;
            let (_, bindings) = prepare_bindings(
                self.source.clone(),
                self.layer_recipes(index)?,
                &unit,
                BackboneUnit::Layer(index),
            )?;
            units.push(bindings);
        }
        let inputs = static_bindings
            .iter()
            .chain(units.iter().flatten())
            .flat_map(|binding| binding.checkpoint_keys().into_iter().map(str::to_owned))
            .collect();
        let source = Arc::new(RestrictedCheckpointSource::including(
            self.source.clone(),
            "voicechat/backbone/target",
            inputs,
        )?);
        Ok((
            model,
            PreparedBackboneBindings {
                source,
                layout,
                static_bindings,
                units,
            },
        ))
    }

    /// Binds the ordinary resident Nemotron runtime from exact retained recipes.
    ///
    /// All units pass recipe and backend admission before the first payload read.
    /// Each unit reads only its own retained physical inputs and establishes exact
    /// materialization completion before replacing slots. No artifact is reopened
    /// and no physical naming or conversion policy is reconstructed. The result
    /// retains learned weights; it does not create caches or select full speech
    /// graph residency, parallelism, or a public realtime session.
    pub fn materialize_resident<B, S, P>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
        materialization: &P::MaterializationContext,
    ) -> Result<ResidentRuntime<nemotron_h::LayeredModel<B>, B, S>, BackboneConstructionError>
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: ParameterBackend<Parameter = B::Tensor>,
    {
        let model = nemotron_h::CompleteNemotronKernel::<B>::new(self.args.clone(), context)?;
        let mut runtime = ResidentRuntime::<_, B, S>::new(model, context)?;
        if runtime.units().len() != 1
            || runtime.units()[0].len() != self.args.num_hidden_layers as usize
        {
            return Err(BackboneConstructionError::Schedule);
        }
        let select = |id, source, bindings| {
            select_bindings::<P>(source, bindings).map_err(|source| {
                BackboneConstructionError::Binding {
                    unit: id,
                    source: BackendFailure::from_error(source),
                }
            })
        };
        let (source, bindings) = prepare_bindings(
            self.source.clone(),
            self.static_recipes(),
            runtime.architecture().kernel().static_modules(),
            BackboneUnit::Static,
        )?;
        let static_plan = select(BackboneUnit::Static, source, bindings)?;
        let mut layers = Vec::with_capacity(runtime.units()[0].len());
        for (index, unit) in runtime.units()[0].iter().enumerate() {
            let id = BackboneUnit::Layer(index);
            let (source, bindings) =
                prepare_bindings(self.source.clone(), self.layer_recipes(index)?, unit, id)?;
            layers.push(select(id, source, bindings)?);
        }
        bind::<P, _>(
            runtime.architecture_mut().kernel_mut().static_modules_mut(),
            static_plan,
            materialization,
            BackboneUnit::Static,
        )?;
        for (index, (unit, selected)) in runtime.units_mut()[0].iter_mut().zip(layers).enumerate() {
            bind::<P, _>(unit, selected, materialization, BackboneUnit::Layer(index))?;
        }
        Ok(runtime)
    }
}

fn bind<P: ParameterBackend, M: Parameterized<P::Parameter>>(
    module: &mut M,
    selected: SelectedBindingPlan<P>,
    context: &P::MaterializationContext,
    unit: BackboneUnit,
) -> Result<(), BackboneConstructionError> {
    materialize_selected_bindings::<P>(selected, context)
        .and_then(|materialized| bind_materialized_unit::<P, _>(module, materialized))
        .map_err(|source| BackboneConstructionError::Binding {
            unit,
            source: BackendFailure::from_error(source),
        })
}

pub(super) fn prepare_bindings<T: 'static, M: Parameterized<T>>(
    source: SharedCheckpointSource,
    mut recipes: BTreeMap<String, DerivedWeightRecipe>,
    module: &M,
    unit: BackboneUnit,
) -> Result<(SharedCheckpointSource, Vec<WeightBinding>), BackboneConstructionError> {
    let topology = validate_parameter_topology(module)?;
    let expected = topology
        .iter()
        .map(|parameter| parameter.id.as_str().to_owned())
        .collect::<BTreeSet<_>>();
    let admitted = recipes.keys().cloned().collect::<BTreeSet<_>>();
    if expected != admitted {
        return Err(BackboneConstructionError::Parameters {
            unit,
            missing: expected.difference(&admitted).cloned().collect(),
            unexpected: admitted.difference(&expected).cloned().collect(),
        });
    }
    let mut aliases = Vec::new();
    for parameter in &topology {
        if let Some(owner) = &parameter.alias_of {
            let alias = parameter.id.as_str();
            if recipes[alias] != recipes[owner.as_str()] {
                return Err(BackboneConstructionError::Alias {
                    alias: alias.into(),
                    owner: owner.as_str().into(),
                });
            }
            aliases.push(RecipeAlias::new(alias, owner.as_str()));
        }
    }
    for parameter in &topology {
        if parameter.alias_of.is_some() {
            recipes.remove(parameter.id.as_str());
        }
    }
    let inputs = recipes
        .values()
        .flat_map(|recipe| recipe.source_keys().into_iter().map(str::to_owned))
        .collect();
    let source: SharedCheckpointSource = Arc::new(RestrictedCheckpointSource::including(
        source,
        format!("voicechat/backbone/{unit:?}"),
        inputs,
    )?);
    let recipes = AtomicRecipeSet::new_with_aliases(source.as_ref(), recipes, aliases)?;
    let bindings = bindings_from_recipe_set(source.as_ref(), recipes)?;
    let bindings = eredu_runtime::apply_parameter_storage_layouts(module, bindings);
    Ok((source, bindings))
}
