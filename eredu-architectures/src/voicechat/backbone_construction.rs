//! Exact retained binding plans for the ordinary Nemotron runtime.

use eredu_core::BackendFailure;
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, GroupedNeuralBackend, Parameterized, Tensor,
};
use eredu_runtime::{
    bind_materialized_unit, materialize_selected_bindings, select_bindings, ExecutionUnitLayout,
    LayerRuntimeState, ParameterBackend, PreparedComponent, ResidentRuntime,
    RuntimeStateComponents, SelectedBindingPlan, WeightBinding,
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
/// Cold preparation validates the Nemotron parameter declaration against retained
/// recipes without native allocation or payload reads. The source exposes only
/// physical inputs used by the target; pinned vocabulary weights keep their
/// canonical owners alongside independently leased block bindings.
pub struct PreparedBackboneBindings {
    prepared: PreparedComponent,
}

impl PreparedBackboneBindings {
    /// Exact retained physical inputs, including source guards and provenance.
    pub fn source(&self) -> &SharedCheckpointSource {
        self.prepared.source()
    }
    /// Existing Nemotron target traversal in canonical block order.
    pub fn layout(&self) -> &ExecutionUnitLayout {
        self.prepared.unit_layout()
    }
    /// Embedding, final normalization and agent readout bindings.
    pub fn static_bindings(&self) -> impl Iterator<Item = &WeightBinding> {
        self.prepared.pinned_bindings().values().flatten()
    }
    /// Independently loadable target block bindings.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        self.prepared.unit_bindings()
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
        let (_, layout, source, units, pinned) = self.prepared.into_parts();
        let mut statics = pinned.into_values().flatten().collect::<Vec<_>>();
        statics.sort_by(|left, right| left.name().cmp(right.name()));
        (source, layout, statics, units)
    }
}

impl PreparedBackboneParameters {
    /// Prepares exact target resources from the source-free Nemotron declaration.
    /// No native module, tensor, device or checkpoint payload is required.
    pub fn prepare_bindings(&self) -> Result<PreparedBackboneBindings, BackbonePreparationError> {
        let description = nemotron_h::parameter_description(&self.args)
            .map_err(|error| BackbonePreparationError::Configuration(error.to_string()))?;
        let prepared = PreparedComponent::prepare(
            "voicechat/backbone/target",
            description,
            self.source.clone(),
            |member| {
                self.parameters
                    .get(member.target())
                    .map(|parameter| parameter.recipe.clone())
                    .ok_or_else(|| {
                        BackbonePreparationError::Configuration(format!(
                            "missing backbone parameter {}",
                            member.target()
                        ))
                    })
            },
        )?;
        Ok(PreparedBackboneBindings { prepared })
    }

    /// Constructs the ordinary model and applies native storage preferences to cold bindings.
    pub fn prepare_layerwise<B, S>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(nemotron_h::LayeredModel<B>, PreparedBackboneBindings), BackboneConstructionError>
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let bindings = self.prepare_bindings()?;
        let model = nemotron_h::CompleteNemotronKernel::<B>::new(self.args.clone(), context)?;
        let prepared = bindings
            .prepared
            .with_storage_preferences::<B, S, _>(&model, context)?;
        Ok((model, PreparedBackboneBindings { prepared }))
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
        let prepared = self.prepare_bindings()?.prepared;
        let static_source = prepared
            .pinned_source()
            .map_err(BackbonePreparationError::from)?;
        let layer_sources = (0..prepared.unit_layout().len())
            .map(|ordinal| {
                prepared.unit_source(
                    prepared
                        .unit_layout()
                        .address(ordinal)
                        .expect("declared unit"),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(BackbonePreparationError::from)?;
        let (_, _, statics, units) = PreparedBackboneBindings { prepared }.into_parts();
        let statics = eredu_runtime::apply_parameter_storage_layouts(
            runtime.architecture().kernel().static_modules(),
            statics,
        );
        let static_plan = select(BackboneUnit::Static, static_source, statics)?;
        let layers = units
            .into_iter()
            .zip(layer_sources)
            .zip(&runtime.units()[0])
            .enumerate()
            .map(|(index, ((bindings, source), unit))| {
                let bindings = eredu_runtime::apply_parameter_storage_layouts(unit, bindings);
                select(BackboneUnit::Layer(index), source, bindings)
            })
            .collect::<Result<Vec<_>, _>>()?;
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
