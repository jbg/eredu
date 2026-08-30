//! Cold-path architecture/backend composition selected by public loaders.

use ref_cast::RefCast;
use safemlx::error::Exception;
use safemlx::Array;

use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use eredu_nn::Parameterized;
use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureParameters, LayeredArchitecture, RuntimeState,
    StaticParameterVisitor, StaticUnitBindings,
};

use crate::{backend::error::Error, backend::nn::shared::MlxNeuralBackend};

/// Derives prompt-cache identity for a complete replicated realization through
/// the architecture's neutral mutable-state contract.
pub(crate) fn replicated_prompt_cache_identity<A>(
    architecture: &A,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error>
where
    A: ArchitectureParameters<MlxNeuralBackend>,
    A::DefinitionError: std::fmt::Display,
{
    let layout = architecture
        .state_layout()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    eredu_runtime::PartitionState::new(layout, 0)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .prompt_cache_identity::<MlxNeuralBackend, A>(architecture, topology)
        .map_err(|error| match error {
            eredu_runtime::ArchitecturePartitionError::ArchitectureState(detail) => {
                Error::ArchitectureModel(detail)
            }
            eredu_runtime::ArchitecturePartitionError::PromptCacheIdentity(detail) => {
                Error::Parallel(detail)
            }
            error => Error::Parallel(error.to_string()),
        })
}

/// Adapts public MLX-array observation to the neutral tensor/error contract.
pub(crate) struct NeutralActivationObserver<'a> {
    inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
}

impl<'a> NeutralActivationObserver<'a> {
    pub(crate) fn new(
        inner: &'a mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Self {
        Self { inner }
    }
}

impl eredu_runtime::ActivationObserver<crate::MlxTensor, eredu_nn::Error>
    for NeutralActivationObserver<'_>
{
    fn observe(&mut self, path: &str, value: &crate::MlxTensor) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe(path, value.as_array())
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn intervene(
        &mut self,
        path: &str,
        value: &crate::MlxTensor,
    ) -> Result<Option<crate::MlxTensor>, eredu_nn::Error> {
        self.inner
            .intervene(path, value.as_array())
            .map(|value| value.map(crate::MlxTensor::from_array))
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }

    fn observe_routing(
        &mut self,
        routing: eredu_runtime::RoutingObservation<'_, crate::MlxTensor>,
    ) -> Result<(), eredu_nn::Error> {
        self.inner
            .observe_routing(eredu_runtime::RoutingObservation {
                path: routing.path,
                selected_experts: routing.selected_experts.as_array(),
                selected_scores: routing.selected_scores.as_array(),
                route_weights: routing.route_weights.as_array(),
                routed_output: routing.routed_output.as_array(),
                local_routed_output: routing.local_routed_output.map(crate::MlxTensor::as_array),
                reduced_routed_output: routing
                    .reduced_routed_output
                    .map(crate::MlxTensor::as_array),
                shared_output: routing.shared_output.map(crate::MlxTensor::as_array),
                combined_output: routing.combined_output.map(crate::MlxTensor::as_array),
                expert_count: routing.expert_count,
            })
            .map_err(|error| eredu_nn::Error::backend(error.to_string()))
    }
}

struct StaticBindingVisitor<'a> {
    store: &'a dyn CheckpointSource,
    recipes: std::collections::BTreeMap<String, DerivedWeightRecipe>,
    selected_roles: Option<std::collections::BTreeSet<String>>,
    units: Vec<StaticUnitBindings>,
}

impl StaticParameterVisitor<MlxNeuralBackend> for StaticBindingVisitor<'_> {
    type Error = Error;

    fn visit<M>(&mut self, role: &str, module: &M) -> Result<(), Self::Error>
    where
        M: Parameterized<crate::MlxTensor>,
    {
        if self
            .selected_roles
            .as_ref()
            .is_some_and(|selected| !selected.contains(role))
        {
            for name in crate::backend::nn::shared::neutral_parameter_refs(module, false)
                .flatten()
                .into_keys()
            {
                self.recipes.remove(name.as_ref());
            }
            return Ok(());
        }
        let bindings =
            crate::backend::runtime::checkpoint::binding::build_neutral_module_bindings_with_recipes(
                module,
                self.store,
                &mut self.recipes,
            )?;
        self.units.push(StaticUnitBindings::new(role, bindings)?);
        Ok(())
    }
}

pub(crate) fn architecture_static_units<A>(
    architecture: &A,
    store: &dyn CheckpointSource,
) -> Result<Vec<StaticUnitBindings>, Error>
where
    A: ArchitectureParameters<MlxNeuralBackend>,
{
    architecture_static_units_selected(architecture, store, None)
}

pub(crate) fn architecture_static_units_for_roles<A>(
    architecture: &A,
    store: &dyn CheckpointSource,
    roles: &[&str],
) -> Result<Vec<StaticUnitBindings>, Error>
where
    A: ArchitectureParameters<MlxNeuralBackend>,
{
    architecture_static_units_selected(architecture, store, Some(roles))
}

fn architecture_static_units_selected<A>(
    architecture: &A,
    store: &dyn CheckpointSource,
    roles: Option<&[&str]>,
) -> Result<Vec<StaticUnitBindings>, Error>
where
    A: ArchitectureParameters<MlxNeuralBackend>,
{
    let recipes = architecture
        .static_parameter_recipes(store)
        .map_err(Error::ArchitectureModel)?;
    let mut visitor = StaticBindingVisitor {
        store,
        recipes,
        selected_roles: roles.map(|roles| roles.iter().map(|role| (*role).to_owned()).collect()),
        units: Vec::new(),
    };
    architecture.visit_static_parameters(&mut visitor)?;
    if !visitor.recipes.is_empty() {
        return Err(Error::ArchitectureModel(format!(
            "architecture declared static recipes for unknown parameters {:?}",
            visitor.recipes.into_keys().collect::<Vec<_>>()
        )));
    }
    Ok(visitor.units)
}

/// Lowers the architecture-owned parameter topology into this rank's native
/// tensor-parallel layout.
pub(crate) fn parallel_layout_from_description(
    build: crate::backend::runtime::distributed::parallel::ParallelBuildContext,
    description: &eredu_runtime::ArchitectureParameterDescription,
) -> Result<eredu_runtime::LocalModelLayout, Error> {
    let mut planner = build.planner();
    for group in description.groups() {
        planner.register(group.group().clone())?;
    }
    planner.finish().map(|(_, layout)| layout)
}

/// Resolves the stable name of the sole execution group with a semantic role.
pub(crate) fn architecture_group_name<A, S>(
    architecture: &A,
    kind: ArchitectureGroupKind,
) -> Result<String, Error>
where
    A: LayeredArchitecture<MlxNeuralBackend, S>,
    S: RuntimeState<MlxNeuralBackend>,
    A::Error: std::fmt::Display,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    unique_group_name(&graph, kind, |group| {
        architecture.group_transport(group).kind
    })
}

fn unique_group_name(
    graph: &eredu_runtime::ExecutionGraph,
    kind: ArchitectureGroupKind,
    mut group_kind: impl FnMut(usize) -> ArchitectureGroupKind,
) -> Result<String, Error> {
    let mut groups = (0..graph.groups().len()).filter(|&group| group_kind(group) == kind);
    let group = groups.next().ok_or_else(|| {
        Error::ArchitectureModel(format!("architecture declares no {kind:?} execution group"))
    })?;
    if groups.next().is_some() {
        return Err(Error::ArchitectureModel(format!(
            "architecture declares multiple {kind:?} execution groups"
        )));
    }
    Ok(graph.groups()[group].id().to_owned())
}

/// Lowers already-selected neutral expert units into native cache entries.
///
/// Callers that realize only part of an architecture must consume neutral owner
/// and distribution policy with [`select_architecture_expert_units`] first.
pub(crate) fn architecture_expert_units(
    units: impl IntoIterator<Item = eredu_architectures::ExpertResidencyUnit>,
    store: &dyn CheckpointSource,
    layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<Vec<crate::backend::runtime::residency::expert_cache::ExpertCatalogEntry>, Error> {
    use crate::backend::runtime::residency::expert_cache::ExpertCatalogEntry;
    use eredu_runtime::{OffloadUnit, WeightBinding};

    units
        .into_iter()
        .map(|unit| {
            let identity = unit.identity();
            let mut bindings = unit
                .into_parameters()
                .into_iter()
                .map(|parameter| {
                    let (binding_name, logical_target, mut recipe, role) = parameter.into_parts();
                    if recipe.infer(store)?.dtype() == &eredu_checkpoint::recipe::RecipeDtype::F4 {
                        recipe = crate::backend::runtime::checkpoint::recipe::lower_mxfp4_recipe(
                            recipe, store,
                        )?;
                    }
                    let metadata = recipe.infer(store)?;
                    let mut binding =
                        WeightBinding::from_recipe(binding_name, recipe, metadata.byte_len())?
                            .with_logical_target(logical_target)?;
                    if let eredu_architectures::ExpertParameterRole::QuantizableProjection {
                        scales_binding,
                        biases_binding,
                    } = role
                    {
                        binding =
                            binding.with_quantization_companions(scales_binding, biases_binding)?;
                    }
                    Ok(binding)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            if let Some(layout) = layout {
                bindings = crate::backend::runtime::execution::layerwise::shard_layer_bindings(
                    bindings, store, layout,
                )?;
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::ArchitectureModel(format!("expert {identity:?} byte total overflowed"))
                })
            })?;
            Ok(ExpertCatalogEntry::new(
                identity,
                OffloadUnit::new(identity.unit_id(), bindings)?,
                bytes,
            )?)
        })
        .collect()
}

/// Retains expert units owned by one realized execution partition and expert rank.
///
/// Ownership and distribution are architecture addresses, so both predicates must
/// run while the neutral declarations are still available. Native cache entries
/// intentionally retain only router identity and materialization state.
pub(crate) fn select_architecture_expert_units(
    units: impl IntoIterator<Item = eredu_architectures::ExpertResidencyUnit>,
    mut owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
    mut owns_expert: impl FnMut(eredu_runtime::ExpertIdentity) -> bool,
) -> impl Iterator<Item = eredu_architectures::ExpertResidencyUnit> {
    units.into_iter().filter(move |unit| {
        owns_unit(unit.owner_group(), unit.owner_unit())
            && match unit.distribution() {
                eredu_architectures::ExpertResidencyDistribution::ExpertParallel => {
                    owns_expert(unit.identity())
                }
                eredu_architectures::ExpertResidencyDistribution::Replicated => true,
            }
    })
}

pub(crate) fn tensor_ref(array: &Array) -> &crate::MlxTensor {
    crate::MlxTensor::ref_cast(array)
}

pub(crate) fn tensor_opt(array: Option<&Array>) -> Option<&crate::MlxTensor> {
    array.map(tensor_ref)
}

pub mod deepseek;
pub mod deepseek_expert;
pub mod gemma4;
pub mod gemma4_expert;
#[cfg(any(feature = "image", feature = "audio"))]
pub mod gemma4_processor;
pub mod gpt_oss;
pub mod inkling;
pub mod inkling_expert;
#[cfg(any(feature = "image", feature = "audio"))]
pub mod inkling_processor;
pub mod kimi_linear;
// MLX adapter only; the neutral family is always available from
// `eredu_architectures::lfm2`.
pub mod lfm2;
pub mod llama;
pub mod mlx;
pub mod moshi;
pub mod muse_glimmer;
pub mod muse_glimmer_expert;
#[cfg(feature = "image")]
pub mod muse_glimmer_processor;
pub mod nemotron_h;
pub mod qwen;

#[cfg(test)]
#[path = "tests/mlx_architecture_conformance.rs"]
mod mlx_architecture_conformance;

#[cfg(test)]
mod expert_selection_tests {
    use eredu_architectures::{
        ExpertParameterRecipe, ExpertParameterRole, ExpertResidencyDistribution,
        ExpertResidencyUnit,
    };
    use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
    use eredu_runtime::{ExecutionGroupId, ExpertIdentity};

    fn expert_unit(
        identity: ExpertIdentity,
        group: &str,
        owner_unit: usize,
        distribution: ExpertResidencyDistribution,
    ) -> ExpertResidencyUnit {
        let parameter = ExpertParameterRecipe::new(
            "weight",
            format!("{group}.{owner_unit}.weight"),
            DerivedWeightRecipe::source("source", TensorSelection::Full),
            ExpertParameterRole::Preserved,
        )
        .unwrap();
        ExpertResidencyUnit::new(
            identity,
            ExecutionGroupId::new(group).unwrap(),
            owner_unit,
            format!("{group}.{owner_unit}"),
            distribution,
            [parameter],
        )
        .unwrap()
    }

    #[test]
    fn expert_selection_uses_owner_address_and_distribution_before_lowering() {
        let units = vec![
            expert_unit(
                ExpertIdentity::new(1, 0),
                "target",
                7,
                ExpertResidencyDistribution::ExpertParallel,
            ),
            expert_unit(
                ExpertIdentity::new(7, 1),
                "mtp.0",
                1,
                ExpertResidencyDistribution::ExpertParallel,
            ),
            expert_unit(
                ExpertIdentity::new(9, 2),
                "mtp.0",
                1,
                ExpertResidencyDistribution::Replicated,
            ),
        ];

        let selected = super::select_architecture_expert_units(
            units,
            |group, unit| group.as_str() == "mtp.0" && unit == 1,
            |identity| identity.global_expert == 1,
        )
        .map(|unit| unit.identity())
        .collect::<Vec<_>>();

        assert_eq!(
            selected,
            vec![ExpertIdentity::new(7, 1), ExpertIdentity::new(9, 2)]
        );
    }
}

#[cfg(test)]
mod architecture_group_tests {
    use eredu_runtime::{ArchitectureGroupKind, ExecutionGraph, ExecutionGroupSpec};

    #[test]
    fn semantic_group_lookup_preserves_architecture_declared_name() {
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("decoder-renamed"),
                ExecutionGroupSpec::with_dependencies("prediction", ["decoder-renamed"]),
            ],
            "prediction",
        )
        .unwrap();

        let name = super::unique_group_name(&graph, ArchitectureGroupKind::Decoder, |group| {
            if group == 0 {
                ArchitectureGroupKind::Decoder
            } else {
                ArchitectureGroupKind::Prediction
            }
        })
        .unwrap();

        assert_eq!(name, "decoder-renamed");
    }
}
