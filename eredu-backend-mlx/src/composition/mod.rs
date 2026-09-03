//! Cold-path architecture/backend composition selected by public loaders.

pub(crate) mod grouped_provider;
// Multi-process adapters are also entry points for manually launched native validation.
#[allow(dead_code)]
pub(crate) mod expert_dispatch;

use ref_cast::RefCast;
use safemlx::error::Exception;
use safemlx::Array;

use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use eredu_nn::Parameterized;
use eredu_runtime::{ArchitectureParameters, StaticParameterVisitor, StaticUnitBindings};

use crate::{backend::error::Error, backend::nn::shared::MlxNeuralBackend};

impl From<eredu_runtime::ParameterBankLoadOptions>
    for crate::backend::runtime::residency::parameter_bank::ParameterBankOptions
{
    fn from(options: eredu_runtime::ParameterBankLoadOptions) -> Self {
        Self::new(
            options.offload(),
            options.compact_bank_scratch_bytes(),
            options.prefill_compact_bank_target_bytes(),
        )
        .expect("architecture-owned expert-cache options were already validated")
    }
}

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
                coefficients: routing.coefficients.as_array(),
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
    build: crate::composition::mlx::distributed::topology::ParallelBuildContext,
    description: &eredu_runtime::ArchitectureParameterDescription,
) -> Result<eredu_runtime::LocalModelLayout, Error> {
    let mut planner = build.planner();
    for group in description.groups() {
        planner.register(group.group().clone())?;
    }
    planner.finish().map(|(_, layout)| layout)
}

/// Lowers already-selected neutral expert units into native cache entries.
///
/// Callers that realize only part of an architecture must consume neutral owner
/// and distribution policy with [`select_architecture_expert_units`] first.
pub(crate) fn architecture_expert_units(
    units: impl IntoIterator<Item = eredu_architectures::ExpertResidencyUnit>,
    store: &dyn CheckpointSource,
    layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<Vec<crate::backend::runtime::residency::parameter_bank::ParameterBankEntry>, Error> {
    use crate::backend::runtime::residency::parameter_bank::ParameterBankEntry;
    use eredu_runtime::{OffloadUnit, WeightBinding};

    units
        .into_iter()
        .map(|unit| {
            let identity = unit.identity();
            let mut bindings = unit
                .into_parameters()
                .into_iter()
                .map(|parameter| {
                    let mut parameter = parameter.into_artifact();
                    let binding_name = parameter.take_binding_name();
                    let logical_target = parameter.take_logical_target();
                    let mut recipe = parameter.take_recipe();
                    let role = parameter.take_role();
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
            let key = crate::backend::runtime::residency::parameter_bank::ParameterBankKey::new(
                identity.unit(),
                identity.member(),
            );
            Ok(ParameterBankEntry::new(
                key,
                OffloadUnit::new(key.unit_id(), bindings)?,
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
    mut owns_expert: impl FnMut(eredu_runtime::ParameterBankKey) -> bool,
) -> impl Iterator<Item = eredu_architectures::ExpertResidencyUnit> {
    units.into_iter().filter(move |unit| {
        owns_unit(unit.owner_group(), unit.owner_unit())
            && match unit.distribution() {
                eredu_architectures::ExpertResidencyDistribution::ExpertParallel => {
                    owns_expert(unit.identity())
                }
                eredu_architectures::ExpertResidencyDistribution::Replicated => true,
                _ => false,
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
pub mod gpt_oss;
pub mod inkling;
pub mod inkling_expert;
pub mod kimi_linear;
// MLX adapter only; the neutral family is always available from
// `eredu_architectures::lfm2`.
pub mod lfm2;
pub mod llama;
pub mod mlx;
pub mod moshi;
pub mod muse_glimmer;
pub mod muse_glimmer_expert;
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
    use eredu_runtime::{ExecutionGroupId, ParameterBankKey};

    fn expert_unit(
        identity: ParameterBankKey,
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
                ParameterBankKey::new(1, 0),
                "target",
                7,
                ExpertResidencyDistribution::ExpertParallel,
            ),
            expert_unit(
                ParameterBankKey::new(7, 1),
                "mtp.0",
                1,
                ExpertResidencyDistribution::ExpertParallel,
            ),
            expert_unit(
                ParameterBankKey::new(9, 2),
                "mtp.0",
                1,
                ExpertResidencyDistribution::Replicated,
            ),
        ];

        let selected = super::select_architecture_expert_units(
            units,
            |group, unit| group.as_str() == "mtp.0" && unit == 1,
            |identity| identity.member() == 1,
        )
        .map(|unit| unit.identity())
        .collect::<Vec<_>>();

        assert_eq!(
            selected,
            vec![ParameterBankKey::new(7, 1), ParameterBankKey::new(9, 2)]
        );
    }
}
