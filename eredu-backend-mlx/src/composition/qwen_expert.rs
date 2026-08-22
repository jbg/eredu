// MLX residency adapter for neutral Qwen routed-expert checkpoint contracts.

use eredu_architectures::qwen::ModelArgs;
use eredu_checkpoint::recipe::DerivedWeightRecipe;
use eredu_runtime::ExpertPass;
use eredu_runtime::{
    ExpertIdentity, LayeredArchitecture, OffloadUnit, ParameterGroupOwner, ParameterRole,
    WeightBinding,
};
use safemlx::{Array, Stream};

use crate::backend::mlx::runtime::residency::expert_cache::ExpertCache;
use crate::backend::mlx::runtime::residency::expert_provider::{
    execute_cached_gated_product_dispatched, CachedGatedProductExpertProvider,
};
use crate::backend::mlx::{
    error::Error,
    runtime::{
        checkpoint::binding_plan::{BindingPlan, PlannedBinding},
        execution::layerwise::shard_layer_bindings,
        residency::expert_cache::ExpertCatalogEntry,
    },
};

pub fn expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    stream: &Stream,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    expert_catalog_cartesian(args, store, None, stream)
}

pub const fn cached_provider<'a>(
    cache: &'a ExpertCache,
    _args: &ModelArgs,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}

pub fn execute_cached_dispatched(
    cache: &ExpertCache,
    args: &ModelArgs,
    layer: usize,
    hidden: &Array,
    global_expert_ids: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    let spec = eredu_architectures::qwen::expert_bank_spec(args, layer)?;
    execute_cached_gated_product_dispatched(
        cache,
        &spec,
        layer,
        hidden,
        global_expert_ids,
        pass,
        stream,
    )
}

/// Builds expert-granular bindings with optional tensor-parallel selection.
pub fn expert_catalog_cartesian(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    stream: &Stream,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "Qwen expert catalog requires Qwen3-MoE arguments".into(),
        ));
    }
    let mut entries = Vec::new();
    let layers = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("Qwen layer count is negative".into()))?;
    let experts = usize::try_from(args.num_experts)
        .map_err(|_| Error::UnsupportedArchitecture("Qwen expert count is negative".into()))?;
    let architecture = eredu_architectures::qwen::LayeredModel::<
        crate::backend::mlx::nn::shared::MlxBackend,
    >::new(args.clone(), stream)
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let description = eredu_architectures::qwen::parameter_description(&architecture, stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let graph = <eredu_architectures::qwen::LayeredModel<
        crate::backend::mlx::nn::shared::MlxBackend,
    > as LayeredArchitecture<
        crate::backend::mlx::nn::shared::MlxBackend,
        super::MlxKeyValueState,
    >>::execution_graph(&architecture)
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    for layer in 0..layers {
        let mut owner_group = None;
        let expert_targets = description
            .groups()
            .iter()
            .filter(|owned| owned.role() == ParameterRole::ExpertIntermediate)
            .filter_map(|owned| match owned.owner() {
                ParameterGroupOwner::ExecutionUnit { group, global_unit }
                    if *global_unit == layer =>
                {
                    owner_group.get_or_insert(group.clone());
                    Some(owned.members())
                }
                _ => None,
            })
            .flatten()
            .map(|member| member.target().to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        let owner_group = owner_group.ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Qwen layer {layer} declares no expert-intermediate owner"
            ))
        })?;
        let group = graph
            .groups()
            .iter()
            .position(|candidate| candidate.id() == owner_group.as_str())
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Qwen expert owner group {owner_group:?} is absent from the execution graph"
                ))
            })?;
        let unit_path = <eredu_architectures::qwen::LayeredModel<
            crate::backend::mlx::nn::shared::MlxBackend,
        > as LayeredArchitecture<
            crate::backend::mlx::nn::shared::MlxBackend,
            super::MlxKeyValueState,
        >>::unit_path(&architecture, group, layer)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        for expert in 0..experts {
            let identity = ExpertIdentity::new(layer, expert);
            let mut bindings = eredu_architectures::qwen::expert_unit_recipes(
                store, args, layer, expert,
            )
            .map_err(Error::UnsupportedArchitecture)?
            .into_iter()
            .map(|(name, recipe)| {
                let candidates = expert_targets
                    .iter()
                    .filter(|target| {
                        target.as_str() == name
                            || target
                                .strip_suffix(name.as_str())
                                .is_some_and(|prefix| prefix.ends_with('.'))
                    })
                    .collect::<Vec<_>>();
                let [target] = candidates.as_slice() else {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "Qwen expert binding {name:?} resolves to {} architecture role targets",
                        candidates.len()
                    )));
                };
                recipe_binding(&name, target, recipe, store)
            })
            .collect::<Result<Vec<_>, _>>()?;
            if let Some(layout) = layout {
                bindings = shard_layer_bindings(bindings, &unit_path, store, layout)?;
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("Qwen expert byte total overflowed".into())
                })
            })?;
            let unit = OffloadUnit::new(identity.unit_id(), bindings)?;
            entries.push(ExpertCatalogEntry::new(identity, unit, bytes)?);
        }
    }
    Ok(entries)
}

fn recipe_binding(
    name: &str,
    logical_target: &str,
    recipe: DerivedWeightRecipe,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<WeightBinding, Error> {
    let metadata = recipe.infer(store)?;
    let mut bindings = BindingPlan::new(vec![PlannedBinding {
        target_name: name.into(),
        expected_shape: metadata.shape().to_vec(),
        expected_dtype: metadata.dtype().clone(),
        recipe,
    }])
    .and_then(|plan| plan.build_bindings(store))
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    Ok(bindings
        .pop()
        .expect("single planned expert binding")
        .with_logical_target(logical_target)?)
}
