//! MLX residency adapter for neutral Qwen routed-expert checkpoint contracts.

use std::collections::BTreeSet;

use eredu_architectures::qwen::ModelArgs;
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
use eredu_runtime::ExpertPass;
use eredu_runtime::{ExpertIdentity, OffloadUnit, WeightBinding};
use safemlx::{Array, Stream};

use crate::backend::mlx::runtime::residency::expert_cache::ExpertCache;
use crate::backend::mlx::runtime::residency::expert_provider::{
    execute_cached_swiglu, CachedSwiGluBankSpec, CachedSwiGluExpertProvider,
};
use crate::backend::mlx::{
    error::Error,
    runtime::{
        checkpoint::binding_plan::{BindingPlan, PlannedBinding},
        execution::layerwise::shard_layer_bindings,
        residency::expert_cache::ExpertCatalogEntry,
    },
};

pub(crate) fn expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    expert_catalog_cartesian(args, store, None)
}

pub(crate) fn cached_provider<'a>(
    cache: &'a ExpertCache,
    args: &'a ModelArgs,
) -> CachedSwiGluExpertProvider<'a, impl FnMut(usize) -> CachedSwiGluBankSpec + 'a> {
    CachedSwiGluExpertProvider::new(cache, move |layer| cached_bank_spec(args, layer))
}

fn cached_bank_spec(args: &ModelArgs, layer: usize) -> CachedSwiGluBankSpec {
    let prefix = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
    CachedSwiGluBankSpec {
        hidden_dimensions: args.hidden_size,
        intermediate_dimensions: args.moe_intermediate_size,
        gate_up_quantization: args.weight_quantization_for(&format!("{prefix}.gate_up_proj")),
        down_quantization: args.weight_quantization_for(&format!("{prefix}.down_proj")),
        activation: eredu_nn::GatedExpertActivation::Silu,
        limit: None,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_cached(
    cache: &ExpertCache,
    args: &ModelArgs,
    layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    route_weights: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    execute_cached_swiglu(
        cache,
        cached_bank_spec(args, layer),
        layer,
        hidden,
        expert_ids,
        route_weights,
        pass,
        stream,
    )
}

/// Builds expert-granular bindings with optional tensor-parallel selection.
pub(crate) fn expert_catalog_cartesian(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "Qwen expert catalog requires Qwen3-MoE arguments".into(),
        ));
    }
    let keys = store.source_keys().into_iter().collect::<BTreeSet<_>>();
    let mut entries = Vec::new();
    let layers = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("Qwen layer count is negative".into()))?;
    let experts = usize::try_from(args.num_experts)
        .map_err(|_| Error::UnsupportedArchitecture("Qwen expert count is negative".into()))?;
    for layer in 0..layers {
        let prefix = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
        let packed_gate_up = format!("{prefix}.gate_up_proj");
        let packed_down = format!("{prefix}.down_proj");
        for expert in 0..experts {
            let identity = ExpertIdentity::new(layer, expert);
            let selection = TensorSelection::Range {
                axis: 0,
                start: expert,
                end: expert + 1,
            };
            let mut bindings = Vec::new();
            if keys.contains(&packed_gate_up) && keys.contains(&packed_down) {
                for (name, key) in [
                    ("gate_up_proj", packed_gate_up.clone()),
                    ("down_proj", packed_down.clone()),
                ] {
                    bindings.push(recipe_binding(
                        name,
                        DerivedWeightRecipe::source(key, selection.clone()),
                        store,
                    )?);
                }
                for (name, key) in [
                    ("gate_up_proj_scales", format!("{packed_gate_up}_scales")),
                    ("gate_up_proj_biases", format!("{packed_gate_up}_biases")),
                    ("down_proj_scales", format!("{packed_down}_scales")),
                    ("down_proj_biases", format!("{packed_down}_biases")),
                ] {
                    if keys.contains(&key) {
                        bindings.push(recipe_binding(
                            name,
                            DerivedWeightRecipe::source(key, selection.clone()),
                            store,
                        )?);
                    }
                }
            } else if keys.contains(&format!("{prefix}.gate_proj"))
                && keys.contains(&format!("{prefix}.up_proj"))
                && keys.contains(&packed_down)
            {
                bindings.push(recipe_binding(
                    "gate_up_proj",
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(
                                format!("{prefix}.gate_proj"),
                                selection.clone(),
                            ),
                            DerivedWeightRecipe::source(
                                format!("{prefix}.up_proj"),
                                selection.clone(),
                            ),
                        ],
                    },
                    store,
                )?);
                bindings.push(recipe_binding(
                    "down_proj",
                    DerivedWeightRecipe::source(packed_down.clone(), selection.clone()),
                    store,
                )?);
                for suffix in ["_scales", "_biases"] {
                    let gate = format!("{prefix}.gate_proj{suffix}");
                    let up = format!("{prefix}.up_proj{suffix}");
                    if keys.contains(&gate) && keys.contains(&up) {
                        bindings.push(recipe_binding(
                            &format!("gate_up_proj{suffix}"),
                            DerivedWeightRecipe::Concatenate {
                                axis: 1,
                                inputs: vec![
                                    DerivedWeightRecipe::source(gate, selection.clone()),
                                    DerivedWeightRecipe::source(up, selection.clone()),
                                ],
                            },
                            store,
                        )?);
                    }
                    let down = format!("{packed_down}{suffix}");
                    if keys.contains(&down) {
                        bindings.push(recipe_binding(
                            &format!("down_proj{suffix}"),
                            DerivedWeightRecipe::source(down, selection.clone()),
                            store,
                        )?);
                    }
                }
            } else {
                if args.weight_quantization_for(&packed_gate_up).is_some()
                    || args.weight_quantization_for(&packed_down).is_some()
                {
                    return Err(Error::Quantization(
                        "split Qwen experts cannot be lazily load-time quantized; use checkpoint-native packed expert weights"
                            .into(),
                    ));
                }
                let gate = split_expert_key(&keys, &prefix, expert, &["gate_proj", "w1"])?;
                let up = split_expert_key(&keys, &prefix, expert, &["up_proj", "w3"])?;
                let down = split_expert_key(&keys, &prefix, expert, &["down_proj", "w2"])?;
                bindings.push(recipe_binding(
                    "gate_up_proj",
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![DerivedWeightRecipe::Concatenate {
                            axis: 0,
                            inputs: vec![
                                DerivedWeightRecipe::source(gate, TensorSelection::Full),
                                DerivedWeightRecipe::source(up, TensorSelection::Full),
                            ],
                        }],
                    },
                    store,
                )?);
                bindings.push(recipe_binding(
                    "down_proj",
                    DerivedWeightRecipe::Stack {
                        axis: 0,
                        inputs: vec![DerivedWeightRecipe::source(down, TensorSelection::Full)],
                    },
                    store,
                )?);
            }
            if let Some(layout) = layout {
                bindings = shard_layer_bindings(bindings, &prefix, store, layout)?;
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
    Ok(bindings.pop().expect("single planned expert binding"))
}

fn split_expert_key(
    keys: &BTreeSet<String>,
    prefix: &str,
    expert: usize,
    projections: &[&str],
) -> Result<String, Error> {
    projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.weight"))
        .find(|key| keys.contains(key))
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Qwen checkpoint is missing split expert {expert} projection {projections:?}"
            ))
        })
}
