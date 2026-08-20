//! MLX expert-residency binding for the neutral DeepSeek architectures.

use std::ops::Range;

use eredu_architectures::deepseek::{self, LayerPolicy, V3Args, V4Args};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
use eredu_runtime::{ExpertIdentity, OffloadUnit, WeightBinding};

use crate::backend::mlx::{
    error::Error,
    runtime::{
        checkpoint::binding_plan::{BindingPlan, PlannedBinding},
        residency::{
            expert_cache::{ExpertCache, ExpertCatalogEntry},
            expert_provider::{CachedSwiGluBankSpec, CachedSwiGluExpertProvider},
        },
    },
};

pub(crate) fn v3_catalog(
    args: &V3Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut entries = Vec::new();
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V3 layer count".into()))?;
    let total = target
        + usize::try_from(args.num_nextn_predict_layers)
            .map_err(|_| Error::UnsupportedArchitecture("invalid V3 prediction count".into()))?;
    let experts = usize::try_from(args.n_routed_experts)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V3 expert count".into()))?;
    for layer in 0..total {
        if layer < target && args.layer_schedule.get(layer) != Some(&LayerPolicy::SparseMoe) {
            continue;
        }
        let recipes = deepseek::v3_expert_recipes(store, args, layer)
            .map_err(Error::UnsupportedArchitecture)?;
        append_entries(
            &mut entries,
            layer,
            experts,
            recipes.gate_up,
            recipes.down,
            None,
            store,
        )?;
    }
    Ok(entries)
}

pub(crate) fn v3_parallel_catalog(
    args: &V3Args,
    intermediate: Range<usize>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    v3_catalog_with_intermediate(args, Some(intermediate), store)
}

fn v3_catalog_with_intermediate(
    args: &V3Args,
    intermediate: Option<Range<usize>>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let mut entries = Vec::new();
    let target = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V3 layer count".into()))?;
    let total = target
        + usize::try_from(args.num_nextn_predict_layers)
            .map_err(|_| Error::UnsupportedArchitecture("invalid V3 prediction count".into()))?;
    let experts = usize::try_from(args.n_routed_experts)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V3 expert count".into()))?;
    for layer in 0..total {
        if layer < target && args.layer_schedule.get(layer) != Some(&LayerPolicy::SparseMoe) {
            continue;
        }
        let recipes = deepseek::v3_expert_recipes(store, args, layer)
            .map_err(Error::UnsupportedArchitecture)?;
        append_entries(
            &mut entries,
            layer,
            experts,
            recipes.gate_up,
            recipes.down,
            intermediate.clone(),
            store,
        )?;
    }
    Ok(entries)
}

pub(crate) fn v4_catalog(
    args: &V4Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V4 layer count".into()))?;
    let experts = usize::try_from(args.n_routed_experts)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V4 expert count".into()))?;
    let mut entries = Vec::new();
    for layer in 0..total {
        let recipes = deepseek::v4_expert_recipes(store, args, layer)
            .map_err(Error::UnsupportedArchitecture)?;
        append_entries(
            &mut entries,
            layer,
            experts,
            recipes.gate_up,
            recipes.down,
            None,
            store,
        )?;
    }
    Ok(entries)
}

pub(crate) fn v4_parallel_catalog(
    args: &V4Args,
    intermediate: Range<usize>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V4 layer count".into()))?;
    let experts = usize::try_from(args.n_routed_experts)
        .map_err(|_| Error::UnsupportedArchitecture("invalid V4 expert count".into()))?;
    let mut entries = Vec::new();
    for layer in 0..total {
        let recipes = deepseek::v4_expert_recipes(store, args, layer)
            .map_err(Error::UnsupportedArchitecture)?;
        append_entries(
            &mut entries,
            layer,
            experts,
            recipes.gate_up,
            recipes.down,
            Some(intermediate.clone()),
            store,
        )?;
    }
    Ok(entries)
}

fn append_entries(
    entries: &mut Vec<ExpertCatalogEntry>,
    layer: usize,
    experts: usize,
    gate_up: DerivedWeightRecipe,
    down: DerivedWeightRecipe,
    intermediate: Option<Range<usize>>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<(), Error> {
    for expert in 0..experts {
        let selection = TensorSelection::Range {
            axis: 0,
            start: expert,
            end: expert + 1,
        };
        let mut gate_up = gate_up.clone().select_bounded(store, selection.clone())?;
        let mut down = down.clone().select_bounded(store, selection)?;
        if let Some(intermediate) = &intermediate {
            let metadata = gate_up.infer(store)?;
            let fused = *metadata.shape().get(1).ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "DeepSeek expert gate/up recipe is not rank three".into(),
                )
            })?;
            if fused % 2 != 0
                || intermediate.start >= intermediate.end
                || intermediate.end > fused / 2
            {
                return Err(Error::UnsupportedArchitecture(format!(
                    "DeepSeek expert intermediate range {intermediate:?} is outside 0..{}",
                    fused / 2
                )));
            }
            let width = fused / 2;
            gate_up = DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![
                    gate_up.clone().select_bounded(
                        store,
                        TensorSelection::Range {
                            axis: 1,
                            start: intermediate.start,
                            end: intermediate.end,
                        },
                    )?,
                    gate_up.select_bounded(
                        store,
                        TensorSelection::Range {
                            axis: 1,
                            start: width + intermediate.start,
                            end: width + intermediate.end,
                        },
                    )?,
                ],
            };
            down = down.select_bounded(
                store,
                TensorSelection::Range {
                    axis: 2,
                    start: intermediate.start,
                    end: intermediate.end,
                },
            )?;
        }
        let bindings = vec![
            recipe_binding("gate_up_proj", gate_up, store)?,
            recipe_binding("down_proj", down, store)?,
        ];
        let bytes = bindings.iter().try_fold(0u64, |total, binding| {
            total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                Error::UnsupportedArchitecture("DeepSeek expert byte total overflowed".into())
            })
        })?;
        let identity = ExpertIdentity::new(layer, expert);
        entries.push(ExpertCatalogEntry::new(
            identity,
            OffloadUnit::new(identity.unit_id(), bindings)?,
            bytes,
        )?);
    }
    Ok(())
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
    Ok(bindings.pop().expect("one expert binding"))
}

pub(crate) fn v3_provider<'a>(
    cache: &'a ExpertCache,
    args: &'a V3Args,
) -> CachedSwiGluExpertProvider<'a, impl FnMut(usize) -> CachedSwiGluBankSpec + 'a> {
    CachedSwiGluExpertProvider::new(cache, move |layer| v3_spec(args, layer))
}

pub(crate) fn v4_provider<'a>(
    cache: &'a ExpertCache,
    args: &'a V4Args,
) -> CachedSwiGluExpertProvider<'a, impl FnMut(usize) -> CachedSwiGluBankSpec + 'a> {
    CachedSwiGluExpertProvider::new(cache, move |layer| v4_spec(args, layer))
}

pub(crate) fn v3_spec(args: &V3Args, layer: usize) -> CachedSwiGluBankSpec {
    let root = format!("model.layers.{layer}.mlp.experts");
    CachedSwiGluBankSpec {
        hidden_dimensions: args.hidden_size,
        intermediate_dimensions: args.moe_intermediate_size,
        gate_up_quantization: args
            .linear_format_for(&format!("{root}.gate_up_proj"))
            .weight_quantization(),
        down_quantization: args
            .linear_format_for(&format!("{root}.down_proj"))
            .weight_quantization(),
        limit: None,
    }
}

pub(crate) fn v4_spec(args: &V4Args, layer: usize) -> CachedSwiGluBankSpec {
    let root = if layer < args.num_hidden_layers as usize {
        format!("layers.{layer}.ffn.switch_mlp")
    } else {
        format!(
            "mtp.{}.ffn.switch_mlp",
            layer - args.num_hidden_layers as usize
        )
    };
    CachedSwiGluBankSpec {
        hidden_dimensions: args.hidden_size,
        intermediate_dimensions: args.moe_intermediate_size,
        gate_up_quantization: args
            .linear_format_for(&format!("{root}.gate_up_proj"))
            .weight_quantization(),
        down_quantization: args
            .linear_format_for(&format!("{root}.down_proj"))
            .weight_quantization(),
        limit: args.swiglu_limit.map(|limit| limit.get()),
    }
}
