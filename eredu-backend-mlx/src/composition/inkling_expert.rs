//! MLX checkpoint and residency adapter for neutral Inkling expert banks.

use std::collections::{BTreeMap, BTreeSet};

use eredu_architectures::inkling::{DecoderLayer, FeedForwardPolicy, ModelArgs};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
use eredu_runtime::{ExpertIdentity, OffloadUnit, WeightBinding};
use safemlx::{module::ModuleParameters, Stream};

use crate::backend::mlx::{
    error::Error,
    nn::shared::{MlxBackend, MlxModule},
    runtime::{
        checkpoint::binding::build_module_bindings_with_recipes_excluding,
        residency::{
            expert_cache::{ExpertCache, ExpertCatalogEntry},
            expert_provider::{CachedGatedProductBankSpec, CachedGatedProductExpertProvider},
        },
    },
};

/// Builds the architecture-owned released-layout recipes applicable to one module.
pub fn module_recipes<M: ModuleParameters>(
    module: &M,
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let keys = store.source_keys().into_iter().collect::<BTreeSet<_>>();
    let parameters = module.parameters().flatten();
    let mut recipes = BTreeMap::new();
    for alias in eredu_architectures::inkling::safetensors_aliases(args)
        .map_err(Error::UnsupportedArchitecture)?
    {
        if parameters.contains_key(alias.target.as_str()) && keys.contains(&alias.source) {
            recipes.insert(
                alias.target,
                DerivedWeightRecipe::source(alias.source, TensorSelection::Full),
            );
        }
    }
    for layer in 0..args.text_config.num_hidden_layers as usize {
        for suffix in [
            "self_attn.k_sconv.weight",
            "self_attn.v_sconv.weight",
            "attn_sconv.weight",
            "mlp_sconv.weight",
        ] {
            let target = format!("model.layers.{layer}.{suffix}");
            let Some(parameter) = parameters.get(target.as_str()) else {
                continue;
            };
            if !keys.contains(&target) {
                continue;
            }
            let metadata = store.source_metadata(&target)?;
            if metadata.logical_shape.len() == 2 && parameter.shape().len() == 3 {
                let shape = parameter
                    .shape()
                    .iter()
                    .map(|&dimension| {
                        usize::try_from(dimension).map_err(|_| {
                            Error::UnsupportedArchitecture(format!(
                                "Inkling convolution target {target:?} has a negative dimension"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                recipes.insert(
                    target.clone(),
                    DerivedWeightRecipe::Reshape {
                        input: Box::new(DerivedWeightRecipe::source(target, TensorSelection::Full)),
                        shape,
                    },
                );
            }
        }
    }
    for layer in 0..args.text_config.num_hidden_layers as usize {
        let canonical = format!("model.layers.{layer}");
        let released = format!("model.llm.layers.{layer}");
        match args
            .text_config
            .layer_policy(layer)
            .map(|policy| policy.feed_forward)
        {
            Some(FeedForwardPolicy::Dense) => {
                let source = format!("{released}.mlp.w13_dn.weight");
                let gate_target = format!("{canonical}.dense.gate_proj.weight");
                let up_target = format!("{canonical}.dense.up_proj.weight");
                if !parameters.contains_key(gate_target.as_str())
                    && !parameters.contains_key(up_target.as_str())
                {
                    continue;
                }
                if keys.contains(&source) {
                    let split = eredu_architectures::inkling::dense_w13_recipes(store, &source)
                        .map_err(Error::UnsupportedArchitecture)?;
                    recipes.insert(gate_target, split.gate);
                    recipes.insert(up_target, split.up);
                }
            }
            Some(FeedForwardPolicy::SparseMoe) => {
                for (released_bank, canonical_bank) in [
                    ("mlp.experts.w13_weight", "moe.experts"),
                    ("mlp.shared_experts.shared_w13_weight", "moe.shared_experts"),
                ] {
                    let source = format!("{released}.{released_bank}");
                    let target = format!("{canonical}.{canonical_bank}.gate_up_proj");
                    if !parameters.contains_key(target.as_str()) {
                        continue;
                    }
                    if keys.contains(&source) {
                        recipes.insert(
                            target,
                            eredu_architectures::inkling::expert_w13_recipe(store, &source)
                                .map_err(Error::UnsupportedArchitecture)?,
                        );
                    } else {
                        let gate = format!("{canonical}.{canonical_bank}.gate_proj");
                        let up = format!("{canonical}.{canonical_bank}.up_proj");
                        if keys.contains(&gate) && keys.contains(&up) {
                            recipes.insert(
                                target,
                                DerivedWeightRecipe::Concatenate {
                                    axis: 1,
                                    inputs: vec![
                                        DerivedWeightRecipe::source(gate, TensorSelection::Full),
                                        DerivedWeightRecipe::source(up, TensorSelection::Full),
                                    ],
                                },
                            );
                        }
                    }
                }
            }
            None => {}
        }
    }
    if let Some(mtp) = &args.mtp_config {
        for depth in 0..usize::try_from(mtp.num_nextn_predict_layers).map_err(|_| {
            Error::UnsupportedArchitecture("Inkling MTP layer count is negative".into())
        })? {
            let root = format!("model.mtp.layers.{depth}.transformer_block");
            let source = format!("{root}.mlp.w13_dn.weight");
            let gate_target = format!("{root}.dense.gate_proj.weight");
            let up_target = format!("{root}.dense.up_proj.weight");
            if (!parameters.contains_key(gate_target.as_str())
                && !parameters.contains_key(up_target.as_str()))
                || !keys.contains(&source)
            {
                continue;
            }
            let split = eredu_architectures::inkling::dense_w13_recipes(store, &source)
                .map_err(Error::UnsupportedArchitecture)?;
            recipes.insert(gate_target, split.gate);
            recipes.insert(up_target, split.up);
        }
    }
    Ok(recipes)
}

/// Returns one independently leasable unit for every routed and shared expert.
pub fn expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    stream: &Stream,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let layers = args.text_config.num_hidden_layers as usize;
    let mut entries = Vec::new();
    for layer in 0..layers {
        if args
            .text_config
            .layer_policy(layer)
            .is_none_or(|policy| policy.feed_forward != FeedForwardPolicy::SparseMoe)
        {
            continue;
        }
        let block = DecoderLayer::<MlxBackend>::new(&args.text_config, layer, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let module = MlxModule::new(block);
        let recipes = module_recipes(&module, args, store)?;
        for (field, cache_layer, count) in [
            (".moe.experts.", layer, args.text_config.n_routed_experts),
            (
                ".moe.shared_experts.",
                layers + layer,
                args.text_config.n_shared_experts,
            ),
        ] {
            let bank = build_module_bindings_with_recipes_excluding(
                &module,
                "",
                store,
                recipes.clone(),
                |name| !name.contains(field),
            )?;
            for expert in 0..usize::try_from(count).map_err(|_| {
                Error::UnsupportedArchitecture("Inkling expert count is negative".into())
            })? {
                let selection = TensorSelection::Range {
                    axis: 0,
                    start: expert,
                    end: expert + 1,
                };
                let bindings = bank
                    .iter()
                    .map(|binding| {
                        let recipe = DerivedWeightRecipe::Select {
                            input: Box::new(binding.source_recipe()),
                            selection: selection.clone(),
                        };
                        let metadata = recipe.infer(store)?;
                        WeightBinding::from_recipe(
                            binding.name().rsplit('.').next().unwrap(),
                            recipe,
                            metadata.byte_len(),
                        )
                        .map_err(Into::into)
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                    total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                        Error::UnsupportedArchitecture("Inkling expert bytes overflowed".into())
                    })
                })?;
                let identity = ExpertIdentity::new(cache_layer, expert);
                entries.push(ExpertCatalogEntry::new(
                    identity,
                    OffloadUnit::new(identity.unit_id(), bindings)?,
                    bytes,
                )?);
            }
        }
    }
    if entries.is_empty() {
        return Err(Error::UnsupportedArchitecture(
            "independent expert caching requires sparse Inkling layers".into(),
        ));
    }
    Ok(entries)
}

pub fn cached_provider<'a>(
    cache: &'a ExpertCache,
    args: &'a ModelArgs,
) -> CachedGatedProductExpertProvider<'a, impl FnMut(usize) -> CachedGatedProductBankSpec + 'a> {
    CachedGatedProductExpertProvider::new(cache, move |cache_layer| {
        let layers = args.text_config.num_hidden_layers as usize;
        let (layer, bank) = if cache_layer < layers {
            (cache_layer, "experts")
        } else {
            (cache_layer - layers, "shared_experts")
        };
        let prefix = format!("model.layers.{layer}.moe.{bank}");
        CachedGatedProductBankSpec {
            hidden_dimensions: args.text_config.hidden_size,
            intermediate_dimensions: args.text_config.moe_intermediate_size(),
            gate_up_quantization: args
                .text_config
                .linear_format_for(&format!("{prefix}.gate_up_proj"))
                .weight_quantization(),
            down_quantization: args
                .text_config
                .linear_format_for(&format!("{prefix}.down_proj"))
                .weight_quantization(),
            gate_up_bias: false,
            down_bias: false,
            policy: eredu_nn::GatedProductPolicy::ordinary_silu(),
        }
    })
}
