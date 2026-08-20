//! MLX checkpoint and residency adapter for neutral Muse-Glimmer experts.

use std::collections::{BTreeMap, BTreeSet};

use eredu_architectures::muse_glimmer::{DecoderConfig, TransformerBlock};
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
            expert_provider::{CachedSwiGluBankSpec, CachedSwiGluExpertProvider},
        },
    },
};

fn existing(keys: &BTreeSet<String>, names: impl IntoIterator<Item = String>) -> Option<String> {
    names.into_iter().find(|name| keys.contains(name))
}

/// Builds released SafeTensors aliases and expert-layout recipes for one module.
pub(crate) fn module_recipes<M: ModuleParameters>(
    module: &M,
    args: &DecoderConfig,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let keys = store.source_keys().into_iter().collect::<BTreeSet<_>>();
    let parameters = module.parameters().flatten();
    let mut recipes = BTreeMap::new();
    for name in parameters.keys() {
        let released = name
            .replace("model.layers.", "model.language_model.layers.")
            .replace("model.embed_tokens.", "model.language_model.embed_tokens.")
            .replace("model.norm.", "model.language_model.norm.");
        let released = if let Some(prefix) = released.strip_suffix("_scales") {
            format!("{prefix}.scales")
        } else if let Some(prefix) = released.strip_suffix("_biases") {
            format!("{prefix}.biases")
        } else {
            released
        };
        if released != name.as_ref() && keys.contains(&released) {
            recipes.insert(
                name.to_string(),
                DerivedWeightRecipe::source(released, TensorSelection::Full),
            );
        }
    }
    if args.is_moe() {
        let experts = usize::try_from(args.num_experts)
            .map_err(|_| Error::UnsupportedArchitecture("invalid expert count".into()))?;
        for layer in 0..args.num_hidden_layers as usize {
            let target = format!("model.layers.{layer}.mlp.experts");
            if !parameters
                .keys()
                .any(|name| name.starts_with(&format!("{target}.")))
            {
                continue;
            }
            let released = format!("model.language_model.layers.{layer}.mlp.experts");
            let packed_gate_up = existing(
                &keys,
                [
                    format!("{target}.gate_up_proj"),
                    format!("{released}.gate_up_proj"),
                    format!("{released}.gate_up_proj.weight"),
                ],
            );
            let gate_up = if let Some(source) = packed_gate_up {
                Some(DerivedWeightRecipe::source(source, TensorSelection::Full))
            } else if let (Some(gate), Some(up)) = (
                existing(
                    &keys,
                    [
                        format!("{target}.gate_proj"),
                        format!("{target}.gate_proj.weight"),
                        format!("{released}.gate_proj"),
                        format!("{released}.gate_proj.weight"),
                    ],
                ),
                existing(
                    &keys,
                    [
                        format!("{target}.up_proj"),
                        format!("{target}.up_proj.weight"),
                        format!("{released}.up_proj"),
                        format!("{released}.up_proj.weight"),
                    ],
                ),
            ) {
                Some(DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        DerivedWeightRecipe::source(gate, TensorSelection::Full),
                        DerivedWeightRecipe::source(up, TensorSelection::Full),
                    ],
                })
            } else {
                let independent = (0..experts)
                    .map(|expert| {
                        let root = format!("{released}.{expert}");
                        let gate = format!("{root}.gate_proj.weight");
                        let up = format!("{root}.up_proj.weight");
                        (keys.contains(&gate) && keys.contains(&up)).then(|| {
                            DerivedWeightRecipe::Concatenate {
                                axis: 0,
                                inputs: vec![
                                    DerivedWeightRecipe::source(gate, TensorSelection::Full),
                                    DerivedWeightRecipe::source(up, TensorSelection::Full),
                                ],
                            }
                        })
                    })
                    .collect::<Option<Vec<_>>>();
                independent.map(|inputs| DerivedWeightRecipe::Stack { axis: 0, inputs })
            };
            if let Some(gate_up) = gate_up {
                recipes.insert(format!("{target}.gate_up_proj"), gate_up);
            }
            let down = existing(
                &keys,
                [
                    format!("{target}.down_proj"),
                    format!("{target}.down_proj.weight"),
                    format!("{released}.down_proj"),
                    format!("{released}.down_proj.weight"),
                ],
            )
            .map(|source| DerivedWeightRecipe::source(source, TensorSelection::Full))
            .or_else(|| {
                (0..experts)
                    .map(|expert| {
                        let source = format!("{released}.{expert}.down_proj.weight");
                        keys.contains(&source)
                            .then(|| DerivedWeightRecipe::source(source, TensorSelection::Full))
                    })
                    .collect::<Option<Vec<_>>>()
                    .map(|inputs| DerivedWeightRecipe::Stack { axis: 0, inputs })
            });
            if let Some(down) = down {
                recipes.insert(format!("{target}.down_proj"), down);
            }
        }
    }
    Ok(recipes)
}

pub(crate) fn expert_catalog(
    args: &DecoderConfig,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    stream: &Stream,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    if !args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "independent expert caching requires sparse Muse-Glimmer".into(),
        ));
    }
    let mut entries = Vec::new();
    for layer in 0..args.num_hidden_layers as usize {
        let block = TransformerBlock::<MlxBackend>::new(args, layer, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let module = MlxModule::new(block);
        let recipes = module_recipes(&module, args, store)?;
        let bank =
            build_module_bindings_with_recipes_excluding(&module, "", store, recipes, |name| {
                !name.contains(".mlp.experts.")
            })?;
        for expert in 0..args.num_experts as usize {
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
                    Error::UnsupportedArchitecture("Muse-Glimmer expert bytes overflowed".into())
                })
            })?;
            let identity = ExpertIdentity::new(layer, expert);
            entries.push(ExpertCatalogEntry::new(
                identity,
                OffloadUnit::new(identity.unit_id(), bindings)?,
                bytes,
            )?);
        }
    }
    Ok(entries)
}

pub(crate) fn cached_provider<'a>(
    cache: &'a ExpertCache,
    args: &'a DecoderConfig,
) -> CachedSwiGluExpertProvider<'a, impl FnMut(usize) -> CachedSwiGluBankSpec + 'a> {
    CachedSwiGluExpertProvider::new(cache, move |layer| {
        let prefix = format!("model.layers.{layer}.mlp.experts");
        CachedSwiGluBankSpec {
            hidden_dimensions: args.hidden_size,
            intermediate_dimensions: args.moe_intermediate_size,
            gate_up_quantization: args
                .linear_format_for(&format!("{prefix}.gate_up_proj"))
                .weight_quantization(),
            down_quantization: args
                .linear_format_for(&format!("{prefix}.down_proj"))
                .weight_quantization(),
            activation: eredu_nn::GatedExpertActivation::Silu,
            limit: None,
        }
    })
}
