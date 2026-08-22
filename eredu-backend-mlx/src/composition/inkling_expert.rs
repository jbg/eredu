//! MLX checkpoint and residency adapter for neutral Inkling expert banks.

use std::collections::BTreeMap;

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
            expert_provider::CachedGatedProductExpertProvider,
        },
    },
};

/// Selects the architecture-owned released-layout recipes used by one module.
pub fn module_recipes<M: ModuleParameters>(
    module: &M,
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let parameters = module.parameters().flatten();
    let mut recipes = eredu_architectures::inkling::safetensors_recipes(args, store)
        .map_err(Error::UnsupportedArchitecture)?;
    recipes.retain(|name, _| parameters.contains_key(name.as_str()));
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

pub const fn cached_provider<'a>(
    cache: &'a ExpertCache,
    _args: &ModelArgs,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}
