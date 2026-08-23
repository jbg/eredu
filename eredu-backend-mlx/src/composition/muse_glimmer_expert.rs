//! MLX checkpoint and residency adapter for neutral Muse-Glimmer experts.

use std::collections::BTreeMap;

use eredu_architectures::muse_glimmer::{DecoderConfig, TransformerBlock};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
use eredu_runtime::{ExpertIdentity, OffloadUnit, ParameterRole, WeightBinding};
use safemlx::{module::ModuleParameters, Stream};

use crate::backend::mlx::{
    error::Error,
    nn::shared::{MlxModule, MlxNeuralBackend},
    runtime::{
        checkpoint::binding::{
            build_module_bindings_with_recipes_excluding, parameter_name_in_targets,
            parameter_role_targets,
        },
        residency::{
            expert_cache::{ExpertCache, ExpertCatalogEntry},
            expert_provider::CachedGatedProductExpertProvider,
        },
    },
};

/// Selects the architecture-owned released-layout recipes used by one module.
pub fn module_recipes<M: ModuleParameters>(
    module: &M,
    args: &DecoderConfig,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, Error> {
    let parameters = module.parameters().flatten();
    let mut recipes = eredu_architectures::muse_glimmer::safetensors_recipes(args, store)
        .map_err(Error::UnsupportedArchitecture)?;
    recipes.retain(|name, _| parameters.contains_key(name.as_str()));
    Ok(recipes)
}

pub fn expert_catalog(
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
        let block = TransformerBlock::<MlxNeuralBackend>::new(args, layer, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let module = MlxModule::new(block);
        let expert_targets = parameter_role_targets(
            &eredu_architectures::muse_glimmer::layer_parameter_groups(args, layer)?,
            ParameterRole::ExpertIntermediate,
        );
        let recipes = module_recipes(&module, args, store)?;
        let bank =
            build_module_bindings_with_recipes_excluding(&module, "", store, recipes, |name| {
                !parameter_name_in_targets(name, &expert_targets)
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

pub const fn cached_provider<'a>(
    cache: &'a ExpertCache,
    _args: &DecoderConfig,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}
