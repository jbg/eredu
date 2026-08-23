//! MLX residency adapter for neutral Gemma 4 routed experts.

use std::collections::BTreeMap;

use eredu_architectures::gemma4::{DenseBlock, ModelArgs};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
use eredu_runtime::{ExpertIdentity, OffloadUnit, ParameterRole, WeightBinding};
use safemlx::Stream;

use crate::backend::{
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

/// Returns one independently leasable cache unit for every routed expert.
pub fn expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    stream: &Stream,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let expert_count = usize::try_from(args.num_experts.ok_or_else(|| {
        Error::UnsupportedArchitecture("Gemma 4 MoE config has no expert count".into())
    })?)
    .map_err(|_| Error::UnsupportedArchitecture("Gemma 4 expert count is negative".into()))?;
    let mut entries = Vec::new();
    for layer in 0..args.num_hidden_layers() {
        if args.layer_policy(layer).is_none_or(|policy| {
            policy.feed_forward
                != eredu_architectures::gemma4::FeedForwardPolicy::DenseWithSparseMoe
        }) {
            continue;
        }
        let resolved = eredu_architectures::gemma4::expert_recipes(
            store,
            args,
            "model.language_model.layers",
            layer,
        )
        .map_err(Error::UnsupportedArchitecture)?;
        let recipes = BTreeMap::from([
            (resolved.target_gate_up, resolved.gate_up),
            (resolved.target_down, resolved.down),
        ]);
        let block = DenseBlock::<MlxNeuralBackend>::new(args, layer, stream)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
        let expert_targets = parameter_role_targets(
            &eredu_architectures::gemma4::layer_parameter_groups(args, layer)?,
            ParameterRole::ExpertIntermediate,
        );
        let bank = build_module_bindings_with_recipes_excluding(
            &MlxModule::new(block),
            "",
            store,
            recipes,
            |name| !parameter_name_in_targets(name, &expert_targets),
        )?;
        if bank.is_empty() {
            return Err(Error::UnsupportedArchitecture(format!(
                "Gemma 4 sparse layer {layer} produced no expert bindings"
            )));
        }
        for expert in 0..expert_count {
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
                    let name = binding
                        .name()
                        .rsplit('.')
                        .next()
                        .expect("validated parameter name");
                    WeightBinding::from_recipe(name, recipe, metadata.byte_len())
                        .map_err(Into::into)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total.checked_add(binding.expected_bytes()).ok_or_else(|| {
                    Error::UnsupportedArchitecture("Gemma 4 expert byte total overflowed".into())
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
    if entries.is_empty() {
        return Err(Error::UnsupportedArchitecture(
            "independent expert caching requires sparse Gemma 4 layers".into(),
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
