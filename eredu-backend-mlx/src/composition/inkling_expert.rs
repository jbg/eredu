//! MLX checkpoint and residency adapter for neutral Inkling expert banks.

use std::collections::BTreeMap;

use eredu_architectures::inkling::ModelArgs;
use eredu_checkpoint::recipe::DerivedWeightRecipe;
use safemlx::module::ModuleParameters;

use crate::backend::{
    error::Error,
    runtime::residency::{
        expert_cache::{ExpertCache, ExpertCatalogEntry},
        expert_provider::CachedGatedProductExpertProvider,
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
        .map_err(Error::ArchitectureModel)?;
    recipes.retain(|name, _| parameters.contains_key(name.as_str()));
    Ok(recipes)
}

/// Returns one independently leasable unit for every routed and shared expert.
pub fn expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = eredu_architectures::inkling::expert_residency_catalog(args, store)
        .map_err(Error::ArchitectureModel)?;
    super::architecture_expert_units(catalog, store, None)
}

pub const fn cached_provider<'a>(
    cache: &'a ExpertCache,
    _args: &ModelArgs,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}
