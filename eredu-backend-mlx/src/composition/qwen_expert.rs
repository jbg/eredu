// MLX residency adapter for neutral Qwen routed-expert checkpoint contracts.

use eredu_architectures::qwen::ModelArgs;
use crate::backend::runtime::residency::expert_cache::ExpertCache;
use crate::backend::runtime::residency::expert_provider::CachedGatedProductExpertProvider;
use crate::backend::{
    error::Error, runtime::residency::expert_cache::ExpertCatalogEntry,
};

pub fn expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = eredu_architectures::qwen::expert_residency_catalog(store, args)
        .map_err(Error::ArchitectureModel)?;
    crate::composition::architecture_expert_units(catalog, store, None)
}

pub const fn cached_provider<'a>(
    cache: &'a ExpertCache,
    _args: &ModelArgs,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}
