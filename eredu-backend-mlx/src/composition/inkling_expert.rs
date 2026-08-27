//! MLX checkpoint and residency adapter for neutral Inkling expert banks.

use eredu_architectures::inkling::ModelArgs;

use crate::backend::{
    error::Error,
    runtime::residency::{
        expert_cache::{ExpertCache, ExpertCatalogEntry},
        expert_provider::CachedGatedProductExpertProvider,
    },
};

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
