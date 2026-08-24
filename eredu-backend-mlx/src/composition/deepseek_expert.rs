//! MLX expert-residency binding for the neutral DeepSeek architectures.

use std::{collections::BTreeSet, ops::Range};

use eredu_architectures::deepseek::{self, V3Args, V4Args};
use eredu_nn::GatedProductExpertBankSpec;

use crate::backend::{
    error::Error,
    runtime::residency::{
        expert_cache::{ExpertCache, ExpertCatalogEntry},
        expert_provider::CachedGatedProductExpertProvider,
    },
};

pub fn v3_catalog(
    args: &V3Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = deepseek::v3_expert_residency_catalog(store, args, None)
        .map_err(Error::ArchitectureModel)?;
    crate::composition::architecture_expert_units(catalog, store, None)
}

/// Physical checkpoint keys owned by the independent V3 expert cache.
///
/// These come from the architecture-declared expert recipes rather than from
/// runtime parameter names, because packed checkpoints may realize one logical
/// projection from multiple source tensors.
pub fn v3_checkpoint_keys(
    args: &V3Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeSet<String>, Error> {
    let catalog = deepseek::v3_expert_residency_catalog(store, args, None)
        .map_err(Error::ArchitectureModel)?;
    Ok(checkpoint_keys(&catalog))
}

pub fn v3_parallel_catalog(
    args: &V3Args,
    intermediate: Range<usize>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = deepseek::v3_expert_residency_catalog(store, args, Some(intermediate))
        .map_err(Error::ArchitectureModel)?;
    crate::composition::architecture_expert_units(catalog, store, None)
}

pub fn v4_catalog(
    args: &V4Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = deepseek::v4_expert_residency_catalog(store, args, None)
        .map_err(Error::ArchitectureModel)?;
    crate::composition::architecture_expert_units(catalog, store, None)
}

/// Physical checkpoint keys owned by the independent V4 expert cache.
pub fn v4_checkpoint_keys(
    args: &V4Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<BTreeSet<String>, Error> {
    let catalog = deepseek::v4_expert_residency_catalog(store, args, None)
        .map_err(Error::ArchitectureModel)?;
    Ok(checkpoint_keys(&catalog))
}

pub fn v4_parallel_catalog(
    args: &V4Args,
    intermediate: Range<usize>,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = deepseek::v4_expert_residency_catalog(store, args, Some(intermediate))
        .map_err(Error::ArchitectureModel)?;
    crate::composition::architecture_expert_units(catalog, store, None)
}

fn checkpoint_keys(catalog: &eredu_architectures::ExpertResidencyCatalog) -> BTreeSet<String> {
    catalog
        .units()
        .iter()
        .flat_map(|unit| unit.parameters())
        .flat_map(|parameter| parameter.recipe().source_keys())
        .map(str::to_owned)
        .collect()
}

pub const fn v3_provider<'a>(
    cache: &'a ExpertCache,
    _args: &V3Args,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}

pub const fn v4_provider<'a>(
    cache: &'a ExpertCache,
    _args: &V4Args,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}

pub fn v3_spec(args: &V3Args, layer: usize) -> Result<GatedProductExpertBankSpec, Error> {
    let policy = deepseek::v3::moe_policy(args, layer)?;
    Ok(deepseek::moe::expert_bank_spec(&policy)?)
}

pub fn v4_spec(args: &V4Args, layer: usize) -> Result<GatedProductExpertBankSpec, Error> {
    Ok(deepseek::v4::expert_bank_spec(args, layer)?)
}
