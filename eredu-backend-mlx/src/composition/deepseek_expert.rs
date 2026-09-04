//! MLX expert-residency binding for the neutral DeepSeek architectures.

use std::collections::BTreeSet;

use eredu_architectures::deepseek::{self, V3Args, V4Args};

use crate::composition::grouped_provider::*;

use crate::backend::{
    error::Error,
    runtime::residency::parameter_bank::{AddressableParameterBank, ParameterBankEntry},
};

pub fn v3_catalog(
    args: &V3Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ParameterBankEntry>, Error> {
    v3_catalog_selected(args, store, |_, _| true)
}

/// Canonical V3 expert catalog for selected architecture-owned units.
pub fn v3_catalog_selected(
    args: &V3Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
) -> Result<Vec<ParameterBankEntry>, Error> {
    let catalog = deepseek::v3_expert_residency_catalog(store, args, None)
        .map_err(Error::ArchitectureModel)?;
    lower_selected_catalog(catalog, store, owns_unit)
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

pub fn v4_catalog(
    args: &V4Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ParameterBankEntry>, Error> {
    v4_catalog_selected(args, store, |_, _| true)
}

/// Canonical V4 expert catalog for selected architecture-owned units.
pub fn v4_catalog_selected(
    args: &V4Args,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
) -> Result<Vec<ParameterBankEntry>, Error> {
    let catalog = deepseek::v4_expert_residency_catalog(store, args, None)
        .map_err(Error::ArchitectureModel)?;
    lower_selected_catalog(catalog, store, owns_unit)
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

fn lower_selected_catalog(
    catalog: eredu_architectures::ExpertResidencyCatalog,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    owns_unit: impl FnMut(&eredu_runtime::ExecutionGroupId, usize) -> bool,
) -> Result<Vec<ParameterBankEntry>, Error> {
    let units = catalog.into_units_selected_by_owner(owns_unit);
    crate::composition::architecture_expert_units(units, store, None)
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
    cache: &'a AddressableParameterBank,
    _args: &V3Args,
) -> CachedGatedProductGroupProvider<'a> {
    CachedGatedProductGroupProvider::new(cache)
}

pub const fn v4_provider<'a>(
    cache: &'a AddressableParameterBank,
    _args: &V4Args,
) -> CachedGatedProductGroupProvider<'a> {
    CachedGatedProductGroupProvider::new(cache)
}
