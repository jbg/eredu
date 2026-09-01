// MLX residency adapter for neutral Qwen routed-expert checkpoint contracts.

use eredu_architectures::qwen::ModelArgs;
use crate::backend::runtime::residency::parameter_bank::AddressableParameterBank;
use crate::composition::grouped_provider::CachedGatedProductGroupProvider;
use crate::backend::{
    error::Error, runtime::residency::parameter_bank::ParameterBankEntry,
};

pub fn expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ParameterBankEntry>, Error> {
    let catalog = eredu_architectures::qwen::expert_residency_catalog(store, args)
        .map_err(Error::ArchitectureModel)?;
    crate::composition::architecture_expert_units(catalog, store, None)
}

pub const fn cached_provider<'a>(
    cache: &'a AddressableParameterBank,
    _args: &ModelArgs,
) -> CachedGatedProductGroupProvider<'a> {
    CachedGatedProductGroupProvider::new(cache)
}
