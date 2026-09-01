//! MLX checkpoint and residency adapter for neutral Muse-Glimmer experts.

use eredu_architectures::muse_glimmer::DecoderConfig;

use crate::composition::grouped_provider::*;

use crate::backend::{
    error::Error,
    runtime::residency::parameter_bank::{AddressableParameterBank, ParameterBankEntry},
};

pub fn expert_catalog(
    args: &DecoderConfig,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ParameterBankEntry>, Error> {
    let catalog = eredu_architectures::muse_glimmer::expert_residency_catalog(store, args)
        .map_err(Error::ArchitectureModel)?;
    crate::composition::architecture_expert_units(catalog, store, None)
}

pub const fn cached_provider<'a>(
    cache: &'a AddressableParameterBank,
    _args: &DecoderConfig,
) -> CachedGatedProductGroupProvider<'a> {
    CachedGatedProductGroupProvider::new(cache)
}
