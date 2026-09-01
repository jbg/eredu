//! MLX checkpoint and residency adapter for neutral Inkling expert banks.

use eredu_architectures::inkling::ModelArgs;

use crate::composition::grouped_provider::*;

use crate::backend::{
    error::Error,
    runtime::residency::parameter_bank::{AddressableParameterBank, ParameterBankEntry},
};

/// Returns one independently leasable unit for every routed and shared expert.
pub fn expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ParameterBankEntry>, Error> {
    let catalog = eredu_architectures::inkling::expert_residency_catalog(args, store)
        .map_err(Error::ArchitectureModel)?;
    super::architecture_expert_units(catalog, store, None)
}

pub const fn cached_provider<'a>(
    cache: &'a AddressableParameterBank,
    _args: &ModelArgs,
) -> CachedGatedProductGroupProvider<'a> {
    CachedGatedProductGroupProvider::new(cache)
}
