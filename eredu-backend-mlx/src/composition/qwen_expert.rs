// MLX residency adapter for neutral Qwen routed-expert checkpoint contracts.

use eredu_architectures::qwen::ModelArgs;
use eredu_runtime::ExpertPass;
use safemlx::{Array, Stream};

use crate::backend::mlx::runtime::residency::expert_cache::ExpertCache;
use crate::backend::mlx::runtime::residency::expert_provider::{
    execute_cached_gated_product_dispatched, CachedGatedProductExpertProvider,
};
use crate::backend::mlx::{
    error::Error, runtime::residency::expert_cache::ExpertCatalogEntry,
};

pub fn expert_catalog(
    args: &ModelArgs,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<Vec<ExpertCatalogEntry>, Error> {
    let catalog = eredu_architectures::qwen::expert_residency_catalog(store, args)
        .map_err(Error::UnsupportedArchitecture)?;
    crate::composition::architecture_expert_units(catalog, store, None)
}

pub const fn cached_provider<'a>(
    cache: &'a ExpertCache,
    _args: &ModelArgs,
) -> CachedGatedProductExpertProvider<'a> {
    CachedGatedProductExpertProvider::new(cache)
}

pub fn execute_cached_dispatched(
    cache: &ExpertCache,
    args: &ModelArgs,
    layer: usize,
    hidden: &Array,
    global_expert_ids: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    let spec = eredu_architectures::qwen::expert_bank_spec(args, layer)?;
    execute_cached_gated_product_dispatched(
        cache,
        &spec,
        layer,
        hidden,
        global_expert_ids,
        pass,
        stream,
    )
}
