//! MLX routed-expert execution shared by every distributed stage.

use eredu_runtime::ExpertPass;
use safemlx::{Array, Stream};

use crate::backend::{error::Error, runtime::residency::expert_cache::ExpertCache};

use crate::backend::runtime::distributed::expert::DispatchedRoutes;

pub(super) fn execute_cached_gated_product(
    spec: &eredu_nn::GatedProductExpertBankSpec,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    crate::backend::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
        cache,
        spec,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub(super) fn execute_cached_relu2(
    spec: &eredu_nn::Relu2ExpertBankSpec,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    crate::backend::runtime::residency::expert_provider::execute_cached_relu2_dispatched(
        cache,
        spec,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}
