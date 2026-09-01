//! MLX routed-expert execution shared by every distributed stage.

use eredu_runtime::ExpertPass;
use safemlx::{Array, Stream};

use crate::backend::{error::Error, runtime::residency::parameter_bank::AddressableParameterBank};

use crate::composition::expert_dispatch::DispatchedRoutes;

pub(super) fn execute_cached_gated_product(
    spec: &eredu_nn::GroupedGatedProductSpec,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &AddressableParameterBank,
    stream: &Stream,
) -> Result<Array, Error> {
    crate::composition::grouped_provider::execute_cached_gated_product_dispatched(
        cache,
        spec,
        layer,
        &routes.hidden,
        &routes.global_group_indices,
        pass,
        stream,
    )
}

pub(super) fn execute_cached_relu2(
    spec: &eredu_nn::GroupedRelu2Spec,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &AddressableParameterBank,
    stream: &Stream,
) -> Result<Array, Error> {
    crate::composition::grouped_provider::execute_cached_relu2_dispatched(
        cache,
        spec,
        layer,
        &routes.hidden,
        &routes.global_group_indices,
        pass,
        stream,
    )
}
