//! MLX routed-expert execution shared by every distributed stage.

use eredu_runtime::ExpertPass;
use safemlx::{Array, Stream};

use crate::backend::{error::Error, runtime::residency::expert_cache::ExpertCache};

use crate::backend::runtime::distributed::expert::DispatchedRoutes;

pub(super) fn execute_cached_neutral_gemma4(
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

pub(super) fn execute_cached_kimi_linear(
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

pub(super) fn execute_cached_neutral_qwen3(
    spec: &eredu_nn::GatedProductExpertBankSpec,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    crate::composition::qwen::expert::execute_cached_dispatched(
        cache,
        spec,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub(super) fn execute_cached_neutral_inkling(
    spec: &eredu_nn::GatedProductExpertBankSpec,
    cache_layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    crate::backend::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
        cache,
        spec,
        cache_layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub(super) fn execute_cached_lfm2(
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

pub(super) fn execute_cached_muse_glimmer(
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

pub(super) fn execute_cached_nemotron_h(
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
