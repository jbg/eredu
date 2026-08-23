//! MLX routed-expert execution shared by every distributed stage.

use eredu_runtime::ExpertPass;
use safemlx::{Array, Stream};

use crate::backend::{error::Error, runtime::residency::expert_cache::ExpertCache};

use crate::backend::runtime::distributed::expert::DispatchedRoutes;

pub(super) fn execute_cached_neutral_gemma4(
    args: &eredu_architectures::gemma4::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let spec = eredu_architectures::gemma4::text::expert_bank_spec(args, layer)?;
    crate::backend::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
        cache,
        &spec,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub(super) fn execute_cached_kimi_linear(
    args: &eredu_architectures::kimi_linear::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let spec = eredu_architectures::kimi_linear::moe::expert_bank_spec(args, layer)?;
    crate::backend::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
        cache,
        &spec,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub(super) fn execute_cached_neutral_qwen3(
    args: &eredu_architectures::qwen::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    crate::composition::qwen::expert::execute_cached_dispatched(
        cache,
        args,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub(super) fn execute_cached_neutral_inkling(
    args: &eredu_architectures::inkling::ModelArgs,
    cache_layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let spec = eredu_architectures::inkling::text::expert_bank_spec(args, cache_layer)?;
    crate::backend::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
        cache,
        &spec,
        cache_layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub(super) fn execute_cached_lfm2(
    args: &eredu_architectures::lfm2::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let spec = eredu_architectures::lfm2::moe::expert_bank_spec(args, layer)?;
    crate::backend::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
        cache,
        &spec,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub(super) fn execute_cached_muse_glimmer(
    args: &eredu_architectures::muse_glimmer::DecoderConfig,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let spec = eredu_architectures::muse_glimmer::text::expert_bank_spec(args, layer)?;
    crate::backend::runtime::residency::expert_provider::execute_cached_gated_product_dispatched(
        cache,
        &spec,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}

pub(super) fn execute_cached_nemotron_h(
    args: &eredu_architectures::nemotron_h::ModelArgs,
    layer: usize,
    routes: &DispatchedRoutes,
    pass: ExpertPass,
    cache: &ExpertCache,
    stream: &Stream,
) -> Result<Array, Error> {
    let spec = eredu_architectures::nemotron_h::expert_bank_spec(args, layer)?;
    crate::backend::runtime::residency::expert_provider::execute_cached_relu2_dispatched(
        cache,
        &spec,
        layer,
        &routes.hidden,
        &routes.global_expert_ids,
        pass,
        stream,
    )
}
