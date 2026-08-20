//! MLX realization of backend-neutral routed-expert providers.

use std::time::Instant;

use eredu_checkpoint::WeightQuantization;
use eredu_runtime::{ExpertPass, RoutedExpertProvider, RoutedExpertRequest};
use safemlx::{module::Param, Array, Stream};

use crate::backend::mlx::nn::moe::PackedSwiGluExperts;
use crate::backend::mlx::nn::shared::MlxBackend;
use crate::backend::mlx::runtime::residency::expert_cache::{ExpertCache, ExpertRouteBatch};
use crate::backend::mlx::Error;

/// Backend geometry and physical encoding for one cached SwiGLU bank.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CachedSwiGluBankSpec {
    pub(crate) hidden_dimensions: i32,
    pub(crate) intermediate_dimensions: i32,
    pub(crate) gate_up_quantization: Option<WeightQuantization>,
    pub(crate) down_quantization: Option<WeightQuantization>,
    pub(crate) limit: Option<f32>,
}

/// Executes independently cached SwiGLU experts through a layer-spec factory.
pub(crate) struct CachedSwiGluExpertProvider<'a, F> {
    cache: &'a ExpertCache,
    spec_for_layer: F,
}

impl<'a, F> CachedSwiGluExpertProvider<'a, F> {
    pub(crate) const fn new(cache: &'a ExpertCache, spec_for_layer: F) -> Self {
        Self {
            cache,
            spec_for_layer,
        }
    }
}

impl<F> RoutedExpertProvider<MlxBackend> for CachedSwiGluExpertProvider<'_, F>
where
    F: FnMut(usize) -> CachedSwiGluBankSpec,
{
    type Error = Error;

    fn forward_routed(
        &mut self,
        _resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::SwiGluExpertBank,
        request: RoutedExpertRequest<'_, Array>,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        execute_cached_swiglu(
            self.cache,
            (self.spec_for_layer)(request.layer),
            request.layer,
            request.input,
            &request.routes.expert_ids,
            &request.routes.route_weights,
            request.pass,
            stream,
        )
    }
}

/// Adapts a distributed callback that owns expert acquisition and collectives.
pub(crate) struct ExpertExecutorProvider<'a, F> {
    execute: &'a mut F,
}

impl<'a, F> ExpertExecutorProvider<'a, F> {
    pub(crate) const fn new(execute: &'a mut F) -> Self {
        Self { execute }
    }
}

impl<F> RoutedExpertProvider<MlxBackend> for ExpertExecutorProvider<'_, F>
where
    F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, safemlx::error::Exception>,
{
    type Error = safemlx::error::Exception;

    fn forward_routed(
        &mut self,
        _resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::SwiGluExpertBank,
        request: RoutedExpertRequest<'_, Array>,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        (self.execute)(
            request.layer,
            request.input,
            &request.routes.expert_ids,
            &request.routes.route_weights,
            stream,
        )
    }
}

/// Adapts a distributed callback that consumes the resident rank-local bank.
pub(crate) struct ResidentExpertExecutorProvider<'a, F> {
    execute: &'a mut F,
}

impl<'a, F> ResidentExpertExecutorProvider<'a, F> {
    pub(crate) const fn new(execute: &'a mut F) -> Self {
        Self { execute }
    }
}

impl<F> RoutedExpertProvider<MlxBackend> for ResidentExpertExecutorProvider<'_, F>
where
    F: FnMut(
        &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::SwiGluExpertBank,
        &Array,
        &Array,
        &Array,
        &Stream,
    ) -> Result<Array, safemlx::error::Exception>,
{
    type Error = safemlx::error::Exception;

    fn forward_routed(
        &mut self,
        resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::SwiGluExpertBank,
        request: RoutedExpertRequest<'_, Array>,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        (self.execute)(
            resident_bank,
            request.input,
            &request.routes.expert_ids,
            &request.routes.route_weights,
            stream,
        )
    }
}

/// Executes one cached route batch with a compact bank retained by the cache.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_cached_swiglu(
    cache: &ExpertCache,
    spec: CachedSwiGluBankSpec,
    layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    route_weights: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    // The neutral router reports one row per token while decoder hidden state
    // retains its leading batch/sequence dimensions. Resident expert banks
    // flatten and restore those dimensions as part of their operator contract;
    // cached banks must do the same before entering the row-oriented cache.
    let original_shape = hidden.shape().to_vec();
    let flattened = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
    let output = cache.execute_routes_bounded(
        ExpertRouteBatch::new(layer, &flattened, expert_ids, route_weights, pass),
        stream,
        |hidden, acquired, weights, stream| {
            let started = Instant::now();
            let load_time = cache.weight_quantization();
            let mut bank = PackedSwiGluExperts::new(
                acquired.identities().len() as i32,
                spec.hidden_dimensions,
                spec.intermediate_dimensions,
                load_time.or(spec.gate_up_quantization),
                load_time.or(spec.down_quantization),
                stream,
            )?;
            if let Some(limit) = spec.limit {
                bank = bank.with_swiglu_limit(limit)?;
            }
            bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
            bank.gate_up_proj_scales =
                Param::new(acquired.optional_compact_binding("gate_up_proj_scales", stream)?);
            bank.gate_up_proj_biases =
                Param::new(acquired.optional_compact_binding("gate_up_proj_biases", stream)?);
            bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
            bank.down_proj_scales =
                Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
            bank.down_proj_biases =
                Param::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
            cache.record_compact_bank(
                acquired.pass(),
                acquired.scratch_bytes(),
                started.elapsed(),
            )?;
            Ok(bank.forward(hidden, acquired.compact_routes(), weights, stream)?)
        },
    )?;
    Ok(output.reshape(&original_shape, stream)?)
}

/// Executes route rows already compacted by distributed ownership dispatch.
///
/// Each input row represents exactly one selected route. The outer dispatcher
/// applies the original route weight while recombining rows, so this compact
/// bank must use a unit weight and return one unweighted output row per input.
pub(crate) fn execute_cached_swiglu_dispatched(
    cache: &ExpertCache,
    spec: CachedSwiGluBankSpec,
    layer: usize,
    hidden: &Array,
    global_expert_ids: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    let expert_ids = global_expert_ids.reshape(&[-1, 1], stream)?;
    let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
    execute_cached_swiglu(
        cache,
        spec,
        layer,
        hidden,
        &expert_ids,
        &weights,
        pass,
        stream,
    )
}
