//! MLX realization of backend-neutral routed-expert providers.

use std::time::Instant;

use eredu_checkpoint::WeightQuantization;
use eredu_nn::TensorParallelExpertOutput;
use eredu_runtime::{
    ExpertPass, RoutedExpertProvider, RoutedExpertRequest, RoutedExpertTensorParallelOutput,
};
use safemlx::{module::Param, ops::indexing::TryIndexOp, Array, Stream};

use crate::backend::mlx::nn::moe::{PackedGatedProductExperts, PackedRelu2Experts};
use crate::backend::mlx::nn::shared::MlxBackend;
use crate::backend::mlx::runtime::residency::expert_cache::{ExpertCache, ExpertRouteBatch};
use crate::backend::mlx::Error;

/// Backend geometry, equation, and physical encoding for one cached gated-product bank.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CachedGatedProductBankSpec {
    pub(crate) hidden_dimensions: i32,
    pub(crate) intermediate_dimensions: i32,
    pub(crate) gate_up_quantization: Option<WeightQuantization>,
    pub(crate) down_quantization: Option<WeightQuantization>,
    pub(crate) gate_up_bias: bool,
    pub(crate) down_bias: bool,
    pub(crate) policy: eredu_nn::GatedProductPolicy,
}

/// Backend geometry and physical encoding for one cached ReLU2 bank.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CachedRelu2BankSpec {
    pub(crate) hidden_dimensions: i32,
    pub(crate) intermediate_dimensions: i32,
    pub(crate) up_quantization: Option<WeightQuantization>,
    pub(crate) down_quantization: Option<WeightQuantization>,
}

/// Executes independently cached ReLU2 experts through a layer-spec factory.
pub(crate) struct CachedRelu2ExpertProvider<'a, F> {
    cache: &'a ExpertCache,
    spec_for_layer: F,
}

impl<'a, F> CachedRelu2ExpertProvider<'a, F> {
    pub(crate) const fn new(cache: &'a ExpertCache, spec_for_layer: F) -> Self {
        Self {
            cache,
            spec_for_layer,
        }
    }
}

impl<F> RoutedExpertProvider<MlxBackend> for CachedRelu2ExpertProvider<'_, F>
where
    F: FnMut(usize) -> CachedRelu2BankSpec,
{
    type Error = Error;

    fn forward_routed(
        &mut self,
        _resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        _request: RoutedExpertRequest<'_, Array>,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        Err(Error::UnsupportedArchitecture(
            "a ReLU2 expert cache cannot execute a gated-product expert bank".into(),
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::Relu2ExpertBank,
        request: RoutedExpertRequest<'_, Array>,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        execute_cached_relu2(
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

/// Executes independently cached gated-product experts through a layer-spec factory.
pub(crate) struct CachedGatedProductExpertProvider<'a, F> {
    cache: &'a ExpertCache,
    spec_for_layer: F,
}

impl<'a, F> CachedGatedProductExpertProvider<'a, F> {
    pub(crate) const fn new(cache: &'a ExpertCache, spec_for_layer: F) -> Self {
        Self {
            cache,
            spec_for_layer,
        }
    }
}

impl<F> RoutedExpertProvider<MlxBackend> for CachedGatedProductExpertProvider<'_, F>
where
    F: FnMut(usize) -> CachedGatedProductBankSpec,
{
    type Error = Error;

    fn forward_routed(
        &mut self,
        _resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, Array>,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        execute_cached_gated_product(
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

    fn forward_routed_tensor_parallel(
        &mut self,
        _resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, Array>,
        partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<Array>, Self::Error> {
        execute_cached_gated_product_tensor_parallel(
            self.cache,
            (self.spec_for_layer)(request.layer),
            request.layer,
            request.input,
            &request.routes.expert_ids,
            &request.routes.route_weights,
            request.pass,
            partitions,
            stream,
        )
        .map(RoutedExpertTensorParallelOutput::Partial)
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::Relu2ExpertBank,
        _request: RoutedExpertRequest<'_, Array>,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        Err(Error::UnsupportedArchitecture(
            "a gated-product expert cache cannot execute a ReLU2 expert bank".into(),
        ))
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

fn execute_routed_callback<F>(
    execute: &mut F,
    request: RoutedExpertRequest<'_, Array>,
    stream: &Stream,
) -> Result<Array, safemlx::error::Exception>
where
    F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, safemlx::error::Exception>,
{
    if request.input.ndim() < 2 {
        return Err(safemlx::error::Exception::custom(format!(
            "routed expert input must have a hidden dimension, got {:?}",
            request.input.shape()
        )));
    }
    let original_shape = request.input.shape().to_vec();
    let hidden = request
        .input
        .reshape(&[-1, request.input.dim(-1)], stream)?;
    let expert_ids = request
        .routes
        .expert_ids
        .reshape(&[-1, request.routes.expert_ids.dim(-1)], stream)?;
    let route_weights = request
        .routes
        .route_weights
        .reshape(&[-1, request.routes.route_weights.dim(-1)], stream)?;
    let output = execute(request.layer, &hidden, &expert_ids, &route_weights, stream)?;
    output.reshape(&original_shape, stream)
}

impl<F> RoutedExpertProvider<MlxBackend> for ExpertExecutorProvider<'_, F>
where
    F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, safemlx::error::Exception>,
{
    type Error = safemlx::error::Exception;

    fn forward_routed(
        &mut self,
        _resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, Array>,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        execute_routed_callback(self.execute, request, stream)
    }

    fn forward_routed_tensor_parallel(
        &mut self,
        _resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, Array>,
        _partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<Array>, Self::Error> {
        execute_routed_callback(self.execute, request, stream).map(|reducible| {
            RoutedExpertTensorParallelOutput::Partial(TensorParallelExpertOutput {
                reducible,
                post_reduce: None,
            })
        })
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::Relu2ExpertBank,
        request: RoutedExpertRequest<'_, Array>,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        execute_routed_callback(self.execute, request, stream)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::Relu2ExpertBank,
        request: RoutedExpertRequest<'_, Array>,
        _partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<Array>, Self::Error> {
        execute_routed_callback(self.execute, request, stream).map(|reducible| {
            RoutedExpertTensorParallelOutput::Partial(TensorParallelExpertOutput {
                reducible,
                post_reduce: None,
            })
        })
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
        &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        &Array,
        &Array,
        &Array,
        usize,
        &Stream,
    ) -> Result<TensorParallelExpertOutput<Array>, safemlx::error::Exception>,
{
    type Error = safemlx::error::Exception;

    fn forward_routed(
        &mut self,
        resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, Array>,
        stream: &Stream,
    ) -> Result<Array, Self::Error> {
        let output = (self.execute)(
            resident_bank,
            request.input,
            &request.routes.expert_ids,
            &request.routes.route_weights,
            1,
            stream,
        )?;
        match output.post_reduce {
            Some(bias) => Ok(output.reducible.add(&bias, stream)?),
            None => Ok(output.reducible),
        }
    }

    fn forward_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, Array>,
        partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<Array>, Self::Error> {
        (self.execute)(
            resident_bank,
            request.input,
            &request.routes.expert_ids,
            &request.routes.route_weights,
            partitions,
            stream,
        )
        .map(RoutedExpertTensorParallelOutput::Partial)
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxBackend as eredu_nn::RoutedNeuralBackend>::Relu2ExpertBank,
        _request: RoutedExpertRequest<'_, Array>,
        _stream: &Stream,
    ) -> Result<Array, Self::Error> {
        Err(safemlx::error::Exception::custom(
            "a resident gated-product executor cannot execute a ReLU2 expert bank",
        ))
    }
}

/// Executes one cached route batch with a compact bank retained by the cache.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_cached_gated_product(
    cache: &ExpertCache,
    spec: CachedGatedProductBankSpec,
    layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    route_weights: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    execute_cached_gated_product_inner(
        cache,
        spec,
        layer,
        hidden,
        expert_ids,
        route_weights,
        pass,
        None,
        stream,
    )
}

/// Executes one cached tensor-parallel route batch with exact-once down bias.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_cached_gated_product_tensor_parallel(
    cache: &ExpertCache,
    spec: CachedGatedProductBankSpec,
    layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    route_weights: &Array,
    pass: ExpertPass,
    partitions: usize,
    stream: &Stream,
) -> Result<TensorParallelExpertOutput<Array>, Error> {
    let original_shape = hidden.shape().to_vec();
    let packed = execute_cached_gated_product_inner(
        cache,
        spec,
        layer,
        hidden,
        expert_ids,
        route_weights,
        pass,
        Some(partitions),
        stream,
    )?;
    let hidden_dimensions = spec.hidden_dimensions;
    let packed = packed.reshape(&[-1, 2 * hidden_dimensions], stream)?;
    let reducible = packed
        .try_index_device((.., ..hidden_dimensions), stream)?
        .reshape(&original_shape, stream)?;
    Ok(TensorParallelExpertOutput {
        reducible,
        post_reduce: spec
            .down_bias
            .then(|| {
                packed
                    .try_index_device((.., hidden_dimensions..), stream)?
                    .reshape(&original_shape, stream)
            })
            .transpose()?,
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_cached_gated_product_inner(
    cache: &ExpertCache,
    spec: CachedGatedProductBankSpec,
    layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    route_weights: &Array,
    pass: ExpertPass,
    partitions: Option<usize>,
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
            let mut bank = PackedGatedProductExperts::new(
                acquired.identities().len() as i32,
                spec.hidden_dimensions,
                spec.intermediate_dimensions,
                spec.gate_up_quantization.or(load_time),
                spec.down_quantization.or(load_time),
                [spec.gate_up_bias, spec.down_bias],
                stream,
            )?
            .with_policy(spec.policy)?;
            bank.gate_up_proj = Param::new(acquired.compact_binding("gate_up_proj", stream)?);
            bank.gate_up_proj_bias =
                Param::new(acquired.optional_compact_binding("gate_up_proj_bias", stream)?);
            bank.gate_up_proj_scales =
                Param::new(acquired.optional_compact_binding("gate_up_proj_scales", stream)?);
            bank.gate_up_proj_biases =
                Param::new(acquired.optional_compact_binding("gate_up_proj_biases", stream)?);
            bank.down_proj = Param::new(acquired.compact_binding("down_proj", stream)?);
            bank.down_proj_bias =
                Param::new(acquired.optional_compact_binding("down_proj_bias", stream)?);
            bank.down_proj_scales =
                Param::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
            bank.down_proj_biases =
                Param::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
            cache.record_compact_bank(
                acquired.pass(),
                acquired.scratch_bytes(),
                started.elapsed(),
            )?;
            Ok(match partitions {
                Some(partitions) => {
                    let output = bank.forward_tensor_parallel(
                        hidden,
                        acquired.compact_routes(),
                        weights,
                        partitions,
                        stream,
                    )?;
                    let post_reduce = match output.post_reduce {
                        Some(bias) => bias,
                        None => safemlx::ops::zeros_dtype(
                            output.reducible.shape(),
                            output.reducible.dtype(),
                            stream,
                        )?,
                    };
                    safemlx::ops::concatenate_axis(&[output.reducible, post_reduce], -1, stream)?
                }
                None => bank.forward(hidden, acquired.compact_routes(), weights, stream)?,
            })
        },
    )?;
    let mut output_shape = original_shape;
    if partitions.is_some() {
        let last = output_shape.last_mut().ok_or_else(|| {
            Error::UnsupportedArchitecture("expert output has no hidden axis".into())
        })?;
        *last = last.checked_mul(2).ok_or_else(|| {
            Error::UnsupportedArchitecture("expert output width overflowed".into())
        })?;
    }
    Ok(output.reshape(&output_shape, stream)?)
}

/// Executes one cached ReLU2 route batch with a compact acquired bank.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_cached_relu2(
    cache: &ExpertCache,
    spec: CachedRelu2BankSpec,
    layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    route_weights: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    let original_shape = hidden.shape().to_vec();
    let flattened = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
    let output = cache.execute_routes_bounded(
        ExpertRouteBatch::new(layer, &flattened, expert_ids, route_weights, pass),
        stream,
        |hidden, acquired, weights, stream| {
            let started = Instant::now();
            let load_time = cache.weight_quantization();
            let mut bank = PackedRelu2Experts::new(
                acquired.identities().len() as i32,
                spec.hidden_dimensions,
                spec.intermediate_dimensions,
                [
                    spec.up_quantization.or(load_time),
                    spec.down_quantization.or(load_time),
                ],
                stream,
            )?;
            bank.up_proj = Param::new(acquired.compact_binding("up_proj", stream)?);
            bank.up_proj_scales =
                Param::new(acquired.optional_compact_binding("up_proj_scales", stream)?);
            bank.up_proj_biases =
                Param::new(acquired.optional_compact_binding("up_proj_biases", stream)?);
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

/// Executes ReLU2 route rows already compacted by distributed ownership dispatch.
pub(crate) fn execute_cached_relu2_dispatched(
    cache: &ExpertCache,
    spec: CachedRelu2BankSpec,
    layer: usize,
    hidden: &Array,
    global_expert_ids: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    let expert_ids = global_expert_ids.reshape(&[-1, 1], stream)?;
    let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
    execute_cached_relu2(
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

/// Executes route rows already compacted by distributed ownership dispatch.
///
/// Each input row represents exactly one selected route. The outer dispatcher
/// applies the original route weight while recombining rows, so this compact
/// bank must use a unit weight and return one unweighted output row per input.
pub(crate) fn execute_cached_gated_product_dispatched(
    cache: &ExpertCache,
    spec: CachedGatedProductBankSpec,
    layer: usize,
    hidden: &Array,
    global_expert_ids: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    let expert_ids = global_expert_ids.reshape(&[-1, 1], stream)?;
    let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
    execute_cached_gated_product(
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
