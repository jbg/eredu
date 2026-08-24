//! MLX realization of backend-neutral routed-expert providers.

use std::time::Instant;

use eredu_nn::{
    GatedProductExpertBankOperator, GatedProductExpertBankSpec, GatedProductExpertLayout,
    Relu2ExpertBankSpec, TensorParallelExpertOutput,
};
use eredu_runtime::{
    ExpertPass, RoutedExpertProvider, RoutedExpertRequest, RoutedExpertTensorParallelOutput,
};
use safemlx::{module::Param, ops::indexing::TryIndexOp, Array, Stream};

use crate::backend::nn::moe::{PackedGatedProductExperts, PackedRelu2Experts};
use crate::backend::nn::shared::MlxNeuralBackend;
use crate::backend::runtime::residency::expert_cache::{ExpertCache, ExpertRouteBatch};
use crate::backend::Error;
use crate::MlxTensor;

fn wrap_parallel_output(
    output: TensorParallelExpertOutput<Array>,
) -> TensorParallelExpertOutput<MlxTensor> {
    TensorParallelExpertOutput {
        reducible: MlxTensor::from_array(output.reducible),
        post_reduce: output.post_reduce.map(MlxTensor::from_array),
    }
}

/// Executes independently cached ReLU2 experts through a layer-spec factory.
pub struct CachedRelu2ExpertProvider<'a, F> {
    cache: &'a ExpertCache,
    spec_for_layer: F,
}

impl<'a, F> CachedRelu2ExpertProvider<'a, F> {
    pub const fn new(cache: &'a ExpertCache, spec_for_layer: F) -> Self {
        Self {
            cache,
            spec_for_layer,
        }
    }
}

impl<F> RoutedExpertProvider<MlxNeuralBackend> for CachedRelu2ExpertProvider<'_, F>
where
    F: FnMut(usize) -> Result<Relu2ExpertBankSpec, Error>,
{
    type Error = Error;

    fn forward_routed(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        _request: RoutedExpertRequest<'_, MlxTensor>,
        _stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        Err(Error::ArchitectureModel(
            "a ReLU2 expert cache cannot execute a gated-product expert bank".into(),
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::Relu2ExpertBank,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        execute_cached_relu2(
            self.cache,
            &(self.spec_for_layer)(request.layer)?,
            request.layer,
            request.input.as_array(),
            request.routes.expert_ids.as_array(),
            request.routes.route_weights.as_array(),
            request.pass,
            stream,
        )
        .map(MlxTensor::from_array)
    }
}

/// Executes independently cached gated-product experts with resident-bank semantics.
pub struct CachedGatedProductExpertProvider<'a> {
    cache: &'a ExpertCache,
}

impl<'a> CachedGatedProductExpertProvider<'a> {
    pub const fn new(cache: &'a ExpertCache) -> Self {
        Self { cache }
    }
}

impl RoutedExpertProvider<MlxNeuralBackend> for CachedGatedProductExpertProvider<'_> {
    type Error = Error;

    fn forward_routed(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        execute_cached_gated_product(
            self.cache,
            resident_bank.spec(),
            request.layer,
            request.input.as_array(),
            request.routes.expert_ids.as_array(),
            request.routes.route_weights.as_array(),
            request.pass,
            stream,
        )
        .map(MlxTensor::from_array)
    }

    fn forward_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, MlxTensor>,
        partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, Self::Error> {
        execute_cached_gated_product_tensor_parallel(
            self.cache,
            resident_bank.spec(),
            request.layer,
            request.input.as_array(),
            request.routes.expert_ids.as_array(),
            request.routes.route_weights.as_array(),
            request.pass,
            partitions,
            stream,
        )
        .map(wrap_parallel_output)
        .map(RoutedExpertTensorParallelOutput::Partial)
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::Relu2ExpertBank,
        _request: RoutedExpertRequest<'_, MlxTensor>,
        _stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        Err(Error::ArchitectureModel(
            "a gated-product expert cache cannot execute a ReLU2 expert bank".into(),
        ))
    }
}

/// Adapts a distributed callback that owns expert acquisition and collectives.
pub struct ExpertExecutorProvider<'a, F> {
    execute: &'a mut F,
}

impl<'a, F> ExpertExecutorProvider<'a, F> {
    pub const fn new(execute: &'a mut F) -> Self {
        Self { execute }
    }
}

fn execute_routed_callback<F>(
    execute: &mut F,
    request: RoutedExpertRequest<'_, MlxTensor>,
    stream: &Stream,
) -> Result<MlxTensor, safemlx::error::Exception>
where
    F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, safemlx::error::Exception>,
{
    if request.input.as_array().ndim() < 2 {
        return Err(safemlx::error::Exception::custom(format!(
            "routed expert input must have a hidden dimension, got {:?}",
            request.input.as_array().shape()
        )));
    }
    let original_shape = request.input.as_array().shape().to_vec();
    let hidden = request
        .input
        .as_array()
        .reshape(&[-1, request.input.as_array().dim(-1)], stream)?;
    let expert_ids = request
        .routes
        .expert_ids
        .as_array()
        .reshape(&[-1, request.routes.expert_ids.as_array().dim(-1)], stream)?;
    let route_weights = request.routes.route_weights.as_array().reshape(
        &[-1, request.routes.route_weights.as_array().dim(-1)],
        stream,
    )?;
    let output = execute(request.layer, &hidden, &expert_ids, &route_weights, stream)?;
    output
        .reshape(&original_shape, stream)
        .map(MlxTensor::from_array)
}

impl<F> RoutedExpertProvider<MlxNeuralBackend> for ExpertExecutorProvider<'_, F>
where
    F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, safemlx::error::Exception>,
{
    type Error = safemlx::error::Exception;

    fn forward_routed(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        execute_routed_callback(self.execute, request, stream)
    }

    fn forward_routed_tensor_parallel(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, MlxTensor>,
        _partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, Self::Error> {
        execute_routed_callback(self.execute, request, stream)
            .map(RoutedExpertTensorParallelOutput::Complete)
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::Relu2ExpertBank,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        execute_routed_callback(self.execute, request, stream)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::Relu2ExpertBank,
        request: RoutedExpertRequest<'_, MlxTensor>,
        _partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, Self::Error> {
        execute_routed_callback(self.execute, request, stream)
            .map(RoutedExpertTensorParallelOutput::Complete)
    }
}

/// Completion contract requested from a gated-product expert callback.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum GatedProductExpertExecutionMode {
    /// Execute the ordinary bank and return a globally complete output.
    Complete,
    /// Execute a rank-local TP contribution and preserve its post-reduce term.
    TensorParallel {
        /// Partition count supplied by the architecture provider call.
        partitions: usize,
    },
}

/// Architecture-owned geometry and tensors supplied to an expert callback.
pub struct GatedProductExpertExecution {
    /// Global decoder layer requesting expert execution.
    pub layer: usize,
    /// Rank-local bank specification retained by the architecture module.
    pub spec: GatedProductExpertBankSpec,
    /// Flattened token rows submitted to the selected experts.
    pub hidden: Array,
    /// Selected global expert identities.
    pub expert_ids: Array,
    /// Route weights aligned with `expert_ids`.
    pub route_weights: Array,
    /// Required complete or rank-local tensor-parallel result contract.
    pub mode: GatedProductExpertExecutionMode,
}

/// Adapts a callback that explicitly preserves complete versus rank-local TP
/// expert semantics and consumes the architecture-owned local bank geometry.
pub struct GatedProductExpertExecutorProvider<'a, F> {
    execute: &'a mut F,
}

impl<'a, F> GatedProductExpertExecutorProvider<'a, F> {
    pub const fn new(execute: &'a mut F) -> Self {
        Self { execute }
    }
}

fn reshape_gated_product_callback_output(
    output: RoutedExpertTensorParallelOutput<Array>,
    original_shape: &[i32],
    stream: &Stream,
) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, safemlx::error::Exception> {
    match output {
        RoutedExpertTensorParallelOutput::Complete(output) => output
            .reshape(original_shape, stream)
            .map(MlxTensor::from_array)
            .map(RoutedExpertTensorParallelOutput::Complete),
        RoutedExpertTensorParallelOutput::Partial(output) => {
            let reducible = output.reducible.reshape(original_shape, stream)?;
            let post_reduce = output
                .post_reduce
                .map(|value| value.reshape(original_shape, stream))
                .transpose()?;
            Ok(RoutedExpertTensorParallelOutput::Partial(
                TensorParallelExpertOutput {
                    reducible: MlxTensor::from_array(reducible),
                    post_reduce: post_reduce.map(MlxTensor::from_array),
                },
            ))
        }
    }
}

fn execute_gated_product_callback<F>(
    execute: &mut F,
    resident_bank: &<MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
    request: RoutedExpertRequest<'_, MlxTensor>,
    mode: GatedProductExpertExecutionMode,
    stream: &Stream,
) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, safemlx::error::Exception>
where
    F: FnMut(
        GatedProductExpertExecution,
        &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<Array>, safemlx::error::Exception>,
{
    if request.input.as_array().ndim() < 2 {
        return Err(safemlx::error::Exception::custom(format!(
            "routed expert input must have a hidden dimension, got {:?}",
            request.input.as_array().shape()
        )));
    }
    let original_shape = request.input.as_array().shape().to_vec();
    let hidden = request
        .input
        .as_array()
        .reshape(&[-1, request.input.as_array().dim(-1)], stream)?;
    let expert_ids = request
        .routes
        .expert_ids
        .as_array()
        .reshape(&[-1, request.routes.expert_ids.as_array().dim(-1)], stream)?;
    let route_weights = request.routes.route_weights.as_array().reshape(
        &[-1, request.routes.route_weights.as_array().dim(-1)],
        stream,
    )?;
    let output = execute(
        GatedProductExpertExecution {
            layer: request.layer,
            spec: resident_bank.spec().clone(),
            hidden,
            expert_ids,
            route_weights,
            mode,
        },
        stream,
    )?;
    reshape_gated_product_callback_output(output, &original_shape, stream)
}

impl<F> RoutedExpertProvider<MlxNeuralBackend> for GatedProductExpertExecutorProvider<'_, F>
where
    F: FnMut(
        GatedProductExpertExecution,
        &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<Array>, safemlx::error::Exception>,
{
    type Error = safemlx::error::Exception;

    fn forward_routed(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        match execute_gated_product_callback(
            self.execute,
            resident_bank,
            request,
            GatedProductExpertExecutionMode::Complete,
            stream,
        )? {
            RoutedExpertTensorParallelOutput::Complete(output) => Ok(output),
            RoutedExpertTensorParallelOutput::Partial(_) => Err(safemlx::error::Exception::custom(
                "ordinary expert execution returned a rank-local tensor-parallel partial",
            )),
        }
    }

    fn forward_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, MlxTensor>,
        partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, Self::Error> {
        execute_gated_product_callback(
            self.execute,
            resident_bank,
            request,
            GatedProductExpertExecutionMode::TensorParallel { partitions },
            stream,
        )
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::Relu2ExpertBank,
        _request: RoutedExpertRequest<'_, MlxTensor>,
        _stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        Err(safemlx::error::Exception::custom(
            "a gated-product executor cannot execute a ReLU2 expert bank",
        ))
    }
}

/// Adapts a distributed callback that consumes the resident rank-local bank.
pub struct ResidentExpertExecutorProvider<'a, F> {
    execute: &'a mut F,
}

impl<'a, F> ResidentExpertExecutorProvider<'a, F> {
    pub const fn new(execute: &'a mut F) -> Self {
        Self { execute }
    }
}

impl<F> RoutedExpertProvider<MlxNeuralBackend> for ResidentExpertExecutorProvider<'_, F>
where
    F: FnMut(
        &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
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
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        let output = (self.execute)(
            resident_bank,
            request.input.as_array(),
            request.routes.expert_ids.as_array(),
            request.routes.route_weights.as_array(),
            1,
            stream,
        )?;
        match output.post_reduce {
            Some(bias) => output
                .reducible
                .add(&bias, stream)
                .map(MlxTensor::from_array),
            None => Ok(MlxTensor::from_array(output.reducible)),
        }
    }

    fn forward_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::GatedProductExpertBank,
        request: RoutedExpertRequest<'_, MlxTensor>,
        partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, Self::Error> {
        (self.execute)(
            resident_bank,
            request.input.as_array(),
            request.routes.expert_ids.as_array(),
            request.routes.route_weights.as_array(),
            partitions,
            stream,
        )
        .map(wrap_parallel_output)
        .map(RoutedExpertTensorParallelOutput::Partial)
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::RoutedNeuralBackend>::Relu2ExpertBank,
        _request: RoutedExpertRequest<'_, MlxTensor>,
        _stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        Err(safemlx::error::Exception::custom(
            "a resident gated-product executor cannot execute a ReLU2 expert bank",
        ))
    }
}

/// Executes one cached route batch with a compact bank retained by the cache.
#[allow(clippy::too_many_arguments)]
pub fn execute_cached_gated_product(
    cache: &ExpertCache,
    spec: &GatedProductExpertBankSpec,
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
pub fn execute_cached_gated_product_tensor_parallel(
    cache: &ExpertCache,
    spec: &GatedProductExpertBankSpec,
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
    let output_dimensions = spec.output_dimensions;
    let packed = packed.reshape(&[-1, 2 * output_dimensions], stream)?;
    let reducible = packed
        .try_index_device((.., ..output_dimensions), stream)?
        .reshape(&original_shape, stream)?;
    Ok(TensorParallelExpertOutput {
        reducible,
        post_reduce: packed_gated_product_projections(spec)?
            .1
            .bias
            .is_some()
            .then(|| {
                packed
                    .try_index_device((.., output_dimensions..), stream)?
                    .reshape(&original_shape, stream)
            })
            .transpose()?,
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_cached_gated_product_inner(
    cache: &ExpertCache,
    spec: &GatedProductExpertBankSpec,
    layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    route_weights: &Array,
    pass: ExpertPass,
    partitions: Option<usize>,
    stream: &Stream,
) -> Result<Array, Error> {
    spec.validate()?;
    if spec.input_dimensions != spec.output_dimensions {
        return Err(Error::ArchitectureModel(
            "MLX cached gated-product experts require equal input and output dimensions".into(),
        ));
    }
    let (gate_up, down) = packed_gated_product_projections(spec)?;
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
                spec.input_dimensions,
                spec.intermediate_dimensions,
                gate_up.format.weight_quantization().or(load_time),
                down.format.weight_quantization().or(load_time),
                [gate_up.bias.is_some(), down.bias.is_some()],
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
        let last = output_shape
            .last_mut()
            .ok_or_else(|| Error::ArchitectureModel("expert output has no hidden axis".into()))?;
        *last = last
            .checked_mul(2)
            .ok_or_else(|| Error::ArchitectureModel("expert output width overflowed".into()))?;
    }
    Ok(output.reshape(&output_shape, stream)?)
}

/// Executes one cached ReLU2 route batch with a compact acquired bank.
#[allow(clippy::too_many_arguments)]
pub fn execute_cached_relu2(
    cache: &ExpertCache,
    spec: &Relu2ExpertBankSpec,
    layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    route_weights: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    spec.validate()?;
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
                    spec.up.format.weight_quantization().or(load_time),
                    spec.down.format.weight_quantization().or(load_time),
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
pub fn execute_cached_relu2_dispatched(
    cache: &ExpertCache,
    spec: &Relu2ExpertBankSpec,
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
pub fn execute_cached_gated_product_dispatched(
    cache: &ExpertCache,
    spec: &GatedProductExpertBankSpec,
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

fn packed_gated_product_projections(
    spec: &GatedProductExpertBankSpec,
) -> Result<
    (
        &eredu_nn::ExpertProjectionSpec,
        &eredu_nn::ExpertProjectionSpec,
    ),
    Error,
> {
    match &spec.layout {
        GatedProductExpertLayout::Packed { gate_up, down } => Ok((gate_up, down)),
        GatedProductExpertLayout::Independent(_) => Err(Error::ArchitectureModel(
            "MLX compact cached banks require a packed architecture expert specification".into(),
        )),
    }
}
