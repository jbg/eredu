//! MLX realization of backend-neutral routed-expert providers.

use std::time::Instant;

use eredu_nn::{
    GatedProductGroupLayout, GroupedGatedProductOperator, GroupedGatedProductSpec,
    GroupedRelu2Spec, TensorParallelGroupedOutput,
};
use eredu_runtime::{
    ExpertPass, IndexedMovement, RoutedExpertProvider, RoutedExpertRequest,
    RoutedExpertTensorParallelOutput, TensorParallelRoutedExpertProvider,
};
use safemlx::{ops::indexing::TryIndexOp, Array, Stream};

use crate::module::PhysicalParam;

use crate::backend::error::Error;
use crate::backend::nn::grouped::{PackedGatedProductGroups, PackedRelu2Groups};
use crate::backend::nn::shared::MlxNeuralBackend;
use crate::backend::runtime::residency::parameter_bank::{
    AcquiredParameterGroups, AddressableParameterBank, AddressableParameterBankError,
    BankAccessClass, MlxIndexedMovement, ParameterBankKey,
};
use crate::MlxTensor;

/// Composition-owned route batch for excluded distributed/composite provider adapters.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ParameterBankSelection<'a> {
    namespace: usize,
    hidden: &'a Array,
    group_indices: &'a Array,
    weights: &'a Array,
    pass: BankAccessClass,
}

impl<'a> ParameterBankSelection<'a> {
    pub(crate) const fn new(
        namespace: usize,
        hidden: &'a Array,
        group_indices: &'a Array,
        weights: &'a Array,
        pass: BankAccessClass,
    ) -> Self {
        Self {
            namespace,
            hidden,
            group_indices,
            weights,
            pass,
        }
    }
}

fn selection_chunk_rows(cache: &AddressableParameterBank, selections_per_row: usize) -> usize {
    if selections_per_row == 0 {
        return 1;
    }
    let bytes_per_row = cache
        .maximum_member_bytes()
        .max(1)
        .saturating_mul(u64::try_from(selections_per_row).unwrap_or(u64::MAX));
    let budget = cache
        .compact_bank_scratch_bytes()
        .min(cache.bulk_compact_bank_target_bytes())
        .max(1);
    usize::try_from(budget.checked_div(bytes_per_row).unwrap_or(0).max(1)).unwrap_or(usize::MAX)
}

/// Owns route interpretation, bounded row partitioning, and compact-id remapping
/// for composition-only cached-provider integrations.
pub(crate) fn execute_selections_bounded<F>(
    cache: &AddressableParameterBank,
    batch: ParameterBankSelection<'_>,
    stream: &Stream,
    mut execute_bank: F,
) -> Result<Array, Error>
where
    F: FnMut(&Array, &AcquiredParameterGroups, &Array, &Array, &Stream) -> Result<Array, Error>,
{
    let ParameterBankSelection {
        namespace,
        hidden: grouped_hidden,
        group_indices: grouped_ids,
        weights: coefficients,
        pass,
    } = batch;
    if grouped_hidden.ndim() == 0
        || grouped_ids.ndim() == 0
        || coefficients.ndim() == 0
        || grouped_hidden.dim(0) != grouped_ids.dim(0)
        || grouped_hidden.dim(0) != coefficients.dim(0)
    {
        return Err(AddressableParameterBankError::GroupedBatchShapeMismatch {
            hidden: grouped_hidden.shape().to_vec(),
            selections: grouped_ids.shape().to_vec(),
            weights: coefficients.shape().to_vec(),
        }
        .into());
    }
    let selections_per_row = grouped_ids.shape()[1..]
        .iter()
        .try_fold(1usize, |total, dimension| {
            usize::try_from(*dimension)
                .ok()
                .and_then(|dimension| total.checked_mul(dimension))
        })
        .ok_or_else(|| {
            AddressableParameterBankError::InvalidSelectionShape(grouped_ids.shape().to_vec())
        })?;
    let global_span = cache
        .namespace_global_span(namespace)
        .ok_or(AddressableParameterBankError::UnknownNamespace { namespace })?;
    let row_count = grouped_hidden.dim(0);
    let chunk_rows = if pass == BankAccessClass::Bulk {
        i32::try_from(selection_chunk_rows(cache, selections_per_row)).unwrap_or(i32::MAX)
    } else {
        row_count.max(1)
    };
    let mut movement = MlxIndexedMovement;
    let mut outputs = Vec::new();
    let mut execute_chunk = |hidden: &Array, selections: &Array, weights: &Array| {
        let indexed = MlxTensor::from_array(selections.clone());
        let demands = movement.index_demands(&indexed, global_span, stream)?;
        let backend_demands = demands
            .iter()
            .map(|(member, count)| (ParameterBankKey::new(namespace, *member), *count))
            .collect::<Vec<_>>();
        let acquired = cache.acquire_entry_demand(&backend_demands, pass, stream)?;
        let mapping = demands
            .iter()
            .enumerate()
            .map(|(compact, (source, _))| (*source, compact))
            .collect::<Vec<_>>();
        let compact = movement.remap_indices(&indexed, &mapping, stream)?;
        let output = execute_bank(hidden, &acquired, compact.as_array(), weights, stream)?;
        if output.ndim() == 0 || output.dim(0) != hidden.dim(0) {
            return Err(
                AddressableParameterBankError::CompactBankOutputShapeMismatch {
                    expected_rows: hidden.dim(0),
                    actual: output.shape().to_vec(),
                }
                .into(),
            );
        }
        cache.complete_acquisition(acquired, &output)?;
        Ok(output)
    };
    let mut start = 0;
    while start < row_count {
        let end = (start + chunk_rows).min(row_count);
        let hidden = grouped_hidden.try_index_device(start..end, stream)?;
        let selections = grouped_ids.try_index_device(start..end, stream)?;
        let weights = coefficients.try_index_device(start..end, stream)?;
        outputs.push(execute_chunk(&hidden, &selections, &weights)?);
        start = end;
    }
    if outputs.is_empty() {
        return execute_chunk(grouped_hidden, grouped_ids, coefficients);
    }
    Ok(safemlx::ops::concatenate_axis(&outputs, 0, stream)?)
}

fn wrap_parallel_output(
    output: TensorParallelGroupedOutput<Array>,
) -> TensorParallelGroupedOutput<MlxTensor> {
    TensorParallelGroupedOutput::new(
        MlxTensor::from_array(output.reducible().clone()),
        output.post_reduce().cloned().map(MlxTensor::from_array),
    )
}

fn bank_access_class(pass: ExpertPass) -> Result<BankAccessClass, Error> {
    match pass {
        ExpertPass::Prefill => Ok(BankAccessClass::Bulk),
        ExpertPass::Decode => Ok(BankAccessClass::Incremental),
        _ => Err(Error::ArchitectureModel(
            "unsupported expert pass requires an explicit bank-access policy".into(),
        )),
    }
}

/// Executes independently cached ReLU2 experts through a layer-spec factory.
pub struct CachedRelu2GroupProvider<'a, F> {
    cache: &'a AddressableParameterBank,
    spec_for_layer: F,
}

impl<'a, F> CachedRelu2GroupProvider<'a, F> {
    /// Creates a cached provider using the supplied layer-spec factory.
    pub const fn new(cache: &'a AddressableParameterBank, spec_for_layer: F) -> Self {
        Self {
            cache,
            spec_for_layer,
        }
    }
}

impl<F> RoutedExpertProvider<MlxNeuralBackend> for CachedRelu2GroupProvider<'_, F>
where
    F: FnMut(usize) -> Result<GroupedRelu2Spec, Error>,
{
    type Error = Error;

    fn forward_grouped(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        _request: RoutedExpertRequest<'_, MlxTensor>,
        _stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        Err(Error::ArchitectureModel(
            "a ReLU2 expert cache cannot execute a gated-product expert bank".into(),
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        execute_cached_relu2(
            self.cache,
            &(self.spec_for_layer)(request.layer)?,
            request.layer,
            request.input.as_array(),
            request.routes.group_indices().as_array(),
            request.routes.coefficients().as_array(),
            request.pass,
            stream,
        )
        .map(MlxTensor::from_array)
    }
}

/// Executes independently cached gated-product experts with resident-bank semantics.
pub struct CachedGatedProductGroupProvider<'a> {
    cache: &'a AddressableParameterBank,
}

impl<'a> CachedGatedProductGroupProvider<'a> {
    /// Creates a gated-product provider backed by an expert cache.
    pub const fn new(cache: &'a AddressableParameterBank) -> Self {
        Self { cache }
    }
}

impl RoutedExpertProvider<MlxNeuralBackend> for CachedGatedProductGroupProvider<'_> {
    type Error = Error;

    fn forward_grouped(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        execute_cached_gated_product(
            self.cache,
            resident_bank.spec(),
            request.layer,
            request.input.as_array(),
            request.routes.group_indices().as_array(),
            request.routes.coefficients().as_array(),
            request.pass,
            stream,
        )
        .map(MlxTensor::from_array)
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
        _request: RoutedExpertRequest<'_, MlxTensor>,
        _stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        Err(Error::ArchitectureModel(
            "a gated-product expert cache cannot execute a ReLU2 expert bank".into(),
        ))
    }
}

impl TensorParallelRoutedExpertProvider<MlxNeuralBackend> for CachedGatedProductGroupProvider<'_> {
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, MlxTensor>,
        partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, Self::Error> {
        execute_cached_gated_product_tensor_parallel(
            self.cache,
            resident_bank.spec(),
            request.layer,
            request.input.as_array(),
            request.routes.group_indices().as_array(),
            request.routes.coefficients().as_array(),
            request.pass,
            partitions,
            stream,
        )
        .map(wrap_parallel_output)
        .map(RoutedExpertTensorParallelOutput::Partial)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
        _request: RoutedExpertRequest<'_, MlxTensor>,
        _partitions: usize,
        _stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, Self::Error> {
        Err(Error::ArchitectureModel(
            "a gated-product parameter bank cannot execute a ReLU2 grouped operation".into(),
        ))
    }
}

/// Completion contract requested from a gated-product expert callback.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum GatedProductGroupExecutionMode {
    /// Execute the ordinary bank and return a globally complete output.
    Complete,
    /// Execute a rank-local TP contribution and preserve its post-reduce term.
    TensorParallel {
        /// Partition count supplied by the architecture provider call.
        partitions: usize,
    },
}

/// Architecture-owned geometry and tensors supplied to an expert callback.
pub struct GatedProductGroupExecution {
    /// Global decoder layer requesting expert execution.
    pub layer: usize,
    /// Rank-local bank specification retained by the architecture module.
    pub spec: GroupedGatedProductSpec,
    /// Flattened token rows submitted to the selected experts.
    pub hidden: Array,
    /// Selected global expert identities.
    pub group_indices: Array,
    /// Route weights aligned with `group_indices`.
    pub coefficients: Array,
    /// Required complete or rank-local tensor-parallel result contract.
    pub mode: GatedProductGroupExecutionMode,
}

/// Adapts a callback that explicitly preserves complete versus rank-local TP
/// expert semantics and consumes the architecture-owned local bank geometry.
pub struct GatedProductGroupedExecutorProvider<'a, F> {
    execute: &'a mut F,
}

impl<'a, F> GatedProductGroupedExecutorProvider<'a, F> {
    /// Wraps a callback preserving complete versus rank-local output semantics.
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
            let (reducible, post_reduce) = output.into_parts();
            let reducible = reducible.reshape(original_shape, stream)?;
            let post_reduce = post_reduce
                .map(|value| value.reshape(original_shape, stream))
                .transpose()?;
            Ok(RoutedExpertTensorParallelOutput::Partial(
                TensorParallelGroupedOutput::new(
                    MlxTensor::from_array(reducible),
                    post_reduce.map(MlxTensor::from_array),
                ),
            ))
        }
    }
}

fn execute_gated_product_callback<F>(
    execute: &mut F,
    resident_bank: &<MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
    request: RoutedExpertRequest<'_, MlxTensor>,
    mode: GatedProductGroupExecutionMode,
    stream: &Stream,
) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, safemlx::error::Exception>
where
    F: FnMut(
        GatedProductGroupExecution,
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
    let group_indices = request.routes.group_indices().as_array().reshape(
        &[-1, request.routes.group_indices().as_array().dim(-1)],
        stream,
    )?;
    let coefficients = request.routes.coefficients().as_array().reshape(
        &[-1, request.routes.coefficients().as_array().dim(-1)],
        stream,
    )?;
    let output = execute(
        GatedProductGroupExecution {
            layer: request.layer,
            spec: resident_bank.spec().clone(),
            hidden,
            group_indices,
            coefficients,
            mode,
        },
        stream,
    )?;
    reshape_gated_product_callback_output(output, &original_shape, stream)
}

impl<F> RoutedExpertProvider<MlxNeuralBackend> for GatedProductGroupedExecutorProvider<'_, F>
where
    F: FnMut(
        GatedProductGroupExecution,
        &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<Array>, safemlx::error::Exception>,
{
    type Error = safemlx::error::Exception;

    fn forward_grouped(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        match execute_gated_product_callback(
            self.execute,
            resident_bank,
            request,
            GatedProductGroupExecutionMode::Complete,
            stream,
        )? {
            RoutedExpertTensorParallelOutput::Complete(output) => Ok(output),
            RoutedExpertTensorParallelOutput::Partial(_) => Err(safemlx::error::Exception::custom(
                "ordinary expert execution returned a rank-local tensor-parallel partial",
            )),
        }
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
        _request: RoutedExpertRequest<'_, MlxTensor>,
        _stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        Err(safemlx::error::Exception::custom(
            "a gated-product executor cannot execute a ReLU2 expert bank",
        ))
    }
}

impl<F> TensorParallelRoutedExpertProvider<MlxNeuralBackend>
    for GatedProductGroupedExecutorProvider<'_, F>
where
    F: FnMut(
        GatedProductGroupExecution,
        &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<Array>, safemlx::error::Exception>,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, MlxTensor>,
        partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, Self::Error> {
        execute_gated_product_callback(
            self.execute,
            resident_bank,
            request,
            GatedProductGroupExecutionMode::TensorParallel { partitions },
            stream,
        )
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
        _request: RoutedExpertRequest<'_, MlxTensor>,
        _partitions: usize,
        _stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, Self::Error> {
        Err(safemlx::error::Exception::custom(
            "a gated-product executor cannot execute a ReLU2 grouped operation",
        ))
    }
}

/// Architecture-owned geometry and tensors supplied to a ReLU2 expert callback.
pub struct Relu2GroupExecution {
    /// Global decoder or prediction layer requesting expert execution.
    pub layer: usize,
    /// Rank-local bank specification retained by the architecture module.
    pub spec: GroupedRelu2Spec,
    /// Flattened token rows submitted to the selected experts.
    pub hidden: Array,
    /// Selected global expert identities.
    pub group_indices: Array,
    /// Route weights aligned with `group_indices`.
    pub coefficients: Array,
}

/// Adapts a distributed ReLU2 callback that consumes the resident unit spec.
pub struct Relu2GroupedExecutorProvider<'a, F> {
    execute: &'a mut F,
}

impl<'a, F> Relu2GroupedExecutorProvider<'a, F> {
    /// Wraps a callback that executes a ReLU2 expert request.
    pub const fn new(execute: &'a mut F) -> Self {
        Self { execute }
    }
}

fn execute_relu2_callback<F>(
    execute: &mut F,
    resident_bank: &<MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
    request: RoutedExpertRequest<'_, MlxTensor>,
    stream: &Stream,
) -> Result<MlxTensor, safemlx::error::Exception>
where
    F: FnMut(Relu2GroupExecution, &Stream) -> Result<Array, safemlx::error::Exception>,
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
    let group_indices = request.routes.group_indices().as_array().reshape(
        &[-1, request.routes.group_indices().as_array().dim(-1)],
        stream,
    )?;
    let coefficients = request.routes.coefficients().as_array().reshape(
        &[-1, request.routes.coefficients().as_array().dim(-1)],
        stream,
    )?;
    execute(
        Relu2GroupExecution {
            layer: request.layer,
            spec: resident_bank.spec().clone(),
            hidden,
            group_indices,
            coefficients,
        },
        stream,
    )?
    .reshape(&original_shape, stream)
    .map(MlxTensor::from_array)
}

impl<F> RoutedExpertProvider<MlxNeuralBackend> for Relu2GroupedExecutorProvider<'_, F>
where
    F: FnMut(Relu2GroupExecution, &Stream) -> Result<Array, safemlx::error::Exception>,
{
    type Error = safemlx::error::Exception;

    fn forward_grouped(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        _request: RoutedExpertRequest<'_, MlxTensor>,
        _stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        Err(safemlx::error::Exception::custom(
            "a ReLU2 expert executor cannot execute a gated-product expert bank",
        ))
    }

    fn forward_relu2_routed(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        execute_relu2_callback(self.execute, resident_bank, request, stream)
    }
}

impl<F> TensorParallelRoutedExpertProvider<MlxNeuralBackend> for Relu2GroupedExecutorProvider<'_, F>
where
    F: FnMut(Relu2GroupExecution, &Stream) -> Result<Array, safemlx::error::Exception>,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        _request: RoutedExpertRequest<'_, MlxTensor>,
        _partitions: usize,
        _stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, Self::Error> {
        Err(safemlx::error::Exception::custom(
            "a ReLU2 executor cannot execute a gated-product grouped operation",
        ))
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
        request: RoutedExpertRequest<'_, MlxTensor>,
        _partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, Self::Error> {
        execute_relu2_callback(self.execute, resident_bank, request, stream)
            .map(RoutedExpertTensorParallelOutput::Complete)
    }
}

/// Adapts a distributed callback that consumes the resident rank-local bank.
pub struct ResidentGroupedExecutorProvider<'a, F> {
    execute: &'a mut F,
}

impl<'a, F> ResidentGroupedExecutorProvider<'a, F> {
    /// Wraps a callback that executes against a resident rank-local bank.
    pub const fn new(execute: &'a mut F) -> Self {
        Self { execute }
    }
}

impl<F> RoutedExpertProvider<MlxNeuralBackend> for ResidentGroupedExecutorProvider<'_, F>
where
    F: FnMut(
        &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        &Array,
        &Array,
        &Array,
        usize,
        &Stream,
    ) -> Result<TensorParallelGroupedOutput<Array>, safemlx::error::Exception>,
{
    type Error = safemlx::error::Exception;

    fn forward_grouped(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        let output = (self.execute)(
            resident_bank,
            request.input.as_array(),
            request.routes.group_indices().as_array(),
            request.routes.coefficients().as_array(),
            1,
            stream,
        )?;
        let (reducible, post_reduce) = output.into_parts();
        match post_reduce {
            Some(bias) => reducible.add(&bias, stream).map(MlxTensor::from_array),
            None => Ok(MlxTensor::from_array(reducible)),
        }
    }

    fn forward_relu2_routed(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
        _request: RoutedExpertRequest<'_, MlxTensor>,
        _stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        Err(safemlx::error::Exception::custom(
            "a resident gated-product executor cannot execute a ReLU2 expert bank",
        ))
    }
}

impl<F> TensorParallelRoutedExpertProvider<MlxNeuralBackend>
    for ResidentGroupedExecutorProvider<'_, F>
where
    F: FnMut(
        &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        &Array,
        &Array,
        &Array,
        usize,
        &Stream,
    ) -> Result<TensorParallelGroupedOutput<Array>, safemlx::error::Exception>,
{
    fn forward_grouped_tensor_parallel(
        &mut self,
        resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::GatedProductGroups,
        request: RoutedExpertRequest<'_, MlxTensor>,
        partitions: usize,
        stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, Self::Error> {
        (self.execute)(
            resident_bank,
            request.input.as_array(),
            request.routes.group_indices().as_array(),
            request.routes.coefficients().as_array(),
            partitions,
            stream,
        )
        .map(wrap_parallel_output)
        .map(RoutedExpertTensorParallelOutput::Partial)
    }

    fn forward_relu2_routed_tensor_parallel(
        &mut self,
        _resident_bank: &mut <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::Relu2Groups,
        _request: RoutedExpertRequest<'_, MlxTensor>,
        _partitions: usize,
        _stream: &Stream,
    ) -> Result<RoutedExpertTensorParallelOutput<MlxTensor>, Self::Error> {
        Err(safemlx::error::Exception::custom(
            "a resident gated-product executor cannot execute a ReLU2 grouped operation",
        ))
    }
}

#[cfg(test)]
#[allow(
    clippy::items_after_test_module,
    reason = "cache-provider tests stay adjacent to provider construction"
)]
mod tests {
    use super::*;
    use crate::backend::ExecutionContext;
    use eredu_nn::{GatedProductGroupLayout, GroupSelection};
    use safemlx::{Device, DeviceType};

    fn localized_qwen_spec() -> GroupedGatedProductSpec {
        let args = eredu_architectures::qwen::model_args_from_config_value(&serde_json::json!({
            "model_type": "qwen3_moe",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 0,
            "moe_intermediate_size": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "tie_word_embeddings": false
        }))
        .unwrap();
        eredu_architectures::qwen::expert_bank_spec(&args, 0)
            .unwrap()
            .with_group_geometry(2, 4)
            .unwrap()
    }

    fn test_routes() -> (MlxTensor, GroupSelection<MlxTensor>) {
        let hidden = MlxTensor::from_array(Array::from_slice(&[0.0_f32; 16], &[1, 16]));
        let group_indices = MlxTensor::from_array(Array::from_slice(&[0_i32], &[1, 1]));
        let weights = MlxTensor::from_array(Array::from_slice(&[1.0_f32], &[1, 1]));
        let selected_scores = weights.clone();
        let routes = GroupSelection::new(group_indices, selected_scores, weights);
        (hidden, routes)
    }

    #[test]
    #[ignore = "requires local MLX Metal execution"]
    fn distributed_callbacks_receive_resident_unit_specs() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = execution.stream();
        let gated_spec = localized_qwen_spec();
        let mut gated_bank =
            <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::grouped_gated_product(
                gated_spec.clone(),
                stream,
            )
            .unwrap();
        let (hidden, routes) = test_routes();
        let mut gated_observed = false;
        let mut execute = |execution: GatedProductGroupExecution, _stream: &Stream| {
            assert_eq!(execution.spec.group_count(), gated_spec.group_count());
            assert_eq!(
                execution.spec.intermediate_dimensions(),
                gated_spec.intermediate_dimensions()
            );
            gated_observed = true;
            Ok(RoutedExpertTensorParallelOutput::Complete(execution.hidden))
        };
        let mut provider = GatedProductGroupedExecutorProvider::new(&mut execute);
        provider
            .forward_grouped(
                &mut gated_bank,
                RoutedExpertRequest {
                    layer: 0,
                    input: &hidden,
                    routes: &routes,
                    pass: ExpertPass::Decode,
                },
                stream,
            )
            .unwrap();
        assert!(gated_observed);

        let GatedProductGroupLayout::Packed { gate_up, down } = gated_spec.layout().clone() else {
            panic!("Qwen test specification must use packed experts")
        };
        let relu_spec = GroupedRelu2Spec::new(2, 16, 4, gate_up, down).unwrap();
        let mut relu_bank = <MlxNeuralBackend as eredu_nn::GroupedNeuralBackend>::grouped_relu2(
            relu_spec.clone(),
            stream,
        )
        .unwrap();
        let (hidden, routes) = test_routes();
        let mut relu_observed = false;
        let mut execute = |execution: Relu2GroupExecution, _stream: &Stream| {
            assert_eq!(execution.spec.group_count(), relu_spec.group_count());
            assert_eq!(
                execution.spec.intermediate_dimensions(),
                relu_spec.intermediate_dimensions()
            );
            relu_observed = true;
            Ok(execution.hidden)
        };
        let mut provider = Relu2GroupedExecutorProvider::new(&mut execute);
        provider
            .forward_relu2_routed(
                &mut relu_bank,
                RoutedExpertRequest {
                    layer: 1,
                    input: &hidden,
                    routes: &routes,
                    pass: ExpertPass::Decode,
                },
                stream,
            )
            .unwrap();
        assert!(relu_observed);
    }
}

/// Executes one cached route batch with a compact bank retained by the cache.
#[allow(clippy::too_many_arguments)]
pub fn execute_cached_gated_product(
    cache: &AddressableParameterBank,
    spec: &GroupedGatedProductSpec,
    layer: usize,
    hidden: &Array,
    group_indices: &Array,
    coefficients: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    execute_cached_gated_product_inner(
        cache,
        spec,
        layer,
        hidden,
        group_indices,
        coefficients,
        pass,
        None,
        stream,
    )
}

/// Executes one cached tensor-parallel route batch with exact-once down bias.
#[allow(clippy::too_many_arguments)]
pub fn execute_cached_gated_product_tensor_parallel(
    cache: &AddressableParameterBank,
    spec: &GroupedGatedProductSpec,
    layer: usize,
    hidden: &Array,
    group_indices: &Array,
    coefficients: &Array,
    pass: ExpertPass,
    partitions: usize,
    stream: &Stream,
) -> Result<TensorParallelGroupedOutput<Array>, Error> {
    let original_shape = hidden.shape().to_vec();
    let packed = execute_cached_gated_product_inner(
        cache,
        spec,
        layer,
        hidden,
        group_indices,
        coefficients,
        pass,
        Some(partitions),
        stream,
    )?;
    let output_dimensions = spec.output_dimensions();
    let packed = packed.reshape(&[-1, 2 * output_dimensions], stream)?;
    let reducible = packed
        .try_index_device((.., ..output_dimensions), stream)?
        .reshape(&original_shape, stream)?;
    Ok(TensorParallelGroupedOutput::new(
        reducible,
        packed_gated_product_projections(spec)?
            .1
            .bias()
            .is_some()
            .then(|| {
                packed
                    .try_index_device((.., output_dimensions..), stream)?
                    .reshape(&original_shape, stream)
            })
            .transpose()?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn execute_cached_gated_product_inner(
    cache: &AddressableParameterBank,
    spec: &GroupedGatedProductSpec,
    layer: usize,
    hidden: &Array,
    group_indices: &Array,
    coefficients: &Array,
    pass: ExpertPass,
    partitions: Option<usize>,
    stream: &Stream,
) -> Result<Array, Error> {
    spec.validate()?;
    if spec.input_dimensions() != spec.output_dimensions() {
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
    let output = execute_selections_bounded(
        cache,
        ParameterBankSelection::new(
            layer,
            &flattened,
            group_indices,
            coefficients,
            bank_access_class(pass)?,
        ),
        stream,
        |hidden, acquired, compact_selections, weights, stream| {
            let started = Instant::now();
            let load_time = cache.weight_quantization();
            let mut bank = PackedGatedProductGroups::new(
                acquired.identities().len() as i32,
                spec.input_dimensions(),
                spec.intermediate_dimensions(),
                gate_up
                    .format()
                    .encoding()
                    .weight_quantization()
                    .or(load_time),
                down.format().encoding().weight_quantization().or(load_time),
                [gate_up.bias().is_some(), down.bias().is_some()],
                stream,
            )?
            .with_policy(spec.policy())?;
            bank.gate_up_proj =
                PhysicalParam::new(acquired.compact_binding("gate_up_proj", stream)?);
            bank.gate_up_proj_bias =
                PhysicalParam::new(acquired.optional_compact_binding("gate_up_proj_bias", stream)?);
            bank.gate_up_proj_scales = PhysicalParam::new(
                acquired.optional_compact_binding("gate_up_proj_scales", stream)?,
            );
            bank.gate_up_proj_biases = PhysicalParam::new(
                acquired.optional_compact_binding("gate_up_proj_biases", stream)?,
            );
            bank.down_proj = PhysicalParam::new(acquired.compact_binding("down_proj", stream)?);
            bank.down_proj_bias =
                PhysicalParam::new(acquired.optional_compact_binding("down_proj_bias", stream)?);
            bank.down_proj_scales =
                PhysicalParam::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
            bank.down_proj_biases =
                PhysicalParam::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
            cache.record_compact_bank(
                acquired.pass(),
                acquired.scratch_bytes(),
                started.elapsed(),
            )?;
            Ok(match partitions {
                Some(partitions) => {
                    let output = bank.forward_tensor_parallel(
                        hidden,
                        compact_selections,
                        weights,
                        partitions,
                        stream,
                    )?;
                    let (reducible, post_reduce) = output.into_parts();
                    let post_reduce = match post_reduce {
                        Some(bias) => bias,
                        None => {
                            safemlx::ops::zeros_dtype(reducible.shape(), reducible.dtype(), stream)?
                        }
                    };
                    safemlx::ops::concatenate_axis(&[reducible, post_reduce], -1, stream)?
                }
                None => bank.forward(hidden, compact_selections, weights, stream)?,
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
    cache: &AddressableParameterBank,
    spec: &GroupedRelu2Spec,
    layer: usize,
    hidden: &Array,
    group_indices: &Array,
    coefficients: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    spec.validate()?;
    let original_shape = hidden.shape().to_vec();
    let flattened = hidden.reshape(&[-1, hidden.dim(-1)], stream)?;
    let output = execute_selections_bounded(
        cache,
        ParameterBankSelection::new(
            layer,
            &flattened,
            group_indices,
            coefficients,
            bank_access_class(pass)?,
        ),
        stream,
        |hidden, acquired, compact_selections, weights, stream| {
            let started = Instant::now();
            let load_time = cache.weight_quantization();
            let mut bank = PackedRelu2Groups::new(
                acquired.identities().len() as i32,
                spec.hidden_dimensions(),
                spec.intermediate_dimensions(),
                [
                    spec.up()
                        .format()
                        .encoding()
                        .weight_quantization()
                        .or(load_time),
                    spec.down()
                        .format()
                        .encoding()
                        .weight_quantization()
                        .or(load_time),
                ],
                stream,
            )?;
            bank.up_proj = PhysicalParam::new(acquired.compact_binding("up_proj", stream)?);
            bank.up_proj_scales =
                PhysicalParam::new(acquired.optional_compact_binding("up_proj_scales", stream)?);
            bank.up_proj_biases =
                PhysicalParam::new(acquired.optional_compact_binding("up_proj_biases", stream)?);
            bank.down_proj = PhysicalParam::new(acquired.compact_binding("down_proj", stream)?);
            bank.down_proj_scales =
                PhysicalParam::new(acquired.optional_compact_binding("down_proj_scales", stream)?);
            bank.down_proj_biases =
                PhysicalParam::new(acquired.optional_compact_binding("down_proj_biases", stream)?);
            cache.record_compact_bank(
                acquired.pass(),
                acquired.scratch_bytes(),
                started.elapsed(),
            )?;
            Ok(bank.forward(hidden, compact_selections, weights, stream)?)
        },
    )?;
    Ok(output.reshape(&original_shape, stream)?)
}

/// Executes ReLU2 route rows already compacted by distributed ownership dispatch.
pub fn execute_cached_relu2_dispatched(
    cache: &AddressableParameterBank,
    spec: &GroupedRelu2Spec,
    layer: usize,
    hidden: &Array,
    global_group_indices: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    let group_indices = global_group_indices.reshape(&[-1, 1], stream)?;
    let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
    execute_cached_relu2(
        cache,
        spec,
        layer,
        hidden,
        &group_indices,
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
    cache: &AddressableParameterBank,
    spec: &GroupedGatedProductSpec,
    layer: usize,
    hidden: &Array,
    global_group_indices: &Array,
    pass: ExpertPass,
    stream: &Stream,
) -> Result<Array, Error> {
    let group_indices = global_group_indices.reshape(&[-1, 1], stream)?;
    let weights = safemlx::ops::ones_dtype(&[hidden.dim(0), 1], hidden.dtype(), stream)?;
    execute_cached_gated_product(
        cache,
        spec,
        layer,
        hidden,
        &group_indices,
        &weights,
        pass,
        stream,
    )
}

fn packed_gated_product_projections(
    spec: &GroupedGatedProductSpec,
) -> Result<
    (
        &eredu_nn::GroupedProjectionSpec,
        &eredu_nn::GroupedProjectionSpec,
    ),
    Error,
> {
    match spec.layout() {
        GatedProductGroupLayout::Packed { gate_up, down } => Ok((gate_up, down)),
        GatedProductGroupLayout::Independent(_) => Err(Error::ArchitectureModel(
            "MLX compact cached banks require a packed architecture expert specification".into(),
        )),
        _ => Err(Error::ArchitectureModel(
            "MLX does not implement this grouped parameter-bank layout".into(),
        )),
    }
}
