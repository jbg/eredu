//! MLX realization of backend-neutral routed-expert providers.
use std::time::Instant;

use eredu_nn::{
    GatedProductGroupLayout, GroupedGatedProductOperator, GroupedGatedProductSpec,
    GroupedRelu2Spec, TensorParallelGroupedOutput,
};
use eredu_runtime::{
    IndexedMovement, ParameterBankAccess, RoutedExpertProvider, RoutedExpertRequest,
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

fn bank_access_class(access: ParameterBankAccess) -> Result<BankAccessClass, Error> {
    match access {
        ParameterBankAccess::Bulk => Ok(BankAccessClass::Bulk),
        ParameterBankAccess::Incremental => Ok(BankAccessClass::Incremental),
        _ => Err(Error::ArchitectureModel(
            "unsupported parameter-bank access class".into(),
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
            request.parameter_bank_access(),
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
            request.parameter_bank_access(),
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
            request.parameter_bank_access(),
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

#[cfg(test)]
#[allow(
    clippy::items_after_test_module,
    reason = "cache-provider tests stay adjacent to provider construction"
)]
mod tests {
    use super::*;

    #[test]
    fn parameter_bank_access_classes_are_exact_storage_inputs() {
        assert_eq!(
            bank_access_class(ParameterBankAccess::Bulk).unwrap(),
            BankAccessClass::Bulk
        );
        assert_eq!(
            bank_access_class(ParameterBankAccess::Incremental).unwrap(),
            BankAccessClass::Incremental
        );
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
    access: ParameterBankAccess,
    stream: &Stream,
) -> Result<Array, Error> {
    execute_cached_gated_product_inner(
        cache,
        spec,
        layer,
        hidden,
        group_indices,
        coefficients,
        access,
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
    access: ParameterBankAccess,
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
        access,
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
    access: ParameterBankAccess,
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
            bank_access_class(access)?,
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
    access: ParameterBankAccess,
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
            bank_access_class(access)?,
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
