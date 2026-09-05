//! Test-only bounded selection harness for generic addressable parameter banks.

use eredu_runtime::IndexedMovement;
use safemlx::{ops::indexing::TryIndexOp, Array, Stream};

use crate::backend::error::Error;
use crate::backend::runtime::residency::parameter_bank::{
    AcquiredParameterGroups, AddressableParameterBank, AddressableParameterBankError,
    BankAccessClass, MlxIndexedMovement, ParameterBankKey,
};
use crate::MlxTensor;

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

/// Exercises bounded row partitioning and compact-id remapping independently of
/// any architecture or model-family adapter.
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
