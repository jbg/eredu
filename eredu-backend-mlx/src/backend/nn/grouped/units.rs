use super::*;

/// Native bridge to an admitted neutral observer. Returned values are the actual
/// input of the selected down projection, before its optional input transform.
pub(crate) trait NativeGroupedUnitObserver {
    fn apply(&mut self, batch: &eredu_nn::GroupedUnitBatch<'_, Array>) -> Result<Array, Exception>;
}

pub(super) fn observe_units(
    values: Array,
    plan: &GroupedSelectionPlan,
    coefficients: &Array,
    token_offset: usize,
    total_token_count: usize,
    group_count: usize,
    observer: &mut Option<&mut dyn NativeGroupedUnitObserver>,
) -> Result<Array, Exception> {
    let Some(observer) = observer else {
        return Ok(values);
    };
    observer.apply(&eredu_nn::GroupedUnitBatch {
        values: &values,
        group_indices: &plan.sorted_group_ids,
        selection_indices: &plan.selection_indices,
        token_indices: &plan.token_indices,
        coefficients,
        token_offset,
        total_token_count,
        group_count,
    })
}
