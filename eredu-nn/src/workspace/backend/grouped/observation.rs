use super::*;

/// Join logical chunk geometry while retaining every possible physical owner.
/// This is metadata for the enclosing Finish equation, not an execution-time
/// concatenation. Original units already carry the sum of all chunk capacities;
/// callback-created allocations remain in the shared completed-span ledger.
pub(super) fn trace(
    batch: GroupedUnitBatch<'_, WorkspaceTensor>,
    schedule: WorkspaceGroupedObservationSchedule,
    observer: &mut dyn GroupedUnitObserver<WorkspaceTensor>,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let chunk = match schedule {
        WorkspaceGroupedObservationSchedule::WholeBatch => {
            return deliver(batch, observer, context)
        }
        WorkspaceGroupedObservationSchedule::TokenChunks(tokens) => tokens.get() as usize,
    };
    if batch.total_token_count <= chunk {
        return deliver(batch, observer, context);
    }
    let k = batch.coefficients.shape()[1] as usize;
    let mut owners = context.metadata_vec(batch.total_token_count.div_ceil(chunk))?;
    for start in (0..batch.total_token_count).step_by(chunk) {
        let end = start.saturating_add(chunk).min(batch.total_token_count);
        // Units has already checked that the entire selected-row count fits i32.
        // The slice is a logical view into separately sorted chunk stores.
        let routes = [Index::Range((start * k) as i32, (end * k) as i32)];
        let values = batch.values.index(&routes, context)?;
        let group_indices = batch.group_indices.index(&routes, context)?;
        let selection_indices = batch.selection_indices.index(&routes, context)?;
        let token_indices = batch.token_indices.index(&routes, context)?;
        let coefficients = batch
            .coefficients
            .index(&[Index::Range(start as i32, end as i32)], context)?;
        let effective = deliver(
            GroupedUnitBatch {
                values: &values,
                group_indices: &group_indices,
                selection_indices: &selection_indices,
                token_indices: &token_indices,
                coefficients: &coefficients,
                token_offset: start,
                total_token_count: batch.total_token_count,
                group_count: batch.group_count,
            },
            observer,
            context,
        )?;
        owners.push(effective.storage);
    }
    Ok(WorkspaceTensor {
        imported_existing: false,
        layout: batch.values.layout.clone(),
        storage: context.try_new_storage(Some(0), owners)?,
        context: context.identity.clone(),
    })
}

fn deliver(
    batch: GroupedUnitBatch<'_, WorkspaceTensor>,
    observer: &mut dyn GroupedUnitObserver<WorkspaceTensor>,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    observer.observe(&batch)?;
    let effective = observer
        .intervene(&batch)?
        .unwrap_or_else(|| batch.values.clone());
    if effective.shape() != batch.values.shape() {
        let mut expected = context.metadata_vec(batch.values.shape().len())?;
        expected.extend_from_slice(batch.values.shape());
        let mut actual = context.metadata_vec(effective.shape().len())?;
        actual.extend_from_slice(effective.shape());
        return Err(
            context.metadata_source(GroupedUnitError::ReplacementShape { expected, actual })
        );
    }
    if effective.layout.dtype != batch.values.layout.dtype {
        return Err(context.metadata_source(GroupedUnitError::ReplacementDtype));
    }
    context.validate_values([&effective])?;
    observer.observe_effective(&batch.with_values(&effective))?;
    Ok(effective)
}
