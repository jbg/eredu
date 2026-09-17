use super::*;

pub(super) fn indexed(
    input: crate::IndexedAttentionInput<'_, WorkspaceTensor>,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    input.validate()?;
    let mut values = context.metadata_vec(
        6 + usize::from(input.local_mask.is_some())
            + usize::from(input.pooled_mask.is_some())
            + usize::from(input.sinks.is_some()),
    )?;
    values.extend([
        input.queries,
        input.local_keys,
        input.local_values,
        input.pooled_keys,
        input.pooled_values,
        input.selected_positions,
    ]);
    values.extend(input.local_mask);
    values.extend(input.pooled_mask);
    values.extend(input.sinks);
    let mut shape = context.metadata_vec(input.queries.shape().len())?;
    shape.extend_from_slice(input.queries.shape());
    shape[3] = input.local_values.shape()[2];
    WorkspaceTensor::operation(
        WorkspaceOperationKind::IndexedAttention {
            scale: input.scale,
            local_mask: input.local_mask.is_some(),
            pooled_mask: input.pooled_mask.is_some(),
            sinks: input.sinks.is_some(),
        },
        &values,
        &shape,
        WorkspaceDtype::Float32,
        context,
    )
}

pub(super) fn pooled(
    input: crate::PooledAttentionInput<'_, WorkspaceTensor>,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let query = input.queries.shape();
    let local = input.local.shape();
    let pooled = input.pooled.shape();
    if query.len() != 4
        || local.len() != 3
        || pooled.len() != 3
        || query[0] != local[0]
        || query[0] != pooled[0]
        || query[3] != local[2]
        || query[3] != pooled[2]
        || !input.scale.is_finite()
        || input.scale <= 0.
    {
        return Err(context.metadata_error(format_args!("invalid pooled attention geometry")));
    }
    let mut values = context.metadata_vec(
        3 + usize::from(input.local_mask.is_some())
            + usize::from(input.pooled_mask.is_some())
            + usize::from(input.sinks.is_some()),
    )?;
    values.extend([input.queries, input.local, input.pooled]);
    values.extend(input.local_mask);
    values.extend(input.pooled_mask);
    values.extend(input.sinks);
    WorkspaceTensor::operation(
        WorkspaceOperationKind::PooledAttention {
            scale: input.scale,
            local_mask: input.local_mask.is_some(),
            pooled_mask: input.pooled_mask.is_some(),
            sinks: input.sinks.is_some(),
        },
        &values,
        query,
        WorkspaceDtype::Float32,
        context,
    )
}

pub(super) fn positions(
    input: crate::PooledPositionInput<'_, WorkspaceTensor>,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let query = input.queries.shape();
    let pooled = input.pooled_keys.shape();
    let weights = input.head_weights.shape();
    if query.len() != 4
        || pooled.len() != 3
        || weights.len() != 3
        || query[0] != pooled[0]
        || weights != [query[0], query[2], query[1]]
        || query[3] != pooled[2]
        || input.top_k <= 0
        || !input.scale.is_finite()
        || input.scale <= 0.
        || !input.head_scale.is_finite()
        || input.head_scale <= 0.
    {
        return Err(
            context.metadata_error(format_args!("invalid pooled position selection geometry"))
        );
    }
    let mut values = context.metadata_vec(3 + usize::from(input.mask.is_some()))?;
    values.extend([input.queries, input.pooled_keys, input.head_weights]);
    values.extend(input.mask);
    WorkspaceTensor::operation(
        WorkspaceOperationKind::PooledPositions {
            top_k: input.top_k,
            scale: input.scale,
            head_scale: input.head_scale,
            masked: input.mask.is_some(),
        },
        &values,
        &[query[0], query[2], input.top_k.min(pooled[1])],
        WorkspaceDtype::Uint32,
        context,
    )
}

pub(super) fn gather(
    mask: &WorkspaceTensor,
    positions: &WorkspaceTensor,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let shape = positions.shape();
    if mask.shape().len() != 2
        || shape.len() != 3
        || mask.shape()[0] != shape[1]
        || !matches!(
            positions.layout().dtype(),
            WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
        )
    {
        return Err(context.metadata_error(format_args!("invalid pooled mask gather geometry")));
    }
    WorkspaceTensor::operation(
        WorkspaceOperationKind::GatherPooledMask,
        &[mask, positions],
        &[shape[0], 1, shape[1], shape[2]],
        mask.layout().dtype(),
        context,
    )
}
