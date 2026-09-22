//! Safe calls made by the selected shared pooled-attention adapter.
use super::*;

pub(super) fn attention(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    let WorkspaceOperationKindView::PooledAttention {
        local_mask,
        pooled_mask,
        sinks,
        ..
    } = operation.kind
    else {
        return None;
    };
    // The owning Metal emitter has already validated query/bank/output geometry
    // and the actual native SDPA numerical source. Keep mask normalization tied
    // to the same shape worker used by MlxNeuralBackend::pooled_attention.
    if operation.inputs.len()
        != 3 + usize::from(local_mask) + usize::from(pooled_mask) + usize::from(sinks)
    {
        return None;
    }
    let local = local_mask.then(|| operation.inputs.get(3)).flatten();
    let pooled = pooled_mask
        .then(|| operation.inputs.get(3 + usize::from(local_mask)))
        .flatten();
    let shapes = crate::backend::nn::attention::pooled_mask_shapes_fixed(
        operation.inputs.get(0)?.shape(),
        *operation.inputs.get(1)?.shape().get(1)?,
        *operation.inputs.get(2)?.shape().get(1)?,
        local.map(|mask| mask.shape()),
        pooled.map(|mask| mask.shape()),
    )
    .ok()?;
    let call = OrdinaryCallControls::call;
    // Both bank owners are transferred into a fixed two-element source array.
    // Concatenate's C ArrayVector is observed by the actual Join wrapper query.
    let mut result = call(OrdinaryRecipeCall::ExpandDims)?
        .repeat(2)?
        .append(call(OrdinaryRecipeCall::Join {
            inputs: 2,
            stack: false,
        })?)?;
    if shapes.is_some() {
        let masks = [
            local.map(|mask| mask.dtype()),
            pooled.map(|mask| mask.dtype()),
        ];
        if masks
            .into_iter()
            .flatten()
            .any(|dtype| !matches!(dtype, WorkspaceDtype::Bool | WorkspaceDtype::Float32))
        {
            return None;
        }
        let additive = masks.contains(&Some(WorkspaceDtype::Float32));
        for mask in masks {
            result = match mask {
                Some(WorkspaceDtype::Bool) if additive => result
                    .append(call(OrdinaryRecipeCall::ScalarF32)?.repeat(2)?)?
                    .append(call(OrdinaryRecipeCall::Select)?)?,
                Some(_) => result.metadata(Array::ordinary_clone_control_bytes()?)?,
                None => result.append(call(if additive {
                    OrdinaryRecipeCall::ScalarF32
                } else {
                    OrdinaryRecipeCall::ScalarBool
                })?)?,
            };
            result = result.append(call(OrdinaryRecipeCall::Broadcast { rank: 4 })?)?;
        }
        result = result.append(call(OrdinaryRecipeCall::Join {
            inputs: 2,
            stack: false,
        })?)?;
        if additive {
            result = result.append(call(OrdinaryRecipeCall::Cast)?)?;
        }
    }
    // This adapter directly borrows the joined mask and sinks into fast SDPA;
    // it does not pass through the separate ordinary AttentionRequest adapter.
    result
        .metadata(safemlx::fast::ordinary_sdpa_control_bytes()?)?
        .metadata(crate::backend::nn::attention::pooled_attention_control_bytes(shapes.is_some())?)
}

/// The fixed contraction and score worker explicitly casts all three floating
/// operands to F32 before selection, so its caller source preserves unknown
/// input precision without choosing another numerical implementation.
pub(super) fn positions(operation: WorkspaceOperationView<'_>) -> Option<OrdinaryCallControls> {
    let (geometry, masked) = super::super::super::pooling::positions_geometry(operation).ok()?;
    if operation
        .inputs
        .slice(0..3)?
        .iter()
        .any(|value| value.dtype() != WorkspaceDtype::Float32)
        || masked && operation.inputs.get(3)?.dtype() != WorkspaceDtype::Bool
    {
        return None;
    }
    let call = OrdinaryCallControls::call;
    let mut result = if geometry.pooled == 0 {
        call(OrdinaryRecipeCall::Fill { rank: 3 })?
    } else {
        call(OrdinaryRecipeCall::Cast)?
            .repeat(3)?
            .append(call(OrdinaryRecipeCall::Broadcast { rank: 4 })?)?
            .append(call(OrdinaryRecipeCall::Broadcast { rank: 3 })?)?
            .append(call(OrdinaryRecipeCall::Transpose { rank: 4 })?.repeat(2)?)?
            .append(call(OrdinaryRecipeCall::Transpose { rank: 3 })?.repeat(2)?)?
            .append(call(OrdinaryRecipeCall::Reshape { rank: 3 })?.repeat(2)?)?
            .append(call(OrdinaryRecipeCall::Reshape { rank: 4 })?)?
            .append(call(OrdinaryRecipeCall::Binary)?.repeat(5)?)?
            .append(call(OrdinaryRecipeCall::ScalarF32)?.repeat(3)?)?
            .append(call(OrdinaryRecipeCall::ExpandDims)?)?
            .append(call(OrdinaryRecipeCall::ReduceAxis)?)?
            .append(call(OrdinaryRecipeCall::ArgPartitionAxis)?)?
            .append(call(OrdinaryRecipeCall::StaticSlice { rank: 3 })?)?
    };
    if geometry.pooled != 0 && masked {
        result = result
            .append(call(OrdinaryRecipeCall::ScalarF32)?)?
            .append(call(OrdinaryRecipeCall::Select)?)?;
    }
    result
        .metadata(crate::backend::nn::attention::pooled_positions::control_bytes(geometry, masked)?)
}
