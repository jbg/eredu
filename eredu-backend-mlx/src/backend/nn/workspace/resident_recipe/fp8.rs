//! Actual ungrouped block-FP8 worker and its floating replacement alternative.
use super::*;

pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let WorkspaceOperationKindView::Projection(format) = operation.kind else {
        return None;
    };
    let eredu_checkpoint::LinearFormat::E4M3BlockFp8(config) = format.encoding() else {
        return None;
    };
    if config.block_rows != 128
        || config.block_columns != 128
        || format.row_layout() != eredu_nn::LinearRowLayout::Contiguous
    {
        return None;
    }
    let input = operation.inputs.get(0)?;
    let weight = operation.inputs.get(1)?;
    if input.dtype() != WorkspaceDtype::Float32 || operation.outputs.len() != 1 {
        return None;
    }
    crate::backend::nn::fp8::original::geometry(input.shape(), weight.shape())?;
    crate::backend::nn::fp8::original::control_bytes()?;
    // FP8: input reshape, two quantizer sibling descriptors, four-input
    // projection, output reshape and dtype restoration: 6 nodes / 11 edges.
    // Ue8m0: source table, cast, gather flatten/restore: 6 / 7 plus one seed.
    // Optional bias: 5 / 6. Retain the existing floating replacement branch
    // (14 / 16 plus its scalar seed), selected by PhysicalLinear itself.
    let mut value = Lowering::product(31, 40, 2, 1);
    // Both fixed custom kernels can compact their actual operands: one input
    // to quantization and four inputs to projection. Dense split/compaction
    // alternatives remain included by Lowering::product.
    value.maximum_births = value.maximum_births.checked_add(5)?;
    // Taking from the scale table with rank-2 indices creates rank-3 Gather.
    value.intermediate_rank = 3;
    value.backend_shells = 4; // retained scale and quantized/output handoffs
    value.unqualified_kernel_owner =
        (!crate::backend::nn::fp8::kernel::source_qualified())
            .then_some(CustomKernelOwner::BlockFp8);
    if bf16_grouped_width(*input.shape().last()?) {
        value.bf16_projection_calls = 1;
        if value.unqualified_kernel_owner.is_none() {
            value.unqualified_kernel_owner = bf16_projection_source_requirement();
        }
    }
    Some(value)
}

/// The same physical worker split at its actual callback. Reconstruction
/// operations stay separate; these two phases never stand in for the factory.
pub(super) fn observed(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    use WorkspaceOperationKindView as K;
    let (format, prepare) = match operation.kind {
        K::ProjectionPrepare(format) => (format, true),
        K::ProjectionFinish(format) => (format, false),
        _ => return None,
    };
    let eredu_checkpoint::LinearFormat::E4M3BlockFp8(config) = format.encoding() else {
        return None;
    };
    if config.block_rows != 128
        || config.block_columns != 128
        || format.row_layout() != eredu_nn::LinearRowLayout::Contiguous
    {
        return None;
    }
    let input = operation.inputs.get(0)?;
    let weight = operation.inputs.get(1)?;
    if input.dtype() != WorkspaceDtype::Float32 {
        return None;
    }
    crate::backend::nn::fp8::original::geometry(input.shape(), weight.shape())?;
    let plan = eredu_nn::BlockFp8InputReconstructionPlan::new(input.shape()).ok()?;
    let original_inputs = if prepare {
        operation.inputs.len()
    } else {
        operation.inputs.len().checked_sub(2)?
    };
    if original_inputs < 3 {
        return None;
    }
    let prepared = if prepare {
        operation.outputs
    } else {
        operation
            .inputs
            .slice(original_inputs..operation.inputs.len())?
    };
    if prepared.len() != 2
        || prepared.get(0)?.dtype() != WorkspaceDtype::Uint8
        || prepared.get(1)?.dtype() != WorkspaceDtype::Float32
        || plan
            .validate_operands(prepared.get(0)?.shape(), prepared.get(1)?.shape())
            .is_err()
        || (!prepare && operation.outputs.len() != 1)
    {
        return None;
    }
    let mut value = if prepare {
        // Input reshape and two quantizer siblings: 3 descriptors / 3 edges.
        // Ue8m0 scale table/cast/take with flatten/restore: 6 / 7 plus one seed.
        // The possible quantizer input compaction is a distinct physical birth.
        let mut v = Lowering::plain(9, 10, 1);
        v.maximum_births = v.maximum_births.checked_add(1)?;
        v.backend_shells = 3; // scale alias and both compact-root handoffs
        v
    } else {
        // Four-input custom projection, output reshape/dtype, optional bias:
        // 8 / 12. The same actual PhysicalLinear floating replacement remains
        // a conservative alternative: 14 / 16 plus its scalar seed.
        let mut v = Lowering::product(22, 28, 1, 1);
        v.maximum_births = v.maximum_births.checked_add(4)?;
        v.backend_shells = 1;
        if bf16_grouped_width(*input.shape().last()?) {
            v.bf16_projection_calls = 1;
        }
        v
    };
    value.intermediate_rank = if prepare { 3 } else { 2 };
    value.unqualified_kernel_owner =
        (!crate::backend::nn::fp8::kernel::source_qualified())
            .then_some(CustomKernelOwner::BlockFp8);
    if value.bf16_projection_calls != 0 && value.unqualified_kernel_owner.is_none() {
        value.unqualified_kernel_owner = bf16_projection_source_requirement();
    }
    Some(value)
}

pub(super) fn decode(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let input = operation.inputs.get(0)?;
    let output = operation.outputs.get(0)?;
    if operation.inputs.len() != 1
        || operation.outputs.len() != 1
        || input.dtype() != WorkspaceDtype::Uint8
        || output.dtype() != WorkspaceDtype::Float32
        || input.shape().len() != 2
        || input.shape() != output.shape()
        || input.shape().iter().any(|&n| n <= 0)
    {
        return None;
    }
    // ops.cpp::from_fp8 creates one ConvertFP8 with its actual compact input.
    // Metal delegates to the ordinary unary kernel, with one F32 destination.
    Some(Lowering::plain(1, 1, 0))
}

pub(super) fn control_bytes(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    use WorkspaceOperationKindView as K;
    match operation.kind {
        K::Projection(_) => crate::backend::nn::fp8::original::control_bytes(),
        K::ProjectionPrepare(_) => {
            // Prepare enters the enclosing native function and offers the
            // retained factory; these frames live through callback and finish.
            crate::backend::nn::fp8::original::control_bytes()?
                .checked_add(crate::backend::nn::fp8::original::reconstruction_control_bytes()?)?
                .checked_add(crate::backend::nn::shared::projection_observation_control_bytes()?)
        }
        // The shared primitive recipes already price these native descriptors.
        K::ProjectionFinish(_) | K::BlockFp8ActivationDecode => Some(0),
        _ => None,
    }
}

/// Projection-specific contribution inside the existing selected-bank equation.
/// The enclosing reducer adds the unchanged bias/activation/sort scratch.
pub(super) fn grouped_linear(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let WorkspaceOperationKindView::Grouped {
        bank: WorkspaceGroupedBank::Linear(spec),
        phase: WorkspaceGroupedPhase::Whole,
        partitions: None,
    } = operation.kind
    else {
        return None;
    };
    let format = spec.projection().format();
    let eredu_checkpoint::LinearFormat::E4M3BlockFp8(config) = format.encoding() else {
        return None;
    };
    if config.block_rows != 128
        || config.block_columns != 128
        || format.row_layout() != eredu_nn::LinearRowLayout::Contiguous
    {
        return None;
    }
    let input = operation.inputs.get(0)?;
    let ids = operation.inputs.get(1)?;
    // The grouped descriptor owns input, IDs, selected scores and coefficients,
    // followed by actual weight/companions/bias; do not synthesize dimensions.
    let weight = operation.inputs.get(4)?;
    let scales = operation.inputs.get(5)?;
    if input.dtype() != WorkspaceDtype::Float32
        || input.shape().len() != 2
        || ids.shape().len() != 2
        || ids.shape()[0] != input.shape()[0]
        || operation.outputs.len() != 1
        || weight.dtype() != WorkspaceDtype::Uint8
        || scales.dtype()
            != match config.scale_encoding {
                eredu_checkpoint::BlockFp8ScaleEncoding::FloatingPoint => WorkspaceDtype::Float32,
                eredu_checkpoint::BlockFp8ScaleEncoding::Ue8m0 => WorkspaceDtype::Uint8,
            }
    {
        return None;
    }
    let routes = input.shape()[0].checked_mul(ids.shape()[1])?;
    crate::backend::nn::fp8::original::grouped_geometry(
        &[routes, input.shape()[1]],
        weight.shape(),
        scales.shape(),
        &[routes],
    )?;
    grouped_control_bytes()?;
    // Common equation excluding dense projection: 75 / 87 / 5. Actual FP8:
    // two quantizer sibling nodes / two edges; five-input grouped kernel;
    // optional dtype restoration; E8M0 table + cast + take flatten/gather/
    // restore (six nodes, seven edges, one seed). Combined 85 / 102 / 6.
    // Two shape-only partition views feed the same source-derived family.
    let mut value = Lowering::plain(87, 104, 6);
    value.validations = 1; // selected IDs; no BF16 grouped-row validation
    value.maximum_operands = 5;
    value.backend_shells = 3; // decoded scale alias and quantizer handoffs
    // Rank-3 scale indices create a rank-4 Gather before its axis squeeze.
    value.intermediate_rank = 4;
    value.unqualified_kernel_owner =
        (!crate::backend::nn::fp8::kernel::source_qualified())
            .then_some(CustomKernelOwner::BlockFp8);
    Some(value)
}
pub(super) fn grouped_control_bytes() -> Option<usize> {
    crate::backend::nn::fp8::original::grouped_control_bytes()?
        .checked_add(crate::backend::nn::shared::MlxGroupedLinear::original_fp8_control_bytes()?)
}

/// Validates both actual packed projection banks in declared parameter order.
/// This supplies a projection fact to the existing common chunk/activation loop.
pub(super) fn packed_sources(operation: WorkspaceOperationView<'_>) -> Option<()> {
    let WorkspaceOperationKindView::Grouped {
        bank: WorkspaceGroupedBank::GatedProduct(spec),
        partitions: None,
        phase,
    } = operation.kind
    else {
        return None;
    };
    let eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } = spec.layout() else {
        return None;
    };
    let input = operation.inputs.get(0)?;
    let ids = operation.inputs.get(1)?;
    let width = *input.shape().last()?;
    if input.dtype() != WorkspaceDtype::Float32
        || width != spec.input_dimensions()
        || width <= 0
        || operation.outputs.len()
            != if phase == WorkspaceGroupedPhase::Units {
                4
            } else {
                1
            }
    {
        return None;
    }
    let tokens = i32::try_from(input.elements().ok()?.checked_div(width as u64)?).ok()?;
    if ids.shape().len() != 2 || ids.shape()[0] != tokens || ids.shape()[1] <= 0 {
        return None;
    }
    use crate::backend::nn::grouped::{
        GROUPED_PROJECTION_CHUNK_THRESHOLD as THRESHOLD, GROUPED_PROJECTION_CHUNK_TOKENS as CHUNK,
    };
    let chunk_tokens = if tokens > THRESHOLD { CHUNK } else { tokens };
    let routes = chunk_tokens.checked_mul(ids.shape()[1])?;
    if gate_up.format().encoding() != down.format().encoding() {
        return None;
    }
    let mut slot = 4usize;
    for (projection, columns, rows) in [
        (
            gate_up,
            spec.input_dimensions(),
            spec.intermediate_dimensions().checked_mul(2)?,
        ),
        (
            down,
            spec.intermediate_dimensions(),
            spec.output_dimensions(),
        ),
    ] {
        let eredu_checkpoint::LinearFormat::E4M3BlockFp8(config) = projection.format().encoding()
        else {
            return None;
        };
        if config.block_rows != 128
            || config.block_columns != 128
            || projection.format().affine_bias().is_some()
        {
            return None;
        }
        let weight = operation.inputs.get(slot)?;
        let scales = operation.inputs.get(slot.checked_add(1)?)?;
        if weight.shape() != [spec.group_count(), rows, columns]
            || weight.dtype() != WorkspaceDtype::Uint8
            || scales.dtype()
                != match config.scale_encoding {
                    eredu_checkpoint::BlockFp8ScaleEncoding::FloatingPoint => {
                        WorkspaceDtype::Float32
                    }
                    eredu_checkpoint::BlockFp8ScaleEncoding::Ue8m0 => WorkspaceDtype::Uint8,
                }
        {
            return None;
        }
        crate::backend::nn::fp8::original::grouped_geometry_with_layout(
            &[routes, columns],
            weight.shape(),
            scales.shape(),
            &[routes],
            projection.format().row_layout(),
        )?;
        slot = slot.checked_add(2)?;
        if projection.bias().is_some() {
            let bias = operation.inputs.get(slot)?;
            if bias.shape() != [spec.group_count(), rows] || bias.dtype() != WorkspaceDtype::Float32
            {
                return None;
            }
            slot = slot.checked_add(1)?;
        }
    }
    // The native down worker is the contiguous grouped projection; gate/up
    // alone has independent component-major origins in this retained module.
    if down.format().row_layout() != eredu_nn::LinearRowLayout::Contiguous
        || operation.inputs.len()
            != slot.checked_add(if phase == WorkspaceGroupedPhase::Finish {
                4
            } else {
                0
            })?
    {
        return None;
    }
    Some(())
}
pub(super) fn packed_control_bytes(operation: WorkspaceOperationView<'_>) -> Option<usize> {
    packed_sources(operation)?;
    let input = operation.inputs.get(0)?;
    let tokens = usize::try_from(
        input
            .elements()
            .ok()?
            .checked_div(*input.shape().last()? as u64)?,
    )
    .ok()?;
    use crate::backend::nn::grouped::{
        GROUPED_PROJECTION_CHUNK_THRESHOLD as THRESHOLD, GROUPED_PROJECTION_CHUNK_TOKENS as CHUNK,
    };
    let chunks = if tokens > THRESHOLD as usize {
        tokens.div_ceil(CHUNK as usize)
    } else {
        1
    };
    let WorkspaceOperationKindView::Grouped { phase, .. } = operation.kind else {
        return None;
    };
    let calls = if phase == WorkspaceGroupedPhase::Whole {
        2
    } else {
        1
    };
    crate::backend::nn::fp8::original::grouped_control_bytes()?
        .checked_mul(chunks.checked_mul(calls)?)?
        .checked_add(if phase == WorkspaceGroupedPhase::Whole {
            crate::backend::nn::shared::MlxGroupedGatedProduct::original_fp8_control_bytes()?
        } else {
            0
        })
}
