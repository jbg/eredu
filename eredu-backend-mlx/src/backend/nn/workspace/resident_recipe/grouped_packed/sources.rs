//! Parameter roles of the shared dense, affine and MXFP4 grouped worker.
use super::*;

/// Complete source roles in the same parameter order consumed by the factory.
/// Floating replacement geometry belongs to the retained parameter publication;
/// these are the original packed companions, never reinterpreted as dense bytes.
pub(super) fn counts(operation: WorkspaceOperationView<'_>) -> Option<[usize; 3]> {
    use eredu_checkpoint::LinearFormat;
    let WorkspaceOperationKindView::Grouped {
        bank: WorkspaceGroupedBank::GatedProduct(spec),
        phase,
        partitions,
    } = operation.kind
    else {
        return None;
    };
    spec.validate_fixed().ok()?;
    let eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } = spec.layout() else {
        return None;
    };
    let input = operation.inputs.get(0)?;
    let ids = operation.inputs.get(1)?;
    let columns = spec.input_dimensions();
    if columns <= 0
        || input.dtype() != WorkspaceDtype::Float32
        || input.shape().last() != Some(&columns)
        || partitions == Some(0)
        || operation.outputs.len()
            != if phase == WorkspaceGroupedPhase::Units {
                4
            } else if partitions.is_some() && down.bias().is_some() {
                2
            } else {
                1
            }
    {
        return None;
    }
    let tokens = usize::try_from(input.elements().ok()?.checked_div(columns as u64)?).ok()?;
    if ids.shape().len() != 2
        || usize::try_from(ids.shape()[0]).ok()? != tokens
        || ids.shape()[1] <= 0
        || !matches!(ids.dtype(), WorkspaceDtype::Uint32 | WorkspaceDtype::Int32)
    {
        return None;
    }
    let mut slot = 4usize;
    // MXFP4 gather, affine gather, affine selected-bank QMM.
    let mut calls = [0usize; 3];
    for (projection, k, n, active) in [
        (
            gate_up,
            columns,
            spec.intermediate_dimensions().checked_mul(2)?,
            phase != WorkspaceGroupedPhase::Finish,
        ),
        (
            down,
            spec.intermediate_dimensions(),
            spec.output_dimensions(),
            phase != WorkspaceGroupedPhase::Units,
        ),
    ] {
        if projection.format().row_layout() != eredu_nn::LinearRowLayout::Contiguous {
            return None;
        }
        let weight = operation.inputs.get(slot)?;
        slot = slot.checked_add(1)?;
        match projection.format().encoding() {
            LinearFormat::MxFp4 => {
                let scales = operation.inputs.get(slot)?;
                slot = slot.checked_add(1)?;
                if k <= 0
                    || k % 32 != 0
                    || projection.format().scale().is_none()
                    || projection.format().affine_bias().is_some()
                    || weight.dtype() != WorkspaceDtype::Uint32
                    || weight.shape() != [spec.group_count(), n, k / 8]
                    || scales.dtype() != WorkspaceDtype::Uint8
                    || scales.shape() != [spec.group_count(), n, k / 32]
                {
                    return None;
                }
                calls[0] = calls[0].checked_add(usize::from(active))?;
            }
            LinearFormat::Affine(config) => {
                config.validate_fixed().ok()?;
                let scales = operation.inputs.get(slot)?;
                let biases = operation.inputs.get(slot.checked_add(1)?)?;
                slot = slot.checked_add(2)?;
                let packed = k.checked_mul(config.bits)?;
                if k <= 0 || k % config.group_size != 0 || packed % 32 != 0
                    || projection.format().scale().is_none()
                    || projection.format().affine_bias().is_none()
                    || weight.dtype() != WorkspaceDtype::Uint32
                    || weight.shape() != [spec.group_count(), n, packed / 32]
                    || [scales, biases].iter().any(|value|
                        value.dtype() != WorkspaceDtype::Float32
                        || value.shape() != [spec.group_count(), n, k / config.group_size])
                { return None; }
                let kind = if config.group_size == 16 { 2 } else { 1 };
                calls[kind] = calls[kind].checked_add(usize::from(active))?;
            }
            LinearFormat::Dense => {
                if projection.format().scale().is_some()
                    || projection.format().affine_bias().is_some()
                    || weight.dtype() != WorkspaceDtype::Float32
                    || weight.shape() != [spec.group_count(), n, k]
                {
                    return None;
                }
            }
            _ => return None,
        }
        if projection.bias().is_some() {
            let bias = operation.inputs.get(slot)?;
            if bias.dtype() != WorkspaceDtype::Float32 || bias.shape() != [spec.group_count(), n] {
                return None;
            }
            slot = slot.checked_add(1)?;
        }
    }
    if operation.inputs.len()
        != slot.checked_add(if phase == WorkspaceGroupedPhase::Finish {
            4
        } else {
            0
        })?
    {
        return None;
    }
    use crate::backend::nn::grouped::{
        GROUPED_PROJECTION_CHUNK_THRESHOLD as THRESHOLD, GROUPED_PROJECTION_CHUNK_TOKENS as CHUNK,
    };
    let chunks = if tokens > THRESHOLD as usize {
        tokens.div_ceil(CHUNK as usize)
    } else { 1 };
    Some([calls[0].checked_mul(chunks)?, calls[1].checked_mul(chunks)?,
        calls[2].checked_mul(chunks)?])
}

