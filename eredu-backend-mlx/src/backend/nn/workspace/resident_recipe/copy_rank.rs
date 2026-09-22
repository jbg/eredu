//! The actual high-rank workers in prepared rotary and ordinary reshape/concat.
//! Every other operation retains its existing native worker-rank qualification.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct Profile {
    pub worker_rank: usize,
    pub extra_extents: usize,
    pub controls: usize,
}

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    intermediate_rank: usize,
) -> Option<Profile> {
    let rank = operation
        .inputs
        .iter()
        .chain(operation.outputs.iter())
        .map(|input| input.shape().len())
        .max()
        .unwrap_or(0)
        .max(intermediate_rank);
    use WorkspaceOperationKindView as K;
    if matches!(operation.kind, K::GeneratedF32Initialization) {
        let source = super::super::basic::generated_f32_plan_view(operation).ok()?;
        return Some(Profile {
            worker_rank: rank,
            extra_extents: 0,
            controls: source
                .worker_control_bytes::<crate::MlxTensor>()?
                .checked_add(std::mem::size_of::<(&[f32], &[i32], &safemlx::Stream)>())?,
        });
    }
    if matches!(operation.kind, K::Convolution { .. }) {
        let layout = super::super::convolution::original_layout(operation)?;
        return Some(Profile {
            worker_rank: rank.max(4),
            extra_extents: 0,
            controls: crate::backend::nn::convolution::original::control_bytes(layout)?,
        });
    }

    if matches!(
        operation.kind,
        K::HyperCollapse(..) | K::HyperExpand | K::HyperHeadCoefficients(_) | K::HyperHeadSum
    ) {
        let p = super::super::hyper::structure(operation).ok()??;
        let extra = safemlx::ops::OriginalCopyWorkerLayout::inspect(rank, p.reshapes, 0, 0)?;
        let mut controls = extra.control_bytes()?.checked_add(p.controls)?;
        if matches!(operation.kind, K::HyperHeadCoefficients(_)) {
            controls = controls
                .checked_add(crate::backend::nn::shared::hyper_observation_control_bytes()?)?;
        }
        return Some(Profile {
            worker_rank: rank.max(4),
            extra_extents: extra.allocation_extents(),
            controls,
        });
    }
    if matches!(operation.kind, K::HostStoreFloating(..)) {
        let dtype = super::super::basic::host_transfer_dtype(operation)?;
        let (controls, extra_extents) =
            safemlx::PreparedHostCopyDestination::original_layout(rank, dtype)?;
        return Some(Profile {
            worker_rank: rank,
            extra_extents,
            controls,
        });
    }
    if matches!(
        operation.kind,
        K::HostTransferFloating(_) | K::HostLoadStoredFloating(..)
    ) {
        let dtype = super::super::basic::host_transfer_dtype(operation)?;
        let (controls, extra_extents) =
            safemlx::ImmutableHostTransferBuffer::original_copy_layout(rank, dtype)?;
        return Some(Profile {
            worker_rank: rank,
            extra_extents,
            controls,
        });
    }
    if let K::View(name) = operation.kind {
        if super::super::byte_view::selected(name).is_some() {
            super::super::byte_view::inspect(operation)?;
            let source = safemlx::ops::OriginalCopyWorkerLayout::inspect_byte_view(rank)?;
            return Some(Profile {
                worker_rank: 0,
                extra_extents: source.allocation_extents(),
                controls: source
                    .control_bytes()?
                    .checked_add(safemlx::Array::view_dtype_control_bytes()?)?,
            });
        }
    }
    if matches!(operation.kind, K::CastFloating(_)) {
        return Some(Profile {
            worker_rank: rank,
            extra_extents: 0,
            controls: safemlx::Array::as_dtype_control_bytes()?,
        });
    }
    if matches!(operation.kind, K::SelectiveStateSpaceScan(..)) {
        let p = super::super::selective_scan::structure(operation).ok()??;
        // The recurrence's pointwise/reduction rank remains four. Every actual
        // indexing reshape, explicit reshape and final concat also runs the
        // shared strided copy worker; none inherits a low-rank rotary shortcut.
        let extra =
            safemlx::ops::OriginalCopyWorkerLayout::inspect(rank, p.reshapes, 1, p.concat_inputs)?;
        return Some(Profile {
            worker_rank: rank,
            extra_extents: extra.allocation_extents(),
            controls: extra.control_bytes()?.checked_add(p.controls)?,
        });
    }
    if matches!(operation.kind, K::PoolingMask(_)) {
        let extra = safemlx::ops::OriginalCopyWorkerLayout::inspect(rank, 2, 0, 0)?;
        return Some(Profile {
            worker_rank: rank,
            extra_extents: extra.allocation_extents(),
            controls: extra.control_bytes()?.checked_add(
                crate::backend::runtime::cache::kv::PoolingCache::mask_control_bytes()?,
            )?,
        });
    }
    if matches!(operation.kind, K::GatherPooledMask) {
        return Some(Profile {
            worker_rank: rank,
            extra_extents: 0,
            controls: crate::backend::nn::attention::gather_mask::control_bytes()?,
        });
    }
    if matches!(operation.kind, K::Elementwise("clip")) {
        return Some(Profile {
            worker_rank: rank,
            extra_extents: 0,
            controls: crate::tensor::clip::control_bytes()?,
        });
    }
    if matches!(operation.kind, K::PooledPositions { .. }) {
        let (geometry, masked) = super::super::pooling::positions_geometry(operation).ok()?;
        let extra = safemlx::ops::OriginalCopyWorkerLayout::inspect(
            rank,
            if geometry.pooled == 0 { 0 } else { 3 },
            0,
            0,
        )?;
        return Some(Profile {
            worker_rank: rank,
            extra_extents: extra.allocation_extents(),
            controls: extra.control_bytes()?.checked_add(
                crate::backend::nn::attention::pooled_positions::control_bytes(geometry, masked)?,
            )?,
        });
    }
    if matches!(operation.kind, K::RelativeAttention { .. }) {
        return super::relative_attention::copy_profile(operation);
    }
    if matches!(operation.kind, K::JointGroupSelection(_)) {
        return super::joint_selection::copy_profile(operation);
    }
    if matches!(operation.kind, K::MaskedOutputProjection { .. }) {
        return super::masked_readout::copy_profile(operation);
    }
    if matches!(operation.kind, K::PooledAttention { .. }) {
        return super::pooled_attention::copy_profile(operation, rank);
    }
    if let K::IndexedAttention {
        local_mask,
        pooled_mask,
        sinks,
        ..
    } = operation.kind
    {
        super::super::pooling::indexed_geometry(operation).ok()?;
        // Three reshapes in each of the four exact contractions, plus the
        // optional sink reshape. The single concat owns two or three sources.
        let extra = safemlx::ops::OriginalCopyWorkerLayout::inspect(
            rank,
            12 + usize::from(sinks),
            1,
            2 + usize::from(sinks),
        )?;
        return Some(Profile {
            worker_rank: rank,
            extra_extents: extra.allocation_extents(),
            controls: extra.control_bytes()?.checked_add(
                crate::backend::nn::attention::indexed::control_bytes(
                    usize::from(local_mask) + usize::from(pooled_mask),
                    sinks,
                )?,
            )?,
        });
    }
    let (worker_rank, reshapes, concatenations, inputs) = match operation.kind {
        // The shared worker reshapes positions to [rows, axes] before its
        // arithmetic and restores the two output prefixes after cosine/sine.
        K::PreparedMultiAxisRotary(_) => (2, 3, 0, 0),
        K::View("reshape") => (0, 1, 0, 0),
        // The existing frontend can cast each input; native concat performs
        // one general/general copy into each temporary output slice.
        K::Concatenate => (0, 0, 1, operation.inputs.len()),
        _ => {
            return Some(Profile {
                worker_rank: rank,
                extra_extents: 0,
                controls: 0,
            });
        }
    };
    let extra =
        safemlx::ops::OriginalCopyWorkerLayout::inspect(rank, reshapes, concatenations, inputs)?;
    Some(Profile {
        worker_rank,
        extra_extents: extra.allocation_extents(),
        controls: extra.control_bytes()?,
    })
}
