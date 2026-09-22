//! Local/pooled bank and mask preparation around the existing fused SDPA recipe.
use super::*;

#[derive(Clone, Copy)]
struct Front {
    // One normalization branch per actual optional mask, including missing
    // bank masks. None means there is no joined mask at all.
    masks: Option<[Option<WorkspaceDtype>; 2]>,
    additive: bool,
}
fn with_child<R>(
    op: WorkspaceOperationView<'_>,
    visit: impl FnOnce(WorkspaceOperationView<'_>, Front) -> Option<R>,
) -> Option<R> {
    let WorkspaceOperationKindView::PooledAttention {
        scale,
        local_mask,
        pooled_mask,
        sinks,
    } = op.kind
    else {
        return None;
    };
    if !scale.is_finite()
        || scale <= 0.0
        || op.inputs.len()
            != 3 + usize::from(local_mask) + usize::from(pooled_mask) + usize::from(sinks)
        || op.outputs.len() != 1
    {
        return None;
    }
    let q: [i32; 4] = op.inputs.get(0)?.shape().try_into().ok()?;
    let l: [i32; 3] = op.inputs.get(1)?.shape().try_into().ok()?;
    let p: [i32; 3] = op.inputs.get(2)?.shape().try_into().ok()?;
    let keys = l[1].checked_add(p[1])?;
    if q.iter().any(|&d| d <= 0)
        || l[1] < 0
        || p[1] < 0
        || keys <= 0
        || q[0] != l[0]
        || q[0] != p[0]
        || q[3] != l[2]
        || q[3] != p[2]
        || op
            .inputs
            .slice(0..3)?
            .iter()
            .any(|a| a.dtype() != WorkspaceDtype::Float32)
        || op.outputs.get(0)?.shape() != q
        || op.outputs.get(0)?.dtype() != WorkspaceDtype::Float32
    {
        return None;
    }
    let local = if local_mask {
        Some(op.inputs.get(3)?)
    } else {
        None
    };
    let pooled = if pooled_mask {
        Some(op.inputs.get(3 + usize::from(local_mask))?)
    } else {
        None
    };
    let shapes = crate::backend::nn::attention::pooled_mask_shapes_fixed(
        &q,
        l[1],
        p[1],
        local.map(|v| v.shape()),
        pooled.map(|v| v.shape()),
    )
    .ok()?;
    let masks = [local.map(|v| v.dtype()), pooled.map(|v| v.dtype())];
    if masks
        .into_iter()
        .flatten()
        .any(|d| !matches!(d, WorkspaceDtype::Float32 | WorkspaceDtype::Bool))
    {
        return None;
    }
    let additive = masks.contains(&Some(WorkspaceDtype::Float32));
    if sinks
        && (op.inputs.last()?.shape() != [q[1]]
            || op.inputs.last()?.dtype() != WorkspaceDtype::Float32)
    {
        return None;
    }
    let bank_shape = [q[0], 1, keys, q[3]];
    let bank = WorkspaceLayoutView::new(&bank_shape, WorkspaceDtype::Float32).ok()?;
    let mask_shape = shapes.map(|(ls, _)| [ls[0], ls[1], ls[2], keys]);
    let mut inputs = [op.inputs.get(0)?, bank, bank, bank, bank];
    let mut count = 3;
    if let Some(shape) = mask_shape.as_ref() {
        inputs[count] = WorkspaceLayoutView::new(
            shape,
            if additive {
                WorkspaceDtype::Float32
            } else {
                WorkspaceDtype::Bool
            },
        )
        .ok()?;
        count += 1;
    }
    if sinks {
        inputs[count] = op.inputs.last()?;
        count += 1;
    }
    let child = WorkspaceOperationView {
        kind: WorkspaceOperationKindView::Attention {
            causal: false,
            window: None,
            sinks,
            softcap: false,
            arithmetic: eredu_nn::AttentionArithmetic::Fused,
        },
        inputs: WorkspaceLayoutList::Views(&inputs[..count]),
        outputs: op.outputs,
    };
    visit(
        child,
        Front {
            masks: shapes.map(|_| masks),
            additive,
        },
    )
}

pub(super) fn lowering(op: WorkspaceOperationView<'_>) -> Option<Lowering> {
    with_child(op, |child, front| {
        // Same native SDPA source/fallback, with the exact joined geometry.
        let mut value = super::lowering(child)?;
        // Both banks expand a head axis, then concatenate (two casts/result).
        let (mut nodes, mut edges, mut seeds) = (5usize, 6usize, 0usize);
        if let Some(masks) = front.masks {
            for mask in masks {
                match mask {
                    Some(WorkspaceDtype::Bool) if front.additive => {
                        nodes += 7;
                        edges += 9;
                        seeds += 2; // Where(mask,0,-inf)
                    }
                    None => seeds += 1, // actual true or additive-zero scalar
                    _ => {}
                }
                nodes += 1;
                edges += 1; // normalized mask Broadcast
            }
            nodes += 3;
            edges += 4; // two casts and joined-mask Concat
            if front.additive {
                nodes += 1;
                edges += 1;
            } // promoted mask AsType
        }
        value.primitives = value.primitives.checked_add(nodes)?;
        value.edges = value.edges.checked_add(edges)?;
        value.seeds = value.seeds.checked_add(seeds)?;
        value.maximum_births = value
            .maximum_births
            .checked_add(nodes)?
            .checked_add(seeds)?;
        // Array-valued masks are borrowed by the native wrapper. Each actual
        // Some(mask) branch retains one safe alias before its Broadcast call.
        value.backend_shells = value
            .backend_shells
            .checked_add(front.masks.map_or(0, |m| {
                m.into_iter()
                    .flatten()
                    .filter(|&d| !front.additive || d != WorkspaceDtype::Bool)
                    .count()
            }))?;
        Some(value)
    })
}

pub(super) fn copy_profile(
    op: WorkspaceOperationView<'_>,
    rank: usize,
) -> Option<copy_rank::Profile> {
    with_child(op, |_, front| {
        let concats = 1 + usize::from(front.masks.is_some());
        // Two bank ExpandDims run the shared reshape worker; each actual
        // Concat owns exactly two input copy descriptors. SDPA's internal
        // population remains in its existing common rank/worker recipe.
        let extra = safemlx::ops::OriginalCopyWorkerLayout::inspect(4, 2, concats, concats * 2)?;
        Some(copy_rank::Profile {
            worker_rank: rank,
            extra_extents: extra.allocation_extents(),
            controls: extra.control_bytes()?.checked_add(
                crate::backend::nn::attention::pooled_attention_control_bytes(
                    front.masks.is_some(),
                )?,
            )?,
        })
    })
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use eredu_nn::{NeuralBackend, PooledAttentionInput, Tensor};
    #[test]
    fn pooled_recipe_joins_actual_empty_banks_masks_and_fused_child() {
        if !crate::tests::support::native_process::enter("qualified-recipe") {
            return;
        }
        let _sources = crate::tests::support::test_utils::initialize_original_sources();
        let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        for (local, pooled, mixed) in [(2, 0, false), (0, 3, true), (2, 3, true)] {
            let context = WorkspaceContext::new(mechanism);
            let tensor = |shape: &[i32], dtype| {
                WorkspaceTensor::existing(WorkspaceLayout::new(shape, dtype).unwrap(), &context)
                    .unwrap()
            };
            let q = tensor(&[1, 2, 2, 8], WorkspaceDtype::Float32);
            let l = tensor(&[1, local, 8], WorkspaceDtype::Float32);
            let p = tensor(&[1, pooled, 8], WorkspaceDtype::Float32);
            let lm = tensor(&[2, local], WorkspaceDtype::Bool);
            let pm = tensor(&[2, pooled], WorkspaceDtype::Float32);
            let sinks = tensor(&[2], WorkspaceDtype::Float32);
            context.begin_span();
            let output = WorkspaceBackend::pooled_attention(
                PooledAttentionInput {
                    queries: &q,
                    local: &l,
                    pooled: &p,
                    scale: 0.25,
                    local_mask: Some(&lm),
                    pooled_mask: mixed.then_some(&pm),
                    sinks: Some(&sinks),
                },
                &context,
            )
            .unwrap();
            assert_eq!(output.shape(), [1, 2, 2, 8]);
            let report = context.report(&[output]).unwrap();
            assert_eq!(report.operations.len(), 1);
            let op = report.operations[0].as_view();
            let callers = mechanism.ordinary_call_controls(op).unwrap().unwrap();
            assert!(callers.metadata_bytes > 0);
            let outer = lowering(op).unwrap();
            let child = with_child(op, |child, _| super::super::lowering(child)).unwrap();
            assert!(outer.primitives > child.primitives);
            assert!(outer.maximum_births > child.maximum_births);
            let profile = copy_profile(op, outer.intermediate_rank).unwrap();
            assert!(profile.controls > 0);
            let recorder = ResidentRecipeRecorder::new(
                InferenceGeometry {
                    batch_size: 1,
                    cached_positions: 0,
                    input_positions: 2,
                    max_output_tokens: 1,
                    prefill_chunk_positions: 2,
                    output: eredu_core::OutputDemand::Sequence,
                },
                mechanism,
            );
            let reduced = recorder.reduce_trace(&report, None, 0, 1).unwrap();
            assert_eq!(reduced.first_missing_operation, None);
            assert!(reduced.graph.is_some() && reduced.dispatch.is_some());
            let mut malformed = report.operations[0].clone();
            malformed.inputs[1] =
                WorkspaceLayout::new(&[1, i32::MAX, 8], WorkspaceDtype::Float32).unwrap();
            malformed.inputs[2] =
                WorkspaceLayout::new(&[1, 1, 8], WorkspaceDtype::Float32).unwrap();
            assert!(lowering(malformed.as_view()).is_none());
        }
    }
}
