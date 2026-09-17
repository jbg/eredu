//! Shared exact operand decoding for native blockwise facts and recipes.
use super::*;
use eredu_nn::{
    operation_geometry::AbsoluteAttentionMaskGeometry,
    workspace::{WorkspaceBlockwisePolicy, WorkspaceBlockwiseStage},
};

#[derive(Clone, Copy)]
pub(in crate::backend::nn::workspace) struct Descriptor<'a> {
    pub policy: WorkspaceBlockwisePolicy,
    pub stage: Stage<'a>,
}
#[derive(Clone, Copy)]
pub(in crate::backend::nn::workspace) enum Stage<'a> {
    Begin {
        q: [i32; 4],
        mask: Option<WorkspaceLayoutView<'a>>,
        sink: Option<WorkspaceLayoutView<'a>>,
    },
    Accumulate {
        q: [i32; 4],
        k: [i32; 4],
        v: [i32; 4],
        mask: Option<WorkspaceLayoutView<'a>>,
        sink: Option<WorkspaceLayoutView<'a>>,
        bias: Option<WorkspaceLayoutView<'a>>,
        previous: bool,
        value_pass: bool,
        absolute: AbsoluteAttentionMaskGeometry,
    },
    Finish {
        shape: [i32; 4],
        input_scores: bool,
    },
}
fn invalid() -> MlxWorkspaceFactError {
    MlxWorkspaceFactError::descriptor("invalid native blockwise stage descriptor")
}
fn shape(value: WorkspaceLayoutView<'_>) -> FactResult<[i32; 4]> {
    if value.dtype() != WorkspaceDtype::Float32
        || value.shape().len() != 4
        || value.shape().iter().any(|n| *n <= 0)
    {
        return Err(invalid());
    }
    value.shape().try_into().map_err(|_| invalid())
}
fn same(value: WorkspaceLayoutView<'_>, shape: &[i32], dtype: WorkspaceDtype) -> FactResult<()> {
    if value.shape() != shape || value.dtype() != dtype {
        return Err(invalid());
    }
    Ok(())
}
fn broadcast(value: WorkspaceLayoutView<'_>, target: &[i32; 4]) -> FactResult<()> {
    if value.shape().len() > 4
        || value.shape().iter().any(|n| *n <= 0)
        || value
            .shape()
            .iter()
            .rev()
            .zip(target.iter().rev())
            .any(|(a, b)| *a != 1 && a != b)
    {
        return Err(invalid());
    }
    Ok(())
}
pub(in crate::backend::nn::workspace) fn decode(
    operation: WorkspaceOperationView<'_>,
) -> FactResult<Option<Descriptor<'_>>> {
    let WorkspaceOperationKindView::BlockwiseAttention { policy, stage } = operation.kind else {
        return Ok(None);
    };
    if policy.options.validate().is_err()
        || !policy.scale.is_finite()
        || policy.scale <= 0.
        || policy.query_start < 0
        || policy.context_end <= policy.query_start
        || policy.prefix_tokens < 0
        || policy.sliding_window.is_some_and(|n| n <= 0)
    {
        return Err(invalid());
    }
    let inputs = operation.inputs;
    let outputs = operation.outputs;
    let result = match stage {
        WorkspaceBlockwiseStage::Begin => {
            let q = shape(inputs.get(0).ok_or_else(invalid)?)?;
            if policy.output_type
                != inputs
                    .get(0)
                    .and_then(|input| input.representation())
                    .map(WorkspaceRepresentation::dtype)
            {
                return Err(invalid());
            }
            if policy
                .query_start
                .checked_add(i64::from(q[2]))
                .is_none_or(|end| end > policy.context_end)
            {
                return Err(invalid());
            }
            let has_mask = policy.mask_origin.is_some();
            let base = 1 + usize::from(has_mask);
            if !(base..=base + 1).contains(&inputs.len()) || outputs.len() != base {
                return Err(invalid());
            }
            same(
                outputs.get(0).ok_or_else(invalid)?,
                &q,
                WorkspaceDtype::Float32,
            )?;
            let mask = if has_mask {
                let mask = inputs.get(1).ok_or_else(invalid)?;
                let origin = policy.mask_origin.unwrap();
                let columns =
                    i32::try_from(policy.context_end.checked_sub(origin).ok_or_else(invalid)?)
                        .map_err(|_| invalid())?;
                let target = [q[0], q[1], q[2], columns];
                if columns <= 0 || mask.shape().last() != Some(&columns) {
                    return Err(invalid());
                }
                broadcast(mask, &target)?;
                if !matches!(mask.dtype(), WorkspaceDtype::Float32 | WorkspaceDtype::Bool) {
                    return Err(invalid());
                }
                same(outputs.get(1).ok_or_else(invalid)?, &target, mask.dtype())?;
                Some(mask)
            } else {
                None
            };
            let sink = if inputs.len() > base {
                let sink = inputs.get(base).ok_or_else(invalid)?;
                same(sink, &[q[1]], WorkspaceDtype::Float32)?;
                Some(sink)
            } else {
                None
            };
            Stage::Begin { q, mask, sink }
        }
        WorkspaceBlockwiseStage::Accumulate {
            start,
            end,
            previous,
            value_pass,
            bias,
        } => {
            let q = shape(inputs.get(0).ok_or_else(invalid)?)?;
            let k = shape(inputs.get(1).ok_or_else(invalid)?)?;
            let v = shape(inputs.get(2).ok_or_else(invalid)?)?;
            if start < 0
                || end > policy.context_end
                || end <= start
                || end.checked_sub(start) != Some(i64::from(k[2]))
                || q[0] != k[0]
                || k[..3] != v[..3]
                || q[3] != k[3]
                || q[1] % k[1] != 0
                || policy
                    .query_start
                    .checked_add(i64::from(q[2]))
                    .is_none_or(|n| n > policy.context_end)
                || (value_pass && policy.options.arithmetic != AttentionArithmetic::InputScores)
            {
                return Err(invalid());
            }
            let absolute = AbsoluteAttentionMaskGeometry::new(
                policy.query_start,
                q[2],
                start,
                end,
                policy.sliding_window,
                policy.prefix_tokens,
            )
            .map_err(|_| invalid())?;
            let running = if value_pass {
                2 + usize::from(previous)
            } else {
                3 * usize::from(previous)
            };
            let base = 3 + usize::from(policy.mask_origin.is_some()) + usize::from(bias) + running;
            if inputs.len() < base
                || inputs.len() > base + usize::from(!value_pass)
                || outputs.len() != if value_pass { 1 } else { 3 }
            {
                return Err(invalid());
            }
            let sinks = inputs.len() > base;
            let mut index = 3;
            let mask = if let Some(origin) = policy.mask_origin {
                if origin > start {
                    return Err(invalid());
                }
                let columns =
                    i32::try_from(policy.context_end.checked_sub(origin).ok_or_else(invalid)?)
                        .map_err(|_| invalid())?;
                let mask = inputs.get(index).ok_or_else(invalid)?;
                index += 1;
                if !matches!(mask.dtype(), WorkspaceDtype::Bool | WorkspaceDtype::Float32) {
                    return Err(invalid());
                }
                same(mask, &[q[0], q[1], q[2], columns], mask.dtype())?;
                Some(mask)
            } else {
                None
            };
            let sink = if sinks {
                let sink = inputs.get(index).ok_or_else(invalid)?;
                index += 1;
                same(sink, &[q[1]], WorkspaceDtype::Float32)?;
                Some(sink)
            } else {
                None
            };
            let bias = if bias {
                let bias = inputs.get(index).ok_or_else(invalid)?;
                index += 1;
                if bias.dtype() != WorkspaceDtype::Float32 {
                    return Err(invalid());
                }
                broadcast(bias, &[q[0], q[1], q[2], k[2]])?;
                Some(bias)
            } else {
                None
            };
            let rows = [q[0], q[1], q[2], 1];
            let value = [q[0], q[1], q[2], v[3]];
            for prior in 0..running {
                same(
                    inputs.get(index + prior).ok_or_else(invalid)?,
                    if prior < 2 { &rows } else { &value },
                    WorkspaceDtype::Float32,
                )?;
            }
            if !value_pass {
                for i in 0..2 {
                    same(
                        outputs.get(i).ok_or_else(invalid)?,
                        &rows,
                        WorkspaceDtype::Float32,
                    )?;
                }
            }
            same(
                outputs.get(outputs.len() - 1).ok_or_else(invalid)?,
                &value,
                WorkspaceDtype::Float32,
            )?;
            Stage::Accumulate {
                q,
                k,
                v,
                mask,
                sink,
                bias,
                previous,
                value_pass,
                absolute,
            }
        }
        WorkspaceBlockwiseStage::Finish => {
            let shape = shape(inputs.get(0).ok_or_else(invalid)?)?;
            let input_scores = policy.options.arithmetic == AttentionArithmetic::InputScores;
            if outputs.len() != 1 || inputs.len() != if input_scores { 1 } else { 2 } {
                return Err(invalid());
            }
            if !input_scores {
                same(
                    inputs.get(1).ok_or_else(invalid)?,
                    &[shape[0], shape[1], shape[2], 1],
                    WorkspaceDtype::Float32,
                )?;
            }
            same(
                outputs.get(0).ok_or_else(invalid)?,
                &shape,
                WorkspaceDtype::Float32,
            )?;
            Stage::Finish {
                shape,
                input_scores,
            }
        }
    };
    Ok(Some(Descriptor {
        policy,
        stage: result,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_nn::{BlockwiseAttentionOptions, workspace::WorkspaceLayoutList};
    fn f32_layout(shape: &[i32]) -> WorkspaceLayoutView<'_> {
        WorkspaceLayoutView::new(shape, WorkspaceDtype::Float32).unwrap()
    }
    #[test]
    fn page_descriptors_reject_overflow_and_mask_column_mismatch_but_accept_scalar_bias() {
        let policy = WorkspaceBlockwisePolicy {
            query_start: 0,
            context_end: 2,
            scale: 0.5,
            sliding_window: None,
            prefix_tokens: 0,
            mask_origin: None,
            output_type: None,
            options: BlockwiseAttentionOptions {
                arithmetic: AttentionArithmetic::Fused,
                softcap: None,
            },
        };
        let inputs = [
            f32_layout(&[1, 2, 2, 4]),
            f32_layout(&[1, 1, 2, 4]),
            f32_layout(&[1, 1, 2, 3]),
            f32_layout(&[]),
        ];
        let outputs = [
            f32_layout(&[1, 2, 2, 1]),
            f32_layout(&[1, 2, 2, 1]),
            f32_layout(&[1, 2, 2, 3]),
        ];
        let operation = |start, end| WorkspaceOperationView {
            kind: WorkspaceOperationKindView::BlockwiseAttention {
                policy,
                stage: WorkspaceBlockwiseStage::Accumulate {
                    start,
                    end,
                    previous: false,
                    value_pass: false,
                    bias: true,
                },
            },
            inputs: WorkspaceLayoutList::Views(&inputs),
            outputs: WorkspaceLayoutList::Views(&outputs),
        };
        assert!(decode(operation(i64::MIN, 1)).is_err());
        assert!(matches!(
            decode(operation(0, 2)).unwrap().unwrap().stage,
            Stage::Accumulate { bias: Some(_), .. }
        ));
        let outputs = [inputs[0], f32_layout(&[1, 2, 2, 2])];
        for mask in [f32_layout(&[]), f32_layout(&[2, 1]), f32_layout(&[2, 0])] {
            let inputs = [inputs[0], mask];
            assert!(
                decode(WorkspaceOperationView {
                    kind: WorkspaceOperationKindView::BlockwiseAttention {
                        policy: WorkspaceBlockwisePolicy {
                            mask_origin: Some(0),
                            ..policy
                        },
                        stage: WorkspaceBlockwiseStage::Begin
                    },
                    inputs: WorkspaceLayoutList::Views(&inputs),
                    outputs: WorkspaceLayoutList::Views(&outputs),
                })
                .is_err()
            );
        }
        let inputs = [inputs[0], f32_layout(&[2, 2])];
        assert!(
            decode(WorkspaceOperationView {
                kind: WorkspaceOperationKindView::BlockwiseAttention {
                    policy: WorkspaceBlockwisePolicy {
                        mask_origin: Some(0),
                        ..policy
                    },
                    stage: WorkspaceBlockwiseStage::Begin
                },
                inputs: WorkspaceLayoutList::Views(&inputs),
                outputs: WorkspaceLayoutList::Views(&outputs),
            })
            .unwrap()
            .is_some()
        );
    }
}
