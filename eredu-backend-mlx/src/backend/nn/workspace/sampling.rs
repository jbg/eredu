//! Sampling buffer bounds for the retained Metal realization. Compound costs
//! retain all children until completion; no donation or early release is assumed.

use super::facts::{
    self, add, buffer_capacity, mul, Aliases, Emitter, FactResult, HostEmitter, Output,
};
use super::{
    indexing::embedding_validation_cost_fixed as embedding_validation_cost,
    reduction::{capacity_fixed as capacity, sum_cost_fixed as sum_cost},
    *,
};
use eredu_nn::EmbeddingLookupPolicy;

fn invalid() -> MlxWorkspaceFactError {
    MlxWorkspaceFactError::descriptor("invalid Metal sampling workspace descriptor")
}
fn pointwise(a: NativeAllocationFacts, elements: u64, inputs: u64) -> FactResult<u64> {
    // One possible <=F32 cast per operand, the result, and each scalar operand.
    add(
        mul(inputs + 1, capacity(a, elements)?)?,
        mul(inputs, capacity(a, 1)?)?,
    )
}
fn fill(a: NativeAllocationFacts, elements: u64) -> FactResult<u64> {
    // F32 fill, optional restoration cast, and source/fill/cast scalars.
    add(mul(2, capacity(a, elements)?)?, mul(3, capacity(a, 1)?)?)
}
pub(super) fn sort(
    a: NativeAllocationFacts,
    elements: u64,
    rows: u64,
    width: u64,
) -> Result<u64, Error> {
    sort_fixed(a, elements, rows, width).map_err(MlxWorkspaceFactError::ordinary)
}

pub(super) fn sort_fixed(
    a: NativeAllocationFacts,
    elements: u64,
    rows: u64,
    width: u64,
) -> facts::FactResult<u64> {
    use super::reduction::capacity_fixed as capacity;
    use facts::{add, mul};
    let output = capacity(a, elements)?;
    if width <= 2048 {
        return Ok(output);
    }
    // Metal's <=4-byte sort uses 512 threads x 4 values above one block.
    // Values and indices each have two ping-pong arrays, reused by every merge;
    // the partition array has one entry per block plus one for each row.
    let partitions = mul(rows, add(width.div_ceil(2048), 1)?)?;
    add(mul(5, output)?, capacity(a, partitions)?)
}

/// Shared validator: finite cardinality, preserved integer coordinates and I32 output.
pub(super) fn token_validation_layouts(
    operation: WorkspaceOperationView<'_>,
) -> Option<[WorkspaceLayoutView<'_>; 2]> {
    let WorkspaceOperationKindView::Sampling(
        WorkspaceSamplingOperation::ValidateToken { cardinality },
    ) = operation.kind else { return None; };
    let [input] = operation.inputs.array()?;
    let [output] = operation.outputs.array()?;
    if *cardinality == 0 || *cardinality > i32::MAX as u32
        || !matches!(input.dtype(), WorkspaceDtype::Int32 | WorkspaceDtype::Uint32)
        || output.shape() != input.shape() || output.dtype() != WorkspaceDtype::Int32
    { return None; }
    Some([input, output])
}

pub(super) fn operation_bound(
    op: &WorkspaceOperation,
    a: NativeAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary(|sink| emit(op.as_view(), a, sink))
}

pub(super) fn emit(
    op: WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let WorkspaceOperationKindView::Sampling(kind) = &op.kind else {
        return Ok(None);
    };
    use WorkspaceSamplingOperation as S;
    let Some([out]) = op.outputs.array() else {
        return Err(invalid());
    };
    if matches!(kind, S::CreateRandomKey) {
        if !op.inputs.is_empty() || out.shape() != [2] || out.dtype() != WorkspaceDtype::Uint32 {
            return Err(invalid());
        }
        sink.output(Output::Allocate(buffer_capacity(a, 8)?))?;
        return sink
            .finish(
                0,
                format_args!(
                    "explicit seed key copies two U32 words into one allocator-owned native buffer"
                ),
            )
            .map(Some);
    }
    let input = op.inputs.first().ok_or_else(invalid)?;
    let mut alias = false;
    let total = match kind {
        S::ReadToken => {
            if op.inputs.len() != 1
                || input.elements()? != 1
                || input.dtype() != WorkspaceDtype::Uint32
                || out != input
            {
                return Err(invalid());
            }
            alias = true;
            0
        }
        S::SelectRandomKey { index } => {
            if op.inputs.len() != 1 || input.dtype() != WorkspaceDtype::Uint32
                || input.shape().len() != 2 || input.shape()[0] <= 0 || input.shape()[1] != 2
                || u64::from(*index) >= input.shape()[0] as u64
                || out.shape() != [2] || out.dtype() != WorkspaceDtype::Uint32 { return Err(invalid()); }
            alias = true;
            0
        }
        S::SplitRandomKey => {
            if op.inputs.len() != 1
                || input.shape() != [2]
                || input.dtype() != WorkspaceDtype::Uint32
                || out.shape().len() != 2
                || out.shape()[0] <= 0 || out.shape()[1] != 2
                || out.dtype() != WorkspaceDtype::Uint32
            {
                return Err(invalid());
            }
            // random::split(key,n) is exactly RandomBits([n,2], U32).
            buffer_capacity(a, mul(out.elements()?, 4)?)?
        }
        S::UniformUnitInterval => {
            if op.inputs.len() != 1 || input.shape() != [2]
                || input.dtype() != WorkspaceDtype::Uint32
                || out.shape() != [1] || out.dtype() != WorkspaceDtype::Float32 {
                return Err(invalid());
            }
            // Four eager low/high/upper/maxval scalars, two initial casts,
            // RandomBits, five binary workers and one restoration cast.
            // Each binary bound includes both operand casts/copies and result;
            // all capacities remain live conservatively through completion.
            add(mul(8, capacity(a, 1)?)?, mul(5, pointwise(a, 1, 2)?)?)?
        }
        S::ValidateToken { .. } => {
            if token_validation_layouts(op).is_none() { return Err(invalid()); }
            add(
                capacity(a, input.elements()?)?,
                embedding_validation_cost(
                    input.elements()?,
                    input.elements()?,
                    EmbeddingLookupPolicy::Strict,
                    a,
                )?,
            )?
        }
        _ => {
            let width = input
                .shape()
                .last()
                .copied()
                .filter(|n| *n > 0)
                .ok_or_else(invalid)? as u64;
            let elements = input.elements()?;
            let rows = elements / width;
            if rows == 0 || input.dtype() != WorkspaceDtype::Float32 {
                return Err(invalid());
            }
            let token_shape = &input.shape()[..input.shape().len() - 1];
            match kind {
                S::Greedy | S::Categorical
                    if out.shape() != token_shape || out.dtype() != WorkspaceDtype::Uint32 =>
                {
                    return Err(invalid())
                }
                S::TokenProbability
                    if !out.shape().is_empty()
                        || out.dtype() != WorkspaceDtype::Float32
                        || input.shape().len() > 3 =>
                {
                    return Err(invalid())
                }
                S::Greedy | S::Categorical | S::TokenProbability => {}
                _ if out != input => return Err(invalid()),
                _ => {}
            }
            if matches!(kind, S::Categorical) {
                if op.inputs.len() != 2
                    || op.inputs.get(1).expect("checked two inputs").shape() != [2]
                    || op.inputs.get(1).expect("checked two inputs").dtype()
                        != WorkspaceDtype::Uint32
                {
                    return Err(invalid());
                }
            } else if op.inputs.len() != 1 {
                return Err(invalid());
            }
            let full = capacity(a, elements)?;
            let row = capacity(a, rows)?;
            let scalar = capacity(a, 1)?;
            let reduction = || sum_cost(a, elements, rows, width);
            match kind {
                S::Greedy => row, // ArgReduce reads strided input; no partial array.
                S::Categorical => {
                    // RandomBits; uniform divide/min/cast/multiply/add; four
                    // Gumbel unary operations; addition to logits; argmax.
                    // Include each possible cast and all uniform/fill scalars.
                    add(add(mul(25, full)?, mul(21, scalar)?)?, row)?
                }
                S::Penalties {
                    repetition,
                    additive,
                    ..
                } => {
                    let mut total = 0;
                    if *repetition {
                        total = add(
                            buffer_capacity(a, elements)?,
                            add(
                                mul(3, pointwise(a, elements, 2)?)?,
                                mul(2, pointwise(a, elements, 3)?)?,
                            )?,
                        )?;
                    }
                    if *additive {
                        total = add(total, add(full, pointwise(a, elements, 2)?)?)?;
                    }
                    // A disabled or zero-window operation can preserve input.
                    total.max(full)
                }
                S::TopK { keep } => {
                    if *keep == 0 || u64::from(*keep) >= width {
                        return Err(invalid());
                    }
                    // Native partition is Metal sort; its sliced result may
                    // retain the entire buffer. Then min, comparison and where.
                    add(
                        add(sort_fixed(a, elements, rows, width)?, reduction()?)?,
                        add(pointwise(a, elements, 2)?, pointwise(a, elements, 3)?)?,
                    )?
                }
                S::TopP => {
                    // Negative, sort, gather (including index/layout copies),
                    // softmax, scan/copy, subtract, compare, mask, fill/cast,
                    // and scatter with source/index/update copies.
                    let mut total = sort_fixed(a, elements, rows, width)?;
                    for bytes in [
                        pointwise(a, elements, 1)?,
                        mul(3, full)?,
                        full,
                        mul(2, full)?,
                        pointwise(a, elements, 2)?,
                        pointwise(a, elements, 2)?,
                        pointwise(a, elements, 3)?,
                        fill(a, elements)?,
                        mul(4, full)?,
                    ] {
                        total = add(total, bytes)?;
                    }
                    total
                }
                S::MinP => {
                    let mut total = add(full, reduction()?)?;
                    for bytes in [
                        pointwise(a, rows, 2)?,
                        pointwise(a, elements, 2)?,
                        pointwise(a, elements, 3)?,
                    ] {
                        total = add(total, bytes)?;
                    }
                    total
                }
                S::TokenFilter | S::OptionalTokenFilter => {
                    add(buffer_capacity(a, elements)?, pointwise(a, elements, 3)?)?
                }
                S::MirostatCutoff => {
                    if rows != 1 {
                        return Err(invalid());
                    }
                    let mut total = add(add(full, reduction()?)?, row)?;
                    for bytes in [
                        scalar,
                        pointwise(a, elements, 2)?,
                        fill(a, elements)?,
                        fill(a, rows)?,
                        mul(4, full)?,
                        pointwise(a, rows, 2)?,
                        pointwise(a, elements, 3)?,
                        pointwise(a, elements, 3)?,
                    ] {
                        total = add(total, bytes)?;
                    }
                    total
                }
                S::TokenProbability => add(full, mul(2, scalar)?)?,
                _ => unreachable!("non-logit descriptors handled above"),
            }
        }
    };
    let output = if alias {
        0
    } else {
        buffer_capacity(a, out.bytes()?.max(4))?
    };
    if total < output {
        return Err(invalid());
    }
    sink.output(if alias {
        Output::AliasInput(0)
    } else if matches!(kind, S::SplitRandomKey | S::UniformUnitInterval | S::Greedy | S::Categorical) {
        Output::Allocate(output)
    } else {
        Output::AllocateOrAliasInputs {
            bytes: output,
            inputs: Aliases::Slice(if out.dtype() == input.dtype() {
                &[0]
            } else {
                &[]
            }),
        }
    })?;
    sink.finish(total - output, format_args!("selected MLX Metal sampling {kind:?}: complete native child graphs, merge-sort ping-pong and partition buffers, possible operand casts/copies, and scalar storage retained through completion; page={} and bounded oversized reuse; excludes separately priced input logits and host payloads", a.page_size())).map(Some)
}

pub(super) fn host_bound(op: &WorkspaceOperation) -> Result<Option<WorkspaceHostBound>, Error> {
    facts::ordinary_host(|sink| emit_host(op.as_view(), sink))
}

pub(super) fn emit_host(
    op: WorkspaceOperationView<'_>,
    sink: &mut HostEmitter<'_>,
) -> FactResult<Option<WorkspaceHostFacts>> {
    let WorkspaceOperationKindView::Sampling(kind) = &op.kind else {
        return Ok(None);
    };
    use WorkspaceSamplingOperation as S;
    let bytes = match kind {
        S::Penalties {
            history_positions, ..
        } => add(
            mul(op.inputs.first().ok_or_else(invalid)?.elements()?, 5)?,
            mul(*history_positions as u64, 4)?,
        )?,
        S::TokenFilter | S::OptionalTokenFilter => add(
            op.inputs.first().ok_or_else(invalid)?.elements()?,
            *op.inputs
                .first()
                .ok_or_else(invalid)?
                .shape()
                .last()
                .ok_or_else(invalid)? as u64,
        )?,
        _ => 0,
    };
    sink.finish(bytes, format_args!("selected MLX sampling: exact boolean/additive vocabulary buffers and an in-place sorted U32 history window; closed filter expansion plus exact row-mask allocation; other primitives use shared native buffers and direct scalar reads; existing history/filter ownership and allocator bookkeeping are outside this operation")).map(Some)
}

#[cfg(test)]
mod tests;
