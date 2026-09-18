//! Source-visible F32 CPU packed grouped equations. Every worker below is the
//! same selected CPU worker used by ordinary grouped execution; bank residency
//! and accepted occurrence authority remain with the existing source provider.
use super::*;
use safemlx::Dtype;
mod descriptor;
mod program;
use descriptor::{Activation, Descriptor, Projection};
use super::program::Program;

fn invalid() -> MlxWorkspaceFactError {
    MlxWorkspaceFactError::descriptor("CPU grouped source geometry differs")
}

fn retained(d: Descriptor, mechanism: MlxCpuWorkspaceMechanisms) -> facts::FactResult<[u64; 4]> {
    let mut bytes = [0u64; 4];
    let capacity = |n: usize| {
        mechanism
            .allocation
            .fixed_buffer_capacity(facts::mul(n as u64, 4)?)
    };
    if d.phase == WorkspaceGroupedPhase::Units {
        for (rows, count) in d.chunks() {
            if count == 0 {
                continue;
            }
            let n = rows
                .checked_mul(d.routes)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
            // A donated gate slice may keep the complete fused read allocation.
            let sizes = [
                capacity(
                    n.checked_mul(d.first.output)
                        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
                )?,
                capacity(n)?,
                capacity(n)?,
                capacity(n)?,
            ];
            for (sum, size) in bytes.iter_mut().zip(sizes) {
                *sum = facts::add(*sum, facts::mul(size, count as u64)?)?;
            }
        }
    } else {
        let n = d
            .tokens
            .checked_mul(d.output)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        bytes[0] = capacity(n)?;
        if d.separate_bias {
            bytes[1] = bytes[0];
        }
    }
    Ok(bytes)
}

pub(super) fn emit_outputs(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
    sink: &mut facts::Emitter<'_>,
) -> facts::FactResult<()> {
    let d = Descriptor::inspect(operation)?.ok_or_else(invalid)?;
    for bytes in retained(d, mechanism)?
        .into_iter()
        .take(operation.outputs.len())
    {
        sink.output(facts::Output::Allocate(bytes))?;
    }
    Ok(())
}

/// Actual output table and callback destinations, independent of CPU arithmetic.
pub(in super::super) fn output_storage(
    operation: WorkspaceOperationView<'_>,
) -> facts::FactResult<crate::backend::nn::tensor::GroupedOutputStorage> {
    use crate::backend::nn::tensor::GroupedOutputStorage;
    let Some(d) = Descriptor::inspect(operation)? else {
        return Ok(GroupedOutputStorage::default());
    };
    let chunks = d
        .chunks()
        .into_iter()
        .try_fold(0usize, |sum, (_, n)| sum.checked_add(n))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    // The actual chunk table is used after units, including a split Finish
    // trace. Unit callbacks are prepared once for the whole logical invocation.
    let chunked = chunks > 1 && d.phase != WorkspaceGroupedPhase::Units;
    Ok(GroupedOutputStorage {
        calls: usize::from(chunked),
        chunks: if chunked { chunks } else { 0 },
        unit_observers: usize::from(d.phase == WorkspaceGroupedPhase::Units),
        observer_shape_rank: d.input_rank.max(2),
    })
}

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let Some(d) = Descriptor::inspect(operation)? else {
        return Ok(None);
    };
    let before = d.phase != WorkspaceGroupedPhase::Finish;
    let after = d.phase != WorkspaceGroupedPhase::Units;
    let source = (|| {
        let mut total = Program::new(mechanism);
        if before && !matches!(d.activation, Activation::Relu2) {
            total.child(
                super::embedding::group_indices(
                    d.index,
                    2,
                    d.tokens.checked_mul(d.routes)?,
                    mechanism,
                )
                .ok()??,
            )?;
        }
        if before && !matches!(d.activation, Activation::Linear(_)) {
            total.reshape(d.input_rank, 2)?;
        }
        let chunks = d.chunks();
        let chunk_count = chunks
            .into_iter()
            .try_fold(0usize, |n, (_, k)| n.checked_add(k))?;
        for (tokens, count) in chunks {
            if count == 0 {
                continue;
            }
            let rows = tokens.checked_mul(d.routes)?;
            let mut chunk = Program::new(mechanism);
            if before {
                if chunk_count > 1 {
                    for _ in 0..3 {
                        chunk.slice(2)?;
                    }
                }
                chunk.selection(
                    tokens,
                    d.routes,
                    if matches!(d.activation, Activation::Relu2) {
                        d.index
                    } else {
                        Dtype::Int32
                    },
                )?;
                chunk.input_rows(tokens, d.routes, d.input)?;
                chunk.projection(rows, d.groups, d.first)?;
                chunk.activation(d, rows)?;
            }
            if after {
                if let Some(down) = d.down {
                    chunk.projection(rows, d.groups, down)?;
                }
                chunk.weighted(tokens, d.routes, d.output, d.reduction)?;
            }
            total.repeat(&chunk, count)?;
        }
        if after {
            total.concatenate(chunk_count, d.tokens.checked_mul(d.output)?)?;
            if d.separate_bias {
                total.bias_tail(d)?;
            }
            if !matches!(d.activation, Activation::Linear(_)) {
                for _ in 0..operation.outputs.len() {
                    total.reshape(2, d.input_rank)?;
                }
            }
        }
        if d.phase == WorkspaceGroupedPhase::Units {
            total.controls(crate::backend::nn::shared::MlxGroupedGatedProduct::original_unit_observation_control_bytes()?)?;
        }
        if before
            && matches!(
                operation.kind,
                WorkspaceOperationKindView::Grouped {
                    bank: WorkspaceGroupedBank::GatedProduct(_),
                    partitions: Some(_),
                    ..
                }
            )
        {
            total.controls(crate::backend::nn::shared::MlxGroupedGatedProduct::original_tensor_parallel_control_bytes()?)?;
        }
        // Fixed borrowed descriptor, native callback, selected plan and result
        // transports. The actual dynamic chunk/observer tables have their own
        // GroupedOutputStorage owner, composed by the CPU trace reducer.
        let frames = [
            size_of::<WorkspaceOperationView<'_>>(),
            size_of::<MlxCpuWorkspaceMechanisms>(),
            size_of::<Descriptor>() * 2,
            size_of::<Option<Descriptor>>(),
            size_of::<facts::FactResult<Option<Descriptor>>>(),
            size_of::<Program>() * 2,
            size_of::<[(usize, usize); 2]>(),
            size_of::<std::array::IntoIter<(usize, usize), 2>>(),
            size_of::<(usize, usize)>(),
            size_of::<usize>() * 12,
            size_of::<bool>() * 3,
            size_of::<WorkspaceLayoutView<'_>>() * 5,
            size_of::<Option<Projection>>() * 2,
            size_of::<(
                &eredu_nn::GroupedProjectionSpec,
                usize,
                usize,
                usize,
                WorkspaceOperationView<'_>,
                &mut usize,
            )>(),
            size_of::<facts::FactResult<Option<Projection>>>(),
            size_of::<[u64; 4]>() * 2,
            size_of::<OperationPlan>(),
            size_of::<Option<OperationPlan>>(),
            size_of::<Option<Program>>(),
            size_of::<crate::backend::nn::grouping::GroupedSelectionPlan>(),
            size_of::<(
                &safemlx::Array,
                &safemlx::Array,
                &safemlx::Array,
                &safemlx::Stream,
            )>(),
            size_of::<safemlx::Array>() * 7,
            size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
            size_of::<crate::backend::nn::tensor::GroupedOutputStorage>(),
            size_of::<facts::FactResult<crate::backend::nn::tensor::GroupedOutputStorage>>(),
        ];
        total.controls(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)?,
        )?;
        Some(total)
    })();
    let Some(total) = source else {
        return Ok(None);
    };
    let output_bytes = retained(d, mechanism)?
        .into_iter()
        .try_fold(0, facts::add)?;
    let scratch_bytes = total.bytes.checked_sub(output_bytes).ok_or_else(invalid)?;
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population: total.native,
        alias_input: None,
        output_bytes,
        scratch_bytes,
        rank: d.input_rank.max(3),
        parameter_shells: total.shells,
        seeds: total.seeds,
        validations: total.validations,
    }))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
