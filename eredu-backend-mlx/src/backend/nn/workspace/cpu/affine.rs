//! Packed CPU affine projections use the native QMM worker and its own Eval bank.
use super::*;
use eredu_checkpoint::LinearFormat;
use eredu_nn::LinearFormatSpec;
use safemlx::Dtype;

fn native(dtype: WorkspaceFloatingType) -> Dtype {
    match dtype {
        WorkspaceFloatingType::Float32 => Dtype::Float32,
        WorkspaceFloatingType::Float16 => Dtype::Float16,
        WorkspaceFloatingType::Bfloat16 => Dtype::Bfloat16,
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
#[path = "affine/tests.rs"]
mod tests;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let WorkspaceOperationKindView::Projection(format) = operation.kind else {
        return Ok(None);
    };
    let LinearFormat::Affine(config) = format.encoding() else {
        return Ok(None);
    };
    format.validate_fixed()?;
    // Shares the physical projection's shape, role and scalar promotion rules.
    let Some(output_representation) = super::super::representation::output(operation, 0) else {
        return Ok(None);
    };
    let input = operation.inputs.get(0).expect("validated affine inputs");
    let weight = operation.inputs.get(1).expect("validated affine inputs");
    let output = operation.outputs.get(0).expect("validated affine output");
    let rank = input.shape().len();
    if !(2..=4).contains(&rank)
        || operation
            .inputs
            .iter()
            .any(|v| v.shape().iter().any(|&n| n <= 0))
    {
        return Ok(None);
    }
    // A published floating edit takes the existing dense branch; its retained
    // packed companions do not participate in arithmetic or widen precision.
    if weight.dtype() == WorkspaceDtype::Float32 {
        let dense_format = LinearFormatSpec::unscaled(LinearFormat::Dense)
            .map_err(|_| MlxWorkspaceFactError::descriptor("invalid dense projection encoding"))?;
        let mut inputs = [input, weight, input];
        let count = if let Some(bias) = operation.inputs.get(4) {
            inputs[2] = bias;
            3
        } else {
            2
        };
        let mut plan = dense::inspect(
            WorkspaceOperationView {
                kind: WorkspaceOperationKindView::Projection(&dense_format),
                inputs: WorkspaceLayoutList::Views(&inputs[..count]),
                ..operation
            },
            mechanism,
        )?;
        if let Some(plan) = plan.as_mut() {
            plan.population.controls = plan
                .population
                .controls
                .checked_add(
                    size_of_val(&inputs)
                        + size_of_val(&dense_format)
                        + size_of::<usize>()
                        + size_of::<WorkspaceOperationView<'_>>()
                        + size_of::<Option<OperationPlan>>(),
                )
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        }
        return Ok(plan);
    }
    let Some(source_dtype) = input.representation().map(|r| r.dtype()) else {
        return Ok(None);
    };
    let mut matrix_dtype = source_dtype;
    for ordinal in [0, 2, 3] {
        let layout = operation
            .inputs
            .get(ordinal)
            .expect("validated affine companions");
        let Some(representation) = layout.representation() else {
            return Ok(None);
        };
        if !representation.row_contiguous() {
            return Ok(None);
        }
        matrix_dtype = super::super::representation::promote(matrix_dtype, representation.dtype());
    }
    if operation
        .inputs
        .get(4)
        .is_some_and(|bias| !bias.representation().is_some_and(|r| r.row_contiguous()))
    {
        return Ok(None);
    }
    let Some(rows) = input.shape()[..rank - 1]
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
    else {
        return Ok(None);
    };
    let width = input.shape()[rank - 1] as usize;
    let columns = output.shape()[rank - 1] as usize;
    let dtype = output_representation.dtype();
    let mut program = program::Program::new(mechanism);
    let accepted = (|| {
        for ordinal in [0, 2, 3] {
            let layout = operation.inputs.get(ordinal)?;
            let from = layout.representation()?.dtype();
            if from != matrix_dtype {
                program.cast(
                    native(from),
                    native(matrix_dtype),
                    layout.shape().len(),
                    usize::try_from(layout.elements().ok()?).ok()?,
                )?;
            }
        }
        let elements = rows.checked_mul(columns)?;
        let qmm = OperationEvent::cpu_affine_quantized_layout(
            native(matrix_dtype),
            rank,
            rows,
            columns,
            width,
            config.group_size,
            config.bits,
            false,
        )?;
        if qmm.backing_births() != 1 {
            return None;
        }
        program.copy(qmm, 4, elements, native(matrix_dtype))?;
        // One output Data and the five actual weak task-array Data aliases.
        program.native.maximum_captures = program.native.maximum_captures.max(6);
        if let Some(bias) = operation.inputs.get(4) {
            if matrix_dtype != dtype {
                program.cast(native(matrix_dtype), native(dtype), rank, elements)?;
            }
            let bias_dtype = bias.representation()?.dtype();
            if bias_dtype != dtype {
                program.cast(native(bias_dtype), native(dtype), 1, columns)?;
            }
            program.native.copy(
                OperationEvent::cpu_broadcast_alias_layout(1, rank, false)?,
                1,
            )?;
            let add = OperationEvent::cpu_binary_layout(
                CpuBinaryOperation::Add,
                native(dtype),
                rank,
                elements,
                false,
            )?;
            program.native.binary(add)?;
            program.bytes = program
                .bytes
                .checked_add(program.capacity(elements, native(dtype))?)?;
        }
        Some(())
    })();
    if accepted.is_none() {
        return Ok(None);
    }
    let output_bytes = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(output.elements()?, 4)?)?;
    let frames = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 4,
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<program::Program>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<Option<()>>(),
        size_of::<WorkspaceFloatingType>() * 5,
        size_of::<Dtype>() * 4,
        size_of::<WorkspaceRepresentation>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<usize>() * 8,
        size_of::<u64>() * 3,
        size_of::<bool>() * 2,
        size_of::<std::array::IntoIter<usize, 3>>(),
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<(
            &safemlx::Array,
            &safemlx::Array,
            &safemlx::Array,
            Option<&safemlx::Array>,
            bool,
            i32,
            i32,
            safemlx::ops::QuantizationMode,
            &safemlx::Stream,
        )>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
    ];
    program
        .controls(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        dtype,
        population: program.native,
        alias_input: None,
        output_bytes,
        scratch_bytes: program
            .bytes
            .checked_sub(output_bytes)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        rank,
        parameter_shells: 1,
        seeds: 0,
        validations: 0,
    }))
}
