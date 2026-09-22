//! Census of the unchanged prepared spatial rotary worker, including widened
//! signed coordinates and its Graph-owned row sink.
use super::program::Program;
use super::*;
use eredu_nn::multimodal::{MultiAxisRotaryLayout, MultiAxisRotarySpecRef};
use safemlx::{CpuUnaryOperation, Dtype};

fn seed(program: &mut Program, dtype: Dtype, count: usize) -> Option<()> {
    program.seeds = program.seeds.checked_add(1)?;
    program.bytes = program.bytes.checked_add(program.capacity(count, dtype)?)?;
    program.controls(size_of::<(&mut Program, Dtype, usize, Option<()>)>())
}

fn join(program: &mut Program, dtype: Dtype, inputs: usize, count: usize) -> Option<()> {
    if inputs == 1 {
        // concatenate's singleton branch retains its input; the row sink is
        // separately counted by its actual OriginalArrayRowsLayout.
        program.shells = program.shells.checked_add(1)?;
    } else {
        let source = OperationEvent::cpu_concatenate_many_layout(dtype, 2, inputs, count, false)?;
        program.bytes = program.bytes.checked_add(
            program
                .capacity(count, dtype)?
                .checked_mul(source.backing_births() as u64)?,
        )?;
        program.native.concatenate(source, inputs)?;
    }
    program.controls(size_of::<(
        &mut Program,
        Dtype,
        usize,
        usize,
        CpuCopyEvalLayout,
        Option<()>,
    )>())
}

fn column(program: &mut Program, rows: usize, axes: usize, signed: bool) -> Option<()> {
    // The same tuple-index worker keeps the sliced column's strides while
    // removing its unit axis. Arithmetic then writes dense I64 coordinates.
    program.slice(2)?;
    let rows_i32 = i32::try_from(rows).ok()?;
    program.copy(
        OperationEvent::cpu_reshape_layout(
            &[rows_i32, 1],
            &[i64::try_from(axes).ok()?, 1],
            &[rows_i32],
            false,
        )?,
        1,
        rows,
        Dtype::Int64,
    )?;
    seed(program, Dtype::Int64, 1)?;
    program.binary(
        CpuBinaryOperation::Add,
        Dtype::Int64,
        1,
        rows,
        1,
        0,
        rows,
        1,
    )?;
    if signed {
        for operation in [CpuBinaryOperation::Maximum, CpuBinaryOperation::Minimum] {
            seed(program, Dtype::Int64, 1)?;
            program.binary(operation, Dtype::Int64, 1, rows, 1, 0, rows, 1)?;
        }
    }
    seed(program, Dtype::Int64, 1)?;
    program.binary(
        CpuBinaryOperation::Maximum,
        Dtype::Int64,
        1,
        rows,
        1,
        0,
        rows,
        1,
    )?;
    program.controls(size_of::<(
        &mut Program,
        usize,
        usize,
        bool,
        i32,
        [i32; 2],
        [i64; 2],
        [i32; 1],
        [CpuBinaryOperation; 2],
        Option<()>,
    )>())
}

fn population(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
    spec: MultiAxisRotarySpecRef<'_>,
    rows: usize,
    dimensions: usize,
) -> Option<OperationPlan> {
    let input = operation.inputs.get(0)?;
    let rank = input.shape().len();
    let count = usize::try_from(input.elements().ok()?).ok()?;
    let full = rows.checked_mul(dimensions)?;
    let half = dimensions / 2;
    let signed = input.dtype() == WorkspaceDtype::Int32;
    let dtype = if signed { Dtype::Int32 } else { Dtype::Uint32 };
    let profile = crate::tensor::PreparedRotaryProfile::inspect(spec, rank)?;
    let capacity = if spec.layout == MultiAxisRotaryLayout::RoundRobinSections {
        half
    } else {
        spec.axes.len()
    };
    let row_bank = safemlx::ops::OriginalArrayRowsLayout::inspect(capacity, rank)?;
    let mut program = Program::new(mechanism);
    // Integer layouts carry no contiguity claim. Quote the real flattening
    // reshape's copy alternative, before its I64 conversion.
    program.copy(
        OperationEvent::cpu_reshape_copy_layout(rank, 2, false)?,
        1,
        count,
        dtype,
    )?;
    program.cast(dtype, Dtype::Int64, 2, count)?;
    match spec.layout {
        MultiAxisRotaryLayout::IndependentAxes | MultiAxisRotaryLayout::SplitHalves => {
            for axis in spec.axes {
                let frequencies = usize::try_from(axis.dimensions / 2).ok()?;
                seed(&mut program, Dtype::Float32, frequencies)?;
                program.copy(
                    OperationEvent::cpu_copy_alias_layout(2, false)?,
                    1,
                    frequencies,
                    Dtype::Float32,
                )?;
                column(&mut program, rows, spec.axes.len(), signed)?;
                program.cast(Dtype::Int64, Dtype::Float32, 1, rows)?;
                program.copy(
                    OperationEvent::cpu_expand_dims_alias_layout(1, 2, false)?,
                    1,
                    0,
                    Dtype::Float32,
                )?;
                let elements = rows.checked_mul(frequencies)?;
                program.binary(
                    CpuBinaryOperation::Multiply,
                    Dtype::Float32,
                    2,
                    elements,
                    2,
                    2,
                    rows,
                    frequencies,
                )?;
                if spec.layout == MultiAxisRotaryLayout::IndependentAxes {
                    join(&mut program, Dtype::Float32, 2, elements.checked_mul(2)?)?;
                }
            }
            let joined = if spec.layout == MultiAxisRotaryLayout::SplitHalves {
                full / 2
            } else {
                full
            };
            join(&mut program, Dtype::Float32, spec.axes.len(), joined)?;
            if spec.layout == MultiAxisRotaryLayout::SplitHalves {
                join(&mut program, Dtype::Float32, 2, full)?;
            }
        }
        MultiAxisRotaryLayout::RoundRobinSections => {
            for _ in 0..half {
                column(&mut program, rows, spec.axes.len(), signed)?;
                program.copy(
                    OperationEvent::cpu_expand_dims_alias_layout(1, 2, false)?,
                    1,
                    0,
                    Dtype::Int64,
                )?;
            }
            let elements = rows.checked_mul(half)?;
            join(&mut program, Dtype::Int64, half, elements)?;
            seed(&mut program, Dtype::Float32, half)?;
            program.copy(
                OperationEvent::cpu_copy_alias_layout(2, false)?,
                1,
                half,
                Dtype::Float32,
            )?;
            program.cast(Dtype::Int64, Dtype::Float32, 2, elements)?;
            program.binary(
                CpuBinaryOperation::Multiply,
                Dtype::Float32,
                2,
                elements,
                2,
                2,
                elements,
                half,
            )?;
            join(&mut program, Dtype::Float32, 2, full)?;
        }
    }
    for operation in [CpuUnaryOperation::Cosine, CpuUnaryOperation::Sine] {
        program.cast(Dtype::Float32, Dtype::Float32, 2, full)?;
        program.unary(operation, Dtype::Float32, 2, full)?;
        program.reshape(2, rank)?;
    }
    program.native.construction_entries = program
        .native
        .construction_entries
        .checked_add(row_bank.resident_controls())?;
    program.native.maximum_operands = program
        .native
        .maximum_operands
        .max(row_bank.maximum_operands());
    program.shells = program.shells.checked_add(profile.cloned_handles)?;
    program.controls(profile.control_bytes()?)?;
    program.controls(size_of::<(
        WorkspaceOperationView<'_>,
        MlxCpuWorkspaceMechanisms,
        MultiAxisRotarySpecRef<'_>,
        WorkspaceLayoutView<'_>,
        crate::tensor::PreparedRotaryProfile,
        safemlx::ops::OriginalArrayRowsLayout,
        Program,
        Option<OperationPlan>,
        [usize; 12],
        [u64; 3],
        Dtype,
        bool,
        std::slice::Iter<'_, eredu_nn::multimodal::RotaryAxisSpec>,
        std::ops::Range<usize>,
        [CpuUnaryOperation; 2],
    )>())?;
    let output_bytes = program.capacity(full, Dtype::Float32)?.checked_mul(2)?;
    Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population: program.native,
        alias_input: None,
        output_bytes,
        scratch_bytes: program.bytes.checked_sub(output_bytes)?,
        rank,
        parameter_shells: program.shells,
        seeds: program.seeds,
        validations: 0,
    })
}

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let WorkspaceOperationKindView::PreparedMultiAxisRotary(spec) = operation.kind else {
        return Ok(None);
    };
    let dimensions = spec.dimensions()?;
    let Some([input]) = operation.inputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU prepared rotary input population differs",
        ));
    };
    let shape = input.shape();
    if shape.len() < 2
        || shape.iter().any(|&n| n < 0)
        || shape.last().copied() != i32::try_from(spec.axes.len()).ok()
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU prepared rotary input geometry differs",
        ));
    }
    if shape.len() > 5
        || shape.contains(&0)
        || !matches!(
            input.dtype(),
            WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
        )
    {
        return Ok(None);
    }
    if operation.outputs.len() != 2
        || operation.outputs.iter().any(|output| {
            output.dtype() != WorkspaceDtype::Float32
                || !output.shape().iter().copied().eq(shape[..shape.len() - 1]
                    .iter()
                    .copied()
                    .chain(std::iter::once(dimensions)))
        })
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU prepared rotary output geometry differs",
        ));
    }
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1usize, |rows, &n| {
            rows.checked_mul(n as usize)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)
        })?;
    if input.elements()? > i32::MAX as u64
        || rows
            .checked_mul(dimensions as usize)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?
            > i32::MAX as usize
    {
        return Ok(None);
    }
    Ok(population(
        operation,
        mechanism,
        spec,
        rows,
        dimensions as usize,
    ))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
