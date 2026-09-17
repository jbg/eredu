//! Exact integer views and the shared isolated text-matrix cast used on resume.
use super::*;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let cast = match operation.kind {
        WorkspaceOperationKindView::View("reshape") => false,
        WorkspaceOperationKindView::Elementwise("cast_u32") => true,
        _ => return Ok(None),
    };
    if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return Ok(None);
    }
    let input = operation.inputs.get(0).expect("one integer source");
    let output = operation.outputs.get(0).expect("one integer output");
    if !matches!(input.dtype(), WorkspaceDtype::Int32 | WorkspaceDtype::Uint32) {
        return Ok(None);
    }
    let (rank, output_rank) = (input.shape().len(), output.shape().len());
    if rank > 5 || output_rank > 5
        || input.shape().iter().chain(output.shape()).any(|&n| n <= 0)
    {
        return Ok(None);
    }
    let elements = input.elements()?;
    if elements > i32::MAX as u64 || elements != output.elements()? {
        return Ok(None);
    }
    let mut population = CpuPopulation::default();
    let mut shells = 0;
    if cast {
        // Reuse the same closed source contract as the native pending worker:
        // isolate, reshape [1,N], then I32 -> U32. This name never denotes a
        // general dtype conversion, nor does it give an integer a floating fact.
        if !basic::is_isolated_text_u32_cast_view(operation) {
            return Ok(None);
        }
        let Some(source) = OperationEvent::cpu_cast_layout(
            safemlx::Dtype::Int32, safemlx::Dtype::Uint32,
            rank, usize::try_from(elements)?, false,
        ) else { return Ok(None); };
        if source.backing_births() != 1 || population.copy(source, 1).is_none() {
            return Ok(None);
        }
    } else {
        if output.dtype() != input.dtype() { return Ok(None); }
        if input.shape() == output.shape() {
            // ops::reshape returns its input before creating a primitive.
            shells = 1;
        } else if elements == 1 {
            // All coordinates are singleton, so every readable physical stride
            // describes the same element. The existing fixed planner covers
            // both the row flag and non-row singleton alias branches. No
            // general integer row/stride representation is inferred here.
            let strides = [1i64; 5];
            let Some(source) = OperationEvent::cpu_reshape_layout(
                input.shape(), &strides[..rank], output.shape(), false,
            ) else { return Ok(None); };
            if source.backing_births() != 0 || population.copy(source, 1).is_none() {
                return Ok(None);
            }
        } else {
            return Ok(None);
        }
    }
    let frames = [
        size_of::<WorkspaceOperationView<'_>>(), size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<MlxCpuWorkspaceMechanisms>(), size_of::<CpuPopulation>(),
        size_of::<OperationPlan>(), size_of::<Option<OperationPlan>>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<CpuCopyEvalLayout>(), size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<WorkspaceDtype>() * 2, size_of::<safemlx::Dtype>() * 2,
        size_of::<usize>() * 4, size_of::<u64>(), size_of::<bool>() * 2,
        size_of::<[i64; 5]>(), size_of::<(usize, usize)>(),
        size_of::<std::slice::Iter<'_, i32>>() * 2,
        size_of::<std::iter::Chain<std::slice::Iter<'_, i32>, std::slice::Iter<'_, i32>>>(),
        size_of::<(&[i32], &[i64], &[i32], bool)>(),
        size_of::<(&safemlx::Array, &[i32], &safemlx::Stream)>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<Result<usize, std::num::TryFromIntError>>(),
        if cast { safemlx::Array::as_dtype_control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)? } else { 0 },
    ];
    population.controls = frames.into_iter().try_fold(
        population.controls.checked_add(size_of_val(&frames))
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        usize::checked_add,
    ).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        // Only the physical four-byte envelope; output_representation never
        // publishes a floating representation for these integer destinations.
        dtype: WorkspaceFloatingType::Float32, population,
        alias_input: if cast { None } else { Some(0) },
        output_bytes: if cast { mechanism.allocation.fixed_buffer_capacity(output.bytes()?)? } else { 0 },
        scratch_bytes: 0, rank: rank.max(output_rank), parameter_shells: shells,
        seeds: 0, validations: 0,
    }))
}
