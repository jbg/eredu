//! The unchanged beta-scaled NeuralBackend softplus, including its Bool branch.
use super::*;
use safemlx::{CpuUnaryOperation, Dtype};

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    if !matches!(
        operation.kind,
        WorkspaceOperationKindView::Elementwise("softplus")
    ) {
        return Ok(None);
    }
    if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU softplus operand population differs",
        ));
    }
    let input = operation.inputs.get(0).expect("one softplus input");
    let output = operation.outputs.get(0).expect("one softplus output");
    if input.shape() != output.shape() {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU softplus output geometry differs",
        ));
    }
    let rank = input.shape().len();
    if rank > 4
        || input.shape().iter().any(|&n| n <= 0)
        || input.dtype() != WorkspaceDtype::Float32
        || output.dtype() != WorkspaceDtype::Float32
    {
        return Ok(None);
    }
    let Some(representation) = input.representation().filter(|r| r.row_contiguous()) else {
        return Ok(None);
    };
    let dtype = representation.dtype();
    let native = match dtype {
        WorkspaceFloatingType::Float32 => Dtype::Float32,
        WorkspaceFloatingType::Bfloat16 => Dtype::Bfloat16,
        WorkspaceFloatingType::Float16 => Dtype::Float16,
    };
    let count = usize::try_from(input.elements()?)?;
    if count > i32::MAX as usize {
        return Ok(None);
    }
    let source = (|| {
        let mut population = CpuPopulation::default();
        // F32 full tensors, Bool full tensors, and stored F32 scalars have
        // different owning capacities. The returned storage envelope is below.
        let mut births = [0usize; 3];
        let wide = OperationEvent::cpu_cast_layout(native, Dtype::Float32, rank, count, false)?;
        births[0] = wide.backing_births();
        population.copy(wide, 1)?;
        for (step, kind) in [
            CpuBinaryOperation::Multiply,
            CpuBinaryOperation::Divide,
            CpuBinaryOperation::Greater,
        ]
        .into_iter()
        .enumerate()
        {
            // Each binary frontend casts and broadcasts its full left operand
            // and actual eager scalar, including identity constructor envelopes.
            for scalar in [false, true] {
                let (r, n, bucket) = if scalar { (0, 1, 2) } else { (rank, count, 0) };
                let cast =
                    OperationEvent::cpu_cast_layout(Dtype::Float32, Dtype::Float32, r, n, false)?;
                births[bucket] = births[bucket].checked_add(cast.backing_births())?;
                population.copy(cast, 1)?;
                population.copy(
                    OperationEvent::cpu_broadcast_alias_layout(r, rank, false)?,
                    1,
                )?;
            }
            let binary =
                OperationEvent::cpu_binary_layout(kind, Dtype::Float32, rank, count, false)?;
            let bucket = if step == 2 { 1 } else { 0 };
            births[bucket] = births[bucket].checked_add(binary.backing_births())?;
            population.binary(binary)?;
            if step == 0 {
                // exp and log1p both call astype(at_least_float) before their
                // unchanged unary task. The scaled input also survives for gt.
                for kind in [CpuUnaryOperation::Exponential, CpuUnaryOperation::Log1p] {
                    let cast = OperationEvent::cpu_cast_layout(
                        Dtype::Float32,
                        Dtype::Float32,
                        rank,
                        count,
                        false,
                    )?;
                    births[0] = births[0].checked_add(cast.backing_births())?;
                    population.copy(cast, 1)?;
                    let unary =
                        OperationEvent::cpu_unary_layout(kind, Dtype::Float32, rank, false)?;
                    births[0] = births[0].checked_add(unary.backing_births())?;
                    population.unary(unary)?;
                }
            }
        }
        // where casts the Bool condition and both F32 branches, then broadcasts
        // all three. Never price the predicate as a floating backing allocation.
        for native in [Dtype::Bool, Dtype::Float32, Dtype::Float32] {
            let cast = OperationEvent::cpu_cast_layout(native, native, rank, count, false)?;
            let bucket = usize::from(native == Dtype::Bool);
            births[bucket] = births[bucket].checked_add(cast.backing_births())?;
            population.copy(cast, 1)?;
            population.copy(
                OperationEvent::cpu_broadcast_alias_layout(rank, rank, false)?,
                1,
            )?;
        }
        let select = OperationEvent::cpu_select_layout(Dtype::Float32, rank, count, false)?;
        births[0] = births[0].checked_add(select.backing_births())?;
        population.copy(select, 3)?;
        let final_cast =
            OperationEvent::cpu_cast_layout(Dtype::Float32, native, rank, count, false)?;
        if final_cast.backing_births() != 1 {
            return None;
        }
        population.copy(final_cast, 1)?;
        // Select publishes its output Data and four retained weak input/output
        // handles in one Eval. Cast/unary/binary workers have smaller peaks.
        population.maximum_captures = population.maximum_captures.max(5);
        Some((population, births))
    })();
    let Some((mut population, births)) = source else {
        return Ok(None);
    };
    let full = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(input.elements()?, 4)?)?;
    let predicate = mechanism
        .allocation
        .fixed_buffer_capacity(input.elements()?)?;
    let scalar = mechanism.allocation.fixed_buffer_capacity(4)?;
    // Workspace floating layouts retain their logical F32 storage floor even
    // when the native result uses BF16/F16. Preserve that shared declaration
    // contract while reporting the actual returned precision through `dtype`.
    let output_bytes = full;
    let scratch_bytes = facts::add(
        facts::add(
            facts::mul(full, u64::try_from(births[0])?)?,
            facts::mul(predicate, u64::try_from(births[1])?)?,
        )?,
        facts::mul(
            scalar,
            u64::try_from(
                births[2]
                    .checked_add(3)
                    .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            )?,
        )?,
    )?;
    let frames = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<WorkspaceFloatingType>(),
        size_of::<Dtype>() * 2,
        size_of::<(&usize, &usize, &Dtype)>(),
        size_of::<CpuPopulation>(),
        size_of::<[usize; 3]>(),
        size_of::<Option<(CpuPopulation, [usize; 3])>>(),
        size_of::<CpuCopyEvalLayout>() * 4,
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<[CpuBinaryOperation; 3]>(),
        size_of::<std::iter::Enumerate<std::array::IntoIter<CpuBinaryOperation, 3>>>(),
        size_of::<(usize, CpuBinaryOperation)>(),
        size_of::<std::array::IntoIter<bool, 2>>(),
        size_of::<[bool; 2]>(),
        size_of::<(usize, usize, usize)>(),
        size_of::<[CpuUnaryOperation; 2]>(),
        size_of::<std::array::IntoIter<CpuUnaryOperation, 2>>(),
        size_of::<CpuUnaryOperation>(),
        size_of::<[Dtype; 3]>(),
        size_of::<std::array::IntoIter<Dtype, 3>>(),
        size_of::<usize>() * 4,
        size_of::<u64>() * 5,
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        // Actual shared native softplus caller and its live result transports.
        size_of::<(crate::MlxTensor, f32, &safemlx::Stream)>(),
        size_of::<safemlx::Array>() * 5,
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>() * 2,
        size_of::<Result<crate::MlxTensor, eredu_nn::Error>>(),
    ];
    population.controls = frames.into_iter().try_fold(
        population
            .controls
            .checked_add(size_of_val(&frames))
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        |n, bytes| {
            n.checked_add(bytes)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)
        },
    )?;
    Ok(Some(OperationPlan {
        dtype,
        population,
        output_bytes,
        scratch_bytes,
        rank,
        alias_input: None,
        parameter_shells: 0,
        seeds: 3,
        validations: 0,
    }))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
