//! Default-zero padding through the existing scalar fill and rectangular copy.
use super::*;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    if !matches!(
        operation.kind,
        WorkspaceOperationKindView::Pad(eredu_nn::PadMode::Constant)
    ) {
        return Ok(None);
    }
    let Some([input]) = operation.inputs.array() else {
        return Ok(None);
    };
    let Some([output]) = operation.outputs.array() else {
        return Ok(None);
    };
    let rank = input.shape().len();
    if !(1..=4).contains(&rank)
        || output.shape().len() != rank
        || input.dtype() != WorkspaceDtype::Float32
        || output.dtype() != input.dtype()
        || input
            .shape()
            .iter()
            .zip(output.shape())
            .any(|(&a, &b)| a <= 0 || b < a)
    {
        return Ok(None);
    }
    let Some(representation) = input.representation() else {
        return Ok(None);
    };
    if !representation.row_contiguous() {
        return Ok(None);
    }
    let dtype = representation.dtype();
    let native = match dtype {
        WorkspaceFloatingType::Float32 => safemlx::Dtype::Float32,
        WorkspaceFloatingType::Float16 => safemlx::Dtype::Float16,
        WorkspaceFloatingType::Bfloat16 => safemlx::Dtype::Bfloat16,
    };
    let elements = usize::try_from(output.elements()?)?;
    let input_elements = usize::try_from(input.elements()?)?;
    let Some(pad) = OperationEvent::cpu_constant_pad_layout(rank, elements, input_elements, false)
    else {
        return Ok(None);
    };
    let Some(cast) = OperationEvent::cpu_cast_layout(safemlx::Dtype::Int32, native, 0, 1, false)
    else {
        return Ok(None);
    };
    let mut population = CpuPopulation::default();
    if population.copy(cast, 1).is_none() || population.copy(pad, 2).is_none() {
        return Ok(None);
    }
    // Pad publishes output Data and two pairs of weak copy captures. Its scalar
    // is an eager I32 seed plus one real dtype conversion; the native second
    // astype is an identity and adds no numerical backing.
    population.maximum_captures = population.maximum_captures.max(5);
    population.construction_entries = population
        .construction_entries
        .checked_add(1)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let frames = [
        safemlx::ops::constant_pad_control_bytes(rank)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        size_of::<(WorkspaceOperationView<'_>, MlxCpuWorkspaceMechanisms)>(),
        size_of::<[WorkspaceLayoutView<'_>; 1]>() * 2,
        size_of::<Option<[WorkspaceLayoutView<'_>; 1]>>() * 2,
        size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<WorkspaceFloatingType>(),
        size_of::<safemlx::Dtype>(),
        size_of::<CpuCopyEvalLayout>() * 2,
        size_of::<Option<CpuCopyEvalLayout>>() * 2,
        size_of::<CpuPopulation>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<usize>() * 3,
        size_of::<Result<usize, std::num::TryFromIntError>>(),
        size_of::<std::iter::Zip<std::slice::Iter<'_, i32>, std::slice::Iter<'_, i32>>>(),
        size_of::<(i32, i32)>(),
        size_of::<u64>() * 2,
        size_of::<bool>(),
    ];
    population.controls = frames.into_iter().try_fold(
        population
            .controls
            .checked_add(size_of_val(&frames))
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        |n, b| {
            n.checked_add(b)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)
        },
    )?;
    Ok(Some(OperationPlan {
        alias_input: None,
        dtype,
        population,
        rank,
        parameter_shells: 0,
        seeds: 1,
        validations: 0,
        output_bytes: mechanism
            .allocation
            .fixed_buffer_capacity(facts::mul(output.elements()?, 4)?)?,
        scratch_bytes: facts::mul(mechanism.allocation.fixed_buffer_capacity(4)?, 2)?,
    }))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use eredu_nn::Tensor;
    #[test]
    fn constant_pad_has_one_destination_and_paid_fill_scalar() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        let context = WorkspaceContext::new(cpu);
        let input = WorkspaceTensor::existing(
            context
                .layout(&[1, 2, 64], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                ))),
            &context,
        )
        .unwrap();
        context.begin_span();
        let output = WorkspaceTensor::pad(
            &input,
            &[(0, 0), (3, 0), (0, 0)],
            eredu_nn::PadMode::Constant,
            &context,
        )
        .unwrap();
        let report = context.report(&[output]).unwrap();
        let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
        assert_eq!(plan.population.births, 2);
        assert_eq!(plan.seeds, 1);
        SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context)
            .unwrap();
    }
}
