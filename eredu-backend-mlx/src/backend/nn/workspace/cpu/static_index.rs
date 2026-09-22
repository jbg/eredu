//! Static integer-axis indexing uses the shared Slice and Reshape workers.
use super::*;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let WorkspaceOperationKindView::Index { selected_axes } = operation.kind else {
        return Ok(None);
    };
    // Rank-preserving ranges carry their exact coordinates in StaticSlice.
    if selected_axes == 0 {
        return Ok(None);
    }
    let Some([input]) = operation.inputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU static index input differs",
        ));
    };
    let Some([output]) = operation.outputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU static index output differs",
        ));
    };
    let rank = input.shape().len();
    let output_rank = output.shape().len();
    if rank.checked_sub(selected_axes) != Some(output_rank) || input.dtype() != output.dtype() {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU static index rank or scalar differs",
        ));
    }
    // Index accepts only Full, Range and At. The shared descriptor constructor
    // checks their coordinates. Every retained range has unit positive stride;
    // this source uses the complete possible Reshape-copy branch because the
    // removed axes and physical slice strides are not retained in this record.
    if !(1..=4).contains(&rank)
        || input.shape().iter().chain(output.shape()).any(|&n| n <= 0)
        || input.elements()? > i32::MAX as u64
        || output.elements()? > input.elements()?
    {
        return Ok(None);
    }
    let Some(slice) = OperationEvent::cpu_slice_layout(rank, false, false) else {
        return Ok(None);
    };
    let Some(reshape) = OperationEvent::cpu_reshape_copy_layout(rank, output_rank, false) else {
        return Ok(None);
    };
    let mut population = CpuPopulation::default();
    if slice.backing_births() != 0
        || reshape.backing_births() != 1
        || population.copy(slice, 1).is_none()
        || population.copy(reshape, 1).is_none()
    {
        return Ok(None);
    }
    let frames = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<[WorkspaceLayoutView<'_>; 1]>() * 2,
        size_of::<Option<[WorkspaceLayoutView<'_>; 1]>>() * 2,
        size_of::<CpuPopulation>(),
        size_of::<CpuCopyEvalLayout>() * 2,
        size_of::<Option<CpuCopyEvalLayout>>() * 2,
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<usize>() * 3,
        size_of::<u64>() * 2,
        size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<WorkspaceRepresentation>(),
        size_of::<std::iter::Chain<std::slice::Iter<'_, i32>, std::slice::Iter<'_, i32>>>(),
    ];
    population.controls = frames
        .into_iter()
        .try_fold(
            population
                .controls
                .checked_add(size_of_val(&frames))
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            usize::checked_add,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        // An absent floating representation stays absent in the public output
        // facts. The normalized Float32 descriptor supplies only its documented
        // four-byte envelope; these dtype-preserving workers never widen it.
        dtype: input
            .representation()
            .map_or(WorkspaceFloatingType::Float32, |r| r.dtype()),
        population,
        alias_input: None,
        output_bytes: mechanism
            .allocation
            .fixed_buffer_capacity(output.bytes()?)?,
        scratch_bytes: 0,
        rank,
        // A whole Slice returns an additional source handle without a primitive.
        // Its unused primitive population remains an allowance, not a birth.
        parameter_shells: 1,
        seeds: 0,
        validations: 0,
    }))
}

pub(super) fn representation(
    operation: WorkspaceOperationView<'_>,
) -> Option<WorkspaceRepresentation> {
    // Either branch preserves precision. Neither the erased axes nor a possible
    // copy establish common row/stride evidence across alias and copy outcomes.
    operation
        .inputs
        .get(0)?
        .representation()
        .map(|r| WorkspaceRepresentation::new(r.dtype(), false))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use eredu_nn::{Index, Tensor};
    #[test]
    fn static_index_preserves_unknown_precision_and_full_alias_custody() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(
            ordinary.allocation(),
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap(),
        );
        for representation in [
            None,
            Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float16,
                true,
            )),
            Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                false,
            )),
        ] {
            let context = WorkspaceContext::new(cpu);
            let storage = WorkspaceExistingStorage::try_new(Some(4096), &context).unwrap();
            let source = WorkspaceTensor::existing_with_storage(
                context
                    .layout(&[2, 3, 8], WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(representation),
                &storage,
                &context,
            )
            .unwrap();
            context.begin_state_span([&source]).unwrap();
            let row = source
                .index(&[Index::Full, Index::At(1), Index::Full], &context)
                .unwrap();
            assert_eq!(row.shape(), [2, 8]);
            assert_eq!(
                row.layout().representation().map(|r| r.dtype()),
                representation.map(|r| r.dtype())
            );
            assert!(!row
                .layout()
                .representation()
                .is_some_and(|r| r.row_contiguous()));
            let report = context.finish_report(&[row]).unwrap();
            let operation = report.operations[0].as_view();
            let plan = cpu.plan(operation).unwrap().unwrap();
            assert_eq!(plan.population.primitives, 2);
            assert_eq!(plan.population.births, 1);
            assert_eq!(plan.alias_input, None);
            assert!(plan.output_bytes >= 64);
            assert!(
                report.closing_storage.bytes.unwrap() >= 4096,
                "possible reshape copy must preserve the original alias backing"
            );
            let malformed = WorkspaceOperationView {
                kind: WorkspaceOperationKindView::Index { selected_axes: 2 },
                ..operation
            };
            assert!(cpu.plan(malformed).is_err());
        }
    }
}
