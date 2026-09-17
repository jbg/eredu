//! Complete rank-one capture count and moment/extremum reduction sources.
use super::*;

pub(super) fn inspect(operation: WorkspaceOperationView<'_>, mechanism: MlxCpuWorkspaceMechanisms)
    -> facts::FactResult<Option<OperationPlan>> {
    let name = match operation.kind {
        WorkspaceOperationKindView::Reduction(name @ ("sum" | "min" | "max"), 0, false) => name,
        _ => return Ok(None),
    };
    let Some([input]) = operation.inputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor("CPU flat reduction input population differs"));
    };
    let Some([output]) = operation.outputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor("CPU flat reduction output population differs"));
    };
    if input.shape().len() != 1 { return Ok(None); }
    if !output.shape().is_empty() || input.dtype() != output.dtype() {
        return Err(MlxWorkspaceFactError::descriptor("CPU flat reduction changes scalar dtype or geometry"));
    }
    let unsigned = input.dtype() == WorkspaceDtype::Uint32;
    if unsigned {
        if name != "sum" { return Ok(None); }
    } else if input.dtype() != WorkspaceDtype::Float32 ||
        input.representation() != Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true)) {
        return Ok(None);
    }
    let Ok(elements) = usize::try_from(input.shape()[0]) else { return Ok(None); };
    if elements == 0 { return Ok(None); }
    let mut population = CpuPopulation::default();
    if elements > 1 {
        let source = match (name, unsigned) {
            ("sum", true) => OperationEvent::cpu_flat_u32_sum_layout(elements, false),
            ("sum", false) => OperationEvent::cpu_row_sum_layout(1, elements, 1, false),
            ("min", false) => OperationEvent::cpu_row_min_layout(1, elements, 1, false),
            ("max", false) => OperationEvent::cpu_flat_max_layout(elements, false),
            _ => None,
        };
        let Some(source) = source else { return Ok(None); };
        if source.backing_births() != 1 || population.copy(source, 1).is_none() { return Ok(None); }
    }
    // The ordinary no-keepdims API always squeezes [1] to []. At width one
    // the Reduce itself is elided, and that squeeze aliases the actual input.
    let Some(squeeze) = OperationEvent::cpu_squeeze_layout(1, false) else { return Ok(None); };
    if squeeze.backing_births() != 0 || population.copy(squeeze, 1).is_none() { return Ok(None); }
    let frames = [size_of::<WorkspaceOperationView<'_>>(), size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<[WorkspaceLayoutView<'_>; 1]>() * 2,
        size_of::<Option<[WorkspaceLayoutView<'_>; 1]>>() * 2,
        size_of::<MlxCpuWorkspaceMechanisms>(), size_of::<&str>(), size_of::<bool>(), size_of::<usize>(),
        size_of::<(&str, bool)>(),
        size_of::<WorkspaceRepresentation>(), size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<CpuPopulation>(), size_of::<CpuCopyEvalLayout>() * 2,
        size_of::<Option<CpuCopyEvalLayout>>() * 2, size_of::<Option<()>>(),
        size_of::<Result<usize, std::num::TryFromIntError>>(), size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(), size_of::<u64>()];
    population.controls = frames.into_iter().try_fold(
        population.controls.checked_add(size_of_val(&frames)).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        |n, b| n.checked_add(b).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
    Ok(Some(OperationPlan { dtype: WorkspaceFloatingType::Float32, population,
        alias_input: (elements == 1).then_some(0),
        output_bytes: if elements == 1 { 0 } else { mechanism.allocation.fixed_buffer_capacity(4)? },
        scratch_bytes: 0, rank: 1, parameter_shells: 0, seeds: 0, validations: 0 }))
}

#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod tests {
    use super::*;
    #[test]
    fn cpu_flat_capture_reductions_preserve_scalar_and_singleton_storage() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected = MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        for elements in [1, 19] {
            for (name, dtype) in [("sum", WorkspaceDtype::Uint32), ("sum", WorkspaceDtype::Float32),
                ("min", WorkspaceDtype::Float32), ("max", WorkspaceDtype::Float32)] {
                let context = WorkspaceContext::new(cpu);
                let storage = WorkspaceExistingStorage::try_new(Some(4096), &context).unwrap();
                let layout = context.layout(&[elements], dtype).unwrap().with_representation(
                    (dtype == WorkspaceDtype::Float32).then_some(
                        WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true)));
                let input = WorkspaceTensor::existing_with_storage(layout, &storage, &context).unwrap();
                context.begin_state_span([&input]).unwrap();
                let output = context.execute(WorkspaceOperationKind::Reduction(name, 0, false), &[&input],
                    vec![context.layout(&[], dtype).unwrap()]).unwrap().remove(0);
                let report = context.finish_report(std::slice::from_ref(&output)).unwrap();
                assert!(report.unpriced_operations.is_empty() && report.unpriced_host_operations.is_empty());
                let recipe = SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context).unwrap();
                assert_eq!(recipe.completion.nested_completions, 0);
                assert_eq!(recipe.kernels, 0);
                assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(if elements == 1 { 0 } else { 4096 }));
                if elements == 1 { assert_eq!(report.state.as_ref().unwrap().retained_bytes, Some(4096)); }
                assert_eq!(output.layout().dtype(), dtype);
            }
        }
    }
}
