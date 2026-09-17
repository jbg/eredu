//! Exact eager F32 scalar and existing Bool/F32 where worker used by summaries.
use super::*;
use safemlx::Dtype;

pub(super) fn inspect(operation: WorkspaceOperationView<'_>, mechanism: MlxCpuWorkspaceMechanisms)
    -> facts::FactResult<Option<OperationPlan>> {
    let scalar = match operation.kind {
        WorkspaceOperationKindView::Elementwise("scalar_f32") => true,
        WorkspaceOperationKindView::Elementwise("where") => false,
        _ => return Ok(None),
    };
    let Some([output]) = operation.outputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor("CPU scalar/selection output population differs"));
    };
    let mut population = CpuPopulation::default();
    let rank = output.shape().len();
    if scalar {
        if !basic::is_scalar_f32(operation) {
            return Err(MlxWorkspaceFactError::descriptor("CPU eager scalar differs from one F32 value"));
        }
    } else {
        let Some([mask, selected, other]) = operation.inputs.array() else {
            return Err(MlxWorkspaceFactError::descriptor("CPU selection requires mask and two values"));
        };
        if output.dtype() != WorkspaceDtype::Float32 || selected.dtype() != output.dtype()
            || other.dtype() != output.dtype() || mask.dtype() != WorkspaceDtype::Bool
            || mask.shape() != output.shape() || selected.shape() != output.shape()
            || (!other.shape().is_empty() && other.shape() != output.shape()) {
            return Err(MlxWorkspaceFactError::descriptor("CPU selection changes its declared mask/value geometry"));
        }
        if rank > 4 || output.shape().iter().any(|&n| n <= 0)
            || !strided_capture::f32_source(selected) || !strided_capture::f32_source(other) {
            return Ok(None);
        }
        let elements = usize::try_from(output.elements()?)?;
        if elements > i32::MAX as usize { return Ok(None); }
        // AsType is identity for Bool/F32/F32. Only the actual scalar-to-row
        // broadcast creates an alias; matching shapes return their inputs.
        let broadcast = rank != 0 && other.shape().is_empty();
        if broadcast {
            let Some(alias) = OperationEvent::cpu_broadcast_alias_layout(0, rank, false) else { return Ok(None); };
            if alias.backing_births() != 0 || population.copy(alias, 1).is_none() { return Ok(None); }
        }
        // Any source-proved non-row layout takes the unchanged General worker.
        // Its collapse/iterator storage is present in the existing broadcast
        // source; logical output size remains separate from retained input span.
        let strided = !selected.representation().expect("qualified F32 source").row_contiguous()
            || !other.representation().expect("qualified F32 source").row_contiguous();
        let source = if strided || (broadcast && elements > 1) {
            OperationEvent::cpu_select_broadcast_layout(rank, elements, false)
        } else { OperationEvent::cpu_select_layout(Dtype::Float32, rank, elements, false) };
        let Some(source) = source else { return Ok(None); };
        if source.backing_births() != 1 || population.copy(source, 3).is_none() { return Ok(None); }
    }
    if scalar {
        population.controls = population.controls.checked_add(
            basic::scalar_f32_control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    }
    let frames = [strided_capture::control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?, size_of::<WorkspaceOperationView<'_>>(), size_of::<WorkspaceLayoutView<'_>>() * 4,
        size_of::<[WorkspaceLayoutView<'_>; 1]>(), size_of::<Option<[WorkspaceLayoutView<'_>; 1]>>(),
        size_of::<[WorkspaceLayoutView<'_>; 3]>(), size_of::<Option<[WorkspaceLayoutView<'_>; 3]>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(), size_of::<CpuPopulation>(), size_of::<bool>() * 3,
        size_of::<usize>() * 2, size_of::<u64>(), size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(), size_of::<CpuCopyEvalLayout>() * 2,
        size_of::<Option<CpuCopyEvalLayout>>() * 2, size_of::<Option<()>>(),
        size_of::<std::slice::Iter<'_, i32>>(), size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        // Where's borrowed condition/value/result transports. Native header,
        // data and task layouts are separately priced as source.
        size_of::<safemlx::Array>() * 4,
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>() * 2,
        size_of::<(&safemlx::Array, &safemlx::Array, &safemlx::Array, &safemlx::Stream)>(),
        size_of::<Result<usize, std::num::TryFromIntError>>()];
    population.controls = frames.into_iter().try_fold(
        population.controls.checked_add(size_of_val(&frames)).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        |n, b| n.checked_add(b).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
    Ok(Some(OperationPlan { dtype: WorkspaceFloatingType::Float32, population,
        alias_input: None, output_bytes: mechanism.allocation.fixed_buffer_capacity(output.bytes()?)?,
        scratch_bytes: 0, rank, parameter_shells: 0, seeds: usize::from(scalar), validations: 0 }))
}

#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod tests {
    use super::*;
    use crate::backend::array_copy::SummaryProgram;
    #[test]
    fn cpu_summary_trace_qualifies_actual_chunk_scalars_masks_and_reductions() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected = MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        for elements in [1, 19, 1025] {
            let context = WorkspaceContext::new(cpu);
            let storage = WorkspaceExistingStorage::try_new(Some(8192), &context).unwrap();
            let flat = WorkspaceTensor::existing_with_storage(context.layout(&[elements], WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true))),
                &storage, &context).unwrap();
            context.begin_state_span([&flat]).unwrap();
            let program = SummaryProgram::new(elements).unwrap();
            let population = program.population().unwrap();
            let mut retained = Vec::new();
            program.trace(&flat, &context, &mut retained).unwrap();
            assert_eq!(retained.len(), population.retained_outputs);
            let report = context.finish_report(std::slice::from_ref(&flat)).unwrap();
            assert!(report.unpriced_operations.is_empty() && report.unpriced_host_operations.is_empty(),
                "missing sources: {:?}", report.unpriced_operations);
            let recipe = SpeculativeNumericalRecipe::inspect_cpu_capture(&report, retained.len(),
                population.scalar_completions, ordinary, cpu, &context).unwrap();
            assert_eq!(recipe.kernels, 0);
            assert_eq!(recipe.completion.nested_completions, population.scalar_completions);
            assert_eq!(recipe.completion.traversal.limits().roots, retained.len() + 1);
            assert!(recipe.graph_capacity > 0 && recipe.record_capacity > 0);
            assert_eq!(report.state.as_ref().unwrap().retained_bytes, Some(8192));
            assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(0));
        }
    }
    #[test]
    fn scalar_source_is_exact_and_where_refuses_unrepresented_values() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected = MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        for context in [WorkspaceContext::new(cpu), WorkspaceContext::new(ordinary)] {
            let scalar = context.execute(WorkspaceOperationKind::Elementwise("scalar_f32"), &[],
                vec![context.layout(&[], WorkspaceDtype::Float32).unwrap()]).unwrap().remove(0);
            assert_eq!(scalar.layout().representation(), Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true)));
            let report = context.finish_report(std::slice::from_ref(&scalar)).unwrap();
            assert!(report.unpriced_operations.is_empty() && report.unpriced_host_operations.is_empty());
            assert_eq!(report.tensor_buffers.retained_bytes, report.tensor_buffers.total_bytes);
            assert!(context.execute(WorkspaceOperationKind::Elementwise("scalar_f32"), &[],
                vec![context.layout(&[1], WorkspaceDtype::Float32).unwrap()]).is_err());
        }
        let input = WorkspaceLayout::new(&[19], WorkspaceDtype::Float32).unwrap();
        let mask = WorkspaceLayout::new(&[19], WorkspaceDtype::Bool).unwrap();
        let other = WorkspaceLayout::new(&[], WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true)));
        let inputs = [mask.as_view(), input.as_view(), other.as_view()];
        let outputs = [input.as_view()];
        assert!(cpu.plan(WorkspaceOperationView { kind: WorkspaceOperationKindView::Elementwise("where"),
            inputs: WorkspaceLayoutList::Views(&inputs), outputs: WorkspaceLayoutList::Views(&outputs) }).unwrap().is_none());
    }
}
