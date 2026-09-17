//! Shared F32 comparisons and Bool predicates, preserving exact CPU workers.
use super::*;
use safemlx::{CpuUnaryOperation, Dtype};

pub(super) fn inspect(operation: WorkspaceOperationView<'_>, mechanism: MlxCpuWorkspaceMechanisms)
    -> facts::FactResult<Option<OperationPlan>> {
    let (binary, logical) = match operation.kind {
        WorkspaceOperationKindView::Elementwise("less") => (Some(CpuBinaryOperation::Less), false),
        WorkspaceOperationKindView::Elementwise("less_equal") => (Some(CpuBinaryOperation::LessEqual), false),
        WorkspaceOperationKindView::Elementwise("greater") => (Some(CpuBinaryOperation::Greater), false),
        WorkspaceOperationKindView::Elementwise("greater_equal") => (Some(CpuBinaryOperation::GreaterEqual), false),
        WorkspaceOperationKindView::Elementwise("logical_and") => (Some(CpuBinaryOperation::LogicalAnd), true),
        WorkspaceOperationKindView::Elementwise("logical_not") => (None, true),
        _ => return Ok(None),
    };
    let count = if binary.is_some() { 2 } else { 1 };
    if operation.inputs.len() != count || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor("CPU comparison operand population differs"));
    }
    let left = operation.inputs.get(0).expect("checked comparison input");
    let output = operation.outputs.get(0).expect("checked comparison output");
    let dtype = if logical { WorkspaceDtype::Bool } else { WorkspaceDtype::Float32 };
    if output.dtype() != WorkspaceDtype::Bool || output.shape() != left.shape() {
        return Err(MlxWorkspaceFactError::descriptor("CPU comparison changes output geometry or Bool dtype"));
    }
    let rank = output.shape().len();
    if rank > 4 || output.shape().iter().any(|&n| n <= 0) { return Ok(None); }
    for value in operation.inputs.iter() {
        if value.dtype() != dtype || (!value.shape().is_empty() && value.shape() != output.shape()) {
            return Err(MlxWorkspaceFactError::descriptor("CPU comparison changes its source dtype or geometry"));
        }
        if !logical && !strided_capture::f32_source(value) { return Ok(None); }
    }
    let elements = usize::try_from(output.elements()?)?;
    if elements > i32::MAX as usize { return Ok(None); }
    let native = if logical { Dtype::Bool } else { Dtype::Float32 };
    let mut population = CpuPopulation::default();
    // Bool operators' conversion and F32 comparison promotion are identities.
    // A differing scalar operand alone introduces a broadcast alias.
    for value in operation.inputs.iter() {
        if rank != 0 && value.shape().is_empty() {
            let Some(source) = OperationEvent::cpu_broadcast_alias_layout(0, rank, false) else { return Ok(None); };
            if source.backing_births() != 0 || population.copy(source, 1).is_none() { return Ok(None); }
        }
    }
    match binary {
        Some(kind) => {
            let Some(source) = OperationEvent::cpu_binary_layout(kind, native, rank, elements, false) else { return Ok(None); };
            if source.backing_births() != 1 || population.binary(source).is_none() { return Ok(None); }
        }
        None => {
            let Some(source) = OperationEvent::cpu_unary_layout(CpuUnaryOperation::LogicalNot, native, rank, false) else { return Ok(None); };
            if source.backing_births() != 1 || population.unary(source).is_none() { return Ok(None); }
        }
    }
    let frames = [strided_capture::control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?, size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<(Option<CpuBinaryOperation>, bool)>(),
        size_of_val(&operation.inputs.iter()) * 2,
        size_of::<WorkspaceOperationView<'_>>(), size_of::<WorkspaceLayoutView<'_>>() * 3,
        size_of::<Option<CpuBinaryOperation>>(), size_of::<bool>(), size_of::<usize>() * 3,
        size_of::<WorkspaceDtype>(), size_of::<Dtype>(), size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(), size_of::<CpuPopulation>(),
        size_of::<CpuCopyEvalLayout>(), size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(), size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(), size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<Option<()>>(), size_of::<Result<usize, std::num::TryFromIntError>>(),
        size_of::<WorkspaceLayoutList<'_>>(), size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<OperationPlan>(), size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<(&safemlx::Array, &safemlx::Array, &safemlx::Stream)>(),
        size_of::<safemlx::Array>() * 3,
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>()];
    population.controls = frames.into_iter().try_fold(
        population.controls.checked_add(size_of_val(&frames)).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        |n, b| n.checked_add(b).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
    Ok(Some(OperationPlan { dtype: WorkspaceFloatingType::Float32, population,
        alias_input: None, output_bytes: mechanism.allocation.fixed_buffer_capacity(output.bytes()?)?,
        scratch_bytes: 0, rank, parameter_shells: 0, seeds: 0, validations: 0 }))
}

#[cfg(all(test, target_vendor="apple", feature="metal", not(feature="cuda")))]
mod tests {
    use super::*;
    use crate::backend::array_copy::HistogramProgram;
    #[test]
    fn cpu_histogram_trace_preserves_edges_and_each_scalar_frontier() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected = MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        let edges = [-2.0, 0.0, 2.0, 4.0];
        for elements in [1, 19, 1025] {
            let context = WorkspaceContext::new(cpu);
            let storage = WorkspaceExistingStorage::try_new(Some(8192), &context).unwrap();
            let flat = WorkspaceTensor::existing_with_storage(context.layout(&[elements], WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true))),
                &storage, &context).unwrap();
            context.begin_state_span([&flat]).unwrap();
            let program = HistogramProgram::new(elements, &edges).unwrap();
            let population = program.population().unwrap();
            let mut retained = Vec::new();
            program.trace(&flat, &context, &mut retained).unwrap();
            assert_eq!(retained.len(), population.retained_outputs);
            let report = context.finish_report(std::slice::from_ref(&flat)).unwrap();
            assert!(report.unpriced_operations.is_empty() && report.unpriced_host_operations.is_empty(),
                "missing histogram sources: {:?}", report.unpriced_operations);
            let recipe = SpeculativeNumericalRecipe::inspect_cpu_capture(&report, retained.len(),
                population.scalar_completions, ordinary, cpu, &context).unwrap();
            assert_eq!(recipe.kernels, 0);
            assert_eq!(recipe.completion.nested_completions, population.scalar_completions);
            assert_eq!(recipe.completion.traversal.limits().roots, retained.len()+1);
            assert_eq!(population.scalar_completions, ((elements as usize + 1023)/1024)*6);
            assert_eq!(report.state.as_ref().unwrap().retained_bytes, Some(8192));
            assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(0));
        }
    }
}
