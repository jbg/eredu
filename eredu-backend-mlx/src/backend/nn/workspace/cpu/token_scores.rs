//! Exact existing rank-one argmax and singleton overwrite in token scoring.
use super::*;

pub(super) fn inspect(operation: WorkspaceOperationView<'_>, mechanism: MlxCpuWorkspaceMechanisms)
    -> facts::FactResult<Option<OperationPlan>> {
    let update = match operation.kind {
        WorkspaceOperationKindView::Sampling(WorkspaceSamplingOperation::Greedy) => false,
        WorkspaceOperationKindView::SliceUpdate { .. } => true,
        _ => return Ok(None),
    };
    if operation.inputs.len() != if update { 2 } else { 1 } || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor("CPU selected-token operand population differs"));
    }
    let input = operation.inputs.get(0).expect("checked token-score input");
    let output = operation.outputs.get(0).expect("checked token-score output");
    if input.shape().len() != 1 || input.shape()[0] < 2 { return Ok(None); }
    let expected = Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true));
    if input.dtype() != WorkspaceDtype::Float32 || input.representation() != expected { return Ok(None); }
    let elements = usize::try_from(input.shape()[0])?;
    let mut population = CpuPopulation::default();
    if update {
        let WorkspaceOperationKindView::SliceUpdate { starts } = operation.kind else { unreachable!() };
        let value = operation.inputs.get(1).expect("checked token-score update");
        if value.shape() != [1] || value.dtype() != WorkspaceDtype::Float32 || starts.len() != 1
            || starts[0] < 0 || starts[0] >= input.shape()[0] || output.shape() != input.shape()
            || output.dtype() != input.dtype() {
            return Err(MlxWorkspaceFactError::descriptor("CPU scalar overwrite changes coordinates or dtype"));
        }
        if value.representation() != expected { return Ok(None); }
        // Trace already includes the actual scalar->[1] reshape. Same-dtype
        // cast and same-shape broadcast elide; this is the remaining primitive.
        let Some(source) = OperationEvent::cpu_scalar_update_layout(elements, false) else { return Ok(None); };
        if source.backing_births() != 1 || population.copy(source, 2).is_none() { return Ok(None); }
    } else {
        if !output.shape().is_empty() || output.dtype() != WorkspaceDtype::Uint32 {
            return Err(MlxWorkspaceFactError::descriptor("CPU selected-token argmax changes scalar output"));
        }
        // Global argmax flattens an already rank-one row without an operation,
        // performs ArgReduce and squeezes [1]. Width one never asks for an
        // alternative, so its distinct zero-fill producer is not substituted.
        let Some(reduce) = OperationEvent::cpu_arg_reduce_layout(1, elements, 1, false) else { return Ok(None); };
        let Some(squeeze) = OperationEvent::cpu_squeeze_layout(1, false) else { return Ok(None); };
        if reduce.backing_births() != 1 || squeeze.backing_births() != 0
            || population.copy(reduce, 1).is_none() || population.copy(squeeze, 1).is_none() { return Ok(None); }
    }
    let frames = [size_of::<WorkspaceOperationView<'_>>(), size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<WorkspaceLayoutView<'_>>() * 3, size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(), size_of::<CpuPopulation>(),
        size_of::<CpuCopyEvalLayout>() * 2, size_of::<Option<CpuCopyEvalLayout>>() * 2,
        size_of::<Option<()>>(), size_of::<usize>(), size_of::<bool>(), size_of::<&[i32]>(),
        size_of::<Result<usize, std::num::TryFromIntError>>(), size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<(&safemlx::Array, &safemlx::Stream)>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>() * 2];
    population.controls = frames.into_iter().try_fold(
        population.controls.checked_add(size_of_val(&frames)).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        |n, b| n.checked_add(b).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
    Ok(Some(OperationPlan { dtype: WorkspaceFloatingType::Float32, population, alias_input: None,
        output_bytes: mechanism.allocation.fixed_buffer_capacity(output.bytes()?)?, scratch_bytes: 0,
        rank: 1, parameter_shells: 0, seeds: 0, validations: 0 }))
}

#[cfg(all(test, target_vendor="apple", feature="metal", not(feature="cuda")))]
mod tests {
    use super::*;
    use crate::backend::array_copy::TokenScoreProgram;
    #[test]
    fn cpu_token_score_trace_joins_full_distribution_and_each_scalar_frontier() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected = MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        for width in [1, 19, 1025] {
            let context = WorkspaceContext::new(cpu);
            let storage = WorkspaceExistingStorage::try_new(Some(16384), &context).unwrap();
            let input = WorkspaceTensor::existing_with_storage(context.layout(&[1, 2, width], WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true))),
                &storage, &context).unwrap();
            context.begin_state_span([&input]).unwrap();
            let ids = if width == 1 { vec![0] } else { vec![width as u32 - 1, 0] };
            let program = TokenScoreProgram::new(width, &ids).unwrap();
            let population = program.population().unwrap();
            let mut retained = context.metadata_vec(population.retained_outputs).unwrap();
            program.trace(&input, &context, &mut retained).unwrap();
            assert_eq!(retained.len(), population.retained_outputs);
            let chunks = (width as usize).div_ceil(1024);
            assert_eq!(population.scalar_completions, 2 + chunks + ids.len() * if width == 1 { 2 } else { 4 });
            let mut roots = context.metadata_vec(retained.len() + 1).unwrap();
            roots.push(input.clone()); roots.extend(retained);
            let report = context.finish_report(&roots).unwrap();
            assert!(report.unpriced_operations.is_empty(), "{:?}", report.unpriced_operations);
            assert!(report.unpriced_host_operations.is_empty());
            let recipe = SpeculativeNumericalRecipe::inspect_cpu_capture(&report, roots.len(), population.scalar_completions,
                ordinary, cpu, &context).unwrap();
            assert_eq!(recipe.kernels, 0);
            assert_eq!(recipe.completion.nested_completions, population.scalar_completions);
            // The shared reducer retains its final output plus the supplied capture frontier.
            assert_eq!(recipe.completion.traversal.limits().roots, 1 + roots.len());
            // Closing roots retain the opening source and every newly allocated
            // result/scalar backing; aliases do not contribute a second birth.
            assert_eq!(report.state.as_ref().unwrap().retained_bytes,
                report.tensor_buffers.retained_bytes.map(|bytes| bytes + 16384));
            assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(0));
        }
        let context = WorkspaceContext::new(cpu);
        let input = WorkspaceTensor::existing(context.layout(&[19], WorkspaceDtype::Float32).unwrap(), &context).unwrap();
        let output = context.execute(WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::Greedy), &[&input],
            vec![context.layout(&[], WorkspaceDtype::Uint32).unwrap()]).unwrap().remove(0);
        let report = context.finish_report(&[output]).unwrap();
        assert!(!report.unpriced_operations.is_empty());
    }
}
