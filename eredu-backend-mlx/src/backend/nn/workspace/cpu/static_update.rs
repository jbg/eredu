//! Actual full alias or partial copy through the ordinary SliceUpdate worker.
use super::*;

pub(super) fn inspect(operation: WorkspaceOperationView<'_>, mechanism:MlxCpuWorkspaceMechanisms)
    -> facts::FactResult<Option<OperationPlan>> {
    let WorkspaceOperationKindView::StaticSliceUpdate { starts, ends, strides } = operation.kind else {
        return Ok(None);
    };
    let Some([source, update]) = operation.inputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor("CPU static update operand population differs"));
    };
    let Some([output]) = operation.outputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor("CPU static update output population differs"));
    };
    let rank = source.shape().len();
    if starts.len() != rank || ends.len() != rank || strides.len() != rank
        || update.shape().len() != rank || output.shape() != source.shape()
        || update.dtype() != source.dtype() || output.dtype() != source.dtype()
        || (0..rank).any(|axis| {
            let (a, b, step, n, width) = (starts[axis], ends[axis], strides[axis], source.shape()[axis], update.shape()[axis]);
            a < 0 || b <= a || b > n || step <= 0
                || b.checked_sub(a).and_then(|d| d.checked_add(step - 1)).map(|d| d / step) != Some(width)
        }) {
        return Err(MlxWorkspaceFactError::descriptor("CPU static update coordinates or representation differ"));
    }
    if !(1..=4).contains(&rank) || source.dtype() != WorkspaceDtype::Float32
        || source.elements()? > i32::MAX as u64 || strides.iter().any(|&n| n != 1) { return Ok(None); }
    if [source, update].iter().any(|value| value.representation()
        .is_none_or(|r| r.dtype() != WorkspaceFloatingType::Float32)) { return Ok(None); }
    let whole=starts.iter().all(|&n|n==0) && ends==source.shape() && update.shape()==source.shape();
    let mut population=CpuPopulation::default();
    if !whole {
        if [source,update].iter().any(|value| !value.representation().expect("checked precision").row_contiguous()) {
            return Ok(None);
        }
        let Some(native)=OperationEvent::cpu_static_update_layout(rank,usize::try_from(source.elements()?)?,
            usize::try_from(update.elements()?)?,false) else {return Ok(None);};
        if native.backing_births()!=1 || population.copy(native,2).is_none() {return Ok(None);}
    }
    // Equal precision/shape make native astype and broadcast identities. Full
    // replacement is one result handle; a partial rectangle has one complete
    // destination copy and one shaped overwrite inside the same primitive.
    let frames = [
        safemlx::Array::static_slice_update_control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        size_of::<MlxCpuWorkspaceMechanisms>(),size_of::<bool>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<Result<usize,std::num::TryFromIntError>>(),size_of::<u64>()*2,
        size_of::<WorkspaceOperationView<'_>>(), size_of::<WorkspaceLayoutView<'_>>() * 3,
        size_of::<Option<[WorkspaceLayoutView<'_>; 2]>>(), size_of::<[WorkspaceLayoutView<'_>; 2]>(),
        size_of::<Option<[WorkspaceLayoutView<'_>; 1]>>(), size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(), size_of::<std::ops::Range<usize>>(),
        size_of::<std::slice::Iter<'_, i32>>(), size_of::<std::slice::Iter<'_, WorkspaceLayoutView<'_>>>(),
        size_of::<usize>(), size_of::<[i32; 5]>(), size_of::<Option<i32>>(),
        size_of::<CpuPopulation>(), size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
    ];
    population.controls=frames.into_iter().try_fold(population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,usize::checked_add)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population, alias_input:whole.then_some(1),
        output_bytes:if whole{0}else{mechanism.allocation.fixed_buffer_capacity(output.bytes()?)?},
        scratch_bytes:0,rank,parameter_shells:usize::from(whole),seeds:0,validations:0,
    }))
}

#[cfg(all(test, target_vendor="apple", feature="metal", not(feature="cuda")))]
mod tests {
    use super::*;
    use eredu_nn::Tensor;
    use crate::composition::mlx::fixture_intervention::PreparedStaticActivation;
    use eredu_core::{capture::ResolvedCaptureSlice, intervention::{InterventionAction, InterventionDtype}};
    #[test]
    fn cpu_static_scale_preserves_full_alias_and_prices_partial_replacement() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected = MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        for width in [1, 19, 1025] {
            let context = WorkspaceContext::new(cpu);
            let storage = WorkspaceExistingStorage::try_new(Some(16384), &context).unwrap();
            let source = WorkspaceTensor::existing_with_storage(context.layout(&[1,1,width], WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, true))),
                &storage, &context).unwrap();
            let shape = [1,1,width as u64];
            let slice = ResolvedCaptureSlice { starts:vec![0;3], ends:shape.to_vec(), strides:vec![1;3], shape:shape.to_vec() };
            let action = InterventionAction::Scale { dtype:InterventionDtype::Float32, factor:-0.5 };
            let program = PreparedStaticActivation::new(&action, &slice, &shape, InterventionDtype::Float32).unwrap();
            let population = program.population().unwrap();
            assert_eq!(population.retained_roots,5); assert_eq!(population.completions,0);
            let mut retained = context.metadata_vec(population.retained_roots).unwrap();
            context.begin_state_span([&source]).unwrap();
            let output = program.trace(&source, &context, &mut retained).unwrap();
            assert_eq!(retained.len(),5); assert_eq!(output.shape(),source.shape());
            let mut roots = context.metadata_vec(retained.len()+1).unwrap(); roots.push(source.clone()); roots.extend(retained);
            let report = context.finish_report(&roots).unwrap();
            assert!(report.unpriced_operations.is_empty(),"{:?}",report.unpriced_operations);
            assert!(report.unpriced_host_operations.is_empty());
            let first = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(first.alias_input,Some(0)); assert_eq!(first.population.primitives,0); assert_eq!(first.parameter_shells,1);
            let last = cpu.plan(report.operations.last().unwrap().as_view()).unwrap().unwrap();
            assert_eq!(last.alias_input,Some(1)); assert_eq!(last.population.primitives,0);
            assert_eq!(last.population.births,0); assert_eq!(last.output_bytes,0); assert_eq!(last.scratch_bytes,0);
            assert_eq!(last.parameter_shells,1);
            let recipe = SpeculativeNumericalRecipe::inspect_cpu_capture(&report, roots.len(), 0, ordinary,cpu,&context).unwrap();
            assert_eq!(recipe.kernels,0); assert_eq!(recipe.completion.nested_completions,0);
            // Closing roots retain the opening source and newly allocated results.
            assert_eq!(report.state.as_ref().unwrap().retained_bytes,
                report.tensor_buffers.retained_bytes.map(|bytes| bytes + 16384));
            assert_eq!(report.state.as_ref().unwrap().displaced_bytes,Some(0));
        }
        let base = WorkspaceLayoutView::new(&[1,1,19],WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)));
        let inputs = [base,base]; let outputs = [base];
        let op = WorkspaceOperationView { kind:WorkspaceOperationKindView::StaticSliceUpdate {
            starts:&[0,0,0], ends:&[1,1,19], strides:&[1,1,1]},
            inputs:WorkspaceLayoutList::Views(&inputs), outputs:WorkspaceLayoutList::Views(&outputs) };
        for bad in [None,Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Bfloat16,true))] {
            for index in 0..2 { let mut changed=inputs; changed[index]=base.with_representation(bad);
                assert!(cpu.plan(WorkspaceOperationView {inputs:WorkspaceLayoutList::Views(&changed),..op}).unwrap().is_none()); }
        }
        let partial = [base,WorkspaceLayoutView::new(&[1,1,2],WorkspaceDtype::Float32).unwrap()
            .with_representation(base.representation())];
        let partial=cpu.plan(WorkspaceOperationView {kind:WorkspaceOperationKindView::StaticSliceUpdate {
            starts:&[0,0,1],ends:&[1,1,3],strides:&[1,1,1]},inputs:WorkspaceLayoutList::Views(&partial),..op}).unwrap().unwrap();
        assert_eq!(partial.alias_input,None);assert_eq!(partial.population.primitives,1);
        assert_eq!(partial.population.births,1);assert_eq!(partial.population.input_edges,2);
        assert_eq!(partial.parameter_shells,0);assert!(partial.output_bytes>=19*4);assert_eq!(partial.scratch_bytes,0);
        assert!(cpu.plan(WorkspaceOperationView {kind:WorkspaceOperationKindView::StaticSliceUpdate {
            starts:&[0,0],ends:&[1,1,19],strides:&[1,1,1]},..op}).is_err());
        let strided = [base,base.with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false)))];
        assert_eq!(cpu.output_representation(WorkspaceOperationView {inputs:WorkspaceLayoutList::Views(&strided),..op},0),strided[1].representation());
    }
    #[test]
    fn cpu_partial_static_update_quotes_batched_rectangles_and_complete_scale_trace() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        let context=WorkspaceContext::new(cpu);
        let original=WorkspaceExistingStorage::try_new(Some(168),&context).unwrap();
        let replacement=WorkspaceExistingStorage::try_new(Some(48),&context).unwrap();
        let known=Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true));
        let source=WorkspaceTensor::existing_with_storage(context.layout(&[2,3,7],WorkspaceDtype::Float32).unwrap()
            .with_representation(known),&original,&context).unwrap();
        let update=WorkspaceTensor::existing_with_storage(context.layout(&[2,2,3],WorkspaceDtype::Float32).unwrap()
            .with_representation(known),&replacement,&context).unwrap();
        context.begin_state_span([&source,&update]).unwrap();
        let output=source.update_static_slice(&update,&[0,1,2],&[2,3,5],&[1;3],&context).unwrap();
        assert_eq!(output.shape(),[2,3,7]);assert_eq!(output.layout().representation(),known);
        let report=context.finish_report(&[source.clone(),update.clone(),output]).unwrap();
        assert!(report.unpriced_operations.is_empty()&&report.unpriced_host_operations.is_empty());
        let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
        assert_eq!(plan.alias_input,None);assert_eq!(plan.population.primitives,1);
        assert_eq!(plan.population.input_edges,2);assert_eq!(plan.population.births,1);
        assert_eq!(plan.output_bytes,ordinary.allocation().fixed_buffer_capacity(168).unwrap());
        assert_eq!(report.state.as_ref().unwrap().retained_bytes,Some(216+plan.output_bytes));
        assert_eq!(report.state.as_ref().unwrap().displaced_bytes,Some(0));
        let recipe=SpeculativeNumericalRecipe::inspect_cpu_capture(&report,3,0,ordinary,cpu,&context).unwrap();
        assert_eq!(recipe.kernels,0);assert_eq!(recipe.completion.nested_completions,0);
        for bad in [None,Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Bfloat16,true)),
            Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false))] {
            let op=report.operations[0].as_view();
            let values=[op.inputs.get(0).unwrap(),op.inputs.get(1).unwrap().with_representation(bad)];
            assert!(cpu.plan(WorkspaceOperationView{inputs:WorkspaceLayoutList::Views(&values),..op}).unwrap().is_none());
        }
        for (dims,starts,ends,selected_shape) in [
            ([1,1,19],[0,0,1],[1,1,3],[1,1,2]),
            ([2,3,7],[0,1,2],[2,3,5],[2,2,3]),
        ] {
        let context=WorkspaceContext::new(cpu);
        let input_bytes=dims.iter().map(|&n|n as u64).product::<u64>()*4;
        let storage=WorkspaceExistingStorage::try_new(Some(input_bytes),&context).unwrap();
        let source=WorkspaceTensor::existing_with_storage(context.layout(&dims,WorkspaceDtype::Float32).unwrap()
            .with_representation(known),&storage,&context).unwrap();
        context.begin_state_span([&source]).unwrap();
        let shape=dims.map(|n|n as u64);
        let slice=ResolvedCaptureSlice{starts:starts.to_vec(),ends:ends.to_vec(),strides:vec![1;3],shape:selected_shape.to_vec()};
        let action=InterventionAction::Scale{dtype:InterventionDtype::Float32,factor:-0.5};
        let program=PreparedStaticActivation::new(&action,&slice,&shape,InterventionDtype::Float32).unwrap();
        let population=program.population().unwrap();
        let mut retained=context.metadata_vec(population.retained_roots).unwrap();
        let output=program.trace(&source,&context,&mut retained).unwrap();
        assert_eq!(output.shape(),source.shape());assert_eq!(retained.len(),5);
        let mut roots=context.metadata_vec(retained.len()+1).unwrap();roots.push(source);roots.extend(retained);
        let report=context.finish_report(&roots).unwrap();
        assert!(report.unpriced_operations.is_empty(),"{:?}",report.unpriced_operations);
        assert!(report.unpriced_host_operations.is_empty());
        let update=cpu.plan(report.operations.last().unwrap().as_view()).unwrap().unwrap();
        assert_eq!(update.population.births,1);assert_eq!(update.alias_input,None);
        assert_eq!(report.state.as_ref().unwrap().retained_bytes,
            report.tensor_buffers.retained_bytes.map(|bytes|bytes+input_bytes));
        let recipe=SpeculativeNumericalRecipe::inspect_cpu_capture(&report,roots.len(),0,ordinary,cpu,&context).unwrap();
        assert_eq!(recipe.kernels,0);assert_eq!(recipe.completion.nested_completions,0);
        }
    }

}
