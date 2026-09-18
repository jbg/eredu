//! Exact CPU children of the ordinary terminal-row sort/take composite.
use super::*;
use safemlx::Dtype;

#[derive(Default)]
struct Program { population: CpuPopulation, bytes: u64, seeds: usize, shells: usize }
impl Program {
    fn child(&mut self, child: OperationPlan) -> Option<()> {
        self.population.add(child.population)?;
        self.bytes=self.bytes.checked_add(child.output_bytes)?.checked_add(child.scratch_bytes)?;
        self.seeds=self.seeds.checked_add(child.seeds)?;
        self.shells=self.shells.checked_add(child.parameter_shells)?;
        Some(())
    }
    fn copy(&mut self, source: CpuCopyEvalLayout, inputs: usize, bytes: u64) -> Option<()> {
        self.bytes=self.bytes.checked_add(bytes.checked_mul(u64::try_from(source.backing_births()).ok()?)?)?;
        self.population.copy(source,inputs)
    }
}
pub(super) fn inspect(operation: WorkspaceOperationView<'_>, mechanism: MlxCpuWorkspaceMechanisms)
    -> facts::FactResult<Option<OperationPlan>> {
    let Some(geometry)=super::super::candidates::geometry(operation)? else {return Ok(None)};
    let input=operation.inputs.get(0).expect("validated candidate input");
    let expected=Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true));
    if input.representation()!=expected || input.elements()?>i32::MAX as u64 {return Ok(None);}
    let width=geometry.vocabulary as usize;let count=geometry.count as usize;
    let shape=[geometry.vocabulary as i32];
    let row=WorkspaceLayoutView::new(&shape,WorkspaceDtype::Float32)?.with_representation(expected);
    let mask=WorkspaceLayoutView::new(&shape,WorkspaceDtype::Bool)?;
    let unsigned=WorkspaceLayoutView::new(&shape,WorkspaceDtype::Uint32)?;
    let scalar=WorkspaceLayoutView::new(&[],WorkspaceDtype::Uint32)?;
    // Reuse the exact classification and flat-count plans. These are borrowed
    // operation descriptors, not another execution or reduction engine.
    let child = |kind, input, output| {
        let inputs=[input];let outputs=[output];
        let op=WorkspaceOperationView {kind,inputs:WorkspaceLayoutList::Views(&inputs),
            outputs:WorkspaceLayoutList::Views(&outputs)};
        match kind {
            WorkspaceOperationKindView::Reduction(..)=>flat_reduction::inspect(op,mechanism),
            _=>classification::inspect(op,mechanism),
        }
    };
    let finite=child(WorkspaceOperationKindView::Elementwise("is_finite"),row,mask)?;
    let cast=child(WorkspaceOperationKindView::Elementwise("bool_to_u32"),mask,unsigned)?;
    let sum=child(WorkspaceOperationKindView::Reduction("sum",0,false),unsigned,scalar)?;
    let (Some(finite),Some(cast),Some(sum))=(finite,cast,sum) else{return Ok(None)};
    let sorted_bytes=mechanism.allocation.fixed_buffer_capacity(u64::from(geometry.vocabulary)*4)?;
    let selected_bytes=mechanism.allocation.fixed_buffer_capacity(u64::from(geometry.count)*4)?;
    let mut p=Program::default();
    let source=(|| {
        // Batch/terminal row is a unit-step rectangle followed by reshape.
        // The entire input rectangle elides Slice when there is only one row.
        if geometry.rows>1 {p.copy(OperationEvent::cpu_slice_layout(3, false, false)?,1,0)?;}
        p.copy(OperationEvent::cpu_reshape_alias_layout(3,1,false)?,1,0)?;
        // F32 AsType is identity, so no conversion backing/task is invented.
        p.child(finite)?;p.child(cast)?;p.child(sum)?;
        p.copy(OperationEvent::cpu_argsort_layout(Dtype::Float32,1,width,1,false)?,1,sorted_bytes)?;
        if count<width {p.copy(OperationEvent::cpu_slice_layout(1, false, false)?,1,0)?;}
        // Contiguous may keep the full sort buffer or copy K entries, based
        // on that actual backing's byte size. Keep both physical envelopes.
        p.copy(OperationEvent::cpu_contiguous_layout(1,false)?,1,selected_bytes)?;
        // One U32 index is already broadcast and typed. take's flatten is
        // identity; Gather owns [K,1], then Squeeze aliases that score backing.
        p.copy(OperationEvent::cpu_gather_layout(Dtype::Float32,Dtype::Uint32,1,1,
            width,count,1,false)?,2,selected_bytes)?;
        p.copy(OperationEvent::cpu_squeeze_layout(2,false)?,1,0)?;
        Some(())
    })();
    if source.is_none(){return Ok(None);}
    let output_bytes=selected_bytes.checked_mul(2).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let scratch_bytes=p.bytes.checked_sub(output_bytes).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let frames=[size_of::<WorkspaceOperationView<'_>>(),size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<super::super::candidates::Geometry>(),size_of::<Option<super::super::candidates::Geometry>>(),
        size_of::<WorkspaceLayoutView<'_>>()*5,size_of::<[WorkspaceLayoutView<'_>;1]>()*2,
        size_of::<WorkspaceRepresentation>(),size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<[i32;1]>(),size_of::<Program>(),size_of::<CpuPopulation>(),
        size_of::<OperationPlan>()*4,size_of::<Option<OperationPlan>>()*3,
        size_of::<(Option<OperationPlan>,Option<OperationPlan>,Option<OperationPlan>)>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),size_of::<Option<()>>(),size_of::<usize>()*2,
        size_of::<u64>()*4,size_of_val(&child),size_of::<(&mut Program,OperationPlan)>(),
        size_of::<(&mut Program,CpuCopyEvalLayout,usize,u64)>(),
        size_of::<(WorkspaceOperationKindView<'_>,WorkspaceLayoutView<'_>,WorkspaceLayoutView<'_>)>(),
        size_of::<&mut Program>(),size_of::<WorkspaceLayoutList<'_>>(),
        size_of_val(&operation.outputs.iter())];
    p.population.controls=frames.into_iter().try_fold(p.population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,|n,b|n.checked_add(b).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
    Ok(Some(OperationPlan {dtype:WorkspaceFloatingType::Float32,population:p.population,
        alias_input:None,output_bytes,scratch_bytes,rank:3,
        parameter_shells:p.shells.checked_add(crate::backend::array_copy::CandidateExtraction::ROOTS)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,seeds:p.seeds,validations:0}))
}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod tests {
    use super::*;
    use crate::backend::array_copy::CandidateExtraction;
    #[test]
    fn cpu_candidate_composite_preserves_two_outputs_full_sort_and_three_reads() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),choice);
        for (width,count,rows) in [(1,1,1),(19,5,2),(4097,3,1),(4097,4097,1)] {
            let context=WorkspaceContext::new(cpu);
            let storage=WorkspaceExistingStorage::try_new(Some(65536),&context).unwrap();
            let input=WorkspaceTensor::existing_with_storage(context.layout(&[1,rows,width],WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&storage,&context).unwrap();
            context.begin_state_span([&input]).unwrap();
            let program=CandidateExtraction::borrowed(&[1,rows,width],count as u64).unwrap();
            let output=program.trace(&input,&context).unwrap();assert_eq!(output.len(),2);
            assert_eq!(output[0].layout().dtype(),WorkspaceDtype::Uint32);
            assert_eq!(output[0].layout().representation(),None);
            assert_eq!(output[1].layout().representation(),Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)));
            let report=context.finish_report(&[input.clone(),output[0].clone(),output[1].clone()]).unwrap();
            assert!(report.unpriced_operations.is_empty()&&report.unpriced_host_operations.is_empty());
            let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(plan.output_bytes,2*ordinary.allocation().fixed_buffer_capacity(count as u64*4).unwrap());
            assert!(plan.scratch_bytes>=ordinary.allocation().fixed_buffer_capacity(width as u64*4).unwrap());
            assert_eq!(plan.seeds,2);assert_eq!(plan.validations,0);
            let recipe=SpeculativeNumericalRecipe::inspect_cpu_capture(&report,1+CandidateExtraction::ROOTS,
                CandidateExtraction::COMPLETIONS,ordinary,cpu,&context).unwrap();
            assert_eq!(recipe.kernels,0);assert_eq!(recipe.completion.nested_completions,3);
            // One final output plus source and the independent candidate roots.
            assert_eq!(recipe.completion.traversal.limits().roots,2+CandidateExtraction::ROOTS);
            // Closing roots retain the opening source and every newly allocated
            // result/scalar backing; aliases do not contribute a second birth.
            assert_eq!(report.state.as_ref().unwrap().retained_bytes,
                report.tensor_buffers.retained_bytes.map(|bytes| bytes + 65536));
            assert_eq!(report.state.as_ref().unwrap().displaced_bytes,Some(0));
        }
        let context=WorkspaceContext::new(cpu);
        let input=WorkspaceTensor::existing(context.layout(&[1,1,19],WorkspaceDtype::Float32).unwrap(),&context).unwrap();
        let output=CandidateExtraction::borrowed(&[1,1,19],5).unwrap().trace(&input,&context).unwrap();
        let report=context.finish_report(&output).unwrap();assert!(!report.unpriced_operations.is_empty());
    }
}
