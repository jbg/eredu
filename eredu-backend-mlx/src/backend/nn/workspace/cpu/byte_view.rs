//! Exact byte reinterpretation using the ordinary CPU View alias/copy worker.
use super::*;
use super::super::byte_view as geometry;
pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    let WorkspaceOperationKindView::View(name)=operation.kind else{return Ok(None)};
    let Some(target)=geometry::selected(name) else{return Ok(None)};
    let Some((bytes,rank))=geometry::output_geometry(operation) else {
        return Err(MlxWorkspaceFactError::descriptor("CPU byte view scalar or byte geometry differs"));
    };
    let input=operation.inputs.get(0).expect("checked byte source");
    let output=operation.outputs.get(0).expect("checked byte result");
    if rank>5||bytes==0||input.shape().iter().chain(output.shape()).any(|&n|n<=0) {return Ok(None)};
    let source=geometry::Dtype::from_layout(input).expect("checked source dtype");
    let same=source==target;
    let alias=source.bytes()==target.bytes()||input.representation().is_some_and(|r|
        r.row_contiguous()||(target.bytes()<source.bytes()&&r.last_axis_contiguous()));
    let copy=!alias;
    let mut population=CpuPopulation::default();
    if !same {
        let Some(native)=OperationEvent::cpu_byte_view_layout(source.native(),target.native(),rank,
            usize::try_from(bytes)?,copy,false) else{return Ok(None)};
        if native.backing_births()!=usize::from(copy)||population.copy(native,1).is_none(){return Ok(None)};
    }
    let frames=[size_of::<WorkspaceOperationView<'_>>(),size_of::<WorkspaceLayoutView<'_>>()*2,
        size_of::<MlxCpuWorkspaceMechanisms>(),size_of::<geometry::Dtype>()*2,
        size_of::<Option<geometry::Dtype>>(),size_of::<Option<(u64,usize)>>(),
        size_of::<CpuPopulation>(),size_of::<OperationPlan>(),size_of::<Option<OperationPlan>>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<Option<WorkspaceRepresentation>>(),size_of::<WorkspaceRepresentation>(),
        size_of::<bool>()*4,size_of::<usize>()*2,size_of::<u64>(),size_of::<&str>(),
        size_of::<std::iter::Chain<std::slice::Iter<'_,i32>,std::slice::Iter<'_,i32>>>(),
        size_of::<(&safemlx::Array,safemlx::Dtype,&safemlx::Stream)>(),
        size_of::<Result<safemlx::Array,safemlx::error::Exception>>()];
    population.controls=frames.into_iter().try_fold(population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,usize::checked_add)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    // Native source geometry above stays in actual scalar bytes. A possible
    // new floating destination also satisfies the workspace's conservative
    // Float32 envelope (four bytes per element, including F16/BF16 results).
    Ok(Some(OperationPlan{dtype:target.floating().unwrap_or(WorkspaceFloatingType::Float32),population,
        alias_input:alias.then_some(0),output_bytes:if copy {mechanism.allocation.fixed_buffer_capacity(bytes.max(output.bytes()?))?}else{0},
        scratch_bytes:0,rank,parameter_shells:usize::from(same),seeds:0,validations:0}))
}
pub(super) fn representation(operation:WorkspaceOperationView<'_>)->Option<WorkspaceRepresentation> {
    geometry::output_geometry(operation)?;
    let WorkspaceOperationKindView::View(name)=operation.kind else{return None};
    let target=geometry::selected(name)?;let dtype=target.floating()?;
    let input=operation.inputs.get(0)?;let source=geometry::Dtype::from_layout(input)?;
    // A widening view either requires an already row-major source or takes
    // its real contiguous temporary. Same-width aliases never prove row order.
    let rows=target.bytes()>source.bytes()||input.representation().is_some_and(|r|r.row_contiguous())||input.shape().is_empty();
    let last=rows||input.representation().is_some_and(|r|r.last_axis_contiguous());
    Some(WorkspaceRepresentation::new(dtype,rows).with_last_axis_contiguous(last))
}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod tests {
    use super::*;
    use eredu_nn::Tensor;
    #[test]
    fn cpu_byte_view_keeps_exact_scalar_ratio_alias_and_possible_copy_source(){
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for dtype in [WorkspaceFloatingType::Float32,WorkspaceFloatingType::Float16,WorkspaceFloatingType::Bfloat16] {
            let context=WorkspaceContext::new(cpu);
            let storage=WorkspaceExistingStorage::try_new(Some(4096),&context).unwrap();
            let source=WorkspaceTensor::existing_with_storage(context.layout(&[2,3],WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&storage,&context).unwrap();
            context.begin_state_span([&source]).unwrap();
            let bytes=geometry::trace(&source,geometry::Dtype::U8,&context).unwrap();
            assert_eq!(bytes.layout().dtype(),WorkspaceDtype::Uint8);assert_eq!(bytes.layout().representation(),None);
            let encoded=context.report(std::slice::from_ref(&bytes)).unwrap();
            let plan=cpu.plan(encoded.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(plan.alias_input,Some(0));assert_eq!(plan.population.births,0);
            assert_eq!(encoded.tensor_buffers.total_bytes,Some(0));assert_eq!(encoded.state.as_ref().unwrap().retained_bytes,Some(4096));
            let target=match dtype {WorkspaceFloatingType::Float32=>geometry::Dtype::F32,
                WorkspaceFloatingType::Float16=>geometry::Dtype::F16,WorkspaceFloatingType::Bfloat16=>geometry::Dtype::Bf16};
            context.begin_span();let decoded=geometry::trace(&bytes,target,&context).unwrap();
            assert_eq!(decoded.shape(),source.shape());assert_eq!(decoded.layout().representation(),Some(WorkspaceRepresentation::new(dtype,true)));
            let report=context.report(&[decoded]).unwrap();let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            // Integer layouts carry no invented row-order evidence. The same
            // native View source can need one temporary; its actual alias path
            // consumes less when the received bytes are already row-major.
            assert_eq!(plan.population.births,1);assert_eq!(plan.alias_input,None);assert!(plan.output_bytes>=report.operations[0].outputs[0].bytes().unwrap());
            assert!(geometry::inspect(report.operations[0].as_view()).is_some());
            // Metal's possible-copy declaration obeys that same represented
            // allocation minimum while retaining the exact native byte ratio.
            let metal=WorkspaceContext::new(ordinary);
            let raw=WorkspaceTensor::existing(metal.layout(bytes.shape(),WorkspaceDtype::Uint8).unwrap(),&metal).unwrap();
            let typed=geometry::trace(&raw,target,&metal).unwrap();
            assert_eq!(typed.shape(),source.shape());
            assert_eq!(typed.layout().representation().unwrap().dtype(),dtype);
            assert!(metal.report(&[typed]).unwrap().unpriced_operations.is_empty());
        }
    }
}
