//! Exact integer byte framing through the ordinary reshape/concat/slice/add workers.
use super::*;
use safemlx::Dtype;

pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    let kind=match operation.kind {
        WorkspaceOperationKindView::Elementwise("scalar_u8")=>0,
        WorkspaceOperationKindView::View("reshape")=>1,
        WorkspaceOperationKindView::Concatenate=>2,
        WorkspaceOperationKindView::StaticSlice {..}=>3,
        WorkspaceOperationKindView::Elementwise("add")=>4,
        _=>return Ok(None),
    };
    if operation.outputs.len()!=1 {return Ok(None);}
    let output=operation.outputs.get(0).expect("one byte output");
    if output.dtype()!=WorkspaceDtype::Uint8{return Ok(None);}
    let rank=output.shape().len();let count=output.elements()?;
    if rank>5||count==0||count>i32::MAX as u64||output.shape().iter().any(|&n|n<=0){return Ok(None);}
    if operation.inputs.iter().any(|input|input.dtype()!=WorkspaceDtype::Uint8||input.shape().len()>5||
        input.shape().iter().any(|&n|n<=0)){return Ok(None);}
    let mut population=CpuPopulation::default();let mut alias=None;let mut shells=0;let mut seeds=0;
    match kind {
        0=>{
            if !basic::is_scalar_u8(operation){return Err(MlxWorkspaceFactError::descriptor("CPU byte seed is not one eager U8 scalar"));}
            population.controls=basic::scalar_u8_control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
            seeds=1;
        }
        1=>{
            if operation.inputs.len()!=1{return Ok(None);}
            let input=operation.inputs.get(0).expect("one reshape source");
            if input.elements()?!=count{return Err(MlxWorkspaceFactError::descriptor("CPU byte reshape changes element count"));}
            if input.shape()==output.shape(){alias=Some(0);shells=1;}
            else {
                // Integer workspace layouts carry no stride claim. The actual
                // native planner validates positive strides/backing and may
                // need its one General copy; price that real branch here.
                let Some(source)=OperationEvent::cpu_reshape_copy_layout(input.shape().len(),rank,false) else{return Ok(None)};
                if source.backing_births()!=1||population.copy(source,1).is_none(){return Ok(None);}
            }
        }
        2=>{
            if operation.inputs.len()!=2||rank!=1{return Ok(None);}
            let left=operation.inputs.get(0).expect("two concatenation sources");
            let right=operation.inputs.get(1).expect("two concatenation sources");
            if left.shape().len()!=1||right.shape().len()!=1||left.shape()[0].checked_add(right.shape()[0])!=Some(output.shape()[0]){
                return Err(MlxWorkspaceFactError::descriptor("CPU byte concatenation dimensions differ"));
            }
            let Some(source)=OperationEvent::cpu_concatenate_layout(Dtype::Uint8,rank,
                usize::try_from(left.elements()?)?,usize::try_from(right.elements()?)?,false) else{return Ok(None)};
            // Same U8 dtype makes both ordinary AsType calls identity. The
            // shared Concatenate owns exactly two copy jobs and one output.
            if source.backing_births()!=1||population.concatenate(source,2).is_none(){return Ok(None);}
        }
        3=>{
            if !basic::is_static_slice(operation){return Err(MlxWorkspaceFactError::descriptor("CPU byte slice coordinates differ"));}
            let input=operation.inputs.get(0).expect("checked slice input");
            let WorkspaceOperationKindView::StaticSlice {strides,..}=operation.kind else{unreachable!()};
            if rank!=1||strides!=[1]{return Ok(None);}
            alias=Some(0);
            if input.shape()==output.shape(){shells=1;}
            else {
                let Some(source)=OperationEvent::cpu_slice_layout(rank, false, false) else{return Ok(None)};
                if source.backing_births()!=0||population.copy(source,1).is_none(){return Ok(None);}
            }
        }
        4=>{
            if operation.inputs.len()!=2||rank!=1{return Ok(None);}
            let input=operation.inputs.get(0).expect("two addition sources");
            let scalar=operation.inputs.get(1).expect("two addition sources");
            if input.shape()!=output.shape()||!scalar.shape().is_empty(){return Ok(None);}
            let Some(broadcast)=OperationEvent::cpu_broadcast_alias_layout(0,rank,false) else{return Ok(None)};
            let Some(add)=OperationEvent::cpu_binary_layout(CpuBinaryOperation::Add,Dtype::Uint8,rank,usize::try_from(count)?,false) else{return Ok(None)};
            if broadcast.backing_births()!=0||add.backing_births()!=1||population.copy(broadcast,1).is_none()||population.binary(add).is_none(){return Ok(None);}
        }
        _=>unreachable!(),
    }
    let frames=[size_of::<WorkspaceOperationView<'_>>(),size_of::<WorkspaceLayoutView<'_>>()*3,
        size_of::<MlxCpuWorkspaceMechanisms>(),size_of::<CpuPopulation>(),size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<Option<usize>>(),size_of::<usize>()*5,size_of::<u64>(),size_of::<i32>(),
        size_of::<std::slice::Iter<'_,i32>>(),size_of::<&[i32]>(),size_of::<[i32;1]>()*3,
        size_of::<Option<i32>>(),size_of::<Result<usize,std::num::TryFromIntError>>(),
        size_of::<(&safemlx::Array,&[i32],&safemlx::Stream)>(),
        size_of::<(&safemlx::Array,&safemlx::Array,&safemlx::Stream)>(),
        size_of::<(&safemlx::Array,&[i32],&[i32],&[i32],&safemlx::Stream)>(),
        size_of::<Result<safemlx::Array,safemlx::error::Exception>>(),
        if kind==2 {safemlx::ops::concatenate_axis_control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?}else{0},
        if kind==3 {crate::tensor::narrow::control_bytes(rank).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?}else{0}];
    population.controls=frames.into_iter().try_fold(population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,usize::checked_add).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {dtype:WorkspaceFloatingType::Float32,population,alias_input:alias,
        output_bytes:if alias.is_some(){0}else{mechanism.allocation.fixed_buffer_capacity(count)?},scratch_bytes:0,
        rank,parameter_shells:shells,seeds,validations:0}))
}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod tests {
    use super::*;
    use eredu_nn::Tensor;
    use crate::backend::nn::boundary_frame;
    #[test]
    fn cpu_boundary_frame_quotes_exact_bytes_alignment_and_typed_payload(){
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),choice);
        for dtype in [WorkspaceFloatingType::Float32,WorkspaceFloatingType::Float16,WorkspaceFloatingType::Bfloat16] {
            for header_len in [3,4] {
                let context=WorkspaceContext::new(cpu);
                let storage=WorkspaceExistingStorage::try_new(Some(4096),&context).unwrap();
                let source=WorkspaceTensor::existing_with_storage(context.layout(&[1,2,3],WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(dtype,true))),&storage,&context).unwrap();
                let header=WorkspaceTensor::existing(context.layout(&[header_len],WorkspaceDtype::Uint8).unwrap(),&context).unwrap();
                context.begin_state_span([&source,&header]).unwrap();
                let frame=boundary_frame::encode(&boundary_frame::Workspace(&context),&source,&header).unwrap();
                let (copied_header,decoded)=boundary_frame::split(&boundary_frame::Workspace(&context),&frame,header_len,&source).unwrap();
                assert_eq!(copied_header.shape(),[header_len]);assert_eq!(decoded.shape(),source.shape());
                assert_eq!(decoded.layout().representation(),Some(WorkspaceRepresentation::new(dtype,true)));
                let report=context.report(&[copied_header,decoded]).unwrap();
                assert!(report.operations.iter().all(|op|cpu.plan(op.as_view()).unwrap().is_some()));
                assert_eq!(report.operations.iter().filter(|op|matches!(op.kind,WorkspaceOperationKind::Elementwise("scalar_u8"))).count(),usize::from(header_len==3));
                assert!(report.operations.iter().any(|op|matches!(op.kind,WorkspaceOperationKind::StaticSlice {..})));
                let recipe=super::super::super::resident_recipe::SpeculativeNumericalRecipe::inspect_cpu_outputs(
                    &report,2,ordinary,cpu,&context).unwrap();
                drop(recipe);
                // Generic initialization is not the exact borrowed byte seed.
                let generic=context.report(&[WorkspaceTensor::initialized(&[],WorkspaceDtype::Uint8,&context).unwrap()]).unwrap();
                let initialize=generic.operations.iter().find(|op|matches!(op.kind,WorkspaceOperationKind::Initialize)).unwrap();
                assert!(cpu.plan(initialize.as_view()).unwrap().is_none());
            }
        }
    }
}
