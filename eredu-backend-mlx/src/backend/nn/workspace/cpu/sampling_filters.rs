//! Exact children of the shared one-row TopK/TopP/MinP workers.
use super::*;
use safemlx::{CpuUnaryOperation,Dtype};

struct Program { population:CpuPopulation, bytes:u64, seeds:usize, allocation:MetalAllocationFacts }
impl Program {
    fn storage(&mut self,elements:usize,itemsize:u64,births:usize)->Option<()> {
        if births==0 {return Some(());}
        let bytes=u64::try_from(elements).ok()?.checked_mul(itemsize)?;
        let capacity=self.allocation.fixed_buffer_capacity(bytes).ok()?;
        self.bytes=self.bytes.checked_add(capacity.checked_mul(u64::try_from(births).ok()?)?)?;Some(())
    }
    fn copy(&mut self,source:CpuCopyEvalLayout,inputs:usize,elements:usize)->Option<()> {
        self.storage(elements,4,source.backing_births())?;self.population.copy(source,inputs)
    }
    fn alias(&mut self,from:usize,to:usize)->Option<()> {
        self.copy(OperationEvent::cpu_broadcast_alias_layout(from,to,false)?,1,0)
    }
    fn binary(&mut self,kind:CpuBinaryOperation,rank:usize,elements:usize,mask:bool)->Option<()> {
        let source=OperationEvent::cpu_binary_layout(kind,Dtype::Float32,rank,elements,false)?;
        self.storage(elements,if mask{1}else{4},source.backing_births())?;self.population.binary(source)
    }
    fn seed(&mut self)->Option<()> {
        self.seeds=self.seeds.checked_add(1)?;self.storage(1,4,1)?;
        self.population.controls=self.population.controls.checked_add(basic::scalar_f32_control_bytes()?)?;Some(())
    }
    fn mask(&mut self,rank:usize,width:usize)->Option<()> {
        self.seed()?;self.alias(0,rank)?;
        // where's other inputs and all three casts are identities. The
        // actual one-element alias takes the ordinary row Select branch.
        let source=if width==1 {OperationEvent::cpu_select_layout(Dtype::Float32,rank,width,false)?}
            else {OperationEvent::cpu_select_broadcast_layout(rank,width,false)?};
        self.copy(source,3,width)
    }
    fn softmax(&mut self,rank:usize,width:usize)->Option<()> {
        self.copy(OperationEvent::cpu_typed_softmax_layout(Dtype::Float32,true,rank,width,1,false)?,1,width)
    }
}

#[cfg(all(test,target_vendor="apple",feature="metal",not(feature="cuda")))]
mod tests {
    use super::*;
    #[test]
    fn cpu_sampler_filters_join_only_their_actual_one_row_sources() {
        let ordinary=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected=MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu=MlxCpuWorkspaceMechanisms::new(ordinary.allocation(),selected);
        for rank in [2,3] {for width in [1,19,65,4097] {
            let shape=if rank==2{vec![1,width]}else{vec![1,1,width]};
            for kind in [WorkspaceSamplingOperation::TopK{keep:1},WorkspaceSamplingOperation::TopK{keep:7},
                WorkspaceSamplingOperation::TopP,WorkspaceSamplingOperation::MinP] {
                if matches!(kind,WorkspaceSamplingOperation::TopK{keep} if keep as i32>=width){continue;}
                let context=WorkspaceContext::new(cpu);
                let input=WorkspaceTensor::existing(context.layout(&shape,WorkspaceDtype::Float32).unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
                let output=context.execute(WorkspaceOperationKind::Sampling(kind.clone()),&[&input],
                    vec![input.layout().clone()]).unwrap().remove(0);
                let report=context.finish_report(&[output]).unwrap();
                assert!(report.unpriced_operations.is_empty()&&report.unpriced_host_operations.is_empty(),"{kind:?} {shape:?}");
                let plan=cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
                assert_eq!(plan.seeds,match kind{WorkspaceSamplingOperation::TopK{..}=>1,WorkspaceSamplingOperation::TopP=>3,_=>2});
                assert_eq!(plan.output_bytes,ordinary.allocation().fixed_buffer_capacity(width as u64*4).unwrap());
                let recipe=SpeculativeNumericalRecipe::inspect_cpu_equations(&report,ordinary,cpu,&context).unwrap();
                assert_eq!(recipe.kernels,0);assert!(recipe.graph_capacity>0&&recipe.record_capacity>0);
                assert_eq!(recipe.completion.nested_completions,0);
                let mut missing=report.operations[0].clone();missing.inputs[0]=missing.inputs[0].clone().with_representation(None);
                assert!(cpu.plan(missing.as_view()).unwrap().is_none());
            }
        }}
        let context=WorkspaceContext::new(cpu);
        let input=WorkspaceTensor::existing(context.layout(&[2,1,65],WorkspaceDtype::Float32).unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true))),&context).unwrap();
        let output=context.execute(WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::TopP),&[&input],
            vec![input.layout().clone()]).unwrap().remove(0);
        let report=context.finish_report(&[output]).unwrap();assert!(!report.unpriced_operations.is_empty());
    }
}

pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    use WorkspaceSamplingOperation as S;
    let WorkspaceOperationKindView::Sampling(kind)=operation.kind else{return Ok(None)};
    if !matches!(kind,S::TopK{..}|S::TopP|S::MinP){return Ok(None);}
    let Some([input])=operation.inputs.array() else{return Err(MlxWorkspaceFactError::descriptor("CPU filter needs one score source"));};
    let Some([output])=operation.outputs.array() else{return Err(MlxWorkspaceFactError::descriptor("CPU filter needs one output"));};
    if input.dtype()!=WorkspaceDtype::Float32||output.dtype()!=input.dtype()||input.shape()!=output.shape(){
        return Err(MlxWorkspaceFactError::descriptor("CPU filter changes its score geometry"));
    }
    let rank=input.shape().len();
    // These are the actual public sampler's rank-two/rank-three complete
    // score rows. Multirow arrays and unrepresented floating inputs stay out.
    if !(2..=3).contains(&rank)||input.shape().iter().any(|&n|n<=0)||
        input.representation()!=Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,true)){
        return Ok(None);
    }
    let width=usize::try_from(*input.shape().last().expect("nonempty rank"))?;
    if input.elements()?!=width as u64||width>i32::MAX as usize{return Ok(None);}
    let mut p=Program{population:CpuPopulation::default(),bytes:0,seeds:0,allocation:mechanism.allocation};
    let source=(|| {
        match kind {
            S::TopK{keep} if *keep!=0&&(*keep as usize)<width=>{
                let keep=*keep as usize;
                p.copy(OperationEvent::cpu_partition_row_layout(rank,width,false)?,1,width)?;
                p.copy(OperationEvent::cpu_slice_layout(rank,false)?,1,0)?;
                if keep>1 {p.copy(OperationEvent::cpu_row_min_layout(rank,keep,1,false)?,1,1)?;}
                p.alias(rank,rank)?;
                p.binary(CpuBinaryOperation::Less,rank,width,true)?;
                p.mask(rank,width)?;
            }
            S::TopP=>{
                let negative=OperationEvent::cpu_unary_layout(CpuUnaryOperation::Negative,Dtype::Float32,rank,false)?;
                p.storage(width,4,negative.backing_births())?;p.population.unary(negative)?;
                p.copy(OperationEvent::cpu_argsort_row_layout(rank,width,false)?,1,width)?;
                // take_along_axis's equal-shape inputs are borrowed unchanged;
                // the real CPU primitive is GatherAxis, not Gather+Squeeze.
                p.copy(OperationEvent::cpu_gather_axis_row_layout(rank,width,false)?,2,width)?;
                p.softmax(rank,width)?;
                p.copy(OperationEvent::cpu_scan_sum_row_layout(rank,width,false)?,1,width)?;
                p.binary(CpuBinaryOperation::Subtract,rank,width,false)?;
                p.seed()?;p.alias(0,rank)?;
                p.binary(CpuBinaryOperation::Greater,rank,width,true)?;
                p.mask(rank,width)?;
                // full(shape, stored minimum) broadcasts the scalar then
                // copies it. The following F32 AsType is an identity.
                p.seed()?;p.alias(0,rank)?;
                p.copy(OperationEvent::cpu_scalar_full_layout(Dtype::Float32,rank,width,false)?,1,width)?;
                // put_along_axis's updates/indices already match dtype/shape;
                // no conversion/broadcast backing is invented for them.
                p.copy(OperationEvent::cpu_scatter_axis_layout(Dtype::Uint32,rank,width,width,false)?,3,width)?;
            }
            S::MinP=>{
                p.softmax(rank,width)?;
                if width>1 {p.copy(OperationEvent::cpu_maximum_row_layout(rank,width,false)?,1,1)?;}
                p.seed()?;p.alias(0,rank)?;
                p.binary(CpuBinaryOperation::Multiply,rank,1,false)?;
                if width>1 {p.alias(rank,rank)?;}
                p.binary(CpuBinaryOperation::Less,rank,width,true)?;
                p.mask(rank,width)?;
            }
            _=>return None,
        }
        Some(())
    })();
    if source.is_none(){return Ok(None);}
    let output_bytes=mechanism.allocation.fixed_buffer_capacity(output.bytes()?)?;
    let scratch_bytes=p.bytes.checked_sub(output_bytes).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let controls=crate::backend::runtime::generation::sampling_filter_control_bytes()
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let frames=[controls,size_of::<Program>(),size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>()*2,size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<CpuPopulation>(),size_of::<OperationPlan>(),size_of::<Option<OperationPlan>>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<usize>()*5,size_of::<u64>()*5,size_of::<Option<()>>(),size_of::<bool>(),
        size_of::<Dtype>(),size_of::<CpuUnaryOperation>(),size_of::<CpuBinaryOperation>(),
        size_of::<(&mut Program,usize,u64,usize)>(),size_of::<(&mut Program,CpuCopyEvalLayout,usize,usize)>(),
        size_of::<(&mut Program,usize,usize)>(),size_of::<(&mut Program,CpuBinaryOperation,usize,usize,bool)>(),
        size_of::<&mut Program>(),size_of::<std::slice::Iter<'_,i32>>(),
        size_of::<[WorkspaceLayoutView<'_>;1]>()*2,size_of::<Option<[WorkspaceLayoutView<'_>;1]>>()*2,
        size_of::<Result<usize,std::num::TryFromIntError>>(),size_of::<facts::FactResult<Option<OperationPlan>>>()];
    p.population.controls=frames.into_iter().try_fold(p.population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,usize::checked_add)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan{dtype:WorkspaceFloatingType::Float32,population:p.population,alias_input:None,
        output_bytes,scratch_bytes,rank,parameter_shells:0,seeds:p.seeds,validations:0}))
}
