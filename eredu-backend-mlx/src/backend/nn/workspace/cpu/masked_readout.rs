//! Existing ordered vocabulary worker, with its actual CPU child sources.
use super::*;
use safemlx::Dtype;
#[derive(Clone,Copy)]
struct Program { native:CpuPopulation, bytes:u64, allocation:MetalAllocationFacts }
impl Program {
    fn physical(&mut self,elements:usize,births:usize)->Option<()> {
        if births==0 {return Some(());}
        let bytes=u64::try_from(elements).ok()?.checked_mul(4)?;
        let capacity=self.allocation.fixed_buffer_capacity(bytes).ok()?;
        self.bytes=self.bytes.checked_add(capacity.checked_mul(u64::try_from(births).ok()?)?)?;
        Some(())
    }
    fn copy(&mut self,source:CpuCopyEvalLayout,inputs:usize,elements:usize)->Option<()> {
        self.physical(elements,source.backing_births())?;self.native.copy(source,inputs)
    }
    fn alias(&mut self,rank:usize,output_rank:usize)->Option<()> {
        self.copy(OperationEvent::cpu_broadcast_alias_layout(rank,output_rank,false)?,1,0)
    }
    fn reshape(&mut self,rank:usize,output_rank:usize)->Option<()> {
        self.copy(OperationEvent::cpu_reshape_alias_layout(rank,output_rank,false)?,1,0)
    }
    fn cast(&mut self,dtype:Dtype,rank:usize,elements:usize)->Option<()> {
        self.copy(OperationEvent::cpu_cast_layout(dtype,dtype,rank,elements,false)?,1,elements)
    }
    fn gather(&mut self,dtype:Dtype,index:Dtype,index_rank:usize,table:usize,
        indices:usize,width:usize)->Option<()> {
        self.alias(index_rank,index_rank)?;self.cast(index,index_rank,indices)?;
        self.copy(OperationEvent::cpu_gather_layout(dtype,index,2,index_rank,table,indices,width,false)?,
            2,indices.checked_mul(width)?)?;
        self.copy(OperationEvent::cpu_squeeze_layout(index_rank+2,false)?,1,0)
    }
}
pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    let Some(geometry)=super::super::readout::geometry(operation)? else {return Ok(None)};
    if geometry.batch<=0||geometry.sequence<=0 {return Ok(None);}
    for input in operation.inputs.iter().take(3) {
        if !input.representation().is_some_and(|r|r.dtype()==WorkspaceFloatingType::Float32&&r.row_contiguous()) {
            return Ok(None);
        }
    }
    let index=match operation.inputs.get(3).expect("validated ordering").dtype() {
        WorkspaceDtype::Int32=>Dtype::Int32,WorkspaceDtype::Uint32=>Dtype::Uint32,_=>return Ok(None),
    };
    let (b,s,h,v,c,k)=(usize::try_from(geometry.batch)?,usize::try_from(geometry.sequence)?,
        usize::try_from(geometry.hidden)?,usize::try_from(geometry.vocabulary)?,
        usize::try_from(geometry.centroids)?,usize::try_from(geometry.selected)?);
    let checked=||->Option<[usize;7]> {
        let rows=b.checked_mul(s)?;let hidden=rows.checked_mul(h)?;
        let centroids=rows.checked_mul(c)?;let picks=rows.checked_mul(k/(v/c))?;
        let selected=rows.checked_mul(k)?;let weights=selected.checked_mul(h)?;
        let output=rows.checked_mul(v)?;
        if [hidden,centroids,picks,selected,weights,output,v.checked_mul(h)?].into_iter().any(|n|n>i32::MAX as usize) {
            return None;
        }
        Some([rows,hidden,centroids,picks,selected,weights,output])
    };
    let Some([rows,_hidden,centroids,picks,selected,_weights,output])=checked() else{return Ok(None)};
    let source=(|| {
        let mut p=Program{native:CpuPopulation::default(),bytes:0,allocation:mechanism.allocation};
        // Full centroid partition owns one U32 output and its existing bank.
        let partition=OperationEvent::cpu_argpartition_source_layout(3,centroids,false)?;
        p.native.add(CpuPopulation{construction_entries:1,primitives:1,input_edges:1,hidden_leaves:0,maximum_operands:1,maximum_captures:0,births:1,
            extents:partition.allocation_extents(),controls:partition.control_bytes()?})?;
        p.physical(centroids,1)?;
        // Static top-centroid slice and unchanged-shape reshape candidate.
        p.copy(OperationEvent::cpu_slice_layout(3, false, false)?,1,0)?;p.reshape(3,3)?;
        p.reshape(1,2)?;
        p.gather(index,Dtype::Uint32,3,v,picks,v/c)?;
        p.reshape(4,1)?;
        p.gather(Dtype::Float32,index,1,v.checked_mul(h)?,selected,h)?;
        p.reshape(2,4)?;
        // Shared hidden NewAxis subscript and selected-weight transpose.
        // The actual selected F32 matmul never copies the complete table.
        p.copy(OperationEvent::cpu_slice_layout(3, false, false)?,1,0)?;p.reshape(3,4)?;
        p.copy(OperationEvent::cpu_transpose_alias_layout(4,false)?,1,0)?;
        p.alias(4,4)?;p.alias(4,4)?;
        let product=mechanism.matmul.selected().geometry(4,1,u32::try_from(k).ok()?,
            u32::try_from(h).ok()?,u32::try_from(rows).ok()?).ok()?;
        p.copy(mechanism.matmul.eval_layout(product,false)?,2,selected)?;
        p.copy(OperationEvent::cpu_squeeze_layout(4,false)?,1,0)?;
        if k>1 {p.copy(OperationEvent::cpu_row_min_layout(3,k,rows,false)?,1,rows)?;}
        // Per-row minimum minus the actual F32 margin scalar, preserving the
        // ordinary cast/broadcast constructor envelope and exact extents.
        p.cast(Dtype::Float32,3,rows)?;p.cast(Dtype::Float32,0,1)?;
        p.alias(3,3)?;p.alias(0,3)?;
        let subtract=OperationEvent::cpu_binary_layout(CpuBinaryOperation::Subtract,Dtype::Float32,3,rows,false)?;
        p.physical(rows,subtract.backing_births())?;p.native.binary(subtract)?;
        p.alias(3,3)?;p.cast(Dtype::Float32,3,output)?;
        p.copy(OperationEvent::cpu_row_full_layout(3,v,rows,false)?,1,output)?;
        p.reshape(4,3)?;
        // put_along_axis promotes updates, broadcasts both updates, then all
        // three inputs outside the selected axis, before its overwrite job.
        p.cast(Dtype::Float32,3,selected)?;
        for _ in 0..5 {p.alias(3,3)?;}
        p.copy(OperationEvent::cpu_scatter_axis_layout(index,3,output,selected,false)?,3,output)?;
        p.physical(1,1)?; // margin seed, separate from Eval births
        p.native.controls=p.native.controls.checked_add(crate::tensor::masked_readout_control_bytes(false)?)?;
        Some(p)
    })();
    let Some(mut source)=source else{return Ok(None)};
    let output_bytes=mechanism.allocation.fixed_buffer_capacity(facts::mul(u64::try_from(output)?,4)?)?;
    let scratch_bytes=source.bytes.checked_sub(output_bytes).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let frames=[size_of::<Program>()*3,size_of::<Option<Program>>(),size_of::<CpuPopulation>()*2,
        size_of::<OperationPlan>(),size_of::<Option<OperationPlan>>(),size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>()*4,size_of::<super::super::readout::ReadoutGeometry>()*2,
        size_of::<Result<Option<super::super::readout::ReadoutGeometry>,MlxWorkspaceFactError>>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<safemlx::CpuArgPartitionLayout>(),size_of::<Option<safemlx::CpuArgPartitionLayout>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),size_of::<Dtype>()*2,size_of::<usize>()*17,
        size_of::<[usize;7]>()*2,size_of::<Option<[usize;7]>>(),size_of::<std::array::IntoIter<usize,7>>(),
        size_of::<(&mut Program,usize,usize)>(),size_of::<(&mut Program,CpuCopyEvalLayout,usize,usize)>(),
        size_of::<(&mut Program,Dtype,usize,usize)>(),size_of::<(&mut Program,Dtype,Dtype,usize,usize,usize,usize)>(),
        size_of::<eredu_nn::CpuMatmulGeometry>(),size_of::<Result<eredu_nn::CpuMatmulGeometry,eredu_nn::CpuMatmulError>>(),
        size_of::<Option<()>>(),size_of::<u64>()*3,size_of_val(&checked),size_of::<std::ops::Range<usize>>()];
    source.native.controls=frames.into_iter().try_fold(source.native.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,|n,b|n.checked_add(b).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
    Ok(Some(OperationPlan{dtype:WorkspaceFloatingType::Float32,population:source.native,
        alias_input:None,output_bytes,scratch_bytes,rank:5,parameter_shells:17,seeds:1,validations:0}))
}
