//! Actual floating CPU SDPA/GQA fallback, with optional explicit Boolean mask.
use super::*;
use eredu_nn::AttentionArithmetic;
use safemlx::Dtype;
fn cast(p:&mut CpuPopulation,source:Dtype,destination:Dtype,rank:usize,count:usize)->Option<()> {
    p.copy(OperationEvent::cpu_cast_layout(source,destination,rank,count,false)?,1)
}
pub(super) fn matmul(p:&mut CpuPopulation,mechanism:MlxCpuWorkspaceMechanisms,dtype:Dtype,rank:usize,
    m:usize,n:usize,k:usize,batches:usize,copies:usize)->Option<()> {
    // The shared frontend receives already-promoted values, then broadcasts
    // the actual batch geometries while retaining both matrix interiors.
    p.copy(OperationEvent::cpu_broadcast_alias_layout(rank,rank,false)?,1)?;
    p.copy(OperationEvent::cpu_broadcast_alias_layout(rank,rank,false)?,1)?;
    let (rank,m,n,k,batches)=(u32::try_from(rank).ok()?,u32::try_from(m).ok()?,
        u32::try_from(n).ok()?,u32::try_from(k).ok()?,u32::try_from(batches).ok()?);
    let selected=mechanism.matmul.selected();
    let geometry=if dtype==Dtype::Float16 {selected.float16_geometry(rank,m,n,k,batches)}
        else {selected.geometry(rank,m,n,k,batches)}.ok()?;
    let native=match dtype {
        Dtype::Float32=>if copies==0 {mechanism.matmul.eval_layout(geometry,false)?}
            else {mechanism.matmul.eval_layout_with_copies(geometry,copies,false)?},
        Dtype::Float16=>mechanism.matmul.float16_eval_layout(geometry,copies,false)?,
        Dtype::Bfloat16=>OperationEvent::cpu_bf16_matmul_copy_layout(rank as usize,m as usize,n as usize,
            k as usize,batches as usize,copies,false)?,
        _=>return None,
    };
    p.copy(native,2)
}
fn native_dtype(dtype:WorkspaceFloatingType)->Dtype {
    match dtype {WorkspaceFloatingType::Float32=>Dtype::Float32,
        WorkspaceFloatingType::Float16=>Dtype::Float16,WorkspaceFloatingType::Bfloat16=>Dtype::Bfloat16}
}
pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    if !matches!(operation.kind,WorkspaceOperationKindView::Attention {causal:false,window:None,sinks:false,
        softcap:false,arithmetic:AttentionArithmetic::Fused}) {return Ok(None);}
    if !(3..=4).contains(&operation.inputs.len())||operation.outputs.len()!=1 {return Ok(None);}
    let q=operation.inputs.get(0).expect("query source");let k=operation.inputs.get(1).expect("key source");
    let v=operation.inputs.get(2).expect("value source");let output=operation.outputs.get(0).expect("attention output");
    for (index,input) in operation.inputs.slice(0..3).expect("three attention operands").iter().enumerate() {
        if input.shape().len()!=4||input.shape().iter().any(|&n|n<=0)||input.dtype()!=WorkspaceDtype::Float32||
            !input.representation().is_some_and(|r|if index==2 {r.last_axis_contiguous()} else {r.row_contiguous()}) {return Ok(None);}
    }
    let query_dtype=q.representation().expect("checked query representation").dtype();
    let key_dtype=k.representation().expect("checked key representation").dtype();
    let value_dtype=v.representation().expect("checked value representation").dtype();
    let dtype=super::super::representation::promote(super::super::representation::promote(query_dtype,key_dtype),value_dtype);
    let native=native_dtype(dtype);
    let (b,h,q_len,d)=(q.shape()[0],q.shape()[1],q.shape()[2],q.shape()[3]);
    let (kv,k_len,dv)=(k.shape()[1],k.shape()[2],v.shape()[3]);
    if k.shape()[0]!=b||v.shape()[0]!=b||v.shape()[1]!=kv||v.shape()[2]!=k_len||k.shape()[3]!=d||h%kv!=0||
        output.shape()!=[b,h,q_len,dv]||output.dtype()!=WorkspaceDtype::Float32 {
        return Err(MlxWorkspaceFactError::descriptor("CPU attention query/key/value geometry differs"));
    }
    let value_copies=usize::from(value_dtype==dtype&&!v.representation().is_some_and(|r|r.row_contiguous()));
    let mask=operation.inputs.get(3);
    if let Some(mask)=mask {
        if mask.dtype()!=WorkspaceDtype::Bool {return Ok(None);}
        if mask.shape().len()>4||mask.shape().iter().any(|n|*n<=0)
            ||mask.shape().iter().rev().zip([b,h,q_len,k_len].iter().rev())
                .any(|(a,b)|*a!=1&&a!=b) {
            return Err(MlxWorkspaceFactError::descriptor("CPU attention mask does not broadcast to scores"));
        }
    }
    let (b,h,q_len,d,kv,k_len,dv)=(usize::try_from(b)?,usize::try_from(h)?,usize::try_from(q_len)?,usize::try_from(d)?,
        usize::try_from(kv)?,usize::try_from(k_len)?,usize::try_from(dv)?);
    let checked=||->Option<(usize,usize,usize,usize,usize,usize)> {
        let batches=b.checked_mul(h)?;
        let queries=batches.checked_mul(q_len)?.checked_mul(d)?;
        let keys=batches.checked_mul(k_len)?.checked_mul(d)?;
        let values=batches.checked_mul(k_len)?.checked_mul(dv)?;
        let scores=batches.checked_mul(q_len)?.checked_mul(k_len)?;
        let output=batches.checked_mul(q_len)?.checked_mul(dv)?;
        if [queries,keys,values,scores,output].into_iter().any(|n|n>i32::MAX as usize) {return None;}
        Some((batches,queries,keys,values,scores,output))
    };
    let Some((batches,queries,keys,values,scores,produced))=checked() else{return Ok(None)};
    let repeats=h/kv;let rank=if repeats>1 {5}else{4};
    let source=(|| {
        let mut p=CpuPopulation::default();
        cast(&mut p,native_dtype(query_dtype),native,4,queries)?;
        cast(&mut p,native_dtype(key_dtype),native,4,keys/repeats)?;
        cast(&mut p,native_dtype(value_dtype),native,4,values/repeats)?;
        cast(&mut p,native,native,0,1)?;cast(&mut p,native,native,4,queries)?;
        p.copy(OperationEvent::cpu_broadcast_alias_layout(0,4,false)?,1)?;
        p.copy(OperationEvent::cpu_broadcast_alias_layout(4,4,false)?,1)?;
        p.binary(OperationEvent::cpu_binary_layout(CpuBinaryOperation::Multiply,native,4,queries,false)?)?;
        if repeats>1 {
            p.copy(OperationEvent::cpu_reshape_alias_layout(4,5,false)?,1)?;
            p.copy(OperationEvent::cpu_expand_dims_alias_layout(4,5,false)?,1)?;
            p.copy(OperationEvent::cpu_expand_dims_alias_layout(4,5,false)?,1)?;
        }
        if let Some(mask)=mask {
            p.copy(OperationEvent::cpu_broadcast_alias_layout(mask.shape().len(),4,false)?,1)?;
        }
        p.copy(OperationEvent::cpu_transpose_alias_layout(rank,false)?,1)?;
        matmul(&mut p,mechanism,native,rank,q_len,k_len,d,batches,0)?;
        if mask.is_some() {
            // The frontend first broadcasts to [B,H,Q,K]. GQA unflattens H
            // into [KV,repeats]; the shared where frontend casts/broadcasts
            // its three actual operands before the unchanged Select worker.
            if repeats>1 {p.copy(OperationEvent::cpu_reshape_alias_layout(4,5,false)?,1)?;}
            p.copy(OperationEvent::cpu_cast_layout(Dtype::Bool,Dtype::Bool,rank,scores,false)?,1)?;
            cast(&mut p,native,native,rank,scores)?;cast(&mut p,native,native,0,1)?;
            p.copy(OperationEvent::cpu_broadcast_alias_layout(rank,rank,false)?,1)?;
            p.copy(OperationEvent::cpu_broadcast_alias_layout(rank,rank,false)?,1)?;
            p.copy(OperationEvent::cpu_broadcast_alias_layout(0,rank,false)?,1)?;
            p.copy(OperationEvent::cpu_typed_select_broadcast_layout(native,rank,scores,false)?,3)?;
        }
        cast(&mut p,native,native,rank,scores)?;
        p.copy(OperationEvent::cpu_typed_softmax_layout(native,true,rank,k_len,batches.checked_mul(q_len)?,false)?,1)?;
        matmul(&mut p,mechanism,native,rank,q_len,dv,k_len,batches,value_copies)?;
        if repeats>1 {p.copy(OperationEvent::cpu_reshape_alias_layout(5,4,false)?,1)?;}
        let controls=if mask.is_some() {
            OperationEvent::cpu_sdpa_array_mask_control_bytes(rank,queries,keys,values,scores)?
        } else {OperationEvent::cpu_sdpa_fallback_control_bytes(rank,queries,keys,values,scores)?};
        p.controls=p.controls.checked_add(controls)?;
        Some(p)
    })();
    let Some(mut population)=source else{return Ok(None)};
    // Every possible cast/score/result backing is bounded by a population from
    // this exact batch/head/position/width geometry, including repeated K/V.
    let maximum=[queries,keys,values,scores,produced].into_iter().max().expect("fixed nonempty extents");
    let capacity=mechanism.allocation.fixed_buffer_capacity(facts::mul(u64::try_from(maximum)?,4)?)?;
    let output_bytes=mechanism.allocation.fixed_buffer_capacity(facts::mul(u64::try_from(produced)?,4)?)?;
    let seeds=1+usize::from(mask.is_some());
    let total=facts::add(facts::mul(capacity,u64::try_from(population.births)?)?,
        facts::mul(mechanism.allocation.fixed_buffer_capacity(4)?,u64::try_from(seeds)?)?)?;
    let scratch_bytes=total.checked_sub(output_bytes).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let frames=[size_of::<CpuPopulation>()*3,size_of::<Option<CpuPopulation>>(),size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),size_of::<Option<WorkspaceLayoutView<'_>>>(),
        size_of::<usize>()*3,size_of::<[i32;4]>(),size_of::<WorkspaceOperationView<'_>>(),size_of::<WorkspaceLayoutView<'_>>()*4,
        size_of::<MlxCpuWorkspaceMechanisms>(),size_of::<usize>()*19,size_of::<[usize;5]>()*2,
        size_of::<std::array::IntoIter<usize,5>>(),size_of::<(usize,usize,usize,usize,usize,usize)>(),
        size_of::<Option<(usize,usize,usize,usize,usize,usize)>>(),size_of::<[i32;4]>(),
        size_of::<(&mut CpuPopulation,MlxCpuWorkspaceMechanisms,Dtype,usize,usize,usize,usize,usize,usize)>(),
        size_of::<(&mut CpuPopulation,Dtype,Dtype,usize,usize)>(),size_of::<WorkspaceFloatingType>()*4,
        size_of::<Dtype>()*3,size_of::<eredu_nn::SelectedCpuMatmul>(),size_of::<eredu_nn::CpuMatmulGeometry>(),
        size_of::<Result<eredu_nn::CpuMatmulGeometry,eredu_nn::CpuMatmulError>>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),size_of_val(&checked),size_of::<u64>()*4,
        size_of::<std::slice::Iter<i32>>(),size_of::<(&crate::MlxTensor,&crate::MlxTensor,&crate::MlxTensor,f32,&safemlx::Stream)>(),
        size_of::<Result<crate::MlxTensor,eredu_nn::Error>>(),
        size_of::<std::iter::Enumerate<eredu_nn::workspace::WorkspaceLayoutIter<'_>>>()];
    population.controls=frames.into_iter().try_fold(population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,|n,b|n.checked_add(b).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
    Ok(Some(OperationPlan{dtype,population,output_bytes,scratch_bytes,
        rank,parameter_shells:0,alias_input:None,seeds,validations:0}))
}
