//! Views retain source identity; a proved reshape copy uses the shared General worker.
use super::*;
pub(super) fn inspect(operation:WorkspaceOperationView<'_>,mechanism:MlxCpuWorkspaceMechanisms)
    ->facts::FactResult<Option<OperationPlan>> {
    let name=match operation.kind {
        WorkspaceOperationKindView::Transpose(_)=>"transpose",
        WorkspaceOperationKindView::View(name@("reshape"|"expand_dims"|"squeeze"|"transpose"))=>name,
        _=>return Ok(None),
    };
    if operation.inputs.len()!=1||operation.outputs.len()!=1 {
        return Err(MlxWorkspaceFactError::descriptor("CPU row view population differs"));
    }
    let input=operation.inputs.get(0).expect("view source");
    let output=operation.outputs.get(0).expect("view output");
    let (rank,output_rank)=(input.shape().len(),output.shape().len());
    if rank>4||output_rank>4||input.dtype()!=WorkspaceDtype::Float32||output.dtype()!=input.dtype()||
        input.shape().iter().chain(output.shape()).any(|&n|n<=0) {return Ok(None);}
    let Some(representation)=input.representation() else{return Ok(None)};
    // The shared native alias, unit-axis and reshape sources validate F16
    // item size and actual strides alongside F32/BF16; preserve that dtype.
    if (name!="transpose"&&name!="reshape"&&!representation.row_contiguous())||!matches!(representation.dtype(),WorkspaceFloatingType::Float32|WorkspaceFloatingType::Float16|WorkspaceFloatingType::Bfloat16) {
        return Ok(None);
    }
    if input.elements()?!=output.elements()? {return Err(MlxWorkspaceFactError::descriptor("CPU row view changes element count"));}
    if input.elements()?>i32::MAX as u64 {return Ok(None);}
    if name!="reshape"&&name!="transpose"&&(!input.shape().iter().filter(|&&n|n!=1).eq(output.shape().iter().filter(|&&n|n!=1))||
        (name=="expand_dims"&&rank>=output_rank)||(name=="squeeze"&&rank<=output_rank)) {
        return Err(MlxWorkspaceFactError::descriptor("CPU unit-axis view changes nonunit geometry"));
    }
    if name=="transpose" {
        if rank!=output_rank {return Err(MlxWorkspaceFactError::descriptor("CPU transpose rank differs"));}
        if let WorkspaceOperationKindView::Transpose(axes)=operation.kind {
            if axes.len()!=rank || axes.iter().enumerate().any(|(i,&axis)|axis>=rank ||
                axes[..i].contains(&axis) || output.shape()[i]!=input.shape()[axis]) {
                return Err(MlxWorkspaceFactError::descriptor("CPU transpose permutation differs"));
            }
        }
        // Legacy shapes omit the permutation. Equal multiset geometry permits every
        // actual native transpose, whose source validates axes and readable
        // strides. It does not prove row-contiguous output.
        for &extent in input.shape() {
            let mut source_count=0usize;let mut output_count=0usize;
            for &value in input.shape(){source_count+=usize::from(value==extent);}
            for &value in output.shape(){output_count+=usize::from(value==extent);}
            if source_count!=output_count{
                return Err(MlxWorkspaceFactError::descriptor("CPU transpose extents differ"));
            }
        }
    }
    let source=match name {
        "reshape"=>if representation.row_contiguous() {
            OperationEvent::cpu_reshape_alias_layout(rank,output_rank,false)
        } else {reshape_source(input,output,representation)},
        "expand_dims"=>OperationEvent::cpu_expand_dims_alias_layout(rank,output_rank,false),
        "squeeze"=>OperationEvent::cpu_squeeze_layout(rank,false),
        "transpose"=>OperationEvent::cpu_transpose_alias_layout(rank,false),
        _=>unreachable!(),
    };
    let Some(source)=source else{return Ok(None)};
    let mut population=CpuPopulation::default();
    let copied=name=="reshape"&&source.backing_births()==1;
    if (source.backing_births()!=0&&!copied)||population.copy(source,1).is_none() {return Ok(None);}
    let frames=[if name=="reshape" && !representation.row_contiguous() {physical_stride_control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?}else{0},
        size_of::<WorkspaceOperationView<'_>>(),size_of::<WorkspaceLayoutView<'_>>()*2,
        size_of::<OperationPlan>(),size_of::<Option<OperationPlan>>(),size_of::<CpuPopulation>(),
        size_of::<CpuCopyEvalLayout>(),size_of::<Option<CpuCopyEvalLayout>>(),size_of::<(usize,usize)>(),
        size_of::<&str>(),size_of::<WorkspaceRepresentation>(),size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<std::slice::Iter<i32>>()*2,size_of::<u64>()*2,size_of::<bool>(),size_of::<std::iter::Enumerate<std::slice::Iter<i32>>>(),
        size_of::<(usize,&i32)>(),size_of::<&[i32]>(),
        size_of::<usize>()*4,size_of::<i32>()*2,size_of::<Option<usize>>()*2,
        size_of::<std::slice::Iter<usize>>(),size_of::<std::iter::Enumerate<std::slice::Iter<usize>>>(),
        size_of::<(usize,&usize)>(),size_of::<&[usize]>(),size_of::<bool>()*2,
        size_of::<(WorkspaceOperationView<'_>,WorkspaceRepresentation)>(),size_of::<WorkspaceRepresentation>(),
        size_of::<[usize;4]>(),size_of::<[i64;4]>(),size_of::<Option<usize>>()*3,
        size_of::<i64>()*2,size_of::<u32>(),size_of::<u16>(),size_of::<usize>()*7,
        size_of::<(&[i32],&[i64],&[i32],bool)>(),size_of::<bool>(),size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<(WorkspaceLayoutView<'_>,WorkspaceLayoutView<'_>,WorkspaceRepresentation)>(),
        size_of::<Option<[i64;4]>>(),size_of::<[u64;1]>(),size_of::<Option<u64>>(),
        size_of::<(WorkspaceOperationView<'_>,WorkspaceRepresentation)>(),size_of::<u64>(),
        size_of::<std::iter::Take<std::iter::Enumerate<std::slice::IterMut<'_,i64>>>>()];
    population.controls=frames.into_iter().try_fold(population.controls.checked_add(size_of_val(&frames))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,|n,b|n.checked_add(b).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
    let output_bytes=if copied {mechanism.allocation.fixed_buffer_capacity(output.bytes()?)?} else {0};
    Ok(Some(OperationPlan{dtype:representation.dtype(),population,alias_input:if copied {None}else{Some(0)},output_bytes,
        scratch_bytes:0,rank:rank.max(output_rank),parameter_shells:0,seeds:0,validations:0}))
}

/// Only dimensions and an existing row-contiguous source can prove this
/// property. Repeated nonunit extents conceal a possible axis exchange.
pub(super) fn transpose_representation(operation:WorkspaceOperationView<'_>,input:WorkspaceRepresentation)->WorkspaceRepresentation {
    let source=operation.inputs.get(0).expect("qualified transpose input");
    let output=operation.outputs.get(0).expect("qualified transpose output");
    if let WorkspaceOperationKindView::Transpose(axes)=operation.kind {
        // An exact permutation preserves row order iff the nonunit source axes
        // remain increasing. Repeated dimensions no longer hide an exchange.
        let mut previous=None;let mut rows=input.row_contiguous();
        for &axis in axes {
            if source.shape()[axis]>1 {
                if previous.is_some_and(|p|p>=axis){rows=false;}
                previous=Some(axis);
            }
        }
        let last=axes.last().copied();
        let last_contiguous=last.is_none_or(|axis|source.shape()[axis]==1) ||
            (last==source.shape().len().checked_sub(1)&&input.last_axis_contiguous()) ||
            (input.row_contiguous()&&last.is_some_and(|axis|
                source.shape()[axis+1..].iter().all(|&n|n==1)));
        let mut representation=WorkspaceRepresentation::new(input.dtype(),rows)
            .with_last_axis_contiguous(last_contiguous);
        let rank=source.shape().len();let mut order=[0usize;4];let mut dense=true;
        for position in 0..rank {
            let Some(source_axis)=input.dense_axis_at(rank,position) else {dense=false;break;};
            let Some(output_axis)=axes.iter().position(|&axis|axis==source_axis) else {dense=false;break;};
            order[position]=output_axis;
        }
        if dense {representation=representation.with_dense_axis_order(&order[..rank]).expect("qualified permutation");}
        return representation;
    }
    let mut rows=input.row_contiguous()&&source.shape().iter().filter(|&&n|n!=1)
        .eq(output.shape().iter().filter(|&&n|n!=1));
    for (axis,&extent) in source.shape().iter().enumerate(){
        if extent!=1&&source.shape()[..axis].contains(&extent){rows=false;}
    }
    WorkspaceRepresentation::new(input.dtype(),rows)
}

/// Derive physical strides only from exact row-major/permutation evidence,
/// then ask the shared native planner whether this real reshape copies.
fn reshape_source(input:WorkspaceLayoutView<'_>,output:WorkspaceLayoutView<'_>,
    representation:WorkspaceRepresentation)->Option<CpuCopyEvalLayout> {
    let rank=input.shape().len();
    let strides=physical_strides(input,representation)?;
    OperationEvent::cpu_reshape_layout(input.shape(),&strides[..rank],output.shape(),false)
}
/// Read retained strides or reconstruct only a proved dense permutation.
/// Singleton coordinates never observe a stride, so their dense canonical
/// value is equivalent for the shared native fixed reshape planner.
pub(super) fn physical_strides(input:WorkspaceLayoutView<'_>,representation:WorkspaceRepresentation)->Option<[i64;4]> {
    let rank=input.shape().len();if rank>4 {return None;}
    let mut strides=[0i64;4];
    if rank>0 && representation.element_stride_at(rank,0).is_some() {
        for (axis,stride) in strides.iter_mut().enumerate().take(rank) {
            *stride=i64::try_from(representation.element_stride_at(rank,axis)?).ok()?;
        }
        return Some(strides);
    }
    let mut stride=1i64;
    for position in (0..rank).rev() {
        let axis=representation.dense_axis_at(rank,position)?;
        strides[axis]=stride;stride=stride.checked_mul(i64::from(input.shape()[axis]))?;
    }
    Some(strides)
}
/// Called only after the native planner proved an alias. A one-dimensional
/// reshape has the step of the source's last nonunit logical axis. A copied
/// reshape takes the ordinary row-major output path instead.
pub(super) fn reshape_representation(operation:WorkspaceOperationView<'_>,input:WorkspaceRepresentation)->WorkspaceRepresentation {
    let source=operation.inputs.get(0).expect("qualified reshape source");
    let output=operation.outputs.get(0).expect("qualified reshape output");
    if source.shape()==output.shape(){return input;}
    if output.shape().len()==1 {
        let strides=physical_strides(source,input).expect("qualified reshape strides");
        let step=source.shape().iter().rposition(|&n|n>1).map_or(1,|axis|strides[axis] as u64);
        return WorkspaceRepresentation::new(input.dtype(),step==1 || output.shape()[0]==1)
            .with_element_strides(&[step]).expect("qualified bounded source stride");
    }
    WorkspaceRepresentation::new(input.dtype(),false)
}

pub(super) fn physical_stride_control_bytes()->Option<usize> {
    let parts=[size_of::<(WorkspaceLayoutView<'_>,WorkspaceRepresentation)>(),size_of::<[i64;4]>(),
        size_of::<Option<[i64;4]>>(),size_of::<i64>(),size_of::<usize>()*3,
        size_of::<Option<u64>>(),size_of::<Option<usize>>(),size_of::<Option<i64>>(),
        size_of::<std::slice::IterMut<'_,i64>>(),
        size_of::<std::iter::Take<std::iter::Enumerate<std::slice::IterMut<'_,i64>>>>(),
        size_of::<(usize,&mut i64)>(),size_of::<std::iter::Rev<std::ops::Range<usize>>>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
