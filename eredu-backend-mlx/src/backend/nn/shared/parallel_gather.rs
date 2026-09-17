//! Native primitives for the shared rank-ordered vocabulary Gather equation.
use super::*;
use eredu_nn::{ParallelGatherError, ParallelGatherOperations};
use std::mem::{size_of, size_of_val};

struct Operations<'a> { group:&'a Group, stream:&'a Stream }
impl Operations<'_> {
    fn native<T>(&self,value:Result<T,safemlx::error::Exception>)->Result<T,ComputeError> {
        value.map_err(|cause|self.group.model_native_error(cause))
    }
    fn controls(&self,parts:&[usize])->Result<(),ComputeError> {
        self.charge(parts.iter().copied().try_fold(size_of_val(parts),usize::checked_add)
            .ok_or_else(||self.invalid(ParallelGatherError::Overflow))?)
    }
}
impl ParallelGatherOperations for Operations<'_> {
    type Value=Array;
    fn shape<'a>(&self,value:&'a Array)->&'a [i32] {value.shape()}
    fn charge(&self,bytes:usize)->Result<(),ComputeError> {
        self.group.model_funding().map_or(Ok(()),|funding|funding.reserve_metadata(bytes)
            .map_err(|cause|eredu_nn::workspace::WorkspaceMetadataError::Funding(cause).into()))
    }
    fn invalid(&self,cause:ParallelGatherError)->ComputeError {self.group.model_gather_error(cause)}
    fn zeros(&self,input:&Array,shape:&[i32])->Result<Array,ComputeError> {
        self.controls(&[size_of::<(&Self,&Array,&[i32])>(),size_of::<Dtype>(),
            size_of::<Result<Array,safemlx::error::Exception>>(),size_of::<Result<Array,ComputeError>>()])?;
        self.native(zeros_dtype(shape,input.dtype(),self.stream))
    }
    fn gather_first(&self,input:&Array,axis:usize,rank:usize,widths:&[usize])->Result<Array,ComputeError> {
        if rank!=self.group.rank() || widths.len()!=self.group.size(){return Err(self.invalid(ParallelGatherError::Identity));}
        self.group.gather_model_first(input,self.stream,axis,widths)
    }
    fn slice(&self,input:&Array,axis:usize,start:i32,end:i32)->Result<Array,ComputeError> {
        self.controls(&[size_of::<(&Self,&Array,usize,i32,i32)>(),size_of::<[Vec<i32>;3]>(),
            size_of::<Result<Array,safemlx::error::Exception>>(),size_of::<Result<Array,ComputeError>>()])?;
        let mut begins=self.destination(input.ndim())?;begins.resize(input.ndim(),0);
        let mut ends=self.destination(input.ndim())?;ends.extend_from_slice(input.shape());
        let mut strides=self.destination(input.ndim())?;strides.resize(input.ndim(),1);
        begins[axis]=start;ends[axis]=end;
        self.native(input.try_slice(&begins,&ends,&strides,self.stream))
    }
    fn concatenate(&self,values:&[&Array],axis:usize)->Result<Array,ComputeError> {
        self.controls(&[size_of::<(&Self,&[&Array],usize)>(),size_of::<Result<Array,ComputeError>>(),
            safemlx::ops::concatenate_axis_control_bytes().ok_or_else(||self.invalid(ParallelGatherError::Overflow))?])?;
        let axis=i32::try_from(axis).map_err(|_|self.invalid(ParallelGatherError::Overflow))?;
        self.native(concatenate_axis(values,axis,self.stream))
    }
}

pub(super) fn widths(range:&VocabularyParallelRange,group:&Group)->Result<Vec<usize>,ComputeError> {
    if let Some(funding)=group.model_funding() {
        let parts=[size_of::<eredu_nn::BalancedVocabularyWidths>(),size_of::<eredu_nn::VocabularyRangeError>(),
            size_of::<Result<eredu_nn::BalancedVocabularyWidths,eredu_nn::VocabularyRangeError>>(),
            size_of::<Vec<usize>>(),size_of::<Result<Vec<usize>,ComputeError>>(),
            size_of::<Result<(),std::collections::TryReserveError>>(),size_of::<(&VocabularyParallelRange,&Group)>()];
        let bytes=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .and_then(|n|std::alloc::Layout::array::<usize>(group.size()).ok().and_then(|l|n.checked_add(l.size())))
            .ok_or_else(||group.model_gather_error(ParallelGatherError::Overflow))?;
        funding.reserve_metadata(bytes).map_err(eredu_nn::workspace::WorkspaceMetadataError::Funding)?;
    }
    let plan=range.balanced_peer_widths_plan(group.size(),group.rank())
        .map_err(|cause|group.model_vocabulary_error(cause))?;
    let mut widths=Vec::new();widths.try_reserve_exact(plan.len())
        .map_err(|_|group.model_gather_error(ParallelGatherError::Allocation))?;
    widths.extend(plan.widths());Ok(widths)
}

pub(super) fn run(input:&Array,widths:&[usize],group:&Group,stream:&Stream)->Result<Array,ComputeError> {
    let ops=Operations{group,stream};
    let axis=input.ndim().checked_sub(1).ok_or_else(||ops.invalid(ParallelGatherError::Identity))?;
    run_axis(input,widths,axis,group,stream)
}

pub(super) fn run_axis(input:&Array,widths:&[usize],axis:usize,group:&Group,stream:&Stream)->Result<Array,ComputeError> {
    let ops=Operations{group,stream};
    // Original invocation uses its exact selected source and primitive binding.
    // Ordinary execution keeps the existing bounded-setup and outer contract.
    let setup=if group.has_original_parallel(){None}else{ops.native(group.begin_bounded_setup())?};
    if !group.has_original_parallel() {
        ops.native(group.validate_tensor(eredu_runtime::CommunicationOperation::AllGatherUneven,input,false))?;
        let total=widths.iter().try_fold(0usize,|n,&v|n.checked_add(v));
        let non_axis=input.shape().iter().enumerate().filter(|(i,_)|*i!=axis)
            .try_fold(1usize,|n,(_, &v)|usize::try_from(v).ok().and_then(|v|n.checked_mul(v)));
        let elements=total.zip(non_axis).and_then(|(a,b)|a.checked_mul(b))
            .ok_or_else(||ops.invalid(ParallelGatherError::Overflow))?;
        ops.native(group.validate_expected_output(eredu_runtime::CommunicationOperation::AllGatherUneven,
            input.dtype(),input.ndim(),elements))?;
        if let Some(setup)=&setup {ops.native(setup.check())?;}
    }
    let output=eredu_nn::gather_uneven_axis(&ops,input,axis,group.rank(),widths)?;
    if !group.has_original_parallel() {
        ops.native(group.validate_tensor(eredu_runtime::CommunicationOperation::AllGatherUneven,&output,true))?;
    }
    Ok(output)
}
