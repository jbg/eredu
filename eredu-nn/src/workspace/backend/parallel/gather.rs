//! Workspace primitives for the same native uneven-axis Gather worker.
use super::*;
use crate::{ParallelGatherError, ParallelGatherOperations};
pub(super) struct Operations<'a>(pub(super) &'a WorkspaceContext);
impl ParallelGatherOperations for Operations<'_> {
    type Value = WorkspaceTensor;
    fn shape<'a>(&self,value:&'a WorkspaceTensor)->&'a [i32] { value.shape() }
    fn charge(&self,bytes:usize)->Result<(),Error> { self.0.charge_metadata(bytes).map_err(Into::into) }
    fn invalid(&self,cause:ParallelGatherError)->Error { self.0.metadata_error(format_args!("{cause}")) }
    fn zeros(&self,input:&WorkspaceTensor,shape:&[i32])->Result<WorkspaceTensor,Error> {
        WorkspaceTensor::zeros_from_prototype(shape,input.layout().as_view(),self.0)
    }
    fn gather_first(&self,input:&WorkspaceTensor,axis:usize,rank:usize,widths:&[usize])->Result<WorkspaceTensor,Error> {
        let mut shape=self.destination(input.shape().len())?;shape.extend_from_slice(input.shape());
        shape[0]=i32::try_from(widths.len()).ok().and_then(|n|n.checked_mul(shape[0]))
            .ok_or_else(||self.invalid(ParallelGatherError::Overflow))?;
        let mut peer_widths=self.destination(widths.len())?;peer_widths.extend_from_slice(widths);
        WorkspaceTensor::operation(WorkspaceOperationKind::Collective(WorkspaceCollective::GatherFirstAxis{
            axis,rank,peer_widths}),&[input],&shape,input.layout.dtype,self.0)
    }
    fn slice(&self,input:&WorkspaceTensor,axis:usize,start:i32,end:i32)->Result<WorkspaceTensor,Error> {
        // The native gather adapter calls its exact rank-preserving Slice.
        // Retain those coordinates instead of erasing them into generic Index.
        input.narrow_axis(axis,start,end,self.0)
    }
    fn concatenate(&self,values:&[&WorkspaceTensor],axis:usize)->Result<WorkspaceTensor,Error> {
        let first=values.first().ok_or_else(||self.invalid(ParallelGatherError::Identity))?;
        let mut shape=self.destination(first.shape().len())?;shape.extend_from_slice(first.shape());
        if axis>=shape.len(){return Err(self.invalid(ParallelGatherError::Identity));}
        shape[axis]=0;
        for value in values {
            if value.shape().len()!=shape.len() || value.layout.dtype!=first.layout.dtype ||
                value.shape().iter().enumerate().any(|(i,&n)|i!=axis&&n!=shape[i]) {
                return Err(self.invalid(ParallelGatherError::Identity));
            }
            shape[axis]=shape[axis].checked_add(value.shape()[axis])
                .ok_or_else(||self.invalid(ParallelGatherError::Overflow))?;
        }
        WorkspaceTensor::operation(WorkspaceOperationKind::Concatenate,values,&shape,first.layout.dtype,self.0)
    }
}
