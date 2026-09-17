//! Shared validation and destinations for architecture-declared boundary schemas.
use super::*;
use eredu_nn::{workspace::WorkspaceContext,Error};
use std::mem::{size_of,size_of_val};

pub(super) enum Failure { Declaration(ArchitectureBoundaryError), Metadata(Error) }
impl From<ArchitectureBoundaryError> for Failure {fn from(value:ArchitectureBoundaryError)->Self{Self::Declaration(value)}}
impl Failure {
    pub(super) fn ordinary(self)->ArchitectureBoundaryError {match self {
        Self::Declaration(value)=>value,Self::Metadata(_)=>unreachable!("ordinary boundary destinations do not fail metadata admission"),
    }}
    pub(super) fn metadata(self,context:&WorkspaceContext)->Error {match self {
        Self::Declaration(value)=>context.metadata_source(value),Self::Metadata(value)=>value,
    }}
}
#[derive(Clone,Copy)]
pub(super) struct Destination<'a>(pub Option<&'a WorkspaceContext>);
impl Destination<'_> {
    pub(super) fn controls<T>(self)->Result<(),Failure>{
        if let Some(context)=self.0 {
            let parts=[size_of::<T>(),size_of::<Result<T,Failure>>(),size_of::<Self>(),size_of::<Failure>()];
            context.charge_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
                .ok_or_else(||Failure::Metadata(eredu_nn::workspace::WorkspaceMetadataError::Overflow.into()))?)
                .map_err(|cause|Failure::Metadata(cause.into()))?;
        }Ok(())
    }
    pub(super) fn vector<T>(self,count:usize)->Result<Vec<T>,Failure>{
        self.controls::<Vec<T>>()?;
        match self.0 {Some(context)=>context.metadata_vec(count).map_err(Failure::Metadata),None=>Ok(Vec::with_capacity(count))}
    }
    pub(super) fn text(self,value:&str)->Result<String,Failure>{
        self.controls::<String>()?;
        match self.0 {Some(context)=>context.metadata_string(format_args!("{value}")).map_err(Failure::Metadata),None=>Ok(value.to_owned())}
    }
}

pub(super) fn construct(identity:&'static str,primary:BoundaryTensorSpec,
    auxiliary:Vec<BoundaryTensorSpec>,destination:Destination<'_>)->Result<BoundaryWireSchema,Failure>{
    type Specs<'a>=std::iter::Chain<std::iter::Once<&'a BoundaryTensorSpec>,std::slice::Iter<'a,BoundaryTensorSpec>>;
    destination.controls::<(BoundaryWireSchema,usize,&BoundaryTensorSpec,std::iter::Enumerate<Specs<'_>>,
        std::iter::Take<Specs<'_>>,std::slice::Iter<'_,BoundaryTensorDimension>,bool)>()?;
    if identity.trim().is_empty(){return Err(ArchitectureBoundaryError::EmptyIdentity.into());}
    if primary.dtype!=BoundaryTensorDtype::Activation {
        return Err(ArchitectureBoundaryError::InvalidPrimaryDtype{boundary:identity}.into());
    }
    // Roles stay in their original canonical order. Borrow the already-owned
    // prefix instead of constructing another unbounded tree of role aliases.
    for (index,tensor) in std::iter::once(&primary).chain(&auxiliary).enumerate(){
        if tensor.role.trim().is_empty(){return Err(ArchitectureBoundaryError::EmptyTensorRole{boundary:identity}.into());}
        if std::iter::once(&primary).chain(&auxiliary).take(index).any(|prior|prior.role==tensor.role){
            return Err(ArchitectureBoundaryError::DuplicateTensorRole{boundary:identity,role:destination.text(&tensor.role)?}.into());
        }
        if tensor.shape.is_empty(){return Err(ArchitectureBoundaryError::EmptyTensorShape{boundary:identity,role:destination.text(&tensor.role)?}.into());}
        if tensor.shape.iter().any(|dim|matches!(dim,BoundaryTensorDimension::Fixed(value) if *value<=0)){
            return Err(ArchitectureBoundaryError::InvalidTensorDimension{boundary:identity,role:destination.text(&tensor.role)?}.into());
        }
    }
    Ok(BoundaryWireSchema{identity,primary,auxiliary})
}

pub(super) fn resolve(schema:&BoundaryWireSchema,batch:i32,sequences:&[i32],destination:Destination<'_>)
    ->Result<ResolvedBoundaryWireSchema,Failure>{
    destination.controls::<(ResolvedBoundaryWireSchema,&BoundaryWireSchema,&[i32],usize,i32,
        std::slice::Iter<'_,i32>,std::iter::Zip<std::slice::Iter<'_,BoundaryTensorSpec>,std::slice::Iter<'_,i32>>)>()?;
    validate_count(schema.identity,1+schema.auxiliary.len(),sequences.len())?;
    if batch<=0 || sequences.iter().any(|sequence|*sequence<=0){
        return Err(ArchitectureBoundaryError::InvalidInvocationGeometry{boundary:schema.identity,batch_size:batch,
            sequence_length:sequences.iter().copied().find(|value|*value<=0).unwrap_or(0)}.into());
    }
    let primary=resolve_tensor(&schema.primary,batch,sequences[0],destination)?;
    let mut auxiliary=destination.vector(schema.auxiliary.len())?;
    for (tensor,&sequence) in schema.auxiliary.iter().zip(&sequences[1..]){
        auxiliary.push(resolve_tensor(tensor,batch,sequence,destination)?);
    }
    Ok(ResolvedBoundaryWireSchema{identity:schema.identity,primary,auxiliary})
}
fn resolve_tensor(tensor:&BoundaryTensorSpec,batch:i32,sequence:i32,destination:Destination<'_>)
    ->Result<ResolvedBoundaryTensorSpec,Failure>{
    destination.controls::<ResolvedBoundaryTensorSpec>()?;
    let role=destination.text(&tensor.role)?;
    let mut shape=destination.vector(tensor.shape.len())?;
    for dimension in &tensor.shape {shape.push(match dimension{
        BoundaryTensorDimension::Batch=>batch,BoundaryTensorDimension::Sequence=>sequence,
        BoundaryTensorDimension::Fixed(value)=>*value,
    });}
    Ok(ResolvedBoundaryTensorSpec{role,shape,dtype:tensor.dtype})
}
pub(super) fn validate_count(boundary:&'static str,expected:usize,actual:usize)->Result<(),ArchitectureBoundaryError>{
    if actual!=expected {Err(ArchitectureBoundaryError::TensorCount{boundary,expected,actual})}else{Ok(())}
}

#[cfg(test)]
#[path = "boundary_construction/tests.rs"]
mod tests;
