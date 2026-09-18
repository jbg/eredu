//! Shared ordinary packing and extraction; native and cold adapters supply only
//! the existing tensor operations and paid shape destinations.
use super::*;
use std::alloc::Layout;
pub(crate) trait PackedOperations:Operations {
    fn shape<'a>(&self,value:&'a Self::Value)->&'a [i32];
    fn shape_buffer(&self,count:usize)->Result<Vec<i32>,Self::Error>;
    fn invalid(&self)->Self::Error;
    fn zeros(&self,prototype:&Self::Value,shape:&[i32])->Result<Self::Value,Self::Error>;
    fn reshape(&self,value:&Self::Value,shape:&[i32])->Result<Self::Value,Self::Error>;
    fn update(&self,value:&Self::Value,update:&Self::Value,starts:&[i32],ends:&[i32],strides:&[i32])->Result<Self::Value,Self::Error>;
    fn at(&self,value:&Self::Value,slot:i32)->Result<Self::Value,Self::Error>;
    fn value_buffer(&self,count:usize)->Result<Vec<Self::Value>,Self::Error>;
    fn stack_members(&self,values:&[Self::Value])->Result<Self::Value,Self::Error>;
}
pub(crate) fn controls<O:PackedOperations>(rank:usize)->Option<usize>{
    let fixed=size_of::<(&O,&O::Value,&O::Value,usize,usize)>();
    fixed.checked_add(size_of::<[Vec<i32>;5]>())?
        .checked_add(size_of::<[O::Value;3]>())?
        .checked_add(size_of::<Result<O::Value,O::Error>>())?
        .checked_add(size_of::<[usize;8]>())?
        .checked_add(size_of::<std::slice::Iter<'_,i32>>())?
        .checked_add(Layout::array::<i32>(rank.checked_add(1)?.checked_mul(5)?).ok()?.size())
}
pub(crate) fn pack<O:PackedOperations>(ops:&O,input:&O::Value,slot:usize,world:usize)->Result<O::Value,O::Error>{
    ops.charge(controls::<O>(ops.shape(input).len()).ok_or_else(||ops.invalid())?)?;
    if slot>=world{return Err(ops.invalid());}
    let mut shape=ops.shape_buffer(ops.shape(input).len()+1)?;
    shape.push(i32::try_from(world).map_err(|_|ops.invalid())?);
    shape.extend_from_slice(ops.shape(input));
    let empty=ops.zeros(input,&shape)?;
    if ops.shape(input).contains(&0){return Ok(empty);}
    let mut update_shape=ops.shape_buffer(shape.len())?;
    update_shape.push(1);update_shape.extend_from_slice(ops.shape(input));
    let update=ops.reshape(input,&update_shape)?;
    let mut starts=ops.shape_buffer(shape.len())?;starts.resize(shape.len(),0);
    starts[0]=i32::try_from(slot).map_err(|_|ops.invalid())?;
    let mut ends=ops.shape_buffer(shape.len())?;ends.extend_from_slice(&shape);
    ends[0]=starts[0].checked_add(1).ok_or_else(||ops.invalid())?;
    let mut strides=ops.shape_buffer(shape.len())?;strides.resize(shape.len(),1);
    ops.update(&empty,&update,&starts,&ends,&strides)
}
pub(crate) fn sum_result<O:PackedOperations>(ops:&O,world:&O::Value,representative:usize)->Result<O::Value,O::Error>{
    ops.charge(controls::<O>(ops.shape(world).len()).ok_or_else(||ops.invalid())?)?;
    ops.at(world,i32::try_from(representative).map_err(|_|ops.invalid())?)
}
/// Extraction uses the already selected, host-validated ordered membership.
/// Static slices preserve the values exactly and introduce no dynamic-index
/// validation graph. Both ordinary execution and its cold quote use this worker.
pub(crate) fn gather_controls<O:PackedOperations>(count:usize)->Option<usize>{
    size_of::<(Vec<O::Value>,&[usize],std::slice::Iter<'_,usize>,usize,i32)>()
        .checked_add(Layout::array::<O::Value>(count).ok()?.size())?
        .checked_add(size_of::<Result<Vec<O::Value>,O::Error>>())
}
pub(crate) fn gather_stacked<O:PackedOperations>(ops:&O,world:&O::Value,members:&[usize])->Result<O::Value,O::Error>{
    ops.charge(controls::<O>(ops.shape(world).len())
        .and_then(|n|n.checked_add(gather_controls::<O>(members.len())?))
        .ok_or_else(||ops.invalid())?)?;
    let extent=ops.shape(world).first().copied().ok_or_else(||ops.invalid())?;
    if members.is_empty(){return Err(ops.invalid());}
    let mut selected=ops.value_buffer(members.len())?;
    for &member in members {
        let slot=i32::try_from(member).map_err(|_|ops.invalid())?;
        if slot>=extent{return Err(ops.invalid());}
        selected.push(ops.at(world,slot)?);
    }
    ops.stack_members(&selected)
}
pub(crate) fn flatten<O:PackedOperations>(ops:&O,stacked:&O::Value,input_shape:&[i32],count:usize)->Result<O::Value,O::Error>{
    ops.charge(controls::<O>(input_shape.len()).ok_or_else(||ops.invalid())?)?;
    if input_shape.is_empty(){return ops.reshape(stacked,&[i32::try_from(count).map_err(|_|ops.invalid())?]);}
    let mut shape=ops.shape_buffer(input_shape.len())?;shape.extend_from_slice(input_shape);
    shape[0]=shape[0].checked_mul(i32::try_from(count).map_err(|_|ops.invalid())?).ok_or_else(||ops.invalid())?;
    ops.reshape(stacked,&shape)
}
impl PackedOperations for Native<'_>{
    fn shape<'a>(&self,value:&'a Array)->&'a [i32]{value.shape()}
    fn shape_buffer(&self,count:usize)->Result<Vec<i32>,Self::Error>{Ok(Vec::with_capacity(count))}
    fn invalid(&self)->Self::Error{safemlx::error::Exception::custom("logical world packing geometry is invalid")}
    fn zeros(&self,prototype:&Array,shape:&[i32])->Result<Array,Self::Error>{safemlx::ops::zeros_dtype(shape,prototype.dtype(),self.0)}
    fn reshape(&self,value:&Array,shape:&[i32])->Result<Array,Self::Error>{value.reshape(shape,self.0)}
    fn update(&self,value:&Array,update:&Array,starts:&[i32],ends:&[i32],strides:&[i32])->Result<Array,Self::Error>{value.try_slice_update(update,starts,ends,strides,self.0)}
    fn at(&self,value:&Array,slot:i32)->Result<Array,Self::Error>{
        let shape=value.shape();let rank=shape.len();
        if rank==0||slot<0||slot>=shape[0]{return Err(self.invalid());}
        // Slice + reshape uses the ordinary static index and exposes
        // its actual coordinates to the cold adapter. StaticSlice's existing
        // narrow-axis control source pays these inline/fallback destinations.
        let selected=if rank<=4 {
            let mut starts=[0;4];let mut ends=[0;4];let strides=[1;4];
            ends[..rank].copy_from_slice(shape);starts[0]=slot;ends[0]=slot+1;
            value.try_slice(&starts[..rank],&ends[..rank],&strides[..rank],self.0)?
        } else {
            let mut starts=vec![0;rank];let mut ends=shape.to_vec();let strides=vec![1;rank];
            starts[0]=slot;ends[0]=slot+1;
            value.try_slice(&starts,&ends,&strides,self.0)?
        };
        selected.reshape(&shape[1..],self.0)
    }

    fn value_buffer(&self,count:usize)->Result<Vec<Array>,Self::Error>{
        let mut values=Vec::new();
        values.try_reserve_exact(count).map_err(|_|self.invalid())?;
        if values.capacity()!=count{return Err(self.invalid());}
        Ok(values)
    }
    fn stack_members(&self,values:&[Array])->Result<Array,Self::Error>{safemlx::ops::stack_axis(values,0,self.0)}
}
impl PackedOperations for Workspace<'_>{
    fn shape<'a>(&self,value:&'a WorkspaceTensor)->&'a [i32]{value.shape()}
    fn shape_buffer(&self,count:usize)->Result<Vec<i32>,Self::Error>{self.0.metadata_vec(count)}
    fn invalid(&self)->Self::Error{self.0.metadata_error(format_args!("logical world packing geometry is invalid"))}
    fn zeros(&self,prototype:&WorkspaceTensor,shape:&[i32])->Result<WorkspaceTensor,Self::Error>{
        super::super::workspace::zero_fill::trace(shape,prototype.layout().as_view(),self.0)
    }

    fn reshape(&self,value:&WorkspaceTensor,shape:&[i32])->Result<WorkspaceTensor,Self::Error>{value.reshape(shape,self.0)}
    fn update(&self,value:&WorkspaceTensor,update:&WorkspaceTensor,starts:&[i32],ends:&[i32],strides:&[i32])->Result<WorkspaceTensor,Self::Error>{value.update_static_slice(update,starts,ends,strides,self.0)}
    fn at(&self,value:&WorkspaceTensor,slot:i32)->Result<WorkspaceTensor,Self::Error>{
        if value.shape().is_empty()||slot<0||slot>=value.shape()[0]{return Err(self.invalid());}
        value.narrow_axis(0,slot,slot+1,self.0)?.reshape(&value.shape()[1..],self.0)
    }
    fn value_buffer(&self,count:usize)->Result<Vec<WorkspaceTensor>,Self::Error>{self.0.metadata_vec(count)}
    fn stack_members(&self,values:&[WorkspaceTensor])->Result<WorkspaceTensor,Self::Error>{WorkspaceTensor::stack(values,0,self.0)}
}
