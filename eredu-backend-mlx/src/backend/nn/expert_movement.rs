//! Shared row movement equations for ordinary execution and cold child records.
use eredu_nn::{Tensor,workspace::{WorkspaceContext,WorkspaceTensor}};
use crate::MlxTensor;
use std::mem::{size_of,size_of_val};
pub(crate) trait Operations {
    type Value;
    type Error;
    fn charge(&self,bytes:usize)->Result<(),Self::Error>;
    fn shape<'a>(&self,value:&'a Self::Value)->&'a [i32];
    fn reshape(&self,value:&Self::Value,shape:&[i32])->Result<Self::Value,Self::Error>;
    fn gather(&self,value:&Self::Value,indices:&Self::Value)->Result<Self::Value,Self::Error>;
    fn zeros(&self,prototype:&Self::Value,shape:&[i32])->Result<Self::Value,Self::Error>;
    fn indexed_add(&self,base:&Self::Value,indices:&Self::Value,updates:&Self::Value)->Result<Self::Value,Self::Error>;
}
pub(crate) fn control_bytes<O:Operations>()->Option<usize> {
    let frames=[size_of::<O>(),size_of::<&O>(),size_of::<[&O::Value;3]>(),size_of::<[O::Value;3]>(),
        size_of::<Result<O::Value,O::Error>>(),size_of::<[i32;2]>(),size_of::<i32>(),size_of::<bool>()];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}
pub(crate) fn gather<O:Operations>(ops:&O,value:&O::Value,indices:&O::Value,route_values:bool)->Result<O::Value,O::Error> {
    ops.charge(control_bytes::<O>().expect("fixed movement controls"))?;
    if route_values {
        let flat=ops.reshape(value,&[-1])?;
        let selected=ops.gather(&flat,indices)?;
        ops.reshape(&selected,&[ops.shape(indices)[0],1])
    }else{ops.gather(value,indices)}
}
pub(crate) fn zeros<O:Operations>(ops:&O,prototype:&O::Value,rows:i32)->Result<O::Value,O::Error> {
    ops.charge(control_bytes::<O>().expect("fixed movement controls"))?;
    ops.zeros(prototype,&[rows,ops.shape(prototype)[1]])
}
pub(crate) fn add<O:Operations>(ops:&O,base:&O::Value,indices:&O::Value,updates:&O::Value)->Result<O::Value,O::Error> {
    ops.charge(control_bytes::<O>().expect("fixed movement controls"))?;
    ops.indexed_add(base,indices,updates)
}
pub(crate) struct Native<'a>(pub(crate) &'a safemlx::Stream);
impl Operations for Native<'_> {
    type Value=MlxTensor;type Error=eredu_nn::Error;
    fn charge(&self,_:usize)->Result<(),Self::Error>{Ok(())}
    fn shape<'a>(&self,value:&'a MlxTensor)->&'a [i32]{value.shape()}
    fn reshape(&self,value:&MlxTensor,shape:&[i32])->Result<MlxTensor,Self::Error>{value.reshape(shape,self.0)}
    fn gather(&self,value:&MlxTensor,indices:&MlxTensor)->Result<MlxTensor,Self::Error>{value.take_axis(indices,0,self.0)}
    fn zeros(&self,prototype:&MlxTensor,shape:&[i32])->Result<MlxTensor,Self::Error>{
        safemlx::ops::zeros_dtype(shape,prototype.as_array().dtype(),self.0)
            .map(MlxTensor::from_array).map_err(eredu_nn::Error::backend_retained_source)
    }
    fn indexed_add(&self,base:&MlxTensor,indices:&MlxTensor,updates:&MlxTensor)->Result<MlxTensor,Self::Error>{
        base.as_array().scatter_add(indices.as_array(),updates.as_array(),0,self.0)
            .map(MlxTensor::from_array).map_err(eredu_nn::Error::backend_retained_source)
    }
}
pub(crate) struct Workspace<'a>(pub(crate) &'a WorkspaceContext);
impl Operations for Workspace<'_> {
    type Value=WorkspaceTensor;type Error=eredu_nn::Error;
    fn charge(&self,bytes:usize)->Result<(),Self::Error>{self.0.charge_metadata(bytes).map_err(Into::into)}
    fn shape<'a>(&self,value:&'a WorkspaceTensor)->&'a [i32]{value.shape()}
    fn reshape(&self,value:&WorkspaceTensor,shape:&[i32])->Result<WorkspaceTensor,Self::Error>{value.reshape(shape,self.0)}
    fn gather(&self,value:&WorkspaceTensor,indices:&WorkspaceTensor)->Result<WorkspaceTensor,Self::Error>{value.take_axis(indices,0,self.0)}
    fn zeros(&self,prototype:&WorkspaceTensor,shape:&[i32])->Result<WorkspaceTensor,Self::Error>{
        super::workspace::zero_fill::trace(shape,prototype.layout().as_view(),self.0)
    }
    fn indexed_add(&self,base:&WorkspaceTensor,indices:&WorkspaceTensor,updates:&WorkspaceTensor)->Result<WorkspaceTensor,Self::Error>{
        base.indexed_row_add(indices,updates,self.0)
    }
}
