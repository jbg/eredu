//! The ordinary boundary byte equations and their pure workspace trace.
//! Canonical header content remains owned by the neutral protocol writer.
use eredu_nn::Tensor;
use eredu_nn::workspace::{WorkspaceTensor,WorkspaceContext,WorkspaceDtype,WorkspaceOperationKind};
use safemlx::{Array,Stream,Dtype};
use std::mem::{size_of,size_of_val};

pub(crate) trait Operations {
    type Value;
    type Error;
    fn charge(&self,bytes:usize)->Result<(),Self::Error>;
    fn bytes(&self,value:&Self::Value)->Result<usize,Self::Error>;
    fn as_bytes(&self,value:&Self::Value)->Result<Self::Value,Self::Error>;
    fn flatten(&self,value:&Self::Value)->Result<Self::Value,Self::Error>;
    fn concatenate(&self,header:&Self::Value,bytes:&Self::Value)->Result<Self::Value,Self::Error>;
    fn header(&self,value:&Self::Value,end:i32)->Result<Self::Value,Self::Error>;
    fn payload(&self,value:&Self::Value,start:i32)->Result<Self::Value,Self::Error>;
    fn materialize_bytes(&self,value:&Self::Value)->Result<Self::Value,Self::Error>;
    fn reinterpret(&self,value:&Self::Value,prototype:&Self::Value)->Result<Self::Value,Self::Error>;
    fn reshape(&self,value:&Self::Value,prototype:&Self::Value)->Result<Self::Value,Self::Error>;
}
/// Generic worker frames, in addition to the existing individual constructors.
/// Original native callers pay these through their retained source before entry.
pub(crate) fn control_bytes<O:Operations>()->Option<usize>{
    let frames=[size_of::<O>(),size_of::<&O>(),size_of::<O::Value>(),size_of::<O::Value>(),
        size_of::<O::Value>(),size_of::<O::Error>(),size_of::<Result<O::Value,O::Error>>(),
        size_of::<(&O::Value,&O::Value)>(),size_of::<i32>(),size_of::<usize>(),size_of::<bool>(),
        size_of::<[i32;1]>()*3,size_of::<&[i32]>()*3,
        size_of::<(&Array,&[i32],&[i32],&[i32],&Stream)>(),size_of::<Result<Array,safemlx::error::Exception>>()];
    frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
}
pub(crate) fn encode<O:Operations>(ops:&O,input:&O::Value,header:&O::Value)
    ->Result<O::Value,O::Error>{
    // This fixed generic sum cannot overflow for materialized Rust types.
    ops.charge(control_bytes::<O>().expect("fixed boundary worker frames"))?;
    let bytes=ops.as_bytes(input)?;
    let bytes=ops.flatten(&bytes)?;
    ops.concatenate(header,&bytes)
}
pub(crate) fn decode<O:Operations>(ops:&O,received:&O::Value,header_len:i32,prototype:&O::Value)
    ->Result<O::Value,O::Error>{
    ops.charge(control_bytes::<O>().expect("fixed boundary worker frames"))?;
    let bytes=ops.payload(received,header_len)?;
    // A byte view can carry an offset which typed kernels round down. Exactly
    // the ordinary zero-add materializes alignment before reinterpretation.
    let bytes=if (header_len as usize).is_multiple_of(ops.bytes(prototype)?) {
        bytes
    } else {ops.materialize_bytes(&bytes)?};
    let typed=ops.reinterpret(&bytes,prototype)?;
    ops.reshape(&typed,prototype)
}

/// Preserve ordinary header-first graph construction and the same payload
/// alignment/interpretation equations in both native and workspace execution.
pub(crate) fn split<O:Operations>(ops:&O,received:&O::Value,header_len:i32,prototype:&O::Value)
    ->Result<(O::Value,O::Value),O::Error>{
    ops.charge(std::mem::size_of::<(O::Value,O::Value)>())?;
    let header=ops.header(received,header_len)?;
    let payload=decode(ops,received,header_len,prototype)?;
    Ok((header,payload))
}

/// Native constructors are shared by ordinary and source-funded invocations.
/// The enclosing original producer supplies its actual Graph/Record/Q/H banks.
pub(crate) struct Native<'a>(pub(crate) &'a Stream);
impl Operations for Native<'_> {
    type Value=Array;type Error=safemlx::error::Exception;
    fn charge(&self,_:usize)->Result<(),Self::Error>{Ok(())}
    fn bytes(&self,value:&Array)->Result<usize,Self::Error>{Ok(value.item_size())}
    fn as_bytes(&self,value:&Array)->Result<Array,Self::Error>{value.view_dtype(Dtype::Uint8,self.0)}
    fn flatten(&self,value:&Array)->Result<Array,Self::Error>{value.reshape(&[-1],self.0)}
    fn concatenate(&self,header:&Array,bytes:&Array)->Result<Array,Self::Error>{safemlx::ops::concatenate(&[header,bytes],self.0)}
    fn header(&self,value:&Array,end:i32)->Result<Array,Self::Error>{value.try_slice(&[0],&[end],&[1],self.0)}
    fn payload(&self,value:&Array,start:i32)->Result<Array,Self::Error>{value.try_slice(&[start],&[value.shape()[0]],&[1],self.0)}
    fn materialize_bytes(&self,value:&Array)->Result<Array,Self::Error>{value.add(Array::try_from_slice(&[0u8],&[])?,self.0)}
    fn reinterpret(&self,value:&Array,prototype:&Array)->Result<Array,Self::Error>{value.view_dtype(prototype.dtype(),self.0)}
    fn reshape(&self,value:&Array,prototype:&Array)->Result<Array,Self::Error>{value.reshape(prototype.shape(),self.0)}
}

pub(crate) struct Workspace<'a>(pub(crate) &'a WorkspaceContext);
impl Workspace<'_> {
    fn dtype(&self,value:&WorkspaceTensor)->Result<super::workspace::byte_view::Dtype,eredu_nn::Error>{
        super::workspace::byte_view::Dtype::from_layout(value.layout().as_view())
            .ok_or_else(||self.0.metadata_error(format_args!("boundary byte view lacks exact scalar representation")))
    }
}
impl Operations for Workspace<'_> {
    type Value=WorkspaceTensor;type Error=eredu_nn::Error;
    fn charge(&self,bytes:usize)->Result<(),Self::Error>{self.0.charge_metadata(bytes).map_err(Into::into)}
    fn bytes(&self,value:&WorkspaceTensor)->Result<usize,Self::Error>{Ok(self.dtype(value)?.bytes() as usize)}
    fn as_bytes(&self,value:&WorkspaceTensor)->Result<WorkspaceTensor,Self::Error>{
        super::workspace::byte_view::trace(value,super::workspace::byte_view::Dtype::U8,self.0)
    }
    fn flatten(&self,value:&WorkspaceTensor)->Result<WorkspaceTensor,Self::Error>{value.reshape(&[-1],self.0)}
    fn concatenate(&self,header:&WorkspaceTensor,bytes:&WorkspaceTensor)->Result<WorkspaceTensor,Self::Error>{
        if header.shape().len()!=1 || bytes.shape().len()!=1 || header.layout().dtype()!=WorkspaceDtype::Uint8
            || bytes.layout().dtype()!=WorkspaceDtype::Uint8 {
            return Err(self.0.metadata_error(format_args!("boundary byte concatenation requires rank-one U8 inputs")));
        }
        let len=header.shape()[0].checked_add(bytes.shape()[0])
            .ok_or_else(||self.0.metadata_error(format_args!("boundary frame length overflow")))?;
        let mut outputs=self.0.metadata_vec(1)?;
        outputs.push(self.0.layout(&[len],WorkspaceDtype::Uint8)?);
        let mut values=self.0.execute(WorkspaceOperationKind::Concatenate,&[header,bytes],outputs)?;
        Ok(values.pop().expect("one declared frame"))
    }
    fn header(&self,value:&WorkspaceTensor,end:i32)->Result<WorkspaceTensor,Self::Error>{
        value.narrow_axis(0,0,end,self.0)
    }
    fn payload(&self,value:&WorkspaceTensor,start:i32)->Result<WorkspaceTensor,Self::Error>{
        if value.shape().len()!=1{return Err(self.0.metadata_error(format_args!("boundary frame must have one byte dimension")));}
        value.narrow_axis(0,start,value.shape()[0],self.0)
    }
    fn materialize_bytes(&self,value:&WorkspaceTensor)->Result<WorkspaceTensor,Self::Error>{
        let mut outputs=self.0.metadata_vec(1)?;
        outputs.push(self.0.layout(&[],WorkspaceDtype::Uint8)?);
        // Exact counterpart of Native's borrowed eager [0u8] seed. Generic
        // Initialize does not establish this construction or value source.
        let mut values=self.0.execute(WorkspaceOperationKind::Elementwise("scalar_u8"),&[],outputs)?;
        value.add(&values.pop().expect("one declared byte scalar"),self.0)
    }
    fn reinterpret(&self,value:&WorkspaceTensor,prototype:&WorkspaceTensor)->Result<WorkspaceTensor,Self::Error>{
        super::workspace::byte_view::trace(value,self.dtype(prototype)?,self.0)
    }
    fn reshape(&self,value:&WorkspaceTensor,prototype:&WorkspaceTensor)->Result<WorkspaceTensor,Self::Error>{
        value.reshape(prototype.shape(),self.0)
    }
}
