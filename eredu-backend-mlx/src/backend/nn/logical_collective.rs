//! One arithmetic worker for ordinary and source-funded logical pair collectives.
//! The exact selected group owns transport and ordered peer identity.
use eredu_nn::{
    workspace::{WorkspaceContext, WorkspaceTensor},
    Tensor,
};
use safemlx::{Array, Stream};
use std::mem::{size_of, size_of_val};

pub(crate) trait Operations {
    type Value;
    type Error;
    fn charge(&self, bytes: usize) -> Result<(), Self::Error>;
    fn zero(&self, prototype: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn multiply(&self, left: &Self::Value, right: &Self::Value)
        -> Result<Self::Value, Self::Error>;
    fn add(&self, left: &Self::Value, right: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn stack(&self, first: &Self::Value, second: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn flatten_peers(
        &self,
        stacked: &Self::Value,
        prototype: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;
}
pub(crate) fn control_bytes<O: Operations>() -> Option<usize> {
    let frames = [
        size_of::<O>(),
        size_of::<&O>(),
        size_of::<[&O::Value; 3]>(),
        size_of::<[O::Value; 2]>(),
        size_of::<Result<O::Value, O::Error>>(),
        size_of::<bool>(),
        size_of::<usize>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
/// Keep the ordinary dependency arithmetic, including NaN/signed-zero behavior.
/// Both sent and received are real outputs of the accepted paired exchange.
pub(crate) fn peer<O: Operations>(
    ops: &O,
    sent: &O::Value,
    received: &O::Value,
    zero: &O::Value,
) -> Result<O::Value, O::Error> {
    ops.charge(control_bytes::<O>().expect("fixed logical arithmetic frames"))?;
    let dependency = ops.multiply(sent, zero)?;
    ops.add(received, &dependency)
}
pub(crate) fn sum<O: Operations>(
    ops: &O,
    input: &O::Value,
    peer: &O::Value,
) -> Result<O::Value, O::Error> {
    ops.charge(control_bytes::<O>().expect("fixed logical arithmetic frames"))?;
    ops.add(input, peer)
}
pub(crate) fn stacked<O: Operations>(
    ops: &O,
    input: &O::Value,
    peer: &O::Value,
    first: bool,
) -> Result<O::Value, O::Error> {
    ops.charge(control_bytes::<O>().expect("fixed logical arithmetic frames"))?;
    if first {
        ops.stack(input, peer)
    } else {
        ops.stack(peer, input)
    }
}
pub(crate) fn gather<O: Operations>(
    ops: &O,
    input: &O::Value,
    peer: &O::Value,
    first: bool,
) -> Result<O::Value, O::Error> {
    ops.charge(control_bytes::<O>().expect("fixed logical arithmetic frames"))?;
    let stacked = stacked(ops, input, peer, first)?;
    ops.flatten_peers(&stacked, input)
}
/// Native worker frames are prepaid by each original caller before entry.
/// Individual tensor constructors use that caller's actual Graph/Record banks.
pub(crate) struct Native<'a>(pub(crate) &'a Stream);
impl Operations for Native<'_> {
    type Value = Array;
    type Error = safemlx::error::Exception;
    fn charge(&self, _: usize) -> Result<(), Self::Error> {
        Ok(())
    }
    fn zero(&self, prototype: &Array) -> Result<Array, Self::Error> {
        safemlx::ops::zeros_dtype(&[], prototype.dtype(), self.0)
    }
    fn multiply(&self, left: &Array, right: &Array) -> Result<Array, Self::Error> {
        left.multiply(right, self.0)
    }
    fn add(&self, left: &Array, right: &Array) -> Result<Array, Self::Error> {
        left.add(right, self.0)
    }
    fn stack(&self, first: &Array, second: &Array) -> Result<Array, Self::Error> {
        safemlx::ops::stack_axis(&[first, second], 0, self.0)
    }
    fn flatten_peers(&self, stacked: &Array, prototype: &Array) -> Result<Array, Self::Error> {
        if prototype.ndim() == 0 {
            return Ok(stacked.clone());
        }
        let mut shape = prototype.shape().to_vec();
        shape[0] = shape[0].checked_mul(2).ok_or_else(|| {
            safemlx::error::Exception::custom("logical all-gather shape exceeds i32")
        })?;
        stacked.reshape(&shape, self.0)
    }
}
pub(crate) struct Workspace<'a>(pub(crate) &'a WorkspaceContext);
impl Operations for Workspace<'_> {
    type Value = WorkspaceTensor;
    type Error = eredu_nn::Error;
    fn charge(&self, bytes: usize) -> Result<(), Self::Error> {
        self.0.charge_metadata(bytes).map_err(Into::into)
    }
    fn zero(&self, prototype: &WorkspaceTensor) -> Result<WorkspaceTensor, Self::Error> {
        super::workspace::zero_fill::trace(&[],prototype.layout().as_view(),self.0)
    }

    fn multiply(
        &self,
        left: &WorkspaceTensor,
        right: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Self::Error> {
        left.multiply(right, self.0)
    }
    fn add(
        &self,
        left: &WorkspaceTensor,
        right: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Self::Error> {
        left.add(right, self.0)
    }
    fn stack(
        &self,
        first: &WorkspaceTensor,
        second: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Self::Error> {
        WorkspaceTensor::stack(&[first.clone(), second.clone()], 0, self.0)
    }
    fn flatten_peers(
        &self,
        stacked: &WorkspaceTensor,
        prototype: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Self::Error> {
        if prototype.shape().is_empty() {
            return Ok(stacked.clone());
        }
        let mut shape = self.0.metadata_vec(prototype.shape().len())?;
        shape.extend_from_slice(prototype.shape());
        shape[0] = shape[0].checked_mul(2).ok_or_else(|| {
            self.0
                .metadata_error(format_args!("logical all-gather shape exceeds i32"))
        })?;
        stacked.reshape(&shape, self.0)
    }
}

pub(crate) mod packed;

pub(crate) mod blocks;

pub(crate) mod axis;

pub(crate) mod variable;

pub(crate) mod routed;
