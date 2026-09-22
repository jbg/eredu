//! The selected ordinary discovery and remapping graph, shared with its census.
use super::*;

/// One call to an existing native worker. Composite calls retain that worker's
/// own constructor and completion census; they are not relabelled equations.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Operation {
    Flatten,
    Scalar(i32),
    Less,
    GreaterEqual,
    LogicalAnd,
    LogicalNot,
    CountNonzero,
    CastI32,
    ZerosI32(i32),
    OnesI32(i32),
    Select,
    Histogram(i32),
    Copy,
    Take,
}

pub(crate) trait Visitor {
    type Value;
    type HostI32: ?Sized;
    fn dtype(&self, value: &Self::Value) -> Dtype;
    fn elements(&self, value: &Self::Value) -> usize;
    fn apply(
        &mut self,
        operation: Operation,
        inputs: &[&Self::Value],
    ) -> Result<Self::Value, Error>;
    fn host_i32(&mut self, source: &Self::HostI32) -> Result<Self::Value, Error>;
}

pub(crate) struct Discovery<V> {
    pub(crate) histogram: V,
    pub(crate) invalid: V,
}

fn indexing_size(value: usize) -> Result<i32, Error> {
    i32::try_from(value)
        .map_err(|_| Error::ArchitectureModel("indexed movement exceeds MLX i32 indexing".into()))
}

pub(crate) fn discover<V: Visitor>(
    visitor: &mut V,
    indices: &V::Value,
    upper_bound: usize,
) -> Result<Discovery<V::Value>, Error> {
    let dtype = visitor.dtype(indices);
    if !matches!(
        dtype,
        Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
    ) {
        return Err(AddressableParameterBankError::InvalidSelectionDtype { actual: dtype }.into());
    }
    let upper = indexing_size(upper_bound)?;
    if upper == 0 {
        return Err(Error::ArchitectureModel(
            "indexed movement upper bound must be nonzero".into(),
        ));
    }
    let elements = indexing_size(visitor.elements(indices))?;
    let flat = visitor.apply(Operation::Flatten, &[indices])?;
    let upper = visitor.apply(Operation::Scalar(upper), &[])?;
    let below = visitor.apply(Operation::Less, &[&flat, &upper])?;
    let valid = if matches!(dtype, Dtype::Uint32 | Dtype::Uint64) {
        below
    } else {
        let zero = visitor.apply(Operation::Scalar(0), &[])?;
        let above = visitor.apply(Operation::GreaterEqual, &[&flat, &zero])?;
        visitor.apply(Operation::LogicalAnd, &[&above, &below])?
    };
    let invalid = visitor.apply(Operation::LogicalNot, &[&valid])?;
    let invalid = visitor.apply(Operation::CountNonzero, &[&invalid])?;
    let normalized = if dtype == Dtype::Int32 {
        flat
    } else {
        visitor.apply(Operation::CastI32, &[&flat])?
    };
    let zeros = visitor.apply(Operation::ZerosI32(elements), &[])?;
    let safe = visitor.apply(Operation::Select, &[&valid, &normalized, &zeros])?;
    let ones = visitor.apply(Operation::OnesI32(elements), &[])?;
    let histogram = visitor.apply(
        Operation::Histogram(indexing_size(upper_bound)?),
        &[&ones, &safe],
    )?;
    Ok(Discovery { histogram, invalid })
}

pub(crate) fn remap<V: Visitor>(
    visitor: &mut V,
    indices: &V::Value,
    lookup: &V::HostI32,
) -> Result<V::Value, Error> {
    let lookup = visitor.host_i32(lookup)?;
    let lookup = visitor.apply(Operation::Copy, &[&lookup])?;
    if visitor.dtype(indices) == Dtype::Int32 {
        visitor.apply(Operation::Take, &[&lookup, indices])
    } else {
        let normalized = visitor.apply(Operation::CastI32, &[indices])?;
        visitor.apply(Operation::Take, &[&lookup, &normalized])
    }
}

pub(crate) struct Native<'a>(pub(crate) &'a Stream);
impl Visitor for Native<'_> {
    type Value = Array;
    type HostI32 = [i32];
    fn dtype(&self, value: &Array) -> Dtype {
        value.dtype()
    }
    fn elements(&self, value: &Array) -> usize {
        value.size()
    }
    fn host_i32(&mut self, values: &[i32]) -> Result<Array, Error> {
        Ok(Array::try_from_slice(
            values,
            &[indexing_size(values.len())?],
        )?)
    }
    fn apply(&mut self, operation: Operation, inputs: &[&Array]) -> Result<Array, Error> {
        let stream = self.0;
        Ok(match operation {
            Operation::Flatten => inputs[0].reshape(&[-1], stream)?,
            Operation::Scalar(value) => Array::try_from_int(value)?,
            Operation::Less => inputs[0].lt(inputs[1], stream)?,
            Operation::GreaterEqual => inputs[0].ge(inputs[1], stream)?,
            Operation::LogicalAnd => inputs[0].logical_and(inputs[1], stream)?,
            Operation::LogicalNot => inputs[0].logical_not(stream)?,
            Operation::CountNonzero => {
                crate::backend::compaction::count_nonzero(inputs[0], stream)?
            }
            Operation::CastI32 => inputs[0].as_dtype(Dtype::Int32, stream)?,
            Operation::ZerosI32(elements) => Array::zeros::<i32>(&[elements], stream)?,
            Operation::OnesI32(elements) => Array::ones::<i32>(&[elements], stream)?,
            Operation::Select => r#where(inputs[0], inputs[1], inputs[2], stream)?,
            Operation::Histogram(bins) => segment_sum(inputs[0], inputs[1], bins, 0, stream)?,
            Operation::Copy => inputs[0].copy(stream)?,
            Operation::Take => inputs[0].take(inputs[1], stream)?,
        })
    }
}

/// Fixed Rust call frames of this same traversal. Native primitive/seed controls
/// and the caller's retained host payloads are quoted by their own producers.
pub(crate) fn traversal_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<Native<'_>>(),
        size_of::<(&mut Native<'_>, &Array, usize)>(),
        size_of::<Discovery<Array>>(),
        size_of::<Result<Discovery<Array>, Error>>(),
        size_of::<(&mut Native<'_>, &Array, &[i32])>(),
        size_of::<Result<Array, Error>>(),
        size_of::<Array>() * 11,
        size_of::<Dtype>(),
        size_of::<i32>() * 3,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
pub(crate) fn operation_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<(&mut Native<'_>, Operation, &[&Array])>(),
        size_of::<&Stream>(),
        size_of::<Result<Array, Error>>(),
        size_of::<Result<Array, safemlx::error::Exception>>(),
        size_of::<Array>(),
        size_of::<[&Array; 3]>(),
        size_of::<[i32; 1]>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
