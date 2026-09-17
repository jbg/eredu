//! One rank-ordered uneven Gather equation for native and workspace execution.
use crate::Error;
use std::{alloc::Layout, mem::{size_of, size_of_val}};

/// Fixed failure of the shared axis Gather geometry or destination producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ParallelGatherError {
    /// Rank, peer widths or input shape do not describe the same invocation.
    #[error("parallel gather rank, widths or input shape differ")]
    Identity,
    /// An extent or a concrete host destination is not representable.
    #[error("parallel gather extent or destination overflow")]
    Overflow,
    /// A pre-admitted host destination could not be allocated.
    #[error("parallel gather destination allocation failed")]
    Allocation,
    /// The actual collective or view returned an unexpected shape.
    #[error("parallel gather output shape differs from its equation")]
    Output,
}

/// Native tensor primitives and their workspace counterparts for the same
/// wrapper equation. Implementations own source admission, exact physical
/// representation, error custody and the primitive's native completion.
pub trait ParallelGatherOperations {
    /// Tensor owner used by this execution of the equation.
    type Value;
    /// Borrows the actual logical shape.
    fn shape<'a>(&self, value: &'a Self::Value) -> &'a [i32];
    /// Charges the next concrete host destination before it is allocated.
    fn charge(&self, bytes: usize) -> Result<(), Error>;
    /// Retains a fixed failure using the caller's existing diagnostic account.
    fn invalid(&self, cause: ParallelGatherError) -> Error;
    /// Creates zeros with the input's physical dtype and the supplied shape.
    fn zeros(&self, input: &Self::Value, shape: &[i32]) -> Result<Self::Value, Error>;
    /// Equal-sized native Gather along the first axis, in peer rank order.
    fn gather_first(&self, input: &Self::Value, axis: usize, rank: usize, widths: &[usize]) -> Result<Self::Value, Error>;
    /// Static view preserving rank, with a half-open selected-axis interval.
    fn slice(&self, input: &Self::Value, axis: usize, start: i32, end: i32) -> Result<Self::Value, Error>;
    /// Concatenates borrowed values without cloning their native owners.
    fn concatenate(&self, values: &[&Self::Value], axis: usize) -> Result<Self::Value, Error>;

    /// Reserves one exact Rust vector destination before allocation. Retired
    /// temporaries never refund the cumulative account supplied by `charge`.
    fn destination<T>(&self, count: usize) -> Result<Vec<T>, Error> {
        let controls = [size_of::<Vec<T>>(), size_of::<Result<Vec<T>, Error>>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<Layout>(), size_of::<usize>(), size_of::<&Self>()];
        let bytes = Layout::array::<T>(count).ok().map(|l| l.size())
            .and_then(|n| n.checked_add(size_of_val(&controls)))
            .and_then(|n| controls.into_iter().try_fold(n, usize::checked_add))
            .ok_or_else(|| self.invalid(ParallelGatherError::Overflow))?;
        self.charge(bytes)?;
        let mut values = Vec::new();
        values.try_reserve_exact(count).map_err(|_| self.invalid(ParallelGatherError::Allocation))?;
        Ok(values)
    }
}

/// Pads each local shard to the same width, gathers along the native first
/// axis, then selects each rank's requested columns before one final concat.
/// The selected outputs are the same rank-ordered values as pad/gather/split/
/// trim. Unused split tails are never materialized by either implementation.
pub fn gather_uneven_axis<O: ParallelGatherOperations>(
    ops: &O, input: &O::Value, axis: usize, rank: usize, widths: &[usize],
) -> Result<O::Value, Error> {
    let controls = [size_of::<(&O, &O::Value, &[usize])>(), size_of::<[usize; 7]>(),
        size_of::<[i32; 6]>(), size_of::<[bool; 4]>(), size_of::<ParallelGatherError>(),
        size_of::<Option<O::Value>>(), size_of::<O::Value>() * 3,
        size_of::<Vec<O::Value>>(), size_of::<Vec<&O::Value>>(), size_of::<Vec<i32>>(),
        size_of::<[&O::Value; 2]>(), size_of::<Result<O::Value, Error>>()];
    let bytes = controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or_else(|| ops.invalid(ParallelGatherError::Overflow))?;
    ops.charge(bytes)?;
    let shape = ops.shape(input);
    if shape.is_empty() || axis >= shape.len() || rank >= widths.len()
        || shape.iter().any(|&d| d < 0)
        || usize::try_from(shape[axis]).ok() != widths.get(rank).copied() {
        return Err(ops.invalid(ParallelGatherError::Identity));
    }
    let max = widths.iter().copied().max().filter(|&n| n != 0)
        .ok_or_else(|| ops.invalid(ParallelGatherError::Identity))?;
    let max_i32 = i32::try_from(max).map_err(|_| ops.invalid(ParallelGatherError::Overflow))?;
    let output_width = widths.iter().try_fold(0usize, |n, &v| n.checked_add(v))
        .and_then(|n| i32::try_from(n).ok())
        .ok_or_else(|| ops.invalid(ParallelGatherError::Overflow))?;
    let peers = i32::try_from(widths.len()).map_err(|_| ops.invalid(ParallelGatherError::Overflow))?;
    let height = if axis == 0 { max_i32 } else { shape[0] };
    let gathered_height = height.checked_mul(peers)
        .ok_or_else(|| ops.invalid(ParallelGatherError::Overflow))?;
    let padded = if widths[rank] != max {
        let mut padding_shape = ops.destination(shape.len())?;
        padding_shape.extend_from_slice(shape);
        padding_shape[axis] = max_i32 - shape[axis];
        let zeros = ops.zeros(input, &padding_shape)?;
        Some(ops.concatenate(&[input, &zeros], axis)?)
    } else { None };
    let gathered = ops.gather_first(padded.as_ref().unwrap_or(input), axis, rank, widths)?;
    if ops.shape(&gathered).len() != shape.len()
        || ops.shape(&gathered).iter().enumerate().any(|(i, &n)| n !=
            if i == 0 { gathered_height } else if i == axis { max_i32 } else { shape[i] }) {
        return Err(ops.invalid(ParallelGatherError::Output));
    }
    let mut shards = ops.destination(widths.len())?;
    for (peer, &width) in widths.iter().enumerate() {
        let start = i32::try_from(peer).ok().and_then(|n| n.checked_mul(height))
            .ok_or_else(|| ops.invalid(ParallelGatherError::Overflow))?;
        let width = i32::try_from(width).map_err(|_| ops.invalid(ParallelGatherError::Overflow))?;
        let end = start.checked_add(if axis == 0 { width } else { height })
            .ok_or_else(|| ops.invalid(ParallelGatherError::Overflow))?;
        let rank_value = ops.slice(&gathered, 0, start, end)?;
        shards.push(if axis != 0 && width != max_i32 {
            ops.slice(&rank_value, axis, 0, width)?
        } else { rank_value });
    }
    let mut refs = ops.destination(shards.len())?;
    refs.extend(shards.iter());
    let output = ops.concatenate(&refs, axis)?;
    if ops.shape(&output).len() != shape.len() || ops.shape(&output).iter().enumerate()
        .any(|(i, &n)| n != if i == axis { output_width } else { shape[i] }) {
        return Err(ops.invalid(ParallelGatherError::Output));
    }
    Ok(output)
}
