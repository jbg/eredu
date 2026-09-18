//! Fixed host payload shared by numerical and metadata tensor initialization.

use crate::Error;

/// Invalid geometry of the fixed F32 initialization worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum F32InitializationError {
    /// Every tensor axis must have a nonnegative extent.
    #[error("F32 initialization axis {axis} has negative extent {extent}")]
    NegativeExtent {
        /// Invalid axis.
        axis: usize,
        /// Supplied negative extent.
        extent: i32,
    },
    /// Product of the tensor axes exceeds the host index domain.
    #[error("F32 initialization element count overflows")]
    ElementCountOverflow,
    /// The actual F32 allocation exceeds the host allocation/accounting domain.
    #[error("F32 initialization buffer capacity overflows")]
    BufferCapacityOverflow,
}

/// Borrowed, allocation-free geometry for one fixed F32 initialization buffer.
///
/// This describes the shared worker's numerical Vec payload, not the native
/// seed/copy, shape/command metadata, allocator overhead or arbitrary callback
/// resources. The scalar expression must obey `Tensor::from_f32_fn`'s contract.
/// It grants no allocation permission or complete inference bound.
#[derive(Debug, Clone, Copy)]
pub struct F32InitializationPlan<'a> {
    shape: &'a [i32],
    elements: usize,
    bytes: u64,
}

impl<'a> F32InitializationPlan<'a> {
    /// Validates all axes and checked storage geometry before any allocation.
    /// An empty shape is a scalar; any zero axis gives an empty buffer.
    pub fn new(shape: &'a [i32]) -> Result<Self, F32InitializationError> {
        for (axis, &extent) in shape.iter().enumerate() {
            if extent < 0 {
                return Err(F32InitializationError::NegativeExtent { axis, extent });
            }
        }
        let elements = if shape.contains(&0) {
            0
        } else {
            shape.iter().try_fold(1usize, |count, &extent| {
                count
                    .checked_mul(extent as usize)
                    .ok_or(F32InitializationError::ElementCountOverflow)
            })?
        };
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(F32InitializationError::BufferCapacityOverflow)?;
        Ok(Self {
            shape,
            elements,
            bytes,
        })
    }

    /// Original validated shape, borrowed without another shape allocation.
    pub fn shape(&self) -> &'a [i32] {
        self.shape
    }
    /// Exact number of scalar invocations and initialized elements.
    pub fn elements(&self) -> usize {
        self.elements
    }
    /// One fixed F32 vector's requested payload capacity.
    pub fn host_buffer_bytes(&self) -> u64 {
        self.bytes
    }

    /// Fixed shared initialization frames while its host buffer and realization
    /// result coexist. The F32 payload is separate, as are the caller's scalar
    /// expression/captures and the selected native constructor's storage.
    /// This describes the concrete shared worker and grants no allocation.
    pub fn worker_control_bytes<T>(&self) -> Option<usize> {
        use std::mem::{size_of,size_of_val};
        let controls=[
            size_of::<Self>(),size_of::<Result<Self,F32InitializationError>>(),
            size_of::<Vec<f32>>(),size_of::<(&Self,&[i32],&[f32])>(),
            size_of::<std::ops::Range<usize>>(),size_of::<usize>(),size_of::<f32>(),
            size_of::<T>(),size_of::<Result<T,Error>>(),size_of::<F32InitializationError>(),
        ];
        controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
    }

    fn values(&self, mut scalar: impl FnMut(usize) -> f32) -> Vec<f32> {
        // The selected/pinned RawVec worker requests Layout::array::<f32>(n)
        // and records n. Exactly n pushes follow, with no grow/collect/shrink
        // pass and no second numerical vector. Native facts depend on this
        // concrete allocation program, not merely the resulting Vec length.
        let mut values = Vec::with_capacity(self.elements);
        #[cfg(test)]
        tests::allocated(values.capacity(), values.as_ptr());
        for index in 0..self.elements {
            values.push(scalar(index));
        }
        values
    }
}

pub(crate) fn initialize<T>(
    shape: &[i32],
    scalar: impl FnMut(usize) -> f32,
    realize: impl FnOnce(&[f32]) -> Result<T, Error>,
) -> Result<T, Error> {
    let plan = F32InitializationPlan::new(shape).map_err(Error::backend_retained_source)?;
    let values = plan.values(scalar);
    // The exact host buffer remains live throughout the native seed/copy call,
    // then retires on either result. The original backend error is unchanged.
    realize(&values)
}

#[cfg(test)]
mod tests;
