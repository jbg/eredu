//! Logical host reads without a tensor-sized compaction or intermediate vector.
use super::*;

pub(crate) struct LayoutSpan {
    pub size: usize,
    pub minimum: isize,
    pub maximum: isize,
    pub contiguous: bool,
}

/// Validates every arithmetic operation used by host pointer displacement.
/// Native tensor construction guarantees initialized storage at each logical
/// coordinate; the safe as_strided wrapper separately proves its source bounds.
pub(crate) fn checked_layout(
    shape: &[i32],
    strides: &[i64],
    element_bytes: usize,
) -> Result<LayoutSpan, AsSliceError> {
    if shape.len() != strides.len() || shape.iter().any(|&n| n < 0) || element_bytes == 0 {
        return Err(AsSliceError::InvalidLayout);
    }
    let mut span = LayoutSpan {
        size: 0,
        minimum: 0,
        maximum: 0,
        contiguous: true,
    };
    if shape.contains(&0) {
        return Ok(span);
    }
    span.size = 1;
    for (&extent, &stride) in shape.iter().zip(strides).rev() {
        span.contiguous &= extent <= 1 || i128::from(stride) == span.size as i128;
        span.size = span
            .size
            .checked_mul(extent as usize)
            .ok_or(AsSliceError::TooLarge)?;
        let delta = i128::from(stride) * i128::from(extent - 1);
        let delta = isize::try_from(delta).map_err(|_| AsSliceError::TooLarge)?;
        let limit = if delta < 0 {
            &mut span.minimum
        } else {
            &mut span.maximum
        };
        *limit = limit.checked_add(delta).ok_or(AsSliceError::TooLarge)?;
    }
    let limit = isize::MAX as u128;
    if span.size as u128 * element_bytes as u128 > limit
        || (span.maximum as i128 - span.minimum as i128 + 1) as u128 * element_bytes as u128 > limit
    {
        return Err(AsSliceError::TooLarge);
    }
    Ok(span)
}

pub(super) struct HostValues<'a, T> {
    data: *const T,
    shape: &'a [i32],
    strides: &'a [i64],
    next: usize,
    end: usize,
    span: LayoutSpan,
}
impl<T: Copy> HostValues<'_, T> {
    fn value_at(&self, mut flat: usize) -> T {
        let offset = if self.span.contiguous {
            flat as isize
        } else {
            let mut offset = 0_isize;
            for (&extent, &stride) in self.shape.iter().zip(self.strides).rev() {
                let coordinate = flat % extent as usize;
                flat /= extent as usize;
                // checked_layout proves each nonzero displacement and the sum
                // of all positive/negative displacements fit pointer arithmetic.
                offset += coordinate as isize * stride as isize;
            }
            offset
        };
        // SAFETY: evaluation retains the native backing and data points at its
        // logical origin, which may be inside the allocation for reversed views.
        // The validated signed displacement selects an initialized logical
        // element. No reference to potentially unaligned foreign data is made.
        unsafe { self.data.offset(offset).read_unaligned() }
    }
}
impl<T: Copy> Iterator for HostValues<'_, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.next == self.end {
            return None;
        }
        let index = self.next;
        self.next += 1;
        Some(self.value_at(index))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.next;
        (remaining, Some(remaining))
    }
}
impl<T: Copy> ExactSizeIterator for HostValues<'_, T> {}
impl<T: Copy> DoubleEndedIterator for HostValues<'_, T> {
    fn next_back(&mut self) -> Option<T> {
        if self.next == self.end {
            return None;
        }
        self.end -= 1;
        Some(self.value_at(self.end))
    }
}

impl EvaluatedArray<'_> {
    /// Read one flattened logical coordinate from this completed native owner.
    /// Reuses the signed-stride validator/reader used by ordinary host iteration.
    /// No native operation, numerical copy, index upload or owning view is made.
    /// Out-of-range and empty reads return None; dtype/layout errors remain typed.
    pub fn try_get<T: ArrayElement + Copy>(&self, index: usize) -> Result<Option<T>, AsSliceError> {
        let values = self.host_values::<T>()?;
        Ok((index < values.end).then(|| values.value_at(index)))
    }

    /// Exact fixed lookup frames in addition to the shared validation/reader.
    /// This describes host control only and grants no evaluation or storage.
    pub fn lookup_control_bytes<T: ArrayElement + Copy>() -> Option<usize> {
        Self::iteration_control_bytes::<T>()?
            .checked_add(std::mem::size_of::<Result<Option<T>, AsSliceError>>())?
            .checked_add(std::mem::size_of::<usize>())
    }

    /// Fixed logical host-iteration frames and result transports, without
    /// numerical storage, compaction, shape copies or evaluation authority.
    /// Uses the same concrete signed-stride iterator returned by `try_iter`.
    pub fn iteration_control_bytes<T: ArrayElement + Copy>() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<HostValues<'static, T>>(),
            size_of::<Result<HostValues<'static, T>, AsSliceError>>(),
            size_of::<LayoutSpan>(),
            size_of::<Result<LayoutSpan, AsSliceError>>(),
            size_of::<AsSliceError>(),
            size_of::<Option<T>>(),
            size_of::<T>(),
            size_of::<(&[i32], &[i64], usize)>(),
            size_of::<(usize, usize, isize, i128, i128)>(),
            size_of::<(usize, Option<usize>)>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    pub(super) fn host_values<T: ArrayElement + Copy>(
        &self,
    ) -> Result<HostValues<'_, T>, AsSliceError> {
        let array = self.as_array();
        if array.dtype() != T::DTYPE {
            return Err(AsSliceError::DtypeMismatch {
                expecting: T::DTYPE,
                found: array.dtype(),
            });
        }
        let span = checked_layout(
            array.shape(),
            array.signed_strides(),
            std::mem::size_of::<T>(),
        )?;
        if span.size != array.size() {
            return Err(AsSliceError::InvalidLayout);
        }
        let data = if span.size == 0 {
            std::ptr::NonNull::<T>::dangling().as_ptr()
        } else {
            let data = T::array_data(array);
            if data.is_null() {
                return Err(AsSliceError::Null);
            }
            data
        };
        Ok(HostValues {
            data,
            shape: array.shape(),
            strides: array.signed_strides(),
            next: 0,
            end: span.size,
            span,
        })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod native_copy_tests;
