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

impl Array {
    /// Observe this array's own readiness without evaluation or a completion
    /// wait. A signalled attached event is detached by the native availability
    /// worker. `None` means the runtime lock is busy; `Some(false)` leaves an
    /// unscheduled or pending source untouched. This is readiness evidence only:
    /// [`Self::try_completed`] still validates actual backing attribution.
    pub fn try_observe_availability(&self) -> crate::error::Result<Option<bool>> {
        let Some(_guard) = crate::utils::runtime_lock::try_enter_for_recovery() else {
            return Ok(None);
        };
        // SAFETY: this retained Array and the runtime loan remain live. The
        // native worker only queries its event; it never evaluates or waits.
        bool::try_from_op(|ready| unsafe {
            safemlx_sys::_mlx_array_is_available(ready, self.as_ptr())
        })
        .map(Some)
    }

    /// Fixed ordinary availability-query and error transports. The selected
    /// event's retained native resources remain with their original producer.
    pub fn ordinary_availability_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        // SAFETY: pure linked sizeof query, with no runtime access.
        let native = unsafe { safemlx_sys::mlx_array_availability_control_bytes() };
        let controls = [
            size_of::<&Self>(),
            size_of::<bool>(),
            size_of::<<bool as Guarded>::Guard>(),
            size_of::<Result<bool, crate::error::Exception>>(),
            size_of::<Result<Option<bool>, crate::error::Exception>>(),
            size_of::<crate::error::RawException>(),
            size_of::<Option<crate::error::RawException>>(),
            size_of::<Option<crate::OriginalScopeObserver>>(),
            size_of::<crate::utils::runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<crate::utils::runtime_lock::RuntimeLockGuard>>(),
            size_of::<safemlx_sys::mlx_array>(),
            size_of::<i32>(),
        ];
        controls.into_iter().try_fold(
            native.checked_add(size_of_val(&controls))?,
            usize::checked_add,
        )
    }

    /// Borrows already completed backing without evaluation, waiting or graph
    /// construction. A lazy or unresolved source remains a typed refusal.
    pub fn try_completed(
        &self,
    ) -> Result<EvaluatedArray<'_>, crate::error::CompletedReadbackError> {
        let descriptor = self.try_descriptor()?;
        if descriptor.facts().allocation().is_none() {
            return Err(crate::error::CompletedReadbackError::Attribution);
        }
        Ok(EvaluatedArray {
            storage: super::EvaluatedArrayStorage::Borrowed(self),
        })
    }

    /// Fixed descriptor inspection and borrowed-result transport controls.
    pub fn completed_borrow_control_bytes() -> Option<usize> {
        Self::descriptor_control_bytes()?
            .checked_add(std::mem::size_of::<&Self>())?
            .checked_add(std::mem::size_of::<
                Result<EvaluatedArray<'_>, crate::error::CompletedReadbackError>,
            >())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod native_copy_tests;

impl EvaluatedArray<'_> {
    /// Copies a flattened logical element range into native-endian bytes.
    /// The destination length must be a multiple of the actual scalar width.
    /// Signed strides and broadcasts use the existing synchronous completed
    /// element worker, including device-to-host copies for CUDA. No graph or
    /// staging allocation is created. Invalid ranges leave the destination
    /// unchanged; a native copy failure may leave a completed prefix.
    pub fn try_copy_native_element_range_into(
        &self,
        start: usize,
        destination: &mut [u8],
    ) -> Result<(), crate::error::NativeBytesCopyError> {
        use crate::error::CompletedReadbackError;
        let array = self.as_array();
        let scalar_bytes = array.item_size();
        let span = checked_layout(array.shape(), array.signed_strides(), scalar_bytes)?;
        if span.size != array.size() || destination.len() % scalar_bytes != 0 {
            return Err(AsSliceError::InvalidLayout.into());
        }
        let count = destination.len() / scalar_bytes;
        let end = start.checked_add(count).ok_or(AsSliceError::TooLarge)?;
        if end > span.size {
            return Err(AsSliceError::InvalidLayout.into());
        }
        if count == 0 {
            return Ok(());
        }
        let Some(_guard) = crate::utils::runtime_lock::try_enter_for_recovery() else {
            return Err(CompletedReadbackError::Busy.into());
        };
        for (index, output) in (start..end).zip(destination.chunks_exact_mut(scalar_bytes)) {
            // SAFETY: range and scalar width were checked before any write.
            // The completed source and exact output chunk remain borrowed for
            // this synchronous call; native code retains no destination access.
            let status = unsafe {
                safemlx_sys::mlx_array_copy_completed_element(
                    array.as_ptr(),
                    index,
                    output.as_mut_ptr().cast(),
                    scalar_bytes,
                )
            };
            match status {
                0 => {}
                -1 => return Err(CompletedReadbackError::Attribution.into()),
                value => return Err(CompletedReadbackError::CudaStatus(value).into()),
            }
        }
        Ok(())
    }

    /// Fixed range, layout and synchronous native-copy controls. The byte
    /// destination and its accounting owner remain the caller's responsibility.
    pub fn native_element_range_readback_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let controls = [
            size_of::<(&Self, &super::Array, &mut [u8])>(),
            size_of::<LayoutSpan>(),
            size_of::<Result<LayoutSpan, AsSliceError>>(),
            size_of::<(&[i32], &[i64], usize)>(),
            size_of::<(usize, usize, isize, i128, i128)>(),
            size_of::<[usize; 5]>(),
            size_of::<Option<usize>>(),
            size_of::<crate::utils::runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<crate::utils::runtime_lock::RuntimeLockGuard>>(),
            size_of::<std::iter::Zip<std::ops::Range<usize>, std::slice::ChunksExactMut<'_, u8>>>(),
            size_of::<Option<(usize, &mut [u8])>>(),
            size_of::<Result<(), crate::error::NativeBytesCopyError>>(),
            size_of::<crate::error::CompletedReadbackError>(),
            size_of::<AsSliceError>(),
            size_of::<i32>(),
        ];
        // SAFETY: pure linked layout query for the same completed scalar worker.
        let native = unsafe { safemlx_sys::mlx_array_completed_readback_control_bytes() };
        controls.into_iter().try_fold(
            native.checked_add(size_of_val(&controls))?,
            usize::checked_add,
        )
    }

    /// Copies completed storage into an exact caller-owned host destination.
    /// No graph, compaction buffer, allocation or implicit source relocation is
    /// created. CUDA device-to-host copies complete before this method returns.
    /// Shape/type refusals leave the destination unchanged; a native copy error
    /// can leave a copied prefix and preserves its native status.
    pub fn try_copy_into<T: ArrayElement + Copy>(
        &self,
        destination: &mut [T],
    ) -> Result<(), crate::error::CompletedReadbackError> {
        use crate::error::CompletedReadbackError;
        let array = self.as_array();
        if array.dtype() != T::DTYPE {
            return Err(AsSliceError::DtypeMismatch {
                expecting: T::DTYPE,
                found: array.dtype(),
            }
            .into());
        }
        let span = checked_layout(
            array.shape(),
            array.signed_strides(),
            std::mem::size_of::<T>(),
        )?;
        if span.size != array.size() {
            return Err(AsSliceError::InvalidLayout.into());
        }
        if destination.len() != span.size {
            return Err(CompletedReadbackError::DestinationLength {
                expected: span.size,
                found: destination.len(),
            });
        }
        if span.size == 0 {
            return Ok(());
        }
        let Some(_guard) = crate::utils::runtime_lock::try_enter_for_recovery() else {
            return Err(CompletedReadbackError::Busy);
        };
        let bytes = destination
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(AsSliceError::TooLarge)?;
        // SAFETY: EvaluatedArray owns completion evidence, the checked logical
        // extent equals the live typed destination, and this synchronous worker
        // retains no pointer after return. It rechecks actual native provenance.
        let status = unsafe {
            safemlx_sys::mlx_array_copy_completed_data(
                array.as_ptr(),
                destination.as_mut_ptr().cast(),
                bytes,
            )
        };
        match status {
            0 => Ok(()),
            -1 => Err(CompletedReadbackError::Attribution),
            value => Err(CompletedReadbackError::CudaStatus(value)),
        }
    }

    /// Fixed controls for the synchronous copy worker; excludes the caller's
    /// destination allocation and its retained accounting owner.
    pub fn completed_readback_control_bytes<T: ArrayElement + Copy>() -> Option<usize> {
        use std::mem::size_of;
        // SAFETY: pure sizeof-based native layout query; no runtime initialization.
        let native = unsafe { safemlx_sys::mlx_array_completed_readback_control_bytes() };
        [
            size_of::<Self>(),
            size_of::<&mut [T]>(),
            size_of::<LayoutSpan>(),
            size_of::<crate::utils::runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<crate::utils::runtime_lock::RuntimeLockGuard>>(),
            size_of::<Result<(), crate::error::CompletedReadbackError>>(),
            size_of::<usize>() * 2,
            size_of::<i32>(),
        ]
        .into_iter()
        .try_fold(native, usize::checked_add)
    }
}

impl EvaluatedArray<'_> {
    /// Copies and converts logical values directly into an admitted destination.
    /// Each source scalar uses an inline temporary and the same synchronous
    /// completed-backing worker. No tensor-sized staging allocation or graph is
    /// created. A conversion/native failure may leave a completed output prefix.
    pub fn try_map_into<T: ArrayElement + Copy + Default, U: Copy>(
        &self,
        destination: &mut [U],
        map: fn(T) -> U,
    ) -> Result<(), crate::error::CompletedReadbackError> {
        use crate::error::CompletedReadbackError;
        let array = self.as_array();
        if array.dtype() != T::DTYPE {
            return Err(AsSliceError::DtypeMismatch {
                expecting: T::DTYPE,
                found: array.dtype(),
            }
            .into());
        }
        let span = checked_layout(
            array.shape(),
            array.signed_strides(),
            std::mem::size_of::<T>(),
        )?;
        if span.size != array.size() {
            return Err(AsSliceError::InvalidLayout.into());
        }
        if destination.len() != span.size {
            return Err(CompletedReadbackError::DestinationLength {
                expected: span.size,
                found: destination.len(),
            });
        }
        if span.size == 0 {
            return Ok(());
        }
        let Some(_guard) = crate::utils::runtime_lock::try_enter_for_recovery() else {
            return Err(CompletedReadbackError::Busy);
        };
        for (index, output) in destination.iter_mut().enumerate() {
            let mut value = T::default();
            // SAFETY: the source is complete, this stack scalar has its exact
            // dtype extent, and the native worker retains no access on return.
            let status = unsafe {
                safemlx_sys::mlx_array_copy_completed_element(
                    array.as_ptr(),
                    index,
                    (&mut value as *mut T).cast(),
                    std::mem::size_of::<T>(),
                )
            };
            match status {
                0 => *output = map(value),
                -1 => return Err(CompletedReadbackError::Attribution),
                value => return Err(CompletedReadbackError::CudaStatus(value)),
            }
        }
        Ok(())
    }

    /// Fixed scalar conversion controls, in addition to the caller's destination.
    pub fn completed_mapped_readback_control_bytes<T: ArrayElement + Copy + Default, U: Copy>()
    -> Option<usize> {
        use std::mem::size_of;
        Self::completed_readback_control_bytes::<T>()?
            .checked_add(size_of::<T>())?
            .checked_add(size_of::<U>())?
            .checked_add(size_of::<fn(T) -> U>())?
            .checked_add(size_of::<&mut [U]>())?
            .checked_add(size_of::<std::iter::Enumerate<std::slice::IterMut<'_, U>>>())?
            .checked_add(size_of::<usize>())
    }

    /// Copies a checked rectangular region through the same completed-scalar
    /// worker used by mapped readback. The source's signed strides remain
    /// authoritative; no slice graph, index upload or staging buffer is made.
    /// Geometry and destination-length errors occur before any write. A native
    /// transfer failure can leave a copied prefix, as with full mapped readback.
    pub fn try_map_region_into<T: ArrayElement + Copy + Default, U: Copy>(
        &self,
        starts: &[u64],
        shape: &[u64],
        destination: &mut [U],
        map: fn(T) -> U,
    ) -> Result<(), crate::error::CompletedReadbackError> {
        use crate::error::CompletedReadbackError;
        let array = self.as_array();
        if array.dtype() != T::DTYPE {
            return Err(AsSliceError::DtypeMismatch {
                expecting: T::DTYPE,
                found: array.dtype(),
            }
            .into());
        }
        let dimensions = array.shape();
        let span = checked_layout(dimensions, array.signed_strides(), std::mem::size_of::<T>())?;
        if span.size != array.size()
            || starts.len() != dimensions.len()
            || shape.len() != dimensions.len()
        {
            return Err(AsSliceError::InvalidLayout.into());
        }
        let mut count = 1usize;
        for ((&start, &extent), &dimension) in starts.iter().zip(shape).zip(dimensions) {
            if start
                .checked_add(extent)
                .is_none_or(|end| end > dimension as u64)
            {
                return Err(AsSliceError::InvalidLayout.into());
            }
            count = count
                .checked_mul(usize::try_from(extent).map_err(|_| AsSliceError::TooLarge)?)
                .ok_or(AsSliceError::TooLarge)?;
        }
        if destination.len() != count {
            return Err(CompletedReadbackError::DestinationLength {
                expected: count,
                found: destination.len(),
            });
        }
        if count == 0 {
            return Ok(());
        }
        let Some(_guard) = crate::utils::runtime_lock::try_enter_for_recovery() else {
            return Err(CompletedReadbackError::Busy);
        };
        for (index, output) in destination.iter_mut().enumerate() {
            let mut remaining = index;
            let mut source_index = 0usize;
            let mut stride = 1usize;
            for ((&start, &extent), &dimension) in starts.iter().zip(shape).zip(dimensions).rev() {
                // Validation proves nonzero extents for this nonempty output,
                // and the complete logical source population fits usize.
                source_index += (start as usize + remaining % extent as usize) * stride;
                remaining /= extent as usize;
                stride *= dimension as usize;
            }
            let mut value = T::default();
            // SAFETY: the complete source and validated logical region retain
            // every selected element. The synchronous worker writes only this
            // exact inline scalar and retains no destination pointer.
            let status = unsafe {
                safemlx_sys::mlx_array_copy_completed_element(
                    array.as_ptr(),
                    source_index,
                    (&mut value as *mut T).cast(),
                    std::mem::size_of::<T>(),
                )
            };
            match status {
                0 => *output = map(value),
                -1 => return Err(CompletedReadbackError::Attribution),
                value => return Err(CompletedReadbackError::CudaStatus(value)),
            }
        }
        Ok(())
    }

    /// Fixed region validation, coordinate and scalar-copy controls. Numerical
    /// destination storage belongs to the caller's independent grant.
    pub fn completed_region_readback_control_bytes<T: ArrayElement + Copy + Default, U: Copy>()
    -> Option<usize> {
        use std::mem::size_of;
        Self::completed_mapped_readback_control_bytes::<T, U>()?
            .checked_add(size_of::<(&[u64], &[u64], &[i32])>())?
            .checked_add(size_of::<[usize; 5]>())?
            .checked_add(size_of::<
                std::iter::Zip<
                    std::iter::Zip<std::slice::Iter<'_, u64>, std::slice::Iter<'_, u64>>,
                    std::slice::Iter<'_, i32>,
                >,
            >())
    }
}
