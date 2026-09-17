use crate::{
    dtype::Dtype,
    error::AsSliceError,
    sealed::Sealed,
    utils::{guard::Guarded, runtime_lock},
    Stream,
};
use element::FromSliceElement;
use num_complex::Complex;
use safemlx_sys::mlx_array;
use std::ffi::c_void;

mod element;
pub(crate) mod host_read;
mod metadata;
mod prepared_clone;
mod prepared_host;
mod prepared_mutable;
mod prepared_prompt;
mod repeated_i32;
pub use prepared_clone::{PreparedArrayClone, PreparedArrayCloneCause};
pub use prepared_host::{
    CompletedOwnedHostCopy, OwnedHostBufferCopyError, OwnedHostCopyBuffer, OwnedHostCopyCause,
    OwnedHostCopyError, OwnedHostCopyFacts, OwnedHostCopyPlan, OwnedHostCopyPreparationError,
    OwnedHostCopyPreparationOwner, OwnedHostCopyStrategy, PreparedOwnedHostCopy,
};
pub use prepared_mutable::{
    OriginalMutablePairCause, OriginalMutablePairCustodies, OriginalMutablePairError,
    OriginalMutablePairFacts, OriginalMutablePairOwner, OriginalMutablePairPlan,
    PreparedOriginalMutablePair, RetainedOriginalMutablePair, UnpreparedOriginalMutablePair,
};
pub use prepared_prompt::{OriginalPromptInputCause, OriginalPromptInputFacts};
pub use repeated_i32::RepeatedI32InputPlan;
mod scoped_evaluation;
pub use scoped_evaluation::{
    original_scoped_deep_copy_control_bytes, original_scoped_evaluation_control_bytes,
};

cfg_safetensors! {
    mod safetensors;
}

pub use element::ArrayElement;
pub use metadata::{
    ArrayDescriptorError, ArrayDescriptorFacts, ArrayDescriptorLoan, ArrayMetadataError,
    ArrayMetadataSnapshot, OrdinaryArrayMetadataGuard, OwnedArrayDescriptorLoan,
};

// Not using Complex64 because `num_complex::Complex64` is actually Complex<f64>

/// Type alias for `num_complex::Complex<f32>`.
#[allow(non_camel_case_types)]
pub type complex64 = Complex<f32>;

/// An n-dimensional array.
///
/// Arrays are lazy MLX graph values. They may be moved or shared across
/// threads; use explicit streams for operations and evaluate before host reads.
#[repr(transparent)]
pub struct Array {
    c_array: mlx_array,
}

/// Exact completed backing storage, distinct from a view's logical byte count.
/// Identity is local to the linked native runtime and never reused, including
/// after physical retirement. It is not persistent across processes/runtimes.
/// It grants no pointer access and makes no statement about future allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocationInfo {
    identity: AllocationIdentity,
    bytes: usize,
}

/// Opaque allocation equality key. Nonzero generations are never reused by the
/// linked native runtime, so deferred charges cannot alias a new allocation
/// that reuses an address. Allocator and host-transfer namespaces are distinct.
/// Allocation-free empty values use a zero sentinel and carry no storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocationIdentity {
    host_transfer: bool,
    generation: u64,
}

/// Allocation information returned by a completed native array.
pub type ArrayAllocationInfo = AllocationInfo;

impl AllocationInfo {
    pub(crate) const fn from_native(generation: u64, bytes: usize, host_transfer: bool) -> Self {
        Self {
            identity: AllocationIdentity {
                host_transfer,
                generation,
            },
            bytes,
        }
    }
    /// Opaque equality key for views of the same retained physical allocation.
    pub const fn identity(&self) -> AllocationIdentity {
        self.identity
    }
    /// Complete certified backing capacity, including padding.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
}

// SAFETY: `Array` owns an MLX C++ array handle. MLX arrays are immutable graph
// values from the Rust API's perspective; mutation is represented by producing
// new arrays. Runtime entry points that touch known MLX global state are guarded
// inside safemlx, and callers still provide explicit streams for execution.
unsafe impl Send for Array {}

// SAFETY: Shared references to `Array` expose only immutable graph/value
// operations. Concurrent evaluation and host-read paths are covered by runtime
// guards where MLX requires them, and stress-tested in safemlx-tests. Allocation
// owner attachment mutates only a sidecar, serialized by the same runtime lock.
unsafe impl Sync for Array {}

unsafe extern "C" fn drop_owned_vec<T>(payload: *mut c_void) {
    unsafe {
        drop(Box::from_raw(payload.cast::<Vec<T>>()));
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod direct_init_tests {
    use super::*;
    use crate::{Device, DeviceType};

    #[test]
    fn group_initializer_publishes_the_same_allocations_with_mixed_types() {
        let mut addresses = Vec::new();
        let values = Array::try_init_many_with(
            &[
                (&[4], Dtype::Uint8),
                (&[2], Dtype::Float32),
                (&[0], Dtype::Int32),
            ],
            |outputs| {
                assert_eq!(
                    outputs
                        .iter()
                        .map(|output| output.len())
                        .collect::<Vec<_>>(),
                    [4, 8, 0]
                );
                assert!(outputs
                    .iter()
                    .all(|output| output.iter().all(|byte| *byte == 0)));
                addresses = outputs.iter().map(|output| output.as_ptr()).collect();
                outputs[0].copy_from_slice(&[1, 2, 3, 4]);
                outputs[1].copy_from_slice(
                    &[1.5f32, -2.0]
                        .into_iter()
                        .flat_map(f32::to_ne_bytes)
                        .collect::<Vec<_>>(),
                );
                Ok::<_, crate::error::Exception>(())
            },
        )
        .unwrap();
        assert_ne!(addresses[0], addresses[1]);
        for (value, address) in values.iter().zip(addresses).take(2) {
            assert_eq!(
                value.evaluated().unwrap().to_native_bytes().len(),
                value.nbytes()
            );
            let evaluated = value.evaluated().unwrap();
            let actual = if value.dtype() == Dtype::Uint8 {
                evaluated.as_slice::<u8>().as_ptr()
            } else {
                evaluated.as_slice::<f32>().as_ptr().cast()
            };
            assert_eq!(actual, address);
        }
        assert_eq!(
            values[0].evaluated().unwrap().as_slice::<u8>(),
            [1, 2, 3, 4]
        );
        assert_eq!(
            values[1].evaluated().unwrap().as_slice::<f32>(),
            [1.5, -2.0]
        );
        assert_eq!(values[2].size(), 0);
    }

    #[test]
    fn group_initializer_discards_partial_results_on_error_or_unwind() {
        let specs = [(&[2][..], Dtype::Uint8), (&[3][..], Dtype::Uint8)];
        let result = Array::try_init_many_with(&specs, |outputs| {
            outputs[0].fill(7);
            Err(crate::error::Exception::custom("second read failed"))
        });
        assert!(result.is_err());
        let result = std::panic::catch_unwind(|| {
            Array::try_init_many_with(&specs, |outputs| -> Result<(), crate::error::Exception> {
                outputs[0].fill(7);
                panic!("initializer unwound");
            })
        });
        assert!(result.is_err());
        let mut called = false;
        let result =
            Array::try_init_many_with(&[(&[2], Dtype::Uint8), (&[-1], Dtype::Uint8)], |_| {
                called = true;
                Ok::<_, crate::error::Exception>(())
            });
        assert!(result.is_err());
        assert!(!called);
        assert!(Array::try_init_many_with(&[], |outputs| {
            assert!(outputs.is_empty());
            Ok::<_, crate::error::Exception>(())
        })
        .unwrap()
        .is_empty());
    }

    #[test]
    fn initializer_writes_the_final_native_allocation() {
        let mut initialized_at = std::ptr::null();
        let value = Array::try_init_with(&[4], Dtype::Uint8, |bytes| {
            assert_eq!(bytes, &[0; 4]);
            initialized_at = bytes.as_ptr();
            bytes.copy_from_slice(&[1, 2, 3, 4]);
            Ok::<_, crate::error::Exception>(())
        })
        .unwrap();
        let clone = value.clone();
        drop(value);
        let evaluated = clone.evaluated().unwrap();
        assert_eq!(evaluated.as_slice::<u8>().as_ptr(), initialized_at);
        assert_eq!(evaluated.as_slice::<u8>(), &[1, 2, 3, 4]);
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let sum = clone
            .sum(false, &stream)
            .unwrap()
            .try_item::<u32>(&stream)
            .unwrap();
        assert_eq!(sum, 10);
        #[cfg(feature = "metal")]
        {
            let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
            let promoted = clone.copy(&stream).unwrap();
            let evaluated = promoted.evaluated().unwrap();
            assert_eq!(evaluated.as_slice::<u8>().as_ptr(), initialized_at);
            assert_eq!(
                promoted
                    .sum(false, &stream)
                    .unwrap()
                    .try_item::<u32>(&stream)
                    .unwrap(),
                10
            );
        }
    }

    #[test]
    fn initializer_errors_and_invalid_shapes_do_not_publish_arrays() {
        let result = Array::try_init_with(&[2], Dtype::Float32, |_bytes| {
            Err(crate::error::Exception::custom("read failed"))
        });
        assert!(result.is_err());
        let mut called = false;
        let result = Array::try_init_with(&[-1], Dtype::Uint8, |_| {
            called = true;
            Ok::<_, crate::error::Exception>(())
        });
        assert!(result.is_err());
        assert!(!called);
        let result = Array::try_init_with(&[0, i32::MAX, i32::MAX, i32::MAX], Dtype::Uint8, |_| {
            called = true;
            Ok::<_, crate::error::Exception>(())
        });
        assert!(result.is_err());
        assert!(!called);
    }
}

#[cfg(feature = "safetensors")]
unsafe extern "C" fn retain_borrowed_data(_: *mut c_void) {}

const fn dtype_byte_width(dtype: Dtype) -> usize {
    match dtype {
        Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
        Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
        Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
        Dtype::Uint64 | Dtype::Int64 | Dtype::Float64 | Dtype::Complex64 => 8,
    }
}

/// An evaluated array with materialized storage available for host reads.
pub struct EvaluatedArray<'a> {
    storage: EvaluatedArrayStorage<'a>,
}

enum EvaluatedArrayStorage<'a> {
    Borrowed(&'a Array),
    Owned(Array),
}

impl Sealed for Array {}

impl Sealed for &Array {}

impl std::fmt::Debug for EvaluatedArray<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvaluatedArray")
            .field("dtype", &self.as_array().dtype())
            .field("shape", &self.as_array().shape())
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Array")
            .field("dtype", &self.dtype())
            .field("shape", &self.shape())
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Array(dtype={:?}, shape={:?})",
            self.dtype(),
            self.shape()
        )
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        // TODO: check memory leak with some tool?

        // Decrease the reference count
        unsafe { safemlx_sys::mlx_array_free(self.as_ptr()) };
    }
}

impl Array {
    /// Create a new array from an existing mlx_array pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure the reference count of the array is properly incremented with
    /// `safemlx_sys::mlx_retain`.
    pub unsafe fn from_ptr(c_array: mlx_array) -> Array {
        Self { c_array }
    }

    /// Get the underlying mlx_array pointer.
    pub fn as_ptr(&self) -> mlx_array {
        self.c_array
    }

    /// New array from a bool scalar.
    pub fn from_bool(val: bool) -> Array {
        let c_array = unsafe { safemlx_sys::mlx_array_new_bool(val) };
        Array { c_array }
    }

    /// New array from an int scalar.
    pub fn from_int(val: i32) -> Array {
        let c_array = unsafe { safemlx_sys::mlx_array_new_int(val) };
        Array { c_array }
    }

    /// New array from a f32 scalar.
    pub fn from_f32(val: f32) -> Array {
        let c_array = unsafe { safemlx_sys::mlx_array_new_float32(val) };
        Array { c_array }
    }

    /// New array from a f64 scalar.
    pub fn from_f64(val: f64) -> Array {
        let c_array = unsafe { safemlx_sys::mlx_array_new_float64(val) };
        Array { c_array }
    }

    /// New array from a complex scalar.
    pub fn from_complex(val: complex64) -> Array {
        let c_array = unsafe { safemlx_sys::mlx_array_new_complex(val.re, val.im) };
        Array { c_array }
    }

    /// Construct a scalar while preserving a native allocation/refusal error.
    pub fn try_from_scalar<T: FromSliceElement>(value: T) -> crate::error::Result<Self> {
        Self::try_from_slice(std::slice::from_ref(&value), &[])
    }

    /// Copy a typed host slice after checked shape validation. A native error
    /// returns its exact source; no null Array is exposed on construction failure.
    pub fn try_from_slice<T: FromSliceElement>(
        data: &[T],
        shape: &[i32],
    ) -> crate::error::Result<Self> {
        let dimensions = i32::try_from(shape.len())
            .map_err(|_| crate::error::Exception::custom("array rank exceeds int32"))?;
        let elements = shape
            .iter()
            .try_fold(1usize, |size, dimension| {
                usize::try_from(*dimension)
                    .ok()
                    .and_then(|dimension| size.checked_mul(dimension))
            })
            .ok_or_else(|| {
                crate::error::Exception::custom("negative or overflowing array shape")
            })?;
        if elements != data.len() {
            return Err(crate::error::Exception::custom(
                "array shape does not match host slice",
            ));
        }
        Self::try_from_op(|output| unsafe {
            safemlx_sys::mlx_array_set_data(
                output,
                data.as_ptr().cast(),
                shape.as_ptr(),
                dimensions,
                T::DTYPE.into(),
            )
        })
    }

    /// Fallible signed integer scalar construction.
    pub fn try_from_int(value: i32) -> crate::error::Result<Self> {
        Self::try_from_scalar(value)
    }
    /// Fallible float scalar construction.
    pub fn try_from_f32(value: f32) -> crate::error::Result<Self> {
        Self::try_from_scalar(value)
    }
    /// Fallible boolean scalar construction.
    pub fn try_from_bool(value: bool) -> crate::error::Result<Self> {
        Self::try_from_scalar(value)
    }

    /// New array from existing buffer.
    ///
    /// Please note that floating point literals are treated as f32 instead of
    /// f64. Use [`Array::from_slice_f64`] for f64.
    ///
    /// # Parameters
    ///
    /// - `data`: A buffer which will be copied.
    /// - `shape`: Shape of the array.
    ///
    /// # Panic
    ///
    /// - Panics if the product of the shape is not equal to the length of the
    ///   data.
    /// - Panics if the shape is too large.
    pub fn from_slice<T: FromSliceElement>(data: &[T], shape: &[i32]) -> Self {
        // Validate data size and shape
        assert_eq!(data.len(), shape.iter().product::<i32>() as usize);

        unsafe { Self::from_raw_data(data.as_ptr() as *const c_void, shape, T::DTYPE) }
    }

    /// New array from a slice of f64.
    ///
    /// A separate method is provided for f64 because f64 is not supported on GPU
    /// and rust defaults to f64 for floating point literals
    pub fn from_slice_f64(data: &[f64], shape: &[i32]) -> Self {
        // Validate data size and shape
        assert_eq!(data.len(), shape.iter().product::<i32>() as usize);

        unsafe { Self::from_raw_data(data.as_ptr() as *const c_void, shape, Dtype::Float64) }
    }

    /// Create a new array from raw data buffer.
    ///
    /// This is a convenience wrapper around [`safemlx_sys::mlx_array_new_data`].
    ///
    /// # Safety
    ///
    /// This is unsafe because the caller must ensure that the data buffer is valid and that the
    /// shape is correct.
    #[inline]
    pub unsafe fn from_raw_data(data: *const c_void, shape: &[i32], dtype: Dtype) -> Self {
        let dim = if shape.len() > i32::MAX as usize {
            panic!("Shape is too large")
        } else {
            shape.len() as i32
        };

        let c_array = safemlx_sys::mlx_array_new_data(data, shape.as_ptr(), dim, dtype.into());
        Array { c_array }
    }

    /// Transfers an owned, typed host allocation into an MLX array, adopting
    /// its storage without copying when the native backend supports it.
    pub fn try_from_owned_data<T: ArrayElement + 'static>(
        values: Vec<T>,
        shape: &[i32],
    ) -> crate::error::Result<Self> {
        if T::DTYPE == Dtype::Bool {
            return Err(crate::error::Exception::custom(
                "owned boolean buffers are unsupported",
            ));
        }
        unsafe { Self::try_from_owned_host_data(values, shape, T::DTYPE) }
    }

    /// Initializes an array directly in its final CPU or shared Metal storage.
    ///
    /// The callback receives exclusive access to zero-initialized bytes. The
    /// array becomes observable only after the callback succeeds; errors and
    /// unwinding release the allocation. No intermediate host allocation or
    /// host-to-device copy is needed. CUDA storage is unsupported.
    pub fn try_init_with<E: From<crate::error::Exception>>(
        shape: &[i32],
        dtype: Dtype,
        initialize: impl FnOnce(&mut [u8]) -> std::result::Result<(), E>,
    ) -> std::result::Result<Self, E> {
        let (value, data, len) = Self::allocate_host_initializer(shape, dtype)?;
        // SAFETY: allocation initialized the checked extent. The exclusively
        // owned array is not exposed until the callback's borrow has ended.
        let bytes = if len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(data.cast::<u8>(), len) }
        };
        initialize(bytes)?;
        Ok(value)
    }

    /// Initializes several final CPU or shared Metal allocations together.
    ///
    /// The callback receives disjoint, zero-initialized buffers in specification
    /// order. No array is published until it succeeds for the complete group.
    /// Failure or unwinding releases all allocations, including partial output.
    pub fn try_init_many_with<E: From<crate::error::Exception>>(
        specifications: &[(&[i32], Dtype)],
        initialize: impl FnOnce(&mut [&mut [u8]]) -> std::result::Result<(), E>,
    ) -> std::result::Result<Vec<Self>, E> {
        let allocations = specifications
            .iter()
            .map(|(shape, dtype)| Self::allocate_host_initializer(shape, *dtype))
            .collect::<crate::error::Result<Vec<_>>>()?;
        let mut buffers = allocations
            .iter()
            .map(|(_, data, len)| {
                // SAFETY: every checked allocation is initialized and exclusively
                // owned here. Live allocations are disjoint; their arrays remain
                // retained and inaccessible throughout the callback.
                if *len == 0 {
                    &mut [][..]
                } else {
                    unsafe { std::slice::from_raw_parts_mut(data.cast::<u8>(), *len) }
                }
            })
            .collect::<Vec<_>>();
        initialize(&mut buffers)?;
        drop(buffers);
        Ok(allocations.into_iter().map(|(array, _, _)| array).collect())
    }

    fn allocate_host_initializer(
        shape: &[i32],
        dtype: Dtype,
    ) -> crate::error::Result<(Self, *mut c_void, usize)> {
        let len = checked_elements(shape)?
            .checked_mul(dtype_byte_width(dtype))
            .filter(|len| *len <= isize::MAX as usize)
            .ok_or_else(|| crate::error::Exception::custom("array byte count overflow"))?;
        let rank = i32::try_from(shape.len())
            .map_err(|_| crate::error::Exception::custom("shape is too large"))?;
        let mut data: *mut c_void = std::ptr::null_mut();
        let value = {
            let _guard = runtime_lock::enter();
            Self::try_from_op(|output| unsafe {
                safemlx_sys::mlx_array_new_host(
                    output,
                    &mut data,
                    shape.as_ptr(),
                    rank,
                    dtype.into(),
                )
            })?
        };
        Ok((value, data, len))
    }

    /// Copies a checked typed slice into a new MLX array.
    pub fn try_from_data<T: ArrayElement>(
        values: &[T],
        shape: &[i32],
    ) -> crate::error::Result<Self> {
        let elements = checked_elements(shape)?;
        if values.len() != elements {
            return Err(crate::error::Exception::custom(
                "array element count does not match shape",
            ));
        }
        Ok(unsafe { Self::from_raw_data(values.as_ptr().cast(), shape, T::DTYPE) })
    }

    unsafe fn try_from_owned_host_data<T: 'static>(
        mut values: Vec<T>,
        shape: &[i32],
        dtype: Dtype,
    ) -> crate::error::Result<Self> {
        if shape.len() > i32::MAX as usize {
            return Err(crate::error::Exception::custom("shape is too large"));
        }
        let elements = shape.iter().try_fold(1usize, |count, dimension| {
            usize::try_from(*dimension)
                .ok()
                .and_then(|dimension| count.checked_mul(dimension))
                .ok_or_else(|| crate::error::Exception::custom("invalid or overflowing shape"))
        })?;
        let expected = elements
            .checked_mul(dtype_byte_width(dtype))
            .ok_or_else(|| crate::error::Exception::custom("array byte count overflow"))?;
        if std::mem::size_of_val(values.as_slice()) != expected {
            return Err(crate::error::Exception::custom(
                "owned array byte count does not match shape and dtype",
            ));
        }
        let alignment = dtype_byte_width(dtype).min(8);
        if !(values.as_ptr() as usize).is_multiple_of(alignment) {
            return Err(crate::error::Exception::custom(
                "owned array storage is not aligned for its dtype",
            ));
        }
        let data = values.as_mut_ptr().cast::<c_void>();
        let payload = Box::into_raw(Box::new(values)).cast::<c_void>();
        let _guard = runtime_lock::enter();
        let c_array = unsafe {
            safemlx_sys::mlx_array_new_data_managed_payload(
                data,
                shape.as_ptr(),
                shape.len() as i32,
                dtype.into(),
                payload,
                Some(drop_owned_vec::<T>),
            )
        };
        Ok(Array { c_array })
    }

    /// Creates a host array that borrows caller-owned storage without copying.
    ///
    /// # Safety
    ///
    /// `data` must remain valid, immutable, and aligned for `dtype` until every
    /// clone of the returned array has been dropped or fully evaluated by a
    /// dependent operation.
    #[cfg(feature = "safetensors")]
    pub(crate) unsafe fn try_from_borrowed_host_data(
        data: *const c_void,
        shape: &[i32],
        dtype: Dtype,
    ) -> crate::error::Result<Self> {
        if data.is_null() || shape.len() > i32::MAX as usize {
            return Err(crate::error::Exception::custom(
                "borrowed array data or shape is invalid",
            ));
        }
        let alignment = dtype_byte_width(dtype).min(8);
        if !(data as usize).is_multiple_of(alignment) {
            return Err(crate::error::Exception::custom(
                "borrowed array storage is not aligned for its dtype",
            ));
        }
        let _guard = runtime_lock::enter();
        let c_array = unsafe {
            safemlx_sys::mlx_array_new_data_managed_payload(
                data.cast_mut(),
                shape.as_ptr(),
                shape.len() as i32,
                dtype.into(),
                std::ptr::null_mut(),
                Some(retain_borrowed_data),
            )
        };
        Ok(Array { c_array })
    }

    /// New array from an iterator.
    ///
    /// Please note that floating point literals are treated as f32 instead of
    /// f64. Use [`Array::from_iter_f64`] for f64.
    ///
    /// This is a convenience method that is equivalent to
    ///
    /// ```rust, ignore
    /// let data: Vec<T> = iter.collect();
    /// Array::from_slice(&data, shape)
    /// ```
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    /// use safemlx::Array;
    ///
    /// let data = vec![1i32, 2, 3, 4, 5];
    /// let mut array = Array::from_iter(data.clone(), &[5]);
    /// let array = array.evaluated().unwrap();
    /// assert_eq!(array.as_slice::<i32>(), &data[..]);
    /// ```
    pub fn from_iter<I: IntoIterator<Item = T>, T: FromSliceElement>(
        iter: I,
        shape: &[i32],
    ) -> Self {
        let data: Vec<T> = iter.into_iter().collect();
        Self::from_slice(&data, shape)
    }

    /// New array from an iterator of f64.
    ///
    /// A separate method is provided for f64 because f64 is not supported on GPU
    /// and rust defaults to f64 for floating point literals
    pub fn from_iter_f64<I: IntoIterator<Item = f64>>(iter: I, shape: &[i32]) -> Self {
        let data: Vec<f64> = iter.into_iter().collect();
        Self::from_slice_f64(&data, shape)
    }

    /// The size of the array’s datatype in bytes.
    pub fn item_size(&self) -> usize {
        unsafe { safemlx_sys::mlx_array_itemsize(self.as_ptr()) }
    }

    /// Number of elements in the array.
    pub fn size(&self) -> usize {
        unsafe { safemlx_sys::mlx_array_size(self.as_ptr()) }
    }

    /// The legacy unsigned strides. Reversed strides wrap as unsigned values;
    /// use [`Self::signed_strides`] when interpreting logical coordinates.
    pub fn strides(&self) -> &[usize] {
        let ndim = self.ndim();
        if ndim == 0 {
            // The data pointer may be null which would panic even if len is 0
            return &[];
        }

        unsafe {
            let data = safemlx_sys::mlx_array_strides(self.as_ptr());
            std::slice::from_raw_parts(data, ndim)
        }
    }

    /// Native signed strides in elements, including negative and zero strides.
    /// Evaluate first to inspect the realized layout of a lazy view.
    pub fn signed_strides(&self) -> &[i64] {
        let ndim = self.ndim();
        if ndim == 0 {
            return &[];
        }
        // SAFETY: the retained native descriptor owns exactly ndim int64 strides.
        // Unlike the legacy size_t API, this preserves the native signed type.
        unsafe {
            let data = safemlx_sys::mlx_array_signed_strides(self.as_ptr());
            std::slice::from_raw_parts(data, ndim)
        }
    }

    /// The number of bytes in the array.
    pub fn nbytes(&self) -> usize {
        unsafe { safemlx_sys::mlx_array_nbytes(self.as_ptr()) }
    }

    /// Reads completed allocator or certified host-transfer storage without
    /// evaluation, polling or waiting. Unfinished or unrecognized storage is unknown.
    /// Exhausted allocation generations also remain unknown; identity queries
    /// never assign a replacement generation or interfere with deallocation.
    /// Logical view size never substitutes for unknown physical capacity.
    pub fn allocation_info(&self) -> crate::error::Result<Option<ArrayAllocationInfo>> {
        let _guard = runtime_lock::enter();
        let (mut known, mut host_transfer, mut identity, mut bytes) = (false, false, 0, 0);
        // SAFETY: all outputs point to live initialized scalars. The native
        // query only reads this retained array and certified native-owned data;
        // it neither exposes nor dereferences a caller-provided data pointer.
        <() as Guarded>::try_from_op(|_| unsafe {
            safemlx_sys::mlx_array_allocation_info(
                &mut known,
                &mut host_transfer,
                &mut identity,
                &mut bytes,
                self.as_ptr(),
            )
        })?;
        Ok(known.then_some(AllocationInfo::from_native(identity, bytes, host_transfer)))
    }

    /// The array’s dimension.
    pub fn ndim(&self) -> usize {
        unsafe { safemlx_sys::mlx_array_ndim(self.as_ptr()) }
    }

    /// The shape of the array.
    ///
    /// Returns: a pointer to the sizes of each dimension.
    pub fn shape(&self) -> &[i32] {
        let ndim = self.ndim();
        if ndim == 0 {
            // The data pointer may be null which would panic even if len is 0
            return &[];
        }

        unsafe {
            let data = safemlx_sys::mlx_array_shape(self.as_ptr());
            std::slice::from_raw_parts(data, ndim)
        }
    }

    /// The shape of the array in a particular dimension.
    ///
    /// # Panic
    ///
    /// - Panics if the array is scalar.
    /// - Panics if `dim` is negative and `dim + ndim` overflows
    /// - Panics if the dimension is out of bounds.
    pub fn dim(&self, dim: i32) -> i32 {
        let dim = if dim.is_negative() {
            (self.ndim() as i32).checked_add(dim).unwrap()
        } else {
            dim
        };

        // This will panic on a scalar array
        unsafe { safemlx_sys::mlx_array_dim(self.as_ptr(), dim) }
    }

    /// The array element type.
    pub fn dtype(&self) -> Dtype {
        let dtype = unsafe { safemlx_sys::mlx_array_dtype(self.as_ptr()) };
        Dtype::try_from(dtype).unwrap()
    }

    /// Evaluate the array and return a borrowed host-readable value.
    pub fn evaluated(&self) -> crate::error::Result<EvaluatedArray<'_>> {
        if let Some(result) = scoped_evaluation::evaluate(self) {
            result?;
            return Ok(EvaluatedArray {
                storage: EvaluatedArrayStorage::Borrowed(self),
            });
        }
        let _guard = runtime_lock::enter();
        <() as Guarded>::try_from_op(|_| unsafe { safemlx_sys::mlx_array_eval(self.as_ptr()) })?;
        Ok(EvaluatedArray {
            storage: EvaluatedArrayStorage::Borrowed(self),
        })
    }

    /// Evaluate the array and return an owned host-readable value.
    pub fn into_evaluated(self) -> crate::error::Result<EvaluatedArray<'static>> {
        if let Some(result) = scoped_evaluation::evaluate(&self) {
            result?;
            return Ok(EvaluatedArray {
                storage: EvaluatedArrayStorage::Owned(self),
            });
        }
        let _guard = runtime_lock::enter();
        <() as Guarded>::try_from_op(|_| unsafe { safemlx_sys::mlx_array_eval(self.as_ptr()) })?;
        Ok(EvaluatedArray {
            storage: EvaluatedArrayStorage::Owned(self),
        })
    }

    /// Evaluate and access the value of a scalar array.
    ///
    /// If `T` does not match the array's dtype, the value is converted on
    /// `stream` before evaluation.
    pub fn item<T: ArrayElement>(self, stream: impl AsRef<Stream>) -> T {
        self.try_item(stream).unwrap()
    }

    /// Evaluate and access the value of a scalar array.
    ///
    /// If `T` does not match the array's dtype, the value is converted on
    /// `stream` before evaluation.
    pub fn try_item<T: ArrayElement>(self, stream: impl AsRef<Stream>) -> crate::error::Result<T> {
        let stream = stream.as_ref();
        let array = if self.dtype() == T::DTYPE {
            self
        } else {
            self.as_dtype(T::DTYPE, stream)?
        };

        array.into_evaluated()?.try_item()
    }

    /// Clone the array by copying the data.
    ///
    /// This is named `deep_clone` to avoid confusion with the `Clone` trait.
    pub fn deep_clone(self) -> crate::error::Result<Self> {
        let clone = self.into_evaluated()?.deep_clone()?;
        Ok(clone
            .into_array()
            .expect("deep cloned evaluated arrays always own their storage"))
    }
}

fn checked_elements(shape: &[i32]) -> crate::error::Result<usize> {
    shape.iter().try_fold(1usize, |count, dimension| {
        usize::try_from(*dimension)
            .ok()
            .and_then(|dimension| count.checked_mul(dimension))
            .ok_or_else(|| crate::error::Exception::custom("invalid or overflowing shape"))
    })
}

impl<'a> EvaluatedArray<'a> {
    /// Return the evaluated array to the lazy array type if this value owns it.
    pub fn into_array(self) -> Option<Array> {
        match self.storage {
            EvaluatedArrayStorage::Borrowed(_) => None,
            EvaluatedArrayStorage::Owned(array) => Some(array),
        }
    }

    /// Borrow the underlying array.
    pub fn as_array(&self) -> &Array {
        match &self.storage {
            EvaluatedArrayStorage::Borrowed(array) => array,
            EvaluatedArrayStorage::Owned(array) => array,
        }
    }

    /// Access the value of a scalar array.
    pub fn item<T: ArrayElement>(&self) -> T {
        self.try_item().unwrap()
    }

    /// Access the value of a scalar array returning an error if the array is not a scalar.
    pub fn try_item<T: ArrayElement>(&self) -> crate::error::Result<T> {
        let array = self.as_array();
        if array.dtype() != T::DTYPE {
            return Err(crate::error::Exception::custom(format!(
                "dtype mismatch: expected {:?}, found {:?}",
                T::DTYPE,
                array.dtype()
            )));
        }

        T::array_item(array)
    }

    /// Returns a slice of the array data without validating the dtype.
    ///
    /// # Safety
    ///
    /// The caller must ensure matching dtype, row-contiguous logical storage,
    /// valid initialized elements, a non-null aligned data pointer, and a byte
    /// length no greater than isize::MAX. Reversed, broadcast and other strided
    /// views cannot be borrowed as a flat Rust slice.
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    /// use safemlx::Array;
    ///
    /// let data = [1i32, 2, 3, 4, 5];
    /// let mut array = Array::from_slice(&data[..], &[5]);
    ///
    /// unsafe {
    ///    let array = array.evaluated().unwrap();
    ///    let slice = array.as_slice_unchecked::<i32>();
    ///    assert_eq!(slice, &[1, 2, 3, 4, 5]);
    /// }
    /// ```
    pub unsafe fn as_slice_unchecked<T: ArrayElement>(&self) -> &[T] {
        unsafe {
            let array = self.as_array();
            let data = T::array_data(array);
            let size = array.size();
            std::slice::from_raw_parts(data, size)
        }
    }

    /// Borrows contiguous logical data, rejecting incompatible dtype, strides or alignment.
    /// Empty arrays return an empty slice without dereferencing native storage.
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    /// use safemlx::Array;
    ///
    /// let data = [1i32, 2, 3, 4, 5];
    /// let mut array = Array::from_slice(&data[..], &[5]);
    ///
    /// let array = array.evaluated().unwrap();
    /// let slice = array.try_as_slice::<i32>();
    /// assert_eq!(slice, Ok(&data[..]));
    /// ```
    pub fn try_as_slice<T: ArrayElement>(&self) -> Result<&[T], AsSliceError> {
        let array = self.as_array();
        if array.dtype() != T::DTYPE {
            return Err(AsSliceError::DtypeMismatch {
                expecting: T::DTYPE,
                found: array.dtype(),
            });
        }

        let span = host_read::checked_layout(
            array.shape(),
            array.signed_strides(),
            std::mem::size_of::<T>(),
        )?;
        if span.size != array.size() {
            return Err(AsSliceError::InvalidLayout);
        }
        if span.size == 0 {
            return Ok(&[]);
        }
        if !span.contiguous {
            return Err(AsSliceError::NonContiguous);
        }
        unsafe {
            let size = span.size;
            let data = T::array_data(array);
            if data.is_null() {
                return Err(AsSliceError::Null);
            }
            if !data.is_aligned() {
                return Err(AsSliceError::Misaligned);
            }
            if size > isize::MAX as usize / std::mem::size_of::<T>() {
                return Err(AsSliceError::TooLarge);
            }

            // Evaluation retains initialized native elements. The checks above
            // establish logical contiguity, matching dtype, alignment and a
            // non-null pointer with a Rust-representable slice length.
            Ok(std::slice::from_raw_parts(data, size))
        }
    }

    /// Returns a slice of the array data.
    /// # Panics
    ///
    /// Panics on incompatible dtype, non-contiguous logical storage or misalignment.
    ///
    /// # Example
    ///
    /// ```rust
    /// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    /// use safemlx::Array;
    ///
    /// let data = [1i32, 2, 3, 4, 5];
    /// let mut array = Array::from_slice(&data[..], &[5]);
    ///
    /// let array = array.evaluated().unwrap();
    /// let slice = array.as_slice::<i32>();
    /// assert_eq!(slice, &data[..]);
    /// ```
    pub fn as_slice<T: ArrayElement>(&self) -> &[T] {
        self.try_as_slice().unwrap()
    }

    /// Iterates evaluated values in logical row order without allocating a
    /// numerical payload. Signed strides and unaligned storage are supported.
    /// The iterator borrows the evaluated array and yields copied elements.
    pub fn try_iter<'b, T: ArrayElement + Copy + 'b>(
        &'b self,
    ) -> Result<impl ExactSizeIterator<Item = T> + DoubleEndedIterator + 'b, AsSliceError> {
        self.host_values::<T>()
    }

    /// Copies evaluated data in logical row order into one owned vector.
    /// Supports reversed, broadcast, strided and unaligned views without a
    /// tensor-sized native compaction or an intermediate host payload.
    pub fn try_to_vec<T: ArrayElement + Copy>(&self) -> Result<Vec<T>, AsSliceError> {
        let values = self.host_values::<T>()?;
        // Generic iterator collection may round tiny vectors up to its minimum
        // growth capacity. Preserve the exact requested host payload capacity.
        let mut output = Vec::with_capacity(values.len());
        output.extend(values);
        Ok(output)
    }

    /// Clone the array by copying the data.
    ///
    /// This is named `deep_clone` to avoid confusion with the `Clone` trait.
    pub fn deep_clone(&self) -> crate::error::Result<EvaluatedArray<'static>> {
        let source = self.as_array();
        if let Some(result) = scoped_evaluation::deep_copy(source) {
            return result;
        }
        let row_contiguous =
            host_read::checked_layout(source.shape(), source.signed_strides(), source.item_size())
                .map_err(|error| crate::error::Exception::custom(error.to_string()))?
                .contiguous;
        // mlx_array_data returns a pointer into physical storage, not a packed
        // logical sequence. Copying shape-sized bytes from a strided view can
        // read cache padding or repeat the wrong batch. This API already makes
        // a synchronous host copy; compact strided input on a CPU stream first
        // and retain that evaluated storage through mlx_array_new_data.
        let compact = if row_contiguous {
            None
        } else {
            let stream = Stream::try_default_cpu()?;
            Some(source.contiguous(false, &stream)?.into_evaluated()?)
        };
        let source = compact.as_ref().map_or(source, |value| value.as_array());
        unsafe {
            let array = source;
            let dtype = array.dtype();
            let shape = array.shape();
            let data = match dtype {
                Dtype::Bool => safemlx_sys::mlx_array_data_bool(array.as_ptr()) as *const c_void,
                Dtype::Uint8 => safemlx_sys::mlx_array_data_uint8(array.as_ptr()) as *const c_void,
                Dtype::Uint16 => {
                    safemlx_sys::mlx_array_data_uint16(array.as_ptr()) as *const c_void
                }
                Dtype::Uint32 => {
                    safemlx_sys::mlx_array_data_uint32(array.as_ptr()) as *const c_void
                }
                Dtype::Uint64 => {
                    safemlx_sys::mlx_array_data_uint64(array.as_ptr()) as *const c_void
                }
                Dtype::Int8 => safemlx_sys::mlx_array_data_int8(array.as_ptr()) as *const c_void,
                Dtype::Int16 => safemlx_sys::mlx_array_data_int16(array.as_ptr()) as *const c_void,
                Dtype::Int32 => safemlx_sys::mlx_array_data_int32(array.as_ptr()) as *const c_void,
                Dtype::Int64 => safemlx_sys::mlx_array_data_int64(array.as_ptr()) as *const c_void,
                Dtype::Float16 => {
                    safemlx_sys::mlx_array_data_float16(array.as_ptr()) as *const c_void
                }
                Dtype::Float32 => {
                    safemlx_sys::mlx_array_data_float32(array.as_ptr()) as *const c_void
                }
                Dtype::Float64 => {
                    safemlx_sys::mlx_array_data_float64(array.as_ptr()) as *const c_void
                }
                Dtype::Bfloat16 => {
                    safemlx_sys::mlx_array_data_bfloat16(array.as_ptr()) as *const c_void
                }
                Dtype::Complex64 => {
                    safemlx_sys::mlx_array_data_complex64(array.as_ptr()) as *const c_void
                }
            };

            let new_c_array = safemlx_sys::mlx_array_new_data(
                data,
                shape.as_ptr(),
                shape.len() as i32,
                dtype.into(),
            );

            Ok(EvaluatedArray {
                storage: EvaluatedArrayStorage::Owned(Array::from_ptr(new_c_array)),
            })
        }
    }
}

impl Array {
    /// Shares the existing native descriptor through one fallible C handle.
    /// This does not evaluate or copy tensor storage. The current original
    /// scope, when present, owns the same handle constructor as `Clone`.
    pub fn try_clone_handle(&self) -> crate::error::Result<Self> {
        // SAFETY: the guard supplies a live destination handle and self retains
        // its valid source handle. mlx_array_set shares the existing descriptor;
        // it neither evaluates the graph nor copies numerical payload.
        Array::try_from_op(|res| unsafe { safemlx_sys::mlx_array_set(res, self.as_ptr()) })
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        // Exception may be thrown when calling `new` in cpp.
        self.try_clone_handle().expect("Failed to clone array")
    }
}

impl EvaluatedArray<'_> {
    /// Copies logical values into one native-endian byte vector.
    pub fn to_native_bytes(&self) -> Vec<u8> {
        let array = self.as_array();
        let span =
            host_read::checked_layout(array.shape(), array.signed_strides(), array.item_size())
                .expect("invalid evaluated host layout");
        assert_eq!(
            span.size,
            array.size(),
            "inconsistent evaluated host layout"
        );
        let mut bytes = Vec::with_capacity(span.size * array.item_size());
        macro_rules! extend_bytes {
            ($ty:ty, $encode:expr) => {{
                for value in self
                    .host_values::<$ty>()
                    .expect("invalid evaluated host data")
                {
                    bytes.extend_from_slice(&$encode(value));
                }
            }};
        }
        match self.as_array().dtype() {
            Dtype::Bool => extend_bytes!(bool, |v| [u8::from(v)]),
            Dtype::Uint8 => extend_bytes!(u8, |v| [v]),
            Dtype::Uint16 => extend_bytes!(u16, u16::to_ne_bytes),
            Dtype::Uint32 => extend_bytes!(u32, u32::to_ne_bytes),
            Dtype::Uint64 => extend_bytes!(u64, u64::to_ne_bytes),
            Dtype::Int8 => extend_bytes!(i8, |v: i8| v.to_ne_bytes()),
            Dtype::Int16 => extend_bytes!(i16, i16::to_ne_bytes),
            Dtype::Int32 => extend_bytes!(i32, i32::to_ne_bytes),
            Dtype::Int64 => extend_bytes!(i64, i64::to_ne_bytes),
            Dtype::Float16 => extend_bytes!(half::f16, |v: half::f16| v.to_bits().to_ne_bytes()),
            Dtype::Float32 => extend_bytes!(f32, f32::to_ne_bytes),
            Dtype::Float64 => extend_bytes!(f64, f64::to_ne_bytes),
            Dtype::Bfloat16 => extend_bytes!(half::bf16, |v: half::bf16| v.to_bits().to_ne_bytes()),
            Dtype::Complex64 => extend_bytes!(complex64, |v: complex64| {
                let mut out = [0_u8; 8];
                out[..4].copy_from_slice(&v.re.to_ne_bytes());
                out[4..].copy_from_slice(&v.im.to_ne_bytes());
                out
            }),
        }
        bytes
    }

    /// Compare two evaluated arrays for equal dtype, shape, and values.
    pub fn equal_values(&self, other: &Self) -> bool {
        if self.as_array().dtype() != other.as_array().dtype()
            || self.as_array().shape() != other.as_array().shape()
        {
            return false;
        }

        macro_rules! eq_slice {
            ($ty:ty) => {{
                self.host_values::<$ty>()
                    .expect("invalid evaluated host data")
                    .eq(other
                        .host_values::<$ty>()
                        .expect("invalid evaluated host data"))
            }};
        }

        match self.as_array().dtype() {
            Dtype::Bool => eq_slice!(bool),
            Dtype::Uint8 => eq_slice!(u8),
            Dtype::Uint16 => eq_slice!(u16),
            Dtype::Uint32 => eq_slice!(u32),
            Dtype::Uint64 => eq_slice!(u64),
            Dtype::Int8 => eq_slice!(i8),
            Dtype::Int16 => eq_slice!(i16),
            Dtype::Int32 => eq_slice!(i32),
            Dtype::Int64 => eq_slice!(i64),
            Dtype::Float16 => eq_slice!(half::f16),
            Dtype::Float32 => eq_slice!(f32),
            Dtype::Float64 => eq_slice!(f64),
            Dtype::Bfloat16 => eq_slice!(half::bf16),
            Dtype::Complex64 => eq_slice!(crate::complex64),
        }
    }
}

#[cfg(test)]
pub(crate) fn eval_vec<T>(array: &Array) -> Vec<T>
where
    T: ArrayElement + Copy,
{
    array.evaluated().unwrap().try_to_vec::<T>().unwrap()
}

#[cfg(test)]
pub(crate) fn eval_equal_values(lhs: &Array, rhs: &Array) -> bool {
    let lhs = lhs.evaluated().unwrap();
    let rhs = rhs.evaluated().unwrap();
    lhs.equal_values(&rhs)
}

/// Stop gradients from being computed.
///
/// The operation is the identity but it prevents gradients from flowing
/// through the array.
pub fn stop_gradient(
    a: impl AsRef<Array>,
    stream: impl AsRef<Stream>,
) -> crate::error::Result<Array> {
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_stop_gradient(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

impl From<bool> for Array {
    fn from(value: bool) -> Self {
        Array::from_bool(value)
    }
}

impl From<i32> for Array {
    fn from(value: i32) -> Self {
        Array::from_int(value)
    }
}

impl From<f32> for Array {
    fn from(value: f32) -> Self {
        Array::from_f32(value)
    }
}

impl From<complex64> for Array {
    fn from(value: complex64) -> Self {
        Array::from_complex(value)
    }
}

impl<T> From<T> for Array
where
    Array: FromNested<T>,
{
    fn from(value: T) -> Self {
        Array::from_nested(value)
    }
}

impl AsRef<Array> for Array {
    fn as_ref(&self) -> &Array {
        self
    }
}

/// A helper trait to construct `Array` from scalar values.
///
/// This trait is intended to be used with the macro [`crate::array!`] but can be used directly if needed.
pub trait FromScalar<T>
where
    T: ArrayElement,
{
    /// Create an array from a scalar value.
    fn from_scalar(val: T) -> Array;
}

impl FromScalar<bool> for Array {
    fn from_scalar(val: bool) -> Array {
        Array::from_bool(val)
    }
}

impl FromScalar<i32> for Array {
    fn from_scalar(val: i32) -> Array {
        Array::from_int(val)
    }
}

impl FromScalar<f32> for Array {
    fn from_scalar(val: f32) -> Array {
        Array::from_f32(val)
    }
}

impl FromScalar<complex64> for Array {
    fn from_scalar(val: complex64) -> Array {
        Array::from_complex(val)
    }
}

/// A helper trait to construct `Array` from nested arrays or slices.
///
/// Given that this is not intended for use other than the macro [`crate::array!`], this trait is added
/// instead of directly implementing `From` for `Array` to avoid conflicts with other `From`
/// implementations.
///
/// This trait is an implementation detail of [`crate::array!`]; applications
/// should construct values through the macro rather than implementing it.
pub trait FromNested<T> {
    /// Create an array from nested arrays or slices.
    fn from_nested(data: T) -> Array;
}

impl<T: FromSliceElement> FromNested<&[T]> for Array {
    fn from_nested(data: &[T]) -> Self {
        Array::from_slice(data, &[data.len() as i32])
    }
}

impl<T: FromSliceElement, const N: usize> FromNested<[T; N]> for Array {
    fn from_nested(data: [T; N]) -> Self {
        Array::from_slice(&data, &[N as i32])
    }
}

impl<T: FromSliceElement, const N: usize> FromNested<&[T; N]> for Array {
    fn from_nested(data: &[T; N]) -> Self {
        Array::from_slice(data, &[N as i32])
    }
}

impl<T: FromSliceElement + Copy> FromNested<&[&[T]]> for Array {
    fn from_nested(data: &[&[T]]) -> Self {
        // check that all rows have the same length
        let row_len = data[0].len();
        assert!(
            data.iter().all(|row| row.len() == row_len),
            "Rows must have the same length"
        );

        let shape = [data.len() as i32, row_len as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter())
            .copied()
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize> FromNested<[&[T]; N]> for Array {
    fn from_nested(data: [&[T]; N]) -> Self {
        // check that all rows have the same length
        let row_len = data[0].len();
        assert!(
            data.iter().all(|row| row.len() == row_len),
            "Rows must have the same length"
        );

        let shape = [N as i32, row_len as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter())
            .copied()
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize> FromNested<&[[T; N]]> for Array {
    fn from_nested(data: &[[T; N]]) -> Self {
        let shape = [data.len() as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().copied())
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize> FromNested<&[&[T; N]]> for Array {
    fn from_nested(data: &[&[T; N]]) -> Self {
        let shape = [data.len() as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().copied())
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize, const M: usize> FromNested<[[T; N]; M]> for Array {
    fn from_nested(data: [[T; N]; M]) -> Self {
        let shape = [M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().copied())
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize, const M: usize> FromNested<&[[T; N]; M]>
    for Array
{
    fn from_nested(data: &[[T; N]; M]) -> Self {
        let shape = [M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().copied())
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize, const M: usize> FromNested<&[&[T; N]; M]>
    for Array
{
    fn from_nested(data: &[&[T; N]; M]) -> Self {
        let shape = [M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().copied())
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy> FromNested<&[&[&[T]]]> for Array {
    fn from_nested(data: &[&[&[T]]]) -> Self {
        // check that 2nd dimension has the same length
        let len_2d = data[0].len();
        assert!(
            data.iter().all(|x| x.len() == len_2d),
            "2nd dimension must have the same length"
        );

        // check that 3rd dimension has the same length
        let len_3d = data[0][0].len();
        assert!(
            data.iter().all(|x| x.iter().all(|y| y.len() == len_3d)),
            "3rd dimension must have the same length"
        );

        let shape = [data.len() as i32, len_2d as i32, len_3d as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize> FromNested<[&[&[T]]; N]> for Array {
    fn from_nested(data: [&[&[T]]; N]) -> Self {
        // check that 2nd dimension has the same length
        let len_2d = data[0].len();
        assert!(
            data.iter().all(|x| x.len() == len_2d),
            "2nd dimension must have the same length"
        );

        // check that 3rd dimension has the same length
        let len_3d = data[0][0].len();
        assert!(
            data.iter().all(|x| x.iter().all(|y| y.len() == len_3d)),
            "3rd dimension must have the same length"
        );

        let shape = [N as i32, len_2d as i32, len_3d as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize> FromNested<&[[&[T]; N]]> for Array {
    fn from_nested(data: &[[&[T]; N]]) -> Self {
        // check that 3rd dimension has the same length
        let len_3d = data[0][0].len();
        assert!(
            data.iter().all(|x| x.iter().all(|y| y.len() == len_3d)),
            "3rd dimension must have the same length"
        );

        let shape = [data.len() as i32, N as i32, len_3d as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize> FromNested<&[&[[T; N]]]> for Array {
    fn from_nested(data: &[&[[T; N]]]) -> Self {
        // check that 2nd dimension has the same length
        let len_2d = data[0].len();
        assert!(
            data.iter().all(|x| x.len() == len_2d),
            "2nd dimension must have the same length"
        );

        let shape = [data.len() as i32, len_2d as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize, const M: usize> FromNested<[[&[T]; N]; M]>
    for Array
{
    fn from_nested(data: [[&[T]; N]; M]) -> Self {
        // check that 3rd dimension has the same length
        let len_3d = data[0][0].len();
        assert!(
            data.iter().all(|x| x.iter().all(|y| y.len() == len_3d)),
            "3rd dimension must have the same length"
        );

        let shape = [M as i32, N as i32, len_3d as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize, const M: usize> FromNested<&[[&[T]; N]; M]>
    for Array
{
    fn from_nested(data: &[[&[T]; N]; M]) -> Self {
        // check that 3rd dimension has the same length
        let len_3d = data[0][0].len();
        assert!(
            data.iter().all(|x| x.iter().all(|y| y.len() == len_3d)),
            "3rd dimension must have the same length"
        );

        let shape = [M as i32, N as i32, len_3d as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize, const M: usize> FromNested<&[&[[T; N]]; M]>
    for Array
{
    fn from_nested(data: &[&[[T; N]]; M]) -> Self {
        // check that 2nd dimension has the same length
        let len_2d = data[0].len();
        assert!(
            data.iter().all(|x| x.len() == len_2d),
            "2nd dimension must have the same length"
        );

        let shape = [M as i32, len_2d as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize, const M: usize, const O: usize>
    FromNested<[[[T; N]; M]; O]> for Array
{
    fn from_nested(data: [[[T; N]; M]; O]) -> Self {
        let shape = [O as i32, M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize, const M: usize, const O: usize>
    FromNested<&[[[T; N]; M]; O]> for Array
{
    fn from_nested(data: &[[[T; N]; M]; O]) -> Self {
        let shape = [O as i32, M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize, const M: usize, const O: usize>
    FromNested<&[&[[T; N]; M]; O]> for Array
{
    fn from_nested(data: &[&[[T; N]; M]; O]) -> Self {
        let shape = [O as i32, M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize, const M: usize, const O: usize>
    FromNested<&[[&[T; N]; M]; O]> for Array
{
    fn from_nested(data: &[[&[T; N]; M]; O]) -> Self {
        let shape = [O as i32, M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

impl<T: FromSliceElement + Copy, const N: usize, const M: usize, const O: usize>
    FromNested<&[&[&[T; N]; M]; O]> for Array
{
    fn from_nested(data: &[&[&[T; N]; M]; O]) -> Self {
        let shape = [O as i32, M as i32, N as i32];
        let data = data
            .iter()
            .flat_map(|x| x.iter().flat_map(|y| y.iter().copied()))
            .collect::<Vec<T>>();
        Array::from_slice(&data, &shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_scalar_array_from_bool() {
        let stream = crate::test_stream();
        let array = Array::from_bool(true);
        assert_eq!(array.item_size(), 1);
        assert_eq!(array.size(), 1);
        assert!(array.strides().is_empty());
        assert_eq!(array.nbytes(), 1);
        assert_eq!(array.ndim(), 0);
        assert!(array.shape().is_empty());
        assert_eq!(array.dtype(), Dtype::Bool);
        assert!(array.item::<bool>(&stream));
    }

    #[test]
    fn new_scalar_array_from_int() {
        let stream = crate::test_stream();
        let array = Array::from_int(42);
        assert_eq!(array.item_size(), 4);
        assert_eq!(array.size(), 1);
        assert!(array.strides().is_empty());
        assert_eq!(array.nbytes(), 4);
        assert_eq!(array.ndim(), 0);
        assert!(array.shape().is_empty());
        assert_eq!(array.dtype(), Dtype::Int32);
        assert_eq!(array.item::<i32>(&stream), 42);
    }

    #[test]
    fn new_scalar_array_from_f32() {
        let stream = crate::test_stream();
        let array = Array::from_f32(3.14);
        assert_eq!(array.item_size(), 4);
        assert_eq!(array.size(), 1);
        assert!(array.strides().is_empty());
        assert_eq!(array.nbytes(), 4);
        assert_eq!(array.ndim(), 0);
        assert!(array.shape().is_empty());
        assert_eq!(array.dtype(), Dtype::Float32);
        assert_eq!(array.item::<f32>(&stream), 3.14);
    }

    #[test]
    fn new_scalar_array_from_f64() {
        let stream = crate::test_stream();
        let array = Array::from_f64(3.14)
            .as_dtype(Dtype::Float64, stream)
            .unwrap();
        assert_eq!(array.item_size(), 8);
        assert_eq!(array.size(), 1);
        assert!(array.strides().is_empty());
        assert_eq!(array.nbytes(), 8);
        assert_eq!(array.ndim(), 0);
        assert!(array.shape().is_empty());
        assert_eq!(array.dtype(), Dtype::Float64);
        float_eq::assert_float_eq!(array.item::<f64>(&stream), 3.14, abs <= 1e-5);
    }

    #[test]
    fn new_array_from_slice_f64() {
        let array = Array::from_slice_f64(&[1.0, 2.0, 3.0], &[3]);
        assert_eq!(array.item_size(), 8);
        assert_eq!(array.size(), 3);
        assert_eq!(array.strides(), &[1]);
        assert_eq!(array.nbytes(), 24);
        assert_eq!(array.ndim(), 1);
        assert_eq!(array.dim(0), 3);
        assert_eq!(array.shape(), &[3]);
        assert_eq!(array.dtype(), Dtype::Float64);
    }

    #[test]
    fn new_scalar_array_from_complex() {
        let stream = crate::test_stream();
        let val = complex64::new(1.0, 2.0);
        let array = Array::from_complex(val);
        assert_eq!(array.item_size(), 8);
        assert_eq!(array.size(), 1);
        assert!(array.strides().is_empty());
        assert_eq!(array.nbytes(), 8);
        assert_eq!(array.ndim(), 0);
        assert!(array.shape().is_empty());
        assert_eq!(array.dtype(), Dtype::Complex64);
        assert_eq!(array.item::<complex64>(&stream), val);
    }

    #[test]
    fn new_array_from_single_element_slice() {
        let data = [1i32];
        let array = Array::from_slice(&data, &[1]);
        assert_eq!(array.item_size(), 4);
        assert_eq!(array.size(), 1);
        assert_eq!(array.strides(), &[1]);
        assert_eq!(array.nbytes(), 4);
        assert_eq!(array.ndim(), 1);
        assert_eq!(array.dim(0), 1);
        assert_eq!(array.shape(), &[1]);
        assert_eq!(array.dtype(), Dtype::Int32);
        assert_eq!(array.evaluated().unwrap().as_slice::<i32>(), &data[..]);
    }

    #[test]
    fn half_precision_scalar_and_data_access_preserve_bits() {
        let stream = crate::test_stream();

        let f16_value = half::f16::from_bits(0x3555);
        let f16_array = Array::from_slice(&[f16_value], &[]);
        assert_eq!(f16_array.clone().item::<half::f16>(&stream), f16_value);
        let f16_clone = f16_array.deep_clone().unwrap().into_evaluated().unwrap();
        assert_eq!(f16_clone.as_slice::<half::f16>(), &[f16_value]);

        let bf16_value = half::bf16::from_bits(0x3eab);
        let bf16_array = Array::from_slice(&[bf16_value], &[]);
        assert_eq!(bf16_array.clone().item::<half::bf16>(&stream), bf16_value);
        let bf16_clone = bf16_array.deep_clone().unwrap().into_evaluated().unwrap();
        assert_eq!(bf16_clone.as_slice::<half::bf16>(), &[bf16_value]);
    }

    #[test]
    fn new_array_from_multi_element_slice() {
        let data = [1i32, 2, 3, 4, 5];
        let array = Array::from_slice(&data, &[5]);
        assert_eq!(array.item_size(), 4);
        assert_eq!(array.size(), 5);
        assert_eq!(array.strides(), &[1]);
        assert_eq!(array.nbytes(), 20);
        assert_eq!(array.ndim(), 1);
        assert_eq!(array.dim(0), 5);
        assert_eq!(array.shape(), &[5]);
        assert_eq!(array.dtype(), Dtype::Int32);
        assert_eq!(array.evaluated().unwrap().as_slice::<i32>(), &data[..]);
    }

    #[test]
    fn evaluated_array_copies_to_owned_vector() {
        let data = [1.25f32, -2.5, 3.75];
        let array = Array::from_slice(&data, &[3]);
        assert_eq!(
            array.evaluated().unwrap().try_to_vec::<f32>(),
            Ok(data.to_vec())
        );
    }

    #[test]
    fn owned_typed_data_and_native_bytes_round_trip() {
        let values = vec![0x0102_0304u32, u32::MAX];
        let array = Array::try_from_owned_data(values.clone(), &[2]).unwrap();
        let evaluated = array.evaluated().unwrap();
        assert_eq!(evaluated.as_slice::<u32>(), values);
        assert_eq!(
            evaluated.to_native_bytes(),
            values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn new_2d_array_from_slice() {
        let data = [1i32, 2, 3, 4, 5, 6];
        let array = Array::from_slice(&data, &[2, 3]);
        assert_eq!(array.item_size(), 4);
        assert_eq!(array.size(), 6);
        assert_eq!(array.strides(), &[3, 1]);
        assert_eq!(array.nbytes(), 24);
        assert_eq!(array.ndim(), 2);
        assert_eq!(array.dim(0), 2);
        assert_eq!(array.dim(1), 3);
        assert_eq!(array.dim(-1), 3); // negative index
        assert_eq!(array.dim(-2), 2); // negative index
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype(), Dtype::Int32);
        assert_eq!(array.evaluated().unwrap().as_slice::<i32>(), &data[..]);
    }

    #[test]
    fn deep_cloned_array_has_different_ptr() {
        let data = [1i32, 2, 3, 4, 5];
        let orig = Array::from_slice(&data, &[5]);
        let clone = orig.clone().deep_clone().unwrap();
        let orig = orig.evaluated().unwrap();
        let clone = clone.evaluated().unwrap();

        // Data should be the same
        assert_eq!(orig.as_slice::<i32>(), clone.as_slice::<i32>());

        // Addr of `mlx_array` should be different
        assert_ne!(orig.as_array().as_ptr().ctx, clone.as_array().as_ptr().ctx);

        // Addr of data should be different
        assert_ne!(
            orig.as_slice::<i32>().as_ptr(),
            clone.as_slice::<i32>().as_ptr()
        );
    }

    #[test]
    fn deep_clone_preserves_logical_order_of_strided_and_broadcast_arrays() {
        use crate::ops::indexing::TryIndexOp;
        let stream = Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0));
        let storage = Array::from_slice(&(0..2 * 5 * 3).collect::<Vec<i32>>(), &[2, 5, 3]);
        let padded = storage.try_index_device((.., ..2, ..), &stream).unwrap();
        let transposed = storage.transpose_axes(&[2, 0, 1], &stream).unwrap();
        let broadcast = crate::ops::broadcast_to(
            Array::from_slice(&[7i32, 11, 13], &[1, 1, 3]),
            &[2, 4, 3],
            &stream,
        )
        .unwrap();
        let expected = [
            vec![0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20],
            (0..3)
                .flat_map(|channel| {
                    (0..2).flat_map(move |batch| {
                        (0..5).map(move |token| batch * 15 + token * 3 + channel)
                    })
                })
                .collect(),
            [7, 11, 13].repeat(8),
        ];
        for (source, expected) in [padded, transposed, broadcast].into_iter().zip(expected) {
            let shape = source.shape().to_vec();
            let evaluated = source.evaluated().unwrap();
            let copied = evaluated.deep_clone().unwrap();
            assert_eq!(copied.as_array().shape(), shape);
            assert_eq!(copied.as_slice::<i32>(), expected);
            let copied = source.deep_clone().unwrap();
            assert_eq!(copied.evaluated().unwrap().as_slice::<i32>(), expected);
        }
    }

    #[test]
    fn allocation_info_preserves_completed_backing_capacity_and_alias_identity() {
        use crate::ops::indexing::TryIndexOp;
        let stream = Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0));
        let root = Array::from_slice(&(0..120).map(|n| n as f32).collect::<Vec<_>>(), &[2, 20, 3]);
        let root_info = root.allocation_info().unwrap().unwrap();
        assert!(root_info.bytes() >= root.nbytes());
        let tail = root.try_index_device((.., 18.., ..), &stream).unwrap();
        assert_eq!(
            tail.allocation_info().unwrap(),
            None,
            "facts must not evaluate a lazy view"
        );
        let evaluated = tail.evaluated().unwrap();
        assert_eq!(tail.allocation_info().unwrap(), Some(root_info));
        assert!(root_info.bytes() > tail.nbytes());
        let clone = evaluated.deep_clone().unwrap();
        let cloned_info = clone.as_array().allocation_info().unwrap().unwrap();
        assert_ne!(root_info.identity(), cloned_info.identity());
        assert!(cloned_info.bytes() >= tail.nbytes());
        let expected = (54..60)
            .chain(114..120)
            .map(|n| n as f32)
            .collect::<Vec<_>>();
        assert_eq!(clone.as_slice::<f32>(), expected);
        let alias = root.clone();
        drop(root);
        assert_eq!(alias.allocation_info().unwrap(), Some(root_info));
        assert_eq!(tail.allocation_info().unwrap(), Some(root_info));
    }

    #[test]
    fn allocation_info_keeps_custom_owned_buffers_unknown_without_touching_them() {
        let values = (0..32).map(|n| n as f32 * 0.25).collect::<Vec<_>>();
        let transferred = values.clone();
        let source = transferred.as_ptr();
        let foreign = Array::try_from_owned_data(transferred, &[4, 8]).unwrap();
        let before = foreign.allocation_info().unwrap();
        let evaluated = foreign.evaluated().unwrap();
        assert_eq!(evaluated.as_slice::<f32>(), values);
        if evaluated.as_slice::<f32>().as_ptr() == source {
            assert_eq!(
                before, None,
                "adopted foreign storage has no allocator size bound"
            );
        } else {
            assert!(
                before.unwrap().bytes() >= foreign.nbytes(),
                "copy fallback owns native storage"
            );
        }
        assert_eq!(foreign.allocation_info().unwrap(), before);
    }

    #[test]
    fn test_array_eq() {
        let data = [1i32, 2, 3, 4, 5];
        let array1 = Array::from_slice(&data, &[5]);
        let array2 = Array::from_slice(&data, &[5]);
        let array3 = Array::from_slice(&[1i32, 2, 3, 4, 6], &[5]);

        let array1 = array1.evaluated().unwrap();
        let array2 = array2.evaluated().unwrap();
        let array3 = array3.evaluated().unwrap();

        assert!(array1.equal_values(&array2));
        assert!(!array1.equal_values(&array3));
    }

    #[test]
    fn test_array_item_non_scalar() {
        let stream = crate::test_stream();
        let data = [1i32, 2, 3, 4, 5];
        let array = Array::from_slice(&data, &[5]);
        assert!(array.try_item::<i32>(&stream).is_err());
    }

    #[test]
    fn test_item_type_conversion() {
        let stream = crate::test_stream();
        let array = Array::from_f32(1.0);
        assert_eq!(array.clone().item::<i32>(&stream), 1);
        assert_eq!(
            array.clone().item::<complex64>(&stream),
            complex64::new(1.0, 0.0)
        );
        assert_eq!(array.clone().item::<u8>(&stream), 1);
        assert_eq!(array.evaluated().unwrap().as_slice::<f32>(), &[1.0]);
    }
}
