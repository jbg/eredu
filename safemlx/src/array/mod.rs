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

cfg_safetensors! {
    mod safetensors;
}

pub use element::ArrayElement;

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

// SAFETY: `Array` owns an MLX C++ array handle. MLX arrays are immutable graph
// values from the Rust API's perspective; mutation is represented by producing
// new arrays. Runtime entry points that touch known MLX global state are guarded
// inside safemlx, and callers still provide explicit streams for execution.
unsafe impl Send for Array {}

// SAFETY: Shared references to `Array` expose only immutable graph/value
// operations. Concurrent evaluation and host-read paths are covered by runtime
// guards where MLX requires them, and stress-tested in safemlx-tests.
unsafe impl Sync for Array {}

unsafe extern "C" fn drop_owned_vec<T>(payload: *mut c_void) {
    unsafe {
        drop(Box::from_raw(payload.cast::<Vec<T>>()));
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

    /// Transfers an owned, typed host allocation into an MLX array without copying it.
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

    /// The strides of the array.
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

    /// The number of bytes in the array.
    pub fn nbytes(&self) -> usize {
        unsafe { safemlx_sys::mlx_array_nbytes(self.as_ptr()) }
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
        let _guard = runtime_lock::enter();
        <() as Guarded>::try_from_op(|_| unsafe { safemlx_sys::mlx_array_eval(self.as_ptr()) })?;
        Ok(EvaluatedArray {
            storage: EvaluatedArrayStorage::Borrowed(self),
        })
    }

    /// Evaluate the array and return an owned host-readable value.
    pub fn into_evaluated(self) -> crate::error::Result<EvaluatedArray<'static>> {
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
    /// This is unsafe because the underlying data ptr is not checked for null or if the desired
    /// dtype matches the actual dtype of the array.
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

    /// Returns a slice of the array data returning an error if the dtype does not match the actual dtype.
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

        unsafe {
            let size = array.size();
            let data = T::array_data(array);
            if data.is_null() || size == 0 {
                return Err(AsSliceError::Null);
            }
            if !data.is_aligned() {
                return Err(AsSliceError::Misaligned);
            }
            if size > isize::MAX as usize / std::mem::size_of::<T>() {
                return Err(AsSliceError::TooLarge);
            }

            Ok(std::slice::from_raw_parts(data, size))
        }
    }

    /// Returns a slice of the array data.
    /// # Panics
    ///
    /// Panics if the array is not evaluated or if the desired dtype does not match the actual dtype
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

    /// Copies evaluated array data into an owned vector.
    ///
    /// Unlike [`Self::try_as_slice`], this accepts MLX views whose data pointer
    /// is not aligned for a Rust slice. Each element is copied with an
    /// unaligned read instead of constructing a reference to that storage.
    pub fn try_to_vec<T: ArrayElement + Copy>(&self) -> Result<Vec<T>, AsSliceError> {
        let array = self.as_array();
        if array.dtype() != T::DTYPE {
            return Err(AsSliceError::DtypeMismatch {
                expecting: T::DTYPE,
                found: array.dtype(),
            });
        }

        let size = array.size();
        if size == 0 {
            return Ok(Vec::new());
        }
        if size > isize::MAX as usize / std::mem::size_of::<T>() {
            return Err(AsSliceError::TooLarge);
        }
        let data = T::array_data(array);
        if data.is_null() {
            return Err(AsSliceError::Null);
        }

        let mut values = Vec::with_capacity(size);
        for index in 0..size {
            // SAFETY: MLX reports `size` initialized elements at `data` after
            // evaluation. `read_unaligned` does not create a reference to the
            // possibly unaligned foreign storage.
            values.push(unsafe { std::ptr::read_unaligned(data.add(index)) });
        }
        Ok(values)
    }

    /// Clone the array by copying the data.
    ///
    /// This is named `deep_clone` to avoid confusion with the `Clone` trait.
    pub fn deep_clone(&self) -> crate::error::Result<EvaluatedArray<'static>> {
        unsafe {
            let array = self.as_array();
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

impl Clone for Array {
    fn clone(&self) -> Self {
        Array::try_from_op(|res| unsafe { safemlx_sys::mlx_array_set(res, self.as_ptr()) })
            // Exception may be thrown when calling `new` in cpp.
            .expect("Failed to clone array")
    }
}

impl EvaluatedArray<'_> {
    /// Copies the evaluated storage into its native-endian byte representation.
    pub fn to_native_bytes(&self) -> Vec<u8> {
        macro_rules! extend_bytes {
            ($values:expr) => {{
                let values = $values;
                let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
                for value in values {
                    bytes.extend_from_slice(&value.to_ne_bytes());
                }
                bytes
            }};
        }

        match self.as_array().dtype() {
            Dtype::Bool => self
                .as_slice::<bool>()
                .iter()
                .map(|value| u8::from(*value))
                .collect(),
            Dtype::Uint8 => self.as_slice::<u8>().to_vec(),
            Dtype::Uint16 => extend_bytes!(self.as_slice::<u16>()),
            Dtype::Uint32 => extend_bytes!(self.as_slice::<u32>()),
            Dtype::Uint64 => extend_bytes!(self.as_slice::<u64>()),
            Dtype::Int8 => self
                .as_slice::<i8>()
                .iter()
                .map(|value| *value as u8)
                .collect(),
            Dtype::Int16 => extend_bytes!(self.as_slice::<i16>()),
            Dtype::Int32 => extend_bytes!(self.as_slice::<i32>()),
            Dtype::Int64 => extend_bytes!(self.as_slice::<i64>()),
            Dtype::Float16 => self
                .as_slice::<half::f16>()
                .iter()
                .flat_map(|value| value.to_bits().to_ne_bytes())
                .collect(),
            Dtype::Float32 => extend_bytes!(self.as_slice::<f32>()),
            Dtype::Float64 => extend_bytes!(self.as_slice::<f64>()),
            Dtype::Bfloat16 => self
                .as_slice::<half::bf16>()
                .iter()
                .flat_map(|value| value.to_bits().to_ne_bytes())
                .collect(),
            Dtype::Complex64 => self
                .as_slice::<complex64>()
                .iter()
                .flat_map(|value| {
                    value
                        .re
                        .to_ne_bytes()
                        .into_iter()
                        .chain(value.im.to_ne_bytes())
                })
                .collect(),
        }
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
                let lhs = self.as_slice::<$ty>();
                let rhs = other.as_slice::<$ty>();
                lhs == rhs
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
    T: ArrayElement + Clone,
{
    array.evaluated().unwrap().as_slice::<T>().to_vec()
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
