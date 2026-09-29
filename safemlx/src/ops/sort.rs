//! Implements bindings for the sorting ops.

use safemlx_internal_macros::generate_macro;

use crate::{error::Result, utils::guard::Guarded, Array, Stream};

/// Returns a sorted copy of the array. Returns an error if the arguments are invalid.
///
/// # Params
///
/// - `array`: input array
/// - `axis`: axis to sort over
///
/// # Example
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
/// use safemlx::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let axis = 0;
/// let result = sort_axis(&a, axis, &stream);
/// ```
#[generate_macro]
pub fn sort_axis(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_sort_axis(res, a.as_ref().as_ptr(), axis, stream.as_ref().as_ptr())
    })
}

/// Returns a sorted copy of the flattened array. Returns an error if the arguments are invalid.
///
/// # Params
///
/// - `array`: input array
///
/// # Example
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
/// use safemlx::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let result = sort(&a, &stream);
/// ```
#[generate_macro]
pub fn sort(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_sort(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Returns the indices that sort the array. Returns an error if the arguments are invalid.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `axis`: axis to sort over
///
/// # Example
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
/// use safemlx::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let axis = 0;
/// let result = argsort_axis(&a, axis, &stream);
/// ```
#[generate_macro]
pub fn argsort_axis(
    a: impl AsRef<Array>,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_argsort_axis(res, a.as_ref().as_ptr(), axis, stream.as_ref().as_ptr())
    })
}

/// Returns the indices that sort the flattened array. Returns an error if the arguments are
/// invalid.
///
/// # Params
///
/// - `a`: The array to sort.
///
/// # Example
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
/// use safemlx::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let result = argsort(&a, &stream);
/// ```
#[generate_macro]
pub fn argsort(a: impl AsRef<Array>, #[optional] stream: impl AsRef<Stream>) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_argsort(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// Returns a partitioned copy of the array such that the smaller `kth` elements are first.
/// Returns an error if the arguments are invalid.
///
/// The ordering of the elements in partitions is undefined.
///
/// # Params
///
/// - `array`: input array
/// - `kth`: Element at the `kth` index will be in its sorted position in the output. All elements
///   before the kth index will be less or equal to the `kth` element and all elements after will be
///   greater or equal to the `kth` element in the output.
/// - `axis`: axis to partition over
///
/// # Example
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
/// use safemlx::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let axis = 0;
/// let result = partition_axis(&a, kth, axis, &stream);
/// ```
#[generate_macro]
pub fn partition_axis(
    a: impl AsRef<Array>,
    kth: i32,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_partition_axis(
            res,
            a.as_ref().as_ptr(),
            kth,
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Returns a partitioned copy of the flattened array such that the smaller `kth` elements are
/// first. Returns an error if the arguments are invalid.
///
/// The ordering of the elements in partitions is undefined.
///
/// # Params
///
/// - `array`: input array
/// - `kth`: Element at the `kth` index will be in its sorted position in the output. All elements
///   before the kth index will be less or equal to the `kth` element and all elements after will be
///   greater or equal to the `kth` element in the output.
///
/// # Example
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
/// use safemlx::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let result = partition(&a, kth, &stream);
/// ```
#[generate_macro]
pub fn partition(
    a: impl AsRef<Array>,
    kth: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_partition(res, a.as_ref().as_ptr(), kth, stream.as_ref().as_ptr())
    })
}

/// Returns the indices that partition the array. Returns an error if the arguments are invalid.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `kth`: element index at the `kth` position in the output will give the sorted position.  All
///   indices before the`kth` position will be of elements less than or equal to the element at the
///   `kth` index and all indices after will be elemenents greater than or equal to the element at
///   the `kth` position.
/// - `axis`: axis to partition over
///
/// # Example
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
/// use safemlx::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let axis = 0;
/// let result = argpartition_axis(&a, kth, axis, &stream);
/// ```
#[generate_macro]
pub fn argpartition_axis(
    a: impl AsRef<Array>,
    kth: i32,
    axis: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_argpartition_axis(
            res,
            a.as_ref().as_ptr(),
            kth,
            axis,
            stream.as_ref().as_ptr(),
        )
    })
}

/// Returns the indices that partition the flattened array. Returns an error if the arguments are
/// invalid.
///
/// The ordering of the elements within a partition in given by the indices is undefined.
///
/// # Params
///
/// - `a`: The array to sort.
/// - `kth`: element index at the `kth` position in the output will give the sorted position.  All
///   indices before the`kth` position will be of elements less than or equal to the element at the
///   `kth` index and all indices after will be elemenents greater than or equal to the element at
///   the `kth` position.
///
/// # Example
///
/// ```rust
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
/// use safemlx::{Array, ops::*};
///
/// let a = Array::from_slice(&[3, 2, 1], &[3]);
/// let kth = 1;
/// let result = argpartition(&a, kth, &stream);
/// ```
#[generate_macro]
pub fn argpartition(
    a: impl AsRef<Array>,
    kth: i32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_argpartition(res, a.as_ref().as_ptr(), kth, stream.as_ref().as_ptr())
    })
}
