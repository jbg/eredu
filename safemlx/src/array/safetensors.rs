//! Implement conversion from safetensors `TensorView` to `Array`.
//!
//! MLX represents checkpoint FP8 and FP4 values as integer arrays containing
//! their encoded bit patterns. Accordingly, `F8_E4M3`, `F8_E8M0`, and `F4`
//! tensor views are loaded losslessly as [`Dtype::Uint8`]. The last dimension
//! of an F4 storage array is half its logical scalar dimension because two
//! values occupy each byte. `F8_E5M2` is not supported.

use std::ffi::c_void;

use bytemuck::cast_slice;
use half::slice::HalfFloatSliceExt;
use safetensors::tensor::TensorView;

use crate::{error::ConversionError, Dtype};

use super::{Array, EvaluatedArray};

impl<'data> TryFrom<TensorView<'data>> for Array {
    type Error = ConversionError;

    fn try_from(value: TensorView<'data>) -> Result<Self, Self::Error> {
        let stored_dtype = value.dtype();
        let dtype = match stored_dtype {
            // MLX's FP8 conversion and kernels use uint8 arrays as the storage
            // type for E4M3. Keep the checkpoint bytes unchanged so callers can
            // pass the array directly to those operations.
            safetensors::Dtype::F8_E4M3
            | safetensors::Dtype::F8_E8M0
            | safetensors::Dtype::F4 => Dtype::Uint8,
            dtype => dtype.try_into()?,
        };
        let mut shape = value.shape()
            .iter()
            .map(|x| *x as i32)
            .collect::<Vec<_>>();
        adjust_packed_shape(stored_dtype, &mut shape)?;

        let data = value.data();

        unsafe {
            Ok(Array::from_raw_data(data.as_ptr() as *const c_void, &shape, dtype))
        }
    }
}

impl Array {
    /// Creates a host array that borrows a SafeTensors view without copying.
    ///
    /// # Safety
    ///
    /// The view's backing bytes must remain valid and immutable until every
    /// clone of the returned array has been dropped or every dependent graph
    /// has completed evaluation.
    pub unsafe fn try_from_borrowed_safetensors(
        value: TensorView<'_>,
    ) -> Result<Self, ConversionError> {
        let stored_dtype = value.dtype();
        let dtype = match stored_dtype {
            safetensors::Dtype::F8_E4M3
            | safetensors::Dtype::F8_E8M0
            | safetensors::Dtype::F4 => Dtype::Uint8,
            dtype => dtype.try_into()?,
        };
        let mut shape = value
            .shape()
            .iter()
            .map(|dimension| {
                i32::try_from(*dimension)
                    .map_err(|_| crate::error::Exception::custom("shape is too large").into())
            })
            .collect::<Result<Vec<_>, ConversionError>>()?;
        adjust_packed_shape(stored_dtype, &mut shape)?;
        Ok(unsafe {
            Self::try_from_borrowed_host_data(value.data().as_ptr().cast(), &shape, dtype)?
        })
    }
}

fn adjust_packed_shape(
    dtype: safetensors::Dtype,
    shape: &mut [i32],
) -> Result<(), ConversionError> {
    if dtype != safetensors::Dtype::F4 {
        return Ok(());
    }
    let last = shape.last_mut().ok_or_else(|| {
        crate::error::Exception::custom("F4 safetensors scalar storage is unsupported")
    })?;
    if *last % 2 != 0 {
        return Err(crate::error::Exception::custom(
            "F4 safetensors final dimension must be even for packed storage",
        )
        .into());
    }
    *last /= 2;
    Ok(())
}

impl EvaluatedArray<'_> {
    /// Borrows the bytes of a row-contiguous array as a little-endian tensor.
    pub fn safetensors_view(&self) -> Result<TensorView<'_>, ConversionError> {
        if cfg!(target_endian = "big") {
            return Err(crate::error::Exception::custom(
                "borrowing SafeTensors data requires little-endian native storage",
            )
            .into());
        }
        let array = self.as_array();
        let dtype = array.dtype().try_into()?;
        let shape = array
            .shape()
            .iter()
            .map(|&dimension| dimension as usize)
            .collect();
        let bytes = match array.dtype() {
            Dtype::Bool => cast_slice(self.try_as_slice::<bool>()?),
            Dtype::Uint8 => self.try_as_slice::<u8>()?,
            Dtype::Uint16 => cast_slice(self.try_as_slice::<u16>()?),
            Dtype::Uint32 => cast_slice(self.try_as_slice::<u32>()?),
            Dtype::Uint64 => cast_slice(self.try_as_slice::<u64>()?),
            Dtype::Int8 => cast_slice(self.try_as_slice::<i8>()?),
            Dtype::Int16 => cast_slice(self.try_as_slice::<i16>()?),
            Dtype::Int32 => cast_slice(self.try_as_slice::<i32>()?),
            Dtype::Int64 => cast_slice(self.try_as_slice::<i64>()?),
            Dtype::Float16 => cast_slice(self.try_as_slice::<half::f16>()?.reinterpret_cast()),
            Dtype::Float32 => cast_slice(self.try_as_slice::<f32>()?),
            Dtype::Float64 => cast_slice(self.try_as_slice::<f64>()?),
            Dtype::Bfloat16 => cast_slice(self.try_as_slice::<half::bf16>()?.reinterpret_cast()),
            Dtype::Complex64 => return Err(ConversionError::MlxDtype(Dtype::Complex64)),
        };
        Ok(TensorView::new(dtype, shape, bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::{array, complex64, Array, Dtype};
    use safetensors::tensor::TensorView;

    // Helper macro to test conversion between Array and TensorView
    macro_rules! assert_conversion {
        ($arr:expr, $dtype:expr) => {
            let stream = crate::Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0));
            let arr = $arr.as_dtype($dtype, &stream).unwrap();
            let arr = arr.evaluated().unwrap();
            let tensor = arr.safetensors_view().unwrap();
            let arr2 = Array::try_from(tensor).unwrap();
            let arr2 = arr2.evaluated().unwrap();

            assert!(arr.equal_values(&arr2));
        };
    }

    #[test]
    fn test_conversion_bool() {
        let arr = array!([[true, false, true], [false, true, false]]);
        assert_conversion!(&arr, crate::Dtype::Bool);
    }

    #[test]
    fn test_conversion_float32() {
        let arr = array!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_conversion!(&arr, crate::Dtype::Float32);
    }

    #[test]
    fn test_conversion_complex64() {
        let stream = crate::Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0));
        let arr = array!([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            .as_type::<complex64>(&stream)
            .unwrap();
        let arr = arr.evaluated().unwrap();
        let tensor = arr.safetensors_view();
        assert!(tensor.is_err());
    }

    #[test]
    fn test_conversion_float8_e4m3_preserves_encoded_bytes() {
        let encoded = [0x00, 0x38, 0xb8, 0x7e, 0xff];
        let tensor = TensorView::new(
            safetensors::Dtype::F8_E4M3,
            vec![1, encoded.len()],
            &encoded,
        )
        .unwrap();

        let stream = crate::Stream::new_with_device(&crate::Device::new(crate::DeviceType::Cpu, 0));
        let array = Array::try_from(tensor).unwrap().copy(&stream).unwrap();
        let array = array.evaluated().unwrap();

        assert_eq!(array.as_array().dtype(), Dtype::Uint8);
        assert_eq!(array.as_array().shape(), &[1, encoded.len() as i32]);
        assert_eq!(array.as_slice::<u8>(), &encoded);
    }

    #[test]
    fn test_conversion_float8_e5m2_remains_unsupported() {
        let encoded = [0x00];
        let tensor = TensorView::new(safetensors::Dtype::F8_E5M2, vec![1], &encoded).unwrap();

        assert!(Array::try_from(tensor).is_err());
    }

    #[test]
    fn test_conversion_float8_e8m0_preserves_encoded_bytes() {
        let encoded = [0x00, 0x7f, 0x80, 0xff];
        let tensor = TensorView::new(
            safetensors::Dtype::F8_E8M0,
            vec![2, 2],
            &encoded,
        )
        .unwrap();
        let converted = Array::try_from(tensor).unwrap();
        let array = converted.evaluated().unwrap();
        assert_eq!(array.as_array().dtype(), Dtype::Uint8);
        assert_eq!(array.as_array().shape(), &[2, 2]);
        assert_eq!(array.as_slice::<u8>(), &encoded);
    }

    #[test]
    fn test_conversion_float4_uses_physical_packed_shape() {
        let encoded = [0x10, 0x32, 0x54, 0x76];
        let tensor = TensorView::new(safetensors::Dtype::F4, vec![2, 4], &encoded).unwrap();
        let converted = Array::try_from(tensor).unwrap();
        let array = converted.evaluated().unwrap();
        assert_eq!(array.as_array().dtype(), Dtype::Uint8);
        assert_eq!(array.as_array().shape(), &[2, 2]);
        assert_eq!(array.as_slice::<u8>(), &encoded);
    }
}
