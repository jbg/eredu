//! Native realization of reproducible weighted-normalization arithmetic.
use safemlx::{error::Exception, Array, Stream};

/// Learned RMS normalization with an input-precision boundary before the gain.
/// Keep the same rounding order as the native low-precision operator, using
/// the shared float32 reduction to make half-way cases reproducible on Metal.
pub(crate) fn input_precision_rms(
    input: &Array,
    weight: &Array,
    epsilon: f32,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.dtype() == weight.dtype()
        && matches!(
            input.dtype(),
            safemlx::Dtype::Bfloat16 | safemlx::Dtype::Float16
        )
        && input
            .shape()
            .last()
            .is_some_and(|width| weight.shape() == [*width])
    {
        let wide = input.as_dtype(safemlx::Dtype::Float32, stream)?;
        if let Some(normalized) = f32_weightless_rms(&wide, epsilon, stream)? {
            return normalized
                .as_dtype(input.dtype(), stream)?
                .multiply(weight, stream);
        }
    }
    safemlx::fast::rms_norm(input, weight, epsilon, stream)
}

pub(crate) fn f32_weightless_rms(
    input: &Array,
    epsilon: f32,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use crate::backend::managed_memory::row_kernels::{self, RowKernel};
        use safemlx::{DeviceType, Dtype};
        if input.dtype() != Dtype::Float32
            || input.size() == 0
            || input.ndim() == 0
            || stream.device_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        row_kernels::validate_call(RowKernel::Rms, input.ndim())?;
        let width = input.dim(-1);
        let rows = i32::try_from(input.size() / width as usize)
            .map_err(|_| Exception::custom("normalization row count overflow"))?;
        let squared = input.square(stream)?;
        let epsilon = Array::try_from_slice(&[epsilon], &[1])?;
        row_kernels::apply(
            RowKernel::Rms,
            [input, &squared, &epsilon],
            input.shape(),
            rows,
            stream,
        )
        .map(Some)
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, epsilon, stream);
        Ok(None)
    }
}

#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod tests {
    use super::*;
    use safemlx::{Device, DeviceType, Dtype};
    #[test]
    fn native_cascade_normalization_matches_independent_bf16_rounding() {
        let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let fixture = Array::load_safetensors(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/validation/bf16_normalization.safetensors"
            ),
            &weights,
        )
        .unwrap();
        for width in [32, 768, 1280, 2560] {
            let input = fixture[&format!("{width}.input")]
                .as_dtype(Dtype::Float32, &stream)
                .unwrap();
            let weight = fixture[&format!("{width}.weight")]
                .as_dtype(Dtype::Float32, &stream)
                .unwrap();
            let normalized = f32_weightless_rms(&input, 1e-6, &stream).unwrap().unwrap();
            let output = normalized
                .multiply(&weight, &stream)
                .unwrap()
                .as_dtype(Dtype::Bfloat16, &stream)
                .unwrap()
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .into_evaluated()
                .unwrap();
            let expected = fixture[&format!("{width}.output")]
                .as_dtype(Dtype::Float32, &stream)
                .unwrap()
                .into_evaluated()
                .unwrap();
            let differences = output
                .as_slice::<f32>()
                .iter()
                .zip(expected.as_slice::<f32>())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(differences, 0, "width {width}");
        }
    }
}

pub(crate) fn f32_sum_last(input: &Array, stream: &Stream) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use crate::backend::managed_memory::row_kernels::{self, RowKernel};
        use safemlx::{DeviceType, Dtype};
        if input.dtype() != Dtype::Float32
            || input.size() == 0
            || input.ndim() == 0
            || stream.device_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        row_kernels::validate_call(RowKernel::Sum, input.ndim())?;
        let width = input.dim(-1);
        let rows = i32::try_from(input.size() / width as usize)
            .map_err(|_| Exception::custom("row sum count overflow"))?;
        let mut fixed_shape = [0; row_kernels::OUTPUT_DIMENSIONS];
        let mut ordinary_shape;
        let shape = if input.ndim() <= fixed_shape.len() {
            fixed_shape[..input.ndim()].copy_from_slice(input.shape());
            &mut fixed_shape[..input.ndim()]
        } else {
            // Only the ordinary path reaches this branch: validate_call checks
            // the fixed native shape capacity before any original construction.
            ordinary_shape = input.shape().to_vec();
            ordinary_shape.as_mut_slice()
        };
        *shape.last_mut().expect("row sum rank") = 1;
        row_kernels::apply(RowKernel::Sum, [input], shape, rows, stream).map(Some)
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, stream);
        Ok(None)
    }
}
