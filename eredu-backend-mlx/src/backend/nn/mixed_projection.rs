//! Mixed F32-activation / narrow-weight projections without full-weight casts.
use safemlx::{error::Exception, Array, Stream};

/// Matches the native F32 GEMV reduction while converting each weight on load.
///
/// Only one logical input row is eligible. Multi-row prefill and speculative
/// verification keep native GEMM, whose reduction depends on its tile/device
/// selection. Column-major weights similarly keep the native transposed GEMV.
/// Unevaluated weights also use native fallback because their strides are not
/// final yet. A contiguous copy, when necessary, stays in the narrow dtype.
/// Conversion-retention estimates remain conservative: these same weights can
/// still need their F32 conversions on a later multi-row invocation.
pub(crate) fn project(
    input: &Array,
    weight: &Array,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use safemlx::{
            fast::{CustomKernelConfig, MetalKernel},
            DeviceType, Dtype,
        };
        use std::cell::RefCell;
        thread_local! {
            static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
        }
        if input.ndim() == 0
            || weight.ndim() != 2
            || input.dtype() != Dtype::Float32
            || !matches!(weight.dtype(), Dtype::Float16 | Dtype::Bfloat16)
            || input.dim(-1) <= 0
            || weight.dim(0) <= 0
            || weight.dim(1) != input.dim(-1)
            || input.size() != input.dim(-1) as usize
            || !weight.is_available()?
            || weight.strides()[1] != 1
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        let width = input.dim(-1);
        let outputs = weight.dim(0);
        // Match MLX's non-transposed GEMV dispatch, including small-K and wide-K
        // reductions; changing these partitions changes floating-point results.
        let (bm, bn, sm, sn) = if width <= 64 {
            (1, 1, 8, 4)
        } else if i64::from(width) >= 16 * i64::from(outputs) {
            (1, 8, 1, 32)
        } else {
            (if outputs >= 4096 { 8 } else { 4 }, 1, 1, 32)
        };
        let tm = if outputs < 4 { 1 } else { 4 };
        let block_rows = bm * sm * tm;
        let blocks = outputs / block_rows + i32::from(outputs % block_rows != 0);
        let Some(grid_x) = 32_i32.checked_mul(blocks) else {
            return Ok(None);
        };
        let config = CustomKernelConfig::new()
            .with_template_arg_int("WIDTH", width)
            .with_template_arg_int("OUTPUTS", outputs)
            .with_template_arg_int("BM", bm)
            .with_template_arg_int("BN", bn)
            .with_template_arg_int("SM", sm)
            .with_template_arg_int("SN", sn)
            .with_template_arg_int("TM", tm)
            .with_grid([grid_x, bn, bm])
            .with_thread_group([32, bn, bm])
            .with_output_arg([outputs], Dtype::Float32);
        let output = KERNEL.with(|cell| -> Result<Array, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "mixed_f32_narrow_gemv",
                    ["input", "weight"],
                    ["output"],
                    include_str!("mixed_projection.metal"),
                    "",
                    true,
                    false,
                )?);
            }
            cell.borrow()
                .as_ref()
                .expect("mixed projection kernel initialized")
                .apply_one_device([input, weight], &config, stream)
        })?;
        let mut shape = input.shape().to_vec();
        *shape.last_mut().expect("nonempty input rank") = outputs;
        output.reshape(&shape, stream).map(Some)
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, weight, stream);
        Ok(None)
    }
}

#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod tests {
    use super::*;
    use safemlx::{Device, DeviceType, Dtype};

    fn values(count: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state & 0xffff) as f32 - 32768.0) / 12345.0
            })
            .collect()
    }

    fn assert_same(input: &Array, weight: &Array, stream: &Stream) {
        safemlx::transforms::eval([weight]).unwrap();
        let actual = project(input, weight, stream).unwrap().unwrap();
        let expected = safemlx::ops::matmul(
            input,
            weight
                .as_dtype(Dtype::Float32, stream)
                .unwrap()
                .transpose(stream)
                .unwrap(),
            stream,
        )
        .unwrap();
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.dtype(), Dtype::Float32);
        let actual = actual.into_evaluated().unwrap();
        let expected = expected.into_evaluated().unwrap();
        let actual = actual.try_as_slice::<f32>().unwrap();
        let expected = expected.try_as_slice::<f32>().unwrap();
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{:?} {:?} {:?} output {index}: {actual} != {expected}",
                input.shape(),
                weight.shape(),
                weight.dtype()
            );
        }
    }

    #[test]
    fn mixed_projection_matches_native_f32_gemv_bits() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        for dtype in [Dtype::Float16, Dtype::Bfloat16] {
            for (width, outputs) in [
                (1, 1),
                (3, 2),
                (31, 3),
                (32, 5),
                (63, 17),
                (64, 33),
                (65, 3),
                (127, 5),
                (128, 17),
                (129, 35),
                (768, 256),
                (1024, 1536),
                (2560, 4097),
                (4096, 17),
                (8192, 256),
            ] {
                let input = Array::from_slice(&values(width as usize, 123), &[1, 1, width]);
                let weight =
                    Array::from_slice(&values((width * outputs) as usize, 567), &[outputs, width])
                        .as_dtype(dtype, &stream)
                        .unwrap();
                assert_same(&input, &weight, &stream);
            }
        }
    }

    #[test]
    fn mixed_projection_copies_strided_vectors_without_changing_the_dot() {
        use safemlx::ops::indexing::TryIndexOp;
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        for dtype in [Dtype::Float16, Dtype::Bfloat16] {
            let input = Array::from_slice(&values(256, 123), &[128, 2])
                .try_index_device((.., 0), &stream)
                .unwrap();
            let weight = Array::from_slice(&values(17 * 128, 567), &[17, 128])
                .as_dtype(dtype, &stream)
                .unwrap();
            assert_same(&input, &weight, &stream);
            let padded = Array::from_slice(&values(17 * 256, 567), &[17, 256])
                .as_dtype(dtype, &stream)
                .unwrap()
                .try_index_device((.., ..128), &stream)
                .unwrap();
            assert_same(&input, &padded, &stream);
        }
    }

    #[test]
    fn mixed_projection_keeps_native_gemm_and_transposed_matrix_fallbacks() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let weight = Array::from_slice(&values(128 * 17, 567), &[128, 17])
            .as_dtype(Dtype::Bfloat16, &stream)
            .unwrap();
        let input = Array::from_slice(&values(128, 123), &[1, 128]);
        let lazy = weight.reshape(&[17, 128], &stream).unwrap();
        assert!(!lazy.is_available().unwrap());
        assert!(project(&input, &lazy, &stream).unwrap().is_none());
        let transposed = weight.transpose(&stream).unwrap();
        safemlx::transforms::eval([&transposed]).unwrap();
        assert!(project(&input, &transposed, &stream).unwrap().is_none());
        let weight = weight.reshape(&[17, 128], &stream).unwrap();
        safemlx::transforms::eval([&weight]).unwrap();
        let batch = Array::from_slice(&values(256, 123), &[2, 128]);
        assert!(project(&batch, &weight, &stream).unwrap().is_none());
        assert!(project(
            &input.as_dtype(Dtype::Bfloat16, &stream).unwrap(),
            &weight,
            &stream
        )
        .unwrap()
        .is_none());
        let cpu = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        assert!(project(&input, &weight, &cpu).unwrap().is_none());
    }

    #[test]
    fn mixed_projection_linear_bias_and_tied_embedding_preserve_retention_accounting() {
        use crate::backend::nn::{
            linear::PhysicalLinear,
            parameter_conversion::{execution_budget, trim_budget, ResidentParameterConversions},
        };
        use crate::module::PhysicalParam;
        use eredu_core::residency::{
            ParameterConversionRetentionEligibility, ParameterConversionRetentionPolicy,
        };
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        for dtype in [Dtype::Float16, Dtype::Bfloat16] {
            let weight = Array::from_slice(&values(128 * 17, 567), &[17, 128])
                .as_dtype(dtype, &stream)
                .unwrap();
            safemlx::transforms::eval([&weight]).unwrap();
            let bias = Array::from_slice(&values(17, 999), &[17]);
            let mut linear = PhysicalLinear::unloaded(
                128,
                17,
                true,
                eredu_checkpoint::LinearFormat::Dense,
                &stream,
            )
            .unwrap();
            linear.weight = PhysicalParam::new(weight.clone());
            linear.bias = PhysicalParam::new(Some(bias.clone()));
            let embedding = crate::nn::Embedding {
                weight: PhysicalParam::new(weight.clone()),
            };
            let budget = execution_budget(
                Some(ParameterConversionRetentionPolicy::Unlimited),
                ParameterConversionRetentionEligibility::Eligible,
            )
            .unwrap();
            let mut owner = ResidentParameterConversions::default();
            owner.register([&weight], &budget).unwrap();
            for rows in [1, 3, 1] {
                let input = Array::from_slice(&values(rows * 128, 123), &[rows as i32, 128]);
                let expected = safemlx::ops::matmul(
                    &input,
                    weight
                        .as_dtype(Dtype::Float32, &stream)
                        .unwrap()
                        .transpose(&stream)
                        .unwrap(),
                    &stream,
                )
                .unwrap();
                let tied = embedding
                    .as_linear(&input, &stream)
                    .unwrap()
                    .into_evaluated()
                    .unwrap();
                assert_eq!(
                    tied.as_slice::<f32>(),
                    expected.evaluated().unwrap().as_slice::<f32>()
                );
                let actual = linear
                    .forward(&input, &stream)
                    .unwrap()
                    .into_evaluated()
                    .unwrap();
                let biased = expected
                    .add(&bias, &stream)
                    .unwrap()
                    .into_evaluated()
                    .unwrap();
                assert_eq!(actual.as_slice::<f32>(), biased.as_slice::<f32>());
                let retained = budget
                    .report()
                    .usage
                    .value()
                    .unwrap()
                    .retained_payload_bytes;
                assert_eq!(retained, if rows == 1 { 0 } else { 17 * 128 * 4 });
                trim_budget(&budget).unwrap();
            }
        }
    }
}
