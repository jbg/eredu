//! Mixed F32-activation / narrow-weight projections without full-weight casts.
use safemlx::{error::Exception, Array, Stream};

/// Shared selector for ordinary linear layers and tied embedding readout.
/// Single rows use native-equivalent GEMV; multiple rows use the pinned native
/// F32 GEMM with narrow weight loading. Neither path evaluates inputs to decide
/// eligibility. Unsupported geometry, weight layout, dtype or native arithmetic
/// path retains the caller's existing conversion-retention/matmul fallback.
/// Forecasts remain conservative until per-invocation coverage is reported.
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
            || !weight.is_available()?
            || weight.strides()[1] != 1
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        if input.size() != input.dim(-1) as usize {
            #[cfg(feature = "projection-profiling")]
            if super::projection_profile::cast_gemm_reference_enabled() {
                return Ok(None);
            }
            return safemlx::fast::try_mixed_storage_gemm(input, weight, stream);
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

#[cfg(feature = "projection-profiling")]
pub(crate) fn profile_path(input: &Array) -> &'static str {
    if input.size() == input.dim(-1) as usize {
        "mixed_gemv"
    } else {
        "mixed_gemm"
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
    fn mixed_projection_selects_gemm_and_keeps_unsupported_fallbacks() {
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
        assert_same(&batch, &weight, &stream);
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
            for rows in [1, 3, 2001, 1] {
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
                // Beyond the admitted GEMM bounds, both callers still use
                // the existing retained-conversion fallback.
                assert_eq!(retained, if rows == 2001 { 17 * 128 * 4 } else { 0 });
                trim_budget(&budget).unwrap();
            }
        }
    }
}

/// Current weight layout plus the native dispatch's row-dependent workspace.
/// This query creates host records only; it neither evaluates nor retains arrays.
pub(crate) fn storage_facts(
    weight: &Array,
    stream: &Stream,
) -> Result<Option<eredu_runtime::projection_memory::ProjectionStorageFacts>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use eredu_core::checkpoint::TensorDtype;
        use eredu_runtime::projection_memory::{ProjectionRowStorage, ProjectionStorageFacts};
        use safemlx::{DeviceType, Dtype};
        if weight.ndim() != 2
            || !matches!(weight.dtype(), Dtype::Float16 | Dtype::Bfloat16)
            || !weight.is_available()?
            || weight.dim(0) <= 1
            || weight.dim(0) > 65536
            || weight.dim(1) <= 0
            || weight.dim(1) > 8192
            || weight.strides() != [weight.dim(1) as usize, 1]
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        #[cfg(feature = "projection-profiling")]
        let reference = super::projection_profile::cast_gemm_reference_enabled();
        #[cfg(not(feature = "projection-profiling"))]
        let reference = false;
        // Geometry-only tables are reusable across equal-shaped resident bindings.
        // The native arithmetic configuration is process-immutable. Keep the cache
        // thread-local and bounded; no array, model, stream or parameter owner is kept.
        thread_local! {
            static TABLES: std::cell::RefCell<std::collections::BTreeMap<(i32, i32, i32), Vec<ProjectionRowStorage>>> = const { std::cell::RefCell::new(std::collections::BTreeMap::new()) };
        }
        let key = (
            stream.get_device()?.get_index()?,
            weight.dim(0),
            weight.dim(1),
        );
        let mut rows = vec![ProjectionRowStorage {
            first: 1,
            last: 1,
            mechanism: "mlx.mixed_f32_narrow_gemv".into(),
            partial_bytes_per_row: 0,
        }];
        if !reference {
            let cached = TABLES.with(|tables| tables.borrow().get(&key).cloned());
            let ranges = if let Some(cached) = cached {
                cached
            } else {
                let mut ranges: Vec<ProjectionRowStorage> = Vec::new();
                for m in 2..=2000 {
                    if let Some(partials) =
                        safemlx::fast::mixed_storage_gemm_workspace(m, key.1, key.2, stream)?
                    {
                        let per_row = partials / m as u64;
                        if let Some(last) = ranges.last_mut().filter(|r| {
                            r.last + 1 == m as u64 && r.partial_bytes_per_row == per_row
                        }) {
                            last.last = m as u64;
                        } else {
                            ranges.push(ProjectionRowStorage {
                                first: m as u64,
                                last: m as u64,
                                mechanism: "mlx.mixed_storage_f32_gemm".into(),
                                partial_bytes_per_row: per_row,
                            });
                        }
                    }
                }
                TABLES.with(|tables| {
                    let mut tables = tables.borrow_mut();
                    if tables.len() >= 256 {
                        tables.clear();
                    }
                    tables.insert(key, ranges.clone());
                });
                ranges
            };
            rows.extend(ranges);
        }
        Ok(Some(ProjectionStorageFacts {
            input: key.2 as u64,
            output: key.1 as u64,
            activation: TensorDtype::F32,
            weight: if weight.dtype() == Dtype::Float16 {
                TensorDtype::F16
            } else {
                TensorDtype::Bf16
            },
            output_scalar_bytes: 4,
            activation_copy_bytes_per_row: key.2 as u64 * 4,
            rows,
        }))
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (weight, stream);
        Ok(None)
    }
}

#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod storage_tests {
    use super::*;
    use eredu_nn::{mechanism_memory::MechanismInvocation, TensorElementType};
    use safemlx::{Device, DeviceType, Dtype};
    #[test]
    fn projection_storage_facts_preserve_laziness_and_report_native_partials() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        for (dtype, element) in [
            (Dtype::Float16, TensorElementType::F16),
            (Dtype::Bfloat16, TensorElementType::Bf16),
        ] {
            let weight = Array::from_slice(&vec![0.25f32; 129 * 257], &[129, 257])
                .as_dtype(dtype, &stream)
                .unwrap();
            assert!(storage_facts(&weight, &stream).unwrap().is_none());
            assert!(!weight.is_available().unwrap());
            safemlx::transforms::eval([&weight]).unwrap();
            stream.synchronize().unwrap();
            let before = safemlx::memory::active_memory().unwrap();
            let facts = storage_facts(&weight, &stream).unwrap().unwrap();
            assert_eq!(before, safemlx::memory::active_memory().unwrap());
            let partial = safemlx::fast::mixed_storage_gemm_workspace(8, 129, 257, &stream)
                .unwrap()
                .unwrap();
            assert_eq!(
                facts
                    .for_invocation(257, 129, &eredu_core::checkpoint::TensorDtype::F32, 8)
                    .unwrap()
                    .partial_bytes_per_row
                    * 8,
                partial
            );
            assert!(facts
                .for_invocation(257, 129, &eredu_core::checkpoint::TensorDtype::F32, 2001)
                .is_none());
            let invocation = MechanismInvocation::Projection {
                rows: 8,
                input: 257,
                output: 129,
                format: eredu_checkpoint::LinearFormat::Dense,
                element: TensorElementType::F32,
                weight_element: Some(element),
                bias: true,
            };
            let contract =
                super::super::memory::describe_bound_projection(&invocation, &weight, &stream)
                    .unwrap();
            assert!(!contract.storage.iter().any(|s| s.name == "promoted_weight"));
            assert_eq!(
                contract
                    .storage
                    .iter()
                    .find(|s| s.name == "split_k_partials")
                    .unwrap()
                    .payload
                    .upper,
                Some(partial)
            );
            assert!(super::super::memory::describe(&invocation)
                .unwrap()
                .storage
                .iter()
                .any(|s| s.name == "promoted_weight"));
            let cpu = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            assert!(storage_facts(&weight, &cpu).unwrap().is_none());
        }
    }
}
