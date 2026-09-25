//! Mixed-storage GEMM exactness and allocation coverage. Run serially: allocator counters are global.
use safemlx::{
    fast::try_mixed_storage_gemm, memory, ops::indexing::TryIndexOp, transforms, Array, Device,
    DeviceType, Dtype, Stream,
};

fn values(count: usize, mut seed: u32) -> Vec<f32> {
    (0..count)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let x = ((seed & 0xffff) as f32 - 32768.0) / 12345.0;
            // Mixed magnitudes and cancellation; avoid NaNs whose payloads have no
            // portable contract. Representable tiny/subnormal narrow values included.
            x * [1.0, 0.001, 100.0, -1.0, 0.0000001][i % 5]
        })
        .collect()
}
fn bits(a: &Array) -> Vec<u32> {
    a.clone()
        .into_evaluated()
        .unwrap()
        .try_as_slice::<f32>()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}
fn same(a: &Array, b: &Array, context: &str) {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), Dtype::Float32);
    let (a, b) = (bits(a), bits(b));
    if let Some((i, (x, y))) = a.iter().zip(&b).enumerate().find(|(_, (x, y))| x != y) {
        panic!("{context}: index {i}: {x:08x} != {y:08x}");
    }
}

#[test]
#[ignore = "requires Metal SIMD GEMM; run explicitly with MLX_ENABLE_TF32=0"]
fn loader_preserves_native_gemm_bits_and_allocations() {
    let s = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    memory::set_cache_limit(0).unwrap();
    let mut cases = Vec::new();
    for k in [1, 3, 15, 16, 17, 63, 64, 65] {
        for (m, n) in [(2, 2), (33, 63), (65, 65)] {
            cases.push((m, n, k));
        }
    }
    // K=1 is the native degenerate NN layout (both weight strides are one).
    // Cover its remaining MN alignment flags as well as small regular NT tails.
    cases.extend([
        (64, 32, 1),
        (64, 33, 1),
        (65, 32, 1),
        (64, 64, 3),
        (64, 65, 3),
        (65, 64, 3),
    ]);
    // Both regular tile sizes, split-K BM16/BM32, aligned and unaligned MN/K,
    // split-K residual partitions, and selector thresholds.
    for m in [
        2, 4, 8, 16, 17, 31, 32, 33, 39, 40, 63, 64, 65, 127, 128, 129,
    ] {
        for (n, k) in [(17, 127), (32, 128), (33, 129), (65, 257), (129, 513)] {
            cases.push((m, n, k));
        }
    }
    for (n, k) in [
        (512, 2048),
        (2048, 2048),
        (6144, 2048),
        (8192, 2048),
        (2048, 8192),
        (65536, 2048),
    ] {
        for m in [2, 8, 16, 128, 2000] {
            cases.push((m, n, k));
        }
    }
    // Threshold around the regular 32/64-row tile decision; tails at large K.
    cases.extend([
        (127, 8191, 2047),
        (128, 8193, 2049),
        (129, 8193, 2049),
        (33, 2047, 8191),
        (2000, 513, 2049),
    ]);
    let mut checked = 0;
    for dtype in [Dtype::Float16, Dtype::Bfloat16] {
        for &(m, n, k) in &cases {
            let a = Array::from_slice(&values(m * k, 123), &[m as i32, k as i32]);
            let w = Array::from_slice(&values(n * k, 567), &[n as i32, k as i32])
                .as_dtype(dtype, &s)
                .unwrap();
            transforms::eval([&a, &w]).unwrap();
            // Settle the reference weight first. Measure only the native GEMM
            // output/partials, using this device's actual dispatch and allocator.
            let converted = w.as_dtype(Dtype::Float32, &s).unwrap();
            transforms::eval([&converted]).unwrap();
            s.synchronize().unwrap();
            let reference_baseline = memory::active_memory().unwrap();
            memory::reset_peak_memory().unwrap();
            let reference = a.matmul(converted.transpose(&s).unwrap(), &s).unwrap();
            let expected = bits(&reference);
            s.synchronize().unwrap();
            let reference_peak = memory::peak_memory().unwrap() - reference_baseline;
            drop(reference);
            drop(converted);
            s.synchronize().unwrap();
            let baseline = memory::active_memory().unwrap();
            memory::reset_peak_memory().unwrap();
            let output = try_mixed_storage_gemm(&a, &w, &s)
                .unwrap()
                .expect("native SIMD GEMM device/configuration");
            transforms::eval([&output]).unwrap();
            s.synchronize().unwrap();
            let peak = memory::peak_memory().unwrap() - baseline;
            let actual = bits(&output);
            if let Some(i) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
                panic!(
                    "{dtype:?} [{m},{n},{k}] index {i}: {:08x} != {:08x}",
                    actual[i], expected[i]
                );
            }
            // Same output and split-K workspace as native preconverted GEMM.
            // This works across native device-specific split-K thresholds and
            // host page sizes, without duplicating either selection policy.
            assert_eq!(peak, reference_peak, "allocation {dtype:?} [{m},{n},{k}]");
            let bias = Array::from_slice(&values(n, 999), &[n as i32]);
            let reference = Array::from_slice(
                &expected
                    .iter()
                    .map(|x| f32::from_bits(*x))
                    .collect::<Vec<_>>(),
                &[m as i32, n as i32],
            );
            same(
                &output.add(&bias, &s).unwrap(),
                &reference.add(&bias, &s).unwrap(),
                "separate bias",
            );
            checked += 1;
        }
    }
    eprintln!("{checked} F16/BF16 exact-bit, separate-bias and allocation cases passed");
}

#[test]
#[ignore = "requires Metal SIMD GEMM; run explicitly with MLX_ENABLE_TF32=0"]
fn unsupported_layouts_and_types_do_not_evaluate_or_cast() {
    let s = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let a = Array::from_slice(&values(4 * 129, 123), &[4, 129]);
    let w = Array::from_slice(&values(65 * 129, 567), &[65, 129])
        .as_dtype(Dtype::Bfloat16, &s)
        .unwrap();
    assert!(!w.is_available().unwrap());
    assert!(try_mixed_storage_gemm(&a, &w, &s).unwrap().is_none());
    assert!(!w.is_available().unwrap());
    transforms::eval([&a, &w]).unwrap();
    assert!(try_mixed_storage_gemm(&a, &w, &s).unwrap().is_some());
    let cpu = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    assert!(try_mixed_storage_gemm(&a, &w, &cpu).unwrap().is_none());
    for bad in [
        w.as_dtype(Dtype::Float32, &s).unwrap(),
        w.transpose(&s).unwrap(),
        w.try_index_device((.., ..128), &s).unwrap(),
    ] {
        transforms::eval([&bad]).unwrap();
        assert!(try_mixed_storage_gemm(&a, &bad, &s).unwrap().is_none());
    }
    let padded = Array::from_slice(&values(65 * 130, 567), &[65, 130])
        .as_dtype(Dtype::Bfloat16, &s)
        .unwrap()
        .try_index_device((.., ..129), &s)
        .unwrap();
    transforms::eval([&padded]).unwrap();
    assert!(try_mixed_storage_gemm(&a, &padded, &s).unwrap().is_none());
    let column_weight = Array::from_slice(&values(129 * 65, 567), &[129, 65])
        .as_dtype(Dtype::Bfloat16, &s)
        .unwrap()
        .transpose(&s)
        .unwrap();
    transforms::eval([&column_weight]).unwrap();
    assert_eq!(column_weight.shape(), w.shape());
    assert!(try_mixed_storage_gemm(&a, &column_weight, &s)
        .unwrap()
        .is_none());
    let narrow_input = a.as_dtype(Dtype::Float16, &s).unwrap();
    transforms::eval([&narrow_input]).unwrap();
    assert!(try_mixed_storage_gemm(&narrow_input, &w, &s)
        .unwrap()
        .is_none());
    let lazy_input = a.add(&a, &s).unwrap();
    assert!(!lazy_input.is_available().unwrap());
    let output = try_mixed_storage_gemm(&lazy_input, &w, &s)
        .unwrap()
        .unwrap();
    assert!(!lazy_input.is_available().unwrap());
    assert!(!output.is_available().unwrap());
    same(
        &output,
        &lazy_input
            .matmul(
                w.as_dtype(Dtype::Float32, &s)
                    .unwrap()
                    .transpose(&s)
                    .unwrap(),
                &s,
            )
            .unwrap(),
        "lazy activation",
    );
    let column_input = Array::from_slice(&values(4 * 129, 123), &[129, 4])
        .transpose(&s)
        .unwrap();
    transforms::eval([&column_input]).unwrap();
    same(
        &try_mixed_storage_gemm(&column_input, &w, &s)
            .unwrap()
            .unwrap(),
        &column_input
            .matmul(
                w.as_dtype(Dtype::Float32, &s)
                    .unwrap()
                    .transpose(&s)
                    .unwrap(),
                &s,
            )
            .unwrap(),
        "column activation",
    );
    let one = a.try_index_device((..1, ..), &s).unwrap();
    transforms::eval([&one]).unwrap();
    assert!(try_mixed_storage_gemm(&one, &w, &s).unwrap().is_none());
    // A contiguous view with a nonzero byte offset is eligible.
    let offset = w.try_index_device((1.., ..), &s).unwrap();
    transforms::eval([&offset]).unwrap();
    same(
        &try_mixed_storage_gemm(&a, &offset, &s).unwrap().unwrap(),
        &a.matmul(
            offset
                .as_dtype(Dtype::Float32, &s)
                .unwrap()
                .transpose(&s)
                .unwrap(),
            &s,
        )
        .unwrap(),
        "offset",
    );
}

#[test]
fn cpu_prototype_rejection_needs_no_gpu_or_evaluation() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let a = Array::from_slice(&[1.0f32; 6], &[2, 3]);
    let w = a.as_dtype(Dtype::Float16, &stream).unwrap();
    assert!(!w.is_available().unwrap());
    assert!(try_mixed_storage_gemm(&a, &w, &stream).unwrap().is_none());
    assert!(!w.is_available().unwrap());
}

#[test]
#[ignore = "requires Metal SIMD GEMM; run explicitly with MLX_ENABLE_TF32=0"]
fn lazy_batched_and_strided_activations_preserve_bits() {
    let s = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    for dtype in [Dtype::Float16, Dtype::Bfloat16] {
        // Exercise transposed-A kernel families, tile/K tails, split-K and the
        // same flatten/copy/unflatten operations as native high-rank matmul.
        for (m, n, k) in [(4, 33, 129), (32, 64, 128), (66, 65, 257), (128, 129, 513)] {
            let w = Array::from_slice(&values(n * k, 567), &[n as i32, k as i32])
                .as_dtype(dtype, &s)
                .unwrap();
            transforms::eval([&w]).unwrap();
            let contiguous = Array::from_slice(&values(m * k, 123), &[m as i32, k as i32]);
            let column = Array::from_slice(&values(m * k, 123), &[k as i32, m as i32])
                .transpose(&s)
                .unwrap();
            let padded = Array::from_slice(&values(m * (k + 3), 123), &[m as i32, (k + 3) as i32])
                .try_index_device((.., ..k as i32), &s)
                .unwrap();
            let sliced = Array::from_slice(&values(m * k * 2, 123), &[m as i32, k as i32, 2])
                .try_index_device((.., .., 0), &s)
                .unwrap();
            for a in [contiguous, column, padded, sliced] {
                for batched in [false, true] {
                    let a = if batched {
                        a.reshape(&[2, 1, (m / 2) as i32, k as i32], &s).unwrap()
                    } else {
                        a.clone()
                    };
                    let output = try_mixed_storage_gemm(&a, &w, &s).unwrap().unwrap();
                    assert!(!output.is_available().unwrap());
                    let expected = a
                        .matmul(
                            w.as_dtype(Dtype::Float32, &s)
                                .unwrap()
                                .transpose(&s)
                                .unwrap(),
                            &s,
                        )
                        .unwrap();
                    same(
                        &output,
                        &expected,
                        &format!("{dtype:?} {m} {n} {k} batched={batched}"),
                    );
                }
            }
        }
    }
}
