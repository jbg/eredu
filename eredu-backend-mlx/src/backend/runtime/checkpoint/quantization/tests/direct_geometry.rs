use super::*;
use safemlx::{Device, DeviceType, Dtype, OperationEvent, Stream};

const CODEBOOK: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];
fn shape(rank: usize, columns: i32) -> Vec<i32> {
    let mut shape = vec![1; rank];
    shape[0] = if rank == 2 { 6 } else { 2 };
    if rank > 2 {
        shape[1] = 3;
    }
    shape[rank - 1] = columns;
    shape
}
fn amplitude(group: usize) -> f32 {
    2.0f32.powi((group % 4) as i32 - 1)
}
fn values(config: WeightQuantization, elements: usize) -> Vec<f32> {
    let group = config.group_size() as usize;
    let bins = (1 << config.bits()) - 1;
    (0..elements)
        .map(|i| {
            let value = if config == WeightQuantization::MxFp4 {
                CODEBOOK[i % 16]
            } else {
                ((i % group) * bins as usize / (group - 1)) as f32
            };
            value * amplitude(i / group)
        })
        .collect()
}
fn input(values: &[f32], shape: &[i32], dtype: Dtype, strided: bool, stream: &Stream) -> Array {
    let rank = shape.len();
    let mut array = match dtype {
        Dtype::Float16 => Array::from_slice(
            &values
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect::<Vec<_>>(),
            shape,
        ),
        Dtype::Bfloat16 => Array::from_slice(
            &values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
            shape,
        ),
        Dtype::Float32 => Array::from_slice(values, shape),
        _ => unreachable!(),
    };
    if strided {
        let mut axes: Vec<i32> = (0..rank as i32).collect();
        axes.swap(0, rank - 1);
        array = array
            .transpose_axes(&axes, stream)
            .unwrap()
            .contiguous(false, stream)
            .unwrap()
            .transpose_axes(&axes, stream)
            .unwrap();
    }
    array.evaluated().unwrap();
    array
}
fn verify(output: &QuantizedTensor, config: WeightQuantization, shape: &[i32], stream: &Stream) {
    let group = config.group_size() as usize;
    let bits = config.bits() as usize;
    let bins = (1u32 << bits) - 1;
    let elements: usize = shape.iter().map(|&x| x as usize).product();
    let mut packed_shape = shape.to_vec();
    *packed_shape.last_mut().unwrap() = shape.last().unwrap() * bits as i32 / 32;
    let mut scale_shape = shape.to_vec();
    *scale_shape.last_mut().unwrap() = shape.last().unwrap() / group as i32;
    assert_eq!(output.weight.shape(), packed_shape);
    assert_eq!(output.scales.shape(), scale_shape);
    let packed = output.weight.evaluated().unwrap();
    let words = packed.try_as_slice::<u32>().unwrap();
    for i in 0..elements {
        let mut code = 0u32;
        for bit in 0..bits {
            let at = i * bits + bit;
            code |= ((words[at / 32] >> (at % 32)) & 1) << bit;
        }
        let expected = if config == WeightQuantization::MxFp4 {
            if i % 16 == 8 { 0 } else { (i % 16) as u32 }
        } else {
            bins - ((i % group) as u32 * bins / (group - 1) as u32)
        };
        assert_eq!(code, expected, "{config:?}, shape={shape:?}, index={i}");
    }
    if config == WeightQuantization::MxFp4 {
        assert!(output.biases.is_none());
        let scales = output.scales.evaluated().unwrap();
        for (g, &value) in scales.try_as_slice::<u8>().unwrap().iter().enumerate() {
            assert_eq!(value, 126 + (g % 4) as u8);
        }
    } else {
        let biases = output.biases.as_ref().unwrap();
        assert_eq!(biases.shape(), scale_shape);
        for (array, bias) in [(&output.scales, false), (biases, true)] {
            let floats = array.as_dtype(Dtype::Float32, stream).unwrap();
            for (g, &value) in floats
                .evaluated()
                .unwrap()
                .try_as_slice::<f32>()
                .unwrap()
                .iter()
                .enumerate()
            {
                assert_eq!(
                    value,
                    if bias {
                        bins as f32 * amplitude(g)
                    } else {
                        -amplitude(g)
                    }
                );
            }
        }
    }
}
fn ordinary(device: DeviceType) {
    let stream = Stream::new_with_device(&Device::new(device, 0));
    let mut configs = vec![WeightQuantization::MxFp4];
    for group in [32, 64, 128] {
        for bits in [2, 3, 4, 5, 6, 8] {
            configs.push(AffineQuantization::new(group, bits).unwrap().into());
        }
    }
    for config in configs {
        for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
            for rank in [2, 3, 4, 11] {
                for strided in [false, true] {
                    let shape = shape(rank, 128);
                    let input = input(&values(config, 6 * 128), &shape, dtype, strided, &stream);
                    let output = quantize_tensor(&input, config, &stream).unwrap();
                    verify(&output, config, &shape, &stream);
                }
            }
        }
    }
}
#[test]
fn direct_quantization_values_cpu() {
    ordinary(DeviceType::Cpu);
}
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn direct_quantization_values_metal() {
    ordinary(DeviceType::Gpu);
}

#[test]
fn direct_quantization_uses_shared_mxfp4_submission_storage() {
    use safemlx::{
        OriginalScopeObserver, PrefillRootsRuntime, PreparedInputRuntime,
        PreparedOriginalBufferBudget, PreparedPrefillFailure, PreparedSubmissionGraphQuota,
        PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner, SubmissionScope,
    };
    let qualified =
        OperationEvent::cpu_mxfp4_quantize_submission_layout(Dtype::Float32, 2, 6, 64).is_some();
    if std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref() == Ok("1") {
        assert!(qualified);
    }
    if !qualified {
        return;
    }
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let allocator = PreparedInputRuntime::prepare().unwrap();
    for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
        for rank in [2, 3, 11, 21] {
            let shape = shape(rank, 64);
            let input = input(
                &values(WeightQuantization::MxFp4, 6 * 64),
                &shape,
                dtype,
                false,
                &stream,
            );
            let layout =
                OperationEvent::cpu_mxfp4_quantize_submission_layout(dtype, rank, 6, 64).unwrap();
            let budget = PreparedOriginalBufferBudget::try_new(
                &allocator,
                layout.payload().physical_capacity(&allocator).unwrap(),
                (),
            )
            .unwrap()
            .try_allocate()
            .unwrap();
            let graph = PreparedSubmissionGraphQuota::try_new(layout.graph_capacity(), ())
                .unwrap()
                .try_allocate()
                .unwrap();
            let records = PreparedSubmissionRecordQuota::try_new(layout.record_capacity(), ())
                .unwrap()
                .try_allocate()
                .unwrap();
            let failure = PreparedPrefillFailure::try_new(())
                .unwrap()
                .try_allocate()
                .unwrap();
            let mut scope = SubmissionScope::try_begin_retaining(
                PreparedSubmissionScopeOwner::try_new(())
                    .unwrap()
                    .with_graph_quota(graph.clone())
                    .with_record_quota(records.clone()),
            )
            .unwrap();
            scope.enable_scoped_observation().unwrap();
            scope.require_original_native_controls().unwrap();
            failure.bind_original_scope(&scope).unwrap();
            scope.enable_original_native_controls().unwrap();
            scope.bind_original_buffer_budget(&budget).unwrap();
            let observer = OriginalScopeObserver::require_current().unwrap();
            OperationEvent::validate_traversal_leaf(&input, &observer).unwrap();
            let bank =
                OperationEvent::prepare_resident_graph(layout.construction().graph(), &observer)
                    .unwrap();
            let output = quantize_tensor(&input, WeightQuantization::MxFp4, &stream).unwrap();
            drop(bank);
            let roots = [&output.weight, &output.scales];
            let event = safemlx::transforms::async_eval_with_original_prepared_traversal(
                roots,
                &observer,
                &stream,
                &layout.traversal(),
            )
            .unwrap();
            event.synchronize().unwrap();
            // Completed-only access avoids another Eval under the original role.
            let packed = output
                .weight
                .completed_in_original_scope(&observer)
                .unwrap();
            for (i, &word) in packed.try_as_slice::<u32>().unwrap().iter().enumerate() {
                assert_eq!(word, if i % 2 == 0 { 0x76543210 } else { 0xfedcba90 });
            }
            let scales = output
                .scales
                .completed_in_original_scope(&observer)
                .unwrap();
            for (g, &scale) in scales.try_as_slice::<u8>().unwrap().iter().enumerate() {
                assert_eq!(scale, 126 + (g % 4) as u8);
            }
            drop((packed, scales));
            safemlx::try_with_submission_retirement(|| drop((event, output))).unwrap();
            safemlx::try_with_submission_retirement(|| drop(input)).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                let (progress, status) = observer.progress().unwrap();
                assert_eq!(progress, safemlx::ScopedSubmissionProgress::Observed);
                assert!(!status.failed() && !status.blocked());
                if status.is_settled() {
                    break;
                }
                assert!(std::time::Instant::now() < deadline);
                std::thread::yield_now();
            }
            observer.retire_completed_records().unwrap();
            scope.seal();
            drop((observer, scope));
            safemlx::reclaim_allocation_owners();
            assert_eq!(budget.occupied_bytes(), 0);
            assert_eq!(graph.occupied_bytes(), 0);
            assert_eq!(records.occupied_bytes(), 0);
        }
    }
}
