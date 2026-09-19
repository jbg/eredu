use super::*;
use crate::{
    CpuAffineQuantizeSubmissionLayout, PreparedInputRuntime, PreparedOriginalBufferBudget,
};

fn qualified() -> bool {
    let known = OperationEvent::affine_quantize_construction_layout(2).is_some();
    if std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref() == Ok("1") {
        assert!(known);
    }
    known
}
fn layout(
    dtype: Dtype,
    companion: Dtype,
    rank: usize,
    group: i32,
    bits: i32,
) -> CpuAffineQuantizeSubmissionLayout {
    OperationEvent::cpu_affine_quantize_submission_layout(
        dtype,
        companion,
        rank,
        3,
        group as usize * 2,
        group,
        bits,
    )
    .unwrap()
}
fn source(
    dtype: Dtype,
    rank: usize,
    group: i32,
    bits: i32,
    strided: bool,
    stream: &Stream,
) -> Array {
    let columns = group as usize * 2;
    let bins = (1u32 << bits) - 1;
    let value = |i: usize| {
        ((i % group as usize) as u32 * bins / (group as u32 - 1)) as f32
            * 2.0f32.powi(((i / group as usize) % 4) as i32 - 1)
    };
    let values: Vec<f32> = (0..3 * columns)
        .map(|i| {
            if strided {
                value((i % 3) * columns + i / 3)
            } else {
                value(i)
            }
        })
        .collect();
    let mut shape = vec![1; rank];
    shape[0] = if strided { columns as i32 } else { 3 };
    shape[rank - 1] = if strided { 3 } else { columns as i32 };
    let input = match dtype {
        Dtype::Float16 => Array::from_slice(
            &values
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect::<Vec<_>>(),
            &shape,
        ),
        Dtype::Bfloat16 => Array::from_slice(
            &values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
            &shape,
        ),
        _ => Array::from_slice(&values, &shape),
    };
    let input = if strided {
        let mut axes: Vec<i32> = (0..rank as i32).collect();
        axes.swap(0, rank - 1);
        input.transpose_axes(&axes, stream).unwrap()
    } else {
        input
    };
    input.evaluated().unwrap();
    // Detach the completed ordinary event so this is an available input leaf.
    input.evaluated().unwrap();
    input
}
fn construct(
    input: &Array,
    companion: Dtype,
    group: i32,
    bits: i32,
    layout: CpuAffineQuantizeSubmissionLayout,
    observer: &OriginalScopeObserver,
    stream: &Stream,
) -> [Array; 3] {
    OperationEvent::validate_traversal_leaf(input, observer).unwrap_or_else(|error| {
        panic!(
            "leaf {:?} {:?}, companion={companion:?}, group={group}, bits={bits}: {error:?}",
            input.dtype(),
            input.shape()
        )
    });
    let bank =
        OperationEvent::prepare_affine_quantize_graph(layout.construction(), observer).unwrap();
    let arrays = crate::ops::quantize_with_mode(
        input,
        group,
        bits,
        crate::ops::QuantizationMode::Affine,
        stream,
    )
    .unwrap();
    drop(bank);
    let mut outputs = [arrays.weight, arrays.scales, arrays.biases.unwrap()];
    if let Some(construction) = layout.companion_construction() {
        let bank = OperationEvent::prepare_resident_graph(construction, observer).unwrap();
        outputs[1] = outputs[1].as_dtype(companion, stream).unwrap();
        outputs[2] = outputs[2].as_dtype(companion, stream).unwrap();
        drop(bank);
    }
    outputs
}
fn verify(
    outputs: &[Array; 3],
    companion: Dtype,
    rank: usize,
    group: i32,
    bits: i32,
    observer: &OriginalScopeObserver,
) {
    let columns = group as usize * 2;
    let bins = (1u32 << bits) - 1;
    let weights = outputs[0].completed_in_original_scope(observer).unwrap();
    let words = weights.try_as_slice::<u32>().unwrap();
    assert_eq!(outputs[0].ndim(), rank);
    assert_eq!(outputs[0].shape()[rank - 1], columns as i32 * bits / 32);
    for i in 0..3 * columns {
        let mut code = 0;
        for bit in 0..bits as usize {
            let at = i * bits as usize + bit;
            code |= ((words[at / 32] >> (at % 32)) & 1) << bit;
        }
        assert_eq!(
            code,
            bins - (i % group as usize) as u32 * bins / (group as u32 - 1)
        );
    }
    for (index, output) in outputs.iter().enumerate().skip(1) {
        assert_eq!(output.dtype(), companion);
        assert_eq!(output.ndim(), rank);
        assert_eq!(output.shape()[rank - 1], 2);
        let view = output.completed_in_original_scope(observer).unwrap();
        for g in 0..6 {
            let actual = match companion {
                Dtype::Float16 => view.try_as_slice::<half::f16>().unwrap()[g].to_f32(),
                Dtype::Bfloat16 => view.try_as_slice::<half::bf16>().unwrap()[g].to_f32(),
                _ => view.try_as_slice::<f32>().unwrap()[g],
            };
            let amplitude = 2.0f32.powi((g % 4) as i32 - 1);
            assert_eq!(
                actual,
                if index == 1 {
                    -amplitude
                } else {
                    bins as f32 * amplitude
                }
            );
        }
    }
}
#[test]
fn affine_submission_validates_geometry_before_resource_creation() {
    for (dtype, companion, rank, rows, columns, group, bits) in [
        (Dtype::Float64, Dtype::Float32, 2, 1, 32, 32, 4),
        (Dtype::Float32, Dtype::Int32, 2, 1, 32, 32, 4),
        (Dtype::Float32, Dtype::Float32, 1, 1, 32, 32, 4),
        (Dtype::Float32, Dtype::Float16, usize::MAX, 1, 32, 32, 4),
        (Dtype::Float32, Dtype::Float32, 2, 0, 32, 32, 4),
        (Dtype::Float32, Dtype::Float16, 2, 1, 0, 32, 4),
        (Dtype::Float32, Dtype::Float16, 2, 1, 33, 32, 4),
        (Dtype::Float32, Dtype::Float16, 2, 1, 64, 16, 4),
        (Dtype::Float32, Dtype::Float16, 2, 1, 64, 32, 7),
        (Dtype::Float32, Dtype::Float16, 2, usize::MAX, 64, 32, 4),
        (Dtype::Float32, Dtype::Float16, 2, 1, usize::MAX, 32, 4),
    ] {
        assert!(OperationEvent::cpu_affine_quantize_submission_layout(
            dtype, companion, rank, rows, columns, group, bits
        )
        .is_none());
    }
}
#[test]
fn affine_submission_covers_sibling_outputs_and_companion_conversions() {
    if !qualified() {
        return;
    }
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let allocator = PreparedInputRuntime::prepare().unwrap();
    for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
        for companion in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
            for rank in [2, 4, 11, 21] {
                for group in [32, 64, 128] {
                    for bits in [2, 3, 4, 5, 6, 8] {
                        for strided in [false, true] {
                            let layout = layout(dtype, companion, rank, group, bits);
                            let input = source(dtype, rank, group, bits, strided, &stream);
                            let capacity = layout.physical_capacity(&allocator).unwrap();
                            let budget =
                                PreparedOriginalBufferBudget::try_new(&allocator, capacity, ())
                                    .unwrap()
                                    .try_allocate()
                                    .unwrap();
                            let mut original = Original::with_capacities(
                                (),
                                layout.graph_capacity(),
                                layout.record_capacity(),
                            );
                            original.scope.bind_original_buffer_budget(&budget).unwrap();
                            let observer = OriginalScopeObserver::require_current().unwrap();
                            let outputs = construct(
                                &input, companion, group, bits, layout, &observer, &stream,
                            );
                            let event =
                                crate::transforms::async_eval_with_original_prepared_traversal(
                                    &outputs,
                                    &observer,
                                    &stream,
                                    &layout.traversal(),
                                )
                                .unwrap();
                            event.synchronize().unwrap();
                            verify(&outputs, companion, rank, group, bits, &observer);
                            assert_eq!(
                                outputs.each_ref().map(Array::nbytes),
                                layout.output_bytes()
                            );
                            assert!(budget.occupied_bytes() <= capacity);
                            let graph = original.graph.clone();
                            let records = original._records.clone();
                            crate::try_with_submission_retirement(|| drop((event, outputs, input)))
                                .unwrap();
                            settle(&observer);
                            original.scope.seal();
                            drop((observer, original));
                            crate::reclaim_allocation_owners();
                            assert_eq!(budget.occupied_bytes(), 0);
                            assert_eq!(graph.occupied_bytes(), 0);
                            assert_eq!(records.occupied_bytes(), 0);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn affine_submission_refuses_native_allocation_prefixes_and_short_records() {
    if !qualified() {
        return;
    }
    let allocator = PreparedInputRuntime::prepare().unwrap();
    for (dtype, companion) in [
        (Dtype::Float16, Dtype::Float32),
        (Dtype::Bfloat16, Dtype::Float16),
        (Dtype::Float32, Dtype::Bfloat16),
    ] {
        let layout = layout(dtype, companion, 11, 128, 8);
        // Refuse compaction and each native sibling allocation before any can retire.
        // The final case funds all payloads but supplies insufficient Record storage.
        for prefix in 0..=4 {
            // An entered failure can terminate its stream; each refusal owns one.
            let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
            let short_record = prefix == 4;
            let input = source(dtype, 11, 128, 8, true, &stream);
            let capacity = layout.request_bytes()[..if short_record {
                layout.request_bytes().len()
            } else {
                prefix
            }]
                .iter()
                .map(|&bytes| {
                    crate::OriginalBufferBudget::request_layout(&allocator, bytes)
                        .unwrap()
                        .capacity()
                })
                .sum();
            let budget = PreparedOriginalBufferBudget::try_new(&allocator, capacity, ())
                .unwrap()
                .try_allocate()
                .unwrap();
            let mut original = Original::with_capacities(
                (),
                layout.graph_capacity(),
                if short_record {
                    SubmissionRecordQuota::fresh_capacity_for_extents(0).unwrap()
                } else {
                    layout.record_capacity()
                },
            );
            original.scope.bind_original_buffer_budget(&budget).unwrap();
            let observer = OriginalScopeObserver::require_current().unwrap();
            let outputs = construct(&input, companion, 128, 8, layout, &observer, &stream);
            let failure = crate::transforms::async_eval_with_original_prepared_traversal(
                &outputs,
                &observer,
                &stream,
                &layout.traversal(),
            )
            .expect_err("incomplete physical or Record capacity must refuse");
            assert!(original._failure.error().is_some());
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let (progress, status) = observer.progress().unwrap();
                assert_eq!(progress, ScopedSubmissionProgress::Observed);
                if status.is_settled() {
                    break;
                }
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
            assert_eq!(
                observer.retire_completed_records().unwrap(),
                SubmissionRetirement::CompleteSnapshot
            );
            assert!(budget.occupied_bytes() <= capacity);
            let graph = original.graph.clone();
            let records = original._records.clone();
            crate::try_with_submission_retirement(|| drop((outputs, input, failure))).unwrap();
            original.scope.seal();
            drop((observer, original));
            crate::reclaim_allocation_owners();
            assert_eq!(budget.occupied_bytes(), 0);
            assert_eq!(graph.occupied_bytes(), 0);
            assert_eq!(records.occupied_bytes(), 0);
        }
    }
}

#[test]
fn affine_submission_companion_refusals_retire_accepted_buffers() {
    if !qualified() {
        return;
    }
    let allocator = PreparedInputRuntime::prepare().unwrap();
    let layout = OperationEvent::cpu_affine_quantize_submission_layout(
        Dtype::Float16,
        Dtype::Float32,
        2,
        4096,
        64,
        32,
        8,
    )
    .unwrap();
    let capacities: Vec<_> = layout
        .request_bytes()
        .iter()
        .map(|&bytes| {
            crate::OriginalBufferBudget::request_layout(&allocator, bytes)
                .unwrap()
                .capacity()
        })
        .collect();
    // Dense inputs require no compaction. Widening casts cannot donate their
    // half-sized source buffers; the first and second conversions each need
    // another full-sized destination even if earlier sources have retired.
    let first_cast_peak = capacities[1] + capacities[2] + capacities[3] + capacities[4];
    let second_cast_peak = capacities[1] + capacities[3] + capacities[4] + capacities[5];
    assert!(second_cast_peak > first_cast_peak);
    for capacity in [first_cast_peak - 1, second_cast_peak - 1] {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
        let values: Vec<_> = (0..4096 * 64)
            .map(|index| half::f16::from_f32((index % 32) as f32))
            .collect();
        let input = Array::from_slice(&values, &[4096, 64]);
        input.evaluated().unwrap();
        input.evaluated().unwrap();
        let budget = PreparedOriginalBufferBudget::try_new(&allocator, capacity, ())
            .unwrap()
            .try_allocate()
            .unwrap();
        let mut original =
            Original::with_capacities((), layout.graph_capacity(), layout.record_capacity());
        original.scope.bind_original_buffer_budget(&budget).unwrap();
        let observer = OriginalScopeObserver::require_current().unwrap();
        let outputs = construct(&input, Dtype::Float32, 32, 8, layout, &observer, &stream);
        let failure = crate::transforms::async_eval_with_original_prepared_traversal(
            &outputs,
            &observer,
            &stream,
            &layout.traversal(),
        )
        .expect_err("companion conversion must refuse insufficient physical capacity");
        assert!(original._failure.error().is_some());
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let (progress, status) = observer.progress().unwrap();
            assert_eq!(progress, ScopedSubmissionProgress::Observed);
            if status.is_settled() {
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert_eq!(
            observer.retire_completed_records().unwrap(),
            SubmissionRetirement::CompleteSnapshot
        );
        assert!(budget.occupied_bytes() <= capacity);
        let graph = original.graph.clone();
        let records = original._records.clone();
        crate::try_with_submission_retirement(|| drop((outputs, input, failure))).unwrap();
        original.scope.seal();
        drop((observer, original));
        crate::reclaim_allocation_owners();
        assert_eq!(budget.occupied_bytes(), 0);
        assert_eq!(graph.occupied_bytes(), 0);
        assert_eq!(records.occupied_bytes(), 0);
    }
}

#[test]
fn affine_submission_outputs_keep_physical_custody_after_scope_and_budget_drop() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    if !qualified() {
        return;
    }
    #[derive(Debug)]
    struct Custody(Arc<AtomicUsize>);
    impl Drop for Custody {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let released = Arc::new(AtomicUsize::new(0));
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let allocator = PreparedInputRuntime::prepare().unwrap();
    let layout = layout(Dtype::Float32, Dtype::Float16, 11, 64, 5);
    let input = source(Dtype::Float32, 11, 64, 5, true, &stream);
    let budget = PreparedOriginalBufferBudget::try_new(
        &allocator,
        layout.physical_capacity(&allocator).unwrap(),
        Custody(released.clone()),
    )
    .unwrap()
    .try_allocate()
    .unwrap();
    let mut original =
        Original::with_capacities((), layout.graph_capacity(), layout.record_capacity());
    original.scope.bind_original_buffer_budget(&budget).unwrap();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let outputs = construct(&input, Dtype::Float16, 64, 5, layout, &observer, &stream);
    let event = crate::transforms::async_eval_with_original_prepared_traversal(
        &outputs,
        &observer,
        &stream,
        &layout.traversal(),
    )
    .unwrap();
    event.synchronize().unwrap();
    verify(&outputs, Dtype::Float16, 11, 64, 5, &observer);
    crate::try_with_submission_retirement(|| drop((event, input))).unwrap();
    settle(&observer);
    original.scope.seal();
    drop((observer, original, budget));
    crate::reclaim_allocation_owners();
    assert_eq!(released.load(Ordering::SeqCst), 0);
    let [weight, scales, biases] = outputs;
    drop((weight, scales));
    crate::reclaim_allocation_owners();
    assert_eq!(released.load(Ordering::SeqCst), 0);
    drop(biases);
    crate::reclaim_allocation_owners();
    assert_eq!(released.load(Ordering::SeqCst), 1);
}
