use super::*;
use crate::{PreparedInputRuntime, PreparedOriginalBufferBudget};

fn qualified() -> bool {
    let known = OperationEvent::resident_graph_layout(1, 0, 3).is_some();
    if std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref() == Ok("1") {
        assert!(known);
    }
    known
}

#[test]
fn quantize_submission_rejects_invalid_geometry_before_creating_resources() {
    for (dtype, rank, rows, columns) in [
        (Dtype::Float64, 2, 1, 32),
        (Dtype::Int32, 2, 1, 32),
        (Dtype::Float32, 1, 1, 32),
        (Dtype::Float32, usize::MAX, 1, 32),
        (Dtype::Float32, 2, 0, 32),
        (Dtype::Float32, 2, 1, 0),
        (Dtype::Float32, 2, 1, 33),
        (Dtype::Float32, 2, usize::MAX, 32),
        (Dtype::Float32, 2, 1, usize::MAX),
    ] {
        assert!(
            OperationEvent::cpu_mxfp4_quantize_submission_layout(dtype, rank, rows, columns)
                .is_none()
        );
    }
}

#[test]
fn quantize_submission_uses_derived_arenas_for_real_packed_outputs() {
    if !qualified() {
        return;
    }
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let allocator = PreparedInputRuntime::prepare().unwrap();
    let codebook = [
        0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
        for rank in [2, 3, 11, 21] {
            for (rows, columns) in [(1usize, 32usize), (2, 64), (17, 256)] {
                let layout = OperationEvent::cpu_mxfp4_quantize_submission_layout(
                    dtype, rank, rows, columns,
                )
                .unwrap();
                let mut shape = vec![1; rank];
                shape[0] = rows as i32;
                shape[rank - 1] = columns as i32;
                let values = (0..rows * columns)
                    .map(|i| codebook[i % 16] * 2.0f32.powi(((i / 32) % 4) as i32 - 1))
                    .collect::<Vec<_>>();
                let input = input(dtype, &values, &shape);
                input.evaluated().unwrap();
                let capacity = layout.payload().physical_capacity(&allocator).unwrap();
                let budget = PreparedOriginalBufferBudget::try_new(&allocator, capacity, ())
                    .unwrap()
                    .try_allocate()
                    .unwrap();
                {
                    let mut original = Original::with_capacities(
                        (),
                        layout.graph_capacity(),
                        layout.record_capacity(),
                    );
                    original.scope.bind_original_buffer_budget(&budget).unwrap();
                    let observer = OriginalScopeObserver::require_current().unwrap();
                    OperationEvent::validate_traversal_leaf(&input, &observer).unwrap();
                    let bank = OperationEvent::prepare_resident_graph(
                        layout.construction().graph(),
                        &observer,
                    )
                    .unwrap();
                    let arrays = crate::ops::quantize_with_mode(
                        &input,
                        32,
                        4,
                        crate::ops::QuantizationMode::MxFp4,
                        &stream,
                    )
                    .unwrap();
                    assert!(arrays.biases.is_none());
                    drop(bank);
                    let roots = [arrays.weight, arrays.scales];
                    let completion =
                        crate::transforms::async_eval_with_original_prepared_traversal(
                            &roots,
                            &observer,
                            &stream,
                            &layout.traversal(),
                        )
                        .unwrap();
                    completion.synchronize().unwrap();
                    let weights = roots[0].completed_in_original_scope(&observer).unwrap();
                    let packed = weights.try_as_slice::<u32>().unwrap();
                    for i in 0..values.len() {
                        let expected = if i % 16 == 8 { 0 } else { i % 16 };
                        assert_eq!(
                            (packed[i / 8] >> (4 * (i % 8))) & 15,
                            expected as u32,
                            "{dtype:?} rank={rank} rows={rows} columns={columns} index={i}"
                        );
                    }
                    let scales = roots[1].completed_in_original_scope(&observer).unwrap();
                    for (i, &scale) in scales.try_as_slice::<u8>().unwrap().iter().enumerate() {
                        assert_eq!(scale, 126 + (i % 4) as u8);
                    }
                    assert!(budget.occupied_bytes() > 0);
                    assert!(budget.occupied_bytes() <= capacity);
                    assert!(layout.control_bytes() > 0);
                    crate::try_with_submission_retirement(|| drop((roots, completion))).unwrap();
                    settle(&observer);
                }
                assert_eq!(budget.occupied_bytes(), 0);
            }
        }
    }
}

fn input(dtype: Dtype, values: &[f32], shape: &[i32]) -> Array {
    match dtype {
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
        _ => Array::from_slice(values, shape),
    }
}

#[test]
fn quantize_submission_refusals_settle_before_releasing_their_prefix() {
    if !qualified() {
        return;
    }
    let allocator = PreparedInputRuntime::prepare().unwrap();
    for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
        for short_record in [false, true] {
            let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
            let layout =
                OperationEvent::cpu_mxfp4_quantize_submission_layout(dtype, 2, 17, 256).unwrap();
            let input = input(dtype, &vec![1.0; 17 * 256], &[17, 256]);
            input.evaluated().unwrap();
            let capacity = if short_record {
                layout.payload().physical_capacity(&allocator).unwrap()
            } else {
                layout
                    .construction()
                    .seed_request_bytes()
                    .into_iter()
                    .map(|request| {
                        crate::OriginalBufferBudget::request_layout(&allocator, request)
                            .unwrap()
                            .capacity()
                    })
                    .sum()
            };
            let budget = PreparedOriginalBufferBudget::try_new(&allocator, capacity, ())
                .unwrap()
                .try_allocate()
                .unwrap();
            {
                let records = if short_record {
                    layout.record_capacity() / 2
                } else {
                    layout.record_capacity()
                };
                let mut original = Original::with_capacities((), layout.graph_capacity(), records);
                original.scope.bind_original_buffer_budget(&budget).unwrap();
                let observer = OriginalScopeObserver::require_current().unwrap();
                OperationEvent::validate_traversal_leaf(&input, &observer).unwrap();
                let bank = OperationEvent::prepare_resident_graph(
                    layout.construction().graph(),
                    &observer,
                )
                .unwrap();
                let arrays = crate::ops::quantize_with_mode(
                    &input,
                    32,
                    4,
                    crate::ops::QuantizationMode::MxFp4,
                    &stream,
                )
                .unwrap();
                drop(bank);
                let roots = [arrays.weight, arrays.scales];
                let failure = match crate::transforms::async_eval_with_original_prepared_traversal(
                    &roots,
                    &observer,
                    &stream,
                    &layout.traversal(),
                ) {
                    Ok(_) => panic!("the underfunded submission must refuse"),
                    Err(failure) => failure,
                };
                assert_eq!(
                    failure.scoped_evaluation_cause(),
                    Some(crate::error::ScopedEvaluationCause::Failed)
                );
                assert!(original._failure.error().is_some());
                assert!(budget.occupied_bytes() > 0);
                if short_record {
                    assert!(!observer.status().has_work());
                }
                let deadline = Instant::now() + Duration::from_secs(5);
                loop {
                    let (progress, status) = observer.progress().unwrap();
                    assert_eq!(progress, ScopedSubmissionProgress::Observed);
                    if !short_record {
                        assert!(status.failed());
                    }
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
                for root in &roots {
                    assert!(root.completed_in_original_scope(&observer).is_err());
                }
                crate::try_with_submission_retirement(|| drop((roots, failure))).unwrap();
            }
            assert_eq!(budget.occupied_bytes(), 0);
        }
    }
}
