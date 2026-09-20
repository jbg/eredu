use super::*;
use safemlx::{
    CpuAffineQuantizeSubmissionLayout, Device, DeviceType, Dtype, OperationEvent,
    PrefillRootsRuntime, PreparedInputRuntime, PreparedOriginalBufferBudget,
    PreparedPrefillFailure, PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota,
    PreparedSubmissionScopeOwner, Stream, SubmissionScope,
};
use std::cell::Cell;

thread_local! { static HOOKS: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOOKS.with(|count| count.set(count.get() + 1));
}
struct Hook;
impl Hook {
    fn new() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOOKS.with(|count| count.set(0));
        Self
    }
}
impl Drop for Hook {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

fn layout() -> Option<CpuAffineQuantizeSubmissionLayout> {
    let layout = OperationEvent::cpu_affine_quantize_submission_layout(
        Dtype::Float32,
        Dtype::Float32,
        2,
        2,
        64,
        32,
        4,
    );
    if std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref() == Ok("1") {
        assert!(layout.is_some());
    }
    layout
}
fn input() -> Array {
    let values: Vec<_> = (0..128).map(|index| (index % 16) as f32).collect();
    let input = Array::from_slice(&values, &[2, 64]);
    input.evaluated().unwrap();
    input
}
fn retire(observer: &OriginalScopeObserver) {
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
    assert_eq!(
        observer.retire_completed_records().unwrap(),
        safemlx::SubmissionRetirement::CompleteSnapshot
    );
}
fn original_outputs(
    input: &Array,
    layout: CpuAffineQuantizeSubmissionLayout,
    observer: &OriginalScopeObserver,
    stream: &Stream,
) -> [Array; 3] {
    OperationEvent::validate_traversal_leaf(input, observer).unwrap();
    let bank =
        OperationEvent::prepare_affine_quantize_graph(layout.construction(), observer).unwrap();
    let quantized = safemlx::ops::quantize_with_mode(
        input,
        32,
        4,
        safemlx::ops::QuantizationMode::Affine,
        stream,
    )
    .unwrap();
    drop(bank);
    [
        quantized.weight,
        quantized.scales,
        quantized.biases.unwrap(),
    ]
}

#[test]
fn ordinary_output_views_preserve_order_and_native_encoding() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let input = Array::from_slice(&[-3.5f32, 1.25, 6.0], &[3]);
    let outputs = [
        input.as_dtype(Dtype::Float16, &stream).unwrap(),
        input,
        Array::from_slice(&[0x12345678u32, 0xabcdef01], &[2]),
        Array::from_slice(&[3u8, 127, 254], &[3]),
    ];
    let expected = [
        [-3.5f32, 1.25, 6.0]
            .into_iter()
            .flat_map(|v| half::f16::from_f32(v).to_bits().to_ne_bytes())
            .collect::<Vec<_>>(),
        [-3.5f32, 1.25, 6.0]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect(),
        [0x12345678u32, 0xabcdef01]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect(),
        vec![3u8, 127, 254],
    ];
    let views = completed_outputs(&outputs, None);
    assert_eq!(views.len(), expected.len());
    for (view, expected) in views.zip(expected) {
        let view = view.unwrap();
        let mut bytes = vec![0; expected.len()];
        view.try_copy_native_bytes_into(&mut bytes).unwrap();
        assert_eq!(bytes, expected);
    }
}

#[test]
fn original_output_views_read_sealed_completion_without_new_work() {
    let Some(layout) = layout() else { return };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let allocator = PreparedInputRuntime::prepare().unwrap();
    let input = input();
    let budget = PreparedOriginalBufferBudget::try_new(
        &allocator,
        layout.physical_capacity(&allocator).unwrap(),
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
    let outputs = original_outputs(&input, layout, &observer, &stream);
    let event = safemlx::transforms::async_eval_with_original_prepared_traversal(
        &outputs,
        &observer,
        &stream,
        &layout.traversal(),
    )
    .unwrap();
    event.synchronize().unwrap();
    scope.seal();
    let before = (
        graph.occupied_bytes(),
        records.occupied_bytes(),
        budget.occupied_bytes(),
    );
    let hook = Hook::new();
    for _ in 0..3 {
        let mut views = completed_outputs(&outputs, Some(&observer));
        assert_eq!(views.len(), 3);
        let packed = views.next().unwrap().unwrap();
        let mut bytes = [0u8; 64];
        packed.try_copy_native_bytes_into(&mut bytes).unwrap();
        for (index, word) in bytes.chunks_exact(4).enumerate() {
            assert_eq!(
                u32::from_ne_bytes(word.try_into().unwrap()),
                if index % 2 == 0 {
                    0x89abcdef
                } else {
                    0x01234567
                }
            );
        }
        assert_eq!(
            views
                .next()
                .unwrap()
                .unwrap()
                .try_as_slice::<f32>()
                .unwrap(),
            &[-1.0; 4]
        );
        assert_eq!(
            views
                .next()
                .unwrap()
                .unwrap()
                .try_as_slice::<f32>()
                .unwrap(),
            &[15.0; 4]
        );
        assert!(views.next().is_none());
    }
    assert_eq!(HOOKS.with(Cell::get), 0);
    drop(hook);
    assert_eq!(
        (
            graph.occupied_bytes(),
            records.occupied_bytes(),
            budget.occupied_bytes()
        ),
        before
    );
    assert!(failure.error().is_none());
    safemlx::try_with_submission_retirement(|| drop((event, outputs, input))).unwrap();
    retire(&observer);
    drop((observer, scope, failure));
    safemlx::reclaim_allocation_owners();
    assert_eq!(
        (
            graph.occupied_bytes(),
            records.occupied_bytes(),
            budget.occupied_bytes()
        ),
        (0, 0, 0)
    );
}

#[test]
fn original_output_views_refuse_unscheduled_and_foreign_work() {
    let Some(layout) = layout() else { return };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let input = input();
    let foreign = [input.as_dtype(Dtype::Float16, &stream).unwrap()];
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
    let observer = OriginalScopeObserver::require_current().unwrap();
    let outputs = original_outputs(&input, layout, &observer, &stream);
    let before = (graph.occupied_bytes(), records.occupied_bytes());
    let hook = Hook::new();
    for roots in [&outputs[..], &foreign[..]] {
        for view in completed_outputs(roots, Some(&observer)) {
            assert!(view.is_err());
        }
    }
    assert_eq!(HOOKS.with(Cell::get), 0);
    drop(hook);
    assert_eq!((graph.occupied_bytes(), records.occupied_bytes()), before);
    assert!(!observer.status().has_work());
    safemlx::try_with_submission_retirement(|| drop((outputs, foreign, input))).unwrap();
    retire(&observer);
    scope.seal();
    drop((observer, scope, failure));
    safemlx::reclaim_allocation_owners();
    assert_eq!((graph.occupied_bytes(), records.occupied_bytes()), (0, 0));
}
