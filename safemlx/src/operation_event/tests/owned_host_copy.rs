use super::*;
use crate::array::{OwnedHostCopyCause, OwnedHostCopyPlan, OwnedHostCopyStrategy};
use crate::PreparedInputRuntime;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct SourceCustody(Arc<AtomicUsize>);
impl Drop for SourceCustody {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn reclaim() {
    crate::allocation_retention::reclaim_allocation_owners();
}
fn copied_completion(kind: DeviceType) {
    let stream = Stream::new_with_device(&Device::new(kind, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let values = vec![2.0_f32, -3.0, 7.0];
    let plan = OwnedHostCopyPlan::<f32>::new(&runtime, &[3], values.capacity()).unwrap();
    assert_eq!(
        plan.facts().strategy(),
        OwnedHostCopyStrategy::CopyToImmutablePreparedBacking
    );
    assert_eq!(plan.facts().copy_bytes(), 12);
    assert_eq!(plan.facts().maximum_input_bytes(), values.capacity() * 4);
    assert!(plan.control_bytes::<SourceCustody>().is_some());
    let retired = Arc::new(AtomicUsize::new(0));
    let preparation = PreparedSubmissionGraphQuota::try_new(
        plan.facts().metadata_bytes(),
        SourceCustody(retired.clone()),
    )
    .unwrap();
    let slot = plan.prepare(preparation).unwrap();
    let original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let graph = original.graph.occupied_bytes();
    let records = original._records.occupied_bytes();
    crate::register_thread_runtime_housekeeping(hook);
    let hooks = Hook;
    HOOKS.with(|value| value.set(0));
    let copied = slot.try_fill(values, &observer).unwrap();
    let doubled = copied.add(&copied, &stream).unwrap();
    let event = crate::transforms::async_eval_with_original_operation_event_on_stream_exact(
        std::slice::from_ref(&doubled),
        &observer,
        &stream,
    )
    .unwrap();
    event.synchronize().unwrap();
    assert_eq!(HOOKS.with(Cell::get), 0);
    drop(hooks);
    settle(&observer);
    observer.validate_completed_array(&doubled).unwrap();
    assert_eq!(
        doubled
            .completed_in_original_scope(&observer)
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap(),
        &[4.0, -6.0, 14.0]
    );
    // A real immutable cache owner can retain the prepared final handle beyond
    // the producer/slot/role-independent source owner wrappers.
    let cache = Arc::new(copied);
    let alias = Arc::clone(&cache);
    crate::try_with_submission_retirement(|| drop((doubled, event))).unwrap();
    settle(&observer);
    drop(cache);
    reclaim();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    std::thread::spawn(move || drop(alias)).join().unwrap();
    reclaim();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert_eq!(original.graph.occupied_bytes(), graph);
    assert_eq!(original._records.occupied_bytes(), records);
}
#[test]
fn prepared_owned_copy_cpu_preserves_completion_and_shared_final_handle_custody() {
    copied_completion(DeviceType::Cpu);
}
#[cfg(all(feature = "metal", target_vendor = "apple", not(feature = "cuda")))]
#[test]
fn prepared_owned_copy_metal_preserves_completion_and_shared_final_handle_custody() {
    copied_completion(DeviceType::Gpu);
}
#[test]
fn prepared_owned_copy_busy_and_foreign_owner_keep_actual_vec_and_ready_slot() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let values = vec![13_i64, -7];
    let pointer = values.as_ptr();
    let capacity = values.capacity();
    let plan = OwnedHostCopyPlan::<i64>::new(&runtime, &[2], capacity).unwrap();
    let retired = Arc::new(AtomicUsize::new(0));
    let preparation = PreparedSubmissionGraphQuota::try_new(
        plan.facts().metadata_bytes(),
        SourceCustody(retired.clone()),
    )
    .unwrap();
    let slot = plan.prepare(preparation).unwrap();
    let mut first = Original::new();
    let first_observer = OriginalScopeObserver::require_current().unwrap();
    let (entered, ready) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        crate::try_with_submission_retirement(|| {
            entered.send(()).unwrap();
            released.recv().unwrap();
        })
        .unwrap();
    });
    ready.recv().unwrap();
    let mut error = slot.try_fill(values, &first_observer).unwrap_err();
    assert_eq!(error.cause(), OwnedHostCopyCause::Busy);
    assert!(!error.attempted());
    assert_eq!(error.values().as_ptr(), pointer);
    assert_eq!(error.values().capacity(), capacity);
    assert_eq!(error.values(), &[13, -7]);
    let busy_message = error.native_source().unwrap().what().as_ptr();
    let busy = error.take_native_source().unwrap();
    assert_eq!(busy.what().as_ptr(), busy_message);
    assert_eq!(
        busy.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::RuntimeBusy)
    );
    assert!(error.take_native_source().is_none());
    assert!(error.native_source().is_none());
    assert!(std::error::Error::source(&error).is_none());
    assert!(!error.attempted());
    assert_eq!(error.values().as_ptr(), pointer);
    assert_eq!(error.values().capacity(), capacity);
    release.send(()).unwrap();
    worker.join().unwrap();
    first.scope.seal();
    let second = Original::new();
    let second_observer = OriginalScopeObserver::require_current().unwrap();
    let mut error = error.retry(&first_observer).unwrap_err();
    assert_eq!(error.cause(), OwnedHostCopyCause::Domain);
    assert!(!error.attempted());
    assert_eq!(error.values().as_ptr(), pointer);
    assert_eq!(error.values().capacity(), capacity);
    let domain = error.take_native_source().unwrap();
    assert_eq!(
        domain.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Domain)
    );
    assert_eq!(
        busy.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::RuntimeBusy)
    );
    assert!(error.take_native_source().is_none());
    reclaim();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    let copied = error.retry(&second_observer).unwrap();
    assert_eq!(
        copied
            .completed_in_original_scope(&second_observer)
            .unwrap()
            .try_as_slice::<i64>()
            .unwrap(),
        &[13, -7]
    );
    drop(copied);
    reclaim();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    settle(&second_observer);
    drop(second);
    settle(&first_observer);
    drop(first);
    // Both actual diagnostic snapshots survive the attempted input's retirement.
    assert_eq!(busy.what().as_ptr(), busy_message);
    assert_eq!(
        domain.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Domain)
    );
    drop((busy, domain));
}
#[test]
fn prepared_owned_copy_input_capacity_refusal_retains_vec_and_cold_owner() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let mut values = Vec::<f32>::with_capacity(8);
    values.push(23.0);
    let pointer = values.as_ptr();
    let capacity = values.capacity();
    let plan = OwnedHostCopyPlan::<f32>::new(&runtime, &[1], 1).unwrap();
    let retired = Arc::new(AtomicUsize::new(0));
    let preparation = PreparedSubmissionGraphQuota::try_new(
        plan.facts().metadata_bytes(),
        SourceCustody(retired.clone()),
    )
    .unwrap();
    let slot = plan.prepare(preparation).unwrap();
    let original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let error = slot.try_fill(values, &observer).unwrap_err();
    assert_eq!(error.cause(), OwnedHostCopyCause::Invalid);
    assert!(!error.attempted());
    assert_eq!(error.values().as_ptr(), pointer);
    assert_eq!(error.values().capacity(), capacity);
    assert_eq!(error.values(), &[23.0]);
    reclaim();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(error);
    reclaim();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    settle(&observer);
    drop(original);
    let plan = OwnedHostCopyPlan::<f32>::new(&runtime, &[], 1).unwrap();
    let wrong = PreparedSubmissionGraphQuota::try_new(
        plan.facts().metadata_bytes() + 64,
        SourceCustody(retired.clone()),
    )
    .unwrap();
    let error = plan.prepare(wrong).unwrap_err();
    assert_eq!(error.cause(), OwnedHostCopyCause::Invalid);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    drop(error);
    assert_eq!(retired.load(Ordering::SeqCst), 2);
    assert_eq!(
        OwnedHostCopyPlan::<bool>::new(&runtime, &[], 1).unwrap_err(),
        OwnedHostCopyCause::Unsupported
    );
}

struct FixedBuffer<T = i64> {
    values: Vec<T>,
    accesses: Arc<AtomicUsize>,
    _receipt: BufferReceipt,
}
struct BufferReceipt(Arc<AtomicUsize>);
impl Drop for BufferReceipt {
    fn drop(&mut self) {
        assert!(crate::utils::runtime_lock::can_reclaim_submission_resources());
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
impl<T: crate::ArrayElement> crate::OwnedHostCopyBuffer<T> for FixedBuffer<T> {
    fn as_slice(&self) -> &[T] {
        assert!(crate::utils::runtime_lock::can_reclaim_submission_resources());
        self.accesses.fetch_add(1, Ordering::SeqCst);
        &self.values
    }
    fn capacity(&self) -> usize {
        assert!(crate::utils::runtime_lock::can_reclaim_submission_resources());
        self.accesses.fetch_add(1, Ordering::SeqCst);
        self.values.capacity()
    }
}
#[test]
fn owning_host_buffer_is_borrowed_before_native_loan_and_retired_after_copy_or_refusal() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let source_retired = Arc::new(AtomicUsize::new(0));
    let buffer_retired = Arc::new(AtomicUsize::new(0));
    let accesses = Arc::new(AtomicUsize::new(0));
    let original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    for refuse in [true, false] {
        let buffer = FixedBuffer {
            values: vec![17, -29],
            accesses: accesses.clone(),
            _receipt: BufferReceipt(buffer_retired.clone()),
        };
        let pointer = buffer.values.as_ptr();
        let capacity = buffer.values.capacity();
        let plan = OwnedHostCopyPlan::<i64>::new(&runtime, &[2], if refuse { 1 } else { capacity });
        // The pure plan rejects too-small input ceilings; use a one-element
        // shape to reach the actual buffer-capacity refusal in the fill worker.
        let plan = if refuse {
            OwnedHostCopyPlan::<i64>::new(&runtime, &[1], 1).unwrap()
        } else {
            plan.unwrap()
        };
        assert!(plan
            .control_bytes_with_buffer::<SourceCustody, FixedBuffer>()
            .is_some());
        let arena = PreparedSubmissionGraphQuota::try_new(
            plan.facts().metadata_bytes(),
            SourceCustody(source_retired.clone()),
        )
        .unwrap();
        let slot = plan.prepare(arena).unwrap();
        let result = slot.try_fill_owned(buffer, &observer);
        assert_eq!(accesses.swap(0, Ordering::SeqCst), 2);
        if refuse {
            let failure = result.unwrap_err();
            assert_eq!(failure.cause(), OwnedHostCopyCause::Invalid);
            assert!(!failure.attempted());
            assert_eq!(failure.buffer().values.as_ptr(), pointer);
            assert_eq!(failure.buffer().values.capacity(), capacity);
            assert_eq!(failure.buffer().values, [17, -29]);
            assert_eq!(buffer_retired.load(Ordering::SeqCst), 0);
            drop(failure);
            reclaim();
            assert_eq!(source_retired.load(Ordering::SeqCst), 1);
        } else {
            let output = result.unwrap();
            assert_eq!(buffer_retired.load(Ordering::SeqCst), 2);
            assert_eq!(
                output
                    .completed_in_original_scope(&observer)
                    .unwrap()
                    .try_as_slice::<i64>()
                    .unwrap(),
                &[17, -29]
            );
            assert_eq!(source_retired.load(Ordering::SeqCst), 1);
            let alias = Arc::new(output);
            let retained = Arc::clone(&alias);
            drop(alias);
            reclaim();
            assert_eq!(source_retired.load(Ordering::SeqCst), 1);
            drop(retained);
            reclaim();
            assert_eq!(source_retired.load(Ordering::SeqCst), 2);
        }
    }
    settle(&observer);
    drop(original);
}

#[test]
fn completed_immutable_copy_checked_attachment_preserves_busy_owner_and_final_custody() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let values = vec![3.5f32, -0.0, -11.25];
    let plan = OwnedHostCopyPlan::<f32>::new(&runtime, &[3], values.capacity()).unwrap();
    let expected_backing = plan.facts().backing_bytes();
    let retired = Arc::new(AtomicUsize::new(0));
    let attached = Arc::new(AtomicUsize::new(0));
    let metadata_bytes = plan.facts().metadata_bytes();
    let slot = plan
        .prepare(
            PreparedSubmissionGraphQuota::try_new(metadata_bytes, SourceCustody(retired.clone()))
                .unwrap(),
        )
        .unwrap();
    let original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let accesses = Arc::new(AtomicUsize::new(0));
    let buffer_retired = Arc::new(AtomicUsize::new(0));
    let input = FixedBuffer {
        values,
        accesses: accesses.clone(),
        _receipt: BufferReceipt(buffer_retired.clone()),
    };
    let completed = slot.try_fill_owned_completed(input, &observer).unwrap();
    assert_eq!(accesses.load(Ordering::SeqCst), 2);
    assert_eq!(buffer_retired.load(Ordering::SeqCst), 1);
    let allocation = completed
        .allocation()
        .expect("actual nonzero immutable birth");
    assert_eq!(allocation.bytes(), expected_backing);
    let crate::ImmutableSourceInspection::Allocation(witness) = completed.observe().unwrap() else {
        panic!("completed nonzero copy must be positively immutable");
    };
    assert_eq!(witness.allocation(), allocation);
    let owner = crate::PreparedAllocationOwner::try_new(SourceCustody(attached.clone())).unwrap();
    let pointer = std::ptr::from_ref(owner.owner());
    let (entered, ready) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _loan = crate::utils::runtime_lock::enter();
        entered.send(()).unwrap();
        released.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    let ready = ready.recv_timeout(Duration::from_secs(5)).is_ok();
    let result = witness.try_attach(owner);
    let sent = release.send(()).is_ok();
    let held = worker.join().unwrap();
    assert!(ready && sent && held);
    let error = result.unwrap_err();
    assert_eq!(error.cause(), crate::OriginalBufferCause::RuntimeBusy);
    assert_eq!(std::ptr::from_ref(error.owner().owner()), pointer);
    assert_eq!(attached.load(Ordering::SeqCst), 0);
    let crate::ImmutableSourceInspection::Allocation(witness) = completed.observe().unwrap() else {
        panic!("unchanged completion");
    };
    witness.try_attach(error.into_parts().1).unwrap();
    let output = completed.into_array();
    assert_eq!(
        output
            .completed_in_original_scope(&observer)
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        [3.5f32, -0.0, -11.25].map(f32::to_bits)
    );
    assert!(output.inspect_original_buffer_alias().unwrap().is_none());
    let alias = Arc::new(output);
    let other = alias.clone();
    settle(&observer);
    drop((observer, original, alias));
    reclaim();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(attached.load(Ordering::SeqCst), 0);
    std::thread::spawn(move || drop(other)).join().unwrap();
    reclaim();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert_eq!(attached.load(Ordering::SeqCst), 1);
}
