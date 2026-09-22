use super::*;
use crate::{
    reclaim_allocation_owners, PreparedPrefillFailure, PreparedSubmissionGraphQuota,
    PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner, SubmissionScope,
};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};

#[derive(Debug)]
struct Owner(Arc<AtomicUsize>);
impl Drop for Owner {
    fn drop(&mut self) {
        assert!(crate::can_reclaim_submission_resources());
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn settle(count: &AtomicUsize, expected: usize) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        crate::memory::clear_cache().unwrap();
        reclaim_allocation_owners();
        let actual = count.load(Ordering::SeqCst);
        if actual == expected {
            return;
        }
        assert!(actual < expected && Instant::now() < deadline);
        std::thread::yield_now();
    }
}
fn allocate<T: Send + 'static>(
    mut prepared: PreparedOriginalBufferBudget<'_, T>,
) -> OriginalBufferBudget {
    let limit = Instant::now() + Duration::from_secs(5);
    loop {
        match prepared.try_allocate() {
            Ok(value) => return value,
            Err(error) => {
                assert_eq!(error.cause(), OriginalBufferCause::RuntimeBusy);
                prepared = error.into_parts().1;
                assert!(Instant::now() < limit);
                std::thread::yield_now();
            }
        }
    }
}

#[test]
fn original_buffer_preparation_busy_preserves_node_and_final_handle_queues_owner() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let prepared =
        PreparedOriginalBufferBudget::try_new(&runtime, 4096, Owner(count.clone())).unwrap();
    let node = std::ptr::from_ref(&**prepared.node.as_ref().unwrap()) as usize;
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        ready_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    let entered = ready_rx.recv_timeout(Duration::from_secs(5)).is_ok();
    let result = prepared.try_allocate();
    let released = release_tx.send(()).is_ok();
    let held = worker.join().unwrap();
    assert!(entered && released && held);
    let error = result.unwrap_err();
    assert_eq!(error.cause(), OriginalBufferCause::RuntimeBusy);
    assert_eq!(
        std::ptr::from_ref(&**error.owner().node.as_ref().unwrap()) as usize,
        node
    );
    let budget = allocate(error.into_parts().1);
    let alias = budget.clone();
    assert!(budget.same_budget(&alias));
    assert_eq!(budget.capacity(), 4096);
    assert_eq!(budget.occupied_bytes(), 0);
    drop(budget);
    reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    // Direct owner destruction on this thread would violate Owner's unlocked
    // assertion. Another ordinary thread may legitimately reclaim the queue.
    let guard = runtime_lock::enter();
    drop(alias);
    drop(guard);
    settle(&count, 1);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn original_buffer_scope_binding_retains_budget_and_refused_setup_keeps_owner() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let budget =
        allocate(PreparedOriginalBufferBudget::try_new(&runtime, 0, Owner(count.clone())).unwrap());
    let rejected_count = Arc::new(AtomicUsize::new(0));
    let prepared =
        PreparedOriginalBufferBudget::try_new(&runtime, 1, Owner(rejected_count.clone())).unwrap();
    let node = std::ptr::from_ref(&**prepared.node.as_ref().unwrap()) as usize;
    let ordinary = crate::OrdinarySubmissionEntry::enter().unwrap();
    let graph = PreparedSubmissionGraphQuota::try_new(64 << 10, ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let records = PreparedSubmissionRecordQuota::try_new(64 << 10, ())
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
    assert_eq!(
        scope.bind_original_buffer_budget(&budget),
        Err(OriginalBufferCause::InvalidScope)
    );
    scope.enable_scoped_observation().unwrap();
    scope.require_original_native_controls().unwrap();
    failure.bind_original_scope(&scope).unwrap();
    scope.enable_original_native_controls().unwrap();
    scope.bind_original_buffer_budget(&budget).unwrap();
    assert_eq!(
        scope.bind_original_buffer_budget(&budget),
        Err(OriginalBufferCause::AlreadyBound)
    );
    let refused = prepared.try_allocate().unwrap_err();
    assert_eq!(refused.cause(), OriginalBufferCause::InvalidScope);
    assert_eq!(
        std::ptr::from_ref(&**refused.owner().node.as_ref().unwrap()) as usize,
        node
    );
    drop(ordinary);
    drop(refused);
    assert_eq!(rejected_count.load(Ordering::SeqCst), 1);
    drop(budget);
    reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    drop((scope, failure, records, graph));
    settle(&count, 1);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn original_buffer_authentication_does_not_adopt_ordinary_or_lazy_backing() {
    let stream = crate::test_stream();
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let budget = allocate(PreparedOriginalBufferBudget::try_new(&runtime, 64 << 10, ()).unwrap());
    let source = Array::from_slice(&[2.0f32, 3.0, 5.0], &[3]);
    let lazy = source.add(&source, stream).unwrap();
    let ordinary = crate::OrdinarySubmissionEntry::enter().unwrap();
    assert!(budget.inspect_array(&source).unwrap().is_none());
    assert!(budget.inspect_array(&lazy).unwrap().is_none());
    assert!(lazy.try_allocation_info().unwrap().is_none());
    assert_eq!(budget.occupied_bytes(), 0);
    drop(ordinary);
    let original = PreparedOriginalBufferBudget::try_new(&runtime, 0, 17u32)
        .unwrap()
        .into_owner();
    assert_eq!(original, 17);
}

#[test]
fn existing_alias_inspection_preserves_busy_and_never_adopts_ordinary_or_lazy_storage() {
    let stream = crate::test_stream();
    let source = Array::from_slice(&[31.0f32, 37.0, 41.0], &[3]);
    let lazy = source.add(&source, stream).unwrap();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        ready_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    let entered = ready_rx.recv_timeout(Duration::from_secs(5)).is_ok();
    let result = source.inspect_original_buffer_alias();
    let released = release_tx.send(()).is_ok();
    let held = worker.join().unwrap();
    assert!(entered && released && held);
    assert_eq!(result.unwrap_err(), OriginalBufferCause::RuntimeBusy);
    let ordinary = crate::OrdinarySubmissionEntry::enter().unwrap();
    assert!(source.inspect_original_buffer_alias().unwrap().is_none());
    assert!(lazy.inspect_original_buffer_alias().unwrap().is_none());
    assert!(lazy.try_allocation_info().unwrap().is_none());
    drop(ordinary);
}

#[test]
fn ordinary_buffer_witness_retains_nonzero_alias_and_never_classifies_lazy_storage() {
    let stream = crate::test_stream();
    let source = Array::from_slice(&[43.0f32, 47.0, 53.0], &[3]);
    let alias = source.clone();
    let lazy = source.add(&source, stream).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let owner = PreparedAllocationOwner::try_new(Owner(count.clone())).unwrap();
    let entry = crate::OrdinarySubmissionEntry::enter().unwrap();
    assert!(matches!(
        lazy.inspect_ordinary_buffer().unwrap(),
        OrdinaryBufferInspection::Unknown
    ));
    assert!(lazy.try_allocation_info().unwrap().is_none());
    let OrdinaryBufferInspection::Allocation(witness) = source.inspect_ordinary_buffer().unwrap()
    else {
        panic!("actual ordinary allocation must have a positive witness");
    };
    assert_eq!(
        Some(witness.allocation()),
        source.try_allocation_info().unwrap()
    );
    witness.try_attach(owner).unwrap();
    drop(entry);
    drop(source);
    reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert_eq!(
        alias.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        &[43.0, 47.0, 53.0]
    );
    drop((lazy, alias));
    settle(&count, 1);
}

#[test]
fn ordinary_buffer_checked_attachment_busy_returns_same_prepared_owner() {
    let source = Array::from_slice(&[59.0f32, 61.0, 67.0], &[3]);
    let count = Arc::new(AtomicUsize::new(0));
    let owner = PreparedAllocationOwner::try_new(Owner(count.clone())).unwrap();
    let owner_address = std::ptr::from_ref(owner.owner());
    let OrdinaryBufferInspection::Allocation(witness) = source.inspect_ordinary_buffer().unwrap()
    else {
        panic!("actual ordinary allocation must have a positive witness");
    };
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        ready_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    let entered = ready_rx.recv_timeout(Duration::from_secs(5)).is_ok();
    let result = witness.try_attach(owner);
    let released = release_tx.send(()).is_ok();
    let held = worker.join().unwrap();
    assert!(entered && released && held);
    let error = result.unwrap_err();
    assert_eq!(error.cause(), OriginalBufferCause::RuntimeBusy);
    assert_eq!(std::ptr::from_ref(error.owner().owner()), owner_address);
    assert_eq!(count.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[cfg(all(target_vendor = "apple", not(feature = "cuda")))]
#[test]
fn original_population_covers_independently_rounded_sources() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let lengths = [1usize, 2, 4096, 4097];
    let actual = lengths
        .iter()
        .map(|&elements| {
            crate::OriginalPromptInputFacts::inspect(&runtime, elements)
                .unwrap()
                .mutable_bytes()
        })
        .sum::<usize>();
    let requested = lengths.iter().sum::<usize>() * size_of::<u32>();
    let population =
        OriginalBufferBudget::population_layout(&runtime, requested, lengths.len()).unwrap();
    assert!(population.capacity() >= actual);
    assert!(population.control_bytes() > 0);
    // Four-byte scalar sources individually occupy a physical page. The
    // ordinary small-buffer allowance (strictly below twice the payload) is
    // insufficient even when its equation has a certified birth population.
    let scalar = OriginalBufferBudget::population_layout(&runtime, 4, 1).unwrap();
    assert!(
        scalar.capacity()
            >= crate::OriginalPromptInputFacts::inspect(&runtime, 1)
                .unwrap()
                .mutable_bytes()
    );
    assert!(scalar.capacity() > 7);
    assert_eq!(
        OriginalBufferBudget::population_layout(&runtime, 0, 0)
            .unwrap()
            .capacity(),
        0
    );
    let empty = OriginalBufferBudget::request_layout(&runtime, 0)
        .unwrap()
        .capacity();
    let empty_population = OriginalBufferBudget::population_layout(&runtime, 0, 8).unwrap();
    assert!(empty_population.capacity() >= 8 * empty);
    if empty == 0 {
        assert_eq!(empty_population.capacity(), 0);
    } else {
        assert!(empty_population.capacity() > 0);
        assert_eq!(
            OriginalBufferBudget::population_layout(&runtime, 0, usize::MAX),
            Err(OriginalBufferCause::InvalidLayout)
        );
    }
    let page = crate::memory::host_page_size().unwrap();
    let header = size_of::<usize>();
    for requests in [
        vec![0],
        vec![0, 0, 0],
        vec![1, 0, page - header, page - header + 1],
        vec![page - 1, page, page + 1, 2 * page + 1],
    ] {
        let actual: usize = requests
            .iter()
            .map(|&bytes| {
                OriginalBufferBudget::request_layout(&runtime, bytes)
                    .unwrap()
                    .capacity()
            })
            .sum();
        let bound = OriginalBufferBudget::population_layout(
            &runtime,
            requests.iter().sum(),
            requests.len(),
        )
        .unwrap()
        .capacity();
        assert!(bound >= actual, "requests={requests:?}: {bound} < {actual}");
        assert!(bound < actual + requests.len() * page);
        let larger_population = OriginalBufferBudget::population_layout(
            &runtime,
            requests.iter().sum(),
            requests.len() + 3,
        )
        .unwrap()
        .capacity();
        assert!(larger_population >= bound);
    }
    assert_eq!(
        OriginalBufferBudget::population_layout(&runtime, 1, 0),
        Err(OriginalBufferCause::InvalidLayout)
    );
    assert_eq!(
        OriginalBufferBudget::population_layout(&runtime, usize::MAX, 1),
        Err(OriginalBufferCause::InvalidLayout)
    );
    assert_eq!(
        OriginalBufferBudget::population_layout(&runtime, 1, usize::MAX),
        Err(OriginalBufferCause::InvalidLayout)
    );
}

#[test]
fn ordinary_attachment_rejects_conflicting_placement_before_consuming_owner() {
    let guard = runtime_lock::enter();
    let source = Array::from_slice(&[2.0f32, -3.0, 5.0], &[3]);
    let OrdinaryBufferInspection::Allocation(mut witness) =
        source.inspect_ordinary_buffer().unwrap()
    else {
        panic!("ordinary allocation must be certified");
    };
    assert_ne!(
        witness.allocation().placement(),
        crate::AllocationPlacement::Unknown
    );
    witness.facts.placement.kind = 0;
    let count = Arc::new(AtomicUsize::new(0));
    let prepared = PreparedAllocationOwner::try_new(Owner(count.clone())).unwrap();
    let rejected = witness.try_attach(prepared).unwrap_err();
    assert_eq!(rejected.cause(), OriginalBufferCause::BirthChanged);
    assert_eq!(count.load(Ordering::SeqCst), 0);
    drop(guard);
    drop(rejected);
    // Failed attachment never arms native retirement: the still-owned payload
    // is reclaimed by the Rust preparation owner immediately outside its loan.
    settle(&count, 1);
}
