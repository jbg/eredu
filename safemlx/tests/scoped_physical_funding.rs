//! Ordinary backing births use their actual native scope, including aliases.
use safemlx::{
    Array, PhysicalBackingCustody, PhysicalBackingObserver, ScopedPhysicalBackingObserver,
    SubmissionScope,
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Barrier,
};

#[derive(Default)]
struct Counts {
    accepted: AtomicUsize,
    live: AtomicUsize,
    payload_accepted: AtomicUsize,
    payload_live: AtomicUsize,
    refuse: AtomicBool,
    observers_retired: AtomicUsize,
}
struct Observer(Arc<Counts>);
impl Drop for Observer {
    fn drop(&mut self) {
        self.0.observers_retired.fetch_add(1, Ordering::SeqCst);
    }
}
struct Charge(Arc<Counts>, bool);
impl Drop for Charge {
    fn drop(&mut self) {
        assert!(self.0.live.fetch_sub(1, Ordering::SeqCst) > 0);
        if self.1 {
            assert!(self.0.payload_live.fetch_sub(1, Ordering::SeqCst) > 0);
        }
    }
}
impl PhysicalBackingObserver for Observer {
    fn admit(
        &self,
        facts: safemlx::AllocationInfo,
    ) -> safemlx::error::Result<PhysicalBackingCustody> {
        if self.0.refuse.load(Ordering::SeqCst) {
            return Err(safemlx::error::Exception::custom(
                "assigned fixture allowance exhausted",
            ));
        }
        self.0.accepted.fetch_add(1, Ordering::SeqCst);
        self.0.live.fetch_add(1, Ordering::SeqCst);
        let payload = facts.bytes() != 0;
        if payload {
            self.0.payload_accepted.fetch_add(1, Ordering::SeqCst);
            self.0.payload_live.fetch_add(1, Ordering::SeqCst);
        }
        Ok(PhysicalBackingCustody::new(Charge(self.0.clone(), payload)))
    }
}
fn reclaim() {
    safemlx::memory::clear_cache().unwrap();
    safemlx::reclaim_allocation_owners();
}

#[test]
fn observer_owner_waits_for_explicit_unlocked_reclamation() {
    reclaim();
    let counts = Arc::new(Counts::default());
    let observer = ScopedPhysicalBackingObserver::new(Arc::new(Observer(counts.clone()))).unwrap();
    let mut scope = SubmissionScope::begin().unwrap();
    scope.bind_physical_observer(&observer).unwrap();
    scope.seal();
    drop((scope, observer));
    assert_eq!(counts.observers_retired.load(Ordering::SeqCst), 0);
    safemlx::reclaim_allocation_owners();
    assert_eq!(counts.observers_retired.load(Ordering::SeqCst), 1);
}

#[test]
fn scopes_route_independently_and_backing_charge_outlives_scope_and_observer() {
    reclaim();
    let previous = safemlx::memory::set_cache_limit(0).unwrap();
    let ready = Arc::new(Barrier::new(2));
    let mut workers = Vec::new();
    for seed in [3.0f32, 7.0] {
        let ready = ready.clone();
        workers.push(std::thread::spawn(move || {
            let counts = Arc::new(Counts::default());
            let observer =
                ScopedPhysicalBackingObserver::new(Arc::new(Observer(counts.clone()))).unwrap();
            let mut scope = SubmissionScope::begin().unwrap();
            scope.bind_physical_observer(&observer).unwrap();
            ready.wait();
            let mut child = SubmissionScope::begin().unwrap();
            let value = Array::try_from_slice(&[seed, -seed, seed + 1.0], &[3]).unwrap();
            let alias = value.try_clone_handle().unwrap();
            assert_eq!(counts.payload_accepted.load(Ordering::SeqCst), 1);
            let accepted = counts.accepted.load(Ordering::SeqCst);
            counts.refuse.store(true, Ordering::SeqCst);
            let refused = Array::try_from_slice(&[seed; 1024], &[1024]).unwrap_err();
            let mut cause: &(dyn std::error::Error + 'static) = &refused;
            while let Some(source) = cause.source() {
                cause = source;
            }
            assert!(cause
                .to_string()
                .contains("assigned fixture allowance exhausted"));
            assert_eq!(counts.payload_accepted.load(Ordering::SeqCst), 1);
            assert_eq!(counts.accepted.load(Ordering::SeqCst), accepted);
            counts.refuse.store(false, Ordering::SeqCst);
            child.seal();
            scope.seal();
            drop((child, scope, observer, value));
            assert_eq!(counts.payload_live.load(Ordering::SeqCst), 1);
            assert_eq!(
                alias.evaluated().unwrap().as_slice::<f32>(),
                &[seed, -seed, seed + 1.0]
            );
            (alias, counts)
        }));
    }
    let retained = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect::<Vec<_>>();
    for (alias, counts) in retained {
        assert_eq!(counts.payload_live.load(Ordering::SeqCst), 1);
        drop(alias);
        reclaim();
        assert_eq!(counts.live.load(Ordering::SeqCst), 0);
        assert_eq!(counts.observers_retired.load(Ordering::SeqCst), 1);
    }
    safemlx::memory::set_cache_limit(previous).unwrap();
}

#[test]
fn queued_cpu_work_keeps_exact_scope_funding_after_public_handles_drop() {
    queued_work_keeps_exact_scope_funding(safemlx::DeviceType::Cpu, false);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires an actual Metal device"]
fn queued_metal_work_retires_scope_funding_after_native_completion() {
    queued_work_keeps_exact_scope_funding(safemlx::DeviceType::Gpu, false);
}

#[test]
fn queued_uncached_cpu_work_retires_after_its_last_completed_alias() {
    queued_work_keeps_exact_scope_funding(safemlx::DeviceType::Cpu, true);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires an actual Metal device"]
fn queued_uncached_metal_work_retires_after_its_last_completed_alias() {
    queued_work_keeps_exact_scope_funding(safemlx::DeviceType::Gpu, true);
}

fn queued_work_keeps_exact_scope_funding(device: safemlx::DeviceType, uncached: bool) {
    reclaim();
    let previous = safemlx::memory::set_cache_limit(if uncached { 1 << 24 } else { 0 }).unwrap();
    let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(device, 0));
    let counts = Arc::new(Counts::default());
    let owner = Arc::new(Observer(counts.clone()));
    let observer = if uncached {
        ScopedPhysicalBackingObserver::new_uncached(owner)
    } else {
        ScopedPhysicalBackingObserver::new(owner)
    }
    .unwrap();
    let mut scope = SubmissionScope::begin().unwrap();
    scope.bind_physical_observer(&observer).unwrap();
    let lhs = Array::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let rhs = Array::try_from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
    let output = lhs.matmul(&rhs, &stream).unwrap();
    let completion = safemlx::transforms::async_eval_with_event([&output]).unwrap();
    scope.seal();
    drop((scope, observer, lhs, rhs));
    completion.synchronize().unwrap();
    let mut actual = [0.0f32; 4];
    output
        .evaluated()
        .unwrap()
        .try_copy_into(&mut actual)
        .unwrap();
    assert_eq!(actual, [58.0, 64.0, 139.0, 154.0]);
    assert!(counts.accepted.load(Ordering::SeqCst) >= 3);
    assert!(counts.live.load(Ordering::SeqCst) > 0);
    drop((output, completion));
    stream.synchronize().unwrap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(counts.live.load(Ordering::SeqCst), 0);
    assert_eq!(counts.observers_retired.load(Ordering::SeqCst), 1);
    safemlx::memory::set_cache_limit(previous).unwrap();
}

#[test]
fn uncached_source_retires_new_roots_but_preserves_reused_cache_custody() {
    reclaim();
    let previous = safemlx::memory::set_cache_limit(1 << 24).unwrap();
    let cached_counts = Arc::new(Counts::default());
    let cached_observer =
        ScopedPhysicalBackingObserver::new(Arc::new(Observer(cached_counts.clone()))).unwrap();
    let mut cached_scope = SubmissionScope::begin().unwrap();
    cached_scope
        .bind_physical_observer(&cached_observer)
        .unwrap();
    let cached = Array::try_from_slice(&[3.0f32; 128], &[128]).unwrap();
    let original = cached.allocation_info().unwrap().unwrap();
    cached_scope.seal();
    drop((cached, cached_scope, cached_observer));
    safemlx::reclaim_allocation_owners();
    assert_eq!(cached_counts.live.load(Ordering::SeqCst), 1);

    let counts = Arc::new(Counts::default());
    let observer =
        ScopedPhysicalBackingObserver::new_uncached(Arc::new(Observer(counts.clone()))).unwrap();
    let mut scope = SubmissionScope::begin().unwrap();
    scope.bind_physical_observer(&observer).unwrap();
    let reused = Array::try_from_slice(&[7.0f32; 128], &[128]).unwrap();
    assert_eq!(reused.allocation_info().unwrap(), Some(original));
    assert_eq!(counts.payload_accepted.load(Ordering::SeqCst), 0);
    let fresh = Array::try_from_slice(&[11.0f32; 32768], &[32768]).unwrap();
    let alias = fresh.try_clone_handle().unwrap();
    assert_eq!(counts.payload_accepted.load(Ordering::SeqCst), 1);
    scope.seal();
    drop((scope, observer, reused, fresh));
    safemlx::reclaim_allocation_owners();
    assert_eq!(counts.payload_live.load(Ordering::SeqCst), 1);
    assert_eq!(cached_counts.live.load(Ordering::SeqCst), 1);
    assert_eq!(alias.evaluated().unwrap().as_slice::<f32>()[0], 11.0);
    drop(alias);
    // Reclaim only actual retired owners: the process cache stays enabled.
    safemlx::reclaim_allocation_owners();
    assert_eq!(counts.live.load(Ordering::SeqCst), 0);
    assert_eq!(counts.observers_retired.load(Ordering::SeqCst), 1);
    assert_eq!(cached_counts.live.load(Ordering::SeqCst), 1);
    reclaim();
    assert_eq!(cached_counts.live.load(Ordering::SeqCst), 0);
    safemlx::memory::set_cache_limit(previous).unwrap();
}

#[test]
fn fresh_cpu_source_bypasses_warmed_cache_and_retires_after_last_alias() {
    fresh_source_bypasses_warmed_cache(safemlx::DeviceType::Cpu);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires an actual Metal device"]
fn fresh_metal_source_bypasses_warmed_cache_and_retires_after_last_alias() {
    fresh_source_bypasses_warmed_cache(safemlx::DeviceType::Gpu);
}

fn fresh_source_bypasses_warmed_cache(device: safemlx::DeviceType) {
    reclaim();
    let previous = safemlx::memory::set_cache_limit(1 << 24).unwrap();
    let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(device, 0));
    let cached_counts = Arc::new(Counts::default());
    let cached_observer =
        ScopedPhysicalBackingObserver::new(Arc::new(Observer(cached_counts.clone()))).unwrap();
    let mut cached_scope = SubmissionScope::begin().unwrap();
    cached_scope
        .bind_physical_observer(&cached_observer)
        .unwrap();
    let left = Array::try_from_slice(&[2.0f32; 128], &[8, 16]).unwrap();
    let right = Array::try_from_slice(&[3.0f32; 128], &[16, 8]).unwrap();
    let output = left.matmul(&right, &stream).unwrap();
    let completion = safemlx::transforms::async_eval_with_event([&output]).unwrap();
    completion.synchronize().unwrap();
    let cached_roots = [&left, &right, &output].map(|array| {
        assert_eq!(array.try_observe_availability().unwrap(), Some(true));
        array.allocation_info().unwrap().unwrap().identity()
    });
    cached_scope.seal();
    drop((
        left,
        right,
        output,
        completion,
        cached_scope,
        cached_observer,
    ));
    stream.synchronize().unwrap();
    safemlx::reclaim_allocation_owners();
    let cached_live = cached_counts.payload_live.load(Ordering::SeqCst);
    assert!(cached_live >= 3);

    let counts = Arc::new(Counts::default());
    let observer =
        ScopedPhysicalBackingObserver::new_fresh(Arc::new(Observer(counts.clone()))).unwrap();
    let mut scope = SubmissionScope::begin().unwrap();
    scope.bind_physical_observer(&observer).unwrap();
    // Even an exact old cache hit must enter the current source's admission.
    counts.refuse.store(true, Ordering::SeqCst);
    assert!(Array::try_from_slice(&[5.0f32; 128], &[8, 16]).is_err());
    assert_eq!(counts.payload_accepted.load(Ordering::SeqCst), 0);
    assert_eq!(
        cached_counts.payload_live.load(Ordering::SeqCst),
        cached_live
    );
    counts.refuse.store(false, Ordering::SeqCst);
    let mut child = SubmissionScope::begin().unwrap();
    let left = Array::try_from_slice(&[5.0f32; 128], &[8, 16]).unwrap();
    let right = Array::try_from_slice(&[7.0f32; 128], &[16, 8]).unwrap();
    let output = left.matmul(&right, &stream).unwrap();
    let alias = output.try_clone_handle().unwrap();
    let completion = safemlx::transforms::async_eval_with_event([&output]).unwrap();
    child.seal();
    scope.seal();
    drop((child, scope, observer));
    completion.synchronize().unwrap();
    for array in [&left, &right, &output] {
        assert_eq!(array.try_observe_availability().unwrap(), Some(true));
        assert!(!cached_roots.contains(&array.allocation_info().unwrap().unwrap().identity()));
    }
    assert!(counts.payload_accepted.load(Ordering::SeqCst) >= 3);
    drop((left, right, output, completion));
    stream.synchronize().unwrap();
    safemlx::reclaim_allocation_owners();
    assert!(counts.payload_live.load(Ordering::SeqCst) > 0);
    assert_eq!(
        cached_counts.payload_live.load(Ordering::SeqCst),
        cached_live
    );
    let mut values = [0.0f32; 64];
    alias
        .evaluated()
        .unwrap()
        .try_copy_into(&mut values)
        .unwrap();
    assert!(values.iter().all(|&value| value == 560.0));
    drop(alias);
    stream.synchronize().unwrap();
    // No cache eviction or global cache-policy change is needed for this source.
    safemlx::reclaim_allocation_owners();
    assert_eq!(counts.live.load(Ordering::SeqCst), 0);
    assert_eq!(counts.observers_retired.load(Ordering::SeqCst), 1);
    assert_eq!(
        cached_counts.payload_live.load(Ordering::SeqCst),
        cached_live
    );
    reclaim();
    assert_eq!(cached_counts.live.load(Ordering::SeqCst), 0);
    safemlx::memory::set_cache_limit(previous).unwrap();
}
