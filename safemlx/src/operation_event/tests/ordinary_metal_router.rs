//! One retained GPU/CPU router pair, with the ordinary physical observer.
#![cfg(all(feature = "metal", not(feature = "cuda")))]

use crate as safemlx;
use safemlx::{
    Array, Device, DeviceType, OperationEvalTraversalLimits, OperationEvent,
    PhysicalBackingCustody, PhysicalBackingObserver, ScopedPhysicalBackingObserver, Stream,
    SubmissionGraphQuota, SubmissionScope,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct Ledger {
    limit: usize,
    total: AtomicUsize,
    live: AtomicUsize,
    refused: AtomicUsize,
    retired: AtomicUsize,
    refuse: AtomicBool,
}
struct Observer(Arc<Ledger>);
struct Charge(Arc<Ledger>, usize);
impl Drop for Observer {
    fn drop(&mut self) {
        self.0.retired.fetch_add(1, Ordering::SeqCst);
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        assert!(self.0.live.fetch_sub(self.1, Ordering::SeqCst) >= self.1);
    }
}
impl PhysicalBackingObserver for Observer {
    fn admit(
        &self,
        facts: safemlx::AllocationInfo,
    ) -> safemlx::error::Result<PhysicalBackingCustody> {
        let bytes = facts.host_control_bytes();
        if self.0.refuse.load(Ordering::SeqCst)
            || self
                .0
                .total
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |old| {
                    old.checked_add(bytes).filter(|sum| *sum <= self.0.limit)
                })
                .is_err()
        {
            self.0.refused.fetch_add(1, Ordering::SeqCst);
            return Err(safemlx::error::Exception::custom(
                "ordinary mixed control source refused",
            ));
        }
        self.0.live.fetch_add(bytes, Ordering::SeqCst);
        Ok(PhysicalBackingCustody::new(Charge(self.0.clone(), bytes)))
    }
}
fn ledger(limit: usize) -> Arc<Ledger> {
    Arc::new(Ledger {
        limit,
        total: 0.into(),
        live: 0.into(),
        refused: 0.into(),
        retired: 0.into(),
        refuse: false.into(),
    })
}
fn graph(input: &Array, gpu: &Stream, cpu: &Stream) -> safemlx::error::Result<Array> {
    let values = input.add(input, gpu)?;
    let indices = safemlx::ops::argpartition_axis(&values, 0, -1, cpu)?;
    let converted = indices.as_type::<f32>(gpu)?;
    converted.add(&converted, gpu)
}
fn reclaim(gpu: &Stream, cpu: &Stream) {
    gpu.synchronize().unwrap();
    cpu.synchronize().unwrap();
    safemlx::try_retire_completed_submissions().unwrap();
    safemlx::reclaim_allocation_owners();
}
fn settle(scope: &SubmissionScope) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !scope.progress().is_settled() {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
}
fn verify(output: &Array) {
    let mut values = [0.0f32; 4];
    output
        .evaluated()
        .unwrap()
        .try_copy_into(&mut values)
        .unwrap();
    assert_eq!(values[0], 0.0);
    values.sort_by(f32::total_cmp);
    assert_eq!(values, [0.0, 2.0, 4.0, 6.0]);
}
fn allowance() -> usize {
    // The actual graph has three GPU numerical entries, one CPU partition and
    // one GPU Synchronizer. The input was constructed before this request.
    let traversal = OperationEvalTraversalLimits {
        roots: 1,
        arrays: 6,
        tape_entries: 5,
        input_edges: 7,
        output_slots: 5,
        streams: 2,
        captures: 6,
    };
    assert!(OperationEvent::ordinary_metal_eval_control_layout(traversal).is_none());
    for streams in [0, 1, 3] {
        assert!(
            OperationEvent::ordinary_metal_router_eval_control_layout(
                OperationEvalTraversalLimits {
                    streams,
                    ..traversal
                }
            )
            .is_none()
        );
    }
    let eval = OperationEvent::ordinary_metal_router_eval_control_layout(traversal).unwrap();
    assert_eq!(eval.platform_events(), 4);
    // Three GPU outputs and the two possible fast Fence backing births; the
    // CPU output/task bank is separate. One real consumer wait follows Eval.
    let worker = OperationEvent::resident_gpu_worker_layout_with_router_frontiers(
        4, 6, 4, 6, 5, 2, 4, 0, 1, 1, 1,
    )
    .unwrap();
    let without_wait = OperationEvent::resident_gpu_worker_layout_with_router_frontiers(
        4, 6, 4, 6, 5, 2, 4, 0, 1, 1, 0,
    )
    .unwrap();
    assert!(worker.allocation_extents() > without_wait.allocation_extents());
    assert!(
        OperationEvent::resident_gpu_worker_layout_with_router_frontiers(
            4, 6, 4, 6, 5, 2, 4, 0, 1, 2, 0,
        )
        .is_none()
    );
    let prologue = OperationEvent::gpu_eval_prologue_layout(6, 0).unwrap();
    let population = prologue.population(4, 6, 0).unwrap();
    let mut extents = worker.allocation_extents();
    for (bytes, _) in prologue.requests()[..2]
        .iter()
        .copied()
        .chain(prologue.owner_requests())
    {
        extents = extents
            .checked_add(
                SubmissionGraphQuota::allocation_population_extent(
                    bytes.checked_mul(4).unwrap(),
                    4,
                )
                .unwrap(),
            )
            .unwrap();
    }
    for (bytes, count) in [
        (
            population.data_slot_bytes(),
            population.vector_allocations(),
        ),
        (population.output_array_bytes(), population.evaluations()),
    ] {
        extents = extents
            .checked_add(SubmissionGraphQuota::allocation_population_extent(bytes, count).unwrap())
            .unwrap();
    }
    extents = extents
        .checked_add(
            OperationEvent::cpu_argpartition_layout(false)
                .unwrap()
                .allocation_extents(),
        )
        .unwrap();
    let sources = [
        OperationEvent::ordinary_frontend_control_layout(4, 0, 2, 4).unwrap(),
        eval,
        OperationEvent::ordinary_dispatch_control_envelope(extents).unwrap(),
        OperationEvent::ordinary_array_vector_control_layout(1).unwrap(),
        OperationEvent::ordinary_metal_wait_record_control_layout().unwrap(),
    ];
    // Numerical payload capacities are a separate source; this test constrains
    // the ordinary native control population and all six possible birth headers.
    sources
        .into_iter()
        .try_fold(
            safemlx::physical_backing_control_bytes()
                .checked_mul(6)
                .unwrap(),
            |total, source| total.checked_add(source.observed_host_control_bytes()),
        )
        .unwrap()
}

#[test]
fn ordinary_metal_router_keeps_finite_controls_alias_refusal_and_recovery() {
    let gpu = Stream::try_new_with_device(&Device::new(DeviceType::Gpu, 0)).unwrap();
    let cpu = Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0)).unwrap();
    let input = Array::try_from_slice(&[1.0f32, 4.0, 2.0, 3.0], &[1, 4]).unwrap();
    // Process worker/kernel sources precede the bounded request.
    let warm = graph(&input, &gpu, &cpu).unwrap();
    verify(&warm);
    drop(warm);
    reclaim(&gpu, &cpu);

    let counts = ledger(allowance());
    let observer =
        ScopedPhysicalBackingObserver::new_uncached(Arc::new(Observer(counts.clone()))).unwrap();
    let mut scope = SubmissionScope::begin().unwrap();
    scope.bind_physical_observer(&observer).unwrap();
    let output = graph(&input, &gpu, &cpu).unwrap();
    let done = safemlx::transforms::async_eval_with_event([&output]).unwrap();
    done.wait_on(&gpu).unwrap();
    done.synchronize().unwrap();
    verify(&output);
    let alias = output.try_clone_handle().unwrap();
    scope.seal();
    settle(&scope);
    drop((output, done, scope, observer));
    reclaim(&gpu, &cpu);
    assert!(counts.live.load(Ordering::SeqCst) > 0);
    assert_eq!(counts.retired.load(Ordering::SeqCst), 0);
    assert_eq!(counts.refused.load(Ordering::SeqCst), 0);
    drop(alias);
    reclaim(&gpu, &cpu);
    assert_eq!(counts.live.load(Ordering::SeqCst), 0);
    assert_eq!(counts.retired.load(Ordering::SeqCst), 1);

    // Refuse after real lazy construction, at completion preparation. Neither
    // the retained graph nor an escaped native cause may lose its payer.
    let refused = ledger(allowance());
    let observer =
        ScopedPhysicalBackingObserver::new_uncached(Arc::new(Observer(refused.clone()))).unwrap();
    let mut scope = SubmissionScope::begin().unwrap();
    scope.bind_physical_observer(&observer).unwrap();
    let output = graph(&input, &gpu, &cpu).unwrap();
    refused.refuse.store(true, Ordering::SeqCst);
    let error = safemlx::transforms::async_eval_with_event([&output]).unwrap_err();
    assert!(refused.refused.load(Ordering::SeqCst) > 0);
    scope.seal();
    settle(&scope);
    drop((scope, observer));
    assert!(refused.live.load(Ordering::SeqCst) > 0);
    assert_eq!(refused.retired.load(Ordering::SeqCst), 0);
    drop((output, error));
    reclaim(&gpu, &cpu);
    assert_eq!(refused.live.load(Ordering::SeqCst), 0);
    assert_eq!(refused.retired.load(Ordering::SeqCst), 1);
    let recovered = graph(&input, &gpu, &cpu).unwrap();
    verify(&recovered);
}
