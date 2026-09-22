//! Actual CPU stream crossings under one finite ordinary physical payer.
use safemlx::{
    Array, CpuBinaryOperation, Device, DeviceType, Dtype, OperationEvalTraversalLimits,
    OperationEvent, PhysicalBackingCustody, PhysicalBackingObserver, ScopedPhysicalBackingObserver,
    Stream, SubmissionScope,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct Ledger {
    limit: usize,
    total: AtomicUsize,
    live: AtomicUsize,
    refused: AtomicUsize,
    retired: AtomicUsize,
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
        assert_eq!(facts.placement(), safemlx::AllocationPlacement::Host);
        let bytes = facts.host_control_bytes();
        if self
            .0
            .total
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |old| {
                old.checked_add(bytes).filter(|sum| *sum <= self.0.limit)
            })
            .is_err()
        {
            self.0.refused.fetch_add(1, Ordering::SeqCst);
            return Err(safemlx::error::Exception::custom(
                "finite ordinary CPU control allowance exhausted",
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
    })
}
fn reclaim(streams: &[&Stream]) {
    for stream in streams {
        stream.synchronize().unwrap();
    }
    safemlx::try_retire_completed_submissions().unwrap();
    safemlx::reclaim_allocation_owners();
}
fn graph(input: &Array, a: &Stream, b: &Stream) -> safemlx::error::Result<Array> {
    let first = input.add(input, a)?;
    let second = first.multiply(input, b)?;
    let third = second.add(input, a)?;
    third.multiply(input, b)
}
fn allowance() -> usize {
    let primitive = |operation| {
        OperationEvent::cpu_binary_layout(operation, Dtype::Float32, 1, 3, false).unwrap()
    };
    let add = primitive(CpuBinaryOperation::Add);
    let multiply = primitive(CpuBinaryOperation::Multiply);
    let completion = OperationEvent::cpu_completion_layout(1).unwrap();
    let extents = add
        .graph_allocation_extents()
        .checked_add(add.worker_graph_allocation_extents())
        .unwrap()
        .checked_add(multiply.graph_allocation_extents())
        .unwrap()
        .checked_add(multiply.worker_graph_allocation_extents())
        .unwrap()
        .checked_mul(2)
        .unwrap()
        .checked_add(completion.graph_allocation_extents())
        .unwrap();
    let sources = [
        OperationEvent::ordinary_frontend_control_layout(4, 0, 1, 4).unwrap(),
        OperationEvent::ordinary_cpu_eval_control_layout(OperationEvalTraversalLimits {
            roots: 1,
            arrays: 6,
            tape_entries: 5,
            input_edges: 9,
            output_slots: 5,
            streams: 2,
            captures: 6,
        })
        .unwrap(),
        OperationEvent::ordinary_cpu_dispatch_envelope(extents).unwrap(),
        OperationEvent::ordinary_array_vector_control_layout(1).unwrap(),
    ];
    let births = add
        .backing_births()
        .checked_add(multiply.backing_births())
        .unwrap()
        .checked_mul(2)
        .unwrap()
        .checked_add(completion.backing_births())
        .unwrap();
    sources
        .into_iter()
        .try_fold(
            births
                .checked_mul(safemlx::physical_backing_control_bytes())
                .unwrap(),
            |total, source| total.checked_add(source.observed_host_control_bytes()),
        )
        .unwrap()
}

#[test]
fn ordinary_cpu_crossings_keep_finite_source_alias_and_refusal() {
    let device = Device::new(DeviceType::Cpu, 0);
    let a = Stream::try_new_with_device(&device).unwrap();
    let b = Stream::try_new_with_device(&device).unwrap();
    let input = Array::try_from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
    // The actual CPU worker registrations are process sources established first.
    for stream in [&a, &b] {
        let warm = input.add(&input, stream).unwrap();
        warm.evaluated().unwrap();
    }
    reclaim(&[&a, &b]);
    let counts = ledger(allowance());
    let observer =
        ScopedPhysicalBackingObserver::new_uncached(Arc::new(Observer(counts.clone()))).unwrap();
    let mut scope = SubmissionScope::begin().unwrap();
    scope.bind_physical_observer(&observer).unwrap();
    let output = graph(&input, &a, &b).unwrap();
    let done = safemlx::transforms::async_eval_with_event([&output]).unwrap();
    done.synchronize().unwrap();
    assert_eq!(
        output.evaluated().unwrap().as_slice::<f32>(),
        &[3.0, 20.0, 63.0]
    );
    let alias = output.try_clone_handle().unwrap();
    scope.seal();
    assert!(scope.progress().is_settled());
    drop((output, done, scope, observer));
    reclaim(&[&a, &b]);
    assert!(counts.live.load(Ordering::SeqCst) > 0);
    assert_eq!(counts.retired.load(Ordering::SeqCst), 0);
    assert_eq!(counts.refused.load(Ordering::SeqCst), 0);
    drop(alias);
    reclaim(&[&a, &b]);
    assert_eq!(counts.live.load(Ordering::SeqCst), 0);
    assert_eq!(counts.retired.load(Ordering::SeqCst), 1);

    let refused = ledger(0);
    let observer =
        ScopedPhysicalBackingObserver::new_uncached(Arc::new(Observer(refused.clone()))).unwrap();
    let mut scope = SubmissionScope::begin().unwrap();
    scope.bind_physical_observer(&observer).unwrap();
    let error = graph(&input, &a, &b).unwrap_err();
    assert!(refused.refused.load(Ordering::SeqCst) > 0);
    assert_eq!(refused.total.load(Ordering::SeqCst), 0);
    scope.seal();
    assert!(scope.progress().is_settled());
    drop((error, scope, observer));
    reclaim(&[&a, &b]);
    assert_eq!(refused.live.load(Ordering::SeqCst), 0);
    assert_eq!(refused.retired.load(Ordering::SeqCst), 1);
    let recovered = graph(&input, &a, &b).unwrap();
    assert_eq!(
        recovered.evaluated().unwrap().as_slice::<f32>(),
        &[3.0, 20.0, 63.0]
    );
}
