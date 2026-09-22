use super::*;
use crate::backend::submission_recovery::{RecoveryBeginError, prefill::PrefillRequestRetention};
use eredu_core::{InferenceGeometry, OutputDemand};
use eredu_runtime::working_memory::{InferenceExecutionIdentity, InferenceRequest};
use safemlx::{Device, DeviceType, SubmissionScopeBeginError};
use std::{cell::Cell, sync::mpsc, time::Duration};

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|n| n.set(n.get() + 1));
}
struct Hook;
impl Drop for Hook {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn actual_mechanism_try_entry_returns_typed_busy_and_exact_request_without_housekeeping() {
    type Architecture = eredu_architectures::llama::LayeredModel<MlxNeuralBackend>;
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut mechanism = MlxReplicatedTextMechanisms::<Architecture, MlxKeyValueState>::new(
        Arc::new(eredu_checkpoint::store::MemoryWeightStore::default()),
        &stream,
        &stream,
    )
    .unwrap();
    // This bridge checks the real method and exact owner. Charged request/array
    // lifetime is covered separately by the recovery tests; no byte proof here.
    let request = crate::memory_fixture::empty_admitted_request(
        &InferenceExecutionIdentity::default(),
        InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: 1,
            max_output_tokens: 0,
            prefill_chunk_positions: 1,
            output: OutputDemand::LastPosition,
        },
    )
    .unwrap();
    let identity = request.clone();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = safemlx::RuntimeCallDeadline::new(Duration::from_secs(5))
            .unwrap()
            .enter()
            .unwrap();
        entered_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let _hook = Hook;
    HOUSEKEEPING.with(|n| n.set(0));
    let error = mechanism
        .begin_prefill_reservation(request)
        .err()
        .expect("one nonblocking try");
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    let Error::Other(owner) = &error else {
        panic!("typed owner");
    };
    let owner = owner
        .downcast_ref::<RecoveryBeginError<PrefillRequestRetention>>()
        .unwrap();
    assert!(matches!(owner.cause(), SubmissionScopeBeginError::Busy));
    owner
        .retention()
        .0
        .validate_same_request(&identity)
        .unwrap();
    let released = release_tx.send(()).is_ok();
    assert!(
        worker.join().unwrap() && released,
        "try-entry waited for runtime lock"
    );
    drop(error);
    let guard = mechanism.begin_prefill_reservation(identity).unwrap();
    mechanism.finish_prefill_reservation(guard).unwrap();
}

#[test]
fn actual_mechanisms_initialize_selected_runtime_before_first_request() {
    type Architecture = eredu_architectures::llama::LayeredModel<MlxNeuralBackend>;
    let device = Device::new(DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let weights = Stream::new_with_device(&device);
    let mechanism = MlxReplicatedTextMechanisms::<Architecture, MlxKeyValueState>::new(
        Arc::new(eredu_checkpoint::store::MemoryWeightStore::default()),
        &stream,
        &weights,
    )
    .unwrap();
    let baseline = mechanism.prefill_roots_runtime.baseline().unwrap();
    assert_eq!(baseline.selected_streams, 2);
    assert_eq!(baseline.cpu_workers, 2);
    assert_eq!(baseline.native_threads, 2);
    assert!(baseline.fixed_header_bytes().unwrap() > 0);
    assert_ne!(baseline.unpriced_populations, 0);
    assert!(mechanism.prefill_controls.borrow().is_none());
    assert!(mechanism.prefill_roots.is_none());
}

#[test]
fn coordinated_ordinary_entry_waits_then_releases_the_runtime_before_returning() {
    for role in [
        None,
        Some(eredu_runtime::prefill::PrefillControlRole::SourcePreparation),
    ] {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (start_tx, start_rx) = mpsc::channel();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            type Architecture = eredu_architectures::llama::LayeredModel<MlxNeuralBackend>;
            let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
            let mut mechanism = MlxReplicatedTextMechanisms::<Architecture, MlxKeyValueState>::new(
                Arc::new(eredu_checkpoint::store::MemoryWeightStore::default()),
                &stream,
                &stream,
            )
            .unwrap();
            let request = crate::memory_fixture::empty_admitted_request(
                &InferenceExecutionIdentity::default(),
                InferenceGeometry {
                    batch_size: 1,
                    cached_positions: 0,
                    input_positions: 1,
                    max_output_tokens: 0,
                    prefill_chunk_positions: 1,
                    output: OutputDemand::LastPosition,
                },
            )
            .unwrap();
            safemlx::register_thread_runtime_housekeeping(housekeeping);
            let _hook = Hook;
            HOUSEKEEPING.with(|n| n.set(0));
            ready_tx.send(()).unwrap();
            start_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let guard = mechanism.coordinate_prefill_entry(request, role).unwrap();
            assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
            entered_tx.send(()).unwrap();
            finish_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            mechanism.finish_prefill_reservation(guard).unwrap();
        });
        ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let owner = safemlx::RuntimeCallDeadline::new(Duration::from_secs(5))
            .unwrap()
            .enter()
            .unwrap();
        start_tx.send(()).unwrap();
        let waited = matches!(
            entered_rx.recv_timeout(Duration::from_millis(20)),
            Err(mpsc::RecvTimeoutError::Timeout)
        );
        drop(owner);
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        // The live reservation guard keeps its resources, but not the runtime
        // loan: workers and later native calls must be able to make progress.
        let available = safemlx::RuntimeCallDeadline::new(Duration::from_millis(100))
            .unwrap()
            .enter();
        let released = available.is_ok();
        drop(available);
        finish_tx.send(()).unwrap();
        worker.join().unwrap();
        assert!(waited, "ordinary caller must coordinate before its attempt");
        assert!(
            released,
            "entry loan must not cover later completion or prefetch"
        );
    }
}
