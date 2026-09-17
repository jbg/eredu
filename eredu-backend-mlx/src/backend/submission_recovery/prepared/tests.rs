use super::*;
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
struct Retained {
    calls: Rc<Cell<usize>>,
    drops: Rc<Cell<usize>>,
    unboxed: usize,
}
impl Retention for Retained {
    fn observe(&self, _: Status) {
        self.calls.set(self.calls.get() + 1);
    }
}
impl Drop for Retained {
    fn drop(&mut self) {
        assert!(!safemlx::can_reclaim_submission_resources());
        assert!(UNBOXED.with(Cell::get) > self.unboxed);
        self.drops.set(self.drops.get() + 1);
        reap();
    }
}
struct Capsule {
    drops: Arc<AtomicUsize>,
    phase: Arc<AtomicUsize>,
}
impl Drop for Capsule {
    fn drop(&mut self) {
        // Unattached host-only custody can drop during ordinary unwinding.
        // Record the phase without risking a second panic in this destructor.
        let phase = usize::from(safemlx::can_reclaim_submission_resources())
            | (usize::from(std::thread::panicking()) << 1);
        self.phase.store(phase, Ordering::SeqCst);
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn never_started_preparation_unboxes_guarded_without_any_observation_even_after_unwind() {
    for unwind in [false, true] {
        let calls = Rc::new(Cell::new(0));
        let drops = Rc::new(Cell::new(0));
        let native = Arc::new(AtomicUsize::new(0));
        let phase = Arc::new(AtomicUsize::new(usize::MAX));
        let ready = PreparedRecovery::new(
            Retained {
                calls: calls.clone(),
                drops: drops.clone(),
                unboxed: UNBOXED.with(Cell::get),
            },
            Capsule {
                drops: native.clone(),
                phase: phase.clone(),
            },
        )
        .ok()
        .unwrap();
        let marker = Arc::new(());
        let expected = marker.clone();
        if unwind {
            let panic = catch_unwind(AssertUnwindSafe(|| {
                let _ready = ready;
                std::panic::panic_any(marker);
            }))
            .unwrap_err();
            assert!(Arc::ptr_eq(
                panic.downcast_ref::<Arc<()>>().unwrap(),
                &expected
            ));
            assert_eq!(drops.get(), 0);
        } else {
            drop(ready);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while drops.get() == 0 {
            reap();
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        safemlx::reclaim_allocation_owners();
        assert_eq!(calls.get(), 0);
        assert_eq!(drops.get(), 1);
        assert_eq!(native.load(Ordering::SeqCst), 1);
        assert_eq!(phase.load(Ordering::SeqCst), if unwind { 2 } else { 1 });
    }
}
#[test]
fn prepared_node_handoff_tracks_real_cpu_work_and_keeps_constructor_custody_until_delete() {
    let calls = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let native = Arc::new(AtomicUsize::new(0));
    let phase = Arc::new(AtomicUsize::new(usize::MAX));
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let input = safemlx::Array::from_slice(&[2_f32, -3., 5.], &[3]);
    let mut ready = PreparedRecovery::new(
        Retained {
            calls: calls.clone(),
            drops: drops.clone(),
            unboxed: UNBOXED.with(Cell::get),
        },
        Capsule {
            drops: native.clone(),
            phase: phase.clone(),
        },
    )
    .ok()
    .unwrap();
    let same = ready.allocation_identity();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut active = loop {
        match ready.try_begin() {
            Ok(active) => break active,
            Err(error) => {
                assert_eq!(error.cause, SubmissionScopeOwnerCause::RuntimeBusy);
                ready = error.pending;
                assert_eq!(ready.allocation_identity(), same);
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    };
    let output = input.square(&stream).unwrap();
    let event = safemlx::transforms::async_eval_with_event([&output]).unwrap();
    active.seal();
    event.synchronize().unwrap();
    // Event signaling precedes the CPU task's terminal frontier publication.
    // Keep polling actual scope evidence; failed or blocked status still fails.
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        let status = active.progress();
        if status.settled || status.failed || status.blocked {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "native scope did not settle after event completion: {status:?}"
        );
        std::thread::yield_now();
    };
    assert!(
        status.settled && !status.failed && !status.blocked,
        "native scope failed after event completion: {status:?}"
    );
    assert!(calls.get() > 0);
    assert_eq!(native.load(Ordering::SeqCst), 0);
    assert_eq!(
        output.evaluated().unwrap().as_slice::<f32>(),
        &[4., 9., 25.]
    );
    drop((event, output, input));
    active.finish();
    while drops.get() == 0 || native.load(Ordering::SeqCst) == 0 {
        reap();
        // Settled native records still own Scope references until this pass.
        safemlx::try_retire_completed_submissions().unwrap();
        safemlx::reclaim_allocation_owners();
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(drops.get(), 1);
    assert_eq!(native.load(Ordering::SeqCst), 1);
    assert_eq!(
        phase.load(Ordering::SeqCst),
        1,
        "attached custody retires unlocked after unwind"
    );
}
