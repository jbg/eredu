use super::*;
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{mpsc, Arc},
    time::{Duration, Instant},
};

struct DropWitness {
    drops: Rc<Cell<usize>>,
    probe_drops: Rc<Cell<usize>>,
}
impl Retention for DropWitness {
    fn observe(&self, _: Status) {}
}
impl Drop for DropWitness {
    fn drop(&mut self) {
        assert_eq!(
            self.probe_drops.get(),
            1,
            "probe must retire before custody"
        );
        assert!(!safemlx::can_reclaim_submission_resources());
        self.drops.set(self.drops.get() + 1);
    }
}
struct ControlledProbe {
    settled: Rc<Cell<bool>>,
    calls: Rc<Cell<usize>>,
    seals: Rc<Cell<usize>>,
    drops: Rc<Cell<usize>>,
    initial_unboxes: usize,
}
impl Probe for ControlledProbe {
    fn seal(&mut self) {
        self.seals.set(self.seals.get() + 1);
    }
    fn progress(&self) -> Status {
        self.calls.set(self.calls.get() + 1);
        Status {
            settled: self.settled.get(),
            failed: false,
            blocked: false,
        }
    }
}
impl Drop for ControlledProbe {
    fn drop(&mut self) {
        assert!(UNBOXED.with(Cell::get) > self.initial_unboxes);
        assert!(!safemlx::can_reclaim_submission_resources());
        self.drops.set(self.drops.get() + 1);
    }
}
fn controlled(
    settled: bool,
) -> (
    Recovery<DropWitness, ControlledProbe>,
    Rc<Cell<bool>>,
    Rc<Cell<usize>>,
    Rc<Cell<usize>>,
    Rc<Cell<usize>>,
) {
    let state = Rc::new(Cell::new(settled));
    let calls = Rc::new(Cell::new(0));
    let seals = Rc::new(Cell::new(0));
    let probe_drops = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let recovery = Recovery::with_probe(
        DropWitness {
            drops: drops.clone(),
            probe_drops: probe_drops.clone(),
        },
        ControlledProbe {
            settled: state.clone(),
            calls: calls.clone(),
            seals: seals.clone(),
            drops: probe_drops,
            initial_unboxes: UNBOXED.with(Cell::get),
        },
    );
    (recovery, state, calls, seals, drops)
}

#[test]
fn concrete_and_erased_nodes_unbox_before_probe_and_custody_under_the_guard() {
    let (mut direct, _, _, seals, drops) = controlled(true);
    direct.seal();
    let status = direct.finish().unwrap();
    assert!(status.settled && !status.failed && !status.blocked);
    assert_eq!(seals.get(), 1);
    assert_eq!(drops.get(), 1);

    let (pending, state, _, seals, drops) = controlled(false);
    drop(pending);
    assert_eq!(seals.get(), 1);
    assert_eq!(drops.get(), 0);
    state.set(true);
    reap();
    assert_eq!(drops.get(), 1);
}

#[test]
fn operation_unwind_seals_once_without_poisoning_or_observing_its_recovery() {
    let (recovery, state, calls, seals, drops) = controlled(false);
    let marker = Arc::new(());
    let panic = catch_unwind(AssertUnwindSafe({
        let marker = marker.clone();
        move || {
            let _recovery = recovery;
            std::panic::panic_any(marker);
        }
    }))
    .unwrap_err();
    assert!(Arc::ptr_eq(
        panic.downcast_ref::<Arc<()>>().unwrap(),
        &marker
    ));
    assert_eq!(seals.get(), 1);
    assert_eq!(calls.get(), 0);
    assert_eq!(drops.get(), 0);
    state.set(true);
    reap();
    assert_eq!(drops.get(), 1, "ordinary unwind is not callback poisoning");
}

struct PanickingRetention {
    armed: Rc<Cell<bool>>,
    calls: Rc<Cell<usize>>,
    drops: Rc<Cell<usize>>,
    marker: Arc<()>,
}
impl Retention for PanickingRetention {
    fn observe(&self, _: Status) {
        self.calls.set(self.calls.get() + 1);
        if self.armed.get() {
            std::panic::panic_any(self.marker.clone());
        }
    }
}
impl Drop for PanickingRetention {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}
struct PlainProbe(Rc<Cell<bool>>);
impl Probe for PlainProbe {
    fn seal(&mut self) {}
    fn progress(&self) -> Status {
        Status {
            settled: self.0.get(),
            failed: false,
            blocked: false,
        }
    }
}

#[test]
fn callback_panic_preserves_the_current_node_and_unvisited_snapshot() {
    let (tail, tail_state, _, _, tail_drops) = controlled(false);
    drop(tail);
    let state = Rc::new(Cell::new(false));
    let armed = Rc::new(Cell::new(false));
    let calls = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let marker = Arc::new(());
    drop(Recovery::with_probe(
        PanickingRetention {
            armed: armed.clone(),
            calls: calls.clone(),
            drops: drops.clone(),
            marker: marker.clone(),
        },
        PlainProbe(state.clone()),
    ));
    armed.set(true);
    state.set(true);
    tail_state.set(true);
    let panic = catch_unwind(AssertUnwindSafe(reap)).unwrap_err();
    assert!(Arc::ptr_eq(
        panic.downcast_ref::<Arc<()>>().unwrap(),
        &marker
    ));
    assert_eq!(drops.get(), 0);
    assert_eq!(
        tail_drops.get(),
        0,
        "unvisited nodes must not unwind as raw Boxes"
    );
    let calls_after_panic = calls.get();
    reap();
    assert_eq!(tail_drops.get(), 1);
    assert_eq!(drops.get(), 0);
    assert_eq!(
        calls.get(),
        calls_after_panic,
        "poisoned callback is never retried"
    );
}

fn with_foreign_runtime(f: impl FnOnce()) {
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let holder = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if safemlx::try_with_submission_retirement(|| {
                let _ = ready_tx.send(());
                let _ = release_rx.recv_timeout(Duration::from_secs(5));
            })
            .is_some()
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "could not acquire fixture runtime lock"
            );
            std::thread::yield_now();
        }
    });
    struct Release(
        Option<mpsc::Sender<()>>,
        Option<std::thread::JoinHandle<()>>,
    );
    impl Drop for Release {
        fn drop(&mut self) {
            if let Some(tx) = self.0.take() {
                let _ = tx.send(());
            }
            if let Some(thread) = self.1.take() {
                let _ = thread.join();
            }
        }
    }
    let _release = Release(Some(release_tx), Some(holder));
    ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    f();
}

#[test]
fn busy_fallback_callback_panic_retains_the_same_poisoned_node() {
    let calls = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let marker = Arc::new(());
    let recovery = Recovery::with_probe(
        PanickingRetention {
            armed: Rc::new(Cell::new(true)),
            calls: calls.clone(),
            drops: drops.clone(),
            marker: marker.clone(),
        },
        PlainProbe(Rc::new(Cell::new(true))),
    );
    with_foreign_runtime(|| {
        let panic = catch_unwind(AssertUnwindSafe(|| drop(recovery))).unwrap_err();
        assert!(Arc::ptr_eq(
            panic.downcast_ref::<Arc<()>>().unwrap(),
            &marker
        ));
        assert_eq!(drops.get(), 0);
    });
    assert_eq!(calls.get(), 1);
    reap();
    assert_eq!(calls.get(), 1);
    assert_eq!(drops.get(), 0);
}

struct SealPanic {
    marker: Arc<()>,
    calls: Rc<Cell<usize>>,
}
impl Probe for SealPanic {
    fn seal(&mut self) {
        self.calls.set(self.calls.get() + 1);
        std::panic::panic_any(self.marker.clone());
    }
    fn progress(&self) -> Status {
        panic!("failed seal must not be observed");
    }
}
#[test]
fn seal_panic_keeps_its_exact_payload_and_preallocated_owner() {
    let marker = Arc::new(());
    let calls = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    let recovery = Recovery::with_probe(
        PanickingRetention {
            armed: Rc::new(Cell::new(false)),
            calls: Rc::new(Cell::new(0)),
            drops: drops.clone(),
            marker: marker.clone(),
        },
        SealPanic {
            marker: marker.clone(),
            calls: calls.clone(),
        },
    );
    let panic = catch_unwind(AssertUnwindSafe(|| drop(recovery))).unwrap_err();
    assert!(Arc::ptr_eq(
        panic.downcast_ref::<Arc<()>>().unwrap(),
        &marker
    ));
    reap();
    assert_eq!(calls.get(), 1);
    assert_eq!(drops.get(), 0);
}

struct NativeProbe {
    scope: Rc<RefCell<SubmissionScope>>,
    ordinary_seals: Rc<Cell<usize>>,
    failure_seals: Rc<Cell<usize>>,
}
impl Probe for NativeProbe {
    fn seal(&mut self) {
        self.ordinary_seals.set(self.ordinary_seals.get() + 1);
        self.scope.borrow_mut().seal();
    }
    fn progress(&self) -> Status {
        let status = self.scope.borrow().progress();
        Status {
            settled: status.is_settled(),
            failed: status.failed(),
            blocked: status.blocked(),
        }
    }
    fn seal_after_callback_failure(&mut self) {
        self.failure_seals.set(self.failure_seals.get() + 1);
        self.scope.borrow_mut().seal();
    }
}
#[test]
fn callback_failure_before_sealing_closes_actual_thread_capture_without_releasing_it() {
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let outer = SubmissionScope::begin().unwrap();
    let scope = Rc::new(RefCell::new(SubmissionScope::begin().unwrap()));
    let ordinary_seals = Rc::new(Cell::new(0));
    let failure_seals = Rc::new(Cell::new(0));
    let marker = Arc::new(());
    let drops = Rc::new(Cell::new(0));
    let calls = Rc::new(Cell::new(0));
    let recovery = Recovery::with_probe(
        PanickingRetention {
            armed: Rc::new(Cell::new(true)),
            calls: calls.clone(),
            drops: drops.clone(),
            marker: marker.clone(),
        },
        NativeProbe {
            scope: scope.clone(),
            ordinary_seals: ordinary_seals.clone(),
            failure_seals: failure_seals.clone(),
        },
    );
    let panic = catch_unwind(AssertUnwindSafe(|| recovery.progress())).unwrap_err();
    assert!(Arc::ptr_eq(
        panic.downcast_ref::<Arc<()>>().unwrap(),
        &marker
    ));
    let blocked = recovery.progress();
    assert!(!blocked.settled && !blocked.failed && blocked.blocked);
    drop(recovery);
    assert_eq!(ordinary_seals.get(), 0);
    assert_eq!(failure_seals.get(), 1);
    assert_eq!(calls.get(), 1);
    let following = SubmissionScope::begin().unwrap();
    let values = safemlx::Array::from_slice(&[2_f32, 3., 5.], &[3]);
    let output = values.square(&stream).unwrap();
    safemlx::transforms::async_eval_with_event([&output])
        .unwrap()
        .synchronize()
        .unwrap();
    assert_eq!(
        output.evaluated().unwrap().as_slice::<f32>(),
        &[4., 9., 25.]
    );
    // A live following scope is a child of outer, not the poisoned scope.
    assert!(!outer.status().is_settled());
    assert!(scope.borrow().status().is_settled());
    assert_eq!(
        drops.get(),
        0,
        "native status does not release a poisoned callback population"
    );
    drop(following);
    reap();
    assert_eq!(drops.get(), 0);
}

struct ReentrantProbe(Rc<Cell<usize>>);
impl Probe for ReentrantProbe {
    fn seal(&mut self) {}
    fn progress(&self) -> Status {
        self.0.set(self.0.get() + 1);
        reap();
        Status {
            settled: true,
            failed: false,
            blocked: false,
        }
    }
}
#[test]
fn retirement_callback_can_inspect_the_orphan_queue_without_a_live_list_borrow() {
    let calls = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    drop(Recovery::with_probe(
        PanickingRetention {
            armed: Rc::new(Cell::new(false)),
            calls: Rc::new(Cell::new(0)),
            drops: drops.clone(),
            marker: Arc::new(()),
        },
        ReentrantProbe(calls.clone()),
    ));
    assert_eq!(calls.get(), 1);
    assert_eq!(drops.get(), 1);
}

struct ProgressPanic(Arc<()>);
impl Probe for ProgressPanic {
    fn seal(&mut self) {}
    fn progress(&self) -> Status {
        std::panic::panic_any(self.0.clone());
    }
}
#[test]
fn progress_panic_during_consuming_finish_preserves_the_original_payload_and_owner() {
    let marker = Arc::new(());
    let drops = Rc::new(Cell::new(0));
    let observations = Rc::new(Cell::new(0));
    let recovery = Recovery::with_probe(
        PanickingRetention {
            armed: Rc::new(Cell::new(false)),
            calls: observations.clone(),
            drops: drops.clone(),
            marker: marker.clone(),
        },
        ProgressPanic(marker.clone()),
    );
    let panic = catch_unwind(AssertUnwindSafe(|| recovery.finish())).unwrap_err();
    assert!(Arc::ptr_eq(
        panic.downcast_ref::<Arc<()>>().unwrap(),
        &marker
    ));
    assert_eq!(observations.get(), 0);
    reap();
    assert_eq!(drops.get(), 0);
}

#[test]
fn busy_queue_retains_every_detached_link_without_invoking_callbacks() {
    let (mut first, _, first_calls, _, first_drops) = controlled(false);
    let (mut second, _, second_calls, _, second_drops) = controlled(false);
    first.seal();
    second.seal();
    let mut first = first.node.take().unwrap().into_pending();
    let second = second.node.take().unwrap().into_pending();
    first.0.as_deref_mut().unwrap().set_next(Some(second));
    ORPHANS.with(|orphans| {
        let _loan = orphans.borrow_mut();
        // No provider operation is needed when the closed snapshot cannot
        // reenter its queue. Both original Boxes remain deliberately retained.
        drop(PendingList(Some(first)));
    });
    reap();
    assert_eq!(first_calls.get(), 0);
    assert_eq!(second_calls.get(), 0);
    assert_eq!(first_drops.get(), 0);
    assert_eq!(second_drops.get(), 0);
}
