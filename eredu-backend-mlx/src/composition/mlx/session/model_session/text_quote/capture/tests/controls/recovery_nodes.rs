use super::*;
use crate::backend::submission_recovery::{self, Probe, Recovery, Retention, Status};
use crate::composition::mlx::session::model_session::ScopeRetention;
use std::sync::Arc;

struct OriginalRetention {
    _root: Array,
    pool: WorkingMemoryPool,
    expected: u64,
    probe_dropped: Rc<Cell<bool>>,
    panic_on_observe: Option<Arc<()>>,
    // The real original work/custody is retained by this final field.
    ticket: ScopeRetention,
}
impl Retention for OriginalRetention {
    fn observe(&self, status: Status) {
        self.ticket.observe(status);
        if let Some(marker) = &self.panic_on_observe {
            std::panic::panic_any(marker.clone());
        }
    }
}
impl Drop for OriginalRetention {
    fn drop(&mut self) {
        assert!(self.probe_dropped.get());
        assert!(!safemlx::can_reclaim_submission_resources());
        assert!(self.pool.used_bytes().unwrap() >= self.expected);
    }
}
struct ProbeOrder {
    settled: Rc<Cell<bool>>,
    dropped: Rc<Cell<bool>>,
    initial_unboxes: usize,
    pool: WorkingMemoryPool,
    expected: u64,
}
impl Probe for ProbeOrder {
    fn seal(&mut self) {}
    fn progress(&self) -> Status {
        Status {
            settled: self.settled.get(),
            failed: false,
            blocked: false,
        }
    }
}
impl Drop for ProbeOrder {
    fn drop(&mut self) {
        assert!(submission_recovery::test_node_unbox_count() > self.initial_unboxes);
        assert!(!safemlx::can_reclaim_submission_resources());
        assert!(self.pool.used_bytes().unwrap() >= self.expected);
        self.dropped.set(true);
    }
}

#[test]
fn recovery_node_keeps_actual_original_custody_through_typed_and_erased_retirement() {
    for terminal_at_finish in [true, false] {
        let (pool, resources, authority, expected) = original_submission_resources();
        let root = Array::from_slice(&[2_f32, 3., 5.], &[3]);
        safemlx::transforms::eval([&root]).unwrap();
        assert_eq!(root.evaluated().unwrap().as_slice::<f32>(), &[2., 3., 5.]);
        let settled = Rc::new(Cell::new(terminal_at_finish));
        let probe_dropped = Rc::new(Cell::new(false));
        let retention = OriginalRetention {
            _root: root,
            pool: pool.clone(),
            expected,
            probe_dropped: probe_dropped.clone(),
            panic_on_observe: None,
            ticket: resources.ticket(),
        };
        let mut recovery = Recovery::with_probe(
            retention,
            ProbeOrder {
                settled: settled.clone(),
                dropped: probe_dropped.clone(),
                initial_unboxes: submission_recovery::test_node_unbox_count(),
                pool: pool.clone(),
                expected,
            },
        );
        resources.request_release();
        drop(resources);
        assert!(authority.require_idle().is_err());
        assert_eq!(pool.used_bytes().unwrap(), expected);
        recovery.seal();
        if terminal_at_finish {
            let status = recovery.finish().unwrap();
            assert!(status.settled && !status.failed && !status.blocked);
        } else {
            let marker = Arc::new(());
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
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
            assert!(!probe_dropped.get());
            assert!(authority.require_idle().is_err());
            assert_eq!(pool.used_bytes().unwrap(), expected);
            settled.set(true);
            submission_recovery::wait_for_retirement(|| authority.require_idle().is_ok());
        }
        assert!(probe_dropped.get());
        assert!(authority.require_idle().is_ok());
        settle_terminal(&pool, 0);
    }
}

#[test]
fn poisoned_real_scope_retains_original_account_and_lease_after_callback_unwind() {
    let (pool, resources, authority, expected) = original_submission_resources();
    let scope = safemlx::SubmissionScope::begin().unwrap();
    let root = Array::from_slice(&[7_u32], &[1]);
    safemlx::transforms::eval([&root]).unwrap();
    assert_eq!(root.evaluated().unwrap().as_slice::<u32>(), &[7]);
    let marker = Arc::new(());
    let mut recovery = Recovery::with_probe(
        OriginalRetention {
            _root: root,
            pool: pool.clone(),
            expected,
            probe_dropped: Rc::new(Cell::new(false)),
            panic_on_observe: Some(marker.clone()),
            ticket: resources.ticket(),
        },
        scope,
    );
    resources.request_release();
    drop(resources);
    let panic =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| recovery.progress())).unwrap_err();
    assert!(Arc::ptr_eq(
        panic.downcast_ref::<Arc<()>>().unwrap(),
        &marker
    ));
    let status = recovery.progress();
    assert!(!status.settled && !status.failed && status.blocked);
    // This invokes the actual SubmissionScope bookkeeping-only override. The
    // node stays poisoned even though native work was already settled.
    recovery.seal();
    drop(recovery);
    for _ in 0..3 {
        submission_recovery::reap();
    }
    assert!(authority.require_idle().is_err());
    assert_eq!(pool.used_bytes().unwrap(), expected);
}
