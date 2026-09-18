//! Exercise the real pre-grant original route without enabling mock native copy.
use super::*;
use eredu_core::HostPreparationAuthority;
use eredu_runtime::{
    execution_control::{PreparedTextHostCopy, TextHostCopyError},
    working_memory::{WorkingMemoryError, WorkspaceCopyLimits},
};
use std::cell::Cell;

thread_local! {
    static PROBE: Cell<Option<bool>> = const { Cell::new(None) };
    static COLD_CALLS: Cell<usize> = const { Cell::new(0) };
}
struct Probe;
impl Drop for Probe {
    fn drop(&mut self) {
        PROBE.with(|probe| probe.set(None));
    }
}
pub(in super::super) fn ordinary_hook() {
    assert!(
        PROBE.with(Cell::get).is_none(),
        "ordinary estimator ran before original host grant"
    );
}
pub(in super::super) fn preparation_bytes() -> Option<u64> {
    // No mock native planner is constructed. The host provider always refuses
    // below, before the separately unpriced native copy could be requested.
    PROBE.with(Cell::get).map(|_| 0)
}
pub(in super::super) fn estimate(
    runtime: &ModelRuntime<MockBackend>,
    pending: Option<PendingTextInput<&Prompt, &MockToken>>,
) -> Option<[SnapshotEstimate; 3]> {
    let known = PROBE.with(Cell::get)?;
    COLD_CALLS.with(|calls| calls.set(calls.get() + 1));
    if !known {
        return None;
    }
    runtime.session().authority.require_idle().ok()?;
    let native =
        64u64.checked_add(u64::try_from(runtime.session().intervention_identity.len()).ok()?)?;
    let input = match pending {
        None => 0,
        Some(PendingTextInput::Decode(_)) => 4,
        Some(PendingTextInput::Prefill(ids)) => {
            24u64.checked_add(u64::try_from(ids.len()).ok()?.checked_mul(4)?)?
        }
    };
    Some([native, 64, input].map(|bytes| SnapshotEstimate {
        retained_bytes: bytes,
        copy_bytes: bytes,
    }))
}

pub(in super::super) fn probe_estimate(
    runtime: &ModelRuntime<MockBackend>,
    pending: Option<PendingTextInput<&Prompt, &MockToken>>,
) -> Option<Option<[SnapshotEstimate; 3]>> {
    PROBE.with(Cell::get).map(|_| estimate(runtime, pending))
}

struct RefusingHost<'a>(&'a Cell<bool>);
impl PreparedTextHostCopy for RefusingHost<'_> {
    type Copied = ();
    fn storage_bytes(&self) -> Option<u64> {
        Some(0)
    }
    fn original_preparation_bytes(&self) -> Option<u64> {
        Some(0)
    }
    fn original_control_bytes(&self) -> Option<usize> {
        TextContinuationSnapshot::<MockBackend, Controller>::original_capture_control_bytes::<()>()
    }
    fn copy(self, _: u64) -> Result<(), TextHostCopyError> {
        panic!("ordinary host copy must not run")
    }
    fn copy_original(self, _: u64) -> Result<((), HostPreparationAuthority), TextHostCopyError> {
        self.0.set(true);
        Err(TextHostCopyError::Admission(
            WorkingMemoryError::UnknownBound,
        ))
    }
}

#[test]
fn original_estimates_bypass_ordinary_hooks_and_keep_budget_admission_order() {
    for known in [false, true] {
        let mut runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = ManagedTextContinuation::root(
            driver
                .start(vec![11, 7, 3].into(), config(), Controller::default())
                .unwrap(),
        );
        assert!(advance(&mut state, &mut driver).is_some());
        let before = state.controller().0.clone();
        let budget = budget();
        let called = Cell::new(false);
        COLD_CALLS.with(|calls| calls.set(0));
        PROBE.with(|probe| probe.set(Some(known)));
        let probe = Probe;
        let result = {
            let mut boundary = state.boundary(&mut driver).unwrap();
            TextContinuationSnapshot::capture_original_host(
                &mut boundary.snapshot_source(),
                &budget,
                RefusingHost(&called),
                WorkspaceCopyLimits::new(1 << 20),
            )
        };
        assert_eq!(COLD_CALLS.with(Cell::get), 1);
        assert_eq!(called.get(), known);
        if known {
            assert!(matches!(
                result,
                Err(TextSnapshotError::HostAdmission(
                    WorkingMemoryError::UnknownBound
                ))
            ));
            assert!(budget.usage().cumulative_copy_bytes > 0);
        } else {
            assert!(matches!(
                result,
                Err(TextSnapshotError::Control(
                    ExecutionControlError::UnknownEstimate
                ))
            ));
            assert_eq!(budget.usage(), SnapshotUsage::default());
        }
        assert_eq!(budget.usage().retained_bytes, 0);
        assert_eq!(budget.usage().snapshots, 0);
        assert_eq!(state.controller().0, before);
        drop(probe);
        assert!(
            advance(&mut state, &mut driver).is_some(),
            "rejection changed continuation readiness"
        );
    }
}
