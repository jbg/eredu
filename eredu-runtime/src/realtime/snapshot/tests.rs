use super::*;
use crate::execution_control::{IsolatedStateSnapshot, SnapshotBudget};
use eredu_core::{
    execution_control::SnapshotLimits, scheduler::SemanticStateTransaction, Completion,
};
use std::{cell::Cell, convert::Infallible, rc::Rc};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Value(u64);
impl SnapshotValue for Value {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        Ok(SnapshotEstimate {
            retained_bytes: 8,
            copy_bytes: 8,
        })
    }
    fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(self.clone())
    }
}
#[derive(Clone)]
struct Schedule {
    identity: Value,
    position: u64,
}
impl RealtimeScheduleState for Schedule {
    type Identity = Value;
    type Error = Failure;
    fn identity(&self) -> &Value {
        &self.identity
    }
    fn validate_successor(&self, next: &Self) -> Result<(), Failure> {
        if next.identity != self.identity || next.position != self.position + 1 {
            return Err(Failure("schedule"));
        }
        Ok(())
    }
}
impl SnapshotValue for Schedule {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        Ok(SnapshotEstimate {
            retained_bytes: 16,
            copy_bytes: 16,
        })
    }
    fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(self.clone())
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct Failure(&'static str);
struct Model {
    value: u64,
    origin: Rc<()>,
    active: Rc<Cell<bool>>,
    reject_restore: Rc<Cell<bool>>,
}
struct Branch {
    value: u64,
    active: Rc<Cell<bool>>,
}
impl Drop for Branch {
    fn drop(&mut self) {
        self.active.set(false);
    }
}
impl SemanticStateTransaction for Model {
    type Branch = Branch;
    type Error = Failure;
    fn branch(&self) -> Result<Branch, Failure> {
        if self.active.replace(true) {
            return Err(Failure("busy"));
        }
        Ok(Branch {
            value: self.value,
            active: self.active.clone(),
        })
    }
    fn commit_branch(&mut self, branch: Branch) -> Result<(), Failure> {
        self.value = branch.value;
        Ok(())
    }
}
impl IsolatedStateCopy for Model {
    type Error = Failure;
    fn estimate_isolated_copy(&self) -> Result<SnapshotEstimate, Failure> {
        Ok(SnapshotEstimate {
            retained_bytes: 128,
            copy_bytes: 128,
        })
    }
    fn copy_isolated(&self, _: &SnapshotReservation) -> Result<Self, Failure> {
        if self.active.get() {
            return Err(Failure("busy"));
        }
        Ok(Self {
            value: self.value,
            origin: self.origin.clone(),
            active: Rc::default(),
            reject_restore: self.reject_restore.clone(),
        })
    }
    fn validate_isolated_restore(&self, saved: &Self) -> Result<(), Failure> {
        if self.active.get() || saved.active.get() {
            return Err(Failure("busy"));
        }
        if !Rc::ptr_eq(&self.origin, &saved.origin) {
            return Err(Failure("foreign"));
        }
        Ok(())
    }
    fn restore_isolated(&mut self, saved: Self) -> Result<(), Failure> {
        self.validate_isolated_restore(&saved)?;
        if self.reject_restore.get() {
            return Err(Failure("restore"));
        }
        self.value = saved.value;
        Ok(())
    }
}
struct Ready;
impl Completion for Ready {
    type Error = Infallible;
    fn is_complete(&self) -> Result<bool, Infallible> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Infallible> {
        Ok(())
    }
}
type Run = RealtimeGenerationState<Model, Schedule, Value, Value, Value, Ready>;
fn run() -> Run {
    Run::new(
        Model {
            value: 11,
            origin: Rc::new(()),
            active: Rc::default(),
            reject_restore: Rc::default(),
        },
        RealtimeGenerationContract::new(Value(5), 1),
        Schedule {
            identity: Value(5),
            position: 0,
        },
        Value(3),
        vec![Value(7)],
        Some(Value(19)),
    )
    .unwrap()
}
fn budget() -> SnapshotBudget {
    SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 2,
        max_branches: 0,
        retained_bytes: 1 << 20,
        cumulative_copy_bytes: 1 << 24,
    })
}
fn advance(run: &mut Run) -> (u64, u64) {
    let mut branch = run.branch().unwrap();
    let decision =
        branch.model_state.value + branch.random_state.as_ref().unwrap().0 + branch.samplers[0].0;
    branch.model_state.value += 7;
    branch.schedule_state.position += 1;
    branch.random_state.as_mut().unwrap().0 += 17;
    branch.samplers[0].0 += decision;
    let position = branch.schedule_state.position;
    branch.attach_submission_completion(Ready).unwrap();
    run.commit_branch(branch).unwrap();
    (decision, position)
}

#[test]
fn realtime_snapshot_replays_schedule_sampler_and_random_state_without_rewinding_revision() {
    let mut run = run();
    let budget = budget();
    advance(&mut run);
    let saved = IsolatedStateSnapshot::capture(&run, &budget).unwrap();
    let first = advance(&mut run);
    let second = advance(&mut run);
    run.replace_sampling(Value(23), vec![Value(29)], Some(Value(31)))
        .unwrap();
    let revision = run.revision();
    let cost = budget.usage().cumulative_copy_bytes;
    saved.restore(&mut run).unwrap();
    assert_eq!(run.revision(), revision + 1);
    assert_eq!(run.sampling(), &Value(3));
    assert!(budget.usage().cumulative_copy_bytes > cost);
    assert_eq!(advance(&mut run), first);
    assert_eq!(advance(&mut run), second);
    let before = budget.usage();
    let pending = run.branch().unwrap();
    assert!(saved.restore(&mut run).is_err());
    assert_eq!(
        budget.usage(),
        before,
        "busy restore must reject before charging"
    );
    Run::discard_branch(pending).unwrap();
    saved.restore(&mut run).unwrap();
    assert_eq!(advance(&mut run), first);
}

#[test]
fn failed_realtime_model_restore_preserves_every_canonical_value_and_copy_usage() {
    let mut run = run();
    let budget = budget();
    let saved = IsolatedStateSnapshot::capture(&run, &budget).unwrap();
    advance(&mut run);
    let before = (
        run.model_state.value,
        run.schedule_state.position,
        run.samplers[0].0,
        run.random_state.as_ref().unwrap().0,
        run.revision,
    );
    let cost = budget.usage().cumulative_copy_bytes;
    run.model_state.reject_restore.set(true);
    assert!(saved.restore(&mut run).is_err());
    assert_eq!(
        (
            run.model_state.value,
            run.schedule_state.position,
            run.samplers[0].0,
            run.random_state.as_ref().unwrap().0,
            run.revision
        ),
        before
    );
    assert!(budget.usage().cumulative_copy_bytes > cost);
    run.model_state.reject_restore.set(false);
    saved.restore(&mut run).unwrap();
    assert_eq!(advance(&mut run), (37, 1));
}
