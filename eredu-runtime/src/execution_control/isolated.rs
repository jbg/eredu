//! Completed isolated state copies with non-rewindable logical reservations.

use super::{SnapshotBudget, SnapshotReservation};
use eredu_core::{
    execution_control::{ExecutionControlError, SnapshotEstimate, SnapshotResourceKind},
    BackendFailure,
};

mod observation;

/// A self-contained value copied alongside an isolated model-state snapshot.
///
/// Unlike `Clone`, this contract requires independent mutable contents and a
/// complete conservative storage estimate. Immutable resources may be shared only
/// when their lifetime and storage remain accounted for by the enclosing owner.
/// Implementations perform no random draws or model transitions.
pub trait SnapshotValue: Sized {
    /// Complete retained and copy cost, including dynamic storage and native data.
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure>;

    /// Makes an exact independent copy and establishes any native copy completion.
    /// Native recovery must retain a clone of `retention` when failed submitted
    /// work can outlive this call. Success leaves no pending copy operation.
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure>;
}

/// Adds complete component estimates to a container's inline metadata, using
/// checked arithmetic. Double-counted inline fields are conservative; callers
/// must still include every independently copied dynamic allocation.
pub fn snapshot_value_estimate<T>(
    components: impl IntoIterator<Item = Result<SnapshotEstimate, BackendFailure>>,
) -> Result<SnapshotEstimate, BackendFailure> {
    let mut total = SnapshotEstimate {
        retained_bytes: std::mem::size_of::<T>() as u64,
        copy_bytes: std::mem::size_of::<T>() as u64,
    };
    for component in components {
        let component = component?;
        total.retained_bytes = total
            .retained_bytes
            .checked_add(component.retained_bytes)
            .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
        total.copy_bytes = total
            .copy_bytes
            .checked_add(component.copy_bytes)
            .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
    }
    Ok(total)
}

impl<T: SnapshotValue> SnapshotValue for Option<T> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>(self.iter().map(SnapshotValue::estimate_snapshot_value))
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        self.as_ref()
            .map(|value| value.copy_snapshot_value(retention))
            .transpose()
    }
}
impl<T: SnapshotValue> SnapshotValue for Vec<T> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        // Estimates also admit live fork successors, whose allocations have not
        // been compacted by copy_snapshot_value. Price spare source slots too.
        let slots = self
            .capacity()
            .checked_mul(std::mem::size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
        snapshot_value_estimate::<Self>(
            std::iter::once(Ok(SnapshotEstimate {
                retained_bytes: slots,
                copy_bytes: 0,
            }))
            .chain(self.iter().map(SnapshotValue::estimate_snapshot_value)),
        )
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        let mut copied = Vec::new();
        copied
            .try_reserve_exact(self.len())
            .map_err(BackendFailure::from_error)?;
        for value in self {
            copied.push(value.copy_snapshot_value(retention)?);
        }
        Ok(copied)
    }
}
macro_rules! scalar_snapshot {
    ($($scalar:ty),*) => { $(
        impl SnapshotValue for $scalar {
            fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
                snapshot_value_estimate::<Self>([])
            }
            fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> { Ok(*self) }
        }
    )* };
}
scalar_snapshot!(i32, u32, usize, f32, ());

/// General state-copy mechanisms, independent of text or realtime family policy.
///
/// Implementations preserve origin identity and cumulative resource accounting
/// across every copy. These methods operate at a quiescent state boundary; they
/// reject outstanding transitions rather than inferring native completion.
pub trait IsolatedStateCopy: Sized {
    /// Native failures may remain concrete inside the mechanism contract.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Conservative complete retained and temporary-copy cost, without allocation.
    fn estimate_isolated_copy(&self) -> Result<SnapshotEstimate, Self::Error>;

    /// Copies every mutable component into independently owned state and establishes
    /// exact completion before returning it. No transition admission remains held
    /// on the source after success. On failure the source is unchanged.
    ///
    /// Clone `retention` into any native recovery owner that can outlive this call.
    /// Pending failed copies must keep both their native resources and logical
    /// reservation until completion or safe teardown. The caller retains the
    /// original reservation for the successfully returned state.
    fn copy_isolated(&self, retention: &SnapshotReservation) -> Result<Self, Self::Error>;

    /// Checks origin, geometry, idle admission and revision capacity before copying
    /// a saved state. No model, random, observation or budget state changes.
    fn validate_isolated_restore(&self, saved: &Self) -> Result<(), Self::Error>;

    /// Revalidates and atomically installs an already completed independent copy.
    /// Rejection leaves the target unchanged. Installation submits no native work
    /// and preserves the target's non-rewindable counters and admission identity.
    fn restore_isolated(&mut self, saved: Self) -> Result<(), Self::Error>;
}

/// Independently runnable descendants of a quiescent, isolated state owner.
///
/// The supplied branch reservation remains retained by the returned owner and
/// every execution descendant or native recovery owner that can outlive it.
/// Releasing a facade handle or a failed completion does not release this
/// accounting while the descendant's native resources remain live. Copies keep
/// originating native cumulative budgets while owning independent admission.
pub trait IsolatedStateFork: IsolatedStateCopy {
    /// Copies a runnable state with the same complete cost as an isolated copy.
    /// Failed copies preserve source state and retain the reservation whenever
    /// submitted native copying remains unresolved.
    fn fork_isolated(&self, retention: &SnapshotReservation) -> Result<Self, Self::Error>;
}

/// Complete retained-state sizing at a completed fork execution boundary.
///
/// The enclosing transaction admits any increase against its existing fork
/// reservation before committing this candidate. Estimation performs no copying,
/// native submission, random draw, or canonical-state mutation.
pub trait ForkStateEstimate: eredu_core::scheduler::SemanticStateTransaction {
    /// Conservative logical retained bytes of every successor component, using
    /// the same storage accounting as isolated snapshot/fork admission.
    fn estimate_fork_branch(branch: &Self::Branch) -> Result<u64, BackendFailure>;
}

/// Portable copy admission or mechanism error, preserving the concrete source.
#[derive(Debug, thiserror::Error)]
pub enum IsolatedStateError {
    /// A known state copy exceeds the shared logical reservation limits.
    #[error(transparent)]
    Admission(#[from] ExecutionControlError),
    /// A native copy, compatibility check or atomic installation failed.
    #[error("isolated state operation failed: {0}")]
    Backend(#[source] BackendFailure),
}

fn backend(error: impl std::error::Error + Send + Sync + 'static) -> IsolatedStateError {
    IsolatedStateError::Backend(BackendFailure::from_error(error))
}

/// Reusable saved component state with its own retained-resource reservation.
///
/// This is a mechanism-level snapshot, not a complete inference-session snapshot.
/// The enclosing driver accounts for host queues, sampling, observations and
/// external delivery before composing it into a session restore operation.
/// Snapshots keep the original budget; restoring cannot select a fresh budget or
/// refund cumulative copying. A saved copy holds no live transition admission.
pub struct IsolatedStateSnapshot<S> {
    state: S,
    budget: SnapshotBudget,
    _reservation: SnapshotReservation,
}

impl<S: IsolatedStateCopy> IsolatedStateSnapshot<S> {
    fn estimate(state: &S) -> Result<SnapshotEstimate, IsolatedStateError> {
        let mut estimate = state.estimate_isolated_copy().map_err(backend)?;
        estimate.retained_bytes = estimate
            .retained_bytes
            .checked_add(std::mem::size_of::<Self>() as u64)
            .ok_or(ExecutionControlError::Overflow)?;
        Ok(estimate)
    }

    /// Reserves complete known storage before copying at an idle boundary.
    /// Failed admitted copies consume cumulative allowance without publication.
    pub fn capture(state: &S, budget: &SnapshotBudget) -> Result<Self, IsolatedStateError> {
        let reservation =
            budget.reserve(SnapshotResourceKind::Snapshot, Some(Self::estimate(state)?))?;
        let state = state.copy_isolated(&reservation).map_err(backend)?;
        Ok(Self {
            state,
            budget: budget.clone(),
            _reservation: reservation,
        })
    }

    /// Restores a compatible idle owner through an independent provisional copy.
    /// The saved state remains reusable. Checks precede reservation; copying and
    /// all fallible mechanism work precede atomic target replacement.
    pub fn restore(&self, state: &mut S) -> Result<(), IsolatedStateError> {
        state
            .validate_isolated_restore(&self.state)
            .map_err(backend)?;
        let reservation = self.budget.reserve(
            SnapshotResourceKind::Restore,
            Some(Self::estimate(&self.state)?),
        )?;
        let candidate = self.state.copy_isolated(&reservation).map_err(backend)?;
        state.restore_isolated(candidate).map_err(backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::execution_control::SnapshotLimits;
    use std::{
        cell::{Cell, RefCell},
        error::Error,
        rc::Rc,
    };

    struct State {
        value: u64,
        origin: Rc<()>,
        fail: Rc<Cell<bool>>,
        recovery: Rc<RefCell<Option<SnapshotReservation>>>,
    }
    impl IsolatedStateCopy for State {
        type Error = std::io::Error;
        fn estimate_isolated_copy(&self) -> Result<SnapshotEstimate, Self::Error> {
            Ok(SnapshotEstimate {
                retained_bytes: 64,
                copy_bytes: 80,
            })
        }
        fn copy_isolated(&self, retention: &SnapshotReservation) -> Result<Self, Self::Error> {
            if self.fail.get() {
                *self.recovery.borrow_mut() = Some(retention.clone());
                return Err(std::io::Error::other("copy awaiting safe retirement"));
            }
            Ok(Self {
                value: self.value,
                origin: self.origin.clone(),
                fail: self.fail.clone(),
                recovery: self.recovery.clone(),
            })
        }
        fn validate_isolated_restore(&self, saved: &Self) -> Result<(), Self::Error> {
            if Rc::ptr_eq(&self.origin, &saved.origin) {
                Ok(())
            } else {
                Err(std::io::Error::other("foreign origin"))
            }
        }
        fn restore_isolated(&mut self, saved: Self) -> Result<(), Self::Error> {
            self.validate_isolated_restore(&saved)?;
            self.value = saved.value;
            Ok(())
        }
    }
    #[test]
    fn failed_restore_retains_recovery_credit_and_never_refunds_copying() {
        let budget = SnapshotBudget::new(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 0,
            retained_bytes: 4096,
            cumulative_copy_bytes: 240,
        });
        let mut state = State {
            value: 17,
            origin: Rc::new(()),
            fail: Rc::new(Cell::new(false)),
            recovery: Rc::default(),
        };
        let snapshot = IsolatedStateSnapshot::capture(&state, &budget).unwrap();
        let retained = budget.usage().retained_bytes;
        state.value = 29;
        state.fail.set(true);
        let error = snapshot.restore(&mut state).unwrap_err();
        assert!(error
            .source()
            .unwrap()
            .source()
            .unwrap()
            .is::<std::io::Error>());
        assert_eq!(state.value, 29);
        assert_eq!(budget.usage().retained_bytes, retained * 2);
        assert_eq!(budget.usage().cumulative_copy_bytes, 160);
        state.recovery.borrow_mut().take();
        assert_eq!(budget.usage().retained_bytes, retained);
        state.fail.set(false);
        snapshot.restore(&mut state).unwrap();
        assert_eq!(state.value, 17);
        state.value = 41;
        assert!(matches!(
            snapshot.restore(&mut state),
            Err(IsolatedStateError::Admission(_))
        ));
        assert_eq!(state.value, 41);
        drop(snapshot);
        assert_eq!(budget.usage().retained_bytes, 0);
        assert_eq!(budget.usage().snapshots, 0);
        assert_eq!(budget.usage().cumulative_copy_bytes, 240);
    }
}
