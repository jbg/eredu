//! One prepared owner for a failed operation and failed checkpoint restoration.
use super::{SpeculativeDriverError, SpeculativeExecutor};
use crate::{BackendFailure, HostPreparationAuthority};
use std::{error::Error, marker::PhantomData, mem::{size_of, size_of_val}};

/// Both failures from one unsuccessful operation and its attempted restoration.
/// The original operation remains the primary source. The second failure is
/// retained independently; this value certifies neither restoration nor reuse.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[error("speculative operation failed: {operation}; checkpoint restoration also failed: {rollback}")]
pub struct SpeculativeRollbackFailure<C: Error + 'static, E: Error + 'static> {
    /// Original operation failure and its actual resource ownership.
    #[source]
    pub operation: C,
    /// Failure to restore the checkpoint, including any owned native prefix.
    pub rollback: E,
}

#[derive(Debug, thiserror::Error)]
#[error("{failure}")]
struct RetainedRollback<C: Error + 'static, E: Error + 'static> {
    #[source]
    failure: SpeculativeRollbackFailure<C, E>,
    // Both concrete causes and the source shell retire before their host grant.
    _host: HostPreparationAuthority,
}

pub(super) struct PreparedRollback<C, E> {
    host: HostPreparationAuthority,
    marker: PhantomData<fn(C, E)>,
}
impl<C: Error + Send + Sync + 'static, E: Error + Send + Sync + 'static> PreparedRollback<C, E> {
    fn control_bytes() -> Option<usize> {
        let parts = [size_of::<Self>(), size_of::<Option<Self>>(),
            size_of::<Result<Self, E>>(), size_of::<Result<Self, SpeculativeDriverError<E>>>(),
            size_of::<SpeculativeRollbackFailure<C, E>>(),
            size_of::<RetainedRollback<C, E>>(), size_of::<(C, E)>(),
            size_of::<BackendFailure>(), size_of::<SpeculativeDriverError<E>>(),
            size_of::<Result<BackendFailure, E>>(), size_of::<Option<usize>>()];
        BackendFailure::source_retention_peak_bytes::<RetainedRollback<C, E>>()?
            .checked_add(size_of_val(&parts))
            .and_then(|bytes| parts.into_iter().try_fold(bytes, usize::checked_add))
    }
    pub(super) fn prepare<X: SpeculativeExecutor<Error = E>>(
        executor: &X, context: X::Context<'_>,
    ) -> Result<Self, E> {
        let bytes = Self::control_bytes().and_then(|bytes|
            bytes.checked_add(size_of::<(&X, X::Context<'_>)>())
                .and_then(|bytes| bytes.checked_add(size_of::<(
                    Self, C, &mut X, &mut X::Cache, &X::CacheCheckpoint, X::Context<'_>,
                )>()))
                .and_then(|bytes| bytes.checked_add(size_of::<Result<(), E>>())));
        let host = executor.driver_host_metadata(bytes, context)?;
        Ok(Self { host, marker: PhantomData })
    }
    #[inline(never)]
    pub(super) fn retain(self, operation: C, rollback: E) -> BackendFailure {
        BackendFailure::from_error(RetainedRollback {
            failure: SpeculativeRollbackFailure { operation, rollback },
            _host: self.host,
        })
    }
}

impl<E: Error + Send + Sync + 'static> PreparedRollback<SpeculativeDriverError<E>, E> {
    // Error-only restoration and its two concrete causes must not occupy the
    // successful prefill/commit frame while it calls a cold model constructor.
    #[inline(never)]
    pub(super) fn restore<X: SpeculativeExecutor<Error = E>>(
        self,
        operation: SpeculativeDriverError<E>,
        executor: &mut X,
        cache: &mut X::Cache,
        checkpoint: &X::CacheCheckpoint,
        context: X::Context<'_>,
    ) -> SpeculativeDriverError<E> {
        match executor.restore_checkpoint(cache, checkpoint, context) {
            Ok(()) => operation,
            Err(rollback) => SpeculativeDriverError::Rollback(self.retain(operation, rollback)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
    #[derive(Debug)]
    struct State { host_retired: AtomicBool, causes_retired: AtomicUsize }
    #[derive(Debug)]
    struct Host(Arc<State>);
    impl Drop for Host {
        fn drop(&mut self) {
            assert_eq!(self.0.causes_retired.load(Ordering::SeqCst), 2);
            self.0.host_retired.store(true, Ordering::SeqCst);
        }
    }
    #[derive(Debug, thiserror::Error)]
    #[error("{name}")]
    struct Cause { name: &'static str, state: Arc<State> }
    impl Drop for Cause {
        fn drop(&mut self) {
            assert!(!self.state.host_retired.load(Ordering::SeqCst));
            self.state.causes_retired.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn both_rollback_causes_retire_before_the_prepared_host_owner() {
        let state = Arc::new(State { host_retired: AtomicBool::new(false), causes_retired: AtomicUsize::new(0) });
        assert!(PreparedRollback::<Cause, Cause>::control_bytes().unwrap() > size_of::<RetainedRollback<Cause, Cause>>());
        let prepared = PreparedRollback::<Cause, Cause> {
            host: HostPreparationAuthority::retain(Host(state.clone())), marker: PhantomData,
        };
        let error = prepared.retain(Cause { name: "model budget", state: state.clone() },
            Cause { name: "copy budget", state: state.clone() });
        let retained = error.source().unwrap().downcast_ref::<RetainedRollback<Cause, Cause>>().unwrap();
        assert_eq!(retained.failure.operation.name, "model budget");
        assert_eq!(retained.failure.rollback.name, "copy budget");
        assert_eq!(retained.failure.source().unwrap().downcast_ref::<Cause>().unwrap().name, "model budget");
        assert!(!state.host_retired.load(Ordering::SeqCst));
        drop(error);
        assert!(state.host_retired.load(Ordering::SeqCst));
    }
}
