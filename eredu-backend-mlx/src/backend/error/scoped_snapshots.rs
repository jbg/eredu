//! Closed original-source catalogue mechanics. Callers own semantic operation
//! keys, payload types, source classification and all admitted storage.
use eredu_core::SharedBackendFailure;
use safemlx::error::{Exception, ScopedEvaluationCause as Code};

pub(crate) const CAUSE_SNAPSHOTS: usize = 10 * 2;

/// The native constructor captures source presence once. An absent snapshot
/// remains absent after the retained carrier publishes its one immutable value.
pub(crate) fn native_key(error: &Exception) -> Option<usize> {
    let code = match error.scoped_evaluation_cause()? {
        Code::Invalid => 0,
        Code::Capacity => 1,
        Code::Allocation => 2,
        Code::Domain => 3,
        Code::Spent => 4,
        Code::Pending => 5,
        Code::Failed => 6,
        Code::NeedsFundedProgress => 7,
        Code::Unobservable => 8,
        Code::RuntimeBusy => 9,
        Code::InvalidStatus(_) => return None,
    };
    let published = std::error::Error::source(error)
        .and_then(|source| source.source())
        .is_some();
    Some(code * 2 + usize::from(published))
}

pub(crate) struct SnapshotCache<const N: usize> {
    slots: [Option<SharedBackendFailure>; N],
    terminal: Option<SharedBackendFailure>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum SnapshotConflict {
    Unclassified,
    Changed(usize),
}

pub(crate) enum SnapshotLookup {
    Retained(SharedBackendFailure),
    Vacant(usize),
    Conflict(SnapshotConflict),
}

impl<const N: usize> SnapshotCache<N> {
    pub(crate) fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
            terminal: None,
        }
    }

    pub(crate) fn terminal(&self) -> Option<SharedBackendFailure> {
        self.terminal.as_ref().map(SharedBackendFailure::retained)
    }

    /// Borrows the incoming payload. A caller releases its cache loan before
    /// dropping a duplicate/contradictory input; this helper cannot retire it.
    pub(crate) fn lookup<E: std::error::Error + 'static>(
        &self,
        key: Option<usize>,
        incoming: &E,
        same: fn(&E, &E) -> bool,
    ) -> SnapshotLookup {
        if let Some(source) = self.terminal() {
            return SnapshotLookup::Retained(source);
        }
        let Some(index) = key.filter(|index| *index < N) else {
            return SnapshotLookup::Conflict(SnapshotConflict::Unclassified);
        };
        match &self.slots[index] {
            None => SnapshotLookup::Vacant(index),
            Some(previous) => {
                let old = previous
                    .source_error()
                    .downcast_ref::<E>()
                    .expect("closed original snapshot payload");
                if same(old, incoming) {
                    SnapshotLookup::Retained(previous.retained())
                } else {
                    SnapshotLookup::Conflict(SnapshotConflict::Changed(index))
                }
            }
        }
    }

    pub(crate) fn publish(
        &mut self,
        index: usize,
        source: SharedBackendFailure,
    ) -> SharedBackendFailure {
        let slot = &mut self.slots[index];
        assert!(slot.is_none(), "original snapshot never replaced");
        let retained = source.retained();
        *slot = Some(source);
        retained
    }

    pub(crate) fn publish_terminal(
        &mut self,
        source: SharedBackendFailure,
    ) -> SharedBackendFailure {
        assert!(self.terminal.is_none(), "first contract cause is immutable");
        let retained = source.retained();
        self.terminal = Some(source);
        retained
    }
}

#[cfg(test)]
mod tests;
