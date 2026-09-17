//! Closed poll snapshot population: two actual polling operation classes, ten defined
//! scoped causes and two immutable carrier-publication snapshots. No poll count.
use super::{Error, OriginalTextControlGuard, SamplingEventCause, SamplingEventFailure};
use crate::backend::error::scoped_snapshots::{
    native_key, SnapshotCache, SnapshotConflict, SnapshotLookup, CAUSE_SNAPSHOTS,
};
use eredu_core::{BackendFailureKind, SharedBackendFailure};
#[cfg(test)]
use safemlx::error::ScopedEvaluationCause as Code;
use std::{cell::RefCell, mem::size_of, rc::Rc};

#[derive(Clone, Copy, Debug)]
pub(super) enum Origin {
    Observer,
    Event,
}
const ORIGINS: usize = 2;
const SLOTS: usize = ORIGINS * CAUSE_SNAPSHOTS;

#[derive(Debug, thiserror::Error)]
enum ContractReason {
    #[error("original sampling returned an unclassified scoped failure")]
    Unclassified,
    #[error("original sampling changed immutable failure snapshot {0}")]
    Changed(usize),
}
#[derive(Debug)]
struct ContractFailure {
    reason: ContractReason,
    // Retain the first actual offending native cause/source, not just a label.
    offending: SamplingEventCause,
    _controls: OriginalTextControlGuard,
}
impl std::fmt::Display for ContractFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.reason, self.offending)
    }
}
impl std::error::Error for ContractFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.offending)
    }
}
type Cache = SnapshotCache<SLOTS>;
/// Completion and escaped token source share this exact unusable-state flag
/// and catalogue. No Rc/Weak or mutable source leaves the closed owner.
pub(super) struct PollOwner(Option<Rc<RefCell<Cache>>>);
impl PollOwner {
    pub(super) fn new() -> Self {
        Self(Some(Rc::new(RefCell::new(Cache::new()))))
    }
    pub(super) fn terminal(&self) -> Option<Error> {
        self.0
            .as_ref()
            .expect("live poll owner")
            .borrow()
            .terminal()
            .map(|source| Error::retained_original(source, false))
    }
    pub(super) fn failure(
        &self,
        origin: Origin,
        cause: SamplingEventCause,
        controls: OriginalTextControlGuard,
    ) -> Error {
        let mut cache = self.0.as_ref().expect("live poll owner").borrow_mut();
        let key = key(origin, &cause);
        let incoming = SamplingEventFailure {
            cause,
            _controls: controls,
        };
        let retained = match cache.lookup(key, &incoming, |a, b| same(&a.cause, &b.cause)) {
            SnapshotLookup::Retained(retained) => {
                drop(cache);
                drop(incoming);
                return Error::retained_original(retained, false);
            }
            SnapshotLookup::Vacant(index) => cache.publish(
                index,
                SharedBackendFailure::new(BackendFailureKind::Other, incoming),
            ),
            SnapshotLookup::Conflict(reason) => {
                let SamplingEventFailure { cause, _controls } = incoming;
                cache.publish_terminal(SharedBackendFailure::new(
                    BackendFailureKind::InvalidSession,
                    ContractFailure {
                        reason: match reason {
                            SnapshotConflict::Unclassified => ContractReason::Unclassified,
                            SnapshotConflict::Changed(index) => ContractReason::Changed(index),
                        },
                        offending: cause,
                        _controls,
                    },
                ))
            }
        };
        drop(cache);
        Error::retained_original(retained, false)
    }
    pub(super) fn control_bytes() -> Option<u64> {
        use std::{alloc::Layout, cell::Cell, mem::ManuallyDrop, rc::Weak};
        let allocation = Layout::new::<[Cell<usize>; 2]>()
            .align_to(2)
            .ok()?
            .pad_to_align()
            .extend(Layout::new::<RefCell<Cache>>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let fixed = [
            size_of::<Cache>(),
            size_of::<RefCell<Cache>>(),
            size_of::<Self>(),
            size_of::<Option<Rc<RefCell<Cache>>>>(),
            size_of::<Rc<RefCell<Cache>>>(),
            size_of::<ManuallyDrop<Rc<RefCell<Cache>>>>(),
            size_of::<Weak<RefCell<Cache>>>(),
            size_of::<Option<RefCell<Cache>>>(),
            size_of::<std::cell::RefMut<'static, Cache>>(),
            size_of::<std::cell::Ref<'static, Cache>>(),
            size_of::<Option<usize>>(),
            size_of::<ContractReason>(),
            size_of::<Option<ContractReason>>(),
            size_of::<Origin>(),
            size_of::<SnapshotConflict>(),
            size_of::<SnapshotLookup>(),
            size_of::<SamplingEventFailure>(),
            size_of::<fn(&SamplingEventFailure, &SamplingEventFailure) -> bool>(),
            size_of::<Option<SharedBackendFailure>>(),
            size_of::<Error>(),
        ]
        .into_iter()
        .try_fold(allocation, usize::checked_add)?;
        let all = SharedBackendFailure::control_bytes::<SamplingEventFailure>()?
            .checked_mul(SLOTS)?
            .checked_add(SharedBackendFailure::control_bytes::<ContractFailure>()?)?
            .checked_add(fixed)?;
        u64::try_from(all).ok()
    }
}
impl Clone for PollOwner {
    fn clone(&self) -> Self {
        Self(Some(self.0.as_ref().expect("live poll owner").clone()))
    }
}
impl Drop for PollOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Rc::into_inner(owner));
        }
    }
}
fn key(origin: Origin, cause: &SamplingEventCause) -> Option<usize> {
    let SamplingEventCause::Native(error) = cause else {
        return None;
    };
    Some(
        match origin {
            Origin::Observer => 0,
            Origin::Event => 1,
        } * CAUSE_SNAPSHOTS
            + native_key(error)?,
    )
}
fn same(left: &SamplingEventCause, right: &SamplingEventCause) -> bool {
    match (left, right) {
        (SamplingEventCause::Native(a), SamplingEventCause::Native(b)) => a == b,
        _ => false,
    }
}

#[cfg(test)]
mod tests;
