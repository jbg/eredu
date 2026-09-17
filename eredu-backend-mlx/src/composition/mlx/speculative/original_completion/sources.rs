//! Three real repeated observation sites, ten fixed causes, two immutable
//! publication states. Native array validation adds no array-specific source.
use super::*;
use crate::backend::{
    error::scoped_snapshots::{
        native_key, SnapshotCache, SnapshotConflict, SnapshotLookup, CAUSE_SNAPSHOTS,
    },
    nn::shared::{OriginalObservationFailure, OriginalObservationSite},
};
use std::cell::RefCell;

const SLOTS: usize = 3 * CAUSE_SNAPSHOTS;

#[derive(Debug)]
struct NativeFailure {
    cause: Exception,
    _controls: OriginalTextControlGuard,
}
impl std::fmt::Display for NativeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for NativeFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
#[derive(Debug)]
struct ContractFailure {
    reason: SnapshotConflict,
    offending: NativeFailure,
}
impl std::fmt::Display for ContractFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "original speculative snapshot contract {:?}: {}",
            self.reason, self.offending
        )
    }
}
impl std::error::Error for ContractFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.offending)
    }
}

pub(super) struct Sources(RefCell<SnapshotCache<SLOTS>>);
impl Sources {
    pub(super) fn new() -> Self {
        Self(RefCell::new(SnapshotCache::new()))
    }
    pub(super) fn terminal(&self) -> Option<Error> {
        self.0
            .borrow()
            .terminal()
            .map(|source| Error::retained_original(source, false))
    }
    pub(super) fn failure(
        &self,
        failure: OriginalObservationFailure,
        controls: OriginalTextControlGuard,
    ) -> Error {
        let base = match failure.site {
            OriginalObservationSite::Observer => 0,
            OriginalObservationSite::Event => 1,
            OriginalObservationSite::Validation => 2,
        } * CAUSE_SNAPSHOTS;
        let key = native_key(&failure.cause).map(|key| base + key);
        let incoming = NativeFailure {
            cause: failure.cause,
            _controls: controls,
        };
        let mut cache = self.0.borrow_mut();
        let retained = match cache.lookup(key, &incoming, |a, b| a.cause == b.cause) {
            SnapshotLookup::Retained(retained) => {
                drop(cache);
                drop(incoming);
                return Error::retained_original(retained, false);
            }
            SnapshotLookup::Vacant(index) => cache.publish(
                index,
                SharedBackendFailure::new(BackendFailureKind::Other, incoming),
            ),
            SnapshotLookup::Conflict(reason) => cache.publish_terminal(SharedBackendFailure::new(
                BackendFailureKind::InvalidSession,
                ContractFailure {
                    reason,
                    offending: incoming,
                },
            )),
        };
        drop(cache);
        Error::retained_original(retained, false)
    }
    pub(super) fn control_bytes() -> Option<u64> {
        let fixed = [
            size_of::<Self>(),
            size_of::<SnapshotCache<SLOTS>>(),
            size_of::<std::cell::RefMut<'static, SnapshotCache<SLOTS>>>(),
            size_of::<std::cell::Ref<'static, SnapshotCache<SLOTS>>>(),
            size_of::<SnapshotLookup>(),
            size_of::<SnapshotConflict>(),
            size_of::<NativeFailure>(),
            size_of::<fn(&NativeFailure, &NativeFailure) -> bool>(),
            size_of::<ContractFailure>(),
            size_of::<OriginalObservationFailure>(),
            size_of::<OriginalObservationSite>(),
            size_of::<Result<bool, OriginalObservationFailure>>(),
            size_of::<Option<usize>>(),
            size_of::<Option<Error>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        u64::try_from(
            SharedBackendFailure::control_bytes::<NativeFailure>()?
                .checked_mul(SLOTS)?
                .checked_add(SharedBackendFailure::control_bytes::<ContractFailure>()?)?
                .checked_add(fixed)?,
        )
        .ok()
    }
}
