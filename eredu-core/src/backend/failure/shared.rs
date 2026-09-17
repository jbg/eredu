use super::{BackendFailure, BackendFailureKind, SourceOwner};
use std::{error::Error, fmt};

/// A closed shared diagnostic source. Construction allocates once; retaining
/// an alias or transferring it to BackendFailure allocates no new source.
///
/// No Arc, Weak, mutable source or custom retirement callback is exported. The
/// concrete source allocation retires before its payload, so its own custody
/// can remain last. This is error ownership only, never admission or completion.
pub struct SharedBackendFailure {
    inner: BackendFailure,
}
impl SharedBackendFailure {
    /// Allocates one shared source with its owner's explicit classification.
    /// An original caller must account for this allocation before constructing.
    pub fn new(kind: BackendFailureKind, source: impl Error + Send + Sync + 'static) -> Self {
        Self {
            inner: BackendFailure {
                kind,
                operation: "backend operation",
                source: SourceOwner::shared(source),
            },
        }
    }
    /// Retains the same immutable source without allocating another wrapper.
    pub fn retained(&self) -> Self {
        Self {
            inner: BackendFailure {
                kind: self.inner.kind,
                operation: self.inner.operation,
                source: self.inner.source.retain_shared(),
            },
        }
    }
    /// Transfers the existing owner directly to the neutral public error.
    pub fn into_failure(self) -> BackendFailure {
        self.inner
    }
    /// Borrows the original concrete source, without an extra wrapper layer.
    pub fn source_error(&self) -> &(dyn Error + Send + Sync + 'static) {
        self.inner.source.error()
    }
    /// Requested allocation and named owner/retirement controls for the pinned
    /// Rust1.98 shared representation. Nested payloads and caller storage are
    /// excluded. Overflow returns None; this fact issues no authority.
    pub fn control_bytes<E: Error + Send + Sync + 'static>() -> Option<usize> {
        super::source::shared_retention_peak_bytes::<E>()
    }
}
impl fmt::Debug for SharedBackendFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, f)
    }
}
impl fmt::Display for SharedBackendFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.source_error(), f)
    }
}
impl Error for SharedBackendFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source_error())
    }
}

#[cfg(test)]
mod tests;

// Cloning retains exactly the existing shared allocation. Equality is source
// identity plus portable classification, not formatted diagnostic equality.
impl Clone for SharedBackendFailure { fn clone(&self) -> Self { self.retained() } }
impl PartialEq for SharedBackendFailure {
    fn eq(&self, other: &Self) -> bool {
        self.inner.kind == other.inner.kind && self.inner.operation == other.inner.operation
            && self.inner.source.same_shared(&other.inner.source)
    }
}
impl Eq for SharedBackendFailure {}
