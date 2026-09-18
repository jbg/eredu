//! Closed ordinary diagnostic ownership; no quota or native completion grant.
use super::CaptureError;
use crate::HostPreparationAuthority;
use std::{fmt, sync::Arc};

struct Inner {
    cause: CaptureError,
    // The concrete cause is always unretained; no predecessor error chain.
    _host: HostPreparationAuthority,
    #[cfg(test)]
    retired_control: Option<Arc<std::sync::atomic::AtomicBool>>,
}

/// Immutable aliases of one ordinary diagnostic and its existing host custody.
/// No raw owner, Weak, mutable borrow or owning cause extraction is available.
#[derive(Clone)]
pub struct RetainedCaptureError(Option<Arc<Inner>>);
impl RetainedCaptureError {
    fn cause(&self) -> &CaptureError {
        &self.0.as_ref().expect("live capture failure").cause
    }
}
impl Drop for RetainedCaptureError {
    fn drop(&mut self) {
        if let Some(inner) = self.0.take() {
            // All strong exits consume their Arc, including concurrent aliases.
            // Control deallocation precedes String/cause and then host retirement.
            if let Some(inner) = Arc::into_inner(inner) {
                #[cfg(test)]
                if let Some(retired) = &inner.retired_control {
                    retired.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                drop(inner);
            }
        }
    }
}
impl fmt::Debug for RetainedCaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.cause(), f)
    }
}
impl fmt::Display for RetainedCaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.cause(), f)
    }
}
impl std::error::Error for RetainedCaptureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause())
    }
}
impl CaptureError {
    /// Retains an ordinary diagnostic under custody acquired before construction.
    /// This grants neither finite bytes nor native completion. An already retained
    /// value keeps its complete existing owner; it never forms a predecessor chain.
    #[doc(hidden)]
    pub fn retain_ordinary(self, host: HostPreparationAuthority) -> Self {
        match self {
            Self::Retained(_) => self,
            cause => Self::Retained(RetainedCaptureError(Some(Arc::new(Inner {
                cause,
                _host: host,
                #[cfg(test)]
                retired_control: None,
            })))),
        }
    }
    /// Borrows the original typed diagnostic without transferring its custody.
    /// Explicitly cloning this raw value creates a separate caller-owned copy.
    pub fn cause(&self) -> &Self {
        match self {
            Self::Retained(error) => error.cause(),
            other => other,
        }
    }
    /// Concrete retained Arc allocation, excluding dynamic diagnostic contents,
    /// allocator overhead and the existing host authority's own representation.
    #[doc(hidden)]
    pub fn ordinary_retained_control_bytes() -> Option<u64> {
        let block = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<Inner>())
            .ok()?
            .0
            .pad_to_align();
        u64::try_from(block.size()).ok()
    }
}
impl PartialEq for CaptureError {
    fn eq(&self, other: &Self) -> bool {
        use CaptureError::*;
        match (self.cause(), other.cause()) {
            (Intervention(a), Intervention(b)) => a == b,
            (AdmissionStorage(a), AdmissionStorage(b)) => a == b,
            (Invalid(a), Invalid(b))
            | (Unsupported(a), Unsupported(b))
            | (MissingPath(a), MissingPath(b)) => a == b,
            (Overflow, Overflow) => true,
            (
                Limit {
                    budget: a,
                    cumulative: x,
                },
                Limit {
                    budget: b,
                    cumulative: y,
                },
            ) => a == b && x == y,
            _ => false,
        }
    }
}
impl Eq for CaptureError {}

#[cfg(test)]
mod tests;
