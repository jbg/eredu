//! Paid parser allocation callbacks with exact error storage and retirement.
mod storage;
pub use storage::ParserStorageError;
use std::{
    alloc::Layout,
    error::Error,
    fmt,
    mem::{size_of, size_of_val},
    sync::{atomic::AtomicUsize, Arc, OnceLock},
};
trait Account: fmt::Debug + Send + Sync {
    fn reserve(&self, bytes: usize) -> bool;
    fn cause(&self) -> Option<&(dyn Error + 'static)>;
    fn retire(self: Arc<Self>);
}
struct Payload<F, E> {
    failure: OnceLock<E>,
    funding: F,
}
impl<F, E> fmt::Debug for Payload<F, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParserAllocationFunding")
            .field("failed", &self.failure.get().is_some())
            .finish()
    }
}
impl<F, E> Account for Payload<F, E>
where
    F: Fn(usize) -> Result<(), E> + Send + Sync + 'static,
    E: Error + Send + Sync + 'static,
{
    fn reserve(&self, bytes: usize) -> bool {
        if self.failure.get().is_some() {
            return false;
        }
        match (self.funding)(bytes) {
            Ok(()) => true,
            Err(error) => {
                let _ = self.failure.set(error);
                false
            }
        }
    }
    fn cause(&self) -> Option<&(dyn Error + 'static)> {
        self.failure.get().map(|e| e as _)
    }
    fn retire(self: Arc<Self>) {
        if let Some(payload) = Arc::into_inner(self) {
            drop(payload);
        }
    }
}
/// Independently paid callback and first-failure storage. Every compiler/table
/// worker must request its real reached allocation before constructing it.
/// This owner supplies funding and custody, never source or execution authority.
pub struct ParserAllocationFunding(Option<Arc<dyn Account>>);
impl Clone for ParserAllocationFunding {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for ParserAllocationFunding {
    fn drop(&mut self) {
        if let Some(account) = self.0.take() {
            account.retire();
        }
    }
}
impl fmt::Debug for ParserAllocationFunding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParserAllocationFunding")
            .field(
                "failed",
                &self.0.as_ref().is_some_and(|a| a.cause().is_some()),
            )
            .finish()
    }
}
/// Inline construction refusal; no callback destination exists on failure.
#[derive(Debug)]
pub enum ParserAllocationPreparationError<E> {
    /// The actual shared callback/error layout cannot be represented.
    Overflow,
    /// The real account refused before allocation.
    Funding(E),
}
impl<E: fmt::Display> fmt::Display for ParserAllocationPreparationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => f.write_str("parser allocation funding owner layout overflow"),
            Self::Funding(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl<E: Error + 'static> Error for ParserAllocationPreparationError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Funding(e) => Some(e),
            _ => None,
        }
    }
}
/// The original concrete refusal remains borrowed from its fixed first-error
/// slot. The final alias frees its Arc shell before dropping error/source/H.
#[derive(Debug)]
pub struct ParserAllocationFailure {
    funding: ParserAllocationFunding,
}
impl fmt::Display for ParserAllocationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.source() {
            Some(cause) => fmt::Display::fmt(cause, f),
            None => f.write_str("parser allocation funding is unavailable"),
        }
    }
}
impl Error for ParserAllocationFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.funding.0.as_ref().and_then(|a| a.cause())
    }
}
impl ParserAllocationFunding {
    /// Exact prospective constructor payment for this actual retained callback
    /// and its error slot. Inspection invokes no callback or allocation.
    pub fn preparation_bytes<F, E>(_: &F) -> Option<usize>
    where F: Fn(usize) -> Result<(), E> + Send + Sync + 'static,
        E: Error + Send + Sync + 'static,
    {
        let layout = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Payload<F, E>>())
            .ok()?
            .0
            .pad_to_align();
        let parts = [
            layout.size(),
            size_of::<Payload<F, E>>(),
            size_of::<F>(),
            size_of::<OnceLock<E>>(),
            size_of::<Self>(),
            size_of::<Arc<Payload<F, E>>>(),
            size_of::<Arc<dyn Account>>(),
            size_of::<ParserAllocationPreparationError<E>>(),
            size_of::<ParserAllocationFailure>(),
            size_of::<Result<Self, ParserAllocationPreparationError<E>>>(),
            size_of::<Result<(), E>>(),
            size_of::<Layout>(),
            size_of::<Option<usize>>(),
            size_of::<Result<E, E>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Reserves the actual callback/error slot and shared shell before allocation.
    /// The callback must retain its real cumulative funding account.
    pub fn prepare<F, E>(funding: F) -> Result<Self, ParserAllocationPreparationError<E>>
    where
        F: Fn(usize) -> Result<(), E> + Send + Sync + 'static,
        E: Error + Send + Sync + 'static,
    {
        let bytes = Self::preparation_bytes(&funding)
            .ok_or(ParserAllocationPreparationError::Overflow)?;
        funding(bytes).map_err(ParserAllocationPreparationError::Funding)?;
        Ok(Self(Some(Arc::new(Payload {
            failure: OnceLock::new(),
            funding,
        }))))
    }
    /// Explicitly unenforced accounting over the same allocation workers.
    pub const fn unenforced() -> Self {
        Self(None)
    }

    /// Whether this source has an actual admission callback. Opaque external
    /// producers cannot be invoked under that policy without their own hooks.
    pub fn is_enforced(&self) -> bool { self.0.is_some() }

    /// Reserve reached storage before allocation. A refusal retains the original
    /// typed cause in the preallocated first-error slot and never refunds work.
    pub fn reserve(&self, bytes: usize) -> Result<(), ParserAllocationFailure> {
        match &self.0 {
            None => Ok(()),
            Some(account) if account.reserve(bytes) => Ok(()),
            Some(_) => Err(ParserAllocationFailure { funding: self.clone() }),
        }
    }

    /// Borrowed dependency hooks may return a fixed refusal marker. Recover the
    /// original typed failure from this same paid first-error slot afterward.
    pub fn failure(&self) -> Option<ParserAllocationFailure> {
        self.0.as_ref().filter(|account| account.cause().is_some())
            .map(|_| ParserAllocationFailure { funding: self.clone() })
    }
}
impl regex_syntax::allocation::Allocation for ParserAllocationFunding {
    fn reserve(&self, bytes: usize) -> Result<(), regex_syntax::allocation::AllocationError> {
        ParserAllocationFunding::reserve(self, bytes)
            .map_err(|_| regex_syntax::allocation::AllocationError::Refused)
    }
}

/// Distinguish resource refusal from optional semantic heuristics.
pub fn is_storage_failure(error: &(dyn Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(cause) = current {
        if cause.is::<ParserStorageError>() || cause.is::<ParserAllocationFailure>() || cause.is::<crate::raw::PreparedExprError>() || cause.is::<crate::raw::HashConsCapacityError>() || cause.is::<crate::RegexAstCopyFailure>() || cause.is::<crate::RegexBuilderCopyFailure>() || cause.is::<std::collections::TryReserveError>() || cause.is::<hashbrown::TryReserveError>() || cause.is::<indexmap::TryReserveError>() { return true; }
        current = cause.source();
    }
    false
}
