//! Private exclusive-only request synchronization. See the pinned std audit.
//!
//! No reader, blocking RwLock acquisition, park, yield, raw inner lock or guard
//! downgrade is used by this module or its two consumers. Contention retries the
//! same allocation-free try-write; it never turns required fencing into Busy.
use super::WorkingMemoryError;
use std::{
    fmt,
    mem::size_of,
    sync::{LockResult, PoisonError, RwLock, RwLockWriteGuard, TryLockError, TryLockResult},
};

pub(super) struct ControlMutex<T> {
    inner: RwLock<T>,
    // Observation of the actual WouldBlock branch, with no callback or heap.
    // Each test owns its lock; this adds no production field or global state.
    #[cfg(test)]
    contended: std::sync::atomic::AtomicBool,
}
impl<T> ControlMutex<T> {
    pub(super) const fn new(value: T) -> Self {
        Self {
            inner: RwLock::new(value),
            #[cfg(test)]
            contended: std::sync::atomic::AtomicBool::new(false),
        }
    }
    pub(super) fn lock(&self) -> LockResult<RwLockWriteGuard<'_, T>> {
        loop {
            match self.inner.try_write() {
                Ok(guard) => return Ok(guard),
                // Preserve the actual std poison error and its acquired guard;
                // no new error, formatting, callback or allocation is involved.
                Err(TryLockError::Poisoned(error)) => return Err(error),
                Err(TryLockError::WouldBlock) => {
                    #[cfg(test)]
                    self.contended
                        .store(true, std::sync::atomic::Ordering::Release);
                    core::hint::spin_loop();
                }
            }
        }
    }
    #[cfg(test)]
    pub(super) fn observed_contention(&self) -> bool {
        self.contended.load(std::sync::atomic::Ordering::Acquire)
    }
}
// Debug must not call RwLock::fmt (a reader) or format a T under the lock.
impl<T> fmt::Debug for ControlMutex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ControlMutex").finish_non_exhaustive()
    }
}

// The audited Rust 1.98 selector uses inline queue/futex/no_threads storage on
// every other selected target. SOLID lazily constructs a kernel object even for
// try_write; it has no priced constructor here. Admission rejects that missing
// attribution under finite and unlimited limits.
pub(super) fn require_known_layout() -> Result<(), WorkingMemoryError> {
    require_layout(!cfg!(target_os = "solid_asp3"))
}
fn require_layout(known: bool) -> Result<(), WorkingMemoryError> {
    if known {
        Ok(())
    } else {
        Err(WorkingMemoryError::UnknownBound)
    }
}

// Concrete simultaneously representable acquisition/return controls; the lock
// body itself is already included in each enclosing Arc/enum layout. No dynamic
// native mutex, waiter, parking queue, Thread handle or formatter is constructed.
pub(super) const fn operation_control_bytes<T: 'static>() -> usize {
    size_of::<RwLockWriteGuard<'static, T>>()
        + size_of::<TryLockResult<RwLockWriteGuard<'static, T>>>()
        + size_of::<TryLockError<RwLockWriteGuard<'static, T>>>()
        + size_of::<PoisonError<RwLockWriteGuard<'static, T>>>()
        + size_of::<LockResult<RwLockWriteGuard<'static, T>>>()
}

#[cfg(test)]
pub(super) mod tests;
