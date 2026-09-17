use super::{ReentrantMutex, RuntimeLockGuard};
use std::{mem::size_of, thread::LocalKey, time::Duration};

/// Concrete representations used by the private native runtime lock.
///
/// Static storage is counted once in its runtime baseline, not once per request.
/// This is not a complete request or thread bound. The native TLS extent needs
/// an actual compiler/target receipt; dependency deadlock-tracking features must
/// also be verified before claiming the selected acquire/release path is closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeLockLayout {
    /// The single pre-existing static reentrant mutex.
    pub mutex_bytes: usize,
    /// LocalKey handle representation for lock_api's constant identity key.
    /// This does not imply a separately allocated static header.
    pub identity_key_header_bytes: usize,
    /// Exact emitted per-thread identity storage, still unknown until verified.
    pub identity_tls_bytes: Option<usize>,
    /// One actual owning Rust guard.
    pub guard_bytes: usize,
    /// The immediate acquisition result representation.
    pub try_guard_bytes: usize,
    /// The finite counter, selected action and duration in the ordinary waiter.
    pub ordinary_wait_bytes: usize,
}

/// Returns layout facts without acquiring the lock, initializing TLS or doing
/// native work. OS sleep internals and ordinary reclaim/hooks are not included;
/// they remain separate ordinary call populations. No authority is constructed.
pub fn runtime_lock_layout() -> RuntimeLockLayout {
    RuntimeLockLayout {
        mutex_bytes: size_of::<ReentrantMutex<()>>(),
        identity_key_header_bytes: size_of::<LocalKey<u8>>(),
        identity_tls_bytes: None,
        guard_bytes: size_of::<RuntimeLockGuard>(),
        try_guard_bytes: size_of::<Option<RuntimeLockGuard>>(),
        ordinary_wait_bytes: size_of::<Wait>() + size_of::<Action>() + size_of::<Duration>(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Spin,
    Yield,
    Sleep,
}

pub(super) struct Wait {
    failures: u8,
}

impl Wait {
    pub(super) const fn new() -> Self {
        Self { failures: 0 }
    }

    fn next(&mut self) -> Action {
        let action = match self.failures {
            0..8 => Action::Spin,
            8..16 => Action::Yield,
            _ => Action::Sleep,
        };
        self.failures = self.failures.saturating_add(1);
        action
    }

    pub(super) fn pause(&mut self) {
        let action = self.next();
        match action {
            Action::Spin => core::hint::spin_loop(),
            Action::Yield => std::thread::yield_now(),
            Action::Sleep => std::thread::sleep(Duration::from_millis(1)),
        }
        #[cfg(test)]
        OBSERVED.store(
            match action {
                Action::Spin => 1,
                Action::Yield => 2,
                Action::Sleep => 3,
            },
            std::sync::atomic::Ordering::SeqCst,
        );
    }
}

#[cfg(test)]
pub(super) static OBSERVED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_wait_reaches_capped_sleep_and_never_wraps_to_active_spin() {
        let mut wait = Wait::new();
        let mut phases = [0; 3];
        for _ in 0..1024 {
            let index = match wait.next() {
                Action::Spin => 0,
                Action::Yield => 1,
                Action::Sleep => 2,
            };
            phases[index] += 1;
        }
        assert_eq!(phases, [8, 8, 1008]);
        assert_eq!(wait.failures, u8::MAX);
    }

    #[test]
    fn runtime_lock_layout_reports_real_controls_without_inventing_tls_extent() {
        let (facts, allocations) = crate::utils::allocation_test::measure(runtime_lock_layout);
        assert_eq!(allocations, 0);
        assert_eq!(facts.mutex_bytes, size_of::<ReentrantMutex<()>>());
        assert_eq!(facts.guard_bytes, size_of::<RuntimeLockGuard>());
        assert_eq!(facts.try_guard_bytes, size_of::<Option<RuntimeLockGuard>>());
        assert_eq!(facts.identity_key_header_bytes, size_of::<LocalKey<u8>>());
        assert_eq!(facts.identity_tls_bytes, None);
        assert_eq!(
            facts.ordinary_wait_bytes,
            size_of::<Wait>() + size_of::<Action>() + size_of::<Duration>()
        );
    }
}
