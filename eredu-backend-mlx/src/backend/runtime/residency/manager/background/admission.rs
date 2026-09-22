//! One physical source birth per selected occurrence; failures close the source.
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Slot {
    Unread,
    Reading,
    Ready,
    Consumed,
}
#[derive(Default)]
pub(super) struct Admission {
    closed: AtomicBool,
}
/// A stack-only failure fence. Its caller retains the actual source owner;
/// unwind and every early return close admission without taking a mutex.
pub(crate) struct BackgroundSourceAttempt<'a> {
    admission: &'a Admission,
    succeeded: bool,
}
impl BackgroundSourceAttempt<'_> {
    pub(crate) fn succeed(mut self) {
        self.succeeded = true;
    }
}
impl Drop for BackgroundSourceAttempt<'_> {
    fn drop(&mut self) {
        if !self.succeeded {
            self.admission.close();
        }
    }
}
impl Admission {
    pub(super) fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }
    pub(super) fn attempt(&self) -> Result<BackgroundSourceAttempt<'_>, ()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(());
        }
        Ok(BackgroundSourceAttempt {
            admission: self,
            succeeded: false,
        })
    }
    pub(super) fn submit(&self, slot: Slot) -> Result<(), ()> {
        let attempt = self.attempt()?;
        if slot == Slot::Consumed {
            return Err(());
        }
        attempt.succeed();
        Ok(())
    }
    /// Returns false only when the exact completed payload remains in its slot.
    /// Once taken, that unit has no second birth in this source generation.
    pub(super) fn begin(&self, slot: &mut Slot) -> Result<bool, ()> {
        let attempt = self.attempt()?;
        let read = match slot {
            Slot::Unread => {
                *slot = Slot::Reading;
                true
            }
            Slot::Ready => false,
            Slot::Reading | Slot::Consumed => return Err(()),
        };
        attempt.succeed();
        Ok(read)
    }
    /// A successful occurrence may be followed by another selected window.
    /// This is never called by ordinary submission or after failure/cancel.
    /// It grants no read: the actual next finite slot and native retirement
    /// allowance remain mandatory in the shared source worker.
    pub(super) fn rearm_window(&self, slot: &mut Slot) -> Result<(), ()> {
        let attempt = self.attempt()?;
        match slot {
            Slot::Consumed => *slot = Slot::Unread,
            Slot::Unread | Slot::Ready => {}
            Slot::Reading => return Err(()),
        }
        attempt.succeed();
        Ok(())
    }
    pub(super) fn complete(&self, slot: &mut Slot) -> Result<(), ()> {
        let attempt = self.attempt()?;
        if *slot != Slot::Reading {
            return Err(());
        }
        *slot = Slot::Ready;
        attempt.succeed();
        Ok(())
    }
    pub(super) fn take(&self, slot: &mut Slot) -> Result<bool, ()> {
        let attempt = self.attempt()?;
        let ready = match slot {
            Slot::Ready => {
                *slot = Slot::Consumed;
                true
            }
            Slot::Unread => false,
            Slot::Reading | Slot::Consumed => return Err(()),
        };
        attempt.succeed();
        Ok(ready)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn consumed_source_cannot_allocate_again_while_its_batch_escapes() {
        let admission = Admission::default();
        let mut slot = Slot::Unread;
        assert!(admission.begin(&mut slot).unwrap());
        admission.complete(&mut slot).unwrap();
        assert!(!admission.begin(&mut slot).unwrap());
        assert!(admission.take(&mut slot).unwrap());
        // A returned batch may remain in a published host owner, transfer, or
        // rollback error. Ordinary resubmission is not a retirement fact.
        assert!(admission.submit(slot).is_err());
        assert!(admission.begin(&mut Slot::Unread).is_err());
    }
    #[test]
    fn next_selected_window_rearms_only_successful_consumption() {
        let admission = Admission::default();
        let mut slot = Slot::Unread;
        admission.rearm_window(&mut slot).unwrap();
        assert!(admission.begin(&mut slot).unwrap());
        admission.complete(&mut slot).unwrap();
        assert!(admission.take(&mut slot).unwrap());
        admission.rearm_window(&mut slot).unwrap();
        assert!(admission.begin(&mut slot).unwrap());
        admission.close();
        assert!(admission.rearm_window(&mut slot).is_err());
        assert!(admission.rearm_window(&mut Slot::Consumed).is_err());
    }
    #[test]
    fn failure_unwind_and_cancellation_close_later_units() {
        for cancel in [false, true] {
            let admission = Admission::default();
            let attempt = admission.attempt().unwrap();
            let mut slot = Slot::Unread;
            assert!(admission.begin(&mut slot).unwrap());
            if cancel {
                admission.close();
            }
            drop(attempt); // Same RAII path as a retained read/publication error.
            assert!(admission.complete(&mut slot).is_err());
            assert!(admission.submit(Slot::Unread).is_err());
            assert!(admission.begin(&mut Slot::Unread).is_err());
        }
        let admission = Admission::default();
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _attempt = admission.attempt().unwrap();
            panic!("source worker panicked after a partial read");
        }));
        assert!(unwind.is_err());
        assert!(admission.submit(Slot::Unread).is_err());
    }
}
