//! Same real SessionPrefill fixture, with the closed successful group installed.
//! These tests do not activate remaining-span/native execution authority.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OpeningMode {
    Success,
    ForeignStamp,
    Busy,
    OriginBeforeInstall,
    OriginAfterInstall,
    Panic,
    Abandon,
}
impl Backend {
    pub(super) fn install_opening(
        &mut self,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        let mode = self.opening_mode.unwrap();
        let segment = self.segment.as_ref().unwrap();
        // The owner is a field before any provider comparison or lock attempt.
        self.opening_pending = Some(self.groups.last().unwrap().clone());
        let pending = &mut self.opening_pending;
        for foreign in [
            self.sibling.as_mut().unwrap(),
            self.origin.as_mut().unwrap(),
        ] {
            assert!(matches!(
                segment.install_opening_group(foreign, context, pending),
                Err(BoundedPinError::Storage(
                    WorkingMemoryError::IdentityMismatch
                ))
            ));
            assert!(pending.is_some());
        }
        let scope = self.scope.as_mut().unwrap();
        if mode == OpeningMode::ForeignStamp && context.chunk().input.start > 0 {
            let mut previous = Some(self.groups[0].clone());
            assert!(matches!(
                segment.install_opening_group(scope, context, &mut previous),
                Err(BoundedPinError::Storage(
                    WorkingMemoryError::IdentityMismatch
                ))
            ));
            assert!(previous.is_some());
        }
        if mode == OpeningMode::Busy {
            let c = self.contention.as_ref().unwrap();
            c.key.0.armed.store(true, AtomicOrdering::SeqCst);
            let key = c.key.clone();
            let pool = scope.pool().clone();
            let worker =
                std::thread::spawn(move || pool.pin_registered_storage([(key, 1)]).unwrap());
            c.entered
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            let result = segment.install_opening_group(scope, context, pending);
            c.release.send(()).unwrap();
            drop(worker.join().unwrap());
            assert!(matches!(result, Err(BoundedPinError::Busy)));
            assert!(pending.is_some());
            assert_eq!(self.slots.spent_rows(), 1);
        }
        if mode == OpeningMode::OriginBeforeInstall {
            drop(self.origin.take());
        }
        if mode == OpeningMode::Panic {
            self.keys[0]
                .probe
                .panic_after
                .store(1, AtomicOrdering::SeqCst);
        }
        let result = segment.install_opening_group(scope, context, pending);
        if mode == OpeningMode::OriginBeforeInstall {
            assert!(matches!(
                result,
                Err(BoundedPinError::Storage(
                    WorkingMemoryError::ExecutionFenced
                ))
            ));
            assert!(pending.is_some());
            return Err(FundedCaptureError::Backend(Error::backend_retained_source(
                Original(self.opening_failure_identity.clone()),
            )));
        }
        result.map_err(|e| FundedCaptureError::Backend(Error::backend_retained_source(e)))?;
        assert!(pending.is_none());
        let mut duplicate = Some(self.groups.last().unwrap().clone());
        assert!(matches!(
            segment.install_opening_group(scope, context, &mut duplicate),
            Err(BoundedPinError::Storage(
                WorkingMemoryError::IdentityMismatch
            ))
        ));
        assert!(duplicate.is_some());
        segment.validate_native_scope(scope).unwrap();
        if mode == OpeningMode::OriginAfterInstall {
            drop(self.origin.take());
        }
        Ok(())
    }
}

#[test]
fn canonical_opening_group_keeps_source_and_original_custody_until_final_parcel() {
    exercise_pins_inner(Mode::Success, 1, Some(OpeningMode::Success));
}
#[test]
fn canonical_opening_group_rejects_foreign_accounts_stamps_and_duplicate_install() {
    exercise_pins_inner(Mode::Success, 3, Some(OpeningMode::ForeignStamp));
}
#[test]
fn canonical_busy_install_preserves_same_owner_and_row_for_successful_retry() {
    exercise_pins_inner(Mode::Success, 1, Some(OpeningMode::Busy));
}
#[test]
fn canonical_origin_closure_before_or_after_install_preserves_original_quarantine() {
    for mode in [
        OpeningMode::OriginBeforeInstall,
        OpeningMode::OriginAfterInstall,
    ] {
        exercise_pins_inner(Mode::Success, 1, Some(mode));
    }
}
#[test]
fn canonical_install_comparison_panic_keeps_caller_pending_owner() {
    exercise_pins_inner(Mode::Success, 1, Some(OpeningMode::Panic));
}
#[test]
fn canonical_retirement_failure_keeps_installed_group_in_original_scope_quarantine() {
    exercise_pins_inner(Mode::Success, 1, Some(OpeningMode::Abandon));
}
