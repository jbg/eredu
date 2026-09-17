use super::*;

struct CompletionProbe(Rc<Cell<Status>>);
impl Probe for CompletionProbe {
    fn seal(&mut self) {}
    fn progress(&self) -> Status {
        self.0.get()
    }
}

#[test]
fn saved_copy_completion_failure_keeps_pending_source_and_submission_authority() {
    for failed in [false, true] {
        let mut authority = eredu_core::SessionAuthority::new();
        let poison = Rc::new(Cell::new(false));
        let roots = Rc::new(RefCell::new(Vec::new()));
        let source = Rc::downgrade(&roots);
        let owner = SubmissionResources::with_purpose(
            authority.begin_submission().unwrap(),
            poison.clone(),
            SubmissionPurpose::SavedArrayCopy(roots),
        );
        let status = Rc::new(Cell::new(Status {
            settled: false,
            failed,
            blocked: !failed,
        }));
        let recovery = Recovery::with_probe(owner.ticket(), CompletionProbe(status.clone()));
        let result = complete_model_operation((), owner, recovery);
        assert!(matches!(
            result,
            Err(Error::SavedCopyCompletion {
                settled: false,
                failed: actual,
                blocked,
            }) if actual == failed && blocked == !failed
        ));
        assert!(poison.get());
        assert!(authority.require_idle().is_err());
        assert!(source.upgrade().is_some());

        // A failed observation alone cannot release the source or its lease.
        // The same retained probe must independently establish settlement.
        status.set(Status {
            settled: true,
            failed,
            blocked: false,
        });
        crate::backend::submission_recovery::wait_for_retirement(|| {
            authority.require_idle().is_ok()
        });
        assert!(authority.require_idle().is_ok());
        assert!(source.upgrade().is_none());
    }
}
