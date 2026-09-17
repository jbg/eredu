//! Explicit numerical realization at the already-authorized loading boundary.

use super::Executable;
use crate::backend::{error::Error, managed_memory::NativeMemoryOwner, MlxCompletion};
use eredu_core::Completion;

impl Executable {
    /// Settles only values already retained by the loaded target/prediction
    /// policies, including existing dormant prediction prototype arrays. No
    /// new lane, prototype or target decoder state is constructed here.
    /// Cold inventory stays observational; a fresh inventory after this operation
    /// determines physical publication, including any still-incomplete traversal.
    ///
    /// The caller holds this loading authority and its existing detached recovery
    /// scope across this call, initial publication, and every error path. The
    /// completion separately retains all actual submitted roots until terminal
    /// native proof. No ownership is permanently attached to these arrays here.
    /// This must not be used to load units or certify whole-model coverage.
    pub(crate) fn settle_loaded_numerical_values(
        &self,
        _owner: &NativeMemoryOwner,
    ) -> Result<(), Error> {
        let storage = {
            // This model is still private to ordinary loading: its new dense
            // workers have no submitted forward windows. Serialize the required
            // inventory against unrelated old workers without changing cold or
            // original inspection into a waiting operation. No manager/source
            // loan precedes entry, and no producer or wait retains the guard.
            let _inspection = safemlx::OrdinaryArrayMetadataGuard::enter()?;
            let mut storage = self.erased().retained_target_module_storage()?;
            storage.merge(self.erased().retained_prediction_storage()?)?;
            storage
        };
        settle_loaded_numerical_values(storage, _owner)
    }

    /// Test only: fail after actual submission, before its wait or publication.
    /// The original typed cause is returned through ordinary loading recovery.
    #[cfg(test)]
    pub(crate) fn reject_next_loaded_helper_finalization_for_test(error: Error) {
        let previous = REJECT_AFTER_SUBMISSION.with(|failure| failure.replace(Some(error)));
        assert!(
            previous.is_none(),
            "one loading failure injection at a time"
        );
    }
}

#[cfg(test)]
thread_local! {
    static REJECT_AFTER_SUBMISSION: std::cell::RefCell<Option<Error>> = const {
        std::cell::RefCell::new(None)
    };
}

/// Shared explicit loading settlement. The caller supplies the already retained
/// numerical inventory and keeps ordinary loading exclusion through publication.
/// This never attaches that exclusion to an allocation or promotes its custody.
pub(in crate::composition) fn settle_loaded_numerical_values(
    storage: crate::backend::runtime::residency::storage::RetainedStorage,
    _owner: &NativeMemoryOwner,
) -> Result<(), Error> {
    let mut roots = storage
        .into_retained_arrays()
        .map_err(|(cause, _storage)| cause)?;
    let Some(first) = roots.next() else {
        return Ok(());
    };
    // This existing completion gathers every supplied descriptor before it
    // enters native submission. Its recovery and the enclosing loading scope
    // cover both submission/wait errors and unwinding before publication.
    let submission = MlxCompletion::submission_retaining(first, roots)?;
    #[cfg(test)]
    if let Some(error) = REJECT_AFTER_SUBMISSION.with(|failure| failure.borrow_mut().take()) {
        return Err(error);
    }
    submission.completion.wait()?;
    Ok(())
}
