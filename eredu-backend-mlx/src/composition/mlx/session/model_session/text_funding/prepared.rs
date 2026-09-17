//! Original preparation retains the same Work allocation before exposing funding.
use super::*;
use eredu_runtime::working_memory::PreparedWorkingMemoryFundingScope;
use std::mem::size_of;

/// Sole activation authority. No Work/Deref/Clone/native-root API is exposed.
/// Scope cancellation precedes the same Work's original full guard.
pub(in crate::composition::mlx::session::model_session) struct PreparedFundedWork {
    scope: Option<PreparedWorkingMemoryFundingScope>,
    work: FundedWorkOwner,
}

/// Recovery's alias retains Q and source owners, but has no activation or Work
/// access. Before activation it reports nothing and certifies nothing.
pub(in crate::composition::mlx::session::model_session) struct PreparedWorkRetention {
    work: FundedWorkOwner,
}

impl PreparedFundedWork {
    pub(in crate::composition::mlx::session::model_session) fn new(
        scope: PreparedWorkingMemoryFundingScope,
        controls: OriginalTextControlGuard,
        rows: Option<crate::composition::mlx::replicated_text::NativeOpeningRowsOwner>,
    ) -> Result<Self, Error> {
        Self::new_with_native(scope, controls, rows, None, None)
    }
    pub(in crate::composition::mlx::session::model_session) fn new_with_native(
        scope: PreparedWorkingMemoryFundingScope,
        controls: OriginalTextControlGuard,
        rows: Option<crate::composition::mlx::replicated_text::NativeOpeningRowsOwner>,
        prepared_source: Option<eredu_runtime::input::OriginalPreparedWorkspaceSource>,
        native_storage: Option<
            crate::backend::runtime::residency::storage::native_storage::BankOwner,
        >,
    ) -> Result<Self, Error> {
        // Stage the closed original owners before validation. On rejection,
        // unused cancellation runs before Work/rows/full-custody destruction.
        let prepared = Self {
            scope: Some(scope),
            work: FundedWork::allocate(None, Some(controls), rows, None, None, prepared_source, native_storage, None),
        };
        let scope = prepared.scope.as_ref().expect("prepared funding scope");
        prepared
            .work
            ._controls
            .as_ref()
            .expect("original controls")
            .validate_prepared_scope(scope)
            .map_err(|e| Error::Other(Box::new(e)))?;
        if let Some(source) = &prepared.work.prepared_source {
            let controls = prepared.work._controls.as_ref().expect("original controls");
            if !controls.metadata_custody().matches_domain(source.pool().shared_storage_domain()) {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
        }
        if let Some(rows) = &prepared.work.opening_rows {
            rows.validate_prepared_scope(scope)?;
        }
        prepared.work.prepare_collectors()?;
        Ok(prepared)
    }

    pub(in crate::composition::mlx::session::model_session) fn retention(
        &self,
    ) -> PreparedWorkRetention {
        PreparedWorkRetention {
            work: self.work.clone(),
        }
    }

    /// Called after actual native begin, before any worker. The slot is private
    /// to this owner and its callback-only aliases release every loan before
    /// returning. Only infallible moves follow extraction of the exact scope.
    pub(in crate::composition::mlx::session::model_session) fn activate(
        mut self,
    ) -> FundedWorkOwner {
        {
            let mut slot = self.work.scope.borrow_mut();
            assert!(slot.is_none(), "unexposed original funding slot");
            let scope = self.scope.take().expect("unconsumed prepared Work");
            *slot = Some(scope.activate());
        }
        self.work
    }

    pub(in crate::composition::mlx::session::model_session) fn control_bytes() -> Option<usize> {
        [
            size_of::<Self>(), // constructor/activation local, not stored quote field
            size_of::<Result<Self, Error>>(),
            size_of::<PreparedWorkRetention>(), // callback alias construction/move
            size_of::<Option<PreparedWorkingMemoryFundingScope>>(), // activation extraction
            size_of::<std::cell::RefMut<'_, Option<WorkingMemoryFundingScope>>>(),
            size_of::<std::cell::Ref<'_, Option<WorkingMemoryFundingScope>>>(),
            size_of::<Result<(), Error>>(), // original validation result
        ]
        .into_iter()
        .try_fold(
            PreparedWorkingMemoryFundingScope::control_bytes()?,
            usize::checked_add,
        )
    }

    #[cfg(test)]
    pub(in crate::composition::mlx::session::model_session) fn identity(&self) -> usize {
        std::ptr::from_ref(self.work.as_ref()) as usize
    }
}

impl PreparedWorkRetention {
    #[cfg(test)]
    pub(in crate::composition::mlx::session::model_session) fn phase(&self) -> (bool, bool) {
        (
            self.work.scope.borrow().is_some(),
            self.work.published.get(),
        )
    }
    pub(in crate::composition::mlx::session::model_session) fn observe(&self, status: Status) {
        let active = { self.work.scope.borrow().is_some() };
        if active {
            self.work.observe_preparation(status);
        }
    }
}
