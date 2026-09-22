//! Actual external request, immutable selected contract and monotonic attempts.
use super::{OriginalSpeculativeNumericalPreparation, OriginalSpeculativeNumericalSources};
use crate::{
    backend::error::Error,
    composition::mlx::{model::retain_planning_error, session::OriginalInterventionDeclaration},
};
use eredu_core::{
    speculative::{SpeculativeActivationOrigin, SpeculativePrefillSpan},
    InferenceGeometry, SpeculativeRequestId,
};
use eredu_nn::workspace::HostMetadataFundingError;
use eredu_runtime::{
    speculative::external_occurrence::{
        ExternalContinuation, ExternalInvocationKind, ExternalOccurrenceClaim,
        ExternalOccurrenceCursor, ExternalOccurrenceError, ExternalSchedulePlan,
    },
    working_memory::{OriginalSpeculativeRequest, WorkingMemoryError},
};
use std::{
    cell::{Cell, RefCell},
    mem::{size_of, size_of_val},
};

/// Lives outside cache/snapshot branches and never resets failed attempts.
pub(crate) struct OriginalExternalSources<'a> {
    cursor: RefCell<ExternalOccurrenceCursor<'a>>,
    scheduler_request: Cell<Option<SpeculativeRequestId>>,
    prefill_scope:
        RefCell<Option<eredu_runtime::working_memory::SpeculativePrefillScheduleAuthority>>,
    numerical: OriginalSpeculativeNumericalSources,
}
impl std::fmt::Debug for OriginalExternalSources<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OriginalExternalSources")
    }
}
impl<'a> OriginalExternalSources<'a> {
    /// The immutable target preparation precedes the selected assistant loan.
    /// This issues the real External request through the common account worker.
    pub(crate) fn prepare_from_source(
        source: OriginalSpeculativeNumericalPreparation,
        schedule: ExternalSchedulePlan<'a>,
        capacity: eredu_core::MemoryLimits,
        declaration: Option<OriginalInterventionDeclaration>,
    ) -> Result<Self, Error> {
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<ExternalSchedulePlan<'a>>(),
            size_of::<(
                OriginalSpeculativeNumericalPreparation,
                u64,
                Option<OriginalInterventionDeclaration>,
            )>(),
        ];
        source
            .metadata_funding()
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let request = OriginalSpeculativeRequest::prepare_external(
            source.pool(),
            source.execution_identity(),
            &schedule,
            capacity,
        )
        .map_err(|error| retain_planning_error(error, source.metadata_funding().clone()))?;
        let numerical = source
            .bind(request)?
            .with_intervention_declaration(declaration)?;
        Ok(Self {
            cursor: RefCell::new(schedule.into_cursor()),
            scheduler_request: Cell::new(None),
            prefill_scope: RefCell::new(None),
            numerical,
        })
    }
}

/// Erases only the lifetime of the retained selection. It cannot manufacture a
/// request, model source or native account from a matching coordinate.
pub(crate) trait ExternalInvocationSource: std::fmt::Debug {
    fn numerical_sources(&self) -> &OriginalSpeculativeNumericalSources;
    fn prefill_schedule(
        &self,
    ) -> Result<eredu_runtime::working_memory::SpeculativePrefillScheduleAuthority, Error>;
    fn claim(
        &self,
        kind: ExternalInvocationKind,
        geometry: InferenceGeometry,
        span: Option<SpeculativePrefillSpan>,
        origin: SpeculativeActivationOrigin,
    ) -> Result<ExternalOccurrenceClaim<'_>, Error>;
    fn prepare_continuation(
        &self,
        committed: usize,
        status: eredu_core::generation::SpeculativeRequestStatus,
    ) -> Result<(), Error>;
}
impl ExternalInvocationSource for OriginalExternalSources<'_> {
    fn numerical_sources(&self) -> &OriginalSpeculativeNumericalSources {
        &self.numerical
    }
    fn prefill_schedule(
        &self,
    ) -> Result<eredu_runtime::working_memory::SpeculativePrefillScheduleAuthority, Error> {
        let mut slot = self
            .prefill_scope
            .try_borrow_mut()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        if let Some(scope) = slot.as_ref() {
            return Ok(scope.clone());
        }
        let cursor = self
            .cursor
            .try_borrow()
            .map_err(|_| Error::PrefillScopeReentrant)?;
        let scope = self
            .numerical
            .request()
            .prepare_external_prefill_schedule(cursor.plan(), self.numerical.metadata_funding())
            .map_err(|error| self.numerical.retain_startup_error(error))?;
        self.numerical.bind_prefill_schedule(&scope)?;
        *slot = Some(scope.clone());
        Ok(scope)
    }
    fn claim(
        &self,
        kind: ExternalInvocationKind,
        geometry: InferenceGeometry,
        span: Option<SpeculativePrefillSpan>,
        origin: SpeculativeActivationOrigin,
    ) -> Result<ExternalOccurrenceClaim<'_>, Error> {
        let frames = [
            size_of::<ExternalOccurrenceClaim<'_>>(),
            size_of::<Result<ExternalOccurrenceClaim<'_>, Error>>(),
            size_of::<std::cell::RefMut<'_, ExternalOccurrenceCursor<'_>>>(),
            size_of::<(
                ExternalInvocationKind,
                InferenceGeometry,
                Option<SpeculativePrefillSpan>,
                SpeculativeActivationOrigin,
            )>(),
            size_of::<Result<ExternalOccurrenceClaim<'_>, ExternalOccurrenceError>>(),
        ];
        self.numerical
            .metadata_funding()
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        if self
            .scheduler_request
            .get()
            .is_some_and(|prior| prior != origin.request)
        {
            return Err(self
                .numerical
                .retain_startup_error(WorkingMemoryError::IdentityMismatch));
        }
        let mut cursor = self.cursor.try_borrow_mut().map_err(|_| {
            self.numerical
                .retain_startup_error(WorkingMemoryError::AccountConstructionBusy)
        })?;
        let invocation = cursor
            .plan()
            .invocation(kind, geometry, span, origin)
            .map_err(|error| self.numerical.retain_startup_error(error))?;
        self.scheduler_request.set(Some(origin.request));
        cursor
            .claim(invocation)
            .map_err(|error| self.numerical.retain_startup_error(error))
    }
    fn prepare_continuation(
        &self,
        committed: usize,
        status: eredu_core::generation::SpeculativeRequestStatus,
    ) -> Result<(), Error> {
        let frames = [
            size_of::<(
                &Self,
                usize,
                eredu_core::generation::SpeculativeRequestStatus,
            )>(),
            size_of::<ExternalContinuation>(),
            size_of::<Result<ExternalContinuation, ExternalOccurrenceError>>(),
            size_of::<std::cell::RefMut<'_, ExternalOccurrenceCursor<'_>>>(),
            size_of::<Result<(), Error>>(),
            size_of::<Result<(), ExternalOccurrenceError>>(),
            size_of::<Result<(), eredu_runtime::working_memory::SpeculativeContinuationError>>(),
        ];
        let funding = self.numerical.metadata_funding();
        funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let mut cursor = self.cursor.try_borrow_mut().map_err(|_| {
            self.numerical
                .retain_startup_error(WorkingMemoryError::AccountConstructionBusy)
        })?;
        let continuation = cursor
            .continuation(committed, status)
            .map_err(|error| self.numerical.retain_startup_error(error))?;
        self.numerical
            .request()
            .prepare_external_continuation(&continuation, funding)
            .map_err(|error| self.numerical.retain_startup_error(error))?;
        cursor
            .install_continuation(continuation)
            .map_err(|error| self.numerical.retain_startup_error(error))
    }
}
