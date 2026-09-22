//! Exact physical span lookup under the unchanged logical scheduled native scope.
use super::*;
use eredu_runtime::{
    intervention::InterventionPrefillWindow as Window, working_memory::InterventionPrefillFragment,
};
impl NativeScheduledCapture<'_> {
    fn prefill_edit_row(
        &self,
        claim: &CaptureInterventionClaim<'_>,
        window: Window,
    ) -> Result<&PreparedModelInterventions, Error> {
        let (phase, prediction) = claim.coordinate();
        let row = self
            .work
            .text_interventions
            .as_ref()
            .ok_or(Error::PrefillScopeUnavailable)?
            .row_for_prefill(phase, prediction, window)
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        row.validate_prefill_span(window)
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        Ok(row)
    }
    pub(in super::super) fn prefill_projection_usage(
        &self,
        source: &Array,
        claim: &CaptureInterventionClaim<'_>,
        window: Window,
    ) -> Result<[CaptureUsage; 2], FundedCaptureError<Error>> {
        let result = (|| {
            let row = self.prefill_edit_row(claim, window)?;
            let scope = self
                .work
                .scope
                .try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?;
            let scope = scope.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
            row.projection_usage_scheduled(source, claim, scope)
                .map_err(|cause| Error::Other(Box::new(cause)))
        })();
        result.map_err(|cause| self.scheduled_failure(cause))
    }
    pub(in super::super) fn prefill_edit_usage(
        &self,
        source: &Array,
        claim: &CaptureInterventionClaim<'_>,
        window: Window,
    ) -> Result<CaptureUsage, FundedCaptureError<Error>> {
        let result = (|| {
            let row = self.prefill_edit_row(claim, window)?;
            let scope = self
                .work
                .scope
                .try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?;
            let scope = scope.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
            row.usage_scheduled(source, claim, scope)
                .map_err(|cause| Error::Other(Box::new(cause)))
        })();
        result.map_err(|cause| self.scheduled_failure(cause))
    }
    pub(in super::super) fn prefill_edit(
        &self,
        source: &Array,
        fragment: InterventionPrefillFragment<'_, '_>,
        charged: CaptureUsage,
        projection: [CaptureUsage; 2],
    ) -> Result<Option<Array>, FundedCaptureError<Error>> {
        let row = self
            .prefill_edit_row(fragment.claim(), fragment.source())
            .map_err(|cause| self.scheduled_failure(cause))?;
        let observer = self
            .capture_observer()
            .and_then(|v| v.ok_or(Error::PrefillScopeUnavailable))
            .map_err(|cause| self.scheduled_failure(cause))?;
        let scope = self
            .work
            .scope
            .try_borrow()
            .map_err(|_| self.scheduled_failure(Error::PrefillScopeReentrant))?;
        let scope = scope
            .as_ref()
            .ok_or_else(|| self.scheduled_failure(Error::PrefillScopeUnavailable))?;
        row.execute_prefill_scheduled(
            source,
            fragment,
            charged,
            projection,
            self.stream,
            &self.work.roots,
            scope,
            &observer,
        )
        .map_err(|cause| self.native_intervention_failure(cause))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    [
        size_of::<Window>(),
        size_of::<InterventionPrefillFragment<'_, '_>>(),
        size_of::<(
            &NativeScheduledCapture<'_>,
            &Array,
            &CaptureInterventionClaim<'_>,
            Window,
        )>(),
        size_of::<Result<Option<Array>, FundedCaptureError<Error>>>(),
        size_of::<Result<[CaptureUsage; 2], FundedCaptureError<Error>>>(),
        size_of::<Result<CaptureUsage, FundedCaptureError<Error>>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
