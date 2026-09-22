//! Scheduled sparse provider loans use the original shared lowering worker.
use super::*;
use eredu_core::capture::RoutedUnitCaptureSource;
use eredu_runtime::{
    intervention::InterventionPrefillWindow, working_memory::RoutedInterventionBatch,
};
impl NativeScheduledCapture<'_> {
    pub(in super::super) fn scheduled_routed_range(
        &self,
        source: &RoutedUnitCaptureSource<'_, Array>,
        claim: &CaptureInterventionClaim<'_>,
        span: Option<InterventionPrefillWindow>,
    ) -> Result<[u64; 2], FundedCaptureError<Error>> {
        let result = (|| {
            let row = self.scheduled_routing_row(claim, span)?;
            let scope = self
                .work
                .scope
                .try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?;
            let scope = scope.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
            row.routed_range_scheduled(source, claim, scope, span)
                .map_err(|cause| Error::Other(Box::new(cause)))
        })();
        result.map_err(|cause| self.scheduled_failure(cause))
    }
    pub(in super::super) fn scheduled_routed_usage(
        &self,
        source: &RoutedUnitCaptureSource<'_, Array>,
        claim: &CaptureInterventionClaim<'_>,
        span: Option<InterventionPrefillWindow>,
    ) -> Result<CaptureUsage, FundedCaptureError<Error>> {
        let result = (|| {
            let row = self.scheduled_routing_row(claim, span)?;
            let scope = self
                .work
                .scope
                .try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?;
            let scope = scope.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
            let usage = row
                .routed_usage_scheduled(source, claim, scope, span)
                .map_err(|cause| Error::Other(Box::new(cause)))?;
            match span {
                Some(_) => self
                    .work
                    .text_interventions
                    .as_ref()
                    .ok_or(Error::PrefillScopeUnavailable)?
                    .prefill_routed_usage(claim.index())
                    .map_err(|cause| Error::Other(Box::new(cause))),
                None => Ok(usage),
            }
        })();
        result.map_err(|cause| self.scheduled_failure(cause))
    }
    pub(in super::super) fn scheduled_routed_apply(
        &mut self,
        source: &RoutedUnitCaptureSource<'_, Array>,
        batch: RoutedInterventionBatch<'_, '_>,
    ) -> Result<Option<Array>, FundedCaptureError<Error>> {
        let row = self
            .scheduled_routing_row(batch.claim(), batch.prefill_window())
            .map_err(|cause| self.scheduled_failure(cause))?;
        let observer = self
            .capture_observer()
            .and_then(|value| value.ok_or(Error::PrefillScopeUnavailable))
            .map_err(|cause| self.scheduled_failure(cause))?;
        let scope = self
            .work
            .scope
            .try_borrow()
            .map_err(|_| self.scheduled_failure(Error::PrefillScopeReentrant))?;
        let scope = scope
            .as_ref()
            .ok_or_else(|| self.scheduled_failure(Error::PrefillScopeUnavailable))?;
        row.execute_routed_scheduled(
            source,
            batch,
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
        size_of::<RoutedUnitCaptureSource<'_, Array>>(),
        size_of::<RoutedInterventionBatch<'_, '_>>(),
        size_of::<Option<InterventionPrefillWindow>>(),
        size_of::<[u64; 2]>(),
        size_of::<(
            &NativeScheduledCapture<'_>,
            &RoutedUnitCaptureSource<'_, Array>,
            &CaptureInterventionClaim<'_>,
            Option<InterventionPrefillWindow>,
        )>(),
        size_of::<Result<[u64; 2], FundedCaptureError<Error>>>(),
        size_of::<Result<CaptureUsage, FundedCaptureError<Error>>>(),
        size_of::<Result<Option<Array>, FundedCaptureError<Error>>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
