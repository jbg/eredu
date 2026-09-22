//! The funded partition program borrows the same sealed original model row.
use super::*;
use crate::composition::mlx::session::intervention::PreparedPartitionModelIntervention;
use eredu_runtime::{
    capture::partition::PartitionInterventionLocalAllowance,
    intervention::InterventionPrefillWindow as Window,
};
impl NativeScheduledCapture<'_> {
    fn partition_edit_source(
        &self,
        claim: &CaptureInterventionClaim<'_>,
        window: Option<Window>,
    ) -> Result<(&PreparedPartitionModelIntervention, usize), Error> {
        let (phase, prediction) = claim.coordinate();
        let rows = self
            .work
            .text_interventions
            .as_ref()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let row = match window {
            Some(window) => rows.row_for_prefill(phase, prediction, window),
            None => rows.row(phase, prediction),
        }
        .map_err(|cause| Error::Other(Box::new(cause)))?;
        let (rank, _, _) = row
            .partition_binding()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let source = row
            .partition_source(claim.index())
            .map_err(|cause| Error::Other(Box::new(cause)))?
            .ok_or(Error::PrefillScopeUnavailable)?;
        if source.window() != window {
            return Err(Error::Other(Box::new(NativeFailure::SourceChanged)));
        }
        Ok((source, rank))
    }
    pub(in super::super) fn partition_edit_validate(
        &self,
        value: &Array,
        claim: &CaptureInterventionClaim<'_>,
        window: Option<Window>,
    ) -> Result<(), FundedCaptureError<Error>> {
        let result = (|| {
            let (source, _) = self.partition_edit_source(claim, window)?;
            let scope = self
                .work
                .scope
                .try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?;
            let scope = scope.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
            source
                .validate_source(value, claim, scope)
                .map_err(|cause| Error::Other(Box::new(cause)))
        })();
        result.map_err(|cause| self.scheduled_failure(cause))
    }
    pub(in super::super) fn partition_evidence_validate(
        &self,
        value: &Array,
        claim: &CaptureInterventionClaim<'_>,
        side: eredu_core::capture::InterventionEvidenceSide,
        window: Option<Window>,
    ) -> Result<(), FundedCaptureError<Error>> {
        let result = (|| {
            let (source, rank) = self.partition_edit_source(claim, window)?;
            let source = source
                .evidence()
                .ok_or_else(|| Error::Other(Box::new(NativeFailure::SourceChanged)))?;
            if source.rank() != rank {
                return Err(Error::Other(Box::new(NativeFailure::ClaimMismatch)));
            }
            let scope = self
                .work
                .scope
                .try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?;
            let scope = scope.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
            source
                .validate_native(value, claim, side, window, scope)
                .map_err(|cause| Error::Other(Box::new(cause)))
        })();
        result.map_err(|cause| self.scheduled_failure(cause))
    }
    pub(in super::super) fn partition_edit_apply(
        &self,
        value: &Array,
        claim: &CaptureInterventionClaim<'_>,
        allowance: &mut PartitionInterventionLocalAllowance,
    ) -> Result<Option<Array>, FundedCaptureError<Error>> {
        let result = (|| {
            let (source, rank) = self.partition_edit_source(claim, allowance.window())?;
            if rank != allowance.rank() {
                return Err(Error::Other(Box::new(NativeFailure::ClaimMismatch)));
            }
            let observer = self
                .capture_observer()?
                .ok_or(Error::PrefillScopeUnavailable)?;
            let scope = self
                .work
                .scope
                .try_borrow()
                .map_err(|_| Error::PrefillScopeReentrant)?;
            let scope = scope.as_ref().ok_or(Error::PrefillScopeUnavailable)?;
            source
                .execute(
                    value,
                    claim,
                    allowance,
                    scope,
                    self.stream,
                    &observer,
                    &self.work.roots,
                )
                .map_err(|cause| Error::Other(Box::new(cause)))
        })();
        result.map_err(|cause| self.scheduled_failure(cause))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    [
        size_of::<Option<Window>>(),
        size_of::<(
            &NativeScheduledCapture<'_>,
            &Array,
            &CaptureInterventionClaim<'_>,
            eredu_core::capture::InterventionEvidenceSide,
            Option<Window>,
        )>(),
        size_of::<(
            &NativeScheduledCapture<'_>,
            &Array,
            &CaptureInterventionClaim<'_>,
            Option<Window>,
        )>(),
        size_of::<(
            &NativeScheduledCapture<'_>,
            &Array,
            &CaptureInterventionClaim<'_>,
            &mut PartitionInterventionLocalAllowance,
        )>(),
        size_of::<Result<(&PreparedPartitionModelIntervention, usize), Error>>(),
        size_of::<Result<Option<Array>, FundedCaptureError<Error>>>(),
        size_of::<Result<(), FundedCaptureError<Error>>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
