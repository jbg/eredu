//! Evidence follows the ordinary ledger and the existing paid record writer.
use super::*;
pub(super) fn observe<T, E: std::error::Error + Send + Sync + 'static>(
    backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
    frame: &mut crate::working_memory::ScheduledCaptureStep<'_>,
    ledger: &mut CaptureLedger,
    operation: usize,
    side: InterventionEvidenceSide,
    value: &T,
) -> Result<(), FundedCaptureError<E>> {
    let Some(claim) = frame.take_intervention_evidence(operation, side)? else {
        return Ok(());
    };
    let mut dtype = None;
    let mut charged = CaptureUsage::default();
    let mut skipped = None;
    let result: Result<
        Option<crate::working_memory::ClaimedInterventionEvidence<'_>>,
        FundedCaptureError<E>,
    > = (|| {
        if let Some(usage) = claim.window_metadata_usage()? {
            if let Some(reason) = ledger.reserve(usage)? {
                drop(claim);
                skipped = Some(reason);
                return Ok(None);
            }
            charged = usage;
        }
        let (actual, usage) = backend.intervention_evidence_usage(value, &claim)?;
        if !matches!(
            actual,
            TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16
        ) {
            return Err(CaptureProtocolError::Transaction.into());
        }
        dtype = Some(actual.clone());
        if claim.window_empty() {
            drop(claim);
            skipped = Some(CaptureSkipReason::NotInvoked);
            return Ok(None);
        }
        if let Some(reason) = ledger.reserve(usage)? {
            drop(claim);
            skipped = Some(reason);
            return Ok(None);
        }
        charged = charged.checked_add(usage)?;
        backend
            .capture_intervention_evidence(value, claim)
            .map(Some)
    })();
    // The exclusive child claim is consumed before borrowing its frame again.
    let result: Result<(), FundedCaptureError<E>> = match result {
        Ok(Some(evidence)) => frame
            .record_intervention_evidence(
                evidence,
                dtype.clone().expect("validated evidence dtype"),
                charged,
            )
            .map_err(Into::into),
        Ok(None) => frame
            .skip_intervention_evidence_with_usage(
                operation,
                side,
                skipped.expect("reserved evidence skip"),
                dtype.clone(),
                charged,
            )
            .map_err(Into::into),
        Err(cause) => Err(cause),
    };
    if let Err(cause) = result {
        let reason = match &cause {
            FundedCaptureError::Admission(error) => policy::failure_reason(error),
            _ => CaptureFailureReason::Native,
        };
        // A failed record writer may already have preserved its own diagnostic.
        // Keep the original cause even if account fencing prevents this update.
        let _ = frame.fail_intervention_evidence(operation, side, reason, dtype, charged);
        return Err(cause);
    }
    Ok(())
}
