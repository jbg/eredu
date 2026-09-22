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
    observe_field(backend, frame, ledger, operation, side, None, value)
}
pub(in crate::capture::funded::observer) fn observe_routing<
    T,
    E: std::error::Error + Send + Sync + 'static,
>(
    backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
    frame: &mut crate::working_memory::ScheduledCaptureStep<'_>,
    ledger: &mut CaptureLedger,
    operation: usize,
    side: InterventionEvidenceSide,
    field: eredu_core::RoutingObservationField,
    value: &T,
) -> Result<(), FundedCaptureError<E>> {
    observe_field(backend, frame, ledger, operation, side, Some(field), value)
}
fn observe_field<T, E: std::error::Error + Send + Sync + 'static>(
    backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
    frame: &mut crate::working_memory::ScheduledCaptureStep<'_>,
    ledger: &mut CaptureLedger,
    operation: usize,
    side: InterventionEvidenceSide,
    field: Option<eredu_core::RoutingObservationField>,
    value: &T,
) -> Result<(), FundedCaptureError<E>> {
    let Some(claim) = (match field {
        Some(field) => frame.take_routing_intervention_evidence(operation, side, field)?,
        None => frame.take_intervention_evidence(operation, side)?,
    }) else {
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
        let valid_dtype = match field {
            Some(eredu_core::RoutingObservationField::SelectedExperts) => matches!(
                actual,
                TensorDtype::U8 | TensorDtype::U16 | TensorDtype::U32 | TensorDtype::U64
            ),
            _ => matches!(
                actual,
                TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16
            ),
        };
        if !valid_dtype {
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
        Ok(None) => match field {
            Some(field) => frame.skip_routing_intervention_evidence_with_usage(
                operation,
                side,
                field,
                skipped.expect("reserved evidence skip"),
                dtype.clone(),
                charged,
            ),
            None => frame.skip_intervention_evidence_with_usage(
                operation,
                side,
                skipped.expect("reserved evidence skip"),
                dtype.clone(),
                charged,
            ),
        }
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
        let _ = match field {
            Some(field) => frame
                .fail_routing_intervention_evidence(operation, side, field, reason, dtype, charged),
            None => frame.fail_intervention_evidence(operation, side, reason, dtype, charged),
        };
        return Err(cause);
    }
    Ok(())
}

/// Uses the original companion's paid full target across actual prompt chunks.
pub(super) fn observe_prefill<T, E: std::error::Error + Send + Sync + 'static>(
    backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
    frame: &mut crate::working_memory::ScheduledCaptureStep<'_>,
    ledger: &mut CaptureLedger,
    operation: usize,
    side: InterventionEvidenceSide,
    value: &T,
    inference: eredu_core::InferenceGeometry,
    chunk: &crate::prefill::PrefillChunk,
) -> Result<(), FundedCaptureError<E>> {
    observe_prefill_field(
        backend, frame, ledger, operation, side, None, value, inference, chunk,
    )
}
pub(in crate::capture::funded::observer) fn observe_routing_prefill<
    T,
    E: std::error::Error + Send + Sync + 'static,
>(
    backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
    frame: &mut crate::working_memory::ScheduledCaptureStep<'_>,
    ledger: &mut CaptureLedger,
    operation: usize,
    side: InterventionEvidenceSide,
    field: eredu_core::RoutingObservationField,
    value: &T,
    inference: eredu_core::InferenceGeometry,
    chunk: &crate::prefill::PrefillChunk,
) -> Result<(), FundedCaptureError<E>> {
    observe_prefill_field(
        backend,
        frame,
        ledger,
        operation,
        side,
        Some(field),
        value,
        inference,
        chunk,
    )
}
fn observe_prefill_field<T, E: std::error::Error + Send + Sync + 'static>(
    backend: &mut dyn ScheduledCaptureBackend<Tensor = T, Error = E>,
    frame: &mut crate::working_memory::ScheduledCaptureStep<'_>,
    ledger: &mut CaptureLedger,
    operation: usize,
    side: InterventionEvidenceSide,
    field: Option<eredu_core::RoutingObservationField>,
    value: &T,
    inference: eredu_core::InferenceGeometry,
    chunk: &crate::prefill::PrefillChunk,
) -> Result<(), FundedCaptureError<E>> {
    use crate::capture::{CapturePrefillHookDecision, CapturePrefillObservationPolicy};
    let Some(plan) = frame.intervention_admission() else {
        return Ok(());
    };
    if plan.plan().operations[operation].evidence
        == eredu_core::intervention::InterventionEvidence::None
    {
        return Ok(());
    }
    let mut loan = frame.take_intervention_evidence_frame(operation)?;
    let index = match field {
        Some(field) => side
            .routing_index(field)
            .ok_or(CaptureProtocolError::Transaction)?,
        None => side.index(),
    };
    let mut dtype = None;
    let result = (|| {
        if loan.frame().prefill_geometry() != Some(inference) {
            return Err(CaptureProtocolError::PrefillAttribution.into());
        }
        let source = loan.source();
        let policy = CapturePrefillObservationPolicy::new(source, inference)
            .map_err(super::super::fragments::progress_error)?;
        let row = policy
            .row(index)
            .map_err(super::super::fragments::progress_error)?;
        let path = &source.admission().plan().selections[index].path;
        let decision = loan.frame_mut().begin_prefill_hook(index, chunk, path)?;
        if decision == CapturePrefillHookDecision::Ignore {
            return Ok(());
        }
        super::super::fragments::observe_fragment_target(
            backend,
            loan.frame_mut(),
            ledger,
            &row,
            index,
            decision,
            inference,
            chunk,
            value,
            &mut dtype,
        )
    })();
    if let Err(error) = &result {
        loan.frame_mut().fail_prefill_hook(index);
        let reason = match error {
            FundedCaptureError::Admission(error) => policy::failure_reason(error),
            _ => CaptureFailureReason::Native,
        };
        if matches!(
            loan.frame().records()[index].outcome,
            CaptureOutcome::Missing
        ) {
            let _ = loan.frame_mut().record_failure(
                index,
                reason,
                "prefill intervention evidence failed",
                dtype,
                CaptureUsage::default(),
            );
        }
    }
    let restored = frame.return_intervention_evidence_frame(loan);
    result.and_then(|()| restored.map_err(Into::into))
}

pub(in crate::capture::funded::observer) fn complete_prefill<
    E: std::error::Error + Send + Sync + 'static,
>(
    frame: &mut crate::working_memory::ScheduledCaptureStep<'_>,
    inference: eredu_core::InferenceGeometry,
    chunk: &crate::prefill::PrefillChunk,
) -> Result<(), FundedCaptureError<E>> {
    let Some(plan) = frame.intervention_admission() else {
        return Ok(());
    };
    for (index, (operation, point)) in plan.plan().operations.iter().zip(plan.points()).enumerate()
    {
        if !operation.schedule.includes(CapturePhase::Prefill, 0)
            || !crate::intervention::InterventionPrefillWindow::row_axis(point)
            || operation.evidence == eredu_core::intervention::InterventionEvidence::None
        {
            continue;
        }
        let mut loan = frame.take_intervention_evidence_frame(index)?;
        let result = (|| {
            if loan.frame().prefill_geometry() != Some(inference) {
                return Err(CaptureProtocolError::PrefillAttribution.into());
            }
            loan.frame_mut()
                .complete_prefill_chunk(chunk.input.start / inference.prefill_chunk_positions)?;
            if chunk.input.end == inference.input_positions {
                loan.frame_mut().finish_prefill_targets()?;
            }
            Ok(())
        })();
        let restored = frame.return_intervention_evidence_frame(loan);
        result.and_then(|()| restored.map_err(FundedCaptureError::from))?;
    }
    Ok(())
}

pub(super) fn prefill_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<crate::working_memory::InterventionEvidenceFrame<'_>>(),
        size_of::<crate::capture::CapturePrefillObservationPolicy<'_>>(),
        size_of::<crate::capture::CapturePrefillObservationRow<'_>>(),
        size_of::<Result<(), FundedCaptureError<std::convert::Infallible>>>(),
        size_of::<Result<(), CaptureRunHostError>>(),
        size_of::<Option<TensorDtype>>(),
        size_of::<(
            &mut dyn ScheduledCaptureBackend<Tensor = (), Error = std::convert::Infallible>,
            &mut crate::working_memory::ScheduledCaptureStep<'_>,
            &mut CaptureLedger,
            usize,
            InterventionEvidenceSide,
            &(),
            eredu_core::InferenceGeometry,
            &crate::prefill::PrefillChunk,
        )>(),
        size_of::<(
            &mut crate::working_memory::ScheduledCaptureStep<'_>,
            eredu_core::InferenceGeometry,
            &crate::prefill::PrefillChunk,
        )>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
