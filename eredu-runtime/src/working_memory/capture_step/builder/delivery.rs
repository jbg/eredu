use super::*;

/// Completely constructed frame with a uniquely owned, preallocated shared
/// destination. It contains no source/claim-row borrow and exposes no payload
/// before the enclosing transaction selects its final outcome.
#[derive(Debug)]
pub struct PreparedCaptureDelivery {
    frame: UnpublishedCapturedStep,
}
impl PreparedCaptureDelivery {
    /// Publish the existing frame with the enclosing transaction's outcome.
    /// This moves ownership without allocation, callback, account mutation or
    /// native certification. It remains valid after parent close/quarantine:
    /// the payload was already completed while that parent was healthy.
    pub fn finish(self, outcome: CaptureStepOutcome) -> SharedCapturedStep {
        self.frame.finish(outcome)
    }
}
/// Fully owned frame and all still-unused payload, before its final custody.
/// No plan/claim-row borrow or execution authority is retained here.
pub(super) struct OwnedCaptureDelivery {
    pub(super) frame: CapturedStep,
    pub(super) buffers: Vec<RecordBuffers>,
    pub(super) intervention_buffers: Vec<Vec<u8>>,
    pub(super) prefill: Option<crate::working_memory::capture_tensor::prefill::PrefillTargets>,
    pub(super) partition_metadata: Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
    pub(super) custody: CaptureTensorCustody,
}
impl fmt::Debug for OwnedCaptureDelivery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedCaptureDelivery")
            .field("records", &self.frame.records.len())
            .finish_non_exhaustive()
    }
}
#[derive(Debug)]
pub(super) struct OwnedCaptureDeliveryError {
    // An error may itself own a diagnostic; retire it before payload custody.
    error: CaptureStepError,
    payload: OwnedCaptureDelivery,
}
impl OwnedCaptureDeliveryError {
    pub(super) fn into_parts(self) -> (OwnedCaptureDelivery, CaptureStepError) {
        (self.payload, self.error)
    }
}

pub(super) fn validate_completion(
    frame: &CapturedStep,
    custody: &CaptureTensorCustody,
    step_usage: CaptureUsage,
    cumulative_usage: CaptureUsage,
    capture_seconds: f64,
) -> Result<(), CaptureStepError> {
    custody.validate()?;
    let step = frame
        .records
        .iter()
        .try_fold(CaptureUsage::default(), |sum, record| {
            sum.checked_add(record.charged)
        })?;
    for (index, record) in frame.interventions.iter().enumerate() {
        if !crate::capture::intervention_fits_encoding(record) {
            return Err(CaptureStepError::InterventionState { index });
        }
    }
    let step = frame.interventions.iter().try_fold(step, |sum, record| {
        record
            .evidence
            .iter()
            .try_fold(sum.checked_add(record.charged)?, |sum, evidence| {
                sum.checked_add(evidence.charged)
            })
    })?;
    if !capture_seconds.is_finite()
        || capture_seconds < 0.0
        || step_usage.captures < step.captures
        || step_usage.retained_bytes < step.retained_bytes
        || step_usage.host_bytes < step.host_bytes
        || step_usage.encoded_bytes < step.encoded_bytes
        || cumulative_usage.captures < step_usage.captures
        || cumulative_usage.retained_bytes < step_usage.retained_bytes
        || cumulative_usage.host_bytes < step_usage.host_bytes
        || cumulative_usage.encoded_bytes < step_usage.encoded_bytes
    {
        return Err(CaptureStepError::InvalidCompletion);
    }
    Ok(())
}
fn validate_record_encoding(
    frame: &mut CapturedStep,
    buffers: &mut [RecordBuffers],
    custody: &CaptureTensorCustody,
    index: usize,
) -> Result<(), CaptureStepError> {
    custody.validate()?;
    let record = frame
        .records
        .get(index)
        .ok_or(CaptureStepError::RecordNotPending { index })?;
    match (&record.outcome, &record.payload) {
        (CaptureOutcome::Captured | CaptureOutcome::Truncated { .. },
            Some(CapturePayload::SharedTensor(_) | CapturePayload::Candidates(_)
                | CapturePayload::TokenScores(_) | CapturePayload::Summary(_)
                | CapturePayload::Histogram(_) | CapturePayload::RoutedUnits(_))) => {}
        // A no-overlap window observed its real source but has no selected
        // rectangle or payload. It still requires the exact encoding check.
        (CaptureOutcome::Skipped { reason: CaptureSkipReason::NotInvoked }, None)
            if record.source_shape.is_some() && record.source_dtype.is_some()
                && record.selected_shape.is_none() => {}
        _ => return Err(CaptureStepError::RecordNotPending { index }),
    }
    if crate::capture::record_fits_encoding(record) {
        return Ok(());
    }
    let buffer = &mut buffers[index];
    // Successful insertion never consumes this fixed diagnostic buffer.
    // The constant fits the same 256-byte capacity priced by the plan.
    const MESSAGE: &[u8] = b"backend underestimated encoded capture size";
    debug_assert!(buffer.diagnostic.is_empty());
    debug_assert!(buffer.diagnostic.capacity() >= MESSAGE.len());
    buffer.diagnostic.extend_from_slice(MESSAGE);
    let message =
        String::from_utf8(std::mem::take(&mut buffer.diagnostic)).expect("static UTF-8 diagnostic");
    let record = &mut frame.records[index];
    record.outcome = CaptureOutcome::Failed {
        reason: CaptureFailureReason::Invalid,
        message,
    };
    let retired = record.payload.take();
    // This frame still owns its host custody; shared tensor aliases retain
    // theirs. Neither logical quota nor a spent constructor is refunded.
    drop(retired);
    Err(CaptureStepError::EncodedSize { index })
}

/// Shared owned worker for precommit sealing and later aborted-frame recovery.
/// On error every payload remains with its custody; no source/row borrow is
/// needed. Successful preparation drops unused buffers before allocating the
/// final shared owner, so allocation unwind cannot outlive their host hold.
pub(super) fn prepare_owned_delivery(
    mut payload: OwnedCaptureDelivery,
    step_usage: CaptureUsage,
    cumulative_usage: CaptureUsage,
    capture_seconds: f64,
) -> Result<PreparedCaptureDelivery, OwnedCaptureDeliveryError> {
    let result = (|| {
        validate_completion(
            &payload.frame,
            &payload.custody,
            step_usage,
            cumulative_usage,
            capture_seconds,
        )?;
        for index in 0..payload.frame.records.len() {
            if matches!(
                payload.frame.records[index].outcome,
                CaptureOutcome::Captured | CaptureOutcome::Truncated { .. }
            ) {
                validate_record_encoding(
                    &mut payload.frame,
                    &mut payload.buffers,
                    &payload.custody,
                    index,
                )?;
            }
        }
        Ok(())
    })();
    if let Err(error) = result {
        return Err(OwnedCaptureDeliveryError { error, payload });
    }
    payload.frame.step_usage = step_usage;
    payload.frame.cumulative_usage = cumulative_usage;
    payload.frame.capture_seconds = capture_seconds;
    let OwnedCaptureDelivery {
        frame,
        buffers,
        intervention_buffers,
        prefill,
        partition_metadata,
        custody,
    } = payload;
    drop(prefill);
    drop(intervention_buffers);
    drop(buffers);
    Ok(PreparedCaptureDelivery {
        frame: UnpublishedCapturedStep::retain(frame, (custody, partition_metadata)),
    })
}

impl<'a> PreparedCaptureStep<'a> {
    /// Check one successful record against its already charged capture-wire
    /// bound. Failure preserves shapes/dtype/charges and the spent constructor,
    /// installs the fixed diagnostic before retiring the former tensor, and
    /// never rewrites another terminal outcome.
    pub fn validate_record_encoding(&mut self, index: usize) -> Result<(), CaptureStepError> {
        validate_record_encoding(&mut self.frame, &mut self.buffers, &self.custody, index)
    }

    /// Complete all fallible storage/usage/wire checks and allocate the final
    /// shared owner before the enclosing observation-delivery vote. Successful
    /// return releases the plan borrow; error retains this exact partial frame.
    /// Missing/skip/failure records retain the existing metadata contract.
    pub fn prepare_delivery(
        mut self,
        step_usage: CaptureUsage,
        cumulative_usage: CaptureUsage,
        capture_seconds: f64,
    ) -> Result<PreparedCaptureDelivery, CaptureStepFinishError<'a>> {
        self.flush_intervention_evidence();
        if let Err(error) = self
            .validate_prefill_completion()
            .and_then(|_| self.validate_interventions_complete())
        {
            return Err(CaptureStepFinishError {
                error,
                builder: self,
            });
        }
        let Self {
            frame,
            buffers,
            intervention_buffers,
            intervention_evidence,
            prefill,
            plan,
            partition_metadata,
            custody,
        } = self;
        drop(intervention_evidence);
        let payload = OwnedCaptureDelivery {
            frame,
            buffers,
            intervention_buffers,
            prefill,
            partition_metadata,
            custody,
        };
        prepare_owned_delivery(payload, step_usage, cumulative_usage, capture_seconds).map_err(
            |error| {
                let (
                    OwnedCaptureDelivery {
                        frame,
                        buffers,
                        intervention_buffers,
                        prefill,
                        partition_metadata,
                        custody,
                    },
                    error,
                ) = error.into_parts();
                CaptureStepFinishError {
                    error,
                    builder: Self {
                        frame,
                        buffers,
                        intervention_buffers,
                        intervention_evidence: Vec::new(),
                        prefill,
                        plan,
                        partition_metadata,
                        custody,
                    },
                }
            },
        )
    }
}
