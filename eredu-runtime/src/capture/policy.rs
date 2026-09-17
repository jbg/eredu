//! Shared transaction, selection and logical reservation policy. Storage-specific
//! collectors supply their own closed frame construction and transformation.
use super::*;
pub(super) mod prefill;

/// Allocation-free observer protocol rejection. Legacy callers retain their
/// existing CaptureError diagnostics; the funded collector keeps this typed cause.
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
pub enum CaptureProtocolError {
    /// A same-run restored collector is already spending this cumulative ledger.
    #[error("capture cumulative source is already in use")]
    CumulativeInUse,
    /// An internal operation poisoned the fixed shared cumulative source.
    #[error("capture cumulative source is poisoned")]
    CumulativePoisoned,
    /// Independent invocation geometry requires its existing dedicated driver.
    #[error("independent capture must prepare explicit invocation geometry")]
    Invocation,
    /// An earlier transaction/delivery is still owned or the epoch is stale.
    #[error("capture transaction requires a fresh epoch and a drained step")]
    Undrained,
    /// The actual invocation does not match its admitted authority.
    #[error("capture invocation geometry/authority mismatch")]
    Geometry,
    /// A prior frame remains owned by this session.
    #[error("previous capture step has not been consumed")]
    PreviousStep,
    /// The finite admitted prediction range is exhausted.
    #[error("generation exceeds admitted prediction range")]
    Prediction,
    /// The hook has no matching active transaction.
    #[error("capture hook has no matching pending transaction")]
    Transaction,
    /// The same sequencing rejection with an allocation-free callback stage.
    #[error("capture hook has no matching pending transaction during {0}")]
    TransactionPhase(&'static str),
    /// A selected observation was emitted more than once.
    #[error("selected observation emitted twice")]
    Duplicate,
    /// This closed collector does not price a generated source factory.
    #[error("generated capture source requires separate original-account admission")]
    GeneratedSource,
    /// Selected prefill tensor attribution needs the shared prompt collector.
    #[error("selected prefill tensor capture requires prompt-span attribution")]
    PrefillAttribution,
}
pub(super) fn legacy_error(error: CaptureProtocolError) -> CaptureError {
    CaptureError::Invalid(error.to_string())
}
impl CaptureSession {
    pub(super) fn claim_step_epoch(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        external_pending: bool,
    ) -> Result<(), CaptureProtocolError> {
        if self.plan.invocation_bounds().is_some() {
            return Err(CaptureProtocolError::Invocation);
        }
        self.claim_prepared_step_epoch(epoch, external_pending, None)
    }
    // Only the fixed funded bank supplies this already-validated physical
    // descriptor. The shared transaction ordering remains identical.
    pub(super) fn claim_prepared_step_epoch(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        external_pending: bool,
        invocation: Option<CaptureInvocationShape>,
    ) -> Result<(), CaptureProtocolError> {
        if self.plan.invocation_bounds().is_some() != invocation.is_some() {
            return Err(CaptureProtocolError::Invocation);
        }
        if external_pending
            || self.records.is_some()
            || self.ordinary_frame.is_some()
            || self.transaction.is_some()
            || self
                .last_transaction_epoch
                .is_some_and(|previous| previous >= epoch)
        {
            return Err(CaptureProtocolError::Undrained);
        }
        // Claim before any fallible metadata/frame reservation.
        self.invocation = invocation;
        self.last_transaction_epoch = Some(epoch);
        self.bind_partition_run_epoch(epoch);
        self.transaction = Some((epoch, CaptureTransactionStatus::Pending));
        self.checkpoint_ready = false;
        Ok(())
    }
    pub(super) fn validate_step_start(&self, prediction: u64) -> Result<(), CaptureProtocolError> {
        if self.plan.invocation_bounds().is_some() != self.invocation.is_some() {
            return Err(CaptureProtocolError::Geometry);
        }
        if self.records.is_some() || self.ordinary_frame.is_some() {
            return Err(CaptureProtocolError::PreviousStep);
        }
        if prediction >= self.plan.request().max_predictions {
            return Err(CaptureProtocolError::Prediction);
        }
        Ok(())
    }
    pub(super) fn reset_step_ledger(&mut self) -> Result<(), CaptureError> {
        self.checkpoint_ready = false;
        self.has_step = true;
        self.ledger.begin_step();
        if self.invocation.is_some() {
            let usage = invocation::INVOCATION_METADATA.checked_mul(
                self.partition
                    .as_ref()
                    .map_or(1, |run| run.world_size() as u64),
            )?;
            reserve_required(&mut self.ledger, usage)?;
        }
        if let Some(partition) = &mut self.partition {
            partition.begin_step();
        }
        Ok(())
    }
}
pub(super) fn reserve_required(
    ledger: &mut CaptureLedger,
    usage: CaptureUsage,
) -> Result<(), CaptureError> {
    if let Some(CaptureSkipReason::Limit { budget, cumulative }) = ledger.reserve(usage)? {
        // A skipped metadata envelope would itself be unaccounted output.
        return Err(CaptureError::Limit { budget, cumulative });
    }
    Ok(())
}
pub(super) fn reserve_metadata(
    ledger: &mut CaptureLedger,
    selection: &CaptureSelection,
    point: &eredu_core::ObservationPoint,
    world_size: u64,
) -> Result<CaptureUsage, CaptureError> {
    let charged = metadata_reservation(selection, point)?;
    reserve_required(ledger, charged.checked_mul(world_size)?)?;
    Ok(charged)
}
pub(super) fn selected_record(
    selection: &CaptureSelection,
    record: &CaptureRecord,
    path: &str,
) -> Result<bool, CaptureProtocolError> {
    select_status(selection, CaptureRecordStatus::from_record(record), path)
}
pub(super) fn failure_reason(error: &CaptureError) -> CaptureFailureReason {
    match error.cause() {
        CaptureError::Limit { budget, cumulative } => CaptureFailureReason::Limit {
            budget: *budget,
            cumulative: *cumulative,
        },
        CaptureError::Unsupported(_) => CaptureFailureReason::Unsupported,
        _ => CaptureFailureReason::Invalid,
    }
}

/// Semantic record state shared by real delivery and cold workspace inspection.
/// This contains no payload, quota credit, receipt or construction permission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureRecordStatus {
    /// A scheduled selection has not emitted a value.
    Missing,
    /// Schedule or logical limit skipped the selection.
    Skipped,
    /// A value or failed attempt already consumed this selection.
    Consumed,
}
impl CaptureRecordStatus {
    /// Read only the existing delivery's semantic status.
    pub fn from_record(record: &CaptureRecord) -> Self {
        match record.outcome {
            CaptureOutcome::Missing => Self::Missing,
            CaptureOutcome::Skipped { .. } => Self::Skipped,
            _ => Self::Consumed,
        }
    }
}
fn select_status(
    selection: &CaptureSelection,
    status: CaptureRecordStatus,
    path: &str,
) -> Result<bool, CaptureProtocolError> {
    if selection.path != path || status == CaptureRecordStatus::Skipped {
        return Ok(false);
    }
    if status != CaptureRecordStatus::Missing {
        return Err(CaptureProtocolError::Duplicate);
    }
    Ok(true)
}

/// Borrowed ordinary prediction policy, independent of storage/transactions.
/// Real collectors retain their original ledger and frame; diagnostics use a
/// fresh ledger. No method claims an epoch, allocates a frame or funds work.
#[derive(Clone, Copy)]
pub struct CaptureObservationStep<'a> {
    source: &'a AdmittedCapturePlan,
    phase: CapturePhase,
    prediction: u64,
    invocation: Option<CaptureInvocationShape>,
    window: Option<CaptureInvocationWindow>,
}
impl<'a> CaptureObservationStep<'a> {
    /// Validate one ordinary coordinate without changing a ledger or session.
    pub fn new(
        source: &'a AdmittedCapturePlan,
        phase: CapturePhase,
        prediction: u64,
    ) -> Result<Self, CaptureProtocolError> {
        if source.text_origin().is_none() {
            return Err(CaptureProtocolError::Invocation);
        }
        if prediction >= source.request().max_predictions {
            return Err(CaptureProtocolError::Prediction);
        }
        if (prediction == 0) != (phase == CapturePhase::Prefill) {
            return Err(CaptureProtocolError::Geometry);
        }
        Ok(Self {
            source,
            phase,
            prediction,
            invocation: None,
            window: None,
        })
    }
    /// Describe the same observation policy at one actual independently shaped
    /// invocation. This carries no claim, source, native or quota authority.
    pub fn with_invocation(
        source: &'a AdmittedCapturePlan,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
    ) -> Result<Self, CaptureProtocolError> {
        let Some(shape) = invocation else {
            return Self::new(source, phase, prediction);
        };
        let bounds = source
            .invocation_bounds()
            .ok_or(CaptureProtocolError::Invocation)?;
        bounds
            .validate_fixed(shape, prediction)
            .map_err(|_| CaptureProtocolError::Geometry)?;
        Ok(Self {
            source,
            phase,
            prediction,
            invocation,
            window: None,
        })
    }
    /// Bind globally anchored slices to this same admitted physical invocation.
    /// Geometry neither spends a claim nor admits source/native construction.
    pub fn with_window(
        mut self,
        window: Option<CaptureInvocationWindow>,
    ) -> Result<Self, CaptureProtocolError> {
        if let Some(window) = window {
            let physical = self.invocation.ok_or(CaptureProtocolError::Invocation)?;
            let logical = window
                .validate(physical)
                .map_err(|_| CaptureProtocolError::Geometry)?;
            self.source
                .invocation_bounds()
                .ok_or(CaptureProtocolError::Invocation)?
                .validate_fixed(logical, self.prediction)
                .map_err(|_| CaptureProtocolError::Geometry)?;
        }
        self.window = window;
        Ok(self)
    }
    /// Exact physical window, with no claim or source authority.
    pub fn window(&self) -> Option<CaptureInvocationWindow> {
        self.window
    }
    /// Existing ordinary fragment metadata policy, separate from physical H.
    pub fn window_metadata_usage(
        &self,
        index: usize,
    ) -> Result<Option<CaptureUsage>, CaptureError> {
        self.window
            .map(|_| {
                let selection = self
                    .source
                    .plan()
                    .selections
                    .get(index)
                    .ok_or(CaptureError::Overflow)?;
                let point = self
                    .source
                    .points()
                    .get(index)
                    .ok_or(CaptureError::Overflow)?;
                super::partition::fragment_metadata_usage(
                    selection,
                    point,
                    point.axes.as_ref().map_or(32, Vec::len),
                )
            })
            .transpose()
    }
    /// Selected prefill hooks require full new-prompt rows. Decode already has
    /// one row; metadata-only phases do not change the ordinary output demand.
    /// Available before observer preparation, from the already-known prediction.
    pub fn requires_sequence_readout(&self) -> bool {
        self.sequence_readout(None)
    }
    /// Same demand for the exact physical phase applicability mask. Excluded
    /// scopes do not request readout work, even when their logical schedule fires.
    pub fn requires_sequence_readout_for(
        &self,
        selected: &[bool],
    ) -> Result<bool, CaptureProtocolError> {
        if selected.len() != self.source.points().len() {
            return Err(CaptureProtocolError::Geometry);
        }
        Ok(self.sequence_readout(Some(selected)))
    }
    fn sequence_readout(&self, selected: Option<&[bool]>) -> bool {
        (self.phase == CapturePhase::Prefill
            || self.invocation.is_some_and(|shape| shape.sequence > 1))
            && self
                .source
                .plan()
                .selections
                .iter()
                .zip(self.source.points())
                .enumerate()
                .any(|(index, (selection, point))| {
                    selected.is_none_or(|mask| mask[index])
                        && (match self.phase {
                            CapturePhase::Prefill => point.prefill,
                            CapturePhase::Decode => point.decode,
                        })
                        && selection.schedule.includes(self.phase, self.prediction)
                        && !matches!(
                            selection.transform,
                            CaptureTransform::TopCandidates { .. }
                                | CaptureTransform::TokenScores { .. }
                        )
                })
    }
    /// Same original p0 scheduling predicate used before source bootstrap.
    /// Empty, decode-only and schedule-skipped plans issue no source segment.
    /// This says nothing about quota acceptance or physical readout demand.
    pub fn has_selected_prefill_hook(&self) -> bool {
        self.phase == CapturePhase::Prefill
            && self
                .source
                .plan()
                .selections
                .iter()
                .zip(self.source.points())
                .any(|(selection, point)| {
                    point.prefill && selection.schedule.includes(self.phase, self.prediction)
                })
    }
    /// Initial delivery status; scheduled but unavailable hooks remain Missing,
    /// matching the ordinary frame protocol rather than silently skipping them.
    pub fn initial_status(
        &self,
        index: usize,
    ) -> Result<CaptureRecordStatus, CaptureProtocolError> {
        let selection = self
            .source
            .plan()
            .selections
            .get(index)
            .ok_or(CaptureProtocolError::Geometry)?;
        Ok(
            if selection.schedule.includes(self.phase, self.prediction) {
                CaptureRecordStatus::Missing
            } else {
                CaptureRecordStatus::Skipped
            },
        )
    }
    /// The exact existing path/duplicate/skip decision without a record DTO.
    pub fn select(
        &self,
        index: usize,
        status: CaptureRecordStatus,
        path: &str,
    ) -> Result<bool, CaptureProtocolError> {
        let selection = self
            .source
            .plan()
            .selections
            .get(index)
            .ok_or(CaptureProtocolError::Geometry)?;
        select_status(selection, status, path)
    }
    /// Reserve the same independent-invocation envelope as the live session.
    /// Cold callers begin their own ledger step; funded reset already calls the
    /// existing world-size-aware worker and must not reserve this a second time.
    pub fn reserve_invocation_metadata(
        &self,
        ledger: &mut CaptureLedger,
    ) -> Result<(), CaptureError> {
        if self.invocation.is_some() {
            reserve_required(ledger, super::invocation::INVOCATION_METADATA)?;
        }
        Ok(())
    }
    /// Reserve all required record envelopes after the caller begins its one
    /// logical step. A skipped metadata envelope is still a typed limit error.
    pub fn reserve_metadata(&self, ledger: &mut CaptureLedger) -> Result<(), CaptureError> {
        for (selection, point) in self
            .source
            .plan()
            .selections
            .iter()
            .zip(self.source.points())
        {
            reserve_metadata(ledger, selection, point, 1)?;
        }
        Ok(())
    }
    /// Derive the same fixed geometry as the funded tensor worker.
    pub fn tensor_geometry(
        &self,
        index: usize,
    ) -> Result<CaptureTensorGeometry<'a>, CaptureTensorGeometryError> {
        match self.window {
            Some(window) => CaptureTensorGeometry::prepare_window(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation
                    .ok_or(CaptureTensorGeometryError::Unsupported)?,
                window,
            ),
            None => CaptureTensorGeometry::prepare(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation,
            ),
        }
    }
    /// Exact routed token/slot/unit coordinates for this ordinary invocation.
    pub fn routed_geometry(&self, index: usize)
        -> Result<CaptureRoutedUnitsGeometry<'a>, CaptureTensorGeometryError> {
        match self.window {
            Some(window) => CaptureRoutedUnitsGeometry::prepare_window(
                self.source, index, self.phase, self.prediction,
                self.invocation.ok_or(CaptureTensorGeometryError::Unsupported)?, window),
            None => CaptureRoutedUnitsGeometry::prepare(
                self.source, index, self.phase, self.prediction, self.invocation),
        }
    }
    /// Same typed finite-only Summary selection at the actual physical window.
    pub fn summary_geometry(
        &self,
        index: usize,
    ) -> Result<CaptureSummaryGeometry<'a>, CaptureTensorGeometryError> {
        match self.window {
            Some(window) => CaptureSummaryGeometry::prepare_window(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation
                    .ok_or(CaptureTensorGeometryError::Unsupported)?,
                window,
            ),
            None => CaptureSummaryGeometry::prepare(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation,
            ),
        }
    }
    /// Same typed Histogram selection; it cannot become a raw tensor claim.
    pub fn histogram_geometry(
        &self,
        index: usize,
    ) -> Result<CaptureHistogramGeometry<'a>, CaptureTensorGeometryError> {
        match self.window {
            Some(window) => CaptureHistogramGeometry::prepare_window(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation
                    .ok_or(CaptureTensorGeometryError::Unsupported)?,
                window,
            ),
            None => CaptureHistogramGeometry::prepare(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation,
            ),
        }
    }
    /// Validate the concrete generated program against this admitted source
    /// geometry and declared logical metadata. This is not source custody,
    /// numerical admission or permission to invoke an arbitrary factory.
    pub fn validate_generated(
        &self,
        index: usize,
        source: &GeneratedCaptureSource,
        program: eredu_nn::GeneratedTensorProgram<'_>,
    ) -> Result<(), CaptureError> {
        let geometry = self.tensor_geometry(index).map_err(|_| {
            CaptureError::Invalid("generated observation geometry is unavailable".into())
        })?;
        let eredu_nn::GeneratedTensorProgram::BlockFp8Input(plan) = program;
        if plan.shape().len() != geometry.source_shape().len()
            || plan
                .shape()
                .iter()
                .zip(geometry.source_shape())
                .any(|(&a, &b)| usize::try_from(a).ok() != Some(b))
        {
            return Err(CaptureError::Invalid(
                "generated observation changed its declared geometry".into(),
            ));
        }
        let actual = plan
            .logical_capture_source()
            .map_err(|_| CaptureError::Overflow)?;
        if source.source_dtype != Some(eredu_core::checkpoint::TensorDtype::F32)
            || source.creation_bytes != actual.creation_bytes
        {
            return Err(CaptureError::Invalid(
                "generated observation source differs from its selected program".into(),
            ));
        }
        Ok(())
    }
    /// Add the unchanged legacy creation charge for every accepted selection,
    /// even when one generated value is reused within the hook.
    pub fn generated_usage(
        &self,
        usage: CaptureUsage,
        source: &GeneratedCaptureSource,
    ) -> Result<CaptureUsage, CaptureError> {
        generated::generated_usage(usage, source)
    }
    /// Whether the pre-transform slice is empty. Preview(max_elements=0) alone
    /// does not suppress a nonempty legacy generated factory.
    pub fn generated_slice_empty(&self, index: usize) -> Result<bool, CaptureError> {
        let geometry = self.tensor_geometry(index).map_err(|_| {
            CaptureError::Invalid("generated observation geometry is unavailable".into())
        })?;
        Ok(geometry.source_shape().contains(&0)
            || self.source.plan().selections[index]
                .slices
                .iter()
                .any(|slice| slice.start == slice.end))
    }

    /// Reuse the existing monotone per-step/cumulative ledger and Skip/Fail rule.
    pub fn reserve_value(
        &self,
        ledger: &mut dyn eredu_core::capture::CaptureReservation,
        usage: CaptureUsage,
    ) -> Result<Option<CaptureSkipReason>, CaptureError> {
        ledger.reserve(usage)
    }
}

#[cfg(test)]
mod tests;
