use super::*;
mod partition;
mod pending;
pub use pending::{PendingCaptureDelivery, PendingCaptureDeliveryError};

mod delivery;
pub use delivery::PreparedCaptureDelivery;
pub(super) const DELIVERY_CONTROL_BYTES: usize = 3
    * (size_of::<PreparedCaptureDelivery>()
        + size_of::<delivery::OwnedCaptureDelivery>()
        + size_of::<delivery::OwnedCaptureDeliveryError>());

pub(super) fn fixed_string(text: &str) -> String {
    let mut bytes = Vec::with_capacity(text.len());
    for &byte in text.as_bytes() {
        bytes.push(byte);
    }
    // UTF-8 validation transfers the exact existing Vec; it does not copy or
    // allocate. The input is already valid UTF-8 and no arbitrary writer runs.
    String::from_utf8(bytes).expect("copied valid UTF-8")
}
pub(in crate::working_memory) fn allocate<'a>(
    plan: CaptureStepHostPlan<'a>,
    custody: CaptureTensorCustody,
) -> Result<PreparedCaptureStep<'a>, CaptureStepError> {
    custody.validate()?;
    #[cfg(test)]
    tests::before_allocate();
    let mut records = Vec::with_capacity(plan.len());
    let mut buffers = Vec::with_capacity(plan.len());
    for (index, (selection, point)) in plan
        .source
        .plan()
        .selections
        .iter()
        .zip(plan.source.points())
        .enumerate()
    {
        let geometry = plan.geometry(index)?;
        let candidates = plan.candidate_geometry(index)?;
        let scores = plan.token_score_geometry(index)?;
        let summary = plan.summary_geometry(index)?;
        let histogram = plan.histogram_geometry(index)?;
        let routed = plan.routed_geometry(index)?;
        let rank = geometry.as_ref().map_or(
            if candidates.is_some() || scores.is_some() {
                3
            } else if let Some(summary) = &summary {
                summary.source_shape().len()
            } else if let Some(routed) = &routed {
                routed.source_shape().len()
            } else if let Some(histogram) = &histogram {
                histogram.source_shape().len()
            } else {
                0
            },
            |g| g.source_shape().len(),
        );
        let mut source = Vec::with_capacity(rank);
        let mut selected = Vec::with_capacity(rank);
        if let Some(geometry) = &geometry {
            for (axis, &dimension) in geometry.source_shape().iter().enumerate() {
                source.push(dimension as u64);
                selected.push(plan::selected_extent(plan.source, index, geometry, axis));
            }
        }
        if let Some(candidates) = &candidates {
            source.extend(candidates.source_shape().iter().map(|&n| n as u64));
            selected.extend(candidates.source_shape().iter().map(|&n| n as u64));
        }
        if let Some(scores) = &scores {
            source.extend(scores.source_shape().iter().map(|&n| n as u64));
            selected.extend(scores.source_shape().iter().map(|&n| n as u64));
        }
        if let Some(summary) = &summary {
            source.extend(summary.source_shape().iter().map(|&n| n as u64));
            selected.extend(summary.shape().iter().map(|&n| n as u64));
        }
        if let Some(routed) = &routed {
            source.extend(routed.source_shape().iter().map(|&n| n as u64));
            selected.extend(routed.shape().iter().map(|&n| n as u64));
        }
        if let Some(histogram) = &histogram {
            source.extend(histogram.source_shape().iter().map(|&n| n as u64));
            selected.extend(histogram.shape().iter().map(|&n| n as u64));
        }
        let diagnostic = Vec::with_capacity(
            if geometry.is_some()
                || candidates.is_some()
                || scores.is_some()
                || summary.is_some()
                || histogram.is_some()
                || routed.is_some()
            {
                plan::DIAGNOSTIC_BYTES
            } else {
                0
            },
        );
        buffers.push(RecordBuffers {
            routed: None,
            source,
            selected,
            diagnostic,
        });
        records.push(CaptureRecord {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selection_id: fixed_string(&selection.id),
            path: fixed_string(&selection.path),
            node_id: fixed_string(&point.node_id),
            position: point.position,
            source_shape: None,
            source_dtype: None,
            selected_shape: None,
            outcome: if !selection.schedule.includes(plan.phase, plan.prediction) {
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Schedule,
                }
            } else if !plan.selected(index) {
                CaptureOutcome::Skipped {
                    reason: plan
                        .skipped
                        .and_then(|rows| rows.reason(index))
                        .unwrap_or(CaptureSkipReason::NotInvoked),
                }
            } else {
                CaptureOutcome::Missing
            },
            payload: None,
            charged: crate::capture::metadata_reservation(selection, point)?,
        });
        #[cfg(test)]
        tests::after_record(index);
    }
    let frame = CapturedStep {
        outcome: CaptureStepOutcome::Untracked,
        phase: plan.phase,
        invocation: plan.invocation,
        prediction_index: plan.prediction,
        records,
        partitions: Vec::new(),
        interventions: Vec::new(),
        step_usage: CaptureUsage::default(),
        cumulative_usage: CaptureUsage::default(),
        capture_seconds: 0.0,
    };
    // A concurrently fenced/closed parent cannot publish a freshly allocated
    // builder. All local frame/sidecar payload retires before custody on error.
    custody.validate()?;
    Ok(PreparedCaptureStep {
        frame,
        buffers,
        intervention_buffers: Vec::new(),
        intervention_evidence: Vec::new(),
        prefill: None,
        partition_metadata: None,
        plan,
        custody,
    })
}

/// Fixed record buffers with indexed mutation and no raw owning export.
///
/// Frame and unused sidecars precede the final private host custody. Each method
/// rechecks the original parent is open and healthy. This owner cannot certify a
/// native scope, refund capture quota, or establish actual tensor provenance.
#[must_use = "finish this frame or retire its protected partial buffers"]
pub struct PreparedCaptureStep<'a> {
    pub(super) frame: CapturedStep,
    pub(super) buffers: Vec<RecordBuffers>,
    pub(super) intervention_buffers: Vec<Vec<u8>>,
    pub(in crate::working_memory) intervention_evidence:
        Vec<Option<super::interventions::evidence::PreparedInterventionEvidence<'a>>>,
    pub(in crate::working_memory) prefill:
        Option<crate::working_memory::capture_tensor::prefill::PrefillTargets>,
    pub(in crate::working_memory) partition_metadata:
        Option<eredu_nn::workspace::HostMetadataFunding>,
    pub(super) plan: CaptureStepHostPlan<'a>,
    pub(super) custody: CaptureTensorCustody,
}
impl PreparedCaptureStep<'_> {
    /// Borrows records without transferring frame storage/custody. Caller DTO
    /// clones are separate allocations and do not inherit frame custody.
    pub fn records(&self) -> &[CaptureRecord] {
        &self.frame.records
    }
    pub(in crate::working_memory) fn routed_target(
        &self,
        index: usize,
    ) -> Option<&crate::working_memory::capture_run::RoutedInvocationTarget> {
        self.buffers
            .get(index)
            .and_then(|buffer| buffer.routed.as_ref())
    }
    pub(in crate::working_memory) fn routed_target_mut(
        &mut self,
        index: usize,
    ) -> Result<
        &mut Option<crate::working_memory::capture_run::RoutedInvocationTarget>,
        CaptureStepError,
    > {
        self.custody.validate()?;
        self.buffers
            .get_mut(index)
            .map(|buffer| &mut buffer.routed)
            .ok_or(CaptureStepError::RecordNotPending { index })
    }
    /// Fixed selection count, including skips and missing records.
    pub fn len(&self) -> usize {
        self.frame.records.len()
    }
    /// Whether this frame has no records.
    pub fn is_empty(&self) -> bool {
        self.frame.records.is_empty()
    }
    /// Original construction hold; numerical tensor custody remains separate.
    pub fn protected_bytes(&self) -> u64 {
        self.plan.initialization_peak_bytes()
    }
    fn pending_charge(
        &self,
        index: usize,
        additional: CaptureUsage,
    ) -> Result<CaptureUsage, CaptureStepError> {
        self.custody.validate()?;
        let record = self
            .frame
            .records
            .get(index)
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        if !matches!(record.outcome, CaptureOutcome::Missing) || !self.plan.active(index)? {
            return Err(CaptureStepError::RecordNotPending { index });
        }
        Ok(record.charged.checked_add(additional)?)
    }
    pub(in crate::working_memory) fn record_candidates(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        shape: [usize; 3],
        candidates: CaptureCandidates,
        additional: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        let charged = self.pending_charge(index, additional)?;
        self.dtype(index, &dtype)?;
        let geometry = self
            .plan
            .candidate_geometry(index)?
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        if shape[0] != 1
            || shape[1] == 0
            || shape[1] > geometry.source_shape()[1]
            || shape[2] != geometry.vocabulary()
            || candidates.candidates.len() != geometry.count()
        {
            return Err(CaptureStepError::TensorMismatch { index });
        }
        // Buffers were already allocated at original frame construction. Preserve
        // actual physical readout geometry rather than substituting prompt width.
        for (axis, &extent) in shape.iter().enumerate() {
            self.buffers[index].source[axis] = extent as u64;
            self.buffers[index].selected[axis] = extent as u64;
        }
        self.observed(index, Some(dtype));
        let record = &mut self.frame.records[index];
        record.payload = Some(CapturePayload::Candidates(candidates));
        record.outcome = CaptureOutcome::Captured;
        record.charged = charged;
        Ok(())
    }
    pub(in crate::working_memory) fn record_token_scores(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        shape: [usize; 3],
        scores: CaptureTokenScores,
        additional: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        let charged = self.pending_charge(index, additional)?;
        self.dtype(index, &dtype)?;
        let geometry = self
            .plan
            .token_score_geometry(index)?
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        if shape[0] != 1
            || shape[1] == 0
            || shape[1] > geometry.source_shape()[1]
            || shape[2] != geometry.vocabulary()
            || scores.scores.len() != geometry.count()
            || scores.vocabulary != geometry.vocabulary() as u64
        {
            return Err(CaptureStepError::TensorMismatch { index });
        }
        // Buffers were already allocated at original frame construction. Preserve
        // actual physical readout geometry rather than substituting prompt width.
        for (axis, &extent) in shape.iter().enumerate() {
            self.buffers[index].source[axis] = extent as u64;
            self.buffers[index].selected[axis] = extent as u64;
        }
        self.observed(index, Some(dtype));
        let record = &mut self.frame.records[index];
        record.payload = Some(CapturePayload::TokenScores(scores));
        record.outcome = CaptureOutcome::Captured;
        record.charged = charged;
        Ok(())
    }
    pub(in crate::working_memory) fn record_summary(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        value: CaptureSummary,
        additional: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        let charged = self.pending_charge(index, additional)?;
        self.dtype(index, &dtype)?;
        let geometry = self
            .plan
            .summary_geometry(index)?
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        crate::capture::reduction::Summary::default()
            .appended_fixed(&value, geometry.elements() as u64)?;
        self.observed(index, Some(dtype));
        let record = &mut self.frame.records[index];
        record.payload = Some(CapturePayload::Summary(value));
        record.outcome = CaptureOutcome::Captured;
        record.charged = charged;
        Ok(())
    }
    pub(in crate::working_memory) fn record_histogram(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        value: CaptureHistogram,
        additional: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        let charged = self.pending_charge(index, additional)?;
        self.dtype(index, &dtype)?;
        let geometry = self
            .plan
            .histogram_geometry(index)?
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        crate::capture::reduction::validate_histogram_fixed(
            &value,
            geometry.edges(),
            geometry.elements() as u64,
        )?;
        self.observed(index, Some(dtype));
        let record = &mut self.frame.records[index];
        record.payload = Some(CapturePayload::Histogram(value));
        record.outcome = CaptureOutcome::Captured;
        record.charged = charged;
        Ok(())
    }
    fn pending(
        &self,
        index: usize,
        additional: CaptureUsage,
    ) -> Result<(CaptureTensorGeometry<'_>, CaptureUsage), CaptureStepError> {
        self.custody.validate()?;
        let record = self
            .frame
            .records
            .get(index)
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        if !matches!(record.outcome, CaptureOutcome::Missing) {
            return Err(CaptureStepError::RecordNotPending { index });
        }
        let geometry = self
            .plan
            .geometry(index)?
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        Ok((geometry, record.charged.checked_add(additional)?))
    }
    fn dtype(&self, index: usize, dtype: &TensorDtype) -> Result<(), CaptureStepError> {
        let semantic = self
            .plan
            .source
            .points()
            .get(index)
            .map(|point| point.dtype);
        let supported = match semantic {
            Some(ObservationDtype::Floating) => matches!(
                dtype,
                TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16
            ),
            Some(ObservationDtype::Integer) => matches!(
                dtype,
                TensorDtype::U8 | TensorDtype::U16 | TensorDtype::U32 | TensorDtype::U64
            ),
            _ => false,
        };
        if !supported {
            return Err(CaptureStepError::TensorMismatch { index });
        }
        Ok(())
    }
    fn observed(&mut self, index: usize, dtype: Option<TensorDtype>) {
        if let Some(dtype) = dtype {
            let buffer = &mut self.buffers[index];
            self.frame.records[index].source_shape = Some(std::mem::take(&mut buffer.source));
            self.frame.records[index].selected_shape = Some(std::mem::take(&mut buffer.selected));
            self.frame.records[index].source_dtype = Some(dtype);
        }
    }
    pub(in crate::working_memory) fn charge_prefill_target(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        additional: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        let charged = self.pending_charge(index, additional)?;
        self.dtype(index, &dtype)?;
        // Shape sidecars remain unused until final success/failure; no duplicate move.
        self.frame.records[index].charged = charged;
        Ok(())
    }
    pub(in crate::working_memory) fn validate_prefill_completion(
        &self,
    ) -> Result<(), CaptureStepError> {
        if self.prefill.as_ref().is_some_and(|p| !p.complete()) {
            return Err(CaptureStepError::InvalidCompletion);
        }
        Ok(())
    }
    /// Attach an existing protected typed tensor without copying its values.
    ///
    /// Exact output shape and declared native scalar category are checked. SharedTensor
    /// alone cannot authenticate which source/selection/invocation produced it;
    /// that is the enclosing closed backend worker's semantic obligation. It must
    /// also supply genuinely charged logical usage and separately fund/settle
    /// native work. Matching values or geometry does not create that authority.
    pub fn record_tensor(
        &mut self,
        index: usize,
        source_dtype: TensorDtype,
        tensor: SharedTensorObservation,
        additional: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        let (geometry, charged) = self.pending(index, additional)?;
        self.dtype(index, &source_dtype)?;
        if tensor.shape() != geometry.shape()
            || !matches!(
                (geometry.value_dtype(), tensor.data()),
                (ObservationDtype::Floating, TensorObservationData::F32(_))
                    | (ObservationDtype::Integer, TensorObservationData::U64(_))
            )
        {
            return Err(CaptureStepError::TensorMismatch { index });
        }
        let available = plan::selected_elements(self.plan.source, index, &geometry)?;
        let outcome = if let CaptureTransform::Preview { max_elements } =
            self.plan.source.plan().selections[index].transform
        {
            if available > max_elements {
                CaptureOutcome::Truncated {
                    available_elements: available,
                    emitted_elements: max_elements,
                }
            } else {
                CaptureOutcome::Captured
            }
        } else {
            CaptureOutcome::Captured
        };
        self.observed(index, Some(source_dtype));
        let record = &mut self.frame.records[index];
        record.payload = Some(CapturePayload::SharedTensor(tensor));
        record.outcome = outcome;
        record.charged = charged;
        Ok(())
    }
    /// Record a terminal failure in the preallocated 256-byte diagnostic buffer.
    /// Input text is borrowed; UTF-8 truncation never invokes Display/callbacks or
    /// grows storage. A known source dtype also installs the exact shape buffers.
    pub fn record_failure(
        &mut self,
        index: usize,
        reason: CaptureFailureReason,
        diagnostic: &str,
        source_dtype: Option<TensorDtype>,
        additional: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        let charged = self.pending_charge(index, additional)?;
        if let Some(dtype) = &source_dtype {
            self.dtype(index, dtype)?;
        }
        let mut end = diagnostic.len().min(plan::DIAGNOSTIC_BYTES);
        while !diagnostic.is_char_boundary(end) {
            end -= 1;
        }
        let buffer = &mut self.buffers[index];
        for &byte in diagnostic[..end].as_bytes() {
            buffer.diagnostic.push(byte);
        }
        let message =
            String::from_utf8(std::mem::take(&mut buffer.diagnostic)).expect("bounded valid UTF-8");
        self.observed(index, source_dtype);
        let record = &mut self.frame.records[index];
        record.outcome = CaptureOutcome::Failed { reason, message };
        record.charged = charged;
        Ok(())
    }
    // A no-overlap physical window observed the actual source but produced no
    // selected rectangle. Move only the already-paid source sidecar, matching
    // the ordinary partition collector; no empty payload/selected shape escapes.
    pub(in crate::working_memory) fn record_window_empty(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        additional: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        let charged = self.pending_charge(index, additional)?;
        self.dtype(index, &dtype)?;
        if self.plan.window.is_none() || !self.buffers[index].selected.contains(&0) {
            return Err(CaptureStepError::InvalidCompletion);
        }
        let record = &mut self.frame.records[index];
        record.source_shape = Some(std::mem::take(&mut self.buffers[index].source));
        record.source_dtype = Some(dtype);
        record.outcome = CaptureOutcome::Skipped {
            reason: CaptureSkipReason::NotInvoked,
        };
        record.charged = charged;
        Ok(())
    }
    /// Record an explicit skip. This reports an already made logical policy
    /// decision; it cannot authorize/refund either host or capture accounting.
    pub fn record_skip(
        &mut self,
        index: usize,
        reason: CaptureSkipReason,
        source_dtype: Option<TensorDtype>,
        additional: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        let charged = self.pending_charge(index, additional)?;
        if let Some(dtype) = &source_dtype {
            self.dtype(index, dtype)?;
        }
        self.observed(index, source_dtype);
        let record = &mut self.frame.records[index];
        record.outcome = CaptureOutcome::Skipped { reason };
        record.charged = charged;
        Ok(())
    }
}
impl<'a> PreparedCaptureStep<'a> {
    fn validate_completion(
        &self,
        step_usage: CaptureUsage,
        cumulative_usage: CaptureUsage,
        capture_seconds: f64,
    ) -> Result<(), CaptureStepError> {
        self.validate_prefill_completion()?;
        delivery::validate_completion(
            &self.frame,
            &self.custody,
            step_usage,
            cumulative_usage,
            capture_seconds,
        )
    }

    /// Move existing frame buffers into shared delivery. Missing records remain
    /// explicit. Transaction outcome and cumulative usage are observations from
    /// the enclosing driver, not permissions. Step usage must cover the record
    /// sum, including any additional invocation/failed-attempt ledger charges;
    /// cumulative usage must cover step usage. No custody or payload is detached on error.
    pub fn finish(
        mut self,
        outcome: CaptureStepOutcome,
        step_usage: CaptureUsage,
        cumulative_usage: CaptureUsage,
        capture_seconds: f64,
    ) -> Result<SharedCapturedStep, CaptureStepFinishError<'a>> {
        self.flush_intervention_evidence();
        let result = self
            .validate_completion(step_usage, cumulative_usage, capture_seconds)
            .and_then(|_| {
                if outcome == CaptureStepOutcome::Aborted {
                    Ok(())
                } else {
                    self.validate_interventions_complete()
                }
            });
        if let Err(error) = result {
            return Err(CaptureStepFinishError {
                builder: self,
                error,
            });
        }
        self.frame.outcome = outcome;
        self.frame.step_usage = step_usage;
        self.frame.cumulative_usage = cumulative_usage;
        self.frame.capture_seconds = capture_seconds;
        let Self {
            frame,
            buffers,
            intervention_buffers,
            intervention_evidence,
            prefill,
            plan: _,
            partition_metadata,
            custody,
        } = self;
        drop(prefill);
        drop(intervention_evidence);
        drop(intervention_buffers);
        drop(buffers); // Unused payload retires before the shared custody move.
        Ok(SharedCapturedStep::retain(
            frame,
            (custody, partition_metadata),
        ))
    }
}
impl fmt::Debug for PreparedCaptureStep<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedCaptureStep")
            .field("records", &self.len())
            .field("protected_bytes", &self.protected_bytes())
            .finish_non_exhaustive()
    }
}
/// Failed completion retains the entire partial frame before its host custody.
pub struct CaptureStepFinishError<'a> {
    error: CaptureStepError,
    builder: PreparedCaptureStep<'a>,
}
impl<'a> CaptureStepFinishError<'a> {
    /// Borrows the typed rejection without detaching payload or custody.
    pub fn error(&self) -> &CaptureStepError {
        &self.error
    }
    pub(crate) fn into_parts(self) -> (PreparedCaptureStep<'a>, CaptureStepError) {
        (self.builder, self.error)
    }
    /// Recover the same fixed owner for cleanup or a corrected completion.
    pub fn into_builder(self) -> PreparedCaptureStep<'a> {
        self.builder
    }
}
impl fmt::Debug for CaptureStepFinishError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CaptureStepFinishError")
            .field("error", &self.error)
            .field("builder", &self.builder)
            .finish()
    }
}
impl fmt::Display for CaptureStepFinishError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl std::error::Error for CaptureStepFinishError<'_> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl PreparedCaptureStep<'_> {
    pub(in crate::working_memory) fn record_routed_units(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        value: RoutedUnitCapture,
        slice: &ResolvedCaptureSlice,
        scratch: &mut [RoutedUnitRowIdentity],
        additional: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        self.record_routed_kind(index, dtype, value, slice, scratch, additional, false)
    }
    pub(in crate::working_memory) fn record_partition_routed_units(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        value: RoutedUnitCapture,
        slice: &ResolvedCaptureSlice,
        scratch: &mut [RoutedUnitRowIdentity],
        additional: CaptureUsage,
    ) -> Result<(), CaptureStepError> {
        self.record_routed_kind(index, dtype, value, slice, scratch, additional, true)
    }
    fn record_routed_kind(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        mut value: RoutedUnitCapture,
        slice: &ResolvedCaptureSlice,
        scratch: &mut [RoutedUnitRowIdentity],
        additional: CaptureUsage,
        partition: bool,
    ) -> Result<(), CaptureStepError> {
        let charged = self.pending_charge(index, additional)?;
        if !partition || dtype != TensorDtype::F64 {
            self.dtype(index, &dtype)?;
        }
        let geometry = self
            .plan
            .routed_geometry(index)?
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        if value.geometry != geometry.bank()
            || slice.starts != geometry.starts()
            || slice.ends != geometry.ends()
            || slice.strides != geometry.strides()
            || slice
                .shape
                .iter()
                .copied()
                .ne(geometry.shape().iter().map(|n| *n as u64))
        {
            return Err(CaptureStepError::InvalidCompletion);
        }
        if partition {
            value.finish_partition_with_scratch(slice, scratch)
        } else {
            value.finish_ordinary_with_scratch(slice, geometry.source_shape()[0] as u64, scratch)
        }
        .map_err(|_| CaptureStepError::InvalidCompletion)?;
        self.observed(index, Some(dtype));
        let record = &mut self.frame.records[index];
        record.payload = Some(CapturePayload::RoutedUnits(value));
        record.outcome = CaptureOutcome::Captured;
        record.charged = charged;
        Ok(())
    }
}
