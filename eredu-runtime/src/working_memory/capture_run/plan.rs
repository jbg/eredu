use super::*;

pub(super) fn phase(prediction: usize) -> CapturePhase {
    if prediction == 0 {
        CapturePhase::Prefill
    } else {
        CapturePhase::Decode
    }
}
fn add(a: u64, b: u64) -> Result<u64, WorkingMemoryError> {
    a.checked_add(b).ok_or(WorkingMemoryError::Overflow)
}
fn extent(count: usize, width: usize) -> Result<u64, WorkingMemoryError> {
    count
        .checked_mul(width)
        .filter(|n| *n <= isize::MAX as usize)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)
}

/// Allocation-free cumulative host program bound to one actual immutable plan.
///
/// Prices every frame (including prefill metadata and schedule gaps), every
/// eligible tensor, and a fixed finite claim table/control envelope. Ordinary
/// text origin and context growth are resolved through the admitted geometry.
/// Unknown active geometry is rejected. Native transforms, the source plan's
/// existing physical storage and facade delivery envelopes are not priced here.
#[derive(Debug, Clone)]
pub struct CaptureRunHostPlan<'a> {
    pub(super) source: &'a SharedCapturePlan,
    pub(super) interventions: Option<super::super::OriginalInterventionSource>,
    end_prediction: u64,
    pub(super) invocation: Option<Invocation<'a>>,
    pub(super) continuation: bool,
    pub(super) first_prediction: usize,
    pub(super) predictions: usize,
    pub(super) width: usize,
    pub(super) slots: usize,
    frames: u64,
    tensors: u64,
    controls: u64,
    pub(super) peak: u64,
}
#[derive(Debug, Clone, Copy)]
pub(super) struct Invocation<'a> {
    pub(super) phase: CapturePhase,
    pub(super) shape: CaptureInvocationShape,
    pub(super) window: Option<CaptureInvocationWindow>,
    pub(super) selected: &'a [bool],
    pub(super) skipped: Option<&'a [Option<CaptureSkipReason>]>,
}
impl<'a> CaptureRunHostPlan<'a> {
    /// Resolve the original finite ordinary schedule without allocating or
    /// cloning any semantic plan/history/tensor payload.
    pub fn prepare(source: &'a SharedCapturePlan) -> Result<Self, CaptureRunHostError> {
        Self::prepare_range(
            source,
            0,
            source.admission().request().max_predictions,
            false,
        )
    }

    // Only the closed funded checkpoint can select a later first coordinate.
    // The actual frame workers continue to use the source's absolute schedule.
    pub(crate) fn prepare_range(
        source: &'a SharedCapturePlan,
        first_prediction: u64,
        end_prediction: u64,
        continuation: bool,
    ) -> Result<Self, CaptureRunHostError> {
        Self::prepare_sequence(
            source,
            first_prediction,
            end_prediction,
            continuation,
            None,
            false,
        )
    }

    /// Price one exact independent invocation using the same finite frame and
    /// transformation constructors as ordinary capture. Physical phase/shape do
    /// not follow the logical prediction coordinate. The scope mask is borrowed
    /// from the caller's prepared selection and copied only after real funding.
    /// This descriptor supplies no source, native or invocation authority.
    pub fn prepare_invocation(
        source: &'a SharedCapturePlan,
        phase: CapturePhase,
        prediction: u64,
        shape: CaptureInvocationShape,
        selected: &'a [bool],
    ) -> Result<Self, CaptureRunHostError> {
        Self::prepare_invocation_window(source, phase, prediction, shape, selected, None)
    }
    /// Price the same physical frame with globally anchored slice coordinates.
    /// The window supplies no source, quota, native or occurrence authority.
    pub fn prepare_invocation_window(
        source: &'a SharedCapturePlan,
        phase: CapturePhase,
        prediction: u64,
        shape: CaptureInvocationShape,
        selected: &'a [bool],
        window: Option<CaptureInvocationWindow>,
    ) -> Result<Self, CaptureRunHostError> {
        if source.admission().invocation_bounds().is_none()
            || selected.len() != source.admission().plan().selections.len()
        {
            return Err(CaptureRunHostError::ExplicitInvocation);
        }
        let end = prediction
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        Self::prepare_sequence(
            source,
            prediction,
            end,
            false,
            Some(Invocation {
                phase,
                shape,
                window,
                selected,
                skipped: None,
            }),
            false,
        )
    }

    /// Add one exact original edit source to the ordinary finite text schedule.
    /// Every prediction retains its outcome rows, including inactive edits and
    /// capture plans with no tensor selections. Native edits and their completion
    /// still require the selected backend's original numerical quotation.
    pub fn with_interventions(
        mut self,
        source: &super::super::OriginalInterventionSource,
    ) -> Result<Self, CaptureRunHostError> {
        if self.interventions.is_some()
            || self.invocation.is_some()
            || source.plan().admission().request() != self.source.admission().request()
            || source.plan().admission().text_origin() != self.source.admission().text_origin()
            || source.plan().admission().invocation_bounds().is_some()
            || self.source.admission().invocation_bounds().is_some()
            || source.plan().admission().points().len()
                != source.plan().admission().plan().operations.len()
        {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        // An empty capture plan alone has no frames. Intervention records still
        // belong to every logical prediction, using the same frame constructors.
        if self.predictions == self.first_prediction
            && self.end_prediction > self.first_prediction as u64
        {
            self = Self::prepare_sequence(
                self.source,
                self.first_prediction as u64,
                self.end_prediction,
                self.continuation,
                None,
                true,
            )?;
        }
        let mut extra = 0u64;
        for prediction in self.first_prediction..self.predictions {
            let step = super::interventions::StepPlan::prepare(
                self.source,
                source,
                self.phase(prediction),
                prediction as u64,
            )?;
            extra = add(extra, step.peak())?;
        }
        self.peak = add(self.peak, extra)?;
        let additional_slots = self
            .predictions
            .checked_sub(self.first_prediction)
            .and_then(|count| count.checked_mul(source.plan().admission().points().len()))
            .ok_or(WorkingMemoryError::Overflow)?;
        self.slots = self
            .slots
            .checked_add(additional_slots)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.interventions = Some(source.clone());
        Ok(self)
    }

    /// Exact source alias carried into the accepted host bank. This borrow
    /// grants neither execution nor a new source account.
    pub fn intervention_source(&self) -> Option<&super::super::OriginalInterventionSource> {
        self.interventions.as_ref()
    }

    /// Retain exact logical skip decisions alongside an already inactive scope
    /// mask. This changes only record attribution; it cannot enable a claim or
    /// reduce the existing frame, native, or cumulative capture obligations.
    pub fn with_skip_reasons(
        mut self,
        skipped: &'a [Option<CaptureSkipReason>],
    ) -> Result<Self, CaptureRunHostError> {
        let invocation = self
            .invocation
            .as_mut()
            .ok_or(CaptureRunHostError::ExplicitInvocation)?;
        if invocation.skipped.is_some()
            || skipped.len() != invocation.selected.len()
            || skipped
                .iter()
                .zip(invocation.selected)
                .any(|(reason, selected)| reason.is_some() && *selected)
        {
            return Err(CaptureRunHostError::ExplicitInvocation);
        }
        let added = add(
            extent(skipped.len(), size_of::<Option<CaptureSkipReason>>())?,
            extent(
                1,
                size_of::<(
                    Vec<Option<CaptureSkipReason>>,
                    Box<[Option<CaptureSkipReason>]>,
                    Option<CaptureSkipReason>,
                    Result<Self, CaptureRunHostError>,
                )>(),
            )?,
        )?;
        self.controls = add(self.controls, added)?;
        self.peak = add(self.peak, added)?;
        invocation.skipped = Some(skipped);
        Ok(self)
    }

    fn prepare_sequence(
        source: &'a SharedCapturePlan,
        first_prediction: u64,
        end_prediction: u64,
        continuation: bool,
        invocation: Option<Invocation<'a>>,
        include_empty_frames: bool,
    ) -> Result<Self, CaptureRunHostError> {
        let admission = source.admission();
        if first_prediction > end_prediction || end_prediction > admission.request().max_predictions
        {
            return Err(CaptureRunHostError::Coordinate);
        }
        let first_prediction =
            usize::try_from(first_prediction).map_err(|_| WorkingMemoryError::Overflow)?;
        if invocation.is_none() && admission.text_origin().is_none() {
            return Err(CaptureRunHostError::ExplicitInvocation);
        }
        let selections = admission.plan().selections.len();
        let predictions = if selections == 0 && invocation.is_none() && !include_empty_frames {
            first_prediction
        } else {
            usize::try_from(end_prediction).map_err(|_| WorkingMemoryError::Overflow)?
        };
        let width = selections
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        let slots = predictions
            .checked_sub(first_prediction)
            .and_then(|count| count.checked_mul(width))
            .ok_or(WorkingMemoryError::Overflow)?;
        let table = extent(slots, size_of::<ClaimState>())?;
        // Explicit owned controls plus two moves of each. All simultaneous
        // constructors are exclusive; finished DTO/data storage is priced in
        // the child P sums. Arc/account/registry bookkeeping and compiler stack
        // retain their existing classification. Transfer K is always behind the
        // existing storage Arc (never stored inline), so u8 measures the exact
        // generic handle representation. No arbitrary added allocation is
        // excluded by this formula. The temporary Vec header and fill values are
        // included, as are the borrowed geometry/closed plan controls.
        let control = size_of::<PreparedCaptureRun>()
            .checked_add(size_of::<CaptureRunHostPlan<'_>>())
            .and_then(|n| n.checked_add(size_of::<CaptureStepClaim<'_>>()))
            .and_then(|n| n.checked_add(size_of::<ScheduledCaptureStep<'_>>()))
            .and_then(|n| n.checked_add(size_of::<CaptureTensorClaim<'_, '_>>()))
            // One simultaneously active capture source channel. Its actual
            // inline owner/slot controls and moves are included; Arc/registry
            // source bundles retain the existing bookkeeping classification.
            .and_then(|n| n.checked_add(size_of::<crate::working_memory::CaptureSourceSegment>()))
            .and_then(|n| n.checked_add(super::super::funding::capture_source_control_bytes()))
            .and_then(|n| n.checked_add(size_of::<CapturePrefillSourceBootstrap<'_>>()))
            .and_then(|n| n.checked_add(size_of::<ClaimedCaptureTensor>()))
            .and_then(|n| n.checked_add(size_of::<ScheduledCaptureTensorFailure>()))
            .and_then(|n| n.checked_add(size_of::<ScheduledCaptureTensor<'_, '_>>()))
            .and_then(|n| {
                n.checked_add(size_of::<ScheduledCaptureTensorTransfer<'_, '_, '_, u8>>())
            })
            .and_then(|n| n.checked_add(size_of::<ScheduledCaptureStepFinishError<'_>>()))
            .and_then(|n| n.checked_add(size_of::<ScheduledCaptureTensorFinishError<'_, '_>>()))
            .and_then(|n| {
                n.checked_add(size_of::<
                    ScheduledCaptureTensorTransferFinishError<'_, '_, '_, u8>,
                >())
            })
            .and_then(|n| n.checked_add(size_of::<CaptureTensorHostPlan<'_>>()))
            .and_then(|n| n.checked_add(size_of::<CaptureStepHostPlan<'_>>()))
            .and_then(|n| n.checked_add(size_of::<Vec<ClaimState>>()))
            .and_then(|n| n.checked_mul(3))
            .ok_or(WorkingMemoryError::Overflow)?;
        let controls = add(table, extent(1, control)?)?;
        let controls = add(
            controls,
            if invocation.is_some() {
                add(
                    extent(selections, size_of::<bool>())?,
                    extent(
                        1,
                        size_of::<Invocation<'_>>()
                            .checked_add(size_of::<super::OwnedInvocation>())
                            .and_then(|n| n.checked_add(size_of::<Vec<bool>>()))
                            .and_then(|n| n.checked_add(size_of::<Box<[bool]>>()))
                            .ok_or(WorkingMemoryError::Overflow)?,
                    )?,
                )?
            } else {
                0
            },
        )?;
        // p0 may keep every target partially filled simultaneously. The actual
        // slot box owns fixed destination headers, progress, optional completed
        // alias and original-H custody. Numeric/shape payload is already counted
        // once by tensor P below. No fragment allocates another numeric buffer.
        let controls = add(
            controls,
            if invocation.is_some() || first_prediction != 0 || predictions == 0 {
                0
            } else {
                prefill_target_bytes(selections)?
            },
        )?;
        // One consuming optional session factory and one lexical observer. Its
        // actual inner identity payload, wrapper, delivery slot and moves are
        // included before even the claim table is allocated.
        let controls = add(controls, crate::capture::funded::control_bytes()?)?;
        let controls = add(
            controls,
            if continuation {
                crate::capture::funded::checkpoint::continuation_control_bytes()?
            } else {
                0
            },
        )?;
        let controls = add(
            controls,
            if slots == 0 {
                0
            } else {
                2 * size_of::<ClaimState>() as u64
            },
        )?;
        let mut result = Self {
            source,
            interventions: None,
            end_prediction,
            invocation,
            continuation,
            first_prediction,
            predictions,
            width,
            slots,
            frames: 0,
            tensors: 0,
            controls,
            peak: 0,
        };
        for p in first_prediction..predictions {
            let frame = result.frame(p)?;
            result.frames = add(result.frames, frame.initialization_peak_bytes())?;
            for index in 0..selections {
                if let Some(geometry) = frame.candidate_geometry(index)? {
                    let candidates = CaptureCandidateHostPlan::prepare(geometry)?;
                    result.tensors = add(result.tensors, candidates.initialization_peak_bytes())?;
                }
                if let Some(geometry) = frame.token_score_geometry(index)? {
                    let scores = CaptureTokenScoreHostPlan::prepare(geometry)?;
                    result.tensors = add(result.tensors, scores.initialization_peak_bytes())?;
                }
                if let Some(geometry) = frame.summary_geometry(index)? {
                    let summary = CaptureSummaryHostPlan::prepare(geometry)?;
                    result.tensors = add(result.tensors, summary.initialization_peak_bytes())?;
                }
                if let Some(geometry) = frame.routed_geometry(index)? {
                    let routed = CaptureRoutedHostPlan::prepare(geometry)?;
                    result.tensors = add(result.tensors, routed.initialization_peak_bytes())?;
                }
                if let Some(geometry) = frame.histogram_geometry(index)? {
                    let histogram = CaptureHistogramHostPlan::prepare(geometry)?;
                    result.tensors = add(result.tensors, histogram.initialization_peak_bytes())?;
                }
                if let Some(geometry) = frame.geometry(index)? {
                    let tensor = CaptureTensorHostPlan::prepare(geometry)?;
                    result.tensors = add(result.tensors, tensor.initialization_peak_bytes())?;
                    // Every completed receipt may escape simultaneously. Its
                    // semantic coordinate/owner control is not assumed to be the
                    // single exclusive constructor. Two moved receipts are
                    // already included in the fixed controls above.
                    result.controls = add(
                        result.controls,
                        size_of::<ClaimedCaptureTensor>()
                            .max(size_of::<ScheduledCaptureTensorFailure>())
                            as u64,
                    )?;
                }
            }
        }
        result.peak = add(add(result.frames, result.tensors)?, result.controls)?;
        Ok(result)
    }
    pub(super) fn frame(
        &self,
        prediction: usize,
    ) -> Result<CaptureStepHostPlan<'a>, CaptureStepError> {
        CaptureStepHostPlan::prepare_selected_window_skips(
            self.source.admission(),
            self.phase(prediction),
            prediction as u64,
            self.invocation.map(|value| value.shape),
            self.invocation.map(|value| value.selected),
            self.invocation.and_then(|value| value.window),
            self.invocation.and_then(|value| value.skipped),
        )
    }
    pub(super) fn phase(&self, prediction: usize) -> CapturePhase {
        self.invocation
            .map_or_else(|| phase(prediction), |value| value.phase)
    }
    /// Borrow the very same shared source used by the eventual claims.
    pub fn source(&self) -> &'a SharedCapturePlan {
        self.source
    }
    /// Sum of all actual frame construction envelopes, including skipped records.
    pub fn frame_peak_bytes(&self) -> u64 {
        self.frames
    }
    /// Sum of all eligible tensor construction envelopes with context growth.
    pub fn tensor_peak_bytes(&self) -> u64 {
        self.tensors
    }
    /// Fixed claim table, every potentially retained coordinate receipt, and
    /// explicit exclusive constructor/error controls and initialization moves.
    pub fn control_peak_bytes(&self) -> u64 {
        self.controls
    }
    /// Exact number of fixed one-byte claim states constructed by this worker.
    pub fn claim_slots(&self) -> usize {
        self.slots
    }
    /// Complete protected H retained through the final output alias. No native
    /// budget, source registration, capture quota or run stage is created.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.peak
    }
}

/// Same actual target box and worker controls for a parent or original evidence
/// frame. Numeric destinations remain in their existing tensor/summary source.
pub(in crate::working_memory) fn prefill_target_bytes(
    selections: usize,
) -> Result<u64, WorkingMemoryError> {
    let partial = extent(selections, size_of::<capture_tensor::prefill::TargetSlot>())?;
    let moves = size_of::<capture_tensor::prefill::PrefillTargets>()
        .checked_add(size_of::<Vec<capture_tensor::prefill::TargetSlot>>())
        .and_then(|n| n.checked_add(size_of::<capture_tensor::prefill::TargetSlot>()))
        .and_then(|n| n.checked_add(size_of::<capture_tensor::prefill::OwnedPrefillTensor>()))
        // The shared allocator now borrows either the original
        // whole geometry or an authenticated receipt projection.
        .and_then(|n| {
            n.checked_add(size_of::<(&CaptureTensorGeometry<'_>, CaptureTensorCustody)>())
        })
        .and_then(|n| {
            n.checked_add(size_of::<
                Result<capture_tensor::prefill::OwnedPrefillTensor, WorkingMemoryError>,
            >())
        })
        .and_then(|n| n.checked_add(size_of::<CapturePrefillFragmentClaim<'_, '_, '_, '_>>()))
        .and_then(|n| n.checked_add(size_of::<CapturePrefillFragmentWriter<'_, '_, '_, '_>>()))
        .and_then(|n| {
            n.checked_add(size_of::<
                CapturePrefillFragmentTransfer<'_, '_, '_, '_, '_, u8>,
            >())
        })
        .and_then(|n| n.checked_add(super::super::funding::capture_source_rollback_control_bytes()))
        .and_then(|n| n.checked_add(size_of::<CapturePrefillHostError>()))
        .and_then(|n| n.checked_add(size_of::<CaptureRunHostError>()))
        .and_then(|n| {
            n.checked_add(size_of::<crate::capture::CapturePrefillObservationPolicy<'_>>())
        })
        .and_then(|n| n.checked_add(size_of::<crate::capture::CapturePrefillObservationRow<'_>>()))
        .and_then(|n| n.checked_add(size_of::<CapturePrefillRowAssembly<'_>>()))
        .and_then(|n| n.checked_add(size_of::<CapturePrefillFragment<'_, '_>>()))
        .and_then(|n| n.checked_add(size_of::<CapturePrefillElement>()))
        .and_then(|n| n.checked_mul(3))
        .ok_or(WorkingMemoryError::Overflow)?;
    add(partial, extent(1, moves)?)
}
