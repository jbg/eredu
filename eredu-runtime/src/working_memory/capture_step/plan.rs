use super::*;

const DIAGNOSTIC_CAPACITY: usize = 256;
pub(super) const DIAGNOSTIC_BYTES: usize = DIAGNOSTIC_CAPACITY;

#[derive(Clone, Copy, Debug)]
pub(super) enum CaptureSkipRows<'a> {
    Selections(&'a [Option<CaptureSkipReason>]),
    EvidenceSides(&'a [Option<CaptureSkipReason>; 2], usize),
}
impl CaptureSkipRows<'_> {
    fn len(self) -> usize {
        match self {
            Self::Selections(rows) => rows.len(),
            Self::EvidenceSides(_, fields) => 2 * fields,
        }
    }
    pub(super) fn reason(self, index: usize) -> Option<CaptureSkipReason> {
        match self {
            Self::Selections(rows) => rows.get(index),
            Self::EvidenceSides(rows, fields) => rows.get(index / fields),
        }
        .cloned()
        .flatten()
    }
}

/// Allocation-free description of one actual admitted tensor/candidate frame.
///
/// Borrows the immutable admission and derives all record/string/buffer counts.
/// The source plan itself and nested tensor payloads are separate source and
/// destination ownership obligations. This plan proves no execution permission,
/// actual tensor provenance, native work bound, capture quota, or source charge.
#[derive(Debug)]
pub struct CaptureStepHostPlan<'a> {
    pub(super) source: &'a AdmittedCapturePlan,
    pub(super) phase: CapturePhase,
    pub(super) prediction: u64,
    pub(super) invocation: Option<CaptureInvocationShape>,
    pub(super) window: Option<CaptureInvocationWindow>,
    pub(super) selected: Option<&'a [bool]>,
    pub(super) skipped: Option<CaptureSkipRows<'a>>,
    retained: u64,
    allocated: u64,
    peak: u64,
}
fn add(a: u64, b: u64) -> Result<u64, WorkingMemoryError> {
    a.checked_add(b).ok_or(WorkingMemoryError::Overflow)
}
pub(super) fn extent(count: usize, width: usize) -> Result<u64, WorkingMemoryError> {
    count
        .checked_mul(width)
        .filter(|n| *n <= isize::MAX as usize)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)
}
impl<'a> CaptureStepHostPlan<'a> {
    /// Resolve one actual coordinate. Unknown active geometry is rejected; no
    /// record, string, shape or diagnostic buffer is allocated by preparation.
    pub fn prepare(
        source: &'a AdmittedCapturePlan,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
    ) -> Result<Self, CaptureStepError> {
        Self::prepare_selected(source, phase, prediction, invocation, None)
    }
    // A scope mask is descriptive only. The original caller authenticates it
    // against the retained plan's exact phase applicability before admission.
    pub(in crate::working_memory) fn prepare_selected(
        source: &'a AdmittedCapturePlan,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        selected: Option<&'a [bool]>,
    ) -> Result<Self, CaptureStepError> {
        Self::prepare_selected_window(source, phase, prediction, invocation, selected, None)
    }
    pub(in crate::working_memory) fn prepare_selected_window(
        source: &'a AdmittedCapturePlan,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        selected: Option<&'a [bool]>,
        window: Option<CaptureInvocationWindow>,
    ) -> Result<Self, CaptureStepError> {
        Self::prepare_selected_window_skips(
            source, phase, prediction, invocation, selected, window, None,
        )
    }
    pub(in crate::working_memory) fn prepare_selected_window_skips(
        source: &'a AdmittedCapturePlan,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        selected: Option<&'a [bool]>,
        window: Option<CaptureInvocationWindow>,
        skipped: Option<&'a [Option<CaptureSkipReason>]>,
    ) -> Result<Self, CaptureStepError> {
        Self::prepare_with_skips(
            source,
            phase,
            prediction,
            invocation,
            selected,
            window,
            skipped.map(CaptureSkipRows::Selections),
        )
    }
    pub(in crate::working_memory) fn prepare_intervention_window_skips(
        source: &'a AdmittedCapturePlan,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        selected: &'a [bool],
        window: Option<CaptureInvocationWindow>,
        skipped: Option<&'a [Option<CaptureSkipReason>; 2]>,
    ) -> Result<Self, CaptureStepError> {
        let fields = match source.plan().selections.len() {
            2 => 1,
            4 => 2,
            _ => return Err(CaptureStepError::InvalidCompletion),
        };
        Self::prepare_with_skips(
            source,
            phase,
            prediction,
            invocation,
            Some(selected),
            window,
            skipped.map(|rows| CaptureSkipRows::EvidenceSides(rows, fields)),
        )
    }
    fn prepare_with_skips(
        source: &'a AdmittedCapturePlan,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        selected: Option<&'a [bool]>,
        window: Option<CaptureInvocationWindow>,
        skipped: Option<CaptureSkipRows<'a>>,
    ) -> Result<Self, CaptureStepError> {
        if skipped.is_some_and(|rows| {
            rows.len() != source.plan().selections.len()
                || selected.is_none_or(|mask| {
                    mask.len() != rows.len()
                        || mask
                            .iter()
                            .enumerate()
                            .any(|(index, active)| rows.reason(index).is_some() && *active)
                })
        }) {
            return Err(CaptureStepError::InvalidCompletion);
        }
        if selected.is_some_and(|rows| rows.len() != source.plan().selections.len()) {
            return Err(CaptureStepError::InvalidCompletion);
        }
        source.geometry_at(phase, prediction, invocation)?;
        if let Some(window) = window {
            let physical = invocation.ok_or(CaptureStepError::InvalidCompletion)?;
            if source.invocation_bounds().is_none() {
                return Err(CaptureStepError::InvalidCompletion);
            }
            source.geometry_at(phase, prediction, Some(window.validate(physical)?))?;
        }
        if prediction >= source.request().max_predictions {
            return Err(CaptureTensorGeometryError::Inactive.into());
        }
        let count = source.plan().selections.len();
        let mut retained = add(
            size_of::<CapturedStep>() as u64,
            extent(count, size_of::<CaptureRecord>())?,
        )?;
        let mut result = Self {
            source,
            phase,
            prediction,
            invocation,
            window,
            selected,
            skipped,
            retained: 0,
            allocated: 0,
            peak: 0,
        };
        for (index, (selection, point)) in source
            .plan()
            .selections
            .iter()
            .zip(source.points())
            .enumerate()
        {
            if result.selected(index)
                && (!matches!(
                    point.dtype,
                    ObservationDtype::Floating | ObservationDtype::Integer
                ) || !matches!(
                    (&point.value_type, &selection.transform),
                    (ObservationValueType::Tensor, _)
                        | (
                            ObservationValueType::RoutedUnits { .. },
                            CaptureTransform::RoutedUnits
                        )
                ) || !matches!(
                    selection.transform,
                    CaptureTransform::FullTensor
                        | CaptureTransform::Slice
                        | CaptureTransform::Preview { .. }
                        | CaptureTransform::TopCandidates { .. }
                        | CaptureTransform::TokenScores { .. }
                        | CaptureTransform::Summary
                        | CaptureTransform::Histogram { .. }
                        | CaptureTransform::RoutedUnits
                ))
            {
                return Err(CaptureStepError::UnsupportedSelection { index });
            }
            // Validate existing logical metadata arithmetic without treating it
            // as a physical byte proof or reserving logical quota.
            crate::capture::metadata_reservation(selection, point)?;
            for text in [&selection.id, &selection.path, &point.node_id] {
                retained = add(retained, extent(text.len(), size_of::<u8>())?)?;
            }
            if let Some(geometry) = result.geometry(index)? {
                retained = add(
                    retained,
                    extent(geometry.source_shape().len(), 2 * size_of::<u64>())?,
                )?;
                retained = add(retained, DIAGNOSTIC_CAPACITY as u64)?;
                selected_elements(source, index, &geometry)?;
            } else if let Some(geometry) = result.summary_geometry(index)? {
                retained = add(
                    retained,
                    extent(geometry.source_shape().len(), 2 * size_of::<u64>())?,
                )?;
                retained = add(retained, DIAGNOSTIC_CAPACITY as u64)?;
            } else if let Some(geometry) = result.histogram_geometry(index)? {
                retained = add(
                    retained,
                    extent(geometry.source_shape().len(), 2 * size_of::<u64>())?,
                )?;
                retained = add(retained, DIAGNOSTIC_CAPACITY as u64)?;
            } else if let Some(geometry) = result.routed_geometry(index)? {
                retained = add(
                    retained,
                    extent(geometry.source_shape().len(), 2 * size_of::<u64>())?,
                )?;
                retained = add(retained, DIAGNOSTIC_CAPACITY as u64)?;
            } else if result.candidate_geometry(index)?.is_some()
                || result.token_score_geometry(index)?.is_some()
            {
                retained = add(retained, extent(3, 2 * size_of::<u64>())?)?;
                retained = add(retained, DIAGNOSTIC_CAPACITY as u64)?;
            }
        }
        let retained = add(
            retained,
            UnpublishedCapturedStep::retained_control_bytes::<CaptureFrameCustody>()
                .ok_or(WorkingMemoryError::Overflow)?,
        )?;
        let allocated = add(
            add(retained, extent(count, size_of::<RecordBuffers>())?)?,
            size_of::<Vec<RecordBuffers>>() as u64,
        )?;
        // Explicit initialized destinations plus conservative moved DTO/record/
        // sidecar and geometry values. No compiler-frame or allocator metadata
        // claim. P covers unused sidecars until finish/error as well as populated
        // record fields, and is deliberately retained through the last alias.
        let moves = 3 * size_of::<Vec<Option<super::interventions::evidence::PreparedInterventionEvidence<'static>>>>()
            + 2 * size_of::<CapturedStep>()
            + 2 * size_of::<CaptureRecord>()
            + 2 * size_of::<RecordBuffers>()
            + 3 * size_of::<Vec<Vec<u8>>>()
            + 2 * size_of::<CaptureTensorGeometry<'_>>()
            + 2 * size_of::<CaptureTokenScoreGeometry<'_>>()
            + 2 * size_of::<CaptureSummaryGeometry<'_>>()
            + 2 * size_of::<CaptureHistogramGeometry<'_>>()
            + 2 * size_of::<CaptureRoutedUnitsGeometry<'_>>()
            + 2 * size_of::<u64>()
            + 2 * size_of::<u8>()
            // Precommit delivery is an additional owned handle with two moves;
            // finalization reuses its already allocated shared destination.
            + builder::DELIVERY_CONTROL_BYTES
            + crate::capture::RECORD_ENCODING_CONTROL_BYTES
            // Every detached abort/error can retain its own frame concurrently.
            + 3 * size_of::<PendingCaptureDelivery>()
            + 3 * size_of::<PendingCaptureDeliveryError>();
        result.retained = retained;
        result.allocated = allocated;
        result.peak = add(
            allocated,
            u64::try_from(moves).map_err(|_| WorkingMemoryError::Overflow)?,
        )?;
        Ok(result)
    }
    pub(in crate::working_memory) fn selected(&self, index: usize) -> bool {
        self.selected
            .is_none_or(|rows| rows.get(index).copied().unwrap_or(false))
    }
    /// Exact immutable semantic source; borrowing supplies no source charge.
    pub fn admission(&self) -> &'a AdmittedCapturePlan {
        self.source
    }
    /// All selection records, including schedule skips and missing points.
    pub fn len(&self) -> usize {
        self.source.plan().selections.len()
    }
    /// Whether the admitted frame has no selection records.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Maximum final frame-owned inline, buffer and closed shared-control capacity.
    /// Unused shapes/diagnostics may retire at finish. Numerical SharedTensor
    /// data and candidate Vec storage (priced by their destination plans) are excluded.
    pub fn retained_payload_bytes(&self) -> u64 {
        self.retained
    }
    /// Exact initialized frame/sidecar inline, shared-control and requested buffer capacities.
    /// Excludes source plans, nested numerical payloads and allocator bookkeeping.
    pub fn allocated_payload_bytes(&self) -> u64 {
        self.allocated
    }
    /// Protected construction envelope including explicit moved values.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.peak
    }
    pub(in crate::working_memory) fn candidate_geometry(
        &self,
        index: usize,
    ) -> Result<Option<CaptureCandidateGeometry<'a>>, CaptureStepError> {
        if !self.selected(index) {
            return Ok(None);
        }
        let selection = self
            .source
            .plan()
            .selections
            .get(index)
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        let point = &self.source.points()[index];
        if !matches!(selection.transform, CaptureTransform::TopCandidates { .. })
            || !selection.schedule.includes(self.phase, self.prediction)
            || !match self.phase {
                CapturePhase::Prefill => point.prefill,
                CapturePhase::Decode => point.decode,
            }
        {
            return Ok(None);
        }
        if self.window.is_some() {
            return Err(CaptureStepError::UnsupportedSelection { index });
        }
        Ok(Some(CaptureCandidateGeometry::prepare(
            self.source,
            index,
            self.phase,
            self.prediction,
            self.invocation,
        )?))
    }
    pub(in crate::working_memory) fn token_score_geometry(
        &self,
        index: usize,
    ) -> Result<Option<CaptureTokenScoreGeometry<'a>>, CaptureStepError> {
        if !self.selected(index) {
            return Ok(None);
        }
        let selection = self
            .source
            .plan()
            .selections
            .get(index)
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        let point = &self.source.points()[index];
        if !matches!(selection.transform, CaptureTransform::TokenScores { .. })
            || !selection.schedule.includes(self.phase, self.prediction)
            || !match self.phase {
                CapturePhase::Prefill => point.prefill,
                CapturePhase::Decode => point.decode,
            }
        {
            return Ok(None);
        }
        if self.window.is_some() {
            return Err(CaptureStepError::UnsupportedSelection { index });
        }
        Ok(Some(CaptureTokenScoreGeometry::prepare(
            self.source,
            index,
            self.phase,
            self.prediction,
            self.invocation,
        )?))
    }
    pub(in crate::working_memory) fn routed_geometry(
        &self,
        index: usize,
    ) -> Result<Option<CaptureRoutedUnitsGeometry<'a>>, CaptureStepError> {
        if !self.selected(index) {
            return Ok(None);
        }
        let selection = self
            .source
            .plan()
            .selections
            .get(index)
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        let point = &self.source.points()[index];
        if !matches!(selection.transform, CaptureTransform::RoutedUnits)
            || !selection.schedule.includes(self.phase, self.prediction)
            || !match self.phase {
                CapturePhase::Prefill => point.prefill,
                CapturePhase::Decode => point.decode,
            }
        {
            return Ok(None);
        }
        Ok(Some(match self.window {
            Some(window) => CaptureRoutedUnitsGeometry::prepare_window(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation.ok_or(CaptureStepError::InvalidCompletion)?,
                window,
            )?,
            None => CaptureRoutedUnitsGeometry::prepare(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation,
            )?,
        }))
    }
    pub(in crate::working_memory) fn summary_geometry(
        &self,
        index: usize,
    ) -> Result<Option<CaptureSummaryGeometry<'a>>, CaptureStepError> {
        if !self.selected(index) {
            return Ok(None);
        }
        let selection = self
            .source
            .plan()
            .selections
            .get(index)
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        let point = &self.source.points()[index];
        if !matches!(selection.transform, CaptureTransform::Summary)
            || !selection.schedule.includes(self.phase, self.prediction)
            || !match self.phase {
                CapturePhase::Prefill => point.prefill,
                CapturePhase::Decode => point.decode,
            }
        {
            return Ok(None);
        }
        Ok(Some(match self.window {
            Some(window) => CaptureSummaryGeometry::prepare_window(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation.ok_or(CaptureStepError::InvalidCompletion)?,
                window,
            )?,
            None => CaptureSummaryGeometry::prepare(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation,
            )?,
        }))
    }
    pub(in crate::working_memory) fn histogram_geometry(
        &self,
        index: usize,
    ) -> Result<Option<CaptureHistogramGeometry<'a>>, CaptureStepError> {
        if !self.selected(index) {
            return Ok(None);
        }
        let selection = self
            .source
            .plan()
            .selections
            .get(index)
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        let point = &self.source.points()[index];
        if !matches!(selection.transform, CaptureTransform::Histogram { .. })
            || !selection.schedule.includes(self.phase, self.prediction)
            || !match self.phase {
                CapturePhase::Prefill => point.prefill,
                CapturePhase::Decode => point.decode,
            }
        {
            return Ok(None);
        }
        Ok(Some(match self.window {
            Some(window) => CaptureHistogramGeometry::prepare_window(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation.ok_or(CaptureStepError::InvalidCompletion)?,
                window,
            )?,
            None => CaptureHistogramGeometry::prepare(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation,
            )?,
        }))
    }
    pub(in crate::working_memory) fn active(&self, index: usize) -> Result<bool, CaptureStepError> {
        Ok(self.geometry(index)?.is_some()
            || self.candidate_geometry(index)?.is_some()
            || self.token_score_geometry(index)?.is_some()
            || self.summary_geometry(index)?.is_some()
            || self.histogram_geometry(index)?.is_some()
            || self.routed_geometry(index)?.is_some())
    }
    pub(in crate::working_memory) fn geometry(
        &self,
        index: usize,
    ) -> Result<Option<CaptureTensorGeometry<'a>>, CaptureStepError> {
        if !self.selected(index) {
            return Ok(None);
        }
        let selection = self
            .source
            .plan()
            .selections
            .get(index)
            .ok_or(CaptureStepError::RecordNotPending { index })?;
        let point = &self.source.points()[index];
        if !selection.schedule.includes(self.phase, self.prediction)
            || !match self.phase {
                CapturePhase::Prefill => point.prefill,
                CapturePhase::Decode => point.decode,
            }
        {
            return Ok(None);
        }
        if matches!(
            selection.transform,
            CaptureTransform::TopCandidates { .. }
                | CaptureTransform::TokenScores { .. }
                | CaptureTransform::Summary
                | CaptureTransform::Histogram { .. }
                | CaptureTransform::RoutedUnits
        ) {
            return Ok(None);
        }
        Ok(Some(match self.window {
            Some(window) => CaptureTensorGeometry::prepare_window(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation.ok_or(CaptureStepError::InvalidCompletion)?,
                window,
            )?,
            None => CaptureTensorGeometry::prepare(
                self.source,
                index,
                self.phase,
                self.prediction,
                self.invocation,
            )?,
        }))
    }
}
// CaptureTensorGeometry already validated these actual source extents/slices.
// Record selected_shape retains pre-preview rank; the tensor output may flatten.
pub(super) fn selected_extent(
    source: &AdmittedCapturePlan,
    index: usize,
    geometry: &CaptureTensorGeometry<'_>,
    axis: usize,
) -> u64 {
    debug_assert!(std::ptr::eq(source, geometry.admission()));
    debug_assert_eq!(index, geometry.selection_index());
    (geometry.ends()[axis] - geometry.starts()[axis]).div_ceil(geometry.strides()[axis])
}
pub(super) fn selected_elements(
    source: &AdmittedCapturePlan,
    index: usize,
    geometry: &CaptureTensorGeometry<'_>,
) -> Result<u64, WorkingMemoryError> {
    (0..geometry.source_shape().len()).try_fold(1u64, |n, axis| {
        n.checked_mul(selected_extent(source, index, geometry, axis))
            .ok_or(WorkingMemoryError::Overflow)
    })
}
