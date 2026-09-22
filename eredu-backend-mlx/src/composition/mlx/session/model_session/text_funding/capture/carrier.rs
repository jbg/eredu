//! Capture-only native custody inside the original work; no new execution scope.
use super::*;
use crate::backend::array_copy::PreparedCaptureFragment;
use eredu_runtime::{
    inspection::{
        PrefillChunkRetentionContext, PreparedPrefillChunkRetention, SettledPrefillChunkRetention,
    },
    working_memory::{
        CapturePrefillFragmentClaim, CapturePrefillSourceBootstrap, CaptureSourceSegment,
        SettledCaptureSourceParcel, WorkingMemoryError,
    },
};
use std::mem::{size_of, take};
mod candidates;
mod generated;
mod histogram;
mod routed;
mod summary;
mod token_scores;

#[derive(Debug, thiserror::Error)]
pub(super) enum CaptureCarrierError {
    #[error("capture source carrier is missing or already active")]
    Identity,
    #[error("capture native work has not completed successfully")]
    Incomplete,
    #[error("completed native submission retirement is busy")]
    RetirementBusy,
    #[error("capture retirement requires an unlocked ordinary host boundary")]
    RetirementBoundary,
}
fn error<E: std::error::Error + Send + Sync + 'static>(e: E) -> Error {
    Error::Other(Box::new(e))
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Progress {
    Ready,
    Running,
    Failed,
}

// Payload fields precede the segment's original host custody. This owner is
// never published as an allocation sidecar and cannot create an Array cycle.
pub(in crate::composition::mlx::session::model_session::text_funding) struct CaptureCarrier {
    roots: RefCell<Vec<Array>>,
    publications: RefCell<Vec<RetainedStoragePublication>>,
    // The genuine bootstrap runs inside this chunk's SpanOuter. Native record
    // membership includes its InputTransaction descendants, never the later
    // cancellation boundary. Absence preserves the legacy ordinary path.
    native_ancestor: Option<safemlx::OriginalScopeObserver>,
    segment: Option<CaptureSourceSegment>,
    progress: Cell<Progress>,
}
// The single Box never escapes this closed owner. All moves, including the
// empty canonical carrier and failed/unwinding FundedWork, retire identically.
pub(in crate::composition::mlx::session::model_session::text_funding) struct CaptureCarrierOwner(
    Option<Box<CaptureCarrier>>,
);
impl CaptureCarrierOwner {
    fn prepared(
        source: &eredu_core::capture::SharedCapturePlan,
        interventions: Option<&eredu_runtime::working_memory::OriginalInterventionSource>,
    ) -> Result<Self, Error> {
        Ok(Self(Some(Box::new(CaptureCarrier::prepared(
            source,
            interventions,
        )?))))
    }
}
impl std::ops::Deref for CaptureCarrierOwner {
    type Target = CaptureCarrier;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live capture carrier")
    }
}
impl std::ops::DerefMut for CaptureCarrierOwner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_deref_mut().expect("live capture carrier")
    }
}
// Returning from this consuming function deallocates the Box before the caller
// can destroy the moved payload. No callback/native work precedes that return.
fn unbox_carrier(owner: Box<CaptureCarrier>) -> CaptureCarrier {
    *owner
}
impl Drop for CaptureCarrierOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            let payload = unbox_carrier(owner);
            drop(payload);
        }
    }
}

impl CaptureCarrier {
    fn prepared(
        source: &eredu_core::capture::SharedCapturePlan,
        interventions: Option<&eredu_runtime::working_memory::OriginalInterventionSource>,
    ) -> Result<Self, Error> {
        let requirements = collector_requirements(source, interventions)?;
        let (descriptors, publications) = (requirements.descriptors, requirements.publications);
        let mut roots = Vec::new();
        let mut retained_publications = Vec::new();
        // Fresh collectors allocate once. Neither grows after bootstrap; the
        // leaf's reserve_exact is a no-op after our preflight capacity check.
        roots.try_reserve_exact(descriptors).map_err(error)?;
        retained_publications
            .try_reserve_exact(publications)
            .map_err(error)?;
        if roots.capacity() != descriptors || retained_publications.capacity() != publications {
            return Err(error(WorkingMemoryError::UnknownBound));
        }
        Ok(Self {
            roots: RefCell::new(roots),
            publications: RefCell::new(retained_publications),
            segment: None,
            progress: Cell::new(Progress::Failed),
            native_ancestor: safemlx::OriginalScopeObserver::try_current()?,
        })
    }
    fn require_capacity(&self, descriptors: usize) -> Result<(), Error> {
        let roots = self.roots.try_borrow().map_err(error)?;
        let publications = self.publications.try_borrow().map_err(error)?;
        if roots.capacity().saturating_sub(roots.len()) < descriptors
            || publications.len() == publications.capacity()
        {
            return Err(error(WorkingMemoryError::UnknownBound));
        }
        Ok(())
    }
}
struct CaptureAttempt<'a> {
    progress: &'a Cell<Progress>,
    complete: bool,
}
impl<'a> CaptureAttempt<'a> {
    fn new(progress: &'a Cell<Progress>) -> Result<Self, Error> {
        if progress.get() != Progress::Ready {
            return Err(error(CaptureCarrierError::Incomplete));
        }
        progress.set(Progress::Running);
        Ok(Self {
            progress,
            complete: false,
        })
    }
    fn complete(mut self) {
        self.progress.set(Progress::Ready);
        self.complete = true;
    }
}
impl Drop for CaptureAttempt<'_> {
    fn drop(&mut self) {
        if !self.complete {
            self.progress.set(Progress::Failed);
        }
    }
}
struct CapturePayload {
    roots: Vec<Array>,
    publications: Vec<RetainedStoragePublication>,
    opening: Option<crate::composition::mlx::replicated_text::RetiredOpeningRow>,
}
struct SettledCapture {
    payload: Option<CapturePayload>,
    native_ancestor: Option<safemlx::OriginalScopeObserver>,
    parcel: Option<SettledCaptureSourceParcel>,
    segment: Option<CaptureSourceSegment>,
    ticket: Option<SettledPrefillChunkRetention>,
}
impl SettledCapture {
    fn retire(mut self) {
        // No Usage, native lock or FundedWork RefCell loan survives here.
        // On destructor unwind Rust still drops payload fields before the parcel.
        drop(self.payload.take());
        crate::backend::ordinary_retirement::reclaim();
        safemlx::reclaim_allocation_owners();
        // The exact subtree's records retired before the parcel was taken.
        // Keep that native owner through all captured payload/publication drops
        // and retire its own final reference before releasing source custody.
        drop(self.native_ancestor.take());
        drop(self.parcel.take());
        drop(self.segment.take());
        drop(self.ticket.take());
        // No byte credit is assumed. Other graph/Array/publication owners may
        // survive; the next original admission uses actual current accounting.
    }
}

fn retire_capture_records(ancestor: Option<&safemlx::OriginalScopeObserver>) -> Result<(), Error> {
    let retired = match ancestor {
        None => safemlx::try_retire_completed_submissions()?,
        Some(ancestor) => {
            let (progress, status) = ancestor.progress()?;
            if let Some(cause) = ancestor.observation_error(progress) {
                return Err(error(cause));
            }
            if status.failed() {
                return Err(ancestor
                    .retained_failure()
                    .map(error)
                    .unwrap_or_else(|| error(CaptureCarrierError::Incomplete)));
            }
            if status.blocked() || !status.is_settled() {
                return Err(error(CaptureCarrierError::Incomplete));
            }
            ancestor.retire_completed_records()?
        }
    };
    match retired {
        safemlx::SubmissionRetirement::CompleteSnapshot => Ok(()),
        _ => Err(error(CaptureCarrierError::RetirementBusy)),
    }
}

/// Introduced common representation only. The original quote composers include
/// this for each work owner; it is not a reservation or retained payload grant.
pub(in crate::composition::mlx::session::model_session) fn common_control_bytes(
) -> Result<u64, Error> {
    size_of::<RefCell<Option<CaptureCarrierOwner>>>()
        .checked_add(size_of::<
            Option<eredu_runtime::working_memory::OriginalTextControlGuard>,
        >())
        .and_then(|n| {
            n.checked_add(size_of::<
                Option<crate::composition::mlx::replicated_text::NativeOpeningRowsOwner>,
            >())
        })
        .and_then(|n| u64::try_from(n).ok())
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| error(WorkingMemoryError::Overflow))
}
/// One active boxed carrier and its terminal movement/attempt controls. Numeric
/// arrays remain in the existing equation trace; registry nodes are not arrays.
pub(in crate::composition::mlx::session::model_session) fn active_control_bytes(
    source: &eredu_core::capture::SharedCapturePlan,
    interventions: Option<&eredu_runtime::working_memory::OriginalInterventionSource>,
) -> Result<u64, Error> {
    let requirements = collector_requirements(source, interventions)?;
    let (descriptors, publications) = (requirements.descriptors, requirements.publications);
    if publications == 0 {
        return Ok(0);
    }
    // The prepared leaf owns one snapshot and the fixed native Selection. A
    // simultaneous revalidation owns one more snapshot; each source rank is
    // bounded by the selected program's 32 axes. Shape Vec payload is explicit.
    let controls = size_of::<SettledCapture>()
        .checked_add(size_of::<CaptureAttempt<'_>>())
        // The lexical generated producer also keeps its final native descriptor
        // beside the retained collector slots; runtime's generic T=() control
        // measurement deliberately leaves this native payload to its caller.
        .and_then(|n| n.checked_add(size_of::<Option<Array>>()))
        .and_then(|n| n.checked_add(size_of::<crate::backend::array_copy::CandidateExtraction>()))
        .and_then(|n| n.checked_add(size_of::<(Array, Array)>()))
        .and_then(|n| {
            n.checked_add(size_of::<
                eredu_runtime::working_memory::ScheduledCaptureCandidatesTransfer<'_, '_, '_, u8>,
            >())
        })
        .and_then(|n| n.checked_add(size_of::<PreparedCaptureFragment<'_, '_, '_, '_>>()))
        .and_then(|n| n.checked_add(size_of::<safemlx::ArrayMetadataSnapshot>()))
        .and_then(|n| n.checked_add(size_of::<RetainedStorage>()))
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_add(carrier_owner_control_bytes()?))
        // collector_counts builds these fixed values during both quotation and
        // once-only carrier construction. The frames are reused across rows;
        // numerical controls remain in each actual observed span's population.
        .and_then(|n| n.checked_add(requirements.summary_controls))
        .and_then(|n| n.checked_add(requirements.histogram_controls))
        .and_then(|n| n.checked_add(collector_requirement_control_bytes()?))
        .and_then(|n| n.checked_add(routed::control_bytes()?))
        // Inline retained aliases are already in Carrier/SettledCapture. This
        // additional named report covers native/Rust observation call controls;
        // it allocates no Scope, carrier, event or record and grants no arena fit.
        .and_then(|n| n.checked_add(safemlx::OriginalScopeObserver::control_bytes()?))
        .and_then(|n| n.checked_add(2 * 32 * size_of::<i32>()))
        .and_then(|n| n.checked_add(descriptors.checked_mul(size_of::<Array>())?))
        .and_then(|n| {
            n.checked_add(publications.checked_mul(size_of::<RetainedStoragePublication>())?)
        })
        .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
    u64::try_from(controls).map_err(|_| error(WorkingMemoryError::Overflow))
}

// Count the Box payload once, plus its preparation and Box::new moves.
// The other terms cover owner/result and unboxing controls, not another
// carrier allocation.
fn carrier_owner_control_bytes() -> Option<usize> {
    size_of::<CaptureCarrier>() // existing Box payload
        .checked_add(size_of::<CaptureCarrier>())? // prepared value
        .checked_add(size_of::<CaptureCarrier>())? // Box::new parameter
        .checked_add(size_of::<CaptureCarrierOwner>())?
        .checked_add(size_of::<Option<CaptureCarrierOwner>>())?
        .checked_add(size_of::<Result<CaptureCarrierOwner, Error>>())?
        .checked_add(size_of::<Option<Box<CaptureCarrier>>>())?
        .checked_add(size_of::<Box<CaptureCarrier>>())?
        .checked_add(size_of::<CaptureCarrier>())? // unbox return
        .checked_add(size_of::<CaptureCarrier>()) // payload destruction argument
}

fn summary_collector_control_bytes(
    source: &eredu_core::capture::SharedCapturePlan,
) -> Option<usize> {
    use crate::backend::array_copy::{
        CaptureNativePopulation, CaptureTensorNativeError, PreparedCaptureSummary,
    };
    use eredu_core::capture::{CapturePhase, CaptureSummaryGeometry, CaptureTensorGeometryError};
    if !source
        .admission()
        .plan()
        .selections
        .iter()
        .zip(source.admission().points())
        .any(|(selection, point)| {
            point.prefill
                && selection.schedule.includes(CapturePhase::Prefill, 0)
                && matches!(
                    selection.transform,
                    eredu_core::capture::CaptureTransform::Summary
                )
        })
    {
        return Some(0);
    }
    let frames = [
        size_of::<CaptureSummaryGeometry<'static>>(),
        size_of::<Result<CaptureSummaryGeometry<'static>, CaptureTensorGeometryError>>(),
        size_of::<PreparedCaptureSummary>(),
        size_of::<Result<PreparedCaptureSummary, CaptureTensorNativeError>>(),
        size_of::<Option<CaptureNativePopulation>>(),
        size_of::<Result<usize, Error>>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

fn histogram_collector_control_bytes(
    source: &eredu_core::capture::SharedCapturePlan,
) -> Option<usize> {
    use crate::backend::array_copy::{
        CaptureNativePopulation, CaptureTensorNativeError, PreparedCaptureHistogram,
    };
    use eredu_core::capture::{CaptureHistogramGeometry, CapturePhase, CaptureTensorGeometryError};
    if !source
        .admission()
        .plan()
        .selections
        .iter()
        .zip(source.admission().points())
        .any(|(selection, point)| {
            point.prefill
                && selection.schedule.includes(CapturePhase::Prefill, 0)
                && matches!(
                    selection.transform,
                    eredu_core::capture::CaptureTransform::Histogram { .. }
                )
        })
    {
        return Some(0);
    }
    let frames = [
        size_of::<CaptureHistogramGeometry<'static>>(),
        size_of::<Result<CaptureHistogramGeometry<'static>, CaptureTensorGeometryError>>(),
        size_of::<PreparedCaptureHistogram<'static>>(),
        size_of::<Result<PreparedCaptureHistogram<'static>, CaptureTensorNativeError>>(),
        size_of::<Option<CaptureNativePopulation>>(),
        size_of::<Result<usize, Error>>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

fn selection_descriptors(transform: &eredu_core::capture::CaptureTransform) -> usize {
    if matches!(
        transform,
        eredu_core::capture::CaptureTransform::TopCandidates { .. }
    ) {
        return 1 + crate::backend::array_copy::CandidateExtraction::ROOTS;
    }
    4 + 2 * usize::from(matches!(
        transform,
        eredu_core::capture::CaptureTransform::Preview { .. }
    ))
}
// One collector retains roots from the ordinary capture and every exact
// before/after companion admitted with that original intervention source.
// This only sizes host slots; the segment and claims separately authenticate
// the source and preserve the parent's cumulative evidence allowances.
#[derive(Default)]
struct CollectorRequirements {
    descriptors: usize,
    publications: usize,
    summary_controls: usize,
    histogram_controls: usize,
}
impl CollectorRequirements {
    fn include(&mut self, source: &eredu_core::capture::SharedCapturePlan) -> Result<(), Error> {
        let (descriptors, publications) = collector_counts(source)?;
        self.descriptors = self
            .descriptors
            .checked_add(descriptors)
            .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
        self.publications = self
            .publications
            .checked_add(publications)
            .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
        // These source-inspection frames are reused serially. Roots and
        // publication slots, in contrast, stay live together through retirement.
        self.summary_controls = self.summary_controls.max(
            summary_collector_control_bytes(source)
                .ok_or_else(|| error(WorkingMemoryError::Overflow))?,
        );
        self.histogram_controls = self.histogram_controls.max(
            histogram_collector_control_bytes(source)
                .ok_or_else(|| error(WorkingMemoryError::Overflow))?,
        );
        Ok(())
    }
}
fn collector_requirements(
    source: &eredu_core::capture::SharedCapturePlan,
    interventions: Option<&eredu_runtime::working_memory::OriginalInterventionSource>,
) -> Result<CollectorRequirements, Error> {
    let mut requirements = CollectorRequirements::default();
    requirements.include(source)?;
    if let Some(original) = interventions {
        for operation in 0..original.plan().admission().plan().operations.len() {
            if let Some(companion) = original.plan().evidence(operation) {
                requirements.include(companion.shared_geometry_source())?;
            }
        }
    }
    Ok(requirements)
}
fn collector_requirement_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<CollectorRequirements>() * 3,
        size_of::<Result<CollectorRequirements, Error>>(),
        size_of::<(
            &eredu_core::capture::SharedCapturePlan,
            Option<&eredu_runtime::working_memory::OriginalInterventionSource>,
        )>() * 3,
        size_of::<Option<&eredu_core::capture::InterventionEvidenceCompanion>>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<(usize, usize)>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}
fn collector_counts(
    source: &eredu_core::capture::SharedCapturePlan,
) -> Result<(usize, usize), Error> {
    use eredu_core::capture::CapturePhase;
    let mut descriptors = 0usize;
    let mut publications = 0usize;
    for (index, (selection, point)) in source
        .admission()
        .plan()
        .selections
        .iter()
        .zip(source.admission().points())
        .enumerate()
    {
        if matches!(
            selection.transform,
            eredu_core::capture::CaptureTransform::RoutedUnits
        ) {
            let prefill = point.prefill && selection.schedule.includes(CapturePhase::Prefill, 0);
            let decode = point.decode
                && selection
                    .schedule
                    .count_and_last(
                        CapturePhase::Decode,
                        source.admission().request().max_predictions,
                    )
                    .map_err(error)?
                    .is_some();
            if !prefill && !decode {
                continue;
            }
            // The shared sparse progression requires disjoint nonempty provider
            // ranges. Thus at most one batch per logical token can reach this
            // selection in a physical chunk. Cold native populations count
            // actual callbacks separately; these are only fixed collector slots.
            let positions = source.admission().invocation_bounds().map_or(
                if prefill {
                    source.admission().request().prompt_tokens
                } else {
                    1
                },
                |bounds| bounds.max_sequence,
            );
            let tokens = source
                .admission()
                .request()
                .batch
                .checked_mul(positions)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
            let slots = tokens
                .checked_mul(5)
                .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
            descriptors = descriptors
                .checked_add(slots)
                .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
            publications = publications
                .checked_add(slots)
                .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
            continue;
        }
        if !point.prefill || !selection.schedule.includes(CapturePhase::Prefill, 0) {
            continue;
        }
        // One source ingress root, leaf source+slice, optional reshape/prefix,
        // and a conservative nonempty half cast. Dtype/zero specialization can
        // lower actual usage but never raises this admitted representation bound.
        // The admitted retained factory protocol has exactly two compact
        // operands and seven outputs. The catalog does not classify whether a
        // hook emits that protocol, so reserve that bounded descriptor envelope
        // for each eligible selection. Shared factories consume it only once;
        // this overcounts handles, never numerical bytes or admission authority.
        let count = if matches!(
            selection.transform,
            eredu_core::capture::CaptureTransform::TokenScores { .. }
        ) {
            let geometry = eredu_core::capture::CaptureTokenScoreGeometry::prepare(
                source.admission(),
                index,
                CapturePhase::Prefill,
                0,
                None,
            )
            .map_err(error)?;
            crate::backend::array_copy::TokenScoreProgram::from_geometry(&geometry)
                .and_then(|p| p.population())
                .map_err(error)?
                .retained_outputs
                .checked_add(1)
                .ok_or_else(|| error(WorkingMemoryError::Overflow))?
        } else if matches!(
            selection.transform,
            eredu_core::capture::CaptureTransform::Summary
        ) {
            let geometry = eredu_core::capture::CaptureSummaryGeometry::prepare(
                source.admission(),
                index,
                CapturePhase::Prefill,
                0,
                None,
            )
            .map_err(error)?;
            let program =
                crate::backend::array_copy::PreparedCaptureSummary::from_geometry(&geometry)
                    .map_err(error)?;
            // Each carrier belongs to one physical chunk. Its selected values
            // are a subset of this exact logical selection; the shared summary
            // worker has a fixed root population per nonempty 1024-value chunk.
            // Thus the complete selected geometry bounds every physical fragment,
            // including strided selections and a shorter terminal chunk. The
            // same count prices the Vec and constructs it before native work.
            program
                .population()
                .ok_or_else(|| error(WorkingMemoryError::Overflow))?
                .retained_roots
        } else if matches!(
            selection.transform,
            eredu_core::capture::CaptureTransform::Histogram { .. }
        ) {
            let geometry = eredu_core::capture::CaptureHistogramGeometry::prepare(
                source.admission(),
                index,
                CapturePhase::Prefill,
                0,
                None,
            )
            .map_err(error)?;
            let program =
                crate::backend::array_copy::PreparedCaptureHistogram::from_geometry(&geometry)
                    .map_err(error)?;
            // Each carrier belongs to one physical chunk. Its selected values
            // are a subset of this exact logical selection; the shared histogram
            // worker has a fixed root population per nonempty 1024-value chunk.
            // Thus the complete selected geometry bounds every physical fragment,
            // including strided selections and a shorter terminal chunk. The
            // same count prices the Vec and constructs it before native work.
            program
                .population()
                .ok_or_else(|| error(WorkingMemoryError::Overflow))?
                .retained_roots
        } else {
            selection_descriptors(&selection.transform)
                + if matches!(
                    selection.transform,
                    eredu_core::capture::CaptureTransform::TopCandidates { .. }
                ) {
                    0
                } else {
                    9
                }
        };
        descriptors = descriptors
            .checked_add(count)
            .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
        publications = publications
            .checked_add(1)
            .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
    }
    Ok((descriptors, publications))
}

impl FundedWork {
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_is_active(
        &self,
    ) -> Result<bool, Error> {
        Ok(self.capture.try_borrow().map_err(error)?.is_some())
    }
    pub(in crate::composition::mlx::session::model_session::text_funding) fn require_no_capture_carrier(
        &self,
    ) -> Result<(), Error> {
        if self.capture_is_active()? {
            return Err(error(CaptureCarrierError::Incomplete));
        }
        Ok(())
    }
    pub(in crate::composition::mlx::session::model_session::text_funding) fn include_capture_roots(
        &self,
        storage: &mut RetainedStorage,
    ) -> Result<(), Error> {
        let capture = self.capture.try_borrow().map_err(error)?;
        if let Some(capture) = capture.as_ref() {
            let roots = capture.roots.try_borrow().map_err(error)?;
            for root in roots.iter() {
                storage.include_array(root)?;
            }
        }
        Ok(())
    }
    pub(in crate::composition::mlx::session::model_session::text_funding) fn prepare_capture_chunk(
        &self,
        bootstrap: CapturePrefillSourceBootstrap<'_>,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<PreparedPrefillChunkRetention, Error> {
        let mut capture = self.capture.try_borrow_mut().map_err(error)?;
        if capture.is_some() {
            return Err(error(CaptureCarrierError::Identity));
        }
        let mut scope = self.scope.try_borrow_mut().map_err(error)?;
        let scope = scope
            .as_mut()
            .ok_or_else(|| error(WorkingMemoryError::ExecutionFenced))?;
        // Allocate measured control before bootstrap can install the native slot.
        // A failed/partial bootstrap leaves this carrier in original recovery.
        *capture = Some(CaptureCarrierOwner::prepared(
            bootstrap.source(),
            bootstrap.intervention_source(),
        )?);
        self.published.set(false);
        let (segment, registration) = bootstrap.begin_segment(scope, context).map_err(error)?;
        let capture: &mut CaptureCarrier = capture.as_mut().expect("installed carrier");
        capture.segment = Some(segment);
        capture.progress.set(Progress::Ready);
        if let Some(rows) = &self.opening_rows {
            rows.pin_opening(
                context,
                scope,
                capture.segment.as_ref().expect("installed segment"),
            )?;
        }
        Ok(registration)
    }
    pub(in crate::composition::mlx::session::model_session::text_funding) fn retire_capture_chunk(
        &self,
        ticket: SettledPrefillChunkRetention,
    ) -> Result<(), Error> {
        self.published.set(false);
        let mut capture_slot = self.capture.try_borrow_mut().map_err(error)?;
        let capture: &mut CaptureCarrier = capture_slot
            .as_mut()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        let mut scope_loan = self.scope.try_borrow_mut().map_err(error)?;
        let scope = scope_loan
            .as_mut()
            .ok_or_else(|| error(WorkingMemoryError::ExecutionFenced))?;
        let mut roots = capture.roots.try_borrow_mut().map_err(error)?;
        let mut publications = capture.publications.try_borrow_mut().map_err(error)?;
        if capture.progress.get() != Progress::Ready {
            return Err(error(CaptureCarrierError::Incomplete));
        }
        let segment = capture
            .segment
            .as_ref()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        segment
            .validate_settled_ticket(scope, &ticket)
            .map_err(error)?;
        if !safemlx::can_reclaim_submission_resources() {
            return Err(error(CaptureCarrierError::RetirementBoundary));
        }
        #[cfg(test)]
        if FORCE_RETIREMENT_BUSY.replace(false) {
            return Err(error(CaptureCarrierError::RetirementBusy));
        }
        retire_capture_records(capture.native_ancestor.as_ref())?;
        // Every fallible caller loan and native progress check precedes the
        // one-use slot take. Its own atomic source validation may still reject.
        let (opening, parcel) = match &self.opening_rows {
            Some(rows) => {
                rows.publish_completed(&ticket, scope, segment)?;
                let (row, parcel) = rows.retire_published(&ticket, || {
                    // Keep the native parcel error typed until the rows loan
                    // ends; the complete owning envelope is outside this carrier.
                    segment.take_settled_sources(scope, &ticket)
                })?;
                (Some(row), parcel)
            }
            None => (
                None,
                segment
                    .take_settled_sources(scope, &ticket)
                    .map_err(error)?,
            ),
        };
        let payload = CapturePayload {
            opening,
            roots: take(&mut *roots),
            publications: take(&mut *publications),
        };
        drop(roots);
        drop(publications);
        let segment = capture.segment.take();
        let native_ancestor = capture.native_ancestor.take();
        let empty_carrier = capture_slot.take();
        let retired = SettledCapture {
            payload: Some(payload),
            native_ancestor,
            parcel: Some(parcel),
            segment,
            ticket: Some(ticket),
        };
        drop(capture_slot);
        // Drop the actual RefMut, not merely the borrowed scope reference.
        drop(scope_loan);
        drop(empty_carrier);
        retired.retire();
        Ok(())
    }

    /// Private until original p0 progression supplies the actual claim and the
    /// original quote covers this full native span. This never assumes reuse
    /// credit, reduces a bound or starts another scope. Source remains borrowed
    /// and immutable through settlement. The bound observer caller additionally
    /// requires the original full-span capacity proof before evaluation.
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_prefill_fragment(
        &self,
        source: &Array,
        claim: CapturePrefillFragmentClaim<'_, '_, '_, '_>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.capture_prefill_fragment_with_completion(
            source,
            claim,
            stream,
            CaptureCompletion::Ordinary,
        )
    }
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_prefill_fragment_with_completion(
        &self,
        source: &Array,
        claim: CapturePrefillFragmentClaim<'_, '_, '_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
    ) -> Result<(), Error> {
        self.validate_capture_completion(completion)?;
        let _activity = publication_scope::Activity::begin(&self.publishing)?;
        let mut capture_slot = self.capture.try_borrow_mut().map_err(error)?;
        let capture: &mut CaptureCarrier = capture_slot
            .as_mut()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        let mut scope_owner = publication_scope::OwnedScope::take(&self.scope)?;
        let scope = scope_owner.get_mut();
        let geometry = claim.fragment().assembly().logical_geometry();
        let selection = &geometry.admission().plan().selections[geometry.selection_index()];
        capture.require_capacity(selection_descriptors(&selection.transform))?;
        let segment = capture
            .segment
            .as_mut()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        segment
            .validate_prefill_fragment(claim.fragment())
            .map_err(error)?;
        segment.validate_native_scope(scope).map_err(error)?;
        PreparedCaptureFragment::validate_borrowed_source(source, claim.fragment())
            .map_err(error)?;
        // Reject unsupported rank/signed geometry before a metadata snapshot or
        // any possible evaluation. The same closed program is constructed below.
        crate::backend::array_copy::CaptureTensorSelection::from_fragment(claim.fragment())
            .map_err(error)?;
        PreparedCaptureFragment::validate_source_geometry(source, claim.fragment())
            .map_err(error)?;
        PreparedCaptureTensor::validate_stream(stream).map_err(error)?;
        let attempt = CaptureAttempt::new(&capture.progress)?;
        let retained = completion.clone_array(source).map_err(error)?;
        capture.roots.borrow_mut().push(retained);
        self.published.set(false);
        drop(completion.settle(source, stream).map_err(error)?);
        segment.validate_native_scope(scope).map_err(error)?;
        let publication = self.publish_capture_source(source, scope, completion)?;
        let _publication = match completion {
            CaptureCompletion::Original(_) => Some(publication),
            CaptureCompletion::Ordinary => {
                capture.publications.borrow_mut().push(publication);
                None
            }
        };
        let prepared = PreparedCaptureFragment::new(source, claim.fragment()).map_err(error)?;
        prepared
            .transfer_with_completion(
                claim,
                scope,
                segment,
                stream,
                &capture.roots,
                completion,
                self.original_metadata_custody().as_ref(),
            )
            .map_err(error)?;
        attempt.complete();
        Ok(())
    }
}

#[cfg(test)]
impl FundedWork {
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_state_for_test(
        &self,
    ) -> Result<Option<(usize, usize, bool)>, Error> {
        let capture = self.capture.try_borrow().map_err(error)?;
        capture
            .as_ref()
            .map(|c| {
                Ok((
                    c.roots.try_borrow().map_err(error)?.len(),
                    c.publications.try_borrow().map_err(error)?.len(),
                    c.progress.get() != Progress::Ready,
                ))
            })
            .transpose()
    }
    /// Test-only composition over an already materialized, independently
    /// registered F32 source. The fixture establishes settlement before entry;
    /// no cast, source allocation or fabricated chunk/claim authority occurs.
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_settled_external<
        'a,
        'c,
    >(
        &self,
        source: &Array,
        claim: CaptureTensorClaim<'a, 'c>,
        stream: &Stream,
    ) -> Result<ClaimedCaptureTensor, Error> {
        use crate::backend::runtime::residency::storage::StorageIdentity;
        let mut capture_slot = self.capture.try_borrow_mut().map_err(error)?;
        let capture: &mut CaptureCarrier = capture_slot
            .as_mut()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        let mut scope_loan = self.scope.try_borrow_mut().map_err(error)?;
        let scope = scope_loan
            .as_mut()
            .ok_or_else(|| error(WorkingMemoryError::ExecutionFenced))?;
        capture.require_capacity(1)?;
        let segment = capture
            .segment
            .as_mut()
            .ok_or_else(|| error(CaptureCarrierError::Identity))?;
        claim.validate_native_scope(scope).map_err(error)?;
        segment.validate_native_scope(scope).map_err(error)?;
        let observed = PreparedCaptureTensor::validate_source_geometry(source, claim.geometry())
            .map_err(error)?;
        PreparedCaptureTensor::validate_stream(stream).map_err(error)?;
        if observed.dtype() != safemlx::Dtype::Float32 {
            return Err(error(CaptureCarrierError::Identity));
        }
        let allocation = observed
            .allocation()
            .ok_or_else(|| error(WorkingMemoryError::UnknownBound))?;
        let bytes =
            u64::try_from(allocation.bytes()).map_err(|_| error(WorkingMemoryError::Overflow))?;
        let prepared =
            crate::backend::runtime::residency::storage::prepare_funded_storage_publication(
                scope, 1,
            )
            .map_err(error)?;
        let source_pin = prepared
            .pin_registered_storage([(StorageIdentity::Native(allocation.identity()), bytes)])
            .map_err(error)?;
        let attempt = CaptureAttempt::new(&capture.progress)?;
        capture
            .roots
            .borrow_mut()
            .push(source.try_clone_for_inspection().map_err(error)?);
        self.published.set(false);
        let mut storage = RetainedStorage::default();
        storage.include_array(source)?;
        capture
            .publications
            .borrow_mut()
            .push(storage.publish_funded(scope)?);
        let mut destination = claim
            .prepare_with_segment_source(scope, segment, source_pin)
            .map_err(error)?;
        let settled = source.evaluated()?;
        destination.validate().map_err(error)?;
        for value in settled.try_iter::<f32>().map_err(error)? {
            destination.push_f32(value).map_err(error)?;
        }
        let result = destination
            .finish()
            .map_err(|e| error(e.into_owned_error()))?;
        attempt.complete();
        Ok(result)
    }
}

#[cfg(test)]
thread_local! { static FORCE_RETIREMENT_BUSY: Cell<bool> = const { Cell::new(false) }; }
#[cfg(test)]
impl FundedWork {
    /// Failure-only test seam. Successful paths always call actual native B.
    pub(in crate::composition::mlx::session::model_session::text_funding) fn force_capture_retirement_busy_for_test(
    ) {
        FORCE_RETIREMENT_BUSY.set(true);
    }
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::{FundingFixture as _, StorageFixture as _};
