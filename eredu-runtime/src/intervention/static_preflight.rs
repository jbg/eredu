//! Fixed original scratch around the same schedule and budget preflight.
use super::*;
use eredu_core::HostPreparationAuthority;
use std::mem::{size_of, size_of_val};
const RANK: usize = 32;
mod invocation;
/// Fixed allocation/refusal keeps the actual scratch-construction custody.
#[derive(Debug, thiserror::Error)]
#[error("prepared intervention scratch construction failed")]
pub struct StaticInterventionScratchError {
    #[source]
    cause: Option<std::collections::TryReserveError>,
    _host: HostPreparationAuthority,
}
/// Fixed preparation refusals preserve the original geometry or budget cause.
#[derive(Debug, thiserror::Error)]
pub enum StaticInterventionPreflightError {
    /// Shared budget/estimator refusal.
    #[error("{0}")]
    Capture(#[from] CaptureError),
    /// Different admission coordinates or mode.
    #[error("capture/intervention geometry mismatch")]
    Request,
    /// The actual immutable companion producer is not yet installed.
    #[error("prepared static evidence producer is not installed")]
    Profile,
    /// Static metadata exceeds the native rank contract.
    #[error("static intervention metadata exceeds the rank bound")]
    Rank,
    /// Static activation requires a declared exact dtype.
    #[error("static action requires an exact dtype")]
    Dtype,
    /// Source axis refusal from the shared resolver.
    #[error("{0}")]
    Axes(#[from] CaptureAxisError),
    /// Slice refusal from the shared resolver.
    #[error("{0}")]
    Slice(#[from] CaptureSliceDestinationError),
    /// Admitted edit geometry differs from the source.
    #[error("{0}")]
    Geometry(#[from] InterventionGeometryError),
}
/// Four exact rank destinations reused for every declaration/phase; source
/// shapes are fixed stack metadata. A host token retains paid storage only.
#[derive(Debug)]
pub struct StaticInterventionPreflight {
    slice: ResolvedCaptureSlice,
    _host: HostPreparationAuthority,
}
impl StaticInterventionPreflight {
    /// Exact scratch payload and constructor/query/worker control population.
    pub fn required_bytes() -> Option<usize> {
        let frames = [
            size_of::<Option<&SharedInterventionPlan>>(),
            size_of::<(
                &mut Self,
                &mut crate::capture::PreflightBudget<'_>,
                &AdmittedCapturePlan,
                CapturePhase,
                &dyn InterventionEstimator,
            )>(),
            size_of::<(
                &mut Self,
                &AdmittedCapturePlan,
                &SharedInterventionPlan,
                &dyn InterventionEstimator,
            )>(),
            size_of::<Option<&InterventionEvidenceCompanion>>(),
            size_of::<Result<(), StaticInterventionPreflightError>>(),
            size_of::<[u64; RANK]>(),
            size_of::<Self>(),
            size_of::<Result<Self, StaticInterventionScratchError>>(),
            size_of::<StaticInterventionScratchError>(),
            size_of::<[Vec<u64>; 4]>(),
            size_of::<[u64; RANK]>(),
            size_of::<crate::capture::PreflightBudget<'static>>(),
            invocation::control_bytes()?,
            size_of::<[CaptureUsage; 4]>(),
            size_of::<[CapturePhase; 2]>(),
            size_of::<Result<(), CaptureError>>(),
            size_of::<CaptureAxisError>(),
            size_of::<StaticInterventionPreflightError>(),
            size_of::<Result<(), StaticInterventionPreflightError>>(),
            size_of::<(
                &Self,
                &AdmittedCapturePlan,
                &AdmittedInterventionPlan,
                &dyn InterventionEstimator,
            )>(),
            size_of::<
                std::iter::Zip<
                    std::slice::Iter<'_, InterventionOperation>,
                    std::slice::Iter<'_, InterventionPoint>,
                >,
            >(),
            size_of::<
                std::iter::Zip<
                    std::slice::Iter<'_, CaptureSelection>,
                    std::slice::Iter<'_, eredu_core::ObservationPoint>,
                >,
            >(),
            size_of::<Result<bool, CaptureAxisError>>(),
            size_of::<Result<(), InterventionGeometryError>>(),
            size_of::<[&InterventionPoint; 2]>(),
            size_of::<[&InterventionOperation; 2]>(),
            size_of::<[&eredu_core::ObservationPoint; 2]>(),
            size_of::<[&CaptureSelection; 2]>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?
            .checked_add(4usize.checked_mul(RANK)?.checked_mul(size_of::<u64>())?)?
            // The shared ordinary origin check / estimator-contract refusal can
            // allocate one of these exact fixed diagnostics. Metadata arithmetic
            // and all prepared geometry refusals are allocation-free.
            .checked_add(
                "intervention admission does not yet retain ordinary cached text origin"
                    .len()
                    .max("activation estimate must exclude capture and record encoding".len()),
            )
    }
    /// Construct only after its exact extent has been paid by the caller's
    /// authenticated request. Existing caller buffers are never adopted.
    pub fn prepare(host: HostPreparationAuthority) -> Result<Self, StaticInterventionScratchError> {
        let mut buffers: [Vec<u64>; 4] = std::array::from_fn(|_| Vec::new());
        for buffer in &mut buffers {
            if let Err(cause) = buffer.try_reserve_exact(RANK) {
                return Err(StaticInterventionScratchError {
                    cause: Some(cause),
                    _host: host,
                });
            }
            if buffer.capacity() != RANK {
                return Err(StaticInterventionScratchError {
                    cause: None,
                    _host: host,
                });
            }
        }
        let [starts, ends, strides, shape] = buffers;
        Ok(Self {
            slice: ResolvedCaptureSlice {
                starts,
                ends,
                strides,
                shape,
            },
            _host: host,
        })
    }
    fn rank(&mut self, rank: usize) -> Result<(), StaticInterventionPreflightError> {
        if rank > RANK {
            return Err(StaticInterventionPreflightError::Rank);
        }
        self.slice.starts.resize(rank, 0);
        self.slice.ends.resize(rank, 0);
        self.slice.strides.resize(rank, 0);
        self.slice.shape.resize(rank, 0);
        Ok(())
    }
    /// Validates a capture-only source using the same scheduled geometry and
    /// cumulative budget worker, without constructing an intervention source.
    pub fn run_capture(
        &mut self, capture: &AdmittedCapturePlan, estimator: &dyn InterventionEstimator,
    ) -> Result<(), StaticInterventionPreflightError> {
        if capture.invocation_bounds().is_some() { return Err(StaticInterventionPreflightError::Request); }
        let mut base = CaptureUsage::default();
        for (selection, point) in capture.plan().selections.iter().zip(capture.points()) {
            base = base.checked_add(crate::capture::metadata_reservation(selection, point)?)?;
        }
        let mut budget = crate::capture::PreflightBudget::new(capture, base, 0, CaptureUsage::default())?;
        for phase in [CapturePhase::Prefill, CapturePhase::Decode] {
            if budget.begin_phase(phase) {
                self.add_selections(&mut budget, capture, phase, estimator)?;
                budget.finish_phase()?;
            }
        }
        budget.finish()?;
        Ok(())
    }
    /// Existing static admission without source-owned evidence companions.
    pub fn run(
        &mut self,
        capture: &AdmittedCapturePlan,
        intervention: &AdmittedInterventionPlan,
        estimator: &dyn InterventionEstimator,
    ) -> Result<(), StaticInterventionPreflightError> {
        self.run_impl(capture, intervention, None, estimator)
    }
    /// Use the actual paid immutable companion inventory. Its zero limits grant
    /// no allowance: the enclosing capture plan supplies the only budget.
    pub fn run_with_source(
        &mut self,
        capture: &AdmittedCapturePlan,
        source: &SharedInterventionPlan,
        estimator: &dyn InterventionEstimator,
    ) -> Result<(), StaticInterventionPreflightError> {
        self.run_impl(capture, source.admission(), Some(source), estimator)
    }
    fn run_impl(
        &mut self,
        capture: &AdmittedCapturePlan,
        intervention: &AdmittedInterventionPlan,
        source: Option<&SharedInterventionPlan>,
        estimator: &dyn InterventionEstimator,
    ) -> Result<(), StaticInterventionPreflightError> {
        use StaticInterventionPreflightError as E;
        validate_capture_origin(capture, intervention)?;
        if capture.request() != intervention.request()
            || capture.invocation_bounds() != intervention.invocation_bounds()
            || intervention.invocation_bounds().is_some()
        {
            return Err(E::Request);
        }
        let mut base = CaptureUsage::default();
        for (index, (operation, point)) in intervention
            .plan()
            .operations
            .iter()
            .zip(intervention.points())
            .enumerate()
        {
            if point.routing.is_some() || point.routed_units.is_some() {
                return Err(E::Profile);
            }
            base = base.checked_add(intervention_metadata(
                operation,
                point,
                intervention.identity(),
            )?)?;
            match (
                &operation.evidence,
                source.and_then(|source| source.evidence(index)),
            ) {
                (InterventionEvidence::None, None) => (),
                (
                    InterventionEvidence::Preview { .. } | InterventionEvidence::Summary,
                    Some(companion),
                ) if companion.operation() == index
                    && companion.geometry_source().plan().selections.len() == 2 =>
                {
                    for (selection, point) in companion
                        .geometry_source()
                        .plan()
                        .selections
                        .iter()
                        .zip(companion.geometry_source().points())
                    {
                        base = base
                            .checked_add(crate::capture::metadata_reservation(selection, point)?)?;
                    }
                }
                _ => return Err(E::Profile),
            }
        }
        for (selection, point) in capture.plan().selections.iter().zip(capture.points()) {
            base = base.checked_add(crate::capture::metadata_reservation(selection, point)?)?;
        }
        let mut budget =
            crate::capture::PreflightBudget::new(capture, base, 0, CaptureUsage::default())?;
        for phase in [CapturePhase::Prefill, CapturePhase::Decode] {
            if !budget.begin_phase(phase) {
                continue;
            }
            for (index, (operation, point)) in intervention
                .plan()
                .operations
                .iter()
                .zip(intervention.points())
                .enumerate()
            {
                let Some((count, last)) = operation.schedule.count_and_last_from(
                    phase,
                    0,
                    intervention.request().max_predictions,
                )?
                else {
                    continue;
                };
                let mut shape = [0u64; RANK];
                self.rank(point.axes.len())?;
                let shape = &mut shape[..point.axes.len()];
                let actual = intervention.geometry_at(phase, last, None)?;
                if !actual.resolve_axes_into(&point.axes, shape)? {
                    continue;
                }
                intervention.resolve_prepared_at(
                    index,
                    phase,
                    last,
                    shape,
                    operation.action.dtype().ok_or(E::Dtype)?,
                    &mut self.slice,
                )?;
                estimator.validate_geometry(shape, &self.slice)?;
                budget.add(
                    activation_cost(estimator, shape, &self.slice, &operation.action)?,
                    count,
                )?;
                if let Some(companion) = source.and_then(|source| source.evidence(index)) {
                    self.add_selections(
                        &mut budget,
                        companion.geometry_source(),
                        phase,
                        estimator,
                    )?;
                }
            }
            self.add_selections(&mut budget, capture, phase, estimator)?;
            budget.finish_phase()?;
        }
        budget.finish()?;
        Ok(())
    }
    fn add_selections(
        &mut self,
        budget: &mut crate::capture::PreflightBudget<'_>,
        source: &AdmittedCapturePlan,
        phase: CapturePhase,
        estimator: &dyn InterventionEstimator,
    ) -> Result<(), StaticInterventionPreflightError> {
        for (selection, point) in source.plan().selections.iter().zip(source.points()) {
            let Some((count, last)) = selection.schedule.count_and_last_from(
                phase,
                0,
                source.request().max_predictions,
            )?
            else {
                continue;
            };
            let Some(axes) = point.axes.as_deref() else {
                continue;
            };
            let mut shape = [0u64; RANK];
            self.rank(axes.len())?;
            let shape = &mut shape[..axes.len()];
            let actual = source.geometry_at(phase, last, None)?;
            if !actual.resolve_axes_into(axes, shape)? {
                continue;
            }
            resolve_slice_into(
                Some(axes),
                &selection.slices,
                shape,
                &mut self.slice.starts,
                &mut self.slice.ends,
                &mut self.slice.strides,
                &mut self.slice.shape,
            )?;
            budget.add_selection(
                estimator.capture_usage(shape, selection, &self.slice)?,
                count,
            )?;
        }
        Ok(())
    }
}
