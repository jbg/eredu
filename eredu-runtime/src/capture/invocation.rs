//! Independently shaped forwards use the existing capture owner and ledger.
use super::*;

/// Explicit applicability within the already admitted selections/operations.
/// Masks are in admission order. None enables every phase-eligible entry; it
/// never discovers or implicitly selects additional capture points.
#[derive(Clone, Copy, Default)]
pub struct CaptureInvocationSelection<'a> {
    /// Applicability of each admitted capture selection.
    pub captures: Option<&'a [bool]>,
    /// Applicability of each admitted intervention operation.
    pub interventions: Option<&'a [bool]>,
}
impl CaptureInvocationSelection<'_> {
    fn capture(self, index: usize) -> bool {
        self.captures.is_none_or(|mask| mask[index])
    }
    fn intervention(self, index: usize) -> bool {
        self.interventions.is_none_or(|mask| mask[index])
    }
}

pub(super) const INVOCATION_METADATA: CaptureUsage = CaptureUsage {
    captures: 0,
    retained_bytes: 0,
    host_bytes: 64,
    encoded_bytes: 192,
};
impl CaptureSession {
    /// Resolves the current invocation's physical axes without using prediction
    /// position as a sequence width or inferring a prediction-local cache length.
    pub(crate) fn tensor_geometry(&self) -> Result<CaptureInvocationShape, CaptureError> {
        self.plan
            .geometry_at(self.phase, self.prediction, self.invocation)
    }

    /// Admits one actual forward before capture work. Reusing a prediction
    /// coordinate starts another charged invocation; restore never refunds it.
    pub fn begin_invocation(
        &mut self,
        phase: CapturePhase,
        prediction: u64,
        shape: CaptureInvocationShape,
        selection: CaptureInvocationSelection<'_>,
    ) -> Result<(), CaptureError> {
        self.begin_invocation_with_window(phase, prediction, shape, selection, None)
    }

    /// Begins one attributed row window using the original selection and shared
    /// cumulative ledger. It does not authorize new captures or managed work.
    pub fn begin_invocation_window(
        &mut self,
        phase: CapturePhase,
        prediction: u64,
        shape: CaptureInvocationShape,
        selection: CaptureInvocationSelection<'_>,
        window: CaptureInvocationWindow,
    ) -> Result<(), CaptureError> {
        self.begin_invocation_with_window(phase, prediction, shape, selection, Some(window))
    }

    fn begin_invocation_with_window(
        &mut self,
        phase: CapturePhase,
        prediction: u64,
        shape: CaptureInvocationShape,
        selection: CaptureInvocationSelection<'_>,
        window: Option<CaptureInvocationWindow>,
    ) -> Result<(), CaptureError> {
        let host = self.ordinary_error_custody.clone();
        let result = self
            .begin_invocation_with_window_unretained(phase, prediction, shape, selection, window);
        ordinary_error::retain_result(host, result)
    }

    fn begin_invocation_with_window_unretained(
        &mut self,
        phase: CapturePhase,
        prediction: u64,
        shape: CaptureInvocationShape,
        selection: CaptureInvocationSelection<'_>,
        window: Option<CaptureInvocationWindow>,
    ) -> Result<(), CaptureError> {
        let bounds = self.plan.invocation_bounds().ok_or_else(|| {
            CaptureError::Invalid(
                "ordinary capture authority cannot admit independent invocations".into(),
            )
        })?;
        bounds.validate(shape, prediction)?;
        let logical = window
            .map(|window| window.validate(shape))
            .transpose()?
            .unwrap_or(shape);
        bounds.validate(logical, prediction)?;
        if window.is_some() && self.partition.is_some() {
            return Err(CaptureError::Unsupported(
                "row-window plus native partition projection is not composed".into(),
            ));
        }
        if self.records.is_some() || self.transaction.is_some() {
            return Err(CaptureError::Invalid(
                "invocation requires a drained capture step".into(),
            ));
        }
        if selection
            .captures
            .is_some_and(|mask| mask.len() != self.plan.points().len())
            || selection.interventions.is_some_and(|mask| {
                mask.len()
                    != self
                        .interventions
                        .as_ref()
                        .map_or(0, |run| run.plan.points().len())
            })
        {
            return Err(CaptureError::Invalid(
                "invocation applicability does not match admission".into(),
            ));
        }
        for (index, (entry, point)) in self
            .plan
            .plan()
            .selections
            .iter()
            .zip(self.plan.points())
            .enumerate()
        {
            if !selection.capture(index) || !entry.schedule.includes(phase, prediction) {
                continue;
            }
            logical.validate_slices(point, &entry.slices)?;
            if window.is_some()
                && !matches!(
                    entry.transform,
                    CaptureTransform::Slice
                        | CaptureTransform::FullTensor
                        | CaptureTransform::RoutedUnits
                )
                && !(self.window_reductions
                    && matches!(
                        entry.transform,
                        CaptureTransform::Summary
                            | CaptureTransform::Histogram { .. }
                            | CaptureTransform::Preview { .. }
                    ))
            {
                return Err(CaptureError::Unsupported("split invocation requires a composed row-fragment collector; selected aggregation is unfinished".into()));
            }
            if let Some(actual) = logical.resolve(point)? {
                resolve_slice(point, entry, &actual)?;
            }
            if window.is_some() {
                // Only inspect borrowed declarations here. Actual global shape,
                // slice and fragment construction follows the metadata ledger
                // reservation in capture_window_value.
                let axes = point.axes.as_ref().ok_or_else(|| {
                    CaptureError::Unsupported("window capture requires declared row axes".into())
                })?;
                let mut rows = axes.iter().filter(|axis| {
                    matches!(
                        axis.dimension,
                        eredu_core::SymbolicDimension::Sequence
                            | eredu_core::SymbolicDimension::TokenRows
                    )
                });
                if rows.next().is_none() {
                    return Err(CaptureError::Unsupported(
                        "window capture has no declared row axis".into(),
                    ));
                }
                if rows.next().is_some() {
                    return Err(CaptureError::Unsupported(
                        "window capture requires one declared row axis".into(),
                    ));
                }
                // Window validation already proves start/end <= this extent.
                usize::try_from(logical.sequence).map_err(|_| CaptureError::Overflow)?;
            }
        }
        if let Some(run) = &self.interventions {
            for (index, (operation, point)) in run
                .plan
                .plan()
                .operations
                .iter()
                .zip(run.plan.points())
                .enumerate()
            {
                if !selection.intervention(index) || !operation.schedule.includes(phase, prediction)
                {
                    continue;
                }
                let point = point.observation_geometry();
                logical.validate_slices(&point, &operation.slices)?;
                if let Some(actual) = logical.resolve(&point)? {
                    run.plan.validate_at(
                        index,
                        phase,
                        prediction,
                        Some(logical),
                        &actual,
                        operation.action.dtype(),
                    )?;
                }
            }
        }
        self.invocation = Some(shape);
        self.invocation_window = window;
        self.begin_step_inner(phase, prediction)?;
        for (index, record) in self.records.iter_mut().flatten().enumerate() {
            if !selection.capture(index) && record.outcome == CaptureOutcome::Missing {
                record.outcome = CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::NotInvoked,
                };
            }
        }
        if let Some(run) = &mut self.interventions {
            for (index, record) in run.records.iter_mut().flatten().enumerate() {
                if !selection.intervention(index) {
                    record.outcome = eredu_core::intervention::InterventionOutcome::Inactive;
                    for evidence in &mut record.evidence {
                        if evidence.outcome == CaptureOutcome::Missing {
                            evidence.outcome = CaptureOutcome::Skipped {
                                reason: CaptureSkipReason::NotInvoked,
                            };
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Independent phases have no predetermined invocation count. Check one worst
/// case step; the shared cumulative ledger admits every actual repetition.
pub(super) fn preflight_invocations(
    plan: &AdmittedCapturePlan,
    extra: &[(CaptureSelection, eredu_core::ObservationPoint)],
    base: CaptureUsage,
    scheduled_costs: &[(CaptureSchedule, [CaptureUsage; 2])],
    inherited: CaptureUsage,
    mut estimate: impl FnMut(
        &[u64],
        &CaptureSelection,
        &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError>,
) -> Result<(), CaptureError> {
    let has_work = !plan.is_empty() || !extra.is_empty() || !scheduled_costs.is_empty()
        || base != CaptureUsage::default();
    let mut budget = InvocationPreflightBudget::new(plan, base, inherited, has_work)?;
    if !has_work { return budget.finish(); }
    let bounds = plan.invocation_bounds().expect("invocation admission");
    let geometry = bounds.maximum()?;
    let active = |schedule: &CaptureSchedule, phase| {
        schedule
            .count_coordinates(phase, 0, bounds.max_predictions)
            .map(|range| range.is_some())
    };
    for (selection, point) in plan
        .plan()
        .selections
        .iter()
        .zip(plan.points())
        .chain(extra.iter().map(|(selection, point)| (selection, point)))
    {
        budget.add(metadata_reservation(selection, point)?)?;
        if (active(&selection.schedule, CapturePhase::Prefill)?
            || active(&selection.schedule, CapturePhase::Decode)?)
            && plan.plan().limits.on_limit == CaptureLimitPolicy::Fail
        {
            if let Some(shape) = geometry.resolve(point)? {
                let slice = resolve_slice(point, selection, &shape)?;
                budget.add(estimate(&shape, selection, &slice)?)?;
            }
        }
    }
    for (schedule, costs) in scheduled_costs {
        let a = if active(schedule, CapturePhase::Prefill)? {
            costs[0]
        } else {
            CaptureUsage::default()
        };
        let b = if active(schedule, CapturePhase::Decode)? {
            costs[1]
        } else {
            CaptureUsage::default()
        };
        budget.add(CaptureUsage {
            captures: a.captures.max(b.captures),
            retained_bytes: a.retained_bytes.max(b.retained_bytes),
            host_bytes: a.host_bytes.max(b.host_bytes),
            encoded_bytes: a.encoded_bytes.max(b.encoded_bytes),
        })?;
    }
    budget.finish()
}

/// Same one-invocation budget for ordinary dynamic and original fixed scratch
/// producers. Repeated coordinates are charged later by the actual live ledger.
pub(crate) struct InvocationPreflightBudget<'a> {
    plan: &'a AdmittedCapturePlan,
    step: CaptureUsage,
    inherited: CaptureUsage,
}
impl<'a> InvocationPreflightBudget<'a> {
    pub(crate) fn new(plan: &'a AdmittedCapturePlan, base: CaptureUsage,
        inherited: CaptureUsage, has_work: bool) -> Result<Self, CaptureError>
    {
        Ok(Self { plan, step: if has_work { base.checked_add(INVOCATION_METADATA)? } else { base }, inherited })
    }
    pub(crate) fn reserves_selections(&self) -> bool {
        self.plan.plan().limits.on_limit == CaptureLimitPolicy::Fail
    }
    pub(crate) fn add(&mut self, cost: CaptureUsage) -> Result<(), CaptureError> {
        self.step = self.step.checked_add(cost)?;
        Ok(())
    }
    pub(crate) fn finish(self) -> Result<(), CaptureError> {
        if let Some(budget) = self.step.exceeded(self.plan.plan().limits.per_step) {
            return Err(CaptureError::Limit { budget, cumulative: false });
        }
        if let Some(budget) = self.inherited.checked_add(self.step)?.exceeded(self.plan.plan().limits.cumulative) {
            return Err(CaptureError::Limit { budget, cumulative: true });
        }
        Ok(())
    }
}
