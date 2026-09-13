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
        match (self.plan.invocation_bounds(), self.invocation) {
            (Some(bounds), Some(shape)) => {
                bounds.validate(shape, self.prediction)?;
                Ok(shape)
            }
            (None, None) => self
                .plan
                .request()
                .invocation_shape(self.phase, self.prediction),
            _ => Err(CaptureError::Invalid(
                "capture invocation geometry is not active".into(),
            )),
        }
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
        let bounds = self.plan.invocation_bounds().ok_or_else(|| {
            CaptureError::Invalid(
                "ordinary capture authority cannot admit independent invocations".into(),
            )
        })?;
        bounds.validate(shape, prediction)?;
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
            shape.validate_slices(point, &entry.slices)?;
            if let Some(actual) = shape.resolve(point)? {
                resolve_slice(point, entry, &actual)?;
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
                shape.validate_slices(&point, &operation.slices)?;
                if let Some(actual) = shape.resolve(&point)? {
                    run.plan.validate_at(
                        index,
                        phase,
                        prediction,
                        Some(shape),
                        &actual,
                        operation.action.dtype(),
                    )?;
                }
            }
        }
        self.invocation = Some(shape);
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
    if plan.is_empty()
        && extra.is_empty()
        && scheduled_costs.is_empty()
        && base == CaptureUsage::default()
    {
        if let Some(budget) = inherited.exceeded(plan.plan().limits.cumulative) {
            return Err(CaptureError::Limit {
                budget,
                cumulative: true,
            });
        }
        return Ok(());
    }
    let bounds = plan.invocation_bounds().expect("invocation admission");
    let geometry = bounds.maximum()?;
    let mut step = base.checked_add(INVOCATION_METADATA)?;
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
        step = step.checked_add(metadata_reservation(selection, point)?)?;
        if (active(&selection.schedule, CapturePhase::Prefill)?
            || active(&selection.schedule, CapturePhase::Decode)?)
            && plan.plan().limits.on_limit == CaptureLimitPolicy::Fail
        {
            if let Some(shape) = geometry.resolve(point)? {
                let slice = resolve_slice(point, selection, &shape)?;
                step = step.checked_add(estimate(&shape, selection, &slice)?)?;
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
        step = step.checked_add(CaptureUsage {
            captures: a.captures.max(b.captures),
            retained_bytes: a.retained_bytes.max(b.retained_bytes),
            host_bytes: a.host_bytes.max(b.host_bytes),
            encoded_bytes: a.encoded_bytes.max(b.encoded_bytes),
        })?;
    }
    if let Some(budget) = step.exceeded(plan.plan().limits.per_step) {
        return Err(CaptureError::Limit {
            budget,
            cumulative: false,
        });
    }
    if let Some(budget) = inherited
        .checked_add(step)?
        .exceeded(plan.plan().limits.cumulative)
    {
        return Err(CaptureError::Limit {
            budget,
            cumulative: true,
        });
    }
    Ok(())
}
