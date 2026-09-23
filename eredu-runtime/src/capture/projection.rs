//! Reusable ordinary capture sizing, shared by preflight and memory forecasts.
use super::*;
use serde::{Deserialize, Serialize};

/// One admitted schedule and conservative per-occurrence native/host/wire costs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduledCaptureUsage {
    /// Absolute prediction schedule; its frequency origin never rewinds.
    pub schedule: CaptureSchedule,
    /// Prefill and decode costs at the largest selected shape in the coverage range.
    /// None means geometry or complete source/transform costing is unavailable.
    pub costs: [Option<CaptureUsage>; 2],
    /// Mandatory cost even under skip-on-limit admission (e.g. an intervention).
    pub required: bool,
}

/// Descriptive sizing facts, not admission authority. Costs must include source
/// creation when applicable and bound all earlier shapes in the coverage range.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureUsageProjection {
    /// First absolute prediction covered by the observations.
    pub first_prediction: u64,
    /// Exclusive end of shape coverage; longer alternatives use quota fallbacks.
    pub max_predictions: u64,
    /// Diagnostics/metadata emitted each step, including missing/skipped selections.
    pub per_step_metadata: CaptureUsage,
    /// Per-selection and optional intervention schedules.
    pub selections: Vec<ScheduledCaptureUsage>,
}

/// Known costs for an absolute prediction range. Unknown values do not erase the
/// known contributions, but must not be treated as a complete forecast bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaptureUsageOutlook {
    /// Peak per phase; enabled selections may conservatively coincide.
    pub phases: [CaptureUsage; 2],
    /// Cumulative future costs, excluding history already charged by a live run.
    pub total: CaptureUsage,
    /// False if any enabled source lacks full geometry or native cost coverage.
    pub complete: bool,
}

impl CaptureUsageProjection {
    /// Recounts selected occurrences without advancing or charging the capture owner.
    /// The retained maximum-shape costs remain conservative for shorter horizons.
    pub fn outlook(
        &self,
        first: u64,
        end: u64,
    ) -> Result<Option<CaptureUsageOutlook>, CaptureError> {
        self.known_outlook(first, end, true)
    }

    pub(super) fn known_outlook(
        &self,
        first: u64,
        end: u64,
        include_optional: bool,
    ) -> Result<Option<CaptureUsageOutlook>, CaptureError> {
        if end < first {
            return Err(CaptureError::Invalid(
                "reversed capture forecast range".into(),
            ));
        }
        if first < self.first_prediction || end > self.max_predictions {
            return Ok(None);
        }
        let mut result = CaptureUsageOutlook {
            phases: [CaptureUsage::default(); 2],
            total: self.per_step_metadata.checked_mul(end - first)?,
            complete: true,
        };
        for (index, phase) in [CapturePhase::Prefill, CapturePhase::Decode]
            .into_iter()
            .enumerate()
        {
            if end == first
                || (phase == CapturePhase::Prefill && first > 0)
                || (phase == CapturePhase::Decode && end <= 1)
            {
                continue;
            }
            result.phases[index] = self.per_step_metadata;
            for selection in &self.selections {
                let Some((count, _)) = selection.schedule.count_and_last_from(phase, first, end)?
                else {
                    continue;
                };
                if !include_optional && !selection.required {
                    continue;
                }
                match selection.costs[index] {
                    Some(cost) => {
                        result.phases[index] = result.phases[index].checked_add(cost)?;
                        result.total = result.total.checked_add(cost.checked_mul(count)?)?;
                    }
                    None => result.complete = false,
                }
            }
        }
        Ok(Some(result))
    }
}

/// Sizes ordinary selections at their maximum scheduled shape. Independent
/// invocations need their own phase-aware projection and return None. The native
/// callback must return None when source creation or transform costs are uncovered.
pub fn project_capture_usage(
    plan: &AdmittedCapturePlan,
    first_prediction: u64,
    estimate: impl FnMut(
        &[u64],
        &CaptureSelection,
        &ResolvedCaptureSlice,
    ) -> Result<Option<CaptureUsage>, CaptureError>,
) -> Result<Option<CaptureUsageProjection>, CaptureError> {
    if plan.invocation_bounds().is_some() {
        return Ok(None);
    }
    build_projection(
        plan,
        &[],
        CaptureUsage::default(),
        &[],
        first_prediction,
        None,
        estimate,
    )
    .map(Some)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_projection(
    plan: &AdmittedCapturePlan,
    extra: &[(CaptureSelection, eredu_core::ObservationPoint)],
    mut base: CaptureUsage,
    scheduled: &[(CaptureSchedule, [CaptureUsage; 2])],
    first: u64,
    metadata_limit: Option<CaptureUsage>,
    mut estimate: impl FnMut(
        &[u64],
        &CaptureSelection,
        &ResolvedCaptureSlice,
    ) -> Result<Option<CaptureUsage>, CaptureError>,
) -> Result<CaptureUsageProjection, CaptureError> {
    let end = plan.request().max_predictions;
    if first > end {
        return Err(CaptureError::Invalid(
            "continuation exceeds admitted prediction range".into(),
        ));
    }
    let entries = plan
        .plan()
        .selections
        .iter()
        .zip(plan.points())
        .chain(extra.iter().map(|(selection, point)| (selection, point)))
        .collect::<Vec<_>>();
    for (selection, point) in &entries {
        base = base.checked_add(metadata_reservation(selection, point)?)?;
    }
    if let Some(budget) = metadata_limit.and_then(|limit| base.exceeded(limit)) {
        return Err(CaptureError::Limit {
            budget,
            cumulative: false,
        });
    }
    let mut selections = scheduled
        .iter()
        .map(|(schedule, costs)| ScheduledCaptureUsage {
            schedule: schedule.clone(),
            costs: costs.map(Some),
            required: true,
        })
        .collect::<Vec<_>>();
    for (selection, point) in entries {
        let mut costs = [Some(CaptureUsage::default()); 2];
        for (index, phase) in [CapturePhase::Prefill, CapturePhase::Decode]
            .into_iter()
            .enumerate()
        {
            let Some((_, last)) = selection.schedule.count_and_last_from(phase, first, end)? else {
                continue;
            };
            costs[index] = match plan.estimate_shape(point, phase, last)? {
                Some(shape) => {
                    estimate(&shape, selection, &resolve_slice(point, selection, &shape)?)?
                }
                None => None,
            };
        }
        selections.push(ScheduledCaptureUsage {
            schedule: selection.schedule.clone(),
            costs,
            required: false,
        });
    }
    Ok(CaptureUsageProjection {
        first_prediction: first,
        max_predictions: end,
        per_step_metadata: base,
        selections,
    })
}
