//! Fixed destinations for the existing one-worst-invocation continuation policy.
use super::*;
use crate::capture::InvocationPreflightBudget;

pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<InvocationPreflightBudget<'_>>(),
        size_of::<CaptureInvocationShape>(),
        size_of::<Option<CaptureInvocationBounds>>(),
        size_of::<[CaptureUsage; 2]>(),
        size_of::<[u64; RANK]>(),
        size_of::<[CapturePhase; 2]>(),
        size_of::<Result<(), StaticInterventionPreflightError>>(),
        size_of::<(
            &mut StaticInterventionPreflight,
            &AdmittedCapturePlan,
            &AdmittedInterventionPlan,
            Option<&SharedInterventionPlan>,
            &dyn InterventionEstimator,
            CaptureUsage,
        )>(),
        size_of::<(
            &mut StaticInterventionPreflight,
            &mut InvocationPreflightBudget<'_>,
            &AdmittedCapturePlan,
            CaptureInvocationShape,
            &dyn InterventionEstimator,
        )>(),
        size_of::<Result<Option<(u64, u64)>, CaptureError>>(),
        size_of::<(CaptureUsage, CaptureUsage, CaptureUsage)>(),
        size_of::<Option<&eredu_core::intervention::RoutedUnitInterventionPoint>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
impl StaticInterventionPreflight {
    /// Shared invocation preflight using immutable source-owned evidence rows.
    /// Current spending is inherited; this does not reserve or refund a callback.
    pub fn run_invocation_with_source(
        &mut self,
        capture: &AdmittedCapturePlan,
        source: &SharedInterventionPlan,
        estimator: &dyn InterventionEstimator,
        inherited: CaptureUsage,
    ) -> Result<(), StaticInterventionPreflightError> {
        self.run_invocation(
            capture,
            source.admission(),
            Some(source),
            estimator,
            inherited,
        )
    }
    /// Same policy for an admitted plan requiring no evidence companions.
    pub fn run_invocation_without_evidence(
        &mut self,
        capture: &AdmittedCapturePlan,
        intervention: &AdmittedInterventionPlan,
        estimator: &dyn InterventionEstimator,
        inherited: CaptureUsage,
    ) -> Result<(), StaticInterventionPreflightError> {
        self.run_invocation(capture, intervention, None, estimator, inherited)
    }
    fn run_invocation(
        &mut self,
        capture: &AdmittedCapturePlan,
        intervention: &AdmittedInterventionPlan,
        source: Option<&SharedInterventionPlan>,
        estimator: &dyn InterventionEstimator,
        inherited: CaptureUsage,
    ) -> Result<(), StaticInterventionPreflightError> {
        use StaticInterventionPreflightError as E;
        let bounds = capture.invocation_bounds().ok_or(E::Request)?;
        if capture.request() != intervention.request()
            || Some(bounds) != intervention.invocation_bounds()
        {
            return Err(E::Request);
        }
        let geometry = bounds.maximum()?;
        let mut base = CaptureUsage::default();
        for (operation, point) in intervention
            .plan()
            .operations
            .iter()
            .zip(intervention.points())
        {
            base = base.checked_add(intervention_metadata(
                operation,
                point,
                intervention.identity(),
            )?)?;
        }
        let mut budget = InvocationPreflightBudget::new(
            capture,
            base,
            inherited,
            !capture.is_empty() || !intervention.is_empty(),
        )?;
        self.add_invocation_selections(&mut budget, capture, geometry, estimator)?;
        for (index, (operation, point)) in intervention
            .plan()
            .operations
            .iter()
            .zip(intervention.points())
            .enumerate()
        {
            let companion = source.and_then(|source| source.evidence(index));
            match (&operation.evidence, companion) {
                (InterventionEvidence::None, None) => (),
                (
                    InterventionEvidence::Preview { .. } | InterventionEvidence::Summary,
                    Some(companion),
                ) if companion.operation() == index
                    && companion.geometry_source().points().len()
                        == if point.routing.is_some() { 4 } else { 2 } =>
                {
                    self.add_invocation_selections(
                        &mut budget,
                        companion.geometry_source(),
                        geometry,
                        estimator,
                    )?;
                }
                _ => return Err(E::Profile),
            }
            let mut costs = [CaptureUsage::default(); 2];
            for (at, phase) in [CapturePhase::Prefill, CapturePhase::Decode]
                .into_iter()
                .enumerate()
            {
                let Some((_, last)) =
                    operation
                        .schedule
                        .count_coordinates(phase, 0, bounds.max_predictions)?
                else {
                    continue;
                };
                self.rank(point.axes.len())?;
                let mut shape = [0; RANK];
                let shape = &mut shape[..point.axes.len()];
                if !geometry.resolve_axes_into(&point.axes, shape)? {
                    continue;
                }
                costs[at] = self.resolve_cost(
                    intervention,
                    index,
                    phase,
                    last,
                    Some(geometry),
                    shape,
                    estimator,
                )?;
            }
            let [a, b] = costs;
            budget.add(CaptureUsage {
                captures: a.captures.max(b.captures),
                retained_bytes: a.retained_bytes.max(b.retained_bytes),
                host_bytes: a.host_bytes.max(b.host_bytes),
                encoded_bytes: a.encoded_bytes.max(b.encoded_bytes),
            })?;
        }
        budget.finish()?;
        Ok(())
    }
    fn add_invocation_selections(
        &mut self,
        budget: &mut InvocationPreflightBudget<'_>,
        source: &AdmittedCapturePlan,
        geometry: CaptureInvocationShape,
        estimator: &dyn InterventionEstimator,
    ) -> Result<(), StaticInterventionPreflightError> {
        for (selection, point) in source.plan().selections.iter().zip(source.points()) {
            budget.add(crate::capture::metadata_reservation(selection, point)?)?;
            let bounds = source
                .invocation_bounds()
                .ok_or(StaticInterventionPreflightError::Request)?;
            let active = selection
                .schedule
                .count_coordinates(CapturePhase::Prefill, 0, bounds.max_predictions)?
                .is_some()
                || selection
                    .schedule
                    .count_coordinates(CapturePhase::Decode, 0, bounds.max_predictions)?
                    .is_some();
            // The enclosing admission supplies Skip/Fail, including companions
            // whose geometry source deliberately has no independent allowance.
            if !active || !budget.reserves_selections() {
                continue;
            }
            let Some(axes) = point.axes.as_deref() else {
                continue;
            };
            self.rank(axes.len())?;
            let mut shape = [0; RANK];
            let shape = &mut shape[..axes.len()];
            if !geometry.resolve_axes_into(axes, shape)? {
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
            budget.add(estimator.capture_usage(shape, selection, &self.slice)?)?;
        }
        Ok(())
    }
}
