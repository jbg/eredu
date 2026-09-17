//! Original source revalidation over retained architecture declarations.
use super::PreparedModelDiscovery;
use crate::speculative_execution::SpeculativeActivationExecution;
use eredu_core::{
    capture::{AdmittedCapturePlan, CapturePhase},
    speculative::{
        AdmittedSpeculativeActivations, SpeculativeActivationSourceError as E,
        SpeculativeCaptureScope,
    },
};

struct PhaseValidation<'a> {
    descriptor: &'a eredu_core::ArchitectureDescriptor,
    context: eredu_runtime::inspection::ObservationExecutionContext,
}
impl PhaseValidation<'_> {
    fn allows(&self, point: &eredu_core::ObservationPoint, phase: CapturePhase) -> bool {
        let Ok(scope) =
            SpeculativeActivationExecution::retained_scope(self.descriptor, &point.node_id)
        else {
            return false;
        };
        eredu_runtime::inspection::observation_phase_is_admissible(
            point,
            phase,
            self.context,
            scope != SpeculativeCaptureScope::Target,
        )
    }
}

impl PreparedModelDiscovery {
    /// Revalidate an immutable internal capture admission against the actual
    /// retained catalog, invocation hooks and selected collector facts. Identity
    /// must already have been resolved by ordinary discovery/admission. This
    /// borrows all declarations; it neither reconstructs discovery nor issues
    /// native, invocation, geometry or host-allocation authority.
    ///
    /// Partition producers remain a distinct projection. This capture-only
    /// entry requires no edits; the facts-bearing entry below qualifies static
    /// internal edits through the same retained declaration checks.
    pub fn validate_original_speculative_activations(
        &self,
        admitted: &AdmittedSpeculativeActivations,
        execution: &SpeculativeActivationExecution,
        session: &str,
        overlay: Option<&str>,
    ) -> Result<(), E> {
        self.validate_original_activations(admitted, execution, session, overlay, None, None)
    }
    /// Same loaded revalidation with actual static native mechanism facts. The
    /// source contains no execution authority; exact trace/claim binding follows.
    pub fn validate_original_speculative_activations_with_interventions(
        &self,
        admitted: &AdmittedSpeculativeActivations,
        execution: &SpeculativeActivationExecution,
        session: &str,
        overlay: Option<&str>,
        facts: eredu_core::intervention::InterventionMechanismFacts<'_>,
    ) -> Result<(), E> {
        self.validate_original_activations(admitted, execution, session, overlay, Some(facts), None)
    }
    /// The same retained declarations with the actual original sparse worker's
    /// operation/dtype profile. These facts are descriptive; exact selected row
    /// counts, source loans and execution funding are checked at the native hook.
    pub fn validate_original_speculative_activations_with_routed_interventions(
        &self, admitted: &AdmittedSpeculativeActivations,
        execution: &SpeculativeActivationExecution, session: &str, overlay: Option<&str>,
        facts: eredu_core::intervention::InterventionMechanismFacts<'_>,
        routed: eredu_core::intervention::InterventionMechanismFacts<'_>,
    ) -> Result<(), E> {
        self.validate_original_activations(admitted, execution, session, overlay, Some(facts), Some(routed))
    }
    fn validate_original_activations(
        &self,
        admitted: &AdmittedSpeculativeActivations,
        execution: &SpeculativeActivationExecution,
        session: &str,
        overlay: Option<&str>,
        facts: Option<eredu_core::intervention::InterventionMechanismFacts<'_>>,
        routed: Option<eredu_core::intervention::InterventionMechanismFacts<'_>>,
    ) -> Result<(), E> {
        if self.partition_selection.is_some()
            || self.observation_context.partitioned
            || !self.identity.is_resolved()
        {
            return Err(E::Identity);
        }
        let identity = self.identity.resolve().map_err(|_| E::Identity)?;
        admitted.validate_source_identity(identity, self.execution_identity(), overlay, session)?;
        let prediction = self.prediction.as_ref().ok_or(E::Declaration)?;
        let descriptor = &prediction.descriptor;
        // Ordinary speculative discovery checks every retained point's declared
        // invocation, even points not selected by the requested capture plan.
        for point in &descriptor.observations.points {
            let scope = SpeculativeActivationExecution::retained_scope(descriptor, &point.node_id)?;
            if !execution.supports_scope(scope) {
                return Err(E::Declaration);
            }
        }
        if !admitted.interventions().is_empty() && facts.is_none() {
            return Err(E::UnqualifiedIntervention);
        }
        if admitted.capture_scopes().len() != admitted.captures().points().len()
            || admitted.intervention_scopes().len() != admitted.interventions().points().len()
            || admitted.captures().invocation_bounds().is_none()
            || admitted.captures().invocation_bounds()
                != admitted.interventions().invocation_bounds()
        {
            return Err(E::Declaration);
        }
        for (point, expected) in admitted
            .captures()
            .points()
            .iter()
            .zip(admitted.capture_scopes())
        {
            let actual =
                SpeculativeActivationExecution::retained_scope(descriptor, &point.node_id)?;
            if actual != *expected || !execution.supports_scope(actual) {
                return Err(E::Declaration);
            }
        }
        let mut context = self.observation_context;
        context.prediction_inspection = true;
        let phase_validation = PhaseValidation {
            descriptor,
            context,
        };
        if let Some(facts) = facts {
            use eredu_core::intervention::{InterventionEvidence, InterventionStage};
            for ((operation, point), scope) in admitted
                .interventions()
                .plan()
                .operations
                .iter()
                .zip(admitted.interventions().points())
                .zip(admitted.intervention_scopes())
            {
                let actual =
                    SpeculativeActivationExecution::retained_scope(descriptor, &point.node_id)?;
                if actual != *scope || !execution.supports_scope(actual) {
                    return Err(E::Declaration);
                }
                if point.stage != InterventionStage::Activation
                    || point.routing.is_some()
                    || operation.action.dtype().is_none()
                    || !matches!(operation.evidence, InterventionEvidence::None | InterventionEvidence::Preview { .. } | InterventionEvidence::Summary)
                {
                    return Err(E::UnqualifiedIntervention);
                }
                if point.routed_units.is_some() && (operation.evidence != InterventionEvidence::None
                    || routed.is_none_or(|profile| !profile.routed_units
                        || !profile.operations.contains(&operation.action.kind())
                        || operation.action.dtype().is_none_or(|dtype| !profile.dtypes.contains(&dtype)))) {
                    return Err(E::UnqualifiedIntervention);
                }
            }
            eredu_runtime::inspection::validate_activation_intervention_declarations_with_phases(
                admitted.interventions(),
                &prediction.intervention_points,
                facts,
                |actual, expected| {
                    let mut points = descriptor.observations.points.iter().filter(|point| {
                        point.path == actual.path && point.node_id == actual.node_id
                    });
                    let Some(point) = points.next() else {
                        return false;
                    };
                    if points.next().is_some() {
                        return false;
                    }
                    let Ok(scope) =
                        SpeculativeActivationExecution::retained_scope(descriptor, &point.node_id)
                    else {
                        return false;
                    };
                    [CapturePhase::Prefill, CapturePhase::Decode]
                        .into_iter()
                        .all(|phase| {
                            eredu_runtime::inspection::observation_phase_matches(
                                point,
                                phase,
                                context,
                                scope != SpeculativeCaptureScope::Target,
                                match phase {
                                    CapturePhase::Prefill => &expected.prefill,
                                    CapturePhase::Decode => &expected.decode,
                                },
                            )
                        })
                },
            )
            .map_err(|_| E::Declaration)?;
        }
        admitted
            .captures()
            .revalidate_borrowed_declarations(
                &descriptor.observations,
                self.support.schema_version,
                &self.support.capture,
                |point, phase| phase_validation.allows(point, phase),
            )
            .map_err(|_| E::Declaration)
    }

    /// Fixed validation frames and shared helper controls. The enclosing original
    /// source constructor charges these before calling the borrowed validator.
    /// No deep catalog, source payload or native population is represented here.
    pub fn original_speculative_activation_validation_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<(
                &Self,
                &AdmittedSpeculativeActivations,
                &SpeculativeActivationExecution,
                &str,
                Option<&str>,
            )>(),
            size_of::<
                Result<
                    eredu_core::artifact::ArtifactIdentity,
                    std::sync::Arc<eredu_core::artifact::ArtifactError>,
                >,
            >(),
            size_of::<(
                &super::PreparedPredictionDiscovery,
                &eredu_core::ArchitectureDescriptor,
            )>(),
            size_of::<std::slice::Iter<'static, eredu_core::ObservationPoint>>(),
            size_of::<
                std::iter::Zip<
                    std::slice::Iter<'static, eredu_core::ObservationPoint>,
                    std::slice::Iter<'static, SpeculativeCaptureScope>,
                >,
            >(),
            size_of::<(&eredu_core::ObservationPoint, &SpeculativeCaptureScope)>(),
            size_of::<eredu_runtime::inspection::ObservationExecutionContext>(),
            // Actual phase closure captures one borrow of these fixed controls.
            size_of::<PhaseValidation<'static>>(),
            size_of::<&PhaseValidation<'static>>(),
            size_of::<(&eredu_core::ObservationPoint, CapturePhase)>(),
            size_of::<SpeculativeCaptureScope>(),
            size_of::<Result<(), E>>(),
            size_of::<[Option<eredu_core::intervention::InterventionMechanismFacts<'static>>; 2]>(),
            size_of::<(
                &eredu_core::ArchitectureDescriptor,
                &eredu_runtime::inspection::ObservationExecutionContext,
            )>(),
            size_of::<[CapturePhase; 2]>(),
            size_of::<(
                &eredu_core::intervention::InterventionPoint,
                &eredu_core::intervention::InterventionPoint,
            )>(),
            eredu_runtime::inspection::static_intervention_validation_control_bytes()?,
            size_of::<Result<(), eredu_core::capture::CaptureRevalidationError>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?
            .checked_add(AdmittedSpeculativeActivations::source_validation_control_bytes()?)?
            .checked_add(AdmittedCapturePlan::borrowed_declaration_validation_control_bytes()?)?
            .checked_add(SpeculativeActivationExecution::scope_validation_control_bytes()?)?
            .checked_add(eredu_runtime::inspection::observation_phase_validation_control_bytes()?)
    }
}

#[cfg(test)]
mod tests;
