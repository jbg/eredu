//! Loaded-session authority for explicitly scoped internal speculative evidence.
use super::SpeculativeActivationPhase;
use crate::{capture::*, intervention::*};
use serde::{Deserialize, Serialize};
#[cfg(test)]
mod tests;
mod source;
pub use source::SpeculativeActivationSourceError;

/// Wire version for internal speculative capture plans and discovery.
pub const SPECULATIVE_ACTIVATION_SCHEMA_VERSION: u32 = 2;

/// Architecture-supplied applicability of an observation or intervention node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpeculativeCaptureScope {
    /// Ordinary target stack and readout.
    Target,
    /// Independent sequential prediction invocation.
    Prediction {
        /// Architecture-declared zero-based depth.
        depth: usize,
    },
    /// Preparation of prediction caches from accepted target context. It does
    /// not imply that the predictor's decoder writes or score head execute.
    PredictionContext,
    /// One fused proposal invocation, potentially spanning several physical
    /// decoder blocks. Proposal row count is independent of block count.
    FusedProposal,
}
impl SpeculativeCaptureScope {
    /// Whether the declared node participates in this physical phase.
    pub fn applies(self, phase: SpeculativeActivationPhase) -> bool {
        use SpeculativeActivationPhase as P;
        match (self, phase) {
            (Self::Target, P::TargetPrefill | P::Verification | P::TargetReplay) => true,
            (Self::Prediction { depth }, P::Proposal { depth: actual }) => depth == actual,
            (Self::Prediction { .. }, P::PredictionPrefill | P::PredictionReplay) => true,
            (Self::PredictionContext, P::PredictionPrefill | P::PredictionReplay) => true,
            (Self::FusedProposal, P::FusedProposal) => true,
            _ => false,
        }
    }
}

/// An exact architecture node and the invocation scope that executes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculativeCaptureBinding {
    /// Stable architecture identity, independent of checkpoint naming.
    pub node_id: String,
    /// Architecture-projected invocation applicability.
    pub scope: SpeculativeCaptureScope,
}

/// Internal capture support for the actual selected speculative call path.
/// Ordinary tensor support does not establish this report. Conditional capture
/// requirements must be resolved by the owner of the retained invocation scope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeActivationDiscovery {
    /// Discovery wire version; see `SPECULATIVE_ACTIVATION_SCHEMA_VERSION`.
    pub schema_version: u32,
    /// Retained selected execution, including any effective parameter version.
    pub execution_identity: String,
    /// Exact supported internal points and native transformations.
    pub captures: CaptureDiscovery,
    /// Actual mutable points with the realized session identity.
    pub interventions: InterventionDiscovery,
    /// Invocation binding for each selectable architecture node.
    pub bindings: Vec<SpeculativeCaptureBinding>,
}

/// Serializable internal capture/edit request, separate from sampler logits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeActivationPlan {
    /// Wire version; see `SPECULATIVE_ACTIVATION_SCHEMA_VERSION`.
    pub schema_version: u32,
    /// Exact selections and shared native/host/encoded allowances.
    pub captures: CapturePlan,
    /// Optional edits with original/effective evidence under the same allowances.
    pub interventions: InterventionPlan,
    /// Maximum physical geometry and independent prediction schedule domain.
    pub bounds: CaptureInvocationBounds,
}

/// Immutable authority bound to actual loaded model, execution and session facts.
/// Snapshot restoration does not recreate its cumulative runtime allowance.
#[derive(Debug, Clone)]
pub struct AdmittedSpeculativeActivations {
    identity: String,
    execution_identity: String,
    artifact_identity: String,
    session_identity: String,
    captures: AdmittedCapturePlan,
    interventions: AdmittedInterventionPlan,
    capture_scopes: Vec<SpeculativeCaptureScope>,
    intervention_scopes: Vec<SpeculativeCaptureScope>,
}

impl SpeculativeActivationPlan {
    /// Validates exact invocation geometry, loaded support and all scope joins
    /// before a native observer may be constructed. Native estimators still
    /// preflight and reserve actual transformations before work.
    pub fn admit(
        self,
        discovery: &SpeculativeActivationDiscovery,
    ) -> Result<AdmittedSpeculativeActivations, CaptureError> {
        if self.schema_version != SPECULATIVE_ACTIVATION_SCHEMA_VERSION
            || discovery.schema_version != SPECULATIVE_ACTIVATION_SCHEMA_VERSION
            || discovery.execution_identity.is_empty()
            || discovery.captures.artifact_identity.is_empty()
            || discovery.captures.artifact_identity != discovery.interventions.artifact_identity
        {
            return Err(CaptureError::Invalid(
                "inconsistent speculative activation discovery or schema".into(),
            ));
        }
        let session = discovery
            .interventions
            .session_identity
            .as_deref()
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                CaptureError::Invalid(
                    "speculative activation admission requires a realized session".into(),
                )
            })?;
        let mut bindings = std::collections::BTreeMap::new();
        for binding in &discovery.bindings {
            if binding.node_id.is_empty()
                || bindings
                    .insert(binding.node_id.as_str(), binding.scope)
                    .is_some()
            {
                return Err(CaptureError::Invalid(
                    "duplicate or empty speculative node binding".into(),
                ));
            }
        }
        let captures = self.captures.admit_invocations(
            &discovery.captures.catalog,
            &discovery.captures.support,
            &discovery.captures.support.capture,
            self.bounds,
        )?;
        let interventions =
            self.interventions
                .admit_invocations(&discovery.interventions, self.bounds, session)?;
        let scope = |node: &str| {
            bindings.get(node).copied().ok_or_else(|| {
                CaptureError::Invalid(format!(
                    "speculative node has no invocation binding: {node}"
                ))
            })
        };
        let capture_scopes = captures
            .points()
            .iter()
            .map(|p| scope(&p.node_id))
            .collect::<Result<Vec<_>, _>>()?;
        let intervention_scopes = interventions
            .points()
            .iter()
            .map(|p| scope(&p.node_id))
            .collect::<Result<Vec<_>, _>>()?;
        use sha2::{Digest, Sha256};
        let encoded = serde_json::to_vec(&(
            "eredu.speculative.activations.v2",
            &discovery.execution_identity,
            &discovery.captures.artifact_identity,
            session,
            captures.identity(),
            interventions.identity(),
            &capture_scopes,
            &intervention_scopes,
        ))
        .map_err(|error| CaptureError::Invalid(error.to_string()))?;
        let identity = Sha256::digest(encoded)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Ok(AdmittedSpeculativeActivations {
            identity,
            execution_identity: discovery.execution_identity.clone(),
            artifact_identity: discovery.captures.artifact_identity.clone(),
            session_identity: session.into(),
            captures,
            interventions,
            capture_scopes,
            intervention_scopes,
        })
    }
}

impl AdmittedSpeculativeActivations {
    /// Stable digest of this exact session, geometry, selections, edits and scopes.
    pub fn identity(&self) -> &str {
        &self.identity
    }
    /// Selected execution/effective-parameter version bound at admission.
    pub fn execution_identity(&self) -> &str {
        &self.execution_identity
    }
    /// Exact loaded artifact identity.
    pub fn artifact_identity(&self) -> &str {
        &self.artifact_identity
    }
    /// Realized backend session identity.
    pub fn session_identity(&self) -> &str {
        &self.session_identity
    }
    /// Immutable capture authority and its shared allowances.
    pub fn captures(&self) -> &AdmittedCapturePlan {
        &self.captures
    }
    /// Immutable edits, including an explicitly empty edit set.
    pub fn interventions(&self) -> &AdmittedInterventionPlan {
        &self.interventions
    }
    /// Scope applicability in admitted capture order.
    pub fn capture_scopes(&self) -> &[SpeculativeCaptureScope] {
        &self.capture_scopes
    }
    /// Scope applicability in admitted edit order.
    pub fn intervention_scopes(&self) -> &[SpeculativeCaptureScope] {
        &self.intervention_scopes
    }
    /// Whether the caller can preserve a completely absent internal observer.
    pub fn is_empty(&self) -> bool {
        self.captures.is_empty() && self.interventions.is_empty()
    }
    /// Rechecks all retained authority against the actual loaded call path.
    pub fn validate(&self, discovery: &SpeculativeActivationDiscovery) -> Result<(), CaptureError> {
        let current = SpeculativeActivationPlan {
            schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
            captures: self.captures.plan().clone(),
            interventions: self.interventions.plan().clone(),
            bounds: self.captures.invocation_bounds().ok_or_else(|| {
                CaptureError::Invalid("missing speculative invocation bounds".into())
            })?,
        }
        .admit(discovery)?;
        if current.identity != self.identity {
            return Err(CaptureError::Invalid(
                "speculative activation authority differs from the loaded session or execution"
                    .into(),
            ));
        }
        Ok(())
    }
}
