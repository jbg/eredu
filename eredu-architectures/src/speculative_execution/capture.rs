use eredu_core::{
    capture::CaptureError, component::ComponentExecutionScopeKind, ArchitectureDescriptor,
    ArchitectureNodeKind,
};
use eredu_runtime::capture::SpeculativeCaptureScope;

/// Complete internal hook coverage of one architecture-selected prediction
/// executor. Only the sealed materialized extension contract constructs it.
#[derive(Debug, Clone)]
pub struct SpeculativeActivationExecution {
    pub(crate) depth: usize,
    pub(crate) strategy: eredu_runtime::SpeculativeStrategyClass,
}

impl SpeculativeActivationExecution {
    pub(crate) fn selected(
        complete_hooks: bool,
        depth: usize,
        selected: &eredu_runtime::SelectedSpeculativeRealization,
    ) -> Option<Self> {
        let strategy = selected.requirements().strategy().class();
        (complete_hooks
            && depth > 0
            && matches!(
                strategy,
                eredu_runtime::SpeculativeStrategyClass::EmbeddedSequential
                    | eredu_runtime::SpeculativeStrategyClass::EmbeddedFused
            ))
        .then_some(Self { depth, strategy })
    }

    pub(crate) fn validate_scope(
        &self,
        scope: SpeculativeCaptureScope,
    ) -> Result<(), CaptureError> {
        use eredu_runtime::SpeculativeStrategyClass as Strategy;
        let valid = match scope {
            SpeculativeCaptureScope::Target => true,
            SpeculativeCaptureScope::Prediction { depth } => {
                self.strategy == Strategy::EmbeddedSequential && depth < self.depth
            }
            SpeculativeCaptureScope::PredictionContext | SpeculativeCaptureScope::FusedProposal => {
                self.strategy == Strategy::EmbeddedFused
            }
        };
        if valid {
            Ok(())
        } else {
            Err(CaptureError::Invalid(
                "invocation scope differs from selected prediction strategy or depth".into(),
            ))
        }
    }
}

/// Resolves one observation/intervention node through declared architecture
/// ancestry. No checkpoint-name or activation-path parsing selects semantics.
pub fn speculative_capture_scope(
    descriptor: &ArchitectureDescriptor,
    node_id: &str,
) -> Result<SpeculativeCaptureScope, CaptureError> {
    let mut declared = std::collections::BTreeMap::new();
    for binding in &descriptor.speculative_invocations {
        if descriptor.node(&binding.node_id).is_none()
            || declared
                .insert(binding.node_id.as_str(), binding.scope)
                .is_some()
        {
            return Err(CaptureError::Invalid(
                "duplicate or unknown speculative invocation root".into(),
            ));
        }
    }
    for scope in &descriptor.component_scopes {
        let expected = match scope.kind {
            ComponentExecutionScopeKind::Prediction { depth } => {
                SpeculativeCaptureScope::Prediction { depth }
            }
            ComponentExecutionScopeKind::FusedPrediction => SpeculativeCaptureScope::FusedProposal,
        };
        if declared
            .get(scope.node_id.as_str())
            .is_some_and(|actual| *actual != expected)
        {
            return Err(CaptureError::Invalid(
                "component score scope conflicts with its invocation declaration".into(),
            ));
        }
    }
    let mut next = Some(node_id);
    for _ in 0..=descriptor.nodes.len() {
        let Some(id) = next else {
            return Ok(SpeculativeCaptureScope::Target);
        };
        let node = descriptor
            .nodes
            .iter()
            .find(|n| n.id == id)
            .ok_or_else(|| {
                CaptureError::Invalid(format!(
                    "speculative capture node absent from architecture: {id}"
                ))
            })?;
        if let Some(scope) = declared.get(id) {
            return Ok(*scope);
        }
        if let Some(scope) = descriptor
            .component_scopes
            .iter()
            .find(|scope| scope.node_id == id)
        {
            return Ok(match scope.kind {
                ComponentExecutionScopeKind::Prediction { depth } => {
                    SpeculativeCaptureScope::Prediction { depth }
                }
                ComponentExecutionScopeKind::FusedPrediction => {
                    SpeculativeCaptureScope::FusedProposal
                }
            });
        }
        if node.kind == ArchitectureNodeKind::Prediction {
            return Err(CaptureError::Unsupported(
                "prediction observation has no declared invocation scope".into(),
            ));
        }
        next = node.parent.as_deref();
    }
    Err(CaptureError::Invalid(
        "cycle in speculative capture node ancestry".into(),
    ))
}
