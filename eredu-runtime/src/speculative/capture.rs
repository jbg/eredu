use crate::capture::SpeculativeCaptureScope;
use crate::inspection::SpeculativeActivationObserver;
use eredu_core::speculative::SpeculativeActivationCapture;
use eredu_core::{
    capture::CaptureError, component::ComponentExecutionScopeKind, ArchitectureDescriptor,
    ArchitectureNodeKind,
};

/// Complete internal hook coverage of an architecture-selected generation source.
/// Embedded construction and retained shared-backbone declarations supply its semantics.
#[derive(Debug, Clone)]
pub struct SpeculativeActivationExecution {
    depth: usize,
    strategy: Option<crate::SpeculativeStrategyClass>,
}

impl SpeculativeActivationExecution {
    /// Admits complete embedded hook coverage for the selected strategy and depth.
    pub fn selected(
        complete_hooks: bool,
        depth: usize,
        selected: &crate::SelectedSpeculativeRealization,
    ) -> Option<Self> {
        let strategy = selected.requirements().strategy().class();
        (complete_hooks
            && depth > 0
            && matches!(
                strategy,
                crate::SpeculativeStrategyClass::EmbeddedSequential
                    | crate::SpeculativeStrategyClass::EmbeddedFused
            ))
        .then_some(Self {
            depth,
            strategy: Some(strategy),
        })
    }

    /// Uses one set of declared hooks across shared-backbone invocations.
    pub fn shared_backbone() -> Self {
        Self {
            depth: 0,
            strategy: None,
        }
    }

    /// The causal target, ordinary refinement and masked drafts invoke the same
    /// declared hooks and producer placement, with independent invocation records.
    pub fn is_shared_backbone(&self) -> bool {
        self.strategy.is_none()
    }

    /// Number of ordered prediction units accepted by this contract.
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Embedded strategy for this contract, when it uses a prediction extension.
    pub const fn strategy(&self) -> Option<crate::SpeculativeStrategyClass> {
        self.strategy
    }

    /// Checks that the requested scope exists in this invocation contract.
    pub fn validate_scope(&self, scope: SpeculativeCaptureScope) -> Result<(), CaptureError> {
        use crate::SpeculativeStrategyClass as Strategy;
        let valid = match scope {
            SpeculativeCaptureScope::Target => true,
            SpeculativeCaptureScope::Prediction { depth } => {
                self.strategy == Some(Strategy::EmbeddedSequential) && depth < self.depth
            }
            SpeculativeCaptureScope::PredictionContext | SpeculativeCaptureScope::FusedProposal => {
                self.strategy == Some(Strategy::EmbeddedFused)
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

/// Observation path for the physical target capture consumed by embedded prediction.
pub const EMBEDDED_TARGET_CAPTURE_PATH: &str = "embedded_prediction.target_capture";
/// Observation path for a sequential prediction hidden output.
pub const EMBEDDED_PREDICTION_OUTPUT_PATH: &str = "embedded_prediction.output";
/// Observation path for logits consumed by speculative proposal sampling.
pub const EMBEDDED_PROPOSAL_LOGITS_PATH: &str = "embedded_prediction.proposal_logits";
/// Observation path for the full target verification logits tensor.
pub const EMBEDDED_VERIFICATION_LOGITS_PATH: &str = "embedded_prediction.verification_logits";

/// Production-carried observers for architecture-owned embedded prediction boundaries.
pub struct EmbeddedPredictionObservers<T, L, E> {
    tensors: Box<dyn crate::ActivationObserver<T, E>>,
    logits: Box<dyn crate::ActivationObserver<L, E>>,
    internal: Option<Box<dyn SpeculativeActivationObserver<T, E>>>,
}

impl<T, L, E> EmbeddedPredictionObservers<T, L, E> {
    /// Installs independent tensor and proposal-logit observers.
    pub fn new(
        tensors: impl crate::ActivationObserver<T, E> + 'static,
        logits: impl crate::ActivationObserver<L, E> + 'static,
    ) -> Self {
        Self {
            tensors: Box::new(tensors),
            logits: Box::new(logits),
            internal: None,
        }
    }

    /// Adds explicitly admitted internal phase instrumentation. Existing outer
    /// observers do not enable this route or incur component tensor work.
    pub fn with_internal(
        mut self,
        observer: impl SpeculativeActivationObserver<T, E> + 'static,
    ) -> Self {
        self.internal = Some(Box::new(observer));
        self
    }

    /// Drains one admitted internal record after the executor scope has returned.
    pub fn take_activation_capture(&mut self) -> Option<SpeculativeActivationCapture> {
        self.internal()?.take_activation_capture()
    }

    /// Recovers the original portable failure from native error propagation.
    pub fn take_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        self.internal()?.take_activation_error()
    }

    /// Borrows retained observation authority for snapshot accounting.
    pub fn internal_ref(&self) -> Option<&dyn SpeculativeActivationObserver<T, E>> {
        self.internal.as_deref()
    }

    /// Replaces invocation authority and returns the prior observer for restoration.
    pub fn replace_internal(
        &mut self,
        observer: Option<Box<dyn SpeculativeActivationObserver<T, E>>>,
    ) -> Option<Box<dyn SpeculativeActivationObserver<T, E>>> {
        std::mem::replace(&mut self.internal, observer)
    }

    /// Borrows the admitted observer for one actual target or prediction invocation.
    pub fn internal(&mut self) -> Option<&mut dyn SpeculativeActivationObserver<T, E>> {
        match self.internal.as_mut() {
            Some(observer) => Some(&mut **observer),
            None => None,
        }
    }

    /// Observes and applies interventions to the capture consumed by prediction.
    /// Streaming prefill calls this once per chunk before seeding prediction state.
    pub fn target_capture(&mut self, value: &T) -> Result<T, E>
    where
        T: Clone,
    {
        self.tensor(EMBEDDED_TARGET_CAPTURE_PATH, value)
    }

    /// Observes or patches a declared tensor boundary.
    pub fn tensor(&mut self, path: &str, value: &T) -> Result<T, E>
    where
        T: Clone,
    {
        crate::observe_and_intervene(self.tensors.as_mut(), path, value)
    }

    /// Observes or patches the logits passed to proposal sampling.
    pub fn logits(&mut self, value: &L) -> Result<L, E>
    where
        L: Clone,
    {
        crate::observe_and_intervene(self.logits.as_mut(), EMBEDDED_PROPOSAL_LOGITS_PATH, value)
    }
}

impl<T, L, E> Default for EmbeddedPredictionObservers<T, L, E> {
    fn default() -> Self {
        Self::new(crate::NoopObserver, crate::NoopObserver)
    }
}
