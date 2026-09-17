use eredu_core::{
    ArchitectureDescriptor, ArchitectureNodeKind, capture::CaptureError,
    component::ComponentExecutionScopeKind,
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

    pub(crate) fn supports_scope(&self, scope: SpeculativeCaptureScope) -> bool {
        use eredu_runtime::SpeculativeStrategyClass as Strategy;
        match scope {
            SpeculativeCaptureScope::Target => true,
            SpeculativeCaptureScope::Prediction { depth } => {
                self.strategy == Strategy::EmbeddedSequential && depth < self.depth
            }
            SpeculativeCaptureScope::PredictionContext | SpeculativeCaptureScope::FusedProposal => {
                self.strategy == Strategy::EmbeddedFused
            }
        }
    }

    pub(crate) fn retained_scope(
        descriptor: &ArchitectureDescriptor,
        node_id: &str,
    ) -> Result<SpeculativeCaptureScope, eredu_core::speculative::SpeculativeActivationSourceError>
    {
        resolve_scope(descriptor, node_id)
            .map_err(|_| eredu_core::speculative::SpeculativeActivationSourceError::Declaration)
    }

    pub(crate) fn scope_validation_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<(&ArchitectureDescriptor, &str)>(),
            size_of::<
                std::iter::Enumerate<
                    std::slice::Iter<'static, eredu_core::speculative::SpeculativeCaptureBinding>,
                >,
            >(),
            size_of::<std::slice::Iter<'static, eredu_core::speculative::SpeculativeCaptureBinding>>(
            ),
            size_of::<std::slice::Iter<'static, eredu_core::ArchitectureNode>>(),
            size_of::<std::slice::Iter<'static, eredu_core::component::ComponentExecutionScope>>(),
            size_of::<(&eredu_core::ArchitectureNode, Option<&str>)>(),
            size_of::<(usize, SpeculativeCaptureScope)>(),
            size_of::<std::ops::RangeInclusive<usize>>(),
            size_of::<ScopeError<'static>>(),
            size_of::<Result<SpeculativeCaptureScope, ScopeError<'static>>>(),
            size_of::<
                Result<
                    SpeculativeCaptureScope,
                    eredu_core::speculative::SpeculativeActivationSourceError,
                >,
            >(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    pub(crate) fn validate_scope(
        &self,
        scope: SpeculativeCaptureScope,
    ) -> Result<(), CaptureError> {
        let valid = self.supports_scope(scope);
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
    resolve_scope(descriptor, node_id).map_err(|error| match error {
        ScopeError::Duplicate => {
            CaptureError::Invalid("duplicate or unknown speculative invocation root".into())
        }
        ScopeError::Conflict => CaptureError::Invalid(
            "component score scope conflicts with its invocation declaration".into(),
        ),
        ScopeError::Missing(id) => CaptureError::Invalid(format!(
            "speculative capture node absent from architecture: {id}"
        )),
        ScopeError::Undeclared => CaptureError::Unsupported(
            "prediction observation has no declared invocation scope".into(),
        ),
        ScopeError::Cycle => {
            CaptureError::Invalid("cycle in speculative capture node ancestry".into())
        }
    })
}
#[derive(Clone, Copy)]
enum ScopeError<'a> {
    Duplicate,
    Conflict,
    Missing(&'a str),
    Undeclared,
    Cycle,
}
fn resolve_scope<'a>(
    descriptor: &'a ArchitectureDescriptor,
    node_id: &'a str,
) -> Result<SpeculativeCaptureScope, ScopeError<'a>> {
    // The declarations are immutable and normally small. Borrowed prefix scans
    // preserve duplicate validation without reconstructing an ownership map.
    for (index, binding) in descriptor.speculative_invocations.iter().enumerate() {
        if descriptor.node(&binding.node_id).is_none()
            || descriptor.speculative_invocations[..index]
                .iter()
                .any(|earlier| earlier.node_id == binding.node_id)
        {
            return Err(ScopeError::Duplicate);
        }
    }
    for scope in &descriptor.component_scopes {
        let expected = component_scope(&scope.kind);
        if descriptor
            .speculative_invocations
            .iter()
            .find(|binding| binding.node_id == scope.node_id)
            .is_some_and(|binding| binding.scope != expected)
        {
            return Err(ScopeError::Conflict);
        }
    }
    let mut next = Some(node_id);
    for _ in 0..=descriptor.nodes.len() {
        let Some(id) = next else {
            return Ok(SpeculativeCaptureScope::Target);
        };
        let node = descriptor.node(id).ok_or(ScopeError::Missing(id))?;
        if let Some(binding) = descriptor
            .speculative_invocations
            .iter()
            .find(|binding| binding.node_id == id)
        {
            return Ok(binding.scope);
        }
        if let Some(scope) = descriptor
            .component_scopes
            .iter()
            .find(|scope| scope.node_id == id)
        {
            return Ok(component_scope(&scope.kind));
        }
        if node.kind == ArchitectureNodeKind::Prediction {
            return Err(ScopeError::Undeclared);
        }
        next = node.parent.as_deref();
    }
    Err(ScopeError::Cycle)
}
fn component_scope(kind: &ComponentExecutionScopeKind) -> SpeculativeCaptureScope {
    match kind {
        ComponentExecutionScopeKind::Prediction { depth } => {
            SpeculativeCaptureScope::Prediction { depth: *depth }
        }
        ComponentExecutionScopeKind::FusedPrediction => SpeculativeCaptureScope::FusedProposal,
    }
}
