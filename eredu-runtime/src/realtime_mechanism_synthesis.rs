//! Requirement enumeration for realtime mechanisms, independent of native resources.

use std::num::NonZeroUsize;

use eredu_core::{cache::StateComponentPolicy, SessionCapabilities};
use eredu_nn::NeuralOperatorCapabilities;

use crate::{
    CommunicationCompletionCapabilities, ExecutionResidency, RealtimeArchitectureRequirements,
    RealtimeIdentity, RealtimeMechanism, RealtimeMechanismCapabilities, StateComponentPlacement,
    StateLifecycleCapabilities, WeightLoweringCapability, WeightLoweringDescriptor,
    WeightLoweringKind,
};

/// Collection-independent facilities of one realtime backend implementation.
pub struct RealtimeMechanismFacts {
    operators: NeuralOperatorCapabilities,
    mechanisms: Vec<RealtimeMechanism>,
    residencies: Vec<ExecutionResidency>,
    maximum_tensor_parallel_size: NonZeroUsize,
    completion: CommunicationCompletionCapabilities,
    session: SessionCapabilities,
    state: StateLifecycleCapabilities,
    observations: Vec<RealtimeIdentity>,
}

impl RealtimeMechanismFacts {
    /// Declares exact mechanisms; state lifecycle and named observations remain absent.
    pub fn new(
        operators: NeuralOperatorCapabilities,
        mechanisms: impl IntoIterator<Item = RealtimeMechanism>,
        residencies: impl IntoIterator<Item = ExecutionResidency>,
        maximum_tensor_parallel_size: NonZeroUsize,
        completion: CommunicationCompletionCapabilities,
        session: SessionCapabilities,
    ) -> Self {
        Self {
            operators,
            mechanisms: mechanisms.into_iter().collect(),
            residencies: residencies.into_iter().collect(),
            maximum_tensor_parallel_size,
            completion,
            session,
            state: StateLifecycleCapabilities::new(),
            observations: Vec::new(),
        }
    }

    /// Declares exact state lifecycle support independently of component enumeration.
    pub const fn with_state_lifecycle(mut self, state: StateLifecycleCapabilities) -> Self {
        self.state = state;
        self
    }

    /// Declares exact named observation mechanisms implemented by this backend.
    pub fn with_observation_identities(
        mut self,
        observations: impl IntoIterator<Item = RealtimeIdentity>,
    ) -> Self {
        self.observations = observations.into_iter().collect();
        self
    }
}

/// Side-effect-free realtime support facts and per-descriptor native predicates.
///
/// Providers never enumerate architecture requirement collections, open payloads,
/// allocate tensors, or create native queues/groups while answering these queries.
pub trait RealtimeMechanismSupport {
    /// Reports implemented facilities independently of requested architecture mechanisms.
    fn facts(&self) -> RealtimeMechanismFacts;

    /// Reports support for one validated exact descriptor and lowering kind.
    fn supports_lowering(
        &self,
        descriptor: &WeightLoweringDescriptor,
        kind: WeightLoweringKind,
    ) -> bool;

    /// Reports exact placements for device-policy and paged-policy execution, respectively.
    fn state_component_placements(
        &self,
        component: &StateComponentPolicy,
    ) -> (
        Option<StateComponentPlacement>,
        Option<StateComponentPlacement>,
    );
}

/// Synthesizes a stable exact report from every source/execution alternative.
///
/// Equal supported lowering descriptors retain their first occurrence. State
/// traversal and lifecycle propagation are shared with replicated-text synthesis.
pub fn synthesize_realtime_capabilities(
    requirements: &RealtimeArchitectureRequirements,
    support: &impl RealtimeMechanismSupport,
) -> RealtimeMechanismCapabilities {
    let facts = support.facts();
    let mut lowerings = Vec::new();
    for lowering in requirements
        .executions()
        .iter()
        .flat_map(|execution| execution.weight_lowerings())
    {
        let descriptor = lowering.descriptor();
        let valid = match lowering.kind() {
            WeightLoweringKind::Direct | WeightLoweringKind::Derived => {
                descriptor.has_valid_direct_geometry()
            }
            WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform => {
                descriptor.has_valid_transform_geometry()
            }
        };
        if valid && support.supports_lowering(descriptor, lowering.kind()) {
            let capability =
                WeightLoweringCapability::new(lowering.descriptor().clone(), lowering.kind());
            if !lowerings.contains(&capability) {
                lowerings.push(capability);
            }
        }
    }
    let state = crate::mechanism_synthesis::synthesize_state_components(
        requirements.state_layout(),
        facts.state,
        |component| support.state_component_placements(component),
    );
    RealtimeMechanismCapabilities::new(
        facts.operators,
        facts.mechanisms,
        facts.residencies,
        lowerings,
        state,
        facts.maximum_tensor_parallel_size,
        facts.completion,
        facts.session,
    )
    .with_observation_identities(facts.observations)
}
