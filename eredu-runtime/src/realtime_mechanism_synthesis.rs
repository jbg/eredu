//! Requirement enumeration for realtime mechanisms, independent of native resources.

pub use eredu_core::realtime_mechanisms::{
    RealtimeLoadSupport, RealtimeMechanismFacts, RealtimeMechanismSupport,
};

use crate::{
    ExecutionResidency, RealtimeArchitectureRequirements, RealtimeMechanismCapabilities,
    WeightLoweringCapability, WeightLoweringKind, WeightResidencyMechanism,
};

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
                descriptor.supports_word_transform()
            }
        };
        if valid && support.supports_lowering(descriptor, lowering.kind().transforms_payload()) {
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
        facts
            .residencies
            .into_iter()
            .filter_map(|residency| match residency {
                WeightResidencyMechanism::Resident => Some(ExecutionResidency::FullyResident),
                WeightResidencyMechanism::Windowed => Some(ExecutionResidency::LayerwiseHost),
                WeightResidencyMechanism::DiskStreamed => Some(ExecutionResidency::DenseDiskStream),
                _ => None,
            }),
        lowerings,
        state,
        facts.maximum_tensor_parallel_size,
        facts.completion,
        facts.session,
    )
    .with_observation_identities(facts.observations)
}
