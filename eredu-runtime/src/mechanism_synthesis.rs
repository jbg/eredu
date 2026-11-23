//! Requirement-driven synthesis of exact backend mechanism capabilities.

#[cfg(test)]
use crate::{
    AddressableStorageCapabilities, GroupedOperationRequirement, WeightLoweringDescriptor,
    WeightResidencyMechanism,
};
use eredu_checkpoint::{LinearFormat, SourceTensorEncoding, StoredDtype};
use eredu_core::cache::{StateComponentPolicy, StateResidencyClass};
#[cfg(test)]
use eredu_core::{cache::StateTensorDimension, checkpoint::TensorDtype, SessionCapabilities};
#[cfg(test)]
use eredu_nn::NeuralOperatorCapabilities;

use crate::{
    BackendMechanismCapabilities, CacheResidencyPolicy, ReplicatedTextParameterRole,
    ReplicatedTextRequirements, ReplicatedTextSelectionRequest, StateComponentMechanism,
    StateComponentPlacement, StateLayout, StateMechanismCapabilities, StateStorageDtype,
    WeightLoweringCapability, WeightLoweringKind,
};

pub use eredu_core::preparation::{
    BackendMechanismFacts, ReplicatedTextMechanismSupport, StateLifecycleCapabilities,
};

/// Enumerates exact lowering and state mechanisms in deterministic requirement order.
///
/// Invalid architecture transforms are left for the selector's structured
/// diagnostic; they never become backend support queries. Absent and tied
/// parameters introduce no descriptor. Equal descriptors retain their first
/// supported occurrence, including across primary and auxiliary requirements.
pub fn synthesize_replicated_text_capabilities(
    requirements: &ReplicatedTextRequirements,
    request: &ReplicatedTextSelectionRequest,
    support: &(impl ReplicatedTextMechanismSupport + ?Sized),
) -> BackendMechanismCapabilities {
    let facts = support.facts(state_placement(request.state()));
    let mut weight_lowerings = Vec::new();
    for parameter in requirements
        .parameters()
        .iter()
        .chain(requirements.auxiliary_parameters())
    {
        if !parameter.has_lowering_source() {
            continue;
        }
        let requested = request
            .quantization()
            .and_then(|requested| parameter.transform_target(requested).ok().flatten())
            .map(|target| target.executable());
        for executable in std::iter::once(parameter.native_executable()).chain(requested) {
            let Ok(descriptor) = parameter.lowering_descriptor(executable) else {
                continue;
            };
            if parameter.role() == ReplicatedTextParameterRole::LinearWeight
                && matches!(
                    descriptor.source(),
                    SourceTensorEncoding::Safetensors(StoredDtype::U8)
                )
                && executable == LinearFormat::Dense
            {
                continue;
            }
            let kind = if executable == parameter.native_executable()
                && descriptor.has_valid_direct_geometry()
                && support.supports_direct(&descriptor)
            {
                Some(WeightLoweringKind::Direct)
            } else if descriptor.supports_word_transform()
                && support.supports_transform(&descriptor)
            {
                Some(WeightLoweringKind::Transform)
            } else {
                None
            };
            if let Some(kind) = kind {
                let capability = WeightLoweringCapability::new(descriptor, kind);
                if !weight_lowerings.contains(&capability) {
                    weight_lowerings.push(capability);
                }
            }
        }
    }

    let state = state_capabilities(&requirements.state_requirements(), facts.state, support);
    let capabilities = BackendMechanismCapabilities::new(
        facts.operators,
        weight_lowerings,
        facts.weight_residencies,
        state,
    )
    .with_session(facts.session)
    .with_prompt_cache(facts.prompt_cache)
    .with_exact_completion(facts.exact_completion)
    .with_chunked_prefill(facts.chunked_prefill)
    .with_grouped_operations(facts.grouped_operations)
    .with_indexed_movement(facts.indexed_movement);
    match facts.addressable_storage {
        Some(storage) => capabilities.with_addressable_storage(storage),
        None => capabilities,
    }
}

/// Enumerates one independent state role using existing backend support predicates.
/// This performs no immutable-weight queries and introduces no native resources.
pub fn synthesize_state_capabilities(
    requirements: &crate::StateRealizationRequirements,
    policy: &CacheResidencyPolicy,
    support: &(impl ReplicatedTextMechanismSupport + ?Sized),
) -> StateMechanismCapabilities {
    state_capabilities(
        requirements,
        support.facts(state_placement(policy)).state,
        support,
    )
}

fn state_placement(policy: &CacheResidencyPolicy) -> StateComponentPlacement {
    match policy {
        CacheResidencyPolicy::Paged(_) => StateComponentPlacement::Paged,
        CacheResidencyPolicy::Device => StateComponentPlacement::Device,
    }
}

fn state_capabilities(
    requirements: &crate::StateRealizationRequirements,
    lifecycle: StateLifecycleCapabilities,
    support: &(impl ReplicatedTextMechanismSupport + ?Sized),
) -> StateMechanismCapabilities {
    let floating_dtype = requirements
        .floating_source()
        .and_then(|source| support.floating_state_dtype(source))
        .filter(|dtype| dtype.is_floating());
    let mut state = synthesize_state_components(requirements.layout(), lifecycle, |component| {
        let Some(dtype) = StateStorageDtype::resolve(component.dtype(), floating_dtype) else {
            return (None, None);
        };
        let device = support
            .supports_state_component(component.shape(), dtype, StateComponentPlacement::Device)
            .then_some(StateComponentPlacement::Device);
        let paged = match component.residency() {
            StateResidencyClass::SealablePaged => support
                .supports_state_component(component.shape(), dtype, StateComponentPlacement::Paged)
                .then_some(StateComponentPlacement::Paged),
            StateResidencyClass::AlwaysDeviceMutable
            | StateResidencyClass::LayerScopedOffloadable => device,
        };
        (device, paged)
    });

    if let (Some(source), Some(dtype)) = (requirements.floating_source(), floating_dtype) {
        state = state.with_floating_state_dtype(source.clone(), dtype);
    }
    state
}

/// Shared collection traversal; native providers answer only one component at a time.
pub(crate) fn synthesize_state_components(
    layout: &StateLayout,
    lifecycle: StateLifecycleCapabilities,
    placements: impl Fn(
        &StateComponentPolicy,
    ) -> (
        Option<StateComponentPlacement>,
        Option<StateComponentPlacement>,
    ),
) -> StateMechanismCapabilities {
    StateMechanismCapabilities::new((0..layout.len()).flat_map(|layer| {
        let placements = &placements;
        layout
            .components(layer)
            .expect("validated state layout exposes every layer")
            .iter()
            .filter_map(move |component| {
                let (device, paged) = placements(component);
                (device.is_some() || paged.is_some())
                    .then(|| StateComponentMechanism::new(layer, component.clone(), device, paged))
            })
    }))
    .with_transactions(lifecycle.checkpoint, lifecycle.rollback)
    .with_reset(lifecycle.reset)
    .with_prompt_cache(lifecycle.prompt_cache)
    .with_observation_retention(lifecycle.observation_retention)
}

#[cfg(test)]
mod tests;
