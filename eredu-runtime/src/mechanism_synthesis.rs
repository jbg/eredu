//! Requirement-driven synthesis of exact backend mechanism capabilities.

use eredu_checkpoint::{LinearFormat, SourceTensorEncoding, StoredDtype};
use eredu_core::{
    cache::{StateComponentPolicy, StateResidencyClass},
    checkpoint::TensorDtype,
    SessionCapabilities,
};
use eredu_nn::NeuralOperatorCapabilities;

use crate::{
    AddressableStorageCapabilities, BackendMechanismCapabilities, CacheResidencyPolicy,
    GroupedOperationRequirement, ReplicatedTextParameterRole, ReplicatedTextRequirements,
    ReplicatedTextSelectionRequest, StateComponentMechanism, StateComponentPlacement, StateLayout,
    StateMechanismCapabilities, StateStorageDtype, WeightLoweringCapability,
    WeightLoweringDescriptor, WeightLoweringKind, WeightResidencyMechanism,
};

/// Collection-independent facilities of a backend's mutable-state implementation.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct StateLifecycleCapabilities {
    checkpoint: bool,
    rollback: bool,
    reset: bool,
    prompt_cache: bool,
    observation_retention: bool,
}

impl StateLifecycleCapabilities {
    /// Creates a fail-closed declaration of state lifecycle facilities.
    pub const fn new() -> Self {
        Self {
            checkpoint: false,
            rollback: false,
            reset: false,
            prompt_cache: false,
            observation_retention: false,
        }
    }

    /// Declares exact checkpoint and rollback support.
    pub const fn with_transactions(mut self, checkpoint: bool, rollback: bool) -> Self {
        self.checkpoint = checkpoint;
        self.rollback = rollback;
        self
    }

    /// Declares complete reset support.
    pub const fn with_reset(mut self, supported: bool) -> Self {
        self.reset = supported;
        self
    }

    /// Declares persistence and restoration support for the requested state policy.
    pub const fn with_prompt_cache(mut self, supported: bool) -> Self {
        self.prompt_cache = supported;
        self
    }

    /// Declares retention of state components through observed submissions.
    pub const fn with_observation_retention(mut self, supported: bool) -> Self {
        self.observation_retention = supported;
        self
    }
}

/// Exact backend facts independent of architecture parameter and state collections.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BackendMechanismFacts {
    operators: NeuralOperatorCapabilities,
    weight_residencies: Vec<WeightResidencyMechanism>,
    state: StateLifecycleCapabilities,
    session: SessionCapabilities,
    prompt_cache: bool,
    exact_completion: bool,
    grouped_operations: Vec<GroupedOperationRequirement>,
    indexed_movement: bool,
    addressable_storage: Option<AddressableStorageCapabilities>,
}

impl BackendMechanismFacts {
    /// Declares neural, ordinary residency, and state lifecycle mechanisms.
    /// Optional facilities remain absent until explicitly declared.
    pub fn new(
        operators: NeuralOperatorCapabilities,
        weight_residencies: impl IntoIterator<Item = WeightResidencyMechanism>,
        state: StateLifecycleCapabilities,
    ) -> Self {
        Self {
            operators,
            weight_residencies: weight_residencies.into_iter().collect(),
            state,
            session: SessionCapabilities::default(),
            prompt_cache: false,
            exact_completion: false,
            grouped_operations: Vec::new(),
            indexed_movement: false,
            addressable_storage: None,
        }
    }

    /// Declares the exact capabilities of the resulting session implementation.
    pub const fn with_session(mut self, session: SessionCapabilities) -> Self {
        self.session = session;
        self
    }

    /// Declares prompt-cache persistence mechanisms.
    pub const fn with_prompt_cache(mut self, supported: bool) -> Self {
        self.prompt_cache = supported;
        self
    }

    /// Declares exact completion ownership.
    pub const fn with_exact_completion(mut self, supported: bool) -> Self {
        self.exact_completion = supported;
        self
    }

    /// Declares grouped operation mechanisms.
    pub fn with_grouped_operations(
        mut self,
        operations: impl IntoIterator<Item = GroupedOperationRequirement>,
    ) -> Self {
        self.grouped_operations = operations.into_iter().collect();
        self
    }

    /// Declares indexed discovery, slicing, remapping, and concatenation.
    pub const fn with_indexed_movement(mut self, supported: bool) -> Self {
        self.indexed_movement = supported;
        self
    }

    /// Declares independently addressable storage and exact capacity limits.
    pub const fn with_addressable_storage(
        mut self,
        capabilities: AddressableStorageCapabilities,
    ) -> Self {
        self.addressable_storage = Some(capabilities);
        self
    }
}

/// Side-effect-free native support queries for neutral requirements.
///
/// Implementations report facts about one descriptor or state component. They
/// must not open payloads, allocate tensors, create native resources, or select
/// an architecture. Requirement enumeration and semantic validation belong to
/// [`synthesize_replicated_text_capabilities`].
pub trait ReplicatedTextMechanismSupport {
    /// Reports collection-independent facts for the requested state implementation.
    fn facts(&self, state_policy: &CacheResidencyPolicy) -> BackendMechanismFacts;

    /// Reports native realization of one semantically valid direct descriptor.
    fn supports_direct(&self, descriptor: &WeightLoweringDescriptor) -> bool;

    /// Reports native realization of one semantically valid transform descriptor.
    fn supports_transform(&self, descriptor: &WeightLoweringDescriptor) -> bool;

    /// Resolves native floating-state storage from one architecture-selected source dtype.
    /// Packed embeddings may produce a different native scalar representation.
    /// This metadata-only query must not inspect names or acquire payloads.
    fn floating_state_dtype(&self, source: &TensorDtype) -> Option<StateStorageDtype>;

    /// Reports support for the exact component geometry, native storage dtype, and placement.
    fn supports_state_component(
        &self,
        component: &StateComponentPolicy,
        storage_dtype: StateStorageDtype,
        placement: StateComponentPlacement,
    ) -> bool;
}

/// Enumerates exact lowering and state mechanisms in deterministic requirement order.
///
/// Invalid architecture transforms are left for the selector's structured
/// diagnostic; they never become backend support queries. Absent and tied
/// parameters introduce no descriptor. Equal descriptors retain their first
/// supported occurrence, including across primary and auxiliary requirements.
pub fn synthesize_replicated_text_capabilities(
    requirements: &ReplicatedTextRequirements,
    request: &ReplicatedTextSelectionRequest,
    support: &impl ReplicatedTextMechanismSupport,
) -> BackendMechanismCapabilities {
    let facts = support.facts(request.state());
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
            } else if descriptor.has_valid_transform_geometry()
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

    let floating_dtype = requirements
        .floating_state_source()
        .and_then(|source| support.floating_state_dtype(source))
        .filter(|dtype| dtype.is_floating());
    let mut state =
        synthesize_state_components(requirements.state_layout(), facts.state, |component| {
            let Some(dtype) = StateStorageDtype::resolve(component.dtype(), floating_dtype) else {
                return (None, None);
            };
            let device = support
                .supports_state_component(component, dtype, StateComponentPlacement::Device)
                .then_some(StateComponentPlacement::Device);
            let paged = match component.residency() {
                StateResidencyClass::SealablePaged => support
                    .supports_state_component(component, dtype, StateComponentPlacement::Paged)
                    .then_some(StateComponentPlacement::Paged),
                StateResidencyClass::AlwaysDeviceMutable
                | StateResidencyClass::LayerScopedOffloadable => device,
            };
            (device, paged)
        });

    if let (Some(source), Some(dtype)) = (requirements.floating_state_source(), floating_dtype) {
        state = state.with_floating_state_dtype(source.clone(), dtype);
    }
    let capabilities = BackendMechanismCapabilities::new(
        facts.operators,
        weight_lowerings,
        facts.weight_residencies,
        state,
    )
    .with_session(facts.session)
    .with_prompt_cache(facts.prompt_cache)
    .with_exact_completion(facts.exact_completion)
    .with_grouped_operations(facts.grouped_operations)
    .with_indexed_movement(facts.indexed_movement);
    match facts.addressable_storage {
        Some(storage) => capabilities.with_addressable_storage(storage),
        None => capabilities,
    }
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

impl WeightLoweringDescriptor {
    /// Validates source encoding geometry without choosing a native implementation.
    pub fn has_valid_direct_geometry(&self) -> bool {
        if self.executable().validate().is_err() {
            return false;
        }
        if self.executable() != LinearFormat::Dense
            && (self.packed_axis().is_none()
                || self.packed_axis() != self.logical_shape().len().checked_sub(1))
        {
            return false;
        }
        let same_unpacked_dimensions = |packed_axis: usize| {
            self.physical_shape()
                .iter()
                .zip(self.logical_shape())
                .enumerate()
                .all(|(axis, (physical, logical))| axis == packed_axis || physical == logical)
        };
        match self.source() {
            SourceTensorEncoding::Gguf { ggml_type, .. } => ggml_type
                .block_and_bytes()
                .ok()
                .and_then(|(block, _)| usize::try_from(block).ok())
                .is_some_and(|block| match self.packed_axis() {
                    Some(axis) if same_unpacked_dimensions(axis) => {
                        let physical = self.physical_shape()[axis];
                        let logical = self.logical_shape()[axis];
                        physical >= logical
                            && physical.is_multiple_of(block)
                            && physical - logical < block
                    }
                    Some(_) => false,
                    None => self.physical_shape() == self.logical_shape(),
                }),
            SourceTensorEncoding::Safetensors(StoredDtype::U32)
            | SourceTensorEncoding::RecipeOutput(StoredDtype::U32) => {
                let Some(axis) = self.packed_axis() else {
                    return false;
                };
                if !same_unpacked_dimensions(axis) {
                    return false;
                }
                let bits = match self.executable() {
                    LinearFormat::Affine(format) => usize::try_from(format.bits).ok(),
                    LinearFormat::MxFp4 => Some(4),
                    _ => None,
                };
                bits.is_some_and(|bits| {
                    self.physical_shape()[axis]
                        .checked_mul(32)
                        .zip(self.logical_shape()[axis].checked_mul(bits))
                        .is_some_and(|(physical, logical)| physical == logical)
                }) && self.has_valid_packed_geometry()
            }
            SourceTensorEncoding::Safetensors(StoredDtype::U8)
            | SourceTensorEncoding::RecipeOutput(StoredDtype::U8)
                if self.executable() == LinearFormat::MxFp4 =>
            {
                self.physical_shape() == self.logical_shape()
            }
            _ => {
                self.physical_shape() == self.logical_shape()
                    && (self.executable() == LinearFormat::Dense
                        || self.has_valid_packed_geometry())
            }
        }
    }

    /// Validates an unpacked source and the selected transform's packing geometry.
    pub fn has_valid_transform_geometry(&self) -> bool {
        self.physical_shape() == self.logical_shape()
            && self.packed_axis() == self.logical_shape().len().checked_sub(1)
            && self.has_valid_packed_geometry()
    }

    fn has_valid_packed_geometry(&self) -> bool {
        if self.executable() != LinearFormat::Dense
            && (self.packed_axis().is_none()
                || self.packed_axis() != self.logical_shape().len().checked_sub(1))
        {
            return false;
        }
        let Some(extent) = self.packed_extent() else {
            return false;
        };
        match self.executable() {
            LinearFormat::Affine(format) => {
                format.validate().is_ok()
                    && usize::try_from(format.group_size)
                        .ok()
                        .is_some_and(|group| extent.is_multiple_of(group))
            }
            LinearFormat::MxFp4 => extent.is_multiple_of(32),
            LinearFormat::GgufIQuant { ggml_type, .. } => ggml_type
                .block_and_bytes()
                .ok()
                .and_then(|(block, _)| usize::try_from(block).ok())
                .is_some_and(|block| extent.is_multiple_of(block)),
            LinearFormat::E4M3BlockFp8(format) => format.validate().is_ok(),
            LinearFormat::Dense => true,
        }
    }
}

#[cfg(test)]
mod tests;
