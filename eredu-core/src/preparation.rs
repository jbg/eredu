//! Side-effect-free native preparation facts. Selection belongs to the caller.

/// Exact geometry and native support for row decoding.
pub mod rows;

use eredu_checkpoint::lowering::WeightLoweringDescriptor;
use eredu_nn::{GroupedOperationRequirement, NeuralOperatorCapabilities};

use crate::{
    cache::{StateTensorDimension, StateTensorDtype},
    checkpoint::TensorDtype,
    residency::AddressableStorageCapabilities,
    PreparationMechanismCapabilities, SessionCapabilities,
};

/// Weight-residency mechanism implemented by a backend.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum WeightResidencyMechanism {
    /// All parameters remain device resident.
    Resident,
    /// A bounded device window is staged from host storage.
    Windowed,
    /// Bounded host and device windows are populated from disk.
    DiskStreamed,
}

/// Physical placement selected for one semantic mutable-state component.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum StateComponentPlacement {
    /// The mutable component remains on the execution device.
    Device,
    /// The append-only component is managed by bounded paged storage.
    Paged,
}

/// Physical scalar representation selected for native mutable state.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum StateStorageDtype {
    /// IEEE half precision.
    F16,
    /// Brain floating point.
    Bf16,
    /// IEEE single precision.
    F32,
    /// IEEE double precision.
    F64,
    /// Two IEEE single-precision components.
    Complex64,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer.
    U32,
}

impl StateStorageDtype {
    /// Exact neutral tensor scalar representation used by typed state bindings.
    pub const fn element_type(self) -> eredu_nn::TensorElementType {
        use eredu_nn::TensorElementType as Element;
        match self {
            Self::F16 => Element::F16,
            Self::Bf16 => Element::Bf16,
            Self::F32 => Element::F32,
            Self::F64 => Element::F64,
            Self::Complex64 => Element::Complex64,
            Self::I32 => Element::I32,
            Self::U32 => Element::U32,
        }
    }
    /// Exact bytes occupied by one native state element.
    pub const fn bytes(self) -> std::num::NonZeroU8 {
        let bytes = match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::Complex64 => 8,
        };
        std::num::NonZeroU8::new(bytes).unwrap()
    }

    /// Whether this representation belongs to the model's floating state family.
    pub const fn is_floating(self) -> bool {
        !matches!(self, Self::I32 | Self::U32)
    }

    /// Resolves an architecture dtype policy without overriding fixed-width tensors.
    pub const fn resolve(policy: StateTensorDtype, floating: Option<Self>) -> Option<Self> {
        match policy {
            StateTensorDtype::Floating => match floating {
                Some(dtype) if dtype.is_floating() => Some(dtype),
                _ => None,
            },
            StateTensorDtype::Float32 => Some(Self::F32),
            StateTensorDtype::Int32 => Some(Self::I32),
            StateTensorDtype::Uint32 => Some(Self::U32),
        }
    }
}

/// Optional observations requested from the backend's residency mechanisms.
///
/// Deriving a request never samples memory or creates a native resource.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ResidencyDiagnostics {
    backend_memory: bool,
    process_memory: bool,
}

impl ResidencyDiagnostics {
    /// Selects backend allocator and process-memory observations independently.
    pub const fn new(backend_memory: bool, process_memory: bool) -> Self {
        Self {
            backend_memory,
            process_memory,
        }
    }
}

impl ResidencyDiagnostics {
    /// Whether to sample native allocator counters.
    pub const fn backend_memory(self) -> bool {
        self.backend_memory
    }
    /// Whether to sample process resident memory.
    pub const fn process_memory(self) -> bool {
        self.process_memory
    }
}
/// Collection-independent facilities of a backend's mutable-state implementation.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct StateLifecycleCapabilities {
    /// Can checkpoint mutable state.
    pub checkpoint: bool,
    /// Can roll back mutable state.
    pub rollback: bool,
    /// Can reset mutable state.
    pub reset: bool,
    /// Can persist and restore prompt state.
    pub prompt_cache: bool,
    /// Retains state through an observed submission.
    pub observation_retention: bool,
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
    /// Supports settled causal-prefix submission and prompt slicing.
    pub chunked_prefill: bool,
    /// Exact supported neural operations.
    pub operators: NeuralOperatorCapabilities,
    /// Physical ordinary weight residency facilities.
    pub weight_residencies: Vec<WeightResidencyMechanism>,
    /// Native state lifecycle facilities.
    pub state: StateLifecycleCapabilities,
    /// Facilities that the configured execution can expose.
    pub session: SessionCapabilities,
    /// Native prompt-cache persistence support.
    pub prompt_cache: bool,
    /// Retains native resources through exact completion.
    pub exact_completion: bool,
    /// Exact grouped operator semantics.
    pub grouped_operations: Vec<GroupedOperationRequirement>,
    /// Supports indexed slicing, remapping and concatenation.
    pub indexed_movement: bool,
    /// Independently addressable storage and its bounds.
    pub addressable_storage: Option<AddressableStorageCapabilities>,
}

impl BackendMechanismFacts {
    /// Declares settled causal-prefix submission and compatible prompt slicing.
    pub const fn with_chunked_prefill(mut self, supported: bool) -> Self {
        self.chunked_prefill = supported;
        self
    }

    /// Declares neural, ordinary residency, and state lifecycle mechanisms.
    /// Optional facilities remain absent until explicitly declared.
    pub fn new(
        operators: NeuralOperatorCapabilities,
        weight_residencies: impl IntoIterator<Item = WeightResidencyMechanism>,
        state: StateLifecycleCapabilities,
    ) -> Self {
        Self {
            chunked_prefill: false,
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
/// the portable capability synthesizer.
pub trait ReplicatedTextMechanismSupport {
    /// Reports collection-independent facts for an exact native storage placement.
    fn facts(&self, placement: StateComponentPlacement) -> BackendMechanismFacts;

    /// Reports native realization of one semantically valid direct descriptor.
    fn supports_direct(&self, descriptor: &WeightLoweringDescriptor) -> bool;

    /// Reports native realization of one semantically valid transform descriptor.
    fn supports_transform(&self, descriptor: &WeightLoweringDescriptor) -> bool;

    /// Resolves native floating-state storage from one architecture-selected source dtype.
    /// Packed embeddings may produce a different native scalar representation.
    /// This metadata-only query must not inspect names or acquire payloads.
    fn floating_state_dtype(&self, source: &TensorDtype) -> Option<StateStorageDtype>;

    /// Reports support for the exact tensor geometry, native storage dtype, and placement.
    fn supports_state_component(
        &self,
        shape: &[StateTensorDimension],
        storage_dtype: StateStorageDtype,
        placement: StateComponentPlacement,
    ) -> bool;
}

/// Side-effect-free support facts supplied for portable preparation.
pub trait NativePreparation {
    /// Floating activation and state representation produced by materializing the
    /// declared embedding dtype. This is an immutable native mechanism fact.
    fn floating_state_dtype(&self, source: &TensorDtype) -> Option<StateStorageDtype> {
        self.replicated_text_support()?.floating_state_dtype(source)
    }

    /// Physical support for exact bounded row decoding.
    fn row_lookup_support(&self) -> Option<&dyn rows::NativeRowLookupSupport> {
        None
    }

    /// Native temporary geometry for explicit input-score attention. No family
    /// identity, tensor, stream or device is consulted.
    fn input_score_attention_workspace(
        &self,
    ) -> Option<crate::attention::InputScoreAttentionMechanism> {
        None
    }

    /// Conservative live buffers for one physical recipe, including native
    /// copies and index buffers. This must inspect metadata only.
    fn recipe_materialization_workspace<C: eredu_checkpoint::recipe::RecipeCatalog + ?Sized>(
        &self,
        _recipe: &eredu_checkpoint::recipe::DerivedWeightRecipe,
        _source: &C,
    ) -> Result<u64, String> {
        Err("backend did not report native recipe workspace".into())
    }

    /// Host observation collector facts, independent of model family and devices.
    fn observation_mechanisms(&self) -> crate::ObservationMechanisms {
        crate::ObservationMechanisms::default()
    }

    /// Bounded native transformations, without allocating or selecting a device.
    fn capture_capabilities(&self) -> crate::capture::CaptureCapabilities {
        Default::default()
    }

    /// Portable artifact, residency, topology, modality, completion, and session mechanisms.
    fn preparation_capabilities(&self) -> PreparationMechanismCapabilities;

    /// Whether one exact grouped-compute semantic is implemented.
    fn supports_grouped_operation(&self, requirement: GroupedOperationRequirement) -> bool;

    /// Exact operator and state support for portable text admission, when implemented.
    fn replicated_text_support(&self) -> Option<&dyn ReplicatedTextMechanismSupport> {
        None
    }

    /// Portable media and prepared-tensor primitives.
    fn processor_capabilities(&self) -> crate::media::MediaPrimitiveCapabilities;

    /// Portable speculative execution mechanisms.
    fn speculative_capabilities(&self) -> crate::speculative::SpeculativeMechanismCapabilities;

    /// Portable communication operations, bounds, framing, and completion support.
    fn communication_capabilities(&self) -> crate::communication::CommunicationCapabilities;
}

/// Backend observations used by facade inspection and memory forecasts.
pub trait BackendInspection: PlanMechanismFacts {
    /// Determines placement from a backend device identifier without allocating it.
    fn device_uses_host_memory(
        &self,
        device: &crate::DevicePlan,
    ) -> Result<bool, crate::BackendFailure>;

    /// Resets the allocator high-water mark when supported.
    fn reset_allocator_peak(&self) -> Result<bool, crate::BackendFailure> {
        Ok(false)
    }

    /// Validates a proposed explicit cache limit without changing allocator state.
    /// Returns the effective ceiling in bytes for cold forecasts.
    fn proposed_allocator_cache_limit(&self, _bytes: u64) -> Result<u64, crate::BackendFailure> {
        Err(crate::BackendFailure::from_error(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "the selected backend does not expose an allocator cache limit",
        ))
        .with_operation("allocator cache policy"))
    }

    /// Applies an explicit allocator cache limit at application startup.
    fn set_allocator_cache_limit(&self, _bytes: u64) -> Result<(), crate::BackendFailure> {
        Err(crate::BackendFailure::from_error(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "the selected backend does not allow changing its allocator cache limit",
        ))
        .with_operation("allocator cache policy"))
    }

    /// Observes allocator overhead inputs if the backend exposes a retained cache.
    fn allocator_observation(
        &self,
    ) -> Result<
        Option<(crate::AllocatorCachePolicyReport, crate::AllocatorTelemetry)>,
        crate::BackendFailure,
    > {
        Ok(None)
    }
}

/// Side-effect-free mechanism inputs supplied to portable model planning.
pub trait PlanMechanismFacts: crate::automatic::ColdPlanPlacement {
    /// Immutable native mechanism support for the portable catalog.
    type Preparation: NativePreparation;

    /// Reports mechanism support without selecting a candidate.
    fn preparation_mechanisms(&self) -> Self::Preparation;
    /// Reports the configured diagnostic observations.
    fn residency_diagnostics(&self) -> ResidencyDiagnostics;
    /// Reports the available prepared-input conversion operations.
    fn media_features(&self) -> crate::MediaFeatureAvailability;
}
