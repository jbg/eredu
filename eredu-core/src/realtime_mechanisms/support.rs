//! Exact native realtime preparation support.

use std::num::NonZeroUsize;

use eredu_checkpoint::lowering::WeightLoweringDescriptor;
use eredu_nn::NeuralOperatorCapabilities;

use crate::{
    cache::StateComponentPolicy,
    communication::CommunicationCompletionCapabilities,
    preparation::{
        ReplicatedTextMechanismSupport, StateComponentPlacement, StateLifecycleCapabilities,
        WeightResidencyMechanism,
    },
    SessionCapabilities,
};

/// A native or portable realtime identity must contain non-whitespace text.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
#[error("realtime identity must not be empty")]
pub struct EmptyRealtimeIdentity;

/// Exact nonempty identity used by neutral realtime contracts.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RealtimeIdentity(String);

impl RealtimeIdentity {
    /// Creates an identity while preserving its exact nonempty text.
    pub fn new(value: impl Into<String>) -> Result<Self, EmptyRealtimeIdentity> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(EmptyRealtimeIdentity);
        }
        Ok(Self(value))
    }

    /// Returns the exact identity text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RealtimeIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// One generic mechanism that a neutral realtime executor may require.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum RealtimeMechanism {
    /// Generic tensor construction, indexing, stacking, and slicing.
    TensorOperations,
    /// Backend-neutral tensor operations and architecture neural operations.
    NeuralOperations,
    /// Exact selected checkpoint recipes and source-to-execution lowering.
    ParameterMaterialization,
    /// Resident or bounded immutable-parameter storage.
    ParameterStorage,
    /// Mutable model-state storage matching architecture geometry.
    StateStorage,
    /// Typed coordinate payload storage and retention.
    CoordinateStorage,
    /// Generic logits processing and token sampling.
    Sampling,
    /// Transactional random-state storage and advancement.
    Randomness,
    /// Portable-host to opaque-tensor conversion.
    HostConversion,
    /// Exact completion and resource-retention tracking.
    ExactCompletion,
    /// Submitted tensor, store, validation, queue, and collective retention.
    ResourceRetention,
    /// Generic device or rank-local transfer.
    Transfer,
    /// Named activation observation and intervention.
    Observation,
    /// Opaque collective communication.
    Collectives,
    /// Optional execution timing.
    Timing,
}

/// Collection-independent facilities of one realtime backend implementation.
pub struct RealtimeMechanismFacts {
    /// Exact supported neural primitives.
    pub operators: NeuralOperatorCapabilities,
    /// Supported realtime operations.
    pub mechanisms: Vec<RealtimeMechanism>,
    /// Exact physical parameter storage mechanisms.
    pub residencies: Vec<WeightResidencyMechanism>,
    /// Largest supported tensor-collective cohort.
    pub maximum_tensor_parallel_size: NonZeroUsize,
    /// Native completion and cancellation capabilities.
    pub completion: CommunicationCompletionCapabilities,
    /// General session facilities exposed by the backend.
    pub session: SessionCapabilities,
    /// Native mutable state lifecycle operations.
    pub state: StateLifecycleCapabilities,
    /// Named native observation facilities.
    pub observations: Vec<RealtimeIdentity>,
}

impl RealtimeMechanismFacts {
    /// Declares exact mechanisms; state lifecycle and named observations remain absent.
    pub fn new(
        operators: NeuralOperatorCapabilities,
        mechanisms: impl IntoIterator<Item = RealtimeMechanism>,
        residencies: impl IntoIterator<Item = WeightResidencyMechanism>,
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

    /// Whether the backend supports the exact physical weight storage mechanism.
    pub fn supports_residency(&self, residency: WeightResidencyMechanism) -> bool {
        self.residencies.contains(&residency)
    }

    /// Whether the backend supports collectives for this exact tensor cohort.
    pub fn supports_collectives(&self, tensor_size: usize) -> bool {
        self.mechanisms.contains(&RealtimeMechanism::Collectives)
            && tensor_size <= self.maximum_tensor_parallel_size.get()
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

    /// Reports support for the exact descriptor, including a requested payload transform.
    fn supports_lowering(
        &self,
        descriptor: &WeightLoweringDescriptor,
        transforms_payload: bool,
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

/// Cold support for realtime graphs with both categorical and independent cache profiles.
pub trait RealtimeLoadSupport: RealtimeMechanismSupport {
    /// Side-effect-free cache support for independent selected state profiles.
    fn independent_state_support(&self) -> &impl ReplicatedTextMechanismSupport;
}
