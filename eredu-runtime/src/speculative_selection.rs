//! Family-blind selection and construction gates for speculative execution.

use std::{collections::BTreeSet, num::NonZeroUsize};

use eredu_core::{
    SpeculativeExecutionTopology, SpeculativeLifecycleObserver, SpeculativeLifecycleStage,
    SpeculativeOutputError, TokenizerCompatibilityProof,
};

/// Stable neutral identity used by speculative selection and cache keys.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SpeculativeIdentity(String);

impl SpeculativeIdentity {
    /// Creates a nonempty stable identity.
    pub fn new(value: impl Into<String>) -> Result<Self, SpeculativeContractError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(SpeculativeContractError::new(
                "speculative identity must not be empty",
            ));
        }
        Ok(Self(value))
    }

    /// Returns the stable identity text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One architecture-selected tensor in an ordered target capture.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpeculativeCaptureEntry {
    path: SpeculativeIdentity,
    shape: Vec<usize>,
    bounded_dimensions: BTreeSet<usize>,
    owner: SpeculativeIdentity,
    observation: SpeculativeIdentity,
}

impl SpeculativeCaptureEntry {
    /// Creates one exact capture entry.
    pub fn new(
        path: SpeculativeIdentity,
        shape: Vec<usize>,
        owner: SpeculativeIdentity,
        observation: SpeculativeIdentity,
    ) -> Result<Self, SpeculativeContractError> {
        if shape.is_empty() || shape.contains(&0) {
            return Err(SpeculativeContractError::new(
                "speculative capture shape must have positive extents",
            ));
        }
        Ok(Self {
            path,
            shape,
            bounded_dimensions: BTreeSet::new(),
            owner,
            observation,
        })
    }

    /// Marks one dimension as request-sized up to its construction-time bound.
    pub fn with_bounded_dimension(
        mut self,
        dimension: usize,
    ) -> Result<Self, SpeculativeContractError> {
        if dimension >= self.shape.len() {
            return Err(SpeculativeContractError::new(
                "bounded capture dimension is outside the declared shape",
            ));
        }
        self.bounded_dimensions.insert(dimension);
        Ok(self)
    }

    /// Returns the architecture observation path.
    pub const fn path(&self) -> &SpeculativeIdentity {
        &self.path
    }

    /// Returns the exact logical tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns dimensions whose exact positive extent is closed by the request.
    pub const fn bounded_dimensions(&self) -> &BTreeSet<usize> {
        &self.bounded_dimensions
    }

    /// Returns the rank or component that owns publication.
    pub const fn owner(&self) -> &SpeculativeIdentity {
        &self.owner
    }

    /// Returns the observation seam identity.
    pub const fn observation(&self) -> &SpeculativeIdentity {
        &self.observation
    }
}

/// Exact ordered target-capture contract declared by an architecture.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpeculativeCaptureSchema {
    identity: SpeculativeIdentity,
    entries: Vec<SpeculativeCaptureEntry>,
}

impl SpeculativeCaptureSchema {
    /// Creates a nonempty schema with unique paths and observation identities.
    pub fn new(
        identity: SpeculativeIdentity,
        entries: impl IntoIterator<Item = SpeculativeCaptureEntry>,
    ) -> Result<Self, SpeculativeContractError> {
        let entries = entries.into_iter().collect::<Vec<_>>();
        if entries.is_empty() {
            return Err(SpeculativeContractError::new(
                "speculative capture schema must not be empty",
            ));
        }
        let paths = entries
            .iter()
            .map(SpeculativeCaptureEntry::path)
            .collect::<BTreeSet<_>>();
        if paths.len() != entries.len() {
            return Err(SpeculativeContractError::new(
                "speculative capture paths must be unique",
            ));
        }
        let observations = entries
            .iter()
            .map(SpeculativeCaptureEntry::observation)
            .collect::<BTreeSet<_>>();
        if observations.len() != entries.len() {
            return Err(SpeculativeContractError::new(
                "speculative capture observation identities must be unique",
            ));
        }
        Ok(Self { identity, entries })
    }

    /// Returns the stable schema identity.
    pub const fn identity(&self) -> &SpeculativeIdentity {
        &self.identity
    }

    /// Returns capture entries in semantic consumption order.
    pub fn entries(&self) -> &[SpeculativeCaptureEntry] {
        &self.entries
    }

    /// Closes request-sized dimensions while retaining paths, order, ownership, and observations.
    pub fn instantiate(
        &self,
        shapes: impl IntoIterator<Item = Vec<usize>>,
    ) -> Result<Self, SpeculativeCaptureError> {
        let shapes = shapes.into_iter().collect::<Vec<_>>();
        if shapes.len() != self.entries.len() {
            return Err(SpeculativeCaptureError::ValueCount {
                expected: self.entries.len(),
                actual: shapes.len(),
            });
        }
        let mut entries = Vec::with_capacity(self.entries.len());
        for (expected, shape) in self.entries.iter().zip(shapes) {
            if shape.len() != expected.shape.len()
                || shape.contains(&0)
                || shape.iter().enumerate().any(|(dimension, actual)| {
                    let maximum = expected.shape[dimension];
                    if expected.bounded_dimensions.contains(&dimension) {
                        *actual > maximum
                    } else {
                        *actual != maximum
                    }
                })
            {
                return Err(SpeculativeCaptureError::ShapeMismatch);
            }
            let mut entry = expected.clone();
            entry.shape = shape;
            entry.bounded_dimensions.clear();
            entries.push(entry);
        }
        Ok(Self {
            identity: self.identity.clone(),
            entries,
        })
    }

    fn admits(&self, actual: &Self) -> bool {
        self.identity == actual.identity
            && self.entries.len() == actual.entries.len()
            && self
                .entries
                .iter()
                .zip(&actual.entries)
                .all(|(expected, actual)| {
                    expected.path == actual.path
                        && expected.owner == actual.owner
                        && expected.observation == actual.observation
                        && expected.shape.len() == actual.shape.len()
                        && actual.bounded_dimensions.is_empty()
                        && expected
                            .shape
                            .iter()
                            .enumerate()
                            .all(|(dimension, maximum)| {
                                let actual = actual.shape[dimension];
                                actual > 0
                                    && if expected.bounded_dimensions.contains(&dimension) {
                                        actual <= *maximum
                                    } else {
                                        actual == *maximum
                                    }
                            })
                })
    }
}

/// Capture metadata proven by the already selected ordinary target.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpeculativeCaptureMetadata {
    schema: SpeculativeCaptureSchema,
    generation: u64,
}

impl SpeculativeCaptureMetadata {
    /// Couples an exact schema to the ordinary-target generation that produced it.
    pub const fn new(schema: SpeculativeCaptureSchema, generation: u64) -> Self {
        Self { schema, generation }
    }

    /// Returns the exact ordered schema.
    pub const fn schema(&self) -> &SpeculativeCaptureSchema {
        &self.schema
    }

    /// Returns the ordinary-target generation.
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

/// Architecture-selected tensors paired with exact capture metadata.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpeculativeCaptureEnvelope<T> {
    metadata: SpeculativeCaptureMetadata,
    values: Vec<T>,
}

impl<T> SpeculativeCaptureEnvelope<T> {
    /// Creates an envelope with exactly one value per ordered schema entry.
    pub fn new(
        metadata: SpeculativeCaptureMetadata,
        values: Vec<T>,
    ) -> Result<Self, SpeculativeCaptureError> {
        if values.len() != metadata.schema.entries.len() {
            return Err(SpeculativeCaptureError::ValueCount {
                expected: metadata.schema.entries.len(),
                actual: values.len(),
            });
        }
        Ok(Self { metadata, values })
    }

    /// Returns the schema and generation proven for these values.
    pub const fn metadata(&self) -> &SpeculativeCaptureMetadata {
        &self.metadata
    }

    /// Returns values in schema order.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Consumes the envelope into ordered values.
    pub fn into_values(self) -> Vec<T> {
        self.values
    }

    /// Validates exact schema and generation equality before proposal state changes.
    pub fn validate_against(
        &self,
        schema: &SpeculativeCaptureSchema,
        generation: u64,
    ) -> Result<(), SpeculativeCaptureError> {
        if !schema.admits(&self.metadata.schema) {
            return Err(SpeculativeCaptureError::SchemaMismatch);
        }
        if self.metadata.generation != generation {
            return Err(SpeculativeCaptureError::GenerationMismatch {
                expected: generation,
                actual: self.metadata.generation,
            });
        }
        Ok(())
    }
}

/// Invalid target-capture envelope.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum SpeculativeCaptureError {
    /// The envelope cardinality differs from its schema.
    #[error("speculative capture expected {expected} values, received {actual}")]
    ValueCount {
        /// Required entry count.
        expected: usize,
        /// Supplied value count.
        actual: usize,
    },
    /// Paths, order, shapes, ownership, or observation identities differ.
    #[error("speculative capture schema does not match the selected architecture")]
    SchemaMismatch,
    /// One request-sized tensor shape exceeds or disagrees with its selected contract.
    #[error("speculative capture shape does not match the selected architecture")]
    ShapeMismatch,
    /// The capture belongs to another ordinary-target generation.
    #[error("speculative capture generation {actual} does not match {expected}")]
    GenerationMismatch {
        /// Required generation.
        expected: u64,
        /// Supplied generation.
        actual: u64,
    },
    /// The lane was created for another selected model realization.
    #[error("speculative lane identity does not match the selected realization")]
    RealizationMismatch,
}

/// Family-neutral prediction strategy class.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum SpeculativeStrategyClass {
    /// Ordered checkpoint-embedded prediction depths.
    EmbeddedSequential,
    /// One checkpoint-embedded fused proposal block.
    EmbeddedFused,
    /// A separately materialized assistant.
    External,
}

/// Exact architecture-owned strategy requirements.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpeculativeStrategyRequirements {
    class: SpeculativeStrategyClass,
    identity: SpeculativeIdentity,
    proposal_capacity: NonZeroUsize,
    tokenizer_fingerprint: Option<[u8; 32]>,
}

impl SpeculativeStrategyRequirements {
    /// Creates an embedded strategy, which needs no external-tokenizer proof.
    pub fn embedded(
        class: SpeculativeStrategyClass,
        identity: SpeculativeIdentity,
        proposal_capacity: NonZeroUsize,
    ) -> Result<Self, SpeculativeContractError> {
        match class {
            SpeculativeStrategyClass::EmbeddedSequential
            | SpeculativeStrategyClass::EmbeddedFused => Ok(Self {
                class,
                identity,
                proposal_capacity,
                tokenizer_fingerprint: None,
            }),
            SpeculativeStrategyClass::External => Err(SpeculativeContractError::new(
                "external strategy requires an external-tokenizer identity",
            )),
        }
    }

    /// Creates an external strategy with the exact tokenizer compatibility identity.
    pub const fn external(
        identity: SpeculativeIdentity,
        proposal_capacity: NonZeroUsize,
        tokenizer_fingerprint: [u8; 32],
    ) -> Self {
        Self {
            class: SpeculativeStrategyClass::External,
            identity,
            proposal_capacity,
            tokenizer_fingerprint: Some(tokenizer_fingerprint),
        }
    }

    /// Returns the selected semantic strategy class.
    pub const fn class(&self) -> SpeculativeStrategyClass {
        self.class
    }

    /// Returns the exact architecture strategy identity.
    pub const fn identity(&self) -> &SpeculativeIdentity {
        &self.identity
    }

    /// Returns the maximum proposal count admitted by the architecture.
    pub const fn proposal_capacity(&self) -> NonZeroUsize {
        self.proposal_capacity
    }

    /// Returns the external tokenizer fingerprint, when required.
    pub const fn tokenizer_fingerprint(&self) -> Option<[u8; 32]> {
        self.tokenizer_fingerprint
    }
}

/// Generic backend mechanism needed by speculative composition.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum SpeculativeMechanism {
    /// Backend-native tensor indexing, concatenation, and shape-preserving movement.
    TensorOperations,
    /// General neural operators needed to construct and execute the target or draft.
    NeuralOperations,
    /// Routed or grouped neural operators required by a selected model family.
    GroupedNeuralOperations,
    /// Hyper-connection and hyper-head operators required by a selected model family.
    HyperNeuralOperations,
    /// Checkpoint payload decoding and backend-native parameter materialization.
    PayloadMaterialization,
    /// Generic logits processing and probability construction.
    LogitsProcessing,
    /// Generic categorical or deterministic token selection.
    Sampling,
    /// Forkable random state and stable random substreams.
    Randomness,
    /// Architecture-shaped prediction or assistant state storage.
    StateStorage,
    /// Placement and residency of materialized payload and mutable speculative state.
    StorageResidency,
    /// Exact completion retaining submitted resources.
    ExactCompletion,
    /// Architecture-declared activation observation and replacement.
    Observation,
    /// Component timing collection and attribution.
    Timing,
    /// Native target and draft execution queue binding.
    QueueBinding,
    /// Communication required by an architecture-declared distributed realization.
    Communication,
    /// Exact target/draft agreement before canonical state is committed.
    Agreement,
    /// Publication of only canonically committed tokens and semantic events.
    Publication,
    /// Ordered handoff between distinct queues on one device.
    SameDeviceHandoff,
    /// Shape- and dtype-preserving transfer between devices.
    CrossDeviceTransfer,
}

impl SpeculativeMechanism {
    const BASE: [Self; 13] = [
        Self::TensorOperations,
        Self::NeuralOperations,
        Self::PayloadMaterialization,
        Self::LogitsProcessing,
        Self::Sampling,
        Self::Randomness,
        Self::StateStorage,
        Self::StorageResidency,
        Self::ExactCompletion,
        Self::Observation,
        Self::QueueBinding,
        Self::Agreement,
        Self::Publication,
    ];
}

/// Exact generic mechanisms required by one strategy.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpeculativeMechanismRequirements {
    mechanisms: BTreeSet<SpeculativeMechanism>,
}

impl SpeculativeMechanismRequirements {
    /// Creates requirements containing the mandatory transaction mechanisms and additions.
    pub fn new(additional: impl IntoIterator<Item = SpeculativeMechanism>) -> Self {
        Self {
            mechanisms: SpeculativeMechanism::BASE
                .into_iter()
                .chain(additional)
                .collect(),
        }
    }

    /// Returns required mechanisms in stable order.
    pub const fn mechanisms(&self) -> &BTreeSet<SpeculativeMechanism> {
        &self.mechanisms
    }
}

impl Default for SpeculativeMechanismRequirements {
    fn default() -> Self {
        Self::new([])
    }
}

/// Generic speculative mechanisms implemented by a backend.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpeculativeMechanismCapabilities {
    mechanisms: BTreeSet<SpeculativeMechanism>,
}

impl SpeculativeMechanismCapabilities {
    /// Creates a fail-closed capability set.
    pub fn new(mechanisms: impl IntoIterator<Item = SpeculativeMechanism>) -> Self {
        Self {
            mechanisms: mechanisms.into_iter().collect(),
        }
    }

    /// Returns supported mechanisms in stable order.
    pub const fn mechanisms(&self) -> &BTreeSet<SpeculativeMechanism> {
        &self.mechanisms
    }

    /// Returns whether one exact generic mechanism is supported.
    pub fn supports(&self, mechanism: SpeculativeMechanism) -> bool {
        self.mechanisms.contains(&mechanism)
    }
}

/// Facade-requested target/draft placement, resolved before queue construction.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum SpeculativePlacementRequest {
    /// Target and draft share one ordered queue.
    Single,
    /// Target and draft use distinct queues on one device.
    SameDeviceSplit,
    /// Target and draft use queues on different devices.
    CrossDeviceSplit,
}

impl SpeculativePlacementRequest {
    /// Reconstructs a selection request from portable execution-plan topology.
    pub fn from_topology(
        topology: SpeculativeExecutionTopology,
    ) -> Result<Self, SpeculativeContractError> {
        match topology {
            SpeculativeExecutionTopology::Single => Ok(Self::Single),
            SpeculativeExecutionTopology::SameDeviceSplit => Ok(Self::SameDeviceSplit),
            SpeculativeExecutionTopology::CrossDeviceSplit => Ok(Self::CrossDeviceSplit),
            _ => Err(SpeculativeContractError::new(
                "unsupported speculative execution topology",
            )),
        }
    }

    fn topology(self) -> SpeculativeExecutionTopology {
        match self {
            Self::Single => SpeculativeExecutionTopology::Single,
            Self::SameDeviceSplit => SpeculativeExecutionTopology::SameDeviceSplit,
            Self::CrossDeviceSplit => SpeculativeExecutionTopology::CrossDeviceSplit,
        }
    }

    fn required_mechanism(self) -> Option<SpeculativeMechanism> {
        match self {
            Self::Single => None,
            Self::SameDeviceSplit => Some(SpeculativeMechanism::SameDeviceHandoff),
            Self::CrossDeviceSplit => Some(SpeculativeMechanism::CrossDeviceTransfer),
        }
    }
}

/// Placement proven against generic backend mechanisms.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SelectedSpeculativePlacement {
    topology: SpeculativeExecutionTopology,
}

impl SelectedSpeculativePlacement {
    /// Returns the portable selected execution topology.
    pub const fn topology(self) -> SpeculativeExecutionTopology {
        self.topology
    }

    /// Whether two distinct target/draft queues are selected.
    pub const fn is_split(self) -> bool {
        !matches!(self.topology, SpeculativeExecutionTopology::Single)
    }

    /// Whether selected tensors require physical cross-device transfer.
    pub const fn crosses_devices(self) -> bool {
        matches!(
            self.topology,
            SpeculativeExecutionTopology::CrossDeviceSplit
        )
    }
}

/// Stable semantic inputs to target, prediction, assistant, and cache identity.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpeculativeStateCacheIdentityIngredients {
    target: SpeculativeIdentity,
    strategy: SpeculativeIdentity,
    assistant: Option<SpeculativeIdentity>,
    tokenizer: Option<[u8; 32]>,
    artifact: SpeculativeIdentity,
    format: SpeculativeIdentity,
    model_topology: SpeculativeIdentity,
    rank: usize,
    processor: SpeculativeIdentity,
    state_components: Vec<SpeculativeIdentity>,
}

impl SpeculativeStateCacheIdentityIngredients {
    /// Creates exact identity ingredients with unique, nonempty state components.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: SpeculativeIdentity,
        strategy: SpeculativeIdentity,
        assistant: Option<SpeculativeIdentity>,
        tokenizer: Option<[u8; 32]>,
        artifact: SpeculativeIdentity,
        format: SpeculativeIdentity,
        model_topology: SpeculativeIdentity,
        rank: usize,
        processor: SpeculativeIdentity,
        state_components: Vec<SpeculativeIdentity>,
    ) -> Result<Self, SpeculativeContractError> {
        if state_components.is_empty() {
            return Err(SpeculativeContractError::new(
                "speculative state identity must contain at least one component",
            ));
        }
        if state_components.iter().collect::<BTreeSet<_>>().len() != state_components.len() {
            return Err(SpeculativeContractError::new(
                "speculative state component identities must be unique",
            ));
        }
        Ok(Self {
            target,
            strategy,
            assistant,
            tokenizer,
            artifact,
            format,
            model_topology,
            rank,
            processor,
            state_components,
        })
    }

    /// Returns the ordinary target identity.
    pub const fn target(&self) -> &SpeculativeIdentity {
        &self.target
    }

    /// Returns the embedded or external strategy identity.
    pub const fn strategy(&self) -> &SpeculativeIdentity {
        &self.strategy
    }

    /// Returns the assistant identity for separate drafting.
    pub const fn assistant(&self) -> Option<&SpeculativeIdentity> {
        self.assistant.as_ref()
    }

    /// Returns the tokenizer compatibility identity when required.
    pub const fn tokenizer(&self) -> Option<[u8; 32]> {
        self.tokenizer
    }

    /// Returns the admitted artifact identity.
    pub const fn artifact(&self) -> &SpeculativeIdentity {
        &self.artifact
    }

    /// Returns the admitted checkpoint-format identity.
    pub const fn format(&self) -> &SpeculativeIdentity {
        &self.format
    }

    /// Returns the ordinary target's model topology identity.
    pub const fn model_topology(&self) -> &SpeculativeIdentity {
        &self.model_topology
    }

    /// Returns the architecture-global rank identity.
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the selected input-processor contract identity.
    pub const fn processor(&self) -> &SpeculativeIdentity {
        &self.processor
    }

    /// Returns architecture state components in persistence order.
    pub fn state_components(&self) -> &[SpeculativeIdentity] {
        &self.state_components
    }
}

/// Architecture compatibility proof produced before backend construction.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpeculativeArchitectureCompatibilityProof {
    target: SpeculativeIdentity,
    strategy: SpeculativeIdentity,
    capture_schema: SpeculativeIdentity,
}

impl SpeculativeArchitectureCompatibilityProof {
    /// Creates a proof over target, strategy, and capture schema identities.
    pub const fn new(
        target: SpeculativeIdentity,
        strategy: SpeculativeIdentity,
        capture_schema: SpeculativeIdentity,
    ) -> Self {
        Self {
            target,
            strategy,
            capture_schema,
        }
    }
}

/// Complete neutral requirements for one speculative realization.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpeculativeRealizationRequirements {
    target: SpeculativeIdentity,
    strategy: SpeculativeStrategyRequirements,
    capture: SpeculativeCaptureSchema,
    mechanisms: SpeculativeMechanismRequirements,
    state: SpeculativeStateCacheIdentityIngredients,
}

impl SpeculativeRealizationRequirements {
    /// Creates requirements and validates cross-contract identity equality.
    pub fn new(
        target: SpeculativeIdentity,
        strategy: SpeculativeStrategyRequirements,
        capture: SpeculativeCaptureSchema,
        mechanisms: SpeculativeMechanismRequirements,
        state: SpeculativeStateCacheIdentityIngredients,
    ) -> Result<Self, SpeculativeContractError> {
        if state.target != target || state.strategy != strategy.identity {
            return Err(SpeculativeContractError::new(
                "speculative state identity differs from target or strategy requirements",
            ));
        }
        if state.tokenizer != strategy.tokenizer_fingerprint {
            return Err(SpeculativeContractError::new(
                "speculative tokenizer identity differs between strategy and cache state",
            ));
        }
        match strategy.class {
            SpeculativeStrategyClass::External if state.assistant.is_none() => {
                return Err(SpeculativeContractError::new(
                    "external strategy requires an assistant cache identity",
                ));
            }
            SpeculativeStrategyClass::EmbeddedSequential
            | SpeculativeStrategyClass::EmbeddedFused
                if state.assistant.is_some() =>
            {
                return Err(SpeculativeContractError::new(
                    "embedded strategy cannot carry an external assistant identity",
                ));
            }
            _ => {}
        }
        Ok(Self {
            target,
            strategy,
            capture,
            mechanisms,
            state,
        })
    }

    /// Returns the already selected ordinary target identity.
    pub const fn target(&self) -> &SpeculativeIdentity {
        &self.target
    }

    /// Returns exact strategy requirements.
    pub const fn strategy(&self) -> &SpeculativeStrategyRequirements {
        &self.strategy
    }

    /// Returns the required target-capture schema.
    pub const fn capture(&self) -> &SpeculativeCaptureSchema {
        &self.capture
    }

    /// Returns required generic mechanisms.
    pub const fn mechanisms(&self) -> &SpeculativeMechanismRequirements {
        &self.mechanisms
    }

    /// Returns exact state/cache identity ingredients.
    pub const fn state(&self) -> &SpeculativeStateCacheIdentityIngredients {
        &self.state
    }
}

/// Preconstruction proofs and placement requested for one realization.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpeculativeSelectionRequest {
    placement: SpeculativePlacementRequest,
    architecture: Option<SpeculativeArchitectureCompatibilityProof>,
    tokenizer: Option<TokenizerCompatibilityProof>,
    capture: SpeculativeCaptureSchema,
}

impl SpeculativeSelectionRequest {
    /// Creates a fail-closed request without compatibility proofs.
    pub const fn new(
        placement: SpeculativePlacementRequest,
        capture: SpeculativeCaptureSchema,
    ) -> Self {
        Self {
            placement,
            architecture: None,
            tokenizer: None,
            capture,
        }
    }

    /// Supplies the architecture-owned target/strategy compatibility proof.
    pub fn with_architecture_proof(
        mut self,
        proof: SpeculativeArchitectureCompatibilityProof,
    ) -> Self {
        self.architecture = Some(proof);
        self
    }

    /// Supplies the facade-owned tokenizer compatibility proof.
    pub fn with_tokenizer_proof(mut self, proof: TokenizerCompatibilityProof) -> Self {
        self.tokenizer = Some(proof);
        self
    }
}

/// Selected generic sampling mechanisms.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SelectedSpeculativeSampling;

/// Selected exact-completion policy.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SelectedSpeculativeCompletion;

/// State/cache identity paired with selected target/draft placement.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedSpeculativeState {
    identity: SpeculativeStateCacheIdentityIngredients,
    placement: SpeculativeExecutionTopology,
}

/// Request-local cache identity completed after input preparation.
///
/// Model realization deliberately excludes these values because neither the
/// prepared input nor a target capture generation exists while payloads,
/// modules, state storage, and queues are selected. The lane identity closes
/// that gap immediately before execution.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SpeculativeLaneIdentity {
    realization: SelectedSpeculativeState,
    prepared_input: SpeculativeIdentity,
    capture_generation: u64,
}

impl SpeculativeLaneIdentity {
    /// Returns the model-level state identity and selected placement.
    pub const fn realization(&self) -> &SelectedSpeculativeState {
        &self.realization
    }

    /// Returns the exact prepared-input identity for this lane.
    pub const fn prepared_input(&self) -> &SpeculativeIdentity {
        &self.prepared_input
    }

    /// Returns the ordinary-target generation expected by this lane.
    pub const fn capture_generation(&self) -> u64 {
        self.capture_generation
    }
}

impl SelectedSpeculativeState {
    /// Returns exact architecture and artifact identity ingredients.
    pub const fn identity(&self) -> &SpeculativeStateCacheIdentityIngredients {
        &self.identity
    }

    /// Returns target/draft placement included in cache identity.
    pub const fn placement(&self) -> SpeculativeExecutionTopology {
        self.placement
    }
}

/// One authoritative speculative realization selected before native work.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedSpeculativeRealization {
    requirements: SpeculativeRealizationRequirements,
    placement: SelectedSpeculativePlacement,
    sampling: SelectedSpeculativeSampling,
    state: SelectedSpeculativeState,
    completion: SelectedSpeculativeCompletion,
}

impl SelectedSpeculativeRealization {
    /// Returns the complete architecture requirements selected once.
    pub const fn requirements(&self) -> &SpeculativeRealizationRequirements {
        &self.requirements
    }

    /// Returns the selected target/draft placement.
    pub const fn placement(&self) -> SelectedSpeculativePlacement {
        self.placement
    }

    /// Returns selected generic sampling facilities.
    pub const fn sampling(&self) -> SelectedSpeculativeSampling {
        self.sampling
    }

    /// Returns selected state and cache identity.
    pub const fn state(&self) -> &SelectedSpeculativeState {
        &self.state
    }

    /// Returns selected exact-completion facilities.
    pub const fn completion(&self) -> SelectedSpeculativeCompletion {
        self.completion
    }

    /// Completes cache identity with request-local input and capture values.
    pub fn lane_identity(
        &self,
        prepared_input: SpeculativeIdentity,
        capture_generation: u64,
    ) -> SpeculativeLaneIdentity {
        SpeculativeLaneIdentity {
            realization: self.state.clone(),
            prepared_input,
            capture_generation,
        }
    }

    /// Validates an actual target capture at the lane boundary.
    pub fn validate_capture<T>(
        &self,
        lane: &SpeculativeLaneIdentity,
        capture: &SpeculativeCaptureEnvelope<T>,
    ) -> Result<(), SpeculativeCaptureError> {
        if lane.realization != self.state {
            return Err(SpeculativeCaptureError::RealizationMismatch);
        }
        capture.validate_against(&self.requirements.capture, lane.capture_generation)
    }
}

/// Fail-closed speculative selection diagnostic.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("speculative realization is unsupported: {issues}", issues = .issues.join("; "))]
pub struct SpeculativeSelectionError {
    issues: Vec<String>,
}

impl SpeculativeSelectionError {
    /// Returns every mismatch in stable validation order.
    pub fn issues(&self) -> &[String] {
        &self.issues
    }
}

/// Selects the complete speculative realization without invoking native construction.
pub fn select_speculative_realization(
    requirements: &SpeculativeRealizationRequirements,
    request: &SpeculativeSelectionRequest,
    capabilities: &SpeculativeMechanismCapabilities,
) -> Result<SelectedSpeculativeRealization, SpeculativeSelectionError> {
    let mut issues = Vec::new();
    match &request.architecture {
        Some(proof)
            if proof.target == requirements.target
                && proof.strategy == requirements.strategy.identity
                && proof.capture_schema == requirements.capture.identity => {}
        Some(_) => issues.push("architecture compatibility proof identity mismatch".into()),
        None => issues.push("architecture compatibility proof is missing".into()),
    }
    match (
        requirements.strategy.tokenizer_fingerprint,
        request.tokenizer.as_ref(),
    ) {
        (Some(expected), Some(proof)) if proof.fingerprint() == expected => {}
        (Some(_), Some(_)) => issues.push("tokenizer compatibility proof identity mismatch".into()),
        (Some(_), None) => issues.push("tokenizer compatibility proof is missing".into()),
        (None, Some(_)) => {
            issues.push("embedded strategy received an external tokenizer proof".into())
        }
        (None, None) => {}
    }
    if request.capture != requirements.capture {
        issues.push("target capture path, order, shape, owner, or observation mismatch".into());
    }
    for mechanism in requirements.mechanisms.mechanisms() {
        if !capabilities.supports(*mechanism) {
            issues.push(format!("missing speculative mechanism {mechanism:?}"));
        }
    }
    if let Some(mechanism) = request.placement.required_mechanism() {
        if !capabilities.supports(mechanism) {
            issues.push(format!(
                "placement {:?} requires speculative mechanism {mechanism:?}",
                request.placement
            ));
        }
    }
    if !issues.is_empty() {
        return Err(SpeculativeSelectionError { issues });
    }
    let placement = SelectedSpeculativePlacement {
        topology: request.placement.topology(),
    };
    Ok(SelectedSpeculativeRealization {
        requirements: requirements.clone(),
        placement,
        sampling: SelectedSpeculativeSampling,
        state: SelectedSpeculativeState {
            identity: requirements.state.clone(),
            placement: placement.topology,
        },
        completion: SelectedSpeculativeCompletion,
    })
}

/// Resources constructed only after a speculative realization is fully selected.
#[derive(Debug)]
pub struct ConstructedSpeculativeResources<P, M, S, Q, T> {
    payload: P,
    modules: M,
    state: S,
    queues: Q,
    transfer: Option<T>,
}

impl<P, M, S, Q, T> ConstructedSpeculativeResources<P, M, S, Q, T> {
    /// Consumes resources in construction order.
    pub fn into_parts(self) -> (P, M, S, Q, Option<T>) {
        (
            self.payload,
            self.modules,
            self.state,
            self.queues,
            self.transfer,
        )
    }
}

/// A selected realization paired with backend-native mechanism resources.
#[derive(Debug)]
pub struct PreparedSpeculativeRealization<P, M, S, Q, T> {
    selected: SelectedSpeculativeRealization,
    resources: ConstructedSpeculativeResources<P, M, S, Q, T>,
}

impl<P, M, S, Q, T> PreparedSpeculativeRealization<P, M, S, Q, T> {
    /// Returns the authoritative neutral selection.
    pub const fn selected(&self) -> &SelectedSpeculativeRealization {
        &self.selected
    }

    /// Consumes the prepared realization into selection and resources.
    pub fn into_parts(
        self,
    ) -> (
        SelectedSpeculativeRealization,
        ConstructedSpeculativeResources<P, M, S, Q, T>,
    ) {
        (self.selected, self.resources)
    }
}

/// Failure after or during the preconstruction selection gate.
#[derive(Debug, thiserror::Error)]
pub enum SpeculativePreparationError<E> {
    /// Explicit lifecycle observation rejected work before the named boundary.
    #[error(transparent)]
    Lifecycle(#[from] SpeculativeOutputError),
    /// Neutral selection rejected before any construction callback.
    #[error(transparent)]
    Selection(#[from] SpeculativeSelectionError),
    /// Checkpoint payload construction failed.
    #[error("speculative payload construction failed")]
    Payload(#[source] E),
    /// Architecture module construction failed.
    #[error("speculative module construction failed")]
    Modules(#[source] E),
    /// Prediction or assistant state construction failed.
    #[error("speculative state construction failed")]
    State(#[source] E),
    /// Target/draft queue construction failed.
    #[error("speculative queue construction failed")]
    Queues(#[source] E),
    /// Cross-device transfer construction failed.
    #[error("speculative transfer construction failed")]
    Transfer(#[source] E),
}

/// Selects once, then invokes family-blind construction callbacks in dependency order.
#[allow(clippy::too_many_arguments)]
pub fn select_and_prepare_speculative_realization<P, M, S, Q, T, E>(
    requirements: &SpeculativeRealizationRequirements,
    request: &SpeculativeSelectionRequest,
    capabilities: &SpeculativeMechanismCapabilities,
    payload: impl FnOnce(&SelectedSpeculativeRealization) -> Result<P, E>,
    modules: impl FnOnce(&SelectedSpeculativeRealization, &P) -> Result<M, E>,
    state: impl FnOnce(&SelectedSpeculativeRealization, &M) -> Result<S, E>,
    queues: impl FnOnce(&SelectedSpeculativeRealization) -> Result<Q, E>,
    transfer: impl FnOnce(&SelectedSpeculativeRealization, &Q) -> Result<T, E>,
) -> Result<PreparedSpeculativeRealization<P, M, S, Q, T>, SpeculativePreparationError<E>> {
    select_and_prepare_speculative_realization_observed(
        requirements,
        request,
        capabilities,
        &|_| Ok(()),
        payload,
        modules,
        state,
        queues,
        transfer,
    )
}

/// Selects once, then observes and constructs every native resource in dependency order.
///
/// Lifecycle observation runs before its associated work. In particular, an
/// admission or compatibility failure invokes no construction callback, and a
/// transfer failure occurs before any cross-device transfer callback.
#[allow(clippy::too_many_arguments)]
pub fn select_and_prepare_speculative_realization_observed<P, M, S, Q, T, E>(
    requirements: &SpeculativeRealizationRequirements,
    request: &SpeculativeSelectionRequest,
    capabilities: &SpeculativeMechanismCapabilities,
    observer: &impl SpeculativeLifecycleObserver,
    payload: impl FnOnce(&SelectedSpeculativeRealization) -> Result<P, E>,
    modules: impl FnOnce(&SelectedSpeculativeRealization, &P) -> Result<M, E>,
    state: impl FnOnce(&SelectedSpeculativeRealization, &M) -> Result<S, E>,
    queues: impl FnOnce(&SelectedSpeculativeRealization) -> Result<Q, E>,
    transfer: impl FnOnce(&SelectedSpeculativeRealization, &Q) -> Result<T, E>,
) -> Result<PreparedSpeculativeRealization<P, M, S, Q, T>, SpeculativePreparationError<E>> {
    observer.observe(SpeculativeLifecycleStage::Admission)?;
    let selected = select_speculative_realization(requirements, request, capabilities)?;
    observer.observe(SpeculativeLifecycleStage::Compatibility)?;
    observer.observe(SpeculativeLifecycleStage::Input)?;
    let payload = payload(&selected).map_err(SpeculativePreparationError::Payload)?;
    observer.observe(SpeculativeLifecycleStage::Execution)?;
    let modules = modules(&selected, &payload).map_err(SpeculativePreparationError::Modules)?;
    let state = state(&selected, &modules).map_err(SpeculativePreparationError::State)?;
    let queues = queues(&selected).map_err(SpeculativePreparationError::Queues)?;
    let transfer = if selected.placement.crosses_devices() {
        observer.observe(SpeculativeLifecycleStage::Transfer)?;
        Some(transfer(&selected, &queues).map_err(SpeculativePreparationError::Transfer)?)
    } else {
        None
    };
    Ok(PreparedSpeculativeRealization {
        selected,
        resources: ConstructedSpeculativeResources {
            payload,
            modules,
            state,
            queues,
            transfer,
        },
    })
}

/// Invalid neutral speculative contract.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("invalid speculative contract: {message}")]
pub struct SpeculativeContractError {
    message: String,
}

impl SpeculativeContractError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::{Cell, RefCell},
        convert::Infallible,
        rc::Rc,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };

    use super::*;

    fn id(value: &str) -> SpeculativeIdentity {
        SpeculativeIdentity::new(value).unwrap()
    }

    fn capture() -> SpeculativeCaptureSchema {
        SpeculativeCaptureSchema::new(
            id("capture-v1"),
            [
                SpeculativeCaptureEntry::new(
                    id("layers.1.output"),
                    vec![1, 2, 8],
                    id("output-rank"),
                    id("layer-1-seam"),
                )
                .unwrap(),
                SpeculativeCaptureEntry::new(
                    id("layers.3.output"),
                    vec![1, 2, 8],
                    id("output-rank"),
                    id("layer-3-seam"),
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    fn requirements(class: SpeculativeStrategyClass) -> SpeculativeRealizationRequirements {
        requirements_with_mechanisms(class, [])
    }

    fn requirements_with_mechanisms(
        class: SpeculativeStrategyClass,
        additional_mechanisms: impl IntoIterator<Item = SpeculativeMechanism>,
    ) -> SpeculativeRealizationRequirements {
        let target = id("target-v1");
        let strategy_id = id("strategy-v1");
        let (strategy, assistant, tokenizer) = match class {
            SpeculativeStrategyClass::External => (
                SpeculativeStrategyRequirements::external(
                    strategy_id.clone(),
                    NonZeroUsize::new(4).unwrap(),
                    [7; 32],
                ),
                Some(id("assistant-v1")),
                Some([7; 32]),
            ),
            _ => (
                SpeculativeStrategyRequirements::embedded(
                    class,
                    strategy_id.clone(),
                    NonZeroUsize::new(4).unwrap(),
                )
                .unwrap(),
                None,
                None,
            ),
        };
        let state = SpeculativeStateCacheIdentityIngredients::new(
            target.clone(),
            strategy_id,
            assistant,
            tokenizer,
            id("artifact-v1"),
            id("safetensors"),
            id("tp2-rank0"),
            0,
            id("text-processor-v1"),
            vec![id("target-state"), id("prediction-state")],
        )
        .unwrap();
        SpeculativeRealizationRequirements::new(
            target,
            strategy,
            capture(),
            SpeculativeMechanismRequirements::new(additional_mechanisms),
            state,
        )
        .unwrap()
    }

    fn capabilities() -> SpeculativeMechanismCapabilities {
        SpeculativeMechanismCapabilities::new(SpeculativeMechanism::BASE.into_iter().chain([
            SpeculativeMechanism::GroupedNeuralOperations,
            SpeculativeMechanism::HyperNeuralOperations,
            SpeculativeMechanism::Timing,
            SpeculativeMechanism::Communication,
            SpeculativeMechanism::SameDeviceHandoff,
            SpeculativeMechanism::CrossDeviceTransfer,
        ]))
    }

    fn request(
        requirements: &SpeculativeRealizationRequirements,
        placement: SpeculativePlacementRequest,
    ) -> SpeculativeSelectionRequest {
        let mut request = SpeculativeSelectionRequest::new(placement, requirements.capture.clone())
            .with_architecture_proof(SpeculativeArchitectureCompatibilityProof::new(
                requirements.target.clone(),
                requirements.strategy.identity.clone(),
                requirements.capture.identity.clone(),
            ));
        if let Some(tokenizer) = requirements.strategy.tokenizer_fingerprint {
            request = request.with_tokenizer_proof(
                TokenizerCompatibilityProof::prove(tokenizer, tokenizer).unwrap(),
            );
        }
        request
    }

    fn construction_counter() -> (Rc<Cell<usize>>, impl Fn() -> usize) {
        let counter = Rc::new(Cell::new(0));
        let read = Rc::clone(&counter);
        (counter, move || read.get())
    }

    #[test]
    fn exact_selection_precedes_every_construction_callback() {
        let requirements = requirements(SpeculativeStrategyClass::External);
        let request = request(&requirements, SpeculativePlacementRequest::CrossDeviceSplit);
        let (counter, calls) = construction_counter();
        let prepared = select_and_prepare_speculative_realization(
            &requirements,
            &request,
            &capabilities(),
            {
                let counter = Rc::clone(&counter);
                move |_| {
                    assert_eq!(counter.get(), 0);
                    counter.set(1);
                    Ok::<_, Infallible>("payload")
                }
            },
            {
                let counter = Rc::clone(&counter);
                move |_, payload| {
                    assert_eq!((*payload, counter.get()), ("payload", 1));
                    counter.set(2);
                    Ok::<_, Infallible>("modules")
                }
            },
            {
                let counter = Rc::clone(&counter);
                move |_, modules| {
                    assert_eq!((*modules, counter.get()), ("modules", 2));
                    counter.set(3);
                    Ok::<_, Infallible>("state")
                }
            },
            {
                let counter = Rc::clone(&counter);
                move |_| {
                    assert_eq!(counter.get(), 3);
                    counter.set(4);
                    Ok::<_, Infallible>("queues")
                }
            },
            {
                let counter = Rc::clone(&counter);
                move |selected, queues| {
                    assert!(selected.placement().crosses_devices());
                    assert_eq!((*queues, counter.get()), ("queues", 4));
                    counter.set(5);
                    Ok::<_, Infallible>("transfer")
                }
            },
        )
        .unwrap();
        assert_eq!(calls(), 5);
        assert_eq!(
            prepared.selected().state().placement(),
            SpeculativeExecutionTopology::CrossDeviceSplit
        );
        let (_, resources) = prepared.into_parts();
        assert_eq!(
            resources.into_parts(),
            ("payload", "modules", "state", "queues", Some("transfer"))
        );
    }

    #[test]
    fn production_lifecycle_observation_precedes_each_construction_boundary() {
        let requirements = requirements(SpeculativeStrategyClass::External);
        let request = request(&requirements, SpeculativePlacementRequest::CrossDeviceSplit);
        let observed = Arc::new(Mutex::new(Vec::new()));
        let observer = {
            let observed = Arc::clone(&observed);
            move |stage| {
                observed.lock().unwrap().push(stage);
                Ok(())
            }
        };
        select_and_prepare_speculative_realization_observed(
            &requirements,
            &request,
            &capabilities(),
            &observer,
            |_| Ok::<_, Infallible>("payload"),
            |_, _| Ok("modules"),
            |_, _| Ok("state"),
            |_| Ok("queues"),
            |_, _| Ok("transfer"),
        )
        .unwrap();
        assert_eq!(
            *observed.lock().unwrap(),
            [
                SpeculativeLifecycleStage::Admission,
                SpeculativeLifecycleStage::Compatibility,
                SpeculativeLifecycleStage::Input,
                SpeculativeLifecycleStage::Execution,
                SpeculativeLifecycleStage::Transfer,
            ]
        );
    }

    #[test]
    fn lifecycle_failure_prevents_its_native_construction_boundary() {
        let requirements = requirements(SpeculativeStrategyClass::External);
        let request = request(&requirements, SpeculativePlacementRequest::CrossDeviceSplit);
        for (failure, expected_callbacks) in [
            (SpeculativeLifecycleStage::Admission, 0),
            (SpeculativeLifecycleStage::Compatibility, 0),
            (SpeculativeLifecycleStage::Input, 0),
            (SpeculativeLifecycleStage::Execution, 1),
            (SpeculativeLifecycleStage::Transfer, 4),
        ] {
            let callbacks = Arc::new(AtomicUsize::new(0));
            let observer = move |stage| {
                if stage == failure {
                    Err(SpeculativeOutputError::semantic(
                        "lifecycle observation",
                        format!("failed at {stage:?}"),
                    ))
                } else {
                    Ok(())
                }
            };
            let increment = || {
                callbacks.fetch_add(1, Ordering::SeqCst);
                Ok::<_, Infallible>(())
            };
            let result = select_and_prepare_speculative_realization_observed(
                &requirements,
                &request,
                &capabilities(),
                &observer,
                |_| increment(),
                |_, _| increment(),
                |_, _| increment(),
                |_| increment(),
                |_, _| increment(),
            );
            assert!(matches!(
                result,
                Err(SpeculativePreparationError::Lifecycle(_))
            ));
            assert_eq!(callbacks.load(Ordering::SeqCst), expected_callbacks);
        }

        let observed = Arc::new(Mutex::new(Vec::new()));
        let observer = {
            let observed = Arc::clone(&observed);
            move |stage| {
                observed.lock().unwrap().push(stage);
                Ok(())
            }
        };
        let mut incompatible = request.clone();
        incompatible.architecture = None;
        let result = select_and_prepare_speculative_realization_observed(
            &requirements,
            &incompatible,
            &capabilities(),
            &observer,
            |_| Ok::<_, Infallible>(()),
            |_, _| Ok(()),
            |_, _| Ok(()),
            |_| Ok(()),
            |_, _| Ok(()),
        );
        assert!(matches!(
            result,
            Err(SpeculativePreparationError::Selection(_))
        ));
        assert_eq!(
            *observed.lock().unwrap(),
            [SpeculativeLifecycleStage::Admission]
        );
    }

    #[test]
    fn missing_mechanisms_and_placement_fail_before_native_work() {
        let requirements = requirements_with_mechanisms(
            SpeculativeStrategyClass::EmbeddedSequential,
            [
                SpeculativeMechanism::GroupedNeuralOperations,
                SpeculativeMechanism::HyperNeuralOperations,
                SpeculativeMechanism::Timing,
                SpeculativeMechanism::Communication,
            ],
        );
        let cases = SpeculativeMechanism::BASE
            .into_iter()
            .chain([
                SpeculativeMechanism::GroupedNeuralOperations,
                SpeculativeMechanism::HyperNeuralOperations,
                SpeculativeMechanism::Timing,
                SpeculativeMechanism::Communication,
            ])
            .map(|mechanism| (mechanism, SpeculativePlacementRequest::Single))
            .chain([
                (
                    SpeculativeMechanism::SameDeviceHandoff,
                    SpeculativePlacementRequest::SameDeviceSplit,
                ),
                (
                    SpeculativeMechanism::CrossDeviceTransfer,
                    SpeculativePlacementRequest::CrossDeviceSplit,
                ),
            ]);
        for (missing, placement) in cases {
            let supported = capabilities();
            let available = supported
                .mechanisms()
                .iter()
                .copied()
                .filter(|mechanism| *mechanism != missing);
            let (counter, calls) = construction_counter();
            let result = select_and_prepare_speculative_realization(
                &requirements,
                &request(&requirements, placement),
                &SpeculativeMechanismCapabilities::new(available),
                {
                    let counter = Rc::clone(&counter);
                    move |_| {
                        counter.set(counter.get() + 1);
                        Ok::<_, Infallible>(())
                    }
                },
                |_, _| Ok(()),
                |_, _| Ok(()),
                |_| Ok(()),
                |_, _| Ok(()),
            );
            assert!(matches!(
                result,
                Err(SpeculativePreparationError::Selection(_))
            ));
            assert_eq!(calls(), 0, "missing {missing:?} reached payload work");
        }
    }

    #[test]
    fn construction_failure_stops_each_later_stage() {
        let requirements = requirements(SpeculativeStrategyClass::External);
        let request = request(&requirements, SpeculativePlacementRequest::CrossDeviceSplit);
        for (failure, expected_work) in [
            ("payload", &["payload"][..]),
            ("modules", &["payload", "modules"][..]),
            ("state", &["payload", "modules", "state"][..]),
            ("queues", &["payload", "modules", "state", "queues"][..]),
            (
                "transfer",
                &["payload", "modules", "state", "queues", "transfer"][..],
            ),
        ] {
            let trace = Rc::new(RefCell::new(Vec::new()));
            let result = select_and_prepare_speculative_realization(
                &requirements,
                &request,
                &capabilities(),
                {
                    let trace = Rc::clone(&trace);
                    move |_| {
                        trace.borrow_mut().push("payload");
                        if failure == "payload" {
                            Err("payload failure")
                        } else {
                            Ok("payload")
                        }
                    }
                },
                {
                    let trace = Rc::clone(&trace);
                    move |_, _| {
                        trace.borrow_mut().push("modules");
                        if failure == "modules" {
                            Err("modules failure")
                        } else {
                            Ok("modules")
                        }
                    }
                },
                {
                    let trace = Rc::clone(&trace);
                    move |_, _| {
                        trace.borrow_mut().push("state");
                        if failure == "state" {
                            Err("state failure")
                        } else {
                            Ok("state")
                        }
                    }
                },
                {
                    let trace = Rc::clone(&trace);
                    move |_| {
                        trace.borrow_mut().push("queues");
                        if failure == "queues" {
                            Err("queues failure")
                        } else {
                            Ok("queues")
                        }
                    }
                },
                {
                    let trace = Rc::clone(&trace);
                    move |_, _| {
                        trace.borrow_mut().push("transfer");
                        if failure == "transfer" {
                            Err("transfer failure")
                        } else {
                            Ok("transfer")
                        }
                    }
                },
            );
            assert_eq!(&*trace.borrow(), expected_work, "failure at {failure}");
            assert!(
                matches!(
                    (failure, result),
                    (
                        "payload",
                        Err(SpeculativePreparationError::Payload("payload failure"))
                    ) | (
                        "modules",
                        Err(SpeculativePreparationError::Modules("modules failure"))
                    ) | (
                        "state",
                        Err(SpeculativePreparationError::State("state failure"))
                    ) | (
                        "queues",
                        Err(SpeculativePreparationError::Queues("queues failure"))
                    ) | (
                        "transfer",
                        Err(SpeculativePreparationError::Transfer("transfer failure"))
                    )
                ),
                "unexpected result for {failure}"
            );
        }
    }

    #[test]
    fn proof_and_capture_perturbations_fail_before_construction() {
        let requirements = requirements(SpeculativeStrategyClass::External);
        let base = request(&requirements, SpeculativePlacementRequest::Single);
        let mut reordered = requirements.capture.clone();
        reordered.entries.swap(0, 1);
        let mut wrong_shape = requirements.capture.clone();
        wrong_shape.entries[0].shape[2] += 1;
        let mut wrong_owner = requirements.capture.clone();
        wrong_owner.entries[0].owner = id("other-output-rank");
        let mut wrong_observation = requirements.capture.clone();
        wrong_observation.entries[0].observation = id("other-observation-seam");
        let requests = [
            SpeculativeSelectionRequest {
                architecture: None,
                ..base.clone()
            },
            SpeculativeSelectionRequest {
                architecture: Some(SpeculativeArchitectureCompatibilityProof::new(
                    id("other-target"),
                    requirements.strategy.identity.clone(),
                    requirements.capture.identity.clone(),
                )),
                ..base.clone()
            },
            SpeculativeSelectionRequest {
                tokenizer: None,
                ..base.clone()
            },
            SpeculativeSelectionRequest {
                tokenizer: Some(TokenizerCompatibilityProof::prove([8; 32], [8; 32]).unwrap()),
                ..base.clone()
            },
            SpeculativeSelectionRequest {
                capture: reordered,
                ..base.clone()
            },
            SpeculativeSelectionRequest {
                capture: wrong_shape,
                ..base.clone()
            },
            SpeculativeSelectionRequest {
                capture: wrong_owner,
                ..base.clone()
            },
            SpeculativeSelectionRequest {
                capture: wrong_observation,
                ..base
            },
        ];
        for request in requests {
            let trace = Rc::new(RefCell::new(Vec::new()));
            let result = select_and_prepare_speculative_realization(
                &requirements,
                &request,
                &capabilities(),
                {
                    let trace = Rc::clone(&trace);
                    move |_| {
                        trace.borrow_mut().push("payload");
                        Ok::<_, Infallible>(())
                    }
                },
                {
                    let trace = Rc::clone(&trace);
                    move |_, _| {
                        trace.borrow_mut().push("modules");
                        Ok(())
                    }
                },
                {
                    let trace = Rc::clone(&trace);
                    move |_, _| {
                        trace.borrow_mut().push("state");
                        Ok(())
                    }
                },
                {
                    let trace = Rc::clone(&trace);
                    move |_| {
                        trace.borrow_mut().push("queues");
                        Ok(())
                    }
                },
                {
                    let trace = Rc::clone(&trace);
                    move |_, _| {
                        trace.borrow_mut().push("transfer");
                        Ok(())
                    }
                },
            );
            assert!(matches!(
                result,
                Err(SpeculativePreparationError::Selection(_))
            ));
            assert!(trace.borrow().is_empty());
        }
    }

    #[test]
    fn capture_envelope_validates_cardinality_schema_and_generation() {
        let schema = capture();
        let envelope = SpeculativeCaptureEnvelope::new(
            SpeculativeCaptureMetadata::new(schema.clone(), 11),
            vec![10, 30],
        )
        .unwrap();
        envelope.validate_against(&schema, 11).unwrap();
        assert!(matches!(
            envelope.validate_against(&schema, 12),
            Err(SpeculativeCaptureError::GenerationMismatch { .. })
        ));
        let mut reordered = schema.clone();
        reordered.entries.swap(0, 1);
        assert_eq!(
            envelope.validate_against(&reordered, 11),
            Err(SpeculativeCaptureError::SchemaMismatch)
        );
        assert!(matches!(
            SpeculativeCaptureEnvelope::new(SpeculativeCaptureMetadata::new(schema, 11), vec![10]),
            Err(SpeculativeCaptureError::ValueCount { .. })
        ));
    }

    #[test]
    fn request_sized_capture_dimensions_close_exactly_at_the_lane_boundary() {
        let schema = SpeculativeCaptureSchema::new(
            id("bounded-capture-v1"),
            [SpeculativeCaptureEntry::new(
                id("layers.1.output"),
                vec![1, 128, 8],
                id("output-rank"),
                id("layer-1-seam"),
            )
            .unwrap()
            .with_bounded_dimension(1)
            .unwrap()],
        )
        .unwrap();
        let actual = schema.instantiate([vec![1, 7, 8]]).unwrap();
        let envelope =
            SpeculativeCaptureEnvelope::new(SpeculativeCaptureMetadata::new(actual, 4), vec![10])
                .unwrap();
        envelope.validate_against(&schema, 4).unwrap();

        assert_eq!(
            schema.instantiate([vec![1, 129, 8]]),
            Err(SpeculativeCaptureError::ShapeMismatch)
        );
        assert_eq!(
            schema.instantiate([vec![1, 7, 9]]),
            Err(SpeculativeCaptureError::ShapeMismatch)
        );
    }

    #[test]
    fn request_local_identity_is_closed_only_after_model_selection() {
        let requirements = requirements(SpeculativeStrategyClass::EmbeddedSequential);
        let selected = select_speculative_realization(
            &requirements,
            &request(&requirements, SpeculativePlacementRequest::Single),
            &capabilities(),
        )
        .unwrap();
        let lane = selected.lane_identity(id("prepared-input-42"), 11);
        let envelope = SpeculativeCaptureEnvelope::new(
            SpeculativeCaptureMetadata::new(requirements.capture.clone(), 11),
            vec![10, 30],
        )
        .unwrap();
        selected.validate_capture(&lane, &envelope).unwrap();
        assert_eq!(lane.prepared_input().as_str(), "prepared-input-42");
        assert_eq!(lane.capture_generation(), 11);

        let other = select_speculative_realization(
            &requirements,
            &request(&requirements, SpeculativePlacementRequest::SameDeviceSplit),
            &capabilities(),
        )
        .unwrap();
        assert_eq!(
            other.validate_capture(&lane, &envelope),
            Err(SpeculativeCaptureError::RealizationMismatch)
        );
    }
}
