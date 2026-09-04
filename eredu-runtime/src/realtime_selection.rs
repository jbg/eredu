//! Family-blind realtime requirements, selection, and construction gating.

use std::{collections::BTreeSet, num::NonZeroUsize};

use eredu_core::{ParallelTopology, RealtimeSpeechConfig, SessionCapabilities};
use eredu_nn::NeuralOperatorCapabilities;

use crate::{
    ArchitectureParameterDescription, CacheResidencyPolicy, ExecutionGraph, ExecutionResidency,
    ExecutionUnitLayout, LayerWeightResidency, StateComponentMechanism, StateComponentPlacement,
    StateLayout, StateMechanismCapabilities, WeightLoweringCapability, WeightLoweringDescriptor,
    WeightLoweringKind,
};

/// Exact nonempty identity used by neutral realtime contracts.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RealtimeIdentity(String);

impl RealtimeIdentity {
    /// Creates an identity while preserving its exact nonempty text.
    pub fn new(value: impl Into<String>) -> Result<Self, RealtimeContractError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(RealtimeContractError::EmptyIdentity);
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

/// Semantic role of one physical component in an atomic matrix lowering.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RealtimeWeightComponentRole {
    /// Primary matrix weight.
    Primary,
    /// Scale values required by the executable format.
    Scale,
    /// Affine bias values required by the executable format.
    AffineBias,
}

/// One exact physical or transform-generated component of a matrix lowering.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeWeightComponentRequirement {
    target: RealtimeIdentity,
    recipe_owner: Option<RealtimeIdentity>,
    recipe_identity: Option<RealtimeIdentity>,
    recipe: Option<eredu_checkpoint::recipe::DerivedWeightRecipe>,
    recipe_output: Option<eredu_checkpoint::recipe::RecipeMetadata>,
    source_occurrences: Vec<RealtimeIdentity>,
    physical_shape: Vec<usize>,
    role: RealtimeWeightComponentRole,
}

impl RealtimeWeightComponentRequirement {
    /// Creates one recipe-backed source component or one generated target component.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: RealtimeIdentity,
        recipe_owner: Option<RealtimeIdentity>,
        recipe_identity: Option<RealtimeIdentity>,
        recipe: Option<eredu_checkpoint::recipe::DerivedWeightRecipe>,
        recipe_output: Option<eredu_checkpoint::recipe::RecipeMetadata>,
        source_occurrences: impl IntoIterator<Item = RealtimeIdentity>,
        physical_shape: impl Into<Vec<usize>>,
        role: RealtimeWeightComponentRole,
    ) -> Result<Self, RealtimeContractError> {
        let source_occurrences = source_occurrences.into_iter().collect::<Vec<_>>();
        let physical_shape = physical_shape.into();
        if physical_shape.is_empty() || physical_shape.contains(&0) {
            return Err(RealtimeContractError::InvalidWeightComponentShape);
        }
        match (
            &recipe_owner,
            &recipe_identity,
            &recipe,
            &recipe_output,
            source_occurrences.is_empty(),
        ) {
            (Some(_), Some(_), Some(recipe), Some(output), false) => {
                let declared = source_occurrences
                    .iter()
                    .map(RealtimeIdentity::as_str)
                    .collect::<Vec<_>>();
                if recipe.source_occurrences() != declared || output.shape != physical_shape {
                    return Err(RealtimeContractError::WeightComponentRecipeMismatch);
                }
            }
            (Some(_), Some(_), Some(_), Some(_), true) => {
                return Err(RealtimeContractError::EmptyRecipeComponentSources);
            }
            (None, None, None, None, true) => {}
            (None, None, None, None, false) => {
                return Err(RealtimeContractError::GeneratedComponentHasSources);
            }
            _ => return Err(RealtimeContractError::IncompleteWeightComponentRecipe),
        }
        Ok(Self {
            target,
            recipe_owner,
            recipe_identity,
            recipe,
            recipe_output,
            source_occurrences,
            physical_shape,
            role,
        })
    }

    /// Returns the exact executable component target.
    pub const fn target(&self) -> &RealtimeIdentity {
        &self.target
    }

    /// Returns the canonical recipe owner for a source-backed component.
    pub const fn recipe_owner(&self) -> Option<&RealtimeIdentity> {
        self.recipe_owner.as_ref()
    }

    /// Returns the exact recipe identity for a source-backed component.
    pub const fn recipe_identity(&self) -> Option<&RealtimeIdentity> {
        self.recipe_identity.as_ref()
    }

    /// Returns the exact selected architecture recipe for a source-backed component.
    pub const fn recipe(&self) -> Option<&eredu_checkpoint::recipe::DerivedWeightRecipe> {
        self.recipe.as_ref()
    }

    /// Returns exact admission-time output metadata for a source-backed component.
    pub const fn recipe_output(&self) -> Option<&eredu_checkpoint::recipe::RecipeMetadata> {
        self.recipe_output.as_ref()
    }

    /// Returns source identities in recipe traversal order, retaining duplicates.
    pub fn source_occurrences(&self) -> &[RealtimeIdentity] {
        &self.source_occurrences
    }

    /// Returns exact component geometry before rank-local construction.
    pub fn physical_shape(&self) -> &[usize] {
        &self.physical_shape
    }

    /// Returns the component's semantic matrix-family role.
    pub const fn role(&self) -> RealtimeWeightComponentRole {
        self.role
    }

    /// Returns whether this component is produced by an architecture recipe.
    pub const fn is_recipe_backed(&self) -> bool {
        self.recipe_identity.is_some()
    }
}

/// One exact named source-to-execution weight lowering required by an architecture.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeWeightLoweringRequirement {
    target: RealtimeIdentity,
    components: Vec<RealtimeWeightComponentRequirement>,
    descriptor: WeightLoweringDescriptor,
    kind: WeightLoweringKind,
}

impl RealtimeWeightLoweringRequirement {
    /// Binds one executable matrix target to its complete atomic component family.
    pub fn new(
        target: RealtimeIdentity,
        components: impl IntoIterator<Item = RealtimeWeightComponentRequirement>,
        descriptor: WeightLoweringDescriptor,
        kind: WeightLoweringKind,
    ) -> Result<Self, RealtimeContractError> {
        let components = components.into_iter().collect::<Vec<_>>();
        let primary = components
            .iter()
            .filter(|component| component.role == RealtimeWeightComponentRole::Primary)
            .collect::<Vec<_>>();
        match primary.as_slice() {
            [] => return Err(RealtimeContractError::MissingPrimaryWeightComponent),
            [primary] if primary.target != target => {
                return Err(RealtimeContractError::PrimaryWeightComponentTargetMismatch);
            }
            [_] => {}
            _ => return Err(RealtimeContractError::DuplicatePrimaryWeightComponent),
        }
        let targets = components
            .iter()
            .map(RealtimeWeightComponentRequirement::target)
            .collect::<BTreeSet<_>>();
        if targets.len() != components.len() {
            return Err(RealtimeContractError::DuplicateWeightComponentTarget);
        }
        for role in [
            RealtimeWeightComponentRole::Scale,
            RealtimeWeightComponentRole::AffineBias,
        ] {
            if components
                .iter()
                .filter(|component| component.role == role)
                .count()
                > 1
            {
                return Err(RealtimeContractError::DuplicateWeightComponentRole { role });
            }
        }
        Ok(Self {
            target,
            components,
            descriptor,
            kind,
        })
    }

    /// Returns the exact canonical executable parameter target.
    pub const fn target(&self) -> &RealtimeIdentity {
        &self.target
    }

    /// Returns the complete atomic component family in architecture order.
    pub fn components(&self) -> &[RealtimeWeightComponentRequirement] {
        &self.components
    }

    /// Returns the unique primary component.
    pub fn primary(&self) -> &RealtimeWeightComponentRequirement {
        self.component(RealtimeWeightComponentRole::Primary)
            .expect("validated realtime lowering contains one primary component")
    }

    /// Returns the optional scale companion.
    pub fn scale(&self) -> Option<&RealtimeWeightComponentRequirement> {
        self.component(RealtimeWeightComponentRole::Scale)
    }

    /// Returns the optional affine-bias companion.
    pub fn affine_bias(&self) -> Option<&RealtimeWeightComponentRequirement> {
        self.component(RealtimeWeightComponentRole::AffineBias)
    }

    fn component(
        &self,
        role: RealtimeWeightComponentRole,
    ) -> Option<&RealtimeWeightComponentRequirement> {
        self.components
            .iter()
            .find(|component| component.role == role)
    }

    /// Returns the exact geometry-bearing backend lowering query.
    pub const fn descriptor(&self) -> &WeightLoweringDescriptor {
        &self.descriptor
    }

    /// Returns the required direct, derived, transforming, or combined route.
    pub const fn kind(&self) -> WeightLoweringKind {
        self.kind
    }
}

/// One architecture-admitted execution configuration and its exact lowerings.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeExecutionRequirements {
    identity: RealtimeIdentity,
    parameters: ArchitectureParameterDescription,
    weight_lowerings: Vec<RealtimeWeightLoweringRequirement>,
}

impl RealtimeExecutionRequirements {
    /// Binds a stable execution identity to every geometry-bearing weight lowering.
    pub fn new(
        identity: RealtimeIdentity,
        parameters: ArchitectureParameterDescription,
        weight_lowerings: Vec<RealtimeWeightLoweringRequirement>,
    ) -> Result<Self, RealtimeContractError> {
        if weight_lowerings.is_empty() {
            return Err(RealtimeContractError::EmptyWeightLowerings);
        }
        let targets = weight_lowerings
            .iter()
            .map(RealtimeWeightLoweringRequirement::target)
            .collect::<BTreeSet<_>>();
        if targets.len() != weight_lowerings.len() {
            return Err(RealtimeContractError::DuplicateWeightLoweringTarget);
        }
        let members = parameters
            .groups()
            .iter()
            .flat_map(|group| group.group().members())
            .collect::<Vec<_>>();
        let expected_targets = members
            .iter()
            .filter(|member| member.linear_companion().is_none())
            .map(|member| member.target())
            .collect::<BTreeSet<_>>();
        let actual_targets = weight_lowerings
            .iter()
            .map(|lowering| lowering.target().as_str())
            .collect::<BTreeSet<_>>();
        if expected_targets != actual_targets {
            return Err(RealtimeContractError::WeightLoweringParameterMismatch);
        }
        for lowering in &weight_lowerings {
            let expected_components = members
                .iter()
                .filter(|member| {
                    member.target() == lowering.target().as_str()
                        || member.linear_companion_of() == Some(lowering.target().as_str())
                })
                .map(|member| member.target())
                .collect::<BTreeSet<_>>();
            let actual_components = lowering
                .components()
                .iter()
                .map(|component| component.target().as_str())
                .collect::<BTreeSet<_>>();
            if expected_components != actual_components {
                return Err(RealtimeContractError::WeightComponentParameterMismatch {
                    target: lowering.target().clone(),
                });
            }
        }
        Ok(Self {
            identity,
            parameters,
            weight_lowerings,
        })
    }

    /// Returns the stable selected execution identity.
    pub const fn identity(&self) -> &RealtimeIdentity {
        &self.identity
    }

    /// Returns exact parameter ownership and geometry for this execution format.
    pub const fn parameters(&self) -> &ArchitectureParameterDescription {
        &self.parameters
    }

    /// Returns exact source-to-execution lowerings for every materialized tensor.
    pub fn weight_lowerings(&self) -> &[RealtimeWeightLoweringRequirement] {
        &self.weight_lowerings
    }
}

/// Exact generic mechanisms required by an architecture.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeMechanismRequirements {
    mechanisms: BTreeSet<RealtimeMechanism>,
}

impl RealtimeMechanismRequirements {
    /// Creates a nonempty requirement set.
    pub fn new(
        mechanisms: impl IntoIterator<Item = RealtimeMechanism>,
    ) -> Result<Self, RealtimeContractError> {
        let mechanisms = mechanisms.into_iter().collect::<BTreeSet<_>>();
        if mechanisms.is_empty() {
            return Err(RealtimeContractError::EmptyMechanismRequirements);
        }
        Ok(Self { mechanisms })
    }

    /// Returns required mechanisms in stable order.
    pub const fn mechanisms(&self) -> &BTreeSet<RealtimeMechanism> {
        &self.mechanisms
    }

    /// Returns whether this contract requires a mechanism.
    pub fn requires(&self, mechanism: RealtimeMechanism) -> bool {
        self.mechanisms.contains(&mechanism)
    }
}

/// Architecture-admitted replicated and pure tensor-parallel topology policy.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeTopologyPolicy {
    identity: RealtimeIdentity,
    replicated: bool,
    pure_tensor_parallel_sizes: BTreeSet<usize>,
}

impl RealtimeTopologyPolicy {
    /// Creates an exact fail-closed topology policy.
    pub fn new(
        identity: RealtimeIdentity,
        replicated: bool,
        pure_tensor_parallel_sizes: impl IntoIterator<Item = usize>,
    ) -> Result<Self, RealtimeContractError> {
        let pure_tensor_parallel_sizes = pure_tensor_parallel_sizes
            .into_iter()
            .collect::<BTreeSet<_>>();
        if let Some(size) = pure_tensor_parallel_sizes
            .iter()
            .copied()
            .find(|size| *size < 2)
        {
            return Err(RealtimeContractError::InvalidPureTensorParallelSize { size });
        }
        if !replicated && pure_tensor_parallel_sizes.is_empty() {
            return Err(RealtimeContractError::EmptyTopologyPolicy);
        }
        Ok(Self {
            identity,
            replicated,
            pure_tensor_parallel_sizes,
        })
    }

    /// Returns the stable architecture topology-policy identity.
    pub const fn identity(&self) -> &RealtimeIdentity {
        &self.identity
    }

    /// Returns whether replicated execution is admitted.
    pub const fn admits_replicated(&self) -> bool {
        self.replicated
    }

    /// Returns admitted pure tensor-parallel sizes.
    pub const fn pure_tensor_parallel_sizes(&self) -> &BTreeSet<usize> {
        &self.pure_tensor_parallel_sizes
    }

    /// Returns whether an exact topology is architecture-admitted.
    pub fn admits(&self, topology: ParallelTopology) -> bool {
        if topology.is_replicated() {
            return self.replicated;
        }
        topology.pipeline() == 1
            && topology.expert() == 1
            && topology.data() == 1
            && self.pure_tensor_parallel_sizes.contains(&topology.tensor())
    }
}

/// Exact architecture requirements used for realtime selection.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeArchitectureRequirements {
    architecture: RealtimeIdentity,
    source: RealtimeIdentity,
    execution_graph: ExecutionGraph,
    execution_units: ExecutionUnitLayout,
    source_parameters: ArchitectureParameterDescription,
    executions: Vec<RealtimeExecutionRequirements>,
    operators: NeuralOperatorCapabilities,
    speech_schedule_identity: RealtimeIdentity,
    speech_schedule: RealtimeSpeechConfig,
    state_layout_identity: RealtimeIdentity,
    state_layout: StateLayout,
    mechanisms: RealtimeMechanismRequirements,
    topology: RealtimeTopologyPolicy,
    residencies: Vec<ExecutionResidency>,
}

impl RealtimeArchitectureRequirements {
    /// Creates one exact architecture-owned realtime requirement contract.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        architecture: RealtimeIdentity,
        source: RealtimeIdentity,
        execution_graph: ExecutionGraph,
        execution_units: ExecutionUnitLayout,
        source_parameters: ArchitectureParameterDescription,
        executions: impl IntoIterator<Item = RealtimeExecutionRequirements>,
        operators: NeuralOperatorCapabilities,
        speech_schedule_identity: RealtimeIdentity,
        speech_schedule: RealtimeSpeechConfig,
        state_layout_identity: RealtimeIdentity,
        state_layout: StateLayout,
        mechanisms: RealtimeMechanismRequirements,
        topology: RealtimeTopologyPolicy,
        residencies: impl IntoIterator<Item = ExecutionResidency>,
    ) -> Result<Self, RealtimeContractError> {
        if execution_graph.groups().len() != execution_units.group_count()
            || execution_graph
                .groups()
                .iter()
                .enumerate()
                .any(|(index, group)| {
                    execution_units
                        .group_id(index)
                        .is_none_or(|identity| identity.as_str() != group.id())
                })
        {
            return Err(RealtimeContractError::ExecutionGraphLayoutMismatch);
        }
        if source_parameters.graph() != &execution_graph
            || source_parameters.unit_layout() != &execution_units
        {
            return Err(RealtimeContractError::SourceParameterDescriptionMismatch);
        }
        let executions = executions.into_iter().collect::<Vec<_>>();
        if executions.is_empty() {
            return Err(RealtimeContractError::EmptyExecutionIdentities);
        }
        let execution_identities = executions
            .iter()
            .map(RealtimeExecutionRequirements::identity)
            .collect::<BTreeSet<_>>();
        if execution_identities.len() != executions.len() {
            return Err(RealtimeContractError::DuplicateExecutionIdentity);
        }
        if let Some(execution) = executions.iter().find(|execution| {
            execution.parameters.graph() != &execution_graph
                || execution.parameters.unit_layout() != &execution_units
        }) {
            return Err(
                RealtimeContractError::ExecutionParameterDescriptionMismatch {
                    execution: execution.identity.clone(),
                },
            );
        }
        let residencies = residencies.into_iter().collect::<Vec<_>>();
        if residencies.is_empty() {
            return Err(RealtimeContractError::EmptyResidencies);
        }
        if residencies
            .iter()
            .enumerate()
            .any(|(index, residency)| residencies[..index].contains(residency))
        {
            return Err(RealtimeContractError::DuplicateResidency);
        }
        Ok(Self {
            architecture,
            source,
            execution_graph,
            execution_units,
            source_parameters,
            executions,
            operators,
            speech_schedule_identity,
            speech_schedule,
            state_layout_identity,
            state_layout,
            mechanisms,
            topology,
            residencies,
        })
    }

    /// Returns the exact architecture identity.
    pub const fn architecture(&self) -> &RealtimeIdentity {
        &self.architecture
    }

    /// Returns the exact admitted source artifact identity.
    pub const fn source(&self) -> &RealtimeIdentity {
        &self.source
    }

    /// Returns the exact temporal/depth execution dependency graph.
    pub const fn execution_graph(&self) -> &ExecutionGraph {
        &self.execution_graph
    }

    /// Returns the exact temporal/depth execution-unit layout.
    pub const fn execution_units(&self) -> &ExecutionUnitLayout {
        &self.execution_units
    }

    /// Returns exact architecture-owned source parameter geometry and ownership.
    pub const fn source_parameters(&self) -> &ArchitectureParameterDescription {
        &self.source_parameters
    }

    /// Returns architecture-admitted execution identities.
    pub fn executions(&self) -> &[RealtimeExecutionRequirements] {
        &self.executions
    }

    /// Returns optional neural operators required by the architecture equations.
    pub const fn operators(&self) -> NeuralOperatorCapabilities {
        self.operators
    }

    /// Returns the stable speech schedule identity.
    pub const fn speech_schedule_identity(&self) -> &RealtimeIdentity {
        &self.speech_schedule_identity
    }

    /// Returns exact portable speech geometry and delays.
    pub const fn speech_schedule(&self) -> &RealtimeSpeechConfig {
        &self.speech_schedule
    }

    /// Returns the stable state-layout identity.
    pub const fn state_layout_identity(&self) -> &RealtimeIdentity {
        &self.state_layout_identity
    }

    /// Returns complete architecture-owned mutable-state geometry.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Returns required generic mechanisms.
    pub const fn mechanisms(&self) -> &RealtimeMechanismRequirements {
        &self.mechanisms
    }

    /// Returns architecture-admitted topology policy.
    pub const fn topology(&self) -> &RealtimeTopologyPolicy {
        &self.topology
    }

    /// Returns architecture-admitted residency policies.
    pub fn residencies(&self) -> &[ExecutionResidency] {
        &self.residencies
    }
}

/// Family-blind backend capability report for realtime construction.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeMechanismCapabilities {
    operators: NeuralOperatorCapabilities,
    mechanisms: BTreeSet<RealtimeMechanism>,
    residencies: Vec<ExecutionResidency>,
    weight_lowerings: Vec<WeightLoweringCapability>,
    observation_identities: BTreeSet<RealtimeIdentity>,
    state: StateMechanismCapabilities,
    maximum_tensor_parallel_size: NonZeroUsize,
    completion: crate::CommunicationCompletionCapabilities,
    session: SessionCapabilities,
}

impl RealtimeMechanismCapabilities {
    /// Creates one exact generic mechanism capability report.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        operators: NeuralOperatorCapabilities,
        mechanisms: impl IntoIterator<Item = RealtimeMechanism>,
        residencies: impl IntoIterator<Item = ExecutionResidency>,
        weight_lowerings: Vec<WeightLoweringCapability>,
        state: StateMechanismCapabilities,
        maximum_tensor_parallel_size: NonZeroUsize,
        completion: crate::CommunicationCompletionCapabilities,
        session: SessionCapabilities,
    ) -> Self {
        Self {
            operators,
            mechanisms: mechanisms.into_iter().collect(),
            residencies: residencies.into_iter().collect(),
            weight_lowerings,
            observation_identities: BTreeSet::new(),
            state,
            maximum_tensor_parallel_size,
            completion,
            session,
        }
    }

    /// Returns optional neural operators implemented by this backend path.
    pub const fn operators(&self) -> NeuralOperatorCapabilities {
        self.operators
    }

    /// Returns whether a generic mechanism is implemented.
    pub fn supports(&self, mechanism: RealtimeMechanism) -> bool {
        self.mechanisms.contains(&mechanism)
    }

    /// Returns supported residency policies.
    pub fn residencies(&self) -> &[ExecutionResidency] {
        &self.residencies
    }

    /// Returns exact geometry-bearing source-to-execution lowering mechanisms.
    pub fn weight_lowerings(&self) -> &[WeightLoweringCapability] {
        &self.weight_lowerings
    }

    /// Adds exact named activation observations implemented by this backend path.
    pub fn with_observation_identities(
        mut self,
        observations: impl IntoIterator<Item = RealtimeIdentity>,
    ) -> Self {
        self.observation_identities = observations.into_iter().collect();
        self
    }

    /// Returns exact named activation observations implemented by this backend path.
    pub const fn observation_identities(&self) -> &BTreeSet<RealtimeIdentity> {
        &self.observation_identities
    }

    /// Returns whether one exact named activation observation is implemented.
    pub fn supports_observation(&self, observation: &RealtimeIdentity) -> bool {
        self.observation_identities.contains(observation)
    }

    /// Returns exact mutable-state component and transaction mechanisms.
    pub const fn state(&self) -> &StateMechanismCapabilities {
        &self.state
    }

    /// Returns the maximum supported pure tensor-parallel size.
    pub const fn maximum_tensor_parallel_size(&self) -> NonZeroUsize {
        self.maximum_tensor_parallel_size
    }

    /// Returns supported exact-completion timeout dispositions.
    pub const fn completion(&self) -> &crate::CommunicationCompletionCapabilities {
        &self.completion
    }

    /// Returns generic prepared-session capabilities.
    pub const fn session(&self) -> SessionCapabilities {
        self.session
    }
}

/// Architecture-issued proof for the exact topology and rank request.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeArchitectureProof {
    architecture: RealtimeIdentity,
    speech_schedule: RealtimeIdentity,
    state_layout: RealtimeIdentity,
    topology_policy: RealtimeIdentity,
    topology: ParallelTopology,
    rank: usize,
}

impl RealtimeArchitectureProof {
    /// Binds exact architecture contracts to one requested topology and rank.
    pub fn new(
        architecture: RealtimeIdentity,
        speech_schedule: RealtimeIdentity,
        state_layout: RealtimeIdentity,
        topology_policy: RealtimeIdentity,
        topology: ParallelTopology,
        rank: usize,
    ) -> Self {
        Self {
            architecture,
            speech_schedule,
            state_layout,
            topology_policy,
            topology,
            rank,
        }
    }

    /// Returns the proven topology.
    pub const fn topology(&self) -> ParallelTopology {
        self.topology
    }

    /// Returns the proven global rank.
    pub const fn rank(&self) -> usize {
        self.rank
    }
}

/// Host-visible observation facilities requested at construction time.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct RealtimeObservationRequirements {
    output: bool,
    activations: BTreeSet<RealtimeIdentity>,
}

impl RealtimeObservationRequirements {
    /// Creates exact observation requirements.
    pub fn new(output: bool, activations: impl IntoIterator<Item = RealtimeIdentity>) -> Self {
        Self {
            output,
            activations: activations.into_iter().collect(),
        }
    }

    /// Returns whether completed host output is required.
    pub const fn output(&self) -> bool {
        self.output
    }

    /// Returns exact requested activation observations in stable identity order.
    pub const fn activations(&self) -> &BTreeSet<RealtimeIdentity> {
        &self.activations
    }

    /// Returns whether any named activation observation is required.
    pub fn requires_activations(&self) -> bool {
        !self.activations.is_empty()
    }
}

/// Exact source, execution, placement, and observation selection request.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeSelectionRequest {
    source: RealtimeIdentity,
    execution: RealtimeIdentity,
    residency: LayerWeightResidency,
    state: CacheResidencyPolicy,
    architecture_proof: Option<RealtimeArchitectureProof>,
    completion: crate::CommunicationCompletionPolicy,
    observations: RealtimeObservationRequirements,
}

impl RealtimeSelectionRequest {
    /// Creates one complete fail-closed selection request.
    pub fn new(
        source: RealtimeIdentity,
        execution: RealtimeIdentity,
        residency: LayerWeightResidency,
        state: CacheResidencyPolicy,
        architecture_proof: Option<RealtimeArchitectureProof>,
        completion: crate::CommunicationCompletionPolicy,
        observations: RealtimeObservationRequirements,
    ) -> Self {
        Self {
            source,
            execution,
            residency,
            state,
            architecture_proof,
            completion,
            observations,
        }
    }
}

/// One exact mutable-state component selected before native allocation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedRealtimeStateComponentRealization {
    mechanism: StateComponentMechanism,
    placement: StateComponentPlacement,
}

impl SelectedRealtimeStateComponentRealization {
    /// Returns the architecture-global state layer.
    pub const fn layer(&self) -> usize {
        self.mechanism.layer()
    }

    /// Returns the exact architecture-declared semantic component.
    pub const fn component(&self) -> &eredu_core::cache::StateComponentPolicy {
        self.mechanism.component()
    }

    /// Returns the exact placement resolved during selection.
    pub const fn placement(&self) -> StateComponentPlacement {
        self.placement
    }
}

/// Authoritative mutable-state realization selected before native allocation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedRealtimeStateRealization {
    layout: StateLayout,
    policy: CacheResidencyPolicy,
    components: Vec<SelectedRealtimeStateComponentRealization>,
    checkpoint: bool,
    rollback: bool,
    reset: bool,
    observation_retention: bool,
}

impl SelectedRealtimeStateRealization {
    /// Returns the exact architecture-owned mutable-state layout.
    pub const fn layout(&self) -> &StateLayout {
        &self.layout
    }

    /// Returns the exact selected state residency policy.
    pub const fn policy(&self) -> &CacheResidencyPolicy {
        &self.policy
    }

    /// Returns state components in architecture layer/component order.
    pub fn components(&self) -> &[SelectedRealtimeStateComponentRealization] {
        &self.components
    }

    /// Returns whether state checkpoints are guaranteed.
    pub const fn checkpoint(&self) -> bool {
        self.checkpoint
    }

    /// Returns whether checkpoint rollback is guaranteed.
    pub const fn rollback(&self) -> bool {
        self.rollback
    }

    /// Returns whether complete state reset is guaranteed.
    pub const fn reset(&self) -> bool {
        self.reset
    }

    /// Returns whether observed submissions retain every live state component.
    pub const fn observation_retention(&self) -> bool {
        self.observation_retention
    }
}

/// One immutable realtime realization selected before native construction.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedRealtimeRealization {
    requirements: RealtimeArchitectureRequirements,
    source: RealtimeIdentity,
    execution: RealtimeExecutionRequirements,
    residency: LayerWeightResidency,
    state: SelectedRealtimeStateRealization,
    topology: ParallelTopology,
    rank: usize,
    completion: crate::CommunicationCompletionPolicy,
    observations: RealtimeObservationRequirements,
}

impl SelectedRealtimeRealization {
    /// Returns the complete architecture requirements selected once.
    pub const fn requirements(&self) -> &RealtimeArchitectureRequirements {
        &self.requirements
    }

    /// Returns the exact source identity.
    pub const fn source(&self) -> &RealtimeIdentity {
        &self.source
    }

    /// Returns the exact selected temporal/depth execution graph.
    pub const fn execution_graph(&self) -> &ExecutionGraph {
        self.requirements.execution_graph()
    }

    /// Returns the exact selected temporal/depth execution-unit layout.
    pub const fn execution_units(&self) -> &ExecutionUnitLayout {
        self.requirements.execution_units()
    }

    /// Returns exact source parameter ownership and geometry.
    pub const fn source_parameters(&self) -> &ArchitectureParameterDescription {
        self.requirements.source_parameters()
    }

    /// Returns optional neural operators required by the selected architecture.
    pub const fn operators(&self) -> NeuralOperatorCapabilities {
        self.requirements.operators()
    }

    /// Returns the selected execution identity.
    pub const fn execution(&self) -> &RealtimeIdentity {
        self.execution.identity()
    }

    /// Returns the exact execution contract selected for materialization.
    pub const fn execution_requirements(&self) -> &RealtimeExecutionRequirements {
        &self.execution
    }

    /// Returns exact selected execution parameter ownership and geometry.
    pub const fn execution_parameters(&self) -> &ArchitectureParameterDescription {
        self.execution.parameters()
    }

    /// Returns the exact named lowering requirements retained by this selection.
    pub fn weight_lowerings(&self) -> &[RealtimeWeightLoweringRequirement] {
        self.execution.weight_lowerings()
    }

    /// Returns selected residency.
    pub const fn residency(&self) -> LayerWeightResidency {
        self.residency
    }

    /// Returns the exact selected mutable-state realization.
    pub const fn state(&self) -> &SelectedRealtimeStateRealization {
        &self.state
    }

    /// Returns the architecture-proven topology.
    pub const fn topology(&self) -> ParallelTopology {
        self.topology
    }

    /// Returns the architecture-proven global rank.
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the one selected bounded-wait and timeout disposition policy.
    pub const fn completion(&self) -> crate::CommunicationCompletionPolicy {
        self.completion
    }

    /// Returns selected observation requirements.
    pub const fn observations(&self) -> &RealtimeObservationRequirements {
        &self.observations
    }
}

/// One stable fail-closed realtime selection issue.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeSelectionIssue {
    /// No architecture proof was supplied.
    #[error("architecture topology and rank proof is missing")]
    MissingArchitectureProof,
    /// Proof names another architecture.
    #[error("architecture proof identity mismatch")]
    ArchitectureIdentityMismatch,
    /// Proof names another schedule.
    #[error("speech schedule proof identity mismatch")]
    SpeechScheduleIdentityMismatch,
    /// Proof names another state layout.
    #[error("state layout proof identity mismatch")]
    StateLayoutIdentityMismatch,
    /// Proof names another topology policy.
    #[error("topology policy proof identity mismatch")]
    TopologyPolicyIdentityMismatch,
    /// Request names another source artifact.
    #[error("source artifact identity mismatch")]
    SourceIdentityMismatch,
    /// Requested execution configuration was not admitted.
    #[error("execution identity is not architecture-admitted")]
    ExecutionIdentityNotAdmitted,
    /// A required optional neural operator is unavailable.
    #[error("required neural operator {operator} is unavailable")]
    MissingNeuralOperator {
        /// Stable generic operator name.
        operator: String,
    },
    /// A required named geometry-bearing weight lowering is unavailable.
    #[error("execution weight lowering {index} for target {target} is unavailable")]
    WeightLoweringUnavailable {
        /// Stable ordinal in the architecture execution contract.
        index: usize,
        /// Exact canonical executable target.
        target: RealtimeIdentity,
    },
    /// Requested residency was not admitted by the architecture.
    #[error("residency is not architecture-admitted")]
    ResidencyNotAdmitted,
    /// Backend cannot implement requested residency.
    #[error("residency mechanism is unavailable")]
    ResidencyUnavailable,
    /// Architecture did not admit the proven topology.
    #[error("topology is not architecture-admitted as replicated or pure tensor parallel")]
    TopologyNotAdmitted,
    /// Proven rank lies outside the topology.
    #[error("rank lies outside the proven topology")]
    RankOutOfRange,
    /// Backend cannot realize the requested tensor-parallel width.
    #[error("tensor-parallel width exceeds backend capability")]
    TensorParallelWidthUnavailable,
    /// A required generic mechanism is unavailable.
    #[error("required realtime mechanism {0:?} is unavailable")]
    MissingMechanism(RealtimeMechanism),
    /// Backend cannot safely apply the requested completion timeout disposition.
    #[error("requested completion timeout disposition is unavailable")]
    CompletionPolicyUnavailable,
    /// No unique compatible mechanism exists for one architecture state component.
    #[error("state component {component} at layer {layer} is unavailable")]
    StateComponentUnavailable {
        /// Architecture-global state layer.
        layer: usize,
        /// Stable component role.
        component: String,
    },
    /// State checkpointing is required for unpublished realtime branches.
    #[error("state checkpoint mechanism is unavailable")]
    MissingStateCheckpoint,
    /// State rollback is required for discarded realtime branches.
    #[error("state rollback mechanism is unavailable")]
    MissingStateRollback,
    /// State reset is required at architecture-declared frame seams.
    #[error("state reset mechanism is unavailable")]
    MissingStateReset,
    /// Observation must retain every referenced state component.
    #[error("state observation retention is unavailable")]
    MissingStateObservationRetention,
    /// Persistent model-state storage is unavailable.
    #[error("persistent session state is unavailable")]
    MissingPersistentState,
    /// Requested completed-output observation is unavailable.
    #[error("completed output observation is unavailable")]
    MissingOutputObservation,
    /// Requested activation inspection is unavailable.
    #[error("activation inspection is unavailable")]
    MissingActivationInspection,
    /// One exact requested activation observation is unavailable.
    #[error("requested activation observation {observation} is unavailable")]
    ObservationUnavailable {
        /// Exact architecture-owned observation identity.
        observation: RealtimeIdentity,
    },
}

/// Fail-closed diagnostic containing issues in stable validation order.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("realtime realization is unsupported: {message}", message = selection_message(.issues))]
pub struct RealtimeSelectionError {
    issues: Vec<RealtimeSelectionIssue>,
}

impl RealtimeSelectionError {
    /// Returns every mismatch in stable validation order.
    pub fn issues(&self) -> &[RealtimeSelectionIssue] {
        &self.issues
    }
}

fn selection_message(issues: &[RealtimeSelectionIssue]) -> String {
    issues
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

/// Selects the complete realtime realization without invoking native construction.
pub fn select_realtime_realization(
    requirements: &RealtimeArchitectureRequirements,
    request: &RealtimeSelectionRequest,
    capabilities: &RealtimeMechanismCapabilities,
) -> Result<SelectedRealtimeRealization, RealtimeSelectionError> {
    let mut issues = Vec::new();
    let proof = request.architecture_proof.as_ref();
    match proof {
        Some(proof) => {
            if proof.architecture != requirements.architecture {
                issues.push(RealtimeSelectionIssue::ArchitectureIdentityMismatch);
            }
            if proof.speech_schedule != requirements.speech_schedule_identity {
                issues.push(RealtimeSelectionIssue::SpeechScheduleIdentityMismatch);
            }
            if proof.state_layout != requirements.state_layout_identity {
                issues.push(RealtimeSelectionIssue::StateLayoutIdentityMismatch);
            }
            if proof.topology_policy != *requirements.topology.identity() {
                issues.push(RealtimeSelectionIssue::TopologyPolicyIdentityMismatch);
            }
        }
        None => issues.push(RealtimeSelectionIssue::MissingArchitectureProof),
    }
    if request.source != requirements.source {
        issues.push(RealtimeSelectionIssue::SourceIdentityMismatch);
    }
    let execution = requirements
        .executions
        .iter()
        .find(|execution| execution.identity == request.execution);
    if execution.is_none() {
        issues.push(RealtimeSelectionIssue::ExecutionIdentityNotAdmitted);
    }
    for operator in capabilities
        .operators
        .missing_capability_names(requirements.operators)
    {
        issues.push(RealtimeSelectionIssue::MissingNeuralOperator {
            operator: operator.to_owned(),
        });
    }
    if let Some(execution) = execution {
        for (index, required) in execution.weight_lowerings().iter().enumerate() {
            if !capabilities.weight_lowerings.iter().any(|available| {
                available.descriptor() == required.descriptor()
                    && available.kind() == required.kind()
            }) {
                issues.push(RealtimeSelectionIssue::WeightLoweringUnavailable {
                    index,
                    target: required.target().clone(),
                });
            }
        }
    }
    let execution_residency = request.residency.execution_residency();
    if !requirements.residencies.contains(&execution_residency) {
        issues.push(RealtimeSelectionIssue::ResidencyNotAdmitted);
    }
    if !capabilities.residencies.contains(&execution_residency) {
        issues.push(RealtimeSelectionIssue::ResidencyUnavailable);
    }
    if let Some(proof) = proof {
        if !requirements.topology.admits(proof.topology) {
            issues.push(RealtimeSelectionIssue::TopologyNotAdmitted);
        }
        if proof.rank >= proof.topology.world_size() {
            issues.push(RealtimeSelectionIssue::RankOutOfRange);
        }
        if proof.topology.tensor() > capabilities.maximum_tensor_parallel_size.get() {
            issues.push(RealtimeSelectionIssue::TensorParallelWidthUnavailable);
        }
    }
    for mechanism in requirements.mechanisms.mechanisms() {
        if !capabilities.supports(*mechanism) {
            issues.push(RealtimeSelectionIssue::MissingMechanism(*mechanism));
        }
    }
    if !capabilities.completion.supports(request.completion) {
        issues.push(RealtimeSelectionIssue::CompletionPolicyUnavailable);
    }
    let mut state_components = Vec::new();
    for layer in 0..requirements.state_layout.len() {
        for component in requirements
            .state_layout
            .components(layer)
            .expect("state layout exposes every validated layer")
        {
            let matches = capabilities
                .state
                .components()
                .iter()
                .filter(|mechanism| {
                    mechanism.layer() == layer && mechanism.component() == component
                })
                .cloned()
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [mechanism] => match mechanism.placement(&request.state) {
                    Some(placement)
                        if crate::replicated_text::placement_is_compatible(
                            component,
                            &request.state,
                            placement,
                        ) =>
                    {
                        state_components.push(SelectedRealtimeStateComponentRealization {
                            mechanism: mechanism.clone(),
                            placement,
                        });
                    }
                    _ => issues.push(RealtimeSelectionIssue::StateComponentUnavailable {
                        layer,
                        component: component.role().stable_name().to_owned(),
                    }),
                },
                _ => issues.push(RealtimeSelectionIssue::StateComponentUnavailable {
                    layer,
                    component: component.role().stable_name().to_owned(),
                }),
            }
        }
    }
    if !capabilities.state.checkpoint() {
        issues.push(RealtimeSelectionIssue::MissingStateCheckpoint);
    }
    if !capabilities.state.rollback() {
        issues.push(RealtimeSelectionIssue::MissingStateRollback);
    }
    if !capabilities.state.reset() {
        issues.push(RealtimeSelectionIssue::MissingStateReset);
    }
    if (request.observations.output() || request.observations.requires_activations())
        && !capabilities.state.observation_retention()
    {
        issues.push(RealtimeSelectionIssue::MissingStateObservationRetention);
    }
    if let Some(proof) = proof {
        if !proof.topology.is_replicated()
            && !capabilities.supports(RealtimeMechanism::Collectives)
            && !requirements
                .mechanisms
                .requires(RealtimeMechanism::Collectives)
        {
            issues.push(RealtimeSelectionIssue::MissingMechanism(
                RealtimeMechanism::Collectives,
            ));
        }
    }
    if !capabilities.session.persistent_cache() {
        issues.push(RealtimeSelectionIssue::MissingPersistentState);
    }
    if request.observations.output() && !capabilities.session.output_observation() {
        issues.push(RealtimeSelectionIssue::MissingOutputObservation);
    }
    if request.observations.requires_activations() && !capabilities.session.activation_inspection()
    {
        issues.push(RealtimeSelectionIssue::MissingActivationInspection);
    }
    for observation in request.observations.activations() {
        if !capabilities.supports_observation(observation) {
            issues.push(RealtimeSelectionIssue::ObservationUnavailable {
                observation: observation.clone(),
            });
        }
    }
    if !issues.is_empty() {
        return Err(RealtimeSelectionError { issues });
    }
    let proof = proof.expect("successful realtime selection has architecture proof");
    let state = SelectedRealtimeStateRealization {
        layout: requirements.state_layout.clone(),
        policy: request.state.clone(),
        components: state_components,
        checkpoint: capabilities.state.checkpoint(),
        rollback: capabilities.state.rollback(),
        reset: capabilities.state.reset(),
        observation_retention: capabilities.state.observation_retention(),
    };
    Ok(SelectedRealtimeRealization {
        requirements: requirements.clone(),
        source: request.source.clone(),
        execution: execution
            .expect("successful realtime selection has admitted execution")
            .clone(),
        residency: request.residency,
        state,
        topology: proof.topology,
        rank: proof.rank,
        completion: request.completion,
        observations: request.observations.clone(),
    })
}

/// Resources constructed only after realtime selection succeeds.
#[derive(Debug)]
pub struct ConstructedRealtimeResources<P, M, S, Q, G> {
    payload: P,
    modules: M,
    state: S,
    queue: Q,
    group: Option<G>,
}

impl<P, M, S, Q, G> ConstructedRealtimeResources<P, M, S, Q, G> {
    /// Consumes resources in construction order.
    pub fn into_parts(self) -> (P, M, S, Q, Option<G>) {
        (
            self.payload,
            self.modules,
            self.state,
            self.queue,
            self.group,
        )
    }
}

/// A selected realtime realization paired with native mechanism resources.
#[derive(Debug)]
pub struct PreparedRealtimeRealization<P, M, S, Q, G> {
    selected: SelectedRealtimeRealization,
    resources: ConstructedRealtimeResources<P, M, S, Q, G>,
}

impl<P, M, S, Q, G> PreparedRealtimeRealization<P, M, S, Q, G> {
    /// Returns the authoritative neutral selection.
    pub const fn selected(&self) -> &SelectedRealtimeRealization {
        &self.selected
    }

    /// Consumes the prepared realization into selection and resources.
    pub fn into_parts(
        self,
    ) -> (
        SelectedRealtimeRealization,
        ConstructedRealtimeResources<P, M, S, Q, G>,
    ) {
        (self.selected, self.resources)
    }
}

/// Failure during selection or one post-selection construction stage.
#[derive(Debug, thiserror::Error)]
pub enum RealtimePreparationError<E> {
    /// Neutral selection rejected before every construction callback.
    #[error(transparent)]
    Selection(#[from] RealtimeSelectionError),
    /// Checkpoint payload construction failed.
    #[error("realtime payload construction failed")]
    Payload(#[source] E),
    /// Architecture module construction failed.
    #[error("realtime module construction failed")]
    Modules(#[source] E),
    /// Mutable model-state construction failed.
    #[error("realtime state construction failed")]
    State(#[source] E),
    /// Execution queue construction failed.
    #[error("realtime queue construction failed")]
    Queue(#[source] E),
    /// Communication group construction failed.
    #[error("realtime communication group construction failed")]
    Group(#[source] E),
}

/// Selects once, then constructs family-blind resources in dependency order.
#[allow(clippy::too_many_arguments)]
pub fn select_and_prepare_realtime_realization<P, M, S, Q, G, E>(
    requirements: &RealtimeArchitectureRequirements,
    request: &RealtimeSelectionRequest,
    capabilities: &RealtimeMechanismCapabilities,
    payload: impl FnOnce(&SelectedRealtimeRealization) -> Result<P, E>,
    modules: impl FnOnce(&SelectedRealtimeRealization, &P) -> Result<M, E>,
    state: impl FnOnce(&SelectedRealtimeRealization, &M) -> Result<S, E>,
    queue: impl FnOnce(&SelectedRealtimeRealization) -> Result<Q, E>,
    group: impl FnOnce(&SelectedRealtimeRealization) -> Result<G, E>,
) -> Result<PreparedRealtimeRealization<P, M, S, Q, G>, RealtimePreparationError<E>> {
    let selected = select_realtime_realization(requirements, request, capabilities)?;
    let payload = payload(&selected).map_err(RealtimePreparationError::Payload)?;
    let modules = modules(&selected, &payload).map_err(RealtimePreparationError::Modules)?;
    let state = state(&selected, &modules).map_err(RealtimePreparationError::State)?;
    let queue = queue(&selected).map_err(RealtimePreparationError::Queue)?;
    let group = if selected.topology().is_replicated() {
        None
    } else {
        Some(group(&selected).map_err(RealtimePreparationError::Group)?)
    };
    Ok(PreparedRealtimeRealization {
        selected,
        resources: ConstructedRealtimeResources {
            payload,
            modules,
            state,
            queue,
            group,
        },
    })
}

/// Invalid neutral realtime contract.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeContractError {
    /// Stable identities must contain non-whitespace text.
    #[error("realtime identity must not be empty")]
    EmptyIdentity,
    /// A realtime architecture must require generic mechanisms.
    #[error("realtime mechanism requirements must not be empty")]
    EmptyMechanismRequirements,
    /// At least one execution identity must be admitted.
    #[error("realtime execution identities must not be empty")]
    EmptyExecutionIdentities,
    /// The retained execution graph and execution-unit layout must name the same groups.
    #[error("realtime execution graph and unit layout differ")]
    ExecutionGraphLayoutMismatch,
    /// Source parameters must be described against the exact retained graph and layout.
    #[error("realtime source parameter description differs from the retained graph or layout")]
    SourceParameterDescriptionMismatch,
    /// Execution parameters must be described against the exact retained graph and layout.
    #[error(
        "realtime execution parameter description for {execution} differs from the retained graph or layout"
    )]
    ExecutionParameterDescriptionMismatch {
        /// Admitted execution whose description is incoherent.
        execution: RealtimeIdentity,
    },
    /// Execution identities must be unique within one architecture contract.
    #[error("realtime execution identities must be unique")]
    DuplicateExecutionIdentity,
    /// Every admitted execution must describe its exact checkpoint lowerings.
    #[error("realtime execution weight lowerings must not be empty")]
    EmptyWeightLowerings,
    /// Physical component geometry must have positive nonempty extents.
    #[error("realtime weight component shape must contain only positive extents")]
    InvalidWeightComponentShape,
    /// Recipe owner and recipe identity must either both be present or both be absent.
    #[error("realtime weight component recipe owner and identity must appear together")]
    IncompleteWeightComponentRecipe,
    /// A recipe-backed physical component must retain at least one source occurrence.
    #[error("realtime recipe-backed weight component sources must not be empty")]
    EmptyRecipeComponentSources,
    /// A retained recipe or its exact output metadata differs from the component declaration.
    #[error("realtime weight component recipe does not match its sources and physical geometry")]
    WeightComponentRecipeMismatch,
    /// A transform-generated component cannot claim physical recipe sources.
    #[error("realtime generated weight component must not contain recipe sources")]
    GeneratedComponentHasSources,
    /// Every matrix lowering must contain its primary physical component.
    #[error("realtime weight lowering has no primary component")]
    MissingPrimaryWeightComponent,
    /// Every matrix lowering must contain exactly one primary physical component.
    #[error("realtime weight lowering has more than one primary component")]
    DuplicatePrimaryWeightComponent,
    /// The primary component target must equal the matrix lowering target.
    #[error("realtime primary weight component does not match its lowering target")]
    PrimaryWeightComponentTargetMismatch,
    /// Atomic component targets must be unique within one matrix lowering.
    #[error("realtime weight component targets must be unique within a lowering")]
    DuplicateWeightComponentTarget,
    /// Scale and affine-bias roles may each occur at most once.
    #[error("realtime weight lowering repeats component role {role:?}")]
    DuplicateWeightComponentRole {
        /// Repeated companion role.
        role: RealtimeWeightComponentRole,
    },
    /// One execution cannot bind the same canonical target more than once.
    #[error("realtime execution weight lowering targets must be unique")]
    DuplicateWeightLoweringTarget,
    /// Lowering targets must exactly equal primary execution parameters.
    #[error("realtime weight lowerings differ from execution parameters")]
    WeightLoweringParameterMismatch,
    /// One lowering's components must exactly equal its execution parameter family.
    #[error("realtime weight components differ from execution parameters for {target}")]
    WeightComponentParameterMismatch {
        /// Affected primary lowering target.
        target: RealtimeIdentity,
    },
    /// At least one residency must be admitted.
    #[error("realtime residencies must not be empty")]
    EmptyResidencies,
    /// Residency declarations must not be duplicated.
    #[error("realtime residencies must be unique")]
    DuplicateResidency,
    /// A topology policy must admit replicated or pure tensor-parallel execution.
    #[error("realtime topology policy must not be empty")]
    EmptyTopologyPolicy,
    /// Pure tensor-parallel sizes must contain more than one rank.
    #[error("pure tensor-parallel size {size} must be at least two")]
    InvalidPureTensorParallelSize {
        /// Invalid requested size.
        size: usize,
    },
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, convert::Infallible, num::NonZeroUsize, rc::Rc, time::Duration};

    use eredu_checkpoint::{LinearFormat, SourceTensorEncoding, StoredDtype};
    use eredu_core::{
        cache::LayerCachePolicy, AttentionPolicy, CompletionCancellationMode, LayerSchedule,
        RealtimeFrameConvention,
    };

    use super::*;
    fn identity(value: &str) -> RealtimeIdentity {
        RealtimeIdentity::new(value).unwrap()
    }

    fn execution_graph() -> ExecutionGraph {
        ExecutionGraph::chain(["temporal", "depth"]).unwrap()
    }

    fn execution_units(graph: &ExecutionGraph) -> ExecutionUnitLayout {
        ExecutionUnitLayout::new(graph, [2, 2]).unwrap()
    }

    fn parameter_description(
        graph: &ExecutionGraph,
        units: &ExecutionUnitLayout,
    ) -> ArchitectureParameterDescription {
        let group = crate::ParameterGroupSpec::new(
            "target",
            crate::ParameterRole::Replicated,
            [crate::ParameterMemberSpec::new(
                "target.weight",
                [8, 8],
                crate::MemberSharding::Replicated,
            )],
        )
        .unwrap();
        ArchitectureParameterDescription::new(
            graph,
            units,
            [group.clone()],
            [crate::OwnedParameterGroupSpec::new(
                crate::ParameterGroupOwner::static_role("target"),
                group,
            )],
        )
        .unwrap()
    }

    fn required_operators() -> NeuralOperatorCapabilities {
        NeuralOperatorCapabilities::EXP.union(NeuralOperatorCapabilities::SOFTMAX_AXIS)
    }

    fn state_layout() -> StateLayout {
        StateLayout::new(
            LayerSchedule::new(
                2,
                vec![
                    LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 8).unwrap(),
                    LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 8).unwrap(),
                ],
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn schedule() -> RealtimeSpeechConfig {
        RealtimeSpeechConfig::new(
            4,
            2,
            2,
            2,
            0,
            1,
            RealtimeFrameConvention::FeedbackAlignedHistory,
            vec![0, 1, 2, 1, 2],
        )
        .unwrap()
    }

    fn required_mechanisms() -> Vec<RealtimeMechanism> {
        vec![
            RealtimeMechanism::TensorOperations,
            RealtimeMechanism::NeuralOperations,
            RealtimeMechanism::ParameterMaterialization,
            RealtimeMechanism::ParameterStorage,
            RealtimeMechanism::StateStorage,
            RealtimeMechanism::CoordinateStorage,
            RealtimeMechanism::Sampling,
            RealtimeMechanism::Randomness,
            RealtimeMechanism::HostConversion,
            RealtimeMechanism::ExactCompletion,
            RealtimeMechanism::ResourceRetention,
            RealtimeMechanism::Transfer,
            RealtimeMechanism::Collectives,
        ]
    }

    fn completion_policy() -> crate::CommunicationCompletionPolicy {
        crate::CommunicationCompletionPolicy::new(
            Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap()
    }

    fn completion_capabilities() -> crate::CommunicationCompletionCapabilities {
        crate::CommunicationCompletionCapabilities::new([
            CompletionCancellationMode::QuarantineUntilComplete,
        ])
        .unwrap()
    }

    fn lowering(executable: LinearFormat) -> WeightLoweringDescriptor {
        WeightLoweringDescriptor::new(
            SourceTensorEncoding::Safetensors(StoredDtype::F32),
            executable,
            vec![8, 8],
            vec![8, 8],
            Some(1),
        )
        .unwrap()
    }

    fn lowering_requirement(
        target: &str,
        executable: LinearFormat,
        kind: WeightLoweringKind,
    ) -> RealtimeWeightLoweringRequirement {
        let primary = RealtimeWeightComponentRequirement::new(
            identity(target),
            Some(identity("canonical-owner")),
            Some(identity("recipe-v1")),
            Some(eredu_checkpoint::recipe::DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: ["source-a", "source-b"]
                    .into_iter()
                    .map(|key| {
                        eredu_checkpoint::recipe::DerivedWeightRecipe::source(
                            key,
                            eredu_checkpoint::store::TensorSelection::Full,
                        )
                    })
                    .collect(),
            }),
            Some(eredu_checkpoint::recipe::RecipeMetadata {
                shape: vec![8, 8],
                dtype: eredu_checkpoint::recipe::RecipeDtype::F32,
                byte_len: 256,
            }),
            [identity("source-a"), identity("source-b")],
            [8, 8],
            RealtimeWeightComponentRole::Primary,
        )
        .unwrap();
        RealtimeWeightLoweringRequirement::new(
            identity(target),
            [primary],
            lowering(executable),
            kind,
        )
        .unwrap()
    }

    fn execution(
        value: &str,
        parameters: ArchitectureParameterDescription,
    ) -> RealtimeExecutionRequirements {
        RealtimeExecutionRequirements::new(
            identity(value),
            parameters,
            vec![lowering_requirement(
                "target.weight",
                LinearFormat::Dense,
                WeightLoweringKind::Direct,
            )],
        )
        .unwrap()
    }

    fn requirements() -> RealtimeArchitectureRequirements {
        let graph = execution_graph();
        let units = execution_units(&graph);
        let parameters = parameter_description(&graph, &units);
        requirements_with_model_contract(
            graph,
            units,
            parameters.clone(),
            vec![
                execution("execution-native", parameters.clone()),
                execution("execution-transformed", parameters),
            ],
        )
        .unwrap()
    }

    fn requirements_with_model_contract(
        graph: ExecutionGraph,
        units: ExecutionUnitLayout,
        source_parameters: ArchitectureParameterDescription,
        executions: Vec<RealtimeExecutionRequirements>,
    ) -> Result<RealtimeArchitectureRequirements, RealtimeContractError> {
        RealtimeArchitectureRequirements::new(
            identity("architecture-7"),
            identity("artifact-4"),
            graph,
            units,
            source_parameters,
            executions,
            required_operators(),
            identity("schedule-3"),
            schedule(),
            identity("state-layout-8"),
            state_layout(),
            RealtimeMechanismRequirements::new(required_mechanisms()).unwrap(),
            RealtimeTopologyPolicy::new(identity("topology-plan-2"), true, [2]).unwrap(),
            [
                ExecutionResidency::FullyResident,
                ExecutionResidency::LayerwiseHost,
            ],
        )
    }

    fn proof(requirements: &RealtimeArchitectureRequirements) -> RealtimeArchitectureProof {
        RealtimeArchitectureProof::new(
            requirements.architecture().clone(),
            requirements.speech_schedule_identity().clone(),
            requirements.state_layout_identity().clone(),
            requirements.topology().identity().clone(),
            ParallelTopology::new(2, 1, 1, 1).unwrap(),
            1,
        )
    }

    fn request(requirements: &RealtimeArchitectureRequirements) -> RealtimeSelectionRequest {
        RealtimeSelectionRequest::new(
            requirements.source().clone(),
            identity("execution-transformed"),
            LayerWeightResidency::LayerwiseHost(crate::LayerwiseLoadOptions::default()),
            CacheResidencyPolicy::Device,
            Some(proof(requirements)),
            completion_policy(),
            RealtimeObservationRequirements::new(
                true,
                [identity("temporal.layer.0"), identity("depth.slice.0")],
            ),
        )
    }

    fn state_capabilities() -> StateMechanismCapabilities {
        StateMechanismCapabilities::new((0..state_layout().len()).flat_map(|layer| {
            state_layout()
                .components(layer)
                .unwrap()
                .iter()
                .cloned()
                .map(move |component| {
                    crate::StateComponentMechanism::new(
                        layer,
                        component,
                        Some(crate::StateComponentPlacement::Device),
                        None,
                    )
                })
                .collect::<Vec<_>>()
        }))
        .with_transactions(true, true)
        .with_reset(true)
        .with_observation_retention(true)
    }

    fn capabilities() -> RealtimeMechanismCapabilities {
        RealtimeMechanismCapabilities::new(
            required_operators(),
            required_mechanisms(),
            [
                ExecutionResidency::FullyResident,
                ExecutionResidency::LayerwiseHost,
            ],
            vec![WeightLoweringCapability::new(
                lowering(LinearFormat::Dense),
                WeightLoweringKind::Direct,
            )],
            state_capabilities(),
            NonZeroUsize::new(2).unwrap(),
            completion_capabilities(),
            SessionCapabilities::new(true, true, true),
        )
        .with_observation_identities([identity("temporal.layer.0"), identity("depth.slice.0")])
    }

    #[test]
    fn identities_and_requirement_sets_are_exact_and_nonempty() {
        assert_eq!(
            RealtimeIdentity::new(" \t"),
            Err(RealtimeContractError::EmptyIdentity)
        );
        let exact = RealtimeIdentity::new("  exact identity  ").unwrap();
        assert_eq!(exact.as_str(), "  exact identity  ");
        assert_eq!(
            RealtimeMechanismRequirements::new([]),
            Err(RealtimeContractError::EmptyMechanismRequirements)
        );
        assert_eq!(
            RealtimeTopologyPolicy::new(identity("none"), false, []),
            Err(RealtimeContractError::EmptyTopologyPolicy)
        );
        let duplicate = lowering_requirement(
            "duplicate.target",
            LinearFormat::Dense,
            WeightLoweringKind::Direct,
        );
        let graph = execution_graph();
        let units = execution_units(&graph);
        assert_eq!(
            RealtimeExecutionRequirements::new(
                identity("duplicate-execution"),
                parameter_description(&graph, &units),
                vec![duplicate.clone(), duplicate],
            ),
            Err(RealtimeContractError::DuplicateWeightLoweringTarget)
        );
    }

    #[test]
    fn model_construction_contract_requires_one_exact_graph_and_layout() {
        let graph = execution_graph();
        let units = execution_units(&graph);
        let parameters = parameter_description(&graph, &units);
        let other_graph = ExecutionGraph::chain(["other"]).unwrap();
        let other_units = ExecutionUnitLayout::new(&other_graph, [1]).unwrap();
        let other_parameters = parameter_description(&other_graph, &other_units);

        assert_eq!(
            requirements_with_model_contract(
                graph.clone(),
                other_units,
                parameters.clone(),
                vec![execution("execution", parameters.clone())],
            ),
            Err(RealtimeContractError::ExecutionGraphLayoutMismatch)
        );
        assert_eq!(
            requirements_with_model_contract(
                graph.clone(),
                units.clone(),
                other_parameters.clone(),
                vec![execution("execution", parameters.clone())],
            ),
            Err(RealtimeContractError::SourceParameterDescriptionMismatch)
        );
        assert_eq!(
            requirements_with_model_contract(
                graph,
                units,
                parameters.clone(),
                vec![execution("execution", other_parameters)],
            ),
            Err(
                RealtimeContractError::ExecutionParameterDescriptionMismatch {
                    execution: identity("execution"),
                }
            )
        );
    }

    #[test]
    fn weight_components_enforce_recipe_geometry_and_atomic_family_invariants() {
        let component = |target: &str,
                         owner: Option<&str>,
                         recipe: Option<&str>,
                         sources: Vec<&str>,
                         shape: Vec<usize>,
                         role| {
            let retained_recipe = (owner.is_some() && recipe.is_some()).then(|| {
                let mut inputs = sources
                    .iter()
                    .map(|key| {
                        eredu_checkpoint::recipe::DerivedWeightRecipe::source(
                            *key,
                            eredu_checkpoint::store::TensorSelection::Full,
                        )
                    })
                    .collect::<Vec<_>>();
                if inputs.len() == 1 {
                    inputs.pop().expect("one source recipe")
                } else {
                    eredu_checkpoint::recipe::DerivedWeightRecipe::Concatenate { axis: 0, inputs }
                }
            });
            let output = retained_recipe.as_ref().map(|_| {
                let elements = shape.iter().copied().product::<usize>();
                eredu_checkpoint::recipe::RecipeMetadata {
                    shape: shape.clone(),
                    dtype: eredu_checkpoint::recipe::RecipeDtype::F32,
                    byte_len: u64::try_from(elements.saturating_mul(4)).unwrap(),
                }
            });
            RealtimeWeightComponentRequirement::new(
                identity(target),
                owner.map(identity),
                recipe.map(identity),
                retained_recipe,
                output,
                sources.into_iter().map(identity),
                shape,
                role,
            )
        };
        assert_eq!(
            component(
                "weight",
                Some("owner"),
                Some("recipe"),
                vec!["source"],
                vec![],
                RealtimeWeightComponentRole::Primary,
            ),
            Err(RealtimeContractError::InvalidWeightComponentShape)
        );
        assert_eq!(
            component(
                "weight",
                Some("owner"),
                Some("recipe"),
                vec!["source"],
                vec![8, 0],
                RealtimeWeightComponentRole::Primary,
            ),
            Err(RealtimeContractError::InvalidWeightComponentShape)
        );
        assert_eq!(
            component(
                "weight",
                Some("owner"),
                None,
                vec!["source"],
                vec![8, 8],
                RealtimeWeightComponentRole::Primary,
            ),
            Err(RealtimeContractError::IncompleteWeightComponentRecipe)
        );
        assert_eq!(
            component(
                "weight",
                None,
                Some("recipe"),
                vec!["source"],
                vec![8, 8],
                RealtimeWeightComponentRole::Primary,
            ),
            Err(RealtimeContractError::IncompleteWeightComponentRecipe)
        );
        assert_eq!(
            component(
                "weight",
                Some("owner"),
                Some("recipe"),
                vec![],
                vec![8, 8],
                RealtimeWeightComponentRole::Primary,
            ),
            Err(RealtimeContractError::EmptyRecipeComponentSources)
        );
        assert_eq!(
            component(
                "weight",
                None,
                None,
                vec!["source"],
                vec![8, 8],
                RealtimeWeightComponentRole::Primary,
            ),
            Err(RealtimeContractError::GeneratedComponentHasSources)
        );

        let primary = component(
            "weight",
            Some("owner"),
            Some("recipe"),
            vec!["source", "source"],
            vec![8, 8],
            RealtimeWeightComponentRole::Primary,
        )
        .unwrap();
        let generated_scale = component(
            "scales",
            None,
            None,
            vec![],
            vec![8, 1],
            RealtimeWeightComponentRole::Scale,
        )
        .unwrap();
        let generated_bias = component(
            "biases",
            None,
            None,
            vec![],
            vec![8, 1],
            RealtimeWeightComponentRole::AffineBias,
        )
        .unwrap();
        let packed = RealtimeWeightLoweringRequirement::new(
            identity("weight"),
            [
                primary.clone(),
                generated_scale.clone(),
                generated_bias.clone(),
            ],
            lowering(LinearFormat::Dense),
            WeightLoweringKind::Transform,
        )
        .unwrap();
        assert_eq!(packed.primary(), &primary);
        assert_eq!(packed.scale(), Some(&generated_scale));
        assert_eq!(packed.affine_bias(), Some(&generated_bias));
        assert_eq!(
            packed.components(),
            &[
                primary.clone(),
                generated_scale.clone(),
                generated_bias.clone(),
            ]
        );
        assert_eq!(
            packed.primary().source_occurrences(),
            &[identity("source"), identity("source")]
        );
        assert!(!packed.scale().unwrap().is_recipe_backed());
        assert_eq!(packed.scale().unwrap().recipe_owner(), None);
        assert_eq!(packed.scale().unwrap().recipe_identity(), None);
        assert!(packed.scale().unwrap().source_occurrences().is_empty());
        assert_eq!(packed.scale().unwrap().physical_shape(), &[8, 1]);

        assert_eq!(
            RealtimeWeightLoweringRequirement::new(
                identity("weight"),
                [generated_scale.clone()],
                lowering(LinearFormat::Dense),
                WeightLoweringKind::Transform,
            ),
            Err(RealtimeContractError::MissingPrimaryWeightComponent)
        );
        let other_primary = component(
            "other",
            Some("owner"),
            Some("recipe"),
            vec!["source"],
            vec![8, 8],
            RealtimeWeightComponentRole::Primary,
        )
        .unwrap();
        assert_eq!(
            RealtimeWeightLoweringRequirement::new(
                identity("weight"),
                [other_primary.clone()],
                lowering(LinearFormat::Dense),
                WeightLoweringKind::Direct,
            ),
            Err(RealtimeContractError::PrimaryWeightComponentTargetMismatch)
        );
        assert_eq!(
            RealtimeWeightLoweringRequirement::new(
                identity("weight"),
                [primary.clone(), other_primary],
                lowering(LinearFormat::Dense),
                WeightLoweringKind::Direct,
            ),
            Err(RealtimeContractError::DuplicatePrimaryWeightComponent)
        );
        let second_scale = component(
            "other.scales",
            None,
            None,
            vec![],
            vec![8, 1],
            RealtimeWeightComponentRole::Scale,
        )
        .unwrap();
        assert_eq!(
            RealtimeWeightLoweringRequirement::new(
                identity("weight"),
                [primary.clone(), generated_scale.clone(), second_scale],
                lowering(LinearFormat::Dense),
                WeightLoweringKind::Transform,
            ),
            Err(RealtimeContractError::DuplicateWeightComponentRole {
                role: RealtimeWeightComponentRole::Scale,
            })
        );
        let second_bias = component(
            "other.biases",
            None,
            None,
            vec![],
            vec![8, 1],
            RealtimeWeightComponentRole::AffineBias,
        )
        .unwrap();
        assert_eq!(
            RealtimeWeightLoweringRequirement::new(
                identity("weight"),
                [primary.clone(), generated_bias, second_bias],
                lowering(LinearFormat::Dense),
                WeightLoweringKind::Transform,
            ),
            Err(RealtimeContractError::DuplicateWeightComponentRole {
                role: RealtimeWeightComponentRole::AffineBias,
            })
        );
        let duplicate_target = component(
            "weight",
            None,
            None,
            vec![],
            vec![8, 1],
            RealtimeWeightComponentRole::Scale,
        )
        .unwrap();
        assert_eq!(
            RealtimeWeightLoweringRequirement::new(
                identity("weight"),
                [primary, duplicate_target],
                lowering(LinearFormat::Dense),
                WeightLoweringKind::Transform,
            ),
            Err(RealtimeContractError::DuplicateWeightComponentTarget)
        );
    }

    #[test]
    fn exact_selection_closes_architecture_and_mechanism_choices_once() {
        let requirements = requirements();
        let selected =
            select_realtime_realization(&requirements, &request(&requirements), &capabilities())
                .unwrap();

        assert_eq!(selected.requirements(), &requirements);
        assert_eq!(selected.source().as_str(), "artifact-4");
        assert_eq!(selected.execution().as_str(), "execution-transformed");
        assert_eq!(selected.execution_graph(), requirements.execution_graph());
        assert_eq!(selected.execution_units(), requirements.execution_units());
        assert_eq!(
            selected.source_parameters(),
            requirements.source_parameters()
        );
        assert_eq!(selected.operators(), required_operators());
        assert_eq!(capabilities().operators(), required_operators());
        assert_eq!(
            selected.execution_parameters(),
            requirements.executions()[1].parameters()
        );
        assert_eq!(
            selected.residency(),
            LayerWeightResidency::LayerwiseHost(crate::LayerwiseLoadOptions::default())
        );
        assert_eq!(
            selected.topology(),
            ParallelTopology::new(2, 1, 1, 1).unwrap()
        );
        assert_eq!(selected.rank(), 1);
        assert_eq!(selected.completion(), completion_policy());
        assert_eq!(
            selected.requirements().speech_schedule(),
            requirements.speech_schedule()
        );
        assert_eq!(
            selected.requirements().state_layout(),
            requirements.state_layout()
        );
        assert!(selected.observations().output());
        assert_eq!(
            selected
                .observations()
                .activations()
                .iter()
                .map(RealtimeIdentity::as_str)
                .collect::<Vec<_>>(),
            vec!["depth.slice.0", "temporal.layer.0"]
        );
        let selected_lowering = &selected.execution_requirements().weight_lowerings()[0];
        assert_eq!(selected_lowering.target().as_str(), "target.weight");
        let primary = selected_lowering.primary();
        assert_eq!(primary.target().as_str(), "target.weight");
        assert_eq!(primary.recipe_owner().unwrap().as_str(), "canonical-owner");
        assert_eq!(primary.recipe_identity().unwrap().as_str(), "recipe-v1");
        assert_eq!(
            primary
                .source_occurrences()
                .iter()
                .map(RealtimeIdentity::as_str)
                .collect::<Vec<_>>(),
            vec!["source-a", "source-b"]
        );
        assert_eq!(primary.physical_shape(), &[8, 8]);
        assert_eq!(primary.role(), RealtimeWeightComponentRole::Primary);
        assert_eq!(
            selected_lowering.components(),
            std::slice::from_ref(primary)
        );
        assert_eq!(selected_lowering.kind(), WeightLoweringKind::Direct);
        assert_eq!(selected_lowering.scale(), None);
        assert_eq!(selected_lowering.affine_bias(), None);

        let selected_state = selected.state();
        assert_eq!(selected_state.layout(), requirements.state_layout());
        assert_eq!(selected_state.policy(), &CacheResidencyPolicy::Device);
        let expected_components = (0..requirements.state_layout().len()).flat_map(|layer| {
            requirements
                .state_layout()
                .components(layer)
                .unwrap()
                .iter()
                .map(move |component| (layer, component))
        });
        let expected_count = expected_components.clone().count();
        assert_eq!(selected_state.components().len(), expected_count);
        for (selected_component, (layer, component)) in
            selected_state.components().iter().zip(expected_components)
        {
            assert_eq!(selected_component.layer(), layer);
            assert_eq!(selected_component.component(), component);
            assert_eq!(
                selected_component.placement(),
                StateComponentPlacement::Device
            );
        }
        assert!(selected_state.checkpoint());
        assert!(selected_state.rollback());
        assert!(selected_state.reset());
        assert!(selected_state.observation_retention());
    }

    #[test]
    fn selection_reports_stable_fail_closed_diagnostics() {
        let requirements = requirements();
        let request = RealtimeSelectionRequest::new(
            identity("wrong-artifact"),
            identity("unknown-execution"),
            LayerWeightResidency::DenseDiskStream(crate::DenseDiskStreamLoadOptions::default()),
            CacheResidencyPolicy::Device,
            None,
            completion_policy(),
            RealtimeObservationRequirements::new(true, [identity("temporal.layer.0")]),
        );
        let capabilities = RealtimeMechanismCapabilities::new(
            NeuralOperatorCapabilities::NONE,
            [RealtimeMechanism::NeuralOperations],
            [ExecutionResidency::FullyResident],
            Vec::new(),
            StateMechanismCapabilities::new([]),
            NonZeroUsize::new(1).unwrap(),
            crate::CommunicationCompletionCapabilities::new([
                CompletionCancellationMode::NativeCancel,
            ])
            .unwrap(),
            SessionCapabilities::default(),
        );
        let error =
            select_realtime_realization(&requirements, &request, &capabilities).unwrap_err();

        assert_eq!(
            error.issues(),
            &[
                RealtimeSelectionIssue::MissingArchitectureProof,
                RealtimeSelectionIssue::SourceIdentityMismatch,
                RealtimeSelectionIssue::ExecutionIdentityNotAdmitted,
                RealtimeSelectionIssue::MissingNeuralOperator {
                    operator: "exp".into(),
                },
                RealtimeSelectionIssue::MissingNeuralOperator {
                    operator: "softmax_axis".into(),
                },
                RealtimeSelectionIssue::ResidencyNotAdmitted,
                RealtimeSelectionIssue::ResidencyUnavailable,
                RealtimeSelectionIssue::MissingMechanism(RealtimeMechanism::TensorOperations),
                RealtimeSelectionIssue::MissingMechanism(
                    RealtimeMechanism::ParameterMaterialization,
                ),
                RealtimeSelectionIssue::MissingMechanism(RealtimeMechanism::ParameterStorage),
                RealtimeSelectionIssue::MissingMechanism(RealtimeMechanism::StateStorage),
                RealtimeSelectionIssue::MissingMechanism(RealtimeMechanism::CoordinateStorage),
                RealtimeSelectionIssue::MissingMechanism(RealtimeMechanism::Sampling),
                RealtimeSelectionIssue::MissingMechanism(RealtimeMechanism::Randomness),
                RealtimeSelectionIssue::MissingMechanism(RealtimeMechanism::HostConversion),
                RealtimeSelectionIssue::MissingMechanism(RealtimeMechanism::ExactCompletion),
                RealtimeSelectionIssue::MissingMechanism(RealtimeMechanism::ResourceRetention),
                RealtimeSelectionIssue::MissingMechanism(RealtimeMechanism::Transfer),
                RealtimeSelectionIssue::MissingMechanism(RealtimeMechanism::Collectives),
                RealtimeSelectionIssue::CompletionPolicyUnavailable,
                RealtimeSelectionIssue::StateComponentUnavailable {
                    layer: 0,
                    component: "attention.keys".into(),
                },
                RealtimeSelectionIssue::StateComponentUnavailable {
                    layer: 0,
                    component: "attention.values".into(),
                },
                RealtimeSelectionIssue::StateComponentUnavailable {
                    layer: 1,
                    component: "attention.keys".into(),
                },
                RealtimeSelectionIssue::StateComponentUnavailable {
                    layer: 1,
                    component: "attention.values".into(),
                },
                RealtimeSelectionIssue::MissingStateCheckpoint,
                RealtimeSelectionIssue::MissingStateRollback,
                RealtimeSelectionIssue::MissingStateReset,
                RealtimeSelectionIssue::MissingStateObservationRetention,
                RealtimeSelectionIssue::MissingPersistentState,
                RealtimeSelectionIssue::MissingOutputObservation,
                RealtimeSelectionIssue::MissingActivationInspection,
                RealtimeSelectionIssue::ObservationUnavailable {
                    observation: identity("temporal.layer.0"),
                },
            ]
        );
        assert_eq!(
            error.to_string(),
            "realtime realization is unsupported: architecture topology and rank proof is missing; source artifact identity mismatch; execution identity is not architecture-admitted; required neural operator exp is unavailable; required neural operator softmax_axis is unavailable; residency is not architecture-admitted; residency mechanism is unavailable; required realtime mechanism TensorOperations is unavailable; required realtime mechanism ParameterMaterialization is unavailable; required realtime mechanism ParameterStorage is unavailable; required realtime mechanism StateStorage is unavailable; required realtime mechanism CoordinateStorage is unavailable; required realtime mechanism Sampling is unavailable; required realtime mechanism Randomness is unavailable; required realtime mechanism HostConversion is unavailable; required realtime mechanism ExactCompletion is unavailable; required realtime mechanism ResourceRetention is unavailable; required realtime mechanism Transfer is unavailable; required realtime mechanism Collectives is unavailable; requested completion timeout disposition is unavailable; state component attention.keys at layer 0 is unavailable; state component attention.values at layer 0 is unavailable; state component attention.keys at layer 1 is unavailable; state component attention.values at layer 1 is unavailable; state checkpoint mechanism is unavailable; state rollback mechanism is unavailable; state reset mechanism is unavailable; state observation retention is unavailable; persistent session state is unavailable; completed output observation is unavailable; activation inspection is unavailable; requested activation observation temporal.layer.0 is unavailable"
        );
    }

    #[test]
    fn exact_weight_lowering_must_be_implemented_before_construction() {
        let requirements = requirements();
        let capabilities = RealtimeMechanismCapabilities::new(
            required_operators(),
            required_mechanisms(),
            [
                ExecutionResidency::FullyResident,
                ExecutionResidency::LayerwiseHost,
            ],
            vec![WeightLoweringCapability::new(
                lowering(LinearFormat::Dense),
                WeightLoweringKind::Transform,
            )],
            state_capabilities(),
            NonZeroUsize::new(2).unwrap(),
            completion_capabilities(),
            SessionCapabilities::new(true, true, true),
        )
        .with_observation_identities([identity("temporal.layer.0"), identity("depth.slice.0")]);
        let error =
            select_realtime_realization(&requirements, &request(&requirements), &capabilities)
                .unwrap_err();

        assert_eq!(
            error.issues(),
            &[RealtimeSelectionIssue::WeightLoweringUnavailable {
                index: 0,
                target: identity("target.weight"),
            }]
        );
    }

    #[test]
    fn topology_and_rank_are_fail_closed() {
        let requirements = requirements();
        let invalid_proof = RealtimeArchitectureProof::new(
            requirements.architecture().clone(),
            requirements.speech_schedule_identity().clone(),
            requirements.state_layout_identity().clone(),
            requirements.topology().identity().clone(),
            ParallelTopology::new(2, 2, 1, 1).unwrap(),
            4,
        );
        let request = RealtimeSelectionRequest::new(
            requirements.source().clone(),
            identity("execution-native"),
            LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
            Some(invalid_proof),
            completion_policy(),
            RealtimeObservationRequirements::default(),
        );
        let error =
            select_realtime_realization(&requirements, &request, &capabilities()).unwrap_err();
        assert_eq!(
            error.issues(),
            &[
                RealtimeSelectionIssue::TopologyNotAdmitted,
                RealtimeSelectionIssue::RankOutOfRange,
            ]
        );
    }

    #[test]
    fn selection_failure_prevents_every_construction_callback() {
        let requirements = requirements();
        let invalid_request = RealtimeSelectionRequest::new(
            identity("wrong-artifact"),
            identity("execution-native"),
            LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
            Some(proof(&requirements)),
            completion_policy(),
            RealtimeObservationRequirements::default(),
        );
        let calls = Rc::new(Cell::new(0));
        let callback = || {
            let calls = Rc::clone(&calls);
            move |_: &SelectedRealtimeRealization| {
                calls.set(calls.get() + 1);
                Ok::<_, Infallible>(())
            }
        };
        let module_calls = Rc::clone(&calls);
        let state_calls = Rc::clone(&calls);
        let error = select_and_prepare_realtime_realization(
            &requirements,
            &invalid_request,
            &capabilities(),
            callback(),
            move |_, _| {
                module_calls.set(module_calls.get() + 1);
                Ok::<_, Infallible>(())
            },
            move |_, _| {
                state_calls.set(state_calls.get() + 1);
                Ok::<_, Infallible>(())
            },
            callback(),
            callback(),
        )
        .unwrap_err();

        assert!(matches!(error, RealtimePreparationError::Selection(_)));
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn successful_gate_constructs_in_dependency_order() {
        let requirements = requirements();
        let calls = Rc::new(Cell::new(0));
        let prepared = select_and_prepare_realtime_realization(
            &requirements,
            &request(&requirements),
            &capabilities(),
            {
                let calls = Rc::clone(&calls);
                move |_| {
                    assert_eq!(calls.replace(1), 0);
                    Ok::<_, Infallible>("payload")
                }
            },
            {
                let calls = Rc::clone(&calls);
                move |_, payload| {
                    assert_eq!((*payload, calls.replace(2)), ("payload", 1));
                    Ok::<_, Infallible>("modules")
                }
            },
            {
                let calls = Rc::clone(&calls);
                move |_, modules| {
                    assert_eq!((*modules, calls.replace(3)), ("modules", 2));
                    Ok::<_, Infallible>("state")
                }
            },
            {
                let calls = Rc::clone(&calls);
                move |_| {
                    assert_eq!(calls.replace(4), 3);
                    Ok::<_, Infallible>("queue")
                }
            },
            {
                let calls = Rc::clone(&calls);
                move |selected| {
                    assert_eq!(selected.topology().tensor(), 2);
                    assert_eq!(calls.replace(5), 4);
                    Ok::<_, Infallible>("group")
                }
            },
        )
        .unwrap();

        assert_eq!(calls.get(), 5);
        let (_, resources) = prepared.into_parts();
        assert_eq!(
            resources.into_parts(),
            ("payload", "modules", "state", "queue", Some("group"))
        );
    }
}
