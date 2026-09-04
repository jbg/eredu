//! Selection contracts for replicated text architectures.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use eredu_checkpoint::{LinearFormat, SourceTensorEncoding, StoredDtype};
use eredu_core::{
    cache::StateComponentPolicy, ParallelTopology, QuantizationRequest, SessionCapabilities,
};
use eredu_nn::{NeuralBackend, NeuralOperatorCapabilities};

use crate::{
    ArchitectureGroupTransport, ArchitectureParameterDescription, ArchitecturePartition,
    CacheResidencyPolicy, ExecutionGraph, ExecutionUnitLayout, LayerWeightResidency,
    LayeredArchitecture, ParameterGroupOwner, ParameterGroupSpec, RuntimeState, StateLayout,
};

/// Statically dispatched text-input seam for a layered decoder.
///
/// Routed, composite, partitioned, prediction, and realtime execution use
/// separate extension contracts rather than adding requirements here.
pub trait ReplicatedTextArchitecture<B, S>: LayeredArchitecture<B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Forms the architecture-owned borrowed input for one text pass.
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a>;

    /// Declares how a causal-text session projects a complete architecture output.
    fn text_output_selection(&self) -> ReplicatedTextOutputSelection {
        ReplicatedTextOutputSelection::LastSequencePosition
    }
}

/// Architecture-declared projection from complete logits to one causal-text output.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum ReplicatedTextOutputSelection {
    /// Selects the final position on the architecture's sequence axis.
    LastSequencePosition,
}

impl ReplicatedTextOutputSelection {
    /// Returns the mechanical sequence-axis index requested from a backend tensor.
    pub const fn sequence_index(self) -> i32 {
        match self {
            Self::LastSequencePosition => -1,
        }
    }
}

/// Backend implementation route for one source-to-executable weight lowering.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum WeightLoweringKind {
    /// The admitted source encoding is retained by the executable operator.
    Direct,
    /// An architecture-owned recipe derives the executable tensor from the admitted source.
    Derived,
    /// Payload materialization performs an admitted transformation.
    Transform,
    /// An architecture recipe derives a tensor that payload materialization then transforms.
    DerivedTransform,
}

/// One exact weight lowering implemented by a backend.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightLoweringCapability {
    /// Exact neutral lowering request implemented by this capability.
    descriptor: WeightLoweringDescriptor,
    /// Whether the lowering is direct or transforming.
    kind: WeightLoweringKind,
}

impl WeightLoweringCapability {
    /// Creates one exact backend lowering mechanism.
    pub fn new(descriptor: WeightLoweringDescriptor, kind: WeightLoweringKind) -> Self {
        Self { descriptor, kind }
    }

    /// Returns the admitted source encoding.
    pub const fn source(&self) -> &SourceTensorEncoding {
        self.descriptor.source()
    }

    /// Returns the executable format produced by this mechanism.
    pub const fn executable(&self) -> LinearFormat {
        self.descriptor.executable()
    }

    /// Returns whether materialization retains or transforms the source.
    pub const fn kind(&self) -> WeightLoweringKind {
        self.kind
    }

    /// Returns the exact geometry-bearing lowering request.
    pub const fn descriptor(&self) -> &WeightLoweringDescriptor {
        &self.descriptor
    }
}

/// Exact source-to-executable lowering query presented to a backend.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightLoweringDescriptor {
    source: SourceTensorEncoding,
    executable: LinearFormat,
    physical_shape: Vec<usize>,
    logical_shape: Vec<usize>,
    packed_axis: Option<usize>,
}

impl WeightLoweringDescriptor {
    /// Creates a geometry-bearing lowering query.
    pub fn new(
        source: SourceTensorEncoding,
        executable: LinearFormat,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
        packed_axis: Option<usize>,
    ) -> Result<Self, ReplicatedTextContractError> {
        if physical_shape.contains(&0)
            || logical_shape.contains(&0)
            || physical_shape.len() != logical_shape.len()
        {
            return Err(ReplicatedTextContractError::invalid(
                "weight lowering requires positive extents and equal physical and logical ranks",
            ));
        }
        if packed_axis.is_some_and(|axis| axis >= logical_shape.len()) {
            return Err(ReplicatedTextContractError::invalid(
                "weight lowering packed axis is outside the logical shape",
            ));
        }
        Ok(Self {
            source,
            executable,
            physical_shape,
            logical_shape,
            packed_axis,
        })
    }

    /// Returns the admitted source encoding.
    pub const fn source(&self) -> &SourceTensorEncoding {
        &self.source
    }

    /// Returns the selected executable format.
    pub const fn executable(&self) -> LinearFormat {
        self.executable
    }

    /// Returns the admitted physical source shape.
    pub fn physical_shape(&self) -> &[usize] {
        &self.physical_shape
    }

    /// Returns the architecture-declared logical shape.
    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    /// Returns the executable packing axis, when the parameter is packable.
    pub const fn packed_axis(&self) -> Option<usize> {
        self.packed_axis
    }

    /// Returns the exact extent along the packing axis.
    pub fn packed_extent(&self) -> Option<usize> {
        self.packed_axis.map(|axis| self.logical_shape[axis])
    }
}

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
#[non_exhaustive]
pub enum StateComponentPlacement {
    /// The mutable component remains on the execution device.
    Device,
    /// The append-only component is managed by bounded paged storage.
    Paged,
}

/// Exact state component and placements implemented by a backend mechanism.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StateComponentMechanism {
    layer: usize,
    component: StateComponentPolicy,
    device_placement: Option<StateComponentPlacement>,
    paged_placement: Option<StateComponentPlacement>,
}

impl StateComponentMechanism {
    /// Describes support for one exact architecture-declared component.
    pub fn new(
        layer: usize,
        component: StateComponentPolicy,
        device_placement: Option<StateComponentPlacement>,
        paged_placement: Option<StateComponentPlacement>,
    ) -> Self {
        Self {
            layer,
            component,
            device_placement,
            paged_placement,
        }
    }

    /// Returns the architecture-global state layer.
    pub const fn layer(&self) -> usize {
        self.layer
    }

    /// Returns the exact semantic component contract.
    pub const fn component(&self) -> &StateComponentPolicy {
        &self.component
    }

    /// Returns the placement used for a requested state policy.
    pub const fn placement(
        &self,
        policy: &CacheResidencyPolicy,
    ) -> Option<StateComponentPlacement> {
        match policy {
            CacheResidencyPolicy::Device => self.device_placement,
            CacheResidencyPolicy::Paged(_) => self.paged_placement,
        }
    }
}

/// Exact, family-neutral mutable-state mechanisms reported by a backend.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StateMechanismCapabilities {
    components: Vec<StateComponentMechanism>,
    checkpoint: bool,
    rollback: bool,
    reset: bool,
    prompt_cache: bool,
    observation_retention: bool,
}

impl StateMechanismCapabilities {
    /// Creates a fail-closed report for exact architecture-declared components.
    pub fn new(components: impl IntoIterator<Item = StateComponentMechanism>) -> Self {
        Self {
            components: components.into_iter().collect(),
            checkpoint: false,
            rollback: false,
            reset: false,
            prompt_cache: false,
            observation_retention: false,
        }
    }

    /// Declares transactional checkpoint and rollback facilities.
    pub const fn with_transactions(mut self, checkpoint: bool, rollback: bool) -> Self {
        self.checkpoint = checkpoint;
        self.rollback = rollback;
        self
    }

    /// Declares complete state reset support.
    pub const fn with_reset(mut self, supported: bool) -> Self {
        self.reset = supported;
        self
    }

    /// Declares prompt-cache persistence and restoration support.
    pub const fn with_prompt_cache(mut self, supported: bool) -> Self {
        self.prompt_cache = supported;
        self
    }

    /// Declares that observed submissions retain every live component.
    pub const fn with_observation_retention(mut self, supported: bool) -> Self {
        self.observation_retention = supported;
        self
    }

    /// Returns exact supported component mechanisms.
    pub fn components(&self) -> &[StateComponentMechanism] {
        &self.components
    }

    /// Returns whether state checkpoints are implemented.
    pub const fn checkpoint(&self) -> bool {
        self.checkpoint
    }

    /// Returns whether checkpoint rollback is implemented.
    pub const fn rollback(&self) -> bool {
        self.rollback
    }

    /// Returns whether complete reset is implemented.
    pub const fn reset(&self) -> bool {
        self.reset
    }

    /// Returns whether prompt-cache persistence is implemented.
    pub const fn prompt_cache(&self) -> bool {
        self.prompt_cache
    }

    /// Returns whether observation retains every live component.
    pub const fn observation_retention(&self) -> bool {
        self.observation_retention
    }
}

/// Architecture-valid transform target for one linear parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterTransformTarget {
    /// Requested load-time transform.
    request: QuantizationRequest,
    /// Executable format produced for this parameter.
    executable: LinearFormat,
    /// Exact geometry that the backend lowering must accept.
    descriptor: WeightLoweringDescriptor,
}

impl ParameterTransformTarget {
    /// Creates one architecture-admitted load-time transform target.
    fn new(
        request: QuantizationRequest,
        executable: LinearFormat,
        descriptor: WeightLoweringDescriptor,
    ) -> Self {
        Self {
            request,
            executable,
            descriptor,
        }
    }

    /// Returns the caller request selecting this transform.
    pub const fn request(&self) -> QuantizationRequest {
        self.request
    }

    /// Returns the architecture-admitted executable format.
    pub const fn executable(&self) -> LinearFormat {
        self.executable
    }

    /// Returns the exact neutral lowering query.
    pub const fn descriptor(&self) -> &WeightLoweringDescriptor {
        &self.descriptor
    }
}

/// Architecture declaration of whether and how a parameter may be transformed.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum ParameterTransformConstraint {
    /// This parameter is not an executable affine projection weight.
    None,
    /// The declared axis is the input/packing axis of a linear parameter.
    Linear {
        /// Axis whose extent is grouped or blocked by executable packing.
        packed_axis: usize,
    },
}

/// Architecture-owned semantic role of one logical parameter.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum ReplicatedTextParameterRole {
    /// Token lookup table.
    Embedding,
    /// Executable affine projection weight.
    LinearWeight,
    /// Learned affine projection bias.
    LinearBias,
    /// Learned normalization scale or offset.
    Normalization,
    /// Physical scale, zero-point, or packed-format companion.
    FormatCompanion,
    /// Another architecture-declared non-linear parameter.
    Other,
}

/// Architecture-owned location of one replicated-text parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum ReplicatedTextParameterOwner {
    /// Pinned module selected by a stable architecture role.
    StaticRole(String),
    /// One architecture-global execution unit.
    ExecutionUnit {
        /// Stable execution-group identity.
        group: String,
        /// Group-local architecture-global unit index.
        unit: usize,
    },
}

/// Exact admitted presence or derivation of one logical parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum ReplicatedTextParameterPresence {
    /// A required physical source was selected.
    Required,
    /// An optional physical source was present and selected.
    OptionalPresent,
    /// An optional architecture parameter is absent from this artifact.
    OptionalAbsent,
    /// The value is tied to another canonical logical parameter.
    Tied {
        /// Canonical identity supplying the value.
        target: String,
    },
    /// The value is produced by an architecture-owned recipe.
    Derived {
        /// Stable recipe identity.
        recipe: String,
    },
}

impl ReplicatedTextParameterPresence {
    /// Returns whether selection must choose a backend lowering.
    pub fn has_physical_source(&self) -> bool {
        matches!(self, Self::Required | Self::OptionalPresent)
    }
}

/// Exact admitted source and executable constraints for one logical parameter.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReplicatedTextPhysicalSource {
    tensor: String,
    shard: PathBuf,
    output: String,
}

impl ReplicatedTextPhysicalSource {
    /// Records one exact physical tensor, admitted shard, and selected output.
    pub fn new(
        tensor: impl Into<String>,
        shard: impl Into<PathBuf>,
        output: impl Into<String>,
    ) -> Result<Self, ReplicatedTextContractError> {
        let tensor = tensor.into();
        let shard = shard.into();
        let output = output.into();
        if tensor.trim().is_empty() || shard.as_os_str().is_empty() || output.trim().is_empty() {
            return Err(ReplicatedTextContractError::invalid(
                "physical source tensor, shard, and output must be non-empty",
            ));
        }
        Ok(Self {
            tensor,
            shard,
            output,
        })
    }

    /// Physical tensor identity in the admitted container.
    pub fn tensor(&self) -> &str {
        &self.tensor
    }
    /// Canonical admitted payload shard.
    pub fn shard(&self) -> &Path {
        &self.shard
    }
    /// Exact logical output selected from the physical tensor.
    pub fn output(&self) -> &str {
        &self.output
    }
}

/// Exact admitted source and executable constraints for one logical parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextParameterRequirement {
    /// Canonical logical parameter identity.
    name: String,
    /// Physical outputs admitted as sources for this logical parameter.
    sources: Vec<String>,
    /// Exact shard and multi-output provenance for the physical input.
    physical_sources: Vec<ReplicatedTextPhysicalSource>,
    /// All admitted aliases for the logical parameter.
    aliases: Vec<String>,
    /// Encoding of the selected physical source, when present.
    source_encoding: Option<SourceTensorEncoding>,
    /// Exact selected physical source shape, when present.
    physical_shape: Option<Vec<usize>>,
    /// Architecture-declared logical tensor shape.
    logical_shape: Vec<usize>,
    /// Architecture-owned semantic parameter role.
    role: ReplicatedTextParameterRole,
    /// Architecture-owned static/group/unit location.
    owner: ReplicatedTextParameterOwner,
    /// Exact artifact presence, tie, or derivation.
    presence: ReplicatedTextParameterPresence,
    /// Architecture-selected native executable format.
    native_executable: LinearFormat,
    /// Exact architecture-owned transform eligibility and packing axis.
    transform: ParameterTransformConstraint,
}

impl ReplicatedTextParameterRequirement {
    /// Creates one exact logical-parameter requirement.
    #[allow(
        clippy::too_many_arguments,
        reason = "the constructor validates one complete immutable catalog record"
    )]
    pub fn new(
        name: impl Into<String>,
        sources: Vec<String>,
        physical_sources: Vec<ReplicatedTextPhysicalSource>,
        aliases: Vec<String>,
        source_encoding: Option<SourceTensorEncoding>,
        physical_shape: Option<Vec<usize>>,
        logical_shape: Vec<usize>,
        native_executable: LinearFormat,
        role: ReplicatedTextParameterRole,
        owner: ReplicatedTextParameterOwner,
        presence: ReplicatedTextParameterPresence,
        transform: ParameterTransformConstraint,
    ) -> Result<Self, ReplicatedTextContractError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ReplicatedTextContractError::invalid(
                "logical parameter identity is empty",
            ));
        }
        if sources.iter().any(|source| source.trim().is_empty())
            || aliases.iter().any(|alias| alias.trim().is_empty())
        {
            return Err(ReplicatedTextContractError::invalid(format!(
                "logical parameter {name:?} has an empty physical identity"
            )));
        }
        let has_source = !sources.is_empty();
        let has_physical_facts = source_encoding.is_some() && physical_shape.is_some();
        if source_encoding.is_some() != physical_shape.is_some()
            || (has_source && !has_physical_facts)
        {
            return Err(ReplicatedTextContractError::invalid(format!(
                "logical parameter {name:?} has inconsistent source presence"
            )));
        }
        match presence {
            ReplicatedTextParameterPresence::Required
            | ReplicatedTextParameterPresence::OptionalPresent
                if !has_source =>
            {
                return Err(ReplicatedTextContractError::invalid(format!(
                    "physical logical parameter {name:?} has no lowering source"
                )));
            }
            ReplicatedTextParameterPresence::OptionalAbsent
            | ReplicatedTextParameterPresence::Tied { .. }
                if has_source =>
            {
                return Err(ReplicatedTextContractError::invalid(format!(
                    "source-free logical parameter {name:?} has a lowering source"
                )));
            }
            _ => {}
        }
        let provenance_required =
            has_source || matches!(presence, ReplicatedTextParameterPresence::Derived { .. });
        if provenance_required != !physical_sources.is_empty() {
            return Err(ReplicatedTextContractError::invalid(format!(
                "logical parameter {name:?} has inconsistent physical provenance"
            )));
        }
        if physical_sources.is_empty() && has_physical_facts {
            return Err(ReplicatedTextContractError::invalid(format!(
                "logical parameter {name:?} has physical facts without provenance"
            )));
        }
        if physical_shape
            .as_ref()
            .is_some_and(|shape| shape.contains(&0))
        {
            return Err(ReplicatedTextContractError::invalid(format!(
                "logical parameter {name:?} has an invalid physical shape"
            )));
        }
        if logical_shape.contains(&0) {
            return Err(ReplicatedTextContractError::invalid(format!(
                "logical parameter {name:?} has an invalid shape {logical_shape:?}"
            )));
        }
        if let ParameterTransformConstraint::Linear { packed_axis } = transform {
            if packed_axis >= logical_shape.len() {
                return Err(ReplicatedTextContractError::invalid(format!(
                    "logical parameter {name:?} has packing axis {packed_axis} outside shape {logical_shape:?}"
                )));
            }
        }
        Ok(Self {
            name,
            sources,
            physical_sources,
            aliases,
            source_encoding,
            physical_shape,
            logical_shape,
            role,
            owner,
            presence,
            native_executable,
            transform,
        })
    }

    /// Returns the canonical logical identity.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns exact admitted physical source identities.
    pub fn sources(&self) -> &[String] {
        &self.sources
    }

    /// Returns exact admitted shard and multi-output provenance.
    pub fn physical_sources(&self) -> &[ReplicatedTextPhysicalSource] {
        &self.physical_sources
    }

    /// Returns all architecture-admitted alternative source identities.
    pub fn aliases(&self) -> &[String] {
        &self.aliases
    }

    /// Returns the admitted physical source encoding.
    pub const fn source_encoding(&self) -> Option<&SourceTensorEncoding> {
        self.source_encoding.as_ref()
    }

    /// Returns the selected physical shape, when a source is present.
    pub fn physical_shape(&self) -> Option<&[usize]> {
        self.physical_shape.as_deref()
    }

    /// Returns the architecture-declared logical shape.
    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    /// Returns the architecture-owned semantic role.
    pub const fn role(&self) -> ReplicatedTextParameterRole {
        self.role
    }

    /// Returns the architecture-owned static/group/unit location.
    pub const fn owner(&self) -> &ReplicatedTextParameterOwner {
        &self.owner
    }

    /// Returns exact artifact presence, tie, or derivation.
    pub const fn presence(&self) -> &ReplicatedTextParameterPresence {
        &self.presence
    }

    /// Returns whether this logical value selects a physical source lowering.
    ///
    /// Architecture-derived values may retain a physical lowering source when
    /// a recipe splits one encoded tensor into several logical parameters.
    pub fn has_lowering_source(&self) -> bool {
        !self.sources.is_empty()
    }

    /// Returns exact transform eligibility and packing geometry.
    pub const fn transform_constraint(&self) -> ParameterTransformConstraint {
        self.transform
    }

    /// Returns the architecture-native executable format.
    pub const fn native_executable(&self) -> LinearFormat {
        self.native_executable
    }

    /// Resolves a caller transform through architecture-owned constraints.
    pub fn transform_target(
        &self,
        request: QuantizationRequest,
    ) -> Result<Option<ParameterTransformTarget>, ReplicatedTextContractError> {
        let packed_axis = match self.transform {
            ParameterTransformConstraint::None => return Ok(None),
            ParameterTransformConstraint::Linear { packed_axis } => packed_axis,
        };
        let extent = self.logical_shape[packed_axis];
        let executable = match request {
            QuantizationRequest::Affine { group_size, bits } => {
                let group_size = i32::try_from(group_size).map_err(|_| {
                    ReplicatedTextContractError::invalid("affine group size exceeds i32")
                })?;
                let format = eredu_checkpoint::AffineQuantization::new(group_size, i32::from(bits))
                    .map_err(|error| ReplicatedTextContractError::invalid(error.to_string()))?;
                let group_size = usize::try_from(format.group_size).map_err(|_| {
                    ReplicatedTextContractError::invalid("affine group size is negative")
                })?;
                if group_size > extent || !extent.is_multiple_of(group_size) {
                    return Err(ReplicatedTextContractError::invalid(format!(
                        "affine group size {group_size} does not divide packed extent {extent}"
                    )));
                }
                LinearFormat::Affine(format)
            }
            QuantizationRequest::MxFp4 => {
                const MXFP4_BLOCK_SIZE: usize = 32;
                if !extent.is_multiple_of(MXFP4_BLOCK_SIZE) {
                    return Err(ReplicatedTextContractError::invalid(format!(
                        "MXFP4 packed extent {extent} is not divisible by block size {MXFP4_BLOCK_SIZE}"
                    )));
                }
                LinearFormat::MxFp4
            }
            _ => {
                return Err(ReplicatedTextContractError::invalid(
                    "unknown load-time transform request",
                ))
            }
        };
        let descriptor = self.lowering_descriptor(executable)?;
        Ok(Some(ParameterTransformTarget::new(
            request, executable, descriptor,
        )))
    }

    /// Forms the exact backend lowering query for one admitted executable format.
    pub fn lowering_descriptor(
        &self,
        executable: LinearFormat,
    ) -> Result<WeightLoweringDescriptor, ReplicatedTextContractError> {
        let packed_axis = match self.transform {
            ParameterTransformConstraint::None => None,
            ParameterTransformConstraint::Linear { packed_axis } => Some(packed_axis),
        };
        let packed_axis = packed_axis
            .or_else(|| {
                (self.role == ReplicatedTextParameterRole::Embedding
                    && executable != LinearFormat::Dense)
                    .then(|| self.logical_shape.len() - 1)
            })
            .or_else(|| {
                (matches!(
                    self.presence,
                    ReplicatedTextParameterPresence::Derived { .. }
                ) && executable != LinearFormat::Dense)
                    .then(|| {
                        self.physical_shape
                            .as_ref()
                            .expect("derived lowering source has shape")
                            .len()
                            - 1
                    })
            });
        let alias_backed_packed_safetensors = matches!(
            self.source_encoding,
            Some(SourceTensorEncoding::Safetensors(StoredDtype::U32))
        );
        let lowering_shape = if matches!(
            self.presence,
            ReplicatedTextParameterPresence::Derived { .. }
        ) && !alias_backed_packed_safetensors
        {
            self.physical_shape.as_ref().unwrap_or(&self.logical_shape)
        } else {
            &self.logical_shape
        };
        WeightLoweringDescriptor::new(
            self.source_encoding.clone().ok_or_else(|| {
                ReplicatedTextContractError::invalid(format!(
                    "logical parameter {:?} has no physical lowering source",
                    self.name
                ))
            })?,
            executable,
            self.physical_shape.clone().ok_or_else(|| {
                ReplicatedTextContractError::invalid(format!(
                    "logical parameter {:?} has no physical source shape",
                    self.name
                ))
            })?,
            lowering_shape.clone(),
            packed_axis,
        )
    }
}

/// Invalid public replicated-text contract construction.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("invalid replicated text contract: {message}")]
pub struct ReplicatedTextContractError {
    message: String,
}

/// Static state-access semantics required by an admitted replicated text graph.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ReplicatedTextStateAccess {
    /// No mutable token state.
    Stateless,
    /// Ordinary key/value attention state.
    KeyValue,
    /// Architecture-declared recurrent or convolutional components only.
    Fixed,
    /// Key/value attention plus architecture-declared fixed components.
    AttentionWithFixed,
    /// Compressed-latent attention state without fixed components.
    CompressedAttention,
    /// Compressed-latent attention plus architecture-declared fixed components.
    CompressedAttentionWithFixed,
}

impl ReplicatedTextContractError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the stable semantic diagnostic.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Exact architecture and artifact requirements for replicated text execution.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextRequirements {
    architecture_identity: String,
    /// Optional neural operations required by the architecture equations.
    operators: NeuralOperatorCapabilities,
    /// Stable architecture-owned execution graph.
    execution_graph: ExecutionGraph,
    /// Exact group-major execution-unit geometry.
    execution_units: ExecutionUnitLayout,
    /// Architecture-owned transport semantics in graph-group order.
    group_transports: Vec<ArchitectureGroupTransport>,
    /// Complete architecture-owned mutable-state geometry.
    state_layout: StateLayout,
    /// Static state-access semantics used by architecture traversal.
    state_access: ReplicatedTextStateAccess,
    /// Canonical logical parameter requirements.
    parameters: Vec<ReplicatedTextParameterRequirement>,
    derived_recipes: BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
    derived_recipe_outputs: BTreeMap<String, eredu_checkpoint::recipe::RecipeMetadata>,
    grouped_operations: Vec<GroupedOperationRequirement>,
}

impl ReplicatedTextRequirements {
    /// Creates exact requirements from architecture and admitted-artifact facts only.
    #[allow(
        clippy::too_many_arguments,
        reason = "the constructor validates one complete immutable architecture contract"
    )]
    pub fn new(
        architecture_identity: impl Into<String>,
        operators: NeuralOperatorCapabilities,
        execution_graph: ExecutionGraph,
        execution_units: ExecutionUnitLayout,
        group_transports: Vec<ArchitectureGroupTransport>,
        state_layout: StateLayout,
        state_access: ReplicatedTextStateAccess,
        parameters: Vec<ReplicatedTextParameterRequirement>,
    ) -> Result<Self, ReplicatedTextContractError> {
        let architecture_identity = architecture_identity.into();
        if architecture_identity.trim().is_empty() {
            return Err(ReplicatedTextContractError::invalid(
                "architecture identity is empty",
            ));
        }
        if group_transports.len() != execution_graph.groups().len() {
            return Err(ReplicatedTextContractError::invalid(format!(
                "{} group transports do not match {} execution groups",
                group_transports.len(),
                execution_graph.groups().len()
            )));
        }
        if execution_units.group_count() != execution_graph.groups().len()
            || execution_graph
                .groups()
                .iter()
                .enumerate()
                .any(|(index, group)| {
                    execution_units
                        .group_id(index)
                        .is_none_or(|id| id.as_str() != group.id())
                })
        {
            return Err(ReplicatedTextContractError::invalid(
                "execution-unit layout group identities differ from the execution graph",
            ));
        }
        validate_state_access_profile(&state_layout, state_access)?;
        let mut names = BTreeSet::new();
        if parameters
            .iter()
            .any(|parameter| !names.insert(parameter.name()))
        {
            return Err(ReplicatedTextContractError::invalid(
                "logical parameter identities are not unique",
            ));
        }
        Ok(Self {
            architecture_identity,
            operators,
            execution_graph,
            execution_units,
            group_transports,
            state_layout,
            state_access,
            parameters,
            derived_recipes: BTreeMap::new(),
            derived_recipe_outputs: BTreeMap::new(),
            grouped_operations: Vec::new(),
        })
    }

    /// Attaches the exact architecture-owned derivations selected for this artifact.
    pub fn with_derived_recipes(
        mut self,
        recipes: BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        outputs: BTreeMap<String, eredu_checkpoint::recipe::RecipeMetadata>,
    ) -> Result<Self, ReplicatedTextContractError> {
        if recipes.keys().ne(outputs.keys()) {
            return Err(ReplicatedTextContractError::invalid(
                "derived recipe targets and inferred outputs differ",
            ));
        }
        for target in recipes.keys() {
            let recipe = recipes
                .get(target)
                .expect("recipe target came from the same map");
            let parameter = self
                .parameters
                .iter_mut()
                .find(|parameter| parameter.name == *target)
                .ok_or_else(|| {
                    ReplicatedTextContractError::invalid(format!(
                        "derived recipe target {target:?} is not a declared parameter"
                    ))
                })?;
            if matches!(
                parameter.presence,
                ReplicatedTextParameterPresence::OptionalAbsent
                    | ReplicatedTextParameterPresence::Tied { .. }
            ) {
                return Err(ReplicatedTextContractError::invalid(format!(
                    "derived recipe target {target:?} has no independent artifact value"
                )));
            }
            parameter.presence = ReplicatedTextParameterPresence::Derived {
                recipe: "architecture.recipe".into(),
            };
            parameter.sources = recipe
                .source_keys()
                .into_iter()
                .map(str::to_owned)
                .collect();
        }
        self.derived_recipes = recipes;
        self.derived_recipe_outputs = outputs;
        Ok(self)
    }

    /// Returns the normalized architecture identity bound during admission.
    pub fn architecture_identity(&self) -> &str {
        &self.architecture_identity
    }

    /// Declares exact grouped operations required by this architecture path.
    pub fn with_grouped_operations(
        mut self,
        operations: impl IntoIterator<Item = GroupedOperationRequirement>,
    ) -> Self {
        self.grouped_operations = operations.into_iter().collect();
        self
    }

    /// Returns required optional neural-operation semantics.
    pub const fn operators(&self) -> NeuralOperatorCapabilities {
        self.operators
    }
    /// Returns the architecture-owned execution graph.
    pub const fn execution_graph(&self) -> &ExecutionGraph {
        &self.execution_graph
    }
    /// Returns group-major execution-unit geometry.
    pub const fn execution_units(&self) -> &ExecutionUnitLayout {
        &self.execution_units
    }
    /// Returns architecture-owned group transports.
    pub fn group_transports(&self) -> &[ArchitectureGroupTransport] {
        &self.group_transports
    }
    /// Returns complete mutable-state geometry.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }
    /// Returns the state-access semantics used by typed traversal.
    pub const fn state_access(&self) -> ReplicatedTextStateAccess {
        self.state_access
    }
    /// Returns canonical logical parameter requirements.
    pub fn parameters(&self) -> &[ReplicatedTextParameterRequirement] {
        &self.parameters
    }
    /// Returns exact derivations that are part of the selected artifact contract.
    pub fn derived_recipes(
        &self,
    ) -> &BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe> {
        &self.derived_recipes
    }
    /// Returns admission-time output metadata for every exact derivation.
    pub fn derived_recipe_outputs(
        &self,
    ) -> &BTreeMap<String, eredu_checkpoint::recipe::RecipeMetadata> {
        &self.derived_recipe_outputs
    }
    /// Returns exact grouped operations required before construction.
    pub fn grouped_operations(&self) -> &[GroupedOperationRequirement] {
        &self.grouped_operations
    }
}

fn validate_state_access_profile(
    layout: &StateLayout,
    access: ReplicatedTextStateAccess,
) -> Result<(), ReplicatedTextContractError> {
    use eredu_core::cache::StateComponentRole;

    let roles = (0..layout.len())
        .flat_map(|layer| {
            layout
                .components(layer)
                .expect("validated state layout exposes every layer")
        })
        .map(StateComponentPolicy::role)
        .collect::<Vec<_>>();
    let ordinary = |role| {
        matches!(
            role,
            StateComponentRole::AttentionKeys | StateComponentRole::AttentionValues
        )
    };
    let compressed = |role| {
        matches!(
            role,
            StateComponentRole::CompressedLatent | StateComponentRole::RotaryKeys
        )
    };
    let fixed = |role| matches!(role, StateComponentRole::Fixed(_));
    let has_ordinary = roles.iter().copied().any(ordinary);
    let has_compressed = roles.iter().copied().any(compressed);
    let has_fixed = roles.iter().copied().any(fixed);
    let coherent = match access {
        ReplicatedTextStateAccess::Stateless => roles.is_empty(),
        ReplicatedTextStateAccess::KeyValue => roles.iter().copied().all(ordinary) && has_ordinary,
        ReplicatedTextStateAccess::Fixed => roles.iter().copied().all(fixed) && has_fixed,
        ReplicatedTextStateAccess::AttentionWithFixed => {
            roles
                .iter()
                .copied()
                .all(|role| ordinary(role) || fixed(role))
                && has_ordinary
                && has_fixed
        }
        ReplicatedTextStateAccess::CompressedAttention => {
            roles.iter().copied().all(compressed) && has_compressed
        }
        ReplicatedTextStateAccess::CompressedAttentionWithFixed => {
            roles
                .iter()
                .copied()
                .all(|role| compressed(role) || fixed(role))
                && has_compressed
                && has_fixed
        }
    };
    if !coherent {
        return Err(ReplicatedTextContractError::invalid(format!(
            "state access profile {access:?} does not match component roles {roles:?}"
        )));
    }
    Ok(())
}

/// One required grouped-compute mechanism.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum GroupedOperationRequirement {
    /// Ordinary grouped gated-product output.
    GatedProduct,
    /// Rank-local gated-product partial with an explicit post-reduce term.
    GatedProductTensorParallelPartial,
    /// Ordinary grouped ReLU-squared output.
    Relu2,
    /// Rank-local ReLU-squared partial with an explicit post-reduce term.
    Relu2TensorParallelPartial,
}

/// Generic independently addressable storage facilities implemented by a backend.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct AddressableStorageCapabilities {
    bulk_access: bool,
    incremental_access: bool,
    lease_completion: bool,
    maximum_compact_bytes: u64,
    tiers: AddressableStorageTiers,
}

/// Generic storage tiers usable by an independently addressable bank.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct AddressableStorageTiers {
    device: bool,
    host: bool,
    disk: bool,
}

impl AddressableStorageTiers {
    /// Creates one exact tier capability set.
    pub const fn new(device: bool, host: bool, disk: bool) -> Self {
        Self { device, host, disk }
    }

    /// Returns whether executable device storage is available.
    pub const fn device(self) -> bool {
        self.device
    }

    /// Returns whether host staging storage is available.
    pub const fn host(self) -> bool {
        self.host
    }

    /// Returns whether lazy checkpoint-backed storage is available.
    pub const fn disk(self) -> bool {
        self.disk
    }
}

impl AddressableStorageCapabilities {
    /// Creates an exact addressable-storage capability report.
    pub const fn new(
        bulk_access: bool,
        incremental_access: bool,
        lease_completion: bool,
        maximum_compact_bytes: u64,
    ) -> Self {
        Self {
            bulk_access,
            incremental_access,
            lease_completion,
            maximum_compact_bytes,
            tiers: AddressableStorageTiers::new(true, true, true),
        }
    }

    /// Replaces the exact supported storage-tier set.
    pub const fn with_tiers(mut self, tiers: AddressableStorageTiers) -> Self {
        self.tiers = tiers;
        self
    }

    /// Returns whether bounded multi-row access is implemented.
    pub const fn bulk_access(self) -> bool {
        self.bulk_access
    }

    /// Returns whether latency-sensitive incremental access is implemented.
    pub const fn incremental_access(self) -> bool {
        self.incremental_access
    }

    /// Returns whether acquisitions remain leased through native completion.
    pub const fn lease_completion(self) -> bool {
        self.lease_completion
    }

    /// Returns the largest supported per-acquisition compact bank.
    pub const fn maximum_compact_bytes(self) -> u64 {
        self.maximum_compact_bytes
    }

    /// Returns the exact independently addressable storage tiers.
    pub const fn tiers(self) -> AddressableStorageTiers {
        self.tiers
    }
}

/// Family- and execution-class-neutral backend mechanism report.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BackendMechanismCapabilities {
    /// Optional neural operations implemented by the backend.
    operators: NeuralOperatorCapabilities,
    /// Exact admitted source-to-executable lowerings.
    weight_lowerings: Vec<WeightLoweringCapability>,
    /// Ordinary parameter residency mechanisms.
    weight_residencies: Vec<WeightResidencyMechanism>,
    /// Exact mutable-state component and lifecycle mechanisms.
    state: StateMechanismCapabilities,
    /// Exact session facilities implemented by the constructed session.
    session: SessionCapabilities,
    /// Prompt-cache persistence mechanism is available.
    prompt_cache: bool,
    /// Exact completion ownership is implemented for submitted work.
    exact_completion: bool,
    grouped_operations: Vec<GroupedOperationRequirement>,
    indexed_movement: bool,
    addressable_storage: Option<AddressableStorageCapabilities>,
}

impl BackendMechanismCapabilities {
    /// Creates a fail-closed mechanism report.
    pub fn new(
        operators: NeuralOperatorCapabilities,
        weight_lowerings: Vec<WeightLoweringCapability>,
        weight_residencies: Vec<WeightResidencyMechanism>,
        state: StateMechanismCapabilities,
    ) -> Self {
        Self {
            operators,
            weight_lowerings,
            weight_residencies,
            state,
            session: SessionCapabilities::default(),
            prompt_cache: false,
            exact_completion: false,
            grouped_operations: Vec::new(),
            indexed_movement: false,
            addressable_storage: None,
        }
    }

    /// Adds supported session-observation and persistence mechanisms.
    pub const fn with_session(mut self, session: SessionCapabilities) -> Self {
        self.session = session;
        self
    }
    /// Declares prompt-cache persistence support.
    pub const fn with_prompt_cache(mut self, supported: bool) -> Self {
        self.prompt_cache = supported;
        self
    }
    /// Declares exact native-completion ownership support.
    pub const fn with_exact_completion(mut self, supported: bool) -> Self {
        self.exact_completion = supported;
        self
    }
    /// Declares exact grouped operation mechanisms.
    pub fn with_grouped_operations(
        mut self,
        operations: impl IntoIterator<Item = GroupedOperationRequirement>,
    ) -> Self {
        self.grouped_operations = operations.into_iter().collect();
        self
    }
    /// Declares generic indexed discovery, slicing, remapping, and concatenation.
    pub const fn with_indexed_movement(mut self, supported: bool) -> Self {
        self.indexed_movement = supported;
        self
    }
    /// Declares generic independently addressable storage facilities.
    pub const fn with_addressable_storage(
        mut self,
        capabilities: AddressableStorageCapabilities,
    ) -> Self {
        self.addressable_storage = Some(capabilities);
        self
    }
    /// Returns neural-operation mechanisms.
    pub const fn operators(&self) -> NeuralOperatorCapabilities {
        self.operators
    }
    /// Returns source-to-executable weight-lowering mechanisms.
    pub fn weight_lowerings(&self) -> &[WeightLoweringCapability] {
        &self.weight_lowerings
    }
    /// Returns weight-residency mechanisms.
    pub fn weight_residencies(&self) -> &[WeightResidencyMechanism] {
        &self.weight_residencies
    }
    /// Returns exact mutable-state component and lifecycle mechanisms.
    pub const fn state(&self) -> &StateMechanismCapabilities {
        &self.state
    }
    /// Returns session-observation and persistence mechanisms.
    pub const fn session(&self) -> SessionCapabilities {
        self.session
    }
    /// Returns whether prompt-cache persistence is supported.
    pub const fn prompt_cache(&self) -> bool {
        self.prompt_cache
    }
    /// Returns whether exact native-completion ownership is supported.
    pub const fn exact_completion(&self) -> bool {
        self.exact_completion
    }
    /// Returns exact grouped operation mechanisms.
    pub fn grouped_operations(&self) -> &[GroupedOperationRequirement] {
        &self.grouped_operations
    }
    /// Returns whether generic indexed movement is implemented.
    pub const fn indexed_movement(&self) -> bool {
        self.indexed_movement
    }
    /// Returns independently addressable storage facilities, when implemented.
    pub const fn addressable_storage(&self) -> Option<AddressableStorageCapabilities> {
        self.addressable_storage
    }
}

/// Caller choices resolved while selecting one replicated text realization.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextSelectionRequest {
    /// Requested execution topology.
    topology: Option<ParallelTopology>,
    /// Requested ordinary parameter residency.
    residency: LayerWeightResidency,
    /// Requested mutable-state implementation and its exact residency policy.
    state: CacheResidencyPolicy,
    /// Optional load-time transform.
    quantization: Option<QuantizationRequest>,
    /// Requested optional session facilities.
    session: SessionCapabilities,
    /// Whether prompt-cache persistence is requested.
    prompt_cache: bool,
    /// Whether exact completion ownership is requested.
    exact_completion: bool,
}

impl ReplicatedTextSelectionRequest {
    /// Creates a replicated request with fail-closed optional facilities.
    pub fn new(residency: LayerWeightResidency, state: CacheResidencyPolicy) -> Self {
        Self {
            topology: None,
            residency,
            state,
            quantization: None,
            session: SessionCapabilities::default(),
            prompt_cache: false,
            exact_completion: false,
        }
    }
    /// Sets the requested topology.
    pub const fn with_topology(mut self, topology: ParallelTopology) -> Self {
        self.topology = Some(topology);
        self
    }
    /// Sets the optional load-time transform.
    pub const fn with_quantization(mut self, quantization: QuantizationRequest) -> Self {
        self.quantization = Some(quantization);
        self
    }
    /// Sets requested session facilities.
    pub const fn with_session(mut self, session: SessionCapabilities) -> Self {
        self.session = session;
        self
    }
    /// Requests prompt-cache persistence.
    pub const fn with_prompt_cache(mut self, required: bool) -> Self {
        self.prompt_cache = required;
        self
    }
    /// Requests exact completion ownership.
    pub const fn with_exact_completion(mut self, required: bool) -> Self {
        self.exact_completion = required;
        self
    }
    /// Returns the requested topology, where `None` means replicated.
    pub const fn topology(&self) -> Option<ParallelTopology> {
        self.topology
    }
    /// Returns the requested weight residency.
    pub const fn residency(&self) -> LayerWeightResidency {
        self.residency
    }
    /// Returns the requested state policy.
    pub const fn state(&self) -> &CacheResidencyPolicy {
        &self.state
    }
    /// Returns the requested transform.
    pub const fn quantization(&self) -> Option<QuantizationRequest> {
        self.quantization
    }
    /// Returns requested session facilities.
    pub const fn session(&self) -> SessionCapabilities {
        self.session
    }
    /// Returns whether prompt-cache persistence is requested.
    pub const fn prompt_cache(&self) -> bool {
        self.prompt_cache
    }
    /// Returns whether exact completion ownership is requested.
    pub const fn exact_completion(&self) -> bool {
        self.exact_completion
    }
}

/// Selected lowering for one canonical logical parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedParameterRealization {
    /// Canonical logical parameter identity.
    name: String,
    /// Physical outputs admitted as sources for this logical parameter.
    sources: Vec<String>,
    physical_sources: Vec<ReplicatedTextPhysicalSource>,
    /// Admitted physical encoding.
    source_encoding: SourceTensorEncoding,
    /// Exact executable format used to construct the architecture module.
    executable: LinearFormat,
    /// Backend lowering selected for materialization.
    lowering: WeightLoweringKind,
}

/// Exact backend work item for one selected logical parameter.
///
/// This value joins architecture-owned topology and artifact facts with the
/// authoritative selected lowering. A materializer may batch these tasks, but
/// it must not replace them with one model-wide transform choice.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextMaterializationTask {
    name: String,
    sources: Vec<String>,
    physical_sources: Vec<ReplicatedTextPhysicalSource>,
    aliases: Vec<String>,
    source_encoding: SourceTensorEncoding,
    physical_shape: Vec<usize>,
    logical_shape: Vec<usize>,
    role: ReplicatedTextParameterRole,
    owner: ReplicatedTextParameterOwner,
    presence: ReplicatedTextParameterPresence,
    executable: LinearFormat,
    lowering: WeightLoweringKind,
    lowering_descriptor: WeightLoweringDescriptor,
    derived_recipe: Option<eredu_checkpoint::recipe::DerivedWeightRecipe>,
    derived_output: Option<eredu_checkpoint::recipe::RecipeMetadata>,
    output_companions: Vec<ReplicatedTextOutputCompanion>,
}

/// Architecture-declared output companion for one materialized linear weight.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextOutputCompanion {
    name: String,
    role: eredu_nn::LinearCompanionRole,
    logical_shape: Vec<usize>,
    owner: ParameterGroupOwner,
    materialization_task: Option<Box<ReplicatedTextMaterializationTask>>,
    catalog_source: Option<ReplicatedTextPhysicalSource>,
    derived_recipe: Option<eredu_checkpoint::recipe::DerivedWeightRecipe>,
    derived_output: Option<eredu_checkpoint::recipe::RecipeMetadata>,
}

impl ReplicatedTextOutputCompanion {
    /// Creates one exact output companion identity and semantic role.
    pub fn new(
        name: impl Into<String>,
        role: eredu_nn::LinearCompanionRole,
        logical_shape: Vec<usize>,
        owner: ParameterGroupOwner,
    ) -> Result<Self, ReplicatedTextContractError> {
        let name = name.into();
        if name.trim().is_empty() || logical_shape.is_empty() || logical_shape.contains(&0) {
            return Err(ReplicatedTextContractError::invalid(
                "materialization output companion identity or geometry is invalid",
            ));
        }
        Ok(Self {
            name,
            role,
            logical_shape,
            owner,
            materialization_task: None,
            catalog_source: None,
            derived_recipe: None,
            derived_output: None,
        })
    }

    pub(crate) fn with_derived_recipe(
        mut self,
        recipe: eredu_checkpoint::recipe::DerivedWeightRecipe,
        output: eredu_checkpoint::recipe::RecipeMetadata,
    ) -> Self {
        self.derived_recipe = Some(recipe);
        self.derived_output = Some(output);
        self
    }

    /// Returns the exact architecture-declared parameter identity.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the companion's role in the encoded linear parameter.
    pub const fn role(&self) -> eredu_nn::LinearCompanionRole {
        self.role
    }

    /// Returns the exact architecture-declared companion geometry.
    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    /// Returns the exact architecture-declared companion owner.
    pub const fn owner(&self) -> &ParameterGroupOwner {
        &self.owner
    }

    pub(crate) fn with_materialization_task(
        mut self,
        task: ReplicatedTextMaterializationTask,
    ) -> Result<Self, ReplicatedTextContractError> {
        let names_output =
            task.name() == self.name || task.aliases().iter().any(|alias| alias == &self.name);
        if !names_output || !task.output_companions().is_empty() {
            return Err(ReplicatedTextContractError::invalid(format!(
                "companion {:?} has an inconsistent standalone materialization task",
                self.name
            )));
        }
        self.materialization_task = Some(Box::new(task));
        Ok(self)
    }

    pub(crate) fn with_catalog_source(mut self, source: ReplicatedTextPhysicalSource) -> Self {
        self.catalog_source = Some(source);
        self
    }

    /// Returns the standalone selected materialization task, when one exists.
    ///
    /// Generated transform outputs and translated checkpoint catalog outputs
    /// instead retain their causal source on the primary task or companion.
    pub fn materialization_task(&self) -> Option<&ReplicatedTextMaterializationTask> {
        self.materialization_task.as_deref()
    }

    /// Returns exact translated-catalog provenance for this companion.
    pub const fn catalog_source(&self) -> Option<&ReplicatedTextPhysicalSource> {
        self.catalog_source.as_ref()
    }

    /// Returns the architecture-owned companion derivation, when required.
    pub const fn derived_recipe(&self) -> Option<&eredu_checkpoint::recipe::DerivedWeightRecipe> {
        self.derived_recipe.as_ref()
    }

    /// Returns admission-time metadata for the derived companion output.
    pub const fn derived_output(&self) -> Option<&eredu_checkpoint::recipe::RecipeMetadata> {
        self.derived_output.as_ref()
    }
}

impl ReplicatedTextMaterializationTask {
    pub(crate) fn set_output_companions(
        &mut self,
        mut companions: Vec<ReplicatedTextOutputCompanion>,
    ) -> Result<(), ReplicatedTextContractError> {
        companions.sort_by(|left, right| {
            left.role
                .cmp(&right.role)
                .then_with(|| left.name.cmp(&right.name))
        });
        if companions
            .windows(2)
            .any(|pair| pair[0].name == pair[1].name || pair[0].role == pair[1].role)
        {
            return Err(ReplicatedTextContractError::invalid(format!(
                "materialization task {:?} has duplicate output companions",
                self.name
            )));
        }
        let roles = companions
            .iter()
            .map(|companion| companion.role)
            .collect::<Vec<_>>();
        let expected = match self.executable {
            LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => Vec::new(),
            LinearFormat::MxFp4 | LinearFormat::E4M3BlockFp8(_) => {
                vec![eredu_nn::LinearCompanionRole::Scale]
            }
            LinearFormat::Affine(_) => vec![
                eredu_nn::LinearCompanionRole::Scale,
                eredu_nn::LinearCompanionRole::AffineBias,
            ],
        };
        let mut expected = expected;
        expected.sort();
        if roles != expected {
            return Err(ReplicatedTextContractError::invalid(format!(
                "materialization task {:?} executable {:?} requires companion roles {:?}, got {:?}",
                self.name, self.executable, expected, roles
            )));
        }
        self.output_companions = companions;
        Ok(())
    }

    /// Returns the canonical logical parameter identity.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns every admitted physical source identity.
    pub fn sources(&self) -> &[String] {
        &self.sources
    }

    /// Returns exact shard and translated-output provenance.
    pub fn physical_sources(&self) -> &[ReplicatedTextPhysicalSource] {
        &self.physical_sources
    }

    /// Returns every architecture-admitted alias.
    pub fn aliases(&self) -> &[String] {
        &self.aliases
    }

    /// Returns the exact admitted source encoding.
    pub const fn source_encoding(&self) -> &SourceTensorEncoding {
        &self.source_encoding
    }

    /// Returns the admitted physical source geometry.
    pub fn physical_shape(&self) -> &[usize] {
        &self.physical_shape
    }

    /// Returns the architecture-declared logical geometry.
    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    /// Returns the architecture-owned semantic parameter role.
    pub const fn role(&self) -> ReplicatedTextParameterRole {
        self.role
    }

    /// Returns the architecture-owned module location.
    pub const fn owner(&self) -> &ReplicatedTextParameterOwner {
        &self.owner
    }

    /// Returns the exact admitted presence or derivation.
    pub const fn presence(&self) -> &ReplicatedTextParameterPresence {
        &self.presence
    }

    /// Returns the selected executable format.
    pub const fn executable(&self) -> LinearFormat {
        self.executable
    }

    /// Returns the selected backend lowering mechanism.
    pub const fn lowering(&self) -> WeightLoweringKind {
        self.lowering
    }

    /// Returns the complete geometry-bearing lowering request.
    pub const fn lowering_descriptor(&self) -> &WeightLoweringDescriptor {
        &self.lowering_descriptor
    }

    /// Returns the architecture-owned derivation, when this output is derived.
    pub const fn derived_recipe(&self) -> Option<&eredu_checkpoint::recipe::DerivedWeightRecipe> {
        self.derived_recipe.as_ref()
    }

    /// Returns admission-time metadata for the derived output.
    pub const fn derived_output(&self) -> Option<&eredu_checkpoint::recipe::RecipeMetadata> {
        self.derived_output.as_ref()
    }

    /// Returns exact output companion identities declared by the architecture.
    pub fn output_companions(&self) -> &[ReplicatedTextOutputCompanion] {
        &self.output_companions
    }

    /// Returns the exact source recipe selected for this task.
    ///
    /// Direct tasks are represented as a full selection of their single
    /// admitted source. Derived tasks return the architecture-owned recipe
    /// without reconstructing it from a checkpoint catalog.
    pub fn source_recipe(
        &self,
    ) -> Result<eredu_checkpoint::recipe::DerivedWeightRecipe, ReplicatedTextContractError> {
        let expects_recipe = matches!(
            self.lowering,
            WeightLoweringKind::Derived | WeightLoweringKind::DerivedTransform
        );
        match (expects_recipe, self.derived_recipe.as_ref()) {
            (true, Some(recipe)) => {
                let declared = self
                    .sources
                    .iter()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>();
                let consumed = recipe.source_keys().into_iter().collect::<BTreeSet<_>>();
                if declared != consumed {
                    return Err(ReplicatedTextContractError::invalid(format!(
                        "materialization task {:?} recipe sources differ from its exact source catalog",
                        self.name
                    )));
                }
                Ok(recipe.clone())
            }
            (false, None) => {
                let [source] = self.sources.as_slice() else {
                    return Err(ReplicatedTextContractError::invalid(format!(
                        "direct materialization task {:?} must name exactly one source",
                        self.name
                    )));
                };
                Ok(eredu_checkpoint::recipe::DerivedWeightRecipe::source(
                    source.clone(),
                    eredu_checkpoint::store::TensorSelection::Full,
                ))
            }
            (true, None) => Err(ReplicatedTextContractError::invalid(format!(
                "derived materialization task {:?} has no exact recipe",
                self.name
            ))),
            (false, Some(_)) => Err(ReplicatedTextContractError::invalid(format!(
                "direct materialization task {:?} unexpectedly carries a recipe",
                self.name
            ))),
        }
    }
}

/// Projects an authoritative selection into exact materialization work.
///
/// Every selected parameter must agree with its immutable requirement. The
/// returned sequence preserves selected-parameter order and contains no
/// model-wide quantization or transform value.
pub fn replicated_text_materialization_tasks(
    selected: &SelectedReplicatedTextRealization,
) -> Result<Vec<ReplicatedTextMaterializationTask>, ReplicatedTextContractError> {
    let requirements = selected.requirements();
    selected
        .parameters()
        .iter()
        .map(|realization| {
            let requirement = requirements
                .parameters()
                .iter()
                .find(|requirement| requirement.name() == realization.name())
                .ok_or_else(|| {
                    ReplicatedTextContractError::invalid(format!(
                        "selected parameter {:?} has no architecture requirement",
                        realization.name()
                    ))
                })?;
            if requirement.sources() != realization.sources()
                || requirement.physical_sources() != realization.physical_sources()
                || requirement.source_encoding() != Some(realization.source_encoding())
            {
                return Err(ReplicatedTextContractError::invalid(format!(
                    "selected parameter {:?} changed admitted source provenance",
                    realization.name()
                )));
            }
            let physical_shape = requirement.physical_shape().ok_or_else(|| {
                ReplicatedTextContractError::invalid(format!(
                    "selected parameter {:?} has no physical geometry",
                    realization.name()
                ))
            })?;
            let lowering_descriptor = requirement.lowering_descriptor(realization.executable())?;
            if lowering_descriptor.source() != realization.source_encoding() {
                return Err(ReplicatedTextContractError::invalid(format!(
                    "selected parameter {:?} changed its lowering source encoding",
                    realization.name()
                )));
            }
            let derived_recipe = requirements
                .derived_recipes()
                .get(realization.name())
                .cloned();
            let derived_output = requirements
                .derived_recipe_outputs()
                .get(realization.name())
                .cloned();
            if derived_recipe.is_some() != derived_output.is_some() {
                return Err(ReplicatedTextContractError::invalid(format!(
                    "selected parameter {:?} has incomplete derived metadata",
                    realization.name()
                )));
            }
            Ok(ReplicatedTextMaterializationTask {
                name: realization.name().to_owned(),
                sources: realization.sources().to_vec(),
                physical_sources: realization.physical_sources().to_vec(),
                aliases: requirement.aliases().to_vec(),
                source_encoding: realization.source_encoding().clone(),
                physical_shape: physical_shape.to_vec(),
                logical_shape: requirement.logical_shape().to_vec(),
                role: requirement.role(),
                owner: requirement.owner().clone(),
                presence: requirement.presence().clone(),
                executable: realization.executable(),
                lowering: realization.lowering(),
                lowering_descriptor,
                derived_recipe,
                derived_output,
                output_companions: Vec::new(),
            })
        })
        .collect()
}

/// Projects selected text materialization into one exact architecture partition.
///
/// Encoded-linear companions are reconstructed from the architecture's
/// validated physical parameter groups and remain atomic with their primary
/// task. If a partition would own only part of such a physical family, the
/// complete projection fails instead of retaining an unowned output.
pub fn partitioned_replicated_text_materialization_tasks<G, A>(
    selected: &SelectedReplicatedTextRealization,
    parameters: &ArchitectureParameterDescription,
    partition: &ArchitecturePartition<G, A>,
) -> Result<Vec<ReplicatedTextMaterializationTask>, ReplicatedTextContractError> {
    let mut companions = BTreeMap::<String, Vec<ReplicatedTextOutputCompanion>>::new();
    let mut all_targets = BTreeSet::new();
    let mut owned_targets = BTreeSet::new();
    for tagged in parameters.groups() {
        let local = partition.parameter_bindings().iter().any(|binding| {
            binding.owner() == tagged.owner()
                && parameter_groups_have_same_members(binding.group(), tagged.group())
        });
        let group_targets = tagged
            .members()
            .iter()
            .map(|member| member.target())
            .collect::<BTreeSet<_>>();
        for member in tagged.members() {
            if !all_targets.insert(member.target().to_owned()) {
                return Err(ReplicatedTextContractError::invalid(format!(
                    "architecture parameter target {:?} appears more than once",
                    member.target()
                )));
            }
            if local {
                owned_targets.insert(member.target().to_owned());
            }
            match (member.linear_companion(), member.linear_companion_of()) {
                (None, None) => {}
                (Some(role), Some(primary)) if group_targets.contains(primary) => {
                    companions.entry(primary.to_owned()).or_default().push(
                        ReplicatedTextOutputCompanion::new(
                            member.target(),
                            role,
                            member.global_shape().to_vec(),
                            tagged.owner().clone(),
                        )?,
                    );
                }
                (Some(_), Some(primary)) => {
                    return Err(ReplicatedTextContractError::invalid(format!(
                        "physical companion {:?} names primary {primary:?} outside its atomic parameter group",
                        member.target()
                    )));
                }
                _ => {
                    return Err(ReplicatedTextContractError::invalid(format!(
                        "physical parameter {:?} has incomplete companion metadata",
                        member.target()
                    )));
                }
            }
        }
    }

    let mut tasks = replicated_text_materialization_tasks(selected)?;
    let mut topology_targets = BTreeMap::<String, String>::new();
    let mut target_claims = BTreeMap::<String, String>::new();
    for task in &tasks {
        let matches = std::iter::once(task.name())
            .chain(task.aliases().iter().map(String::as_str))
            .filter(|candidate| all_targets.contains(*candidate))
            .collect::<BTreeSet<_>>();
        if matches.len() != 1 {
            return Err(ReplicatedTextContractError::invalid(format!(
                "selected materialization output {:?} resolves to {} architecture topology targets through its canonical identity and admitted aliases: {:?}",
                task.name(),
                matches.len(),
                matches
            )));
        }
        let target = matches.first().expect("one topology target was validated");
        if let Some(previous) = target_claims.insert((*target).to_owned(), task.name().to_owned()) {
            return Err(ReplicatedTextContractError::invalid(format!(
                "selected materialization outputs {previous:?} and {:?} ambiguously resolve to architecture target {target:?}",
                task.name()
            )));
        }
        topology_targets.insert(task.name().to_owned(), (*target).to_owned());
    }
    let companion_names = companions
        .values()
        .flatten()
        .map(|companion| companion.name().to_owned())
        .collect::<BTreeSet<_>>();
    let standalone = tasks
        .iter()
        .filter_map(|task| {
            let target = topology_targets
                .get(task.name())
                .expect("every task has one validated topology target");
            companion_names
                .contains(target)
                .then(|| (target.clone(), task.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    for task in &mut tasks {
        let topology_target = topology_targets
            .get(task.name())
            .expect("every task has one validated topology target");
        let attached = companions
            .remove(topology_target)
            .unwrap_or_default()
            .into_iter()
            .map(
                |companion| match standalone.get(companion.name()).cloned() {
                    Some(exact) => companion.with_materialization_task(exact),
                    None => Ok(companion),
                },
            )
            .collect::<Result<Vec<_>, _>>()?;
        task.set_output_companions(attached)?;
    }
    if !companions.is_empty() {
        return Err(ReplicatedTextContractError::invalid(format!(
            "architecture companions name missing primary tasks: {:?}",
            companions.keys().collect::<Vec<_>>()
        )));
    }
    tasks.retain(|task| {
        !companion_names.contains(
            topology_targets
                .get(task.name())
                .expect("every task has one validated topology target"),
        )
    });

    let mut projected = Vec::new();
    for task in tasks {
        let topology_target = topology_targets
            .get(task.name())
            .expect("every task has one validated topology target");
        let emitted = std::iter::once(topology_target.as_str())
            .chain(
                task.output_companions()
                    .iter()
                    .map(ReplicatedTextOutputCompanion::name),
            )
            .collect::<Vec<_>>();
        let local = emitted
            .iter()
            .filter(|target| owned_targets.contains(**target))
            .count();
        match local {
            0 => {}
            count if count == emitted.len() => projected.push(task),
            count => {
                return Err(ReplicatedTextContractError::invalid(format!(
                    "materialization task {:?} would emit {count} of {} outputs into this partition",
                    task.name(),
                    emitted.len()
                )));
            }
        }
    }
    Ok(projected)
}

/// Parameter visitation order is an implementation detail of a local module,
/// while an architecture parameter group is an atomic, target-keyed contract.
/// Compare that contract without making otherwise-identical local ownership
/// depend on whether a backend-neutral module visits a bias before its weight.
fn parameter_groups_have_same_members(
    left: &ParameterGroupSpec,
    right: &ParameterGroupSpec,
) -> bool {
    left.logical_name() == right.logical_name()
        && left.role() == right.role()
        && left.partition_units() == right.partition_units()
        && left.members().len() == right.members().len()
        && left.members().iter().all(|left_member| {
            right.members().iter().any(|right_member| {
                left_member.target() == right_member.target()
                    && left_member.global_shape() == right_member.global_shape()
                    && left_member.sharding() == right_member.sharding()
                    && left_member.linear_companion() == right_member.linear_companion()
                    && left_member.linear_companion_of() == right_member.linear_companion_of()
            })
        })
}

impl SelectedParameterRealization {
    /// Returns the canonical logical identity.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Returns admitted physical source identities.
    pub fn sources(&self) -> &[String] {
        &self.sources
    }
    /// Returns the exact selected shard and multi-output provenance.
    pub fn physical_sources(&self) -> &[ReplicatedTextPhysicalSource] {
        &self.physical_sources
    }
    /// Returns the admitted source encoding.
    pub const fn source_encoding(&self) -> &SourceTensorEncoding {
        &self.source_encoding
    }
    /// Returns the selected executable format.
    pub const fn executable(&self) -> LinearFormat {
        self.executable
    }
    /// Returns the selected backend lowering kind.
    pub const fn lowering(&self) -> WeightLoweringKind {
        self.lowering
    }
}

/// Selected physical realization of one exact semantic state component.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedStateComponentRealization {
    layer: usize,
    component: StateComponentPolicy,
    placement: StateComponentPlacement,
}

impl SelectedStateComponentRealization {
    /// Returns the architecture-global state layer.
    pub const fn layer(&self) -> usize {
        self.layer
    }

    /// Returns the exact architecture-declared component contract.
    pub const fn component(&self) -> &StateComponentPolicy {
        &self.component
    }

    /// Returns the selected physical placement.
    pub const fn placement(&self) -> StateComponentPlacement {
        self.placement
    }
}

/// Authoritative mutable-state realization selected before allocation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedStateRealization {
    layout: StateLayout,
    access: ReplicatedTextStateAccess,
    policy: CacheResidencyPolicy,
    components: Vec<SelectedStateComponentRealization>,
    checkpoint: bool,
    rollback: bool,
    reset: bool,
    prompt_cache: bool,
    observation_retention: bool,
}

impl SelectedStateRealization {
    /// Returns the exact architecture-owned state layout.
    pub const fn layout(&self) -> &StateLayout {
        &self.layout
    }

    /// Selects the exact rank-local state interval while preserving global ownership proof.
    ///
    /// Component ordinals are rebased to the local layout consumed by a rank-local runtime;
    /// prompt-cache identity retains the global offset separately through [`crate::PartitionState`].
    pub fn for_partition(
        &self,
        partition: &crate::PartitionState,
    ) -> Result<Self, ReplicatedTextContractError> {
        let range = partition.global_layers();
        let expected = self
            .layout
            .slice(range.clone())
            .map_err(|error| ReplicatedTextContractError::invalid(error.to_string()))?;
        if &expected != partition.layout() {
            return Err(ReplicatedTextContractError::invalid(
                "partition state layout differs from the selected global interval",
            ));
        }
        let components = self
            .components
            .iter()
            .filter(|component| range.contains(&component.layer))
            .cloned()
            .map(|mut component| {
                component.layer -= range.start;
                component
            })
            .collect::<Vec<_>>();
        let expected_components = (0..partition.layout().len())
            .map(|layer| {
                partition
                    .layout()
                    .components(layer)
                    .expect("validated local state layout contains every layer")
                    .len()
            })
            .sum::<usize>();
        if components.len() != expected_components {
            return Err(ReplicatedTextContractError::invalid(
                "partition state components differ from the selected global interval",
            ));
        }
        Ok(Self {
            layout: partition.layout().clone(),
            access: self.access,
            policy: self.policy.clone(),
            components,
            checkpoint: self.checkpoint,
            rollback: self.rollback,
            reset: self.reset,
            prompt_cache: self.prompt_cache,
            observation_retention: self.observation_retention,
        })
    }

    /// Selects a rank-local interval whose tensor-parallel component shapes were authored by the
    /// validated architecture partition.
    ///
    /// Pipeline ownership must still name the same global layer interval. Tensor-parallel
    /// geometry may narrow fixed dimensions, but it cannot change component roles, dtype,
    /// residency, presence, ordering, or selected physical placement.
    pub fn for_partitioned_geometry(
        &self,
        partition: &crate::PartitionState,
    ) -> Result<Self, ReplicatedTextContractError> {
        let range = partition.global_layers();
        let global = self
            .layout
            .slice(range.clone())
            .map_err(|error| ReplicatedTextContractError::invalid(error.to_string()))?;
        if global.len() != partition.layout().len() {
            return Err(ReplicatedTextContractError::invalid(
                "partition state layer count differs from the selected global interval",
            ));
        }
        let mut components = Vec::new();
        for local_layer in 0..partition.layout().len() {
            let global_components = global
                .components(local_layer)
                .expect("validated selected state contains every local layer");
            let local_components = partition
                .layout()
                .components(local_layer)
                .expect("validated partition state contains every local layer");
            if global_components.len() != local_components.len() {
                return Err(ReplicatedTextContractError::invalid(
                    "partition state component count differs from selected state",
                ));
            }
            let global_layer = range.start + local_layer;
            let selected_components = self
                .components
                .iter()
                .filter(|component| component.layer == global_layer)
                .collect::<Vec<_>>();
            if selected_components.len() != local_components.len() {
                return Err(ReplicatedTextContractError::invalid(
                    "partition state components differ from the selected global interval",
                ));
            }
            for ((global_policy, local_policy), selected) in global_components
                .iter()
                .zip(local_components)
                .zip(selected_components)
            {
                if global_policy.role() != local_policy.role()
                    || global_policy.dtype() != local_policy.dtype()
                    || global_policy.residency() != local_policy.residency()
                    || global_policy.presence() != local_policy.presence()
                    || selected.component != *global_policy
                {
                    return Err(ReplicatedTextContractError::invalid(
                        "partition state component semantics differ from selected state",
                    ));
                }
                components.push(SelectedStateComponentRealization {
                    layer: local_layer,
                    component: local_policy.clone(),
                    placement: selected.placement,
                });
            }
        }
        Ok(Self {
            layout: partition.layout().clone(),
            access: self.access,
            policy: self.policy.clone(),
            components,
            checkpoint: self.checkpoint,
            rollback: self.rollback,
            reset: self.reset,
            prompt_cache: self.prompt_cache,
            observation_retention: self.observation_retention,
        })
    }

    /// Returns the state-access semantics selected for typed traversal.
    pub const fn access(&self) -> ReplicatedTextStateAccess {
        self.access
    }

    /// Returns the selected residency policy.
    pub const fn policy(&self) -> &CacheResidencyPolicy {
        &self.policy
    }

    /// Returns exact selected component realizations in layer/component order.
    pub fn components(&self) -> &[SelectedStateComponentRealization] {
        &self.components
    }

    /// Returns whether state checkpoints are selected.
    pub const fn checkpoint(&self) -> bool {
        self.checkpoint
    }

    /// Returns whether checkpoint rollback is selected.
    pub const fn rollback(&self) -> bool {
        self.rollback
    }

    /// Returns whether complete reset is selected.
    pub const fn reset(&self) -> bool {
        self.reset
    }

    /// Returns whether prompt-cache persistence is selected.
    pub const fn prompt_cache(&self) -> bool {
        self.prompt_cache
    }

    /// Returns whether observation retains every live component.
    pub const fn observation_retention(&self) -> bool {
        self.observation_retention
    }
}

/// Authoritative realization selected before architecture or payload construction.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedReplicatedTextRealization {
    requirements: ReplicatedTextRequirements,
    /// Exact selected execution topology.
    topology: ParallelTopology,
    /// Selected ordinary parameter residency.
    residency: LayerWeightResidency,
    /// Selected exact mutable-state implementation.
    state: SelectedStateRealization,
    /// Exact per-parameter source, executable format, and lowering.
    parameters: Vec<SelectedParameterRealization>,
    /// Required observation facilities admitted by the backend.
    session: SessionCapabilities,
    /// Prompt-cache persistence is selected for this lifecycle.
    prompt_cache: bool,
    /// Exact completion ownership selected for this lifecycle.
    exact_completion: bool,
    grouped_operations: Vec<GroupedOperationRequirement>,
}

impl SelectedReplicatedTextRealization {
    /// Returns the exact architecture/artifact requirements selected together.
    pub const fn requirements(&self) -> &ReplicatedTextRequirements {
        &self.requirements
    }
    /// Returns the exact selected topology.
    pub const fn topology(&self) -> ParallelTopology {
        self.topology
    }
    /// Returns selected weight residency.
    pub const fn residency(&self) -> LayerWeightResidency {
        self.residency
    }
    /// Returns the authoritative selected mutable-state realization.
    pub const fn state(&self) -> &SelectedStateRealization {
        &self.state
    }
    /// Returns exact per-parameter realizations.
    pub fn parameters(&self) -> &[SelectedParameterRealization] {
        &self.parameters
    }
    /// Returns selected session facilities.
    pub const fn session(&self) -> SessionCapabilities {
        self.session
    }
    /// Returns whether prompt-cache persistence was selected.
    pub const fn prompt_cache(&self) -> bool {
        self.prompt_cache
    }
    /// Returns whether exact completion ownership was selected.
    pub const fn exact_completion(&self) -> bool {
        self.exact_completion
    }
    /// Returns selected grouped operation mechanisms.
    pub fn grouped_operations(&self) -> &[GroupedOperationRequirement] {
        &self.grouped_operations
    }
}

/// Complete fail-closed selection diagnostic.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("replicated text realization is unsupported: {issues}", issues = .issues.join("; "))]
pub struct ReplicatedTextSelectionError {
    issues: Vec<String>,
}

impl ReplicatedTextSelectionError {
    /// Every missing semantic or mechanism requirement in stable order.
    pub fn issues(&self) -> &[String] {
        &self.issues
    }
}

/// Deterministically selects one realization without constructing backend payloads.
pub fn select_replicated_text_realization(
    requirements: &ReplicatedTextRequirements,
    request: &ReplicatedTextSelectionRequest,
    capabilities: &BackendMechanismCapabilities,
) -> Result<SelectedReplicatedTextRealization, ReplicatedTextSelectionError> {
    let mut issues = Vec::new();
    if request
        .topology
        .is_some_and(|topology| !topology.is_replicated())
    {
        issues.push("replicated execution topology".into());
    }
    if !capabilities.operators.contains(requirements.operators) {
        issues.extend(
            capabilities
                .operators
                .missing_capability_names(requirements.operators)
                .into_iter()
                .map(|name| format!("neural operation {name}")),
        );
    }
    for operation in &requirements.grouped_operations {
        if !capabilities.grouped_operations.contains(operation) {
            issues.push(format!("grouped operation {operation:?}"));
        }
    }
    let residency_mechanism = match request.residency {
        LayerWeightResidency::FullyResident => WeightResidencyMechanism::Resident,
        LayerWeightResidency::LayerwiseHost(_) => WeightResidencyMechanism::Windowed,
        LayerWeightResidency::DenseDiskStream(_) => WeightResidencyMechanism::DiskStreamed,
    };
    if !capabilities
        .weight_residencies
        .contains(&residency_mechanism)
    {
        issues.push(format!("weight residency {residency_mechanism:?}"));
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
                .components
                .iter()
                .filter(|mechanism| mechanism.layer == layer && mechanism.component == *component)
                .collect::<Vec<_>>();
            let role = component.role().stable_name();
            match matches.as_slice() {
                [mechanism] => match mechanism.placement(&request.state) {
                    Some(placement) if placement_is_compatible(component, &request.state, placement) => {
                        state_components.push(SelectedStateComponentRealization {
                            layer,
                            component: component.clone(),
                            placement,
                        });
                    }
                    Some(placement) => issues.push(format!(
                        "state component {role} at layer {layer} has incompatible {placement:?} placement for {:?} and {:?} residency",
                        request.state,
                        component.residency()
                    )),
                    None => issues.push(format!(
                        "state component {role} at layer {layer} for {:?}",
                        request.state
                    )),
                },
                [] => issues.push(format!(
                    "state component {role} at layer {layer} with shape {:?} and dtype {:?}",
                    component.shape(),
                    component.dtype()
                )),
                _ => issues.push(format!(
                    "unique state component mechanism {role} at layer {layer}"
                )),
            }
        }
    }
    for (supported, name) in [
        (capabilities.state.checkpoint, "state checkpoint"),
        (capabilities.state.rollback, "state rollback"),
        (capabilities.state.reset, "state reset"),
    ] {
        if !supported {
            issues.push(name.into());
        }
    }
    if request.prompt_cache && !capabilities.state.prompt_cache {
        issues.push("state prompt-cache persistence".into());
    }
    if (request.session.output_observation() || request.session.activation_inspection())
        && !capabilities.state.observation_retention
    {
        issues.push("state observation retention".into());
    }
    for (required, supported, name) in [
        (
            request.session.persistent_cache(),
            capabilities.session.persistent_cache(),
            "persistent_cache",
        ),
        (
            request.session.output_observation(),
            capabilities.session.output_observation(),
            "output_observation",
        ),
        (
            request.session.activation_inspection(),
            capabilities.session.activation_inspection(),
            "activation_inspection",
        ),
    ] {
        if required && !supported {
            issues.push(format!("session capability {name}"));
        }
    }
    if request.prompt_cache && !capabilities.prompt_cache {
        issues.push("prompt-cache persistence".into());
    }
    if request.exact_completion && !capabilities.exact_completion {
        issues.push("exact completion ownership".into());
    }

    let mut parameters = Vec::with_capacity(requirements.parameters.len());
    let mut names = BTreeSet::new();
    for parameter in &requirements.parameters {
        if parameter.name.trim().is_empty() || !names.insert(parameter.name.as_str()) {
            issues.push(format!(
                "unique nonempty logical parameter identity {:?}",
                parameter.name
            ));
            continue;
        }
        if !parameter.has_lowering_source() {
            continue;
        }
        let candidate = match request.quantization {
            Some(request) => match parameter.transform_target(request) {
                Ok(Some(target)) => Some((target.executable(), target.descriptor().clone())),
                Ok(None) => Some((
                    parameter.native_executable,
                    parameter
                        .lowering_descriptor(parameter.native_executable)
                        .expect("validated parameter forms native descriptor"),
                )),
                Err(error) => {
                    issues.push(error.to_string());
                    None
                }
            },
            None => Some((
                parameter.native_executable,
                parameter
                    .lowering_descriptor(parameter.native_executable)
                    .expect("validated parameter forms native descriptor"),
            )),
        };
        let Some((executable, descriptor)) = candidate else {
            issues.push(format!(
                "architecture transform {:?} for {:?}",
                request.quantization, parameter.name
            ));
            continue;
        };
        let Some(lowering) = capabilities
            .weight_lowerings
            .iter()
            .find(|lowering| lowering.descriptor == descriptor)
        else {
            issues.push(format!(
                "weight lowering {:?} -> {:?} for {:?}",
                parameter.source_encoding, executable, parameter.name
            ));
            continue;
        };
        parameters.push(SelectedParameterRealization {
            name: parameter.name.clone(),
            sources: parameter.sources.clone(),
            physical_sources: parameter.physical_sources.clone(),
            source_encoding: parameter
                .source_encoding
                .clone()
                .expect("physical parameter has a source encoding"),
            executable,
            lowering: match (&parameter.presence, lowering.kind) {
                (
                    ReplicatedTextParameterPresence::Derived { .. },
                    WeightLoweringKind::Transform,
                ) => WeightLoweringKind::DerivedTransform,
                (ReplicatedTextParameterPresence::Derived { .. }, _) => WeightLoweringKind::Derived,
                (_, kind) => kind,
            },
        });
    }
    if !issues.is_empty() {
        return Err(ReplicatedTextSelectionError { issues });
    }
    Ok(SelectedReplicatedTextRealization {
        requirements: requirements.clone(),
        topology: request
            .topology
            .unwrap_or_else(|| ParallelTopology::new(1, 1, 1, 1).expect("replicated topology")),
        residency: request.residency,
        state: SelectedStateRealization {
            layout: requirements.state_layout.clone(),
            access: requirements.state_access,
            policy: request.state.clone(),
            components: state_components,
            checkpoint: true,
            rollback: true,
            reset: true,
            prompt_cache: request.prompt_cache,
            observation_retention: request.session.output_observation()
                || request.session.activation_inspection(),
        },
        parameters,
        session: request.session,
        prompt_cache: request.prompt_cache,
        exact_completion: request.exact_completion,
        grouped_operations: requirements.grouped_operations.clone(),
    })
}

fn placement_is_compatible(
    component: &StateComponentPolicy,
    policy: &CacheResidencyPolicy,
    placement: StateComponentPlacement,
) -> bool {
    use eredu_core::cache::StateResidencyClass;

    let expected = match (policy, component.residency()) {
        (CacheResidencyPolicy::Device, _) => StateComponentPlacement::Device,
        (CacheResidencyPolicy::Paged(_), StateResidencyClass::SealablePaged) => {
            StateComponentPlacement::Paged
        }
        (
            CacheResidencyPolicy::Paged(_),
            StateResidencyClass::AlwaysDeviceMutable | StateResidencyClass::LayerScopedOffloadable,
        ) => StateComponentPlacement::Device,
    };
    placement == expected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
        ArchitectureMergeDestination, ArchitectureParameterDescription, ArchitecturePartition,
        ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule, DenseDiskStreamLoadOptions,
        ExecutionGroupSpec, ExecutionUnitLayout, LayerwiseLoadOptions, MemberSharding,
        NoAuxiliaryBoundarySchema, OwnedParameterGroupSpec, ParameterGroupSpec,
        ParameterMemberSpec, ParameterRole, PartitionOwnership, StateLayout,
    };
    use eredu_checkpoint::{AffineQuantization, StoredDtype};
    use eredu_core::{
        cache::{
            LayerCachePolicy, MutableStateResidency, StateTensorDimension, StateTensorDtype,
            StateTensorPolicy, StateTensorRole,
        },
        AttentionPolicy, LayerSchedule,
    };

    fn paged_state() -> CacheResidencyPolicy {
        CacheResidencyPolicy::Paged(
            crate::PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        )
    }

    fn physical_source(name: &str) -> ReplicatedTextPhysicalSource {
        ReplicatedTextPhysicalSource::new(name, "/checkpoint/model.safetensors", name).unwrap()
    }

    fn requirements() -> ReplicatedTextRequirements {
        let graph =
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
        let execution_units = ExecutionUnitLayout::new(&graph, [1]).unwrap();
        ReplicatedTextRequirements::new(
            "test.replicated-text",
            NeuralOperatorCapabilities::EXP,
            graph,
            execution_units,
            vec![ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: ArchitectureGroupKind::Decoder,
                first_owner_static_roles: vec!["embedding".into()],
                last_owner_static_roles: vec!["output".into()],
                merge_destination: ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            }],
            StateLayout::new(
                LayerSchedule::new(
                    1,
                    vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap()],
                )
                .unwrap(),
            )
            .unwrap(),
            ReplicatedTextStateAccess::KeyValue,
            vec![
                ReplicatedTextParameterRequirement::new(
                    "model.layers.0.mlp.weight",
                    vec!["blk.0.ffn.weight".into()],
                    vec![physical_source("blk.0.ffn.weight")],
                    Vec::new(),
                    Some(SourceTensorEncoding::Safetensors(StoredDtype::F16)),
                    Some(vec![64, 64]),
                    vec![64, 64],
                    LinearFormat::Dense,
                    ReplicatedTextParameterRole::LinearWeight,
                    ReplicatedTextParameterOwner::ExecutionUnit {
                        group: "decoder".into(),
                        unit: 0,
                    },
                    ReplicatedTextParameterPresence::Required,
                    ParameterTransformConstraint::Linear { packed_axis: 1 },
                )
                .unwrap(),
                ReplicatedTextParameterRequirement::new(
                    "model.layers.0.mlp.bias",
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    None,
                    None,
                    vec![64],
                    LinearFormat::Dense,
                    ReplicatedTextParameterRole::LinearBias,
                    ReplicatedTextParameterOwner::ExecutionUnit {
                        group: "decoder".into(),
                        unit: 0,
                    },
                    ReplicatedTextParameterPresence::OptionalAbsent,
                    ParameterTransformConstraint::None,
                )
                .unwrap(),
                ReplicatedTextParameterRequirement::new(
                    "model.layers.0.norm.weight",
                    vec!["blk.0.norm.weight".into()],
                    vec![physical_source("blk.0.norm.weight")],
                    Vec::new(),
                    Some(SourceTensorEncoding::Safetensors(StoredDtype::F16)),
                    Some(vec![64]),
                    vec![64],
                    LinearFormat::Dense,
                    ReplicatedTextParameterRole::Normalization,
                    ReplicatedTextParameterOwner::ExecutionUnit {
                        group: "decoder".into(),
                        unit: 0,
                    },
                    ReplicatedTextParameterPresence::Required,
                    ParameterTransformConstraint::None,
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn requirements_reject_unit_layout_from_an_equally_sized_different_graph() {
        let baseline = requirements();
        let other_graph =
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("mutated")], "mutated").unwrap();
        let other_layout = ExecutionUnitLayout::new(&other_graph, [1]).unwrap();
        let error = ReplicatedTextRequirements::new(
            baseline.architecture_identity.clone(),
            baseline.operators,
            baseline.execution_graph.clone(),
            other_layout,
            baseline.group_transports.clone(),
            baseline.state_layout.clone(),
            baseline.state_access,
            baseline.parameters.clone(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("layout group identities differ"));
    }

    #[test]
    fn parameter_requirement_preserves_every_admitted_alias() {
        let requirement = ReplicatedTextParameterRequirement::new(
            "model.layers.0.mlp.weight",
            vec!["released.layers.0.mlp.weight".into()],
            vec![physical_source("released.layers.0.mlp.weight")],
            vec![
                "legacy.layers.0.mlp.weight".into(),
                "vendor.layers.0.mlp.weight".into(),
            ],
            Some(SourceTensorEncoding::Safetensors(StoredDtype::F16)),
            Some(vec![64, 64]),
            vec![64, 64],
            LinearFormat::Dense,
            ReplicatedTextParameterRole::LinearWeight,
            ReplicatedTextParameterOwner::ExecutionUnit {
                group: "decoder".into(),
                unit: 0,
            },
            ReplicatedTextParameterPresence::Required,
            ParameterTransformConstraint::Linear { packed_axis: 1 },
        )
        .unwrap();

        assert_eq!(
            requirement.aliases(),
            ["legacy.layers.0.mlp.weight", "vendor.layers.0.mlp.weight"]
        );
        assert_eq!(requirement.sources(), ["released.layers.0.mlp.weight"]);

        let absent_bias = ReplicatedTextParameterRequirement::new(
            "model.layers.0.mlp.bias",
            Vec::new(),
            Vec::new(),
            vec!["released.layers.0.mlp.bias".into()],
            None,
            None,
            vec![64],
            LinearFormat::Dense,
            ReplicatedTextParameterRole::LinearBias,
            ReplicatedTextParameterOwner::ExecutionUnit {
                group: "decoder".into(),
                unit: 0,
            },
            ReplicatedTextParameterPresence::OptionalAbsent,
            ParameterTransformConstraint::None,
        )
        .unwrap();
        assert_eq!(
            absent_bias.presence(),
            &ReplicatedTextParameterPresence::OptionalAbsent
        );
        assert!(absent_bias.sources().is_empty());
        assert_eq!(
            absent_bias.transform_constraint(),
            ParameterTransformConstraint::None
        );
    }

    #[test]
    fn scalar_parameter_requirement_preserves_rank_zero_geometry() {
        let requirement = ReplicatedTextParameterRequirement::new(
            "model.audio_tower.input_max",
            vec!["model.audio_tower.input_max".into()],
            vec![physical_source("model.audio_tower.input_max")],
            Vec::new(),
            Some(SourceTensorEncoding::Safetensors(StoredDtype::F32)),
            Some(Vec::new()),
            Vec::new(),
            LinearFormat::Dense,
            ReplicatedTextParameterRole::Other,
            ReplicatedTextParameterOwner::StaticRole("audio".into()),
            ReplicatedTextParameterPresence::Required,
            ParameterTransformConstraint::None,
        )
        .unwrap();

        let descriptor = requirement
            .lowering_descriptor(LinearFormat::Dense)
            .unwrap();
        assert!(descriptor.physical_shape().is_empty());
        assert!(descriptor.logical_shape().is_empty());
        assert_eq!(descriptor.packed_axis(), None);
    }

    #[test]
    fn physical_provenance_distinguishes_outputs_from_one_sharded_tensor() {
        let shard = "/checkpoint/model-00002-of-00003.gguf";
        let weight = ReplicatedTextPhysicalSource::new(
            "blk.0.ffn_gate.weight",
            shard,
            "blk.0.ffn_gate.weight",
        )
        .unwrap();
        let scales = ReplicatedTextPhysicalSource::new(
            "blk.0.ffn_gate.weight",
            shard,
            "blk.0.ffn_gate.scales",
        )
        .unwrap();
        assert_eq!(weight.tensor(), scales.tensor());
        assert_eq!(weight.shard(), scales.shard());
        assert_ne!(weight.output(), scales.output());
    }

    fn capabilities() -> BackendMechanismCapabilities {
        let source = SourceTensorEncoding::Safetensors(StoredDtype::F16);
        let requirements = requirements();
        let state = StateMechanismCapabilities::new(
            (0..requirements.state_layout().len()).flat_map(|layer| {
                requirements
                    .state_layout()
                    .components(layer)
                    .unwrap()
                    .iter()
                    .cloned()
                    .map(move |component| {
                        let paged = match component.role() {
                            eredu_core::cache::StateComponentRole::AttentionKeys
                            | eredu_core::cache::StateComponentRole::AttentionValues
                            | eredu_core::cache::StateComponentRole::CompressedLatent
                            | eredu_core::cache::StateComponentRole::RotaryKeys => {
                                StateComponentPlacement::Paged
                            }
                            eredu_core::cache::StateComponentRole::Fixed(_) => {
                                StateComponentPlacement::Device
                            }
                        };
                        StateComponentMechanism::new(
                            layer,
                            component,
                            Some(StateComponentPlacement::Device),
                            Some(paged),
                        )
                    })
            }),
        )
        .with_transactions(true, true)
        .with_reset(true)
        .with_prompt_cache(true)
        .with_observation_retention(true);
        BackendMechanismCapabilities::new(
            NeuralOperatorCapabilities::EXP,
            vec![
                WeightLoweringCapability::new(
                    WeightLoweringDescriptor::new(
                        source.clone(),
                        LinearFormat::Dense,
                        vec![64, 64],
                        vec![64, 64],
                        Some(1),
                    )
                    .unwrap(),
                    WeightLoweringKind::Direct,
                ),
                WeightLoweringCapability::new(
                    WeightLoweringDescriptor::new(
                        source,
                        LinearFormat::Affine(AffineQuantization::new(64, 4).unwrap()),
                        vec![64, 64],
                        vec![64, 64],
                        Some(1),
                    )
                    .unwrap(),
                    WeightLoweringKind::Transform,
                ),
                WeightLoweringCapability::new(
                    WeightLoweringDescriptor::new(
                        SourceTensorEncoding::Safetensors(StoredDtype::F16),
                        LinearFormat::Dense,
                        vec![64],
                        vec![64],
                        None,
                    )
                    .unwrap(),
                    WeightLoweringKind::Direct,
                ),
            ],
            vec![
                WeightResidencyMechanism::Resident,
                WeightResidencyMechanism::Windowed,
                WeightResidencyMechanism::DiskStreamed,
            ],
            state,
        )
        .with_session(SessionCapabilities::new(true, true, true))
        .with_prompt_cache(true)
        .with_exact_completion(true)
    }

    fn request(residency: LayerWeightResidency) -> ReplicatedTextSelectionRequest {
        ReplicatedTextSelectionRequest::new(residency, paged_state())
            .with_session(SessionCapabilities::new(true, true, true))
            .with_prompt_cache(true)
            .with_exact_completion(true)
    }

    #[test]
    fn complete_requirements_are_invariant_across_all_caller_policy_dimensions() {
        let baseline = requirements();
        let disk = DenseDiskStreamLoadOptions::new(4096, 8192, 2, 1).unwrap();
        let requests = [
            ReplicatedTextSelectionRequest::new(
                LayerWeightResidency::FullyResident,
                CacheResidencyPolicy::Device,
            ),
            ReplicatedTextSelectionRequest::new(
                LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::default()),
                paged_state(),
            )
            .with_topology(ParallelTopology::new(2, 1, 1, 1).unwrap())
            .with_quantization(QuantizationRequest::Affine {
                group_size: 64,
                bits: 4,
            })
            .with_session(SessionCapabilities::new(true, true, true))
            .with_prompt_cache(true)
            .with_exact_completion(true),
            ReplicatedTextSelectionRequest::new(
                LayerWeightResidency::DenseDiskStream(disk),
                CacheResidencyPolicy::Device,
            )
            .with_quantization(QuantizationRequest::MxFp4),
        ];

        for _request in &requests {
            assert_eq!(requirements(), baseline);
        }
        assert_eq!(requests[0].state(), &CacheResidencyPolicy::Device);
        assert!(matches!(
            requests[1].residency(),
            LayerWeightResidency::LayerwiseHost(_)
        ));
        assert_eq!(requests[1].topology().unwrap().tensor(), 2);
        assert_eq!(
            requests[1].quantization(),
            Some(QuantizationRequest::Affine {
                group_size: 64,
                bits: 4,
            })
        );
        assert!(requests[1].prompt_cache());
        assert!(requests[1].exact_completion());
        assert!(requests[1].session().activation_inspection());
        assert_eq!(
            requests[2].residency(),
            LayerWeightResidency::DenseDiskStream(disk)
        );
        assert_eq!(requests[2].quantization(), Some(QuantizationRequest::MxFp4));
    }

    #[test]
    fn partitioned_tasks_keep_encoded_companions_atomic_and_reject_split_groups() {
        let request = request(LayerWeightResidency::FullyResident).with_quantization(
            QuantizationRequest::Affine {
                group_size: 64,
                bits: 4,
            },
        );
        let selected =
            select_replicated_text_realization(&requirements(), &request, &capabilities()).unwrap();
        let graph = selected.requirements().execution_graph().clone();
        let layout = selected.requirements().execution_units().clone();
        let format = eredu_nn::LinearFormatSpec::affine(
            LinearFormat::Affine(AffineQuantization::new(64, 4).unwrap()),
            eredu_nn::ParameterSpec::trainable("model.layers.0.mlp.scales").unwrap(),
            eredu_nn::ParameterSpec::trainable("model.layers.0.mlp.biases").unwrap(),
        )
        .unwrap();
        let [physical] = crate::expand_linear_format_parameter_groups(
            vec![ParameterGroupSpec::new(
                "mlp",
                ParameterRole::FeedForwardIntermediate,
                [ParameterMemberSpec::new(
                    "model.layers.0.mlp.weight",
                    vec![64, 64],
                    MemberSharding::Replicated,
                )],
            )
            .unwrap()],
            |_| Ok(Some(format.clone())),
        )
        .unwrap()
        .try_into()
        .unwrap();
        let norm = ParameterGroupSpec::new(
            "norm",
            ParameterRole::Replicated,
            [ParameterMemberSpec::new(
                "model.layers.0.norm.weight",
                vec![64],
                MemberSharding::Replicated,
            )],
        )
        .unwrap();
        let owner = ParameterGroupOwner::execution_unit(layout.group_id(0).unwrap().clone(), 0);
        let description = ArchitectureParameterDescription::new(
            &graph,
            &layout,
            [physical.clone(), norm.clone()],
            [
                OwnedParameterGroupSpec::new(owner.clone(), physical.clone()),
                OwnedParameterGroupSpec::new(owner.clone(), norm.clone()),
            ],
        )
        .unwrap();
        let ownership =
            PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap();
        let state = selected.requirements().state_layout();
        let state_plan =
            ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
                0,
                0..state.len(),
            )]);
        let partition = ArchitecturePartition::from_description(
            &description,
            [(layout.group_id(0).unwrap().as_str(), 0..1)],
            ownership.clone(),
            state,
            &state_plan,
            (),
            NoAuxiliaryBoundarySchema::new(64),
        )
        .unwrap();
        let tasks =
            partitioned_replicated_text_materialization_tasks(&selected, &description, &partition)
                .unwrap();
        let task = tasks
            .iter()
            .find(|task| task.name() == "model.layers.0.mlp.weight")
            .unwrap();
        assert_eq!(task.output_companions().len(), 2);

        let members = physical.members();
        let primary = ParameterGroupSpec::new(
            "primary",
            ParameterRole::FeedForwardIntermediate,
            [members[0].clone()],
        )
        .unwrap();
        let companions = ParameterGroupSpec::new(
            "companions",
            ParameterRole::FeedForwardIntermediate,
            members[1..].to_vec(),
        )
        .unwrap();
        let malformed = ArchitectureParameterDescription::new(
            &graph,
            &layout,
            [primary.clone(), companions.clone(), norm.clone()],
            [
                OwnedParameterGroupSpec::new(owner.clone(), primary),
                OwnedParameterGroupSpec::new(owner.clone(), companions),
                OwnedParameterGroupSpec::new(owner, norm),
            ],
        )
        .unwrap();
        let malformed_partition = ArchitecturePartition::from_description(
            &malformed,
            [(layout.group_id(0).unwrap().as_str(), 0..1)],
            ownership,
            state,
            &state_plan,
            (),
            NoAuxiliaryBoundarySchema::new(64),
        )
        .unwrap();
        let error = partitioned_replicated_text_materialization_tasks(
            &selected,
            &malformed,
            &malformed_partition,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("outside its atomic parameter group"));
    }

    #[test]
    fn partitioned_tasks_require_one_canonical_or_admitted_alias_topology_target() {
        let mut requirements = requirements();
        requirements.parameters[0].aliases = vec!["architecture.mlp.weight".into()];
        let selected = select_replicated_text_realization(
            &requirements,
            &request(LayerWeightResidency::FullyResident),
            &capabilities(),
        )
        .unwrap();
        let graph = selected.requirements().execution_graph().clone();
        let layout = selected.requirements().execution_units().clone();
        let owner = ParameterGroupOwner::execution_unit(layout.group_id(0).unwrap().clone(), 0);
        let ownership =
            PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap();
        let state = selected.requirements().state_layout();
        let state_plan =
            ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
                0,
                0..state.len(),
            )]);

        let project = |primary_targets: &[&str]| {
            let mut groups = primary_targets
                .iter()
                .enumerate()
                .map(|(index, target)| {
                    ParameterGroupSpec::new(
                        format!("mlp-{index}"),
                        ParameterRole::FeedForwardIntermediate,
                        [ParameterMemberSpec::new(
                            *target,
                            vec![64, 64],
                            MemberSharding::Replicated,
                        )],
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            groups.push(
                ParameterGroupSpec::new(
                    "norm",
                    ParameterRole::Replicated,
                    [ParameterMemberSpec::new(
                        "model.layers.0.norm.weight",
                        vec![64],
                        MemberSharding::Replicated,
                    )],
                )
                .unwrap(),
            );
            let description = ArchitectureParameterDescription::new(
                &graph,
                &layout,
                groups.clone(),
                groups
                    .into_iter()
                    .map(|group| OwnedParameterGroupSpec::new(owner.clone(), group)),
            )
            .unwrap();
            let partition = ArchitecturePartition::from_description(
                &description,
                [(layout.group_id(0).unwrap().as_str(), 0..1)],
                ownership.clone(),
                state,
                &state_plan,
                (),
                NoAuxiliaryBoundarySchema::new(64),
            )
            .unwrap();
            partitioned_replicated_text_materialization_tasks(&selected, &description, &partition)
        };

        let canonical = project(&["model.layers.0.mlp.weight"]).unwrap();
        assert!(canonical
            .iter()
            .any(|task| task.name() == "model.layers.0.mlp.weight"));

        let aliased = project(&["architecture.mlp.weight"]).unwrap();
        let task = aliased
            .iter()
            .find(|task| task.name() == "model.layers.0.mlp.weight")
            .unwrap();
        assert_eq!(task.aliases(), ["architecture.mlp.weight"]);

        let error = project(&["model.layers.0.mlp.weight", "architecture.mlp.weight"]).unwrap_err();
        assert!(error
            .to_string()
            .contains("resolves to 2 architecture topology targets"));
    }

    #[test]
    fn selection_is_deterministic_and_keeps_source_format_distinct() {
        let disk = DenseDiskStreamLoadOptions::new(1234, 5678, 3, 2).unwrap();
        let request = request(LayerWeightResidency::DenseDiskStream(disk)).with_quantization(
            QuantizationRequest::Affine {
                group_size: 64,
                bits: 4,
            },
        );
        let left =
            select_replicated_text_realization(&requirements(), &request, &capabilities()).unwrap();
        let right =
            select_replicated_text_realization(&requirements(), &request, &capabilities()).unwrap();
        assert_eq!(left, right);
        assert_eq!(
            left.residency(),
            LayerWeightResidency::DenseDiskStream(disk)
        );
        assert_eq!(left.state().policy(), &paged_state());
        assert_eq!(left.state().layout(), requirements().state_layout());
        assert_eq!(left.parameters().len(), 2);
        assert_eq!(requirements().parameters().len(), 3);
        assert!(matches!(
            requirements().parameters()[1].presence(),
            ReplicatedTextParameterPresence::OptionalAbsent
        ));
        assert!(matches!(
            requirements().parameters()[2].role(),
            ReplicatedTextParameterRole::Normalization
        ));
        assert_eq!(requirements().parameters()[2].logical_shape(), [64]);
        assert_eq!(
            requirements().parameters()[2].transform_constraint(),
            ParameterTransformConstraint::None
        );
        assert_eq!(
            left.parameters()[0].lowering(),
            WeightLoweringKind::Transform
        );
        assert_ne!(
            format!("{:?}", left.parameters()[0].source_encoding()),
            format!("{:?}", left.parameters()[0].executable())
        );
    }

    #[test]
    fn exact_tasks_are_the_authority_for_direct_derived_and_transform_sources() {
        use eredu_checkpoint::recipe::{DerivedWeightRecipe, RecipeDtype, RecipeMetadata};

        let direct = select_replicated_text_realization(
            &requirements(),
            &request(LayerWeightResidency::FullyResident),
            &capabilities(),
        )
        .unwrap();
        let direct_tasks = replicated_text_materialization_tasks(&direct).unwrap();
        assert_eq!(
            direct_tasks[0].source_recipe().unwrap(),
            DerivedWeightRecipe::source(
                "blk.0.ffn.weight",
                eredu_checkpoint::store::TensorSelection::Full,
            )
        );

        let recipe = DerivedWeightRecipe::source(
            "blk.0.ffn.weight",
            eredu_checkpoint::store::TensorSelection::Full,
        );
        let outputs = BTreeMap::from([(
            "model.layers.0.mlp.weight".into(),
            RecipeMetadata {
                shape: vec![64, 64],
                dtype: RecipeDtype::F16,
                byte_len: 64 * 64 * 2,
            },
        )]);
        let derived_requirements = requirements()
            .with_derived_recipes(
                BTreeMap::from([("model.layers.0.mlp.weight".into(), recipe.clone())]),
                outputs,
            )
            .unwrap();
        let derived = select_replicated_text_realization(
            &derived_requirements,
            &request(LayerWeightResidency::FullyResident),
            &capabilities(),
        )
        .unwrap();
        let derived_tasks = replicated_text_materialization_tasks(&derived).unwrap();
        assert_eq!(derived_tasks[0].lowering(), WeightLoweringKind::Derived);
        assert_eq!(derived_tasks[0].source_recipe().unwrap(), recipe);

        let transformed = select_replicated_text_realization(
            &derived_requirements,
            &request(LayerWeightResidency::FullyResident).with_quantization(
                QuantizationRequest::Affine {
                    group_size: 64,
                    bits: 4,
                },
            ),
            &capabilities(),
        )
        .unwrap();
        let transformed_tasks = replicated_text_materialization_tasks(&transformed).unwrap();
        assert_eq!(
            transformed_tasks[0].lowering(),
            WeightLoweringKind::DerivedTransform
        );
        assert_eq!(transformed_tasks[0].source_recipe().unwrap(), recipe);

        // These corruptions fail while projecting the cold exact plan; no
        // backend mechanism or checkpoint payload is available to perform work.
        let mut corrupt_direct = direct_tasks[0].clone();
        corrupt_direct.sources.push("unselected.weight".into());
        assert!(corrupt_direct.source_recipe().is_err());
        let mut corrupt_kind = derived_tasks[0].clone();
        corrupt_kind.lowering = WeightLoweringKind::Direct;
        assert!(corrupt_kind.source_recipe().is_err());
        let mut corrupt_recipe = derived_tasks[0].clone();
        corrupt_recipe.derived_recipe = Some(DerivedWeightRecipe::source(
            "unselected.weight",
            eredu_checkpoint::store::TensorSelection::Full,
        ));
        assert!(corrupt_recipe.source_recipe().is_err());
    }

    #[test]
    fn selection_reports_all_missing_mechanisms_together() {
        let capabilities = BackendMechanismCapabilities::new(
            NeuralOperatorCapabilities::NONE,
            Vec::new(),
            Vec::new(),
            StateMechanismCapabilities::new(Vec::new()),
        );
        let error = select_replicated_text_realization(
            &requirements(),
            &request(LayerWeightResidency::LayerwiseHost(
                LayerwiseLoadOptions::default(),
            )),
            &capabilities,
        )
        .unwrap_err();
        assert!(error.issues().len() >= 7, "{:?}", error.issues());
        assert!(error.issues().iter().any(|issue| issue.contains("exp")));
        assert!(error
            .issues()
            .iter()
            .any(|issue| issue.contains("weight lowering")));
    }

    #[test]
    fn selection_rejects_paged_fixed_component_placement_even_when_reported() {
        let fixed = StateTensorPolicy::new(
            StateTensorRole::Recurrent,
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::fixed(8).unwrap(),
            ],
            StateTensorDtype::Float32,
            MutableStateResidency::LayerScopedOffloadable,
        )
        .unwrap();
        let mut requirements = requirements();
        requirements.state_layout = StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value_with_fixed_state(
                    AttentionPolicy::Full,
                    1,
                    8,
                    vec![fixed],
                )
                .unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
        requirements.state_access = ReplicatedTextStateAccess::AttentionWithFixed;
        let mut capabilities = capabilities();
        capabilities.state.components = (0..requirements.state_layout.len())
            .flat_map(|layer| {
                requirements
                    .state_layout
                    .components(layer)
                    .unwrap()
                    .iter()
                    .cloned()
                    .map(move |component| {
                        StateComponentMechanism::new(
                            layer,
                            component,
                            Some(StateComponentPlacement::Device),
                            Some(StateComponentPlacement::Paged),
                        )
                    })
            })
            .collect();

        let error = select_replicated_text_realization(
            &requirements,
            &request(LayerWeightResidency::FullyResident),
            &capabilities,
        )
        .unwrap_err();
        assert!(error
            .issues()
            .iter()
            .any(|issue| issue.contains("incompatible Paged placement")));
    }

    #[test]
    fn requirements_reject_state_layout_and_access_profile_mismatch() {
        let fixed = StateTensorPolicy::new(
            StateTensorRole::Recurrent,
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::fixed(8).unwrap(),
            ],
            StateTensorDtype::Float32,
            MutableStateResidency::LayerScopedOffloadable,
        )
        .unwrap();
        let layout = StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value_with_fixed_state(
                    AttentionPolicy::Full,
                    1,
                    8,
                    vec![fixed],
                )
                .unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
        let base = requirements();
        let error = ReplicatedTextRequirements::new(
            base.architecture_identity,
            base.operators,
            base.execution_graph,
            base.execution_units,
            base.group_transports,
            layout,
            ReplicatedTextStateAccess::KeyValue,
            base.parameters,
        )
        .unwrap_err();
        assert!(error.message().contains("does not match component roles"));
    }

    #[test]
    fn transform_selection_rejects_incompatible_exact_geometry() {
        for quantization in [
            QuantizationRequest::Affine {
                group_size: 96,
                bits: 4,
            },
            QuantizationRequest::Affine {
                group_size: 256,
                bits: 4,
            },
            QuantizationRequest::Affine {
                group_size: 0,
                bits: 4,
            },
            QuantizationRequest::Affine {
                group_size: u32::MAX,
                bits: 4,
            },
            QuantizationRequest::Affine {
                group_size: 32,
                bits: 0,
            },
            QuantizationRequest::Affine {
                group_size: 32,
                bits: 7,
            },
        ] {
            let error = select_replicated_text_realization(
                &requirements(),
                &request(LayerWeightResidency::FullyResident).with_quantization(quantization),
                &capabilities(),
            )
            .unwrap_err();
            assert!(error
                .issues()
                .iter()
                .any(|issue| issue.contains("invalid replicated text contract")));
        }

        let mut indivisible = requirements();
        indivisible.parameters[0].logical_shape = vec![64, 48];
        let error = select_replicated_text_realization(
            &indivisible,
            &request(LayerWeightResidency::FullyResident)
                .with_quantization(QuantizationRequest::MxFp4),
            &capabilities(),
        )
        .unwrap_err();
        assert!(error
            .issues()
            .iter()
            .any(|issue| issue.contains("MXFP4 packed extent 48")));
    }

    #[test]
    fn exact_source_and_physical_geometry_fail_before_construction_or_payload() {
        for mutate in [
            |requirement: &mut ReplicatedTextParameterRequirement| {
                requirement.source_encoding =
                    Some(SourceTensorEncoding::Safetensors(StoredDtype::U8));
            },
            |requirement: &mut ReplicatedTextParameterRequirement| {
                requirement.physical_shape = Some(vec![64, 32]);
            },
        ] {
            let mut requirements = requirements();
            mutate(&mut requirements.parameters[0]);
            let selected = select_replicated_text_realization(
                &requirements,
                &request(LayerWeightResidency::FullyResident),
                &capabilities(),
            );
            let error = selected.unwrap_err();
            assert!(error
                .issues()
                .iter()
                .any(|issue| issue.contains("weight lowering")));
        }
    }

    #[test]
    fn missing_tensor_parallel_grouped_partial_fails_before_construction_or_forward() {
        let requirements = requirements().with_grouped_operations([
            GroupedOperationRequirement::GatedProduct,
            GroupedOperationRequirement::GatedProductTensorParallelPartial,
        ]);
        let capabilities =
            capabilities().with_grouped_operations([GroupedOperationRequirement::GatedProduct]);
        let selected = select_replicated_text_realization(
            &requirements,
            &request(LayerWeightResidency::FullyResident),
            &capabilities,
        );
        let error = selected.unwrap_err();
        assert!(error
            .issues()
            .iter()
            .any(|issue| { issue.contains("GatedProductTensorParallelPartial") }));
    }
}
