//! Selection contracts for replicated text architectures.

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use eredu_checkpoint::{LinearFormat, SourceTensorEncoding};
use eredu_core::{ParallelTopology, QuantizationRequest, SessionCapabilities};
use eredu_nn::{NeuralBackend, NeuralOperatorCapabilities};

use crate::{
    ArchitectureGroupTransport, CacheResidencyPolicy, ExecutionGraph, ExecutionUnitLayout,
    LayerWeightResidency, LayeredArchitecture, RuntimeState, StateLayout,
};

/// Statically dispatched text-input seam for an ordinary layered decoder.
///
/// Hybrid, routed, composite, partitioned, prediction, and realtime execution
/// use separate extension contracts rather than adding requirements here.
pub trait ReplicatedTextArchitecture<B, S>: LayeredArchitecture<B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Forms the architecture-owned borrowed input for one text pass.
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a>;
}

/// Backend implementation route for one source-to-executable weight lowering.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum WeightLoweringKind {
    /// The admitted source encoding is retained by the executable operator.
    Direct,
    /// Payload materialization performs an admitted transformation.
    Transform,
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
        if physical_shape.is_empty()
            || physical_shape.contains(&0)
            || logical_shape.is_empty()
            || logical_shape.contains(&0)
            || physical_shape.len() != logical_shape.len()
        {
            return Err(ReplicatedTextContractError::invalid(
                "weight lowering requires positive physical and logical shapes of equal rank",
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

/// Mutable-state residency mechanism implemented by a backend.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum StateResidencyMechanism {
    /// State tensors remain on the execution device.
    Device,
    /// State blocks use bounded paged storage.
    Paged,
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
        let has_source = presence.has_physical_source();
        if has_source
            != (!sources.is_empty() && source_encoding.is_some() && physical_shape.is_some())
        {
            return Err(ReplicatedTextContractError::invalid(format!(
                "logical parameter {name:?} has inconsistent source presence"
            )));
        }
        let provenance_required =
            has_source || matches!(presence, ReplicatedTextParameterPresence::Derived { .. });
        if provenance_required != !physical_sources.is_empty() {
            return Err(ReplicatedTextContractError::invalid(format!(
                "logical parameter {name:?} has inconsistent physical provenance"
            )));
        }
        if has_source
            && physical_sources
                .iter()
                .any(|source| !sources.iter().any(|name| name == source.tensor()))
        {
            return Err(ReplicatedTextContractError::invalid(format!(
                "logical parameter {name:?} has provenance outside its selected sources"
            )));
        }
        if physical_shape
            .as_ref()
            .is_some_and(|shape| shape.is_empty() || shape.contains(&0))
        {
            return Err(ReplicatedTextContractError::invalid(format!(
                "logical parameter {name:?} has an invalid physical shape"
            )));
        }
        if logical_shape.is_empty() || logical_shape.contains(&0) {
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
        let packed_axis = packed_axis.or_else(|| {
            (self.role == ReplicatedTextParameterRole::Embedding
                && executable != LinearFormat::Dense)
                .then(|| self.logical_shape.len() - 1)
        });
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
            self.logical_shape.clone(),
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
    /// Canonical logical parameter requirements.
    parameters: Vec<ReplicatedTextParameterRequirement>,
    grouped_operations: Vec<GroupedOperationRequirement>,
}

impl ReplicatedTextRequirements {
    /// Creates exact requirements from architecture and admitted-artifact facts only.
    pub fn new(
        operators: NeuralOperatorCapabilities,
        execution_graph: ExecutionGraph,
        execution_units: ExecutionUnitLayout,
        group_transports: Vec<ArchitectureGroupTransport>,
        state_layout: StateLayout,
        parameters: Vec<ReplicatedTextParameterRequirement>,
    ) -> Result<Self, ReplicatedTextContractError> {
        if group_transports.len() != execution_graph.groups().len() {
            return Err(ReplicatedTextContractError::invalid(format!(
                "{} group transports do not match {} execution groups",
                group_transports.len(),
                execution_graph.groups().len()
            )));
        }
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
            operators,
            execution_graph,
            execution_units,
            group_transports,
            state_layout,
            parameters,
            grouped_operations: Vec::new(),
        })
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
    /// Returns canonical logical parameter requirements.
    pub fn parameters(&self) -> &[ReplicatedTextParameterRequirement] {
        &self.parameters
    }
    /// Returns exact grouped operations required before construction.
    pub fn grouped_operations(&self) -> &[GroupedOperationRequirement] {
        &self.grouped_operations
    }
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

/// Family- and execution-class-neutral backend mechanism report.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BackendMechanismCapabilities {
    /// Optional neural operations implemented by the backend.
    operators: NeuralOperatorCapabilities,
    /// Exact admitted source-to-executable lowerings.
    weight_lowerings: Vec<WeightLoweringCapability>,
    /// Ordinary parameter residency mechanisms.
    weight_residencies: Vec<WeightResidencyMechanism>,
    /// Mutable-state residency mechanisms.
    state_residencies: Vec<StateResidencyMechanism>,
    /// Exact session facilities implemented by the constructed session.
    session: SessionCapabilities,
    /// Prompt-cache persistence mechanism is available.
    prompt_cache: bool,
    /// Exact completion ownership is implemented for submitted work.
    exact_completion: bool,
    grouped_operations: Vec<GroupedOperationRequirement>,
}

impl BackendMechanismCapabilities {
    /// Creates a fail-closed mechanism report.
    pub fn new(
        operators: NeuralOperatorCapabilities,
        weight_lowerings: Vec<WeightLoweringCapability>,
        weight_residencies: Vec<WeightResidencyMechanism>,
        state_residencies: Vec<StateResidencyMechanism>,
    ) -> Self {
        Self {
            operators,
            weight_lowerings,
            weight_residencies,
            state_residencies,
            session: SessionCapabilities::default(),
            prompt_cache: false,
            exact_completion: false,
            grouped_operations: Vec::new(),
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
    /// Returns mutable-state residency mechanisms.
    pub fn state_residencies(&self) -> &[StateResidencyMechanism] {
        &self.state_residencies
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

/// Authoritative realization selected before architecture or payload construction.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedReplicatedTextRealization {
    /// Exact selected execution topology.
    topology: ParallelTopology,
    /// Selected ordinary parameter residency.
    residency: LayerWeightResidency,
    /// Selected mutable-state implementation and its exact residency policy.
    state: CacheResidencyPolicy,
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
    /// Returns the exact selected topology.
    pub const fn topology(&self) -> ParallelTopology {
        self.topology
    }
    /// Returns selected weight residency.
    pub const fn residency(&self) -> LayerWeightResidency {
        self.residency
    }
    /// Returns selected mutable-state policy.
    pub const fn state(&self) -> &CacheResidencyPolicy {
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
    let state_residency = match &request.state {
        CacheResidencyPolicy::Device => StateResidencyMechanism::Device,
        CacheResidencyPolicy::Paged(_) => StateResidencyMechanism::Paged,
    };
    if !capabilities.state_residencies.contains(&state_residency) {
        issues.push(format!("state residency {state_residency:?}"));
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
        if !parameter.presence.has_physical_source() {
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
            lowering: lowering.kind,
        });
    }
    if !issues.is_empty() {
        return Err(ReplicatedTextSelectionError { issues });
    }
    Ok(SelectedReplicatedTextRealization {
        topology: request
            .topology
            .unwrap_or_else(|| ParallelTopology::new(1, 1, 1, 1).expect("replicated topology")),
        residency: request.residency,
        state: request.state.clone(),
        parameters,
        session: request.session,
        prompt_cache: request.prompt_cache,
        exact_completion: request.exact_completion,
        grouped_operations: requirements.grouped_operations.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
        ArchitectureMergeDestination, DenseDiskStreamLoadOptions, ExecutionGroupSpec,
        ExecutionUnitLayout, LayerwiseLoadOptions, StateLayout,
    };
    use eredu_checkpoint::{AffineQuantization, StoredDtype};
    use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, LayerSchedule};

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
            vec![
                StateResidencyMechanism::Device,
                StateResidencyMechanism::Paged,
            ],
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
        assert_eq!(left.state(), &paged_state());
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
    fn selection_reports_all_missing_mechanisms_together() {
        let capabilities = BackendMechanismCapabilities::new(
            NeuralOperatorCapabilities::NONE,
            Vec::new(),
            Vec::new(),
            Vec::new(),
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
