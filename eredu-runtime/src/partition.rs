//! Backend-neutral ownership of one rank-local architecture partition.

use std::ops::Deref;
use std::{collections::BTreeMap, collections::BTreeSet, ops::Range};

use crate::{
    ExecutionGraph, ExecutionGroupId, ExecutionUnitLayout, LayeredForwardState,
    LayeredPartitionInput, LayeredPartitionOutput, ParameterGroupSpec,
    PartitionedLayeredArchitecture, RuntimeState, StateLayout,
};

/// Architecture-owned location of one neutral parameter group.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum ParameterGroupOwner {
    /// A pinned module selected by an explicit architecture static role.
    StaticRole(String),
    /// A shared pinned module selected when any declared static consumer is local.
    StaticAnyOf(Vec<String>),
    /// One architecture-global unit in a canonical execution group.
    ExecutionUnit {
        /// Canonical execution-group identity.
        group: ExecutionGroupId,
        /// Group-local architecture-global unit index.
        global_unit: usize,
    },
}

impl ParameterGroupOwner {
    /// Creates static-module ownership with a stable, non-empty role.
    pub fn static_role(role: impl Into<String>) -> Self {
        Self::StaticRole(role.into())
    }

    /// Creates shared static-module ownership across explicit consumer roles.
    pub fn static_any_of(roles: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::StaticAnyOf(roles.into_iter().map(Into::into).collect())
    }

    /// Creates execution-unit ownership in the architecture-global index space.
    pub fn execution_unit(group: ExecutionGroupId, global_unit: usize) -> Self {
        Self::ExecutionUnit { group, global_unit }
    }

    fn is_local<G, A>(&self, partition: &ArchitecturePartition<G, A>) -> bool {
        match self {
            Self::StaticRole(role) => partition.ownership().owns_static_role(role),
            Self::StaticAnyOf(roles) => roles
                .iter()
                .any(|role| partition.ownership().owns_static_role(role)),
            Self::ExecutionUnit { group, global_unit } => {
                partition.owns_unit(group.as_str(), *global_unit)
            }
        }
    }

    fn is_local_partition_parts(
        &self,
        groups: &[PartitionGroup],
        ownership: &PartitionOwnership,
    ) -> bool {
        match self {
            Self::StaticRole(role) => ownership.owns_static_role(role),
            Self::StaticAnyOf(roles) => roles.iter().any(|role| ownership.owns_static_role(role)),
            Self::ExecutionUnit { group, global_unit } => groups
                .iter()
                .any(|owned| owned.group() == group && owned.contains(*global_unit)),
        }
    }

    fn static_storage_role(&self) -> Option<&str> {
        match self {
            Self::StaticRole(role) => Some(role),
            Self::StaticAnyOf(roles) => roles.first().map(String::as_str),
            Self::ExecutionUnit { .. } => None,
        }
    }
}

/// One neutral parameter group tagged with its architecture-owned location.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct OwnedParameterGroupSpec {
    owner: ParameterGroupOwner,
    group: ParameterGroupSpec,
}

impl OwnedParameterGroupSpec {
    /// Tags a group with one explicit owner.
    pub fn new(owner: ParameterGroupOwner, group: ParameterGroupSpec) -> Self {
        Self { owner, group }
    }

    /// Returns the architecture-owned location.
    pub const fn owner(&self) -> &ParameterGroupOwner {
        &self.owner
    }

    /// Returns the neutral placement group.
    pub const fn group(&self) -> &ParameterGroupSpec {
        &self.group
    }

    /// Consumes the tag and returns the neutral placement group.
    pub fn into_group(self) -> ParameterGroupSpec {
        self.group
    }
}

impl Deref for OwnedParameterGroupSpec {
    type Target = ParameterGroupSpec;

    fn deref(&self) -> &Self::Target {
        &self.group
    }
}

/// Complete, validated parameter-ownership declaration for an architecture.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ArchitectureParameterDescription {
    graph: ExecutionGraph,
    unit_layout: ExecutionUnitLayout,
    groups: Vec<OwnedParameterGroupSpec>,
}

impl ArchitectureParameterDescription {
    /// Validates explicit ownership against the canonical graph/layout and an
    /// authoritative set of neutral parameter groups.
    pub fn new(
        graph: &ExecutionGraph,
        layout: &ExecutionUnitLayout,
        expected: impl IntoIterator<Item = ParameterGroupSpec>,
        groups: impl IntoIterator<Item = OwnedParameterGroupSpec>,
    ) -> Result<Self, ArchitectureParameterError> {
        validate_canonical_layout(graph, layout)
            .map_err(|error| ArchitectureParameterError::InvalidLayout(error.to_string()))?;
        let expected = parameter_targets(expected)?;
        let groups = groups.into_iter().collect::<Vec<_>>();
        let mut actual = BTreeMap::new();
        for tagged in &groups {
            match tagged.owner() {
                ParameterGroupOwner::StaticRole(role) => {
                    if role.trim().is_empty() {
                        return Err(ArchitectureParameterError::EmptyStaticRole);
                    }
                }
                ParameterGroupOwner::StaticAnyOf(roles) => {
                    if roles.is_empty() || roles.iter().any(|role| role.trim().is_empty()) {
                        return Err(ArchitectureParameterError::EmptyStaticRole);
                    }
                    let unique = roles.iter().collect::<BTreeSet<_>>();
                    if unique.len() != roles.len() {
                        return Err(ArchitectureParameterError::DuplicateStaticRole);
                    }
                }
                ParameterGroupOwner::ExecutionUnit { group, global_unit } => {
                    let Some(group_index) = graph
                        .groups()
                        .iter()
                        .position(|candidate| candidate.id() == group.as_str())
                    else {
                        return Err(ArchitectureParameterError::UnknownExecutionGroup(
                            group.as_str().to_owned(),
                        ));
                    };
                    let available = layout
                        .group_range(group_index)
                        .expect("validated canonical layout contains every group")
                        .len();
                    if *global_unit >= available {
                        return Err(ArchitectureParameterError::UnitOutOfRange {
                            group: group.as_str().to_owned(),
                            global_unit: *global_unit,
                            available,
                        });
                    }
                }
            }
            for member in tagged.group().members() {
                if let Some(previous) = actual.insert(member.target().to_owned(), tagged.owner()) {
                    return Err(ArchitectureParameterError::DuplicateOwnership {
                        target: member.target().to_owned(),
                        first: previous.clone(),
                        second: tagged.owner().clone(),
                    });
                }
            }
        }
        let actual_targets = actual.keys().cloned().collect::<BTreeSet<_>>();
        let expected_targets = expected.keys().cloned().collect::<BTreeSet<_>>();
        if let Some(target) = expected_targets.difference(&actual_targets).next() {
            return Err(ArchitectureParameterError::MissingOwnership(target.clone()));
        }
        if let Some(target) = actual_targets.difference(&expected_targets).next() {
            return Err(ArchitectureParameterError::UnexpectedOwnership(
                target.clone(),
            ));
        }
        Ok(Self {
            graph: graph.clone(),
            unit_layout: layout.clone(),
            groups,
        })
    }

    /// Returns the canonical execution graph that owns these parameter groups.
    pub const fn graph(&self) -> &ExecutionGraph {
        &self.graph
    }

    /// Returns the canonical architecture-global execution-unit layout that
    /// owns these parameter groups.
    pub const fn unit_layout(&self) -> &ExecutionUnitLayout {
        &self.unit_layout
    }

    /// Proves that this description still matches a concrete neutral architecture.
    pub fn validate_architecture<B, S, M>(
        &self,
        architecture: &M,
    ) -> Result<(), ArchitecturePartitionError>
    where
        B: eredu_nn::NeuralBackend,
        S: crate::RuntimeState<B>,
        M: crate::LayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
    {
        let (graph, unit_layout) = canonical_architecture_layout::<B, S, M>(architecture)?;
        if graph != self.graph {
            return Err(ArchitecturePartitionError::ArchitectureGraphMismatch);
        }
        if unit_layout != self.unit_layout {
            return Err(ArchitecturePartitionError::ArchitectureUnitLayoutMismatch);
        }
        Ok(())
    }

    /// Returns every explicitly tagged neutral parameter group.
    pub fn groups(&self) -> &[OwnedParameterGroupSpec] {
        &self.groups
    }

    /// Returns every physical target owned by groups with the supplied semantic role.
    ///
    /// Selection happens after architecture ownership has been assigned, so callers
    /// do not need to rediscover families, aliases, or packed companions from target
    /// name syntax.
    pub fn targets_for_role(&self, role: crate::ParameterRole) -> BTreeSet<String> {
        self.groups
            .iter()
            .filter(|owned| owned.group().role() == role)
            .flat_map(|owned| owned.group().members())
            .map(|member| member.target().to_owned())
            .collect()
    }

    /// Selects rank-owned groups without discarding their architecture owner.
    pub fn select_owned<G, A>(
        &self,
        partition: &ArchitecturePartition<G, A>,
    ) -> Vec<OwnedParameterGroupSpec> {
        self.groups
            .iter()
            .filter(|tagged| tagged.owner().is_local(partition))
            .cloned()
            .collect()
    }

    /// Returns canonical static storage roles selected for one partition.
    ///
    /// Shared owners always return their first declared role, while any later
    /// roles only act as ownership consumers.
    pub fn select_static_roles<'a, G, A>(
        &'a self,
        partition: &ArchitecturePartition<G, A>,
    ) -> Vec<&'a str> {
        self.groups
            .iter()
            .filter(|tagged| tagged.owner().is_local(partition))
            .filter_map(|tagged| tagged.owner().static_storage_role())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

fn parameter_targets(
    groups: impl IntoIterator<Item = ParameterGroupSpec>,
) -> Result<BTreeMap<String, String>, ArchitectureParameterError> {
    let mut targets = BTreeMap::new();
    for group in groups {
        for member in group.members() {
            if let Some(previous) =
                targets.insert(member.target().to_owned(), group.logical_name().to_owned())
            {
                return Err(ArchitectureParameterError::DuplicateExpectedTarget {
                    target: member.target().to_owned(),
                    first: previous,
                    second: group.logical_name().to_owned(),
                });
            }
        }
    }
    Ok(targets)
}

/// Invalid architecture-owned parameter ownership declaration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ArchitectureParameterError {
    /// The supplied graph/layout is not canonical.
    #[error("invalid architecture parameter layout: {0}")]
    InvalidLayout(String),
    /// A pinned parameter group has no semantic role.
    #[error("architecture parameter static role must not be empty")]
    EmptyStaticRole,
    /// A shared pinned parameter repeats one consumer role.
    #[error("architecture shared parameter owner repeats a static role")]
    DuplicateStaticRole,
    /// A unit owner names no canonical graph group.
    #[error("architecture parameter owner names unknown execution group {0:?}")]
    UnknownExecutionGroup(String),
    /// A unit owner exceeds its canonical group size.
    #[error("architecture parameter owner {group}:{global_unit} exceeds {available} units")]
    UnitOutOfRange {
        /// Canonical execution-group identity.
        group: String,
        /// Invalid group-local global unit.
        global_unit: usize,
        /// Canonical unit count.
        available: usize,
    },
    /// The authoritative neutral group set itself repeats a target.
    #[error("expected parameter target {target:?} appears in both {first:?} and {second:?}")]
    DuplicateExpectedTarget {
        /// Repeated physical target.
        target: String,
        /// First logical group.
        first: String,
        /// Second logical group.
        second: String,
    },
    /// Two explicit owners claim one physical target.
    #[error("parameter target {target:?} is owned by both {first:?} and {second:?}")]
    DuplicateOwnership {
        /// Repeated physical target.
        target: String,
        /// First explicit owner.
        first: ParameterGroupOwner,
        /// Second explicit owner.
        second: ParameterGroupOwner,
    },
    /// An authoritative target was left unowned.
    #[error("parameter target {0:?} has no architecture owner")]
    MissingOwnership(String),
    /// An ownership tag names a target outside the authoritative set.
    #[error("parameter target {0:?} is not present in the authoritative parameter groups")]
    UnexpectedOwnership(String),
}

/// Logical scalar kind carried by one architecture-owned boundary tensor.
///
/// `Activation` is resolved by a concrete backend to the execution dtype
/// selected for the surrounding pipeline activation. Integer kinds are exact.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum BoundaryTensorDtype {
    /// The selected execution activation dtype.
    Activation,
    /// Exact unsigned 32-bit integer values.
    Uint32,
    /// Exact signed 32-bit integer values.
    Int32,
}

/// Portable floating-point dtype carried between pipeline stages.
///
/// This is execution transport policy, not checkpoint storage metadata. A
/// concrete backend must lower the selected dtype to its native tensor dtype
/// and normalize outgoing activations to it before transport.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum PipelineActivationDtype {
    /// IEEE 16-bit floating point.
    Float16,
    /// Brain 16-bit floating point.
    Bfloat16,
    /// IEEE 32-bit floating point.
    Float32,
}

/// Backend-neutral wire contract shared by every stage of one pipeline.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct PipelineWireContract {
    activation_dtype: PipelineActivationDtype,
}

impl PipelineWireContract {
    /// Declares the exact dtype used by hidden activations and auxiliary
    /// tensors whose boundary dtype is [`BoundaryTensorDtype::Activation`].
    pub const fn new(activation_dtype: PipelineActivationDtype) -> Self {
        Self { activation_dtype }
    }

    /// Returns the exact floating-point dtype transported between stages.
    pub const fn activation_dtype(self) -> PipelineActivationDtype {
        self.activation_dtype
    }
}

/// One symbolic dimension in an architecture-owned boundary tensor.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum BoundaryTensorDimension {
    /// Invocation batch size.
    Batch,
    /// Invocation sequence length.
    Sequence,
    /// Positive architecture-defined extent.
    Fixed(i32),
}

/// Semantic role, symbolic shape, and logical dtype of one boundary tensor.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct BoundaryTensorSpec {
    role: String,
    shape: Vec<BoundaryTensorDimension>,
    dtype: BoundaryTensorDtype,
}

impl BoundaryTensorSpec {
    /// Declares one tensor in canonical transport order.
    pub fn new(
        role: impl Into<String>,
        shape: impl IntoIterator<Item = BoundaryTensorDimension>,
        dtype: BoundaryTensorDtype,
    ) -> Self {
        Self {
            role: role.into(),
            shape: shape.into_iter().collect(),
            dtype,
        }
    }

    /// Returns the stable semantic role.
    pub fn role(&self) -> &str {
        &self.role
    }

    /// Returns the symbolic shape.
    pub fn shape(&self) -> &[BoundaryTensorDimension] {
        &self.shape
    }

    /// Returns the logical scalar kind.
    pub const fn dtype(&self) -> BoundaryTensorDtype {
        self.dtype
    }
}

/// One boundary tensor after invocation-dependent dimensions are resolved.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct ResolvedBoundaryTensorSpec {
    role: String,
    shape: Vec<i32>,
    dtype: BoundaryTensorDtype,
}

impl ResolvedBoundaryTensorSpec {
    /// Returns the stable semantic role.
    pub fn role(&self) -> &str {
        &self.role
    }

    /// Returns the concrete transport shape.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }

    /// Returns the logical scalar kind.
    pub const fn dtype(&self) -> BoundaryTensorDtype {
        self.dtype
    }
}

/// Complete ordered auxiliary wire schema for one architecture boundary.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct BoundaryWireSchema {
    identity: &'static str,
    tensors: Vec<BoundaryTensorSpec>,
}

impl BoundaryWireSchema {
    /// Creates and validates an architecture-owned wire schema.
    pub fn new(
        identity: &'static str,
        tensors: impl IntoIterator<Item = BoundaryTensorSpec>,
    ) -> Result<Self, ArchitectureBoundaryError> {
        if identity.trim().is_empty() {
            return Err(ArchitectureBoundaryError::EmptyIdentity);
        }
        let tensors = tensors.into_iter().collect::<Vec<_>>();
        let mut roles = BTreeSet::new();
        for tensor in &tensors {
            if tensor.role.trim().is_empty() {
                return Err(ArchitectureBoundaryError::EmptyTensorRole { boundary: identity });
            }
            if !roles.insert(tensor.role.as_str()) {
                return Err(ArchitectureBoundaryError::DuplicateTensorRole {
                    boundary: identity,
                    role: tensor.role.clone(),
                });
            }
            if tensor.shape.is_empty() {
                return Err(ArchitectureBoundaryError::EmptyTensorShape {
                    boundary: identity,
                    role: tensor.role.clone(),
                });
            }
            if tensor
                .shape
                .iter()
                .any(|dimension| matches!(dimension, BoundaryTensorDimension::Fixed(value) if *value <= 0))
            {
                return Err(ArchitectureBoundaryError::InvalidTensorDimension {
                    boundary: identity,
                    role: tensor.role.clone(),
                });
            }
        }
        Ok(Self { identity, tensors })
    }

    /// Returns the stable schema identity.
    pub const fn identity(&self) -> &'static str {
        self.identity
    }

    /// Returns tensor declarations in canonical transport order.
    pub fn tensors(&self) -> &[BoundaryTensorSpec] {
        &self.tensors
    }

    /// Resolves invocation-dependent dimensions without backend family logic.
    pub fn resolve(
        &self,
        batch_size: i32,
        sequence_length: i32,
    ) -> Result<Vec<ResolvedBoundaryTensorSpec>, ArchitectureBoundaryError> {
        if batch_size <= 0 || sequence_length <= 0 {
            return Err(ArchitectureBoundaryError::InvalidInvocationGeometry {
                boundary: self.identity,
                batch_size,
                sequence_length,
            });
        }
        Ok(self
            .tensors
            .iter()
            .map(|tensor| ResolvedBoundaryTensorSpec {
                role: tensor.role.clone(),
                shape: tensor
                    .shape
                    .iter()
                    .map(|dimension| match dimension {
                        BoundaryTensorDimension::Batch => batch_size,
                        BoundaryTensorDimension::Sequence => sequence_length,
                        BoundaryTensorDimension::Fixed(value) => *value,
                    })
                    .collect(),
                dtype: tensor.dtype,
            })
            .collect())
    }
}

/// Typed architecture-owned tensor state and wire geometry carried across one
/// partition boundary.
///
/// The runtime and a backend transport may resolve and move the encoded tensor
/// vector, but only this family schema assigns semantic roles, shape, dtype,
/// cardinality, or reconstructs the typed value.
pub trait ArchitectureBoundary: Sized {
    /// Typed family value transported by this schema.
    type Boundary<T>;

    /// Stable non-empty semantic identity used in diagnostics and wire schema
    /// validation.
    const IDENTITY: &'static str;

    /// Tensor declarations in exact encoded order.
    fn tensor_specs(&self) -> Vec<BoundaryTensorSpec>;

    /// Consumes this typed value into transport-order tensors.
    fn encode<T>(&self, boundary: Self::Boundary<T>) -> Result<Vec<T>, ArchitectureBoundaryError>;

    /// Reconstructs the typed value from transport-order tensors.
    fn decode<T>(&self, tensors: Vec<T>) -> Result<Self::Boundary<T>, ArchitectureBoundaryError>;

    /// Returns the validated backend-neutral wire schema.
    fn wire_schema(&self) -> Result<BoundaryWireSchema, ArchitectureBoundaryError> {
        BoundaryWireSchema::new(Self::IDENTITY, self.tensor_specs())
    }
}

/// Explicit declaration that an architecture partition carries no auxiliary
/// tensors across its boundary.
///
/// This marker is preferable to `()` because it still participates in the
/// typed boundary contract and rejects any unexpected transported tensor.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct NoAuxiliaryBoundary;

impl ArchitectureBoundary for NoAuxiliaryBoundary {
    type Boundary<T> = NoAuxiliaryBoundary;

    const IDENTITY: &'static str = "none";

    fn tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
        Vec::new()
    }

    fn encode<T>(&self, _boundary: Self::Boundary<T>) -> Result<Vec<T>, ArchitectureBoundaryError> {
        Ok(Vec::new())
    }

    fn decode<T>(&self, tensors: Vec<T>) -> Result<Self::Boundary<T>, ArchitectureBoundaryError> {
        validate_boundary_tensor_count(self, &tensors)?;
        Ok(Self)
    }
}

/// Validates the number of tensors before a family boundary decodes any
/// positional value.
pub fn validate_boundary_tensor_count<B, T>(
    boundary: &B,
    tensors: &[T],
) -> Result<(), ArchitectureBoundaryError>
where
    B: ArchitectureBoundary,
{
    let expected = boundary.wire_schema()?.tensors().len();
    let actual = tensors.len();
    if actual != expected {
        return Err(ArchitectureBoundaryError::TensorCount {
            boundary: B::IDENTITY,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Invalid architecture-owned partition boundary declaration or payload.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ArchitectureBoundaryError {
    /// A family boundary omitted its stable identity.
    #[error("architecture boundary identity must not be empty")]
    EmptyIdentity,
    /// A family boundary declared an empty tensor role.
    #[error("architecture boundary {boundary:?} contains an empty tensor role")]
    EmptyTensorRole {
        /// Stable boundary identity.
        boundary: &'static str,
    },
    /// A family boundary declared one tensor role more than once.
    #[error("architecture boundary {boundary:?} repeats tensor role {role:?}")]
    DuplicateTensorRole {
        /// Stable boundary identity.
        boundary: &'static str,
        /// Repeated semantic tensor role.
        role: String,
    },
    /// A family boundary declared a rank-zero tensor.
    #[error("architecture boundary {boundary:?} tensor {role:?} has no dimensions")]
    EmptyTensorShape {
        /// Stable boundary identity.
        boundary: &'static str,
        /// Tensor semantic role.
        role: String,
    },
    /// A family boundary declared a non-positive fixed dimension.
    #[error("architecture boundary {boundary:?} tensor {role:?} has a non-positive dimension")]
    InvalidTensorDimension {
        /// Stable boundary identity.
        boundary: &'static str,
        /// Tensor semantic role.
        role: String,
    },
    /// A caller supplied non-positive invocation dimensions.
    #[error(
        "architecture boundary {boundary:?} requires positive invocation geometry, got batch {batch_size} and sequence {sequence_length}"
    )]
    InvalidInvocationGeometry {
        /// Stable boundary identity.
        boundary: &'static str,
        /// Invalid batch size.
        batch_size: i32,
        /// Invalid sequence length.
        sequence_length: i32,
    },
    /// A transported payload has the wrong tensor cardinality.
    #[error(
        "architecture boundary {boundary:?} expected {expected} tensors but received {actual}"
    )]
    TensorCount {
        /// Stable boundary identity.
        boundary: &'static str,
        /// Declared tensor count.
        expected: usize,
        /// Transported tensor count.
        actual: usize,
    },
    /// Family-specific boundary validation failed.
    #[error("architecture boundary {boundary:?} is invalid: {detail}")]
    Invalid {
        /// Stable boundary identity.
        boundary: &'static str,
        /// Family-owned failure detail.
        detail: String,
    },
}

/// Input, output, and pinned static-module ownership for one partition.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PartitionOwnership {
    input: bool,
    output: bool,
    static_roles: Vec<String>,
}

impl PartitionOwnership {
    /// Creates validated boundary and static-module ownership.
    pub fn new(
        input: bool,
        output: bool,
        static_roles: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, ArchitecturePartitionError> {
        let static_roles = static_roles.into_iter().map(Into::into).collect::<Vec<_>>();
        let mut unique = BTreeSet::new();
        for role in &static_roles {
            if role.trim().is_empty() {
                return Err(ArchitecturePartitionError::EmptyStaticRole);
            }
            if !unique.insert(role.clone()) {
                return Err(ArchitecturePartitionError::DuplicateStaticRole(
                    role.clone(),
                ));
            }
        }
        Ok(Self {
            input,
            output,
            static_roles,
        })
    }

    /// Returns whether this partition owns model input preparation.
    pub const fn owns_input(&self) -> bool {
        self.input
    }

    /// Returns whether this partition owns model output production.
    pub const fn owns_output(&self) -> bool {
        self.output
    }

    /// Returns pinned static roles in architecture declaration order.
    pub fn static_roles(&self) -> &[String] {
        &self.static_roles
    }

    /// Returns whether this partition owns a named pinned static role.
    pub fn owns_static_role(&self, role: &str) -> bool {
        self.static_roles.iter().any(|candidate| candidate == role)
    }
}

/// Rank-local mutable-state geometry and its architecture-global layer range.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PartitionState {
    layout: StateLayout,
    global_layers: Range<usize>,
}

impl PartitionState {
    /// Attaches a local state layout at one architecture-global layer offset.
    pub fn new(
        layout: StateLayout,
        global_layer_offset: usize,
    ) -> Result<Self, ArchitecturePartitionError> {
        let end = global_layer_offset.checked_add(layout.len()).ok_or(
            ArchitecturePartitionError::StateOffsetOverflow {
                offset: global_layer_offset,
                layers: layout.len(),
            },
        )?;
        Ok(Self {
            layout,
            global_layers: global_layer_offset..end,
        })
    }

    /// Returns the exact rank-local state layout.
    pub const fn layout(&self) -> &StateLayout {
        &self.layout
    }

    /// Returns the first architecture-global layer represented by the layout.
    pub const fn global_layer_offset(&self) -> usize {
        self.global_layers.start
    }

    /// Returns the architecture-global state-layer range.
    pub fn global_layers(&self) -> Range<usize> {
        self.global_layers.clone()
    }
}

/// One validated architecture group and its group-local global unit range.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PartitionGroup {
    group: ExecutionGroupId,
    group_index: usize,
    global_units: Range<usize>,
}

impl PartitionGroup {
    /// Returns the canonical execution-group identity.
    pub const fn group(&self) -> &ExecutionGroupId {
        &self.group
    }

    /// Returns the canonical execution-group slot.
    pub const fn group_index(&self) -> usize {
        self.group_index
    }

    /// Returns owned unit indices in the architecture group's global index space.
    pub fn global_units(&self) -> Range<usize> {
        self.global_units.clone()
    }

    /// Returns whether the range contains one group-local global unit index.
    pub fn contains(&self, global_unit: usize) -> bool {
        self.global_units.contains(&global_unit)
    }
}

/// Complete backend-neutral realization of one rank's architecture ownership.
///
/// `G` is family-owned local construction geometry. `A` is the family-owned
/// auxiliary wire schema carried by the partition realization.
#[derive(Debug, Clone)]
pub struct ArchitecturePartition<G, A> {
    graph: ExecutionGraph,
    unit_layout: ExecutionUnitLayout,
    groups: Vec<PartitionGroup>,
    ownership: PartitionOwnership,
    state: Option<PartitionState>,
    local_geometry: G,
    auxiliary_boundary: A,
    parameter_bindings: Vec<OwnedParameterGroupSpec>,
}

impl<G, A> ArchitecturePartition<G, A> {
    /// Creates a partition from the topology declared by one concrete neutral
    /// architecture.
    ///
    /// This is the only public constructor: it derives the graph and unit
    /// layout from `architecture`, preventing a backend realization from
    /// publishing a parallel topology that merely resembles, but is not the
    /// canonical topology of, the architecture it will execute.
    #[allow(clippy::too_many_arguments)]
    pub fn from_architecture<B, S, M, N>(
        architecture: &M,
        group_ranges: impl IntoIterator<Item = (N, Range<usize>)>,
        ownership: PartitionOwnership,
        state: Option<PartitionState>,
        local_geometry: G,
        auxiliary_boundary: A,
        parameter_bindings: impl IntoIterator<Item = OwnedParameterGroupSpec>,
    ) -> Result<Self, ArchitecturePartitionError>
    where
        B: eredu_nn::NeuralBackend,
        S: crate::RuntimeState<B>,
        M: crate::LayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
        N: Into<String>,
        A: ArchitectureBoundary,
    {
        let (graph, unit_layout) = canonical_architecture_layout::<B, S, M>(architecture)?;
        auxiliary_boundary.wire_schema()?;
        Self::new(
            graph,
            unit_layout,
            group_ranges,
            ownership,
            state,
            local_geometry,
            auxiliary_boundary,
            parameter_bindings,
        )
    }

    /// Creates one validated rank-local architecture partition after the
    /// authoritative architecture topology has already been derived.
    #[allow(clippy::too_many_arguments)]
    fn new<S>(
        graph: ExecutionGraph,
        unit_layout: ExecutionUnitLayout,
        group_ranges: impl IntoIterator<Item = (S, Range<usize>)>,
        ownership: PartitionOwnership,
        state: Option<PartitionState>,
        local_geometry: G,
        auxiliary_boundary: A,
        parameter_bindings: impl IntoIterator<Item = OwnedParameterGroupSpec>,
    ) -> Result<Self, ArchitecturePartitionError>
    where
        S: Into<String>,
    {
        validate_canonical_layout(&graph, &unit_layout)?;
        let mut seen_groups = BTreeSet::new();
        let mut groups = Vec::new();
        for (group, global_units) in group_ranges {
            let group = group.into();
            let group_index = graph
                .groups()
                .iter()
                .position(|candidate| candidate.id() == group)
                .ok_or_else(|| ArchitecturePartitionError::UnknownGroup(group.clone()))?;
            if !seen_groups.insert(group.clone()) {
                return Err(ArchitecturePartitionError::DuplicateGroup(group));
            }
            if global_units.is_empty() {
                return Err(ArchitecturePartitionError::EmptyGroupRange { group });
            }
            let available = unit_layout
                .group_range(group_index)
                .expect("canonical layout contains every graph group")
                .len();
            if global_units.end > available {
                return Err(ArchitecturePartitionError::GroupRangeOutOfBounds {
                    group,
                    start: global_units.start,
                    end: global_units.end,
                    available,
                });
            }
            groups.push(PartitionGroup {
                group: unit_layout
                    .group_id(group_index)
                    .expect("canonical layout contains every graph group identity")
                    .clone(),
                group_index,
                global_units,
            });
        }
        groups.sort_by_key(PartitionGroup::group_index);

        let parameter_bindings = parameter_bindings.into_iter().collect::<Vec<_>>();
        let mut targets = BTreeSet::new();
        for binding in &parameter_bindings {
            if !binding
                .owner()
                .is_local_partition_parts(&groups, &ownership)
            {
                return Err(ArchitecturePartitionError::NonLocalParameterOwner(
                    binding.owner().clone(),
                ));
            }
            for member in binding.members() {
                if !targets.insert(member.target().to_owned()) {
                    return Err(ArchitecturePartitionError::DuplicateParameterTarget(
                        member.target().to_owned(),
                    ));
                }
            }
        }

        Ok(Self {
            graph,
            unit_layout,
            groups,
            ownership,
            state,
            local_geometry,
            auxiliary_boundary,
            parameter_bindings,
        })
    }

    /// Returns the canonical architecture execution graph.
    pub const fn graph(&self) -> &ExecutionGraph {
        &self.graph
    }

    /// Returns the canonical complete execution-unit layout.
    pub const fn unit_layout(&self) -> &ExecutionUnitLayout {
        &self.unit_layout
    }

    /// Returns groups and group-local global unit ranges owned by this rank.
    pub fn groups(&self) -> &[PartitionGroup] {
        &self.groups
    }

    /// Traverses rank-owned execution units in canonical architecture order.
    pub fn units(&self) -> impl Iterator<Item = crate::ExecutionUnitAddress> + '_ {
        self.groups.iter().flat_map(move |owned| {
            let group = owned.group_index;
            let base = self
                .unit_layout
                .group_range(group)
                .expect("partition group belongs to its canonical layout")
                .start;
            owned.global_units.clone().map(move |index| {
                self.unit_layout
                    .address(base + index)
                    .expect("partition unit belongs to its canonical layout")
            })
        })
    }

    /// Returns whether this rank owns one group-local global unit.
    pub fn owns_unit(&self, group: &str, global_unit: usize) -> bool {
        self.groups
            .iter()
            .any(|owned| owned.group.as_str() == group && owned.contains(global_unit))
    }

    /// Returns input, output, and static-module ownership.
    pub const fn ownership(&self) -> &PartitionOwnership {
        &self.ownership
    }

    /// Returns rank-local state geometry when this partition owns mutable state.
    pub const fn state(&self) -> Option<&PartitionState> {
        self.state.as_ref()
    }

    /// Returns family-owned rank-local construction geometry.
    pub const fn local_geometry(&self) -> &G {
        &self.local_geometry
    }

    /// Returns the family-owned auxiliary boundary schema.
    pub const fn auxiliary_boundary(&self) -> &A {
        &self.auxiliary_boundary
    }

    /// Mutably returns the family-owned auxiliary boundary schema.
    pub fn auxiliary_boundary_mut(&mut self) -> &mut A {
        &mut self.auxiliary_boundary
    }

    /// Returns neutral semantic parameter bindings owned by this rank.
    pub fn parameter_bindings(&self) -> &[OwnedParameterGroupSpec] {
        &self.parameter_bindings
    }

    /// Returns the exact neutral groups assigned to one architecture owner.
    pub fn parameter_bindings_for_owner<'a>(
        &'a self,
        owner: &'a ParameterGroupOwner,
    ) -> impl Iterator<Item = &'a ParameterGroupSpec> + 'a {
        self.parameter_bindings
            .iter()
            .filter(move |binding| binding.owner() == owner)
            .map(OwnedParameterGroupSpec::group)
    }

    /// Proves that this partition still describes the supplied concrete
    /// neutral architecture.
    ///
    /// Loaders may use this when a partition crosses a backend boundary or is
    /// restored from a prepared plan. Both dependency edges and exact unit
    /// counts are compared; matching group names alone are insufficient.
    pub fn validate_architecture<B, S, M>(
        &self,
        architecture: &M,
    ) -> Result<(), ArchitecturePartitionError>
    where
        B: eredu_nn::NeuralBackend,
        S: crate::RuntimeState<B>,
        M: crate::LayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
    {
        let (graph, unit_layout) = canonical_architecture_layout::<B, S, M>(architecture)?;
        if graph != self.graph {
            return Err(ArchitecturePartitionError::ArchitectureGraphMismatch);
        }
        if unit_layout != self.unit_layout {
            return Err(ArchitecturePartitionError::ArchitectureUnitLayoutMismatch);
        }
        Ok(())
    }
}

/// Validated execution metadata for one rank-local layered partition.
///
/// This driver is the single owner of partition input/output checks, canonical
/// storage and state ranges, execution-group setup/completion, and final output
/// projection. Concrete backends retain only state storage and unit residency.
#[derive(Debug, Clone)]
pub struct LayeredPartitionDriver {
    group: usize,
    range: Range<usize>,
    state_layout: StateLayout,
    owns_input: bool,
    owns_output: bool,
}

impl LayeredPartitionDriver {
    /// Validates a canonical partition against its concrete unit storage.
    pub fn new<G, A>(
        partition: &ArchitecturePartition<G, A>,
        group_index: usize,
        storage_range: Range<usize>,
    ) -> Result<Self, LayeredPartitionError> {
        let group = partition
            .groups()
            .iter()
            .find(|group| group.group_index() == group_index)
            .ok_or(LayeredPartitionError::GroupNotOwned { group: group_index })?;
        let range = group.global_units();
        if storage_range != range {
            return Err(LayeredPartitionError::StorageRange {
                storage: storage_range,
                partition: range,
            });
        }
        let state = partition
            .state()
            .ok_or(LayeredPartitionError::MissingState)?;
        if state.global_layers().start > range.start || state.global_layers().end < range.end {
            return Err(LayeredPartitionError::StateRange {
                state: state.global_layers(),
                partition: range,
            });
        }
        Ok(Self {
            group: group.group_index(),
            range,
            state_layout: state.layout().clone(),
            owns_input: partition.ownership().owns_input(),
            owns_output: partition.ownership().owns_output(),
        })
    }

    /// Returns the canonical group-local global unit range.
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Returns the canonical architecture execution-group slot.
    pub const fn group_index(&self) -> usize {
        self.group
    }

    /// Returns the architecture-global state layout for this partition.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Validates input form against architecture boundary ownership.
    pub fn input<'a, T, A>(
        &self,
        input: LayeredPartitionInput<'a, T, A>,
    ) -> Result<LayeredPartitionInput<'a, T, A>, LayeredPartitionError> {
        match (&input, self.owns_input) {
            (LayeredPartitionInput::Tokens(_), true)
            | (LayeredPartitionInput::Hidden { .. }, _) => Ok(input),
            (LayeredPartitionInput::Tokens(_), false) => {
                Err(LayeredPartitionError::TokensOnNonInputOwner)
            }
        }
    }

    /// Prepares the partition and starts its canonical execution group.
    #[allow(clippy::too_many_arguments)]
    pub fn begin<'a, B, S, M>(
        &self,
        architecture: &mut M,
        input: LayeredPartitionInput<
            'a,
            B::Tensor,
            <M::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, M::ForwardContext>, M::Error>
    where
        B: eredu_nn::NeuralBackend,
        S: RuntimeState<B>,
        M: PartitionedLayeredArchitecture<B, S>,
    {
        let mut forward = match parallel {
            Some(parallel) => architecture.begin_partition_parallel(
                input,
                mask,
                state,
                &self.state_layout,
                self.range.start,
                parallel,
                context,
            ),
            None => architecture.begin_partition(
                input,
                mask,
                state,
                &self.state_layout,
                self.range.start,
                context,
            ),
        }?;
        forward.hidden = architecture.enter_partition_group(
            self.group,
            &forward.hidden,
            state,
            &mut forward.context,
            parallel,
            context,
        )?;
        Ok(forward)
    }

    /// Completes the canonical group and applies output projection only on its owner.
    #[allow(clippy::too_many_arguments)]
    pub fn finish<B, S, M>(
        &self,
        architecture: &mut M,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut M::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        LayeredPartitionOutput<
            B::Tensor,
            <M::Boundary as ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        M::Error,
    >
    where
        B: eredu_nn::NeuralBackend,
        S: RuntimeState<B>,
        M: PartitionedLayeredArchitecture<B, S>,
    {
        let hidden = architecture
            .leave_partition_group(self.group, hidden, state, forward, parallel, context)?;
        architecture.finish_partition(&hidden, state, forward, self.owns_output, parallel, context)
    }
}

/// Invalid concrete realization or boundary use of a layered partition.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum LayeredPartitionError {
    /// The selected architecture execution group is not owned by this partition.
    #[error("layered partition does not own execution group {group}")]
    GroupNotOwned {
        /// Canonical architecture group index.
        group: usize,
    },
    /// Concrete unit storage does not match canonical ownership.
    #[error("partition storage range {storage:?} disagrees with canonical range {partition:?}")]
    StorageRange {
        /// Concrete backend storage range.
        storage: Range<usize>,
        /// Canonical partition range.
        partition: Range<usize>,
    },
    /// Partition omitted mutable state geometry.
    #[error("layered partition has no runtime state")]
    MissingState,
    /// Mutable state geometry does not match canonical unit ownership.
    #[error("partition state range {state:?} disagrees with canonical range {partition:?}")]
    StateRange {
        /// Architecture-global state range.
        state: Range<usize>,
        /// Canonical partition range.
        partition: Range<usize>,
    },
    /// Token ids were supplied after the architecture input boundary.
    #[error("non-input partition received token ids")]
    TokensOnNonInputOwner,
}

fn canonical_architecture_layout<B, S, M>(
    architecture: &M,
) -> Result<(ExecutionGraph, ExecutionUnitLayout), ArchitecturePartitionError>
where
    B: eredu_nn::NeuralBackend,
    S: crate::RuntimeState<B>,
    M: crate::LayeredArchitecture<B, S>,
    M::Error: std::fmt::Display,
{
    let graph = architecture
        .execution_graph()
        .map_err(|error| ArchitecturePartitionError::ArchitectureTopology(error.to_string()))?;
    let mut counts = Vec::with_capacity(graph.groups().len());
    let mut paths = BTreeSet::new();
    for group in 0..graph.groups().len() {
        let count = architecture
            .group_unit_count(group)
            .map_err(|error| ArchitecturePartitionError::ArchitectureTopology(error.to_string()))?;
        counts.push(count);
        for index in 0..count {
            let path = architecture.unit_path(group, index).map_err(|error| {
                ArchitecturePartitionError::ArchitectureTopology(error.to_string())
            })?;
            if path.trim().is_empty() {
                return Err(ArchitecturePartitionError::EmptyArchitectureUnitPath { group, index });
            }
            if !paths.insert(path.clone()) {
                return Err(ArchitecturePartitionError::DuplicateArchitectureUnitPath(
                    path,
                ));
            }
        }
    }
    let unit_layout = ExecutionUnitLayout::new(&graph, counts)
        .map_err(|error| ArchitecturePartitionError::ArchitectureTopology(error.to_string()))?;
    Ok((graph, unit_layout))
}

fn validate_canonical_layout(
    graph: &ExecutionGraph,
    layout: &ExecutionUnitLayout,
) -> Result<(), ArchitecturePartitionError> {
    if graph.groups().len() != layout.group_count() {
        return Err(ArchitecturePartitionError::LayoutGroupCountMismatch {
            graph: graph.groups().len(),
            layout: layout.group_count(),
        });
    }
    for (index, group) in graph.groups().iter().enumerate() {
        let layout_group = layout
            .group_id(index)
            .expect("matching group counts provide every layout identity");
        if layout_group.as_str() != group.id() {
            return Err(ArchitecturePartitionError::LayoutGroupMismatch {
                index,
                graph: group.id().to_owned(),
                layout: layout_group.as_str().to_owned(),
            });
        }
    }
    Ok(())
}

/// Invalid backend-neutral architecture partition declaration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ArchitecturePartitionError {
    /// The architecture supplied an invalid partition-boundary wire schema.
    #[error("invalid architecture partition boundary: {0}")]
    InvalidBoundary(#[from] ArchitectureBoundaryError),
    /// The neutral architecture could not declare a canonical graph, unit
    /// count, or unit path.
    #[error("neutral architecture topology is invalid: {0}")]
    ArchitectureTopology(String),
    /// A neutral architecture exposed an empty stable unit path.
    #[error("neutral architecture unit {group}:{index} has an empty path")]
    EmptyArchitectureUnitPath {
        /// Canonical execution-group slot.
        group: usize,
        /// Group-local unit index.
        index: usize,
    },
    /// Two canonical architecture units exposed the same stable path.
    #[error("neutral architecture repeats unit path {0:?}")]
    DuplicateArchitectureUnitPath(String),
    /// The partition dependency graph differs from the concrete architecture.
    #[error("architecture partition dependency graph differs from the neutral architecture")]
    ArchitectureGraphMismatch,
    /// The partition unit counts differ from the concrete architecture.
    #[error("architecture partition unit layout differs from the neutral architecture")]
    ArchitectureUnitLayoutMismatch,
    /// The graph and complete unit layout contain different group counts.
    #[error("execution graph contains {graph} groups but its unit layout contains {layout}")]
    LayoutGroupCountMismatch {
        /// Canonical graph group count.
        graph: usize,
        /// Unit-layout group count.
        layout: usize,
    },
    /// A unit-layout group identity differs from the graph at the same slot.
    #[error("execution group {index} is {graph:?} in the graph but {layout:?} in the unit layout")]
    LayoutGroupMismatch {
        /// Canonical group slot.
        index: usize,
        /// Graph identity.
        graph: String,
        /// Unit-layout identity.
        layout: String,
    },
    /// A rank-local unit range names no canonical architecture group.
    #[error("architecture partition names unknown execution group {0:?}")]
    UnknownGroup(String),
    /// A canonical architecture group was declared more than once.
    #[error("architecture partition repeats execution group {0:?}")]
    DuplicateGroup(String),
    /// A group owns no execution units.
    #[error("architecture partition declares an empty unit range for group {group:?}")]
    EmptyGroupRange {
        /// Canonical group identity.
        group: String,
    },
    /// A group-local global unit range exceeds the canonical group size.
    #[error(
        "architecture partition range {start}..{end} for group {group:?} exceeds {available} units"
    )]
    GroupRangeOutOfBounds {
        /// Canonical group identity.
        group: String,
        /// Invalid range start.
        start: usize,
        /// Invalid range end.
        end: usize,
        /// Canonical group unit count.
        available: usize,
    },
    /// A static ownership role is blank.
    #[error("architecture partition static role must not be empty")]
    EmptyStaticRole,
    /// A static ownership role was repeated.
    #[error("architecture partition repeats static role {0:?}")]
    DuplicateStaticRole(String),
    /// A local state layout cannot be placed in the global layer index space.
    #[error("state layer offset {offset} plus {layers} local layers overflowed usize")]
    StateOffsetOverflow {
        /// Requested global layer offset.
        offset: usize,
        /// Local state-layer count.
        layers: usize,
    },
    /// Two semantic parameter groups claim the same physical target.
    #[error("architecture partition repeats parameter target {0:?}")]
    DuplicateParameterTarget(String),
    /// A supplied parameter owner is not part of this rank-local partition.
    #[error("architecture partition includes non-local parameter owner {0:?}")]
    NonLocalParameterOwner(ParameterGroupOwner),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemberSharding, ParameterMemberSpec, ParameterRole};
    use eredu_core::{cache::LayerCachePolicy, LayerSchedule};

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct Geometry(&'static str);

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct Boundary {
        route: usize,
    }

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct PairBoundary<T> {
        tokens: T,
        embedded: T,
    }

    #[derive(Debug, Clone, Copy)]
    struct PairBoundarySchema;

    impl ArchitectureBoundary for PairBoundarySchema {
        type Boundary<T> = PairBoundary<T>;

        const IDENTITY: &'static str = "fixture.target";

        fn tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
            vec![
                BoundaryTensorSpec::new(
                    "tokens",
                    [
                        BoundaryTensorDimension::Batch,
                        BoundaryTensorDimension::Sequence,
                    ],
                    BoundaryTensorDtype::Uint32,
                ),
                BoundaryTensorSpec::new(
                    "embedded",
                    [
                        BoundaryTensorDimension::Batch,
                        BoundaryTensorDimension::Sequence,
                        BoundaryTensorDimension::Fixed(16),
                    ],
                    BoundaryTensorDtype::Activation,
                ),
            ]
        }

        fn encode<T>(
            &self,
            boundary: Self::Boundary<T>,
        ) -> Result<Vec<T>, ArchitectureBoundaryError> {
            Ok(vec![boundary.tokens, boundary.embedded])
        }

        fn decode<T>(
            &self,
            mut tensors: Vec<T>,
        ) -> Result<Self::Boundary<T>, ArchitectureBoundaryError> {
            validate_boundary_tensor_count(self, &tensors)?;
            let embedded = tensors.pop().expect("validated embedded tensor");
            let tokens = tensors.pop().expect("validated token tensor");
            Ok(PairBoundary { tokens, embedded })
        }
    }

    fn graph() -> ExecutionGraph {
        ExecutionGraph::chain(["primary", "prediction"]).unwrap()
    }

    fn layout(graph: &ExecutionGraph) -> ExecutionUnitLayout {
        ExecutionUnitLayout::new(graph, [4, 3]).unwrap()
    }

    fn state_layout(layers: usize) -> StateLayout {
        StateLayout::new(
            LayerSchedule::new(layers, vec![LayerCachePolicy::NoState; layers]).unwrap(),
        )
        .unwrap()
    }

    fn parameter(logical: &str, target: &str) -> ParameterGroupSpec {
        ParameterGroupSpec::new(
            logical,
            ParameterRole::Replicated,
            [ParameterMemberSpec::new(
                target,
                vec![2, 2],
                MemberSharding::Replicated,
            )],
        )
        .unwrap()
    }

    fn valid_partition() -> ArchitecturePartition<Geometry, Boundary> {
        let graph = graph();
        let layout = layout(&graph);
        ArchitecturePartition::new(
            graph,
            layout,
            [("prediction", 0..2), ("primary", 1..4)],
            PartitionOwnership::new(true, false, ["embedding", "normalization"]).unwrap(),
            Some(PartitionState::new(state_layout(2), 7).unwrap()),
            Geometry("local"),
            Boundary { route: 3 },
            [
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role("embedding"),
                    parameter("model.embed_tokens", "model.embed_tokens.weight"),
                ),
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(
                        ExecutionGroupId::new("primary").unwrap(),
                        1,
                    ),
                    parameter("model.layers.1", "model.layers.1.weight"),
                ),
            ],
        )
        .unwrap()
    }

    fn parameter_description(
        expected: Vec<ParameterGroupSpec>,
        groups: Vec<OwnedParameterGroupSpec>,
    ) -> Result<ArchitectureParameterDescription, ArchitectureParameterError> {
        let graph = graph();
        ArchitectureParameterDescription::new(&graph, &layout(&graph), expected, groups)
    }

    #[test]
    fn parameter_description_selects_static_roles_and_canonical_units() {
        let embedding = parameter("embedding", "model.embed_tokens.weight");
        let layer = parameter("layer", "model.layers.1.weight");
        let description = parameter_description(
            vec![embedding.clone(), layer.clone()],
            vec![
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role("embedding"),
                    embedding,
                ),
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(
                        ExecutionGroupId::new("primary").unwrap(),
                        1,
                    ),
                    layer,
                ),
            ],
        )
        .unwrap();
        let partition = valid_partition();
        assert_eq!(description.graph(), partition.graph());
        assert_eq!(description.unit_layout(), partition.unit_layout());
        let selected = description.select_owned(&partition);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].logical_name(), "embedding");
        assert_eq!(selected[1].logical_name(), "layer");
        assert_eq!(
            selected[0].owner(),
            &ParameterGroupOwner::static_role("embedding")
        );
        assert_eq!(
            selected[1].owner(),
            &ParameterGroupOwner::execution_unit(ExecutionGroupId::new("primary").unwrap(), 1,)
        );
    }

    #[test]
    fn parameter_description_selects_every_owned_target_for_a_role() {
        let expert = ParameterGroupSpec::new(
            "model.layers.1.expert_intermediate",
            ParameterRole::ExpertIntermediate,
            [
                ParameterMemberSpec::new(
                    "model.layers.1.moe.packed.weight",
                    vec![4, 2],
                    MemberSharding::Replicated,
                ),
                ParameterMemberSpec::new(
                    "model.layers.1.moe.packed.scales",
                    vec![4, 1],
                    MemberSharding::Replicated,
                ),
                ParameterMemberSpec::new(
                    "model.layers.1.moe.alias.biases",
                    vec![4, 1],
                    MemberSharding::Replicated,
                ),
            ],
        )
        .unwrap();
        let replicated = parameter("router", "model.layers.1.moe.router.weight");
        let owner =
            ParameterGroupOwner::execution_unit(ExecutionGroupId::new("primary").unwrap(), 1);
        let description = parameter_description(
            vec![expert.clone(), replicated.clone()],
            vec![
                OwnedParameterGroupSpec::new(owner.clone(), expert),
                OwnedParameterGroupSpec::new(owner, replicated),
            ],
        )
        .unwrap();

        assert_eq!(
            description.targets_for_role(ParameterRole::ExpertIntermediate),
            BTreeSet::from([
                "model.layers.1.moe.alias.biases".to_owned(),
                "model.layers.1.moe.packed.scales".to_owned(),
                "model.layers.1.moe.packed.weight".to_owned(),
            ])
        );
    }

    #[test]
    fn parameter_description_selects_shared_static_owner_by_any_consumer() {
        let embedding = parameter("embedding", "model.embed_tokens.weight");
        let description = parameter_description(
            vec![embedding.clone()],
            vec![OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_any_of(["output", "embedding"]),
                embedding,
            )],
        )
        .unwrap();
        assert_eq!(description.select_owned(&valid_partition()).len(), 1);

        let duplicate = parameter("embedding", "model.embed_tokens.weight");
        assert_eq!(
            parameter_description(
                vec![duplicate.clone()],
                vec![OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_any_of(["embedding", "embedding"]),
                    duplicate,
                )],
            )
            .unwrap_err(),
            ArchitectureParameterError::DuplicateStaticRole,
        );
    }

    #[test]
    fn partition_rejects_parameter_owner_outside_local_unit_ranges() {
        let graph = graph();
        let error = ArchitecturePartition::new(
            graph.clone(),
            layout(&graph),
            [("primary", 1..4)],
            PartitionOwnership::new(false, false, ["embedding"]).unwrap(),
            None,
            (),
            (),
            [OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(
                    ExecutionGroupId::new("prediction").unwrap(),
                    0,
                ),
                parameter("prediction", "prediction.weight"),
            )],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ArchitecturePartitionError::NonLocalParameterOwner(
                ParameterGroupOwner::ExecutionUnit { .. }
            )
        ));
    }

    #[test]
    fn parameter_description_rejects_missing_duplicate_and_out_of_range_ownership() {
        let embedding = parameter("embedding", "model.embed_tokens.weight");
        let layer = parameter("layer", "model.layers.1.weight");
        assert_eq!(
            parameter_description(
                vec![embedding.clone(), layer.clone()],
                vec![OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role("embedding"),
                    embedding.clone(),
                )],
            )
            .unwrap_err(),
            ArchitectureParameterError::MissingOwnership("model.layers.1.weight".into())
        );
        assert!(matches!(
            parameter_description(
                vec![embedding.clone()],
                vec![
                    OwnedParameterGroupSpec::new(
                        ParameterGroupOwner::static_role("embedding"),
                        embedding.clone(),
                    ),
                    OwnedParameterGroupSpec::new(
                        ParameterGroupOwner::static_role("output"),
                        embedding.clone(),
                    ),
                ],
            )
            .unwrap_err(),
            ArchitectureParameterError::DuplicateOwnership { .. }
        ));
        assert_eq!(
            parameter_description(
                vec![layer.clone()],
                vec![OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(
                        ExecutionGroupId::new("prediction").unwrap(),
                        3,
                    ),
                    layer,
                )],
            )
            .unwrap_err(),
            ArchitectureParameterError::UnitOutOfRange {
                group: "prediction".into(),
                global_unit: 3,
                available: 3,
            }
        );
    }

    #[test]
    fn retains_canonical_topology_ownership_and_typed_family_values() {
        let mut partition = valid_partition();
        assert_eq!(partition.graph().groups().len(), 2);
        assert_eq!(partition.unit_layout().len(), 7);
        assert_eq!(partition.groups()[0].group().as_str(), "primary");
        assert_eq!(partition.groups()[0].group_index(), 0);
        assert_eq!(partition.groups()[0].global_units(), 1..4);
        assert!(partition.owns_unit("primary", 3));
        assert!(!partition.owns_unit("primary", 0));
        assert!(partition.ownership().owns_input());
        assert!(!partition.ownership().owns_output());
        assert!(partition.ownership().owns_static_role("embedding"));
        assert_eq!(
            partition
                .units()
                .map(|unit| (unit.group(), unit.index()))
                .collect::<Vec<_>>(),
            [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
        );
        assert_eq!(partition.state().unwrap().global_layers(), 7..9);
        assert_eq!(partition.local_geometry(), &Geometry("local"));
        partition.auxiliary_boundary_mut().route = 5;
        assert_eq!(partition.auxiliary_boundary().route, 5);
        assert_eq!(partition.parameter_bindings().len(), 2);
    }

    #[test]
    fn typed_boundary_owns_roles_order_and_atomic_cardinality_validation() {
        let boundary = PairBoundary {
            tokens: 3,
            embedded: 7,
        };
        let schema = PairBoundarySchema;
        let tensors = schema.encode(boundary).unwrap();
        assert_eq!(tensors, [3, 7]);
        assert_eq!(
            schema.decode(tensors).unwrap(),
            PairBoundary {
                tokens: 3,
                embedded: 7
            }
        );
        let resolved = schema.wire_schema().unwrap().resolve(2, 3).unwrap();
        assert_eq!(resolved[0].shape(), [2, 3]);
        assert_eq!(resolved[0].dtype(), BoundaryTensorDtype::Uint32);
        assert_eq!(resolved[1].shape(), [2, 3, 16]);
        assert_eq!(resolved[1].dtype(), BoundaryTensorDtype::Activation);
        assert_eq!(
            schema.decode(vec![3]).unwrap_err(),
            ArchitectureBoundaryError::TensorCount {
                boundary: "fixture.target",
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn boundary_schema_rejects_role_and_geometry_drift_before_transport() {
        let duplicate = BoundaryWireSchema::new(
            "fixture.invalid",
            [
                BoundaryTensorSpec::new(
                    "state",
                    [BoundaryTensorDimension::Fixed(1)],
                    BoundaryTensorDtype::Activation,
                ),
                BoundaryTensorSpec::new(
                    "state",
                    [BoundaryTensorDimension::Fixed(2)],
                    BoundaryTensorDtype::Activation,
                ),
            ],
        )
        .unwrap_err();
        assert_eq!(
            duplicate,
            ArchitectureBoundaryError::DuplicateTensorRole {
                boundary: "fixture.invalid",
                role: "state".into(),
            }
        );

        let invalid = BoundaryWireSchema::new(
            "fixture.invalid",
            [BoundaryTensorSpec::new(
                "state",
                [BoundaryTensorDimension::Fixed(0)],
                BoundaryTensorDtype::Activation,
            )],
        )
        .unwrap_err();
        assert_eq!(
            invalid,
            ArchitectureBoundaryError::InvalidTensorDimension {
                boundary: "fixture.invalid",
                role: "state".into(),
            }
        );
    }

    #[test]
    fn rejects_noncanonical_unknown_and_duplicate_groups() {
        let graph = graph();
        let mismatched_graph = ExecutionGraph::chain(["primary", "other"]).unwrap();
        let error = ArchitecturePartition::new(
            graph.clone(),
            layout(&mismatched_graph),
            [("primary", 0..1)],
            PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap(),
            None,
            (),
            (),
            std::iter::empty(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ArchitecturePartitionError::LayoutGroupMismatch { .. }
        ));

        let error = ArchitecturePartition::new(
            graph.clone(),
            layout(&graph),
            [("missing", 0..1)],
            PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap(),
            None,
            (),
            (),
            std::iter::empty(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ArchitecturePartitionError::UnknownGroup("missing".into())
        );

        let error = ArchitecturePartition::new(
            graph.clone(),
            layout(&graph),
            [("primary", 0..1), ("primary", 1..2)],
            PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap(),
            None,
            (),
            (),
            std::iter::empty(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ArchitecturePartitionError::DuplicateGroup("primary".into())
        );
    }

    #[test]
    fn rejects_empty_and_out_of_bounds_group_ranges() {
        let graph = graph();
        let error = ArchitecturePartition::new(
            graph.clone(),
            layout(&graph),
            [("primary", 2..2)],
            PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap(),
            None,
            (),
            (),
            std::iter::empty(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ArchitecturePartitionError::EmptyGroupRange { .. }
        ));

        let error = ArchitecturePartition::new(
            graph.clone(),
            layout(&graph),
            [("prediction", 1..4)],
            PartitionOwnership::new(false, false, std::iter::empty::<String>()).unwrap(),
            None,
            (),
            (),
            std::iter::empty(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ArchitecturePartitionError::GroupRangeOutOfBounds { .. }
        ));
    }

    #[test]
    fn rejects_state_offset_overflow() {
        assert_eq!(
            PartitionState::new(state_layout(2), usize::MAX).unwrap_err(),
            ArchitecturePartitionError::StateOffsetOverflow {
                offset: usize::MAX,
                layers: 2,
            }
        );
    }

    #[test]
    fn rejects_empty_static_roles_and_duplicate_parameter_targets() {
        assert_eq!(
            PartitionOwnership::new(false, false, [" "]).unwrap_err(),
            ArchitecturePartitionError::EmptyStaticRole
        );

        let graph = graph();
        let error = ArchitecturePartition::new(
            graph.clone(),
            layout(&graph),
            [("primary", 0..1)],
            PartitionOwnership::new(false, false, ["embedding", "normalization"]).unwrap(),
            None,
            (),
            (),
            [
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role("embedding"),
                    parameter("first", "shared.weight"),
                ),
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role("normalization"),
                    parameter("second", "shared.weight"),
                ),
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            ArchitecturePartitionError::DuplicateParameterTarget("shared.weight".into())
        );
    }

    fn layered_partition(
        storage_state: Range<usize>,
        owns_input: bool,
    ) -> ArchitecturePartition<(), ()> {
        let graph = ExecutionGraph::chain(["decoder"]).unwrap();
        let layout = ExecutionUnitLayout::new(&graph, [4]).unwrap();
        ArchitecturePartition::new(
            graph,
            layout,
            [("decoder", 1..3)],
            PartitionOwnership::new(owns_input, false, std::iter::empty::<String>()).unwrap(),
            Some(
                PartitionState::new(state_layout(storage_state.len()), storage_state.start)
                    .unwrap(),
            ),
            (),
            (),
            std::iter::empty(),
        )
        .unwrap()
    }

    #[test]
    fn layered_driver_rejects_storage_and_state_range_drift() {
        let partition = layered_partition(1..3, true);
        assert!(LayeredPartitionDriver::new(&partition, 0, 1..3).is_ok());
        assert_eq!(
            LayeredPartitionDriver::new(&partition, 0, 0..2).unwrap_err(),
            LayeredPartitionError::StorageRange {
                storage: 0..2,
                partition: 1..3,
            }
        );

        let partition = layered_partition(0..2, true);
        assert_eq!(
            LayeredPartitionDriver::new(&partition, 0, 1..3).unwrap_err(),
            LayeredPartitionError::StateRange {
                state: 0..2,
                partition: 1..3,
            }
        );
    }

    #[test]
    fn layered_driver_restricts_tokens_but_accepts_architecture_prepared_hidden() {
        let input_owner =
            LayeredPartitionDriver::new(&layered_partition(1..3, true), 0, 1..3).unwrap();
        assert!(matches!(
            input_owner.input(LayeredPartitionInput::<i32, NoAuxiliaryBoundary>::Tokens(
                &7
            )),
            Ok(LayeredPartitionInput::Tokens(7))
        ));
        assert!(matches!(
            input_owner.input(LayeredPartitionInput::Hidden {
                hidden: 7,
                auxiliary: NoAuxiliaryBoundary,
            }),
            Ok(LayeredPartitionInput::Hidden {
                hidden: 7,
                auxiliary: NoAuxiliaryBoundary,
            })
        ));

        let hidden_owner =
            LayeredPartitionDriver::new(&layered_partition(1..3, false), 0, 1..3).unwrap();
        assert_eq!(
            hidden_owner
                .input(LayeredPartitionInput::<i32, NoAuxiliaryBoundary>::Tokens(
                    &7
                ))
                .unwrap_err(),
            LayeredPartitionError::TokensOnNonInputOwner
        );
        assert!(matches!(
            hidden_owner.input(LayeredPartitionInput::Hidden {
                hidden: 7,
                auxiliary: NoAuxiliaryBoundary,
            }),
            Ok(LayeredPartitionInput::Hidden {
                hidden: 7,
                auxiliary: NoAuxiliaryBoundary,
            })
        ));
    }
}
