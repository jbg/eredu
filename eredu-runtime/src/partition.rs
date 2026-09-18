//! Backend-neutral ownership of one rank-local architecture partition.

mod description;
mod ownership;

use std::ops::Deref;
use std::{collections::BTreeSet, ops::Range};

use crate::{
    ArchitectureStatePartitionError, ArchitectureStatePartitionPlan, ArchitectureStatePlacement,
    ExecutionGraph, ExecutionGroupId, ExecutionUnitLayout, LayeredForwardState,
    LayeredPartitionInput, LayeredPartitionOutput, ParameterGroupSpec,
    PartitionedLayeredArchitecture, RuntimeState, StateLayout,
};

/// Architecture-owned location of one neutral parameter group.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
#[non_exhaustive]
pub enum ParameterGroupOwner {
    /// A pinned module selected by an explicit architecture static role.
    StaticRole(String),
    /// A shared pinned module selected when any declared static consumer is local.
    StaticAnyOf(Vec<String>),
    /// A pinned module used by explicit logical units. Each partition containing
    /// a consumer stores one copy, regardless of how many consumers it owns.
    StaticUnitConsumers {
        /// Canonical static binding role.
        role: String,
        /// Canonical group and unit identities that consume the module.
        consumers: Vec<(ExecutionGroupId, usize)>,
    },
    /// One architecture-global unit in a canonical execution group.
    #[non_exhaustive]
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

    /// Pins one shared module on each partition executing a declared consumer.
    pub fn static_unit_consumers(
        role: impl Into<String>,
        consumers: impl IntoIterator<Item = (ExecutionGroupId, usize)>,
    ) -> Self {
        Self::StaticUnitConsumers {
            role: role.into(),
            consumers: consumers.into_iter().collect(),
        }
    }

    /// Whether this declaration binds through the given static storage role.
    pub fn accepts_static_role(&self, expected: &str) -> bool {
        match self {
            Self::StaticRole(role) | Self::StaticUnitConsumers { role, .. } => role == expected,
            Self::StaticAnyOf(roles) => roles.iter().any(|role| role == expected),
            Self::ExecutionUnit { .. } => false,
        }
    }

    /// Compares exact ownership, allowing a constructed consumer declaration
    /// to refine a selected static binding role.
    pub fn refines_storage_owner(&self, selected: &Self) -> bool {
        self == selected
            || matches!(selected, Self::StaticRole(role) if self.accepts_static_role(role))
    }

    /// Creates execution-unit ownership in the architecture-global index space.
    pub fn execution_unit(group: ExecutionGroupId, global_unit: usize) -> Self {
        Self::ExecutionUnit { group, global_unit }
    }

    pub(crate) fn validate(
        &self,
        graph: &ExecutionGraph,
        layout: &ExecutionUnitLayout,
    ) -> Result<(), ArchitectureParameterError> {
        description::owner(self, graph, layout).map_err(description::Issue::into_owned)
    }

    fn is_local<G, A>(&self, partition: &ArchitecturePartition<G, A>) -> bool {
        self.is_stored_by(partition.ownership(), |group, unit| {
            partition.owns_unit(group.as_str(), unit)
        })
    }

    fn is_local_partition_parts(
        &self,
        groups: &[PartitionGroup],
        ownership: &PartitionOwnership,
    ) -> bool {
        self.is_stored_by(ownership, |group, unit| {
            groups
                .iter()
                .any(|owned| owned.group() == group && owned.contains(unit))
        })
    }

    /// Tests static consumers and unit ownership without constructing a module.
    pub fn is_owned_by(
        &self,
        ownership: &PartitionOwnership,
        owns_unit: impl Fn(&ExecutionGroupId, usize) -> bool,
    ) -> bool {
        match self {
            Self::StaticRole(role) => ownership.owns_static_role(role),
            Self::StaticAnyOf(roles) => roles.iter().any(|role| ownership.owns_static_role(role)),
            Self::StaticUnitConsumers { consumers, .. } => consumers
                .iter()
                .any(|(group, unit)| owns_unit(group, *unit)),
            Self::ExecutionUnit { group, global_unit } => owns_unit(group, *global_unit),
        }
    }

    /// Tests parameter storage, including replicas for auxiliary invocations.
    /// Use `is_owned_by` for the ordinary model's observation ownership.
    pub fn is_stored_by(
        &self,
        ownership: &PartitionOwnership,
        owns_unit: impl Fn(&ExecutionGroupId, usize) -> bool,
    ) -> bool {
        match self {
            Self::StaticRole(role) => ownership.stores_static_role(role),
            Self::StaticAnyOf(roles) => roles.iter().any(|role| ownership.stores_static_role(role)),
            Self::StaticUnitConsumers { role, consumers } => {
                ownership.replicated_static_roles().contains(role)
                    || consumers
                        .iter()
                        .any(|(group, unit)| owns_unit(group, *unit))
            }
            Self::ExecutionUnit { group, global_unit } => owns_unit(group, *global_unit),
        }
    }

    fn static_storage_role(&self) -> Option<&str> {
        match self {
            Self::StaticRole(role) => Some(role),
            Self::StaticAnyOf(roles) => roles.first().map(String::as_str),
            Self::StaticUnitConsumers { role, .. } => Some(role),
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
    /// Moves an owned declaration or copies a borrowed one through the actual destination.
    /// A borrowed description never silently allocates when a metadata owner is supplied.
    pub fn into_owned(description:std::borrow::Cow<'_,Self>,metadata:Option<&eredu_nn::workspace::WorkspaceContext>)->Result<Self,eredu_nn::Error>{
        let Some(context)=metadata.filter(|context|context.uses_checked_metadata()) else{return Ok(description.into_owned());};
        context.charge_metadata(std::mem::size_of::<(std::borrow::Cow<'_,Self>,Option<&eredu_nn::workspace::WorkspaceContext>,Self,Result<Self,eredu_nn::Error>)>())?;
        match description{std::borrow::Cow::Owned(value)=>Ok(value),std::borrow::Cow::Borrowed(value)=>description::copy(value,context)}
    }

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
        let expected_groups = expected.into_iter().collect::<Vec<_>>();
        let destination = description::Destination(None);
        let expected = description::expected(expected_groups.iter(), destination)
            .map_err(description::Failure::ordinary)?;
        let groups = groups.into_iter().collect::<Vec<_>>();
        description::owned(graph, layout, &groups, &expected, destination)
            .map_err(description::Failure::ordinary)?;
        Ok(Self {
            graph: graph.clone(),
            unit_layout: layout.clone(),
            groups,
        })
    }

    /// Consumes the complete groups emitted by one architecture constructor.
    ///
    /// The same expected-target and owner validation is used by `new`. Its
    /// authoritative expected groups are the emitted groups themselves, avoiding
    /// a deep duplicate of names, shapes, companions, and placement metadata.
    /// Graph, layout, and groups are moved into the description.
    pub fn from_owned(
        graph: ExecutionGraph,
        layout: ExecutionUnitLayout,
        groups: Vec<OwnedParameterGroupSpec>,
    ) -> Result<Self, ArchitectureParameterError> {
        Self::from_owned_destination(graph, layout, groups, description::Destination(None))
            .map_err(description::Failure::ordinary)
    }

    /// Consumes complete emitted groups using the caller's metadata destination.
    ///
    /// Uses the same validation and ownership transfer as `from_owned`. The
    /// caller retains the Context's metadata custody while the returned
    /// description is in use.
    pub fn from_owned_with_metadata(
        graph: ExecutionGraph,
        layout: ExecutionUnitLayout,
        groups: Vec<OwnedParameterGroupSpec>,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        Self::from_owned_destination(graph, layout, groups, description::Destination(Some(context)))
            .map_err(description::Failure::metadata)
    }

    fn from_owned_destination(
        graph: ExecutionGraph,
        layout: ExecutionUnitLayout,
        groups: Vec<OwnedParameterGroupSpec>,
        destination: description::Destination<'_>,
    ) -> Result<Self, description::Failure> {
        destination.controls::<Self>()?;
        description::canonical(&graph, &layout)
            .map_err(|cause| destination.issue(description::Issue::Layout(cause)))?;
        let expected = description::expected(
            groups.iter().map(OwnedParameterGroupSpec::group),
            destination,
        )?;
        description::owned(&graph, &layout, &groups, &expected, destination)?;
        drop(expected);
        Ok(Self { graph, unit_layout: layout, groups })
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

/// Invalid architecture-owned parameter ownership declaration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
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
    /// A unit-consumed static module has no declared invocation.
    #[error("architecture pinned parameter has no unit consumer")]
    EmptyStaticConsumers,
    /// A shared pinned module repeats a logical invocation.
    #[error("architecture pinned parameter repeats a unit consumer")]
    DuplicateStaticConsumer,
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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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

    /// Copies a finite borrowed role and symbolic dimensions into counted storage.
    pub fn new_with_metadata(role:&str,shape:&[BoundaryTensorDimension],dtype:BoundaryTensorDtype,
        context:&eredu_nn::workspace::WorkspaceContext)->Result<Self,eredu_nn::Error>{
        let destination=boundary_construction::Destination(Some(context));
        destination.controls::<Self>().map_err(|cause|cause.metadata(context))?;
        let role=destination.text(role).map_err(|cause|cause.metadata(context))?;
        let mut copied=destination.vector(shape.len()).map_err(|cause|cause.metadata(context))?;
        copied.extend_from_slice(shape);
        Ok(Self {role,shape:copied,dtype})
    }

    /// Declares the standard evolving batch/sequence/hidden activation.
    pub fn primary_activation(hidden_size: i32) -> Self {
        Self::new(
            "hidden",
            [
                BoundaryTensorDimension::Batch,
                BoundaryTensorDimension::Sequence,
                BoundaryTensorDimension::Fixed(hidden_size),
            ],
            BoundaryTensorDtype::Activation,
        )
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

/// Complete primary and ordered auxiliary wire schema for one architecture boundary.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct BoundaryWireSchema {
    identity: &'static str,
    primary: BoundaryTensorSpec,
    auxiliary: Vec<BoundaryTensorSpec>,
}

impl BoundaryWireSchema {
    /// Creates and validates an architecture-owned wire schema.
    pub fn new(
        identity: &'static str,
        primary: BoundaryTensorSpec,
        auxiliary: impl IntoIterator<Item = BoundaryTensorSpec>,
    ) -> Result<Self, ArchitectureBoundaryError> {
        if identity.trim().is_empty(){return Err(ArchitectureBoundaryError::EmptyIdentity);}
        if primary.dtype!=BoundaryTensorDtype::Activation {
            return Err(ArchitectureBoundaryError::InvalidPrimaryDtype{boundary:identity});
        }
        boundary_construction::construct(identity, primary, auxiliary.into_iter().collect(),
            boundary_construction::Destination(None)).map_err(boundary_construction::Failure::ordinary)
    }

    /// Moves declarations from their already paid producers and validates them
    /// through the same worker as `new`. No source or native authority is minted.
    pub fn from_owned_with_metadata(identity: &'static str, primary: BoundaryTensorSpec,
        auxiliary: Vec<BoundaryTensorSpec>, context: &eredu_nn::workspace::WorkspaceContext)
        -> Result<Self, eredu_nn::Error> {
        boundary_construction::construct(identity, primary, auxiliary,
            boundary_construction::Destination(Some(context)))
            .map_err(|cause|cause.metadata(context))
    }

    /// Returns the stable schema identity.
    pub const fn identity(&self) -> &'static str {
        self.identity
    }

    /// Returns the primary evolving activation declaration.
    pub const fn primary(&self) -> &BoundaryTensorSpec {
        &self.primary
    }

    /// Returns auxiliary tensor declarations in canonical transport order.
    pub fn auxiliary(&self) -> &[BoundaryTensorSpec] {
        &self.auxiliary
    }

    /// Resolves invocation-dependent dimensions without backend family logic.
    pub fn resolve(
        &self,
        batch_size: i32,
        sequence_length: i32,
    ) -> Result<ResolvedBoundaryWireSchema, ArchitectureBoundaryError> {
        self.resolve_each(
            batch_size,
            std::iter::repeat_n(sequence_length, 1 + self.auxiliary.len()),
        )
    }

    /// Resolves one exact sequence extent per primary/auxiliary tensor.
    ///
    /// This is used by composite boundaries whose evolving internal activation
    /// and learned side outputs have different sequence geometries. The family
    /// supplies values in canonical schema order; the runtime only validates and
    /// substitutes the declared symbolic dimensions.
    pub fn resolve_each(
        &self,
        batch_size: i32,
        sequence_lengths: impl IntoIterator<Item = i32>,
    ) -> Result<ResolvedBoundaryWireSchema, ArchitectureBoundaryError> {
        let sequence_lengths = sequence_lengths.into_iter().collect::<Vec<_>>();
        boundary_construction::resolve(self, batch_size, &sequence_lengths,
            boundary_construction::Destination(None)).map_err(boundary_construction::Failure::ordinary)
    }

    /// Resolves the same symbolic dimensions using counted role and shape destinations.
    pub fn resolve_with_metadata(&self, batch_size:i32, sequence_length:i32,
        context:&eredu_nn::workspace::WorkspaceContext)->Result<ResolvedBoundaryWireSchema,eredu_nn::Error>{
        let count=self.auxiliary.len().checked_add(1)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        let mut sequences=context.metadata_vec(count)?;
        sequences.resize(count,sequence_length);
        self.resolve_each_with_metadata(batch_size,&sequences,context)
    }

    /// Resolves an exact borrowed sequence extent per primary/auxiliary tensor.
    pub fn resolve_each_with_metadata(&self,batch_size:i32,sequence_lengths:&[i32],
        context:&eredu_nn::workspace::WorkspaceContext)->Result<ResolvedBoundaryWireSchema,eredu_nn::Error>{
        boundary_construction::resolve(self,batch_size,sequence_lengths,
            boundary_construction::Destination(Some(context))).map_err(|cause|cause.metadata(context))
    }

}

/// One architecture boundary after invocation-dependent dimensions are resolved.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct ResolvedBoundaryWireSchema {
    identity: &'static str,
    primary: ResolvedBoundaryTensorSpec,
    auxiliary: Vec<ResolvedBoundaryTensorSpec>,
}

impl ResolvedBoundaryWireSchema {
    /// Copies this validated concrete schema into admitted metadata destinations.
    pub fn clone_with_metadata(&self,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<Self,eredu_nn::Error> {
        context.charge_metadata(std::mem::size_of::<(Self,&Self,ResolvedBoundaryTensorSpec,
            Vec<ResolvedBoundaryTensorSpec>,String,Vec<i32>)>())?;
        let copy=|value:&ResolvedBoundaryTensorSpec|->Result<ResolvedBoundaryTensorSpec,eredu_nn::Error>{
            let role=context.metadata_string(format_args!("{}",value.role))?;
            let mut shape=context.metadata_vec(value.shape.len())?;shape.extend_from_slice(&value.shape);
            Ok(ResolvedBoundaryTensorSpec{role,shape,dtype:value.dtype})
        };
        context.charge_metadata(std::mem::size_of_val(&copy))?;
        let primary=copy(&self.primary)?;
        let mut auxiliary=context.metadata_vec(self.auxiliary.len())?;
        for value in &self.auxiliary {auxiliary.push(copy(value)?);}
        Ok(Self{identity:self.identity,primary,auxiliary})
    }

    /// Returns the stable schema identity.
    pub const fn identity(&self) -> &'static str {
        self.identity
    }

    /// Returns the resolved primary evolving activation declaration.
    pub const fn primary(&self) -> &ResolvedBoundaryTensorSpec {
        &self.primary
    }

    /// Returns resolved auxiliary declarations in canonical transport order.
    pub fn auxiliary(&self) -> &[ResolvedBoundaryTensorSpec] {
        &self.auxiliary
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

    /// Primary evolving activation declaration.
    fn primary_tensor_spec(&self) -> BoundaryTensorSpec;

    /// Auxiliary tensor declarations in exact encoded order.
    fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec>;

    /// Consumes this typed value into exact role-tagged transport tensors.
    ///
    /// Roles are assigned while the architecture-owned typed value is
    /// decomposed; neutral execution must never reconstruct them positionally.
    fn encode<T>(
        &self,
        boundary: Self::Boundary<T>,
    ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError>;

    /// Reconstructs the typed value from transport-order tensors.
    fn decode<T>(&self, tensors: Vec<T>) -> Result<Self::Boundary<T>, ArchitectureBoundaryError>;

    /// Produces this actual schema using the caller's metadata destination.
    /// A custom schema must qualify its own finite producer before original execution.
    fn wire_schema_with_metadata(&self, context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<BoundaryWireSchema,eredu_nn::Error>{
        Err(context.metadata_source(eredu_nn::workspace::WorkspaceMetadataError::Unqualified))
    }

    /// Decomposes the same typed auxiliary value into paid role-tagged storage.
    fn encode_with_metadata<T>(&self,boundary:Self::Boundary<T>,
        context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<Vec<ArchitectureBoundaryValue<T>>,eredu_nn::Error>{
        let _=boundary;
        Err(context.metadata_source(eredu_nn::workspace::WorkspaceMetadataError::Unqualified))
    }

    /// Reconstructs this exact typed auxiliary value using a qualified producer.
    fn decode_with_metadata<T>(&self,tensors:Vec<T>,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<Self::Boundary<T>,eredu_nn::Error>{
        let _=tensors;
        Err(context.metadata_source(eredu_nn::workspace::WorkspaceMetadataError::Unqualified))
    }

    /// Returns the validated backend-neutral wire schema.
    fn wire_schema(&self) -> Result<BoundaryWireSchema, ArchitectureBoundaryError> {
        BoundaryWireSchema::new(
            Self::IDENTITY,
            self.primary_tensor_spec(),
            self.auxiliary_tensor_specs(),
        )
    }
}

/// One architecture-tagged auxiliary boundary value.
///
/// The semantic role is assigned while the family-owned typed boundary is
/// decomposed. Keeping it coupled to the tensor prevents a neutral executor
/// from silently reassigning roles by positional zipping.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ArchitectureBoundaryValue<T> {
    role: String,
    tensor: T,
}

impl<T> ArchitectureBoundaryValue<T> {
    /// Couples a non-empty architecture role to its exact tensor value.
    pub fn new(role: impl Into<String>, tensor: T) -> Result<Self, ArchitectureBoundaryError> {
        let role = role.into();
        if role.trim().is_empty() {
            return Err(ArchitectureBoundaryError::EmptyTaggedTensorRole);
        }
        Ok(Self { role, tensor })
    }

    /// Copies an exact borrowed role through its paid destination, preserving
    /// the same nonempty-role validation and moving the existing tensor owner.
    pub fn new_with_metadata(role:&str,tensor:T,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<Self,eredu_nn::Error>{
        context.charge_metadata(std::mem::size_of::<(Self,Result<Self,eredu_nn::Error>)>())?;
        let role=context.metadata_string(format_args!("{role}"))?;
        Self::new(role,tensor).map_err(|cause|context.metadata_source(cause))
    }

    /// Architecture-owned semantic role.
    pub fn role(&self) -> &str {
        &self.role
    }

    /// Borrows the exact tensor assigned to this role.
    pub const fn tensor(&self) -> &T {
        &self.tensor
    }

    /// Decomposes this value without cloning the tensor.
    pub fn into_parts(self) -> (String, T) {
        (self.role, self.tensor)
    }
}

/// Explicit declaration that an architecture partition carries no auxiliary
/// tensors across its boundary.
///
/// This marker is preferable to `()` because it still participates in the
/// typed boundary contract and rejects any unexpected transported tensor.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct NoAuxiliaryBoundary;

/// Schema for an evolving decoder activation with no auxiliary tensors.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct NoAuxiliaryBoundarySchema {
    hidden_size: i32,
}

impl NoAuxiliaryBoundarySchema {
    /// Declares a standard batch/sequence/hidden activation boundary.
    pub const fn new(hidden_size: i32) -> Self {
        Self { hidden_size }
    }
}

impl ArchitectureBoundary for NoAuxiliaryBoundarySchema {
    type Boundary<T> = NoAuxiliaryBoundary;

    const IDENTITY: &'static str = "none";

    fn wire_schema_with_metadata(&self,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<BoundaryWireSchema,eredu_nn::Error>{
        let primary=BoundaryTensorSpec::new_with_metadata("hidden",&[
            BoundaryTensorDimension::Batch,BoundaryTensorDimension::Sequence,
            BoundaryTensorDimension::Fixed(self.hidden_size)],BoundaryTensorDtype::Activation,context)?;
        let auxiliary=context.metadata_vec(0)?;
        BoundaryWireSchema::from_owned_with_metadata(Self::IDENTITY,primary,auxiliary,context)
    }

    fn encode_with_metadata<T>(&self,boundary:Self::Boundary<T>,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<Vec<ArchitectureBoundaryValue<T>>,eredu_nn::Error>{
        context.charge_metadata(std::mem::size_of::<(NoAuxiliaryBoundary,
            Vec<ArchitectureBoundaryValue<T>>,Result<Vec<ArchitectureBoundaryValue<T>>,eredu_nn::Error>)>())?;
        self.encode::<T>(boundary).map_err(|cause|context.metadata_source(cause))
    }

    fn decode_with_metadata<T>(&self,tensors:Vec<T>,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<Self::Boundary<T>,eredu_nn::Error>{
        context.charge_metadata(std::mem::size_of::<(Vec<T>,Result<NoAuxiliaryBoundary,eredu_nn::Error>)>())?;
        // The same schema validation precedes cardinality on the ordinary path.
        let schema=self.wire_schema_with_metadata(context)?;
        boundary_construction::validate_count(Self::IDENTITY,schema.auxiliary.len(),tensors.len())
            .map_err(|cause|context.metadata_source(cause))?;
        Ok(NoAuxiliaryBoundary)
    }

    fn primary_tensor_spec(&self) -> BoundaryTensorSpec {
        BoundaryTensorSpec::primary_activation(self.hidden_size)
    }

    fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
        Vec::new()
    }

    fn encode<T>(
        &self,
        _boundary: Self::Boundary<T>,
    ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError> {
        Ok(Vec::new())
    }

    fn decode<T>(&self, tensors: Vec<T>) -> Result<Self::Boundary<T>, ArchitectureBoundaryError> {
        validate_boundary_tensor_count(self, &tensors)?;
        Ok(NoAuxiliaryBoundary)
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
    let expected = boundary.wire_schema()?.auxiliary().len();
    boundary_construction::validate_count(B::IDENTITY, expected, tensors.len())
}

/// Invalid architecture-owned partition boundary declaration or payload.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ArchitectureBoundaryError {
    /// A family boundary omitted its stable identity.
    #[error("architecture boundary identity must not be empty")]
    EmptyIdentity,
    /// A role-tagged family value omitted its semantic identity.
    #[error("architecture boundary value contains an empty tensor role")]
    EmptyTaggedTensorRole,
    /// A family boundary assigned a non-activation dtype to its primary tensor.
    #[error("architecture boundary {boundary:?} primary tensor must use activation dtype")]
    InvalidPrimaryDtype {
        /// Stable boundary identity.
        boundary: &'static str,
    },
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
    replicated_static_roles: Vec<String>,
}

impl PartitionOwnership {
    /// Auxiliary storage dependencies, distinct from ordinary invocation roles.
    pub fn replicated_static_roles(&self) -> &[String] {
        &self.replicated_static_roles
    }

    /// Whether a static parameter is materialized here, including shared replicas.
    pub fn stores_static_role(&self, role: &str) -> bool {
        self.owns_static_role(role)
            || self
                .replicated_static_roles
                .iter()
                .any(|stored| stored == role)
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

    /// Derives prompt-cache identity from this canonical state partition.
    pub fn prompt_cache_identity<B, M>(
        &self,
        architecture: &M,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, ArchitecturePartitionError>
    where
        B: eredu_nn::NeuralBackend,
        M: crate::ArchitectureParameters<B>,
        M::DefinitionError: std::fmt::Display,
    {
        architecture
            .state_identity(self, topology, None)
            .map_err(|error| ArchitecturePartitionError::ArchitectureState(error.to_string()))?
            .prompt_cache_identity(self.layout())
            .map_err(|error| ArchitecturePartitionError::PromptCacheIdentity(error.to_string()))
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
/// primary and auxiliary wire schema carried by the partition realization.
#[derive(Debug, Clone)]
pub struct ArchitecturePartition<G, A> {
    graph: ExecutionGraph,
    unit_layout: ExecutionUnitLayout,
    groups: Vec<PartitionGroup>,
    ownership: PartitionOwnership,
    state: Option<PartitionState>,
    local_geometry: G,
    boundary_schema: A,
    parameter_bindings: Vec<OwnedParameterGroupSpec>,
}

impl<G, A> ArchitecturePartition<G, A> {
    /// Creates a partition from the topology declared by one concrete neutral
    /// architecture.
    ///
    /// This constructor derives the graph and unit layout from `architecture`,
    /// preventing a backend realization from publishing a parallel topology
    /// that merely resembles, but is not the canonical topology of, the
    /// architecture it will execute. Pre-allocation selection should use
    /// [`Self::from_description`] with the architecture-authored declaration.
    #[allow(clippy::too_many_arguments)]
    pub fn from_architecture<B, S, M, N>(
        architecture: &M,
        group_ranges: impl IntoIterator<Item = (N, Range<usize>)>,
        ownership: PartitionOwnership,
        local_geometry: G,
        boundary_schema: A,
        parameters: &ArchitectureParameterDescription,
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
        boundary_schema.wire_schema()?;
        if parameters.graph() != &graph {
            return Err(ArchitecturePartitionError::ArchitectureGraphMismatch);
        }
        if parameters.unit_layout() != &unit_layout {
            return Err(ArchitecturePartitionError::ArchitectureUnitLayoutMismatch);
        }
        let complete_state = architecture
            .state_layout(None)
            .map_err(|error| ArchitecturePartitionError::ArchitectureState(error.to_string()))?;
        let plan = architecture.state_partition_plan(&complete_state);
        let mut partition = Self::new(
            graph,
            unit_layout,
            group_ranges,
            ownership,
            None,
            local_geometry,
            boundary_schema,
            std::iter::empty(),
        )?;
        partition.state = partition
            .resolve_state_partition(&complete_state, &plan)
            .map_err(|error| ArchitecturePartitionError::ArchitectureState(error.to_string()))?;
        partition.parameter_bindings = parameters.select_owned(&partition);
        Ok(partition)
    }

    /// Creates an authoritative partition from architecture-authored static declarations.
    ///
    /// This path exists for pre-materialization selection: it validates and consumes the
    /// canonical graph/unit declaration, state plan, boundary schema, local geometry, and
    /// parameter topology without constructing a backend module merely to rediscover those
    /// facts. Concrete backends must not synthesize any of these inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn from_description<N>(
        parameters: &ArchitectureParameterDescription,
        group_ranges: impl IntoIterator<Item = (N, Range<usize>)>,
        ownership: PartitionOwnership,
        complete_state: &StateLayout,
        state_plan: &ArchitectureStatePartitionPlan,
        local_geometry: G,
        boundary_schema: A,
    ) -> Result<Self, ArchitecturePartitionError>
    where
        N: Into<String>,
        A: ArchitectureBoundary,
    {
        let graph = parameters.graph().clone();
        let unit_layout = parameters.unit_layout().clone();
        validate_canonical_layout(&graph, &unit_layout)?;
        boundary_schema.wire_schema()?;
        let mut partition = Self::new(
            graph,
            unit_layout,
            group_ranges,
            ownership,
            None,
            local_geometry,
            boundary_schema,
            std::iter::empty(),
        )?;
        partition.state = partition
            .resolve_state_partition(complete_state, state_plan)
            .map_err(|error| ArchitecturePartitionError::ArchitectureState(error.to_string()))?;
        partition.parameter_bindings = parameters.select_owned(&partition);
        Ok(partition)
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
        boundary_schema: A,
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
            boundary_schema,
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

    /// Derives prompt-cache identity from this partition's canonical state.
    pub fn prompt_cache_identity<B, M>(
        &self,
        architecture: &M,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, ArchitecturePartitionError>
    where
        B: eredu_nn::NeuralBackend,
        M: crate::ArchitectureParameters<B>,
        M::DefinitionError: std::fmt::Display,
    {
        let state = self
            .state()
            .ok_or(ArchitecturePartitionError::MissingArchitectureState)?;
        state.prompt_cache_identity::<B, M>(architecture, topology)
    }

    /// Resolves the architecture-authored state plan for this realized partition.
    ///
    /// The current partition representation stores one contiguous global state
    /// interval. A valid plan may describe multiple semantic ranges, but the
    /// ranges selected by any one partition must be adjacent.
    pub fn resolve_state_partition(
        &self,
        complete: &StateLayout,
        plan: &ArchitectureStatePartitionPlan,
    ) -> Result<Option<PartitionState>, ArchitectureStatePartitionError> {
        if plan.rules().is_empty() {
            return Err(ArchitectureStatePartitionError::EmptyPlan);
        }

        let mut rules = plan.rules().iter().collect::<Vec<_>>();
        rules.sort_by_key(|rule| rule.layers().start);
        let mut frontier = 0usize;
        for rule in &rules {
            let layers = rule.layers();
            if layers.is_empty() {
                return Err(ArchitectureStatePartitionError::EmptyRange {
                    start: layers.start,
                    end: layers.end,
                });
            }
            if layers.end > complete.len() {
                return Err(ArchitectureStatePartitionError::RangeOutOfBounds {
                    start: layers.start,
                    end: layers.end,
                    layers: complete.len(),
                });
            }
            if layers.start < frontier {
                return Err(ArchitectureStatePartitionError::OverlappingRange {
                    start: layers.start,
                    frontier,
                });
            }
            if layers.start > frontier {
                return Err(ArchitectureStatePartitionError::UnassignedLayer { layer: frontier });
            }
            if let ArchitectureStatePlacement::GroupUnits { group } = rule.placement() {
                let units = self
                    .unit_layout
                    .group_range(group)
                    .ok_or(ArchitectureStatePartitionError::UnknownGroup { group })?
                    .len();
                if layers.len() != units {
                    return Err(ArchitectureStatePartitionError::GroupLengthMismatch {
                        group,
                        start: layers.start,
                        end: layers.end,
                        units,
                    });
                }
            }
            frontier = layers.end;
        }
        if frontier != complete.len() {
            return Err(ArchitectureStatePartitionError::UnassignedLayer { layer: frontier });
        }

        let mut selected = Vec::new();
        for rule in plan.rules() {
            let layers = rule.layers();
            match rule.placement() {
                ArchitectureStatePlacement::GroupUnits { group } => {
                    if let Some(owned) = self
                        .groups
                        .iter()
                        .find(|owned| owned.group_index() == group)
                    {
                        let units = owned.global_units();
                        selected.push(layers.start + units.start..layers.start + units.end);
                    }
                }
                ArchitectureStatePlacement::OutputOwner if self.ownership.owns_output() => {
                    selected.push(layers);
                }
                ArchitectureStatePlacement::OutputOwner => {}
            }
        }
        if selected.is_empty() {
            return Ok(None);
        }
        selected.sort_by_key(|layers| layers.start);
        let start = selected[0].start;
        let mut end = selected[0].end;
        for layers in selected.iter().skip(1) {
            if layers.start != end {
                return Err(ArchitectureStatePartitionError::DiscontiguousSelection {
                    frontier: end,
                    start: layers.start,
                });
            }
            end = layers.end;
        }
        let layout = complete
            .slice(start..end)
            .map_err(|error| ArchitectureStatePartitionError::InvalidLayout(error.to_string()))?;
        PartitionState::new(layout, start)
            .map(Some)
            .map_err(|error| ArchitectureStatePartitionError::InvalidLayout(error.to_string()))
    }

    /// Returns family-owned rank-local construction geometry.
    pub const fn local_geometry(&self) -> &G {
        &self.local_geometry
    }

    /// Returns the family-owned primary and auxiliary boundary schema.
    pub const fn boundary_schema(&self) -> &A {
        &self.boundary_schema
    }

    /// Mutably returns the family-owned primary and auxiliary boundary schema.
    pub fn boundary_schema_mut(&mut self) -> &mut A {
        &mut self.boundary_schema
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
    state_layout: Option<StateLayout>,
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
        Self::new_with_state_ownership(partition, group_index, storage_range, true)
    }

    /// Validates a partition while explicitly declaring whether this group owns state slots.
    ///
    /// Architecture selection must pass `false` for parameter-only composite roots. A rank can
    /// still own decoder state for another local group without falsely comparing that state range
    /// with this group's unrelated unit indices.
    pub fn new_with_state_ownership<G, A>(
        partition: &ArchitecturePartition<G, A>,
        group_index: usize,
        storage_range: Range<usize>,
        group_owns_state: bool,
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
        if group_owns_state {
            let state = partition
                .state()
                .ok_or(LayeredPartitionError::MissingState)?;
            if state.global_layers().start > range.start || state.global_layers().end < range.end {
                return Err(LayeredPartitionError::StateRange {
                    state: state.global_layers(),
                    partition: range,
                });
            }
        }
        Ok(Self {
            group: group.group_index(),
            range,
            state_layout: group_owns_state
                .then(|| partition.state().map(|state| state.layout().clone()))
                .flatten(),
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

    /// Returns state geometry for a driver created through the strict stateful constructor.
    pub fn state_layout(&self) -> &StateLayout {
        self.state_layout
            .as_ref()
            .expect("state_layout requires a state-owning layered partition driver")
    }

    /// Returns rank-local state geometry, or `None` for stateless roots and ranks.
    pub const fn optional_state_layout(&self) -> Option<&StateLayout> {
        self.state_layout.as_ref()
    }

    /// Returns whether this partition receives request ingress directly.
    pub const fn owns_input(&self) -> bool {
        self.owns_input
    }

    /// Returns whether this partition owns architecture output projection.
    pub const fn owns_output(&self) -> bool {
        self.owns_output
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

    /// Transports this partition's realized boundary through the selected
    /// opaque collective group.
    ///
    /// Keeping the operation on the validated driver makes boundary movement
    /// part of partition execution rather than an unrelated backend call.
    pub fn exchange_boundary<B>(
        &self,
        value: B::Tensor,
        group: &B::Group,
        executor: &B::Executor,
    ) -> Result<B::Tensor, B::CollectiveError>
    where
        B: crate::CollectiveBackend,
    {
        B::all_to_all(value, group, executor)
    }

    /// Prepares the partition and starts its canonical execution group.
    #[allow(
        clippy::too_many_arguments,
        clippy::type_complexity,
        reason = "the result preserves the concrete architecture error without erased dispatch"
    )]
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
    ) -> Result<
        LayeredForwardState<B::Tensor, M::ForwardContext>,
        LayeredPartitionBeginError<M::Error>,
    >
    where
        B: eredu_nn::NeuralBackend,
        S: RuntimeState<B>,
        M: PartitionedLayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
    {
        self.begin_with_optional_observer(architecture, input, mask, state, parallel, context, None)
    }

    /// Prepares the partition with internal input observations before group entry.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn begin_observed<'a, B, S, M, O>(
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
        observer: &mut O,
    ) -> Result<
        LayeredForwardState<B::Tensor, M::ForwardContext>,
        LayeredPartitionBeginError<M::Error>,
    >
    where
        B: eredu_nn::NeuralBackend,
        S: RuntimeState<B>,
        M: PartitionedLayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
        O: crate::ActivationObserver<B::Tensor, M::Error> + ?Sized,
    {
        let mut observer = crate::BorrowedActivationObserver(observer);
        self.begin_with_optional_observer(
            architecture,
            input,
            mask,
            state,
            parallel,
            context,
            Some(&mut observer),
        )
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn begin_with_optional_observer<'a, B, S, M>(
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
        observer: Option<&mut dyn crate::ActivationObserver<B::Tensor, M::Error>>,
    ) -> Result<
        LayeredForwardState<B::Tensor, M::ForwardContext>,
        LayeredPartitionBeginError<M::Error>,
    >
    where
        B: eredu_nn::NeuralBackend,
        S: RuntimeState<B>,
        M: PartitionedLayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
    {
        let expected = self
            .state_layout
            .as_ref()
            .ok_or(LayeredPartitionBeginError::MissingState { group: self.group })?;
        // `state` is the partition-local allocation selected by `PartitionState`.
        // Global ownership is carried separately by that partition's offset, so
        // architecture code must index this allocation from local ordinal zero.
        let mut forward = match observer.filter(|observer| observer.observes_activations()) {
            Some(observer) => architecture.begin_partition_observed(
                input, mask, state, expected, 0, parallel, context, observer,
            ),
            None => match parallel {
                Some(parallel) => architecture
                    .begin_partition_parallel(input, mask, state, expected, 0, parallel, context),
                None => architecture.begin_partition(input, mask, state, expected, 0, context),
            },
        }
        .map_err(LayeredPartitionBeginError::Architecture)?;
        forward.hidden = architecture
            .enter_partition_group(
                self.group,
                &forward.hidden,
                state,
                &mut forward.context,
                parallel,
                context,
            )
            .map_err(LayeredPartitionBeginError::Architecture)?;
        Ok(forward)
    }

    /// Completes the canonical group and applies output projection only on its owner.
    #[allow(
        clippy::too_many_arguments,
        clippy::type_complexity,
        reason = "the signature exposes the backend and architecture boundary types explicitly"
    )]
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
        self.finish_with_optional_observer(
            architecture,
            hidden,
            state,
            forward,
            parallel,
            context,
            None,
            eredu_core::OutputDemand::Sequence,
        )
    }

    /// Completes the group and observes readout only at the selected output owner.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn finish_observed<B, S, M, O>(
        &self,
        architecture: &mut M,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut M::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
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
        O: crate::ActivationObserver<B::Tensor, M::Error> + ?Sized,
    {
        self.finish_observed_with_readout(
            architecture,
            hidden,
            state,
            forward,
            parallel,
            context,
            observer,
            eredu_core::OutputDemand::Sequence,
        )
    }

    /// Completes the same partition with explicit vocabulary demand.
    pub fn finish_observed_with_readout<B, S, M, O>(
        &self,
        architecture: &mut M,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut M::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
        demand: eredu_core::OutputDemand,
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
        O: crate::ActivationObserver<B::Tensor, M::Error> + ?Sized,
    {
        let mut observer = crate::BorrowedActivationObserver(observer);
        self.finish_with_optional_observer(
            architecture,
            hidden,
            state,
            forward,
            parallel,
            context,
            Some(&mut observer),
            demand,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn finish_with_optional_observer<B, S, M>(
        &self,
        architecture: &mut M,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut M::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: Option<&mut dyn crate::ActivationObserver<B::Tensor, M::Error>>,
        demand: eredu_core::OutputDemand,
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
        match observer {
            Some(observer) => architecture.finish_partition_with_readout(
                &hidden,
                state,
                forward,
                self.owns_output,
                parallel,
                context,
                observer,
                demand,
            ),
            None => architecture.finish_partition_with_readout(
                &hidden,
                state,
                forward,
                self.owns_output,
                parallel,
                context,
                &mut crate::NoopObserver,
                demand,
            ),
        }
    }
}

/// Failure to enter one concrete rank-local partition group.
#[derive(Debug, thiserror::Error)]
pub enum LayeredPartitionBeginError<E>
where
    E: std::fmt::Display,
{
    /// This group was declared stateless and cannot use the stateful partition entry API.
    #[error("stateless partition group {group} requires an architecture stateless entry strategy")]
    MissingState {
        /// Canonical architecture group slot.
        group: usize,
    },
    /// Architecture-owned partition entry failed.
    #[error("partition architecture entry failed: {0}")]
    Architecture(E),
}

/// Invalid concrete realization or boundary use of a layered partition.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
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
        .map_err(|error| ArchitecturePartitionError::ArchitectureTopology(error.to_string()))?.into_owned();
    let primary = architecture.primary_execution_group();
    let primary_index = graph.group_index(primary).ok_or_else(|| {
        ArchitecturePartitionError::ArchitectureTopology(format!(
            "primary execution group {primary:?} is not present in the canonical graph"
        ))
    })?;
    let primary_transport = architecture.group_transport(primary_index);
    if primary_transport.kind != crate::ArchitectureGroupKind::Decoder
        || primary_transport.placement != crate::ArchitectureGroupPlacement::Pipeline
    {
        return Err(ArchitecturePartitionError::ArchitectureTopology(format!(
            "primary execution group {primary:?} must be a pipeline decoder"
        )));
    }
    let mut declared_groups = BTreeSet::from([primary.to_owned()]);
    for prediction in architecture.prediction_execution_groups() {
        let prediction_index = graph.group_index(&prediction).ok_or_else(|| {
            ArchitecturePartitionError::ArchitectureTopology(format!(
                "prediction execution group {prediction:?} is not present in the canonical graph"
            ))
        })?;
        let prediction_transport = architecture.group_transport(prediction_index);
        if prediction_transport.kind != crate::ArchitectureGroupKind::Prediction
            || prediction_transport.placement != crate::ArchitectureGroupPlacement::OutputOwner
        {
            return Err(ArchitecturePartitionError::ArchitectureTopology(format!(
                "prediction execution group {prediction:?} must be an output-owner prediction"
            )));
        }
        if !declared_groups.insert(prediction.clone()) {
            return Err(ArchitecturePartitionError::ArchitectureTopology(format!(
                "execution group {prediction:?} is declared as a primary or prediction group more than once"
            )));
        }
    }
    let mut counts = Vec::with_capacity(graph.groups().len());
    let mut paths = BTreeSet::new();
    for group in 0..graph.groups().len() {
        let count = architecture
            .group_unit_count(group, None)
            .map_err(|error| ArchitecturePartitionError::ArchitectureTopology(error.to_string()))?;
        counts.push(count);
        for index in 0..count {
            let path = architecture.unit_path(group, index, None).map_err(|error| {
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
    description::canonical(graph, layout).map_err(description::LayoutIssue::into_owned)
}

/// Invalid backend-neutral architecture partition declaration.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ArchitecturePartitionError {
    /// The actual source account refused a metadata producer.
    #[error("{0}")]
    MetadataFunding(#[from] eredu_core::HostMetadataFundingError),
    /// The host allocator refused an admitted metadata request.
    #[error("partition metadata allocation: {0}")]
    MetadataAllocation(#[from] std::collections::TryReserveError),
    /// The selected allocator returned an unqualified capacity.
    #[error("partition metadata capacity differs from its request")]
    MetadataCapacity,

    /// The architecture supplied an invalid partition-boundary wire schema.
    #[error("invalid architecture partition boundary: {0}")]
    InvalidBoundary(#[from] ArchitectureBoundaryError),
    /// The neutral architecture could not declare a canonical graph, unit
    /// count, or unit path.
    #[error("neutral architecture topology is invalid: {0}")]
    ArchitectureTopology(String),
    /// The architecture could not declare or partition its mutable state.
    #[error("neutral architecture state is invalid: {0}")]
    ArchitectureState(String),
    /// The realized partition owns no mutable architecture state.
    #[error("architecture partition owns no mutable state")]
    MissingArchitectureState,
    /// The architecture state could not be converted to prompt-cache identity.
    #[error("architecture prompt-cache identity is invalid: {0}")]
    PromptCacheIdentity(String),
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

        fn primary_tensor_spec(&self) -> BoundaryTensorSpec {
            BoundaryTensorSpec::primary_activation(8)
        }

        fn auxiliary_tensor_specs(&self) -> Vec<BoundaryTensorSpec> {
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
        ) -> Result<Vec<ArchitectureBoundaryValue<T>>, ArchitectureBoundaryError> {
            Ok(vec![
                ArchitectureBoundaryValue::new("tokens", boundary.tokens)?,
                ArchitectureBoundaryValue::new("embedded", boundary.embedded)?,
            ])
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

    fn state_plan_partition(
        primary: Range<usize>,
        ownership: PartitionOwnership,
    ) -> ArchitecturePartition<(), ()> {
        let graph = graph();
        ArchitecturePartition::new(
            graph.clone(),
            layout(&graph),
            [("primary", primary)],
            ownership,
            None,
            (),
            (),
            [],
        )
        .unwrap()
    }

    #[test]
    fn architecture_state_plan_attaches_declared_tail_to_output_owner() {
        let complete = state_layout(6);
        let plan = ArchitectureStatePartitionPlan::new([
            crate::ArchitectureStatePartitionRule::group_units(0, 0..4),
            crate::ArchitectureStatePartitionRule::output_owner(4..6),
        ]);
        let interior = state_plan_partition(
            1..3,
            PartitionOwnership::new(false, false, std::iter::empty::<&str>()).unwrap(),
        );
        let output = state_plan_partition(
            3..4,
            PartitionOwnership::new(false, true, std::iter::empty::<&str>()).unwrap(),
        );

        assert_eq!(
            interior
                .resolve_state_partition(&complete, &plan)
                .unwrap()
                .unwrap()
                .global_layers(),
            1..3
        );
        assert_eq!(
            output
                .resolve_state_partition(&complete, &plan)
                .unwrap()
                .unwrap()
                .global_layers(),
            3..6
        );
    }

    #[test]
    fn architecture_state_plan_rejects_noncontiguous_local_state() {
        let complete = state_layout(6);
        let plan = ArchitectureStatePartitionPlan::new([
            crate::ArchitectureStatePartitionRule::output_owner(0..2),
            crate::ArchitectureStatePartitionRule::group_units(0, 2..6),
        ]);
        let output = state_plan_partition(
            3..4,
            PartitionOwnership::new(false, true, std::iter::empty::<&str>()).unwrap(),
        );

        assert_eq!(
            output.resolve_state_partition(&complete, &plan),
            Err(ArchitectureStatePartitionError::DiscontiguousSelection {
                frontier: 2,
                start: 5,
            })
        );
    }

    fn parameter_description(
        expected: Vec<ParameterGroupSpec>,
        groups: Vec<OwnedParameterGroupSpec>,
    ) -> Result<ArchitectureParameterDescription, ArchitectureParameterError> {
        let graph = graph();
        ArchitectureParameterDescription::new(&graph, &layout(&graph), expected, groups)
    }

    #[test]
    fn description_driven_partition_selects_state_and_parameters_before_construction() {
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
        let ownership = PartitionOwnership::new(true, false, ["embedding"]).unwrap();
        let state = state_layout(4);
        let state_plan = ArchitectureStatePartitionPlan::new([
            crate::ArchitectureStatePartitionRule::group_units(0, 0..4),
        ]);

        let partition = ArchitecturePartition::from_description(
            &description,
            [("primary", 1..3)],
            ownership,
            &state,
            &state_plan,
            Geometry("selected-before-allocation"),
            PairBoundarySchema,
        )
        .unwrap();

        assert_eq!(partition.groups()[0].global_units(), 1..3);
        assert_eq!(partition.state().unwrap().global_layers(), 1..3);
        assert_eq!(
            partition.local_geometry(),
            &Geometry("selected-before-allocation")
        );
        assert_eq!(partition.parameter_bindings().len(), 2);
        assert_eq!(
            partition
                .parameter_bindings()
                .iter()
                .flat_map(|group| group.members())
                .map(ParameterMemberSpec::target)
                .collect::<Vec<_>>(),
            ["model.embed_tokens.weight", "model.layers.1.weight"]
        );
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
    fn pinned_unit_consumers_select_one_copy_and_canonical_binding_role() {
        let primary = ExecutionGroupId::new("primary").unwrap();
        let shared = parameter("shared", "shared.weight");
        let owner = ParameterGroupOwner::static_unit_consumers(
            "projector",
            [(primary.clone(), 1), (primary.clone(), 2)],
        );
        let description = parameter_description(
            vec![shared.clone()],
            vec![OwnedParameterGroupSpec::new(owner.clone(), shared.clone())],
        )
        .unwrap();
        let partition = valid_partition();
        assert!(!partition.ownership().owns_static_role("projector"));
        assert_eq!(description.select_owned(&partition).len(), 1);
        assert_eq!(description.select_static_roles(&partition), ["projector"]);
        assert!(owner.is_owned_by(partition.ownership(), |group, unit| {
            partition.owns_unit(group.as_str(), unit)
        }));
        assert!(owner.refines_storage_owner(&ParameterGroupOwner::static_role("projector")));
        assert!(!owner.refines_storage_owner(&ParameterGroupOwner::static_role("embedding")));
        let replica = PartitionOwnership::new(false, false, Vec::<String>::new())
            .unwrap()
            .with_replicated_static_roles(["projector"])
            .unwrap();
        assert!(owner.is_stored_by(&replica, |_, _| false));
        assert!(!owner.is_owned_by(&replica, |_, _| false));
        let unowned = parameter_description(
            vec![shared.clone()],
            vec![OwnedParameterGroupSpec::new(
                ParameterGroupOwner::static_unit_consumers("embedding", [(primary, 0)]),
                shared,
            )],
        )
        .unwrap();
        assert!(partition.ownership().owns_static_role("embedding"));
        assert!(unowned.select_owned(&partition).is_empty());
        assert!(unowned.select_static_roles(&partition).is_empty());
    }

    #[test]
    fn pinned_unit_consumers_validate_exact_graph_invocations() {
        let primary = ExecutionGroupId::new("primary").unwrap();
        let make = |consumers| {
            let shared = parameter("shared", "shared.weight");
            parameter_description(
                vec![shared.clone()],
                vec![OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_unit_consumers("projector", consumers),
                    shared,
                )],
            )
        };
        assert_eq!(
            make(vec![]).unwrap_err(),
            ArchitectureParameterError::EmptyStaticConsumers
        );
        assert_eq!(
            make(vec![(primary.clone(), 1), (primary.clone(), 1)]).unwrap_err(),
            ArchitectureParameterError::DuplicateStaticConsumer
        );
        assert!(matches!(
            make(vec![(primary, usize::MAX)]),
            Err(ArchitectureParameterError::UnitOutOfRange { .. })
        ));
        assert!(matches!(
            make(vec![(ExecutionGroupId::new("absent").unwrap(), 0)]),
            Err(ArchitectureParameterError::UnknownExecutionGroup(_))
        ));
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
    fn auxiliary_static_replicas_materialize_without_owning_target_hooks() {
        let ownership = PartitionOwnership::new(false, true, ["output"])
            .unwrap()
            .with_replicated_static_roles(["embedding"])
            .unwrap();
        let owner = ParameterGroupOwner::static_role("embedding");
        assert!(!ownership.owns_input());
        assert!(!owner.is_owned_by(&ownership, |_, _| false));
        assert!(owner.is_stored_by(&ownership, |_, _| false));
        assert!(ParameterGroupOwner::static_any_of(["embedding", "unused"])
            .is_stored_by(&ownership, |_, _| false));
        let partition = state_plan_partition(1..4, ownership);
        let embedding = parameter("embedding", "model.embed_tokens.weight");
        let description = parameter_description(
            vec![embedding.clone()],
            vec![OwnedParameterGroupSpec::new(owner, embedding)],
        )
        .unwrap();
        let selected = description.select_owned(&partition);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].logical_name(), "embedding");
        assert!(PartitionOwnership::new(false, false, [] as [&str; 0])
            .unwrap()
            .with_replicated_static_roles(["embedding", "embedding"])
            .is_err());
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
        partition.boundary_schema_mut().route = 5;
        assert_eq!(partition.boundary_schema().route, 5);
        assert_eq!(partition.parameter_bindings().len(), 2);
    }

    #[test]
    fn typed_boundary_owns_roles_order_and_atomic_cardinality_validation() {
        let boundary = PairBoundary {
            tokens: 3,
            embedded: 7,
        };
        let schema = PairBoundarySchema;
        let values = schema.encode(boundary).unwrap();
        assert_eq!(values[0].role(), "tokens");
        assert_eq!(values[1].role(), "embedded");
        let tensors = values
            .into_iter()
            .map(ArchitectureBoundaryValue::into_parts)
            .map(|(_, tensor)| tensor)
            .collect();
        assert_eq!(
            schema.decode(tensors).unwrap(),
            PairBoundary {
                tokens: 3,
                embedded: 7
            }
        );
        let resolved = schema.wire_schema().unwrap().resolve(2, 3).unwrap();
        assert_eq!(resolved.primary().shape(), [2, 3, 8]);
        assert_eq!(resolved.primary().dtype(), BoundaryTensorDtype::Activation);
        assert_eq!(resolved.auxiliary()[0].shape(), [2, 3]);
        assert_eq!(resolved.auxiliary()[0].dtype(), BoundaryTensorDtype::Uint32);
        assert_eq!(resolved.auxiliary()[1].shape(), [2, 3, 16]);
        assert_eq!(
            resolved.auxiliary()[1].dtype(),
            BoundaryTensorDtype::Activation
        );
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
        let invalid_primary = BoundaryWireSchema::new(
            "fixture.invalid",
            BoundaryTensorSpec::new(
                "hidden",
                [BoundaryTensorDimension::Fixed(8)],
                BoundaryTensorDtype::Uint32,
            ),
            [],
        )
        .unwrap_err();
        assert_eq!(
            invalid_primary,
            ArchitectureBoundaryError::InvalidPrimaryDtype {
                boundary: "fixture.invalid",
            }
        );

        let duplicate = BoundaryWireSchema::new(
            "fixture.invalid",
            BoundaryTensorSpec::primary_activation(8),
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
            BoundaryTensorSpec::primary_activation(8),
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
    fn layered_driver_represents_stateless_root_without_borrowing_decoder_state() {
        let graph = ExecutionGraph::chain(["vision", "decoder"]).unwrap();
        let layout = ExecutionUnitLayout::new(&graph, [1, 2]).unwrap();
        let partition = ArchitecturePartition::new(
            graph,
            layout,
            [("vision", 0..1), ("decoder", 0..2)],
            PartitionOwnership::new(true, true, std::iter::empty::<String>()).unwrap(),
            Some(PartitionState::new(state_layout(1), 1).unwrap()),
            (),
            (),
            std::iter::empty(),
        )
        .unwrap();

        let vision =
            LayeredPartitionDriver::new_with_state_ownership(&partition, 0, 0..1, false).unwrap();
        assert!(vision.optional_state_layout().is_none());
        assert_eq!(vision.group_index(), 0);

        let without_state = ArchitecturePartition::new(
            ExecutionGraph::chain(["vision"]).unwrap(),
            ExecutionUnitLayout::new(&ExecutionGraph::chain(["vision"]).unwrap(), [1]).unwrap(),
            [("vision", 0..1)],
            PartitionOwnership::new(true, false, std::iter::empty::<String>()).unwrap(),
            None,
            (),
            (),
            std::iter::empty(),
        )
        .unwrap();
        assert_eq!(
            LayeredPartitionDriver::new(&without_state, 0, 0..1).unwrap_err(),
            LayeredPartitionError::MissingState
        );
        assert!(
            LayeredPartitionDriver::new_with_state_ownership(&without_state, 0, 0..1, false)
                .unwrap()
                .optional_state_layout()
                .is_none()
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

#[path = "partition/boundary_construction.rs"]
mod boundary_construction;
