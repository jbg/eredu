//! Opaque communication manifests projected from semantic parallel topology.
//!
//! This module is the ownership seam between neutral topology planning and a
//! concrete communication implementation. Projection may inspect semantic
//! axes, but the descriptors handed to a backend contain only ordered world
//! ranks, opaque identities, operations, and mechanical limits.

use std::{collections::BTreeSet, ops::Range};

use eredu_core::{
    checkpoint::TensorDtype, consensus::ConsensusTransport, CollectiveGroupDescriptor,
    CollectiveGroupId, CompletionCancellationMode, ParallelAxis, ParallelRankTopology,
    ParallelTopology,
};
use serde::{Deserialize, Serialize};

/// Stable opaque identity of one communication group in a selected session.
///
/// This alias preserves the established core identity consumed by backend
/// communicator maps while the runtime manifest adds ordering and requirements.
pub type CommunicationGroupId = CollectiveGroupId;

/// Exact rank-local send and receive element counts for a variable exchange.
///
/// Counts are ordered by opaque communication-group membership. Zero is a
/// valid count for an idle peer; the peer cardinality itself must be positive.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct CommunicationPeerCounts {
    send: Vec<usize>,
    receive: Vec<usize>,
}

impl CommunicationPeerCounts {
    /// Validates one send and receive count for every ordered group member.
    pub fn new(
        send: Vec<usize>,
        receive: Vec<usize>,
        group_size: usize,
    ) -> Result<Self, CommunicationManifestError> {
        if group_size == 0 || send.len() != group_size || receive.len() != group_size {
            return Err(CommunicationManifestError::InvalidPeerCounts {
                group_size,
                send: send.len(),
                receive: receive.len(),
            });
        }
        Ok(Self { send, receive })
    }

    /// Elements submitted to each peer in ordered membership order.
    pub fn send(&self) -> &[usize] {
        &self.send
    }

    /// Elements expected from each peer in ordered membership order.
    pub fn receive(&self) -> &[usize] {
        &self.receive
    }

    /// Ordered communication-group cardinality.
    pub fn group_size(&self) -> usize {
        self.send.len()
    }
}

impl<'de> Deserialize<'de> for CommunicationPeerCounts {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            send: Vec<usize>,
            receive: Vec<usize>,
        }

        let raw = Raw::deserialize(deserializer)?;
        let group_size = raw.send.len();
        Self::new(raw.send, raw.receive, group_size).map_err(serde::de::Error::custom)
    }
}

/// Stable opaque identity of one directed communication route.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CommunicationRouteId(u64);

impl CommunicationRouteId {
    /// Creates an opaque route identity.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the stable numeric representation used by backend maps.
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// One exact mechanism required by a communication group or route.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum CommunicationOperation {
    /// Elementwise sum reduction over an ordered group.
    AllReduceSum,
    /// Equal-size gather with concatenation in group-member order.
    AllGatherEven,
    /// Unequal-size gather with concatenation in group-member order.
    AllGatherUneven,
    /// All-to-all with an exact count supplied for every ordered peer.
    VariableAllToAll,
    /// Directed point-to-point transfer.
    SendReceive,
    /// Root-to-group broadcast.
    Broadcast,
    /// Group synchronization without a tensor payload.
    Barrier,
    /// All-rank boolean conjunction used to propagate a local phase failure.
    FailureAgreement,
}

/// Maximum tensor and per-peer counts admitted for one operation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
pub struct CommunicationTensorLimits {
    max_tensors: usize,
    max_tensor_rank: usize,
    max_tensor_elements: usize,
    max_output_tensor_elements: usize,
    max_count_per_peer: Option<usize>,
}

impl<'de> Deserialize<'de> for CommunicationTensorLimits {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            max_tensors: usize,
            max_tensor_rank: usize,
            max_tensor_elements: usize,
            #[serde(default)]
            max_output_tensor_elements: Option<usize>,
            max_count_per_peer: Option<usize>,
        }

        let raw = Raw::deserialize(deserializer)?;
        let limits = Self::new(
            raw.max_tensors,
            raw.max_tensor_rank,
            raw.max_tensor_elements,
            raw.max_count_per_peer,
        )
        .map_err(serde::de::Error::custom)?;
        match raw.max_output_tensor_elements {
            Some(elements) => limits
                .with_output_tensor_elements(elements)
                .map_err(serde::de::Error::custom),
            None => Ok(limits),
        }
    }
}

impl CommunicationTensorLimits {
    /// Validates positive payload limits. Tensor rank may be zero for scalars.
    pub fn new(
        max_tensors: usize,
        max_tensor_rank: usize,
        max_tensor_elements: usize,
        max_count_per_peer: Option<usize>,
    ) -> Result<Self, CommunicationManifestError> {
        if max_tensors == 0 || max_tensor_elements == 0 || max_count_per_peer == Some(0) {
            return Err(CommunicationManifestError::InvalidOperationLimits);
        }
        Ok(Self {
            max_tensors,
            max_tensor_rank,
            max_tensor_elements,
            max_output_tensor_elements: max_tensor_elements,
            max_count_per_peer,
        })
    }

    /// Selects a distinct upper bound for the completed result tensor.
    ///
    /// Gather and variable-count operations can return more elements than one rank submits.
    pub fn with_output_tensor_elements(
        mut self,
        max_output_tensor_elements: usize,
    ) -> Result<Self, CommunicationManifestError> {
        if max_output_tensor_elements == 0 {
            return Err(CommunicationManifestError::InvalidOperationLimits);
        }
        self.max_output_tensor_elements = max_output_tensor_elements;
        Ok(self)
    }

    /// Maximum tensors submitted in one operation.
    pub const fn max_tensors(self) -> usize {
        self.max_tensors
    }

    /// Maximum logical rank of one tensor. Zero admits scalars only.
    pub const fn max_tensor_rank(self) -> usize {
        self.max_tensor_rank
    }

    /// Maximum logical elements in one tensor.
    pub const fn max_tensor_elements(self) -> usize {
        self.max_tensor_elements
    }

    /// Maximum logical elements in one completed result tensor.
    pub const fn max_output_tensor_elements(self) -> usize {
        self.max_output_tensor_elements
    }

    /// Maximum elements addressed to or received from one peer, when used.
    pub const fn max_count_per_peer(self) -> Option<usize> {
        self.max_count_per_peer
    }

    fn covers(self, required: Self) -> bool {
        self.max_tensors >= required.max_tensors
            && self.max_tensor_rank >= required.max_tensor_rank
            && self.max_tensor_elements >= required.max_tensor_elements
            && self.max_output_tensor_elements >= required.max_output_tensor_elements
            && match (self.max_count_per_peer, required.max_count_per_peer) {
                (_, None) => true,
                (Some(available), Some(required)) => available >= required,
                (None, Some(_)) => false,
            }
    }
}

/// Fine-grained operation, dtype, limit, and completion requirement.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct CommunicationOperationRequirement {
    operation: CommunicationOperation,
    dtypes: Vec<TensorDtype>,
    limits: Option<CommunicationTensorLimits>,
    exact_completion: bool,
}

impl CommunicationOperationRequirement {
    /// Validates one tensor-carrying operation requirement.
    pub fn tensors(
        operation: CommunicationOperation,
        dtypes: impl IntoIterator<Item = TensorDtype>,
        limits: CommunicationTensorLimits,
        exact_completion: bool,
    ) -> Result<Self, CommunicationManifestError> {
        if matches!(
            operation,
            CommunicationOperation::Barrier | CommunicationOperation::FailureAgreement
        ) {
            return Err(CommunicationManifestError::InvalidOperationLimits);
        }
        if operation == CommunicationOperation::VariableAllToAll
            && limits.max_count_per_peer().is_none()
        {
            return Err(CommunicationManifestError::InvalidOperationLimits);
        }
        if operation != CommunicationOperation::VariableAllToAll
            && limits.max_count_per_peer().is_some()
        {
            return Err(CommunicationManifestError::InvalidOperationLimits);
        }
        let dtypes = dtypes.into_iter().collect::<Vec<_>>();
        if dtypes.is_empty() || contains_duplicate_dtypes(&dtypes) {
            return Err(CommunicationManifestError::InvalidOperationDtypes);
        }
        Ok(Self {
            operation,
            dtypes,
            limits: Some(limits),
            exact_completion,
        })
    }

    /// Creates a payload-free barrier requirement.
    pub const fn barrier(exact_completion: bool) -> Self {
        Self {
            operation: CommunicationOperation::Barrier,
            dtypes: Vec::new(),
            limits: None,
            exact_completion,
        }
    }

    /// Creates a payload-free all-rank failure-agreement requirement.
    pub const fn failure_agreement(exact_completion: bool) -> Self {
        Self {
            operation: CommunicationOperation::FailureAgreement,
            dtypes: Vec::new(),
            limits: None,
            exact_completion,
        }
    }

    /// Required operation semantics.
    pub const fn operation(&self) -> CommunicationOperation {
        self.operation
    }

    /// Exact admitted logical tensor element types.
    pub fn dtypes(&self) -> &[TensorDtype] {
        &self.dtypes
    }

    /// Payload limits, absent for payload-free synchronization operations.
    pub const fn limits(&self) -> Option<CommunicationTensorLimits> {
        self.limits
    }

    /// Whether the operation requires an exact completion object.
    pub const fn exact_completion(&self) -> bool {
        self.exact_completion
    }

    fn validate(&self) -> Result<(), CommunicationManifestError> {
        match (self.operation, self.limits) {
            (CommunicationOperation::Barrier | CommunicationOperation::FailureAgreement, None)
                if self.dtypes.is_empty() =>
            {
                Ok(())
            }
            (CommunicationOperation::VariableAllToAll, Some(limits))
                if limits.max_count_per_peer().is_some()
                    && !self.dtypes.is_empty()
                    && !contains_duplicate_dtypes(&self.dtypes) =>
            {
                Ok(())
            }
            (operation, Some(limits))
                if operation != CommunicationOperation::Barrier
                    && operation != CommunicationOperation::FailureAgreement
                    && operation != CommunicationOperation::VariableAllToAll
                    && limits.max_count_per_peer().is_none()
                    && !self.dtypes.is_empty()
                    && !contains_duplicate_dtypes(&self.dtypes) =>
            {
                Ok(())
            }
            _ => Err(CommunicationManifestError::InvalidOperationLimits),
        }
    }
}

impl<'de> Deserialize<'de> for CommunicationOperationRequirement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            operation: CommunicationOperation,
            dtypes: Vec<TensorDtype>,
            limits: Option<CommunicationTensorLimits>,
            exact_completion: bool,
        }

        let raw = Raw::deserialize(deserializer)?;
        let requirement = Self {
            operation: raw.operation,
            dtypes: raw.dtypes,
            limits: raw.limits,
            exact_completion: raw.exact_completion,
        };
        requirement.validate().map_err(serde::de::Error::custom)?;
        Ok(requirement)
    }
}

/// Exact operations required on one opaque group.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct CommunicationGroupRequirements {
    operations: Vec<CommunicationOperationRequirement>,
}

impl<'de> Deserialize<'de> for CommunicationGroupRequirements {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            operations: Vec<CommunicationOperationRequirement>,
        }

        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.operations).map_err(serde::de::Error::custom)
    }
}

impl CommunicationGroupRequirements {
    /// Validates a non-empty, operation-unique requirement list.
    pub fn new(
        operations: impl IntoIterator<Item = CommunicationOperationRequirement>,
    ) -> Result<Self, CommunicationManifestError> {
        let operations = operations.into_iter().collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        if operations.is_empty()
            || operations.iter().any(|requirement| {
                requirement.validate().is_err() || !seen.insert(requirement.operation())
            })
        {
            return Err(CommunicationManifestError::DuplicateOrMissingOperation);
        }
        Ok(Self { operations })
    }

    /// Operations in deterministic declaration order.
    pub fn operations(&self) -> &[CommunicationOperationRequirement] {
        &self.operations
    }
}

/// Ordered membership and exact requirements for one opaque group.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct CommunicationGroupDescriptor {
    id: CommunicationGroupId,
    creation_order: usize,
    members: Vec<usize>,
    local_index: Option<usize>,
    requirements: CommunicationGroupRequirements,
}

impl CommunicationGroupDescriptor {
    /// Validates local ordered membership independent of world geometry.
    pub fn new(
        id: CommunicationGroupId,
        creation_order: usize,
        members: Vec<usize>,
        local_index: Option<usize>,
        requirements: CommunicationGroupRequirements,
    ) -> Result<Self, CommunicationManifestError> {
        let unique = members.iter().copied().collect::<BTreeSet<_>>();
        if members.is_empty() {
            return Err(CommunicationManifestError::EmptyGroup { id });
        }
        if unique.len() != members.len() {
            return Err(CommunicationManifestError::DuplicateGroupMember { id });
        }
        if local_index.is_some_and(|index| index >= members.len()) {
            return Err(CommunicationManifestError::WrongLocalIndex { id });
        }
        Ok(Self {
            id,
            creation_order,
            members,
            local_index,
            requirements,
        })
    }

    /// Opaque stable group identity.
    pub const fn id(&self) -> CommunicationGroupId {
        self.id
    }

    /// Canonical group-creation position shared by every world rank.
    pub const fn creation_order(&self) -> usize {
        self.creation_order
    }

    /// Ordered world-rank membership.
    pub fn members(&self) -> &[usize] {
        &self.members
    }

    /// This manifest rank's position, or `None` when it does not participate.
    pub const fn local_index(&self) -> Option<usize> {
        self.local_index
    }

    /// Exact operations and limits required on the group.
    pub const fn requirements(&self) -> &CommunicationGroupRequirements {
        &self.requirements
    }

    /// Returns the established core descriptor when this rank participates.
    ///
    /// Backends can consume this narrow local descriptor without learning the
    /// semantic topology that produced it.
    pub fn collective_descriptor(&self) -> Option<CollectiveGroupDescriptor> {
        self.local_index.map(|local_index| {
            CollectiveGroupDescriptor::new(self.id, self.members.clone(), local_index)
                .expect("runtime communication descriptor already validates local membership")
        })
    }
}

impl<'de> Deserialize<'de> for CommunicationGroupDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            id: CommunicationGroupId,
            creation_order: usize,
            members: Vec<usize>,
            local_index: Option<usize>,
            requirements: CommunicationGroupRequirements,
        }

        let raw = Raw::deserialize(deserializer)?;
        Self::new(
            raw.id,
            raw.creation_order,
            raw.members,
            raw.local_index,
            raw.requirements,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Directed route with exact mechanical transfer requirements.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct CommunicationRouteDescriptor {
    id: CommunicationRouteId,
    submission_order: usize,
    source: usize,
    destination: usize,
    requirement: CommunicationOperationRequirement,
    boundary: Option<RoleExactBoundaryContract>,
}

/// Versioned in-band framing protocol selected for a boundary route.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryFramingProtocol {
    /// One U8 message containing the full canonical header followed by payload bytes.
    RoleExactV1,
}

/// One exact ordered tensor role on a boundary route.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct BoundaryRoleContract {
    role: String,
    dtype: TensorDtype,
    shape: Vec<BoundaryDimensionContract>,
}

impl<'de> Deserialize<'de> for BoundaryRoleContract {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            role: String,
            dtype: TensorDtype,
            shape: Vec<BoundaryDimensionContract>,
        }

        let raw = Raw::deserialize(deserializer)?;
        Self::symbolic(raw.role, raw.dtype, raw.shape).map_err(serde::de::Error::custom)
    }
}

/// One manifest-selected boundary dimension constraint.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryDimensionContract {
    /// Invocation dimension bounded by the admitted maximum.
    Variable {
        /// Positive admission-time upper bound for this invocation dimension.
        maximum: usize,
    },
    /// Architecture dimension that must match exactly.
    Fixed(usize),
}

impl BoundaryRoleContract {
    /// Creates one concrete role contract.
    pub fn new(
        role: impl Into<String>,
        dtype: TensorDtype,
        shape: Vec<usize>,
    ) -> Result<Self, CommunicationManifestError> {
        let role = role.into();
        if role.trim().is_empty() || shape.is_empty() || shape.contains(&0) {
            return Err(CommunicationManifestError::InvalidBoundaryContract);
        }
        tensor_dtype_width(&dtype).ok_or(CommunicationManifestError::InvalidBoundaryContract)?;
        Ok(Self {
            role,
            dtype,
            shape: shape
                .into_iter()
                .map(BoundaryDimensionContract::Fixed)
                .collect(),
        })
    }

    /// Creates one role with symbolic invocation and exact fixed dimensions.
    pub fn symbolic(
        role: impl Into<String>,
        dtype: TensorDtype,
        shape: Vec<BoundaryDimensionContract>,
    ) -> Result<Self, CommunicationManifestError> {
        let role = role.into();
        if role.trim().is_empty()
            || shape.is_empty()
            || shape.iter().any(|dimension| match dimension {
                BoundaryDimensionContract::Variable { maximum } => *maximum == 0,
                BoundaryDimensionContract::Fixed(value) => *value == 0,
            })
        {
            return Err(CommunicationManifestError::InvalidBoundaryContract);
        }
        tensor_dtype_width(&dtype).ok_or(CommunicationManifestError::InvalidBoundaryContract)?;
        Ok(Self { role, dtype, shape })
    }

    /// Stable semantic role.
    pub fn role(&self) -> &str {
        &self.role
    }
    /// Exact scalar dtype.
    pub const fn dtype(&self) -> &TensorDtype {
        &self.dtype
    }
    /// Per-axis admitted maximum shape.
    pub fn shape(&self) -> &[BoundaryDimensionContract] {
        &self.shape
    }
}

/// Architecture-selected role/schema contract embedded in a route manifest.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct RoleExactBoundaryContract {
    protocol: BoundaryFramingProtocol,
    schema: String,
    roles: Vec<BoundaryRoleContract>,
}

impl<'de> Deserialize<'de> for RoleExactBoundaryContract {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            protocol: BoundaryFramingProtocol,
            schema: String,
            roles: Vec<BoundaryRoleContract>,
        }

        let raw = Raw::deserialize(deserializer)?;
        if raw.protocol != BoundaryFramingProtocol::RoleExactV1 {
            return Err(serde::de::Error::custom(
                CommunicationManifestError::InvalidBoundaryContract,
            ));
        }
        Self::new(raw.schema, raw.roles).map_err(serde::de::Error::custom)
    }
}

impl RoleExactBoundaryContract {
    /// Creates a non-empty ordered role contract.
    pub fn new(
        schema: impl Into<String>,
        roles: impl IntoIterator<Item = BoundaryRoleContract>,
    ) -> Result<Self, CommunicationManifestError> {
        let schema = schema.into();
        let roles = roles.into_iter().collect::<Vec<_>>();
        let mut names = BTreeSet::new();
        if schema.trim().is_empty()
            || roles.is_empty()
            || roles.iter().any(|role| !names.insert(role.role()))
        {
            return Err(CommunicationManifestError::InvalidBoundaryContract);
        }
        Ok(Self {
            protocol: BoundaryFramingProtocol::RoleExactV1,
            schema,
            roles,
        })
    }

    /// Stable architecture schema identity.
    pub fn schema(&self) -> &str {
        &self.schema
    }
    /// Ordered primary then auxiliary role contracts.
    pub fn roles(&self) -> &[BoundaryRoleContract] {
        &self.roles
    }

    fn validate_actual_roles(
        &self,
        actual_roles: &[BoundaryRoleContract],
    ) -> Result<(), CommunicationManifestError> {
        if actual_roles.len() != self.roles.len() {
            return Err(CommunicationManifestError::InvalidBoundaryContract);
        }
        for (actual, admitted) in actual_roles.iter().zip(&self.roles) {
            if actual.role != admitted.role
                || actual.dtype != admitted.dtype
                || actual.shape.len() != admitted.shape.len()
                || actual
                    .shape
                    .iter()
                    .zip(&admitted.shape)
                    .any(|(actual, admitted)| {
                        let BoundaryDimensionContract::Fixed(actual) = actual else {
                            return true;
                        };
                        match admitted {
                            BoundaryDimensionContract::Variable { maximum } => actual > maximum,
                            BoundaryDimensionContract::Fixed(expected) => actual != expected,
                        }
                    })
            {
                return Err(CommunicationManifestError::InvalidBoundaryContract);
            }
        }
        Ok(())
    }

    /// Validates invocation-resolved roles without constructing or submitting payloads.
    pub fn validate_invocation(
        &self,
        actual_roles: &[BoundaryRoleContract],
    ) -> Result<(), CommunicationManifestError> {
        self.validate_actual_roles(actual_roles)
    }

    /// Builds canonical headers for an already locally validated tensor bundle.
    pub fn frame_values<T>(
        &self,
        route: CommunicationRouteId,
        actual_roles: &[BoundaryRoleContract],
        values: Vec<T>,
    ) -> Result<Vec<crate::RoleExactBoundaryValue<T>>, CommunicationManifestError> {
        if values.len() != self.roles.len() {
            return Err(CommunicationManifestError::InvalidBoundaryContract);
        }
        self.validate_actual_roles(actual_roles)?;
        values
            .into_iter()
            .zip(actual_roles)
            .enumerate()
            .map(|(ordinal, (tensor, actual))| {
                Ok(crate::RoleExactBoundaryValue::new(
                    boundary_frame_header(route, &self.schema, ordinal, actual)?,
                    tensor,
                ))
            })
            .collect()
    }
}

impl CommunicationRouteDescriptor {
    /// Validates a directed point-to-point route independent of world geometry.
    pub fn new(
        id: CommunicationRouteId,
        submission_order: usize,
        source: usize,
        destination: usize,
        requirement: CommunicationOperationRequirement,
    ) -> Result<Self, CommunicationManifestError> {
        if source == destination {
            return Err(CommunicationManifestError::InvalidRouteEndpoints { id });
        }
        if requirement.operation() != CommunicationOperation::SendReceive {
            return Err(CommunicationManifestError::InvalidRouteOperation { id });
        }
        requirement.validate()?;
        Ok(Self {
            id,
            submission_order,
            source,
            destination,
            requirement,
            boundary: None,
        })
    }

    /// Attaches the exact role/schema framing contract selected for this route.
    pub fn with_boundary_contract(
        mut self,
        boundary: RoleExactBoundaryContract,
    ) -> Result<Self, CommunicationManifestError> {
        let Some(limits) = self.requirement.limits() else {
            return Err(CommunicationManifestError::InvalidBoundaryContract);
        };
        if !self.requirement.exact_completion() || boundary.roles().len() > limits.max_tensors() {
            return Err(CommunicationManifestError::InvalidBoundaryContract);
        }
        let mut aggregate_bytes = 0usize;
        for role in boundary.roles() {
            if !self.requirement.dtypes().contains(role.dtype())
                || role.shape().len() > limits.max_tensor_rank()
            {
                return Err(CommunicationManifestError::InvalidBoundaryContract);
            }
            let elements = role
                .shape()
                .iter()
                .try_fold(1usize, |elements, dimension| {
                    let extent = match dimension {
                        BoundaryDimensionContract::Variable { maximum } => *maximum,
                        BoundaryDimensionContract::Fixed(value) => *value,
                    };
                    elements.checked_mul(extent)
                })
                .ok_or(CommunicationManifestError::InvalidBoundaryContract)?;
            if elements > limits.max_tensor_elements()
                || elements > limits.max_output_tensor_elements()
            {
                return Err(CommunicationManifestError::InvalidBoundaryContract);
            }
            let bytes = elements
                .checked_mul(
                    tensor_dtype_width(role.dtype())
                        .ok_or(CommunicationManifestError::InvalidBoundaryContract)?,
                )
                .ok_or(CommunicationManifestError::InvalidBoundaryContract)?;
            aggregate_bytes = aggregate_bytes
                .checked_add(bytes)
                .ok_or(CommunicationManifestError::InvalidBoundaryContract)?;
        }
        self.boundary = Some(boundary);
        Ok(self)
    }

    /// Opaque stable route identity.
    pub const fn id(&self) -> CommunicationRouteId {
        self.id
    }

    /// Canonical route-submission position shared by every world rank.
    pub const fn submission_order(&self) -> usize {
        self.submission_order
    }

    /// Source world rank.
    pub const fn source(&self) -> usize {
        self.source
    }

    /// Destination world rank.
    pub const fn destination(&self) -> usize {
        self.destination
    }

    /// Exact transfer dtype, tensor-count, shape, and completion requirement.
    pub const fn requirement(&self) -> &CommunicationOperationRequirement {
        &self.requirement
    }

    /// Exact architecture-selected framing contract, when this is a boundary route.
    pub const fn boundary_contract(&self) -> Option<&RoleExactBoundaryContract> {
        self.boundary.as_ref()
    }
}

impl<'de> Deserialize<'de> for CommunicationRouteDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            id: CommunicationRouteId,
            submission_order: usize,
            source: usize,
            destination: usize,
            requirement: CommunicationOperationRequirement,
            #[serde(default)]
            boundary: Option<RoleExactBoundaryContract>,
        }

        let raw = Raw::deserialize(deserializer)?;
        let route = Self::new(
            raw.id,
            raw.submission_order,
            raw.source,
            raw.destination,
            raw.requirement,
        )
        .map_err(serde::de::Error::custom)?;
        match raw.boundary {
            Some(boundary) => route
                .with_boundary_contract(boundary)
                .map_err(serde::de::Error::custom),
            None => Ok(route),
        }
    }
}

fn tensor_dtype_width(dtype: &TensorDtype) -> Option<usize> {
    Some(match dtype {
        TensorDtype::Bool | TensorDtype::I8 | TensorDtype::U8 => 1,
        TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::U16 | TensorDtype::I16 => 2,
        TensorDtype::F32 | TensorDtype::U32 | TensorDtype::I32 => 4,
        TensorDtype::F64 | TensorDtype::U64 | TensorDtype::I64 | TensorDtype::Complex64 => 8,
        TensorDtype::Encoded(_) => return None,
    })
}

fn dtype_tag(dtype: &TensorDtype) -> Result<u8, CommunicationManifestError> {
    Ok(match dtype {
        TensorDtype::Bool => 0,
        TensorDtype::F32 => 1,
        TensorDtype::F16 => 2,
        TensorDtype::Bf16 => 3,
        TensorDtype::I8 => 4,
        TensorDtype::U8 => 5,
        TensorDtype::U16 => 6,
        TensorDtype::U32 => 7,
        TensorDtype::U64 => 8,
        TensorDtype::I16 => 9,
        TensorDtype::I32 => 10,
        TensorDtype::I64 => 11,
        TensorDtype::F64 => 12,
        TensorDtype::Complex64 => 13,
        TensorDtype::Encoded(_) => return Err(CommunicationManifestError::InvalidBoundaryContract),
    })
}

fn boundary_frame_header(
    route: CommunicationRouteId,
    schema: &str,
    ordinal: usize,
    role: &BoundaryRoleContract,
) -> Result<Vec<u8>, CommunicationManifestError> {
    let exact_shape = role
        .shape
        .iter()
        .map(|dimension| match dimension {
            BoundaryDimensionContract::Fixed(value) => Ok(*value),
            BoundaryDimensionContract::Variable { .. } => {
                Err(CommunicationManifestError::InvalidBoundaryContract)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let payload_elements = exact_shape
        .iter()
        .try_fold(1usize, |value, dimension| value.checked_mul(*dimension))
        .ok_or(CommunicationManifestError::InvalidBoundaryContract)?;
    let payload_bytes = payload_elements
        .checked_mul(
            tensor_dtype_width(&role.dtype)
                .ok_or(CommunicationManifestError::InvalidBoundaryContract)?,
        )
        .ok_or(CommunicationManifestError::InvalidBoundaryContract)?;
    let ordinal =
        u32::try_from(ordinal).map_err(|_| CommunicationManifestError::InvalidBoundaryContract)?;
    let schema_len = u32::try_from(schema.len())
        .map_err(|_| CommunicationManifestError::InvalidBoundaryContract)?;
    let role_len = u32::try_from(role.role.len())
        .map_err(|_| CommunicationManifestError::InvalidBoundaryContract)?;
    let rank = u32::try_from(exact_shape.len())
        .map_err(|_| CommunicationManifestError::InvalidBoundaryContract)?;
    let payload_bytes = u64::try_from(payload_bytes)
        .map_err(|_| CommunicationManifestError::InvalidBoundaryContract)?;
    let mut header = b"EREDUBND".to_vec();
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&route.value().to_le_bytes());
    header.extend_from_slice(&ordinal.to_le_bytes());
    header.push(dtype_tag(&role.dtype)?);
    header.extend_from_slice(&schema_len.to_le_bytes());
    header.extend_from_slice(schema.as_bytes());
    header.extend_from_slice(&role_len.to_le_bytes());
    header.extend_from_slice(role.role.as_bytes());
    header.extend_from_slice(&rank.to_le_bytes());
    for dimension in &exact_shape {
        header.extend_from_slice(
            &u64::try_from(*dimension)
                .map_err(|_| CommunicationManifestError::InvalidBoundaryContract)?
                .to_le_bytes(),
        );
    }
    header.extend_from_slice(&payload_bytes.to_le_bytes());
    Ok(header)
}

/// Selected deadline and safe disposition for exact communication completion.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
pub struct CommunicationCompletionPolicy {
    timeout_millis: u64,
    cancellation: CompletionCancellationMode,
}

impl CommunicationCompletionPolicy {
    /// Selects a positive relative deadline and the required timeout disposition.
    pub fn new(
        timeout: std::time::Duration,
        cancellation: CompletionCancellationMode,
    ) -> Result<Self, CommunicationManifestError> {
        let timeout_millis = u64::try_from(timeout.as_millis())
            .map_err(|_| CommunicationManifestError::InvalidCompletionPolicy)?;
        if timeout_millis == 0 || std::time::Instant::now().checked_add(timeout).is_none() {
            return Err(CommunicationManifestError::InvalidCompletionPolicy);
        }
        Ok(Self {
            timeout_millis,
            cancellation,
        })
    }

    /// Positive selected relative deadline.
    pub const fn timeout(self) -> std::time::Duration {
        std::time::Duration::from_millis(self.timeout_millis)
    }

    /// Required safe disposition for work live at the deadline.
    pub const fn cancellation(self) -> CompletionCancellationMode {
        self.cancellation
    }

    /// Converts the serialized manifest contract into the core completion policy.
    pub fn bounded_wait(self) -> eredu_core::BoundedCompletionWait {
        eredu_core::BoundedCompletionWait::new(self.timeout(), self.cancellation)
            .expect("checked communication completion policy has a positive timeout")
    }
}

impl<'de> Deserialize<'de> for CommunicationCompletionPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            timeout_millis: u64,
            cancellation: CompletionCancellationMode,
        }

        let raw = Raw::deserialize(deserializer)?;
        Self::new(
            std::time::Duration::from_millis(raw.timeout_millis),
            raw.cancellation,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Backend support for bounded completion and safe timeout disposition.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CommunicationCompletionCapabilities {
    cancellation_modes: Vec<CompletionCancellationMode>,
}

impl CommunicationCompletionCapabilities {
    /// Advertises the safe timeout dispositions implemented by a backend.
    pub fn new(
        cancellation_modes: impl IntoIterator<Item = CompletionCancellationMode>,
    ) -> Result<Self, CommunicationManifestError> {
        let cancellation_modes = cancellation_modes.into_iter().collect::<Vec<_>>();
        let unique = cancellation_modes.iter().copied().collect::<BTreeSet<_>>();
        if cancellation_modes.is_empty() || unique.len() != cancellation_modes.len() {
            return Err(CommunicationManifestError::InvalidCompletionCapabilities);
        }
        Ok(Self { cancellation_modes })
    }

    /// Whether this backend can safely apply the policy's timeout disposition.
    pub fn supports(&self, policy: CommunicationCompletionPolicy) -> bool {
        self.cancellation_modes.contains(&policy.cancellation())
    }
}

/// Complete opaque communication description as viewed by one world rank.
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct CommunicationManifest {
    world_size: usize,
    rank: usize,
    groups: Vec<CommunicationGroupDescriptor>,
    routes: Vec<CommunicationRouteDescriptor>,
    completion: Option<CommunicationCompletionPolicy>,
}

impl CommunicationManifest {
    /// Validates world geometry, membership, local positions, endpoints, and order.
    pub fn new(
        world_size: usize,
        rank: usize,
        groups: Vec<CommunicationGroupDescriptor>,
        routes: Vec<CommunicationRouteDescriptor>,
    ) -> Result<Self, CommunicationManifestError> {
        if world_size == 0 || rank >= world_size {
            return Err(CommunicationManifestError::RankOutOfRange { rank, world_size });
        }
        let mut group_ids = BTreeSet::new();
        for (order, group) in groups.iter().enumerate() {
            if !group_ids.insert(group.id()) {
                return Err(CommunicationManifestError::DuplicateGroupId { id: group.id() });
            }
            if group.creation_order() != order {
                return Err(CommunicationManifestError::WrongGroupOrder {
                    id: group.id(),
                    expected: order,
                    actual: group.creation_order(),
                });
            }
            if group.members().iter().any(|member| *member >= world_size) {
                return Err(CommunicationManifestError::GroupMemberOutOfRange {
                    id: group.id(),
                    world_size,
                });
            }
            let expected = group.members().iter().position(|member| *member == rank);
            if group.local_index() != expected {
                return Err(CommunicationManifestError::WrongLocalIndex { id: group.id() });
            }
        }

        let mut route_ids = BTreeSet::new();
        for (order, route) in routes.iter().enumerate() {
            if !route_ids.insert(route.id()) {
                return Err(CommunicationManifestError::DuplicateRouteId { id: route.id() });
            }
            if route.submission_order() != order {
                return Err(CommunicationManifestError::WrongRouteOrder {
                    id: route.id(),
                    expected: order,
                    actual: route.submission_order(),
                });
            }
            if route.source() >= world_size || route.destination() >= world_size {
                return Err(CommunicationManifestError::RouteEndpointOutOfRange {
                    id: route.id(),
                    world_size,
                });
            }
        }
        Ok(Self {
            world_size,
            rank,
            groups,
            routes,
            completion: None,
        })
    }

    /// Attaches the exact bounded-completion policy selected before realization.
    pub fn with_completion_policy(mut self, policy: CommunicationCompletionPolicy) -> Self {
        self.completion = Some(policy);
        self
    }

    /// Total process count.
    pub const fn world_size(&self) -> usize {
        self.world_size
    }

    /// World rank whose local positions this manifest carries.
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// This rank's opaque groups in canonical collective-creation order.
    pub fn groups(&self) -> &[CommunicationGroupDescriptor] {
        &self.groups
    }

    /// Directed routes in canonical submission order.
    pub fn routes(&self) -> &[CommunicationRouteDescriptor] {
        &self.routes
    }

    /// Returns the manifest-proven route submission waves in canonical order.
    ///
    /// A multi-route wave is selected only when one contiguous batch has an
    /// identical mechanical contract, disjoint endpoints, and covers every
    /// world rank exactly once. Such logical endpoint groups may share one
    /// world-backed native collective ordinal. Routes without that proof stay
    /// in singleton waves and retain ordinary sequential point-to-point order.
    pub(crate) fn route_submission_waves(&self) -> Vec<Range<usize>> {
        let mut waves = Vec::new();
        let mut start = 0;
        while start < self.routes.len() {
            let reference = &self.routes[start];
            let mut endpoints = vec![false; self.world_size];
            let mut end = start;
            while end < self.routes.len() {
                let route = &self.routes[end];
                if route.requirement() != reference.requirement()
                    || route.boundary_contract() != reference.boundary_contract()
                    || endpoints[route.source()]
                    || endpoints[route.destination()]
                {
                    break;
                }
                endpoints[route.source()] = true;
                endpoints[route.destination()] = true;
                end += 1;
                if endpoints.iter().all(|endpoint| *endpoint) {
                    break;
                }
            }
            if !endpoints.iter().all(|endpoint| *endpoint) {
                end = start + 1;
            }
            waves.push(start..end);
            start = end;
        }
        waves
    }

    /// Selected bounded-completion policy, if this manifest requires one.
    pub const fn completion_policy(&self) -> Option<CommunicationCompletionPolicy> {
        self.completion
    }

    /// Invokes a mechanism-only callback for every descriptor in creation order.
    pub fn try_create_groups<T, E>(
        &self,
        mut create: impl FnMut(&CommunicationGroupDescriptor) -> Result<T, E>,
    ) -> Result<Vec<T>, E> {
        self.groups.iter().map(&mut create).collect()
    }

    /// Invokes a mechanism-only callback for every route in submission order.
    pub fn try_create_routes<T, E>(
        &self,
        mut create: impl FnMut(&CommunicationRouteDescriptor) -> Result<T, E>,
    ) -> Result<Vec<T>, E> {
        self.routes.iter().map(&mut create).collect()
    }
}

impl<'de> Deserialize<'de> for CommunicationManifest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            world_size: usize,
            rank: usize,
            groups: Vec<CommunicationGroupDescriptor>,
            routes: Vec<CommunicationRouteDescriptor>,
            #[serde(default)]
            completion: Option<CommunicationCompletionPolicy>,
        }

        let raw = Raw::deserialize(deserializer)?;
        let manifest = Self::new(raw.world_size, raw.rank, raw.groups, raw.routes)
            .map_err(serde::de::Error::custom)?;
        Ok(match raw.completion {
            Some(policy) => manifest.with_completion_policy(policy),
            None => manifest,
        })
    }
}

/// Semantic inputs used only while projecting opaque communication descriptors.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct TopologyCommunicationPlan {
    session_group: Option<CommunicationGroupRequirements>,
    tensor_groups: Option<CommunicationGroupRequirements>,
    pipeline_groups: Option<CommunicationGroupRequirements>,
    expert_groups: Option<CommunicationGroupRequirements>,
    data_groups: Option<CommunicationGroupRequirements>,
    pipeline_routes: Option<CommunicationOperationRequirement>,
    completion: Option<CommunicationCompletionPolicy>,
}

impl TopologyCommunicationPlan {
    /// Creates an empty projection plan.
    pub const fn new() -> Self {
        Self {
            session_group: None,
            tensor_groups: None,
            pipeline_groups: None,
            expert_groups: None,
            data_groups: None,
            pipeline_routes: None,
            completion: None,
        }
    }

    /// Selects one bounded completion policy for every operation in the manifest.
    pub fn with_completion_policy(mut self, policy: CommunicationCompletionPolicy) -> Self {
        self.completion = Some(policy);
        self
    }

    /// Requests one group containing every participant in world-rank order.
    ///
    /// Session groups are projected before axis groups, so their opaque ID is
    /// stable across the set of active Cartesian axes.
    pub fn with_session_group(mut self, requirements: CommunicationGroupRequirements) -> Self {
        self.session_group = Some(requirements);
        self
    }

    /// Exact opaque ID assigned to the requested session group.
    pub const fn session_group_id(&self) -> Option<CommunicationGroupId> {
        if self.session_group.is_some() {
            Some(CommunicationGroupId::new(1))
        } else {
            None
        }
    }

    /// Requests groups varying the tensor coordinate.
    pub fn with_tensor_groups(mut self, requirements: CommunicationGroupRequirements) -> Self {
        self.tensor_groups = Some(requirements);
        self
    }

    /// Exact opaque tensor-group ID assigned to one rank by this projection plan.
    pub fn tensor_group_id(
        &self,
        topology: ParallelTopology,
        rank: ParallelRankTopology,
    ) -> Result<Option<CommunicationGroupId>, CommunicationManifestError> {
        if self.tensor_groups.is_none() {
            return Ok(None);
        }
        if rank.topology() != topology {
            return Err(CommunicationManifestError::TopologyMismatch);
        }
        let groups = unique_axis_groups(topology, ParallelAxis::Tensor)?;
        let subgroup = groups
            .iter()
            .position(|members| members.contains(&rank.global_rank()))
            .ok_or(CommunicationManifestError::InvalidTopologyProjection)?;
        let first = 1usize + usize::from(self.session_group.is_some());
        let numeric = first
            .checked_add(subgroup)
            .ok_or(CommunicationManifestError::DescriptorCountOverflow)?;
        Ok(Some(CommunicationGroupId::new(
            u32::try_from(numeric)
                .map_err(|_| CommunicationManifestError::DescriptorCountOverflow)?,
        )))
    }

    /// Requests groups varying the pipeline coordinate.
    pub fn with_pipeline_groups(mut self, requirements: CommunicationGroupRequirements) -> Self {
        self.pipeline_groups = Some(requirements);
        self
    }

    /// Requests groups varying the expert coordinate.
    pub fn with_expert_groups(mut self, requirements: CommunicationGroupRequirements) -> Self {
        self.expert_groups = Some(requirements);
        self
    }

    /// Represents future groups varying the data coordinate.
    ///
    /// This is a topology projection facility, not product admission of data
    /// parallel execution. Product selection must continue to reject it until
    /// its semantics are implemented.
    pub fn with_data_groups(mut self, requirements: CommunicationGroupRequirements) -> Self {
        self.data_groups = Some(requirements);
        self
    }

    /// Requests directed routes between adjacent pipeline coordinates.
    pub fn with_pipeline_routes(
        mut self,
        requirement: CommunicationOperationRequirement,
    ) -> Result<Self, CommunicationManifestError> {
        if requirement.operation() != CommunicationOperation::SendReceive {
            return Err(CommunicationManifestError::InvalidRouteOperation {
                id: CommunicationRouteId::new(0),
            });
        }
        self.pipeline_routes = Some(requirement);
        Ok(self)
    }
}

/// Projects one rank's semantic Cartesian topology into opaque descriptors.
pub fn project_communication_manifest(
    topology: ParallelTopology,
    rank: ParallelRankTopology,
    plan: &TopologyCommunicationPlan,
) -> Result<CommunicationManifest, CommunicationManifestError> {
    if rank.topology() != topology {
        return Err(CommunicationManifestError::TopologyMismatch);
    }

    let mut groups = Vec::new();
    let mut next_group_id = 1usize;
    if let Some(requirements) = &plan.session_group {
        let id = CommunicationGroupId::new(1);
        groups.push(CommunicationGroupDescriptor::new(
            id,
            0,
            (0..topology.world_size()).collect(),
            Some(rank.global_rank()),
            requirements.clone(),
        )?);
        next_group_id = 2;
    }
    for (axis, requirements) in [
        (ParallelAxis::Tensor, plan.tensor_groups.as_ref()),
        (ParallelAxis::Pipeline, plan.pipeline_groups.as_ref()),
        (ParallelAxis::Expert, plan.expert_groups.as_ref()),
        (ParallelAxis::Data, plan.data_groups.as_ref()),
    ] {
        let Some(requirements) = requirements else {
            continue;
        };
        let axis_groups = unique_axis_groups(topology, axis)?;
        let (subgroup_index, members) = axis_groups
            .iter()
            .enumerate()
            .find(|(_, members)| members.contains(&rank.global_rank()))
            .ok_or(CommunicationManifestError::InvalidTopologyProjection)?;
        let numeric_id = next_group_id
            .checked_add(subgroup_index)
            .ok_or(CommunicationManifestError::DescriptorCountOverflow)?;
        let id = CommunicationGroupId::new(
            u32::try_from(numeric_id)
                .map_err(|_| CommunicationManifestError::DescriptorCountOverflow)?,
        );
        let local_index = members
            .iter()
            .position(|member| *member == rank.global_rank());
        groups.push(CommunicationGroupDescriptor::new(
            id,
            groups.len(),
            members.clone(),
            local_index,
            requirements.clone(),
        )?);
        next_group_id = next_group_id
            .checked_add(axis_groups.len())
            .ok_or(CommunicationManifestError::DescriptorCountOverflow)?;
    }

    let mut routes = Vec::new();
    if let Some(requirement) = &plan.pipeline_routes {
        for source in 0..topology.world_size() {
            let coordinates = topology
                .coordinates(source)
                .map_err(|_| CommunicationManifestError::InvalidTopologyProjection)?;
            if coordinates.pipeline() + 1 == topology.pipeline() {
                continue;
            }
            let destination = topology
                .rank_for(coordinates.with_pipeline(coordinates.pipeline() + 1))
                .map_err(|_| CommunicationManifestError::InvalidTopologyProjection)?;
            let id = CommunicationRouteId::new(routes.len() as u64);
            routes.push(CommunicationRouteDescriptor::new(
                id,
                routes.len(),
                source,
                destination,
                requirement.clone(),
            )?);
        }
    }

    let manifest =
        CommunicationManifest::new(topology.world_size(), rank.global_rank(), groups, routes)?;
    Ok(match plan.completion {
        Some(policy) => manifest.with_completion_policy(policy),
        None => manifest,
    })
}

/// Projects and cross-validates compatible opaque descriptors for every rank.
pub fn project_all_communication_manifests(
    topology: ParallelTopology,
    plan: &TopologyCommunicationPlan,
) -> Result<Vec<CommunicationManifest>, CommunicationManifestError> {
    let manifests = (0..topology.world_size())
        .map(|global_rank| {
            let rank = ParallelRankTopology::new(topology, global_rank)
                .map_err(|_| CommunicationManifestError::InvalidTopologyProjection)?;
            project_communication_manifest(topology, rank, plan)
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_compatible_communication_manifests(&manifests)?;
    Ok(manifests)
}

/// Proves that all ranks derived identical global descriptors and exact local indices.
pub fn validate_compatible_communication_manifests(
    manifests: &[CommunicationManifest],
) -> Result<(), CommunicationManifestError> {
    let Some(reference) = manifests.first() else {
        return Err(CommunicationManifestError::MissingRankManifest { rank: 0 });
    };
    if manifests.len() != reference.world_size() {
        return Err(CommunicationManifestError::WrongManifestCount {
            expected: reference.world_size(),
            actual: manifests.len(),
        });
    }
    let mut ranks = vec![None; reference.world_size()];
    for manifest in manifests {
        if manifest.world_size() != reference.world_size()
            || manifest.rank() >= reference.world_size()
        {
            return Err(CommunicationManifestError::IncompatibleRankManifest {
                rank: manifest.rank(),
            });
        }
        let rank = manifest.rank();
        if ranks[rank].replace(manifest).is_some() {
            return Err(CommunicationManifestError::DuplicateRankManifest { rank });
        }
    }
    let ranks = ranks
        .into_iter()
        .enumerate()
        .map(|(rank, manifest)| {
            manifest.ok_or(CommunicationManifestError::MissingRankManifest { rank })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let group_count = reference.groups().len();
    let mut ids_across_orders = BTreeSet::new();
    for (rank, manifest) in ranks.iter().copied().enumerate() {
        if manifest.routes() != reference.routes()
            || manifest.groups().len() != group_count
            || manifest.completion_policy() != reference.completion_policy()
        {
            return Err(CommunicationManifestError::IncompatibleRankManifest { rank });
        }
    }
    for order in 0..group_count {
        let mut subgroup_ids = BTreeSet::new();
        let mut covered = vec![false; reference.world_size()];
        for (rank, manifest) in ranks.iter().copied().enumerate() {
            let group = &manifest.groups()[order];
            if group.creation_order() != order
                || group.local_index() != group.members().iter().position(|member| *member == rank)
            {
                return Err(CommunicationManifestError::IncompatibleRankManifest { rank });
            }
            if subgroup_ids.insert(group.id()) {
                if !ids_across_orders.insert(group.id()) {
                    return Err(CommunicationManifestError::IncompatibleRankManifest { rank });
                }
                for member in group.members() {
                    if covered[*member] {
                        return Err(CommunicationManifestError::IncompatibleRankManifest { rank });
                    }
                    covered[*member] = true;
                }
            }
            for member in group.members() {
                if *member >= ranks.len() {
                    return Err(CommunicationManifestError::IncompatibleRankManifest { rank });
                }
                let peer = &ranks[*member].groups()[order];
                if peer.id() != group.id()
                    || peer.creation_order() != group.creation_order()
                    || peer.members() != group.members()
                    || peer.requirements() != group.requirements()
                {
                    return Err(CommunicationManifestError::IncompatibleRankManifest {
                        rank: *member,
                    });
                }
            }
        }
        if covered.contains(&false) {
            return Err(CommunicationManifestError::IncompatibleRankManifest { rank: 0 });
        }
    }
    for (rank, manifest) in ranks.iter().copied().enumerate() {
        for group in manifest.groups() {
            if group.local_index().is_none() {
                return Err(CommunicationManifestError::IncompatibleRankManifest { rank });
            }
        }
    }
    Ok(())
}

/// Gathers and cross-validates one complete opaque manifest from every world rank.
///
/// The transport only needs an equal-word all-gather. A fixed two-word length
/// gather determines the largest encoded manifest, then every rank pads its
/// complete payload to that size for a second equal-word gather. Validation is
/// deliberately deferred until both collectives complete, so a rank-local
/// world, rank, descriptor, or limit mismatch cannot strand its peers before
/// the shared compatibility result is available.
pub fn validate_communication_manifest_consensus<T: ConsensusTransport>(
    transport: &T,
    local: &CommunicationManifest,
) -> Result<Vec<CommunicationManifest>, CommunicationManifestConsensusError> {
    let participants = transport.participant_count();
    if participants == 0 {
        return Err(CommunicationManifestConsensusError::EmptyTopology);
    }

    let encoded = serde_json::to_vec(local)
        .map_err(|error| CommunicationManifestConsensusError::Encoding(error.to_string()))?;
    let encoded_len = u64::try_from(encoded.len()).map_err(|_| {
        CommunicationManifestConsensusError::MetadataOverflow("encoded manifest length")
    })?;
    let length_words = [encoded_len as u32, (encoded_len >> 32) as u32];
    let gathered_lengths =
        gather_manifest_words(transport, &length_words, participants, "manifest lengths")?;

    let lengths = gathered_lengths
        .as_chunks::<2>()
        .0
        .iter()
        .enumerate()
        .map(|(rank, words)| {
            let length = u64::from(words[0]) | (u64::from(words[1]) << 32);
            usize::try_from(length).map_err(|_| {
                CommunicationManifestConsensusError::PayloadLengthOverflow { rank, length }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let payload_words = lengths
        .iter()
        .copied()
        .map(words_for_bytes)
        .max()
        .unwrap_or(0);
    let local_words = encode_manifest_words(&encoded, payload_words);
    let gathered_payloads =
        gather_manifest_words(transport, &local_words, participants, "manifest payloads")?;

    let mut manifests = Vec::with_capacity(participants);
    for (rank, &length) in lengths.iter().enumerate() {
        let available = payload_words.checked_mul(4).ok_or(
            CommunicationManifestConsensusError::MetadataOverflow("manifest payload bytes"),
        )?;
        if length > available {
            return Err(CommunicationManifestConsensusError::InvalidPayloadLength {
                rank,
                length,
                available,
            });
        }
        let start = rank.checked_mul(payload_words).ok_or(
            CommunicationManifestConsensusError::MetadataOverflow("manifest payload offset"),
        )?;
        let end = start.checked_add(payload_words).ok_or(
            CommunicationManifestConsensusError::MetadataOverflow("manifest payload end"),
        )?;
        let mut bytes = Vec::with_capacity(available);
        for word in &gathered_payloads[start..end] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.truncate(length);
        manifests.push(serde_json::from_slice(&bytes).map_err(|error| {
            CommunicationManifestConsensusError::InvalidEncoding {
                rank,
                message: error.to_string(),
            }
        })?);
    }

    validate_compatible_communication_manifests(&manifests)?;
    Ok(manifests)
}

fn words_for_bytes(bytes: usize) -> usize {
    bytes / 4 + usize::from(!bytes.is_multiple_of(4))
}

fn encode_manifest_words(encoded: &[u8], padded_words: usize) -> Vec<u32> {
    let mut words = Vec::with_capacity(padded_words);
    for chunk in encoded.chunks(4) {
        let mut word = [0; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(word));
    }
    words.resize(padded_words, 0);
    words
}

fn gather_manifest_words<T: ConsensusTransport>(
    transport: &T,
    local: &[u32],
    participants: usize,
    stage: &'static str,
) -> Result<Vec<u32>, CommunicationManifestConsensusError> {
    let expected = local.len().checked_mul(participants).ok_or(
        CommunicationManifestConsensusError::MetadataOverflow("gathered manifest word count"),
    )?;
    let gathered = transport
        .all_gather_words(local)
        .map_err(|error| CommunicationManifestConsensusError::Transport(error.to_string()))?;
    if gathered.len() != expected {
        return Err(CommunicationManifestConsensusError::MalformedGather {
            stage,
            expected,
            actual: gathered.len(),
            participants,
        });
    }
    Ok(gathered)
}

/// Backend capability limits for exact opaque communication mechanisms.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CommunicationCapabilities {
    operations: Vec<CommunicationOperationRequirement>,
    completion: Option<CommunicationCompletionCapabilities>,
    boundary_framing: Vec<BoundaryFramingProtocol>,
}

impl CommunicationCapabilities {
    /// Validates one capability entry per operation.
    pub fn new(
        operations: impl IntoIterator<Item = CommunicationOperationRequirement>,
    ) -> Result<Self, CommunicationManifestError> {
        let operations = operations.into_iter().collect::<Vec<_>>();
        let mut seen = BTreeSet::new();
        if operations.iter().any(|capability| {
            capability.validate().is_err() || !seen.insert(capability.operation())
        }) {
            return Err(CommunicationManifestError::DuplicateOrMissingOperation);
        }
        Ok(Self {
            operations,
            completion: None,
            boundary_framing: Vec::new(),
        })
    }

    /// Advertises exact in-band boundary protocols implemented by the backend.
    pub fn with_boundary_framing(
        mut self,
        protocols: impl IntoIterator<Item = BoundaryFramingProtocol>,
    ) -> Result<Self, CommunicationManifestError> {
        let protocols = protocols.into_iter().collect::<Vec<_>>();
        if protocols.is_empty()
            || protocols
                .iter()
                .enumerate()
                .any(|(index, protocol)| protocols[..index].contains(protocol))
        {
            return Err(CommunicationManifestError::InvalidBoundaryContract);
        }
        self.boundary_framing = protocols;
        Ok(self)
    }

    /// Attaches backend support for bounded completion timeout dispositions.
    pub fn with_completion_capabilities(
        mut self,
        completion: CommunicationCompletionCapabilities,
    ) -> Self {
        self.completion = Some(completion);
        self
    }

    /// Validates every manifest requirement against these generic mechanisms.
    pub fn validate_manifest(
        &self,
        manifest: &CommunicationManifest,
    ) -> Result<(), CommunicationCapabilityError> {
        if !manifest.groups().is_empty() || !manifest.routes().is_empty() {
            let policy = manifest
                .completion_policy()
                .ok_or(CommunicationCapabilityError::MissingSelectedCompletionPolicy)?;
            let completion = self
                .completion
                .as_ref()
                .ok_or(CommunicationCapabilityError::MissingBoundedCompletion)?;
            if !completion.supports(policy) {
                return Err(CommunicationCapabilityError::UnsupportedCancellationMode {
                    cancellation: policy.cancellation(),
                });
            }
        }
        for group in manifest.groups() {
            for requirement in group.requirements().operations() {
                self.validate_requirement(requirement)?;
            }
        }
        for route in manifest.routes() {
            self.validate_requirement(route.requirement())?;
            if let Some(boundary) = route.boundary_contract() {
                if !self.boundary_framing.contains(&boundary.protocol) {
                    return Err(CommunicationCapabilityError::MissingBoundaryFraming {
                        protocol: boundary.protocol,
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_requirement(
        &self,
        requirement: &CommunicationOperationRequirement,
    ) -> Result<(), CommunicationCapabilityError> {
        let capability = self
            .operations
            .iter()
            .find(|capability| capability.operation() == requirement.operation())
            .ok_or(CommunicationCapabilityError::MissingOperation {
                operation: requirement.operation(),
            })?;
        for dtype in requirement.dtypes() {
            if !capability.dtypes().contains(dtype) {
                return Err(CommunicationCapabilityError::UnsupportedDtype {
                    operation: requirement.operation(),
                    dtype: dtype.clone(),
                });
            }
        }
        match (capability.limits(), requirement.limits()) {
            (Some(available), Some(required)) if available.covers(required) => {}
            (None, None) => {}
            _ => {
                return Err(CommunicationCapabilityError::InsufficientLimits {
                    operation: requirement.operation(),
                })
            }
        }
        if requirement.exact_completion() && !capability.exact_completion() {
            return Err(CommunicationCapabilityError::MissingExactCompletion {
                operation: requirement.operation(),
            });
        }
        Ok(())
    }
}

fn unique_axis_groups(
    topology: ParallelTopology,
    axis: ParallelAxis,
) -> Result<Vec<Vec<usize>>, CommunicationManifestError> {
    let mut seen = BTreeSet::new();
    let mut groups = Vec::new();
    for rank in 0..topology.world_size() {
        let members = topology
            .axis_members(rank, axis)
            .map_err(|_| CommunicationManifestError::InvalidTopologyProjection)?;
        if seen.insert(members.clone()) {
            groups.push(members);
        }
    }
    Ok(groups)
}

fn contains_duplicate_dtypes(dtypes: &[TensorDtype]) -> bool {
    dtypes
        .iter()
        .enumerate()
        .any(|(index, dtype)| dtypes[..index].contains(dtype))
}

/// Invalid opaque communication manifest or topology projection.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CommunicationManifestError {
    /// A role-exact route schema is empty, incoherent, or not representable.
    #[error("communication boundary framing contract is invalid")]
    InvalidBoundaryContract,
    /// Manifest rank is outside the declared world.
    #[error("communication rank {rank} is outside world size {world_size}")]
    RankOutOfRange {
        /// Invalid rank.
        rank: usize,
        /// Declared world size.
        world_size: usize,
    },
    /// Rank topology and complete topology disagree.
    #[error("rank topology does not match the topology being projected")]
    TopologyMismatch,
    /// A validated core topology could not be projected consistently.
    #[error("parallel topology could not be projected into communication descriptors")]
    InvalidTopologyProjection,
    /// Tensor/count limits were absent or incoherent for the operation.
    #[error("communication operation has invalid tensor or per-peer limits")]
    InvalidOperationLimits,
    /// Dtype requirements were empty or repeated.
    #[error("communication operation tensor dtypes must be non-empty and unique")]
    InvalidOperationDtypes,
    /// A bounded completion policy has no positive representable deadline.
    #[error("communication completion policy must have a positive millisecond deadline")]
    InvalidCompletionPolicy,
    /// Completion capability modes were absent or repeated.
    #[error("communication completion capabilities must be non-empty and unique")]
    InvalidCompletionCapabilities,
    /// A group requirement was empty or repeated an operation.
    #[error("communication group operations must be non-empty and unique")]
    DuplicateOrMissingOperation,
    /// A group has no members.
    #[error("communication group {id:?} has no members")]
    EmptyGroup {
        /// Invalid group.
        id: CommunicationGroupId,
    },
    /// A group repeats a world rank.
    #[error("communication group {id:?} repeats a world rank")]
    DuplicateGroupMember {
        /// Invalid group.
        id: CommunicationGroupId,
    },
    /// A group member is outside the manifest world.
    #[error("communication group {id:?} contains a rank outside world size {world_size}")]
    GroupMemberOutOfRange {
        /// Invalid group.
        id: CommunicationGroupId,
        /// Declared world size.
        world_size: usize,
    },
    /// A group local index does not identify the manifest rank exactly.
    #[error("communication group {id:?} has the wrong local member index")]
    WrongLocalIndex {
        /// Invalid group.
        id: CommunicationGroupId,
    },
    /// Group IDs are not unique.
    #[error("communication group ID {id:?} is repeated")]
    DuplicateGroupId {
        /// Repeated group identity.
        id: CommunicationGroupId,
    },
    /// Group descriptors are not in their declared canonical order.
    #[error("communication group {id:?} has creation order {actual}, expected {expected}")]
    WrongGroupOrder {
        /// Invalid group.
        id: CommunicationGroupId,
        /// Required position.
        expected: usize,
        /// Declared position.
        actual: usize,
    },
    /// A route is not directed between distinct endpoints.
    #[error("communication route {id:?} must have distinct endpoints")]
    InvalidRouteEndpoints {
        /// Invalid route.
        id: CommunicationRouteId,
    },
    /// A route carries a non-point-to-point operation.
    #[error("communication route {id:?} must require send/receive")]
    InvalidRouteOperation {
        /// Invalid route.
        id: CommunicationRouteId,
    },
    /// A route endpoint is outside the manifest world.
    #[error("communication route {id:?} contains an endpoint outside world size {world_size}")]
    RouteEndpointOutOfRange {
        /// Invalid route.
        id: CommunicationRouteId,
        /// Declared world size.
        world_size: usize,
    },
    /// Route IDs are not unique.
    #[error("communication route ID {id:?} is repeated")]
    DuplicateRouteId {
        /// Repeated route identity.
        id: CommunicationRouteId,
    },
    /// Route descriptors are not in their declared canonical order.
    #[error("communication route {id:?} has submission order {actual}, expected {expected}")]
    WrongRouteOrder {
        /// Invalid route.
        id: CommunicationRouteId,
        /// Required position.
        expected: usize,
        /// Declared position.
        actual: usize,
    },
    /// The number of per-rank projections does not match the world.
    #[error("communication projection has {actual} manifests, expected {expected}")]
    WrongManifestCount {
        /// Expected world size.
        expected: usize,
        /// Actual manifest count.
        actual: usize,
    },
    /// No manifest was supplied for a world rank.
    #[error("communication projection is missing rank {rank}")]
    MissingRankManifest {
        /// Missing world rank.
        rank: usize,
    },
    /// More than one manifest was supplied for a world rank.
    #[error("communication projection repeats rank {rank}")]
    DuplicateRankManifest {
        /// Repeated world rank.
        rank: usize,
    },
    /// A rank derived different global descriptors or local positions.
    #[error("communication projection for rank {rank} is incompatible with its peers")]
    IncompatibleRankManifest {
        /// Incompatible world rank.
        rank: usize,
    },
    /// The projected descriptor count cannot be represented by the stable ID.
    #[error("communication descriptor count exceeds its stable ID representation")]
    DescriptorCountOverflow,
    /// Variable exchange counts do not match opaque group membership.
    #[error(
        "communication peer counts have send length {send} and receive length {receive}, expected {group_size}"
    )]
    InvalidPeerCounts {
        /// Expected opaque group cardinality.
        group_size: usize,
        /// Supplied send-count cardinality.
        send: usize,
        /// Supplied receive-count cardinality.
        receive: usize,
    },
}

/// Failure while exchanging or validating complete rank-local manifests.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CommunicationManifestConsensusError {
    /// A consensus topology cannot be empty.
    #[error("communication manifest consensus topology has no participants")]
    EmptyTopology,
    /// Portable metadata exceeded its wire or host representation.
    #[error("communication manifest consensus {0} overflowed")]
    MetadataOverflow(&'static str),
    /// The local manifest could not be encoded.
    #[error("communication manifest encoding failed: {0}")]
    Encoding(String),
    /// The control-plane collective failed.
    #[error("communication manifest consensus transport failed: {0}")]
    Transport(String),
    /// An equal-word gather returned a malformed rank-major frame.
    #[error(
        "communication manifest {stage} gather returned {actual} words; expected {expected} for {participants} ranks"
    )]
    MalformedGather {
        /// Protocol stage being gathered.
        stage: &'static str,
        /// Expected gathered word count.
        expected: usize,
        /// Actual gathered word count.
        actual: usize,
        /// Expected participant count.
        participants: usize,
    },
    /// A peer encoded a payload length that this host cannot represent.
    #[error("communication manifest payload length {length} from rank {rank} exceeds usize")]
    PayloadLengthOverflow {
        /// Rank that advertised the length.
        rank: usize,
        /// Advertised encoded byte length.
        length: u64,
    },
    /// A peer advertised more bytes than fit in the gathered padded frame.
    #[error(
        "communication manifest payload from rank {rank} has length {length}, but only {available} bytes were gathered"
    )]
    InvalidPayloadLength {
        /// Rank that advertised the invalid length.
        rank: usize,
        /// Advertised encoded byte length.
        length: usize,
        /// Bytes available in one gathered rank frame.
        available: usize,
    },
    /// A peer payload was not a valid checked opaque manifest.
    #[error("communication manifest payload from rank {rank} is invalid: {message}")]
    InvalidEncoding {
        /// Rank whose payload failed decoding or checked construction.
        rank: usize,
        /// Serialization or manifest validation detail.
        message: String,
    },
    /// The reconstructed rank manifests are not mutually compatible.
    #[error(transparent)]
    Manifest(#[from] CommunicationManifestError),
}

/// Missing or insufficient generic communication mechanism capability.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CommunicationCapabilityError {
    /// A route selected an in-band framing protocol unavailable in the backend.
    #[error("boundary framing protocol {protocol:?} is unavailable")]
    MissingBoundaryFraming {
        /// Selected protocol.
        protocol: BoundaryFramingProtocol,
    },
    /// The manifest requires bounded completion but the backend did not advertise it.
    #[error("bounded communication completion is unavailable")]
    MissingBoundedCompletion,
    /// A manifest selects communication resources but no bounded completion policy.
    #[error("communication manifest has no selected bounded completion policy")]
    MissingSelectedCompletionPolicy,
    /// The backend cannot safely apply the selected timeout disposition.
    #[error("communication completion cannot apply cancellation mode {cancellation:?}")]
    UnsupportedCancellationMode {
        /// Selected but unavailable timeout disposition.
        cancellation: CompletionCancellationMode,
    },
    /// The exact operation is unavailable.
    #[error("communication operation {operation:?} is unavailable")]
    MissingOperation {
        /// Missing operation.
        operation: CommunicationOperation,
    },
    /// The operation cannot carry one required dtype.
    #[error("communication operation {operation:?} does not support dtype {dtype:?}")]
    UnsupportedDtype {
        /// Required operation.
        operation: CommunicationOperation,
        /// Unsupported dtype.
        dtype: TensorDtype,
    },
    /// Tensor shape or count limits are too small.
    #[error("communication operation {operation:?} has insufficient tensor or count limits")]
    InsufficientLimits {
        /// Required operation.
        operation: CommunicationOperation,
    },
    /// Exact completion is required but unavailable.
    #[error("communication operation {operation:?} lacks exact completion")]
    MissingExactCompletion {
        /// Required operation.
        operation: CommunicationOperation,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_completion_policy() -> CommunicationCompletionPolicy {
        CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap()
    }

    fn test_completion_capabilities() -> CommunicationCompletionCapabilities {
        CommunicationCompletionCapabilities::new([
            CompletionCancellationMode::QuarantineUntilComplete,
        ])
        .unwrap()
    }

    struct ScriptedManifestTransport {
        encoded: Vec<Vec<u8>>,
        calls: std::cell::Cell<usize>,
    }

    impl ScriptedManifestTransport {
        fn new(manifests: &[CommunicationManifest]) -> Self {
            Self {
                encoded: manifests
                    .iter()
                    .map(|manifest| serde_json::to_vec(manifest).unwrap())
                    .collect(),
                calls: std::cell::Cell::new(0),
            }
        }
    }

    impl ConsensusTransport for ScriptedManifestTransport {
        type Error = std::convert::Infallible;

        fn participant_count(&self) -> usize {
            self.encoded.len()
        }

        fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            match call {
                0 => {
                    assert_eq!(local.len(), 2);
                    Ok(self
                        .encoded
                        .iter()
                        .flat_map(|payload| {
                            let length = u64::try_from(payload.len()).unwrap();
                            [length as u32, (length >> 32) as u32]
                        })
                        .collect())
                }
                1 => {
                    let padded_words = self
                        .encoded
                        .iter()
                        .map(|payload| words_for_bytes(payload.len()))
                        .max()
                        .unwrap();
                    assert_eq!(local.len(), padded_words);
                    Ok(self
                        .encoded
                        .iter()
                        .flat_map(|payload| encode_manifest_words(payload, padded_words))
                        .collect())
                }
                _ => panic!("manifest consensus performs exactly two gathers"),
            }
        }
    }

    fn limits() -> CommunicationTensorLimits {
        CommunicationTensorLimits::new(2, 3, 4096, None).unwrap()
    }

    fn requirement(operation: CommunicationOperation) -> CommunicationOperationRequirement {
        CommunicationOperationRequirement::tensors(
            operation,
            [TensorDtype::F32, TensorDtype::Bf16],
            if operation == CommunicationOperation::VariableAllToAll {
                CommunicationTensorLimits::new(2, 3, 4096, Some(2048)).unwrap()
            } else {
                limits()
            },
            true,
        )
        .unwrap()
    }

    fn group_requirements(operation: CommunicationOperation) -> CommunicationGroupRequirements {
        CommunicationGroupRequirements::new([requirement(operation)]).unwrap()
    }

    fn route_requirement() -> CommunicationOperationRequirement {
        requirement(CommunicationOperation::SendReceive)
    }

    fn projection_plan() -> TopologyCommunicationPlan {
        TopologyCommunicationPlan::new()
            .with_tensor_groups(group_requirements(CommunicationOperation::AllReduceSum))
            .with_expert_groups(group_requirements(CommunicationOperation::VariableAllToAll))
            .with_data_groups(group_requirements(CommunicationOperation::AllGatherEven))
            .with_pipeline_routes(route_requirement())
            .unwrap()
    }

    fn publication_requirements() -> CommunicationGroupRequirements {
        CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::tensors(
                CommunicationOperation::Broadcast,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, 8192, None).unwrap(),
                true,
            )
            .unwrap(),
            CommunicationOperationRequirement::barrier(true),
        ])
        .unwrap()
    }

    #[test]
    fn session_group_is_first_stable_world_group_with_exact_publication_requirements() {
        let topology = ParallelTopology::new(2, 2, 1, 1).unwrap();
        let plan = TopologyCommunicationPlan::new()
            .with_completion_policy(test_completion_policy())
            .with_session_group(publication_requirements())
            .with_tensor_groups(group_requirements(CommunicationOperation::AllReduceSum));
        let selected_id = plan.session_group_id().unwrap();
        let manifests = project_all_communication_manifests(topology, &plan).unwrap();

        assert_eq!(selected_id, CommunicationGroupId::new(1));
        for (rank, manifest) in manifests.iter().enumerate() {
            let session = &manifest.groups()[0];
            assert_eq!(session.id(), selected_id);
            assert_eq!(session.creation_order(), 0);
            assert_eq!(session.members(), [0, 1, 2, 3]);
            assert_eq!(session.local_index(), Some(rank));
            assert_eq!(session.requirements(), &publication_requirements());
            assert_ne!(manifest.groups()[1].id(), selected_id);
        }

        let capabilities = CommunicationCapabilities::new([
            requirement(CommunicationOperation::AllReduceSum),
            CommunicationOperationRequirement::tensors(
                CommunicationOperation::Broadcast,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, 8192, None).unwrap(),
                true,
            )
            .unwrap(),
            CommunicationOperationRequirement::barrier(true),
        ])
        .unwrap()
        .with_completion_capabilities(test_completion_capabilities());
        capabilities.validate_manifest(&manifests[0]).unwrap();

        let graph =
            crate::ExecutionGraph::new(vec![crate::ExecutionGroupSpec::root("decoder")], "decoder")
                .unwrap();
        let execution = crate::PartitionedExecutionPlan::new(
            graph,
            vec![(crate::ArchitectureGroupKind::Decoder, false)],
            vec![None],
            Vec::new(),
            Some(crate::PartitionOutputPublication {
                group: selected_id,
                owner_rank: 3,
            }),
            Some(selected_id),
            crate::PipelineWireContract::new(crate::PipelineActivationDtype::Float32),
        )
        .unwrap();
        assert_eq!(execution.publication().unwrap().group, selected_id);
        assert_eq!(execution.commit_barrier(), Some(selected_id));
    }

    #[test]
    fn group_and_manifest_validation_reject_membership_and_local_index_corruption() {
        let requirements = group_requirements(CommunicationOperation::AllReduceSum);
        assert_eq!(
            CommunicationGroupDescriptor::new(
                CommunicationGroupId::new(4),
                0,
                vec![0, 0],
                Some(0),
                requirements.clone(),
            ),
            Err(CommunicationManifestError::DuplicateGroupMember {
                id: CommunicationGroupId::new(4)
            })
        );

        let out_of_range = CommunicationGroupDescriptor::new(
            CommunicationGroupId::new(4),
            0,
            vec![0, 3],
            Some(0),
            requirements.clone(),
        )
        .unwrap();
        assert_eq!(
            CommunicationManifest::new(2, 0, vec![out_of_range], vec![]),
            Err(CommunicationManifestError::GroupMemberOutOfRange {
                id: CommunicationGroupId::new(4),
                world_size: 2,
            })
        );

        let wrong_local = CommunicationGroupDescriptor::new(
            CommunicationGroupId::new(4),
            0,
            vec![0, 1],
            Some(1),
            requirements,
        )
        .unwrap();
        assert_eq!(
            CommunicationManifest::new(2, 0, vec![wrong_local], vec![]),
            Err(CommunicationManifestError::WrongLocalIndex {
                id: CommunicationGroupId::new(4)
            })
        );
    }

    #[test]
    fn route_and_order_validation_rejects_wrong_endpoints_and_sequences() {
        let route = CommunicationRouteDescriptor::new(
            CommunicationRouteId::new(8),
            0,
            0,
            3,
            route_requirement(),
        )
        .unwrap();
        assert_eq!(
            CommunicationManifest::new(2, 0, vec![], vec![route]),
            Err(CommunicationManifestError::RouteEndpointOutOfRange {
                id: CommunicationRouteId::new(8),
                world_size: 2,
            })
        );

        let group = CommunicationGroupDescriptor::new(
            CommunicationGroupId::new(2),
            1,
            vec![0, 1],
            Some(0),
            group_requirements(CommunicationOperation::AllReduceSum),
        )
        .unwrap();
        assert_eq!(
            CommunicationManifest::new(2, 0, vec![group], vec![]),
            Err(CommunicationManifestError::WrongGroupOrder {
                id: CommunicationGroupId::new(2),
                expected: 0,
                actual: 1,
            })
        );

        let route = CommunicationRouteDescriptor::new(
            CommunicationRouteId::new(8),
            1,
            0,
            1,
            route_requirement(),
        )
        .unwrap();
        assert_eq!(
            CommunicationManifest::new(2, 0, vec![], vec![route]),
            Err(CommunicationManifestError::WrongRouteOrder {
                id: CommunicationRouteId::new(8),
                expected: 0,
                actual: 1,
            })
        );
    }

    #[test]
    fn cartesian_projection_is_deterministic_and_compatible_for_every_rank() {
        let topology = ParallelTopology::new(2, 3, 2, 2).unwrap();
        let manifests = project_all_communication_manifests(topology, &projection_plan()).unwrap();
        assert_eq!(manifests.len(), topology.world_size());
        assert_eq!(manifests[0].groups().len(), 3);
        assert_eq!(manifests[0].routes().len(), 16);
        assert_eq!(manifests[0].groups()[0].members(), [0, 2]);
        assert_eq!(manifests[0].groups()[0].local_index(), Some(0));
        assert_eq!(manifests[1].groups()[0].members(), [1, 3]);
        assert_eq!(manifests[1].groups()[0].local_index(), Some(0));
        assert_eq!(manifests[0].routes()[0].source(), 0);
        assert_eq!(manifests[0].routes()[0].destination(), 4);

        let again = project_all_communication_manifests(topology, &projection_plan()).unwrap();
        assert_eq!(manifests, again);
        validate_compatible_communication_manifests(&manifests).unwrap();
    }

    #[test]
    fn cross_rank_validation_detects_descriptor_disagreement() {
        let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
        let mut manifests = project_all_communication_manifests(
            topology,
            &TopologyCommunicationPlan::new()
                .with_completion_policy(test_completion_policy())
                .with_tensor_groups(group_requirements(CommunicationOperation::AllReduceSum)),
        )
        .unwrap();
        manifests[1].groups[0].members.swap(0, 1);
        manifests[1].groups[0].local_index = Some(0);
        assert_eq!(
            validate_compatible_communication_manifests(&manifests),
            Err(CommunicationManifestError::IncompatibleRankManifest { rank: 1 })
        );
    }

    #[test]
    fn manifest_consensus_gathers_variable_payloads_and_rejects_limit_disagreement() {
        let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
        let mut manifests = project_all_communication_manifests(
            topology,
            &TopologyCommunicationPlan::new()
                .with_completion_policy(test_completion_policy())
                .with_tensor_groups(group_requirements(CommunicationOperation::AllReduceSum)),
        )
        .unwrap();
        let rank_one = &manifests[1].groups[0];
        manifests[1].groups[0] = CommunicationGroupDescriptor::new(
            rank_one.id(),
            rank_one.creation_order(),
            rank_one.members().to_vec(),
            rank_one.local_index(),
            CommunicationGroupRequirements::new([CommunicationOperationRequirement::tensors(
                CommunicationOperation::AllReduceSum,
                [TensorDtype::F32, TensorDtype::Bf16],
                CommunicationTensorLimits::new(2, 3, 32, None).unwrap(),
                true,
            )
            .unwrap()])
            .unwrap(),
        )
        .unwrap();
        let transport = ScriptedManifestTransport::new(&manifests);

        assert_eq!(
            validate_communication_manifest_consensus(&transport, &manifests[0]),
            Err(CommunicationManifestConsensusError::Manifest(
                CommunicationManifestError::IncompatibleRankManifest { rank: 1 }
            ))
        );
        assert_eq!(transport.calls.get(), 2);
        assert_ne!(transport.encoded[0].len(), transport.encoded[1].len());
    }

    #[test]
    fn manifest_consensus_accepts_complete_compatible_rank_artifacts() {
        let manifests = project_all_communication_manifests(
            ParallelTopology::new(2, 1, 1, 1).unwrap(),
            &TopologyCommunicationPlan::new()
                .with_tensor_groups(group_requirements(CommunicationOperation::AllReduceSum)),
        )
        .unwrap();
        let transport = ScriptedManifestTransport::new(&manifests);

        validate_communication_manifest_consensus(&transport, &manifests[0]).unwrap();
        assert_eq!(transport.calls.get(), 2);
    }

    #[test]
    fn manifest_consensus_defers_world_validation_until_after_both_gathers() {
        let manifests = vec![
            CommunicationManifest::new(2, 0, Vec::new(), Vec::new()).unwrap(),
            CommunicationManifest::new(3, 1, Vec::new(), Vec::new()).unwrap(),
        ];
        let transport = ScriptedManifestTransport::new(&manifests);

        assert_eq!(
            validate_communication_manifest_consensus(&transport, &manifests[0]),
            Err(CommunicationManifestConsensusError::Manifest(
                CommunicationManifestError::IncompatibleRankManifest { rank: 1 }
            ))
        );
        assert_eq!(transport.calls.get(), 2);
    }

    #[test]
    fn capability_validation_is_fine_grained_and_fail_closed() {
        let topology = ParallelTopology::new(2, 1, 1, 1).unwrap();
        let manifest = project_all_communication_manifests(
            topology,
            &TopologyCommunicationPlan::new()
                .with_completion_policy(test_completion_policy())
                .with_tensor_groups(group_requirements(CommunicationOperation::AllReduceSum)),
        )
        .unwrap()
        .remove(0);

        let missing = CommunicationCapabilities::new([])
            .unwrap()
            .with_completion_capabilities(test_completion_capabilities());
        assert_eq!(
            missing.validate_manifest(&manifest),
            Err(CommunicationCapabilityError::MissingOperation {
                operation: CommunicationOperation::AllReduceSum,
            })
        );

        let narrow = CommunicationCapabilities::new([CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 2, 32, None).unwrap(),
            false,
        )
        .unwrap()])
        .unwrap()
        .with_completion_capabilities(test_completion_capabilities());
        assert_eq!(
            narrow.validate_manifest(&manifest),
            Err(CommunicationCapabilityError::UnsupportedDtype {
                operation: CommunicationOperation::AllReduceSum,
                dtype: TensorDtype::Bf16,
            })
        );
    }

    #[test]
    fn capability_validation_covers_gather_result_size_separately() {
        let required = CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllGatherUneven,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, 32, None)
                .unwrap()
                .with_output_tensor_elements(64)
                .unwrap(),
            true,
        )
        .unwrap();
        let available =
            CommunicationCapabilities::new([CommunicationOperationRequirement::tensors(
                CommunicationOperation::AllGatherUneven,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, 64, None).unwrap(),
                true,
            )
            .unwrap()])
            .unwrap();
        assert!(available.validate_requirement(&required).is_ok());

        let insufficient =
            CommunicationCapabilities::new([CommunicationOperationRequirement::tensors(
                CommunicationOperation::AllGatherUneven,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, 32, None).unwrap(),
                true,
            )
            .unwrap()])
            .unwrap();
        assert_eq!(
            insufficient.validate_requirement(&required),
            Err(CommunicationCapabilityError::InsufficientLimits {
                operation: CommunicationOperation::AllGatherUneven,
            })
        );
    }

    #[test]
    fn backend_callbacks_receive_only_opaque_descriptors_in_declared_order() {
        let topology = ParallelTopology::new(2, 2, 1, 1).unwrap();
        let manifest = project_all_communication_manifests(topology, &projection_plan())
            .unwrap()
            .remove(3);

        let groups = manifest
            .try_create_groups(|descriptor| {
                Ok::<_, std::convert::Infallible>((
                    descriptor.id(),
                    descriptor.creation_order(),
                    descriptor.members().to_vec(),
                    descriptor.local_index(),
                ))
            })
            .unwrap();
        assert!(groups
            .iter()
            .enumerate()
            .all(|(index, (_, order, _, _))| index == *order));

        let routes = manifest
            .try_create_routes(|descriptor| {
                Ok::<_, std::convert::Infallible>((
                    descriptor.id(),
                    descriptor.source(),
                    descriptor.destination(),
                    descriptor.requirement().operation(),
                ))
            })
            .unwrap();
        assert!(routes.iter().all(|(_, source, destination, operation)| {
            source != destination && *operation == CommunicationOperation::SendReceive
        }));
    }

    #[test]
    fn variable_peer_counts_preserve_zeroes_and_exact_member_order() {
        let counts = CommunicationPeerCounts::new(vec![0, 3, 1], vec![2, 0, 2], 3).unwrap();
        assert_eq!(counts.send(), [0, 3, 1]);
        assert_eq!(counts.receive(), [2, 0, 2]);
        assert_eq!(counts.group_size(), 3);
        assert_eq!(
            CommunicationPeerCounts::new(vec![1], vec![1, 0], 2),
            Err(CommunicationManifestError::InvalidPeerCounts {
                group_size: 2,
                send: 1,
                receive: 2,
            })
        );
    }

    #[test]
    fn failure_agreement_is_payload_free_and_not_covered_by_a_barrier() {
        let required = CommunicationOperationRequirement::failure_agreement(true);
        assert_eq!(
            required.operation(),
            CommunicationOperation::FailureAgreement
        );
        assert!(required.dtypes().is_empty());
        assert_eq!(required.limits(), None);

        let barrier_only =
            CommunicationCapabilities::new([CommunicationOperationRequirement::barrier(true)])
                .unwrap();
        assert_eq!(
            barrier_only.validate_requirement(&required),
            Err(CommunicationCapabilityError::MissingOperation {
                operation: CommunicationOperation::FailureAgreement,
            })
        );
    }

    #[test]
    fn completion_policy_deserialization_revalidates_positive_deadline() {
        assert!(
            serde_json::from_value::<CommunicationCompletionPolicy>(serde_json::json!({
                "timeout_millis": 0,
                "cancellation": "quarantine_until_complete"
            }))
            .is_err()
        );
    }

    #[test]
    fn communication_resources_require_selected_supported_completion() {
        let group = CommunicationGroupDescriptor::new(
            CommunicationGroupId::new(1),
            0,
            vec![0],
            Some(0),
            CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)])
                .unwrap(),
        )
        .unwrap();
        let manifest = CommunicationManifest::new(1, 0, vec![group], Vec::new()).unwrap();
        let capabilities =
            CommunicationCapabilities::new([CommunicationOperationRequirement::barrier(true)])
                .unwrap()
                .with_completion_capabilities(
                    CommunicationCompletionCapabilities::new([
                        CompletionCancellationMode::QuarantineUntilComplete,
                    ])
                    .unwrap(),
                );
        assert_eq!(
            capabilities.validate_manifest(&manifest),
            Err(CommunicationCapabilityError::MissingSelectedCompletionPolicy)
        );
        let manifest = manifest.with_completion_policy(
            CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(1),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
        capabilities.validate_manifest(&manifest).unwrap();
    }

    #[test]
    fn role_exact_frames_use_actual_variable_dimensions_and_reject_role_or_fixed_drift() {
        let admitted = RoleExactBoundaryContract::new(
            "nemotron_h.target",
            [
                BoundaryRoleContract::symbolic(
                    "hidden",
                    TensorDtype::F32,
                    vec![
                        BoundaryDimensionContract::Variable { maximum: 4 },
                        BoundaryDimensionContract::Variable { maximum: 16 },
                        BoundaryDimensionContract::Fixed(12),
                    ],
                )
                .unwrap(),
                BoundaryRoleContract::symbolic(
                    "embedded",
                    TensorDtype::F32,
                    vec![
                        BoundaryDimensionContract::Variable { maximum: 4 },
                        BoundaryDimensionContract::Variable { maximum: 16 },
                        BoundaryDimensionContract::Fixed(12),
                    ],
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let actual = vec![
            BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![1, 3, 12]).unwrap(),
            BoundaryRoleContract::new("embedded", TensorDtype::F32, vec![1, 3, 12]).unwrap(),
        ];
        let framed = admitted
            .frame_values(CommunicationRouteId::new(7), &actual, vec![1_u8, 2])
            .unwrap();
        assert_ne!(framed[0].header(), framed[1].header());

        let swapped = vec![actual[1].clone(), actual[0].clone()];
        assert_eq!(
            admitted.frame_values(CommunicationRouteId::new(7), &swapped, vec![1_u8, 2]),
            Err(CommunicationManifestError::InvalidBoundaryContract)
        );
        let fixed_shrink = vec![
            BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![1, 3, 11]).unwrap(),
            actual[1].clone(),
        ];
        assert_eq!(
            admitted.frame_values(CommunicationRouteId::new(7), &fixed_shrink, vec![1_u8, 2],),
            Err(CommunicationManifestError::InvalidBoundaryContract)
        );
    }

    #[test]
    fn role_exact_framing_capability_and_byte_overflow_fail_before_submission() {
        let requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::SendReceive,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, 4096, None).unwrap(),
            true,
        )
        .unwrap();
        let role = BoundaryRoleContract::symbolic(
            "hidden",
            TensorDtype::F32,
            vec![BoundaryDimensionContract::Fixed(1)],
        )
        .unwrap();
        let route = CommunicationRouteDescriptor::new(
            CommunicationRouteId::new(3),
            0,
            0,
            1,
            requirement.clone(),
        )
        .unwrap()
        .with_boundary_contract(RoleExactBoundaryContract::new("none", [role]).unwrap())
        .unwrap();
        let manifest = CommunicationManifest::new(2, 0, Vec::new(), vec![route])
            .unwrap()
            .with_completion_policy(
                CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(1),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            );
        let capability = CommunicationCapabilities::new([requirement.clone()])
            .unwrap()
            .with_completion_capabilities(
                CommunicationCompletionCapabilities::new([
                    CompletionCancellationMode::QuarantineUntilComplete,
                ])
                .unwrap(),
            );
        assert!(matches!(
            capability.validate_manifest(&manifest),
            Err(CommunicationCapabilityError::MissingBoundaryFraming {
                protocol: BoundaryFramingProtocol::RoleExactV1,
            })
        ));

        for invalid in [
            BoundaryRoleContract::new("hidden", TensorDtype::I32, vec![1]).unwrap(),
            BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![1, 1, 1, 1]).unwrap(),
            BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![4097]).unwrap(),
        ] {
            assert_eq!(
                CommunicationRouteDescriptor::new(
                    CommunicationRouteId::new(30),
                    0,
                    0,
                    1,
                    requirement.clone(),
                )
                .unwrap()
                .with_boundary_contract(
                    RoleExactBoundaryContract::new("invalid", [invalid]).unwrap(),
                ),
                Err(CommunicationManifestError::InvalidBoundaryContract),
            );
        }
        let inexact = CommunicationOperationRequirement::tensors(
            CommunicationOperation::SendReceive,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, 4096, None).unwrap(),
            false,
        )
        .unwrap();
        assert_eq!(
            CommunicationRouteDescriptor::new(CommunicationRouteId::new(31), 0, 0, 1, inexact,)
                .unwrap()
                .with_boundary_contract(
                    RoleExactBoundaryContract::new(
                        "inexact",
                        [BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![1]).unwrap()],
                    )
                    .unwrap(),
                ),
            Err(CommunicationManifestError::InvalidBoundaryContract),
        );

        let exact = RoleExactBoundaryContract::new(
            "overflow",
            [BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![usize::MAX, 2]).unwrap()],
        )
        .unwrap();
        assert_eq!(
            exact.frame_values(CommunicationRouteId::new(4), exact.roles(), vec![0_u8],),
            Err(CommunicationManifestError::InvalidBoundaryContract)
        );
    }

    #[test]
    fn role_exact_contract_deserialization_revalidates_roles_dimensions_and_dtypes() {
        let contract = RoleExactBoundaryContract::new(
            "fixture",
            [BoundaryRoleContract::symbolic(
                "hidden",
                TensorDtype::F32,
                vec![BoundaryDimensionContract::Fixed(4)],
            )
            .unwrap()],
        )
        .unwrap();
        let valid = serde_json::to_value(&contract).unwrap();

        let mut empty_role = valid.clone();
        empty_role["roles"][0]["role"] = serde_json::json!("");
        assert!(serde_json::from_value::<RoleExactBoundaryContract>(empty_role).is_err());

        let mut duplicate_role = valid.clone();
        let repeated = duplicate_role["roles"][0].clone();
        duplicate_role["roles"]
            .as_array_mut()
            .unwrap()
            .push(repeated);
        assert!(serde_json::from_value::<RoleExactBoundaryContract>(duplicate_role).is_err());

        let mut zero_dimension = valid.clone();
        zero_dimension["roles"][0]["shape"][0] = serde_json::json!({ "fixed": 0 });
        assert!(serde_json::from_value::<RoleExactBoundaryContract>(zero_dimension).is_err());

        let mut encoded_dtype = valid;
        encoded_dtype["roles"][0]["dtype"] = serde_json::json!({ "encoded": "q4" });
        assert!(serde_json::from_value::<RoleExactBoundaryContract>(encoded_dtype).is_err());
    }

    #[test]
    fn route_submission_waves_require_complete_disjoint_world_batches() {
        let requirement = CommunicationOperationRequirement::tensors(
            CommunicationOperation::SendReceive,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, 64, None).unwrap(),
            true,
        )
        .unwrap();
        let boundary = RoleExactBoundaryContract::new(
            "hidden-v1",
            [BoundaryRoleContract::new("hidden", TensorDtype::F32, vec![1, 2, 4]).unwrap()],
        )
        .unwrap();
        let route = |order: usize, source: usize, destination: usize| {
            CommunicationRouteDescriptor::new(
                CommunicationRouteId::new(order as u64),
                order,
                source,
                destination,
                requirement.clone(),
            )
            .unwrap()
            .with_boundary_contract(boundary.clone())
            .unwrap()
        };

        let complete = CommunicationManifest::new(
            8,
            0,
            Vec::new(),
            (0..4).map(|rank| route(rank, rank + 4, rank)).collect(),
        )
        .unwrap();
        assert_eq!(complete.route_submission_waves().len(), 1);
        assert_eq!(complete.route_submission_waves()[0], 0..4);

        let omitted = CommunicationManifest::new(
            8,
            0,
            Vec::new(),
            (0..3).map(|rank| route(rank, rank + 4, rank)).collect(),
        )
        .unwrap();
        assert_eq!(omitted.route_submission_waves(), [0..1, 1..2, 2..3]);

        let overlapping = CommunicationManifest::new(
            8,
            0,
            Vec::new(),
            vec![
                route(0, 4, 0),
                route(1, 4, 1),
                route(2, 6, 2),
                route(3, 7, 3),
            ],
        )
        .unwrap();
        assert_eq!(
            overlapping.route_submission_waves(),
            [0..1, 1..2, 2..3, 3..4]
        );
    }
}
