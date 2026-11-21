//! Narrow capability contracts implemented by execution backends.

use crate::{
    checkpoint::TensorDtype, BoundedCompletion, Completion, Submission, SubmissionBackend,
};
use eredu_nn::NeuralBackend;
use serde::{Deserialize, Serialize};

/// Backend facts and transport for a single portable communication setup.
pub trait CommunicationSetupTransport: crate::consensus::ConsensusTransport {
    /// Fresh nonce for this setup, or no nonce when local issuance fails.
    fn setup_nonce(&self) -> Option<[u8; 32]>;
    /// Opaque operation, framing and completion capabilities.
    fn capabilities(&self) -> CommunicationCapabilities;
    /// Exact native group and route reachability.
    fn topology_capabilities(&self) -> CommunicationTopologyCapabilities;
}

/// Collective operations available to distributed runtime policies.
pub trait CollectiveBackend: SubmissionBackend {
    /// Backend-native collective group.
    type Group: ?Sized;
    /// Backend-specific collective failure.
    type CollectiveError: std::error::Error + Send + Sync + 'static;

    /// Reduces a tensor across the selected group.
    fn all_reduce(
        value: Self::Tensor,
        group: &Self::Group,
        executor: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError>;

    /// Gathers a tensor across the selected group.
    fn all_gather(
        value: Self::Tensor,
        group: &Self::Group,
        executor: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError>;

    /// Exchanges tensor partitions across the selected group.
    fn all_to_all(
        value: Self::Tensor,
        group: &Self::Group,
        executor: &Self::Executor,
    ) -> Result<Self::Tensor, Self::CollectiveError>;
}

/// Common opaque handles and exact completion used by communication extensions.
///
/// This trait deliberately declares no operation. Backends implement only the
/// fine-grained operation traits selected for a concrete architecture.
pub trait CommunicationBackend: SubmissionBackend {
    /// Backend-native realization of one opaque communication group.
    type CommunicationGroup: ?Sized;
    /// Backend-native realization of one opaque directed route.
    type CommunicationRoute: ?Sized;
    /// Exact completion retaining tensors, buffers, groups, routes, and streams.
    type CommunicationCompletion: Completion<Error = Self::CommunicationError>
        + BoundedCompletion<Error = Self::CommunicationError>;
    /// Stable mechanism failure with no architecture-family policy.
    type CommunicationError: std::error::Error + Send + Sync + 'static;

    /// Submits evaluation of rank-local tensor dependencies before a
    /// communication-readiness agreement.
    ///
    /// The returned communication completion must retain every submitted
    /// tensor and native execution resource through exact completion or safe
    /// cancellation teardown. This operation does not select or infer a
    /// collective group.
    fn submit_local_dependencies<'a, I>(
        values: I,
        executor: &Self::Executor,
    ) -> Result<Submission<(), Self::CommunicationCompletion>, Self::CommunicationError>
    where
        Self::Tensor: 'a,
        I: IntoIterator<Item = &'a Self::Tensor>;
}

/// Sum reduction on an opaque communication group.
pub trait SumReductionBackend: CommunicationBackend {
    /// Submits one elementwise sum and returns its exact completion.
    fn all_reduce_sum(
        value: Self::Tensor,
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<Submission<Self::Tensor, Self::CommunicationCompletion>, Self::CommunicationError>;
}

/// Equal-size gathering on an opaque communication group.
pub trait EvenGatherBackend: CommunicationBackend {
    /// Gathers equal-sized values and concatenates in member order on `axis`.
    fn all_gather_even(
        value: Self::Tensor,
        axis: usize,
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<Submission<Self::Tensor, Self::CommunicationCompletion>, Self::CommunicationError>;
}

/// Unequal-size gathering on an opaque communication group.
pub trait UnevenGatherBackend: CommunicationBackend {
    /// Gathers values and concatenates in member order using exact element counts.
    fn all_gather_uneven(
        value: Self::Tensor,
        counts: &[usize],
        axis: usize,
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<Submission<Self::Tensor, Self::CommunicationCompletion>, Self::CommunicationError>;
}

/// Variable-count exchange on an opaque communication group.
pub trait VariableAllToAllBackend: CommunicationBackend {
    /// Exchanges exact per-peer partitions on `axis` and returns exact completion.
    fn variable_all_to_all(
        value: Self::Tensor,
        counts: &CommunicationPeerCounts,
        axis: usize,
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<Submission<Self::Tensor, Self::CommunicationCompletion>, Self::CommunicationError>;
}

/// Ordered point-to-point boundary transfer on one opaque route.
pub trait PointToPointBackend: CommunicationBackend {
    /// Sends or receives the route's exact ordered tensor bundle.
    #[allow(
        clippy::type_complexity,
        reason = "the signature exposes the tensor bundle and exact completion without erasure"
    )]
    fn send_receive(
        values: Vec<RoleExactBoundaryValue<Self::Tensor>>,
        route: &Self::CommunicationRoute,
        executor: &Self::Executor,
    ) -> Result<
        Submission<Vec<Self::Tensor>, Self::CommunicationCompletion>,
        Self::CommunicationError,
    >;
}

/// One logical boundary tensor coupled to the exact in-band header that must
/// be transmitted with its payload.
///
/// A point-to-point implementation must place `header` and the byte
/// representation of `tensor` in the same native message. On receive it must
/// compare the bytes actually received with `header` before its completion can
/// report success. Returning a backend-synthesized tag does not satisfy this
/// contract.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RoleExactBoundaryValue<T> {
    header: Vec<u8>,
    tensor: T,
}

impl<T> RoleExactBoundaryValue<T> {
    /// Couples a tensor to the exact header chosen by the portable framing protocol.
    pub fn new(header: Vec<u8>, tensor: T) -> Self {
        Self { header, tensor }
    }

    /// Exact expected in-band header bytes.
    pub fn header(&self) -> &[u8] {
        &self.header
    }

    /// Logical tensor payload.
    pub const fn tensor(&self) -> &T {
        &self.tensor
    }

    /// Consumes the framed value into its expected header and payload.
    pub fn into_parts(self) -> (Vec<u8>, T) {
        (self.header, self.tensor)
    }
}

/// Root-to-group publication on an opaque communication group.
pub trait BroadcastBackend: CommunicationBackend {
    /// Broadcasts one tensor from an ordered member index.
    fn broadcast(
        value: Self::Tensor,
        root: usize,
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<Submission<Self::Tensor, Self::CommunicationCompletion>, Self::CommunicationError>;
}

/// Payload-free agreement on an opaque communication group.
pub trait BarrierBackend: CommunicationBackend {
    /// Submits a barrier and returns its exact completion.
    fn barrier(
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<Self::CommunicationCompletion, Self::CommunicationError>;
}

/// All-rank success-status agreement on an opaque communication group.
///
/// Unlike a barrier, this operation carries one boolean status from every
/// member and returns `true` only when every submitted status was `true` and
/// any supplied fixed descriptor agrees exactly across members.
pub trait FailureAgreementBackend: CommunicationBackend {
    /// Backend-owned result whose host boolean becomes authoritative only after
    /// exact communication completion.
    type FailureAgreementOutput;

    /// Submits one local phase status without reading the lazy result eagerly.
    /// Members must choose the same descriptor mode for a collective phase.
    /// A descriptor compares all integer bits without float conversion or hashing;
    /// even a locally failed participant must submit its fixed descriptor.
    fn agree_success(
        local_success: bool,
        descriptor: Option<[u64; 8]>,
        group: &Self::CommunicationGroup,
        executor: &Self::Executor,
    ) -> Result<
        Submission<Self::FailureAgreementOutput, Self::CommunicationCompletion>,
        Self::CommunicationError,
    >;

    /// Resolves the completed backend result without starting new native work.
    fn resolve_failure_agreement(
        output: Self::FailureAgreementOutput,
    ) -> Result<bool, Self::CommunicationError>;
}

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
    ) -> Result<Self, CommunicationPeerCountsError> {
        if group_size == 0 || send.len() != group_size || receive.len() != group_size {
            return Err(CommunicationPeerCountsError {
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

/// Variable exchange counts do not match opaque group membership.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("communication peer counts have send length {send} and receive length {receive}, expected {group_size}")]
pub struct CommunicationPeerCountsError {
    /// Expected opaque group cardinality.
    pub group_size: usize,
    /// Supplied send-count cardinality.
    pub send: usize,
    /// Supplied receive-count cardinality.
    pub receive: usize,
}

/// Backend-native tensor metadata used only for mechanical communication validation.
pub trait CommunicationTensorMetadata<B: NeuralBackend> {
    /// Returns the portable logical dtype of one native tensor.
    fn dtype(&self, tensor: &B::Tensor) -> TensorDtype;

    /// Returns the exact logical shape of one native tensor.
    fn shape(&self, tensor: &B::Tensor) -> Vec<usize>;
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

/// Logical scalar kind carried by one architecture-owned boundary tensor.
///
/// `Activation` is resolved by a concrete backend to the execution dtype
/// selected for the surrounding pipeline activation. Integer kinds are exact.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum BoundaryTensorDtype {
    /// Exact FP32 auxiliary values, independent of activation transport precision.
    Float32,
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

/// Mechanical backend allocation used only for receive/publication placeholders.
pub trait PartitionTensorAllocator<B>
where
    B: eredu_nn::NeuralBackend,
{
    /// Converts one source activation to the selected pipeline wire dtype.
    ///
    /// This runs while the source execution group is still local work, before
    /// distributed readiness agreement. Implementations must reject unsupported
    /// conversions instead of leaving a native-dtype tensor for the route.
    fn tensor_to_wire(
        &mut self,
        _tensor: B::Tensor,
        _logical_dtype: BoundaryTensorDtype,
        _activation_dtype: PipelineActivationDtype,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error> {
        Err(eredu_nn::Error::backend(
            "partition tensor allocator does not support pipeline wire conversion",
        ))
    }

    /// Allocates an activation tensor with the selected pipeline wire dtype.
    fn tensor_placeholder(
        &mut self,
        shape: &[i32],
        logical_dtype: BoundaryTensorDtype,
        activation_dtype: PipelineActivationDtype,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, eredu_nn::Error>;
}

impl PipelineActivationDtype {
    /// Returns the neural element type used to prepare outbound activations.
    pub const fn element_type(self) -> eredu_nn::TensorElementType {
        match self {
            Self::Float16 => eredu_nn::TensorElementType::F16,
            Self::Bfloat16 => eredu_nn::TensorElementType::Bf16,
            Self::Float32 => eredu_nn::TensorElementType::F32,
        }
    }

    /// Returns the dtype recorded in the transport descriptor.
    pub const fn tensor_dtype(self) -> TensorDtype {
        match self {
            Self::Float16 => TensorDtype::F16,
            Self::Bfloat16 => TensorDtype::Bf16,
            Self::Float32 => TensorDtype::F32,
        }
    }
}

impl BoundaryTensorDtype {
    /// Resolves a logical boundary dtype against the selected activation precision.
    pub const fn tensor_dtype(self, activation: PipelineActivationDtype) -> TensorDtype {
        match self {
            Self::Activation => activation.tensor_dtype(),
            Self::Float32 => TensorDtype::F32,
            Self::Uint32 => TensorDtype::U32,
            Self::Int32 => TensorDtype::I32,
        }
    }
}

mod manifest;
pub use manifest::{
    validate_compatible_communication_manifests, BoundaryDimensionContract,
    BoundaryFramingProtocol, BoundaryRoleContract, CommunicationCapabilities,
    CommunicationCapabilityError, CommunicationCompletionCapabilities,
    CommunicationCompletionPolicy, CommunicationGroupDescriptor, CommunicationGroupRequirements,
    CommunicationManifest, CommunicationManifestError, CommunicationOperationRequirement,
    CommunicationRouteDescriptor, CommunicationRouteId, CommunicationTensorLimits,
    CommunicationTopologyCapabilities, RoleExactBoundaryContract,
};

mod binding;
/// Native resources for portable component pipelines.
pub mod pipeline;
pub use binding::{
    CommunicationBindingError, CommunicationBindingGeometryError, CommunicationBindings,
    RankCommunication, RankCommunicationResources, RealizedCommunicationGroup,
    RealizedCommunicationRoute, WorldCommunicationBinding,
};

mod authority;
pub use authority::{CommunicationAuthority, CommunicationError, CommunicationPoisoned};

/// Common identity of one successful communication setup, not execution authority.
///
/// It distinguishes repeated setups with identical manifests when native
/// composition supplies fresh instance nonces. Copying or serializing this host
/// value never grants access to a communicator, model, run, or capture quota.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize)]
pub struct CommunicationSessionIdentity {
    digest: [u8; 32],
    participants: std::num::NonZeroUsize,
}

impl CommunicationSessionIdentity {
    /// Records the digest and participant count of a validated setup transcript.
    pub const fn from_transcript_digest(
        digest: [u8; 32],
        participants: std::num::NonZeroUsize,
    ) -> Self {
        Self {
            digest,
            participants,
        }
    }

    /// Fixed-width identity suitable for bounded host provenance records.
    pub const fn bytes(&self) -> &[u8; 32] {
        &self.digest
    }
    /// Participant count committed by this exact setup transcript.
    pub const fn participant_count(&self) -> usize {
        self.participants.get()
    }
}

impl std::fmt::Display for CommunicationSessionIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.digest {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}
