//! Versioned logical architecture and capture discovery, independent of execution storage.
//!
//! Node IDs, canonical parameter group prefixes, and activation paths are separate
//! namespaces. A node need not have an observation point. `None` means unknown,
//! never zero; completeness describes omissions at the stated scope.

use serde::{Deserialize, Serialize};

/// Current wire schema for architecture and observation discovery.
pub const DISCOVERY_SCHEMA_VERSION: u32 = 1;

/// Coverage of a descriptor, node, or catalog.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", content = "reasons", rename_all = "snake_case")]
pub enum DescriptionCompleteness {
    /// All details in the documented logical abstraction are represented.
    Complete,
    /// Known omissions; represented facts remain authoritative.
    Partial(Vec<String>),
    /// No supported description at this scope.
    Unsupported(Vec<String>),
}

/// Semantic dimension; symbols refer to the current operation, not a fixed batch.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum SymbolicDimension {
    /// An exact positive or zero extent.
    Known(usize),
    /// Batch size of the current operation.
    Batch,
    /// Sequence length of this operation, usually one for decode.
    Sequence,
    /// Flattened batch times sequence rows, used by expert routing tensors.
    TokenRows,
    /// Visible prefix length including cached positions.
    Context,
    /// Input-dependent media feature positions.
    MediaPositions,
    /// Extent is not described.
    Unknown,
}

/// Named tensor axis in storage order.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct TensorAxis {
    /// Semantic axis name; independent of checkpoint field spelling.
    pub name: String,
    /// Known extent or operation-dependent symbolic dimension.
    pub dimension: SymbolicDimension,
}

/// Operation class, interpreted without matching a model-family name.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ArchitectureNodeKind {
    /// Token or codebook embedding lookup.
    Embedding,
    /// One architecture-defined decoder execution unit.
    DecoderBlock,
    /// Normalization of activations.
    Normalization,
    /// An attention operation.
    Attention,
    /// A stateful token mixer.
    Mixer,
    /// Dense feed-forward computation.
    FeedForward,
    /// Routed and optional shared expert computation.
    MixtureOfExperts,
    /// Expert selection and coefficient computation.
    Router,
    /// Bank of selected experts.
    RoutedExperts,
    /// Always-on expert branch.
    SharedExperts,
    /// Addition of a sublayer contribution and its bypass input.
    ResidualAdd,
    /// Combination of parallel branch contributions.
    Sum,
    /// Projection into an output domain such as vocabulary logits.
    OutputHead,
    /// Host or tensor input processing.
    Processor,
    /// Media feature encoder.
    Encoder,
    /// Projection of encoded features into decoder space.
    Projector,
    /// Assembly of text and media features.
    ModalityMerge,
    /// Checkpoint-embedded prediction component.
    Prediction,
    /// Temporal/depth frame execution.
    Realtime,
    /// Operation whose internal semantics are not described.
    Opaque,
}

/// Query/key-value head sharing, independent of receptive field and mechanism.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadSharing {
    /// One key/value head per query head.
    MultiHead,
    /// A single shared key/value head.
    MultiQuery,
    /// Multiple query heads share each of several key/value heads.
    GroupedQuery,
}

/// Reach of one attention operation.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReceptiveField {
    /// The complete causal prefix.
    Full,
    /// A bounded trailing window.
    Sliding {
        /// Maximum number of visible positions.
        window: usize,
    },
    /// A local neighborhood.
    Local {
        /// Neighborhood size, when known.
        window: Option<usize>,
    },
}

/// Attention equation class; recurrent linear attention can use both flags below.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionMechanism {
    /// Softmax over all candidate scores.
    Softmax,
    /// Linear attention with optional recurrent state.
    Linear,
    /// Attention over compressed latent key/value representations.
    Latent,
    /// Attention over local and compressed sparse history.
    CompressedSparse,
}

/// Positional encoding class. Exact scaling may remain outside this abstraction.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionalEncoding {
    /// Rotary position encoding.
    Rotary,
    /// Relative positional features.
    Relative,
    /// Learned positional embeddings.
    Learned,
    /// No such transformation is applied.
    None,
}

/// Independent optional semantic facts for one layer's attention.
#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct AttentionAttributes {
    /// Relationship between query heads and key/value heads, when applicable.
    pub head_sharing: Option<HeadSharing>,
    /// Number of query heads.
    pub query_heads: Option<usize>,
    /// Number of key/value heads.
    pub key_value_heads: Option<usize>,
    /// Per-head query/key width.
    pub key_head_dimension: Option<usize>,
    /// Per-head value width.
    pub value_head_dimension: Option<usize>,
    /// Positions visible to this layer; unknown for mixed compressed-history policies.
    pub receptive_field: Option<ReceptiveField>,
    /// Equation class of this operator.
    pub mechanism: Option<AttentionMechanism>,
    /// Whether execution maintains recurrent state.
    pub recurrent: Option<bool>,
    /// Whether future positions are masked.
    pub causal: Option<bool>,
    /// Position encoding used by this layer.
    pub positional_encoding: Option<PositionalEncoding>,
}

/// Stateful token-mixing equation class.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MixerMechanism {
    /// Gated delta recurrent attention.
    GatedDelta,
    /// Selective state-space mixer.
    SelectiveStateSpace,
    /// Gated causal short convolution.
    ShortConvolution,
}

/// Optional geometry for a stateful token mixer.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct MixerAttributes {
    /// Equation class of this operator.
    pub mechanism: MixerMechanism,
    /// Whether execution maintains recurrent state.
    pub recurrent: bool,
    /// Causal convolution kernel width, when present.
    pub convolution_width: Option<usize>,
}

/// Semantic granularity of expert selection.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingGranularity {
    /// One decision per token.
    Token,
    /// One decision per sequence.
    Sequence,
}

/// Router score transformation, independent of top-k normalization.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingScoreTransform {
    /// Softmax over all candidate scores.
    Softmax,
    /// Softmax applied only after top-k selection.
    SelectedSoftmax,
    /// Independent sigmoid-transformed scores.
    Sigmoid,
    /// Square root of softplus-transformed scores.
    SqrtSoftplus,
    /// Untransformed scores.
    Identity,
}

/// Normalization policy for selected route coefficients.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingNormalization {
    /// Selected scores are normalized by their sum.
    SelectedSum,
    /// Joint normalization over selected routed experts and always-on experts.
    SelectedAndSharedSum,
    /// No such transformation is applied.
    None,
}

/// Routed and always-on experts are distinct branches, also linked by graph edges.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct MoeAttributes {
    /// Number of independently routed experts.
    pub routed_experts: usize,
    /// Number of experts selected per routing decision.
    pub selected_experts: usize,
    /// Number of always-on experts; zero explicitly means none.
    pub shared_experts: Option<usize>,
    /// Intermediate width of the shared branch, when known.
    pub shared_expert_width: Option<usize>,
    /// Whether a learned gate scales the shared contribution.
    pub shared_expert_gated: Option<bool>,
    /// Unit on which expert decisions are made.
    pub granularity: Option<RoutingGranularity>,
    /// Transformation of router logits into scores.
    pub score_transform: Option<RoutingScoreTransform>,
    /// Normalization applied after selecting experts.
    pub normalization: Option<RoutingNormalization>,
}

/// A semantic node; containment is separate from data-flow edges.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureNode {
    /// Stable identity within this descriptor.
    pub id: String,
    /// Human-readable operation label.
    pub label: String,
    /// Typed semantic category.
    pub kind: ArchitectureNodeKind,
    /// Containing node, independent of data-flow dependencies.
    pub parent: Option<String>,
    /// Zero-based physical layer ordinal, if this node is a decoder unit.
    pub layer_index: Option<usize>,
    /// Canonical parameter-group identities referenced by this node.
    pub parameter_groups: Vec<String>,
    /// Exact activation selectors associated with this node.
    pub observation_paths: Vec<String>,
    /// Output axes in storage order; absent when rank or shape semantics are unknown.
    pub output_axes: Option<Vec<TensorAxis>>,
    /// Independent attention properties for this exact layer.
    pub attention: Option<AttentionAttributes>,
    /// Stateful mixer properties, when applicable.
    pub mixer: Option<MixerAttributes>,
    /// Expert topology and routing policy, when applicable.
    pub moe: Option<MoeAttributes>,
    /// Explicit omissions at this scope.
    pub completeness: DescriptionCompleteness,
}

/// Semantic role of a logical graph edge.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchitectureEdgeKind {
    /// Ordinary activation flow.
    Data,
    /// Bypass activation consumed by a residual join.
    Residual,
    /// Expert-selection control flow.
    Routing,
    /// Persistent state dependency.
    State,
}

/// Directed logical data flow, including the bypass input of a residual addition.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureEdge {
    /// Source node identity.
    pub from: String,
    /// Destination node identity.
    pub to: String,
    /// Typed semantic category.
    pub kind: ArchitectureEdgeKind,
}

/// Architecture-declared canonical parameter namespace, not an activation selector.
/// Physical checkpoint aliases and encoding companions remain in checkpoint schemas.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureParameterGroup {
    /// Stable identity within this descriptor.
    pub id: String,
    /// Canonical logical checkpoint module prefix; not a physical source key or activation selector.
    pub canonical_prefix: String,
}

/// Backend-independent logical graph from an admitted architecture plan.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureDescriptor {
    /// Version of the discovery wire schema.
    pub schema_version: u32,
    /// Logical operations and their containment.
    pub nodes: Vec<ArchitectureNode>,
    /// Directed logical data flow.
    pub edges: Vec<ArchitectureEdge>,
    /// Canonical parameter groups referenced by nodes in this graph.
    pub parameter_groups: Vec<ArchitectureParameterGroup>,
    /// Architecture-declared observation points, independent of execution support.
    pub observations: ObservationCatalog,
    /// Explicit omissions at this scope.
    pub completeness: DescriptionCompleteness,
}

impl ArchitectureDescriptor {
    /// Finds a node by its stable descriptor identity.
    pub fn node(&self, id: &str) -> Option<&ArchitectureNode> {
        self.nodes.iter().find(|node| node.id == id)
    }
}

/// Shared declaration used by execution traversal and catalog generation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitObservation {
    /// Input boundary of an execution unit.
    Input,
    /// Output boundary of an execution unit.
    Output,
}

impl UnitObservation {
    /// Formats the exact selector used by instrumentation and discovery.
    pub fn path(self, unit: &str) -> String {
        format!(
            "{unit}.{}",
            match self {
                Self::Input => "input",
                Self::Output => "output",
            }
        )
    }
}

/// Fields of the runtime's normalized routing event, shared with host collectors.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingObservationField {
    /// Selected expert identifiers.
    SelectedExperts,
    /// Selected scores before top-k renormalization.
    SelectedScores,
    /// Final coefficients applied to routed contributions.
    Coefficients,
    /// Combined routed-expert contribution.
    RoutedOutput,
    /// Rank-local routed contribution.
    LocalRoutedOutput,
    /// Routed contribution after collective reduction.
    ReducedRoutedOutput,
    /// Shared-expert contribution including any shared gate.
    SharedOutput,
    /// Combined routed and shared contribution.
    CombinedOutput,
}

impl RoutingObservationField {
    /// Formats the exact selector used by instrumentation and discovery.
    pub fn path(self, module: &str) -> String {
        let field = match self {
            Self::SelectedExperts => "selected_experts",
            Self::SelectedScores => "selected_scores",
            Self::Coefficients => "coefficients",
            Self::RoutedOutput => "routed_output",
            Self::LocalRoutedOutput => "local_routed_output",
            Self::ReducedRoutedOutput => "reduced_routed_output",
            Self::SharedOutput => "shared_output",
            Self::CombinedOutput => "combined_output",
        };
        format!("{module}.routing.{field}")
    }
}

/// Tensor element category before native-to-host conversion.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationDtype {
    /// Floating-point tensor; native precision is execution-dependent.
    Floating,
    /// Integer tensor; signedness is supplied by the captured host value.
    Integer,
    /// Boolean tensor.
    Boolean,
    /// Element category is not described.
    Unknown,
}

/// Portable value category produced by an advertised observation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationValueType {
    /// A complete tensor, materialized through `TensorObservation`.
    Tensor,
}

/// Capture timing relative to an intervention at the same path.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationPosition {
    /// Value passed to observe, before replacement at this exact point.
    BeforeIntervention,
    /// A read-only event after dispatch; no replacement at this point.
    ReadOnly,
    /// Value after replacement at this point.
    AfterIntervention,
}

/// Conditional mechanism or input required to emit a point.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationRequirement {
    /// Requires activation capture through instrumented execution.
    ActivationHooks,
    /// Requires normalized routing-event capture.
    RoutingEvents,
    /// Requires the corresponding media input.
    MediaInput,
    /// Requires the corresponding prediction group to execute.
    PredictionExecution,
}

/// One exact implemented tensor observation, not a hypothetical internal value.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ObservationPoint {
    /// Exact activation path accepted by ObservationSelector::Exact.
    pub path: String,
    /// Associated architecture node identity.
    pub node_id: String,
    /// Semantic meaning of the captured value.
    pub meaning: String,
    /// Portable value category.
    pub value_type: ObservationValueType,
    /// Semantic dtype before backend-specific host conversion.
    pub dtype: ObservationDtype,
    /// Tensor axes in storage order, when known.
    pub axes: Option<Vec<TensorAxis>>,
    /// Availability during a prefill operation.
    pub prefill: bool,
    /// Availability during a cached decode operation.
    pub decode: bool,
    /// Conditions in addition to phase availability.
    pub requirements: Vec<ObservationRequirement>,
    /// Position relative to an intervention at this exact point.
    pub position: ObservationPosition,
    /// Full tensor retained until host materialization; bytes depend on runtime shape/dtype.
    pub retained_bytes: Option<u64>,
    /// Host materialization bytes when reliably known; otherwise unknown.
    pub host_bytes: Option<u64>,
}

/// Versioned set of implemented architecture-declared capture points.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ObservationCatalog {
    /// Version of the discovery wire schema.
    pub schema_version: u32,
    /// Observation points in deterministic path order.
    pub points: Vec<ObservationPoint>,
    /// Explicit omissions at this scope.
    pub completeness: DescriptionCompleteness,
}

impl ObservationCatalog {
    /// Finds an implemented observation by its exact selectable path.
    pub fn get(&self, path: &str) -> Option<&ObservationPoint> {
        self.points.iter().find(|point| point.path == path)
    }
}

/// Side-effect-free host collector facts. Unknown is the conservative default.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ObservationMechanisms {
    /// The backend collects ordinary activation tensors.
    pub activation_tensors: bool,
    /// The backend collects normalized routing-event tensors.
    pub routing_tensors: bool,
    /// Floating observations are converted to portable F32 host values.
    pub floating_to_f32: bool,
}

/// Phase-specific support under one selected execution configuration.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", content = "reason", rename_all = "snake_case")]
pub enum ObservationSupportStatus {
    /// The selected execution emits this point in this phase.
    Supported,
    /// Supported if the stated input or execution condition holds.
    Conditional(String),
    /// The selected execution cannot emit this observation.
    Unsupported(String),
    /// Available facts do not prove capture support.
    Unverified(String),
}

/// Support for one architecture-declared path under an exact selected execution.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ObservationSupport {
    /// Exact activation path accepted by ObservationSelector::Exact.
    pub path: String,
    /// Availability during a prefill operation.
    pub prefill: ObservationSupportStatus,
    /// Availability during a cached decode operation.
    pub decode: ObservationSupportStatus,
    /// Host conversion fact; integer signedness is reported by the captured value.
    pub floating_to_f32: bool,
}

/// Versioned per-path support facts, separate from the logical architecture.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct ObservationSupportReport {
    /// Version of the discovery wire schema.
    pub schema_version: u32,
    /// Observation points in deterministic path order.
    pub points: Vec<ObservationSupport>,
    /// Bounded transformation mechanisms and their execution/storage conditions.
    #[serde(default)]
    pub capture: crate::capture::CaptureCapabilities,
}
