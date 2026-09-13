//! Architecture-declared scalar component identities and parameter relationships.
//! Declarations never imply that a selected execution supplies observations or edits.

use serde::{Deserialize, Serialize};

mod coordinates;
pub use coordinates::{ComponentCoordinateError, ComponentCoordinateMap};
mod routed;
mod routed_read;
mod streams;
mod transforms;
mod write;
pub use routed::{
    RoutedComponentCoordinateMap, RoutedComponentError, RoutedComponentGroup, RoutedComponentId,
    RoutedComponentParameter, RoutedComponentParameterName, RoutedComponentRead,
    RoutedComponentSelection,
};
pub use routed_read::ComponentRoutedRead;
pub use streams::{
    ComponentStreamBase, ComponentStreamCoefficients, ComponentStreamCycle, ComponentStreamHead,
    ComponentStreamResidual,
};
pub use transforms::{
    ComponentTensorTransform, ComponentTensorTransformEquation, ComponentTransformParameter,
};
pub use write::{
    ComponentGroupedWriteProjection, ComponentParameterSelection, ComponentWriteColumn,
    ComponentWriteError,
};

/// Version of the portable component topology.
pub const COMPONENT_SCHEMA_VERSION: u32 = 13;

/// Architecture readout equation. Decomposition is of the affine score before
/// `output_transform`; a nonlinear transform cannot be distributed over components.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentReadout {
    /// Actual input embeddings entering the decoder (with `.effective` companion).
    /// Conditional execution includes assembled media features; the observation's
    /// architecture node identifies that assembly. These captured values are the
    /// primary residual base for score reconstruction.
    pub embedding: String,
    /// Parameter used by the token lookup branch of the input embeddings.
    pub embedding_weight: String,
    /// Scale applied before the embedding observation.
    pub embedding_scale: ComponentScalar,
    /// Optional normalization of the assembled embedding tensor after scaling
    /// and before its observation. The captured tensor is the residual base;
    /// it must not be replaced by unnormalized token-lookup rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_normalization: Option<ComponentNormalization>,
    /// Optional normalization of token-lookup rows before text/media assembly.
    /// Media features are not normalized by this token-only equation. The
    /// captured `embedding` remains the authoritative assembled residual base.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_embedding_normalization: Option<ComponentNormalization>,
    /// Whether output and embedding parameters share the same source.
    pub tied_embeddings: bool,
    /// The primary head equation. Flattened serialization preserves the existing
    /// primary readout fields; field access also remains available through Deref.
    #[serde(flatten)]
    pub equation: ComponentReadoutEquation,
}

impl std::ops::Deref for ComponentReadout {
    type Target = ComponentReadoutEquation;

    fn deref(&self) -> &Self::Target {
        &self.equation
    }
}

impl std::ops::DerefMut for ComponentReadout {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.equation
    }
}

/// Score equation shared by primary and separately invoked prediction heads.
/// The enclosing readout declares the residual base and component membership.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentReadoutEquation {
    /// Optional multi-stream residual equation. Its measured mixing coefficients
    /// replace a plain sum of scalar-scaled writes; they are not derivatives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_residual: Option<ComponentStreamResidual>,
    /// Residual immediately before final normalization (with `.effective` companion).
    pub residual: String,
    /// Actual normalized residual consumed by the output head.
    pub normalized: String,
    /// Read-only actual output-head multiplication input after any selected input
    /// transform. Its difference from `normalized` is a separate readout term.
    #[serde(default)]
    pub projection_input: Option<String>,
    /// Affine vocabulary scores before optional nonlinear output transformation.
    pub linear_scores: String,
    /// Final scores before sampling policy.
    pub logits: String,
    /// Final normalization equation; measured factors depend on the current trial.
    pub normalization: ComponentNormalization,
    /// Canonical effective output matrix `[vocabulary, residual_width]`.
    pub weight: String,
    /// Optional additive vocabulary bias, separate from component contributions.
    pub bias: Option<String>,
    /// Dynamic vocabulary-space writes added after the primary head's effective
    /// affine scores and before `output_transform`. These are whole score terms,
    /// not residual writes or static vocabulary biases.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub score_writes: Vec<ComponentScoreWrite>,
    /// Transformation after the decomposable affine scores.
    pub output_transform: ComponentOutputTransform,
    /// Ordered normalizations of entire block residuals, in addition to sublayer norms.
    pub block_normalizations: Vec<ComponentResidualNormalization>,
    /// Ordered transformations of the entire running residual, after the writes
    /// and any `block_normalizations` at the same logical layer. Apply these in
    /// listed order before processing later layers. They affect every preceding
    /// contribution, including the residual base, not just this layer's writes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub block_transforms: Vec<ComponentResidualTransform>,
    /// Residual writes retained as whole terms because no scalar component group
    /// describes them completely. A component group at or below this write's
    /// node (for example, a shared expert in a sparse sum) is a constituent, not
    /// an additional residual term. Include the whole write once, or replace it
    /// with a complete decomposition of its declared child branches.
    #[serde(default)]
    pub other_writes: Vec<ComponentResidualWrite>,
}

/// Components belonging to a separately invoked execution group and score head.
/// These groups are excluded from the primary target component lists; combining
/// different scopes is not an additive decomposition of either head.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentExecutionScope {
    /// Stable identity of this component/readout scope.
    pub id: String,
    /// Root architecture node containing this invocation.
    pub node_id: String,
    /// Ordered physical execution groups participating in this score equation.
    /// A fused proposal may evolve its residual through several such groups.
    pub execution_groups: Vec<String>,
    /// Pinned parameter roles consumed by this invocation. Their storage is
    /// distinct from the ordered physical decoder groups above.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub static_parameter_roles: Vec<String>,
    /// Invocation semantics and its position in prediction execution.
    pub kind: ComponentExecutionScopeKind,
    /// Dense or shared component groups belonging only to this score equation.
    pub components: Vec<ComponentGroup>,
    /// Routed component groups belonging only to this score equation.
    pub routed_components: Vec<RoutedComponentGroup>,
    /// Actual residual base before this scope's decoder writes.
    pub residual_base: ComponentResidualBase,
    /// This scope's head, independent of the primary target head.
    pub readout: ComponentReadoutEquation,
}

/// Why an auxiliary component scope executes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentExecutionScopeKind {
    /// One zero-based sequential prediction depth; its scores are tentative
    /// until the ordinary target verification policy commits output.
    Prediction {
        /// Zero-based sequential prediction depth.
        depth: usize,
    },
    /// One fused proposal score equation. The number of proposal rows does not
    /// equal the number of physical groups contributing to this equation.
    FusedPrediction,
}

/// Base of a separately scoped residual decomposition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentResidualBase {
    /// An actual input, such as token embeddings, expanded without a projection.
    Source {
        /// Semantic source and effective parameter identity where applicable.
        source: ComponentFusionSource,
        /// Source actually consumed before the declared expansion.
        input: String,
        /// Exact geometry change, without modifying source values.
        expansion: ComponentFusionExpansion,
        /// Expanded input before intervention.
        output: String,
        /// Expanded input consumed by this scope's decoder.
        effective_output: String,
    },
    /// Independently normalized and projected inputs, expanded as declared and
    /// added elementwise. Each projection retains its own effective parameters
    /// and multiplication input; there is no implied concatenated matrix.
    ProjectedSum {
        /// Ordered terms of the sum.
        inputs: Vec<ComponentProjectedFusionInput>,
        /// Sum before intervention.
        output: String,
        /// Sum actually consumed by the decoder.
        effective_output: String,
    },
    /// A linear projection of concatenated, independently normalized inputs.
    /// The output itself is a whole residual term. Its input normalizations must
    /// not be distributed over preceding components as if they were affine.
    LinearFusion {
        /// Ordered inputs and their exact columns in the concatenation.
        inputs: Vec<ComponentFusionInput>,
        /// Canonical effective fusion matrix `[residual_width, input_width]`.
        weight: String,
        /// Canonical identity shared by aliases or repeated invocations.
        shared_weight: String,
        /// Owning architecture parameter group.
        parameter_group: String,
        /// Optional fusion projection bias.
        bias: Option<String>,
        /// Actual multiplication input after any selected input transformation.
        projection_input: String,
        /// Fusion output before intervention.
        output: String,
        /// Fusion output actually consumed by this scope's decoder.
        effective_output: String,
    },
}

/// An input-dependent vocabulary-space term in a score equation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentScoreWrite {
    /// Node that executes this projection, independent of shared parameters.
    pub node_id: String,
    /// Input origin before any intervention at the input seam.
    pub source: ComponentFusionSource,
    /// Effective input consumed by the projection's selected input transform.
    pub input: String,
    /// Actual multiplication input after input rounding or other transforms.
    pub projection_input: String,
    /// Canonical effective matrix `[vocabulary, input_width]`.
    pub weight: String,
    /// Physical parameter identity shared by aliases or repeated invocations.
    pub shared_weight: String,
    /// Architecture parameter group owning the projection.
    pub parameter_group: String,
    /// Optional additive projection bias.
    pub bias: Option<String>,
    /// Projected score term before intervention or broadcasting.
    pub output: String,
    /// Actual score term consumed by broadcasting and addition.
    pub effective_output: String,
    /// Existing singleton axes broadcast to the corresponding final-score axes.
    /// All other axes must match exactly; no reduction or projection is implied.
    pub broadcast_axes: Vec<String>,
}

/// One independently projected term of a residual fusion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentProjectedFusionInput {
    /// Actual source value before this branch's normalization (read-only).
    pub input: String,
    /// Origin before normalization.
    pub source: ComponentFusionSource,
    /// Normalization applied over each input's final axis.
    pub normalization: ComponentNormalization,
    /// Effective normalized input before the selected multiplication transform.
    pub normalized: String,
    /// Canonical effective projection matrix `[output_width, input_width]`.
    pub weight: String,
    /// Physical parameter identity shared by aliases or repeated invocations.
    pub shared_weight: String,
    /// Owning architecture parameter group.
    pub parameter_group: String,
    /// Optional additive projection bias.
    pub bias: Option<String>,
    /// Actual multiplication input, including any selected input rounding.
    pub projection_input: String,
    /// Projection output before intervention or expansion.
    pub output: String,
    /// Projection output consumed by the expansion and sum.
    pub effective_output: String,
    /// Geometry applied after the projection, without changing its values.
    pub expansion: ComponentFusionExpansion,
}

/// Exact expansion of a projected fusion term before elementwise addition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentFusionExpansion {
    /// Projection already has the sum's geometry.
    Identity,
    /// Insert a new axis at this storage position and repeat along it. All other
    /// axes retain their original order and extent.
    BroadcastAxis {
        /// Zero-based insertion position, at most the original rank.
        axis: usize,
        /// Semantic name of the inserted axis.
        name: String,
        /// Positive extent of the inserted axis.
        extent: usize,
    },
}

/// One normalized input to a declared residual fusion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentFusionInput {
    /// Origin of the input before normalization.
    pub source: ComponentFusionSource,
    /// Exact normalization applied independently to this input.
    pub normalization: ComponentNormalization,
    /// Effective normalized input consumed by the fusion.
    pub output: String,
    /// Contiguous columns occupied by this input in the concatenation.
    pub columns: std::ops::Range<usize>,
}

/// Semantic input origin without implying a fused base is an embedding lookup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentFusionSource {
    /// Embeddings of this invocation's token IDs, before normalization.
    TokenEmbedding {
        /// Canonical effective embedding parameter.
        weight: String,
        /// Shared physical parameter identity.
        shared_weight: String,
        /// Owning architecture parameter group.
        parameter_group: String,
        /// Scale applied before the embedding's own normalization.
        scale: ComponentScalar,
        /// Optional source-embedding normalization, before the enclosing fusion
        /// input's normalization. These are separate learned operations.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        normalization: Option<ComponentNormalization>,
    },
    /// A previously computed tensor supplied to this invocation.
    Observation {
        /// Exact observation of that input, not a reconstructed diagnostic.
        path: String,
    },
}

/// An architecture-declared whole residual contribution, such as a convolution
/// or routed-expert sum. It is not a fabricated scalar component decomposition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentResidualWrite {
    /// Actual operator input, when declared, before this whole residual write.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    /// Logical invocation ordinal.
    pub layer_index: usize,
    /// Owning architecture node.
    pub node_id: String,
    /// Original operator output immediately before residual addition.
    pub output: String,
    /// Effective operator output consumed after interventions.
    pub effective_output: String,
    /// Scale applied to the observed output at residual addition.
    pub residual_scale: ComponentScalar,
}

/// A normalization of the running residual changes all preceding contributions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentResidualNormalization {
    /// Logical invocation ordinal.
    pub layer_index: usize,
    /// Pre-normalization residual observation.
    pub input: String,
    /// Normalized block output observation.
    pub output: String,
    /// Exact architecture normalization convention.
    pub normalization: ComponentNormalization,
}

/// One whole-residual transformation in the ordered readout equation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentResidualTransform {
    /// Logical layer after whose writes this transformation executes.
    pub layer_index: usize,
    /// Exact identity in `ArchitectureDescriptor::component_transforms`. That
    /// declaration supplies consumed input, original/effective output and the
    /// loaded parameter identities; this reference is not an additive write.
    pub transform_id: String,
}

/// Vocabulary-score transformation, applied after the affine readout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentOutputTransform {
    /// No further transform.
    Identity,
    /// `cap * tanh(score / cap)`; additive contributions describe its input.
    Softcap {
        /// Positive architecture cap.
        cap: ComponentScalar,
    },
    /// `cap * tanh(score * scale / cap)`. Contributions describe the unscaled
    /// affine score. Scaling is applied to that complete score after the actual
    /// output projection, including its bias and selected input transformation.
    ScaledSoftcap {
        /// Architecture output multiplier.
        scale: ComponentScalar,
        /// Positive architecture cap.
        cap: ComponentScalar,
    },
}

/// A scalar component is addressed without enumerating every unit in a model.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ComponentId {
    /// Stable component-group identity, joined to an architecture invocation.
    pub group: String,
    /// Zero-based scalar within that group's `component` axis.
    pub index: usize,
}

/// Exact binary32 architecture scalar, preserving equality and wire round trips.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentScalar(u32);
impl ComponentScalar {
    /// Records the architecture's actual binary32 constant.
    pub fn new(value: f32) -> Self {
        Self(value.to_bits())
    }
    /// Returns the constant without a decimal serialization round trip.
    pub fn value(self) -> f32 {
        f32::from_bits(self.0)
    }
}

/// Normalization equation, including the parameters needed for exact centering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentNormalization {
    /// Identity, RMS, L2 or centered LayerNorm before optional gain and bias.
    pub kind: ComponentNormalizationKind,
    /// Epsilon added inside the square root.
    pub epsilon: ComponentScalar,
    /// Canonical effective gain parameter, if learned.
    pub gain: Option<String>,
    /// Constant added to the learned gain before multiplying normalized input.
    pub gain_offset: ComponentScalar,
    /// Canonical additive normalization bias, if present.
    pub bias: Option<String>,
    /// Number of independent contiguous normalization groups.
    pub groups: usize,
}

/// Normalization of a measured activation; factors must be recomputed per trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentNormalizationKind {
    /// No data-dependent normalization; optional gain and bias remain affine.
    /// With no learned gain or bias, the input passes through unchanged.
    Identity,
    /// `x / sqrt(mean(x*x) + epsilon)`.
    Rms,
    /// `(x - mean(x)) / sqrt(mean((x-mean(x))^2) + epsilon)`.
    Layer,
    /// `x / sqrt(sum(x*x) + epsilon)`, with additive epsilon, not a clamp.
    L2,
}

/// Activation applied before the output projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentActivation {
    /// An ordinary affine read followed by one activation.
    Unary {
        /// Activation equation.
        activation: ComponentNonlinearity,
    },
    /// Activated gate multiplied by a separately projected value.
    Gated {
        /// Gate activation.
        activation: ComponentNonlinearity,
        /// Optional upper clamp on the gate read before activation.
        gate_upper_bound: Option<ComponentScalar>,
        /// Optional symmetric clamp on the value read.
        value_absolute_bound: Option<ComponentScalar>,
        /// Offset added to the value read after clamping.
        value_offset: ComponentScalar,
    },
    /// Value aggregation over positions; optional gating follows aggregation.
    Attention {
        /// Query head count.
        query_heads: usize,
        /// Key/value head count.
        key_value_heads: usize,
        /// Channels per query head.
        head_width: usize,
        /// Activation on a separate output-gate projection.
        output_gate: Option<ComponentNonlinearity>,
    },
    /// Per-head gated delta recurrence, followed by head normalization and an
    /// output gate. This is stateful linear attention, without a softmax over
    /// positions. Query/key projection transforms and normalization are declared
    /// by their reads before the scales below are applied.
    ///
    /// For a key-by-value state matrix, `D = diag(exp(log_decay)) * S`,
    /// `S = D + sigmoid(update) * k * (v - k^T * D)^T`, and `y = q^T * S`.
    /// `log_decay = -exp(decay_rate) * softplus(decay + decay_bias)`.
    /// Decay and update affine dependencies use the corresponding read roles.
    GatedDeltaAttention {
        /// Projected query/key heads before contiguous repetition to value heads.
        key_heads: usize,
        /// Independent recurrent state and output heads.
        value_heads: usize,
        /// Query/key channels per head (state rows).
        key_head_width: usize,
        /// Value/output channels per head (state columns).
        value_head_width: usize,
        /// Decay broadcast within each state matrix.
        decay_layout: ComponentDeltaDecay,
        /// Scale following the declared query head normalization.
        query_scale: ComponentScalar,
        /// Scale following the declared key head normalization.
        key_scale: ComponentScalar,
        /// Learned log transition rate, one scalar per head.
        decay_rate: ComponentTransformParameter,
        /// Learned decay offset, with the declared per-head or per-key layout.
        decay_bias: ComponentTransformParameter,
        /// Normalization of `y`, before its elementwise output gate. This is
        /// distinct from normalization after a group's summed affine write.
        channel_normalization: ComponentHeadNormalization,
        /// Activation on the separate `OutputGate` read, multiplied into the
        /// normalized recurrent output before the component boundary.
        output_gate: ComponentNonlinearity,
    },
}

/// Broadcast geometry of a gated-delta state's decay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentDeltaDecay {
    /// One decay scalar per value head, multiplying the entire state matrix.
    Head,
    /// One decay value per key channel in each value head, multiplying state rows.
    KeyChannel,
}

/// Exact family-declared activation convention.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentNonlinearity {
    /// `x * sigmoid(multiplier * x)`.
    Silu {
        /// Multiplier inside sigmoid.
        multiplier: ComponentScalar,
    },
    /// `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`.
    GeluApproximate,
    /// Gaussian-CDF GELU.
    Gelu,
    /// `max(x, 0)`.
    Relu,
    /// `max(x, 0)^2`.
    ReluSquared,
    /// Logistic sigmoid.
    Sigmoid,
    /// `softplus(beta*x)/beta`.
    Softplus {
        /// Positive softplus scale.
        beta: ComponentScalar,
    },
}

fn component_unit_scale() -> ComponentScalar {
    ComponentScalar::new(1.0)
}

/// Normalization of projected attention heads before positional transformation
/// and score construction. This is distinct from sublayer input normalization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentHeadNormalization {
    /// Architecture constant multiplying each normalized channel. A missing
    /// serialized field means one; this is separate from any learned gain.
    #[serde(default = "component_unit_scale")]
    pub output_scale: ComponentScalar,
    /// Number of projected heads, before grouped-query repetition.
    pub heads: usize,
    /// Features normalized together within each head.
    pub head_width: usize,
    /// Whether the gain contains a separate vector for every head. Otherwise one
    /// head-width gain vector is reused for all heads.
    pub independent_gains: bool,
    /// Exact equation and parameter identity. Independent gains use the flattened
    /// head vector with `groups = heads`; shared gains apply separately to each head.
    pub normalization: ComponentNormalization,
}

/// Role of an effective affine read parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentReadRole {
    /// Non-gated FFN read.
    Input,
    /// FFN activation/gate read.
    Gate,
    /// FFN multiplicative value or attention value read.
    Value,
    /// Attention query read.
    Query,
    /// Attention key read.
    Key,
    /// Gate controlling a component contribution. The enclosing activation or
    /// output-gate declaration identifies its position in the equation.
    OutputGate,
    /// Key-channel decay read in a declared recurrent aggregation equation.
    Decay,
    /// Headwise state-update read in a declared recurrent aggregation equation.
    Update,
}

/// Maps a scalar component index into the logical effective projection rows.
/// Query/key dependencies span a head; value and FFN dependencies select one row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentRowMapping {
    /// `offset + component`; also describes a segment of a fused projection.
    Direct {
        /// Start row in the effective projection.
        offset: usize,
    },
    /// Scalar rows arranged in equally spaced blocks, such as output-gate rows
    /// interleaved with query rows inside each attention head.
    Blocked {
        /// First selected row in the effective projection.
        offset: usize,
        /// Number of consecutive selected rows in each block.
        block_width: usize,
        /// Distance between the starts of adjacent blocks, at least `block_width`.
        block_stride: usize,
    },
    /// `offset + (head / queries_per_kv)*head_width + channel`.
    GroupedQuery {
        /// Start row of the value segment.
        offset: usize,
        /// Channels per query head.
        head_width: usize,
        /// Number of query heads sharing one KV head.
        queries_per_kv: usize,
    },
    /// All rows in the query/key head used to construct a component's attention
    /// weights. The component (value) head width may differ from the read width.
    HeadRows {
        /// Start row of this segment in an effective, possibly fused projection.
        offset: usize,
        /// Output channels per component head.
        component_head_width: usize,
        /// Projection rows per query/key head.
        read_head_width: usize,
        /// Number of component heads sharing a read head (one for queries).
        component_heads_per_read_head: usize,
        /// Distance between head starts for an interleaved projection. Absent means
        /// `read_head_width`, preserving ordinary contiguous head geometry.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        read_head_stride: Option<usize>,
    },
}
impl ComponentRowMapping {
    /// Computes a scalar read row. Multirow head dependencies return `None`;
    /// use `row_range` for those. Malformed and overflowed mappings also fail.
    pub fn row(&self, component: usize) -> Option<usize> {
        match *self {
            Self::Direct { offset } => offset.checked_add(component),
            Self::Blocked {
                offset,
                block_width,
                block_stride,
            } => {
                if block_width == 0 || block_stride < block_width {
                    return None;
                }
                offset
                    .checked_add((component / block_width).checked_mul(block_stride)?)?
                    .checked_add(component % block_width)
            }
            Self::GroupedQuery {
                offset,
                head_width,
                queries_per_kv,
            } => {
                if head_width == 0 || queries_per_kv == 0 {
                    return None;
                }
                let head = component / head_width / queries_per_kv;
                offset
                    .checked_add(head.checked_mul(head_width)?)?
                    .checked_add(component % head_width)
            }
            Self::HeadRows {
                read_head_width: 1, ..
            } => self.row_range(component).map(|rows| rows.start),
            Self::HeadRows { .. } => None,
        }
    }

    /// Computes the complete contiguous read dependency using checked arithmetic.
    /// The caller must first validate the component against its group's count.
    pub fn row_range(&self, component: usize) -> Option<std::ops::Range<usize>> {
        let (start, width) = match *self {
            Self::HeadRows {
                offset,
                component_head_width,
                read_head_width,
                component_heads_per_read_head,
                read_head_stride,
            } => {
                let stride = read_head_stride.unwrap_or(read_head_width);
                if component_head_width == 0
                    || read_head_width == 0
                    || component_heads_per_read_head == 0
                    || stride < read_head_width
                {
                    return None;
                }
                let head = component / component_head_width / component_heads_per_read_head;
                (
                    offset.checked_add(head.checked_mul(stride)?)?,
                    read_head_width,
                )
            }
            _ => (self.row(component)?, 1),
        };
        Some(start..start.checked_add(width)?)
    }
}

/// Read rows refer to effective affine geometry, never raw packed tensor geometry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentRead {
    /// Override for a read whose input and retained history belong to another
    /// component invocation. Absent means this group's normalized input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ComponentReadSource>,
    /// Semantic use of this read.
    pub role: ComponentReadRole,
    /// Canonical architecture parameter name.
    pub weight: String,
    /// Architecture parameter group containing the weight.
    pub parameter_group: String,
    /// Original shared parameter name; identical for shared logical invocations.
    pub shared_weight: String,
    /// Canonical optional bias vector, using the same row mapping.
    pub bias: Option<String>,
    /// Logical effective row selection.
    pub rows: ComponentRowMapping,
    /// Actual affine projection output, when captured. Follow architecture
    /// `component_transforms` from this path for any intervening causal operations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_output: Option<String>,
    /// Optional head normalization before attention scoring, after any declared
    /// intervening tensor transforms. A read row alone is not the complete
    /// normalized query/key equation.
    #[serde(default)]
    pub head_normalization: Option<ComponentHeadNormalization>,
    /// Ordered projections between the source's normalized input and this read.
    /// Empty means this matrix reads the normalized sublayer input directly.
    /// A normalization in this chain prevents replacing it with a fixed product
    /// of weight matrices; its measured factors must be recomputed for each trial.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_projections: Vec<ComponentInputProjection>,
}

/// Origin of an affine read whose state is shared across logical invocations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentReadSource {
    /// The read executes at this publishing attention group and its normalized
    /// input, then enters that publisher's retained attention state. A consumer
    /// does not project its own input using these weights. Capturing the current
    /// publisher invocation does not recover historical cached positions.
    PublishedAttentionState {
        /// Stable component-group identity of the actual publishing invocation.
        component_group: String,
    },
}

/// A stage producing the input of a component read, such as a normalized latent
/// bottleneck. Parameters use the same effective geometry as bounded queries.
/// Stages execute in order, starting from the component group's normalized input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentInputProjection {
    /// Canonical effective matrix; selected rows are taken before normalization.
    pub weight: String,
    /// Architecture parameter group containing this projection.
    pub parameter_group: String,
    /// Shared physical identity across logical invocations.
    pub shared_weight: String,
    /// Optional affine bias, selected with the same row range.
    pub bias: Option<String>,
    /// Nonempty half-open range of effective output rows.
    pub rows: std::ops::Range<usize>,
    /// Normalization after the selected affine output, when present.
    pub normalization: Option<ComponentNormalization>,
    /// Actual effective stage output for the current input positions. A cache may
    /// retain this value for later reads; this is not a capture of past positions.
    pub output: String,
}

/// Placement of the group's write/output hooks under tensor parallel execution.
/// This declares equations, not verified hook or collector support.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentWritePartition {
    /// Each executing rank observes the already reduced complete write.
    #[default]
    Complete,
    /// Each TP rank supplies its additive full-hidden-width term before the
    /// ordinary reduction, possibly fused with another branch. True replicas
    /// belong to separate sums. There is no intervening output normalization;
    /// each sum contains any affine offset exactly once.
    TensorParallelSum,
}

/// A measured scalar gate applied after a component group's affine write and
/// optional output normalization, immediately before its `output` observation.
/// Each batch/sequence row has one scalar, broadcast across the hidden axis.
/// This factor must be remeasured for every trial; it is not a static residual
/// scale and cannot be moved across the projection in floating-point execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentOutputGate {
    /// Effective input to the gate's affine read, before its input transform.
    pub input: String,
    /// Actual multiplication input after the selected parameter input transform.
    pub projection_input: String,
    /// Exact affine parameter and component-to-read-row relationship. The role
    /// is `OutputGate`; every component must select the same single output row.
    pub read: ComponentRead,
    /// Activation applied to the affine result to produce the scalar gate.
    pub activation: ComponentNonlinearity,
    /// Gate after activation and before intervention, shaped `[batch, sequence, 1]`.
    pub output: String,
    /// Effective gate actually multiplied into the normalized write.
    pub effective_output: String,
}

/// A compact group of scalar FFN units or attention output channels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentGroup {
    /// Stable group ID. A `ComponentId` adds a zero-based scalar index.
    pub id: String,
    /// Owning logical architecture node (invocations remain distinct).
    pub node_id: String,
    /// Logical execution ordinal.
    pub layer_index: usize,
    /// Number of scalar components.
    pub count: usize,
    /// Pre-intervention component observation and intervention target.
    pub activation: String,
    /// Read-only post-intervention values supplied to the selected projection.
    /// Its input arithmetic may transform these; `write_input` reports that result.
    pub effective_activation: String,
    /// Read-only multiplication input after any selected projection input transform.
    /// Use this evidence with effective write columns to reconstruct the write.
    #[serde(default)]
    pub write_input: Option<String>,
    /// Logical complete affine write after projection, before output normalization.
    /// `write_partition` describes whether a TP hook supplies the complete tensor
    /// or an additive term. Has a `.effective` intervention companion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_output: Option<String>,
    /// Complete contribution after output normalization and before addition to
    /// the residual or a larger operator sum. Has an `.effective` companion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Architecture equation at the write/output hooks when tensor parallelism
    /// is selected. Capture evidence separately identifies its assembly arithmetic.
    #[serde(default)]
    pub write_partition: ComponentWritePartition,
    /// Normalized sublayer input observation.
    pub input: String,
    /// Exact read relationship, including both branches of a gated FFN.
    pub reads: Vec<ComponentRead>,
    /// Input-dependent mixtures of activated expert reads, such as routed
    /// attention values. These are distinct from ordinary affine read rows.
    #[serde(default)]
    pub routed_reads: Vec<ComponentRoutedRead>,
    /// Effective final output projection. Its input width is `count` for a
    /// direct write, or `groups * rank` when `write_input_projection` is present.
    pub write_weight: String,
    /// Optional grouped linear stage between the scalar components and the final
    /// write matrix. This describes actual parameter factors, not a materialized
    /// dense product. Absent in component schemas before version five.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub write_input_projection: Option<ComponentGroupedWriteProjection>,
    /// Parameter group containing the output projection.
    pub write_parameter_group: String,
    /// Original shared output weight; logical invocations keep separate component IDs.
    pub shared_write_weight: String,
    /// Output bias is a separate additive term, not one bias per component.
    pub write_bias: Option<String>,
    /// Scalar activation/aggregation equation.
    pub activation_equation: ComponentActivation,
    /// Normalization before the reads.
    pub input_normalization: ComponentNormalization,
    /// Optional normalization of the summed affine write before residual addition.
    pub output_normalization: Option<ComponentNormalization>,
    /// Optional input-dependent scalar gate after output normalization and
    /// before `output`. Individual contributions include its effective value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_gate: Option<ComponentOutputGate>,
    /// Scalar applied to the sublayer write before residual addition.
    pub residual_scale: ComponentScalar,
}
impl ComponentGroup {
    /// Validates a component identity against this compact group.
    pub fn contains(&self, id: &ComponentId) -> bool {
        id.group == self.id && id.index < self.count
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn head_scale_and_scaled_softcap_preserve_wire_equations() {
        let legacy = r#"{"heads":2,"head_width":4,"independent_gains":false,"normalization":{"kind":"rms","epsilon":0,"gain":null,"gain_offset":0,"bias":null,"groups":1}}"#;
        let mut value = serde_json::from_str::<super::ComponentHeadNormalization>(legacy).unwrap();
        assert_eq!(value.output_scale.value(), 1.0);
        value.output_scale = super::ComponentScalar::new(1.7);
        assert_eq!(
            value,
            serde_json::from_str(&serde_json::to_string(&value).unwrap()).unwrap()
        );
        let transform = super::ComponentOutputTransform::ScaledSoftcap {
            scale: super::ComponentScalar::new(1.9),
            cap: super::ComponentScalar::new(7.0),
        };
        assert_eq!(
            transform,
            serde_json::from_str(&serde_json::to_string(&transform).unwrap()).unwrap()
        );
    }

    use super::ComponentRowMapping;

    #[test]
    fn attention_head_reads_cover_all_rows_and_shared_heads() {
        let query = ComponentRowMapping::HeadRows {
            offset: 0,
            component_head_width: 3,
            read_head_width: 5,
            component_heads_per_read_head: 1,
            read_head_stride: None,
        };
        let key = ComponentRowMapping::HeadRows {
            offset: 20,
            component_head_width: 3,
            read_head_width: 5,
            component_heads_per_read_head: 2,
            read_head_stride: None,
        };
        let value = ComponentRowMapping::GroupedQuery {
            offset: 30,
            head_width: 3,
            queries_per_kv: 2,
        };
        for component in 0..12 {
            let query_head = component / 3;
            let kv_head = query_head / 2;
            assert_eq!(
                query.row_range(component),
                Some(query_head * 5..(query_head + 1) * 5)
            );
            assert_eq!(
                key.row_range(component),
                Some(20 + kv_head * 5..20 + (kv_head + 1) * 5)
            );
            assert_eq!(query.row(component), None);
            assert_eq!(key.row(component), None);
            let row = 30 + kv_head * 3 + component % 3;
            assert_eq!(value.row(component), Some(row));
            assert_eq!(value.row_range(component), Some(row..row + 1));
        }
        for mapping in [query, key, value] {
            let encoded = serde_json::to_string(&mapping).unwrap();
            assert_eq!(
                serde_json::from_str::<ComponentRowMapping>(&encoded).unwrap(),
                mapping
            );
        }
    }

    #[test]
    fn component_read_ranges_reject_malformed_geometry_and_overflow() {
        for (offset, component_width, read_width, sharing, component) in [
            (0, 0, 2, 1, 0),
            (0, 2, 0, 1, 0),
            (0, 2, 2, 0, 0),
            (usize::MAX, 2, 2, 1, 0),
            (0, 1, 2, 1, usize::MAX),
        ] {
            let mapping = ComponentRowMapping::HeadRows {
                offset,
                component_head_width: component_width,
                read_head_width: read_width,
                component_heads_per_read_head: sharing,
                read_head_stride: None,
            };
            assert_eq!(mapping.row_range(component), None);
        }
        assert_eq!(
            ComponentRowMapping::Direct { offset: usize::MAX }.row_range(0),
            None
        );
        for (head_width, queries_per_kv) in [(0, 1), (1, 0)] {
            assert_eq!(
                ComponentRowMapping::GroupedQuery {
                    offset: 0,
                    head_width,
                    queries_per_kv
                }
                .row_range(0),
                None
            );
        }
        let scalar_head = ComponentRowMapping::HeadRows {
            offset: 3,
            component_head_width: 2,
            read_head_width: 1,
            component_heads_per_read_head: 2,
            read_head_stride: None,
        };
        assert_eq!(scalar_head.row(4), Some(4));
        assert_eq!(scalar_head.row_range(4), Some(4..5));
    }
    #[test]
    fn interleaved_query_and_gate_rows_preserve_head_dependencies() {
        let query = ComponentRowMapping::HeadRows {
            offset: 0,
            component_head_width: 3,
            read_head_width: 3,
            component_heads_per_read_head: 1,
            read_head_stride: Some(6),
        };
        let gate = ComponentRowMapping::Blocked {
            offset: 3,
            block_width: 3,
            block_stride: 6,
        };
        // Independently laid out rows: [Q0(3), G0(3), Q1(3), G1(3)].
        let values = [11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43];
        for (component, expected_query, expected_gate) in [
            (0, [11, 12, 13], 21),
            (1, [11, 12, 13], 22),
            (2, [11, 12, 13], 23),
            (3, [31, 32, 33], 41),
            (4, [31, 32, 33], 42),
            (5, [31, 32, 33], 43),
        ] {
            assert_eq!(values[query.row_range(component).unwrap()], expected_query);
            assert_eq!(values[gate.row(component).unwrap()], expected_gate);
            let row = gate.row(component).unwrap();
            assert_eq!(gate.row_range(component), Some(row..row + 1));
            assert_eq!(query.row(component), None);
        }
        for mapping in [query, gate] {
            assert_eq!(
                serde_json::from_str::<ComponentRowMapping>(
                    &serde_json::to_string(&mapping).unwrap()
                )
                .unwrap(),
                mapping
            );
        }
        let legacy: ComponentRowMapping = serde_json::from_str(r#"{"kind":"head_rows","offset":0,"component_head_width":3,"read_head_width":3,"component_heads_per_read_head":1}"#).unwrap();
        assert_eq!(legacy.row_range(3), Some(3..6));
        for (offset, width, stride, index) in [
            (0, 0, 1, 0),
            (0, 3, 2, 0),
            (usize::MAX, 1, 1, 0),
            (0, 1, 2, usize::MAX),
        ] {
            assert_eq!(
                ComponentRowMapping::Blocked {
                    offset,
                    block_width: width,
                    block_stride: stride
                }
                .row_range(index),
                None
            );
            assert_eq!(
                ComponentRowMapping::HeadRows {
                    offset,
                    component_head_width: 1,
                    read_head_width: width,
                    component_heads_per_read_head: 1,
                    read_head_stride: Some(stride),
                }
                .row_range(index),
                None
            );
        }
    }
}
