//! Backend-neutral neural computation contracts.
//!
//! Architecture implementations use opaque backend tensors through this
//! interface. Tensor storage, physical layout, laziness, device placement, and
//! synchronization remain owned by the selected backend.

#![warn(missing_docs)]

extern crate self as eredu_nn;

use std::fmt::Debug;

pub use eredu_checkpoint::LinearFormat;
use eredu_checkpoint::WeightQuantization;

pub use eredu_nn_macros::Parameterized;

#[cfg(feature = "mlx")]
mod mlx;

/// Backend operation failure.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct Error {
    message: String,
}

impl Error {
    /// Creates a backend operation failure without exposing backend-native
    /// exception types through architecture code.
    pub fn backend(error: impl std::fmt::Display) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

/// One axis of a backend-neutral tensor view.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Index {
    /// Retains the complete axis.
    Full,
    /// Selects one element and removes the axis.
    At(i32),
    /// Retains the half-open interval `[start, end)`.
    Range(i32, i32),
}

/// Padding behavior for convolutional architecture components.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PadMode {
    /// Pads with zeroes.
    Constant,
    /// Repeats the nearest edge value.
    Edge,
}

/// Attention masking selected by architecture code.
#[derive(Debug, Clone, Copy)]
pub enum AttentionMask<'a, T> {
    /// No attention mask.
    None,
    /// Standard causal mask.
    Causal,
    /// Backend tensor added to attention logits.
    Tensor(&'a T),
}

/// Typed sparse/indexed attention request.
///
/// Architecture code owns how positions are selected, causal eligibility,
/// compression ratios, top-k, and sink policy. A backend may fuse gathering,
/// shared-softmax attention, and value reduction without exposing an
/// accelerator-specific indexing API.
#[derive(Debug, Clone, Copy)]
pub struct IndexedAttentionInput<'a, T> {
    /// Queries shaped `[batch, heads, query_tokens, key_dimensions]`.
    pub queries: &'a T,
    /// Bounded local keys shaped `[batch, local_tokens, key_dimensions]`.
    pub local_keys: &'a T,
    /// Bounded local values shaped `[batch, local_tokens, value_dimensions]`.
    pub local_values: &'a T,
    /// Compressed/indexable keys shaped `[batch, pooled_tokens, key_dimensions]`.
    pub pooled_keys: &'a T,
    /// Compressed/indexable values shaped `[batch, pooled_tokens, value_dimensions]`.
    pub pooled_values: &'a T,
    /// Selected pooled positions shaped `[batch, query_tokens, selected]`.
    pub selected_positions: &'a T,
    /// Query/key score multiplier.
    pub scale: f32,
    /// Optional mask broadcastable to local scores.
    pub local_mask: Option<&'a T>,
    /// Optional mask broadcastable to selected pooled scores.
    pub pooled_mask: Option<&'a T>,
    /// Optional learned per-head sink logits.
    pub sinks: Option<&'a T>,
}

/// Dense attention over bounded local keys plus complete pooled history.
#[derive(Debug, Clone, Copy)]
pub struct PooledAttentionInput<'a, T> {
    /// Queries shaped `[batch, heads, query_tokens, dimensions]`.
    pub queries: &'a T,
    /// Bounded local keys and values shaped `[batch, local_tokens, dimensions]`.
    pub local: &'a T,
    /// Complete pooled keys and values shaped `[batch, pooled_tokens, dimensions]`.
    pub pooled: &'a T,
    /// Query/key score multiplier.
    pub scale: f32,
    /// Optional mask broadcastable to local scores.
    pub local_mask: Option<&'a T>,
    /// Optional mask broadcastable to pooled scores.
    pub pooled_mask: Option<&'a T>,
    /// Optional learned per-head sink logits.
    pub sinks: Option<&'a T>,
}

/// Architecture-selected pooled-position scoring request.
///
/// Backends may fuse score construction, causal masking, and top-k selection
/// without exposing device-specific partition or gather operations.
#[derive(Debug, Clone, Copy)]
pub struct PooledPositionInput<'a, T> {
    /// Rotary queries shaped `[batch, heads, query_tokens, dimensions]`.
    pub queries: &'a T,
    /// Compressed index keys shaped `[batch, pooled_tokens, dimensions]`.
    pub pooled_keys: &'a T,
    /// Per-token head weights shaped `[batch, query_tokens, heads]`.
    pub head_weights: &'a T,
    /// Optional eligibility mask shaped `[query_tokens, pooled_tokens]` or a
    /// broadcast-compatible batch variant.
    pub mask: Option<&'a T>,
    /// Number of pooled positions selected per query token.
    pub top_k: i32,
    /// Score multiplier applied after nonnegative clamping.
    pub scale: f32,
    /// Head-weight multiplier applied before reducing the head axis.
    pub head_scale: f32,
}

impl<T: Tensor> IndexedAttentionInput<'_, T> {
    /// Validates semantic ranks and exact non-broadcast geometry without
    /// materializing any backend values.
    pub fn validate(&self) -> Result<(), Error> {
        let query = self.queries.shape();
        let local_keys = self.local_keys.shape();
        let local_values = self.local_values.shape();
        let pooled_keys = self.pooled_keys.shape();
        let pooled_values = self.pooled_values.shape();
        let selected = self.selected_positions.shape();
        if query.len() != 4
            || local_keys.len() != 3
            || local_values.len() != 3
            || pooled_keys.len() != 3
            || pooled_values.len() != 3
            || selected.len() != 3
            || query[0] != local_keys[0]
            || query[0] != local_values[0]
            || query[0] != pooled_keys[0]
            || query[0] != pooled_values[0]
            || query[0] != selected[0]
            || query[2] != selected[1]
            || query[3] != local_keys[2]
            || query[3] != pooled_keys[2]
            || local_keys[1] != local_values[1]
            || pooled_keys[1] != pooled_values[1]
            || local_values[2] != pooled_values[2]
            || selected[2] <= 0
            || pooled_keys[1] <= 0
        {
            return Err(Error::backend(format!(
                "invalid indexed-attention geometry: queries={query:?} local_keys={local_keys:?} local_values={local_values:?} pooled_keys={pooled_keys:?} pooled_values={pooled_values:?} selected={selected:?}"
            )));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(Error::backend(format!(
                "indexed-attention scale must be finite and positive, got {}",
                self.scale
            )));
        }
        if let Some(sinks) = self.sinks {
            if sinks.shape() != [query[1]] {
                return Err(Error::backend(format!(
                    "indexed-attention sinks require shape [{}], got {:?}",
                    query[1],
                    sinks.shape()
                )));
            }
        }
        Ok(())
    }
}

/// Stable authoritative identity of one logical model parameter.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ParameterId(String);

impl ParameterId {
    /// Creates a non-empty parameter identity.
    pub fn new(id: impl Into<String>) -> Result<Self, ParameterTopologyError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(ParameterTopologyError::EmptyId);
        }
        Ok(Self(id))
    }

    /// Returns the stable checkpoint-facing identity.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ParameterId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Complete logical declaration for one parameter slot.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterSpec {
    /// Stable authoritative identity.
    pub id: ParameterId,
    /// Whether optimizers may update this parameter by default.
    pub trainable: bool,
    /// Authoritative destination when this slot aliases a tied parameter.
    pub alias_of: Option<ParameterId>,
    /// Optional atomic encoding or sharding group.
    pub group: Option<String>,
}

impl ParameterSpec {
    /// Declares an ordinary trainable parameter.
    pub fn trainable(id: impl Into<String>) -> Result<Self, ParameterTopologyError> {
        Ok(Self {
            id: ParameterId::new(id)?,
            trainable: true,
            alias_of: None,
            group: None,
        })
    }
}

/// Parameter metadata observed during traversal.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterMetadata {
    /// Stable authoritative identity.
    pub id: ParameterId,
    /// Current trainable state.
    pub trainable: bool,
    /// Authoritative destination for a tied alias.
    pub alias_of: Option<ParameterId>,
    /// Optional atomic parameter group.
    pub group: Option<String>,
}

impl ParameterMetadata {
    /// Creates traversal metadata from a construction specification.
    pub fn from_spec(spec: &ParameterSpec, trainable: bool) -> Self {
        Self {
            id: spec.id.clone(),
            trainable,
            alias_of: spec.alias_of.clone(),
            group: spec.group.clone(),
        }
    }
}

/// Invalid stable parameter topology.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ParameterTopologyError {
    /// A parameter identity is empty.
    #[error("parameter identity must not be empty")]
    EmptyId,
    /// The same stable identity was visited more than once.
    #[error("parameter identity {0} is duplicated")]
    DuplicateId(ParameterId),
    /// A tied alias points to an identity that is not in the topology.
    #[error("parameter alias {alias} points to missing destination {destination}")]
    MissingAliasDestination {
        /// Identity of the aliasing slot.
        alias: ParameterId,
        /// Missing authoritative destination.
        destination: ParameterId,
    },
    /// A tied alias points to another alias instead of an authoritative slot.
    #[error("parameter alias {alias} points to non-authoritative alias {destination}")]
    AliasTargetsAlias {
        /// Identity of the aliasing slot.
        alias: ParameterId,
        /// Non-authoritative destination.
        destination: ParameterId,
    },
}

/// Immutable statically dispatched parameter visitor.
pub trait ParameterVisitor<'a, T: 'a> {
    /// Visits one authoritative parameter slot.
    fn visit(&mut self, metadata: ParameterMetadata, value: &'a T);
}

/// Mutable statically dispatched parameter visitor.
pub trait ParameterVisitorMut<'a, T: 'a> {
    /// Visits one authoritative mutable parameter slot.
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut T);
}

/// Backend-neutral parameter topology for a module or operator.
pub trait Parameterized<T: 'static> {
    /// Visits every parameter exactly once using stable identities.
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, T>;

    /// Mutably visits every parameter exactly once using stable identities.
    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, T>;

    /// Updates whether all parameters in this module are trainable.
    fn set_trainable(&mut self, trainable: bool);
}

/// Collects and validates the stable parameter topology exposed by a module.
pub fn validate_parameter_topology<T: 'static, M>(
    module: &M,
) -> Result<Vec<ParameterMetadata>, ParameterTopologyError>
where
    M: Parameterized<T>,
{
    struct Collector(Vec<ParameterMetadata>);
    impl<'a, T: 'a> ParameterVisitor<'a, T> for Collector {
        fn visit(&mut self, metadata: ParameterMetadata, _value: &'a T) {
            self.0.push(metadata);
        }
    }

    let mut collector = Collector(Vec::new());
    module.visit_parameters(&mut collector);
    let mut topology = std::collections::BTreeMap::new();
    for metadata in &collector.0 {
        if topology.insert(metadata.id.clone(), metadata).is_some() {
            return Err(ParameterTopologyError::DuplicateId(metadata.id.clone()));
        }
    }
    for metadata in &collector.0 {
        let Some(destination) = &metadata.alias_of else {
            continue;
        };
        let Some(target) = topology.get(destination) else {
            return Err(ParameterTopologyError::MissingAliasDestination {
                alias: metadata.id.clone(),
                destination: destination.clone(),
            });
        };
        if target.alias_of.is_some() {
            return Err(ParameterTopologyError::AliasTargetsAlias {
                alias: metadata.id.clone(),
                destination: destination.clone(),
            });
        }
    }
    Ok(collector.0)
}

/// Shape and checkpoint identity for an affine projection.
#[derive(Debug, Clone)]
pub struct LinearSpec {
    /// Input feature count.
    pub input: i32,
    /// Output feature count.
    pub output: i32,
    /// Stable weight slot.
    pub weight: ParameterSpec,
    /// Optional stable bias slot.
    pub bias: Option<ParameterSpec>,
    /// Complete physical checkpoint encoding selected for this parameter.
    pub format: LinearFormat,
}

/// Complete construction specification for a token embedding table.
#[derive(Debug, Clone)]
pub struct EmbeddingSpec {
    /// Vocabulary row count.
    pub vocabulary: i32,
    /// Embedding width.
    pub dimensions: i32,
    /// Stable embedding weight slot.
    pub weight: ParameterSpec,
    /// Physical checkpoint encoding selected for this parameter.
    pub quantization: Option<WeightQuantization>,
}

/// Complete construction specification for normalization.
#[derive(Debug, Clone)]
pub struct NormalizationSpec {
    /// Normalized feature count.
    pub dimensions: i32,
    /// Numerical stability epsilon.
    pub epsilon: f32,
    /// Stable scale parameter slot.
    pub weight: ParameterSpec,
}

/// One backend-neutral RoPE configuration value.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(untagged)]
pub enum RopeValue {
    /// Floating-point metadata.
    Float(f32),
    /// String metadata.
    String(String),
    /// Boolean metadata.
    Bool(bool),
}

/// Canonical metadata tag for piecewise wavelength-based rotary scaling.
pub const FREQUENCY_SCALED_ROPE_TYPE: &str = "llama3";

/// Complete backend-neutral rotary-position construction specification.
#[derive(Debug, Clone, Copy)]
pub struct RotarySpec<'a> {
    /// Rotated head dimensions.
    pub dimensions: i32,
    /// Base frequency.
    pub base: f32,
    /// Whether adjacent pairs are rotated instead of split halves.
    pub traditional: bool,
    /// Maximum configured position count.
    pub max_positions: i32,
    /// Optional normalized scaling metadata.
    pub scaling: Option<&'a std::collections::HashMap<String, RopeValue>>,
}

/// Backend-native affine projection used by shared architectures.
pub trait LinearOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Applies the projection without host materialization.
    fn forward(&mut self, input: &T, context: &T::Context) -> Result<T, Error>;
}

/// Backend-native token embedding used by shared architectures.
pub trait EmbeddingOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Looks up token embeddings.
    fn forward(&mut self, input: &T, context: &T::Context) -> Result<T, Error>;
    /// Projects hidden states through the transposed embedding table.
    fn as_linear(&mut self, input: &T, context: &T::Context) -> Result<T, Error>;
}

/// Backend-native normalization operator.
pub trait NormalizationOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Applies normalization.
    fn forward(&mut self, input: &T, context: &T::Context) -> Result<T, Error>;
}

/// Construction specification for a normalized low-rank projection.
#[derive(Debug, Clone)]
pub struct LowRankProjectionSpec {
    /// Optional input-to-rank projection. When omitted, the input is already
    /// represented in rank space.
    pub first: Option<LinearSpec>,
    /// Normalization applied in rank space.
    pub normalization: NormalizationSpec,
    /// Rank-to-output projection.
    pub second: LinearSpec,
}

impl LowRankProjectionSpec {
    /// Validates that both projections and normalization agree on rank width.
    pub fn validate(&self) -> Result<(), Error> {
        let rank = self.normalization.dimensions;
        if rank <= 0 {
            return Err(Error::backend(format!(
                "low-rank normalization dimensions must be positive, got {rank}"
            )));
        }
        if self.second.input != rank {
            return Err(Error::backend(format!(
                "low-rank second projection expects {} inputs but rank width is {rank}",
                self.second.input
            )));
        }
        if let Some(first) = &self.first {
            if first.output != rank {
                return Err(Error::backend(format!(
                    "low-rank first projection emits {} values but rank width is {rank}",
                    first.output
                )));
            }
        }
        Ok(())
    }
}

/// Reusable normalized low-rank projection with statically dispatched backend
/// operators.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct LowRankProjection<B: NeuralBackend> {
    /// Optional input-to-rank projection.
    pub first: Option<B::Linear>,
    /// Rank-space normalization.
    pub normalization: B::Normalization,
    /// Rank-to-output projection.
    pub second: B::Linear,
}

impl<B: NeuralBackend> LowRankProjection<B> {
    /// Builds an unloaded low-rank projection from architecture-owned
    /// parameter identities and physical formats.
    pub fn new(
        spec: LowRankProjectionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        Ok(Self {
            first: spec
                .first
                .map(|projection| B::linear(projection, context))
                .transpose()?,
            normalization: B::rms_norm(spec.normalization, context)?,
            second: B::linear(spec.second, context)?,
        })
    }

    /// Applies the optional first projection, rank normalization, and second
    /// projection without backend-value conversion.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let rank = match &mut self.first {
            Some(first) => first.forward(input, context)?,
            None => input.clone(),
        };
        let rank = self.normalization.forward(&rank, context)?;
        self.second.forward(&rank, context)
    }
}

/// Backend-native rotary-position operator.
pub trait RotaryOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Applies the architecture-selected rotary position source.
    fn forward(
        &mut self,
        input: &T,
        position: RotaryPosition<'_, T>,
        context: &T::Context,
    ) -> Result<T, Error>;

    /// Applies rotary positions only to the selected final-axis subspace and
    /// leaves every other feature unchanged.
    fn forward_subspace(
        &mut self,
        input: &T,
        subspace: RotarySubspace,
        position: RotaryPosition<'_, T>,
        context: &T::Context,
    ) -> Result<T, Error> {
        let width = *input
            .shape()
            .last()
            .ok_or_else(|| Error::backend("rotary input must have a feature axis"))?;
        let (start, dimensions) = subspace.resolve(width)?;
        if start == 0 && dimensions == width {
            return self.forward(input, position, context);
        }
        let end = start + dimensions;
        let mut indexes = vec![Index::Full; input.shape().len()];
        indexes[input.shape().len() - 1] = Index::Range(start, end);
        let selected = input.index(&indexes, context)?;
        let rotated = self.forward(&selected, position, context)?;
        let mut pieces = Vec::with_capacity(3);
        if start > 0 {
            indexes[input.shape().len() - 1] = Index::Range(0, start);
            pieces.push(input.index(&indexes, context)?);
        }
        pieces.push(rotated);
        if end < width {
            indexes[input.shape().len() - 1] = Index::Range(end, width);
            pieces.push(input.index(&indexes, context)?);
        }
        T::concatenate(&pieces, -1, context)
    }
}

/// Final-axis feature range selected for rotary position encoding.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RotarySubspace {
    /// Rotate the complete final feature axis.
    Full,
    /// Rotate one contiguous half-open final-axis range.
    Range {
        /// First rotated feature.
        start: i32,
        /// Number of rotated features.
        dimensions: i32,
    },
}

impl RotarySubspace {
    fn resolve(self, width: i32) -> Result<(i32, i32), Error> {
        let (start, dimensions) = match self {
            Self::Full => (0, width),
            Self::Range { start, dimensions } => (start, dimensions),
        };
        if width <= 0
            || start < 0
            || dimensions <= 0
            || dimensions % 2 != 0
            || start > width - dimensions
        {
            return Err(Error::backend(format!(
                "rotary subspace start={start} dimensions={dimensions} is invalid for width {width}"
            )));
        }
        Ok((start, dimensions))
    }
}

/// Position data supplied to a backend-native rotary operator.
#[derive(Debug)]
pub enum RotaryPosition<'a, T> {
    /// Ordinary contiguous sequence positions beginning at this offset.
    Offset(i32),
    /// Caller-provided cosine and sine tensors for explicit positions.
    Embeddings {
        /// Cosine values shaped for the input sequence and rotary dimensions.
        cosine: &'a T,
        /// Sine values shaped for the input sequence and rotary dimensions.
        sine: &'a T,
    },
}

impl<T> Copy for RotaryPosition<'_, T> {}

impl<T> Clone for RotaryPosition<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Architecture-selected scoring policy for routed experts.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RoutingScoring {
    /// Apply softmax across all expert logits before selecting routes.
    Softmax,
    /// Apply elementwise sigmoid scores before grouped selection.
    Sigmoid,
    /// Apply the square root of softplus before selection.
    SqrtSoftplus,
}

/// Backend-neutral top-k routing semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopKRoutingSpec {
    expert_count: i32,
    top_k: i32,
    scoring: RoutingScoring,
    normalize_selected: bool,
    normalization_epsilon: f32,
    routed_scaling: f32,
    expert_groups: i32,
    selected_groups: i32,
}

/// Construction specification for a learned top-k router projection.
#[derive(Debug, Clone)]
pub struct TopKRouterSpec {
    /// Hidden width consumed by the router projection.
    pub input_dimensions: i32,
    /// Stable router projection parameter identity.
    pub weight: ParameterSpec,
    /// Optional correction bias used only to choose expert IDs. Gathered
    /// route scores remain unbiased.
    pub correction_bias: Option<ParameterSpec>,
    /// Optional physical encoding of the router projection.
    pub quantization: Option<WeightQuantization>,
    /// Architecture-selected scoring and selection semantics.
    pub routing: TopKRoutingSpec,
}

impl TopKRouterSpec {
    /// Validates positive input geometry.
    pub fn validate(&self) -> Result<(), Error> {
        if self.input_dimensions <= 0 {
            return Err(Error::backend(format!(
                "router input dimensions must be positive, got {}",
                self.input_dimensions
            )));
        }
        Ok(())
    }
}

impl TopKRoutingSpec {
    /// Creates a validated routed-expert selection policy.
    pub fn new(
        expert_count: i32,
        top_k: i32,
        scoring: RoutingScoring,
        normalize_selected: bool,
    ) -> Result<Self, Error> {
        if expert_count <= 0 {
            return Err(Error::backend(format!(
                "expert count must be positive, got {expert_count}"
            )));
        }
        if top_k <= 0 || top_k > expert_count {
            return Err(Error::backend(format!(
                "top-k route count must be in 1..={expert_count}, got {top_k}"
            )));
        }
        Ok(Self {
            expert_count,
            top_k,
            scoring,
            normalize_selected,
            normalization_epsilon: 0.0,
            routed_scaling: 1.0,
            expert_groups: 1,
            selected_groups: 1,
        })
    }

    /// Selects grouped routing semantics.
    pub fn with_groups(mut self, expert_groups: i32, selected_groups: i32) -> Result<Self, Error> {
        if expert_groups <= 0
            || selected_groups <= 0
            || selected_groups > expert_groups
            || self.expert_count % expert_groups != 0
            || self.top_k > selected_groups * (self.expert_count / expert_groups)
        {
            return Err(Error::backend(format!(
                "invalid grouped routing geometry: experts={} top_k={} groups={expert_groups} selected_groups={selected_groups}",
                self.expert_count, self.top_k
            )));
        }
        self.expert_groups = expert_groups;
        self.selected_groups = selected_groups;
        Ok(self)
    }

    /// Selects denominator epsilon and final routed contribution scale.
    pub fn with_weight_policy(
        mut self,
        normalization_epsilon: f32,
        routed_scaling: f32,
    ) -> Result<Self, Error> {
        if !normalization_epsilon.is_finite()
            || normalization_epsilon < 0.0
            || !routed_scaling.is_finite()
            || routed_scaling <= 0.0
        {
            return Err(Error::backend(
                "routing normalization epsilon must be finite and nonnegative and routed scaling must be finite and positive",
            ));
        }
        self.normalization_epsilon = normalization_epsilon;
        self.routed_scaling = routed_scaling;
        Ok(self)
    }

    /// Returns the total number of routed experts.
    pub const fn expert_count(self) -> i32 {
        self.expert_count
    }

    /// Returns the number of selected experts per token.
    pub const fn top_k(self) -> i32 {
        self.top_k
    }

    /// Returns the score transformation applied before selection.
    pub const fn scoring(self) -> RoutingScoring {
        self.scoring
    }

    /// Returns whether selected scores are renormalized to sum to one.
    pub const fn normalize_selected(self) -> bool {
        self.normalize_selected
    }

    /// Returns the epsilon added to selected-score normalization.
    pub const fn normalization_epsilon(self) -> f32 {
        self.normalization_epsilon
    }

    /// Returns the final routed contribution multiplier.
    pub const fn routed_scaling(self) -> f32 {
        self.routed_scaling
    }

    /// Returns the number of equal contiguous expert groups.
    pub const fn expert_groups(self) -> i32 {
        self.expert_groups
    }

    /// Returns the number of groups eligible for expert selection.
    pub const fn selected_groups(self) -> i32 {
        self.selected_groups
    }
}

/// Backend-native result of one top-k route selection.
#[derive(Debug, Clone)]
pub struct RoutingResult<T> {
    /// Selected expert IDs shaped `[..., top_k]`.
    pub expert_ids: T,
    /// Selected scores before optional top-k renormalization.
    pub selected_scores: T,
    /// Final normalized or unnormalized route weights.
    pub route_weights: T,
}

/// Optional bound applied to the inputs of a SwiGLU product.
///
/// The gate branch is capped above before SiLU and the up branch is clamped
/// symmetrically to the same magnitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwiGluLimit(f32);

impl SwiGluLimit {
    /// Creates a finite positive SwiGLU bound.
    pub fn new(limit: f32) -> Result<Self, Error> {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(Error::backend(format!(
                "SwiGLU limit must be finite and positive, got {limit}"
            )));
        }
        Ok(Self(limit))
    }

    /// Returns the gate upper bound and up-branch absolute bound.
    pub const fn get(self) -> f32 {
        self.0
    }
}

/// Statically dispatched top-k router.
pub trait RoutingOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Selects expert IDs and route weights without host materialization.
    fn route(&mut self, logits: &T, context: &T::Context) -> Result<RoutingResult<T>, Error>;

    /// Computes scores and weights for caller-selected global expert IDs.
    fn route_selected(
        &mut self,
        input: &T,
        expert_ids: &T,
        context: &T::Context,
    ) -> Result<RoutingResult<T>, Error>;
}

/// Parameter identities for one SwiGLU expert or one packed expert axis.
#[derive(Debug, Clone)]
pub struct SwiGluExpertParameters {
    /// Gating projection weight.
    pub gate: SwiGluExpertProjection,
    /// Up projection weight.
    pub up: SwiGluExpertProjection,
    /// Down projection weight.
    pub down: SwiGluExpertProjection,
}

/// One expert projection identity and optional physical encoding.
#[derive(Debug, Clone)]
pub struct SwiGluExpertProjection {
    /// Stable logical parameter identity.
    pub weight: ParameterSpec,
    /// Complete physical checkpoint encoding.
    pub format: LinearFormat,
}

/// Logical parameter layout for a SwiGLU expert bank.
#[derive(Debug, Clone)]
pub enum SwiGluExpertLayout {
    /// Fused gate/up and down tensors whose leading axis indexes experts.
    Packed {
        /// Concatenated gate/up projection.
        gate_up: SwiGluExpertProjection,
        /// Down projection.
        down: SwiGluExpertProjection,
    },
    /// Independently materialized expert parameter triples in expert-ID order.
    Independent(Vec<SwiGluExpertParameters>),
}

/// Complete architecture-owned construction specification for routed SwiGLU experts.
#[derive(Debug, Clone)]
pub struct SwiGluExpertBankSpec {
    /// Number of routed experts.
    pub expert_count: i32,
    /// Input hidden width.
    pub input_dimensions: i32,
    /// Per-expert intermediate width.
    pub intermediate_dimensions: i32,
    /// Output hidden width.
    pub output_dimensions: i32,
    /// Optional pre-activation bound shared by every expert.
    pub limit: Option<SwiGluLimit>,
    /// Stable logical parameter identities and storage organization.
    pub layout: SwiGluExpertLayout,
}

impl SwiGluExpertBankSpec {
    /// Validates positive geometry and exact independent-expert cardinality.
    pub fn validate(&self) -> Result<(), Error> {
        for (name, value) in [
            ("expert_count", self.expert_count),
            ("input_dimensions", self.input_dimensions),
            ("intermediate_dimensions", self.intermediate_dimensions),
            ("output_dimensions", self.output_dimensions),
        ] {
            if value <= 0 {
                return Err(Error::backend(format!(
                    "SwiGLU expert-bank {name} must be positive, got {value}"
                )));
            }
        }
        if let SwiGluExpertLayout::Independent(experts) = &self.layout {
            let expected = usize::try_from(self.expert_count).map_err(Error::backend)?;
            if experts.len() != expected {
                return Err(Error::backend(format!(
                    "independent SwiGLU bank has {} experts, expected {expected}",
                    experts.len()
                )));
            }
        }
        Ok(())
    }
}

/// Statically dispatched routed SwiGLU expert bank.
pub trait SwiGluExpertBankOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Executes selected experts and combines their outputs by route weight.
    fn forward_routed(
        &mut self,
        input: &T,
        routes: &RoutingResult<T>,
        context: &T::Context,
    ) -> Result<T, Error>;
}

/// Neural backend extension for routed expert architectures.
pub trait RoutedNeuralBackend: NeuralBackend {
    /// Concrete top-k router.
    type Router: RoutingOperator<Self::Tensor>;
    /// Concrete packed or independently materialized expert bank.
    type SwiGluExpertBank: SwiGluExpertBankOperator<Self::Tensor>;

    /// Builds a router with architecture-selected top-k semantics.
    fn top_k_router(
        spec: TopKRouterSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Router, Error>;

    /// Builds a routed SwiGLU expert bank.
    fn swiglu_expert_bank(
        spec: SwiGluExpertBankSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::SwiGluExpertBank, Error>;
}

/// Complete construction specification for one multi-stream residual mix.
#[derive(Debug, Clone)]
pub struct HyperConnectionSpec {
    /// Number of parallel residual streams.
    pub streams: i32,
    /// Hidden width of each stream.
    pub hidden_size: i32,
    /// Sinkhorn row/column normalization passes.
    pub sinkhorn_iterations: usize,
    /// Positive numerical epsilon used by Sinkhorn normalization.
    pub epsilon: f32,
    /// Projection producing pre, post, and stream-mixing logits.
    pub function: ParameterSpec,
    /// Additive base for the mixing logits.
    pub base: ParameterSpec,
    /// Learned scales for pre, post, and matrix logits.
    pub scale: ParameterSpec,
}

impl HyperConnectionSpec {
    /// Validates geometry and numerical policy without inspecting parameters.
    pub fn validate(&self) -> Result<(), Error> {
        if self.streams <= 0 || self.hidden_size <= 0 {
            return Err(Error::backend(
                "hyper-connection streams and hidden size must be positive",
            ));
        }
        if self.sinkhorn_iterations == 0 {
            return Err(Error::backend(
                "hyper-connection Sinkhorn iteration count must be positive",
            ));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::backend(
                "hyper-connection epsilon must be finite and positive",
            ));
        }
        Ok(())
    }
}

/// Complete construction specification for the final stream collapse.
#[derive(Debug, Clone)]
pub struct HyperHeadSpec {
    /// Number of parallel residual streams.
    pub streams: i32,
    /// Hidden width of each stream.
    pub hidden_size: i32,
    /// RMS preparation epsilon.
    pub norm_epsilon: f32,
    /// Positive offset added to collapse coefficients.
    pub epsilon: f32,
    /// Projection producing per-stream collapse logits.
    pub function: ParameterSpec,
    /// Additive base for collapse logits.
    pub base: ParameterSpec,
    /// Learned collapse-logit scale.
    pub scale: ParameterSpec,
}

impl HyperHeadSpec {
    /// Validates geometry and numerical policy without inspecting parameters.
    pub fn validate(&self) -> Result<(), Error> {
        if self.streams <= 0 || self.hidden_size <= 0 {
            return Err(Error::backend(
                "hyper-head streams and hidden size must be positive",
            ));
        }
        if !self.norm_epsilon.is_finite()
            || self.norm_epsilon <= 0.0
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
        {
            return Err(Error::backend(
                "hyper-head epsilons must be finite and positive",
            ));
        }
        Ok(())
    }
}

/// Coefficients and collapsed value produced before one residual sublayer.
#[derive(Debug, Clone)]
pub struct HyperConnectionState<T> {
    /// Residual streams reduced to one sublayer input.
    pub collapsed: T,
    /// Coefficients used for the pre-sublayer reduction.
    pub pre: T,
    /// Coefficients used to inject the sublayer result into each stream.
    pub post: T,
    /// Doubly-stochastic stream-mixing matrix.
    pub combination: T,
}

/// Backend-native operator for one hyper-connected residual cycle.
pub trait HyperConnectionOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Performs FP32 RMS preparation, predicts mixing coefficients, applies
    /// Sinkhorn normalization, and collapses streams for a sublayer.
    fn collapse(
        &mut self,
        residual: &T,
        norm_epsilon: f32,
        context: &T::Context,
    ) -> Result<HyperConnectionState<T>, Error>;

    /// Injects a sublayer result and mixes the previous residual streams.
    fn expand(
        &mut self,
        sublayer: &T,
        residual: &T,
        state: &HyperConnectionState<T>,
        context: &T::Context,
    ) -> Result<T, Error>;
}

/// Backend-native operator for the final learned stream collapse.
pub trait HyperHeadOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Collapses `[batch, tokens, streams, hidden]` into one hidden state.
    fn forward(&mut self, residual: &T, context: &T::Context) -> Result<T, Error>;
}

/// Neural backend extension for hyper-connected residual architectures.
pub trait HyperNeuralBackend: NeuralBackend {
    /// Concrete multi-stream residual operator.
    type HyperConnection: HyperConnectionOperator<Self::Tensor>;
    /// Concrete final stream-collapse operator.
    type HyperHead: HyperHeadOperator<Self::Tensor>;

    /// Builds one unloaded hyper-connection operator.
    fn hyper_connection(
        spec: HyperConnectionSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::HyperConnection, Error>;

    /// Builds one unloaded final hyper-head operator.
    fn hyper_head(
        spec: HyperHeadSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::HyperHead, Error>;
}

/// Neutral, statically dispatched multi-stream residual layer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct HyperConnection<B: HyperNeuralBackend> {
    operator: B::HyperConnection,
}

impl<B: HyperNeuralBackend> HyperConnection<B> {
    /// Builds the backend operator from architecture-owned parameter slots.
    pub fn new(
        spec: HyperConnectionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        Ok(Self {
            operator: B::hyper_connection(spec, context)?,
        })
    }

    /// Collapses residual streams for one sublayer.
    pub fn collapse(
        &mut self,
        residual: &B::Tensor,
        norm_epsilon: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<HyperConnectionState<B::Tensor>, Error> {
        self.operator.collapse(residual, norm_epsilon, context)
    }

    /// Expands a sublayer result back into residual streams.
    pub fn expand(
        &mut self,
        sublayer: &B::Tensor,
        residual: &B::Tensor,
        state: &HyperConnectionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.operator.expand(sublayer, residual, state, context)
    }
}

/// Neutral, statically dispatched final stream-collapse layer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct HyperHead<B: HyperNeuralBackend> {
    operator: B::HyperHead,
}

impl<B: HyperNeuralBackend> HyperHead<B> {
    /// Builds the backend operator from architecture-owned parameter slots.
    pub fn new(
        spec: HyperHeadSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        Ok(Self {
            operator: B::hyper_head(spec, context)?,
        })
    }

    /// Collapses all residual streams to one hidden state.
    pub fn forward(
        &mut self,
        residual: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.operator.forward(residual, context)
    }
}

/// Backend-native key/value cache operations required by attention.
pub trait AttentionCache<T: Tensor> {
    /// Current absolute sequence offset.
    fn offset(&self) -> i32;
    /// Maximum retained history for a sliding cache.
    fn max_size(&self) -> Option<i32>;
    /// Appends projected keys and values and returns the tensors used by attention.
    fn update_for_attention(
        &mut self,
        keys: T,
        values: T,
        context: &T::Context,
    ) -> Result<(T, T), Error>;
    /// Runs cache-aware attention, including paged or quantized kernels where applicable.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        queries: T,
        keys: T,
        values: T,
        scale: f32,
        mask: Option<&T>,
        context: &T::Context,
    ) -> Result<T, Error>;
}

/// Semantic latent and rotary components retained by compressed attention.
#[derive(Debug, Clone)]
pub struct CompressedAttentionState<T> {
    /// Head-independent latent key/value representation.
    pub latent: T,
    /// Rotary-key representation aligned to the same token range.
    pub rotary: T,
}

/// Result of appending compressed state.
#[derive(Debug, Clone)]
pub enum CompressedAttentionView<T> {
    /// Complete resident state used directly by prefill or decode attention.
    Resident(CompressedAttentionState<T>),
    /// Paged state; the appended components are retained for diagnostics while
    /// attention scans semantic blocks through the cache operator.
    Paged {
        /// Components appended by this update.
        appended: CompressedAttentionState<T>,
    },
}

impl<T> CompressedAttentionView<T> {
    /// Returns the resident state when paging is not active.
    pub const fn resident(&self) -> Option<&CompressedAttentionState<T>> {
        match self {
            Self::Resident(state) => Some(state),
            Self::Paged { .. } => None,
        }
    }

    /// Returns components suitable for non-materializing observation.
    pub const fn observable(&self) -> &CompressedAttentionState<T> {
        match self {
            Self::Resident(state) | Self::Paged { appended: state } => state,
        }
    }

    /// Returns whether block-addressable paging is active.
    pub const fn is_paged(&self) -> bool {
        matches!(self, Self::Paged { .. })
    }
}

/// One absolute token range supplied to blockwise compressed attention.
#[derive(Debug, Clone)]
pub struct CompressedAttentionBlock<T> {
    /// Inclusive absolute token start.
    pub start: i64,
    /// Exclusive absolute token end.
    pub end: i64,
    /// Latent and rotary components for this range.
    pub state: CompressedAttentionState<T>,
}

/// Aggregate observations from one paged blockwise attention scan.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct CompressedAttentionScan {
    /// Number of scanned sealed and tail blocks.
    pub blocks: u64,
    /// Number of persistent compressed bytes scanned.
    pub bytes: u64,
    /// Maximum caller-reported transient reconstruction bytes.
    pub reconstruction_scratch_bytes: u64,
}

/// Fixed request metadata for an online blockwise attention recurrence.
#[derive(Debug, Clone, Copy)]
pub struct BlockwiseAttentionSpec<'a, T> {
    /// Queries shaped `[batch, heads, query_tokens, dimensions]`.
    pub queries: &'a T,
    /// Query/key score multiplier.
    pub scale: f32,
    /// Optional complete-context mask sliced by the backend for each block.
    pub mask: Option<&'a T>,
    /// Absolute position of the first query.
    pub query_start: i64,
    /// Absolute exclusive end of the complete visible context.
    pub context_end: i64,
    /// Optional causal sliding-window width.
    pub sliding_window: Option<i32>,
    /// Number of prefix tokens visible outside a sliding window.
    pub prefix_tokens: i64,
    /// Optional learned per-head sink logits.
    pub sinks: Option<&'a T>,
}

/// Typed backend fusion for exact online softmax across ordered attention
/// blocks. Persistent caches remain compressed; only one reconstructed block
/// is live at a time.
pub trait BlockwiseAttentionBackend: NeuralBackend {
    /// Backend-native running maximum, normalization, and value accumulator.
    type BlockwiseAccumulator;

    /// Starts an empty online attention recurrence.
    fn begin_blockwise_attention(
        spec: BlockwiseAttentionSpec<'_, Self::Tensor>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::BlockwiseAccumulator, Error>;

    /// Incorporates one absolute key/value block and reports transient bytes.
    fn accumulate_blockwise_attention(
        accumulator: &mut Self::BlockwiseAccumulator,
        start: i64,
        end: i64,
        keys: Self::Tensor,
        values: Self::Tensor,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<u64, Error>;

    /// Finishes the online recurrence.
    fn finish_blockwise_attention(
        accumulator: Self::BlockwiseAccumulator,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;
}

/// Backend-owned compressed-attention state over semantic components.
///
/// Checkpoints are cheap backend snapshots used by speculative fork/rollback.
/// `finalize` seals mutable state before runtime prompt-cache persistence.
pub trait CompressedAttentionCache<T: Tensor>: Clone + Debug {
    /// Backend snapshot preserving paging and residency identity.
    type Checkpoint: Clone + Debug;

    /// Current absolute token frontier.
    fn offset(&self) -> i32;
    /// Whether state is block-addressable rather than fully resident.
    fn is_paged(&self) -> bool;
    /// Appends aligned latent and rotary components.
    fn append(
        &mut self,
        state: CompressedAttentionState<T>,
        context: &T::Context,
    ) -> Result<CompressedAttentionView<T>, Error>;
    /// Visits all paged blocks in absolute order without host tensor
    /// materialization. The callback reports its transient scratch usage.
    fn visit_blocks<F>(
        &mut self,
        query_tokens: i32,
        context: &T::Context,
        visitor: F,
    ) -> Result<CompressedAttentionScan, Error>
    where
        F: FnMut(CompressedAttentionBlock<T>) -> Result<u64, Error>;
    /// Captures a cheap speculative checkpoint.
    fn checkpoint(&self) -> Self::Checkpoint;
    /// Restores a previous checkpoint, removing any later paged state.
    fn restore(&mut self, checkpoint: &Self::Checkpoint, context: &T::Context)
        -> Result<(), Error>;
    /// Seals mutable tails before prompt-cache snapshot persistence.
    fn finalize(&mut self) -> Result<(), Error>;
    /// Clears all resident or paged state.
    fn clear(&mut self) -> Result<(), Error>;
}

/// Complete source windows emitted by an append-only gated-pooling stream.
#[derive(Debug, Clone)]
pub struct PoolingWindows<T> {
    /// Source values shaped `[batch, complete_source_tokens, width]`.
    pub values: T,
    /// Gate logits aligned to `values`.
    pub gates: T,
    /// Absolute source-token position of the first returned value.
    pub base_position: i32,
}

/// Previous overlap carried between adjacent gated-pooling windows.
#[derive(Debug, Clone)]
pub struct PoolingOverlap<T> {
    /// Previous-window source values, when a complete predecessor exists.
    pub values: Option<T>,
    /// Previous-window gate logits aligned to `values`.
    pub gates: Option<T>,
}

/// Backend-owned local-key and compressed-pooling state used by architectures
/// that mix bounded local attention with append-only pooled history.
///
/// Stream ordinals are architecture-owned. Implementations preserve pending,
/// pooled, and overlap components through checkpoint, rollback, prompt-cache,
/// resident, and paged realizations without exposing storage classes here.
pub trait PoolingAttentionCache<T: Tensor>: Clone + Debug {
    /// Backend snapshot preserving all local and pooling state.
    type Checkpoint: Clone + Debug;

    /// Current absolute source-token frontier.
    fn offset(&self) -> i32;
    /// Returns the configured source-token ratio for one pooling stream.
    fn pooling_ratio(&self, stream: u32) -> Option<i32>;
    /// Appends local keys and returns the bounded history used by attention,
    /// shaped `[batch, local_tokens, dimensions]`.
    fn append_local(&mut self, keys: T, context: &T::Context) -> Result<T, Error>;
    /// Builds causal eligibility for the currently retained local history.
    fn local_mask(&self, query_tokens: i32, offset: i32, context: &T::Context) -> Result<T, Error>;
    /// Accumulates source values and gates, retaining any incomplete suffix.
    fn accumulate_pooling_windows(
        &mut self,
        stream: u32,
        values: T,
        gates: T,
        absolute_offset: i32,
        context: &T::Context,
    ) -> Result<PoolingWindows<T>, Error>;
    /// Replaces overlap carried by one stream and returns its previous pair.
    fn replace_pooling_overlap(
        &mut self,
        stream: u32,
        values: T,
        gates: T,
    ) -> Result<PoolingOverlap<T>, Error>;
    /// Appends newly pooled values and returns the complete pooled history.
    fn append_pooled(&mut self, stream: u32, values: T, context: &T::Context) -> Result<T, Error>;
    /// Builds causal eligibility for one stream's complete pooled history.
    fn pooling_mask(
        &self,
        stream: u32,
        query_tokens: i32,
        offset: i32,
        context: &T::Context,
    ) -> Result<Option<T>, Error>;
    /// Captures a cheap speculative checkpoint.
    fn checkpoint(&self) -> Self::Checkpoint;
    /// Restores a previous checkpoint.
    fn restore(&mut self, checkpoint: &Self::Checkpoint, context: &T::Context)
        -> Result<(), Error>;
    /// Seals mutable local and pooled tails before persistence.
    fn finalize(&mut self) -> Result<(), Error>;
    /// Clears all local and pooling state.
    fn clear(&mut self) -> Result<(), Error>;
}

/// General neural-operator family selected by a shared architecture.
///
/// Associated concrete types make calls statically dispatched. Implementations
/// retain ownership of tensor storage, fusion, quantization, and collectives.
pub trait NeuralBackend: Sized + 'static {
    /// Backend tensor handle.
    type Tensor: Tensor;
    /// Native affine projection, including packed quantized variants.
    type Linear: LinearOperator<Self::Tensor>;
    /// Native embedding table.
    type Embedding: EmbeddingOperator<Self::Tensor>;
    /// Native normalization operator.
    type Normalization: NormalizationOperator<Self::Tensor>;
    /// Native rotary-position operator.
    type Rotary: RotaryOperator<Self::Tensor>;
    /// Backend collective context used by tensor-parallel execution.
    type ParallelContext: ?Sized;

    /// Builds one affine projection.
    fn linear(
        spec: LinearSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Linear, Error>;
    /// Builds one token embedding table.
    fn embedding(
        spec: EmbeddingSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Embedding, Error>;
    /// Builds one RMS normalization operator.
    fn rms_norm(
        spec: NormalizationSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Normalization, Error>;
    /// Builds the model's rotary-position operator.
    fn rotary(
        spec: RotarySpec<'_>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Rotary, Error>;
    /// Applies SiLU using a backend-native implementation.
    fn silu(
        input: Self::Tensor,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;
    /// Runs sparse attention over bounded local state and caller-selected
    /// compressed positions, optionally sharing softmax normalization with
    /// learned attention sinks.
    fn indexed_attention(
        input: IndexedAttentionInput<'_, Self::Tensor>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (input, context);
        Err(Error::backend(
            "indexed attention is not implemented by this backend",
        ))
    }
    /// Runs dense shared-softmax attention over local and pooled history.
    fn pooled_attention(
        input: PooledAttentionInput<'_, Self::Tensor>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (input, context);
        Err(Error::backend(
            "pooled attention is not implemented by this backend",
        ))
    }
    /// Selects pooled positions for indexed attention.
    fn select_pooled_positions(
        input: PooledPositionInput<'_, Self::Tensor>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (input, context);
        Err(Error::backend(
            "pooled-position selection is not implemented by this backend",
        ))
    }
    /// Gathers one full pooled-eligibility mask at selected positions.
    fn gather_pooled_mask(
        mask: &Self::Tensor,
        selected_positions: &Self::Tensor,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (mask, selected_positions, context);
        Err(Error::backend(
            "pooled-mask gathering is not implemented by this backend",
        ))
    }
    /// Runs dense attention with optional learned per-head sink logits.
    #[allow(clippy::too_many_arguments)]
    fn attention_with_sinks(
        queries: Self::Tensor,
        keys: Self::Tensor,
        values: Self::Tensor,
        scale: f32,
        mask: Option<&Self::Tensor>,
        sinks: Option<&Self::Tensor>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        if sinks.is_some() {
            return Err(Error::backend(
                "attention sinks are not implemented by this backend",
            ));
        }
        Self::attention(queries, keys, values, scale, mask, context)
    }
    /// Applies RMS normalization without a learned scale.
    fn rms_norm_without_weight(
        input: &Self::Tensor,
        epsilon: f32,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (input, epsilon, context);
        Err(Error::backend(
            "weightless RMS normalization is not implemented by this backend",
        ))
    }
    /// Applies one packed block-diagonal projection independently across an
    /// explicit group axis.
    fn grouped_linear(
        linear: &mut Self::Linear,
        input: &Self::Tensor,
        groups: i32,
        output_per_group: i32,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (linear, input, groups, output_per_group, context);
        Err(Error::backend(
            "grouped linear projection is not implemented by this backend",
        ))
    }
    /// Applies the shared SwiGLU activation and optional pre-activation bound.
    fn swiglu(
        gate: Self::Tensor,
        up: Self::Tensor,
        limit: Option<SwiGluLimit>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;
    /// Runs un-cached scaled dot-product attention.
    fn attention(
        queries: Self::Tensor,
        keys: Self::Tensor,
        values: Self::Tensor,
        scale: f32,
        mask: Option<&Self::Tensor>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;
    /// Runs causal sliding-window prefill attention without a square mask.
    #[allow(clippy::too_many_arguments)]
    fn sliding_window_attention(
        queries: Self::Tensor,
        keys: Self::Tensor,
        values: Self::Tensor,
        scale: f32,
        window: i32,
        position_offset: i32,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;
    /// Builds the backend-native boolean causal mask used for a prefill.
    ///
    /// The returned value remains a lazy/backend-owned tensor. Calling this
    /// method must not synchronize or materialize mask contents on the host.
    fn causal_mask(
        sequence: i32,
        offset: i32,
        window: Option<i32>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;
    /// Applies a row-parallel projection and its collective reduction.
    fn row_parallel_linear(
        linear: &mut Self::Linear,
        input: &Self::Tensor,
        parallel: &Self::ParallelContext,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error>;
}

/// Opaque tensor handle and the neural operations required by shared Eredu
/// architectures.
///
/// Implementations must preserve backend-native execution semantics. None of
/// these operations imply host materialization or synchronization.
pub trait Tensor: Clone + Debug + Sized + 'static {
    /// Backend execution context, such as an MLX stream.
    type Context: ?Sized;

    /// Logical tensor shape maintained without materializing tensor values.
    fn shape(&self) -> &[i32];

    /// Returns one logical dimension.
    fn dim(&self, axis: usize) -> i32 {
        self.shape()[axis]
    }

    /// Allocates an unloaded floating-point parameter tensor.
    fn unloaded_f32(shape: &[i32], context: &Self::Context) -> Result<Self, Error>;
    /// Allocates an unloaded signed 32-bit integer parameter tensor.
    fn unloaded_i32(shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        let _ = (shape, context);
        Err(Error::backend(
            "I32 parameter allocation is not implemented by this backend",
        ))
    }
    /// Creates a floating-point tensor from host initialization data.
    fn from_f32_slice(
        values: &[f32],
        shape: &[i32],
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Creates a floating-point tensor filled with one scalar.
    fn full_f32(value: f32, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        let _ = (value, shape, context);
        Err(Error::backend(
            "filled tensor construction is not implemented by this backend",
        ))
    }
    /// Creates a signed 32-bit integer tensor filled with one scalar.
    fn full_i32(value: i32, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        let _ = (value, shape, context);
        Err(Error::backend(
            "filled I32 tensor construction is not implemented by this backend",
        ))
    }
    /// Elementwise addition.
    fn add(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error>;
    /// Elementwise subtraction.
    fn subtract(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error>;
    /// Elementwise multiplication.
    fn multiply(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error>;
    /// Elementwise multiplication by a floating-point scalar.
    fn multiply_scalar(&self, rhs: f32, context: &Self::Context) -> Result<Self, Error>;
    /// Elementwise division.
    fn divide(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error>;
    /// Elementwise square.
    fn square(&self, context: &Self::Context) -> Result<Self, Error>;
    /// Elementwise maximum with a scalar.
    fn maximum_scalar(&self, rhs: f32, context: &Self::Context) -> Result<Self, Error>;

    /// Softmax over one axis.
    fn softmax_axis(
        &self,
        axis: i32,
        precise: bool,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        let _ = (axis, precise, context);
        Err(Error::backend(
            "softmax is not implemented by this tensor backend",
        ))
    }

    /// Reshapes without changing logical element order.
    fn reshape(&self, shape: &[i32], context: &Self::Context) -> Result<Self, Error>;
    /// Broadcasts to a compatible target shape.
    fn broadcast_to(&self, shape: &[i32], context: &Self::Context) -> Result<Self, Error> {
        let _ = (shape, context);
        Err(Error::backend(
            "broadcasting is not implemented by this tensor backend",
        ))
    }
    /// Permutes axes.
    fn transpose_axes(&self, axes: &[i32], context: &Self::Context) -> Result<Self, Error>;
    /// Swaps two axes.
    fn swap_axes(&self, left: i32, right: i32, context: &Self::Context) -> Result<Self, Error>;
    /// Reverses the axes of a rank-two tensor.
    fn transpose(&self, context: &Self::Context) -> Result<Self, Error>;
    /// Inserts one unit dimension.
    fn expand_dims(&self, axis: i32, context: &Self::Context) -> Result<Self, Error>;
    /// Removes unit dimensions.
    fn squeeze_axes(&self, axes: &[i32], context: &Self::Context) -> Result<Self, Error>;
    /// Creates a tensor view using backend-neutral axis indexes.
    fn index(&self, indexes: &[Index], context: &Self::Context) -> Result<Self, Error>;
    /// Takes rows along one axis using a backend index tensor.
    fn take_axis(&self, indexes: &Self, axis: i32, context: &Self::Context) -> Result<Self, Error>;

    /// Applies rotary positions using caller-supplied reciprocal frequencies.
    fn rope_with_frequencies(
        &self,
        dimensions: i32,
        traditional: bool,
        offset: i32,
        frequencies: &Self,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        let _ = (dimensions, traditional, offset, frequencies, context);
        Err(Error::backend(
            "explicit-frequency rotary positions are not implemented by this tensor backend",
        ))
    }

    /// Concatenates tensors along an axis.
    fn concatenate(values: &[Self], axis: i32, context: &Self::Context) -> Result<Self, Error>;
    /// Stacks tensors along a new axis.
    fn stack(values: &[Self], axis: i32, context: &Self::Context) -> Result<Self, Error>;
    /// Matrix multiplication.
    fn matmul(lhs: &Self, rhs: &Self, context: &Self::Context) -> Result<Self, Error>;
    /// Reduction sum.
    fn sum_axis(
        value: &Self,
        axis: i32,
        keep_dims: bool,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Reduction mean.
    fn mean_axis(
        value: &Self,
        axis: i32,
        keep_dims: bool,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        let width = value
            .shape()
            .get(if axis < 0 {
                usize::try_from(value.shape().len() as i32 + axis).unwrap_or(usize::MAX)
            } else {
                usize::try_from(axis).unwrap_or(usize::MAX)
            })
            .copied()
            .ok_or_else(|| Error::backend(format!("mean axis {axis} is out of range")))?;
        Self::sum_axis(value, axis, keep_dims, context)?
            .multiply_scalar(1.0 / width as f32, context)
    }
    /// Reduction argmin.
    fn argmin_axis(
        value: &Self,
        axis: i32,
        keep_dims: bool,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Pads a tensor.
    fn pad(
        value: &Self,
        widths: &[(i32, i32)],
        mode: PadMode,
        context: &Self::Context,
    ) -> Result<Self, Error>;

    /// One-dimensional convolution.
    #[allow(clippy::too_many_arguments)]
    fn conv1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// One-dimensional transposed convolution.
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose1d(
        input: &Self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        output_padding: i32,
        groups: i32,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Affine linear projection. Backends may use fused or quantized paths.
    fn linear(
        input: &Self,
        weight: &Self,
        bias: Option<&Self>,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Layer normalization.
    fn layer_norm(
        input: &Self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        epsilon: f32,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Gaussian error linear unit.
    fn gelu(input: &Self, context: &Self::Context) -> Result<Self, Error>;
    /// Exponential linear unit.
    fn elu(input: &Self, alpha: f32, context: &Self::Context) -> Result<Self, Error>;
    /// Rotary positional encoding.
    #[allow(clippy::too_many_arguments)]
    fn rope(
        input: &Self,
        dimensions: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: i32,
        context: &Self::Context,
    ) -> Result<Self, Error>;
    /// Scaled dot-product attention. Backends retain control over fusion.
    fn scaled_dot_product_attention(
        queries: &Self,
        keys: &Self,
        values: &Self,
        scale: f32,
        mask: AttentionMask<'_, Self>,
        context: &Self::Context,
    ) -> Result<Self, Error>;
}

/// One trainable parameter owned by a generic architecture module.
#[derive(Debug, Clone)]
pub struct Parameter<T> {
    spec: ParameterSpec,
    trainable: bool,
    value: T,
}

impl<T> Parameter<T> {
    /// Wraps a backend tensor as an authoritative parameter slot.
    pub fn new(spec: ParameterSpec, value: T) -> Self {
        let trainable = spec.trainable;
        Self {
            spec,
            trainable,
            value,
        }
    }
    /// Borrows the backend tensor.
    pub const fn as_ref(&self) -> &T {
        &self.value
    }
    /// Replaces the backend tensor.
    pub fn replace(&mut self, value: T) {
        self.value = value;
    }
}

impl<T: Tensor> Parameter<T> {
    /// Creates an unloaded floating-point parameter.
    pub fn unloaded(
        spec: ParameterSpec,
        shape: &[i32],
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self::new(spec, T::unloaded_f32(shape, context)?))
    }

    /// Creates an unloaded signed 32-bit integer parameter.
    pub fn unloaded_i32(
        spec: ParameterSpec,
        shape: &[i32],
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self::new(spec, T::unloaded_i32(shape, context)?))
    }
}

impl<T: 'static> Parameterized<T> for Parameter<T> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, T>,
    {
        visitor.visit(
            ParameterMetadata::from_spec(&self.spec, self.trainable),
            &self.value,
        );
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, T>,
    {
        visitor.visit_mut(
            ParameterMetadata::from_spec(&self.spec, self.trainable),
            &mut self.value,
        );
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.trainable = trainable;
    }
}

impl<T: 'static, M: Parameterized<T>> Parameterized<T> for Vec<M> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, T>,
    {
        for module in self {
            module.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, T>,
    {
        for module in self {
            module.visit_parameters_mut(visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        for module in self {
            module.set_trainable(trainable);
        }
    }
}

impl<T: 'static, M: Parameterized<T>> Parameterized<T> for Option<M> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, T>,
    {
        if let Some(module) = self {
            module.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, T>,
    {
        if let Some(module) = self {
            module.visit_parameters_mut(visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        if let Some(module) = self {
            module.set_trainable(trainable);
        }
    }
}

#[cfg(test)]
mod routed_contract_tests {
    use super::*;

    fn parameters(prefix: &str) -> SwiGluExpertParameters {
        SwiGluExpertParameters {
            gate: SwiGluExpertProjection {
                weight: ParameterSpec::trainable(format!("{prefix}.gate.weight")).unwrap(),
                format: LinearFormat::Dense,
            },
            up: SwiGluExpertProjection {
                weight: ParameterSpec::trainable(format!("{prefix}.up.weight")).unwrap(),
                format: LinearFormat::Dense,
            },
            down: SwiGluExpertProjection {
                weight: ParameterSpec::trainable(format!("{prefix}.down.weight")).unwrap(),
                format: LinearFormat::Dense,
            },
        }
    }

    #[test]
    fn top_k_routing_policy_rejects_invalid_selection_counts() {
        assert!(TopKRoutingSpec::new(8, 2, RoutingScoring::Softmax, true).is_ok());
        assert!(TopKRoutingSpec::new(0, 1, RoutingScoring::Softmax, false).is_err());
        assert!(TopKRoutingSpec::new(8, 9, RoutingScoring::Softmax, false).is_err());
    }

    #[test]
    fn independent_expert_layout_requires_exact_cardinality() {
        let valid = SwiGluExpertBankSpec {
            expert_count: 2,
            input_dimensions: 16,
            intermediate_dimensions: 8,
            output_dimensions: 16,
            limit: None,
            layout: SwiGluExpertLayout::Independent(vec![parameters("e0"), parameters("e1")]),
        };
        assert!(valid.validate().is_ok());
        let invalid = SwiGluExpertBankSpec {
            layout: SwiGluExpertLayout::Independent(vec![parameters("e0")]),
            ..valid
        };
        assert!(invalid.validate().is_err());
    }
}

/// Generic affine projection.
#[derive(Debug, Clone)]
pub struct Linear<T> {
    /// Projection weights shaped `[output, input]`.
    pub weight: Parameter<T>,
    /// Optional output bias.
    pub bias: Option<Parameter<T>>,
}

impl<T: Tensor> Linear<T> {
    /// Creates unloaded projection parameters.
    pub fn unloaded(spec: LinearSpec, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            weight: Parameter::unloaded(spec.weight, &[spec.output, spec.input], context)?,
            bias: spec
                .bias
                .map(|bias| Parameter::unloaded(bias, &[spec.output], context))
                .transpose()?,
        })
    }

    /// Applies the projection without materializing backend tensors.
    pub fn forward(&self, input: &T, context: &T::Context) -> Result<T, Error> {
        T::linear(
            input,
            self.weight.as_ref(),
            self.bias.as_ref().map(Parameter::as_ref),
            context,
        )
    }
}

impl<T: 'static> Parameterized<T> for Linear<T> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, T>,
    {
        self.weight.visit_parameters(visitor);
        if let Some(bias) = &self.bias {
            bias.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, T>,
    {
        self.weight.visit_parameters_mut(visitor);
        if let Some(bias) = &mut self.bias {
            bias.visit_parameters_mut(visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.weight.set_trainable(trainable);
        if let Some(bias) = &mut self.bias {
            bias.set_trainable(trainable);
        }
    }
}

/// Generic affine layer normalization.
#[derive(Debug, Clone)]
pub struct LayerNorm<T> {
    /// Numerical stability epsilon.
    pub epsilon: f32,
    /// Optional trainable scale.
    pub weight: Option<Parameter<T>>,
    /// Optional trainable bias.
    pub bias: Option<Parameter<T>>,
}

impl<T: Tensor> LayerNorm<T> {
    /// Creates an unloaded affine layer normalization.
    pub fn unloaded(
        dimensions: i32,
        epsilon: f32,
        weight: Option<ParameterSpec>,
        bias: Option<ParameterSpec>,
        context: &T::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            epsilon,
            weight: weight
                .map(|weight| Parameter::unloaded(weight, &[dimensions], context))
                .transpose()?,
            bias: bias
                .map(|bias| Parameter::unloaded(bias, &[dimensions], context))
                .transpose()?,
        })
    }

    /// Applies layer normalization through the selected backend.
    pub fn forward(&self, input: &T, context: &T::Context) -> Result<T, Error> {
        T::layer_norm(
            input,
            self.weight.as_ref().map(Parameter::as_ref),
            self.bias.as_ref().map(Parameter::as_ref),
            self.epsilon,
            context,
        )
    }
}

impl<T: 'static> Parameterized<T> for LayerNorm<T> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, T>,
    {
        if let Some(weight) = &self.weight {
            weight.visit_parameters(visitor);
        }
        if let Some(bias) = &self.bias {
            bias.visit_parameters(visitor);
        }
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, T>,
    {
        if let Some(weight) = &mut self.weight {
            weight.visit_parameters_mut(visitor);
        }
        if let Some(bias) = &mut self.bias {
            bias.visit_parameters_mut(visitor);
        }
    }

    fn set_trainable(&mut self, trainable: bool) {
        if let Some(weight) = &mut self.weight {
            weight.set_trainable(trainable);
        }
        if let Some(bias) = &mut self.bias {
            bias.set_trainable(trainable);
        }
    }
}

#[cfg(test)]
mod parameter_topology_tests {
    use super::*;

    #[derive(Parameterized)]
    #[parameterized(tensor = "i32")]
    struct DerivedModule {
        first: Parameter<i32>,
        second: Option<Parameter<i32>>,
        #[parameter(skip)]
        label: &'static str,
    }

    #[derive(Parameterized)]
    #[parameterized(tensor = "i32")]
    enum DerivedChoice {
        Present(Parameter<i32>),
        Empty,
    }

    fn parameter(id: &str, value: i32) -> Parameter<i32> {
        Parameter::new(ParameterSpec::trainable(id).unwrap(), value)
    }

    #[test]
    fn derive_recurses_through_structs_options_and_enums() {
        let mut module = DerivedModule {
            first: parameter("first.weight", 1),
            second: Some(parameter("second.weight", 2)),
            label: "not a parameter",
        };
        assert_eq!(module.label, "not a parameter");
        let metadata = validate_parameter_topology::<i32, _>(&module).unwrap();
        assert_eq!(
            metadata
                .iter()
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            ["first.weight", "second.weight"]
        );

        module.set_trainable(false);
        assert!(validate_parameter_topology::<i32, _>(&module)
            .unwrap()
            .iter()
            .all(|entry| !entry.trainable));

        let choice = DerivedChoice::Present(parameter("choice.weight", 3));
        assert_eq!(
            validate_parameter_topology::<i32, _>(&choice).unwrap()[0]
                .id
                .as_str(),
            "choice.weight"
        );
        assert!(validate_parameter_topology::<i32, _>(&DerivedChoice::Empty)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn validation_rejects_duplicates_and_invalid_aliases() {
        let duplicate = vec![parameter("same.weight", 1), parameter("same.weight", 2)];
        assert!(matches!(
            validate_parameter_topology::<i32, _>(&duplicate),
            Err(ParameterTopologyError::DuplicateId(id)) if id.as_str() == "same.weight"
        ));

        let alias = Parameter::new(
            ParameterSpec {
                id: ParameterId::new("alias.weight").unwrap(),
                trainable: true,
                alias_of: Some(ParameterId::new("missing.weight").unwrap()),
                group: None,
            },
            1,
        );
        assert!(matches!(
            validate_parameter_topology::<i32, _>(&alias),
            Err(ParameterTopologyError::MissingAliasDestination { .. })
        ));
    }
}

/// Generic rotary positional encoding configuration.
#[derive(Debug, Clone, Copy)]
pub struct Rope {
    dimensions: i32,
    traditional: bool,
    base: f32,
    scale: f32,
}

impl Rope {
    /// Creates a rotary positional encoding.
    pub const fn new(dimensions: i32, traditional: bool, base: f32, scale: f32) -> Self {
        Self {
            dimensions,
            traditional,
            base,
            scale,
        }
    }

    /// Applies rotary positional encoding through the selected backend.
    pub fn forward<T: Tensor>(
        &self,
        input: &T,
        offset: i32,
        context: &T::Context,
    ) -> Result<T, Error> {
        T::rope(
            input,
            self.dimensions,
            self.traditional,
            self.base,
            self.scale,
            offset,
            context,
        )
    }
}
