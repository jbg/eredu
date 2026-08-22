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

/// Reusable patch projection and multi-axis position operations.
pub mod multimodal;
/// Pure sequence layouts shared by patch-based encoders.
pub mod sequence_layout;

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

/// Validated expansion from a smaller logical head axis to a repeated head
/// axis. Repetition preserves grouped order: each source head is repeated
/// contiguously `target_heads / source_heads` times.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct HeadExpansion {
    /// Tensor axis containing source heads.
    pub axis: usize,
    /// Number of source heads.
    pub source_heads: i32,
    /// Number of heads after expansion.
    pub target_heads: i32,
}

impl HeadExpansion {
    /// Validates the head counts and the selected tensor axis.
    pub fn validate<T: Tensor>(&self, input: &T) -> Result<(), Error> {
        let shape = input.shape();
        if self.source_heads <= 0
            || self.target_heads <= 0
            || self.target_heads % self.source_heads != 0
            || shape.get(self.axis).copied() != Some(self.source_heads)
        {
            return Err(Error::backend(format!(
                "invalid head expansion axis={} source={} target={} shape={shape:?}",
                self.axis, self.source_heads, self.target_heads
            )));
        }
        Ok(())
    }

    /// Number of adjacent copies of each source head.
    pub const fn repeats(self) -> i32 {
        self.target_heads / self.source_heads
    }
}

/// One unmasked attention request over validated contiguous sequence segments.
#[derive(Debug, Clone, Copy)]
pub struct SegmentedAttentionInput<'a, T> {
    /// Queries shaped `[tokens, heads, dimensions]`.
    pub queries: &'a T,
    /// Keys shaped `[tokens, heads, dimensions]`.
    pub keys: &'a T,
    /// Values shaped `[tokens, heads, value_dimensions]`.
    pub values: &'a T,
    /// Positive contiguous segment lengths whose sum equals `tokens`.
    pub segment_lengths: &'a [i32],
    /// Query/key score multiplier.
    pub scale: f32,
}

impl<T: Tensor> SegmentedAttentionInput<'_, T> {
    /// Validates tensor and segment geometry without inspecting tensor values.
    pub fn validate(&self) -> Result<(), Error> {
        let query = self.queries.shape();
        let key = self.keys.shape();
        let value = self.values.shape();
        if query.len() != 3
            || key.len() != 3
            || value.len() != 3
            || query[0] <= 0
            || query[1] <= 0
            || query[2] <= 0
            || query[0] != key[0]
            || query[0] != value[0]
            || query[1] != key[1]
            || query[1] != value[1]
            || query[2] != key[2]
            || value[2] <= 0
            || !self.scale.is_finite()
            || self.scale <= 0.0
        {
            return Err(Error::backend(format!(
                "invalid segmented attention geometry q={query:?} k={key:?} v={value:?} scale={}",
                self.scale
            )));
        }
        validate_segment_lengths(query[0], self.segment_lengths)
    }
}

/// Validates positive contiguous segment lengths and their exact total.
pub fn validate_segment_lengths(total: i32, segment_lengths: &[i32]) -> Result<(), Error> {
    if total <= 0 || segment_lengths.is_empty() {
        return Err(Error::backend(format!(
            "segmented attention requires a positive total and at least one segment, got total={total} segments={segment_lengths:?}"
        )));
    }
    let mut sum = 0i32;
    for &length in segment_lengths {
        if length <= 0 {
            return Err(Error::backend(format!(
                "segmented attention lengths must be positive, got {segment_lengths:?}"
            )));
        }
        sum = sum.checked_add(length).ok_or_else(|| {
            Error::backend("segmented attention length total overflowed signed 32-bit geometry")
        })?;
        if sum > total {
            return Err(Error::backend(format!(
                "segmented attention lengths exceed total {total}: {segment_lengths:?}"
            )));
        }
    }
    if sum != total {
        return Err(Error::backend(format!(
            "segmented attention lengths sum to {sum}, expected {total}"
        )));
    }
    Ok(())
}

/// Deterministic host reference for grouped head repetition.
pub fn reference_expand_heads(
    values: &[f32],
    shape: &[usize],
    axis: usize,
    target_heads: usize,
) -> Result<(Vec<f32>, Vec<usize>), Error> {
    let source_heads = shape.get(axis).copied().unwrap_or(0);
    if source_heads == 0 || target_heads == 0 || !target_heads.is_multiple_of(source_heads) {
        return Err(Error::backend(format!(
            "invalid reference head expansion axis={axis} target={target_heads} shape={shape:?}"
        )));
    }
    let elements = shape.iter().try_fold(1usize, |total, width| {
        total
            .checked_mul(*width)
            .ok_or_else(|| Error::backend("reference head expansion element count overflowed"))
    })?;
    if elements != values.len() {
        return Err(Error::backend(format!(
            "reference head expansion expected {elements} values, got {}",
            values.len()
        )));
    }
    let outer = shape[..axis].iter().product::<usize>();
    let inner = shape[axis + 1..].iter().product::<usize>();
    let repeats = target_heads / source_heads;
    let mut output = Vec::with_capacity(outer * target_heads * inner);
    for outer_index in 0..outer {
        for source in 0..source_heads {
            let start = (outer_index * source_heads + source) * inner;
            for _ in 0..repeats {
                output.extend_from_slice(&values[start..start + inner]);
            }
        }
    }
    let mut output_shape = shape.to_vec();
    output_shape[axis] = target_heads;
    Ok((output, output_shape))
}

/// Deterministic host reference for unmasked segmented scaled-dot-product
/// attention. Inputs and output use token-major `[tokens, heads, dimensions]`
/// storage.
#[allow(clippy::too_many_arguments)]
pub fn reference_segmented_attention(
    tokens: usize,
    heads: usize,
    dimensions: usize,
    value_dimensions: usize,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    segment_lengths: &[i32],
    scale: f32,
) -> Result<Vec<f32>, Error> {
    let tokens_i32 = i32::try_from(tokens)
        .map_err(|_| Error::backend("reference segmented attention token count exceeds i32"))?;
    validate_segment_lengths(tokens_i32, segment_lengths)?;
    if heads == 0
        || dimensions == 0
        || value_dimensions == 0
        || !scale.is_finite()
        || scale <= 0.0
        || queries.len() != tokens * heads * dimensions
        || keys.len() != tokens * heads * dimensions
        || values.len() != tokens * heads * value_dimensions
    {
        return Err(Error::backend(
            "invalid reference segmented attention geometry",
        ));
    }
    let mut output = vec![0.0f32; tokens * heads * value_dimensions];
    let mut segment_start = 0usize;
    for &length in segment_lengths {
        let length = usize::try_from(length).expect("validated positive segment length");
        let segment_end = segment_start + length;
        for query_token in segment_start..segment_end {
            for head in 0..heads {
                let mut scores = Vec::with_capacity(length);
                for key_token in segment_start..segment_end {
                    let mut score = 0.0f32;
                    for dimension in 0..dimensions {
                        let query_index = (query_token * heads + head) * dimensions + dimension;
                        let key_index = (key_token * heads + head) * dimensions + dimension;
                        score += queries[query_index] * keys[key_index];
                    }
                    scores.push(score * scale);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator = scores
                    .iter_mut()
                    .map(|score| {
                        *score = (*score - maximum).exp();
                        *score
                    })
                    .sum::<f32>();
                for value_dimension in 0..value_dimensions {
                    let mut result = 0.0f32;
                    for (relative, key_token) in (segment_start..segment_end).enumerate() {
                        let value_index =
                            (key_token * heads + head) * value_dimensions + value_dimension;
                        result += scores[relative] / denominator * values[value_index];
                    }
                    let output_index =
                        (query_token * heads + head) * value_dimensions + value_dimension;
                    output[output_index] = result;
                }
            }
        }
        segment_start = segment_end;
    }
    Ok(output)
}

/// Value source for an attention unit that owns key/value state.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum AttentionValueSource {
    /// Values are produced by an independent projection.
    Projected,
    /// Projected keys are reused as values.
    ReuseKey,
}

/// Typed source and publication policy for attention key/value state.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum AttentionStateSource {
    /// Own projections and retain state only for this attention unit.
    Local {
        /// Value projection topology.
        value: AttentionValueSource,
    },
    /// Own projections and publish state for later consumers.
    Publish {
        /// Value projection topology.
        value: AttentionValueSource,
    },
    /// Consume state published by another unit or supplied externally.
    Shared,
}

impl AttentionStateSource {
    /// Returns whether the attention unit owns projections and mutable state.
    pub const fn owns_state(self) -> bool {
        !matches!(self, Self::Shared)
    }

    /// Returns whether the resulting state must be published.
    pub const fn publishes_state(self) -> bool {
        matches!(self, Self::Publish { .. })
    }

    /// Returns the value source for a state-owning unit.
    pub const fn value(self) -> Option<AttentionValueSource> {
        match self {
            Self::Local { value } | Self::Publish { value } => Some(value),
            Self::Shared => None,
        }
    }
}

#[cfg(test)]
mod attention_state_source_tests {
    use super::{AttentionStateSource, AttentionValueSource};

    #[test]
    fn ownership_publication_and_key_as_value_are_independent() {
        let local = AttentionStateSource::Local {
            value: AttentionValueSource::Projected,
        };
        let publisher = AttentionStateSource::Publish {
            value: AttentionValueSource::ReuseKey,
        };
        assert!(local.owns_state());
        assert!(!local.publishes_state());
        assert_eq!(local.value(), Some(AttentionValueSource::Projected));
        assert!(publisher.owns_state());
        assert!(publisher.publishes_state());
        assert_eq!(publisher.value(), Some(AttentionValueSource::ReuseKey));
        assert!(!AttentionStateSource::Shared.owns_state());
        assert_eq!(AttentionStateSource::Shared.value(), None);
    }
}

#[cfg(test)]
mod recurrent_encoder_contract_tests {
    use super::{
        reference_expand_heads, reference_segmented_attention, validate_segment_lengths,
        NormalizationConstructionSpec, NormalizationScale,
    };

    #[test]
    fn normalization_construction_rejects_invalid_geometry_and_scalars() {
        assert!(NormalizationConstructionSpec {
            dimensions: 8,
            epsilon: 1e-6,
            scale: NormalizationScale::Unit,
        }
        .validate()
        .is_ok());
        assert!(NormalizationConstructionSpec {
            dimensions: 0,
            epsilon: 1e-6,
            scale: NormalizationScale::Unit,
        }
        .validate()
        .is_err());
        assert!(NormalizationConstructionSpec {
            dimensions: 8,
            epsilon: f32::NAN,
            scale: NormalizationScale::Unit,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn head_expansion_reference_preserves_grouped_row_order() {
        let (values, shape) =
            reference_expand_heads(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2], 1, 4).unwrap();
        assert_eq!(shape, vec![1, 4, 2]);
        assert_eq!(values, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
        assert!(reference_expand_heads(&[1.0, 2.0], &[1, 2], 1, 3).is_err());
    }

    #[test]
    fn segmented_attention_reference_is_independent_per_contiguous_segment() {
        let output = reference_segmented_attention(
            3,
            1,
            1,
            1,
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[2.0, 4.0, 9.0],
            &[2, 1],
            1.0,
        )
        .unwrap();
        assert_eq!(output, vec![3.0, 3.0, 9.0]);
        assert!(validate_segment_lengths(3, &[]).is_err());
        assert!(validate_segment_lengths(3, &[2, 0, 1]).is_err());
        assert!(validate_segment_lengths(3, &[2]).is_err());
        assert!(validate_segment_lengths(3, &[2, 2]).is_err());
        assert!(validate_segment_lengths(i32::MAX, &[i32::MAX, 1]).is_err());
    }
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

/// Causal attention with a learned relative-position profile.
///
/// Architecture code projects per-token relative features into `profiles`;
/// the backend owns position-index gathering and score materialization so no
/// device values cross the host boundary.
#[derive(Debug, Clone, Copy)]
pub struct RelativeAttentionInput<'a, T> {
    /// Normalized queries shaped `[batch, heads, query_tokens, dimensions]`.
    pub queries: &'a T,
    /// Normalized keys shaped `[batch, kv_heads, key_tokens, dimensions]`.
    pub keys: &'a T,
    /// Values shaped `[batch, kv_heads, key_tokens, dimensions]`.
    pub values: &'a T,
    /// Learned profiles shaped `[batch, heads, query_tokens, relative_extent]`.
    pub profiles: &'a T,
    /// Absolute position of the first query token.
    pub query_offset: i32,
    /// Absolute position of the first retained key token.
    pub key_offset: i32,
    /// Optional causal sliding-window width.
    pub window: Option<i32>,
    /// Optional position floor for logarithmic global-attention scaling.
    pub log_scaling_floor: Option<i32>,
    /// Multiplier applied to the logarithmic scale above the floor.
    pub log_scaling_alpha: f32,
}

impl<T: Tensor> RelativeAttentionInput<'_, T> {
    /// Validates exact head, sequence, and relative-profile geometry.
    pub fn validate(&self) -> Result<(), Error> {
        let query = self.queries.shape();
        let key = self.keys.shape();
        let value = self.values.shape();
        let profiles = self.profiles.shape();
        if query.len() != 4
            || key.len() != 4
            || value.len() != 4
            || profiles.len() != 4
            || query[0] != key[0]
            || key != value
            || query[2] != profiles[2]
            || query[0] != profiles[0]
            || query[1] != profiles[1]
            || query[3] != key[3]
            || query[1] % key[1] != 0
            || profiles[3] <= 0
            || self.window.is_some_and(|window| window <= 0)
            || self.log_scaling_floor.is_some_and(|floor| floor <= 0)
            || !self.log_scaling_alpha.is_finite()
        {
            return Err(Error::backend(format!(
                "invalid relative attention geometry q={query:?} k={key:?} v={value:?} profiles={profiles:?} window={:?} floor={:?} alpha={}",
                self.window, self.log_scaling_floor, self.log_scaling_alpha
            )));
        }
        Ok(())
    }
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

/// One rank's validated contiguous vocabulary ownership.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct VocabularyParallelRange {
    /// Complete logical vocabulary size.
    pub global_vocabulary: usize,
    /// Half-open row range materialized by this rank.
    pub local: std::ops::Range<usize>,
}

impl VocabularyParallelRange {
    /// Validates non-empty in-bounds ownership.
    pub fn validate(&self) -> Result<(), Error> {
        if self.global_vocabulary == 0
            || self.local.is_empty()
            || self.local.end > self.global_vocabulary
        {
            return Err(Error::backend(format!(
                "invalid vocabulary-parallel range {:?} of {}",
                self.local, self.global_vocabulary
            )));
        }
        Ok(())
    }

    /// Validates that an operator's declared global row count is exactly this
    /// ownership range's global vocabulary.
    pub fn validate_global_rows(&self, rows: i32) -> Result<(), Error> {
        self.validate()?;
        if usize::try_from(rows).ok() != Some(self.global_vocabulary) {
            return Err(Error::backend(format!(
                "vocabulary-parallel operator declares {rows} rows but ownership covers {}",
                self.global_vocabulary
            )));
        }
        Ok(())
    }
}

/// Token validation and sentinel behavior for one embedding lookup.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EmbeddingLookupPolicy {
    /// Every token must be a non-negative row index in the embedding table.
    Strict,
    /// One negative token value produces an exact zero row instead of indexing
    /// the embedding table. Every other token remains subject to strict lookup.
    ZeroSentinel(i32),
}

impl EmbeddingLookupPolicy {
    /// Validates that the optional sentinel cannot alias an ordinary row.
    pub fn validate(self) -> Result<(), Error> {
        if let Self::ZeroSentinel(sentinel) = self {
            if sentinel >= 0 {
                return Err(Error::backend(format!(
                    "embedding zero sentinel must be negative, got {sentinel}"
                )));
            }
        }
        Ok(())
    }
}

/// One named, positive-width component of a fused projection output.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FusedProjectionSegment {
    name: String,
    width: i32,
}

impl FusedProjectionSegment {
    /// Creates a validated component declaration.
    pub fn new(name: impl Into<String>, width: i32) -> Result<Self, Error> {
        let name = name.into();
        if name.trim().is_empty() || width <= 0 {
            return Err(Error::backend(format!(
                "fused projection segments require a name and positive width, got name={name:?} width={width}"
            )));
        }
        Ok(Self { name, width })
    }

    /// Returns the stable component name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the component width on the final projection axis.
    pub const fn width(&self) -> i32 {
        self.width
    }
}

/// Validated component-major layout of one fused affine projection.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FusedProjectionLayout {
    segments: Vec<FusedProjectionSegment>,
    output_width: i32,
}

impl FusedProjectionLayout {
    /// Validates ordered unique components and checked total width.
    pub fn new(segments: impl IntoIterator<Item = FusedProjectionSegment>) -> Result<Self, Error> {
        let segments = segments.into_iter().collect::<Vec<_>>();
        if segments.is_empty() {
            return Err(Error::backend(
                "fused projection layout must contain at least one segment",
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        let mut output_width = 0i32;
        for segment in &segments {
            if !names.insert(segment.name.clone()) {
                return Err(Error::backend(format!(
                    "fused projection segment {:?} is duplicated",
                    segment.name
                )));
            }
            output_width = output_width.checked_add(segment.width).ok_or_else(|| {
                Error::backend("fused projection output width overflowed signed 32-bit geometry")
            })?;
        }
        Ok(Self {
            segments,
            output_width,
        })
    }

    /// Returns component declarations in physical output order.
    pub fn segments(&self) -> &[FusedProjectionSegment] {
        &self.segments
    }

    /// Returns the checked total output width.
    pub const fn output_width(&self) -> i32 {
        self.output_width
    }

    /// Splits one fused result into component-major final-axis views.
    pub fn split<T: Tensor>(&self, output: &T, context: &T::Context) -> Result<Vec<T>, Error> {
        let actual = output
            .shape()
            .last()
            .copied()
            .ok_or_else(|| Error::backend("fused projection output has no feature axis"))?;
        if actual != self.output_width {
            return Err(Error::backend(format!(
                "fused projection emitted width {actual}, expected {}",
                self.output_width
            )));
        }
        let mut start = 0i32;
        let mut indexes = vec![Index::Full; output.shape().len()];
        self.segments
            .iter()
            .map(|segment| {
                let end = start + segment.width;
                let last = indexes.len() - 1;
                indexes[last] = Index::Range(start, end);
                let selected = output.index(&indexes, context);
                start = end;
                selected
            })
            .collect()
    }
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

/// Parameterization policy for a reusable RMS normalization operator.
///
/// Architectures select the semantic scale form while the backend retains the
/// tensor implementation. In particular, learned-offset scales are evaluated
/// as `offset + weight`; the offset is not folded into checkpoint storage.
#[derive(Debug, Clone)]
pub enum NormalizationScale {
    /// Ordinary learned multiplicative scale.
    Learned(ParameterSpec),
    /// Learned scale offset by a fixed scalar at execution time.
    LearnedOffset {
        /// Stable checkpoint slot containing the learned offset tensor.
        weight: ParameterSpec,
        /// Fixed scalar added to every learned scale value.
        offset: f32,
    },
    /// RMS normalization without a learned scale.
    Unit,
}

/// Complete construction policy for an RMS normalization operator.
#[derive(Debug, Clone)]
pub struct NormalizationConstructionSpec {
    /// Normalized feature count.
    pub dimensions: i32,
    /// Numerical stability epsilon.
    pub epsilon: f32,
    /// Learned, learned-offset, or weightless scale policy.
    pub scale: NormalizationScale,
}

impl NormalizationConstructionSpec {
    /// Validates feature geometry and fixed scalar policy.
    pub fn validate(&self) -> Result<(), Error> {
        let offset = match &self.scale {
            NormalizationScale::LearnedOffset { offset, .. } => Some(*offset),
            NormalizationScale::Learned(_) | NormalizationScale::Unit => None,
        };
        if self.dimensions <= 0
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || offset.is_some_and(|offset| !offset.is_finite())
        {
            return Err(Error::backend(format!(
                "invalid RMS normalization construction: dimensions={} epsilon={} offset={offset:?}",
                self.dimensions, self.epsilon
            )));
        }
        Ok(())
    }
}

impl From<NormalizationSpec> for NormalizationConstructionSpec {
    fn from(spec: NormalizationSpec) -> Self {
        Self {
            dimensions: spec.dimensions,
            epsilon: spec.epsilon,
            scale: NormalizationScale::Learned(spec.weight),
        }
    }
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
    /// Looks up token embeddings under an explicit validation/sentinel policy.
    fn lookup(
        &mut self,
        input: &T,
        policy: EmbeddingLookupPolicy,
        context: &T::Context,
    ) -> Result<T, Error> {
        policy.validate()?;
        match policy {
            EmbeddingLookupPolicy::Strict => self.forward(input, context),
            EmbeddingLookupPolicy::ZeroSentinel(sentinel) => Err(Error::backend(format!(
                "embedding backend does not implement zero sentinel {sentinel}"
            ))),
        }
    }
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
    /// Select the largest raw logits, then softmax only the selected routes.
    SelectedSoftmax,
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
    /// Optional ordinary projection bias. This contributes to router logits
    /// before scoring, selection, and selected-route normalization.
    pub bias: Option<ParameterSpec>,
    /// Optional correction bias used only to choose expert IDs. Gathered
    /// route scores remain unbiased.
    pub correction_bias: Option<ParameterSpec>,
    /// Optional learned scaling applied after weightless RMS normalization of
    /// the router input.
    pub input_transform: Option<RouterInputTransformSpec>,
    /// Optional learned per-expert multiplier gathered after route selection.
    pub route_scale: Option<ParameterSpec>,
    /// Optional physical encoding of the router projection.
    pub quantization: Option<WeightQuantization>,
    /// Architecture-selected scoring and selection semantics.
    pub routing: TopKRoutingSpec,
}

/// Learned normalization and scale applied before a router projection.
#[derive(Debug, Clone)]
pub struct RouterInputTransformSpec {
    /// Weightless RMS-normalization epsilon.
    pub epsilon: f32,
    /// Learned feature-wise multiplier.
    pub scale: ParameterSpec,
    /// Whether to additionally multiply by `1 / sqrt(input_dimensions)`.
    pub inverse_sqrt_dimensions: bool,
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
        if self
            .input_transform
            .as_ref()
            .is_some_and(|transform| !transform.epsilon.is_finite() || transform.epsilon < 0.0)
        {
            return Err(Error::backend(
                "router input RMS epsilon must be finite and nonnegative",
            ));
        }
        if self
            .bias
            .as_ref()
            .zip(self.correction_bias.as_ref())
            .is_some_and(|(bias, correction_bias)| bias.id == correction_bias.id)
        {
            return Err(Error::backend(
                "router projection bias and correction bias require distinct parameter identities",
            ));
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

/// Joint sigmoid routing request with selected routed experts and always-on
/// shared experts normalized in one probability distribution.
#[derive(Debug, Clone, Copy)]
pub struct JointExpertRoutingInput<'a, T> {
    /// Hidden states shaped `[..., hidden]`.
    pub hidden: &'a T,
    /// Projection shaped `[routed_experts + shared_experts, hidden]`.
    pub weight: &'a T,
    /// Correction bias used only for routed top-k selection.
    pub correction_bias: &'a T,
    /// Learned scalar multiplier applied to all final route weights.
    pub global_scale: &'a T,
    /// Number of selectable routed experts.
    pub routed_experts: i32,
    /// Number of always-on shared experts.
    pub shared_experts: i32,
    /// Number of routed experts selected per token.
    pub top_k: i32,
    /// Fixed multiplier applied before the learned global scale.
    pub route_scale: f32,
}

impl<T: Tensor> JointExpertRoutingInput<'_, T> {
    /// Validates exact projection, bias, and scalar geometry.
    pub fn validate(&self) -> Result<(), Error> {
        let hidden = self.hidden.shape();
        let weight = self.weight.shape();
        let bias = self.correction_bias.shape();
        let scale = self.global_scale.shape();
        let hidden_width = hidden.last().copied().unwrap_or(0);
        if hidden.len() < 2
            || weight != [self.routed_experts + self.shared_experts, hidden_width]
            || bias != [self.routed_experts]
            || scale != [1]
            || self.routed_experts <= 0
            || self.shared_experts <= 0
            || self.top_k <= 0
            || self.top_k > self.routed_experts
            || !self.route_scale.is_finite()
            || self.route_scale <= 0.0
        {
            return Err(Error::backend(format!(
                "invalid joint expert routing geometry hidden={hidden:?} weight={weight:?} bias={bias:?} scale={scale:?} routed={} shared={} top_k={} route_scale={}",
                self.routed_experts,
                self.shared_experts,
                self.top_k,
                self.route_scale
            )));
        }
        Ok(())
    }
}

/// Backend-native result of joint routed and shared expert weighting.
#[derive(Debug, Clone)]
pub struct JointExpertRoutingResult<T> {
    /// Selected routed expert IDs shaped `[tokens, top_k]`; integer dtype is
    /// backend-defined and accepted by the corresponding expert operator.
    pub routed_ids: T,
    /// Routed expert weights shaped `[tokens, top_k]`.
    pub routed_weights: T,
    /// Always-on shared expert weights shaped `[tokens, shared_experts]`.
    pub shared_weights: T,
}

/// Activation applied to the gate branch of a routed gated product.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum GatedProductActivation {
    /// `gate * sigmoid(sigmoid_multiplier * gate)`.
    Silu,
    /// Approximate Gaussian error linear unit.
    GeluApproximate,
}

/// Validated equation policy for a gated product expert.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GatedProductPolicy {
    activation: GatedProductActivation,
    gate_upper_bound: Option<f32>,
    up_absolute_bound: Option<f32>,
    sigmoid_multiplier: f32,
    up_offset: f32,
}

impl GatedProductPolicy {
    /// Creates a validated gated-product equation.
    pub fn new(
        activation: GatedProductActivation,
        gate_upper_bound: Option<f32>,
        up_absolute_bound: Option<f32>,
        sigmoid_multiplier: f32,
        up_offset: f32,
    ) -> Result<Self, Error> {
        let policy = Self {
            activation,
            gate_upper_bound,
            up_absolute_bound,
            sigmoid_multiplier,
            up_offset,
        };
        policy.validate()?;
        Ok(policy)
    }

    /// Ordinary unbounded SiLU gating.
    pub const fn ordinary_silu() -> Self {
        Self {
            activation: GatedProductActivation::Silu,
            gate_upper_bound: None,
            up_absolute_bound: None,
            sigmoid_multiplier: 1.0,
            up_offset: 0.0,
        }
    }

    /// Ordinary unbounded approximate-GELU gating.
    pub const fn ordinary_gelu_approximate() -> Self {
        Self {
            activation: GatedProductActivation::GeluApproximate,
            ..Self::ordinary_silu()
        }
    }

    /// Creates ordinary SiLU gating with the same positive gate and up bound.
    pub fn bounded_silu(bound: f32) -> Result<Self, Error> {
        Self::new(
            GatedProductActivation::Silu,
            Some(bound),
            Some(bound),
            1.0,
            0.0,
        )
    }

    /// Validates finite scalars and positive bounds/multiplier.
    pub fn validate(self) -> Result<(), Error> {
        if self
            .gate_upper_bound
            .is_some_and(|bound| !bound.is_finite() || bound <= 0.0)
            || self
                .up_absolute_bound
                .is_some_and(|bound| !bound.is_finite() || bound <= 0.0)
            || !self.sigmoid_multiplier.is_finite()
            || self.sigmoid_multiplier <= 0.0
            || !self.up_offset.is_finite()
        {
            return Err(Error::backend(format!(
                "invalid gated-product policy: {self:?}"
            )));
        }
        Ok(())
    }

    /// Gate activation.
    pub const fn activation(self) -> GatedProductActivation {
        self.activation
    }

    /// Optional upper bound applied to the gate branch before activation.
    pub const fn gate_upper_bound(self) -> Option<f32> {
        self.gate_upper_bound
    }

    /// Optional symmetric absolute bound applied to the up branch.
    pub const fn up_absolute_bound(self) -> Option<f32> {
        self.up_absolute_bound
    }

    /// Multiplier inside the sigmoid for SiLU gating.
    pub const fn sigmoid_multiplier(self) -> f32 {
        self.sigmoid_multiplier
    }

    /// Offset added to the up branch after optional clipping.
    pub const fn up_offset(self) -> f32 {
        self.up_offset
    }
}

impl Default for GatedProductPolicy {
    fn default() -> Self {
        Self::ordinary_silu()
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

/// Parameter identities for one gated-product expert or one packed expert axis.
#[derive(Debug, Clone)]
pub struct GatedProductExpertParameters {
    /// Gating projection weight.
    pub gate: ExpertProjectionSpec,
    /// Up projection weight.
    pub up: ExpertProjectionSpec,
    /// Down projection weight.
    pub down: ExpertProjectionSpec,
}

/// One expert projection identity and optional physical encoding.
#[derive(Debug, Clone)]
pub struct ExpertProjectionSpec {
    /// Stable logical parameter identity.
    pub weight: ParameterSpec,
    /// Optional ordinary per-output projection bias.
    pub bias: Option<ParameterSpec>,
    /// Complete physical checkpoint encoding.
    pub format: LinearFormat,
}

/// Logical parameter layout for a gated-product expert bank.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)] // Packed layout stays inline on the construction hot path.
pub enum GatedProductExpertLayout {
    /// Component-major fused gate-then-up and down tensors whose leading axis
    /// indexes experts.
    Packed {
        /// Concatenated gate/up projection.
        gate_up: ExpertProjectionSpec,
        /// Down projection.
        down: ExpertProjectionSpec,
    },
    /// Independently materialized expert parameter triples in expert-ID order.
    Independent(Vec<GatedProductExpertParameters>),
}

/// Complete architecture-owned construction specification for routed gated-product experts.
#[derive(Debug, Clone)]
pub struct GatedProductExpertBankSpec {
    /// Number of routed experts.
    pub expert_count: i32,
    /// Input hidden width.
    pub input_dimensions: i32,
    /// Per-expert intermediate width.
    pub intermediate_dimensions: i32,
    /// Output hidden width.
    pub output_dimensions: i32,
    /// Exact gate activation, bounds, multiplier, and up offset.
    pub policy: GatedProductPolicy,
    /// Stable logical parameter identities and storage organization.
    pub layout: GatedProductExpertLayout,
}

impl GatedProductExpertBankSpec {
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
                    "gated-product expert-bank {name} must be positive, got {value}"
                )));
            }
        }
        self.policy.validate()?;
        if let GatedProductExpertLayout::Independent(experts) = &self.layout {
            let expected = usize::try_from(self.expert_count).map_err(Error::backend)?;
            if experts.len() != expected {
                return Err(Error::backend(format!(
                    "independent gated-product bank has {} experts, expected {expected}",
                    experts.len()
                )));
            }
        }
        let projections = match &self.layout {
            GatedProductExpertLayout::Packed { gate_up, down } => vec![gate_up, down],
            GatedProductExpertLayout::Independent(experts) => experts
                .iter()
                .flat_map(|expert| [&expert.gate, &expert.up, &expert.down])
                .collect(),
        };
        let mut identities = std::collections::BTreeSet::new();
        for projection in projections {
            for identity in std::iter::once(&projection.weight.id)
                .chain(projection.bias.as_ref().map(|bias| &bias.id))
            {
                if !identities.insert(identity) {
                    return Err(Error::backend(format!(
                        "gated-product expert parameter identity {identity} is duplicated"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Rank-local routed expert output split around the tensor-parallel reduction.
#[derive(Debug, Clone)]
pub struct TensorParallelExpertOutput<T> {
    /// Rank-local projection contribution to all-sum.
    pub reducible: T,
    /// Replicated route-weighted down bias added once after all-sum.
    pub post_reduce: Option<T>,
}

/// Statically dispatched routed gated-product expert bank.
pub trait GatedProductExpertBankOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Returns the architecture-owned construction specification used by this bank.
    ///
    /// Runtime providers use this metadata when they substitute cached or remote
    /// parameters for the resident bank, so every execution path retains the
    /// same geometry, encoding, bias, and activation policy.
    fn spec(&self) -> &GatedProductExpertBankSpec;

    /// Executes selected experts and combines their outputs by route weight.
    fn forward_routed(
        &mut self,
        input: &T,
        routes: &RoutingResult<T>,
        context: &T::Context,
    ) -> Result<T, Error>;

    /// Executes a rank-local tensor-parallel partial and separates replicated
    /// route-weighted down bias for one literal post-reduction addition.
    fn forward_routed_tensor_parallel(
        &mut self,
        input: &T,
        routes: &RoutingResult<T>,
        partitions: usize,
        context: &T::Context,
    ) -> Result<TensorParallelExpertOutput<T>, Error> {
        if partitions == 1 {
            return self
                .forward_routed(input, routes, context)
                .map(|reducible| TensorParallelExpertOutput {
                    reducible,
                    post_reduce: None,
                });
        }
        Err(Error::backend(
            "tensor-parallel gated-product experts are not implemented by this backend",
        ))
    }
}

/// Complete construction specification for packed routed ReLU-squared experts.
#[derive(Debug, Clone)]
pub struct Relu2ExpertBankSpec {
    /// Number of routed experts.
    pub expert_count: i32,
    /// Input and output hidden width.
    pub hidden_dimensions: i32,
    /// Per-expert intermediate width.
    pub intermediate_dimensions: i32,
    /// Packed up-projection identity and physical format.
    pub up: ExpertProjectionSpec,
    /// Packed down-projection identity and physical format.
    pub down: ExpertProjectionSpec,
}

impl Relu2ExpertBankSpec {
    /// Validates positive geometry.
    pub fn validate(&self) -> Result<(), Error> {
        if self.expert_count <= 0
            || self.hidden_dimensions <= 0
            || self.intermediate_dimensions <= 0
        {
            return Err(Error::backend("invalid ReLU2 expert-bank geometry"));
        }
        Ok(())
    }
}

/// Statically dispatched routed ReLU-squared expert bank.
pub trait Relu2ExpertBankOperator<T: Tensor>: Clone + Debug + Parameterized<T> {
    /// Executes selected experts and combines their outputs by route weight.
    fn forward_routed(
        &mut self,
        input: &T,
        routes: &RoutingResult<T>,
        context: &T::Context,
    ) -> Result<T, Error>;

    /// Executes a rank-local tensor-parallel partial and separates replicated
    /// route-weighted down bias for one literal post-reduction addition.
    fn forward_routed_tensor_parallel(
        &mut self,
        input: &T,
        routes: &RoutingResult<T>,
        partitions: usize,
        context: &T::Context,
    ) -> Result<TensorParallelExpertOutput<T>, Error> {
        if partitions == 1 {
            return self
                .forward_routed(input, routes, context)
                .map(|reducible| TensorParallelExpertOutput {
                    reducible,
                    post_reduce: None,
                });
        }
        Err(Error::backend(
            "tensor-parallel ReLU2 experts are not implemented by this backend",
        ))
    }
}

/// Neural backend extension for routed expert architectures.
pub trait RoutedNeuralBackend: NeuralBackend {
    /// Concrete top-k router.
    type Router: RoutingOperator<Self::Tensor>;
    /// Concrete packed or independently materialized expert bank.
    type GatedProductExpertBank: GatedProductExpertBankOperator<Self::Tensor>;
    /// Concrete packed routed ReLU-squared expert bank.
    type Relu2ExpertBank: Relu2ExpertBankOperator<Self::Tensor>;

    /// Builds a router with architecture-selected top-k semantics.
    fn top_k_router(
        spec: TopKRouterSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Router, Error>;

    /// Builds a routed gated-product expert bank.
    fn gated_product_expert_bank(
        spec: GatedProductExpertBankSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::GatedProductExpertBank, Error>;

    /// Builds a routed ReLU-squared expert bank.
    fn relu2_expert_bank(
        spec: Relu2ExpertBankSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Relu2ExpertBank, Error>;
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

/// One backend-native scaled-dot-product attention request.
///
/// Projected queries, keys, and values remain owned backend tensors so cached,
/// uncached, paged, and sliding implementations can consume the same request
/// without cloning or host materialization. Masks and learned per-query-head
/// sink logits are borrowed architecture state.
#[derive(Debug)]
pub struct AttentionRequest<'a, T> {
    /// Queries shaped `[batch, query_heads, query_tokens, head_dimensions]`.
    pub queries: T,
    /// Keys shaped `[batch, key_value_heads, key_tokens, head_dimensions]`.
    pub keys: T,
    /// Values shaped `[batch, key_value_heads, key_tokens, value_dimensions]`.
    pub values: T,
    /// Positive finite score scale.
    pub scale: f32,
    /// Optional additive or boolean attention mask.
    pub mask: Option<&'a T>,
    /// Optional learned sink logit for every query head.
    pub sinks: Option<&'a T>,
}

impl<T: Tensor> AttentionRequest<'_, T> {
    /// Validates common grouped-query and sink geometry without inspecting values.
    pub fn validate(&self) -> Result<(), Error> {
        let queries = self.queries.shape();
        let keys = self.keys.shape();
        let values = self.values.shape();
        if queries.len() != 4
            || keys.len() != 4
            || values.len() != 4
            || queries[0] != keys[0]
            || keys[..3] != values[..3]
            || queries[3] != keys[3]
            || queries[1] <= 0
            || keys[1] <= 0
            || queries[1] % keys[1] != 0
            || queries[2] <= 0
            || keys[2] <= 0
            || values[3] <= 0
            || !self.scale.is_finite()
            || self.scale <= 0.0
        {
            return Err(Error::backend(format!(
                "invalid attention request geometry queries={queries:?} keys={keys:?} values={values:?} scale={}",
                self.scale
            )));
        }
        if let Some(sinks) = self.sinks {
            if sinks.shape() != [queries[1]] {
                return Err(Error::backend(format!(
                    "attention sinks require shape [{}], got {:?}",
                    queries[1],
                    sinks.shape()
                )));
            }
        }
        Ok(())
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
    fn attention(
        &mut self,
        request: AttentionRequest<'_, T>,
        context: &T::Context,
    ) -> Result<T, Error>;
}

/// Attention cache with a fixed set of bounded causal-convolution histories.
///
/// Slot identity is architecture policy; storage, persistence, and device
/// realization remain runtime/backend concerns.
pub trait AuxiliaryConvolutionState<T: Tensor>: AttentionCache<T> {
    /// Borrows one bounded convolution-history slot.
    fn convolution_state(&mut self, slot: u32) -> Result<&mut Option<T>, Error>;
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
pub trait CompressedAttentionCache<T: Tensor>: Debug {
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
pub trait PoolingAttentionCache<T: Tensor>: Debug {
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

/// Explicit forward-pass operators a backend promises to execute.
///
/// These capabilities cover optional methods on [`NeuralBackend`] whose
/// default implementations fail closed. Architectures validate their required
/// set while constructing modules, before parameters are loaded or a forward
/// pass can begin.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct NeuralOperatorCapabilities(u64);

impl NeuralOperatorCapabilities {
    /// No optional forward operators.
    pub const NONE: Self = Self(0);
    /// Tanh-approximated GELU.
    pub const GELU_APPROXIMATE: Self = Self(1 << 0);
    /// Logistic sigmoid.
    pub const SIGMOID: Self = Self(1 << 1);
    /// Softplus.
    pub const SOFTPLUS: Self = Self(1 << 2);
    /// Natural exponential.
    pub const EXP: Self = Self(1 << 3);
    /// Gated grouped RMS normalization.
    pub const GATED_GROUP_RMS_NORM: Self = Self(1 << 4);
    /// L2 normalization.
    pub const L2_NORMALIZE: Self = Self(1 << 5);
    /// SiLU-gated grouped RMS normalization.
    pub const SILU_GATED_GROUP_RMS_NORM: Self = Self(1 << 6);
    /// Segmented attention.
    pub const SEGMENTED_ATTENTION: Self = Self(1 << 7);
    /// Gated-delta recurrent scan.
    pub const GATED_DELTA_SCAN: Self = Self(1 << 8);
    /// Selective state-space scan.
    pub const SELECTIVE_STATE_SPACE_SCAN: Self = Self(1 << 9);
    /// Indexed sparse attention.
    pub const INDEXED_ATTENTION: Self = Self(1 << 10);
    /// Dense pooled attention.
    pub const POOLED_ATTENTION: Self = Self(1 << 11);
    /// Pooled-position selection.
    pub const POOLED_POSITION_SELECTION: Self = Self(1 << 12);
    /// Pooled-mask gathering.
    pub const POOLED_MASK_GATHER: Self = Self(1 << 13);
    /// Attention with learned sink logits.
    pub const ATTENTION_SINKS: Self = Self(1 << 14);
    /// Learned relative-profile attention.
    pub const RELATIVE_ATTENTION: Self = Self(1 << 15);
    /// Joint routed/shared expert selection.
    pub const JOINT_EXPERT_ROUTING: Self = Self(1 << 16);
    /// Weightless RMS normalization.
    pub const RMS_NORM_WITHOUT_WEIGHT: Self = Self(1 << 17);
    /// Grouped block-diagonal linear projection.
    pub const GROUPED_LINEAR: Self = Self(1 << 18);
    /// Tensor-parallel sum reduction.
    pub const SUM_PARALLEL: Self = Self(1 << 19);
    /// Every currently declared optional forward operator.
    pub const ALL: Self = Self((1 << 20) - 1);

    /// Returns the union of two capability sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns whether this set contains every required capability.
    pub const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    fn missing_names(self, required: Self) -> Vec<&'static str> {
        const NAMES: &[(NeuralOperatorCapabilities, &str)] = &[
            (
                NeuralOperatorCapabilities::GELU_APPROXIMATE,
                "gelu_approximate",
            ),
            (NeuralOperatorCapabilities::SIGMOID, "sigmoid"),
            (NeuralOperatorCapabilities::SOFTPLUS, "softplus"),
            (NeuralOperatorCapabilities::EXP, "exp"),
            (
                NeuralOperatorCapabilities::GATED_GROUP_RMS_NORM,
                "gated_group_rms_norm",
            ),
            (NeuralOperatorCapabilities::L2_NORMALIZE, "l2_normalize"),
            (
                NeuralOperatorCapabilities::SILU_GATED_GROUP_RMS_NORM,
                "silu_gated_group_rms_norm",
            ),
            (
                NeuralOperatorCapabilities::SEGMENTED_ATTENTION,
                "segmented_attention",
            ),
            (
                NeuralOperatorCapabilities::GATED_DELTA_SCAN,
                "gated_delta_scan",
            ),
            (
                NeuralOperatorCapabilities::SELECTIVE_STATE_SPACE_SCAN,
                "selective_state_space_scan",
            ),
            (
                NeuralOperatorCapabilities::INDEXED_ATTENTION,
                "indexed_attention",
            ),
            (
                NeuralOperatorCapabilities::POOLED_ATTENTION,
                "pooled_attention",
            ),
            (
                NeuralOperatorCapabilities::POOLED_POSITION_SELECTION,
                "select_pooled_positions",
            ),
            (
                NeuralOperatorCapabilities::POOLED_MASK_GATHER,
                "gather_pooled_mask",
            ),
            (
                NeuralOperatorCapabilities::ATTENTION_SINKS,
                "attention_sinks",
            ),
            (
                NeuralOperatorCapabilities::RELATIVE_ATTENTION,
                "relative_attention",
            ),
            (
                NeuralOperatorCapabilities::JOINT_EXPERT_ROUTING,
                "joint_expert_routing",
            ),
            (
                NeuralOperatorCapabilities::RMS_NORM_WITHOUT_WEIGHT,
                "rms_norm_without_weight",
            ),
            (NeuralOperatorCapabilities::GROUPED_LINEAR, "grouped_linear"),
            (NeuralOperatorCapabilities::SUM_PARALLEL, "sum_parallel"),
        ];
        NAMES
            .iter()
            .filter_map(|(capability, name)| {
                (required.contains(*capability) && !self.contains(*capability)).then_some(*name)
            })
            .collect()
    }
}

/// General neural-operator family selected by a shared architecture.
///
/// Associated concrete types make calls statically dispatched. Implementations
/// retain ownership of tensor storage, fusion, quantization, and collectives.
pub trait NeuralBackend: Sized + 'static {
    /// Optional forward operators explicitly supported by this backend.
    const OPERATOR_CAPABILITIES: NeuralOperatorCapabilities = NeuralOperatorCapabilities::NONE;

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

    /// Rejects an architecture before module construction when an optional
    /// forward operator is unavailable.
    fn require_operator_capabilities(
        architecture: &'static str,
        required: NeuralOperatorCapabilities,
    ) -> Result<(), Error> {
        let available = Self::OPERATOR_CAPABILITIES;
        if available.contains(required) {
            return Ok(());
        }
        Err(Error::backend(format!(
            "{architecture} requires unsupported backend operators: {}",
            available.missing_names(required).join(", ")
        )))
    }

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
    /// Builds one rank-local vocabulary embedding under validated ownership.
    fn vocabulary_parallel_embedding(
        spec: EmbeddingSpec,
        range: VocabularyParallelRange,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Embedding, Error> {
        let _ = (spec, range, context);
        Err(Error::backend(
            "backend does not implement vocabulary-parallel embeddings",
        ))
    }
    /// Builds one rank-local vocabulary output projection.
    fn vocabulary_parallel_linear(
        spec: LinearSpec,
        range: VocabularyParallelRange,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Linear, Error> {
        let _ = (spec, range, context);
        Err(Error::backend(
            "backend does not implement vocabulary-parallel projections",
        ))
    }
    /// Looks up global token IDs and sums this rank's local contribution.
    fn vocabulary_parallel_lookup(
        _embedding: &mut Self::Embedding,
        _input: &Self::Tensor,
        _policy: EmbeddingLookupPolicy,
        _parallel: &Self::ParallelContext,
        _context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        Err(Error::backend(
            "backend does not implement vocabulary-parallel lookup",
        ))
    }
    /// Projects to local vocabulary rows and gathers complete logits.
    fn vocabulary_parallel_project(
        _linear: &mut Self::Linear,
        _input: &Self::Tensor,
        _parallel: &Self::ParallelContext,
        _context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        Err(Error::backend(
            "backend does not implement vocabulary-parallel projection",
        ))
    }
    /// Projects through a rank-local vocabulary embedding and gathers complete logits.
    ///
    /// This is the tied-output counterpart of [`Self::vocabulary_parallel_project`]:
    /// it preserves the embedding operator and its physical parameter storage while
    /// using the same vocabulary ownership established at construction time.
    fn vocabulary_parallel_embedding_project(
        _embedding: &mut Self::Embedding,
        _input: &Self::Tensor,
        _parallel: &Self::ParallelContext,
        _context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        Err(Error::backend(
            "backend does not implement tied vocabulary-parallel projection",
        ))
    }
    /// Builds one RMS normalization operator.
    fn rms_norm(
        spec: NormalizationSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Normalization, Error>;
    /// Builds an RMS normalization with an explicit scale policy.
    ///
    /// Backends predating the generalized construction surface continue to
    /// support ordinary learned scales through `rms_norm`; other policies fail
    /// closed until implemented by that backend.
    fn normalization(
        spec: NormalizationConstructionSpec,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Normalization, Error> {
        spec.validate()?;
        match spec.scale {
            NormalizationScale::Learned(weight) => Self::rms_norm(
                NormalizationSpec {
                    dimensions: spec.dimensions,
                    epsilon: spec.epsilon,
                    weight,
                },
                context,
            ),
            NormalizationScale::LearnedOffset { .. } => Err(Error::backend(
                "learned-offset RMS normalization is not implemented by this backend",
            )),
            NormalizationScale::Unit => Err(Error::backend(
                "weightless RMS normalization construction is not implemented by this backend",
            )),
        }
    }
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
    /// Applies the tanh-approximated GELU used by some encoder MLPs.
    fn gelu_approximate(
        input: Self::Tensor,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (input, context);
        Err(Error::backend(
            "approximate GELU is not implemented by this backend",
        ))
    }
    /// Applies the logistic sigmoid elementwise.
    fn sigmoid(
        input: Self::Tensor,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (input, context);
        Err(Error::backend("sigmoid is not implemented by this backend"))
    }
    /// Applies softplus elementwise.
    fn softplus(
        input: Self::Tensor,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (input, context);
        Err(Error::backend(
            "softplus is not implemented by this backend",
        ))
    }
    /// Applies the natural exponential elementwise.
    fn exp(
        input: Self::Tensor,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (input, context);
        Err(Error::backend(
            "exponential is not implemented by this backend",
        ))
    }
    /// Applies SiLU gating followed by grouped RMS normalization and scale.
    fn gated_group_rms_norm(
        input: &Self::Tensor,
        gate: &Self::Tensor,
        weight: &Self::Tensor,
        groups: i32,
        epsilon: f32,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (input, gate, weight, groups, epsilon, context);
        Err(Error::backend(
            "gated grouped RMS normalization is not implemented by this backend",
        ))
    }
    /// Applies L2 normalization over the final axis.
    fn l2_normalize(
        input: &Self::Tensor,
        epsilon: f32,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (input, epsilon, context);
        Err(Error::backend(
            "L2 normalization is not implemented by this backend",
        ))
    }
    /// Applies grouped RMS normalization to `input`, multiplies by a learned
    /// scale, and then modulates the result by `silu(gate)`.
    fn silu_gated_group_rms_norm(
        input: &Self::Tensor,
        gate: &Self::Tensor,
        weight: &Self::Tensor,
        groups: i32,
        epsilon: f32,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (input, gate, weight, groups, epsilon, context);
        Err(Error::backend(
            "SiLU-gated grouped RMS normalization is not implemented by this backend",
        ))
    }
    /// Repeats a validated head axis using backend-native reshape and
    /// broadcast operations.
    fn expand_heads(
        input: &Self::Tensor,
        expansion: HeadExpansion,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        expansion.validate(input)?;
        if expansion.source_heads == expansion.target_heads {
            return Ok(input.clone());
        }
        let mut expanded_shape = input.shape().to_vec();
        expanded_shape.insert(expansion.axis + 1, 1);
        let expanded = input.reshape(&expanded_shape, context)?;
        expanded_shape[expansion.axis + 1] = expansion.repeats();
        let expanded = expanded.broadcast_to(&expanded_shape, context)?;
        expanded_shape[expansion.axis] = expansion.target_heads;
        expanded_shape.remove(expansion.axis + 1);
        expanded.reshape(&expanded_shape, context)
    }
    /// Runs unmasked attention independently over contiguous validated
    /// segments without exposing backend slicing to architecture code.
    fn segmented_attention(
        input: SegmentedAttentionInput<'_, Self::Tensor>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        input.validate()?;
        let _ = context;
        Err(Error::backend(
            "segmented attention is not implemented by this backend",
        ))
    }
    /// Adds a residual branch, optionally retaining the accumulator in FP32.
    fn add_residual(
        residual: &Self::Tensor,
        branch: &Self::Tensor,
        fp32: bool,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = fp32;
        residual.add(branch, context)
    }
    /// Executes a gated-delta recurrent scan over projected head-major values.
    ///
    /// Inputs are `[batch, sequence, heads, dimensions]` except `beta`, which
    /// is `[batch, sequence, heads]`. The optional initial and returned state
    /// are `[batch, heads, dimensions, dimensions]` in FP32 semantic storage.
    fn gated_delta_scan(
        input: GatedDeltaScanInput<'_, Self::Tensor>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<GatedDeltaScanOutput<Self::Tensor>, Error> {
        let _ = (input, context);
        Err(Error::backend(
            "gated-delta scan is not implemented by this backend",
        ))
    }
    /// Executes a grouped selective state-space recurrence in FP32 state.
    fn selective_state_space_scan(
        input: SelectiveStateSpaceScanInput<'_, Self::Tensor>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<SelectiveStateSpaceScanOutput<Self::Tensor>, Error> {
        let _ = (input, context);
        Err(Error::backend(
            "selective state-space scan is not implemented by this backend",
        ))
    }
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
    fn attention_with_sinks(
        request: AttentionRequest<'_, Self::Tensor>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        request.validate()?;
        if request.sinks.is_some() {
            return Err(Error::backend(
                "attention sinks are not implemented by this backend",
            ));
        }
        Self::attention(
            request.queries,
            request.keys,
            request.values,
            request.scale,
            request.mask,
            context,
        )
    }
    /// Runs causal sliding-window prefill attention with optional learned sinks.
    fn sliding_window_attention_with_sinks(
        request: AttentionRequest<'_, Self::Tensor>,
        window: i32,
        position_offset: i32,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        request.validate()?;
        if request.sinks.is_some() {
            return Err(Error::backend(
                "sliding-window attention sinks are not implemented by this backend",
            ));
        }
        Self::sliding_window_attention(
            request.queries,
            request.keys,
            request.values,
            request.scale,
            window,
            position_offset,
            context,
        )
    }
    /// Runs causal attention with caller-projected learned relative profiles.
    fn relative_attention(
        input: RelativeAttentionInput<'_, Self::Tensor>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (input, context);
        Err(Error::backend(
            "relative-profile attention is not implemented by this backend",
        ))
    }
    /// Selects routed experts and jointly normalizes them with always-on shared
    /// experts without materializing indices or logits on the host.
    fn joint_expert_routing(
        input: JointExpertRoutingInput<'_, Self::Tensor>,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<JointExpertRoutingResult<Self::Tensor>, Error> {
        let _ = (input, context);
        Err(Error::backend(
            "joint routed/shared expert weighting is not implemented by this backend",
        ))
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
    /// Applies RMS normalization with a caller-owned learned scale.
    ///
    /// The default keeps the operation portable by composing weightless
    /// normalization and multiplication. Backends may override it with a
    /// fused kernel when their tensor runtime provides one.
    fn rms_norm_with_weight(
        input: &Self::Tensor,
        weight: &Self::Tensor,
        epsilon: f32,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        Self::rms_norm_without_weight(input, epsilon, context)?.multiply(weight, context)
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
    /// Applies a validated gated-product equation.
    fn gated_product(
        gate: Self::Tensor,
        up: Self::Tensor,
        policy: GatedProductPolicy,
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
    /// Sums a rank-local tensor contribution across the tensor-parallel group.
    fn sum_parallel(
        value: Self::Tensor,
        parallel: &Self::ParallelContext,
        context: &<Self::Tensor as Tensor>::Context,
    ) -> Result<Self::Tensor, Error> {
        let _ = (value, parallel, context);
        Err(Error::backend(
            "tensor-parallel sum is not implemented by this backend",
        ))
    }

    /// Number of participants in a tensor-parallel collective context.
    fn parallel_size(_parallel: &Self::ParallelContext) -> usize {
        1
    }
}

/// Projected inputs to one gated-delta recurrent scan.
#[derive(Debug, Clone, Copy)]
pub struct GatedDeltaScanInput<'a, T> {
    /// Normalized queries `[batch, sequence, heads, dimensions]`.
    pub query: &'a T,
    /// Normalized keys `[batch, sequence, heads, dimensions]`.
    pub key: &'a T,
    /// Values `[batch, sequence, heads, dimensions]`.
    pub value: &'a T,
    /// Log transition decay, scalar- or vector-valued per head.
    pub log_decay: &'a T,
    /// Update strength `[batch, sequence, heads]`.
    pub beta: &'a T,
    /// Optional FP32 recurrent matrix.
    pub initial_state: Option<&'a T>,
}

/// Final state and per-token output of a gated-delta scan.
#[derive(Debug, Clone)]
pub struct GatedDeltaScanOutput<T> {
    /// FP32 recurrent state after the final token.
    pub state: T,
    /// Recurrent output `[batch, sequence, heads, dimensions]`.
    pub output: T,
}

/// Projected inputs to one Mamba-style selective state-space scan.
#[derive(Debug, Clone, Copy)]
pub struct SelectiveStateSpaceScanInput<'a, T> {
    /// Values `[batch, sequence, heads, head_dimensions]`.
    pub values: &'a T,
    /// Expanded input state vectors `[batch, sequence, heads, state_dimensions]`.
    pub input_state: &'a T,
    /// Expanded output state vectors `[batch, sequence, heads, state_dimensions]`.
    pub output_state: &'a T,
    /// Unnormalized timesteps `[batch, sequence, heads]`.
    pub time_step: &'a T,
    /// Per-head timestep bias `[heads]`.
    pub time_step_bias: &'a T,
    /// Per-head logarithmic transition magnitude `[heads]`.
    pub transition_log: &'a T,
    /// Per-head direct skip coefficient `[heads]`.
    pub skip: &'a T,
    /// Optional FP32 state `[batch, heads, head_dimensions, state_dimensions]`.
    pub initial_state: Option<&'a T>,
    /// Lower bound applied after softplus timestep discretization.
    pub time_step_floor: f32,
    /// Maximum number of prefill tokens processed per backend chunk.
    pub chunk_size: usize,
}

/// Final state and per-token output of a selective state-space scan.
#[derive(Debug, Clone)]
pub struct SelectiveStateSpaceScanOutput<T> {
    /// FP32 recurrent state after the final token.
    pub state: T,
    /// Scan output `[batch, sequence, heads, head_dimensions]`.
    pub output: T,
}

/// Deterministic host reference for the selective state-space recurrence.
#[allow(clippy::too_many_arguments)]
pub fn reference_selective_state_space_scan(
    batch: usize,
    sequence: usize,
    heads: usize,
    head_dimensions: usize,
    state_dimensions: usize,
    values: &[f32],
    input_state: &[f32],
    output_state: &[f32],
    time_step: &[f32],
    time_step_bias: &[f32],
    transition_log: &[f32],
    skip: &[f32],
    time_step_floor: f32,
    initial_state: Option<&[f32]>,
) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let groups = batch * sequence * heads;
    let values_len = groups * head_dimensions;
    let vectors_len = groups * state_dimensions;
    let state_len = batch * heads * head_dimensions * state_dimensions;
    if values.len() != values_len
        || input_state.len() != vectors_len
        || output_state.len() != vectors_len
        || time_step.len() != groups
        || time_step_bias.len() != heads
        || transition_log.len() != heads
        || skip.len() != heads
        || initial_state.is_some_and(|state| state.len() != state_len)
        || !time_step_floor.is_finite()
        || time_step_floor < 0.0
    {
        return Err(Error::backend(
            "invalid selective state-space reference geometry",
        ));
    }
    let mut state = initial_state.map_or_else(|| vec![0.0; state_len], <[f32]>::to_vec);
    let mut output = vec![0.0; values_len];
    for batch_index in 0..batch {
        for token in 0..sequence {
            for head in 0..heads {
                let group = (batch_index * sequence + token) * heads + head;
                let dt =
                    ((time_step[group] + time_step_bias[head]).exp().ln_1p()).max(time_step_floor);
                let transition = (-transition_log[head].exp() * dt).exp();
                let vector_base = group * state_dimensions;
                for dimension in 0..head_dimensions {
                    let value_index = group * head_dimensions + dimension;
                    let state_base =
                        (batch_index * heads + head) * head_dimensions * state_dimensions
                            + dimension * state_dimensions;
                    let value = values[value_index];
                    let mut projected = 0.0f32;
                    for state_dimension in 0..state_dimensions {
                        let state_index = state_base + state_dimension;
                        state[state_index] = state[state_index] * transition
                            + dt * input_state[vector_base + state_dimension] * value;
                        projected +=
                            state[state_index] * output_state[vector_base + state_dimension];
                    }
                    output[value_index] = projected + value * skip[head];
                }
            }
        }
    }
    Ok((state, output))
}

/// Deterministic host reference for the gated-delta recurrence.
///
/// Query, key, and value use flattened `[batch, sequence, heads, dimensions]`
/// storage. Decay is either `[batch, sequence, heads]` or the same vector
/// shape as query/key. Returned state is `[batch, heads, key_dim, value_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn reference_gated_delta_scan(
    batch: usize,
    sequence: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    log_decay: &[f32],
    vector_decay: bool,
    beta: &[f32],
    initial_state: Option<&[f32]>,
) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let key_values = batch * sequence * heads * key_dim;
    let values = batch * sequence * heads * value_dim;
    let groups = batch * sequence * heads;
    let state_values = batch * heads * key_dim * value_dim;
    if query.len() != key_values
        || key.len() != key_values
        || value.len() != values
        || beta.len() != groups
        || log_decay.len() != if vector_decay { key_values } else { groups }
        || initial_state.is_some_and(|state| state.len() != state_values)
    {
        return Err(Error::backend("invalid gated-delta reference geometry"));
    }
    let mut state = initial_state.map_or_else(|| vec![0.0; state_values], <[f32]>::to_vec);
    let mut output = vec![0.0; values];
    for batch_index in 0..batch {
        for token in 0..sequence {
            for head in 0..heads {
                let group = (batch_index * sequence + token) * heads + head;
                let state_group = (batch_index * heads + head) * key_dim * value_dim;
                for value_index in 0..value_dim {
                    let mut memory = 0.0f32;
                    for key_index in 0..key_dim {
                        let vector_index = group * key_dim + key_index;
                        let decay = if vector_decay {
                            log_decay[vector_index]
                        } else {
                            log_decay[group]
                        }
                        .exp();
                        let state_index = state_group + key_index * value_dim + value_index;
                        state[state_index] *= decay;
                        memory += state[state_index] * key[vector_index];
                    }
                    let value_index_flat = group * value_dim + value_index;
                    let delta = (value[value_index_flat] - memory) * beta[group];
                    let mut accumulated = 0.0f32;
                    for key_index in 0..key_dim {
                        let vector_index = group * key_dim + key_index;
                        let state_index = state_group + key_index * value_dim + value_index;
                        state[state_index] += key[vector_index] * delta;
                        accumulated += state[state_index] * query[vector_index];
                    }
                    output[value_index_flat] = accumulated;
                }
            }
        }
    }
    Ok((state, output))
}

#[cfg(test)]
mod gated_delta_reference_tests {
    use super::reference_gated_delta_scan;

    #[test]
    fn chunked_continuation_matches_one_scan() {
        let query = [0.5, -0.25, 0.1, 0.2, -0.4, 0.8];
        let key = [0.3, 0.4, -0.2, 0.7, 0.6, -0.1];
        let value = [1.0, -0.5, 0.25, 0.75, -0.3, 0.9];
        let decay = [-0.2, -0.1, -0.4, -0.3, -0.5, -0.25];
        let beta = [0.8, 0.6, 0.4];
        let (expected_state, expected) = reference_gated_delta_scan(
            1, 3, 1, 2, 2, &query, &key, &value, &decay, true, &beta, None,
        )
        .unwrap();
        let (state, mut actual) = reference_gated_delta_scan(
            1,
            2,
            1,
            2,
            2,
            &query[..4],
            &key[..4],
            &value[..4],
            &decay[..4],
            true,
            &beta[..2],
            None,
        )
        .unwrap();
        let (actual_state, tail) = reference_gated_delta_scan(
            1,
            1,
            1,
            2,
            2,
            &query[4..],
            &key[4..],
            &value[4..],
            &decay[4..],
            true,
            &beta[2..],
            Some(&state),
        )
        .unwrap();
        actual.extend(tail);
        assert!(expected
            .iter()
            .zip(actual)
            .all(|(left, right)| (left - right).abs() < 1e-6));
        assert!(expected_state
            .iter()
            .zip(actual_state)
            .all(|(left, right)| (left - right).abs() < 1e-6));
    }
}

#[cfg(test)]
mod selective_state_space_reference_tests {
    use super::reference_selective_state_space_scan;

    #[test]
    fn continuation_matches_one_scan() {
        let values = [0.2, -0.4, 0.8, 0.5, -0.3, 0.7];
        let input_state = [0.1, 0.3, -0.2, 0.4, 0.6, -0.5];
        let output_state = [0.7, -0.1, 0.2, 0.5, -0.4, 0.9];
        let time_step = [-0.3, 0.1, -0.2];
        let bias = [0.05];
        let transition = [-0.4];
        let skip = [0.25];
        let (expected_state, expected) = reference_selective_state_space_scan(
            1,
            3,
            1,
            2,
            2,
            &values,
            &input_state,
            &output_state,
            &time_step,
            &bias,
            &transition,
            &skip,
            0.001,
            None,
        )
        .unwrap();
        let (state, mut actual) = reference_selective_state_space_scan(
            1,
            2,
            1,
            2,
            2,
            &values[..4],
            &input_state[..4],
            &output_state[..4],
            &time_step[..2],
            &bias,
            &transition,
            &skip,
            0.001,
            None,
        )
        .unwrap();
        let (actual_state, tail) = reference_selective_state_space_scan(
            1,
            1,
            1,
            2,
            2,
            &values[4..],
            &input_state[4..],
            &output_state[4..],
            &time_step[2..],
            &bias,
            &transition,
            &skip,
            0.001,
            Some(&state),
        )
        .unwrap();
        actual.extend(tail);
        assert!(expected
            .iter()
            .zip(actual)
            .all(|(left, right)| (left - right).abs() < 1e-6));
        assert!(expected_state
            .iter()
            .zip(actual_state)
            .all(|(left, right)| (left - right).abs() < 1e-6));
    }
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
    /// Creates a signed 32-bit integer tensor from host initialization data.
    fn from_i32_slice(
        values: &[i32],
        shape: &[i32],
        context: &Self::Context,
    ) -> Result<Self, Error> {
        let _ = (values, shape, context);
        Err(Error::backend(
            "I32 tensor construction is not implemented by this backend",
        ))
    }
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
    /// Elementwise hyperbolic tangent.
    fn tanh(&self, context: &Self::Context) -> Result<Self, Error> {
        let _ = context;
        Err(Error::backend(
            "tanh is not implemented by this tensor backend",
        ))
    }
    /// Elementwise maximum with a scalar.
    fn maximum_scalar(&self, rhs: f32, context: &Self::Context) -> Result<Self, Error>;
    /// Elementwise clamp using backend tensor bounds that may be scalar or broadcastable.
    fn clip(&self, minimum: &Self, maximum: &Self, context: &Self::Context) -> Result<Self, Error> {
        let _ = (minimum, maximum, context);
        Err(Error::backend(
            "clip is not implemented by this tensor backend",
        ))
    }

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
    /// Creates a zero tensor with the same shape and physical dtype.
    fn zeros_like(&self, context: &Self::Context) -> Result<Self, Error> {
        let _ = context;
        Err(Error::backend(
            "dtype-preserving zero allocation is not implemented by this tensor backend",
        ))
    }
    /// Compares every element with one signed integer scalar.
    fn equal_i32(&self, value: i32, context: &Self::Context) -> Result<Self, Error> {
        let _ = (value, context);
        Err(Error::backend(
            "integer scalar comparison is not implemented by this tensor backend",
        ))
    }
    /// Elementwise logical disjunction over two boolean tensors.
    fn logical_or(&self, rhs: &Self, context: &Self::Context) -> Result<Self, Error> {
        let _ = (rhs, context);
        Err(Error::backend(
            "logical disjunction is not implemented by this tensor backend",
        ))
    }
    /// Selects elements from two broadcast-compatible tensors using a boolean condition.
    fn where_condition(
        condition: &Self,
        when_true: &Self,
        when_false: &Self,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        let _ = (condition, when_true, when_false, context);
        Err(Error::backend(
            "conditional selection is not implemented by this tensor backend",
        ))
    }
    /// Scatters source rows into the true entries of a boolean mask.
    fn masked_scatter(
        &self,
        mask: &Self,
        source: &Self,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        let _ = (mask, source, context);
        Err(Error::backend(
            "masked scatter is not implemented by this tensor backend",
        ))
    }

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
    /// Two-dimensional convolution over canonical NHWC inputs and OHWI weights.
    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        input: &Self,
        weight: &Self,
        stride: (i32, i32),
        padding: (i32, i32),
        dilation: (i32, i32),
        groups: i32,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        let _ = (input, weight, stride, padding, dilation, groups, context);
        Err(Error::backend(
            "two-dimensional convolution is not implemented by this backend",
        ))
    }
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
    /// Builds cosine and sine tensors for explicit multi-axis positions.
    fn multi_axis_rotary_embeddings(
        position_ids: &Self,
        spec: &multimodal::MultiAxisRotarySpec,
        context: &Self::Context,
    ) -> Result<(Self, Self), Error> {
        let _ = (position_ids, spec, context);
        Err(Error::backend(
            "multi-axis rotary embeddings are not implemented by this backend",
        ))
    }
    /// Projects selected output rows and scatters them into vocabulary order.
    fn masked_output_projection(
        input: multimodal::MaskedOutputProjectionInput<'_, Self>,
        context: &Self::Context,
    ) -> Result<Self, Error> {
        let _ = (input, context);
        Err(Error::backend(
            "masked output projection is not implemented by this backend",
        ))
    }
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

/// Post-convolution activation selected by architecture policy.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ConvolutionActivation {
    /// Preserve the affine convolution output.
    Identity,
    /// Apply the sigmoid linear unit.
    Silu,
}

/// Geometry and parameter identity for a causal depthwise convolution.
#[derive(Debug, Clone)]
pub struct CausalDepthwiseConvolutionSpec {
    /// Number of independent channels.
    pub channels: i32,
    /// Causal kernel width, including the current token.
    pub kernel_size: i32,
    /// Checkpoint-facing kernel stored as `[channels, 1, kernel]`.
    pub weight: ParameterSpec,
    /// Optional per-channel affine bias.
    pub bias: Option<ParameterSpec>,
    /// Activation applied after convolution and bias.
    pub activation: ConvolutionActivation,
}

impl CausalDepthwiseConvolutionSpec {
    /// Validates positive geometry before allocating backend parameters.
    pub fn validate(&self) -> Result<(), Error> {
        if self.channels <= 0 {
            return Err(Error::backend(format!(
                "causal depthwise convolution channels must be positive, got {}",
                self.channels
            )));
        }
        if self.kernel_size <= 0 {
            return Err(Error::backend(format!(
                "causal depthwise convolution kernel size must be positive, got {}",
                self.kernel_size
            )));
        }
        Ok(())
    }
}

/// Output and exact bounded history produced by one causal convolution call.
#[derive(Debug, Clone)]
pub struct CausalDepthwiseConvolutionOutput<T> {
    /// Activated convolution output shaped like the input.
    pub output: T,
    /// Last `kernel_size - 1` inputs, or `None` for a width-one kernel.
    pub history: Option<T>,
}

/// Backend-neutral causal depthwise convolution.
///
/// The layer owns only parameters and equations. Callers keep the returned
/// bounded history in their runtime state realization.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct CausalDepthwiseConvolution<B: NeuralBackend> {
    /// Checkpoint-layout kernel shaped `[channels, 1, kernel]`.
    pub weight: Parameter<B::Tensor>,
    /// Optional per-channel bias.
    pub bias: Option<Parameter<B::Tensor>>,
    #[parameter(skip)]
    channels: i32,
    #[parameter(skip)]
    kernel_size: i32,
    #[parameter(skip)]
    activation: ConvolutionActivation,
}

impl<B: NeuralBackend> CausalDepthwiseConvolution<B> {
    /// Creates unloaded convolution parameters.
    pub fn new(
        spec: CausalDepthwiseConvolutionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        Ok(Self {
            weight: Parameter::unloaded(
                spec.weight,
                &[spec.channels, 1, spec.kernel_size],
                context,
            )?,
            bias: spec
                .bias
                .map(|bias| Parameter::unloaded(bias, &[spec.channels], context))
                .transpose()?,
            channels: spec.channels,
            kernel_size: spec.kernel_size,
            activation: spec.activation,
        })
    }

    /// Returns the exact retained causal-history length.
    pub const fn history_len(&self) -> i32 {
        self.kernel_size - 1
    }

    /// Applies the convolution and returns the replacement bounded history.
    pub fn forward(
        &self,
        input: &B::Tensor,
        history: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<CausalDepthwiseConvolutionOutput<B::Tensor>, Error> {
        let shape = input.shape();
        if shape.len() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] != self.channels {
            return Err(Error::backend(format!(
                "causal depthwise convolution expects [batch, sequence, {}], got {shape:?}",
                self.channels
            )));
        }
        let history_len = self.history_len();
        let padded = if history_len == 0 {
            if history.is_some() {
                return Err(Error::backend(
                    "width-one causal convolution does not accept history",
                ));
            }
            input.clone()
        } else if let Some(history) = history {
            let expected = [shape[0], history_len, self.channels];
            if history.shape() != expected {
                return Err(Error::backend(format!(
                    "causal depthwise convolution history must have shape {expected:?}, got {:?}",
                    history.shape()
                )));
            }
            B::Tensor::concatenate(&[history.clone(), input.clone()], 1, context)?
        } else {
            B::Tensor::pad(
                input,
                &[(0, 0), (history_len, 0), (0, 0)],
                PadMode::Constant,
                context,
            )?
        };
        let execution_weight = self.weight.as_ref().swap_axes(1, 2, context)?;
        let mut output =
            B::Tensor::conv1d(&padded, &execution_weight, 1, 0, 1, self.channels, context)?;
        if output.shape() != shape {
            return Err(Error::backend(format!(
                "causal depthwise convolution backend returned shape {:?}, expected {shape:?}",
                output.shape()
            )));
        }
        if let Some(bias) = &self.bias {
            let bias = bias
                .as_ref()
                .reshape(&[1, 1, self.channels], context)?
                .broadcast_to(shape, context)?;
            output = output.add(&bias, context)?;
        }
        if self.activation == ConvolutionActivation::Silu {
            output = B::silu(output, context)?;
        }
        let history = (history_len > 0)
            .then(|| {
                padded.index(
                    &[
                        Index::Full,
                        Index::Range(shape[1], shape[1] + history_len),
                        Index::Full,
                    ],
                    context,
                )
            })
            .transpose()?;
        Ok(CausalDepthwiseConvolutionOutput { output, history })
    }
}

/// Construction policy for a gated causal short convolution.
#[derive(Debug, Clone)]
pub struct GatedShortConvolutionSpec {
    /// Hidden width accepted by the fused input projection.
    pub input_dimensions: i32,
    /// Rank-local convolution channel count.
    pub channels: i32,
    /// Hidden width returned by the output projection.
    pub output_dimensions: i32,
    /// Fused B/C/x projection with output width `3 * channels`.
    pub input_projection: LinearSpec,
    /// Projection from convolution channels to the output width.
    pub output_projection: LinearSpec,
    /// Shared causal depthwise convolution parameters.
    pub convolution: CausalDepthwiseConvolutionSpec,
}

impl GatedShortConvolutionSpec {
    /// Validates the fused segment and convolution geometry.
    pub fn validate(&self) -> Result<(), Error> {
        self.convolution.validate()?;
        let fused = self
            .channels
            .checked_mul(3)
            .ok_or_else(|| Error::backend("gated short-convolution width overflowed"))?;
        if self.input_dimensions <= 0
            || self.channels <= 0
            || self.output_dimensions <= 0
            || self.convolution.channels != self.channels
            || self.input_projection.input != self.input_dimensions
            || self.input_projection.output != fused
            || self.output_projection.input != self.channels
            || self.output_projection.output != self.output_dimensions
        {
            return Err(Error::backend(format!(
                "invalid gated short-convolution geometry input={} channels={} output={} fused_projection={}x{} output_projection={}x{} convolution_channels={}",
                self.input_dimensions,
                self.channels,
                self.output_dimensions,
                self.input_projection.input,
                self.input_projection.output,
                self.output_projection.input,
                self.output_projection.output,
                self.convolution.channels,
            )));
        }
        Ok(())
    }
}

/// Output and replacement bounded state from a gated short convolution.
#[derive(Debug, Clone)]
pub struct GatedShortConvolutionOutput<T> {
    /// Projected hidden states.
    pub output: T,
    /// Last causal input values retained by the depthwise convolution.
    pub history: Option<T>,
}

/// Fused gated short-convolution layer shared by hybrid decoders.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct GatedShortConvolution<B: NeuralBackend> {
    /// Fused B/C/x input projection.
    pub input_projection: B::Linear,
    /// Shared causal depthwise convolution.
    pub convolution: CausalDepthwiseConvolution<B>,
    /// Output projection.
    pub output_projection: B::Linear,
    #[parameter(skip)]
    channels: i32,
}

impl<B: NeuralBackend> GatedShortConvolution<B> {
    /// Builds one unloaded gated short convolution.
    pub fn new(
        spec: GatedShortConvolutionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        Ok(Self {
            input_projection: B::linear(spec.input_projection, context)?,
            convolution: CausalDepthwiseConvolution::new(spec.convolution, context)?,
            output_projection: B::linear(spec.output_projection, context)?,
            channels: spec.channels,
        })
    }

    fn hidden(
        &mut self,
        input: &B::Tensor,
        history: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, Option<B::Tensor>), Error> {
        let projected = self.input_projection.forward(input, context)?;
        let rank = projected.shape().len();
        if rank == 0 || projected.shape()[rank - 1] != 3 * self.channels {
            return Err(Error::backend(format!(
                "gated short-convolution projection returned shape {:?}, expected final width {}",
                projected.shape(),
                3 * self.channels
            )));
        }
        let mut segment = vec![Index::Full; rank];
        segment[rank - 1] = Index::Range(0, self.channels);
        let b = projected.index(&segment, context)?;
        segment[rank - 1] = Index::Range(self.channels, 2 * self.channels);
        let c = projected.index(&segment, context)?;
        segment[rank - 1] = Index::Range(2 * self.channels, 3 * self.channels);
        let x = projected.index(&segment, context)?;
        let convolution = self
            .convolution
            .forward(&b.multiply(&x, context)?, history, context)?;
        Ok((
            c.multiply(&convolution.output, context)?,
            convolution.history,
        ))
    }

    /// Executes the replicated layer and returns replacement bounded state.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        history: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<GatedShortConvolutionOutput<B::Tensor>, Error> {
        let (hidden, history) = self.hidden(input, history, context)?;
        Ok(GatedShortConvolutionOutput {
            output: self.output_projection.forward(&hidden, context)?,
            history,
        })
    }

    /// Executes the same layer with a row-parallel output projection.
    pub fn forward_parallel(
        &mut self,
        input: &B::Tensor,
        history: Option<&B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<GatedShortConvolutionOutput<B::Tensor>, Error> {
        let (hidden, history) = self.hidden(input, history, context)?;
        Ok(GatedShortConvolutionOutput {
            output: B::row_parallel_linear(
                &mut self.output_projection,
                &hidden,
                parallel,
                context,
            )?,
            history,
        })
    }
}

#[cfg(test)]
mod routed_contract_tests {
    use super::*;

    fn parameters(prefix: &str) -> GatedProductExpertParameters {
        GatedProductExpertParameters {
            gate: ExpertProjectionSpec {
                weight: ParameterSpec::trainable(format!("{prefix}.gate.weight")).unwrap(),
                bias: None,
                format: LinearFormat::Dense,
            },
            up: ExpertProjectionSpec {
                weight: ParameterSpec::trainable(format!("{prefix}.up.weight")).unwrap(),
                bias: None,
                format: LinearFormat::Dense,
            },
            down: ExpertProjectionSpec {
                weight: ParameterSpec::trainable(format!("{prefix}.down.weight")).unwrap(),
                bias: None,
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
    fn gated_product_policy_rejects_malformed_scalars() {
        assert!(
            GatedProductPolicy::new(GatedProductActivation::Silu, Some(0.0), None, 1.0, 0.0,)
                .is_err()
        );
        assert!(GatedProductPolicy::new(
            GatedProductActivation::Silu,
            None,
            Some(f32::NAN),
            1.0,
            0.0,
        )
        .is_err());
        assert!(
            GatedProductPolicy::new(GatedProductActivation::Silu, None, None, 0.0, 0.0,).is_err()
        );
        assert!(GatedProductPolicy::new(
            GatedProductActivation::Silu,
            None,
            None,
            1.0,
            f32::INFINITY,
        )
        .is_err());
    }

    #[test]
    fn router_projection_and_correction_biases_require_distinct_identities() {
        let shared_bias = ParameterSpec::trainable("router.bias").unwrap();
        let spec = TopKRouterSpec {
            input_dimensions: 4,
            weight: ParameterSpec::trainable("router.weight").unwrap(),
            bias: Some(shared_bias.clone()),
            correction_bias: Some(shared_bias),
            input_transform: None,
            route_scale: None,
            quantization: None,
            routing: TopKRoutingSpec::new(2, 1, RoutingScoring::SelectedSoftmax, false).unwrap(),
        };

        assert!(spec.validate().is_err());
    }

    #[test]
    fn independent_expert_layout_requires_exact_cardinality() {
        let valid = GatedProductExpertBankSpec {
            expert_count: 2,
            input_dimensions: 16,
            intermediate_dimensions: 8,
            output_dimensions: 16,
            policy: eredu_nn::GatedProductPolicy::ordinary_silu(),
            layout: GatedProductExpertLayout::Independent(vec![parameters("e0"), parameters("e1")]),
        };
        assert!(valid.validate().is_ok());
        let invalid = GatedProductExpertBankSpec {
            layout: GatedProductExpertLayout::Independent(vec![parameters("e0")]),
            ..valid
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn gated_product_bank_rejects_reused_projection_bias_identity() {
        let shared = ParameterSpec::trainable("experts.gate_up").unwrap();
        let spec = GatedProductExpertBankSpec {
            expert_count: 1,
            input_dimensions: 4,
            intermediate_dimensions: 2,
            output_dimensions: 4,
            policy: GatedProductPolicy::ordinary_silu(),
            layout: GatedProductExpertLayout::Packed {
                gate_up: ExpertProjectionSpec {
                    weight: shared.clone(),
                    bias: Some(shared),
                    format: LinearFormat::Dense,
                },
                down: ExpertProjectionSpec {
                    weight: ParameterSpec::trainable("experts.down").unwrap(),
                    bias: None,
                    format: LinearFormat::Dense,
                },
            },
        };

        assert!(spec.validate().is_err());
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

#[cfg(test)]
mod fused_projection_layout_tests {
    use super::*;

    #[test]
    fn component_major_layout_is_checked_and_stable() {
        let layout = FusedProjectionLayout::new([
            FusedProjectionSegment::new("query", 8).unwrap(),
            FusedProjectionSegment::new("key", 4).unwrap(),
            FusedProjectionSegment::new("value", 4).unwrap(),
        ])
        .unwrap();
        assert_eq!(layout.output_width(), 16);
        assert_eq!(
            layout
                .segments()
                .iter()
                .map(|segment| (segment.name(), segment.width()))
                .collect::<Vec<_>>(),
            [("query", 8), ("key", 4), ("value", 4)]
        );
        assert!(FusedProjectionLayout::new(Vec::new()).is_err());
        assert!(FusedProjectionLayout::new([
            FusedProjectionSegment::new("same", 1).unwrap(),
            FusedProjectionSegment::new("same", 1).unwrap(),
        ])
        .is_err());
        assert!(FusedProjectionSegment::new("", 1).is_err());
        assert!(FusedProjectionSegment::new("bad", 0).is_err());
    }

    #[test]
    fn zero_sentinel_cannot_alias_an_embedding_row() {
        EmbeddingLookupPolicy::Strict.validate().unwrap();
        EmbeddingLookupPolicy::ZeroSentinel(-1).validate().unwrap();
        assert!(EmbeddingLookupPolicy::ZeroSentinel(0).validate().is_err());
    }

    #[test]
    fn vocabulary_parallel_ownership_requires_exact_global_rows() {
        let range = VocabularyParallelRange {
            global_vocabulary: 5,
            local: 0..3,
        };
        range.validate_global_rows(5).unwrap();
        assert!(range.validate_global_rows(4).is_err());
        assert!(range.validate_global_rows(-1).is_err());
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
