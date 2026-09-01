//! Durable cache identity, geometry, and state-residency contracts.

use std::{collections::BTreeSet, num::NonZeroU32};

use serde::{Deserialize, Serialize};

use crate::attention::AttentionPolicy;

/// Representation stored atomically in one cache block.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheRepresentation {
    /// Standard attention keys and values.
    KeyValue,
    /// Compressed latent state and rotary keys.
    CompressedLatentRotary,
}

/// Optional rank identity included in a stable cache block identifier.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct CacheRankIdentity {
    /// Ordered-stage rank, when stage partitioning is active.
    stage_rank: Option<usize>,
    /// State-shard rank, when cache state is partitioned.
    shard_rank: Option<usize>,
    /// Addressable-group rank for replicated cache state.
    addressable_rank: Option<usize>,
}

impl CacheRankIdentity {
    /// Creates a generic rank identity for persisted cache state.
    pub const fn new(
        stage_rank: Option<usize>,
        shard_rank: Option<usize>,
        addressable_rank: Option<usize>,
    ) -> Self {
        Self {
            stage_rank,
            shard_rank,
            addressable_rank,
        }
    }

    /// Returns the ordered-stage rank, when present.
    pub const fn stage_rank(&self) -> Option<usize> {
        self.stage_rank
    }

    /// Returns the state-shard rank, when present.
    pub const fn shard_rank(&self) -> Option<usize> {
        self.shard_rank
    }

    /// Returns the addressable-group rank, when present.
    pub const fn addressable_rank(&self) -> Option<usize> {
        self.addressable_rank
    }
}

/// Stable identity for one immutable sealed cache block.
#[derive(Debug, Clone, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct CacheBlockId {
    /// Identity shared by every block in one live cache.
    pub session_id: u64,
    /// Architecture-global decoder layer index.
    pub global_layer: usize,
    /// Stored attention representation.
    pub representation: CacheRepresentation,
    /// Inclusive absolute token position.
    pub start: i64,
    /// Exclusive absolute token position.
    pub end: i64,
    /// Rank-local ownership identity.
    pub rank: Option<CacheRankIdentity>,
}

/// Logical location of a sealed cache block.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTier {
    /// Available to execution without a catalog load.
    Device,
    /// Evaluated CPU-resident state without an execution-device copy.
    Host,
    /// Stored in a backend-owned persistent shard.
    Disk,
}

/// Exact state kind, attention policy, and tensor geometry for one decoder layer.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerCachePolicy {
    /// This layer contributes no independently persisted state.
    NoState,
    /// Ordinary attention keys and values.
    KeyValue {
        /// Exact full or sliding attention range.
        attention: AttentionPolicy,
        /// Rank-local key/value head count.
        num_key_value_heads: NonZeroU32,
        /// Per-head key/value dimension.
        head_dim: NonZeroU32,
    },
    /// Attention history whose value payload is intentionally empty.
    KeyOnly {
        /// Exact full or sliding attention range.
        attention: AttentionPolicy,
        /// Rank-local key head count.
        num_key_heads: NonZeroU32,
        /// Per-head key dimension.
        head_dim: NonZeroU32,
    },
    /// Compressed latent state plus rotary keys.
    CompressedLatentRotary {
        /// Exact full or sliding attention range.
        attention: AttentionPolicy,
        /// Compressed latent width.
        latent_dim: NonZeroU32,
        /// Rotary-key width.
        rotary_dim: NonZeroU32,
    },
    /// Fixed-size recurrent or convolution state without attention.
    FixedState {
        /// Ordered tensors required to resume this layer.
        tensors: Vec<StateTensorPolicy>,
    },
    /// Ordinary attention plus fixed-size state.
    KeyValueWithFixedState {
        /// Exact full or sliding attention range.
        attention: AttentionPolicy,
        /// Rank-local key/value head count.
        num_key_value_heads: NonZeroU32,
        /// Per-head key/value dimension.
        head_dim: NonZeroU32,
        /// Ordered additional tensors.
        tensors: Vec<StateTensorPolicy>,
    },
    /// Key-only attention plus fixed-size state.
    KeyOnlyWithFixedState {
        /// Exact full or sliding attention range.
        attention: AttentionPolicy,
        /// Rank-local key head count.
        num_key_heads: NonZeroU32,
        /// Per-head key dimension.
        head_dim: NonZeroU32,
        /// Ordered additional tensors.
        tensors: Vec<StateTensorPolicy>,
    },
}

/// Semantic role of one non-attention cache tensor.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTensorRole {
    /// Bounded causal-convolution history.
    Convolution {
        /// Stable slot within the layer's convolution states.
        slot: u32,
    },
    /// Recurrent transition or linear-attention state.
    Recurrent,
    /// Prepared multimodal prefix embeddings.
    PrefixEmbedding,
    /// Model-global multimodal position offset.
    PositionDelta,
    /// One tensor in an append-only token-pooling stream.
    Pooling {
        /// Stable stream slot within the owning layer.
        stream: u32,
        /// Exact component of the pooling state.
        component: PoolingStateComponent,
    },
}

/// Semantic component of one append-only pooling stream.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingStateComponent {
    /// Source values waiting for a complete pooling group.
    PendingValues,
    /// Source gate logits waiting for a complete pooling group.
    PendingGates,
    /// Complete pooled output history.
    Pooled,
    /// Source values retained for an overlapping group.
    OverlapValues,
    /// Source gate logits retained for an overlapping group.
    OverlapGates,
}

/// Runtime ownership behavior for live model state.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateResidencyClass {
    /// Small mutable state that remains on the execution device.
    AlwaysDeviceMutable,
    /// Append-only state that becomes immutable blocks before paging.
    SealablePaged,
    /// Mutable state promoted only for its owning layer.
    LayerScopedOffloadable,
}

/// Residency behaviors valid for mutable fixed-state tensors.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MutableStateResidency {
    /// Small mutable state that remains on the execution device.
    AlwaysDeviceMutable,
    /// Mutable state promoted only for its owning layer.
    LayerScopedOffloadable,
}

impl From<MutableStateResidency> for StateResidencyClass {
    fn from(value: MutableStateResidency) -> Self {
        match value {
            MutableStateResidency::AlwaysDeviceMutable => Self::AlwaysDeviceMutable,
            MutableStateResidency::LayerScopedOffloadable => Self::LayerScopedOffloadable,
        }
    }
}

/// One dimension in a persisted fixed-state tensor.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTensorDimension {
    /// Manifest batch size.
    Batch,
    /// Exact prompt token count.
    PrefixTokens,
    /// Quotient of prompt tokens and a positive divisor.
    PrefixTokensDiv(NonZeroU32),
    /// Remainder of prompt tokens and a positive divisor.
    PrefixTokensRem(NonZeroU32),
    /// Positive architecture-defined dimension.
    Fixed(NonZeroU32),
    /// Scalar dimension list marker; valid only as the sole entry.
    Scalar,
}

/// Condition under which a state tensor must be materialized.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTensorPresence {
    /// Every persisted cache materializes the tensor.
    Required,
    /// The tensor may be present independently of prefix geometry.
    Optional,
    /// Present exactly when the prefix has a non-zero remainder.
    PrefixRemainderNonZero(NonZeroU32),
    /// Present exactly when the prefix contains one complete group.
    PrefixAtLeast(NonZeroU32),
}

/// Accepted dtype family for one fixed-state tensor.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTensorDtype {
    /// Any floating dtype.
    Floating,
    /// Exactly IEEE F32.
    Float32,
    /// Signed 32-bit integer.
    Int32,
    /// Unsigned 32-bit integer.
    Uint32,
}

/// Exact semantic role, symbolic shape, and dtype contract for a state tensor.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct StateTensorPolicy {
    /// Meaning of this tensor within its owner.
    pub role: StateTensorRole,
    /// Symbolic shape resolved from batch and prefix geometry.
    pub shape: Vec<StateTensorDimension>,
    /// Accepted dtype family.
    pub dtype: StateTensorDtype,
    /// Authoritative live-state residency behavior.
    pub residency: StateResidencyClass,
    /// Condition under which persisted caches materialize this tensor.
    pub presence: StateTensorPresence,
}

/// Owner of one persisted non-attention state tensor.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTensorOwner {
    /// Architecture-global decoder layer index.
    Layer(usize),
}

/// Stable semantic identity for one independently managed runtime component.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateComponentRole {
    /// Ordinary attention keys.
    AttentionKeys,
    /// Ordinary attention values.
    AttentionValues,
    /// Head-independent compressed latent key/value state.
    CompressedLatent,
    /// Rotary keys paired with compressed latent state.
    RotaryKeys,
    /// One fixed-size or pooling tensor.
    Fixed(StateTensorRole),
}

impl StateComponentRole {
    /// Returns a stable checkpoint/runtime component name.
    pub fn stable_name(self) -> String {
        match self {
            Self::AttentionKeys => "attention.keys".into(),
            Self::AttentionValues => "attention.values".into(),
            Self::CompressedLatent => "attention.compressed_latent".into(),
            Self::RotaryKeys => "attention.rotary_keys".into(),
            Self::Fixed(StateTensorRole::Convolution { slot }) => {
                format!("state.convolution.{slot}")
            }
            Self::Fixed(StateTensorRole::Recurrent) => "state.recurrent".into(),
            Self::Fixed(StateTensorRole::PrefixEmbedding) => "state.prefix_embedding".into(),
            Self::Fixed(StateTensorRole::PositionDelta) => "state.position_delta".into(),
            Self::Fixed(StateTensorRole::Pooling { stream, component }) => {
                let component = match component {
                    PoolingStateComponent::PendingValues => "pending_values",
                    PoolingStateComponent::PendingGates => "pending_gates",
                    PoolingStateComponent::Pooled => "pooled",
                    PoolingStateComponent::OverlapValues => "overlap_values",
                    PoolingStateComponent::OverlapGates => "overlap_gates",
                };
                format!("state.pooling.{stream}.{component}")
            }
        }
    }
}

/// Symbolic geometry and persistence behavior of one named state component.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct StateComponentPolicy {
    /// Stable semantic role.
    role: StateComponentRole,
    /// Symbolic shape resolved from batch and prefix geometry.
    shape: Vec<StateTensorDimension>,
    /// Accepted persisted dtype family.
    dtype: StateTensorDtype,
    /// Runtime residency behavior.
    residency: StateResidencyClass,
    /// Condition under which persisted state materializes this component.
    presence: StateTensorPresence,
}

impl StateComponentPolicy {
    /// Returns the stable semantic component role.
    pub const fn role(&self) -> StateComponentRole {
        self.role
    }

    /// Returns the symbolic component shape.
    pub fn shape(&self) -> &[StateTensorDimension] {
        &self.shape
    }

    /// Returns the accepted persisted dtype family.
    pub const fn dtype(&self) -> StateTensorDtype {
        self.dtype
    }

    /// Returns the runtime residency class.
    pub const fn residency(&self) -> StateResidencyClass {
        self.residency
    }

    /// Returns the conditional persistence rule.
    pub const fn presence(&self) -> StateTensorPresence {
        self.presence
    }
}

impl LayerCachePolicy {
    /// Returns the residency behavior of this layer's attention payload.
    pub const fn attention_residency_class(&self) -> Option<StateResidencyClass> {
        match self {
            Self::NoState | Self::FixedState { .. } => None,
            Self::KeyValue { .. }
            | Self::KeyOnly { .. }
            | Self::CompressedLatentRotary { .. }
            | Self::KeyValueWithFixedState { .. }
            | Self::KeyOnlyWithFixedState { .. } => Some(StateResidencyClass::SealablePaged),
        }
    }

    /// Constructs validated ordinary key/value state geometry.
    pub fn key_value(
        attention: AttentionPolicy,
        num_key_value_heads: i32,
        head_dim: i32,
    ) -> Result<Self, CachePolicyError> {
        let policy = Self::KeyValue {
            attention,
            num_key_value_heads: positive_u32(num_key_value_heads, "key/value head count")?,
            head_dim: positive_u32(head_dim, "key/value head dimension")?,
        };
        policy.validate()?;
        Ok(policy)
    }

    /// Constructs validated key-only state geometry.
    pub fn key_only(
        attention: AttentionPolicy,
        num_key_heads: i32,
        head_dim: i32,
    ) -> Result<Self, CachePolicyError> {
        let policy = Self::KeyOnly {
            attention,
            num_key_heads: positive_u32(num_key_heads, "key head count")?,
            head_dim: positive_u32(head_dim, "key head dimension")?,
        };
        policy.validate()?;
        Ok(policy)
    }

    /// Constructs validated compressed-latent state geometry.
    pub fn compressed_latent_rotary(
        attention: AttentionPolicy,
        latent_dim: i32,
        rotary_dim: i32,
    ) -> Result<Self, CachePolicyError> {
        let policy = Self::CompressedLatentRotary {
            attention,
            latent_dim: positive_u32(latent_dim, "compressed latent dimension")?,
            rotary_dim: positive_u32(rotary_dim, "rotary-key dimension")?,
        };
        policy.validate()?;
        Ok(policy)
    }

    /// Constructs a validated fixed-state-only policy.
    pub fn fixed_only(tensors: Vec<StateTensorPolicy>) -> Result<Self, CachePolicyError> {
        let policy = Self::FixedState { tensors };
        policy.validate()?;
        Ok(policy)
    }

    /// Constructs validated key/value plus fixed-state geometry.
    pub fn key_value_with_fixed_state(
        attention: AttentionPolicy,
        num_key_value_heads: i32,
        head_dim: i32,
        tensors: Vec<StateTensorPolicy>,
    ) -> Result<Self, CachePolicyError> {
        let policy = Self::KeyValueWithFixedState {
            attention,
            num_key_value_heads: positive_u32(num_key_value_heads, "key/value head count")?,
            head_dim: positive_u32(head_dim, "key/value head dimension")?,
            tensors,
        };
        policy.validate()?;
        Ok(policy)
    }

    /// Constructs validated key-only plus fixed-state geometry.
    pub fn key_only_with_fixed_state(
        attention: AttentionPolicy,
        num_key_heads: i32,
        head_dim: i32,
        tensors: Vec<StateTensorPolicy>,
    ) -> Result<Self, CachePolicyError> {
        let policy = Self::KeyOnlyWithFixedState {
            attention,
            num_key_heads: positive_u32(num_key_heads, "key head count")?,
            head_dim: positive_u32(head_dim, "key head dimension")?,
            tensors,
        };
        policy.validate()?;
        Ok(policy)
    }

    /// Returns the exact attention policy when present.
    pub const fn attention(&self) -> Option<AttentionPolicy> {
        match self {
            Self::NoState | Self::FixedState { .. } => None,
            Self::KeyValue { attention, .. }
            | Self::KeyOnly { attention, .. }
            | Self::CompressedLatentRotary { attention, .. }
            | Self::KeyValueWithFixedState { attention, .. }
            | Self::KeyOnlyWithFixedState { attention, .. } => Some(*attention),
        }
    }

    /// Returns ordered non-attention tensor policies.
    pub fn fixed_state(&self) -> &[StateTensorPolicy] {
        match self {
            Self::FixedState { tensors }
            | Self::KeyValueWithFixedState { tensors, .. }
            | Self::KeyOnlyWithFixedState { tensors, .. } => tensors,
            _ => &[],
        }
    }

    /// Expands this layer policy into ordered, stably named semantic
    /// components shared by runtime residency and prompt persistence.
    pub fn components(&self) -> Vec<StateComponentPolicy> {
        let mut components = Vec::new();
        let floating = StateTensorDtype::Floating;
        let required = StateTensorPresence::Required;
        match self {
            Self::NoState | Self::FixedState { .. } => {}
            Self::KeyValue {
                num_key_value_heads,
                head_dim,
                ..
            }
            | Self::KeyValueWithFixedState {
                num_key_value_heads,
                head_dim,
                ..
            } => {
                let shape = vec![
                    StateTensorDimension::Batch,
                    StateTensorDimension::Fixed(*num_key_value_heads),
                    StateTensorDimension::PrefixTokens,
                    StateTensorDimension::Fixed(*head_dim),
                ];
                for role in [
                    StateComponentRole::AttentionKeys,
                    StateComponentRole::AttentionValues,
                ] {
                    components.push(StateComponentPolicy {
                        role,
                        shape: shape.clone(),
                        dtype: floating,
                        residency: StateResidencyClass::SealablePaged,
                        presence: required,
                    });
                }
            }
            Self::KeyOnly {
                num_key_heads,
                head_dim,
                ..
            }
            | Self::KeyOnlyWithFixedState {
                num_key_heads,
                head_dim,
                ..
            } => components.push(StateComponentPolicy {
                role: StateComponentRole::AttentionKeys,
                shape: vec![
                    StateTensorDimension::Batch,
                    StateTensorDimension::Fixed(*num_key_heads),
                    StateTensorDimension::PrefixTokens,
                    StateTensorDimension::Fixed(*head_dim),
                ],
                dtype: floating,
                residency: StateResidencyClass::SealablePaged,
                presence: required,
            }),
            Self::CompressedLatentRotary {
                latent_dim,
                rotary_dim,
                ..
            } => {
                for (role, dimension) in [
                    (StateComponentRole::CompressedLatent, *latent_dim),
                    (StateComponentRole::RotaryKeys, *rotary_dim),
                ] {
                    components.push(StateComponentPolicy {
                        role,
                        shape: vec![
                            StateTensorDimension::Batch,
                            StateTensorDimension::PrefixTokens,
                            StateTensorDimension::Fixed(dimension),
                        ],
                        dtype: floating,
                        residency: StateResidencyClass::SealablePaged,
                        presence: required,
                    });
                }
            }
        }
        components.extend(
            self.fixed_state()
                .iter()
                .map(|tensor| StateComponentPolicy {
                    role: StateComponentRole::Fixed(tensor.role),
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype,
                    residency: tensor.residency_class(),
                    presence: tensor.presence,
                }),
        );
        components
    }

    /// Validates dimensions and fixed-state invariants after deserialization.
    pub fn validate(&self) -> Result<(), CachePolicyError> {
        if let Some(attention) = self.attention() {
            attention
                .sliding_window_i32()
                .map_err(|error| CachePolicyError::Invalid(error.to_string()))?;
        }
        let validate_dimension = |dimension: NonZeroU32| {
            (dimension.get() <= i32::MAX as u32)
                .then_some(())
                .ok_or_else(|| {
                    CachePolicyError::Invalid(format!(
                        "prompt-cache layer dimension {dimension} exceeds the runtime i32 range"
                    ))
                })
        };
        match self {
            Self::NoState | Self::FixedState { .. } => {}
            Self::KeyValue {
                num_key_value_heads,
                head_dim,
                ..
            }
            | Self::KeyValueWithFixedState {
                num_key_value_heads,
                head_dim,
                ..
            } => {
                validate_dimension(*num_key_value_heads)?;
                validate_dimension(*head_dim)?;
            }
            Self::KeyOnly {
                num_key_heads,
                head_dim,
                ..
            }
            | Self::KeyOnlyWithFixedState {
                num_key_heads,
                head_dim,
                ..
            } => {
                validate_dimension(*num_key_heads)?;
                validate_dimension(*head_dim)?;
            }
            Self::CompressedLatentRotary {
                latent_dim,
                rotary_dim,
                ..
            } => {
                validate_dimension(*latent_dim)?;
                validate_dimension(*rotary_dim)?;
            }
        }
        let tensors = self.fixed_state();
        if tensors.is_empty()
            && matches!(
                self,
                Self::FixedState { .. }
                    | Self::KeyValueWithFixedState { .. }
                    | Self::KeyOnlyWithFixedState { .. }
            )
        {
            return Err(CachePolicyError::Invalid(
                "fixed-state cache policy must contain at least one tensor".into(),
            ));
        }
        validate_state_tensor_policies(tensors)
    }
}

impl StateTensorDimension {
    /// Constructs a positive fixed dimension.
    pub fn fixed(value: i32) -> Result<Self, CachePolicyError> {
        positive_u32(value, "fixed-state tensor dimension").map(Self::Fixed)
    }
}

impl StateTensorPolicy {
    /// Constructs and validates a state-tensor policy.
    pub fn new(
        role: StateTensorRole,
        shape: Vec<StateTensorDimension>,
        dtype: StateTensorDtype,
        residency: MutableStateResidency,
    ) -> Result<Self, CachePolicyError> {
        Self::new_with_residency(role, shape, dtype, residency.into())
    }

    /// Constructs state with an explicit pageable or mutable residency class.
    pub fn new_with_residency(
        role: StateTensorRole,
        shape: Vec<StateTensorDimension>,
        dtype: StateTensorDtype,
        residency: StateResidencyClass,
    ) -> Result<Self, CachePolicyError> {
        let policy = Self {
            role,
            shape,
            dtype,
            residency,
            presence: StateTensorPresence::Required,
        };
        validate_state_tensor_policies(std::slice::from_ref(&policy))?;
        Ok(policy)
    }

    /// Marks this tensor as optional.
    pub const fn optional(mut self) -> Self {
        self.presence = StateTensorPresence::Optional;
        self
    }

    /// Requires the tensor when the prefix has a non-zero remainder.
    pub const fn when_prefix_remainder_nonzero(mut self, divisor: NonZeroU32) -> Self {
        self.presence = StateTensorPresence::PrefixRemainderNonZero(divisor);
        self
    }

    /// Requires the tensor when the prefix contains a complete group.
    pub const fn when_prefix_at_least(mut self, divisor: NonZeroU32) -> Self {
        self.presence = StateTensorPresence::PrefixAtLeast(divisor);
        self
    }

    /// Returns whether the tensor is required for an exact prefix length.
    pub fn is_required_for(&self, prefix_tokens: usize) -> bool {
        match self.presence {
            StateTensorPresence::Required => true,
            StateTensorPresence::Optional => false,
            StateTensorPresence::PrefixRemainderNonZero(divisor) => {
                !prefix_tokens.is_multiple_of(divisor.get() as usize)
            }
            StateTensorPresence::PrefixAtLeast(divisor) => prefix_tokens >= divisor.get() as usize,
        }
    }

    /// Returns the unified residency classification.
    pub fn residency_class(&self) -> StateResidencyClass {
        self.residency
    }

    /// Resolves symbolic dimensions for an exact batch and prefix length.
    pub fn resolved_shape(
        &self,
        batch_size: usize,
        prefix_tokens: usize,
    ) -> Result<Vec<i32>, CachePolicyError> {
        self.shape
            .iter()
            .map(|dimension| match dimension {
                StateTensorDimension::Batch => i32::try_from(batch_size),
                StateTensorDimension::PrefixTokens => i32::try_from(prefix_tokens),
                StateTensorDimension::PrefixTokensDiv(divisor) => {
                    i32::try_from(prefix_tokens / divisor.get() as usize)
                }
                StateTensorDimension::PrefixTokensRem(divisor) => {
                    i32::try_from(prefix_tokens % divisor.get() as usize)
                }
                StateTensorDimension::Fixed(value) => i32::try_from(value.get()),
                StateTensorDimension::Scalar => Ok(1),
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| {
                CachePolicyError::Invalid(
                    "fixed-state tensor dimension exceeds runtime i32 range".into(),
                )
            })
    }

    /// Tests a stable serialized dtype name against this policy.
    pub fn accepts_dtype_name(&self, dtype: &str) -> bool {
        match self.dtype {
            StateTensorDtype::Floating => {
                matches!(dtype, "Float16" | "Bfloat16" | "Float32" | "Float64")
            }
            StateTensorDtype::Float32 => dtype == "Float32",
            StateTensorDtype::Int32 => dtype == "Int32",
            StateTensorDtype::Uint32 => dtype == "Uint32",
        }
    }
}

fn positive_u32(value: i32, field: &str) -> Result<NonZeroU32, CachePolicyError> {
    u32::try_from(value)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| {
            CachePolicyError::Invalid(format!(
                "prompt-cache {field} must be positive and fit u32, got {value}"
            ))
        })
}

fn validate_state_tensor_policies(tensors: &[StateTensorPolicy]) -> Result<(), CachePolicyError> {
    let mut roles = BTreeSet::new();
    for tensor in tensors {
        if !roles.insert(tensor.role) {
            return Err(CachePolicyError::Invalid(format!(
                "duplicate fixed-state tensor role {:?}",
                tensor.role
            )));
        }
        if tensor.shape.is_empty()
            || (tensor.shape.contains(&StateTensorDimension::Scalar)
                && tensor.shape.as_slice() != [StateTensorDimension::Scalar])
        {
            return Err(CachePolicyError::Invalid(format!(
                "invalid fixed-state tensor shape for role {:?}",
                tensor.role
            )));
        }
        let expected = match tensor.role {
            StateTensorRole::Recurrent => StateResidencyClass::LayerScopedOffloadable,
            StateTensorRole::Convolution { .. }
            | StateTensorRole::PrefixEmbedding
            | StateTensorRole::PositionDelta => StateResidencyClass::AlwaysDeviceMutable,
            StateTensorRole::Pooling {
                component: PoolingStateComponent::Pooled,
                ..
            } => StateResidencyClass::SealablePaged,
            StateTensorRole::Pooling { .. } => StateResidencyClass::AlwaysDeviceMutable,
        };
        if tensor.residency != expected {
            return Err(CachePolicyError::Invalid(format!(
                "fixed-state tensor role {:?} requires {:?} residency, got {:?}",
                tensor.role, expected, tensor.residency
            )));
        }
    }
    Ok(())
}

/// Invalid cache geometry or state-residency contract.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum CachePolicyError {
    /// A cache policy violates a structural or representational invariant.
    #[error("{0}")]
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_layer_and_fixed_state_contracts() {
        let recurrent = StateTensorPolicy::new(
            StateTensorRole::Recurrent,
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::fixed(16).unwrap(),
            ],
            StateTensorDtype::Floating,
            MutableStateResidency::LayerScopedOffloadable,
        )
        .unwrap();
        let layer = LayerCachePolicy::key_value_with_fixed_state(
            AttentionPolicy::sliding(128).unwrap(),
            8,
            64,
            vec![recurrent.clone()],
        )
        .unwrap();
        assert_eq!(
            layer.attention_residency_class(),
            Some(StateResidencyClass::SealablePaged)
        );
        assert_eq!(recurrent.resolved_shape(2, 9).unwrap(), vec![2, 16]);
        assert!(recurrent.accepts_dtype_name("Float16"));
        assert!(!recurrent.accepts_dtype_name("Int32"));
    }

    #[test]
    fn rejects_invalid_policy_without_a_backend() {
        assert!(LayerCachePolicy::key_value(AttentionPolicy::Full, 0, 64).is_err());
        assert!(StateTensorPolicy::new(
            StateTensorRole::Recurrent,
            vec![StateTensorDimension::Scalar, StateTensorDimension::Batch],
            StateTensorDtype::Floating,
            MutableStateResidency::LayerScopedOffloadable,
        )
        .is_err());
    }

    #[test]
    fn policy_schema_round_trips() {
        let policy = LayerCachePolicy::key_only(AttentionPolicy::Full, 4, 32).unwrap();
        let json = serde_json::to_string(&policy).unwrap();
        assert_eq!(
            serde_json::from_str::<LayerCachePolicy>(&json).unwrap(),
            policy
        );
    }
}
