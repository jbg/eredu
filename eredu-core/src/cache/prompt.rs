//! Portable reusable prompt-cache identity, catalog, and validation.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::attention::{AttentionPolicy, LayerSchedule};

use super::{
    CachePolicyError, CacheRankIdentity, CacheRepresentation, LayerCachePolicy, StateTensorOwner,
    StateTensorRole,
};

/// Current reusable prompt-cache schema version.
pub const PROMPT_CACHE_SCHEMA_VERSION: u32 = 8;

/// One named contiguous state range in a portable prompt-cache identity.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheStateSegment {
    id: String,
    layers: Range<usize>,
}

impl PromptCacheStateSegment {
    /// Creates a named non-empty local state range.
    pub fn new(id: impl Into<String>, layers: Range<usize>) -> Result<Self, PromptCacheError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(PromptCacheError::Malformed(
                "prompt-cache state segment identity must not be empty".into(),
            ));
        }
        if layers.is_empty() {
            return Err(PromptCacheError::Malformed(format!(
                "prompt-cache state segment {id:?} has an empty range"
            )));
        }
        Ok(Self { id, layers })
    }

    /// Returns the architecture-declared stable segment identity.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the segment's local range in the identity's ordered layout.
    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }
}

/// Caller-supplied identity and geometry for a reusable prefix cache.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct PromptCacheDescriptor {
    /// Stable architecture family.
    pub model_family: String,
    /// Effective normalized model type.
    pub effective_model_type: String,
    /// Caller-verified checkpoint identity.
    pub checkpoint_fingerprint: String,
    /// Identity of all content that produced the cached activations.
    pub prefix_content_fingerprint: String,
    /// Cache-relevant architecture identity.
    pub architecture_fingerprint: String,
    /// Total model layer count.
    pub layer_count: usize,
    /// Inclusive first global layer stored by this rank.
    pub global_layer_start: usize,
    /// Exclusive global layer boundary stored by this rank.
    pub global_layer_end: usize,
    /// Prefix batch size.
    pub batch_size: usize,
    /// Ordered cache layout for the owned layer range.
    pub layer_layout: LayerSchedule<LayerCachePolicy>,
    /// Per-layer processed-token delta relative to the persisted prefix.
    pub layer_prefix_offsets: Vec<i32>,
    /// Architecture-declared named ranges in the ordered state layout.
    pub state_segments: Vec<PromptCacheStateSegment>,
    /// Attention sink or pinned-prefix token count.
    pub sink_tokens: usize,
    /// Distributed rank-local layout.
    pub topology: PromptCacheTopology,
}

impl PromptCacheDescriptor {
    /// Validates the complete portable identity and cache geometry.
    pub fn validate(&self) -> Result<(), PromptCacheError> {
        IdentityLayout {
            layer_count: self.layer_count,
            global_layer_start: self.global_layer_start,
            global_layer_end: self.global_layer_end,
            batch_size: self.batch_size,
            layer_layout: &self.layer_layout,
            layer_prefix_offsets: &self.layer_prefix_offsets,
            state_segments: &self.state_segments,
            topology: &self.topology,
        }
        .validate("prompt-cache descriptor")
    }
}

/// Cache-relevant structure derived from a prepared model.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct PromptCacheModelIdentity {
    /// Stable architecture family.
    pub model_family: String,
    /// Effective normalized model type.
    pub effective_model_type: String,
    /// Cache-relevant architecture identity.
    pub architecture_fingerprint: String,
    /// Total model layer count.
    pub layer_count: usize,
    /// Inclusive first global layer owned by this model instance.
    pub global_layer_start: usize,
    /// Exclusive global layer boundary owned by this model instance.
    pub global_layer_end: usize,
    /// Attention sink or pinned-prefix token count.
    pub sink_tokens: usize,
    /// Distributed rank-local layout.
    pub topology: PromptCacheTopology,
    /// Ordered cache layout for the owned layer range.
    pub layer_layout: LayerSchedule<LayerCachePolicy>,
    /// Per-layer processed-token delta relative to the persisted prefix.
    pub layer_prefix_offsets: Vec<i32>,
    /// Architecture-declared named ranges in the ordered state layout.
    pub state_segments: Vec<PromptCacheStateSegment>,
}

impl PromptCacheModelIdentity {
    /// Builds an ordered ordinary key/value layout from runtime window values.
    pub fn key_value_layouts(
        sliding_windows: impl IntoIterator<Item = Option<i32>>,
        num_key_value_heads: i32,
        head_dim: i32,
    ) -> Result<LayerSchedule<LayerCachePolicy>, PromptCacheError> {
        let policies = sliding_windows
            .into_iter()
            .map(|window| {
                let attention = AttentionPolicy::from_sliding_window(window)
                    .map_err(|error| PromptCacheError::Malformed(error.to_string()))?;
                LayerCachePolicy::key_value(attention, num_key_value_heads, head_dim)
                    .map_err(PromptCacheError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        LayerSchedule::new(policies.len(), policies)
            .map_err(|error| PromptCacheError::Malformed(error.to_string()))
    }

    /// Builds a uniform compressed-latent layout.
    pub fn compressed_layouts(
        layer_count: usize,
        latent_dim: i32,
        rotary_dim: i32,
    ) -> Result<LayerSchedule<LayerCachePolicy>, PromptCacheError> {
        let policies = (0..layer_count)
            .map(|_| {
                LayerCachePolicy::compressed_latent_rotary(
                    AttentionPolicy::Full,
                    latent_dim,
                    rotary_dim,
                )
                .map_err(PromptCacheError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        LayerSchedule::new(layer_count, policies)
            .map_err(|error| PromptCacheError::Malformed(error.to_string()))
    }

    /// Validates the owned layer range and every policy.
    pub fn validate(&self) -> Result<(), PromptCacheError> {
        IdentityLayout {
            layer_count: self.layer_count,
            global_layer_start: self.global_layer_start,
            global_layer_end: self.global_layer_end,
            batch_size: 1,
            layer_layout: &self.layer_layout,
            layer_prefix_offsets: &self.layer_prefix_offsets,
            state_segments: &self.state_segments,
            topology: &self.topology,
        }
        .validate("loaded model")
    }

    /// Returns one architecture-declared state segment by stable identity.
    pub fn state_segment(&self, id: &str) -> Result<&PromptCacheStateSegment, PromptCacheError> {
        self.validate()?;
        self.state_segments
            .iter()
            .find(|segment| segment.id() == id)
            .ok_or_else(|| {
                PromptCacheError::Incompatible(format!(
                    "loaded model has no prompt-cache state segment {id:?}"
                ))
            })
    }

    /// Selects one named state segment as a validated standalone identity.
    pub fn select_state_segment(&self, id: &str) -> Result<Self, PromptCacheError> {
        let layers = self.state_segment(id)?.layers();
        let length = layers.len();
        let global_layer_start = self
            .global_layer_start
            .checked_add(layers.start)
            .ok_or_else(|| PromptCacheError::Malformed("state segment range overflowed".into()))?;
        let global_layer_end = global_layer_start
            .checked_add(length)
            .ok_or_else(|| PromptCacheError::Malformed("state segment range overflowed".into()))?;
        let layer_layout = LayerSchedule::new(
            length,
            self.layer_layout
                .iter()
                .skip(layers.start)
                .take(length)
                .cloned()
                .collect(),
        )
        .map_err(|error| PromptCacheError::Malformed(error.to_string()))?;
        let layer_prefix_offsets = self
            .layer_prefix_offsets
            .get(layers.clone())
            .ok_or_else(|| PromptCacheError::Malformed("state segment range is invalid".into()))?
            .to_vec();
        let selected = Self {
            model_family: self.model_family.clone(),
            effective_model_type: self.effective_model_type.clone(),
            architecture_fingerprint: self.architecture_fingerprint.clone(),
            layer_count: self.layer_count,
            global_layer_start,
            global_layer_end,
            sink_tokens: self.sink_tokens,
            topology: self.topology.clone(),
            layer_layout,
            layer_prefix_offsets,
            state_segments: vec![PromptCacheStateSegment::new(id, 0..length)?],
        };
        selected.validate()?;
        Ok(selected)
    }
}

struct IdentityLayout<'a> {
    layer_count: usize,
    global_layer_start: usize,
    global_layer_end: usize,
    batch_size: usize,
    layer_layout: &'a LayerSchedule<LayerCachePolicy>,
    layer_prefix_offsets: &'a [i32],
    state_segments: &'a [PromptCacheStateSegment],
    topology: &'a PromptCacheTopology,
}

impl IdentityLayout<'_> {
    fn validate(&self, subject: &str) -> Result<(), PromptCacheError> {
        let owned = self
            .global_layer_end
            .checked_sub(self.global_layer_start)
            .ok_or_else(|| {
                PromptCacheError::Incompatible(format!("{subject} has an invalid layer range"))
            })?;
        if self.layer_count == 0
            || self.global_layer_start >= self.global_layer_end
            || self.global_layer_end > self.layer_count
            || self.batch_size == 0
            || self.batch_size > i32::MAX as usize
            || self.layer_layout.len() != owned
            || self.layer_prefix_offsets.len() != owned
            || self.layer_prefix_offsets.iter().any(|offset| *offset > 0)
        {
            return Err(PromptCacheError::Incompatible(format!(
                "{subject} supplied {} cache layouts and {} layer prefix offsets for {owned} owned layers",
                self.layer_layout.len(),
                self.layer_prefix_offsets.len()
            )));
        }
        self.topology.validate()?;
        validate_state_segments(self.state_segments, owned)
            .map_err(|error| PromptCacheError::Incompatible(format!("{subject} {error}")))?;
        for policy in self.layer_layout.iter() {
            policy.validate()?;
        }
        Ok(())
    }
}

/// Verifies that a caller descriptor was derived from the prepared model.
pub fn validate_prompt_cache_model_identity(
    expected: &PromptCacheDescriptor,
    model: &PromptCacheModelIdentity,
) -> Result<(), PromptCacheError> {
    expected.validate()?;
    model.validate()?;
    macro_rules! require_equal {
        ($field:ident) => {
            if expected.$field != model.$field {
                return Err(PromptCacheError::Incompatible(format!(
                    "caller descriptor {} does not match the loaded model",
                    stringify!($field)
                )));
            }
        };
    }
    require_equal!(model_family);
    require_equal!(effective_model_type);
    require_equal!(architecture_fingerprint);
    require_equal!(layer_count);
    require_equal!(global_layer_start);
    require_equal!(global_layer_end);
    require_equal!(sink_tokens);
    require_equal!(topology);
    require_equal!(layer_layout);
    require_equal!(layer_prefix_offsets);
    require_equal!(state_segments);
    Ok(())
}

fn validate_state_segments(
    segments: &[PromptCacheStateSegment],
    owned: usize,
) -> Result<(), String> {
    if segments.is_empty() {
        return Err("has no named state segments".into());
    }
    let mut ids = BTreeSet::new();
    let mut next = 0;
    for segment in segments {
        if segment.id.trim().is_empty() {
            return Err("has an empty state segment identity".into());
        }
        if !ids.insert(segment.id.as_str()) {
            return Err(format!(
                "has duplicate state segment identity {:?}",
                segment.id
            ));
        }
        if segment.layers.start != next
            || segment.layers.end <= segment.layers.start
            || segment.layers.end > owned
        {
            return Err(format!(
                "state segment {:?} range {}..{} does not continue an exact partition of {owned} owned layers",
                segment.id, segment.layers.start, segment.layers.end
            ));
        }
        next = segment.layers.end;
    }
    if next != owned {
        return Err(format!(
            "state segments cover {next} of {owned} owned layers"
        ));
    }
    Ok(())
}

/// Rank-local topology recorded in a prompt-cache manifest.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheTopology {
    /// Pipeline world size and rank.
    pub pipeline: Option<(usize, usize)>,
    /// Tensor-parallel world size and rank.
    pub tensor_parallel: Option<(usize, usize)>,
    /// Expert-parallel world size and rank.
    pub expert_parallel: Option<(usize, usize)>,
    /// Whether attention state is replicated on the expert-parallel axis.
    pub expert_parallel_cache_replicated: bool,
}

impl Default for PromptCacheTopology {
    fn default() -> Self {
        Self {
            pipeline: None,
            tensor_parallel: None,
            expert_parallel: None,
            expert_parallel_cache_replicated: true,
        }
    }
}

impl PromptCacheTopology {
    /// Validates every optional world-size/rank pair.
    pub fn validate(&self) -> Result<(), PromptCacheError> {
        for (name, axis) in [
            ("pipeline", self.pipeline),
            ("tensor parallel", self.tensor_parallel),
            ("expert parallel", self.expert_parallel),
        ] {
            if axis.is_some_and(|(size, rank)| size == 0 || rank >= size) {
                return Err(PromptCacheError::Malformed(format!(
                    "invalid {name} topology"
                )));
            }
        }
        Ok(())
    }

    /// Returns the rank identity stored on cache blocks, if distributed.
    pub fn cache_rank_identity(&self) -> Option<CacheRankIdentity> {
        (self.pipeline.is_some()
            || self.tensor_parallel.is_some()
            || self.expert_parallel.is_some())
        .then(|| CacheRankIdentity {
            pipeline_rank: self.pipeline.map(|(_, rank)| rank),
            tensor_parallel_rank: self.tensor_parallel.map(|(_, rank)| rank),
            expert_parallel_rank: self.expert_parallel.map(|(_, rank)| rank),
        })
    }
}

/// Explicit publication behavior for a reusable prefix cache.
#[derive(Debug, Clone, Default)]
pub struct PromptCacheOptions {
    /// Optional application grouping label; never used for compatibility.
    pub application_namespace: Option<String>,
    /// Allows atomically replacing an existing destination.
    pub replace_existing: bool,
}

/// Versioned metadata inspectable without loading backend arrays.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheManifest {
    /// Persistence schema version.
    pub schema_version: u32,
    /// Model architecture family.
    pub model_family: String,
    /// Effective normalized model type.
    pub effective_model_type: String,
    /// Caller-selected checkpoint identity.
    pub checkpoint_fingerprint: String,
    /// Identity of all content that produced this prefix.
    pub prefix_content_fingerprint: String,
    /// Cache-relevant architecture identity.
    pub architecture_fingerprint: String,
    /// Total model layer count.
    pub layer_count: usize,
    /// Inclusive first global layer represented locally.
    pub global_layer_start: usize,
    /// Exclusive global layer boundary represented locally.
    pub global_layer_end: usize,
    /// Block size used by the producer.
    pub block_size_tokens: i32,
    /// Prefix batch size.
    pub batch_size: usize,
    /// Exact prefix token count.
    pub total_prefix_tokens: usize,
    /// SHA-256 over little-endian prefix token IDs.
    pub prefix_sha256: String,
    /// Ordered cache layout for the owned layer range.
    pub layer_layout: LayerSchedule<LayerCachePolicy>,
    /// Per-layer processed-token delta relative to the prefix.
    pub layer_prefix_offsets: Vec<i32>,
    /// Architecture-declared named ranges in the ordered state layout.
    pub state_segments: Vec<PromptCacheStateSegment>,
    /// Pinned prefix or sink token count.
    pub sink_tokens: usize,
    /// Distributed rank-local representation.
    pub topology: PromptCacheTopology,
    /// Optional non-authoritative application grouping label.
    pub application_namespace: Option<String>,
    /// Ordered immutable cache blocks.
    pub blocks: Vec<PromptCacheBlock>,
    /// Ordered fixed-size state tensors.
    pub state_tensors: Vec<PromptCacheStateTensor>,
}

/// One independently validated fixed-size state tensor catalog entry.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheStateTensor {
    /// Layer owner.
    pub owner: StateTensorOwner,
    /// Semantic role declared by the canonical layout.
    pub role: StateTensorRole,
    /// Safe relative backend shard path.
    pub shard: String,
    /// Array name within the shard.
    pub array: String,
    /// Exact stored shape.
    pub shape: Vec<i32>,
    /// Exact stored dtype.
    pub dtype: String,
    /// Logical bytes in the array.
    pub logical_bytes: u64,
    /// SHA-256 of the exact payload bytes.
    pub payload_sha256: String,
}

/// One cache block catalog entry in a prompt-cache manifest.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheBlock {
    /// Architecture-global layer identity.
    pub global_layer: usize,
    /// Stored attention representation.
    pub representation: CacheRepresentation,
    /// Inclusive absolute token position.
    pub start: i64,
    /// Exclusive absolute token position.
    pub end: i64,
    /// Optional rank identity.
    pub rank: Option<CacheRankIdentity>,
    /// Safe relative backend shard path.
    pub shard: String,
    /// First array name.
    pub first_array: String,
    /// Second array name.
    pub second_array: String,
    /// First array shape.
    pub first_shape: Vec<i32>,
    /// Second array shape.
    pub second_shape: Vec<i32>,
    /// First array dtype.
    pub first_dtype: String,
    /// Second array dtype.
    pub second_dtype: String,
    /// Logical bytes in both arrays.
    pub logical_bytes: u64,
    /// SHA-256 of the exact payload bytes.
    pub payload_sha256: String,
}

impl PromptCacheManifest {
    /// Validates all backend-independent schema, geometry, and coverage rules.
    pub fn validate(&self) -> Result<(), PromptCacheError> {
        if self.schema_version != PROMPT_CACHE_SCHEMA_VERSION {
            return Err(PromptCacheError::UnsupportedSchema(self.schema_version));
        }
        let owned = self.global_layer_end.checked_sub(self.global_layer_start);
        if self.prefix_content_fingerprint.is_empty()
            || self.block_size_tokens <= 0
            || self.layer_count == 0
            || self.global_layer_start >= self.global_layer_end
            || self.global_layer_end > self.layer_count
            || owned != Some(self.layer_layout.len())
            || owned != Some(self.layer_prefix_offsets.len())
            || self.batch_size == 0
            || self.batch_size > i32::MAX as usize
            || self.total_prefix_tokens == 0
            || !is_sha256_hex(&self.prefix_sha256)
        {
            return Err(PromptCacheError::Malformed(
                "invalid global cache dimensions".into(),
            ));
        }
        self.topology.validate()?;
        validate_state_segments(&self.state_segments, self.layer_layout.len())
            .map_err(PromptCacheError::Malformed)?;
        for (index, offset) in self.layer_prefix_offsets.iter().enumerate() {
            layer_prefix_tokens(self.total_prefix_tokens, *offset).map_err(|error| {
                PromptCacheError::Malformed(format!(
                    "invalid prefix frontier for global layer {}: {error}",
                    self.global_layer_start + index
                ))
            })?;
        }
        for (index, policy) in self.layer_layout.iter().enumerate() {
            policy.validate().map_err(|error| {
                PromptCacheError::Malformed(format!(
                    "invalid policy for global layer {}: {error}",
                    self.global_layer_start + index
                ))
            })?;
        }
        self.validate_blocks()?;
        self.validate_state_tensors()?;
        self.validate_coverage()
    }

    /// Validates compatibility with a caller descriptor and exact prefix IDs.
    pub fn validate_compatibility(
        &self,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<(), PromptCacheError> {
        self.validate()?;
        expected.validate()?;
        macro_rules! require_equal {
            ($field:ident) => {
                if self.$field != expected.$field {
                    return Err(PromptCacheError::Incompatible(format!(
                        "{} mismatch",
                        stringify!($field)
                    )));
                }
            };
        }
        require_equal!(model_family);
        require_equal!(effective_model_type);
        require_equal!(checkpoint_fingerprint);
        require_equal!(prefix_content_fingerprint);
        require_equal!(architecture_fingerprint);
        require_equal!(layer_count);
        require_equal!(global_layer_start);
        require_equal!(global_layer_end);
        require_equal!(batch_size);
        require_equal!(layer_layout);
        require_equal!(layer_prefix_offsets);
        require_equal!(state_segments);
        require_equal!(sink_tokens);
        require_equal!(topology);
        if self.total_prefix_tokens != prefix_token_ids.len()
            || self.prefix_sha256 != prompt_cache_token_fingerprint(prefix_token_ids)
        {
            return Err(PromptCacheError::PrefixIdentityMismatch);
        }
        Ok(())
    }

    fn validate_blocks(&self) -> Result<(), PromptCacheError> {
        let mut previous = None;
        for block in &self.blocks {
            let layer_index = block
                .global_layer
                .checked_sub(self.global_layer_start)
                .filter(|index| *index < self.layer_layout.len())
                .ok_or_else(|| {
                    PromptCacheError::Malformed(format!(
                        "cache block layer {} is outside the owned range",
                        block.global_layer
                    ))
                })?;
            let layer_tokens = layer_prefix_tokens(
                self.total_prefix_tokens,
                self.layer_prefix_offsets[layer_index],
            )?;
            if block.start < 0
                || block.end <= block.start
                || block.end > layer_tokens as i64
                || block.logical_bytes == 0
                || block.first_shape.is_empty()
                || block.second_shape.is_empty()
                || !is_sha256_hex(&block.payload_sha256)
                || !safe_relative_path(&block.shard)
            {
                return Err(PromptCacheError::Malformed(format!(
                    "invalid block at layer {} range {}..{}",
                    block.global_layer, block.start, block.end
                )));
            }
            let order = (block.global_layer, block.start, block.end);
            if previous.is_some_and(|value| value >= order) {
                return Err(PromptCacheError::Malformed(format!(
                    "prompt-cache blocks are reordered or duplicated at layer {} range {}..{}",
                    block.global_layer, block.start, block.end
                )));
            }
            previous = Some(order);
            let policy = self.layer_layout.get(layer_index).expect("bounded");
            let (representation, first_shape, second_shape) =
                block_geometry(policy, self.batch_size, block.end - block.start)?;
            if block.representation != representation
                || block.first_shape != first_shape
                || block.second_shape != second_shape
            {
                return Err(PromptCacheError::Malformed(format!(
                    "global layer {} payload geometry does not match its policy: actual {:?}/{:?}/{:?}, expected {:?}/{first_shape:?}/{second_shape:?}",
                    block.global_layer,
                    block.representation,
                    block.first_shape,
                    block.second_shape,
                    representation,
                )));
            }
            if block.rank != self.topology.cache_rank_identity() {
                return Err(PromptCacheError::Malformed(
                    "block rank identity does not match the recorded topology".into(),
                ));
            }
            let names = array_names(block.representation);
            if block.first_array != names.0
                || block.second_array != names.1
                || block.first_dtype != block.second_dtype
            {
                return Err(PromptCacheError::Malformed(
                    "block array names or dtypes do not match its representation".into(),
                ));
            }
        }
        Ok(())
    }

    fn validate_state_tensors(&self) -> Result<(), PromptCacheError> {
        let actual = self
            .state_tensors
            .iter()
            .map(|entry| (entry.owner, entry.role))
            .collect::<BTreeSet<_>>();
        if actual.len() != self.state_tensors.len() {
            return Err(PromptCacheError::Malformed(
                "fixed-state tensors contain duplicate owner/role entries".into(),
            ));
        }
        let mut expected = Vec::new();
        for (index, layer) in self.layer_layout.iter().enumerate() {
            let owner = StateTensorOwner::Layer(self.global_layer_start + index);
            let tokens =
                layer_prefix_tokens(self.total_prefix_tokens, self.layer_prefix_offsets[index])?;
            for policy in layer.fixed_state() {
                // A zero-token frontier has no materialized recurrent value,
                // even when that value is required once execution begins.
                if (tokens != 0 && policy.is_required_for(tokens))
                    || actual.contains(&(owner, policy.role))
                {
                    expected.push((owner, policy, tokens));
                }
            }
        }
        if self.state_tensors.len() != expected.len() {
            return Err(PromptCacheError::Malformed(format!(
                "fixed-state tensor count {} does not match layout count {}",
                self.state_tensors.len(),
                expected.len()
            )));
        }
        for (entry, (owner, policy, tokens)) in self.state_tensors.iter().zip(expected) {
            if entry.owner != owner
                || entry.role != policy.role
                || entry.shape != policy.resolved_shape(self.batch_size, tokens)?
                || !policy.accepts_dtype_name(&entry.dtype)
                || entry.logical_bytes == 0
                || !is_sha256_hex(&entry.payload_sha256)
                || entry.array != "state"
                || !safe_relative_path(&entry.shard)
            {
                return Err(PromptCacheError::Malformed(format!(
                    "fixed-state tensor {:?} for {:?} does not match its policy: shape {:?} and dtype {}, expected shape {:?}",
                    entry.role,
                    entry.owner,
                    entry.shape,
                    entry.dtype,
                    policy.resolved_shape(self.batch_size, tokens)?,
                )));
            }
        }
        Ok(())
    }

    fn validate_coverage(&self) -> Result<(), PromptCacheError> {
        let mut by_layer: BTreeMap<usize, Vec<&PromptCacheBlock>> = BTreeMap::new();
        for block in &self.blocks {
            by_layer.entry(block.global_layer).or_default().push(block);
        }
        for (index, policy) in self.layer_layout.iter().enumerate() {
            let layer = self.global_layer_start + index;
            let tokens =
                layer_prefix_tokens(self.total_prefix_tokens, self.layer_prefix_offsets[index])?;
            let mut blocks = by_layer.remove(&layer).unwrap_or_default();
            if policy.attention().is_none() {
                if !blocks.is_empty() {
                    return Err(PromptCacheError::Malformed(format!(
                        "stateless global layer {layer} has unexpected blocks"
                    )));
                }
                continue;
            }
            if blocks.is_empty() {
                if tokens == 0 {
                    continue;
                }
                return Err(PromptCacheError::Malformed(format!(
                    "missing blocks for global layer {layer}"
                )));
            }
            blocks.sort_by_key(|block| block.start);
            let required = required_persisted_start(policy, tokens)?;
            let mut end = blocks[0].start;
            if end > required
                || (matches!(policy.attention(), Some(AttentionPolicy::Full)) && end != 0)
            {
                return Err(PromptCacheError::Malformed(format!(
                    "global layer {layer} starts at {end}, but its policy requires history from {required}"
                )));
            }
            for block in blocks {
                if block.start != end {
                    return Err(PromptCacheError::Malformed(format!(
                        "gap or overlap at global layer {layer}: expected {end}, found {}",
                        block.start
                    )));
                }
                end = block.end;
            }
            if end != tokens as i64 {
                return Err(PromptCacheError::Malformed(format!(
                    "global layer {layer} ends at {end}, expected {tokens}"
                )));
            }
        }
        Ok(())
    }
}

fn block_geometry(
    policy: &LayerCachePolicy,
    batch_size: usize,
    token_count: i64,
) -> Result<(CacheRepresentation, Vec<i32>, Vec<i32>), PromptCacheError> {
    let batch = i32::try_from(batch_size)
        .map_err(|_| PromptCacheError::Malformed("prompt-cache batch exceeds i32".into()))?;
    let tokens = i32::try_from(token_count)
        .map_err(|_| PromptCacheError::Malformed("cache block token count exceeds i32".into()))?;
    match policy {
        LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => Err(
            PromptCacheError::Malformed("stateless layer has an attention payload".into()),
        ),
        LayerCachePolicy::KeyValue {
            num_key_value_heads,
            head_dim,
            ..
        }
        | LayerCachePolicy::KeyValueWithFixedState {
            num_key_value_heads,
            head_dim,
            ..
        } => {
            let shape = vec![
                batch,
                num_key_value_heads.get() as i32,
                tokens,
                head_dim.get() as i32,
            ];
            Ok((CacheRepresentation::KeyValue, shape.clone(), shape))
        }
        LayerCachePolicy::KeyOnly {
            num_key_heads,
            head_dim,
            ..
        }
        | LayerCachePolicy::KeyOnlyWithFixedState {
            num_key_heads,
            head_dim,
            ..
        } => Ok((
            CacheRepresentation::KeyValue,
            vec![
                batch,
                num_key_heads.get() as i32,
                tokens,
                head_dim.get() as i32,
            ],
            vec![batch, num_key_heads.get() as i32, tokens, 1],
        )),
        LayerCachePolicy::CompressedLatentRotary {
            latent_dim,
            rotary_dim,
            ..
        } => Ok((
            CacheRepresentation::CompressedLatentRotary,
            vec![batch, tokens, latent_dim.get() as i32],
            vec![batch, tokens, rotary_dim.get() as i32],
        )),
    }
}

fn required_persisted_start(
    policy: &LayerCachePolicy,
    total_prefix_tokens: usize,
) -> Result<i64, PromptCacheError> {
    let total = i64::try_from(total_prefix_tokens).map_err(|_| {
        PromptCacheError::Malformed("prompt-cache prefix length exceeds i64".into())
    })?;
    match policy.attention() {
        None | Some(AttentionPolicy::Full) => Ok(0),
        Some(AttentionPolicy::Sliding { window }) => {
            Ok((total - i64::from(window.get() - 1)).max(0))
        }
    }
}

fn layer_prefix_tokens(total: usize, offset: i32) -> Result<usize, PromptCacheError> {
    if offset > 0 {
        return Err(PromptCacheError::Malformed(
            "layer prefix offsets must not advance beyond the persisted prefix".into(),
        ));
    }
    total
        .checked_sub(offset.unsigned_abs() as usize)
        .ok_or_else(|| {
            PromptCacheError::Malformed(format!(
                "layer prefix offset {offset} precedes the start of a {total}-token prefix"
            ))
        })
}

fn array_names(representation: CacheRepresentation) -> (&'static str, &'static str) {
    match representation {
        CacheRepresentation::KeyValue => ("keys", "values"),
        CacheRepresentation::CompressedLatentRotary => ("latent", "rotary_key"),
    }
}

fn safe_relative_path(value: &str) -> bool {
    !value.is_empty()
        && !value.starts_with('/')
        && value
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
        && !value.contains('\\')
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Derives a stable cache architecture fingerprint from ordered semantic fields.
pub fn derive_prompt_cache_architecture_fingerprint<I, K, V>(
    model_family: &str,
    fields: I,
) -> String
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<String>,
{
    let mut fields = fields
        .into_iter()
        .map(|(key, value)| (key.into(), value.into()))
        .collect::<Vec<_>>();
    fields.sort_unstable();
    let mut hasher = Sha256::new();
    hash_component(&mut hasher, b"eredu-prompt-cache-architecture-v1");
    hash_component(&mut hasher, model_family.as_bytes());
    for (key, value) in fields {
        hash_component(&mut hasher, key.as_bytes());
        hash_component(&mut hasher, value.as_bytes());
    }
    format!("sha256:{}", hex(hasher.finalize()))
}

/// Hashes exact prefix token IDs as little-endian `u32` values.
pub fn prompt_cache_token_fingerprint(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    hex(hasher.finalize())
}

fn hash_component(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn hex(digest: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(digest.as_ref().len() * 2);
    for &byte in digest.as_ref() {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

/// Invalid reusable prompt-cache identity, schema, or catalog.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum PromptCacheError {
    /// A layer or state policy is invalid.
    #[error(transparent)]
    Policy(#[from] CachePolicyError),
    /// The persistence schema version is unsupported.
    #[error("unsupported prompt cache schema version {0}")]
    UnsupportedSchema(u32),
    /// The portable manifest structure is malformed.
    #[error("malformed prompt cache manifest: {0}")]
    Malformed(String),
    /// The prepared model or caller identity differs from the producer.
    #[error("incompatible prompt cache: {0}")]
    Incompatible(String),
    /// Exact prefix IDs differ from the persisted identity.
    #[error("prompt cache prefix token identity does not match")]
    PrefixIdentityMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> PromptCacheManifest {
        let layout = LayerSchedule::new(
            1,
            vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 2, 4).unwrap()],
        )
        .unwrap();
        PromptCacheManifest {
            schema_version: PROMPT_CACHE_SCHEMA_VERSION,
            model_family: "llama".into(),
            effective_model_type: "llama".into(),
            checkpoint_fingerprint: "checkpoint".into(),
            prefix_content_fingerprint: "content".into(),
            architecture_fingerprint: "architecture".into(),
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            block_size_tokens: 2,
            batch_size: 1,
            total_prefix_tokens: 2,
            prefix_sha256: prompt_cache_token_fingerprint(&[7, 8]),
            layer_layout: layout,
            layer_prefix_offsets: vec![0],
            state_segments: vec![PromptCacheStateSegment::new("state", 0..1).unwrap()],
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
            application_namespace: None,
            blocks: vec![PromptCacheBlock {
                global_layer: 0,
                representation: CacheRepresentation::KeyValue,
                start: 0,
                end: 2,
                rank: None,
                shard: "blocks/layer-0.safetensors".into(),
                first_array: "keys".into(),
                second_array: "values".into(),
                first_shape: vec![1, 2, 2, 4],
                second_shape: vec![1, 2, 2, 4],
                first_dtype: "Float16".into(),
                second_dtype: "Float16".into(),
                logical_bytes: 64,
                payload_sha256: "0".repeat(64),
            }],
            state_tensors: vec![],
        }
    }

    #[test]
    fn manifest_round_trips_and_validates_without_a_backend() {
        let manifest = manifest();
        manifest.validate().unwrap();
        let json = serde_json::to_string(&manifest).unwrap();
        let restored: PromptCacheManifest = serde_json::from_str(&json).unwrap();
        restored.validate().unwrap();
        assert_eq!(restored, manifest);
    }

    #[test]
    fn architecture_fingerprint_uses_the_eredu_domain() {
        let fingerprint = derive_prompt_cache_architecture_fingerprint(
            "llama",
            [("layers", "32"), ("hidden_size", "4096")],
        );
        assert_eq!(
            fingerprint,
            "sha256:9ee0b30ea8687d04eb4b65db3a58ccfff0a72bdd502805e9fdd6edb223ca5949"
        );
    }

    #[test]
    fn zero_frontier_prediction_state_needs_no_materialized_tensor() {
        let recurrent = crate::cache::StateTensorPolicy::new(
            StateTensorRole::Recurrent,
            vec![crate::cache::StateTensorDimension::Batch],
            crate::cache::StateTensorDtype::Floating,
            crate::cache::MutableStateResidency::LayerScopedOffloadable,
        )
        .unwrap();
        let mut value = manifest();
        value.total_prefix_tokens = 1;
        value.prefix_sha256 = prompt_cache_token_fingerprint(&[7]);
        value.layer_prefix_offsets = vec![-1];
        value.layer_layout = LayerSchedule::new(
            1,
            vec![LayerCachePolicy::fixed_only(vec![recurrent]).unwrap()],
        )
        .unwrap();
        value.blocks.clear();
        value.state_tensors.clear();
        value.validate().unwrap();
    }

    #[test]
    fn rejects_bad_topology_geometry_coverage_and_paths() {
        let mut value = manifest();
        value.topology.tensor_parallel = Some((1, 1));
        assert!(value.validate().is_err());
        let mut value = manifest();
        value.blocks[0].first_shape[2] = 1;
        assert!(value.validate().is_err());
        let mut value = manifest();
        value.blocks[0].shard = "../escape".into();
        assert!(value.validate().is_err());
    }

    #[test]
    fn identity_and_prefix_compatibility_fail_closed() {
        let manifest = manifest();
        let descriptor = PromptCacheDescriptor {
            model_family: manifest.model_family.clone(),
            effective_model_type: manifest.effective_model_type.clone(),
            checkpoint_fingerprint: manifest.checkpoint_fingerprint.clone(),
            prefix_content_fingerprint: manifest.prefix_content_fingerprint.clone(),
            architecture_fingerprint: manifest.architecture_fingerprint.clone(),
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            batch_size: 1,
            layer_layout: manifest.layer_layout.clone(),
            layer_prefix_offsets: vec![0],
            state_segments: manifest.state_segments.clone(),
            sink_tokens: 0,
            topology: PromptCacheTopology::default(),
        };
        manifest
            .validate_compatibility(&descriptor, &[7, 8])
            .unwrap();
        assert!(manifest
            .validate_compatibility(&descriptor, &[8, 7])
            .is_err());
        let mut renamed = descriptor.clone();
        renamed.state_segments = vec![PromptCacheStateSegment::new("renamed", 0..1).unwrap()];
        assert!(matches!(
            manifest.validate_compatibility(&renamed, &[7, 8]),
            Err(PromptCacheError::Incompatible(_))
        ));
        let mut invalid = descriptor;
        invalid.layer_prefix_offsets[0] = 1;
        assert!(matches!(
            invalid.validate(),
            Err(PromptCacheError::Incompatible(_))
        ));

        let mut malformed = manifest.clone();
        malformed.state_segments = vec![PromptCacheStateSegment::new("state", 0..2).unwrap()];
        assert!(matches!(
            malformed.validate(),
            Err(PromptCacheError::Malformed(_))
        ));
    }
}
