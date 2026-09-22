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
///
/// Version 9 rejects states produced before source-precision-preserving affine
/// materialization. Earlier metadata cannot distinguish that numerical policy
/// from the former unloaded-placeholder precision.
pub const PROMPT_CACHE_SCHEMA_VERSION: u32 = 9;

/// One named contiguous state range in a portable prompt-cache identity.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheStateSegment {
    id: String,
    layers: Range<usize>,
}

/// Existing prompt-cache diagnostic classes used by allocation-aware constructors.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PromptCacheDiagnosticKind {
    /// Invalid serialized or supplied identity geometry.
    Malformed,
    /// Identity does not describe the selected model range.
    Incompatible,
    /// An individual cache policy violates its contract.
    Policy,
}
impl PromptCacheDiagnosticKind {
    /// Attaches already-owned diagnostic text without formatting or copying it.
    pub fn into_error(self, text: String) -> PromptCacheError {
        match self {
            Self::Malformed => PromptCacheError::Malformed(text),
            Self::Incompatible => PromptCacheError::Incompatible(text),
            Self::Policy => PromptCacheError::Policy(CachePolicyError::Invalid(text)),
        }
    }
}

impl PromptCacheStateSegment {
    /// Creates a named non-empty local state range.
    pub fn new(id: impl Into<String>, layers: Range<usize>) -> Result<Self, PromptCacheError> {
        Self::new_with_diagnostic(id.into(), layers, |text| {
            PromptCacheError::Malformed(text.to_string())
        })
    }

    /// Validates the same segment after its final name has been constructed.
    pub fn new_with_diagnostic<E>(
        id: String,
        layers: Range<usize>,
        mut error: impl FnMut(std::fmt::Arguments<'_>) -> E,
    ) -> Result<Self, E> {
        if id.trim().is_empty() {
            return Err(error(format_args!(
                "prompt-cache state segment identity must not be empty"
            )));
        }
        if layers.is_empty() {
            return Err(error(format_args!(
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
    model_family: String,
    /// Effective normalized model type.
    effective_model_type: String,
    /// Caller-verified checkpoint identity.
    checkpoint_fingerprint: String,
    /// Identity of all content that produced the cached activations.
    prefix_content_fingerprint: String,
    /// Cache-relevant architecture identity.
    architecture_fingerprint: String,
    /// Total model layer count.
    layer_count: usize,
    /// Inclusive first global layer stored by this rank.
    global_layer_start: usize,
    /// Exclusive global layer boundary stored by this rank.
    global_layer_end: usize,
    /// Prefix batch size.
    batch_size: usize,
    /// Ordered cache layout for the owned layer range.
    layer_layout: LayerSchedule<LayerCachePolicy>,
    /// Per-layer processed-token delta relative to the persisted prefix.
    layer_prefix_offsets: Vec<i32>,
    /// Architecture-declared named ranges in the ordered state layout.
    state_segments: Vec<PromptCacheStateSegment>,
    /// Attention sink or pinned-prefix token count.
    sink_tokens: usize,
    /// Distributed rank-local layout.
    topology: PromptCacheTopology,
    /// Last distributed commit observation associated with the persisted state.
    distributed_commit: Option<crate::DistributedCommitOutcome>,
}

impl PromptCacheDescriptor {
    /// Creates and validates a complete reusable prefix-cache descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_family: impl Into<String>,
        effective_model_type: impl Into<String>,
        checkpoint_fingerprint: impl Into<String>,
        prefix_content_fingerprint: impl Into<String>,
        architecture_fingerprint: impl Into<String>,
        layer_count: usize,
        global_layer_start: usize,
        global_layer_end: usize,
        batch_size: usize,
        layer_layout: LayerSchedule<LayerCachePolicy>,
        layer_prefix_offsets: Vec<i32>,
        state_segments: Vec<PromptCacheStateSegment>,
        sink_tokens: usize,
        topology: PromptCacheTopology,
    ) -> Result<Self, PromptCacheError> {
        let descriptor = Self {
            model_family: model_family.into(),
            effective_model_type: effective_model_type.into(),
            checkpoint_fingerprint: checkpoint_fingerprint.into(),
            prefix_content_fingerprint: prefix_content_fingerprint.into(),
            architecture_fingerprint: architecture_fingerprint.into(),
            layer_count,
            global_layer_start,
            global_layer_end,
            batch_size,
            layer_layout,
            layer_prefix_offsets,
            state_segments,
            sink_tokens,
            topology,
            distributed_commit: None,
        };
        for value in [
            &descriptor.model_family,
            &descriptor.effective_model_type,
            &descriptor.checkpoint_fingerprint,
            &descriptor.prefix_content_fingerprint,
            &descriptor.architecture_fingerprint,
        ] {
            if value.trim().is_empty() {
                return Err(PromptCacheError::Malformed(
                    "prompt-cache identity strings must be non-empty".into(),
                ));
            }
        }
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Stable architecture family.
    pub fn model_family(&self) -> &str {
        &self.model_family
    }
    /// Effective normalized model type.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }
    /// Caller-verified checkpoint identity.
    pub fn checkpoint_fingerprint(&self) -> &str {
        &self.checkpoint_fingerprint
    }
    /// Prefix-content identity.
    pub fn prefix_content_fingerprint(&self) -> &str {
        &self.prefix_content_fingerprint
    }
    /// Cache-relevant architecture identity.
    pub fn architecture_fingerprint(&self) -> &str {
        &self.architecture_fingerprint
    }
    /// Total model layer count.
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }
    /// Inclusive first global layer stored by this rank.
    pub const fn global_layer_start(&self) -> usize {
        self.global_layer_start
    }
    /// Exclusive global layer boundary stored by this rank.
    pub const fn global_layer_end(&self) -> usize {
        self.global_layer_end
    }
    /// Prefix batch size.
    pub const fn batch_size(&self) -> usize {
        self.batch_size
    }
    /// Ordered cache layout.
    pub const fn layer_layout(&self) -> &LayerSchedule<LayerCachePolicy> {
        &self.layer_layout
    }
    /// Per-layer processed-token deltas.
    pub fn layer_prefix_offsets(&self) -> &[i32] {
        &self.layer_prefix_offsets
    }
    /// Named state-layout ranges.
    pub fn state_segments(&self) -> &[PromptCacheStateSegment] {
        &self.state_segments
    }
    /// Attention sink token count.
    pub const fn sink_tokens(&self) -> usize {
        self.sink_tokens
    }
    /// Distributed rank-local layout.
    pub const fn topology(&self) -> &PromptCacheTopology {
        &self.topology
    }
    /// Last distributed commit observation attached by the owning session.
    pub const fn distributed_commit(&self) -> Option<crate::DistributedCommitOutcome> {
        self.distributed_commit
    }
    /// Attaches the owning session's exact distributed commit observation.
    pub const fn with_distributed_commit(
        mut self,
        outcome: Option<crate::DistributedCommitOutcome>,
    ) -> Self {
        self.distributed_commit = outcome;
        self
    }
    /// Replaces the distributed topology and revalidates the descriptor.
    pub fn with_topology(
        mut self,
        topology: PromptCacheTopology,
    ) -> Result<Self, PromptCacheError> {
        self.topology = topology;
        self.validate()?;
        Ok(self)
    }
    /// Replaces the cache-relevant architecture fingerprint.
    pub fn with_architecture_fingerprint(
        mut self,
        architecture_fingerprint: impl Into<String>,
    ) -> Result<Self, PromptCacheError> {
        self.architecture_fingerprint = architecture_fingerprint.into();
        if self.architecture_fingerprint.trim().is_empty() {
            return Err(PromptCacheError::Malformed(
                "prompt-cache architecture fingerprint must be non-empty".into(),
            ));
        }
        self.validate()?;
        Ok(self)
    }
    /// Replaces the total model layer count while preserving the owned range.
    pub fn with_layer_count(mut self, layer_count: usize) -> Result<Self, PromptCacheError> {
        self.layer_count = layer_count;
        self.validate()?;
        Ok(self)
    }
    /// Derives every model-owned field from a prepared model identity.
    ///
    /// The checkpoint and prefix-content fingerprints remain caller-owned
    /// because they identify the concrete weights and processed input rather
    /// than model structure.
    pub fn from_model_identity(
        model: PromptCacheModelIdentity,
        checkpoint_fingerprint: impl Into<String>,
        prefix_content_fingerprint: impl Into<String>,
        batch_size: usize,
    ) -> Result<Self, PromptCacheError> {
        let descriptor = Self {
            model_family: model.model_family,
            effective_model_type: model.effective_model_type,
            checkpoint_fingerprint: checkpoint_fingerprint.into(),
            prefix_content_fingerprint: prefix_content_fingerprint.into(),
            architecture_fingerprint: model.architecture_fingerprint,
            layer_count: model.layer_count,
            global_layer_start: model.global_layer_start,
            global_layer_end: model.global_layer_end,
            batch_size,
            layer_layout: model.layer_layout,
            layer_prefix_offsets: model.layer_prefix_offsets,
            state_segments: model.state_segments,
            sink_tokens: model.sink_tokens,
            topology: model.topology,
            distributed_commit: None,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    /// Moves the validated identity and already-constructed payload declarations
    /// into their persistent manifest without cloning their owned allocations.
    /// The caller owns validation, allocation funding and publication authority.
    #[allow(clippy::too_many_arguments)]
    pub fn into_manifest(
        self,
        block_size_tokens: i32,
        total_prefix_tokens: usize,
        prefix_sha256: String,
        application_namespace: Option<String>,
        blocks: Vec<PromptCacheBlock>,
        state_tensors: Vec<PromptCacheStateTensor>,
    ) -> PromptCacheManifest {
        PromptCacheManifest {
            schema_version: PROMPT_CACHE_SCHEMA_VERSION,
            model_family: self.model_family,
            effective_model_type: self.effective_model_type,
            checkpoint_fingerprint: self.checkpoint_fingerprint,
            prefix_content_fingerprint: self.prefix_content_fingerprint,
            architecture_fingerprint: self.architecture_fingerprint,
            layer_count: self.layer_count,
            global_layer_start: self.global_layer_start,
            global_layer_end: self.global_layer_end,
            block_size_tokens,
            batch_size: self.batch_size,
            total_prefix_tokens,
            prefix_sha256,
            layer_layout: self.layer_layout,
            layer_prefix_offsets: self.layer_prefix_offsets,
            state_segments: self.state_segments,
            sink_tokens: self.sink_tokens,
            topology: self.topology,
            distributed_commit: self.distributed_commit,
            application_namespace,
            blocks,
            state_tensors,
        }
    }

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
    model_family: String,
    /// Effective normalized model type.
    effective_model_type: String,
    /// Cache-relevant architecture identity.
    architecture_fingerprint: String,
    /// Total model layer count.
    layer_count: usize,
    /// Inclusive first global layer owned by this model instance.
    global_layer_start: usize,
    /// Exclusive global layer boundary owned by this model instance.
    global_layer_end: usize,
    /// Attention sink or pinned-prefix token count.
    sink_tokens: usize,
    /// Distributed rank-local layout.
    topology: PromptCacheTopology,
    /// Ordered cache layout for the owned layer range.
    layer_layout: LayerSchedule<LayerCachePolicy>,
    /// Per-layer processed-token delta relative to the persisted prefix.
    layer_prefix_offsets: Vec<i32>,
    /// Architecture-declared named ranges in the ordered state layout.
    state_segments: Vec<PromptCacheStateSegment>,
}

impl PromptCacheModelIdentity {
    /// Creates and validates cache-relevant prepared-model identity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_family: impl Into<String>,
        effective_model_type: impl Into<String>,
        architecture_fingerprint: impl Into<String>,
        layer_count: usize,
        global_layer_start: usize,
        global_layer_end: usize,
        sink_tokens: usize,
        topology: PromptCacheTopology,
        layer_layout: LayerSchedule<LayerCachePolicy>,
        layer_prefix_offsets: Vec<i32>,
        state_segments: Vec<PromptCacheStateSegment>,
    ) -> Result<Self, PromptCacheError> {
        Self::new_with_diagnostic(
            model_family.into(),
            effective_model_type.into(),
            architecture_fingerprint.into(),
            layer_count,
            global_layer_start,
            global_layer_end,
            sink_tokens,
            topology,
            layer_layout,
            layer_prefix_offsets,
            state_segments,
            |kind, text| kind.into_error(text.to_string()),
        )
    }

    /// Consumes already-owned identity fields and validates them using the same
    /// policy worker. The caller supplies any failure-text allocation destination.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_diagnostic<E>(
        model_family: String,
        effective_model_type: String,
        architecture_fingerprint: String,
        layer_count: usize,
        global_layer_start: usize,
        global_layer_end: usize,
        sink_tokens: usize,
        topology: PromptCacheTopology,
        layer_layout: LayerSchedule<LayerCachePolicy>,
        layer_prefix_offsets: Vec<i32>,
        state_segments: Vec<PromptCacheStateSegment>,
        mut error: impl FnMut(PromptCacheDiagnosticKind, std::fmt::Arguments<'_>) -> E,
    ) -> Result<Self, E> {
        let identity = Self {
            model_family,
            effective_model_type,
            architecture_fingerprint,
            layer_count,
            global_layer_start,
            global_layer_end,
            sink_tokens,
            topology,
            layer_layout,
            layer_prefix_offsets,
            state_segments,
        };
        for value in [
            &identity.model_family,
            &identity.effective_model_type,
            &identity.architecture_fingerprint,
        ] {
            if value.trim().is_empty() {
                return Err(error(
                    PromptCacheDiagnosticKind::Malformed,
                    format_args!("prompt-cache model identity strings must be non-empty"),
                ));
            }
        }
        identity.validate_with_diagnostic(error)?;
        Ok(identity)
    }

    /// Stable architecture family.
    pub fn model_family(&self) -> &str {
        &self.model_family
    }
    /// Effective normalized model type.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }
    /// Cache-relevant architecture identity.
    pub fn architecture_fingerprint(&self) -> &str {
        &self.architecture_fingerprint
    }
    /// Total model layer count.
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }
    /// Inclusive first global layer owned locally.
    pub const fn global_layer_start(&self) -> usize {
        self.global_layer_start
    }
    /// Exclusive global layer boundary owned locally.
    pub const fn global_layer_end(&self) -> usize {
        self.global_layer_end
    }
    /// Attention sink token count.
    pub const fn sink_tokens(&self) -> usize {
        self.sink_tokens
    }
    /// Distributed rank-local layout.
    pub const fn topology(&self) -> &PromptCacheTopology {
        &self.topology
    }
    /// Ordered cache layout.
    pub const fn layer_layout(&self) -> &LayerSchedule<LayerCachePolicy> {
        &self.layer_layout
    }
    /// Per-layer processed-token deltas.
    pub fn layer_prefix_offsets(&self) -> &[i32] {
        &self.layer_prefix_offsets
    }
    /// Named state-layout ranges.
    pub fn state_segments(&self) -> &[PromptCacheStateSegment] {
        &self.state_segments
    }
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
        self.validate_with_diagnostic(|kind, text| kind.into_error(text.to_string()))
    }

    fn validate_with_diagnostic<E>(
        &self,
        error: impl FnMut(PromptCacheDiagnosticKind, std::fmt::Arguments<'_>) -> E,
    ) -> Result<(), E> {
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
        .validate_with("loaded model", error)
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
        self.validate_with(subject, |kind, text| kind.into_error(text.to_string()))
    }

    fn validate_with<E>(
        &self,
        subject: &str,
        mut error: impl FnMut(PromptCacheDiagnosticKind, std::fmt::Arguments<'_>) -> E,
    ) -> Result<(), E> {
        let owned = self
            .global_layer_end
            .checked_sub(self.global_layer_start)
            .ok_or_else(|| {
                error(
                    PromptCacheDiagnosticKind::Incompatible,
                    format_args!("{subject} has an invalid layer range"),
                )
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
            return Err(error(
                PromptCacheDiagnosticKind::Incompatible,
                format_args!(
                    "{subject} supplied {} cache layouts and {} layer prefix offsets for {owned} owned layers",
                    self.layer_layout.len(),
                    self.layer_prefix_offsets.len()
                ),
            ));
        }
        self.topology
            .validate_with_diagnostic(|text| error(PromptCacheDiagnosticKind::Malformed, text))?;
        validate_state_segments_with(self.state_segments, owned, |text| {
            error(
                PromptCacheDiagnosticKind::Incompatible,
                format_args!("{subject} {text}"),
            )
        })?;
        for policy in self.layer_layout.iter() {
            policy
                .validate_with_diagnostic(|text| error(PromptCacheDiagnosticKind::Policy, text))?;
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
    validate_state_segments_with(segments, owned, |text| text.to_string())
}

fn validate_state_segments_with<E>(
    segments: &[PromptCacheStateSegment],
    owned: usize,
    mut error: impl FnMut(std::fmt::Arguments<'_>) -> E,
) -> Result<(), E> {
    if segments.is_empty() {
        return Err(error(format_args!("has no named state segments")));
    }
    let mut next = 0;
    for (index, segment) in segments.iter().enumerate() {
        if segment.id.trim().is_empty() {
            return Err(error(format_args!("has an empty state segment identity")));
        }
        if segments[..index]
            .iter()
            .any(|earlier| earlier.id == segment.id)
        {
            return Err(error(format_args!(
                "has duplicate state segment identity {:?}",
                segment.id
            )));
        }
        if segment.layers.start != next
            || segment.layers.end <= segment.layers.start
            || segment.layers.end > owned
        {
            return Err(error(format_args!(
                "state segment {:?} range {}..{} does not continue an exact partition of {owned} owned layers",
                segment.id, segment.layers.start, segment.layers.end
            )));
        }
        next = segment.layers.end;
    }
    if next != owned {
        return Err(error(format_args!(
            "state segments cover {next} of {owned} owned layers"
        )));
    }
    Ok(())
}

/// Rank-local topology recorded in a prompt-cache manifest.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct PromptCacheTopology {
    /// Ordered-stage partition size and rank.
    stage: Option<(usize, usize)>,
    /// State-shard partition size and rank.
    shard: Option<(usize, usize)>,
    /// Addressable-group size and rank.
    addressable: Option<(usize, usize)>,
    /// Whether cache state is replicated on the addressable axis.
    addressable_state_replicated: bool,
}

impl Default for PromptCacheTopology {
    fn default() -> Self {
        Self {
            stage: None,
            shard: None,
            addressable: None,
            addressable_state_replicated: true,
        }
    }
}

impl PromptCacheTopology {
    /// Creates and validates an exact cache-placement topology.
    pub fn new(
        stage: Option<(usize, usize)>,
        shard: Option<(usize, usize)>,
        addressable: Option<(usize, usize)>,
        addressable_state_replicated: bool,
    ) -> Result<Self, PromptCacheError> {
        let topology = Self {
            stage,
            shard,
            addressable,
            addressable_state_replicated,
        };
        topology.validate()?;
        Ok(topology)
    }

    /// Returns ordered-stage size and rank when partitioned.
    pub const fn stage(&self) -> Option<(usize, usize)> {
        self.stage
    }

    /// Returns state-shard size and rank when partitioned.
    pub const fn shard(&self) -> Option<(usize, usize)> {
        self.shard
    }

    /// Returns addressable-group size and rank when partitioned.
    pub const fn addressable(&self) -> Option<(usize, usize)> {
        self.addressable
    }

    /// Returns whether cache state is replicated on the addressable axis.
    pub const fn addressable_state_replicated(&self) -> bool {
        self.addressable_state_replicated
    }

    /// Validates every optional world-size/rank pair.
    pub fn validate(&self) -> Result<(), PromptCacheError> {
        self.validate_with_diagnostic(|text| PromptCacheError::Malformed(text.to_string()))
    }

    /// Validates placement through the same allocation-free diagnostic callback.
    pub fn validate_with_diagnostic<E>(
        &self,
        mut error: impl FnMut(std::fmt::Arguments<'_>) -> E,
    ) -> Result<(), E> {
        for (name, axis) in [
            ("stage", self.stage),
            ("state shard", self.shard),
            ("addressable group", self.addressable),
        ] {
            if axis.is_some_and(|(size, rank)| size == 0 || rank >= size) {
                return Err(error(format_args!("invalid {name} topology")));
            }
        }
        Ok(())
    }

    /// Returns the rank identity stored on cache blocks, if distributed.
    pub fn cache_rank_identity(&self) -> Option<CacheRankIdentity> {
        (self.stage.is_some() || self.shard.is_some() || self.addressable.is_some()).then(|| {
            CacheRankIdentity::new(
                self.stage.map(|(_, rank)| rank),
                self.shard.map(|(_, rank)| rank),
                self.addressable.map(|(_, rank)| rank),
            )
        })
    }
}

/// Explicit publication behavior for a reusable prefix cache.
#[derive(Debug, Clone, Default)]
pub struct PromptCacheOptions {
    /// Optional application grouping label; never used for compatibility.
    application_namespace: Option<String>,
    /// Allows atomically replacing an existing destination.
    replace_existing: bool,
}

impl PromptCacheOptions {
    /// Creates validated prompt-cache publication options.
    pub fn new(
        application_namespace: Option<String>,
        replace_existing: bool,
    ) -> Result<Self, PromptCacheError> {
        if application_namespace
            .as_deref()
            .is_some_and(|namespace| namespace.trim().is_empty())
        {
            return Err(PromptCacheError::Malformed(
                "prompt-cache application namespace must not be empty".into(),
            ));
        }
        Ok(Self {
            application_namespace,
            replace_existing,
        })
    }

    /// Returns the optional application grouping label.
    pub fn application_namespace(&self) -> Option<&str> {
        self.application_namespace.as_deref()
    }

    /// Returns whether an existing destination may be replaced atomically.
    pub const fn replace_existing(&self) -> bool {
        self.replace_existing
    }
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
    /// Durable rank-local observation of the distributed commit epoch.
    #[serde(default)]
    pub distributed_commit: Option<crate::DistributedCommitOutcome>,
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

impl AsRef<PromptCacheManifest> for PromptCacheManifest {
    fn as_ref(&self) -> &PromptCacheManifest {
        self
    }
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
    PromptCacheArchitectureFingerprint::new(model_family, &mut fields).to_string()
}

/// Fixed digest of the exact sorted semantic fields used by prompt-cache identity.
/// Constructing and displaying this value do not allocate.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PromptCacheArchitectureFingerprint([u8; 32]);
impl PromptCacheArchitectureFingerprint {
    /// Fixed hashing and formatting controls, excluding caller-owned fields/output.
    pub const fn construction_bytes() -> usize {
        std::mem::size_of::<Sha256>()
            + std::mem::size_of::<Self>()
            + std::mem::size_of::<[u8; 32]>()
            + std::mem::size_of::<[u8; 8]>()
            + std::mem::size_of::<&str>()
            + std::mem::size_of::<&mut Sha256>()
            + std::mem::size_of::<u8>()
    }
    /// Hashes the same key-then-value ordering as the owned fingerprint helper.
    /// The caller supplies all field storage; duplicate fields remain significant.
    pub fn new<K: AsRef<str>, V: AsRef<str>>(model_family: &str, fields: &mut [(K, V)]) -> Self {
        fields.sort_unstable_by(|(left_key, left_value), (right_key, right_value)| {
            (left_key.as_ref(), left_value.as_ref())
                .cmp(&(right_key.as_ref(), right_value.as_ref()))
        });
        let mut hasher = Sha256::new();
        hash_component(&mut hasher, b"eredu-prompt-cache-architecture-v1");
        hash_component(&mut hasher, model_family.as_bytes());
        for (key, value) in fields {
            hash_component(&mut hasher, key.as_ref().as_bytes());
            hash_component(&mut hasher, value.as_ref().as_bytes());
        }
        Self(hasher.finalize().into())
    }
}
impl std::fmt::Display for PromptCacheArchitectureFingerprint {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("sha256:")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Hashes exact prefix token IDs as little-endian `u32` values.
pub fn prompt_cache_token_fingerprint(tokens: &[u32]) -> String {
    let mut encoded = String::with_capacity(64);
    assert!(prompt_cache_token_fingerprint_into(tokens, &mut encoded));
    encoded
}

/// Named hashing/hexadecimal controls for the fixed 64-byte token fingerprint.
/// The caller separately funds the destination before invoking the writer.
pub const fn prompt_cache_token_fingerprint_control_bytes() -> usize {
    std::mem::size_of::<Sha256>()
        + std::mem::size_of::<[u8; 32]>()
        + std::mem::size_of::<[u8; 4]>()
        + std::mem::size_of::<std::slice::Iter<'static, u32>>()
        + std::mem::size_of::<std::slice::Iter<'static, u8>>()
        + std::mem::size_of::<(&[u32], &mut String)>()
        + std::mem::size_of::<(u8, usize, char, bool)>()
}

/// Appends the ordinary token fingerprint into an already prepared destination.
/// Returns false before hashing or mutation unless 64 spare bytes are present.
/// This worker never grows the destination or acquires source authority.
pub fn prompt_cache_token_fingerprint_into(tokens: &[u32], encoded: &mut String) -> bool {
    if encoded.capacity().saturating_sub(encoded.len()) < 64 {
        return false;
    }
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    append_hex(hasher.finalize(), encoded);
    true
}

fn hash_component(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn hex(digest: impl AsRef<[u8]>) -> String {
    let mut encoded = String::with_capacity(digest.as_ref().len() * 2);
    append_hex(digest, &mut encoded);
    encoded
}

fn append_hex(digest: impl AsRef<[u8]>, encoded: &mut String) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for &byte in digest.as_ref() {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
}

/// Invalid reusable prompt-cache identity, schema, or catalog.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum PromptCacheError {
    /// The cache operation requires the exact prepared or committed input identity.
    #[error("prompt-cache operation requires the prepared-input semantic identity")]
    PreparedInputIdentityRequired,
    /// Loading requires paged state selected during model preparation.
    #[error("prompt-cache loading requires paged state selected during preparation")]
    PagedStateRequired,
    /// The selected local partition contains no cache state to persist or restore.
    #[error("this partition rank owns no prompt-cache state")]
    RankHasNoState,
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
            distributed_commit: None,
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
    fn shared_manifest_aliases_preserve_payload_and_both_metadata_accounts() {
        use crate::cache::PreparedPromptCacheManifest;
        use crate::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
        use std::sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc,
        };
        #[derive(Debug)]
        struct Account {
            spent: Arc<AtomicUsize>,
            retired: Arc<AtomicBool>,
        }
        impl HostMetadataAccount for Account {
            fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
                self.spent.fetch_add(bytes, Ordering::SeqCst);
                Ok(())
            }
        }
        impl Drop for Account {
            fn drop(&mut self) {
                self.retired.store(true, Ordering::SeqCst);
            }
        }
        let spent = Arc::new(AtomicUsize::new(0));
        let retired = Arc::new(AtomicBool::new(false));
        let dependency_retired = Arc::new(AtomicBool::new(false));
        let funding = HostMetadataFunding::new(Account {
            spent: spent.clone(),
            retired: retired.clone(),
        })
        .unwrap();
        let dependency = HostMetadataFunding::new(Account {
            spent: Arc::new(AtomicUsize::new(0)),
            retired: dependency_retired.clone(),
        })
        .unwrap();
        let before = spent.load(Ordering::SeqCst);
        let prepared =
            PreparedPromptCacheManifest::prepare_with_dependency(funding, dependency).unwrap();
        assert_eq!(
            spent.load(Ordering::SeqCst) - before,
            PreparedPromptCacheManifest::control_bytes().unwrap()
        );
        let original = manifest();
        let blocks = original.blocks.as_ptr();
        let encoded = serde_json::to_string(&original).unwrap();
        let owner = prepared.publish(original);
        let alias = owner.clone();
        assert!(owner.same_storage(&alias));
        assert_eq!(alias.blocks.as_ptr(), blocks);
        assert_eq!(serde_json::to_string(&alias).unwrap(), encoded);
        assert_eq!(
            spent.load(Ordering::SeqCst) - before,
            PreparedPromptCacheManifest::control_bytes().unwrap()
        );
        drop(owner);
        assert!(!retired.load(Ordering::SeqCst));
        assert!(!dependency_retired.load(Ordering::SeqCst));
        drop(alias);
        assert!(retired.load(Ordering::SeqCst));
        assert!(dependency_retired.load(Ordering::SeqCst));
    }

    #[test]
    fn legacy_materialization_cache_is_rejected_before_payload_validation() {
        let mut legacy = manifest();
        legacy.schema_version = 8;
        legacy.blocks[0].payload_sha256 = "not a payload digest".into();
        let encoded = serde_json::to_string(&legacy).unwrap();
        let decoded: PromptCacheManifest = serde_json::from_str(&encoded).unwrap();
        assert!(matches!(
            decoded.validate(),
            Err(PromptCacheError::UnsupportedSchema(8))
        ));
    }

    #[test]
    fn descriptor_derives_every_model_owned_field_from_identity() {
        let manifest = manifest();
        let identity = PromptCacheModelIdentity {
            model_family: manifest.model_family.clone(),
            effective_model_type: manifest.effective_model_type.clone(),
            architecture_fingerprint: manifest.architecture_fingerprint.clone(),
            layer_count: manifest.layer_count,
            global_layer_start: manifest.global_layer_start,
            global_layer_end: manifest.global_layer_end,
            sink_tokens: manifest.sink_tokens,
            topology: manifest.topology.clone(),
            layer_layout: manifest.layer_layout.clone(),
            layer_prefix_offsets: manifest.layer_prefix_offsets.clone(),
            state_segments: manifest.state_segments.clone(),
        };

        let descriptor = PromptCacheDescriptor::from_model_identity(
            identity.clone(),
            "caller-checkpoint",
            "caller-prefix-content",
            3,
        )
        .unwrap();

        validate_prompt_cache_model_identity(&descriptor, &identity).unwrap();
        assert_eq!(descriptor.checkpoint_fingerprint, "caller-checkpoint");
        assert_eq!(
            descriptor.prefix_content_fingerprint,
            "caller-prefix-content"
        );
        assert_eq!(descriptor.batch_size, 3);
        assert!(
            PromptCacheDescriptor::from_model_identity(identity, "checkpoint", "prefix", 0)
                .is_err()
        );
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
        value.topology.shard = Some((1, 1));
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
            distributed_commit: None,
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
