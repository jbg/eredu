//! Portable model capabilities, runtime-state accounting, and admission policy.

use crate::{
    cache::{
        LayerCachePolicy, StateTensorDimension, StateTensorDtype, StateTensorPolicy,
        StateTensorPresence, StateTensorRole,
    },
    AttentionPolicy, LayerSchedule, ObservationKind, Observed,
};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU8;

/// Model inputs accepted by a prepared model.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputModalities {
    /// Ordinary tokenizer IDs.
    pub text: bool,
    /// Prepared image inputs.
    pub image: bool,
    /// Prepared audio inputs.
    pub audio: bool,
    /// Prepared video inputs.
    pub video: bool,
}

impl InputModalities {
    /// Text-only input support.
    pub const TEXT: Self = Self {
        text: true,
        image: false,
        audio: false,
        video: false,
    };
}

/// Persistent decoder-state strategy used by a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum CacheStateStrategy {
    /// Ordinary full-context K/V attention.
    FullKv,
    /// Every attention cache is bounded by a sliding window.
    SlidingKv {
        /// Maximum retained positions per attention layer.
        window: u64,
    },
    /// Full-context and sliding-window attention layers.
    MixedKv {
        /// Number of full-context layers.
        full_layers: u64,
        /// Bounded layer counts grouped by retained window.
        sliding: Vec<SlidingWindowLayerCount>,
    },
    /// Full-context KV backing with layers that reuse earlier K/V state.
    SharedFullKv {
        /// Layers that allocate their own K/V state.
        cached_layers: u64,
        /// Layers that reuse K/V produced by an earlier layer.
        shared_layers: u64,
        /// Total full-attention layer count.
        full_attention_layers: u64,
        /// Sliding-mask layers, which do not bound KV allocation.
        sliding_attention: Vec<SlidingWindowLayerCount>,
    },
    /// Multi-head latent attention compressed state.
    CompressedMla {
        /// Compressed latent width per layer and position.
        latent_width: u64,
        /// Shared rotary-key width per layer and position.
        rotary_width: u64,
    },
    /// Attention combined with bounded convolution or recurrent state.
    HybridRecurrent {
        /// Full-context attention layer count.
        full_attention_layers: u64,
        /// Bounded attention layers grouped by exact window.
        sliding_attention: Vec<SlidingWindowLayerCount>,
        /// Recurrent/linear-attention layer count.
        recurrent_layers: u64,
    },
    /// Multimodal preparation feeding positions into a decoder strategy.
    Multimodal {
        /// Underlying decoder state.
        decoder: Box<CacheStateStrategy>,
        /// Whether media embeddings consume persistent decoder positions.
        media_consumes_decoder_positions: bool,
    },
}

/// Sliding-attention layer count sharing one retained window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlidingWindowLayerCount {
    /// Exact positive retained positions, including the current token.
    pub window: u64,
    /// Number of layers using this window.
    pub layers: u64,
}

/// Coverage of a runtime-state estimate.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimationCompleteness {
    /// Persistent request state is modeled exactly.
    Complete,
    /// The estimate is a complete safe upper bound.
    Conservative,
    /// Persistent state is covered but execution transients are not.
    PersistentStateOnly,
}

/// Capabilities derived from validated model configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Effective architecture/model type.
    pub model_type: String,
    /// Original trained context before a supported extension.
    pub native_max_context: Observed<u64>,
    /// Maximum positions accepted by the prepared model.
    pub effective_max_context: Observed<u64>,
    /// Persistent cache or recurrent-state model.
    pub state_strategy: CacheStateStrategy,
    /// Accepted input modalities.
    pub modalities: InputModalities,
    /// Runtime-state estimator coverage.
    pub estimation: EstimationCompleteness,
}

/// Accounting for a tokenized or backend-prepared input.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputTokenCount {
    /// Ordinary tokenizer IDs present in the input.
    pub text_tokens: u64,
    /// Positions inserted by prepared media.
    pub media_positions: u64,
    /// Total decoder/model positions consumed by prefill.
    pub model_positions: u64,
    /// Semantics of the model-position count.
    pub kind: ObservationKind,
    media_execution_workspace_bytes: u64,
    media_execution_workspace_kind: ObservationKind,
}

impl InputTokenCount {
    /// Creates an exact count for tokenized text.
    pub const fn text(tokens: u64) -> Self {
        Self {
            text_tokens: tokens,
            media_positions: 0,
            model_positions: tokens,
            kind: ObservationKind::Exact,
            media_execution_workspace_bytes: 0,
            media_execution_workspace_kind: ObservationKind::Exact,
        }
    }

    /// Creates an exact position count for backend-prepared input.
    pub const fn prepared(
        text_tokens: u64,
        media_positions: u64,
        model_positions: u64,
        media_execution_workspace_bytes: u64,
        media_execution_workspace_kind: ObservationKind,
    ) -> Self {
        Self {
            text_tokens,
            media_positions,
            model_positions,
            kind: ObservationKind::Exact,
            media_execution_workspace_bytes,
            media_execution_workspace_kind,
        }
    }

    /// Conservative media-tower workspace attributed to this input.
    pub const fn media_execution_workspace_bytes(&self) -> u64 {
        self.media_execution_workspace_bytes
    }

    /// Measurement semantics of the media workspace.
    pub const fn media_execution_workspace_kind(&self) -> ObservationKind {
        self.media_execution_workspace_kind
    }
}

/// Floating-dtype and request assumptions used by state estimation.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct StateMemoryAssumptions {
    /// Bytes per architecture-declared generic floating-state scalar.
    ///
    /// Fixed-dtype tensors use their own widths from the exact state policy.
    pub floating_state_dtype_bytes: NonZeroU8,
    /// Logical request batch size.
    pub batch_size: u64,
    /// Total requested positions, including output allowance.
    pub requested_positions: u64,
    /// Distinct sliding-window bounds in ascending order.
    pub sliding_window_bounds: Vec<u64>,
    /// Backing-array growth granularity for unbounded caches.
    pub allocation_granularity: u64,
}

/// Persistent and transient runtime-state estimate for one request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStateEstimate {
    /// Context-independent recurrent/convolution state.
    pub fixed_state_bytes: u64,
    /// Unbounded bytes added per position before multiplying by batch.
    pub bytes_per_position_per_batch: u64,
    /// Persistent context-dependent bytes at the requested length.
    pub context_state_bytes: u64,
    /// Prepared-media embedding bytes retained during prefill.
    pub multimodal_embedding_bytes: u64,
    /// Conservative media-tower execution workspace.
    pub media_execution_workspace_bytes: u64,
    /// Total modeled state for prompt plus output allowance.
    pub requested_state_bytes: u64,
    /// Estimator assumptions.
    pub assumptions: StateMemoryAssumptions,
    /// Estimator coverage.
    pub completeness: EstimationCompleteness,
}

/// Physical relationship between logical host and device tiers.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalMemorySemantics {
    /// Host and accelerator allocations share physical capacity.
    Unified,
    /// Host and accelerator memory are physically separate.
    SeparateTiers,
    /// The backend cannot determine the relationship.
    Unknown,
}

/// Static checkpoint and current residency observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticMemoryReport {
    /// Logical bytes in parameters or the complete residency plan.
    pub logical_parameter_bytes: Observed<u64>,
    /// Current logical host-resident bytes.
    pub current_host_resident_bytes: Observed<u64>,
    /// Current logical device-resident bytes.
    pub current_device_resident_bytes: Observed<u64>,
    /// Planned logical disk-backed bytes.
    pub planned_disk_backed_bytes: Observed<u64>,
    /// Process-global backend active allocation counter.
    pub backend_active_allocation_bytes: Observed<u64>,
    /// Process-global backend allocator-cache counter.
    pub backend_allocator_cache_bytes: Observed<u64>,
    /// Whether logical host/device tiers share physical capacity.
    pub physical_semantics: PhysicalMemorySemantics,
    /// Currently retained memory mappings.
    pub currently_mapped_shards: Observed<u64>,
}

/// System memory usable as an admission signal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvailableMemory {
    /// Unified/host physical memory.
    pub physical_memory_bytes: Observed<u64>,
    /// Defensible point-in-time availability estimate.
    pub available_memory_bytes: Observed<u64>,
    /// Physical tier semantics.
    pub physical_semantics: PhysicalMemorySemantics,
}

/// One pre-generation admission request.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct AdmissionRequest {
    /// Authoritative prompt accounting.
    pub input: InputTokenCount,
    /// Maximum generated-token allowance.
    pub max_output_tokens: u64,
    /// Logical batch size.
    pub batch_size: u64,
    /// Caller-selected reserve added to modeled state.
    pub safety_reserve_bytes: u64,
    /// Optional application budget for incremental state plus reserve.
    pub application_memory_budget_bytes: Option<u64>,
    /// Reject estimates that omit execution transients.
    pub require_complete_estimate: bool,
}

/// Detailed successful admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Admission {
    /// Prompt plus output allowance.
    pub requested_positions: u64,
    /// Runtime-state estimate.
    pub state: RuntimeStateEstimate,
    /// State plus caller reserve.
    pub incremental_required_bytes: u64,
    /// Availability signal used, when supplied.
    pub available_memory_bytes: Option<u64>,
}

/// Structured reason a request was rejected before generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AdmissionRejection {
    /// Prompt alone exceeds configured context.
    PromptExceedsContext {
        /// Prompt model positions.
        prompt_positions: u64,
        /// Effective model limit.
        maximum_positions: u64,
    },
    /// Prompt fits but output allowance does not.
    OutputHeadroomExceedsContext {
        /// Prompt model positions.
        prompt_positions: u64,
        /// Requested maximum output tokens.
        output_tokens: u64,
        /// Effective model limit.
        maximum_positions: u64,
    },
    /// Application budget is smaller than modeled state plus reserve.
    MemoryBudgetExceeded {
        /// Required incremental bytes.
        required_bytes: u64,
        /// Caller-supplied budget.
        budget_bytes: u64,
    },
    /// Current availability is smaller than modeled state plus reserve.
    InsufficientAvailableMemory {
        /// Required incremental bytes.
        required_bytes: u64,
        /// Observed available bytes.
        available_bytes: u64,
    },
    /// A requested availability check could not be performed.
    AvailableMemoryUnavailable {
        /// Platform report detail.
        reason: String,
    },
    /// Policy requires estimator coverage the model cannot provide.
    EstimationUnsupported {
        /// Coverage detail.
        reason: String,
    },
}

/// Admission outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AdmissionResult {
    /// Request may proceed.
    Admitted(Admission),
    /// Request was rejected.
    Rejected(AdmissionRejection),
}

/// Structured capability and accounting failures.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum CapabilityError {
    /// A validated architecture exposed an invalid value.
    #[error("invalid model capability field {field}: {detail}")]
    InvalidConfiguration {
        /// Invalid field name.
        field: &'static str,
        /// Invalid-value detail.
        detail: String,
    },
    /// Checked byte or position arithmetic overflowed.
    #[error("capability arithmetic overflow while computing {operation}")]
    ArithmeticOverflow {
        /// Stable operation label.
        operation: &'static str,
    },
    /// Prepared input does not match the loaded architecture.
    #[error("unsupported prepared input for {architecture}: {reason}")]
    UnsupportedInput {
        /// Effective architecture name.
        architecture: String,
        /// Unsupported-input detail.
        reason: String,
    },
    /// A runtime observation could not be obtained.
    #[error("capability observation failed: {0}")]
    Observation(String),
}

/// Memory-accounting view of an executable runtime-state layout.
///
/// The ordered layer policies are copied directly from the architecture's
/// executable [`LayerSchedule`]. They are intentionally not summarized into a
/// second scalar geometry, so execution and admission share one semantic
/// source for every state-bearing layer and component.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct StateMemoryLayout {
    /// Exact ordered executable state policies.
    layer_layout: LayerSchedule<LayerCachePolicy>,
    /// Executable processed-token offset for every state layer.
    layer_prefix_offsets: Vec<i32>,
    /// Model hidden width used by retained media embeddings.
    pub hidden_size: u64,
    /// Allocation granularity for unbounded caches.
    pub allocation_granularity: u64,
    /// Coverage supplied by the layout.
    pub completeness: EstimationCompleteness,
}

impl StateMemoryLayout {
    /// Creates accounting metadata around an exact executable layer schedule.
    pub fn new(
        layer_layout: LayerSchedule<LayerCachePolicy>,
        layer_prefix_offsets: Vec<i32>,
        hidden_size: u64,
        allocation_granularity: u64,
        completeness: EstimationCompleteness,
    ) -> Result<Self, CapabilityError> {
        if layer_layout.is_empty()
            || layer_prefix_offsets.len() != layer_layout.len()
            || layer_prefix_offsets.iter().any(|offset| *offset > 0)
            || hidden_size == 0
            || allocation_granularity == 0
        {
            let (field, detail) = if layer_layout.is_empty() {
                (
                    "layer_layout",
                    "must contain at least one executable state layer",
                )
            } else if layer_prefix_offsets.len() != layer_layout.len() {
                (
                    "layer_prefix_offsets",
                    "must contain one entry per executable state layer",
                )
            } else if layer_prefix_offsets.iter().any(|offset| *offset > 0) {
                (
                    "layer_prefix_offsets",
                    "must not advance beyond the request token frontier",
                )
            } else if hidden_size == 0 {
                ("hidden_size", "must be positive")
            } else {
                ("allocation_granularity", "must be positive")
            };
            return Err(CapabilityError::InvalidConfiguration {
                field,
                detail: detail.into(),
            });
        }
        for (layer, policy) in layer_layout.iter().enumerate() {
            policy
                .validate()
                .map_err(|error| CapabilityError::InvalidConfiguration {
                    field: "layer_layout",
                    detail: format!("invalid state policy at layer {layer}: {error}"),
                })?;
        }
        Ok(Self {
            layer_layout,
            layer_prefix_offsets,
            hidden_size,
            allocation_granularity,
            completeness,
        })
    }

    /// Borrows the exact ordered executable state policies.
    pub const fn layer_layout(&self) -> &LayerSchedule<LayerCachePolicy> {
        &self.layer_layout
    }

    /// Returns executable processed-token offsets in state-layer order.
    pub fn layer_prefix_offsets(&self) -> &[i32] {
        &self.layer_prefix_offsets
    }
}

fn checked_add(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_add(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn checked_mul(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_mul(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn attention_scalars_per_position(policy: &LayerCachePolicy) -> Result<u64, CapabilityError> {
    let scalars = match policy {
        LayerCachePolicy::KeyValue {
            num_key_value_heads,
            head_dim,
            ..
        }
        | LayerCachePolicy::KeyValueWithFixedState {
            num_key_value_heads,
            head_dim,
            ..
        } => checked_mul(
            checked_mul(
                u64::from(num_key_value_heads.get()),
                u64::from(head_dim.get()),
                "key/value heads times head dimension",
            )?,
            2,
            "key plus value scalars",
        )?,
        LayerCachePolicy::KeyOnly {
            num_key_heads,
            head_dim,
            ..
        }
        | LayerCachePolicy::KeyOnlyWithFixedState {
            num_key_heads,
            head_dim,
            ..
        } => checked_mul(
            u64::from(num_key_heads.get()),
            u64::from(head_dim.get()),
            "key heads times head dimension",
        )?,
        LayerCachePolicy::CompressedLatentRotary {
            latent_dim,
            rotary_dim,
            ..
        } => checked_add(
            u64::from(latent_dim.get()),
            u64::from(rotary_dim.get()),
            "compressed latent plus rotary width",
        )?,
        LayerCachePolicy::NoState | LayerCachePolicy::FixedState { .. } => 0,
    };
    Ok(scalars)
}

fn is_context_dependent_dimension(dimension: &StateTensorDimension) -> bool {
    matches!(
        dimension,
        StateTensorDimension::PrefixTokens
            | StateTensorDimension::PrefixTokensDiv(_)
            | StateTensorDimension::PrefixTokensRem(_)
    )
}

fn state_tensor_dtype_bytes(tensor: &StateTensorPolicy, floating_scalar_bytes: u64) -> u64 {
    match tensor.dtype {
        StateTensorDtype::Floating => floating_scalar_bytes,
        StateTensorDtype::Float32 | StateTensorDtype::Int32 | StateTensorDtype::Uint32 => 4,
    }
}

fn state_tensor_is_present(tensor: &StateTensorPolicy, prefix_tokens: usize) -> bool {
    match tensor.presence {
        StateTensorPresence::Required => true,
        // Prepared prefix embeddings are accounted once through the input's
        // authoritative media-position count below. Any other optional state
        // is included conservatively because its request-time presence is not
        // otherwise represented in the portable input descriptor.
        StateTensorPresence::Optional => !matches!(tensor.role, StateTensorRole::PrefixEmbedding),
        StateTensorPresence::PrefixRemainderNonZero(divisor) => {
            !prefix_tokens.is_multiple_of(divisor.get() as usize)
        }
        StateTensorPresence::PrefixAtLeast(divisor) => prefix_tokens >= divisor.get() as usize,
    }
}

fn state_tensor_bytes(
    tensor: &StateTensorPolicy,
    batch_size: usize,
    prefix_tokens: usize,
    floating_scalar_bytes: u64,
) -> Result<u64, CapabilityError> {
    if !state_tensor_is_present(tensor, prefix_tokens) {
        return Ok(0);
    }
    let shape = tensor
        .resolved_shape(batch_size, prefix_tokens)
        .map_err(|error| CapabilityError::InvalidConfiguration {
            field: "layer_layout",
            detail: error.to_string(),
        })?;
    let scalars = shape.into_iter().try_fold(1_u64, |scalars, dimension| {
        checked_mul(
            scalars,
            u64::try_from(dimension).map_err(|_| CapabilityError::InvalidConfiguration {
                field: "layer_layout",
                detail: "runtime state tensor has a negative resolved dimension".into(),
            })?,
            "runtime state tensor scalar count",
        )
    })?;
    checked_mul(
        scalars,
        state_tensor_dtype_bytes(tensor, floating_scalar_bytes),
        "runtime state tensor bytes",
    )
}

fn state_tensor_bytes_per_position_per_batch(
    tensor: &StateTensorPolicy,
    floating_scalar_bytes: u64,
) -> Result<u64, CapabilityError> {
    let mut scalars = 1_u64;
    let mut divisor = 1_u64;
    let mut unbounded = false;
    for dimension in &tensor.shape {
        match dimension {
            StateTensorDimension::Batch | StateTensorDimension::Scalar => {}
            StateTensorDimension::Fixed(value) => {
                scalars =
                    checked_mul(scalars, u64::from(value.get()), "state growth scalar count")?;
            }
            StateTensorDimension::PrefixTokens => unbounded = true,
            StateTensorDimension::PrefixTokensDiv(value) => {
                unbounded = true;
                divisor = checked_mul(divisor, u64::from(value.get()), "state growth divisor")?;
            }
            StateTensorDimension::PrefixTokensRem(_) => return Ok(0),
        }
    }
    if !unbounded {
        return Ok(0);
    }
    let bytes = checked_mul(
        scalars,
        state_tensor_dtype_bytes(tensor, floating_scalar_bytes),
        "state growth bytes",
    )?;
    Ok(bytes.div_ceil(divisor))
}

/// Estimates request state from exact executable layer policies.
pub fn estimate_runtime_state(
    layout: &StateMemoryLayout,
    input: InputTokenCount,
    max_output_tokens: u64,
    batch_size: u64,
    floating_state_dtype_bytes: NonZeroU8,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    if batch_size == 0 {
        return Err(CapabilityError::InvalidConfiguration {
            field: "batch_size",
            detail: "must be positive".into(),
        });
    }
    let requested_positions = checked_add(
        input.model_positions,
        max_output_tokens,
        "prompt plus output positions",
    )?;
    let floating_scalar_bytes = u64::from(floating_state_dtype_bytes.get());
    let batch_size_usize =
        usize::try_from(batch_size).map_err(|_| CapabilityError::InvalidConfiguration {
            field: "batch_size",
            detail: "exceeds the runtime state shape range".into(),
        })?;
    let mut fixed_state_bytes = 0;
    let mut context_state_bytes = 0;
    let mut unbounded_per_position = 0;
    let mut sliding_window_bounds = Vec::new();
    for (layer, policy) in layout.layer_layout.iter().enumerate() {
        let layer_positions = requested_positions
            .saturating_sub(u64::from(layout.layer_prefix_offsets[layer].unsigned_abs()));
        let layer_positions_usize = usize::try_from(layer_positions).map_err(|_| {
            CapabilityError::InvalidConfiguration {
                field: "requested_positions",
                detail: "exceeds the runtime state shape range".into(),
            }
        })?;
        if let Some(attention) = policy.attention() {
            let per_position = attention_scalars_per_position(policy)?;
            let retained = match attention {
                AttentionPolicy::Sliding { window } => {
                    let window = u64::from(window.get());
                    sliding_window_bounds.push(window);
                    layer_positions.min(window)
                }
                AttentionPolicy::Full => {
                    let adjustment = layout.allocation_granularity - 1;
                    checked_add(layer_positions, adjustment, "cache allocation rounding")?
                        / layout.allocation_granularity
                        * layout.allocation_granularity
                }
            };
            let bytes = checked_mul(
                checked_mul(
                    checked_mul(per_position, retained, "attention context scalars")?,
                    batch_size,
                    "attention context batch",
                )?,
                floating_scalar_bytes,
                "attention context bytes",
            )?;
            context_state_bytes =
                checked_add(context_state_bytes, bytes, "context state byte total")?;
            if matches!(attention, AttentionPolicy::Full) {
                unbounded_per_position = checked_add(
                    unbounded_per_position,
                    checked_mul(
                        per_position,
                        floating_scalar_bytes,
                        "unbounded bytes per position",
                    )?,
                    "unbounded bytes-per-position total",
                )?;
            }
        }
        for tensor in policy.fixed_state() {
            let bytes = state_tensor_bytes(
                tensor,
                batch_size_usize,
                layer_positions_usize,
                floating_scalar_bytes,
            )?;
            if tensor.shape.iter().any(is_context_dependent_dimension) {
                context_state_bytes =
                    checked_add(context_state_bytes, bytes, "context state byte total")?;
                unbounded_per_position = checked_add(
                    unbounded_per_position,
                    state_tensor_bytes_per_position_per_batch(tensor, floating_scalar_bytes)?,
                    "unbounded bytes-per-position total",
                )?;
            } else {
                fixed_state_bytes =
                    checked_add(fixed_state_bytes, bytes, "fixed state byte total")?;
            }
        }
    }
    sliding_window_bounds.sort_unstable();
    sliding_window_bounds.dedup();
    let multimodal_embedding_bytes = checked_mul(
        checked_mul(
            checked_mul(
                input.media_positions,
                layout.hidden_size,
                "media positions times hidden size",
            )?,
            batch_size,
            "media embeddings times batch",
        )?,
        floating_scalar_bytes,
        "media embedding bytes",
    )?;
    let media_execution_workspace_bytes = checked_mul(
        input.media_execution_workspace_bytes,
        batch_size,
        "media execution workspace times batch",
    )?;
    let requested_state_bytes = checked_add(
        checked_add(
            checked_add(
                fixed_state_bytes,
                context_state_bytes,
                "fixed plus context state",
            )?,
            multimodal_embedding_bytes,
            "persistent plus multimodal embedding state",
        )?,
        media_execution_workspace_bytes,
        "persistent plus media execution workspace",
    )?;
    let completeness = if input.media_positions == 0
        || input.media_execution_workspace_kind == ObservationKind::Exact
    {
        layout.completeness
    } else {
        EstimationCompleteness::Conservative
    };
    Ok(RuntimeStateEstimate {
        fixed_state_bytes,
        bytes_per_position_per_batch: unbounded_per_position,
        context_state_bytes,
        multimodal_embedding_bytes,
        media_execution_workspace_bytes,
        requested_state_bytes,
        assumptions: StateMemoryAssumptions {
            floating_state_dtype_bytes,
            batch_size,
            requested_positions,
            sliding_window_bounds,
            allocation_granularity: layout.allocation_granularity,
        },
        completeness,
    })
}

/// Applies context and memory policy to an already-computed state estimate.
pub fn apply_admission_policy(
    capabilities: &ModelCapabilities,
    request: AdmissionRequest,
    state: RuntimeStateEstimate,
    available: Option<&AvailableMemory>,
) -> Result<AdmissionResult, CapabilityError> {
    let maximum = match &capabilities.effective_max_context {
        Observed::Available { value, .. } => *value,
        Observed::Unsupported { reason } | Observed::Unavailable { reason } => {
            return Ok(AdmissionResult::Rejected(
                AdmissionRejection::EstimationUnsupported {
                    reason: reason.clone(),
                },
            ));
        }
    };
    if request.input.model_positions > maximum {
        return Ok(AdmissionResult::Rejected(
            AdmissionRejection::PromptExceedsContext {
                prompt_positions: request.input.model_positions,
                maximum_positions: maximum,
            },
        ));
    }
    let requested_positions = checked_add(
        request.input.model_positions,
        request.max_output_tokens,
        "admission prompt plus output",
    )?;
    if requested_positions > maximum {
        return Ok(AdmissionResult::Rejected(
            AdmissionRejection::OutputHeadroomExceedsContext {
                prompt_positions: request.input.model_positions,
                output_tokens: request.max_output_tokens,
                maximum_positions: maximum,
            },
        ));
    }
    if request.require_complete_estimate
        && state.completeness == EstimationCompleteness::PersistentStateOnly
    {
        return Ok(AdmissionResult::Rejected(
            AdmissionRejection::EstimationUnsupported {
                reason: format!(
                    "architecture estimator coverage is {:?}",
                    state.completeness
                ),
            },
        ));
    }
    let incremental_required_bytes = checked_add(
        state.requested_state_bytes,
        request.safety_reserve_bytes,
        "state plus safety reserve",
    )?;
    if let Some(budget_bytes) = request.application_memory_budget_bytes {
        if incremental_required_bytes > budget_bytes {
            return Ok(AdmissionResult::Rejected(
                AdmissionRejection::MemoryBudgetExceeded {
                    required_bytes: incremental_required_bytes,
                    budget_bytes,
                },
            ));
        }
    }
    let available_memory_bytes = match available {
        Some(report) => match &report.available_memory_bytes {
            Observed::Available { value, .. } => Some(*value),
            Observed::Unsupported { reason } | Observed::Unavailable { reason } => {
                return Ok(AdmissionResult::Rejected(
                    AdmissionRejection::AvailableMemoryUnavailable {
                        reason: reason.clone(),
                    },
                ))
            }
        },
        None => None,
    };
    if let Some(available_bytes) = available_memory_bytes {
        if incremental_required_bytes > available_bytes {
            return Ok(AdmissionResult::Rejected(
                AdmissionRejection::InsufficientAvailableMemory {
                    required_bytes: incremental_required_bytes,
                    available_bytes,
                },
            ));
        }
    }
    Ok(AdmissionResult::Admitted(Admission {
        requested_positions,
        state,
        incremental_required_bytes,
        available_memory_bytes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_estimation_and_admission_are_backend_independent() {
        let policies = (0..2)
            .map(|_| LayerCachePolicy::key_only(AttentionPolicy::Full, 1, 8).unwrap())
            .collect::<Vec<_>>();
        let layout = StateMemoryLayout::new(
            LayerSchedule::new(2, policies).unwrap(),
            vec![0; 2],
            32,
            8,
            EstimationCompleteness::Complete,
        )
        .unwrap();
        let input = InputTokenCount::text(5);
        let state =
            estimate_runtime_state(&layout, input, 2, 1, NonZeroU8::new(4).unwrap()).unwrap();
        assert_eq!(state.assumptions.requested_positions, 7);
        assert_eq!(state.context_state_bytes, 512);
        let capabilities = ModelCapabilities {
            model_type: "mock".into(),
            native_max_context: Observed::exact(16, "mock"),
            effective_max_context: Observed::exact(16, "mock"),
            state_strategy: CacheStateStrategy::FullKv,
            modalities: InputModalities::TEXT,
            estimation: EstimationCompleteness::Complete,
        };
        assert!(matches!(
            apply_admission_policy(
                &capabilities,
                AdmissionRequest {
                    input,
                    max_output_tokens: 2,
                    batch_size: 1,
                    safety_reserve_bytes: 0,
                    application_memory_budget_bytes: Some(1024),
                    require_complete_estimate: true
                },
                state,
                None
            )
            .unwrap(),
            AdmissionResult::Admitted(_)
        ));
    }

    #[test]
    fn admission_rejections_are_portable_and_fail_closed() {
        let capabilities = ModelCapabilities {
            model_type: "mock".into(),
            native_max_context: Observed::exact(8, "mock"),
            effective_max_context: Observed::exact(8, "mock"),
            state_strategy: CacheStateStrategy::FullKv,
            modalities: InputModalities::TEXT,
            estimation: EstimationCompleteness::Complete,
        };
        let state = RuntimeStateEstimate {
            fixed_state_bytes: 0,
            bytes_per_position_per_batch: 0,
            context_state_bytes: 0,
            multimodal_embedding_bytes: 0,
            media_execution_workspace_bytes: 0,
            requested_state_bytes: 0,
            assumptions: StateMemoryAssumptions {
                floating_state_dtype_bytes: NonZeroU8::new(4).unwrap(),
                batch_size: 1,
                requested_positions: 9,
                sliding_window_bounds: Vec::new(),
                allocation_granularity: 1,
            },
            completeness: EstimationCompleteness::Complete,
        };
        let request = AdmissionRequest {
            input: InputTokenCount::text(7),
            max_output_tokens: 2,
            batch_size: 1,
            safety_reserve_bytes: 0,
            application_memory_budget_bytes: None,
            require_complete_estimate: true,
        };
        assert!(matches!(
            apply_admission_policy(&capabilities, request, state, None).unwrap(),
            AdmissionResult::Rejected(AdmissionRejection::OutputHeadroomExceedsContext { .. })
        ));

        let unavailable = AvailableMemory {
            physical_memory_bytes: Observed::unavailable("not reported"),
            available_memory_bytes: Observed::unavailable("not reported"),
            physical_semantics: PhysicalMemorySemantics::Unknown,
        };
        let request = AdmissionRequest {
            input: InputTokenCount::text(1),
            max_output_tokens: 0,
            batch_size: 1,
            safety_reserve_bytes: 0,
            application_memory_budget_bytes: None,
            require_complete_estimate: true,
        };
        let state = estimate_runtime_state(
            &StateMemoryLayout::new(
                LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
                vec![0],
                1,
                1,
                EstimationCompleteness::Complete,
            )
            .unwrap(),
            request.input,
            0,
            1,
            NonZeroU8::new(4).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            apply_admission_policy(&capabilities, request, state, Some(&unavailable)).unwrap(),
            AdmissionResult::Rejected(AdmissionRejection::AvailableMemoryUnavailable { .. })
        ));
    }

    #[test]
    fn capability_and_memory_schemas_round_trip_without_a_backend() {
        let report = StaticMemoryReport {
            logical_parameter_bytes: Observed::exact(1_024, "mock catalog"),
            current_host_resident_bytes: Observed::exact(512, "mock ledger"),
            current_device_resident_bytes: Observed::exact(512, "mock ledger"),
            planned_disk_backed_bytes: Observed::exact(0, "mock plan"),
            backend_active_allocation_bytes: Observed::unavailable("no allocator probe"),
            backend_allocator_cache_bytes: Observed::unsupported("no allocator cache"),
            physical_semantics: PhysicalMemorySemantics::SeparateTiers,
            currently_mapped_shards: Observed::exact(1, "mock store"),
        };
        let encoded = serde_json::to_string(&report).unwrap();
        let decoded: StaticMemoryReport = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, report);
    }
}
