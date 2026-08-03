//! Architecture-independent model capability, accounting, and admission APIs.

use std::{collections::BTreeMap, num::NonZeroU8};

use safemlx::{module::ModuleParameters, Array, Stream};

use super::{
    deepseek_v3, gemma4, gpt_oss, inkling, kimi_linear, lfm2, nemotron_h, qwen3_5_moe, Model,
    PreparedModelInput,
};
use crate::{
    architectures::qwen::hybrid::qwen3_5::LayerPolicy as QwenHybridLayerPolicy,
    nn::rope::FloatOrString,
    runtime::{
        attention::AttentionPolicy,
        media::input::{InputPayload, Modality},
        residency::policy::MemoryTier,
    },
};

/// Confidence/semantics attached to a reported number.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MeasurementKind {
    /// Derived exactly from validated configuration or an exact counter.
    Exact,
    /// An upper bound intentionally chosen to avoid underestimating admission cost.
    Conservative,
    /// A point-in-time runtime observation rather than ownership accounting.
    Observational,
    /// A platform-derived estimate whose value can change immediately.
    Estimated,
}

/// A value that may be unsupported or unavailable without inventing a numeric default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityValue<T> {
    /// A usable value and its documented measurement semantics.
    Available {
        /// Reported value.
        value: T,
        /// Exact, conservative, observational, or estimated semantics.
        kind: MeasurementKind,
        /// Stable description of the source.
        source: &'static str,
    },
    /// The architecture or platform cannot produce this value.
    Unsupported {
        /// Human-readable reason.
        reason: String,
    },
    /// The value is meaningful in principle but was not available now.
    Unavailable {
        /// Human-readable reason.
        reason: String,
    },
}

impl<T> CapabilityValue<T> {
    /// Borrows the available value, returning `None` for unsupported/unavailable values.
    pub const fn value(&self) -> Option<&T> {
        match self {
            Self::Available { value, .. } => Some(value),
            Self::Unsupported { .. } | Self::Unavailable { .. } => None,
        }
    }
}

/// Model inputs accepted by the loaded architecture.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct InputModalities {
    /// Ordinary tokenizer IDs.
    pub text: bool,
    /// Prepared image tensors.
    pub image: bool,
    /// Prepared audio tensors.
    pub audio: bool,
    /// Prepared video tensors.
    pub video: bool,
}

impl InputModalities {
    const TEXT: Self = Self {
        text: true,
        image: false,
        audio: false,
        video: false,
    };
}

/// Persistent decoder-state strategy used by a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheStateStrategy {
    /// Ordinary full-context K/V attention.
    FullKv,
    /// Every attention cache is bounded by a sliding window.
    SlidingKv {
        /// Maximum retained positions per attention layer.
        window: u64,
    },
    /// A model combines full-context and sliding-window attention layers.
    MixedKv {
        /// Number of full-context layers.
        full_layers: u64,
        /// Bounded layer counts grouped by exact retained window.
        sliding: Vec<SlidingWindowLayerCount>,
    },
    /// Full-context KV backing with some layers using sliding attention masks
    /// and later layers reusing earlier K/V state.
    SharedFullKv {
        /// Layers that allocate and retain their own K/V state.
        cached_layers: u64,
        /// Layers that reuse K/V produced by an earlier layer.
        shared_layers: u64,
        /// Total full-attention layer count, including shared layers.
        full_attention_layers: u64,
        /// Total sliding-mask layers grouped by exact attention window,
        /// including shared layers. These windows do not bound KV allocation.
        sliding_attention: Vec<SlidingWindowLayerCount>,
    },
    /// Multi-head latent attention stores compressed latent and rotary state.
    CompressedMla {
        /// Compressed latent width stored per layer and position.
        latent_width: u64,
        /// Shared rotary-key width stored per layer and position.
        rotary_width: u64,
    },
    /// Attention is combined with bounded convolution or recurrent state.
    HybridRecurrent {
        /// Full-context attention layer count.
        full_attention_layers: u64,
        /// Bounded attention layers grouped by exact window.
        sliding_attention: Vec<SlidingWindowLayerCount>,
        /// Recurrent/linear-attention layer count.
        recurrent_layers: u64,
    },
    /// Multimodal preparation feeds model positions into a decoder state strategy.
    Multimodal {
        /// Underlying decoder state.
        decoder: Box<CacheStateStrategy>,
        /// Media embeddings consume persistent decoder positions.
        media_consumes_decoder_positions: bool,
    },
}

/// Sliding-attention layer count sharing one exact retained window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlidingWindowLayerCount {
    /// Exact positive retained positions, including the current token.
    pub window: u64,
    /// Number of layers using this window.
    pub layers: u64,
}

/// Whether an estimator covers all persistent and transient runtime state.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EstimationCompleteness {
    /// Persistent request state is modeled exactly for the stated dtype and cache layout.
    Complete,
    /// The estimate is a complete, safe upper bound for the stated assumptions.
    Conservative,
    /// Persistent decoder state is covered, but some architecture-visible transients are not.
    PersistentStateOnly,
}

/// Public capability report derived from validated config and the loaded architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCapabilities {
    /// Effective architecture/model type from the loaded model.
    pub model_type: String,
    /// Original trained context before a supported extension, when identifiable.
    pub native_max_context: CapabilityValue<u64>,
    /// Maximum model positions accepted by the configured architecture.
    pub effective_max_context: CapabilityValue<u64>,
    /// Persistent cache or recurrent-state model.
    pub state_strategy: CacheStateStrategy,
    /// Accepted input modalities.
    pub modalities: InputModalities,
    /// Coverage of the runtime-state estimator.
    pub estimation: EstimationCompleteness,
}

/// Accounting for a tokenized or prepared input.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct InputTokenCount {
    /// Ordinary tokenizer IDs present in the input.
    pub text_tokens: u64,
    /// Positions inserted by prepared media.
    pub media_positions: u64,
    /// Total decoder/model positions consumed by prefill.
    pub model_positions: u64,
    /// Semantics of the model-position count.
    pub kind: MeasurementKind,
    media_execution_workspace_bytes: u64,
    media_execution_workspace_kind: MeasurementKind,
}

impl InputTokenCount {
    /// Creates an exact count for an already-tokenized text prompt.
    pub const fn text(tokens: u64) -> Self {
        Self {
            text_tokens: tokens,
            media_positions: 0,
            model_positions: tokens,
            kind: MeasurementKind::Exact,
            media_execution_workspace_bytes: 0,
            media_execution_workspace_kind: MeasurementKind::Exact,
        }
    }

    fn prepared(
        text_tokens: u64,
        media_positions: u64,
        model_positions: u64,
        media_execution_workspace_bytes: u64,
        media_execution_workspace_kind: MeasurementKind,
    ) -> Self {
        Self {
            text_tokens,
            media_positions,
            model_positions,
            kind: MeasurementKind::Exact,
            media_execution_workspace_bytes,
            media_execution_workspace_kind,
        }
    }

    /// Conservative media-tower workspace attributed to this prepared input.
    pub const fn media_execution_workspace_bytes(&self) -> u64 {
        self.media_execution_workspace_bytes
    }

    /// Exact or conservative semantics of the media workspace value.
    pub const fn media_execution_workspace_kind(&self) -> MeasurementKind {
        self.media_execution_workspace_kind
    }
}

/// Dtype and request assumptions used by state estimation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StateMemoryAssumptions {
    /// Bytes per retained cache/state scalar.
    pub state_dtype_bytes: NonZeroU8,
    /// Logical request batch size.
    pub batch_size: u64,
    /// Total requested model positions, including output allowance.
    pub requested_positions: u64,
    /// Distinct sliding-window bounds applied by the estimator, in ascending order.
    pub sliding_window_bounds: Vec<u64>,
    /// Backing-array growth granularity used for unbounded caches.
    pub allocation_granularity: u64,
}

/// Persistent and transient runtime-state estimate for one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeStateEstimate {
    /// Context-independent recurrent/convolution state.
    pub fixed_state_bytes: u64,
    /// Unbounded bytes added per additional position before multiplying by batch.
    pub bytes_per_position_per_batch: u64,
    /// Persistent context-dependent bytes at the requested length.
    pub context_state_bytes: u64,
    /// Prepared-media embedding bytes retained during multimodal prefill.
    pub multimodal_embedding_bytes: u64,
    /// Conservative model-visible media-tower execution workspace.
    pub media_execution_workspace_bytes: u64,
    /// Total modeled state for prompt plus output allowance.
    pub requested_state_bytes: u64,
    /// Estimator assumptions.
    pub assumptions: StateMemoryAssumptions,
    /// Estimator coverage.
    pub completeness: EstimationCompleteness,
}

/// Physical relationship between logical host and device tiers.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PhysicalMemorySemantics {
    /// Apple/UMA host and GPU allocations draw from one physical capacity.
    Unified,
    /// Host and accelerator memory may be physically separate.
    SeparateTiers,
    /// The runtime cannot determine the physical relationship.
    Unknown,
}

/// Static checkpoint and current residency observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticMemoryReport {
    /// Logical bytes in loaded parameters or the complete residency plan.
    pub logical_parameter_bytes: CapabilityValue<u64>,
    /// Current logical host-resident bytes tracked by bounded residency.
    pub current_host_resident_bytes: CapabilityValue<u64>,
    /// Current logical device-resident bytes tracked by bounded residency.
    pub current_device_resident_bytes: CapabilityValue<u64>,
    /// Planned logical disk-backed bytes.
    pub planned_disk_backed_bytes: CapabilityValue<u64>,
    /// Process-global MLX active allocation counter, not model ownership or RSS.
    pub mlx_active_allocation_bytes: CapabilityValue<u64>,
    /// Process-global MLX allocator-cache counter.
    pub mlx_allocator_cache_bytes: CapabilityValue<u64>,
    /// Whether logical host/device tiers share one physical capacity.
    pub physical_semantics: PhysicalMemorySemantics,
    /// Number of currently retained memory mappings, when bounded residency is used.
    pub currently_mapped_shards: CapabilityValue<u64>,
}

/// System memory usable as an admission signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvailableMemory {
    /// Unified/host physical memory.
    pub physical_memory_bytes: CapabilityValue<u64>,
    /// Defensible point-in-time availability estimate.
    pub available_memory_bytes: CapabilityValue<u64>,
    /// Physical tier semantics for host and accelerator allocations.
    pub physical_semantics: PhysicalMemorySemantics,
}

/// One pre-generation admission request.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct AdmissionRequest {
    /// Authoritative prompt accounting.
    pub input: InputTokenCount,
    /// Maximum generated-token allowance.
    pub max_output_tokens: u64,
    /// Logical batch size.
    pub batch_size: u64,
    /// Caller-selected reserve added to modeled incremental state.
    pub safety_reserve_bytes: u64,
    /// Optional application budget for incremental state plus reserve.
    pub application_memory_budget_bytes: Option<u64>,
    /// Reject estimates that omit execution scratch or media-tower transients.
    pub require_complete_estimate: bool,
}

/// Detailed successful admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admission {
    /// Prompt plus output allowance.
    pub requested_positions: u64,
    /// Runtime-state estimate.
    pub state: RuntimeStateEstimate,
    /// State plus caller reserve compared with budgets/availability.
    pub incremental_required_bytes: u64,
    /// Memory signal used for the availability check, when supplied.
    pub available_memory_bytes: Option<u64>,
}

/// Structured reason a request was rejected before allocation/generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionRejection {
    /// Prompt alone exceeds the configured context.
    PromptExceedsContext {
        /// Prompt model positions.
        prompt_positions: u64,
        /// Effective model limit.
        maximum_positions: u64,
    },
    /// Prompt fits, but output allowance does not.
    OutputHeadroomExceedsContext {
        /// Prompt model positions.
        prompt_positions: u64,
        /// Requested maximum output tokens.
        output_tokens: u64,
        /// Effective model limit.
        maximum_positions: u64,
    },
    /// The supplied application budget is smaller than modeled state plus reserve.
    MemoryBudgetExceeded {
        /// Required incremental bytes.
        required_bytes: u64,
        /// Caller-supplied budget.
        budget_bytes: u64,
    },
    /// Current platform availability is smaller than modeled state plus reserve.
    InsufficientAvailableMemory {
        /// Required incremental bytes.
        required_bytes: u64,
        /// Observed/estimated available bytes.
        available_bytes: u64,
    },
    /// A requested platform-availability check could not be performed.
    AvailableMemoryUnavailable {
        /// Platform report detail.
        reason: String,
    },
    /// Admission policy requires coverage the architecture estimator cannot provide.
    EstimationUnsupported {
        /// Coverage detail.
        reason: String,
    },
}

/// Admission outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionResult {
    /// Request may proceed under the supplied policy.
    Admitted(Admission),
    /// Request was rejected before model allocation/generation.
    Rejected(AdmissionRejection),
}

/// Structured failures from capability and checked-memory accounting.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum CapabilityError {
    /// A validated architecture exposed an invalid value to the estimator.
    #[error("invalid model capability field {field}: {detail}")]
    InvalidConfiguration {
        /// Field name.
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

#[derive(Debug, Clone)]
struct GrowingState {
    layers: u64,
    scalars_per_position: u64,
    window: Option<u64>,
}

#[derive(Debug, Clone)]
struct ArchitectureEstimate {
    fixed_scalars_per_batch: u64,
    growing: Vec<GrowingState>,
    hidden_size: u64,
    allocation_granularity: u64,
    completeness: EstimationCompleteness,
}

fn positive(value: i32, field: &'static str) -> Result<u64, CapabilityError> {
    u64::try_from(value).map_err(|_| CapabilityError::InvalidConfiguration {
        field,
        detail: format!("expected a non-negative value, got {value}"),
    })
}

fn nonzero_positive(value: i32, field: &'static str) -> Result<u64, CapabilityError> {
    let value = positive(value, field)?;
    if value == 0 {
        return Err(CapabilityError::InvalidConfiguration {
            field,
            detail: "expected a positive value, got zero".into(),
        });
    }
    Ok(value)
}

fn checked_add(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_add(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn checked_mul(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_mul(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn rounded_cache_positions(
    requested_positions: u64,
    allocation_granularity: u64,
) -> Result<u64, CapabilityError> {
    let adjustment =
        allocation_granularity
            .checked_sub(1)
            .ok_or(CapabilityError::InvalidConfiguration {
                field: "allocation_granularity",
                detail: "cache allocation granularity must be nonzero".into(),
            })?;
    Ok(
        checked_add(requested_positions, adjustment, "cache allocation rounding")?
            / allocation_granularity
            * allocation_granularity,
    )
}

fn config_number(
    config: &std::collections::HashMap<String, FloatOrString>,
    key: &str,
) -> Option<f32> {
    match config.get(key) {
        Some(FloatOrString::Float(value)) => Some(*value),
        Some(FloatOrString::String(value)) => value.parse().ok(),
        Some(FloatOrString::Bool(_)) | None => None,
    }
}

fn context_from_rope(
    effective: i32,
    rope: Option<&std::collections::HashMap<String, FloatOrString>>,
) -> Result<(CapabilityValue<u64>, CapabilityValue<u64>), CapabilityError> {
    let effective = positive(effective, "max_position_embeddings")?;
    let original =
        match rope.and_then(|rope| config_number(rope, "original_max_position_embeddings")) {
            Some(value)
                if value.is_finite()
                    && value > 0.0
                    && value.fract() == 0.0
                    && value <= u64::MAX as f32 =>
            {
                Some(value as u64)
            }
            Some(_) => {
                return Ok((
                    CapabilityValue::Unsupported {
                        reason: "RoPE original context is not a positive integer".into(),
                    },
                    CapabilityValue::Available {
                        value: effective,
                        kind: MeasurementKind::Exact,
                        source: "validated model configuration",
                    },
                ))
            }
            None => None,
        };
    let native = original.unwrap_or(effective);
    let effective = if original.is_none() {
        match rope
            .and_then(|rope| config_number(rope, "factor"))
            .filter(|factor| factor.is_finite() && *factor > 1.0)
        {
            Some(factor) => {
                let scaled = effective as f64 * f64::from(factor);
                if !scaled.is_finite() || scaled.fract() != 0.0 || scaled > u64::MAX as f64 {
                    return Ok((
                        CapabilityValue::Available {
                            value: native,
                            kind: MeasurementKind::Exact,
                            source: "validated model configuration",
                        },
                        CapabilityValue::Unsupported {
                            reason:
                                "RoPE factor does not produce an exact integer effective context"
                                    .into(),
                        },
                    ));
                }
                scaled as u64
            }
            None => effective,
        }
    } else {
        effective
    };
    Ok((
        CapabilityValue::Available {
            value: native,
            kind: MeasurementKind::Exact,
            source: "validated model configuration",
        },
        CapabilityValue::Available {
            value: effective,
            kind: MeasurementKind::Exact,
            source: "validated model configuration and supported RoPE setup",
        },
    ))
}

fn plain_context(
    maximum: i32,
) -> Result<(CapabilityValue<u64>, CapabilityValue<u64>), CapabilityError> {
    context_from_rope(maximum, None)
}

fn kv_scalars(kv_heads: i32, head_dim: i32) -> Result<u64, CapabilityError> {
    let one = checked_mul(
        positive(kv_heads, "num_key_value_heads")?,
        positive(head_dim, "head_dim")?,
        "K/V heads times head dimension",
    )?;
    checked_mul(one, 2, "key plus value scalars")
}

fn text_modalities() -> InputModalities {
    InputModalities::TEXT
}

impl Model {
    fn capabilities_and_estimate(
        &self,
    ) -> Result<(ModelCapabilities, ArchitectureEstimate), CapabilityError> {
        let model_type = self.model_type().to_string();
        let result = match self {
            Self::Llama(model) => llama_spec(&model.args, false)?,
            Self::LlamaLayerwise(model) => llama_spec(model.args(), false)?,
            Self::DenseQwen(model) => dense_qwen_spec(&model.args, false)?,
            Self::DenseQwenLayerwise(model) => dense_qwen_spec(model.args(), false)?,
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => {
                dense_qwen_spec(&model.args.text_config, true)?
            }
            Self::Qwen3VlLayerwise(model) | Self::Qwen3VlMoeLayerwise(model) => {
                dense_qwen_spec(&model.args().text_config, true)?
            }
            Self::DeepSeekV3(model) => deepseek_spec(&model.args)?,
            Self::DeepSeekV3Layerwise(model) => deepseek_spec(model.args())?,
            Self::GptOss(model) => gpt_oss_spec(&model.args)?,
            Self::GptOssLayerwise(model) => gpt_oss_spec(model.args())?,
            Self::Gemma4(model) => {
                let modalities = InputModalities {
                    text: true,
                    image: model.image_token_id.is_some(),
                    audio: model.audio_token_id.is_some(),
                    video: model.video_token_id.is_some(),
                };
                gemma4_spec(&model.args, modalities)?
            }
            Self::Gemma4Layerwise(model) => {
                let (_, _, image, audio, video) = model.media_accounting();
                gemma4_spec(
                    model.args(),
                    InputModalities {
                        text: true,
                        image,
                        audio,
                        video,
                    },
                )?
            }
            Self::Inkling(model) => inkling_spec(&model.args)?,
            Self::InklingLayerwise(model) => inkling_spec(model.args())?,
            Self::KimiLinear(model) => kimi_linear_spec(&model.args)?,
            Self::KimiLinearLayerwise(model) => kimi_linear_spec(model.args())?,
            Self::Lfm2(model) => lfm2_spec(&model.args)?,
            Self::Lfm2Layerwise(model) => lfm2_spec(model.args())?,
            Self::NemotronH(model) => nemotron_spec(&model.args)?,
            Self::NemotronHLayerwise(model) => nemotron_spec(model.args())?,
            Self::Qwen3Next(model) | Self::Qwen35Moe(model) => {
                qwen_hybrid_spec(&model.args, model.vision_args.is_some())?
            }
            Self::Qwen3NextLayerwise(model) | Self::Qwen35MoeLayerwise(model) => {
                qwen_hybrid_spec(model.args(), model.vision_spatial_merge_size().is_some())?
            }
        };
        let (native_max_context, effective_max_context, state_strategy, modalities, estimate) =
            result;
        Ok((
            ModelCapabilities {
                model_type,
                native_max_context,
                effective_max_context,
                state_strategy,
                modalities,
                estimation: if modalities.image || modalities.audio || modalities.video {
                    EstimationCompleteness::Conservative
                } else {
                    estimate.completeness
                },
            },
            estimate,
        ))
    }
}

type Spec = (
    CapabilityValue<u64>,
    CapabilityValue<u64>,
    CacheStateStrategy,
    InputModalities,
    ArchitectureEstimate,
);

fn llama_spec(args: &super::llama::ModelArgs, multimodal: bool) -> Result<Spec, CapabilityError> {
    let context = context_from_rope(args.max_position_embeddings, args.rope_scaling.as_ref())?;
    let layers = positive(args.num_hidden_layers, "num_hidden_layers")?;
    let scalars = kv_scalars(args.num_key_value_heads, args.head_dim)?;
    let window = args
        .sliding_window
        .map(|value| positive(value, "sliding_window"))
        .transpose()?;
    let base = match window {
        Some(window) => CacheStateStrategy::SlidingKv { window },
        None => CacheStateStrategy::FullKv,
    };
    let strategy = if multimodal {
        CacheStateStrategy::Multimodal {
            decoder: Box::new(base),
            media_consumes_decoder_positions: true,
        }
    } else {
        base
    };
    let completeness = EstimationCompleteness::Complete;
    Ok((
        context.0,
        context.1,
        strategy,
        if multimodal {
            InputModalities {
                text: true,
                image: true,
                audio: false,
                video: true,
            }
        } else {
            text_modalities()
        },
        ArchitectureEstimate {
            fixed_scalars_per_batch: 0,
            growing: vec![GrowingState {
                layers,
                scalars_per_position: scalars,
                window,
            }],
            hidden_size: positive(args.hidden_size, "hidden_size")?,
            allocation_granularity: 1,
            completeness,
        },
    ))
}

fn dense_qwen_spec(
    args: &super::dense_qwen::DecoderConfig,
    multimodal: bool,
) -> Result<Spec, CapabilityError> {
    let mut spec = llama_spec(
        &super::llama::ModelArgs {
            model_type: args.model_type.clone(),
            hidden_size: args.hidden_size,
            num_hidden_layers: args.num_hidden_layers,
            intermediate_size: args.intermediate_size,
            num_attention_heads: args.num_attention_heads,
            rms_norm_eps: args.rms_norm_eps,
            vocab_size: args.vocab_size,
            num_key_value_heads: args.num_key_value_heads,
            max_position_embeddings: args.max_position_embeddings,
            rope_theta: args.rope_theta,
            rope_traditional: false,
            head_dim: args.head_dim,
            tie_word_embeddings: args.tie_word_embeddings,
            attention_bias: args.qkv_bias(),
            mlp_bias: false,
            rope_scaling: args.rope_scaling.clone(),
            sliding_window: None,
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        },
        multimodal,
    )?;
    let full_layers = args.attention_schedule.full_layer_count() as u64;
    let sliding = args
        .attention_schedule
        .sliding_windows()
        .into_iter()
        .map(|(window, layers)| SlidingWindowLayerCount {
            window: u64::from(window.get()),
            layers: layers as u64,
        })
        .collect::<Vec<_>>();
    if !sliding.is_empty() {
        let scalars = kv_scalars(args.num_key_value_heads, args.head_dim)?;
        spec.2 = if full_layers == 0 && sliding.len() == 1 {
            CacheStateStrategy::SlidingKv {
                window: sliding[0].window,
            }
        } else {
            CacheStateStrategy::MixedKv {
                full_layers,
                sliding: sliding.clone(),
            }
        };
        spec.4.growing = std::iter::once(GrowingState {
            layers: full_layers,
            scalars_per_position: scalars,
            window: None,
        })
        .filter(|state| state.layers > 0)
        .chain(sliding.into_iter().map(|group| GrowingState {
            layers: group.layers,
            scalars_per_position: scalars,
            window: Some(group.window),
        }))
        .collect();
    }
    Ok(spec)
}

fn deepseek_spec(args: &deepseek_v3::ModelArgs) -> Result<Spec, CapabilityError> {
    let effective = positive(args.max_position_embeddings, "max_position_embeddings")?;
    let native = args
        .rope_scaling
        .as_ref()
        .map(|rope| {
            positive(
                rope.original_max_position_embeddings,
                "original_max_position_embeddings",
            )
        })
        .transpose()?
        .unwrap_or(effective);
    let layers = positive(args.num_hidden_layers, "num_hidden_layers")?;
    let latent = positive(args.kv_lora_rank, "kv_lora_rank")?;
    let rotary = positive(args.qk_rope_head_dim, "qk_rope_head_dim")?;
    let width = checked_add(latent, rotary, "MLA latent plus rotary width")?;
    Ok((
        CapabilityValue::Available {
            value: native,
            kind: MeasurementKind::Exact,
            source: "validated DeepSeek YaRN configuration",
        },
        CapabilityValue::Available {
            value: effective,
            kind: MeasurementKind::Exact,
            source: "validated DeepSeek configuration",
        },
        CacheStateStrategy::CompressedMla {
            latent_width: latent,
            rotary_width: rotary,
        },
        text_modalities(),
        ArchitectureEstimate {
            fixed_scalars_per_batch: 0,
            growing: vec![GrowingState {
                layers,
                scalars_per_position: width,
                window: None,
            }],
            hidden_size: positive(args.hidden_size, "hidden_size")?,
            allocation_granularity: 256,
            completeness: EstimationCompleteness::Complete,
        },
    ))
}

fn kimi_linear_spec(args: &kimi_linear::ModelArgs) -> Result<Spec, CapabilityError> {
    let context = plain_context(args.model_max_length)?;
    let attention = args.linear_attn_config.full_attn_layers.len() as u64;
    let recurrent = args.linear_attn_config.kda_layers.len() as u64;
    let heads = positive(
        args.linear_attn_config.num_heads,
        "linear_attn_config.num_heads",
    )?;
    let head_dim = positive(
        args.linear_attn_config.head_dim,
        "linear_attn_config.head_dim",
    )?;
    let projection = checked_mul(heads, head_dim, "KDA projected width")?;
    let conv_state = checked_mul(
        checked_mul(
            positive(
                args.linear_attn_config.short_conv_kernel_size - 1,
                "linear_attn_config.short_conv_kernel_size",
            )?,
            projection,
            "KDA convolution history width",
        )?,
        3,
        "KDA Q/K/V convolution states",
    )?;
    let recurrent_state = checked_mul(
        checked_mul(heads, head_dim, "KDA recurrent heads times key width")?,
        head_dim,
        "KDA recurrent state",
    )?;
    let fixed = checked_mul(
        recurrent,
        checked_add(conv_state, recurrent_state, "KDA fixed layer state")?,
        "all KDA fixed state",
    )?;
    let mla_width = checked_add(
        positive(args.kv_lora_rank, "kv_lora_rank")?,
        positive(args.qk_rope_head_dim, "qk_rope_head_dim")?,
        "Kimi MLA latent plus identity positional width",
    )?;
    Ok((
        context.0,
        context.1,
        CacheStateStrategy::HybridRecurrent {
            full_attention_layers: attention,
            sliding_attention: Vec::new(),
            recurrent_layers: recurrent,
        },
        text_modalities(),
        ArchitectureEstimate {
            fixed_scalars_per_batch: fixed,
            growing: vec![GrowingState {
                layers: attention,
                scalars_per_position: mla_width,
                window: None,
            }],
            hidden_size: positive(args.hidden_size, "hidden_size")?,
            allocation_granularity: 256,
            completeness: EstimationCompleteness::Complete,
        },
    ))
}

fn gpt_oss_spec(args: &gpt_oss::ModelArgs) -> Result<Spec, CapabilityError> {
    let context = context_from_rope(args.max_position_embeddings, args.rope_scaling.as_ref())?;
    positive(args.num_hidden_layers, "num_hidden_layers")?;
    let full = args.attention_schedule.full_layer_count() as u64;
    let sliding = args
        .attention_schedule
        .sliding_windows()
        .into_iter()
        .map(|(window, layers)| SlidingWindowLayerCount {
            layers: layers as u64,
            window: u64::from(window.get()),
        })
        .collect::<Vec<_>>();
    let scalars = kv_scalars(args.num_key_value_heads, args.head_dim)?;
    let state_strategy = match (full, sliding.as_slice()) {
        (_, []) => CacheStateStrategy::FullKv,
        (0, [only]) => CacheStateStrategy::SlidingKv {
            window: only.window,
        },
        _ => CacheStateStrategy::MixedKv {
            full_layers: full,
            sliding: sliding.clone(),
        },
    };
    Ok((
        context.0,
        context.1,
        state_strategy,
        text_modalities(),
        ArchitectureEstimate {
            fixed_scalars_per_batch: 0,
            growing: std::iter::once(GrowingState {
                layers: full,
                scalars_per_position: scalars,
                window: None,
            })
            .filter(|state| state.layers > 0)
            .chain(sliding.into_iter().map(|group| GrowingState {
                layers: group.layers,
                scalars_per_position: scalars,
                window: Some(group.window),
            }))
            .collect(),
            hidden_size: positive(args.hidden_size, "hidden_size")?,
            allocation_granularity: 1,
            completeness: EstimationCompleteness::Complete,
        },
    ))
}

fn gemma4_spec(
    args: &gemma4::ModelArgs,
    modalities: InputModalities,
) -> Result<Spec, CapabilityError> {
    let context = context_from_rope(args.max_position_embeddings, args.rope_scaling.as_ref())?;
    let layers = positive(args.num_hidden_layers, "num_hidden_layers")?;
    let shared = positive(args.num_kv_shared_layers, "num_kv_shared_layers")?;
    if shared > layers {
        return Err(CapabilityError::InvalidConfiguration {
            field: "num_kv_shared_layers",
            detail: format!("{shared} shared layers exceed {layers} total layers"),
        });
    }
    let cached = layers - shared;
    let mut sliding_by_window = BTreeMap::<u64, u64>::new();
    for policy in args.attention_schedule.iter() {
        if let Some(window) = policy.window() {
            *sliding_by_window
                .entry(u64::from(window.get()))
                .or_default() += 1;
        }
    }
    let total_sliding = sliding_by_window.values().sum::<u64>();
    let full_attention_layers = layers - total_sliding;
    let mut cached_sliding = 0;
    for policy in args.attention_schedule.iter().take(cached as usize) {
        if policy.window().is_some() {
            cached_sliding += 1;
        }
    }
    let cached_full = cached - cached_sliding;
    let sliding_attention = sliding_by_window
        .into_iter()
        .map(|(window, layers)| SlidingWindowLayerCount { window, layers })
        .collect();
    let local_scalars = kv_scalars(args.num_key_value_heads, args.head_dim)?;
    let global_scalars = kv_scalars(
        args.num_global_key_value_heads
            .unwrap_or(args.num_key_value_heads),
        args.global_head_dim.unwrap_or(args.head_dim),
    )?;
    let decoder = if shared > 0 || total_sliding > 0 {
        CacheStateStrategy::SharedFullKv {
            cached_layers: cached,
            shared_layers: shared,
            full_attention_layers,
            sliding_attention,
        }
    } else {
        CacheStateStrategy::FullKv
    };
    let has_media = modalities.image || modalities.audio || modalities.video;
    Ok((
        context.0,
        context.1,
        if has_media {
            CacheStateStrategy::Multimodal {
                decoder: Box::new(decoder),
                media_consumes_decoder_positions: true,
            }
        } else {
            decoder
        },
        modalities,
        ArchitectureEstimate {
            fixed_scalars_per_batch: 0,
            growing: vec![
                GrowingState {
                    layers: cached_full,
                    scalars_per_position: global_scalars,
                    window: None,
                },
                GrowingState {
                    layers: cached_sliding,
                    scalars_per_position: local_scalars,
                    // Gemma masks sliding attention but retains full KV.
                    window: None,
                },
            ],
            hidden_size: positive(args.hidden_size, "hidden_size")?,
            allocation_granularity: 256,
            completeness: EstimationCompleteness::Complete,
        },
    ))
}

fn inkling_spec(args: &inkling::ModelArgs) -> Result<Spec, CapabilityError> {
    let text = &args.text_config;
    let context = match text.model_max_length {
        Some(maximum) => plain_context(maximum)?,
        None => (
            CapabilityValue::Unsupported {
                reason: "Inkling configuration does not expose a native maximum context".into(),
            },
            CapabilityValue::Unsupported {
                reason: "Inkling configuration does not expose an effective maximum context".into(),
            },
        ),
    };
    let layers = positive(text.num_hidden_layers, "num_hidden_layers")?;
    let global = text
        .layer_schedule
        .iter()
        .filter(|policy| policy.attention == AttentionPolicy::Full)
        .count() as u64;
    let sliding = text
        .layer_schedule
        .iter()
        .filter_map(|policy| policy.attention.window())
        .fold(BTreeMap::<u64, u64>::new(), |mut groups, window| {
            *groups.entry(u64::from(window.get())).or_default() += 1;
            groups
        })
        .into_iter()
        .map(|(window, layers)| SlidingWindowLayerCount { layers, window })
        .collect::<Vec<_>>();
    let local_kv = kv_scalars(
        text.swa_num_key_value_heads
            .unwrap_or(text.num_key_value_heads),
        text.swa_head_dim.unwrap_or(text.head_dim),
    )?;
    let global_kv = kv_scalars(text.num_key_value_heads, text.head_dim)?;
    let conv_width = checked_mul(
        positive(text.hidden_size, "hidden_size")?,
        4,
        "Inkling convolution widths",
    )?;
    let fixed = checked_mul(
        checked_mul(
            layers,
            positive(text.sconv_kernel_size - 1, "sconv_kernel_size")?,
            "Inkling layers times convolution state",
        )?,
        conv_width,
        "Inkling fixed convolution state",
    )?;
    let modalities = InputModalities {
        text: true,
        image: args.vision_config.is_some(),
        audio: args.audio_config.is_some(),
        video: false,
    };
    let decoder = match (global, sliding.as_slice()) {
        (_, []) => CacheStateStrategy::FullKv,
        (0, [only]) => CacheStateStrategy::SlidingKv {
            window: only.window,
        },
        _ => CacheStateStrategy::MixedKv {
            full_layers: global,
            sliding: sliding.clone(),
        },
    };
    Ok((
        context.0,
        context.1,
        CacheStateStrategy::Multimodal {
            decoder: Box::new(decoder),
            media_consumes_decoder_positions: true,
        },
        modalities,
        ArchitectureEstimate {
            fixed_scalars_per_batch: fixed,
            growing: std::iter::once(GrowingState {
                layers: global,
                scalars_per_position: global_kv,
                window: None,
            })
            .filter(|state| state.layers > 0)
            .chain(sliding.into_iter().map(|group| GrowingState {
                layers: group.layers,
                scalars_per_position: local_kv,
                window: Some(group.window),
            }))
            .collect(),
            hidden_size: positive(text.hidden_size, "hidden_size")?,
            allocation_granularity: 1,
            completeness: EstimationCompleteness::Complete,
        },
    ))
}

fn lfm2_spec(args: &lfm2::ModelArgs) -> Result<Spec, CapabilityError> {
    let context = plain_context(args.max_position_embeddings)?;
    let attention = args
        .layer_schedule
        .iter()
        .filter(|policy| matches!(policy, lfm2::LayerPolicy::SelfAttention(_)))
        .count() as u64;
    let conv = args
        .layer_schedule
        .iter()
        .filter(|policy| matches!(policy, lfm2::LayerPolicy::CausalConvolution))
        .count() as u64;
    let head_dim = args.hidden_size / args.num_attention_heads;
    let fixed = checked_mul(
        checked_mul(
            conv,
            positive(args.conv_l_cache - 1, "conv_L_cache")?,
            "LFM convolution layers times history",
        )?,
        positive(args.hidden_size, "hidden_size")?,
        "LFM fixed convolution state",
    )?;
    Ok((
        context.0,
        context.1,
        CacheStateStrategy::HybridRecurrent {
            full_attention_layers: attention,
            sliding_attention: Vec::new(),
            recurrent_layers: conv,
        },
        text_modalities(),
        ArchitectureEstimate {
            fixed_scalars_per_batch: fixed,
            growing: vec![GrowingState {
                layers: attention,
                scalars_per_position: kv_scalars(args.num_key_value_heads, head_dim)?,
                window: None,
            }],
            hidden_size: positive(args.hidden_size, "hidden_size")?,
            allocation_granularity: 256,
            completeness: EstimationCompleteness::Complete,
        },
    ))
}

fn nemotron_spec(args: &nemotron_h::ModelArgs) -> Result<Spec, CapabilityError> {
    let context = plain_context(args.max_position_embeddings)?;
    let mamba = args
        .layer_schedule
        .iter()
        .filter(|policy| **policy == nemotron_h::LayerPolicy::Mamba)
        .count() as u64;
    let mut attention_groups = BTreeMap::<Option<u64>, u64>::new();
    for policy in args.layer_schedule.iter() {
        if let nemotron_h::LayerPolicy::SelfAttention(policy) = policy {
            let window = policy.window().map(|window| u64::from(window.get()));
            *attention_groups.entry(window).or_default() += 1;
        }
    }
    let intermediate = checked_mul(
        positive(args.mamba_num_heads, "mamba_num_heads")?,
        positive(args.mamba_head_dim, "mamba_head_dim")?,
        "Mamba intermediate width",
    )?;
    let conv_dim = checked_add(
        intermediate,
        checked_mul(
            checked_mul(2, positive(args.n_groups, "n_groups")?, "Mamba B/C groups")?,
            positive(args.ssm_state_size, "ssm_state_size")?,
            "Mamba B/C state width",
        )?,
        "Mamba convolution width",
    )?;
    let conv_state = checked_mul(
        positive(args.conv_kernel - 1, "conv_kernel")?,
        conv_dim,
        "Mamba convolution state",
    )?;
    let ssm_state = checked_mul(
        checked_mul(
            positive(args.mamba_num_heads, "mamba_num_heads")?,
            positive(args.mamba_head_dim, "mamba_head_dim")?,
            "Mamba heads times head dimension",
        )?,
        positive(args.ssm_state_size, "ssm_state_size")?,
        "Mamba SSM state",
    )?;
    let fixed = checked_mul(
        mamba,
        checked_add(conv_state, ssm_state, "Mamba fixed layer state")?,
        "all Mamba fixed state",
    )?;
    let full_attention_layers = attention_groups.get(&None).copied().unwrap_or(0);
    let sliding_attention = attention_groups
        .iter()
        .filter_map(|(window, layers)| {
            window.map(|window| SlidingWindowLayerCount {
                layers: *layers,
                window,
            })
        })
        .collect();
    Ok((
        context.0,
        context.1,
        CacheStateStrategy::HybridRecurrent {
            full_attention_layers,
            sliding_attention,
            recurrent_layers: mamba,
        },
        text_modalities(),
        ArchitectureEstimate {
            fixed_scalars_per_batch: fixed,
            growing: attention_groups
                .into_iter()
                .map(|(window, layers)| {
                    Ok(GrowingState {
                        layers,
                        scalars_per_position: kv_scalars(args.num_key_value_heads, args.head_dim)?,
                        window,
                    })
                })
                .collect::<Result<Vec<_>, CapabilityError>>()?,
            hidden_size: positive(args.hidden_size, "hidden_size")?,
            allocation_granularity: 1,
            completeness: EstimationCompleteness::Complete,
        },
    ))
}

fn qwen_hybrid_spec(
    args: &qwen3_5_moe::ModelArgs,
    multimodal: bool,
) -> Result<Spec, CapabilityError> {
    let configured = positive(args.max_position_embeddings, "max_position_embeddings")?;
    let original = args
        .rope_scaling
        .as_ref()
        .and_then(|config| config.get("original_max_position_embeddings"))
        .and_then(serde_json::Value::as_u64);
    let native = original.unwrap_or(configured);
    let effective = if original.is_some() {
        CapabilityValue::Available {
            value: configured,
            kind: MeasurementKind::Exact,
            source: "validated Qwen RoPE configuration",
        }
    } else {
        match args
            .rope_scaling
            .as_ref()
            .and_then(|config| config.get("factor"))
            .and_then(serde_json::Value::as_f64)
            .filter(|factor| factor.is_finite() && *factor > 1.0)
        {
            Some(factor) => {
                let scaled = configured as f64 * factor;
                if scaled.is_finite() && scaled.fract() == 0.0 && scaled <= u64::MAX as f64 {
                    CapabilityValue::Available {
                        value: scaled as u64,
                        kind: MeasurementKind::Exact,
                        source: "validated Qwen configuration and supported RoPE setup",
                    }
                } else {
                    CapabilityValue::Unsupported {
                        reason:
                            "Qwen RoPE factor does not produce an exact integer effective context"
                                .into(),
                    }
                }
            }
            None => CapabilityValue::Available {
                value: configured,
                kind: MeasurementKind::Exact,
                source: "validated Qwen configuration",
            },
        }
    };
    let layers = positive(args.num_hidden_layers, "num_hidden_layers")?;
    let attention = args
        .layer_schedule
        .iter()
        .filter(|policy| {
            matches!(
                policy,
                QwenHybridLayerPolicy::SelfAttention(AttentionPolicy::Full)
            )
        })
        .count() as u64;
    let recurrent = layers.saturating_sub(attention);
    let key_dim = checked_mul(
        positive(args.linear_num_key_heads, "linear_num_key_heads")?,
        positive(args.linear_key_head_dim, "linear_key_head_dim")?,
        "linear key width",
    )?;
    let value_dim = checked_mul(
        positive(args.linear_num_value_heads, "linear_num_value_heads")?,
        positive(args.linear_value_head_dim, "linear_value_head_dim")?,
        "linear value width",
    )?;
    let conv_dim = checked_add(
        checked_mul(2, key_dim, "linear query/key width")?,
        value_dim,
        "linear convolution width",
    )?;
    let conv_state = checked_mul(
        positive(args.linear_conv_kernel_dim - 1, "linear_conv_kernel_dim")?,
        conv_dim,
        "linear convolution state",
    )?;
    let recurrent_state = checked_mul(
        checked_mul(
            positive(args.linear_num_value_heads, "linear_num_value_heads")?,
            positive(args.linear_key_head_dim, "linear_key_head_dim")?,
            "linear recurrent heads times key width",
        )?,
        positive(args.linear_value_head_dim, "linear_value_head_dim")?,
        "linear recurrent state",
    )?;
    let fixed = checked_mul(
        recurrent,
        checked_add(conv_state, recurrent_state, "linear fixed layer state")?,
        "all linear-attention fixed state",
    )?;
    let base = CacheStateStrategy::HybridRecurrent {
        full_attention_layers: attention,
        sliding_attention: Vec::new(),
        recurrent_layers: recurrent,
    };
    Ok((
        CapabilityValue::Available {
            value: native,
            kind: MeasurementKind::Exact,
            source: "validated Qwen RoPE configuration",
        },
        effective,
        if multimodal {
            CacheStateStrategy::Multimodal {
                decoder: Box::new(base),
                media_consumes_decoder_positions: true,
            }
        } else {
            base
        },
        if multimodal {
            InputModalities {
                text: true,
                image: true,
                audio: false,
                video: true,
            }
        } else {
            text_modalities()
        },
        ArchitectureEstimate {
            fixed_scalars_per_batch: fixed,
            growing: vec![GrowingState {
                layers: attention,
                scalars_per_position: kv_scalars(args.num_key_value_heads, args.head_dim)?,
                window: None,
            }],
            hidden_size: positive(args.hidden_size, "hidden_size")?,
            allocation_granularity: 1,
            completeness: EstimationCompleteness::Complete,
        },
    ))
}

fn parameter_bytes<M: ModuleParameters>(model: &M) -> Result<u64, CapabilityError> {
    model
        .parameters()
        .flatten()
        .values()
        .try_fold(0u64, |total, parameter| {
            let bytes = u64::try_from(parameter.nbytes()).map_err(|_| {
                CapabilityError::ArithmeticOverflow {
                    operation: "parameter byte conversion",
                }
            })?;
            checked_add(total, bytes, "logical parameter byte total")
        })
}

impl Model {
    fn resident_parameter_bytes(&self) -> Result<Option<u64>, CapabilityError> {
        let value = match self {
            Self::DeepSeekV3(model) => Some(parameter_bytes(model)?),
            Self::Gemma4(model) => Some(parameter_bytes(model)?),
            Self::GptOss(model) => Some(parameter_bytes(model)?),
            Self::Inkling(model) => Some(parameter_bytes(model)?),
            Self::KimiLinear(model) => Some(parameter_bytes(model)?),
            Self::Llama(model) => Some(parameter_bytes(model)?),
            Self::Lfm2(model) => Some(parameter_bytes(model)?),
            Self::NemotronH(model) => Some(parameter_bytes(model)?),
            Self::DenseQwen(model) => Some(parameter_bytes(model)?),
            Self::Qwen3Next(model) => Some(parameter_bytes(model)?),
            Self::Qwen3Vl(model) => Some(parameter_bytes(model)?),
            Self::Qwen3VlMoe(model) => Some(parameter_bytes(model)?),
            Self::Qwen35Moe(model) => Some(parameter_bytes(model)?),
            Self::LlamaLayerwise(model) => model
                .resident_parameter_bytes()
                .transpose()
                .map_err(|detail| CapabilityError::Observation(detail.into()))?,
            Self::DeepSeekV3Layerwise(_)
            | Self::Gemma4Layerwise(_)
            | Self::GptOssLayerwise(_)
            | Self::InklingLayerwise(_)
            | Self::KimiLinearLayerwise(_)
            | Self::Lfm2Layerwise(_)
            | Self::NemotronHLayerwise(_)
            | Self::DenseQwenLayerwise(_)
            | Self::Qwen3NextLayerwise(_)
            | Self::Qwen3VlLayerwise(_)
            | Self::Qwen3VlMoeLayerwise(_)
            | Self::Qwen35MoeLayerwise(_) => None,
        };
        Ok(value)
    }
}

fn unavailable_counter(error: safemlx::error::Exception) -> CapabilityValue<u64> {
    CapabilityValue::Unavailable {
        reason: error.to_string(),
    }
}

fn runtime_counter(
    function: fn() -> Result<usize, safemlx::error::Exception>,
    source: &'static str,
) -> CapabilityValue<u64> {
    match function() {
        Ok(value) => match u64::try_from(value) {
            Ok(value) => CapabilityValue::Available {
                value,
                kind: MeasurementKind::Observational,
                source,
            },
            Err(_) => CapabilityValue::Unavailable {
                reason: "counter does not fit u64".into(),
            },
        },
        Err(error) => unavailable_counter(error),
    }
}

fn bool_count(array: &Array) -> Result<u64, CapabilityError> {
    let evaluated = array
        .evaluated()
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let values = evaluated
        .try_as_slice::<bool>()
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    u64::try_from(values.iter().filter(|value| **value).count()).map_err(|_| {
        CapabilityError::ArithmeticOverflow {
            operation: "boolean mask count",
        }
    })
}

fn array_bytes(array: &Array, operation: &'static str) -> Result<u64, CapabilityError> {
    u64::try_from(array.nbytes()).map_err(|_| CapabilityError::ArithmeticOverflow { operation })
}

fn four_byte_scalars(scalars: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    checked_mul(scalars, 4, operation)
}

fn qwen_attention_chunk_squares(
    grid: &[(i32, i32, i32)],
    merge: u64,
    window_size: i32,
    patch_size: i32,
) -> Result<(u64, u64), CapabilityError> {
    let patch = nonzero_positive(patch_size, "Qwen vision patch size")?;
    let window_pixels = nonzero_positive(window_size, "Qwen vision window size")?;
    let merger_window = window_pixels / merge / patch;
    if merger_window == 0 {
        return Err(CapabilityError::InvalidConfiguration {
            field: "window_size",
            detail: format!(
                "Qwen vision window {window_pixels} is too small for merge {merge} and patch {patch}"
            ),
        });
    }
    let merge_area = checked_mul(merge, merge, "Qwen attention merge area")?;
    let merge_area_square =
        checked_mul(merge_area, merge_area, "Qwen attention merge-area square")?;
    let mut full_squares = 0u64;
    let mut window_squares = 0u64;
    for (time, height, width) in grid {
        let time = nonzero_positive(*time, "Qwen grid time")?;
        let height = nonzero_positive(*height, "Qwen grid height")?;
        let width = nonzero_positive(*width, "Qwen grid width")?;
        if height % merge != 0 || width % merge != 0 {
            return Err(CapabilityError::InvalidConfiguration {
                field: "qwen_grid_thw",
                detail: format!(
                    "Qwen grid ({height}, {width}) is not divisible by spatial merge {merge}"
                ),
            });
        }
        let full_length = checked_mul(height, width, "Qwen full-attention chunk length")?;
        full_squares = checked_add(
            full_squares,
            checked_mul(
                time,
                checked_mul(full_length, full_length, "Qwen full-attention chunk square")?,
                "Qwen full-attention temporal chunks",
            )?,
            "Qwen full-attention chunk-square total",
        )?;

        let merged_height = height / merge;
        let merged_width = width / merge;
        let height_full = merged_height / merger_window;
        let height_remainder = merged_height % merger_window;
        let width_full = merged_width / merger_window;
        let width_remainder = merged_width % merger_window;
        let window_square = checked_mul(merger_window, merger_window, "Qwen merger-window square")?;
        let height_square_sum = checked_add(
            checked_mul(
                height_full,
                window_square,
                "Qwen full height-window squares",
            )?,
            checked_mul(
                height_remainder,
                height_remainder,
                "Qwen remainder height-window square",
            )?,
            "Qwen height-window square sum",
        )?;
        let width_square_sum = checked_add(
            checked_mul(width_full, window_square, "Qwen full width-window squares")?,
            checked_mul(
                width_remainder,
                width_remainder,
                "Qwen remainder width-window square",
            )?,
            "Qwen width-window square sum",
        )?;
        let item_window_squares = checked_mul(
            checked_mul(
                height_square_sum,
                width_square_sum,
                "Qwen merged window-area squares",
            )?,
            merge_area_square,
            "Qwen patch window-area squares",
        )?;
        window_squares = checked_add(
            window_squares,
            checked_mul(time, item_window_squares, "Qwen temporal window chunks")?,
            "Qwen window-attention chunk-square total",
        )?;
    }
    Ok((full_squares, window_squares))
}

fn gemma_valid_patch_count(positions: &Array, architecture: &str) -> Result<u64, CapabilityError> {
    if positions.ndim() != 3 || positions.dim(0) != 1 || positions.dim(2) != 2 {
        return Err(CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: format!(
                "Gemma patch positions must be [1, patches, 2], got {:?}",
                positions.shape()
            ),
        });
    }
    let evaluated = positions
        .evaluated()
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let values = evaluated
        .try_as_slice::<i32>()
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    u64::try_from(
        values
            .chunks_exact(2)
            .filter(|pair| pair[0] >= 0 && pair[1] >= 0)
            .count(),
    )
    .map_err(|_| CapabilityError::ArithmeticOverflow {
        operation: "Gemma valid patch count",
    })
}

fn qwen_vision_workspace(
    config: &crate::architectures::qwen::vl::vision::VisionConfig,
    modality: Modality,
    payload: &Array,
    metadata: super::input::InputMetadata<'_>,
    stream: &Stream,
    architecture: &str,
) -> Result<(u64, u64), CapabilityError> {
    if !matches!(modality, Modality::Image | Modality::Video) {
        return Err(CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: format!("{} is not a Qwen vision modality", modality.as_str()),
        });
    }
    if payload.ndim() != 2 {
        return Err(CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: format!(
                "Qwen prepared vision tensor must be [patches, patch_dims], got {:?}",
                payload.shape()
            ),
        });
    }
    let patches = positive(payload.dim(0), "Qwen prepared patch count")?;
    let merge = nonzero_positive(config.spatial_merge_size, "spatial_merge_size")?;
    let patch = nonzero_positive(config.patch_size, "Qwen vision patch size")?;
    let expected_patch_dims = checked_mul(
        checked_mul(
            nonzero_positive(config.in_channels, "Qwen vision input channels")?,
            nonzero_positive(
                config.temporal_patch_size,
                "Qwen vision temporal patch size",
            )?,
            "Qwen temporal input channels",
        )?,
        checked_mul(patch, patch, "Qwen vision patch area")?,
        "Qwen vision patch dimensions",
    )?;
    if positive(payload.dim(1), "Qwen prepared patch dimensions")? != expected_patch_dims {
        return Err(CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: format!(
                "Qwen prepared patches have width {}, expected {expected_patch_dims}",
                payload.dim(1)
            ),
        });
    }
    let merge_area = checked_mul(merge, merge, "Qwen spatial merge area")?;
    if patches % merge_area != 0 {
        return Err(CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: format!("Qwen patch count {patches} is not divisible by {merge_area}"),
        });
    }
    let positions = patches / merge_area;
    let grid = metadata
        .qwen_grid_thw
        .ok_or_else(|| CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: "prepared Qwen media has no grid_thw metadata".into(),
        })
        .and_then(|grid| {
            crate::architectures::qwen::vl::vision::grid_thw_from_array(grid, stream).map_err(
                |error| CapabilityError::UnsupportedInput {
                    architecture: architecture.into(),
                    reason: error.to_string(),
                },
            )
        })?;
    let described_patches = grid.iter().try_fold(0u64, |total, (time, height, width)| {
        let item_patches = checked_mul(
            checked_mul(
                nonzero_positive(*time, "Qwen grid time")?,
                nonzero_positive(*height, "Qwen grid height")?,
                "Qwen grid time-height",
            )?,
            nonzero_positive(*width, "Qwen grid width")?,
            "Qwen grid item patches",
        )?;
        checked_add(total, item_patches, "Qwen described patch total")
    })?;
    if described_patches != patches {
        return Err(CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: format!(
                "Qwen grid describes {described_patches} patches but payload has {patches}"
            ),
        });
    }
    let (full_chunk_squares, window_chunk_squares) =
        qwen_attention_chunk_squares(&grid, merge, config.window_size, config.patch_size)?;
    let depth = positive(config.depth, "Qwen vision depth")?;
    let full_blocks = u64::try_from(config.fullatt_block_indexes.len())
        .map_err(|_| CapabilityError::ArithmeticOverflow {
            operation: "Qwen full-attention block count",
        })?
        .min(depth);
    let window_blocks = depth - full_blocks;
    let heads = positive(config.num_heads, "Qwen vision heads")?;
    let hidden = positive(config.hidden_size, "Qwen vision hidden size")?;
    let intermediate = positive(config.intermediate_size, "Qwen vision intermediate size")?;
    let out_hidden = positive(config.out_hidden_size, "Qwen vision output size")?;

    // Conservative model-visible graph bound: normalization/rotary/residual
    // temporaries, QKV and MLP outputs, fused-attention inputs/outputs, every
    // configured block, and all patch-merger/deepstack outputs.
    let patch_hidden = checked_mul(patches, hidden, "Qwen patch hidden elements")?;
    let patch_intermediate =
        checked_mul(patches, intermediate, "Qwen patch intermediate elements")?;
    let per_block = checked_add(
        checked_mul(32, patch_hidden, "Qwen block hidden workspace")?,
        checked_mul(6, patch_intermediate, "Qwen block intermediate workspace")?,
        "Qwen block workspace",
    )?;
    let block_workspace = checked_mul(depth, per_block, "Qwen all-block workspace")?;
    let full_attention = checked_mul(
        checked_mul(
            checked_mul(
                full_blocks,
                full_chunk_squares,
                "Qwen full-attention blocks",
            )?,
            heads,
            "Qwen full-attention heads",
        )?,
        2,
        "Qwen full-attention score/probability bound",
    )?;
    let window_attention = checked_mul(
        checked_mul(
            checked_mul(
                window_blocks,
                window_chunk_squares,
                "Qwen window-attention blocks",
            )?,
            heads,
            "Qwen window-attention heads",
        )?,
        2,
        "Qwen window-attention score/probability bound",
    )?;
    let merge_width = checked_mul(hidden, merge_area, "Qwen merger width")?;
    let merger_output = checked_mul(
        positions,
        checked_add(
            checked_mul(12, merge_width, "Qwen merger hidden workspace")?,
            checked_mul(6, out_hidden, "Qwen merger output workspace")?,
            "Qwen merger per-position workspace",
        )?,
        "Qwen merger workspace",
    )?;
    let mergers = checked_add(
        1,
        u64::try_from(config.deepstack_visual_indexes.len()).map_err(|_| {
            CapabilityError::ArithmeticOverflow {
                operation: "Qwen deepstack merger count",
            }
        })?,
        "Qwen merger count",
    )?;
    let graph_scalars = checked_add(
        checked_add(
            checked_mul(16, patch_hidden, "Qwen vision setup workspace")?,
            block_workspace,
            "Qwen setup plus blocks",
        )?,
        checked_add(
            checked_add(full_attention, window_attention, "Qwen attention workspace")?,
            checked_mul(mergers, merger_output, "Qwen all-merger workspace")?,
            "Qwen attention plus mergers",
        )?,
        "Qwen vision graph workspace",
    )?;
    Ok((
        positions,
        checked_add(
            array_bytes(payload, "Qwen prepared media bytes")?,
            four_byte_scalars(graph_scalars, "Qwen vision graph bytes")?,
            "Qwen total vision workspace",
        )?,
    ))
}

fn gemma_vision_workspace(
    config: &crate::architectures::gemma4::vision::Gemma4VisionConfig,
    text_hidden: u64,
    payload: &Array,
    metadata: super::input::InputMetadata<'_>,
    architecture: &str,
) -> Result<(u64, u64), CapabilityError> {
    if payload.ndim() != 3 || payload.dim(0) != 1 {
        return Err(CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: format!(
                "Gemma prepared vision tensor must be [1, patches, patch_dims], got {:?}",
                payload.shape()
            ),
        });
    }
    let patch = nonzero_positive(config.patch_size, "Gemma vision patch size")?;
    let expected_patch_dims = checked_mul(
        3,
        checked_mul(patch, patch, "Gemma vision patch area")?,
        "Gemma vision patch dimensions",
    )?;
    if positive(payload.dim(2), "Gemma prepared patch dimensions")? != expected_patch_dims {
        return Err(CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: format!(
                "Gemma prepared patches have width {}, expected {expected_patch_dims}",
                payload.dim(2)
            ),
        });
    }
    let position_ids =
        metadata
            .patch_position_ids
            .ok_or_else(|| CapabilityError::UnsupportedInput {
                architecture: architecture.into(),
                reason: "prepared Gemma media has no patch positions".into(),
            })?;
    let valid_patches = gemma_valid_patch_count(position_ids, architecture)?;
    let pool = nonzero_positive(config.pooling_kernel_size, "Gemma pooling kernel")?;
    let pool_area = checked_mul(pool, pool, "Gemma pooling area")?;
    if valid_patches % pool_area != 0 {
        return Err(CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: format!(
                "Gemma valid patch count {valid_patches} is not divisible by {pool_area}"
            ),
        });
    }
    let positions = valid_patches / pool_area;
    let padded_patches = positive(payload.dim(1), "Gemma padded patch count")?;
    if positive(position_ids.dim(1), "Gemma patch-position count")? != padded_patches {
        return Err(CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: format!(
                "Gemma patch positions {:?} do not match prepared vision payload {:?}",
                position_ids.shape(),
                payload.shape()
            ),
        });
    }
    let hidden = positive(config.hidden_size, "Gemma vision hidden size")?;
    let intermediate = positive(config.intermediate_size, "Gemma vision intermediate size")?;
    let depth = positive(config.num_hidden_layers, "Gemma vision depth")?;
    let patch_hidden = checked_mul(padded_patches, hidden, "Gemma vision patch hidden elements")?;
    let per_layer = checked_add(
        checked_mul(48, patch_hidden, "Gemma vision layer hidden workspace")?,
        checked_mul(
            8,
            checked_mul(
                padded_patches,
                intermediate,
                "Gemma vision intermediate elements",
            )?,
            "Gemma vision MLP workspace",
        )?,
        "Gemma vision layer workspace",
    )?;
    let output_workspace = checked_mul(
        positions,
        checked_add(
            checked_mul(8, hidden, "Gemma pooled vision workspace")?,
            checked_mul(8, text_hidden, "Gemma projected vision workspace")?,
            "Gemma vision output workspace per position",
        )?,
        "Gemma vision output workspace",
    )?;
    let graph_scalars = checked_add(
        checked_mul(20, patch_hidden, "Gemma vision setup workspace")?,
        checked_add(
            checked_mul(depth, per_layer, "Gemma all vision layers")?,
            output_workspace,
            "Gemma layers plus output workspace",
        )?,
        "Gemma vision graph workspace",
    )?;
    let input_bytes = checked_add(
        array_bytes(payload, "Gemma prepared vision bytes")?,
        array_bytes(position_ids, "Gemma patch-position bytes")?,
        "Gemma prepared vision input bytes",
    )?;
    Ok((
        positions,
        checked_add(
            input_bytes,
            four_byte_scalars(graph_scalars, "Gemma vision graph bytes")?,
            "Gemma total vision workspace",
        )?,
    ))
}

fn gemma_audio_workspace(
    config: &crate::architectures::gemma4::audio::Gemma4AudioConfig,
    text_hidden: u64,
    payload: &Array,
    metadata: super::input::InputMetadata<'_>,
    architecture: &str,
) -> Result<(u64, u64), CapabilityError> {
    if payload.ndim() != 3 || payload.dim(0) != 1 || payload.dim(2) != 128 {
        return Err(CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: format!(
                "Gemma prepared audio tensor must be [1, frames, 128], got {:?}",
                payload.shape()
            ),
        });
    }
    let mask = metadata
        .audio_mask
        .ok_or_else(|| CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: "prepared Gemma audio has no frame mask".into(),
        })?;
    let frames = positive(payload.dim(1), "Gemma padded audio frames")?;
    if mask.ndim() != 2
        || mask.dim(0) != 1
        || positive(mask.dim(1), "Gemma audio mask frames")? != frames
    {
        return Err(CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: format!(
                "Gemma audio mask must be [1, {frames}], got {:?}",
                mask.shape()
            ),
        });
    }
    let positions = bool_count(mask)?.div_ceil(4);
    let sequence = frames.div_ceil(4);
    let hidden = positive(config.hidden_size, "Gemma audio hidden size")?;
    let depth = positive(config.num_hidden_layers, "Gemma audio depth")?;
    let heads = positive(config.num_attention_heads, "Gemma audio heads")?;
    let chunk = nonzero_positive(config.attention_chunk_size, "Gemma audio attention chunk")?;
    let past = nonzero_positive(
        config.attention_context_left.checked_sub(1).ok_or(
            CapabilityError::ArithmeticOverflow {
                operation: "Gemma audio left context",
            },
        )?,
        "Gemma audio left context",
    )?;
    let padded_sequence = checked_mul(
        sequence.div_ceil(chunk),
        chunk,
        "Gemma padded audio sequence",
    )?;
    let chunks = padded_sequence / chunk;
    let attention_elements = checked_mul(
        checked_mul(
            checked_mul(
                checked_mul(chunks, heads, "Gemma audio attention chunk heads")?,
                chunk,
                "Gemma audio attention queries",
            )?,
            checked_add(chunk, past, "Gemma audio attention key bound")?,
            "Gemma audio attention scores",
        )?,
        4,
        "Gemma audio logits/relative/mask/probability workspace",
    )?;
    let layer_workspace = checked_add(
        checked_mul(
            80,
            checked_mul(sequence, hidden, "Gemma audio hidden elements")?,
            "Gemma audio layer hidden workspace",
        )?,
        attention_elements,
        "Gemma audio layer workspace",
    )?;
    let first_frames = frames.div_ceil(2);
    let first_channels = positive(
        *config.subsampling_conv_channels.first().ok_or_else(|| {
            CapabilityError::InvalidConfiguration {
                field: "subsampling_conv_channels",
                detail: "Gemma audio has no first convolution channel count".into(),
            }
        })?,
        "Gemma audio first convolution channels",
    )?;
    let second_channels = positive(
        *config.subsampling_conv_channels.get(1).ok_or_else(|| {
            CapabilityError::InvalidConfiguration {
                field: "subsampling_conv_channels",
                detail: "Gemma audio has no second convolution channel count".into(),
            }
        })?,
        "Gemma audio second convolution channels",
    )?;
    let conv_workspace = checked_add(
        checked_mul(
            6,
            checked_mul(
                checked_mul(first_frames, 64, "Gemma first convolution grid")?,
                first_channels,
                "Gemma first convolution elements",
            )?,
            "Gemma first convolution workspace",
        )?,
        checked_mul(
            6,
            checked_mul(
                checked_mul(sequence, 32, "Gemma second convolution grid")?,
                second_channels,
                "Gemma second convolution elements",
            )?,
            "Gemma second convolution workspace",
        )?,
        "Gemma convolution workspace",
    )?;
    let output = positive(config.output_proj_dims, "Gemma audio output size")?;
    let output_workspace = checked_mul(
        positions,
        checked_add(
            checked_mul(8, output, "Gemma audio output workspace")?,
            checked_mul(8, text_hidden, "Gemma audio text projection workspace")?,
            "Gemma audio output workspace per position",
        )?,
        "Gemma audio projected output workspace",
    )?;
    let graph_scalars = checked_add(
        conv_workspace,
        checked_add(
            checked_mul(depth, layer_workspace, "Gemma all audio layers")?,
            output_workspace,
            "Gemma audio layers plus output",
        )?,
        "Gemma audio graph workspace",
    )?;
    let input_bytes = checked_add(
        array_bytes(payload, "Gemma prepared audio bytes")?,
        array_bytes(mask, "Gemma audio mask bytes")?,
        "Gemma prepared audio input bytes",
    )?;
    Ok((
        positions,
        checked_add(
            input_bytes,
            four_byte_scalars(graph_scalars, "Gemma audio graph bytes")?,
            "Gemma total audio workspace",
        )?,
    ))
}

fn inkling_workspace(
    args: &inkling::ModelArgs,
    modality: Modality,
    payload: &Array,
    metadata: super::input::InputMetadata<'_>,
    architecture: &str,
) -> Result<(u64, u64), CapabilityError> {
    match modality {
        Modality::Image => {
            let config =
                args.vision_config
                    .as_ref()
                    .ok_or_else(|| CapabilityError::UnsupportedInput {
                        architecture: architecture.into(),
                        reason: "loaded Inkling model has no vision configuration".into(),
                    })?;
            if payload.ndim() != 5 || payload.shape()[1..] != [2, 40, 40, 3] {
                return Err(CapabilityError::UnsupportedInput {
                    architecture: architecture.into(),
                    reason: format!(
                        "Inkling image patches must be [patches, 2, 40, 40, 3], got {:?}",
                        payload.shape()
                    ),
                });
            }
            let patches = positive(payload.dim(0), "Inkling image patch count")?;
            let text_hidden = positive(config.text_hidden_size, "Inkling vision output size")?;
            let layer_outputs = [
                checked_mul(
                    checked_mul(
                        checked_mul(patches, 2, "Inkling vision time")?,
                        8 * 8,
                        "Inkling vision grid",
                    )?,
                    128,
                    "Inkling vision layer 1",
                )?,
                checked_mul(
                    checked_mul(
                        checked_mul(patches, 2, "Inkling vision time")?,
                        4 * 4,
                        "Inkling vision grid",
                    )?,
                    512,
                    "Inkling vision layer 2",
                )?,
                checked_mul(
                    checked_mul(patches, 2, "Inkling vision time")?,
                    4_800,
                    "Inkling vision layer 3",
                )?,
                checked_mul(patches, text_hidden, "Inkling vision layer 4")?,
            ];
            let graph_scalars = layer_outputs.iter().try_fold(0u64, |total, value| {
                checked_add(
                    total,
                    checked_mul(12, *value, "Inkling vision layer workspace")?,
                    "Inkling vision graph workspace",
                )
            })?;
            Ok((
                patches,
                checked_add(
                    array_bytes(payload, "Inkling prepared image bytes")?,
                    four_byte_scalars(graph_scalars, "Inkling vision graph bytes")?,
                    "Inkling total vision workspace",
                )?,
            ))
        }
        Modality::Audio => {
            let config =
                args.audio_config
                    .as_ref()
                    .ok_or_else(|| CapabilityError::UnsupportedInput {
                        architecture: architecture.into(),
                        reason: "loaded Inkling model has no audio configuration".into(),
                    })?;
            let (padded_frames, payload_codebooks) = match payload.ndim() {
                2 => (
                    positive(payload.dim(0), "Inkling audio frame count")?,
                    positive(payload.dim(1), "Inkling audio payload codebooks")?,
                ),
                3 if payload.dim(0) == 1 => (
                    positive(payload.dim(1), "Inkling audio frame count")?,
                    positive(payload.dim(2), "Inkling audio payload codebooks")?,
                ),
                _ => {
                    return Err(CapabilityError::UnsupportedInput {
                        architecture: architecture.into(),
                        reason: format!(
                            "Inkling audio tokens must be [frames, codebooks] or [1, frames, codebooks], got {:?}",
                            payload.shape()
                        ),
                    });
                }
            };
            let codebooks = positive(config.num_codebooks, "Inkling audio codebooks")?;
            if payload_codebooks != codebooks {
                return Err(CapabilityError::UnsupportedInput {
                    architecture: architecture.into(),
                    reason: format!(
                        "Inkling audio payload has {payload_codebooks} codebooks, expected {codebooks}"
                    ),
                });
            }
            let frames = if let Some(mask) = metadata.audio_mask {
                if mask.ndim() != 2
                    || mask.dim(0) != 1
                    || positive(mask.dim(1), "Inkling audio mask frames")? != padded_frames
                {
                    return Err(CapabilityError::UnsupportedInput {
                        architecture: architecture.into(),
                        reason: format!(
                            "Inkling audio mask must be [1, {padded_frames}], got {:?}",
                            mask.shape()
                        ),
                    });
                }
                bool_count(mask)?
            } else {
                padded_frames
            };
            let hidden = positive(config.text_hidden_size, "Inkling audio hidden size")?;
            let embedded = checked_mul(
                checked_mul(padded_frames, codebooks, "Inkling audio frame codebooks")?,
                hidden,
                "Inkling audio embedding elements",
            )?;
            let reduced = checked_mul(padded_frames, hidden, "Inkling audio reduced elements")?;
            let graph_scalars = checked_add(
                checked_mul(4, embedded, "Inkling audio embedding workspace")?,
                checked_mul(12, reduced, "Inkling audio reduction/norm workspace")?,
                "Inkling audio graph workspace",
            )?;
            let mut input_bytes = array_bytes(payload, "Inkling prepared audio bytes")?;
            if let Some(mask) = metadata.audio_mask {
                input_bytes = checked_add(
                    input_bytes,
                    array_bytes(mask, "Inkling audio mask bytes")?,
                    "Inkling prepared audio input bytes",
                )?;
            }
            Ok((
                frames,
                checked_add(
                    input_bytes,
                    four_byte_scalars(graph_scalars, "Inkling audio graph bytes")?,
                    "Inkling total audio workspace",
                )?,
            ))
        }
        Modality::Video => Err(CapabilityError::UnsupportedInput {
            architecture: architecture.into(),
            reason: "video is not a supported Inkling modality".into(),
        }),
        Modality::Text => unreachable!("text handled separately"),
    }
}

impl Model {
    fn prepared_media_accounting(
        &self,
        modality: Modality,
        payload: &Array,
        metadata: super::input::InputMetadata<'_>,
        stream: &Stream,
    ) -> Result<(u64, u64), CapabilityError> {
        match self {
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => qwen_vision_workspace(
                &model.args.vision_config,
                modality,
                payload,
                metadata,
                stream,
                self.model_type(),
            ),
            Self::Qwen3VlLayerwise(model) | Self::Qwen3VlMoeLayerwise(model) => {
                qwen_vision_workspace(
                    &model.args().vision_config,
                    modality,
                    payload,
                    metadata,
                    stream,
                    self.model_type(),
                )
            }
            Self::Qwen3Next(model) | Self::Qwen35Moe(model) => model
                .vision_args
                .as_ref()
                .ok_or_else(|| CapabilityError::UnsupportedInput {
                    architecture: self.model_type().into(),
                    reason: "loaded model has no vision configuration".into(),
                })
                .and_then(|config| {
                    qwen_vision_workspace(
                        config,
                        modality,
                        payload,
                        metadata,
                        stream,
                        self.model_type(),
                    )
                }),
            Self::Qwen3NextLayerwise(model) | Self::Qwen35MoeLayerwise(model) => model
                .vision_config()
                .ok_or_else(|| CapabilityError::UnsupportedInput {
                    architecture: self.model_type().into(),
                    reason: "loaded model has no vision configuration".into(),
                })
                .and_then(|config| {
                    qwen_vision_workspace(
                        config,
                        modality,
                        payload,
                        metadata,
                        stream,
                        self.model_type(),
                    )
                }),
            Self::Gemma4(model) => match modality {
                Modality::Image | Modality::Video => model
                    .model
                    .vision_tower
                    .as_ref()
                    .ok_or_else(|| CapabilityError::UnsupportedInput {
                        architecture: self.model_type().into(),
                        reason: "loaded model has no vision tower".into(),
                    })
                    .and_then(|tower| {
                        gemma_vision_workspace(
                            &tower.config,
                            positive(model.args.hidden_size, "Gemma text hidden size")?,
                            payload,
                            metadata,
                            self.model_type(),
                        )
                    }),
                Modality::Audio => model
                    .model
                    .audio_tower
                    .as_ref()
                    .ok_or_else(|| CapabilityError::UnsupportedInput {
                        architecture: self.model_type().into(),
                        reason: "loaded model has no audio tower".into(),
                    })
                    .and_then(|tower| {
                        gemma_audio_workspace(
                            &tower.config,
                            positive(model.args.hidden_size, "Gemma text hidden size")?,
                            payload,
                            metadata,
                            self.model_type(),
                        )
                    }),
                Modality::Text => unreachable!("text handled separately"),
            },
            Self::Gemma4Layerwise(model) => {
                let (vision, audio, _, _, _) = model.media_accounting();
                match modality {
                    Modality::Image | Modality::Video => vision
                        .ok_or_else(|| CapabilityError::UnsupportedInput {
                            architecture: self.model_type().into(),
                            reason: "loaded model has no vision tower".into(),
                        })
                        .and_then(|config| {
                            gemma_vision_workspace(
                                config,
                                positive(model.args().hidden_size, "Gemma text hidden size")?,
                                payload,
                                metadata,
                                self.model_type(),
                            )
                        }),
                    Modality::Audio => audio
                        .ok_or_else(|| CapabilityError::UnsupportedInput {
                            architecture: self.model_type().into(),
                            reason: "loaded model has no audio tower".into(),
                        })
                        .and_then(|config| {
                            gemma_audio_workspace(
                                config,
                                positive(model.args().hidden_size, "Gemma text hidden size")?,
                                payload,
                                metadata,
                                self.model_type(),
                            )
                        }),
                    Modality::Text => unreachable!("text handled separately"),
                }
            }
            Self::Inkling(model) => {
                inkling_workspace(&model.args, modality, payload, metadata, self.model_type())
            }
            Self::InklingLayerwise(model) => {
                inkling_workspace(model.args(), modality, payload, metadata, self.model_type())
            }
            Self::DeepSeekV3(_)
            | Self::DeepSeekV3Layerwise(_)
            | Self::GptOss(_)
            | Self::GptOssLayerwise(_)
            | Self::KimiLinear(_)
            | Self::KimiLinearLayerwise(_)
            | Self::Llama(_)
            | Self::LlamaLayerwise(_)
            | Self::Lfm2(_)
            | Self::Lfm2Layerwise(_)
            | Self::NemotronH(_)
            | Self::NemotronHLayerwise(_)
            | Self::DenseQwen(_)
            | Self::DenseQwenLayerwise(_) => Err(CapabilityError::UnsupportedInput {
                architecture: self.model_type().into(),
                reason: format!("{} media is not supported", modality.as_str()),
            }),
        }
    }
}

fn estimate_architecture_state(
    estimate: &ArchitectureEstimate,
    input: InputTokenCount,
    max_output_tokens: u64,
    batch_size: u64,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    if batch_size == 0 {
        return Err(CapabilityError::InvalidConfiguration {
            field: "batch_size",
            detail: "batch size must be positive".into(),
        });
    }
    let requested_positions = checked_add(
        input.model_positions,
        max_output_tokens,
        "prompt plus output positions",
    )?;
    let dtype = NonZeroU8::new(4).expect("four is nonzero");
    let scalar_bytes = u64::from(dtype.get());
    let fixed_state_bytes = checked_mul(
        checked_mul(
            estimate.fixed_scalars_per_batch,
            batch_size,
            "fixed state times batch",
        )?,
        scalar_bytes,
        "fixed state bytes",
    )?;
    let mut context_state_bytes = 0u64;
    let mut unbounded_per_position = 0u64;
    let mut sliding_window_bounds = Vec::new();
    for component in &estimate.growing {
        let per_position = checked_mul(
            component.layers,
            component.scalars_per_position,
            "component scalars per position",
        )?;
        let retained = component.window.map_or_else(
            || rounded_cache_positions(requested_positions, estimate.allocation_granularity),
            |window| Ok(requested_positions.min(window)),
        )?;
        let component_bytes = checked_mul(
            checked_mul(
                checked_mul(per_position, retained, "component context scalars")?,
                batch_size,
                "component context batch",
            )?,
            scalar_bytes,
            "component context bytes",
        )?;
        context_state_bytes = checked_add(
            context_state_bytes,
            component_bytes,
            "context state byte total",
        )?;
        if component.window.is_none() {
            unbounded_per_position = checked_add(
                unbounded_per_position,
                checked_mul(per_position, scalar_bytes, "unbounded bytes per position")?,
                "unbounded bytes-per-position total",
            )?;
        }
        if let Some(window) = component.window {
            sliding_window_bounds.push(window);
        }
    }
    let multimodal_embedding_bytes = checked_mul(
        checked_mul(
            checked_mul(
                input.media_positions,
                estimate.hidden_size,
                "media positions times hidden size",
            )?,
            batch_size,
            "media embeddings times batch",
        )?,
        scalar_bytes,
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
    let completeness = if input.media_positions == 0 {
        estimate.completeness
    } else {
        match input.media_execution_workspace_kind {
            MeasurementKind::Exact => estimate.completeness,
            MeasurementKind::Conservative
            | MeasurementKind::Observational
            | MeasurementKind::Estimated => EstimationCompleteness::Conservative,
        }
    };
    Ok(RuntimeStateEstimate {
        fixed_state_bytes,
        bytes_per_position_per_batch: unbounded_per_position,
        context_state_bytes,
        multimodal_embedding_bytes,
        media_execution_workspace_bytes,
        requested_state_bytes,
        assumptions: StateMemoryAssumptions {
            state_dtype_bytes: dtype,
            batch_size,
            requested_positions,
            sliding_window_bounds: {
                sliding_window_bounds.sort_unstable();
                sliding_window_bounds.dedup();
                sliding_window_bounds
            },
            allocation_granularity: estimate.allocation_granularity,
        },
        completeness,
    })
}

impl super::LoadedModel {
    /// Returns architecture-independent capabilities derived from validated loaded state.
    pub fn capabilities(&self) -> Result<ModelCapabilities, CapabilityError> {
        self.model
            .capabilities_and_estimate()
            .map(|(capabilities, _)| capabilities)
    }

    /// Counts an ordinary encoded prompt exactly.
    pub fn count_token_ids(&self, token_ids: &[u32]) -> Result<InputTokenCount, CapabilityError> {
        let tokens =
            u64::try_from(token_ids.len()).map_err(|_| CapabilityError::ArithmeticOverflow {
                operation: "token-id length",
            })?;
        Ok(InputTokenCount::text(tokens))
    }

    /// Tokenizes and counts a rendered text prompt exactly.
    pub fn count_text(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<InputTokenCount, CapabilityError> {
        let ids = self
            .encode(text, add_special_tokens)
            .map_err(|error| CapabilityError::Observation(error.to_string()))?;
        self.count_token_ids(&ids)
    }

    /// Counts the exact rendered prompt stored in a prepared chat.
    pub fn count_prepared_chat(
        &self,
        chat: &crate::runtime::chat::PreparedChat,
    ) -> Result<InputTokenCount, CapabilityError> {
        self.count_text(chat.rendered_prompt(), false)
    }

    /// Counts text IDs and actual model positions in processor-prepared multimodal input.
    ///
    /// Media positions are derived from prepared patch grids, pooling metadata,
    /// or valid audio masks used by the loaded architecture. Tensor media also
    /// carries a conservative execution-workspace bound derived from those
    /// prepared shapes and the loaded tower configuration.
    pub fn count_prepared_input(
        &self,
        prepared: &PreparedModelInput,
        stream: &Stream,
    ) -> Result<InputTokenCount, CapabilityError> {
        let mut text_tokens = 0u64;
        let mut media_positions = 0u64;
        let mut media_execution_workspace_bytes = 0u64;
        let mut media_execution_workspace_kind = MeasurementKind::Exact;
        for part in prepared.input_parts() {
            match (part.modality, part.payload) {
                (Modality::Text, InputPayload::TokenIds(tokens)) => {
                    if tokens.ndim() != 2 || tokens.dim(0) != 1 {
                        return Err(CapabilityError::UnsupportedInput {
                            architecture: self.model_type().into(),
                            reason: format!(
                                "prepared text token IDs must be [1, sequence], got {:?}",
                                tokens.shape()
                            ),
                        });
                    }
                    text_tokens = checked_add(
                        text_tokens,
                        positive(tokens.dim(1), "prepared text sequence")?,
                        "prepared text-token total",
                    )?;
                }
                (Modality::Text, _) => {
                    return Err(CapabilityError::UnsupportedInput {
                        architecture: self.model_type().into(),
                        reason: "prepared text is not represented by tokenizer IDs".into(),
                    });
                }
                (_modality, InputPayload::Embeddings(embeddings)) => {
                    if embeddings.ndim() != 3 || embeddings.dim(0) != 1 {
                        return Err(CapabilityError::UnsupportedInput {
                            architecture: self.model_type().into(),
                            reason: format!(
                                "prepared media embeddings must be [1, sequence, hidden], got {:?}",
                                embeddings.shape()
                            ),
                        });
                    }
                    media_positions = checked_add(
                        media_positions,
                        positive(embeddings.dim(1), "prepared embedding sequence")?,
                        "prepared media-position total",
                    )?;
                }
                (modality, InputPayload::Tensor(tensor)) => {
                    let (positions, workspace_bytes) = self.model.prepared_media_accounting(
                        modality,
                        tensor,
                        part.metadata,
                        stream,
                    )?;
                    media_positions =
                        checked_add(media_positions, positions, "prepared media-position total")?;
                    media_execution_workspace_bytes = checked_add(
                        media_execution_workspace_bytes,
                        workspace_bytes,
                        "prepared media-workspace total",
                    )?;
                    media_execution_workspace_kind = MeasurementKind::Conservative;
                }
                (_, InputPayload::TokenIds(_)) => {
                    return Err(CapabilityError::UnsupportedInput {
                        architecture: self.model_type().into(),
                        reason: "non-text prepared input cannot contain tokenizer IDs".into(),
                    });
                }
            }
        }
        Ok(InputTokenCount::prepared(
            text_tokens,
            media_positions,
            checked_add(
                text_tokens,
                media_positions,
                "prepared model-position total",
            )?,
            media_execution_workspace_bytes,
            media_execution_workspace_kind,
        ))
    }

    /// Estimates persistent request state and prepared-media execution workspace
    /// with checked arithmetic.
    ///
    /// Cache/state scalars are conservatively modeled as four-byte values.
    /// This matches current float32 cache construction and avoids understating
    /// models whose checkpoint weights use a narrower storage dtype.
    pub fn estimate_runtime_state(
        &self,
        input: InputTokenCount,
        max_output_tokens: u64,
        batch_size: u64,
    ) -> Result<RuntimeStateEstimate, CapabilityError> {
        let (_, estimate) = self.model.capabilities_and_estimate()?;
        estimate_architecture_state(&estimate, input, max_output_tokens, batch_size)
    }

    /// Reports logical checkpoint/residency accounting and MLX allocator observations.
    pub fn static_memory(&self) -> Result<StaticMemoryReport, CapabilityError> {
        let residency = self
            .model
            .residency_report()
            .map_err(|error| CapabilityError::Observation(error.to_string()))?;
        let resident = self.model.resident_parameter_bytes()?;
        let (logical, host, device, disk, mappings) = if let Some(report) = residency {
            let planned = report.offload().planned_bytes();
            let resident = report.offload().resident_bytes();
            let logical = checked_add(
                checked_add(
                    planned.get(MemoryTier::Host),
                    planned.get(MemoryTier::Device),
                    "planned host plus device parameters",
                )?,
                planned.get(MemoryTier::Disk),
                "complete planned parameter bytes",
            )?;
            (
                CapabilityValue::Available {
                    value: logical,
                    kind: MeasurementKind::Exact,
                    source: "validated bounded-residency plan",
                },
                CapabilityValue::Available {
                    value: resident.get(MemoryTier::Host),
                    kind: MeasurementKind::Exact,
                    source: "bounded-residency manager",
                },
                CapabilityValue::Available {
                    value: resident.get(MemoryTier::Device),
                    kind: MeasurementKind::Exact,
                    source: "bounded-residency manager",
                },
                CapabilityValue::Available {
                    value: planned.get(MemoryTier::Disk),
                    kind: MeasurementKind::Exact,
                    source: "bounded-residency plan",
                },
                CapabilityValue::Available {
                    value: report.weight_store().currently_mapped_shards as u64,
                    kind: MeasurementKind::Observational,
                    source: "checkpoint-store mapping cache",
                },
            )
        } else if let Some(bytes) = resident {
            (
                CapabilityValue::Available {
                    value: bytes,
                    kind: MeasurementKind::Exact,
                    source: "loaded module parameter arrays",
                },
                CapabilityValue::Unsupported {
                    reason: "fully resident MLX arrays do not expose independent host ownership"
                        .into(),
                },
                CapabilityValue::Unavailable {
                    reason: "MLX active memory is process-global, not attributable per model"
                        .into(),
                },
                CapabilityValue::Available {
                    value: 0,
                    kind: MeasurementKind::Exact,
                    source: "fully resident load policy",
                },
                CapabilityValue::Unsupported {
                    reason: "fully resident checkpoint does not retain a bounded mapping cache"
                        .into(),
                },
            )
        } else {
            (
                CapabilityValue::Unavailable {
                    reason: "loaded model exposes neither resident parameters nor a residency plan"
                        .into(),
                },
                CapabilityValue::Unavailable {
                    reason: "host residency unavailable".into(),
                },
                CapabilityValue::Unavailable {
                    reason: "device residency unavailable".into(),
                },
                CapabilityValue::Unavailable {
                    reason: "disk residency unavailable".into(),
                },
                CapabilityValue::Unavailable {
                    reason: "mapping information unavailable".into(),
                },
            )
        };
        Ok(StaticMemoryReport {
            logical_parameter_bytes: logical,
            current_host_resident_bytes: host,
            current_device_resident_bytes: device,
            planned_disk_backed_bytes: disk,
            mlx_active_allocation_bytes: runtime_counter(
                safemlx::memory::active_memory,
                "process-global MLX active allocation counter",
            ),
            mlx_allocator_cache_bytes: runtime_counter(
                safemlx::memory::cache_memory,
                "process-global MLX allocator cache counter",
            ),
            physical_semantics: if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                PhysicalMemorySemantics::Unified
            } else {
                PhysicalMemorySemantics::Unknown
            },
            currently_mapped_shards: mappings,
        })
    }

    /// Applies context and memory policy without allocating a model cache.
    pub fn admit(
        &self,
        request: AdmissionRequest,
        available: Option<&AvailableMemory>,
    ) -> Result<AdmissionResult, CapabilityError> {
        let capabilities = self.capabilities()?;
        let maximum = match capabilities.effective_max_context {
            CapabilityValue::Available { value, .. } => value,
            CapabilityValue::Unsupported { reason } | CapabilityValue::Unavailable { reason } => {
                return Ok(AdmissionResult::Rejected(
                    AdmissionRejection::EstimationUnsupported { reason },
                ))
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
        let state = self.estimate_runtime_state(
            request.input,
            request.max_output_tokens,
            request.batch_size,
        )?;
        apply_admission_memory_policy(request, requested_positions, state, available)
    }
}

fn apply_admission_memory_policy(
    request: AdmissionRequest,
    requested_positions: u64,
    state: RuntimeStateEstimate,
    available: Option<&AvailableMemory>,
) -> Result<AdmissionResult, CapabilityError> {
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
    let available_bytes = match available {
        Some(report) => match &report.available_memory_bytes {
            CapabilityValue::Available { value, .. } => Some(*value),
            CapabilityValue::Unsupported { reason } | CapabilityValue::Unavailable { reason } => {
                return Ok(AdmissionResult::Rejected(
                    AdmissionRejection::AvailableMemoryUnavailable {
                        reason: reason.clone(),
                    },
                ))
            }
        },
        None => None,
    };
    if let Some(available_bytes) = available_bytes {
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
        available_memory_bytes: available_bytes,
    }))
}

#[cfg(target_os = "macos")]
fn macos_memory() -> Result<AvailableMemory, CapabilityError> {
    unsafe extern "C" {
        fn os_proc_available_memory() -> usize;
    }

    let name = c"hw.memsize";
    let mut total = 0u64;
    let mut size = std::mem::size_of::<u64>();
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut total as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    let physical_memory_bytes = if status == 0 && size == std::mem::size_of::<u64>() {
        CapabilityValue::Available {
            value: total,
            kind: MeasurementKind::Exact,
            source: "macOS sysctl hw.memsize",
        }
    } else {
        CapabilityValue::Unavailable {
            reason: std::io::Error::last_os_error().to_string(),
        }
    };
    let available = unsafe { os_proc_available_memory() };
    let available_memory_bytes = match u64::try_from(available) {
        Ok(value) if value > 0 => CapabilityValue::Available {
            value,
            kind: MeasurementKind::Estimated,
            source: "macOS os_proc_available_memory",
        },
        _ => CapabilityValue::Unavailable {
            reason: "os_proc_available_memory returned no usable value".into(),
        },
    };
    Ok(AvailableMemory {
        physical_memory_bytes,
        available_memory_bytes,
        physical_semantics: if cfg!(target_arch = "aarch64") {
            PhysicalMemorySemantics::Unified
        } else {
            PhysicalMemorySemantics::Unknown
        },
    })
}

#[cfg(target_os = "linux")]
fn linux_memory() -> Result<AvailableMemory, CapabilityError> {
    let contents = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let value = |name: &str| -> Option<u64> {
        contents.lines().find_map(|line| {
            let (key, rest) = line.split_once(':')?;
            if key != name {
                return None;
            }
            let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            kib.checked_mul(1024)
        })
    };
    Ok(AvailableMemory {
        physical_memory_bytes: value("MemTotal").map_or_else(
            || CapabilityValue::Unavailable {
                reason: "/proc/meminfo has no MemTotal".into(),
            },
            |value| CapabilityValue::Available {
                value,
                kind: MeasurementKind::Exact,
                source: "Linux /proc/meminfo MemTotal",
            },
        ),
        available_memory_bytes: value("MemAvailable").map_or_else(
            || CapabilityValue::Unavailable {
                reason: "/proc/meminfo has no MemAvailable".into(),
            },
            |value| CapabilityValue::Available {
                value,
                kind: MeasurementKind::Estimated,
                source: "Linux /proc/meminfo MemAvailable",
            },
        ),
        physical_semantics: PhysicalMemorySemantics::Unknown,
    })
}

#[cfg(target_os = "windows")]
fn windows_memory() -> Result<AvailableMemory, CapabilityError> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_physical: u64,
        available_physical: u64,
        total_page_file: u64,
        available_page_file: u64,
        total_virtual: u64,
        available_virtual: u64,
        available_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }

    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_physical: 0,
        available_physical: 0,
        total_page_file: 0,
        available_page_file: 0,
        total_virtual: 0,
        available_virtual: 0,
        available_extended_virtual: 0,
    };
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return Ok(AvailableMemory {
            physical_memory_bytes: CapabilityValue::Unavailable {
                reason: "GlobalMemoryStatusEx failed".into(),
            },
            available_memory_bytes: CapabilityValue::Unavailable {
                reason: "GlobalMemoryStatusEx failed".into(),
            },
            physical_semantics: PhysicalMemorySemantics::Unknown,
        });
    }
    Ok(AvailableMemory {
        physical_memory_bytes: CapabilityValue::Available {
            value: status.total_physical,
            kind: MeasurementKind::Exact,
            source: "Windows GlobalMemoryStatusEx ullTotalPhys",
        },
        available_memory_bytes: CapabilityValue::Available {
            value: status.available_physical,
            kind: MeasurementKind::Estimated,
            source: "Windows GlobalMemoryStatusEx ullAvailPhys",
        },
        physical_semantics: PhysicalMemorySemantics::Unknown,
    })
}

/// Queries system memory that can be used as an admission signal.
///
/// Apple Silicon reports one unified physical capacity; logical host/device
/// residency tiers must not be added as independent physical capacities.
pub fn available_memory() -> Result<AvailableMemory, CapabilityError> {
    #[cfg(target_os = "macos")]
    {
        macos_memory()
    }
    #[cfg(target_os = "linux")]
    {
        linux_memory()
    }
    #[cfg(target_os = "windows")]
    {
        windows_memory()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        Ok(AvailableMemory {
            physical_memory_bytes: CapabilityValue::Unavailable {
                reason: "portable physical-memory query is not implemented on this platform".into(),
            },
            available_memory_bytes: CapabilityValue::Unavailable {
                reason: "portable available-memory query is not implemented on this platform"
                    .into(),
            },
            physical_semantics: PhysicalMemorySemantics::Unknown,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::llama;
    use safemlx::{Device, DeviceType};
    use serde_json::json;

    #[test]
    fn qwen2_runtime_state_splits_full_and_sliding_gqa_layers() {
        let args = crate::api::dense_qwen::config_from_hf_value(&json!({
            "model_type": "qwen2", "hidden_size": 16, "num_hidden_layers": 6,
            "intermediate_size": 32, "num_attention_heads": 4,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 64,
            "max_position_embeddings": 128, "rope_theta": 10000.0,
            "tie_word_embeddings": false, "use_sliding_window": true,
            "sliding_window": 8, "max_window_layers": 4
        }))
        .unwrap();
        let (_, _, strategy, _, estimate) = dense_qwen_spec(&args, false).unwrap();
        assert_eq!(
            strategy,
            CacheStateStrategy::MixedKv {
                full_layers: 4,
                sliding: vec![SlidingWindowLayerCount {
                    layers: 2,
                    window: 8,
                }],
            }
        );
        assert_eq!(estimate.growing.len(), 2);
        assert_eq!(estimate.growing[0].layers, 4);
        assert_eq!(estimate.growing[0].window, None);
        assert_eq!(estimate.growing[1].layers, 2);
        assert_eq!(estimate.growing[1].window, Some(8));
        // 2 KV heads x 4 values per head x key/value.
        assert_eq!(estimate.growing[1].scalars_per_position, 16);
    }

    #[test]
    fn qwen2_runtime_state_groups_arbitrary_distinct_windows_exactly() {
        let mut args = crate::api::dense_qwen::config_from_hf_value(&json!({
            "model_type": "qwen2", "hidden_size": 16, "num_hidden_layers": 4,
            "intermediate_size": 32, "num_attention_heads": 4,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 64,
            "max_position_embeddings": 128, "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }))
        .unwrap();
        args.attention_schedule = crate::runtime::attention::LayerSchedule::new(
            4,
            vec![
                crate::runtime::attention::AttentionPolicy::sliding(4).unwrap(),
                crate::runtime::attention::AttentionPolicy::Full,
                crate::runtime::attention::AttentionPolicy::sliding(8).unwrap(),
                crate::runtime::attention::AttentionPolicy::sliding(4).unwrap(),
            ],
        )
        .unwrap();
        let (_, _, strategy, _, layout) = dense_qwen_spec(&args, false).unwrap();
        assert_eq!(
            strategy,
            CacheStateStrategy::MixedKv {
                full_layers: 1,
                sliding: vec![
                    SlidingWindowLayerCount {
                        layers: 2,
                        window: 4,
                    },
                    SlidingWindowLayerCount {
                        layers: 1,
                        window: 8,
                    },
                ],
            }
        );
        let state = estimate_architecture_state(&layout, InputTokenCount::text(10), 0, 2).unwrap();
        assert_eq!(state.assumptions.sliding_window_bounds, vec![4, 8]);
        assert_eq!(state.context_state_bytes, (10 + 2 * 4 + 8) * 16 * 2 * 4);
    }

    #[test]
    fn lfm2_runtime_state_uses_the_normalized_hybrid_schedule() {
        let args = lfm2::model_args_from_config_value(&json!({
            "model_type": "lfm2", "vocab_size": 32, "hidden_size": 16,
            "intermediate_size": 24, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "max_position_embeddings": 128, "norm_eps": 1e-5,
            "conv_L_cache": 3, "block_auto_adjust_ff_dim": false,
            "layer_types": ["conv", "full_attention", "conv"]
        }))
        .unwrap();
        let (_, _, strategy, _, estimate) = lfm2_spec(&args).unwrap();
        assert_eq!(
            strategy,
            CacheStateStrategy::HybridRecurrent {
                full_attention_layers: 1,
                sliding_attention: Vec::new(),
                recurrent_layers: 2,
            }
        );
        assert_eq!(estimate.fixed_scalars_per_batch, 64);
        assert_eq!(estimate.growing.len(), 1);
        assert_eq!(estimate.growing[0].layers, 1);
        assert_eq!(estimate.growing[0].scalars_per_position, 16);
        assert_eq!(estimate.growing[0].window, None);
    }

    #[test]
    fn gpt_oss_runtime_state_uses_exact_schedule_and_distinct_windows() {
        use crate::runtime::attention::{AttentionPolicy, LayerSchedule};

        let mut args = gpt_oss::model_args_from_config_value(&json!({
            "model_type": "gpt_oss", "hidden_size": 32,
            "intermediate_size": 32, "num_hidden_layers": 4,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 16, "vocab_size": 32, "num_local_experts": 2,
            "num_experts_per_tok": 1, "rms_norm_eps": 1e-5,
            "sliding_window": 5, "max_position_embeddings": 128,
            "layer_types": [
                "sliding_attention", "full_attention",
                "sliding_attention", "full_attention"
            ],
            "quantization_config": {"quant_method": "mxfp4"}
        }))
        .unwrap();
        args.attention_schedule = LayerSchedule::new(
            4,
            vec![
                AttentionPolicy::sliding(3).unwrap(),
                AttentionPolicy::Full,
                AttentionPolicy::sliding(5).unwrap(),
                AttentionPolicy::Full,
            ],
        )
        .unwrap();
        let (_, _, strategy, _, estimate) = gpt_oss_spec(&args).unwrap();
        assert_eq!(
            strategy,
            CacheStateStrategy::MixedKv {
                full_layers: 2,
                sliding: vec![
                    SlidingWindowLayerCount {
                        layers: 1,
                        window: 3,
                    },
                    SlidingWindowLayerCount {
                        layers: 1,
                        window: 5,
                    },
                ],
            }
        );
        assert_eq!(estimate.growing.len(), 3);
        assert_eq!(estimate.growing[0].window, None);
        assert_eq!(estimate.growing[1].window, Some(3));
        assert_eq!(estimate.growing[2].window, Some(5));
    }

    #[test]
    fn nemotron_runtime_state_tracks_mixed_recurrent_kv_and_stateless_layers() {
        let args = nemotron_h::model_args_from_config_value(&json!({
            "model_type": "nemotron_h", "vocab_size": 32, "hidden_size": 8,
            "intermediate_size": 12, "num_hidden_layers": 4,
            "hybrid_override_pattern": "M*-E", "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 4,
            "max_position_embeddings": 128, "sliding_window": 5,
            "mamba_num_heads": 2, "mamba_head_dim": 4, "n_groups": 1,
            "ssm_state_size": 4, "conv_kernel": 3, "chunk_size": 2,
            "moe_intermediate_size": 6,
            "moe_shared_expert_intermediate_size": 10,
            "n_routed_experts": 2, "n_shared_experts": 1,
            "num_experts_per_tok": 2, "mlp_hidden_act": "relu2",
            "mamba_hidden_act": "silu"
        }))
        .unwrap();
        let (_, _, strategy, _, estimate) = nemotron_spec(&args).unwrap();
        assert_eq!(
            strategy,
            CacheStateStrategy::HybridRecurrent {
                full_attention_layers: 0,
                sliding_attention: vec![SlidingWindowLayerCount {
                    layers: 1,
                    window: 5,
                }],
                recurrent_layers: 1,
            }
        );
        assert_eq!(estimate.fixed_scalars_per_batch, 64);
        assert_eq!(estimate.growing.len(), 1);
        assert_eq!(estimate.growing[0].layers, 1);
        assert_eq!(estimate.growing[0].scalars_per_position, 8);
        assert_eq!(estimate.growing[0].window, Some(5));
        let state =
            estimate_architecture_state(&estimate, InputTokenCount::text(10), 0, 2).unwrap();
        assert_eq!(state.fixed_state_bytes, 512);
        assert_eq!(state.context_state_bytes, 320);
        assert_eq!(state.bytes_per_position_per_batch, 0);
        assert_eq!(state.assumptions.sliding_window_bounds, vec![5]);
    }

    #[test]
    fn qwen_hybrid_runtime_state_uses_the_normalized_schedule() {
        let args = qwen3_5_moe::model_args_from_config_value(&json!({
            "model_type": "qwen3_next", "vocab_size": 32, "hidden_size": 16,
            "num_hidden_layers": 4, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 8,
            "max_position_embeddings": 128, "intermediate_size": 32,
            "num_experts": 0, "linear_conv_kernel_dim": 3,
            "linear_key_head_dim": 4, "linear_value_head_dim": 4,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "layer_types": [
                "full_attention", "linear_attention",
                "linear_attention", "full_attention"
            ]
        }))
        .unwrap();
        let (_, _, strategy, _, estimate) = qwen_hybrid_spec(&args, false).unwrap();
        assert_eq!(
            strategy,
            CacheStateStrategy::HybridRecurrent {
                full_attention_layers: 2,
                sliding_attention: Vec::new(),
                recurrent_layers: 2,
            }
        );
        assert_eq!(estimate.fixed_scalars_per_batch, 160);
        assert_eq!(estimate.growing.len(), 1);
        assert_eq!(estimate.growing[0].layers, 2);
        assert_eq!(estimate.growing[0].scalars_per_position, 16);
    }

    fn tiny_llama(kv_heads: i32, sliding_window: Option<i32>) -> llama::ModelArgs {
        llama::ModelArgs {
            model_type: "mistral".into(),
            hidden_size: 32,
            num_hidden_layers: 2,
            intermediate_size: 64,
            num_attention_heads: 4,
            rms_norm_eps: 1e-5,
            vocab_size: 64,
            num_key_value_heads: kv_heads,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            rope_traditional: false,
            head_dim: 8,
            tie_word_embeddings: true,
            attention_bias: false,
            mlp_bias: false,
            rope_scaling: None,
            sliding_window,
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        }
    }

    fn tiny_gemma4() -> gemma4::ModelArgs {
        gemma4::ModelArgs {
            model_type: "gemma4_unified".into(),
            hidden_size: 8,
            num_hidden_layers: 4,
            intermediate_size: 16,
            use_double_wide_mlp: false,
            feed_forward_lengths: None,
            num_attention_heads: 2,
            rms_norm_eps: 1e-5,
            vocab_size: 32,
            pad_token_id: 0,
            num_key_value_heads: 1,
            num_global_key_value_heads: Some(1),
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            head_dim: 4,
            global_head_dim: Some(4),
            tie_word_embeddings: true,
            attention_bias: false,
            attention_k_eq_v: false,
            quantized: false,
            weight_quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
            quantization_group_size: 64,
            quantization_bits: 4,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            num_kv_shared_layers: 1,
            attention_schedule: crate::runtime::attention::LayerSchedule::new(
                4,
                vec![
                    AttentionPolicy::sliding(4).unwrap(),
                    AttentionPolicy::Full,
                    AttentionPolicy::sliding(4).unwrap(),
                    AttentionPolicy::Full,
                ],
            )
            .unwrap(),
            final_logit_softcapping: None,
            enable_moe_block: false,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            rope_scaling: None,
            rope_parameters: None,
        }
    }

    fn estimate(
        fixed: u64,
        components: Vec<GrowingState>,
        positions: u64,
        batch: u64,
    ) -> Result<RuntimeStateEstimate, CapabilityError> {
        estimate_architecture_state(
            &ArchitectureEstimate {
                fixed_scalars_per_batch: fixed,
                growing: components,
                hidden_size: 1,
                allocation_granularity: 1,
                completeness: EstimationCompleteness::Complete,
            },
            InputTokenCount::text(positions),
            0,
            batch,
        )
    }

    #[test]
    fn standard_kv_and_gqa_use_kv_head_count() {
        let (_, _, strategy, _, llama_layout) = llama_spec(&tiny_llama(4, None), false).unwrap();
        assert_eq!(strategy, CacheStateStrategy::FullKv);
        let llama =
            estimate_architecture_state(&llama_layout, InputTokenCount::text(10), 0, 1).unwrap();
        assert_eq!(llama.requested_state_bytes, 2 * 2 * 4 * 8 * 10 * 4);

        let (_, _, _, _, gqa_layout) = llama_spec(&tiny_llama(1, None), false).unwrap();
        let gqa =
            estimate_architecture_state(&gqa_layout, InputTokenCount::text(10), 0, 1).unwrap();
        assert_eq!(gqa.requested_state_bytes, llama.requested_state_bytes / 4);
    }

    #[test]
    fn sliding_window_bounds_only_bounded_layers() {
        let estimate = estimate(
            0,
            vec![
                GrowingState {
                    layers: 1,
                    scalars_per_position: 16,
                    window: None,
                },
                GrowingState {
                    layers: 3,
                    scalars_per_position: 16,
                    window: Some(4),
                },
            ],
            10,
            1,
        )
        .unwrap();
        assert_eq!(estimate.context_state_bytes, (10 + 3 * 4) * 16 * 4);
        assert_eq!(estimate.bytes_per_position_per_batch, 16 * 4);
    }

    #[test]
    fn compressed_mla_uses_latent_plus_rotary_width() {
        let estimate = estimate(
            0,
            vec![GrowingState {
                layers: 3,
                scalars_per_position: 12 + 4,
                window: None,
            }],
            5,
            2,
        )
        .unwrap();
        assert_eq!(estimate.requested_state_bytes, 3 * 16 * 5 * 2 * 4);
    }

    #[test]
    fn kimi_linear_accounts_for_bounded_kda_and_growing_mla_state() {
        let args = kimi_linear::parse_config_value(json!({
            "model_type": "kimi_linear",
            "vocab_size": 64,
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 16,
            "head_dim": 4,
            "model_max_length": 128,
            "linear_attn_config": {
                "kda_layers": [1],
                "full_attn_layers": [2],
                "num_heads": 2,
                "head_dim": 4,
                "short_conv_kernel_size": 2
            },
            "num_experts": 4,
            "moe_intermediate_size": 8,
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "v_head_dim": 2,
            "mla_use_nope": true,
            "num_experts_per_token": 2,
            "routed_scaling_factor": 1.0,
            "first_k_dense_replace": 1,
            "num_expert_group": 1,
            "topk_group": 1
        }))
        .unwrap();
        let (native, effective, strategy, modalities, estimate) = kimi_linear_spec(&args).unwrap();
        assert_eq!(native.value(), Some(&128));
        assert_eq!(effective.value(), Some(&128));
        assert_eq!(
            strategy,
            CacheStateStrategy::HybridRecurrent {
                full_attention_layers: 1,
                sliding_attention: Vec::new(),
                recurrent_layers: 1,
            }
        );
        assert_eq!(modalities, InputModalities::TEXT);
        assert_eq!(estimate.fixed_scalars_per_batch, 56);
        assert_eq!(estimate.growing.len(), 1);
        assert_eq!(estimate.growing[0].layers, 1);
        assert_eq!(estimate.growing[0].scalars_per_position, 6);
        assert_eq!(estimate.allocation_granularity, 256);
    }

    #[test]
    fn hybrid_fixed_and_attention_state_are_separate() {
        let estimate = estimate(
            100,
            vec![GrowingState {
                layers: 2,
                scalars_per_position: 8,
                window: None,
            }],
            5,
            3,
        )
        .unwrap();
        assert_eq!(estimate.fixed_state_bytes, 100 * 3 * 4);
        assert_eq!(estimate.context_state_bytes, 2 * 8 * 5 * 3 * 4);
    }

    #[test]
    fn multimodal_positions_are_distinct_from_text_tokens() {
        let count = InputTokenCount::prepared(7, 12, 19, 1_024, MeasurementKind::Conservative);
        assert_eq!(
            count.text_tokens + count.media_positions,
            count.model_positions
        );
        assert_eq!(count.media_execution_workspace_bytes(), 1_024);
        assert_eq!(
            count.media_execution_workspace_kind(),
            MeasurementKind::Conservative
        );
    }

    #[test]
    fn gemma4_prepared_patch_positions_ignore_padding() {
        let positions = Array::from_slice(&[0, 0, 1, 0, 0, 1, 1, 1, -1, -1], &[1, 5, 2]);
        assert_eq!(gemma_valid_patch_count(&positions, "gemma4").unwrap(), 4);
    }

    #[test]
    fn gemma4_prepared_media_bounds_vision_and_audio_workspaces() {
        let vision_config = crate::architectures::gemma4::vision::Gemma4VisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            patch_size: 1,
            pooling_kernel_size: 2,
            position_embedding_size: 4,
            rms_norm_eps: 1e-5,
            hidden_activation: "gelu_pytorch_tanh".into(),
            standardize: false,
            rope_parameters: None,
        };
        let vision = Array::from_slice(&[0.0f32; 15], &[1, 5, 3]);
        let patch_positions = Array::from_slice(&[0, 0, 1, 0, 0, 1, 1, 1, -1, -1], &[1, 5, 2]);
        let (vision_positions, vision_workspace) = gemma_vision_workspace(
            &vision_config,
            8,
            &vision,
            super::super::input::InputMetadata::patch_position_ids(&patch_positions),
            "gemma4",
        )
        .unwrap();
        assert_eq!(vision_positions, 1);
        assert!(vision_workspace > array_bytes(&vision, "test vision bytes").unwrap());

        let audio_config = crate::architectures::gemma4::audio::Gemma4AudioConfig {
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            output_proj_dims: 8,
            conv_kernel_size: 3,
            attention_chunk_size: 4,
            attention_context_left: 3,
            attention_context_right: 0,
            attention_invalid_logits_value: -1e9,
            attention_logit_cap: 50.0,
            residual_weight: 1.0,
            rms_norm_eps: 1e-5,
            subsampling_conv_channels: vec![4, 8],
        };
        let audio = Array::from_slice(&[0.0f32; 8 * 128], &[1, 8, 128]);
        let audio_mask =
            Array::from_slice(&[true, true, true, true, true, true, false, false], &[1, 8]);
        let (audio_positions, audio_workspace) = gemma_audio_workspace(
            &audio_config,
            8,
            &audio,
            super::super::input::InputMetadata::audio_mask(&audio_mask),
            "gemma4",
        )
        .unwrap();
        assert_eq!(audio_positions, 2);
        assert!(audio_workspace > array_bytes(&audio, "test audio bytes").unwrap());
    }

    #[test]
    fn qwen_prepared_grid_bounds_vision_workspace() {
        let config = crate::architectures::qwen::vl::vision::VisionConfig {
            depth: 2,
            hidden_size: 8,
            hidden_act: "silu".into(),
            intermediate_size: 16,
            num_heads: 2,
            num_position_embeddings: 4,
            in_channels: 3,
            patch_size: 1,
            spatial_merge_size: 2,
            temporal_patch_size: 1,
            window_size: 4,
            out_hidden_size: 8,
            fullatt_block_indexes: vec![1],
            deepstack_visual_indexes: vec![0],
        };
        let payload = Array::from_slice(&[0.0f32; 12], &[4, 3]);
        let grid = Array::from_slice(&[1, 2, 2], &[1, 3]);
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let (positions, workspace) = qwen_vision_workspace(
            &config,
            Modality::Image,
            &payload,
            super::super::input::InputMetadata::qwen_grid_thw(&grid),
            &stream,
            "qwen3_vl",
        )
        .unwrap();
        assert_eq!(positions, 1);
        assert!(workspace > payload.nbytes() as u64);
    }

    fn tiny_inkling() -> inkling::ModelArgs {
        inkling::model_args_from_config_value(&json!({
            "model_type":"inkling_mm_model",
            "text_config":{
                "hidden_size":32,"num_hidden_layers":3,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
                "sliding_window_size":8,"local_layer_ids":[0,1],"dense_mlp_idx":1,
                "sconv_kernel_size":4,"d_rel":4,"rel_extent":16,
                "intermediate_size":24,"dense_intermediate_size":48,
                "n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
                "route_scale":8.0,"use_sconv":true,"use_embed_norm":true,
                "shared_expert_sink":true,"use_gate_bias":true,"norm_after_topk":true,
                "use_global_scale":true,"gate_activation":"sigmoid"
            },
            "audio_config":{
                "decoder_dmodel":32,"n_mel_bins":80,"mel_vocab_size":16
            },
            "vision_config":{
                "decoder_dmodel":32,"patch_size":40,"temporal_patch_size":2,
                "n_channels":3,"n_layers":4
            }
        }))
        .unwrap()
    }

    #[test]
    fn inkling_runtime_state_groups_the_exact_ordered_schedule() {
        use crate::architectures::inkling::model::{FeedForwardPolicy, LayerPolicy};
        use crate::runtime::attention::LayerSchedule;

        let mut args = tiny_inkling();
        args.text_config.num_key_value_heads = 1;
        args.text_config.swa_num_key_value_heads = Some(2);
        args.text_config.layer_schedule = LayerSchedule::new(
            3,
            vec![
                LayerPolicy {
                    attention: AttentionPolicy::Full,
                    feed_forward: FeedForwardPolicy::Dense,
                },
                LayerPolicy {
                    attention: AttentionPolicy::sliding(3).unwrap(),
                    feed_forward: FeedForwardPolicy::SparseMoe,
                },
                LayerPolicy {
                    attention: AttentionPolicy::sliding(5).unwrap(),
                    feed_forward: FeedForwardPolicy::SparseMoe,
                },
            ],
        )
        .unwrap();

        let (_, _, strategy, _, layout) = inkling_spec(&args).unwrap();
        assert_eq!(
            strategy,
            CacheStateStrategy::Multimodal {
                decoder: Box::new(CacheStateStrategy::MixedKv {
                    full_layers: 1,
                    sliding: vec![
                        SlidingWindowLayerCount {
                            layers: 1,
                            window: 3,
                        },
                        SlidingWindowLayerCount {
                            layers: 1,
                            window: 5,
                        },
                    ],
                }),
                media_consumes_decoder_positions: true,
            }
        );
        let state = estimate_architecture_state(&layout, InputTokenCount::text(10), 0, 2).unwrap();
        assert_eq!(state.assumptions.sliding_window_bounds, vec![3, 5]);
        // Full KV: 1 x 10 x (1 head x 8 x K/V). Sliding KV: (3 + 5) x
        // (2 heads x 8 x K/V), all for two batches of f32 state.
        assert_eq!(state.context_state_bytes, (10 * 16 + (3 + 5) * 32) * 2 * 4);
    }

    #[test]
    fn inkling_prepared_media_bounds_hmlp_and_dmel_workspaces() {
        let args = tiny_inkling();
        let image = Array::from_slice(&[0.0f32; 2 * 40 * 40 * 3], &[1, 2, 40, 40, 3]);
        let (image_positions, image_workspace) = inkling_workspace(
            &args,
            Modality::Image,
            &image,
            super::super::input::InputMetadata::empty(),
            "inkling",
        )
        .unwrap();
        assert_eq!(image_positions, 1);
        assert!(image_workspace > image.nbytes() as u64);

        let audio = Array::from_slice(&[0u32; 3 * 80], &[1, 3, 80]);
        let mask = Array::from_slice(&[true, true, false], &[1, 3]);
        let (audio_positions, audio_workspace) = inkling_workspace(
            &args,
            Modality::Audio,
            &audio,
            super::super::input::InputMetadata::audio_mask(&mask),
            "inkling",
        )
        .unwrap();
        assert_eq!(audio_positions, 2);
        assert!(audio_workspace > audio.nbytes() as u64);
    }

    #[test]
    fn gemma4_shared_and_sliding_layers_use_full_chunked_kv_backing() {
        let modalities = InputModalities {
            text: true,
            image: true,
            audio: false,
            video: false,
        };
        let (_, _, strategy, _, layout) = gemma4_spec(&tiny_gemma4(), modalities).unwrap();
        assert_eq!(
            strategy,
            CacheStateStrategy::Multimodal {
                decoder: Box::new(CacheStateStrategy::SharedFullKv {
                    cached_layers: 3,
                    shared_layers: 1,
                    full_attention_layers: 2,
                    sliding_attention: vec![SlidingWindowLayerCount {
                        window: 4,
                        layers: 2,
                    }],
                }),
                media_consumes_decoder_positions: true,
            }
        );
        let estimate = estimate_architecture_state(
            &layout,
            InputTokenCount::prepared(5, 3, 8, 1_024, MeasurementKind::Conservative),
            2,
            2,
        )
        .unwrap();
        assert_eq!(estimate.assumptions.allocation_granularity, 256);
        assert!(estimate.assumptions.sliding_window_bounds.is_empty());
        assert_eq!(estimate.context_state_bytes, 3 * 2 * 2 * 4 * 256 * 4);
        assert_eq!(estimate.multimodal_embedding_bytes, 3 * 8 * 2 * 4);
        assert_eq!(estimate.media_execution_workspace_bytes, 2_048);
        assert_eq!(estimate.completeness, EstimationCompleteness::Conservative);
    }

    #[test]
    fn gemma4_capabilities_report_each_exact_sliding_window() {
        let mut args = tiny_gemma4();
        args.attention_schedule = crate::runtime::attention::LayerSchedule::new(
            4,
            vec![
                AttentionPolicy::sliding(3).unwrap(),
                AttentionPolicy::Full,
                AttentionPolicy::sliding(5).unwrap(),
                AttentionPolicy::Full,
            ],
        )
        .unwrap();
        let (_, _, strategy, _, _) = gemma4_spec(&args, text_modalities()).unwrap();
        assert_eq!(
            strategy,
            CacheStateStrategy::SharedFullKv {
                cached_layers: 3,
                shared_layers: 1,
                full_attention_layers: 2,
                sliding_attention: vec![
                    SlidingWindowLayerCount {
                        window: 3,
                        layers: 1,
                    },
                    SlidingWindowLayerCount {
                        window: 5,
                        layers: 1,
                    },
                ],
            }
        );
    }

    #[test]
    fn checked_arithmetic_reports_overflow() {
        assert_eq!(
            checked_mul(u64::MAX, 2, "synthetic overflow"),
            Err(CapabilityError::ArithmeticOverflow {
                operation: "synthetic overflow"
            })
        );

        let layout = ArchitectureEstimate {
            fixed_scalars_per_batch: 0,
            growing: Vec::new(),
            hidden_size: 1,
            allocation_granularity: 1,
            completeness: EstimationCompleteness::Complete,
        };
        assert!(matches!(
            estimate_architecture_state(
                &layout,
                InputTokenCount::prepared(0, 1, 1, u64::MAX, MeasurementKind::Conservative,),
                0,
                2,
            ),
            Err(CapabilityError::ArithmeticOverflow {
                operation: "media execution workspace times batch"
            })
        ));
    }

    #[test]
    fn unavailable_memory_is_not_zero() {
        let value: CapabilityValue<u64> = CapabilityValue::Unavailable {
            reason: "synthetic".into(),
        };
        assert_eq!(value.value(), None);
    }

    #[test]
    fn apple_unified_semantics_do_not_create_two_capacities() {
        let report = AvailableMemory {
            physical_memory_bytes: CapabilityValue::Available {
                value: 16,
                kind: MeasurementKind::Exact,
                source: "test",
            },
            available_memory_bytes: CapabilityValue::Available {
                value: 8,
                kind: MeasurementKind::Estimated,
                source: "test",
            },
            physical_semantics: PhysicalMemorySemantics::Unified,
        };
        assert_eq!(report.physical_memory_bytes.value(), Some(&16));
        assert_eq!(report.physical_semantics, PhysicalMemorySemantics::Unified);
    }

    #[test]
    fn dtype_assumption_is_explicit() {
        let estimate = estimate(0, Vec::new(), 1, 1).unwrap();
        assert_eq!(estimate.assumptions.state_dtype_bytes.get(), 4);
    }

    #[test]
    fn capability_value_never_invents_default() {
        let unsupported: CapabilityValue<u64> = CapabilityValue::Unsupported {
            reason: "not supported".into(),
        };
        assert!(unsupported.value().is_none());
    }

    #[test]
    fn context_scaling_distinguishes_native_and_effective_limits() {
        let linear =
            std::collections::HashMap::from([("factor".into(), FloatOrString::Float(4.0))]);
        let (native, effective) = context_from_rope(2_048, Some(&linear)).unwrap();
        assert_eq!(native.value(), Some(&2_048));
        assert_eq!(effective.value(), Some(&8_192));

        let yarn = std::collections::HashMap::from([
            ("factor".into(), FloatOrString::Float(40.0)),
            (
                "original_max_position_embeddings".into(),
                FloatOrString::Float(4_096.0),
            ),
        ]);
        let (native, effective) = context_from_rope(163_840, Some(&yarn)).unwrap();
        assert_eq!(native.value(), Some(&4_096));
        assert_eq!(effective.value(), Some(&163_840));
    }

    #[test]
    fn cache_allocation_rounding_is_checked_and_explicit() {
        assert_eq!(rounded_cache_positions(0, 256).unwrap(), 0);
        assert_eq!(rounded_cache_positions(257, 256).unwrap(), 512);
        assert!(matches!(
            rounded_cache_positions(u64::MAX, 256),
            Err(CapabilityError::ArithmeticOverflow {
                operation: "cache allocation rounding"
            })
        ));
        assert!(matches!(
            rounded_cache_positions(1, 0),
            Err(CapabilityError::InvalidConfiguration {
                field: "allocation_granularity",
                ..
            })
        ));
    }

    fn request(prompt: u64, output: u64, budget: Option<u64>) -> AdmissionRequest {
        AdmissionRequest {
            input: InputTokenCount::text(prompt),
            max_output_tokens: output,
            batch_size: 1,
            safety_reserve_bytes: 10,
            application_memory_budget_bytes: budget,
            require_complete_estimate: true,
        }
    }

    fn reject_context(
        maximum: u64,
        request: AdmissionRequest,
    ) -> Result<Option<AdmissionRejection>, CapabilityError> {
        if request.input.model_positions > maximum {
            return Ok(Some(AdmissionRejection::PromptExceedsContext {
                prompt_positions: request.input.model_positions,
                maximum_positions: maximum,
            }));
        }
        let total = checked_add(
            request.input.model_positions,
            request.max_output_tokens,
            "test admission positions",
        )?;
        Ok(
            (total > maximum).then_some(AdmissionRejection::OutputHeadroomExceedsContext {
                prompt_positions: request.input.model_positions,
                output_tokens: request.max_output_tokens,
                maximum_positions: maximum,
            }),
        )
    }

    #[test]
    fn context_limit_rejection_is_structured() {
        assert_eq!(
            reject_context(8, request(9, 0, None)).unwrap(),
            Some(AdmissionRejection::PromptExceedsContext {
                prompt_positions: 9,
                maximum_positions: 8,
            })
        );
    }

    #[test]
    fn output_headroom_rejection_is_structured() {
        assert_eq!(
            reject_context(8, request(7, 2, None)).unwrap(),
            Some(AdmissionRejection::OutputHeadroomExceedsContext {
                prompt_positions: 7,
                output_tokens: 2,
                maximum_positions: 8,
            })
        );
    }

    #[test]
    fn memory_budget_rejection_is_structured() {
        let state = estimate(0, Vec::new(), 1, 1).unwrap();
        assert_eq!(
            apply_admission_memory_policy(request(1, 0, Some(9)), 1, state, None).unwrap(),
            AdmissionResult::Rejected(AdmissionRejection::MemoryBudgetExceeded {
                required_bytes: 10,
                budget_bytes: 9,
            })
        );
    }

    #[test]
    fn complete_policy_accepts_conservative_media_bound() {
        let layout = ArchitectureEstimate {
            fixed_scalars_per_batch: 0,
            growing: Vec::new(),
            hidden_size: 8,
            allocation_granularity: 1,
            completeness: EstimationCompleteness::Complete,
        };
        let input = InputTokenCount::prepared(1, 1, 2, 100, MeasurementKind::Conservative);
        let state = estimate_architecture_state(&layout, input, 0, 1).unwrap();
        assert_eq!(state.completeness, EstimationCompleteness::Conservative);
        let admission = apply_admission_memory_policy(
            AdmissionRequest {
                input,
                max_output_tokens: 0,
                batch_size: 1,
                safety_reserve_bytes: 0,
                application_memory_budget_bytes: Some(state.requested_state_bytes),
                require_complete_estimate: true,
            },
            2,
            state,
            None,
        )
        .unwrap();
        assert!(matches!(admission, AdmissionResult::Admitted(_)));
    }

    #[test]
    fn unavailable_platform_memory_rejects_when_check_was_requested() {
        let state = estimate(0, Vec::new(), 1, 1).unwrap();
        let unavailable = AvailableMemory {
            physical_memory_bytes: CapabilityValue::Unavailable {
                reason: "synthetic total".into(),
            },
            available_memory_bytes: CapabilityValue::Unavailable {
                reason: "synthetic availability".into(),
            },
            physical_semantics: PhysicalMemorySemantics::Unknown,
        };
        assert_eq!(
            apply_admission_memory_policy(request(1, 0, None), 1, state, Some(&unavailable))
                .unwrap(),
            AdmissionResult::Rejected(AdmissionRejection::AvailableMemoryUnavailable {
                reason: "synthetic availability".into(),
            })
        );
    }
}
