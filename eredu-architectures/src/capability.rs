//! Backend-neutral model capability and runtime-state estimates.
//!
//! Normalized architecture configurations own context limits, accepted
//! modalities, cache/recurrent strategy, and executable state geometry. Concrete
//! backends apply physical scalar widths and report live memory observations.

use std::collections::BTreeMap;

use crate::rotary::RopeValue;
use eredu_core::{
    cache::{LayerCachePolicy, StateTensorRole},
    CacheStateStrategy, CapabilityError, EstimationCompleteness, InputModalities,
    ModelCapabilities, ObservationKind, Observed, SlidingWindowLayerCount, SpeculativeDraftSource,
    StateMemoryLayout,
};
use eredu_runtime::StateLayout as RuntimeStateLayout;

use crate::{
    gemma4, gpt_oss, kimi_linear, lfm2,
    llama::ModelArgs as LlamaModelArgs,
    nemotron_h,
    qwen::{
        hybrid::{HybridConfig as QwenHybridConfig, HybridLayerPolicy as QwenHybridLayerPolicy},
        ModelArgs as QwenModelArgs, QwenVariant,
    },
};
use eredu_core::attention::AttentionPolicy;

fn config_number(config: &std::collections::HashMap<String, RopeValue>, key: &str) -> Option<f32> {
    match config.get(key) {
        Some(RopeValue::Float(value)) => Some(*value),
        Some(RopeValue::String(value)) => value.parse().ok(),
        Some(RopeValue::Bool(_)) | None => None,
    }
}

fn context_from_rope(
    effective: i32,
    rope: Option<&std::collections::HashMap<String, RopeValue>>,
) -> Result<(Observed<u64>, Observed<u64>), CapabilityError> {
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
                    Observed::Unsupported {
                        reason: "RoPE original context is not a positive integer".into(),
                    },
                    Observed::Available {
                        value: effective,
                        kind: ObservationKind::Exact,
                        source: "validated model configuration".into(),
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
                        Observed::Available {
                            value: native,
                            kind: ObservationKind::Exact,
                            source: "validated model configuration".into(),
                        },
                        Observed::Unsupported {
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
        Observed::Available {
            value: native,
            kind: ObservationKind::Exact,
            source: "validated model configuration".into(),
        },
        Observed::Available {
            value: effective,
            kind: ObservationKind::Exact,
            source: "validated model configuration and supported RoPE setup".into(),
        },
    ))
}

fn plain_context(maximum: i32) -> Result<(Observed<u64>, Observed<u64>), CapabilityError> {
    context_from_rope(maximum, None)
}

fn text_modalities() -> InputModalities {
    InputModalities::TEXT
}

fn positive(value: i32, field: &'static str) -> Result<u64, CapabilityError> {
    u64::try_from(value).map_err(|_| CapabilityError::InvalidConfiguration {
        field,
        detail: format!("expected a non-negative value, got {value}"),
    })
}

type Spec = (
    Observed<u64>,
    Observed<u64>,
    CacheStateStrategy,
    InputModalities,
    StateMemoryLayout,
);

fn state_memory_layout<E: std::fmt::Display>(
    layout: Result<RuntimeStateLayout, E>,
    hidden_size: i32,
    allocation_granularity: u64,
    completeness: EstimationCompleteness,
) -> Result<StateMemoryLayout, CapabilityError> {
    let layout = layout.map_err(|error| CapabilityError::InvalidConfiguration {
        field: "state_layout",
        detail: error.to_string(),
    })?;
    state_memory_layout_from_layout(layout, hidden_size, allocation_granularity, completeness)
}

fn state_memory_layout_from_layout(
    layout: RuntimeStateLayout,
    hidden_size: i32,
    allocation_granularity: u64,
    completeness: EstimationCompleteness,
) -> Result<StateMemoryLayout, CapabilityError> {
    StateMemoryLayout::new(
        layout.layers().clone(),
        layout.layer_prefix_offsets(),
        positive(hidden_size, "hidden_size")?,
        allocation_granularity,
        completeness,
    )
}

fn llama_spec(args: &LlamaModelArgs, multimodal: bool) -> Result<Spec, CapabilityError> {
    let context = context_from_rope(args.max_position_embeddings, args.rope_scaling.as_ref())?;
    let layers = positive(args.num_hidden_layers, "num_hidden_layers")?;
    if args.attention_schedule.len() != layers as usize {
        return Err(CapabilityError::InvalidConfiguration {
            field: "attention_schedule",
            detail: format!(
                "has {} layers, expected {layers}",
                args.attention_schedule.len()
            ),
        });
    }
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
    let base = match (full, sliding.as_slice()) {
        (_, []) => CacheStateStrategy::FullKv,
        (0, [only]) => CacheStateStrategy::SlidingKv {
            window: only.window,
        },
        _ => CacheStateStrategy::MixedKv {
            full_layers: full,
            sliding: sliding.clone(),
        },
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
        state_memory_layout(
            crate::llama::state_layout(args),
            args.hidden_size,
            1,
            completeness,
        )?,
    ))
}

fn qwen_spec(args: &QwenModelArgs, multimodal: bool) -> Result<Spec, CapabilityError> {
    let mut spec = llama_spec(
        &LlamaModelArgs {
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
            attention_bias: args.variant == QwenVariant::Qwen2,
            mlp_bias: false,
            rope_scaling: args.rope_scaling.clone(),
            attention_schedule: args.attention_schedule.clone(),
            quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        },
        multimodal,
    )?;
    spec.4 = state_memory_layout(
        crate::qwen::state_layout(args),
        args.hidden_size,
        1,
        EstimationCompleteness::Complete,
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
    }
    Ok(spec)
}

fn muse_glimmer_spec(args: &crate::muse_glimmer::DecoderConfig) -> Result<Spec, CapabilityError> {
    let context = plain_context(args.max_position_embeddings)?;
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
    let state_strategy = CacheStateStrategy::MixedKv {
        full_layers,
        sliding: sliding.clone(),
    };
    Ok((
        context.0,
        context.1,
        state_strategy,
        InputModalities {
            text: true,
            image: true,
            audio: false,
            video: args.weight_convention == crate::muse_glimmer::WeightConvention::HuggingFace,
        },
        state_memory_layout(
            crate::muse_glimmer::state_layout(args),
            args.hidden_size,
            1,
            EstimationCompleteness::Complete,
        )?,
    ))
}

fn neutral_deepseek_v3_spec(args: &crate::deepseek::V3Args) -> Result<Spec, CapabilityError> {
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
    let latent = positive(args.kv_lora_rank, "kv_lora_rank")?;
    let rotary = positive(args.qk_rope_head_dim, "qk_rope_head_dim")?;
    Ok((
        Observed::Available {
            value: native,
            kind: ObservationKind::Exact,
            source: "validated neutral DeepSeek YaRN configuration".into(),
        },
        Observed::Available {
            value: effective,
            kind: ObservationKind::Exact,
            source: "validated neutral DeepSeek configuration".into(),
        },
        CacheStateStrategy::CompressedMla {
            latent_width: latent,
            rotary_width: rotary,
        },
        text_modalities(),
        state_memory_layout(
            crate::deepseek::v3::state_layout(args),
            args.hidden_size,
            256,
            EstimationCompleteness::Complete,
        )?,
    ))
}

fn neutral_deepseek_v4_spec(args: &crate::deepseek::V4Args) -> Result<Spec, CapabilityError> {
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
    let layout = crate::deepseek::v4::state_layout(args).map_err(|error| {
        CapabilityError::InvalidConfiguration {
            field: "state_layout",
            detail: error.to_string(),
        }
    })?;
    let mut window = None;
    let mut pooling_layers = 0_u64;
    for (layer, policy) in layout.layers().iter().enumerate() {
        let attention = match policy {
            LayerCachePolicy::KeyOnly { attention, .. } => attention,
            LayerCachePolicy::KeyOnlyWithFixedState {
                attention, tensors, ..
            } if !tensors.is_empty()
                && tensors
                    .iter()
                    .all(|tensor| matches!(tensor.role, StateTensorRole::Pooling { .. })) =>
            {
                pooling_layers += 1;
                attention
            }
            _ => {
                return Err(CapabilityError::InvalidConfiguration {
                    field: "state_layout",
                    detail: format!(
                        "DeepSeek-V4 layer {layer} is not key-only attention with optional pooling state"
                    ),
                })
            }
        };
        let layer_window = match attention {
            AttentionPolicy::Sliding { window } => u64::from(window.get()),
            AttentionPolicy::Full => {
                return Err(CapabilityError::InvalidConfiguration {
                    field: "state_layout",
                    detail: format!("DeepSeek-V4 layer {layer} has unbounded attention state"),
                })
            }
        };
        if window.is_some_and(|window| window != layer_window) {
            return Err(CapabilityError::InvalidConfiguration {
                field: "state_layout",
                detail: format!("DeepSeek-V4 layer {layer} has an inconsistent attention window"),
            });
        }
        window = Some(layer_window);
    }
    let window = window.ok_or_else(|| CapabilityError::InvalidConfiguration {
        field: "state_layout",
        detail: "DeepSeek-V4 state layout is empty".into(),
    })?;
    let layers = u64::try_from(layout.layers().len()).map_err(|_| {
        CapabilityError::InvalidConfiguration {
            field: "state_layout",
            detail: "DeepSeek-V4 state layer count exceeds the capability range".into(),
        }
    })?;
    Ok((
        Observed::Available {
            value: native,
            kind: ObservationKind::Exact,
            source: "validated neutral DeepSeek-V4 YaRN configuration".into(),
        },
        Observed::Available {
            value: effective,
            kind: ObservationKind::Exact,
            source: "validated neutral DeepSeek-V4 configuration".into(),
        },
        CacheStateStrategy::SlidingKey {
            window,
            layers,
            pooling_layers,
        },
        text_modalities(),
        state_memory_layout_from_layout(
            layout,
            args.hidden_size,
            128,
            EstimationCompleteness::Conservative,
        )?,
    ))
}

fn kimi_linear_spec(args: &kimi_linear::ModelArgs) -> Result<Spec, CapabilityError> {
    let context = plain_context(args.model_max_length)?;
    let attention = args
        .layer_schedule
        .iter()
        .filter(|policy| policy.attention == kimi_linear::AttentionKind::Mla)
        .count() as u64;
    let recurrent = args.layer_schedule.len() as u64 - attention;
    Ok((
        context.0,
        context.1,
        CacheStateStrategy::HybridRecurrent {
            full_attention_layers: attention,
            sliding_attention: Vec::new(),
            recurrent_layers: recurrent,
        },
        text_modalities(),
        state_memory_layout(
            kimi_linear::state_layout(args),
            args.hidden_size,
            256,
            EstimationCompleteness::Complete,
        )?,
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
        state_memory_layout(
            gpt_oss::state_layout(args),
            args.hidden_size,
            1,
            EstimationCompleteness::Complete,
        )?,
    ))
}

fn gemma4_spec(
    args: &gemma4::ModelArgs,
    modalities: InputModalities,
) -> Result<Spec, CapabilityError> {
    let context = context_from_rope(args.max_position_embeddings, args.rope_scaling.as_ref())?;
    let layers = args.num_hidden_layers() as u64;
    let shared = args
        .layer_schedule
        .iter()
        .filter(|policy| !policy.key_value.owns_state())
        .count() as u64;
    let cached = layers - shared;
    let mut sliding_by_window = BTreeMap::<u64, u64>::new();
    for policy in args.layer_schedule.iter() {
        if let Some(window) = policy.attention.window() {
            *sliding_by_window
                .entry(u64::from(window.get()))
                .or_default() += 1;
        }
    }
    let total_sliding = sliding_by_window.values().sum::<u64>();
    let full_attention_layers = layers - total_sliding;
    let sliding_attention = sliding_by_window
        .into_iter()
        .map(|(window, layers)| SlidingWindowLayerCount { window, layers })
        .collect();
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
        state_memory_layout(
            gemma4::state_layout(args),
            args.hidden_size,
            256,
            EstimationCompleteness::Complete,
        )?,
    ))
}

fn inkling_spec(args: &crate::inkling::ModelArgs) -> Result<Spec, CapabilityError> {
    let text = &args.text_config;
    let context = match text.model_max_length {
        Some(maximum) => plain_context(maximum)?,
        None => (
            Observed::Unsupported {
                reason: "Inkling configuration does not expose a native maximum context".into(),
            },
            Observed::Unsupported {
                reason: "Inkling configuration does not expose an effective maximum context".into(),
            },
        ),
    };
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
    let modalities = args.input_modalities();
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
        state_memory_layout(
            (|| {
                let target = crate::inkling::state_layout(args)?;
                let prediction = crate::inkling::mtp_state_layout(args)?;
                crate::inkling::composite_state_layout(&target, prediction.as_ref())
            })(),
            text.hidden_size,
            1,
            EstimationCompleteness::Complete,
        )?,
    ))
}

fn lfm2_spec(args: &lfm2::ModelArgs) -> Result<Spec, CapabilityError> {
    let context = plain_context(args.max_position_embeddings)?;
    let attention = args
        .layer_schedule
        .iter()
        .filter(|policy| matches!(policy.operator, lfm2::OperatorPolicy::SelfAttention(_)))
        .count() as u64;
    let conv = args
        .layer_schedule
        .iter()
        .filter(|policy| matches!(policy.operator, lfm2::OperatorPolicy::CausalConvolution))
        .count() as u64;
    Ok((
        context.0,
        context.1,
        CacheStateStrategy::HybridRecurrent {
            full_attention_layers: attention,
            sliding_attention: Vec::new(),
            recurrent_layers: conv,
        },
        text_modalities(),
        state_memory_layout(
            lfm2::state_layout(args),
            args.hidden_size,
            256,
            EstimationCompleteness::Complete,
        )?,
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
        state_memory_layout(
            nemotron_h::state_layout(args),
            args.hidden_size,
            1,
            EstimationCompleteness::Complete,
        )?,
    ))
}

fn qwen_hybrid_spec(args: &QwenHybridConfig, multimodal: bool) -> Result<Spec, CapabilityError> {
    let configured = positive(args.max_position_embeddings, "max_position_embeddings")?;
    let original = args
        .rope_scaling
        .as_ref()
        .and_then(|config| config.get("original_max_position_embeddings"))
        .and_then(serde_json::Value::as_u64);
    let native = original.unwrap_or(configured);
    let effective = if original.is_some() {
        Observed::Available {
            value: configured,
            kind: ObservationKind::Exact,
            source: "validated Qwen RoPE configuration".into(),
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
                    Observed::Available {
                        value: scaled as u64,
                        kind: ObservationKind::Exact,
                        source: "validated Qwen configuration and supported RoPE setup".into(),
                    }
                } else {
                    Observed::Unsupported {
                        reason:
                            "Qwen RoPE factor does not produce an exact integer effective context"
                                .into(),
                    }
                }
            }
            None => Observed::Available {
                value: configured,
                kind: ObservationKind::Exact,
                source: "validated Qwen configuration".into(),
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
    let base = CacheStateStrategy::HybridRecurrent {
        full_attention_layers: attention,
        sliding_attention: Vec::new(),
        recurrent_layers: recurrent,
    };
    Ok((
        Observed::Available {
            value: native,
            kind: ObservationKind::Exact,
            source: "validated Qwen RoPE configuration".into(),
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
        state_memory_layout(
            crate::qwen::hybrid::state_layout(args),
            args.hidden_size,
            1,
            EstimationCompleteness::Complete,
        )?,
    ))
}

/// Complete portable capability and runtime-state estimate for one architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityEstimate {
    capabilities: ModelCapabilities,
    state_layout: StateMemoryLayout,
    draft_source: Option<SpeculativeDraftSource>,
}

impl CapabilityEstimate {
    /// Returns validated portable model capabilities.
    pub const fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }

    /// Returns memory metadata around the exact executable layer schedule.
    pub const fn state_layout(&self) -> &StateMemoryLayout {
        &self.state_layout
    }

    /// Architecture-declared checkpoint form for executable draft weights.
    ///
    /// `None` means that this exact normalized configuration exposes no
    /// speculative drafting graph. Concrete backends decide whether they
    /// implement the declared graph; they do not infer family policy themselves.
    pub const fn speculative_draft_source(&self) -> Option<SpeculativeDraftSource> {
        self.draft_source
    }

    /// Splits the estimate into its portable capability and state values.
    pub fn into_parts(self) -> (ModelCapabilities, StateMemoryLayout) {
        (self.capabilities, self.state_layout)
    }
}

fn finish(effective_model_type: String, spec: Spec) -> CapabilityEstimate {
    let (native_max_context, effective_max_context, state_strategy, modalities, state_layout) =
        spec;
    let estimation = if modalities.image || modalities.audio || modalities.video {
        EstimationCompleteness::Conservative
    } else {
        state_layout.completeness
    };
    CapabilityEstimate {
        capabilities: ModelCapabilities {
            effective_model_type,
            native_max_context,
            effective_max_context,
            state_strategy,
            modalities,
            estimation,
        },
        state_layout,
        draft_source: None,
    }
}

fn with_speculative_draft_source(
    mut estimate: CapabilityEstimate,
    draft_source: Option<SpeculativeDraftSource>,
) -> CapabilityEstimate {
    estimate.draft_source = draft_source;
    estimate
}

fn embedded_mtp_draft_source(layers: i32) -> Option<SpeculativeDraftSource> {
    (layers > 0).then_some(SpeculativeDraftSource::Embedded)
}

/// Derives Llama/Mistral capabilities from normalized architecture policy.
pub fn llama(args: &crate::llama::ModelArgs) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(finish(args.model_type.clone(), llama_spec(args, false)?))
}

/// Derives Qwen text capabilities from normalized architecture policy.
pub fn qwen(args: &crate::qwen::ModelArgs) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(finish(args.model_type.clone(), qwen_spec(args, false)?))
}

/// Derives Qwen3-VL capabilities from its complete normalized family policy.
pub fn qwen_vl(args: &crate::qwen::vl::ModelArgs) -> Result<CapabilityEstimate, CapabilityError> {
    let mut spec = qwen_spec(&args.text, true)?;
    spec.4 = state_memory_layout(
        crate::qwen::vl::state_layout(args),
        args.text.hidden_size,
        1,
        EstimationCompleteness::Complete,
    )?;
    Ok(finish(args.effective_model_type().into(), spec))
}

/// Derives Muse-Glimmer capabilities from normalized architecture policy.
pub fn muse_glimmer(
    args: &crate::muse_glimmer::DecoderConfig,
) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(with_speculative_draft_source(
        finish(args.model_type.clone(), muse_glimmer_spec(args)?),
        Some(SpeculativeDraftSource::Separate),
    ))
}

/// Derives DeepSeek-V3 capabilities from normalized architecture policy.
pub fn deepseek_v3(args: &crate::deepseek::V3Args) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(with_speculative_draft_source(
        finish(args.model_type.clone(), neutral_deepseek_v3_spec(args)?),
        embedded_mtp_draft_source(args.num_nextn_predict_layers),
    ))
}

/// Derives DeepSeek-V4 capabilities from normalized architecture policy.
pub fn deepseek_v4(args: &crate::deepseek::V4Args) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(with_speculative_draft_source(
        finish(args.model_type.clone(), neutral_deepseek_v4_spec(args)?),
        embedded_mtp_draft_source(args.num_nextn_predict_layers),
    ))
}

/// Derives Kimi Linear capabilities from normalized architecture policy.
pub fn kimi_linear(
    args: &crate::kimi_linear::ModelArgs,
) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(finish(args.model_type.clone(), kimi_linear_spec(args)?))
}

/// Derives GPT-OSS capabilities from normalized architecture policy.
pub fn gpt_oss(args: &crate::gpt_oss::ModelArgs) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(finish(args.model_type.clone(), gpt_oss_spec(args)?))
}

/// Derives Gemma 4 capabilities from its complete normalized family policy.
pub fn gemma4(args: &crate::gemma4::FamilyConfig) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(with_speculative_draft_source(
        finish(
            args.effective_model_type().into(),
            gemma4_spec(&args.text, args.input_modalities())?,
        ),
        Some(SpeculativeDraftSource::Separate),
    ))
}

/// Derives Inkling capabilities from normalized architecture policy.
pub fn inkling(args: &crate::inkling::ModelArgs) -> Result<CapabilityEstimate, CapabilityError> {
    let layers = args
        .mtp_config
        .as_ref()
        .map_or(0, |mtp| mtp.num_nextn_predict_layers);
    Ok(with_speculative_draft_source(
        finish(args.model_type.clone(), inkling_spec(args)?),
        embedded_mtp_draft_source(layers),
    ))
}

/// Derives LFM2 capabilities from normalized architecture policy.
pub fn lfm2(args: &crate::lfm2::ModelArgs) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(finish(args.model_type.clone(), lfm2_spec(args)?))
}

/// Derives Nemotron-H capabilities from normalized architecture policy.
pub fn nemotron_h(
    args: &crate::nemotron_h::ModelArgs,
) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(with_speculative_draft_source(
        finish(args.model_type.clone(), nemotron_spec(args)?),
        embedded_mtp_draft_source(args.num_nextn_predict_layers),
    ))
}

/// Derives Qwen hybrid text/vision capabilities from complete normalized policy.
pub fn qwen_hybrid(
    args: &crate::qwen::hybrid::ParsedHybridConfig,
) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(with_speculative_draft_source(
        finish(
            args.text.model_type.clone(),
            qwen_hybrid_spec(&args.text, args.vision.is_some())?,
        ),
        embedded_mtp_draft_source(args.text.mtp_num_hidden_layers),
    ))
}

/// Derives Qwen hybrid capabilities from normalized text-only policy.
pub fn qwen_hybrid_text(
    args: &crate::qwen::hybrid::HybridConfig,
) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(with_speculative_draft_source(
        finish(args.model_type.clone(), qwen_hybrid_spec(args, false)?),
        embedded_mtp_draft_source(args.mtp_num_hidden_layers),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::num::NonZeroU8;

    #[test]
    fn context_scaling_distinguishes_native_and_effective_limits() {
        let linear = std::collections::HashMap::from([("factor".into(), RopeValue::Float(4.0))]);
        let (native, effective) = context_from_rope(2_048, Some(&linear)).unwrap();
        assert_eq!(native.value(), Some(&2_048));
        assert_eq!(effective.value(), Some(&8_192));

        let yarn = std::collections::HashMap::from([
            ("factor".into(), RopeValue::Float(40.0)),
            (
                "original_max_position_embeddings".into(),
                RopeValue::Float(4_096.0),
            ),
        ]);
        let (native, effective) = context_from_rope(163_840, Some(&yarn)).unwrap();
        assert_eq!(native.value(), Some(&4_096));
        assert_eq!(effective.value(), Some(&163_840));
    }

    #[test]
    fn qwen_accounting_uses_the_executable_layer_schedule() {
        let args = crate::qwen::model_args_from_config_value(&json!({
            "model_type": "qwen2",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "use_sliding_window": true,
            "sliding_window": 8,
            "max_window_layers": 2
        }))
        .unwrap();

        let estimate = qwen(&args).unwrap();
        assert_eq!(
            estimate.capabilities().state_strategy,
            CacheStateStrategy::MixedKv {
                full_layers: 2,
                sliding: vec![SlidingWindowLayerCount {
                    layers: 2,
                    window: 8,
                }],
            }
        );
        let executable = crate::qwen::state_layout(&args).unwrap();
        assert_eq!(estimate.state_layout().layer_layout(), executable.layers());
    }

    #[test]
    fn kimi_linear_mixed_dtype_state_uses_per_tensor_widths() {
        let args = crate::kimi_linear::model_args_from_config_value(&json!({
            "model_type": "kimi_linear",
            "vocab_size": 64,
            "hidden_size": 8,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 16,
            "head_dim": 4,
            "model_max_length": 128,
            "linear_attn_config": {
                "kda_layers": [1, 3],
                "full_attn_layers": [2, 4],
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
        let estimate = kimi_linear(&args).unwrap();
        let executable = crate::kimi_linear::state_layout(&args).unwrap();
        assert_eq!(estimate.state_layout().layer_layout(), executable.layers());

        let state = eredu_core::estimate_runtime_state(
            estimate.state_layout(),
            eredu_core::InputTokenCount::text(1),
            0,
            1,
            NonZeroU8::new(2).unwrap(),
        )
        .unwrap();

        // Two KDA layers each retain 24 BF16 convolution scalars and 32 FP32
        // recurrent scalars: 2 * (24 * 2 + 32 * 4) = 352 bytes.
        assert_eq!(state.fixed_state_bytes, 352);
    }

    #[test]
    fn embedded_mtp_accounting_contains_every_executable_state_layer() {
        let v3_args = crate::deepseek::parse_v3_config(&json!({
            "hidden_size": 8, "intermediate_size": 16, "moe_intermediate_size": 8,
            "num_hidden_layers": 2, "num_attention_heads": 2, "vocab_size": 31,
            "max_position_embeddings": 64, "kv_lora_rank": 4, "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2, "v_head_dim": 2, "first_k_dense_replace": 1,
            "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
            "n_group": 2, "topk_group": 1, "num_nextn_predict_layers": 1,
            "tie_word_embeddings": false
        }))
        .unwrap();
        let v3 = deepseek_v3(&v3_args).unwrap();
        let v3_executable = crate::deepseek::v3::state_layout(&v3_args).unwrap();
        assert_eq!(v3.state_layout().layer_layout(), v3_executable.layers());
        assert_eq!(v3.state_layout().layer_layout().len(), 3);
        let v3_state = eredu_core::estimate_runtime_state(
            v3.state_layout(),
            eredu_core::InputTokenCount::text(3),
            0,
            1,
            NonZeroU8::new(2).unwrap(),
        )
        .unwrap();
        assert_eq!(v3_state.context_state_bytes, 3 * (4 + 2) * 256 * 2);
        let target_only_budget = 2 * (4 + 2) * 256 * 2;
        assert!(matches!(
            eredu_core::apply_admission_policy(
                v3.capabilities(),
                eredu_core::AdmissionRequest {
                    input: eredu_core::InputTokenCount::text(3),
                    max_output_tokens: 0,
                    batch_size: 1,
                    safety_reserve_bytes: 0,
                    application_memory_budget_bytes: Some(target_only_budget),
                    require_complete_estimate: true,
                },
                v3_state,
                None,
            )
            .unwrap(),
            eredu_core::AdmissionResult::Rejected(
                eredu_core::AdmissionRejection::MemoryBudgetExceeded { .. }
            )
        ));

        let v4_args = crate::deepseek::parse_v4_config(&json!({
            "hidden_size": 8, "moe_intermediate_size": 8, "num_hidden_layers": 3,
            "num_attention_heads": 2, "head_dim": 4, "qk_rope_head_dim": 2,
            "q_lora_rank": 4, "o_lora_rank": 2, "o_groups": 2, "vocab_size": 31,
            "max_position_embeddings": 64, "sliding_window": 8,
            "compress_ratios": [0, 4, 128, 0], "index_n_heads": 2,
            "index_head_dim": 4, "index_topk": 1, "hc_mult": 2,
            "hc_sinkhorn_iters": 2, "n_routed_experts": 4, "num_experts_per_tok": 2,
            "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
            "norm_topk_prob": true, "num_nextn_predict_layers": 1
        }))
        .unwrap();
        let v4 = deepseek_v4(&v4_args).unwrap();
        assert_eq!(
            v4.capabilities().state_strategy,
            CacheStateStrategy::SlidingKey {
                window: 8,
                layers: 4,
                pooling_layers: 2,
            }
        );
        assert_eq!(
            serde_json::to_value(v4.capabilities()).unwrap()["state_strategy"],
            json!({
                "strategy": "sliding_key",
                "window": 8,
                "layers": 4,
                "pooling_layers": 2
            })
        );
        assert_eq!(
            v4.state_layout().layer_layout(),
            crate::deepseek::v4::state_layout(&v4_args)
                .unwrap()
                .layers()
        );
        assert_eq!(v4.state_layout().layer_layout().len(), 4);

        let inkling_args = crate::inkling::ModelArgs::from_hf_json(
            br#"{
              "model_type":"inkling_mm_model",
              "text_config":{
                "hidden_size":16,"num_hidden_layers":1,"vocab_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
                "sliding_window_size":8,"local_layer_ids":[0],
                "mlp_layer_types":["dense"],"sconv_kernel_size":4,
                "d_rel":2,"intermediate_size":12,"n_routed_experts":4,
                "num_experts_per_tok":2,"n_shared_experts":1
              },
              "mtp_config":{
                "num_nextn_predict_layers":2,"local_layer_ids":[1],
                "chain_hidden_post_norm":true,"dense_intermediate_size":12
              }
            }"#,
        )
        .unwrap();
        let inkling = inkling(&inkling_args).unwrap();
        assert_eq!(inkling.state_layout().layer_layout().len(), 3);

        let nemotron_args = crate::nemotron_h::model_args_from_config_value(&json!({
            "model_type":"nemotron_h", "vocab_size":32, "hidden_size":16,
            "intermediate_size":24, "num_hidden_layers":4,
            "hybrid_override_pattern":"M*-E", "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
            "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
            "conv_kernel":3, "n_routed_experts":4, "n_shared_experts":1,
            "moe_intermediate_size":8, "moe_shared_expert_intermediate_size":8,
            "num_experts_per_tok":2, "n_group":2, "topk_group":1,
            "num_nextn_predict_layers":1, "mtp_hybrid_override_pattern":"*E",
            "tie_word_embeddings":false
        }))
        .unwrap();
        let nemotron = nemotron_h(&nemotron_args).unwrap();
        assert!(nemotron.state_layout().layer_layout().len() > 4);
        assert_eq!(
            nemotron.state_layout().layer_layout(),
            crate::nemotron_h::state_layout(&nemotron_args)
                .unwrap()
                .layers()
        );

        let qwen_args = crate::qwen::hybrid::model_args_from_config_value(&json!({
            "model_type": "qwen3_5_text", "vocab_size": 8, "hidden_size": 8,
            "num_hidden_layers": 2, "mtp_num_hidden_layers": 2,
            "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 8,
            "max_position_embeddings": 16, "intermediate_size": 16,
            "num_experts": 0, "tie_word_embeddings": true,
            "layer_types": ["full_attention", "full_attention"]
        }))
        .unwrap();
        let qwen = qwen_hybrid(&qwen_args).unwrap();
        assert_eq!(qwen.state_layout().layer_layout().len(), 4);
        assert_eq!(
            qwen.state_layout().layer_layout(),
            crate::qwen::hybrid::state_layout(&qwen_args.text)
                .unwrap()
                .layers()
        );
    }
}
