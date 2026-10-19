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
                ));
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

fn muse_spec(args: &crate::muse::DecoderConfig) -> Result<Spec, CapabilityError> {
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
            image: args.vision_config.is_some(),
            audio: args.audio_config.is_some(),
            video: args.vision_config.as_ref().is_some_and(|vision|
                vision.input_policy == crate::muse::VisionInputPolicy::ImagesAndVideo),
        },
        state_memory_layout(
            crate::muse::state_layout(args),
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
            LayerCachePolicy::KeyOnlyWithState {
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
                });
            }
        };
        let layer_window = match attention {
            AttentionPolicy::Sliding { window } => u64::from(window.get()),
            AttentionPolicy::Full => {
                return Err(CapabilityError::InvalidConfiguration {
                    field: "state_layout",
                    detail: format!("DeepSeek-V4 layer {layer} has unbounded attention state"),
                });
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
    missing_media_companions: Vec<(eredu_core::InputModality, &'static str)>,
    draft_source: Option<SpeculativeDraftSource>,
    shared_backbone: Option<eredu_runtime::speculative::shared_backbone::SharedBackboneExecutionPolicy>,
    ordinary_refinement: Option<eredu_runtime::generation::refinement::architecture::RefinementExecutionPolicy>,
}

impl CapabilityEstimate {
    /// Attaches an admitted shared-backbone draft policy to the family declaration.
    pub(crate) fn with_shared_backbone_draft(mut self, policy: eredu_runtime::speculative::shared_backbone::SharedBackboneExecutionPolicy) -> Self {
        self.ordinary_refinement = Some(eredu_runtime::generation::refinement::architecture::RefinementExecutionPolicy {
            geometry: policy.geometry(), policy: policy.diffusion_policy(0.0).expect("validated family policy"),
            causal_sampling: true, seed_first: policy.seed_first(),
        });
        self.shared_backbone = Some(policy);
        self.draft_source = Some(SpeculativeDraftSource::SharedBackbone);
        self
    }
    /// Source-free masked draft defaults consumed by native construction adapters.
    pub const fn shared_backbone_draft(&self) -> Option<eredu_runtime::speculative::shared_backbone::SharedBackboneExecutionPolicy> { self.shared_backbone }
    /// Retained ordinary numerical equations, independent of any draft source.
    pub const fn ordinary_refinement(&self) -> Option<eredu_runtime::generation::refinement::architecture::RefinementExecutionPolicy> { self.ordinary_refinement }
    /// Complete target-stack observation semantics for a retained shared draft.
    /// All modes reuse ordinary target hook identities and producer ownership.
    pub fn shared_backbone_activation_execution(&self) -> Option<crate::speculative_execution::SpeculativeActivationExecution> {
        self.shared_backbone.map(|_| crate::speculative_execution::SpeculativeActivationExecution::shared_backbone())
    }
    /// Projects memory accounting onto the exact selected rank-local state.
    /// Family capabilities and context limits remain global model properties.
    pub(crate) fn for_selected_state(
        mut self,
        layout: Option<&RuntimeStateLayout>,
    ) -> Result<Self, CapabilityError> {
        self.state_layout = StateMemoryLayout::new(
            layout.map_or_else(eredu_core::LayerSchedule::empty, |layout| {
                layout.layers().clone()
            }),
            layout.map_or_else(Vec::new, |layout| layout.layer_prefix_offsets()),
            self.state_layout.hidden_size,
            self.state_layout.allocation_granularity,
            self.state_layout.completeness,
        )?;
        Ok(self)
    }

    /// Reuses an exact mechanism/state estimate for an architecture extension
    /// that implements the same validated neutral geometry under a new identity.
    pub fn for_architecture_extension(mut self, effective_model_type: impl Into<String>) -> Self {
        self.capabilities.effective_model_type = effective_model_type.into();
        self
    }

    /// Returns validated portable model capabilities.
    pub const fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }

    /// Admits decoded media against selected component coverage before processor lookup.
    /// Token-only segments require no media components or processor assets.
    pub fn admit_raw_media_request(
        &self,
        request: &eredu_core::TokenizedMultimodalRequest,
        processor_available: bool,
    ) -> Result<(), CapabilityError> {
        use eredu_core::{InputModality, Media, TokenizedMultimodalSegment};
        for segment in request.segments() {
            let modality = match segment {
                TokenizedMultimodalSegment::TokenIds(_) => continue,
                TokenizedMultimodalSegment::Media(Media::Image(_)) => InputModality::Image,
                TokenizedMultimodalSegment::Media(Media::Video(_)) => InputModality::Video,
                TokenizedMultimodalSegment::Media(Media::Audio(_)) => InputModality::Audio,
            };
            if let Some((_, component)) = self
                .missing_media_companions
                .iter()
                .find(|(missing, _)| *missing == modality)
            {
                return Err(CapabilityError::MissingCompanion {
                    architecture: self.capabilities.effective_model_type.clone(),
                    modality,
                    component,
                });
            }
            let supported = match modality {
                InputModality::Text => self.capabilities.modalities.text,
                InputModality::Image => self.capabilities.modalities.image,
                InputModality::Video => self.capabilities.modalities.video,
                InputModality::Audio => self.capabilities.modalities.audio,
                _ => false,
            };
            if !supported {
                return Err(CapabilityError::UnsupportedInput {
                    architecture: self.capabilities.effective_model_type.clone(),
                    reason: format!(
                        "selected architecture does not accept {} input",
                        modality.as_str()
                    ),
                });
            }
            if !processor_available {
                return Err(CapabilityError::MissingCompanion {
                    architecture: self.capabilities.effective_model_type.clone(),
                    modality,
                    component: "processor assets",
                });
            }
        }
        Ok(())
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
        missing_media_companions: Vec::new(),
        draft_source: None,
        shared_backbone: None,
        ordinary_refinement: None,
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

/// Derives one shared causal cache and the retained masked-draft declaration.
/// Image input is advertised only for construction with admitted vision sources.
pub fn nemotron_labs_diffusion(args: &crate::nemotron_labs_diffusion::FamilyConfig, image_input: bool) -> Result<CapabilityEstimate, CapabilityError> {
    use crate::decoder::Config;
    if image_input && args.vision.is_none() {
        return Err(CapabilityError::InvalidConfiguration { field: "vision", detail: "image input requires an admitted Pixtral tower".into() });
    }
    let estimate = finish(args.model_identity().to_owned(), llama_spec(args.dense(), image_input)?);
    if args.block_size.get() < 2 { return Ok(estimate); }
    let policy = args.shared_backbone_draft_policy().map_err(|error| CapabilityError::InvalidConfiguration { field: "shared_backbone", detail: error.to_string() })?;
    Ok(estimate.with_shared_backbone_draft(policy))
}

/// Derives state memory for every logical decoder invocation.
pub fn nanbeige(args: &crate::nanbeige::ModelArgs) -> Result<CapabilityEstimate, CapabilityError> {
    let dense = args.dense_config();
    let mut spec = llama_spec(dense, false)?;
    spec.4 = state_memory_layout(
        crate::nanbeige::state_layout(args),
        dense.hidden_size,
        1,
        EstimationCompleteness::Complete,
    )?;
    Ok(finish("nanbeige".into(), spec))
}

/// Derives context and ordinary KV geometry from the normalized K2 schedule.
pub fn k2_horizon(
    args: &crate::k2_horizon::ModelArgs,
) -> Result<CapabilityEstimate, CapabilityError> {
    let context = context_from_rope(
        args.max_position_embeddings,
        args.rope_parameters.as_ref().or(args.rope_scaling.as_ref()),
    )?;
    let sliding = args
        .schedule
        .sliding_windows()
        .into_iter()
        .map(|(window, layers)| SlidingWindowLayerCount {
            window: u64::from(window.get()),
            layers: layers as u64,
        })
        .collect::<Vec<_>>();
    let full_layers = args.schedule.full_layer_count() as u64;
    let strategy = match (full_layers, sliding.as_slice()) {
        (_, []) => CacheStateStrategy::FullKv,
        (0, [only]) => CacheStateStrategy::SlidingKv {
            window: only.window,
        },
        _ => CacheStateStrategy::MixedKv {
            full_layers,
            sliding,
        },
    };
    let state = state_memory_layout(
        crate::decoder::state_layout(args),
        args.hidden_size,
        1,
        EstimationCompleteness::Complete,
    )?;
    Ok(with_speculative_draft_source(
        finish(
            "k2_horizon".into(),
            (context.0, context.1, strategy, text_modalities(), state),
        ),
        Some(SpeculativeDraftSource::Separate),
    ))
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

/// Derives Muse capabilities from selected component coverage and normalized policy.
pub fn muse(args: &crate::muse::DecoderConfig) -> Result<CapabilityEstimate, CapabilityError> {
    let mut estimate = with_speculative_draft_source(
        finish(args.identity.text_model_type().into(), muse_spec(args)?),
        Some(SpeculativeDraftSource::Separate),
    );
    estimate.missing_media_companions = muse_missing_media_companions(args);
    Ok(estimate)
}

/// Source omissions that can be satisfied by independently admitted Muse companions.
pub(crate) fn muse_missing_media_companions(
    args: &crate::muse::DecoderConfig,
) -> Vec<(eredu_core::InputModality, &'static str)> {
    let mut missing = Vec::new();
    if args.vision_config.is_none() {
        for modality in [
            eredu_core::InputModality::Image,
            eredu_core::InputModality::Video,
        ] {
            missing.push((modality, "vision encoder and projector"));
        }
    }
    if args.identity == crate::muse::MuseIdentity::Spark && args.audio_config.is_none() {
        missing.push((
            eredu_core::InputModality::Audio,
            "waveform encoder and perception projection",
        ));
    }
    missing
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

/// Derives shared-context state and input geometry for block refinement.
/// Causal speculative verification requires conditional target probabilities;
/// refinement position scores do not provide that distribution.
pub fn diffusion_gemma(
    args: &crate::diffusion_gemma::FamilyConfig,
) -> Result<CapabilityEstimate, CapabilityError> {
    let mut spec = gemma4_spec(&args.text, args.input_modalities())?;
    spec.4 = state_memory_layout(
        crate::diffusion_gemma::state_layout(args),
        args.text.hidden_size,
        256,
        EstimationCompleteness::Complete,
    )?;
    let mut estimate = finish("diffusion_gemma_text".into(), spec);
    estimate.ordinary_refinement = Some(eredu_runtime::generation::refinement::architecture::RefinementExecutionPolicy {
        geometry: eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry {
            canvas_length: args.canvas_length, vocabulary_size: std::num::NonZeroU32::new(u32::try_from(args.text.vocab_size).map_err(|_| CapabilityError::InvalidConfiguration { field: "vocab_size", detail: "outside refinement domain".into() })?).ok_or_else(|| CapabilityError::InvalidConfiguration { field: "vocab_size", detail: "empty refinement domain".into() })?,
        },
        policy: eredu_core::refinement::RefinementPolicy::default(), causal_sampling: false, seed_first: false,
    });
    Ok(estimate)
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

/// Describes the prepared Flash-Next target role using its executable state layout.
/// Media processors and prediction state require their own prepared roles. Native
/// allocation overhead and execution workspaces are not persistent-state payload.
pub fn qwen4_exp_target(
    spec: &crate::qwen4_exp::target::TargetSpec,
) -> Result<CapabilityEstimate, CapabilityError> {
    let config = &spec.config;
    let configured = positive(config.max_positions, "max_position_embeddings")?;
    let effective = configured.min(positive(spec.limits.history_tokens, "history_tokens")?);
    let native = match config.attention.rotary.algorithm {
        eredu_nn::RotaryAlgorithm::Yarn {
            original_max_positions,
            ..
        }
        | eredu_nn::RotaryAlgorithm::Llama3 {
            original_max_positions,
            ..
        } => positive(original_max_positions, "original_max_position_embeddings")?,
        _ => configured,
    };
    let context = |value| Observed::Available {
        value,
        kind: ObservationKind::Exact,
        source: "validated qwen4_exp configuration and retained execution limits".into(),
    };
    let attention = config
        .layers
        .iter()
        .filter(|kind| matches!(kind, crate::qwen4_exp::config::LayerKind::Indexed))
        .count() as u64;
    Ok(finish(
        "qwen4_exp_text".into(),
        (
            context(native),
            context(effective),
            CacheStateStrategy::HybridRecurrent {
                full_attention_layers: attention,
                sliding_attention: Vec::new(),
                recurrent_layers: config.layers.len() as u64 - attention,
            },
            text_modalities(),
            state_memory_layout(
                spec.state_layout(),
                config.hidden_size,
                1,
                EstimationCompleteness::PersistentStateOnly,
            )?,
        ),
    ))
}

/// Joint target and embedded predictor use separately owned persistent state.
/// Reuses the ordinary state estimate without treating predictor caches as target aliases.
pub(crate) fn qwen4_exp_prediction(
    target: &crate::qwen4_exp::target::TargetSpec,
    prediction: &crate::qwen4_exp::mtp::PredictionSpec,
) -> Result<CapabilityEstimate, CapabilityError> {
    let mut estimate = qwen4_exp_target(target)?;
    let prediction_state =
        prediction
            .state_layout()
            .map_err(|error| CapabilityError::InvalidConfiguration {
                field: "prediction_state_layout",
                detail: error.to_string(),
            })?;
    let layers = estimate
        .state_layout
        .layer_layout()
        .iter()
        .cloned()
        .chain(prediction_state.layers().iter().cloned())
        .collect::<Vec<_>>();
    let offsets = estimate
        .state_layout
        .layer_prefix_offsets()
        .iter()
        .copied()
        .chain(prediction_state.layer_prefix_offsets())
        .collect();
    estimate.state_layout = StateMemoryLayout::new(
        eredu_core::LayerSchedule::new(layers.len(), layers).map_err(|error| {
            CapabilityError::InvalidConfiguration {
                field: "prediction_state_layout",
                detail: error.to_string(),
            }
        })?,
        offsets,
        estimate.state_layout.hidden_size,
        estimate.state_layout.allocation_granularity,
        EstimationCompleteness::PersistentStateOnly,
    )?;
    if let CacheStateStrategy::HybridRecurrent {
        full_attention_layers,
        recurrent_layers,
        ..
    } = &mut estimate.capabilities.state_strategy
    {
        for unit in &prediction.units {
            match &unit.mixer {
                crate::qwen4_exp::target::MixerSpec::Indexed(_) => *full_attention_layers += 1,
                crate::qwen4_exp::target::MixerSpec::Recurrent(_) => *recurrent_layers += 1,
            }
        }
    }
    estimate.draft_source = Some(SpeculativeDraftSource::Embedded);
    Ok(estimate)
}

/// Prepared joint image/video and target equations. Host processor dispatch and
/// complete request workspace admission remain distinct from persistent state sizing.
pub(crate) fn qwen4_exp_conditional(
    spec: &crate::qwen4_exp::target::TargetSpec,
    prediction: Option<&crate::qwen4_exp::mtp::PredictionSpec>,
) -> Result<CapabilityEstimate, CapabilityError> {
    let mut estimate = match prediction {
        Some(prediction) => qwen4_exp_prediction(spec, prediction)?,
        None => qwen4_exp_target(spec)?,
    };
    estimate.capabilities.effective_model_type = "qwen4_exp".into();
    estimate.capabilities.modalities.image = true;
    estimate.capabilities.modalities.video = true;
    Ok(estimate)
}

/// Derives Gemma 2 context and mixed full/sliding KV residency from its schedule.
pub fn gemma2(args: &crate::gemma2::ModelArgs) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(finish(
        "gemma2".into(),
        llama_spec(args.dense_config(), false)?,
    ))
}
