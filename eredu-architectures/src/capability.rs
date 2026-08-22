//! Backend-neutral model capability and runtime-state estimates.
//!
//! Normalized architecture configurations own context limits, accepted
//! modalities, cache/recurrent strategy, and scalar state geometry. Concrete
//! backends apply physical scalar widths and report live memory observations.

use std::collections::BTreeMap;

use eredu_core::{
    CacheStateStrategy, CapabilityError, EstimationCompleteness, GrowingState, InputModalities,
    ModelCapabilities, ObservationKind, Observed, SlidingWindowLayerCount, StateLayout,
};
use eredu_nn::RopeValue;

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

fn positive(value: i32, field: &'static str) -> Result<u64, CapabilityError> {
    u64::try_from(value).map_err(|_| CapabilityError::InvalidConfiguration {
        field,
        detail: format!("expected a non-negative value, got {value}"),
    })
}

fn checked_add(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_add(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn checked_mul(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_mul(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

type Spec = (
    Observed<u64>,
    Observed<u64>,
    CacheStateStrategy,
    InputModalities,
    StateLayout,
);

fn llama_spec(args: &LlamaModelArgs, multimodal: bool) -> Result<Spec, CapabilityError> {
    let context = context_from_rope(args.max_position_embeddings, args.rope_scaling.as_ref())?;
    let layers = positive(args.num_hidden_layers, "num_hidden_layers")?;
    let scalars = kv_scalars(args.num_key_value_heads, args.head_dim)?;
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
        StateLayout {
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
            completeness,
        },
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
    let scalars = kv_scalars(args.num_key_value_heads, args.head_dim)?;
    let state_strategy = CacheStateStrategy::MixedKv {
        full_layers,
        sliding: sliding.clone(),
    };
    let growing = std::iter::once(GrowingState {
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
        StateLayout {
            fixed_scalars_per_batch: 0,
            growing,
            hidden_size: positive(args.hidden_size, "hidden_size")?,
            allocation_granularity: 1,
            completeness: EstimationCompleteness::Complete,
        },
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
    let layers = u64::try_from(args.layer_schedule.len()).map_err(|_| {
        CapabilityError::InvalidConfiguration {
            field: "layer_schedule",
            detail: "decoder layer count exceeds runtime-state accounting range".into(),
        }
    })?;
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
        StateLayout {
            fixed_scalars_per_batch: 0,
            growing: vec![GrowingState {
                layers,
                scalars_per_position: checked_add(latent, rotary, "MLA latent plus rotary width")?,
                window: None,
            }],
            hidden_size: positive(args.hidden_size, "hidden_size")?,
            allocation_granularity: 256,
            completeness: EstimationCompleteness::Complete,
        },
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
    let layers = positive(args.num_hidden_layers, "num_hidden_layers")?;
    let compressed = args
        .attention_schedule
        .iter()
        .filter(|policy| {
            matches!(
                policy,
                crate::deepseek::V4AttentionPolicy::Compressed { .. }
            )
        })
        .count() as u64;
    let head_dim = positive(args.head_dim, "head_dim")?;
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
        CacheStateStrategy::MixedKv {
            full_layers: compressed,
            sliding: vec![SlidingWindowLayerCount {
                layers,
                window: positive(args.sliding_window, "sliding_window")?,
            }],
        },
        text_modalities(),
        StateLayout {
            fixed_scalars_per_batch: 0,
            growing: vec![
                GrowingState {
                    layers,
                    scalars_per_position: head_dim,
                    window: Some(positive(args.sliding_window, "sliding_window")?),
                },
                GrowingState {
                    layers: compressed,
                    scalars_per_position: head_dim,
                    window: None,
                },
            ],
            hidden_size: positive(args.hidden_size, "hidden_size")?,
            allocation_granularity: 128,
            completeness: EstimationCompleteness::Conservative,
        },
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
    let heads = positive(args.kda_config.num_heads, "kda_config.num_heads")?;
    let head_dim = positive(args.kda_config.head_dim, "kda_config.head_dim")?;
    let projection = checked_mul(heads, head_dim, "KDA projected width")?;
    let conv_state = checked_mul(
        checked_mul(
            positive(
                args.kda_config.short_conv_kernel_size - 1,
                "kda_config.short_conv_kernel_size",
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
        StateLayout {
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
        StateLayout {
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
        StateLayout {
            fixed_scalars_per_batch: 0,
            growing: {
                let mut groups = BTreeMap::<u64, u64>::new();
                for policy in args
                    .layer_schedule
                    .iter()
                    .filter(|policy| policy.key_value.owns_state())
                {
                    let scalars = 2
                        * u64::from(policy.num_key_value_heads.get())
                        * u64::from(policy.head_dim.get());
                    *groups.entry(scalars).or_default() += 1;
                }
                groups
                    .into_iter()
                    .map(|(scalars_per_position, layers)| GrowingState {
                        layers,
                        scalars_per_position,
                        // Gemma masks sliding attention but retains full KV backing.
                        window: None,
                    })
                    .collect()
            },
            hidden_size: positive(args.hidden_size, "hidden_size")?,
            allocation_granularity: 256,
            completeness: EstimationCompleteness::Complete,
        },
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
        StateLayout {
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
        .filter(|policy| matches!(policy.operator, lfm2::OperatorPolicy::SelfAttention(_)))
        .count() as u64;
    let conv = args
        .layer_schedule
        .iter()
        .filter(|policy| matches!(policy.operator, lfm2::OperatorPolicy::CausalConvolution))
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
        StateLayout {
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
        StateLayout {
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
        StateLayout {
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

/// Complete portable capability and scalar-state estimate for one architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityEstimate {
    capabilities: ModelCapabilities,
    state_layout: StateLayout,
}

impl CapabilityEstimate {
    /// Returns validated portable model capabilities.
    pub const fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }

    /// Returns backend-neutral scalar state geometry.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Splits the estimate into its portable capability and state values.
    pub fn into_parts(self) -> (ModelCapabilities, StateLayout) {
        (self.capabilities, self.state_layout)
    }
}

fn finish(model_type: String, spec: Spec) -> CapabilityEstimate {
    let (native_max_context, effective_max_context, state_strategy, modalities, state_layout) =
        spec;
    let estimation = if modalities.image || modalities.audio || modalities.video {
        EstimationCompleteness::Conservative
    } else {
        state_layout.completeness
    };
    CapabilityEstimate {
        capabilities: ModelCapabilities {
            model_type,
            native_max_context,
            effective_max_context,
            state_strategy,
            modalities,
            estimation,
        },
        state_layout,
    }
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
    Ok(finish(
        args.model_type.clone(),
        qwen_spec(&args.text, true)?,
    ))
}

/// Derives Muse-Glimmer capabilities from normalized architecture policy.
pub fn muse_glimmer(
    args: &crate::muse_glimmer::DecoderConfig,
) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(finish(args.model_type.clone(), muse_glimmer_spec(args)?))
}

/// Derives DeepSeek-V3 capabilities from normalized architecture policy.
pub fn deepseek_v3(args: &crate::deepseek::V3Args) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(finish(
        args.model_type.clone(),
        neutral_deepseek_v3_spec(args)?,
    ))
}

/// Derives DeepSeek-V4 capabilities from normalized architecture policy.
pub fn deepseek_v4(args: &crate::deepseek::V4Args) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(finish(
        args.model_type.clone(),
        neutral_deepseek_v4_spec(args)?,
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
    let modalities = InputModalities {
        text: true,
        image: args.image_token_id.is_some(),
        audio: args.audio_token_id.is_some(),
        video: args.video_token_id.is_some(),
    };
    Ok(finish(
        args.model_type.clone(),
        gemma4_spec(&args.text, modalities)?,
    ))
}

/// Derives Inkling capabilities from normalized architecture policy.
pub fn inkling(args: &crate::inkling::ModelArgs) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(finish(args.model_type.clone(), inkling_spec(args)?))
}

/// Derives LFM2 capabilities from normalized architecture policy.
pub fn lfm2(args: &crate::lfm2::ModelArgs) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(finish(args.model_type.clone(), lfm2_spec(args)?))
}

/// Derives Nemotron-H capabilities from normalized architecture policy.
pub fn nemotron_h(
    args: &crate::nemotron_h::ModelArgs,
) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(finish(args.model_type.clone(), nemotron_spec(args)?))
}

/// Derives Qwen hybrid text/vision capabilities from complete normalized policy.
pub fn qwen_hybrid(
    args: &crate::qwen::hybrid::ParsedHybridConfig,
) -> Result<CapabilityEstimate, CapabilityError> {
    Ok(finish(
        args.text.model_type.clone(),
        qwen_hybrid_spec(&args.text, args.vision.is_some())?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
    fn qwen_catalog_owns_window_grouping_and_scalar_geometry() {
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
        assert_eq!(
            estimate.state_layout().growing,
            vec![
                GrowingState {
                    layers: 2,
                    scalars_per_position: 16,
                    window: None,
                },
                GrowingState {
                    layers: 2,
                    scalars_per_position: 16,
                    window: Some(8),
                },
            ]
        );
    }
}
