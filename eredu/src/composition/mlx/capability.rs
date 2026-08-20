//! MLX model capability derivation, resource observation, and admission adapter.

use std::{collections::BTreeMap, num::NonZeroU8};

use eredu_architectures::{
    llama::ModelArgs as LlamaModelArgs,
    qwen::{ModelArgs as QwenModelArgs, QwenVariant},
};
use eredu_core::{
    estimate_runtime_state, AvailableMemory, CacheStateStrategy, CapabilityError,
    EstimationCompleteness, GrowingState, InputModalities, InputTokenCount, ModelCapabilities,
    ModelCapabilityBackend, ModelRuntime, ObservationKind, Observed, PhysicalMemorySemantics,
    RuntimeStateEstimate, SlidingWindowLayerCount, StateLayout, StaticMemoryReport,
};
use eredu_nn::RopeValue;
use safemlx::{Array, Stream};

use super::{MlxBackend, MlxModelInput, MlxModelSession, Model};
use crate::{
    backend::mlx::runtime::media::input::{self, InputPayload, Modality},
    composition::mlx_architectures::{
        gemma4::model as gemma4,
        gpt_oss::model as gpt_oss,
        inkling::model as inkling,
        kimi_linear::model as kimi_linear,
        lfm2::model as lfm2,
        muse_glimmer,
        nemotron_h::model as nemotron_h,
        qwen::hybrid::qwen3_5::{self, LayerPolicy as QwenHybridLayerPolicy},
    },
    core::attention::AttentionPolicy,
    core::residency::MemoryTier,
};

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

fn estimate_mlx_runtime_state(
    layout: &StateLayout,
    input: InputTokenCount,
    max_output_tokens: u64,
    batch_size: u64,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    estimate_runtime_state(
        layout,
        input,
        max_output_tokens,
        batch_size,
        NonZeroU8::new(4).expect("MLX cache scalar width is nonzero"),
    )
}

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

impl Model {
    fn capabilities_and_estimate(
        &self,
    ) -> Result<(ModelCapabilities, StateLayout), CapabilityError> {
        let model_type = self.model_type().to_string();
        let result = match self {
            Self::DeepSeek(model) => {
                if let Some(args) = model.v3_args() {
                    neutral_deepseek_v3_spec(args)?
                } else {
                    neutral_deepseek_v4_spec(model.v4_args().expect("DeepSeek family"))?
                }
            }
            Self::Llama(model) => llama_spec(model.args(), false)?,
            Self::Qwen(model) => qwen_spec(model.args(), false)?,
            Self::MuseGlimmer(model) => muse_glimmer_spec(model.args())?,
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => {
                qwen_spec(&model.args().text_config, true)?
            }
            Self::GptOss(model) => gpt_oss_spec(model.args())?,
            Self::Gemma4(model) => {
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
            Self::Inkling(model) => inkling_spec(model.args())?,
            Self::KimiLinear(model) => kimi_linear_spec(model.args())?,
            Self::Lfm2(model) => lfm2_spec(model.args())?,
            Self::NemotronH(model) => nemotron_spec(model.args())?,
            Self::Qwen3Next(model) | Self::Qwen35(model) => {
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

fn muse_glimmer_spec(args: &muse_glimmer::DecoderConfig) -> Result<Spec, CapabilityError> {
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
            image: args.vision_config.is_some(),
            audio: false,
            video: args.vision_config.is_some()
                && args.weight_convention == muse_glimmer::WeightConvention::HuggingFace,
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

fn neutral_deepseek_v3_spec(
    args: &eredu_architectures::deepseek::V3Args,
) -> Result<Spec, CapabilityError> {
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

fn neutral_deepseek_v4_spec(
    args: &eredu_architectures::deepseek::V4Args,
) -> Result<Spec, CapabilityError> {
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
                eredu_architectures::deepseek::V4AttentionPolicy::Compressed { .. }
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
    let layers = positive(args.num_hidden_layers, "num_hidden_layers")?;
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

fn inkling_spec(args: &inkling::ModelArgs) -> Result<Spec, CapabilityError> {
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

fn qwen_hybrid_spec(args: &qwen3_5::ModelArgs, multimodal: bool) -> Result<Spec, CapabilityError> {
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

fn unavailable_counter(error: safemlx::error::Exception) -> Observed<u64> {
    Observed::Unavailable {
        reason: error.to_string(),
    }
}

fn runtime_counter(
    function: fn() -> Result<usize, safemlx::error::Exception>,
    source: &'static str,
) -> Observed<u64> {
    match function() {
        Ok(value) => match u64::try_from(value) {
            Ok(value) => Observed::Available {
                value,
                kind: ObservationKind::Observational,
                source: source.into(),
            },
            Err(_) => Observed::Unavailable {
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
    config: &crate::composition::mlx_architectures::qwen::vl::vision::VisionConfig,
    modality: Modality,
    payload: &Array,
    metadata: input::InputMetadata<'_>,
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
            crate::composition::mlx_architectures::qwen::vl::vision::grid_thw_from_array(
                grid, stream,
            )
            .map_err(|error| CapabilityError::UnsupportedInput {
                architecture: architecture.into(),
                reason: error.to_string(),
            })
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
    let depth =
        u64::try_from(config.layer_count()).map_err(|_| CapabilityError::ArithmeticOverflow {
            operation: "Qwen vision depth",
        })?;
    let full_blocks = u64::try_from(
        config
            .layer_schedule
            .iter()
            .filter(|policy| {
                matches!(
                    policy.attention,
                    crate::composition::mlx_architectures::qwen::vl::vision::VisionAttentionPolicy::Full
                )
            })
            .count(),
    )
    .map_err(|_| CapabilityError::ArithmeticOverflow {
        operation: "Qwen full-attention block count",
    })?;
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
        u64::try_from(config.deepstack_layer_count()).map_err(|_| {
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
    config: &crate::composition::mlx_architectures::gemma4::vision::Gemma4VisionConfig,
    text_hidden: u64,
    payload: &Array,
    metadata: input::InputMetadata<'_>,
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
    config: &crate::composition::mlx_architectures::gemma4::audio::Gemma4AudioConfig,
    text_hidden: u64,
    payload: &Array,
    metadata: input::InputMetadata<'_>,
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
    metadata: input::InputMetadata<'_>,
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
        metadata: input::InputMetadata<'_>,
        stream: &Stream,
    ) -> Result<(u64, u64), CapabilityError> {
        match self {
            Self::Qwen3Vl(model) | Self::Qwen3VlMoe(model) => qwen_vision_workspace(
                &model.args().vision_config,
                modality,
                payload,
                metadata,
                stream,
                self.model_type(),
            ),
            Self::Qwen3Next(model) | Self::Qwen35(model) => model
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
            Self::Gemma4(model) => {
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
                inkling_workspace(model.args(), modality, payload, metadata, self.model_type())
            }
            Self::DeepSeek(_)
            | Self::GptOss(_)
            | Self::KimiLinear(_)
            | Self::Llama(_)
            | Self::Lfm2(_)
            | Self::NemotronH(_)
            | Self::Qwen(_) => Err(CapabilityError::UnsupportedInput {
                architecture: self.model_type().into(),
                reason: format!("{} media is not supported", modality.as_str()),
            }),
            Self::MuseGlimmer(model) => {
                let config = model.args().vision_config.as_ref().ok_or_else(|| {
                    CapabilityError::UnsupportedInput {
                        architecture: self.model_type().into(),
                        reason: "loaded Muse-Glimmer artifact has no vision projector".into(),
                    }
                })?;
                if modality == Modality::Audio
                    || (modality == Modality::Video
                        && model.args().weight_convention == muse_glimmer::WeightConvention::Gguf)
                {
                    return Err(CapabilityError::UnsupportedInput {
                        architecture: self.model_type().into(),
                        reason: format!(
                            "loaded Muse-Glimmer artifact does not support {}",
                            modality.as_str()
                        ),
                    });
                }
                let grid =
                    metadata
                        .vision_grid_thw
                        .ok_or_else(|| CapabilityError::UnsupportedInput {
                            architecture: self.model_type().into(),
                            reason: "Muse-Glimmer media requires vision_grid_thw metadata".into(),
                        })?;
                if grid.ndim() != 2 || grid.dim(1) != 3 {
                    return Err(CapabilityError::UnsupportedInput {
                        architecture: self.model_type().into(),
                        reason: format!(
                            "Muse-Glimmer vision_grid_thw must be [items, 3], got {:?}",
                            grid.shape()
                        ),
                    });
                }
                let evaluated = grid
                    .evaluated()
                    .map_err(|error| CapabilityError::Observation(error.to_string()))?;
                let values = evaluated
                    .try_as_slice::<i32>()
                    .map_err(|error| CapabilityError::Observation(error.to_string()))?;
                if values.len() % 3 != 0 {
                    return Err(CapabilityError::UnsupportedInput {
                        architecture: self.model_type().into(),
                        reason: "Muse-Glimmer vision_grid_thw has an incomplete row".into(),
                    });
                }
                let mut patches = 0u64;
                let mut positions = 0u64;
                for entry in values.chunks_exact(3) {
                    if entry.iter().any(|value| *value <= 0)
                        || entry[1] % config.merge_size != 0
                        || entry[2] % config.merge_size != 0
                    {
                        return Err(CapabilityError::UnsupportedInput {
                            architecture: self.model_type().into(),
                            reason:
                                "Muse-Glimmer vision grids must be positive and merge-divisible"
                                    .into(),
                        });
                    }
                    let t = entry[0] as u64;
                    let h = entry[1] as u64;
                    let w = entry[2] as u64;
                    patches = checked_add(
                        patches,
                        checked_mul(
                            checked_mul(t, h, "Muse vision t*h")?,
                            w,
                            "Muse vision patches",
                        )?,
                        "Muse vision patch total",
                    )?;
                    positions = checked_add(
                        positions,
                        checked_mul(
                            checked_mul(t, h / config.merge_size as u64, "Muse merged t*h")?,
                            w / config.merge_size as u64,
                            "Muse merged positions",
                        )?,
                        "Muse merged position total",
                    )?;
                }
                if positive(payload.dim(0), "Muse vision payload patches")? != patches {
                    return Err(CapabilityError::UnsupportedInput {
                        architecture: self.model_type().into(),
                        reason: format!(
                            "Muse-Glimmer payload has {} patches but metadata describes {patches}",
                            payload.dim(0)
                        ),
                    });
                }
                let input = array_bytes(payload, "Muse-Glimmer prepared media bytes")?;
                let graph = checked_mul(
                    patches,
                    positive(config.hidden_size, "Muse vision hidden size")?,
                    "Muse vision activation scalars",
                )?;
                Ok((
                    positions,
                    checked_add(
                        input,
                        four_byte_scalars(
                            checked_mul(graph, 8, "Muse vision graph multiplier")?,
                            "Muse vision graph bytes",
                        )?,
                        "Muse total media workspace",
                    )?,
                ))
            }
        }
    }
}

pub(crate) fn model_capabilities(model: &Model) -> Result<ModelCapabilities, CapabilityError> {
    model
        .capabilities_and_estimate()
        .map(|(capabilities, _)| capabilities)
}

pub(crate) fn count_prepared_input(
    model: &Model,
    prepared: input::ModelInput<'_>,
    stream: &Stream,
) -> Result<InputTokenCount, CapabilityError> {
    let mut text_tokens = 0u64;
    let mut media_positions = 0u64;
    let mut media_execution_workspace_bytes = 0u64;
    let mut media_execution_workspace_kind = ObservationKind::Exact;
    for part in prepared.parts {
        match (part.modality, part.payload) {
            (Modality::Text, InputPayload::TokenIds(tokens)) => {
                if tokens.ndim() != 2 || tokens.dim(0) != 1 {
                    return Err(CapabilityError::UnsupportedInput {
                        architecture: model.model_type().into(),
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
                    architecture: model.model_type().into(),
                    reason: "prepared text is not represented by tokenizer IDs".into(),
                });
            }
            (_modality, InputPayload::Embeddings(embeddings)) => {
                if embeddings.ndim() != 3 || embeddings.dim(0) != 1 {
                    return Err(CapabilityError::UnsupportedInput {
                        architecture: model.model_type().into(),
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
                let (positions, workspace_bytes) =
                    model.prepared_media_accounting(modality, tensor, part.metadata, stream)?;
                media_positions =
                    checked_add(media_positions, positions, "prepared media-position total")?;
                media_execution_workspace_bytes = checked_add(
                    media_execution_workspace_bytes,
                    workspace_bytes,
                    "prepared media-workspace total",
                )?;
                media_execution_workspace_kind = ObservationKind::Conservative;
            }
            (_, InputPayload::TokenIds(_)) => {
                return Err(CapabilityError::UnsupportedInput {
                    architecture: model.model_type().into(),
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

pub(crate) fn model_runtime_state(
    model: &Model,
    input: InputTokenCount,
    max_output_tokens: u64,
    batch_size: u64,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    let (_, estimate) = model.capabilities_and_estimate()?;
    estimate_mlx_runtime_state(&estimate, input, max_output_tokens, batch_size)
}

pub(crate) fn static_model_memory(
    session: &MlxModelSession<'_>,
) -> Result<StaticMemoryReport, CapabilityError> {
    let residency = session
        .residency_report()
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
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
            Observed::Available {
                value: logical,
                kind: ObservationKind::Exact,
                source: "validated bounded-residency plan".into(),
            },
            Observed::Available {
                value: resident.get(MemoryTier::Host),
                kind: ObservationKind::Exact,
                source: "bounded-residency manager".into(),
            },
            Observed::Available {
                value: resident.get(MemoryTier::Device),
                kind: ObservationKind::Exact,
                source: "bounded-residency manager".into(),
            },
            Observed::Available {
                value: planned.get(MemoryTier::Disk),
                kind: ObservationKind::Exact,
                source: "bounded-residency plan".into(),
            },
            Observed::Available {
                value: report.weight_store().currently_mapped_shards as u64,
                kind: ObservationKind::Observational,
                source: "checkpoint-store mapping cache".into(),
            },
        )
    } else {
        (
            Observed::Unavailable {
                reason: "loaded model exposes neither resident parameters nor a residency plan"
                    .into(),
            },
            Observed::Unavailable {
                reason: "host residency unavailable".into(),
            },
            Observed::Unavailable {
                reason: "device residency unavailable".into(),
            },
            Observed::Unavailable {
                reason: "disk residency unavailable".into(),
            },
            Observed::Unavailable {
                reason: "mapping information unavailable".into(),
            },
        )
    };
    Ok(StaticMemoryReport {
        logical_parameter_bytes: logical,
        current_host_resident_bytes: host,
        current_device_resident_bytes: device,
        planned_disk_backed_bytes: disk,
        backend_active_allocation_bytes: runtime_counter(
            safemlx::memory::active_memory,
            "process-global MLX active allocation counter",
        ),
        backend_allocator_cache_bytes: runtime_counter(
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

fn complete_model<'a>(session: &'a MlxModelSession<'_>) -> Result<&'a Model, CapabilityError> {
    session.complete_model_for_capabilities().ok_or_else(|| {
        CapabilityError::Observation(
            "MLX capability derivation for distributed model sessions is unavailable".into(),
        )
    })
}

impl<'a> ModelCapabilityBackend for MlxBackend<'a> {
    fn model_capabilities(
        runtime: &ModelRuntime<Self>,
    ) -> Result<ModelCapabilities, CapabilityError> {
        model_capabilities(complete_model(runtime.session())?)
    }

    fn count_prepared_input(
        runtime: &ModelRuntime<Self>,
        prepared: &MlxModelInput,
    ) -> Result<InputTokenCount, CapabilityError> {
        let model = complete_model(runtime.session())?;
        prepared
            .with_borrowed(|input| count_prepared_input(model, input, runtime.backend().stream()))
    }

    fn estimate_runtime_state(
        runtime: &ModelRuntime<Self>,
        input: InputTokenCount,
        max_output_tokens: u64,
        batch_size: u64,
    ) -> Result<RuntimeStateEstimate, CapabilityError> {
        model_runtime_state(
            complete_model(runtime.session())?,
            input,
            max_output_tokens,
            batch_size,
        )
    }

    fn static_memory(runtime: &ModelRuntime<Self>) -> Result<StaticMemoryReport, CapabilityError> {
        static_model_memory(runtime.session())
    }
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
        Observed::Available {
            value: total,
            kind: ObservationKind::Exact,
            source: "macOS sysctl hw.memsize".into(),
        }
    } else {
        Observed::Unavailable {
            reason: std::io::Error::last_os_error().to_string(),
        }
    };
    let available = unsafe { os_proc_available_memory() };
    let available_memory_bytes = match u64::try_from(available) {
        Ok(value) if value > 0 => Observed::Available {
            value,
            kind: ObservationKind::Estimated,
            source: "macOS os_proc_available_memory".into(),
        },
        _ => Observed::Unavailable {
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
            || Observed::Unavailable {
                reason: "/proc/meminfo has no MemTotal".into(),
            },
            |value| Observed::Available {
                value,
                kind: ObservationKind::Exact,
                source: "Linux /proc/meminfo MemTotal".into(),
            },
        ),
        available_memory_bytes: value("MemAvailable").map_or_else(
            || Observed::Unavailable {
                reason: "/proc/meminfo has no MemAvailable".into(),
            },
            |value| Observed::Available {
                value,
                kind: ObservationKind::Estimated,
                source: "Linux /proc/meminfo MemAvailable".into(),
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
            physical_memory_bytes: Observed::Unavailable {
                reason: "GlobalMemoryStatusEx failed".into(),
            },
            available_memory_bytes: Observed::Unavailable {
                reason: "GlobalMemoryStatusEx failed".into(),
            },
            physical_semantics: PhysicalMemorySemantics::Unknown,
        });
    }
    Ok(AvailableMemory {
        physical_memory_bytes: Observed::Available {
            value: status.total_physical,
            kind: ObservationKind::Exact,
            source: "Windows GlobalMemoryStatusEx ullTotalPhys".into(),
        },
        available_memory_bytes: Observed::Available {
            value: status.available_physical,
            kind: ObservationKind::Estimated,
            source: "Windows GlobalMemoryStatusEx ullAvailPhys".into(),
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
            physical_memory_bytes: Observed::Unavailable {
                reason: "portable physical-memory query is not implemented on this platform".into(),
            },
            available_memory_bytes: Observed::Unavailable {
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
    use safemlx::{Device, DeviceType};
    use serde_json::json;

    #[test]
    fn qwen2_runtime_state_splits_full_and_sliding_gqa_layers() {
        let args = eredu_architectures::qwen::model_args_from_config_value(&json!({
                "model_type": "qwen2", "hidden_size": 16, "num_hidden_layers": 6,
                "intermediate_size": 32, "num_attention_heads": 4,
                "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 64,
                "max_position_embeddings": 128, "rope_theta": 10000.0,
                "tie_word_embeddings": false, "use_sliding_window": true,
                "sliding_window": 8, "max_window_layers": 4
        }))
        .unwrap();
        let (_, _, strategy, _, estimate) = qwen_spec(&args, false).unwrap();
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
        let mut args = eredu_architectures::qwen::model_args_from_config_value(&json!({
                "model_type": "qwen2", "hidden_size": 16, "num_hidden_layers": 4,
                "intermediate_size": 32, "num_attention_heads": 4,
                "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 64,
                "max_position_embeddings": 128, "rope_theta": 10000.0,
                "tie_word_embeddings": false
        }))
        .unwrap();
        args.attention_schedule = crate::core::attention::LayerSchedule::new(
            4,
            vec![
                crate::core::attention::AttentionPolicy::sliding(4).unwrap(),
                crate::core::attention::AttentionPolicy::Full,
                crate::core::attention::AttentionPolicy::sliding(8).unwrap(),
                crate::core::attention::AttentionPolicy::sliding(4).unwrap(),
            ],
        )
        .unwrap();
        let (_, _, strategy, _, layout) = qwen_spec(&args, false).unwrap();
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
        let state = estimate_mlx_runtime_state(&layout, InputTokenCount::text(10), 0, 2).unwrap();
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
        use crate::core::attention::{AttentionPolicy, LayerSchedule};

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
        let state_layout = gpt_oss::state_layout(&args).unwrap();
        assert_eq!(state_layout.len(), 4);
        for (layer, attention) in args.attention_schedule.iter().copied().enumerate() {
            assert_eq!(
                state_layout.layer(layer),
                Some(
                    &eredu_core::cache::LayerCachePolicy::key_value(
                        attention,
                        args.num_key_value_heads,
                        args.head_dim,
                    )
                    .unwrap()
                )
            );
        }
        let cache = gpt_oss::Cache::new_device(&args).unwrap();
        assert_eq!(cache.layout.as_ref(), Some(&state_layout));
        assert_eq!(cache.layers.len(), state_layout.len());
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
        let state = estimate_mlx_runtime_state(&estimate, InputTokenCount::text(10), 0, 2).unwrap();
        assert_eq!(state.fixed_state_bytes, 512);
        assert_eq!(state.context_state_bytes, 320);
        assert_eq!(state.bytes_per_position_per_batch, 0);
        assert_eq!(state.assumptions.sliding_window_bounds, vec![5]);
    }

    #[test]
    fn qwen_hybrid_runtime_state_uses_the_normalized_schedule() {
        let args = qwen3_5::model_args_from_config_value(&json!({
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

    fn tiny_llama(kv_heads: i32, sliding_window: Option<i32>) -> LlamaModelArgs {
        LlamaModelArgs {
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
            attention_schedule: match sliding_window {
                Some(window) => crate::core::attention::LayerSchedule::all_sliding(
                    2,
                    u32::try_from(window).unwrap(),
                )
                .unwrap(),
                None => crate::core::attention::LayerSchedule::all_full(2).unwrap(),
            },
            quantization: None,
            quantization_config: None,
            quantized_weights: None,
            quantized_weight_configs: None,
        }
    }

    fn tiny_gemma4() -> gemma4::ModelArgs {
        let layer = |attention, key_value| gemma4::LayerPolicy {
            attention,
            head_dim: std::num::NonZeroU32::new(4).unwrap(),
            num_key_value_heads: std::num::NonZeroU32::new(1).unwrap(),
            key_value,
            intermediate_size: std::num::NonZeroU32::new(16).unwrap(),
            feed_forward: gemma4::FeedForwardPolicy::Dense,
        };
        gemma4::ModelArgs {
            model_type: "gemma4_unified".into(),
            hidden_size: 8,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            rms_norm_eps: 1e-5,
            vocab_size: 32,
            pad_token_id: 0,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            attention_bias: false,
            quantized: false,
            weight_quantization: None,
            quantized_weights: None,
            quantized_weight_configs: None,
            quantization_group_size: 64,
            quantization_bits: 4,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            layer_schedule: crate::core::attention::LayerSchedule::new(
                4,
                vec![
                    layer(
                        AttentionPolicy::sliding(4).unwrap(),
                        gemma4::KeyValuePolicy::Local {
                            value: gemma4::ValuePolicy::Projected,
                        },
                    ),
                    layer(
                        AttentionPolicy::Full,
                        gemma4::KeyValuePolicy::Publish {
                            value: gemma4::ValuePolicy::Projected,
                        },
                    ),
                    layer(
                        AttentionPolicy::sliding(4).unwrap(),
                        gemma4::KeyValuePolicy::Local {
                            value: gemma4::ValuePolicy::Projected,
                        },
                    ),
                    layer(AttentionPolicy::Full, gemma4::KeyValuePolicy::Shared),
                ],
            )
            .unwrap(),
            final_logit_softcapping: None,
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
        estimate_mlx_runtime_state(
            &StateLayout {
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
            estimate_mlx_runtime_state(&llama_layout, InputTokenCount::text(10), 0, 1).unwrap();
        assert_eq!(llama.requested_state_bytes, 2 * 2 * 4 * 8 * 10 * 4);

        let (_, _, _, _, gqa_layout) = llama_spec(&tiny_llama(1, None), false).unwrap();
        let gqa = estimate_mlx_runtime_state(&gqa_layout, InputTokenCount::text(10), 0, 1).unwrap();
        assert_eq!(gqa.requested_state_bytes, llama.requested_state_bytes / 4);
    }

    #[test]
    fn llama_runtime_state_groups_exact_per_layer_windows() {
        use crate::core::attention::{AttentionPolicy, LayerSchedule};

        let mut args = tiny_llama(2, None);
        args.attention_schedule = LayerSchedule::new(
            2,
            vec![AttentionPolicy::Full, AttentionPolicy::sliding(3).unwrap()],
        )
        .unwrap();
        let (_, _, strategy, _, layout) = llama_spec(&args, false).unwrap();
        assert_eq!(
            strategy,
            CacheStateStrategy::MixedKv {
                full_layers: 1,
                sliding: vec![SlidingWindowLayerCount {
                    layers: 1,
                    window: 3,
                }],
            }
        );
        let estimate =
            estimate_mlx_runtime_state(&layout, InputTokenCount::text(10), 0, 2).unwrap();
        assert_eq!(estimate.context_state_bytes, (10 + 3) * 2 * 8 * 2 * 2 * 4);
        assert_eq!(estimate.assumptions.sliding_window_bounds, vec![3]);
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
        let args = kimi_linear::model_args_from_config_value(&json!({
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
        let (native, effective, strategy, modalities, estimate) = kimi_linear_spec(&args).unwrap();
        assert_eq!(native.value(), Some(&128));
        assert_eq!(effective.value(), Some(&128));
        assert_eq!(
            strategy,
            CacheStateStrategy::HybridRecurrent {
                full_attention_layers: 2,
                sliding_attention: Vec::new(),
                recurrent_layers: 2,
            }
        );
        assert_eq!(modalities, InputModalities::TEXT);
        assert_eq!(estimate.fixed_scalars_per_batch, 112);
        assert_eq!(estimate.growing.len(), 1);
        assert_eq!(estimate.growing[0].layers, 2);
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
        let count = InputTokenCount::prepared(7, 12, 19, 1_024, ObservationKind::Conservative);
        assert_eq!(
            count.text_tokens + count.media_positions,
            count.model_positions
        );
        assert_eq!(count.media_execution_workspace_bytes(), 1_024);
        assert_eq!(
            count.media_execution_workspace_kind(),
            ObservationKind::Conservative
        );
    }

    #[test]
    fn gemma4_prepared_patch_positions_ignore_padding() {
        let positions = Array::from_slice(&[0, 0, 1, 0, 0, 1, 1, 1, -1, -1], &[1, 5, 2]);
        assert_eq!(gemma_valid_patch_count(&positions, "gemma4").unwrap(), 4);
    }

    #[test]
    fn gemma4_prepared_media_bounds_vision_and_audio_workspaces() {
        let vision_config =
            crate::composition::mlx_architectures::gemma4::vision::Gemma4VisionConfig {
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
                weight_quantization: None,
            };
        let vision = Array::from_slice(&[0.0f32; 15], &[1, 5, 3]);
        let patch_positions = Array::from_slice(&[0, 0, 1, 0, 0, 1, 1, 1, -1, -1], &[1, 5, 2]);
        let (vision_positions, vision_workspace) = gemma_vision_workspace(
            &vision_config,
            8,
            &vision,
            super::input::InputMetadata::patch_position_ids(&patch_positions),
            "gemma4",
        )
        .unwrap();
        assert_eq!(vision_positions, 1);
        assert!(vision_workspace > array_bytes(&vision, "test vision bytes").unwrap());

        let audio_config =
            crate::composition::mlx_architectures::gemma4::audio::Gemma4AudioConfig {
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
                weight_quantization: None,
            };
        let audio = Array::from_slice(&[0.0f32; 8 * 128], &[1, 8, 128]);
        let audio_mask =
            Array::from_slice(&[true, true, true, true, true, true, false, false], &[1, 8]);
        let (audio_positions, audio_workspace) = gemma_audio_workspace(
            &audio_config,
            8,
            &audio,
            super::input::InputMetadata::audio_mask(&audio_mask),
            "gemma4",
        )
        .unwrap();
        assert_eq!(audio_positions, 2);
        assert!(audio_workspace > array_bytes(&audio, "test audio bytes").unwrap());
    }

    #[test]
    fn qwen_prepared_grid_bounds_vision_workspace() {
        let config = crate::composition::mlx_architectures::qwen::vl::vision::VisionConfig {
            layer_schedule: crate::core::attention::LayerSchedule::new(
                2,
                vec![
                    crate::composition::mlx_architectures::qwen::vl::vision::VisionLayerPolicy {
                        attention:
                            crate::composition::mlx_architectures::qwen::vl::vision::VisionAttentionPolicy::Windowed,
                        deepstack_merger: Some(0),
                    },
                    crate::composition::mlx_architectures::qwen::vl::vision::VisionLayerPolicy {
                        attention:
                            crate::composition::mlx_architectures::qwen::vl::vision::VisionAttentionPolicy::Full,
                        deepstack_merger: None,
                    },
                ],
            )
            .unwrap(),
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
            quantized_weight_configs: Default::default(),
        };
        let payload = Array::from_slice(&[0.0f32; 12], &[4, 3]);
        let grid = Array::from_slice(&[1, 2, 2], &[1, 3]);
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let (positions, workspace) = qwen_vision_workspace(
            &config,
            Modality::Image,
            &payload,
            super::input::InputMetadata::qwen_grid_thw(&grid),
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
        use crate::composition::mlx_architectures::inkling::model::{
            FeedForwardPolicy, LayerPolicy,
        };
        use crate::core::attention::LayerSchedule;

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
        let state = estimate_mlx_runtime_state(&layout, InputTokenCount::text(10), 0, 2).unwrap();
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
            super::input::InputMetadata::empty(),
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
            super::input::InputMetadata::audio_mask(&mask),
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
        let estimate = estimate_mlx_runtime_state(
            &layout,
            InputTokenCount::prepared(5, 3, 8, 1_024, ObservationKind::Conservative),
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
        let attentions = [
            AttentionPolicy::sliding(3).unwrap(),
            AttentionPolicy::Full,
            AttentionPolicy::sliding(5).unwrap(),
            AttentionPolicy::Full,
        ];
        args.layer_schedule = crate::core::attention::LayerSchedule::new(
            4,
            args.layer_schedule
                .iter()
                .copied()
                .zip(attentions)
                .map(|(policy, attention)| gemma4::LayerPolicy {
                    attention,
                    ..policy
                })
                .collect(),
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
    fn gemma4_runtime_state_uses_each_scheduled_kv_geometry() {
        let mut args = tiny_gemma4();
        let mut policies = args.layer_schedule.iter().copied().collect::<Vec<_>>();
        policies[0].head_dim = std::num::NonZeroU32::new(8).unwrap();
        args.layer_schedule = crate::core::attention::LayerSchedule::new(4, policies).unwrap();

        let (_, _, _, _, layout) = gemma4_spec(&args, text_modalities()).unwrap();
        assert_eq!(layout.growing.len(), 2);
        assert_eq!(layout.growing[0].layers, 2);
        assert_eq!(layout.growing[0].scalars_per_position, 8);
        assert_eq!(layout.growing[0].window, None);
        assert_eq!(layout.growing[1].layers, 1);
        assert_eq!(layout.growing[1].scalars_per_position, 16);
        assert_eq!(layout.growing[1].window, None);
    }

    #[test]
    fn checked_arithmetic_reports_overflow() {
        assert_eq!(
            checked_mul(u64::MAX, 2, "synthetic overflow"),
            Err(CapabilityError::ArithmeticOverflow {
                operation: "synthetic overflow"
            })
        );

        let layout = StateLayout {
            fixed_scalars_per_batch: 0,
            growing: Vec::new(),
            hidden_size: 1,
            allocation_granularity: 1,
            completeness: EstimationCompleteness::Complete,
        };
        assert!(matches!(
            estimate_mlx_runtime_state(
                &layout,
                InputTokenCount::prepared(0, 1, 1, u64::MAX, ObservationKind::Conservative,),
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
        let value: Observed<u64> = Observed::Unavailable {
            reason: "synthetic".into(),
        };
        assert_eq!(value.value(), None);
    }

    #[test]
    fn apple_unified_semantics_do_not_create_two_capacities() {
        let report = AvailableMemory {
            physical_memory_bytes: Observed::Available {
                value: 16,
                kind: ObservationKind::Exact,
                source: "test".into(),
            },
            available_memory_bytes: Observed::Available {
                value: 8,
                kind: ObservationKind::Estimated,
                source: "test".into(),
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
        let unsupported: Observed<u64> = Observed::Unsupported {
            reason: "not supported".into(),
        };
        assert!(unsupported.value().is_none());
    }

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
}
