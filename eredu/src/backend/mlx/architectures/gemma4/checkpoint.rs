//! Architecture-owned checkpoint contracts for Gemma 4 text and media weights.

use eredu_checkpoint::{StoredDtype, WeightQuantization};

use std::collections::HashMap;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::{
    audio::Gemma4AudioConfig,
    model::{self, FeedForwardPolicy, Gemma4MmprojGguf, ModelArgs, ValuePolicy},
    vision::Gemma4VisionConfig,
};
use crate::backend::mlx::runtime::checkpoint::{store::SafetensorsWeightStore, validation};
use eredu_checkpoint::schema::{
    AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::validation::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation};

#[derive(Debug, Clone, Eq, PartialEq)]
struct MediaSpec {
    name: String,
    shape: Vec<usize>,
    required: bool,
    quantizable: bool,
}

pub(crate) fn validate_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    bounded: bool,
    expert_cache: bool,
) -> CheckpointValidation {
    if expert_cache {
        return invalid_geometry(
            "Gemma 4 SafeTensors does not expose a sparse-expert-cache load route".into(),
        );
    }
    let (args, vision, _, _, audio, _) = match model::model_config_from_value(config) {
        Ok(config) => config,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match safetensors_plan(&args, vision.as_ref(), audio.as_ref(), bounded) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validation::validate_safetensors_plan(store, &plan)
}

pub(crate) fn safetensors_plan(
    args: &ModelArgs,
    vision: Option<&Gemma4VisionConfig>,
    audio: Option<&Gemma4AudioConfig>,
    bounded: bool,
) -> Result<SafetensorsCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let layers = dimension(args.num_hidden_layers, "layer count")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let quantization = args.weight_quantization();
    let mut common = Vec::new();
    let mut groups = Vec::new();
    add_text_tensor(
        &mut common,
        "model.language_model.embed_tokens.weight",
        vec![vocab, hidden],
        TensorOperation::Matrix,
        quantization,
        true,
    )?;
    add_text_tensor(
        &mut common,
        "model.language_model.norm.weight",
        vec![hidden],
        TensorOperation::Vector,
        quantization,
        true,
    )?;
    if !args.tie_word_embeddings {
        add_text_tensor(
            &mut common,
            "lm_head.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
            quantization,
            false,
        )?;
    }
    if args.hidden_size_per_layer_input > 0 {
        let per_layer = dimension(
            args.hidden_size_per_layer_input,
            "per-layer input hidden size",
        )?;
        let combined = checked_mul(layers, per_layer, "combined per-layer input width")?;
        let per_layer_vocab = dimension(
            args.vocab_size_per_layer_input.unwrap_or(args.vocab_size),
            "per-layer input vocabulary size",
        )?;
        for (name, shape, operation) in [
            (
                "model.language_model.embed_tokens_per_layer.weight",
                vec![per_layer_vocab, combined],
                TensorOperation::Matrix,
            ),
            (
                "model.language_model.per_layer_model_projection.weight",
                vec![combined, hidden],
                TensorOperation::Matrix,
            ),
            (
                "model.language_model.per_layer_projection_norm.weight",
                vec![per_layer],
                TensorOperation::Vector,
            ),
        ] {
            add_text_tensor(&mut common, name, shape, operation, quantization, true)?;
        }
    }

    for layer in 0..layers {
        let prefix = format!("model.language_model.layers.{layer}");
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| format!("Gemma 4 layer {layer} has no normalized policy"))?;
        let head_dim = policy.head_dim.get() as usize;
        let kv_heads = policy.num_key_value_heads.get() as usize;
        let query = checked_mul(
            dimension(args.num_attention_heads, "attention head count")?,
            head_dim,
            "query projection width",
        )?;
        let key_value = checked_mul(kv_heads, head_dim, "key/value projection width")?;
        let shared_kv = !policy.key_value.owns_state();
        let attention_k_eq_v = policy.key_value.value() == Some(ValuePolicy::ReuseKey);
        let intermediate = policy.intermediate_size.get() as usize;
        for name in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ] {
            add_text_tensor(
                &mut common,
                &format!("{prefix}.{name}"),
                vec![hidden],
                TensorOperation::Vector,
                quantization,
                true,
            )?;
        }
        add_text_tensor(
            &mut common,
            &format!("{prefix}.layer_scalar"),
            vec![1],
            TensorOperation::Vector,
            quantization,
            true,
        )?;
        for (name, shape, operation) in [
            (
                "self_attn.q_proj.weight",
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.o_proj.weight",
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.q_norm.weight",
                vec![head_dim],
                TensorOperation::Vector,
            ),
            (
                "mlp.gate_proj.weight",
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            (
                "mlp.up_proj.weight",
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            (
                "mlp.down_proj.weight",
                vec![hidden, intermediate],
                TensorOperation::Matrix,
            ),
        ] {
            add_text_tensor(
                &mut common,
                &format!("{prefix}.{name}"),
                shape,
                operation,
                quantization,
                true,
            )?;
        }
        if !shared_kv {
            for (name, shape, operation) in [
                (
                    "self_attn.k_proj.weight",
                    vec![key_value, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "self_attn.k_norm.weight",
                    vec![head_dim],
                    TensorOperation::Vector,
                ),
            ] {
                add_text_tensor(
                    &mut common,
                    &format!("{prefix}.{name}"),
                    shape,
                    operation,
                    quantization,
                    true,
                )?;
            }
            if !attention_k_eq_v {
                add_text_tensor(
                    &mut common,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    vec![key_value, hidden],
                    TensorOperation::Matrix,
                    quantization,
                    true,
                )?;
            }
        }
        if args.hidden_size_per_layer_input > 0 {
            let per_layer = dimension(
                args.hidden_size_per_layer_input,
                "per-layer input hidden size",
            )?;
            for (name, shape, operation) in [
                (
                    "per_layer_input_gate.weight",
                    vec![per_layer, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "per_layer_projection.weight",
                    vec![hidden, per_layer],
                    TensorOperation::Matrix,
                ),
                (
                    "post_per_layer_input_norm.weight",
                    vec![hidden],
                    TensorOperation::Vector,
                ),
            ] {
                add_text_tensor(
                    &mut common,
                    &format!("{prefix}.{name}"),
                    shape,
                    operation,
                    quantization,
                    true,
                )?;
            }
        }
        if policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe {
            let experts = dimension(
                args.num_experts.ok_or_else(|| {
                    "Gemma 4 sparse layer has no configured expert count".to_string()
                })?,
                "expert count",
            )?;
            let moe = dimension(
                args.moe_intermediate_size.ok_or_else(|| {
                    "Gemma 4 sparse layer has no expert intermediate size".to_string()
                })?,
                "expert intermediate size",
            )?;
            for (name, shape, operation) in [
                (
                    "router.proj.weight",
                    vec![experts, hidden],
                    TensorOperation::Matrix,
                ),
                ("router.scale", vec![hidden], TensorOperation::Vector),
                (
                    "router.per_expert_scale",
                    vec![experts],
                    TensorOperation::Vector,
                ),
                (
                    "post_feedforward_layernorm_1.weight",
                    vec![hidden],
                    TensorOperation::Vector,
                ),
                (
                    "pre_feedforward_layernorm_2.weight",
                    vec![hidden],
                    TensorOperation::Vector,
                ),
                (
                    "post_feedforward_layernorm_2.weight",
                    vec![hidden],
                    TensorOperation::Vector,
                ),
            ] {
                add_text_tensor(
                    &mut common,
                    &format!("{prefix}.{name}"),
                    shape,
                    operation,
                    quantization,
                    true,
                )?;
            }
            add_expert_layout(
                &mut common,
                &mut groups,
                &prefix,
                experts,
                hidden,
                moe,
                quantization,
                bounded,
            )?;
        }
    }

    if let Some(config) = vision {
        for spec in vision_specs(config, hidden, "model.vision_tower", "model.embed_vision")? {
            add_media_tensor(&mut common, spec, quantization)?;
        }
    }
    if let Some(config) = audio {
        for spec in audio_specs(
            config,
            hidden,
            "model.audio_tower",
            "model.embed_audio",
            false,
        )? {
            add_media_tensor(&mut common, spec, quantization)?;
        }
    }
    let mut catalog = CatalogPolicy::strict();
    catalog.allowed_prefixes.extend([
        "multi_modal_projector.".into(),
        "model.multi_modal_projector.".into(),
        "model.vision_embedder.".into(),
    ]);
    SafetensorsCheckpointPlan::new("Gemma 4 SafeTensors", common, groups, catalog)
        .map_err(|error| error.to_string())
}

fn add_text_tensor(
    output: &mut Vec<SafetensorsTensorConstraint>,
    canonical: &str,
    shape: Vec<usize>,
    operation: TensorOperation,
    quantization: Option<WeightQuantization>,
    released_alias: bool,
) -> Result<(), String> {
    output.extend(text_constraints(
        canonical,
        shape,
        (operation == TensorOperation::Matrix)
            .then_some(quantization)
            .flatten(),
        true,
        released_alias,
    )?);
    Ok(())
}

fn text_constraints(
    canonical: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
    required: bool,
    released_alias: bool,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    if let Some(released) = released_aliases(canonical, released_alias)
        .into_iter()
        .next()
    {
        safetensors_constraints(
            &released,
            vec![canonical.into()],
            shape,
            quantization,
            required,
        )
    } else {
        safetensors_constraints(canonical, Vec::new(), shape, quantization, required)
    }
}

fn add_media_tensor(
    output: &mut Vec<SafetensorsTensorConstraint>,
    spec: MediaSpec,
    quantization: Option<WeightQuantization>,
) -> Result<(), String> {
    let aliases = spec
        .name
        .strip_prefix("model.")
        .map(|name| vec![name.to_string()])
        .unwrap_or_default();
    output.extend(safetensors_constraints(
        &spec.name,
        aliases,
        spec.shape,
        spec.quantizable.then_some(quantization).flatten(),
        spec.required,
    )?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_expert_layout(
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
    layer_prefix: &str,
    experts: usize,
    hidden: usize,
    intermediate: usize,
    quantization: Option<WeightQuantization>,
    bounded: bool,
) -> Result<(), String> {
    let root = format!("{layer_prefix}.experts.switch_glu");
    let mut variants = Vec::new();
    let canonical_gate = format!("{root}.gate_proj.weight");
    let canonical_up = format!("{root}.up_proj.weight");
    let gate = released_aliases(&canonical_gate, true)
        .into_iter()
        .next()
        .expect("Gemma text expert has a released alias");
    let up = released_aliases(&canonical_up, true)
        .into_iter()
        .next()
        .expect("Gemma text expert has a released alias");
    let mut split = text_constraints(
        &canonical_gate,
        vec![experts, intermediate, hidden],
        quantization,
        true,
        true,
    )?;
    split.extend(text_constraints(
        &canonical_up,
        vec![experts, intermediate, hidden],
        quantization,
        true,
        true,
    )?);
    variants.push(LayoutVariant {
        id: "separate gate and up".into(),
        tensors: split,
        discriminator_keys: vec![gate, up],
    });
    if bounded {
        let fused = format!("{root}.gate_up_proj.weight");
        variants.push(LayoutVariant {
            id: "single fused gate and up".into(),
            tensors: safetensors_constraints(
                &fused,
                Vec::new(),
                vec![
                    experts,
                    checked_mul(2, intermediate, "fused expert width")?,
                    hidden,
                ],
                quantization,
                true,
            )?,
            discriminator_keys: vec![fused],
        });
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("{root} gate/up storage"),
        required: true,
        variants,
    });
    let down = format!("{root}.down_proj.weight");
    common.extend(text_constraints(
        &down,
        vec![experts, hidden, intermediate],
        quantization,
        true,
        true,
    )?);
    Ok(())
}

fn safetensors_constraints(
    canonical: &str,
    mut aliases: Vec<String>,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
    required: bool,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    if let Some(quantization) = quantization {
        let input = *shape
            .last()
            .ok_or_else(|| format!("Gemma 4 quantized tensor {canonical:?} has scalar shape"))?;
        let bits = quantization.bits() as usize;
        let group = quantization.group_size() as usize;
        let packed_bits = checked_mul(input, bits, "affine packing")?;
        if !input.is_multiple_of(group)
            || !input.is_multiple_of(32)
            || !packed_bits.is_multiple_of(32)
        {
            return Err(format!(
                "Gemma 4 quantized tensor {canonical:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
            ));
        }
        let released = aliases.clone();
        aliases.extend(
            std::iter::once(canonical.to_string())
                .chain(released.iter().cloned())
                .filter_map(|name| {
                    name.strip_suffix(".weight")
                        .map(|prefix| format!("{prefix}.inner.weight"))
                }),
        );
        aliases.retain(|alias| alias != canonical);
        aliases.sort();
        aliases.dedup();
        let mut packed = shape.clone();
        *packed.last_mut().expect("matrix shape") = packed_bits / 32;
        let mut companion_shape = shape;
        *companion_shape.last_mut().expect("matrix shape") = input / group;
        let outer = canonical.strip_suffix(".weight").unwrap_or(canonical);
        let companion_aliases = released
            .iter()
            .map(|name| name.strip_suffix(".weight").unwrap_or(name))
            .collect::<Vec<_>>();
        let companion_dtype = || {
            StoredDtypeConstraint::OneOf(vec![
                StoredDtype::F16,
                StoredDtype::BF16,
                StoredDtype::F32,
                StoredDtype::U8,
            ])
        };
        let mut result = vec![SafetensorsTensorConstraint::required(
            canonical,
            packed,
            StoredDtypeConstraint::Exact(StoredDtype::U32),
        )
        .with_aliases(aliases)];
        result.push(
            SafetensorsTensorConstraint::required(
                format!("{outer}.scales"),
                companion_shape.clone(),
                companion_dtype(),
            )
            .with_aliases(
                companion_aliases
                    .iter()
                    .map(|outer| format!("{outer}.scales")),
            )
            .companion(),
        );
        if quantization.has_biases() {
            result.push(
                SafetensorsTensorConstraint::required(
                    format!("{outer}.biases"),
                    companion_shape,
                    companion_dtype(),
                )
                .with_aliases(
                    companion_aliases
                        .iter()
                        .map(|outer| format!("{outer}.biases")),
                )
                .companion(),
            );
        }
        return Ok(result);
    }
    let constraint =
        SafetensorsTensorConstraint::required(canonical, shape, StoredDtypeConstraint::Floating)
            .with_aliases(aliases);
    Ok(vec![if required {
        constraint
    } else {
        constraint.optional()
    }])
}

fn released_aliases(canonical: &str, enabled: bool) -> Vec<String> {
    enabled
        .then(|| {
            canonical
                .strip_prefix("model.language_model.")
                .map(|rest| format!("language_model.model.{rest}"))
        })
        .flatten()
        .into_iter()
        .collect()
}

fn vision_specs(
    config: &Gemma4VisionConfig,
    text_hidden: usize,
    root: &str,
    projection_root: &str,
) -> Result<Vec<MediaSpec>, String> {
    let hidden = dimension(config.hidden_size, "vision hidden size")?;
    let intermediate = dimension(config.intermediate_size, "vision intermediate size")?;
    let head_dim = dimension(config.head_dim, "vision head width")?;
    let query = checked_mul(
        dimension(config.num_attention_heads, "vision attention head count")?,
        head_dim,
        "vision query width",
    )?;
    let key_value = checked_mul(
        dimension(config.num_key_value_heads, "vision KV head count")?,
        head_dim,
        "vision key/value width",
    )?;
    let patch = dimension(config.patch_size, "vision patch size")?;
    let patch_area = checked_mul(patch, patch, "vision patch area")?;
    let mut specs = vec![
        media(
            format!("{root}.patch_embedder.input_proj.weight"),
            vec![
                hidden,
                checked_mul(3, patch_area, "vision patch input width")?,
            ],
        ),
        media(
            format!("{root}.patch_embedder.position_embedding_table"),
            vec![
                2,
                dimension(
                    config.position_embedding_size,
                    "vision position embedding size",
                )?,
                hidden,
            ],
        ),
    ];
    if config.standardize {
        for name in ["std_bias", "std_scale"] {
            specs.push(media(format!("{root}.{name}"), vec![hidden]));
        }
    }
    for layer in 0..dimension(config.num_hidden_layers, "vision layer count")? {
        let prefix = format!("{root}.encoder.layers.{layer}");
        for name in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ] {
            specs.push(media(format!("{prefix}.{name}"), vec![hidden]));
        }
        for (name, shape) in [
            ("self_attn.q_proj", vec![query, hidden]),
            ("self_attn.k_proj", vec![key_value, hidden]),
            ("self_attn.v_proj", vec![key_value, hidden]),
            ("self_attn.o_proj", vec![hidden, query]),
            ("mlp.gate_proj", vec![intermediate, hidden]),
            ("mlp.up_proj", vec![intermediate, hidden]),
            ("mlp.down_proj", vec![hidden, intermediate]),
        ] {
            add_clipped_specs(&mut specs, &format!("{prefix}.{name}"), shape);
        }
        for name in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
            specs.push(media(format!("{prefix}.{name}"), vec![head_dim]));
        }
    }
    specs.push(MediaSpec {
        name: format!("{projection_root}.embedding_projection.weight"),
        shape: vec![text_hidden, hidden],
        required: true,
        quantizable: true,
    });
    Ok(specs)
}

fn audio_specs(
    config: &Gemma4AudioConfig,
    text_hidden: usize,
    root: &str,
    projection_root: &str,
    require_output_bias: bool,
) -> Result<Vec<MediaSpec>, String> {
    let hidden = dimension(config.hidden_size, "audio hidden size")?;
    let heads = dimension(config.num_attention_heads, "audio attention head count")?;
    if !hidden.is_multiple_of(heads) {
        return Err(format!(
            "Gemma 4 audio hidden size {hidden} is not divisible by {heads} attention heads"
        ));
    }
    let head = hidden / heads;
    let [first, second] = config.subsampling_conv_channels.as_slice() else {
        return Err("Gemma 4 audio requires exactly two subsampling convolution channels".into());
    };
    let first = dimension(*first, "first audio subsampling channel count")?;
    let second = dimension(*second, "second audio subsampling channel count")?;
    let output = dimension(config.output_proj_dims, "audio output projection width")?;
    let mut specs = vec![
        media(
            format!("{root}.subsample_conv_projection.layer0.conv.weight"),
            vec![first, 3, 3, 1],
        ),
        media(
            format!("{root}.subsample_conv_projection.layer0.norm.weight"),
            vec![first],
        ),
        media(
            format!("{root}.subsample_conv_projection.layer1.conv.weight"),
            vec![second, 3, 3, first],
        ),
        media(
            format!("{root}.subsample_conv_projection.layer1.norm.weight"),
            vec![second],
        ),
        media(
            format!("{root}.subsample_conv_projection.input_proj_linear.weight"),
            vec![hidden, checked_mul(32, second, "audio subsampling width")?],
        ),
        media(format!("{root}.output_proj.weight"), vec![output, hidden]),
        MediaSpec {
            name: format!("{root}.output_proj.bias"),
            shape: vec![output],
            required: require_output_bias,
            quantizable: false,
        },
    ];
    for layer in 0..dimension(config.num_hidden_layers, "audio layer count")? {
        let prefix = format!("{root}.layers.{layer}");
        for name in [
            "feed_forward1.pre_layer_norm.weight",
            "feed_forward1.post_layer_norm.weight",
            "norm_pre_attn.weight",
            "norm_post_attn.weight",
            "lconv1d.pre_layer_norm.weight",
            "lconv1d.conv_norm.weight",
            "feed_forward2.pre_layer_norm.weight",
            "feed_forward2.post_layer_norm.weight",
            "norm_out.weight",
        ] {
            specs.push(media(format!("{prefix}.{name}"), vec![hidden]));
        }
        let four_hidden = checked_mul(4, hidden, "audio feed-forward width")?;
        let two_hidden = checked_mul(2, hidden, "audio convolution gate width")?;
        for (name, shape) in [
            ("feed_forward1.ffw_layer_1", vec![four_hidden, hidden]),
            ("feed_forward1.ffw_layer_2", vec![hidden, four_hidden]),
            ("self_attn.q_proj", vec![hidden, hidden]),
            ("self_attn.k_proj", vec![hidden, hidden]),
            ("self_attn.v_proj", vec![hidden, hidden]),
            ("self_attn.post", vec![hidden, hidden]),
            ("lconv1d.linear_start", vec![two_hidden, hidden]),
            ("lconv1d.linear_end", vec![hidden, hidden]),
            ("feed_forward2.ffw_layer_1", vec![four_hidden, hidden]),
            ("feed_forward2.ffw_layer_2", vec![hidden, four_hidden]),
        ] {
            add_clipped_specs(&mut specs, &format!("{prefix}.{name}"), shape);
        }
        for (name, shape) in [
            ("self_attn.relative_k_proj.weight", vec![hidden, hidden]),
            ("self_attn.per_dim_scale", vec![head]),
            (
                "lconv1d.depthwise_conv1d.weight",
                vec![
                    hidden,
                    dimension(config.conv_kernel_size, "audio convolution kernel size")?,
                    1,
                ],
            ),
        ] {
            specs.push(media(format!("{prefix}.{name}"), shape));
        }
    }
    specs.push(MediaSpec {
        name: format!("{projection_root}.embedding_projection.weight"),
        shape: vec![text_hidden, output],
        required: true,
        quantizable: true,
    });
    Ok(specs)
}

fn media(name: impl Into<String>, shape: Vec<usize>) -> MediaSpec {
    MediaSpec {
        name: name.into(),
        shape,
        required: true,
        quantizable: false,
    }
}

fn add_clipped_specs(output: &mut Vec<MediaSpec>, prefix: &str, shape: Vec<usize>) {
    output.push(media(format!("{prefix}.linear.weight"), shape));
    for suffix in ["input_min", "input_max", "output_min", "output_max"] {
        output.push(media(format!("{prefix}.{suffix}"), Vec::new()));
    }
}

pub(crate) fn validate_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> CheckpointValidation {
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(model::translate_gguf_weight_name)
    {
        return conflicting_layout(error.to_string());
    }
    let args = match model::gemma4_args_from_gguf_catalog(checkpoint, metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > checkpoint.catalog().physical_tensor_count() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor GGUF catalog",
            args.num_hidden_layers,
            checkpoint.catalog().physical_tensor_count()
        ));
    }
    let plan = match gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validation::validate_gguf_plan(checkpoint, &plan)
}

pub(crate) fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let layers = dimension(args.num_hidden_layers, "layer count")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let mut common = vec![
        gguf(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf("output_norm.weight", vec![hidden], TensorOperation::Vector),
    ];
    let mut groups = Vec::new();
    let mut catalog = CatalogPolicy::strict();
    catalog.allowed_prefixes.push("rope_freqs.".into());
    if !args.tie_word_embeddings {
        common.push(gguf(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ));
    }
    if args.hidden_size_per_layer_input > 0 {
        let per_layer = dimension(
            args.hidden_size_per_layer_input,
            "per-layer input hidden size",
        )?;
        let combined = checked_mul(layers, per_layer, "combined per-layer input width")?;
        let per_layer_vocab = dimension(
            args.vocab_size_per_layer_input.unwrap_or(args.vocab_size),
            "per-layer input vocabulary size",
        )?;
        common.extend([
            gguf(
                "per_layer_token_embd.weight",
                vec![per_layer_vocab, combined],
                TensorOperation::Matrix,
            ),
            gguf(
                "per_layer_model_proj.weight",
                vec![combined, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                "per_layer_proj_norm.weight",
                vec![per_layer],
                TensorOperation::Vector,
            ),
        ]);
    }
    let attention_heads = dimension(args.num_attention_heads, "attention head count")?;
    for layer in 0..layers {
        let root = format!("blk.{layer}");
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| format!("Gemma 4 layer {layer} has no normalized policy"))?;
        let head_dim = policy.head_dim.get() as usize;
        let query = checked_mul(attention_heads, head_dim, "query projection width")?;
        let key_value = checked_mul(
            policy.num_key_value_heads.get() as usize,
            head_dim,
            "key/value projection width",
        )?;
        let shared_kv = !policy.key_value.owns_state();
        let attention_k_eq_v = policy.key_value.value() == Some(ValuePolicy::ReuseKey);
        let intermediate = policy.intermediate_size.get() as usize;
        for name in [
            "attn_norm.weight",
            "post_attention_norm.weight",
            "ffn_norm.weight",
            "post_ffw_norm.weight",
        ] {
            common.push(gguf(
                format!("{root}.{name}"),
                vec![hidden],
                TensorOperation::Vector,
            ));
        }
        common.extend([
            gguf(
                format!("{root}.layer_output_scale.weight"),
                vec![1],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.attn_q.weight"),
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.attn_output.weight"),
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.attn_q_norm.weight"),
                vec![head_dim],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{root}.ffn_gate.weight"),
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.ffn_up.weight"),
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{root}.ffn_down.weight"),
                vec![hidden, intermediate],
                TensorOperation::Matrix,
            ),
        ]);
        if shared_kv {
            catalog.allowed_prefixes.extend([
                format!("{root}.attn_k."),
                format!("{root}.attn_v."),
                format!("{root}.attn_k_norm."),
            ]);
        } else {
            common.extend([
                gguf(
                    format!("{root}.attn_k.weight"),
                    vec![key_value, hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.attn_k_norm.weight"),
                    vec![head_dim],
                    TensorOperation::Vector,
                ),
            ]);
            if !attention_k_eq_v {
                common.push(gguf(
                    format!("{root}.attn_v.weight"),
                    vec![key_value, hidden],
                    TensorOperation::Matrix,
                ));
            }
        }
        if args.hidden_size_per_layer_input > 0 {
            let per_layer = dimension(
                args.hidden_size_per_layer_input,
                "per-layer input hidden size",
            )?;
            common.extend([
                gguf(
                    format!("{root}.inp_gate.weight"),
                    vec![per_layer, hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.proj.weight"),
                    vec![hidden, per_layer],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.post_norm.weight"),
                    vec![hidden],
                    TensorOperation::Vector,
                ),
            ]);
        }
        if policy.feed_forward == FeedForwardPolicy::DenseWithSparseMoe {
            let experts = dimension(
                args.num_experts
                    .ok_or_else(|| "Gemma 4 MoE has no expert count".to_string())?,
                "expert count",
            )?;
            let moe = dimension(
                args.moe_intermediate_size
                    .ok_or_else(|| "Gemma 4 MoE has no expert width".to_string())?,
                "expert intermediate size",
            )?;
            common.extend([
                gguf(
                    format!("{root}.ffn_gate_inp.weight"),
                    vec![experts, hidden],
                    TensorOperation::Matrix,
                ),
                gguf(
                    format!("{root}.ffn_gate_inp.scale"),
                    vec![hidden],
                    TensorOperation::Vector,
                ),
                gguf(
                    format!("{root}.ffn_down_exps.scale"),
                    vec![experts],
                    TensorOperation::Vector,
                ),
                gguf(
                    format!("{root}.post_ffw_norm_1.weight"),
                    vec![hidden],
                    TensorOperation::Vector,
                ),
                gguf(
                    format!("{root}.pre_ffw_norm_2.weight"),
                    vec![hidden],
                    TensorOperation::Vector,
                ),
                gguf(
                    format!("{root}.post_ffw_norm_2.weight"),
                    vec![hidden],
                    TensorOperation::Vector,
                ),
                gguf(
                    format!("{root}.ffn_down_exps.weight"),
                    vec![experts, hidden, moe],
                    TensorOperation::Matrix,
                ),
            ]);
            let fused = format!("{root}.ffn_gate_up_exps.weight");
            let gate = format!("{root}.ffn_gate_exps.weight");
            let up = format!("{root}.ffn_up_exps.weight");
            groups.push(AlternativeLayoutGroup {
                id: format!("{root} expert gate/up storage"),
                required: true,
                variants: vec![
                    LayoutVariant {
                        id: "separate gate and up".into(),
                        tensors: vec![
                            gguf(&gate, vec![experts, moe, hidden], TensorOperation::Matrix),
                            gguf(&up, vec![experts, moe, hidden], TensorOperation::Matrix),
                        ],
                        discriminator_keys: vec![gate, up],
                    },
                    LayoutVariant {
                        id: "single fused gate and up".into(),
                        tensors: vec![gguf(
                            &fused,
                            vec![experts, checked_mul(2, moe, "fused expert width")?, hidden],
                            TensorOperation::Matrix,
                        )],
                        discriminator_keys: vec![fused],
                    },
                ],
            });
        }
    }
    GgufCheckpointPlan::new("Gemma 4 GGUF", common, groups, catalog)
        .map_err(|error| error.to_string())
}

pub(crate) fn validate_mmproj_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: &Gemma4MmprojGguf,
) -> CheckpointValidation {
    if let Err(error) = model::validate_mmproj_metadata(&mmproj.metadata) {
        return invalid_geometry(error.to_string());
    }
    if let Err(error) = mmproj
        .checkpoint
        .catalog()
        .translated_outputs(model::translate_mmproj_weight_name)
    {
        return invalid_geometry(error.to_string());
    }
    let mut args = match model::gemma4_args_from_gguf_catalog(model_checkpoint, model_metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let (vision, _, _, audio, _) = match model::apply_mmproj_args(&mut args, model_metadata, mmproj)
    {
        Ok(parts) => parts,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match mmproj_plan(&args, vision.as_ref(), audio.as_ref()) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validation::validate_gguf_plan(&mmproj.checkpoint, &plan)
}

pub(crate) fn mmproj_plan(
    args: &ModelArgs,
    vision: Option<&Gemma4VisionConfig>,
    audio: Option<&Gemma4AudioConfig>,
) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let mut tensors = Vec::new();
    if let Some(config) = vision {
        for spec in vision_specs(config, hidden, "vision_tower", "embed_vision")? {
            tensors.push(gguf(spec.name, spec.shape, TensorOperation::Dense));
        }
    }
    if let Some(config) = audio {
        for spec in audio_specs(config, hidden, "audio_tower", "embed_audio", true)? {
            tensors.push(gguf(spec.name, spec.shape, TensorOperation::Dense));
        }
    }
    GgufCheckpointPlan::new(
        "Gemma 4 mmproj GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

fn gguf(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Gemma 4 {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Gemma 4 {name} geometry overflows"))
}

fn invalid_geometry(detail: String) -> CheckpointValidation {
    CheckpointValidation::Invalid(vec![CheckpointIssue {
        kind: CheckpointIssueKind::InvalidGeometry,
        detail,
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    }])
}

fn conflicting_layout(detail: String) -> CheckpointValidation {
    CheckpointValidation::Invalid(vec![CheckpointIssue {
        kind: CheckpointIssueKind::ConflictingLayout,
        detail,
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    }])
}
