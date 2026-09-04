//! Pure checkpoint naming, schemas, and derived-weight recipes for DeepSeek.

use std::{collections::BTreeMap, ops::Range};

use eredu_checkpoint::schema::{
    matrix_for_linear_format, AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan,
    GgufTensorConstraint, GgufTypeConstraint, LayoutVariant, MatrixScaleNames,
    SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::{
    expert::{
        resolve_gated_product_expert_recipes, GatedProductExpertLayoutNames,
        GatedProductExpertRecipes, IndependentGatedProductExpertNames,
    },
    recipe::RecipeCatalog,
};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
use eredu_checkpoint::{
    BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat, StoredDtype, WeightQuantization,
};

use super::config::{ExpertFormat, LayerPolicy, V3Args, V4Args, V4AttentionPolicy};

/// Recipes for one independently resident DeepSeek routed expert.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertUnitRecipes {
    /// Fused gate/up projection, optionally restricted to one rank's channels.
    pub gate_up: DerivedWeightRecipe,
    /// Down projection, optionally restricted to one rank's input channels.
    pub down: DerivedWeightRecipe,
}

/// Translates canonical llama.cpp DeepSeek2/V3 tensor names to neutral model
/// parameter identities.
pub fn translate_v3_gguf_weight_name(name: &str) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("output_norm", "model.norm"),
        ("output", "lm_head"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_string();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_string();
    };
    for (source, target) in [
        ("ffn_gate_exps", "mlp.experts.gate_proj"),
        ("ffn_up_exps", "mlp.experts.up_proj"),
        ("ffn_down_exps", "mlp.experts.down_proj"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let suffix = match parameter.strip_prefix(source).unwrap_or_default() {
                ".weight" => "",
                ".scales" => "_scales",
                ".biases" => "_biases",
                other => other,
            };
            return format!("model.layers.{layer}.{target}{suffix}");
        }
    }
    if matches!(parameter, "exp_probs_b.bias" | "ffn_exp_probs_b.bias") {
        return format!("model.layers.{layer}.mlp.gate.e_score_correction_bias");
    }
    for (source, target) in [
        ("attn_q", "self_attn.q_proj"),
        ("attn_q_a", "self_attn.q_a_proj"),
        ("attn_q_b", "self_attn.q_b_proj"),
        ("attn_kv_a_mqa", "self_attn.kv_a_proj_with_mqa"),
        ("attn_kv_b", "self_attn.kv_b_proj"),
        ("attn_k_b", "self_attn.k_b_proj"),
        ("attn_v_b", "self_attn.v_b_proj"),
        ("attn_q_a_norm", "self_attn.q_a_layernorm"),
        ("attn_kv_a_norm", "self_attn.kv_a_layernorm"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("ffn_gate_shexp", "mlp.shared_experts.gate_proj"),
        ("ffn_up_shexp", "mlp.shared_experts.up_proj"),
        ("ffn_down_shexp", "mlp.shared_experts.down_proj"),
        ("ffn_gate_inp", "mlp.gate"),
        ("rope_freqs", "rope_freqs"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.to_string()
}

/// Translates canonical llama.cpp DeepSeek-V4 tensor names to neutral model
/// parameter identities.
pub fn translate_v4_gguf_weight_name(name: &str) -> String {
    for (source, target, strip_weight) in [
        ("token_embd", "embed", false),
        ("output_norm", "norm", false),
        ("output", "head", false),
        ("output_hc_fn", "hc_head_fn", true),
        ("output_hc_base", "hc_head_base", true),
        ("output_hc_scale", "hc_head_scale", true),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            let mut translated = name.replacen(source, target, 1);
            if strip_weight && translated.ends_with(".weight") {
                translated.truncate(translated.len() - ".weight".len());
            }
            return translated;
        }
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_string();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_string();
    };
    let root = format!("layers.{layer}");
    for (source, target, strip_weight) in [
        ("attn_norm", "attn_norm", false),
        ("attn_sinks", "attn.attn_sink", true),
        ("attn_q_a", "attn.wq_a", false),
        ("attn_q_a_norm", "attn.q_norm", false),
        ("attn_q_b", "attn.wq_b", false),
        ("attn_kv", "attn.wkv", false),
        ("attn_kv_a_norm", "attn.kv_norm", false),
        ("attn_output_a", "attn.wo_a", false),
        ("attn_output_b", "attn.wo_b", false),
        ("hc_attn_fn", "hc_attn_fn", true),
        ("hc_attn_base", "hc_attn_base", true),
        ("hc_attn_scale", "hc_attn_scale", true),
        ("hc_ffn_fn", "hc_ffn_fn", true),
        ("hc_ffn_base", "hc_ffn_base", true),
        ("hc_ffn_scale", "hc_ffn_scale", true),
        ("attn_compressor_kv", "attn.compressor.wkv", false),
        ("attn_compressor_gate", "attn.compressor.wgate", false),
        ("attn_compressor_ape", "attn.compressor.ape", true),
        ("attn_compressor_norm", "attn.compressor.norm", false),
        ("indexer.proj", "attn.indexer.weights_proj", false),
        ("indexer.attn_q_b", "attn.indexer.wq_b", false),
        (
            "indexer_compressor_kv",
            "attn.indexer.compressor.wkv",
            false,
        ),
        (
            "indexer_compressor_gate",
            "attn.indexer.compressor.wgate",
            false,
        ),
        (
            "indexer_compressor_ape",
            "attn.indexer.compressor.ape",
            true,
        ),
        (
            "indexer_compressor_norm",
            "attn.indexer.compressor.norm",
            false,
        ),
        ("ffn_norm", "ffn_norm", false),
        ("ffn_gate_inp", "ffn.gate", false),
        ("exp_probs_b", "ffn.gate", false),
        ("ffn_gate_tid2eid", "ffn.gate.tid2eid", true),
        ("ffn_gate_shexp", "ffn.shared_experts.w1", false),
        ("ffn_down_shexp", "ffn.shared_experts.w2", false),
        ("ffn_up_shexp", "ffn.shared_experts.w3", false),
        ("ffn_gate_exps", "ffn.expert_banks.w1", false),
        ("ffn_down_exps", "ffn.expert_banks.w2", false),
        ("ffn_up_exps", "ffn.expert_banks.w3", false),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let mut translated = format!("{root}.{}", parameter.replacen(source, target, 1));
            if target.contains("expert_banks") && translated.ends_with(".scales") {
                translated.truncate(translated.len() - ".scales".len());
                translated.push_str(".scale");
            }
            if strip_weight && translated.ends_with(".weight") {
                translated.truncate(translated.len() - ".weight".len());
            }
            return translated;
        }
    }
    name.to_string()
}

/// Adds canonical fused runtime aliases for translated DeepSeek-V3 GGUF formats.
pub fn normalize_v3_weight_formats<V: Clone>(args: &V3Args, formats: &mut BTreeMap<String, V>) {
    for layer in 0..args.layer_schedule.len() {
        let root = format!("model.layers.{layer}.mlp");
        if let Some(format) = formats.get(&format!("{root}.experts.gate_proj")).cloned() {
            formats.insert(format!("{root}.experts.gate_up_proj"), format);
        }
    }
}

/// Adds canonical fused runtime aliases for translated DeepSeek-V4 GGUF formats.
pub fn normalize_v4_weight_formats<V: Clone>(args: &V4Args, formats: &mut BTreeMap<String, V>) {
    for layer in 0..args.attention_schedule.len() {
        let root = format!("layers.{layer}.ffn");
        if let Some(format) = formats
            .get(&format!("{root}.expert_banks.w1.weight"))
            .or_else(|| formats.get(&format!("{root}.expert_banks.w1")))
            .cloned()
        {
            formats.insert(format!("{root}.switch_mlp.gate_up_proj"), format);
        }
        if let Some(format) = formats
            .get(&format!("{root}.expert_banks.w2.weight"))
            .or_else(|| formats.get(&format!("{root}.expert_banks.w2")))
            .cloned()
        {
            formats.insert(format!("{root}.switch_mlp.down_proj"), format);
        }
    }
}

/// Derives a V3 configuration whose physical matrix formats reflect load-time quantization.
pub fn v3_load_time_quantization(
    args: &V3Args,
    quantization: WeightQuantization,
) -> Result<V3Args, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = args.clone();
    target.linear_format = quantization.into();
    target.linear_formats.clear();
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Derives a V4 configuration whose target and MTP expert formats reflect load-time quantization.
pub fn v4_load_time_quantization(
    args: &V4Args,
    quantization: WeightQuantization,
) -> Result<V4Args, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = args.clone();
    target.linear_format = quantization.into();
    target.linear_formats.clear();
    let target_layers = count(target.num_hidden_layers, "V4 target layer count")?;
    let prediction_layers = count(target.num_nextn_predict_layers, "V4 prediction layer count")?;
    let total = target_layers
        .checked_add(prediction_layers)
        .ok_or_else(|| "V4 load-time quantization layer count overflowed".to_string())?;
    for layer in 0..total {
        let (_, expert_root) = v4_expert_roots(&target, layer)?;
        for projection in ["gate_up_proj", "down_proj"] {
            target
                .linear_formats
                .insert(format!("{expert_root}.{projection}"), quantization.into());
        }
    }
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical checkpoint formats to a complete DeepSeek V3
/// configuration.
pub fn v3_with_checkpoint_formats(
    args: &V3Args,
    mut formats: BTreeMap<String, LinearFormat>,
) -> Result<V3Args, String> {
    normalize_v3_weight_formats(args, &mut formats);
    let mut target = args.clone();
    target.linear_formats = formats;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical checkpoint formats to a complete DeepSeek V4
/// configuration.
pub fn v4_with_checkpoint_formats(
    args: &V4Args,
    mut formats: BTreeMap<String, LinearFormat>,
) -> Result<V4Args, String> {
    normalize_v4_weight_formats(args, &mut formats);
    let mut target = args.clone();
    target.linear_formats = formats;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Builds the canonical llama.cpp DeepSeek2 GGUF plan. Fused and split MLA
/// KV-B storage are one global alternative layout so a checkpoint cannot mix
/// the two representations across layers.
pub fn v3_gguf_plan(args: &V3Args) -> Result<GgufCheckpointPlan, String> {
    args.validate().map_err(|error| error.to_string())?;
    if args.num_nextn_predict_layers != 0 {
        return Err("DeepSeek2 base GGUF does not embed MTP layers".into());
    }
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let heads = dimension(args.num_attention_heads, "attention heads")?;
    let nope = dimension(args.qk_nope_head_dim, "non-rotary head width")?;
    let rope = dimension(args.qk_rope_head_dim, "rotary head width")?;
    let value = dimension(args.v_head_dim, "value head width")?;
    let query_head = checked_add(nope, rope, "query head width")?;
    let kv_rank = dimension(args.kv_lora_rank, "KV rank")?;
    let mut tensors = vec![
        gguf_tensor(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf_tensor("output_norm.weight", vec![hidden], TensorOperation::Vector),
        gguf_tensor(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
    ];
    let mut fused = Vec::new();
    let mut fused_keys = Vec::new();
    let mut split = Vec::new();
    let mut split_keys = Vec::new();
    for (layer, policy) in args.layer_schedule.iter().copied().enumerate() {
        let root = format!("blk.{layer}");
        for (local, shape, operation) in [
            ("attn_norm.weight", vec![hidden], TensorOperation::Vector),
            ("ffn_norm.weight", vec![hidden], TensorOperation::Vector),
        ] {
            tensors.push(gguf_tensor(format!("{root}.{local}"), shape, operation));
        }
        if let Some(rank) = args.q_lora_rank {
            let rank = dimension(rank, "query rank")?;
            for (local, shape, operation) in [
                (
                    "attn_q_a.weight",
                    vec![rank, hidden],
                    TensorOperation::Matrix,
                ),
                ("attn_q_a_norm.weight", vec![rank], TensorOperation::Vector),
                (
                    "attn_q_b.weight",
                    vec![checked_mul(heads, query_head, "query width")?, rank],
                    TensorOperation::Matrix,
                ),
            ] {
                tensors.push(gguf_tensor(format!("{root}.{local}"), shape, operation));
            }
        } else {
            tensors.push(gguf_tensor(
                format!("{root}.attn_q.weight"),
                vec![checked_mul(heads, query_head, "query width")?, hidden],
                TensorOperation::Matrix,
            ));
        }
        tensors.extend([
            gguf_tensor(
                format!("{root}.attn_kv_a_mqa.weight"),
                vec![checked_add(kv_rank, rope, "KV-A width")?, hidden],
                TensorOperation::Matrix,
            ),
            gguf_tensor(
                format!("{root}.attn_kv_a_norm.weight"),
                vec![kv_rank],
                TensorOperation::Vector,
            ),
            gguf_tensor(
                format!("{root}.attn_output.weight"),
                vec![hidden, checked_mul(heads, value, "output width")?],
                TensorOperation::Matrix,
            ),
        ]);
        let fused_key = format!("{root}.attn_kv_b.weight");
        fused_keys.push(fused_key.clone());
        fused.push(gguf_tensor(
            fused_key,
            vec![
                checked_mul(
                    heads,
                    checked_add(nope, value, "KV-B head width")?,
                    "KV-B width",
                )?,
                kv_rank,
            ],
            TensorOperation::Matrix,
        ));
        for (local, shape) in [
            ("attn_k_b.weight", vec![heads, kv_rank, nope]),
            ("attn_v_b.weight", vec![heads, value, kv_rank]),
        ] {
            let key = format!("{root}.{local}");
            split_keys.push(key.clone());
            split.push(gguf_tensor(key, shape, TensorOperation::Matrix));
        }
        if policy == LayerPolicy::SparseMoe {
            let experts = dimension(args.n_routed_experts, "expert count")?;
            let intermediate = dimension(args.moe_intermediate_size, "expert width")?;
            let shared = checked_mul(
                intermediate,
                dimension(args.n_shared_experts, "shared expert count")?,
                "shared expert width",
            )?;
            for (local, shape, operation) in [
                (
                    "ffn_gate_inp.weight",
                    vec![experts, hidden],
                    TensorOperation::Matrix,
                ),
                ("exp_probs_b.bias", vec![experts], TensorOperation::Vector),
                (
                    "ffn_gate_shexp.weight",
                    vec![shared, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "ffn_up_shexp.weight",
                    vec![shared, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "ffn_down_shexp.weight",
                    vec![hidden, shared],
                    TensorOperation::Matrix,
                ),
                (
                    "ffn_gate_exps.weight",
                    vec![experts, intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "ffn_up_exps.weight",
                    vec![experts, intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "ffn_down_exps.weight",
                    vec![experts, hidden, intermediate],
                    TensorOperation::Matrix,
                ),
            ] {
                tensors.push(gguf_tensor(format!("{root}.{local}"), shape, operation));
            }
        } else {
            let intermediate = dimension(args.intermediate_size, "dense MLP width")?;
            for (local, shape) in [
                ("ffn_gate.weight", vec![intermediate, hidden]),
                ("ffn_up.weight", vec![intermediate, hidden]),
                ("ffn_down.weight", vec![hidden, intermediate]),
            ] {
                tensors.push(gguf_tensor(
                    format!("{root}.{local}"),
                    shape,
                    TensorOperation::Matrix,
                ));
            }
        }
    }
    let layouts = vec![AlternativeLayoutGroup {
        id: "DeepSeek2 MLA KV-B storage".into(),
        required: true,
        variants: vec![
            LayoutVariant {
                id: "fused".into(),
                tensors: fused,
                discriminator_keys: fused_keys,
            },
            LayoutVariant {
                id: "split".into(),
                tensors: split,
                discriminator_keys: split_keys,
            },
        ],
    }];
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.push("rope_freqs.".into());
    GgufCheckpointPlan::new("DeepSeek2 GGUF", tensors, layouts, policy)
        .map_err(|error| error.to_string())
}

/// Builds the canonical fused KV-B loading recipe for one DeepSeek2 GGUF
/// layer. Split K/V storage is transposed and concatenated into the exact
/// parameter identity consumed by neutral MLA.
pub fn v3_gguf_kv_b_recipe(
    args: &V3Args,
    layer: usize,
    split: bool,
) -> Result<DerivedWeightRecipe, String> {
    args.validate().map_err(|error| error.to_string())?;
    let layers = count(args.num_hidden_layers, "target layer count")?;
    if layer >= layers {
        return Err(format!(
            "DeepSeek2 layer {layer} is outside {layers} layers"
        ));
    }
    let root = format!("blk.{layer}");
    if !split {
        return Ok(DerivedWeightRecipe::source(
            format!("{root}.attn_kv_b.weight"),
            TensorSelection::Full,
        ));
    }
    let heads = dimension(args.num_attention_heads, "attention heads")?;
    let rank = dimension(args.kv_lora_rank, "KV rank")?;
    let width = checked_mul(
        heads,
        checked_add(
            dimension(args.qk_nope_head_dim, "non-rotary width")?,
            dimension(args.v_head_dim, "value width")?,
            "KV-B head width",
        )?,
        "KV-B width",
    )?;
    Ok(DerivedWeightRecipe::Reshape {
        input: Box::new(DerivedWeightRecipe::Concatenate {
            axis: 1,
            inputs: vec![
                DerivedWeightRecipe::Transpose {
                    input: Box::new(DerivedWeightRecipe::source(
                        format!("{root}.attn_k_b.weight"),
                        TensorSelection::Full,
                    )),
                    axes: vec![0, 2, 1],
                },
                DerivedWeightRecipe::source(
                    format!("{root}.attn_v_b.weight"),
                    TensorSelection::Full,
                ),
            ],
        }),
        shape: vec![width, rank],
    })
}

/// Returns the complete neutral recipe catalog for one DeepSeek V3 unit.
pub fn v3_unit_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &V3Args,
    layer: usize,
    include_experts: bool,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let mut recipes = BTreeMap::new();
    let physical = format!("blk.{layer}");
    let logical = format!("model.layers.{layer}.self_attn");
    if catalog
        .tensor_metadata(&format!("{logical}.k_b_proj.weight"))
        .is_ok()
    {
        let heads = dimension(args.num_attention_heads, "attention heads")?;
        let rank = dimension(args.kv_lora_rank, "KV rank")?;
        let width = checked_mul(
            heads,
            checked_add(
                dimension(args.qk_nope_head_dim, "non-rotary width")?,
                dimension(args.v_head_dim, "value width")?,
                "KV-B head width",
            )?,
            "KV-B width",
        )?;
        recipes.insert(
            format!("{logical}.kv_b_proj.weight"),
            DerivedWeightRecipe::Reshape {
                input: Box::new(DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        DerivedWeightRecipe::Transpose {
                            input: Box::new(DerivedWeightRecipe::source(
                                format!("{logical}.k_b_proj.weight"),
                                TensorSelection::Full,
                            )),
                            axes: vec![0, 2, 1],
                        },
                        DerivedWeightRecipe::source(
                            format!("{logical}.v_b_proj.weight"),
                            TensorSelection::Full,
                        ),
                    ],
                }),
                shape: vec![width, rank],
            },
        );
    } else if catalog
        .tensor_metadata(&format!("{physical}.attn_k_b.weight"))
        .is_ok()
    {
        recipes.insert(
            format!("{logical}.kv_b_proj.weight"),
            v3_gguf_kv_b_recipe(args, layer, true)?,
        );
    }

    let target = count(args.num_hidden_layers, "target layer count")?;
    if include_experts
        && (layer >= target || args.layer_schedule.get(layer) == Some(&LayerPolicy::SparseMoe))
    {
        let expert = v3_expert_recipes(catalog, args, layer)?;
        recipes.insert(expert.target_gate_up, expert.gate_up);
        recipes.insert(expert.target_down, expert.down);
    }
    Ok(recipes)
}

/// Resolves independent, split-bank, or fused V3 experts into the one packed
/// gate/up and down representation consumed by the neutral expert operator.
pub fn v3_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &V3Args,
    layer: usize,
) -> Result<GatedProductExpertRecipes, String> {
    let total = count(
        args.num_hidden_layers + args.num_nextn_predict_layers,
        "V3 layer count",
    )?;
    if layer >= total {
        return Err(format!("V3 expert layer {layer} is outside {total} layers"));
    }
    let count = dimension(args.n_routed_experts, "expert count")?;
    let root = format!("model.layers.{layer}.mlp.experts");
    resolve_gated_product_expert_recipes(
        catalog,
        &GatedProductExpertLayoutNames {
            target_gate_up: format!("{root}.gate_up_proj"),
            target_down: format!("{root}.down_proj"),
            packed_gate_up: format!("{root}.gate_up_proj"),
            packed_down: format!("{root}.down_proj"),
            separate_gate: format!("{root}.gate_proj"),
            separate_up: format!("{root}.up_proj"),
            separate_down: format!("{root}.down_proj"),
            independent: (0..count)
                .map(|expert| IndependentGatedProductExpertNames {
                    gate: format!("{root}.{expert}.gate_proj.weight"),
                    up: format!("{root}.{expert}.up_proj.weight"),
                    down: format!("{root}.{expert}.down_proj.weight"),
                })
                .collect(),
        },
    )
    .map_err(|error| error.to_string())
}

/// Resolves SafeTensors independent experts or GGUF projection banks into the
/// packed V4 expert representation consumed by target, MTP, and DSpark blocks.
pub fn v4_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &V4Args,
    layer: usize,
) -> Result<GatedProductExpertRecipes, String> {
    let count = dimension(args.n_routed_experts, "expert count")?;
    let (physical, target) = v4_expert_roots(args, layer)?;
    resolve_gated_product_expert_recipes(
        catalog,
        &GatedProductExpertLayoutNames {
            target_gate_up: format!("{target}.gate_up_proj"),
            target_down: format!("{target}.down_proj"),
            packed_gate_up: format!("{target}.gate_up_proj"),
            packed_down: format!("{target}.down_proj"),
            separate_gate: format!("{physical}.expert_banks.w1"),
            separate_up: format!("{physical}.expert_banks.w3"),
            separate_down: format!("{physical}.expert_banks.w2"),
            independent: (0..count)
                .map(|expert| IndependentGatedProductExpertNames {
                    gate: format!("{physical}.experts.{expert}.w1.weight"),
                    up: format!("{physical}.experts.{expert}.w3.weight"),
                    down: format!("{physical}.experts.{expert}.w2.weight"),
                })
                .collect(),
        },
    )
    .map_err(|error| error.to_string())
}

fn v4_expert_roots(args: &V4Args, layer: usize) -> Result<(String, String), String> {
    let target_layers = count(args.num_hidden_layers, "V4 target layer count")?;
    let prediction_layers = count(args.num_nextn_predict_layers, "V4 prediction layer count")?;
    let total = target_layers
        .checked_add(prediction_layers)
        .ok_or_else(|| "V4 expert layer count overflowed".to_string())?;
    if layer >= total {
        return Err(format!("V4 expert layer {layer} is outside {total} layers"));
    }
    let block = if layer < target_layers {
        format!("layers.{layer}")
    } else {
        format!("mtp.{}", layer - target_layers)
    };
    let physical = format!("{block}.ffn");
    let target = format!("{physical}.switch_mlp");
    Ok((physical, target))
}

/// Derives one independently resident expert from a canonical DeepSeek bank.
///
/// When `intermediate` is present, its semantic range is applied independently
/// to the gate and up segments and to the down-projection input axis. This is
/// the recipe counterpart of the architecture's `PartitionedSegments` plan;
/// concrete backends only materialize the recipes returned here.
pub fn expert_unit_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    bank: &GatedProductExpertRecipes,
    expert: usize,
    intermediate: Option<Range<usize>>,
) -> Result<ExpertUnitRecipes, String> {
    let expert_selection = TensorSelection::Range {
        axis: 0,
        start: expert,
        end: expert
            .checked_add(1)
            .ok_or_else(|| "DeepSeek expert index overflowed".to_string())?,
    };
    let mut gate_up = bank
        .gate_up
        .select_bounded(catalog, expert_selection.clone())
        .map_err(|error| error.to_string())?;
    let mut down = bank
        .down
        .select_bounded(catalog, expert_selection)
        .map_err(|error| error.to_string())?;

    if let Some(intermediate) = intermediate {
        let metadata = gate_up.infer(catalog).map_err(|error| error.to_string())?;
        let fused = *metadata
            .shape()
            .get(1)
            .ok_or_else(|| "DeepSeek expert gate/up recipe is not rank three".to_string())?;
        if fused % 2 != 0 || intermediate.start >= intermediate.end || intermediate.end > fused / 2
        {
            return Err(format!(
                "DeepSeek expert intermediate range {intermediate:?} is outside 0..{}",
                fused / 2
            ));
        }
        let width = fused / 2;
        gate_up = DerivedWeightRecipe::Concatenate {
            axis: 1,
            inputs: vec![
                gate_up
                    .clone()
                    .select_bounded(
                        catalog,
                        TensorSelection::Range {
                            axis: 1,
                            start: intermediate.start,
                            end: intermediate.end,
                        },
                    )
                    .map_err(|error| error.to_string())?,
                gate_up
                    .select_bounded(
                        catalog,
                        TensorSelection::Range {
                            axis: 1,
                            start: width + intermediate.start,
                            end: width + intermediate.end,
                        },
                    )
                    .map_err(|error| error.to_string())?,
            ],
        };
        down = down
            .select_bounded(
                catalog,
                TensorSelection::Range {
                    axis: 2,
                    start: intermediate.start,
                    end: intermediate.end,
                },
            )
            .map_err(|error| error.to_string())?;
    }

    Ok(ExpertUnitRecipes { gate_up, down })
}

/// Builds the complete V3/R1 schedule for independently resident routed experts.
pub fn v3_expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &V3Args,
    intermediate: Option<Range<usize>>,
) -> Result<crate::ExpertResidencyCatalog, String> {
    let target = count(args.num_hidden_layers, "V3 target layer count")?;
    let prediction = count(args.num_nextn_predict_layers, "V3 prediction layer count")?;
    let total = target
        .checked_add(prediction)
        .ok_or_else(|| "V3 expert residency layer count overflowed".to_string())?;
    let experts = dimension(args.n_routed_experts, "V3 expert count")?;
    let mut units = Vec::new();
    for layer in 0..total {
        if layer < target && args.layer_schedule.get(layer) != Some(&LayerPolicy::SparseMoe) {
            continue;
        }
        let (owner, owner_unit) = if layer < target {
            ("target".to_string(), layer)
        } else {
            (format!("mtp.{}", layer - target), 0)
        };
        let unit_path = format!("model.layers.{layer}");
        let bank = v3_expert_recipes(catalog, args, layer)?;
        append_expert_residency_units(
            &mut units,
            catalog,
            layer,
            &owner,
            owner_unit,
            &unit_path,
            experts,
            &bank,
            intermediate.clone(),
        )?;
    }
    crate::ExpertResidencyCatalog::new(units)
        .and_then(|residency| residency.with_inferred_byte_geometry(catalog))
        .map_err(|error| error.to_string())
}

/// Builds the complete V4 schedule for independently resident routed experts.
pub fn v4_expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &V4Args,
    intermediate: Option<Range<usize>>,
) -> Result<crate::ExpertResidencyCatalog, String> {
    let target = count(args.num_hidden_layers, "V4 target layer count")?;
    let prediction = count(args.num_nextn_predict_layers, "V4 prediction layer count")?;
    let total = target
        .checked_add(prediction)
        .ok_or_else(|| "V4 expert residency layer count overflowed".to_string())?;
    let experts = dimension(args.n_routed_experts, "V4 expert count")?;
    let capacity = total
        .checked_mul(experts)
        .ok_or_else(|| "V4 expert residency catalog size overflowed".to_string())?;
    let mut units = Vec::with_capacity(capacity);
    for layer in 0..total {
        let (owner, owner_unit, unit_path) = if layer < target {
            ("target".to_string(), layer, format!("layers.{layer}"))
        } else {
            let depth = layer - target;
            (format!("mtp.{depth}"), 0, format!("mtp.{depth}"))
        };
        let bank = v4_expert_recipes(catalog, args, layer)?;
        append_expert_residency_units(
            &mut units,
            catalog,
            layer,
            &owner,
            owner_unit,
            &unit_path,
            experts,
            &bank,
            intermediate.clone(),
        )?;
    }
    crate::ExpertResidencyCatalog::new(units)
        .and_then(|residency| residency.with_inferred_byte_geometry(catalog))
        .map_err(|error| error.to_string())
}

#[allow(clippy::too_many_arguments)]
fn append_expert_residency_units<C: RecipeCatalog + ?Sized>(
    units: &mut Vec<crate::ExpertResidencyUnit>,
    catalog: &C,
    identity_layer: usize,
    owner_group: &str,
    owner_unit: usize,
    unit_path: &str,
    experts: usize,
    bank: &GatedProductExpertRecipes,
    intermediate: Option<Range<usize>>,
) -> Result<(), String> {
    let owner_group =
        eredu_runtime::ExecutionGroupId::new(owner_group).map_err(|error| error.to_string())?;
    for expert in 0..experts {
        let recipes = expert_unit_recipes(catalog, bank, expert, intermediate.clone())?;
        let parameters = [
            ("gate_up_proj", bank.target_gate_up.clone(), recipes.gate_up),
            ("down_proj", bank.target_down.clone(), recipes.down),
        ]
        .into_iter()
        .map(|(binding, target, recipe)| {
            let role = match binding {
                "gate_up_proj" => crate::ExpertParameterRole::quantizable_projection(
                    "gate_up_proj_scales",
                    "gate_up_proj_biases",
                ),
                "down_proj" => crate::ExpertParameterRole::quantizable_projection(
                    "down_proj_scales",
                    "down_proj_biases",
                ),
                _ => crate::ExpertParameterRole::Preserved,
            };
            crate::ExpertParameterRecipe::new(binding, target, recipe, role)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
        units.push(
            crate::ExpertResidencyUnit::new(
                eredu_runtime::ParameterBankKey::new(identity_layer, expert),
                owner_group.clone(),
                owner_unit,
                unit_path,
                crate::ExpertResidencyDistribution::ExpertParallel,
                parameters,
            )
            .map_err(|error| error.to_string())?,
        );
    }
    Ok(())
}

/// Builds the canonical DeepSeek-V3/R1 SafeTensors catalog plan.
pub fn v3_safetensors_plan(
    args: &V3Args,
    allow_packed_experts: bool,
) -> Result<SafetensorsCheckpointPlan, String> {
    args.validate().map_err(|error| error.to_string())?;
    let mut tensors = Vec::new();
    let mut groups = Vec::new();
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    add_v3_tensor(
        &mut tensors,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
        TensorOperation::Matrix,
        LinearFormat::Dense,
    )?;
    add_v3_tensor(
        &mut tensors,
        "model.norm.weight",
        vec![hidden],
        TensorOperation::Vector,
        LinearFormat::Dense,
    )?;
    add_v3_tensor(
        &mut tensors,
        "lm_head.weight",
        vec![vocab, hidden],
        TensorOperation::Matrix,
        LinearFormat::Dense,
    )?;
    for (layer, policy) in args.layer_schedule.iter().copied().enumerate() {
        add_v3_layer(
            args,
            layer,
            policy,
            &mut tensors,
            &mut groups,
            allow_packed_experts,
        )?;
    }
    for prediction in 0..count(args.num_nextn_predict_layers, "MTP layer count")? {
        let layer = dimension(args.num_hidden_layers, "layer count")?
            .checked_add(prediction)
            .ok_or_else(|| "MTP layer index overflowed".to_string())?;
        add_v3_layer(
            args,
            layer,
            LayerPolicy::SparseMoe,
            &mut tensors,
            &mut groups,
            allow_packed_experts,
        )?;
        let root = format!("model.layers.{layer}");
        for (name, shape, operation) in [
            ("enorm.weight", vec![hidden], TensorOperation::Vector),
            ("hnorm.weight", vec![hidden], TensorOperation::Vector),
            (
                "eh_proj.weight",
                vec![hidden, checked_mul(2, hidden, "MTP input width")?],
                TensorOperation::Matrix,
            ),
            (
                "shared_head.norm.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            (
                "shared_head.head.weight",
                vec![vocab, hidden],
                TensorOperation::Matrix,
            ),
        ] {
            let full = format!("{root}.{name}");
            add_v3_tensor(
                &mut tensors,
                &full,
                shape,
                operation,
                v3_format(args, &full, operation),
            )?;
        }
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.push("rope_freqs.".into());
    policy.allowed_suffixes.push("rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("DeepSeek-V3 SafeTensors", tensors, groups, policy)
        .map_err(|error| error.to_string())
}

fn add_v3_layer(
    args: &V3Args,
    layer: usize,
    policy: LayerPolicy,
    tensors: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
    allow_packed_experts: bool,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let heads = dimension(args.num_attention_heads, "attention heads")?;
    let nope = dimension(args.qk_nope_head_dim, "non-rotary head width")?;
    let rope = dimension(args.qk_rope_head_dim, "rotary head width")?;
    let value = dimension(args.v_head_dim, "value head width")?;
    let kv_rank = dimension(args.kv_lora_rank, "KV rank")?;
    let query_head = nope
        .checked_add(rope)
        .ok_or_else(|| "query head width overflowed".to_string())?;
    let root = format!("model.layers.{layer}");
    let mut entries = vec![
        (
            "input_layernorm.weight".to_string(),
            vec![hidden],
            TensorOperation::Vector,
        ),
        (
            "post_attention_layernorm.weight".to_string(),
            vec![hidden],
            TensorOperation::Vector,
        ),
    ];
    if let Some(rank) = args.q_lora_rank {
        let rank = dimension(rank, "query rank")?;
        entries.extend([
            (
                "self_attn.q_a_proj.weight".into(),
                vec![rank, hidden],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.q_a_layernorm.weight".into(),
                vec![rank],
                TensorOperation::Vector,
            ),
            (
                "self_attn.q_b_proj.weight".into(),
                vec![checked_mul(heads, query_head, "query width")?, rank],
                TensorOperation::Matrix,
            ),
        ]);
    } else {
        entries.push((
            "self_attn.q_proj.weight".into(),
            vec![checked_mul(heads, query_head, "query width")?, hidden],
            TensorOperation::Matrix,
        ));
    }
    entries.extend([
        (
            "self_attn.kv_a_proj_with_mqa.weight".into(),
            vec![kv_rank + rope, hidden],
            TensorOperation::Matrix,
        ),
        (
            "self_attn.kv_a_layernorm.weight".into(),
            vec![kv_rank],
            TensorOperation::Vector,
        ),
        (
            "self_attn.kv_b_proj.weight".into(),
            vec![checked_mul(heads, nope + value, "KV-B width")?, kv_rank],
            TensorOperation::Matrix,
        ),
        (
            "self_attn.o_proj.weight".into(),
            vec![hidden, checked_mul(heads, value, "attention output")?],
            TensorOperation::Matrix,
        ),
    ]);
    if policy == LayerPolicy::SparseMoe {
        let experts = dimension(args.n_routed_experts, "expert count")?;
        let intermediate = dimension(args.moe_intermediate_size, "expert width")?;
        let shared = checked_mul(
            intermediate,
            dimension(args.n_shared_experts, "shared expert count")?,
            "shared expert width",
        )?;
        entries.extend([
            (
                "mlp.gate.weight".into(),
                vec![experts, hidden],
                TensorOperation::Matrix,
            ),
            (
                "mlp.gate.e_score_correction_bias".into(),
                vec![experts],
                TensorOperation::Vector,
            ),
            (
                "mlp.shared_experts.gate_proj.weight".into(),
                vec![shared, hidden],
                TensorOperation::Matrix,
            ),
            (
                "mlp.shared_experts.up_proj.weight".into(),
                vec![shared, hidden],
                TensorOperation::Matrix,
            ),
            (
                "mlp.shared_experts.down_proj.weight".into(),
                vec![hidden, shared],
                TensorOperation::Matrix,
            ),
        ]);
        for (projection, split, packed) in [
            (
                "gate_proj",
                vec![intermediate, hidden],
                vec![experts, intermediate, hidden],
            ),
            (
                "up_proj",
                vec![intermediate, hidden],
                vec![experts, intermediate, hidden],
            ),
            (
                "down_proj",
                vec![hidden, intermediate],
                vec![experts, hidden, intermediate],
            ),
        ] {
            groups.push(v3_expert_group(
                args,
                &format!("{root}.mlp.experts"),
                projection,
                experts,
                split,
                packed,
                allow_packed_experts,
            )?);
        }
    } else {
        let intermediate = dimension(args.intermediate_size, "dense MLP width")?;
        for projection in ["gate_proj", "up_proj"] {
            entries.push((
                format!("mlp.{projection}.weight"),
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ));
        }
        entries.push((
            "mlp.down_proj.weight".into(),
            vec![hidden, intermediate],
            TensorOperation::Matrix,
        ));
    }
    for (local, shape, operation) in entries {
        let name = format!("{root}.{local}");
        add_v3_tensor(
            tensors,
            &name,
            shape,
            operation,
            v3_format(args, &name, operation),
        )?;
    }
    Ok(())
}

fn v3_expert_group(
    args: &V3Args,
    root: &str,
    projection: &str,
    experts: usize,
    split_shape: Vec<usize>,
    packed_shape: Vec<usize>,
    allow_packed: bool,
) -> Result<AlternativeLayoutGroup<SafetensorsTensorConstraint>, String> {
    let mut split = Vec::new();
    let mut split_keys = Vec::new();
    for expert in 0..experts {
        let name = format!("{root}.{expert}.{projection}.weight");
        split_keys.push(name.clone());
        split.extend(v3_matrix_constraints(
            &name,
            split_shape.clone(),
            args.linear_format_for(&name),
        )?);
    }
    let mut variants = vec![LayoutVariant {
        id: "split".into(),
        tensors: split,
        discriminator_keys: split_keys,
    }];
    if allow_packed {
        let name = format!("{root}.{projection}");
        variants.push(LayoutVariant {
            id: "packed".into(),
            discriminator_keys: vec![name.clone()],
            tensors: v3_matrix_constraints(&name, packed_shape, args.linear_format_for(&name))?,
        });
    }
    Ok(AlternativeLayoutGroup {
        id: format!("{root} {projection} storage"),
        required: true,
        variants,
    })
}

fn v3_format(args: &V3Args, name: &str, operation: TensorOperation) -> LinearFormat {
    if operation != TensorOperation::Matrix
        || name.ends_with(".mlp.gate.weight")
        || (matches!(args.linear_format_for(name), LinearFormat::E4M3BlockFp8(_))
            && matches!(name, "model.embed_tokens.weight" | "lm_head.weight"))
    {
        LinearFormat::Dense
    } else {
        args.linear_format_for(name)
    }
}

fn add_v3_tensor(
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
    operation: TensorOperation,
    format: LinearFormat,
) -> Result<(), String> {
    if operation == TensorOperation::Matrix {
        output.extend(v3_matrix_constraints(name, shape, format)?);
    } else {
        output.push(SafetensorsTensorConstraint::required(
            name,
            shape,
            StoredDtypeConstraint::Floating,
        ));
    }
    Ok(())
}

fn v3_matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    format: LinearFormat,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let scale = matches!(format, LinearFormat::E4M3BlockFp8(_)).then(|| {
        let prefix = name.strip_suffix(".weight").unwrap_or(name);
        MatrixScaleNames {
            key: format!("{prefix}.weight_scale_inv"),
            aliases: Vec::new(),
        }
    });
    matrix_for_linear_format(name, std::iter::empty::<String>(), shape, format, scale)
        .map_err(|error| error.to_string())
}

#[derive(Debug, Clone)]
struct V4TensorSpec {
    name: String,
    shape: Vec<usize>,
    operation: TensorOperation,
    format: LinearFormat,
}

/// Builds the canonical DeepSeek-V4 SafeTensors catalog plan, including
/// native UE8M0 block-FP8 companions and typed integer hash tables.
pub fn v4_safetensors_plan(args: &V4Args) -> Result<SafetensorsCheckpointPlan, String> {
    args.validate().map_err(|error| error.to_string())?;
    let mut specs = v4_target_specs(args)?;
    append_v4_draft_specs(&mut specs, args)?;
    let mut tensors = Vec::new();
    for spec in specs {
        match spec.operation {
            TensorOperation::Matrix | TensorOperation::MxFp4Matrix => {
                tensors.extend(v4_matrix_constraints(&spec.name, spec.shape, spec.format)?)
            }
            TensorOperation::I32 => tensors.push(SafetensorsTensorConstraint::required(
                spec.name,
                spec.shape,
                StoredDtypeConstraint::Exact(StoredDtype::I32),
            )),
            TensorOperation::Vector | TensorOperation::Dense => {
                tensors.push(SafetensorsTensorConstraint::required(
                    spec.name,
                    spec.shape,
                    StoredDtypeConstraint::Floating,
                ));
            }
        }
    }
    let mut policy = CatalogPolicy::non_strict();
    policy.allowed_suffixes.push("rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("DeepSeek-V4 SafeTensors", tensors, Vec::new(), policy)
        .map_err(|error| error.to_string())
}

/// Builds the strict llama.cpp DeepSeek-V4 base-model GGUF plan. Routed
/// experts are packed MXFP4 projection banks in this representation.
pub fn v4_gguf_plan(args: &V4Args) -> Result<GgufCheckpointPlan, String> {
    args.validate().map_err(|error| error.to_string())?;
    if args.num_nextn_predict_layers != 0 {
        return Err("DeepSeek-V4 MTP GGUF files are companion artifacts, not base models".into());
    }
    let specs = v4_target_specs(args)?;
    let mut tensors = Vec::new();
    for spec in specs
        .into_iter()
        .filter(|spec| !spec.name.contains(".ffn.experts."))
    {
        let physical = v4_gguf_physical_name(&spec.name)?;
        tensors.push(gguf_tensor(physical, spec.shape, spec.operation));
    }
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let intermediate = dimension(args.moe_intermediate_size, "expert width")?;
    let experts = dimension(args.n_routed_experts, "expert count")?;
    for layer in 0..dimension(args.num_hidden_layers, "target layers")? {
        let root = format!("blk.{layer}");
        for (local, shape) in [
            ("ffn_gate_exps.weight", vec![experts, intermediate, hidden]),
            ("ffn_down_exps.weight", vec![experts, hidden, intermediate]),
            ("ffn_up_exps.weight", vec![experts, intermediate, hidden]),
        ] {
            tensors.push(gguf_tensor(
                format!("{root}.{local}"),
                shape,
                TensorOperation::MxFp4Matrix,
            ));
        }
    }
    GgufCheckpointPlan::new(
        "DeepSeek-V4 GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

fn v4_gguf_physical_name(name: &str) -> Result<String, String> {
    for (logical, physical) in [
        ("embed.weight", "token_embd.weight"),
        ("norm.weight", "output_norm.weight"),
        ("head.weight", "output.weight"),
        ("hc_head_fn", "output_hc_fn.weight"),
        ("hc_head_base", "output_hc_base.weight"),
        ("hc_head_scale", "output_hc_scale.weight"),
    ] {
        if name == logical {
            return Ok(physical.into());
        }
    }
    let rest = name
        .strip_prefix("layers.")
        .ok_or_else(|| format!("no DeepSeek-V4 GGUF name for {name:?}"))?;
    let (layer, local) = rest
        .split_once('.')
        .ok_or_else(|| format!("invalid DeepSeek-V4 layer name {name:?}"))?;
    let physical = match local {
        "attn.wq_a.weight" => "attn_q_a.weight",
        "attn.q_norm.weight" => "attn_q_a_norm.weight",
        "attn.wq_b.weight" => "attn_q_b.weight",
        "attn.wkv.weight" => "attn_kv.weight",
        "attn.kv_norm.weight" => "attn_kv_a_norm.weight",
        "attn.wo_a.weight" => "attn_output_a.weight",
        "attn.wo_b.weight" => "attn_output_b.weight",
        "attn.attn_sink" => "attn_sinks.weight",
        "attn_norm.weight" => "attn_norm.weight",
        "ffn_norm.weight" => "ffn_norm.weight",
        "hc_attn_fn" => "hc_attn_fn.weight",
        "hc_attn_base" => "hc_attn_base.weight",
        "hc_attn_scale" => "hc_attn_scale.weight",
        "hc_ffn_fn" => "hc_ffn_fn.weight",
        "hc_ffn_base" => "hc_ffn_base.weight",
        "hc_ffn_scale" => "hc_ffn_scale.weight",
        "ffn.gate.weight" => "ffn_gate_inp.weight",
        "ffn.gate.tid2eid" => "ffn_gate_tid2eid.weight",
        "ffn.gate.bias" => "exp_probs_b.bias",
        "ffn.shared_experts.w1.weight" => "ffn_gate_shexp.weight",
        "ffn.shared_experts.w2.weight" => "ffn_down_shexp.weight",
        "ffn.shared_experts.w3.weight" => "ffn_up_shexp.weight",
        "attn.compressor.wkv.weight" => "attn_compressor_kv.weight",
        "attn.compressor.wgate.weight" => "attn_compressor_gate.weight",
        "attn.compressor.ape" => "attn_compressor_ape.weight",
        "attn.compressor.norm.weight" => "attn_compressor_norm.weight",
        "attn.indexer.wq_b.weight" => "indexer.attn_q_b.weight",
        "attn.indexer.weights_proj.weight" => "indexer.proj.weight",
        "attn.indexer.compressor.wkv.weight" => "indexer_compressor_kv.weight",
        "attn.indexer.compressor.wgate.weight" => "indexer_compressor_gate.weight",
        "attn.indexer.compressor.ape" => "indexer_compressor_ape.weight",
        "attn.indexer.compressor.norm.weight" => "indexer_compressor_norm.weight",
        _ => return Err(format!("no DeepSeek-V4 GGUF name for {name:?}")),
    };
    Ok(format!("blk.{layer}.{physical}"))
}

fn v4_target_specs(args: &V4Args) -> Result<Vec<V4TensorSpec>, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let streams = dimension(args.hc_mult, "hyper streams")?;
    let stream_hidden = checked_mul(streams, hidden, "hyper width")?;
    let mix = checked_mul(
        checked_add(streams, 2, "hyper mixing input")?,
        streams,
        "hyper mixing width",
    )?;
    let mut specs = vec![
        v4_spec(
            "embed.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
            LinearFormat::Dense,
        ),
        v4_spec(
            "norm.weight",
            vec![hidden],
            TensorOperation::Vector,
            LinearFormat::Dense,
        ),
        v4_spec(
            "head.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
            args.linear_format,
        ),
        v4_spec(
            "hc_head_fn",
            vec![streams, stream_hidden],
            TensorOperation::Matrix,
            LinearFormat::Dense,
        ),
        v4_spec(
            "hc_head_base",
            vec![streams],
            TensorOperation::Vector,
            LinearFormat::Dense,
        ),
        v4_spec(
            "hc_head_scale",
            vec![1],
            TensorOperation::Vector,
            LinearFormat::Dense,
        ),
    ];
    let heads = dimension(args.num_attention_heads, "attention heads")?;
    let head_dim = dimension(args.head_dim, "head width")?;
    let query_rank = dimension(args.q_lora_rank, "query rank")?;
    let output_rank = dimension(args.o_lora_rank, "output rank")?;
    let output_groups = dimension(args.o_groups, "output groups")?;
    let experts = dimension(args.n_routed_experts, "expert count")?;
    let routed = dimension(args.num_experts_per_tok, "routes per token")?;
    let shared = checked_mul(
        dimension(args.moe_intermediate_size, "expert width")?,
        dimension(args.n_shared_experts, "shared expert count")?,
        "shared expert width",
    )?;
    let index_heads = dimension(args.index_n_heads, "index heads")?;
    let index_dim = dimension(args.index_head_dim, "index head width")?;
    let query_width = checked_mul(heads, head_dim, "query width")?;
    let output_a_rows = checked_mul(output_groups, output_rank, "output A rows")?;
    let output_a_columns = query_width / output_groups;
    for layer in 0..dimension(args.num_hidden_layers, "target layers")? {
        let root = format!("layers.{layer}");
        for (local, shape, operation) in [
            (
                "attn.wq_a.weight",
                vec![query_rank, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn.q_norm.weight",
                vec![query_rank],
                TensorOperation::Vector,
            ),
            (
                "attn.wq_b.weight",
                vec![query_width, query_rank],
                TensorOperation::Matrix,
            ),
            (
                "attn.wkv.weight",
                vec![head_dim, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn.kv_norm.weight",
                vec![head_dim],
                TensorOperation::Vector,
            ),
            (
                "attn.wo_a.weight",
                vec![output_a_rows, output_a_columns],
                TensorOperation::Matrix,
            ),
            (
                "attn.wo_b.weight",
                vec![hidden, output_a_rows],
                TensorOperation::Matrix,
            ),
            ("attn.attn_sink", vec![heads], TensorOperation::Vector),
            ("attn_norm.weight", vec![hidden], TensorOperation::Vector),
            ("ffn_norm.weight", vec![hidden], TensorOperation::Vector),
            (
                "hc_attn_fn",
                vec![mix, stream_hidden],
                TensorOperation::Matrix,
            ),
            ("hc_attn_base", vec![mix], TensorOperation::Vector),
            ("hc_attn_scale", vec![3], TensorOperation::Vector),
            (
                "hc_ffn_fn",
                vec![mix, stream_hidden],
                TensorOperation::Matrix,
            ),
            ("hc_ffn_base", vec![mix], TensorOperation::Vector),
            ("hc_ffn_scale", vec![3], TensorOperation::Vector),
            (
                "ffn.gate.weight",
                vec![experts, hidden],
                TensorOperation::Matrix,
            ),
            (
                "ffn.shared_experts.w1.weight",
                vec![shared, hidden],
                TensorOperation::Matrix,
            ),
            (
                "ffn.shared_experts.w2.weight",
                vec![hidden, shared],
                TensorOperation::Matrix,
            ),
            (
                "ffn.shared_experts.w3.weight",
                vec![shared, hidden],
                TensorOperation::Matrix,
            ),
        ] {
            let format = if operation == TensorOperation::Matrix
                && !local.starts_with("hc_")
                && local != "ffn.gate.weight"
            {
                args.linear_format
            } else {
                LinearFormat::Dense
            };
            specs.push(v4_spec(format!("{root}.{local}"), shape, operation, format));
        }
        if layer < args.num_hash_layers as usize {
            specs.push(v4_spec(
                format!("{root}.ffn.gate.tid2eid"),
                vec![vocab, routed],
                TensorOperation::I32,
                LinearFormat::Dense,
            ));
        } else {
            specs.push(v4_spec(
                format!("{root}.ffn.gate.bias"),
                vec![experts],
                TensorOperation::Vector,
                LinearFormat::Dense,
            ));
        }
        let policy = args
            .attention_policy(layer)
            .ok_or_else(|| format!("missing attention policy for layer {layer}"))?;
        if let V4AttentionPolicy::Compressed { ratio } = policy {
            let output = checked_mul(head_dim, if ratio == 4 { 2 } else { 1 }, "pool output")?;
            let ratio = dimension(ratio, "compression ratio")?;
            for (local, shape, operation) in [
                (
                    "attn.compressor.wkv.weight",
                    vec![output, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "attn.compressor.wgate.weight",
                    vec![output, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "attn.compressor.ape",
                    vec![ratio, output],
                    TensorOperation::Dense,
                ),
                (
                    "attn.compressor.norm.weight",
                    vec![head_dim],
                    TensorOperation::Vector,
                ),
            ] {
                let format = if operation == TensorOperation::Matrix {
                    args.linear_format
                } else {
                    LinearFormat::Dense
                };
                specs.push(v4_spec(format!("{root}.{local}"), shape, operation, format));
            }
            if ratio == 4 {
                let index_output = checked_mul(2, index_dim, "index compressor width")?;
                for (local, shape, operation) in [
                    (
                        "attn.indexer.wq_b.weight",
                        vec![
                            checked_mul(index_heads, index_dim, "index query width")?,
                            query_rank,
                        ],
                        TensorOperation::Matrix,
                    ),
                    (
                        "attn.indexer.weights_proj.weight",
                        vec![index_heads, hidden],
                        TensorOperation::Matrix,
                    ),
                    (
                        "attn.indexer.compressor.wkv.weight",
                        vec![index_output, hidden],
                        TensorOperation::Matrix,
                    ),
                    (
                        "attn.indexer.compressor.wgate.weight",
                        vec![index_output, hidden],
                        TensorOperation::Matrix,
                    ),
                    (
                        "attn.indexer.compressor.ape",
                        vec![4, index_output],
                        TensorOperation::Dense,
                    ),
                    (
                        "attn.indexer.compressor.norm.weight",
                        vec![index_dim],
                        TensorOperation::Vector,
                    ),
                ] {
                    let format = if operation == TensorOperation::Matrix {
                        args.linear_format
                    } else {
                        LinearFormat::Dense
                    };
                    specs.push(v4_spec(format!("{root}.{local}"), shape, operation, format));
                }
            }
        }
        append_v4_experts(&mut specs, args, &root, experts, hidden)?;
    }
    Ok(specs)
}

fn append_v4_experts(
    specs: &mut Vec<V4TensorSpec>,
    args: &V4Args,
    root: &str,
    experts: usize,
    hidden: usize,
) -> Result<(), String> {
    let intermediate = dimension(args.moe_intermediate_size, "expert width")?;
    let format = match args.expert_format {
        ExpertFormat::Dense => LinearFormat::Dense,
        ExpertFormat::MxFp4 => LinearFormat::MxFp4,
        ExpertFormat::BlockFp8 => match args.linear_format {
            format @ LinearFormat::E4M3BlockFp8(_) => format,
            _ => LinearFormat::E4M3BlockFp8(
                BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::Ue8m0)
                    .map_err(|error| error.to_string())?,
            ),
        },
    };
    for expert in 0..experts {
        for (projection, shape) in [
            ("w1", vec![intermediate, hidden]),
            ("w2", vec![hidden, intermediate]),
            ("w3", vec![intermediate, hidden]),
        ] {
            specs.push(v4_spec(
                format!("{root}.ffn.experts.{expert}.{projection}.weight"),
                shape,
                TensorOperation::Matrix,
                format,
            ));
        }
    }
    Ok(())
}

fn append_v4_draft_specs(specs: &mut Vec<V4TensorSpec>, args: &V4Args) -> Result<(), String> {
    let draft_layers = count(args.num_nextn_predict_layers, "draft layer count")?;
    if draft_layers == 0 {
        return Ok(());
    }
    let target_layers = dimension(args.num_hidden_layers, "target layer count")?;
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let streams = dimension(args.hc_mult, "hyper streams")?;
    let stream_hidden = checked_mul(streams, hidden, "hyper width")?;
    let experts = dimension(args.n_routed_experts, "expert count")?;
    let routed = dimension(args.num_experts_per_tok, "routes per token")?;
    let template_root = format!("layers.{}", target_layers - 1);
    let template_prefix = format!("{template_root}.");
    let template = specs
        .iter()
        .filter(|spec| spec.name.starts_with(&template_prefix))
        .filter(|spec| {
            !spec.name.contains(".attn.compressor.")
                && !spec.name.contains(".attn.indexer.")
                && !spec.name.ends_with(".ffn.gate.bias")
                && !spec.name.ends_with(".ffn.gate.tid2eid")
        })
        .cloned()
        .collect::<Vec<_>>();
    for depth in 0..draft_layers {
        let root = format!("mtp.{depth}");
        for source in &template {
            let suffix = source.name.strip_prefix(&template_prefix).unwrap();
            let mut cloned = source.clone();
            cloned.name = format!("{root}.{suffix}");
            specs.push(cloned);
        }
        if target_layers + depth < args.num_hash_layers as usize {
            specs.push(v4_spec(
                format!("{root}.ffn.gate.tid2eid"),
                vec![vocab, routed],
                TensorOperation::I32,
                LinearFormat::Dense,
            ));
        } else {
            specs.push(v4_spec(
                format!("{root}.ffn.gate.bias"),
                vec![experts],
                TensorOperation::Vector,
                LinearFormat::Dense,
            ));
        }
    }
    if let Some(dspark) = &args.dspark {
        let last = draft_layers - 1;
        let markov = dimension(dspark.markov_rank, "DSpark Markov rank")?;
        let captured = checked_mul(
            hidden,
            args.target_capture_policy
                .as_ref()
                .expect("validated DSpark target capture policy")
                .len(),
            "captured hidden",
        )?;
        for (name, shape, operation, format) in [
            (
                "mtp.0.main_proj.weight".to_string(),
                vec![hidden, captured],
                TensorOperation::Matrix,
                args.linear_format,
            ),
            (
                "mtp.0.main_norm.weight".to_string(),
                vec![hidden],
                TensorOperation::Vector,
                LinearFormat::Dense,
            ),
            (
                format!("mtp.{last}.norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
                LinearFormat::Dense,
            ),
            (
                format!("mtp.{last}.hc_head_fn"),
                vec![streams, stream_hidden],
                TensorOperation::Matrix,
                args.linear_format,
            ),
            (
                format!("mtp.{last}.hc_head_base"),
                vec![streams],
                TensorOperation::Vector,
                LinearFormat::Dense,
            ),
            (
                format!("mtp.{last}.hc_head_scale"),
                vec![1],
                TensorOperation::Vector,
                LinearFormat::Dense,
            ),
            (
                format!("mtp.{last}.markov_head.markov_w1.weight"),
                vec![vocab, markov],
                TensorOperation::Matrix,
                LinearFormat::Dense,
            ),
            (
                format!("mtp.{last}.markov_head.markov_w2.weight"),
                vec![vocab, markov],
                TensorOperation::Matrix,
                args.linear_format,
            ),
            (
                format!("mtp.{last}.confidence_head.proj.weight"),
                vec![1, checked_add(hidden, markov, "DSpark confidence width")?],
                TensorOperation::Matrix,
                args.linear_format,
            ),
        ] {
            specs.push(v4_spec(name, shape, operation, format));
        }
    } else {
        for depth in 0..draft_layers {
            let root = format!("mtp.{depth}");
            for (local, shape, operation, format) in [
                (
                    "e_proj.weight",
                    vec![hidden, hidden],
                    TensorOperation::Matrix,
                    args.linear_format,
                ),
                (
                    "h_proj.weight",
                    vec![hidden, hidden],
                    TensorOperation::Matrix,
                    args.linear_format,
                ),
                (
                    "enorm.weight",
                    vec![hidden],
                    TensorOperation::Vector,
                    LinearFormat::Dense,
                ),
                (
                    "hnorm.weight",
                    vec![hidden],
                    TensorOperation::Vector,
                    LinearFormat::Dense,
                ),
                (
                    "norm.weight",
                    vec![hidden],
                    TensorOperation::Vector,
                    LinearFormat::Dense,
                ),
                (
                    "hc_head_fn",
                    vec![streams, stream_hidden],
                    TensorOperation::Matrix,
                    args.linear_format,
                ),
                (
                    "hc_head_base",
                    vec![streams],
                    TensorOperation::Vector,
                    LinearFormat::Dense,
                ),
                (
                    "hc_head_scale",
                    vec![1],
                    TensorOperation::Vector,
                    LinearFormat::Dense,
                ),
            ] {
                specs.push(v4_spec(format!("{root}.{local}"), shape, operation, format));
            }
        }
    }
    Ok(())
}

fn v4_spec(
    name: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
    format: LinearFormat,
) -> V4TensorSpec {
    V4TensorSpec {
        name: name.into(),
        shape,
        operation,
        format,
    }
}

fn v4_matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    format: LinearFormat,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let scale = matches!(format, LinearFormat::E4M3BlockFp8(_) | LinearFormat::MxFp4).then(|| {
        let prefix = name.strip_suffix(".weight").unwrap_or(name);
        if format == LinearFormat::MxFp4 {
            MatrixScaleNames {
                key: format!("{prefix}.scale"),
                aliases: vec![format!("{prefix}.scales")],
            }
        } else {
            MatrixScaleNames {
                key: format!("{prefix}.weight_scale_inv"),
                aliases: vec![format!("{prefix}.scale")],
            }
        }
    });
    matrix_for_linear_format(name, std::iter::empty::<String>(), shape, format, scale)
        .map_err(|error| error.to_string())
}

fn gguf_tensor(
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
        .ok_or_else(|| format!("{name} must be positive, got {value}"))
}

fn count(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{name} must be nonnegative, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{name} overflowed"))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("{name} overflowed"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek::{parse_v3_config, parse_v4_config};

    #[test]
    fn expert_unit_recipe_owns_segmented_rank_local_selection() {
        struct Catalog(BTreeMap<String, eredu_checkpoint::store::TensorMetadata>);
        impl RecipeCatalog for Catalog {
            fn tensor_metadata(
                &self,
                key: &str,
            ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError>
            {
                self.0.get(key).cloned().ok_or_else(|| {
                    eredu_checkpoint::store::StoreError::UnknownTensor { key: key.into() }
                })
            }
        }
        let metadata = |name: &str, shape: Vec<usize>| eredu_checkpoint::store::TensorMetadata {
            name: name.into(),
            encoded_byte_len: shape.iter().product::<usize>() as u64 * 4,
            physical_shape: shape.clone(),
            logical_shape: shape,
            stored_dtype: StoredDtype::F32,
            backing_shard: None,
        };
        let catalog = Catalog(
            [
                ("gate_up".into(), metadata("gate_up", vec![4, 128, 96])),
                ("down".into(), metadata("down", vec![4, 96, 64])),
            ]
            .into_iter()
            .collect(),
        );
        let bank = GatedProductExpertRecipes {
            target_gate_up: "experts.gate_up_proj".into(),
            target_down: "experts.down_proj".into(),
            layout: eredu_checkpoint::expert::GatedProductExpertStorageLayout::Packed,
            gate_up: DerivedWeightRecipe::source("gate_up", TensorSelection::Full),
            down: DerivedWeightRecipe::source("down", TensorSelection::Full),
        };

        let local = expert_unit_recipes(&catalog, &bank, 2, Some(16..48)).unwrap();
        assert_eq!(local.gate_up.infer(&catalog).unwrap().shape(), [1, 64, 96]);
        assert_eq!(local.down.infer(&catalog).unwrap().shape(), [1, 96, 32]);
        assert!(matches!(
            local.gate_up,
            DerivedWeightRecipe::Concatenate { axis: 1, ref inputs }
                if inputs.len() == 2
        ));
        assert!(expert_unit_recipes(&catalog, &bank, 2, Some(48..65)).is_err());
    }

    #[test]
    fn v3_residency_catalog_owns_sparse_mtp_topology_and_local_selection() {
        struct Catalog(BTreeMap<String, eredu_checkpoint::store::TensorMetadata>);
        impl RecipeCatalog for Catalog {
            fn tensor_metadata(
                &self,
                key: &str,
            ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError>
            {
                self.0.get(key).cloned().ok_or_else(|| {
                    eredu_checkpoint::store::StoreError::UnknownTensor { key: key.into() }
                })
            }
        }
        let args = parse_v3_config(&serde_json::json!({
            "model_type": "deepseek_v3",
            "hidden_size": 128,
            "intermediate_size": 256,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 1,
            "num_attention_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "q_lora_rank": 128,
            "kv_lora_rank": 128,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 64,
            "v_head_dim": 64,
            "first_k_dense_replace": 1,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "tie_word_embeddings": false
        }))
        .unwrap();
        let metadata = |name: &str, shape: Vec<usize>| eredu_checkpoint::store::TensorMetadata {
            name: name.into(),
            encoded_byte_len: shape.iter().product::<usize>() as u64 * 4,
            physical_shape: shape.clone(),
            logical_shape: shape,
            stored_dtype: StoredDtype::F32,
            backing_shard: None,
        };
        let mut tensors = BTreeMap::new();
        for layer in [1, 2] {
            let root = format!("model.layers.{layer}.mlp.experts");
            for (binding, shape) in [
                ("gate_up_proj", vec![4, 128, 128]),
                ("down_proj", vec![4, 128, 64]),
            ] {
                let name = format!("{root}.{binding}");
                tensors.insert(name.clone(), metadata(&name, shape));
            }
        }
        let catalog = Catalog(tensors);
        let residency = v3_expert_residency_catalog(&catalog, &args, Some(16..48)).unwrap();
        assert_eq!(residency.units().len(), 8);
        let target = &residency.units()[0];
        assert_eq!(
            target.identity(),
            eredu_runtime::ParameterBankKey::new(1, 0)
        );
        assert_eq!(target.owner_group().as_str(), "target");
        assert_eq!(target.owner_unit(), 1);
        assert_eq!(target.unit_path(), "model.layers.1");
        assert_eq!(
            target.parameters()[0]
                .recipe()
                .infer(&catalog)
                .unwrap()
                .shape(),
            [1, 64, 128]
        );
        assert_eq!(
            target.parameters()[1]
                .recipe()
                .infer(&catalog)
                .unwrap()
                .shape(),
            [1, 128, 32]
        );
        let prediction = &residency.units()[4];
        assert_eq!(
            prediction.identity(),
            eredu_runtime::ParameterBankKey::new(2, 0)
        );
        assert_eq!(prediction.owner_group().as_str(), "mtp.0");
        assert_eq!(prediction.owner_unit(), 0);
        assert_eq!(prediction.unit_path(), "model.layers.2");
        assert_eq!(
            residency.units()[7].identity(),
            eredu_runtime::ParameterBankKey::new(2, 3)
        );
        let prediction = residency
            .into_units_selected_by_owner(|group, unit| group.as_str() == "mtp.0" && unit == 0)
            .collect::<Vec<_>>();
        assert_eq!(prediction.len(), 4);
        assert!(prediction.iter().all(|unit| {
            unit.identity().unit() == 2
                && unit.owner_group().as_str() == "mtp.0"
                && unit.owner_unit() == 0
        }));
    }

    #[test]
    fn translates_v3_fused_split_and_expert_names() {
        assert_eq!(
            translate_v3_gguf_weight_name("blk.2.attn_k_b.weight"),
            "model.layers.2.self_attn.k_b_proj.weight"
        );
        assert_eq!(
            translate_v3_gguf_weight_name("blk.3.ffn_gate_exps.scales"),
            "model.layers.3.mlp.experts.gate_proj_scales"
        );
    }

    #[test]
    fn translates_v4_hyper_index_and_expert_names() {
        assert_eq!(
            translate_v4_gguf_weight_name("blk.1.hc_attn_fn.weight"),
            "layers.1.hc_attn_fn"
        );
        assert_eq!(
            translate_v4_gguf_weight_name("blk.1.indexer.proj.weight"),
            "layers.1.attn.indexer.weights_proj.weight"
        );
        assert_eq!(
            translate_v4_gguf_weight_name("blk.1.ffn_gate_exps.scales"),
            "layers.1.ffn.expert_banks.w1.scale"
        );
    }

    #[test]
    fn v3_plan_uses_general_block_fp8_and_expert_layouts() {
        let args = parse_v3_config(&serde_json::json!({
            "model_type": "deepseek_v3",
            "hidden_size": 128,
            "intermediate_size": 256,
            "moe_intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "q_lora_rank": 128,
            "kv_lora_rank": 128,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 64,
            "v_head_dim": 64,
            "first_k_dense_replace": 1,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "tie_word_embeddings": false,
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128]
            }
        }))
        .unwrap();
        let plan = v3_safetensors_plan(&args, true).unwrap();
        assert!(plan
            .common_tensors
            .iter()
            .any(|tensor| { tensor.key == "model.layers.0.self_attn.q_a_proj.weight_scale_inv" }));
        assert_eq!(plan.layout_groups.len(), 3);
        assert!(plan
            .layout_groups
            .iter()
            .all(|group| group.variants.len() == 2));
    }

    #[test]
    fn v4_plan_distinguishes_native_linears_hyper_state_and_mxfp4_experts() {
        let args = parse_v4_config(&serde_json::json!({
            "model_type": "deepseek_v4",
            "hidden_size": 128,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "qk_rope_head_dim": 8,
            "q_lora_rank": 32,
            "o_lora_rank": 16,
            "o_groups": 2,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "sliding_window": 128,
            "compress_ratios": [0, 4, 0, 0],
            "index_n_heads": 4,
            "index_head_dim": 16,
            "index_topk": 2,
            "hc_mult": 2,
            "hc_sinkhorn_iters": 4,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "num_hash_layers": 1,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": true,
            "routed_scaling_factor": 1.0,
            "num_nextn_predict_layers": 1,
            "expert_dtype": "fp4",
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "scale_fmt": "ue8m0"
            },
            "dspark_block_size": 4,
            "dspark_noise_token_id": 0,
            "dspark_target_layer_ids": [0, 2],
            "dspark_markov_rank": 32
        }))
        .unwrap();
        let plan = v4_safetensors_plan(&args).unwrap();
        let tensor = |name: &str| {
            plan.common_tensors
                .iter()
                .find(|tensor| tensor.key == name)
                .unwrap_or_else(|| panic!("missing {name}"))
        };

        assert_eq!(
            tensor("layers.0.attn.wq_a.weight_scale_inv").dtype,
            StoredDtypeConstraint::Exact(StoredDtype::F8E8M0)
        );
        assert_eq!(
            tensor("layers.0.hc_attn_fn").dtype,
            StoredDtypeConstraint::Floating
        );
        assert!(!plan
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "layers.0.hc_attn_fn.weight_scale_inv"));
        assert_eq!(
            tensor("layers.0.ffn.gate.tid2eid").dtype,
            StoredDtypeConstraint::Exact(StoredDtype::I32)
        );
        assert_eq!(
            tensor("layers.0.ffn.experts.0.w1.weight").dtype,
            StoredDtypeConstraint::Exact(StoredDtype::U32)
        );
        assert_eq!(tensor("layers.0.ffn.experts.0.w1.scale").shape, vec![64, 4]);
        assert!(plan
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "mtp.0.main_proj.weight_scale_inv"));
        assert_eq!(
            tensor("mtp.0.markov_head.markov_w1.weight").dtype,
            StoredDtypeConstraint::Floating
        );

        let quantization =
            WeightQuantization::Affine(eredu_checkpoint::AffineQuantization::new(32, 4).unwrap());
        let target = v4_load_time_quantization(&args, quantization).unwrap();
        assert_eq!(target.linear_format, quantization.into());
        assert_eq!(
            target.linear_formats,
            BTreeMap::from([
                (
                    "layers.0.ffn.switch_mlp.down_proj".into(),
                    quantization.into(),
                ),
                (
                    "layers.0.ffn.switch_mlp.gate_up_proj".into(),
                    quantization.into(),
                ),
                (
                    "layers.1.ffn.switch_mlp.down_proj".into(),
                    quantization.into(),
                ),
                (
                    "layers.1.ffn.switch_mlp.gate_up_proj".into(),
                    quantization.into(),
                ),
                (
                    "layers.2.ffn.switch_mlp.down_proj".into(),
                    quantization.into(),
                ),
                (
                    "layers.2.ffn.switch_mlp.gate_up_proj".into(),
                    quantization.into(),
                ),
                ("mtp.0.ffn.switch_mlp.down_proj".into(), quantization.into(),),
                (
                    "mtp.0.ffn.switch_mlp.gate_up_proj".into(),
                    quantization.into(),
                ),
            ])
        );
    }

    #[test]
    fn v3_gguf_plan_has_one_global_fused_or_split_kv_layout() {
        let args = parse_v3_config(&serde_json::json!({
            "model_type": "deepseek_v3",
            "hidden_size": 128,
            "intermediate_size": 256,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 24,
            "qk_rope_head_dim": 8,
            "v_head_dim": 24,
            "first_k_dense_replace": 1,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
            "tie_word_embeddings": false
        }))
        .unwrap();
        let plan = v3_gguf_plan(&args).unwrap();
        assert_eq!(plan.layout_groups.len(), 1);
        assert_eq!(plan.layout_groups[0].variants.len(), 2);
        assert_eq!(plan.layout_groups[0].variants[0].tensors.len(), 2);
        assert_eq!(plan.layout_groups[0].variants[1].tensors.len(), 4);

        let prefix = "model.layers.1.mlp.experts";
        let mut formats = BTreeMap::from([(format!("{prefix}.gate_proj"), 1)]);
        normalize_v3_weight_formats(&args, &mut formats);
        assert_eq!(formats.get(&format!("{prefix}.gate_up_proj")), Some(&1));
        assert_eq!(formats.get(&format!("{prefix}.gate_proj")), Some(&1));

        struct Catalog(std::collections::BTreeMap<String, eredu_checkpoint::store::TensorMetadata>);
        impl eredu_checkpoint::recipe::RecipeCatalog for Catalog {
            fn tensor_metadata(
                &self,
                key: &str,
            ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError>
            {
                self.0.get(key).cloned().ok_or_else(|| {
                    eredu_checkpoint::store::StoreError::UnknownTensor { key: key.into() }
                })
            }
        }
        let metadata = |name: &str, shape: Vec<usize>| eredu_checkpoint::store::TensorMetadata {
            name: name.into(),
            encoded_byte_len: shape.iter().product::<usize>() as u64 * 4,
            physical_shape: shape.clone(),
            logical_shape: shape,
            stored_dtype: StoredDtype::F32,
            backing_shard: None,
        };
        let catalog = Catalog(
            [
                (
                    "blk.0.attn_k_b.weight".into(),
                    metadata("blk.0.attn_k_b.weight", vec![4, 32, 24]),
                ),
                (
                    "blk.0.attn_v_b.weight".into(),
                    metadata("blk.0.attn_v_b.weight", vec![4, 24, 32]),
                ),
                (
                    "blk.0.attn_kv_b.weight".into(),
                    metadata("blk.0.attn_kv_b.weight", vec![192, 32]),
                ),
                (
                    "model.layers.0.self_attn.k_b_proj.weight".into(),
                    metadata("model.layers.0.self_attn.k_b_proj.weight", vec![4, 32, 24]),
                ),
                (
                    "model.layers.0.self_attn.v_b_proj.weight".into(),
                    metadata("model.layers.0.self_attn.v_b_proj.weight", vec![4, 24, 32]),
                ),
            ]
            .into_iter()
            .collect(),
        );
        for split in [false, true] {
            let inferred = v3_gguf_kv_b_recipe(&args, 0, split)
                .unwrap()
                .infer(&catalog)
                .unwrap();
            assert_eq!(inferred.shape(), [192, 32]);
        }
        let unit = v3_unit_recipes(&catalog, &args, 0, false).unwrap();
        let recipe = unit
            .get("model.layers.0.self_attn.kv_b_proj.weight")
            .unwrap();
        assert!(matches!(
            recipe,
            DerivedWeightRecipe::Reshape { input, shape }
                if shape == &[192, 32]
                    && matches!(input.as_ref(), DerivedWeightRecipe::Concatenate { axis: 1, .. })
        ));
        assert_eq!(recipe.infer(&catalog).unwrap().shape(), [192, 32]);
    }

    #[test]
    fn v4_gguf_plan_uses_physical_names_and_packed_mxfp4_banks() {
        let args = parse_v4_config(&serde_json::json!({
            "model_type": "deepseek_v4",
            "hidden_size": 128,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "qk_rope_head_dim": 8,
            "q_lora_rank": 32,
            "o_lora_rank": 16,
            "o_groups": 2,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "sliding_window": 128,
            "compress_ratios": [0, 4, 0],
            "index_n_heads": 4,
            "index_head_dim": 16,
            "index_topk": 2,
            "hc_mult": 2,
            "hc_sinkhorn_iters": 4,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "num_hash_layers": 1,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": true,
            "expert_dtype": "fp4"
        }))
        .unwrap();
        let plan = v4_gguf_plan(&args).unwrap();
        let tensor = |name: &str| {
            plan.common_tensors
                .iter()
                .find(|tensor| tensor.key == name)
                .unwrap_or_else(|| panic!("missing {name}"))
        };
        assert_eq!(tensor("blk.0.ffn_gate_tid2eid.weight").shape, vec![128, 2]);
        assert_eq!(
            tensor("blk.1.ffn_gate_exps.weight").encoding,
            GgufTypeConstraint::OperationClass(TensorOperation::MxFp4Matrix)
        );
        assert_eq!(
            tensor("blk.1.indexer_compressor_ape.weight").shape,
            vec![4, 32]
        );

        let root = "layers.1.ffn";
        let mut formats = BTreeMap::from([
            (format!("{root}.expert_banks.w1.weight"), 1),
            (format!("{root}.expert_banks.w2"), 2),
        ]);
        normalize_v4_weight_formats(&args, &mut formats);
        assert_eq!(
            formats.get(&format!("{root}.switch_mlp.gate_up_proj")),
            Some(&1)
        );
        assert_eq!(
            formats.get(&format!("{root}.switch_mlp.down_proj")),
            Some(&2)
        );
    }
}
