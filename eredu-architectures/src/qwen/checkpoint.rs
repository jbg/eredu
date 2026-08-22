//! Pure checkpoint schemas, name translation, and recipes for Qwen text models.

use std::collections::BTreeMap;

use eredu_checkpoint::schema::{
    matrix_for_linear_format, AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan,
    GgufTensorConstraint, GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan,
    SafetensorsTensorConstraint, TensorOperation,
};
use eredu_checkpoint::{
    expert::{
        resolve_gated_product_expert_recipes, GatedProductExpertLayoutNames,
        GatedProductExpertRecipes, IndependentGatedProductExpertNames,
    },
    recipe::RecipeCatalog,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe, store::TensorSelection, LinearFormat, WeightQuantization,
};

use super::{ModelArgs, QwenVariant};

/// Builds the canonical Qwen SafeTensors catalog plan.
pub fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
    safetensors_plan_with_root(args, &args.parameter_root, true)
}

/// Builds a SafeTensors plan under an embedding-specific decoder root.
pub fn safetensors_plan_with_root(
    args: &ModelArgs,
    root: &str,
    allow_derived_expert_layouts: bool,
) -> Result<SafetensorsCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden_size")?;
    let vocab = dimension(args.vocab_size, "vocab_size")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "num_attention_heads")?,
        dimension(args.head_dim, "head_dim")?,
        "query width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "num_key_value_heads")?,
        dimension(args.head_dim, "head_dim")?,
        "key/value width",
    )?;
    let head = dimension(args.head_dim, "head_dim")?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    add_tensor(
        args,
        &mut common,
        format!("{root}.embed_tokens.weight"),
        vec![vocab, hidden],
        TensorOperation::Matrix,
        None,
    )?;
    add_tensor(
        args,
        &mut common,
        format!("{root}.norm.weight"),
        vec![hidden],
        TensorOperation::Vector,
        None,
    )?;
    if args.tie_word_embeddings {
        let tensors = matrix_constraints(
            "lm_head.weight",
            vec![vocab, hidden],
            args.weight_quantization_for("lm_head.weight"),
            Vec::new(),
        )?;
        groups.push(AlternativeLayoutGroup {
            id: "redundant tied output head".into(),
            required: false,
            variants: vec![LayoutVariant {
                id: "present".into(),
                discriminator_keys: tensors.iter().map(|tensor| tensor.key.clone()).collect(),
                tensors,
            }],
        });
    } else {
        add_tensor(
            args,
            &mut common,
            "lm_head.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
            None,
        )?;
    }
    for layer in 0..dimension(args.num_hidden_layers, "num_hidden_layers")? {
        let block = format!("{root}.layers.{layer}");
        for (name, shape, operation) in [
            (
                "input_layernorm.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            (
                "post_attention_layernorm.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            (
                "self_attn.q_proj.weight",
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.k_proj.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.v_proj.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.o_proj.weight",
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
        ] {
            add_tensor(
                args,
                &mut common,
                format!("{block}.{name}"),
                shape,
                operation,
                None,
            )?;
        }
        if args.variant != QwenVariant::Qwen2 {
            for name in ["q_norm.weight", "k_norm.weight"] {
                add_tensor(
                    args,
                    &mut common,
                    format!("{block}.self_attn.{name}"),
                    vec![head],
                    TensorOperation::Vector,
                    None,
                )?;
            }
        }
        if args.variant == QwenVariant::Qwen2 {
            for (name, size) in [
                ("q_proj.bias", query),
                ("k_proj.bias", key_value),
                ("v_proj.bias", key_value),
            ] {
                add_tensor(
                    args,
                    &mut common,
                    format!("{block}.self_attn.{name}"),
                    vec![size],
                    TensorOperation::Vector,
                    None,
                )?;
            }
        }
        if args.is_moe() {
            let experts = dimension(args.num_experts, "num_experts")?;
            let intermediate = dimension(args.moe_intermediate_size, "moe_intermediate_size")?;
            add_tensor(
                args,
                &mut common,
                format!("{block}.mlp.gate.weight"),
                vec![experts, hidden],
                TensorOperation::Matrix,
                Some(None),
            )?;
            groups.push(expert_layout_group(
                args,
                &format!("{block}.mlp.experts"),
                experts,
                hidden,
                intermediate,
                allow_derived_expert_layouts,
            )?);
        } else {
            let intermediate = dimension(args.intermediate_size, "intermediate_size")?;
            for (projection, shape) in [
                ("gate_proj", vec![intermediate, hidden]),
                ("up_proj", vec![intermediate, hidden]),
                ("down_proj", vec![hidden, intermediate]),
            ] {
                add_tensor(
                    args,
                    &mut common,
                    format!("{block}.mlp.{projection}.weight"),
                    shape,
                    TensorOperation::Matrix,
                    None,
                )?;
            }
        }
    }
    SafetensorsCheckpointPlan::new(
        format!("{} SafeTensors", args.model_type),
        common,
        groups,
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

/// Resolves packed, separate-packed, or per-expert sources into one canonical bank.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
    root: &str,
    layer: usize,
) -> Result<GatedProductExpertRecipes, String> {
    if !args.is_moe() {
        return Err("Qwen expert recipes require Qwen3-MoE arguments".into());
    }
    let layer_count = dimension(args.num_hidden_layers, "num_hidden_layers")?;
    if layer >= layer_count {
        return Err(format!(
            "Qwen expert recipe layer {layer} is outside {layer_count} layers"
        ));
    }
    let count = dimension(args.num_experts, "num_experts")?;
    let prefix = format!("{root}.layers.{layer}.mlp.experts");
    let separate_suffix = if catalog
        .tensor_metadata(&format!("{prefix}.gate_proj"))
        .is_ok()
    {
        ""
    } else {
        ".weight"
    };
    let names = GatedProductExpertLayoutNames {
        target_gate_up: format!("{prefix}.gate_up_proj"),
        target_down: format!("{prefix}.down_proj"),
        packed_gate_up: format!("{prefix}.gate_up_proj"),
        packed_down: format!("{prefix}.down_proj"),
        separate_gate: format!("{prefix}.gate_proj{separate_suffix}"),
        separate_up: format!("{prefix}.up_proj{separate_suffix}"),
        separate_down: format!("{prefix}.down_proj{separate_suffix}"),
        independent: (0..count)
            .map(|expert| IndependentGatedProductExpertNames {
                gate: format!("{prefix}.{expert}.gate_proj.weight"),
                up: format!("{prefix}.{expert}.up_proj.weight"),
                down: format!("{prefix}.{expert}.down_proj.weight"),
            })
            .collect(),
    };
    resolve_gated_product_expert_recipes(catalog, &names).map_err(|error| error.to_string())
}

fn recipe_source_key<C: RecipeCatalog + ?Sized>(catalog: &C, base: &str) -> Option<String> {
    if catalog.tensor_metadata(base).is_ok() {
        return Some(base.to_owned());
    }
    let weight = format!("{base}.weight");
    catalog.tensor_metadata(&weight).is_ok().then_some(weight)
}

fn split_expert_source<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    prefix: &str,
    expert: usize,
    projections: &[&str],
) -> Result<String, String> {
    projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.weight"))
        .find(|key| catalog.tensor_metadata(key).is_ok())
        .ok_or_else(|| {
            format!("Qwen checkpoint is missing split expert {expert} projection {projections:?}")
        })
}

/// Returns neutral lazy-loading recipes for one Qwen routed expert.
pub fn expert_unit_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
    layer: usize,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    if !args.is_moe() {
        return Err("Qwen expert recipes require Qwen3-MoE arguments".into());
    }
    let layers = dimension(args.num_hidden_layers, "num_hidden_layers")?;
    if layer >= layers {
        return Err(format!(
            "Qwen expert recipe layer {layer} is outside {layers} layers"
        ));
    }
    let experts = dimension(args.num_experts, "num_experts")?;
    if expert >= experts {
        return Err(format!(
            "Qwen expert {expert} is outside {experts} experts in layer {layer}"
        ));
    }
    let prefix = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
    let packed_gate_up = format!("{prefix}.gate_up_proj");
    let packed_down = format!("{prefix}.down_proj");
    let packed_gate_up_source = recipe_source_key(catalog, &packed_gate_up);
    let packed_down_source = recipe_source_key(catalog, &packed_down);
    let split_gate_source = recipe_source_key(catalog, &format!("{prefix}.gate_proj"));
    let split_up_source = recipe_source_key(catalog, &format!("{prefix}.up_proj"));
    let selection = TensorSelection::Range {
        axis: 0,
        start: expert,
        end: expert + 1,
    };
    let mut recipes = BTreeMap::new();
    if let (Some(gate_up), Some(down)) = (packed_gate_up_source, packed_down_source.as_ref()) {
        recipes.insert(
            "gate_up_proj".into(),
            DerivedWeightRecipe::source(gate_up, selection.clone()),
        );
        recipes.insert(
            "down_proj".into(),
            DerivedWeightRecipe::source(down, selection.clone()),
        );
        for (target, source) in [
            ("gate_up_proj_scales", format!("{packed_gate_up}_scales")),
            ("gate_up_proj_biases", format!("{packed_gate_up}_biases")),
            ("down_proj_scales", format!("{packed_down}_scales")),
            ("down_proj_biases", format!("{packed_down}_biases")),
        ] {
            if catalog.tensor_metadata(&source).is_ok() {
                recipes.insert(
                    target.into(),
                    DerivedWeightRecipe::source(source, selection.clone()),
                );
            }
        }
        return Ok(recipes);
    }
    if let (Some(gate), Some(up), Some(down)) =
        (split_gate_source, split_up_source, packed_down_source)
    {
        recipes.insert(
            "gate_up_proj".into(),
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![
                    DerivedWeightRecipe::source(gate, selection.clone()),
                    DerivedWeightRecipe::source(up, selection.clone()),
                ],
            },
        );
        recipes.insert(
            "down_proj".into(),
            DerivedWeightRecipe::source(down, selection.clone()),
        );
        for suffix in ["_scales", "_biases"] {
            let gate = format!("{prefix}.gate_proj{suffix}");
            let up = format!("{prefix}.up_proj{suffix}");
            if catalog.tensor_metadata(&gate).is_ok() && catalog.tensor_metadata(&up).is_ok() {
                recipes.insert(
                    format!("gate_up_proj{suffix}"),
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(gate, selection.clone()),
                            DerivedWeightRecipe::source(up, selection.clone()),
                        ],
                    },
                );
            }
            let down = format!("{packed_down}{suffix}");
            if catalog.tensor_metadata(&down).is_ok() {
                recipes.insert(
                    format!("down_proj{suffix}"),
                    DerivedWeightRecipe::source(down, selection.clone()),
                );
            }
        }
        return Ok(recipes);
    }
    if args.weight_quantization_for(&packed_gate_up).is_some()
        || args.weight_quantization_for(&packed_down).is_some()
    {
        return Err(
            "split Qwen experts cannot be lazily load-time quantized; use checkpoint-native packed expert weights"
                .into(),
        );
    }
    let gate = split_expert_source(catalog, &prefix, expert, &["gate_proj", "w1"])?;
    let up = split_expert_source(catalog, &prefix, expert, &["up_proj", "w3"])?;
    let down = split_expert_source(catalog, &prefix, expert, &["down_proj", "w2"])?;
    recipes.insert(
        "gate_up_proj".into(),
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![
                    DerivedWeightRecipe::source(gate, TensorSelection::Full),
                    DerivedWeightRecipe::source(up, TensorSelection::Full),
                ],
            }],
        },
    );
    recipes.insert(
        "down_proj".into(),
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![DerivedWeightRecipe::source(down, TensorSelection::Full)],
        },
    );
    Ok(recipes)
}

fn expert_layout_group(
    args: &ModelArgs,
    prefix: &str,
    experts: usize,
    hidden: usize,
    intermediate: usize,
    allow_derived_layouts: bool,
) -> Result<AlternativeLayoutGroup<SafetensorsTensorConstraint>, String> {
    let gate_up_quantization = args.weight_quantization_for(&format!("{prefix}.gate_up_proj"));
    let down_quantization = args.weight_quantization_for(&format!("{prefix}.down_proj"));
    let packed_gate_up = format!("{prefix}.gate_up_proj");
    let packed_down = format!("{prefix}.down_proj");
    let packed = expert_variant(
        "packed",
        vec![packed_gate_up.clone()],
        [
            (
                packed_gate_up,
                vec![
                    experts,
                    checked_mul(2, intermediate, "packed expert width")?,
                    hidden,
                ],
                gate_up_quantization,
                Vec::new(),
            ),
            (
                packed_down.clone(),
                vec![experts, hidden, intermediate],
                down_quantization,
                Vec::new(),
            ),
        ],
    )?;
    let gate = format!("{prefix}.gate_proj");
    let up = format!("{prefix}.up_proj");
    let separate = expert_variant(
        "separate-packed",
        vec![gate.clone(), up.clone()],
        [
            (
                gate.clone(),
                vec![experts, intermediate, hidden],
                gate_up_quantization,
                Vec::new(),
            ),
            (
                up.clone(),
                vec![experts, intermediate, hidden],
                gate_up_quantization,
                Vec::new(),
            ),
            (
                packed_down,
                vec![experts, hidden, intermediate],
                down_quantization,
                Vec::new(),
            ),
        ],
    )?;
    let mut variants = vec![packed];
    if allow_derived_layouts {
        variants.push(separate);
    }
    if allow_derived_layouts && gate_up_quantization.is_none() && down_quantization.is_none() {
        let mut tensors = Vec::with_capacity(checked_mul(experts, 3, "expert tensor count")?);
        let mut discriminators = Vec::with_capacity(tensors.capacity());
        for expert in 0..experts {
            for (canonical, aliases, shape) in [
                (
                    format!("{prefix}.{expert}.gate_proj.weight"),
                    vec![format!("{prefix}.{expert}.w1.weight")],
                    vec![intermediate, hidden],
                ),
                (
                    format!("{prefix}.{expert}.up_proj.weight"),
                    vec![format!("{prefix}.{expert}.w3.weight")],
                    vec![intermediate, hidden],
                ),
                (
                    format!("{prefix}.{expert}.down_proj.weight"),
                    vec![format!("{prefix}.{expert}.w2.weight")],
                    vec![hidden, intermediate],
                ),
            ] {
                discriminators.push(canonical.clone());
                tensors.push((canonical, shape, None, aliases));
            }
        }
        variants.push(expert_variant("split", discriminators, tensors)?);
    }
    Ok(AlternativeLayoutGroup {
        id: format!("{prefix} storage"),
        required: true,
        variants,
    })
}

fn expert_variant(
    id: &str,
    discriminator_keys: Vec<String>,
    tensors: impl IntoIterator<Item = (String, Vec<usize>, Option<WeightQuantization>, Vec<String>)>,
) -> Result<LayoutVariant<SafetensorsTensorConstraint>, String> {
    let mut constraints = Vec::new();
    for (name, shape, quantization, aliases) in tensors {
        constraints.extend(matrix_constraints(&name, shape, quantization, aliases)?);
    }
    Ok(LayoutVariant {
        id: id.into(),
        tensors: constraints,
        discriminator_keys,
    })
}

fn add_tensor(
    args: &ModelArgs,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: impl AsRef<str>,
    shape: Vec<usize>,
    operation: TensorOperation,
    quantization_override: Option<Option<WeightQuantization>>,
) -> Result<(), String> {
    let name = name.as_ref();
    let quantization = if operation == TensorOperation::Matrix {
        quantization_override.unwrap_or_else(|| args.weight_quantization_for(name))
    } else {
        None
    };
    output.extend(matrix_constraints(name, shape, quantization, Vec::new())?);
    Ok(())
}

fn matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
    aliases: Vec<String>,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    matrix_for_linear_format(name, aliases, shape, LinearFormat::from(quantization), None)
        .map_err(|error| error.to_string())
}

/// Builds the canonical Qwen GGUF catalog plan.
pub fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden_size")?;
    let vocab = dimension(args.vocab_size, "vocab_size")?;
    let head = dimension(args.head_dim, "head_dim")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "num_attention_heads")?,
        head,
        "query width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "num_key_value_heads")?,
        head,
        "key/value width",
    )?;
    let mut tensors = vec![
        gguf_tensor(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf_tensor("output_norm.weight", vec![hidden], TensorOperation::Vector),
    ];
    if !args.tie_word_embeddings {
        tensors.push(gguf_tensor(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ));
    }
    for layer in 0..dimension(args.num_hidden_layers, "num_hidden_layers")? {
        let block = format!("blk.{layer}");
        for (name, shape, operation) in [
            ("attn_norm.weight", vec![hidden], TensorOperation::Vector),
            ("ffn_norm.weight", vec![hidden], TensorOperation::Vector),
            (
                "attn_q.weight",
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_k.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_v.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_output.weight",
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
        ] {
            tensors.push(gguf_tensor(format!("{block}.{name}"), shape, operation));
        }
        if args.variant != QwenVariant::Qwen2 {
            for name in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                tensors.push(gguf_tensor(
                    format!("{block}.{name}"),
                    vec![head],
                    TensorOperation::Vector,
                ));
            }
        }
        if args.variant == QwenVariant::Qwen2 {
            for (name, size) in [
                ("attn_q.bias", query),
                ("attn_k.bias", key_value),
                ("attn_v.bias", key_value),
            ] {
                tensors.push(gguf_tensor(
                    format!("{block}.{name}"),
                    vec![size],
                    TensorOperation::Vector,
                ));
            }
        }
        if args.is_moe() {
            let experts = dimension(args.num_experts, "num_experts")?;
            let intermediate = dimension(args.moe_intermediate_size, "moe_intermediate_size")?;
            for (name, shape) in [
                ("ffn_gate_inp.weight", vec![experts, hidden]),
                ("ffn_gate_exps.weight", vec![experts, intermediate, hidden]),
                ("ffn_up_exps.weight", vec![experts, intermediate, hidden]),
                ("ffn_down_exps.weight", vec![experts, hidden, intermediate]),
            ] {
                tensors.push(gguf_tensor(
                    format!("{block}.{name}"),
                    shape,
                    TensorOperation::Matrix,
                ));
            }
        } else {
            let intermediate = dimension(args.intermediate_size, "intermediate_size")?;
            for (name, shape) in [
                ("ffn_gate.weight", vec![intermediate, hidden]),
                ("ffn_up.weight", vec![intermediate, hidden]),
                ("ffn_down.weight", vec![hidden, intermediate]),
            ] {
                tensors.push(gguf_tensor(
                    format!("{block}.{name}"),
                    shape,
                    TensorOperation::Matrix,
                ));
            }
        }
    }
    GgufCheckpointPlan::new(
        format!("{:?} GGUF", args.variant),
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
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
        .ok_or_else(|| format!("Qwen {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Qwen {name} geometry overflows"))
}

/// Translates one GGUF tensor name into the canonical Qwen parameter identity.
pub fn translate_gguf_weight_name(name: &str, is_moe: bool) -> String {
    const ROOTS: [(&str, &str); 3] = [
        ("token_embd", "model.embed_tokens"),
        ("output_norm", "model.norm"),
        ("output", "lm_head"),
    ];
    for (source, target) in ROOTS {
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
    if is_moe {
        for (source, target) in [
            ("ffn_gate_inp", "mlp.gate"),
            ("ffn_gate_exps", "mlp.experts.gate_proj"),
            ("ffn_up_exps", "mlp.experts.up_proj"),
            ("ffn_down_exps", "mlp.experts.down_proj"),
        ] {
            if parameter == source || parameter.starts_with(&format!("{source}.")) {
                let suffix = parameter.strip_prefix(source).unwrap_or_default();
                return format!("model.layers.{layer}.{target}{suffix}");
            }
        }
    }
    for (source, target) in [
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("ffn_up", "mlp.up_proj"),
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

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::qwen::model_args_from_config_value;
    use eredu_checkpoint::{
        expert::GatedProductExpertStorageLayout,
        recipe::DerivedWeightRecipe,
        store::{StoreError, TensorMetadata},
        StoredDtype,
    };

    struct Catalog(BTreeMap<String, TensorMetadata>);

    impl RecipeCatalog for Catalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
    }

    fn metadata(name: &str, shape: Vec<usize>) -> TensorMetadata {
        TensorMetadata {
            name: name.into(),
            encoded_byte_len: shape.iter().product::<usize>() as u64 * 2,
            logical_shape: shape.clone(),
            physical_shape: shape,
            stored_dtype: StoredDtype::F16,
            backing_shard: None,
        }
    }

    fn args(model_type: &str, tied: bool) -> ModelArgs {
        let mut value = serde_json::json!({
            "model_type": model_type,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "tie_word_embeddings": tied
        });
        if model_type == "qwen3_moe" {
            value["intermediate_size"] = 0.into();
            value["moe_intermediate_size"] = 8.into();
            value["num_experts"] = 4.into();
            value["num_experts_per_tok"] = 2.into();
            value["norm_topk_prob"] = true.into();
        }
        model_args_from_config_value(&value).unwrap()
    }

    #[test]
    fn plans_qwen2_biases_and_qwen3_norms() {
        let qwen2 = safetensors_plan(&args("qwen2", false)).unwrap();
        assert!(qwen2
            .common_tensors
            .iter()
            .any(|tensor| tensor.key.ends_with("self_attn.q_proj.bias")));
        let qwen3 = safetensors_plan(&args("qwen3", true)).unwrap();
        assert!(qwen3
            .common_tensors
            .iter()
            .any(|tensor| tensor.key.ends_with("self_attn.q_norm.weight")));
        assert!(!qwen3
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "lm_head.weight"));
    }

    #[test]
    fn moe_plan_admits_packed_separate_and_split_experts() {
        let plan = safetensors_plan(&args("qwen3_moe", false)).unwrap();
        let experts = plan
            .layout_groups
            .iter()
            .find(|group| group.id.contains("mlp.experts"))
            .unwrap();
        assert_eq!(
            experts
                .variants
                .iter()
                .map(|variant| variant.id.as_str())
                .collect::<Vec<_>>(),
            ["packed", "separate-packed", "split"]
        );
    }

    #[test]
    fn translates_dense_and_moe_gguf_names() {
        assert_eq!(
            translate_gguf_weight_name("blk.1.attn_q_norm.weight", false),
            "model.layers.1.self_attn.q_norm.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.1.ffn_gate_exps.weight", true),
            "model.layers.1.mlp.experts.gate_proj.weight"
        );
    }

    #[test]
    fn safetensors_parameter_ids_are_stable_for_qwen2_and_tied_qwen3() {
        let mut qwen2_args = args("qwen2", false);
        qwen2_args.num_hidden_layers = 1;
        let qwen2 = safetensors_plan(&qwen2_args).unwrap();
        let keys = qwen2
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            keys,
            BTreeSet::from([
                "lm_head.weight",
                "model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.mlp.down_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.layers.0.self_attn.k_proj.bias",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.0.self_attn.q_proj.bias",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.v_proj.bias",
                "model.layers.0.self_attn.v_proj.weight",
                "model.norm.weight",
            ])
        );

        let mut qwen3_args = args("qwen3", true);
        qwen3_args.num_hidden_layers = 1;
        let qwen3 = safetensors_plan(&qwen3_args).unwrap();
        assert!(!qwen3
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "lm_head.weight"));
        let tied = qwen3
            .layout_groups
            .iter()
            .find(|group| group.id == "redundant tied output head")
            .unwrap();
        assert!(!tied.required);
        assert_eq!(tied.variants[0].discriminator_keys, ["lm_head.weight"]);
    }

    #[test]
    fn quantized_moe_plan_has_atomic_companions_and_no_split_variant() {
        let mut args = args("qwen3_moe", false);
        args.hidden_size = 64;
        args.head_dim = 16;
        args.num_attention_heads = 4;
        args.num_key_value_heads = 2;
        args.vocab_size = 64;
        args.moe_intermediate_size = 64;
        args.num_hidden_layers = 1;
        args.quantization = Some(WeightQuantization::MxFp4);
        let plan = safetensors_plan(&args).unwrap();
        let query = plan
            .common_tensors
            .iter()
            .filter(|tensor| tensor.key.starts_with("model.layers.0.self_attn.q_proj"))
            .map(|tensor| tensor.key.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            query,
            [
                "model.layers.0.self_attn.q_proj.scales",
                "model.layers.0.self_attn.q_proj.weight",
            ]
        );
        let experts = plan
            .layout_groups
            .iter()
            .find(|group| group.id.ends_with("mlp.experts storage"))
            .unwrap();
        assert_eq!(
            experts
                .variants
                .iter()
                .map(|variant| variant.id.as_str())
                .collect::<Vec<_>>(),
            ["packed", "separate-packed"]
        );
        for variant in &experts.variants {
            assert!(variant
                .tensors
                .iter()
                .any(|tensor| tensor.key.ends_with("gate_up_proj.scales")
                    || tensor.key.ends_with("gate_proj.scales")));
            assert!(variant
                .tensors
                .iter()
                .any(|tensor| tensor.key.ends_with("down_proj.scales")));
        }
    }

    #[test]
    fn gguf_plan_golden_covers_qwen3_moe_catalog_ids() {
        let mut args = args("qwen3_moe", false);
        args.num_hidden_layers = 1;
        let plan = gguf_plan(&args).unwrap();
        let keys = plan
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<BTreeSet<_>>();
        for required in [
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
        ] {
            assert!(
                keys.contains(required),
                "missing GGUF golden key {required}"
            );
        }
        assert!(plan.layout_groups.is_empty());
    }

    #[test]
    fn split_expert_recipe_golden_preserves_targets_and_source_ids() {
        let args = args("qwen3_moe", false);
        let prefix = "model.layers.0.mlp.experts";
        let mut tensors = BTreeMap::new();
        for expert in 0..4 {
            for (projection, shape) in [
                ("gate_proj", vec![8, 16]),
                ("up_proj", vec![8, 16]),
                ("down_proj", vec![16, 8]),
            ] {
                let name = format!("{prefix}.{expert}.{projection}.weight");
                tensors.insert(name.clone(), metadata(&name, shape));
            }
        }
        let catalog = Catalog(tensors);
        let recipes = expert_recipes(&catalog, &args, "model", 0).unwrap();
        assert_eq!(recipes.layout, GatedProductExpertStorageLayout::Independent);
        assert_eq!(recipes.target_gate_up, format!("{prefix}.gate_up_proj"));
        assert_eq!(recipes.target_down, format!("{prefix}.down_proj"));
        assert_eq!(
            recipes.gate_up.infer(&catalog).unwrap().shape(),
            &[4, 16, 16]
        );
        let DerivedWeightRecipe::Stack { axis, inputs } = &recipes.gate_up else {
            panic!("split experts must stack canonical gate/up recipes")
        };
        assert_eq!(*axis, 0);
        assert_eq!(inputs.len(), 4);
        let DerivedWeightRecipe::Concatenate {
            axis,
            inputs: first,
        } = &inputs[0]
        else {
            panic!("each split expert must concatenate gate and up")
        };
        assert_eq!(*axis, 0);
        assert!(matches!(
            &first[0],
            DerivedWeightRecipe::Source { key, .. }
                if key == "model.layers.0.mlp.experts.0.gate_proj.weight"
        ));
        assert!(matches!(
            &first[1],
            DerivedWeightRecipe::Source { key, .. }
                if key == "model.layers.0.mlp.experts.0.up_proj.weight"
        ));
        let unit = expert_unit_recipes(&catalog, &args, 0, 2).unwrap();
        assert!(matches!(
            unit.get("gate_up_proj"),
            Some(DerivedWeightRecipe::Stack { axis: 0, inputs }) if inputs.len() == 1
        ));
        assert_eq!(
            unit["gate_up_proj"].infer(&catalog).unwrap().shape(),
            &[1, 16, 16]
        );
        assert_eq!(
            unit["down_proj"].infer(&catalog).unwrap().shape(),
            &[1, 16, 8]
        );
    }

    #[test]
    fn expert_unit_catalog_owns_separate_bank_packing() {
        let args = args("qwen3_moe", false);
        let prefix = "model.layers.0.mlp.experts";
        let catalog = Catalog(BTreeMap::from([
            (
                format!("{prefix}.gate_proj"),
                metadata(&format!("{prefix}.gate_proj"), vec![4, 8, 16]),
            ),
            (
                format!("{prefix}.up_proj"),
                metadata(&format!("{prefix}.up_proj"), vec![4, 8, 16]),
            ),
            (
                format!("{prefix}.down_proj"),
                metadata(&format!("{prefix}.down_proj"), vec![4, 16, 8]),
            ),
        ]));
        let recipes = expert_unit_recipes(&catalog, &args, 0, 2).unwrap();
        assert!(matches!(
            recipes.get("gate_up_proj"),
            Some(DerivedWeightRecipe::Concatenate { axis: 1, inputs })
                if inputs.len() == 2
        ));
        assert_eq!(
            recipes["gate_up_proj"].infer(&catalog).unwrap().shape(),
            &[1, 16, 16]
        );
    }
}
