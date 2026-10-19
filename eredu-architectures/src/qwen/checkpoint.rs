//! Pure checkpoint schemas, name translation, and recipes for Qwen text models.

use std::collections::{BTreeMap, HashMap};

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

/// Derives a Qwen configuration whose physical matrix formats reflect
/// load-time quantization.
pub fn load_time_quantization(
    args: &ModelArgs,
    quantization: WeightQuantization,
) -> Result<ModelArgs, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = args.clone();
    target.quantization = Some(quantization);
    target.quantized_weights = None;
    target.quantized_weight_configs = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical checkpoint format metadata to a complete Qwen
/// configuration.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    mut formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    if args.is_moe() {
        for layer in 0..args.num_hidden_layers {
            let prefix = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
            let gate = formats.get(&format!("{prefix}.gate_proj.weight"));
            let up = formats.get(&format!("{prefix}.up_proj.weight"));
            if (gate.is_some() || up.is_some()) && gate != up {
                return Err(format!("split expert gate/up formats disagree at {prefix}"));
            }
        }
    }
    normalize_weight_formats(args, &mut formats);
    let mut target = args.clone();
    target.quantized_weights = Some(formats.keys().cloned().collect());
    target.quantized_weight_configs = Some(formats);
    target.quantization = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

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
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.push("rope_freqs.".into());
    policy.allowed_suffixes.push(".rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new(
        format!("{} SafeTensors", args.model_type),
        common,
        groups,
        policy,
    )
    .map_err(|error| error.to_string())
}

/// Resolves packed, separate-packed, or per-expert sources into one canonical
/// bank under the normalized configuration's checkpoint root.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
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
    let prefix = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
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

/// Returns architecture-owned recipes for one rank-local Qwen expert bank.
///
/// Expert identity is resolved through canonical per-expert recipes and then
/// assembled in rank order, preserving packed companions and adapting to
/// packed, split, or independently stored sources.
pub fn rank_local_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
    layer: usize,
    group_indices: &[usize],
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    if group_indices.is_empty() {
        return Err("rank-local Qwen expert recipes require at least one expert".into());
    }
    let mut unique = std::collections::BTreeSet::new();
    let mut grouped = BTreeMap::<String, Vec<DerivedWeightRecipe>>::new();
    for &expert in group_indices {
        if !unique.insert(expert) {
            return Err(format!(
                "rank-local Qwen expert recipe contains duplicate expert {expert}"
            ));
        }
        for (binding, recipe) in expert_unit_recipes(catalog, args, layer, expert)? {
            grouped.entry(binding).or_default().push(recipe);
        }
    }
    let prefix = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
    grouped
        .into_iter()
        .map(|(binding, mut inputs)| {
            let recipe = if inputs.len() == 1 {
                inputs.pop().expect("one rank-local expert recipe")
            } else {
                DerivedWeightRecipe::Concatenate { axis: 0, inputs }
            };
            recipe.infer(catalog).map_err(|error| error.to_string())?;
            Ok((format!("{prefix}.{binding}"), recipe))
        })
        .collect()
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

/// Builds the complete architecture-owned schedule for independent expert residency.
pub fn expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &ModelArgs,
) -> Result<crate::ExpertResidencyCatalog, String> {
    if !args.is_moe() {
        return Err("Qwen expert residency requires Qwen3-MoE arguments".into());
    }
    let layers = dimension(args.num_hidden_layers, "num_hidden_layers")?;
    let experts = dimension(args.num_experts, "num_experts")?;
    let capacity = layers
        .checked_mul(experts)
        .ok_or_else(|| "Qwen expert residency catalog size overflowed".to_owned())?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new("text_decoder").map_err(|error| error.to_string())?;
    let mut units = Vec::with_capacity(capacity);
    for layer in 0..layers {
        let unit_path = format!("{}.layers.{layer}", args.parameter_root);
        let expert_root = format!("{unit_path}.mlp.experts");
        for expert in 0..experts {
            let recipes = expert_unit_recipes(catalog, args, layer, expert)?;
            let gate_up_quantizable = !recipes.contains_key("gate_up_proj_scales")
                && !recipes.contains_key("gate_up_proj_biases");
            let down_quantizable = !recipes.contains_key("down_proj_scales")
                && !recipes.contains_key("down_proj_biases");
            let parameters = recipes
                .into_iter()
                .map(|(binding, recipe)| {
                    let target = format!("{expert_root}.{binding}");
                    let role = match binding.as_str() {
                        "gate_up_proj" if gate_up_quantizable => {
                            crate::ExpertParameterRole::quantizable_projection(
                                "gate_up_proj_scales",
                                "gate_up_proj_biases",
                            )
                        }
                        "down_proj" if down_quantizable => {
                            crate::ExpertParameterRole::quantizable_projection(
                                "down_proj_scales",
                                "down_proj_biases",
                            )
                        }
                        _ => crate::ExpertParameterRole::Preserved,
                    };
                    crate::ExpertParameterRecipe::new(binding, target, recipe, role)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            units.push(
                crate::ExpertResidencyUnit::new(
                    eredu_runtime::ParameterBankKey::new(0, layer, expert),
                    owner_group.clone(),
                    layer,
                    &unit_path,
                    crate::ExpertResidencyDistribution::ExpertParallel,
                    parameters,
                )
                .map_err(|error| error.to_string())?,
            );
        }
    }
    crate::ExpertResidencyCatalog::new(units)
        .and_then(|residency| residency.with_inferred_byte_geometry(catalog))
        .map_err(|error| error.to_string())
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
    let mut policy = CatalogPolicy::non_strict();
    policy.allowed_prefixes.push("rope_freqs.".into());
    GgufCheckpointPlan::new(
        format!("{:?} GGUF", args.variant),
        tensors,
        Vec::new(),
        policy,
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

/// Rehomes split expert GGUF formats onto the canonical fused runtime weights.
pub fn normalize_weight_formats<V>(args: &ModelArgs, formats: &mut HashMap<String, V>) {
    if !args.is_moe() {
        return;
    }
    for layer in 0..args.num_hidden_layers {
        let prefix = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
        if let Some(format) = formats.remove(&format!("{prefix}.gate_proj.weight")) {
            formats.remove(&format!("{prefix}.up_proj.weight"));
            formats.insert(format!("{prefix}.gate_up_proj"), format);
        }
        if let Some(format) = formats.remove(&format!("{prefix}.down_proj.weight")) {
            formats.insert(format!("{prefix}.down_proj"), format);
        }
    }
}
