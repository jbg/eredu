//! SafeTensors contracts and fused-projection recipes for the hybrid decoder.

use std::collections::BTreeMap;

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, DerivedWeightRecipe, RecipeCatalog},
    schema::{
        matrix_for_linear_format, AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan,
        GgufTensorConstraint, GgufTypeConstraint, LayoutVariant, MatrixScaleNames,
        SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint,
        TensorOperation,
    },
    store::{CheckpointSource, TensorSelection, WeightStoreBackend},
};

/// Returns all derived recipes owned by Qwen hybrid static modules.
pub fn static_recipes(
    store: &dyn CheckpointSource,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let mut recipes = BTreeMap::new();
    let patch0 = "model.visual.patch_embed.proj.weight.0";
    let patch1 = "model.visual.patch_embed.proj.weight.1";
    if store.source_metadata(patch0).is_ok() && store.source_metadata(patch1).is_ok() {
        recipes.insert(
            "model.visual.patch_embed.proj.weight".into(),
            DerivedWeightRecipe::Stack {
                axis: 2,
                inputs: vec![
                    DerivedWeightRecipe::source(patch0, TensorSelection::Full),
                    DerivedWeightRecipe::source(patch1, TensorSelection::Full),
                ],
            },
        );
    }
    if store
        .source_diagnostics()
        .map_err(|error| error.to_string())?
        .backend
        == WeightStoreBackend::Gguf
    {
        let name = "model.norm.weight";
        if store.source_metadata(name).is_ok() {
            recipes.insert(
                name.into(),
                DerivedWeightRecipe::SubtractOne {
                    input: Box::new(DerivedWeightRecipe::source(name, TensorSelection::Full)),
                },
            );
        }
    }
    Ok(recipes)
}

/// Returns the complete derived-weight catalog for one target or MTP unit.
pub fn unit_recipes(
    store: &dyn CheckpointSource,
    config: &HybridConfig,
    flat: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let target_layers = usize::try_from(config.num_hidden_layers)
        .map_err(|_| "invalid Qwen hybrid layer count".to_string())?;
    let mut recipes = BTreeMap::new();
    if flat < target_layers
        && config.variant == HybridVariant::Qwen3Next
        && matches!(
            config.layer_schedule.get(flat),
            Some(HybridLayerPolicy::LinearAttention)
        )
    {
        let fused = format!("model.layers.{flat}.linear_attn.in_proj_qkvz.weight");
        if store.tensor_metadata(&fused).is_ok() {
            recipes.extend(
                qwen3_next_fused_recipes(store, config, flat)?
                    .iter()
                    .map(|(name, recipe)| (name.to_owned(), recipe.clone())),
            );
        }
    }
    if store
        .source_diagnostics()
        .map_err(|error| error.to_string())?
        .backend
        == WeightStoreBackend::Gguf
    {
        add_gguf_unit_transforms(&mut recipes, store, config, flat)?;
    }
    if !config.is_moe() {
        return Ok(recipes);
    }
    let root = if flat < target_layers {
        format!("model.layers.{flat}.mlp.experts")
    } else {
        format!("mtp.layers.{}.mlp.experts", flat - target_layers)
    };
    let gate_up = format!("{root}.gate_up_proj");
    if store.tensor_metadata(&gate_up).is_err() {
        let gate = format!("{root}.gate_proj");
        let up = format!("{root}.up_proj");
        if store.tensor_metadata(&gate).is_ok() && store.tensor_metadata(&up).is_ok() {
            recipes.insert(
                gate_up,
                DerivedWeightRecipe::Concatenate {
                    axis: 1,
                    inputs: vec![
                        DerivedWeightRecipe::source(gate, TensorSelection::Full),
                        DerivedWeightRecipe::source(up, TensorSelection::Full),
                    ],
                },
            );
        } else {
            let inputs = (0..config.num_experts)
                .map(|expert| DerivedWeightRecipe::Concatenate {
                    axis: 0,
                    inputs: vec![
                        DerivedWeightRecipe::source(
                            format!("{root}.{expert}.gate_proj.weight"),
                            TensorSelection::Full,
                        ),
                        DerivedWeightRecipe::source(
                            format!("{root}.{expert}.up_proj.weight"),
                            TensorSelection::Full,
                        ),
                    ],
                })
                .collect();
            recipes.insert(gate_up, DerivedWeightRecipe::Stack { axis: 0, inputs });
        }
    }
    let down = format!("{root}.down_proj");
    if store.tensor_metadata(&down).is_err() {
        recipes.insert(
            down,
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: (0..config.num_experts)
                    .map(|expert| {
                        DerivedWeightRecipe::source(
                            format!("{root}.{expert}.down_proj.weight"),
                            TensorSelection::Full,
                        )
                    })
                    .collect(),
            },
        );
    }
    Ok(recipes)
}

/// Returns canonical lazy-loading recipes for one routed target or MTP expert.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &HybridConfig,
    layer: usize,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    if !config.is_moe() {
        return Err("independent expert recipes require a routed Qwen hybrid model".into());
    }
    let target = config.num_hidden_layers as usize;
    let total = target
        .checked_add(config.mtp_num_hidden_layers as usize)
        .ok_or_else(|| "Qwen hybrid layer count overflowed".to_string())?;
    if layer >= total {
        return Err(format!("Qwen hybrid has no routed layer {layer}"));
    }
    if expert >= config.num_experts as usize {
        return Err(format!(
            "Qwen hybrid has no expert {expert} in layer {layer}"
        ));
    }
    let root = if layer < target {
        format!("model.layers.{layer}.mlp.experts")
    } else {
        format!("mtp.layers.{}.mlp.experts", layer - target)
    };
    let selection = TensorSelection::Range {
        axis: 0,
        start: expert,
        end: expert + 1,
    };
    let packed = catalog
        .tensor_metadata(&format!("{root}.gate_up_proj"))
        .is_ok();
    let split_banks = ["gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .all(|name| catalog.tensor_metadata(&format!("{root}.{name}")).is_ok());
    let mut recipes = BTreeMap::new();
    if packed {
        for (target_name, required) in [
            ("gate_up_proj", true),
            ("gate_up_proj_scale_inv", false),
            ("gate_up_proj_scales", false),
            ("gate_up_proj_biases", false),
            ("down_proj", true),
            ("down_proj_scale_inv", false),
            ("down_proj_scales", false),
            ("down_proj_biases", false),
        ] {
            let source = format!("{root}.{target_name}");
            if catalog.tensor_metadata(&source).is_err() {
                if required {
                    return Err(format!("missing packed Qwen hybrid expert tensor {source}"));
                }
                continue;
            }
            recipes.insert(
                target_name.into(),
                DerivedWeightRecipe::source(source, selection.clone()),
            );
        }
    } else if split_banks {
        recipes.insert(
            "gate_up_proj".into(),
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: ["gate_proj", "up_proj"]
                    .into_iter()
                    .map(|name| {
                        DerivedWeightRecipe::source(format!("{root}.{name}"), selection.clone())
                    })
                    .collect(),
            },
        );
        recipes.insert(
            "down_proj".into(),
            DerivedWeightRecipe::source(format!("{root}.down_proj"), selection.clone()),
        );
        for suffix in ["scales", "biases", "scale_inv"] {
            let gate = format!("{root}.gate_proj_{suffix}");
            let up = format!("{root}.up_proj_{suffix}");
            if catalog.tensor_metadata(&gate).is_ok() && catalog.tensor_metadata(&up).is_ok() {
                recipes.insert(
                    format!("gate_up_proj_{suffix}"),
                    DerivedWeightRecipe::Concatenate {
                        axis: 1,
                        inputs: vec![
                            DerivedWeightRecipe::source(gate, selection.clone()),
                            DerivedWeightRecipe::source(up, selection.clone()),
                        ],
                    },
                );
            }
            let down = format!("{root}.down_proj_{suffix}");
            if catalog.tensor_metadata(&down).is_ok() {
                recipes.insert(
                    format!("down_proj_{suffix}"),
                    DerivedWeightRecipe::source(down, selection.clone()),
                );
            }
        }
    } else {
        let projection = |names: &[&str], suffix: &str| {
            names
                .iter()
                .map(|name| format!("{root}.{expert}.{name}.{suffix}"))
                .find(|name| catalog.tensor_metadata(name).is_ok())
                .map(|name| DerivedWeightRecipe::source(name, TensorSelection::Full))
                .ok_or_else(|| {
                    format!("missing split Qwen hybrid expert {expert} tensor under {root}")
                })
        };
        let gate = projection(&["gate_proj", "w1"], "weight")?;
        let up = projection(&["up_proj", "w3"], "weight")?;
        let down = projection(&["down_proj", "w2"], "weight")?;
        recipes.insert(
            "gate_up_proj".into(),
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: vec![DerivedWeightRecipe::Concatenate {
                    axis: 0,
                    inputs: vec![gate, up],
                }],
            },
        );
        recipes.insert(
            "down_proj".into(),
            DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: vec![down],
            },
        );
        if let (Ok(gate), Ok(up), Ok(down)) = (
            projection(&["gate_proj", "w1"], "weight_scale_inv"),
            projection(&["up_proj", "w3"], "weight_scale_inv"),
            projection(&["down_proj", "w2"], "weight_scale_inv"),
        ) {
            recipes.insert(
                "gate_up_proj_scale_inv".into(),
                DerivedWeightRecipe::Stack {
                    axis: 0,
                    inputs: vec![DerivedWeightRecipe::Concatenate {
                        axis: 0,
                        inputs: vec![gate, up],
                    }],
                },
            );
            recipes.insert(
                "down_proj_scale_inv".into(),
                DerivedWeightRecipe::Stack {
                    axis: 0,
                    inputs: vec![down],
                },
            );
        }
    }
    Ok(recipes)
}

/// Builds the complete architecture-owned schedule for independently resident experts.
pub fn expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &HybridConfig,
) -> Result<crate::ExpertResidencyCatalog, String> {
    config.validate().map_err(|error| error.to_string())?;
    if !config.is_moe() {
        return Err("Qwen hybrid expert residency requires a routed model".into());
    }
    let target = usize::try_from(config.num_hidden_layers)
        .map_err(|_| "invalid Qwen hybrid target layer count".to_string())?;
    let prediction = usize::try_from(config.mtp_num_hidden_layers)
        .map_err(|_| "invalid Qwen hybrid MTP layer count".to_string())?;
    let experts = usize::try_from(config.num_experts)
        .map_err(|_| "invalid Qwen hybrid expert count".to_string())?;
    let total = target
        .checked_add(prediction)
        .ok_or_else(|| "Qwen hybrid layer count overflowed".to_string())?;
    let capacity = total
        .checked_mul(experts)
        .ok_or_else(|| "Qwen hybrid expert residency catalog size overflowed".to_string())?;
    let mut units = Vec::with_capacity(capacity);
    for layer in 0..total {
        let (owner_group, owner_unit, unit_path) = if layer < target {
            ("target".to_owned(), layer, format!("model.layers.{layer}"))
        } else {
            let depth = layer - target;
            (format!("mtp.{depth}"), 0, format!("mtp.layers.{depth}"))
        };
        let owner_group =
            eredu_runtime::ExecutionGroupId::new(owner_group).map_err(|error| error.to_string())?;
        let expert_root = format!("{unit_path}.mlp.experts");
        for expert in 0..experts {
            let recipes = expert_recipes(catalog, config, layer, expert)?;
            let gate_up_quantizable = !recipes.contains_key("gate_up_proj_scales")
                && !recipes.contains_key("gate_up_proj_biases")
                && !recipes.contains_key("gate_up_proj_scale_inv");
            let down_quantizable = !recipes.contains_key("down_proj_scales")
                && !recipes.contains_key("down_proj_biases")
                && !recipes.contains_key("down_proj_scale_inv");
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
                    eredu_runtime::ExpertIdentity::new(layer, expert),
                    owner_group.clone(),
                    owner_unit,
                    &unit_path,
                    crate::ExpertResidencyDistribution::ExpertParallel,
                    parameters,
                )
                .map_err(|error| error.to_string())?,
            );
        }
    }
    crate::ExpertResidencyCatalog::new(units).map_err(|error| error.to_string())
}

fn add_gguf_unit_transforms(
    recipes: &mut BTreeMap<String, DerivedWeightRecipe>,
    store: &dyn CheckpointSource,
    config: &HybridConfig,
    flat: usize,
) -> Result<(), String> {
    let target_layers = usize::try_from(config.num_hidden_layers)
        .map_err(|_| "invalid Qwen hybrid layer count".to_string())?;
    let root = if flat < target_layers {
        format!("model.layers.{flat}")
    } else {
        format!("mtp.layers.{}", flat - target_layers)
    };
    for suffix in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ] {
        let name = format!("{root}.{suffix}");
        if store.source_metadata(&name).is_ok() {
            recipes.insert(
                name.clone(),
                DerivedWeightRecipe::SubtractOne {
                    input: Box::new(DerivedWeightRecipe::source(name, TensorSelection::Full)),
                },
            );
        }
    }
    let a_log = format!("{root}.linear_attn.A_log");
    if store.source_metadata(&a_log).is_ok() {
        recipes.insert(
            a_log.clone(),
            DerivedWeightRecipe::NegLog {
                input: Box::new(DerivedWeightRecipe::source(a_log, TensorSelection::Full)),
            },
        );
    }
    let shared_gate = format!("{root}.mlp.shared_expert_gate.weight");
    if let Ok(metadata) = store.source_metadata(&shared_gate) {
        if metadata.logical_shape.len() == 1 {
            recipes.insert(
                shared_gate.clone(),
                DerivedWeightRecipe::Reshape {
                    input: Box::new(DerivedWeightRecipe::source(
                        shared_gate,
                        TensorSelection::Full,
                    )),
                    shape: vec![1, metadata.logical_shape[0]],
                },
            );
        }
    }
    let conv = format!("{root}.linear_attn.conv1d.weight");
    if let Ok(metadata) = store.source_metadata(&conv) {
        if metadata.logical_shape.len() == 2 {
            let source = DerivedWeightRecipe::source(conv.clone(), TensorSelection::Full);
            let source = if config.variant == HybridVariant::Qwen3Next {
                source
            } else {
                qwen35_value_head_recipe(
                    "linear_attn.in_proj_qkv.weight",
                    source,
                    &metadata.logical_shape,
                    config,
                )?
                .unwrap_or_else(|| DerivedWeightRecipe::source(&conv, TensorSelection::Full))
            };
            recipes.insert(
                conv.clone(),
                DerivedWeightRecipe::Reshape {
                    input: Box::new(source),
                    shape: vec![metadata.logical_shape[0], 1, metadata.logical_shape[1]],
                },
            );
        }
    }
    if config.variant == HybridVariant::Qwen3Next {
        return Ok(());
    }
    for suffix in [
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.dt_bias",
        "linear_attn.A_log",
        "linear_attn.out_proj.weight",
    ] {
        let name = format!("{root}.{suffix}");
        let Ok(metadata) = store.source_metadata(&name) else {
            continue;
        };
        let base = recipes
            .remove(&name)
            .unwrap_or_else(|| DerivedWeightRecipe::source(name.clone(), TensorSelection::Full));
        if let Some(recipe) =
            qwen35_value_head_recipe(suffix, base, &metadata.logical_shape, config)?
        {
            recipes.insert(name, recipe);
        }
    }
    Ok(())
}

fn qwen35_value_head_recipe(
    suffix: &str,
    recipe: DerivedWeightRecipe,
    shape: &[usize],
    config: &HybridConfig,
) -> Result<Option<DerivedWeightRecipe>, String> {
    let num_k = usize::try_from(config.linear_num_key_heads)
        .map_err(|_| "invalid Qwen3.5 key-head count".to_string())?;
    let num_v = usize::try_from(config.linear_num_value_heads)
        .map_err(|_| "invalid Qwen3.5 value-head count".to_string())?;
    if num_k == 0 || num_v == 0 || num_v % num_k != 0 {
        return Err("invalid Qwen3.5 value-head grouping".into());
    }
    let repeats = num_v / num_k;
    let reorder =
        |input: DerivedWeightRecipe, axis: usize, head_width: usize, original: Vec<usize>| {
            let mut expanded = original.clone();
            expanded.splice(axis..=axis, [repeats, num_k, head_width]);
            let mut axes = (0..expanded.len()).collect::<Vec<_>>();
            axes.swap(axis, axis + 1);
            DerivedWeightRecipe::Reshape {
                input: Box::new(DerivedWeightRecipe::Transpose {
                    input: Box::new(DerivedWeightRecipe::Reshape {
                        input: Box::new(input),
                        shape: expanded,
                    }),
                    axes,
                }),
                shape: original,
            }
        };
    if suffix.ends_with("in_proj_qkv.weight") {
        if shape.len() != 2 {
            return Ok(None);
        }
        let prefix = 2usize
            .checked_mul(num_k)
            .and_then(|value| value.checked_mul(config.linear_key_head_dim as usize))
            .ok_or_else(|| "Qwen3.5 value-tail width overflow".to_string())?;
        if prefix >= shape[0] || !(shape[0] - prefix).is_multiple_of(num_v) {
            return Ok(None);
        }
        let leading = DerivedWeightRecipe::Select {
            input: Box::new(recipe.clone()),
            selection: TensorSelection::Range {
                axis: 0,
                start: 0,
                end: prefix,
            },
        };
        let tail = DerivedWeightRecipe::Select {
            input: Box::new(recipe),
            selection: TensorSelection::Range {
                axis: 0,
                start: prefix,
                end: shape[0],
            },
        };
        return Ok(Some(DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![
                leading,
                reorder(
                    tail,
                    0,
                    (shape[0] - prefix) / num_v,
                    vec![shape[0] - prefix, shape[1]],
                ),
            ],
        }));
    }
    let axis = if suffix.ends_with("out_proj.weight") {
        1
    } else {
        0
    };
    if axis >= shape.len() || !shape[axis].is_multiple_of(num_v) {
        return Ok(None);
    }
    let admitted = suffix.ends_with("in_proj_z.weight")
        || suffix.ends_with("in_proj_a.weight")
        || suffix.ends_with("in_proj_b.weight")
        || suffix.ends_with("dt_bias")
        || suffix.ends_with("A_log")
        || suffix.ends_with("out_proj.weight");
    Ok(admitted.then(|| reorder(recipe, axis, shape[axis] / num_v, shape.to_vec())))
}

use super::{
    fp8_block_row_widths, fused_projection_widths, HybridConfig, HybridLayerPolicy, HybridVariant,
};

/// Translates one hybrid projector GGUF name, including split patch weights.
pub fn translate_vision_gguf_weight_name(name: &str, deepstack: &[i32]) -> String {
    match name {
        "v.patch_embd.weight" => "model.visual.patch_embed.proj.weight.0".into(),
        "v.patch_embd.weight.1" => "model.visual.patch_embed.proj.weight.1".into(),
        _ => crate::qwen::vision::translate_gguf_weight_name(name, deepstack),
    }
}

/// Builds the strict hybrid text SafeTensors catalog.
pub fn safetensors_plan(config: &HybridConfig) -> Result<SafetensorsCheckpointPlan, String> {
    config.validate().map_err(|error| error.to_string())?;
    let hidden = dim(config.hidden_size, "hidden_size")?;
    let vocab = dim(config.vocab_size, "vocab_size")?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    matrix(
        config,
        &mut common,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
    )?;
    vector(&mut common, "model.norm.weight", hidden);
    if !config.tie_word_embeddings {
        matrix(config, &mut common, "lm_head.weight", vec![vocab, hidden])?;
    }
    for (layer, policy) in config.layer_schedule.iter().copied().enumerate() {
        add_block(config, layer, policy, &mut common, &mut groups)?;
    }
    let mtp_layers = usize::try_from(config.mtp_num_hidden_layers)
        .map_err(|_| "mtp_num_hidden_layers must be non-negative")?;
    if mtp_layers > 0 {
        vector(&mut common, "mtp.pre_fc_norm_hidden.weight", hidden);
        vector(&mut common, "mtp.pre_fc_norm_embedding.weight", hidden);
        common.extend(
            matrix_for_linear_format(
                "mtp.fc.weight",
                Vec::<String>::new(),
                vec![hidden, mul(2, hidden)?],
                eredu_checkpoint::LinearFormat::Dense,
                None,
            )
            .map_err(|error| error.to_string())?,
        );
        vector(&mut common, "mtp.norm.weight", hidden);
        for depth in 0..mtp_layers {
            add_block_at(
                config,
                config.num_hidden_layers as usize + depth,
                &format!("mtp.layers.{depth}"),
                HybridLayerPolicy::SelfAttention(eredu_core::attention::AttentionPolicy::Full),
                &mut common,
                &mut groups,
            )?;
        }
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_suffixes.push("rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new(
        format!("{} SafeTensors", config.model_type),
        common,
        groups,
        policy,
    )
    .map_err(|error| error.to_string())
}

/// Builds the canonical llama.cpp Qwen3-Next/Qwen3.5 tensor catalog.
pub fn gguf_plan(config: &HybridConfig) -> Result<GgufCheckpointPlan, String> {
    config.validate().map_err(|error| error.to_string())?;
    let hidden = dim(config.hidden_size, "hidden_size")?;
    let vocab = dim(config.vocab_size, "vocab_size")?;
    let query = mul(
        dim(config.num_attention_heads, "num_attention_heads")?,
        dim(config.head_dim, "head_dim")?,
    )?;
    let key_value = mul(
        dim(config.num_key_value_heads, "num_key_value_heads")?,
        dim(config.head_dim, "head_dim")?,
    )?;
    let key = mul(
        dim(config.linear_num_key_heads, "linear_num_key_heads")?,
        dim(config.linear_key_head_dim, "linear_key_head_dim")?,
    )?;
    let value = mul(
        dim(config.linear_num_value_heads, "linear_num_value_heads")?,
        dim(config.linear_value_head_dim, "linear_value_head_dim")?,
    )?;
    let value_heads = dim(config.linear_num_value_heads, "linear_num_value_heads")?;
    let mut tensors = vec![
        gguf_tensor(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf_tensor("output_norm.weight", vec![hidden], TensorOperation::Vector),
    ];
    if !config.tie_word_embeddings {
        tensors.push(gguf_tensor(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ));
    }
    let mut groups = Vec::new();
    for (layer, policy) in config.layer_schedule.iter().copied().enumerate() {
        let root = format!("blk.{layer}");
        tensors.extend([
            gguf_tensor(
                format!("{root}.attn_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
            gguf_tensor(
                format!("{root}.post_attention_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
        ]);
        match policy {
            HybridLayerPolicy::SelfAttention(_) => {
                tensors.extend([
                    gguf_tensor(
                        format!("{root}.attn_q.weight"),
                        vec![mul(2, query)?, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf_tensor(
                        format!("{root}.attn_k.weight"),
                        vec![key_value, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf_tensor(
                        format!("{root}.attn_v.weight"),
                        vec![key_value, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf_tensor(
                        format!("{root}.attn_output.weight"),
                        vec![hidden, query],
                        TensorOperation::Matrix,
                    ),
                    gguf_tensor(
                        format!("{root}.attn_q_norm.weight"),
                        vec![dim(config.head_dim, "head_dim")?],
                        TensorOperation::Vector,
                    ),
                    gguf_tensor(
                        format!("{root}.attn_k_norm.weight"),
                        vec![dim(config.head_dim, "head_dim")?],
                        TensorOperation::Vector,
                    ),
                ]);
                if config.attention_bias {
                    tensors.extend([
                        gguf_tensor(
                            format!("{root}.attn_q.bias"),
                            vec![mul(2, query)?],
                            TensorOperation::Vector,
                        ),
                        gguf_tensor(
                            format!("{root}.attn_k.bias"),
                            vec![key_value],
                            TensorOperation::Vector,
                        ),
                        gguf_tensor(
                            format!("{root}.attn_v.bias"),
                            vec![key_value],
                            TensorOperation::Vector,
                        ),
                        gguf_tensor(
                            format!("{root}.attn_output.bias"),
                            vec![hidden],
                            TensorOperation::Vector,
                        ),
                    ]);
                }
            }
            HybridLayerPolicy::LinearAttention => {
                if config.variant == HybridVariant::Qwen3Next {
                    tensors.extend([
                        gguf_tensor(
                            format!("{root}.attn_qkvz.weight"),
                            vec![add(mul(2, key)?, mul(2, value)?)?, hidden],
                            TensorOperation::Matrix,
                        ),
                        gguf_tensor(
                            format!("{root}.ssm_ba.weight"),
                            vec![mul(2, value_heads)?, hidden],
                            TensorOperation::Matrix,
                        ),
                    ]);
                } else {
                    tensors.extend([
                        gguf_tensor(
                            format!("{root}.attn_qkv.weight"),
                            vec![add(mul(2, key)?, value)?, hidden],
                            TensorOperation::Matrix,
                        ),
                        gguf_tensor(
                            format!("{root}.attn_gate.weight"),
                            vec![value, hidden],
                            TensorOperation::Matrix,
                        ),
                        gguf_tensor(
                            format!("{root}.ssm_beta.weight"),
                            vec![value_heads, hidden],
                            TensorOperation::Matrix,
                        ),
                        gguf_tensor(
                            format!("{root}.ssm_alpha.weight"),
                            vec![value_heads, hidden],
                            TensorOperation::Matrix,
                        ),
                    ]);
                }
                tensors.extend([
                    gguf_tensor(
                        format!("{root}.ssm_conv1d.weight"),
                        vec![
                            add(mul(2, key)?, value)?,
                            dim(config.linear_conv_kernel_dim, "linear_conv_kernel_dim")?,
                        ],
                        TensorOperation::Dense,
                    ),
                    gguf_tensor(
                        format!("{root}.ssm_dt.bias"),
                        vec![value_heads],
                        TensorOperation::Vector,
                    ),
                    gguf_tensor(
                        format!("{root}.ssm_a"),
                        vec![value_heads],
                        TensorOperation::Dense,
                    ),
                    gguf_tensor(
                        format!("{root}.ssm_norm.weight"),
                        vec![dim(config.linear_value_head_dim, "linear_value_head_dim")?],
                        TensorOperation::Vector,
                    ),
                    gguf_tensor(
                        format!("{root}.ssm_out.weight"),
                        vec![hidden, value],
                        TensorOperation::Matrix,
                    ),
                ]);
            }
        }
        if config.is_moe() {
            let experts = dim(config.num_experts, "num_experts")?;
            let intermediate = dim(config.moe_intermediate_size, "moe_intermediate_size")?;
            let shared = dim(
                config.shared_expert_intermediate_size,
                "shared_expert_intermediate_size",
            )?;
            tensors.extend([
                gguf_tensor(
                    format!("{root}.ffn_gate_inp.weight"),
                    vec![experts, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_gate_shexp.weight"),
                    vec![shared, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_up_shexp.weight"),
                    vec![shared, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_down_shexp.weight"),
                    vec![hidden, shared],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_gate_exps.weight"),
                    vec![experts, intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_up_exps.weight"),
                    vec![experts, intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_down_exps.weight"),
                    vec![experts, hidden, intermediate],
                    TensorOperation::Matrix,
                ),
            ]);
            let shared_gate = format!("{root}.ffn_gate_inp_shexp.weight");
            groups.push(AlternativeLayoutGroup {
                id: format!("{root} shared-expert gate rank"),
                required: true,
                variants: vec![
                    LayoutVariant {
                        id: "matrix".into(),
                        tensors: vec![gguf_tensor(
                            shared_gate.clone(),
                            vec![1, hidden],
                            TensorOperation::Matrix,
                        )],
                        discriminator_keys: vec![shared_gate.clone()],
                    },
                    LayoutVariant {
                        id: "vector".into(),
                        tensors: vec![gguf_tensor(
                            shared_gate.clone(),
                            vec![hidden],
                            TensorOperation::Vector,
                        )],
                        discriminator_keys: vec![shared_gate],
                    },
                ],
            });
        } else {
            let intermediate = dim(config.intermediate_size, "intermediate_size")?;
            tensors.extend([
                gguf_tensor(
                    format!("{root}.ffn_gate.weight"),
                    vec![intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_up.weight"),
                    vec![intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{root}.ffn_down.weight"),
                    vec![hidden, intermediate],
                    TensorOperation::Matrix,
                ),
            ]);
        }
    }
    let mut catalog = CatalogPolicy::non_strict();
    catalog.allowed_prefixes.push("rope_freqs.".into());
    GgufCheckpointPlan::new(
        format!("{} GGUF", config.model_type),
        tensors,
        groups,
        catalog,
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

/// Translates llama.cpp Qwen hybrid tensor identities to canonical parameters.
pub fn translate_gguf_weight_name(name: &str) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("output_norm", "model.norm"),
        ("output", "lm_head"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    let Some((layer, parameter)) = name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
    else {
        return name.to_owned();
    };
    for (source, target) in [
        ("attn_norm", "input_layernorm"),
        ("post_attention_norm", "post_attention_layernorm"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_qkvz", "linear_attn.in_proj_qkvz"),
        ("attn_qkv", "linear_attn.in_proj_qkv"),
        ("attn_gate", "linear_attn.in_proj_z"),
        ("ssm_beta", "linear_attn.in_proj_b"),
        ("ssm_alpha", "linear_attn.in_proj_a"),
        ("ssm_ba", "linear_attn.in_proj_ba"),
        ("ssm_conv1d", "linear_attn.conv1d"),
        ("ssm_dt.bias", "linear_attn.dt_bias"),
        ("ssm_a", "linear_attn.A_log"),
        ("ssm_norm", "linear_attn.norm"),
        ("ssm_out", "linear_attn.out_proj"),
        ("ffn_gate_inp_shexp", "mlp.shared_expert_gate"),
        ("ffn_gate_inp", "mlp.gate"),
        ("ffn_gate_shexp", "mlp.shared_expert.gate_proj"),
        ("ffn_up_shexp", "mlp.shared_expert.up_proj"),
        ("ffn_down_shexp", "mlp.shared_expert.down_proj"),
        ("ffn_gate_exps", "mlp.experts.gate_proj"),
        ("ffn_up_exps", "mlp.experts.up_proj"),
        ("ffn_down_exps", "mlp.experts.down_proj"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("rope_freqs", "rope_freqs"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            let mut suffix = parameter.strip_prefix(source).unwrap_or_default();
            if target.starts_with("mlp.experts.") {
                suffix = match suffix {
                    ".weight" => "",
                    ".scales" => "_scales",
                    ".biases" => "_biases",
                    other => other,
                };
            }
            return format!("model.layers.{layer}.{target}{suffix}");
        }
    }
    name.to_owned()
}

fn add_block(
    config: &HybridConfig,
    layer: usize,
    policy: HybridLayerPolicy,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    add_block_at(
        config,
        layer,
        &format!("model.layers.{layer}"),
        policy,
        common,
        groups,
    )
}

fn add_block_at(
    config: &HybridConfig,
    _layer: usize,
    root: &str,
    policy: HybridLayerPolicy,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let hidden = dim(config.hidden_size, "hidden_size")?;
    vector(common, format!("{root}.input_layernorm.weight"), hidden);
    vector(
        common,
        format!("{root}.post_attention_layernorm.weight"),
        hidden,
    );
    match policy {
        HybridLayerPolicy::SelfAttention(_) => add_attention(config, root, common)?,
        HybridLayerPolicy::LinearAttention => add_linear_attention(config, root, common, groups)?,
    }
    add_feed_forward(config, root, common, groups)
}

fn add_attention(
    config: &HybridConfig,
    root: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let hidden = dim(config.hidden_size, "hidden_size")?;
    let head = dim(config.head_dim, "head_dim")?;
    let query = mul(
        dim(config.num_attention_heads, "num_attention_heads")?,
        head,
    )?;
    let kv = mul(
        dim(config.num_key_value_heads, "num_key_value_heads")?,
        head,
    )?;
    let prefix = format!("{root}.self_attn");
    for (field, shape) in [
        ("q_proj", vec![mul(2, query)?, hidden]),
        ("k_proj", vec![kv, hidden]),
        ("v_proj", vec![kv, hidden]),
        ("o_proj", vec![hidden, query]),
    ] {
        matrix(config, common, format!("{prefix}.{field}.weight"), shape)?;
    }
    vector(common, format!("{prefix}.q_norm.weight"), head);
    vector(common, format!("{prefix}.k_norm.weight"), head);
    if config.attention_bias {
        for (field, width) in [
            ("q_proj", mul(2, query)?),
            ("k_proj", kv),
            ("v_proj", kv),
            ("o_proj", hidden),
        ] {
            vector(common, format!("{prefix}.{field}.bias"), width);
        }
    }
    Ok(())
}

fn add_linear_attention(
    config: &HybridConfig,
    root: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let hidden = dim(config.hidden_size, "hidden_size")?;
    let key = mul(
        dim(config.linear_num_key_heads, "linear_num_key_heads")?,
        dim(config.linear_key_head_dim, "linear_key_head_dim")?,
    )?;
    let value = mul(
        dim(config.linear_num_value_heads, "linear_num_value_heads")?,
        dim(config.linear_value_head_dim, "linear_value_head_dim")?,
    )?;
    let heads = dim(config.linear_num_value_heads, "linear_num_value_heads")?;
    let prefix = format!("{root}.linear_attn");
    let split = layout(
        config,
        "split",
        [
            (
                format!("{prefix}.in_proj_qkv.weight"),
                vec![add(mul(2, key)?, value)?, hidden],
            ),
            (format!("{prefix}.in_proj_z.weight"), vec![value, hidden]),
            (format!("{prefix}.in_proj_b.weight"), vec![heads, hidden]),
            (format!("{prefix}.in_proj_a.weight"), vec![heads, hidden]),
        ],
    )?;
    let mut variants = vec![split];
    if config.variant == HybridVariant::Qwen3Next {
        variants.push(layout(
            config,
            "fused",
            [
                (
                    format!("{prefix}.in_proj_qkvz.weight"),
                    vec![add(mul(2, key)?, mul(2, value)?)?, hidden],
                ),
                (
                    format!("{prefix}.in_proj_ba.weight"),
                    vec![mul(2, heads)?, hidden],
                ),
            ],
        )?);
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("{root} linear-attention inputs"),
        required: true,
        variants,
    });
    common.push(SafetensorsTensorConstraint::required(
        format!("{prefix}.conv1d.weight"),
        vec![
            add(mul(2, key)?, value)?,
            1,
            dim(config.linear_conv_kernel_dim, "linear_conv_kernel_dim")?,
        ],
        StoredDtypeConstraint::Floating,
    ));
    vector(common, format!("{prefix}.dt_bias"), heads);
    vector(common, format!("{prefix}.A_log"), heads);
    vector(
        common,
        format!("{prefix}.norm.weight"),
        dim(config.linear_value_head_dim, "linear_value_head_dim")?,
    );
    matrix(
        config,
        common,
        format!("{prefix}.out_proj.weight"),
        vec![hidden, value],
    )
}

fn add_feed_forward(
    config: &HybridConfig,
    root: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let hidden = dim(config.hidden_size, "hidden_size")?;
    let prefix = format!("{root}.mlp");
    if !config.is_moe() {
        let intermediate = dim(config.intermediate_size, "intermediate_size")?;
        for (field, shape) in [
            ("gate_proj", vec![intermediate, hidden]),
            ("up_proj", vec![intermediate, hidden]),
            ("down_proj", vec![hidden, intermediate]),
        ] {
            matrix(config, common, format!("{prefix}.{field}.weight"), shape)?;
        }
        return Ok(());
    }
    let experts = dim(config.num_experts, "num_experts")?;
    let intermediate = dim(config.moe_intermediate_size, "moe_intermediate_size")?;
    let shared = dim(
        config.shared_expert_intermediate_size,
        "shared_expert_intermediate_size",
    )?;
    for (field, shape) in [
        ("gate", vec![experts, hidden]),
        ("shared_expert.gate_proj", vec![shared, hidden]),
        ("shared_expert.up_proj", vec![shared, hidden]),
        ("shared_expert.down_proj", vec![hidden, shared]),
        ("shared_expert_gate", vec![1, hidden]),
    ] {
        matrix(config, common, format!("{prefix}.{field}.weight"), shape)?;
    }
    let expert_prefix = format!("{prefix}.experts");
    let packed = layout(
        config,
        "packed",
        [
            (
                format!("{expert_prefix}.gate_up_proj"),
                vec![experts, mul(2, intermediate)?, hidden],
            ),
            (
                format!("{expert_prefix}.down_proj"),
                vec![experts, hidden, intermediate],
            ),
        ],
    )?;
    let mut independent = Vec::with_capacity(mul(experts, 3)?);
    for expert in 0..experts {
        independent.extend([
            (
                format!("{expert_prefix}.{expert}.gate_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{expert_prefix}.{expert}.up_proj.weight"),
                vec![intermediate, hidden],
            ),
            (
                format!("{expert_prefix}.{expert}.down_proj.weight"),
                vec![hidden, intermediate],
            ),
        ]);
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("{expert_prefix} storage"),
        required: true,
        variants: vec![packed, layout(config, "independent", independent)?],
    });
    Ok(())
}

/// Resolves a Qwen3-Next fused QKVZ/BA family into canonical split parameters.
///
/// Group-major physical rows are gathered into component-major canonical rows.
/// Weight, affine companions, and FP8 inverse scales become visible atomically.
pub fn qwen3_next_fused_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    config: &HybridConfig,
    layer: usize,
) -> Result<AtomicRecipeSet, String> {
    if config.variant != HybridVariant::Qwen3Next {
        return Err("fused hybrid recipes require Qwen3-Next policy".into());
    }
    let prefix = format!("model.layers.{layer}.linear_attn");
    let (widths, ba_width) = fused_projection_widths(config).map_err(|error| error.to_string())?;
    let groups = dim(config.linear_num_key_heads, "linear_num_key_heads")?;
    let mut outputs = Vec::new();
    for suffix in ["weight", "scales", "biases"] {
        let qkvz = format!("{prefix}.in_proj_qkvz.{suffix}");
        if catalog.tensor_metadata(&qkvz).is_ok() {
            add_grouped_outputs(
                catalog,
                &mut outputs,
                &qkvz,
                groups,
                &widths.map(|width| usize::try_from(width).unwrap()),
                [
                    (format!("{prefix}.in_proj_qkv.{suffix}"), vec![0, 1, 2]),
                    (format!("{prefix}.in_proj_z.{suffix}"), vec![3]),
                ],
            )?;
        }
        let ba = format!("{prefix}.in_proj_ba.{suffix}");
        if catalog.tensor_metadata(&ba).is_ok() {
            let width = usize::try_from(ba_width).map_err(|_| "invalid BA width")?;
            add_grouped_outputs(
                catalog,
                &mut outputs,
                &ba,
                groups,
                &[width, width],
                [
                    (format!("{prefix}.in_proj_b.{suffix}"), vec![0]),
                    (format!("{prefix}.in_proj_a.{suffix}"), vec![1]),
                ],
            )?;
        }
    }
    let scale = format!("{prefix}.in_proj_qkvz.weight_scale_inv");
    if catalog.tensor_metadata(&scale).is_ok() {
        let block_widths = fp8_block_row_widths(&widths)
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|width| usize::try_from(width).map_err(|_| "invalid FP8 block width"))
            .collect::<Result<Vec<_>, _>>()?;
        add_grouped_outputs(
            catalog,
            &mut outputs,
            &scale,
            groups,
            &block_widths,
            [
                (
                    format!("{prefix}.in_proj_qkv.weight_scale_inv"),
                    vec![0, 1, 2],
                ),
                (format!("{prefix}.in_proj_z.weight_scale_inv"), vec![3]),
            ],
        )?;
    }
    if catalog
        .tensor_metadata(&format!("{prefix}.in_proj_ba.weight_scale_inv"))
        .is_ok()
    {
        return Err("Qwen3-Next fused BA must remain dense and cannot carry inverse scales".into());
    }
    if outputs.is_empty() {
        return Err(format!("no fused projections found for layer {layer}"));
    }
    AtomicRecipeSet::new(catalog, outputs).map_err(|error| error.to_string())
}

fn add_grouped_outputs<C, const N: usize>(
    catalog: &C,
    outputs: &mut Vec<(String, DerivedWeightRecipe)>,
    source: &str,
    groups: usize,
    widths: &[usize],
    targets: [(String, Vec<usize>); N],
) -> Result<(), String>
where
    C: RecipeCatalog + ?Sized,
{
    let group_width = widths.iter().try_fold(0usize, |sum, width| {
        sum.checked_add(*width).ok_or("grouped row width overflow")
    })?;
    for (target, components) in targets {
        if catalog.tensor_metadata(&target).is_ok() {
            return Err(format!(
                "fused output {target:?} collides with a physical tensor"
            ));
        }
        let mut indices = Vec::new();
        for component in components {
            let start = widths
                .get(..component)
                .ok_or("invalid grouped component")?
                .iter()
                .sum::<usize>();
            let width = *widths.get(component).ok_or("invalid grouped component")?;
            for group in 0..groups {
                let base = group
                    .checked_mul(group_width)
                    .and_then(|base| base.checked_add(start))
                    .ok_or("grouped row index overflow")?;
                indices.extend(base..base + width);
            }
        }
        outputs.push((
            target,
            DerivedWeightRecipe::source(source, TensorSelection::Indices { axis: 0, indices }),
        ));
    }
    Ok(())
}

fn layout(
    config: &HybridConfig,
    id: &str,
    tensors: impl IntoIterator<Item = (String, Vec<usize>)>,
) -> Result<LayoutVariant<SafetensorsTensorConstraint>, String> {
    let tensors = tensors.into_iter().collect::<Vec<_>>();
    let discriminator_keys = tensors.iter().map(|(name, _)| name.clone()).collect();
    let mut constraints = Vec::new();
    for (name, shape) in tensors {
        matrix(config, &mut constraints, name, shape)?;
    }
    Ok(LayoutVariant {
        id: id.into(),
        discriminator_keys,
        tensors: constraints,
    })
}

fn matrix(
    config: &HybridConfig,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: impl Into<String>,
    shape: Vec<usize>,
) -> Result<(), String> {
    let name = name.into();
    let aliases = aliases(&name);
    let format = config.linear_format(&name);
    let scale = matches!(format, eredu_checkpoint::LinearFormat::E4M3BlockFp8(_)).then(|| {
        MatrixScaleNames {
            key: format!("{name}_scale_inv"),
            aliases: aliases
                .iter()
                .map(|name| format!("{name}_scale_inv"))
                .collect(),
        }
    });
    output.extend(
        matrix_for_linear_format(&name, aliases, shape, format, scale)
            .map_err(|error| error.to_string())?,
    );
    Ok(())
}

fn vector(output: &mut Vec<SafetensorsTensorConstraint>, name: impl Into<String>, width: usize) {
    let name = name.into();
    output.push(
        SafetensorsTensorConstraint::required(&name, vec![width], StoredDtypeConstraint::Floating)
            .with_aliases(aliases(&name)),
    );
}

fn aliases(name: &str) -> Vec<String> {
    name.strip_prefix("model.")
        .map(|rest| {
            vec![
                format!("model.language_model.{rest}"),
                format!("language_model.{rest}"),
                format!("model.model.{rest}"),
            ]
        })
        .unwrap_or_default()
}

fn dim(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{name} must be positive"))
}

fn mul(left: usize, right: usize) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| "checkpoint dimension overflow".into())
}

fn add(left: usize, right: usize) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| "checkpoint dimension overflow".into())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use eredu_checkpoint::{
        recipe::RecipeCatalog,
        store::{MemoryWeightStore, StoreError, TensorMetadata},
        StoredDtype,
    };
    use safetensors::tensor::Dtype;
    use serde_json::json;

    use super::*;
    use crate::qwen::hybrid::model_args_from_config_value;

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

    fn config() -> HybridConfig {
        model_args_from_config_value(&json!({
            "model_type": "qwen3_next",
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "intermediate_size": 48,
            "layer_types": ["linear_attention", "full_attention"]
        }))
        .unwrap()
        .text
    }

    fn moe_config() -> HybridConfig {
        model_args_from_config_value(&json!({
            "model_type": "qwen3_next",
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "mtp_num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "intermediate_size": 48,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 24,
            "num_experts_per_tok": 2,
            "num_experts": 4,
            "layer_types": ["linear_attention", "full_attention"]
        }))
        .unwrap()
        .text
    }

    fn memory_store(tensors: impl IntoIterator<Item = (String, Vec<usize>)>) -> MemoryWeightStore {
        MemoryWeightStore::from_safetensors(tensors.into_iter().map(|(name, shape)| {
            let bytes = vec![0; shape.iter().product::<usize>() * 2];
            (name, Dtype::F16, shape, bytes)
        }))
        .unwrap()
    }

    #[test]
    fn plan_freezes_fused_or_split_recurrent_and_gated_attention_shapes() {
        let plan = safetensors_plan(&config()).unwrap();
        let recurrent = plan
            .layout_groups
            .iter()
            .find(|group| group.id == "model.layers.0 linear-attention inputs")
            .unwrap();
        assert_eq!(recurrent.variants.len(), 2);
        assert!(recurrent.variants.iter().any(|variant| {
            variant
                .discriminator_keys
                .contains(&"model.layers.0.linear_attn.in_proj_qkvz.weight".into())
                && variant
                    .discriminator_keys
                    .contains(&"model.layers.0.linear_attn.in_proj_ba.weight".into())
        }));
        let query = plan
            .common_tensors
            .iter()
            .find(|tensor| tensor.key == "model.layers.1.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(query.shape, [64, 32]);
    }

    #[test]
    fn grouped_fused_recipes_restore_component_major_rows_atomically() {
        let config = config();
        let prefix = "model.layers.0.linear_attn";
        let mut tensors = BTreeMap::new();
        for (name, shape) in [
            (format!("{prefix}.in_proj_qkvz.weight"), vec![96, 32]),
            (format!("{prefix}.in_proj_ba.weight"), vec![8, 32]),
        ] {
            tensors.insert(name.clone(), metadata(&name, shape));
        }
        let catalog = Catalog(tensors);
        let recipes = qwen3_next_fused_recipes(&catalog, &config, 0).unwrap();
        assert_eq!(recipes.iter().count(), 4);
        assert_eq!(
            recipes
                .get(&format!("{prefix}.in_proj_qkv.weight"))
                .unwrap()
                .infer(&catalog)
                .unwrap()
                .shape(),
            &[64, 32]
        );
        assert_eq!(
            recipes
                .get(&format!("{prefix}.in_proj_z.weight"))
                .unwrap()
                .infer(&catalog)
                .unwrap()
                .shape(),
            &[32, 32]
        );
    }

    #[test]
    fn neutral_unit_catalog_owns_fused_recurrent_transforms() {
        let config = config();
        let prefix = "model.layers.0.linear_attn";
        let store = memory_store([
            (format!("{prefix}.in_proj_qkvz.weight"), vec![96, 32]),
            (format!("{prefix}.in_proj_ba.weight"), vec![8, 32]),
        ]);
        let recipes = unit_recipes(&store, &config, 0).unwrap();
        for suffix in [
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_a.weight",
            "in_proj_b.weight",
        ] {
            assert!(
                recipes.contains_key(&format!("{prefix}.{suffix}")),
                "missing {suffix}"
            );
        }
        assert_eq!(
            translate_vision_gguf_weight_name("v.patch_embd.weight.1", &[0]),
            "model.visual.patch_embed.proj.weight.1"
        );
    }

    #[test]
    fn fused_recipes_reject_physical_split_collision() {
        let config = config();
        let prefix = "model.layers.0.linear_attn";
        let mut tensors = BTreeMap::new();
        for (name, shape) in [
            (format!("{prefix}.in_proj_qkvz.weight"), vec![96, 32]),
            (format!("{prefix}.in_proj_qkv.weight"), vec![64, 32]),
        ] {
            tensors.insert(name.clone(), metadata(&name, shape));
        }
        assert!(qwen3_next_fused_recipes(&Catalog(tensors), &config, 0).is_err());
    }

    #[test]
    fn residency_catalog_owns_target_and_mtp_expert_topology() {
        let config = moe_config();
        let mut tensors = BTreeMap::new();
        for root in [
            "model.layers.0.mlp.experts".to_owned(),
            "model.layers.1.mlp.experts".to_owned(),
            "mtp.layers.0.mlp.experts".to_owned(),
        ] {
            for (name, shape) in [
                ("gate_up_proj", vec![4, 32, 32]),
                ("down_proj", vec![4, 32, 16]),
            ] {
                let target = format!("{root}.{name}");
                tensors.insert(target.clone(), metadata(&target, shape));
            }
        }
        let catalog = expert_residency_catalog(&Catalog(tensors), &config).unwrap();
        assert_eq!(catalog.units().len(), 12);
        let target = &catalog.units()[0];
        assert_eq!(target.identity(), eredu_runtime::ExpertIdentity::new(0, 0));
        assert_eq!(target.owner_group().as_str(), "target");
        assert_eq!(target.owner_unit(), 0);
        assert_eq!(target.unit_path(), "model.layers.0");
        assert_eq!(
            target
                .parameters()
                .iter()
                .map(|parameter| (parameter.binding_name(), parameter.logical_target()))
                .collect::<Vec<_>>(),
            [
                ("down_proj", "model.layers.0.mlp.experts.down_proj"),
                ("gate_up_proj", "model.layers.0.mlp.experts.gate_up_proj"),
            ]
        );
        let prediction = &catalog.units()[8];
        assert_eq!(
            prediction.identity(),
            eredu_runtime::ExpertIdentity::new(2, 0)
        );
        assert_eq!(prediction.owner_group().as_str(), "mtp.0");
        assert_eq!(prediction.owner_unit(), 0);
        assert_eq!(prediction.unit_path(), "mtp.layers.0");
        assert_eq!(
            prediction.distribution(),
            crate::ExpertResidencyDistribution::ExpertParallel
        );

        let prediction_units = catalog
            .into_units_selected_by_owner(|group, unit| group.as_str() == "mtp.0" && unit == 0)
            .collect::<Vec<_>>();
        assert_eq!(prediction_units.len(), 4);
        assert!(prediction_units
            .iter()
            .all(|unit| unit.identity().layer == 2));
    }
}
