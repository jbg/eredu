//! Pure Kimi Linear checkpoint plans, naming, and derived-weight recipes.

use std::collections::{BTreeMap, HashMap};

use eredu_checkpoint::schema::{
    AlternativeLayoutGroup, CatalogPolicy, DepthwiseConvolutionSchema, GgufCheckpointPlan,
    GgufTensorConstraint, GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan,
    SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{CheckpointSource, TensorSelection, WeightStoreBackend},
    StoredDtype, WeightQuantization,
};

use super::{AttentionKind, FeedForwardPolicy, ModelArgs};

/// Derives a Kimi Linear configuration whose physical matrix formats reflect
/// load-time quantization instead of checkpoint-specific format selections.
pub fn load_time_quantization(
    args: &ModelArgs,
    quantization: WeightQuantization,
) -> Result<ModelArgs, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = args.clone();
    target.weight_quantization = Some(quantization);
    target.quantized_weight_configs = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical checkpoint format metadata to a complete Kimi Linear
/// configuration.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    mut formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    normalize_weight_formats(args, &mut formats);
    let mut target = args.clone();
    target.quantized_weight_configs = Some(formats);
    target.weight_quantization = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

fn canonical_recipe_name(name: &str) -> String {
    name.replace(".block_sparse_moe.", ".mlp.")
}

fn normalized_checkpoint_keys(store: &dyn CheckpointSource) -> BTreeMap<String, String> {
    store
        .source_keys()
        .into_iter()
        .map(|raw| (canonical_recipe_name(&raw), raw))
        .collect()
}

fn expert_source(
    normalized: &BTreeMap<String, String>,
    prefix: &str,
    expert: usize,
    projections: &[&str],
) -> Result<DerivedWeightRecipe, String> {
    let runtime = projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.weight"))
        .find(|candidate| normalized.contains_key(candidate))
        .ok_or_else(|| {
            format!("Kimi Linear checkpoint is missing expert {expert} projection under {prefix}")
        })?;
    Ok(DerivedWeightRecipe::source(
        normalized
            .get(&runtime)
            .expect("normalized Kimi expert key exists"),
        TensorSelection::Full,
    ))
}

/// Returns the complete neutral recipe catalog for one Kimi execution unit.
pub fn unit_recipes(
    store: &dyn CheckpointSource,
    args: &ModelArgs,
    layer: usize,
    include_experts: bool,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| format!("Kimi Linear has no layer policy {layer}"))?;
    let mut recipes = BTreeMap::new();
    let root = format!("model.layers.{layer}");
    let expert_prefix = format!("{root}.mlp.experts");
    let normalized = normalized_checkpoint_keys(store);
    for (runtime, raw) in &normalized {
        if runtime.starts_with(&format!("{root}.mlp."))
            && !runtime.starts_with(&expert_prefix)
            && runtime != raw
        {
            recipes.insert(
                runtime.clone(),
                DerivedWeightRecipe::source(raw, TensorSelection::Full),
            );
        }
    }

    let attention = format!("{root}.self_attn");
    let projection = checked_mul(
        dimension(args.kda_config.num_heads, "KDA head count")?,
        dimension(args.kda_config.head_dim, "KDA head width")?,
        "KDA projection width",
    )?;
    let kernel = dimension(
        args.kda_config.short_conv_kernel_size,
        "KDA convolution width",
    )?;
    for local in ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"] {
        let name = format!("{attention}.{local}");
        if store.source_metadata(&name).is_ok() {
            recipes.insert(
                name.clone(),
                DerivedWeightRecipe::Reshape {
                    input: Box::new(DerivedWeightRecipe::source(&name, TensorSelection::Full)),
                    shape: vec![projection, 1, kernel],
                },
            );
        }
    }
    let a_log = format!("{attention}.A_log");
    if store.source_metadata(&a_log).is_ok() {
        let mut recipe = DerivedWeightRecipe::Reshape {
            input: Box::new(DerivedWeightRecipe::source(&a_log, TensorSelection::Full)),
            shape: vec![
                1,
                1,
                dimension(args.kda_config.num_heads, "KDA head count")?,
                1,
            ],
        };
        if store
            .source_diagnostics()
            .map_err(|error| error.to_string())?
            .backend
            == WeightStoreBackend::Gguf
        {
            recipe = DerivedWeightRecipe::NegLog {
                input: Box::new(recipe),
            };
        }
        recipes.insert(a_log, recipe);
    }

    if !include_experts || policy.feed_forward != FeedForwardPolicy::SparseMoe {
        return Ok(recipes);
    }
    let gate_up = format!("{expert_prefix}.gate_up_proj");
    let down = format!("{expert_prefix}.down_proj");
    if let (Some(gate_up_source), Some(down_source)) =
        (normalized.get(&gate_up), normalized.get(&down))
    {
        for (target, source) in [(&gate_up, gate_up_source), (&down, down_source)] {
            if source != target {
                recipes.insert(
                    target.clone(),
                    DerivedWeightRecipe::source(source, TensorSelection::Full),
                );
            }
        }
        return Ok(recipes);
    }
    let gate = format!("{expert_prefix}.gate_proj");
    let up = format!("{expert_prefix}.up_proj");
    if normalized.contains_key(&gate) && normalized.contains_key(&up) {
        recipes.insert(
            gate_up,
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: [gate, up]
                    .into_iter()
                    .map(|runtime| {
                        DerivedWeightRecipe::source(
                            normalized.get(&runtime).expect("normalized projection key"),
                            TensorSelection::Full,
                        )
                    })
                    .collect(),
            },
        );
        return Ok(recipes);
    }
    let experts = dimension(args.num_experts, "expert count")?;
    let mut gate_up_inputs = Vec::with_capacity(experts);
    let mut down_inputs = Vec::with_capacity(experts);
    for expert in 0..experts {
        gate_up_inputs.push(DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![
                expert_source(&normalized, &expert_prefix, expert, &["w1", "gate_proj"])?,
                expert_source(&normalized, &expert_prefix, expert, &["w3", "up_proj"])?,
            ],
        });
        down_inputs.push(expert_source(
            &normalized,
            &expert_prefix,
            expert,
            &["w2", "down_proj"],
        )?);
    }
    recipes.insert(
        gate_up,
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: gate_up_inputs,
        },
    );
    recipes.insert(
        down,
        DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: down_inputs,
        },
    );
    Ok(recipes)
}

/// Returns neutral lazy-loading recipes for one Kimi routed expert.
pub fn expert_recipes(
    store: &dyn CheckpointSource,
    args: &ModelArgs,
    layer: usize,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| format!("Kimi Linear has no layer {layer}"))?;
    if policy.feed_forward != FeedForwardPolicy::SparseMoe {
        return Err(format!("Kimi Linear layer {layer} is not sparse MoE"));
    }
    let experts = dimension(args.num_experts, "expert count")?;
    if expert >= experts {
        return Err(format!(
            "Kimi Linear has no expert {expert} in layer {layer}"
        ));
    }
    let normalized = normalized_checkpoint_keys(store);
    let prefix = format!("model.layers.{layer}.mlp.experts");
    let packed_gate_up = format!("{prefix}.gate_up_proj");
    let packed_down = format!("{prefix}.down_proj");
    let selection = TensorSelection::Range {
        axis: 0,
        start: expert,
        end: expert + 1,
    };
    let mut recipes = BTreeMap::new();
    if let (Some(gate_up), Some(down)) = (
        normalized.get(&packed_gate_up),
        normalized.get(&packed_down),
    ) {
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
            if let Some(source) = normalized.get(&source) {
                recipes.insert(
                    target.into(),
                    DerivedWeightRecipe::source(source, selection.clone()),
                );
            }
        }
        return Ok(recipes);
    }
    let gate = format!("{prefix}.gate_proj");
    let up = format!("{prefix}.up_proj");
    if let (Some(gate), Some(up), Some(down)) = (
        normalized.get(&gate),
        normalized.get(&up),
        normalized.get(&packed_down),
    ) {
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
            if let (Some(gate), Some(up)) = (normalized.get(&gate), normalized.get(&up)) {
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
            if let Some(down) = normalized.get(&down) {
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
        return Err("split Kimi Linear experts cannot be lazily load-time quantized".into());
    }
    let gate = expert_source(&normalized, &prefix, expert, &["w1", "gate_proj"])?;
    let up = expert_source(&normalized, &prefix, expert, &["w3", "up_proj"])?;
    let down = expert_source(&normalized, &prefix, expert, &["w2", "down_proj"])?;
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
    Ok(recipes)
}

/// Builds the complete architecture-owned schedule for independently resident experts.
pub fn expert_residency_catalog(
    store: &dyn CheckpointSource,
    args: &ModelArgs,
) -> Result<crate::ExpertResidencyCatalog, String> {
    if !args.has_sparse_moe_layers() {
        return Err("independent expert residency requires Kimi Linear-MoE".into());
    }
    let experts = dimension(args.num_experts, "expert count")?;
    let owner_group = eredu_runtime::ExecutionGroupId::new(crate::decoder::TARGET_EXECUTION_GROUP)
        .map_err(|error| error.to_string())?;
    let mut units = Vec::new();
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if policy.feed_forward != FeedForwardPolicy::SparseMoe {
            continue;
        }
        let unit_path = format!("model.layers.{layer}");
        let expert_root = format!("{unit_path}.mlp.experts");
        for expert in 0..experts {
            let recipes = expert_recipes(store, args, layer, expert)?;
            let gate_up_quantizable = !recipes.contains_key("gate_up_proj_scales")
                && !recipes.contains_key("gate_up_proj_biases");
            let down_quantizable = !recipes.contains_key("down_proj_scales")
                && !recipes.contains_key("down_proj_biases");
            let parameters = recipes
                .into_iter()
                .map(|(binding, recipe)| {
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
                    let target = format!("{expert_root}.{binding}");
                    crate::ExpertParameterRecipe::new(binding, target, recipe, role)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            units.push(
                crate::ExpertResidencyUnit::new(
                    eredu_runtime::ExpertIdentity::new(layer, expert),
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
    crate::ExpertResidencyCatalog::new(units).map_err(|error| error.to_string())
}

/// Builds the strict SafeTensors catalog plan.
pub fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    add_safe_matrix(
        args,
        &mut common,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
    )?;
    common.push(safe_aliases("model.norm.weight", vec![hidden]));
    if !args.tie_word_embeddings {
        add_safe_matrix(args, &mut common, "lm_head.weight", vec![vocab, hidden])?;
    }

    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        let root = format!("model.layers.{layer}");
        common.extend([
            safe_aliases(&format!("{root}.input_layernorm.weight"), vec![hidden]),
            safe_aliases(
                &format!("{root}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
        ]);
        match policy.attention {
            AttentionKind::Kda => add_safe_kda(args, &root, &mut common)?,
            AttentionKind::Mla => add_safe_mla(args, &root, &mut common)?,
        }
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let intermediate = dimension(args.intermediate_size, "dense intermediate size")?;
                for (local, shape) in [
                    ("gate_proj.weight", vec![intermediate, hidden]),
                    ("up_proj.weight", vec![intermediate, hidden]),
                    ("down_proj.weight", vec![hidden, intermediate]),
                ] {
                    add_safe_matrix(args, &mut common, &format!("{root}.mlp.{local}"), shape)?;
                }
            }
            FeedForwardPolicy::SparseMoe => {
                add_safe_moe(args, layer, &root, &mut common, &mut groups)?;
            }
        }
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.push("model.mtp.".into());
    policy.allowed_suffixes.push(".rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("Kimi Linear SafeTensors", common, groups, policy)
        .map_err(|error| error.to_string())
}

fn add_safe_kda(
    args: &ModelArgs,
    root: &str,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let heads = dimension(args.kda_config.num_heads, "KDA head count")?;
    let head = dimension(args.kda_config.head_dim, "KDA head width")?;
    let kernel = dimension(
        args.kda_config.short_conv_kernel_size,
        "KDA convolution width",
    )?;
    let projection = checked_mul(heads, head, "KDA projection width")?;
    let prefix = format!("{root}.self_attn");
    for (local, shape) in [
        ("q_proj.weight", vec![projection, hidden]),
        ("k_proj.weight", vec![projection, hidden]),
        ("v_proj.weight", vec![projection, hidden]),
        ("f_a_proj.weight", vec![head, hidden]),
        ("f_b_proj.weight", vec![projection, head]),
        ("b_proj.weight", vec![heads, hidden]),
        ("g_a_proj.weight", vec![head, hidden]),
        ("g_b_proj.weight", vec![projection, head]),
        ("o_proj.weight", vec![hidden, projection]),
    ] {
        add_safe_matrix(args, output, &format!("{prefix}.{local}"), shape)?;
    }
    output.extend([
        safe_aliases(&format!("{prefix}.dt_bias"), vec![projection]),
        safe_aliases(&format!("{prefix}.o_norm.weight"), vec![head]),
    ]);
    let convolution_elements = checked_mul(projection, kernel, "KDA convolution elements")?;
    let convolution = DepthwiseConvolutionSchema::new(projection, kernel, false)?;
    for name in ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"] {
        output.push(
            safe_aliases(&format!("{prefix}.{name}"), convolution.storage_shape())
                .with_element_count(convolution_elements),
        );
    }
    output.push(
        safe_aliases(&format!("{prefix}.A_log"), vec![1, 1, heads, 1]).with_element_count(heads),
    );
    Ok(())
}

fn add_safe_mla(
    args: &ModelArgs,
    root: &str,
    output: &mut Vec<SafetensorsTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let heads = dimension(args.num_attention_heads, "attention head count")?;
    let nope = dimension(args.qk_nope_head_dim, "MLA no-PE width")?;
    let rope = dimension(args.qk_rope_head_dim, "MLA RoPE width")?;
    let value = dimension(args.v_head_dim, "MLA value width")?;
    let kv_rank = dimension(args.kv_lora_rank, "KV LoRA rank")?;
    let query_head = checked_add(nope, rope, "MLA query head width")?;
    let prefix = format!("{root}.self_attn");
    if let Some(rank) = args.q_lora_rank {
        let rank = dimension(rank, "query LoRA rank")?;
        add_safe_matrix(
            args,
            output,
            &format!("{prefix}.q_a_proj.weight"),
            vec![rank, hidden],
        )?;
        output.push(safe_aliases(
            &format!("{prefix}.q_a_layernorm.weight"),
            vec![rank],
        ));
        add_safe_matrix(
            args,
            output,
            &format!("{prefix}.q_b_proj.weight"),
            vec![
                checked_mul(heads, query_head, "query projection width")?,
                rank,
            ],
        )?;
    } else {
        add_safe_matrix(
            args,
            output,
            &format!("{prefix}.q_proj.weight"),
            vec![
                checked_mul(heads, query_head, "query projection width")?,
                hidden,
            ],
        )?;
    }
    add_safe_matrix(
        args,
        output,
        &format!("{prefix}.kv_a_proj_with_mqa.weight"),
        vec![checked_add(kv_rank, rope, "KV-A width")?, hidden],
    )?;
    output.push(safe_aliases(
        &format!("{prefix}.kv_a_layernorm.weight"),
        vec![kv_rank],
    ));
    if args.split_kv_b {
        add_safe_matrix(
            args,
            output,
            &format!("{prefix}.k_b_proj.weight"),
            vec![checked_mul(heads, nope, "K-B width")?, kv_rank],
        )?;
        add_safe_matrix(
            args,
            output,
            &format!("{prefix}.v_b_proj.weight"),
            vec![checked_mul(heads, value, "V-B width")?, kv_rank],
        )?;
    } else {
        add_safe_matrix(
            args,
            output,
            &format!("{prefix}.kv_b_proj.weight"),
            vec![
                checked_mul(
                    heads,
                    checked_add(nope, value, "KV-B head width")?,
                    "KV-B width",
                )?,
                kv_rank,
            ],
        )?;
    }
    add_safe_matrix(
        args,
        output,
        &format!("{prefix}.o_proj.weight"),
        vec![hidden, checked_mul(heads, value, "attention output width")?],
    )
}

fn add_safe_moe(
    args: &ModelArgs,
    layer: usize,
    root: &str,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let experts = dimension(args.num_experts, "expert count")?;
    let intermediate = dimension(args.moe_intermediate_size, "MoE intermediate size")?;
    for (local, shape, matrix) in [
        ("gate.weight", vec![experts, hidden], true),
        ("gate.e_score_correction_bias", vec![experts], false),
        (
            "shared_experts.gate_proj.weight",
            vec![intermediate, hidden],
            true,
        ),
        (
            "shared_experts.up_proj.weight",
            vec![intermediate, hidden],
            true,
        ),
        (
            "shared_experts.down_proj.weight",
            vec![hidden, intermediate],
            true,
        ),
    ] {
        let name = format!("{root}.mlp.{local}");
        if matrix {
            add_safe_matrix(args, common, &name, shape)?;
        } else {
            common.push(safe_aliases(&name, shape));
        }
    }

    let prefix = format!("{root}.mlp.experts");
    let gate_up = format!("{prefix}.gate_up_proj");
    let down = format!("{prefix}.down_proj");
    let mut packed_tensors = safe_matrix_constraints(
        &gate_up,
        vec![
            experts,
            checked_mul(2, intermediate, "fused expert width")?,
            hidden,
        ],
        args.weight_quantization_for(&gate_up),
    )?;
    packed_tensors.extend(safe_matrix_constraints(
        &down,
        vec![experts, hidden, intermediate],
        args.weight_quantization_for(&down),
    )?);
    let mut split_tensors = Vec::new();
    let mut split_discriminators = Vec::new();
    for expert in 0..experts {
        for (projection, shape) in [
            ("w1", vec![intermediate, hidden]),
            ("w2", vec![hidden, intermediate]),
            ("w3", vec![intermediate, hidden]),
        ] {
            let name = format!("{prefix}.{expert}.{projection}.weight");
            split_discriminators.push(name.clone());
            split_tensors.push(safe_aliases(&name, shape));
        }
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("Kimi Linear layer {layer} expert storage"),
        required: true,
        variants: vec![
            LayoutVariant {
                id: "packed".into(),
                tensors: packed_tensors,
                discriminator_keys: vec![gate_up, down],
            },
            LayoutVariant {
                id: "split".into(),
                tensors: split_tensors,
                discriminator_keys: split_discriminators,
            },
        ],
    });
    Ok(())
}

fn add_safe_matrix(
    args: &ModelArgs,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    output.extend(safe_matrix_constraints(
        name,
        shape,
        args.weight_quantization_for(name),
    )?);
    Ok(())
}

fn safe_matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let Some(quantization) = quantization else {
        return Ok(vec![safe_aliases(name, shape)]);
    };
    let input = *shape
        .last()
        .ok_or_else(|| format!("Kimi Linear matrix {name:?} has scalar shape"))?;
    let bits = dimension(quantization.bits(), "affine bit width")?;
    let group = dimension(quantization.group_size(), "affine group size")?;
    let packed_bits = checked_mul(input, bits, "affine packing")?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(format!(
            "Kimi Linear quantized tensor {name:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
        ));
    }
    let mut packed = shape.clone();
    *packed.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion_shape = shape;
    *companion_shape.last_mut().expect("matrix shape") = input / group;
    let canonical_prefix = outer_prefix(name);
    let aliases_for = |suffix: &str| {
        let canonical = format!("{canonical_prefix}.{suffix}");
        physical_names(name)
            .into_iter()
            .map(|candidate| format!("{}.{suffix}", outer_prefix(&candidate)))
            .filter(|candidate| candidate != &canonical)
            .collect::<Vec<_>>()
    };
    let companion_dtype = || {
        StoredDtypeConstraint::OneOf(vec![
            StoredDtype::F16,
            StoredDtype::BF16,
            StoredDtype::F32,
            StoredDtype::U8,
        ])
    };
    let mut constraints = vec![safe_aliases_with_dtype(
        name,
        packed,
        StoredDtypeConstraint::Exact(StoredDtype::U32),
    )];
    constraints.push(
        SafetensorsTensorConstraint::required(
            format!("{canonical_prefix}.scales"),
            companion_shape.clone(),
            companion_dtype(),
        )
        .with_aliases(aliases_for("scales"))
        .companion(),
    );
    if quantization.has_biases() {
        constraints.push(
            SafetensorsTensorConstraint::required(
                format!("{canonical_prefix}.biases"),
                companion_shape,
                companion_dtype(),
            )
            .with_aliases(aliases_for("biases"))
            .companion(),
        );
    }
    Ok(constraints)
}

fn safe_aliases(key: &str, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    safe_aliases_with_dtype(key, shape, StoredDtypeConstraint::Floating)
}

fn safe_aliases_with_dtype(
    key: &str,
    shape: Vec<usize>,
    dtype: StoredDtypeConstraint,
) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, dtype).with_aliases(
        physical_names(key)
            .into_iter()
            .filter(|candidate| candidate != key),
    )
}

fn physical_names(canonical: &str) -> Vec<String> {
    let mut names = vec![canonical.to_string()];
    if canonical.contains(".mlp.") {
        names.push(canonical.replace(".mlp.", ".block_sparse_moe."));
    }
    names.sort();
    names.dedup();
    names
}

fn outer_prefix(name: &str) -> &str {
    name.strip_suffix(".weight").unwrap_or(name)
}

/// Builds the GGUF physical catalog plan.
pub fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let mut tensors = vec![
        gguf(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf("output_norm.weight", vec![hidden], TensorOperation::Vector),
    ];
    if !args.tie_word_embeddings {
        tensors.push(gguf(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ));
    }
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        let block = format!("blk.{layer}");
        tensors.extend([
            gguf(
                format!("{block}.attn_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
            gguf(
                format!("{block}.ffn_norm.weight"),
                vec![hidden],
                TensorOperation::Vector,
            ),
        ]);
        match policy.attention {
            AttentionKind::Kda => add_gguf_kda(args, &block, &mut tensors)?,
            AttentionKind::Mla => add_gguf_mla(args, &block, &mut tensors)?,
        }
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let intermediate = dimension(args.intermediate_size, "dense intermediate size")?;
                for (local, shape) in [
                    ("ffn_gate.weight", vec![intermediate, hidden]),
                    ("ffn_up.weight", vec![intermediate, hidden]),
                    ("ffn_down.weight", vec![hidden, intermediate]),
                ] {
                    tensors.push(gguf(
                        format!("{block}.{local}"),
                        shape,
                        TensorOperation::Matrix,
                    ));
                }
            }
            FeedForwardPolicy::SparseMoe => {
                let experts = dimension(args.num_experts, "expert count")?;
                let intermediate = dimension(args.moe_intermediate_size, "MoE intermediate size")?;
                for (local, shape, operation) in [
                    (
                        "ffn_gate_inp.weight",
                        vec![experts, hidden],
                        TensorOperation::Matrix,
                    ),
                    ("exp_probs_b.bias", vec![experts], TensorOperation::Vector),
                    (
                        "ffn_gate_shexp.weight",
                        vec![intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    (
                        "ffn_up_shexp.weight",
                        vec![intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    (
                        "ffn_down_shexp.weight",
                        vec![hidden, intermediate],
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
                    tensors.push(gguf(format!("{block}.{local}"), shape, operation));
                }
            }
        }
    }
    GgufCheckpointPlan::new(
        "Kimi Linear GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .map_err(|error| error.to_string())
}

fn add_gguf_kda(
    args: &ModelArgs,
    block: &str,
    output: &mut Vec<GgufTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let heads = dimension(args.kda_config.num_heads, "KDA head count")?;
    let head = dimension(args.kda_config.head_dim, "KDA head width")?;
    let kernel = dimension(
        args.kda_config.short_conv_kernel_size,
        "KDA convolution width",
    )?;
    let projection = checked_mul(heads, head, "KDA projection width")?;
    for (local, shape, operation) in [
        (
            "attn_q.weight",
            vec![projection, hidden],
            TensorOperation::Matrix,
        ),
        (
            "attn_k.weight",
            vec![projection, hidden],
            TensorOperation::Matrix,
        ),
        (
            "attn_v.weight",
            vec![projection, hidden],
            TensorOperation::Matrix,
        ),
        (
            "ssm_f_a.weight",
            vec![head, hidden],
            TensorOperation::Matrix,
        ),
        (
            "ssm_f_b.weight",
            vec![projection, head],
            TensorOperation::Matrix,
        ),
        (
            "ssm_beta.weight",
            vec![heads, hidden],
            TensorOperation::Matrix,
        ),
        (
            "ssm_g_a.weight",
            vec![head, hidden],
            TensorOperation::Matrix,
        ),
        (
            "ssm_g_b.weight",
            vec![projection, head],
            TensorOperation::Matrix,
        ),
        ("ssm_dt.bias", vec![projection], TensorOperation::Vector),
        ("ssm_norm.weight", vec![head], TensorOperation::Vector),
        (
            "attn_output.weight",
            vec![hidden, projection],
            TensorOperation::Matrix,
        ),
    ] {
        output.push(gguf(format!("{block}.{local}"), shape, operation));
    }
    let convolution_elements = checked_mul(projection, kernel, "KDA convolution elements")?;
    for local in [
        "ssm_conv1d_q.weight",
        "ssm_conv1d_k.weight",
        "ssm_conv1d_v.weight",
    ] {
        output.push(
            gguf(
                format!("{block}.{local}"),
                vec![projection, 1, kernel],
                TensorOperation::Dense,
            )
            .with_element_count(convolution_elements),
        );
    }
    output.push(
        gguf(
            format!("{block}.ssm_a"),
            vec![heads],
            TensorOperation::Vector,
        )
        .with_aliases([format!("{block}.ssm_a.weight")])
        .with_element_count(heads),
    );
    Ok(())
}

fn add_gguf_mla(
    args: &ModelArgs,
    block: &str,
    output: &mut Vec<GgufTensorConstraint>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let heads = dimension(args.num_attention_heads, "attention head count")?;
    let nope = dimension(args.qk_nope_head_dim, "MLA no-PE width")?;
    let rope = dimension(args.qk_rope_head_dim, "MLA RoPE width")?;
    let value = dimension(args.v_head_dim, "MLA value width")?;
    let kv_rank = dimension(args.kv_lora_rank, "KV LoRA rank")?;
    let query_head = checked_add(nope, rope, "MLA query head width")?;
    if let Some(rank) = args.q_lora_rank {
        let rank = dimension(rank, "query LoRA rank")?;
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
            output.push(gguf(format!("{block}.{local}"), shape, operation));
        }
    } else {
        output.push(gguf(
            format!("{block}.attn_q.weight"),
            vec![checked_mul(heads, query_head, "query width")?, hidden],
            TensorOperation::Matrix,
        ));
    }
    output.extend([
        gguf(
            format!("{block}.attn_kv_a_mqa.weight"),
            vec![checked_add(kv_rank, rope, "KV-A width")?, hidden],
            TensorOperation::Matrix,
        ),
        gguf(
            format!("{block}.attn_kv_a_norm.weight"),
            vec![kv_rank],
            TensorOperation::Vector,
        ),
    ]);
    if args.split_kv_b {
        output.extend([
            gguf(
                format!("{block}.attn_k_b.weight"),
                vec![checked_mul(heads, nope, "K-B width")?, kv_rank],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{block}.attn_v_b.weight"),
                vec![checked_mul(heads, value, "V-B width")?, kv_rank],
                TensorOperation::Matrix,
            ),
        ]);
    } else {
        output.push(gguf(
            format!("{block}.attn_kv_b.weight"),
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
    }
    output.push(gguf(
        format!("{block}.attn_output.weight"),
        vec![hidden, checked_mul(heads, value, "attention output width")?],
        TensorOperation::Matrix,
    ));
    Ok(())
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
        .ok_or_else(|| format!("Kimi Linear {name} must be positive, got {value}"))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("Kimi Linear {name} geometry overflows"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Kimi Linear {name} geometry overflows"))
}

/// Translates a physical GGUF tensor name to canonical Kimi parameter identity.
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
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.to_owned();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.to_owned();
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
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_q_a", "self_attn.q_a_proj"),
        ("attn_q_b", "self_attn.q_b_proj"),
        ("attn_kv_a_mqa", "self_attn.kv_a_proj_with_mqa"),
        ("attn_kv_b", "self_attn.kv_b_proj"),
        ("attn_k_b", "self_attn.k_b_proj"),
        ("attn_v_b", "self_attn.v_b_proj"),
        ("attn_q_a_norm", "self_attn.q_a_layernorm"),
        ("attn_kv_a_norm", "self_attn.kv_a_layernorm"),
        ("attn_output", "self_attn.o_proj"),
        ("ssm_conv1d_q", "self_attn.q_conv1d"),
        ("ssm_conv1d_k", "self_attn.k_conv1d"),
        ("ssm_conv1d_v", "self_attn.v_conv1d"),
        ("ssm_f_a", "self_attn.f_a_proj"),
        ("ssm_f_b", "self_attn.f_b_proj"),
        ("ssm_beta", "self_attn.b_proj"),
        ("ssm_g_a", "self_attn.g_a_proj"),
        ("ssm_g_b", "self_attn.g_b_proj"),
        ("ssm_norm", "self_attn.o_norm"),
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("ffn_gate_shexp", "mlp.shared_experts.gate_proj"),
        ("ffn_up_shexp", "mlp.shared_experts.up_proj"),
        ("ffn_down_shexp", "mlp.shared_experts.down_proj"),
        ("ffn_gate_inp", "mlp.gate"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    if parameter == "ssm_a" || parameter.starts_with("ssm_a.") {
        let suffix = parameter.strip_prefix("ssm_a").unwrap_or_default();
        let suffix = if suffix == ".weight" { "" } else { suffix };
        return format!("model.layers.{layer}.self_attn.A_log{suffix}");
    }
    if parameter == "ssm_dt.bias" || parameter == "ssm_dt" {
        return format!("model.layers.{layer}.self_attn.dt_bias");
    }
    name.to_owned()
}

/// Rehomes split expert GGUF formats onto the canonical fused runtime weights.
pub fn normalize_weight_formats<V>(args: &ModelArgs, formats: &mut HashMap<String, V>) {
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if policy.feed_forward != FeedForwardPolicy::SparseMoe {
            continue;
        }
        let prefix = format!("model.layers.{layer}.mlp.experts");
        if let Some(format) = formats.remove(&format!("{prefix}.gate_proj")) {
            formats.remove(&format!("{prefix}.up_proj"));
            formats.insert(format!("{prefix}.gate_up_proj"), format);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use eredu_checkpoint::store::{
        CheckpointLease, StoreError, TensorMetadata, TensorReadRequest, WeightStoreDiagnostics,
    };

    use super::*;

    struct Catalog {
        tensors: BTreeMap<String, TensorMetadata>,
        backend: WeightStoreBackend,
    }

    impl CheckpointSource for Catalog {
        fn source_keys(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }

        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.tensors
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }

        fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            Err(StoreError::UnknownTensor { key: request.key })
        }

        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            Ok(WeightStoreDiagnostics {
                backend: self.backend,
                mapping_hits: 0,
                mapping_misses: 0,
                evictions: 0,
                currently_mapped_shards: 0,
                touched_shard_paths: Vec::new(),
                physical_reads: 0,
                physical_read_bytes: 0,
                coalesced_group_hits: 0,
            })
        }
    }

    fn metadata(name: &str, shape: Vec<usize>) -> TensorMetadata {
        TensorMetadata {
            name: name.into(),
            encoded_byte_len: shape.iter().product::<usize>() as u64 * 4,
            logical_shape: shape.clone(),
            physical_shape: shape,
            stored_dtype: StoredDtype::F32,
            backing_shard: None,
        }
    }

    fn fixture(split_kv_b: bool, q_lora_rank: Option<i32>) -> ModelArgs {
        let mut value = serde_json::json!({
            "model_type":"kimi_linear", "vocab_size":16, "hidden_size":12,
            "num_hidden_layers":2, "num_attention_heads":3, "num_key_value_heads":3,
            "intermediate_size":17, "head_dim":4, "model_max_length":64,
            "linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],"num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
            "num_experts":2, "moe_intermediate_size":9, "kv_lora_rank":6,
            "qk_nope_head_dim":4, "qk_rope_head_dim":2, "v_head_dim":4,
            "mla_use_nope":true, "num_experts_per_token":1, "num_shared_experts":1,
            "routed_scaling_factor":1.0, "first_k_dense_replace":1,
            "num_expert_group":1, "topk_group":1, "tie_word_embeddings":false,
            "split_kv_b": split_kv_b
        });
        if let Some(rank) = q_lora_rank {
            value["q_lora_rank"] = rank.into();
        }
        let mut args = super::super::config::model_args_from_config_value(&value).unwrap();
        args.split_kv_b = split_kv_b;
        args
    }

    #[test]
    fn architecture_derives_checkpoint_and_load_time_format_policy() {
        let source = fixture(false, None);
        let name = "model.layers.0.mlp.gate_proj.weight".to_string();
        let checkpoint =
            WeightQuantization::Affine(eredu_checkpoint::AffineQuantization::new(32, 8).unwrap());
        let checkpoint_args =
            with_checkpoint_formats(&source, HashMap::from([(name.clone(), checkpoint)])).unwrap();
        assert_eq!(
            checkpoint_args.weight_quantization_for(&name),
            Some(checkpoint)
        );

        let requested =
            WeightQuantization::Affine(eredu_checkpoint::AffineQuantization::new(32, 4).unwrap());
        let target = load_time_quantization(&checkpoint_args, requested).unwrap();
        assert_eq!(target.weight_quantization, Some(requested));
        assert_eq!(target.quantized_weight_configs, None);
        assert_eq!(target.weight_quantization_for(&name), Some(requested));
    }

    #[test]
    fn plans_kda_mla_low_rank_and_alternative_expert_artifacts() {
        for (split, query_rank) in [(false, None), (true, Some(5))] {
            let plan = safetensors_plan(&fixture(split, query_rank)).unwrap();
            let names = plan
                .common_tensors
                .iter()
                .map(|tensor| tensor.key.as_str())
                .collect::<Vec<_>>();
            assert!(names.contains(&"model.layers.0.self_attn.q_conv1d.weight"));
            assert!(names.contains(&"model.layers.0.self_attn.A_log"));
            assert_eq!(
                names.contains(&"model.layers.1.self_attn.kv_b_proj.weight"),
                !split
            );
            assert_eq!(
                names.contains(&"model.layers.1.self_attn.k_b_proj.weight"),
                split
            );
            assert_eq!(
                names.contains(&"model.layers.1.self_attn.q_a_proj.weight"),
                query_rank.is_some()
            );
            let experts = plan
                .layout_groups
                .iter()
                .find(|group| group.id.contains("expert storage"))
                .unwrap();
            assert_eq!(
                experts
                    .variants
                    .iter()
                    .map(|variant| variant.id.as_str())
                    .collect::<Vec<_>>(),
                ["packed", "split"]
            );
        }
    }

    #[test]
    fn gguf_plan_and_translation_cover_hybrid_physical_names() {
        let args = fixture(false, None);
        let plan = gguf_plan(&args).unwrap();
        let names = plan
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"blk.0.ssm_conv1d_q.weight"));
        assert!(names.contains(&"blk.1.attn_kv_b.weight"));
        assert!(names.contains(&"blk.1.ffn_gate_exps.weight"));
        assert_eq!(
            translate_gguf_weight_name("blk.0.ssm_a.weight"),
            "model.layers.0.self_attn.A_log"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.1.ffn_gate_exps.scales"),
            "model.layers.1.mlp.experts.gate_proj_scales"
        );

        let prefix = "model.layers.1.mlp.experts";
        let mut formats = HashMap::from([
            (format!("{prefix}.gate_proj"), 1),
            (format!("{prefix}.up_proj"), 2),
        ]);
        normalize_weight_formats(&args, &mut formats);
        assert_eq!(formats.get(&format!("{prefix}.gate_up_proj")), Some(&1));
        assert!(!formats.contains_key(&format!("{prefix}.gate_proj")));
        assert!(!formats.contains_key(&format!("{prefix}.up_proj")));
    }

    #[test]
    fn portable_names_do_not_normalize_module_tree_paths() {
        assert_eq!(
            canonical_recipe_name("projection.inner.weight"),
            "projection.inner.weight"
        );
        assert_eq!(physical_names("projection.weight"), ["projection.weight"]);
    }

    #[test]
    fn neutral_catalog_owns_kda_and_split_expert_transformations() {
        let args = fixture(false, None);
        let mut tensors = BTreeMap::from([
            (
                "model.layers.0.self_attn.q_conv1d.weight".into(),
                metadata("model.layers.0.self_attn.q_conv1d.weight", vec![12, 3]),
            ),
            (
                "model.layers.0.self_attn.A_log".into(),
                metadata("model.layers.0.self_attn.A_log", vec![3]),
            ),
        ]);
        for expert in 0..2 {
            for (projection, shape) in [
                ("w1", vec![9, 12]),
                ("w2", vec![12, 9]),
                ("w3", vec![9, 12]),
            ] {
                let name =
                    format!("model.layers.1.block_sparse_moe.experts.{expert}.{projection}.weight");
                tensors.insert(name.clone(), metadata(&name, shape));
            }
        }
        let catalog = Catalog {
            tensors,
            backend: WeightStoreBackend::Gguf,
        };
        let kda = unit_recipes(&catalog, &args, 0, true).unwrap();
        assert!(matches!(
            kda.get("model.layers.0.self_attn.q_conv1d.weight"),
            Some(DerivedWeightRecipe::Reshape { shape, .. }) if shape == &[12, 1, 3]
        ));
        assert!(matches!(
            kda.get("model.layers.0.self_attn.A_log"),
            Some(DerivedWeightRecipe::NegLog { .. })
        ));
        let sparse = unit_recipes(&catalog, &args, 1, true).unwrap();
        assert!(matches!(
            sparse.get("model.layers.1.mlp.experts.gate_up_proj"),
            Some(DerivedWeightRecipe::Stack { axis: 0, inputs }) if inputs.len() == 2
        ));
        assert!(!sparse
            .keys()
            .any(|name| name.contains(".mlp.experts.0.") || name.contains(".mlp.experts.1.")));
        assert!(matches!(
            expert_recipes(&catalog, &args, 1, 0)
                .unwrap()
                .get("gate_up_proj"),
            Some(DerivedWeightRecipe::Stack { axis: 0, inputs }) if inputs.len() == 1
        ));
        let residency = expert_residency_catalog(&catalog, &args).unwrap();
        assert_eq!(residency.units().len(), 2);
        assert_eq!(residency.units()[0].unit_path(), "model.layers.1");
        assert!(residency.units()[0]
            .parameters()
            .iter()
            .all(|parameter| matches!(
                parameter.role(),
                crate::ExpertParameterRole::QuantizableProjection { .. }
            )));
        assert_eq!(residency.units()[0].owner_group().as_str(), "target");
        let owned = residency
            .into_units_selected_by_owner(|group, unit| group.as_str() == "target" && unit == 1)
            .count();
        assert_eq!(owned, 2);
    }
}
