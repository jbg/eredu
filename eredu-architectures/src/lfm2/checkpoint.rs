//! Pure SafeTensors and GGUF checkpoint plans for LFM2.

use std::collections::{BTreeMap, HashMap};

use eredu_checkpoint::schema::{
    AlternativeLayoutGroup, CatalogPolicy, DepthwiseConvolutionSchema, FusedProjectionSegment,
    FusedSegmentedProjectionSchema, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
    LayoutVariant, SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint,
    TensorOperation,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{CheckpointSource, TensorSelection},
};
use eredu_checkpoint::{StoredDtype, WeightQuantization};
use eredu_core::AttentionPolicy;

use super::config::{FeedForwardPolicy, ModelArgs, OperatorPolicy};

/// Derives an LFM2 configuration whose physical matrix formats reflect
/// load-time quantization instead of checkpoint-specific format selections.
pub fn load_time_quantization(
    args: &ModelArgs,
    quantization: WeightQuantization,
) -> Result<ModelArgs, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = args.clone();
    target.weight_quantization = Some(quantization);
    target.quantized_weights = None;
    target.quantized_weight_configs = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical checkpoint format metadata to a complete LFM2
/// configuration.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    mut formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    normalize_weight_formats(args, &mut formats);
    let mut target = args.clone();
    target.quantized_weights = Some(formats.keys().cloned().collect());
    target.quantized_weight_configs = Some(formats);
    target.weight_quantization = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

fn expert_source(
    store: &dyn CheckpointSource,
    prefix: &str,
    expert: i32,
    projections: &[&str],
) -> Result<DerivedWeightRecipe, String> {
    let keys = store.source_keys();
    let key = projections
        .iter()
        .map(|projection| format!("{prefix}.{expert}.{projection}.weight"))
        .find(|candidate| keys.contains(candidate))
        .ok_or_else(|| {
            format!("LFM2 checkpoint is missing expert {expert} projection under {prefix}")
        })?;
    Ok(DerivedWeightRecipe::source(key, TensorSelection::Full))
}

/// Returns the complete derived-weight catalog for one LFM2 execution unit.
pub fn unit_recipes(
    store: &dyn CheckpointSource,
    args: &ModelArgs,
    layer: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let mut recipes = BTreeMap::new();
    let root = format!("model.layers.{layer}");
    let conv = format!("{root}.conv.conv.weight");
    if let Ok(metadata) = store.source_metadata(&conv) {
        let expected = vec![args.hidden_size as usize, 1, args.conv_l_cache as usize];
        let recipe = if metadata.logical_shape.len() == 2 {
            Some(DerivedWeightRecipe::Reshape {
                input: Box::new(DerivedWeightRecipe::source(&conv, TensorSelection::Full)),
                shape: expected.clone(),
            })
        } else if metadata.logical_shape == [expected[0], expected[2], expected[1]]
            && metadata.logical_shape != expected
        {
            Some(DerivedWeightRecipe::Transpose {
                input: Box::new(DerivedWeightRecipe::source(&conv, TensorSelection::Full)),
                axes: vec![0, 2, 1],
            })
        } else {
            None
        };
        if let Some(recipe) = recipe {
            recipes.insert(conv.clone(), recipe);
        }
    }
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| format!("LFM2 has no layer policy {layer}"))?;
    if policy.feed_forward != FeedForwardPolicy::SparseMoe {
        return Ok(recipes);
    }
    let prefix = format!("{root}.feed_forward.experts");
    let keys = store.source_keys();
    let gate_up = format!("{prefix}.gate_up_proj");
    let down = format!("{prefix}.down_proj");
    if keys.contains(&gate_up) {
        return Ok(recipes);
    }
    if keys.contains(&format!("{prefix}.gate_proj")) && keys.contains(&format!("{prefix}.up_proj"))
    {
        recipes.insert(
            gate_up,
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: ["gate_proj", "up_proj"]
                    .into_iter()
                    .map(|name| {
                        DerivedWeightRecipe::source(
                            format!("{prefix}.{name}"),
                            TensorSelection::Full,
                        )
                    })
                    .collect(),
            },
        );
        return Ok(recipes);
    }
    let mut gate_up_inputs = Vec::new();
    let mut down_inputs = Vec::new();
    for expert in 0..args.num_experts {
        gate_up_inputs.push(DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![
                expert_source(store, &prefix, expert, &["w1", "gate_proj"])?,
                expert_source(store, &prefix, expert, &["w3", "up_proj"])?,
            ],
        });
        down_inputs.push(expert_source(store, &prefix, expert, &["w2", "down_proj"])?);
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

/// Returns canonical lazy-loading recipes for one LFM2 routed expert.
pub fn expert_recipes(
    store: &dyn CheckpointSource,
    args: &ModelArgs,
    layer: usize,
    expert: usize,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let policy = args
        .layer_policy(layer)
        .ok_or_else(|| format!("LFM2 has no layer policy {layer}"))?;
    if policy.feed_forward != FeedForwardPolicy::SparseMoe {
        return Err(format!("LFM2 layer {layer} is not sparse MoE"));
    }
    if expert >= args.num_experts as usize {
        return Err(format!("LFM2 has no expert {expert} in layer {layer}"));
    }
    let prefix = format!("model.layers.{layer}.feed_forward.experts");
    let packed_gate_up = format!("{prefix}.gate_up_proj");
    let packed_down = format!("{prefix}.down_proj");
    let keys = store.source_keys();
    let selection = TensorSelection::Range {
        axis: 0,
        start: expert,
        end: expert + 1,
    };
    let mut recipes = BTreeMap::new();
    if keys.contains(&packed_gate_up) && keys.contains(&packed_down) {
        for (name, key) in [
            ("gate_up_proj", packed_gate_up.clone()),
            ("down_proj", packed_down.clone()),
        ] {
            recipes.insert(
                name.into(),
                DerivedWeightRecipe::source(key, selection.clone()),
            );
        }
        for (name, key) in [
            ("gate_up_proj_scales", format!("{packed_gate_up}_scales")),
            ("gate_up_proj_biases", format!("{packed_gate_up}_biases")),
            ("down_proj_scales", format!("{packed_down}_scales")),
            ("down_proj_biases", format!("{packed_down}_biases")),
        ] {
            if keys.contains(&key) {
                recipes.insert(
                    name.into(),
                    DerivedWeightRecipe::source(key, selection.clone()),
                );
            }
        }
    } else if keys.contains(&format!("{prefix}.gate_proj"))
        && keys.contains(&format!("{prefix}.up_proj"))
        && keys.contains(&packed_down)
    {
        recipes.insert(
            "gate_up_proj".into(),
            DerivedWeightRecipe::Concatenate {
                axis: 1,
                inputs: vec![
                    DerivedWeightRecipe::source(format!("{prefix}.gate_proj"), selection.clone()),
                    DerivedWeightRecipe::source(format!("{prefix}.up_proj"), selection.clone()),
                ],
            },
        );
        recipes.insert(
            "down_proj".into(),
            DerivedWeightRecipe::source(packed_down.clone(), selection.clone()),
        );
        for suffix in ["_scales", "_biases"] {
            let gate = format!("{prefix}.gate_proj{suffix}");
            let up = format!("{prefix}.up_proj{suffix}");
            if keys.contains(&gate) && keys.contains(&up) {
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
            if keys.contains(&down) {
                recipes.insert(
                    format!("down_proj{suffix}"),
                    DerivedWeightRecipe::source(down, selection.clone()),
                );
            }
        }
    } else {
        if args.weight_quantization_for(&packed_gate_up).is_some()
            || args.weight_quantization_for(&packed_down).is_some()
        {
            return Err("split LFM2 experts cannot be lazily load-time quantized".into());
        }
        let gate = expert_source(store, &prefix, expert as i32, &["w1", "gate_proj"])?;
        let up = expert_source(store, &prefix, expert as i32, &["w3", "up_proj"])?;
        let down = expert_source(store, &prefix, expert as i32, &["w2", "down_proj"])?;
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
    }
    Ok(recipes)
}

/// Builds the complete architecture-owned schedule for independently resident experts.
pub fn expert_residency_catalog(
    store: &dyn CheckpointSource,
    args: &ModelArgs,
) -> Result<crate::ExpertResidencyCatalog, String> {
    if !args.has_sparse_moe_layers() {
        return Err("independent expert residency requires LFM2-MoE".into());
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
        let expert_root = format!("{unit_path}.feed_forward.experts");
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

/// Rehomes split expert GGUF formats onto their canonical fused runtime weights.
pub fn normalize_weight_formats<V>(args: &ModelArgs, formats: &mut HashMap<String, V>) {
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if policy.feed_forward != FeedForwardPolicy::SparseMoe {
            continue;
        }
        let prefix = format!("model.layers.{layer}.feed_forward.experts");
        if let Some(format) = formats.remove(&format!("{prefix}.gate_proj")) {
            formats.remove(&format!("{prefix}.up_proj"));
            formats.insert(format!("{prefix}.gate_up_proj"), format);
        }
    }
}

/// Builds the complete alternative-layout SafeTensors contract.
pub fn safetensors_plan(
    args: &ModelArgs,
    allow_derived_packed: bool,
) -> Result<SafetensorsCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let layers = dimension(args.num_hidden_layers, "layer count")?;
    let heads = dimension(args.num_attention_heads, "attention head count")?;
    let kv_heads = dimension(args.num_key_value_heads, "key/value head count")?;
    if !hidden.is_multiple_of(heads) {
        return Err(format!(
            "LFM2 hidden size {hidden} is not divisible by attention head count {heads}"
        ));
    }
    let head = hidden / heads;
    let key_value = checked_mul(kv_heads, head, "key/value projection width")?;
    let mut common = Vec::new();
    let mut groups = Vec::new();
    add_safe_matrix(
        args,
        &mut common,
        "model.embed_tokens.weight",
        vec![vocab, hidden],
        true,
    )?;
    common.push(safe("model.embedding_norm.weight", vec![hidden]));
    if !args.tie_word_embeddings {
        add_safe_matrix(
            args,
            &mut common,
            "lm_head.weight",
            vec![vocab, hidden],
            true,
        )?;
    }

    for layer in 0..layers {
        let root = format!("model.layers.{layer}");
        common.extend([
            safe(format!("{root}.operator_norm.weight"), vec![hidden]),
            safe(format!("{root}.ffn_norm.weight"), vec![hidden]),
        ]);
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| format!("LFM2 layer schedule has no policy for layer {layer}"))?;
        match policy.operator {
            OperatorPolicy::CausalConvolution => {
                let kernel = dimension(args.conv_l_cache, "short-convolution width")?;
                let convolution = DepthwiseConvolutionSchema::new(hidden, kernel, args.conv_bias)?;
                let fused = FusedSegmentedProjectionSchema::new(
                    hidden,
                    ["gate", "value", "convolution"]
                        .into_iter()
                        .map(|name| FusedProjectionSegment::new(name, hidden))
                        .collect::<Result<Vec<_>, _>>()?,
                )?;
                common.push(
                    safe(
                        format!("{root}.conv.conv.weight"),
                        convolution.storage_shape(),
                    )
                    .with_alternate_shapes([
                        DepthwiseConvolutionSchema::with_axes(
                            hidden,
                            kernel,
                            eredu_checkpoint::schema::DepthwiseKernelAxes::ChannelsKernelSingleton,
                            eredu_checkpoint::schema::DepthwiseKernelAxes::ChannelsSingletonKernel,
                            args.conv_bias,
                        )?
                        .storage_shape(),
                    ]),
                );
                add_safe_matrix(
                    args,
                    &mut common,
                    &format!("{root}.conv.in_proj.weight"),
                    fused.matrix_shape(),
                    true,
                )?;
                add_safe_matrix(
                    args,
                    &mut common,
                    &format!("{root}.conv.out_proj.weight"),
                    vec![hidden, hidden],
                    true,
                )?;
                if args.conv_bias {
                    common.extend([
                        safe(format!("{root}.conv.conv.bias"), vec![hidden]),
                        safe(format!("{root}.conv.in_proj.bias"), fused.bias_shape()),
                        safe(format!("{root}.conv.out_proj.bias"), vec![hidden]),
                    ]);
                }
            }
            OperatorPolicy::SelfAttention(AttentionPolicy::Full) => {
                for (local, shape, matrix) in [
                    ("q_proj.weight", vec![hidden, hidden], true),
                    ("k_proj.weight", vec![key_value, hidden], true),
                    ("v_proj.weight", vec![key_value, hidden], true),
                    ("out_proj.weight", vec![hidden, hidden], true),
                    ("q_layernorm.weight", vec![head], false),
                    ("k_layernorm.weight", vec![head], false),
                ] {
                    let name = format!("{root}.self_attn.{local}");
                    if matrix {
                        add_safe_matrix(args, &mut common, &name, shape, true)?;
                    } else {
                        common.push(safe(name, shape));
                    }
                }
            }
            OperatorPolicy::SelfAttention(AttentionPolicy::Sliding { .. }) => {
                return Err("LFM2 structural admission does not support sliding attention".into());
            }
        }

        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let intermediate =
                    dimension(args.dense_intermediate_size, "dense intermediate size")?;
                for (local, shape) in [
                    ("w1.weight", vec![intermediate, hidden]),
                    ("w2.weight", vec![hidden, intermediate]),
                    ("w3.weight", vec![intermediate, hidden]),
                ] {
                    add_safe_matrix(
                        args,
                        &mut common,
                        &format!("{root}.feed_forward.{local}"),
                        shape,
                        true,
                    )?;
                }
            }
            FeedForwardPolicy::SparseMoe => {
                add_safe_moe(
                    args,
                    layer,
                    &root,
                    allow_derived_packed,
                    &mut common,
                    &mut groups,
                )?;
            }
        }
    }

    let mut policy = CatalogPolicy::non_strict();
    policy.allowed_suffixes.push(".rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("LFM2 SafeTensors", common, groups, policy)
        .map_err(|error| error.to_string())
}

fn add_safe_moe(
    args: &ModelArgs,
    layer: usize,
    root: &str,
    allow_derived_packed: bool,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let experts = dimension(args.num_experts, "expert count")?;
    let intermediate = dimension(args.moe_intermediate_size, "MoE intermediate size")?;
    common.push(safe(
        format!("{root}.feed_forward.gate.weight"),
        vec![experts, hidden],
    ));
    if args.use_expert_bias {
        common.push(safe(
            format!("{root}.feed_forward.expert_bias"),
            vec![experts],
        ));
    }
    let prefix = format!("{root}.feed_forward.experts");
    let gate_up = format!("{prefix}.gate_up_proj");
    let gate = format!("{prefix}.gate_proj");
    let up = format!("{prefix}.up_proj");
    let down = format!("{prefix}.down_proj");
    let gate_up_quantization = args.weight_quantization_for(&gate_up);
    let down_quantization = args.weight_quantization_for(&down);
    let mut variants = vec![LayoutVariant {
        id: "fused packed".into(),
        tensors: [
            safe_matrix_constraints(
                &gate_up,
                vec![
                    experts,
                    checked_mul(2, intermediate, "fused expert width")?,
                    hidden,
                ],
                gate_up_quantization,
            )?,
            safe_matrix_constraints(
                &down,
                vec![experts, hidden, intermediate],
                down_quantization,
            )?,
        ]
        .concat(),
        discriminator_keys: vec![gate_up.clone()],
    }];
    if allow_derived_packed {
        variants.push(LayoutVariant {
            id: "separate packed".into(),
            tensors: [
                safe_matrix_constraints(
                    &gate,
                    vec![experts, intermediate, hidden],
                    gate_up_quantization,
                )?,
                safe_matrix_constraints(
                    &up,
                    vec![experts, intermediate, hidden],
                    gate_up_quantization,
                )?,
                safe_matrix_constraints(
                    &down,
                    vec![experts, hidden, intermediate],
                    down_quantization,
                )?,
            ]
            .concat(),
            discriminator_keys: vec![gate, up],
        });
    }
    if gate_up_quantization.is_none() && down_quantization.is_none() {
        let mut tensors = Vec::new();
        let mut discriminators = Vec::new();
        for expert in 0..experts {
            for (canonical, alias, shape) in [
                (
                    format!("{prefix}.{expert}.w1.weight"),
                    format!("{prefix}.{expert}.gate_proj.weight"),
                    vec![intermediate, hidden],
                ),
                (
                    format!("{prefix}.{expert}.w2.weight"),
                    format!("{prefix}.{expert}.down_proj.weight"),
                    vec![hidden, intermediate],
                ),
                (
                    format!("{prefix}.{expert}.w3.weight"),
                    format!("{prefix}.{expert}.up_proj.weight"),
                    vec![intermediate, hidden],
                ),
            ] {
                discriminators.push(canonical.clone());
                tensors.push(safe(canonical, shape).with_aliases([alias]));
            }
        }
        variants.push(LayoutVariant {
            id: "per-expert split".into(),
            tensors,
            discriminator_keys: discriminators,
        });
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("LFM2 layer {layer} routed expert storage"),
        required: true,
        variants,
    });
    Ok(())
}

fn add_safe_matrix(
    args: &ModelArgs,
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
    quantizable: bool,
) -> Result<(), String> {
    let quantization = quantizable
        .then(|| args.weight_quantization_for(name))
        .flatten();
    output.extend(safe_matrix_constraints(name, shape, quantization)?);
    Ok(())
}

fn safe_matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let Some(quantization) = quantization else {
        return Ok(vec![safe(name, shape)]);
    };
    let input = *shape
        .last()
        .ok_or_else(|| format!("LFM2 matrix {name:?} has scalar shape"))?;
    let bits = dimension(quantization.bits(), "quantization bit width")?;
    let group = dimension(quantization.group_size(), "quantization group size")?;
    let packed_bits = checked_mul(input, bits, "affine packing")?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(format!(
            "quantized tensor {name:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
        ));
    }
    let mut packed = shape.clone();
    *packed.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion = shape;
    *companion.last_mut().expect("matrix shape") = input / group;
    let prefix = outer_prefix(name);
    let companion_dtype = || {
        StoredDtypeConstraint::OneOf(vec![
            StoredDtype::F16,
            StoredDtype::BF16,
            StoredDtype::F32,
            StoredDtype::U8,
        ])
    };
    let mut constraints = vec![SafetensorsTensorConstraint::required(
        name,
        packed,
        StoredDtypeConstraint::Exact(StoredDtype::U32),
    )];
    constraints.push(
        SafetensorsTensorConstraint::required(
            format!("{prefix}.scales"),
            companion.clone(),
            companion_dtype(),
        )
        .companion(),
    );
    if quantization.has_biases() {
        constraints.push(
            SafetensorsTensorConstraint::required(
                format!("{prefix}.biases"),
                companion,
                companion_dtype(),
            )
            .companion(),
        );
    }
    Ok(constraints)
}

fn safe(key: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
}

fn outer_prefix(name: &str) -> &str {
    name.strip_suffix(".weight").unwrap_or(name)
}

/// Builds the complete physical GGUF tensor contract.
pub fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let heads = dimension(args.num_attention_heads, "attention head count")?;
    let kv_heads = dimension(args.num_key_value_heads, "key/value head count")?;
    if !hidden.is_multiple_of(heads) {
        return Err(format!(
            "LFM2 hidden size {hidden} is not divisible by attention head count {heads}"
        ));
    }
    let head = hidden / heads;
    let key_value = checked_mul(kv_heads, head, "key/value projection width")?;
    let mut tensors = vec![
        gguf(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf(
            "token_embd_norm.weight",
            vec![hidden],
            TensorOperation::Vector,
        ),
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
        match policy.operator {
            OperatorPolicy::CausalConvolution => {
                let kernel = dimension(args.conv_l_cache, "short-convolution width")?;
                tensors.push(gguf(
                    format!("{block}.shortconv.conv.weight"),
                    vec![hidden, kernel],
                    TensorOperation::Dense,
                ));
                tensors.extend([
                    gguf(
                        format!("{block}.shortconv.in_proj.weight"),
                        vec![
                            checked_mul(3, hidden, "short-convolution input width")?,
                            hidden,
                        ],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{block}.shortconv.out_proj.weight"),
                        vec![hidden, hidden],
                        TensorOperation::Matrix,
                    ),
                ]);
                if args.conv_bias {
                    tensors.extend([
                        gguf(
                            format!("{block}.shortconv.conv.bias"),
                            vec![hidden],
                            TensorOperation::Vector,
                        ),
                        gguf(
                            format!("{block}.shortconv.in_proj.bias"),
                            vec![checked_mul(3, hidden, "short-convolution bias width")?],
                            TensorOperation::Vector,
                        ),
                        gguf(
                            format!("{block}.shortconv.out_proj.bias"),
                            vec![hidden],
                            TensorOperation::Vector,
                        ),
                    ]);
                }
            }
            OperatorPolicy::SelfAttention(AttentionPolicy::Full) => {
                for (local, shape, operation) in [
                    (
                        "attn_q.weight",
                        vec![hidden, hidden],
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
                        vec![hidden, hidden],
                        TensorOperation::Matrix,
                    ),
                    ("attn_q_norm.weight", vec![head], TensorOperation::Vector),
                    ("attn_k_norm.weight", vec![head], TensorOperation::Vector),
                ] {
                    tensors.push(gguf(format!("{block}.{local}"), shape, operation));
                }
            }
            OperatorPolicy::SelfAttention(AttentionPolicy::Sliding { .. }) => {
                return Err("LFM2 structural admission does not support sliding attention".into());
            }
        }
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                let intermediate =
                    dimension(args.dense_intermediate_size, "dense intermediate size")?;
                for (local, shape) in [
                    ("ffn_gate.weight", vec![intermediate, hidden]),
                    ("ffn_down.weight", vec![hidden, intermediate]),
                    ("ffn_up.weight", vec![intermediate, hidden]),
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
                tensors.extend([
                    gguf(
                        format!("{block}.ffn_gate_inp.weight"),
                        vec![experts, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{block}.ffn_gate_exps.weight"),
                        vec![experts, intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{block}.ffn_up_exps.weight"),
                        vec![experts, intermediate, hidden],
                        TensorOperation::Matrix,
                    ),
                    gguf(
                        format!("{block}.ffn_down_exps.weight"),
                        vec![experts, hidden, intermediate],
                        TensorOperation::Matrix,
                    ),
                ]);
                if args.use_expert_bias {
                    tensors.push(
                        gguf(
                            format!("{block}.ffn_exp_probs_b.bias"),
                            vec![experts],
                            TensorOperation::Vector,
                        )
                        .with_aliases([format!("{block}.exp_probs_b.bias")]),
                    );
                }
            }
        }
    }
    GgufCheckpointPlan::new(
        "LFM2 GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
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
        .ok_or_else(|| format!("LFM2 {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("LFM2 {name} geometry overflows"))
}

/// Translates one physical GGUF tensor name to its canonical parameter identity.
pub fn translate_gguf_weight_name(name: &str, is_moe: bool) -> String {
    for (source, target) in [
        ("token_embd", "model.embed_tokens"),
        ("token_embd_norm", "model.embedding_norm"),
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
    if is_moe {
        for (source, target) in [
            ("ffn_gate_inp", "feed_forward.gate"),
            ("ffn_gate_exps", "feed_forward.experts.gate_proj"),
            ("ffn_up_exps", "feed_forward.experts.up_proj"),
            ("ffn_down_exps", "feed_forward.experts.down_proj"),
            ("ffn_exp_probs_b", "feed_forward.expert_bias"),
            ("exp_probs_b", "feed_forward.expert_bias"),
        ] {
            if parameter == source || parameter.starts_with(&format!("{source}.")) {
                let suffix = parameter.strip_prefix(source).unwrap_or_default();
                let suffix = if target.ends_with("expert_bias") && suffix == ".bias" {
                    ""
                } else if target.contains("experts.") {
                    match suffix {
                        ".weight" => "",
                        ".scales" => "_scales",
                        ".biases" => "_biases",
                        other => other,
                    }
                } else {
                    suffix
                };
                return format!("model.layers.{layer}.{target}{suffix}");
            }
        }
    }
    for (source, target) in [
        ("shortconv.conv", "conv.conv"),
        ("shortconv.in_proj", "conv.in_proj"),
        ("shortconv.out_proj", "conv.out_proj"),
        ("attn_q_norm", "self_attn.q_layernorm"),
        ("attn_k_norm", "self_attn.k_layernorm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.out_proj"),
        ("attn_norm", "operator_norm"),
        ("ffn_norm", "ffn_norm"),
        ("ffn_gate", "feed_forward.w1"),
        ("ffn_down", "feed_forward.w2"),
        ("ffn_up", "feed_forward.w3"),
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
    use eredu_checkpoint::store::MemoryWeightStore;
    use safetensors::tensor::Dtype;

    use super::*;

    fn memory_store(tensors: impl IntoIterator<Item = (String, Vec<usize>)>) -> MemoryWeightStore {
        MemoryWeightStore::from_safetensors(tensors.into_iter().map(|(name, shape)| {
            let bytes = vec![0; shape.iter().product::<usize>() * 2];
            (name, Dtype::F16, shape, bytes)
        }))
        .unwrap()
    }

    fn fixture() -> ModelArgs {
        super::super::config::model_args_from_config_value(&serde_json::json!({
            "model_type": "lfm2_moe", "vocab_size": 32, "hidden_size": 16,
            "intermediate_size": 24, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "layer_types": ["conv", "full_attention", "conv"],
            "conv_L_cache": 3, "block_auto_adjust_ff_dim": false,
            "num_dense_layers": 1, "moe_intermediate_size": 8,
            "num_experts": 4, "num_experts_per_tok": 2,
            "use_expert_bias": true
        }))
        .unwrap()
    }

    #[test]
    fn load_time_quantization_replaces_checkpoint_format_policy() {
        let mut source = fixture();
        source.quantized_weights = Some(Default::default());
        source.quantized_weight_configs = Some(Default::default());
        let requested =
            WeightQuantization::Affine(eredu_checkpoint::AffineQuantization::new(32, 4).unwrap());

        let target = load_time_quantization(&source, requested).unwrap();

        assert_eq!(target.weight_quantization, Some(requested));
        assert_eq!(target.quantized_weights, None);
        assert_eq!(target.quantized_weight_configs, None);
        assert_eq!(
            target.weight_quantization_for("model.layers.0.feed_forward.w1.weight"),
            Some(requested)
        );
        assert!(source
            .weight_quantization_for("model.layers.0.feed_forward.w1.weight")
            .is_none());
    }

    #[test]
    fn plans_every_scheduled_operator_and_feed_forward_policy() {
        let args = fixture();
        let safe = safetensors_plan(&args, true).unwrap();
        let safe_names = safe
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<Vec<_>>();
        assert!(safe_names.contains(&"model.layers.0.conv.conv.weight"));
        assert!(safe_names.contains(&"model.layers.1.self_attn.q_proj.weight"));
        assert!(safe_names.contains(&"model.layers.0.feed_forward.w1.weight"));
        assert!(safe_names.contains(&"model.layers.1.feed_forward.gate.weight"));

        let gguf = gguf_plan(&args).unwrap();
        let gguf_names = gguf
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<Vec<_>>();
        assert!(gguf_names.contains(&"blk.0.shortconv.conv.weight"));
        assert!(gguf_names.contains(&"blk.1.attn_q.weight"));
        assert!(gguf_names.contains(&"blk.1.ffn_gate_exps.weight"));
    }

    #[test]
    fn translates_dense_sparse_and_short_convolution_names() {
        assert_eq!(
            translate_gguf_weight_name("blk.2.shortconv.in_proj.weight", false),
            "model.layers.2.conv.in_proj.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.2.ffn_gate_exps.weight", true),
            "model.layers.2.feed_forward.experts.gate_proj"
        );
    }

    #[test]
    fn packed_weights_use_only_canonical_checkpoint_names() {
        let quantization =
            WeightQuantization::Affine(eredu_checkpoint::AffineQuantization::new(32, 4).unwrap());
        let constraints =
            safe_matrix_constraints("projection.weight", vec![64, 64], Some(quantization)).unwrap();

        assert!(constraints[0].aliases.is_empty());
    }

    #[test]
    fn neutral_catalog_owns_split_expert_stacking() {
        let args = fixture();
        let prefix = "model.layers.1.feed_forward.experts";
        let mut tensors = Vec::new();
        for layer in 1..3 {
            let layer_prefix = format!("model.layers.{layer}.feed_forward.experts");
            for expert in 0..args.num_experts {
                for (name, shape) in [
                    ("w1", vec![8, 16]),
                    ("w3", vec![8, 16]),
                    ("w2", vec![16, 8]),
                ] {
                    tensors.push((format!("{layer_prefix}.{expert}.{name}.weight"), shape));
                }
            }
        }
        let store = memory_store(tensors);
        let units = unit_recipes(&store, &args, 1).unwrap();
        assert!(matches!(
            units.get(&format!("{prefix}.gate_up_proj")),
            Some(DerivedWeightRecipe::Stack { axis: 0, inputs }) if inputs.len() == 4
        ));
        let expert = expert_recipes(&store, &args, 1, 2).unwrap();
        assert_eq!(expert.len(), 2);
        let residency = expert_residency_catalog(&store, &args).unwrap();
        assert_eq!(residency.units().len(), 8);
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
        assert_eq!(owned, 4);

        let mut formats = HashMap::from([
            (format!("{prefix}.gate_proj"), 1),
            (format!("{prefix}.up_proj"), 2),
        ]);
        normalize_weight_formats(&args, &mut formats);
        assert_eq!(formats.get(&format!("{prefix}.gate_up_proj")), Some(&1));
    }
}
