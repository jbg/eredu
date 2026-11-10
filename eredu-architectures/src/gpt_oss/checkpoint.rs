//! Pure checkpoint schemas, name translation, and expert normalization for GPT-OSS.

use std::collections::{BTreeMap, HashMap};

use eredu_checkpoint::{
    expert::{
        canonical_gated_expert_projection_family_recipes, ExpertProjectionFamilyNames,
        GatedExpertProjectionFamilyNames, GatedExpertProjectionFamilyRecipes,
    },
    recipe::{DerivedWeightRecipe, RecipeCatalog},
    schema::{
        matrix_for_linear_format, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
        GgufTypeConstraint, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint, TensorOperation,
    },
    validation::{validate_gguf_plan, validate_matching_gguf_encodings, CheckpointValidation},
    LinearFormat, StoredDtype, WeightQuantization,
};
use eredu_gguf::Checkpoint as GgufCheckpoint;

use super::config::ModelArgs;
use eredu_nn::{DeclareLogicalParameters, LogicalParameterRole};

const MXFP4_GROUP_SIZE: usize = 32;
const MXFP4_BLOCK_BYTES: usize = 16;

/// Derives a GPT-OSS configuration whose ordinary dense matrix formats
/// reflect load-time quantization while preserving native MXFP4 experts.
pub fn load_time_quantization(
    args: &ModelArgs,
    quantization: WeightQuantization,
) -> Result<ModelArgs, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut target = args.clone();
    target.quantization = Some(quantization);
    target.quantized_weight_configs = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Applies canonical checkpoint format metadata to a complete GPT-OSS
/// configuration while preserving native expert formats.
pub fn with_checkpoint_formats(
    args: &ModelArgs,
    mut formats: HashMap<String, WeightQuantization>,
) -> Result<ModelArgs, String> {
    let expert_targets = gguf_expert_quantization_targets(args)?
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    formats.retain(|name, _| !expert_targets.contains(name));
    let mut target = args.clone();
    target.quantized_weight_configs = Some(formats);
    target.quantization = None;
    target.validate().map_err(|error| error.to_string())?;
    Ok(target)
}

/// Builds the exact published GPT-OSS SafeTensors catalog.
pub fn safetensors_plan(
    declaration: &super::Declaration,
) -> Result<SafetensorsCheckpointPlan, String> {
    let declarations = super::parallel::parameter_description(
        declaration,
        crate::decoder::ParameterEncoding::Logical,
    )
    .map_err(|error| error.to_string())?;
    let mut tensors = Vec::new();
    let routers = declaration
        .projections()
        .owner_layers()
        .map(|layer| {
            declaration
                .projections()
                .parameters(layer)
                .map(|spec| spec.selector.weight().id.as_str().to_owned())
                .map_err(|error| error.to_string())
        })
        .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
    for group in declarations.groups() {
        if group.role() == eredu_runtime::ParameterRole::ExpertIntermediate {
            continue;
        }
        for parameter in group.members() {
            if let Some(format) = parameter.logical_linear_format() {
                let source_format = if routers.contains(parameter.target()) {
                    LinearFormat::Dense
                } else {
                    format.encoding()
                };
                tensors.extend(
                    matrix_for_linear_format(
                        parameter.target(),
                        Vec::<String>::new(),
                        parameter.global_shape().to_vec(),
                        source_format,
                        None,
                    )
                    .map_err(|error| error.to_string())?,
                );
            } else {
                tensors.push(floating(
                    parameter.target(),
                    parameter.global_shape().to_vec(),
                ));
            }
        }
    }
    tensors.extend(safetensors_expert_tensors(declaration)?);
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.push("rope_freqs.".into());
    policy.allowed_suffixes.push(".rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("GPT-OSS SafeTensors", tensors, Vec::new(), policy)
        .map_err(|error| error.to_string())
}

/// Returns the exact published SafeTensors storage owned by routed experts.
///
/// This keeps physical native-MXFP4 names and companions in the architecture
/// checkpoint contract instead of requiring backends to classify path text.
pub fn safetensors_expert_tensors(
    declaration: &super::Declaration,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let mut tensors = Vec::new();
    for bank in declaration
        .expert_banks()
        .map_err(|error| error.to_string())?
        .values()
    {
        tensors.extend(safetensors_expert_constraints(bank)?);
    }
    Ok(tensors)
}

/// Builds the exact canonical llama.cpp GPT-OSS GGUF catalog.
pub fn gguf_plan(declaration: &super::Declaration) -> Result<GgufCheckpointPlan, String> {
    let args = declaration.args();
    let declarations = super::parallel::parameter_description(
        declaration,
        crate::decoder::ParameterEncoding::Logical,
    )
    .map_err(|error| error.to_string())?;
    let mut tensors = Vec::new();
    for group in declarations.groups() {
        if group.role() == eredu_runtime::ParameterRole::ExpertIntermediate {
            continue;
        }
        for parameter in group.members() {
            let key = gguf_parameter_name(parameter.target(), &args.parameter_root)?;
            let constraint = gguf(
                &key,
                parameter.global_shape().to_vec(),
                if parameter.logical_linear_format().is_some() {
                    TensorOperation::Matrix
                } else {
                    TensorOperation::Vector
                },
            );
            tensors.push(
                if let Some(prefix) = key.strip_suffix(".attn_sinks.weight") {
                    constraint.with_aliases([format!("{prefix}.attn_sinks")])
                } else {
                    constraint
                },
            );
        }
    }
    for (layer, bank) in declaration
        .expert_banks()
        .map_err(|error| error.to_string())?
    {
        let source = bank
            .packed_parameters()
            .map_err(|error| error.to_string())?;
        for declaration in bank
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            let fused = declaration.parameter.id == source.gate_up().parameter.id
                || source
                    .gate_up_bias()
                    .is_some_and(|bias| bias.parameter.id == declaration.parameter.id);
            let shape = if fused {
                if declaration.role == LogicalParameterRole::LinearWeight {
                    source.unfused_shape()
                } else {
                    source
                        .unfused_bias_shape()
                        .ok_or("undeclared GPT-OSS gate/up bias")?
                }
            } else {
                declaration.shape
            };
            let (suffix, operation) = if declaration.role == LogicalParameterRole::LinearWeight {
                ("weight", TensorOperation::MxFp4Matrix)
            } else {
                ("bias", TensorOperation::Vector)
            };
            for field in if fused {
                &["ffn_gate_exps", "ffn_up_exps"][..]
            } else {
                &["ffn_down_exps"][..]
            } {
                tensors.push(gguf(
                    format!("blk.{layer}.{field}.{suffix}"),
                    shape.clone(),
                    operation,
                ));
            }
        }
    }
    GgufCheckpointPlan::new("GPT-OSS GGUF", tensors, Vec::new(), CatalogPolicy::strict())
        .map_err(|error| error.to_string())
}

/// Returns translated GGUF weight targets whose storage belongs to routed experts.
///
/// These are source-layout targets, before the architecture's expert recipes
/// normalize separate gate/up/down matrices into the runtime expert bank.
pub fn gguf_expert_quantization_targets(args: &ModelArgs) -> Result<Vec<String>, String> {
    args.validate().map_err(|error| error.to_string())?;
    Ok((0..dimension(args.num_hidden_layers, "layer count")?)
        .flat_map(|layer| {
            ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].map(move |projection| {
                translate_gguf_weight_name(&format!("blk.{layer}.{projection}.weight"))
            })
        })
        .collect())
}

/// Validates the portable GGUF catalog and matching gate/up physical encodings.
pub fn validate_gguf(
    checkpoint: &GgufCheckpoint,
    declaration: &super::Declaration,
) -> CheckpointValidation {
    let args = declaration.args();
    let plan = match gguf_plan(declaration) {
        Ok(plan) => plan,
        Err(error) => {
            return CheckpointValidation::Invalid(vec![
                eredu_checkpoint::validation::CheckpointIssue {
                    kind: eredu_checkpoint::validation::CheckpointIssueKind::InvalidGeometry,
                    detail: error,
                    tensor_name: None,
                    tensor_type_code: None,
                    metadata_key: None,
                },
            ])
        }
    };
    let mut issues = match validate_gguf_plan(checkpoint, &plan) {
        CheckpointValidation::Exact => Vec::new(),
        CheckpointValidation::Invalid(issues) => issues,
        CheckpointValidation::Unverified(issue) => vec![issue],
    };
    issues.extend(validate_matching_gguf_encodings(
        checkpoint,
        (0..args.num_hidden_layers.max(0) as usize).map(|layer| {
            (
                format!("blk.{layer}.ffn_gate_exps.weight"),
                format!("blk.{layer}.ffn_up_exps.weight"),
            )
        }),
        "GPT-OSS",
    ));
    CheckpointValidation::from_issues(issues)
}

const GGUF_ROOT_NAMES: &[(&str, &str)] = &[
    ("token_embd", "model.embed_tokens"),
    ("output_norm", "model.norm"),
    ("output", "lm_head"),
];
const GGUF_LAYER_NAMES: &[(&str, &str)] = &[
    ("attn_norm", "input_layernorm"),
    ("attn_post_norm", "post_attention_layernorm"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("ffn_gate_inp", "mlp.router"),
    ("ffn_gate_exps", "mlp.experts.gate_proj"),
    ("ffn_up_exps", "mlp.experts.up_proj"),
    ("ffn_down_exps", "mlp.experts.down_proj"),
];

/// Translates a physical llama.cpp tensor name to its neutral parameter identity.
pub fn translate_gguf_weight_name(name: &str) -> String {
    for &(source, target) in GGUF_ROOT_NAMES {
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
    if matches!(parameter, "attn_sinks" | "attn_sinks.weight") {
        return format!("model.layers.{layer}.self_attn.sinks");
    }
    for &(source, target) in GGUF_LAYER_NAMES {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.to_string()
}

fn gguf_parameter_name(name: &str, root: &str) -> Result<String, String> {
    for &(physical, canonical) in GGUF_ROOT_NAMES {
        let canonical = canonical.replacen("model", root, 1);
        if name == canonical
            || name
                .strip_prefix(&canonical)
                .is_some_and(|suffix| suffix.starts_with('.'))
        {
            return Ok(name.replacen(&canonical, physical, 1));
        }
    }
    if let Some((layer, parameter)) = name
        .strip_prefix(&format!("{root}.layers."))
        .and_then(|rest| rest.split_once('.'))
    {
        if parameter == "self_attn.sinks" {
            return Ok(format!("blk.{layer}.attn_sinks.weight"));
        }
        for &(physical, canonical) in GGUF_LAYER_NAMES {
            if parameter == canonical
                || parameter
                    .strip_prefix(canonical)
                    .is_some_and(|suffix| suffix.starts_with('.'))
            {
                return Ok(format!(
                    "blk.{layer}.{}",
                    parameter.replacen(canonical, physical, 1)
                ));
            }
        }
    }
    Err(format!(
        "GPT-OSS GGUF has no storage declaration for {name}"
    ))
}

/// Resolves SafeTensors alternating rows or translated GGUF separate matrices
/// into one atomic component-major logical-MXFP4 expert family.
pub fn expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    declaration: &super::Declaration,
    layer: usize,
) -> Result<GatedExpertProjectionFamilyRecipes, String> {
    let bank = declaration
        .expert_bank(layer)
        .map_err(|error| error.to_string())?;
    expert_recipes_for_bank(catalog, &bank)
}

fn expert_recipes_for_bank<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    bank: &eredu_nn::GroupedGatedProductSpec,
) -> Result<GatedExpertProjectionFamilyRecipes, String> {
    let source = bank
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    canonical_gated_expert_projection_family_recipes(catalog, &expert_names(&source)?)
        .map_err(|error| error.to_string())
}

/// Returns architecture-owned recipes for one rank-local GPT-OSS expert bank.
pub fn rank_local_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    declaration: &super::Declaration,
    layer: usize,
    group_indices: &[usize],
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    let declared = declaration
        .expert_bank(layer)
        .map_err(|error| error.to_string())?;
    let experts = declared
        .packed_parameters()
        .map_err(|error| error.to_string())?
        .group_count();
    let outputs = expert_recipes_for_bank(catalog, &declared)?
        .into_outputs()
        .into_outputs();
    eredu_runtime::select_parameter_bank_recipes(catalog, experts, 0, group_indices, outputs)
        .map_err(|error| error.to_string())
}

/// Builds the complete architecture-owned schedule for independently resident experts.
pub fn expert_residency_catalog<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    declaration: &super::Declaration,
) -> Result<crate::ExpertResidencyCatalog, String> {
    let args = declaration.args();
    let banks = declaration
        .expert_banks()
        .map_err(|error| error.to_string())?;
    let capacity = banks.values().try_fold(0usize, |capacity, bank| {
        capacity
            .checked_add(bank.group_count() as usize)
            .ok_or_else(|| "GPT-OSS expert residency catalog size overflowed".to_string())
    })?;
    let owner_group =
        eredu_runtime::ExecutionGroupId::new("text_decoder").map_err(|error| error.to_string())?;
    let mut units = Vec::with_capacity(capacity);
    for (layer, declared) in banks {
        let unit_path = format!("{}.layers.{layer}", args.parameter_root);
        let source = declared
            .packed_parameters()
            .map_err(|error| error.to_string())?;
        let expert_root = source
            .gate_up()
            .parameter
            .id
            .as_str()
            .strip_suffix(".gate_up_proj")
            .ok_or("GPT-OSS bank has no declared source root")?;
        let experts = source.group_count();
        let canonical = expert_recipes_for_bank(catalog, &declared)?
            .into_outputs()
            .into_outputs();
        let bank = eredu_runtime::ParameterBankRecipes::new(catalog, canonical, experts)
            .map_err(|error| error.to_string())?;
        for expert in 0..experts {
            let parameters = bank
                .member(expert)
                .expect("validated expert bank member")
                .iter()
                .map(|(target, recipe)| {
                    let binding = target
                        .strip_prefix(expert_root)
                        .and_then(|suffix| suffix.strip_prefix('.'))
                        .ok_or_else(|| {
                            format!(
                                "GPT-OSS expert target {target:?} is outside bank {expert_root:?}"
                            )
                        })?;
                    crate::ExpertParameterRecipe::new(
                        binding,
                        target,
                        recipe.clone(),
                        crate::ExpertParameterRole::Preserved,
                    )
                    .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, String>>()?;
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

fn expert_names(
    source: &eredu_nn::PackedGatedProductParameters<'_>,
) -> Result<GatedExpertProjectionFamilyNames, String> {
    let projection = |weight: String, scales: String, bias: String| ExpertProjectionFamilyNames {
        weight,
        scales,
        bias,
    };
    let declared = |weight: &eredu_nn::LogicalParameterDeclaration<'_>,
                    bias: Option<&eredu_nn::LogicalParameterDeclaration<'_>>|
     -> Result<ExpertProjectionFamilyNames, String> {
        let scale = weight
            .linear_format
            .and_then(|format| format.scale())
            .ok_or("GPT-OSS expert scale is undeclared")?;
        let bias = bias.ok_or("GPT-OSS expert additive bias is undeclared")?;
        Ok(projection(
            weight.parameter.id.to_string(),
            scale.id.to_string(),
            bias.parameter.id.to_string(),
        ))
    };
    let target_gate_up = declared(source.gate_up(), source.gate_up_bias())?;
    let target_down = declared(source.down(), source.down_bias())?;
    let alternating = |target: &ExpertProjectionFamilyNames| {
        projection(
            format!("{}_blocks", target.weight),
            target.scales.clone(),
            target.bias.clone(),
        )
    };
    let root = target_gate_up
        .weight
        .strip_suffix(".gate_up_proj")
        .ok_or("GPT-OSS bank has no declared source root")?;
    Ok(GatedExpertProjectionFamilyNames {
        alternating_gate_up: alternating(&target_gate_up),
        alternating_down: alternating(&target_down),
        separate_gate: projection(
            format!("{root}.gate_proj.weight"),
            format!("{root}.gate_proj.scales"),
            format!("{root}.gate_proj.bias"),
        ),
        separate_up: projection(
            format!("{root}.up_proj.weight"),
            format!("{root}.up_proj.scales"),
            format!("{root}.up_proj.bias"),
        ),
        separate_down: projection(
            format!("{root}.down_proj.weight"),
            format!("{root}.down_proj.scales"),
            format!("{root}.down_proj.bias"),
        ),
        target_gate_up,
        target_down,
    })
}

fn safetensors_expert_constraints(
    bank: &eredu_nn::GroupedGatedProductSpec,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let source = bank
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    let mut tensors = Vec::new();
    for (weight, bias) in [
        (source.gate_up(), source.gate_up_bias()),
        (source.down(), source.down_bias()),
    ] {
        let format = weight
            .linear_format
            .ok_or("GPT-OSS expert encoding declaration is missing")?;
        if format.encoding() != LinearFormat::MxFp4 {
            return Err("GPT-OSS released expert storage requires MXFP4".into());
        }
        let mut blocks = weight.shape.clone();
        let input = blocks
            .last_mut()
            .ok_or_else(|| "GPT-OSS expert weight is scalar".to_owned())?;
        if !input.is_multiple_of(MXFP4_GROUP_SIZE) {
            return Err("GPT-OSS MXFP4 expert input splits an encoded block".into());
        }
        *input /= MXFP4_GROUP_SIZE;
        let scales = blocks.clone();
        blocks.push(MXFP4_BLOCK_BYTES);
        let key = format!("{}_blocks", weight.parameter.id);
        let scale = format
            .scale()
            .ok_or_else(|| "GPT-OSS MXFP4 scale declaration is missing".to_owned())?;
        tensors.push(
            SafetensorsTensorConstraint::required(
                &key,
                blocks,
                StoredDtypeConstraint::Exact(StoredDtype::U8),
            )
            .with_linear_format(format.encoding()),
        );
        tensors.push(
            SafetensorsTensorConstraint::required(
                scale.id.as_str(),
                scales,
                StoredDtypeConstraint::Exact(StoredDtype::U8),
            )
            .linear_companion(&key, eredu_checkpoint::schema::LinearCompanionKind::Scale),
        );
        if let Some(bias) = bias {
            tensors.push(floating(bias.parameter.id.as_str(), bias.shape.clone()));
        }
    }
    Ok(tensors)
}

fn floating(key: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
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
        .ok_or_else(|| format!("GPT-OSS {name} must be positive, got {value}"))
}
