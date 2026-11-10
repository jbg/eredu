//! Pure checkpoint naming, schemas, and derived-weight recipes for DeepSeek.

use std::{collections::BTreeMap, ops::Range};

use eredu_checkpoint::schema::{
    matrix_for_linear_format, AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan,
    GgufTensorConstraint, GgufTypeConstraint, LayoutVariant, MatrixScaleNames,
    SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::{
    expert::{resolve_gated_product_expert_recipes, GatedProductExpertRecipes},
    recipe::RecipeCatalog,
};
use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::TensorSelection};
use eredu_checkpoint::{LinearFormat, StoredDtype, WeightQuantization};

use super::config::{LayerPolicy, V3Args, V4Args};
use eredu_nn::{DeclareLogicalParameters, LogicalParameterRole};

mod expert_sources;

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
    for (source, target) in V3_GGUF_ROOTS {
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
    for (source, target) in V3_GGUF_EXPERT_PARAMETERS {
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
    for (source, target) in V3_GGUF_PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.to_string()
}

const V3_GGUF_ROOTS: [(&str, &str); 3] = [
    ("token_embd", "model.embed_tokens"),
    ("output_norm", "model.norm"),
    ("output", "lm_head"),
];

const V3_GGUF_EXPERT_PARAMETERS: [(&str, &str); 3] = [
    ("ffn_gate_exps", "mlp.experts.gate_proj"),
    ("ffn_up_exps", "mlp.experts.up_proj"),
    ("ffn_down_exps", "mlp.experts.down_proj"),
];

const V3_GGUF_PARAMETERS: [(&str, &str); 20] = [
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
];

fn v3_gguf_parameter_name(parameter: &eredu_nn::ParameterSpec) -> Result<String, String> {
    let canonical = parameter.id.as_str();
    for (source, target) in V3_GGUF_ROOTS {
        if canonical == target || canonical.starts_with(&format!("{target}.")) {
            return Ok(canonical.replacen(target, source, 1));
        }
    }
    if let Some((layer, parameter)) = canonical
        .strip_prefix("model.layers.")
        .and_then(|s| s.split_once('.'))
    {
        if parameter == "mlp.gate.e_score_correction_bias" {
            return Ok(format!("blk.{layer}.exp_probs_b.bias"));
        }
        for (source, target) in V3_GGUF_PARAMETERS {
            if parameter == target || parameter.starts_with(&format!("{target}.")) {
                return Ok(format!(
                    "blk.{layer}.{}",
                    parameter.replacen(target, source, 1)
                ));
            }
        }
    }
    Err(format!("unresolved V3 GGUF declaration {canonical}"))
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
    args.with_formats(quantization.into(), BTreeMap::new())
        .map_err(|e| e.to_string())
}

/// Derives a V4 configuration whose target and MTP expert formats reflect load-time quantization.
pub fn v4_load_time_quantization(
    args: &V4Args,
    quantization: WeightQuantization,
) -> Result<V4Args, String> {
    quantization.validate().map_err(|error| error.to_string())?;
    let mut formats = BTreeMap::new();
    for bank in super::v4::expert_bank_specs(args)
        .map_err(|e| e.to_string())?
        .values()
    {
        let eredu_nn::GatedProductGroupLayout::Packed { gate_up, down } = bank.layout() else {
            return Err("V4 requires packed experts".into());
        };
        for p in [gate_up, down] {
            formats.insert(p.weight().id.to_string(), quantization.into());
        }
    }
    args.with_formats(quantization.into(), formats)
        .map_err(|e| e.to_string())
}

/// Applies canonical checkpoint formats to a complete DeepSeek V3
/// configuration.
pub fn v3_with_checkpoint_formats(
    args: &V3Args,
    mut formats: BTreeMap<String, LinearFormat>,
) -> Result<V3Args, String> {
    normalize_v3_weight_formats(args, &mut formats);
    args.with_formats(args.linear_format, formats)
        .map_err(|e| e.to_string())
}

/// Applies canonical checkpoint formats to a complete DeepSeek V4
/// configuration.
pub fn v4_with_checkpoint_formats(
    args: &V4Args,
    mut formats: BTreeMap<String, LinearFormat>,
) -> Result<V4Args, String> {
    normalize_v4_weight_formats(args, &mut formats);
    args.with_formats(args.linear_format, formats)
        .map_err(|e| e.to_string())
}

pub(super) fn validate_v3_format_references(args: &V3Args) -> Result<(), eredu_nn::Error> {
    crate::format_references::validate(
        args.parameters(),
        args.linear_formats
            .iter()
            .map(|(name, format)| (name.as_str(), Some(*format))),
        || {
            let target = args.prediction_target().map_err(|e| e.to_string())?;
            Ok(gguf_format_references(
                v3_gguf_plan(&target)?,
                translate_v3_gguf_weight_name,
                |name| args.linear_format_for(name),
            ))
        },
        || v3_safetensors_plan(args, true),
    )
}

pub(super) fn validate_v4_format_references(args: &V4Args) -> Result<(), eredu_nn::Error> {
    crate::format_references::validate(
        args.parameters(),
        args.linear_formats
            .iter()
            .map(|(name, format)| (name.as_str(), Some(*format))),
        || {
            let target = args.prediction_target().map_err(|e| e.to_string())?;
            let mut references = gguf_format_references(
                v4_gguf_plan(&target)?,
                translate_v4_gguf_weight_name,
                |name| args.linear_format_for(name),
            );
            // Both normalized packed-bank spellings resolve to the same native source.
            references.extend(
                references
                    .iter()
                    .filter(|(name, _)| name.contains(".expert_banks."))
                    .filter_map(|(name, format)| {
                        name.strip_suffix(".weight")
                            .map(|name| (name.to_owned(), *format))
                    })
                    .collect::<Vec<_>>(),
            );
            Ok(references)
        },
        || v4_safetensors_plan(args),
    )
}

/// Projects primary GGUF matrix references without applying SafeTensors word packing.
fn gguf_format_references(
    plan: GgufCheckpointPlan,
    translate: impl Fn(&str) -> String,
    ordinary_format: impl Fn(&str) -> LinearFormat,
) -> Vec<(String, LinearFormat)> {
    plan.common_tensors
        .into_iter()
        .chain(
            plan.layout_groups
                .into_iter()
                .flat_map(|group| group.variants)
                .flat_map(|variant| variant.tensors),
        )
        .filter_map(|tensor| {
            let name = translate(&tensor.key);
            let format = match tensor.encoding {
                GgufTypeConstraint::OperationClass(TensorOperation::Matrix) => {
                    ordinary_format(&name)
                }
                GgufTypeConstraint::OperationClass(TensorOperation::MxFp4Matrix) => {
                    LinearFormat::MxFp4
                }
                _ => return None,
            };
            Some((name, format))
        })
        .collect()
}

/// Builds the canonical llama.cpp DeepSeek2 GGUF plan. Fused and split MLA
/// KV-B storage are one global alternative layout so a checkpoint cannot mix
/// the two representations across layers.
pub fn v3_gguf_plan(args: &V3Args) -> Result<GgufCheckpointPlan, String> {
    args.validate().map_err(|error| error.to_string())?;
    if args.num_nextn_predict_layers != 0 {
        return Err("DeepSeek2 base GGUF does not embed MTP layers".into());
    }
    let statics = &args.parameters().statics;
    let mut tensors = Vec::new();
    add_v3_gguf_declared(&mut tensors, statics.embedding_spec())?;
    add_v3_gguf_declared(&mut tensors, statics.normalization_spec())?;
    add_v3_gguf_declared(
        &mut tensors,
        statics.head_spec().expect("untied V3 readout"),
    )?;
    let mut fused = Vec::new();
    let mut fused_keys = Vec::new();
    let mut split = Vec::new();
    let mut split_keys = Vec::new();
    for (layer, policy) in args.layer_schedule.iter().copied().enumerate() {
        let root = format!("model.layers.{layer}");
        let block = format!("blk.{layer}");
        for field in ["input_layernorm", "post_attention_layernorm"] {
            let spec = super::v3_specs::normalization_spec(
                args,
                &format!("{root}.{field}.weight"),
                args.hidden_size,
            )
            .map_err(|e| e.to_string())?;
            add_v3_gguf_declared(&mut tensors, &spec)?;
        }
        let attention = super::v3_specs::attention_spec(args, &format!("{root}.self_attn"))
            .map_err(|e| e.to_string())?;
        add_v3_gguf_declared(&mut tensors, &attention.query)?;
        add_v3_gguf_declared(&mut tensors, &attention.kv_a)?;
        add_v3_gguf_declared(&mut tensors, &attention.kv_norm)?;
        add_v3_gguf_declared(&mut tensors, &attention.output)?;
        let parameter = attention
            .kv_b
            .logical_parameters()
            .map_err(|e| e.to_string())?
            .into_iter()
            .next()
            .expect("declared KV-B weight");
        let fused_key = v3_gguf_parameter_name(parameter.parameter)?;
        fused_keys.push(fused_key.clone());
        fused.push(gguf_tensor(
            fused_key,
            parameter.shape,
            TensorOperation::Matrix,
        ));
        // GGUF split keys transpose their final two axes before concatenation.
        for (local, shape) in [
            (
                "attn_k_b.weight",
                vec![
                    attention.geometry.heads() as usize,
                    attention.kv_b.input as usize,
                    attention.geometry.non_positional_dimensions() as usize,
                ],
            ),
            (
                "attn_v_b.weight",
                vec![
                    attention.geometry.heads() as usize,
                    attention.geometry.value_dimensions() as usize,
                    attention.kv_b.input as usize,
                ],
            ),
        ] {
            let key = format!("{block}.{local}");
            split_keys.push(key.clone());
            split.push(gguf_tensor(key, shape, TensorOperation::Matrix));
        }
        if policy == LayerPolicy::SparseMoe {
            let moe = super::v3::routed_spec(args, layer, None).map_err(|e| e.to_string())?;
            add_v3_gguf_declared(&mut tensors, &moe.parameters.selector)?;
            add_v3_gguf_declared(&mut tensors, &moe.shared)?;
            let bank = &moe.parameters.experts;
            let source = bank
                .packed_parameters()
                .map_err(|error| error.to_string())?;
            let read = source.unfused_shape();
            for (local, shape) in [
                ("ffn_gate_exps.weight", read.clone()),
                ("ffn_up_exps.weight", read.clone()),
                ("ffn_down_exps.weight", source.down().shape.clone()),
            ] {
                tensors.push(gguf_tensor(
                    format!("{block}.{local}"),
                    shape,
                    TensorOperation::Matrix,
                ));
            }
        } else {
            let parameters = super::v3_specs::dense_parameters(args, &format!("{root}.mlp"))
                .map_err(|e| e.to_string())?;
            add_v3_gguf_declared(&mut tensors, &parameters)?;
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

fn add_v3_gguf_declared(
    tensors: &mut Vec<GgufTensorConstraint>,
    spec: &impl DeclareLogicalParameters,
) -> Result<(), String> {
    for parameter in spec.logical_parameters().map_err(|e| e.to_string())? {
        tensors.push(gguf_tensor(
            v3_gguf_parameter_name(parameter.parameter)?,
            parameter.shape,
            declared_operation(parameter.role),
        ));
    }
    Ok(())
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
    let spec = super::v3_specs::attention_spec(args, &format!("model.layers.{layer}.self_attn"))
        .map_err(|e| e.to_string())?;
    let (_, recipe) = v3_split_kv_b_recipe(
        &spec.kv_b,
        format!("{root}.attn_k_b.weight"),
        format!("{root}.attn_v_b.weight"),
    )?;
    Ok(recipe)
}

/// Applies the split storage axes to the construction-owned fused matrix.
fn v3_split_kv_b_recipe(
    spec: &eredu_nn::LinearSpec,
    key: String,
    value: String,
) -> Result<(String, DerivedWeightRecipe), String> {
    let parameter = spec
        .logical_parameters()
        .map_err(|error| error.to_string())?
        .into_iter()
        .find(|parameter| parameter.parameter.id == spec.weight.id)
        .ok_or_else(|| "unresolved declared KV-B weight".to_string())?;
    let recipe = DerivedWeightRecipe::Reshape {
        input: Box::new(DerivedWeightRecipe::Concatenate {
            axis: 1,
            inputs: vec![
                DerivedWeightRecipe::Transpose {
                    input: Box::new(DerivedWeightRecipe::source(key, TensorSelection::Full)),
                    axes: vec![0, 2, 1],
                },
                DerivedWeightRecipe::source(value, TensorSelection::Full),
            ],
        }),
        shape: parameter.shape,
    };
    Ok((parameter.parameter.id.to_string(), recipe))
}

/// Returns the complete neutral recipe catalog for one DeepSeek V3 unit.
pub fn v3_unit_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &V3Args,
    layer: usize,
    include_experts: bool,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    args.validate().map_err(|error| error.to_string())?;
    let target = count(args.num_hidden_layers, "target layer count")?;
    let prediction = count(args.num_nextn_predict_layers, "prediction layer count")?;
    let total = target
        .checked_add(prediction)
        .ok_or_else(|| "V3 layer count overflowed".to_string())?;
    if layer >= total {
        return Err(format!(
            "V3 unit {layer} is outside {total} declared layers"
        ));
    }
    let mut recipes = BTreeMap::new();
    let physical = format!("blk.{layer}");
    let logical = format!("model.layers.{layer}.self_attn");
    let sources = if catalog
        .tensor_metadata(&format!("{logical}.k_b_proj.weight"))
        .is_ok()
    {
        Some((
            format!("{logical}.k_b_proj.weight"),
            format!("{logical}.v_b_proj.weight"),
        ))
    } else if catalog
        .tensor_metadata(&format!("{physical}.attn_k_b.weight"))
        .is_ok()
    {
        // The published GGUF layout contains target layers only.
        if layer >= target {
            return Err(format!(
                "DeepSeek2 layer {layer} is outside {target} layers"
            ));
        }
        Some((
            format!("{physical}.attn_k_b.weight"),
            format!("{physical}.attn_v_b.weight"),
        ))
    } else {
        None
    };
    if let Some((key, value)) = sources {
        let spec =
            super::v3_specs::attention_spec(args, &logical).map_err(|error| error.to_string())?;
        let (target, recipe) = v3_split_kv_b_recipe(&spec.kv_b, key, value)?;
        recipes.insert(target, recipe);
    }

    if include_experts
        && (layer >= target || args.layer_schedule.get(layer) == Some(&LayerPolicy::SparseMoe))
    {
        let expert = v3_expert_recipes(catalog, args, layer)?;
        recipes.extend(
            eredu_checkpoint::expert::gated_product_materialization_recipes(catalog, &expert)?,
        );
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
    let bank = super::v3::expert_bank_spec(args, layer).map_err(|error| error.to_string())?;
    let source = bank
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    let root = format!("model.layers.{layer}.mlp.experts");
    resolve_gated_product_expert_recipes(catalog, &expert_sources::v3_names(&source, &root))
        .map_err(|error| error.to_string())
}

/// Resolves SafeTensors independent experts or GGUF projection banks into the
/// packed V4 expert representation consumed by target, MTP, and DSpark blocks.
pub fn v4_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    args: &V4Args,
    layer: usize,
) -> Result<GatedProductExpertRecipes, String> {
    let bank = super::v4::expert_bank_spec(args, layer).map_err(|e| e.to_string())?;
    let source = bank
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    let physical = v4_expert_source_root(args, layer)?;
    resolve_gated_product_expert_recipes(catalog, &expert_sources::v4_names(&source, &physical))
        .map_err(|error| error.to_string())
}

fn v4_expert_source_root(args: &V4Args, layer: usize) -> Result<String, String> {
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
    Ok(format!("{block}.ffn"))
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
    let mut gate_up = eredu_runtime::select_parameter_recipe(
        catalog,
        &bank.target_gate_up,
        &bank.gate_up,
        Some(std::slice::from_ref(&expert_selection)),
    )
    .map_err(|error| error.to_string())?;
    let mut down = eredu_runtime::select_parameter_recipe(
        catalog,
        &bank.target_down,
        &bank.down,
        Some(std::slice::from_ref(&expert_selection)),
    )
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
        gate_up = eredu_runtime::select_parameter_recipe(
            catalog,
            &bank.target_gate_up,
            &gate_up,
            Some(&[
                TensorSelection::Range {
                    axis: 1,
                    start: intermediate.start,
                    end: intermediate.end,
                },
                TensorSelection::Range {
                    axis: 1,
                    start: width + intermediate.start,
                    end: width + intermediate.end,
                },
            ]),
        )
        .map_err(|error| error.to_string())?;
        down = eredu_runtime::select_parameter_recipe(
            catalog,
            &bank.target_down,
            &down,
            Some(&[TensorSelection::Range {
                axis: 2,
                start: intermediate.start,
                end: intermediate.end,
            }]),
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
    let (_, layout) = super::parallel::execution_layout(
        count(args.num_hidden_layers, "V3 target layer count")?,
        count(args.num_nextn_predict_layers, "V3 prediction layer count")?,
    )
    .map_err(|error| error.to_string())?;
    let mut units = Vec::new();
    for (layer, declared) in
        super::v3::expert_bank_specs(args).map_err(|error| error.to_string())?
    {
        let (owner, owner_unit) =
            super::expert_unit_owner(&layout, layer).map_err(|error| error.to_string())?;
        let unit_path = format!("model.layers.{layer}");
        let bank = v3_expert_recipes(catalog, args, layer)?;
        let source = declared
            .packed_parameters()
            .map_err(|error| error.to_string())?;
        append_expert_residency_units(
            &mut units,
            catalog,
            layer,
            &owner,
            owner_unit,
            &unit_path,
            source.group_count(),
            &bank,
            intermediate.clone(),
            source.encodings().map_err(|error| error.to_string())?,
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
    let (_, layout) = super::parallel::execution_layout(
        count(args.num_hidden_layers, "V4 target layer count")?,
        count(args.num_nextn_predict_layers, "V4 prediction layer count")?,
    )
    .map_err(|error| error.to_string())?;
    let banks = super::v4::expert_bank_specs(args).map_err(|error| error.to_string())?;
    let capacity = banks.values().try_fold(0usize, |capacity, bank| {
        capacity
            .checked_add(bank.group_count() as usize)
            .ok_or_else(|| "V4 expert residency catalog size overflowed".to_string())
    })?;
    let mut units = Vec::with_capacity(capacity);
    for (layer, declared) in banks {
        let (owner, owner_unit) =
            super::expert_unit_owner(&layout, layer).map_err(|error| error.to_string())?;
        let target = layout.group_range(0).expect("declared target group").len();
        let unit_path = if layer < target {
            format!("layers.{layer}")
        } else {
            format!("mtp.{}", layer - target)
        };
        let bank = v4_expert_recipes(catalog, args, layer)?;
        let source = declared
            .packed_parameters()
            .map_err(|error| error.to_string())?;
        append_expert_residency_units(
            &mut units,
            catalog,
            layer,
            &owner,
            owner_unit,
            &unit_path,
            source.group_count(),
            &bank,
            intermediate.clone(),
            source.encodings().map_err(|error| error.to_string())?,
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
    owner_group: &eredu_runtime::ExecutionGroupId,
    owner_unit: usize,
    unit_path: &str,
    experts: usize,
    bank: &GatedProductExpertRecipes,
    intermediate: Option<Range<usize>>,
    formats: [LinearFormat; 2],
) -> Result<(), String> {
    let outputs = eredu_checkpoint::expert::gated_product_materialization_recipes(catalog, bank)?;
    let selected_outputs = outputs
        .into_iter()
        .filter(|(target, _)| {
            [&bank.target_gate_up, &bank.target_down]
                .into_iter()
                .any(|base| {
                    ["", "_scales", "_biases"]
                        .into_iter()
                        .any(|suffix| *target == format!("{base}{suffix}"))
                })
        })
        .collect();
    let outputs = eredu_runtime::ParameterBankRecipes::new(catalog, selected_outputs, experts)
        .map_err(|error| error.to_string())?;
    for expert in 0..experts {
        let parameters = super::encoded_expert_residency::unit_parameters(
            catalog,
            bank,
            &outputs,
            expert,
            intermediate.clone(),
            formats,
        )?;
        units.push(
            crate::ExpertResidencyUnit::new(
                eredu_runtime::ParameterBankKey::new(0, identity_layer, expert),
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
    let statics = &args.parameters().statics;
    for parameters in [
        statics.embedding_spec().logical_parameters(),
        statics.normalization_spec().logical_parameters(),
        statics
            .head_spec()
            .expect("untied V3 readout")
            .logical_parameters(),
    ] {
        for parameter in parameters.map_err(|e| e.to_string())? {
            add_v3_tensor(
                &mut tensors,
                parameter.parameter.id.as_str(),
                parameter.shape,
                declared_operation(parameter.role),
                LinearFormat::Dense,
            )?;
        }
    }
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
        let prediction =
            super::v3_specs::prediction_spec(args, &root).map_err(|e| e.to_string())?;
        add_v3_declared(args, &mut tensors, &prediction)?;
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
    let root = format!("model.layers.{layer}");
    for field in ["input_layernorm", "post_attention_layernorm"] {
        let spec = super::v3_specs::normalization_spec(
            args,
            &format!("{root}.{field}.weight"),
            args.hidden_size,
        )
        .map_err(|e| e.to_string())?;
        add_v3_declared(args, tensors, &spec)?;
    }
    let attention = super::v3_specs::attention_spec(args, &format!("{root}.self_attn"))
        .map_err(|e| e.to_string())?;
    add_v3_declared(args, tensors, &attention)?;
    if policy == LayerPolicy::SparseMoe {
        let moe = super::v3::routed_spec(args, layer, None).map_err(|e| e.to_string())?;
        add_v3_declared(args, tensors, &moe.parameters.selector)?;
        add_v3_declared(args, tensors, &moe.shared)?;
        let bank = &moe.parameters.experts;
        let source = bank
            .packed_parameters()
            .map_err(|error| error.to_string())?;
        let expert_root = format!("{root}.mlp.experts");
        visit_v3_expert_sources(
            &source,
            &expert_root,
            |projection, packed_name, independent, shape| {
                groups.push(v3_expert_group(
                    args,
                    &expert_root,
                    projection,
                    packed_name,
                    independent,
                    shape,
                    allow_packed_experts,
                )?);
                Ok(())
            },
        )?;
    } else {
        let parameters = super::v3_specs::dense_parameters(args, &format!("{root}.mlp"))
            .map_err(|e| e.to_string())?;
        add_v3_declared(args, tensors, &parameters)?;
    }
    Ok(())
}

fn declared_operation(role: LogicalParameterRole) -> TensorOperation {
    if matches!(
        role,
        LogicalParameterRole::LinearWeight | LogicalParameterRole::Embedding
    ) {
        TensorOperation::Matrix
    } else {
        TensorOperation::Vector
    }
}

fn add_v3_declared(
    args: &V3Args,
    tensors: &mut Vec<SafetensorsTensorConstraint>,
    spec: &impl DeclareLogicalParameters,
) -> Result<(), String> {
    for parameter in spec.logical_parameters().map_err(|e| e.to_string())? {
        let operation = declared_operation(parameter.role);
        let name = parameter.parameter.id.as_str();
        add_v3_tensor(
            tensors,
            name,
            parameter.shape,
            operation,
            v3_format(args, name, operation),
        )?;
    }
    Ok(())
}

/// Logical ordinary and independent-expert source matrix geometry. Storage
/// layouts reference the retained construction slots before scalar encoding.
pub(crate) fn v3_source_linear_shapes(
    args: &V3Args,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    let declaration = super::parallel::v3_parameter_description(args).map_err(|e| e.to_string())?;
    let mut shapes = declaration
        .groups()
        .iter()
        .flat_map(|group| group.members())
        .filter(|member| member.logical_role() == Some(LogicalParameterRole::LinearWeight))
        .filter_map(|member| {
            member
                .logical_shape()
                .filter(|shape| shape.len() == 2)
                .map(|shape| (member.target().to_owned(), shape.to_vec()))
        })
        .collect::<BTreeMap<_, _>>();
    for bank in super::v3::expert_bank_specs(args)
        .map_err(|e| e.to_string())?
        .values()
    {
        let source = bank.packed_parameters().map_err(|e| e.to_string())?;
        let root = source
            .gate_up()
            .parameter
            .id
            .as_str()
            .strip_suffix(".gate_up_proj")
            .ok_or("DeepSeek-V3 expert source has no declared gate/up root")?;
        visit_v3_expert_sources(&source, root, |_, _, independent, shape| {
            for name in independent {
                shapes.insert(name.to_owned(), shape[1..].to_vec());
            }
            Ok(())
        })?;
    }
    Ok(shapes)
}

fn visit_v3_expert_sources(
    source: &eredu_nn::PackedGatedProductParameters<'_>,
    root: &str,
    mut visit: impl FnMut(&str, &str, Vec<&str>, Vec<usize>) -> Result<(), String>,
) -> Result<(), String> {
    let names = expert_sources::v3_names(source, root);
    let read = source.unfused_shape();
    for (projection, packed_name, independent, shape) in [
        (
            "gate_proj",
            names.separate_gate.as_str(),
            names
                .independent
                .iter()
                .map(|expert| expert.gate.as_str())
                .collect::<Vec<_>>(),
            read.clone(),
        ),
        (
            "up_proj",
            names.separate_up.as_str(),
            names
                .independent
                .iter()
                .map(|expert| expert.up.as_str())
                .collect::<Vec<_>>(),
            read,
        ),
        (
            "down_proj",
            names.separate_down.as_str(),
            names
                .independent
                .iter()
                .map(|expert| expert.down.as_str())
                .collect::<Vec<_>>(),
            source.down().shape.clone(),
        ),
    ] {
        visit(projection, packed_name, independent, shape)?;
    }
    Ok(())
}

fn v3_expert_group(
    args: &V3Args,
    root: &str,
    projection: &str,
    packed_name: &str,
    independent: Vec<&str>,
    packed_shape: Vec<usize>,
    allow_packed: bool,
) -> Result<AlternativeLayoutGroup<SafetensorsTensorConstraint>, String> {
    let mut split = Vec::new();
    let mut split_keys = Vec::new();
    for name in independent {
        split_keys.push(name.to_owned());
        split.extend(v3_matrix_constraints(
            name,
            packed_shape[1..].to_vec(),
            args.linear_format_for(name),
        )?);
    }
    let mut variants = vec![LayoutVariant {
        id: "split".into(),
        tensors: split,
        discriminator_keys: split_keys,
    }];
    if allow_packed {
        variants.push(LayoutVariant {
            id: "packed".into(),
            discriminator_keys: vec![packed_name.to_owned()],
            tensors: v3_matrix_constraints(
                packed_name,
                packed_shape,
                args.linear_format_for(packed_name),
            )?,
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
    role: LogicalParameterRole,
}

/// Logical matrices consumed by V4's configurable projection operators.
/// The checkpoint also contains rank-two integer routing tables, additive
/// compressor positions and dense hyper-connection coefficients. Those retain
/// their own storage and must never acquire a packed-linear transform merely
/// because their shapes have two axes.
pub(crate) fn v4_source_linear_shapes(
    args: &V4Args,
) -> Result<BTreeMap<String, Vec<usize>>, String> {
    args.validate().map_err(|error| error.to_string())?;
    let mut specs = v4_target_specs(args, V4ExpertStorage::Independent)?;
    append_v4_draft_specs(&mut specs, args)?;
    Ok(specs
        .into_iter()
        .filter(|spec| spec.role == LogicalParameterRole::LinearWeight)
        .map(|spec| (spec.name, spec.shape))
        .collect())
}

/// Builds the canonical DeepSeek-V4 SafeTensors catalog plan, including
/// native UE8M0 block-FP8 companions and typed integer hash tables.
pub fn v4_safetensors_plan(args: &V4Args) -> Result<SafetensorsCheckpointPlan, String> {
    args.validate().map_err(|error| error.to_string())?;
    let mut specs = v4_target_specs(args, V4ExpertStorage::Independent)?;
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
    let specs = v4_target_specs(args, V4ExpertStorage::GgufPacked)?;
    let mut tensors = Vec::new();
    for spec in specs {
        let physical = if spec.operation == TensorOperation::MxFp4Matrix {
            spec.name
        } else {
            v4_gguf_physical_name(&spec.name)?
        };
        tensors.push(gguf_tensor(physical, spec.shape, spec.operation));
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

#[derive(Clone, Copy)]
enum V4ExpertStorage {
    Independent,
    GgufPacked,
}

fn v4_target_specs(args: &V4Args, experts: V4ExpertStorage) -> Result<Vec<V4TensorSpec>, String> {
    let mut specs = v4_declared_specs(args, &args.parameters().statics)?;
    specs.extend(v4_declared_specs(
        args,
        &super::v4_specs::hyper_head_spec(args, "hc_head").map_err(|e| e.to_string())?,
    )?);
    for layer in 0..dimension(args.num_hidden_layers, "target layers")? {
        specs.extend(v4_block_specs(
            args,
            layer,
            &format!("layers.{layer}"),
            experts,
        )?);
    }
    Ok(specs)
}

fn v4_block_specs(
    args: &V4Args,
    layer: usize,
    root: &str,
    expert_storage: V4ExpertStorage,
) -> Result<Vec<V4TensorSpec>, String> {
    let mut specs = v4_declared_specs(
        args,
        &super::v4_specs::attention_spec(args, layer, &format!("{root}.attn"))
            .map_err(|e| e.to_string())?,
    )?;
    for normalization in
        super::v4_specs::block_normalizations(args, root).map_err(|e| e.to_string())?
    {
        specs.extend(v4_declared_specs(args, &normalization)?);
    }
    specs.extend(v4_declared_specs(
        args,
        &super::v4_specs::hyper_block_spec(args, root).map_err(|e| e.to_string())?,
    )?);
    let moe = super::v4_specs::moe_spec(args, layer, &format!("{root}.ffn"), None)
        .map_err(|e| e.to_string())?;
    // Published routers retain floating source storage, independently of materialization overrides.
    let mut router = v4_declared_specs(args, &moe.routed.parameters.selector)?;
    for tensor in &mut router {
        tensor.format = LinearFormat::Dense;
    }
    specs.extend(router);
    specs.extend(v4_declared_specs(args, &moe.routed.shared)?);
    if let Some(table) = &moe.token_table {
        specs.extend(v4_declared_specs(args, table)?);
    }
    let bank = &moe.routed.parameters.experts;
    let source = bank
        .packed_parameters()
        .map_err(|error| error.to_string())?;
    let read = source.unfused_shape();
    // Independent SafeTensors and packed GGUF banks are explicit layouts of the declared bank.
    match expert_storage {
        V4ExpertStorage::Independent => {
            let names = expert_sources::v4_names(&source, &format!("{root}.ffn"));
            for expert in &names.independent {
                for (name, shape) in [
                    (&expert.gate, read[1..].to_vec()),
                    (&expert.down, source.down().shape[1..].to_vec()),
                    (&expert.up, read[1..].to_vec()),
                ] {
                    specs.push(v4_spec(
                        name,
                        shape,
                        TensorOperation::Matrix,
                        args.expert_linear_format(),
                        LogicalParameterRole::LinearWeight,
                    ));
                }
            }
        }
        V4ExpertStorage::GgufPacked => {
            for (projection, shape) in [
                ("ffn_gate_exps.weight", read.clone()),
                ("ffn_down_exps.weight", source.down().shape.clone()),
                ("ffn_up_exps.weight", read),
            ] {
                specs.push(v4_spec(
                    format!("blk.{layer}.{projection}"),
                    shape,
                    TensorOperation::MxFp4Matrix,
                    LinearFormat::MxFp4,
                    LogicalParameterRole::LinearWeight,
                ));
            }
        }
    }
    Ok(specs)
}

fn append_v4_draft_specs(specs: &mut Vec<V4TensorSpec>, args: &V4Args) -> Result<(), String> {
    let drafts = count(args.num_nextn_predict_layers, "draft layer count")?;
    if drafts == 0 {
        return Ok(());
    }
    let targets = dimension(args.num_hidden_layers, "target layer count")?;
    for depth in 0..drafts {
        specs.extend(v4_block_specs(
            args,
            targets + depth,
            &format!("mtp.{depth}"),
            V4ExpertStorage::Independent,
        )?);
    }
    if args.dspark.is_some() {
        specs.extend(v4_declared_specs(
            args,
            &super::v4_specs::dspark_spec(args).map_err(|e| e.to_string())?,
        )?);
    } else {
        for depth in 0..drafts {
            specs.extend(v4_declared_specs(
                args,
                &super::v4_specs::prediction_spec(args, &format!("mtp.{depth}"))
                    .map_err(|e| e.to_string())?,
            )?);
        }
    }
    Ok(())
}

fn v4_declared_specs(
    args: &V4Args,
    spec: &impl DeclareLogicalParameters,
) -> Result<Vec<V4TensorSpec>, String> {
    spec.logical_parameters()
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|parameter| {
            let (operation, format) = match parameter.role {
                LogicalParameterRole::LinearWeight => (TensorOperation::Matrix, args.linear_format),
                LogicalParameterRole::Embedding | LogicalParameterRole::ResidualMixWeight => {
                    (TensorOperation::Matrix, LinearFormat::Dense)
                }
                LogicalParameterRole::RoutingTable => (TensorOperation::I32, LinearFormat::Dense),
                LogicalParameterRole::PositionBias => (TensorOperation::Dense, LinearFormat::Dense),
                LogicalParameterRole::Normalization
                | LogicalParameterRole::RoutingCorrection
                | LogicalParameterRole::AttentionSink
                | LogicalParameterRole::ResidualMixBias
                | LogicalParameterRole::ResidualMixScale => {
                    (TensorOperation::Vector, LinearFormat::Dense)
                }
                _ => return Err("unsupported V4 checkpoint declaration role".into()),
            };
            Ok(v4_spec(
                parameter.parameter.id.as_str(),
                parameter.shape,
                operation,
                format,
                parameter.role,
            ))
        })
        .collect()
}

fn v4_spec(
    name: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
    format: LinearFormat,
    role: LogicalParameterRole,
) -> V4TensorSpec {
    V4TensorSpec {
        name: name.into(),
        shape,
        operation,
        format,
        role,
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
