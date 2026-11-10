//! Pure SafeTensors contract for the shared Qwen vision encoder.

use eredu_checkpoint::schema::{
    matrix_for_linear_format, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorOperation,
};

use super::VisionConfig;
use eredu_nn::{DeclareLogicalParameters, LogicalParameterRole};

/// Source matrix geometry from the same typed tower declarations as construction
/// and checkpoint admission. Patch kernels and position tables keep their roles.
pub(crate) fn source_linear_shapes(
    declaration: &super::Declaration,
) -> Result<std::collections::BTreeMap<String, Vec<usize>>, String> {
    Ok(declaration
        .logical_parameters()
        .map_err(|error| error.to_string())?
        .into_iter()
        .filter(|parameter| {
            parameter.role == LogicalParameterRole::LinearWeight && parameter.shape.len() == 2
        })
        .map(|parameter| (parameter.parameter.id.to_string(), parameter.shape))
        .collect())
}

/// Builds the strict canonical vision catalog below an optional artifact prefix.
pub fn safetensors_plan(
    declaration: &super::Declaration,
) -> Result<SafetensorsCheckpointPlan, String> {
    let mut tensors = Vec::new();
    for spec in &declaration.slots {
        for declaration in spec
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            if let Some(format) = declaration
                .linear_format
                .filter(|_| declaration.role == LogicalParameterRole::LinearWeight)
            {
                tensors.extend(
                    matrix_for_linear_format(
                        declaration.parameter.id.as_str(),
                        Vec::<String>::new(),
                        declaration.shape,
                        format.encoding(),
                        format
                            .scale()
                            .map(|scale| eredu_checkpoint::schema::MatrixScaleNames {
                                key: scale.id.to_string(),
                                aliases: Vec::new(),
                            }),
                    )
                    .map_err(|error| error.to_string())?,
                );
            } else {
                tensors.push(dense(
                    declaration.parameter.id.as_str().to_owned(),
                    declaration.shape,
                ));
            }
        }
    }
    SafetensorsCheckpointPlan::new(
        "Qwen shared vision SafeTensors",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

const GGUF_STATIC_NAMES: &[(&str, &str)] = &[
    ("v.position_embd", "pos_embed"),
    ("v.patch_embd", "patch_embed.proj"),
    ("v.post_ln", "merger.norm"),
    ("mm.0", "merger.linear_fc1"),
    ("mm.2", "merger.linear_fc2"),
];
const GGUF_BLOCK_FIELDS: &[(&str, &str)] = &[
    ("ln1", "norm1"),
    ("ln2", "norm2"),
    ("attn_qkv", "attn.qkv"),
    ("attn_out", "attn.proj"),
    ("ffn_up", "mlp.linear_fc1"),
    ("ffn_down", "mlp.linear_fc2"),
];
const GGUF_MERGER_FIELDS: &[(&str, &str)] = &[("fc1", "linear_fc1"), ("fc2", "linear_fc2")];

/// Translates llama.cpp Qwen projector names into the shared vision namespace.
pub fn translate_gguf_weight_name(name: &str, deepstack_layers: &[i32]) -> String {
    if let Some((layer, suffix)) = name
        .strip_prefix("v.deepstack.")
        .and_then(|rest| rest.split_once('.'))
    {
        if let Ok(layer) = layer.parse::<i32>() {
            if let Some(index) = deepstack_layers.iter().position(|value| *value == layer) {
                let suffix = GGUF_MERGER_FIELDS
                    .iter()
                    .fold(suffix.to_owned(), |suffix, (physical, canonical)| {
                        suffix.replacen(physical, canonical, 1)
                    });
                return format!("model.visual.deepstack_merger_list.{index}.{suffix}");
            }
        }
    }
    for &(source, target) in GGUF_STATIC_NAMES
        .iter()
        .chain(std::iter::once(&("v.blk", "blocks")))
    {
        if name == source || name.starts_with(&format!("{source}.")) {
            let mut translated = name.replacen(source, &format!("model.visual.{target}"), 1);
            if source == "v.blk" {
                for &(physical, canonical) in GGUF_BLOCK_FIELDS {
                    translated =
                        translated.replace(&format!(".{physical}."), &format!(".{canonical}."));
                }
            }
            return translated;
        }
    }
    name.to_owned()
}

/// Builds the split Qwen3-VL/Qwen3.5 projector GGUF contract.
pub(crate) fn gguf_plan(declaration: &super::Declaration) -> Result<GgufCheckpointPlan, String> {
    let configured = declaration.config();
    let scope_root = declaration.root();
    let mut tensors = Vec::new();
    for spec in &declaration.slots {
        for declaration in spec
            .logical_parameters()
            .map_err(|error| error.to_string())?
        {
            let name = declaration.parameter.id.as_str();
            let name = name
                .strip_prefix(&format!("{}.", scope_root))
                .unwrap_or(name);
            if name == "patch_embed.proj.weight" {
                // The published GGUF layout stores two temporal Conv2D slices.
                if declaration.shape[1] != 3 || declaration.shape[2] != 2 {
                    return Err(
                        "Qwen projector GGUF requires three channels and two temporal patch slices"
                            .into(),
                    );
                }
                let mut shape = declaration.shape;
                shape.remove(2);
                for key in ["v.patch_embd.weight", "v.patch_embd.weight.1"] {
                    tensors.push(gguf_tensor(key, shape.clone(), TensorOperation::Dense));
                }
            } else {
                tensors.push(gguf_tensor(
                    gguf_parameter_name(name, &configured)?,
                    declaration.shape,
                    if declaration.linear_format.is_some()
                        && declaration.role == LogicalParameterRole::LinearWeight
                    {
                        TensorOperation::Matrix
                    } else {
                        TensorOperation::Dense
                    },
                ));
            }
        }
    }
    GgufCheckpointPlan::new(
        "Qwen vision projector GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
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

fn dense(name: String, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(name, shape, StoredDtypeConstraint::Floating)
}
fn gguf_parameter_name(name: &str, config: &VisionConfig) -> Result<String, String> {
    for &(physical, canonical) in GGUF_STATIC_NAMES {
        if name
            .strip_prefix(canonical)
            .is_some_and(|suffix| suffix.starts_with('.'))
        {
            return Ok(name.replacen(canonical, physical, 1));
        }
    }
    if let Some((layer, suffix)) = name
        .strip_prefix("blocks.")
        .and_then(|rest| rest.split_once('.'))
    {
        for &(physical, canonical) in GGUF_BLOCK_FIELDS {
            if suffix
                .strip_prefix(canonical)
                .is_some_and(|suffix| suffix.starts_with('.'))
            {
                return Ok(format!(
                    "v.blk.{layer}.{}",
                    suffix.replacen(canonical, physical, 1)
                ));
            }
        }
    }
    if let Some((index, suffix)) = name
        .strip_prefix("deepstack_merger_list.")
        .and_then(|rest| rest.split_once('.'))
    {
        let index = index.parse::<usize>().map_err(|error| error.to_string())?;
        let layer = *config
            .deepstack_layers()
            .get(index)
            .ok_or("Qwen vision DeepStack declaration is unresolved")?;
        let suffix = GGUF_MERGER_FIELDS
            .iter()
            .fold(suffix.to_owned(), |suffix, (physical, canonical)| {
                suffix.replacen(canonical, physical, 1)
            });
        return Ok(format!("v.deepstack.{layer}.{suffix}"));
    }
    Err(format!(
        "Qwen vision GGUF has no storage declaration for {name}"
    ))
}

/// Reassembles the published two temporal Conv2D slices as one Conv3D kernel.
/// Callers retain exact source identity, encoding and dtype in the input recipes.
pub(crate) fn temporal_patch_recipe(
    inputs: [eredu_checkpoint::recipe::DerivedWeightRecipe; 2],
) -> eredu_checkpoint::recipe::DerivedWeightRecipe {
    eredu_checkpoint::recipe::DerivedWeightRecipe::Stack {
        axis: 2,
        inputs: inputs.into(),
    }
}
