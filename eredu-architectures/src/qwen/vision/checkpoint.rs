//! Pure SafeTensors contract for the shared Qwen vision encoder.

use eredu_checkpoint::schema::{
    matrix_for_linear_format, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
    GgufTypeConstraint, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
    StoredDtypeConstraint, TensorOperation,
};

use super::{VisionConfig, VisionMode};

/// Builds the strict canonical vision catalog below an optional artifact prefix.
pub fn safetensors_plan(
    config: &VisionConfig,
    mode: VisionMode,
    prefix: &str,
) -> Result<SafetensorsCheckpointPlan, String> {
    config.validate(mode).map_err(|error| error.to_string())?;
    let hidden = dim(config.hidden_size)?;
    let intermediate = dim(config.intermediate_size)?;
    let merge = dim(config.spatial_merge_size)?;
    let merged = hidden
        .checked_mul(merge * merge)
        .ok_or("vision merger width overflow")?;
    let name = |relative: &str| {
        if prefix.is_empty() {
            relative.to_owned()
        } else {
            format!("{prefix}.{relative}")
        }
    };
    let mut tensors = vec![
        dense(
            name("pos_embed.weight"),
            vec![dim(config.num_position_embeddings)?, hidden],
        ),
        dense(
            name("patch_embed.proj.weight"),
            vec![
                hidden,
                dim(config.in_channels)?,
                dim(config.temporal_patch_size)?,
                dim(config.patch_size)?,
                dim(config.patch_size)?,
            ],
        ),
        dense(name("patch_embed.proj.bias"), vec![hidden]),
    ];
    for layer in 0..config.layer_count() {
        let root = format!("blocks.{layer}");
        for field in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"] {
            tensors.push(dense(name(&format!("{root}.{field}")), vec![hidden]));
        }
        matrix(
            config,
            &mut tensors,
            name(&format!("{root}.attn.qkv.weight")),
            "blocks",
            vec![3 * hidden, hidden],
        )?;
        tensors.push(dense(
            name(&format!("{root}.attn.qkv.bias")),
            vec![3 * hidden],
        ));
        matrix(
            config,
            &mut tensors,
            name(&format!("{root}.attn.proj.weight")),
            "blocks",
            vec![hidden, hidden],
        )?;
        tensors.push(dense(name(&format!("{root}.attn.proj.bias")), vec![hidden]));
        matrix(
            config,
            &mut tensors,
            name(&format!("{root}.mlp.linear_fc1.weight")),
            "blocks",
            vec![intermediate, hidden],
        )?;
        tensors.push(dense(
            name(&format!("{root}.mlp.linear_fc1.bias")),
            vec![intermediate],
        ));
        matrix(
            config,
            &mut tensors,
            name(&format!("{root}.mlp.linear_fc2.weight")),
            "blocks",
            vec![hidden, intermediate],
        )?;
        tensors.push(dense(
            name(&format!("{root}.mlp.linear_fc2.bias")),
            vec![hidden],
        ));
    }
    for merger in std::iter::once("merger".to_owned()).chain(
        (0..config.deepstack_layer_count()).map(|index| format!("deepstack_merger_list.{index}")),
    ) {
        let norm_width = if merger == "merger" { hidden } else { merged };
        tensors.push(dense(
            name(&format!("{merger}.norm.weight")),
            vec![norm_width],
        ));
        tensors.push(dense(
            name(&format!("{merger}.norm.bias")),
            vec![norm_width],
        ));
        matrix(
            config,
            &mut tensors,
            name(&format!("{merger}.linear_fc1.weight")),
            &merger,
            vec![merged, merged],
        )?;
        tensors.push(dense(
            name(&format!("{merger}.linear_fc1.bias")),
            vec![merged],
        ));
        matrix(
            config,
            &mut tensors,
            name(&format!("{merger}.linear_fc2.weight")),
            &merger,
            vec![dim(config.out_hidden_size)?, merged],
        )?;
        tensors.push(dense(
            name(&format!("{merger}.linear_fc2.bias")),
            vec![dim(config.out_hidden_size)?],
        ));
    }
    SafetensorsCheckpointPlan::new(
        "Qwen shared vision SafeTensors",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

/// Translates llama.cpp Qwen projector names into the shared vision namespace.
pub fn translate_gguf_weight_name(name: &str, deepstack_layers: &[i32]) -> String {
    if let Some(rest) = name.strip_prefix("v.deepstack.") {
        if let Some((layer, suffix)) = rest.split_once('.') {
            if let Ok(layer) = layer.parse::<i32>() {
                if let Some(index) = deepstack_layers.iter().position(|value| *value == layer) {
                    let suffix =
                        suffix
                            .replacen("fc1", "linear_fc1", 1)
                            .replacen("fc2", "linear_fc2", 1);
                    return format!("model.visual.deepstack_merger_list.{index}.{suffix}");
                }
            }
        }
    }
    for (source, target) in [
        ("v.position_embd", "model.visual.pos_embed"),
        ("v.patch_embd", "model.visual.patch_embed.proj"),
        ("v.post_ln", "model.visual.merger.norm"),
        ("mm.0", "model.visual.merger.linear_fc1"),
        ("mm.2", "model.visual.merger.linear_fc2"),
        ("v.blk", "model.visual.blocks"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            let mut translated = name.replacen(source, target, 1);
            if source == "v.blk" {
                translated = translated
                    .replace(".attn_out.", ".attn.proj.")
                    .replace(".attn_qkv.", ".attn.qkv.")
                    .replace(".ffn_up.", ".mlp.linear_fc1.")
                    .replace(".ffn_down.", ".mlp.linear_fc2.")
                    .replace(".ln1.", ".norm1.")
                    .replace(".ln2.", ".norm2.");
            }
            return translated;
        }
    }
    name.to_owned()
}

/// Builds the split Qwen3-VL/Qwen3.5 projector GGUF contract.
pub fn gguf_plan(
    vision: &VisionConfig,
    text_hidden_size: i32,
) -> Result<GgufCheckpointPlan, String> {
    let hidden = dim(vision.hidden_size)?;
    let intermediate = dim(vision.intermediate_size)?;
    let text_hidden = dim(text_hidden_size)?;
    let patch = dim(vision.patch_size)?;
    let merge = dim(vision.spatial_merge_size)?;
    let merger_hidden = hidden
        .checked_mul(merge)
        .and_then(|value| value.checked_mul(merge))
        .ok_or_else(|| "vision merger width overflows".to_string())?;
    let mut tensors = vec![
        gguf_tensor(
            "v.position_embd.weight",
            vec![dim(vision.num_position_embeddings)?, hidden],
            TensorOperation::Dense,
        ),
        gguf_tensor(
            "v.patch_embd.weight",
            vec![hidden, 3, patch, patch],
            TensorOperation::Dense,
        ),
        gguf_tensor(
            "v.patch_embd.weight.1",
            vec![hidden, 3, patch, patch],
            TensorOperation::Dense,
        ),
        gguf_tensor("v.patch_embd.bias", vec![hidden], TensorOperation::Dense),
    ];
    for layer in 0..vision.layer_count() {
        let root = format!("v.blk.{layer}");
        for (name, shape, operation) in [
            ("ln1.weight", vec![hidden], TensorOperation::Dense),
            ("ln1.bias", vec![hidden], TensorOperation::Dense),
            (
                "attn_qkv.weight",
                vec![3 * hidden, hidden],
                TensorOperation::Matrix,
            ),
            ("attn_qkv.bias", vec![3 * hidden], TensorOperation::Dense),
            (
                "attn_out.weight",
                vec![hidden, hidden],
                TensorOperation::Matrix,
            ),
            ("attn_out.bias", vec![hidden], TensorOperation::Dense),
            ("ln2.weight", vec![hidden], TensorOperation::Dense),
            ("ln2.bias", vec![hidden], TensorOperation::Dense),
            (
                "ffn_up.weight",
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            ("ffn_up.bias", vec![intermediate], TensorOperation::Dense),
            (
                "ffn_down.weight",
                vec![hidden, intermediate],
                TensorOperation::Matrix,
            ),
            ("ffn_down.bias", vec![hidden], TensorOperation::Dense),
        ] {
            tensors.push(gguf_tensor(format!("{root}.{name}"), shape, operation));
        }
    }
    for (name, shape, operation) in [
        ("v.post_ln.weight", vec![hidden], TensorOperation::Dense),
        ("v.post_ln.bias", vec![hidden], TensorOperation::Dense),
        (
            "mm.0.weight",
            vec![merger_hidden, merger_hidden],
            TensorOperation::Matrix,
        ),
        ("mm.0.bias", vec![merger_hidden], TensorOperation::Dense),
        (
            "mm.2.weight",
            vec![text_hidden, merger_hidden],
            TensorOperation::Matrix,
        ),
        ("mm.2.bias", vec![text_hidden], TensorOperation::Dense),
    ] {
        tensors.push(gguf_tensor(name, shape, operation));
    }
    for layer in vision.deepstack_layers() {
        let root = format!("v.deepstack.{layer}");
        for (name, shape, operation) in [
            ("norm.weight", vec![merger_hidden], TensorOperation::Dense),
            ("norm.bias", vec![merger_hidden], TensorOperation::Dense),
            (
                "fc1.weight",
                vec![merger_hidden, merger_hidden],
                TensorOperation::Matrix,
            ),
            ("fc1.bias", vec![merger_hidden], TensorOperation::Dense),
            (
                "fc2.weight",
                vec![text_hidden, merger_hidden],
                TensorOperation::Matrix,
            ),
            ("fc2.bias", vec![text_hidden], TensorOperation::Dense),
        ] {
            tensors.push(gguf_tensor(format!("{root}.{name}"), shape, operation));
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

fn matrix(
    config: &VisionConfig,
    output: &mut Vec<SafetensorsTensorConstraint>,
    physical: String,
    relative_root: &str,
    shape: Vec<usize>,
) -> Result<(), String> {
    let relative = physical
        .find(relative_root)
        .map(|index| &physical[index..])
        .unwrap_or(&physical);
    let format = config.linear_format(relative);
    output.extend(
        matrix_for_linear_format(physical, Vec::<String>::new(), shape, format, None)
            .map_err(|error| error.to_string())?,
    );
    Ok(())
}
fn dense(name: String, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(name, shape, StoredDtypeConstraint::Floating)
}
fn dim(value: i32) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|v| *v > 0)
        .ok_or_else(|| "vision dimension must be positive".into())
}
