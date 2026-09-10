//! Exact family parameter encodings derived from neutral checkpoint metadata.
use super::{ConfigError, ModelArgs};
use eredu_checkpoint::LinearFormat;

pub(crate) fn normalize_source_formats(args: &mut ModelArgs) -> Result<(), ConfigError> {
    let Some(metadata) = args.fp8.clone() else {
        return Ok(());
    };
    for (name, shape) in super::parameter_shapes(args, false).map_err(ConfigError::Invalid)? {
        // The publisher's quantization target is Linear; embeddings are a
        // distinct operator even though they also own a rank-two matrix.
        if shape.len() == 2 && name != "model.embed_tokens.weight" && metadata.includes(&name) {
            args.formats
                .insert(name, LinearFormat::E4M3BlockFp8(metadata.format()));
        }
    }
    for layer in 0..args.num_hidden_layers as usize {
        if !args.is_sparse_layer(layer) {
            continue;
        }
        let root = format!("model.layers.{layer}.mlp.experts");
        if args.moe_intermediate_size % metadata.format().block_rows != 0
            && matches!(
                args.linear_format_for(&format!("{root}.0.gate_proj.weight")),
                LinearFormat::E4M3BlockFp8(_)
            )
        {
            return Err(ConfigError::Invalid(
                "packed FP8 gate and up projections must end at scale-block boundaries".into(),
            ));
        }
        for (target, fields) in [
            ("gate_up_proj", &["gate_proj", "up_proj"][..]),
            ("down_proj", &["down_proj"][..]),
        ] {
            let expected = args.linear_format_for(&format!("{root}.0.{}.weight", fields[0]));
            for expert in 0..args.num_experts {
                for field in fields {
                    if args.linear_format_for(&format!("{root}.{expert}.{field}.weight"))
                        != expected
                    {
                        return Err(ConfigError::Invalid(
                            "packed SwiGLU projections require homogeneous source encodings".into(),
                        ));
                    }
                }
            }
            args.formats.insert(format!("{root}.{target}"), expected);
        }
        if args.is_mova_layer(layer) {
            let root = format!("model.layers.{layer}.self_attn.v_experts");
            let expected = args.linear_format_for(&format!("{root}.0.weight"));
            for expert in 0..args.mova_num_experts {
                if args.linear_format_for(&format!("{root}.{expert}.weight")) != expected {
                    return Err(ConfigError::Invalid(
                        "packed value projections require homogeneous source encodings".into(),
                    ));
                }
            }
            args.formats.insert(format!("{root}.weight"), expected);
        }
    }
    args.validate()
}

/// Retains each GGUF encoding, then proves the two independently packed banks.
pub(crate) fn normalize_gguf_formats(
    args: &mut ModelArgs,
    checkpoint: &eredu_gguf::Checkpoint,
) -> Result<(), String> {
    for shard in checkpoint.shards() {
        for tensor in shard.tensors() {
            let name = super::translate_gguf_weight_name(&tensor.descriptor().name);
            if name.ends_with(".weight") && tensor.descriptor().dimensions.len() >= 2 {
                args.formats.insert(
                    name,
                    crate::linear_format::gguf_tensor_format(tensor, shard.endian())?,
                );
            }
        }
    }
    for layer in 0..args.num_hidden_layers as usize {
        if !args.is_sparse_layer(layer) {
            continue;
        }
        let root = format!("model.layers.{layer}.mlp.experts");
        let gate = args.linear_format_for(&format!("{root}.gate_proj.weight"));
        if gate != args.linear_format_for(&format!("{root}.up_proj.weight")) {
            return Err("packed SwiGLU gate and up require matching GGUF encodings".into());
        }
        args.formats.insert(format!("{root}.gate_up_proj"), gate);
        args.formats.insert(
            format!("{root}.down_proj"),
            args.linear_format_for(&format!("{root}.down_proj.weight")),
        );
    }
    args.validate().map_err(|e| e.to_string())
}
