//! Qwen3-Next text model support.
//!
//! Qwen3-Next and Qwen3.5 share the same hybrid Gated DeltaNet/full-attention
//! decoder and shared-expert MoE building blocks. This module exposes the
//! architecture-specific loading API while reusing that implementation.

#[cfg(test)]
use eredu_checkpoint::AffineQuantization;

use std::path::Path;

#[cfg(test)]
use safemlx::{
    ops::{concatenate_axis, indexing::TryIndexOp, quantized_packed_dimension},
    transforms::eval,
    Array, Dtype, Stream,
};
use tokenizers::Tokenizer;

pub use super::qwen3_5::{
    sample, Cache, Generate, LayerCache, LayerPolicy, LinearAttentionCache, Model, ModelArgs,
    ModelInput,
};
use crate::backend::mlx::error::Error;

/// Reads and normalizes Qwen3-Next model arguments from `config.json`.
pub fn get_qwen3_next_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model_dir.as_ref().join("config.json"))?)?;
    model_args_from_config_value(&config)
}

/// Normalizes a Qwen3-Next JSON configuration into executable text geometry.
pub fn model_args_from_config_value(config: &serde_json::Value) -> Result<ModelArgs, Error> {
    let (args, image_token_id, video_token_id, vision_config) =
        super::qwen3_5::parse_qwen3_5_config_value(config.clone())?;
    if image_token_id.is_some() || video_token_id.is_some() || vision_config.is_some() {
        return Err(Error::UnsupportedArchitecture(
            "qwen3_next is a text-only architecture".into(),
        ));
    }
    super::qwen3_5::validate_text_model_args(&args, "Qwen3-Next")?;
    fused_projection_widths(&args)?;
    Ok(args)
}

/// Loads `tokenizer.json` from a Qwen3-Next model directory.
pub fn load_qwen3_next_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    super::qwen3_5::load_qwen3_5_tokenizer(model_dir)
}

#[cfg(test)]
pub(crate) fn split_fused_projection(
    key: &str,
    value: Array,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<Vec<(String, Array)>, Error> {
    split_fused_projection_with_affine(key, value, None, args, stream)
}

#[cfg(test)]
pub(crate) fn split_fused_projection_with_affine(
    key: &str,
    value: Array,
    affine: Option<AffineQuantization>,
    args: &ModelArgs,
    stream: &Stream,
) -> Result<Vec<(String, Array)>, Error> {
    let (qkvz_widths, ba_width) = fused_projection_widths(args)?;

    let qkvz_scale_suffix = "linear_attn.in_proj_qkvz.weight_scale_inv";
    if let Some(prefix) = key.strip_suffix(qkvz_scale_suffix) {
        let block_widths = fp8_block_row_widths(&qkvz_widths)?;
        let parts = split_grouped_rows(value, args.linear_num_key_heads, &block_widths, stream)?;
        let qkv = concatenate_axis(&parts[..3], 0, stream)?;
        return evaluate_fused_projection_outputs(vec![
            (
                format!("{prefix}linear_attn.in_proj_qkv.weight_scale_inv"),
                qkv,
            ),
            (
                format!("{prefix}linear_attn.in_proj_z.weight_scale_inv"),
                parts[3].clone(),
            ),
        ]);
    }

    if key.ends_with("linear_attn.in_proj_ba.weight_scale_inv") {
        return Err(Error::UnsupportedArchitecture(
            "Qwen3-Next in_proj_ba must remain dense BF16 and cannot carry FP8 inverse scales"
                .into(),
        ));
    }

    for suffix in ["weight", "scales", "biases"] {
        let qkvz_suffix = format!("linear_attn.in_proj_qkvz.{suffix}");
        if let Some(prefix) = key.strip_suffix(&qkvz_suffix) {
            if suffix == "weight" && args.uses_fp8() {
                fp8_block_row_widths(&qkvz_widths)?;
            }
            if let Some(affine) = affine {
                validate_affine_fused_component(key, &value, suffix, affine, args.hidden_size)?;
            }
            let parts = split_grouped_rows(value, args.linear_num_key_heads, &qkvz_widths, stream)?;
            let qkv = concatenate_axis(&parts[..3], 0, stream)?;
            return evaluate_fused_projection_outputs(vec![
                (format!("{prefix}linear_attn.in_proj_qkv.{suffix}"), qkv),
                (
                    format!("{prefix}linear_attn.in_proj_z.{suffix}"),
                    parts[3].clone(),
                ),
            ]);
        }

        let ba_suffix = format!("linear_attn.in_proj_ba.{suffix}");
        if let Some(prefix) = key.strip_suffix(&ba_suffix) {
            if let Some(affine) = affine {
                validate_affine_fused_component(key, &value, suffix, affine, args.hidden_size)?;
            }
            let parts = split_grouped_rows(
                value,
                args.linear_num_key_heads,
                &[ba_width, ba_width],
                stream,
            )?;
            return evaluate_fused_projection_outputs(vec![
                (
                    format!("{prefix}linear_attn.in_proj_b.{suffix}"),
                    parts[0].clone(),
                ),
                (
                    format!("{prefix}linear_attn.in_proj_a.{suffix}"),
                    parts[1].clone(),
                ),
            ]);
        }
    }
    Ok(vec![(key.to_string(), value)])
}

#[cfg(test)]
fn validate_affine_fused_component(
    key: &str,
    value: &Array,
    component: &str,
    affine: AffineQuantization,
    input_dims: i32,
) -> Result<(), Error> {
    if input_dims <= 0 || input_dims % affine.group_size != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3-Next affine fused projection {key:?} has input dimension {input_dims}, which is not divisible by group size {}",
            affine.group_size
        )));
    }
    let (expected_trailing, expected_dtype) = match component {
        "weight" => (
            quantized_packed_dimension(input_dims, affine.bits),
            Dtype::Uint32,
        ),
        "scales" | "biases" => (input_dims / affine.group_size, Dtype::Float16),
        other => {
            return Err(Error::UnsupportedArchitecture(format!(
                "unsupported Qwen3-Next affine fused projection component {other:?}"
            )))
        }
    };
    if value.ndim() != 2 || value.dim(1) != expected_trailing || value.dtype() != expected_dtype {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen3-Next affine fused projection {key:?} has shape {:?} and dtype {:?}; expected rank-2 trailing dimension {expected_trailing} and dtype {expected_dtype:?} for {}-bit groups of {}",
            value.shape(),
            value.dtype(),
            affine.bits,
            affine.group_size
        )));
    }
    Ok(())
}

pub(crate) fn split_fused_projection_configs<T: Copy>(
    configs: &mut std::collections::HashMap<String, T>,
) -> Result<(), Error> {
    let fused = configs
        .keys()
        .filter_map(|key| {
            [
                (
                    "linear_attn.in_proj_qkvz.weight",
                    [
                        "linear_attn.in_proj_qkv.weight",
                        "linear_attn.in_proj_z.weight",
                    ],
                ),
                (
                    "linear_attn.in_proj_ba.weight",
                    [
                        "linear_attn.in_proj_b.weight",
                        "linear_attn.in_proj_a.weight",
                    ],
                ),
            ]
            .into_iter()
            .find_map(|(suffix, outputs)| {
                key.strip_suffix(suffix)
                    .map(|prefix| (key.clone(), prefix.to_string(), outputs))
            })
        })
        .collect::<Vec<_>>();
    for (source, prefix, outputs) in fused {
        let config = configs
            .remove(&source)
            .expect("fused affine config key was collected from this map");
        for output in outputs {
            let output = format!("{prefix}{output}");
            if configs.insert(output.clone(), config).is_some() {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Qwen3-Next affine projection {source:?} collides with {output:?}"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn evaluate_fused_projection_outputs(
    outputs: Vec<(String, Array)>,
) -> Result<Vec<(String, Array)>, Error> {
    // Detach every split from its fused checkpoint source before the loader
    // advances. Otherwise all source arrays remain reachable through lazy MLX
    // graphs until the final model-wide evaluation.
    eval(outputs.iter().map(|(_, value)| value))?;
    Ok(outputs)
}

pub(crate) fn fused_projection_widths(args: &ModelArgs) -> Result<([i32; 4], i32), Error> {
    if args.linear_num_key_heads <= 0
        || args.linear_num_value_heads <= 0
        || args.linear_value_head_dim <= 0
        || args.linear_num_value_heads % args.linear_num_key_heads != 0
    {
        return Err(Error::UnsupportedArchitecture(
            "invalid grouped Qwen3-Next fused projection dimensions".into(),
        ));
    }
    let value_dim = args
        .linear_num_value_heads
        .checked_mul(args.linear_value_head_dim)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("Qwen3-Next fused projection dimension overflow".into())
        })?;
    if value_dim % args.linear_num_key_heads != 0 {
        return Err(Error::UnsupportedArchitecture(
            "invalid grouped Qwen3-Next fused projection dimensions".into(),
        ));
    }
    let value_per_key = value_dim / args.linear_num_key_heads;
    Ok((
        [
            args.linear_key_head_dim,
            args.linear_key_head_dim,
            value_per_key,
            value_per_key,
        ],
        args.linear_num_value_heads / args.linear_num_key_heads,
    ))
}

/// Converts grouped FP8 component widths from tensor rows to 128-row scale
/// blocks. Every component boundary must be exactly block-aligned so a fused
/// checkpoint scale tensor can be split without changing quantization groups.
pub(crate) fn fp8_block_row_widths(widths: &[i32]) -> Result<Vec<i32>, Error> {
    widths
        .iter()
        .map(|width| {
            if *width <= 0 || *width % 128 != 0 {
                return Err(Error::UnsupportedArchitecture(format!(
                    "Qwen3-Next FP8 fused projection component width {width} is not divisible by 128"
                )));
            }
            Ok(*width / 128)
        })
        .collect()
}

#[cfg(test)]
fn split_grouped_rows(
    value: Array,
    groups: i32,
    widths: &[i32],
    stream: &Stream,
) -> Result<Vec<Array>, Error> {
    if value.ndim() != 2 || groups <= 0 || widths.iter().any(|width| *width <= 0) {
        return Err(Error::UnsupportedArchitecture(format!(
            "invalid fused Qwen3-Next projection shape {:?}",
            value.shape()
        )));
    }
    let group_width = widths.iter().sum::<i32>();
    if value.dim(0) != groups * group_width {
        return Err(Error::UnsupportedArchitecture(format!(
            "fused Qwen3-Next projection has shape {:?}; expected {} output rows",
            value.shape(),
            groups * group_width
        )));
    }
    let trailing = value.dim(1);
    let grouped = value.reshape(&[groups, group_width, trailing], stream)?;
    let mut start = 0;
    widths
        .iter()
        .map(|width| {
            let part = grouped
                .try_index_device((.., start..start + *width, ..), stream)?
                .reshape(&[-1, trailing], stream)?;
            start += *width;
            Ok(part)
        })
        .collect::<Result<Vec<_>, safemlx::error::Exception>>()
        .map_err(Into::into)
}

pub(crate) fn validate_model_config_value(config: &serde_json::Value) -> Result<(), Error> {
    model_args_from_config_value(config).map(|_| ())
}
