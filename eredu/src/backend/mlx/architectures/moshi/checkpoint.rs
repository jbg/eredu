//! Architecture-owned checkpoint contract for native Moshi MLX weights.

//!
//! Moshi owns the physical tensor names, temporal/depth geometry, and
//! quantization companion declarations here. Generic checkpoint code only
//! evaluates the resulting declarative plan.

use eredu_checkpoint::validation::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation};
use eredu_checkpoint::{StoredDtype, WeightQuantization};

use std::path::{Path, PathBuf};

use super::model::ModelArgs;
use crate::{
    backend::mlx::error::Error,
    backend::mlx::runtime::checkpoint::store::{SafetensorsWeightStore, WeightStore},
};
use eredu_checkpoint::schema::{
    CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint,
};
use eredu_checkpoint::validation;

pub(crate) fn validate_safetensors_path(
    path: impl AsRef<Path>,
    args: &ModelArgs,
) -> Result<(), Error> {
    let store =
        SafetensorsWeightStore::open(path).map_err(|error| Error::Other(Box::new(error)))?;
    validate_safetensors(args, &store)
        .into_loader_result()
        .map_err(Error::from)
}

pub(crate) fn source_path(model_dir: &Path, args: &ModelArgs) -> PathBuf {
    let weights_name = args.moshi_name.as_deref().unwrap_or("model.safetensors");
    if weights_name == "model.safetensors"
        && model_dir.join("model.safetensors.index.json").exists()
    {
        model_dir.to_path_buf()
    } else {
        model_dir.join(weights_name)
    }
}

pub(crate) fn validate_safetensors(
    args: &ModelArgs,
    store: &SafetensorsWeightStore,
) -> CheckpointValidation {
    if let Err(error) = args.validate() {
        return invalid_geometry(error.to_string());
    }
    if args.num_layers as usize > store.keys().len()
        || args.depformer_num_layers as usize > store.keys().len()
    {
        return invalid_geometry(format!(
            "configured Moshi temporal/depth layer counts {}/{} exceed the entire {}-tensor checkpoint catalog",
            args.num_layers,
            args.depformer_num_layers,
            store.keys().len()
        ));
    }
    let plan = match safetensors_plan(args) {
        Ok(plan) => plan,
        Err(PlanError::Geometry(detail)) => return invalid_geometry(detail),
        Err(PlanError::Companion { name, detail }) => {
            return CheckpointValidation::Invalid(vec![CheckpointIssue {
                kind: CheckpointIssueKind::CompanionMismatch,
                detail,
                tensor_name: Some(name),
                tensor_type_code: None,
                metadata_key: Some("quantization".into()),
            }]);
        }
    };
    validation::validate_safetensors_plan(store, &plan)
}

pub(crate) fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, PlanError> {
    args.validate()
        .map_err(|error| PlanError::Geometry(error.to_string()))?;
    if matches!(
        args.quantization,
        Some(WeightQuantization::GgufIQuant { .. })
    ) {
        return Err(PlanError::Geometry(
            "native Moshi SafeTensors cannot use checkpoint-native GGUF quantization".into(),
        ));
    }

    let quantization = args.quantization;
    let dim = dimension(args.dim, "temporal hidden size")?;
    let depth_dim = dimension(args.depformer_dim, "depth hidden size")?;
    let text_card = dimension(args.text_card, "text vocabulary size")?;
    let audio_card = dimension(args.card, "audio vocabulary size")?;
    let temporal_hidden = mlp_hidden(
        dim,
        optional_dimension(args.dim_feedforward, "temporal feed-forward size")?,
    )?;
    let depth_hidden = mlp_hidden(
        depth_dim,
        optional_dimension(args.depformer_dim_feedforward, "depth feed-forward size")?,
    )?;
    let temporal_layers = dimension(args.num_layers, "temporal layer count")?;
    let depth_layers = dimension(args.depformer_num_layers, "depth layer count")?;
    let audio_codebooks = dimension(args.n_q, "audio embedding count")?;
    let depth_slices = dimension(args.dep_q, "depth codebook count")?;
    let text_input = checked_add(text_card, 1, "text input vocabulary")?;
    let audio_input = checked_add(audio_card, 1, "audio input vocabulary")?;
    let mut tensors = Vec::new();

    add_matrix(
        &mut tensors,
        "text_emb.weight",
        vec![text_input, dim],
        quantization,
    )?;
    add_matrix(
        &mut tensors,
        "text_linear.weight",
        vec![text_card, dim],
        quantization,
    )?;
    tensors.push(norm("out_norm.weight", dim));
    for codebook in 0..audio_codebooks {
        add_matrix(
            &mut tensors,
            &format!("audio_embs.{codebook}.weight"),
            vec![audio_input, dim],
            quantization,
        )?;
    }

    for layer in 0..temporal_layers {
        let prefix = format!("transformer.layers.{layer}");
        tensors.push(norm(format!("{prefix}.norm1.weight"), dim));
        tensors.push(norm(format!("{prefix}.norm2.weight"), dim));
        for (local, shape) in [
            (
                "self_attn.in_proj.weight",
                vec![checked_mul(3, dim, "temporal attention input width")?, dim],
            ),
            ("self_attn.out_proj.weight", vec![dim, dim]),
            (
                "gating.linear_in.weight",
                vec![
                    checked_mul(2, temporal_hidden, "temporal gated input width")?,
                    dim,
                ],
            ),
            ("gating.linear_out.weight", vec![dim, temporal_hidden]),
        ] {
            add_matrix(
                &mut tensors,
                &format!("{prefix}.{local}"),
                shape,
                quantization,
            )?;
        }
    }

    for slice in 0..depth_slices {
        let prefix = format!("depformer.slices.{slice}");
        add_matrix(
            &mut tensors,
            &format!("{prefix}.emb.weight"),
            vec![if slice == 0 { text_input } else { audio_input }, depth_dim],
            quantization,
        )?;
        add_matrix(
            &mut tensors,
            &format!("{prefix}.linear_in.weight"),
            vec![depth_dim, dim],
            quantization,
        )?;
        add_matrix(
            &mut tensors,
            &format!("{prefix}.linear_out.weight"),
            vec![audio_card, depth_dim],
            quantization,
        )?;

        for layer in 0..depth_layers {
            let layer_prefix = format!("{prefix}.transformer.layers.{layer}");
            tensors.push(norm(format!("{layer_prefix}.norm1.weight"), depth_dim));
            tensors.push(norm(format!("{layer_prefix}.norm2.weight"), depth_dim));
            for (local, shape) in [
                (
                    "self_attn.in_proj.weight",
                    vec![
                        checked_mul(3, depth_dim, "depth attention input width")?,
                        depth_dim,
                    ],
                ),
                ("self_attn.out_proj.weight", vec![depth_dim, depth_dim]),
                (
                    "gating.linear_in.weight",
                    vec![
                        checked_mul(2, depth_hidden, "depth gated input width")?,
                        depth_dim,
                    ],
                ),
                ("gating.linear_out.weight", vec![depth_dim, depth_hidden]),
            ] {
                add_matrix(
                    &mut tensors,
                    &format!("{layer_prefix}.{local}"),
                    shape,
                    quantization,
                )?;
            }
        }
    }

    SafetensorsCheckpointPlan::new(
        "Moshi native MLX SafeTensors",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| PlanError::Geometry(error.to_string()))
}

fn add_matrix(
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
) -> Result<(), PlanError> {
    let Some(quantization) = quantization else {
        output.push(SafetensorsTensorConstraint::required(
            name,
            shape,
            StoredDtypeConstraint::Floating,
        ));
        return Ok(());
    };

    let input = *shape
        .last()
        .ok_or_else(|| PlanError::Geometry(format!("Moshi matrix {name:?} is scalar")))?;
    let group = dimension(quantization.group_size(), "quantization group size")?;
    let bits = dimension(quantization.bits(), "quantization bit width")?;
    let packed_bits = checked_mul(input, bits, "quantized matrix packing")?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(PlanError::Companion {
            name: name.into(),
            detail: format!(
                "quantized Moshi tensor {name:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
            ),
        });
    }

    let mut packed = shape.clone();
    *packed.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion = shape;
    *companion.last_mut().expect("matrix shape") = input / group;
    output.push(SafetensorsTensorConstraint::required(
        name,
        packed,
        StoredDtypeConstraint::Exact(StoredDtype::U32),
    ));
    let prefix = name.trim_end_matches(".weight");
    let companion_dtype = match quantization {
        WeightQuantization::Affine(_) => StoredDtypeConstraint::Floating,
        WeightQuantization::MxFp4 => StoredDtypeConstraint::Exact(StoredDtype::U8),
        WeightQuantization::GgufIQuant { .. } => unreachable!("rejected before plan expansion"),
    };
    output.push(
        SafetensorsTensorConstraint::required(
            format!("{prefix}.scales"),
            companion.clone(),
            companion_dtype.clone(),
        )
        .companion(),
    );
    if quantization.has_biases() {
        output.push(
            SafetensorsTensorConstraint::required(
                format!("{prefix}.biases"),
                companion,
                StoredDtypeConstraint::Floating,
            )
            .companion(),
        );
    }
    Ok(())
}

fn norm(key: impl Into<String>, elements: usize) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, vec![elements], StoredDtypeConstraint::Floating)
}

fn mlp_hidden(dim: usize, feed_forward: Option<usize>) -> Result<usize, PlanError> {
    let four_dim = checked_mul(4, dim, "default feed-forward width")?;
    let feed_forward = feed_forward.unwrap_or(four_dim);
    if feed_forward == four_dim {
        checked_mul(11, dim, "gated feed-forward numerator").map(|value| value / 4)
    } else {
        checked_mul(2, feed_forward, "gated feed-forward numerator").map(|value| value / 3)
    }
}

fn optional_dimension(value: Option<i32>, name: &str) -> Result<Option<usize>, PlanError> {
    value.map(|value| dimension(value, name)).transpose()
}

fn dimension(value: i32, name: &str) -> Result<usize, PlanError> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| PlanError::Geometry(format!("Moshi {name} must be positive, got {value}")))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, PlanError> {
    left.checked_add(right)
        .ok_or_else(|| PlanError::Geometry(format!("Moshi {name} geometry overflows")))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, PlanError> {
    left.checked_mul(right)
        .ok_or_else(|| PlanError::Geometry(format!("Moshi {name} geometry overflows")))
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum PlanError {
    Geometry(String),
    Companion { name: String, detail: String },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Geometry(detail) | Self::Companion { detail, .. } => formatter.write_str(detail),
        }
    }
}

fn invalid_geometry(detail: String) -> CheckpointValidation {
    CheckpointValidation::Invalid(vec![CheckpointIssue {
        kind: CheckpointIssueKind::InvalidGeometry,
        detail,
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::AffineQuantization;

    fn args() -> ModelArgs {
        serde_json::from_value(serde_json::json!({
            "dim": 32, "text_card": 32, "n_q": 4, "dep_q": 2, "card": 8,
            "num_heads": 4, "num_layers": 2, "dim_feedforward": 96,
            "causal": true, "context": 16, "max_period": 10000,
            "positional_embedding": "rope", "depformer_dim": 32,
            "depformer_dim_feedforward": 96, "depformer_num_heads": 4,
            "depformer_num_layers": 2, "depformer_context": 2,
            "depformer_pos_emb": "none", "delays": [0, 0, 1, 0, 1]
        }))
        .unwrap()
    }

    #[test]
    fn native_plan_owns_per_slice_depth_transformers() {
        let plan = safetensors_plan(&args()).unwrap();
        let names = plan
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(names.contains("audio_embs.3.weight"));
        assert!(names.contains("depformer.slices.1.transformer.layers.1.self_attn.in_proj.weight"));
        assert!(!names.contains("depformer.layers.1.self_attn.in_proj_weight"));
    }

    #[test]
    fn released_moshika_default_constructs_the_native_contract() {
        let plan = safetensors_plan(&ModelArgs::v0_1()).unwrap();
        assert_eq!(plan.identity, "Moshi native MLX SafeTensors");
        assert!(plan.common_tensors.iter().any(|tensor| {
            tensor.key == "depformer.slices.7.transformer.layers.5.gating.linear_out.weight"
        }));
    }

    #[test]
    fn affine_plan_declares_packed_weights_and_floating_companions() {
        let mut args = args();
        args.quantization = Some(WeightQuantization::Affine(
            AffineQuantization::new(32, 4).unwrap(),
        ));
        let plan = safetensors_plan(&args).unwrap();
        let weight = plan
            .common_tensors
            .iter()
            .find(|tensor| tensor.key == "text_emb.weight")
            .unwrap();
        assert_eq!(weight.shape, [33, 4]);
        assert_eq!(weight.dtype, StoredDtypeConstraint::Exact(StoredDtype::U32));
        let scales = plan
            .common_tensors
            .iter()
            .find(|tensor| tensor.key == "text_emb.scales")
            .unwrap();
        assert_eq!(scales.shape, [33, 1]);
        assert_eq!(scales.dtype, StoredDtypeConstraint::Floating);
    }

    #[test]
    fn mxfp4_plan_requires_u8_scales_without_biases() {
        let mut args = args();
        args.quantization = Some(WeightQuantization::MxFp4);
        let plan = safetensors_plan(&args).unwrap();
        let scales = plan
            .common_tensors
            .iter()
            .find(|tensor| tensor.key == "text_emb.scales")
            .unwrap();
        assert_eq!(scales.dtype, StoredDtypeConstraint::Exact(StoredDtype::U8));
        assert!(!plan
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "text_emb.biases"));
    }
}
