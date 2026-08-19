//! Architecture-owned checkpoint contract for PersonaPlex.

//!
//! PersonaPlex owns the released PyTorch SafeTensors catalog, temporal and
//! depth geometry, physical aliases, flattened norm layouts, and affine
//! companion declarations here. Generic checkpoint code evaluates those
//! declarations without knowing about realtime audio or codebook models.

use eredu_checkpoint::{StoredDtype, WeightQuantization};

use serde_json::Value;

use super::{model::ModelArgs, personaplex};
use crate::backend::mlx::runtime::checkpoint::store::{SafetensorsWeightStore, WeightStore};
use eredu_checkpoint::schema::{
    CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint,
};
use eredu_checkpoint::validation;
use eredu_checkpoint::validation::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation};

pub(crate) fn validate_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> CheckpointValidation {
    let metadata = match personaplex::model_metadata_from_config_value(config) {
        Ok(metadata) => metadata,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let mut args = personaplex::model_args_7b_v1();
    args.quantization = metadata.quantization;
    if args.num_layers as usize > store.keys().len()
        || args.depformer_num_layers as usize > store.keys().len()
    {
        return invalid_geometry(format!(
            "configured PersonaPlex temporal/depth layer counts {}/{} exceed the entire {}-tensor checkpoint catalog",
            args.num_layers,
            args.depformer_num_layers,
            store.keys().len()
        ));
    }
    let plan = match safetensors_plan(&args) {
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
        ["text_emb.weight"],
        "text_emb",
        vec![text_input, dim],
        quantization,
    )?;
    add_matrix(
        &mut tensors,
        ["text_linear.weight"],
        "text_linear",
        vec![text_card, dim],
        quantization,
    )?;
    tensors.push(norm("out_norm.alpha", dim));
    for codebook in 0..audio_codebooks {
        let name = format!("emb.{codebook}.weight");
        add_matrix(
            &mut tensors,
            [name.as_str()],
            name.trim_end_matches(".weight"),
            vec![audio_input, dim],
            quantization,
        )?;
    }

    for layer in 0..temporal_layers {
        let prefix = format!("transformer.layers.{layer}");
        for norm_name in ["norm1", "norm2"] {
            tensors.push(norm(format!("{prefix}.{norm_name}.alpha"), dim));
        }
        add_matrix(
            &mut tensors,
            [
                format!("{prefix}.self_attn.in_proj_weight"),
                format!("{prefix}.self_attn.in_proj.weight"),
            ],
            &format!("{prefix}.self_attn.in_proj"),
            vec![checked_mul(3, dim, "temporal attention input width")?, dim],
            quantization,
        )?;
        for (local, shape) in [
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
            let name = format!("{prefix}.{local}");
            add_matrix(
                &mut tensors,
                [name.as_str()],
                name.trim_end_matches(".weight"),
                shape,
                quantization,
            )?;
        }
    }

    for slice in 0..depth_slices {
        let embedding = if slice == 0 {
            "depformer_text_emb.weight".to_string()
        } else {
            format!("depformer_emb.{}.weight", slice - 1)
        };
        add_matrix(
            &mut tensors,
            [embedding.as_str()],
            embedding.trim_end_matches(".weight"),
            vec![if slice == 0 { text_input } else { audio_input }, depth_dim],
            quantization,
        )?;
        for (name, shape) in [
            (format!("depformer_in.{slice}.weight"), vec![depth_dim, dim]),
            (
                format!("linears.{slice}.weight"),
                vec![audio_card, depth_dim],
            ),
        ] {
            add_matrix(
                &mut tensors,
                [name.as_str()],
                name.trim_end_matches(".weight"),
                shape,
                quantization,
            )?;
        }
    }

    for layer in 0..depth_layers {
        let prefix = format!("depformer.layers.{layer}");
        for norm_name in ["norm1", "norm2"] {
            tensors.push(norm(format!("{prefix}.{norm_name}.alpha"), depth_dim));
        }
        add_matrix(
            &mut tensors,
            [
                format!("{prefix}.self_attn.in_proj_weight"),
                format!("{prefix}.self_attn.in_proj.weight"),
            ],
            &format!("{prefix}.self_attn.in_proj"),
            vec![
                checked_mul(
                    checked_mul(depth_slices, 3, "depth attention QKV count")?,
                    depth_dim,
                    "depth attention input width",
                )?,
                depth_dim,
            ],
            quantization,
        )?;
        let out_proj = format!("{prefix}.self_attn.out_proj.weight");
        add_matrix(
            &mut tensors,
            [out_proj.as_str()],
            out_proj.trim_end_matches(".weight"),
            vec![
                checked_mul(depth_slices, depth_dim, "depth attention output width")?,
                depth_dim,
            ],
            quantization,
        )?;
        for slice in 0..depth_slices {
            for (name, shape) in [
                (
                    format!("{prefix}.gating.{slice}.linear_in.weight"),
                    vec![
                        checked_mul(2, depth_hidden, "depth gated input width")?,
                        depth_dim,
                    ],
                ),
                (
                    format!("{prefix}.gating.{slice}.linear_out.weight"),
                    vec![depth_dim, depth_hidden],
                ),
            ] {
                add_matrix(
                    &mut tensors,
                    [name.as_str()],
                    name.trim_end_matches(".weight"),
                    shape,
                    quantization,
                )?;
            }
        }
    }

    SafetensorsCheckpointPlan::new(
        "PersonaPlex released PyTorch SafeTensors",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| PlanError::Geometry(error.to_string()))
}

fn add_matrix<I, S>(
    output: &mut Vec<SafetensorsTensorConstraint>,
    aliases: I,
    companion_prefix: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
) -> Result<(), PlanError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut aliases = aliases.into_iter().map(Into::into).collect::<Vec<_>>();
    let primary = aliases
        .first()
        .cloned()
        .ok_or_else(|| PlanError::Geometry("PersonaPlex matrix has no physical name".into()))?;
    let physical_aliases = aliases.drain(1..).collect::<Vec<_>>();
    let Some(quantization) = quantization else {
        output.push(
            SafetensorsTensorConstraint::required(primary, shape, StoredDtypeConstraint::Floating)
                .with_aliases(physical_aliases),
        );
        return Ok(());
    };

    let input = *shape
        .last()
        .ok_or_else(|| PlanError::Geometry(format!("PersonaPlex matrix {primary:?} is scalar")))?;
    let group = dimension(quantization.group_size(), "quantization group size")?;
    let bits = dimension(quantization.bits(), "quantization bit width")?;
    let packed_bits = checked_mul(input, bits, "quantized matrix packing")?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(PlanError::Companion {
            name: primary.clone(),
            detail: format!(
                "quantized PersonaPlex tensor {primary:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
            ),
        });
    }
    let mut packed = shape.clone();
    *packed.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion = shape;
    *companion.last_mut().expect("matrix shape") = input / group;
    output.push(
        SafetensorsTensorConstraint::required(
            primary,
            packed,
            StoredDtypeConstraint::Exact(StoredDtype::U32),
        )
        .with_aliases(physical_aliases),
    );
    let companion_dtype = || {
        StoredDtypeConstraint::OneOf(vec![
            StoredDtype::F16,
            StoredDtype::BF16,
            StoredDtype::F32,
            StoredDtype::U8,
        ])
    };
    output.push(
        SafetensorsTensorConstraint::required(
            format!("{companion_prefix}.scales"),
            companion.clone(),
            companion_dtype(),
        )
        .companion(),
    );
    if quantization.has_biases() {
        output.push(
            SafetensorsTensorConstraint::required(
                format!("{companion_prefix}.biases"),
                companion,
                companion_dtype(),
            )
            .companion(),
        );
    }
    Ok(())
}

fn norm(key: impl Into<String>, elements: usize) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, vec![elements], StoredDtypeConstraint::Floating)
        .with_element_count(elements)
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
        .ok_or_else(|| {
            PlanError::Geometry(format!("PersonaPlex {name} must be positive, got {value}"))
        })
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, PlanError> {
    left.checked_add(right)
        .ok_or_else(|| PlanError::Geometry(format!("PersonaPlex {name} geometry overflows")))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, PlanError> {
    left.checked_mul(right)
        .ok_or_else(|| PlanError::Geometry(format!("PersonaPlex {name} geometry overflows")))
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

    #[test]
    fn released_plan_owns_attention_aliases_and_flattened_norms() {
        let args = personaplex::model_args_7b_v1();
        let plan = safetensors_plan(&args).unwrap();
        let attention = plan
            .common_tensors
            .iter()
            .find(|tensor| tensor.key == "transformer.layers.0.self_attn.in_proj_weight")
            .unwrap();
        assert_eq!(
            attention.aliases,
            ["transformer.layers.0.self_attn.in_proj.weight"]
        );
        let norm = plan
            .common_tensors
            .iter()
            .find(|tensor| tensor.key == "out_norm.alpha")
            .unwrap();
        assert_eq!(norm.element_count, Some(args.dim as usize));
    }

    #[test]
    fn affine_plan_declares_packed_weights_and_ordinary_companions() {
        let mut args = personaplex::model_args_7b_v1();
        args.quantization = Some(WeightQuantization::Affine(
            AffineQuantization::new(32, 4).unwrap(),
        ));
        let plan = safetensors_plan(&args).unwrap();
        let weight = plan
            .common_tensors
            .iter()
            .find(|tensor| tensor.key == "text_emb.weight")
            .unwrap();
        assert_eq!(weight.shape, [32_001, 512]);
        assert_eq!(weight.dtype, StoredDtypeConstraint::Exact(StoredDtype::U32));
        let scales = plan
            .common_tensors
            .iter()
            .find(|tensor| tensor.key == "text_emb.scales")
            .unwrap();
        assert_eq!(scales.shape, [32_001, 128]);
        assert_eq!(
            scales.dtype,
            StoredDtypeConstraint::OneOf(vec![
                StoredDtype::BF16,
                StoredDtype::F16,
                StoredDtype::F32,
                StoredDtype::U8,
            ])
        );
    }
}
