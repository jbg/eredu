//! Pure checkpoint-structure plans shared by inspection and high-level loading.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::{GgufArchitecture, ModelKind, ModelLoadOptions};
use crate::{
    architectures::{
        deepseek_v3::checkpoint as deepseek_v3_checkpoint,
        deepseek_v4::checkpoint as deepseek_v4_checkpoint,
        gemma4::{checkpoint as gemma4_checkpoint, model as gemma4},
        gpt_oss::checkpoint as gpt_oss_checkpoint,
        inkling::{checkpoint as inkling_checkpoint, model as inkling},
        kimi_linear::checkpoint as kimi_linear_checkpoint,
        lfm2::checkpoint as lfm2_checkpoint,
        llama::checkpoint as llama_checkpoint,
        moshi::personaplex,
        muse_glimmer::checkpoint as muse_glimmer_checkpoint,
        nemotron_h::checkpoint as nemotron_h_checkpoint,
        qwen::{
            dense::checkpoint as dense_qwen_checkpoint,
            hybrid::checkpoint as qwen_hybrid_checkpoint, vl::checkpoint as qwen_vl_checkpoint,
        },
    },
    error::Error,
    runtime::checkpoint::{
        schema::{
            CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
            StoredDtypeConstraint,
        },
        store::{SafetensorsWeightStore, StoredDtype, WeightStore},
        validation as checkpoint_validation,
    },
};

pub(crate) use crate::runtime::checkpoint::contract::{
    CheckpointIssue as StructuralIssue, CheckpointIssueKind as StructuralIssueKind,
    CheckpointValidation as StructuralValidation,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)] // Reserved for fail-closed structural policies.
pub(crate) enum StructuralValidationPolicy {
    Exact,
    Unverified,
}

/// Exhaustive policy table for high-level SafeTensors loader families.
pub(crate) const fn safetensors_policy(kind: ModelKind) -> StructuralValidationPolicy {
    match kind {
        ModelKind::DeepSeekV3
        | ModelKind::DeepSeekV4
        | ModelKind::Gemma4
        | ModelKind::GptOss
        | ModelKind::Inkling
        | ModelKind::KimiLinear
        | ModelKind::Lfm2
        | ModelKind::Llama
        | ModelKind::MuseGlimmer
        | ModelKind::NemotronH
        | ModelKind::PersonaPlex
        | ModelKind::Qwen2
        | ModelKind::Qwen3
        | ModelKind::Qwen3Next
        | ModelKind::Qwen3Vl
        | ModelKind::Qwen3VlMoe
        | ModelKind::Qwen35 => StructuralValidationPolicy::Exact,
    }
}

/// Exhaustive policy table for concrete GGUF loader architectures.
pub(crate) const fn gguf_policy(architecture: GgufArchitecture) -> StructuralValidationPolicy {
    match architecture {
        GgufArchitecture::Llama
        | GgufArchitecture::Mistral
        | GgufArchitecture::MuseGlimmer
        | GgufArchitecture::DeepSeek2
        | GgufArchitecture::DeepSeek4
        | GgufArchitecture::Lfm2
        | GgufArchitecture::Lfm2Moe
        | GgufArchitecture::GptOss
        | GgufArchitecture::Gemma4
        | GgufArchitecture::Inkling
        | GgufArchitecture::Qwen2
        | GgufArchitecture::Qwen3
        | GgufArchitecture::Qwen3Moe
        | GgufArchitecture::NemotronH
        | GgufArchitecture::NemotronHMoe
        | GgufArchitecture::Qwen35
        | GgufArchitecture::Qwen35Moe
        | GgufArchitecture::Qwen3Next
        | GgufArchitecture::Qwen3Vl
        | GgufArchitecture::Qwen3VlMoe
        | GgufArchitecture::KimiLinear => StructuralValidationPolicy::Exact,
    }
}

pub(crate) fn validate_safetensors(
    kind: ModelKind,
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let validation = match safetensors_policy(kind) {
        StructuralValidationPolicy::Exact => match kind {
            ModelKind::DeepSeekV3 => validate_deepseek_v3_safetensors(config, store, options),
            ModelKind::DeepSeekV4 => validate_deepseek_v4_safetensors(config, store),
            ModelKind::Gemma4 => validate_gemma4_safetensors(config, store, options),
            ModelKind::GptOss => validate_gpt_oss_safetensors(config, store),
            ModelKind::Inkling => validate_inkling_safetensors(config, store),
            ModelKind::KimiLinear => kimi_linear_checkpoint::validate_safetensors(config, store),
            ModelKind::Lfm2 => lfm2_checkpoint::validate_safetensors(
                config,
                store,
                !options.weight_residency.is_fully_resident(),
            ),
            ModelKind::Llama => llama_checkpoint::validate_safetensors(config, store),
            ModelKind::MuseGlimmer => muse_glimmer_checkpoint::validate_safetensors(config, store),
            ModelKind::NemotronH => nemotron_h_checkpoint::validate_safetensors(config, store),
            ModelKind::PersonaPlex => validate_personaplex_safetensors(config, store),
            ModelKind::Qwen2 => validate_dense_qwen_safetensors(config, store),
            ModelKind::Qwen3 => validate_dense_qwen_safetensors(config, store),
            ModelKind::Qwen3Next => validate_qwen3_next_safetensors(config, store, options),
            ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => {
                validate_qwen3_vl_safetensors(kind, config, store, options)
            }
            ModelKind::Qwen35 => validate_qwen35_safetensors(config, store, options),
        },
        StructuralValidationPolicy::Unverified => unverified(kind.model_type_name()),
    };
    validation.with_strict_catalog(options.weight_residency.strict_loading())
}

pub(crate) fn validate_safetensors_load_path(
    kind: ModelKind,
    model_dir: &Path,
    options: ModelLoadOptions,
) -> Result<(), Error> {
    let config: Value = serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
    let store =
        SafetensorsWeightStore::open(model_dir).map_err(|error| Error::Other(Box::new(error)))?;
    validate_safetensors(kind, &config, &store, options).into_loader_result()
}

pub(crate) fn validate_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let validation = match gguf_policy(architecture) {
        StructuralValidationPolicy::Exact => match architecture {
            GgufArchitecture::DeepSeek2 => validate_deepseek2_gguf(checkpoint, metadata),
            GgufArchitecture::DeepSeek4 => validate_deepseek4_gguf(checkpoint, metadata),
            GgufArchitecture::GptOss => validate_gpt_oss_gguf(checkpoint, metadata),
            GgufArchitecture::Gemma4 => validate_gemma4_gguf(checkpoint, metadata, options),
            GgufArchitecture::Inkling => validate_inkling_gguf(checkpoint, metadata, options),
            GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
                let variant = if architecture == GgufArchitecture::Lfm2Moe {
                    lfm2_checkpoint::GgufVariant::Moe
                } else {
                    lfm2_checkpoint::GgufVariant::Dense
                };
                lfm2_checkpoint::validate_gguf(variant, checkpoint, metadata)
            }
            GgufArchitecture::Llama | GgufArchitecture::Mistral => {
                llama_checkpoint::validate_gguf(checkpoint, metadata)
            }
            GgufArchitecture::MuseGlimmer => {
                muse_glimmer_checkpoint::validate_gguf(checkpoint, metadata)
            }
            GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
                if let Err(error) = architecture.validate_load_policy(options) {
                    invalid_geometry(error.to_string())
                } else {
                    let variant = if architecture == GgufArchitecture::NemotronHMoe {
                        nemotron_h_checkpoint::GgufVariant::Moe
                    } else {
                        nemotron_h_checkpoint::GgufVariant::Dense
                    };
                    nemotron_h_checkpoint::validate_gguf(variant, checkpoint, metadata)
                }
            }
            GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
                validate_dense_qwen_gguf(architecture, checkpoint, metadata)
            }
            architecture @ (GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe) => {
                validate_qwen3_vl_gguf(architecture, checkpoint, metadata, options)
            }
            GgufArchitecture::KimiLinear => {
                kimi_linear_checkpoint::validate_gguf(checkpoint, metadata)
            }
            GgufArchitecture::Qwen35
            | GgufArchitecture::Qwen35Moe
            | GgufArchitecture::Qwen3Next => {
                validate_qwen35_gguf(architecture, checkpoint, metadata, options)
            }
        },
        StructuralValidationPolicy::Unverified => unverified(architecture.metadata_name()),
    };
    validation.with_strict_catalog(options.weight_residency.strict_loading())
}

fn unverified(architecture: &str) -> StructuralValidation {
    StructuralValidation::Unverified(StructuralIssue {
        kind: StructuralIssueKind::ValidationUnavailable,
        detail: format!(
            "exact header-only structural validation is not yet implemented for {architecture}"
        ),
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    })
}

fn validate_inkling_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    inkling_checkpoint::validate_safetensors(config, store)
}

fn validate_gemma4_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    gemma4_checkpoint::validate_safetensors(
        config,
        store,
        !options.weight_residency.is_fully_resident(),
        options.weight_residency.expert_cache().is_some(),
    )
}

fn validate_deepseek_v4_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    deepseek_v4_checkpoint::validate_safetensors(config, store)
}

fn validate_deepseek_v3_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    deepseek_v3_checkpoint::validate_safetensors(
        config,
        store,
        !options.weight_residency.is_fully_resident(),
    )
}

fn validate_gpt_oss_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    gpt_oss_checkpoint::validate_safetensors(config, store)
}

fn append_structural_issues(validation: StructuralValidation, issues: &mut Vec<StructuralIssue>) {
    match validation {
        StructuralValidation::Exact => {}
        StructuralValidation::Invalid(found) => issues.extend(found),
        StructuralValidation::Unverified(_) => {
            unreachable!("pure tensor plan is always exact or invalid")
        }
    }
}

fn validate_qwen3_next_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    qwen_hybrid_checkpoint::validate_qwen3_next_safetensors(
        config,
        store,
        options.weight_residency.expert_cache().is_some(),
    )
}

fn validate_qwen35_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    qwen_hybrid_checkpoint::validate_qwen35_safetensors(
        config,
        store,
        options.weight_residency.expert_cache().is_some(),
    )
}

fn validate_dense_qwen_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    dense_qwen_checkpoint::validate_safetensors(config, store)
}

fn validate_personaplex_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
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

    let quantization = args.quantization;
    let dim = args.dim as usize;
    let depth_dim = args.depformer_dim as usize;
    let temporal_hidden = moshi_mlp_hidden(dim, args.dim_feedforward.map(|value| value as usize));
    let depth_hidden = moshi_mlp_hidden(
        depth_dim,
        args.depformer_dim_feedforward.map(|value| value as usize),
    );
    let mut allowed = BTreeSet::new();
    let mut issues = Vec::new();

    for (name, shape) in [
        (
            "text_emb.weight".to_string(),
            vec![args.text_card as usize + 1, dim],
        ),
        (
            "text_linear.weight".to_string(),
            vec![args.text_card as usize, dim],
        ),
    ] {
        validate_personaplex_matrix(
            store,
            std::slice::from_ref(&name),
            name.trim_end_matches(".weight"),
            &shape,
            quantization,
            &mut allowed,
            &mut issues,
        );
    }
    validate_personaplex_norm(store, "out_norm.alpha", dim, &mut allowed, &mut issues);
    for codebook in 0..args.n_q as usize {
        let name = format!("emb.{codebook}.weight");
        validate_personaplex_matrix(
            store,
            std::slice::from_ref(&name),
            name.trim_end_matches(".weight"),
            &[args.card as usize + 1, dim],
            quantization,
            &mut allowed,
            &mut issues,
        );
    }

    for layer in 0..args.num_layers as usize {
        let prefix = format!("transformer.layers.{layer}");
        for norm in ["norm1", "norm2"] {
            validate_personaplex_norm(
                store,
                &format!("{prefix}.{norm}.alpha"),
                dim,
                &mut allowed,
                &mut issues,
            );
        }
        let in_proj = [
            format!("{prefix}.self_attn.in_proj_weight"),
            format!("{prefix}.self_attn.in_proj.weight"),
        ];
        validate_personaplex_matrix(
            store,
            &in_proj,
            &format!("{prefix}.self_attn.in_proj"),
            &[3 * dim, dim],
            quantization,
            &mut allowed,
            &mut issues,
        );
        for (name, shape) in [
            (
                format!("{prefix}.self_attn.out_proj.weight"),
                vec![dim, dim],
            ),
            (
                format!("{prefix}.gating.linear_in.weight"),
                vec![2 * temporal_hidden, dim],
            ),
            (
                format!("{prefix}.gating.linear_out.weight"),
                vec![dim, temporal_hidden],
            ),
        ] {
            validate_personaplex_matrix(
                store,
                std::slice::from_ref(&name),
                name.trim_end_matches(".weight"),
                &shape,
                quantization,
                &mut allowed,
                &mut issues,
            );
        }
    }

    for slice in 0..args.dep_q as usize {
        let embedding = if slice == 0 {
            "depformer_text_emb.weight".to_string()
        } else {
            format!("depformer_emb.{}.weight", slice - 1)
        };
        let input_vocab = if slice == 0 {
            args.text_card as usize + 1
        } else {
            args.card as usize + 1
        };
        validate_personaplex_matrix(
            store,
            std::slice::from_ref(&embedding),
            embedding.trim_end_matches(".weight"),
            &[input_vocab, depth_dim],
            quantization,
            &mut allowed,
            &mut issues,
        );
        for (name, shape) in [
            (format!("depformer_in.{slice}.weight"), vec![depth_dim, dim]),
            (
                format!("linears.{slice}.weight"),
                vec![args.card as usize, depth_dim],
            ),
        ] {
            validate_personaplex_matrix(
                store,
                std::slice::from_ref(&name),
                name.trim_end_matches(".weight"),
                &shape,
                quantization,
                &mut allowed,
                &mut issues,
            );
        }
    }

    let depth_slices = args.dep_q as usize;
    for layer in 0..args.depformer_num_layers as usize {
        let prefix = format!("depformer.layers.{layer}");
        for norm in ["norm1", "norm2"] {
            validate_personaplex_norm(
                store,
                &format!("{prefix}.{norm}.alpha"),
                depth_dim,
                &mut allowed,
                &mut issues,
            );
        }
        let in_proj = [
            format!("{prefix}.self_attn.in_proj_weight"),
            format!("{prefix}.self_attn.in_proj.weight"),
        ];
        validate_personaplex_matrix(
            store,
            &in_proj,
            &format!("{prefix}.self_attn.in_proj"),
            &[depth_slices * 3 * depth_dim, depth_dim],
            quantization,
            &mut allowed,
            &mut issues,
        );
        let out_proj = format!("{prefix}.self_attn.out_proj.weight");
        validate_personaplex_matrix(
            store,
            std::slice::from_ref(&out_proj),
            out_proj.trim_end_matches(".weight"),
            &[depth_slices * depth_dim, depth_dim],
            quantization,
            &mut allowed,
            &mut issues,
        );
        for slice in 0..depth_slices {
            for (name, shape) in [
                (
                    format!("{prefix}.gating.{slice}.linear_in.weight"),
                    vec![2 * depth_hidden, depth_dim],
                ),
                (
                    format!("{prefix}.gating.{slice}.linear_out.weight"),
                    vec![depth_dim, depth_hidden],
                ),
            ] {
                validate_personaplex_matrix(
                    store,
                    std::slice::from_ref(&name),
                    name.trim_end_matches(".weight"),
                    &shape,
                    quantization,
                    &mut allowed,
                    &mut issues,
                );
            }
        }
    }

    for key in store.keys() {
        if !allowed.contains(&key) {
            issues.push(unexpected_layout(
                &key,
                "PersonaPlex released PyTorch SafeTensors",
            ));
        }
    }
    finish(issues)
}

fn moshi_mlp_hidden(dim: usize, feed_forward: Option<usize>) -> usize {
    let feed_forward = feed_forward.unwrap_or(4 * dim);
    if feed_forward == 4 * dim {
        11 * dim / 4
    } else {
        2 * feed_forward / 3
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_personaplex_matrix(
    store: &SafetensorsWeightStore,
    aliases: &[String],
    companion_prefix: &str,
    shape: &[usize],
    quantization: Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
) {
    allowed.extend(aliases.iter().cloned());
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let present = aliases
        .iter()
        .filter(|name| keys.contains(*name))
        .collect::<Vec<_>>();
    if present.len() > 1 {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: format!(
                "PersonaPlex checkpoint contains multiple aliases for {:?}: {present:?}",
                aliases[0]
            ),
            tensor_name: Some(present[1].clone()),
            tensor_type_code: None,
            metadata_key: None,
        });
    }
    let Some(name) = present.first().map(|name| name.as_str()) else {
        issues.push(missing(&aliases[0]));
        return;
    };
    let Some(quantization) = quantization else {
        validate_safetensor(store, name, shape, false, issues);
        return;
    };

    let input = *shape.last().expect("PersonaPlex matrix shape");
    let group_size = quantization.group_size() as usize;
    let bits = quantization.bits() as usize;
    if !input.is_multiple_of(group_size)
        || !input.is_multiple_of(32)
        || !input.saturating_mul(bits).is_multiple_of(32)
    {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::QuantizationCompanionMismatch,
            detail: format!(
                "quantized PersonaPlex tensor {name:?} input dimension {input} is incompatible with group size {group_size} and {bits}-bit packing"
            ),
            tensor_name: Some(name.into()),
            tensor_type_code: None,
            metadata_key: Some("quantization".into()),
        });
        return;
    }
    let mut packed = shape.to_vec();
    *packed.last_mut().expect("PersonaPlex matrix shape") = input * bits / 32;
    validate_safetensor(store, name, &packed, true, issues);
    let mut companions = shape.to_vec();
    *companions.last_mut().expect("PersonaPlex matrix shape") = input / group_size;
    let scales = format!("{companion_prefix}.scales");
    allowed.insert(scales.clone());
    validate_quantization_companion(store, &scales, &companions, issues);
    if quantization.has_biases() {
        let biases = format!("{companion_prefix}.biases");
        allowed.insert(biases.clone());
        validate_quantization_companion(store, &biases, &companions, issues);
    }
}

fn validate_personaplex_norm(
    store: &SafetensorsWeightStore,
    name: &str,
    elements: usize,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
) {
    allowed.insert(name.into());
    let metadata = match store.metadata(name) {
        Ok(metadata) => metadata,
        Err(crate::runtime::checkpoint::store::WeightStoreError::UnknownTensor { .. }) => {
            issues.push(missing(name));
            return;
        }
        Err(error) => {
            issues.push(layout(name, error.to_string()));
            return;
        }
    };
    if metadata.shape.iter().product::<usize>() != elements {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::ShapeMismatch,
            detail: format!(
                "PersonaPlex norm tensor {name:?} must contain {elements} elements for loader reshape, got shape {:?}",
                metadata.shape
            ),
            tensor_name: Some(name.into()),
            tensor_type_code: None,
            metadata_key: None,
        });
    }
    if !is_float_dtype(&metadata.stored_dtype) {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::UnsupportedEncoding,
            detail: format!(
                "tensor {name:?} uses unsupported SafeTensors dtype {:?}",
                metadata.stored_dtype
            ),
            tensor_name: Some(name.into()),
            tensor_type_code: None,
            metadata_key: None,
        });
    }
}

fn validate_qwen3_vl_safetensors(
    kind: ModelKind,
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    qwen_vl_checkpoint::validate_safetensors(
        kind == ModelKind::Qwen3VlMoe,
        config,
        store,
        !options.weight_residency.is_fully_resident(),
    )
}

fn validate_safetensor(
    store: &SafetensorsWeightStore,
    name: &str,
    shape: &[usize],
    packed: bool,
    issues: &mut Vec<StructuralIssue>,
) {
    let dtype = if packed {
        StoredDtypeConstraint::Exact(StoredDtype::U32)
    } else {
        StoredDtypeConstraint::Floating
    };
    let plan = SafetensorsCheckpointPlan::new(
        format!("SafeTensors tensor {name}"),
        vec![SafetensorsTensorConstraint::required(
            name,
            shape.to_vec(),
            dtype,
        )],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .expect("legacy structural tensor constraints are valid");
    append_structural_issues(
        checkpoint_validation::validate_safetensors_plan(store, &plan),
        issues,
    );
}

fn validate_quantization_companion(
    store: &SafetensorsWeightStore,
    name: &str,
    shape: &[usize],
    issues: &mut Vec<StructuralIssue>,
) {
    let plan = SafetensorsCheckpointPlan::new(
        format!("SafeTensors companion {name}"),
        vec![SafetensorsTensorConstraint::required(
            name,
            shape.to_vec(),
            StoredDtypeConstraint::OneOf(vec![
                StoredDtype::F16,
                StoredDtype::BF16,
                StoredDtype::F32,
                StoredDtype::U8,
            ]),
        )
        .companion()],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .expect("legacy companion constraints are valid");
    append_structural_issues(
        checkpoint_validation::validate_safetensors_plan(store, &plan),
        issues,
    );
}

fn validate_deepseek2_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    deepseek_v3_checkpoint::validate_gguf(checkpoint, metadata)
}

fn validate_deepseek4_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    deepseek_v4_checkpoint::validate_gguf(checkpoint, metadata)
}

fn validate_inkling_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    if let Err(error) = GgufArchitecture::Inkling.validate_load_policy(options) {
        return invalid_geometry(error.to_string());
    }
    inkling_checkpoint::validate_gguf(checkpoint, metadata)
}

pub(crate) fn validate_inkling_mmproj_gguf(
    model_metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: &inkling::InklingMmprojGguf,
) -> StructuralValidation {
    inkling_checkpoint::validate_mmproj_gguf(model_metadata, mmproj)
}

pub(crate) fn validate_gemma4_mmproj_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: &gemma4::Gemma4MmprojGguf,
) -> StructuralValidation {
    gemma4_checkpoint::validate_mmproj_gguf(model_checkpoint, model_metadata, mmproj)
}

fn validate_gemma4_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    if let Err(error) = GgufArchitecture::Gemma4.validate_load_policy(options) {
        return invalid_geometry(error.to_string());
    }
    gemma4_checkpoint::validate_gguf(checkpoint, metadata)
}

fn validate_gpt_oss_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    gpt_oss_checkpoint::validate_gguf(checkpoint, metadata)
}

fn validate_dense_qwen_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let variant = match architecture {
        GgufArchitecture::Qwen2 => dense_qwen_checkpoint::GgufVariant::Qwen2,
        GgufArchitecture::Qwen3 => dense_qwen_checkpoint::GgufVariant::Qwen3,
        GgufArchitecture::Qwen3Moe => dense_qwen_checkpoint::GgufVariant::Qwen3Moe,
        _ => unreachable!("dense Qwen GGUF validator received another architecture"),
    };
    dense_qwen_checkpoint::validate_gguf(variant, checkpoint, metadata)
}

pub(crate) fn validate_muse_glimmer_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    muse_glimmer_checkpoint::validate_projector_gguf(
        model_checkpoint,
        model_metadata,
        checkpoint,
        metadata,
    )
}

fn validate_qwen3_vl_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    if let Err(error) = architecture.validate_load_policy(options) {
        return invalid_geometry(error.to_string());
    }
    let variant = match architecture {
        GgufArchitecture::Qwen3Vl => qwen_vl_checkpoint::GgufVariant::Dense,
        GgufArchitecture::Qwen3VlMoe => qwen_vl_checkpoint::GgufVariant::Moe,
        _ => unreachable!("Qwen-VL GGUF validator received another architecture"),
    };
    qwen_vl_checkpoint::validate_gguf(variant, checkpoint, metadata)
}

pub(crate) fn validate_qwen3_vl_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    qwen_vl_checkpoint::validate_projector_gguf(
        model_checkpoint,
        model_metadata,
        checkpoint,
        metadata,
    )
}

pub(crate) fn validate_qwen35_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    qwen_hybrid_checkpoint::validate_projector_gguf(
        model_checkpoint,
        model_metadata,
        checkpoint,
        metadata,
    )
}
fn validate_qwen35_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let variant = match architecture {
        GgufArchitecture::Qwen35 => qwen_hybrid_checkpoint::GgufVariant::Qwen35,
        GgufArchitecture::Qwen35Moe => qwen_hybrid_checkpoint::GgufVariant::Qwen35Moe,
        GgufArchitecture::Qwen3Next => qwen_hybrid_checkpoint::GgufVariant::Qwen3Next,
        _ => unreachable!("Qwen hybrid GGUF validator received another architecture"),
    };
    qwen_hybrid_checkpoint::validate_gguf(
        variant,
        checkpoint,
        metadata,
        options.weight_residency.expert_cache().is_some(),
    )
}

fn is_float_dtype(dtype: &StoredDtype) -> bool {
    matches!(
        dtype,
        StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::F32
    )
}

fn finish(issues: Vec<StructuralIssue>) -> StructuralValidation {
    if issues.is_empty() {
        StructuralValidation::Exact
    } else {
        StructuralValidation::Invalid(issues)
    }
}

fn invalid_geometry(detail: String) -> StructuralValidation {
    StructuralValidation::Invalid(vec![StructuralIssue {
        kind: StructuralIssueKind::InvalidGeometry,
        detail,
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    }])
}

fn missing(name: &str) -> StructuralIssue {
    StructuralIssue {
        kind: StructuralIssueKind::MissingTensor,
        detail: format!("checkpoint is missing required tensor {name:?}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}

fn layout(name: &str, detail: String) -> StructuralIssue {
    StructuralIssue {
        kind: StructuralIssueKind::ConflictingLayout,
        detail: format!("could not validate tensor {name:?}: {detail}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}

fn unexpected_layout(name: &str, loader_name: &str) -> StructuralIssue {
    StructuralIssue {
        kind: StructuralIssueKind::UnexpectedTensor,
        detail: format!("{loader_name} catalog contains unexpected tensor {name:?}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}

#[cfg(test)]
mod admission_policy_tests {
    use super::*;

    #[test]
    fn non_strict_catalog_ignores_only_unexpected_tensors() {
        let unexpected = unexpected_layout("unrelated.weight", "test");
        let malformed =
            crate::runtime::checkpoint::contract::shape_mismatch("model.weight", &[2, 2], &[1]);
        assert_eq!(
            StructuralValidation::Invalid(vec![unexpected.clone(), malformed.clone()])
                .with_strict_catalog(false),
            StructuralValidation::Invalid(vec![malformed])
        );

        let error = StructuralValidation::Invalid(vec![unexpected])
            .into_loader_result()
            .unwrap_err();
        assert!(matches!(
            error,
            Error::StrictLoadValidation { missing, unused }
                if missing.is_empty() && unused == ["unrelated.weight"]
        ));
    }
}

#[cfg(test)]
mod dense_qwen_tests {
    use super::*;
    use crate::architectures::qwen::dense;

    fn qwen2_args(tied: bool) -> dense::DecoderConfig {
        dense::config_from_hf_value(&serde_json::json!({
            "model_type": "qwen2", "hidden_size": 8, "num_hidden_layers": 2,
            "intermediate_size": 16, "num_attention_heads": 4,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 32,
            "max_position_embeddings": 64, "rope_theta": 10000.0,
            "tie_word_embeddings": tied, "use_sliding_window": false
        }))
        .unwrap()
    }

    #[test]
    fn qwen2_plan_is_exactly_biased_and_has_no_qk_norms() {
        let tied = dense_qwen_checkpoint::safetensors_plan(&qwen2_args(true)).unwrap();
        let names = tied
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("model.layers.0.self_attn.q_proj.bias"));
        assert!(names.contains("model.layers.0.self_attn.k_proj.bias"));
        assert!(names.contains("model.layers.0.self_attn.v_proj.bias"));
        assert!(!names.contains("model.layers.0.self_attn.q_norm.weight"));
        assert!(!names.contains("model.layers.0.self_attn.k_norm.weight"));
        assert!(!names.contains("lm_head.weight"));

        let untied = dense_qwen_checkpoint::safetensors_plan(&qwen2_args(false)).unwrap();
        assert!(untied
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "lm_head.weight"));
    }
}
