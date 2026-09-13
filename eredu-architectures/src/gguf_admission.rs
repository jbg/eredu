//! Complete backend-neutral GGUF family admission.

use std::collections::{BTreeSet, HashMap};

use eredu_checkpoint::validation::{validate_gguf_plan, CheckpointValidation};
use eredu_gguf::{Checkpoint, MetadataValue};

use crate::{
    configuration::{GgufArchitecturePlan, GgufModelConfig},
    GgufArchitecture,
};

pub(crate) fn resolve(
    architecture: GgufArchitecture,
    checkpoint: &Checkpoint,
) -> Result<(GgufArchitecturePlan, CheckpointValidation), String> {
    let (model, plan, validation) = resolve_family(architecture, checkpoint)?;
    let tensor_mapping = canonical_tensor_mapping(checkpoint, &model)?;
    let validation = validation.unwrap_or_else(|| validate_gguf_plan(checkpoint, &plan));
    Ok((
        GgufArchitecturePlan::new(architecture, model, plan, tensor_mapping),
        validation,
    ))
}

fn canonical_tensor_mapping(
    checkpoint: &Checkpoint,
    model: &GgufModelConfig,
) -> Result<Vec<eredu_gguf::TranslatedTensorLayout>, String> {
    let mapping = match model {
        GgufModelConfig::DeepSeekV3(_) => {
            checkpoint.translated_outputs(crate::deepseek::translate_v3_gguf_weight_name)
        }
        GgufModelConfig::DeepSeekV4(_) => {
            checkpoint.translated_outputs(crate::deepseek::translate_v4_gguf_weight_name)
        }
        GgufModelConfig::Gemma4(_) => {
            checkpoint.translated_outputs(crate::gemma4::translate_family_gguf_weight_name)
        }
        GgufModelConfig::GptOss(_) => {
            checkpoint.translated_outputs(crate::gpt_oss::translate_gguf_weight_name)
        }
        GgufModelConfig::Inkling(args) => checkpoint.translated_outputs(|name| {
            crate::inkling::translate_gguf_weight_name_for_model(name, args)
        }),
        GgufModelConfig::K2Horizon(_) => {
            checkpoint.translated_outputs(crate::k2_horizon::translate_gguf_weight_name)
        }
        GgufModelConfig::KimiLinear(_) => {
            checkpoint.translated_outputs(crate::kimi_linear::translate_gguf_weight_name)
        }
        GgufModelConfig::Lfm2(args) => checkpoint.translated_outputs(|name| {
            crate::lfm2::translate_gguf_weight_name(name, args.has_sparse_moe_layers())
        }),
        GgufModelConfig::Gemma2(_) => {
            checkpoint.translated_outputs(crate::gemma2::translate_gguf_weight_name)
        }
        GgufModelConfig::Llama(_) | GgufModelConfig::Nanbeige(_) => {
            checkpoint.translated_outputs(crate::llama::translate_gguf_weight_name)
        }
        GgufModelConfig::MuseGlimmer(_) => {
            checkpoint.translated_outputs(crate::muse_glimmer::translate_text_gguf_name)
        }
        GgufModelConfig::NemotronH(_) => {
            checkpoint.translated_outputs(crate::nemotron_h::translate_gguf_weight_name)
        }
        GgufModelConfig::Qwen(args) => checkpoint.translated_outputs(|name| {
            crate::qwen::translate_gguf_weight_name(name, args.is_moe())
        }),
        GgufModelConfig::QwenHybrid(_) => {
            checkpoint.translated_outputs(crate::qwen::hybrid::translate_gguf_weight_name)
        }
    };
    mapping.map_err(|error| error.to_string())
}

// Derived expert banks and embeddings need their exact physical format before
// cold recipes and parameter topology are constructed, including composite roots.
fn qwen_checkpoint_formats(
    args: &crate::qwen::ModelArgs,
    checkpoint: &Checkpoint,
) -> Result<crate::qwen::ModelArgs, String> {
    let mut formats = HashMap::new();
    for shard in checkpoint.shards() {
        for tensor in shard.tensors() {
            if tensor.descriptor().dimensions.len() < 2 {
                continue;
            }
            if let Some(format) = crate::linear_format::gguf_tensor_format(tensor, shard.endian())?
                .weight_quantization()
            {
                let name = crate::qwen::translate_gguf_weight_name(
                    &tensor.descriptor().name,
                    args.is_moe(),
                );
                let name = name.strip_prefix("model.").map_or_else(
                    || name.clone(),
                    |local| format!("{}.{local}", args.parameter_root),
                );
                if formats.insert(name.clone(), format).is_some() {
                    return Err(format!("duplicate Qwen GGUF parameter format {name:?}"));
                }
            }
        }
    }
    crate::qwen::with_checkpoint_formats(args, formats)
}

fn resolve_family(
    architecture: GgufArchitecture,
    checkpoint: &Checkpoint,
) -> Result<
    (
        GgufModelConfig,
        eredu_checkpoint::schema::GgufCheckpointPlan,
        Option<CheckpointValidation>,
    ),
    String,
> {
    let mut metadata = checkpoint
        .metadata()
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<HashMap<String, MetadataValue>>();
    let vocabulary_key = format!("{}.vocab_size", architecture.metadata_name());
    if let std::collections::hash_map::Entry::Vacant(entry) = metadata.entry(vocabulary_key) {
        if let Some(vocabulary) = checkpoint
            .logical_outputs()
            .find(|output| output.name == "token_embd.weight")
            .and_then(|output| output.shape.first().copied())
        {
            entry.insert(MetadataValue::Uint64(vocabulary));
        }
    }
    match architecture {
        GgufArchitecture::DeepSeek2 => {
            let args = crate::deepseek::parse_v3_gguf(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            let plan = crate::deepseek::v3_gguf_plan(&args)?;
            Ok((GgufModelConfig::DeepSeekV3(args), plan, None))
        }
        GgufArchitecture::DeepSeek4 => {
            let args =
                crate::deepseek::parse_v4_gguf(&metadata).map_err(|error| error.to_string())?;
            let plan = crate::deepseek::v4_gguf_plan(&args)?;
            Ok((GgufModelConfig::DeepSeekV4(args), plan, None))
        }
        GgufArchitecture::Gemma4 => {
            let names = checkpoint
                .logical_outputs()
                .map(|output| output.name.clone())
                .collect::<BTreeSet<_>>();
            let text = crate::gemma4::ModelArgs::from_gguf_metadata(&names, &metadata)
                .map_err(|error| error.to_string())?;
            let family = crate::gemma4::family_from_gguf_metadata(text, &metadata, None)
                .map_err(|error| error.to_string())?;
            let plan = crate::gemma4::gguf_plan(&family.text)?;
            Ok((GgufModelConfig::Gemma4(family), plan, None))
        }
        GgufArchitecture::GptOss => {
            let args = crate::gpt_oss::model_args_from_gguf_catalog(&metadata)
                .map_err(|error| error.to_string())?;
            let plan = crate::gpt_oss::gguf_plan(&args)?;
            let validation = crate::gpt_oss::validate_gguf(checkpoint, &args);
            Ok((GgufModelConfig::GptOss(args), plan, Some(validation)))
        }
        GgufArchitecture::Inkling => {
            let args = crate::inkling::ModelArgs::from_gguf_metadata(&metadata)
                .map_err(|error| error.to_string())?;
            let plan = crate::inkling::gguf_plan(&args)?;
            Ok((GgufModelConfig::Inkling(args), plan, None))
        }
        GgufArchitecture::KimiLinear => {
            let args = crate::kimi_linear::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            let plan = crate::kimi_linear::gguf_plan(&args)?;
            Ok((GgufModelConfig::KimiLinear(args), plan, None))
        }
        GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
            let args = crate::lfm2::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            let plan = crate::lfm2::gguf_plan(&args)?;
            Ok((GgufModelConfig::Lfm2(args), plan, None))
        }
        GgufArchitecture::Gemma2 => {
            let args = crate::gemma2::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|e| e.to_string())?;
            let plan = crate::gemma2::gguf_plan(&args)?;
            Ok((GgufModelConfig::Gemma2(args), plan, None))
        }
        GgufArchitecture::Nanbeige => {
            let args = crate::nanbeige::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|e| e.to_string())?;
            let plan = crate::llama::gguf_plan(args.dense_config())?;
            Ok((GgufModelConfig::Nanbeige(args), plan, None))
        }
        GgufArchitecture::Llama | GgufArchitecture::Mistral => {
            let args = crate::llama::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            if args.num_hidden_layers as usize > checkpoint.physical_tensor_count() {
                return Err(format!(
                    "configured layer count {} exceeds the entire {}-tensor GGUF catalog",
                    args.num_hidden_layers,
                    checkpoint.physical_tensor_count()
                ));
            }
            let plan = crate::llama::gguf_plan(&args)?;
            Ok((GgufModelConfig::Llama(args), plan, None))
        }
        GgufArchitecture::MuseGlimmer => {
            let args = crate::muse_glimmer::DecoderConfig::from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            let plan = crate::muse_glimmer::gguf_plan(&args)?;
            Ok((GgufModelConfig::MuseGlimmer(args), plan, None))
        }
        GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
            let args = crate::nemotron_h::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            let plan = crate::nemotron_h::gguf_plan(&args)?;
            Ok((GgufModelConfig::NemotronH(args), plan, None))
        }
        GgufArchitecture::K2Horizon => {
            let mut args = crate::k2_horizon::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|e| e.to_string())?;
            crate::k2_horizon::normalize_gguf_formats(&mut args, checkpoint)?;
            let plan = crate::k2_horizon::gguf_plan(&args)?;
            Ok((GgufModelConfig::K2Horizon(args), plan, None))
        }
        GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
            let args = crate::qwen::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            let args = qwen_checkpoint_formats(&args, checkpoint)?;
            let plan = crate::qwen::gguf_plan(&args)?;
            Ok((GgufModelConfig::Qwen(args), plan, None))
        }
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
            let is_moe = architecture == GgufArchitecture::Qwen3VlMoe;
            let context = if is_moe {
                crate::qwen::TextConfigContext::Qwen3VlMoe
            } else {
                crate::qwen::TextConfigContext::Qwen3Vl
            };
            let args = crate::qwen::model_args_from_gguf_catalog_with_context(
                checkpoint, &metadata, context,
            )
            .map_err(|error| error.to_string())?;
            if args.is_moe() != is_moe {
                return Err("Qwen3-VL GGUF architecture and expert geometry disagree".into());
            }
            let args = qwen_checkpoint_formats(&args, checkpoint)?;
            let plan = crate::qwen::gguf_plan(&args)?;
            Ok((GgufModelConfig::Qwen(args), plan, None))
        }
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe | GgufArchitecture::Qwen3Next => {
            let parsed = crate::qwen::hybrid::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            let plan = crate::qwen::hybrid::gguf_plan(&parsed.text)?;
            Ok((GgufModelConfig::QwenHybrid(parsed), plan, None))
        }
    }
}

#[cfg(test)]
mod gemma_tests {
    use super::*;
    use eredu_gguf::{GgmlType, MetadataArray, TensorInput, Writer};
    use eredu_runtime::{ReplicatedTextParameterOwner, ReplicatedTextParameterRole};
    use std::collections::BTreeMap;

    #[test]
    fn gemma_gguf_admission_preserves_family_embedding_and_unit_ownership() {
        for sparse in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("gemma.gguf");
            let mut metadata = BTreeMap::from([
                (
                    "general.architecture".into(),
                    MetadataValue::String("gemma4".into()),
                ),
                ("gemma4.block_count".into(), MetadataValue::Uint32(2)),
                ("gemma4.embedding_length".into(), MetadataValue::Uint32(32)),
                (
                    "gemma4.embedding_length_per_layer_input".into(),
                    MetadataValue::Uint32(8),
                ),
                (
                    "gemma4.feed_forward_length".into(),
                    MetadataValue::Uint32(64),
                ),
                (
                    "gemma4.attention.head_count".into(),
                    MetadataValue::Uint32(2),
                ),
                (
                    "gemma4.attention.head_count_kv".into(),
                    MetadataValue::Uint32(2),
                ),
                (
                    "gemma4.attention.key_length".into(),
                    MetadataValue::Uint32(16),
                ),
                (
                    "gemma4.attention.layer_norm_rms_epsilon".into(),
                    MetadataValue::Float32(1e-5),
                ),
                (
                    "gemma4.attention.sliding_window_pattern".into(),
                    MetadataValue::Array(MetadataArray::Bool(vec![false, false])),
                ),
                ("gemma4.vocab_size".into(), MetadataValue::Uint32(32)),
                ("gemma4.context_length".into(), MetadataValue::Uint32(64)),
            ]);
            if sparse {
                metadata.insert("gemma4.expert_count".into(), MetadataValue::Uint32(4));
                metadata.insert("gemma4.expert_used_count".into(), MetadataValue::Uint32(2));
                metadata.insert(
                    "gemma4.expert_feed_forward_length".into(),
                    MetadataValue::Uint32(64),
                );
            }
            let args = crate::gemma4::ModelArgs::from_gguf_metadata(
                &BTreeSet::from(["output.weight".to_owned()]),
                &metadata.clone().into_iter().collect(),
            )
            .unwrap();
            let plan = crate::gemma4::gguf_plan(&args).unwrap();
            let tensors = plan
                .common_tensors
                .iter()
                .chain(
                    plan.layout_groups
                        .iter()
                        .filter_map(|group| group.variants.first())
                        .flat_map(|variant| &variant.tensors),
                )
                .map(|tensor| {
                    (
                        tensor.key.clone(),
                        tensor
                            .shape
                            .iter()
                            .rev()
                            .map(|&n| n as u64)
                            .collect::<Vec<_>>(),
                        vec![0.25f32.to_le_bytes(); tensor.shape.iter().product()].concat(),
                    )
                })
                .collect::<Vec<_>>();
            let inputs = tensors
                .iter()
                .map(|(name, dimensions, data)| TensorInput {
                    name,
                    dimensions,
                    ggml_type: GgmlType::F32,
                    data,
                })
                .collect::<Vec<_>>();
            Writer::default()
                .write(std::fs::File::create(&path).unwrap(), &metadata, &inputs)
                .unwrap();
            let inspection = crate::configuration::inspect_artifact(&path).unwrap();
            let requirements =
                crate::replicated_text::composite_text_requirements(&inspection).unwrap();
            let parameters = requirements.execution().parameters();
            let family = crate::gemma4::family_from_gguf_metadata(
                args,
                &metadata.into_iter().collect(),
                None,
            )
            .unwrap();
            let schema = crate::gemma4::safetensors_plan(&family).unwrap();
            let targets = schema
                .common_tensors
                .iter()
                .chain(
                    schema
                        .layout_groups
                        .iter()
                        .flat_map(|group| &group.variants)
                        .flat_map(|variant| &variant.tensors),
                )
                .flat_map(|tensor| {
                    std::iter::once(tensor.key.as_str())
                        .chain(tensor.aliases.iter().map(String::as_str))
                })
                .collect::<BTreeSet<_>>();
            for mapping in inspection
                .architecture_plan()
                .gguf_plan()
                .unwrap()
                .tensor_mapping()
            {
                assert!(
                    targets.contains(mapping.layout.name.as_str()),
                    "GGUF output {} must identify a family parameter",
                    mapping.layout.name
                );
            }
            let find = |name: &str| {
                parameters
                    .iter()
                    .find(|parameter| parameter.name() == name)
                    .unwrap()
            };
            let embedding = find("model.language_model.embed_tokens.weight");
            assert_eq!(embedding.role(), ReplicatedTextParameterRole::Embedding);
            assert_eq!(embedding.logical_shape(), [32, 32]);
            assert!(
                matches!(embedding.owner(), ReplicatedTextParameterOwner::StaticRole(role) if role == "embedding")
            );
            assert!(
                matches!(find("model.language_model.embed_tokens_per_layer.weight").owner(), ReplicatedTextParameterOwner::StaticRole(role) if role == "per_layer_embedding")
            );
            for layer in 0..2 {
                let name = format!("model.language_model.layers.{layer}.self_attn.q_proj.weight");
                let projection = find(&name);
                assert_eq!(projection.role(), ReplicatedTextParameterRole::LinearWeight);
                assert!(
                    matches!(projection.owner(), ReplicatedTextParameterOwner::ExecutionUnit { group, unit } if group == crate::gemma4::TEXT_EXECUTION_GROUP && *unit == layer)
                );
            }
        }
    }
}

#[cfg(test)]
mod packed_qwen_tests {
    use super::*;
    use eredu_checkpoint::WeightQuantization;
    use eredu_gguf::{GgmlType, TensorInput, Writer};
    use std::collections::BTreeMap;

    #[test]
    fn gguf_admission_retains_expert_encodings_before_recipe_selection() {
        for architecture in [GgufArchitecture::Qwen3Moe, GgufArchitecture::Qwen3VlMoe] {
            for mismatch in [false, true] {
                let root = tempfile::tempdir().unwrap();
                let path = root.path().join("packed.gguf");
                let metadata = BTreeMap::from([
                    (
                        "general.architecture".into(),
                        MetadataValue::String("qwen3moe".into()),
                    ),
                    (
                        "qwen3moe.embedding_length".into(),
                        MetadataValue::Uint32(32),
                    ),
                    ("qwen3moe.block_count".into(), MetadataValue::Uint32(2)),
                    (
                        "qwen3moe.attention.head_count".into(),
                        MetadataValue::Uint32(4),
                    ),
                    (
                        "qwen3moe.attention.head_count_kv".into(),
                        MetadataValue::Uint32(2),
                    ),
                    (
                        "qwen3moe.attention.layer_norm_rms_epsilon".into(),
                        MetadataValue::Float32(1e-6),
                    ),
                    ("qwen3moe.vocab_size".into(), MetadataValue::Uint32(32)),
                    ("qwen3moe.context_length".into(), MetadataValue::Uint32(128)),
                    (
                        "qwen3moe.expert_feed_forward_length".into(),
                        MetadataValue::Uint32(64),
                    ),
                    ("qwen3moe.expert_count".into(), MetadataValue::Uint32(4)),
                    (
                        "qwen3moe.expert_used_count".into(),
                        MetadataValue::Uint32(2),
                    ),
                ]);
                let metadata = metadata
                    .into_iter()
                    .map(|(key, value): (String, MetadataValue)| {
                        let key = key.replace("qwen3moe", architecture.metadata_name());
                        let value = if key == "general.architecture" {
                            MetadataValue::String(architecture.metadata_name().into())
                        } else {
                            value
                        };
                        (key, value)
                    })
                    .collect::<BTreeMap<_, _>>();
                let mut tensors = Vec::new();
                tensors.push((
                    "token_embd.weight".into(),
                    vec![32, 32],
                    GgmlType::Q8_0,
                    vec![1u8; 32 * 34],
                ));
                for layer in 0..2 {
                    for projection in ["gate", "up", "down"] {
                        let ty = if layer == 0 && !(mismatch && projection == "up") {
                            GgmlType::Q8_0
                        } else {
                            GgmlType::IQ4NL
                        };
                        let dimensions = if projection == "down" {
                            vec![64u64, 32, 4]
                        } else {
                            vec![32, 64, 4]
                        };
                        let (block, bytes) = ty.block_and_bytes().unwrap();
                        let data = vec![
                            1u8;
                            (dimensions.iter().product::<u64>() / block * bytes)
                                as usize
                        ];
                        tensors.push((
                            format!("blk.{layer}.ffn_{projection}_exps.weight"),
                            dimensions,
                            ty,
                            data,
                        ));
                    }
                }
                let inputs: Vec<_> = tensors
                    .iter()
                    .map(|(name, dimensions, ggml_type, data)| TensorInput {
                        name,
                        dimensions,
                        ggml_type: *ggml_type,
                        data,
                    })
                    .collect();
                Writer::default()
                    .write(std::fs::File::create(&path).unwrap(), &metadata, &inputs)
                    .unwrap();
                let checkpoint = Checkpoint::open(&path).unwrap();
                let result = resolve_family(architecture, &checkpoint);
                if mismatch {
                    assert!(result.unwrap_err().contains("gate/up formats disagree"));
                    continue;
                }
                let (GgufModelConfig::Qwen(args), _, _) = result.unwrap() else {
                    panic!("Qwen family");
                };
                assert_eq!(
                    args.weight_quantization_for(&format!(
                        "{}.embed_tokens.weight",
                        args.parameter_root
                    )),
                    Some(WeightQuantization::GgufIQuant {
                        ggml_type: GgmlType::Q8_0,
                        endian: eredu_gguf::Endian::Little
                    })
                );
                for (layer, ggml_type) in [(0, GgmlType::Q8_0), (1, GgmlType::IQ4NL)] {
                    for projection in ["gate_up_proj", "down_proj"] {
                        assert_eq!(
                            args.weight_quantization_for(&format!(
                                "{}.layers.{layer}.mlp.experts.{projection}",
                                args.parameter_root
                            )),
                            Some(WeightQuantization::GgufIQuant {
                                ggml_type,
                                endian: eredu_gguf::Endian::Little
                            })
                        );
                    }
                }
            }
        }
    }
}
