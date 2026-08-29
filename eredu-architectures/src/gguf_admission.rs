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
            checkpoint.translated_outputs(crate::gemma4::translate_gguf_weight_name)
        }
        GgufModelConfig::GptOss(_) => {
            checkpoint.translated_outputs(crate::gpt_oss::translate_gguf_weight_name)
        }
        GgufModelConfig::Inkling(args) => checkpoint.translated_outputs(|name| {
            crate::inkling::translate_gguf_weight_name_for_model(name, args)
        }),
        GgufModelConfig::KimiLinear(_) => {
            checkpoint.translated_outputs(crate::kimi_linear::translate_gguf_weight_name)
        }
        GgufModelConfig::Lfm2(args) => checkpoint.translated_outputs(|name| {
            crate::lfm2::translate_gguf_weight_name(name, args.has_sparse_moe_layers())
        }),
        GgufModelConfig::Llama(_) => {
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
        GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
            let args = crate::qwen::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
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
