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
    let (model, plan, validation, lfm2, lfm3, kimi, nemotron, dense, gpt_oss, qwen, k2) =
        resolve_family(architecture, checkpoint)?;
    let tensor_mapping = canonical_tensor_mapping(checkpoint, &model)?;
    let validation = validation.unwrap_or_else(|| validate_gguf_plan(checkpoint, &plan));
    Ok((
        GgufArchitecturePlan::new(
            architecture,
            model,
            plan,
            tensor_mapping,
            lfm2,
            lfm3,
            kimi,
            nemotron,
            dense,
            gpt_oss,
            qwen,
            k2,
        ),
        validation,
    ))
}

fn canonical_tensor_mapping(
    checkpoint: &Checkpoint,
    model: &GgufModelConfig,
) -> Result<Vec<eredu_gguf::TranslatedTensorLayout>, String> {
    let mapping = match model {
        GgufModelConfig::NemotronLabsDiffusion(plan) => return Ok(plan.mapping().to_vec()),
        GgufModelConfig::Qwen4Exp(plan) => return Ok(plan.text_plan().mapping().to_vec()),
        GgufModelConfig::VoiceChat(_) => checkpoint.translated_outputs(str::to_owned),
        GgufModelConfig::DeepSeekV3(_) => {
            checkpoint.translated_outputs(crate::deepseek::translate_v3_gguf_weight_name)
        }
        GgufModelConfig::DeepSeekV4(_) => {
            checkpoint.translated_outputs(crate::deepseek::translate_v4_gguf_weight_name)
        }
        GgufModelConfig::DiffusionGemma(config) => {
            let names = crate::diffusion_gemma::GgufNames::new(config)?;
            checkpoint.translated_outputs(|name| names.translate(name))
        }
        GgufModelConfig::Gemma4(config) => {
            let names = crate::gemma4::TextGgufNames::new(
                &config.text,
                config.pinned(),
                config.blocks(),
                config.dense_scopes(),
                config.attention_scopes(),
                config.routed_scopes(),
            )?;
            checkpoint.translated_outputs(|name| names.translate_family(name))
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
        GgufModelConfig::Lfm3(_) => {
            checkpoint.translated_outputs(crate::lfm3::translate_gguf_weight_name)
        }
        GgufModelConfig::Gemma2(_) => {
            checkpoint.translated_outputs(crate::gemma2::translate_gguf_weight_name)
        }
        GgufModelConfig::Llama(args) => checkpoint.translated_outputs(|name| {
            crate::llama::translate_gguf_weight_name_for_model(name, args)
        }),
        GgufModelConfig::Nanbeige(_) => {
            checkpoint.translated_outputs(crate::llama::translate_gguf_weight_name)
        }
        GgufModelConfig::Muse(args) => {
            let names = crate::muse::TextGgufNames::new(args)?;
            checkpoint.translated_outputs(|name| names.translate(name))
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

fn checkpoint_linear_formats(
    checkpoint: &Checkpoint,
    translate: impl Fn(&str) -> String,
) -> Result<HashMap<String, eredu_checkpoint::WeightQuantization>, String> {
    let mut formats = HashMap::new();
    for shard in checkpoint.shards() {
        for tensor in shard
            .tensors()
            .iter()
            .filter(|tensor| tensor.descriptor().dimensions.len() >= 2)
        {
            if let Some(format) = crate::linear_format::gguf_tensor_format(tensor, shard.endian())?
                .weight_quantization()
            {
                let name = translate(&tensor.descriptor().name);
                if formats.insert(name.clone(), format).is_some() {
                    return Err(format!("duplicate GGUF matrix format {name:?}"));
                }
            }
        }
    }
    Ok(formats)
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
        Option<std::sync::Arc<crate::lfm2::Declaration>>,
        Option<std::sync::Arc<crate::lfm3::Declaration>>,
        Option<std::sync::Arc<crate::kimi_linear::Declaration>>,
        Option<std::sync::Arc<crate::nemotron_h::Declaration>>,
        Option<std::sync::Arc<crate::decoder::DecoderDeclaration>>,
        Option<std::sync::Arc<crate::gpt_oss::Declaration>>,
        Option<std::sync::Arc<crate::qwen::Declaration>>,
        Option<std::sync::Arc<crate::k2_horizon::Declaration>>,
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
    let mut lfm2 = None;
    let mut lfm3 = None;
    let mut kimi = None;
    let mut nemotron = None;
    let mut dense = None;
    let mut gpt_oss = None;
    let mut qwen = None;
    let mut k2 = None;
    let result = match architecture {
        GgufArchitecture::NemotronLabsDiffusion => {
            let admitted =
                crate::nemotron_labs_diffusion::gguf::TextGgufPlan::from_checkpoint(checkpoint)
                    .map_err(|error| error.to_string())?;
            let plan = admitted.checkpoint().clone();
            Ok((GgufModelConfig::NemotronLabsDiffusion(admitted), plan, None))
        }

        GgufArchitecture::VoiceChat => {
            use crate::voicechat::checkpoint::{
                config_from_gguf_metadata, gguf_plan, validate_gguf_metadata, GgufLayout,
            };
            let config = config_from_gguf_metadata(checkpoint.metadata())?;
            validate_gguf_metadata(&config, GgufLayout::Complete, checkpoint.metadata())?;
            let plan = gguf_plan(&config, GgufLayout::Complete)?;
            Ok((GgufModelConfig::VoiceChat(config), plan, None))
        }
        GgufArchitecture::Qwen4Exp => {
            let header = crate::qwen4_exp::prepared::GgufTargetPlan::prepare(checkpoint)
                .map_err(|error| error.to_string())?;
            let plan = header.text_plan().checkpoint_plan().clone();
            Ok((
                GgufModelConfig::Qwen4Exp(std::sync::Arc::new(header)),
                plan,
                Some(CheckpointValidation::Exact),
            ))
        }
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
        GgufArchitecture::DiffusionGemma => {
            let config =
                crate::diffusion_gemma::FamilyConfig::from_gguf_metadata(checkpoint, &metadata)
                    .map_err(|error| error.to_string())?;
            let names = crate::diffusion_gemma::GgufNames::new(&config)?;
            let formats = checkpoint_linear_formats(checkpoint, |name| names.translate(name))?;
            let config = config
                .with_checkpoint_formats(formats)
                .map_err(|e| e.to_string())?;
            let plan = crate::diffusion_gemma::gguf_plan(&config)?;
            Ok((GgufModelConfig::DiffusionGemma(config), plan, None))
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
            let plan = crate::gemma4::gguf_plan(&family)?;
            Ok((GgufModelConfig::Gemma4(family), plan, None))
        }
        GgufArchitecture::GptOss => {
            let args = crate::gpt_oss::model_args_from_gguf_catalog(&metadata)
                .map_err(|error| error.to_string())?;
            let declaration = std::sync::Arc::new(
                crate::gpt_oss::Declaration::new(args.clone()).map_err(|e| e.to_string())?,
            );
            let plan = crate::gpt_oss::gguf_plan(&declaration)?;
            let validation = crate::gpt_oss::validate_gguf(checkpoint, &declaration);
            gpt_oss = Some(declaration);
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
            let declaration = std::sync::Arc::new(
                crate::kimi_linear::Declaration::new(args.clone()).map_err(|e| e.to_string())?,
            );
            let plan = crate::kimi_linear::gguf_plan(&declaration)?;
            kimi = Some(declaration);
            Ok((GgufModelConfig::KimiLinear(args), plan, None))
        }
        GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
            let args = crate::lfm2::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            let declaration = std::sync::Arc::new(
                crate::lfm2::Declaration::new(args.clone()).map_err(|e| e.to_string())?,
            );
            let plan = crate::lfm2::gguf_plan(&declaration)?;
            lfm2 = Some(declaration);
            Ok((GgufModelConfig::Lfm2(args), plan, None))
        }
        GgufArchitecture::Lfm3 => {
            let args = crate::lfm3::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            crate::lfm3::checkpoint::validate_gguf_rope(checkpoint, &args)?;
            let declaration = std::sync::Arc::new(
                crate::lfm3::Declaration::new(args.clone()).map_err(|e| e.to_string())?,
            );
            let plan = if crate::lfm3::vision::gguf::is_released_backbone(checkpoint) {
                crate::lfm3::vision::gguf::vl_backbone_plan(checkpoint, &declaration)?
            } else {
                if checkpoint
                    .tensors()
                    .any(|tensor| tensor.descriptor().ggml_type == eredu_gguf::GgmlType::NvFp4)
                {
                    return Err("LFM3 NVFP4 requires the released VL scale layout".into());
                }
                crate::lfm3::gguf_plan(&declaration)?
            };
            lfm3 = Some(declaration);
            Ok((GgufModelConfig::Lfm3(args), plan, None))
        }
        GgufArchitecture::Gemma2 => {
            let args = crate::gemma2::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|e| e.to_string())?;
            let declaration = std::sync::Arc::new(
                crate::decoder::DecoderDeclaration::new(&args).map_err(|e| e.to_string())?,
            );
            let plan = crate::gemma2::gguf_plan(&args, &declaration)?;
            dense = Some(declaration);
            Ok((GgufModelConfig::Gemma2(args), plan, None))
        }
        GgufArchitecture::Nanbeige => {
            let args = crate::nanbeige::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|e| e.to_string())?;
            let declaration = std::sync::Arc::new(
                crate::decoder::DecoderDeclaration::new(&args).map_err(|e| e.to_string())?,
            );
            let plan = crate::nanbeige::gguf_plan(&args, &declaration)?;
            dense = Some(declaration);
            Ok((GgufModelConfig::Nanbeige(args), plan, None))
        }
        GgufArchitecture::Llama | GgufArchitecture::Mistral | GgufArchitecture::Phi3 => {
            let args = crate::llama::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            if args.num_hidden_layers as usize > checkpoint.physical_tensor_count() {
                return Err(format!(
                    "configured layer count {} exceeds the entire {}-tensor GGUF catalog",
                    args.num_hidden_layers,
                    checkpoint.physical_tensor_count()
                ));
            }
            let declaration = std::sync::Arc::new(
                crate::decoder::DecoderDeclaration::new(&args).map_err(|e| e.to_string())?,
            );
            let plan = crate::llama::gguf_plan(&args, &declaration)?;
            dense = Some(declaration);
            Ok((GgufModelConfig::Llama(args), plan, None))
        }
        GgufArchitecture::MuseGlimmer | GgufArchitecture::MuseSpark => {
            let args = crate::muse::DecoderConfig::from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            let plan = crate::muse::gguf_plan(&args)?;
            Ok((GgufModelConfig::Muse(args), plan, None))
        }
        GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
            let args = crate::nemotron_h::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            let declaration = std::sync::Arc::new(
                crate::nemotron_h::Declaration::new(args.clone()).map_err(|e| e.to_string())?,
            );
            let plan = declaration.gguf_plan()?;
            nemotron = Some(declaration);
            Ok((GgufModelConfig::NemotronH(args), plan, None))
        }
        GgufArchitecture::K2Horizon => {
            let mut args = crate::k2_horizon::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|e| e.to_string())?;
            crate::k2_horizon::normalize_gguf_formats(&mut args, checkpoint)?;
            let declaration = std::sync::Arc::new(
                crate::k2_horizon::Declaration::new(args.clone()).map_err(|e| e.to_string())?,
            );
            let plan = crate::k2_horizon::gguf_plan(&declaration)?;
            dense = (!args.is_moe()).then(|| declaration.common().clone());
            k2 = Some(declaration);
            Ok((GgufModelConfig::K2Horizon(args), plan, None))
        }
        GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
            let args = crate::qwen::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            let args = qwen_checkpoint_formats(&args, checkpoint)?;
            let declaration = std::sync::Arc::new(
                crate::qwen::Declaration::new(args.clone()).map_err(|e| e.to_string())?,
            );
            let plan = crate::qwen::gguf_plan(&declaration)?;
            dense = (!args.is_moe()).then(|| declaration.common().clone());
            qwen = Some(declaration);
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
            let declaration = std::sync::Arc::new(
                crate::qwen::Declaration::new(args.clone()).map_err(|e| e.to_string())?,
            );
            let plan = crate::qwen::gguf_plan(&declaration)?;
            dense = (!args.is_moe()).then(|| declaration.common().clone());
            qwen = Some(declaration);
            Ok((GgufModelConfig::Qwen(args), plan, None))
        }
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe | GgufArchitecture::Qwen3Next => {
            let parsed = crate::qwen::hybrid::model_args_from_gguf_catalog(checkpoint, &metadata)
                .map_err(|error| error.to_string())?;
            let plan = crate::qwen::hybrid::gguf_plan(&parsed.text)?;
            Ok((GgufModelConfig::QwenHybrid(parsed), plan, None))
        }
    };
    result.map(|(model, plan, validation)| {
        (
            model, plan, validation, lfm2, lfm3, kimi, nemotron, dense, gpt_oss, qwen, k2,
        )
    })
}
