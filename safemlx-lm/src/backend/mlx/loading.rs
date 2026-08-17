//! MLX checkpoint materialization after backend-neutral planning.

use std::path::Path;

use crate::architectures::{
    deepseek_v3::model as deepseek_v3,
    deepseek_v4::model as deepseek_v4,
    gemma4::model as gemma4,
    gpt_oss::model as gpt_oss,
    inkling::model as inkling,
    kimi_linear::model as kimi_linear,
    lfm2::model as lfm2,
    llama::model as llama,
    qwen::{
        dense as dense_qwen,
        hybrid::{qwen3_5, qwen3_next},
        vl::{model as qwen3_vl, moe as qwen3_vl_moe},
    },
};
use safemlx::{
    ops::{GgufCheckpoint, GgufMetadataValue},
    Stream,
};
use safemlx_lm_core::{GgufArchitecture, ModelArtifact, ModelKind, ModelPreparationPlan};

use crate::{
    api::{Model, ModelLoadOptions},
    backend::mlx::structural,
    error::Error,
    runtime::checkpoint::quantization::WeightQuantization,
};

/// MLX arrays/modules plus architecture-derived side data from one GGUF artifact.
pub(crate) struct MaterializedGgufModel {
    pub(crate) model: Model,
    #[cfg(feature = "media-processing")]
    pub(crate) processor: Option<crate::runtime::media::ModelProcessor>,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) fn materialize_gguf_model(
    gguf_file: &Path,
    checkpoint: &safemlx::ops::GgufCheckpoint,
    metadata: &std::collections::HashMap<String, safemlx::ops::GgufMetadataValue>,
    gguf_architecture: GgufArchitecture,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MaterializedGgufModel, Error> {
    #[cfg(feature = "media-processing")]
    let mut processor = None;

    let (model, architecture_eos_token_ids) = if let Some(quantization) = options
        .quantization
        .filter(|_| options.weight_residency.is_fully_resident())
    {
        match gguf_architecture {
            GgufArchitecture::KimiLinear => {
                let loaded = kimi_linear::load_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model = crate::architectures::kimi_linear::layerwise::execute_transformed_kimi_linear_model(
                        loaded.model, stream, weights_stream,
                    )?;
                (Model::KimiLinear(model), loaded.eos_token_ids)
            }
            GgufArchitecture::DeepSeek2 => {
                let loaded = deepseek_v3::load_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model = crate::architectures::deepseek_v3::layerwise::execute_transformed_deepseek_v3_model(
                        loaded.model, stream, weights_stream,
                    )?;
                (Model::DeepSeekV3(model), loaded.eos_token_ids)
            }
            GgufArchitecture::DeepSeek4 => {
                let (loaded, eos_token_ids) = crate::architectures::deepseek_v4::layerwise::load_deepseek_v4_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        options.weight_residency,
                        Some(quantization),
                        stream,
                        weights_stream,
                    )?;
                (Model::DeepSeekV4Layerwise(Box::new(loaded)), eos_token_ids)
            }
            GgufArchitecture::GptOss => {
                let loaded = gpt_oss::load_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model =
                    crate::architectures::gpt_oss::layerwise::execute_transformed_gpt_oss_model(
                        loaded.model,
                        stream,
                        weights_stream,
                    )?;
                (Model::GptOss(model), loaded.eos_token_ids)
            }
            GgufArchitecture::Gemma4 => {
                let mmproj = gemma4::open_sibling_mmproj(gguf_file)?;
                #[cfg(any(feature = "image-processing", feature = "audio-processing"))]
                if let Some(mmproj) = &mmproj {
                    processor = Some(ModelProcessor::load_gemma4_gguf(
                        &metadata,
                        &mmproj.metadata,
                    )?);
                }
                let loaded = gemma4::load_gemma4_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    mmproj.as_ref(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model =
                        crate::architectures::gemma4::layerwise::execute_transformed_gemma4_model_with_modalities(
                            loaded.model,
                            loaded.vision_config,
                            loaded.audio_config,
                            stream,
                            weights_stream,
                        )?;
                (Model::Gemma4(Box::new(model)), loaded.eos_token_ids)
            }
            GgufArchitecture::Llama | GgufArchitecture::Mistral => {
                let loaded = llama::load_llama_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model =
                    crate::architectures::llama::layerwise::execute_transformed_llama_model(
                        loaded.model,
                        stream,
                        weights_stream,
                    )?;
                (Model::Llama(model), loaded.eos_token_ids)
            }
            GgufArchitecture::MuseGlimmer => {
                #[cfg(feature = "image-processing")]
                if let Some(mmproj) =
                    crate::architectures::muse_glimmer::open_sibling_mmproj(gguf_file)?
                {
                    processor = Some(ModelProcessor::load_muse_glimmer_gguf(&mmproj.metadata)?);
                }
                let (loaded, eos_token_ids) =
                    crate::architectures::muse_glimmer::layerwise::load_gguf_checkpoint(
                        &checkpoint,
                        &metadata,
                        gguf_architecture.metadata_name(),
                        options.weight_residency,
                        Some(quantization),
                        stream,
                        weights_stream,
                    )?;
                (Model::MuseGlimmer(loaded), eos_token_ids)
            }
            GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
                let loaded = lfm2::load_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model = crate::architectures::lfm2::layerwise::execute_transformed_lfm2_model(
                    loaded.model,
                    stream,
                    weights_stream,
                )?;
                (Model::Lfm2(model), loaded.eos_token_ids)
            }
            GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
                let loaded = dense_qwen::load_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model =
                    crate::architectures::qwen::dense::layerwise::execute_transformed_model(
                        loaded.model,
                        stream,
                        weights_stream,
                    )?;
                (Model::DenseQwen(model), loaded.eos_token_ids)
            }
            GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
                let mmproj_file = qwen3_vl::find_qwen3_vl_mmproj(gguf_file)?;
                let vision_checkpoint = GgufCheckpoint::open(mmproj_file)?;
                let vision_metadata =
                    crate::runtime::checkpoint::load::gguf_metadata(&vision_checkpoint);
                let loaded = qwen3_vl::load_qwen3_vl_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    &vision_checkpoint,
                    vision_metadata,
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let model =
                    crate::architectures::qwen::vl::layerwise::execute_transformed_qwen3_vl_model(
                        loaded.model,
                        stream,
                        weights_stream,
                    )?;
                let model = if gguf_architecture == GgufArchitecture::Qwen3VlMoe {
                    Model::Qwen3VlMoe(model)
                } else {
                    Model::Qwen3Vl(model)
                };
                (model, loaded.eos_token_ids)
            }
            GgufArchitecture::Qwen35
            | GgufArchitecture::Qwen35Moe
            | GgufArchitecture::Qwen3Next => {
                let mmproj = if gguf_architecture == GgufArchitecture::Qwen3Next {
                    None
                } else {
                    qwen3_5::open_sibling_mmproj(gguf_file)?
                };
                #[cfg(feature = "image-processing")]
                if mmproj.is_some() {
                    processor = ModelProcessor::load_qwen(gguf_sidecar_dir(gguf_file))?;
                }
                let loaded = qwen3_5::load_qwen3_5_gguf_checkpoint(
                    &checkpoint,
                    metadata.clone(),
                    mmproj.as_ref(),
                    Some(quantization),
                    stream,
                    weights_stream,
                )?;
                let is_next = gguf_architecture == GgufArchitecture::Qwen3Next;
                let model = crate::architectures::qwen::hybrid::layerwise::execute_transformed_qwen_hybrid_model(
                        loaded.model, quantization, stream, weights_stream,
                    )?;
                let model = if is_next {
                    Model::Qwen3Next(model)
                } else {
                    Model::Qwen35(model)
                };
                (model, loaded.eos_token_ids)
            }
            GgufArchitecture::Inkling => {
                let mmproj = inkling::open_sibling_mmproj(gguf_file)?;
                #[cfg(feature = "media-processing")]
                if mmproj.is_some() {
                    processor = Some(ModelProcessor::load_inkling_gguf(&metadata)?);
                }
                let (loaded, eos_token_ids) =
                    crate::architectures::inkling::layerwise::load_inkling_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        mmproj.as_ref(),
                        options.weight_residency,
                        Some(quantization),
                        stream,
                        weights_stream,
                    )?;
                (Model::Inkling(loaded), eos_token_ids)
            }
            GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
                let (loaded, eos_token_ids) = crate::architectures::nemotron_h::layerwise::load_nemotron_h_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        options.weight_residency,
                        Some(quantization),
                        stream,
                        weights_stream,
                    )?;
                (Model::NemotronH(loaded), eos_token_ids)
            }
        }
    } else {
        match gguf_architecture {
            GgufArchitecture::KimiLinear => {
                let (loaded, eos_token_ids) =
                        crate::architectures::kimi_linear::layerwise::load_kimi_linear_gguf_layerwise_model(
                            &checkpoint,
                            &metadata,
                            options.weight_residency,
                            options.quantization,
                            stream,
                            weights_stream,
                        )?;
                (Model::KimiLinear(loaded), eos_token_ids)
            }
            GgufArchitecture::DeepSeek2 => {
                let (loaded, eos_token_ids) =
                        crate::architectures::deepseek_v3::layerwise::load_deepseek_v3_gguf_layerwise_model(
                            &checkpoint,
                            &metadata,
                            options.weight_residency,
                            options.quantization,
                            stream,
                            weights_stream,
                        )?;
                (Model::DeepSeekV3(loaded), eos_token_ids)
            }
            GgufArchitecture::DeepSeek4 => {
                let (loaded, eos_token_ids) = crate::architectures::deepseek_v4::layerwise::load_deepseek_v4_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::DeepSeekV4Layerwise(Box::new(loaded)), eos_token_ids)
            }
            GgufArchitecture::GptOss => {
                let (loaded, eos_token_ids) =
                    crate::architectures::gpt_oss::layerwise::load_gpt_oss_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::GptOss(loaded), eos_token_ids)
            }
            GgufArchitecture::Inkling => {
                let mmproj = inkling::open_sibling_mmproj(gguf_file)?;
                #[cfg(feature = "media-processing")]
                if mmproj.is_some() {
                    processor = Some(ModelProcessor::load_inkling_gguf(&metadata)?);
                }
                let (loaded, eos_token_ids) =
                    crate::architectures::inkling::layerwise::load_inkling_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        mmproj.as_ref(),
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::Inkling(loaded), eos_token_ids)
            }
            GgufArchitecture::Gemma4 => {
                let mmproj = gemma4::open_sibling_mmproj(gguf_file)?;
                #[cfg(any(feature = "image-processing", feature = "audio-processing"))]
                if let Some(mmproj) = &mmproj {
                    processor = Some(ModelProcessor::load_gemma4_gguf(
                        &metadata,
                        &mmproj.metadata,
                    )?);
                }
                let (loaded, eos_token_ids) =
                    crate::architectures::gemma4::layerwise::load_gemma4_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        mmproj.as_ref(),
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::Gemma4(Box::new(loaded)), eos_token_ids)
            }
            GgufArchitecture::Llama | GgufArchitecture::Mistral => {
                let (loaded, eos_token_ids) =
                    crate::architectures::llama::layerwise::load_llama_gguf_model(
                        &checkpoint,
                        &metadata,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::Llama(loaded), eos_token_ids)
            }
            GgufArchitecture::MuseGlimmer => {
                #[cfg(feature = "image-processing")]
                if let Some(mmproj) =
                    crate::architectures::muse_glimmer::open_sibling_mmproj(gguf_file)?
                {
                    processor = Some(ModelProcessor::load_muse_glimmer_gguf(&mmproj.metadata)?);
                }
                let (loaded, eos_token_ids) =
                    crate::architectures::muse_glimmer::layerwise::load_gguf_checkpoint(
                        &checkpoint,
                        &metadata,
                        gguf_architecture.metadata_name(),
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::MuseGlimmer(loaded), eos_token_ids)
            }
            GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
                let (loaded, eos_token_ids) =
                    crate::architectures::lfm2::layerwise::load_lfm2_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::Lfm2(loaded), eos_token_ids)
            }
            GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
                let (loaded, eos_token_ids) =
                    crate::architectures::nemotron_h::layerwise::load_nemotron_h_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::NemotronH(loaded), eos_token_ids)
            }
            GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
                let (loaded, eos_token_ids) =
                    crate::architectures::qwen::dense::layerwise::load_gguf_checkpoint(
                        &checkpoint,
                        &metadata,
                        gguf_architecture.metadata_name(),
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                (Model::DenseQwen(loaded), eos_token_ids)
            }
            GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
                let mmproj_file = qwen3_vl::find_qwen3_vl_mmproj(gguf_file)?;
                let vision_checkpoint = GgufCheckpoint::open(mmproj_file)?;
                let vision_metadata =
                    crate::runtime::checkpoint::load::gguf_metadata(&vision_checkpoint);
                let (loaded, eos_token_ids) =
                    crate::architectures::qwen::vl::layerwise::load_qwen3_vl_gguf_layerwise_model(
                        &checkpoint,
                        &metadata,
                        &vision_checkpoint,
                        &vision_metadata,
                        options.weight_residency,
                        options.quantization,
                        stream,
                        weights_stream,
                    )?;
                let model = if gguf_architecture == GgufArchitecture::Qwen3VlMoe {
                    Model::Qwen3VlMoe(loaded)
                } else {
                    Model::Qwen3Vl(loaded)
                };
                (model, eos_token_ids)
            }
            GgufArchitecture::Qwen35
            | GgufArchitecture::Qwen35Moe
            | GgufArchitecture::Qwen3Next => {
                let mmproj = if gguf_architecture == GgufArchitecture::Qwen3Next {
                    None
                } else {
                    qwen3_5::open_sibling_mmproj(gguf_file)?
                };
                #[cfg(feature = "image-processing")]
                if mmproj.is_some() {
                    processor = ModelProcessor::load_qwen(gguf_sidecar_dir(gguf_file))?;
                }
                let (loaded, eos_token_ids, is_next) =
                        crate::architectures::qwen::hybrid::layerwise::load_qwen_hybrid_gguf_layerwise_model(
                            &checkpoint,
                            &metadata,
                            mmproj.as_ref(),
                            options.weight_residency,
                            options.quantization,
                            stream,
                            weights_stream,
                        )?;
                let model = if is_next {
                    Model::Qwen3Next(loaded)
                } else {
                    Model::Qwen35(loaded)
                };
                (model, eos_token_ids)
            }
        }
    };
    Ok(MaterializedGgufModel {
        model,
        #[cfg(feature = "media-processing")]
        processor,
        eos_token_ids: architecture_eos_token_ids,
    })
}

pub(super) fn materialize_model_plan(
    plan: ModelPreparationPlan,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    validate_plan_options(&plan, options)?;
    let (artifact, _policy, _route) = plan.into_parts();
    match artifact {
        artifact @ ModelArtifact::Gguf { .. } => {
            Ok(materialize_gguf_artifact(artifact, options, stream, weights_stream)?.model)
        }
        ModelArtifact::SafeTensors {
            path,
            configuration,
            ..
        } => materialize_safetensors(configuration.kind, &path, options, stream, weights_stream),
    }
}

/// Materializes a core-owned GGUF plan for the combined model/tokenizer facade.
pub(crate) fn materialize_gguf_plan(
    plan: ModelPreparationPlan,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MaterializedGgufModel, Error> {
    validate_plan_options(&plan, options)?;
    let (artifact, _policy, _route) = plan.into_parts();
    materialize_gguf_artifact(artifact, options, stream, weights_stream)
}

fn validate_plan_options(
    plan: &ModelPreparationPlan,
    options: ModelLoadOptions,
) -> Result<(), Error> {
    if plan.policy() != options.preparation_policy()? {
        return Err(Error::UnsupportedArchitecture(
            "MLX materialization options do not match the backend-neutral preparation plan".into(),
        ));
    }
    Ok(())
}

fn materialize_gguf_artifact(
    artifact: ModelArtifact,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MaterializedGgufModel, Error> {
    let ModelArtifact::Gguf {
        path,
        configuration,
        checkpoint,
        ..
    } = artifact
    else {
        return Err(Error::UnsupportedArchitecture(
            "MLX GGUF materializer received a SafeTensors plan".into(),
        ));
    };
    let checkpoint = safemlx::ops::GgufCheckpoint::from_portable(checkpoint);
    let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
    let architecture = configuration.gguf_architecture.ok_or_else(|| {
        Error::UnsupportedArchitecture("backend-neutral GGUF plan omitted its architecture".into())
    })?;
    structural::validate_gguf(architecture, &checkpoint, &metadata, options)
        .into_loader_result()?;
    validate_gguf_quantization_source(&checkpoint, &metadata, options.quantization)?;
    materialize_gguf_model(
        &path,
        &checkpoint,
        &metadata,
        architecture,
        options,
        stream,
        weights_stream,
    )
}

pub(crate) fn validate_gguf_quantization_source<
    S: crate::runtime::checkpoint::load::GgufTensorNames,
>(
    source: &S,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
) -> Result<(), Error> {
    let Some(quantization) = quantization else {
        return Ok(());
    };
    quantization.validate()?;

    let has_packed_companions = source.has_affine_gguf_tensor();
    if has_packed_companions {
        return Err(Error::Quantization(
            "load-time quantization accepts only unquantized F32/F16/BF16 GGUF weights; packed GGUF tensors cannot be implicitly transcoded"
                .into(),
        ));
    }

    let file_type = metadata
        .get("general.file_type")
        .ok_or_else(|| {
            Error::Quantization(
                "GGUF general.file_type metadata is required to verify that load-time quantization is not transcoding packed weights"
                    .into(),
            )
        })?
        .as_i64()
        .ok_or_else(|| {
            Error::Quantization("GGUF general.file_type metadata must be an integer".into())
        })?;
    // llama.cpp's unquantized file types: ALL_F32, MOSTLY_F16, and MOSTLY_BF16.
    if !matches!(file_type, 0 | 1 | 32) {
        return Err(Error::Quantization(format!(
            "load-time quantization accepts only unquantized F32/F16/BF16 GGUF weights; general.file_type={file_type} is already quantized"
        )));
    }
    Ok(())
}

pub(super) fn materialize_safetensors(
    kind: ModelKind,
    model_dir: &Path,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    structural::validate_safetensors_load_path(kind, model_dir, options)?;
    if let (Some(expert_cache), Some(non_expert)) = (
        options.weight_residency.expert_cache(),
        options.weight_residency.non_experts(),
    ) {
        return match kind {
            ModelKind::KimiLinear => Ok(Model::KimiLinear(
                crate::architectures::kimi_linear::layerwise::load_kimi_linear_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3(
                crate::architectures::deepseek_v3::layerwise::load_deepseek_v3_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::DeepSeekV4 => Ok(Model::DeepSeekV4Layerwise(Box::new(
                crate::architectures::deepseek_v4::layerwise::load_deepseek_v4_expert_cache_model(
                    model_dir,
                    non_expert,
                    expert_cache,
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            ))),
            ModelKind::GptOss => Ok(Model::GptOss(
                crate::architectures::gpt_oss::layerwise::load_gpt_oss_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Inkling => Ok(Model::Inkling(
                crate::architectures::inkling::layerwise::load_inkling_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Lfm2 => Ok(Model::Lfm2(
                crate::architectures::lfm2::layerwise::load_lfm2_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::NemotronH => Ok(Model::NemotronH(
                crate::architectures::nemotron_h::layerwise::load_nemotron_h_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen2 => Err(Error::UnsupportedArchitecture(
                "Qwen2 is dense and does not support sparse expert-cache residency".into(),
            )),
            ModelKind::Qwen3 => Ok(Model::DenseQwen(
                crate::architectures::qwen::dense::layerwise::load_qwen3_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen3Next => Ok(Model::Qwen3Next(
                crate::architectures::qwen::hybrid::layerwise::load_qwen3_next_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoe(
                crate::architectures::qwen::vl::layerwise::load_qwen3_vl_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Qwen35 => Ok(Model::Qwen35(
                crate::architectures::qwen::hybrid::layerwise::load_qwen35_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            _ => Err(Error::UnsupportedArchitecture(format!(
                "independent expert caching requires a supported safetensors MoE architecture, not {}",
                kind.model_type_name()
            ))),
        };
    }
    let execution = options.weight_residency.layers();
    if let Some(quantization) = options.quantization {
        quantization.validate()?;
        return match kind {
            ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3(
                crate::architectures::deepseek_v3::layerwise::execute_transformed_deepseek_v3_model(
                    deepseek_v3::load_model_quantized(model_dir, quantization, stream, weights_stream)?,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::DeepSeekV4 => Ok(Model::DeepSeekV4Layerwise(Box::new(
                crate::architectures::deepseek_v4::layerwise::load_deepseek_v4_layerwise_model(
                    model_dir,
                    execution,
                    Some(quantization),
                    stream,
                    weights_stream,
                )?,
            ))),
            ModelKind::Gemma4 => Ok(Model::Gemma4(Box::new(
                crate::architectures::gemma4::layerwise::execute_transformed_gemma4_model(
                    model_dir,
                    gemma4::load_gemma4_model_quantized(model_dir, quantization, stream, weights_stream)?,
                    stream,
                    weights_stream,
                )?,
            ))),
            ModelKind::Inkling => Ok(Model::Inkling(
                crate::architectures::inkling::layerwise::load_inkling_layerwise_model(
                    model_dir,
                    execution,
                    Some(quantization),
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::GptOss => Ok(Model::GptOss(
                crate::architectures::gpt_oss::layerwise::execute_transformed_gpt_oss_model(
                    gpt_oss::load_model_quantized(model_dir, quantization, stream, weights_stream)?,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::KimiLinear => Ok(Model::KimiLinear(
                crate::architectures::kimi_linear::layerwise::execute_transformed_kimi_linear_model(
                    kimi_linear::load_model_quantized(model_dir, quantization, stream, weights_stream)?,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Llama => Ok(Model::Llama(
                crate::architectures::llama::layerwise::execute_transformed_llama_model(
                    llama::load_resident_llama_model_quantized(model_dir, quantization, stream, weights_stream)?,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::MuseGlimmer => Ok(Model::MuseGlimmer(
                crate::architectures::muse_glimmer::layerwise::load_safetensors_quantized_residency(
                    model_dir,
                    execution,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Lfm2 => Ok(Model::Lfm2(
                crate::architectures::lfm2::layerwise::execute_transformed_lfm2_model(
                    lfm2::load_model_quantized(model_dir, quantization, stream, weights_stream)?,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::NemotronH => Ok(Model::NemotronH(
                crate::architectures::nemotron_h::layerwise::load_nemotron_h_layerwise_model(
                    model_dir,
                    execution,
                    Some(quantization),
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen2 | ModelKind::Qwen3 => Ok(Model::DenseQwen(
                crate::architectures::qwen::dense::layerwise::load_safetensors_quantized_residency(
                    model_dir,
                    execution,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3Next => Ok(Model::Qwen3Next(
                crate::architectures::qwen::hybrid::layerwise::execute_transformed_qwen_hybrid_model(
                    qwen3_next::load_qwen3_next_model_quantized(model_dir, quantization, stream, weights_stream)?,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3Vl => Ok(Model::Qwen3Vl(
                crate::architectures::qwen::vl::layerwise::execute_transformed_qwen3_vl_model(
                    qwen3_vl::load_qwen3_vl_model_quantized(model_dir, quantization, stream, weights_stream)?,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoe(
                crate::architectures::qwen::vl::layerwise::execute_transformed_qwen3_vl_model(
                    qwen3_vl_moe::load_qwen3_vl_moe_model_quantized(model_dir, quantization, stream, weights_stream)?,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen35 => Ok(Model::Qwen35(
                crate::architectures::qwen::hybrid::layerwise::execute_transformed_qwen_hybrid_model(
                    qwen3_5::load_qwen3_5_model_quantized(model_dir, quantization, stream, weights_stream)?,
                    quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::PersonaPlex => {
                unreachable!("load policy rejects unsupported load-time transformations")
            }
        };
    }
    match kind {
        ModelKind::DeepSeekV3 => Ok(Model::DeepSeekV3(
            crate::architectures::deepseek_v3::layerwise::load_deepseek_v3_layerwise_model(
                model_dir,
                execution,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::DeepSeekV4 if execution.is_fully_resident() => Ok(Model::DeepSeekV4(Box::new(
            deepseek_v4::load_model(model_dir, stream, weights_stream)?,
        ))),
        ModelKind::DeepSeekV4 => Ok(Model::DeepSeekV4Layerwise(Box::new(
            crate::architectures::deepseek_v4::layerwise::load_deepseek_v4_layerwise_model(
                model_dir,
                execution,
                None,
                stream,
                weights_stream,
            )?,
        ))),
        ModelKind::Gemma4 => Ok(Model::Gemma4(Box::new(
            crate::architectures::gemma4::layerwise::load_gemma4_layerwise_model(
                model_dir,
                execution,
                stream,
                weights_stream,
            )?,
        ))),
        ModelKind::Inkling => Ok(Model::Inkling(
            crate::architectures::inkling::layerwise::load_inkling_layerwise_model(
                model_dir,
                execution,
                None,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::KimiLinear => Ok(Model::KimiLinear(
            crate::architectures::kimi_linear::layerwise::load_kimi_linear_layerwise_model(
                model_dir,
                execution,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Llama => Ok(Model::Llama(
            crate::architectures::llama::layerwise::load_llama_safetensors_mlx(
                model_dir,
                execution.weight_residency(),
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::MuseGlimmer => Ok(Model::MuseGlimmer(
            crate::architectures::muse_glimmer::layerwise::load_safetensors(
                model_dir,
                execution,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen2 | ModelKind::Qwen3 => Ok(Model::DenseQwen(
            crate::architectures::qwen::dense::layerwise::load_safetensors(
                model_dir,
                execution,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::GptOss => Ok(Model::GptOss(
            crate::architectures::gpt_oss::layerwise::load_gpt_oss_layerwise_model(
                model_dir,
                execution,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Lfm2 => Ok(Model::Lfm2(
            crate::architectures::lfm2::layerwise::load_lfm2_layerwise_model(
                model_dir,
                execution,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::NemotronH => Ok(Model::NemotronH(
            crate::architectures::nemotron_h::layerwise::load_nemotron_h_layerwise_model(
                model_dir,
                execution,
                None,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen3Next => Ok(Model::Qwen3Next(
            crate::architectures::qwen::hybrid::layerwise::load_qwen3_next_layerwise_model(
                model_dir,
                execution,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen3Vl => Ok(Model::Qwen3Vl(
            crate::architectures::qwen::vl::layerwise::load_qwen3_vl_layerwise_model(
                model_dir,
                execution,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoe(
            crate::architectures::qwen::vl::layerwise::load_qwen3_vl_layerwise_model(
                model_dir,
                execution,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen35 => Ok(Model::Qwen35(
            crate::architectures::qwen::hybrid::layerwise::load_qwen35_layerwise_model(
                model_dir,
                execution,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::PersonaPlex => Err(Error::UnsupportedArchitecture(
            "PersonaPlex bounded layer residency is selected through the realtime loader".into(),
        )),
    }
}
