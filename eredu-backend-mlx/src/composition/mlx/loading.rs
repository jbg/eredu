//! MLX checkpoint materialization after backend-neutral planning.

use eredu_checkpoint::WeightQuantization;

use std::path::Path;

use eredu_core::{GgufArchitecture, ModelArtifact, ModelKind, ModelPreparationPlan};
use safemlx::{
    ops::{GgufCheckpoint, GgufMetadataValue},
    Stream,
};

#[cfg(feature = "media")]
use crate::composition::mlx::{load_processor, ModelProcessor};

pub fn gguf_eos_token_ids(
    metadata: &std::collections::HashMap<String, eredu_gguf::MetadataValue>,
) -> Result<Vec<u32>, Error> {
    const KEY: &str = "tokenizer.ggml.eos_token_id";
    Ok(eredu_core::gguf_u32_metadata_values(
        KEY,
        metadata.get(KEY),
    )?)
}
use crate::{
    backend::mlx::error::Error,
    backend::mlx::{MlxModel, ModelLoadOptions},
    composition::mlx::{structural, Model},
};

/// MLX arrays/modules plus backend-owned preprocessing from one GGUF artifact.
struct MaterializedGgufModel {
    model: Model,
    #[cfg(feature = "media")]
    processor: Option<ModelProcessor>,
}

fn materialize_gguf_model(
    gguf_file: &Path,
    checkpoint: &safemlx::ops::GgufCheckpoint,
    metadata: &std::collections::HashMap<String, safemlx::ops::GgufMetadataValue>,
    gguf_architecture: GgufArchitecture,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MaterializedGgufModel, Error> {
    #[cfg(feature = "media")]
    let mut processor = None;

    let (model, _architecture_eos_token_ids) = match gguf_architecture {
        GgufArchitecture::KimiLinear => {
            let (loaded, eos_token_ids) =
                crate::composition::kimi_linear::load_kimi_linear_gguf_model(
                    checkpoint,
                    metadata,
                    options.weight_residency,
                    options.quantization,
                    stream,
                    weights_stream,
                )?;
            (Model::KimiLinear(loaded), eos_token_ids)
        }
        GgufArchitecture::DeepSeek2 => {
            if options.quantization.is_some() {
                return Err(Error::UnsupportedArchitecture(
                    "load-time quantization is not yet supported by neutral DeepSeek GGUF composition".into(),
                ));
            }
            let (loaded, eos_token_ids) = crate::composition::deepseek::load_gguf(
                checkpoint,
                metadata,
                false,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            (Model::DeepSeek(Box::new(loaded)), eos_token_ids)
        }
        GgufArchitecture::DeepSeek4 => {
            if options.quantization.is_some() {
                return Err(Error::UnsupportedArchitecture(
                    "load-time quantization is not yet supported by neutral DeepSeek GGUF composition".into(),
                ));
            }
            let (loaded, eos_token_ids) = crate::composition::deepseek::load_gguf(
                checkpoint,
                metadata,
                true,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            (Model::DeepSeek(Box::new(loaded)), eos_token_ids)
        }
        GgufArchitecture::GptOss => {
            let (loaded, eos_token_ids) =
                crate::composition::gpt_oss::load_gpt_oss_gguf_layerwise_model(
                    checkpoint,
                    metadata,
                    options.weight_residency,
                    options.quantization,
                    stream,
                    weights_stream,
                )?;
            (Model::GptOss(loaded), eos_token_ids)
        }
        GgufArchitecture::Inkling => {
            #[cfg(feature = "media")]
            if crate::composition::mlx::artifact::find_sibling_mmproj(gguf_file, "inkling")?
                .is_some()
            {
                processor = Some(ModelProcessor::load_inkling_gguf(metadata)?);
            }
            if options.quantization.is_some() {
                return Err(Error::UnsupportedArchitecture(
                    "load-time Inkling quantization is not bound on the neutral loader".into(),
                ));
            }
            let loaded = crate::composition::inkling::load_gguf(
                gguf_file,
                checkpoint,
                metadata,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            let eos_token_ids = gguf_eos_token_ids(metadata)?;
            (Model::Inkling(loaded), eos_token_ids)
        }
        GgufArchitecture::Gemma4 => {
            #[cfg(any(feature = "image", feature = "audio"))]
            if let Some(mmproj_path) =
                crate::composition::mlx::artifact::find_sibling_mmproj(gguf_file, "gemma4")?
            {
                let mmproj = GgufCheckpoint::open(mmproj_path)?;
                processor = Some(ModelProcessor::load_gemma4_gguf(
                    metadata,
                    &crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&mmproj),
                )?);
            }
            if options.quantization.is_some() {
                return Err(Error::UnsupportedArchitecture(
                    "load-time Gemma 4 quantization is not bound on the neutral loader".into(),
                ));
            }
            let loaded = crate::composition::gemma4::load_gguf(
                gguf_file,
                checkpoint,
                metadata,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            (Model::Gemma4(loaded), gguf_eos_token_ids(metadata)?)
        }
        GgufArchitecture::Llama | GgufArchitecture::Mistral => {
            let (loaded, eos_token_ids) = crate::composition::llama::load_llama_gguf_model(
                checkpoint,
                metadata,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            (Model::Llama(loaded), eos_token_ids)
        }
        GgufArchitecture::MuseGlimmer => {
            #[cfg(feature = "image")]
            if let Some(mmproj) =
                crate::composition::mlx::artifact::find_sibling_mmproj(gguf_file, "muse-glimmer")?
            {
                let checkpoint = GgufCheckpoint::open(mmproj)?;
                processor = Some(ModelProcessor::load_muse_glimmer_gguf(
                    &crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint),
                )?);
            }
            if options.quantization.is_some() {
                return Err(Error::UnsupportedArchitecture(
                    "load-time Muse-Glimmer quantization is not bound on the neutral loader".into(),
                ));
            }
            let loaded = crate::composition::muse_glimmer::load_gguf(
                gguf_file,
                checkpoint,
                metadata,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            let eos_token_ids = gguf_eos_token_ids(metadata)?;
            (Model::MuseGlimmer(loaded), eos_token_ids)
        }
        GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
            let (loaded, eos_token_ids) = crate::composition::lfm2::load_lfm2_gguf_model(
                checkpoint,
                metadata,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            (Model::Lfm2(loaded), eos_token_ids)
        }
        GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
            let (loaded, eos_token_ids) =
                crate::composition::nemotron_h::load_nemotron_h_gguf_model(
                    checkpoint,
                    metadata,
                    options.weight_residency,
                    options.quantization,
                    stream,
                    weights_stream,
                )?;
            (Model::NemotronH(loaded), eos_token_ids)
        }
        GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
            let (loaded, eos_token_ids) = crate::composition::qwen::load_qwen_gguf_model(
                checkpoint,
                metadata,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            (Model::Qwen(loaded), eos_token_ids)
        }
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
            let (loaded, eos_token_ids) = crate::composition::qwen::vl::load_gguf(
                gguf_file,
                checkpoint,
                metadata,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            let variant = if gguf_architecture == GgufArchitecture::Qwen3VlMoe {
                Model::Qwen3VlMoe(loaded)
            } else {
                Model::Qwen3Vl(loaded)
            };
            (variant, eos_token_ids)
        }
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe | GgufArchitecture::Qwen3Next => {
            let (loaded, eos_token_ids) = crate::composition::qwen::hybrid::load_gguf(
                gguf_file,
                checkpoint,
                metadata,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            let model = if gguf_architecture == GgufArchitecture::Qwen3Next {
                Model::Qwen3Next(loaded)
            } else {
                Model::Qwen35(loaded)
            };
            (model, eos_token_ids)
        }
    };
    Ok(MaterializedGgufModel {
        model,
        #[cfg(feature = "media")]
        processor,
    })
}

pub fn materialize_model_plan(
    plan: ModelPreparationPlan,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    validate_plan_options(&plan, options)?;
    let runtime_state_dtype_bytes =
        inspected_runtime_state_dtype_bytes(plan.inspection().tensors());
    let (artifact, _policy, _route) = plan.into_parts();
    if let Some(topology) = options
        .parallel
        .filter(|topology| !topology.is_replicated())
    {
        let path = match &artifact {
            ModelArtifact::Gguf { path, .. } | ModelArtifact::SafeTensors { path, .. } => path,
        };
        let kind = match &artifact {
            ModelArtifact::Gguf { configuration, .. }
            | ModelArtifact::SafeTensors { configuration, .. } => configuration.kind,
        };
        if topology.pipeline_parallel_size > 1
            || (topology.expert_parallel_size > 1
                && matches!(
                    kind,
                    ModelKind::Gemma4 | ModelKind::MuseGlimmer | ModelKind::Inkling
                ))
            || matches!(
                kind,
                ModelKind::Qwen3Next
                    | ModelKind::Qwen35
                    | ModelKind::Qwen3Vl
                    | ModelKind::Qwen3VlMoe
            )
        {
            let model =
                crate::composition::mlx::distributed::pipeline::load_pipeline_model_with_options(
                    path,
                    options,
                    stream,
                    weights_stream,
                )
                .map(|model| MlxModel::pipeline(model, runtime_state_dtype_bytes))?;
            return attach_artifact_processor(model, &artifact);
        }
        if topology.expert_parallel_size > 1 {
            let model =
                crate::composition::mlx::distributed::expert::load_expert_parallel_model_with_options(
                    path,
                    options,
                    stream,
                    weights_stream,
                )
                .map(|model| MlxModel::expert(model, runtime_state_dtype_bytes))?;
            return attach_artifact_processor(model, &artifact);
        }
        if let ModelArtifact::SafeTensors {
            path,
            configuration,
            ..
        } = &artifact
        {
            let model = materialize_tensor_parallel(
                configuration.kind,
                path,
                options,
                stream,
                weights_stream,
            )
            .map(|model| MlxModel::complete(model, runtime_state_dtype_bytes))?;
            return attach_artifact_processor(model, &artifact);
        }
    }
    match artifact {
        artifact @ ModelArtifact::Gguf { .. } => {
            materialize_gguf_artifact(artifact, options, stream, weights_stream)
                .map(|model| complete_gguf_model(model, runtime_state_dtype_bytes))
        }
        ModelArtifact::SafeTensors {
            path,
            configuration,
            ..
        } => {
            let model =
                materialize_safetensors(configuration.kind, &path, options, stream, weights_stream)
                    .map(|model| MlxModel::complete(model, runtime_state_dtype_bytes))?;
            attach_safetensors_processor(model, &path)
        }
    }
}

fn inspected_runtime_state_dtype_bytes(
    tensors: &eredu_core::checkpoint::TensorCatalog,
) -> std::num::NonZeroU8 {
    // Token embeddings establish the ordinary decoder activation dtype, and
    // MLX caches retain those activations without casting. Encoded embeddings
    // have no portable runtime width, so keep the conservative FP32 fallback.
    let embedding = tensors.descriptors().find(|tensor| {
        tensor.name == "model.embed_tokens.weight"
            || tensor.name.ends_with(".embed_tokens.weight")
            || matches!(
                tensor.name.as_str(),
                "model.llm.embed.weight"
                    | "backbone.embeddings.weight"
                    | "model.embeddings.weight"
                    | "embed.weight"
            )
    });
    let bytes = match embedding.map(|tensor| &tensor.dtype) {
        Some(
            eredu_core::checkpoint::TensorDtype::F16 | eredu_core::checkpoint::TensorDtype::Bf16,
        ) => 2,
        Some(
            eredu_core::checkpoint::TensorDtype::F64
            | eredu_core::checkpoint::TensorDtype::Complex64,
        ) => 8,
        _ => 4,
    };
    std::num::NonZeroU8::new(bytes).expect("supported MLX activation widths are nonzero")
}

#[cfg(test)]
mod runtime_state_dtype_tests {
    use super::inspected_runtime_state_dtype_bytes;
    use eredu_core::checkpoint::{TensorCatalog, TensorDescriptor, TensorDtype};

    fn catalog(name: &str, dtype: TensorDtype) -> TensorCatalog {
        TensorCatalog::new([TensorDescriptor {
            name: name.into(),
            shape: vec![32, 16],
            dtype,
            storage: None,
        }])
        .unwrap()
    }

    #[test]
    fn floating_text_embedding_width_selects_runtime_state_width() {
        for dtype in [TensorDtype::F16, TensorDtype::Bf16] {
            assert_eq!(
                inspected_runtime_state_dtype_bytes(&catalog(
                    "model.language_model.embed_tokens.weight",
                    dtype,
                ))
                .get(),
                2
            );
        }
    }

    #[test]
    fn encoded_embedding_keeps_conservative_fp32_width() {
        assert_eq!(
            inspected_runtime_state_dtype_bytes(&catalog(
                "model.embed_tokens.weight",
                TensorDtype::Encoded("Q4_K".into()),
            ))
            .get(),
            4
        );
    }
}

fn attach_artifact_processor(model: MlxModel, artifact: &ModelArtifact) -> Result<MlxModel, Error> {
    if let ModelArtifact::SafeTensors { path, .. } = artifact {
        return attach_safetensors_processor(model, path);
    }
    Ok(model)
}

fn attach_safetensors_processor(model: MlxModel, path: &Path) -> Result<MlxModel, Error> {
    #[cfg(feature = "media")]
    {
        Ok(model.with_processor(load_processor(path)?))
    }
    #[cfg(not(feature = "media"))]
    {
        let _ = path;
        Ok(model)
    }
}

fn complete_gguf_model(
    materialized: MaterializedGgufModel,
    runtime_state_dtype_bytes: std::num::NonZeroU8,
) -> MlxModel {
    let model = MlxModel::complete(materialized.model, runtime_state_dtype_bytes);
    #[cfg(feature = "media")]
    let model = model.with_processor(materialized.processor);
    model
}

fn materialize_tensor_parallel(
    kind: ModelKind,
    path: &Path,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let topology = options.parallel.ok_or_else(|| {
        Error::Parallel("tensor-parallel materialization requires a topology".into())
    })?;
    if topology.tensor_parallel_size <= 1
        || topology.pipeline_parallel_size != 1
        || topology.expert_parallel_size != 1
    {
        return Err(Error::Parallel(
            "complete Model materialization supports pure tensor parallelism only".into(),
        ));
    }
    if options.weight_residency.expert_cache().is_some() {
        return Err(Error::Parallel(
            "tensor-parallel model materialization does not compose with independent expert caching"
                .into(),
        ));
    }
    if options.quantization.is_some() && kind != ModelKind::DeepSeekV4 {
        return Err(Error::Quantization(format!(
            "load-time quantization is not implemented for tensor-parallel {} materialization",
            kind.model_type_name()
        )));
    }
    let execution = options.weight_residency.layers();
    let build = crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext::new(
        topology,
        eredu_runtime::ShardingPolicy::Require,
    );
    match kind {
        ModelKind::DeepSeekV3 | ModelKind::DeepSeekV4 => Ok(Model::DeepSeek(Box::new(
            crate::composition::deepseek::load_safetensors(
                path,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?,
        ))),
        ModelKind::Gemma4 => Ok(Model::Gemma4(
            crate::composition::gemma4::load_safetensors_tensor_parallel(
                path,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::GptOss => Ok(Model::GptOss(
            crate::composition::gpt_oss::load_gpt_oss_tensor_parallel_model(
                path,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Inkling => Ok(Model::Inkling(
            crate::composition::inkling::load_safetensors_tensor_parallel(
                path,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::KimiLinear => Ok(Model::KimiLinear(
            crate::composition::kimi_linear::load_kimi_linear_tensor_parallel_model(
                path,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Llama => Ok(Model::Llama(
            crate::composition::llama::load_llama_tensor_parallel_model(
                path,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::MuseGlimmer => Ok(Model::MuseGlimmer(
            crate::composition::muse_glimmer::load_safetensors_tensor_parallel(
                path,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Lfm2 => Ok(Model::Lfm2(
            crate::composition::lfm2::load_lfm2_tensor_parallel_model(
                path,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::NemotronH => Ok(Model::NemotronH(
            crate::composition::nemotron_h::load_nemotron_h_tensor_parallel_model(
                path,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen2 | ModelKind::Qwen3 => Ok(Model::Qwen(
            crate::composition::qwen::load_qwen_tensor_parallel_model(
                path,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen3Next => Err(Error::UnsupportedArchitecture(
            "neutral Qwen hybrid tensor-parallel binding is not initialized".into(),
        )),
        ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => Err(Error::UnsupportedArchitecture(
            "neutral Qwen3-VL tensor-parallel binding is not initialized".into(),
        )),
        ModelKind::Qwen35 => Err(Error::UnsupportedArchitecture(
            "neutral Qwen3.5 tensor-parallel binding is not initialized".into(),
        )),
        ModelKind::Moshi => Err(Error::UnsupportedArchitecture(
            "Moshi-family models do not use the text Model tensor-parallel session".into(),
        )),
    }
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
    super::structural::validate_inspected_preparation(plan.inspection(), plan.policy())?;
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
    let metadata = crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(&checkpoint);
    let architecture = configuration.gguf_architecture.ok_or_else(|| {
        Error::UnsupportedArchitecture("backend-neutral GGUF plan omitted its architecture".into())
    })?;
    structural::validate_gguf(architecture, &checkpoint, &metadata, options)
        .into_loader_result()?;
    validate_gguf_quantization_source(&checkpoint, &metadata, options.quantization)?;
    if options
        .parallel
        .is_some_and(|topology| !topology.is_replicated())
    {
        let (model, _eos_token_ids) = materialize_gguf_tensor_parallel(
            &path,
            &checkpoint,
            &metadata,
            architecture,
            options,
            stream,
            weights_stream,
        )?;
        #[cfg(feature = "media")]
        let processor = match architecture {
            GgufArchitecture::Inkling
                if crate::composition::mlx::artifact::find_sibling_mmproj(&path, "inkling")?
                    .is_some() =>
            {
                Some(ModelProcessor::load_inkling_gguf(&metadata)?)
            }
            #[cfg(any(feature = "image", feature = "audio"))]
            GgufArchitecture::Gemma4 => {
                crate::composition::mlx::artifact::find_sibling_mmproj(&path, "gemma4")?
                    .map(GgufCheckpoint::open)
                    .transpose()?
                    .as_ref()
                    .map(|checkpoint| {
                        ModelProcessor::load_gemma4_gguf(
                            &metadata,
                            &crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(
                                checkpoint,
                            ),
                        )
                    })
                    .transpose()?
            }
            #[cfg(feature = "image")]
            GgufArchitecture::MuseGlimmer => {
                crate::composition::mlx::artifact::find_sibling_mmproj(&path, "muse-glimmer")?
                    .map(GgufCheckpoint::open)
                    .transpose()?
                    .as_ref()
                    .map(|checkpoint| {
                        ModelProcessor::load_muse_glimmer_gguf(
                            &crate::backend::mlx::runtime::checkpoint::load::gguf_metadata(
                                checkpoint,
                            ),
                        )
                    })
                    .transpose()?
            }
            #[cfg(feature = "image")]
            GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe
                if crate::composition::mlx::artifact::find_sibling_mmproj(&path, "qwen35")?
                    .is_some() =>
            {
                ModelProcessor::load_qwen(path.parent().unwrap_or_else(|| Path::new(".")))?
            }
            _ => None,
        };
        return Ok(MaterializedGgufModel {
            model,
            #[cfg(feature = "media")]
            processor,
        });
    }
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

fn materialize_gguf_tensor_parallel(
    gguf_path: &Path,
    checkpoint: &GgufCheckpoint,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    architecture: GgufArchitecture,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Model, Vec<u32>), Error> {
    let topology = options.parallel.ok_or_else(|| {
        Error::Parallel("tensor-parallel GGUF materialization requires a topology".into())
    })?;
    if options.quantization.is_some() && architecture != GgufArchitecture::DeepSeek4 {
        return Err(Error::Quantization(format!(
            "load-time quantization is not implemented for tensor-parallel {} GGUF materialization",
            architecture.metadata_name()
        )));
    }
    let residency = options.weight_residency.layers();
    let build = crate::backend::mlx::runtime::distributed::parallel::ParallelBuildContext::new(
        topology,
        eredu_runtime::ShardingPolicy::Require,
    );
    match architecture {
        GgufArchitecture::KimiLinear => {
            let (model, eos) =
                crate::composition::kimi_linear::load_kimi_linear_gguf_tensor_parallel_model(
                    checkpoint,
                    metadata,
                    residency,
                    build,
                    stream,
                    weights_stream,
                )?;
            Ok((Model::KimiLinear(model), eos))
        }
        GgufArchitecture::DeepSeek2 | GgufArchitecture::DeepSeek4 => {
            let (model, eos) = crate::composition::deepseek::load_gguf(
                checkpoint,
                metadata,
                architecture == GgufArchitecture::DeepSeek4,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Ok((Model::DeepSeek(Box::new(model)), eos))
        }
        GgufArchitecture::GptOss => {
            let (model, eos) =
                crate::composition::gpt_oss::load_gpt_oss_gguf_tensor_parallel_model(
                    checkpoint,
                    metadata,
                    residency,
                    build,
                    stream,
                    weights_stream,
                )?;
            Ok((Model::GptOss(model), eos))
        }
        GgufArchitecture::Inkling => {
            let (model, eos) = crate::composition::inkling::load_gguf_tensor_parallel(
                gguf_path,
                checkpoint,
                metadata,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok((Model::Inkling(model), eos))
        }
        GgufArchitecture::Gemma4 => {
            let (model, eos) = crate::composition::gemma4::load_gguf_tensor_parallel(
                gguf_path,
                checkpoint,
                metadata,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok((Model::Gemma4(model), eos))
        }
        GgufArchitecture::Llama | GgufArchitecture::Mistral => {
            let (model, eos) = crate::composition::llama::load_llama_gguf_tensor_parallel_model(
                checkpoint,
                metadata,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok((Model::Llama(model), eos))
        }
        GgufArchitecture::MuseGlimmer => {
            let (model, eos) = crate::composition::muse_glimmer::load_gguf_tensor_parallel(
                gguf_path,
                checkpoint,
                metadata,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok((Model::MuseGlimmer(model), eos))
        }
        GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
            let (model, eos) = crate::composition::lfm2::load_lfm2_gguf_tensor_parallel_model(
                checkpoint,
                metadata,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok((Model::Lfm2(model), eos))
        }
        GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
            let (model, eos) =
                crate::composition::nemotron_h::load_nemotron_h_gguf_tensor_parallel_model(
                    checkpoint,
                    metadata,
                    residency,
                    build,
                    stream,
                    weights_stream,
                )?;
            Ok((Model::NemotronH(model), eos))
        }
        GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
            let (model, eos) = crate::composition::qwen::load_qwen_gguf_tensor_parallel_model(
                checkpoint,
                metadata,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok((Model::Qwen(model), eos))
        }
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
            Err(Error::UnsupportedArchitecture(
                "neutral Qwen3-VL GGUF tensor-parallel binding is not initialized".into(),
            ))
        }
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe | GgufArchitecture::Qwen3Next => {
            Err(Error::UnsupportedArchitecture(
                "neutral Qwen hybrid GGUF tensor-parallel binding is not initialized".into(),
            ))
        }
    }
}

pub fn validate_gguf_quantization_source<
    S: crate::backend::mlx::runtime::checkpoint::load::GgufTensorNames,
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
    if let (Some(expert_cache), Some(non_expert)) = (
        options.weight_residency.expert_cache(),
        options.weight_residency.non_experts(),
    ) {
        return match kind {
            ModelKind::KimiLinear => Ok(Model::KimiLinear(
                crate::composition::kimi_linear::load_kimi_linear_model(
                    model_dir,
                    eredu_runtime::WeightResidency::with_expert_cache(
                        non_expert,
                        expert_cache,
                    ),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::DeepSeekV3 | ModelKind::DeepSeekV4 => {
                Ok(Model::DeepSeek(Box::new(
                    crate::composition::deepseek::load_safetensors(
                        model_dir,
                        eredu_runtime::WeightResidency::with_expert_cache(
                            non_expert,
                            expert_cache,
                        ),
                        options.quantization,
                        stream,
                        weights_stream,
                    )?,
                )))
            }
            ModelKind::GptOss => Ok(Model::GptOss(
                crate::composition::gpt_oss::load_gpt_oss_expert_cache_model(
                    model_dir, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Gemma4 => Ok(Model::Gemma4(
                crate::composition::gemma4::load_safetensors(
                    model_dir,
                    eredu_runtime::WeightResidency::with_expert_cache(
                        non_expert,
                        expert_cache,
                    ),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Inkling => Ok(Model::Inkling(
                crate::composition::inkling::load_safetensors(
                    model_dir,
                    eredu_runtime::WeightResidency::with_expert_cache(
                        non_expert,
                        expert_cache,
                    ),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Lfm2 => Ok(Model::Lfm2(
                crate::composition::lfm2::load_lfm2_model(
                    model_dir,
                    eredu_runtime::WeightResidency::with_expert_cache(non_expert, expert_cache),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::MuseGlimmer => Ok(Model::MuseGlimmer(
                crate::composition::muse_glimmer::load_safetensors(
                    model_dir,
                    eredu_runtime::WeightResidency::with_expert_cache(
                        non_expert,
                        expert_cache,
                    ),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::NemotronH => Ok(Model::NemotronH(
                crate::composition::nemotron_h::load_nemotron_h_model(
                    model_dir,
                    eredu_runtime::WeightResidency::with_expert_cache(non_expert, expert_cache),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen2 => Err(Error::UnsupportedArchitecture(
                "Qwen2 is dense and does not support sparse expert-cache residency".into(),
            )),
            ModelKind::Qwen3 => Ok(Model::Qwen(
                crate::composition::qwen::load_qwen_safetensors_mlx(
                    model_dir,
                    eredu_runtime::WeightResidency::with_expert_cache(non_expert, expert_cache),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3Next => Ok(Model::Qwen3Next(
                crate::composition::qwen::hybrid::load_safetensors_with_residency(
                    model_dir,
                    eredu_runtime::WeightResidency::with_expert_cache(
                        non_expert,
                        expert_cache,
                    ),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoe(
                crate::composition::qwen::vl::load_safetensors_with_residency(
                    model_dir,
                    eredu_runtime::WeightResidency::with_expert_cache(
                        non_expert,
                        expert_cache,
                    ),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen35 => Ok(Model::Qwen35(
                crate::composition::qwen::hybrid::load_safetensors_with_residency(
                    model_dir,
                    eredu_runtime::WeightResidency::with_expert_cache(
                        non_expert,
                        expert_cache,
                    ),
                    options.quantization,
                    stream,
                    weights_stream,
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
    }
    match kind {
        ModelKind::DeepSeekV3 | ModelKind::DeepSeekV4 => Ok(Model::DeepSeek(Box::new(
            crate::composition::deepseek::load_safetensors(
                model_dir,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?,
        ))),
        ModelKind::Gemma4 => Ok(Model::Gemma4(crate::composition::gemma4::load_safetensors(
            model_dir,
            eredu_runtime::WeightResidency::with_layers(execution),
            options.quantization,
            stream,
            weights_stream,
        )?)),
        ModelKind::Inkling => Ok(Model::Inkling(
            crate::composition::inkling::load_safetensors(
                model_dir,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::KimiLinear => Ok(Model::KimiLinear(
            crate::composition::kimi_linear::load_kimi_linear_model(
                model_dir,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Llama => Ok(Model::Llama(
            crate::composition::llama::load_llama_safetensors_mlx(
                model_dir,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::MuseGlimmer => Ok(Model::MuseGlimmer(
            crate::composition::muse_glimmer::load_safetensors(
                model_dir,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen2 | ModelKind::Qwen3 => Ok(Model::Qwen(
            crate::composition::qwen::load_qwen_safetensors_mlx(
                model_dir,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::GptOss => Ok(Model::GptOss(
            crate::composition::gpt_oss::load_gpt_oss_layerwise_model(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Lfm2 => Ok(Model::Lfm2(crate::composition::lfm2::load_lfm2_model(
            model_dir,
            eredu_runtime::WeightResidency::with_layers(execution),
            options.quantization,
            stream,
            weights_stream,
        )?)),
        ModelKind::NemotronH => Ok(Model::NemotronH(
            crate::composition::nemotron_h::load_nemotron_h_model(
                model_dir,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen3Next => Ok(Model::Qwen3Next(
            crate::composition::qwen::hybrid::load_safetensors(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen3Vl => Ok(Model::Qwen3Vl(
            crate::composition::qwen::vl::load_safetensors(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoe(
            crate::composition::qwen::vl::load_safetensors(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen35 => Ok(Model::Qwen35(
            crate::composition::qwen::hybrid::load_safetensors(
                model_dir,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Moshi => Err(Error::UnsupportedArchitecture(
            "Moshi-family bounded layer residency is selected through the realtime loader".into(),
        )),
    }
}
