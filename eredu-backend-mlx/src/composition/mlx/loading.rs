//! MLX checkpoint materialization after backend-neutral planning.

use eredu_checkpoint::WeightQuantization;

use eredu_architectures::{GgufArchitecture, ModelKind};
use eredu_core::{ModelArtifact, ModelPreparationPlan};
use safemlx::ops::GgufCheckpoint;
use safemlx::{ops::GgufMetadataValue, Stream};

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
    backend::error::Error,
    backend::{MlxModel, ModelLoadOptions},
    composition::mlx::{structural, Model},
};

/// MLX arrays/modules plus backend-owned preprocessing from one GGUF artifact.
struct MaterializedGgufModel {
    model: Model,
    #[cfg(feature = "media")]
    processor: Option<ModelProcessor>,
}

fn materialize_gguf_model(
    source: &structural::AdmittedGguf,
    projector: Option<&GgufCheckpoint>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let checkpoint = source.checkpoint();
    let metadata = source.metadata();
    let kind = source.architecture().model_kind();
    let (model, _architecture_eos_token_ids) = match source.architecture() {
        GgufArchitecture::KimiLinear => {
            let (loaded, eos_token_ids) =
                crate::composition::kimi_linear::load_kimi_linear_gguf_model(
                    source,
                    options.weight_residency,
                    options.quantization,
                    stream,
                    weights_stream,
                )?;
            (Model::KimiLinear(kind, loaded), eos_token_ids)
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
            (Model::DeepSeek(kind, Box::new(loaded)), eos_token_ids)
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
            (Model::DeepSeek(kind, Box::new(loaded)), eos_token_ids)
        }
        GgufArchitecture::GptOss => {
            let (loaded, eos_token_ids) =
                crate::composition::gpt_oss::load_gpt_oss_gguf_layerwise_model(
                    source,
                    options.weight_residency,
                    options.quantization,
                    stream,
                    weights_stream,
                )?;
            (Model::GptOss(kind, loaded), eos_token_ids)
        }
        GgufArchitecture::Inkling => {
            if options.quantization.is_some() {
                return Err(Error::UnsupportedArchitecture(
                    "load-time Inkling quantization is not bound on the neutral loader".into(),
                ));
            }
            let loaded = crate::composition::inkling::load_gguf(
                checkpoint,
                projector,
                metadata,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            let eos_token_ids = gguf_eos_token_ids(metadata)?;
            (Model::Inkling(kind, loaded), eos_token_ids)
        }
        GgufArchitecture::Gemma4 => {
            if options.quantization.is_some() {
                return Err(Error::UnsupportedArchitecture(
                    "load-time Gemma 4 quantization is not bound on the neutral loader".into(),
                ));
            }
            let loaded = crate::composition::gemma4::load_gguf(
                checkpoint,
                projector,
                metadata,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            (Model::Gemma4(kind, loaded), gguf_eos_token_ids(metadata)?)
        }
        GgufArchitecture::Llama | GgufArchitecture::Mistral => {
            let (loaded, eos_token_ids) = crate::composition::llama::load_llama_gguf_model(
                source,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            (Model::Llama(kind, loaded), eos_token_ids)
        }
        GgufArchitecture::MuseGlimmer => {
            let projector = projector.ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Muse-Glimmer preparation omitted its required media projector".into(),
                )
            })?;
            if options.quantization.is_some() {
                return Err(Error::UnsupportedArchitecture(
                    "load-time Muse-Glimmer quantization is not bound on the neutral loader".into(),
                ));
            }
            let loaded = crate::composition::muse_glimmer::load_gguf(
                checkpoint,
                projector,
                metadata,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            let eos_token_ids = gguf_eos_token_ids(metadata)?;
            (Model::MuseGlimmer(kind, loaded), eos_token_ids)
        }
        GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
            let (loaded, eos_token_ids) = crate::composition::lfm2::load_lfm2_gguf_model(
                source,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            (Model::Lfm2(kind, loaded), eos_token_ids)
        }
        GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
            let (loaded, eos_token_ids) =
                crate::composition::nemotron_h::load_nemotron_h_gguf_model(
                    source,
                    options.weight_residency,
                    options.quantization,
                    stream,
                    weights_stream,
                )?;
            (Model::NemotronH(kind, loaded), eos_token_ids)
        }
        GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
            let (loaded, eos_token_ids) = crate::composition::qwen::load_qwen_gguf_model(
                source,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            (Model::Qwen(kind, loaded), eos_token_ids)
        }
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
            let projector = projector.ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Qwen3-VL preparation omitted its required media projector".into(),
                )
            })?;
            let (loaded, eos_token_ids) = crate::composition::qwen::vl::load_gguf(
                source.architecture(),
                checkpoint,
                projector,
                metadata,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            let variant = if source.architecture() == GgufArchitecture::Qwen3VlMoe {
                Model::Qwen3VlMoe(kind, loaded)
            } else {
                Model::Qwen3Vl(kind, loaded)
            };
            (variant, eos_token_ids)
        }
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe | GgufArchitecture::Qwen3Next => {
            let (loaded, eos_token_ids) = crate::composition::qwen::hybrid::load_gguf(
                source,
                projector,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            let model = if source.architecture() == GgufArchitecture::Qwen3Next {
                Model::Qwen3Next(kind, loaded)
            } else {
                Model::Qwen35(kind, loaded)
            };
            (model, eos_token_ids)
        }
    };
    Ok(model)
}

pub fn materialize_model_plan(
    plan: ModelPreparationPlan,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    validate_plan_options(&plan, options)?;
    let runtime_state_dtype_bytes = inspected_runtime_state_dtype_bytes(plan.inspection())?;
    if let Some(topology) = options
        .parallel
        .filter(|topology| !topology.is_replicated())
    {
        let kind = ModelKind::resolve_family(&plan.inspection().configuration().family)?;
        if requires_distributed_stage_loader(kind, topology) {
            #[cfg(feature = "media")]
            let processor = match plan.inspection().format() {
                eredu_core::ArtifactFormat::SafeTensors => {
                    let config =
                        plan.inspection()
                            .configuration()
                            .json
                            .as_ref()
                            .ok_or_else(|| {
                                Error::UnsupportedArchitecture(
                                "SafeTensors preparation plan omitted normalized JSON configuration"
                                    .into(),
                            )
                            })?;
                    load_processor(kind, plan.inspection().path(), config)?
                }
                eredu_core::ArtifactFormat::Gguf => {
                    load_inspected_gguf_processor(plan.inspection())?
                }
            };
            let model =
                crate::composition::mlx::distributed::pipeline::load_pipeline_model_with_options(
                    plan,
                    options,
                    stream,
                    weights_stream,
                )
                .map(|model| MlxModel::pipeline(model, runtime_state_dtype_bytes))?;
            #[cfg(feature = "media")]
            let model = model.with_processor(processor);
            return Ok(model);
        }
        let (artifact, _policy, _route) = plan.into_parts();
        return match artifact {
            artifact @ ModelArtifact::Gguf { .. } => {
                materialize_gguf_artifact(artifact, options, stream, weights_stream)
                    .map(|model| complete_gguf_model(model, runtime_state_dtype_bytes))
            }
            ModelArtifact::SafeTensors {
                path,
                configuration,
                tensors,
            } => {
                let prepared = super::artifact::PreparedSafetensorsArtifact::open(
                    path,
                    configuration,
                    tensors,
                    options.weight_residency.max_mapped_shards(),
                )?;
                let model = materialize_tensor_parallel(
                    ModelKind::resolve_family(&prepared.configuration().family)?,
                    &prepared,
                    options,
                    stream,
                    weights_stream,
                )
                .map(|model| MlxModel::complete(model, runtime_state_dtype_bytes))?;
                attach_safetensors_processor(model, &prepared)
            }
        };
    }
    let (artifact, _policy, _route) = plan.into_parts();
    match artifact {
        artifact @ ModelArtifact::Gguf { .. } => {
            materialize_gguf_artifact(artifact, options, stream, weights_stream)
                .map(|model| complete_gguf_model(model, runtime_state_dtype_bytes))
        }
        ModelArtifact::SafeTensors {
            path,
            configuration,
            tensors,
        } => {
            let kind = ModelKind::resolve_family(&configuration.family)?;
            let prepared = super::artifact::PreparedSafetensorsArtifact::open(
                path,
                configuration,
                tensors,
                options.weight_residency.max_mapped_shards(),
            )?;
            let model = materialize_safetensors(kind, &prepared, options, stream, weights_stream)
                .map(|model| MlxModel::complete(model, runtime_state_dtype_bytes))?;
            attach_safetensors_processor(model, &prepared)
        }
    }
}

fn requires_distributed_stage_loader(
    kind: ModelKind,
    topology: crate::backend::MlxParallelContext,
) -> bool {
    topology.pipeline_parallel_size > 1
        || topology.expert_parallel_size > 1
        || matches!(
            kind,
            ModelKind::DeepSeekV3
                | ModelKind::DeepSeekV4
                | ModelKind::Qwen3Next
                | ModelKind::Qwen35
                | ModelKind::Qwen3Vl
                | ModelKind::Qwen3VlMoe
        )
}

fn inspected_runtime_state_dtype_bytes(
    inspection: &eredu_core::ArtifactInspection,
) -> Result<std::num::NonZeroU8, Error> {
    if inspection.format() == eredu_core::ArtifactFormat::Gguf {
        // MLX native GGUF embeddings dequantize to Float32. Their catalog dtype
        // describes packed storage rather than the resulting activation.
        return Ok(std::num::NonZeroU8::new(4).expect("Float32 width is nonzero"));
    }
    let configuration = inspection.configuration();
    let config = configuration.json.as_ref().ok_or_else(|| {
        Error::UnsupportedArchitecture(
            "SafeTensors inspection omitted normalized JSON configuration".into(),
        )
    })?;
    let source = eredu_architectures::preparation::safetensors_runtime_state_dtype_source(
        ModelKind::resolve_family(&configuration.family)?,
        config,
        inspection.tensors(),
    )
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    mlx_runtime_state_dtype_bytes(source.dtype()).map_err(|dtype| {
        Error::UnsupportedArchitecture(format!(
            "runtime-state dtype source {:?} has unsupported MLX activation dtype {dtype:?}",
            source.checkpoint_tensor()
        ))
    })
}

fn mlx_runtime_state_dtype_bytes(
    dtype: &eredu_core::checkpoint::TensorDtype,
) -> Result<std::num::NonZeroU8, eredu_core::checkpoint::TensorDtype> {
    use eredu_core::checkpoint::TensorDtype;

    let bytes = match dtype {
        TensorDtype::F16 | TensorDtype::Bf16 => 2,
        TensorDtype::F32 => 4,
        TensorDtype::F64 | TensorDtype::Complex64 => 8,
        // MLX materializes supported packed SafeTensors embeddings as Float32
        // activations. These cases are reached only after the architecture
        // schema resolved the exact embedding parameter; they are not a
        // fallback for an unknown checkpoint name.
        TensorDtype::U32 | TensorDtype::Encoded(_) => 4,
        dtype => return Err(dtype.clone()),
    };
    Ok(std::num::NonZeroU8::new(bytes).expect("supported MLX activation widths are nonzero"))
}

#[cfg(test)]
mod runtime_state_dtype_tests {
    #[cfg(feature = "image")]
    use super::load_gguf_processor;
    use super::{
        mlx_runtime_state_dtype_bytes, reject_complete_tensor_parallel_quantization,
        requires_distributed_stage_loader,
    };
    use crate::backend::{DeviceAssignment, MlxParallelContext};
    #[cfg(feature = "image")]
    use eredu_architectures::GgufArchitecture;
    use eredu_architectures::ModelKind;
    use eredu_checkpoint::WeightQuantization;
    use eredu_core::checkpoint::TensorDtype;
    use safemlx::DeviceType;

    #[test]
    fn deepseek_pure_tp_uses_distributed_stage_loader() {
        let topology =
            MlxParallelContext::for_rank(0, 2, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        for kind in [ModelKind::DeepSeekV3, ModelKind::DeepSeekV4] {
            assert!(requires_distributed_stage_loader(kind, topology));
        }
    }

    #[test]
    fn complete_tensor_parallel_loader_rejects_unbound_quantization() {
        reject_complete_tensor_parallel_quantization(None, "deepseek4").unwrap();
        let error = reject_complete_tensor_parallel_quantization(
            Some(WeightQuantization::MxFp4),
            "deepseek4",
        )
        .unwrap_err();
        assert!(matches!(
            error,
            crate::backend::error::Error::Quantization(message)
                if message.contains("deepseek4")
        ));
    }

    #[test]
    fn specialized_qwen_tp_uses_distributed_stage_loader() {
        let topology =
            MlxParallelContext::for_rank(0, 2, 1, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        for kind in [
            ModelKind::Qwen3Next,
            ModelKind::Qwen35,
            ModelKind::Qwen3Vl,
            ModelKind::Qwen3VlMoe,
        ] {
            assert!(requires_distributed_stage_loader(kind, topology));
        }
        assert!(!requires_distributed_stage_loader(
            ModelKind::Qwen3,
            topology
        ));
    }

    #[test]
    fn expert_parallel_topology_unconditionally_uses_distributed_stage_loader() {
        let topology =
            MlxParallelContext::for_rank(0, 1, 1, 2, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        assert!(requires_distributed_stage_loader(
            ModelKind::Llama,
            topology
        ));
    }

    #[cfg(feature = "image")]
    #[test]
    fn shared_gguf_processor_loader_covers_qwen_vl_and_qwen35() {
        let directory = tempfile::tempdir().unwrap();
        let model_path = directory.path().join("model.gguf");
        std::fs::write(
            directory.path().join("config.json"),
            br#"{"vision_start_token_id":44,"vision_end_token_id":45}"#,
        )
        .unwrap();
        std::fs::write(
            directory.path().join("preprocessor_config.json"),
            br#"{
                "size":{"shortest_edge":16,"longest_edge":16},
                "patch_size":2,"temporal_patch_size":2,"merge_size":2,
                "image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0],
                "min_frames":1,"max_frames":8
            }"#,
        )
        .unwrap();
        let metadata = std::collections::HashMap::new();

        for architecture in [
            GgufArchitecture::Qwen3Vl,
            GgufArchitecture::Qwen3VlMoe,
            GgufArchitecture::Qwen35,
            GgufArchitecture::Qwen35Moe,
        ] {
            assert!(
                load_gguf_processor(&model_path, architecture, &metadata, Some(&metadata),)
                    .unwrap()
                    .is_some()
            );
        }
        assert!(
            load_gguf_processor(&model_path, GgufArchitecture::Qwen35, &metadata, None,)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn resolved_floating_dtype_selects_runtime_state_width() {
        for (dtype, bytes) in [
            (TensorDtype::F16, 2),
            (TensorDtype::Bf16, 2),
            (TensorDtype::F32, 4),
            (TensorDtype::F64, 8),
        ] {
            assert_eq!(mlx_runtime_state_dtype_bytes(&dtype).unwrap().get(), bytes);
        }
    }

    #[test]
    fn packed_embedding_dtype_uses_known_mlx_materialization_width() {
        assert_eq!(
            mlx_runtime_state_dtype_bytes(&TensorDtype::Encoded("F8_E4M3".into()))
                .unwrap()
                .get(),
            4
        );
        assert_eq!(
            mlx_runtime_state_dtype_bytes(&TensorDtype::U32)
                .unwrap()
                .get(),
            4
        );
    }

    #[test]
    fn invalid_activation_dtype_does_not_silently_default() {
        assert_eq!(
            mlx_runtime_state_dtype_bytes(&TensorDtype::U8),
            Err(TensorDtype::U8)
        );
    }
}

fn attach_safetensors_processor(
    model: MlxModel,
    artifact: &super::artifact::PreparedSafetensorsArtifact,
) -> Result<MlxModel, Error> {
    #[cfg(feature = "media")]
    {
        let kind = ModelKind::resolve_family(&artifact.configuration().family)?;
        Ok(model.with_processor(load_processor(kind, artifact.path(), artifact.config()?)?))
    }
    #[cfg(not(feature = "media"))]
    {
        let _ = artifact;
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

#[cfg(feature = "media")]
fn load_gguf_processor(
    _model_path: &std::path::Path,
    architecture: GgufArchitecture,
    model_metadata: &std::collections::HashMap<String, GgufMetadataValue>,
    projector_metadata: Option<&std::collections::HashMap<String, GgufMetadataValue>>,
) -> Result<Option<ModelProcessor>, Error> {
    match architecture {
        GgufArchitecture::Inkling if projector_metadata.is_some() => {
            ModelProcessor::load_inkling_gguf(model_metadata).map(Some)
        }
        #[cfg(any(feature = "image", feature = "audio"))]
        GgufArchitecture::Gemma4 => projector_metadata
            .map(|metadata| ModelProcessor::load_gemma4_gguf(model_metadata, metadata))
            .transpose(),
        #[cfg(feature = "image")]
        GgufArchitecture::MuseGlimmer => projector_metadata
            .map(ModelProcessor::load_muse_glimmer_gguf)
            .transpose(),
        #[cfg(feature = "image")]
        GgufArchitecture::Qwen3Vl
        | GgufArchitecture::Qwen3VlMoe
        | GgufArchitecture::Qwen35
        | GgufArchitecture::Qwen35Moe
            if projector_metadata.is_some() =>
        {
            ModelProcessor::load_qwen_directory(
                _model_path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new(".")),
            )
        }
        _ => Ok(None),
    }
}

#[cfg(feature = "media")]
fn load_inspected_gguf_processor(
    inspection: &eredu_core::ArtifactInspection,
) -> Result<Option<ModelProcessor>, Error> {
    let validated = inspection.validated_gguf().ok_or_else(|| {
        Error::UnsupportedArchitecture(
            "GGUF preparation plan omitted its validated checkpoint".into(),
        )
    })?;
    let checkpoint = GgufCheckpoint::from_portable(validated.checkpoint().clone());
    let model_metadata = crate::backend::runtime::checkpoint::load::gguf_metadata(&checkpoint);
    let projector_metadata = validated
        .companion(&eredu_core::GgufCompanionRole::MediaProjector)
        .map(|companion| {
            let checkpoint = GgufCheckpoint::from_portable(companion.checkpoint().clone());
            crate::backend::runtime::checkpoint::load::gguf_metadata(&checkpoint)
        });
    load_gguf_processor(
        inspection.path(),
        GgufArchitecture::resolve(&inspection.configuration().declared_model_type)?,
        &model_metadata,
        projector_metadata.as_ref(),
    )
}

fn materialize_tensor_parallel(
    kind: ModelKind,
    artifact: &super::artifact::PreparedSafetensorsArtifact,
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
    reject_complete_tensor_parallel_quantization(options.quantization, kind.canonical_name())?;
    let execution = options.weight_residency.layers();
    let build = crate::backend::runtime::distributed::parallel::ParallelBuildContext::new(
        topology,
        eredu_runtime::ShardingPolicy::Require,
    );
    match kind {
        ModelKind::DeepSeekV3 | ModelKind::DeepSeekV4 => Err(Error::Parallel(
            "DeepSeek tensor parallelism requires distributed-stage materialization".into(),
        )),
        ModelKind::Gemma4 => Ok(Model::Gemma4(
            kind,
            crate::composition::gemma4::load_safetensors_tensor_parallel(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::GptOss => Ok(Model::GptOss(
            kind,
            crate::composition::gpt_oss::load_gpt_oss_tensor_parallel_model(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Inkling => Ok(Model::Inkling(
            kind,
            crate::composition::inkling::load_safetensors_tensor_parallel(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::KimiLinear => Ok(Model::KimiLinear(
            kind,
            crate::composition::kimi_linear::load_kimi_linear_tensor_parallel_model(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Llama => Ok(Model::Llama(
            kind,
            crate::composition::llama::load_llama_tensor_parallel_model(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::MuseGlimmer => Ok(Model::MuseGlimmer(
            kind,
            crate::composition::muse_glimmer::load_safetensors_tensor_parallel(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Lfm2 => Ok(Model::Lfm2(
            kind,
            crate::composition::lfm2::load_lfm2_tensor_parallel_model(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::NemotronH => Ok(Model::NemotronH(
            kind,
            crate::composition::nemotron_h::load_nemotron_h_tensor_parallel_model(
                artifact,
                execution,
                build,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen2 | ModelKind::Qwen3 => Ok(Model::Qwen(
            kind,
            crate::composition::qwen::load_qwen_tensor_parallel_model(
                artifact,
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

pub(super) fn validate_plan_options(
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
        path: _model_path,
        configuration,
        checkpoint,
        mut companions,
        ..
    } = artifact
    else {
        return Err(Error::UnsupportedArchitecture(
            "MLX GGUF materializer received a SafeTensors plan".into(),
        ));
    };
    let checkpoint = safemlx::ops::GgufCheckpoint::from_portable(checkpoint);
    let projector = companions
        .remove(&eredu_core::GgufCompanionRole::MediaProjector)
        .map(|companion| GgufCheckpoint::from_portable(companion.checkpoint().clone()));
    let metadata = crate::backend::runtime::checkpoint::load::gguf_metadata(&checkpoint);
    let architecture = GgufArchitecture::resolve(&configuration.declared_model_type)?;
    let source = structural::admit_gguf(architecture, checkpoint, metadata, options)?;
    let checkpoint = source.checkpoint();
    let metadata = source.metadata();
    validate_gguf_quantization_source(checkpoint, metadata, options.quantization)?;
    #[cfg(feature = "media")]
    let processor = {
        let projector_metadata = projector
            .as_ref()
            .map(|checkpoint| crate::backend::runtime::checkpoint::load::gguf_metadata(checkpoint));
        load_gguf_processor(
            &_model_path,
            architecture,
            metadata,
            projector_metadata.as_ref(),
        )?
    };
    if options
        .parallel
        .is_some_and(|topology| !topology.is_replicated())
    {
        let (model, _eos_token_ids) = materialize_gguf_tensor_parallel(
            &source,
            projector.as_ref(),
            options,
            stream,
            weights_stream,
        )?;
        return Ok(MaterializedGgufModel {
            model,
            #[cfg(feature = "media")]
            processor,
        });
    }
    let model =
        materialize_gguf_model(&source, projector.as_ref(), options, stream, weights_stream)?;
    Ok(MaterializedGgufModel {
        model,
        #[cfg(feature = "media")]
        processor,
    })
}

fn materialize_gguf_tensor_parallel(
    source: &structural::AdmittedGguf,
    projector: Option<&GgufCheckpoint>,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<(Model, Vec<u32>), Error> {
    let checkpoint = source.checkpoint();
    let metadata = source.metadata();
    let architecture = source.architecture();
    let kind = architecture.model_kind();
    let topology = options.parallel.ok_or_else(|| {
        Error::Parallel("tensor-parallel GGUF materialization requires a topology".into())
    })?;
    reject_complete_tensor_parallel_quantization(
        options.quantization,
        architecture.metadata_name(),
    )?;
    let residency = options.weight_residency.layers();
    let build = crate::backend::runtime::distributed::parallel::ParallelBuildContext::new(
        topology,
        eredu_runtime::ShardingPolicy::Require,
    );
    match architecture {
        GgufArchitecture::KimiLinear => {
            let (model, eos) =
                crate::composition::kimi_linear::load_kimi_linear_gguf_tensor_parallel_model(
                    source,
                    residency,
                    build,
                    stream,
                    weights_stream,
                )?;
            Ok((Model::KimiLinear(kind, model), eos))
        }
        GgufArchitecture::DeepSeek2 | GgufArchitecture::DeepSeek4 => Err(Error::Parallel(
            "DeepSeek GGUF tensor parallelism requires distributed-stage materialization".into(),
        )),
        GgufArchitecture::GptOss => {
            let (model, eos) =
                crate::composition::gpt_oss::load_gpt_oss_gguf_tensor_parallel_model(
                    source,
                    residency,
                    build,
                    stream,
                    weights_stream,
                )?;
            Ok((Model::GptOss(kind, model), eos))
        }
        GgufArchitecture::Inkling => {
            let (model, eos) = crate::composition::inkling::load_gguf_tensor_parallel(
                checkpoint,
                projector,
                metadata,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok((Model::Inkling(kind, model), eos))
        }
        GgufArchitecture::Gemma4 => {
            let (model, eos) = crate::composition::gemma4::load_gguf_tensor_parallel(
                checkpoint,
                projector,
                metadata,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok((Model::Gemma4(kind, model), eos))
        }
        GgufArchitecture::Llama | GgufArchitecture::Mistral => {
            let (model, eos) = crate::composition::llama::load_llama_gguf_tensor_parallel_model(
                source,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok((Model::Llama(kind, model), eos))
        }
        GgufArchitecture::MuseGlimmer => {
            let projector = projector.ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Muse-Glimmer preparation omitted its required media projector".into(),
                )
            })?;
            let (model, eos) = crate::composition::muse_glimmer::load_gguf_tensor_parallel(
                checkpoint,
                projector,
                metadata,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok((Model::MuseGlimmer(kind, model), eos))
        }
        GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
            let (model, eos) = crate::composition::lfm2::load_lfm2_gguf_tensor_parallel_model(
                source,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok((Model::Lfm2(kind, model), eos))
        }
        GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
            let (model, eos) =
                crate::composition::nemotron_h::load_nemotron_h_gguf_tensor_parallel_model(
                    source,
                    residency,
                    build,
                    stream,
                    weights_stream,
                )?;
            Ok((Model::NemotronH(kind, model), eos))
        }
        GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
            let (model, eos) = crate::composition::qwen::load_qwen_gguf_tensor_parallel_model(
                source,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok((Model::Qwen(kind, model), eos))
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

fn reject_complete_tensor_parallel_quantization(
    quantization: Option<WeightQuantization>,
    architecture: &str,
) -> Result<(), Error> {
    if quantization.is_some() {
        return Err(Error::Quantization(format!(
            "load-time quantization is not implemented for complete tensor-parallel {architecture} materialization"
        )));
    }
    Ok(())
}

pub fn validate_gguf_quantization_source<
    S: crate::backend::runtime::checkpoint::load::GgufTensorNames,
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
    artifact: &super::artifact::PreparedSafetensorsArtifact,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    if let (Some(expert_cache), Some(non_expert)) = (
        options.weight_residency.expert_cache(),
        options.weight_residency.non_experts(),
    ) {
        return match kind {
            ModelKind::KimiLinear => Ok(Model::KimiLinear(kind,
                crate::composition::kimi_linear::load_kimi_linear_model(
                    artifact,
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
                Ok(Model::DeepSeek(kind, Box::new(
                    crate::composition::deepseek::load_safetensors(
                        artifact,
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
            ModelKind::GptOss => Ok(Model::GptOss(kind,
                crate::composition::gpt_oss::load_gpt_oss_expert_cache_model(
                    artifact, non_expert, expert_cache, options.quantization, stream, weights_stream,
                )?,
            )),
            ModelKind::Gemma4 => Ok(Model::Gemma4(kind,
                crate::composition::gemma4::load_safetensors(
                    artifact,
                    eredu_runtime::WeightResidency::with_expert_cache(
                        non_expert,
                        expert_cache,
                    ),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Inkling => Ok(Model::Inkling(kind,
                crate::composition::inkling::load_safetensors(
                    artifact,
                    eredu_runtime::WeightResidency::with_expert_cache(
                        non_expert,
                        expert_cache,
                    ),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Lfm2 => Ok(Model::Lfm2(kind,
                crate::composition::lfm2::load_lfm2_model(
                    artifact,
                    eredu_runtime::WeightResidency::with_expert_cache(non_expert, expert_cache),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::MuseGlimmer => Ok(Model::MuseGlimmer(kind,
                crate::composition::muse_glimmer::load_safetensors(
                    artifact,
                    eredu_runtime::WeightResidency::with_expert_cache(
                        non_expert,
                        expert_cache,
                    ),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::NemotronH => Ok(Model::NemotronH(kind,
                crate::composition::nemotron_h::load_nemotron_h_model(
                    artifact,
                    eredu_runtime::WeightResidency::with_expert_cache(non_expert, expert_cache),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen2 => Err(Error::UnsupportedArchitecture(
                "Qwen2 is dense and does not support sparse expert-cache residency".into(),
            )),
            ModelKind::Qwen3 => Ok(Model::Qwen(kind,
                crate::composition::qwen::load_qwen_safetensors_mlx(
                    artifact,
                    eredu_runtime::WeightResidency::with_expert_cache(non_expert, expert_cache),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3Next => Ok(Model::Qwen3Next(kind,
                crate::composition::qwen::hybrid::load_safetensors_with_residency(
                    artifact,
                    eredu_runtime::WeightResidency::with_expert_cache(
                        non_expert,
                        expert_cache,
                    ),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoe(kind,
                crate::composition::qwen::vl::load_safetensors_with_residency(
                    artifact,
                    eredu_runtime::WeightResidency::with_expert_cache(
                        non_expert,
                        expert_cache,
                    ),
                    options.quantization,
                    stream,
                    weights_stream,
                )?,
            )),
            ModelKind::Qwen35 => Ok(Model::Qwen35(kind,
                crate::composition::qwen::hybrid::load_safetensors_with_residency(
                    artifact,
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
                kind.canonical_name()
            ))),
        };
    }
    let execution = options.weight_residency.layers();
    if let Some(quantization) = options.quantization {
        quantization.validate()?;
    }
    match kind {
        ModelKind::DeepSeekV3 | ModelKind::DeepSeekV4 => Ok(Model::DeepSeek(
            kind,
            Box::new(crate::composition::deepseek::load_safetensors(
                artifact,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?),
        )),
        ModelKind::Gemma4 => Ok(Model::Gemma4(
            kind,
            crate::composition::gemma4::load_safetensors(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Inkling => Ok(Model::Inkling(
            kind,
            crate::composition::inkling::load_safetensors(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::KimiLinear => Ok(Model::KimiLinear(
            kind,
            crate::composition::kimi_linear::load_kimi_linear_model(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Llama => Ok(Model::Llama(
            kind,
            crate::composition::llama::load_llama_safetensors_mlx(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::MuseGlimmer => Ok(Model::MuseGlimmer(
            kind,
            crate::composition::muse_glimmer::load_safetensors(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen2 | ModelKind::Qwen3 => Ok(Model::Qwen(
            kind,
            crate::composition::qwen::load_qwen_safetensors_mlx(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::GptOss => Ok(Model::GptOss(
            kind,
            crate::composition::gpt_oss::load_gpt_oss_layerwise_model(
                artifact,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Lfm2 => Ok(Model::Lfm2(
            kind,
            crate::composition::lfm2::load_lfm2_model(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::NemotronH => Ok(Model::NemotronH(
            kind,
            crate::composition::nemotron_h::load_nemotron_h_model(
                artifact,
                eredu_runtime::WeightResidency::with_layers(execution),
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen3Next => Ok(Model::Qwen3Next(
            kind,
            crate::composition::qwen::hybrid::load_safetensors(
                artifact,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen3Vl => Ok(Model::Qwen3Vl(
            kind,
            crate::composition::qwen::vl::load_safetensors(
                artifact,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen3VlMoe => Ok(Model::Qwen3VlMoe(
            kind,
            crate::composition::qwen::vl::load_safetensors(
                artifact,
                execution,
                options.quantization,
                stream,
                weights_stream,
            )?,
        )),
        ModelKind::Qwen35 => Ok(Model::Qwen35(
            kind,
            crate::composition::qwen::hybrid::load_safetensors(
                artifact,
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
