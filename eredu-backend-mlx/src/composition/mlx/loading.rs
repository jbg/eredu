//! MLX checkpoint materialization after backend-neutral planning.

use eredu_checkpoint::WeightQuantization;

use eredu_architectures::processor_plan::ArtifactArchitecturePlan;
use eredu_architectures::{GgufArchitecture, ModelKind};
use eredu_core::{ModelArtifact, ModelPreparationPlan};
use safemlx::ops::GgufCheckpoint;
use safemlx::{ops::GgufMetadataValue, Stream};

#[cfg(feature = "media")]
use crate::composition::mlx::ModelProcessor;

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
    let kind = source.architecture().model_kind();
    structural::validate_complete_gguf_quantization(kind, options.quantization.is_some())?;
    let model = match source.architecture() {
        GgufArchitecture::KimiLinear => {
            let loaded = crate::composition::kimi_linear::load_kimi_linear_gguf_model(
                source,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            Model::KimiLinear(kind, loaded)
        }
        GgufArchitecture::DeepSeek2 => {
            let loaded = crate::composition::deepseek::load_gguf(
                source,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Model::DeepSeek(kind, Box::new(loaded))
        }
        GgufArchitecture::DeepSeek4 => {
            let loaded = crate::composition::deepseek::load_gguf(
                source,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Model::DeepSeek(kind, Box::new(loaded))
        }
        GgufArchitecture::GptOss => {
            let loaded = crate::composition::gpt_oss::load_gpt_oss_gguf_layerwise_model(
                source,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            Model::GptOss(kind, loaded)
        }
        GgufArchitecture::Inkling => {
            let loaded = crate::composition::inkling::load_gguf(
                source,
                projector,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Model::Inkling(kind, loaded)
        }
        GgufArchitecture::Gemma4 => {
            let loaded = crate::composition::gemma4::load_gguf(
                source,
                projector,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Model::Gemma4(kind, loaded)
        }
        GgufArchitecture::Llama | GgufArchitecture::Mistral => {
            let loaded = crate::composition::llama::load_llama_gguf_model(
                source,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            Model::Llama(kind, loaded)
        }
        GgufArchitecture::MuseGlimmer => {
            let projector = projector.ok_or_else(|| {
                Error::ArchitectureModel(
                    "Muse-Glimmer preparation omitted its required media projector".into(),
                )
            })?;
            let loaded = crate::composition::muse_glimmer::load_gguf(
                source,
                projector,
                options.weight_residency,
                stream,
                weights_stream,
            )?;
            Model::MuseGlimmer(kind, loaded)
        }
        GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
            let loaded = crate::composition::lfm2::load_lfm2_gguf_model(
                source,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            Model::Lfm2(kind, loaded)
        }
        GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
            let loaded = crate::composition::nemotron_h::load_nemotron_h_gguf_model(
                source,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            Model::NemotronH(kind, loaded)
        }
        GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
            let loaded = crate::composition::qwen::load_qwen_gguf_model(
                source,
                options.weight_residency,
                options.quantization,
                stream,
                weights_stream,
            )?;
            Model::Qwen(kind, loaded)
        }
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => {
            let projector = projector.ok_or_else(|| {
                Error::ArchitectureModel(
                    "Qwen3-VL preparation omitted its required media projector".into(),
                )
            })?;
            let loaded = crate::composition::qwen::vl::load_gguf(
                source,
                projector,
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
            variant
        }
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe | GgufArchitecture::Qwen3Next => {
            let loaded = crate::composition::qwen::hybrid::load_gguf(
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
            model
        }
    };
    Ok(model)
}

pub fn materialize_model_plan(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
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
        let kind = prepared_model_kind(plan.inspection().architecture_plan());
        if structural::requires_distributed_stage_loader(kind, topology.topology()) {
            #[cfg(feature = "media")]
            let processor = ModelProcessor::from_plan(plan.inspection().architecture_plan());
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
        let (artifact, architecture_plan, _policy, _route) = plan.into_parts();
        return match artifact {
            artifact @ ModelArtifact::Gguf { .. } => materialize_gguf_artifact(
                artifact,
                architecture_plan,
                options,
                stream,
                weights_stream,
            )
            .map(|model| complete_gguf_model(model, runtime_state_dtype_bytes)),
            ModelArtifact::SafeTensors {
                path,
                configuration,
                tensors,
            } => {
                let prepared = super::artifact::PreparedSafetensorsArtifact::open(
                    path,
                    configuration,
                    prepared_safetensors_architecture(&architecture_plan)?.clone(),
                    tensors,
                    options.weight_residency.max_mapped_shards(),
                )?;
                let model = materialize_tensor_parallel(&prepared, options, stream, weights_stream)
                    .map(|model| MlxModel::complete(model, runtime_state_dtype_bytes))?;
                attach_processor(model, &architecture_plan)
            }
        };
    }
    let (artifact, architecture_plan, _policy, _route) = plan.into_parts();
    match artifact {
        artifact @ ModelArtifact::Gguf { .. } => {
            materialize_gguf_artifact(artifact, architecture_plan, options, stream, weights_stream)
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
                prepared_safetensors_architecture(&architecture_plan)?.clone(),
                tensors,
                options.weight_residency.max_mapped_shards(),
            )?;
            let model = materialize_safetensors(&prepared, options, stream, weights_stream)
                .map(|model| MlxModel::complete(model, runtime_state_dtype_bytes))?;
            attach_processor(model, &architecture_plan)
        }
    }
}

fn inspected_runtime_state_dtype_bytes(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<std::num::NonZeroU8, Error> {
    if inspection.format() == eredu_core::ArtifactFormat::Gguf {
        // MLX native GGUF embeddings dequantize to Float32. Their catalog dtype
        // describes packed storage rather than the resulting activation.
        return Ok(std::num::NonZeroU8::new(4).expect("Float32 width is nonzero"));
    }
    let source = eredu_architectures::preparation::prepared_safetensors_runtime_state_dtype_source(
        prepared_safetensors_architecture(inspection.architecture_plan())?,
        inspection.tensors(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    mlx_runtime_state_dtype_bytes(source.dtype()).map_err(|dtype| {
        Error::ArchitectureModel(format!(
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
    use super::{mlx_runtime_state_dtype_bytes, reject_complete_tensor_parallel_quantization};
    use crate::backend::{DeviceAssignment, MlxParallelContext};
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
            assert!(super::structural::requires_distributed_stage_loader(
                kind,
                topology.topology()
            ));
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
            assert!(super::structural::requires_distributed_stage_loader(
                kind,
                topology.topology()
            ));
        }
        assert!(!super::structural::requires_distributed_stage_loader(
            ModelKind::Qwen3,
            topology.topology()
        ));
    }

    #[test]
    fn expert_parallel_topology_unconditionally_uses_distributed_stage_loader() {
        let topology =
            MlxParallelContext::for_rank(0, 1, 1, 2, DeviceAssignment::new(DeviceType::Cpu, 0))
                .unwrap();
        assert!(super::structural::requires_distributed_stage_loader(
            ModelKind::Llama,
            topology.topology()
        ));
    }

    #[cfg(feature = "image")]
    #[test]
    fn mlx_processor_consumes_retained_qwen_plan_after_sidecar_removal() {
        use eredu_core::{ArtifactFormat, ModelConfigurationResolver};

        let root = tempfile::tempdir().unwrap();
        let sidecar = root
            .path()
            .join(eredu_architectures::processor_plan::PROCESSOR_CONFIG_FILENAME);
        std::fs::write(
            &sidecar,
            br#"{
                "size":{"shortest_edge":16,"longest_edge":64},
                "patch_size":2,"temporal_patch_size":2,"merge_size":2,
                "image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0]
            }"#,
        )
        .unwrap();
        let config = serde_json::json!({
            "model_type": "qwen3_vl", "image_token_id": 61, "video_token_id": 62,
            "vision_start_token_id": 44, "vision_end_token_id": 45,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "qwen3_vl_text", "hidden_size": 32,
                "num_hidden_layers": 3, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                "rms_norm_eps": 0.000001, "vocab_size": 64,
                "max_position_embeddings": 128, "rope_theta": 1000000.0,
                "rope_scaling": {"mrope_section": [2, 1, 1], "mrope_interleaved": true}
            },
            "vision_config": {
                "depth": 4, "hidden_size": 16, "intermediate_size": 24,
                "num_heads": 4, "num_position_embeddings": 16, "in_channels": 3,
                "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "out_hidden_size": 32, "deepstack_visual_indexes": [1, 3]
            }
        });
        let (configuration, resolved_plan) =
            eredu_architectures::configuration::MODEL_CONFIGURATIONS
                .resolve_safetensors(&config)
                .unwrap()
                .into_parts();
        let architecture_plan = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .artifact_plan(
                root.path(),
                ArtifactFormat::SafeTensors,
                &configuration,
                None,
                resolved_plan,
            )
            .unwrap();
        std::fs::remove_file(sidecar).unwrap();

        assert_eq!(architecture_plan.model_kind(), ModelKind::Qwen3Vl);
        assert!(crate::composition::mlx::ModelProcessor::from_plan(&architecture_plan).is_some());
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

fn prepared_model_kind(plan: &ArtifactArchitecturePlan) -> ModelKind {
    plan.model_kind()
}

pub(super) fn prepared_safetensors_architecture(
    plan: &ArtifactArchitecturePlan,
) -> Result<&eredu_architectures::configuration::SafetensorsArchitecturePlan, Error> {
    plan.safetensors_architecture().ok_or_else(|| {
        Error::ArchitectureModel(
            "SafeTensors preparation omitted its validated architecture plan".into(),
        )
    })
}

fn prepared_gguf_plan(
    plan: &ArtifactArchitecturePlan,
) -> Result<&eredu_architectures::configuration::GgufArchitecturePlan, Error> {
    plan.gguf_plan().ok_or_else(|| {
        Error::ArchitectureModel("GGUF preparation omitted its validated architecture plan".into())
    })
}

fn attach_processor(
    model: MlxModel,
    architecture_plan: &ArtifactArchitecturePlan,
) -> Result<MlxModel, Error> {
    #[cfg(feature = "media")]
    {
        Ok(model.with_processor(ModelProcessor::from_plan(architecture_plan)))
    }
    #[cfg(not(feature = "media"))]
    {
        let _ = architecture_plan;
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
    artifact: &super::artifact::PreparedSafetensorsArtifact,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let kind = artifact.architecture().model_kind();
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
        ModelKind::Qwen3Next => Err(Error::ArchitectureModel(
            "neutral Qwen hybrid tensor-parallel binding is not initialized".into(),
        )),
        ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => Err(Error::ArchitectureModel(
            "neutral Qwen3-VL tensor-parallel binding is not initialized".into(),
        )),
        ModelKind::Qwen35 => Err(Error::ArchitectureModel(
            "neutral Qwen3.5 tensor-parallel binding is not initialized".into(),
        )),
        ModelKind::Moshi => Err(Error::ArchitectureModel(
            "Moshi-family models do not use the text Model tensor-parallel session".into(),
        )),
    }
}

pub(super) fn validate_plan_options(
    plan: &ModelPreparationPlan<ArtifactArchitecturePlan>,
    options: ModelLoadOptions,
) -> Result<(), Error> {
    if plan.policy() != options.preparation_policy()? {
        return Err(Error::ArchitectureModel(
            "MLX materialization options do not match the backend-neutral preparation plan".into(),
        ));
    }
    super::structural::validate_inspected_preparation(plan.inspection(), plan.policy())?;
    Ok(())
}

fn materialize_gguf_artifact(
    artifact: ModelArtifact,
    architecture_plan: ArtifactArchitecturePlan,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MaterializedGgufModel, Error> {
    let ModelArtifact::Gguf {
        path: _,
        configuration: _,
        validated,
        ..
    } = artifact
    else {
        return Err(Error::ArchitectureModel(
            "MLX GGUF materializer received a SafeTensors plan".into(),
        ));
    };
    let architecture = prepared_gguf_plan(&architecture_plan)?.clone();
    let (source, mut companions) =
        structural::AdmittedGguf::from_admission(architecture, validated);
    let projector = companions
        .remove(&eredu_core::GgufCompanionRole::MediaProjector)
        .map(|companion| GgufCheckpoint::from_portable(companion.checkpoint().clone()));
    let checkpoint = source.checkpoint();
    let metadata = source.metadata();
    validate_gguf_quantization_source(checkpoint, metadata, options.quantization)?;
    #[cfg(feature = "media")]
    let processor = ModelProcessor::from_plan(&architecture_plan);
    if options
        .parallel
        .is_some_and(|topology| !topology.is_replicated())
    {
        let model = materialize_gguf_tensor_parallel(
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
) -> Result<Model, Error> {
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
            let model =
                crate::composition::kimi_linear::load_kimi_linear_gguf_tensor_parallel_model(
                    source,
                    residency,
                    build,
                    stream,
                    weights_stream,
                )?;
            Ok(Model::KimiLinear(kind, model))
        }
        GgufArchitecture::DeepSeek2 | GgufArchitecture::DeepSeek4 => Err(Error::Parallel(
            "DeepSeek GGUF tensor parallelism requires distributed-stage materialization".into(),
        )),
        GgufArchitecture::GptOss => {
            let model = crate::composition::gpt_oss::load_gpt_oss_gguf_tensor_parallel_model(
                source,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok(Model::GptOss(kind, model))
        }
        GgufArchitecture::Inkling => {
            let model = crate::composition::inkling::load_gguf_tensor_parallel(
                source,
                projector,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok(Model::Inkling(kind, model))
        }
        GgufArchitecture::Gemma4 => {
            let model = crate::composition::gemma4::load_gguf_tensor_parallel(
                source,
                projector,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok(Model::Gemma4(kind, model))
        }
        GgufArchitecture::Llama | GgufArchitecture::Mistral => {
            let model = crate::composition::llama::load_llama_gguf_tensor_parallel_model(
                source,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok(Model::Llama(kind, model))
        }
        GgufArchitecture::MuseGlimmer => {
            let projector = projector.ok_or_else(|| {
                Error::ArchitectureModel(
                    "Muse-Glimmer preparation omitted its required media projector".into(),
                )
            })?;
            let model = crate::composition::muse_glimmer::load_gguf_tensor_parallel(
                source,
                projector,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok(Model::MuseGlimmer(kind, model))
        }
        GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
            let model = crate::composition::lfm2::load_lfm2_gguf_tensor_parallel_model(
                source,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok(Model::Lfm2(kind, model))
        }
        GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
            let model = crate::composition::nemotron_h::load_nemotron_h_gguf_tensor_parallel_model(
                source,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok(Model::NemotronH(kind, model))
        }
        GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
            let model = crate::composition::qwen::load_qwen_gguf_tensor_parallel_model(
                source,
                residency,
                build,
                stream,
                weights_stream,
            )?;
            Ok(Model::Qwen(kind, model))
        }
        GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe => Err(Error::ArchitectureModel(
            "neutral Qwen3-VL GGUF tensor-parallel binding is not initialized".into(),
        )),
        GgufArchitecture::Qwen35 | GgufArchitecture::Qwen35Moe | GgufArchitecture::Qwen3Next => {
            Err(Error::ArchitectureModel(
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
    artifact: &super::artifact::PreparedSafetensorsArtifact,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let kind = artifact.architecture().model_kind();
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
            ModelKind::Qwen2 => Err(Error::ArchitectureModel(
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
            _ => Err(Error::ArchitectureModel(format!(
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
        ModelKind::Moshi => Err(Error::ArchitectureModel(
            "Moshi-family bounded layer residency is selected through the realtime loader".into(),
        )),
    }
}
