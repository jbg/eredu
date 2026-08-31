use std::{ops::Range, sync::Arc};

use crate::backend::runtime::distributed::Group;
use eredu_architectures::ModelKind;
use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
use eredu_core::cache::{CacheRankIdentity, PromptCacheModelIdentity};
use eredu_nn::RoutedNeuralBackend;
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureParameters, ExpertCacheLoadOptions, ExpertPass,
    LayeredArchitecture,
};
use safemlx::{error::Exception, Array, Stream};

use crate::{
    backend::{
        error::Error,
        nn::shared::{MlxModule, MlxNeuralBackend},
        runtime::{
            cache::{residency::CacheResidencyManager, state::MlxHybridState},
            checkpoint::{
                binding::populate_module_from_lease, quantization::should_quantize_on_load,
            },
            distributed::{
                completion::synchronize_outputs,
                expert::{
                    dispatch_local_with, dispatch_replicated_with, ExpertAssignment,
                    RoutingStatistics,
                },
                parallel::ParallelExecutionContext,
            },
            execution::layerwise::PipelineStageQuantizationSelection,
            residency::{
                expert_cache::ExpertCache,
                expert_provider::{
                    GatedProductExpertExecution, GatedProductExpertExecutorProvider,
                    ResidentExpertExecutorProvider,
                },
            },
        },
        MlxParallelContext,
    },
    composition::mlx::distributed::pipeline::{
        architecture_decoder_group, architecture_group_by_id, architecture_group_id,
        architecture_group_unit_count, architecture_parallel_layout,
        architecture_parameter_unit_owner, architecture_partition_group_range,
        architecture_prediction_group, architecture_single_prediction_units, base_info,
        build_pipeline_expert_cache, build_pipeline_layer_storage, checkpoint_backing_shards,
        checkpoint_unit_backing_shards, construct_qwen_partition_unit,
        decoder_architecture_transport, execute_neutral_decoder_partition_observed,
        execute_neutral_routed_decoder_partition_observed, execute_neutral_routed_output_group,
        execute_resident_distributed_experts, execute_routed_layered_partition_observed,
        load_architecture_static_parameters, materialize_pipeline_cache_layers,
        media_architecture_transport, partition_owns_architecture_units, pipeline_binding_units,
        prediction_architecture_transport, quantize_pipeline_stage_store,
        validate_admitted_pipeline_kind, validate_pipeline_expert_dispatch, BoundPipelineBindings,
        DecoderPipelineBuilder, MlxPlacedGroupExecutor, PipelineAuxiliaryState,
        PipelineEmbeddedMtp, PipelineExpertStorage, PipelineForward, PipelineLayerCache,
        PipelineLayerController, PipelineLayerLoadOptions, PipelineLayerStorage,
        PipelineLoadAccumulator, PipelineModel, PipelineMtpCache, PipelinePartitionMetadata,
        PipelinePayload, PipelineRangeState, PipelineStageInput, PipelineStageOutput, PipelineStep,
        QwenConditionalPipelinePartition, QwenHybridPipelinePartition, QwenPipelinePartition,
        QwenVlPipelinePartition,
    },
    composition::{
        mlx::speculative::embedded::EmbeddedMtpOutput,
        qwen::{
            hybrid::{QwenConditionalPipelineBindings, QwenHybridPipelineBindings},
            vl::QwenVlPipelineBindings,
        },
    },
};

#[cfg(test)]
use crate::composition::mlx::distributed::pipeline::validate_distributed_stage_capabilities;

#[cfg(test)]
#[test]
fn nested_qwen35_moe_capabilities_pass_cartesian_pipeline_preflight() {
    let mut config = serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_moe",
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "intermediate_size": 0,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 24,
            "num_experts_per_tok": 2,
            "num_experts": 8,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ]
        }
    });
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    assert_eq!(resolved.effective_model_type, "qwen3_5_moe");
    let capabilities =
        eredu_architectures::preparation::prepared_safetensors_capabilities(&resolved.architecture)
            .unwrap();
    let topology = MlxParallelContext::for_rank(
        0,
        2,
        2,
        2,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
    )
    .unwrap();

    validate_distributed_stage_capabilities(
        capabilities,
        topology,
        true,
        "SafeTensors",
        &resolved.effective_model_type,
    )
    .unwrap();

    config["text_config"]["model_type"] = serde_json::json!("qwen3_5_text");
    config["text_config"]["intermediate_size"] = serde_json::json!(48);
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    assert_eq!(resolved.effective_model_type, "qwen3_5_text");
    let capabilities =
        eredu_architectures::preparation::prepared_safetensors_capabilities(&resolved.architecture)
            .unwrap();
    let error = validate_distributed_stage_capabilities(
        capabilities,
        topology,
        false,
        "SafeTensors",
        &resolved.effective_model_type,
    )
    .unwrap_err();
    assert!(error.to_string().contains("expert-parallel"));

    let topology = MlxParallelContext::for_rank(
        0,
        2,
        2,
        1,
        crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
    )
    .unwrap();
    let error = validate_distributed_stage_capabilities(
        capabilities,
        topology,
        true,
        "SafeTensors",
        &resolved.effective_model_type,
    )
    .unwrap_err();
    assert!(error.to_string().contains("independent expert-residency"));
}

impl PipelinePartitionMetadata for QwenPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::qwen(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part(
            "qwen",
            input,
            &crate::backend::runtime::media::input::MlxInputInspector,
        )
    }

    fn boundary_wire_schema(&self) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        self.partition
            .boundary_schema()
            .wire_schema()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_cache.as_ref()
    }
}

impl PipelineForward for QwenPipelinePartition {
    pipeline_observed_forward!();
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if self.expert_cache.is_none() && self.expert_assignment.is_none() {
            execute_neutral_decoder_partition_observed(
                self, input, step, mask, cache, None, stream, None,
            )
        } else if self.expert_cache.is_some() {
            self.forward_external_experts_neutral(
                input, step, mask, cache, None, None, stream, None,
            )
        } else {
            Err(Error::Parallel(
                "resident Qwen expert parallelism requires its EP communicator".into(),
            ))
        }
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        if let Some(group) = expert_group {
            if self.expert_cache.is_some() {
                return self.forward_external_experts_neutral(
                    input,
                    step,
                    mask,
                    cache,
                    execution,
                    Some(group),
                    stream,
                    observer,
                );
            }
            return self.forward_resident_experts_neutral(
                input, step, mask, cache, execution, group, stream, observer,
            );
        }
        if self
            .expert_assignment
            .as_ref()
            .is_some_and(|assignment| assignment.group_size() > 1)
        {
            return Err(Error::Parallel(
                "Qwen expert assignment requires its EP communicator".into(),
            ));
        }
        match execution {
            Some(execution)
                if execution.is_tensor_parallel()
                    && self.expert_cache.is_none()
                    && self.expert_assignment.is_none() =>
            {
                execute_neutral_decoder_partition_observed(
                    self,
                    input,
                    step,
                    mask,
                    cache,
                    Some(execution),
                    execution.stream(),
                    observer,
                )
            }
            Some(execution) if execution.is_tensor_parallel() => {
                if self.expert_cache.is_some() {
                    self.forward_external_experts_neutral(
                        input,
                        step,
                        mask,
                        cache,
                        Some(execution),
                        None,
                        execution.stream(),
                        observer,
                    )
                } else {
                    execute_neutral_decoder_partition_observed(
                        self,
                        input,
                        step,
                        mask,
                        cache,
                        Some(execution),
                        execution.stream(),
                        observer,
                    )
                }
            }
            _ if self.expert_cache.is_some() => self.forward_external_experts_neutral(
                input, step, mask, cache, None, None, stream, observer,
            ),
            _ => execute_neutral_decoder_partition_observed(
                self, input, step, mask, cache, None, stream, observer,
            ),
        }
    }
}

impl QwenVlPipelinePartition {
    fn args(&self) -> &eredu_architectures::qwen::vl::ModelArgs {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_architectures::qwen::vl::TEXT_EXECUTION_GROUP)
    }

    fn vision_range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_architectures::qwen::vl::VISION_EXECUTION_GROUP)
    }

    fn boundary_schema(
        &self,
    ) -> Result<eredu_architectures::qwen::vl::PipelineBoundarySchema, Error> {
        Ok(*self.partition.boundary_schema())
    }

    fn new(
        architecture: eredu_architectures::qwen::vl::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Option<Arc<eredu_architectures::qwen::vl::LocalGeometry>>,
            eredu_architectures::qwen::vl::PipelineBoundarySchema,
        >,
        external_experts: bool,
    ) -> Result<Self, Error> {
        let adapter = if external_experts {
            QwenVlPipelineBindings::new_external_experts()
        } else {
            QwenVlPipelineBindings::new()
        };
        Ok(Self {
            architecture,
            partition,
            adapter,
            vision_layers: Vec::new(),
            audio_layers: Vec::new(),
            layers: Vec::new(),
            prediction_layers: Vec::new(),
            dense_layers: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
            ingress_state: None,
        })
    }

    fn begin_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        offset: i32,
        delta: Option<&Array>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<eredu_architectures::qwen::vl::PipelineVisionState<crate::MlxTensor>, Error> {
        self.adapter.begin_pipeline_ingress(
            &mut self.architecture,
            input,
            offset,
            delta,
            execution
                .filter(|execution| execution.is_tensor_parallel())
                .and_then(ParallelExecutionContext::group),
            stream,
        )
    }

    fn execute_vision_state(
        &mut self,
        state: &mut eredu_architectures::qwen::vl::PipelineVisionState<crate::MlxTensor>,
        tensor_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<(), Error> {
        if let Some(storage) = self.dense_layers.as_ref() {
            let forward_guard = match &storage.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.forward_guard(true, &storage.residency)?)
                }
            };
            let group_guard = match &storage.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.group_guard(&storage.residency, "pipeline_stage"))
                }
            };
            let vision_group = architecture_group_by_id::<_, MlxHybridState>(
                &self.architecture,
                eredu_architectures::qwen::vl::VISION_EXECUTION_GROUP,
            )?;
            let mut window = storage.transfer_window(0..self.vision_range().len(), true)?;
            for (ordinal, index) in self.vision_range().clone().enumerate() {
                let transfer = window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = transfer
                    .is_none()
                    .then(|| storage.prepare_layerwise_absolute(ordinal))
                    .transpose()?;
                let mut layer = self
                    .architecture
                    .construct_unit(vision_group, index, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("Qwen3-VL placed vision residency lease"),
                )?;
                let eredu_architectures::qwen::vl::Unit::Vision(block) = &mut *layer else {
                    return Err(Error::Parallel(format!(
                        "Qwen3-VL vision range contains text unit {index}"
                    )));
                };
                self.architecture
                    .forward_pipeline_vision(index, block, state, tensor_group, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                synchronize_outputs(
                    eredu_architectures::qwen::vl::LayeredModel::<MlxNeuralBackend>::pipeline_retained_values(state)
                        .iter()
                        .map(crate::MlxTensor::as_array),
                )?;
                drop(transfer);
                drop(lease);
                if let Some(window) = &mut window {
                    window.refill()?;
                } else {
                    storage.trim_after_absolute(ordinal)?;
                }
            }
            storage.complete_forward()?;
            if let Some(guard) = group_guard {
                guard.complete()?;
            }
            if let Some(guard) = forward_guard {
                guard.complete()?;
            }
        } else {
            for (index, layer) in self.vision_range().clone().zip(&mut self.vision_layers) {
                let eredu_architectures::qwen::vl::Unit::Vision(block) = &mut **layer else {
                    return Err(Error::Parallel(format!(
                        "Qwen3-VL vision range contains text unit {index}"
                    )));
                };
                self.architecture
                    .forward_pipeline_vision(index, block, state, tensor_group, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_decoder(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.range().len() {
            return Err(Error::Parallel(format!(
                "Qwen3-VL stage cache has {} entries, expected {}",
                caches.len(),
                self.range().len()
            )));
        }
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let assignment = self.expert_assignment.clone();
        if let Some(assignment) = assignment.as_ref() {
            validate_pipeline_expert_dispatch(
                assignment,
                expert_group,
                self.expert_storage.is_external(),
            )?;
        }
        self.routing_statistics = RoutingStatistics::default();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let cache = self.expert_storage.cache();
        let decoder_range = self.range();
        if let Some(expert_cache) = cache {
            let assignment = assignment.as_ref().ok_or_else(|| {
                Error::Parallel("Qwen3-VL external experts have no assignment".into())
            })?;
            let mut execute = |execution: GatedProductExpertExecution, stream: &Stream| {
                execute_pipeline_cached_qwen3(
                    &execution.spec,
                    execution.layer,
                    &execution.hidden,
                    &execution.expert_ids,
                    &execution.route_weights,
                    pass,
                    expert_cache,
                    assignment,
                    expert_group,
                    tensor_group,
                    &mut self.routing_statistics,
                    stream,
                )
                .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
                .map_err(|error| Exception::custom(error.to_string()))
            };
            let mut provider = GatedProductExpertExecutorProvider::new(&mut execute);
            execute_routed_layered_partition_observed(
                &mut self.architecture,
                &self.partition,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                input,
                step,
                explicit_mask,
                caches,
                execution,
                pass,
                &mut provider,
                stream,
                observer,
            )
        } else {
            execute_routed_layered_partition_observed(
                &mut self.architecture,
                &self.partition,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                input,
                step,
                explicit_mask,
                caches,
                execution,
                pass,
                &mut eredu_runtime::ResidentExpertProvider,
                stream,
                observer,
            )
        }
    }
}

impl PipelinePartitionMetadata for QwenVlPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::qwen_vl(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::qwen_vl_input_part(
            self.args(),
            input,
            &crate::backend::runtime::media::input::MlxInputInspector,
        )
        .map(Into::into)
    }

    fn boundary_wire_schema(&self) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        self.partition
            .boundary_schema()
            .wire_schema()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }
}

impl MlxPlacedGroupExecutor for QwenVlPipelinePartition {
    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_ingress(input, 0, None, execution, stream)?);
        Ok(())
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.begin_placed_ingress(input, execution, stream)
    }

    fn placed_ingress_active(&self, _group: &str) -> Result<bool, Error> {
        let state = self
            .ingress_state
            .as_ref()
            .ok_or_else(|| Error::Parallel("Qwen3-VL ingress state is unavailable".into()))?;
        Ok(eredu_architectures::qwen::vl::LayeredModel::<
            MlxNeuralBackend,
        >::pipeline_vision_active(state))
    }

    fn placed_ingress_arrays(&self, _group: &str) -> Result<Vec<Array>, Error> {
        let state = self
            .ingress_state
            .as_ref()
            .ok_or_else(|| Error::Parallel("Qwen3-VL ingress state is unavailable".into()))?;
        Ok(
            eredu_architectures::qwen::vl::LayeredModel::<MlxNeuralBackend>::pipeline_retained_values(
                state,
            )
            .into_iter()
            .map(crate::MlxTensor::into_array)
            .collect(),
        )
    }

    fn replace_placed_ingress_arrays(
        &mut self,
        _group: &str,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = self
            .ingress_state
            .as_mut()
            .ok_or_else(|| Error::Parallel("Qwen3-VL ingress state is unavailable".into()))?;
        eredu_architectures::qwen::vl::LayeredModel::<MlxNeuralBackend>::replace_pipeline_retained_values(
            state,
            arrays
                .into_iter()
                .map(crate::MlxTensor::from_array)
                .collect(),
        )
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn merge_placed_ingress_arrays(&mut self, arrays: Vec<Array>) -> Result<(), Error> {
        let group = architecture_group_id::<_, MlxHybridState>(
            &self.architecture,
            eredu_architectures::qwen::vl::VISION_EXECUTION_GROUP,
        )?;
        self.replace_placed_ingress_arrays(&group, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let vision_group = architecture_group_id::<_, MlxHybridState>(
            &self.architecture,
            eredu_architectures::qwen::vl::VISION_EXECUTION_GROUP,
        )?;
        if group != vision_group {
            return Ok(());
        }
        let mut state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Qwen3-VL ingress state is unavailable".into()))?;
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let result = self.execute_vision_state(&mut state, tensor_group, stream);
        self.ingress_state = Some(state);
        result
    }

    fn finish_placed_ingress(
        &mut self,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Qwen3-VL ingress state is unavailable".into()))?;
        let prepared = self
            .architecture
            .finish_pipeline(
                state,
                execution
                    .filter(|execution| execution.is_tensor_parallel())
                    .and_then(ParallelExecutionContext::group),
                stream,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let (hidden, boundary) =
            eredu_architectures::qwen::vl::PipelineBoundary::from_prepared(prepared);
        Ok(PipelinePayload {
            hidden: hidden.into_array(),
            auxiliary: PipelineAuxiliaryState::new(
                self.boundary_schema()?
                    .encode(boundary)
                    .map_err(|error| Error::Parallel(error.to_string()))?
                    .into_iter()
                    .map(crate::MlxTensor::into_array)
                    .collect(),
            ),
        })
    }

    fn prefill(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        let mut state = self.begin_ingress(input, 0, None, execution, stream)?;
        let group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        if eredu_architectures::qwen::vl::LayeredModel::<MlxNeuralBackend>::pipeline_vision_active(
            &state,
        ) {
            self.execute_vision_state(&mut state, group, stream)?;
        }
        let prepared = self
            .architecture
            .finish_pipeline(state, group, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let (hidden, boundary) =
            eredu_architectures::qwen::vl::PipelineBoundary::from_prepared(prepared);
        let payload = PipelinePayload {
            hidden: hidden.into_array(),
            auxiliary: PipelineAuxiliaryState::new(
                self.boundary_schema()?
                    .encode(boundary)
                    .map_err(|error| Error::Parallel(error.to_string()))?
                    .into_iter()
                    .map(crate::MlxTensor::into_array)
                    .collect(),
            ),
        };
        self.forward_decoder(
            PipelineStageInput::Hidden(&payload),
            step,
            mask,
            cache,
            execution,
            expert_group,
            stream,
            observer,
        )
    }
}

impl PipelineForward for QwenVlPipelinePartition {
    pipeline_observed_forward!();
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_decoder(input, step, mask, cache, None, None, stream, None)
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_decoder(
            input,
            step,
            mask,
            cache,
            execution,
            expert_group,
            stream,
            observer,
        )
    }
}

impl QwenConditionalPipelinePartition {
    fn args(&self) -> &eredu_architectures::qwen::hybrid::ParsedHybridConfig {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        let group = architecture_decoder_group::<_, MlxHybridState>(&self.architecture)
            .expect("validated conditional Qwen target decoder group");
        architecture_partition_group_range(&self.partition, group)
    }

    fn vision_range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(
            eredu_architectures::qwen::hybrid::VISION_EXECUTION_GROUP,
        )
    }

    fn boundary_schema(
        &self,
    ) -> Result<eredu_architectures::qwen::hybrid::ConditionalPipelineBoundarySchema, Error> {
        Ok(*self.partition.boundary_schema())
    }

    fn new(
        architecture: eredu_architectures::qwen::hybrid::ConditionalLayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Option<Arc<eredu_architectures::qwen::hybrid::ConditionalLocalGeometry>>,
            eredu_architectures::qwen::hybrid::ConditionalPipelineBoundarySchema,
        >,
        external_experts: bool,
    ) -> Result<Self, Error> {
        let adapter = if external_experts {
            QwenConditionalPipelineBindings::new_external_experts()
        } else {
            QwenConditionalPipelineBindings::new()
        };
        Ok(Self {
            architecture,
            partition,
            adapter,
            vision_layers: Vec::new(),
            audio_layers: Vec::new(),
            layers: Vec::new(),
            prediction_layers: Vec::new(),
            dense_layers: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
            ingress_state: None,
        })
    }

    fn begin_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        offset: i32,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<
        eredu_architectures::qwen::hybrid::ConditionalPipelineVisionState<crate::MlxTensor>,
        Error,
    > {
        self.adapter.begin_pipeline_ingress(
            &mut self.architecture,
            input,
            offset,
            execution
                .filter(|execution| execution.is_tensor_parallel())
                .and_then(ParallelExecutionContext::group),
            stream,
        )
    }

    fn execute_vision_state(
        &mut self,
        state: &mut eredu_architectures::qwen::hybrid::ConditionalPipelineVisionState<
            crate::MlxTensor,
        >,
        tensor_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<(), Error> {
        if let Some(storage) = self.dense_layers.as_ref() {
            let forward_guard = match &storage.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.forward_guard(true, &storage.residency)?)
                }
            };
            let group_guard = match &storage.controller {
                PipelineLayerController::LayerwiseHost(_) => None,
                PipelineLayerController::DenseDiskStream(controller) => {
                    Some(controller.group_guard(&storage.residency, "pipeline_stage"))
                }
            };
            let vision_group = architecture_group_by_id::<_, MlxHybridState>(
                &self.architecture,
                eredu_architectures::qwen::hybrid::VISION_EXECUTION_GROUP,
            )?;
            let mut window = storage.transfer_window(0..self.vision_range().len(), true)?;
            for (ordinal, index) in self.vision_range().clone().enumerate() {
                let transfer = window
                    .as_mut()
                    .map(|window| window.next(stream))
                    .transpose()?;
                let lease = transfer
                    .is_none()
                    .then(|| storage.prepare_layerwise_absolute(ordinal))
                    .transpose()?;
                let mut layer = self
                    .architecture
                    .construct_unit(vision_group, index, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("conditional Qwen3.5 vision residency lease"),
                )?;
                let eredu_architectures::qwen::hybrid::ConditionalUnit::Vision(block) = &mut *layer
                else {
                    return Err(Error::Parallel(format!(
                        "conditional Qwen3.5 vision range contains text unit {index}"
                    )));
                };
                self.architecture
                    .forward_pipeline_vision(index, block, state, tensor_group, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
                synchronize_outputs(
                    eredu_architectures::qwen::hybrid::ConditionalLayeredModel::<MlxNeuralBackend>::pipeline_retained_values(state)
                        .iter()
                        .map(crate::MlxTensor::as_array),
                )?;
                drop(transfer);
                drop(lease);
                if let Some(window) = &mut window {
                    window.refill()?;
                } else {
                    storage.trim_after_absolute(ordinal)?;
                }
            }
            storage.complete_forward()?;
            if let Some(guard) = group_guard {
                guard.complete()?;
            }
            if let Some(guard) = forward_guard {
                guard.complete()?;
            }
        } else {
            for (index, layer) in self.vision_range().clone().zip(&mut self.vision_layers) {
                let eredu_architectures::qwen::hybrid::ConditionalUnit::Vision(block) =
                    &mut **layer
                else {
                    return Err(Error::Parallel(format!(
                        "conditional Qwen3.5 vision range contains text unit {index}"
                    )));
                };
                self.architecture
                    .forward_pipeline_vision(index, block, state, tensor_group, stream)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_decoder(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.range().len() {
            return Err(Error::Parallel(format!(
                "conditional Qwen3.5 stage cache has {} entries, expected {}",
                caches.len(),
                self.range().len()
            )));
        }
        let assignment = self.expert_assignment.clone();
        if let Some(assignment) = assignment.as_ref() {
            validate_pipeline_expert_dispatch(
                assignment,
                expert_group,
                self.expert_storage.is_external(),
            )?;
        }
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.routing_statistics = RoutingStatistics::default();
        let expert_cache = self.expert_storage.cache();
        let decoder_range = self.range();
        if let Some(expert_cache) = expert_cache {
            let assignment = assignment.as_ref().ok_or_else(|| {
                Error::Parallel("conditional Qwen external experts have no assignment".into())
            })?;
            let mut execute = |execution: GatedProductExpertExecution, stream: &Stream| {
                execute_pipeline_cached_neutral_qwen_hybrid(
                    &execution.spec,
                    execution.layer,
                    &execution.hidden,
                    &execution.expert_ids,
                    &execution.route_weights,
                    pass,
                    expert_cache,
                    assignment,
                    expert_group,
                    &mut self.routing_statistics,
                    stream,
                )
                .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
                .map_err(|error| Exception::custom(error.to_string()))
            };
            let mut provider = GatedProductExpertExecutorProvider::new(&mut execute);
            execute_routed_layered_partition_observed(
                &mut self.architecture,
                &self.partition,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                input,
                step,
                explicit_mask,
                caches,
                execution,
                pass,
                &mut provider,
                stream,
                observer,
            )
        } else {
            execute_routed_layered_partition_observed(
                &mut self.architecture,
                &self.partition,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                input,
                step,
                explicit_mask,
                caches,
                execution,
                pass,
                &mut eredu_runtime::ResidentExpertProvider,
                stream,
                observer,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn forward_mtp_draft_neutral(
        &mut self,
        prior: &Array,
        tokens: &Array,
        depth: usize,
        state: &mut MlxHybridState,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let units = self.prediction_layers.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!("conditional Qwen3.5 has no MTP depth {depth}"))
        })?;
        let prediction_group =
            architecture_prediction_group::<_, MlxHybridState>(&self.architecture, depth)?;
        let mut execute = |execution: GatedProductExpertExecution, stream: &Stream| {
            let cache = self.expert_storage.cache().ok_or_else(|| {
                Exception::custom("conditional Qwen3.5 MTP external expert cache is unavailable")
            })?;
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Exception::custom("conditional Qwen3.5 MTP external experts have no assignment")
            })?;
            execute_pipeline_cached_neutral_qwen_hybrid(
                &execution.spec,
                execution.layer,
                &execution.hidden,
                &execution.expert_ids,
                &execution.route_weights,
                ExpertPass::Decode,
                cache,
                assignment,
                expert_group,
                &mut self.routing_statistics,
                stream,
            )
            .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
            .map_err(|error| Exception::custom(error.to_string()))
        };
        let input = eredu_architectures::qwen::hybrid::ConditionalInput::Draft {
            tokens: crate::composition::tensor_ref(tokens),
            hidden: crate::composition::tensor_ref(prior),
            depth,
        };
        let (logits, hidden) = if self.expert_storage.cache().is_some() {
            let mut provider = GatedProductExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_output_group(
                &mut self.architecture,
                input,
                prediction_group,
                units,
                state,
                ExpertPass::Decode,
                &mut provider,
                tensor_group,
                stream,
            )
        } else {
            execute_neutral_routed_output_group(
                &mut self.architecture,
                input,
                prediction_group,
                units,
                state,
                ExpertPass::Decode,
                &mut eredu_runtime::ResidentExpertProvider,
                tensor_group,
                stream,
            )
        }?;
        Ok(EmbeddedMtpOutput {
            logits,
            hidden,
            tokens: crate::MlxTensor::from_array(tokens.clone()),
        })
    }
}

impl PipelinePartitionMetadata for QwenConditionalPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::qwen_hybrid(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::qwen_hybrid_input_part(
            self.args(),
            input,
            &crate::backend::runtime::media::input::MlxInputInspector,
        )
        .map(Into::into)
    }

    fn boundary_wire_schema(&self) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        self.partition
            .boundary_schema()
            .wire_schema()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        let target_identity = identity
            .select_state_segment(eredu_architectures::qwen::hybrid::TARGET_STATE_SEGMENT)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        materialize_pipeline_cache_layers(&target_identity, paged)
    }
}

impl MlxPlacedGroupExecutor for QwenConditionalPipelinePartition {
    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_ingress(input, 0, execution, stream)?);
        Ok(())
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.begin_placed_ingress(input, execution, stream)
    }

    fn placed_ingress_active(&self, _group: &str) -> Result<bool, Error> {
        let state = self.ingress_state.as_ref().ok_or_else(|| {
            Error::Parallel("conditional Qwen3.5 ingress state is unavailable".into())
        })?;
        Ok(eredu_architectures::qwen::hybrid::ConditionalLayeredModel::<
                MlxNeuralBackend,
            >::pipeline_vision_active(state))
    }

    fn placed_ingress_arrays(&self, _group: &str) -> Result<Vec<Array>, Error> {
        let state = self.ingress_state.as_ref().ok_or_else(|| {
            Error::Parallel("conditional Qwen3.5 ingress state is unavailable".into())
        })?;
        Ok(eredu_architectures::qwen::hybrid::ConditionalLayeredModel::<
                MlxNeuralBackend,
            >::pipeline_retained_values(state)
            .into_iter()
            .map(crate::MlxTensor::into_array)
            .collect())
    }

    fn replace_placed_ingress_arrays(
        &mut self,
        _group: &str,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = self.ingress_state.as_mut().ok_or_else(|| {
            Error::Parallel("conditional Qwen3.5 ingress state is unavailable".into())
        })?;
        eredu_architectures::qwen::hybrid::ConditionalLayeredModel::<MlxNeuralBackend>::replace_pipeline_retained_values(
            state,
            arrays.into_iter().map(crate::MlxTensor::from_array).collect(),
        )
                .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn merge_placed_ingress_arrays(&mut self, arrays: Vec<Array>) -> Result<(), Error> {
        let group = architecture_group_id::<_, MlxHybridState>(
            &self.architecture,
            eredu_architectures::qwen::hybrid::VISION_EXECUTION_GROUP,
        )?;
        self.replace_placed_ingress_arrays(&group, arrays)
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let vision_group = architecture_group_id::<_, MlxHybridState>(
            &self.architecture,
            eredu_architectures::qwen::hybrid::VISION_EXECUTION_GROUP,
        )?;
        if group != vision_group {
            return Ok(());
        }
        let mut state = self.ingress_state.take().ok_or_else(|| {
            Error::Parallel("conditional Qwen3.5 ingress state is unavailable".into())
        })?;
        let group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let result = self.execute_vision_state(&mut state, group, stream);
        self.ingress_state = Some(state);
        result
    }

    fn finish_placed_ingress(
        &mut self,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let state = self.ingress_state.take().ok_or_else(|| {
            Error::Parallel("conditional Qwen3.5 ingress state is unavailable".into())
        })?;
        let group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let prepared = self
            .architecture
            .finish_pipeline_target(state, group, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let (hidden, boundary) =
            eredu_architectures::qwen::hybrid::ConditionalPipelineBoundary::from_prepared(prepared);
        Ok(PipelinePayload {
            hidden: hidden.into_array(),
            auxiliary: PipelineAuxiliaryState::new(
                self.boundary_schema()?
                    .encode(boundary)
                    .map_err(|error| Error::Parallel(error.to_string()))?
                    .into_iter()
                    .map(crate::MlxTensor::into_array)
                    .collect(),
            ),
        })
    }

    fn prefill(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        let mut state = self.begin_ingress(input, 0, execution, stream)?;
        let group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        if eredu_architectures::qwen::hybrid::ConditionalLayeredModel::<
                MlxNeuralBackend,
            >::pipeline_vision_active(&state)
            {
                self.execute_vision_state(&mut state, group, stream)?;
            }
        let prepared = self
            .architecture
            .finish_pipeline_target(state, group, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let (hidden, boundary) =
            eredu_architectures::qwen::hybrid::ConditionalPipelineBoundary::from_prepared(prepared);
        let payload = PipelinePayload {
            hidden: hidden.into_array(),
            auxiliary: PipelineAuxiliaryState::new(
                self.boundary_schema()?
                    .encode(boundary)
                    .map_err(|error| Error::Parallel(error.to_string()))?
                    .into_iter()
                    .map(crate::MlxTensor::into_array)
                    .collect(),
            ),
        };
        self.forward_decoder(
            PipelineStageInput::Hidden(&payload),
            step,
            mask,
            cache,
            execution,
            expert_group,
            stream,
            observer,
        )
    }
}

impl PipelineEmbeddedMtp for QwenConditionalPipelinePartition {
    fn embedded_mtp_len(&self) -> usize {
        self.prediction_layers.len()
    }

    fn embedded_mtp_state_segment(&self) -> Option<&'static str> {
        Some(eredu_architectures::qwen::hybrid::PREDICTION_STATE_SEGMENT)
    }

    fn prefill_token_identity(
        &self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        crate::composition::qwen::hybrid::prompt_token_ids(self.args(), input, stream)
            .map_err(Into::into)
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let layout = eredu_runtime::ArchitectureParameters::state_layout(&self.architecture)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let state = match paged {
            Some((manager, rank)) => MlxHybridState::paged(layout, manager, rank)?,
            None => MlxHybridState::device(layout)?,
        };
        Ok(PipelineMtpCache::Hybrid(state))
    }

    fn forward_embedded_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let PipelineMtpCache::Hybrid(cache) = cache else {
            return Err(Error::Parallel(
                "conditional Qwen3.5 pipeline MTP cache mismatch".into(),
            ));
        };
        self.forward_mtp_draft_neutral(
            hidden,
            tokens,
            depth,
            cache,
            execution,
            expert_group,
            stream,
        )
    }

    fn prefill_embedded_mtp_cache(
        &mut self,
        _output: &EmbeddedMtpOutput,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }

    fn fused_embedded_mtp_logits(
        &mut self,
        _hidden: &Array,
        _last_token: u32,
        _proposal_capacity: usize,
        _cache: &mut PipelineMtpCache,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        _stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        Ok(None)
    }

    fn adjust_fused_embedded_mtp_logits(
        &mut self,
        logits: Array,
        _last_token: u32,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(logits)
    }

    fn advance_embedded_mtp_cache(
        &mut self,
        _hidden: &Array,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }
}

impl PipelineForward for QwenConditionalPipelinePartition {
    pipeline_observed_forward!();
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_decoder(input, step, mask, cache, None, None, stream, None)
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_decoder(
            input,
            step,
            mask,
            cache,
            execution,
            expert_group,
            stream,
            observer,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_qwen_pipeline(
    source_args: eredu_architectures::qwen::ModelArgs,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    wire_contract: eredu_runtime::PipelineWireContract,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::Qwen2, ModelKind::Qwen3], "Qwen")?;
    let binding_adapter = if expert_cache_options.is_some() {
        crate::composition::qwen::QwenPipelineBindings::new_external_experts()
    } else {
        crate::composition::qwen::QwenPipelineBindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "Qwen pipeline",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_args = quantize_on_load.map_or_else(
        || Ok(source_args.clone()),
        |quantization| {
            eredu_architectures::qwen::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    let expert_quantization = quantize_on_load;
    let seed_architecture = eredu_architectures::qwen::RoutedLayeredModel::<MlxNeuralBackend>::new(
        target_args.clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = seed_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let decoder_group =
        architecture_decoder_group::<_, PipelineRangeState<'_>>(&seed_architecture)?;
    let target_units = architecture_group_unit_count(
        &binding_parameter_description,
        decoder_group,
        "Qwen decoder",
    )?;
    let seed_expert_realization = eredu_architectures::qwen::expert_realization_plan(
        &seed_architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    topology.preflight(
        Some(target_units),
        seed_expert_realization
            .as_ref()
            .map(eredu_architectures::ExpertRealizationPlan::global_expert_count),
    )?;
    let range = topology.layer_range(target_units)?;
    let mut stage = QwenPipelinePartition::new(
        seed_architecture,
        range.clone(),
        expert_cache_options.is_some(),
        stream,
    )?;
    let seed_architecture = stage
        .architecture
        .take()
        .expect("Qwen partition constructor owns a neutral architecture");
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry = eredu_architectures::qwen::local_geometry(&target_args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        stage.architecture = Some(
            eredu_architectures::qwen::RoutedLayeredModel::<MlxNeuralBackend>::new_parallel(
                target_args.clone(),
                geometry,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        );
        Some(layout)
    } else {
        stage.architecture = Some(seed_architecture);
        None
    };
    stage.expert_realization = eredu_architectures::qwen::expert_realization_plan(
        stage.architecture.as_ref().unwrap(),
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let placement = Arc::new(decoder_architecture_transport::<_, PipelineRangeState<'_>>(
        stage.architecture.as_ref().unwrap(),
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        wire_contract,
        range.clone(),
        placement,
        eredu_architectures::decoder::TEXT_DECODER_EXECUTION_GROUP,
        model_kind,
    );
    let expert_assignment =
        binding_adapter.expert_parallel_assignment(stage.expert_realization.as_ref())?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let geometry = stage
        .architecture
        .as_ref()
        .unwrap()
        .shared_parallel_geometry();
    let parameter_description = stage
        .architecture
        .as_ref()
        .unwrap()
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, PipelineRangeState<'_>, _, _, _>(
            stage.architecture.as_ref().unwrap(),
            info.pipeline_stage,
            geometry,
            &parameter_description,
        )?;
    let mut stage = stage.finish(partition);
    stage.layers = range
        .clone()
        .map(|global_layer| {
            construct_qwen_partition_unit(
                &stage.architecture,
                &stage.bindings,
                global_layer,
                stage.expert_realization.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let source_architecture = eredu_architectures::qwen::RoutedLayeredModel::<
                MlxNeuralBackend,
            >::new(source_args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &source_architecture);
            let target_quantization =
                BoundPipelineBindings::new(&stage.bindings, &stage.architecture);
            let decoder_group =
                architecture_decoder_group::<_, PipelineRangeState<'_>>(&stage.architecture)?;
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                stage.partition.parameter_bindings(),
                PipelineStageQuantizationSelection::new(
                    &static_roles,
                    decoder_group,
                    range.clone(),
                ),
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let quantize_on_load = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &stage.bindings
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    // Checkpoint bindings describe the global tensors. Build them against a
    // global architecture, then apply the rank-local layout while loading the
    // local architecture below.
    let binding_architecture =
        eredu_architectures::qwen::RoutedLayeredModel::<MlxNeuralBackend>::new(
            target_args.clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &binding_architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Qwen", &stage.partition);
    let decoder_group =
        architecture_decoder_group::<_, PipelineRangeState<'_>>(&stage.architecture)?;
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        quantize_on_load,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        for (global_layer, layer) in range.clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                decoder_group,
                global_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
                stream,
            )?;
            if expert_cache_options.is_some() {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, PipelineRangeState<'_>>(
                        &stage.architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, PipelineRangeState<'_>>(
                        &stage.architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_realization = stage.expert_realization.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let architecture = &stage.architecture;
        let bindings = &stage.bindings;
        let dense_layers = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if expert_cache_options.is_some() {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            range.clone(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                construct_qwen_partition_unit(
                    architecture,
                    bindings,
                    global_layer,
                    streamed_realization.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, _layer, store| {
                binding_adapter.cartesian_layer_bindings(
                    &binding_architecture,
                    decoder_group,
                    global_layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer| {
                architecture_parameter_unit_owner::<_, PipelineRangeState<'_>>(
                    architecture,
                    decoder_group,
                    global_layer,
                )
            },
        )?;
        stage.dense_layers = Some(dense_layers);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("Qwen pipeline planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let catalog =
            eredu_architectures::qwen::expert_residency_catalog(store.as_ref(), &source_args)
                .map_err(Error::ArchitectureModel)?;
        let units = crate::composition::select_architecture_expert_units(
            catalog,
            |group, unit| stage.partition.owns_unit(group.as_str(), unit),
            |identity| {
                stage.expert_assignment.as_ref().is_none_or(|assignment| {
                    assignment.owner(identity.global_expert) == Some(assignment.rank())
                })
            },
        );
        let entries = crate::composition::architecture_expert_units(
            units,
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        let cache = build_pipeline_expert_cache(
            Arc::clone(&store),
            entries,
            Some(options),
            expert_quantization,
            weights_stream,
            stream,
        )?;
        let owned_expert_bytes = cache.report()?.owned_bytes;
        info.planned_owned_parameter_bytes = info
            .planned_owned_parameter_bytes
            .checked_add(owned_expert_bytes)
            .ok_or_else(|| Error::Parallel("Qwen pipeline expert byte total overflowed".into()))?;
        stage.expert_cache = Some(cache);
    }
    let checkpoint_diagnostics = store.source_diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.payload_shard_paths.clone();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_neutral_qwen_vl_pipeline(
    source_args: eredu_architectures::qwen::vl::ModelArgs,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    wire_contract: eredu_runtime::PipelineWireContract,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(
        model_kind,
        &[ModelKind::Qwen3Vl, ModelKind::Qwen3VlMoe],
        "Qwen3-VL",
    )?;
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let external_experts = expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        QwenVlPipelineBindings::new_external_experts()
    } else {
        QwenVlPipelineBindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "Qwen3-VL pipeline",
                source_args.text.weight_quantization(),
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_args = quantize_on_load.map_or_else(
        || Ok(source_args.clone()),
        |quantization| {
            eredu_architectures::qwen::vl::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    let target_adapter = if external_experts {
        QwenVlPipelineBindings::new_external_experts()
    } else {
        QwenVlPipelineBindings::new()
    };
    let binding_architecture =
        eredu_architectures::qwen::vl::LayeredModel::new(target_args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut architecture =
        eredu_architectures::qwen::vl::LayeredModel::new(target_args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = binding_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let vision_group = architecture_group_by_id::<_, MlxHybridState>(
        &binding_architecture,
        eredu_architectures::qwen::vl::VISION_EXECUTION_GROUP,
    )?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&binding_architecture)?;
    let target_units = architecture_group_unit_count(
        &binding_parameter_description,
        decoder_group,
        "Qwen3-VL decoder",
    )?;
    let seed_expert_realization = eredu_architectures::qwen::vl::expert_realization_plan(
        &binding_architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    topology.preflight(
        Some(target_units),
        seed_expert_realization
            .as_ref()
            .map(eredu_architectures::ExpertRealizationPlan::global_expert_count),
    )?;
    let range = topology.layer_range(target_units)?;
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry = eredu_architectures::qwen::vl::local_geometry(&target_args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        architecture = eredu_architectures::qwen::vl::LayeredModel::new_parallel(
            target_args.clone(),
            geometry,
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Some(layout)
    } else {
        None
    };
    let placement = Arc::new(media_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        wire_contract,
        range.clone(),
        placement,
        eredu_architectures::qwen::vl::TEXT_EXECUTION_GROUP,
        model_kind,
    );
    let expert_realization = eredu_architectures::qwen::vl::expert_realization_plan(
        &architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_assignment =
        binding_adapter.expert_parallel_assignment(expert_realization.as_ref())?;
    if let Some(assignment) = expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            architecture.shared_parallel_geometry(),
            &parameter_description,
        )?;
    let mut stage = QwenVlPipelinePartition::new(architecture, partition, external_experts)?;
    stage.expert_assignment = expert_assignment;
    stage.vision_layers = stage
        .vision_range()
        .map(|index| {
            stage
                .architecture
                .construct_unit(vision_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    stage.layers = stage
        .range()
        .map(|index| {
            stage
                .architecture
                .construct_unit(decoder_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) =
        match quantize_on_load {
            Some(quantization) => {
                let source_architecture = eredu_architectures::qwen::vl::LayeredModel::<
                    MlxNeuralBackend,
                >::new(source_args.clone(), stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let source_quantization =
                    BoundPipelineBindings::new(&binding_adapter, &source_architecture);
                let target_quantization =
                    BoundPipelineBindings::new(&target_adapter, &binding_architecture);
                let (store, report) = quantize_pipeline_stage_store(
                    store,
                    &source_quantization,
                    &target_quantization,
                    stage.partition.parameter_bindings(),
                    PipelineStageQuantizationSelection::new(
                        &static_roles,
                        decoder_group,
                        stage.range().clone(),
                    )
                    .with_layer_group(vision_group, stage.vision_range().clone()),
                    quantization,
                    stream,
                )?;
                (store, Some(report))
            }
            None => (store, None),
        };
    let quantize_on_load = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &target_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &binding_architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Qwen3-VL", &stage.partition);
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        quantize_on_load,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        for (index, layer) in stage.vision_range().clone().zip(&mut stage.vision_layers) {
            let binding_layer = binding_architecture
                .construct_unit(vision_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                vision_group,
                index,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                None,
            )?;
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    &stage.architecture,
                    vision_group,
                    index,
                )?,
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.range().clone().zip(&mut stage.layers) {
            let binding_layer = binding_architecture
                .construct_unit(decoder_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                decoder_group,
                index,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
            )?;
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        &stage.architecture,
                        decoder_group,
                        index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        &stage.architecture,
                        decoder_group,
                        index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    quantize_on_load,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    let diagnostics = store.source_diagnostics()?;
    if let Some(options) = dense_stream {
        let layout = parallel_layout.clone();
        let assignment = stage.expert_assignment.clone();
        let adapter = &stage.adapter;
        let architecture = &stage.architecture;
        let streamed_units = stage
            .partition
            .units()
            .filter(|address| {
                <eredu_architectures::qwen::vl::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::group_transport(architecture, address.group())
                .placement
                    == eredu_runtime::ArchitectureGroupPlacement::Pipeline
            })
            .collect::<Vec<_>>();
        let execution_offset = streamed_units
            .iter()
            .position(|address| address.group() == decoder_group)
            .unwrap_or(streamed_units.len());
        let dense = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if external_experts {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            0..streamed_units.len(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                let address = streamed_units[ordinal];
                architecture
                    .construct_unit(address.group(), address.index(), stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            },
            |ordinal, _layer, store| {
                let address = streamed_units[ordinal];
                let binding_layer = binding_architecture
                    .construct_unit(address.group(), address.index(), stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                adapter.cartesian_layer_bindings(
                    &binding_architecture,
                    address.group(),
                    address.index(),
                    &binding_layer,
                    store,
                    layout.as_ref(),
                    assignment.as_ref(),
                )
            },
            |ordinal| {
                let address = streamed_units[ordinal];
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    address.group(),
                    address.index(),
                )
            },
        )?
        .with_execution_offset(execution_offset)?;
        stage.dense_layers = Some(dense);
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?)
            .ok_or_else(|| Error::Parallel("Qwen3-VL planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let catalog =
            eredu_architectures::qwen::expert_residency_catalog(store.as_ref(), &source_args.text)
                .map_err(Error::ArchitectureModel)?;
        let units = crate::composition::select_architecture_expert_units(
            catalog,
            |group, unit| stage.partition.owns_unit(group.as_str(), unit),
            |identity| {
                stage.expert_assignment.as_ref().is_none_or(|assignment| {
                    assignment.owner(identity.global_expert) == Some(assignment.rank())
                })
            },
        );
        let entries = crate::composition::architecture_expert_units(
            units,
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
        let cache = build_pipeline_expert_cache(
            Arc::clone(&store),
            entries,
            Some(options),
            quantize_on_load,
            weights_stream,
            stream,
        )?;
        info.planned_owned_parameter_bytes = info
            .planned_owned_parameter_bytes
            .checked_add(cache.report()?.owned_bytes)
            .ok_or_else(|| Error::Parallel("Qwen3-VL expert bytes overflowed".into()))?;
        stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
    }
    let mut materialized_shards = if info.materialization.is_some() {
        store.materialized_source_shards()
    } else {
        Vec::new()
    };
    materialized_shards.extend(checkpoint_backing_shards(
        store.as_ref(),
        info.owned_tensors.iter().map(String::as_str),
    )?);
    if dense_stream.is_some() {
        materialized_shards.extend(checkpoint_unit_backing_shards::<_, MlxHybridState>(
            store.as_ref(),
            &stage.architecture,
            decoder_group,
            stage.range().clone(),
        )?);
    }
    materialized_shards.sort();
    materialized_shards.dedup();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

impl QwenPipelinePartition {
    fn args(&self) -> &eredu_architectures::qwen::ModelArgs {
        self.architecture.args()
    }

    fn new(
        architecture: eredu_architectures::qwen::RoutedLayeredModel<MlxNeuralBackend>,
        range: Range<usize>,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<
        DecoderPipelineBuilder<
            eredu_architectures::qwen::RoutedLayeredModel<MlxNeuralBackend>,
            eredu_architectures::qwen::LocalGeometry,
            crate::composition::qwen::QwenPipelineBindings,
            MlxModule<eredu_architectures::qwen::RoutedTransformerBlock<MlxNeuralBackend>>,
        >,
        Error,
    > {
        let bindings = if external_experts {
            crate::composition::qwen::QwenPipelineBindings::new_external_experts()
        } else {
            crate::composition::qwen::QwenPipelineBindings::new()
        };
        let layers = range
            .clone()
            .map(|layer| {
                architecture
                    .construct_unit(layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DecoderPipelineBuilder {
            architecture: Some(architecture),
            bindings,
            layers,
            dense_layers: None,
            expert_realization: None,
            expert_assignment: None,
            expert_cache: None,
            routing_statistics: RoutingStatistics::default(),
            _geometry: std::marker::PhantomData,
        })
    }
}

impl QwenPipelinePartition {
    #[allow(clippy::too_many_arguments)]
    fn forward_resident_experts_neutral(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: &Group,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self.expert_assignment.clone().ok_or_else(|| {
            Error::Parallel("resident Qwen experts have no rank-local assignment".into())
        })?;
        validate_pipeline_expert_dispatch(&assignment, Some(expert_group), false)?;
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut execute =
            |bank: &mut <MlxNeuralBackend as RoutedNeuralBackend>::GatedProductExpertBank,
             hidden: &Array,
             ids: &Array,
             weights: &Array,
             partitions: usize,
             context: &Stream| {
                execute_resident_distributed_experts(
                    bank,
                    hidden,
                    ids,
                    weights,
                    partitions,
                    &assignment,
                    expert_group,
                    &mut statistics,
                    context,
                )
            };
        let mut provider = ResidentExpertExecutorProvider::new(&mut execute);
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let result = execute_neutral_routed_decoder_partition_observed(
            self,
            input,
            step,
            explicit_mask,
            caches,
            execution,
            pass,
            &mut provider,
            stream,
            observer,
        );
        drop(provider);
        self.routing_statistics = statistics;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_external_experts_neutral(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        let assignment = self.expert_assignment.clone().ok_or_else(|| {
            Error::Parallel("external Qwen experts have no rank-local assignment".into())
        })?;
        validate_pipeline_expert_dispatch(&assignment, expert_group, true)?;
        let cache = self
            .expert_cache
            .take()
            .ok_or_else(|| Error::Parallel("external Qwen expert cache is unavailable".into()))?;
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut execute = |execution: GatedProductExpertExecution, context: &Stream| {
            execute_pipeline_cached_qwen3(
                &execution.spec,
                execution.layer,
                &execution.hidden,
                &execution.expert_ids,
                &execution.route_weights,
                pass,
                &cache,
                &assignment,
                expert_group,
                tensor_group,
                &mut statistics,
                context,
            )
            .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
            .map_err(|error| Exception::custom(error.to_string()))
        };
        let mut provider = GatedProductExpertExecutorProvider::new(&mut execute);
        let result = execute_neutral_routed_decoder_partition_observed(
            self,
            input,
            step,
            explicit_mask,
            caches,
            execution,
            pass,
            &mut provider,
            stream,
            observer,
        );
        self.routing_statistics = statistics;
        self.expert_cache = Some(cache);
        result
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_qwen3(
    spec: &eredu_nn::GatedProductExpertBankSpec,
    global_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    _tensor_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let execute = |routes: &crate::backend::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        crate::composition::mlx::distributed::expert::execute_cached_gated_product(
            spec,
            global_layer,
            routes,
            pass,
            cache,
            stream,
        )
    };
    let returned = match expert_group {
        Some(group) => dispatch_replicated_with(
            hidden, expert_ids, weights, assignment, group, stream, execute,
        )?,
        None => dispatch_local_with(hidden, expert_ids, weights, assignment, stream, execute)?,
    };
    statistics.accumulate(&returned.statistics);
    Ok(returned.reduced_output)
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_neutral_qwen_hybrid(
    spec: &eredu_nn::GatedProductExpertBankSpec,
    global_layer: usize,
    hidden: &Array,
    expert_ids: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &ExpertCache,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let execute = |routes: &crate::backend::runtime::distributed::expert::DispatchedRoutes,
                   stream: &Stream| {
        crate::composition::mlx::distributed::expert::execute_cached_gated_product(
            spec,
            global_layer,
            routes,
            pass,
            cache,
            stream,
        )
    };
    let returned = match expert_group {
        Some(group) => dispatch_replicated_with(
            hidden, expert_ids, weights, assignment, group, stream, execute,
        )?,
        None => dispatch_local_with(hidden, expert_ids, weights, assignment, stream, execute)?,
    };
    statistics.accumulate(&returned.statistics);
    Ok(returned.reduced_output)
}

impl QwenHybridPipelinePartition {
    fn args(&self) -> &eredu_architectures::qwen::hybrid::HybridConfig {
        self.architecture.config()
    }

    fn new(
        architecture: eredu_architectures::qwen::hybrid::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Option<Arc<eredu_architectures::qwen::hybrid::LocalGeometry>>,
            eredu_runtime::NoAuxiliaryBoundarySchema,
        >,
        external_experts: bool,
    ) -> Result<Self, Error> {
        Ok(Self {
            architecture,
            partition,
            layers: Vec::new(),
            prediction_layers: Vec::new(),
            dense_layers: None,
            expert_assignment: None,
            expert_storage: if external_experts {
                PipelineExpertStorage::ExternalEmpty
            } else {
                PipelineExpertStorage::LayerLocal
            },
            routing_statistics: RoutingStatistics::default(),
        })
    }

    fn forward_target(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        explicit_mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "Qwen hybrid stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        if let Some(assignment) = self.expert_assignment.as_ref() {
            validate_pipeline_expert_dispatch(
                assignment,
                expert_group,
                self.expert_storage.is_external(),
            )?;
        }
        self.routing_statistics = RoutingStatistics::default();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let assignment = self.expert_assignment.clone();
        let expert_cache = self.expert_storage.cache();
        let decoder_range = self.range();
        if let Some(expert_cache) = expert_cache {
            let assignment = assignment.as_ref().ok_or_else(|| {
                Error::Parallel("Qwen hybrid external experts have no assignment".into())
            })?;
            let mut execute = |execution: GatedProductExpertExecution, stream: &Stream| {
                execute_pipeline_cached_neutral_qwen_hybrid(
                    &execution.spec,
                    execution.layer,
                    &execution.hidden,
                    &execution.expert_ids,
                    &execution.route_weights,
                    pass,
                    expert_cache,
                    assignment,
                    expert_group,
                    &mut self.routing_statistics,
                    stream,
                )
                .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
                .map_err(|error| Exception::custom(error.to_string()))
            };
            let mut provider = GatedProductExpertExecutorProvider::new(&mut execute);
            execute_routed_layered_partition_observed(
                &mut self.architecture,
                &self.partition,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                input,
                step,
                explicit_mask,
                caches,
                execution,
                pass,
                &mut provider,
                stream,
                observer,
            )
        } else {
            let mut provider = eredu_runtime::ResidentExpertProvider;
            execute_routed_layered_partition_observed(
                &mut self.architecture,
                &self.partition,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                input,
                step,
                explicit_mask,
                caches,
                execution,
                pass,
                &mut provider,
                stream,
                observer,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_mtp_draft_neutral(
        &mut self,
        prior: &Array,
        tokens: &Array,
        depth: usize,
        state: &mut MlxHybridState,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let units = self
            .prediction_layers
            .get_mut(depth)
            .ok_or_else(|| Error::Parallel(format!("Qwen hybrid has no MTP depth {depth}")))?;
        let prediction_group =
            architecture_prediction_group::<_, MlxHybridState>(&self.architecture, depth)?;
        let mut execute = |execution: GatedProductExpertExecution, stream: &Stream| {
            let cache = self.expert_storage.cache().ok_or_else(|| {
                Exception::custom("Qwen hybrid MTP external expert cache is unavailable")
            })?;
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Exception::custom("Qwen hybrid MTP external experts have no assignment")
            })?;
            execute_pipeline_cached_neutral_qwen_hybrid(
                &execution.spec,
                execution.layer,
                &execution.hidden,
                &execution.expert_ids,
                &execution.route_weights,
                ExpertPass::Decode,
                cache,
                assignment,
                expert_group,
                &mut self.routing_statistics,
                stream,
            )
            .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
            .map_err(|error| Exception::custom(error.to_string()))
        };
        let input = eredu_architectures::qwen::hybrid::EmbeddedInput::Draft {
            tokens: crate::composition::tensor_ref(tokens),
            hidden: crate::composition::tensor_ref(prior),
            depth,
        };
        let (logits, hidden) = if self.expert_storage.cache().is_some() {
            let mut provider = GatedProductExpertExecutorProvider::new(&mut execute);
            execute_neutral_routed_output_group(
                &mut self.architecture,
                input,
                prediction_group,
                units,
                state,
                ExpertPass::Decode,
                &mut provider,
                tensor_group,
                stream,
            )
        } else {
            execute_neutral_routed_output_group(
                &mut self.architecture,
                input,
                prediction_group,
                units,
                state,
                ExpertPass::Decode,
                &mut eredu_runtime::ResidentExpertProvider,
                tensor_group,
                stream,
            )
        }?;
        Ok(EmbeddedMtpOutput {
            logits,
            hidden,
            tokens: crate::MlxTensor::from_array(tokens.clone()),
        })
    }
}

impl PipelinePartitionMetadata for QwenHybridPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::qwen_hybrid_text(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::qwen_hybrid_text_input_part(
            self.args(),
            input,
            &crate::backend::runtime::media::input::MlxInputInspector,
        )
        .map(Into::into)
    }

    fn boundary_wire_schema(&self) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
        self.partition
            .boundary_schema()
            .wire_schema()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn dense_layers(&self) -> Option<&PipelineLayerStorage> {
        self.dense_layers.as_ref()
    }

    fn expert_cache(&self) -> Option<&ExpertCache> {
        self.expert_storage.cache()
    }

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        let target_identity = identity
            .select_state_segment(eredu_architectures::qwen::hybrid::TARGET_STATE_SEGMENT)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        materialize_pipeline_cache_layers(&target_identity, paged)
    }
}

impl PipelineEmbeddedMtp for QwenHybridPipelinePartition {
    fn embedded_mtp_len(&self) -> usize {
        self.prediction_layers.len()
    }

    fn embedded_mtp_state_segment(&self) -> Option<&'static str> {
        Some(eredu_architectures::qwen::hybrid::PREDICTION_STATE_SEGMENT)
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let layout = eredu_runtime::ArchitectureParameters::state_layout(&self.architecture)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let state = match paged {
            Some((manager, rank)) => MlxHybridState::paged(layout, manager, rank)?,
            None => MlxHybridState::device(layout)?,
        };
        Ok(PipelineMtpCache::Hybrid(state))
    }

    fn forward_embedded_mtp_draft(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let PipelineMtpCache::Hybrid(cache) = cache else {
            return Err(Error::Parallel(
                "Qwen hybrid pipeline MTP cache mismatch".into(),
            ));
        };
        self.forward_mtp_draft_neutral(
            hidden,
            tokens,
            depth,
            cache,
            execution,
            expert_group,
            stream,
        )
    }

    fn prefill_embedded_mtp_cache(
        &mut self,
        _output: &EmbeddedMtpOutput,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }

    fn fused_embedded_mtp_logits(
        &mut self,
        _hidden: &Array,
        _last_token: u32,
        _proposal_capacity: usize,
        _cache: &mut PipelineMtpCache,
        _execution: Option<&ParallelExecutionContext<'_>>,
        _expert_group: Option<&Group>,
        _stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        Ok(None)
    }

    fn adjust_fused_embedded_mtp_logits(
        &mut self,
        logits: Array,
        _last_token: u32,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(logits)
    }

    fn advance_embedded_mtp_cache(
        &mut self,
        _hidden: &Array,
        _tokens: &Array,
        _cache: &mut PipelineMtpCache,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }
}

impl PipelineForward for QwenHybridPipelinePartition {
    pipeline_observed_forward!();
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_target(input, step, mask, cache, None, None, stream, None)
    }

    fn forward_with_execution(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_target(
            input,
            step,
            mask,
            cache,
            execution,
            expert_group,
            stream,
            observer,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_neutral_qwen_hybrid_pipeline(
    source_args: eredu_architectures::qwen::hybrid::HybridConfig,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    wire_contract: eredu_runtime::PipelineWireContract,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(
        model_kind,
        &[ModelKind::Qwen3Next, ModelKind::Qwen35],
        "Qwen hybrid",
    )?;
    let explicit_expert_cache = expert_cache_options.is_some();
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let external_experts = expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        QwenHybridPipelineBindings::new_external_experts()
    } else {
        QwenHybridPipelineBindings::new()
    };
    if requested_quantization.is_some() && source_args.fp8.is_some() {
        return Err(Error::Quantization(
            "Qwen hybrid pipeline cannot implicitly transcode checkpoint-native FP8 weights".into(),
        ));
    }
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "Qwen hybrid pipeline",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_args = quantize_on_load.map_or_else(
        || Ok(source_args.clone()),
        |quantization| {
            eredu_architectures::qwen::hybrid::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    let target_binding_adapter = if external_experts {
        QwenHybridPipelineBindings::new_external_experts()
    } else {
        QwenHybridPipelineBindings::new()
    };
    let binding_architecture =
        eredu_architectures::qwen::hybrid::LayeredModel::<MlxNeuralBackend>::new(
            target_args.clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut architecture =
        eredu_architectures::qwen::hybrid::LayeredModel::<MlxNeuralBackend>::new(
            target_args.clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = binding_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&binding_architecture)?;
    let target_units = binding_parameter_description
        .unit_layout()
        .group_range(decoder_group)
        .ok_or_else(|| Error::Parallel("Qwen hybrid parameter plan has no target group".into()))?
        .len();
    let seed_expert_realization = eredu_architectures::qwen::hybrid::expert_realization_plan(
        &binding_architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    topology.preflight(
        Some(target_units),
        seed_expert_realization
            .as_ref()
            .map(eredu_architectures::ExpertRealizationPlan::global_expert_count),
    )?;
    let range = topology.layer_range(target_units)?;
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry = eredu_architectures::qwen::hybrid::local_geometry(&target_args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        architecture =
            eredu_architectures::qwen::hybrid::LayeredModel::<MlxNeuralBackend>::new_parallel(
                target_args.clone(),
                geometry,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Some(layout)
    } else {
        None
    };
    let placement = Arc::new(prediction_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        wire_contract,
        range.clone(),
        placement,
        eredu_architectures::decoder::TARGET_EXECUTION_GROUP,
        model_kind,
    );
    let expert_realization = eredu_architectures::qwen::hybrid::expert_realization_plan(
        &architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_assignment =
        binding_adapter.expert_parallel_assignment(expert_realization.as_ref())?;
    if let Some(assignment) = expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let geometry = architecture.shared_parallel_geometry();
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            geometry,
            &parameter_description,
        )?;
    let mut stage = QwenHybridPipelinePartition::new(architecture, partition, external_experts)?;
    stage.expert_assignment = expert_assignment;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&stage.architecture)?;
    let prediction_units = architecture_single_prediction_units::<_, MlxHybridState>(
        &stage.architecture,
        &parameter_description,
    )?;
    stage.layers = range
        .clone()
        .map(|global_layer| {
            stage
                .architecture
                .construct_unit(decoder_group, global_layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let owns_mtp = partition_owns_architecture_units(
        &stage.partition,
        prediction_units
            .iter()
            .map(|(group, index)| (*group, *index..*index + 1)),
    );
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp { prediction_units.len() } else { 0 };
    info.global_embedded_mtp_layers = prediction_units.len();
    if owns_mtp {
        for &(group, index) in &prediction_units {
            let unit = <eredu_architectures::qwen::hybrid::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::build_unit(&stage.architecture, group, index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            stage.prediction_layers.push(vec![unit]);
        }
    }
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let mut selection = PipelineStageQuantizationSelection::new(
                &static_roles,
                decoder_group,
                stage.range().clone(),
            );
            if owns_mtp {
                for &(group, index) in &prediction_units {
                    selection = selection.with_layer_group(group, index..index + 1);
                }
            }
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &binding_architecture);
            let target_quantization =
                BoundPipelineBindings::new(&target_binding_adapter, &binding_architecture);
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                stage.partition.parameter_bindings(),
                selection,
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let expert_quantization = quantize_on_load;
    let requested = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &target_binding_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &binding_architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Qwen hybrid", &stage.partition);
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        requested,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        let architecture = &stage.architecture;
        for (global_layer, layer) in stage.range().clone().zip(&mut stage.layers) {
            let binding_layer = binding_architecture
                .construct_unit(decoder_group, global_layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                decoder_group,
                global_layer,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
            )?;
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    if owns_mtp {
        let architecture = &stage.architecture;
        for (&(prediction_group, prediction_index), layers) in
            prediction_units.iter().zip(&mut stage.prediction_layers)
        {
            let layer = &mut layers[0];
            let binding_layer = binding_architecture
                .construct_unit(prediction_group, prediction_index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                prediction_group,
                prediction_index,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
            )?;
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        prediction_group,
                        prediction_index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        prediction_group,
                        prediction_index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    let checkpoint_diagnostics_before_deferred = store.source_diagnostics()?;
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let streamed_architecture = &stage.architecture;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if external_experts {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            stage.range().clone(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_architecture
                    .construct_unit(decoder_group, global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            },
            |global_layer, _layer, store| {
                let binding_layer = binding_architecture
                    .construct_unit(decoder_group, global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                binding_adapter.cartesian_layer_bindings(
                    &binding_architecture,
                    decoder_group,
                    global_layer,
                    &binding_layer,
                    store,
                    streamed_layout.as_ref(),
                    streamed_assignment.as_ref(),
                )
            },
            |global_layer| {
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    streamed_architecture,
                    decoder_group,
                    global_layer,
                )
            },
        )?);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("Qwen hybrid pipeline bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let entries = crate::composition::qwen::hybrid::expert_catalog_selected(
            &source_args,
            store.as_ref(),
            parallel_layout.as_ref(),
            |group, unit| stage.partition.owns_unit(group.as_str(), unit),
        )?
        .into_iter()
        .filter(|entry| {
            stage.expert_assignment.as_ref().is_none_or(|assignment| {
                assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
            })
        })
        .collect::<Vec<_>>();
        if !entries.is_empty() {
            let cache = build_pipeline_expert_cache(
                Arc::clone(&store),
                entries,
                expert_cache_options,
                expert_quantization,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes)
                .ok_or_else(|| Error::Parallel("Qwen hybrid expert bytes overflowed".into()))?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    let checkpoint_diagnostics = if explicit_expert_cache {
        store.source_diagnostics()?
    } else {
        checkpoint_diagnostics_before_deferred
    };
    let mut materialized_shards = if info.materialization.is_some() {
        store.materialized_source_shards()
    } else {
        Vec::new()
    };
    materialized_shards.extend(checkpoint_backing_shards(
        store.as_ref(),
        info.owned_tensors.iter().map(String::as_str),
    )?);
    if dense_stream.is_some() {
        materialized_shards.extend(checkpoint_unit_backing_shards::<_, MlxHybridState>(
            store.as_ref(),
            &stage.architecture,
            decoder_group,
            stage.range().clone(),
        )?);
    }
    materialized_shards.sort();
    materialized_shards.dedup();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_neutral_qwen_conditional_pipeline(
    source: eredu_architectures::qwen::hybrid::ParsedHybridConfig,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    wire_contract: eredu_runtime::PipelineWireContract,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    expert_cache_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::Qwen35], "conditional Qwen3.5")?;
    let explicit_expert_cache = expert_cache_options.is_some();
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let external_experts = expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        QwenConditionalPipelineBindings::new_external_experts()
    } else {
        QwenConditionalPipelineBindings::new()
    };
    if requested_quantization.is_some() && source.text.fp8.is_some() {
        return Err(Error::Quantization(
            "conditional Qwen3.5 pipeline cannot implicitly transcode checkpoint-native FP8 weights"
                .into(),
        ));
    }
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "conditional Qwen3.5 pipeline",
                source.text.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target = quantize_on_load.map_or_else(
        || Ok(source.clone()),
        |quantization| {
            eredu_architectures::qwen::hybrid::conditional_load_time_quantization(
                &source,
                quantization,
            )
            .map_err(Error::ArchitectureModel)
        },
    )?;
    let target_adapter = if external_experts {
        QwenConditionalPipelineBindings::new_external_experts()
    } else {
        QwenConditionalPipelineBindings::new()
    };
    let binding_architecture =
        eredu_architectures::qwen::hybrid::ConditionalLayeredModel::new(target.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut architecture =
        eredu_architectures::qwen::hybrid::ConditionalLayeredModel::new(target.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = binding_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let vision_group = architecture_group_by_id::<_, MlxHybridState>(
        &binding_architecture,
        eredu_architectures::qwen::hybrid::VISION_EXECUTION_GROUP,
    )?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&binding_architecture)?;
    let target_units = binding_parameter_description
        .unit_layout()
        .group_range(decoder_group)
        .ok_or_else(|| {
            Error::Parallel("conditional Qwen parameter plan has no target group".into())
        })?
        .len();
    let seed_expert_realization =
        eredu_architectures::qwen::hybrid::conditional_expert_realization_plan(
            &binding_architecture,
            topology.rank_topology(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    topology.preflight(
        Some(target_units),
        seed_expert_realization
            .as_ref()
            .map(eredu_architectures::ExpertRealizationPlan::global_expert_count),
    )?;
    let range = topology.layer_range(target_units)?;
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry =
            eredu_architectures::qwen::hybrid::conditional_local_geometry(&target, &layout)
                .map_err(|error| Error::Parallel(error.to_string()))?;
        architecture = eredu_architectures::qwen::hybrid::ConditionalLayeredModel::<
                MlxNeuralBackend,
            >::new_parallel(target.clone(), geometry, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Some(layout)
    } else {
        None
    };
    let placement = Arc::new(media_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        wire_contract,
        range.clone(),
        placement,
        eredu_architectures::decoder::TARGET_EXECUTION_GROUP,
        model_kind,
    );
    let expert_realization =
        eredu_architectures::qwen::hybrid::conditional_expert_realization_plan(
            &architecture,
            topology.rank_topology(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_assignment =
        binding_adapter.expert_parallel_assignment(expert_realization.as_ref())?;
    if let Some(assignment) = expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            architecture.shared_parallel_geometry(),
            &parameter_description,
        )?;
    let mut stage =
        QwenConditionalPipelinePartition::new(architecture, partition, external_experts)?;
    stage.expert_assignment = expert_assignment;
    let prediction_units = architecture_single_prediction_units::<_, MlxHybridState>(
        &stage.architecture,
        &parameter_description,
    )?;
    stage.vision_layers = stage
        .vision_range()
        .map(|index| {
            stage
                .architecture
                .construct_unit(vision_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    stage.layers = stage
        .range()
        .map(|index| {
            stage
                .architecture
                .construct_unit(decoder_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let owns_mtp = partition_owns_architecture_units(
        &stage.partition,
        prediction_units
            .iter()
            .map(|(group, index)| (*group, *index..*index + 1)),
    );
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp { prediction_units.len() } else { 0 };
    info.global_embedded_mtp_layers = prediction_units.len();
    if owns_mtp {
        for &(prediction_group, prediction_index) in &prediction_units {
            let unit = stage
                .architecture
                .construct_unit(prediction_group, prediction_index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            stage.prediction_layers.push(vec![unit]);
        }
    }
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let mut selection = PipelineStageQuantizationSelection::new(
                &static_roles,
                decoder_group,
                stage.range().clone(),
            )
            .with_layer_group(vision_group, stage.vision_range().clone());
            if owns_mtp {
                for &(prediction_group, prediction_index) in &prediction_units {
                    selection = selection
                        .with_layer_group(prediction_group, prediction_index..prediction_index + 1);
                }
            }
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &binding_architecture);
            let target_quantization =
                BoundPipelineBindings::new(&target_adapter, &binding_architecture);
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                stage.partition.parameter_bindings(),
                selection,
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let expert_quantization = quantize_on_load;
    let requested = materialization
        .is_none()
        .then_some(quantize_on_load)
        .flatten();
    let binding_adapter = if materialization.is_some() {
        &target_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &binding_architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("conditional Qwen3.5", &stage.partition);
    load_architecture_static_parameters(
        &mut stage.architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        requested,
        weights_stream,
        stream,
    )?;
    if dense_stream.is_none() {
        let architecture = &stage.architecture;
        for (index, layer) in stage.vision_range().clone().zip(&mut stage.vision_layers) {
            let binding_layer = binding_architecture
                .construct_unit(vision_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                vision_group,
                index,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    vision_group,
                    index,
                )?,
                layer,
                store.as_ref(),
                &bindings,
                requested,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.range().clone().zip(&mut stage.layers) {
            let binding_layer = binding_architecture
                .construct_unit(decoder_group, index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                decoder_group,
                index,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        decoder_group,
                        index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    if owns_mtp {
        let architecture = &stage.architecture;
        for (&(prediction_group, prediction_index), layers) in
            prediction_units.iter().zip(&mut stage.prediction_layers)
        {
            let layer = &mut layers[0];
            let binding_layer = binding_architecture
                .construct_unit(prediction_group, prediction_index, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &binding_architecture,
                prediction_group,
                prediction_index,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
            )?;
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        prediction_group,
                        prediction_index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        architecture,
                        prediction_group,
                        prediction_index,
                    )?,
                    layer,
                    store.as_ref(),
                    &bindings,
                    requested,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    let diagnostics_before_deferred = store.source_diagnostics()?;
    if let Some(options) = dense_stream {
        let layout = parallel_layout.clone();
        let adapter = &stage.adapter;
        let streamed_architecture = &stage.architecture;
        let streamed_units = stage
            .partition
            .units()
            .filter(|address| {
                <eredu_architectures::qwen::hybrid::ConditionalLayeredModel<
                    MlxNeuralBackend,
                > as LayeredArchitecture<MlxNeuralBackend, MlxHybridState>>::group_transport(
                    streamed_architecture,
                    address.group(),
                )
                .placement
                    == eredu_runtime::ArchitectureGroupPlacement::Pipeline
            })
            .collect::<Vec<_>>();
        let execution_offset = streamed_units
            .iter()
            .position(|address| address.group() == decoder_group)
            .ok_or_else(|| {
                Error::Parallel("conditional Qwen partition traversal has no target unit".into())
            })?;
        let dense = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if external_experts {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            0..streamed_units.len(),
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                let address = streamed_units[ordinal];
                streamed_architecture
                    .construct_unit(address.group(), address.index(), stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            },
            |ordinal, _layer, store| {
                let address = streamed_units[ordinal];
                let binding_layer = binding_architecture
                    .construct_unit(address.group(), address.index(), stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                adapter.cartesian_layer_bindings(
                    &binding_architecture,
                    address.group(),
                    address.index(),
                    &binding_layer,
                    store,
                    layout.as_ref(),
                )
            },
            |ordinal| {
                let address = streamed_units[ordinal];
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    streamed_architecture,
                    address.group(),
                    address.index(),
                )
            },
        )?
        .with_execution_offset(execution_offset)?;
        stage.dense_layers = Some(dense);
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?)
            .ok_or_else(|| Error::Parallel("conditional Qwen3.5 bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let entries = crate::composition::qwen::hybrid::expert_catalog_selected(
            &source.text,
            store.as_ref(),
            parallel_layout.as_ref(),
            |group, unit| stage.partition.owns_unit(group.as_str(), unit),
        )?
        .into_iter()
        .filter(|entry| {
            stage.expert_assignment.as_ref().is_none_or(|assignment| {
                assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
            })
        })
        .collect::<Vec<_>>();
        if !entries.is_empty() {
            let cache = build_pipeline_expert_cache(
                Arc::clone(&store),
                entries,
                expert_cache_options,
                expert_quantization,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes)
                .ok_or_else(|| {
                    Error::Parallel("conditional Qwen3.5 expert bytes overflowed".into())
                })?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    let diagnostics = if explicit_expert_cache {
        store.source_diagnostics()?
    } else {
        diagnostics_before_deferred
    };
    let mut materialized_shards = if info.materialization.is_some() {
        store.materialized_source_shards()
    } else {
        Vec::new()
    };
    materialized_shards.extend(checkpoint_backing_shards(
        store.as_ref(),
        info.owned_tensors.iter().map(String::as_str),
    )?);
    if dense_stream.is_some() {
        materialized_shards.extend(checkpoint_unit_backing_shards::<_, MlxHybridState>(
            store.as_ref(),
            &stage.architecture,
            decoder_group,
            stage.range().clone(),
        )?);
    }
    materialized_shards.sort();
    materialized_shards.dedup();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}
