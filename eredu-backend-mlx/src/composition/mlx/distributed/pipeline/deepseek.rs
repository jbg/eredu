use std::{ops::Range, sync::Arc};

use crate::backend::runtime::distributed::Group;
use eredu_architectures::ModelKind;
use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
use eredu_core::cache::{CacheRankIdentity, PromptCacheModelIdentity};
use eredu_runtime::{ArchitectureBoundary, ExpertCacheLoadOptions, ExpertPass, WeightBinding};
use safemlx::{error::Exception, Array, Stream};

use crate::{
    backend::{
        error::Error,
        nn::shared::{MlxModule, MlxNeuralBackend},
        runtime::{
            cache::{
                kv::CompressedLatentCache,
                residency::CacheResidencyManager,
                state::{MlxHybridState, MlxPoolingAttentionCache},
            },
            checkpoint::binding::build_module_bindings,
            distributed::{
                completion::synchronize_outputs,
                expert::{
                    dispatch_local_with, dispatch_replicated_with, ExpertAssignment,
                    RoutingStatistics,
                },
                parallel::{routed_expert_intermediate_range, ParallelExecutionContext},
            },
            execution::layerwise::shard_layer_bindings,
            residency::{
                expert_cache::ExpertCache,
                expert_provider::{
                    GatedProductExpertExecution, GatedProductExpertExecutorProvider,
                },
            },
        },
        MlxParallelContext,
    },
    composition::mlx::distributed::pipeline::{
        architecture_decoder_group, architecture_parallel_layout,
        architecture_parameter_unit_owner, architecture_prediction_group,
        architecture_single_prediction_units, base_info, build_pipeline_expert_cache,
        build_pipeline_layer_storage, execute_routed_layered_partition_observed,
        load_architecture_static_parameters, localized_gated_expert_width,
        partition_owns_architecture_units, prediction_architecture_transport,
        split_static_binding_units_by_owner, validate_admitted_pipeline_kind,
        validate_pipeline_expert_dispatch, DeepSeekV3PipelinePartition,
        DeepSeekV4PipelinePartition, NeutralV3Architecture, NeutralV4Architecture,
        PipelineEmbeddedMtp, PipelineExpertStorage, PipelineForward, PipelineLayerCache,
        PipelineLayerLoadOptions, PipelineLayerStorage, PipelineLoadAccumulator, PipelineModel,
        PipelineMtpCache, PipelinePartitionMetadata, PipelineRangeState, PipelineStageInput,
        PipelineStageOutput, PipelineStep,
    },
    composition::mlx::speculative::embedded::EmbeddedMtpOutput,
};

impl DeepSeekV3PipelinePartition {
    fn args(&self) -> &eredu_architectures::deepseek::V3Args {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        let group = architecture_decoder_group::<_, PipelineRangeState<'_>>(&self.architecture)
            .expect("validated DeepSeek V3 decoder group");
        self.partition
            .groups()
            .iter()
            .find(|owned| owned.group_index() == group)
            .map(|owned| owned.global_units())
            .unwrap_or(0..0)
    }

    fn forward_stage(
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
                "neutral DeepSeek V3 stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let decoder_range = self.range();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.routing_statistics = RoutingStatistics::default();
        let expert_cache = self.expert_storage.cache();
        let assignment = self.expert_assignment.as_ref();
        let statistics = &mut self.routing_statistics;
        if let Some(expert_cache) = expert_cache {
            let assignment = assignment.ok_or_else(|| {
                Error::Parallel("neutral DeepSeek V3 external experts have no assignment".into())
            })?;
            let mut execute = |execution: GatedProductExpertExecution, context: &Stream| {
                execute_pipeline_cached_neutral_deepseek(
                    &execution.spec,
                    execution.layer,
                    &execution.hidden,
                    &execution.expert_ids,
                    &execution.route_weights,
                    pass,
                    expert_cache,
                    assignment,
                    expert_group,
                    statistics,
                    context,
                )
                .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
                .map_err(|error: Error| Exception::custom(error.to_string()))
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
            if expert_group.is_some() && self.owns_routed_units {
                return Err(Error::Parallel(
                    "neutral DeepSeek V3 received EP without external experts".into(),
                ));
            }
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
}

impl PipelinePartitionMetadata for DeepSeekV3PipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::deepseek_v3(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part(
            "deepseek_v3",
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
        self.expert_storage.cache()
    }
}

impl PipelineEmbeddedMtp for DeepSeekV3PipelinePartition {
    fn embedded_mtp_len(&self) -> usize {
        self.mtp_layers.len()
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let caches = (0..self.mtp_layers.len())
            .map(|depth| -> Result<_, Error> {
                let group =
                    architecture_prediction_group::<_, MlxHybridState>(&self.architecture, depth)?;
                let ordinal = self
                    .partition
                    .unit_layout()
                    .ordinal(group, 0)
                    .ok_or_else(|| {
                        Error::Parallel(format!("V3 parameter layout has no MTP depth {depth}"))
                    })?;
                match &paged {
                    Some((manager, rank)) => {
                        CompressedLatentCache::new_paged(manager.clone(), ordinal, *rank)
                            .map_err(Into::into)
                    }
                    None => Ok(CompressedLatentCache::new()),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PipelineMtpCache::DeepSeek(caches))
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
        let tensor_execution = execution.filter(|execution| execution.is_tensor_parallel());
        let tensor_group = tensor_execution
            .map(|execution| {
                execution.group().ok_or_else(|| {
                    Error::Parallel("neutral DeepSeek V3 MTP has no TP communicator".into())
                })
            })
            .transpose()?;
        let PipelineMtpCache::DeepSeek(caches) = cache else {
            return Err(Error::Parallel(
                "neutral DeepSeek V3 MTP cache mismatch".into(),
            ));
        };
        let prediction_group =
            architecture_prediction_group::<_, MlxHybridState>(&self.architecture, depth)?;
        let layer = self
            .partition
            .unit_layout()
            .ordinal(prediction_group, 0)
            .ok_or_else(|| {
                Error::Parallel(format!("V3 parameter layout has no MTP depth {depth}"))
            })?;
        let unit = self.mtp_layers.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!(
                "neutral DeepSeek V3 MTP depth {depth} is unavailable"
            ))
        })?;
        let layer_cache = caches.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!(
                "neutral DeepSeek V3 MTP cache depth {depth} is unavailable"
            ))
        })?;
        let output = if let Some(expert_cache) = self.expert_storage.cache() {
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Error::Parallel("neutral DeepSeek V3 MTP experts have no assignment".into())
            })?;
            let mut execute = |execution: GatedProductExpertExecution, context: &Stream| {
                execute_pipeline_cached_neutral_deepseek(
                    &execution.spec,
                    execution.layer,
                    &execution.hidden,
                    &execution.expert_ids,
                    &execution.route_weights,
                    ExpertPass::Decode,
                    expert_cache,
                    assignment,
                    expert_group,
                    &mut self.routing_statistics,
                    context,
                )
                .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
                .map_err(|error: Error| Exception::custom(error.to_string()))
            };
            let mut provider = GatedProductExpertExecutorProvider::new(&mut execute);
            match tensor_group {
                Some(group) => self
                    .architecture
                    .pipeline_forward_prediction_neutral_parallel_with_provider(
                        &mut unit.inner,
                        crate::composition::tensor_ref(hidden),
                        crate::composition::tensor_ref(tokens),
                        layer_cache,
                        ExpertPass::Decode,
                        &mut provider,
                        group,
                        stream,
                    ),
                None => self.architecture.pipeline_forward_prediction_with_provider(
                    &mut unit.inner,
                    crate::composition::tensor_ref(hidden),
                    crate::composition::tensor_ref(tokens),
                    layer_cache,
                    ExpertPass::Decode,
                    &mut provider,
                    stream,
                ),
            }
        } else {
            if expert_group.is_some() {
                return Err(Error::Parallel(
                    "neutral DeepSeek V3 MTP received EP without external experts".into(),
                ));
            }
            match tensor_group {
                Some(group) => self
                    .architecture
                    .pipeline_forward_prediction_neutral_parallel(
                        &mut unit.inner,
                        crate::composition::tensor_ref(hidden),
                        crate::composition::tensor_ref(tokens),
                        layer_cache,
                        group,
                        stream,
                    ),
                None => self.architecture.pipeline_forward_prediction(
                    &mut unit.inner,
                    crate::composition::tensor_ref(hidden),
                    crate::composition::tensor_ref(tokens),
                    layer_cache,
                    stream,
                ),
            }
        }
        .map_err(|error| Error::Parallel(format!("V3 MTP layer {layer}: {error}")))?;
        Ok(EmbeddedMtpOutput {
            logits: output.logits,
            hidden: output.hidden,
            tokens: output.tokens,
        })
    }

    fn embedded_mtp_state_segment(&self) -> Option<&'static str> {
        None
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

impl PipelineForward for DeepSeekV3PipelinePartition {
    pipeline_observed_forward!();
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_stage(input, step, mask, cache, None, None, stream, None)
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
        self.forward_stage(
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

impl DeepSeekV4PipelinePartition {
    fn args(&self) -> &eredu_architectures::deepseek::V4Args {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        let group = architecture_decoder_group::<_, PipelineRangeState<'_>>(&self.architecture)
            .expect("validated DeepSeek V4 decoder group");
        self.partition
            .groups()
            .iter()
            .find(|owned| owned.group_index() == group)
            .map(|owned| owned.global_units())
            .unwrap_or(0..0)
    }

    fn forward_stage(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::Parallel(format!(
                "neutral DeepSeek V4 stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let decoder_range = self.range();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.routing_statistics = RoutingStatistics::default();
        let expert_cache = self.expert_storage.cache();
        let assignment = self.expert_assignment.as_ref();
        let statistics = &mut self.routing_statistics;
        if let Some(expert_cache) = expert_cache {
            let assignment = assignment.ok_or_else(|| {
                Error::Parallel("neutral DeepSeek V4 external experts have no assignment".into())
            })?;
            let mut execute = |execution: GatedProductExpertExecution, context: &Stream| {
                execute_pipeline_cached_neutral_deepseek(
                    &execution.spec,
                    execution.layer,
                    &execution.hidden,
                    &execution.expert_ids,
                    &execution.route_weights,
                    pass,
                    expert_cache,
                    assignment,
                    expert_group,
                    statistics,
                    context,
                )
                .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
                .map_err(|error: Error| Exception::custom(error.to_string()))
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
                mask,
                caches,
                execution,
                pass,
                &mut provider,
                stream,
                observer,
            )
        } else {
            if expert_group.is_some() && self.owns_routed_units {
                return Err(Error::Parallel(
                    "neutral DeepSeek V4 received EP without external experts".into(),
                ));
            }
            let mut provider = eredu_runtime::ResidentExpertProvider;
            execute_routed_layered_partition_observed(
                &mut self.architecture,
                &self.partition,
                decoder_range,
                &mut self.layers,
                self.dense_layers.as_ref(),
                input,
                step,
                mask,
                caches,
                execution,
                pass,
                &mut provider,
                stream,
                observer,
            )
        }
    }
}

impl PipelinePartitionMetadata for DeepSeekV4PipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::deepseek_v4(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part(
            "deepseek_v4",
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
        self.expert_storage.cache()
    }

    fn persisted_prompt_cache_identity(
        &self,
        identity: PromptCacheModelIdentity,
    ) -> Result<PromptCacheModelIdentity, Error> {
        // V4 prediction state is allocated transactionally and is not
        // populated by an ordinary target-only prefill. Persist the target
        // segment, matching the state actually owned by `PipelineCache::layers`.
        identity
            .select_state_segment(eredu_architectures::deepseek::TARGET_STATE_SEGMENT)
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        let pinned_prefix_tokens = i32::try_from(identity.sink_tokens)
            .map_err(|_| Error::Parallel("V4 attention sink count exceeds i32".into()))?;
        let range = self.range();
        if (identity.global_layer_start..identity.global_layer_end) != range
            || identity.layer_layout.len() != range.len()
        {
            return Err(Error::Parallel(format!(
                "V4 prompt-cache identity owns layers {}..{} with {} policies, expected stage range {range:?}",
                identity.global_layer_start,
                identity.global_layer_end,
                identity.layer_layout.len()
            )));
        }
        range
            .zip(identity.layer_layout.iter())
            .map(|(global_layer, policy)| {
                let cache = match &paged {
                    Some((manager, rank)) => MlxPoolingAttentionCache::paged_from_policy(
                        global_layer,
                        policy,
                        manager.clone(),
                        global_layer,
                        pinned_prefix_tokens,
                        *rank,
                    )?,
                    None => MlxPoolingAttentionCache::resident_from_policy(global_layer, policy)?,
                };
                Ok(PipelineLayerCache::PoolingAttention {
                    global_layer,
                    cache,
                })
            })
            .collect()
    }
}

impl PipelineEmbeddedMtp for DeepSeekV4PipelinePartition {
    fn embedded_mtp_len(&self) -> usize {
        self.mtp_layers.len()
    }

    fn draft_proposal_capacity(&self, _global_prediction_layers: usize) -> usize {
        self.architecture.draft_proposal_capacity()
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let layout = eredu_runtime::ArchitectureParameters::state_layout(&self.architecture)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let caches = (0..self.mtp_layers.len())
            .map(|depth| {
                let group = architecture_prediction_group::<
                    _,
                    eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
                >(&self.architecture, depth)?;
                let layer = self
                    .partition
                    .unit_layout()
                    .ordinal(group, 0)
                    .ok_or_else(|| {
                        Error::Parallel(format!("V4 parameter layout has no MTP depth {depth}"))
                    })?;
                let policy = layout.layer(layer).ok_or_else(|| {
                    Error::Parallel(format!("missing V4 MTP state layout layer {layer}"))
                })?;
                match &paged {
                    Some((manager, rank)) => MlxPoolingAttentionCache::paged_from_policy(
                        layer,
                        policy,
                        manager.clone(),
                        layer,
                        0,
                        *rank,
                    )
                    .map_err(Into::into),
                    None => MlxPoolingAttentionCache::resident_from_policy(layer, policy)
                        .map_err(Into::into),
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(PipelineMtpCache::NeutralDeepSeekV4(caches))
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
        if self.args().dspark.is_some() {
            return Err(Error::Parallel(
                "neutral DSpark uses fused proposals, not sequential MTP".into(),
            ));
        }
        let tensor_execution = execution.filter(|execution| execution.is_tensor_parallel());
        let tensor_group = tensor_execution
            .map(|execution| {
                execution.group().ok_or_else(|| {
                    Error::Parallel("neutral DeepSeek V4 MTP has no TP communicator".into())
                })
            })
            .transpose()?;
        let PipelineMtpCache::NeutralDeepSeekV4(caches) = cache else {
            return Err(Error::Parallel(
                "neutral DeepSeek V4 MTP cache mismatch".into(),
            ));
        };
        let prediction_group = architecture_prediction_group::<
            _,
            eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
        >(&self.architecture, depth)?;
        let layer = self
            .partition
            .unit_layout()
            .ordinal(prediction_group, 0)
            .ok_or_else(|| {
                Error::Parallel(format!("V4 parameter layout has no MTP depth {depth}"))
            })?;
        let unit = self.mtp_layers.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!(
                "neutral DeepSeek V4 MTP depth {depth} is unavailable"
            ))
        })?;
        let layer_cache = caches.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!(
                "neutral DeepSeek V4 MTP cache depth {depth} is unavailable"
            ))
        })?;
        let hidden = self
            .architecture
            .begin_partition_prediction_hidden(crate::composition::tensor_ref(hidden), stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let output = if let Some(expert_cache) = self.expert_storage.cache() {
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Error::Parallel("neutral DeepSeek V4 MTP experts have no assignment".into())
            })?;
            let mut execute = |execution: GatedProductExpertExecution, context: &Stream| {
                execute_pipeline_cached_neutral_deepseek(
                    &execution.spec,
                    execution.layer,
                    &execution.hidden,
                    &execution.expert_ids,
                    &execution.route_weights,
                    ExpertPass::Decode,
                    expert_cache,
                    assignment,
                    expert_group,
                    &mut self.routing_statistics,
                    context,
                )
                .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
                .map_err(|error: Error| Exception::custom(error.to_string()))
            };
            let mut provider = GatedProductExpertExecutorProvider::new(&mut execute);
            match tensor_group {
                Some(group) => self
                    .architecture
                    .pipeline_forward_prediction_neutral_parallel_with_provider(
                        &mut unit.inner,
                        &hidden,
                        crate::composition::tensor_ref(tokens),
                        layer_cache,
                        ExpertPass::Decode,
                        &mut provider,
                        group,
                        stream,
                    ),
                None => self.architecture.pipeline_forward_prediction_with_provider(
                    &mut unit.inner,
                    &hidden,
                    crate::composition::tensor_ref(tokens),
                    layer_cache,
                    ExpertPass::Decode,
                    &mut provider,
                    stream,
                ),
            }
        } else {
            if expert_group.is_some() {
                return Err(Error::Parallel(
                    "neutral DeepSeek V4 MTP received EP without external experts".into(),
                ));
            }
            match tensor_group {
                Some(group) => self
                    .architecture
                    .pipeline_forward_prediction_neutral_parallel(
                        &mut unit.inner,
                        &hidden,
                        crate::composition::tensor_ref(tokens),
                        layer_cache,
                        group,
                        stream,
                    ),
                None => self.architecture.pipeline_forward_prediction(
                    &mut unit.inner,
                    &hidden,
                    crate::composition::tensor_ref(tokens),
                    layer_cache,
                    stream,
                ),
            }
        }
        .map_err(|error| Error::Parallel(format!("V4 MTP layer {layer}: {error}")))?;
        let output = self
            .architecture
            .finish_partition_prediction_output(output, stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(EmbeddedMtpOutput {
            logits: output.logits,
            hidden: output.hidden,
            tokens: output.tokens,
        })
    }

    fn prefill_embedded_mtp_cache(
        &mut self,
        output: &EmbeddedMtpOutput,
        tokens: &Array,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<bool, Error> {
        if self.args().dspark.is_some() {
            let PipelineMtpCache::NeutralDeepSeekV4(caches) = cache else {
                return Err(Error::Parallel("neutral DSpark cache mismatch".into()));
            };
            self.architecture
                .pipeline_prefill_dspark_context(
                    &mut self.mtp_layers,
                    &output.hidden,
                    caches,
                    stream,
                )
                .map_err(|error| Error::Parallel(error.to_string()))?;
            return Ok(true);
        }
        if self.architecture.shared_parallel_geometry().is_some()
            || self
                .expert_assignment
                .as_ref()
                .is_some_and(|assignment| assignment.group_size() > 1)
        {
            // The speculative wrapper will replay these prefixes through
            // `forward_draft`, which carries the live TP/EP execution context.
            return Ok(false);
        }
        let Some((hidden, next)) = self
            .architecture
            .prepare_partition_prediction_replay(
                &output.hidden,
                crate::composition::tensor_ref(tokens),
                stream,
            )
            .map_err(|error| Error::Parallel(error.to_string()))?
        else {
            return Ok(true);
        };
        for depth in 0..self.mtp_layers.len() {
            let output = self.forward_embedded_mtp_draft(
                hidden.as_array(),
                next.as_array(),
                depth,
                cache,
                None,
                None,
                stream,
            )?;
            synchronize_outputs([output.hidden.as_array(), output.logits.as_array()])?;
        }
        Ok(true)
    }

    fn fused_embedded_mtp_logits(
        &mut self,
        _hidden: &Array,
        last_token: u32,
        proposal_capacity: usize,
        cache: &mut PipelineMtpCache,
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<Option<Array>, Error> {
        if self.args().dspark.is_none() {
            return Ok(None);
        }
        let tensor_execution = execution.filter(|execution| execution.is_tensor_parallel());
        let tensor_group = tensor_execution
            .map(|execution| {
                execution
                    .group()
                    .ok_or_else(|| Error::Parallel("neutral DSpark has no TP communicator".into()))
            })
            .transpose()?;
        let PipelineMtpCache::NeutralDeepSeekV4(caches) = cache else {
            return Err(Error::Parallel("neutral DSpark cache mismatch".into()));
        };
        let mut proposal = caches.clone();
        let anchor = crate::MlxTensor::from_array(Array::from_slice(&[last_token], &[1, 1]));
        let logits = if let Some(expert_cache) = self.expert_storage.cache() {
            let assignment = self.expert_assignment.as_ref().ok_or_else(|| {
                Error::Parallel("neutral DSpark experts have no assignment".into())
            })?;
            let mut execute = |execution: GatedProductExpertExecution, context: &Stream| {
                execute_pipeline_cached_neutral_deepseek(
                    &execution.spec,
                    execution.layer,
                    &execution.hidden,
                    &execution.expert_ids,
                    &execution.route_weights,
                    ExpertPass::Decode,
                    expert_cache,
                    assignment,
                    expert_group,
                    &mut self.routing_statistics,
                    context,
                )
                .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
                .map_err(|error: Error| Exception::custom(error.to_string()))
            };
            let mut provider = GatedProductExpertExecutorProvider::new(&mut execute);
            match tensor_group {
                Some(group) => self
                    .architecture
                    .pipeline_dspark_proposal_neutral_parallel_with_provider(
                        &mut self.mtp_layers,
                        &anchor,
                        proposal_capacity,
                        &mut proposal,
                        ExpertPass::Decode,
                        &mut provider,
                        group,
                        stream,
                    ),
                None => self.architecture.pipeline_dspark_proposal_with_provider(
                    &mut self.mtp_layers,
                    &anchor,
                    proposal_capacity,
                    &mut proposal,
                    ExpertPass::Decode,
                    &mut provider,
                    stream,
                ),
            }
        } else {
            if expert_group.is_some() {
                return Err(Error::Parallel(
                    "neutral DSpark received EP without external experts".into(),
                ));
            }
            match tensor_group {
                Some(group) => self.architecture.pipeline_dspark_proposal_neutral_parallel(
                    &mut self.mtp_layers,
                    &anchor,
                    proposal_capacity,
                    &mut proposal,
                    group,
                    stream,
                ),
                None => self.architecture.pipeline_dspark_proposal(
                    &mut self.mtp_layers,
                    &anchor,
                    proposal_capacity,
                    &mut proposal,
                    stream,
                ),
            }
        }
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(Some(logits.into_array()))
    }

    fn advance_embedded_mtp_cache(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        cache: &mut PipelineMtpCache,
        stream: &Stream,
    ) -> Result<bool, Error> {
        if self.args().dspark.is_some() {
            let PipelineMtpCache::NeutralDeepSeekV4(caches) = cache else {
                return Err(Error::Parallel("neutral DSpark cache mismatch".into()));
            };
            self.architecture
                .pipeline_prefill_dspark_context(
                    &mut self.mtp_layers,
                    crate::composition::tensor_ref(hidden),
                    caches,
                    stream,
                )
                .map_err(|error| Error::Parallel(error.to_string()))?;
            return Ok(true);
        }
        if self.architecture.shared_parallel_geometry().is_some()
            || self
                .expert_assignment
                .as_ref()
                .is_some_and(|assignment| assignment.group_size() > 1)
        {
            return Ok(false);
        }
        for depth in 0..self.mtp_layers.len() {
            let _ =
                self.forward_embedded_mtp_draft(hidden, tokens, depth, cache, None, None, stream)?;
        }
        Ok(true)
    }

    fn embedded_mtp_state_segment(&self) -> Option<&'static str> {
        None
    }

    fn adjust_fused_embedded_mtp_logits(
        &mut self,
        logits: Array,
        _last_token: u32,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Ok(logits)
    }
}

impl PipelineForward for DeepSeekV4PipelinePartition {
    pipeline_observed_forward!();
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        self.forward_stage(input, step, mask, cache, None, None, stream, None)
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
        self.forward_stage(
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
fn execute_pipeline_cached_neutral_deepseek(
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

fn v3_sharded_unit_bindings(
    args: &eredu_architectures::deepseek::V3Args,
    ordinal: usize,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    external_experts: bool,
    layout: &eredu_runtime::LocalModelLayout,
    stream: &Stream,
) -> Result<Vec<WeightBinding>, Error> {
    let probe = crate::composition::deepseek::new_v3_unit(args, ordinal, external_experts, stream)?;
    let bindings = crate::composition::deepseek::v3_unit_bindings(
        args,
        ordinal,
        &probe,
        store,
        external_experts,
    )?;
    shard_layer_bindings(bindings, store, layout)
}

fn v4_sharded_unit_bindings(
    args: &eredu_architectures::deepseek::V4Args,
    ordinal: usize,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    external_experts: bool,
    layout: &eredu_runtime::LocalModelLayout,
    stream: &Stream,
) -> Result<Vec<WeightBinding>, Error> {
    let probe = crate::composition::deepseek::new_v4_unit(args, ordinal, external_experts, stream)?;
    let bindings = crate::composition::deepseek::v4_unit_bindings(
        args,
        ordinal,
        &probe,
        store,
        external_experts,
    )?;
    shard_layer_bindings(bindings, store, layout)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_neutral_deepseek_v3_pipeline(
    source_args: eredu_architectures::deepseek::V3Args,
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
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::DeepSeekV3], "DeepSeek-V3")?;
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    let (store, args, materialization) = match requested_quantization {
        Some(quantization) => {
            let (store, args, report) = crate::composition::deepseek::quantize_v3_store(
                store,
                &source_args,
                quantization,
                stream,
            )?;
            (store, args, Some(report))
        }
        None => (store, source_args, None),
    };
    let seed_architecture = NeutralV3Architecture::new(args.clone(), stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let seed_expert_realization = eredu_architectures::deepseek::v3_expert_realization_plan(
        &seed_architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let parameter_description =
        eredu_architectures::deepseek::parallel::v3_parameter_description(&args)
            .map_err(|error| Error::Parallel(error.to_string()))?;
    parameter_description
        .validate_architecture::<MlxNeuralBackend, MlxHybridState, _>(&seed_architecture)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&seed_architecture)?;
    let target_units = parameter_description
        .unit_layout()
        .group_range(decoder_group)
        .ok_or_else(|| Error::Parallel("V3 parameter description has no target group".into()))?
        .len();
    let prediction_units = architecture_single_prediction_units::<_, MlxHybridState>(
        &seed_architecture,
        &parameter_description,
    )?;
    topology.preflight(
        Some(target_units),
        external_experts
            .then(|| {
                seed_expert_realization
                    .as_ref()
                    .map(eredu_architectures::ExpertRealizationPlan::global_expert_count)
                    .ok_or_else(|| {
                        Error::Parallel(
                            "DeepSeek V3 external experts require an architecture realization"
                                .into(),
                        )
                    })
            })
            .transpose()?,
    )?;
    let range = topology.layer_range(target_units)?;
    let tensor_parallel = topology.tensor_parallel_size > 1;
    let parallel_layout = tensor_parallel
        .then(|| architecture_parallel_layout(&parameter_description, topology))
        .transpose()?;
    let seed_static_module = MlxModule::new(seed_architecture.static_modules().clone());
    let all_static_bindings = build_module_bindings(&seed_static_module, "", store.as_ref())?;
    let mut architecture = match parallel_layout.as_ref() {
        Some(layout) => {
            let geometry =
                eredu_architectures::deepseek::parallel::v3_local_geometry(&args, layout)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
            NeutralV3Architecture::new_parallel(args.clone(), geometry, stream)
                .map_err(|error| Error::Parallel(error.to_string()))?
        }
        None => seed_architecture,
    };
    let expert_realization = eredu_architectures::deepseek::v3_expert_realization_plan(
        &architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    if let Some(realization) = expert_realization.clone() {
        architecture.install_expert_realization(realization);
    }
    let expert_assignment = external_experts
        .then(|| {
            ExpertAssignment::from_realization(expert_realization.as_ref().ok_or_else(|| {
                Error::Parallel(
                    "DeepSeek V3 external experts require an architecture realization".into(),
                )
            })?)
        })
        .transpose()?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&architecture)?;
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
    info.global_embedded_mtp_layers = prediction_units.len();
    if let Some(assignment) = &expert_assignment {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    info.materialization = materialization.clone();
    let geometry = architecture.shared_parallel_geometry();
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            geometry,
            &parameter_description,
        )?;
    let owns_mtp = partition_owns_architecture_units(
        &partition,
        prediction_units
            .iter()
            .map(|(group, index)| (*group, *index..*index + 1)),
    );
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp { prediction_units.len() } else { 0 };
    let static_roles = parameter_description.select_static_roles(&partition);
    let static_units = split_static_binding_units_by_owner(
        partition.parameter_bindings(),
        &all_static_bindings,
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("neutral DeepSeek V3", &partition);
    load_architecture_static_parameters(
        &mut architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        None,
        weights_stream,
        stream,
    )?;
    let mut layers = range
        .clone()
        .map(|layer| {
            architecture
                .construct_unit(decoder_group, layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::Parallel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if dense_stream.is_none() {
        for (global_layer, unit) in range.clone().zip(&mut layers) {
            let bindings = match &parallel_layout {
                Some(layout) => v3_sharded_unit_bindings(
                    &args,
                    global_layer,
                    store.as_ref(),
                    external_experts,
                    layout,
                    stream,
                )?,
                None => crate::composition::deepseek::v3_unit_bindings(
                    &args,
                    global_layer,
                    unit,
                    store.as_ref(),
                    external_experts,
                )?,
            };
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        &architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    unit,
                    store.as_ref(),
                    &bindings,
                    None,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        &architecture,
                        decoder_group,
                        global_layer,
                    )?,
                    unit,
                    store.as_ref(),
                    &bindings,
                    None,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let mut mtp_layers = if owns_mtp {
        prediction_units
            .iter()
            .map(|&(prediction_group, _)| {
                architecture
                    .construct_unit(prediction_group, 0, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::Parallel(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    for ((prediction_group, ordinal), unit) in
        prediction_units.iter().copied().zip(mtp_layers.iter_mut())
    {
        let flat_ordinal = parameter_description
            .unit_layout()
            .ordinal(prediction_group, ordinal)
            .ok_or_else(|| {
                Error::Parallel(format!(
                    "DeepSeek V3 parameter layout has no prediction unit {prediction_group}:{ordinal}"
                ))
            })?;
        let bindings = match &parallel_layout {
            Some(layout) => v3_sharded_unit_bindings(
                &args,
                flat_ordinal,
                store.as_ref(),
                external_experts,
                layout,
                stream,
            )?,
            None => crate::composition::deepseek::v3_unit_bindings(
                &args,
                flat_ordinal,
                unit,
                store.as_ref(),
                external_experts,
            )?,
        };
        if external_experts {
            loaded.load_excluding_roles(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    &architecture,
                    prediction_group,
                    0,
                )?,
                unit,
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
                &[eredu_runtime::ParameterRole::ExpertIntermediate],
            )?;
        } else {
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    &architecture,
                    prediction_group,
                    0,
                )?,
                unit,
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
            )?;
        }
    }
    let static_device_bytes = loaded.finish(&mut info)?;
    let streamed_architecture = &architecture;
    let dense_layers = dense_stream
        .map(|options| {
            let global_binding_args = args.clone();
            let binding_layout = parallel_layout.clone();
            let binding_stream = stream.clone();
            build_pipeline_layer_storage(
                Arc::clone(&store),
                partition.parameter_bindings(),
                if external_experts {
                    &[eredu_runtime::ParameterRole::ExpertIntermediate]
                } else {
                    &[]
                },
                range.clone(),
                options,
                static_device_bytes,
                materialization.clone(),
                stream,
                weights_stream,
                |layer, stream| {
                    streamed_architecture
                        .construct_unit(decoder_group, layer, stream)
                        .map(MlxModule::new)
                        .map_err(|error| Error::Parallel(error.to_string()))
                },
                {
                    move |layer, unit, store| match &binding_layout {
                        Some(layout) => v3_sharded_unit_bindings(
                            &global_binding_args,
                            layer,
                            store,
                            external_experts,
                            layout,
                            &binding_stream,
                        ),
                        None => crate::composition::deepseek::v3_unit_bindings(
                            &global_binding_args,
                            layer,
                            unit,
                            store,
                            external_experts,
                        ),
                    }
                },
                |layer| {
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
                        streamed_architecture,
                        decoder_group,
                        layer,
                    )
                },
            )
        })
        .transpose()?;
    info.planned_owned_parameter_bytes = static_device_bytes
        .checked_add(
            dense_layers
                .as_ref()
                .map(PipelineLayerStorage::planned_layer_bytes)
                .transpose()?
                .unwrap_or(0),
        )
        .ok_or_else(|| Error::Parallel("neutral DeepSeek V3 owned byte total overflowed".into()))?;
    let mut expert_storage = if external_experts {
        PipelineExpertStorage::ExternalEmpty
    } else {
        PipelineExpertStorage::LayerLocal
    };
    if external_experts {
        let assignment = expert_assignment
            .as_ref()
            .expect("external expert assignment");
        let catalog = match &parallel_layout {
            Some(layout) => {
                let realization = expert_realization
                    .as_ref()
                    .expect("external V3 experts have an architecture realization");
                let intermediate = routed_expert_intermediate_range(
                    layout,
                    realization.global_expert_count(),
                    localized_gated_expert_width(realization, "DeepSeek V3")?,
                )?;
                crate::composition::deepseek_expert::v3_parallel_catalog_selected(
                    &args,
                    intermediate,
                    store.as_ref(),
                    |group, unit| partition.owns_unit(group.as_str(), unit),
                )?
            }
            None => crate::composition::deepseek_expert::v3_catalog_selected(
                &args,
                store.as_ref(),
                |group, unit| partition.owns_unit(group.as_str(), unit),
            )?,
        };
        let entries = catalog
            .into_iter()
            .filter(|entry| {
                assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
            })
            .collect::<Vec<_>>();
        if !entries.is_empty() {
            let cache = build_pipeline_expert_cache(
                Arc::clone(&store),
                entries,
                expert_cache_options,
                None,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes)
                .ok_or_else(|| {
                    Error::Parallel("neutral DeepSeek V3 expert byte total overflowed".into())
                })?;
            expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    let diagnostics = store.source_diagnostics()?;
    info.opened_checkpoint_shards = diagnostics.payload_shard_paths.clone();
    info.checkpoint_diagnostics = Some(diagnostics);
    let owns_routed_units = expert_realization.as_ref().is_some_and(|realization| {
        realization
            .unit_specs()
            .keys()
            .any(|(group, unit)| partition.owns_unit(group.as_str(), *unit))
    });
    if !owns_routed_units {
        info.local_expert_ids.clear();
    }
    let stage = DeepSeekV3PipelinePartition {
        architecture,
        partition,
        layers,
        mtp_layers,
        dense_layers,
        expert_assignment,
        expert_storage,
        owns_routed_units,
        routing_statistics: RoutingStatistics::default(),
    };
    PipelineModel::from_adapter(topology, info, stage)
}
pub(super) fn load_neutral_deepseek_v4_pipeline(
    source_args: eredu_architectures::deepseek::V4Args,
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
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::DeepSeekV4], "DeepSeek-V4")?;
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    let (store, args, materialization) = match requested_quantization {
        Some(quantization) => {
            let (store, args, report) = crate::composition::deepseek::quantize_v4_store(
                store,
                &source_args,
                quantization,
                stream,
            )?;
            (store, args, Some(report))
        }
        None => (store, source_args, None),
    };
    let seed_architecture = NeutralV4Architecture::new(args.clone(), stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let seed_expert_realization = eredu_architectures::deepseek::v4_expert_realization_plan(
        &seed_architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let parameter_description =
        eredu_architectures::deepseek::parallel::v4_parameter_description(&args)
            .map_err(|error| Error::Parallel(error.to_string()))?;
    parameter_description
        .validate_architecture::<
            MlxNeuralBackend,
            eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
            _,
        >(&seed_architecture)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let decoder_group = architecture_decoder_group::<
        _,
        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
    >(&seed_architecture)?;
    let target_units = parameter_description
        .unit_layout()
        .group_range(decoder_group)
        .ok_or_else(|| Error::Parallel("V4 parameter description has no target group".into()))?
        .len();
    let prediction_units = architecture_single_prediction_units::<
        _,
        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
    >(&seed_architecture, &parameter_description)?;
    topology.preflight(
        Some(target_units),
        external_experts.then_some(seed_expert_realization.global_expert_count()),
    )?;
    let range = topology.layer_range(target_units)?;
    let tensor_parallel = topology.tensor_parallel_size > 1;
    let parallel_layout = tensor_parallel
        .then(|| architecture_parallel_layout(&parameter_description, topology))
        .transpose()?;
    let seed_static_module = MlxModule::new(seed_architecture.static_modules().clone());
    let all_static_bindings = build_module_bindings(&seed_static_module, "", store.as_ref())?;
    let mut architecture = match parallel_layout.as_ref() {
        Some(layout) => {
            let geometry =
                eredu_architectures::deepseek::parallel::v4_local_geometry(&args, layout)
                    .map_err(|error| Error::Parallel(error.to_string()))?;
            NeutralV4Architecture::new_parallel(args.clone(), geometry, stream)
                .map_err(|error| Error::Parallel(error.to_string()))?
        }
        None => seed_architecture,
    };
    let expert_realization = eredu_architectures::deepseek::v4_expert_realization_plan(
        &architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    architecture.install_expert_realization(expert_realization.clone());
    let expert_assignment = external_experts
        .then(|| ExpertAssignment::from_realization(&expert_realization))
        .transpose()?;
    let decoder_group = architecture_decoder_group::<
        _,
        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
    >(&architecture)?;
    let placement = Arc::new(prediction_architecture_transport::<
        _,
        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
    >(&architecture, topology.pipeline_parallel_size)?);
    let mut info = base_info(
        topology,
        wire_contract,
        range.clone(),
        placement,
        eredu_architectures::decoder::TARGET_EXECUTION_GROUP,
        model_kind,
    );
    info.global_embedded_mtp_layers = prediction_units.len();
    if let Some(assignment) = &expert_assignment {
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
    }
    info.materialization = materialization.clone();
    let geometry = architecture.shared_parallel_geometry();
    let partition = info.placement.realize_architecture_partition::<
        MlxNeuralBackend,
        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
        _,
        _,
        _,
    >(
        &architecture,
        info.pipeline_stage,
        geometry,
        &parameter_description,
    )?;
    let owns_mtp = partition_owns_architecture_units(
        &partition,
        prediction_units
            .iter()
            .map(|(group, index)| (*group, *index..*index + 1)),
    );
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp { prediction_units.len() } else { 0 };
    let static_roles = parameter_description.select_static_roles(&partition);
    let static_units = split_static_binding_units_by_owner(
        partition.parameter_bindings(),
        &all_static_bindings,
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("neutral DeepSeek V4", &partition);
    load_architecture_static_parameters(
        &mut architecture,
        &static_roles,
        &static_units,
        &mut loaded,
        store.as_ref(),
        parallel_layout.as_ref(),
        None,
        weights_stream,
        stream,
    )?;
    let mut layers = range
        .clone()
        .map(|layer| {
            architecture
                .construct_unit(decoder_group, layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::Parallel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if dense_stream.is_none() {
        for (global_layer, unit) in range.clone().zip(&mut layers) {
            let bindings = match &parallel_layout {
                Some(layout) => v4_sharded_unit_bindings(
                    &args,
                    global_layer,
                    store.as_ref(),
                    external_experts,
                    layout,
                    stream,
                )?,
                None => crate::composition::deepseek::v4_unit_bindings(
                    &args,
                    global_layer,
                    unit,
                    store.as_ref(),
                    external_experts,
                )?,
            };
            if external_experts {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<
                        _,
                        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
                    >(&architecture, decoder_group, global_layer)?,
                    unit,
                    store.as_ref(),
                    &bindings,
                    None,
                    weights_stream,
                    stream,
                    &[eredu_runtime::ParameterRole::ExpertIntermediate],
                )?;
            } else {
                loaded.load(
                    architecture_parameter_unit_owner::<
                        _,
                        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
                    >(&architecture, decoder_group, global_layer)?,
                    unit,
                    store.as_ref(),
                    &bindings,
                    None,
                    weights_stream,
                    stream,
                )?;
            }
        }
    }
    let mut mtp_layers = if owns_mtp {
        prediction_units
            .iter()
            .map(|&(prediction_group, _)| {
                architecture
                    .construct_unit(prediction_group, 0, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::Parallel(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    for ((prediction_group, ordinal), unit) in
        prediction_units.iter().copied().zip(mtp_layers.iter_mut())
    {
        let flat_ordinal = parameter_description
            .unit_layout()
            .ordinal(prediction_group, ordinal)
            .ok_or_else(|| {
                Error::Parallel(format!(
                    "DeepSeek V4 parameter layout has no prediction unit {prediction_group}:{ordinal}"
                ))
            })?;
        let bindings = match &parallel_layout {
            Some(layout) => v4_sharded_unit_bindings(
                &args,
                flat_ordinal,
                store.as_ref(),
                external_experts,
                layout,
                stream,
            )?,
            None => crate::composition::deepseek::v4_unit_bindings(
                &args,
                flat_ordinal,
                unit,
                store.as_ref(),
                external_experts,
            )?,
        };
        if external_experts {
            loaded.load_excluding_roles(
                architecture_parameter_unit_owner::<
                    _,
                    eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
                >(&architecture, prediction_group, 0)?,
                unit,
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
                &[eredu_runtime::ParameterRole::ExpertIntermediate],
            )?;
        } else {
            loaded.load(
                architecture_parameter_unit_owner::<
                    _,
                    eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
                >(&architecture, prediction_group, 0)?,
                unit,
                store.as_ref(),
                &bindings,
                None,
                weights_stream,
                stream,
            )?;
        }
    }
    let static_device_bytes = loaded.finish(&mut info)?;
    let streamed_architecture = &architecture;
    let dense_layers = dense_stream
        .map(|options| {
            let global_binding_args = args.clone();
            let binding_layout = parallel_layout.clone();
            let binding_stream = stream.clone();
            build_pipeline_layer_storage(
                Arc::clone(&store),
                partition.parameter_bindings(),
                if external_experts {
                    &[eredu_runtime::ParameterRole::ExpertIntermediate]
                } else {
                    &[]
                },
                range.clone(),
                options,
                static_device_bytes,
                materialization.clone(),
                stream,
                weights_stream,
                |layer, stream| {
                    streamed_architecture
                        .construct_unit(decoder_group, layer, stream)
                        .map(MlxModule::new)
                        .map_err(|error| Error::Parallel(error.to_string()))
                },
                {
                    move |layer, unit, store| match &binding_layout {
                        Some(layout) => v4_sharded_unit_bindings(
                            &global_binding_args,
                            layer,
                            store,
                            external_experts,
                            layout,
                            &binding_stream,
                        ),
                        None => crate::composition::deepseek::v4_unit_bindings(
                            &global_binding_args,
                            layer,
                            unit,
                            store,
                            external_experts,
                        ),
                    }
                },
                |layer| {
                    architecture_parameter_unit_owner::<
                        _,
                        eredu_runtime::DeviceState<MlxNeuralBackend, MlxPoolingAttentionCache>,
                    >(streamed_architecture, decoder_group, layer)
                },
            )
        })
        .transpose()?;
    info.planned_owned_parameter_bytes = static_device_bytes
        .checked_add(
            dense_layers
                .as_ref()
                .map(PipelineLayerStorage::planned_layer_bytes)
                .transpose()?
                .unwrap_or(0),
        )
        .ok_or_else(|| Error::Parallel("neutral DeepSeek V4 owned byte total overflowed".into()))?;
    let mut expert_storage = if external_experts {
        PipelineExpertStorage::ExternalEmpty
    } else {
        PipelineExpertStorage::LayerLocal
    };
    if external_experts {
        let assignment = expert_assignment
            .as_ref()
            .expect("external expert assignment");
        let catalog = match &parallel_layout {
            Some(layout) => {
                let intermediate = routed_expert_intermediate_range(
                    layout,
                    expert_realization.global_expert_count(),
                    localized_gated_expert_width(&expert_realization, "DeepSeek V4")?,
                )?;
                crate::composition::deepseek_expert::v4_parallel_catalog_selected(
                    &args,
                    intermediate,
                    store.as_ref(),
                    |group, unit| partition.owns_unit(group.as_str(), unit),
                )?
            }
            None => crate::composition::deepseek_expert::v4_catalog_selected(
                &args,
                store.as_ref(),
                |group, unit| partition.owns_unit(group.as_str(), unit),
            )?,
        };
        let entries = catalog
            .into_iter()
            .filter(|entry| {
                assignment.owner(entry.identity().global_expert) == Some(assignment.rank())
            })
            .collect::<Vec<_>>();
        if !entries.is_empty() {
            let cache = build_pipeline_expert_cache(
                Arc::clone(&store),
                entries,
                expert_cache_options,
                None,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes)
                .ok_or_else(|| {
                    Error::Parallel("neutral DeepSeek V4 expert byte total overflowed".into())
                })?;
            expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    let diagnostics = store.source_diagnostics()?;
    info.opened_checkpoint_shards = diagnostics.payload_shard_paths.clone();
    info.checkpoint_diagnostics = Some(diagnostics);
    let owns_routed_units = expert_realization
        .unit_specs()
        .keys()
        .any(|(group, unit)| partition.owns_unit(group.as_str(), *unit));
    if !owns_routed_units {
        info.local_expert_ids.clear();
    }
    let stage = DeepSeekV4PipelinePartition {
        architecture,
        partition,
        layers,
        mtp_layers,
        dense_layers,
        expert_assignment,
        expert_storage,
        owns_routed_units,
        routing_statistics: RoutingStatistics::default(),
    };
    PipelineModel::from_adapter(topology, info, stage)
}
