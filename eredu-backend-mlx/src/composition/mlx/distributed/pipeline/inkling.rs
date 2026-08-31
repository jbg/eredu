use std::{ops::Range, sync::Arc};

use crate::backend::runtime::distributed::Group;
use eredu_architectures::ModelKind;
use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
use eredu_core::cache::{CacheRankIdentity, PromptCacheModelIdentity};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureParameters, ExpertCacheLoadOptions, ExpertPass,
    LayeredArchitecture, ParallelLayeredArchitecture,
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
                },
            },
        },
        MlxParallelContext,
    },
    composition::mlx::distributed::pipeline::{
        architecture_decoder_group, architecture_group_by_kind, architecture_group_id_by_kind,
        architecture_group_unit_count, architecture_parallel_layout,
        architecture_parameter_unit_owner, base_info, build_pipeline_expert_cache,
        build_pipeline_layer_storage, checkpoint_backing_shards,
        execute_routed_layered_partition_observed, load_architecture_static_parameters,
        materialize_pipeline_cache_layers, media_architecture_transport, pipeline_binding_units,
        preflight_pipeline_realization, quantize_pipeline_stage_store,
        validate_admitted_pipeline_kind, validate_pipeline_expert_dispatch, BoundPipelineBindings,
        InklingIngressState, InklingPipelinePartition, MlxPlacedGroupExecutor,
        PipelineAuxiliaryState, PipelineEmbeddedMtp, PipelineExpertStorage, PipelineForward,
        PipelineLayerCache, PipelineLayerLoadOptions, PipelineLayerStorage,
        PipelineLoadAccumulator, PipelineModel, PipelineMtpCache, PipelinePartitionMetadata,
        PipelinePayload, PipelineStageInput, PipelineStageOutput, PipelineStep,
    },
    composition::{
        inkling::{InklingBindings, InklingPipelineUnit, PreparedInklingInput},
        mlx::speculative::embedded::EmbeddedMtpOutput,
    },
};

impl InklingPipelinePartition {
    fn args(&self) -> &eredu_architectures::inkling::ModelArgs {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_runtime::ArchitectureGroupKind::Decoder)
    }

    fn vision_range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_runtime::ArchitectureGroupKind::VisionEncoder)
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<InklingPipelineUnit, Error> {
        <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::build_unit(&self.architecture, group, index, stream)
        .map(MlxModule::new)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn begin_ingress(
        &mut self,
        typed: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<InklingIngressState, Error> {
        let prepared = PreparedInklingInput::new(self.args(), typed, stream)?;
        let parts = prepared
            .tokens
            .iter()
            .zip(prepared.kinds.iter().copied())
            .zip(&prepared.projected)
            .map(|((tokens, kind), projected)| match projected {
                Some(embeddings) => {
                    eredu_architectures::inkling::DecoderInputPart::Projected { tokens, embeddings }
                }
                None => match kind {
                    eredu_core::InputModality::Text => {
                        eredu_architectures::inkling::DecoderInputPart::Text(tokens)
                    }
                    eredu_core::InputModality::Image => {
                        eredu_architectures::inkling::DecoderInputPart::Image(tokens)
                    }
                    eredu_core::InputModality::Audio => {
                        eredu_architectures::inkling::DecoderInputPart::Audio(tokens)
                    }
                    eredu_core::InputModality::Video => unreachable!(),
                },
            })
            .collect::<Vec<_>>();
        let audio =
            prepared
                .audio
                .as_ref()
                .map(|code_ids| eredu_architectures::inkling::AudioInput {
                    code_ids,
                    valid_frames: code_ids.as_array().dim(1),
                });
        let input = eredu_architectures::inkling::ModelInput {
            parts: &parts,
            vision_patches: prepared.images.as_ref(),
            audio,
        };
        // Media ingress executes before decoder pipeline ownership is applied.
        // The neutral architecture owns this transient geometry independently
        // of the composite target-plus-prediction persistence layout.
        let ingress_layout = self
            .architecture
            .ingress_state_layout()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let mut state = MlxHybridState::device(ingress_layout)?;
        let forward = match execution.and_then(ParallelExecutionContext::group) {
            Some(parallel) => <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as ParallelLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::begin_forward_parallel(&mut self.architecture, input, &mut state, parallel, stream),
            None => <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::begin_forward(&mut self.architecture, input, &mut state, stream),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(InklingIngressState { forward, state })
    }

    fn ingress_active(&self, state: &InklingIngressState) -> bool {
        let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )
        .expect("validated Inkling vision group");
        <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::should_execute_group(&self.architecture, vision_group, &state.forward.context)
    }

    fn replace_ingress_arrays(
        &self,
        state: &mut InklingIngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let [hidden]: [Array; 1] = arrays.try_into().map_err(|arrays: Vec<Array>| {
            Error::Parallel(format!(
                "Inkling placed ingress expected one activation, got {}",
                arrays.len()
            ))
        })?;
        state.replace_hidden(crate::MlxTensor::from_array(hidden));
        Ok(())
    }

    fn forward_vision_unit(
        &mut self,
        index: usize,
        layer: &mut InklingPipelineUnit,
        state: &mut InklingIngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        state.forward.hidden = match execution.and_then(ParallelExecutionContext::group) {
            Some(parallel) => <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as ParallelLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::forward_unit_parallel(
                &mut self.architecture,
                vision_group,
                index,
                &mut **layer,
                &state.forward.hidden,
                &mut state.state,
                &mut state.forward.context,
                parallel,
                stream,
            ),
            None => <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::forward_unit(
                &mut self.architecture,
                vision_group,
                index,
                &mut **layer,
                &state.forward.hidden,
                &mut state.state,
                &mut state.forward.context,
                stream,
            ),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(())
    }

    fn finish_ingress(
        &mut self,
        mut state: InklingIngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if self.ingress_active(&state) {
            let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
                &self.architecture,
                eredu_runtime::ArchitectureGroupKind::VisionEncoder,
            )?;
            state.forward.hidden = match execution.and_then(ParallelExecutionContext::group) {
                Some(parallel) => <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as ParallelLayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::complete_execution_group_parallel(
                    &mut self.architecture,
                    vision_group,
                    &state.forward.hidden,
                    &mut state.state,
                    &mut state.forward.context,
                    parallel,
                    stream,
                ),
                None => <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::complete_execution_group(
                    &mut self.architecture,
                    vision_group,
                    &state.forward.hidden,
                    &mut state.state,
                    &mut state.forward.context,
                    stream,
                ),
            }
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        }
        Ok(state.forward.hidden.into_array())
    }

    fn forward_pipeline_mtp(
        &mut self,
        hidden: &Array,
        tokens: &Array,
        depth: usize,
        state: &mut MlxHybridState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let output = self
            .architecture
            .forward_partition_mtp(
                crate::composition::tensor_ref(hidden),
                crate::composition::tensor_ref(tokens),
                depth,
                state.layers_mut(),
                execution.and_then(ParallelExecutionContext::group),
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(EmbeddedMtpOutput {
            logits: output.logits,
            hidden: output.hidden,
            tokens: output.tokens,
        })
    }
}

impl PipelinePartitionMetadata for InklingPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::inkling(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::inkling_input_part(
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
        // Predictor state is appended to the output owner's persistence
        // identity, but it is materialized in the transactional MTP cache.
        let target_identity = identity
            .select_state_segment(eredu_architectures::inkling::TARGET_STATE_SEGMENT)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        materialize_pipeline_cache_layers(&target_identity, paged)
    }
}

impl MlxPlacedGroupExecutor for InklingPipelinePartition {
    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_ingress(input, execution, stream)?);
        Ok(())
    }

    fn placed_ingress_active(&self, _group: &str) -> Result<bool, Error> {
        // Inkling's media group is also the pass-through producer for already
        // projected image embeddings and text-only requests. Keep the placed
        // route active even when there are no raw vision units to execute.
        self.ingress_state
            .as_ref()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state is unavailable".into()))?;
        Ok(true)
    }

    fn placed_ingress_arrays(&self, _group: &str) -> Result<Vec<Array>, Error> {
        let state = self
            .ingress_state
            .as_ref()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state is unavailable".into()))?;
        Ok(vec![state.hidden().clone()])
    }

    fn replace_placed_ingress_arrays(
        &mut self,
        _group: &str,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let mut state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state is unavailable".into()))?;
        let result = self.replace_ingress_arrays(&mut state, arrays);
        self.ingress_state = Some(state);
        result
    }

    fn merge_placed_ingress_arrays(&mut self, arrays: Vec<Array>) -> Result<(), Error> {
        let group = architecture_group_id_by_kind::<_, MlxHybridState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
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
        let vision_group = architecture_group_id_by_kind::<_, MlxHybridState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        if group != vision_group {
            return Ok(());
        }
        let mut state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state is unavailable".into()))?;
        let result = self.execute_placed_vision(&mut state, execution, stream);
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
            .ok_or_else(|| Error::Parallel("Inkling placed ingress state is unavailable".into()))?;
        Ok(PipelinePayload {
            hidden: self.finish_ingress(state, execution, stream)?,
            auxiliary: PipelineAuxiliaryState::default(),
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
        if mask.is_some() {
            return Err(Error::Parallel(
                "Inkling relative attention does not accept an additive mask".into(),
            ));
        }
        let mut state = self.begin_ingress(input, execution, stream)?;
        if self.ingress_active(&state) {
            let mut layers = std::mem::take(&mut self.vision_layers);
            let result =
                self.vision_range()
                    .clone()
                    .zip(&mut layers)
                    .try_for_each(|(index, layer)| {
                        self.forward_vision_unit(index, layer, &mut state, execution, stream)
                    });
            self.vision_layers = layers;
            result?;
        }
        let payload = PipelinePayload {
            hidden: self.finish_ingress(state, execution, stream)?,
            auxiliary: PipelineAuxiliaryState::default(),
        };
        self.forward_decoder(
            PipelineStageInput::Hidden(&payload),
            step,
            cache,
            execution,
            expert_group,
            stream,
            observer,
        )
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.begin_placed_ingress(input, execution, stream)
    }
}

impl PipelineEmbeddedMtp for InklingPipelinePartition {
    fn embedded_mtp_len(&self) -> usize {
        self.architecture.mtp_len()
    }

    fn new_embedded_mtp_cache(
        &self,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<PipelineMtpCache, Error> {
        let state = self
            .partition
            .local_geometry()
            .prediction_state()
            .ok_or_else(|| {
                Error::ArchitectureModel("Inkling checkpoint has no embedded MTP predictor".into())
            })?;
        let layout = state.layout().clone();
        let global_layer_start = state.global_layer_offset();
        let state = match paged {
            Some((manager, rank)) => MlxHybridState::paged_with_global_layer_start(
                layout,
                manager,
                rank,
                global_layer_start,
            )?,
            None => MlxHybridState::device_with_global_layer_start(layout, global_layer_start)?,
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
        _expert_group: Option<&Group>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error> {
        let PipelineMtpCache::Hybrid(cache) = cache else {
            return Err(Error::Parallel(
                "Inkling pipeline MTP cache mismatch".into(),
            ));
        };
        self.forward_pipeline_mtp(hidden, tokens, depth, cache, execution, stream)
    }

    fn embedded_mtp_state_segment(&self) -> Option<&'static str> {
        Some(eredu_architectures::inkling::PREDICTION_STATE_SEGMENT)
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

impl PipelineForward for InklingPipelinePartition {
    pipeline_observed_forward!();
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if mask.is_some() {
            return Err(Error::Parallel(
                "Inkling relative attention does not accept an additive mask".into(),
            ));
        }
        self.forward_decoder(input, step, cache, None, None, stream, None)
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
        if mask.is_some() {
            return Err(Error::Parallel(
                "Inkling relative attention does not accept an additive mask".into(),
            ));
        }
        self.forward_decoder(
            input,
            step,
            cache,
            execution,
            expert_group,
            stream,
            observer,
        )
    }
}

impl InklingPipelinePartition {
    fn new(
        architecture: eredu_architectures::inkling::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Arc<eredu_architectures::inkling::LocalGeometry>,
            eredu_runtime::NoAuxiliaryBoundarySchema,
        >,
    ) -> Result<Self, Error> {
        Ok(Self {
            architecture,
            partition,
            adapter: (),
            vision_layers: Vec::new(),
            audio_layers: Vec::new(),
            layers: Vec::new(),
            prediction_layers: Vec::new(),
            dense_layers: None,
            expert_assignment: None,
            expert_storage: PipelineExpertStorage::LayerLocal,
            routing_statistics: RoutingStatistics::default(),
            ingress_state: None,
        })
    }

    fn execute_placed_vision(
        &mut self,
        state: &mut InklingIngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        if !self.ingress_active(state) {
            return Ok(());
        }
        if let Some(storage) = self.dense_layers.take() {
            let result = (|| {
                let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
                    &self.architecture,
                    eredu_runtime::ArchitectureGroupKind::VisionEncoder,
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
                    let mut layer = self.build_unit(vision_group, index, stream)?;
                    populate_module_from_lease(
                        &mut layer,
                        transfer
                            .as_ref()
                            .map(|transfer| transfer.lease())
                            .or(lease.as_ref())
                            .expect("Inkling placed vision residency lease"),
                    )?;
                    self.forward_vision_unit(index, &mut layer, state, execution, stream)?;
                    synchronize_outputs([state.hidden()])?;
                    drop(transfer);
                    drop(lease);
                    if let Some(window) = &mut window {
                        window.refill()?;
                    } else {
                        storage.trim_after_absolute(ordinal)?;
                    }
                }
                storage.complete_forward()
            })();
            self.dense_layers = Some(storage);
            result?;
        } else {
            let mut layers = std::mem::take(&mut self.vision_layers);
            let result =
                self.vision_range()
                    .clone()
                    .zip(&mut layers)
                    .try_for_each(|(index, layer)| {
                        self.forward_vision_unit(index, layer, state, execution, stream)
                    });
            self.vision_layers = layers;
            result?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_decoder(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        caches: &mut [PipelineLayerCache],
        execution: Option<&ParallelExecutionContext<'_>>,
        expert_group: Option<&Group>,
        stream: &Stream,
        observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
    ) -> Result<PipelineStageOutput, Error> {
        if caches.len() != self.range().len() {
            return Err(Error::Parallel(format!(
                "Inkling stage has {} cache entries for {} layers",
                caches.len(),
                self.range().len()
            )));
        }
        let assignment = self.expert_assignment.clone();
        let expert_cache = self.expert_storage.cache();
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
        let decoder_range = self.range();
        if let Some(expert_cache) = expert_cache {
            let assignment = assignment.as_ref().ok_or_else(|| {
                Error::Parallel("Inkling external experts have no assignment".into())
            })?;
            let mut execute = |execution: GatedProductExpertExecution, stream: &Stream| {
                execute_pipeline_cached_neutral_inkling(
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
                None,
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
                None,
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

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_neutral_inkling(
    spec: &eredu_nn::GatedProductExpertBankSpec,
    cache_layer: usize,
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
        crate::composition::mlx::distributed::expert::execute_cached_neutral_inkling(
            spec,
            cache_layer,
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
pub(super) fn load_neutral_inkling_pipeline(
    source_args: eredu_architectures::inkling::ModelArgs,
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
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::Inkling], "Inkling")?;
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        InklingBindings::new_external_experts()
    } else {
        InklingBindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "Inkling pipeline",
                source_args.text_config.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let expert_quantization = quantize_on_load;
    let target_args = quantize_on_load
        .map(|quantization| {
            eredu_architectures::inkling::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        })
        .transpose()?
        .unwrap_or_else(|| source_args.clone());
    let target_binding_adapter = if external_experts {
        InklingBindings::new_external_experts()
    } else {
        InklingBindings::new()
    };
    let global_architecture = eredu_architectures::inkling::LayeredModel::<MlxNeuralBackend>::new(
        target_args.clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = global_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let global_decoder_group =
        architecture_decoder_group::<_, MlxHybridState>(&global_architecture)?;
    let target_units = architecture_group_unit_count(
        &binding_parameter_description,
        global_decoder_group,
        "Inkling decoder",
    )?;
    let planned_layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
    let source_architecture = if quantize_on_load.is_some() {
        let source_global = eredu_architectures::inkling::LayeredModel::<MlxNeuralBackend>::new(
            source_args.clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let source_description = source_global
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let source_layout = architecture_parallel_layout(&source_description, topology)?;
        let source_geometry = Arc::new(
            eredu_architectures::inkling::local_geometry(&source_args, &source_layout)
                .map_err(|error| Error::Parallel(error.to_string()))?,
        );
        Some(
            eredu_architectures::inkling::LayeredModel::<MlxNeuralBackend>::new_parallel(
                source_args.clone(),
                source_geometry,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        )
    } else {
        None
    };
    let geometry = Arc::new(
        eredu_architectures::inkling::local_geometry(&target_args, &planned_layout)
            .map_err(|error| Error::Parallel(error.to_string()))?,
    );
    let architecture =
        eredu_architectures::inkling::LayeredModel::<MlxNeuralBackend>::new_parallel(
            target_args.clone(),
            Arc::clone(&geometry),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_realization = eredu_architectures::inkling::expert_realization_plan(
        &architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    preflight_pipeline_realization(
        topology,
        target_units,
        expert_realization.as_ref(),
        external_experts,
        "Inkling",
    )?;
    let range = topology.layer_range(target_units)?;
    let neutral_placement = Arc::new(media_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        wire_contract,
        range.clone(),
        Arc::clone(&neutral_placement),
        model_kind,
    );
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let partition = neutral_placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            topology.pipeline_parallel_rank,
            Arc::clone(&geometry),
            &parameter_description,
        )?;
    let mut stage = InklingPipelinePartition::new(architecture, partition)?;
    if external_experts {
        let assignment =
            ExpertAssignment::from_realization(expert_realization.as_ref().ok_or_else(|| {
                Error::Parallel(
                    "Inkling external experts require an architecture realization".into(),
                )
            })?)?;
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        stage.expert_assignment = Some(assignment);
        stage.expert_storage = PipelineExpertStorage::ExternalEmpty;
    }
    let parallel_layout = (topology.tensor_parallel_size > 1).then_some(planned_layout.clone());
    let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
        &stage.architecture,
        eredu_runtime::ArchitectureGroupKind::VisionEncoder,
    )?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&stage.architecture)?;
    stage.vision_layers = stage
        .vision_range()
        .map(|index| stage.build_unit(vision_group, index, stream))
        .collect::<Result<Vec<_>, _>>()?;
    stage.layers = stage
        .range()
        .map(|index| stage.build_unit(decoder_group, index, stream))
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let embedded_mtp_layers = stage.architecture.mtp_len();
    let owns_embedded_mtp = embedded_mtp_layers > 0
        && static_roles.contains(&eredu_architectures::inkling::MTP_STATIC_ROLE);
    info.owns_embedded_mtp = owns_embedded_mtp;
    info.embedded_mtp_layers = if owns_embedded_mtp {
        embedded_mtp_layers
    } else {
        0
    };
    info.global_embedded_mtp_layers = embedded_mtp_layers;
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let source_quantization = BoundPipelineBindings::new(
                &binding_adapter,
                source_architecture
                    .as_ref()
                    .expect("load-time quantization source architecture"),
            );
            let target_quantization =
                BoundPipelineBindings::new(&target_binding_adapter, &stage.architecture);
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
        &target_binding_adapter
    } else {
        &binding_adapter
    };
    info.materialization = materialization;
    let static_units = pipeline_binding_units(
        &BoundPipelineBindings::new(binding_adapter, &global_architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Inkling", &stage.partition);
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
    let inkling_resident_layers = dense_stream.is_none();
    if inkling_resident_layers {
        let architecture = &stage.architecture;
        for (index, layer) in stage.vision_range().clone().zip(&mut stage.vision_layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                &global_architecture,
                vision_group,
                index,
                store.as_ref(),
                parallel_layout.as_ref(),
                stream,
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
                quantize_on_load,
                weights_stream,
                stream,
            )?;
        }
        for (index, layer) in stage.range().clone().zip(&mut stage.layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                &global_architecture,
                decoder_group,
                index,
                store.as_ref(),
                parallel_layout.as_ref(),
                stream,
            )?;
            loaded.load_excluding_roles(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    decoder_group,
                    index,
                )?,
                layer,
                store.as_ref(),
                &bindings,
                quantize_on_load,
                weights_stream,
                stream,
                if external_experts {
                    &[eredu_runtime::ParameterRole::ExpertIntermediate]
                } else {
                    &[]
                },
            )?;
        }
    }
    let static_bytes = loaded.finish(&mut info)?;
    if let Some(options) = dense_stream {
        let layout = parallel_layout.clone();
        let architecture = &stage.architecture;
        let vision_start = stage.vision_range().start;
        let vision_count = stage.vision_range().len();
        let text_start = stage.range().start;
        let unit_count = vision_count + stage.range().len();
        let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
            architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        let decoder_group = architecture_decoder_group::<_, MlxHybridState>(architecture)?;
        let storage = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if external_experts {
                &[eredu_runtime::ParameterRole::ExpertIntermediate]
            } else {
                &[]
            },
            0..unit_count,
            options,
            static_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |ordinal, stream| {
                if ordinal < vision_count {
                    <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::build_unit(architecture, vision_group, vision_start + ordinal, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
                } else {
                    <eredu_architectures::inkling::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::build_unit(
                        architecture,
                        decoder_group,
                        text_start + ordinal - vision_count,
                        stream,
                    )
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
                }
            },
            |ordinal, _layer, store| {
                if ordinal < vision_count {
                    binding_adapter.cartesian_layer_bindings(
                        &global_architecture,
                        vision_group,
                        vision_start + ordinal,
                        store,
                        layout.as_ref(),
                        stream,
                    )
                } else {
                    binding_adapter.cartesian_layer_bindings(
                        &global_architecture,
                        decoder_group,
                        text_start + ordinal - vision_count,
                        store,
                        layout.as_ref(),
                        stream,
                    )
                }
            },
            |ordinal| {
                let (group, index) = if ordinal < vision_count {
                    (vision_group, vision_start + ordinal)
                } else {
                    (decoder_group, text_start + ordinal - vision_count)
                };
                architecture_parameter_unit_owner::<_, MlxHybridState>(architecture, group, index)
            },
        )?
        .with_execution_offset(vision_count)?;
        stage.dense_layers = Some(storage);
        let bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::Parallel("Inkling pipeline bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let assignment = stage
            .expert_assignment
            .as_ref()
            .expect("Inkling assignment");
        let catalog =
            eredu_architectures::inkling::expert_residency_catalog(&source_args, store.as_ref())
                .map_err(Error::ArchitectureModel)?;
        let units = crate::composition::select_architecture_expert_units(
            catalog,
            |group, unit| stage.partition.owns_unit(group.as_str(), unit),
            |identity| assignment.owner(identity.global_expert) == Some(assignment.rank()),
        );
        let entries = crate::composition::architecture_expert_units(
            units,
            store.as_ref(),
            parallel_layout.as_ref(),
        )?;
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
            .ok_or_else(|| Error::Parallel("Inkling expert bytes overflowed".into()))?;
        stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
    }
    let diagnostics = store.source_diagnostics()?;
    let mut materialized_shards = if info.materialization.is_some() {
        let mut shards = store.materialized_source_shards();
        shards.extend(checkpoint_backing_shards(
            store.as_ref(),
            info.owned_tensors.iter().map(String::as_str),
        )?);
        shards
    } else {
        checkpoint_backing_shards(
            store.as_ref(),
            info.owned_tensors.iter().map(String::as_str),
        )?
    };
    materialized_shards.sort();
    materialized_shards.dedup();
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}
