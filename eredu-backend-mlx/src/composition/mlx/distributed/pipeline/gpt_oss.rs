use std::{ops::Range, sync::Arc};

use crate::backend::runtime::distributed::Group;
use eredu_architectures::{gpt_oss as gpt_oss_arch, ModelKind};
use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
use eredu_nn::RoutedNeuralBackend;
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureParameters, ExpertCacheLoadOptions, ExpertPass,
};
use safemlx::{error::Exception, Array, Stream};

use crate::{
    backend::{
        error::Error,
        nn::shared::{MlxModule, MlxNeuralBackend},
        runtime::{
            distributed::{expert::RoutingStatistics, parallel::ParallelExecutionContext},
            execution::layerwise::PipelineStageQuantizationSelection,
            residency::{
                expert_cache::ExpertCache, expert_provider::ResidentExpertExecutorProvider,
            },
        },
        MlxParallelContext,
    },
    composition::gpt_oss as neutral_gpt_oss,
    composition::mlx::distributed::pipeline::{
        architecture_decoder_group, architecture_group_unit_count, architecture_parallel_layout,
        architecture_parameter_unit_owner, base_info, build_pipeline_expert_cache,
        build_pipeline_layer_storage, construct_gpt_oss_partition_unit,
        decoder_architecture_transport, execute_neutral_decoder_partition_observed,
        execute_neutral_routed_decoder_partition_observed, execute_resident_distributed_experts,
        load_architecture_static_parameters, pipeline_binding_units, quantize_pipeline_stage_store,
        validate_admitted_pipeline_kind, validate_pipeline_expert_dispatch, BoundPipelineBindings,
        DecoderPipelineBuilder, GptOssPipelinePartition, PipelineForward, PipelineLayerCache,
        PipelineLayerLoadOptions, PipelineLayerStorage, PipelineLoadAccumulator, PipelineModel,
        PipelinePartitionMetadata, PipelineRangeState, PipelineStageInput, PipelineStageOutput,
        PipelineStep,
    },
};

impl PipelinePartitionMetadata for GptOssPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::gpt_oss(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part(
            "gpt_oss",
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

impl PipelineForward for GptOssPipelinePartition {
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
                "resident GPT-OSS expert parallelism requires its EP communicator".into(),
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
                "GPT-OSS expert assignment requires its EP communicator".into(),
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

#[allow(clippy::too_many_arguments)]
pub(super) fn load_gpt_oss_pipeline(
    source_args: gpt_oss_arch::ModelArgs,
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
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::GptOss], "GPT-OSS")?;
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let binding_adapter = if expert_cache_options.is_some() {
        neutral_gpt_oss::GptOssPipelineBindings::new_external_experts()
    } else {
        neutral_gpt_oss::GptOssPipelineBindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "GPT-OSS pipeline dense matrices",
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
            eredu_architectures::gpt_oss::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    // Native expert banks remain checkpoint MXFP4. A load-time request applies
    // only to ordinary dense matrices selected by the neutral block schema.
    let expert_quantization = None;
    let target_binding_adapter = if expert_cache_options.is_some() {
        neutral_gpt_oss::GptOssPipelineBindings::new_external_experts()
    } else {
        neutral_gpt_oss::GptOssPipelineBindings::new()
    };
    let seed_architecture =
        gpt_oss_arch::LayeredModel::<MlxNeuralBackend>::new(target_args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_architecture =
        gpt_oss_arch::LayeredModel::<MlxNeuralBackend>::new(target_args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let source_binding_architecture = quantize_on_load
        .map(|_| {
            gpt_oss_arch::LayeredModel::<MlxNeuralBackend>::new(source_args.clone(), stream)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .transpose()?;
    let binding_parameter_description = seed_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let decoder_group =
        architecture_decoder_group::<_, PipelineRangeState<'_>>(&seed_architecture)?;
    let target_units = architecture_group_unit_count(
        &binding_parameter_description,
        decoder_group,
        "GPT-OSS decoder",
    )?;
    let seed_expert_realization = eredu_architectures::gpt_oss::expert_realization_plan(
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
    let mut stage = GptOssPipelinePartition::new(
        seed_architecture,
        range.clone(),
        expert_cache_options.is_some(),
        stream,
    )?;
    let seed_architecture = stage
        .architecture
        .take()
        .expect("GPT-OSS neutral architecture");
    let parallel_layout = if topology.tensor_parallel_size > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry = gpt_oss_arch::local_geometry(&target_args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        stage.architecture = Some(
            gpt_oss_arch::LayeredModel::<MlxNeuralBackend>::new_parallel(
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
    let placement = Arc::new(decoder_architecture_transport::<_, PipelineRangeState<'_>>(
        stage.architecture.as_ref().unwrap(),
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        wire_contract,
        range.clone(),
        placement,
        model_kind,
    );
    stage.expert_realization = eredu_architectures::gpt_oss::expert_realization_plan(
        stage.architecture.as_ref().unwrap(),
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
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
    let parameter_description = binding_parameter_description;
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
            construct_gpt_oss_partition_unit(
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
            let source_quantization = BoundPipelineBindings::new(
                &binding_adapter,
                source_binding_architecture
                    .as_ref()
                    .expect("load-time quantization source architecture"),
            );
            let target_quantization =
                BoundPipelineBindings::new(&target_binding_adapter, &binding_architecture);
            let decoder_group =
                architecture_decoder_group::<_, PipelineRangeState<'_>>(&binding_architecture)?;
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
    let mut loaded = PipelineLoadAccumulator::new("GPT-OSS", &stage.partition);
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
        let architecture = &stage.architecture;
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
                        architecture,
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
                        architecture,
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
    let checkpoint_diagnostics = store.source_diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.payload_shard_paths.clone();
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_realization = stage.expert_realization.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let streamed_architecture = &stage.architecture;
        let binding_architecture = &binding_architecture;
        let streamed_bindings = &stage.bindings;
        stage.dense_layers = Some(build_pipeline_layer_storage(
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
                construct_gpt_oss_partition_unit(
                    streamed_architecture,
                    streamed_bindings,
                    global_layer,
                    streamed_realization.as_ref(),
                    streamed_assignment.as_ref(),
                    stream,
                )
            },
            |global_layer, _layer, store| {
                binding_adapter.cartesian_layer_bindings(
                    binding_architecture,
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
                    streamed_architecture,
                    architecture_decoder_group::<_, PipelineRangeState<'_>>(streamed_architecture)?,
                    global_layer,
                )
            },
        )?);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("GPT-OSS pipeline planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let catalog =
            eredu_architectures::gpt_oss::expert_residency_catalog(store.as_ref(), &source_args)
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
        info.planned_owned_parameter_bytes = info
            .planned_owned_parameter_bytes
            .checked_add(cache.report()?.owned_bytes)
            .ok_or_else(|| {
                Error::Parallel("GPT-OSS pipeline expert byte total overflowed".into())
            })?;
        stage.expert_cache = Some(cache);
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

impl GptOssPipelinePartition {
    fn args(&self) -> &gpt_oss_arch::ModelArgs {
        self.architecture.args()
    }

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
            Error::Parallel("resident GPT-OSS experts have no rank-local assignment".into())
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
        let assignment = self
            .expert_assignment
            .clone()
            .ok_or_else(|| Error::Parallel("external GPT-OSS experts have no assignment".into()))?;
        validate_pipeline_expert_dispatch(&assignment, expert_group, true)?;
        let cache = self.expert_cache.take().ok_or_else(|| {
            Error::Parallel("external GPT-OSS expert cache is unavailable".into())
        })?;
        let local_args = self
            .architecture
            .shared_parallel_geometry()
            .and_then(|geometry| geometry.block(self.range().start).cloned())
            .unwrap_or_else(|| self.args().clone());
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut provider = neutral_gpt_oss::expert::distributed_provider(
            &local_args,
            &assignment,
            expert_group,
            &cache,
            &mut statistics,
        );
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
        self.expert_cache = Some(cache);
        result
    }

    fn new(
        architecture: gpt_oss_arch::LayeredModel<MlxNeuralBackend>,
        range: Range<usize>,
        external_experts: bool,
        stream: &Stream,
    ) -> Result<
        DecoderPipelineBuilder<
            gpt_oss_arch::LayeredModel<MlxNeuralBackend>,
            gpt_oss_arch::LocalGeometry,
            neutral_gpt_oss::GptOssPipelineBindings,
            MlxModule<gpt_oss_arch::TransformerBlock<MlxNeuralBackend>>,
        >,
        Error,
    > {
        let bindings = if external_experts {
            neutral_gpt_oss::GptOssPipelineBindings::new_external_experts()
        } else {
            neutral_gpt_oss::GptOssPipelineBindings::new()
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
