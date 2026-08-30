use std::sync::Arc;

use crate::backend::runtime::distributed::Group;
use eredu_architectures::ModelKind;
use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
use eredu_core::cache::{CacheRankIdentity, PromptCacheModelIdentity};
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
            cache::{residency::CacheResidencyManager, state::MlxHybridState},
            distributed::{
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
                    Relu2ExpertExecution, Relu2ExpertExecutorProvider,
                    ResidentExpertExecutorProvider,
                },
            },
        },
        MlxParallelContext,
    },
    composition::mlx::distributed::pipeline::{
        architecture_decoder_group, architecture_parallel_layout,
        architecture_parameter_unit_owner, architecture_prediction_group,
        architecture_prediction_unit_ranges, base_info, build_pipeline_expert_cache,
        build_pipeline_layer_storage, execute_neutral_routed_output_group,
        execute_resident_distributed_experts, execute_routed_layered_partition_observed,
        load_architecture_static_parameters, local_architecture_parameter_bindings,
        materialize_pipeline_cache_layers, partition_owns_architecture_units,
        pipeline_binding_units, prediction_architecture_transport, preflight_pipeline_realization,
        quantize_pipeline_stage_store, validate_admitted_pipeline_kind,
        validate_pipeline_expert_dispatch, BoundPipelineBindings, NemotronHPipelinePartition,
        PipelineEmbeddedMtp, PipelineExpertStorage, PipelineForward, PipelineLayerCache,
        PipelineLayerLoadOptions, PipelineLayerStorage, PipelineLoadAccumulator, PipelineModel,
        PipelineMtpCache, PipelinePartitionMetadata, PipelineStageInput, PipelineStageOutput,
        PipelineStep,
    },
    composition::{mlx::speculative::embedded::EmbeddedMtpOutput, nemotron_h::NemotronHBindings},
};

impl PipelinePartitionMetadata for NemotronHPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::nemotron_h(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part(
            "nemotron_h",
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

    fn new_cache_layers(
        &self,
        identity: &PromptCacheModelIdentity,
        paged: Option<(CacheResidencyManager, Option<CacheRankIdentity>)>,
    ) -> Result<Vec<PipelineLayerCache>, Error> {
        // The final rank owns appended prediction state in its persisted
        // identity, but ordinary pipeline execution addresses target units
        // only; prediction groups use the transactional hybrid cache below.
        let target_identity = identity
            .select_state_segment(eredu_architectures::nemotron_h::TARGET_STATE_SEGMENT)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        materialize_pipeline_cache_layers(&target_identity, paged)
    }
}

impl PipelineEmbeddedMtp for NemotronHPipelinePartition {
    fn embedded_mtp_len(&self) -> usize {
        self.architecture.mtp_len()
    }

    fn embedded_mtp_state_segment(&self) -> Option<&'static str> {
        Some(eredu_architectures::nemotron_h::PREDICTION_STATE_SEGMENT)
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
    ) -> Result<crate::composition::mlx::speculative::embedded::EmbeddedMtpOutput, Error> {
        let PipelineMtpCache::Hybrid(cache) = cache else {
            return Err(Error::Parallel(
                "Nemotron-H pipeline MTP cache mismatch".into(),
            ));
        };
        if matches!(self.expert_storage, PipelineExpertStorage::External(_)) {
            let assignment = self.expert_assignment.clone().ok_or_else(|| {
                Error::Parallel("Nemotron-H pipeline MTP expert cache has no assignment".into())
            })?;
            let storage = std::mem::replace(
                &mut self.expert_storage,
                PipelineExpertStorage::ExternalEmpty,
            );
            let PipelineExpertStorage::External(expert_cache) = storage else {
                unreachable!("checked external Nemotron-H expert storage")
            };
            let mut statistics = std::mem::take(&mut self.routing_statistics);
            let mut execute = |execution: Relu2ExpertExecution, stream: &Stream| {
                execute_pipeline_cached_nemotron_h(
                    &execution.spec,
                    execution.layer,
                    &execution.hidden,
                    &execution.expert_ids,
                    &execution.route_weights,
                    ExpertPass::Decode,
                    expert_cache.as_ref(),
                    &assignment,
                    expert_group,
                    &mut statistics,
                    stream,
                )
                .map_err(|error| Exception::custom(error.to_string()))
            };
            let result = self.forward_mtp_draft_neutral(
                hidden,
                tokens,
                depth,
                cache,
                execution,
                Some(&mut execute),
                stream,
            );
            self.routing_statistics = statistics;
            self.expert_storage = PipelineExpertStorage::External(expert_cache);
            return result;
        }
        if expert_group.is_some() && self.mtp_depth_has_sparse(depth)? {
            return Err(Error::Parallel(
                "Nemotron-H pipeline MTP with EP requires rank-owned expert residency".into(),
            ));
        }
        self.forward_mtp_draft_neutral::<
                    fn(Relu2ExpertExecution, &Stream) -> Result<Array, Exception>,
                >(hidden, tokens, depth, cache, execution, None, stream)
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

impl PipelineForward for NemotronHPipelinePartition {
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
fn execute_pipeline_cached_nemotron_h(
    spec: &eredu_nn::Relu2ExpertBankSpec,
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
        crate::composition::mlx::distributed::expert::execute_cached_nemotron_h(
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

pub(super) fn load_nemotron_h_pipeline(
    source_args: eredu_architectures::nemotron_h::ModelArgs,
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
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::NemotronH], "Nemotron-H")?;
    let explicit_expert_cache = expert_cache_options.is_some();
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let external_experts = expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        NemotronHBindings::new_external_experts()
    } else {
        NemotronHBindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "Nemotron-H pipeline",
                source_args.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_args = quantize_on_load.map_or_else(
        || Ok(source_args.clone()),
        |quantization| {
            eredu_architectures::nemotron_h::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    let expert_quantization = quantize_on_load;
    let target_binding_adapter = if external_experts {
        NemotronHBindings::new_external_experts()
    } else {
        NemotronHBindings::new()
    };
    let global_architecture =
        eredu_architectures::nemotron_h::LayeredModel::<MlxNeuralBackend>::new(
            target_args.clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    // Released recipes describe global tensors. Bind global architecture probes
    // first, then apply the planner layout exactly once before loading local modules.
    let global_source_architecture = quantize_on_load
        .map(|_| {
            eredu_architectures::nemotron_h::LayeredModel::<MlxNeuralBackend>::new(
                source_args.clone(),
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .transpose()?;
    let binding_parameter_description = global_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&global_architecture)?;
    let target_units = binding_parameter_description
        .unit_layout()
        .group_range(decoder_group)
        .ok_or_else(|| Error::Parallel("Nemotron-H parameter plan has no target group".into()))?
        .len();
    let prediction_units = architecture_prediction_unit_ranges::<_, MlxHybridState>(
        &global_architecture,
        &binding_parameter_description,
    )?;
    let planned_layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
    let geometry = Arc::new(
        eredu_architectures::nemotron_h::local_geometry(&target_args, &planned_layout)
            .map_err(|error| Error::Parallel(error.to_string()))?,
    );
    let mut architecture =
        eredu_architectures::nemotron_h::LayeredModel::<MlxNeuralBackend>::new_parallel(
            target_args.clone(),
            (*geometry).clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_realization = eredu_architectures::nemotron_h::expert_realization_plan(
        &architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    preflight_pipeline_realization(
        topology,
        target_units,
        expert_realization.as_ref(),
        external_experts,
        "Nemotron-H",
    )?;
    let range = topology.layer_range(target_units)?;
    let neutral_placement = Arc::new(prediction_architecture_transport::<_, MlxHybridState>(
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
    if let Some(realization) = expert_realization.clone() {
        architecture.install_expert_realization(realization);
    }
    let mut stage = NemotronHPipelinePartition::new(architecture, partition, external_experts)?;
    let expert_assignment =
        binding_adapter.expert_parallel_assignment(expert_realization.as_ref())?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        if expert_realization.as_ref().is_some_and(|realization| {
            realization
                .unit_specs()
                .keys()
                .any(|(group, unit)| stage.partition.owns_unit(group.as_str(), *unit))
        }) {
            info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        }
    }
    let parallel_layout = (topology.tensor_parallel_size > 1).then_some(planned_layout.clone());
    stage.layers = stage
        .range()
        .map(|global_layer| stage.build_unit(decoder_group, global_layer, stream))
        .collect::<Result<Vec<_>, _>>()?;
    let owns_mtp = partition_owns_architecture_units(
        &stage.partition,
        prediction_units
            .iter()
            .map(|(group, units)| (*group, units.clone())),
    );
    info.owns_embedded_mtp = owns_mtp;
    info.embedded_mtp_layers = if owns_mtp { prediction_units.len() } else { 0 };
    info.global_embedded_mtp_layers = prediction_units.len();
    if owns_mtp {
        for (group, units) in &prediction_units {
            stage.prediction_layers.push(
                units
                    .clone()
                    .map(|index| stage.build_unit(*group, index, stream))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
    }
    let requested = quantize_on_load;
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match requested {
        Some(quantization) => {
            let mut selection = PipelineStageQuantizationSelection::new(
                &static_roles,
                decoder_group,
                stage.range().clone(),
            );
            if owns_mtp {
                for (group, units) in &prediction_units {
                    selection = selection.with_layer_group(*group, units.clone());
                }
            }
            let source_architecture = global_source_architecture
                .as_ref()
                .expect("requested Nemotron-H conversion has a source architecture");
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, source_architecture);
            let source_parameter_description = source_architecture
                .parameter_description(stream)
                .map_err(|error| Error::Parallel(error.to_string()))?;
            let source_binding_authority = local_architecture_parameter_bindings(
                &source_parameter_description,
                &stage.partition,
            );
            let target_quantization =
                BoundPipelineBindings::new(&target_binding_adapter, &global_architecture);
            let (store, report) = quantize_pipeline_stage_store(
                store,
                &source_quantization,
                &target_quantization,
                &source_binding_authority,
                selection,
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let requested = materialization.is_none().then_some(requested).flatten();
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
    let mut loaded = PipelineLoadAccumulator::new("Nemotron-H", &stage.partition);
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
            let binding_layer = global_architecture
                .construct_unit(decoder_group, global_layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &global_architecture,
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
        for (depth, layers) in stage.prediction_layers.iter_mut().enumerate() {
            let (group, units) = &prediction_units[depth];
            for (index, layer) in units.clone().zip(layers) {
                let binding_layer = global_architecture
                    .construct_unit(*group, index, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                let bindings = binding_adapter.cartesian_layer_bindings(
                    &global_architecture,
                    *group,
                    index,
                    &binding_layer,
                    store.as_ref(),
                    parallel_layout.as_ref(),
                    stage.expert_assignment.as_ref(),
                )?;
                if external_experts {
                    loaded.load_excluding_roles(
                        architecture_parameter_unit_owner::<_, MlxHybridState>(
                            architecture,
                            *group,
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
                            *group,
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
    }
    let static_bytes = loaded.finish(&mut info)?;
    let checkpoint_diagnostics_before_deferred = store.source_diagnostics()?;
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let architecture = &stage.architecture;
        let binding_architecture = &global_architecture;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if expert_cache_options.is_some() {
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
                architecture
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
                    binding_architecture,
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
                    architecture,
                    decoder_group,
                    global_layer,
                )
            },
        )?);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes =
            static_bytes.checked_add(layer_bytes).ok_or_else(|| {
                Error::Parallel("Nemotron-H pipeline planned bytes overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let entries = crate::composition::nemotron_h::expert_catalog_selected(
            &source_args,
            store.as_ref(),
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
                    Error::Parallel("Nemotron-H pipeline expert byte total overflowed".into())
                })?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    let checkpoint_diagnostics = if explicit_expert_cache {
        store.source_diagnostics()?
    } else {
        checkpoint_diagnostics_before_deferred
    };
    info.opened_checkpoint_shards = checkpoint_diagnostics.payload_shard_paths.clone();
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

impl NemotronHPipelinePartition {
    fn args(&self) -> &eredu_architectures::nemotron_h::ModelArgs {
        self.architecture.args()
    }

    fn mtp_depth_has_sparse(&self, depth: usize) -> Result<bool, Error> {
        let layers = self.prediction_layers.get(depth).ok_or_else(|| {
            Error::Parallel(format!("Nemotron-H has no MTP prediction depth {depth}"))
        })?;
        Ok(layers.iter().any(|layer| {
            matches!(
                &**layer,
                eredu_architectures::nemotron_h::Unit::Prediction(unit)
                    if matches!(
                        &unit.block.operator,
                        eredu_architectures::nemotron_h::Operator::Sparse(_)
                    )
            )
        }))
    }

    fn new(
        architecture: eredu_architectures::nemotron_h::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Arc<eredu_architectures::nemotron_h::LocalGeometry>,
            eredu_architectures::nemotron_h::TargetBoundarySchema,
        >,
        external_experts: bool,
    ) -> Result<Self, Error> {
        let target_group = architecture_decoder_group::<_, MlxHybridState>(&architecture)?;
        partition
            .groups()
            .iter()
            .find(|group| group.group_index() == target_group)
            .map(|group| group.global_units())
            .ok_or_else(|| Error::Parallel("Nemotron-H partition has no target group".into()))?;
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

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<MlxModule<eredu_architectures::nemotron_h::Unit<MlxNeuralBackend>>, Error> {
        self.architecture
            .construct_unit(group, index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
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
                "Nemotron-H stage cache has {} entries, expected {}",
                caches.len(),
                self.layers.len()
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
        self.routing_statistics = RoutingStatistics::default();
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let expert_cache = self.expert_storage.cache();
        let decoder_range = self.range();
        if let Some(expert_cache) = expert_cache {
            let assignment = assignment.as_ref().ok_or_else(|| {
                Error::Parallel("Nemotron-H external experts have no assignment".into())
            })?;
            let mut execute = |execution: Relu2ExpertExecution, stream: &Stream| {
                execute_pipeline_cached_nemotron_h(
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
                .map_err(|error| Exception::custom(error.to_string()))
            };
            let mut provider = Relu2ExpertExecutorProvider::new(&mut execute);
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
        } else if let Some(assignment) = assignment
            .as_ref()
            .filter(|_| !self.expert_storage.is_external())
        {
            let expert_group = expert_group.ok_or_else(|| {
                Error::Parallel("Nemotron-H expert assignment requires its EP communicator".into())
            })?;
            let mut execute =
                |bank: &mut <MlxNeuralBackend as RoutedNeuralBackend>::GatedProductExpertBank,
                 routed_hidden: &Array,
                 ids: &Array,
                 weights: &Array,
                 partitions: usize,
                 context: &Stream| {
                    execute_resident_distributed_experts(
                        bank,
                        routed_hidden,
                        ids,
                        weights,
                        partitions,
                        assignment,
                        expert_group,
                        &mut self.routing_statistics,
                        context,
                    )
                };
            let mut provider = ResidentExpertExecutorProvider::new(&mut execute);
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
    fn forward_mtp_draft_neutral<F>(
        &mut self,
        prior: &Array,
        tokens: &Array,
        depth: usize,
        state: &mut MlxHybridState,
        execution: Option<&ParallelExecutionContext<'_>>,
        execute: Option<&mut F>,
        stream: &Stream,
    ) -> Result<EmbeddedMtpOutput, Error>
    where
        F: FnMut(Relu2ExpertExecution, &Stream) -> Result<Array, Exception>,
    {
        let tensor_group = execution
            .filter(|execution| execution.is_tensor_parallel())
            .and_then(ParallelExecutionContext::group);
        let layers = self.prediction_layers.get_mut(depth).ok_or_else(|| {
            Error::Parallel(format!("Nemotron-H has no MTP prediction depth {depth}"))
        })?;
        let prediction_group =
            architecture_prediction_group::<_, MlxHybridState>(&self.architecture, depth)?;
        if layers.is_empty() {
            return Err(Error::Parallel(format!(
                "Nemotron-H MTP prediction depth {depth} is empty"
            )));
        }
        let input = eredu_architectures::nemotron_h::EmbeddedInput::Draft {
            tokens: crate::composition::tensor_ref(tokens),
            hidden: crate::composition::tensor_ref(prior),
            depth,
        };
        let (logits, hidden) = if let Some(execute) = execute {
            let mut provider = Relu2ExpertExecutorProvider::new(execute);
            execute_neutral_routed_output_group(
                &mut self.architecture,
                input,
                prediction_group,
                layers,
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
                layers,
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
