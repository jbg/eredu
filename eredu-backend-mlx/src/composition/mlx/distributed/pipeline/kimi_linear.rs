use std::{ops::Range, sync::Arc};

use eredu_architectures::ModelKind;
use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
use eredu_nn::{RoutedNeuralBackend, TensorParallelExpertOutput};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureParameters, ExpertCacheLoadOptions, ExpertPass,
};
use safemlx::{distributed::Group, error::Exception, Array, Stream};

use crate::{
    backend::{
        error::Error,
        nn::shared::{MlxModule, MlxNeuralBackend},
        runtime::{
            cache::state::MlxHybridState,
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
                    GatedProductExpertExecution, GatedProductExpertExecutionMode,
                    GatedProductExpertExecutorProvider, ResidentExpertExecutorProvider,
                },
            },
        },
        MlxParallelContext,
    },
    composition::kimi_linear::KimiLinearBindings,
    composition::mlx::distributed::pipeline::{
        architecture_decoder_group, architecture_group_unit_count, architecture_parallel_layout,
        architecture_parameter_unit_owner, base_info, build_pipeline_expert_cache,
        build_pipeline_layer_storage, checkpoint_backing_shards, decoder_architecture_transport,
        execute_resident_distributed_experts, execute_routed_layered_partition_observed,
        load_architecture_static_parameters, pipeline_binding_units, quantize_pipeline_stage_store,
        validate_admitted_pipeline_kind, validate_pipeline_expert_dispatch, BoundPipelineBindings,
        KimiLinearPipelinePartition, PipelineExpertStorage, PipelineForward, PipelineLayerCache,
        PipelineLayerLoadOptions, PipelineLayerStorage, PipelineLoadAccumulator, PipelineModel,
        PipelinePartitionMetadata, PipelineStageInput, PipelineStageOutput, PipelineStep,
    },
};

#[allow(clippy::too_many_arguments)]
fn execute_neutral_routed_kimi_partition_observed<P>(
    stage: &mut KimiLinearPipelinePartition,
    input: PipelineStageInput<'_>,
    step: PipelineStep,
    explicit_mask: Option<&Array>,
    caches: &mut [PipelineLayerCache],
    execution: Option<&ParallelExecutionContext<'_>>,
    pass: ExpertPass,
    provider: &mut P,
    stream: &Stream,
    observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
) -> Result<PipelineStageOutput, Error>
where
    P: eredu_runtime::RoutedExpertProvider<MlxNeuralBackend>,
    P::Error: std::fmt::Display,
{
    let storage_range = stage.range();
    execute_routed_layered_partition_observed(
        &mut stage.architecture,
        &stage.partition,
        storage_range,
        &mut stage.layers,
        stage.dense_layers.as_ref(),
        input,
        step,
        explicit_mask,
        caches,
        execution,
        pass,
        provider,
        stream,
        observer,
    )
}

impl PipelinePartitionMetadata for KimiLinearPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::kimi_linear(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part(
            "kimi_linear",
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

impl PipelineForward for KimiLinearPipelinePartition {
    pipeline_observed_forward!();
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if self.expert_storage.is_external() {
            self.forward_external_experts_neutral(
                input, step, mask, cache, None, None, stream, None,
            )
        } else {
            let pass = if step.sequence_length > 1 {
                ExpertPass::Prefill
            } else {
                ExpertPass::Decode
            };
            execute_neutral_routed_kimi_partition_observed(
                self,
                input,
                step,
                mask,
                cache,
                None,
                pass,
                &mut eredu_runtime::ResidentExpertProvider,
                stream,
                None,
            )
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
            if self.expert_storage.cache().is_some() {
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
                "Kimi expert assignment requires its EP communicator".into(),
            ));
        }
        match execution {
            Some(execution) if execution.is_tensor_parallel() => {
                if self.expert_storage.cache().is_some() {
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
                    let pass = if step.sequence_length > 1 {
                        ExpertPass::Prefill
                    } else {
                        ExpertPass::Decode
                    };
                    execute_neutral_routed_kimi_partition_observed(
                        self,
                        input,
                        step,
                        mask,
                        cache,
                        Some(execution),
                        pass,
                        &mut eredu_runtime::ResidentExpertProvider,
                        execution.stream(),
                        observer,
                    )
                }
            }
            _ if self.expert_storage.is_external() => self.forward_external_experts_neutral(
                input, step, mask, cache, None, None, stream, observer,
            ),
            _ => {
                let pass = if step.sequence_length > 1 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                execute_neutral_routed_kimi_partition_observed(
                    self,
                    input,
                    step,
                    mask,
                    cache,
                    None,
                    pass,
                    &mut eredu_runtime::ResidentExpertProvider,
                    stream,
                    observer,
                )
            }
        }
    }
}

/// An executable, rank-local piece of a pipeline-parallel model.

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_kimi_linear(
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
        crate::composition::mlx::distributed::expert::execute_cached_kimi_linear(
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
    if returned.reduced_output.shape() != hidden.shape() {
        return Err(Error::Parallel(format!(
            "cached Kimi expert output shape {:?} does not match hidden shape {:?}",
            returned.reduced_output.shape(),
            hidden.shape(),
        )));
    }
    Ok(returned.reduced_output)
}

pub(super) fn load_kimi_linear_pipeline(
    source_args: eredu_architectures::kimi_linear::ModelArgs,
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
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::KimiLinear], "Kimi Linear")?;
    let expert_cache_options = expert_cache_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let binding_adapter = if expert_cache_options.is_some() {
        KimiLinearBindings::new_external_experts()
    } else {
        KimiLinearBindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "Kimi Linear pipeline",
                source_args.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let target_args = quantize_on_load
        .map(|quantization| {
            eredu_architectures::kimi_linear::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        })
        .transpose()?
        .unwrap_or_else(|| source_args.clone());
    let target_binding_adapter = if expert_cache_options.is_some() {
        KimiLinearBindings::new_external_experts()
    } else {
        KimiLinearBindings::new()
    };
    let global_architecture =
        eredu_architectures::kimi_linear::LayeredModel::<MlxNeuralBackend>::new(
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
        "Kimi Linear decoder",
    )?;
    let seed_expert_realization = eredu_architectures::kimi_linear::expert_realization_plan(
        &global_architecture,
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
    let planned_layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
    let geometry = Arc::new(
        eredu_architectures::kimi_linear::local_geometry(&target_args, &planned_layout)
            .map_err(|error| Error::Parallel(error.to_string()))?,
    );
    let architecture =
        eredu_architectures::kimi_linear::LayeredModel::<MlxNeuralBackend>::new_parallel(
            target_args.clone(),
            (*geometry).clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let placement = Arc::new(decoder_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size,
    )?);
    let mut info = base_info(
        topology,
        wire_contract,
        range.clone(),
        placement,
        model_kind,
    );
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxHybridState, _, _, _>(
            &architecture,
            topology.pipeline_parallel_rank,
            Arc::clone(&geometry),
            &parameter_description,
        )?;
    let mut stage =
        KimiLinearPipelinePartition::new(architecture, partition, expert_cache_options.is_some())?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&stage.architecture)?;
    let expert_realization = eredu_architectures::kimi_linear::expert_realization_plan(
        &stage.architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_assignment =
        binding_adapter.expert_parallel_assignment(expert_realization.as_ref())?;
    stage.expert_assignment = expert_assignment;
    if let Some(assignment) = stage.expert_assignment.as_ref() {
        info.global_expert_count = Some(assignment.global_expert_count());
        if stage.range().any(|layer| {
            source_args.layer_policy(layer).is_some_and(|policy| {
                policy.feed_forward
                    == eredu_architectures::kimi_linear::FeedForwardPolicy::SparseMoe
            })
        }) {
            info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        }
    }
    let parallel_layout = (topology.tensor_parallel_size > 1).then_some(planned_layout.clone());
    stage.layers = stage
        .range()
        .map(|global_layer| stage.build_unit(global_layer, stream))
        .collect::<Result<Vec<_>, _>>()?;
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &stage.architecture);
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
                    stage.range(),
                ),
                quantization,
                stream,
            )?;
            (store, Some(report))
        }
        None => (store, None),
    };
    let expert_quantization = quantize_on_load;
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
    let mut loaded = PipelineLoadAccumulator::new("Kimi Linear", &stage.partition);
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
        for (global_layer, layer) in stage.range().zip(&mut stage.layers) {
            let binding_layer = global_architecture
                .construct_unit(global_decoder_group, global_layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let bindings = binding_adapter.cartesian_layer_bindings(
                &global_architecture,
                global_decoder_group,
                global_layer,
                &binding_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stage.expert_assignment.as_ref(),
            )?;
            if expert_cache_options.is_some() {
                loaded.load_excluding_roles(
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
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
                    architecture_parameter_unit_owner::<_, MlxHybridState>(
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
    let mut materialized_shards = if info.materialization.is_some() {
        store.materialized_source_shards()
    } else {
        Vec::new()
    };
    materialized_shards.extend(checkpoint_backing_shards(
        store.as_ref(),
        info.owned_tensors.iter().map(String::as_str),
    )?);
    materialized_shards.sort();
    materialized_shards.dedup();
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
            stage.range(),
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
                    .construct_unit(global_decoder_group, global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                binding_adapter.cartesian_layer_bindings(
                    binding_architecture,
                    global_decoder_group,
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
                Error::Parallel("Kimi Linear pipeline planned bytes overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = expert_cache_options {
        let catalog = eredu_architectures::kimi_linear::expert_residency_catalog(
            store.as_ref(),
            &source_args,
        )
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
        if !entries.is_empty() {
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
                    Error::Parallel("Kimi Linear pipeline expert byte total overflowed".into())
                })?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

impl KimiLinearPipelinePartition {
    fn args(&self) -> &eredu_architectures::kimi_linear::ModelArgs {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        self.partition.groups()[0].global_units()
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
            Error::Parallel("resident Kimi experts have no rank-local assignment".into())
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
        let result = execute_neutral_routed_kimi_partition_observed(
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
            .ok_or_else(|| Error::Parallel("external Kimi experts have no assignment".into()))?;
        validate_pipeline_expert_dispatch(&assignment, expert_group, true)?;
        let storage = std::mem::replace(
            &mut self.expert_storage,
            PipelineExpertStorage::ExternalEmpty,
        );
        let PipelineExpertStorage::External(cache) = storage else {
            self.expert_storage = storage;
            return Err(Error::Parallel(
                "external Kimi expert cache is unavailable".into(),
            ));
        };
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut execute = |execution: GatedProductExpertExecution, context: &Stream| {
            let mode = execution.mode;
            execute_pipeline_cached_kimi_linear(
                &execution.spec,
                execution.layer,
                &execution.hidden,
                &execution.expert_ids,
                &execution.route_weights,
                pass,
                &cache,
                &assignment,
                expert_group,
                &mut statistics,
                context,
            )
            .map(|reducible| match mode {
                GatedProductExpertExecutionMode::Complete => {
                    eredu_runtime::RoutedExpertTensorParallelOutput::Complete(reducible)
                }
                GatedProductExpertExecutionMode::TensorParallel { .. } => {
                    eredu_runtime::RoutedExpertTensorParallelOutput::Partial(
                        TensorParallelExpertOutput {
                            reducible,
                            post_reduce: None,
                        },
                    )
                }
            })
            .map_err(|error| Exception::custom(error.to_string()))
        };
        let mut provider = GatedProductExpertExecutorProvider::new(&mut execute);
        let result = execute_neutral_routed_kimi_partition_observed(
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
        self.expert_storage = PipelineExpertStorage::External(cache);
        result
    }

    fn new(
        architecture: eredu_architectures::kimi_linear::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Arc<eredu_architectures::kimi_linear::LocalGeometry>,
            eredu_runtime::NoAuxiliaryBoundarySchema,
        >,
        external_experts: bool,
    ) -> Result<Self, Error> {
        let [_group] = partition.groups() else {
            return Err(Error::Parallel(format!(
                "Kimi partition owns {} groups, expected one",
                partition.groups().len()
            )));
        };
        Ok(Self {
            architecture,
            partition,
            layers: Vec::new(),
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
        index: usize,
        stream: &Stream,
    ) -> Result<MlxModule<eredu_architectures::kimi_linear::Block<MlxNeuralBackend>>, Error> {
        let group = architecture_decoder_group::<_, MlxHybridState>(&self.architecture)?;
        self.architecture
            .construct_unit(group, index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }
}
