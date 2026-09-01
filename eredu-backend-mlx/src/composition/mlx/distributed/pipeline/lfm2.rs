use std::{ops::Range, sync::Arc};

use crate::composition::grouped_provider::*;

use crate::backend::runtime::distributed::Group;
use eredu_architectures::ModelKind;
use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
use eredu_nn::GroupedNeuralBackend;
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureParameters, ExpertCacheLoadOptions, ExpertPass,
};
use safemlx::{error::Exception, Array, Stream};

use crate::{
    backend::{
        error::Error,
        nn::shared::{MlxModule, MlxNeuralBackend},
        runtime::{
            cache::state::MlxHybridState, distributed::parallel::ParallelExecutionContext,
            execution::layerwise::PipelineStageQuantizationSelection,
            residency::parameter_bank::AddressableParameterBank,
        },
        MlxParallelContext,
    },
    composition::expert_dispatch::{
        dispatch_local_with, dispatch_replicated_with, ExpertAssignment, RoutingStatistics,
    },
    composition::lfm2::Lfm2Bindings,
    composition::mlx::distributed::pipeline::{
        architecture_decoder_group, architecture_group_unit_count, architecture_parallel_layout,
        architecture_parameter_unit_owner, base_info, build_pipeline_layer_storage,
        build_pipeline_parameter_bank, decoder_architecture_transport,
        execute_layered_partition_observed, execute_resident_distributed_experts,
        execute_routed_layered_partition_observed, load_architecture_static_parameters,
        pipeline_binding_units, quantize_pipeline_stage_store, validate_admitted_pipeline_kind,
        validate_pipeline_expert_dispatch, BoundPipelineBindings, Lfm2PipelinePartition,
        PipelineExpertStorage, PipelineForward, PipelineLayerCache, PipelineLayerLoadOptions,
        PipelineLayerStorage, PipelineLoadAccumulator, PipelineModel, PipelinePartitionMetadata,
        PipelineStageInput, PipelineStageOutput, PipelineStep,
    },
};

/// Executes an LFM2 partition through its neutral hybrid architecture.
#[allow(clippy::too_many_arguments)]
fn execute_neutral_lfm2_partition_observed(
    stage: &mut Lfm2PipelinePartition,
    input: PipelineStageInput<'_>,
    step: PipelineStep,
    explicit_mask: Option<&Array>,
    caches: &mut [PipelineLayerCache],
    execution: Option<&ParallelExecutionContext<'_>>,
    stream: &Stream,
    observer: Option<&mut dyn eredu_runtime::ActivationObserver<Array, Exception>>,
) -> Result<PipelineStageOutput, Error> {
    let storage_range = stage.range();
    execute_layered_partition_observed(
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
        stream,
        observer,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_neutral_routed_lfm2_partition_observed<P>(
    stage: &mut Lfm2PipelinePartition,
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
    P: eredu_runtime::TensorParallelRoutedExpertProvider<MlxNeuralBackend>,
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

impl PipelinePartitionMetadata for Lfm2PipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::lfm2(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part(
            "lfm2",
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

    fn parameter_bank(&self) -> Option<&AddressableParameterBank> {
        self.expert_storage.cache()
    }
}

impl PipelineForward for Lfm2PipelinePartition {
    pipeline_observed_forward!();
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        if self.expert_storage.cache().is_some() {
            self.forward_external_experts_neutral(
                input, step, mask, cache, None, None, stream, None,
            )
        } else if self
            .expert_assignment
            .as_ref()
            .is_some_and(|assignment| assignment.group_size() > 1)
        {
            Err(Error::Parallel(
                "resident LFM2 expert parallelism requires its EP communicator".into(),
            ))
        } else {
            execute_neutral_lfm2_partition_observed(
                self, input, step, mask, cache, None, stream, None,
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
                "LFM2 expert assignment requires its EP communicator".into(),
            ));
        }
        match execution {
            Some(execution)
                if execution.is_tensor_parallel()
                    && !self.expert_storage.is_external()
                    && self.expert_assignment.is_none() =>
            {
                execute_neutral_lfm2_partition_observed(
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
                    execute_neutral_lfm2_partition_observed(
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
            _ if self.expert_storage.cache().is_some() => self.forward_external_experts_neutral(
                input, step, mask, cache, None, None, stream, observer,
            ),
            _ => execute_neutral_lfm2_partition_observed(
                self, input, step, mask, cache, None, stream, observer,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_lfm2(
    spec: &eredu_nn::GroupedGatedProductSpec,
    global_layer: usize,
    hidden: &Array,
    group_indices: &Array,
    weights: &Array,
    pass: ExpertPass,
    cache: &AddressableParameterBank,
    assignment: &ExpertAssignment,
    expert_group: Option<&Group>,
    statistics: &mut RoutingStatistics,
    stream: &Stream,
) -> Result<Array, Error> {
    validate_pipeline_expert_dispatch(assignment, expert_group, true)?;
    let execute = |routes: &crate::composition::expert_dispatch::DispatchedRoutes,
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
            hidden,
            group_indices,
            weights,
            assignment,
            group,
            stream,
            execute,
        )?,
        None => dispatch_local_with(hidden, group_indices, weights, assignment, stream, execute)?,
    };
    statistics.accumulate(&returned.statistics);
    Ok(returned.reduced_output)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_lfm2_pipeline(
    source_args: eredu_architectures::lfm2::ModelArgs,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelContext,
    wire_contract: eredu_runtime::PipelineWireContract,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    parameter_bank_options: Option<ExpertCacheLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::Lfm2], "LFM2")?;
    let parameter_bank_options = parameter_bank_options
        .or_else(|| (topology.expert_parallel_size > 1).then(ExpertCacheLoadOptions::default));
    let binding_adapter = if parameter_bank_options.is_some() {
        Lfm2Bindings::new_external_experts()
    } else {
        Lfm2Bindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "LFM2 pipeline",
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
            eredu_architectures::lfm2::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    let expert_quantization = quantize_on_load;
    let target_binding_adapter = if parameter_bank_options.is_some() {
        Lfm2Bindings::new_external_experts()
    } else {
        Lfm2Bindings::new()
    };
    let global_architecture = eredu_architectures::lfm2::LayeredModel::<MlxNeuralBackend>::new(
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
        "LFM2 decoder",
    )?;
    let seed_expert_realization = eredu_architectures::lfm2::expert_realization_plan(
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
    let source_architecture = if quantize_on_load.is_some() {
        let source_global = eredu_architectures::lfm2::LayeredModel::<MlxNeuralBackend>::new(
            source_args.clone(),
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let source_description = source_global
            .parameter_description(stream)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let source_layout = architecture_parallel_layout(&source_description, topology)?;
        let source_geometry =
            eredu_architectures::lfm2::local_geometry(&source_args, &source_layout)
                .map_err(|error| Error::Parallel(error.to_string()))?;
        Some(
            eredu_architectures::lfm2::LayeredModel::<MlxNeuralBackend>::new_parallel(
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
        eredu_architectures::lfm2::local_geometry(&target_args, &planned_layout)
            .map_err(|error| Error::Parallel(error.to_string()))?,
    );
    let architecture = eredu_architectures::lfm2::LayeredModel::<MlxNeuralBackend>::new_parallel(
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
        eredu_architectures::decoder::TARGET_EXECUTION_GROUP,
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
        Lfm2PipelinePartition::new(architecture, partition, parameter_bank_options.is_some())?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&stage.architecture)?;
    let expert_realization = eredu_architectures::lfm2::expert_realization_plan(
        &stage.architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
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
            info.local_group_indices = assignment.local_global_group_indices().to_vec();
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
                    stage.range(),
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
        &BoundPipelineBindings::new(binding_adapter, &global_architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("LFM2", &stage.partition);
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
            if parameter_bank_options.is_some() {
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
    let materialized_shards = checkpoint_diagnostics.payload_shard_paths.clone();
    if let Some(options) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_assignment = stage.expert_assignment.clone();
        let architecture = &stage.architecture;
        let binding_architecture = &global_architecture;
        stage.dense_layers = Some(build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if parameter_bank_options.is_some() {
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
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| Error::Parallel("LFM2 pipeline planned bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if let Some(options) = parameter_bank_options {
        let catalog =
            eredu_architectures::lfm2::expert_residency_catalog(store.as_ref(), &source_args)
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
            let cache = build_pipeline_parameter_bank(
                Arc::clone(&store),
                entries,
                Some(options),
                expert_quantization,
                weights_stream,
                stream,
            )?;
            info.planned_owned_parameter_bytes = info
                .planned_owned_parameter_bytes
                .checked_add(cache.report()?.owned_bytes())
                .ok_or_else(|| {
                    Error::Parallel("LFM2 pipeline expert byte total overflowed".into())
                })?;
            stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
        }
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}

impl Lfm2PipelinePartition {
    fn args(&self) -> &eredu_architectures::lfm2::ModelArgs {
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
            Error::Parallel("resident LFM2 experts have no rank-local assignment".into())
        })?;
        validate_pipeline_expert_dispatch(&assignment, Some(expert_group), false)?;
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut execute =
            |bank: &mut <MlxNeuralBackend as GroupedNeuralBackend>::GatedProductGroups,
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
        let mut provider = ResidentGroupedExecutorProvider::new(&mut execute);
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let result = execute_neutral_routed_lfm2_partition_observed(
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
            .ok_or_else(|| Error::Parallel("external LFM2 experts have no assignment".into()))?;
        validate_pipeline_expert_dispatch(&assignment, expert_group, true)?;
        let storage = std::mem::replace(
            &mut self.expert_storage,
            PipelineExpertStorage::ExternalEmpty,
        );
        let PipelineExpertStorage::External(cache) = storage else {
            self.expert_storage = storage;
            return Err(Error::Parallel(
                "external LFM2 expert cache is unavailable".into(),
            ));
        };
        let pass = if step.sequence_length > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        let mut statistics = std::mem::take(&mut self.routing_statistics);
        let mut execute = |execution: GatedProductGroupExecution, context: &Stream| {
            execute_pipeline_cached_lfm2(
                &execution.spec,
                execution.layer,
                &execution.hidden,
                &execution.group_indices,
                &execution.coefficients,
                pass,
                &cache,
                &assignment,
                expert_group,
                &mut statistics,
                context,
            )
            .map(eredu_runtime::RoutedExpertTensorParallelOutput::Complete)
            .map_err(|error| Exception::custom(error.to_string()))
        };
        let mut provider = GatedProductGroupedExecutorProvider::new(&mut execute);
        let result = execute_neutral_routed_lfm2_partition_observed(
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
        architecture: eredu_architectures::lfm2::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Arc<eredu_architectures::lfm2::LocalGeometry>,
            eredu_runtime::NoAuxiliaryBoundarySchema,
        >,
        external_experts: bool,
    ) -> Result<Self, Error> {
        let [_group] = partition.groups() else {
            return Err(Error::Parallel(format!(
                "LFM2 partition owns {} groups, expected one",
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
    ) -> Result<MlxModule<eredu_architectures::lfm2::Block<MlxNeuralBackend>>, Error> {
        let group = architecture_decoder_group::<_, MlxHybridState>(&self.architecture)?;
        self.architecture
            .construct_unit(group, index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }
}
