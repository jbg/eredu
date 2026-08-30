use std::{ops::Range, sync::Arc};

use eredu_architectures::{muse_glimmer as muse_glimmer_arch, ModelKind};
use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureParameters, ExpertCacheLoadOptions, ExpertPass,
    LayeredArchitecture, ParallelLayeredArchitecture,
};
use safemlx::{distributed::Group, error::Exception, Array, Stream};

use crate::{
    backend::{
        error::Error,
        nn::shared::{MlxModule, MlxNeuralBackend},
        runtime::{
            cache::state::MlxKeyValueState,
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
        architecture_decoder_group, architecture_group_by_kind, architecture_group_unit_count,
        architecture_parallel_layout, architecture_parameter_unit_owner,
        architecture_partition_range, base_info, build_pipeline_expert_cache,
        build_pipeline_layer_storage, checkpoint_backing_shards,
        execute_routed_layered_partition_observed, load_architecture_static_parameters,
        media_architecture_transport, pipeline_binding_units, preflight_pipeline_realization,
        quantize_pipeline_stage_store, validate_admitted_pipeline_kind,
        validate_pipeline_expert_dispatch, validate_scheduled_pipeline_kv_cache,
        BoundPipelineBindings, MuseGlimmerPipelinePartition, PipelineAuxiliaryState,
        PipelineExpertStorage, PipelineForward, PipelineLayerCache, PipelineLayerController,
        PipelineLayerLoadOptions, PipelineLayerStorage, PipelineLoadAccumulator, PipelineModel,
        PipelinePartitionMetadata, PipelinePayload, PipelinePlacedIngress, PipelineStageInput,
        PipelineStageOutput, PipelineStep,
    },
    composition::muse_glimmer::{MuseGlimmerPipelineBindings, MuseGlimmerPlacedState},
};

impl PipelinePartitionMetadata for MuseGlimmerPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::muse_glimmer(self.architecture.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::muse_glimmer_input_part(
            self.architecture.args(),
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

impl PipelinePlacedIngress for MuseGlimmerPipelinePartition {
    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_placed_input(input, execution, stream)?);
        Ok(())
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_placed_input(input, None, stream)?);
        Ok(())
    }

    fn placed_ingress_active(&self, _group: &str) -> Result<bool, Error> {
        let state = self.ingress_state.as_ref().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer placed ingress state is unavailable".into())
        })?;
        let vision_group = architecture_group_by_kind::<_, MlxKeyValueState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        Ok(
            <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
            >>::should_execute_group(
                &self.architecture, vision_group, &state.forward.context
            ),
        )
    }

    fn placed_ingress_arrays(&self, _group: &str) -> Result<Vec<Array>, Error> {
        let state = self.ingress_state.as_ref().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer placed ingress state is unavailable".into())
        })?;
        Ok(vec![state.hidden().as_array().clone()])
    }

    fn replace_placed_ingress_arrays(
        &mut self,
        _group: &str,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let state = self.ingress_state.as_mut().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer placed ingress state is unavailable".into())
        })?;
        let [hidden]: [Array; 1] = arrays.try_into().map_err(|arrays: Vec<Array>| {
            Error::Parallel(format!(
                "Muse-Glimmer placed ingress expected one activation, got {}",
                arrays.len()
            ))
        })?;
        state.replace_hidden(crate::MlxTensor::from_array(hidden));
        Ok(())
    }

    fn merge_placed_ingress_arrays(&mut self, arrays: Vec<Array>) -> Result<(), Error> {
        let state = self.ingress_state.as_mut().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer placed ingress state is unavailable".into())
        })?;
        let [hidden]: [Array; 1] = arrays.try_into().map_err(|arrays: Vec<Array>| {
            Error::Parallel(format!(
                "Muse-Glimmer placed ingress expected one activation, got {}",
                arrays.len()
            ))
        })?;
        state.replace_hidden(crate::MlxTensor::from_array(hidden));
        Ok(())
    }

    fn execute_placed_ingress(
        &mut self,
        _group: &str,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let mut state = self.ingress_state.take().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer placed ingress state is unavailable".into())
        })?;
        let result = self.execute_placed_vision(&mut state, execution, stream);
        self.ingress_state = Some(state);
        result
    }

    fn finish_placed_ingress(
        &mut self,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let mut state = self.ingress_state.take().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer placed ingress state is unavailable".into())
        })?;
        let vision_group = architecture_group_by_kind::<_, MlxKeyValueState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        if <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
            MlxNeuralBackend,
            MlxKeyValueState,
        >>::should_execute_group(&self.architecture, vision_group, &state.forward.context)
        {
            state.forward.hidden =
                <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxKeyValueState,
                >>::complete_execution_group(
                    &mut self.architecture,
                    vision_group,
                    &state.forward.hidden,
                    &mut state.state,
                    &mut state.forward.context,
                    stream,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        }
        let hidden = state.forward.hidden;
        Ok(PipelinePayload {
            hidden: hidden.into_array(),
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
        let mut ingress = self.begin_placed_input(input, execution, stream)?;
        self.execute_placed_vision(&mut ingress, execution, stream)?;
        let vision_group = architecture_group_by_kind::<_, MlxKeyValueState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        if <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
            MlxNeuralBackend,
            MlxKeyValueState,
        >>::should_execute_group(
            &self.architecture, vision_group, &ingress.forward.context
        ) {
            ingress.forward.hidden =
                <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxKeyValueState,
                >>::complete_execution_group(
                    &mut self.architecture,
                    vision_group,
                    &ingress.forward.hidden,
                    &mut ingress.state,
                    &mut ingress.forward.context,
                    stream,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        }
        let payload = PipelinePayload {
            hidden: ingress.forward.hidden.into_array(),
            auxiliary: PipelineAuxiliaryState::default(),
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

impl PipelineForward for MuseGlimmerPipelinePartition {
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

pub(super) fn load_muse_glimmer_pipeline(
    source_args: muse_glimmer_arch::DecoderConfig,
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
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::MuseGlimmer], "Muse-Glimmer")?;
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    let binding_adapter = if external_experts {
        MuseGlimmerPipelineBindings::new_external_experts()
    } else {
        MuseGlimmerPipelineBindings::new()
    };
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load("Muse-Glimmer pipeline", source_args.quantization, requested)
                .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let expert_quantization = quantize_on_load;
    let target_args = quantize_on_load
        .map(|quantization| {
            eredu_architectures::muse_glimmer::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        })
        .transpose()?
        .unwrap_or_else(|| source_args.clone());
    let target_binding_adapter = if external_experts {
        MuseGlimmerPipelineBindings::new_external_experts()
    } else {
        MuseGlimmerPipelineBindings::new()
    };
    let seed_architecture =
        muse_glimmer_arch::LayeredModel::<MlxNeuralBackend>::new(target_args.clone(), stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = seed_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let binding_decoder_group =
        architecture_decoder_group::<_, MlxKeyValueState>(&seed_architecture)?;
    let target_units = architecture_group_unit_count(
        &binding_parameter_description,
        binding_decoder_group,
        "Muse-Glimmer decoder",
    )?;
    let (architecture, parallel_layout) = if topology.tensor_parallel_size > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry = muse_glimmer_arch::local_geometry(&target_args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let architecture = muse_glimmer_arch::LayeredModel::<MlxNeuralBackend>::new_parallel(
            target_args.clone(),
            geometry,
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        (architecture, Some(layout))
    } else {
        (seed_architecture, None)
    };
    let expert_realization = eredu_architectures::muse_glimmer::expert_realization_plan(
        &architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    preflight_pipeline_realization(
        topology,
        target_units,
        expert_realization.as_ref(),
        external_experts,
        "Muse-Glimmer",
    )?;
    let range = topology.layer_range(target_units)?;
    let placement = Arc::new(media_architecture_transport::<_, MlxKeyValueState>(
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
    let geometry = architecture.shared_parallel_geometry();
    let parameter_description = architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let partition = info
        .placement
        .realize_architecture_partition::<MlxNeuralBackend, MlxKeyValueState, _, _, _>(
            &architecture,
            info.pipeline_stage,
            geometry,
            &parameter_description,
        )?;
    let vision_group = architecture_group_by_kind::<_, MlxKeyValueState>(
        &architecture,
        eredu_runtime::ArchitectureGroupKind::VisionEncoder,
    )?;
    let decoder_group = architecture_decoder_group::<_, MlxKeyValueState>(&architecture)?;
    let vision_range = architecture_partition_range::<_, MlxKeyValueState, _>(
        &architecture,
        &partition,
        eredu_runtime::ArchitectureGroupKind::VisionEncoder,
    );
    let decoder_range = architecture_partition_range::<_, MlxKeyValueState, _>(
        &architecture,
        &partition,
        eredu_runtime::ArchitectureGroupKind::Decoder,
    );
    let vision_layers = vision_range
        .clone()
        .map(|index| {
            <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
            >>::build_unit(&architecture, vision_group, index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let layers = decoder_range
        .map(|index| {
            <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
            >>::build_unit(&architecture, decoder_group, index, stream)
            .map(MlxModule::new)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut stage = MuseGlimmerPipelinePartition {
        architecture,
        partition,
        adapter: (),
        vision_layers,
        audio_layers: Vec::new(),
        layers,
        prediction_layers: Vec::new(),
        dense_layers: None,
        expert_assignment: None,
        expert_storage: PipelineExpertStorage::LayerLocal,
        routing_statistics: RoutingStatistics::default(),
        ingress_state: None,
    };
    if external_experts {
        let assignment =
            ExpertAssignment::from_realization(expert_realization.as_ref().ok_or_else(|| {
                Error::Parallel(
                    "Muse-Glimmer external experts require an architecture realization".into(),
                )
            })?)?;
        info.global_expert_count = Some(assignment.global_expert_count());
        info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        stage.expert_assignment = Some(assignment);
        stage.expert_storage = PipelineExpertStorage::ExternalEmpty;
    }
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let source_architecture = match stage.architecture.shared_parallel_geometry() {
                Some(geometry) => {
                    muse_glimmer_arch::LayeredModel::<MlxNeuralBackend>::new_parallel(
                        source_args.clone(),
                        (*geometry).clone(),
                        stream,
                    )
                }
                None => muse_glimmer_arch::LayeredModel::<MlxNeuralBackend>::new(
                    source_args.clone(),
                    stream,
                ),
            }
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            let source_quantization =
                BoundPipelineBindings::new(&binding_adapter, &source_architecture);
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
        &BoundPipelineBindings::new(binding_adapter, &stage.architecture),
        &stage.partition,
        store.as_ref(),
        &static_roles,
    )?;
    let mut loaded = PipelineLoadAccumulator::new("Muse-Glimmer", &stage.partition);
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
        for (index, layer) in stage.vision_range().clone().zip(&mut stage.vision_layers) {
            let bindings = binding_adapter.cartesian_layer_bindings(
                architecture,
                decoder_group,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                None,
            )?;
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxKeyValueState>(
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
                architecture,
                decoder_group,
                index,
                layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                None,
            )?;
            loaded.load_excluding_roles(
                architecture_parameter_unit_owner::<_, MlxKeyValueState>(
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
        let vision_start = stage.vision_range().start;
        let vision_count = stage.vision_range().len();
        let text_start = stage.range().start;
        let unit_count = vision_count + stage.range().len();
        let architecture = &stage.architecture;
        let dense_layers = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            if expert_cache_options.is_some() {
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
                let (group, index) = if ordinal < vision_count {
                    (vision_group, vision_start + ordinal)
                } else {
                    (decoder_group, text_start + ordinal - vision_count)
                };
                <eredu_architectures::muse_glimmer::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxKeyValueState,
                >>::build_unit(architecture, group, index, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            },
            |ordinal, layer, store| {
                if ordinal < vision_count {
                    binding_adapter.cartesian_layer_bindings(
                        architecture,
                        vision_group,
                        vision_start + ordinal,
                        layer,
                        store,
                        layout.as_ref(),
                        None,
                    )
                } else {
                    binding_adapter.cartesian_layer_bindings(
                        architecture,
                        decoder_group,
                        text_start + ordinal - vision_count,
                        layer,
                        store,
                        layout.as_ref(),
                        None,
                    )
                }
            },
            |ordinal| {
                let (group, index) = if ordinal < vision_count {
                    (vision_group, vision_start + ordinal)
                } else {
                    (decoder_group, text_start + ordinal - vision_count)
                };
                architecture_parameter_unit_owner::<_, MlxKeyValueState>(architecture, group, index)
            },
        )?
        .with_execution_offset(vision_count)?;
        stage.dense_layers = Some(dense_layers);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes =
            static_bytes.checked_add(layer_bytes).ok_or_else(|| {
                Error::Parallel("Muse-Glimmer pipeline planned bytes overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let assignment = stage.expert_assignment.as_ref().ok_or_else(|| {
            Error::Parallel("Muse-Glimmer external experts have no assignment".into())
        })?;
        let catalog = eredu_architectures::muse_glimmer::expert_residency_catalog(
            store.as_ref(),
            &source_args,
        )
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
            .ok_or_else(|| Error::Parallel("Muse-Glimmer expert bytes overflowed".into()))?;
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

impl MuseGlimmerPipelinePartition {
    fn range(&self) -> Range<usize> {
        self.media_range::<MlxKeyValueState>(eredu_runtime::ArchitectureGroupKind::Decoder)
    }

    fn vision_range(&self) -> Range<usize> {
        self.media_range::<MlxKeyValueState>(eredu_runtime::ArchitectureGroupKind::VisionEncoder)
    }

    fn begin_placed_input(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<MuseGlimmerPlacedState, Error> {
        let prepared = crate::composition::muse_glimmer::prepare_muse_input(
            self.architecture.args(),
            input,
            stream,
        )?;
        let parts = prepared
            .tokens
            .iter()
            .zip(&prepared.media)
            .map(|(tokens, media)| {
                if *media {
                    muse_glimmer_arch::DecoderInputPart::Media(tokens)
                } else {
                    muse_glimmer_arch::DecoderInputPart::Text(tokens)
                }
            })
            .collect::<Vec<_>>();
        let mut state = MlxKeyValueState::device(self.architecture.state_layout()?)?;
        let model_input = muse_glimmer_arch::ModelInput {
            parts: &parts,
            vision: prepared
                .pixels
                .as_ref()
                .map(|pixels| muse_glimmer_arch::VisionInput {
                    pixels,
                    grid: &prepared.grid,
                }),
            mask: None,
        };
        let forward = if let Some(execution) = execution.filter(|value| value.is_tensor_parallel())
        {
            <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as ParallelLayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
            >>::begin_forward_parallel(
                &mut self.architecture,
                model_input,
                &mut state,
                execution
                    .group()
                    .ok_or_else(|| Error::Parallel("Muse-Glimmer TP group is missing".into()))?,
                stream,
            )
        } else {
            <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                MlxNeuralBackend,
                MlxKeyValueState,
            >>::begin_forward(&mut self.architecture, model_input, &mut state, stream)
        };
        let forward = forward.map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(MuseGlimmerPlacedState::new(forward, state))
    }

    fn execute_placed_vision(
        &mut self,
        state: &mut MuseGlimmerPlacedState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let vision_group = architecture_group_by_kind::<_, MlxKeyValueState>(
            &self.architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
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
                let mut layer = MlxModule::new(
                    <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxKeyValueState,
                    >>::build_unit(
                        &self.architecture, vision_group, index, stream
                    )
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
                );
                populate_module_from_lease(
                    &mut layer,
                    transfer
                        .as_ref()
                        .map(|transfer| transfer.lease())
                        .or(lease.as_ref())
                        .expect("Muse-Glimmer placed vision residency lease"),
                )?;
                let hidden = if let Some(execution) =
                    execution.filter(|value| value.is_tensor_parallel())
                {
                    <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as ParallelLayeredArchitecture<
                        MlxNeuralBackend,
                        MlxKeyValueState,
                    >>::forward_unit_parallel(
                        &mut self.architecture,
                        vision_group,
                        index,
                        &mut *layer,
                        &state.forward.hidden,
                        &mut state.state,
                        &mut state.forward.context,
                        execution.group().ok_or_else(|| {
                            Error::Parallel("Muse-Glimmer TP group is missing".into())
                        })?,
                        stream,
                    )
                } else {
                    <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxKeyValueState,
                    >>::forward_unit(
                        &mut self.architecture,
                        vision_group,
                        index,
                        &mut *layer,
                        &state.forward.hidden,
                        &mut state.state,
                        &mut state.forward.context,
                        stream,
                    )
                }
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                state.forward.hidden = hidden;
                synchronize_outputs([state.hidden().as_array()])?;
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
                let hidden = if let Some(execution) =
                    execution.filter(|value| value.is_tensor_parallel())
                {
                    <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as ParallelLayeredArchitecture<
                        MlxNeuralBackend,
                        MlxKeyValueState,
                    >>::forward_unit_parallel(
                        &mut self.architecture,
                        vision_group,
                        index,
                        &mut **layer,
                        &state.forward.hidden,
                        &mut state.state,
                        &mut state.forward.context,
                        execution.group().ok_or_else(|| {
                            Error::Parallel("Muse-Glimmer TP group is missing".into())
                        })?,
                        stream,
                    )
                } else {
                    <muse_glimmer_arch::LayeredModel<MlxNeuralBackend> as LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxKeyValueState,
                    >>::forward_unit(
                        &mut self.architecture,
                        vision_group,
                        index,
                        &mut **layer,
                        &state.forward.hidden,
                        &mut state.state,
                        &mut state.forward.context,
                        stream,
                    )
                }
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                state.forward.hidden = hidden;
            }
        }
        Ok(())
    }

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
        validate_scheduled_pipeline_kv_cache(
            "Muse-Glimmer",
            self.range().clone(),
            &self.architecture.args().attention_schedule,
            caches,
        )?;
        let assignment = self.expert_assignment.clone();
        let expert_cache = self.expert_storage.cache();
        if let Some(assignment) = assignment.as_ref() {
            validate_pipeline_expert_dispatch(
                assignment,
                expert_group,
                self.expert_storage.is_external(),
            )?;
        } else if expert_group.is_some() || expert_cache.is_some() {
            return Err(Error::Parallel(
                "Muse-Glimmer stage has expert transport without an ownership assignment".into(),
            ));
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
                Error::Parallel("Muse-Glimmer external experts have no assignment".into())
            })?;
            let mut execute = |execution: GatedProductExpertExecution, stream: &Stream| {
                execute_pipeline_cached_muse_glimmer(
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
}

#[allow(clippy::too_many_arguments)]
fn execute_pipeline_cached_muse_glimmer(
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
        crate::composition::mlx::distributed::expert::execute_cached_muse_glimmer(
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
