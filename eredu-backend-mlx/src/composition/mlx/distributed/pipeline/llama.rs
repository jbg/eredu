use std::sync::Arc;

use crate::backend::runtime::distributed::Group;
use eredu_architectures::{llama::ModelArgs as LlamaModelArgs, ModelKind};
use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
use eredu_runtime::{ArchitectureBoundary, ArchitectureParameters};
use safemlx::{error::Exception, Array, Stream};

use crate::{
    backend::{
        error::Error,
        nn::shared::{MlxModule, MlxNeuralBackend},
        runtime::{
            cache::state::MlxHybridState, distributed::parallel::ParallelExecutionContext,
            residency::parameter_bank::AddressableParameterBank,
        },
    },
    composition::expert_dispatch::RoutingStatistics,
    composition::mlx::distributed::pipeline::{
        architecture_decoder_group, architecture_group_unit_count, architecture_parallel_layout,
        architecture_parameter_unit_owner, base_info, build_pipeline_layer_storage,
        decoder_architecture_transport, execute_neutral_decoder_partition_observed,
        load_architecture_static_parameters, select_static_binding_units_by_owner,
        validate_admitted_pipeline_kind, LlamaPipelinePartition, PipelineForward,
        PipelineLayerCache, PipelineLayerLoadOptions, PipelineLayerStorage,
        PipelineLoadAccumulator, PipelineModel, PipelinePartitionMetadata, PipelineStageInput,
        PipelineStageOutput, PipelineStep,
    },
    composition::mlx::distributed::topology::MlxParallelPlan,
};

impl PipelinePartitionMetadata for LlamaPipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::llama(self.architecture.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::text_only_input_part(
            "llama",
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
        self.parameter_bank.as_ref()
    }
}

impl PipelineForward for LlamaPipelinePartition {
    pipeline_observed_forward!();
    fn forward(
        &mut self,
        input: PipelineStageInput<'_>,
        step: PipelineStep,
        mask: Option<&Array>,
        cache: &mut [PipelineLayerCache],
        stream: &Stream,
    ) -> Result<PipelineStageOutput, Error> {
        execute_neutral_decoder_partition_observed(
            self, input, step, mask, cache, None, stream, None,
        )
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
        if expert_group.is_some() {
            return Err(Error::Parallel(
                "Llama/Mistral pipeline stages do not contain routed experts".into(),
            ));
        }
        execute_neutral_decoder_partition_observed(
            self, input, step, mask, cache, execution, stream, observer,
        )
    }
}

pub(super) fn load_llama_pipeline(
    source_args: LlamaModelArgs,
    model_kind: ModelKind,
    store: SharedCheckpointSource,
    topology: MlxParallelPlan,
    wire_contract: eredu_runtime::PipelineWireContract,
    requested_quantization: Option<WeightQuantization>,
    dense_stream: Option<PipelineLayerLoadOptions>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<PipelineModel, Error> {
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::Llama], "Llama")?;
    let quantize_on_load = requested_quantization
        .map(|requested| {
            crate::backend::runtime::checkpoint::quantization::should_quantize_on_load(
                "Llama pipeline",
                source_args.quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let (store, target_args, materialization) = match quantize_on_load {
        Some(quantization) => {
            let (store, args, report) = crate::composition::llama::quantize_neutral_llama_store(
                store,
                &source_args,
                quantization,
                stream,
            )?;
            (store, args, Some(report))
        }
        None => (store, source_args.clone(), None),
    };
    let binding_adapter = crate::composition::llama::LlamaPipelineBindings::new();
    let seed_architecture = eredu_architectures::llama::LayeredModel::<MlxNeuralBackend>::new(
        target_args.clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let binding_parameter_description = seed_architecture
        .parameter_description(stream)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&seed_architecture)?;
    let target_units = architecture_group_unit_count(
        &binding_parameter_description,
        decoder_group,
        "Llama decoder",
    )?;
    topology.preflight(Some(target_units), None)?;
    let range = topology.layer_range(target_units)?;
    let (architecture, parallel_layout) = if topology.tensor_parallel_size() > 1 {
        let layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
        let geometry = eredu_architectures::llama::local_geometry(&target_args, &layout)
            .map_err(|error| Error::Parallel(error.to_string()))?;
        let architecture =
            eredu_architectures::llama::LayeredModel::<MlxNeuralBackend>::new_parallel(
                target_args.clone(),
                geometry,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        (architecture, Some(layout))
    } else {
        (seed_architecture, None)
    };
    let placement = Arc::new(decoder_architecture_transport::<_, MlxHybridState>(
        &architecture,
        topology.pipeline_parallel_size(),
    )?);
    let mut info = base_info(
        topology,
        wire_contract,
        range.clone(),
        placement,
        eredu_architectures::decoder::TEXT_DECODER_EXECUTION_GROUP,
        model_kind,
    );
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
    let layers = range
        .clone()
        .map(|global_layer| {
            architecture
                .construct_unit(global_layer, stream)
                .map(MlxModule::new)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut stage = LlamaPipelinePartition {
        architecture,
        partition,
        bindings: binding_adapter,
        layers,
        dense_layers: None,
        expert_realization: None,
        expert_assignment: None,
        parameter_bank: None,
        routing_statistics: RoutingStatistics::default(),
    };
    let static_roles = parameter_description.select_static_roles(&stage.partition);
    info.materialization = materialization;
    let static_units = select_static_binding_units_by_owner(
        stage.partition.parameter_bindings(),
        stage
            .bindings
            .static_units(&stage.architecture, store.as_ref())?,
        &static_roles,
    )?;
    let quantize_on_load = None;
    let mut loaded = PipelineLoadAccumulator::new("Llama", &stage.partition);
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&stage.architecture)?;
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
            let bindings = stage.bindings.cartesian_layer_bindings(
                &stage.architecture,
                global_layer,
                store.as_ref(),
                parallel_layout.as_ref(),
                stream,
            )?;
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
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
    let static_device_bytes = loaded.finish(&mut info)?;
    let checkpoint_diagnostics = store.source_diagnostics()?;
    let materialized_shards = checkpoint_diagnostics.payload_shard_paths.clone();
    if let Some(dense_stream) = dense_stream {
        let streamed_layout = parallel_layout.clone();
        let streamed_architecture = &stage.architecture;
        let dense_layers = build_pipeline_layer_storage(
            Arc::clone(&store),
            stage.partition.parameter_bindings(),
            &[],
            range.clone(),
            dense_stream,
            static_device_bytes,
            info.materialization.clone(),
            stream,
            weights_stream,
            |global_layer, stream| {
                streamed_architecture
                    .construct_unit(global_layer, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            },
            |global_layer, _layer, store| {
                stage.bindings.cartesian_layer_bindings(
                    streamed_architecture,
                    global_layer,
                    store,
                    streamed_layout.as_ref(),
                    stream,
                )
            },
            |global_layer| {
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    streamed_architecture,
                    decoder_group,
                    global_layer,
                )
            },
        )?;
        stage.dense_layers = Some(dense_layers);
        let layer_bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_device_bytes
            .checked_add(layer_bytes)
            .ok_or_else(|| {
                Error::Parallel("pipeline planned-owned byte total overflowed".into())
            })?;
    } else {
        info.planned_owned_parameter_bytes = static_device_bytes;
    }
    info.opened_checkpoint_shards = materialized_shards;
    info.checkpoint_diagnostics = Some(checkpoint_diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}
