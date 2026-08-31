use std::{ops::Range, sync::Arc};

use crate::backend::runtime::distributed::Group;
use eredu_architectures::ModelKind;
use eredu_checkpoint::{store::SharedCheckpointSource, WeightQuantization};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureParameters, ExpertCacheLoadOptions, ExpertPass,
};
use safemlx::{error::Exception, ops::indexing::TryIndexOp, Array, Stream};

use crate::{
    backend::{
        error::Error,
        nn::shared::{MlxModule, MlxNeuralBackend},
        runtime::{
            cache::state::MlxHybridState,
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
    composition::gemma4::{
        Gemma4Bindings, Gemma4PipelineUnit, PreparedParts as Gemma4PreparedParts,
    },
    composition::mlx::distributed::pipeline::{
        architecture_decoder_group, architecture_group_by_kind, architecture_group_id_by_kind,
        architecture_group_unit_count, architecture_parallel_layout,
        architecture_parameter_unit_owner, base_info, build_pipeline_expert_cache,
        build_pipeline_layer_storage, execute_routed_layered_partition_observed,
        load_architecture_static_parameters, media_architecture_transport, pipeline_binding_units,
        preflight_pipeline_realization, quantize_pipeline_stage_store,
        validate_admitted_pipeline_kind, validate_pipeline_expert_dispatch, BoundPipelineBindings,
        Gemma4IngressState, Gemma4PipelinePartition, MlxPlacedGroupExecutor,
        PipelineAuxiliaryState, PipelineExpertStorage, PipelineForward, PipelineLayerCache,
        PipelineLayerLoadOptions, PipelineLayerStorage, PipelineLoadAccumulator, PipelineModel,
        PipelinePartitionMetadata, PipelinePayload, PipelineStageInput, PipelineStageOutput,
        PipelineStep,
    },
};

impl Gemma4PipelinePartition {
    fn args(&self) -> &eredu_architectures::gemma4::FamilyConfig {
        self.architecture.args()
    }

    fn range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_runtime::ArchitectureGroupKind::Decoder)
    }

    fn vision_range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_runtime::ArchitectureGroupKind::VisionEncoder)
    }

    fn audio_range(&self) -> Range<usize> {
        self.media_range::<MlxHybridState>(eredu_runtime::ArchitectureGroupKind::AudioEncoder)
    }

    fn state_layout(&self) -> Result<eredu_runtime::StateLayout, Error> {
        self.partition
            .state()
            .map(|state| state.layout().clone())
            .ok_or_else(|| Error::Parallel("Gemma 4 partition has no runtime state".into()))
    }

    fn ingress_state_layout(&self) -> Result<eredu_runtime::StateLayout, Error> {
        eredu_runtime::ArchitectureParameters::state_layout(&self.architecture)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn static_modules(&self) -> &eredu_architectures::gemma4::StaticModules<MlxNeuralBackend> {
        <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::static_modules(&self.architecture)
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        stream: &Stream,
    ) -> Result<Gemma4PipelineUnit, Error> {
        <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::build_unit(&self.architecture, group, index, stream)
        .map(MlxModule::new)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn ingress_kind(&self, id: &str) -> Result<eredu_runtime::ArchitectureGroupKind, Error> {
        let graph = self.canonical_graph()?;
        let group = graph
            .groups()
            .iter()
            .position(|group| group.id() == id)
            .ok_or_else(|| Error::Parallel(format!("Gemma 4 has no placed group {id:?}")))?;
        Ok(self.group_kind(group))
    }

    fn canonical_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::execution_graph(&self.architecture)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn group_kind(&self, group: usize) -> eredu_runtime::ArchitectureGroupKind {
        <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
        >>::group_transport(&self.architecture, group)
        .kind
    }

    fn ingress_active(&self, group: &str, state: &Gemma4IngressState) -> Result<bool, Error> {
        match self.ingress_kind(group)? {
            eredu_runtime::ArchitectureGroupKind::VisionEncoder => {
                Ok(state.vision_hidden.is_some())
            }
            eredu_runtime::ArchitectureGroupKind::AudioEncoder => Ok(state.audio_hidden.is_some()),
            _ => Err(Error::Parallel(format!(
                "Gemma 4 has no placed media group {group:?}"
            ))),
        }
    }

    fn ingress_arrays(&self, group: &str, state: &Gemma4IngressState) -> Result<Vec<Array>, Error> {
        let hidden = match self.ingress_kind(group)? {
            eredu_runtime::ArchitectureGroupKind::VisionEncoder => state.vision_hidden.as_ref(),
            eredu_runtime::ArchitectureGroupKind::AudioEncoder => state.audio_hidden.as_ref(),
            _ => {
                return Err(Error::Parallel(format!(
                    "Gemma 4 has no placed media group {group:?}"
                )))
            }
        };
        Ok(hidden
            .map(|hidden| hidden.as_array().clone())
            .into_iter()
            .collect())
    }

    fn replace_ingress_arrays(
        &self,
        group: &str,
        state: &mut Gemma4IngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let slot = match self.ingress_kind(group)? {
            eredu_runtime::ArchitectureGroupKind::VisionEncoder => &mut state.vision_hidden,
            eredu_runtime::ArchitectureGroupKind::AudioEncoder => &mut state.audio_hidden,
            _ => {
                return Err(Error::Parallel(format!(
                    "Gemma 4 has no placed media group {group:?}"
                )))
            }
        };
        match (slot.is_some(), arrays.as_slice()) {
            (true, [hidden]) => {
                *slot = Some(crate::MlxTensor::from_array(hidden.clone()));
                Ok(())
            }
            (false, []) => Ok(()),
            (active, _) => Err(Error::Parallel(format!(
                "Gemma 4 {group} payload has {} arrays for active={active}",
                arrays.len()
            ))),
        }
    }

    fn merge_ingress_arrays(
        &self,
        state: &mut Gemma4IngressState,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let expected =
            usize::from(state.vision_hidden.is_some()) + usize::from(state.audio_hidden.is_some());
        if arrays.len() != expected {
            return Err(Error::Parallel(format!(
                "Gemma 4 media merger produced {} arrays, expected {expected}",
                arrays.len()
            )));
        }
        let mut arrays = arrays.into_iter();
        if state.vision_hidden.is_some() {
            state.vision_hidden = arrays.next().map(crate::MlxTensor::from_array);
        }
        if state.audio_hidden.is_some() {
            state.audio_hidden = arrays.next().map(crate::MlxTensor::from_array);
        }
        Ok(())
    }

    fn begin_ingress(
        &mut self,
        typed: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<Gemma4IngressState, Error> {
        crate::backend::runtime::media::input::validate(typed)?;
        let prepared = Gemma4PreparedParts::new(self.args(), typed, stream)?;
        let parts = prepared.decoder_parts();
        let mut state = MlxHybridState::device(
            if execution.is_some_and(ParallelExecutionContext::is_tensor_parallel) {
                self.state_layout()?
            } else {
                self.ingress_state_layout()?
            },
        )?;
        let input = eredu_architectures::gemma4::ModelInput {
            parts: &parts,
            vision: prepared.vision_input(),
            audio: prepared.audio_input(),
            per_layer_tokens: None,
            mask: None,
        };
        let mut forward = match execution.and_then(ParallelExecutionContext::group) {
            Some(group) => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::ParallelLayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::begin_forward_parallel(&mut self.architecture, input, &mut state, group, stream),
            None => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::begin_forward(&mut self.architecture, input, &mut state, stream),
        }
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let graph = self.canonical_graph()?;
        let media_group = |kind| {
            graph
                .groups()
                .iter()
                .enumerate()
                .find_map(|(index, _)| (self.group_kind(index) == kind).then_some(index))
                .ok_or_else(|| Error::Parallel(format!("Gemma 4 graph has no {kind:?} group")))
        };
        let vision_group = media_group(eredu_runtime::ArchitectureGroupKind::VisionEncoder)?;
        let audio_group = media_group(eredu_runtime::ArchitectureGroupKind::AudioEncoder)?;
        let mut begin_group = |group_index| {
            if !<eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::should_execute_group(&self.architecture, group_index, &forward.context)
            {
                return Ok::<Option<crate::MlxTensor>, Error>(None);
            }
            let hidden = match execution.and_then(ParallelExecutionContext::group) {
                Some(group) => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::ParallelLayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::begin_execution_group_parallel(
                    &mut self.architecture,
                    group_index,
                    &forward.hidden,
                    &[],
                    &mut state,
                    &mut forward.context,
                    group,
                    stream,
                ),
                None => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::begin_execution_group(
                    &mut self.architecture,
                    group_index,
                    &forward.hidden,
                    &[],
                    &mut state,
                    &mut forward.context,
                    stream,
                ),
            }
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            Ok(Some(hidden))
        };
        let vision_hidden = begin_group(vision_group)?;
        let audio_hidden = begin_group(audio_group)?;
        Ok(Gemma4IngressState {
            forward: Some(forward),
            state,
            vision_hidden,
            vision_state: None,
            audio_hidden,
            audio_valid: None,
        })
    }

    fn begin_ingress_continuation(
        &mut self,
        typed: crate::backend::runtime::media::input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Gemma4IngressState, Error> {
        crate::backend::runtime::media::input::validate(typed)?;
        let prepared = Gemma4PreparedParts::new(self.args(), typed, stream)?;
        let vision_hidden = prepared.vision_input().map(|input| input.patches.clone());
        let vision_state = prepared
            .vision_input()
            .map(|input| {
                self.static_modules()
                    .vision
                    .as_ref()
                    .ok_or_else(|| Error::ArchitectureModel("Gemma 4 has no vision tower".into()))?
                    .prepare_state(input, stream)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
            .transpose()?;
        let audio_hidden = prepared.audio_input().map(|input| input.features.clone());
        let audio_valid = prepared
            .audio_input()
            .map(|input| input.valid_subsampled_frames.to_vec());
        Ok(Gemma4IngressState {
            forward: None,
            state: MlxHybridState::device(self.ingress_state_layout()?)?,
            vision_hidden,
            vision_state,
            audio_hidden,
            audio_valid,
        })
    }

    fn forward_media_unit(
        &mut self,
        group: usize,
        index: usize,
        layer: &mut Gemma4PipelineUnit,
        state: &mut Gemma4IngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let kind = self.group_kind(group);
        let hidden = match kind {
            eredu_runtime::ArchitectureGroupKind::VisionEncoder => state.vision_hidden.as_ref(),
            eredu_runtime::ArchitectureGroupKind::AudioEncoder => state.audio_hidden.as_ref(),
            _ => None,
        }
        .ok_or_else(|| Error::Parallel("Gemma 4 media group has no activation".into()))?
        .clone();
        let output = if let Some(forward) = state.forward.as_mut() {
            match execution.and_then(ParallelExecutionContext::group) {
                Some(parallel) => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::ParallelLayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::forward_unit_parallel(
                    &mut self.architecture,
                    group,
                    index,
                    &mut **layer,
                    &hidden,
                    &mut state.state,
                    &mut forward.context,
                    parallel,
                    stream,
                ),
                None => <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                    MlxNeuralBackend,
                    MlxHybridState,
                >>::forward_unit(
                    &mut self.architecture,
                    group,
                    index,
                    &mut **layer,
                    &hidden,
                    &mut state.state,
                    &mut forward.context,
                    stream,
                ),
            }
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        } else {
            self.architecture
                .forward_partition_media_continuation(
                    &mut **layer,
                    &hidden,
                    state.vision_state.as_ref(),
                    state.audio_valid.as_deref(),
                    stream,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        };
        match kind {
            eredu_runtime::ArchitectureGroupKind::VisionEncoder => {
                state.vision_hidden = Some(output)
            }
            eredu_runtime::ArchitectureGroupKind::AudioEncoder => state.audio_hidden = Some(output),
            _ => unreachable!(),
        }
        Ok(())
    }

    fn finish_ingress(
        &mut self,
        mut state: Gemma4IngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<PipelinePayload, Error> {
        let forward = state.forward.take().ok_or_else(|| {
            Error::Parallel("Gemma 4 media finalization requires the primary ingress state".into())
        })?;
        let (hidden, mut per_layer_inputs) = self
            .architecture
            .finish_partition_media_ingress(
                forward,
                &mut state.state,
                state.vision_hidden.take(),
                state.audio_hidden.take(),
                execution.and_then(ParallelExecutionContext::group),
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        if let Some(inputs) = &per_layer_inputs {
            let range = self.partition.local_geometry().per_layer_range().clone();
            per_layer_inputs = Some(crate::MlxTensor::from_array(
                inputs
                    .as_array()
                    .try_index_device((.., .., .., range), stream)?,
            ));
        }
        Ok(PipelinePayload {
            hidden: hidden.into_array(),
            auxiliary: PipelineAuxiliaryState::new(
                self.partition
                    .boundary_schema()
                    .encode(eredu_architectures::gemma4::TextBoundary::new(
                        per_layer_inputs,
                    ))
                    .map_err(|error| Error::Parallel(error.to_string()))?
                    .into_iter()
                    .map(crate::MlxTensor::into_array)
                    .collect(),
            ),
        })
    }
}

impl PipelinePartitionMetadata for Gemma4PipelinePartition {
    fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        eredu_architectures::capability::gemma4(self.args())
    }

    fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        eredu_architectures::media_plan::gemma4_input_part(
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

impl MlxPlacedGroupExecutor for Gemma4PipelinePartition {
    fn begin_placed_ingress(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_ingress(input, execution, stream)?);
        Ok(())
    }

    fn begin_placed_ingress_continuation(
        &mut self,
        input: crate::backend::runtime::media::input::ModelInput<'_>,
        _execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        self.ingress_state = Some(self.begin_ingress_continuation(input, stream)?);
        Ok(())
    }

    fn placed_ingress_active(&self, group: &str) -> Result<bool, Error> {
        let state = self
            .ingress_state
            .as_ref()
            .ok_or_else(|| Error::Parallel("Gemma 4 placed ingress state is unavailable".into()))?;
        self.ingress_active(group, state)
    }

    fn placed_ingress_arrays(&self, group: &str) -> Result<Vec<Array>, Error> {
        let state = self
            .ingress_state
            .as_ref()
            .ok_or_else(|| Error::Parallel("Gemma 4 placed ingress state is unavailable".into()))?;
        self.ingress_arrays(group, state)
    }

    fn replace_placed_ingress_arrays(
        &mut self,
        group: &str,
        arrays: Vec<Array>,
    ) -> Result<(), Error> {
        let mut state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Gemma 4 placed ingress state is unavailable".into()))?;
        let result = self.replace_ingress_arrays(group, &mut state, arrays);
        self.ingress_state = Some(state);
        result
    }

    fn merge_placed_ingress_arrays(&mut self, arrays: Vec<Array>) -> Result<(), Error> {
        let mut state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Gemma 4 placed ingress state is unavailable".into()))?;
        let result = self.merge_ingress_arrays(&mut state, arrays);
        self.ingress_state = Some(state);
        result
    }

    fn execute_placed_ingress(
        &mut self,
        group: &str,
        _step: PipelineStep,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let mut state = self
            .ingress_state
            .take()
            .ok_or_else(|| Error::Parallel("Gemma 4 placed ingress state is unavailable".into()))?;
        let result = self.execute_placed_media(group, &mut state, execution, stream);
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
            .ok_or_else(|| Error::Parallel("Gemma 4 placed ingress state is unavailable".into()))?;
        self.finish_ingress(state, execution, stream)
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
        let mut state = self.begin_ingress(input, execution, stream)?;
        let graph = self.canonical_graph()?;
        let media = graph
            .groups()
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                matches!(
                    self.group_kind(*index),
                    eredu_runtime::ArchitectureGroupKind::VisionEncoder
                        | eredu_runtime::ArchitectureGroupKind::AudioEncoder
                )
            })
            .map(|(_, group)| group.id().to_owned())
            .collect::<Vec<_>>();
        for group in media {
            self.execute_placed_media(&group, &mut state, execution, stream)?;
        }
        let payload = self.finish_ingress(state, execution, stream)?;
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

impl PipelineForward for Gemma4PipelinePartition {
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

impl Gemma4PipelinePartition {
    fn new(
        architecture: eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend>,
        partition: eredu_runtime::ArchitecturePartition<
            Arc<eredu_architectures::gemma4::LocalGeometry>,
            eredu_architectures::gemma4::TextBoundarySchema,
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

    fn execute_placed_media(
        &mut self,
        group: &str,
        state: &mut Gemma4IngressState,
        execution: Option<&ParallelExecutionContext<'_>>,
        stream: &Stream,
    ) -> Result<(), Error> {
        let graph = self.canonical_graph()?;
        let group_index = graph
            .groups()
            .iter()
            .position(|candidate| candidate.id() == group)
            .ok_or_else(|| Error::Parallel(format!("Gemma 4 has no placed group {group:?}")))?;
        let kind = self.group_kind(group_index);
        let range = match kind {
            eredu_runtime::ArchitectureGroupKind::VisionEncoder => self.vision_range().clone(),
            eredu_runtime::ArchitectureGroupKind::AudioEncoder => self.audio_range().clone(),
            _ => return Ok(()),
        };
        let ordinal_start = self
            .partition
            .groups()
            .iter()
            .filter(|placed| placed.group_index() < group_index)
            .map(|placed| placed.global_units().len())
            .sum::<usize>();
        if !self.ingress_active(group, state)? {
            return Ok(());
        }
        if let Some(storage) = self.dense_layers.take() {
            let result = (|| {
                let ordinals = ordinal_start..ordinal_start + range.len();
                let mut window = storage.transfer_window(ordinals.clone(), true)?;
                for (ordinal, index) in ordinals.zip(range) {
                    let transfer = window
                        .as_mut()
                        .map(|window| window.next(stream))
                        .transpose()?;
                    let lease = transfer
                        .is_none()
                        .then(|| storage.prepare_layerwise_absolute(ordinal))
                        .transpose()?;
                    let mut layer = self.build_unit(group_index, index, stream)?;
                    populate_module_from_lease(
                        &mut layer,
                        transfer
                            .as_ref()
                            .map(|transfer| transfer.lease())
                            .or(lease.as_ref())
                            .expect("Gemma 4 placed media residency lease"),
                    )?;
                    self.forward_media_unit(
                        group_index,
                        index,
                        &mut layer,
                        state,
                        execution,
                        stream,
                    )?;
                    let outputs = self.ingress_arrays(group, state)?;
                    synchronize_outputs(outputs.iter())?;
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
            let mut resident = match kind {
                eredu_runtime::ArchitectureGroupKind::VisionEncoder => {
                    std::mem::take(&mut self.vision_layers)
                }
                eredu_runtime::ArchitectureGroupKind::AudioEncoder => {
                    std::mem::take(&mut self.audio_layers)
                }
                _ => unreachable!(),
            };
            let result = range.zip(&mut resident).try_for_each(|(index, layer)| {
                self.forward_media_unit(group_index, index, layer, state, execution, stream)
            });
            match kind {
                eredu_runtime::ArchitectureGroupKind::VisionEncoder => {
                    self.vision_layers = resident
                }
                eredu_runtime::ArchitectureGroupKind::AudioEncoder => self.audio_layers = resident,
                _ => unreachable!(),
            }
            result?;
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
                "Gemma 4 stage has {} cache entries for {} layers",
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
        let statistics = &mut self.routing_statistics;
        if let Some(expert_cache) = expert_cache {
            let assignment = assignment.as_ref().ok_or_else(|| {
                Error::Parallel("Gemma 4 external experts have no assignment".into())
            })?;
            let mut execute = |execution: GatedProductExpertExecution, stream: &Stream| {
                execute_pipeline_cached_neutral_gemma4(
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
fn execute_pipeline_cached_neutral_gemma4(
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
        crate::composition::mlx::distributed::expert::execute_cached_neutral_gemma4(
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

pub(super) fn load_neutral_gemma4_pipeline(
    source_args: eredu_architectures::gemma4::FamilyConfig,
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
    validate_admitted_pipeline_kind(model_kind, &[ModelKind::Gemma4], "Gemma 4")?;
    let external_experts = topology.expert_parallel_size > 1 || expert_cache_options.is_some();
    let binding_adapter = Gemma4Bindings::new(external_experts);
    let quantize_on_load = requested_quantization
        .map(|requested| {
            should_quantize_on_load(
                "Gemma 4 pipeline",
                source_args.text.weight_quantization,
                requested,
            )
            .map(|required| required.then_some(requested))
        })
        .transpose()?
        .flatten();
    let expert_quantization = quantize_on_load;
    let target_args = quantize_on_load.map_or_else(
        || Ok(source_args.clone()),
        |quantization| {
            eredu_architectures::gemma4::load_time_quantization(&source_args, quantization)
                .map_err(Error::ArchitectureModel)
        },
    )?;
    let target_binding_adapter = Gemma4Bindings::new(external_experts);
    let global_architecture = eredu_architectures::gemma4::LayeredModel::<MlxNeuralBackend>::new(
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
        "Gemma 4 decoder",
    )?;
    let planned_layout = architecture_parallel_layout(&binding_parameter_description, topology)?;
    let geometry = Arc::new(
        eredu_architectures::gemma4::local_geometry(&target_args, &planned_layout)
            .map_err(|error| Error::Parallel(error.to_string()))?,
    );
    let architecture = eredu_architectures::gemma4::LayeredModel::<MlxNeuralBackend>::new_parallel(
        target_args.clone(),
        (*geometry).clone(),
        stream,
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let expert_realization = eredu_architectures::gemma4::expert_realization_plan(
        &architecture,
        topology.rank_topology(),
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    preflight_pipeline_realization(
        topology,
        target_units,
        expert_realization.as_ref(),
        external_experts,
        "Gemma 4",
    )?;
    let ranges = target_args
        .text
        .pipeline_layer_ranges(topology.pipeline_parallel_size)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    if ranges.iter().map(Range::len).sum::<usize>() != target_units {
        return Err(Error::Parallel(
            "Gemma 4 architecture pipeline ranges disagree with its parameter description".into(),
        ));
    }
    let range = ranges
        .get(topology.pipeline_parallel_rank)
        .cloned()
        .ok_or_else(|| Error::Parallel("Gemma 4 pipeline rank has no layer range".into()))?;
    let decoder_group_id = architecture_group_id_by_kind::<_, MlxHybridState>(
        &architecture,
        eredu_runtime::ArchitectureGroupKind::Decoder,
    )?;
    let neutral_placement = Arc::new(
        media_architecture_transport::<_, MlxHybridState>(
            &architecture,
            topology.pipeline_parallel_size,
        )?
        .with_group_unit_ranges(&decoder_group_id, ranges.clone())?,
    );
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
    let mut stage = Gemma4PipelinePartition::new(architecture, partition)?;
    if external_experts {
        let assignment =
            ExpertAssignment::from_realization(expert_realization.as_ref().ok_or_else(|| {
                Error::Parallel(
                    "Gemma 4 external experts require an architecture realization".into(),
                )
            })?)?;
        info.global_expert_count = Some(assignment.global_expert_count());
        if expert_realization.as_ref().is_some_and(|realization| {
            realization
                .unit_specs()
                .keys()
                .any(|(group, unit)| stage.partition.owns_unit(group.as_str(), *unit))
        }) {
            info.local_expert_ids = assignment.local_global_expert_ids().to_vec();
        }
        stage.expert_assignment = Some(assignment);
        stage.expert_storage = PipelineExpertStorage::ExternalEmpty;
    }
    let parallel_layout = (topology.tensor_parallel_size > 1).then_some(planned_layout.clone());
    let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
        &stage.architecture,
        eredu_runtime::ArchitectureGroupKind::VisionEncoder,
    )?;
    let audio_group = architecture_group_by_kind::<_, MlxHybridState>(
        &stage.architecture,
        eredu_runtime::ArchitectureGroupKind::AudioEncoder,
    )?;
    let decoder_group = architecture_decoder_group::<_, MlxHybridState>(&stage.architecture)?;
    stage.vision_layers = stage
        .vision_range()
        .map(|index| stage.build_unit(vision_group, index, stream))
        .collect::<Result<Vec<_>, _>>()?;
    stage.audio_layers = stage
        .audio_range()
        .map(|index| stage.build_unit(audio_group, index, stream))
        .collect::<Result<Vec<_>, _>>()?;
    stage.layers = stage
        .range()
        .map(|index| stage.build_unit(decoder_group, index, stream))
        .collect::<Result<Vec<_>, _>>()?;

    let static_roles = parameter_description.select_static_roles(&stage.partition);
    let (store, materialization) = match quantize_on_load {
        Some(quantization) => {
            let selection = PipelineStageQuantizationSelection::new(
                &static_roles,
                decoder_group,
                stage.range().clone(),
            )
            .with_layer_group(vision_group, stage.vision_range().clone())
            .with_layer_group(audio_group, stage.audio_range().clone());
            let source_architecture =
                eredu_architectures::gemma4::LayeredModel::<MlxNeuralBackend>::new_parallel(
                    source_args.clone(),
                    (*geometry).clone(),
                    stream,
                )
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
                selection,
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
    let mut loaded = PipelineLoadAccumulator::new("Gemma 4", &stage.partition);
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
    let gemma4_resident_layers = dense_stream.is_none();
    if gemma4_resident_layers {
        let architecture = &stage.architecture;
        for (index, layer) in stage.vision_range().clone().zip(&mut stage.vision_layers) {
            let bindings = binding_adapter.layer_bindings(
                architecture,
                vision_group,
                index,
                layer,
                store.as_ref(),
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
        for (index, layer) in stage.audio_range().clone().zip(&mut stage.audio_layers) {
            let bindings = binding_adapter.layer_bindings(
                architecture,
                audio_group,
                index,
                layer,
                store.as_ref(),
            )?;
            loaded.load(
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    audio_group,
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
        let audio_start = stage.audio_range().start;
        let audio_count = stage.audio_range().len();
        let text_start = stage.range().start;
        let media_count = vision_count + audio_count;
        let unit_count = media_count + stage.range().len();
        let vision_group = architecture_group_by_kind::<_, MlxHybridState>(
            architecture,
            eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        )?;
        let audio_group = architecture_group_by_kind::<_, MlxHybridState>(
            architecture,
            eredu_runtime::ArchitectureGroupKind::AudioEncoder,
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
                    <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::build_unit(architecture, vision_group, vision_start + ordinal, stream)
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
                } else if ordinal < media_count {
                    <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::build_unit(
                        architecture,
                        audio_group,
                        audio_start + ordinal - vision_count,
                        stream,
                    )
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
                } else {
                    <eredu_architectures::gemma4::LayeredModel<MlxNeuralBackend> as eredu_runtime::LayeredArchitecture<
                        MlxNeuralBackend,
                        MlxHybridState,
                    >>::build_unit(
                        architecture,
                        decoder_group,
                        text_start + ordinal - media_count,
                        stream,
                    )
                    .map(MlxModule::new)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
                }
            },
            |ordinal, layer, store| {
                if ordinal < vision_count {
                    binding_adapter.layer_bindings(
                        architecture,
                        vision_group,
                        vision_start + ordinal,
                        layer,
                        store,
                    )
                } else if ordinal < media_count {
                    binding_adapter.layer_bindings(
                        architecture,
                        audio_group,
                        audio_start + ordinal - vision_count,
                        layer,
                        store,
                    )
                } else {
                    binding_adapter.cartesian_layer_bindings(
                        architecture,
                        decoder_group,
                        text_start + ordinal - media_count,
                        layer,
                        store,
                        layout.as_ref(),
                    )
                }
            },
            |ordinal| {
                let (group, index) = if ordinal < vision_count {
                    (vision_group, vision_start + ordinal)
                } else if ordinal < media_count {
                    (audio_group, audio_start + ordinal - vision_count)
                } else {
                    (decoder_group, text_start + ordinal - media_count)
                };
                architecture_parameter_unit_owner::<_, MlxHybridState>(
                    architecture,
                    group,
                    index,
                )
            },
        )?
        .with_execution_offset(media_count)?;
        stage.dense_layers = Some(storage);
        let bytes = stage.dense_layers.as_ref().unwrap().planned_layer_bytes()?;
        info.planned_owned_parameter_bytes = static_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::Parallel("Gemma 4 pipeline bytes overflowed".into()))?;
    } else {
        info.planned_owned_parameter_bytes = static_bytes;
    }
    if external_experts {
        let assignment = stage
            .expert_assignment
            .as_ref()
            .expect("Gemma 4 expert assignment");
        let catalog = eredu_architectures::gemma4::expert_residency_catalog(
            store.as_ref(),
            &source_args.text,
        )
        .map_err(Error::ArchitectureModel)?;
        let units = crate::composition::select_architecture_expert_units(
            catalog,
            |group, unit| stage.partition.owns_unit(group.as_str(), unit),
            |identity| assignment.owner(identity.global_expert) == Some(assignment.rank()),
        );
        let entries = crate::composition::architecture_expert_units(units, store.as_ref(), None)?;
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
            .ok_or_else(|| Error::Parallel("Gemma 4 expert bytes overflowed".into()))?;
        stage.expert_storage = PipelineExpertStorage::External(Box::new(cache));
    }
    let diagnostics = store.source_diagnostics()?;
    info.opened_checkpoint_shards = diagnostics.payload_shard_paths.clone();
    info.checkpoint_diagnostics = Some(diagnostics);
    PipelineModel::from_adapter(topology, info, stage)
}
