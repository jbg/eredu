//! One neutral Muse multimodal model for resident and bounded runtimes.

mod observation;
mod parameters;
pub use parameters::component_definition;
mod prefill;
mod waveform;
use prefill::DecoderPrefill;

use crate::decoder::DecoderStaticResources;
use eredu_runtime::{ComponentEquations as _, ParallelComponentKernel as _};

use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, GroupedNeuralBackend,
    Parameterized, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionUnitLayout, ExpertPass, LayerRuntimeState,
    LayeredForwardState, LayeredPartitionInput, LayeredPartitionOutput, OwnedParameterGroupSpec,
    ParameterGroupOwner, ParameterProvider, RuntimeStateComponents, StateLayout,
};

use crate::{
    composite_execution::{
        CompositeArchitecture, ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
        ExternalPredictionTargetOperation, PreparedCompositeInput,
    },
    media_plan::MuseInputPartPlan,
};

use super::{
    layer_parameter_groups, state_layout, static_parameter_groups, vision_layer_parameter_groups,
    DecoderConfig, LocalGeometry, StaticModules as TextStaticModules, TransformerBlock,
    VisionBlock, VisionInput, VisionState, VisionStatic,
};

/// Prepared encoder products survive bounded decoder-prefill transactions.
pub const RETAINED_PREFILL: bool = true;

/// Exact decoder and waveform state ownership used by cold and live partitions.
pub(crate) fn state_partition_plan(
    args: &DecoderConfig,
    layout: &StateLayout,
) -> eredu_runtime::ArchitectureStatePartitionPlan {
    if args.audio_config.is_some() {
        let decoder = args.num_hidden_layers as usize;
        eredu_runtime::ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::group_units(1, 0..decoder),
            eredu_runtime::ArchitectureStatePartitionRule::group_units(2, decoder..layout.len()),
        ])
        .with_packing(eredu_runtime::StatePartitionPacking::PreserveGlobalOrdinals)
    } else {
        crate::transport::pipeline_state(1, layout)
    }
}

/// Stable execution-group identity for Muse-Glimmer vision ingress.
pub const VISION_EXECUTION_GROUP: &str = "vision_encoder";
/// Stable execution-group identity for Muse-Glimmer text decoding.
pub const TEXT_EXECUTION_GROUP: &str = "text_decoder";

/// Shared cold and loaded transport ownership for an optional Muse vision root.
pub(crate) fn vision_group_transport(
    args: &DecoderConfig,
) -> eredu_runtime::ArchitectureGroupTransport {
    eredu_runtime::ArchitectureGroupTransport {
        placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
        kind: eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        first_owner_static_roles: args
            .vision_config
            .as_ref()
            .map_or_else(Vec::new, |_| vec!["vision".into()]),
        last_owner_static_roles: Vec::new(),
        merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
        parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
        request_optional: true,
    }
}

/// Shared cold and loaded ownership for small replicated waveform units.
pub(crate) fn audio_group_transport() -> eredu_runtime::ArchitectureGroupTransport {
    eredu_runtime::ArchitectureGroupTransport {
        placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
        kind: eredu_runtime::ArchitectureGroupKind::AudioEncoder,
        first_owner_static_roles: vec![],
        last_owner_static_roles: vec![],
        merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
        parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
        request_optional: true,
    }
}

/// Exact flattened patch or projected-media wire geometry for a vision edge.
pub(crate) fn vision_partition_boundary_schema(
    args: &DecoderConfig,
    continuation: bool,
) -> Result<eredu_runtime::BoundaryWireSchema, Error> {
    use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
    let vision = args
        .vision_config
        .as_ref()
        .ok_or_else(|| Error::backend("Muse vision boundary has no vision configuration"))?;
    eredu_runtime::BoundaryWireSchema::new(
        if continuation {
            "muse.vision_continuation"
        } else {
            "muse.vision_to_decoder"
        },
        eredu_runtime::BoundaryTensorSpec::new(
            "hidden",
            [
                Dim::Sequence,
                Dim::Fixed(if continuation {
                    vision.hidden_size
                } else {
                    args.hidden_size
                }),
            ],
            Dtype::Activation,
        ),
        [],
    )
    .map_err(Error::backend)
}

/// Proves one DFlash assistant against a target and returns exact ordered capture paths.
pub fn external_assistant_capture_request(
    target: &DecoderConfig,
    assistant: &super::assistant::DFlashConfig,
) -> Result<ExternalPredictionCaptureRequest, String> {
    let proof = assistant
        .prove_compatibility(target)
        .map_err(|error| error.to_string())?;
    let target_layers = proof.target_layer_ids().to_vec().into_boxed_slice();
    let target_paths = target_layers
        .iter()
        .map(|index| format!("model.layers.{index}.output"))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    Ok(ExternalPredictionCaptureRequest::Muse {
        target_layers,
        target_paths,
    })
}

/// One ordered decoder-ingress segment.
pub enum DecoderInputPart<'a, T> {
    /// Ordinary text token IDs.
    Text(&'a T),
    /// Image/video placeholder token IDs matching the next projected media span.
    Media(&'a T),
    /// Waveform placeholders matching the next projected audio span.
    Audio(&'a T),
}

/// Prepared ordered text/media request.
pub struct ModelInput<'a, T> {
    /// Ordered token segments at decoder ingress.
    pub parts: &'a [DecoderInputPart<'a, T>],
    /// Optional packed raw image/video patches and host grid metadata.
    pub vision: Option<VisionInput<'a, T>>,
    /// Raw waveform with explicit clip or continuing-chunk boundaries.
    pub audio: Option<super::audio::execution::AudioInput<'a, T>>,
    /// Optional explicit decoder attention mask.
    pub mask: Option<&'a T>,
}

/// Architecture-prepared Muse-Glimmer decoder and vision ingress.
#[derive(Clone, Copy)]
enum IngressKind {
    Text,
    Vision,
    Audio,
}

/// Retained ordered decoder spans and separately packed vision and waveform inputs.
pub struct PreparedCompositeIngress<T> {
    tokens: Vec<T>,
    media: Vec<IngressKind>,
    pixels: Option<T>,
    grid: Vec<(i32, i32, i32)>,
    waveform: Option<T>,
    audio_segments: Vec<super::audio::execution::AudioSegment>,
}

impl<T> PreparedCompositeIngress<T> {
    /// Borrows ordered decoder segments with architecture-created placeholders.
    pub fn decoder_parts(&self) -> Vec<DecoderInputPart<'_, T>> {
        self.tokens
            .iter()
            .zip(&self.media)
            .map(|(tokens, media)| match media {
                IngressKind::Text => DecoderInputPart::Text(tokens),
                IngressKind::Vision => DecoderInputPart::Media(tokens),
                IngressKind::Audio => DecoderInputPart::Audio(tokens),
            })
            .collect()
    }

    /// Borrows independently padded audio clips with explicit convolution resets.
    pub fn audio_input(&self) -> Option<super::audio::execution::AudioInput<'_, T>> {
        self.waveform
            .as_ref()
            .map(|waveform| super::audio::execution::AudioInput::Segments {
                waveform,
                segments: &self.audio_segments,
            })
    }

    /// Borrows the packed vision input when media is present.
    pub fn vision_input(&self) -> Option<VisionInput<'_, T>> {
        self.pixels.as_ref().map(|pixels| VisionInput {
            pixels,
            grid: &self.grid,
        })
    }
}

/// Interprets one admitted Muse-Glimmer input using neutral tensor operations.
pub fn prepare_composite_ingress<B>(
    input: PreparedCompositeInput<'_, B::Tensor, MuseInputPartPlan>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedCompositeIngress<B::Tensor>, Error>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    let prepared = input.prepared();
    let admitted = input.admitted();
    if prepared.identity() != admitted.identity() || prepared.len() != admitted.parts().len() {
        return Err(Error::backend(
            "Muse-Glimmer prepared input no longer matches its admission",
        ));
    }

    let mut tokens = Vec::with_capacity(prepared.len());
    let mut media = Vec::with_capacity(prepared.len());
    let mut pixels = Vec::new();
    let mut grid = Vec::new();
    let mut waveforms = Vec::new();
    let mut audio_segments = Vec::new();
    for (part, plan) in prepared.parts().iter().zip(admitted.parts()) {
        match plan {
            MuseInputPartPlan::TextTokens { .. } => {
                let eredu_runtime::PreparedInputPayload::TokenIds(value) = part.payload() else {
                    return Err(Error::backend(
                        "Muse-Glimmer admitted text part lost its token payload",
                    ));
                };
                tokens.push(value.clone());
                media.push(IngressKind::Text);
            }
            MuseInputPartPlan::Audio { ingress, .. } => {
                let eredu_runtime::PreparedInputPayload::Tensor(value) = part.payload() else {
                    return Err(Error::backend(
                        "Muse admitted waveform lost its tensor payload",
                    ));
                };
                let count = i32::try_from(ingress.placeholder_count).map_err(Error::backend)?;
                let token = i32::try_from(ingress.placeholder_token_id).map_err(Error::backend)?;
                tokens.push(B::Tensor::full_i32(token, &[1, count], context)?);
                media.push(IngressKind::Audio);
                waveforms.push(value.clone());
                audio_segments.push(super::audio::execution::AudioSegment {
                    samples: i32::try_from(ingress.padded_samples).map_err(Error::backend)?,
                    carry: super::audio::execution::AudioCarry::NewClip,
                });
            }
            MuseInputPartPlan::Vision { ingress, .. } => {
                let eredu_runtime::PreparedInputPayload::Tensor(value) = part.payload() else {
                    return Err(Error::backend(
                        "Muse-Glimmer admitted media part lost its tensor payload",
                    ));
                };
                let count = i32::try_from(ingress.placeholder_count)
                    .map_err(|_| Error::backend("Muse-Glimmer media span exceeds I32"))?;
                let token = i32::try_from(ingress.placeholder_token_id)
                    .map_err(|_| Error::backend("Muse-Glimmer placeholder ID exceeds I32"))?;
                tokens.push(B::Tensor::full_i32(token, &[1, count], context)?);
                media.push(IngressKind::Vision);
                pixels.push(value.clone());
                grid.extend_from_slice(&ingress.patch_grid);
            }
        }
    }
    let pixels = match pixels.len() {
        0 => None,
        1 => pixels.pop(),
        _ => Some(B::Tensor::concatenate(&pixels, 0, context)?),
    };
    let waveform = match waveforms.len() {
        0 => None,
        1 => waveforms.pop(),
        _ => Some(B::Tensor::concatenate(&waveforms, 1, context)?),
    };
    Ok(PreparedCompositeIngress {
        tokens,
        media,
        pixels,
        grid,
        waveform,
        audio_segments,
    })
}

impl<B, S> crate::composite_execution::ParallelCompositeArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
}

impl<B, S> CompositeArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn supports_retained_prefill() -> bool {
        RETAINED_PREFILL
    }

    fn prefill_decoder_continuation(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        forward
            .prefill
            .as_ref()
            .and_then(|prefill| prefill.assembled.as_ref())
    }

    fn prepared_input<'a>(
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    ) -> Self::Input<'a> {
        crate::composite_execution::CompositeModelInput::Prepared(input)
    }
    fn prepared_view<'a>(
        input: &Self::Input<'a>,
    ) -> Option<
        crate::composite_execution::PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    > {
        match input {
            crate::composite_execution::CompositeModelInput::Prepared(input) => Some(*input),
            _ => None,
        }
    }

    type InputPartPlan = MuseInputPartPlan;
    type PrefillRequest =
        crate::composite_execution::DecoderPrefillRequest<B::Tensor, Self::InputPartPlan>;

    fn prepare_prefill_request(
        prepared: eredu_runtime::PreparedModelInput<B::Tensor>,
        admitted: crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::PrefillRequest, Self::Error> {
        crate::composite_execution::DecoderPrefillRequest::new(prepared, admitted)
    }
    type AdmissionConfig = DecoderConfig;

    fn admission_config(&self) -> Self::AdmissionConfig {
        let model = self.kernel();
        model.args.clone()
    }

    fn external_assistant_target_profile(
        config: &Self::AdmissionConfig,
    ) -> Option<crate::external_assistant::ExternalAssistantTargetProfile> {
        Some(crate::external_assistant::ExternalAssistantTargetProfile::Muse(config.clone()))
    }

    fn admit_prepared_input(
        config: &Self::AdmissionConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
    ) -> Result<
        crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>,
        eredu_core::CapabilityError,
    > {
        crate::media_plan::admit_muse_input(config, input, inspector)
    }

    fn should_execute_prepared_group(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> bool {
        if matches!(group, 0 | 2)
            && input
                .decoder_chunk()
                .is_some_and(|chunk| chunk.range.start > 0)
        {
            return false;
        }
        match group {
            2 => input
                .admitted()
                .parts()
                .iter()
                .any(|part| matches!(part, MuseInputPartPlan::Audio { .. })),
            0 => input
                .admitted()
                .parts()
                .iter()
                .any(|part| matches!(part, MuseInputPartPlan::Vision { .. })),
            _ => true,
        }
    }

    fn prepared_group_boundary_sequence(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<i32, String> {
        let positions = if group == 2 {
            input
                .admitted()
                .parts()
                .iter()
                .filter_map(|part| match part {
                    MuseInputPartPlan::Audio { ingress, .. } => Some(ingress.placeholder_count),
                    _ => None,
                })
                .try_fold(0u64, |sum, positions| sum.checked_add(positions))
                .ok_or_else(|| "Muse audio positions overflowed".to_owned())?
        } else if group == 0 {
            input
                .admitted()
                .parts()
                .iter()
                .filter_map(|part| match part {
                    MuseInputPartPlan::Vision { ingress, .. } => Some(ingress.placeholder_count),
                    _ => None,
                })
                .try_fold(0_u64, |total, positions| total.checked_add(positions))
                .ok_or_else(|| "Muse projected media positions overflowed".to_owned())?
        } else {
            input
                .decoder_chunk()
                .map_or(input.admitted().decoder_positions(), |chunk| {
                    chunk.range.len() as u64
                })
        };
        i32::try_from(positions)
            .map_err(|_| "Muse prepared boundary sequence exceeds i32".to_owned())
    }

    fn prepared_group_continuation_geometry(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<Option<crate::composite_execution::CompositeContinuationGeometry>, String> {
        let model = self.kernel();
        if group == 2 {
            let samples = input
                .admitted()
                .parts()
                .iter()
                .filter_map(|part| match part {
                    MuseInputPartPlan::Audio { ingress, .. } => Some(ingress.padded_samples),
                    _ => None,
                })
                .try_fold(0u64, |sum, samples| sum.checked_add(samples))
                .ok_or_else(|| "Muse waveform sample count overflowed".to_owned())?;
            if samples == 0 {
                return Ok(None);
            }
            let config = model
                .args
                .audio_config
                .as_ref()
                .ok_or("Muse audio continuation has no admitted tower")?;
            return super::audio::execution::AudioKernel::<B>::new(config.clone())
                .and_then(|audio| {
                    audio
                        .kernel()
                        .continuation_geometry(i32::try_from(samples).map_err(Error::backend)?)
                })
                .map(Some)
                .map_err(|error| error.to_string());
        }
        if group != 0 {
            return Ok(None);
        }
        let patches = input
            .admitted()
            .parts()
            .iter()
            .filter_map(|part| match part {
                MuseInputPartPlan::Vision { ingress, .. } => Some(&ingress.patch_grid),
                _ => None,
            })
            .flatten()
            .try_fold(0_u64, |total, &(time, height, width)| {
                [time, height, width]
                    .into_iter()
                    .try_fold(1_u64, |product, value| {
                        u64::try_from(value)
                            .ok()
                            .and_then(|value| product.checked_mul(value))
                    })
                    .and_then(|patches| total.checked_add(patches))
                    .ok_or_else(|| "Muse continuation patch geometry overflowed".to_owned())
            })?;
        if patches == 0 {
            return Ok(None);
        }
        let patches = i32::try_from(patches)
            .map_err(|_| "Muse continuation patch count exceeds i32".to_owned())?;
        let width = model
            .args
            .vision_config
            .as_ref()
            .ok_or_else(|| "Muse continuation has no vision configuration".to_owned())?
            .hidden_size;
        Ok(Some((patches, width).into()))
    }

    fn prepared_group_continuation_batched(&self, _group: usize) -> bool {
        false
    }

    fn partition_boundary_schema(
        &self,
        source_group: usize,
        destination_group: usize,
        _selected: &eredu_runtime::ResolvedBoundaryWireSchema,
        batch: i32,
        source_sequence: i32,
        _group_sequences: &[i32],
        continuation: Option<(i32, i32)>,
    ) -> Result<Option<eredu_runtime::ResolvedBoundaryWireSchema>, Error> {
        let model = self.kernel();
        if source_group == 2 && matches!(destination_group, 1 | 2) {
            use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
            let (sequence, width) =
                continuation.unwrap_or((source_sequence, model.args.hidden_size));
            return eredu_runtime::BoundaryWireSchema::new(
                if destination_group == 2 {
                    "muse.audio_continuation"
                } else {
                    "muse.audio_to_decoder"
                },
                eredu_runtime::BoundaryTensorSpec::new(
                    "hidden",
                    [Dim::Batch, Dim::Sequence, Dim::Fixed(width)],
                    Dtype::Activation,
                ),
                [],
            )
            .and_then(|schema| schema.resolve(batch, sequence))
            .map(Some)
            .map_err(Error::backend);
        }
        if source_group != 0 || !matches!(destination_group, 0 | 1) {
            return Ok(None);
        }
        let sequence = continuation.map_or(source_sequence, |(sequence, _)| sequence);
        vision_partition_boundary_schema(&model.args, source_group == destination_group)?
            .resolve(batch, sequence)
            .map(Some)
            .map_err(Error::backend)
    }

    fn accept_partition_boundary(
        &mut self,
        source_group: usize,
        destination_group: usize,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        values: Vec<B::Tensor>,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error> {
        if !matches!(
            (source_group, destination_group),
            (0, 0) | (0, 1) | (1, 1) | (2, 2) | (2, 1)
        ) {
            return Ok(None);
        }
        if values.len() != 1 || !schema.auxiliary().is_empty() {
            return Err(Error::backend(
                "Muse boundary must contain exactly its primary activation",
            ));
        }
        // A rank that already executed vision retains its admitted request
        // context. Resume the incoming decoder value without reassembling it.
        Ok(values.into_iter().next())
    }

    fn prepared_group_collective_waves(
        &self,
        group: usize,
        _input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        tensor_partitions: usize,
        stage_units: &[std::ops::Range<usize>],
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, String>
    {
        let pipeline_stages = stage_units.len();
        // Muse-Glimmer's vision blocks and projector are replicated equations.
        // Under TP+PP they still execute on every tensor rank, but emit no tensor
        // collective; the explicit empty waves keep that fact architecture-owned.
        Ok(
            (matches!(group, 0 | 2) && tensor_partitions > 1 && pipeline_stages > 1)
                .then(|| vec![Vec::new(); pipeline_stages]),
        )
    }

    fn routed_tensor_reductions(
        &self,
        _unit: usize,
        routed: bool,
    ) -> Result<eredu_runtime::RoutedTensorReductions, Self::Error> {
        let model = self.kernel();
        Ok(super::parallel::routed_tensor_reductions(
            &model.args,
            routed,
        ))
    }

    fn prepared_primary_ingress_collectives(
        &self,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        tensor_partitions: usize,
    ) -> Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>, String> {
        let model = self.kernel();
        if input
            .decoder_chunk()
            .is_some_and(|chunk| chunk.range.start > 0)
        {
            return Ok((tensor_partitions > 1).then(Vec::new));
        }
        crate::composite_execution::segmented_token_ingress_collectives(
            input
                .admitted()
                .parts()
                .iter()
                .filter_map(|part| match part {
                    MuseInputPartPlan::TextTokens { positions } => Some(*positions),
                    MuseInputPartPlan::Vision { .. } | MuseInputPartPlan::Audio { .. } => None,
                }),
            model.args.hidden_size,
            tensor_partitions,
        )
    }

    fn primary_ingress_collectives_pending(&self, forward: &Self::ForwardContext) -> bool {
        forward
            .parts
            .iter()
            .any(|part| matches!(part, PreparedPart::PendingText { .. }))
    }

    fn external_prediction_capture_paths(
        request: &ExternalPredictionCaptureRequest,
    ) -> Result<Option<Vec<String>>, Self::Error> {
        let ExternalPredictionCaptureRequest::Muse {
            target_layers,
            target_paths,
        } = request
        else {
            return Ok(None);
        };
        if target_layers.is_empty() {
            return Err(Error::backend(
                "Muse assistant capture has no target layers",
            ));
        }
        if target_paths.len() != target_layers.len() {
            return Err(Error::backend(
                "Muse assistant capture path count differs from target layers",
            ));
        }
        Ok(Some(target_paths.to_vec()))
    }

    fn external_prediction_capture(
        request: &ExternalPredictionCaptureRequest,
        _forward: &Self::ForwardContext,
        observed: Vec<B::Tensor>,
    ) -> Result<Option<ExternalPredictionTargetCapture<B::Tensor>>, Self::Error> {
        let ExternalPredictionCaptureRequest::Muse { target_layers, .. } = request else {
            return Ok(None);
        };
        if observed.len() != target_layers.len() {
            return Err(Error::backend(format!(
                "Muse assistant capture expected {} target states, received {}",
                target_layers.len(),
                observed.len()
            )));
        }
        Ok(Some(ExternalPredictionTargetCapture::Muse {
            target_states: observed,
        }))
    }

    fn external_prediction_target_contribution(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        let model = self.kernel_mut();
        let text = &mut model.static_modules.text;
        match operation {
            ExternalPredictionTargetOperation::TokenEmbeddings(tokens)
            | ExternalPredictionTargetOperation::RawTokenEmbeddings(tokens) => {
                text.raw_embeddings(tokens, parallel, context).map(Some)
            }
            ExternalPredictionTargetOperation::ProjectLogits(hidden) => {
                let hidden = text.final_hidden(hidden, context)?;
                text.raw_vocabulary_projection(&hidden, parallel, context)
                    .map(Some)
            }
            ExternalPredictionTargetOperation::RawVocabularyProjection(hidden) => text
                .raw_vocabulary_projection(hidden, parallel, context)
                .map(Some),
        }
    }

    fn finish_external_prediction_target_operation(
        &self,
        operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        combined: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        let model = self.kernel();
        let text = &model.static_modules.text;
        match operation {
            ExternalPredictionTargetOperation::TokenEmbeddings(_) => {
                text.normalize_embeddings(&combined, context).map(Some)
            }
            ExternalPredictionTargetOperation::ProjectLogits(_) => {
                text.finish_logits(combined, context).map(Some)
            }
            ExternalPredictionTargetOperation::RawTokenEmbeddings(_)
            | ExternalPredictionTargetOperation::RawVocabularyProjection(_) => Ok(Some(combined)),
        }
    }
}

/// Typed decoder input for one pipeline partition.
pub enum TextPartitionInput<'a, T> {
    /// Token identities owned by the first decoder partition.
    Tokens(&'a T),
    /// Embedded activation received from an upstream decoder partition.
    Hidden(T),
}

enum PreparedPart<T> {
    Text { tokens: T, embeddings: T },
    PendingText { tokens: T },
    Media { tokens: T },
    Audio { tokens: T },
}

/// Forward-pass values retained across streamed unit submissions.
pub struct ForwardContext<T> {
    mask: Option<T>,
    parts: Vec<PreparedPart<T>>,
    vision: Option<VisionState<T>>,
    vision_active: bool,
    audio: Option<super::audio::execution::AudioForward>,
    waveform: Option<T>,
    decoder_positions: i32,
    prefill: Option<DecoderPrefill<T>>,
}

/// Pinned text and media modules shared by every storage policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Text embedding, final norm, and output head.
    pub text: TextStaticModules<B>,
    /// Optional patch/position modules, merge adapter, and language projection.
    pub vision: Option<VisionStatic<B>>,
}

/// A streamable native-vision block or decoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Vision encoder block.
    Vision(VisionBlock<B>),
    /// Text decoder block.
    Text(TransformerBlock<B>),
    /// One patch, causal block or final audio projection.
    Audio(super::AudioUnit<B>),
}

impl<B, S> eredu_runtime::RoutedComponentKernel<B, S> for MuseKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn routed_unit_observations(&self) -> bool {
        true
    }
    fn routed_sparse_observations(&self) -> bool {
        true
    }

    fn routing_points(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
    ) -> Result<Option<eredu_runtime::RoutedObservationPoints>, Error> {
        let (group, index) = (address.group(), address.index());
        let path = &self.definition.unit(group, index)?.path;
        if group != 1 || !self.args.is_moe() {
            return Ok(None);
        }
        self.local_state_ordinal(index)?;
        Ok(Some(eredu_runtime::RoutedObservationPoints::new(
            eredu_runtime::RoutedBankId::new(0),
            format!("{path}.routing"),
            self.args.num_experts,
        )))
    }

    fn execute_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_text_observed(
            address, unit, hidden, state, forward, pass, provider, context, observer,
        )
    }

    fn execute_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let (group, index) = (address.group(), address.index());
        match (group, unit) {
            (1, Unit::Text(unit)) => self.forward_text_unit_with_provider(
                index, unit, hidden, state, forward, pass, provider, context,
            ),
            (_, unit) => <Self as eredu_runtime::ComponentEquations<B, S>>::execute(
                self, address, unit, hidden, state, forward, context,
            ),
        }
    }
}

impl<B, S> eredu_runtime::ParallelRoutedComponentKernel<B, S> for MuseKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        true
    }
    fn parallel_routed_sparse_observations(&self) -> bool {
        true
    }

    fn execute_parallel_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_text_parallel_observed(
            address, unit, hidden, state, forward, pass, provider, parallel, context, observer,
        )
    }

    fn execute_parallel_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let (group, index) = (address.group(), address.index());
        match (group, unit) {
            (1, Unit::Text(unit)) => self.forward_text_unit_parallel_with_provider(
                index, unit, hidden, state, forward, pass, provider, parallel, context,
            ),
            (_, unit) => <Self as eredu_runtime::ParallelComponentKernel<B, S>>::execute_parallel(
                self, address, unit, hidden, state, forward, parallel, context,
            ),
        }
    }
}

/// Shared execution contracts over Muse's typed equations and exact pinned resources.
pub type LayeredModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    MuseKernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// Muse equations, selected geometry, and native-independent resource construction.
pub struct MuseKernel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    args: DecoderConfig,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    static_modules: StaticModules<B>,
    parallel_geometry: Option<std::sync::Arc<LocalGeometry>>,
    selection: Option<eredu_runtime::ComponentSelection>,
    owns_input: bool,
    expert_realization:
        Option<std::sync::Arc<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>>,
}

impl<B> eredu_runtime::ComponentKernel<B> for MuseKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    type Input<'a> = crate::composite_execution::CompositeModelInput<
        'a,
        B::Tensor,
        MuseInputPartPlan,
        ModelInput<'a, B::Tensor>,
    >;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::Error> {
        if state
            .selected_layout(&self.state_layout_impl()?)
            .map_err(Error::backend)?
            != *state.layout()
        {
            return Err(Error::backend(
                "Muse state identity changes selected component geometry",
            ));
        }
        super::state_identity(
            &self.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
        .map_err(Error::backend_source)
    }
    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        crate::static_parameters::module_recipes(
            &self.static_modules,
            super::static_safetensors_recipes(&self.args, source)?,
        )
    }
    fn build_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        let (group, index) = (address.group(), address.index());
        match group {
            2 => {
                let audio = self
                    .args
                    .audio_config
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse has no audio tower"))?;
                let kind = audio
                    .execution_units()?
                    .get(index)
                    .copied()
                    .ok_or_else(|| Error::backend("Muse audio unit exceeds its group"))?;
                Ok(Unit::Audio(super::AudioUnit::new(audio, kind, context)?))
            }
            0 => Ok(Unit::Vision(VisionBlock::new(
                self.args
                    .vision_config
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?,
                index,
                context,
            )?)),
            1 => {
                let args = self
                    .parallel_geometry
                    .as_ref()
                    .map(|geometry| {
                        geometry.text_block(index).ok_or_else(|| {
                            Error::backend(format!(
                                "Muse-Glimmer text unit {index} has no rank-local geometry"
                            ))
                        })
                    })
                    .transpose()?
                    .unwrap_or(&self.args);
                let routed_spec = self
                    .expert_realization
                    .as_ref()
                    .and_then(|plan| plan.unit_spec(TEXT_EXECUTION_GROUP, index))
                    .cloned();
                if self.expert_realization.is_some() && routed_spec.is_none() {
                    return Err(Error::backend(format!(
                        "Muse-Glimmer expert realization omits text unit {index}"
                    )));
                }
                Ok(Unit::Text(TransformerBlock::new_with_routed_spec(
                    args,
                    index,
                    routed_spec,
                    context,
                )?))
            }
            _ => Err(Error::backend(
                "Muse execution group is outside its admitted graph",
            )),
        }
    }
    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        match group {
            0 => forward.vision_active,
            2 => forward.audio.is_some(),
            _ => true,
        }
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _address: eredu_runtime::ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = Vec::new();
        values.extend(forward.mask.iter());
        values.extend(forward.waveform.iter());
        values.extend(
            forward
                .prefill
                .as_ref()
                .and_then(|prefill| prefill.assembled.as_ref()),
        );
        for part in &forward.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => {
                    values.extend([tokens, embeddings]);
                }
                PreparedPart::Media { tokens }
                | PreparedPart::Audio { tokens }
                | PreparedPart::PendingText { tokens } => values.push(tokens),
            }
        }
        if let Some(vision) = &forward.vision {
            values.extend(vision.retained_values());
        }
        values.into_iter()
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> MuseKernel<B> {
    #[allow(clippy::too_many_arguments)]
    fn begin_text_partition<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        if self.selection.is_some() {
            self.validate_partition_state(state)?;
        }
        if !self.owns_input && matches!(&input, LayeredPartitionInput::Tokens(_)) {
            return Err(Error::backend(
                "Muse decoder receiver requires transported ingress",
            ));
        }
        if state.layout() != expected {
            return Err(Error::backend(
                "Muse-Glimmer partition state layout mismatch",
            ));
        }
        let (input, sequence) = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                (TextPartitionInput::Tokens(tokens), tokens.dim(1))
            }
            LayeredPartitionInput::Hidden { hidden, .. } => {
                let sequence = hidden.dim(1);
                (TextPartitionInput::Hidden(hidden), sequence)
            }
        };
        // A decoder receiver can own waveform histories even when no media
        // group executes in this prefix. Advance their logical clock together
        // with the incoming decoder activation.
        self.align_media_clocks(false, sequence, state)?;
        let offset = state
            .layer(self.local_state_ordinal(first_state_ordinal)?)
            .map_err(Error::backend)?
            .offset();
        self.begin_routed_text_partition(input, mask, sequence, offset, parallel, context)
    }

    /// Builds unloaded pinned modules.
    pub fn new(
        args: DecoderConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        Self::materialize(args, None, None, context)
    }

    /// Builds the canonical graph with planner-derived tensor-local geometry.
    pub fn new_parallel(
        args: DecoderConfig,
        geometry: LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        Self::materialize(args, Some(geometry), None, context)
    }

    /// Builds exactly the resources and global unit view retained by an authoritative partition.
    pub fn from_partition(
        args: DecoderConfig,
        geometry: LocalGeometry,
        partition: &eredu_runtime::ArchitecturePartition<
            super::PartitionLocalGeometry,
            eredu_runtime::NoAuxiliaryBoundarySchema,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        Self::materialize(args, Some(geometry), Some(partition), context)
    }

    fn materialize(
        args: DecoderConfig,
        geometry: Option<LocalGeometry>,
        partition: Option<
            &eredu_runtime::ArchitecturePartition<
                super::PartitionLocalGeometry,
                eredu_runtime::NoAuxiliaryBoundarySchema,
            >,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredModel<B>, Error> {
        let definition = std::sync::Arc::new(component_definition(&args, geometry.as_ref())?);
        let selection = partition
            .map(|partition| {
                super::PartitionLocalFoundation::from_partition(&args, partition)
                    .map_err(Error::backend_source)?;
                partition
                    .local_geometry()
                    .validate_complete_geometry(
                        &args,
                        geometry.as_ref().ok_or_else(|| {
                            Error::backend("selected Muse has no complete local geometry")
                        })?,
                    )
                    .map_err(Error::backend_source)?;
                definition.select_partition(partition).map_err(Error::from)
            })
            .transpose()?;
        crate::operator_requirements::require::<B>(
            args.identity.model_type(),
            crate::operator_requirements::muse(&args),
        )?;
        let text = match partition {
            Some(partition) => TextStaticModules::from_parameter_bindings(
                &args,
                geometry.as_ref(),
                partition.parameter_bindings(),
                context,
            )?,
            None => match &geometry {
                Some(geometry) => TextStaticModules::new_parallel(&args, geometry, context)?,
                None => TextStaticModules::new(&args, context)?,
            },
        };
        let vision = match args.vision_config.as_ref() {
            Some(config) => {
                let selected = match partition {
                    Some(partition) => super::VisionStaticSelection::from_parameter_bindings(
                        &args,
                        partition.parameter_bindings(),
                    )?,
                    None => super::VisionStaticSelection::all(config.clone()),
                };
                Some(VisionStatic::from_selection(selected, context)?)
            }
            None => None,
        };
        let kernel = Self {
            args,
            definition: std::sync::Arc::clone(&definition),
            static_modules: StaticModules { text, vision },
            parallel_geometry: geometry.map(std::sync::Arc::new),
            selection,
            owns_input: partition.is_none_or(|partition| partition.ownership().owns_input()),
            expert_realization: None,
        };
        match partition {
            Some(partition) => {
                eredu_runtime::ComponentAdapter::with_partition_resources::<B, _, _>(
                    definition, kernel, partition,
                )
            }
            None => eredu_runtime::ComponentAdapter::with_pinned_resources::<B>(definition, kernel),
        }
        .map_err(Error::from)
    }

    fn selected_state(&self) -> Option<&eredu_runtime::PartitionState> {
        self.selection
            .as_ref()
            .and_then(|selection| selection.state())
    }

    fn execution_address(
        &self,
        group: usize,
        index: usize,
    ) -> Result<eredu_runtime::ExecutionUnitAddress, Error> {
        match &self.selection {
            Some(selection) => selection.address(group, index),
            None => self.definition.unit_address(group, index),
        }
        .map_err(Error::from)
    }

    fn local_state_ordinal(&self, global: usize) -> Result<usize, Error> {
        match self.selected_state() {
            Some(state) if state.owns_global_layer(global) => global
                .checked_sub(state.global_layer_offset())
                .ok_or_else(|| Error::backend("Muse unit precedes its partition state")),
            Some(_) => Err(Error::backend(
                "Muse unit does not own its partition state slot",
            )),
            None => Ok(global),
        }
    }

    fn validate_partition_state<S: LayerRuntimeState<B>>(&self, state: &S) -> Result<(), Error> {
        let complete = self.state_layout_impl()?;
        let expected = self
            .selected_state()
            .map_or(&complete, |state| state.layout());
        if state.layout() != expected {
            return Err(Error::backend(
                "Muse rank-local state layout differs from retained ownership",
            ));
        }
        Ok(())
    }

    /// Binds the exact selected compact expert banks used by partition-local units.
    pub fn set_expert_realization(
        &mut self,
        realization: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<(), Error> {
        if !self.args.is_moe() {
            return Err(Error::backend(
                "dense Muse-Glimmer cannot bind an expert realization",
            ));
        }
        self.expert_realization = Some(std::sync::Arc::new(realization));
        Ok(())
    }

    /// Returns normalized configuration.
    pub const fn args(&self) -> &DecoderConfig {
        &self.args
    }

    /// Returns the replicated or planner-derived mutable-state layout.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        Ok(self.definition.state_layout().clone())
    }

    /// Applies the architecture-owned tensor-parallel token embedding boundary.
    pub fn pipeline_embed_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        B::vocabulary_parallel_lookup(
            self.static_modules.text.embedding_mut()?,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )
    }

    /// Applies final normalization and architecture-owned tensor-parallel
    /// vocabulary projection, including released logit transforms.
    pub fn pipeline_finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.text.final_hidden(hidden, context)?;
        let logits =
            self.static_modules
                .text
                .raw_vocabulary_projection(&hidden, Some(parallel), context)?;
        self.static_modules.text.finish_logits(logits, context)
    }

    /// Enters the serial decoder partition through the family embedding boundary.
    pub fn begin_partition_text(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.token_embeddings(tokens, context)
    }

    /// Enters the tensor-parallel decoder partition through lookup and input normalization.
    pub fn begin_partition_text_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.pipeline_embed_parallel(tokens, parallel, context)?;
        self.static_modules
            .text
            .normalize_embeddings(&hidden, context)
    }

    /// Resumes an already embedded decoder partition in the canonical layered context.
    pub fn resume_partition_text(
        &self,
        hidden: B::Tensor,
        mask: Option<B::Tensor>,
    ) -> LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>> {
        LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                parts: Vec::new(),
                vision: None,
                vision_active: false,
                audio: None,
                waveform: None,
                decoder_positions: 0,
                prefill: None,
            },
        }
    }

    /// Enters or resumes a routed text partition with architecture-owned
    /// causal-mask construction.
    pub fn begin_routed_text_partition(
        &mut self,
        input: TextPartitionInput<'_, B::Tensor>,
        explicit_mask: Option<&B::Tensor>,
        sequence: i32,
        offset: i32,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        let hidden = match input {
            TextPartitionInput::Tokens(tokens) => match parallel {
                Some(parallel) => self.begin_partition_text_parallel(tokens, parallel, context)?,
                None => self.begin_partition_text(tokens, context)?,
            },
            TextPartitionInput::Hidden(hidden) => hidden,
        };
        let mask = match explicit_mask {
            Some(mask) => Some(mask.clone()),
            None if sequence > 1 => Some(B::causal_mask(sequence, offset, None, context)?),
            None => None,
        };
        Ok(self.resume_partition_text(hidden, mask))
    }

    /// Finishes the serial decoder partition through the family output boundary.
    pub fn finish_partition_text(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.project_logits(hidden, context)
    }

    /// Finishes the tensor-parallel decoder partition through the family output boundary.
    pub fn finish_partition_text_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish_parallel(hidden, parallel, context)
    }

    /// Returns planner-derived geometry for a rank-local realization.
    pub fn parallel_geometry(&self) -> Option<&LocalGeometry> {
        self.parallel_geometry.as_deref()
    }

    /// Shares authoritative local geometry with a backend residency policy.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    /// Executes one decoder block with rank-local projections and collectives.
    pub fn forward_text_unit_parallel<S: LayerRuntimeState<B>>(
        &mut self,
        index: usize,
        unit: &mut TransformerBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        self.execution_address(1, index)?;
        unit.forward_parallel(
            hidden,
            forward.mask.as_ref(),
            Some(
                state
                    .layer(self.local_state_ordinal(index)?)
                    .map_err(Error::backend)?,
            ),
            parallel,
            context,
        )
    }

    /// Executes one rank-local decoder block with a runtime-owned routed bank.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_text_unit_parallel_with_provider<S, P>(
        &mut self,
        index: usize,
        unit: &mut TransformerBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.execution_address(1, index)?;
        unit.forward_parallel_with_provider(
            hidden,
            forward.mask.as_ref(),
            Some(
                state
                    .layer(self.local_state_ordinal(index)?)
                    .map_err(Error::backend)?,
            ),
            pass,
            provider,
            parallel,
            context,
        )
    }

    /// Applies the replicated final normalization before a sharded output head.
    pub fn final_parallel_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.final_hidden(hidden, context)
    }

    /// Applies released output scaling and softcapping after vocab gather.
    pub fn finish_parallel_logits(
        &self,
        logits: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.finish_logits(logits, context)
    }

    /// Applies the target's ordinary token embedding and input normalization.
    pub fn token_embeddings(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.embed(tokens, context)
    }

    /// Applies the target-owned final normalization and vocabulary head.
    pub fn project_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.logits(hidden, context)
    }

    /// Executes one text unit while delegating its routed bank to runtime residency.
    pub fn forward_text_unit_with_provider<S, P>(
        &mut self,
        index: usize,
        unit: &mut TransformerBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.execution_address(1, index)?;
        unit.forward_with_provider(
            hidden,
            forward.mask.as_ref(),
            Some(
                state
                    .layer(self.local_state_ordinal(index)?)
                    .map_err(Error::backend)?,
            ),
            pass,
            provider,
            context,
        )
    }

    fn prepare_parts(
        &mut self,
        parts: &[DecoderInputPart<'_, B::Tensor>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<PreparedPart<B::Tensor>>, Error> {
        if parts.is_empty() {
            return Err(Error::backend("Muse-Glimmer input has no ordered parts"));
        }
        parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) if !self.owns_input => {
                    Ok(PreparedPart::PendingText {
                        tokens: (*tokens).clone(),
                    })
                }
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: self.static_modules.text.embed(tokens, context)?,
                }),
                DecoderInputPart::Audio(tokens) => Ok(PreparedPart::Audio {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Media(tokens) => {
                    if tokens.shape().len() != 2 {
                        return Err(Error::backend(
                            "Muse-Glimmer media token IDs must have rank two",
                        ));
                    }
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
                    })
                }
            })
            .collect()
    }

    fn prepare_parts_parallel(
        &mut self,
        parts: &[DecoderInputPart<'_, B::Tensor>],
        defer_embeddings: bool,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<PreparedPart<B::Tensor>>, Error> {
        if parts.is_empty() {
            return Err(Error::backend("Muse-Glimmer input has no ordered parts"));
        }
        parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) if defer_embeddings || !self.owns_input => {
                    Ok(PreparedPart::PendingText {
                        tokens: (*tokens).clone(),
                    })
                }
                DecoderInputPart::Text(tokens) => {
                    let embeddings = B::vocabulary_parallel_lookup(
                        self.static_modules.text.embedding_mut()?,
                        tokens,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?;
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: self
                            .static_modules
                            .text
                            .normalize_embeddings(&embeddings, context)?,
                    })
                }
                DecoderInputPart::Audio(tokens) => Ok(PreparedPart::Audio {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Media(tokens) => {
                    if tokens.shape().len() != 2 {
                        return Err(Error::backend(
                            "Muse-Glimmer media token IDs must have rank two",
                        ));
                    }
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
                    })
                }
            })
            .collect()
    }

    fn assemble(
        &self,
        parts: &[PreparedPart<B::Tensor>],
        vision: Option<&B::Tensor>,
        audio: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let media_tokens = parts
            .iter()
            .filter_map(|part| match part {
                PreparedPart::Media { tokens } => Some(tokens.dim(1)),
                PreparedPart::Text { .. }
                | PreparedPart::PendingText { .. }
                | PreparedPart::Audio { .. } => None,
            })
            .sum::<i32>();
        match vision {
            Some(vision) if vision.shape() != [media_tokens, self.args.hidden_size] => {
                return Err(Error::backend(format!(
                    "Muse-Glimmer projected media has shape {:?}, expected [{media_tokens}, {}]",
                    vision.shape(),
                    self.args.hidden_size
                )))
            }
            None if media_tokens != 0 => {
                return Err(Error::backend(
                    "Muse-Glimmer media placeholders require projected media",
                ))
            }
            _ => {}
        }
        let audio_tokens = parts.iter().try_fold(0i32, |total, part| {
            let length = match part {
                PreparedPart::Audio { tokens } => tokens.dim(1),
                _ => 0,
            };
            total
                .checked_add(length)
                .ok_or_else(|| Error::backend("Muse audio span exceeds i32"))
        })?;
        let batch = parts
            .first()
            .map(|part| match part {
                PreparedPart::Text { tokens, .. }
                | PreparedPart::PendingText { tokens }
                | PreparedPart::Media { tokens }
                | PreparedPart::Audio { tokens } => tokens.dim(0),
            })
            .ok_or_else(|| Error::backend("Muse input has no ordered parts"))?;
        match audio {
            Some(value) if value.shape() != [batch, audio_tokens, self.args.hidden_size] => {
                return Err(Error::backend(
                    "Muse projected waveform geometry differs from its ordered placeholders",
                ))
            }
            None if audio_tokens != 0 => {
                return Err(Error::backend(
                    "Muse audio placeholders require projected waveform input",
                ))
            }
            _ => {}
        }
        let mut owned_embeddings = Vec::with_capacity(parts.len());
        let media_element = match self.args.media_embedding_precision {
            super::config::MediaEmbeddingPrecision::Preserve => None,
            super::config::MediaEmbeddingPrecision::Decoder
                if media_tokens != 0 || audio_tokens != 0 =>
            {
                Some(
                    self.static_modules
                        .text
                        .embedding()?
                        .output_element_type()
                        .ok_or_else(|| {
                            Error::backend("Muse decoder embedding output precision is unavailable")
                        })?,
                )
            }
            super::config::MediaEmbeddingPrecision::Decoder => None,
        };
        let cast_media = |value: B::Tensor| match media_element {
            Some(element) if value.element_type() != Some(element) => {
                value.cast_float(element, context)
            }
            _ => Ok(value),
        };
        let mut offset = 0;
        let mut audio_offset = 0;
        for part in parts {
            match part {
                PreparedPart::PendingText { .. } => {
                    return Err(Error::backend(
                        "Muse text embeddings must be completed in the decoder ingress wave",
                    ));
                }
                PreparedPart::Text { embeddings, .. } => {
                    owned_embeddings.push(embeddings.clone());
                }
                PreparedPart::Audio { tokens } => {
                    let end = audio_offset + tokens.dim(1);
                    owned_embeddings.push(cast_media(
                        audio.expect("validated waveform exists").index(
                            &[
                                eredu_nn::Index::Full,
                                eredu_nn::Index::Range(audio_offset, end),
                                eredu_nn::Index::Full,
                            ],
                            context,
                        )?,
                    )?);
                    audio_offset = end;
                }
                PreparedPart::Media { tokens } => {
                    let length = tokens.dim(1);
                    let media = vision
                        .expect("validated media exists")
                        .index(
                            &[
                                eredu_nn::Index::Range(offset, offset + length),
                                eredu_nn::Index::Full,
                            ],
                            context,
                        )?
                        .expand_dims(0, context)?;
                    owned_embeddings.push(cast_media(media)?);
                    offset += length;
                }
            }
        }
        let batch = parts
            .first()
            .map(|part| match part {
                PreparedPart::Text { tokens, .. }
                | PreparedPart::PendingText { tokens }
                | PreparedPart::Media { tokens }
                | PreparedPart::Audio { tokens } => tokens.dim(0),
            })
            .ok_or_else(|| Error::backend("Muse-Glimmer input has no ordered parts"))?;
        for (index, (part, embeddings)) in parts.iter().zip(&owned_embeddings).enumerate() {
            let tokens = match part {
                PreparedPart::Text { tokens, .. }
                | PreparedPart::PendingText { tokens }
                | PreparedPart::Media { tokens }
                | PreparedPart::Audio { tokens } => tokens,
            };
            if tokens.shape().len() != 2
                || embeddings.shape().len() != 3
                || tokens.dim(0) != batch
                || embeddings.dim(0) != batch
                || tokens.dim(1) != embeddings.dim(1)
                || embeddings.dim(2) != self.args.hidden_size
            {
                return Err(Error::backend(format!(
                    "Muse-Glimmer ordered input part {index} has incompatible token/embedding shapes {:?} and {:?}",
                    tokens.shape(),
                    embeddings.shape()
                )));
            }
        }
        B::Tensor::concatenate(&owned_embeddings, 1, context)
    }
}

impl<B, S> eredu_runtime::ComponentEquations<B, S> for MuseKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
    }
    fn execute_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let pass = <Self as eredu_runtime::RoutedComponentKernel<B, S>>::expert_pass(
            self, address, hidden, forward,
        );
        self.forward_text_observed(
            address,
            unit,
            hidden,
            state,
            forward,
            pass,
            &mut eredu_runtime::ResidentExpertProvider,
            context,
            observer,
        )
    }
    fn finish_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.finish_text_observed(hidden, None, context, observer)
    }
    fn group_output_observation_path(&self, group: usize) -> Result<Option<String>, Self::Error> {
        Ok((group == 0).then(|| eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH.to_owned()))
    }
    fn group_input_observation_path(&self, group: usize) -> Result<Option<String>, Self::Error> {
        Ok((group == 1).then(|| eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH.to_owned()))
    }
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        match input {
            crate::composite_execution::CompositeModelInput::Native(input) => {
                self.begin_native(input, state, context)
            }
            crate::composite_execution::CompositeModelInput::Prepared(input) => {
                self.begin_prepared(input, state, context)
            }
        }
    }
    fn begin_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 1 && !self.owns_input {
            return Err(Error::backend(
                "Muse decoder receiver requires transported ingress",
            ));
        }

        if group == 1 {
            if let Some(assembled) = forward
                .prefill
                .as_ref()
                .and_then(|prefill| prefill.assembled.as_ref())
                .cloned()
            {
                return self.retain_prefill_ingress(group, assembled, forward, context);
            }
        }
        let hidden = (|| {
            if group == 1 && self.args.audio_config.is_some() {
                // Ordinary traversal retains inactive dependency values; partitioned
                // traversal transports only active producers. Both carry the same
                // ordered semantic roots: vision, then waveform.
                let (vision, audio) = if let [vision, audio] = dependencies {
                    (
                        forward.vision_active.then_some(*vision),
                        forward.audio.as_ref().map(|_| *audio),
                    )
                } else {
                    let mut values = dependencies.iter().copied();
                    let vision = if forward.vision_active {
                        Some(values.next().ok_or_else(|| {
                            Error::backend("Muse decoder ingress lost its active vision dependency")
                        })?)
                    } else {
                        None
                    };
                    let audio = if forward.audio.is_some() {
                        Some(values.next().ok_or_else(|| {
                            Error::backend("Muse decoder ingress lost its active audio dependency")
                        })?)
                    } else {
                        None
                    };
                    if values.next().is_some() {
                        return Err(Error::backend(
                            "Muse decoder ingress has excess media dependencies",
                        ));
                    }
                    (vision, audio)
                };
                return self.assemble(&forward.parts, vision, audio, context);
            }
            match (group, dependencies) {
                (0, []) => Ok(initial.clone()),
                (2, []) => Ok(forward.waveform.as_ref().unwrap_or(initial).clone()),
                (1, [vision_or_assembled]) if forward.vision_active => {
                    self.assemble(&forward.parts, Some(vision_or_assembled), None, context)
                }
                (1, []) if !forward.vision_active => Ok(initial.clone()),
                (1, [assembled]) => Ok((*assembled).clone()),
                _ => Err(Error::backend(
                    "invalid Muse-Glimmer execution dependencies",
                )),
            }
        })()?;
        self.retain_prefill_ingress(group, hidden, forward, context)
    }
    fn execute(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let (group, index) = (address.group(), address.index());
        match (group, unit) {
            (2, Unit::Audio(unit)) => {
                let audio = self
                    .args
                    .audio_config
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse has no audio configuration"))?;
                let kind = audio.execution_units()?[index];
                let layer = state
                    .layer(self.local_state_ordinal(self.args.num_hidden_layers as usize + index)?)
                    .map_err(Error::backend)?;
                super::audio::execution::execute_audio_unit::<B, _>(
                    audio,
                    kind,
                    index,
                    unit,
                    hidden,
                    layer,
                    forward
                        .audio
                        .as_mut()
                        .ok_or_else(|| Error::backend("Muse waveform context is missing"))?,
                    forward.decoder_positions,
                    context,
                )
            }
            (0, Unit::Vision(unit)) => unit.forward_scheduled(
                hidden,
                self.args
                    .vision_config
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?
                    .schedule[index],
                forward
                    .vision
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer vision state is missing"))?,
                context,
            ),
            (1, Unit::Text(unit)) => {
                let pass = <Self as eredu_runtime::RoutedComponentKernel<B, S>>::expert_pass(
                    self, address, hidden, forward,
                );
                self.forward_text_unit_with_provider(
                    index,
                    unit,
                    hidden,
                    state,
                    forward,
                    pass,
                    &mut eredu_runtime::ResidentExpertProvider,
                    context,
                )
            }
            _ => Err(Error::backend("Muse-Glimmer unit/group mismatch")),
        }
    }
    fn complete_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 2 {
            return match &forward.audio {
                Some(audio) => {
                    let features = super::audio::execution::finish_audio_forward(
                        self.args.audio_config.as_ref().ok_or_else(|| {
                            Error::backend("Muse waveform completion has no admitted tower")
                        })?,
                        hidden,
                        audio,
                    )?;
                    self.static_modules
                        .text
                        .normalize_embeddings(&features, context)
                }
                None => Ok(hidden.clone()),
            };
        }
        match (group, forward.vision.as_ref()) {
            (0, Some(vision)) => {
                let media = self
                    .static_modules
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("Muse-Glimmer model has no vision projector"))?
                    .finish(hidden, vision, context)?;
                Ok(media)
            }
            (0, None) | (1, _) => Ok(hidden.clone()),
            _ => Err(Error::backend("invalid Muse-Glimmer execution group")),
        }
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.static_modules.text.logits(hidden, context)
    }
}

impl<B, S> eredu_runtime::ParallelComponentKernel<B, S> for MuseKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
    }
    fn execute_parallel_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let pass = <Self as eredu_runtime::RoutedComponentKernel<B, S>>::expert_pass(
            self, address, hidden, forward,
        );
        self.forward_text_parallel_observed(
            address,
            unit,
            hidden,
            state,
            forward,
            pass,
            &mut eredu_runtime::ResidentExpertProvider,
            parallel,
            context,
            observer,
        )
    }
    fn finish_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.finish_text_observed(hidden, Some(parallel), context, observer)
    }
    fn begin_group_parallel(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 1 && self.owns_input {
            for part in &mut forward.parts {
                if let PreparedPart::PendingText { tokens } = part {
                    let embeddings = B::vocabulary_parallel_lookup(
                        self.static_modules.text.embedding_mut()?,
                        tokens,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?;
                    *part = PreparedPart::Text {
                        tokens: tokens.clone(),
                        embeddings: self
                            .static_modules
                            .text
                            .normalize_embeddings(&embeddings, context)?,
                    };
                }
            }
        }
        self.begin_group(group, initial, dependencies, state, forward, context)
    }
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        match input {
            crate::composite_execution::CompositeModelInput::Native(input) => {
                self.begin_native_parallel(input, state, parallel, context)
            }
            crate::composite_execution::CompositeModelInput::Prepared(input) => {
                self.begin_prepared_parallel(input, state, parallel, context)
            }
        }
    }
    fn execute_parallel(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let (group, index) = (address.group(), address.index());
        match (group, unit) {
            (1, Unit::Text(unit)) => {
                let pass = <Self as eredu_runtime::RoutedComponentKernel<B, S>>::expert_pass(
                    self, address, hidden, forward,
                );
                self.forward_text_unit_parallel_with_provider(
                    index,
                    unit,
                    hidden,
                    state,
                    forward,
                    pass,
                    &mut eredu_runtime::ResidentExpertProvider,
                    parallel,
                    context,
                )
            }
            (_, unit) => self.execute(address, unit, hidden, state, forward, context),
        }
    }
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if self.parallel_geometry.is_none() {
            return Err(Error::backend(
                "Muse-Glimmer model was not built with rank-local geometry",
            ));
        }
        self.static_modules.text.logits_instrumented(
            hidden,
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
}

impl<B, S> eredu_runtime::PartitionComponentKernel<B, S> for MuseKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn partition_observation_hooks(
        &self,
        _tensor_parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
    }

    fn finish_partition_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredPartitionOutput<B::Tensor>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, false, parallel, context);
        }
        Ok(LayeredPartitionOutput::Final {
            output: self.finish_text_observed(hidden, parallel, context, observer)?,
            retained: None,
        })
    }

    type Boundary = eredu_runtime::NoAuxiliaryBoundarySchema;

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(eredu_runtime::NoAuxiliaryBoundarySchema::new(
            self.args().hidden_size,
        ))
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = selected
            .expected
            .ok_or_else(|| Error::backend("Muse text partition has no selected state geometry"))?;
        self.begin_text_partition(
            input,
            mask,
            state,
            expected,
            selected.first_state_ordinal,
            selected.parallel,
            selected.native,
        )
    }

    fn enter_partition_group(
        &mut self,
        _group: usize,
        initial: &B::Tensor,
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _parallel: Option<&B::ParallelContext>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Ok(initial.clone())
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor>, Self::Error> {
        if owns_output {
            let output = match parallel {
                Some(parallel) => {
                    self.finish_parallel(hidden, state, forward, parallel, context)?
                }
                None => self.finish(hidden, state, forward, context)?,
            };
            Ok(LayeredPartitionOutput::Final {
                output,
                retained: None,
            })
        } else {
            Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: eredu_runtime::NoAuxiliaryBoundary,
            })
        }
    }
}

impl<B> eredu_runtime::ComponentPinnedResources<B> for MuseKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    type Modules = StaticModules<B>;
    fn pinned_modules(&self) -> &Self::Modules {
        &self.static_modules
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        &mut self.static_modules
    }
    fn visit_pinned<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        if let Some(vision) = self
            .static_modules
            .vision
            .as_ref()
            .filter(|vision| vision.has_parameters())
        {
            visitor.visit("vision", vision)?;
        }
        self.static_modules.text.decoder_resources().visit(visitor)
    }
    fn visit_pinned_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        if let Some(vision) = self
            .static_modules
            .vision
            .as_mut()
            .filter(|vision| vision.has_parameters())
        {
            visitor.visit_mut("vision", vision)?;
        }
        self.static_modules
            .text
            .decoder_resources_mut()
            .visit(visitor)
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> MuseKernel<B> {
    fn begin_prepared<'a, S>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, MuseInputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let model = self;
        if let Some(resumed) = model.resume_prefill(input.decoder_chunk(), state, context)? {
            return Ok(resumed);
        }
        model.validate_partition_state(state)?;
        let chunk = input.decoder_chunk();
        let positions = chunk
            .as_ref()
            .map(|chunk| i32::try_from(chunk.range.len()).map_err(Error::backend))
            .transpose()?;
        let prepared = prepare_composite_ingress::<B>(input, context)?;
        let decoder_parts = prepared.decoder_parts();
        let parts = model.prepare_parts(&decoder_parts, context)?;
        let mut forward = model.begin_media(
            prepared.vision_input(),
            prepared.audio_input(),
            None,
            parts,
            positions,
            state,
            context,
        )?;
        forward.context.prefill = chunk.map(|chunk| DecoderPrefill {
            range: chunk.range,
            assembled: None,
        });
        Ok(forward)
    }
    fn begin_prepared_parallel<'a, S>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, MuseInputPartPlan>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        let model = self;
        if let Some(resumed) = model.resume_prefill(input.decoder_chunk(), state, context)? {
            return Ok(resumed);
        }
        model.validate_partition_state(state)?;
        let chunk = input.decoder_chunk();
        let positions = chunk
            .as_ref()
            .map(|chunk| i32::try_from(chunk.range.len()).map_err(Error::backend))
            .transpose()?;
        let prepared = prepare_composite_ingress::<B>(input, context)?;
        let decoder_parts = prepared.decoder_parts();
        if model.parallel_geometry.is_none() {
            return Err(Error::backend("Muse model has no rank-local geometry"));
        }
        let parts = model.prepare_parts_parallel(
            &decoder_parts,
            prepared.vision_input().is_some() || prepared.audio_input().is_some(),
            parallel,
            context,
        )?;
        let mut forward = model.begin_media(
            prepared.vision_input(),
            prepared.audio_input(),
            None,
            parts,
            positions,
            state,
            context,
        )?;
        forward.context.prefill = chunk.map(|chunk| DecoderPrefill {
            range: chunk.range,
            assembled: None,
        });
        Ok(forward)
    }
    fn begin_native<'a, S>(
        &mut self,
        input: ModelInput<'a, B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.validate_partition_state(state)?;
        let parts = self.prepare_parts(input.parts, context)?;
        self.begin_media(
            input.vision,
            input.audio,
            input.mask,
            parts,
            None,
            state,
            context,
        )
    }
    fn begin_native_parallel<'a, S>(
        &mut self,
        input: ModelInput<'a, B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        if self.parallel_geometry.is_none() {
            return Err(Error::backend(
                "Muse-Glimmer model was not built with rank-local geometry",
            ));
        }
        self.validate_partition_state(state)?;
        // Text lookup reductions belong to the primary decoder wave, after the
        // replicated vision equations. Other pipeline ranks can then enter the
        // same exact segmented sums before the decoder's routed collectives.
        let parts = self.prepare_parts_parallel(
            input.parts,
            input.vision.is_some() || input.audio.is_some(),
            parallel,
            context,
        )?;
        self.begin_media(
            input.vision,
            input.audio,
            input.mask,
            parts,
            None,
            state,
            context,
        )
    }
}
