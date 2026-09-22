//! One neutral Inkling multimodal model for resident and bounded runtimes.

mod construction;
mod media_prefill;
pub(crate) use construction::RetainedModelSource;
pub use media_prefill::{MediaIngress, MediaPrefillPlan};

use std::sync::Arc;

use eredu_core::cache::PromptCacheTopology;
use eredu_nn::{
    AuxiliaryConvolutionState, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, Error,
    GroupedNeuralBackend, Index, LinearSpec, NormalizationConstructionSpec, NormalizationOperator,
    ParameterSpec, Parameterized, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout, ExpertPass,
    LayerRuntimeState, LayeredArchitecture, LayeredForwardState, LayeredPartitionInput,
    LayeredPartitionOutput, ModelStateIdentity, OwnedParameterGroupSpec,
    ParallelLayeredArchitecture, ParallelRoutedLayeredArchitecture, ParameterGroupOwner,
    PartitionState, PartitionedLayeredArchitecture, RoutedExpertProvider,
    RoutedLayeredArchitecture, StateLayout,
};

use super::{
    composite_state_layout, layer_parameter_groups, mtp_parameter_groups, mtp_state_layout,
    state_layout, static_parameter_groups, vision_layer_parameter_groups, AudioInput, AudioTower,
    DecoderLayer, LocalGeometry, ModelArgs, MtpModel, MtpOutput, VisionLayer, VisionStatic,
};
use crate::{
    composite_execution::{CompositeArchitecture, PreparedCompositeInput},
    media_plan::InklingInputPartPlan,
};

/// Canonical static-role identity for the embedded prediction modules and
/// their persistent state owner.
pub const MTP_STATIC_ROLE: &str = "mtp";
/// Stable execution-group identity for Inkling vision ingress.
pub const VISION_EXECUTION_GROUP: &str = "vision";
/// Stable execution-group identity for Inkling audio ingress.
pub const AUDIO_EXECUTION_GROUP: &str = "audio";
/// Stable execution-group identity for Inkling text decoding.
pub const TEXT_EXECUTION_GROUP: &str = "text_decoder";

/// The hMLP static role contains only final normalization, executed after the
/// last folded projection. Cold selection and constructed execution share this
/// ownership declaration; the completed image still merges on the first owner.
pub(crate) fn vision_group_transport() -> eredu_runtime::ArchitectureGroupTransport {
    eredu_runtime::ArchitectureGroupTransport {
        placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
        kind: eredu_runtime::ArchitectureGroupKind::VisionEncoder,
        first_owner_static_roles: Vec::new(),
        last_owner_static_roles: vec!["vision".into()],
        merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
        parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
        request_optional: true,
    }
}

/// Pinned text, audio, and image modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Token embedding table.
    pub embeddings: B::Embedding,
    /// Required normalization after complete text/media assembly.
    pub embedding_norm: B::Normalization,
    /// Final decoder normalization.
    pub final_norm: B::Normalization,
    /// Untied output projection.
    pub output: B::Linear,
    /// Optional checkpoint-embedded multi-token predictor.
    pub mtp: Option<MtpModel<B>>,
    /// Optional pinned dMel projector.
    pub audio: Option<AudioTower<B>>,
    /// Optional pinned hMLP final normalization.
    pub vision: Option<VisionStatic<B>>,
}

/// One ordered decoder-ingress segment.
pub enum DecoderInputPart<'a, T> {
    /// Ordinary text token IDs.
    Text(&'a T),
    /// Image placeholder IDs matching projected hMLP output.
    Image(&'a T),
    /// Audio placeholder IDs matching projected dMel frames.
    Audio(&'a T),
    /// Caller-supplied decoder-width embeddings with explicit token identity.
    Projected {
        /// Semantic token IDs used for cache and position identity.
        tokens: &'a T,
        /// Decoder-width embeddings that bypass native media towers.
        embeddings: &'a T,
    },
}

/// Prepared token and optional native media input.
pub struct ModelInput<'a, T> {
    /// Ordered text/image/audio segments.
    pub parts: &'a [DecoderInputPart<'a, T>],
    /// Optional hMLP patches shaped `[patches, 2, 40, 40, 3]`.
    pub vision_patches: Option<&'a T>,
    /// Optional prepared dMel code IDs and valid-frame extent.
    pub audio: Option<AudioInput<'a, T>>,
}

/// Architecture-owned tensor assembly for one admitted Inkling request.
pub struct PreparedInput<T> {
    tokens: Vec<T>,
    modalities: Vec<eredu_core::InputModality>,
    projected: Vec<Option<T>>,
    images: Option<T>,
    audio: Option<T>,
    audio_frames: i32,
}

impl<T> PreparedInput<T> {
    /// Borrows the assembled request through the canonical model input vocabulary.
    pub fn with_model_input<R>(&self, apply: impl FnOnce(ModelInput<'_, T>) -> R) -> R {
        let parts = self
            .tokens
            .iter()
            .zip(&self.modalities)
            .zip(&self.projected)
            .map(|((tokens, modality), projected)| match projected {
                Some(embeddings) => DecoderInputPart::Projected { tokens, embeddings },
                None => match modality {
                    eredu_core::InputModality::Text => DecoderInputPart::Text(tokens),
                    eredu_core::InputModality::Image => DecoderInputPart::Image(tokens),
                    eredu_core::InputModality::Audio => DecoderInputPart::Audio(tokens),
                    _ => unreachable!("Inkling admission rejects other modalities"),
                },
            })
            .collect::<Vec<_>>();
        let audio = self.audio.as_ref().map(|code_ids| AudioInput {
            code_ids,
            valid_frames: self.audio_frames,
        });
        apply(ModelInput {
            parts: &parts,
            vision_patches: self.images.as_ref(),
            audio,
        })
    }
}

fn visit_semantic_tokens<T: Tensor>(
    input: PreparedCompositeInput<'_, T, InklingInputPartPlan>,
    visitor: &mut dyn FnMut(
        crate::composite_execution::PredictionTokenPart<'_, T>,
    ) -> Result<(), Error>,
) -> Result<(), Error> {
    let metadata = crate::decoder::identity::Metadata::new(input.metadata());
    metadata.controls::<(
        InklingInputPartPlan,
        crate::composite_execution::PredictionTokenPart<'_, T>,
    )>()?;
    for (part, plan) in input
        .prepared()
        .parts()
        .iter()
        .zip(input.admitted().inkling_parts())
    {
        let placeholder = match plan {
            InklingInputPartPlan::TextTokens { .. } => None,
            InklingInputPartPlan::Projected {
                placeholder_token_id,
                positions,
                ..
            } => Some((placeholder_token_id, positions)),
            InklingInputPartPlan::Media { ingress, .. } => {
                Some((ingress.placeholder_token_id, ingress.placeholder_count))
            }
        };
        let token = if let Some((token, positions)) = placeholder {
            crate::composite_execution::PredictionTokenPart::Repeated { token, positions }
        } else {
            match (part.modality(), part.payload()) {
                (
                    eredu_core::InputModality::Text,
                    eredu_runtime::PreparedInputPayload::TokenIds(tokens),
                ) => crate::composite_execution::PredictionTokenPart::Tokens(tokens),
                _ => {
                    return Err(
                        metadata.error(format_args!("Inkling text part has no token identity"))
                    )
                }
            }
        };
        visitor(token)?;
    }
    Ok(())
}

fn prepared_semantic_tokens<T: Tensor>(
    input: PreparedCompositeInput<'_, T, InklingInputPartPlan>,
    context: &T::Context,
) -> Result<Vec<T>, Error> {
    let metadata = crate::decoder::identity::Metadata::new(input.metadata());
    let mut parts = metadata.vector(input.prepared().len())?;
    visit_semantic_tokens(input, &mut |part| {
        parts.push(crate::composite_execution::prediction_tokens::materialize(
            part, context, metadata,
        )?);
        Ok(())
    })?;
    Ok(parts)
}

/// Materializes placeholders, retained media extents, and ordered segments from
/// an architecture admission.
pub fn prepare_input<T: Tensor>(
    input: PreparedCompositeInput<'_, T, InklingInputPartPlan>,
    context: &T::Context,
) -> Result<PreparedInput<T>, Error> {
    let metadata = crate::decoder::ModuleMetadata::destination(input.metadata());
    metadata.controls::<(PreparedInput<T>, InklingInputPartPlan, usize, i32)>()?;
    let prepared = input.prepared();
    let admitted = input.admitted();
    let tokens = prepared_semantic_tokens(input, context)?;
    let mut modalities = metadata.vector(prepared.len())?;
    let mut projected = metadata.vector(prepared.len())?;
    let mut images = metadata.vector(prepared.len())?;
    let mut audio = metadata.vector(prepared.len())?;
    let mut audio_frames = 0i32;
    for (part, plan) in prepared.parts().iter().zip(admitted.inkling_parts()) {
        match plan {
            InklingInputPartPlan::TextTokens { .. } => {
                modalities.push(eredu_core::InputModality::Text);
                projected.push(None);
            }
            InklingInputPartPlan::Projected { modality, .. } => {
                let eredu_runtime::PreparedInputPayload::Embeddings(value) = part.payload() else {
                    return Err(Error::backend(
                        "Inkling admitted projected part lost its embedding payload",
                    ));
                };

                modalities.push(modality);
                projected.push(Some(value.clone()));
            }
            InklingInputPartPlan::Media {
                modality, ingress, ..
            } => {
                let eredu_runtime::PreparedInputPayload::Tensor(value) = part.payload() else {
                    return Err(Error::backend(
                        "Inkling admitted media part lost its tensor payload",
                    ));
                };
                let count = i32::try_from(ingress.placeholder_count)
                    .map_err(|_| Error::backend("Inkling media span exceeds I32"))?;

                modalities.push(modality);
                projected.push(None);
                match modality {
                    eredu_core::InputModality::Image => images.push(value.clone()),
                    eredu_core::InputModality::Audio => {
                        audio.push(
                            value.index(
                                &[Index::Full, Index::Range(0, count), Index::Full],
                                context,
                            )?,
                        );
                        audio_frames = audio_frames
                            .checked_add(count)
                            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
                    }
                    _ => {
                        return Err(Error::backend(
                            "Inkling media admission contains an unsupported modality",
                        ));
                    }
                }
            }
        }
    }
    let images = match images.len() {
        0 => None,
        1 => images.pop(),
        _ => Some(T::concatenate(&images, 0, context)?),
    };
    let audio = match audio.len() {
        0 => None,
        1 => audio.pop(),
        _ => Some(T::concatenate(&audio, 1, context)?),
    };
    Ok(PreparedInput {
        tokens,
        modalities,
        projected,
        images,
        audio,
        audio_frames,
    })
}

impl<B, S> CompositeArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
{
    type InputPartPlan = InklingInputPartPlan;
    type AdmissionConfig = crate::replicated_text::SharedCompositeConfig<ModelArgs>;

    fn admission_config(&self) -> Self::AdmissionConfig {
        self.construction.args().clone()
    }

    fn retain_admission_config_with_metadata(
        config: &Self::AdmissionConfig,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self::AdmissionConfig, Error> {
        crate::decoder::identity::Metadata::new(Some(context))
            .controls::<(Self::AdmissionConfig, &Self::AdmissionConfig)>()?;
        Ok(config.clone())
    }

    fn admit_prepared_input(
        config: &Self::AdmissionConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
    ) -> Result<
        crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>,
        eredu_core::CapabilityError,
    > {
        crate::media_plan::admit_inkling_input(config, input, inspector)
    }

    fn admit_prepared_input_with_metadata(
        config: &Self::AdmissionConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>, Error> {
        crate::media_plan::admission::inkling(config, input, inspector, context)
    }

    fn visit_prepared_prediction_tokens(
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        visitor: &mut dyn FnMut(
            crate::composite_execution::PredictionTokenPart<'_, B::Tensor>,
        ) -> Result<(), Error>,
    ) -> Result<(), Error> {
        visit_semantic_tokens(input, visitor)
    }

    fn should_execute_prepared_group(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> bool {
        match group {
            0 => input.admitted().inkling_parts().any(|part| {
                matches!(
                    part,
                    InklingInputPartPlan::Media {
                        modality: eredu_core::InputModality::Image,
                        ..
                    }
                )
            }),
            1 => input.admitted().inkling_parts().any(|part| {
                matches!(
                    part,
                    InklingInputPartPlan::Media {
                        modality: eredu_core::InputModality::Audio,
                        ..
                    }
                )
            }),
            2 => true,
            _ => false,
        }
    }

    fn prepared_group_boundary_sequence(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<i32, String> {
        let positions = if group < 2 {
            let modality = if group == 0 {
                eredu_core::InputModality::Image
            } else {
                eredu_core::InputModality::Audio
            };
            input
                .admitted()
                .inkling_parts()
                .filter_map(|part| match part {
                    InklingInputPartPlan::Media {
                        modality: actual,
                        shape,
                        ..
                    } if actual == modality => Some(shape.decoder_positions),
                    _ => None,
                })
                .try_fold(0u64, |n, v| n.checked_add(v))
                .ok_or("Inkling boundary sequence overflowed")?
        } else {
            input.admitted().decoder_positions()
        };
        i32::try_from(positions).map_err(|_| "Inkling boundary sequence exceeds i32".to_owned())
    }

    fn group_continuation_geometry_at(
        &self,
        group: usize,
        source_unit_end: Option<usize>,
        source_sequence: i32,
        prepared: Option<(i32, i32)>,
    ) -> Result<Option<(i32, i32)>, Error> {
        if group != 0 {
            return Ok(prepared);
        }
        let end = source_unit_end.ok_or_else(|| {
            Error::backend("Inkling hMLP continuation requires an exact selected cut")
        })?;
        let shape = self.hmlp_cut_shape(source_sequence, end)?;
        let rows = shape[..4]
            .iter()
            .try_fold(1i32, |n, d| n.checked_mul(*d))
            .ok_or_else(|| Error::backend("Inkling continuation rows overflowed"))?;
        Ok(Some((rows, shape[4])))
    }

    fn encode_group_continuation_at(
        &self,
        group: usize,
        source_unit_end: Option<usize>,
        source_sequence: i32,
        hidden: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group != 0 {
            return Ok(hidden);
        }
        let end = source_unit_end.ok_or_else(|| {
            Error::backend("Inkling hMLP continuation requires an exact selected cut")
        })?;
        let shape = self.hmlp_cut_shape(source_sequence, end)?;
        if hidden.shape() != shape {
            return Err(Error::backend(
                "Inkling hMLP source shape differs from selected cut",
            ));
        }
        let rows = shape[..4]
            .iter()
            .try_fold(1i32, |n, d| n.checked_mul(*d))
            .ok_or_else(|| Error::backend("Inkling continuation rows overflowed"))?;
        hidden.reshape(&[1, rows, shape[4]], context)
    }

    fn decode_group_continuation_at(
        &self,
        group: usize,
        source_unit_end: usize,
        source_sequence: i32,
        hidden: B::Tensor,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group != 0 {
            return Ok(hidden);
        }
        let shape = self.hmlp_cut_shape(source_sequence, source_unit_end)?;
        let rows = shape[..4]
            .iter()
            .try_fold(1i32, |n, d| n.checked_mul(*d))
            .ok_or_else(|| Error::backend("Inkling continuation rows overflowed"))?;
        if hidden.shape() != [1, rows, shape[4]] {
            return Err(Error::backend(
                "Inkling hMLP wire shape differs from selected cut",
            ));
        }
        hidden.reshape(&shape, context)
    }

    fn accept_partition_boundary(
        &mut self,
        source_group: usize,
        destination_group: usize,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        values: Vec<B::Tensor>,
        forward: &mut Self::ForwardContext,
    ) -> Result<Option<B::Tensor>, Error> {
        if !matches!(
            (source_group, destination_group),
            (0, 0) | (0, 2) | (1, 2) | (2, 2)
        ) {
            return Ok(None);
        }
        if values.len() != 1 || !schema.auxiliary().is_empty() {
            return Err(Error::backend(
                "Inkling boundary requires exactly its primary activation",
            ));
        }
        let hidden = values.into_iter().next().expect("validated primary");
        match (source_group, destination_group) {
            // A received encoder result is already completed and projected.
            // Install it before the shared retained-ingress cut consumes this
            // context; an hMLP continuation must never populate this final slot.
            (0, 2) => forward.vision_output = Some(hidden.clone()),
            (1, 2) => forward.audio_output = Some(hidden.clone()),
            // The common executor restores hMLP rank5 at its exact next unit.
            // Decoder continuation resumes this already-retained context and
            // must not assemble or normalize the incoming activation again.
            (0, 0) | (2, 2) => {}
            _ => unreachable!("matched supported boundary"),
        }
        Ok(Some(hidden))
    }

    fn prepared_group_collective_waves(
        &self,
        group: usize,
        _input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        tensor_partitions: usize,
        pipeline_stages: usize,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, Error>
    {
        // Inkling's hMLP units and projector are replicated equations even
        // when their parameters are selected with TP-local ownership. The
        // explicit empty schedule prevents the generic driver from inventing
        // row reductions for this group.
        let destination = crate::composite_execution::graph::Destination(context);
        destination.controls::<(
            &Self,
            usize,
            PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
            usize,
            usize,
            Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>,
        )>()?;
        if group != 0 || tensor_partitions <= 1 || pipeline_stages <= 1 {
            return Ok(None);
        }
        destination
            .collect((0..pipeline_stages).map(|_| Vec::new()))
            .map(Some)
    }

    fn prepared_primary_ingress_collectives(
        &self,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        tensor_partitions: usize,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>, Error> {
        let destination = crate::composite_execution::graph::Destination(context);
        destination.controls::<(
            &Self,
            PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
            usize,
        )>()?;
        crate::composite_execution::segmented_token_ingress_collectives_in(
            input
                .admitted()
                .inkling_parts()
                .filter_map(|part| match part {
                    InklingInputPartPlan::TextTokens { positions } => Some(positions),
                    InklingInputPartPlan::Projected { .. } | InklingInputPartPlan::Media { .. } => {
                        None
                    }
                }),
            self.args.text_config.hidden_size,
            tensor_partitions,
            destination,
        )
    }

    fn routed_tensor_output_width(&self) -> Result<Option<usize>, Self::Error> {
        usize::try_from(self.args.text_config.vocab_size)
            .map(Some)
            .map_err(|_| Error::backend("Inkling physical vocabulary width exceeds usize"))
    }

    fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        forward.target_hidden()
    }

    fn prediction_target_placeholder_shape(
        &self,
        forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<i32>>, Self::Error> {
        Ok(Some(vec![
            forward.tokens().dim(0),
            forward.tokens().dim(1),
            self.args.text_config.hidden_size,
        ]))
    }

    fn begin_composite_forward<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let prepared = prepare_input(input, context)?;
        prepared.with_model_input(|input| {
            <Self as LayeredArchitecture<B, S>>::begin_forward(self, input, state, context)
        })
    }

    fn begin_composite_forward_parallel<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        prepare_input(input, context)?.with_model_input(|input| {
            <Self as ParallelLayeredArchitecture<B, S>>::begin_forward_parallel(
                self, input, state, parallel, context,
            )
        })
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    fn build_execution_graph(
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<ExecutionGraph, Error> {
        let destination = crate::composite_execution::graph::Destination(context);
        destination.controls::<ExecutionGraph>()?;
        let mut groups = destination.vector(3)?;
        groups.push(destination.group(VISION_EXECUTION_GROUP, &[])?);
        groups.push(destination.group(AUDIO_EXECUTION_GROUP, &[])?);
        groups.push(destination.group(
            TEXT_EXECUTION_GROUP,
            &[VISION_EXECUTION_GROUP, AUDIO_EXECUTION_GROUP],
        )?);
        destination.finish(groups, TEXT_EXECUTION_GROUP)
    }

    fn hmlp_cut_shape(&self, patches: i32, end: usize) -> Result<[i32; 5], Error> {
        let vision = self
            .args
            .vision_config
            .as_ref()
            .ok_or_else(|| Error::backend("Inkling continuation has no vision tower"))?;
        if end >= vision.layer_specs().len() {
            return Err(Error::backend(
                "Inkling continuation must precede the final hMLP unit",
            ));
        }
        vision.folded_shape(patches, end).map_err(Error::backend)
    }
}

/// Typed text-decoder input for one pipeline partition.
pub enum TextPartitionInput<'a, T> {
    /// Token identities owned by the first decoder partition.
    Tokens(&'a T),
    /// Decoder activation received from an upstream partition.
    Hidden(T),
}

/// Completed architecture-owned embedded prediction partition.
pub struct PartitionMtpOutput<T> {
    /// Complete vocabulary logits.
    pub logits: T,
    /// Prediction-space hidden state.
    pub hidden: T,
    /// Token identity associated with the prediction.
    pub tokens: T,
}

enum PreparedPart<T> {
    Text { tokens: T, embeddings: T },
    Image { tokens: T },
    Audio { tokens: T },
    Projected { tokens: T, embeddings: T },
}

/// A streamable hMLP stage or text decoder layer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Folded hMLP image stage.
    Vision(VisionLayer<B>),
    /// Stateful text decoder layer.
    Text(DecoderLayer<B>),
}

impl<B, S> RoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
{
    fn routed_unit_observations(&self) -> bool {
        true
    }
    fn routed_sparse_observations(&self) -> bool {
        true
    }

    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let output = match (group, unit) {
            (2, Unit::Text(unit)) => self.forward_text_unit_with_provider(
                index, unit, hidden, state, pass, provider, context,
            ),
            (_, unit) => <Self as LayeredArchitecture<B, S>>::forward_unit(
                self, group, index, unit, hidden, state, forward, context,
            ),
        }?;
        if group == 2 && index + 1 == self.args.text_config.num_hidden_layers as usize {
            forward.capture_target_hidden(output.clone());
        }
        Ok(output)
    }

    fn forward_unit_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let output = match (group, unit) {
            (2, Unit::Text(unit)) => {
                let path = format!("model.layers.{index}");
                let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
                let observed_embedding = if index == 0 {
                    Some(
                        crate::decoder::ComponentInstrumentation::new("readout", &mut observer)
                            .apply("embedding", hidden.clone())?,
                    )
                } else {
                    None
                };
                let hidden = observed_embedding.as_ref().unwrap_or(hidden);
                unit.forward_with_provider_instrumented(
                    hidden,
                    Some(
                        state
                            .layer(self.local_state_ordinal(index)?)
                            .map_err(Error::backend)?,
                    ),
                    pass,
                    provider,
                    context,
                    &mut crate::decoder::ComponentInstrumentation::new(&path, &mut observer),
                )
            }
            (_, unit) => <Self as LayeredArchitecture<B, S>>::forward_unit(
                self, group, index, unit, hidden, state, forward, context,
            ),
        }?;
        if group == 2 && index + 1 == self.args.text_config.num_hidden_layers as usize {
            forward.capture_target_hidden(output.clone());
        }
        Ok(output)
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        true
    }
    fn parallel_routed_sparse_observations(&self) -> bool {
        true
    }

    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
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
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let output = match (group, unit) {
            (2, Unit::Text(unit)) => self.forward_text_unit_parallel_with_provider(
                index, unit, hidden, state, pass, provider, parallel, context,
            ),
            (_, unit) => <Self as ParallelLayeredArchitecture<B, S>>::forward_unit_parallel(
                self, group, index, unit, hidden, state, forward, parallel, context,
            ),
        }?;
        if group == 2 && index + 1 == self.args.text_config.num_hidden_layers as usize {
            forward.capture_target_hidden(output.clone());
        }
        Ok(output)
    }

    fn forward_unit_parallel_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let output = match (group, unit) {
            (2, Unit::Text(unit)) => {
                let path = format!("model.layers.{index}");
                let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
                let observed_embedding = if index == 0 {
                    Some(
                        crate::decoder::ComponentInstrumentation::new("readout", &mut observer)
                            .apply("embedding", hidden.clone())?,
                    )
                } else {
                    None
                };
                let hidden = observed_embedding.as_ref().unwrap_or(hidden);
                unit.forward_parallel_with_provider_instrumented(
                    hidden,
                    Some(
                        state
                            .layer(self.local_state_ordinal(index)?)
                            .map_err(Error::backend)?,
                    ),
                    pass,
                    provider,
                    parallel,
                    context,
                    &mut crate::decoder::ComponentInstrumentation::new(&path, &mut observer),
                )
            }
            (_, unit) => <Self as ParallelLayeredArchitecture<B, S>>::forward_unit_parallel(
                self, group, index, unit, hidden, state, forward, parallel, context,
            ),
        }?;
        if group == 2 && index + 1 == self.args.text_config.num_hidden_layers as usize {
            forward.capture_target_hidden(output.clone());
        }
        Ok(output)
    }
}

/// Transient values retained across component groups.
pub struct ForwardContext<T> {
    parts: Vec<PreparedPart<T>>,
    tokens: T,
    audio_input: Option<T>,
    audio_valid_frames: Option<i32>,
    audio_output: Option<T>,
    vision_output: Option<T>,
    has_vision: bool,
    target_hidden: Option<T>,
    pending_media: Option<PreparedInput<T>>,
    media_span: bool,
}

impl<T> ForwardContext<T> {
    /// Whether this request carries raw vision input.
    pub const fn has_vision_input(&self) -> bool {
        self.has_vision
    }

    /// Whether this request carries raw audio input.
    pub const fn has_audio_input(&self) -> bool {
        self.audio_input.is_some()
    }

    /// Installs the architecture-declared vision projector output received from another owner.
    pub fn replace_vision_output(&mut self, output: T) {
        self.vision_output = Some(output);
    }

    /// Installs the architecture-declared audio projector output received from another owner.
    pub fn replace_audio_output(&mut self, output: T) {
        self.audio_output = Some(output);
    }
}

/// Inkling architecture shared by resident, layerwise, and streamed runtimes.
pub struct LayeredModel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    args: crate::replicated_text::SharedCompositeConfig<ModelArgs>,
    construction: RetainedModelSource,
    static_modules: StaticModules<B>,
    parallel_geometry: Option<Arc<LocalGeometry>>,
    partition_state_offset: usize,
    expert_realization: Option<Arc<crate::ExpertRealizationPlan<super::ExpertBankRealization>>>,
    execution_graph: ExecutionGraph,
}

/// Architecture-owned target, embedded-prediction, and composite state geometry.
#[derive(Debug, Clone)]
pub struct InklingStateLayouts {
    target: StateLayout,
    prediction: Option<PartitionState>,
    composite: StateLayout,
}

impl InklingStateLayouts {
    fn new(target: StateLayout, prediction: Option<PartitionState>) -> Result<Self, Error> {
        if prediction
            .as_ref()
            .is_some_and(|state| state.global_layer_offset() != target.len())
        {
            return Err(Error::backend(
                "Inkling prediction state does not follow target state",
            ));
        }
        let composite =
            composite_state_layout(&target, prediction.as_ref().map(PartitionState::layout))
                .map_err(Error::backend)?;
        Ok(Self {
            target,
            prediction,
            composite,
        })
    }

    /// Returns the target decoder layout for this realization.
    pub const fn target(&self) -> &StateLayout {
        &self.target
    }

    /// Returns embedded-prediction state and its architecture-global placement.
    pub const fn prediction(&self) -> Option<&PartitionState> {
        self.prediction.as_ref()
    }

    /// Returns the complete target-plus-prediction persistence layout.
    pub const fn composite(&self) -> &StateLayout {
        &self.composite
    }

    /// Selects and assembles the exact state owned by one pipeline partition.
    pub fn partition(
        &self,
        target: &PartitionState,
        include_prediction: bool,
    ) -> Result<PartitionState, Error> {
        let range = target.global_layers();
        let expected = self.target.slice(range.clone()).map_err(Error::backend)?;
        if target.layout() != &expected {
            return Err(Error::backend(
                "Inkling partition target state differs from architecture geometry",
            ));
        }
        let prediction = include_prediction
            .then_some(self.prediction.as_ref())
            .flatten();
        if let Some(prediction) = prediction {
            if prediction.global_layer_offset() != range.end {
                return Err(Error::backend(
                    "Inkling prediction state is not contiguous with the target partition",
                ));
            }
        }
        let layout =
            composite_state_layout(target.layout(), prediction.map(PartitionState::layout))
                .map_err(Error::backend)?;
        PartitionState::new(layout, target.global_layer_offset()).map_err(Error::backend)
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::ArchitectureParameters<B> for LayeredModel<B>
{
    type DefinitionError = Error;

    fn state_layout(
        &self,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<StateLayout, Self::DefinitionError> {
        match context {
            Some(context) => self
                .checked_units(context)?
                .layouts
                .composite()
                .clone_workspace(context),
            None => Ok(self.state_layouts()?.composite().clone()),
        }
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: PromptCacheTopology,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<ModelStateIdentity, Self::DefinitionError> {
        match context {
            Some(context) => {
                let source = self.checked_units(context)?;
                state_identity_with_count(
                    &self.args,
                    state.layout(),
                    state.global_layer_offset(),
                    topology,
                    source.global_state_layers,
                    crate::decoder::identity::Metadata::new(Some(context)),
                )
            }
            None => state_identity(
                &self.args,
                state.layout(),
                state.global_layer_offset(),
                topology,
            ),
        }
    }

    fn parameter_description(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<std::borrow::Cow<'_, ArchitectureParameterDescription>, Self::DefinitionError> {
        crate::decoder::identity::Metadata::new(B::construction_metadata(context)).controls::<(
            std::borrow::Cow<'_, ArchitectureParameterDescription>,
            &Self,
        )>()?;
        if let Some(source) = self.construction.units() {
            source.validate(
                self,
                crate::decoder::identity::Metadata::new(B::construction_metadata(context)),
            )?;
            return Ok(std::borrow::Cow::Borrowed(&source.description));
        }
        (|| {
            if B::construction_metadata(context)
                .is_some_and(|metadata| metadata.uses_checked_metadata())
            {
                return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
            }
            self.parameter_description_impl(context)
        })()
        .map(std::borrow::Cow::Owned)
    }

    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        super::static_safetensors_recipes(&self.args, source)
    }

    fn retained_static_value_slot_bound(&self) -> Option<usize> {
        eredu_nn::Parameterized::retained_value_slot_bound(&self.static_modules)
    }

    fn visit_retained_static_values(&self, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
        eredu_nn::Parameterized::visit_retained_values(&self.static_modules, visitor)
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        visitor.visit("embedding", &self.static_modules.embeddings)?;
        visitor.visit("embedding_norm", &self.static_modules.embedding_norm)?;
        visitor.visit("norm", &self.static_modules.final_norm)?;
        visitor.visit("output", &self.static_modules.output)?;
        if let Some(module) = &self.static_modules.audio {
            visitor.visit("audio", module)?;
        }
        if let Some(module) = &self.static_modules.vision {
            visitor.visit("vision", module)?;
        }
        if let Some(module) = &self.static_modules.mtp {
            visitor.visit(MTP_STATIC_ROLE, module)?;
        }
        Ok(())
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        visitor.visit_mut("embedding", &mut self.static_modules.embeddings)?;
        visitor.visit_mut("embedding_norm", &mut self.static_modules.embedding_norm)?;
        visitor.visit_mut("norm", &mut self.static_modules.final_norm)?;
        visitor.visit_mut("output", &mut self.static_modules.output)?;
        if let Some(module) = &mut self.static_modules.audio {
            visitor.visit_mut("audio", module)?;
        }
        if let Some(module) = &mut self.static_modules.vision {
            visitor.visit_mut("vision", module)?;
        }
        if let Some(module) = &mut self.static_modules.mtp {
            visitor.visit_mut(MTP_STATIC_ROLE, module)?;
        }
        Ok(())
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> LayeredModel<B> {
    /// Enters or resumes a routed text partition through the family embedding
    /// boundary.
    pub fn begin_routed_text_partition(
        &mut self,
        input: TextPartitionInput<'_, B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        let hidden = match input {
            TextPartitionInput::Tokens(tokens) => match parallel {
                Some(parallel) => self.mtp_token_embeddings_parallel(tokens, parallel, context)?,
                None => self.mtp_token_embeddings(tokens, context)?,
            },
            TextPartitionInput::Hidden(hidden) => hidden,
        };
        Ok(self.resume_partition_text(hidden))
    }

    /// Resumes an already assembled text partition in the canonical layered context.
    pub fn resume_partition_text(
        &self,
        hidden: B::Tensor,
    ) -> LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>> {
        LayeredForwardState {
            context: ForwardContext {
                tokens: hidden.clone(),
                parts: Vec::new(),
                audio_input: None,
                audio_valid_frames: None,
                audio_output: None,
                vision_output: None,
                has_vision: false,
                target_hidden: None,
                pending_media: None,
                media_span: false,
            },
            hidden,
        }
    }

    /// Builds unloaded pinned modules from normalized family configuration.
    pub fn new(args: ModelArgs, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        Self::require_construction(context)?;
        let source = RetainedModelSource::prepare::<B>(args, context)?;
        Self::from_source(source, None, context)
    }

    fn canonical_group_transport(group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        match group {
            0 => vision_group_transport(),
            1 => eredu_runtime::ArchitectureGroupTransport {
                placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
                kind: eredu_runtime::ArchitectureGroupKind::AudioEncoder,
                first_owner_static_roles: vec!["audio".into()],
                last_owner_static_roles: Vec::new(),
                merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
                parallel_subgroup: None,
                request_optional: true,
            },
            _ => eredu_runtime::ArchitectureGroupTransport {
                placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
                kind: eredu_runtime::ArchitectureGroupKind::Decoder,
                first_owner_static_roles: vec!["embedding".into(), "embedding_norm".into()],
                last_owner_static_roles: vec![
                    "norm".into(),
                    "output".into(),
                    MTP_STATIC_ROLE.into(),
                ],
                merge_destination: eredu_runtime::ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::Decoder),
                request_optional: false,
            },
        }
    }

    fn require_construction(context: &<B::Tensor as Tensor>::Context) -> Result<(), Error> {
        crate::operator_requirements::require_with_metadata::<B>(
            "Inkling",
            crate::operator_requirements::INKLING,
            crate::decoder::identity::Metadata::new(B::construction_metadata(context)),
        )
    }

    pub(crate) fn construction_source(&self) -> &RetainedModelSource {
        &self.construction
    }

    pub(crate) fn new_with_source(
        source: RetainedModelSource,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::require_construction(context)?;
        Self::from_source(source, None, context)
    }

    pub(crate) fn new_parallel_with_source(
        source: RetainedModelSource,
        geometry: Arc<LocalGeometry>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::require_construction(context)?;
        geometry.validate_for(&source).map_err(Error::backend)?;
        Self::from_source(source, Some(geometry), context)
    }

    fn from_source(
        source: RetainedModelSource,
        geometry: Option<Arc<LocalGeometry>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::decoder::ModuleMetadata::new::<B>(context).controls::<(
            RetainedModelSource,
            Option<Arc<LocalGeometry>>,
            Self,
        )>()?;
        let static_modules = source.instantiate::<B>(geometry.as_deref(), context)?;
        let args = source.args().clone();
        Ok(Self {
            args,
            execution_graph: Self::build_execution_graph(B::construction_metadata(context))?,
            construction: source,
            static_modules,
            parallel_geometry: geometry,
            partition_state_offset: 0,
            expert_realization: None,
        })
    }

    /// Builds one planner-derived rank-local realization of the canonical graph.
    pub fn new_parallel(
        args: ModelArgs,
        geometry: Arc<LocalGeometry>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::require_construction(context)?;
        geometry.validate_for(&args).map_err(Error::backend)?;
        let source = RetainedModelSource::prepare::<B>(args, context)?;
        Self::from_source(source, Some(geometry), context)
    }

    /// Retains the architecture-global ordinal of this pipeline partition's first text state.
    pub(crate) fn with_partition_state_offset(mut self, offset: usize) -> Result<Self, Error> {
        if offset >= self.args.text_config.num_hidden_layers as usize {
            return Err(Error::backend(
                "Inkling partition state offset is outside the target decoder",
            ));
        }
        if self
            .args
            .mtp_config
            .as_ref()
            .is_some_and(|prediction| prediction.num_nextn_predict_layers > 0)
        {
            return Err(Error::backend(
                "partitioned Inkling cannot attach embedded-prediction state",
            ));
        }
        self.partition_state_offset = offset;
        Ok(self)
    }

    fn local_state_ordinal(&self, global: usize) -> Result<usize, Error> {
        global
            .checked_sub(self.partition_state_offset)
            .ok_or_else(|| Error::backend("Inkling unit precedes the partition state offset"))
    }

    /// Binds exact routed and replicated-shared compact banks to local text units.
    pub fn with_expert_realization(
        mut self,
        realization: crate::ExpertRealizationPlan<super::ExpertBankRealization>,
    ) -> Result<Self, Error> {
        if !self.args.text_config.has_sparse_moe_layers() {
            return Err(Error::backend(
                "dense Inkling cannot bind an expert realization",
            ));
        }
        if !self
            .construction
            .units()
            .is_some_and(|source| source.matches_realization(&realization))
        {
            self.construction.clear_units();
        }
        self.expert_realization = Some(Arc::new(realization));
        Ok(self)
    }

    /// Returns normalized family configuration.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Describes pinned media/text modules and every graph unit with explicit
    /// architecture ownership.
    fn parameter_description_impl(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let graph = match B::construction_metadata(context) {
            Some(context) => self.execution_graph.clone_with_metadata(context)?,
            None => self.execution_graph.clone(),
        };
        let counts = [
            self.args
                .vision_config
                .as_ref()
                .map_or(0, |vision| vision.num_hidden_layers as usize),
            0,
            self.args.text_config.num_hidden_layers as usize,
        ];
        let layout = ExecutionUnitLayout::new(&graph, counts).map_err(Error::backend)?;
        let static_groups = static_parameter_groups(&self.args).map_err(Error::backend)?;
        let mut static_roles = vec!["embedding", "embedding_norm", "norm", "output"];
        if self.args.audio_config.is_some() {
            static_roles.extend(["audio", "audio"]);
        }
        if self.args.vision_config.is_some() {
            static_roles.extend(std::iter::repeat_n(
                "vision",
                static_groups.len() - static_roles.len(),
            ));
        }
        debug_assert_eq!(static_groups.len(), static_roles.len());
        let mtp_groups = mtp_parameter_groups(&self.args).map_err(Error::backend)?;
        let mut expected = static_groups
            .iter()
            .chain(&mtp_groups)
            .cloned()
            .collect::<Vec<_>>();
        let mut owned = static_groups
            .into_iter()
            .zip(static_roles)
            .map(|(group, role)| {
                let owner = if role == "embedding" {
                    ParameterGroupOwner::static_any_of(["embedding", MTP_STATIC_ROLE])
                } else {
                    ParameterGroupOwner::static_role(role)
                };
                OwnedParameterGroupSpec::new(owner, group)
            })
            .chain(mtp_groups.into_iter().map(|group| {
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role(MTP_STATIC_ROLE),
                    group,
                )
            }))
            .collect::<Vec<_>>();
        for (group_index, &count) in counts.iter().enumerate() {
            let owner_group = layout
                .group_id(group_index)
                .expect("Inkling layout group")
                .clone();
            for index in 0..count {
                let groups = if group_index == 0 {
                    let vision = self
                        .args
                        .vision_config
                        .as_ref()
                        .expect("vision group configured");
                    let layer =
                        VisionLayer::<B>::new(vision, index, vision.layer_specs()[index], context)?;
                    vision_layer_parameter_groups::<B>(&layer, index)
                } else {
                    layer_parameter_groups(&self.args, index)
                }
                .map_err(Error::backend)?;
                expected.extend(groups.iter().cloned());
                owned.extend(groups.into_iter().map(|group| {
                    OwnedParameterGroupSpec::new(
                        ParameterGroupOwner::execution_unit(owner_group.clone(), index),
                        group,
                    )
                }));
            }
        }
        ArchitectureParameterDescription::new(&graph, &layout, expected, owned)
            .map_err(Error::backend)
    }

    /// Returns the transient target state required to enter this realization.
    ///
    /// Ordinary realizations use the global target geometry. Parallel
    /// realizations use the rank-local target geometry. Embedded-prediction
    /// state is persisted separately and is never part of decoder ingress.
    pub fn ingress_state_layout(&self) -> Result<StateLayout, Error> {
        match &self.parallel_geometry {
            Some(geometry) => Ok(geometry.state_layout().clone()),
            None => state_layout(&self.args).map_err(Error::backend),
        }
    }

    /// Returns every state layout consumed by execution and persistence.
    pub fn state_layouts(&self) -> Result<InklingStateLayouts, Error> {
        let target = self.ingress_state_layout()?;
        let prediction = match &self.parallel_geometry {
            Some(geometry) => geometry.prediction_state().cloned(),
            None => mtp_state_layout(&self.args)
                .map_err(Error::backend)?
                .map(|layout| PartitionState::new(layout, target.len()))
                .transpose()
                .map_err(Error::backend)?,
        };
        InklingStateLayouts::new(target, prediction)
    }

    fn accepts_execution_state(&self, layout: &StateLayout) -> Result<bool, Error> {
        let complete = self.ingress_state_layout()?;
        if self.partition_state_offset != 0 || layout.len() < complete.len() {
            let end = self
                .partition_state_offset
                .checked_add(layout.len())
                .ok_or_else(|| Error::backend("Inkling partition state interval overflow"))?;
            let expected = complete
                .slice(self.partition_state_offset..end)
                .map_err(Error::backend)?;
            let expected = composite_state_layout(&expected, None).map_err(Error::backend)?;
            return Ok(layout == &expected);
        }
        let layouts = self.state_layouts()?;
        Ok(layout == layouts.target() || layout == layouts.composite())
    }

    /// Shares validated planner-derived geometry with backend residency policy.
    pub fn shared_parallel_geometry(&self) -> Option<Arc<LocalGeometry>> {
        self.parallel_geometry.clone()
    }

    /// Starts a text-only pass from a rank-local vocabulary embedding.
    pub fn begin_parallel_text<S: LayerRuntimeState<B>>(
        &mut self,
        tokens: &B::Tensor,
        embeddings: B::Tensor,
        state: &S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
    {
        if !self.accepts_execution_state(state.layout())? {
            return Err(Error::backend("Inkling rank-local state layout mismatch"));
        }
        let hidden = self
            .static_modules
            .embedding_norm
            .forward(&embeddings, context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                parts: vec![PreparedPart::Text {
                    tokens: tokens.clone(),
                    embeddings,
                }],
                tokens: tokens.clone(),
                audio_input: None,
                audio_valid_frames: None,
                audio_output: None,
                vision_output: None,
                has_vision: false,
                target_hidden: None,
                pending_media: None,
                media_span: false,
            },
        })
    }

    /// Starts a multimodal pass from rank-local text embeddings while keeping
    /// hMLP, dMel, normalization, and ordered assembly in the neutral model.
    pub fn begin_parallel_input<S: LayerRuntimeState<B>>(
        &mut self,
        input: ModelInput<'_, B::Tensor>,
        text_embeddings: &[B::Tensor],
        vision_layers: &mut [VisionLayer<B>],
        state: &S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
    {
        if !self.accepts_execution_state(state.layout())? {
            return Err(Error::backend("Inkling rank-local state layout mismatch"));
        }
        let mut next_embedding = text_embeddings.iter();
        let parts = input
            .parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: next_embedding
                        .next()
                        .ok_or_else(|| Error::backend("missing Inkling rank-local text embedding"))?
                        .clone(),
                }),
                DecoderInputPart::Image(tokens) => Ok(PreparedPart::Image {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Audio(tokens) => Ok(PreparedPart::Audio {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Projected { tokens, embeddings } => Ok(PreparedPart::Projected {
                    tokens: (*tokens).clone(),
                    embeddings: (*embeddings).clone(),
                }),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        if next_embedding.next().is_some() {
            return Err(Error::backend("unused Inkling rank-local text embedding"));
        }
        let tokens = ordered_tokens(&parts, context)?;
        let audio = match input.audio {
            Some(audio) => Some(
                self.static_modules
                    .audio
                    .as_mut()
                    .ok_or_else(|| Error::backend("Inkling has no audio tower"))?
                    .forward(audio, context)?,
            ),
            None => None,
        };
        let vision = match input.vision_patches {
            Some(patches) => {
                if patches.shape().len() != 5 || patches.shape()[1..] != [2, 40, 40, 3] {
                    return Err(Error::backend("invalid Inkling hMLP patch geometry"));
                }
                let expected = self
                    .args
                    .vision_config
                    .as_ref()
                    .map_or(0, |vision| vision.num_hidden_layers as usize);
                if vision_layers.len() != expected {
                    return Err(Error::backend("Inkling hMLP layer count mismatch"));
                }
                let mut hidden = patches.clone();
                for layer in vision_layers {
                    hidden = layer.forward(&hidden, context)?;
                }
                Some(
                    self.static_modules
                        .vision
                        .as_mut()
                        .ok_or_else(|| Error::backend("Inkling has no vision tower"))?
                        .finish(&hidden, context)?,
                )
            }
            None => None,
        };
        let hidden = self.assemble(&parts, vision.as_ref(), audio.as_ref(), context)?;
        let has_vision = vision.is_some();
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                parts,
                tokens,
                audio_input: None,
                audio_valid_frames: None,
                audio_output: audio,
                vision_output: vision,
                has_vision,
                target_hidden: None,
                pending_media: None,
                media_span: false,
            },
        })
    }

    /// Executes one ordinary text layer with rank-local projections.
    pub fn forward_text_unit_parallel<S: LayerRuntimeState<B>>(
        &mut self,
        index: usize,
        unit: &mut DecoderLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        unit.forward_parallel(
            hidden,
            Some(
                state
                    .layer(self.local_state_ordinal(index)?)
                    .map_err(Error::backend)?,
            ),
            parallel,
            context,
        )
    }

    /// Executes one TP text layer while runtime owns routed expert residency.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_text_unit_parallel_with_provider<S, P>(
        &mut self,
        index: usize,
        unit: &mut DecoderLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        unit.forward_parallel_with_provider(
            hidden,
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

    /// Applies replicated final normalization and muP scaling before the
    /// vocabulary-parallel output projection.
    pub fn final_parallel_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules
            .final_norm
            .forward(hidden, context)?
            .multiply_scalar(
                self.args.text_config.logits_mup_width_multiplier.recip(),
                context,
            )
    }

    /// Returns the checkpoint-embedded prediction depth count.
    pub fn mtp_len(&self) -> usize {
        self.static_modules.mtp.as_ref().map_or(0, MtpModel::len)
    }

    /// Returns one embedded prediction depth's attention policy.
    pub fn mtp_policy(&self, depth: usize) -> Option<eredu_core::AttentionPolicy> {
        self.static_modules
            .mtp
            .as_ref()
            .and_then(|mtp| mtp.policy(depth))
    }

    /// Embeds predictor token identities through the ordinary shared ingress.
    pub fn mtp_token_embeddings(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let embeddings = self.static_modules.embeddings.forward(tokens, context)?;
        self.static_modules
            .embedding_norm
            .forward(&embeddings, context)
    }

    /// Embeds predictor token identities through the rank-local shared ingress.
    pub fn mtp_token_embeddings_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if self.parallel_geometry.is_none() {
            return Err(Error::backend(
                "Inkling model was not built with local geometry",
            ));
        }
        let embeddings = B::vocabulary_parallel_lookup(
            &mut self.static_modules.embeddings,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        self.static_modules
            .embedding_norm
            .forward(&embeddings, context)
    }

    /// Applies the target muP scale before the sharded MTP output projection.
    pub fn final_mtp_parallel_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        hidden.multiply_scalar(
            self.args.text_config.logits_mup_width_multiplier.recip(),
            context,
        )
    }

    /// Executes one checkpoint-embedded prediction depth.
    pub fn forward_mtp_step<S>(
        &mut self,
        hidden: &B::Tensor,
        embeddings: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut [S],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MtpOutput<B::Tensor>, Error>
    where
        S: eredu_nn::AttentionCache<B::Tensor> + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    {
        self.static_modules
            .mtp
            .as_mut()
            .ok_or_else(|| Error::backend("Inkling checkpoint has no MTP predictor"))?
            .forward_step(hidden, embeddings, tokens, depth, state, context)
    }

    /// Executes one complete embedded-prediction partition, including shared
    /// embedding, predictor recurrence, and vocabulary projection.
    pub fn forward_partition_mtp<S>(
        &mut self,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        depth: usize,
        state: &mut [S],
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PartitionMtpOutput<B::Tensor>, Error>
    where
        S: eredu_nn::AttentionCache<B::Tensor> + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    {
        let embeddings = match parallel {
            Some(parallel) => self.mtp_token_embeddings_parallel(tokens, parallel, context)?,
            None => self.mtp_token_embeddings(tokens, context)?,
        };
        let output = self.forward_mtp_step(hidden, &embeddings, tokens, depth, state, context)?;
        let logits = match parallel {
            Some(parallel) => {
                self.project_mtp_logits_parallel(&output.hidden, parallel, context)?
            }
            None => self.project_mtp_logits(&output.hidden, context)?,
        };
        Ok(PartitionMtpOutput {
            logits,
            hidden: output.hidden,
            tokens: output.tokens,
        })
    }

    /// Applies the target-owned muP-scaled vocabulary projection to MTP hidden state.
    pub fn project_mtp_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.project_mtp_logits_instrumented(
            hidden,
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    /// Applies the rank-local output projection to one MTP hidden state.
    pub fn project_mtp_logits_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.project_mtp_logits_instrumented(
            hidden,
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    /// Observes the actual muP-scaled MTP vocabulary input and effective scores.
    pub fn project_mtp_logits_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        if parallel.is_some() && self.parallel_geometry.is_none() {
            return Err(Error::backend(
                "Inkling model was not built with local geometry",
            ));
        }
        let hidden = self.final_mtp_parallel_hidden(hidden, context)?;
        instrumentation.observe("scaled", &hidden)?;
        let logits = match parallel {
            Some(parallel) => instrumentation.project_vocabulary::<B>(
                "projection_input",
                &mut self.static_modules.output,
                &hidden,
                parallel,
                context,
            )?,
            None => instrumentation.project::<B>(
                "projection_input",
                &mut self.static_modules.output,
                &hidden,
                None,
                context,
            )?,
        };
        instrumentation.apply("linear", self.trim_logits(logits, context)?)
    }

    fn target_logits_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        if parallel.is_some() && self.parallel_geometry.is_none() {
            return Err(Error::backend(
                "Inkling model was not built with local geometry",
            ));
        }
        let hidden = instrumentation.apply("residual", hidden.clone())?;
        let normalized = self.static_modules.final_norm.forward(&hidden, context)?;
        let normalized = instrumentation.apply("normalized", normalized)?;
        let scaled = normalized.multiply_scalar(
            self.args.text_config.logits_mup_width_multiplier.recip(),
            context,
        )?;
        instrumentation.observe("scaled", &scaled)?;
        let logits = match parallel {
            Some(parallel) => instrumentation.project_vocabulary::<B>(
                "projection_input",
                &mut self.static_modules.output,
                &scaled,
                parallel,
                context,
            )?,
            None => instrumentation.project::<B>(
                "projection_input",
                &mut self.static_modules.output,
                &scaled,
                None,
                context,
            )?,
        };
        instrumentation.apply("linear", self.trim_logits(logits, context)?)
    }

    /// Applies final target normalization and the rank-local output projection.
    pub fn project_target_logits_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.target_logits_instrumented(
            hidden,
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    /// Executes one text unit while delegating routed and shared banks to runtime residency.
    pub fn forward_text_unit_with_provider<S, P>(
        &mut self,
        index: usize,
        unit: &mut DecoderLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        unit.forward_with_provider(
            hidden,
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
            return Err(Error::backend("Inkling input has no ordered parts"));
        }
        parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: self.static_modules.embeddings.forward(tokens, context)?,
                }),
                DecoderInputPart::Image(tokens) => Ok(PreparedPart::Image {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Audio(tokens) => Ok(PreparedPart::Audio {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Projected { tokens, embeddings } => Ok(PreparedPart::Projected {
                    tokens: (*tokens).clone(),
                    embeddings: (*embeddings).clone(),
                }),
            })
            .collect()
    }

    fn prepare_parts_parallel(
        &mut self,
        parts: &[DecoderInputPart<'_, B::Tensor>],
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<PreparedPart<B::Tensor>>, Error> {
        if parts.is_empty() {
            return Err(Error::backend("Inkling input has no ordered parts"));
        }
        parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: B::vocabulary_parallel_lookup(
                        &mut self.static_modules.embeddings,
                        tokens,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?,
                }),
                DecoderInputPart::Image(tokens) => Ok(PreparedPart::Image {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Audio(tokens) => Ok(PreparedPart::Audio {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Projected { tokens, embeddings } => Ok(PreparedPart::Projected {
                    tokens: (*tokens).clone(),
                    embeddings: (*embeddings).clone(),
                }),
            })
            .collect()
    }

    fn trim_logits(
        &self,
        logits: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let vocabulary = self
            .args
            .text_config
            .unpadded_vocab_size
            .unwrap_or(self.args.text_config.vocab_size);
        if vocabulary == self.args.text_config.vocab_size {
            return Ok(logits);
        }
        let mut indexes = vec![Index::Full; logits.shape().len()];
        *indexes.last_mut().expect("logits have vocabulary axis") = Index::Range(0, vocabulary);
        logits.index(&indexes, context)
    }

    fn assemble(
        &mut self,
        parts: &[PreparedPart<B::Tensor>],
        vision: Option<&B::Tensor>,
        audio: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.assemble_with_metadata(
            parts,
            vision,
            audio,
            context,
            crate::decoder::ModuleMetadata::new::<B>(context),
        )
    }
    fn assemble_with_metadata(
        &mut self,
        parts: &[PreparedPart<B::Tensor>],
        vision: Option<&B::Tensor>,
        audio: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: crate::decoder::ModuleMetadata<'_>,
    ) -> Result<B::Tensor, Error> {
        metadata.controls::<(B::Tensor, Vec<B::Tensor>, i32, i32, i32, i32, usize)>()?;
        let image_tokens =
            part_token_count(parts, |part| matches!(part, PreparedPart::Image { .. }));
        let audio_tokens =
            part_token_count(parts, |part| matches!(part, PreparedPart::Audio { .. }));
        validate_component_with_metadata(
            "image",
            vision,
            image_tokens,
            self.args.text_config.hidden_size,
            metadata,
        )?;
        validate_component_with_metadata(
            "audio",
            audio,
            audio_tokens,
            self.args.text_config.hidden_size,
            metadata,
        )?;
        let mut image_offset = 0;
        let mut audio_offset = 0;
        let mut embeddings = metadata.vector(parts.len())?;
        for part in parts {
            match part {
                PreparedPart::Text {
                    embeddings: value, ..
                } => embeddings.push(value.clone()),
                PreparedPart::Image { tokens } => {
                    let length = tokens.dim(1);
                    embeddings.push(slice_component(
                        vision.expect("validated image component"),
                        image_offset,
                        length,
                        context,
                    )?);
                    image_offset += length;
                }
                PreparedPart::Audio { tokens } => {
                    let length = tokens.dim(1);
                    embeddings.push(slice_component(
                        audio.expect("validated audio component"),
                        audio_offset,
                        length,
                        context,
                    )?);
                    audio_offset += length;
                }
                PreparedPart::Projected {
                    embeddings: value, ..
                } => embeddings.push(value.clone()),
            }
        }
        let batch = parts
            .first()
            .map(prepared_part_tokens)
            .ok_or_else(|| metadata.error(format_args!("Inkling input has no ordered parts")))?
            .dim(0);
        for (index, (part, embeddings)) in parts.iter().zip(&embeddings).enumerate() {
            let tokens = prepared_part_tokens(part);
            if tokens.shape().len() != 2
                || embeddings.shape().len() != 3
                || tokens.dim(0) != batch
                || embeddings.dim(0) != batch
                || tokens.dim(1) != embeddings.dim(1)
                || embeddings.dim(2) != self.args.text_config.hidden_size
            {
                return Err(metadata.error(format_args!(
                    "Inkling ordered input part {index} has incompatible token/embedding shapes {:?} and {:?}",
                    tokens.shape(),
                    embeddings.shape()
                )));
            }
        }
        let hidden = B::Tensor::concatenate(&embeddings, 1, context)?;
        self.static_modules.embedding_norm.forward(&hidden, context)
    }
}

fn prepared_part_tokens<T>(part: &PreparedPart<T>) -> &T {
    match part {
        PreparedPart::Text { tokens, .. }
        | PreparedPart::Image { tokens }
        | PreparedPart::Audio { tokens }
        | PreparedPart::Projected { tokens, .. } => tokens,
    }
}

impl<B, S> LayeredArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
{
    fn media_prefill_observation_declarations(
        &self,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Self::Error> {
        let metadata = crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(
            &Self,
            Option<&eredu_nn::workspace::WorkspaceContext>,
            usize,
            usize,
            String,
            Vec<eredu_runtime::layered::PrefillObservationDeclaration>,
            std::ops::Range<usize>,
            Option<eredu_runtime::RoutedObservationPoints>,
            Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Error>,
        )>()?;

        // The validated ingress preserves media offsets and causal hMLP/attention state; encoder rows are not declared.
        let units =
            <Self as LayeredArchitecture<B, S>>::group_unit_count(self, 2, metadata_context)?;
        let mut declarations = crate::decoder::media_prefill_observation_declarations(
            (0..units).map(|index| {
                <Self as LayeredArchitecture<B, S>>::unit_path(self, 2, index, metadata_context)
            }),
            metadata_context,
        )?;
        // The retained-media ingress changes decoder inputs and positions, but
        // executes the same row-local routed banks as the ordinary target.
        // Reuse those exact architecture declarations; encoder hooks remain
        // outside this decoder contract.
        let ordinary = <Self as LayeredArchitecture<B, S>>::prefill_observation_declarations(
            self,
            metadata_context,
        )?;
        metadata.controls::<(
            Vec<eredu_runtime::layered::PrefillObservationDeclaration>,
            usize,
        )>()?;
        let selected = ordinary
            .iter()
            .filter(|declaration| declaration.flattens_batch_tokens());
        metadata.borrowed_controls(&selected)?;
        let count = selected.count();
        if let Some(context) = metadata.context() {
            context.reserve_metadata_vec(&mut declarations, count)?;
        }
        let selected = ordinary
            .into_iter()
            .filter(|declaration| declaration.flattens_batch_tokens());
        metadata.borrowed_controls(&selected)?;
        declarations.extend(selected);
        Ok(declarations)
    }

    fn prefill_observation_declarations(
        &self,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Self::Error> {
        let metadata = crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(
            &Self,
            Option<&eredu_nn::workspace::WorkspaceContext>,
            usize,
            usize,
            String,
            Vec<eredu_runtime::layered::PrefillObservationDeclaration>,
            std::ops::Range<usize>,
            Option<eredu_runtime::RoutedObservationPoints>,
            Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Error>,
        )>()?;

        // The ordinary target uses four causal convolutions and learned-relative
        // attention at each layer's absolute cache offset. Media and prediction
        // invocations receive no row proof from this target-group declaration.
        let units =
            <Self as LayeredArchitecture<B, S>>::group_unit_count(self, 2, metadata_context)?;
        let mut declarations = crate::decoder::ordinary_prefill_observation_declarations(
            (0..units).map(|index| {
                <Self as LayeredArchitecture<B, S>>::unit_path(self, 2, index, metadata_context)
            }),
            true,
            metadata_context,
        )?;
        {
            if let Some(context) = metadata.context() {
                context.reserve_metadata_vec(&mut declarations, 1)?;
            }
            declarations.push(
                eredu_runtime::layered::PrefillObservationDeclaration::causal_ordinary_text(
                    "readout.scaled".into(),
                    1,
                    eredu_runtime::layered::PrefillReadoutStage::ReadoutInput,
                ),
            );
        };
        // Same target bank invocation as observed execution; its expert equations are row-local.
        for index in 0..units {
            let path =
                <Self as LayeredArchitecture<B, S>>::unit_path(self, 2, index, metadata_context)?;
            if self
                .args
                .text_config
                .layer_schedule
                .get(index)
                .is_some_and(|policy| {
                    policy.feed_forward == crate::inkling::FeedForwardPolicy::SparseMoe
                })
            {
                crate::decoder::append_routed_prefill_path(
                    &mut declarations,
                    &metadata.format(format_args!("{path}.routing"))?,
                    metadata_context,
                )?;
                crate::decoder::append_routed_prefill_path(
                    &mut declarations,
                    &metadata.format(format_args!("{path}.shared.routing"))?,
                    metadata_context,
                )?;
            }
        }
        Ok(declarations)
    }

    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }

    type Input<'a> = ModelInput<'a, B::Tensor>;

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        crate::prefill::segmented_token_shape(input.parts.iter().map(|part| match part {
            DecoderInputPart::Text(tokens)
            | DecoderInputPart::Image(tokens)
            | DecoderInputPart::Audio(tokens)
            | DecoderInputPart::Projected { tokens, .. } => *tokens,
        }))
        .map(Some)
    }

    type StaticModules = StaticModules<B>;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        Self::canonical_group_transport(group)
    }

    fn group_transport_matches(
        &self,
        group: usize,
        expected: &eredu_runtime::ArchitectureGroupTransport,
    ) -> bool {
        match self.construction.units() {
            Some(source) => source.transports.get(group) == Some(expected),
            None => Self::canonical_group_transport(group) == *expected,
        }
    }

    fn primary_execution_group(&self) -> &str {
        TEXT_EXECUTION_GROUP
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        let target_layers = layout
            .segments()
            .iter()
            .find(|segment| segment.id().as_str() == super::TARGET_STATE_SEGMENT)
            .map(|segment| segment.layers().end)
            .unwrap_or(layout.len());
        crate::transport::pipeline_with_output_state(2, target_layers, layout)
    }

    fn execution_graph(
        &self,
    ) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Self::Error> {
        Ok(eredu_runtime::ArchitectureExecutionGraph::borrowed(
            &self.execution_graph,
        ))
    }

    fn group_unit_count(
        &self,
        group: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<usize, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        if let Some(context) = metadata_context {
            return self
                .checked_units(context)?
                .description
                .unit_layout()
                .group_range(group)
                .map(|range| range.len())
                .ok_or_else(|| {
                    metadata.error(format_args!("Inkling group is outside retained source"))
                });
        }
        match group {
            0 => Ok(self
                .args
                .vision_config
                .as_ref()
                .map_or(0, |vision| vision.num_hidden_layers as usize)),
            1 => Ok(0),
            2 => Ok(self.args.text_config.num_hidden_layers as usize),
            _ => Err(metadata.error(format_args!("{}", "Inkling has three execution groups"))),
        }
    }

    fn unit_path(
        &self,
        group: usize,
        index: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<String, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(
            &Self,
            usize,
            usize,
            Option<&eredu_nn::workspace::WorkspaceContext>,
        )>()?;

        let count = match group {
            0 => self
                .args
                .vision_config
                .as_ref()
                .map_or(0, |vision| vision.num_hidden_layers as usize),
            1 => 0,
            2 => self.args.text_config.num_hidden_layers as usize,
            _ => {
                return Err(metadata.error(format_args!("{}", "Inkling has three execution groups")))
            }
        };
        if index >= count {
            return Err(metadata.error(format_args!("{}", "Inkling unit is outside its group")));
        }
        match group {
            0 => metadata.text(format_args!("visual.layers.{index}")),
            2 => metadata.text(format_args!("model.layers.{index}")),
            _ => unreachable!(),
        }
    }

    fn group_input_observation_path(
        &self,
        group: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<String>, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        metadata.optional_path(
            (group == 2).then_some(eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH),
        )
    }

    fn group_output_observation_path(
        &self,
        group: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<String>, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        metadata.optional_path(match group {
            0 => Some(eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH),
            1 => Some(eredu_core::AUDIO_PROJECTOR_OUTPUT_OBSERVATION_PATH),
            _ => None,
        })
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(
            &Self,
            usize,
            usize,
            Unit<B>,
            Option<&construction::PreparedUnits>,
            Result<Unit<B>, Error>,
        )>()?;
        if let Some(source) = self.construction.units() {
            source.validate(self, metadata)?;
            return match group {
                0 => source
                    .vision
                    .get(index)
                    .ok_or_else(|| {
                        metadata.error(format_args!(
                            "Inkling vision unit is outside retained source"
                        ))
                    })?
                    .instantiate::<B>(context)
                    .map(Unit::Vision),
                2 => source
                    .targets
                    .get(index)
                    .ok_or_else(|| {
                        metadata.error(format_args!(
                            "Inkling target unit is outside retained source"
                        ))
                    })?
                    .instantiate::<B>(context)
                    .map(Unit::Text),
                _ => Err(metadata.error(format_args!("Inkling group has no retained units"))),
            };
        }
        if metadata.context().is_some() {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        build_ordinary_unit(self, group, index, context)
    }

    fn state_ordinal(&self, group: usize, index: usize, _ordinal: usize) -> usize {
        match group {
            0 => 0,
            2 => index,
            _ => index,
        }
    }

    fn retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        _ordinal: usize,
    ) -> std::ops::Range<usize> {
        match group {
            0 => 0..0,
            2 => index..index + 1,
            _ => 0..0,
        }
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        if !self.accepts_execution_state(state.layout())? {
            return Err(Error::backend("Inkling runtime state layout mismatch"));
        }
        let parts = self.prepare_parts(input.parts, context)?;
        let tokens = ordered_tokens(&parts, context)?;
        let audio_input = input.audio.as_ref().map(|audio| audio.code_ids.clone());
        let audio_valid_frames = input.audio.as_ref().map(|audio| audio.valid_frames);
        let has_vision = input.vision_patches.is_some();
        let hidden = match input.vision_patches {
            Some(patches) => {
                if patches.shape().len() != 5 || patches.shape()[1..] != [2, 40, 40, 3] {
                    return Err(Error::backend("invalid Inkling hMLP patch geometry"));
                }
                patches.clone()
            }
            None if audio_input.is_some() => audio_input.as_ref().expect("checked").clone(),
            None => self.assemble(&parts, None, None, context)?,
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                parts,
                tokens,
                audio_input,
                audio_valid_frames,
                audio_output: None,
                vision_output: None,
                has_vision,
                target_hidden: None,
                pending_media: None,
                media_span: false,
            },
        })
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 2 && forward.media_span {
            return Ok(initial.clone());
        }
        match (group, dependencies) {
            (0, []) => Ok(initial.clone()),
            (1, []) if forward.audio_input.is_some() => {
                Ok(forward.audio_input.as_ref().expect("checked").clone())
            }
            (1, []) => Ok(initial.clone()),
            (2, [vision, audio]) => self.assemble(
                &forward.parts,
                forward.vision_output.as_ref().map(|_| *vision),
                forward.audio_output.as_ref().map(|_| *audio),
                context,
            ),
            (2, dependencies)
                if dependencies.len()
                    == usize::from(forward.vision_output.is_some())
                        + usize::from(forward.audio_output.is_some()) =>
            {
                let mut dependencies = dependencies.iter().copied();
                let vision = forward
                    .vision_output
                    .as_ref()
                    .and_then(|_| dependencies.next());
                let audio = forward
                    .audio_output
                    .as_ref()
                    .and_then(|_| dependencies.next());
                self.assemble(&forward.parts, vision, audio, context)
            }
            _ => Err(Error::backend("invalid Inkling execution dependencies")),
        }
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        match group {
            0 => forward.has_vision,
            1 => forward.audio_input.is_some(),
            _ => true,
        }
    }

    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let output = match (group, unit) {
            (0, Unit::Vision(unit)) => unit.forward(hidden, context),
            (2, Unit::Text(unit)) => unit.forward(
                hidden,
                Some(
                    state
                        .layer(self.local_state_ordinal(index)?)
                        .map_err(Error::backend)?,
                ),
                context,
            ),
            _ => Err(Error::backend("Inkling unit/group mismatch")),
        }?;
        if group == 2 && index + 1 == self.args.text_config.num_hidden_layers as usize {
            forward.capture_target_hidden(output.clone());
        }
        Ok(output)
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match group {
            0 if forward.has_vision => {
                let vision = self
                    .static_modules
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("Inkling vision static modules are missing"))?
                    .finish(hidden, context)?;
                forward.vision_output = Some(vision.clone());
                Ok(vision)
            }
            0 => Ok(hidden.clone()),
            1 if forward.audio_input.is_some() => {
                let input = AudioInput {
                    code_ids: forward.audio_input.as_ref().expect("checked"),
                    valid_frames: forward.audio_valid_frames.expect("audio frame count"),
                };
                let audio = self
                    .static_modules
                    .audio
                    .as_mut()
                    .ok_or_else(|| Error::backend("Inkling has no audio tower"))?
                    .forward(input, context)?;
                forward.audio_output = Some(audio.clone());
                Ok(audio)
            }
            1 | 2 => Ok(hidden.clone()),
            _ => Err(Error::backend("invalid Inkling execution group")),
        }
    }

    fn select_readout_positions(
        &self,
        hidden: &B::Tensor,
        _forward: &Self::ForwardContext,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        crate::readout::select_readout_positions(hidden, demand, 1, context)
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.target_logits_instrumented(
            hidden,
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = Vec::new();
        values.push(&forward.tokens);
        values.extend(forward.audio_input.iter());
        values.extend(forward.audio_output.iter());
        values.extend(forward.vision_output.iter());
        if let Some(pending) = &forward.pending_media {
            media_prefill::visit_pending(pending, &mut |value| values.push(value));
        }
        for part in &forward.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => values.extend([tokens, embeddings]),
                PreparedPart::Image { tokens } | PreparedPart::Audio { tokens } => {
                    values.push(tokens)
                }
                PreparedPart::Projected { tokens, embeddings } => {
                    values.extend([tokens, embeddings])
                }
            }
        }
        values.into_iter()
    }

    fn forward_unit_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        <Self as RoutedLayeredArchitecture<B, S>>::forward_unit_observed_with_provider(
            self,
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            ExpertPass::Prefill,
            &mut eredu_runtime::ResidentExpertProvider,
            context,
            observer,
        )
    }

    fn finish_forward_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.target_logits_instrumented(
            hidden,
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::new("readout", &mut observer),
        )
    }
}

impl<B, S> ParallelLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AuxiliaryConvolutionState<B::Tensor>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }

    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        if self.parallel_geometry.is_none() {
            return Err(Error::backend(
                "Inkling model was not built with local geometry",
            ));
        }
        if !self.accepts_execution_state(state.layout())? {
            return Err(Error::backend("Inkling rank-local state layout mismatch"));
        }
        let parts = self.prepare_parts_parallel(input.parts, parallel, context)?;
        let tokens = ordered_tokens(&parts, context)?;
        let audio_input = input.audio.as_ref().map(|audio| audio.code_ids.clone());
        let audio_valid_frames = input.audio.as_ref().map(|audio| audio.valid_frames);
        let has_vision = input.vision_patches.is_some();
        let hidden = match input.vision_patches {
            Some(patches) => {
                if patches.shape().len() != 5 || patches.shape()[1..] != [2, 40, 40, 3] {
                    return Err(Error::backend("invalid Inkling hMLP patch geometry"));
                }
                patches.clone()
            }
            None if audio_input.is_some() => audio_input.as_ref().expect("checked").clone(),
            None => self.assemble(&parts, None, None, context)?,
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                parts,
                tokens,
                audio_input,
                audio_valid_frames,
                audio_output: None,
                vision_output: None,
                has_vision,
                target_hidden: None,
                pending_media: None,
                media_span: false,
            },
        })
    }

    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let output = match (group, unit) {
            (0, Unit::Vision(unit)) => unit.forward(hidden, context),
            (2, Unit::Text(unit)) => unit.forward_parallel(
                hidden,
                Some(
                    state
                        .layer(self.local_state_ordinal(index)?)
                        .map_err(Error::backend)?,
                ),
                parallel,
                context,
            ),
            _ => Err(Error::backend("Inkling unit/group mismatch")),
        }?;
        if group == 2 && index + 1 == self.args.text_config.num_hidden_layers as usize {
            forward.capture_target_hidden(output.clone());
        }
        Ok(output)
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if self.parallel_geometry.is_none() {
            return Err(Error::backend(
                "Inkling model was not built with local geometry",
            ));
        }
        self.project_target_logits_parallel(hidden, parallel, context)
    }

    fn forward_unit_parallel_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        <Self as ParallelRoutedLayeredArchitecture<B, S>>::forward_unit_parallel_observed_with_provider(
            self, group, index, unit, hidden, state, forward,
            ExpertPass::Prefill, &mut eredu_runtime::ResidentExpertProvider,
            parallel, context, observer,
        )
    }

    fn finish_forward_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.target_logits_instrumented(
            hidden,
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::new("readout", &mut observer),
        )
    }
}

impl<T> ForwardContext<T> {
    /// Returns complete ordered token identity after text/media placeholder assembly.
    pub const fn tokens(&self) -> &T {
        &self.tokens
    }

    /// Retains the complete post-decoder activation for embedded prediction.
    pub fn capture_target_hidden(&mut self, hidden: T) {
        self.target_hidden = Some(hidden);
    }

    /// Returns the post-decoder activation retained by a TP target pass.
    pub const fn target_hidden(&self) -> Option<&T> {
        self.target_hidden.as_ref()
    }
}

/// Declares prompt-cache identity from the exact local state geometry and global offset.
///
/// The global state order is the target decoder followed by any embedded
/// prediction layers. Prediction layers consume shifted token/hidden pairs and
/// therefore persist a frontier one token behind the target decoder.
pub fn state_identity(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    let layer_count = state_layer_count(args)?;
    state_identity_with_count(
        args,
        layout,
        global_layer_start,
        topology,
        layer_count,
        crate::decoder::identity::Metadata::new(None),
    )
}

fn state_layer_count(args: &ModelArgs) -> Result<usize, Error> {
    let target_layer_count =
        usize::try_from(args.text_config.num_hidden_layers).map_err(Error::backend)?;
    let prediction_layer_count = mtp_state_layout(args)
        .map_err(Error::backend)?
        .as_ref()
        .map_or(0, StateLayout::len);
    target_layer_count
        .checked_add(prediction_layer_count)
        .ok_or_else(|| Error::backend("Inkling state layer count overflowed"))
}

fn state_identity_with_count(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: PromptCacheTopology,
    layer_count: usize,
    metadata: crate::decoder::identity::Metadata<'_>,
) -> Result<ModelStateIdentity, Error> {
    metadata.controls::<(
        &ModelArgs,
        &StateLayout,
        usize,
        PromptCacheTopology,
        ModelStateIdentity,
    )>()?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| metadata.error(format_args!("Inkling owned state range overflowed")))?;
    if global_layer_end > layer_count {
        return Err(metadata.error(format_args!(
            "Inkling owns state layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    eredu_runtime::ModelStateIdentity::new_with_diagnostic(
        metadata.text("inkling")?,
        metadata.text(&args.model_type)?,
        args.architecture_fingerprint_with_metadata(metadata)?,
        layer_count,
        global_layer_start,
        0,
        topology,
        |message| metadata.prompt_error(message),
    )
}

fn ordered_tokens<T: Tensor>(parts: &[PreparedPart<T>], context: &T::Context) -> Result<T, Error> {
    ordered_tokens_with_metadata(parts, context, crate::decoder::ModuleMetadata::ordinary())
}
fn ordered_tokens_with_metadata<T: Tensor>(
    parts: &[PreparedPart<T>],
    context: &T::Context,
    metadata: crate::decoder::ModuleMetadata<'_>,
) -> Result<T, Error> {
    let mut tokens = metadata.vector(parts.len())?;
    tokens.extend(parts.iter().map(prepared_part_tokens).cloned());
    T::concatenate(&tokens, 1, context)
}

fn part_token_count<T: Tensor>(
    parts: &[PreparedPart<T>],
    select: impl Fn(&PreparedPart<T>) -> bool,
) -> i32 {
    parts
        .iter()
        .filter(|part| select(part))
        .map(|part| match part {
            PreparedPart::Text { tokens, .. }
            | PreparedPart::Image { tokens }
            | PreparedPart::Audio { tokens }
            | PreparedPart::Projected { tokens, .. } => tokens.dim(1),
        })
        .sum()
}

fn validate_component<T: Tensor>(
    name: &str,
    value: Option<&T>,
    tokens: i32,
    hidden: i32,
) -> Result<(), Error> {
    validate_component_with_metadata(
        name,
        value,
        tokens,
        hidden,
        crate::decoder::ModuleMetadata::ordinary(),
    )
}
fn validate_component_with_metadata<T: Tensor>(
    name: &str,
    value: Option<&T>,
    tokens: i32,
    hidden: i32,
    metadata: crate::decoder::ModuleMetadata<'_>,
) -> Result<(), Error> {
    match value {
        Some(value) if value.shape() == [1, tokens, hidden] => Ok(()),
        None if tokens == 0 => Ok(()),
        Some(value) => Err(metadata.error(format_args!(
            "Inkling {name} output has shape {:?}, expected [1, {tokens}, {hidden}]",
            value.shape()
        ))),
        None => Err(metadata.error(format_args!(
            "Inkling {name} placeholders require projected media"
        ))),
    }
}

fn slice_component<T: Tensor>(
    value: &T,
    offset: i32,
    length: i32,
    context: &T::Context,
) -> Result<T, Error> {
    value.index(
        &[
            Index::Full,
            Index::Range(offset, offset + length),
            Index::Full,
        ],
        context,
    )
}

impl<B, S> PartitionedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + AuxiliaryConvolutionState<B::Tensor>,
{
    fn partition_observation_hooks(
        &self,
        _tensor_parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }

    type Boundary = eredu_runtime::NoAuxiliaryBoundarySchema;

    fn boundary_schema(
        &self,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Self::Boundary, Self::Error> {
        if let Some(metadata) = metadata {
            metadata.charge_metadata(std::mem::size_of::<(
                &Self,
                Option<&eredu_nn::workspace::WorkspaceContext>,
                Self::Boundary,
                Result<Self::Boundary, Self::Error>,
            )>())?;
        }

        Ok(eredu_runtime::NoAuxiliaryBoundarySchema::new(
            self.args().text_config.hidden_size,
        ))
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        _mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        _first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        if state.layout() != expected {
            return Err(Error::backend("Inkling partition state layout mismatch"));
        }
        let input = match input {
            LayeredPartitionInput::Tokens(tokens) => TextPartitionInput::Tokens(tokens),
            LayeredPartitionInput::Hidden { hidden, .. } => TextPartitionInput::Hidden(hidden),
        };
        self.begin_routed_text_partition(input, None, context)
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        _mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        _first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        if state.layout() != expected {
            return Err(Error::backend("Inkling partition state layout mismatch"));
        }
        let input = match input {
            LayeredPartitionInput::Tokens(tokens) => TextPartitionInput::Tokens(tokens),
            LayeredPartitionInput::Hidden { hidden, .. } => TextPartitionInput::Hidden(hidden),
        };
        self.begin_routed_text_partition(input, Some(parallel), context)
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
                    self.finish_forward_parallel(hidden, state, forward, parallel, context)?
                }
                None => self.finish_forward(hidden, state, forward, context)?,
            };
            Ok(LayeredPartitionOutput::Final {
                output,
                retained: Some(hidden.clone()),
            })
        } else {
            Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: eredu_runtime::NoAuxiliaryBoundary,
            })
        }
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
    ) -> Result<LayeredPartitionOutput<B::Tensor>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, false, parallel, context);
        }
        let output = match parallel {
            Some(parallel) => self.finish_forward_parallel_observed(
                hidden, state, forward, parallel, context, observer,
            )?,
            None => self.finish_forward_observed(hidden, state, forward, context, observer)?,
        };
        Ok(LayeredPartitionOutput::Final {
            output,
            retained: Some(hidden.clone()),
        })
    }
}

#[cfg(test)]
mod state_layout_tests {
    use super::*;

    fn layouts() -> InklingStateLayouts {
        let args = ModelArgs::from_hf_json(
            br#"{"text_config":{"hidden_size":16,"num_hidden_layers":4,"vocab_size":32,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"d_rel":2,
            "intermediate_size":24,"n_routed_experts":2,"num_experts_per_tok":1,
            "n_shared_experts":1},"mtp_config":{"num_nextn_predict_layers":2,
            "num_key_value_heads":1,"head_dim":8,"sconv_kernel_size":3}}"#,
        )
        .unwrap();
        let target = state_layout(&args).unwrap();
        let prediction = mtp_state_layout(&args)
            .unwrap()
            .map(|layout| PartitionState::new(layout, target.len()).unwrap());
        InklingStateLayouts::new(target, prediction).unwrap()
    }

    #[test]
    fn output_partition_appends_architecture_placed_prediction_state() {
        let layouts = layouts();
        let target = PartitionState::new(layouts.target().slice(2..4).unwrap(), 2).unwrap();
        let state = layouts.partition(&target, true).unwrap();

        assert_eq!(state.global_layers(), 2..6);
        assert_eq!(state.layout().segments()[0].layers(), 0..2);
        assert_eq!(state.layout().segments()[1].layers(), 2..4);
    }

    #[test]
    fn interior_partition_cannot_claim_noncontiguous_prediction_state() {
        let layouts = layouts();
        let target = PartitionState::new(layouts.target().slice(0..2).unwrap(), 0).unwrap();

        assert!(layouts.partition(&target, true).is_err());
        assert_eq!(
            layouts.partition(&target, false).unwrap().global_layers(),
            0..2
        );
    }
}

// Keep unused ordinary declarations out of the retained-source constructor's
// frame. The metadata gate remains in build_unit before this worker is called.
#[inline(never)]
fn build_ordinary_unit<B>(
    architecture: &LayeredModel<B>,
    group: usize,
    index: usize,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Unit<B>, Error>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    match group {
        0 => {
            let vision = architecture
                .args
                .vision_config
                .as_ref()
                .ok_or_else(|| Error::backend("Inkling has no vision config"))?;
            Ok(Unit::Vision(VisionLayer::new(
                vision,
                index,
                vision.layer_specs()[index],
                context,
            )?))
        }
        2 => {
            let text = match &architecture.parallel_geometry {
                Some(geometry) => geometry
                    .text_layer(index)
                    .ok_or_else(|| Error::backend("missing rank-local Inkling text geometry"))?,
                None => &architecture.args.text_config,
            };
            let realization = architecture
                .expert_realization
                .as_ref()
                .and_then(|plan| plan.unit_spec(TEXT_EXECUTION_GROUP, index))
                .cloned();
            if architecture.expert_realization.is_some()
                && architecture
                    .args
                    .text_config
                    .layer_policy(index)
                    .is_some_and(|policy| {
                        policy.feed_forward == super::FeedForwardPolicy::SparseMoe
                    })
                && realization.is_none()
            {
                return Err(Error::backend(format!(
                    "Inkling expert realization omits text unit {index}"
                )));
            }
            Ok(Unit::Text(match realization {
                Some(realization) => {
                    DecoderLayer::new_with_expert_realization(text, index, realization, context)?
                }
                None => DecoderLayer::new(text, index, context)?,
            }))
        }
        _ => Err(Error::backend("Inkling has three execution groups")),
    }
}
