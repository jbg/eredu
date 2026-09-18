//! Backend-neutral Gemma 4 multimodal model and layered runtime lifecycle.

mod media_prefill;
mod observation;
mod source;
mod unit_construction;
pub(crate) use source::RetainedModelSource;
pub use media_prefill::{MediaIngress, MediaPrefillPlan};

use std::collections::HashMap;

use eredu_nn::{
    multimodal::{assemble_ordered_inputs, OrderedInputPart},
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, GroupedNeuralBackend, Index,
    LinearOperator, NormalizationOperator, PadMode, Parameterized, RotaryPosition, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionUnitLayout,
    ExpertPass, LayerRuntimeState, LayeredArchitecture, LayeredForwardState, LayeredPartitionInput,
    LayeredPartitionOutput, OwnedParameterGroupSpec, ParallelLayeredArchitecture,
    ParallelPlanError, ParallelRoutedLayeredArchitecture, ParameterGroupOwner,
    PartitionedLayeredArchitecture, RoutedExpertProvider, RoutedLayeredArchitecture, StateLayout,
};

use super::{
    audio_layer_parameter_groups, audio_static_parameter_groups, layer_parameter_groups,
    modality_projection_parameter_groups, state_layout, static_parameter_groups,
    vision_layer_parameter_groups, vision_static_parameter_groups,
    AudioIngressPartPlan, AudioInput, AudioLayer, AudioStatic, BlockInput, DenseBlock,
    FamilyConfig, LocalGeometry, ModalityProjector, SharedAttentionStates, SharedAttentionStore,
    VisionIngressPartPlan, VisionInput, VisionLayer, VisionState, VisionStatic,
};
use crate::{
    composite_execution::{
        CompositeArchitecture, ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
        ExternalPredictionTargetOperation, PreparedCompositeInput,
    },
    media_plan::Gemma4InputPartPlan,
};

/// Stable execution-group identity for Gemma 4 vision ingress.
pub const VISION_EXECUTION_GROUP: &str = "vision";
/// Stable execution-group identity for Gemma 4 audio ingress.
pub const AUDIO_EXECUTION_GROUP: &str = "audio";
/// Stable execution-group identity for Gemma 4 text decoding.
pub const TEXT_EXECUTION_GROUP: &str = "text_decoder";

/// Proves one external assistant against a target and returns its exact capture request.
pub fn external_assistant_capture_request(
    target: &FamilyConfig,
    assistant: &super::assistant::AssistantConfig,
) -> Result<ExternalPredictionCaptureRequest, String> {
    let _compatibility = assistant
        .prove_compatibility(&target.text)
        .map_err(|error| error.to_string())?;
    let last = target
        .text
        .num_hidden_layers()
        .checked_sub(1)
        .ok_or_else(|| "Gemma 4 assistant target has no decoder layer".to_owned())?;
    Ok(ExternalPredictionCaptureRequest::Gemma4SharedAttention {
        final_hidden_path: format!("model.language_model.layers.{last}.output"),
    })
}

fn text_static_parameter_ownership(
    args: &super::ModelArgs,
) -> Result<Vec<OwnedParameterGroupSpec>, ParallelPlanError> {
    let text_static = static_parameter_groups(args)?;
    let mut text_roles = vec!["embedding"];
    if args.hidden_size_per_layer_input > 0 {
        text_roles.extend([
            "per_layer_embedding",
            "per_layer_projection",
            "per_layer_norm",
        ]);
    }
    text_roles.push("norm");
    if !args.tie_word_embeddings {
        text_roles.push("output");
    }
    if text_static.len() != text_roles.len() {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "Gemma text static ownership declared {} groups for {} roles",
            text_static.len(),
            text_roles.len()
        )));
    }
    Ok(text_static
        .into_iter()
        .zip(text_roles)
        .map(|(group, role)| {
            let owner = if role == "embedding" && args.tie_word_embeddings {
                ParameterGroupOwner::static_any_of(["embedding", "output"])
            } else {
                ParameterGroupOwner::static_role(role)
            };
            OwnedParameterGroupSpec::new(owner, group)
        })
        .collect())
}

impl<B, S> PartitionedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
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
    ) -> Result<LayeredPartitionOutput<B::Tensor, TextBoundary<B::Tensor>>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, false, parallel, context);
        }
        Ok(LayeredPartitionOutput::Final {
            output: self.finish_components(hidden, parallel, context, observer)?,
            retained: None,
        })
    }

    type Boundary = TextBoundarySchema;

    fn boundary_schema(&self, metadata: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Self::Boundary, Self::Error> {
        let destination = forward::Metadata::new(metadata);
        if metadata.is_some() {
            self.checked_graph(destination)?;
        }
        let geometry = self.parallel_geometry().ok_or_else(|| destination.error(format_args!(
            "Gemma pipeline boundary requires parallel geometry")))?;
        TextBoundarySchema::from_state_layout_in(&self.args.text, geometry.per_layer_width(),
            geometry.state_layout(), crate::composite_execution::graph::Destination(metadata))
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        match input {
            LayeredPartitionInput::Tokens(tokens) => self.begin_pipeline_text(
                tokens,
                mask.cloned(),
                state,
                expected,
                first_state_ordinal,
                None,
                context,
            ),
            LayeredPartitionInput::Hidden { hidden, auxiliary } => self.resume_pipeline_boundary(
                hidden,
                mask.cloned(),
                auxiliary,
                state,
                expected,
                first_state_ordinal,
                context,
            ),
        }
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        match input {
            LayeredPartitionInput::Tokens(tokens) => self.begin_pipeline_text(
                tokens,
                mask.cloned(),
                state,
                expected,
                first_state_ordinal,
                Some(parallel),
                context,
            ),
            LayeredPartitionInput::Hidden { hidden, auxiliary } => self.resume_pipeline_boundary(
                hidden,
                mask.cloned(),
                auxiliary,
                state,
                expected,
                first_state_ordinal,
                context,
            ),
        }
    }

    fn enter_partition_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.install_pipeline_shared(state, forward, context)?;
        if forward.parts.is_empty() {
            return Ok(initial.clone());
        }
        match parallel {
            Some(parallel) => self.begin_execution_group_parallel(
                group,
                initial,
                &[],
                state,
                forward,
                parallel,
                context,
            ),
            None => self.begin_execution_group(group, initial, &[], state, forward, context),
        }
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor, TextBoundary<B::Tensor>>, Self::Error> {
        if owns_output {
            let output = match parallel {
                Some(parallel) => {
                    self.finish_forward_parallel(hidden, state, forward, parallel, context)?
                }
                None => self.finish_forward(hidden, state, forward, context)?,
            };
            Ok(LayeredPartitionOutput::Final {
                output,
                retained: None,
            })
        } else {
            Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: self.pipeline_boundary(hidden, forward, context)?,
            })
        }
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use eredu_runtime::{ArchitectureBoundary, BoundaryTensorDtype};

    #[test]
    fn optional_per_layer_input_owns_complete_wire_geometry() {
        let present = TextBoundarySchema {
            hidden_size: 16,
            per_layer_geometry: Some((4, 8)),
            shared_geometry: Vec::new(),
        };
        let tensors = present.wire_schema().unwrap().resolve(2, 3).unwrap();
        assert_eq!(tensors.primary().shape(), [2, 3, 16]);
        assert_eq!(tensors.auxiliary().len(), 1);
        assert_eq!(tensors.auxiliary()[0].role(), "per_layer_input");
        assert_eq!(tensors.auxiliary()[0].shape(), [2, 3, 4, 8]);
        assert_eq!(
            tensors.auxiliary()[0].dtype(),
            BoundaryTensorDtype::Activation
        );

        let absent = TextBoundarySchema {
            hidden_size: 16,
            per_layer_geometry: None,
            shared_geometry: Vec::new(),
        };
        assert!(absent.wire_schema().unwrap().auxiliary().is_empty());
    }

    #[test]
    fn per_layer_projection_has_independent_static_owner() {
        let args = super::super::ModelArgs::from_hf_json(
            br#"{
              "model_type":"gemma4","hidden_size":16,"num_hidden_layers":2,
              "intermediate_size":32,"num_attention_heads":4,"num_key_value_heads":2,
              "head_dim":4,"rms_norm_eps":0.000001,"vocab_size":64,
              "max_position_embeddings":128,"layer_types":["full_attention","full_attention"],
              "hidden_size_per_layer_input":4,"vocab_size_per_layer_input":64
            }"#,
        )
        .unwrap();
        let authority = text_static_parameter_ownership(&args).unwrap();
        let projection = authority
            .iter()
            .find(|owned| {
                owned.group().members().iter().any(|member| {
                    member.target() == "model.language_model.per_layer_model_projection.weight"
                })
            })
            .expect("projection ownership");
        assert_eq!(
            projection.owner(),
            &ParameterGroupOwner::static_role("per_layer_projection")
        );
        assert_eq!(projection.group().members().len(), 1);
        assert!(!projection.group().members().iter().any(|member| {
            member.target() == "model.language_model.embed_tokens_per_layer.weight"
        }));
    }
}

/// Pinned text and native media modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Main token embedding table and final text projections.
    pub text: model_text::StaticTextModules<B>,
    /// Optional pinned image phases.
    pub vision: Option<VisionStatic<B>>,
    /// Optional image-to-decoder projection.
    pub vision_projection: Option<ModalityProjector<B>>,
    /// Optional pinned audio phases.
    pub audio: Option<AudioStatic<B>>,
    /// Optional audio-to-decoder projection.
    pub audio_projection: Option<ModalityProjector<B>>,
}

/// One ordered decoder-ingress segment.
pub enum DecoderInputPart<'a, T> {
    /// Ordinary text token IDs.
    Text(&'a T),
    /// Image placeholder IDs.
    Image(&'a T),
    /// Video placeholder IDs.
    Video(&'a T),
    /// Audio placeholder IDs.
    Audio(&'a T),
    /// Caller-supplied decoder-width embeddings paired with semantic token IDs.
    Projected {
        /// Token identities used by per-layer embedding and cache policy.
        tokens: &'a T,
        /// Decoder-width embeddings that bypass token/media encoders.
        embeddings: &'a T,
    },
}

/// Prepared text and optional native media input.
pub struct ModelInput<'a, T> {
    /// Ordered text/media token segments.
    pub parts: &'a [DecoderInputPart<'a, T>],
    /// Optional prepared image/video patches.
    pub vision: Option<VisionInput<'a, T>>,
    /// Optional prepared filter-bank input.
    pub audio: Option<AudioInput<'a, T>>,
    /// Optional replacement IDs for per-layer identity embeddings.
    pub per_layer_tokens: Option<&'a T>,
    /// Optional caller-supplied decoder attention mask.
    pub mask: Option<&'a T>,
}

struct PreparedVision<T> {
    patches: T,
    positions: T,
    valid: T,
    key_mask: T,
    grid_extents: Vec<(i32, i32)>,
}

struct PreparedAudio<T> {
    features: T,
    input_mask: T,
    first_stage_mask: T,
    valid: Vec<i32>,
}

/// Architecture-prepared Gemma 4 decoder and media ingress.
pub struct PreparedCompositeIngress<T> {
    tokens: Vec<T>,
    modalities: Vec<eredu_core::InputModality>,
    projected: Vec<Option<T>>,
    vision: Option<PreparedVision<T>>,
    audio: Option<PreparedAudio<T>>,
    // Returned arrays and host containers retire before their exact destination.
    metadata: Option<eredu_nn::workspace::WorkspaceContext>,
}

impl<T> PreparedCompositeIngress<T> {
    /// Borrows ordered decoder segments with architecture-created placeholders.
    pub fn decoder_parts(&self) -> Vec<DecoderInputPart<'_, T>> {
        self.decoder_part_values().collect()
    }

    fn decoder_parts_with_metadata(&self, metadata: forward::Metadata<'_>)
        -> Result<Vec<DecoderInputPart<'_, T>>, Error>
    {
        let metadata = forward::Metadata::new(self.metadata.as_ref().or(metadata.context()));
        metadata.controls::<(Vec<DecoderInputPart<'_, T>>, &Self)>()?;
        let mut parts = metadata.vector(self.tokens.len())?;
        parts.extend(self.decoder_part_values());
        Ok(parts)
    }

    fn decoder_part_values(&self) -> impl Iterator<Item=DecoderInputPart<'_, T>> {
        self.tokens
            .iter()
            .zip(&self.modalities)
            .zip(&self.projected)
            .map(|((tokens, modality), embeddings)| {
                if let Some(embeddings) = embeddings {
                    DecoderInputPart::Projected { tokens, embeddings }
                } else {
                    match modality {
                        eredu_core::InputModality::Text => DecoderInputPart::Text(tokens),
                        eredu_core::InputModality::Image => DecoderInputPart::Image(tokens),
                        eredu_core::InputModality::Video => DecoderInputPart::Video(tokens),
                        eredu_core::InputModality::Audio => DecoderInputPart::Audio(tokens),
                        _ => unreachable!("Gemma admission rejects other modalities"),
                    }
                }
            })
    }

    /// Borrows the padded vision batch when present.
    pub fn vision_input(&self) -> Option<VisionInput<'_, T>> {
        self.vision.as_ref().map(|vision| VisionInput {
            patches: &vision.patches,
            position_ids: &vision.positions,
            position_valid: &vision.valid,
            key_mask: &vision.key_mask,
            grid_extents: &vision.grid_extents,
        })
    }

    /// Borrows the padded audio batch when present.
    pub fn audio_input(&self) -> Option<AudioInput<'_, T>> {
        self.audio.as_ref().map(|audio| AudioInput {
            features: &audio.features,
            input_mask: &audio.input_mask,
            first_stage_mask: &audio.first_stage_mask,
            valid_subsampled_frames: &audio.valid,
        })
    }
}

fn prediction_token_part<'a, T: Tensor>(
    part: &'a eredu_runtime::PreparedInputPart<T>, plan: &Gemma4InputPartPlan,
    metadata: crate::decoder::identity::Metadata<'_>,
) -> Result<crate::composite_execution::PredictionTokenPart<'a, T>, Error> {
    use crate::composite_execution::PredictionTokenPart;
    match plan {
        Gemma4InputPartPlan::TextTokens { .. } => match part.payload() {
            eredu_runtime::PreparedInputPayload::TokenIds(value) => Ok(PredictionTokenPart::Tokens(value)),
            _ => Err(metadata.error(format_args!("Gemma 4 admitted text part lost its token payload"))),
        },
        Gemma4InputPartPlan::Projected { placeholder_token_id, positions, .. } =>
            Ok(PredictionTokenPart::Repeated { token: *placeholder_token_id, positions: *positions }),
        Gemma4InputPartPlan::Vision { placeholder_token_id, ingress, .. } =>
            Ok(PredictionTokenPart::Repeated { token: *placeholder_token_id, positions: ingress.decoder_positions as u64 }),
        Gemma4InputPartPlan::Audio { placeholder_token_id, ingress, .. } =>
            Ok(PredictionTokenPart::Repeated { token: *placeholder_token_id, positions: ingress.decoder_positions as u64 }),
    }
}

/// Interprets one admitted Gemma 4 input using neutral tensor operations.
pub fn prepare_composite_ingress<B>(
    input: PreparedCompositeInput<'_, B::Tensor, Gemma4InputPartPlan>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedCompositeIngress<B::Tensor>, Error>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    struct VisionPart<T> {
        patches: T,
        positions: T,
        plan: VisionIngressPartPlan,
    }
    struct AudioPart<T> {
        features: T,
        plan: AudioIngressPartPlan,
    }

    use super::ingress::{AudioBatchLayout, VisionBatchLayout};
    let metadata = forward::Metadata::new(input.metadata().or_else(|| B::construction_metadata(context)));
    forward::with_metadata(metadata, || {
    metadata.controls::<(PreparedCompositeIngress<B::Tensor>, Option<eredu_nn::workspace::WorkspaceContext>,
        VisionPart<B::Tensor>, AudioPart<B::Tensor>, Vec<VisionPart<B::Tensor>>, Vec<AudioPart<B::Tensor>>,
        Gemma4InputPartPlan, crate::gemma4::VisionIngressPartPlan, crate::gemma4::AudioIngressPartPlan,
        crate::media_plan::MediaShapePlan, [i32;3], usize, u64,
        eredu_runtime::input::host::PreparedHostPart<'_>,
        eredu_runtime::input::host::HostTensorView<'_>)>()?;
    let prepared = input.prepared();
    let admitted = input.admitted();
    if admitted.ordinary().is_some_and(|value| prepared.identity() != value.identity())
        || prepared.len() != admitted.gemma_parts().len() {
        return Err(metadata.error(format_args!("Gemma 4 prepared input no longer matches its admission")));
    }
    let plans = admitted.gemma_parts();
    metadata.borrowed_controls(&plans)?;
    metadata.controls::<Gemma4InputPartPlan>()?;

    metadata.controls::<(
        Vec<VisionIngressPartPlan>, Vec<AudioIngressPartPlan>,
        VisionBatchLayout<'_>, AudioBatchLayout<'_>,
        Result<VisionBatchLayout<'_>, Error>, Result<AudioBatchLayout<'_>, Error>,
        Vec<B::Tensor>, Vec<B::Tensor>, Vec<B::Tensor>, Vec<(i32, i32)>, Vec<i32>,
        PreparedVision<B::Tensor>, PreparedAudio<B::Tensor>,
        &VisionBatchLayout<'_>, &VisionBatchLayout<'_>,
        &AudioBatchLayout<'_>, &AudioBatchLayout<'_>,
        &VisionPart<B::Tensor>, &AudioPart<B::Tensor>,
        forward::Metadata<'_>, &<B::Tensor as Tensor>::Context,
        [i32; 3], [i32; 4], [(i32, i32); 3], usize, i32,
    )>()?;
    let mut tokens = metadata.vector(prepared.len())?;
    let mut modalities = metadata.vector(prepared.len())?;
    let mut projected = metadata.vector(prepared.len())?;
    let vision_count = admitted.gemma_parts().filter(|part| matches!(part, Gemma4InputPartPlan::Vision { .. })).count();
    let audio_count = admitted.gemma_parts().filter(|part| matches!(part, Gemma4InputPartPlan::Audio { .. })).count();
    let mut vision_parts = metadata.vector(vision_count)?;
    let mut audio_parts = metadata.vector(audio_count)?;
    for (part, plan) in prepared.parts().iter().zip(plans) {
        let token_metadata = crate::decoder::identity::Metadata::new(input.metadata().or_else(|| B::construction_metadata(context)));
        tokens.push(crate::composite_execution::prediction_tokens::materialize(
            prediction_token_part(part, &plan, token_metadata)?, context, token_metadata)?);
        match plan {
            Gemma4InputPartPlan::TextTokens { .. } => {
                let eredu_runtime::PreparedInputPayload::TokenIds(value) = part.payload() else {
                    return Err(metadata.error(format_args!("Gemma 4 admitted text part lost its token payload")));
                };
                modalities.push(eredu_core::InputModality::Text);
                projected.push(None);
            }
            Gemma4InputPartPlan::Projected {
                modality,
                ..
            } => {
                let eredu_runtime::PreparedInputPayload::Embeddings(value) = part.payload() else {
                    return Err(metadata.error(format_args!("Gemma 4 admitted projected part lost its embedding payload")));
                };
                modalities.push(modality);
                projected.push(Some(value.clone()));
            }
            Gemma4InputPartPlan::Vision {
                ingress,
                ..
            } => {
                let eredu_runtime::PreparedInputPayload::Tensor(value) = part.payload() else {
                    return Err(metadata.error(format_args!("Gemma 4 admitted vision part lost its tensor payload")));
                };
                let positions = part
                    .metadata_value(eredu_core::InputMetadataKey::PatchPositions)
                    .ok_or_else(|| metadata.error(format_args!("Gemma 4 vision positions disappeared")))?;
                modalities.push(part.modality());
                projected.push(None);
                vision_parts.push(VisionPart {
                    patches: value.clone(),
                    positions: positions.clone(),
                    plan: ingress,
                });
            }
            Gemma4InputPartPlan::Audio {
                ingress,
                ..
            } => {
                let eredu_runtime::PreparedInputPayload::Tensor(value) = part.payload() else {
                    return Err(metadata.error(format_args!("Gemma 4 admitted audio part lost its tensor payload")));
                };
                modalities.push(eredu_core::InputModality::Audio);
                projected.push(None);
                audio_parts.push(AudioPart {
                    features: value.clone(),
                    plan: ingress,
                });
            }
        }
    }

    let vision = if !vision_parts.is_empty() {
        let mut plans = metadata.vector(vision_parts.len())?;
        plans.extend(vision_parts.iter().map(|part| part.plan.clone()));
        let plan = VisionBatchLayout::new(&plans, |detail| metadata.error(detail))?;
        let mut patches = metadata.vector(vision_parts.len())?;
        let mut positions = metadata.vector(vision_parts.len())?;
        let mut grid_extents = metadata.vector(vision_parts.len())?;
        for part in &vision_parts {
            let extra = plan.padded_patches - part.patches.dim(1);
            if extra < 0 {
                return Err(metadata.error(format_args!("Gemma 4 vision payload exceeds admitted batch padding")));
            }
            patches.push(B::Tensor::pad(
                &part.patches, &[(0, 0), (0, extra), (0, 0)], PadMode::Constant, context,
            )?);
            let extra = plan.padded_patches - part.positions.dim(1);
            if extra < 0 {
                return Err(metadata.error(format_args!("Gemma 4 vision positions exceed admitted batch padding")));
            }
            positions.push(B::Tensor::pad(
                &part.positions, &[(0, 0), (0, extra), (0, 0)], PadMode::Constant, context,
            )?.maximum_i32(0, context)?);
            grid_extents.push((part.plan.grid_height, part.plan.grid_width));
        }
        Some(PreparedVision {
            patches: B::Tensor::concatenate(&patches, 0, context)?,
            positions: B::Tensor::concatenate(&positions, 0, context)?,
            valid: B::Tensor::from_f32_fn(
                &plan.position_valid_shape(), |index| plan.position_valid(index), context,
            )?,
            key_mask: B::Tensor::from_f32_fn(
                &plan.key_mask_shape(), |index| plan.key_mask(index), context,
            )?,
            grid_extents,
        })
    } else { None };

    let audio = if !audio_parts.is_empty() {
        let mut plans = metadata.vector(audio_parts.len())?;
        plans.extend(audio_parts.iter().map(|part| part.plan.clone()));
        let plan = AudioBatchLayout::new(&plans, |detail| metadata.error(detail))?;
        let mut features = metadata.vector(audio_parts.len())?;
        let mut valid = metadata.vector(audio_parts.len())?;
        for part in &audio_parts {
            let extra = plan.padded_frames - part.features.dim(1);
            if extra < 0 {
                return Err(metadata.error(format_args!("Gemma 4 audio payload exceeds admitted batch padding")));
            }
            features.push(B::Tensor::pad(
                &part.features, &[(0, 0), (0, extra), (0, 0)], PadMode::Constant, context,
            )?);
            valid.push(part.plan.valid_subsampled_frames);
        }
        Some(PreparedAudio {
            features: B::Tensor::concatenate(&features, 0, context)?,
            input_mask: B::Tensor::from_f32_fn(
                &plan.input_mask_shape(), |index| plan.input_mask(index), context,
            )?,
            first_stage_mask: B::Tensor::from_f32_fn(
                &plan.first_stage_mask_shape(), |index| plan.first_stage_mask(index), context,
            )?,
            valid,
        })
    } else { None };

    Ok(PreparedCompositeIngress {
        tokens,
        modalities,
        projected,
        vision,
        audio,
        metadata: metadata.context().cloned(),
    })
    })
}

impl<B, S> CompositeArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    type InputPartPlan = Gemma4InputPartPlan;
    type AdmissionConfig = crate::replicated_text::SharedCompositeConfig<FamilyConfig>;

    fn admission_config(&self) -> Self::AdmissionConfig {
        self.args.clone()
    }

    fn retain_admission_config_with_metadata(
        config: &Self::AdmissionConfig,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Self::AdmissionConfig, Error> {
        crate::decoder::identity::Metadata::new(Some(context))
            .controls::<(Self::AdmissionConfig, &Self::AdmissionConfig)>()?;
        Ok(config.clone())
    }

    fn external_assistant_target_profile_ref(config:&Self::AdmissionConfig)
        ->Option<crate::external_assistant::ExternalAssistantTargetProfileRef<'_>> {
        Some(crate::external_assistant::ExternalAssistantTargetProfileRef::Gemma4(config))
    }

    fn external_assistant_target_profile(
        config: &Self::AdmissionConfig,
    ) -> Option<crate::external_assistant::ExternalAssistantTargetProfile> {
        Some(crate::external_assistant::ExternalAssistantTargetProfileRef::Gemma4(config).to_owned())
    }

    fn admit_prepared_input(
        config: &Self::AdmissionConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
    ) -> Result<
        crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>,
        eredu_core::CapabilityError,
    > {
        crate::media_plan::admit_gemma4_input(config, input, inspector)
    }

    fn admit_prepared_input_with_metadata(
        config: &Self::AdmissionConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::media_plan::AdmittedCompositeInput<Self::InputPartPlan>, Error> {
        crate::media_plan::admission::gemma::admit(config, input, inspector, context)
    }

    fn visit_prepared_prediction_tokens(
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
        visitor: &mut dyn FnMut(crate::composite_execution::PredictionTokenPart<'_, B::Tensor>) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let metadata = crate::decoder::identity::Metadata::new(input.metadata());
        metadata.controls::<(Gemma4InputPartPlan, crate::composite_execution::PredictionTokenPart<'_, B::Tensor>)>()?;
        for (part, plan) in input.prepared().parts().iter().zip(input.admitted().gemma_parts()) {
            visitor(prediction_token_part(part, &plan, metadata)?)?;
        }
        Ok(())
    }

    fn should_execute_prepared_group(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> bool {
        match group {
            0 => input
                .admitted().gemma_parts()
                .any(|part| matches!(part, Gemma4InputPartPlan::Vision { .. })),
            1 => input
                .admitted().gemma_parts()
                .any(|part| matches!(part, Gemma4InputPartPlan::Audio { .. })),
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
            input
                .admitted().gemma_parts()
                .filter_map(|part| match (group, part) {
                    (0, Gemma4InputPartPlan::Vision { shape, .. })
                    | (1, Gemma4InputPartPlan::Audio { shape, .. }) => {
                        Some(shape.decoder_positions)
                    }
                    _ => None,
                })
                .try_fold(0_u64, |total, count| total.checked_add(count))
                .ok_or_else(|| "Gemma media boundary positions overflowed".to_owned())?
        } else {
            input
                .admitted().decoder_positions()
        };
        i32::try_from(positions).map_err(|_| "Gemma boundary sequence exceeds i32".into())
    }

    fn prepared_group_continuation_geometry(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<Option<(i32, i32)>, String> {
        let extents = input
            .admitted().gemma_parts()
            .filter_map(|part| match (group, part) {
                (0, Gemma4InputPartPlan::Vision { ingress, .. }) => Some(ingress.padded_patches),
                (1, Gemma4InputPartPlan::Audio { ingress, .. }) => {
                    Some((ingress.padded_frames / 2 + ingress.padded_frames % 2 + 1) / 2)
                }
                _ => None,
            })
            .fold((0usize, None::<i32>), |(count, maximum), value| {
                (count + 1, Some(maximum.map_or(value, |old| old.max(value))))
            });
        let extent = extents.1
            .map(|maximum| {
                i32::try_from(extents.0)
                    .ok()
                    .and_then(|count| maximum.checked_mul(count))
                    .ok_or_else(|| "Gemma batched continuation extent exceeds i32".to_owned())
            })
            .transpose()?;
        let width = match group {
            0 => self.args.vision.as_ref().map(|config| config.hidden_size),
            1 => self.args.audio.as_ref().map(|config| config.hidden_size),
            _ => None,
        };
        Ok(extent.zip(width))
    }

    fn prepared_group_continuation_batched(&self, _group: usize) -> bool {
        // Gemma's encoder activations already retain their batch dimension.
        false
    }

    fn encode_group_continuation(
        &self,
        group: usize,
        hidden: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group >= 2 {
            return Ok(hidden);
        }
        hidden.reshape(&[1, -1, hidden.dim(2)], context)
    }

    fn decode_group_continuation(
        &self,
        group: usize,
        hidden: B::Tensor,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if let Some(pending) = &forward.pending_media {
            let shape = match group {
                0 => pending.vision.as_ref().map(|vision| {
                    [
                        vision.patches.dim(0),
                        vision.patches.dim(1),
                        self.args
                            .vision
                            .as_ref()
                            .expect("admitted vision")
                            .hidden_size,
                    ]
                }),
                1 => pending.audio.as_ref().map(|audio| {
                    let frames = audio.features.dim(1);
                    [
                        audio.features.dim(0),
                        (frames / 2 + frames % 2 + 1) / 2,
                        self.args
                            .audio
                            .as_ref()
                            .expect("admitted audio")
                            .hidden_size,
                    ]
                }),
                _ => return Ok(hidden),
            }
            .ok_or_else(|| {
                Error::backend("Gemma media continuation has no original root geometry")
            })?;
            return hidden.reshape(&shape, context);
        }
        let initial = match group {
            0 => forward.vision_initial.as_ref(),
            1 => forward.audio_initial.as_ref(),
            _ => return Ok(hidden),
        }
        .ok_or_else(|| Error::backend("Gemma media continuation has no admitted batch geometry"))?;
        hidden.reshape(initial.shape(), context)
    }

    fn prepared_group_collective_waves(
        &self, group:usize, input:PreparedCompositeInput<'_,B::Tensor,Self::InputPartPlan>,
        tensor_partitions:usize, pipeline_stages:usize,
        context:Option<&eredu_nn::workspace::WorkspaceContext>,
    )->Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>,Error> {
        let destination=crate::composite_execution::graph::Destination(context);
        destination.controls::<(&Self,usize,PreparedCompositeInput<'_,B::Tensor,Self::InputPartPlan>,
            usize,usize,Option<usize>)>()?;
        let first_active=(0..2).find(|group|
            <Self as CompositeArchitecture<B,S>>::should_execute_prepared_group(self,*group,input));
        prepared_group_waves(input,self.args.text.hidden_size,group,first_active,
            tensor_partitions,pipeline_stages,destination)
    }
    fn prepared_primary_ingress_collectives(
        &self,input:PreparedCompositeInput<'_,B::Tensor,Self::InputPartPlan>,tensor_partitions:usize,
        context:Option<&eredu_nn::workspace::WorkspaceContext>,
    )->Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>,Error> {
        let destination=crate::composite_execution::graph::Destination(context);
        destination.controls::<(&Self,PreparedCompositeInput<'_,B::Tensor,Self::InputPartPlan>,usize)>()?;
        prepared_ingress_waves(input,self.args.text.hidden_size,tensor_partitions,destination)
    }

    fn routed_tensor_reductions(
        &self,
        unit: usize,
        routed: bool,
    ) -> Result<(usize, usize), Self::Error> {
        self.args
            .text
            .layer_policy(unit)
            .ok_or_else(|| Error::backend("Gemma collective plan references an unknown layer"))?;
        let per_layer = usize::from(self.args.text.hidden_size_per_layer_input > 0);
        // Attention and the always-present dense branch reduce before routing;
        // the routed write and optional per-layer write reduce afterward.
        Ok(if routed {
            (2, 1 + per_layer)
        } else {
            (1, 1 + per_layer)
        })
    }

    fn accept_partition_boundary(
        &mut self,
        source_group: usize,
        destination_group: usize,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        mut values: Vec<B::Tensor>,
        forward: &mut Self::ForwardContext,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        let source=forward.metadata.clone();
        let metadata=forward::Metadata::new(source.as_ref());
        metadata.controls::<(Vec<B::Tensor>,TextBoundary<B::Tensor>,TextBoundarySchema,
            B::Tensor,Option<B::Tensor>,Option<eredu_nn::workspace::WorkspaceContext>)>()?;
        if values.len() != 1 + schema.auxiliary().len() {
            return Err(metadata.error(format_args!("Gemma boundary has incomplete typed context")));
        }
        let hidden = values.remove(0);
        match (source_group, destination_group) {
            (0, 2) => forward.vision_output = Some(hidden.clone()),
            (1, 2) => forward.audio_output = Some(hidden.clone()),
            (0, 0) | (1, 1) => {}
            (2, 2) => {
                let geometry = self.parallel_geometry().ok_or_else(|| {
                    metadata.error(format_args!("Gemma decoder boundary requires parallel geometry"))
                })?;
                let boundary_schema=TextBoundarySchema::from_state_layout_in(&self.args.text,
                    geometry.per_layer_width(),geometry.state_layout(),
                    crate::composite_execution::graph::Destination(metadata.context()))?;
                let boundary=match metadata.context(){
                    Some(context)=>eredu_runtime::ArchitectureBoundary::decode_with_metadata(&boundary_schema,values,context)?,
                    None=>eredu_runtime::ArchitectureBoundary::decode(&boundary_schema,values).map_err(Error::backend)?,
                };
                forward.per_layer_inputs = boundary.per_layer_input;
                forward.shared = boundary.shared.into();
                forward.shared_pending = true;
                // The incoming activation has already passed decoder assembly.
                forward.parts.clear();
            }
            _ => return Ok(None),
        }
        Ok(Some(hidden))
    }

    fn external_prediction_capture_paths(
        request: &ExternalPredictionCaptureRequest,
    ) -> Result<Option<Vec<String>>, Self::Error> {
        external_capture_paths(request, crate::decoder::ModuleMetadata::ordinary())
    }

    fn external_prediction_capture_paths_with_metadata(
        request: &ExternalPredictionCaptureRequest,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<Vec<String>>, Error> {
        external_capture_paths(request, crate::decoder::ModuleMetadata::funded(context))
    }

    fn external_prediction_capture(
        request: &ExternalPredictionCaptureRequest,
        forward: &Self::ForwardContext,
        observed: Vec<B::Tensor>,
    ) -> Result<Option<ExternalPredictionTargetCapture<B::Tensor>>, Self::Error> {
        external_capture(request, forward, observed, crate::decoder::ModuleMetadata::ordinary())
    }

    fn external_prediction_capture_with_metadata(
        request: &ExternalPredictionCaptureRequest,
        forward: &Self::ForwardContext,
        observed: Vec<B::Tensor>,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<ExternalPredictionTargetCapture<B::Tensor>>, Error> {
        external_capture(request, forward, observed, crate::decoder::ModuleMetadata::funded(context))
    }

    fn external_prediction_target_operation(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error> {
        match operation {
            ExternalPredictionTargetOperation::TokenEmbeddings(tokens) => {
                self.token_embeddings(tokens, context).map(Some)
            }
            ExternalPredictionTargetOperation::ProjectLogits(_) => Ok(None),
        }
    }

    fn begin_composite_forward<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let metadata = forward::Metadata::new(input.metadata().or_else(|| B::construction_metadata(context)));
        forward::with_metadata(metadata, || {
        let prepared = prepare_composite_ingress::<B>(input, context)?;
        let decoder_parts = prepared.decoder_parts_with_metadata(metadata)?;
        self.begin_forward_with_metadata(
            ModelInput {
                parts: &decoder_parts,
                vision: prepared.vision_input(),
                audio: prepared.audio_input(),
                per_layer_tokens: None,
                mask: None,
            },
            None,
            state,
            context,
            metadata,
        )
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
        let metadata = forward::Metadata::new(input.metadata().or_else(|| B::construction_metadata(context)));
        forward::with_metadata(metadata, || {
        let prepared = prepare_composite_ingress::<B>(input, context)?;
        let decoder_parts = prepared.decoder_parts_with_metadata(metadata)?;
        self.begin_forward_with_metadata(
            ModelInput {
                parts: &decoder_parts,
                vision: prepared.vision_input(),
                audio: prepared.audio_input(),
                per_layer_tokens: None,
                mask: None,
            },
            Some(parallel),
            state,
            context,
            metadata,
        )
        })
    }
}

enum PreparedPart<T> {
    Text { tokens: T, embeddings: T },
    Vision { tokens: T },
    Audio { tokens: T },
}

/// A streamable image, audio, or decoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Vision encoder block.
    Vision(VisionLayer<B>),
    /// Audio encoder block.
    Audio(AudioLayer<B>),
    /// Text decoder block.
    Text(DenseBlock<B>),
}

impl<B, S> RoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn routed_unit_observations(&self) -> bool {
        true
    }
    fn routed_sparse_observations(&self) -> bool {
        true
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
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_components(
            group, index, unit, hidden, state, forward, pass, provider, context, observer,
        )
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
        match (group, unit) {
            (2, Unit::Text(unit)) => self.forward_text_unit_with_provider(
                index, unit, hidden, state, forward, pass, provider, context,
            ),
            (_, unit) => <Self as LayeredArchitecture<B, S>>::forward_unit(
                self, group, index, unit, hidden, state, forward, context,
            ),
        }
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        true
    }
    fn parallel_routed_sparse_observations(&self) -> bool {
        true
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
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_components_parallel(
            group, index, unit, hidden, state, forward, pass, provider, parallel, context, observer,
        )
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
        match (group, unit) {
            (2, Unit::Text(unit)) => self.forward_text_unit_parallel_with_provider(
                index, unit, hidden, state, forward, pass, provider, parallel, context,
            ),
            (_, unit) => <Self as ParallelLayeredArchitecture<B, S>>::forward_unit_parallel(
                self, group, index, unit, hidden, state, forward, parallel, context,
            ),
        }
    }
}

/// Values retained across one complete multimodal decoder pass.
pub struct ForwardContext<T> {
    mask: Option<T>,
    position_offset: i32,
    parts: Vec<PreparedPart<T>>,
    per_layer_token_override: Option<T>,
    per_layer_inputs: Option<T>,
    shared: SharedAttentionPublications<T>,
    shared_pending: bool,
    vision_state: Option<VisionState<T>>,
    vision_initial: Option<T>,
    vision_output: Option<T>,
    audio_valid: Option<Vec<i32>>,
    audio_initial: Option<T>,
    audio_output: Option<T>,
    pending_media: Option<PreparedCompositeIngress<T>>,
    media_span: bool,
    // Exact forwarded destination; every tensor/container above retires first.
    metadata: Option<eredu_nn::workspace::WorkspaceContext>,
}

impl<T> ForwardContext<T> {
    /// Pass-local key/value publications consumed by shared-state layers and
    /// external assistants.
    pub fn shared_attention_states(&self) -> &SharedAttentionPublications<T> {
        &self.shared
    }

    /// Returns the decoder-wide per-layer input tensor transported between
    /// pipeline stages, when the checkpoint declares one.
    pub fn pipeline_per_layer_inputs(&self) -> Option<&T> {
        self.per_layer_inputs.as_ref()
    }
}

/// Family-owned schema for decoder-wide per-layer input transport.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TextBoundarySchema {
    hidden_size: i32,
    per_layer_geometry: Option<(i32, i32)>,
    shared_geometry: Vec<(eredu_core::AttentionPolicy, i32, i32)>,
}

impl TextBoundarySchema {
    /// Derives the schema from normalized text and rank-local geometry.
    pub fn from_args(args: &super::ModelArgs, geometry: &LocalGeometry) -> Self {
        Self::from_state_layout(args, geometry.per_layer_width(), geometry.state_layout())
    }

    /// Derives the same wire schema from exact TP/PP-local construction geometry.
    pub fn from_partition_args(
        args: &super::ModelArgs,
        geometry: &super::PartitionLocalGeometry,
    ) -> Self {
        Self::from_state_layout(
            args,
            geometry.per_layer_range().len() as i32,
            geometry.complete_state_layout(),
        )
    }

    pub(crate) fn from_state_layout(args:&super::ModelArgs,width:i32,state:&StateLayout)->Self {
        Self::from_state_layout_in(args,width,state,crate::composite_execution::graph::Destination(None))
            .expect("ordinary validated boundary geometry has infallible destinations")
    }
    fn from_state_layout_in(args:&super::ModelArgs,width:i32,state:&StateLayout,
        destination:crate::composite_execution::graph::Destination<'_>)->Result<Self,Error> {
        use eredu_core::cache::LayerCachePolicy;
        destination.controls::<(Self,&super::ModelArgs,i32,&StateLayout,usize)>()?;
        let shared_geometry = destination.collect(args
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, policy)| policy.key_value.publishes_state())
            .map(|(layer, policy)| {
                let (heads, dim) = match state.layer(layer).expect("publisher state") {
                    LayerCachePolicy::KeyValue {
                        num_key_value_heads,
                        head_dim,
                        ..
                    }
                    | LayerCachePolicy::KeyValueWithFixedState {
                        num_key_value_heads,
                        head_dim,
                        ..
                    } => (num_key_value_heads.get(), head_dim.get()),
                    LayerCachePolicy::KeyOnly {
                        num_key_heads,
                        head_dim,
                        ..
                    }
                    | LayerCachePolicy::KeyOnlyWithFixedState {
                        num_key_heads,
                        head_dim,
                        ..
                    } => (num_key_heads.get(), head_dim.get()),
                    _ => unreachable!("validated Gemma publisher state"),
                };
                (policy.attention, heads as i32, dim as i32)
            })
            )?;
        Ok(Self {
            hidden_size: args.hidden_size,
            per_layer_geometry: (width > 0).then(|| (args.num_hidden_layers() as i32, width)),
            shared_geometry,
        })
    }
}

impl eredu_runtime::ArchitectureBoundary for TextBoundarySchema {
    type Boundary<T> = TextBoundary<T>;

    const IDENTITY: &'static str = "gemma4.text";

    fn primary_tensor_spec(&self) -> eredu_runtime::BoundaryTensorSpec {
        eredu_runtime::BoundaryTensorSpec::primary_activation(self.hidden_size)
    }

    fn auxiliary_tensor_specs(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
        let mut specs: Vec<_> = self
            .per_layer_geometry
            .map(|(layers, width)| {
                eredu_runtime::BoundaryTensorSpec::new(
                    "per_layer_input",
                    [
                        Dim::Batch,
                        Dim::Sequence,
                        Dim::Fixed(layers),
                        Dim::Fixed(width),
                    ],
                    Dtype::Activation,
                )
            })
            .into_iter()
            .collect();
        for (index, (_, heads, dim)) in self.shared_geometry.iter().enumerate() {
            for component in ["keys", "values"] {
                specs.push(eredu_runtime::BoundaryTensorSpec::new(
                    format!("shared.{index}.{component}"),
                    [
                        Dim::Batch,
                        Dim::Fixed(*heads),
                        Dim::Sequence,
                        Dim::Fixed(*dim),
                    ],
                    Dtype::Activation,
                ));
            }
        }
        specs
    }

    fn wire_schema_with_metadata(&self,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<eredu_runtime::BoundaryWireSchema,Error> {
        use eredu_runtime::{BoundaryTensorDimension as Dim,BoundaryTensorDtype as Dtype,
            BoundaryTensorSpec,BoundaryWireSchema};
        let destination=crate::composite_execution::graph::Destination(Some(context));
        destination.controls::<(&Self,BoundaryWireSchema,BoundaryTensorSpec,usize,[Dim;4],String)>()?;
        let primary=BoundaryTensorSpec::new_with_metadata("hidden",&[Dim::Batch,Dim::Sequence,
            Dim::Fixed(self.hidden_size)],Dtype::Activation,context)?;
        let count=self.shared_geometry.len().checked_mul(2)
            .and_then(|n|n.checked_add(usize::from(self.per_layer_geometry.is_some())))
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        let mut auxiliary=destination.vector(count)?;
        if let Some((layers,width))=self.per_layer_geometry {
            auxiliary.push(BoundaryTensorSpec::new_with_metadata("per_layer_input",&[
                Dim::Batch,Dim::Sequence,Dim::Fixed(layers),Dim::Fixed(width)],Dtype::Activation,context)?);
        }
        for (index,(_,heads,dim)) in self.shared_geometry.iter().enumerate() {
            for component in ["keys","values"] {
                let role=destination.text(format_args!("shared.{index}.{component}"))?;
                auxiliary.push(BoundaryTensorSpec::new_with_metadata(&role,&[
                    Dim::Batch,Dim::Fixed(*heads),Dim::Sequence,Dim::Fixed(*dim)],Dtype::Activation,context)?);
            }
        }
        BoundaryWireSchema::from_owned_with_metadata(Self::IDENTITY,primary,auxiliary,context)
    }
    fn decode_with_metadata<T>(&self,tensors:Vec<T>,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<Self::Boundary<T>,Error> {
        let metadata=forward::Metadata::new(Some(context));
        metadata.controls::<(Self::Boundary<T>,Vec<T>,std::vec::IntoIter<T>,usize)>()?;
        let count=self.shared_geometry.len().checked_mul(2)
            .and_then(|n|n.checked_add(usize::from(self.per_layer_geometry.is_some())))
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        if tensors.len()!=count {return Err(metadata.source(eredu_runtime::ArchitectureBoundaryError::TensorCount{
            boundary:Self::IDENTITY,expected:count,actual:tensors.len()}));}
        let mut tensors=tensors.into_iter();
        let per_layer_input=self.per_layer_geometry.is_some().then(||tensors.next().unwrap());
        let mut shared=SharedAttentionPublications::prepare_policies(&self.shared_geometry,metadata)?;
        for (policy,_,_) in &self.shared_geometry {
            shared.publish(*policy,(tensors.next().unwrap(),tensors.next().unwrap()))?;
        }
        Ok(TextBoundary{per_layer_input,shared})
    }
    fn encode_with_metadata<T>(&self,boundary:Self::Boundary<T>,context:&eredu_nn::workspace::WorkspaceContext)
        ->Result<Vec<eredu_runtime::ArchitectureBoundaryValue<T>>,Error> {
        let metadata=forward::Metadata::new(Some(context));
        metadata.controls::<(Self::Boundary<T>,Vec<eredu_runtime::ArchitectureBoundaryValue<T>>,usize,String)>()?;
        let actual=boundary.shared.len().checked_mul(2)
            .and_then(|n|n.checked_add(usize::from(boundary.per_layer_input.is_some())))
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        let expected=self.shared_geometry.len().checked_mul(2)
            .and_then(|n|n.checked_add(usize::from(self.per_layer_geometry.is_some())))
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        if actual!=expected || boundary.per_layer_input.is_some()!=self.per_layer_geometry.is_some()
            || self.shared_geometry.iter().any(|(policy,_,_)|!boundary.shared.contains_key(policy)) {
            return Err(metadata.source(eredu_runtime::ArchitectureBoundaryError::TensorCount{
                boundary:Self::IDENTITY,expected,actual}));
        }
        let mut values=metadata.vector(expected)?;
        if let Some(tensor)=boundary.per_layer_input {
            values.push(eredu_runtime::ArchitectureBoundaryValue::new_with_metadata("per_layer_input",tensor,context)?);
        }
        let mut shared=boundary.shared;
        for (index,(policy,_,_)) in self.shared_geometry.iter().enumerate() {
            let (keys,payload)=shared.remove(policy).expect("validated shared policy");
            for (component,tensor) in [("keys",keys),("values",payload)] {
                let role=context.metadata_string(format_args!("shared.{index}.{component}"))?;
                values.push(eredu_runtime::ArchitectureBoundaryValue::new(role,tensor)
                    .map_err(|cause|metadata.source(cause))?);
            }
        }
        Ok(values)
    }

    /// Decodes the optional per-layer input without positional guessing.
    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<Self::Boundary<T>, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        let mut tensors = tensors.into_iter();
        let per_layer_input = self
            .per_layer_geometry
            .is_some()
            .then(|| tensors.next().unwrap());
        let shared = self
            .shared_geometry
            .iter()
            .map(|(policy, _, _)| (*policy, (tensors.next().unwrap(), tensors.next().unwrap())))
            .collect::<SharedAttentionStates<T>>().into();
        Ok(TextBoundary {
            per_layer_input,
            shared,
        })
    }

    /// Encodes the optional per-layer input after validating the family schema.
    fn encode<T>(
        &self,
        boundary: TextBoundary<T>,
    ) -> Result<
        Vec<eredu_runtime::ArchitectureBoundaryValue<T>>,
        eredu_runtime::ArchitectureBoundaryError,
    > {
        let actual = usize::from(boundary.per_layer_input.is_some()) + 2 * boundary.shared.len();
        let expected =
            usize::from(self.per_layer_geometry.is_some()) + 2 * self.shared_geometry.len();
        if actual != expected
            || boundary.per_layer_input.is_some() != self.per_layer_geometry.is_some()
            || self
                .shared_geometry
                .iter()
                .any(|(policy, _, _)| !boundary.shared.contains_key(policy))
        {
            return Err(eredu_runtime::ArchitectureBoundaryError::TensorCount {
                boundary: "gemma4.text",
                expected,
                actual,
            });
        }
        let mut values: Vec<_> = boundary
            .per_layer_input
            .into_iter()
            .map(|tensor| eredu_runtime::ArchitectureBoundaryValue::new("per_layer_input", tensor))
            .collect::<Result<_, _>>()?;
        let mut shared = boundary.shared;
        for (index, (policy, _, _)) in self.shared_geometry.iter().enumerate() {
            let (keys, payload) = shared.remove(policy).expect("validated shared policy");
            for (component, tensor) in [("keys", keys), ("values", payload)] {
                values.push(eredu_runtime::ArchitectureBoundaryValue::new(
                    format!("shared.{index}.{component}"),
                    tensor,
                )?);
            }
        }
        Ok(values)
    }
}

/// Typed Gemma text context transported alongside decoder activations.
pub struct TextBoundary<T> {
    /// Decoder-wide per-layer input, when configured by the checkpoint.
    pub per_layer_input: Option<T>,
    shared: SharedAttentionPublications<T>,
}

impl<T> TextBoundary<T> {
    /// Creates a typed text boundary.
    pub fn new(per_layer_input: Option<T>) -> Self {
        Self {
            per_layer_input,
            shared: HashMap::new().into(),
        }
    }
}

/// One Gemma architecture used by resident, layerwise, and streamed runtimes.
pub struct LayeredModel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    args: crate::replicated_text::SharedCompositeConfig<FamilyConfig>,
    source: Option<RetainedModelSource>,
    static_modules: StaticModules<B>,
    parallel_geometry: Option<crate::replicated_text::SharedCompositeConfig<LocalGeometry>>,
    partition_state_offset: usize,
    partition_state: Option<crate::replicated_text::SharedCompositeConfig<StateLayout>>,
    partition_media_inputs: [bool; 2],
    expert_realization: Option<crate::replicated_text::SharedCompositeConfig<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>>,
    execution_graph: ExecutionGraph,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::ArchitectureParameters<B> for LayeredModel<B>
{
    type DefinitionError = Error;


    fn state_layout(
        &self, context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<StateLayout, Self::DefinitionError> {
match context { Some(context) => {
        self.checked_graph(crate::decoder::identity::Metadata::new(Some(context)))?
            .state.clone_workspace(context)
    }, None => {
        self.state_layout_impl()
    } }
}

    fn state_identity(
        &self, state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
        context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
match context { Some(context) => {
        self.source_state_identity(state, topology, context)
    }, None => {
        super::state_identity(
            &self.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
        .map_err(|error| Error::backend(error.to_string()))
    } }
}



    fn parameter_description(
        &self, context: &<B::Tensor as Tensor>::Context,
    ) -> Result<std::borrow::Cow<'_, ArchitectureParameterDescription>, Self::DefinitionError> {
        let metadata=crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(std::borrow::Cow<'_, ArchitectureParameterDescription>, &Self)>()?;
        if self.source.is_some() {
            return Ok(std::borrow::Cow::Borrowed(&self.checked_graph(metadata)?.description));
        }
        (|| {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        self.parameter_description_impl(context)
    })().map(std::borrow::Cow::Owned)
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
        let modules = &self.static_modules;
        visitor.visit("embedding", &modules.text.embeddings)?;
        if let Some(module) = &modules.text.per_layer_embeddings {
            visitor.visit("per_layer_embedding", module)?;
        }
        if let Some(module) = &modules.text.per_layer_projection {
            visitor.visit("per_layer_projection", module)?;
        }
        if let Some(module) = &modules.text.per_layer_norm {
            visitor.visit("per_layer_norm", module)?;
        }
        visitor.visit("norm", &modules.text.norm)?;
        if let Some(module) = &modules.text.head {
            visitor.visit("output", module)?;
        }
        if let Some(module) = &modules.vision {
            visitor.visit("vision", module)?;
        }
        if let Some(module) = &modules.vision_projection {
            visitor.visit("vision_projection", module)?;
        }
        if let Some(module) = &modules.audio {
            visitor.visit("audio", module)?;
        }
        if let Some(module) = &modules.audio_projection {
            visitor.visit("audio_projection", module)?;
        }
        Ok(())
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        let modules = &mut self.static_modules;
        visitor.visit_mut("embedding", &mut modules.text.embeddings)?;
        if let Some(module) = &mut modules.text.per_layer_embeddings {
            visitor.visit_mut("per_layer_embedding", module)?;
        }
        if let Some(module) = &mut modules.text.per_layer_projection {
            visitor.visit_mut("per_layer_projection", module)?;
        }
        if let Some(module) = &mut modules.text.per_layer_norm {
            visitor.visit_mut("per_layer_norm", module)?;
        }
        visitor.visit_mut("norm", &mut modules.text.norm)?;
        if let Some(module) = &mut modules.text.head {
            visitor.visit_mut("output", module)?;
        }
        if let Some(module) = &mut modules.vision {
            visitor.visit_mut("vision", module)?;
        }
        if let Some(module) = &mut modules.vision_projection {
            visitor.visit_mut("vision_projection", module)?;
        }
        if let Some(module) = &mut modules.audio {
            visitor.visit_mut("audio", module)?;
        }
        if let Some(module) = &mut modules.audio_projection {
            visitor.visit_mut("audio_projection", module)?;
        }
        Ok(())
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

    /// Executes one media unit for a continuation request whose primary
    /// layered forward context lives on another pipeline owner.
    pub fn forward_partition_media_continuation(
        &self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        vision_state: Option<&VisionState<B::Tensor>>,
        audio_valid: Option<&[i32]>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match unit {
            Unit::Vision(layer) => self
                .static_modules
                .vision
                .as_ref()
                .ok_or_else(|| Error::backend("Gemma 4 vision static is missing"))?
                .forward_layer(
                    layer,
                    hidden,
                    vision_state.ok_or_else(|| {
                        Error::backend("Gemma 4 vision continuation state is missing")
                    })?,
                    context,
                ),
            Unit::Audio(layer) => layer.forward(
                hidden,
                audio_valid
                    .ok_or_else(|| Error::backend("Gemma 4 audio continuation state is missing"))?,
                context,
            ),
            Unit::Text(_) => Err(Error::backend(
                "Gemma 4 text unit cannot execute in a media continuation",
            )),
        }
    }

    /// Completes optional media projectors while retaining their outputs at the family boundary.
    pub fn complete_partition_media_ingress<S>(
        &mut self,
        forward: &mut LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
        state: &mut S,
        vision_hidden: Option<&B::Tensor>,
        audio_hidden: Option<&B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        for (group, hidden) in [(0, vision_hidden), (1, audio_hidden)] {
            let Some(hidden) = hidden else { continue };
            if Self::partition_media_output(forward, group).is_some() {
                continue;
            }
            match parallel {
                Some(parallel) => {
                    <Self as ParallelLayeredArchitecture<B, S>>::complete_execution_group_parallel(
                        self,
                        group,
                        hidden,
                        state,
                        &mut forward.context,
                        parallel,
                        context,
                    )?
                }
                None => <Self as LayeredArchitecture<B, S>>::complete_execution_group(
                    self,
                    group,
                    hidden,
                    state,
                    &mut forward.context,
                    context,
                )?,
            };
        }
        Ok(())
    }

    /// Returns a completed optional media projector output.
    pub fn partition_media_output(
        forward: &LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
        group: usize,
    ) -> Option<&B::Tensor> {
        match group {
            0 => forward.context.vision_output.as_ref(),
            1 => forward.context.audio_output.as_ref(),
            _ => None,
        }
    }

    /// Replaces a completed optional media projector output before decoder assembly.
    pub fn replace_partition_media_output(
        forward: &mut LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
        group: usize,
        output: B::Tensor,
    ) -> Result<(), Error> {
        match group {
            0 => forward.context.vision_output = Some(output),
            1 => forward.context.audio_output = Some(output),
            _ => return Err(Error::backend("invalid Gemma 4 media execution group")),
        }
        Ok(())
    }

    /// Completes optional media groups and assembles the decoder activation
    /// and per-layer conditioning at the family boundary.
    pub fn finish_partition_media_ingress<S>(
        &mut self,
        mut forward: LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
        state: &mut S,
        vision_hidden: Option<B::Tensor>,
        audio_hidden: Option<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, Option<B::Tensor>), Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        self.complete_partition_media_ingress(
            &mut forward,
            state,
            vision_hidden.as_ref(),
            audio_hidden.as_ref(),
            parallel,
            context,
        )?;
        let (hidden, tokens) = self.assemble_pipeline_text(&forward.context, context)?;
        let per_layer = self.pipeline_per_layer_inputs(&tokens, &hidden, context)?;
        Self::set_pipeline_per_layer_inputs(&mut forward.context, per_layer.clone());
        Ok((hidden, per_layer))
    }

    /// Builds unloaded pinned text and configured media modules.
    pub fn new(
        args: FamilyConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Gemma 4",
            crate::operator_requirements::GEMMA4,
        )?;
        let text = model_text::StaticTextModules::new(&args.text, context)?;
        Self::with_text(args, text, None, context)
    }

    /// Builds the same multimodal graph with planner-derived text geometry.
    pub fn new_parallel(
        args: FamilyConfig,
        geometry: LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Gemma 4",
            crate::operator_requirements::GEMMA4,
        )?;
        args.validate().map_err(Error::backend)?;
        geometry.validate_for(&args).map_err(Error::backend)?;
        let text = model_text::StaticTextModules::new_parallel(&args.text, &geometry, context)?;
        Self::with_text(args, text, Some(crate::replicated_text::SharedCompositeConfig::new(geometry, B::construction_metadata(context))?), context)
    }

    fn with_text(
        args: FamilyConfig,
        text: model_text::StaticTextModules<B>,
        parallel_geometry: Option<crate::replicated_text::SharedCompositeConfig<LocalGeometry>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if B::construction_metadata(context).is_some_and(|m| m.uses_checked_metadata()) {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        // Validate child configurations at the actual initial constructor. The
        // checked constructor only projects this exact immutable source later.
        if let Some(config) = &args.vision { config.validate().map_err(Error::backend)?; }
        if let Some(config) = &args.audio { config.validate().map_err(Error::backend)?; }
        let args = crate::replicated_text::SharedCompositeConfig::new(
            args, B::construction_metadata(context),
        )?;
        Self::with_shared_text(args, text, parallel_geometry, context)
    }

    fn with_shared_text(
        args: crate::replicated_text::SharedCompositeConfig<FamilyConfig>,
        text: model_text::StaticTextModules<B>,
        parallel_geometry: Option<crate::replicated_text::SharedCompositeConfig<LocalGeometry>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, Option<crate::replicated_text::SharedCompositeConfig<LocalGeometry>>, model_text::StaticTextModules<B>,
            crate::replicated_text::SharedCompositeConfig<FamilyConfig>,
            Option<VisionStatic<B>>, Option<AudioStatic<B>>, Option<ModalityProjector<B>>,
            Option<ModalityProjector<B>>, Option<&super::vision::VisionConfig>,
            Option<&super::audio::AudioConfig>)>()?;
        let vision = if args.vision.is_some() {
            Some(VisionStatic::from_source(args.project(
                |args| args.vision.as_ref().expect("selected immutable vision config"),
                B::construction_metadata(context),
            )?, context)?)
        } else { None };
        let vision_projection = if let Some(config) = &args.vision {
            Some(ModalityProjector::new_with_format(
                &args.text, "embed_vision", config.hidden_size, config.rms_norm_eps,
                config.linear_format_for("model.embed_vision.embedding_projection.weight", config.hidden_size),
                context,
            )?)
        } else { None };
        let audio = if args.audio.is_some() {
            Some(AudioStatic::from_source(args.project(
                |args| args.audio.as_ref().expect("selected immutable audio config"),
                B::construction_metadata(context),
            )?, context)?)
        } else { None };
        let audio_projection = if let Some(config) = &args.audio {
            Some(ModalityProjector::new_with_format(
                &args.text, "embed_audio", config.output_proj_dims, config.rms_norm_eps,
                config.linear_format_for("model.embed_audio.embedding_projection.weight", config.output_proj_dims),
                context,
            )?)
        } else { None };
        Ok(Self {
            args,
            source: None,
            execution_graph: Self::build_execution_graph(B::construction_metadata(context))?,
            static_modules: StaticModules {
                text,
                vision,
                vision_projection,
                audio,
                audio_projection,
            },
            parallel_geometry,
            partition_state_offset: 0,
            partition_state: None,
            partition_media_inputs: [true; 2],
            expert_realization: None,
        })
    }

    pub(crate) fn with_expert_realization(
        mut self,
        plan: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        if self.args.text.num_experts.map(|count| count as usize)
            != Some(plan.global_expert_count())
        {
            return Err(Error::backend(
                "Gemma 4 expert realization changed the global expert count",
            ));
        }
        let expected = self
            .args
            .text
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, policy)| {
                policy.feed_forward == super::FeedForwardPolicy::DenseWithSparseMoe
            })
            .map(|(layer, _)| {
                (
                    eredu_runtime::ExecutionGroupId::new(TEXT_EXECUTION_GROUP)
                        .expect("static group"),
                    layer,
                )
            })
            .collect::<std::collections::BTreeSet<_>>();
        if plan
            .unit_specs()
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            != expected
        {
            return Err(Error::backend(
                "Gemma 4 expert realization changed its invocation owners",
            ));
        }
        self.expert_realization = Some(crate::replicated_text::SharedCompositeConfig::new(plan, None)?);
        Ok(self)
    }

    /// Retains the architecture-global ordinal of this pipeline partition's first state row.
    pub(crate) fn with_partition_state(
        mut self,
        geometry: &super::PartitionLocalGeometry,
    ) -> Result<Self, Error> {
        self.source = None;
        self.partition_state_offset = geometry.text_units().start;
        self.partition_state = Some(crate::replicated_text::SharedCompositeConfig::new(geometry.complete_state_layout().clone(), None)?);
        self.partition_media_inputs = [
            geometry
                .vision_units()
                .is_some_and(|units| units.start == 0),
            geometry.audio_units().is_some_and(|units| units.start == 0),
        ];
        Ok(self)
    }

    fn pipeline_boundary(
        &self,
        hidden: &B::Tensor,
        forward: &ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TextBoundary<B::Tensor>, Error> {
        let metadata=forward::Metadata::new(forward.metadata.as_ref().or_else(||B::construction_metadata(context)));
        metadata.controls::<(TextBoundary<B::Tensor>,TextBoundarySchema,SharedAttentionPublications<B::Tensor>,
            B::Tensor,(B::Tensor,B::Tensor),[Index;4],[i32;4],usize)>()?;
        let geometry = self
            .parallel_geometry()
            .ok_or_else(|| metadata.error(format_args!("Gemma pipeline boundary requires local geometry")))?;
        let schema = TextBoundarySchema::from_state_layout_in(&self.args.text,geometry.per_layer_width(),
            geometry.state_layout(),crate::composite_execution::graph::Destination(metadata.context()))?;
        let sequence = hidden.dim(1);
        let mut shared = SharedAttentionPublications::prepare_policies(&schema.shared_geometry,metadata)?;
        for &(policy, heads, dim) in &schema.shared_geometry {
            let pair = match forward.shared.get(&policy) {
                Some((keys, values)) => {
                    let tail = |value: &B::Tensor| {
                        let extent = value.dim(2);
                        if extent < sequence {
                            return Err(metadata.error(format_args!("Gemma publication omitted current positions")));
                        }
                        value.index(
                            &[
                                Index::Full,
                                Index::Full,
                                Index::Range(extent - sequence, extent),
                                Index::Full,
                            ],
                            context,
                        )
                    };
                    if let Some(context)=metadata.context(){context.charge_metadata(std::mem::size_of_val(&tail))?;}
                    (tail(keys)?, tail(values)?)
                }
                None => {
                    // A cut may precede this policy's publisher. Its fixed wire
                    // slots are inactive until that publisher overwrites them;
                    // no consumer or receiver cache can read them before then.
                    let zero =
                        B::Tensor::full_f32(0.0, &[hidden.dim(0), heads, sequence, dim], context)?;
                    (zero.clone(), zero)
                }
            };
            shared.publish(policy, pair)?;
        }
        Ok(TextBoundary {
            per_layer_input: forward.per_layer_inputs.clone(),
            shared,
        })
    }

    fn begin_partition_vision(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: forward::Metadata<'_>,
    ) -> Result<(B::Tensor, VisionState<B::Tensor>), Error> {
        metadata.controls::<(&Self, VisionInput<'_, B::Tensor>, VisionState<B::Tensor>, B::Tensor, [i32;3])>()?;
        let vision = self
            .static_modules
            .vision
            .as_mut()
            .ok_or_else(|| metadata.error(format_args!("Gemma 4 has no vision tower")))?;
        if self.partition_media_inputs[0] {
            return vision.begin_with_metadata(input, context, metadata);
        }
        let shape = [
            input.patches.dim(0),
            input.patches.dim(1),
            self.args
                .vision
                .as_ref()
                .expect("vision configuration")
                .hidden_size,
        ];
        let state = vision.prepare_state_with_metadata(input, context, metadata)?;
        let hidden = B::Tensor::full_f32(0.0, &shape, context)?;
        Ok((hidden, state))
    }

    fn begin_partition_audio(
        &mut self,
        input: AudioInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: forward::Metadata<'_>,
    ) -> Result<(B::Tensor, Vec<i32>), Error> {
        metadata.controls::<(&Self, AudioInput<'_, B::Tensor>, Vec<i32>, B::Tensor, [i32;3])>()?;
        let audio = self
            .static_modules
            .audio
            .as_mut()
            .ok_or_else(|| metadata.error(format_args!("Gemma 4 has no audio tower")))?;
        if self.partition_media_inputs[1] {
            return audio.begin_with_metadata(input, context, metadata);
        }
        let frames = input.features.dim(1);
        let frames = (frames / 2 + frames % 2 + 1) / 2;
        let hidden = B::Tensor::full_f32(
            0.0,
            &[input.features.dim(0), frames, audio.config().hidden_size],
            context,
        )?;
        let mut valid = metadata.vector(input.valid_subsampled_frames.len())?;
        valid.extend_from_slice(input.valid_subsampled_frames);
        Ok((hidden, valid))
    }

    fn install_pipeline_shared<S: LayerRuntimeState<B>>(
        &self,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let source=forward.metadata.clone();
        let metadata=forward::Metadata::new(source.as_ref().or_else(||B::construction_metadata(context)));
        metadata.controls::<(&Self,&mut S,&mut ForwardContext<B::Tensor>,Option<eredu_nn::workspace::WorkspaceContext>,
            std::ops::Range<usize>,usize,usize,B::Tensor,(B::Tensor,B::Tensor))>()?;
        if !std::mem::take(&mut forward.shared_pending) {
            return Ok(());
        }
        let units = self.partition_state_offset..self.partition_state_offset.checked_add(state.layout().len())
            .ok_or_else(||metadata.error(format_args!("Gemma receiver state range overflow")))?;
        let receivers=super::pipeline::receiver_cache_iter(&self.args.text,units);
        if let Some(context)=metadata.context(){context.charge_metadata(std::mem::size_of_val(&receivers))?;}
        for (receiver, publisher) in receivers {
            let policy = self
                .args
                .text
                .layer_policy(publisher)
                .expect("validated publisher")
                .attention;
            let (keys, values) = forward
                .shared
                .remove(&policy)
                .ok_or_else(|| metadata.error(format_args!("Gemma boundary omitted a remote publication")))?;
            let cache = state
                .layer(receiver.checked_sub(self.partition_state_offset).ok_or_else(||
                    metadata.error(format_args!("Gemma receiver precedes its local state source")))?)
                .map_err(|cause|metadata.source(cause))?;
            if cache.offset() != forward.position_offset {
                return Err(metadata.error(format_args!("Gemma shared-state receiver frontier differs from decoder")));
            }
            let pair = cache.update_for_attention(keys, values, context)?;
            forward.shared.publish(policy, pair)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn resume_pipeline_boundary<S: LayerRuntimeState<B>>(
        &self,
        hidden: B::Tensor,
        mask: Option<B::Tensor>,
        boundary: TextBoundary<B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let mut forward = self.resume_pipeline_text_with_metadata(
            hidden,
            mask,
            boundary.per_layer_input,
            state,
            expected,
            first_state_ordinal,
            forward::Metadata::new(B::construction_metadata(context)),
        )?;
        forward.context.shared = boundary.shared.into();
        forward.context.shared_pending = true;
        self.install_pipeline_shared(state, &mut forward.context, context)?;
        Ok(forward)
    }

    fn local_state_ordinal(&self, global: usize) -> Result<usize, Error> {
        global
            .checked_sub(self.partition_state_offset)
            .ok_or_else(|| Error::backend("Gemma 4 unit precedes the partition state offset"))
    }

    fn attention_state_ordinal(&self, global: usize, state_len: usize) -> Result<usize, Error> {
        let end = self
            .partition_state_offset
            .checked_add(state_len)
            .ok_or_else(|| Error::backend("Gemma attention state interval overflow"))?;
        let owner = super::pipeline::attention_cache_owner(
            &self.args.text,
            self.partition_state_offset..end,
            global,
        )
        .ok_or_else(|| Error::backend("Gemma attention has no local history owner"))?;
        self.local_state_ordinal(owner)
    }

    fn validate_partition_state<S:LayerRuntimeState<B>>(&self,state:&S)->Result<(),Error>{
        self.validate_partition_state_with_metadata(state,forward::Metadata::new(None))
    }
    fn partition_position_offset<S:LayerRuntimeState<B>>(&self,state:&mut S)->Result<i32,Error>
    where S::LayerState:AttentionCache<B::Tensor>{
        self.partition_position_offset_with_metadata(state,forward::Metadata::new(None))
    }

    /// Applies the target's ordinary scaled token embedding operation.
    ///
    /// External assistants use this method rather than owning or copying a
    /// second target embedding implementation.
    pub fn token_embeddings(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules
            .text
            .embeddings
            .forward(tokens, context)?
            .multiply_scalar((self.args.text.hidden_size as f32).sqrt(), context)
    }

    fn validate_unit_index(&self,group:usize,index:usize,metadata:crate::decoder::ModuleMetadata<'_>)->Result<(),Error> {
        metadata.controls::<(&Self,usize,usize,usize)>()?;
        let count = match group {
            0 => self
                .args
                .vision
                .as_ref()
                .map_or(0, |config| config.num_hidden_layers as usize),
            1 => self
                .args
                .audio
                .as_ref()
                .map_or(0, |config| config.num_hidden_layers as usize),
            2 => self.args.text.num_hidden_layers(),
            _ => return Err(metadata.error(format_args!("Gemma 4 has three execution groups"))),
        };
        if index >= count {
            return Err(metadata.error(format_args!("Gemma 4 unit is outside its group")));
        }
        Ok(())
    }

    fn canonical_group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        match group {
            0 | 1 => super::pipeline::media_transport(&self.args, group),
            _ => eredu_runtime::ArchitectureGroupTransport {
                placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
                kind: eredu_runtime::ArchitectureGroupKind::Decoder,
                first_owner_static_roles: vec![
                    "embedding".into(),
                    "per_layer_embedding".into(),
                    "per_layer_projection".into(),
                    "per_layer_norm".into(),
                ],
                last_owner_static_roles: if self.args.text.tie_word_embeddings {
                    vec!["norm".into(), "embedding".into()]
                } else {
                    vec!["norm".into(), "output".into()]
                },
                merge_destination: eredu_runtime::ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::Decoder),
                request_optional: false,
            },
        }
    }


    /// Returns the normalized composite family configuration.
    pub fn args(&self) -> &FamilyConfig {
        &self.args
    }

    /// Describes every pinned/media/text parameter with explicit canonical
    /// execution ownership.
    fn parameter_description_impl(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let graph = match B::construction_metadata(context) { Some(context) => self.execution_graph.clone_with_metadata(context)?, None => self.execution_graph.clone() };
        let counts = [
            self.args
                .vision
                .as_ref()
                .map_or(0, |config| config.num_hidden_layers as usize),
            self.args
                .audio
                .as_ref()
                .map_or(0, |config| config.num_hidden_layers as usize),
            self.args.text.num_hidden_layers(),
        ];
        let layout = ExecutionUnitLayout::new(&graph, counts).map_err(Error::backend)?;
        let mut owned = text_static_parameter_ownership(&self.args.text).map_err(Error::backend)?;
        let mut expected = owned
            .iter()
            .map(|owned| owned.group().clone())
            .collect::<Vec<_>>();
        {
            let mut add_static = |role: &'static str, groups: Vec<_>| {
                expected.extend(groups.iter().cloned());
                owned.extend(groups.into_iter().map(|group| {
                    OwnedParameterGroupSpec::new(ParameterGroupOwner::static_role(role), group)
                }));
            };
            if let Some(vision) = &self.static_modules.vision {
                add_static(
                    "vision",
                    vision_static_parameter_groups(vision).map_err(Error::backend)?,
                );
            }
            if let Some(projector) = &self.static_modules.vision_projection {
                add_static(
                    "vision_projection",
                    modality_projection_parameter_groups("model.vision_projector", projector)
                        .map_err(Error::backend)?,
                );
            }
            if let Some(audio) = &self.static_modules.audio {
                add_static(
                    "audio",
                    audio_static_parameter_groups(audio).map_err(Error::backend)?,
                );
            }
            if let Some(projector) = &self.static_modules.audio_projection {
                add_static(
                    "audio_projection",
                    modality_projection_parameter_groups("model.audio_projector", projector)
                        .map_err(Error::backend)?,
                );
            }
        }
        for (group_index, &count) in counts.iter().enumerate() {
            let owner_group = layout
                .group_id(group_index)
                .expect("Gemma layout group")
                .clone();
            for index in 0..count {
                let groups = match group_index {
                    0 => vision_layer_parameter_groups(
                        &VisionLayer::<B>::new(
                            self.args.vision.as_ref().expect("vision group configured"),
                            index,
                            context,
                        )?,
                        index,
                    ),
                    1 => audio_layer_parameter_groups(
                        &AudioLayer::<B>::new(
                            self.args.audio.as_ref().expect("audio group configured"),
                            index,
                            context,
                        )?,
                        index,
                    ),
                    _ => layer_parameter_groups(&self.args.text, index),
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

    /// Returns the replicated or rank-local mutable-state layout.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        if let Some(state) = &self.partition_state {
            return Ok((**state).clone());
        }
        self.parallel_geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .map_or_else(|| state_layout(&self.args.text).map_err(Error::backend), Ok)
    }

    /// Returns planner-derived geometry for a rank-local realization.
    pub fn parallel_geometry(&self) -> Option<&LocalGeometry> {
        self.parallel_geometry.as_deref()
    }

    /// Starts a text-only pass from a rank-local vocabulary embedding shard.
    pub fn begin_parallel_text<S: LayerRuntimeState<B>>(
        &mut self,
        tokens: &B::Tensor,
        embeddings: B::Tensor,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout().len() != self.args.text.num_hidden_layers() {
            return Err(Error::backend("Gemma 4 rank-local state layout mismatch"));
        }
        let embeddings =
            embeddings.multiply_scalar((self.args.text.hidden_size as f32).sqrt(), context)?;
        let mut position_offset = 0;
        for local in 0..state.layout().len() {
            position_offset = position_offset.max(AttentionCache::<B::Tensor>::offset(
                state.layer(local).map_err(Error::backend)?,
            ));
        }
        Ok(LayeredForwardState {
            hidden: embeddings.clone(),
            context: ForwardContext {
                mask: None,
                position_offset,
                parts: vec![PreparedPart::Text {
                    tokens: tokens.clone(),
                    embeddings,
                }],
                per_layer_token_override: None,
                per_layer_inputs: None,
                shared: HashMap::new().into(),
                shared_pending: false,
                vision_state: None,
                vision_initial: None,
                vision_output: None,
                audio_valid: None,
                audio_initial: None,
                audio_output: None,
                pending_media: None,
                media_span: false,
                metadata: None,
            },
        })
    }

    /// Starts a text-only pipeline partition against its stage-local state.
    #[allow(clippy::too_many_arguments)]
    fn begin_pipeline_text<S: LayerRuntimeState<B>>(
        &mut self,
        tokens: &B::Tensor,
        mask: Option<B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        _first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let metadata=forward::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(LayeredForwardState<B::Tensor,ForwardContext<B::Tensor>>,
            B::Tensor,Option<B::Tensor>,&mut S,&StateLayout,usize,Option<&B::ParallelContext>)>()?;
        if state.layout() != expected {
            return Err(metadata.error(format_args!("Gemma 4 pipeline state layout mismatch")));
        }
        let input = [DecoderInputPart::Text(tokens)];
        let parts = match parallel {
            Some(parallel) => self.prepare_parts_parallel(&input, parallel, context)?,
            None => self.prepare_parts(&input, context)?,
        };
        let hidden = match &parts[0] {
            PreparedPart::Text { embeddings, .. } => embeddings.clone(),
            PreparedPart::Vision { .. } | PreparedPart::Audio { .. } => {
                unreachable!("text input prepares a text part")
            }
        };
        let mut position_offset = 0;
        for local in 0..state.layout().len() {
            position_offset = position_offset.max(AttentionCache::<B::Tensor>::offset(
                state.layer(local).map_err(|cause|metadata.source(cause))?,
            ));
        }
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                position_offset,
                parts,
                per_layer_token_override: None,
                per_layer_inputs: None,
                shared:SharedAttentionPublications::prepare(&self.args.text,metadata)?,
                shared_pending: false,
                vision_state: None,
                vision_initial: None,
                vision_output: None,
                audio_valid: None,
                audio_initial: None,
                audio_output: None,
                pending_media: None,
                media_span: false,
                metadata: metadata.context().cloned(),
            },
        })
    }

    /// Resumes a decoder-only pipeline stage from transported activations.
    pub fn resume_pipeline_text<S: LayerRuntimeState<B>>(
        &self,
        hidden: B::Tensor,
        mask: Option<B::Tensor>,
        per_layer_inputs: Option<B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        _first_state_ordinal: usize,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        self.resume_pipeline_text_with_metadata(hidden,mask,per_layer_inputs,state,expected,
            _first_state_ordinal,forward::Metadata::new(None))
    }
    fn resume_pipeline_text_with_metadata<S: LayerRuntimeState<B>>(
        &self,
        hidden: B::Tensor,
        mask: Option<B::Tensor>,
        per_layer_inputs: Option<B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        _first_state_ordinal: usize,
        metadata:forward::Metadata<'_>,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        metadata.controls::<(LayeredForwardState<B::Tensor,ForwardContext<B::Tensor>>,
            B::Tensor,Option<B::Tensor>,Option<B::Tensor>,&mut S,&StateLayout,usize)>()?;
        if state.layout() != expected {
            return Err(metadata.error(format_args!("Gemma 4 pipeline state layout mismatch")));
        }
        let mut position_offset = 0;
        for local in 0..state.layout().len() {
            position_offset = position_offset.max(AttentionCache::<B::Tensor>::offset(
                state.layer(local).map_err(|cause|metadata.source(cause))?,
            ));
        }
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                position_offset,
                parts: Vec::new(),
                per_layer_token_override: None,
                per_layer_inputs,
                shared:SharedAttentionPublications::prepare(&self.args.text,metadata)?,
                shared_pending: false,
                vision_state: None,
                vision_initial: None,
                vision_output: None,
                audio_valid: None,
                audio_initial: None,
                audio_output: None,
                pending_media: None,
                media_span: false,
                metadata: metadata.context().cloned(),
            },
        })
    }

    /// Assembles completed media roots into decoder activations and token
    /// identity without imposing a backend-specific per-layer sharding plan.
    pub fn assemble_pipeline_text(
        &self,
        forward: &ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), Error> {
        let assembled = self.assemble(
            &forward.parts,
            forward.vision_output.as_ref(),
            forward.audio_output.as_ref(),
            context,
        )?;
        Ok((assembled.embeddings, assembled.token_ids))
    }

    /// Computes the ordinary replicated per-layer decoder input.
    pub fn pipeline_per_layer_inputs(
        &mut self,
        token_ids: &B::Tensor,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error> {
        self.per_layer_inputs(token_ids, hidden, context)
    }

    /// Installs a backend-planned per-layer input on a resumed decoder context.
    pub fn set_pipeline_per_layer_inputs(
        forward: &mut ForwardContext<B::Tensor>,
        value: Option<B::Tensor>,
    ) {
        forward.per_layer_inputs = value;
    }

    /// Starts a multimodal pass from rank-local text embeddings while the
    /// neutral family retains image/audio equations, assembly, and per-layer
    /// embedding policy.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_parallel_input<S: LayerRuntimeState<B>>(
        &mut self,
        input: ModelInput<'_, B::Tensor>,
        text_embeddings: &[B::Tensor],
        vision_layers: &mut [VisionLayer<B>],
        audio_layers: &mut [AudioLayer<B>],
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout().len() != self.args.text.num_hidden_layers() {
            return Err(Error::backend("Gemma 4 rank-local state layout mismatch"));
        }
        let scale = (self.args.text.hidden_size as f32).sqrt();
        let mut next_embedding = text_embeddings.iter();
        let parts = input
            .parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: next_embedding
                        .next()
                        .ok_or_else(|| Error::backend("missing Gemma 4 rank-local text embedding"))?
                        .multiply_scalar(scale, context)?,
                }),
                DecoderInputPart::Image(tokens) | DecoderInputPart::Video(tokens) => {
                    Ok(PreparedPart::Vision {
                        tokens: (*tokens).clone(),
                    })
                }
                DecoderInputPart::Audio(tokens) => Ok(PreparedPart::Audio {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Projected { tokens, embeddings } => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: (*embeddings).clone(),
                }),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        if next_embedding.next().is_some() {
            return Err(Error::backend("unused Gemma 4 rank-local text embedding"));
        }

        let vision_output = match input.vision {
            Some(vision) => {
                let expected = self
                    .args
                    .vision
                    .as_ref()
                    .map_or(0, |config| config.num_hidden_layers as usize);
                if vision_layers.len() != expected {
                    return Err(Error::backend("Gemma 4 vision layer count mismatch"));
                }
                let static_modules = self
                    .static_modules
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("Gemma 4 has no vision tower"))?;
                let (mut hidden, vision_state) = static_modules.begin(vision, context)?;
                for layer in vision_layers {
                    hidden =
                        static_modules.forward_layer(layer, &hidden, &vision_state, context)?;
                }
                let encoded = static_modules.finish(&hidden, &vision_state, context)?;
                Some(
                    self.static_modules
                        .vision_projection
                        .as_mut()
                        .ok_or_else(|| Error::backend("Gemma 4 has no vision projection"))?
                        .forward(&encoded, context)?,
                )
            }
            None => None,
        };
        let audio_output = match input.audio {
            Some(audio) => {
                let expected = self
                    .args
                    .audio
                    .as_ref()
                    .map_or(0, |config| config.num_hidden_layers as usize);
                if audio_layers.len() != expected {
                    return Err(Error::backend("Gemma 4 audio layer count mismatch"));
                }
                let static_modules = self
                    .static_modules
                    .audio
                    .as_mut()
                    .ok_or_else(|| Error::backend("Gemma 4 has no audio tower"))?;
                let (mut hidden, valid) = static_modules.begin(audio, context)?;
                for layer in audio_layers {
                    hidden = layer.forward(&hidden, &valid, context)?;
                }
                let encoded = static_modules.finish(&hidden, &valid, context)?;
                Some(
                    self.static_modules
                        .audio_projection
                        .as_mut()
                        .ok_or_else(|| Error::backend("Gemma 4 has no audio projection"))?
                        .forward(&encoded, context)?,
                )
            }
            None => None,
        };
        let assembled = self.assemble(
            &parts,
            vision_output.as_ref(),
            audio_output.as_ref(),
            context,
        )?;
        let per_layer_tokens = input.per_layer_tokens.unwrap_or(&assembled.token_ids);
        let per_layer_inputs =
            self.per_layer_inputs(per_layer_tokens, &assembled.embeddings, context)?;
        let mut position_offset = 0;
        for local in 0..state.layout().len() {
            position_offset = position_offset.max(AttentionCache::<B::Tensor>::offset(
                state.layer(local).map_err(Error::backend)?,
            ));
        }
        Ok(LayeredForwardState {
            hidden: assembled.embeddings,
            context: ForwardContext {
                mask: input.mask.cloned(),
                position_offset,
                parts,
                per_layer_token_override: input.per_layer_tokens.cloned(),
                per_layer_inputs,
                shared: HashMap::new().into(),
                shared_pending: false,
                vision_state: None,
                vision_initial: None,
                vision_output,
                audio_valid: None,
                audio_initial: None,
                audio_output,
                pending_media: None,
                media_span: false,
                metadata: None,
            },
        })
    }

    /// Executes one text block with rank-local projections and collectives.
    pub fn forward_text_unit_parallel<S: LayerRuntimeState<B>>(
        &mut self,
        index: usize,
        unit: &mut DenseBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        let (generated_mask, per_layer_input) =
            self.text_block_inputs(index, hidden, forward, context)?;
        unit.forward_parallel(
            BlockInput {
                hidden,
                mask: forward.mask.as_ref().or(generated_mask.as_ref()),
                cache: Some(
                    state
                        .layer(self.attention_state_ordinal(index, state.layout().len())?)
                        .map_err(Error::backend)?,
                ),
                shared: &mut forward.shared,
                per_layer_input: per_layer_input.as_ref(),
                rotary_position: Some(RotaryPosition::Offset(forward.position_offset)),
            },
            parallel,
            context,
        )
    }

    /// Executes one rank-local text block with a runtime-owned routed bank.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_text_unit_parallel_with_provider<S, P>(
        &mut self,
        index: usize,
        unit: &mut DenseBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let (generated_mask, per_layer_input) =
            self.text_block_inputs(index, hidden, forward, context)?;
        unit.forward_parallel_with_provider(
            BlockInput {
                hidden,
                mask: forward.mask.as_ref().or(generated_mask.as_ref()),
                cache: Some(
                    state
                        .layer(self.attention_state_ordinal(index, state.layout().len())?)
                        .map_err(Error::backend)?,
                ),
                shared: &mut forward.shared,
                per_layer_input: per_layer_input.as_ref(),
                rotary_position: Some(RotaryPosition::Offset(forward.position_offset)),
            },
            pass,
            provider,
            parallel,
            context,
        )
    }

    /// Applies the replicated final norm before the vocabulary-parallel projection.
    pub fn final_parallel_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.norm.forward(hidden, context)
    }

    /// Applies final-logit softcapping after vocabulary shards are gathered.
    pub fn finish_parallel_logits(
        &self,
        logits: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        model_text::args_cap::<B>(logits, self.args.text.final_logit_softcapping, context)
    }

    /// Executes one text unit while delegating its routed bank to a runtime-owned provider.
    ///
    /// This is the same architecture equation used by ordinary resident execution; it only
    /// replaces the expert residency policy. Composition layers use it for bounded expert
    /// caching and expert-parallel exchange without owning a second decoder loop.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_text_unit_with_provider<S, P>(
        &mut self,
        index: usize,
        unit: &mut DenseBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let (generated_mask, per_layer_input) =
            self.text_block_inputs(index, hidden, forward, context)?;
        unit.forward_with_provider(
            BlockInput {
                hidden,
                mask: forward.mask.as_ref().or(generated_mask.as_ref()),
                cache: Some(
                    state
                        .layer(self.attention_state_ordinal(index, state.layout().len())?)
                        .map_err(Error::backend)?,
                ),
                shared: &mut forward.shared,
                per_layer_input: per_layer_input.as_ref(),
                rotary_position: Some(RotaryPosition::Offset(forward.position_offset)),
            },
            pass,
            provider,
            context,
        )
    }

    fn prepare_parts(&mut self, parts: &[DecoderInputPart<'_, B::Tensor>],
        context: &<B::Tensor as Tensor>::Context) -> Result<Vec<PreparedPart<B::Tensor>>, Error> {
        self.prepare_parts_with_metadata(parts, None, context,
            forward::Metadata::new(B::construction_metadata(context)))
    }

    fn prepare_parts_parallel(
        &mut self, parts: &[DecoderInputPart<'_, B::Tensor>],
        parallel: &B::ParallelContext, context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<PreparedPart<B::Tensor>>, Error> {
        self.prepare_parts_with_metadata(parts, Some(parallel), context,
            forward::Metadata::new(B::construction_metadata(context)))
    }

    fn assemble(&self, parts: &[PreparedPart<B::Tensor>], vision: Option<&B::Tensor>,
        audio: Option<&B::Tensor>, context: &<B::Tensor as Tensor>::Context)
        -> Result<eredu_nn::multimodal::OrderedModelInput<B::Tensor>, Error> {
        self.assemble_with_metadata(parts, vision, audio, context,
            forward::Metadata::new(B::construction_metadata(context)))
    }

    fn per_layer_inputs(
        &mut self,
        token_ids: &B::Tensor,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error> {
        let (Some(embeddings), Some(projection), Some(norm)) = (
            self.static_modules.text.per_layer_embeddings.as_mut(),
            self.static_modules.text.per_layer_projection.as_mut(),
            self.static_modules.text.per_layer_norm.as_mut(),
        ) else {
            return Ok(None);
        };
        let batch = hidden.dim(0);
        let sequence = hidden.dim(1);
        let layers = self.args.text.num_hidden_layers() as i32;
        let width = self.args.text.hidden_size_per_layer_input;
        let token_identity = embeddings
            .forward(token_ids, context)?
            .multiply_scalar((width as f32).sqrt(), context)?
            .reshape(&[batch, sequence, layers, width], context)?;
        let projected = projection
            .forward(hidden, context)?
            .multiply_scalar((self.args.text.hidden_size as f32).sqrt().recip(), context)?
            .reshape(&[batch, sequence, layers, width], context)?;
        let projected = norm.forward(&projected, context)?;
        let inputs = projected
            .add(&token_identity, context)?
            .multiply_scalar(2.0_f32.powf(-0.5), context)?;
        match self.parallel_geometry.as_ref() {
            Some(geometry) if geometry.per_layer_range() != &(0..width) => inputs
                .index(
                    &[
                        Index::Full,
                        Index::Full,
                        Index::Full,
                        Index::Range(
                            geometry.per_layer_range().start,
                            geometry.per_layer_range().end,
                        ),
                    ],
                    context,
                )
                .map(Some),
            _ => Ok(Some(inputs)),
        }
    }
}

impl<B, S> LayeredArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn media_prefill_observation_declarations(
        &self, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Self::Error> {
        let metadata=crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(&Self,Option<&eredu_nn::workspace::WorkspaceContext>,usize,usize,String,Vec<eredu_runtime::layered::PrefillObservationDeclaration>,std::ops::Range<usize>,Option<eredu_runtime::RoutedObservationPoints>,Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>,Error>)>()?;

        // The validated image/audio ingress retains exact placements and masks; shared-KV decoder rows keep their original offsets.
        let units = <Self as LayeredArchitecture<B, S>>::group_unit_count(self, 2, metadata_context)?;
        let mut declarations = crate::decoder::media_prefill_observation_declarations(
            (0..units).map(|index| <Self as LayeredArchitecture<B, S>>::unit_path(self, 2, index, metadata_context)), metadata_context)?;
        // The retained-media ingress changes decoder inputs and positions, but
        // executes the same row-local routed banks as the ordinary target.
        // Reuse those exact architecture declarations; encoder hooks remain
        // outside this decoder contract.
        let ordinary=<Self as LayeredArchitecture<B,S>>::prefill_observation_declarations(self,metadata_context)?;
        metadata.controls::<(Vec<eredu_runtime::layered::PrefillObservationDeclaration>,usize)>()?;
        let selected = ordinary.iter().filter(|declaration| declaration.flattens_batch_tokens());
        metadata.borrowed_controls(&selected)?;
        let count = selected.count();
        if let Some(context) = metadata.context() { context.reserve_metadata_vec(&mut declarations, count)?; }
        let selected = ordinary.into_iter().filter(|declaration| declaration.flattens_batch_tokens());
        metadata.borrowed_controls(&selected)?;
        declarations.extend(selected);
        Ok(declarations)
    }

    fn prefill_observation_declarations(
        &self, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Self::Error> {
        let metadata=crate::decoder::identity::Metadata::new(metadata_context);
        metadata.controls::<(&Self,Option<&eredu_nn::workspace::WorkspaceContext>,usize,usize,String,Vec<eredu_runtime::layered::PrefillObservationDeclaration>,std::ops::Range<usize>,Option<eredu_runtime::RoutedObservationPoints>,Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>,Error>)>()?;

        // The ordinary target is group two; optional media groups do not share
        // this row contract. Shared consumers read the earlier publisher's
        // causal history at the captured submission-start query offset.
        let units = <Self as LayeredArchitecture<B, S>>::group_unit_count(self, 2, metadata_context)?;
        let mut declarations = crate::decoder::ordinary_prefill_observation_declarations(
            (0..units).map(|index| <Self as LayeredArchitecture<B, S>>::unit_path(self, 2, index, metadata_context)),
            true, metadata_context)?;
        // Same target bank invocation as observed execution; its expert equations are row-local.
        for index in 0..units {
            let path = <Self as LayeredArchitecture<B, S>>::unit_path(self, 2, index, metadata_context)?;
            if self.args.text.layer_schedule.get(index).is_some_and(|policy| policy.feed_forward == crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe) {
                crate::decoder::append_routed_prefill_path(&mut declarations, &metadata.format(format_args!("{path}.routing"))?, metadata_context)?;
            }
        }
        Ok(declarations)
    }

    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
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
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let pass = <Self as RoutedLayeredArchitecture<B, S>>::expert_pass_for_unit(
            self, group, index, hidden, forward,
        );
        self.forward_components(
            group,
            index,
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
    fn finish_forward_observed<O>(
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
        self.finish_components(hidden, None, context, observer)
    }

    type Input<'a> = ModelInput<'a, B::Tensor>;

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        crate::prefill::segmented_token_shape(input.parts.iter().map(|part| match part {
            DecoderInputPart::Text(tokens)
            | DecoderInputPart::Image(tokens)
            | DecoderInputPart::Video(tokens)
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
        self.canonical_group_transport(group)
    }

    fn group_transport_matches(&self, group: usize, expected: &eredu_runtime::ArchitectureGroupTransport) -> bool {
        match self.source.as_ref() {
            Some(_) => self.checked_graph(crate::decoder::identity::Metadata::new(None))
                .ok().and_then(|graph|graph.transports.get(group)).is_some_and(|value|value==expected),
            None => self.canonical_group_transport(group)==*expected,
        }
    }



    fn primary_execution_group(&self) -> &str {
        TEXT_EXECUTION_GROUP
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        crate::transport::pipeline_state(2, layout)
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Self::Error> {
        Ok(eredu_runtime::ArchitectureExecutionGraph::borrowed(
            &self.execution_graph,
        ))
    }

    fn group_unit_count(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<usize, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        if let Some(context) = metadata_context {
            let graph = self.checked_graph(crate::decoder::identity::Metadata::new(Some(context)))?;
            return graph.description.unit_layout().group_range(group).map(|range| range.len())
                .ok_or_else(|| metadata.error(format_args!("Gemma 4 has three execution groups")));
        }
        match group {
            0 => Ok(self
                .args
                .vision
                .as_ref()
                .map_or(0, |config| config.num_hidden_layers as usize)),
            1 => Ok(self
                .args
                .audio
                .as_ref()
                .map_or(0, |config| config.num_hidden_layers as usize)),
            2 => Ok(self.args.text.num_hidden_layers()),
            _ => Err(metadata.error(format_args!("{}", "Gemma 4 has three execution groups"))),
        }
    }

    fn unit_path(&self, group: usize, index: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<String, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        self.validate_unit_index(group,index,metadata)?;
        match group {
            0 => metadata.text(format_args!("model.vision_tower.encoder.layers.{index}")),
            1 => metadata.text(format_args!("model.audio_tower.layers.{index}")),
            2 => metadata.text(format_args!("model.language_model.layers.{index}")),
            _ => unreachable!(),
        }
    }




    fn group_input_observation_path(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Option<String>, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        metadata.optional_path((group == 2).then_some(eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH))
    }

    fn group_output_observation_path(&self, group: usize, metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>) -> Result<Option<String>, Self::Error> {
        let metadata = crate::decoder::ModuleMetadata::destination(metadata_context);
        metadata.controls::<(&Self, usize, Option<&eredu_nn::workspace::WorkspaceContext>)>()?;

        metadata.optional_path(match group {
            0 => Some(eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH),
            1 => Some(eredu_core::AUDIO_PROJECTOR_OUTPUT_OBSERVATION_PATH),
            _ => None,
        })
    }

    fn retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        _ordinal: usize,
    ) -> std::ops::Range<usize> {
        match group {
            2 => index..index + 1,
            _ => 0..0,
        }
    }

    fn state_ordinal(&self, group: usize, index: usize, ordinal: usize) -> usize {
        if group == 2 {
            index
        } else {
            ordinal
        }
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
        self.build_selected_unit(group, index, context)
    }

    fn begin_forward<'a>(&mut self, input: Self::Input<'a>, state: &mut S,
        context: &<B::Tensor as Tensor>::Context)
        -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_forward_with_metadata(input, None, state, context,
            forward::Metadata::new(B::construction_metadata(context)))
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
        let metadata = forward::Metadata::new(forward.metadata.as_ref().or_else(|| B::construction_metadata(context)));
        if group == 2 && forward.media_span {
            return Ok(initial.clone());
        }
        // New retained sources defer independent input projections to the exact
        // existing group owner. Continuations already carry their hidden state.
        if let Some(pending) = &forward.pending_media {
            if group == 0 && forward.vision_initial.is_none() {
                if let Some(input) = pending.vision_input() {
                    let (hidden, state) = self
                        .static_modules
                        .vision
                        .as_mut()
                        .ok_or_else(|| metadata.error(format_args!("Gemma retained source has no vision owner")))?
                        .begin_with_metadata(input, context, metadata)?;
                    forward.vision_initial = Some(hidden);
                    forward.vision_state = Some(state);
                }
            }
            if group == 1 && forward.audio_initial.is_none() {
                if let Some(input) = pending.audio_input() {
                    let (hidden, valid) = self
                        .static_modules
                        .audio
                        .as_mut()
                        .ok_or_else(|| metadata.error(format_args!("Gemma retained source has no audio owner")))?
                        .begin_with_metadata(input, context, metadata)?;
                    forward.audio_initial = Some(hidden);
                    forward.audio_valid = Some(valid);
                }
            }
        }
        match group {
            0 => Ok(forward.vision_initial.as_ref().unwrap_or(initial).clone()),
            1 => Ok(forward.audio_initial.as_ref().unwrap_or(initial).clone()),
            2 => {
                let vision = forward
                    .vision_output
                    .as_ref()
                    .and_then(|_| dependencies.first().copied())
                    .or(forward.vision_output.as_ref());
                let audio = forward
                    .audio_output
                    .as_ref()
                    .and_then(|_| dependencies.get(1).copied())
                    .or(forward.audio_output.as_ref());
                let assembled = self.assemble_with_metadata(&forward.parts, vision, audio, context,
                    forward::Metadata::new(forward.metadata.as_ref()))?;
                let per_layer_tokens = forward
                    .per_layer_token_override
                    .as_ref()
                    .unwrap_or(&assembled.token_ids);
                forward.per_layer_inputs =
                    self.per_layer_inputs(per_layer_tokens, &assembled.embeddings, context)?;
                Ok(assembled.embeddings)
            }
            _ => Err(Error::backend("invalid Gemma 4 execution group")),
        }
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        match group {
            0 => forward.vision_state.is_some(),
            1 => forward.audio_valid.is_some(),
            2 => true,
            _ => false,
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
        let metadata = forward::Metadata::new(forward.metadata.as_ref().or_else(|| B::construction_metadata(context)));
        match (group, unit) {
            (0, Unit::Vision(unit)) => self
                .static_modules
                .vision
                .as_ref()
                .ok_or_else(|| metadata.error(format_args!("Gemma 4 vision static modules are missing")))?
                .forward_layer_with_metadata(
                    unit,
                    hidden,
                    forward
                        .vision_state
                        .as_ref()
                        .ok_or_else(|| metadata.error(format_args!("Gemma 4 vision state is missing")))?,
                    context, metadata,
                ),
            (1, Unit::Audio(unit)) => unit.forward_with_metadata(
                hidden,
                forward
                    .audio_valid
                    .as_deref()
                    .ok_or_else(|| metadata.error(format_args!("Gemma 4 audio extent is missing")))?,
                context, metadata,
            ),
            (2, Unit::Text(unit)) => {
                let (generated_mask, per_layer_input) =
                    self.text_block_inputs(index, hidden, forward, context)?;
                unit.forward(
                    BlockInput {
                        hidden,
                        mask: forward.mask.as_ref().or(generated_mask.as_ref()),
                        cache: Some(
                            state
                                .layer(self.attention_state_ordinal(index, state.layout().len())?)
                                .map_err(Error::backend)?,
                        ),
                        shared: &mut forward.shared,
                        per_layer_input: per_layer_input.as_ref(),
                        rotary_position: Some(RotaryPosition::Offset(forward.position_offset)),
                    },
                    context,
                )
            }
            _ => Err(Error::backend("Gemma 4 unit/group mismatch")),
        }
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        let metadata = forward::Metadata::new(forward.metadata.as_ref().or_else(|| B::construction_metadata(context)));
        match group {
            0 if forward.vision_state.is_some() => {
                let encoded = self
                    .static_modules
                    .vision
                    .as_ref()
                    .expect("validated vision modules")
                    .finish_with_metadata(hidden, forward.vision_state.as_ref().unwrap(), context, metadata)?;
                forward.vision_output = Some(
                    self.static_modules
                        .vision_projection
                        .as_mut()
                        .expect("validated vision projection")
                        .forward(&encoded, context)?,
                );
                Ok(forward.vision_output.as_ref().unwrap().clone())
            }
            1 if forward.audio_valid.is_some() => {
                let encoded = self
                    .static_modules
                    .audio
                    .as_mut()
                    .expect("validated audio modules")
                    .finish_with_metadata(
                        hidden,
                        forward
                            .audio_valid
                            .as_deref()
                            .expect("validated audio extents"),
                        context, metadata,
                    )?;
                forward.audio_output = Some(
                    self.static_modules
                        .audio_projection
                        .as_mut()
                        .expect("validated audio projection")
                        .forward(&encoded, context)?,
                );
                Ok(forward.audio_output.as_ref().unwrap().clone())
            }
            0..=2 => Ok(hidden.clone()),
            _ => Err(Error::backend("invalid Gemma 4 execution group")),
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
        self.static_modules.text.project_logits(
            hidden,
            self.args.text.final_logit_softcapping,
            context,
        )
    }

    fn forward_metadata(&self, forward: &Self::ForwardContext)
        -> Option<eredu_runtime::layered::LayeredMetadata<Error>> {
        forward.metadata.as_ref().map(|context|
            eredu_runtime::layered::LayeredMetadata::new(context, |error| error))
    }

    fn visit_retained_context_values<'a>(&'a self, forward: &'a Self::ForwardContext,
        _group: usize, _index: usize, visitor: &mut dyn FnMut(&'a B::Tensor)) {
        forward.visit_values(visitor);
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = Vec::new();
        forward.visit_values(&mut |value| values.push(value));
        values.into_iter()
    }

}

impl<B, S> ParallelLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
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
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let pass = <Self as RoutedLayeredArchitecture<B, S>>::expert_pass_for_unit(
            self, group, index, hidden, forward,
        );
        self.forward_components_parallel(
            group,
            index,
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
    fn finish_forward_parallel_observed<O>(
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
        self.finish_components(hidden, Some(parallel), context, observer)
    }

    fn begin_forward_parallel<'a>(
        &mut self, input: Self::Input<'a>, state: &mut S,
        parallel: &B::ParallelContext, context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let metadata = forward::Metadata::new(B::construction_metadata(context));
        self.begin_forward_with_metadata(input, Some(parallel), state, context, metadata)
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
        let metadata = forward::Metadata::new(forward.metadata.as_ref().or_else(|| B::construction_metadata(context)));
        match (group, unit) {
            (0, Unit::Vision(unit)) => self
                .static_modules
                .vision
                .as_ref()
                .ok_or_else(|| metadata.error(format_args!("Gemma 4 vision static modules are missing")))?
                .forward_layer_with_metadata(
                    unit,
                    hidden,
                    forward
                        .vision_state
                        .as_ref()
                        .ok_or_else(|| metadata.error(format_args!("Gemma 4 vision state is missing")))?,
                    context, metadata,
                ),
            (1, Unit::Audio(unit)) => unit.forward_with_metadata(
                hidden,
                forward
                    .audio_valid
                    .as_deref()
                    .ok_or_else(|| metadata.error(format_args!("Gemma 4 audio extent is missing")))?,
                context, metadata,
            ),
            (2, Unit::Text(unit)) => self
                .forward_text_unit_parallel(index, unit, hidden, state, forward, parallel, context),
            _ => Err(Error::backend("Gemma 4 parallel unit/group mismatch")),
        }
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
                "Gemma 4 model was not built with local geometry",
            ));
        }
        self.static_modules.text.project_logits_instrumented(
            hidden,
            self.args.text.final_logit_softcapping,
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
}

fn validate_component<T:Tensor>(name:&str,value:Option<&T>,tokens:i32,hidden:i32)->Result<(),Error>{
    forward::validate_component_with_metadata(name,value,tokens,hidden,forward::Metadata::new(None))
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

// Text-only pinned modules are isolated so the composite static tree can add
// media without changing their stable parameter identities.
pub(crate) mod model_text {
    use eredu_nn::{
        EmbeddingSpec, Error, GroupedNeuralBackend, LinearSpec, NormalizationConstructionSpec,
        ParameterSpec, Parameterized, Tensor,
    };

    use super::super::ModelArgs;

    /// Pinned Gemma text modules.
    #[derive(Debug, Clone, Parameterized)]
    #[parameterized(tensor = "B::Tensor")]
    pub struct StaticTextModules<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
        /// Main token embedding table.
        pub embeddings: B::Embedding,
        /// Optional per-layer token identity table.
        pub per_layer_embeddings: Option<B::Embedding>,
        /// Optional decoder-to-per-layer projection.
        pub per_layer_projection: Option<B::Linear>,
        /// Optional projected per-layer normalization.
        pub per_layer_norm: Option<B::Normalization>,
        /// Final decoder norm.
        pub norm: B::Normalization,
        /// Optional untied vocabulary head.
        pub head: Option<B::Linear>,
    }

    impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> StaticTextModules<B> {
        pub(super) fn new(
            args: &ModelArgs,
            context: &<B::Tensor as Tensor>::Context,
        ) -> Result<Self, Error> {
            let metadata=crate::decoder::ModuleMetadata::new::<B>(context);
            metadata.controls::<(Self, EmbeddingSpec, LinearSpec, NormalizationConstructionSpec,
                Option<ParameterSpec>, &ModelArgs, i32, Result<Self,Error>)>()?;
            let embedding = "model.language_model.embed_tokens.weight";
            let per_layer_width = i32::try_from(args.num_hidden_layers()).ok()
                .and_then(|layers|args.hidden_size_per_layer_input.checked_mul(layers))
                .ok_or_else(||metadata.error(format_args!("Gemma 4 per-layer width overflowed")))?;
            let per_layer_embedding = "model.language_model.embed_tokens_per_layer.weight";
            let per_layer_projection = "model.language_model.per_layer_model_projection.weight";
            let head = "lm_head.weight";
            Ok(Self {
                embeddings: B::embedding(
                    EmbeddingSpec {
                        vocabulary: args.vocab_size,
                        dimensions: args.hidden_size,
                        weight: metadata.plain_parameter(embedding)?,
                        format: metadata.format(
                            embedding,
                            args.linear_format_for(embedding),
                        )?,
                    },
                    context,
                )?,
                per_layer_embeddings: (args.hidden_size_per_layer_input > 0)
                    .then(|| {
                        B::embedding(
                            EmbeddingSpec {
                                vocabulary: args
                                    .vocab_size_per_layer_input
                                    .unwrap_or(args.vocab_size),
                                dimensions: per_layer_width,
                                weight: metadata.plain_parameter(per_layer_embedding)?,
                                format: metadata.format(
                                    per_layer_embedding,
                                    args.linear_format_for(per_layer_embedding),
                                )?,
                            },
                            context,
                        )
                    })
                    .transpose()?,
                per_layer_projection: (args.hidden_size_per_layer_input > 0)
                    .then(|| {
                        B::linear(
                            LinearSpec {
                                input: args.hidden_size,
                                output: per_layer_width,
                                weight: metadata.plain_parameter(per_layer_projection)?,
                                bias: None,
                                format: metadata.format(
                                    per_layer_projection,
                                    args.linear_format_for(per_layer_projection),
                                )?,
                            },
                            context,
                        )
                    })
                    .transpose()?,
                per_layer_norm: (args.hidden_size_per_layer_input > 0)
                    .then(|| {
                        B::normalization(
                            NormalizationConstructionSpec::learned(
                                args.hidden_size_per_layer_input,
                                args.rms_norm_eps,
                                metadata.plain_parameter(
                                    "model.language_model.per_layer_projection_norm.weight",
                                )?,
                            ),
                            context,
                        )
                    })
                    .transpose()?,
                norm: B::normalization(
                    NormalizationConstructionSpec::learned(
                        args.hidden_size,
                        args.rms_norm_eps,
                        metadata.plain_parameter("model.language_model.norm.weight")?,
                    ),
                    context,
                )?,
                head: (!args.tie_word_embeddings)
                    .then(|| {
                        B::linear(
                            LinearSpec {
                                input: args.hidden_size,
                                output: args.vocab_size,
                                weight: metadata.plain_parameter(head)?,
                                bias: None,
                                format: metadata.format(
                                    head,
                                    args.linear_format_for(head),
                                )?,
                            },
                            context,
                        )
                    })
                    .transpose()?,
            })
        }

        pub(super) fn new_parallel(
            args: &ModelArgs,
            geometry: &super::super::LocalGeometry,
            context: &<B::Tensor as Tensor>::Context,
        ) -> Result<Self, Error> {
            let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
            metadata.controls::<(Self, &ModelArgs, &super::super::LocalGeometry,
                i32, i32, EmbeddingSpec, LinearSpec, NormalizationConstructionSpec,
                eredu_nn::VocabularyParallelRange, Result<Self, Error>)>()?;
            let embedding = "model.language_model.embed_tokens.weight";
            let per_layer_width = args.hidden_size_per_layer_input;
            let combined_width = per_layer_width
                .checked_mul(args.num_hidden_layers() as i32)
                .ok_or_else(|| Error::backend("Gemma 4 per-layer width overflowed"))?;
            let per_layer_embedding = "model.language_model.embed_tokens_per_layer.weight";
            let per_layer_projection = "model.language_model.per_layer_model_projection.weight";
            let head = "lm_head.weight";
            Ok(Self {
                embeddings: B::vocabulary_parallel_embedding(
                    EmbeddingSpec {
                        vocabulary: args.vocab_size,
                        dimensions: args.hidden_size,
                        weight: metadata.plain_parameter(embedding)?,
                        format: metadata.format(
                            embedding,
                            args.linear_format_for(embedding),
                        )?,
                    },
                    geometry.embedding_range().clone(),
                    context,
                )?,
                per_layer_embeddings: (per_layer_width > 0)
                    .then(|| {
                        B::embedding(
                            EmbeddingSpec {
                                vocabulary: args
                                    .vocab_size_per_layer_input
                                    .unwrap_or(args.vocab_size),
                                dimensions: combined_width,
                                weight: metadata.plain_parameter(per_layer_embedding)?,
                                format: metadata.format(
                                    per_layer_embedding,
                                    args.linear_format_for(per_layer_embedding),
                                )?,
                            },
                            context,
                        )
                    })
                    .transpose()?,
                per_layer_projection: (per_layer_width > 0)
                    .then(|| {
                        B::linear(
                            LinearSpec {
                                input: args.hidden_size,
                                output: combined_width,
                                weight: metadata.plain_parameter(per_layer_projection)?,
                                bias: None,
                                format: metadata.format(
                                    per_layer_projection,
                                    args.linear_format_for(per_layer_projection),
                                )?,
                            },
                            context,
                        )
                    })
                    .transpose()?,
                per_layer_norm: (per_layer_width > 0)
                    .then(|| {
                        B::normalization(
                            NormalizationConstructionSpec::learned(
                                per_layer_width,
                                args.rms_norm_eps,
                                metadata.plain_parameter(
                                    "model.language_model.per_layer_projection_norm.weight",
                                )?,
                            ),
                            context,
                        )
                    })
                    .transpose()?,
                norm: B::normalization(
                    NormalizationConstructionSpec::learned(
                        args.hidden_size,
                        args.rms_norm_eps,
                        metadata.plain_parameter("model.language_model.norm.weight")?,
                    ),
                    context,
                )?,
                head: if args.tie_word_embeddings {
                    None
                } else {
                    Some(B::vocabulary_parallel_linear(
                        LinearSpec {
                            input: args.hidden_size,
                            output: args.vocab_size,
                            weight: metadata.plain_parameter(head)?,
                            bias: None,
                            format: metadata.format(
                                head,
                                args.linear_format_for(head),
                            )?,
                        },
                        geometry.output_range().cloned().ok_or_else(|| {
                            Error::backend("untied Gemma 4 output has no local range")
                        })?,
                        context,
                    )?)
                },
            })
        }

        pub(super) fn project_logits(
            &mut self,
            hidden: &B::Tensor,
            cap: Option<f32>,
            context: &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error> {
            self.project_logits_instrumented(
                hidden,
                cap,
                None,
                context,
                &mut crate::decoder::ComponentInstrumentation::disabled(),
            )
        }

        pub(super) fn project_logits_instrumented(
            &mut self,
            hidden: &B::Tensor,
            cap: Option<f32>,
            parallel: Option<&B::ParallelContext>,
            context: &<B::Tensor as Tensor>::Context,
            instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error> {
            let hidden = instrumentation.normalize_readout(hidden, &mut self.norm, context)?;
            let logits = match (parallel, self.head.as_mut()) {
                (Some(parallel), Some(head)) => instrumentation.project_vocabulary::<B>(
                    "projection_input",
                    head,
                    &hidden,
                    parallel,
                    context,
                )?,
                (Some(parallel), None) => instrumentation.project_vocabulary_embedding::<B>(
                    "projection_input",
                    &mut self.embeddings,
                    &hidden,
                    parallel,
                    context,
                )?,
                (None, Some(head)) => instrumentation.project::<B>(
                    "projection_input",
                    head,
                    &hidden,
                    None,
                    context,
                )?,
                (None, None) => instrumentation.project_embedding(
                    "projection_input",
                    &mut self.embeddings,
                    &hidden,
                    context,
                )?,
            };
            let logits = instrumentation.apply("linear", logits)?;
            args_cap::<B>(logits, cap, context)
        }
    }

    pub(super) fn args_cap<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        logits: B::Tensor,
        cap: Option<f32>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match cap {
            Some(cap) => logits
                .multiply_scalar(cap.recip(), context)?
                .tanh(context)?
                .multiply_scalar(cap, context),
            None => Ok(logits),
        }
    }
}


fn external_capture_paths(
    request: &ExternalPredictionCaptureRequest,
    metadata: crate::decoder::ModuleMetadata<'_>,
) -> Result<Option<Vec<String>>, Error> {
    if !matches!(request, ExternalPredictionCaptureRequest::Gemma4SharedAttention { .. }) {
        return Ok(None);
    }
    request.collect_paths(metadata).map(Some)
}

fn external_capture<T: Clone>(
    request: &ExternalPredictionCaptureRequest,
    forward: &ForwardContext<T>,
    observed: Vec<T>,
    metadata: crate::decoder::ModuleMetadata<'_>,
) -> Result<Option<ExternalPredictionTargetCapture<T>>, Error> {
    metadata.controls::<(Vec<T>, ExternalPredictionTargetCapture<T>, &ForwardContext<T>)>()?;
    if !matches!(request, ExternalPredictionCaptureRequest::Gemma4SharedAttention { .. }) {
        return Ok(None);
    }
    let [hidden]: [T; 1] = observed.try_into().map_err(|observed: Vec<T>| {
        metadata.error(format_args!(
            "Gemma 4 assistant capture expected one final hidden state, received {}", observed.len()
        ))
    })?;
    let shared = forward.shared_attention_states();
    let mut shared_kv = metadata.vector(shared.iter().count())?;
    for (policy, (keys, values)) in shared.iter() {
        metadata.controls::<(eredu_core::AttentionPolicy, T, T)>()?;
        shared_kv.push((*policy, keys.clone(), values.clone()));
    }
    Ok(Some(ExternalPredictionTargetCapture::Gemma4 { hidden, shared_kv }))
}

pub(super) mod forward;
pub use forward::SharedAttentionPublications;

fn prepared_group_waves<T:Tensor>(input:PreparedCompositeInput<'_,T,Gemma4InputPartPlan>,
    hidden:i32, group:usize, first_active:Option<usize>,tensor_partitions:usize,pipeline_stages:usize,
    destination:crate::composite_execution::graph::Destination<'_>)
    ->Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>,Error> {
    use crate::composite_execution::CompositeTensorCollective;
    destination.controls::<(PreparedCompositeInput<'_,T,Gemma4InputPartPlan>,
        Option<Vec<Vec<CompositeTensorCollective>>>,Vec<CompositeTensorCollective>,
        Option<usize>,usize,i32)>()?;
    if group>=2 || tensor_partitions<=1 || pipeline_stages<=1 {return Ok(None);}
    let ingress=if first_active==Some(group) {
        let parts=input.admitted().gemma_parts();
        destination.controls::<Gemma4InputPartPlan>()?;
        destination.try_collect(input.prepared().parts().iter().zip(parts).filter_map(|(part,plan)|{
            match plan {
                Gemma4InputPartPlan::TextTokens{positions}=>Some((|| {
                    let positions=i32::try_from(positions).map_err(|_|destination.error(format_args!(
                        "Gemma text extent exceeds i32")))?;
                    Ok(CompositeTensorCollective::Sum{shape:destination.collect([
                        part.payload().value().dim(0),positions,hidden])?})
                })()),
                _=>None,
            }
        }))?
    }else{destination.vector(0)?};
    let mut waves=destination.collect((0..pipeline_stages).map(|_|Vec::new()))?;
    waves[0]=ingress;Ok(Some(waves))
}
fn prepared_ingress_waves<T>(input:PreparedCompositeInput<'_,T,Gemma4InputPartPlan>,hidden:i32,
    tensor_partitions:usize,destination:crate::composite_execution::graph::Destination<'_>)
    ->Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>,Error> {
    let source=input.admitted();
    destination.controls::<Gemma4InputPartPlan>()?;
    crate::composite_execution::segmented_token_ingress_collectives_in(
        source.gemma_parts().filter_map(|part|match part{
            Gemma4InputPartPlan::TextTokens{positions}=>Some(positions),_=>None,
        }),hidden,tensor_partitions,destination)
}
