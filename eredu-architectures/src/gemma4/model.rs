//! Backend-neutral Gemma 4 multimodal model and layered runtime lifecycle.

use std::collections::HashMap;

use eredu_nn::{
    multimodal::{assemble_ordered_inputs, OrderedInputPart},
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, GroupedNeuralBackend, Index,
    LinearOperator, NormalizationOperator, PadMode, Parameterized, RotaryPosition, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout,
    ExpertPass, LayerRuntimeState, LayeredArchitecture, LayeredForwardState, LayeredPartitionInput,
    LayeredPartitionOutput, OwnedParameterGroupSpec, ParallelLayeredArchitecture,
    ParallelPlanError, ParallelRoutedLayeredArchitecture, ParameterGroupOwner,
    PartitionedLayeredArchitecture, RoutedExpertProvider, RoutedLayeredArchitecture, StateLayout,
};

use super::{
    audio_layer_parameter_groups, audio_static_parameter_groups, layer_parameter_groups,
    modality_projection_parameter_groups, state_layout, static_parameter_groups,
    vision_layer_parameter_groups, vision_static_parameter_groups, AudioIngressBatchPlan,
    AudioIngressPartPlan, AudioInput, AudioLayer, AudioStatic, BlockInput, DenseBlock,
    FamilyConfig, LocalGeometry, ModalityProjector, SharedAttentionStates, VisionIngressBatchPlan,
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
    type Boundary = TextBoundarySchema;

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        let geometry = self
            .parallel_geometry()
            .ok_or_else(|| Error::backend("Gemma pipeline boundary requires parallel geometry"))?;
        Ok(TextBoundarySchema::from_args(&self.args().text, geometry))
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
            LayeredPartitionInput::Hidden { hidden, auxiliary } => self.resume_pipeline_text(
                hidden,
                mask.cloned(),
                auxiliary.per_layer_input,
                state,
                expected,
                first_state_ordinal,
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
            LayeredPartitionInput::Hidden { hidden, auxiliary } => self.resume_pipeline_text(
                hidden,
                mask.cloned(),
                auxiliary.per_layer_input,
                state,
                expected,
                first_state_ordinal,
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
                auxiliary: TextBoundary::new(forward.pipeline_per_layer_inputs().cloned()),
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
}

impl<T> PreparedCompositeIngress<T> {
    /// Borrows ordered decoder segments with architecture-created placeholders.
    pub fn decoder_parts(&self) -> Vec<DecoderInputPart<'_, T>> {
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
            .collect()
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

    let prepared = input.prepared();
    let admitted = input.admitted();
    if prepared.identity() != admitted.identity() || prepared.len() != admitted.parts().len() {
        return Err(Error::backend(
            "Gemma 4 prepared input no longer matches its admission",
        ));
    }

    let mut tokens = Vec::with_capacity(prepared.len());
    let mut modalities = Vec::with_capacity(prepared.len());
    let mut projected = Vec::with_capacity(prepared.len());
    let mut vision_parts = Vec::new();
    let mut audio_parts = Vec::new();
    for (part, plan) in prepared.parts().iter().zip(admitted.parts()) {
        match plan {
            Gemma4InputPartPlan::TextTokens { .. } => {
                let eredu_runtime::PreparedInputPayload::TokenIds(value) = part.payload() else {
                    return Err(Error::backend(
                        "Gemma 4 admitted text part lost its token payload",
                    ));
                };
                tokens.push(value.clone());
                modalities.push(eredu_core::InputModality::Text);
                projected.push(None);
            }
            Gemma4InputPartPlan::Projected {
                modality,
                placeholder_token_id,
                positions,
            } => {
                let eredu_runtime::PreparedInputPayload::Embeddings(value) = part.payload() else {
                    return Err(Error::backend(
                        "Gemma 4 admitted projected part lost its embedding payload",
                    ));
                };
                let count = i32::try_from(*positions)
                    .map_err(|_| Error::backend("Gemma 4 projected span exceeds I32"))?;
                let token = i32::try_from(*placeholder_token_id)
                    .map_err(|_| Error::backend("Gemma 4 placeholder ID exceeds I32"))?;
                tokens.push(B::Tensor::full_i32(token, &[1, count], context)?);
                modalities.push(*modality);
                projected.push(Some(value.clone()));
            }
            Gemma4InputPartPlan::Vision {
                placeholder_token_id,
                ingress,
                ..
            } => {
                let eredu_runtime::PreparedInputPayload::Tensor(value) = part.payload() else {
                    return Err(Error::backend(
                        "Gemma 4 admitted vision part lost its tensor payload",
                    ));
                };
                let positions = part
                    .metadata_value(eredu_core::InputMetadataKey::PatchPositions)
                    .ok_or_else(|| Error::backend("Gemma 4 vision positions disappeared"))?;
                let token = i32::try_from(*placeholder_token_id)
                    .map_err(|_| Error::backend("Gemma 4 placeholder ID exceeds I32"))?;
                tokens.push(B::Tensor::full_i32(
                    token,
                    &[1, ingress.decoder_positions],
                    context,
                )?);
                modalities.push(part.modality());
                projected.push(None);
                vision_parts.push(VisionPart {
                    patches: value.clone(),
                    positions: positions.clone(),
                    plan: ingress.clone(),
                });
            }
            Gemma4InputPartPlan::Audio {
                placeholder_token_id,
                ingress,
                ..
            } => {
                let eredu_runtime::PreparedInputPayload::Tensor(value) = part.payload() else {
                    return Err(Error::backend(
                        "Gemma 4 admitted audio part lost its tensor payload",
                    ));
                };
                let token = i32::try_from(*placeholder_token_id)
                    .map_err(|_| Error::backend("Gemma 4 placeholder ID exceeds I32"))?;
                tokens.push(B::Tensor::full_i32(
                    token,
                    &[1, ingress.decoder_positions],
                    context,
                )?);
                modalities.push(eredu_core::InputModality::Audio);
                projected.push(None);
                audio_parts.push(AudioPart {
                    features: value.clone(),
                    plan: ingress.clone(),
                });
            }
        }
    }

    let vision_plan = (!vision_parts.is_empty())
        .then(|| {
            VisionIngressBatchPlan::new(
                &vision_parts
                    .iter()
                    .map(|part| part.plan.clone())
                    .collect::<Vec<_>>(),
            )
        })
        .transpose()
        .map_err(|error| Error::backend(error.to_string()))?;
    let vision = if let Some(plan) = vision_plan {
        let patches = vision_parts
            .iter()
            .map(|part| {
                let extra = plan.padded_patches - part.patches.dim(1);
                if extra < 0 {
                    return Err(Error::backend(
                        "Gemma 4 vision payload exceeds admitted batch padding",
                    ));
                }
                B::Tensor::pad(
                    &part.patches,
                    &[(0, 0), (0, extra), (0, 0)],
                    PadMode::Constant,
                    context,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let positions = vision_parts
            .iter()
            .map(|part| {
                let extra = plan.padded_patches - part.positions.dim(1);
                if extra < 0 {
                    return Err(Error::backend(
                        "Gemma 4 vision positions exceed admitted batch padding",
                    ));
                }
                B::Tensor::pad(
                    &part.positions,
                    &[(0, 0), (0, extra), (0, 0)],
                    PadMode::Constant,
                    context,
                )?
                .maximum_i32(0, context)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Some(PreparedVision {
            patches: B::Tensor::concatenate(&patches, 0, context)?,
            positions: B::Tensor::concatenate(&positions, 0, context)?,
            valid: B::Tensor::from_f32_slice(
                &plan.position_valid_values,
                &plan.position_valid_shape(),
                context,
            )?,
            key_mask: B::Tensor::from_f32_slice(
                &plan.key_mask_values,
                &plan.key_mask_shape(),
                context,
            )?,
            grid_extents: plan.grid_extents,
        })
    } else {
        None
    };

    let audio_plan = (!audio_parts.is_empty())
        .then(|| {
            AudioIngressBatchPlan::new(
                &audio_parts
                    .iter()
                    .map(|part| part.plan.clone())
                    .collect::<Vec<_>>(),
            )
        })
        .transpose()
        .map_err(|error| Error::backend(error.to_string()))?;
    let audio = if let Some(plan) = audio_plan {
        let features = audio_parts
            .iter()
            .map(|part| {
                let extra = plan.padded_frames - part.features.dim(1);
                if extra < 0 {
                    return Err(Error::backend(
                        "Gemma 4 audio payload exceeds admitted batch padding",
                    ));
                }
                B::Tensor::pad(
                    &part.features,
                    &[(0, 0), (0, extra), (0, 0)],
                    PadMode::Constant,
                    context,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input_mask_values = audio_parts
            .iter()
            .flat_map(|part| {
                (0..plan.padded_frames).map(|frame| {
                    if frame < part.plan.valid_frames {
                        1.0
                    } else {
                        0.0
                    }
                })
            })
            .collect::<Vec<_>>();
        Some(PreparedAudio {
            features: B::Tensor::concatenate(&features, 0, context)?,
            input_mask: B::Tensor::from_f32_slice(
                &input_mask_values,
                &[audio_parts.len() as i32, plan.padded_frames, 1],
                context,
            )?,
            first_stage_mask: B::Tensor::from_f32_slice(
                &plan.first_stage_mask_values,
                &plan.first_stage_mask_shape(),
                context,
            )?,
            valid: plan.valid_subsampled_frames,
        })
    } else {
        None
    };

    Ok(PreparedCompositeIngress {
        tokens,
        modalities,
        projected,
        vision,
        audio,
    })
}

impl<B, S> CompositeArchitecture<B, S> for LayeredModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    type InputPartPlan = Gemma4InputPartPlan;
    type AdmissionConfig = FamilyConfig;

    fn admission_config(&self) -> Self::AdmissionConfig {
        self.args.clone()
    }

    fn external_assistant_target_profile(
        config: &Self::AdmissionConfig,
    ) -> Option<crate::external_assistant::ExternalAssistantTargetProfile> {
        Some(crate::external_assistant::ExternalAssistantTargetProfile::Gemma4(config.clone()))
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

    fn should_execute_prepared_group(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> bool {
        match group {
            0 => input
                .admitted()
                .parts()
                .iter()
                .any(|part| matches!(part, Gemma4InputPartPlan::Vision { .. })),
            1 => input
                .admitted()
                .parts()
                .iter()
                .any(|part| matches!(part, Gemma4InputPartPlan::Audio { .. })),
            2 => true,
            _ => false,
        }
    }

    fn external_prediction_capture_paths(
        request: &ExternalPredictionCaptureRequest,
    ) -> Result<Option<Vec<String>>, Self::Error> {
        match request {
            ExternalPredictionCaptureRequest::Gemma4SharedAttention { final_hidden_path } => {
                Ok(Some(vec![final_hidden_path.clone()]))
            }
            _ => Ok(None),
        }
    }

    fn external_prediction_capture(
        request: &ExternalPredictionCaptureRequest,
        forward: &Self::ForwardContext,
        observed: Vec<B::Tensor>,
    ) -> Result<Option<ExternalPredictionTargetCapture<B::Tensor>>, Self::Error> {
        if !matches!(
            request,
            ExternalPredictionCaptureRequest::Gemma4SharedAttention { .. }
        ) {
            return Ok(None);
        }
        let [hidden]: [B::Tensor; 1] = observed.try_into().map_err(|observed: Vec<_>| {
            Error::backend(format!(
                "Gemma 4 assistant capture expected one final hidden state, received {}",
                observed.len()
            ))
        })?;
        let shared_kv = forward
            .shared_attention_states()
            .iter()
            .map(|(policy, (keys, values))| (*policy, keys.clone(), values.clone()))
            .collect();
        Ok(Some(ExternalPredictionTargetCapture::Gemma4 {
            hidden,
            shared_kv,
        }))
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
        let prepared = prepare_composite_ingress::<B>(input, context)?;
        let decoder_parts = prepared.decoder_parts();
        <Self as LayeredArchitecture<B, S>>::begin_forward(
            self,
            ModelInput {
                parts: &decoder_parts,
                vision: prepared.vision_input(),
                audio: prepared.audio_input(),
                per_layer_tokens: None,
                mask: None,
            },
            state,
            context,
        )
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
        let prepared = prepare_composite_ingress::<B>(input, context)?;
        let decoder_parts = prepared.decoder_parts();
        <Self as ParallelLayeredArchitecture<B, S>>::begin_forward_parallel(
            self,
            ModelInput {
                parts: &decoder_parts,
                vision: prepared.vision_input(),
                audio: prepared.audio_input(),
                per_layer_tokens: None,
                mask: None,
            },
            state,
            parallel,
            context,
        )
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
    shared: SharedAttentionStates<T>,
    vision_state: Option<VisionState<T>>,
    vision_initial: Option<T>,
    vision_output: Option<T>,
    audio_valid: Option<Vec<i32>>,
    audio_initial: Option<T>,
    audio_output: Option<T>,
}

impl<T> ForwardContext<T> {
    /// Pass-local key/value publications consumed by shared-state layers and
    /// external assistants.
    pub fn shared_attention_states(&self) -> &SharedAttentionStates<T> {
        &self.shared
    }

    /// Returns the decoder-wide per-layer input tensor transported between
    /// pipeline stages, when the checkpoint declares one.
    pub fn pipeline_per_layer_inputs(&self) -> Option<&T> {
        self.per_layer_inputs.as_ref()
    }
}

/// Family-owned schema for decoder-wide per-layer input transport.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TextBoundarySchema {
    hidden_size: i32,
    per_layer_geometry: Option<(i32, i32)>,
}

impl TextBoundarySchema {
    /// Derives the schema from normalized text and rank-local geometry.
    pub fn from_args(args: &super::ModelArgs, geometry: &LocalGeometry) -> Self {
        Self {
            hidden_size: args.hidden_size,
            per_layer_geometry: (args.hidden_size_per_layer_input > 0)
                .then(|| (args.num_hidden_layers() as i32, geometry.per_layer_width())),
        }
    }

    /// Derives the same wire schema from exact TP/PP-local construction geometry.
    pub fn from_partition_args(
        args: &super::ModelArgs,
        geometry: &super::PartitionLocalGeometry,
    ) -> Self {
        Self {
            hidden_size: args.hidden_size,
            per_layer_geometry: (args.hidden_size_per_layer_input > 0).then(|| {
                (
                    args.num_hidden_layers() as i32,
                    geometry.per_layer_range().end - geometry.per_layer_range().start,
                )
            }),
        }
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
        self.per_layer_geometry
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
            .collect()
    }

    /// Decodes the optional per-layer input without positional guessing.
    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<Self::Boundary<T>, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        Ok(TextBoundary {
            per_layer_input: tensors.into_iter().next(),
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
        let actual = usize::from(boundary.per_layer_input.is_some());
        let expected = usize::from(self.per_layer_geometry.is_some());
        if actual != expected {
            return Err(eredu_runtime::ArchitectureBoundaryError::TensorCount {
                boundary: "gemma4.text",
                expected,
                actual,
            });
        }
        boundary
            .per_layer_input
            .into_iter()
            .map(|tensor| eredu_runtime::ArchitectureBoundaryValue::new("per_layer_input", tensor))
            .collect()
    }
}

/// Typed Gemma text context transported alongside decoder activations.
pub struct TextBoundary<T> {
    /// Decoder-wide per-layer input, when configured by the checkpoint.
    pub per_layer_input: Option<T>,
}

impl<T> TextBoundary<T> {
    /// Creates a typed text boundary.
    pub const fn new(per_layer_input: Option<T>) -> Self {
        Self { per_layer_input }
    }
}

/// One Gemma architecture used by resident, layerwise, and streamed runtimes.
pub struct LayeredModel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    args: FamilyConfig,
    static_modules: StaticModules<B>,
    parallel_geometry: Option<LocalGeometry>,
    partition_state_offset: usize,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_runtime::ArchitectureParameters<B> for LayeredModel<B>
{
    type DefinitionError = Error;

    fn state_layout(&self) -> Result<StateLayout, Self::DefinitionError> {
        self.state_layout_impl()
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::DefinitionError> {
        super::state_identity(
            &self.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
        .map_err(|error| Error::backend(error.to_string()))
    }

    fn parameter_description(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Self::DefinitionError> {
        self.parameter_description_impl(context)
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
        Self::with_text(args, text, Some(geometry), context)
    }

    fn with_text(
        args: FamilyConfig,
        text: model_text::StaticTextModules<B>,
        parallel_geometry: Option<LocalGeometry>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let vision = args
            .vision
            .as_ref()
            .map(|config| VisionStatic::new(config.clone(), context))
            .transpose()?;
        let vision_projection = args
            .vision
            .as_ref()
            .map(|config| {
                ModalityProjector::new(
                    &args.text,
                    "embed_vision",
                    config.hidden_size,
                    config.rms_norm_eps,
                    context,
                )
            })
            .transpose()?;
        let audio = args
            .audio
            .as_ref()
            .map(|config| AudioStatic::new(config.clone(), context))
            .transpose()?;
        let audio_projection = args
            .audio
            .as_ref()
            .map(|config| {
                ModalityProjector::new(
                    &args.text,
                    "embed_audio",
                    config.output_proj_dims,
                    config.rms_norm_eps,
                    context,
                )
            })
            .transpose()?;
        Ok(Self {
            args,
            static_modules: StaticModules {
                text,
                vision,
                vision_projection,
                audio,
                audio_projection,
            },
            parallel_geometry,
            partition_state_offset: 0,
        })
    }

    /// Retains the architecture-global ordinal of this pipeline partition's first state row.
    pub(crate) fn with_partition_state_offset(mut self, offset: usize) -> Result<Self, Error> {
        if offset >= self.args.text.num_hidden_layers() {
            return Err(Error::backend(
                "Gemma 4 partition state offset is outside the decoder",
            ));
        }
        self.partition_state_offset = offset;
        Ok(self)
    }

    fn local_state_ordinal(&self, global: usize) -> Result<usize, Error> {
        global
            .checked_sub(self.partition_state_offset)
            .ok_or_else(|| Error::backend("Gemma 4 unit precedes the partition state offset"))
    }

    fn validate_partition_state<S: LayerRuntimeState<B>>(&self, state: &S) -> Result<(), Error> {
        let complete = self.state_layout_impl()?;
        let end = self
            .partition_state_offset
            .checked_add(state.layout().len())
            .ok_or_else(|| Error::backend("Gemma 4 partition state interval overflow"))?;
        let expected = complete
            .slice(self.partition_state_offset..end)
            .map_err(Error::backend)?;
        if state.layout() != &expected {
            return Err(Error::backend("Gemma 4 rank-local state layout mismatch"));
        }
        Ok(())
    }

    fn partition_position_offset<S: LayerRuntimeState<B>>(
        &self,
        state: &mut S,
    ) -> Result<i32, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let mut position_offset = 0;
        for (local, policy) in self
            .args
            .text
            .layer_schedule
            .iter()
            .skip(self.partition_state_offset)
            .take(state.layout().len())
            .enumerate()
        {
            if policy.key_value.owns_state() {
                position_offset = position_offset.max(AttentionCache::<B::Tensor>::offset(
                    state.layer(local).map_err(Error::backend)?,
                ));
            }
        }
        Ok(position_offset)
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

    /// Returns the normalized composite family configuration.
    pub const fn args(&self) -> &FamilyConfig {
        &self.args
    }

    /// Describes every pinned/media/text parameter with explicit canonical
    /// execution ownership.
    fn parameter_description_impl(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root(VISION_EXECUTION_GROUP),
                ExecutionGroupSpec::root(AUDIO_EXECUTION_GROUP),
                ExecutionGroupSpec::with_dependencies(
                    TEXT_EXECUTION_GROUP,
                    [VISION_EXECUTION_GROUP, AUDIO_EXECUTION_GROUP],
                ),
            ],
            TEXT_EXECUTION_GROUP,
        )
        .map_err(Error::backend)?;
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
        self.parallel_geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .map_or_else(|| state_layout(&self.args.text).map_err(Error::backend), Ok)
    }

    /// Returns planner-derived geometry for a rank-local realization.
    pub const fn parallel_geometry(&self) -> Option<&LocalGeometry> {
        self.parallel_geometry.as_ref()
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
        for (layer, policy) in self.args.text.layer_schedule.iter().enumerate() {
            if policy.key_value.owns_state() {
                position_offset = position_offset.max(AttentionCache::<B::Tensor>::offset(
                    state.layer(layer).map_err(Error::backend)?,
                ));
            }
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
                shared: HashMap::new(),
                vision_state: None,
                vision_initial: None,
                vision_output: None,
                audio_valid: None,
                audio_initial: None,
                audio_output: None,
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
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout() != expected {
            return Err(Error::backend("Gemma 4 pipeline state layout mismatch"));
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
        for (layer, policy) in self
            .args
            .text
            .layer_schedule
            .iter()
            .enumerate()
            .skip(first_state_ordinal)
            .take(expected.len())
        {
            if policy.key_value.owns_state() {
                position_offset = position_offset.max(AttentionCache::<B::Tensor>::offset(
                    state
                        .layer(layer - first_state_ordinal)
                        .map_err(Error::backend)?,
                ));
            }
        }
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                position_offset,
                parts,
                per_layer_token_override: None,
                per_layer_inputs: None,
                shared: HashMap::new(),
                vision_state: None,
                vision_initial: None,
                vision_output: None,
                audio_valid: None,
                audio_initial: None,
                audio_output: None,
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
        first_state_ordinal: usize,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout() != expected {
            return Err(Error::backend("Gemma 4 pipeline state layout mismatch"));
        }
        let mut position_offset = 0;
        for (layer, policy) in self
            .args
            .text
            .layer_schedule
            .iter()
            .enumerate()
            .skip(first_state_ordinal)
            .take(expected.len())
        {
            if policy.key_value.owns_state() {
                position_offset = position_offset.max(AttentionCache::<B::Tensor>::offset(
                    state
                        .layer(layer - first_state_ordinal)
                        .map_err(Error::backend)?,
                ));
            }
        }
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                position_offset,
                parts: Vec::new(),
                per_layer_token_override: None,
                per_layer_inputs,
                shared: HashMap::new(),
                vision_state: None,
                vision_initial: None,
                vision_output: None,
                audio_valid: None,
                audio_initial: None,
                audio_output: None,
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
        for (layer, policy) in self.args.text.layer_schedule.iter().enumerate() {
            if policy.key_value.owns_state() {
                position_offset = position_offset.max(AttentionCache::<B::Tensor>::offset(
                    state.layer(layer).map_err(Error::backend)?,
                ));
            }
        }
        Ok(LayeredForwardState {
            hidden: assembled.embeddings,
            context: ForwardContext {
                mask: input.mask.cloned(),
                position_offset,
                parts,
                per_layer_token_override: input.per_layer_tokens.cloned(),
                per_layer_inputs,
                shared: HashMap::new(),
                vision_state: None,
                vision_initial: None,
                vision_output,
                audio_valid: None,
                audio_initial: None,
                audio_output,
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
        let policy = self
            .args
            .text
            .layer_policy(index)
            .ok_or_else(|| Error::backend("missing Gemma 4 layer policy"))?;
        let generated_mask = if forward.mask.is_none() && hidden.dim(1) > 1 {
            Some(B::causal_mask(
                hidden.dim(1),
                forward.position_offset,
                policy
                    .attention
                    .window()
                    .map(|window| window.get() as i32 - 1),
                context,
            )?)
        } else {
            None
        };
        let per_layer_input = forward
            .per_layer_inputs
            .as_ref()
            .map(|inputs| {
                inputs.index(
                    &[
                        Index::Full,
                        Index::Full,
                        Index::At(index as i32),
                        Index::Full,
                    ],
                    context,
                )
            })
            .transpose()?;
        unit.forward_parallel(
            BlockInput {
                hidden,
                mask: forward.mask.as_ref().or(generated_mask.as_ref()),
                cache: Some(state.layer(index).map_err(Error::backend)?),
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
        let policy = self
            .args
            .text
            .layer_policy(index)
            .ok_or_else(|| Error::backend("missing Gemma 4 layer policy"))?;
        let generated_mask = if forward.mask.is_none() && hidden.dim(1) > 1 {
            Some(B::causal_mask(
                hidden.dim(1),
                forward.position_offset,
                policy
                    .attention
                    .window()
                    .map(|window| window.get() as i32 - 1),
                context,
            )?)
        } else {
            None
        };
        let per_layer_input = forward
            .per_layer_inputs
            .as_ref()
            .map(|inputs| {
                inputs.index(
                    &[
                        Index::Full,
                        Index::Full,
                        Index::At(index as i32),
                        Index::Full,
                    ],
                    context,
                )
            })
            .transpose()?;
        unit.forward_parallel_with_provider(
            BlockInput {
                hidden,
                mask: forward.mask.as_ref().or(generated_mask.as_ref()),
                cache: Some(state.layer(index).map_err(Error::backend)?),
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
        let policy = self
            .args
            .text
            .layer_policy(index)
            .ok_or_else(|| Error::backend("missing Gemma 4 layer policy"))?;
        let generated_mask = if forward.mask.is_none() && hidden.dim(1) > 1 {
            Some(B::causal_mask(
                hidden.dim(1),
                forward.position_offset,
                policy
                    .attention
                    .window()
                    .map(|window| window.get() as i32 - 1),
                context,
            )?)
        } else {
            None
        };
        let per_layer_input = forward
            .per_layer_inputs
            .as_ref()
            .map(|inputs| {
                inputs.index(
                    &[
                        Index::Full,
                        Index::Full,
                        Index::At(index as i32),
                        Index::Full,
                    ],
                    context,
                )
            })
            .transpose()?;
        unit.forward_with_provider(
            BlockInput {
                hidden,
                mask: forward.mask.as_ref().or(generated_mask.as_ref()),
                cache: Some(state.layer(index).map_err(Error::backend)?),
                shared: &mut forward.shared,
                per_layer_input: per_layer_input.as_ref(),
                rotary_position: Some(RotaryPosition::Offset(forward.position_offset)),
            },
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
            return Err(Error::backend("Gemma 4 input has no ordered parts"));
        }
        parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: self
                        .static_modules
                        .text
                        .embeddings
                        .forward(tokens, context)?
                        .multiply_scalar((self.args.text.hidden_size as f32).sqrt(), context)?,
                }),
                DecoderInputPart::Image(tokens) | DecoderInputPart::Video(tokens) => {
                    Ok(PreparedPart::Vision {
                        tokens: (*tokens).clone(),
                    })
                }
                DecoderInputPart::Audio(tokens) => Ok(PreparedPart::Audio {
                    tokens: (*tokens).clone(),
                }),
                DecoderInputPart::Projected { tokens, embeddings } => {
                    if embeddings.shape()
                        != [tokens.dim(0), tokens.dim(1), self.args.text.hidden_size]
                    {
                        return Err(Error::backend(format!(
                            "Gemma 4 projected input shape {:?} does not match tokens {:?} and hidden width {}",
                            embeddings.shape(),
                            tokens.shape(),
                            self.args.text.hidden_size
                        )));
                    }
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: (*embeddings).clone(),
                    })
                }
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
            return Err(Error::backend("Gemma 4 input has no ordered parts"));
        }
        let scale = (self.args.text.hidden_size as f32).sqrt();
        parts
            .iter()
            .map(|part| match part {
                DecoderInputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: B::vocabulary_parallel_lookup(
                        &mut self.static_modules.text.embeddings,
                        tokens,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?
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
                DecoderInputPart::Projected { tokens, embeddings } => {
                    if embeddings.shape()
                        != [tokens.dim(0), tokens.dim(1), self.args.text.hidden_size]
                    {
                        return Err(Error::backend(format!(
                            "Gemma 4 projected input shape {:?} does not match tokens {:?} and hidden width {}",
                            embeddings.shape(),
                            tokens.shape(),
                            self.args.text.hidden_size
                        )));
                    }
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: (*embeddings).clone(),
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
    ) -> Result<eredu_nn::multimodal::OrderedModelInput<B::Tensor>, Error> {
        let vision_tokens = token_count(parts, |part| matches!(part, PreparedPart::Vision { .. }));
        let audio_tokens = token_count(parts, |part| matches!(part, PreparedPart::Audio { .. }));
        validate_component("vision", vision, vision_tokens, self.args.text.hidden_size)?;
        validate_component("audio", audio, audio_tokens, self.args.text.hidden_size)?;
        let mut vision_offset = 0;
        let mut audio_offset = 0;
        let mut embeddings = Vec::with_capacity(parts.len());
        for part in parts {
            match part {
                PreparedPart::Text {
                    embeddings: value, ..
                } => embeddings.push(value.clone()),
                PreparedPart::Vision { tokens } => {
                    let length = tokens.dim(1);
                    embeddings.push(slice_component(
                        vision.expect("validated vision component"),
                        vision_offset,
                        length,
                        context,
                    )?);
                    vision_offset += length;
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
            }
        }
        let ordered = parts
            .iter()
            .zip(&embeddings)
            .map(|(part, embeddings)| OrderedInputPart {
                token_ids: match part {
                    PreparedPart::Text { tokens, .. }
                    | PreparedPart::Vision { tokens }
                    | PreparedPart::Audio { tokens } => tokens,
                },
                embeddings,
            })
            .collect::<Vec<_>>();
        assemble_ordered_inputs(&ordered, self.args.text.hidden_size, context)
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
    type Input<'a> = ModelInput<'a, B::Tensor>;
    type StaticModules = StaticModules<B>;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        match group {
            0 => eredu_runtime::ArchitectureGroupTransport {
                placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
                kind: eredu_runtime::ArchitectureGroupKind::VisionEncoder,
                first_owner_static_roles: self.args.vision.as_ref().map_or_else(Vec::new, |_| {
                    vec!["vision".into(), "vision_projection".into()]
                }),
                last_owner_static_roles: Vec::new(),
                merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
                parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
                request_optional: true,
            },
            1 => eredu_runtime::ArchitectureGroupTransport {
                placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
                kind: eredu_runtime::ArchitectureGroupKind::AudioEncoder,
                first_owner_static_roles: self.args.audio.as_ref().map_or_else(Vec::new, |_| {
                    vec!["audio".into(), "audio_projection".into()]
                }),
                last_owner_static_roles: Vec::new(),
                merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
                parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
                request_optional: true,
            },
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

    fn primary_execution_group(&self) -> &str {
        TEXT_EXECUTION_GROUP
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        crate::transport::pipeline_state(2, layout)
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error> {
        ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root(VISION_EXECUTION_GROUP),
                ExecutionGroupSpec::root(AUDIO_EXECUTION_GROUP),
                ExecutionGroupSpec::with_dependencies(
                    TEXT_EXECUTION_GROUP,
                    [VISION_EXECUTION_GROUP, AUDIO_EXECUTION_GROUP],
                ),
            ],
            TEXT_EXECUTION_GROUP,
        )
        .map_err(Error::backend)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
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
            _ => Err(Error::backend("Gemma 4 has three execution groups")),
        }
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
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
            _ => return Err(Error::backend("Gemma 4 has three execution groups")),
        };
        if index >= count {
            return Err(Error::backend("Gemma 4 unit is outside its group"));
        }
        match group {
            0 => Ok(format!("model.vision_tower.encoder.layers.{index}")),
            1 => Ok(format!("model.audio_tower.layers.{index}")),
            2 => Ok(format!("model.language_model.layers.{index}")),
            _ => unreachable!(),
        }
    }

    fn group_input_observation_path(&self, group: usize) -> Result<Option<String>, Self::Error> {
        Ok((group == 2).then(|| eredu_core::MODALITY_MERGE_OUTPUT_OBSERVATION_PATH.to_owned()))
    }

    fn group_output_observation_path(&self, group: usize) -> Result<Option<String>, Self::Error> {
        Ok(match group {
            0 => Some(eredu_core::VISION_PROJECTOR_OUTPUT_OBSERVATION_PATH.to_owned()),
            1 => Some(eredu_core::AUDIO_PROJECTOR_OUTPUT_OBSERVATION_PATH.to_owned()),
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
        <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index)?;
        match group {
            0 => Ok(Unit::Vision(VisionLayer::new(
                self.args
                    .vision
                    .as_ref()
                    .ok_or_else(|| Error::backend("Gemma 4 has no vision config"))?,
                index,
                context,
            )?)),
            1 => Ok(Unit::Audio(AudioLayer::new(
                self.args
                    .audio
                    .as_ref()
                    .ok_or_else(|| Error::backend("Gemma 4 has no audio config"))?,
                index,
                context,
            )?)),
            2 => {
                let args = match &self.parallel_geometry {
                    Some(geometry) => geometry.text_block(index).ok_or_else(|| {
                        Error::backend(format!("missing rank-local Gemma 4 text geometry {index}"))
                    })?,
                    None => &self.args.text,
                };
                Ok(Unit::Text(DenseBlock::new(args, index, context)?))
            }
            _ => Err(Error::backend("Gemma 4 has three execution groups")),
        }
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.validate_partition_state(state)?;
        let parts = self.prepare_parts(input.parts, context)?;
        let (vision_initial, vision_state) = match input.vision {
            Some(vision) => {
                let (hidden, state) = self
                    .static_modules
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("Gemma 4 has no vision tower"))?
                    .begin(vision, context)?;
                (Some(hidden), Some(state))
            }
            None => (None, None),
        };
        let (audio_initial, audio_valid) = match input.audio {
            Some(audio) => {
                let (hidden, valid) = self
                    .static_modules
                    .audio
                    .as_mut()
                    .ok_or_else(|| Error::backend("Gemma 4 has no audio tower"))?
                    .begin(audio, context)?;
                (Some(hidden), Some(valid))
            }
            None => (None, None),
        };
        let assembled = self.assemble(&parts, None, None, context);
        let hidden = vision_initial
            .as_ref()
            .or(audio_initial.as_ref())
            .cloned()
            .or_else(|| {
                assembled
                    .as_ref()
                    .ok()
                    .map(|assembled| assembled.embeddings.clone())
            })
            .ok_or_else(|| {
                assembled
                    .err()
                    .unwrap_or_else(|| Error::backend("empty Gemma input"))
            })?;
        let position_offset = self.partition_position_offset(state)?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: input.mask.cloned(),
                position_offset,
                parts,
                per_layer_token_override: input.per_layer_tokens.cloned(),
                per_layer_inputs: None,
                shared: HashMap::new(),
                vision_state,
                vision_initial,
                vision_output: None,
                audio_valid,
                audio_initial,
                audio_output: None,
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
                let assembled = self.assemble(&forward.parts, vision, audio, context)?;
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
        match (group, unit) {
            (0, Unit::Vision(unit)) => self
                .static_modules
                .vision
                .as_ref()
                .ok_or_else(|| Error::backend("Gemma 4 vision static modules are missing"))?
                .forward_layer(
                    unit,
                    hidden,
                    forward
                        .vision_state
                        .as_ref()
                        .ok_or_else(|| Error::backend("Gemma 4 vision state is missing"))?,
                    context,
                ),
            (1, Unit::Audio(unit)) => unit.forward(
                hidden,
                forward
                    .audio_valid
                    .as_deref()
                    .ok_or_else(|| Error::backend("Gemma 4 audio extent is missing"))?,
                context,
            ),
            (2, Unit::Text(unit)) => {
                let policy = self
                    .args
                    .text
                    .layer_policy(index)
                    .ok_or_else(|| Error::backend("missing Gemma 4 layer policy"))?;
                let generated_mask = if forward.mask.is_none() && hidden.dim(1) > 1 {
                    Some(B::causal_mask(
                        hidden.dim(1),
                        forward.position_offset,
                        policy
                            .attention
                            .window()
                            .map(|window| window.get() as i32 - 1),
                        context,
                    )?)
                } else {
                    None
                };
                let per_layer_input = forward
                    .per_layer_inputs
                    .as_ref()
                    .map(|inputs| {
                        inputs.index(
                            &[
                                Index::Full,
                                Index::Full,
                                Index::At(index as i32),
                                Index::Full,
                            ],
                            context,
                        )
                    })
                    .transpose()?;
                unit.forward(
                    BlockInput {
                        hidden,
                        mask: forward.mask.as_ref().or(generated_mask.as_ref()),
                        cache: Some(
                            state
                                .layer(self.local_state_ordinal(index)?)
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
        match group {
            0 if forward.vision_state.is_some() => {
                let encoded = self
                    .static_modules
                    .vision
                    .as_ref()
                    .expect("validated vision modules")
                    .finish(hidden, forward.vision_state.as_ref().unwrap(), context)?;
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
                    .finish(
                        hidden,
                        forward
                            .audio_valid
                            .as_deref()
                            .expect("validated audio extents"),
                        context,
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

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = Vec::new();
        values.extend(forward.mask.iter());
        values.extend(forward.per_layer_token_override.iter());
        values.extend(forward.per_layer_inputs.iter());
        for part in &forward.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => values.extend([tokens, embeddings]),
                PreparedPart::Vision { tokens } | PreparedPart::Audio { tokens } => {
                    values.push(tokens)
                }
            }
        }
        if let Some(state) = &forward.vision_state {
            values.extend(state.retained_values());
        }
        values.extend(forward.vision_initial.iter());
        values.extend(forward.vision_output.iter());
        values.extend(forward.audio_initial.iter());
        values.extend(forward.audio_output.iter());
        for (key, value) in forward.shared.values() {
            values.extend([key, value]);
        }
        values.into_iter()
    }
}

impl<B, S> ParallelLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.parallel_geometry
            .as_ref()
            .ok_or_else(|| Error::backend("Gemma 4 model was not built with local geometry"))?;
        self.validate_partition_state(state)?;
        let parts = self.prepare_parts_parallel(input.parts, parallel, context)?;
        let (vision_initial, vision_state) = match input.vision {
            Some(vision) => {
                let (hidden, state) = self
                    .static_modules
                    .vision
                    .as_mut()
                    .ok_or_else(|| Error::backend("Gemma 4 has no vision tower"))?
                    .begin(vision, context)?;
                (Some(hidden), Some(state))
            }
            None => (None, None),
        };
        let (audio_initial, audio_valid) = match input.audio {
            Some(audio) => {
                let (hidden, valid) = self
                    .static_modules
                    .audio
                    .as_mut()
                    .ok_or_else(|| Error::backend("Gemma 4 has no audio tower"))?
                    .begin(audio, context)?;
                (Some(hidden), Some(valid))
            }
            None => (None, None),
        };
        let assembled = self.assemble(&parts, None, None, context);
        let hidden = vision_initial
            .as_ref()
            .or(audio_initial.as_ref())
            .cloned()
            .or_else(|| {
                assembled
                    .as_ref()
                    .ok()
                    .map(|assembled| assembled.embeddings.clone())
            })
            .ok_or_else(|| {
                assembled
                    .err()
                    .unwrap_or_else(|| Error::backend("empty Gemma input"))
            })?;
        let position_offset = self.partition_position_offset(state)?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask: input.mask.cloned(),
                position_offset,
                parts,
                per_layer_token_override: input.per_layer_tokens.cloned(),
                per_layer_inputs: None,
                shared: HashMap::new(),
                vision_state,
                vision_initial,
                vision_output: None,
                audio_valid,
                audio_initial,
                audio_output: None,
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
        match (group, unit) {
            (0, Unit::Vision(unit)) => self
                .static_modules
                .vision
                .as_ref()
                .ok_or_else(|| Error::backend("Gemma 4 vision static modules are missing"))?
                .forward_layer(
                    unit,
                    hidden,
                    forward
                        .vision_state
                        .as_ref()
                        .ok_or_else(|| Error::backend("Gemma 4 vision state is missing"))?,
                    context,
                ),
            (1, Unit::Audio(unit)) => unit.forward(
                hidden,
                forward
                    .audio_valid
                    .as_deref()
                    .ok_or_else(|| Error::backend("Gemma 4 audio extent is missing"))?,
                context,
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
        let hidden = self.static_modules.text.norm.forward(hidden, context)?;
        let logits = match &mut self.static_modules.text.head {
            Some(head) => B::vocabulary_parallel_project(head, &hidden, parallel, context)?,
            None => B::vocabulary_parallel_embedding_project(
                &mut self.static_modules.text.embeddings,
                &hidden,
                parallel,
                context,
            )?,
        };
        model_text::args_cap::<B>(logits, self.args.text.final_logit_softcapping, context)
    }
}

fn token_count<T: Tensor>(
    parts: &[PreparedPart<T>],
    select: impl Fn(&PreparedPart<T>) -> bool,
) -> i32 {
    parts
        .iter()
        .filter(|part| select(part))
        .map(|part| match part {
            PreparedPart::Text { tokens, .. }
            | PreparedPart::Vision { tokens }
            | PreparedPart::Audio { tokens } => tokens.dim(1),
        })
        .sum()
}

fn validate_component<T: Tensor>(
    name: &str,
    value: Option<&T>,
    tokens: i32,
    hidden: i32,
) -> Result<(), Error> {
    match value {
        Some(value) if value.shape() == [1, tokens, hidden] => Ok(()),
        None if tokens == 0 => Ok(()),
        Some(value) => Err(Error::backend(format!(
            "Gemma 4 {name} output has shape {:?}, expected [1, {tokens}, {hidden}]",
            value.shape()
        ))),
        None => Err(Error::backend(format!(
            "Gemma 4 {name} placeholders require projected media"
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

// Text-only pinned modules are isolated so the composite static tree can add
// media without changing their stable parameter identities.
pub(crate) mod model_text {
    use eredu_nn::{
        EmbeddingSpec, Error, GroupedNeuralBackend, LinearOperator, LinearSpec,
        NormalizationConstructionSpec, ParameterSpec, Parameterized, Tensor,
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
            let embedding = "model.language_model.embed_tokens.weight";
            let per_layer_width =
                args.hidden_size_per_layer_input * args.num_hidden_layers() as i32;
            let per_layer_embedding = "model.language_model.embed_tokens_per_layer.weight";
            let per_layer_projection = "model.language_model.per_layer_model_projection.weight";
            let head = "lm_head.weight";
            Ok(Self {
                embeddings: B::embedding(
                    EmbeddingSpec {
                        vocabulary: args.vocab_size,
                        dimensions: args.hidden_size,
                        weight: ParameterSpec::trainable(embedding).map_err(Error::backend)?,
                        format: crate::linear_format::standard_linear_format(
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
                                weight: ParameterSpec::trainable(per_layer_embedding)
                                    .map_err(Error::backend)?,
                                format: crate::linear_format::standard_linear_format(
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
                                weight: ParameterSpec::trainable(per_layer_projection)
                                    .map_err(Error::backend)?,
                                bias: None,
                                format: crate::linear_format::standard_linear_format(
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
                                ParameterSpec::trainable(
                                    "model.language_model.per_layer_projection_norm.weight",
                                )
                                .map_err(Error::backend)?,
                            ),
                            context,
                        )
                    })
                    .transpose()?,
                norm: B::normalization(
                    NormalizationConstructionSpec::learned(
                        args.hidden_size,
                        args.rms_norm_eps,
                        ParameterSpec::trainable("model.language_model.norm.weight")
                            .map_err(Error::backend)?,
                    ),
                    context,
                )?,
                head: (!args.tie_word_embeddings)
                    .then(|| {
                        B::linear(
                            LinearSpec {
                                input: args.hidden_size,
                                output: args.vocab_size,
                                weight: ParameterSpec::trainable(head).map_err(Error::backend)?,
                                bias: None,
                                format: crate::linear_format::standard_linear_format(
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
                        weight: ParameterSpec::trainable(embedding).map_err(Error::backend)?,
                        format: crate::linear_format::standard_linear_format(
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
                                weight: ParameterSpec::trainable(per_layer_embedding)
                                    .map_err(Error::backend)?,
                                format: crate::linear_format::standard_linear_format(
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
                                weight: ParameterSpec::trainable(per_layer_projection)
                                    .map_err(Error::backend)?,
                                bias: None,
                                format: crate::linear_format::standard_linear_format(
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
                                ParameterSpec::trainable(
                                    "model.language_model.per_layer_projection_norm.weight",
                                )
                                .map_err(Error::backend)?,
                            ),
                            context,
                        )
                    })
                    .transpose()?,
                norm: B::normalization(
                    NormalizationConstructionSpec::learned(
                        args.hidden_size,
                        args.rms_norm_eps,
                        ParameterSpec::trainable("model.language_model.norm.weight")
                            .map_err(Error::backend)?,
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
                            weight: ParameterSpec::trainable(head).map_err(Error::backend)?,
                            bias: None,
                            format: crate::linear_format::standard_linear_format(
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
            use eredu_nn::{EmbeddingOperator, NormalizationOperator};
            let hidden = self.norm.forward(hidden, context)?;
            let logits = match self.head.as_mut() {
                Some(head) => head.forward(&hidden, context)?,
                None => self.embeddings.as_linear(&hidden, context)?,
            };
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
