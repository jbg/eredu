//! Backend-neutral Gemma 4 multimodal model and layered runtime lifecycle.

use std::collections::HashMap;

use eredu_nn::{
    multimodal::{assemble_ordered_inputs, OrderedInputPart},
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, Index, LinearOperator,
    NormalizationOperator, Parameterized, RotaryPosition, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout,
    ExpertPass, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
    OwnedParameterGroupSpec, ParallelLayeredArchitecture, ParallelPlanError,
    ParallelRoutedLayeredArchitecture, ParameterGroupOwner, RoutedExpertProvider,
    RoutedLayeredArchitecture, StateLayout,
};

use super::{
    audio_layer_parameter_groups, audio_static_parameter_groups, layer_parameter_groups,
    modality_projection_parameter_groups, state_layout, static_parameter_groups,
    vision_layer_parameter_groups, vision_static_parameter_groups, AudioInput, AudioLayer,
    AudioStatic, BlockInput, DenseBlock, FamilyConfig, LocalGeometry, ModalityProjector,
    SharedAttentionStates, VisionInput, VisionLayer, VisionState, VisionStatic,
};

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

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use eredu_runtime::{ArchitectureBoundary, BoundaryTensorDtype};

    #[test]
    fn optional_per_layer_input_owns_complete_wire_geometry() {
        let present = TextBoundarySchema {
            per_layer_geometry: Some((4, 8)),
        };
        let tensors = present.wire_schema().unwrap().resolve(2, 3).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].role(), "per_layer_input");
        assert_eq!(tensors[0].shape(), [2, 3, 4, 8]);
        assert_eq!(tensors[0].dtype(), BoundaryTensorDtype::Activation);

        let absent = TextBoundarySchema {
            per_layer_geometry: None,
        };
        assert!(absent.wire_schema().unwrap().tensors().is_empty());
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
pub struct StaticModules<B: RoutedNeuralBackend> {
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

enum PreparedPart<T> {
    Text { tokens: T, embeddings: T },
    Vision { tokens: T },
    Audio { tokens: T },
}

/// A streamable image, audio, or decoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: RoutedNeuralBackend> {
    /// Vision encoder block.
    Vision(VisionLayer<B>),
    /// Audio encoder block.
    Audio(AudioLayer<B>),
    /// Text decoder block.
    Text(DenseBlock<B>),
}

impl<B, S> RoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
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
    B: RoutedNeuralBackend,
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
        P: RoutedExpertProvider<B>,
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

    /// Replaces the caller-supplied decoder mask for a pipeline stage.
    pub fn set_pipeline_mask(&mut self, mask: Option<T>) {
        self.mask = mask;
    }
}

/// Family-owned schema for decoder-wide per-layer input transport.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TextBoundarySchema {
    per_layer_geometry: Option<(i32, i32)>,
}

impl TextBoundarySchema {
    /// Derives the schema from normalized text and rank-local geometry.
    pub fn from_args(args: &super::ModelArgs, geometry: &LocalGeometry) -> Self {
        Self {
            per_layer_geometry: (args.hidden_size_per_layer_input > 0)
                .then(|| (args.num_hidden_layers() as i32, geometry.per_layer_width())),
        }
    }
}

impl eredu_runtime::ArchitectureBoundary for TextBoundarySchema {
    type Boundary<T> = TextBoundary<T>;

    const IDENTITY: &'static str = "gemma4.text";

    fn tensor_specs(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
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
    ) -> Result<Vec<T>, eredu_runtime::ArchitectureBoundaryError> {
        let actual = usize::from(boundary.per_layer_input.is_some());
        let expected = usize::from(self.per_layer_geometry.is_some());
        if actual != expected {
            return Err(eredu_runtime::ArchitectureBoundaryError::TensorCount {
                boundary: "gemma4.text",
                expected,
                actual,
            });
        }
        Ok(boundary.per_layer_input.into_iter().collect())
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
pub struct LayeredModel<B: RoutedNeuralBackend> {
    args: FamilyConfig,
    static_modules: StaticModules<B>,
    parallel_geometry: Option<LocalGeometry>,
}

impl<B: RoutedNeuralBackend> crate::BindableStaticParameters<B> for LayeredModel<B> {
    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: crate::StaticParameterVisitor<B>,
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
        V: crate::StaticParameterVisitorMut<B>,
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

impl<B: RoutedNeuralBackend> LayeredModel<B> {
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
    {
        for (group, hidden) in [(0, vision_hidden), (1, audio_hidden)] {
            let Some(hidden) = hidden else { continue };
            match parallel {
                Some(parallel) => {
                    <Self as ParallelLayeredArchitecture<B, S>>::complete_execution_group_parallel(
                        self,
                        group,
                        &hidden,
                        state,
                        &mut forward.context,
                        parallel,
                        context,
                    )?
                }
                None => <Self as LayeredArchitecture<B, S>>::complete_execution_group(
                    self,
                    group,
                    &hidden,
                    state,
                    &mut forward.context,
                    context,
                )?,
            };
        }
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
        })
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
    pub fn parameter_description(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("vision"),
                ExecutionGroupSpec::root("audio"),
                ExecutionGroupSpec::with_dependencies("text_decoder", ["vision", "audio"]),
            ],
            "text_decoder",
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
    pub fn runtime_state_layout(&self) -> Result<StateLayout, Error> {
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

    /// Resumes a decoder-only pipeline stage from transported activations.
    pub fn resume_pipeline_text<S: LayerRuntimeState<B>>(
        &self,
        hidden: B::Tensor,
        mask: Option<B::Tensor>,
        per_layer_inputs: Option<B::Tensor>,
        state: &mut S,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout().len() != self.args.text.num_hidden_layers() {
            return Err(Error::backend("Gemma 4 pipeline state layout mismatch"));
        }
        let mut position_offset = 0;
        for (layer, policy) in self.args.text.layer_schedule.iter().enumerate() {
            if policy.key_value.owns_state() {
                position_offset = position_offset.max(AttentionCache::<B::Tensor>::offset(
                    state.layer(layer).map_err(Error::backend)?,
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

    /// Executes one ordinary text block through the shared neutral context.
    pub fn forward_text_unit<S: LayerRuntimeState<B>>(
        &mut self,
        index: usize,
        unit: &mut DenseBlock<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
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
        unit.forward(
            BlockInput {
                hidden,
                mask: forward.mask.as_ref().or(generated_mask.as_ref()),
                cache: Some(state.layer(index).map_err(Error::backend)?),
                shared: &mut forward.shared,
                per_layer_input: per_layer_input.as_ref(),
                rotary_position: Some(RotaryPosition::Offset(forward.position_offset)),
            },
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

    /// Applies the replicated final norm before a backend-owned vocabulary head.
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

    /// Applies the ordinary final normalization and vocabulary projection.
    pub fn project_pipeline_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.text.project_logits(
            hidden,
            self.args.text.final_logit_softcapping,
            context,
        )
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
    B: RoutedNeuralBackend,
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
                first_owner_static_roles: vec!["vision".into(), "vision_projection".into()],
                last_owner_static_roles: Vec::new(),
                merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
                parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
                request_optional: true,
            },
            1 => eredu_runtime::ArchitectureGroupTransport {
                placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
                kind: eredu_runtime::ArchitectureGroupKind::AudioEncoder,
                first_owner_static_roles: vec!["audio".into(), "audio_projection".into()],
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

    fn model_identity(&self) -> &str {
        &self.args.model_type
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error> {
        ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("vision"),
                ExecutionGroupSpec::root("audio"),
                ExecutionGroupSpec::with_dependencies("text_decoder", ["vision", "audio"]),
            ],
            "text_decoder",
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
        if state.layout() != &state_layout(&self.args.text).map_err(Error::backend)? {
            return Err(Error::backend("Gemma 4 runtime state layout mismatch"));
        }
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
        let mut position_offset = 0;
        for (layer, policy) in self.args.text.layer_schedule.iter().enumerate() {
            if policy.key_value.owns_state() {
                position_offset = position_offset.max(AttentionCache::<B::Tensor>::offset(
                    state.layer(layer).map_err(Error::backend)?,
                ));
            }
        }
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
        _dependencies: &[&B::Tensor],
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match group {
            0 => Ok(forward.vision_initial.as_ref().unwrap_or(initial).clone()),
            1 => Ok(forward.audio_initial.as_ref().unwrap_or(initial).clone()),
            2 => {
                let assembled = self.assemble(
                    &forward.parts,
                    forward.vision_output.as_ref(),
                    forward.audio_output.as_ref(),
                    context,
                )?;
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
                        cache: Some(state.layer(index).map_err(Error::backend)?),
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
    B: RoutedNeuralBackend,
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
        let expected = self
            .parallel_geometry
            .as_ref()
            .ok_or_else(|| Error::backend("Gemma 4 model was not built with local geometry"))?
            .state_layout();
        if state.layout() != expected {
            return Err(Error::backend("Gemma 4 rank-local state layout mismatch"));
        }
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
        let mut position_offset = 0;
        for (layer, policy) in self.args.text.layer_schedule.iter().enumerate() {
            if policy.key_value.owns_state() {
                position_offset = position_offset.max(AttentionCache::<B::Tensor>::offset(
                    state.layer(layer).map_err(Error::backend)?,
                ));
            }
        }
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
        EmbeddingSpec, Error, LinearOperator, LinearSpec, NormalizationSpec, ParameterSpec,
        Parameterized, RoutedNeuralBackend, Tensor,
    };

    use super::super::ModelArgs;

    /// Pinned Gemma text modules.
    #[derive(Debug, Clone, Parameterized)]
    #[parameterized(tensor = "B::Tensor")]
    pub struct StaticTextModules<B: RoutedNeuralBackend> {
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

    impl<B: RoutedNeuralBackend> StaticTextModules<B> {
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
                        B::rms_norm(
                            NormalizationSpec {
                                dimensions: args.hidden_size_per_layer_input,
                                epsilon: args.rms_norm_eps,
                                weight: ParameterSpec::trainable(
                                    "model.language_model.per_layer_projection_norm.weight",
                                )
                                .map_err(Error::backend)?,
                            },
                            context,
                        )
                    })
                    .transpose()?,
                norm: B::rms_norm(
                    NormalizationSpec {
                        dimensions: args.hidden_size,
                        epsilon: args.rms_norm_eps,
                        weight: ParameterSpec::trainable("model.language_model.norm.weight")
                            .map_err(Error::backend)?,
                    },
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
                        B::rms_norm(
                            NormalizationSpec {
                                dimensions: per_layer_width,
                                epsilon: args.rms_norm_eps,
                                weight: ParameterSpec::trainable(
                                    "model.language_model.per_layer_projection_norm.weight",
                                )
                                .map_err(Error::backend)?,
                            },
                            context,
                        )
                    })
                    .transpose()?,
                norm: B::rms_norm(
                    NormalizationSpec {
                        dimensions: args.hidden_size,
                        epsilon: args.rms_norm_eps,
                        weight: ParameterSpec::trainable("model.language_model.norm.weight")
                            .map_err(Error::backend)?,
                    },
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

    pub(super) fn args_cap<B: RoutedNeuralBackend>(
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
