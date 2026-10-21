//! Prepared image ingress and shared text invocation traversal.
mod partition;
pub use partition::{Boundary, BoundarySchema};

use super::{
    model::{ForwardContext as TextForward, TextInput, TextKernel},
    text::TextLayer,
    vision::ImageStatic,
    EncoderAttention, FamilyConfig,
};
use crate::{
    composite_execution::{
        CompositeArchitecture, PreparedCompositeInput, SingleCompositePrefillRequest,
    },
    gemma4::{
        model::{prepare_composite_ingress, DecoderInputPart},
        VisionLayer, VisionState,
    },
    media_plan::{AdmittedCompositeInput, Gemma4InputPartPlan},
};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, Error, GroupedNeuralBackend, Index, Parameterized,
    ReadOnlyAttentionCache, Tensor,
};
use eredu_runtime::{
    ArchitectureGroupTransport, ArchitectureParameterDescription, ArchitectureParameters,
    ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout, LayerRuntimeState,
    LayeredArchitecture, LayeredForwardState, OwnedParameterGroupSpec, ParameterGroupOwner,
    ParameterProvider, ResidentExpertProvider, RoutedLayeredArchitecture, StateLayout,
};

use eredu_runtime::{
    ComponentEquations as _, ComponentKernel as _, ComponentPinnedResources as _,
    RoutedComponentKernel as _,
};

/// Independently traversable image stack.
pub const VISION_EXECUTION_GROUP: &str = "vision";

/// Shared pinned text parameters and image ingress/projection parameters.
#[derive(Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    text: TextKernel<B>,
    image: ImageStatic<B>,
}

/// One bounded materialization unit.
#[derive(Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    /// One image transformer layer.
    Vision(VisionLayer<B>),
    /// One shared context/refinement transformer layer.
    Text(TextLayer<B>),
}

/// Request-local image products and explicit text invocation state.
pub struct ForwardContext<T> {
    text: TextForward<T>,
    segments: Vec<(T, Option<T>, bool)>,
    vision: Option<VisionState<T>>,
    vision_initial: Option<T>,
    vision_output: Option<T>,
    tokens: T,
    feedback: Option<Option<T>>,
    deferred: bool,
    vision_spans: Vec<(i32, i32)>,
    pending_text_boundary: Option<Boundary<T>>,
}

/// Architecture-owned composition; native adapters remain model independent.
pub struct Model<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    config: FamilyConfig,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    statics: StaticModules<B>,
    geometry: Option<super::parallel::CompositeGeometry>,
}

/// Reuses the common prepared image contract with this family's exact geometry.
pub fn ingress_config(config: &FamilyConfig) -> crate::gemma4::FamilyConfig {
    crate::gemma4::FamilyConfig {
        model_type: "diffusion_gemma".into(),
        text: config.text.clone(),
        vision: config.vision.clone(),
        image_token_id: config.image_token_id.map(|n| n as i32),
        video_token_id: config.image_token_id.map(|n| n as i32),
        audio: None,
        audio_token_id: None,
    }
}

/// Declares per-layer image execution followed by the shared text stack.
pub fn graph(config: &FamilyConfig) -> Result<(ExecutionGraph, ExecutionUnitLayout), Error> {
    let graph = ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root(VISION_EXECUTION_GROUP),
            ExecutionGroupSpec::with_dependencies(
                super::model::TEXT_EXECUTION_GROUP,
                [VISION_EXECUTION_GROUP],
            ),
        ],
        super::model::TEXT_EXECUTION_GROUP,
    )
    .map_err(Error::backend)?;
    let units = ExecutionUnitLayout::new(
        &graph,
        [
            config
                .vision
                .as_ref()
                .map_or(0, |v| v.num_hidden_layers as usize),
            config.text.num_hidden_layers(),
        ],
    )
    .map_err(Error::backend)?;
    Ok((graph, units))
}

/// Canonical shared text and image storage, declared without native modules.
pub fn parameter_description(
    config: &FamilyConfig,
) -> Result<ArchitectureParameterDescription, Error> {
    let (graph, units) = graph(config)?;
    let mut owned = super::parallel::parameter_description(config)
        .map_err(Error::backend_source)?
        .groups()
        .to_vec();
    let vision = config
        .vision
        .as_ref()
        .ok_or_else(|| Error::backend("composite requires an image tower"))?;
    for group in
        crate::gemma4::vision::parameters::static_groups(vision, "model.encoder.vision_tower")
            .map_err(Error::backend_source)?
    {
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("vision"),
            group,
        ));
    }
    let weight = "model.encoder.embed_vision.embedding_projection.weight";
    let projection = eredu_runtime::ParameterGroupSpec::new(
        "model.encoder.embed_vision",
        eredu_runtime::ParameterRole::Replicated,
        [eredu_runtime::ParameterMemberSpec::new(
            weight,
            vec![
                config.text.hidden_size as usize,
                vision.hidden_size as usize,
            ],
            eredu_runtime::MemberSharding::Replicated,
        )],
    )
    .map_err(Error::backend_source)?;
    for group in eredu_runtime::expand_linear_format_parameter_groups(vec![projection], |member| {
        crate::linear_format::standard_parallel_linear_format(
            member,
            config.text.linear_format_for(member.target()),
        )
    })
    .map_err(Error::backend_source)?
    {
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_role("vision_projection"),
            group,
        ));
    }
    for index in 0..vision.num_hidden_layers as usize {
        for group in crate::gemma4::vision::parameters::layer_groups(
            vision,
            "model.encoder.vision_tower",
            index,
        )
        .map_err(Error::backend_source)?
        {
            owned.push(OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(units.group_id(0).unwrap().clone(), index),
                group,
            ));
        }
    }
    ArchitectureParameterDescription::new(
        &graph,
        &units,
        owned.iter().map(|n| n.group().clone()),
        owned.clone(),
    )
    .map_err(Error::backend_source)
}

/// Validates image invocations and shared text state before materialization.
pub fn component_definition(
    config: &FamilyConfig,
    geometry: Option<&super::parallel::CompositeGeometry>,
) -> Result<eredu_runtime::ComponentDefinition, Error> {
    let layers = config.text.num_hidden_layers();
    let vision_layers = config
        .vision
        .as_ref()
        .ok_or_else(|| Error::backend("composite requires an image tower"))?
        .num_hidden_layers as usize;
    let state = match geometry {
        Some(geometry) => {
            geometry
                .text
                .validate_for(config)
                .map_err(Error::backend_source)?;
            let range = geometry.text.units();
            if geometry.vision_units.start > geometry.vision_units.end
                || geometry.vision_units.end > vision_layers
                || geometry.complete_state.len() != layers
                || range.start > range.end
                || range.end > layers
                || range.len() != geometry.text.state_layout().len()
                || range
                    .clone()
                    .zip(geometry.text.state_layout().layers().iter())
                    .any(|(global, local)| {
                        geometry.complete_state.layers().get(global) != Some(local)
                    })
            {
                return Err(Error::backend(
                    "composite local geometry changes declared unit or state ownership",
                ));
            }
            geometry.complete_state.clone()
        }
        None => super::state_layout(config).map_err(Error::backend_source)?,
    };
    eredu_runtime::ComponentDefinition::new(
        "diffusion_gemma_composite",
        parameter_description(config)?,
        (0..vision_layers)
            .map(|index| eredu_runtime::ComponentUnit {
                path: format!("model.encoder.vision_tower.encoder.layers.{index}"),
                state: 0..0,
            })
            .chain((0..layers).map(|index| eredu_runtime::ComponentUnit {
                path: format!("model.decoder.layers.{index}"),
                state: index..index + 1,
            }))
            .collect(),
        vec![vision_transport(), super::model::text_transport()],
        super::model::TEXT_EXECUTION_GROUP,
        vec![],
        state,
        eredu_runtime::ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::group_units(1, 0..layers),
        ]),
    )
    .map_err(Error::from)
}

/// Image parameters are replicated within a tensor group and cut per layer by PP.
pub fn vision_transport() -> ArchitectureGroupTransport {
    use eredu_runtime::*;
    ArchitectureGroupTransport {
        placement: ArchitectureGroupPlacement::Pipeline,
        kind: ArchitectureGroupKind::VisionEncoder,
        first_owner_static_roles: vec!["vision".into()],
        last_owner_static_roles: vec!["vision".into(), "vision_projection".into()],
        merge_destination: ArchitectureMergeDestination::FirstPipelineOwner,
        parallel_subgroup: None,
        request_optional: true,
    }
}

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> Model<B> {
    /// Declares shared text and independently materialized image modules.
    pub fn new(
        config: FamilyConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let definition = std::sync::Arc::new(component_definition(&config, None)?);
        Ok(Self {
            definition,
            statics: StaticModules {
                text: TextKernel::materialize(config.clone(), None, context)?,
                image: ImageStatic::new(&config, context)?,
            },
            config,
            geometry: None,
        })
    }

    /// Retained source-free declaration used by execution and inspection.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }

    fn begin_input<S>(
        &mut self,
        input: PreparedCompositeInput<'_, B::Tensor, Gemma4InputPartPlan>,
        state: &mut S,
        deferred: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    {
        let ingress = prepare_composite_ingress::<B>(input, context)?;
        let mut segments = Vec::new();
        for part in ingress.decoder_parts() {
            segments.push(match part {
                DecoderInputPart::Text(tokens) => (tokens.clone(), None, false),
                DecoderInputPart::Image(tokens) | DecoderInputPart::Video(tokens) => {
                    (tokens.clone(), None, true)
                }
                DecoderInputPart::Projected { tokens, embeddings } => {
                    (tokens.clone(), Some(embeddings.clone()), false)
                }
                DecoderInputPart::Audio(_) => {
                    return Err(Error::backend(
                        "image/text architecture does not admit audio",
                    ))
                }
            });
        }
        let tokens = B::Tensor::concatenate(
            &segments
                .iter()
                .map(|(tokens, _, _)| tokens.clone())
                .collect::<Vec<_>>(),
            1,
            context,
        )?;
        let mut row_spans = vec![(-1, -1); tokens.dim(1) as usize];
        let text = if let Some(feedback) = input.refinement() {
            if segments
                .iter()
                .any(|(_, projected, native)| projected.is_some() || *native)
            {
                return Err(Error::backend("refinement requires tentative token rows"));
            }
            if deferred {
                self.statics.text.begin_invocation_embeddings(
                    B::Tensor::full_f32(
                        0.0,
                        &[1, tokens.dim(1), self.config.text.hidden_size],
                        context,
                    )?,
                    None,
                    super::TextInvocationPhase::Refinement,
                    state,
                )?
            } else {
                self.statics.text.begin(
                    TextInput::Refinement {
                        tokens: &tokens,
                        feedback,
                        mask: None,
                    },
                    state,
                    context,
                )?
            }
        } else {
            let placeholder = if deferred {
                B::Tensor::full_f32(
                    0.0,
                    &[1, tokens.dim(1), self.config.text.hidden_size],
                    context,
                )?
            } else {
                self.statics.text.encode_context_tokens(&tokens, context)?
            };
            let frontier = if state.layout().is_empty() {
                0
            } else {
                state.layer(0).map_err(Error::backend)?.offset()
            };
            let sequence = tokens.dim(1);
            let vision_spans = input
                .admitted()
                .parts()
                .iter()
                .map(|part| {
                    matches!(
                        part,
                        Gemma4InputPartPlan::Vision { .. }
                            | Gemma4InputPartPlan::Projected {
                                modality: eredu_core::InputModality::Image
                                    | eredu_core::InputModality::Video,
                                ..
                            }
                    )
                })
                .collect::<Vec<_>>();
            let mut spans = Vec::new();
            let mut start = 0;
            for (tokens, _, _) in &segments {
                let end = start + tokens.dim(1);
                spans.push((start, end));
                start = end;
            }
            for (vision, (start, end)) in vision_spans.iter().zip(&spans) {
                if *vision {
                    for row in *start..*end {
                        row_spans[row as usize] = (*start, *end);
                    }
                }
            }
            let mut values = Vec::new();
            for query in 0..sequence {
                for key in 0..frontier + sequence {
                    let same_image =
                        vision_spans
                            .iter()
                            .zip(&spans)
                            .any(|(vision, (start, end))| {
                                *vision
                                    && query >= *start
                                    && query < *end
                                    && key >= frontier + *start
                                    && key < frontier + *end
                            });
                    values.push(
                        if self.config.encoder_attention == EncoderAttention::Bidirectional
                            || key <= frontier + query
                            || same_image
                        {
                            1
                        } else {
                            0
                        },
                    );
                }
            }
            let mask = B::Tensor::from_i32_slice(
                &values,
                &[1, 1, sequence, frontier + sequence],
                context,
            )?
            .equal_i32(1, context)?;
            self.statics
                .text
                .begin_context_embeddings(placeholder, Some(mask), state)?
        };
        let (vision_initial, vision) = if input.refinement().is_none() {
            ingress
                .vision_input()
                .map(|input| {
                    if self
                        .geometry
                        .as_ref()
                        .is_none_or(|geometry| geometry.vision_units.start == 0)
                    {
                        self.statics.image.begin(input, context)
                    } else {
                        Ok((
                            B::Tensor::full_f32(
                                0.0,
                                &[
                                    input.patches.dim(0),
                                    input.patches.dim(1),
                                    self.config.vision.as_ref().unwrap().hidden_size,
                                ],
                                context,
                            )?,
                            self.statics.image.tower.prepare_state(input, context)?,
                        ))
                    }
                })
                .transpose()?
                .map_or((None, None), |(hidden, state)| (Some(hidden), Some(state)))
        } else {
            (None, None)
        };
        Ok(LayeredForwardState {
            hidden: text.hidden,
            context: ForwardContext {
                text: text.context,
                segments,
                vision,
                vision_initial,
                vision_output: None,
                tokens,
                feedback: input.refinement().map(|feedback| feedback.cloned()),
                deferred,
                vision_spans: row_spans,
                pending_text_boundary: None,
            },
        })
    }
    fn assemble(
        &mut self,
        forward: &ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let embedding = self
            .statics
            .text
            .encode_context_tokens(&forward.tokens, context)?;
        self.assemble_embeddings(forward, &embedding, context)
    }

    fn assemble_embeddings(
        &self,
        forward: &ForwardContext<B::Tensor>,
        embedding: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let mut token_position = 0;
        let mut vision_position = 0;
        let mut parts = Vec::new();
        for (tokens, projected, native_vision) in &forward.segments {
            let positions = tokens.dim(1);
            parts.push(if let Some(projected) = projected {
                projected.clone()
            } else if *native_vision {
                let output = forward
                    .vision_output
                    .as_ref()
                    .ok_or_else(|| Error::backend("image group has no projected output"))?;
                let value = output
                    .reshape(&[1, -1, self.config.text.hidden_size], context)?
                    .index(
                        &[
                            Index::Full,
                            Index::Range(vision_position, vision_position + positions),
                            Index::Full,
                        ],
                        context,
                    )?;
                vision_position += positions;
                value
            } else {
                embedding.index(
                    &[
                        Index::Full,
                        Index::Range(token_position, token_position + positions),
                        Index::Full,
                    ],
                    context,
                )?
            });
            token_position += positions;
        }
        B::Tensor::concatenate(&parts, 1, context)
    }
}

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> ArchitectureParameters<B> for Model<B> {
    type DefinitionError = Error;
    fn state_layout(&self) -> Result<StateLayout, Error> {
        Ok(self.definition.state_layout().clone())
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        self.statics.text.state_identity(state, topology)
    }
    fn parameter_description(
        &self,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        Ok(self.definition.parameters().clone())
    }

    fn visit_static_parameters<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.statics.text.visit_pinned(visitor)?;
        visitor.visit("vision", &self.statics.image.tower)?;
        visitor.visit("vision_projection", &self.statics.image.projector)
    }
    fn visit_static_parameters_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.statics.text.visit_pinned_mut(visitor)?;
        visitor.visit_mut("vision", &mut self.statics.image.tower)?;
        visitor.visit_mut("vision_projection", &mut self.statics.image.projector)
    }
}

impl<B, S> LayeredArchitecture<B, S> for Model<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    type Input<'a>
        = PreparedCompositeInput<'a, B::Tensor, Gemma4InputPartPlan>
    where
        Self: 'a;
    type StaticModules = StaticModules<B>;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        <TextKernel<B> as eredu_runtime::ComponentEquations<B, S>>::observation_hooks(
            &self.statics.text,
        )
    }
    fn output_score_kind(
        forward: &Self::ForwardContext,
    ) -> Option<eredu_core::refinement::PredictionScoreKind> {
        <TextKernel<B> as eredu_runtime::ComponentKernel<B>>::output_score_kind(&forward.text)
    }
    fn group_transport(&self, group: usize) -> ArchitectureGroupTransport {
        self.definition
            .group_transport(group)
            .expect("validated execution group")
            .clone()
    }
    fn primary_execution_group(&self) -> &str {
        self.definition.primary_group()
    }
    fn state_partition_plan(
        &self,
        _layout: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        self.definition.state_partition_plan().clone()
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        Ok(self.definition.parameters().graph().clone())
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        self.definition.unit_count(group).map_err(Error::from)
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        Ok(self.definition.unit(group, index)?.path.clone())
    }
    fn static_modules(&self) -> &Self::StaticModules {
        &self.statics
    }
    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.statics
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        if group == 0 {
            Ok(Unit::Vision(ImageStatic::<B>::build_layer(
                &self.config,
                index,
                context,
            )?))
        } else if group == 1 {
            Ok(Unit::Text(
                <TextKernel<B> as eredu_runtime::ComponentKernel<B>>::build_unit(
                    &self.statics.text,
                    self.statics.text.definition().unit_address(0, index)?,
                    context,
                )?,
            ))
        } else {
            Err(Error::backend("invalid image/text group"))
        }
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_input(input, state, self.geometry.is_some(), context)
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group == 1 {
            if let Some(output) = dependencies.first() {
                forward.vision_output = Some((*output).clone());
            }
            if forward.deferred {
                forward.deferred = false;
                if let Some(feedback) = &forward.feedback {
                    let prepared = self.statics.text.begin(
                        TextInput::Refinement {
                            tokens: &forward.tokens,
                            feedback: feedback.as_ref(),
                            mask: None,
                        },
                        state,
                        context,
                    )?;
                    forward.text = prepared.context;
                    return Ok(prepared.hidden);
                }
                return self.assemble(forward, context);
            }
        }
        match group {
            0 => Ok(forward.vision_initial.as_ref().unwrap_or(initial).clone()),
            1 => {
                if forward.vision.is_some()
                    || forward
                        .segments
                        .iter()
                        .any(|(_, projected, _)| projected.is_some())
                {
                    self.assemble(forward, context)
                } else {
                    Ok(initial.clone())
                }
            }
            _ => Err(Error::backend("invalid image/text group")),
        }
    }
    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group == 1 || (group == 0 && forward.vision.is_some())
    }
    fn state_ordinal(&self, group: usize, index: usize, ordinal: usize) -> usize {
        if group == 1 {
            index
        } else {
            ordinal
        }
    }
    fn retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        _ordinal: usize,
    ) -> std::ops::Range<usize> {
        if group == 1 {
            index..index + 1
        } else {
            0..0
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
    ) -> Result<B::Tensor, Error> {
        <Self as RoutedLayeredArchitecture<B, S>>::forward_unit_with_provider(
            self,
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            eredu_runtime::ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            context,
        )
    }
    fn forward_unit_observed<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        <Self as RoutedLayeredArchitecture<B, S>>::forward_unit_observed_with_provider(
            self,
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            eredu_runtime::ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            context,
            observer,
        )
    }
    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if let (0, Some(vision)) = (group, forward.vision.as_ref()) {
            let output = self.statics.image.finish(hidden, vision, context)?;
            forward.vision_output = Some(output.clone());
            Ok(output)
        } else {
            Ok(hidden.clone())
        }
    }
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.statics
            .text
            .finish(hidden, state, &forward.text, context)
    }
    fn finish_forward_observed<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        self.statics
            .text
            .finish_observed(hidden, state, &forward.text, context, observer)
    }
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = Vec::new();
        values.extend(
            <TextKernel<B> as eredu_runtime::ComponentKernel<B>>::retained_values(
                &self.statics.text,
                &forward.text,
                self.statics
                    .text
                    .definition()
                    .unit_address(0, 0)
                    .expect("declared shared stack"),
            ),
        );
        if let Some(vision) = &forward.vision {
            values.extend(vision.retained_values());
        }
        values.extend(forward.vision_initial.iter());
        values.extend(forward.vision_output.iter());
        values.push(&forward.tokens);
        values.extend(forward.feedback.as_ref().and_then(Option::as_ref));
        for (tokens, projected, _) in &forward.segments {
            values.push(tokens);
            values.extend(projected.iter());
        }
        values.into_iter()
    }
    fn projects_final_text_position() -> bool {
        false
    }
}

impl<B, S> RoutedLayeredArchitecture<B, S> for Model<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn routed_unit_observations(&self) -> bool {
        true
    }
    fn routed_sparse_observations(&self) -> bool {
        true
    }
    fn forward_unit_with_provider<P: ParameterProvider<B>>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P::Error: std::fmt::Display,
    {
        self.forward_unit_observed_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            pass,
            provider,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn forward_unit_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        match (group, unit) {
            (0, Unit::Vision(unit)) => self.statics.image.tower.forward_layer(
                unit,
                hidden,
                forward
                    .vision
                    .as_ref()
                    .ok_or_else(|| Error::backend("missing image invocation state"))?,
                context,
            ),
            (1, Unit::Text(unit)) => self.statics.text.execute_routed_observed(
                self.statics.text.definition().unit_address(0, index)?,
                unit,
                hidden,
                state,
                &mut forward.text,
                pass,
                provider,
                context,
                observer,
            ),
            _ => Err(Error::backend("unit/group mismatch")),
        }
    }
}

impl<B, S> CompositeArchitecture<B, S> for Model<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn prepared_publication_width(
        &self,
        input: PreparedCompositeInput<'_, B::Tensor, Self::InputPartPlan>,
    ) -> Result<Option<i32>, String> {
        Ok(Some(if input.refinement().is_some() {
            self.config.text.vocab_size
        } else {
            self.config.text.hidden_size
        }))
    }
    fn accept_partition_boundary(
        &mut self,
        source: usize,
        destination: usize,
        schema: &eredu_runtime::ResolvedBoundaryWireSchema,
        values: Vec<B::Tensor>,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error> {
        if values.len() != 1 + schema.auxiliary().len() {
            return Err(Error::backend(
                "composite boundary has incomplete typed context",
            ));
        }
        let mut values = values.into_iter();
        let hidden = values.next().unwrap();
        match (source, destination) {
            (0, 0) | (0, 1) => {}
            (1, 1) => {
                forward.pending_text_boundary = Some(
                    eredu_runtime::ArchitectureBoundary::decode(
                        &BoundarySchema::new(self.config.text.hidden_size),
                        values.collect(),
                    )
                    .map_err(Error::backend)?,
                );
            }
            _ => return Ok(None),
        }
        Ok(Some(hidden))
    }
    fn routed_tensor_reductions(
        &self,
        _unit: usize,
        routed: bool,
    ) -> Result<eredu_runtime::RoutedTensorReductions, Error> {
        Ok(eredu_runtime::RoutedTensorReductions::hidden(
            2,
            usize::from(routed),
        ))
    }
    fn primary_ingress_collectives_pending(&self, forward: &Self::ForwardContext) -> bool {
        forward.deferred
    }
    fn prepared_primary_ingress_collectives(
        &self,
        input: PreparedCompositeInput<'_, B::Tensor, Gemma4InputPartPlan>,
        tensor_partitions: usize,
    ) -> Result<Option<Vec<crate::composite_execution::CompositeTensorCollective>>, String> {
        if tensor_partitions <= 1 {
            return Ok(Some(Vec::new()));
        }
        let rows = i32::try_from(input.admitted().decoder_positions())
            .map_err(|_| "primary ingress extent exceeds i32")?;
        let copies = input
            .refinement()
            .map_or(1, |feedback| 2 + usize::from(feedback.is_some()));
        Ok(Some(vec![
                crate::composite_execution::CompositeTensorCollective::Sum {
                    shape: vec![1, rows, self.config.text.hidden_size]
                };
                copies
            ]))
    }
    fn prepared_group_collective_waves(
        &self,
        group: usize,
        _input: PreparedCompositeInput<'_, B::Tensor, Gemma4InputPartPlan>,
        _tensor_partitions: usize,
        stage_units: &[std::ops::Range<usize>],
    ) -> Result<Option<Vec<Vec<crate::composite_execution::CompositeTensorCollective>>>, String>
    {
        let pipeline_stages = stage_units.len();
        Ok((group == 0).then(|| vec![Vec::new(); pipeline_stages]))
    }
    fn prepared_group_boundary_sequence(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Gemma4InputPartPlan>,
    ) -> Result<i32, String> {
        let positions = if group == 0 {
            input
                .admitted()
                .parts()
                .iter()
                .filter_map(|part| match part {
                    Gemma4InputPartPlan::Vision { shape, .. } => Some(shape.decoder_positions),
                    _ => None,
                })
                .try_fold(0u64, |total, count| total.checked_add(count))
                .ok_or("vision boundary extent overflow")?
        } else {
            input.admitted().decoder_positions()
        };
        i32::try_from(positions).map_err(|_| "group boundary extent exceeds i32".into())
    }
    fn prepared_group_continuation_geometry(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Gemma4InputPartPlan>,
    ) -> Result<Option<crate::composite_execution::CompositeContinuationGeometry>, String> {
        if group != 0 {
            return Ok(None);
        }
        let patches = input
            .admitted()
            .parts()
            .iter()
            .filter_map(|part| match part {
                Gemma4InputPartPlan::Vision { ingress, .. } => Some(ingress.padded_patches),
                _ => None,
            })
            .collect::<Vec<_>>();
        patches
            .iter()
            .max()
            .map(|maximum| {
                i32::try_from(patches.len())
                    .ok()
                    .and_then(|count| maximum.checked_mul(count))
                    .map(|extent| (extent, self.config.vision.as_ref().unwrap().hidden_size).into())
                    .ok_or_else(|| "batched image continuation extent exceeds i32".into())
            })
            .transpose()
    }
    fn prepared_group_continuation_batched(&self, _group: usize) -> bool {
        false
    }
    fn encode_group_continuation(
        &self,
        group: usize,
        hidden: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group == 0 {
            hidden.reshape(&[1, -1, hidden.dim(2)], context)
        } else {
            Ok(hidden)
        }
    }
    fn decode_group_continuation(
        &self,
        group: usize,
        hidden: B::Tensor,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group == 0 {
            hidden.reshape(
                forward
                    .vision_initial
                    .as_ref()
                    .ok_or_else(|| {
                        Error::backend("image continuation has no admitted batch geometry")
                    })?
                    .shape(),
                context,
            )
        } else {
            Ok(hidden)
        }
    }
    fn supports_chunked_text_prefill() -> bool {
        true
    }
    type InputPartPlan = Gemma4InputPartPlan;
    type PrefillRequest = SingleCompositePrefillRequest<B::Tensor, Gemma4InputPartPlan>;
    type AdmissionConfig = FamilyConfig;
    fn block_refinement_geometry(
        config: &FamilyConfig,
    ) -> Option<eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry> {
        Some(
            eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry {
                canvas_length: config.canvas_length,
                vocabulary_size: std::num::NonZeroU32::new(config.text.vocab_size as u32).unwrap(),
            },
        )
    }
    fn prepare_prefill_request(
        prepared: eredu_runtime::PreparedModelInput<B::Tensor>,
        admitted: AdmittedCompositeInput<Gemma4InputPartPlan>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::PrefillRequest, Error> {
        SingleCompositePrefillRequest::new(prepared, admitted)
    }
    fn admission_config(&self) -> FamilyConfig {
        self.config.clone()
    }
    fn admit_prepared_input(
        config: &FamilyConfig,
        input: &eredu_runtime::PreparedModelInput<B::Tensor>,
        inspector: &impl eredu_runtime::PreparedInputInspector<B::Tensor>,
    ) -> Result<AdmittedCompositeInput<Gemma4InputPartPlan>, eredu_core::CapabilityError> {
        crate::media_plan::admit_gemma4_input(&ingress_config(config), input, inspector)
    }
    fn should_execute_prepared_group(
        &self,
        group: usize,
        input: PreparedCompositeInput<'_, B::Tensor, Gemma4InputPartPlan>,
    ) -> bool {
        group == 1
            || (group == 0
                && input.refinement().is_none()
                && input
                    .admitted()
                    .parts()
                    .iter()
                    .any(|part| matches!(part, Gemma4InputPartPlan::Vision { .. })))
    }
    fn begin_composite_forward<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Gemma4InputPartPlan>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_forward(input, state, context)
    }
}
