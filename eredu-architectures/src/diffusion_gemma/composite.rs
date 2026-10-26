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
    ArchitectureGroupTransport, ArchitectureParameterDescription, ExecutionGraph,
    ExecutionGroupSpec, ExecutionUnitLayout, LayerRuntimeState, LayeredForwardState,
    OwnedParameterGroupSpec, ParameterGroupOwner, ParameterProvider, ResidentExpertProvider,
    StateLayout,
};

use eredu_runtime::{
    ComponentEquations as _, LayeredPartitionInput, LayeredPartitionOutput,
    ParallelRoutedComponentKernel as _,
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

/// Shared runtime contracts over the composite's declared resources and typed equations.
pub type Model<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    CompositeKernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// Family-owned image assembly and shared context/refinement equations.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct CompositeKernel<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    #[parameter(skip)]
    config: FamilyConfig,
    #[parameter(skip)]
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    statics: StaticModules<B>,
    #[parameter(skip)]
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
    let projection = eredu_runtime::expand_linear_format_parameter_group(projection, |member| {
        crate::linear_format::standard_parallel_linear_format(
            member,
            config.text.linear_format_for(member.target()),
        )
    })
    .map_err(Error::backend_source)?;
    owned.push(OwnedParameterGroupSpec::new(
        ParameterGroupOwner::static_role("vision_projection"),
        projection,
    ));
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
    ArchitectureParameterDescription::from_owned(&graph, &units, owned)
        .map_err(Error::backend_source)
}

const COMPONENT_ID: &str = "diffusion_gemma_composite";

fn component_units(config: &FamilyConfig) -> Result<Vec<eredu_runtime::ComponentUnit>, Error> {
    let vision = config
        .vision
        .as_ref()
        .ok_or_else(|| Error::backend("composite requires an image tower"))?;
    Ok((0..vision.num_hidden_layers as usize)
        .map(|index| eredu_runtime::ComponentUnit {
            path: format!("model.encoder.vision_tower.encoder.layers.{index}"),
            state: 0..0,
        })
        .chain(
            (0..config.text.num_hidden_layers()).map(|index| eredu_runtime::ComponentUnit {
                path: format!("model.decoder.layers.{index}"),
                state: index..index + 1,
            }),
        )
        .collect())
}

fn state_partition_plan(config: &FamilyConfig) -> eredu_runtime::ArchitectureStatePartitionPlan {
    eredu_runtime::ArchitectureStatePartitionPlan::new([
        eredu_runtime::ArchitectureStatePartitionRule::group_units(
            1,
            0..config.text.num_hidden_layers(),
        ),
    ])
}

fn component_state(
    config: &FamilyConfig,
    geometry: Option<&super::parallel::CompositeGeometry>,
) -> Result<StateLayout, Error> {
    let layers = config.text.num_hidden_layers();
    let vision_layers = config
        .vision
        .as_ref()
        .ok_or_else(|| Error::backend("composite requires an image tower"))?
        .num_hidden_layers as usize;
    match geometry {
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
                || range.len() != geometry.text.state_layout().map_or(0, StateLayout::len)
                || range
                    .clone()
                    .zip(
                        geometry
                            .text
                            .state_layout()
                            .into_iter()
                            .flat_map(|state| state.layers().iter()),
                    )
                    .any(|(global, local)| {
                        geometry.complete_state.layers().get(global) != Some(local)
                    })
            {
                return Err(Error::backend(
                    "composite local geometry changes declared unit or state ownership",
                ));
            }
            Ok(geometry.complete_state.clone())
        }
        None => super::state_layout(config).map_err(Error::backend_source),
    }
}

/// Validates image invocations and shared text state without native resources.
pub fn component_definition(
    config: &FamilyConfig,
    geometry: Option<&super::parallel::CompositeGeometry>,
) -> Result<eredu_runtime::ComponentDefinition, Error> {
    eredu_runtime::ComponentDefinition::new(
        COMPONENT_ID,
        parameter_description(config)?,
        component_units(config)?,
        vec![vision_transport(), super::model::text_transport()],
        super::model::TEXT_EXECUTION_GROUP,
        vec![],
        component_state(config, geometry)?,
        state_partition_plan(config),
    )
    .map_err(Error::from)
}

/// Selects family-authored executable formats while retaining parent semantic ownership.
pub fn project_parameters(
    config: &FamilyConfig,
    definition: &std::sync::Arc<eredu_runtime::ComponentDefinition>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, Error> {
    validate_identity(definition)?;
    let parameters = parameter_description(config)?;
    if definition.parameters() == &parameters {
        return Ok(definition.clone());
    }
    let uses = definition.parameter_uses().clone();
    definition
        .as_ref()
        .clone()
        .with_parameters(parameters, uses)
        .map(std::sync::Arc::new)
        .map_err(Error::from)
}

fn validate_identity(definition: &eredu_runtime::ComponentDefinition) -> Result<(), Error> {
    if definition.id() != COMPONENT_ID {
        return Err(eredu_runtime::ComponentDefinitionError::Invalid {
            component: definition.id().into(),
            resource: "composite component identity".into(),
        }
        .into());
    }
    Ok(())
}

fn select_definition(
    config: &FamilyConfig,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    geometry: Option<&super::parallel::CompositeGeometry>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, Error> {
    validate_identity(&definition)?;
    definition.validate_parameter_topology(&parameter_description(config)?)?;
    definition.validate_execution_contract(
        super::model::TEXT_EXECUTION_GROUP,
        &[],
        &component_units(config)?,
        &[vision_transport(), super::model::text_transport()],
        &state_partition_plan(config),
        |_, _, boundaries| !boundaries.is_kernel_owned(),
    )?;
    let state = component_state(config, geometry)?;
    if definition.state_layout() == &state {
        Ok(definition)
    } else {
        definition
            .as_ref()
            .clone()
            .with_state_layout(state)
            .map(std::sync::Arc::new)
            .map_err(Error::from)
    }
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

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> CompositeKernel<B> {
    /// Declares shared text and independently materialized image modules.
    pub fn from_definition(
        config: FamilyConfig,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Model<B>, Error> {
        let definition = select_definition(&config, definition, None)?;
        let text_definition = super::model::project_from_composite(&definition)?;
        let kernel = Self {
            definition,
            statics: StaticModules {
                text: TextKernel::materialize(config.clone(), text_definition, None, context)?,
                image: ImageStatic::new(&config, context)?,
            },
            config,
            geometry: None,
        };
        eredu_runtime::ComponentAdapter::with_pinned_resources::<B>(
            std::sync::Arc::clone(&kernel.definition),
            kernel,
        )
        .map_err(Error::from)
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
            let frontier = if state.optional_layout().is_none() {
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
                    if self.geometry.as_ref().is_none_or(|geometry| {
                        !geometry.vision_units.is_empty() && geometry.vision_units.start == 0
                    }) {
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
                            self.statics.image.prepare_state(input, context)?,
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
            self.kernel().config.text.vocab_size
        } else {
            self.kernel().config.text.hidden_size
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
                        &BoundarySchema::new(self.kernel().config.text.hidden_size),
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
                    shape: vec![1, rows, self.kernel().config.text.hidden_size]
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
                    .map(|extent| {
                        (
                            extent,
                            self.kernel().config.vision.as_ref().unwrap().hidden_size,
                        )
                            .into()
                    })
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
    fn prepared_input<'a>(
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    ) -> Self::Input<'a> {
        input
    }
    fn prepared_view<'a>(
        input: &Self::Input<'a>,
    ) -> Option<
        crate::composite_execution::PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
    > {
        Some(*input)
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
        self.kernel().config.clone()
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
}

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> eredu_runtime::ComponentKernel<B>
    for CompositeKernel<B>
{
    type Input<'a>
        = PreparedCompositeInput<'a, B::Tensor, Gemma4InputPartPlan>
    where
        Self: 'a;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::vec::IntoIter<&'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        self.statics.text.state_identity(state, topology)
    }
    fn output_score_kind(
        forward: &Self::ForwardContext,
    ) -> Option<eredu_core::refinement::PredictionScoreKind> {
        <TextKernel<B> as eredu_runtime::ComponentKernel<B>>::output_score_kind(&forward.text)
    }
    fn build_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        let group = address.group();
        let index = address.index();
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
    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group == 1 || (group == 0 && forward.vision.is_some())
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _address: eredu_runtime::ExecutionUnitAddress,
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
}
impl<B: GroupedNeuralBackend + DistributedNeuralBackend> eredu_runtime::ComponentPinnedResources<B>
    for CompositeKernel<B>
{
    type Modules = StaticModules<B>;
    fn pinned_modules(&self) -> &Self::Modules {
        &self.statics
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        &mut self.statics
    }
    fn visit_pinned<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.statics.text.visit_pinned(visitor)?;
        if let Some(tower) = &self.statics.image.tower {
            visitor.visit("vision", tower)?;
        }
        if let Some(projector) = &self.statics.image.projector {
            visitor.visit("vision_projection", projector)?;
        }
        Ok(())
    }
    fn visit_pinned_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.statics.text.visit_pinned_mut(visitor)?;
        if let Some(tower) = &mut self.statics.image.tower {
            visitor.visit_mut("vision", tower)?;
        }
        if let Some(projector) = &mut self.statics.image.projector {
            visitor.visit_mut("vision_projection", projector)?;
        }
        Ok(())
    }
}

impl<B, S> eredu_runtime::ComponentEquations<B, S> for CompositeKernel<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        <TextKernel<B> as eredu_runtime::ComponentEquations<B, S>>::observation_hooks(
            &self.statics.text,
        )
    }
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_input(input, state, self.geometry.is_some(), context)
    }
    fn begin_group(
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
    fn complete_group(
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
    fn finish(
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
    fn finish_observed<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
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
    fn execute(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        <Self as eredu_runtime::RoutedComponentKernel<B, S>>::execute_routed(
            self,
            address,
            unit,
            hidden,
            state,
            forward,
            eredu_runtime::ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            context,
        )
    }
    fn execute_observed<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        <Self as eredu_runtime::RoutedComponentKernel<B, S>>::execute_routed_observed(
            self,
            address,
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
}

impl<B, S> eredu_runtime::RoutedComponentKernel<B, S> for CompositeKernel<B>
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
    fn execute_routed<P: ParameterProvider<B>>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        self.execute_routed_observed(
            address,
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
    fn execute_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        let group = address.group();
        let index = address.index();
        match (group, unit) {
            (0, Unit::Vision(unit)) => self.statics.image.forward_layer(
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

impl<B, S> eredu_runtime::ParallelComponentKernel<B, S> for CompositeKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        <TextKernel<B> as eredu_runtime::ParallelComponentKernel<B, S>>::parallel_observation_hooks(
            &self.statics.text,
        )
    }
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        _parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin_input(input, state, true, context)
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
    ) -> Result<B::Tensor, Error> {
        if group != 1 {
            return self.begin_group(group, initial, dependencies, state, forward, context);
        }
        if let Some(output) = dependencies.first() {
            forward.vision_output = Some((*output).clone());
        }
        if !forward.deferred {
            return Ok(initial.clone());
        }
        forward.deferred = false;
        if let Some(feedback) = &forward.feedback {
            let prepared = self.statics.text.begin_parallel(
                TextInput::Refinement {
                    tokens: &forward.tokens,
                    feedback: feedback.as_ref(),
                    mask: None,
                },
                state,
                parallel,
                context,
            )?;
            forward.text = prepared.context;
            return Ok(prepared.hidden);
        }
        let embeddings =
            self.statics
                .text
                .encode_context_tokens_parallel(&forward.tokens, parallel, context)?;
        self.assemble_embeddings(forward, &embeddings, context)
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
    ) -> Result<B::Tensor, Error> {
        self.execute_parallel_routed(
            address,
            unit,
            hidden,
            state,
            forward,
            eredu_runtime::ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            parallel,
            context,
        )
    }
    fn execute_parallel_observed<
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    >(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        self.execute_parallel_routed_observed(
            address,
            unit,
            hidden,
            state,
            forward,
            eredu_runtime::ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            parallel,
            context,
            observer,
        )
    }
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.statics
            .text
            .finish_parallel(hidden, state, &forward.text, parallel, context)
    }
    fn finish_parallel_observed<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        self.statics.text.finish_parallel_observed(
            hidden,
            state,
            &forward.text,
            parallel,
            context,
            observer,
        )
    }
}

impl<B, S> eredu_runtime::ParallelRoutedComponentKernel<B, S> for CompositeKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        true
    }
    fn parallel_routed_sparse_observations(&self) -> bool {
        true
    }
    fn execute_parallel_routed<P: eredu_runtime::TensorParallelParameterProvider<B>>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P::Error: std::fmt::Display,
    {
        self.execute_parallel_routed_observed(
            address,
            unit,
            hidden,
            state,
            forward,
            pass,
            provider,
            parallel,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn execute_parallel_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
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
        let group = address.group();
        let index = address.index();
        match (group, unit) {
            (0, Unit::Vision(unit)) => self.statics.image.forward_layer(
                unit,
                hidden,
                forward
                    .vision
                    .as_ref()
                    .ok_or_else(|| Error::backend("missing image invocation"))?,
                context,
            ),
            (1, Unit::Text(unit)) => self.statics.text.execute_parallel_routed_observed(
                self.statics.text.definition().unit_address(0, index)?,
                unit,
                hidden,
                state,
                &mut forward.text,
                pass,
                provider,
                parallel,
                context,
                observer,
            ),
            _ => Err(Error::backend("unit/group mismatch")),
        }
    }
}

impl<B, S> eredu_runtime::PartitionComponentKernel<B, S> for CompositeKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    type Boundary = BoundarySchema;
    fn boundary_schema(&self) -> Result<BoundarySchema, Error> {
        Ok(BoundarySchema::new(self.config.text.hidden_size))
    }
    fn partition_observation_hooks(
        &self,
        parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        <TextKernel<B> as eredu_runtime::PartitionComponentKernel<B, S>>::partition_observation_hooks(
            &self.statics.text,
            parallel,
        )
    }
    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, Boundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let expected = selected
            .expected
            .ok_or_else(|| Error::backend("text continuation requires selected context slots"))?;
        self.resume(
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
        group: usize,
        initial: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if let Some(auxiliary) = forward.pending_text_boundary.take() {
            if group != 1 {
                return Err(Error::backend(
                    "text continuation entered another execution group",
                ));
            }
            let geometry = self
                .geometry
                .as_ref()
                .ok_or_else(|| Error::backend("text continuation has no retained ownership"))?;
            let expected = geometry
                .text
                .state_layout()
                .ok_or_else(|| Error::backend("text continuation has no owned context slots"))?
                .clone();
            let first = geometry.text.units().start;
            let resumed = self.resume(
                LayeredPartitionInput::Hidden {
                    hidden: initial.clone(),
                    auxiliary,
                },
                None,
                state,
                &expected,
                first,
                parallel,
                context,
            )?;
            *forward = resumed.context;
            return Ok(resumed.hidden);
        }
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
    ) -> Result<LayeredPartitionOutput<B::Tensor, Boundary<B::Tensor>>, Error> {
        self.finish_cut(
            hidden,
            state,
            forward,
            owns_output,
            parallel,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn finish_partition_observed<
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    >(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredPartitionOutput<B::Tensor, Boundary<B::Tensor>>, Error> {
        self.finish_cut(
            hidden,
            state,
            forward,
            owns_output,
            parallel,
            context,
            observer,
        )
    }
}
