//! Composite Qwen3-VL lifecycle over the shared vision tower and ordinary Qwen decoder.

use eredu_core::cache::StateTensorRole;
use eredu_nn::{
    multimodal::{assemble_ordered_inputs, OrderedInputPart},
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, Index, LinearOperator,
    NormalizationOperator, Parameterized, RotaryPosition, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout,
    ExpertPass, LayerRuntimeState, LayeredArchitecture, LayeredForwardState, LayeredPartitionInput,
    LayeredPartitionOutput, OwnedParameterGroupSpec, ParallelLayeredArchitecture,
    ParallelRoutedLayeredArchitecture, ParameterGroupOwner, PartitionedLayeredArchitecture,
    RoutedExpertProvider, RoutedLayeredArchitecture, RuntimeStateComponents, StateLayout,
};

use crate::decoder::static_parallel_parameter_groups;
use crate::qwen::vision::{
    block_parallel_parameter_groups, static_parallel_parameter_groups as vision_parameter_groups,
    VisionBlock, VisionInput, VisionMode, VisionState, VisionStatic,
};
use crate::qwen::{self, AttentionInput};

use super::{
    mrope_embeddings, multimodal_position_ids, position_ids_tensor, LocalGeometry, ModelArgs,
    PositionPart,
};

/// One semantic segment in decoder order.
pub enum InputPart<'a, T> {
    /// Ordinary text token IDs.
    Text(&'a T),
    /// Image placeholders and their unmerged patch grids.
    Image {
        /// Placeholder IDs shaped `[1, merged_patches]`.
        tokens: &'a T,
        /// One or more `(time, height, width)` patch grids.
        grid: &'a [(i32, i32, i32)],
    },
    /// Video placeholders and their unmerged patch grids.
    Video {
        /// Placeholder IDs shaped `[1, merged_patches]`.
        tokens: &'a T,
        /// One or more `(time, height, width)` patch grids.
        grid: &'a [(i32, i32, i32)],
    },
    /// Already projected decoder-width embeddings.
    Projected {
        /// Semantic token identities.
        tokens: &'a T,
        /// Embeddings shaped `[1, sequence, hidden]`.
        embeddings: &'a T,
    },
}

/// Prepared text and optional model-native visual input.
pub struct ModelInput<'a, T> {
    /// Ordered text and media segments.
    pub parts: &'a [InputPart<'a, T>],
    /// Flattened patches for every image/video part in order.
    pub pixels: Option<&'a T>,
    /// Optional explicit text attention mask.
    pub mask: Option<&'a T>,
}

enum PreparedPart<T> {
    Text { tokens: T, embeddings: T },
    Media { tokens: T },
}

/// Pinned text and shared-vision modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: RoutedNeuralBackend> {
    /// Ordinary Qwen embeddings, final norm, and vocabulary head.
    pub text: qwen::StaticModules<B>,
    /// Qwen shared vision patch, position, and merger modules.
    pub vision: VisionStatic<B>,
}

/// One streamable vision or ordinary Qwen text unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: RoutedNeuralBackend> {
    /// Shared vision transformer block.
    Vision(VisionBlock<B>),
    /// Existing neutral ordinary Qwen dense-or-MoE block.
    Text(qwen::RoutedTransformerBlock<B>),
}

impl<B, S> RoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        LayeredModel::forward_unit_with_provider(
            self, group, index, unit, hidden, state, forward, provider, context,
        )
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        LayeredModel::forward_unit_with_provider_parallel(
            self, group, index, unit, hidden, state, forward, provider, parallel, context,
        )
    }
}

/// Architecture-owned values retained for one complete pass.
pub struct ForwardContext<T> {
    mask: Option<T>,
    tokens: Option<T>,
    parts: Vec<PreparedPart<T>>,
    rotary: (T, T),
    position_delta: T,
    vision_state: Option<VisionState<T>>,
    vision_initial: Option<T>,
    vision_output: Option<T>,
    deepstack: Vec<T>,
    visual_mask: Option<T>,
}

/// Request-scoped state transported while pipeline owners execute the shared
/// vision tower.
pub struct PipelineVisionState<T> {
    /// Current vision activation.
    pub hidden: T,
    parts: Vec<PreparedPart<T>>,
    rotary: (T, T),
    delta: T,
    mask: Option<T>,
    vision: Option<VisionState<T>>,
    vision_output: Option<T>,
}

/// Decoder-facing values produced after the placed vision group completes.
pub struct PipelinePrepared<T> {
    /// Assembled text-width activation.
    pub hidden: T,
    /// Text mRoPE cosine.
    pub cosine: T,
    /// Text mRoPE sine.
    pub sine: T,
    /// Persisted decode position delta.
    pub position_delta: T,
    /// Optional explicit or causal mask.
    pub mask: Option<T>,
    /// Selected raw DeepStack features.
    pub deepstack: Vec<T>,
    /// Image-or-video placeholder mask used for DeepStack scatter.
    pub visual_mask: Option<T>,
}

impl<T: Clone> PipelinePrepared<T> {
    /// Converts decoder preparation into the canonical layered forward state.
    pub fn into_layered_forward(self) -> (LayeredForwardState<T, ForwardContext<T>>, T) {
        let position_delta = self.position_delta;
        (
            LayeredForwardState {
                hidden: self.hidden,
                context: ForwardContext {
                    mask: self.mask,
                    tokens: None,
                    parts: Vec::new(),
                    rotary: (self.cosine, self.sine),
                    position_delta: position_delta.clone(),
                    vision_state: None,
                    vision_initial: None,
                    vision_output: None,
                    deepstack: self.deepstack,
                    visual_mask: self.visual_mask,
                },
            },
            position_delta,
        )
    }

    /// Recovers the transport boundary from a completed layered forward.
    pub fn from_layered_forward(
        forward: LayeredForwardState<T, ForwardContext<T>>,
        position_delta: T,
    ) -> (T, PipelineBoundary<T>) {
        (
            forward.hidden,
            PipelineBoundary {
                cosine: forward.context.rotary.0,
                sine: forward.context.rotary.1,
                position_delta,
                deepstack: forward.context.deepstack,
            },
        )
    }
}

/// Family-owned schema for decoder values transported between pipeline ranks.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PipelineBoundarySchema {
    head_dim: i32,
    hidden_size: i32,
    deepstack_count: usize,
}

impl PipelineBoundarySchema {
    /// Derives the boundary schema from the normalized multimodal model.
    pub fn from_args(args: &ModelArgs) -> Self {
        Self {
            head_dim: args.text.head_dim,
            hidden_size: args.text.hidden_size,
            deepstack_count: args.vision.deepstack_layer_count(),
        }
    }

    /// Returns the configured number of DeepStack feature tensors.
    pub const fn deepstack_count(self) -> usize {
        self.deepstack_count
    }
}

impl eredu_runtime::ArchitectureBoundary for PipelineBoundarySchema {
    type Boundary<T> = PipelineBoundary<T>;

    const IDENTITY: &'static str = "qwen_vl.decoder";

    fn primary_tensor_spec(&self) -> eredu_runtime::BoundaryTensorSpec {
        eredu_runtime::BoundaryTensorSpec::primary_activation(self.hidden_size)
    }

    fn auxiliary_tensor_specs(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
        let mut specs = vec![
            eredu_runtime::BoundaryTensorSpec::new(
                "cosine",
                [Dim::Sequence, Dim::Fixed(self.head_dim)],
                Dtype::Activation,
            ),
            eredu_runtime::BoundaryTensorSpec::new(
                "sine",
                [Dim::Sequence, Dim::Fixed(self.head_dim)],
                Dtype::Activation,
            ),
            eredu_runtime::BoundaryTensorSpec::new("position_delta", [Dim::Fixed(1)], Dtype::Int32),
        ];
        specs.extend((0..self.deepstack_count).map(|index| {
            eredu_runtime::BoundaryTensorSpec::new(
                format!("deepstack.{index}"),
                [Dim::Batch, Dim::Sequence, Dim::Fixed(self.hidden_size)],
                Dtype::Activation,
            )
        }));
        specs
    }

    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<Self::Boundary<T>, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        let mut tensors = tensors.into_iter();
        Ok(PipelineBoundary {
            cosine: tensors.next().expect("validated mRoPE cosine"),
            sine: tensors.next().expect("validated mRoPE sine"),
            position_delta: tensors.next().expect("validated position delta"),
            deepstack: tensors.collect(),
        })
    }

    /// Encodes a typed boundary after validating its configured cardinality.
    fn encode<T>(
        &self,
        boundary: PipelineBoundary<T>,
    ) -> Result<Vec<T>, eredu_runtime::ArchitectureBoundaryError> {
        if boundary.deepstack.len() != self.deepstack_count {
            return Err(eredu_runtime::ArchitectureBoundaryError::TensorCount {
                boundary: "qwen_vl.decoder.deepstack",
                expected: self.deepstack_count,
                actual: boundary.deepstack.len(),
            });
        }
        Ok(std::iter::once(boundary.cosine)
            .chain(std::iter::once(boundary.sine))
            .chain(std::iter::once(boundary.position_delta))
            .chain(boundary.deepstack)
            .collect())
    }
}

/// Typed immutable decoder context transported between Qwen-VL partitions.
pub struct PipelineBoundary<T> {
    /// Text mRoPE cosine.
    pub cosine: T,
    /// Text mRoPE sine.
    pub sine: T,
    /// Persisted decode position delta.
    pub position_delta: T,
    /// Per-layer DeepStack features.
    pub deepstack: Vec<T>,
}

impl<T> PipelineBoundary<T> {
    /// Splits prepared decoder state into its evolving activation and boundary.
    pub fn from_prepared(prepared: PipelinePrepared<T>) -> (T, Self) {
        (
            prepared.hidden,
            Self {
                cosine: prepared.cosine,
                sine: prepared.sine,
                position_delta: prepared.position_delta,
                deepstack: prepared.deepstack,
            },
        )
    }

    /// Reconstructs decoder state on a downstream partition.
    pub fn into_prepared(self, hidden: T) -> PipelinePrepared<T> {
        PipelinePrepared {
            hidden,
            cosine: self.cosine,
            sine: self.sine,
            position_delta: self.position_delta,
            mask: None,
            deepstack: self.deepstack,
            visual_mask: None,
        }
    }
}

/// Decoder partition input from the input owner or an upstream pipeline rank.
pub enum PipelinePartitionInput<'a, T> {
    /// Text token identities entering the architecture-owned embedding boundary.
    Tokens {
        /// Token identities.
        tokens: &'a T,
        /// Existing decoder cache offset.
        offset: i32,
        /// Persisted multimodal position delta after a media prefill.
        position_delta: Option<&'a T>,
    },
    /// Evolving activation plus typed immutable decoder context.
    Hidden {
        /// Upstream decoder activation.
        hidden: T,
        /// Family-owned boundary context.
        boundary: PipelineBoundary<T>,
    },
}

/// One neutral composite model for dense and MoE Qwen3-VL.
pub struct LayeredModel<B: RoutedNeuralBackend> {
    args: ModelArgs,
    static_modules: StaticModules<B>,
    parallel_geometry: Option<std::sync::Arc<LocalGeometry>>,
}

impl<B: RoutedNeuralBackend> eredu_runtime::ArchitectureParameters<B> for LayeredModel<B> {
    type DefinitionError = Error;

    fn state_layout(&self) -> Result<StateLayout, Self::DefinitionError> {
        self.state_layout_impl()
    }

    fn parameter_description(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Self::DefinitionError> {
        self.parameter_description_impl(context)
    }

    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        Ok(super::static_recipes(source))
    }

    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitor<B>,
    {
        visitor.visit("vision", &self.static_modules.vision)?;
        visitor.visit("embedding", &self.static_modules.text.embeddings)?;
        visitor.visit("norm", &self.static_modules.text.norm)?;
        if let Some(head) = &self.static_modules.text.lm_head {
            visitor.visit("output", head)?;
        }
        Ok(())
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        visitor.visit_mut("vision", &mut self.static_modules.vision)?;
        visitor.visit_mut("embedding", &mut self.static_modules.text.embeddings)?;
        visitor.visit_mut("norm", &mut self.static_modules.text.norm)?;
        if let Some(head) = &mut self.static_modules.text.lm_head {
            visitor.visit_mut("output", head)?;
        }
        Ok(())
    }
}

impl<B: RoutedNeuralBackend> LayeredModel<B> {
    #[allow(clippy::too_many_arguments)]
    fn begin_distributed_partition<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, PipelineBoundary<B::Tensor>>,
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
        if state.layout() != expected {
            return Err(Error::backend("Qwen3-VL partition state layout mismatch"));
        }
        let offset = state
            .layer(first_state_ordinal)
            .map_err(Error::backend)?
            .position();
        let persisted_delta = if matches!(&input, LayeredPartitionInput::Tokens(_)) {
            state
                .layer(first_state_ordinal)
                .map_err(Error::backend)?
                .fixed_component(StateTensorRole::PositionDelta)
                .map_err(Error::backend)?
                .clone()
        } else {
            None
        };
        let (input, batch, sequence) = match input {
            LayeredPartitionInput::Tokens(tokens) => (
                PipelinePartitionInput::Tokens {
                    tokens,
                    offset,
                    position_delta: persisted_delta.as_ref(),
                },
                tokens.dim(0),
                tokens.dim(1),
            ),
            LayeredPartitionInput::Hidden {
                hidden,
                auxiliary: boundary,
            } => {
                let batch = hidden.dim(0);
                let sequence = hidden.dim(1);
                (
                    PipelinePartitionInput::Hidden { hidden, boundary },
                    batch,
                    sequence,
                )
            }
        };
        let (forward, position_delta) = self
            .begin_routed_text_partition(input, mask, batch, sequence, offset, parallel, context)?;
        if first_state_ordinal == 0 {
            *state
                .layer(first_state_ordinal)
                .map_err(Error::backend)?
                .fixed_component(StateTensorRole::PositionDelta)
                .map_err(Error::backend)? = Some(position_delta);
        }
        Ok(forward)
    }

    /// Prepares one routed decoder partition, including DeepStack defaults,
    /// shape validation, and architecture-owned causal masking.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_routed_text_partition(
        &mut self,
        input: PipelinePartitionInput<'_, B::Tensor>,
        explicit_mask: Option<&B::Tensor>,
        batch: i32,
        sequence: i32,
        offset: i32,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            B::Tensor,
        ),
        Error,
    > {
        let mut prepared = self.begin_partition_text_inner(input, parallel, context)?;
        let deepstack_count = self.args.vision.deepstack_layer_count();
        if prepared.deepstack.is_empty() {
            let zero = prepared.hidden.zeros_like(context)?;
            prepared.deepstack = vec![zero; deepstack_count];
        } else if prepared.deepstack.len() != deepstack_count {
            return Err(Error::backend(format!(
                "Qwen3-VL prepared {} DeepStack tensors, expected {deepstack_count}",
                prepared.deepstack.len()
            )));
        }
        let expected = [batch, sequence, self.args.text.hidden_size];
        if prepared.hidden.shape() != expected {
            return Err(Error::backend(format!(
                "Qwen3-VL decoder input is shaped {:?}, expected {expected:?}",
                prepared.hidden.shape()
            )));
        }
        prepared.mask = match explicit_mask {
            Some(mask) => Some(mask.clone()),
            None if sequence > 1 => Some(B::causal_mask(sequence, offset, None, context)?),
            None => prepared.mask,
        };
        Ok(prepared.into_layered_forward())
    }

    fn begin_partition_text_inner(
        &mut self,
        input: PipelinePartitionInput<'_, B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PipelinePrepared<B::Tensor>, Error> {
        match input {
            PipelinePartitionInput::Tokens {
                tokens,
                offset,
                position_delta,
            } => {
                let parts = [InputPart::Text(tokens)];
                let input = ModelInput {
                    parts: &parts,
                    pixels: None,
                    mask: None,
                };
                let state = match parallel {
                    Some(parallel) => self.begin_pipeline_parallel(
                        input,
                        offset,
                        position_delta,
                        parallel,
                        context,
                    ),
                    None => self.begin_pipeline(input, offset, position_delta, context),
                }?;
                self.finish_pipeline(state, parallel, context)
            }
            PipelinePartitionInput::Hidden { hidden, boundary } => {
                Ok(boundary.into_prepared(hidden))
            }
        }
    }

    /// Finishes the serial text output boundary.
    pub fn finish_partition_text(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_pipeline_logits(hidden, context)
    }

    /// Finishes the tensor-parallel text output boundary.
    pub fn finish_partition_text_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish_parallel(hidden, parallel, context)
    }

    /// Builds unloaded modules with their canonical checkpoint identities.
    pub fn new(args: ModelArgs, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Qwen3-VL",
            crate::operator_requirements::QWEN_VL,
        )?;
        args.vision
            .validate_for(VisionMode::DeepStack)
            .map_err(Error::backend)?;
        let text = qwen::StaticModules::new(&args.text, context)?;
        let vision = VisionStatic::new_with_root(args.vision.clone(), "model.visual", context)?;
        Ok(Self {
            args,
            static_modules: StaticModules { text, vision },
            parallel_geometry: None,
        })
    }

    /// Builds the composite graph with planner-derived text and vision modules.
    pub fn new_parallel(
        args: ModelArgs,
        geometry: LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Qwen3-VL",
            crate::operator_requirements::QWEN_VL,
        )?;
        args.vision
            .validate_for(VisionMode::DeepStack)
            .map_err(Error::backend)?;
        geometry.validate_for(&args).map_err(Error::backend)?;
        let text = qwen::StaticModules::new_parallel(&args.text, geometry.text(), context)?;
        let vision = VisionStatic::new_parallel_with_root(
            args.vision.clone(),
            "model.visual",
            geometry.merger_widths(),
            context,
        )?;
        Ok(Self {
            args,
            static_modules: StaticModules { text, vision },
            parallel_geometry: Some(std::sync::Arc::new(geometry)),
        })
    }

    /// Returns normalized nested text and vision policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Describes shared-vision and text-decoder parameters with canonical
    /// graph-unit ownership.
    fn parameter_description_impl(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("vision"),
                ExecutionGroupSpec::with_dependencies("text_decoder", ["vision"]),
            ],
            "text_decoder",
        )
        .map_err(Error::backend)?;
        let counts = [
            self.args.vision.layer_count(),
            usize::try_from(self.args.text.num_hidden_layers).map_err(Error::backend)?,
        ];
        let layout = ExecutionUnitLayout::new(&graph, counts).map_err(Error::backend)?;
        let text_static = static_parallel_parameter_groups::<B>(
            &self.static_modules.text.embeddings,
            &self.static_modules.text.norm,
            self.static_modules.text.lm_head.as_ref(),
            &self.args.text.parameter_root,
        )
        .map_err(Error::backend)?;
        let vision_static = vision_parameter_groups::<B>(
            &self.static_modules.vision,
            &self.args.vision,
            "model.visual",
        )
        .map_err(Error::backend)?;
        let mut expected = text_static
            .iter()
            .chain(&vision_static)
            .cloned()
            .collect::<Vec<_>>();
        let mut owned = text_static
            .into_iter()
            .enumerate()
            .map(|(index, group)| {
                OwnedParameterGroupSpec::new(
                    if index == 0 && self.args.text.tie_word_embeddings {
                        ParameterGroupOwner::static_any_of(["embedding", "output"])
                    } else {
                        ParameterGroupOwner::static_role(match index {
                            0 => "embedding",
                            1 => "norm",
                            _ => "output",
                        })
                    },
                    group,
                )
            })
            .chain(vision_static.into_iter().map(|group| {
                OwnedParameterGroupSpec::new(ParameterGroupOwner::static_role("vision"), group)
            }))
            .collect::<Vec<_>>();
        for (group_index, &count) in counts.iter().enumerate() {
            let group_id = layout
                .group_id(group_index)
                .expect("Qwen3-VL layout group")
                .clone();
            for index in 0..count {
                let unit = self.construct_unit(group_index, index, context)?;
                let groups = match unit {
                    Unit::Vision(block) => block_parallel_parameter_groups(
                        &block,
                        &self.args.vision,
                        "model.visual",
                        index,
                    ),
                    Unit::Text(block) => {
                        qwen::routed_layer_parallel_parameter_groups(&block, &self.args.text, index)
                    }
                }
                .map_err(Error::backend)?;
                expected.extend(groups.iter().cloned());
                owned.extend(groups.into_iter().map(|group| {
                    OwnedParameterGroupSpec::new(
                        ParameterGroupOwner::execution_unit(group_id.clone(), index),
                        group,
                    )
                }));
            }
        }
        ArchitectureParameterDescription::new(&graph, &layout, expected, owned)
            .map_err(Error::backend)
    }

    /// Returns replicated or planner-derived decoder state geometry.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        self.parallel_geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .map_or_else(
                || super::state_layout(&self.args).map_err(Error::backend),
                Ok,
            )
    }

    /// Applies the architecture-owned tensor-parallel target output boundary.
    pub fn pipeline_finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.text.norm.forward(hidden, context)?;
        match &mut self.static_modules.text.lm_head {
            Some(head) => B::vocabulary_parallel_project(head, &hidden, parallel, context),
            None => B::vocabulary_parallel_embedding_project(
                &mut self.static_modules.text.embeddings,
                &hidden,
                parallel,
                context,
            ),
        }
    }

    /// Shares planner-owned geometry with placed unit factories.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    /// Constructs one canonical vision or text unit using this model's
    /// replicated or planner-derived local geometry.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        let count = match group {
            0 => self.args.vision.layer_count(),
            1 => usize::try_from(self.args.text.num_hidden_layers).map_err(Error::backend)?,
            _ => return Err(Error::backend("Qwen3-VL has two execution groups")),
        };
        if index >= count {
            return Err(Error::backend("Qwen3-VL unit is outside its group"));
        }
        match group {
            0 => {
                let geometry = self
                    .parallel_geometry
                    .as_ref()
                    .and_then(|geometry| geometry.vision_block(index));
                Ok(Unit::Vision(match geometry {
                    Some((heads, intermediate)) => VisionBlock::new_parallel_with_root(
                        &self.args.vision,
                        "model.visual",
                        index,
                        heads,
                        intermediate,
                        context,
                    )?,
                    None => VisionBlock::new_with_root(
                        &self.args.vision,
                        "model.visual",
                        index,
                        context,
                    )?,
                }))
            }
            1 => {
                let args = self
                    .parallel_geometry
                    .as_ref()
                    .and_then(|geometry| geometry.text().block(index))
                    .unwrap_or(&self.args.text);
                Ok(Unit::Text(qwen::new_routed_block(args, index, context)?))
            }
            _ => unreachable!(),
        }
    }

    fn prepare_parts(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Vec<PreparedPart<B::Tensor>>, Vec<(i32, i32, i32)>), Error> {
        if parts.is_empty() {
            return Err(Error::backend("Qwen3-VL input has no ordered parts"));
        }
        let mut grids = Vec::new();
        let prepared = parts
            .iter()
            .map(|part| match part {
                InputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: self
                        .static_modules
                        .text
                        .embeddings
                        .forward(tokens, context)?,
                }),
                InputPart::Image { tokens, grid } | InputPart::Video { tokens, grid } => {
                    grids.extend_from_slice(grid);
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
                    })
                }
                InputPart::Projected { tokens, embeddings } => {
                    if embeddings.shape()
                        != [tokens.dim(0), tokens.dim(1), self.args.text.hidden_size]
                    {
                        return Err(Error::backend("Qwen3-VL projected input geometry mismatch"));
                    }
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: (*embeddings).clone(),
                    })
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok((prepared, grids))
    }

    fn prepare_parts_parallel(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Vec<PreparedPart<B::Tensor>>, Vec<(i32, i32, i32)>), Error> {
        if parts.is_empty() {
            return Err(Error::backend("Qwen3-VL input has no ordered parts"));
        }
        let mut grids = Vec::new();
        let prepared = parts
            .iter()
            .map(|part| match part {
                InputPart::Text(tokens) => Ok(PreparedPart::Text {
                    tokens: (*tokens).clone(),
                    embeddings: B::vocabulary_parallel_lookup(
                        &mut self.static_modules.text.embeddings,
                        tokens,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?,
                }),
                InputPart::Image { tokens, grid } | InputPart::Video { tokens, grid } => {
                    grids.extend_from_slice(grid);
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
                    })
                }
                InputPart::Projected { tokens, embeddings } => {
                    if embeddings.shape()
                        != [tokens.dim(0), tokens.dim(1), self.args.text.hidden_size]
                    {
                        return Err(Error::backend("Qwen3-VL projected input geometry mismatch"));
                    }
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: (*embeddings).clone(),
                    })
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok((prepared, grids))
    }

    fn assemble(
        &self,
        parts: &[PreparedPart<B::Tensor>],
        vision: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_nn::multimodal::OrderedModelInput<B::Tensor>, Error> {
        let media_tokens = parts
            .iter()
            .map(|part| match part {
                PreparedPart::Media { tokens } => tokens.dim(1),
                _ => 0,
            })
            .sum::<i32>();
        match vision {
            Some(value) if value.shape() == [1, media_tokens, self.args.text.hidden_size] => {}
            None if media_tokens == 0 => {}
            Some(value) => {
                return Err(Error::backend(format!(
                    "Qwen3-VL vision output {:?} does not match {media_tokens} placeholders",
                    value.shape()
                )))
            }
            None => {
                return Err(Error::backend(
                    "Qwen3-VL media placeholders require vision output",
                ))
            }
        }
        let mut offset = 0;
        let mut embeddings = Vec::with_capacity(parts.len());
        for part in parts {
            match part {
                PreparedPart::Text {
                    embeddings: value, ..
                } => embeddings.push(value.clone()),
                PreparedPart::Media { tokens } => {
                    let length = tokens.dim(1);
                    embeddings.push(vision.expect("validated vision output").index(
                        &[
                            Index::Full,
                            Index::Range(offset, offset + length),
                            Index::Full,
                        ],
                        context,
                    )?);
                    offset += length;
                }
            }
        }
        let ordered = parts
            .iter()
            .zip(&embeddings)
            .map(|(part, embeddings)| OrderedInputPart {
                token_ids: match part {
                    PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens,
                },
                embeddings,
            })
            .collect::<Vec<_>>();
        assemble_ordered_inputs(&ordered, self.args.text.hidden_size, context)
    }

    fn finish_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.text.norm.forward(hidden, context)?;
        match &mut self.static_modules.text.lm_head {
            Some(head) => head.forward(&hidden, context),
            None => self
                .static_modules
                .text
                .embeddings
                .as_linear(&hidden, context),
        }
    }

    /// Prepares a pipeline request without binding it to a concrete cache
    /// container. The caller supplies the current token offset and persisted
    /// multimodal delta from the generic state slots.
    pub fn begin_pipeline<'a>(
        &mut self,
        input: ModelInput<'a, B::Tensor>,
        offset: i32,
        persisted_delta: Option<&B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PipelineVisionState<B::Tensor>, Error> {
        self.begin_pipeline_inner(input, offset, persisted_delta, None, context)
    }

    /// Prepares the same placed request with rank-local vocabulary lookup.
    pub fn begin_pipeline_parallel<'a>(
        &mut self,
        input: ModelInput<'a, B::Tensor>,
        offset: i32,
        persisted_delta: Option<&B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PipelineVisionState<B::Tensor>, Error> {
        self.begin_pipeline_inner(input, offset, persisted_delta, Some(parallel), context)
    }

    fn begin_pipeline_inner<'a>(
        &mut self,
        input: ModelInput<'a, B::Tensor>,
        offset: i32,
        persisted_delta: Option<&B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PipelineVisionState<B::Tensor>, Error> {
        let (parts, grids) = match parallel {
            Some(parallel) => self.prepare_parts_parallel(input.parts, parallel, context)?,
            None => self.prepare_parts(input.parts, context)?,
        };
        let media = !grids.is_empty();
        if media != input.pixels.is_some() {
            return Err(Error::backend(
                "Qwen3-VL pixels and media metadata must appear together",
            ));
        }
        if media && offset != 0 {
            return Err(Error::backend(
                "Qwen3-VL media input cannot append to a populated cache",
            ));
        }
        let (vision_initial, vision) = match input.pixels {
            Some(pixels) => {
                let (hidden, state) = self.static_modules.vision.begin(
                    VisionInput {
                        pixels,
                        grid: &grids,
                    },
                    context,
                )?;
                (Some(hidden), Some(state))
            }
            None => (None, None),
        };
        let sequence = parts
            .iter()
            .map(|part| match part {
                PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens.dim(1),
            })
            .sum::<i32>();
        let position_parts = input
            .parts
            .iter()
            .map(|part| match part {
                InputPart::Text(tokens) | InputPart::Projected { tokens, .. } => {
                    PositionPart::Text(tokens.dim(1))
                }
                InputPart::Image { grid, .. } | InputPart::Video { grid, .. } => {
                    PositionPart::Media(grid)
                }
            })
            .collect::<Vec<_>>();
        let (mut positions, computed_delta) = multimodal_position_ids(
            &position_parts,
            self.args.vision.spatial_merge_size,
            sequence,
        )
        .map_err(Error::backend)?;
        if !media {
            for axis in &mut positions {
                for position in axis {
                    *position += offset;
                }
            }
        }
        let mut positions = position_ids_tensor::<B::Tensor>(&positions, context)?;
        let delta = match (media, persisted_delta) {
            (false, Some(delta)) => delta.clone(),
            _ => B::Tensor::full_i32(computed_delta, &[1], context)?,
        };
        if !media {
            positions = positions.add(&delta, context)?;
        }
        let rotary = mrope_embeddings(
            &positions,
            self.args.text.head_dim,
            self.args.text.rope_theta,
            &self.args.mrope_section,
            context,
        )?;
        let mask = if let Some(mask) = input.mask {
            Some(mask.clone())
        } else if sequence > 1 {
            Some(B::causal_mask(sequence, offset, None, context)?)
        } else {
            None
        };
        let assembled = self.assemble(&parts, None, context);
        let hidden = match vision_initial {
            Some(hidden) => hidden,
            None => assembled?.embeddings,
        };
        Ok(PipelineVisionState {
            hidden,
            parts,
            rotary,
            delta,
            mask,
            vision,
            vision_output: None,
        })
    }

    /// Whether this request owns visual component work.
    pub fn pipeline_vision_active(state: &PipelineVisionState<B::Tensor>) -> bool {
        state.vision.is_some()
    }

    /// Exports all request tensors needed by a downstream vision owner.
    pub fn pipeline_retained_values(state: &PipelineVisionState<B::Tensor>) -> Vec<B::Tensor> {
        let mut values = vec![
            state.hidden.clone(),
            state.rotary.0.clone(),
            state.rotary.1.clone(),
            state.delta.clone(),
        ];
        values.extend(state.mask.iter().cloned());
        for part in &state.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => {
                    values.extend([tokens.clone(), embeddings.clone()]);
                }
                PreparedPart::Media { tokens } => values.push(tokens.clone()),
            }
        }
        if let Some(vision) = &state.vision {
            values.extend(vision.retained_values().cloned());
        }
        values
    }

    /// Replaces request tensors with values transported from the previous
    /// component owner. Structure is validated against the locally rebuilt
    /// parameter-free request state.
    pub fn replace_pipeline_retained_values(
        state: &mut PipelineVisionState<B::Tensor>,
        values: Vec<B::Tensor>,
    ) -> Result<(), Error> {
        let fixed = 4
            + usize::from(state.mask.is_some())
            + state
                .parts
                .iter()
                .map(|part| match part {
                    PreparedPart::Text { .. } => 2,
                    PreparedPart::Media { .. } => 1,
                })
                .sum::<usize>();
        let minimum = fixed + usize::from(state.vision.is_some()) * 2;
        if values.len() < minimum || (state.vision.is_none() && values.len() != fixed) {
            return Err(Error::backend(format!(
                "Qwen3-VL pipeline continuation received {} tensors, expected at least {minimum}",
                values.len(),
            )));
        }
        let mut values = values.into_iter();
        state.hidden = values.next().expect("validated hidden");
        state.rotary.0 = values.next().expect("validated cosine");
        state.rotary.1 = values.next().expect("validated sine");
        state.delta = values.next().expect("validated delta");
        if state.mask.is_some() {
            state.mask = Some(values.next().expect("validated mask"));
        }
        for part in &mut state.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => {
                    *tokens = values.next().expect("validated tokens");
                    *embeddings = values.next().expect("validated embeddings");
                }
                PreparedPart::Media { tokens } => {
                    *tokens = values.next().expect("validated media tokens");
                }
            }
        }
        if let Some(vision) = &mut state.vision {
            vision.replace_retained_values(values.collect())?;
        }
        Ok(())
    }

    /// Executes one placed shared-vision block.
    pub fn forward_pipeline_vision(
        &mut self,
        index: usize,
        block: &mut VisionBlock<B>,
        state: &mut PipelineVisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        let vision = state
            .vision
            .as_mut()
            .ok_or_else(|| Error::backend("missing Qwen3-VL pipeline vision state"))?;
        state.hidden = match parallel {
            Some(parallel) => self.static_modules.vision.forward_block_parallel(
                block,
                index,
                &state.hidden,
                vision,
                parallel,
                context,
            )?,
            None => self.static_modules.vision.forward_block(
                block,
                index,
                &state.hidden,
                vision,
                context,
            )?,
        };
        Ok(())
    }

    /// Finishes the vision merger and assembles decoder-width input.
    pub fn finish_pipeline(
        &mut self,
        mut state: PipelineVisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PipelinePrepared<B::Tensor>, Error> {
        let deepstack = if let Some(vision) = &mut state.vision {
            let output = match parallel {
                Some(parallel) => self.static_modules.vision.finish_parallel(
                    &state.hidden,
                    vision,
                    parallel,
                    context,
                )?,
                None => self
                    .static_modules
                    .vision
                    .finish(&state.hidden, vision, context)?,
            };
            state.vision_output = Some(output.embeddings);
            output.deepstack_features
        } else {
            Vec::new()
        };
        let assembled = self.assemble(&state.parts, state.vision_output.as_ref(), context)?;
        let visual_mask = if deepstack.is_empty() {
            None
        } else {
            Some(
                assembled
                    .token_ids
                    .equal_i32(self.args.image_token_id, context)?
                    .logical_or(
                        &assembled
                            .token_ids
                            .equal_i32(self.args.video_token_id, context)?,
                        context,
                    )?,
            )
        };
        let deepstack = match visual_mask.as_ref() {
            Some(mask) => deepstack
                .into_iter()
                .map(|features| {
                    assembled.embeddings.zeros_like(context)?.masked_scatter(
                        mask,
                        &features.index(&[Index::At(0), Index::Full, Index::Full], context)?,
                        context,
                    )
                })
                .collect::<Result<Vec<_>, Error>>()?,
            None => deepstack,
        };
        Ok(PipelinePrepared {
            hidden: assembled.embeddings,
            cosine: state.rotary.0,
            sine: state.rotary.1,
            position_delta: state.delta,
            mask: state.mask,
            deepstack,
            visual_mask: None,
        })
    }

    /// Applies the shared final norm and vocabulary projection.
    pub fn finish_pipeline_logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_logits(hidden, context)
    }

    /// Executes one unit while routing ordinary-Qwen MoE banks through a
    /// runtime-owned provider. Vision units retain the shared vision path.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match (group, unit) {
            (0, Unit::Vision(block)) => self.static_modules.vision.forward_block(
                block,
                index,
                hidden,
                forward
                    .vision_state
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing Qwen3-VL vision state"))?,
                context,
            ),
            (1, Unit::Text(block)) => {
                let pass = if hidden.dim(1) > 1 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                let mask = forward.mask.as_ref();
                let cosine = &forward.rotary.0;
                let sine = &forward.rotary.1;
                let mut output = block.forward_with_feed_forward(
                    AttentionInput {
                        hidden,
                        mask,
                        cache: Some(state.layer(index).map_err(Error::backend)?),
                        allow_sliding_prefill: true,
                        rotary_position: Some(RotaryPosition::Embeddings { cosine, sine }),
                    },
                    context,
                    |policy, normalized, context| {
                        let shape = normalized.shape().to_vec();
                        let flat = normalized.reshape(
                            &[-1, normalized.dim(normalized.shape().len() - 1)],
                            context,
                        )?;
                        policy
                            .forward_with_provider(index, pass, &flat, context, provider)?
                            .reshape(&shape, context)
                    },
                )?;
                if let Some(features) = forward.deepstack.get(index) {
                    output = if features.shape() == output.shape() {
                        output.add(features, context)?
                    } else {
                        let source =
                            features.index(&[Index::At(0), Index::Full, Index::Full], context)?;
                        output.add(
                            &output.zeros_like(context)?.masked_scatter(
                                forward.visual_mask.as_ref().ok_or_else(|| {
                                    Error::backend("missing Qwen3-VL visual mask")
                                })?,
                                &source,
                                context,
                            )?,
                            context,
                        )?
                    };
                }
                Ok(output)
            }
            _ => Err(Error::backend("Qwen3-VL unit/group mismatch")),
        }
    }

    /// Executes one local unit through the runtime-owned routed expert provider.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider_parallel<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match (group, unit) {
            (0, Unit::Vision(block)) => self.static_modules.vision.forward_block_parallel(
                block,
                index,
                hidden,
                forward
                    .vision_state
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing Qwen3-VL vision state"))?,
                parallel,
                context,
            ),
            (1, Unit::Text(block)) => {
                let pass = if hidden.dim(1) > 1 {
                    ExpertPass::Prefill
                } else {
                    ExpertPass::Decode
                };
                let mask = forward.mask.as_ref();
                let cosine = &forward.rotary.0;
                let sine = &forward.rotary.1;
                let mut output = block.forward_tensor_parallel_with_feed_forward(
                    AttentionInput {
                        hidden,
                        mask,
                        cache: Some(state.layer(index).map_err(Error::backend)?),
                        allow_sliding_prefill: true,
                        rotary_position: Some(RotaryPosition::Embeddings { cosine, sine }),
                    },
                    parallel,
                    context,
                    |policy, normalized, context| {
                        let shape = normalized.shape().to_vec();
                        let flat = normalized.reshape(
                            &[-1, normalized.dim(normalized.shape().len() - 1)],
                            context,
                        )?;
                        policy
                            .forward_with_provider_parallel(
                                index, pass, &flat, parallel, context, provider,
                            )?
                            .reshape(&shape, context)
                    },
                )?;
                if let Some(features) = forward.deepstack.get(index) {
                    output = if features.shape() == output.shape() {
                        output.add(features, context)?
                    } else {
                        let source =
                            features.index(&[Index::At(0), Index::Full, Index::Full], context)?;
                        output.add(
                            &output.zeros_like(context)?.masked_scatter(
                                forward.visual_mask.as_ref().ok_or_else(|| {
                                    Error::backend("missing Qwen3-VL visual mask")
                                })?,
                                &source,
                                context,
                            )?,
                            context,
                        )?
                    };
                }
                Ok(output)
            }
            _ => Err(Error::backend("Qwen3-VL parallel unit/group mismatch")),
        }
    }
}

impl<B, S> LayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
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
        if group == 0 {
            eredu_runtime::ArchitectureGroupTransport {
                placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
                kind: eredu_runtime::ArchitectureGroupKind::VisionEncoder,
                first_owner_static_roles: vec!["vision".into()],
                last_owner_static_roles: Vec::new(),
                merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
                parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
                request_optional: true,
            }
        } else {
            crate::transport::decoder()
        }
    }

    fn model_identity(&self) -> &str {
        &self.args.model_type
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error> {
        ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("vision"),
                ExecutionGroupSpec::with_dependencies("text_decoder", ["vision"]),
            ],
            "text_decoder",
        )
        .map_err(Error::backend)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        match group {
            0 => Ok(self.args.vision.layer_count()),
            1 => usize::try_from(self.args.text.num_hidden_layers).map_err(Error::backend),
            _ => Err(Error::backend("Qwen3-VL has two execution groups")),
        }
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        let count = match group {
            0 => self.args.vision.layer_count(),
            1 => usize::try_from(self.args.text.num_hidden_layers).map_err(Error::backend)?,
            _ => return Err(Error::backend("Qwen3-VL has two execution groups")),
        };
        if index >= count {
            return Err(Error::backend("Qwen3-VL unit is outside its group"));
        }
        match group {
            0 => Ok(format!("model.visual.blocks.{index}")),
            1 => Ok(format!("{}.layers.{index}", self.args.text.parameter_root)),
            _ => unreachable!(),
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
        self.construct_unit(group, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = super::state_layout(&self.args).map_err(Error::backend)?;
        if state.layout() != &expected {
            return Err(Error::backend("Qwen3-VL runtime state layout mismatch"));
        }
        let (parts, grids) = self.prepare_parts(input.parts, context)?;
        let media = !grids.is_empty();
        if media != input.pixels.is_some() {
            return Err(Error::backend(
                "Qwen3-VL pixels and media metadata must appear together",
            ));
        }
        let (vision_initial, vision_state) = match input.pixels {
            Some(pixels) => {
                let (hidden, state) = self.static_modules.vision.begin(
                    VisionInput {
                        pixels,
                        grid: &grids,
                    },
                    context,
                )?;
                (Some(hidden), Some(state))
            }
            None => (None, None),
        };
        let assembled = self.assemble(&parts, None, context);
        let sequence = parts
            .iter()
            .map(|part| match part {
                PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens.dim(1),
            })
            .sum::<i32>();
        let state_layer = state.layer(0).map_err(Error::backend)?;
        let offset = state_layer.position();
        let mut position_parts = Vec::with_capacity(input.parts.len());
        for part in input.parts {
            match part {
                InputPart::Text(tokens) | InputPart::Projected { tokens, .. } => {
                    position_parts.push(PositionPart::Text(tokens.dim(1)))
                }
                InputPart::Image { grid, .. } | InputPart::Video { grid, .. } => {
                    position_parts.push(PositionPart::Media(grid))
                }
            }
        }
        let (mut positions, computed_delta) = multimodal_position_ids(
            &position_parts,
            self.args.vision.spatial_merge_size,
            sequence,
        )
        .map_err(Error::backend)?;
        if media && offset != 0 {
            return Err(Error::backend(
                "Qwen3-VL media input cannot append to a populated cache",
            ));
        }
        if !media {
            for axis in &mut positions {
                for position in axis {
                    *position += offset;
                }
            }
        }
        let mut positions = position_ids_tensor::<B::Tensor>(&positions, context)?;
        let delta = state_layer
            .fixed_component(StateTensorRole::PositionDelta)
            .map_err(Error::backend)?;
        if media || delta.is_none() {
            *delta = Some(B::Tensor::full_i32(computed_delta, &[1], context)?);
        } else if let Some(delta) = delta.as_ref() {
            positions = positions.add(delta, context)?;
        }
        let position_delta = delta
            .as_ref()
            .expect("Qwen3-VL installs position delta")
            .clone();
        let rotary = mrope_embeddings(
            &positions,
            self.args.text.head_dim,
            self.args.text.rope_theta,
            &self.args.mrope_section,
            context,
        )?;
        let mask = if let Some(mask) = input.mask {
            Some(mask.clone())
        } else if sequence > 1 {
            Some(B::causal_mask(sequence, offset, None, context)?)
        } else {
            None
        };
        let (assembled_tokens, assembled_hidden, assembled_error) = match assembled {
            Ok(value) => (Some(value.token_ids), Some(value.embeddings), None),
            Err(error) => (None, None, Some(error)),
        };
        let hidden = vision_initial
            .as_ref()
            .cloned()
            .or(assembled_hidden)
            .ok_or_else(|| {
                assembled_error.unwrap_or_else(|| Error::backend("empty Qwen3-VL input"))
            })?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                tokens: assembled_tokens,
                parts,
                rotary,
                position_delta,
                vision_state,
                vision_initial,
                vision_output: None,
                deepstack: Vec::new(),
                visual_mask: None,
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
            1 => {
                let assembled =
                    self.assemble(&forward.parts, forward.vision_output.as_ref(), context)?;
                forward.visual_mask = if forward.deepstack.is_empty() {
                    None
                } else {
                    Some(
                        assembled
                            .token_ids
                            .equal_i32(self.args.image_token_id, context)?
                            .logical_or(
                                &assembled
                                    .token_ids
                                    .equal_i32(self.args.video_token_id, context)?,
                                context,
                            )?,
                    )
                };
                forward.tokens = Some(assembled.token_ids);
                Ok(assembled.embeddings)
            }
            _ => Err(Error::backend("invalid Qwen3-VL execution group")),
        }
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        group == 1 || (group == 0 && forward.vision_state.is_some())
    }

    fn state_ordinal(&self, group: usize, index: usize, _ordinal: usize) -> usize {
        match group {
            0 => 0,
            1 => index,
            _ => index,
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
            (0, Unit::Vision(block)) => self.static_modules.vision.forward_block(
                block,
                index,
                hidden,
                forward
                    .vision_state
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing Qwen3-VL vision state"))?,
                context,
            ),
            (1, Unit::Text(block)) => {
                let mut output = block.forward(
                    AttentionInput {
                        hidden,
                        mask: forward.mask.as_ref(),
                        cache: Some(state.layer(index).map_err(|error| {
                            Error::backend(format!(
                                "Qwen3-VL text group {group} unit {index} cache: {error}"
                            ))
                        })?),
                        allow_sliding_prefill: true,
                        rotary_position: Some(RotaryPosition::Embeddings {
                            cosine: &forward.rotary.0,
                            sine: &forward.rotary.1,
                        }),
                    },
                    context,
                )?;
                if let Some(features) = forward.deepstack.get(index) {
                    let source =
                        features.index(&[Index::At(0), Index::Full, Index::Full], context)?;
                    output = output.add(
                        &output.zeros_like(context)?.masked_scatter(
                            forward
                                .visual_mask
                                .as_ref()
                                .ok_or_else(|| Error::backend("missing Qwen3-VL visual mask"))?,
                            &source,
                            context,
                        )?,
                        context,
                    )?;
                }
                Ok(output)
            }
            _ => Err(Error::backend("Qwen3-VL unit/group mismatch")),
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
        if group == 0 {
            if let Some(vision_state) = forward.vision_state.as_mut() {
                let output = self
                    .static_modules
                    .vision
                    .finish(hidden, vision_state, context)?;
                forward.deepstack = output.deepstack_features;
                forward.vision_output = Some(output.embeddings);
                return Ok(forward
                    .vision_output
                    .as_ref()
                    .expect("installed vision output")
                    .clone());
            }
        }
        Ok(hidden.clone())
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_logits(hidden, context)
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = Vec::new();
        values.extend(forward.mask.iter());
        values.extend(forward.tokens.iter());
        values.extend([&forward.rotary.0, &forward.rotary.1]);
        values.push(&forward.position_delta);
        values.extend(forward.vision_initial.iter());
        values.extend(forward.vision_output.iter());
        values.extend(forward.deepstack.iter());
        values.extend(forward.visual_mask.iter());
        if let Some(state) = &forward.vision_state {
            values.extend(state.retained_values());
        }
        for part in &forward.parts {
            match part {
                PreparedPart::Text { tokens, embeddings } => values.extend([tokens, embeddings]),
                PreparedPart::Media { tokens } => values.push(tokens),
            }
        }
        values.into_iter()
    }
}

impl<B, S> ParallelLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
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
            .ok_or_else(|| Error::backend("Qwen3-VL model has no local geometry"))?
            .state_layout();
        if state.layout() != expected {
            return Err(Error::backend("Qwen3-VL rank-local state layout mismatch"));
        }
        let (parts, grids) = self.prepare_parts_parallel(input.parts, parallel, context)?;
        let media = !grids.is_empty();
        if media != input.pixels.is_some() {
            return Err(Error::backend(
                "Qwen3-VL pixels and media metadata must appear together",
            ));
        }
        let (vision_initial, vision_state) = match input.pixels {
            Some(pixels) => {
                let (hidden, state) = self.static_modules.vision.begin(
                    VisionInput {
                        pixels,
                        grid: &grids,
                    },
                    context,
                )?;
                (Some(hidden), Some(state))
            }
            None => (None, None),
        };
        let assembled = self.assemble(&parts, None, context);
        let sequence = parts
            .iter()
            .map(|part| match part {
                PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens.dim(1),
            })
            .sum::<i32>();
        let state_layer = state.layer(0).map_err(Error::backend)?;
        let offset = state_layer.position();
        let mut position_parts = Vec::with_capacity(input.parts.len());
        for part in input.parts {
            match part {
                InputPart::Text(tokens) | InputPart::Projected { tokens, .. } => {
                    position_parts.push(PositionPart::Text(tokens.dim(1)))
                }
                InputPart::Image { grid, .. } | InputPart::Video { grid, .. } => {
                    position_parts.push(PositionPart::Media(grid))
                }
            }
        }
        let (mut positions, computed_delta) = multimodal_position_ids(
            &position_parts,
            self.args.vision.spatial_merge_size,
            sequence,
        )
        .map_err(Error::backend)?;
        if media && offset != 0 {
            return Err(Error::backend(
                "Qwen3-VL media input cannot append to a populated cache",
            ));
        }
        if !media {
            for axis in &mut positions {
                for position in axis {
                    *position += offset;
                }
            }
        }
        let mut positions = position_ids_tensor::<B::Tensor>(&positions, context)?;
        let delta = state_layer
            .fixed_component(StateTensorRole::PositionDelta)
            .map_err(Error::backend)?;
        if media || delta.is_none() {
            *delta = Some(B::Tensor::full_i32(computed_delta, &[1], context)?);
        } else if let Some(delta) = delta.as_ref() {
            positions = positions.add(delta, context)?;
        }
        let position_delta = delta
            .as_ref()
            .expect("Qwen3-VL installs position delta")
            .clone();
        let rotary = mrope_embeddings(
            &positions,
            self.args.text.head_dim,
            self.args.text.rope_theta,
            &self.args.mrope_section,
            context,
        )?;
        let mask = if let Some(mask) = input.mask {
            Some(mask.clone())
        } else if sequence > 1 {
            Some(B::causal_mask(sequence, offset, None, context)?)
        } else {
            None
        };
        let (assembled_tokens, assembled_hidden, assembled_error) = match assembled {
            Ok(value) => (Some(value.token_ids), Some(value.embeddings), None),
            Err(error) => (None, None, Some(error)),
        };
        let hidden = vision_initial
            .as_ref()
            .cloned()
            .or(assembled_hidden)
            .ok_or_else(|| {
                assembled_error.unwrap_or_else(|| Error::backend("empty Qwen3-VL input"))
            })?;
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                tokens: assembled_tokens,
                parts,
                rotary,
                position_delta,
                vision_state,
                vision_initial,
                vision_output: None,
                deepstack: Vec::new(),
                visual_mask: None,
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
        self.forward_unit_with_provider_parallel(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            &mut eredu_runtime::ResidentExpertProvider,
            parallel,
            context,
        )
    }

    fn complete_execution_group_parallel(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 0 {
            if let Some(vision_state) = forward.vision_state.as_mut() {
                let output = self.static_modules.vision.finish_parallel(
                    hidden,
                    vision_state,
                    parallel,
                    context,
                )?;
                forward.deepstack = output.deepstack_features;
                forward.vision_output = Some(output.embeddings);
                return Ok(forward.vision_output.as_ref().unwrap().clone());
            }
        }
        Ok(hidden.clone())
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
            return Err(Error::backend("Qwen3-VL model has no local geometry"));
        }
        let hidden = self.static_modules.text.norm.forward(hidden, context)?;
        match &mut self.static_modules.text.lm_head {
            Some(head) => B::vocabulary_parallel_project(head, &hidden, parallel, context),
            None => B::vocabulary_parallel_embedding_project(
                &mut self.static_modules.text.embeddings,
                &hidden,
                parallel,
                context,
            ),
        }
    }
}

impl<B, S> PartitionedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type Boundary = PipelineBoundarySchema;

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(PipelineBoundarySchema::from_args(self.args()))
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, PipelineBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_distributed_partition(
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            None,
            context,
        )
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, PipelineBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        self.begin_distributed_partition(
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            Some(parallel),
            context,
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
    ) -> Result<LayeredPartitionOutput<B::Tensor, PipelineBoundary<B::Tensor>>, Self::Error> {
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
                auxiliary: PipelineBoundary {
                    cosine: forward.rotary.0.clone(),
                    sine: forward.rotary.1.clone(),
                    position_delta: forward.position_delta.clone(),
                    deepstack: forward.deepstack.clone(),
                },
            })
        }
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use eredu_runtime::{ArchitectureBoundary, BoundaryTensorDtype};

    #[test]
    fn mrope_and_deepstack_wire_geometry_is_family_owned() {
        let schema = PipelineBoundarySchema {
            head_dim: 8,
            hidden_size: 32,
            deepstack_count: 2,
        };
        let tensors = schema.wire_schema().unwrap().resolve(2, 5).unwrap();
        assert_eq!(tensors.primary().shape(), [2, 5, 32]);
        assert_eq!(tensors.auxiliary().len(), 5);
        assert_eq!(tensors.auxiliary()[0].shape(), [5, 8]);
        assert_eq!(tensors.auxiliary()[2].dtype(), BoundaryTensorDtype::Int32);
        assert_eq!(tensors.auxiliary()[3].role(), "deepstack.0");
        assert_eq!(tensors.auxiliary()[3].shape(), [2, 5, 32]);
    }
}
