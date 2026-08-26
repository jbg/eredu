//! Conditional-generation graph over the shared Qwen vision tower and hybrid decoder.

use eredu_nn::{
    multimodal::{assemble_ordered_inputs, OrderedInputPart},
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, Index, LinearOperator,
    NormalizationOperator, Parameterized, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout,
    LayerRuntimeState, LayeredArchitecture, LayeredForwardState, LayeredPartitionInput,
    LayeredPartitionOutput, OwnedParameterGroupSpec, ParallelLayeredArchitecture,
    ParallelRoutedLayeredArchitecture, ParameterGroupOwner, PartitionedLayeredArchitecture,
    ResidentExpertProvider, RoutedExpertProvider, RoutedLayeredArchitecture,
    RuntimeStateComponents, StateLayout,
};

use crate::decoder::static_parallel_parameter_groups;
use crate::qwen::vision::{
    block_parallel_parameter_groups, static_parallel_parameter_groups as vision_parameter_groups,
    VisionBlock, VisionInput, VisionMode, VisionState, VisionStatic,
};
use crate::qwen::vl::InputPart;

use super::{
    unit_parallel_parameter_groups, Block, ConditionalLocalGeometry, ForwardMode,
    ParsedHybridConfig, PredictionUnit, Unit,
};

enum PreparedPart<T> {
    Text { tokens: T, embeddings: T },
    Media { tokens: T },
}

/// Pinned hybrid text and shared-vision modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct ConditionalStaticModules<B: RoutedNeuralBackend> {
    /// Hybrid token embedding, final normalization, and vocabulary head.
    pub text: crate::decoder::StaticModules<B>,
    /// Shared Qwen patch, position, and merger modules.
    pub vision: VisionStatic<B>,
}

/// One conditional vision, target-text, or MTP unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum ConditionalUnit<B: RoutedNeuralBackend> {
    /// Shared vision block.
    Vision(VisionBlock<B>),
    /// Hybrid recurrent/full-attention target block.
    Target(Block<B>),
    /// Configured hybrid prediction depth.
    Prediction(PredictionUnit<B>),
}

impl<B, S> RoutedLayeredArchitecture<B, S> for ConditionalLayeredModel<B>
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
        _pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match group {
            0 => <Self as LayeredArchitecture<B, S>>::forward_unit(
                self, group, index, unit, hidden, state, forward, context,
            ),
            _ => ConditionalLayeredModel::forward_unit_with_provider(
                self, group, index, unit, hidden, state, forward, provider, context,
            ),
        }
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for ConditionalLayeredModel<B>
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
        _pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match group {
            0 => <Self as ParallelLayeredArchitecture<B, S>>::forward_unit_parallel(
                self, group, index, unit, hidden, state, forward, parallel, context,
            ),
            _ => ConditionalLayeredModel::forward_unit_with_provider_parallel(
                self, group, index, unit, hidden, state, forward, provider, parallel, context,
            ),
        }
    }
}

/// Target media input or one embedded prediction request.
pub enum ConditionalInput<'a, T> {
    /// Ordered text/media target input.
    Target {
        /// Ordered semantic segments.
        parts: &'a [InputPart<'a, T>],
        /// Concatenated flattened patches for media segments.
        pixels: Option<&'a T>,
        /// Optional explicit decoder attention mask.
        mask: Option<&'a T>,
    },
    /// One configured MTP depth.
    Draft {
        /// Next-token identities.
        tokens: &'a T,
        /// Target or prior-depth hidden state.
        hidden: &'a T,
        /// Zero-based prediction depth.
        depth: usize,
    },
}

/// Request-local conditional execution values.
pub struct ConditionalForwardContext<T> {
    tokens: T,
    embedded: T,
    mask: Option<T>,
    mode: ForwardMode,
    parts: Vec<PreparedPart<T>>,
    vision_state: Option<VisionState<T>>,
    vision_initial: Option<T>,
    vision_output: Option<T>,
    deepstack: Vec<T>,
    visual_mask: Option<T>,
    target_hidden: Option<T>,
}

/// Request state transported while placed owners execute conditional vision.
pub struct ConditionalPipelineVisionState<T> {
    /// Current shared-vision activation, or a decoder-width placeholder for text-only input.
    pub hidden: T,
    parts: Vec<PreparedPart<T>>,
    mask: Option<T>,
    vision: Option<VisionState<T>>,
}

/// Decoder-facing conditional values after the placed vision group completes.
pub struct ConditionalPipelinePrepared<T> {
    /// Assembled decoder-width activation.
    pub hidden: T,
    /// Optional explicit or causal decoder mask.
    pub mask: Option<T>,
    /// Fixed-shape DeepStack additions indexed by decoder layer.
    pub deepstack: Vec<T>,
}

impl<T: Clone> ConditionalPipelinePrepared<T> {
    /// Converts target preparation into the canonical layered forward state.
    pub fn into_layered_forward(self) -> LayeredForwardState<T, ConditionalForwardContext<T>> {
        LayeredForwardState {
            hidden: self.hidden.clone(),
            context: ConditionalForwardContext {
                tokens: self.hidden.clone(),
                embedded: self.hidden,
                mask: self.mask,
                mode: ForwardMode::Target,
                parts: Vec::new(),
                vision_state: None,
                vision_initial: None,
                vision_output: None,
                deepstack: self.deepstack,
                visual_mask: None,
                target_hidden: None,
            },
        }
    }

    /// Recovers the decoder boundary from a completed layered forward.
    pub fn from_layered_forward(
        forward: LayeredForwardState<T, ConditionalForwardContext<T>>,
    ) -> (T, ConditionalPipelineBoundary<T>) {
        (
            forward.hidden,
            ConditionalPipelineBoundary {
                deepstack: forward.context.deepstack,
            },
        )
    }
}

/// Family-owned schema for conditional decoder values crossing pipeline ranks.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ConditionalPipelineBoundarySchema {
    hidden_size: i32,
    deepstack_count: usize,
}

impl ConditionalPipelineBoundarySchema {
    /// Returns the expected number of transported DeepStack tensors.
    pub const fn deepstack_count(self) -> usize {
        self.deepstack_count
    }
}

impl eredu_runtime::ArchitectureBoundary for ConditionalPipelineBoundarySchema {
    type Boundary<T> = ConditionalPipelineBoundary<T>;

    const IDENTITY: &'static str = "qwen_conditional.decoder";

    fn primary_tensor_spec(&self) -> eredu_runtime::BoundaryTensorSpec {
        eredu_runtime::BoundaryTensorSpec::primary_activation(self.hidden_size)
    }

    fn auxiliary_tensor_specs(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
        (0..self.deepstack_count)
            .map(|index| {
                eredu_runtime::BoundaryTensorSpec::new(
                    format!("deepstack.{index}"),
                    [Dim::Batch, Dim::Sequence, Dim::Fixed(self.hidden_size)],
                    Dtype::Activation,
                )
            })
            .collect()
    }

    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<Self::Boundary<T>, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        Ok(ConditionalPipelineBoundary { deepstack: tensors })
    }

    /// Encodes typed conditional decoder context after validating cardinality.
    fn encode<T>(
        &self,
        boundary: ConditionalPipelineBoundary<T>,
    ) -> Result<Vec<T>, eredu_runtime::ArchitectureBoundaryError> {
        if boundary.deepstack.len() != self.deepstack_count {
            return Err(eredu_runtime::ArchitectureBoundaryError::TensorCount {
                boundary: "qwen_conditional.decoder",
                expected: self.deepstack_count,
                actual: boundary.deepstack.len(),
            });
        }
        Ok(boundary.deepstack)
    }
}

/// Typed immutable decoder context transported between conditional partitions.
pub struct ConditionalPipelineBoundary<T> {
    /// Per-layer DeepStack additions.
    pub deepstack: Vec<T>,
}

impl<T> ConditionalPipelineBoundary<T> {
    /// Splits prepared state into its evolving activation and immutable boundary.
    pub fn from_prepared(prepared: ConditionalPipelinePrepared<T>) -> (T, Self) {
        (
            prepared.hidden,
            Self {
                deepstack: prepared.deepstack,
            },
        )
    }

    /// Reconstructs decoder state on a downstream partition.
    pub fn into_prepared(self, hidden: T) -> ConditionalPipelinePrepared<T> {
        ConditionalPipelinePrepared {
            hidden,
            mask: None,
            deepstack: self.deepstack,
        }
    }
}

/// Conditional target input from the input owner or an upstream pipeline rank.
pub enum ConditionalPartitionInput<'a, T> {
    /// Text token identities entering the target embedding boundary.
    Tokens {
        /// Token identities.
        tokens: &'a T,
        /// Existing target cache offset.
        offset: i32,
    },
    /// Evolving activation plus typed immutable target context.
    Hidden {
        /// Upstream target activation.
        hidden: T,
        /// Family-owned DeepStack boundary.
        boundary: ConditionalPipelineBoundary<T>,
    },
}

impl<T> ConditionalForwardContext<T> {
    /// Hidden state captured after the selected text execution group.
    pub const fn target_hidden(&self) -> Option<&T> {
        self.target_hidden.as_ref()
    }
}

/// One neutral conditional Qwen3.5 graph.
pub struct ConditionalLayeredModel<B: RoutedNeuralBackend> {
    parsed: ParsedHybridConfig,
    static_modules: ConditionalStaticModules<B>,
    target_layers: usize,
    prediction_steps: usize,
    parallel_geometry: Option<std::sync::Arc<ConditionalLocalGeometry>>,
}

impl<B: RoutedNeuralBackend> eredu_runtime::ArchitectureParameters<B>
    for ConditionalLayeredModel<B>
{
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
        let recipes = super::static_recipes(source)?;
        crate::static_parameters::module_recipes(&self.static_modules, recipes)
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

impl<B: RoutedNeuralBackend> ConditionalLayeredModel<B> {
    #[allow(clippy::too_many_arguments)]
    fn begin_distributed_partition<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, ConditionalPipelineBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ConditionalForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        if state.layout() != expected {
            return Err(Error::backend(
                "conditional Qwen partition state layout mismatch",
            ));
        }
        let (input, batch, sequence) = match input {
            LayeredPartitionInput::Tokens(tokens) => (
                ConditionalPartitionInput::Tokens {
                    tokens,
                    offset: state
                        .layer(first_state_ordinal)
                        .map_err(Error::backend)?
                        .position(),
                },
                tokens.dim(0),
                tokens.dim(1),
            ),
            LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                let batch = hidden.dim(0);
                let sequence = hidden.dim(1);
                (
                    ConditionalPartitionInput::Hidden {
                        hidden,
                        boundary: auxiliary,
                    },
                    batch,
                    sequence,
                )
            }
        };
        let offset = state
            .layer(first_state_ordinal)
            .map_err(Error::backend)?
            .position();
        self.begin_routed_target_partition(input, mask, batch, sequence, offset, parallel, context)
    }

    /// Prepares one routed target decoder partition with architecture-owned
    /// shape validation and causal-mask construction.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_routed_target_partition(
        &mut self,
        input: ConditionalPartitionInput<'_, B::Tensor>,
        explicit_mask: Option<&B::Tensor>,
        batch: i32,
        sequence: i32,
        offset: i32,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ConditionalForwardContext<B::Tensor>>, Error> {
        let mut prepared = self.begin_partition_target_inner(input, parallel, context)?;
        let expected = [batch, sequence, self.parsed.text.hidden_size];
        if prepared.hidden.shape() != expected {
            return Err(Error::backend(format!(
                "conditional Qwen3.5 decoder input is shaped {:?}, expected {expected:?}",
                prepared.hidden.shape()
            )));
        }
        prepared.mask = match explicit_mask {
            Some(mask) => Some(mask.clone()),
            None if sequence > 1 => Some(B::causal_mask(sequence, offset, None, context)?),
            None => None,
        };
        Ok(prepared.into_layered_forward())
    }

    fn begin_partition_target_inner(
        &mut self,
        input: ConditionalPartitionInput<'_, B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalPipelinePrepared<B::Tensor>, Error> {
        match input {
            ConditionalPartitionInput::Tokens { tokens, offset } => {
                let parts = [InputPart::Text(tokens)];
                let state = match parallel {
                    Some(parallel) => self.begin_pipeline_target_parallel(
                        &parts, None, None, offset, parallel, context,
                    ),
                    None => self.begin_pipeline_target(&parts, None, None, offset, context),
                }?;
                self.finish_pipeline_target(state, parallel, context)
            }
            ConditionalPartitionInput::Hidden { hidden, boundary } => {
                Ok(boundary.into_prepared(hidden))
            }
        }
    }

    /// Finishes the serial target output boundary.
    pub fn finish_partition_target(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_pipeline_logits(hidden, context)
    }

    /// Finishes the tensor-parallel target output boundary.
    pub fn finish_partition_target_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish_parallel(hidden, true, parallel, context)
    }

    /// Returns the normalized family configuration owned by this architecture.
    pub const fn args(&self) -> &ParsedHybridConfig {
        &self.parsed
    }

    fn canonical_execution_graph(&self) -> Result<ExecutionGraph, Error> {
        let mut groups = vec![
            ExecutionGroupSpec::root("vision"),
            ExecutionGroupSpec::with_dependencies("target", ["vision"]),
        ];
        let mut output = "target".to_owned();
        for depth in 0..self.prediction_steps {
            let id = format!("mtp.{depth}");
            groups.push(ExecutionGroupSpec::with_dependencies(
                id.clone(),
                [output.clone()],
            ));
            output = id;
        }
        ExecutionGraph::new(groups, output).map_err(Error::backend)
    }

    fn canonical_group_unit_count(&self, group: usize) -> Result<usize, Error> {
        match group {
            0 => Ok(self
                .parsed
                .vision
                .as_ref()
                .expect("validated vision")
                .layer_count()),
            1 => Ok(self.target_layers),
            group if group < self.prediction_steps + 2 => Ok(1),
            _ => Err(Error::backend(
                "conditional Qwen3.5 execution group is invalid",
            )),
        }
    }

    /// Returns the architecture-owned vision, target, and prediction traversal.
    pub fn unit_layout(&self) -> Result<ExecutionUnitLayout, Error> {
        let graph = self.canonical_execution_graph()?;
        let counts = (0..graph.groups().len())
            .map(|group| self.canonical_group_unit_count(group))
            .collect::<Result<Vec<_>, _>>()?;
        ExecutionUnitLayout::new(&graph, counts).map_err(Error::backend)
    }

    /// Returns the family-owned activation schema transported between stages.
    pub fn pipeline_boundary_schema(&self) -> ConditionalPipelineBoundarySchema {
        ConditionalPipelineBoundarySchema {
            hidden_size: self.parsed.text.hidden_size,
            deepstack_count: self
                .parsed
                .vision
                .as_ref()
                .expect("validated vision")
                .deepstack_layer_count(),
        }
    }

    /// Describes vision, target, and prediction parameters with explicit
    /// canonical graph ownership.
    fn parameter_description_impl(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let graph = self.canonical_execution_graph()?;
        let vision = self.parsed.vision.as_ref().expect("validated vision");
        let layout = self.unit_layout()?;
        let text_static = static_parallel_parameter_groups::<B>(
            &self.static_modules.text.embeddings,
            &self.static_modules.text.norm,
            self.static_modules.text.lm_head.as_ref(),
            "model",
        )
        .map_err(Error::backend)?;
        let vision_static =
            vision_parameter_groups::<B>(&self.static_modules.vision, vision, "model.visual")
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
                    if index == 0 {
                        let mut roles = vec!["embedding", "mtp"];
                        if self.parsed.text.tie_word_embeddings {
                            roles.push("output");
                        }
                        ParameterGroupOwner::static_any_of(roles)
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
        for group_index in 0..layout.group_count() {
            let group_id = layout
                .group_id(group_index)
                .expect("conditional layout group")
                .clone();
            let count = layout
                .group_range(group_index)
                .expect("conditional layout range")
                .len();
            for index in 0..count {
                let unit = self.construct_unit(group_index, index, context)?;
                let groups = match unit {
                    ConditionalUnit::Vision(block) => {
                        block_parallel_parameter_groups(&block, vision, "model.visual", index)
                    }
                    ConditionalUnit::Target(block) => unit_parallel_parameter_groups(
                        &Unit::Target(block),
                        &self.parsed.text,
                        0,
                        index,
                    ),
                    ConditionalUnit::Prediction(prediction) => unit_parallel_parameter_groups(
                        &Unit::Prediction(prediction),
                        &self.parsed.text,
                        group_index - 1,
                        index,
                    ),
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

    /// Builds the exact configured vision/text/MTP graph.
    pub fn new(
        parsed: ParsedHybridConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Qwen hybrid conditional",
            crate::operator_requirements::QWEN_HYBRID.union(crate::operator_requirements::QWEN_VL),
        )?;
        let vision = parsed
            .vision
            .clone()
            .ok_or_else(|| Error::backend("conditional Qwen3.5 requires vision config"))?;
        vision
            .validate_for(VisionMode::WindowScheduled)
            .map_err(Error::backend)?;
        let text_model = super::LayeredModel::<B>::new(parsed.text.clone(), context)?;
        let text = text_model.into_static_modules();
        let target_layers =
            usize::try_from(parsed.text.num_hidden_layers).map_err(Error::backend)?;
        let prediction_steps =
            usize::try_from(parsed.text.mtp_num_hidden_layers).map_err(Error::backend)?;
        Ok(Self {
            parsed,
            static_modules: ConditionalStaticModules {
                text,
                vision: VisionStatic::new_with_root(vision, "model.visual", context)?,
            },
            target_layers,
            prediction_steps,
            parallel_geometry: None,
        })
    }

    /// Applies the architecture-owned tensor-parallel token embedding boundary.
    pub fn pipeline_embed_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        B::vocabulary_parallel_lookup(
            &mut self.static_modules.text.embeddings,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )
    }

    /// Applies the architecture-owned tensor-parallel output boundary.
    pub fn pipeline_finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        normalize: bool,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = if normalize {
            self.static_modules.text.norm.forward(hidden, context)?
        } else {
            hidden.clone()
        };
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

    /// Builds the conditional graph with planner-derived local modules.
    pub fn new_parallel(
        parsed: ParsedHybridConfig,
        geometry: ConditionalLocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "Qwen hybrid conditional",
            crate::operator_requirements::QWEN_HYBRID.union(crate::operator_requirements::QWEN_VL),
        )?;
        geometry.validate_for(&parsed).map_err(Error::backend)?;
        let vision_config = parsed
            .vision
            .clone()
            .ok_or_else(|| Error::backend("conditional Qwen3.5 requires vision config"))?;
        vision_config
            .validate_for(VisionMode::WindowScheduled)
            .map_err(Error::backend)?;
        let text_model = super::LayeredModel::<B>::new_parallel(
            parsed.text.clone(),
            geometry.text().clone(),
            context,
        )?;
        let target_layers =
            usize::try_from(parsed.text.num_hidden_layers).map_err(Error::backend)?;
        let prediction_steps =
            usize::try_from(parsed.text.mtp_num_hidden_layers).map_err(Error::backend)?;
        Ok(Self {
            parsed,
            static_modules: ConditionalStaticModules {
                text: text_model.into_static_modules(),
                vision: VisionStatic::new_parallel_with_root(
                    vision_config,
                    "model.visual",
                    geometry.merger_widths(),
                    context,
                )?,
            },
            target_layers,
            prediction_steps,
            parallel_geometry: Some(std::sync::Arc::new(geometry)),
        })
    }

    /// Normalized composite policy.
    pub const fn parsed(&self) -> &ParsedHybridConfig {
        &self.parsed
    }

    /// Returns replicated or planner-derived target/MTP state geometry.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        self.parallel_geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .map_or_else(
                || super::state_layout(&self.parsed.text).map_err(Error::backend),
                Ok,
            )
    }

    /// Shares planner-owned local geometry with placed unit factories.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<ConditionalLocalGeometry>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    /// Constructs one canonical vision, target, or prediction unit using this
    /// model's replicated or planner-derived local geometry.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalUnit<B>, Error> {
        let count = match group {
            0 => self
                .parsed
                .vision
                .as_ref()
                .expect("validated vision")
                .layer_count(),
            1 => self.target_layers,
            group if group < self.prediction_steps + 2 => 1,
            _ => return Err(Error::backend("conditional Qwen3.5 group is invalid")),
        };
        if index >= count {
            return Err(Error::backend(
                "conditional Qwen3.5 unit is outside its group",
            ));
        }
        match group {
            0 => {
                let vision = self.parsed.vision.as_ref().expect("validated vision");
                match self
                    .parallel_geometry
                    .as_ref()
                    .and_then(|geometry| geometry.vision_block(index))
                {
                    Some((heads, intermediate)) => VisionBlock::new_parallel_with_root(
                        vision,
                        "model.visual",
                        index,
                        heads,
                        intermediate,
                        context,
                    ),
                    None => VisionBlock::new_with_root(vision, "model.visual", index, context),
                }
                .map(ConditionalUnit::Vision)
            }
            1 => {
                let config = self
                    .parallel_geometry
                    .as_ref()
                    .and_then(|geometry| geometry.text().target(index))
                    .unwrap_or(&self.parsed.text);
                Block::new(config, index, context).map(ConditionalUnit::Target)
            }
            _ => {
                let config = self
                    .parallel_geometry
                    .as_ref()
                    .and_then(|geometry| geometry.text().prediction(group - 2))
                    .unwrap_or(&self.parsed.text);
                PredictionUnit::new(config, group - 2, context).map(ConditionalUnit::Prediction)
            }
        }
    }

    fn prepare_parts(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Vec<PreparedPart<B::Tensor>>, Vec<(i32, i32, i32)>), Error> {
        if parts.is_empty() {
            return Err(Error::backend("conditional Qwen3.5 input has no parts"));
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
                InputPart::Projected { tokens, embeddings } => {
                    if embeddings.shape()
                        != [tokens.dim(0), tokens.dim(1), self.parsed.text.hidden_size]
                    {
                        return Err(Error::backend(
                            "conditional Qwen3.5 projected input geometry mismatch",
                        ));
                    }
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: (*embeddings).clone(),
                    })
                }
                InputPart::Image { tokens, grid } | InputPart::Video { tokens, grid } => {
                    grids.extend_from_slice(grid);
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
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
            return Err(Error::backend("conditional Qwen3.5 input has no parts"));
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
                InputPart::Projected { tokens, embeddings } => {
                    if embeddings.shape()
                        != [tokens.dim(0), tokens.dim(1), self.parsed.text.hidden_size]
                    {
                        return Err(Error::backend(
                            "conditional Qwen3.5 projected input geometry mismatch",
                        ));
                    }
                    Ok(PreparedPart::Text {
                        tokens: (*tokens).clone(),
                        embeddings: (*embeddings).clone(),
                    })
                }
                InputPart::Image { tokens, grid } | InputPart::Video { tokens, grid } => {
                    grids.extend_from_slice(grid);
                    Ok(PreparedPart::Media {
                        tokens: (*tokens).clone(),
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
                PreparedPart::Text { .. } => 0,
            })
            .sum::<i32>();
        match vision {
            Some(value) if value.shape() == [1, media_tokens, self.parsed.text.hidden_size] => {}
            None if media_tokens == 0 => {}
            _ => {
                return Err(Error::backend(
                    "conditional Qwen3.5 media placeholders and vision output disagree",
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
        assemble_ordered_inputs(&ordered, self.parsed.text.hidden_size, context)
    }

    fn state_index(&self, group: usize, index: usize) -> Result<usize, Error> {
        if group == 1 {
            return Ok(index);
        }
        if group >= 2 && group < self.prediction_steps + 2 && index == 0 {
            return Ok(self.target_layers + group - 2);
        }
        Err(Error::backend(
            "conditional Qwen3.5 state address is invalid",
        ))
    }

    /// Starts one pipeline target request before any placed vision block runs.
    pub fn begin_pipeline_target(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        pixels: Option<&B::Tensor>,
        mask: Option<&B::Tensor>,
        offset: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalPipelineVisionState<B::Tensor>, Error> {
        self.begin_pipeline_target_inner(parts, pixels, mask, offset, None, context)
    }

    /// Starts the same target request with rank-local vocabulary lookup.
    pub fn begin_pipeline_target_parallel(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        pixels: Option<&B::Tensor>,
        mask: Option<&B::Tensor>,
        offset: i32,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalPipelineVisionState<B::Tensor>, Error> {
        self.begin_pipeline_target_inner(parts, pixels, mask, offset, Some(parallel), context)
    }

    fn begin_pipeline_target_inner(
        &mut self,
        parts: &[InputPart<'_, B::Tensor>],
        pixels: Option<&B::Tensor>,
        mask: Option<&B::Tensor>,
        offset: i32,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalPipelineVisionState<B::Tensor>, Error> {
        let (parts, grids) = match parallel {
            Some(parallel) => self.prepare_parts_parallel(parts, parallel, context)?,
            None => self.prepare_parts(parts, context)?,
        };
        let has_media = !grids.is_empty();
        if has_media != pixels.is_some() {
            return Err(Error::backend(
                "conditional Qwen3.5 pixels and media metadata must appear together",
            ));
        }
        if has_media && offset != 0 {
            return Err(Error::backend(
                "conditional Qwen3.5 media cannot append to populated state",
            ));
        }
        let (vision_initial, vision) = match pixels {
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
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None if sequence > 1 => Some(B::causal_mask(sequence, offset, None, context)?),
            None => None,
        };
        let assembled = self.assemble(&parts, None, context);
        let hidden = match vision_initial {
            Some(hidden) => hidden,
            None => assembled?.embeddings,
        };
        Ok(ConditionalPipelineVisionState {
            hidden,
            parts,
            mask,
            vision,
        })
    }

    /// Whether the request has shared-vision work.
    pub fn pipeline_vision_active(state: &ConditionalPipelineVisionState<B::Tensor>) -> bool {
        state.vision.is_some()
    }

    /// Exports the tensors required by the next placed vision owner.
    pub fn pipeline_retained_values(
        state: &ConditionalPipelineVisionState<B::Tensor>,
    ) -> Vec<B::Tensor> {
        let mut values = vec![state.hidden.clone()];
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

    /// Installs values transported from the previous placed vision owner.
    pub fn replace_pipeline_retained_values(
        state: &mut ConditionalPipelineVisionState<B::Tensor>,
        values: Vec<B::Tensor>,
    ) -> Result<(), Error> {
        let expected = Self::pipeline_retained_values(state).len();
        if values.len() != expected {
            return Err(Error::backend(format!(
                "conditional Qwen3.5 pipeline continuation received {} tensors, expected {expected}",
                values.len()
            )));
        }
        let mut values = values.into_iter();
        state.hidden = values.next().expect("validated hidden");
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
        state: &mut ConditionalPipelineVisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        let vision = state
            .vision
            .as_mut()
            .ok_or_else(|| Error::backend("missing conditional pipeline vision state"))?;
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

    /// Finishes the shared merger and emits a fixed-shape decoder payload.
    pub fn finish_pipeline_target(
        &mut self,
        mut state: ConditionalPipelineVisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConditionalPipelinePrepared<B::Tensor>, Error> {
        let output = match &mut state.vision {
            Some(vision) => Some(match parallel {
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
            }),
            None => None,
        };
        let assembled = self.assemble(
            &state.parts,
            output.as_ref().map(|output| &output.embeddings),
            context,
        )?;
        let deepstack = match output {
            Some(output) => {
                let image = self.parsed.image_token_id.expect("validated image token");
                let video = self.parsed.video_token_id.expect("validated video token");
                let visual = assembled
                    .token_ids
                    .equal_i32(image, context)?
                    .logical_or(&assembled.token_ids.equal_i32(video, context)?, context)?;
                output
                    .deepstack_features
                    .into_iter()
                    .map(|features| {
                        assembled.embeddings.zeros_like(context)?.masked_scatter(
                            &visual,
                            &features.index(&[Index::At(0), Index::Full, Index::Full], context)?,
                            context,
                        )
                    })
                    .collect::<Result<Vec<_>, Error>>()?
            }
            None => (0..self
                .parsed
                .vision
                .as_ref()
                .expect("validated vision")
                .deepstack_layer_count())
                .map(|_| assembled.embeddings.zeros_like(context))
                .collect::<Result<Vec<_>, Error>>()?,
        };
        Ok(ConditionalPipelinePrepared {
            hidden: assembled.embeddings,
            mask: state.mask,
            deepstack,
        })
    }

    /// Applies the conditional target normalization and vocabulary projection.
    pub fn finish_pipeline_logits(
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

    /// Executes one text unit through a runtime-owned routed-expert provider.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut ConditionalUnit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ConditionalForwardContext<B::Tensor>,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let state_index = self.state_index(group, index)?;
        let mut output = match unit {
            ConditionalUnit::Target(block) if group == 1 => block.forward_with_provider(
                hidden,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                context,
                provider,
            )?,
            ConditionalUnit::Prediction(unit) if group >= 2 => unit.forward_with_provider(
                hidden,
                &forward.embedded,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                context,
                provider,
            )?,
            _ => return Err(Error::backend("conditional Qwen3.5 unit/group mismatch")),
        };
        if group == 1 {
            if let Some(features) = forward.deepstack.get(index) {
                output = if features.shape() == output.shape() {
                    output.add(features, context)?
                } else {
                    let source =
                        features.index(&[Index::At(0), Index::Full, Index::Full], context)?;
                    output.add(
                        &output.zeros_like(context)?.masked_scatter(
                            forward
                                .visual_mask
                                .as_ref()
                                .ok_or_else(|| Error::backend("missing conditional visual mask"))?,
                            &source,
                            context,
                        )?,
                        context,
                    )?
                };
            }
        }
        Ok(output)
    }

    /// Executes one text unit with local projections and required collectives.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider_parallel<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut ConditionalUnit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ConditionalForwardContext<B::Tensor>,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let state_index = self.state_index(group, index)?;
        let mut output = match unit {
            ConditionalUnit::Target(block) if group == 1 => block.forward_parallel(
                hidden,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                parallel,
                context,
                provider,
            )?,
            ConditionalUnit::Prediction(unit) if group >= 2 => unit.forward_parallel(
                hidden,
                &forward.embedded,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                parallel,
                context,
                provider,
            )?,
            _ => {
                return Err(Error::backend(
                    "conditional Qwen3.5 parallel unit/group mismatch",
                ))
            }
        };
        if group == 1 {
            if let Some(features) = forward.deepstack.get(index) {
                output = if features.shape() == output.shape() {
                    output.add(features, context)?
                } else {
                    let source =
                        features.index(&[Index::At(0), Index::Full, Index::Full], context)?;
                    output.add(
                        &output.zeros_like(context)?.masked_scatter(
                            forward
                                .visual_mask
                                .as_ref()
                                .ok_or_else(|| Error::backend("missing conditional visual mask"))?,
                            &source,
                            context,
                        )?,
                        context,
                    )?
                };
            }
        }
        Ok(output)
    }
}

impl<B, S> LayeredArchitecture<B, S> for ConditionalLayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type Input<'a> = ConditionalInput<'a, B::Tensor>;
    type StaticModules = ConditionalStaticModules<B>;
    type Unit = ConditionalUnit<B>;
    type ForwardContext = ConditionalForwardContext<B::Tensor>;
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
                first_owner_static_roles: vec!["vision".into()],
                last_owner_static_roles: Vec::new(),
                merge_destination: eredu_runtime::ArchitectureMergeDestination::FirstPipelineOwner,
                parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::TensorSharded),
                request_optional: true,
            },
            1 => crate::transport::decoder(),
            _ => conditional_prediction_group_transport(group),
        }
    }

    fn model_identity(&self) -> &str {
        &self.parsed.text.model_type
    }

    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        self.canonical_execution_graph()
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        self.canonical_group_unit_count(group)
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if index >= <Self as LayeredArchitecture<B, S>>::group_unit_count(self, group)? {
            return Err(Error::backend(
                "conditional Qwen3.5 unit is outside its group",
            ));
        }
        Ok(match group {
            0 => format!("model.visual.blocks.{index}"),
            1 => format!("model.layers.{index}"),
            _ => format!("mtp.layers.{}", group - 2),
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
    ) -> Result<Self::Unit, Error> {
        self.construct_unit(group, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let expected = super::state_layout(&self.parsed.text).map_err(Error::backend)?;
        if state.layout() != &expected {
            return Err(Error::backend("conditional Qwen3.5 state layout mismatch"));
        }
        match input {
            ConditionalInput::Draft {
                tokens,
                hidden,
                depth,
            } => {
                if depth >= self.prediction_steps {
                    return Err(Error::backend("conditional Qwen3.5 MTP depth is invalid"));
                }
                let embedded = self
                    .static_modules
                    .text
                    .embeddings
                    .forward(tokens, context)?;
                let state_index = self.target_layers + depth;
                let offset = state.layer(state_index).map_err(Error::backend)?.position();
                let mask = if tokens.dim(1) > 1 {
                    Some(B::causal_mask(tokens.dim(1), offset, None, context)?)
                } else {
                    None
                };
                Ok(LayeredForwardState {
                    hidden: hidden.clone(),
                    context: ConditionalForwardContext {
                        tokens: tokens.clone(),
                        embedded,
                        mask,
                        mode: ForwardMode::Draft(depth),
                        parts: Vec::new(),
                        vision_state: None,
                        vision_initial: None,
                        vision_output: None,
                        deepstack: Vec::new(),
                        visual_mask: None,
                        target_hidden: None,
                    },
                })
            }
            ConditionalInput::Target {
                parts,
                pixels,
                mask,
            } => {
                let (prepared, grids) = self.prepare_parts(parts, context)?;
                let has_media = !grids.is_empty();
                if has_media != pixels.is_some() {
                    return Err(Error::backend(
                        "conditional Qwen3.5 pixels and media metadata must appear together",
                    ));
                }
                let offset = state.layer(0).map_err(Error::backend)?.position();
                if has_media && offset != 0 {
                    return Err(Error::backend(
                        "conditional Qwen3.5 media cannot append to populated state",
                    ));
                }
                let (vision_initial, vision_state) = match pixels {
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
                let assembled = self.assemble(&prepared, None, context);
                let sequence = prepared
                    .iter()
                    .map(|part| match part {
                        PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => {
                            tokens.dim(1)
                        }
                    })
                    .sum::<i32>();
                let decoder_mask = match mask {
                    Some(mask) => Some(mask.clone()),
                    None if sequence > 1 => Some(B::causal_mask(sequence, offset, None, context)?),
                    None => None,
                };
                let (tokens, embedded, error) = match assembled {
                    Ok(value) => (Some(value.token_ids), Some(value.embeddings), None),
                    Err(error) => (None, None, Some(error)),
                };
                let hidden = vision_initial
                    .as_ref()
                    .cloned()
                    .or_else(|| embedded.clone())
                    .ok_or_else(|| {
                        error.unwrap_or_else(|| Error::backend("empty conditional input"))
                    })?;
                Ok(LayeredForwardState {
                    hidden,
                    context: ConditionalForwardContext {
                        tokens: tokens.unwrap_or_else(|| hidden_token_placeholder(&prepared)),
                        embedded: embedded
                            .unwrap_or_else(|| hidden_embedding_placeholder(&prepared)),
                        mask: decoder_mask,
                        mode: ForwardMode::Target,
                        parts: prepared,
                        vision_state,
                        vision_initial,
                        vision_output: None,
                        deepstack: Vec::new(),
                        visual_mask: None,
                        target_hidden: None,
                    },
                })
            }
        }
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match group {
            0 => Ok(forward.vision_initial.as_ref().unwrap_or(initial).clone()),
            1 if matches!(forward.mode, ForwardMode::Target) => {
                let assembled =
                    self.assemble(&forward.parts, forward.vision_output.as_ref(), context)?;
                let image = self.parsed.image_token_id.expect("validated image token");
                let video = self.parsed.video_token_id.expect("validated video token");
                forward.visual_mask = if forward.deepstack.is_empty() {
                    None
                } else {
                    Some(
                        assembled
                            .token_ids
                            .equal_i32(image, context)?
                            .logical_or(&assembled.token_ids.equal_i32(video, context)?, context)?,
                    )
                };
                forward.tokens = assembled.token_ids;
                forward.embedded = assembled.embeddings.clone();
                Ok(assembled.embeddings)
            }
            _ => Ok(dependencies.first().copied().unwrap_or(initial).clone()),
        }
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        match forward.mode {
            ForwardMode::Target => group == 1 || (group == 0 && forward.vision_state.is_some()),
            ForwardMode::Draft(depth) => group == depth + 2,
        }
    }

    fn state_ordinal(&self, group: usize, index: usize, _ordinal: usize) -> usize {
        match group {
            0 => 0,
            1 => index,
            _ => self.target_layers + group - 2,
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
        if group == 0 {
            let ConditionalUnit::Vision(block) = unit else {
                return Err(Error::backend("conditional vision unit/group mismatch"));
            };
            self.static_modules.vision.forward_block(
                block,
                index,
                hidden,
                forward
                    .vision_state
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing conditional vision state"))?,
                context,
            )
        } else {
            self.forward_unit_with_provider(
                group,
                index,
                unit,
                hidden,
                state,
                forward,
                &mut ResidentExpertProvider,
                context,
            )
        }
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group == 0 {
            if let Some(vision_state) = forward.vision_state.as_mut() {
                let output = self
                    .static_modules
                    .vision
                    .finish(hidden, vision_state, context)?;
                forward.deepstack = output.deepstack_features;
                forward.vision_output = Some(output.embeddings);
                return Ok(forward.vision_output.as_ref().unwrap().clone());
            }
        }
        if group == 1 || matches!(forward.mode, ForwardMode::Draft(depth) if group == depth + 2) {
            forward.target_hidden = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = match forward.mode {
            ForwardMode::Target => self.static_modules.text.norm.forward(hidden, context)?,
            ForwardMode::Draft(_) => hidden.clone(),
        };
        match &mut self.static_modules.text.lm_head {
            Some(head) => head.forward(&hidden, context),
            None => self
                .static_modules
                .text
                .embeddings
                .as_linear(&hidden, context),
        }
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        let mut values = vec![&forward.tokens, &forward.embedded];
        values.extend(forward.mask.iter());
        values.extend(forward.vision_initial.iter());
        values.extend(forward.vision_output.iter());
        values.extend(forward.deepstack.iter());
        values.extend(forward.visual_mask.iter());
        values.extend(forward.target_hidden.iter());
        if let Some(state) = &forward.vision_state {
            values.extend(state.retained_values());
        }
        values.into_iter()
    }
}

fn conditional_prediction_group_transport(
    group: usize,
) -> eredu_runtime::ArchitectureGroupTransport {
    let mut transport = crate::transport::prediction();
    if group == 2 {
        transport.first_owner_static_roles.push("mtp".into());
    }
    transport
}

#[cfg(test)]
mod transport_tests {
    use super::{conditional_prediction_group_transport, ConditionalPipelineBoundarySchema};
    use eredu_runtime::ArchitectureBoundary;

    #[test]
    fn conditional_deepstack_count_owns_wire_cardinality() {
        let schema = ConditionalPipelineBoundarySchema {
            hidden_size: 48,
            deepstack_count: 3,
        };
        let tensors = schema.wire_schema().unwrap().resolve(2, 4).unwrap();
        assert_eq!(tensors.primary().shape(), [2, 4, 48]);
        assert_eq!(tensors.auxiliary().len(), 3);
        assert_eq!(tensors.auxiliary()[0].role(), "deepstack.0");
        assert_eq!(tensors.auxiliary()[2].shape(), [2, 4, 48]);
    }

    #[test]
    fn first_conditional_prediction_group_owns_shared_mtp_embedding_role_once() {
        assert_eq!(
            conditional_prediction_group_transport(2).first_owner_static_roles,
            ["mtp"]
        );
        assert!(conditional_prediction_group_transport(3)
            .first_owner_static_roles
            .is_empty());
    }
}

impl<B, S> ParallelLayeredArchitecture<B, S> for ConditionalLayeredModel<B>
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
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let expected = self
            .parallel_geometry
            .as_ref()
            .ok_or_else(|| Error::backend("conditional Qwen3.5 has no local geometry"))?
            .state_layout();
        if state.layout() != expected {
            return Err(Error::backend(
                "conditional Qwen3.5 rank-local state layout mismatch",
            ));
        }
        match input {
            ConditionalInput::Draft {
                tokens,
                hidden,
                depth,
            } => {
                if depth >= self.prediction_steps {
                    return Err(Error::backend("conditional Qwen3.5 MTP depth is invalid"));
                }
                let embedded = B::vocabulary_parallel_lookup(
                    &mut self.static_modules.text.embeddings,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?;
                let state_index = self.target_layers + depth;
                let offset = state.layer(state_index).map_err(Error::backend)?.position();
                let mask = if tokens.dim(1) > 1 {
                    Some(B::causal_mask(tokens.dim(1), offset, None, context)?)
                } else {
                    None
                };
                Ok(LayeredForwardState {
                    hidden: hidden.clone(),
                    context: ConditionalForwardContext {
                        tokens: tokens.clone(),
                        embedded,
                        mask,
                        mode: ForwardMode::Draft(depth),
                        parts: Vec::new(),
                        vision_state: None,
                        vision_initial: None,
                        vision_output: None,
                        deepstack: Vec::new(),
                        visual_mask: None,
                        target_hidden: None,
                    },
                })
            }
            ConditionalInput::Target {
                parts,
                pixels,
                mask,
            } => {
                let (prepared, grids) = self.prepare_parts_parallel(parts, parallel, context)?;
                let has_media = !grids.is_empty();
                if has_media != pixels.is_some() {
                    return Err(Error::backend(
                        "conditional Qwen3.5 pixels and media metadata must appear together",
                    ));
                }
                let offset = state.layer(0).map_err(Error::backend)?.position();
                if has_media && offset != 0 {
                    return Err(Error::backend(
                        "conditional Qwen3.5 media cannot append to populated state",
                    ));
                }
                let (vision_initial, vision_state) = match pixels {
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
                let assembled = self.assemble(&prepared, None, context);
                let sequence = prepared
                    .iter()
                    .map(|part| match part {
                        PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => {
                            tokens.dim(1)
                        }
                    })
                    .sum::<i32>();
                let decoder_mask = match mask {
                    Some(mask) => Some(mask.clone()),
                    None if sequence > 1 => Some(B::causal_mask(sequence, offset, None, context)?),
                    None => None,
                };
                let (tokens, embedded, error) = match assembled {
                    Ok(value) => (Some(value.token_ids), Some(value.embeddings), None),
                    Err(error) => (None, None, Some(error)),
                };
                let hidden = vision_initial
                    .as_ref()
                    .cloned()
                    .or_else(|| embedded.clone())
                    .ok_or_else(|| {
                        error.unwrap_or_else(|| Error::backend("empty conditional input"))
                    })?;
                Ok(LayeredForwardState {
                    hidden,
                    context: ConditionalForwardContext {
                        tokens: tokens.unwrap_or_else(|| hidden_token_placeholder(&prepared)),
                        embedded: embedded
                            .unwrap_or_else(|| hidden_embedding_placeholder(&prepared)),
                        mask: decoder_mask,
                        mode: ForwardMode::Target,
                        parts: prepared,
                        vision_state,
                        vision_initial,
                        vision_output: None,
                        deepstack: Vec::new(),
                        visual_mask: None,
                        target_hidden: None,
                    },
                })
            }
        }
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
    ) -> Result<B::Tensor, Error> {
        if group == 0 {
            let ConditionalUnit::Vision(block) = unit else {
                return Err(Error::backend("conditional vision unit/group mismatch"));
            };
            self.static_modules.vision.forward_block_parallel(
                block,
                index,
                hidden,
                forward
                    .vision_state
                    .as_mut()
                    .ok_or_else(|| Error::backend("missing conditional vision state"))?,
                parallel,
                context,
            )
        } else {
            self.forward_unit_with_provider_parallel(
                group,
                index,
                unit,
                hidden,
                state,
                forward,
                &mut ResidentExpertProvider,
                parallel,
                context,
            )
        }
    }

    fn complete_execution_group_parallel(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
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
        if group == 1 || matches!(forward.mode, ForwardMode::Draft(depth) if group == depth + 2) {
            forward.target_hidden = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if self.parallel_geometry.is_none() {
            return Err(Error::backend("conditional Qwen3.5 has no local geometry"));
        }
        let hidden = match forward.mode {
            ForwardMode::Target => self.static_modules.text.norm.forward(hidden, context)?,
            ForwardMode::Draft(_) => hidden.clone(),
        };
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

impl<B, S> PartitionedLayeredArchitecture<B, S> for ConditionalLayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type Boundary = ConditionalPipelineBoundarySchema;

    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(self.pipeline_boundary_schema())
    }

    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, ConditionalPipelineBoundary<B::Tensor>>,
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
        input: LayeredPartitionInput<'a, B::Tensor, ConditionalPipelineBoundary<B::Tensor>>,
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
    ) -> Result<
        LayeredPartitionOutput<B::Tensor, ConditionalPipelineBoundary<B::Tensor>>,
        Self::Error,
    > {
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
                auxiliary: ConditionalPipelineBoundary {
                    deepstack: forward.deepstack.clone(),
                },
            })
        }
    }
}

fn hidden_token_placeholder<T: Clone>(parts: &[PreparedPart<T>]) -> T {
    match &parts[0] {
        PreparedPart::Text { tokens, .. } | PreparedPart::Media { tokens } => tokens.clone(),
    }
}

fn hidden_embedding_placeholder<T: Clone>(parts: &[PreparedPart<T>]) -> T {
    parts
        .iter()
        .find_map(|part| match part {
            PreparedPart::Text { embeddings, .. } => Some(embeddings.clone()),
            PreparedPart::Media { .. } => None,
        })
        .unwrap_or_else(|| hidden_token_placeholder(parts))
}
