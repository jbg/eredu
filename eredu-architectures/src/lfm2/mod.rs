//! Backend-neutral LFM2 and LFM2-MoE architecture policy.

/// Exact heterogeneous block equations and residual order.
pub mod block;
/// Pure SafeTensors/GGUF plans and canonical name translation.
pub mod checkpoint;
/// Strict configuration normalization and heterogeneous state geometry.
pub mod config;
/// Dense and routed feed-forward policy.
pub mod moe;
/// Semantic placement and rank-local construction geometry.
pub mod parallel;

pub use block::{Block, BlockGeometry, TokenMixer};
pub use checkpoint::{
    expert_recipes, expert_residency_catalog, gguf_plan, load_time_quantization,
    normalize_weight_formats, safetensors_plan, translate_gguf_weight_name, unit_recipes,
    with_checkpoint_formats,
};

pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, state_layout, state_layout_with_geometry, ConfigError,
    FeedForwardPolicy, LayerCacheGeometry, LayerPolicy, ModelArgs, OperatorPolicy, RopeConfig,
};
pub use moe::{expert_realization_plan, DenseSwiGlu, FeedForward, RoutedGatedProduct};
pub use parallel::{
    layer_parallel_parameter_groups, local_block_geometry, local_geometry, local_state_geometry,
    static_parallel_parameter_groups, LocalGeometry,
};

use eredu_core::cache::PromptCacheTopology;
use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, NormalizationOperator,
    RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionUnitLayout, ExpertPass, LayerRuntimeState,
    LayeredArchitecture, LayeredForwardState, LayeredPartitionInput, ModelStateIdentity,
    OwnedParameterGroupSpec, ParallelLayeredArchitecture, ParallelRoutedLayeredArchitecture,
    ParameterGroupOwner, PartitionedLayeredArchitecture, RoutedExpertProvider,
    RoutedLayeredArchitecture, RuntimeStateComponents, StateLayout,
};

use crate::{
    decoder::{LayeredInput, StaticModuleSpec, StaticModules},
    hybrid_decoder::HybridDecoder,
};

/// Architecture-owned values retained for one LFM2 forward pass.
pub struct ForwardContext<T> {
    mask: Option<T>,
}

/// Shared layered LFM2 lifecycle over a heterogeneous physical schedule.
pub struct LayeredModel<B: RoutedNeuralBackend> {
    args: ModelArgs,
    decoder: HybridDecoder<B>,
    parallel_geometry: Option<std::sync::Arc<LocalGeometry>>,
}

impl<B: RoutedNeuralBackend> eredu_runtime::ArchitectureParameters<B> for LayeredModel<B> {
    type DefinitionError = Error;

    fn state_layout(&self) -> Result<StateLayout, Self::DefinitionError> {
        self.state_layout_impl()
    }

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: PromptCacheTopology,
    ) -> Result<ModelStateIdentity, Self::DefinitionError> {
        state_identity(
            &self.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
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
        let modules = self.decoder.static_modules();
        visitor.visit("embedding", &modules.embeddings)?;
        visitor.visit("norm", &modules.norm)?;
        if let Some(head) = &modules.lm_head {
            visitor.visit("output", head)?;
        }
        Ok(())
    }

    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: eredu_runtime::StaticParameterVisitorMut<B>,
    {
        let modules = self.decoder.static_modules_mut();
        visitor.visit_mut("embedding", &mut modules.embeddings)?;
        visitor.visit_mut("norm", &mut modules.norm)?;
        if let Some(head) = &mut modules.lm_head {
            visitor.visit_mut("output", head)?;
        }
        Ok(())
    }
}

impl<B: RoutedNeuralBackend> LayeredModel<B> {
    /// Builds unloaded pinned modules and validates the complete schedule.
    pub fn new(args: ModelArgs, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
        let decoder =
            HybridDecoder::new(Self::static_spec(&args), "model.layers", layers, context)?;
        Ok(Self {
            args,
            decoder,
            parallel_geometry: None,
        })
    }

    /// Builds the same lifecycle with planner-derived rank-local modules.
    pub fn new_parallel(
        args: ModelArgs,
        geometry: LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        geometry.validate_for(&args).map_err(Error::backend)?;
        let layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
        let spec = Self::static_spec(&args);
        let mut decoder = HybridDecoder::new(spec.clone(), "model.layers", layers, context)?;
        *decoder.static_modules_mut() = StaticModules::from_parallel_spec(
            spec,
            geometry.embedding_range().clone(),
            geometry.output_range().cloned(),
            context,
        )?;
        Ok(Self {
            args,
            decoder,
            parallel_geometry: Some(std::sync::Arc::new(geometry)),
        })
    }

    fn static_spec(args: &ModelArgs) -> StaticModuleSpec {
        let embedding_name = "model.embed_tokens.weight";
        StaticModuleSpec {
            embedding_weight: embedding_name.into(),
            normalization_weight: "model.embedding_norm.weight".into(),
            head_weight: "lm_head.weight".into(),
            vocabulary: args.vocab_size,
            hidden_size: args.hidden_size,
            normalization_epsilon: args.norm_eps,
            normalization_offset: 0.0,
            embedding_quantization: args.weight_quantization_for(embedding_name),
            head_format: args.weight_quantization_for("lm_head.weight").into(),
            tied_head: args.tie_word_embeddings,
        }
    }

    /// Returns normalized architecture policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Describes pinned modules and heterogeneous units with explicit neutral
    /// graph ownership.
    fn parameter_description_impl(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        let graph = self.decoder.execution_graph()?;
        let count = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?;
        let layout = ExecutionUnitLayout::new(&graph, [count]).map_err(Error::backend)?;
        let static_groups = static_parallel_parameter_groups(self.decoder.static_modules())
            .map_err(Error::backend)?;
        let mut expected = static_groups.clone();
        let mut owned = static_groups
            .into_iter()
            .enumerate()
            .map(|(index, group)| {
                OwnedParameterGroupSpec::new(
                    if index == 0 && self.args.tie_word_embeddings {
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
            .collect::<Vec<_>>();
        let owner_group = layout.group_id(0).expect("LFM2 layout group").clone();
        for index in 0..count {
            let unit = self.construct_unit(0, index, context)?;
            let groups = layer_parallel_parameter_groups(&unit, &self.args, index)
                .map_err(Error::backend)?;
            expected.extend(groups.iter().cloned());
            owned.extend(groups.into_iter().map(|group| {
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(owner_group.clone(), index),
                    group,
                )
            }));
        }
        ArchitectureParameterDescription::new(&graph, &layout, expected, owned)
            .map_err(Error::backend)
    }

    /// Borrows pinned embedding, final normalization, and output modules.
    pub const fn static_modules(&self) -> &StaticModules<B> {
        self.decoder.static_modules()
    }

    /// Mutably borrows pinned embedding, final normalization, and output modules.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        self.decoder.static_modules_mut()
    }

    /// Returns the authoritative heterogeneous state declaration.
    pub fn state_layout(&self) -> Result<StateLayout, Error> {
        state_layout(&self.args).map_err(Error::backend)
    }

    /// State layout for this model's replicated or rank-local construction.
    fn state_layout_impl(&self) -> Result<StateLayout, Error> {
        self.parallel_geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .map_or_else(|| self.state_layout(), Ok)
    }

    /// Returns planner-derived geometry for a rank-local realization.
    pub fn parallel_geometry(&self) -> Option<&LocalGeometry> {
        self.parallel_geometry.as_deref()
    }

    /// Shares authoritative local geometry with a backend residency policy.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<LocalGeometry>> {
        self.parallel_geometry.as_ref().map(std::sync::Arc::clone)
    }

    /// Constructs one canonical execution unit from the model-owned geometry.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Block<B>, Error> {
        self.decoder.unit_path(group, index)?;
        match &self.parallel_geometry {
            Some(geometry) => Block::new_with_geometry(
                &self.args,
                index,
                *geometry.block(index).ok_or_else(|| {
                    Error::backend(format!("missing rank-local LFM2 block geometry {index}"))
                })?,
                context,
            ),
            None => Block::new(&self.args, index, context),
        }
    }

    /// Begins a forward from an embedding supplied by a placement-aware composition.
    pub fn begin_embedded_with_layout<S>(
        &self,
        hidden: B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        self.begin_embedded_partition_with_layout(hidden, mask, state, expected, 0, context)
    }

    /// Starts a pipeline partition from token ids or an upstream hidden state.
    fn prepare_partition<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => self
                .decoder
                .static_modules_mut()
                .embeddings
                .forward(tokens, context)?,
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        self.begin_embedded_partition_with_layout(
            hidden,
            mask,
            state,
            expected,
            first_state_ordinal,
            context,
        )
    }

    /// Starts the tensor-parallel form of the same neutral partition entry.
    #[allow(clippy::too_many_arguments)]
    fn prepare_partition_parallel<S>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => B::vocabulary_parallel_lookup(
                &mut self.decoder.static_modules_mut().embeddings,
                tokens,
                EmbeddingLookupPolicy::Strict,
                parallel,
                context,
            )?,
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        self.begin_embedded_partition_with_layout(
            hidden,
            mask,
            state,
            expected,
            first_state_ordinal,
            context,
        )
    }

    fn begin_embedded_partition_with_layout<S>(
        &self,
        hidden: B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        if state.layout() != expected {
            return Err(Error::backend(format!(
                "LFM2 runtime state layout {:?} does not match architecture layout {expected:?}",
                state.layout()
            )));
        }
        let sequence = hidden.dim(1);
        let mask = if let Some(mask) = mask {
            Some(mask.clone())
        } else if sequence > 1
            && self
                .args
                .layer_schedule
                .iter()
                .any(|policy| matches!(policy.operator, OperatorPolicy::SelfAttention(_)))
        {
            let position = state
                .layer(first_state_ordinal)
                .map_err(Error::backend)?
                .position();
            Some(B::causal_mask(sequence, position, None, context)?)
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext { mask },
        })
    }

    /// Executes one neutral block for placement-aware compositions.
    pub fn forward_block<S>(
        &mut self,
        index: usize,
        block: &mut Block<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.decoder.unit_path(0, index)?;
        block.forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(index).map_err(Error::backend)?,
            context,
        )
    }

    /// Executes one block while delegating routed feed-forward execution.
    pub fn forward_block_with_feed_forward<S, F>(
        &mut self,
        index: usize,
        block: &mut Block<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        F: FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        self.decoder.unit_path(0, index)?;
        block.forward_with_feed_forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(index).map_err(Error::backend)?,
            context,
            feed_forward,
        )
    }

    /// Executes the same block with tensor-parallel projections and collectives.
    pub fn forward_block_parallel<S>(
        &mut self,
        index: usize,
        block: &mut Block<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.decoder.unit_path(0, index)?;
        block.forward_parallel(
            hidden,
            forward.mask.as_ref(),
            state.layer(index).map_err(Error::backend)?,
            parallel,
            context,
        )
    }

    /// Executes one tensor-partitioned block while delegating its
    /// feed-forward policy to a runtime provider.
    pub fn forward_block_parallel_with_feed_forward<S, F>(
        &mut self,
        index: usize,
        block: &mut Block<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        F: FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            &B::ParallelContext,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        self.decoder.unit_path(0, index)?;
        block.forward_parallel_with_feed_forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(index).map_err(Error::backend)?,
            parallel,
            context,
            feed_forward,
        )
    }
}

impl<B, S> LayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type Input<'a> = LayeredInput<'a, B::Tensor>;
    type StaticModules = StaticModules<B>;
    type Unit = Block<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;

    fn group_transport(&self, _group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        crate::transport::decoder()
    }

    fn primary_execution_group(&self) -> &str {
        crate::decoder::TARGET_EXECUTION_GROUP
    }

    fn state_partition_plan(
        &self,
        layout: &eredu_runtime::StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        crate::transport::pipeline_state(0, layout)
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Self::Error> {
        self.decoder.execution_graph()
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        self.decoder.group_unit_count(group)
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        self.decoder.unit_path(group, index)
    }

    fn static_modules(&self) -> &Self::StaticModules {
        self.decoder.static_modules()
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        self.decoder.static_modules_mut()
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
        let expected = self.state_layout()?;
        if state.layout() != &expected {
            return Err(Error::backend(format!(
                "LFM2 runtime state layout {:?} does not match architecture layout {expected:?}",
                state.layout()
            )));
        }
        let hidden = self
            .decoder
            .static_modules_mut()
            .embeddings
            .forward(input.tokens, context)?;
        self.begin_embedded_with_layout(hidden, input.mask, state, &expected, context)
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.decoder.begin_group(group, initial, dependencies)
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
        self.decoder.unit_path(group, index)?;
        self.forward_block(index, unit, hidden, state, forward, context)
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.decoder.finish_logits(hidden, context)
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter()
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
            .ok_or_else(|| Error::backend("LFM2 model was not built with local geometry"))?
            .state_layout()
            .clone();
        let hidden = B::vocabulary_parallel_lookup(
            &mut self.decoder.static_modules_mut().embeddings,
            input.tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        self.begin_embedded_with_layout(hidden, input.mask, state, &expected, context)
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
        self.decoder.unit_path(group, index)?;
        self.forward_block_parallel(index, unit, hidden, state, forward, parallel, context)
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
                "LFM2 model was not built with local geometry",
            ));
        }
        let modules = self.decoder.static_modules_mut();
        let hidden = modules.norm.forward(hidden, context)?;
        match &mut modules.lm_head {
            Some(head) => B::vocabulary_parallel_project(head, &hidden, parallel, context),
            None => B::vocabulary_parallel_embedding_project(
                &mut modules.embeddings,
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
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        LayeredModel::prepare_partition(
            self,
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            context,
        )
    }

    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        LayeredModel::prepare_partition_parallel(
            self,
            input,
            mask,
            state,
            expected,
            first_state_ordinal,
            parallel,
            context,
        )
    }

    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::LayeredPartitionOutput<B::Tensor>, Self::Error> {
        if owns_output {
            let output = match parallel {
                Some(parallel) => {
                    self.finish_forward_parallel(hidden, state, forward, parallel, context)?
                }
                None => self.finish_forward(hidden, state, forward, context)?,
            };
            Ok(eredu_runtime::LayeredPartitionOutput::Final {
                output,
                retained: None,
            })
        } else {
            Ok(eredu_runtime::LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: eredu_runtime::NoAuxiliaryBoundary,
            })
        }
    }
}

impl<B, S> RoutedLayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn routed_observation_point(
        &self,
        group: usize,
        index: usize,
    ) -> Result<Option<eredu_runtime::RoutedObservationPoint>, Self::Error> {
        let unit_path = self.decoder.unit_path(group, index)?;
        Ok(self.args.routed_observation_point(&unit_path, index))
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
        self.decoder.unit_path(group, index)?;
        self.forward_block_with_feed_forward(
            index,
            unit,
            hidden,
            state,
            forward,
            context,
            |policy, normalized, context| {
                policy.forward_with_provider(normalized, pass, context, provider)
            },
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
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.decoder.unit_path(group, index)?;
        self.forward_block_parallel_with_feed_forward(
            index,
            unit,
            hidden,
            state,
            forward,
            parallel,
            context,
            |policy, normalized, parallel, context| {
                policy.forward_parallel_with_provider(normalized, pass, parallel, context, provider)
            },
        )
    }
}

/// Declares cache identity independently of its backend realization.
pub fn state_identity(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    let layer_count = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("LFM2 owned layer range overflowed"))?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "LFM2 owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    Ok(ModelStateIdentity {
        model_family: "lfm2".into(),
        effective_model_type: args.model_type.clone(),
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start,
        sink_tokens: 0,
        topology,
    })
}
