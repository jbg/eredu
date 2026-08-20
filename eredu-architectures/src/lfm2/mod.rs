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
pub use checkpoint::{gguf_plan, safetensors_plan, translate_gguf_weight_name};

pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, state_layout, state_layout_with_geometry, ConfigError,
    FeedForwardPolicy, GgufTensorCatalog, LayerCacheGeometry, LayerPolicy, ModelArgs,
    OperatorPolicy, RopeConfig,
};
pub use moe::{DenseSwiGlu, FeedForward, RoutedSwiGlu};
pub use parallel::{
    layer_parallel_parameter_groups, local_block_geometry, local_state_geometry,
    static_parallel_parameter_groups,
};

use eredu_core::cache::PromptCacheTopology;
use eredu_nn::{AttentionCache, EmbeddingOperator, Error, RoutedNeuralBackend, Tensor};
use eredu_runtime::{
    LayerRuntimeState, LayeredArchitecture, LayeredForwardState, ModelStateIdentity,
    RuntimeStateComponents, StateLayout,
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
}

impl<B: RoutedNeuralBackend> LayeredModel<B> {
    /// Builds unloaded pinned modules and validates the complete schedule.
    pub fn new(args: ModelArgs, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
        let embedding_name = "model.embed_tokens.weight";
        let decoder = HybridDecoder::new(
            StaticModuleSpec {
                embedding_weight: embedding_name.into(),
                normalization_weight: "model.embedding_norm.weight".into(),
                head_weight: "lm_head.weight".into(),
                vocabulary: args.vocab_size,
                hidden_size: args.hidden_size,
                normalization_epsilon: args.norm_eps,
                embedding_quantization: args.weight_quantization_for(embedding_name),
                head_format: args.weight_quantization_for("lm_head.weight").into(),
                tied_head: args.tie_word_embeddings,
            },
            "model.layers",
            layers,
            context,
        )?;
        Ok(Self { args, decoder })
    }

    /// Returns normalized architecture policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
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
            let position = state.layer(0).map_err(Error::backend)?.position();
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

    fn model_identity(&self) -> &str {
        &self.args.model_type
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
        self.decoder.unit_path(group, index)?;
        Block::new(&self.args, index, context)
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

/// Declares cache identity independently of its MLX realization.
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
        layer_prefix_offsets: vec![0; layout.len()],
        topology,
    })
}
