//! Unified layered lifecycle for hybrid target and embedded-MTP execution.

use eredu_core::cache::PromptCacheTopology;
use eredu_nn::{
    AttentionCache, EmbeddingOperator, Error, Parameterized, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{
    LayerRuntimeState, LayeredArchitecture, LayeredForwardState, ModelStateIdentity,
    ResidentExpertProvider, RuntimeStateComponents, StateLayout,
};

use crate::{
    decoder::{StaticModuleSpec, StaticModules},
    hybrid_decoder::HybridDecoder,
};

use super::{
    prompt_cache_architecture_fingerprint, state_layout, Block, EmbeddedInput, ForwardMode,
    HybridConfig, PredictionUnit,
};

/// One target or configured prediction execution unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: RoutedNeuralBackend> {
    /// Ordinary hybrid decoder block.
    Target(Block<B>),
    /// One embedded prediction depth.
    Prediction(PredictionUnit<B>),
}

/// Request-local values retained through one target or prediction pass.
pub struct ForwardContext<T> {
    tokens: T,
    embedded: T,
    mask: Option<T>,
    mode: ForwardMode,
    target_hidden: Option<T>,
}

impl<T> ForwardContext<T> {
    /// Current target or prediction selection.
    pub const fn mode(&self) -> ForwardMode {
        self.mode
    }

    /// Hidden state captured at the selected execution-group boundary.
    pub const fn target_hidden(&self) -> Option<&T> {
        self.target_hidden.as_ref()
    }
}

/// One neutral layered model for Qwen3-Next and every Qwen3.5 text policy.
pub struct LayeredModel<B: RoutedNeuralBackend> {
    config: HybridConfig,
    decoder: HybridDecoder<B>,
    target_layers: usize,
    prediction_steps: usize,
}

impl<B: RoutedNeuralBackend> LayeredModel<B> {
    /// Builds unloaded pinned modules and the exact configured graph.
    pub fn new(
        config: HybridConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        config.validate().map_err(Error::backend)?;
        let target_layers = usize::try_from(config.num_hidden_layers).map_err(Error::backend)?;
        let prediction_steps =
            usize::try_from(config.mtp_num_hidden_layers).map_err(Error::backend)?;
        let embedding_name = "model.embed_tokens.weight";
        let decoder = HybridDecoder::new_with_prediction_groups(
            StaticModuleSpec {
                embedding_weight: embedding_name.into(),
                normalization_weight: "model.norm.weight".into(),
                head_weight: "lm_head.weight".into(),
                vocabulary: config.vocab_size,
                hidden_size: config.hidden_size,
                normalization_epsilon: config.rms_norm_eps,
                normalization_offset: 1.0,
                embedding_quantization: config.quantization,
                head_format: config.linear_format("lm_head.weight"),
                tied_head: config.tie_word_embeddings,
            },
            "model.layers",
            target_layers,
            "mtp.layers",
            prediction_steps,
            1,
            context,
        )?;
        Ok(Self {
            config,
            decoder,
            target_layers,
            prediction_steps,
        })
    }

    /// Returns normalized hybrid policy.
    pub const fn config(&self) -> &HybridConfig {
        &self.config
    }

    /// Borrows pinned text modules.
    pub const fn static_modules(&self) -> &StaticModules<B> {
        self.decoder.static_modules()
    }

    /// Mutably borrows pinned text modules for checkpoint binding.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        self.decoder.static_modules_mut()
    }

    /// Consumes the text graph and returns its pinned decoder modules.
    pub fn into_static_modules(self) -> StaticModules<B> {
        self.decoder.into_static_modules()
    }

    /// Returns target plus configured prediction state policy.
    pub fn state_layout(&self) -> Result<StateLayout, Error> {
        state_layout(&self.config).map_err(Error::backend)
    }

    fn state_index(&self, group: usize, index: usize) -> Result<usize, Error> {
        if group == 0 {
            return Ok(index);
        }
        if index != 0 || group > self.prediction_steps {
            return Err(Error::backend(format!(
                "Qwen hybrid execution address ({group}, {index}) is outside its graph"
            )));
        }
        self.target_layers
            .checked_add(group - 1)
            .ok_or_else(|| Error::backend("Qwen hybrid state index overflowed"))
    }

    /// Executes one graph unit through a runtime-owned routed-expert provider.
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
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.decoder.unit_path(group, index)?;
        let state_index = self.state_index(group, index)?;
        match unit {
            Unit::Target(block) if group == 0 => block.forward_with_provider(
                hidden,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                context,
                provider,
            ),
            Unit::Prediction(unit) if group > 0 => unit.forward_with_provider(
                hidden,
                &forward.embedded,
                forward.mask.as_ref(),
                state.layer(state_index).map_err(Error::backend)?,
                context,
                provider,
            ),
            _ => Err(Error::backend(format!(
                "Qwen hybrid unit does not match execution group {group}"
            ))),
        }
    }
}

impl<B, S> LayeredArchitecture<B, S> for LayeredModel<B>
where
    B: RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type Input<'a> = EmbeddedInput<'a, B::Tensor>;
    type StaticModules = StaticModules<B>;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = RetainedValues<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn model_identity(&self) -> &str {
        &self.config.model_type
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
        if group == 0 {
            Ok(Unit::Target(Block::new(&self.config, index, context)?))
        } else {
            Ok(Unit::Prediction(PredictionUnit::new(
                &self.config,
                group - 1,
                context,
            )?))
        }
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
                "Qwen hybrid runtime state layout {:?} does not match {expected:?}",
                state.layout()
            )));
        }
        let (tokens, supplied_hidden, supplied_mask, mode) = match input {
            EmbeddedInput::Target { tokens, mask } => (tokens, None, mask, ForwardMode::Target),
            EmbeddedInput::Draft {
                tokens,
                hidden,
                depth,
            } => {
                if depth >= self.prediction_steps {
                    return Err(Error::backend(format!(
                        "Qwen hybrid MTP depth {depth} is outside {} configured groups",
                        self.prediction_steps
                    )));
                }
                (tokens, Some(hidden), None, ForwardMode::Draft(depth))
            }
        };
        let embedded = self
            .decoder
            .static_modules_mut()
            .embeddings
            .forward(tokens, context)?;
        let hidden = supplied_hidden.cloned().unwrap_or_else(|| embedded.clone());
        let position_layer = match mode {
            ForwardMode::Target => 0,
            ForwardMode::Draft(depth) => self.target_layers + depth,
        };
        let mask = if let Some(mask) = supplied_mask {
            Some(mask.clone())
        } else if embedded.dim(1) > 1 {
            Some(B::causal_mask(
                embedded.dim(1),
                state
                    .layer(position_layer)
                    .map_err(Error::backend)?
                    .position(),
                None,
                context,
            )?)
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                tokens: tokens.clone(),
                embedded,
                mask,
                mode,
                target_hidden: None,
            },
        })
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

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        match forward.mode {
            ForwardMode::Target => group == 0,
            ForwardMode::Draft(depth) => group == depth + 1,
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

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 0 || matches!(forward.mode, ForwardMode::Draft(depth) if group == depth + 1) {
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
    ) -> Result<B::Tensor, Self::Error> {
        match forward.mode {
            ForwardMode::Target => self.decoder.finish_logits(hidden, context),
            ForwardMode::Draft(_) => self.decoder.project_logits(hidden, context),
        }
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        RetainedValues::new([
            Some(&forward.tokens),
            Some(&forward.embedded),
            forward.mask.as_ref(),
            forward.target_hidden.as_ref(),
        ])
    }
}

/// Allocation-free request-local retained tensor iterator.
pub struct RetainedValues<'a, T> {
    values: [Option<&'a T>; 4],
    next: usize,
}

impl<'a, T> RetainedValues<'a, T> {
    const fn new(values: [Option<&'a T>; 4]) -> Self {
        Self { values, next: 0 }
    }
}

impl<'a, T> Iterator for RetainedValues<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.values.len() {
            let value = self.values[self.next];
            self.next += 1;
            if value.is_some() {
                return value;
            }
        }
        None
    }
}

/// Declares prompt identity independently of concrete state storage.
pub fn state_identity(
    config: &HybridConfig,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    let layer_count = usize::try_from(config.num_hidden_layers)
        .and_then(|target| usize::try_from(config.mtp_num_hidden_layers).map(|mtp| target + mtp))
        .map_err(Error::backend)?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("Qwen hybrid owned layer range overflowed"))?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "Qwen hybrid owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    Ok(ModelStateIdentity {
        model_family: "qwen_hybrid".into(),
        effective_model_type: config.model_type.clone(),
        architecture_fingerprint: prompt_cache_architecture_fingerprint(config),
        layer_count,
        global_layer_start,
        sink_tokens: 0,
        layer_prefix_offsets: vec![0; layout.len()],
        topology,
    })
}
