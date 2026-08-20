//! Graph-visible embedded multi-token prediction units.

use eredu_nn::{
    AttentionCache, Error, LinearOperator, LinearSpec, NormalizationOperator, NormalizationSpec,
    ParameterSpec, Parameterized, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{RoutedExpertProvider, RuntimeStateComponents};

use super::{Block, LayerGeometry, LayerPolicy, ModelArgs};

/// Borrowed input selecting target execution or one MTP prediction depth.
pub enum EmbeddedInput<'a, T> {
    /// Execute the complete target schedule.
    Target {
        /// Token ids shaped `[batch, sequence]`.
        tokens: &'a T,
        /// Optional caller-provided attention mask.
        mask: Option<&'a T>,
    },
    /// Execute exactly one appended MTP prediction group.
    Draft {
        /// Token ids embedded for the proposed position.
        tokens: &'a T,
        /// Target or prior prediction hidden state.
        hidden: &'a T,
        /// Zero-based MTP prediction depth.
        depth: usize,
    },
}

impl<'a, T> EmbeddedInput<'a, T> {
    /// Creates target-model input.
    pub const fn target(tokens: &'a T, mask: Option<&'a T>) -> Self {
        Self::Target { tokens, mask }
    }

    /// Creates one MTP draft input.
    pub const fn draft(tokens: &'a T, hidden: &'a T, depth: usize) -> Self {
        Self::Draft {
            tokens,
            hidden,
            depth,
        }
    }
}

/// Execution selection retained for one model invocation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ForwardMode {
    /// Target physical units execute.
    Target,
    /// Exactly one MTP depth executes.
    Draft(usize),
}

/// One physical operator inside an appended MTP prediction group.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct PredictionUnit<B: RoutedNeuralBackend> {
    /// First-unit normalization of the current token embedding.
    pub embedding_norm: Option<B::Normalization>,
    /// First-unit normalization of the prior hidden state.
    pub hidden_norm: Option<B::Normalization>,
    /// First-unit projection of concatenated embedding and hidden state.
    pub fusion: Option<B::Linear>,
    /// Scheduled attention or sparse-MoE physical unit.
    pub block: Block<B>,
    /// Last-unit normalization before the shared vocabulary projection.
    pub final_norm: Option<B::Normalization>,
}

impl<B: RoutedNeuralBackend> PredictionUnit<B> {
    /// Builds one physical unit at a prediction-depth-relative position.
    pub fn new(
        args: &ModelArgs,
        depth: usize,
        relative: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let steps = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?;
        if depth >= steps {
            return Err(Error::backend(format!(
                "Nemotron-H MTP depth {depth} is outside {steps} prediction steps"
            )));
        }
        let policies = args.mtp_policies().map_err(Error::backend)?;
        let pattern_len = policies
            .len()
            .checked_div(steps)
            .filter(|length| *length > 0)
            .ok_or_else(|| Error::backend("Nemotron-H MTP pattern is empty"))?;
        if relative >= pattern_len {
            return Err(Error::backend(format!(
                "Nemotron-H MTP unit {relative} is outside pattern length {pattern_len}"
            )));
        }
        let physical = depth
            .checked_mul(pattern_len)
            .and_then(|start| start.checked_add(relative))
            .ok_or_else(|| Error::backend("Nemotron-H MTP physical index overflowed"))?;
        let policy = policies[physical];
        let geometry = match policy {
            LayerPolicy::SelfAttention(_) => LayerGeometry::Attention {
                query_heads: args.num_attention_heads,
                kv_heads: args.num_key_value_heads,
            },
            LayerPolicy::SparseMoe => LayerGeometry::SparseMoe {
                routed: args.moe_intermediate_size,
                shared: args.moe_shared_expert_intermediate_size,
            },
            _ => {
                return Err(Error::backend(format!(
                    "unsupported Nemotron-H MTP policy {policy:?}"
                )))
            }
        };
        Self::new_with_geometry(args, depth, relative, policy, geometry, context)
    }

    /// Builds one prediction unit from placement-resolved local geometry.
    pub fn new_with_geometry(
        args: &ModelArgs,
        depth: usize,
        relative: usize,
        policy: LayerPolicy,
        geometry: LayerGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let steps = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?;
        let policies = args.mtp_policies().map_err(Error::backend)?;
        let pattern_len = policies
            .len()
            .checked_div(steps)
            .filter(|length| *length > 0)
            .ok_or_else(|| Error::backend("Nemotron-H MTP pattern is empty"))?;
        if depth >= steps || relative >= pattern_len {
            return Err(Error::backend(
                "Nemotron-H MTP unit is outside its schedule",
            ));
        }
        let physical = depth
            .checked_mul(pattern_len)
            .and_then(|start| start.checked_add(relative))
            .ok_or_else(|| Error::backend("Nemotron-H MTP physical index overflowed"))?;
        if policies[physical] != policy {
            return Err(Error::backend(format!(
                "Nemotron-H MTP policy {policy:?} does not match schedule {:?}",
                policies[physical]
            )));
        }
        let root = format!("model.mtp.layers.{physical}");
        let parameter = |name: String| ParameterSpec::trainable(name).map_err(Error::backend);
        let norm = |name: String| {
            B::rms_norm(
                NormalizationSpec {
                    dimensions: args.hidden_size,
                    epsilon: args.layer_norm_epsilon,
                    weight: parameter(name)?,
                },
                context,
            )
        };
        let first = relative == 0;
        let last = relative + 1 == pattern_len;
        let fusion_name = format!("{root}.eh_proj.weight");
        Ok(Self {
            embedding_norm: first
                .then(|| norm(format!("{root}.enorm.weight")))
                .transpose()?,
            hidden_norm: first
                .then(|| norm(format!("{root}.hnorm.weight")))
                .transpose()?,
            fusion: first
                .then(|| {
                    B::linear(
                        LinearSpec {
                            input: args.hidden_size * 2,
                            output: args.hidden_size,
                            weight: parameter(fusion_name.clone())?,
                            bias: None,
                            format: args.weight_quantization_for(&fusion_name).into(),
                        },
                        context,
                    )
                })
                .transpose()?,
            block: Block::new_mtp_with_geometry(args, physical, policy, geometry, context)?,
            final_norm: last
                .then(|| norm(format!("{root}.final_layernorm.weight")))
                .transpose()?,
        })
    }

    /// Executes one physical prediction unit with resident experts.
    pub fn forward<S>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.forward_with_provider(
            hidden,
            embedded,
            mask,
            state,
            context,
            &mut eredu_runtime::ResidentExpertProvider,
        )
    }

    /// Executes one physical prediction unit through a runtime expert provider.
    pub fn forward_with_provider<S, P>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let hidden = match (
            &mut self.embedding_norm,
            &mut self.hidden_norm,
            &mut self.fusion,
        ) {
            (Some(embedding_norm), Some(hidden_norm), Some(fusion)) => {
                let embedded = embedding_norm.forward(embedded, context)?;
                let hidden = hidden_norm.forward(hidden, context)?;
                let fused = B::Tensor::concatenate(&[embedded, hidden], -1, context)?;
                fusion.forward(&fused, context)?
            }
            (None, None, None) => hidden.clone(),
            _ => return Err(Error::backend("incomplete Nemotron-H MTP fusion unit")),
        };
        let hidden = self
            .block
            .forward_with_provider(&hidden, mask, state, context, provider)?;
        match &mut self.final_norm {
            Some(norm) => norm.forward(&hidden, context),
            None => Ok(hidden),
        }
    }

    /// Executes one physical prediction unit with tensor collectives.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider<S, P>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let hidden = match (
            &mut self.embedding_norm,
            &mut self.hidden_norm,
            &mut self.fusion,
        ) {
            (Some(embedding_norm), Some(hidden_norm), Some(fusion)) => {
                let embedded = embedding_norm.forward(embedded, context)?;
                let hidden = hidden_norm.forward(hidden, context)?;
                let fused = B::Tensor::concatenate(&[embedded, hidden], -1, context)?;
                fusion.forward(&fused, context)?
            }
            (None, None, None) => hidden.clone(),
            _ => return Err(Error::backend("incomplete Nemotron-H MTP fusion unit")),
        };
        let hidden = self
            .block
            .forward_parallel(&hidden, mask, state, parallel, context, provider)?;
        match &mut self.final_norm {
            Some(norm) => norm.forward(&hidden, context),
            None => Ok(hidden),
        }
    }
}

/// Allocation-free retained tensor iterator for target and draft forwards.
pub struct RetainedValues<'a, T> {
    values: [Option<&'a T>; 5],
    next: usize,
}

impl<'a, T> RetainedValues<'a, T> {
    /// Creates the iterator from fixed request-local slots.
    pub const fn new(values: [Option<&'a T>; 5]) -> Self {
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
