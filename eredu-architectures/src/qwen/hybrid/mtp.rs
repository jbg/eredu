//! Graph-visible Qwen embedded multi-token prediction component.

use eredu_nn::{
    AttentionCache, Error, GroupedNeuralBackend, LinearOperator, LinearSpec,
    NormalizationConstructionSpec, NormalizationOperator, NormalizationScale, ParameterSpec,
    Parameterized, Tensor,
};
use eredu_runtime::{ResidentExpertProvider, RoutedExpertProvider, RuntimeStateComponents};

use super::{Block, HybridConfig};

/// Borrowed input selecting target execution or one prediction depth.
pub enum EmbeddedInput<'a, T> {
    /// Execute the target decoder.
    Target {
        /// Token ids shaped `[batch, sequence]`.
        tokens: &'a T,
        /// Optional caller-provided mask.
        mask: Option<&'a T>,
    },
    /// Execute one configured MTP prediction layer.
    Draft {
        /// Token ids embedded for the proposed positions.
        tokens: &'a T,
        /// Target or previous draft hidden state.
        hidden: &'a T,
        /// Zero-based configured prediction depth.
        depth: usize,
    },
}

impl<'a, T> EmbeddedInput<'a, T> {
    /// Creates target input.
    pub const fn target(tokens: &'a T, mask: Option<&'a T>) -> Self {
        Self::Target { tokens, mask }
    }

    /// Creates one prediction-depth input.
    pub const fn draft(tokens: &'a T, hidden: &'a T, depth: usize) -> Self {
        Self::Draft {
            tokens,
            hidden,
            depth,
        }
    }
}

/// Execution selection retained for a layered invocation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ForwardMode {
    /// Target decoder layers execute.
    Target,
    /// One MTP group executes.
    Draft(usize),
}

/// One configured prediction depth using the checkpoint-shared fusion policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct PredictionUnit<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Learned-offset normalization of the target hidden state.
    pub hidden_norm: B::Normalization,
    /// Learned-offset normalization of the next-token embedding.
    pub embedding_norm: B::Normalization,
    /// Dense fusion of normalized embedding and hidden state.
    pub fusion: B::Linear,
    /// Full-attention decoder block for this prediction depth.
    pub block: Block<B>,
    /// Learned-offset final normalization before shared logits projection.
    pub final_norm: B::Normalization,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> PredictionUnit<B> {
    /// Builds one actual configured prediction depth.
    pub fn new(
        config: &HybridConfig,
        depth: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let steps = usize::try_from(config.mtp_num_hidden_layers).map_err(Error::backend)?;
        if depth >= steps {
            return Err(Error::backend(format!(
                "Qwen hybrid MTP depth {depth} is outside {steps} configured layers"
            )));
        }
        let norm = |name: &str| {
            B::normalization(
                NormalizationConstructionSpec {
                    dimensions: config.hidden_size,
                    epsilon: config.rms_norm_eps,
                    scale: NormalizationScale::LearnedOffset {
                        weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                        offset: 1.0,
                    },
                },
                context,
            )
        };
        Ok(Self {
            hidden_norm: norm("mtp.pre_fc_norm_hidden.weight")?,
            embedding_norm: norm("mtp.pre_fc_norm_embedding.weight")?,
            fusion: B::linear(
                LinearSpec {
                    input: config.hidden_size * 2,
                    output: config.hidden_size,
                    weight: ParameterSpec::trainable("mtp.fc.weight").map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        "mtp.fc.weight",
                        eredu_checkpoint::LinearFormat::Dense,
                    )?,
                },
                context,
            )?,
            block: Block::new_mtp(config, depth, context)?,
            final_norm: norm("mtp.norm.weight")?,
        })
    }

    /// Executes with resident experts.
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
            &mut ResidentExpertProvider,
        )
    }

    /// Executes through the runtime-owned routed-expert provider.
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
        let embedded = self.embedding_norm.forward(embedded, context)?;
        let hidden = self.hidden_norm.forward(hidden, context)?;
        let joined = B::Tensor::concatenate(&[embedded, hidden], -1, context)?;
        let fused = self.fusion.forward(&joined, context)?;
        let predicted = self
            .block
            .forward_with_provider(&fused, mask, state, context, provider)?;
        self.final_norm.forward(&predicted, context)
    }

    /// Executes the local prediction projections with one reduction per
    /// row-parallel block output.
    pub fn forward_parallel<S, P>(
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
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let embedded = self.embedding_norm.forward(embedded, context)?;
        let hidden = self.hidden_norm.forward(hidden, context)?;
        let joined = B::Tensor::concatenate(&[embedded, hidden], -1, context)?;
        let fused = self.fusion.forward(&joined, context)?;
        let predicted = self
            .block
            .forward_parallel(&fused, mask, state, parallel, context, provider)?;
        self.final_norm.forward(&predicted, context)
    }
}
