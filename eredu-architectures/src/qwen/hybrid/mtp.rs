//! Graph-visible Qwen embedded multi-token prediction component.

use eredu_nn::{
    AttentionCache, Error, GroupedNeuralBackend, LinearSpec, NormalizationConstructionSpec,
    NormalizationOperator, NormalizationScale, ParameterSpec, Parameterized, Tensor,
};
use eredu_runtime::{ResidentExpertProvider, RoutedExpertProvider, RuntimeStateComponents};

use crate::decoder::ComponentInstrumentation;

use super::{Block, HybridConfig};
pub(crate) mod construction;
pub(crate) use construction::{PredictionSharedSpec, PredictionUnitSpec};

/// Concatenates decoder token identity for embedded prediction in request order.
pub fn prompt_token_identity<T: Tensor>(tokens: &[T], context: &T::Context) -> Result<T, Error> {
    if tokens.is_empty() {
        return Err(Error::backend(
            "Qwen hybrid embedded prediction requires token identity",
        ));
    }
    T::concatenate(tokens, 1, context)
}

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

/// Checkpoint-shared fusion and normalization, pinned once per prediction replica.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct PredictionShared<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Learned-offset normalization of the target hidden state.
    pub hidden_norm: B::Normalization,
    /// Learned-offset normalization of the next-token embedding.
    pub embedding_norm: B::Normalization,
    /// Dense fusion of normalized embedding and hidden state.
    pub fusion: B::Linear,
    /// Learned-offset final normalization before shared logits projection.
    pub final_norm: B::Normalization,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Clone for PredictionShared<B> {
    fn clone(&self) -> Self {
        Self {
            hidden_norm: self.hidden_norm.clone(),
            embedding_norm: self.embedding_norm.clone(),
            fusion: self.fusion.clone(),
            final_norm: self.final_norm.clone(),
        }
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> PredictionShared<B> {
    /// Builds the canonical shared modules used by every configured depth.
    pub fn new(config: &HybridConfig, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        PredictionSharedSpec::new(config.hidden_size, config.rms_norm_eps).instantiate::<B>(context)
    }
    pub(super) fn from_dimensions(
        hidden_size: i32,
        rms_norm_eps: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, NormalizationConstructionSpec, LinearSpec)>()?;
        let norm = |name: &str| {
            metadata.controls::<NormalizationConstructionSpec>()?;
            B::normalization(
                NormalizationConstructionSpec {
                    groups: None,
                    dimensions: hidden_size,
                    epsilon: rms_norm_eps,
                    scale: NormalizationScale::LearnedOffset {
                        weight: metadata.plain_parameter(name)?,
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
                    input: hidden_size * 2,
                    output: hidden_size,
                    weight: metadata.plain_parameter("mtp.fc.weight")?,
                    bias: None,
                    format: metadata.format("mtp.fc.weight", eredu_checkpoint::LinearFormat::Dense)?,
                },
                context,
            )?,
            final_norm: norm("mtp.norm.weight")?,
        })
    }
}

/// One prediction depth with independently streamable decoder parameters.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct PredictionUnit<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Full-attention decoder block for this prediction depth.
    pub block: Block<B>,
    #[parameter(skip, metadata)]
    experts: i32,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> PredictionUnit<B> {
    /// Builds one actual configured prediction depth.
    pub fn new(
        config: &HybridConfig,
        depth: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        super::block::construction::require_source_compiler::<B>(context)?;
        PredictionUnitSpec::new(config, depth)?.instantiate::<B>(context)
    }

    pub(crate) fn observation_points(
        &self,
        depth: usize,
    ) -> eredu_runtime::RoutedObservationPoints {
        eredu_runtime::RoutedObservationPoints::new(
            eredu_runtime::RoutedBankId::new(0),
            format!("mtp.layers.{depth}.mlp"),
            self.experts,
        )
    }

    /// Executes with resident experts.
    pub fn forward<S>(
        &mut self,
        shared: &mut PredictionShared<B>,
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
            shared,
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
        shared: &mut PredictionShared<B>,
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
        self.forward_with_decoder(
            shared,
            hidden,
            embedded,
            context,
            &mut ComponentInstrumentation::disabled(),
            |block, fused, context, _| {
                block.forward_with_provider(fused, mask, state, context, provider)
            },
        )
    }

    /// Executes the same fusion and decoder with causal component observations.
    /// The effective final normalized value also conditions the next depth.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_observed_with_provider<S, P, O>(
        &mut self,
        shared: &mut PredictionShared<B>,
        path: &str,
        point: eredu_runtime::RoutedObservationPoints,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_with_decoder(
            shared,
            hidden,
            embedded,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |block, fused, context, instrumentation| {
                block.forward_components_with_provider(
                    path,
                    point,
                    fused,
                    mask,
                    state,
                    context,
                    provider,
                    instrumentation.observer().expect("observed prediction"),
                )
            },
        )
    }

    /// Executes the local prediction projections with one reduction per
    /// row-parallel block output.
    pub fn forward_parallel<S, P>(
        &mut self,
        shared: &mut PredictionShared<B>,
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
        self.forward_with_decoder(
            shared,
            hidden,
            embedded,
            context,
            &mut ComponentInstrumentation::disabled(),
            |block, fused, context, _| {
                block.forward_parallel(fused, mask, state, parallel, context, provider)
            },
        )
    }

    /// Observes the same prediction fusion and tensor-parallel decoder while
    /// preserving its ordinary routed/shared reduction schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_observed_with_provider<S, P, O>(
        &mut self,
        shared: &mut PredictionShared<B>,
        path: &str,
        points: eredu_runtime::RoutedObservationPoints,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_with_decoder(
            shared,
            hidden,
            embedded,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |block, fused, context, instrumentation| {
                block.forward_components_parallel_with_provider(
                    path,
                    points,
                    fused,
                    mask,
                    state,
                    parallel,
                    context,
                    provider,
                    instrumentation.observer().expect("observed prediction"),
                )
            },
        )
    }

    fn forward_with_decoder<F>(
        &mut self,
        shared: &mut PredictionShared<B>,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        decoder: F,
    ) -> Result<B::Tensor, Error>
    where
        F: FnOnce(
            &mut Block<B>,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        instrumentation.observe("prediction.hidden", hidden)?;
        instrumentation.observe("prediction.embedding", embedded)?;
        let embedded = instrumentation.apply(
            "prediction.embedding.normalized",
            shared.embedding_norm.forward(embedded, context)?,
        )?;
        let hidden = instrumentation.apply(
            "prediction.hidden.normalized",
            shared.hidden_norm.forward(hidden, context)?,
        )?;
        let joined = B::Tensor::concatenate(&[embedded, hidden], -1, context)?;
        let fused = instrumentation.project::<B>(
            "prediction.fusion.input",
            &mut shared.fusion,
            &joined,
            None,
            context,
        )?;
        let fused = instrumentation.apply("prediction.fusion.output", fused)?;
        let predicted = decoder(&mut self.block, &fused, context, instrumentation)?;
        let predicted = instrumentation.apply("prediction.readout.residual", predicted)?;
        instrumentation.apply(
            "prediction.readout.normalized",
            shared.final_norm.forward(&predicted, context)?,
        )
    }
}
