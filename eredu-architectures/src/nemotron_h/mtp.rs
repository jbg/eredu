//! Graph-visible embedded multi-token prediction units.

use eredu_nn::{
    AttentionCache, Error, GroupedNeuralBackend, LinearSpec, NormalizationConstructionSpec,
    NormalizationOperator, ParameterSpec, Parameterized, Tensor,
};
use eredu_runtime::{RoutedExpertProvider, RuntimeStateComponents};

use crate::decoder::ComponentInstrumentation;

use super::{Block, LayerGeometry, LayerPolicy, ModelArgs};
mod construction;
pub(crate) use construction::PredictionUnitSpec;

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
pub struct PredictionUnit<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
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
    #[parameter(skip, metadata)]
    experts: i32,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> PredictionUnit<B> {
    /// Builds one physical unit at a prediction-depth-relative position.
    pub fn new(
        args: &ModelArgs,
        depth: usize,
        relative: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        PredictionUnitSpec::new(args, depth, relative)?.instantiate::<B>(context)
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
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        PredictionUnitSpec::with_geometry(args, depth, relative, policy, geometry)?
            .instantiate::<B>(context)
    }

    /// Admitted global expert count for routed observation geometry.
    pub(crate) fn expert_count(&self) -> i32 {
        self.experts
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
        self.forward_with_decoder(
            hidden,
            embedded,
            context,
            &mut ComponentInstrumentation::disabled(),
            |block, fused, context, _| {
                block.forward_with_provider(fused, mask, state, context, provider)
            },
        )
    }

    /// Observes the actual fusion, scheduled operator and final normalization.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_observed_with_provider<S, P, O>(
        &mut self,
        path: &str,
        expert_count: i32,
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
            hidden,
            embedded,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |block, fused, context, instrumentation| {
                block.forward_observed_with_provider(
                    path,
                    expert_count,
                    fused,
                    mask,
                    state,
                    context,
                    instrumentation.observer().expect("observed prediction"),
                    provider,
                )
            },
        )
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
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_with_decoder(
            hidden,
            embedded,
            context,
            &mut ComponentInstrumentation::disabled(),
            |block, fused, context, _| {
                block.forward_parallel(fused, mask, state, parallel, context, provider)
            },
        )
    }

    /// Observes the same prediction graph with the ordinary TP reduction order.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_observed_with_provider<S, P, O>(
        &mut self,
        path: &str,
        expert_count: i32,
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
            hidden,
            embedded,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |block, fused, context, instrumentation| {
                block.forward_parallel_observed_with_provider(
                    path,
                    expert_count,
                    fused,
                    mask,
                    state,
                    parallel,
                    context,
                    instrumentation.observer().expect("observed prediction"),
                    provider,
                )
            },
        )
    }

    fn forward_with_decoder<F>(
        &mut self,
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
        let hidden = match (
            &mut self.embedding_norm,
            &mut self.hidden_norm,
            &mut self.fusion,
        ) {
            (Some(embedding_norm), Some(hidden_norm), Some(fusion)) => {
                instrumentation.observe("prediction.hidden", hidden)?;
                instrumentation.observe("prediction.embedding", embedded)?;
                let embedded = instrumentation.apply(
                    "prediction.embedding.normalized",
                    embedding_norm.forward(embedded, context)?,
                )?;
                let hidden = instrumentation.apply(
                    "prediction.hidden.normalized",
                    hidden_norm.forward(hidden, context)?,
                )?;
                let joined = B::Tensor::concatenate(&[embedded, hidden], -1, context)?;
                let fused = instrumentation.project::<B>(
                    "prediction.fusion.input",
                    fusion,
                    &joined,
                    None,
                    context,
                )?;
                instrumentation.apply("prediction.fusion.output", fused)?
            }
            (None, None, None) => hidden.clone(),
            _ => return Err(Error::backend("incomplete Nemotron-H MTP fusion unit")),
        };
        let hidden = decoder(&mut self.block, &hidden, context, instrumentation)?;
        match &mut self.final_norm {
            Some(norm) => {
                let hidden = instrumentation.apply("prediction.readout.residual", hidden)?;
                instrumentation.apply(
                    "prediction.readout.normalized",
                    norm.forward(&hidden, context)?,
                )
            }
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
