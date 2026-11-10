//! Exact Nemotron-H physical-unit normalization and residual order.

use eredu_nn::{
    AttentionCache, Error, GroupedNeuralBackend, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, ParameterSpec, Parameterized, Tensor,
};
use eredu_runtime::{ParameterProvider, ResidentExpertProvider, RuntimeStateComponents};

use crate::decoder::{Attention, AttentionInput};

use super::{DenseMlp, LayerPolicy, Mamba2, ModelArgs, SparseMoe};

pub(crate) fn normalization_spec(
    args: &ModelArgs,
    parameter: &str,
) -> Result<NormalizationConstructionSpec, Error> {
    Ok(NormalizationConstructionSpec::learned(
        args.hidden_size,
        args.layer_norm_epsilon,
        ParameterSpec::trainable(parameter).map_err(Error::backend)?,
    ))
}

/// One scheduled physical operator.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Operator<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Mamba2 state-space operator.
    Mamba(Mamba2<B>),
    /// No-positional grouped-query attention.
    Attention(Attention<B>),
    /// Dense ReLU² MLP.
    Dense(DenseMlp<B>),
    /// Routed plus shared ReLU² experts.
    Sparse(SparseMoe<B>),
}

/// One pre-normalized residual Nemotron-H physical unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Block<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Scheduled operator.
    pub operator: Operator<B>,
    /// Unit pre-normalization.
    pub norm: B::Normalization,
    #[parameter(skip)]
    residual_in_fp32: bool,
}

/// Non-routed Nemotron-H physical operator admitted by replicated text composition.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(crate) enum ReplicatedOperator<B: NeuralBackend> {
    Mamba(Mamba2<B>),
    Attention(Attention<B>),
    Dense(DenseMlp<B>),
}

/// Dense-only Nemotron-H unit used by replicated text composition.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(crate) struct ReplicatedBlock<B: NeuralBackend> {
    operator: ReplicatedOperator<B>,
    norm: B::Normalization,
    #[parameter(skip)]
    residual_in_fp32: bool,
}

impl<B: NeuralBackend> ReplicatedBlock<B> {
    pub(crate) fn from_parameters(
        args: &ModelArgs,
        layer: usize,
        parameters: super::declarations::TargetBlockParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        use super::declarations::TargetOperatorParameters as P;
        let operator = match parameters.operator {
            P::Mamba(spec) => ReplicatedOperator::Mamba(Mamba2::from_spec(spec, context)?),
            P::Attention(spec) => {
                let Some(LayerPolicy::SelfAttention(policy)) = args.layer_schedule.get(layer)
                else {
                    return Err(Error::backend(
                        "attention scope differs from target schedule",
                    ));
                };
                ReplicatedOperator::Attention(spec.construct::<B>(*policy, context)?)
            }
            P::Dense(spec) => ReplicatedOperator::Dense(spec.construct::<B>(context)?),
            P::Sparse(_) => {
                return Err(Error::backend(
                    "replicated Nemotron-H unit rejects routed computation",
                ))
            }
        };
        Ok(Self {
            operator,
            norm: B::normalization(parameters.norm.with_groups(1)?, context)?,
            residual_in_fp32: args.residual_in_fp32,
        })
    }

    pub(crate) fn forward<S>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.forward_instrumented(
            hidden,
            mask,
            state,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    pub(crate) fn forward_instrumented<S>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let boundary = match &self.operator {
            ReplicatedOperator::Attention(_) => Some("attention"),
            ReplicatedOperator::Dense(_) => Some("feed_forward"),
            ReplicatedOperator::Mamba(_) => Some("mixer"),
        };
        let normalized = self.norm.forward(hidden, context)?;
        let normalized = match boundary {
            Some("attention") => instrumentation.apply("attention.input", normalized)?,
            Some("mixer") => instrumentation.apply("mixer.input", normalized)?,
            Some(_) => instrumentation.apply("feed_forward.input", normalized)?,
            None => normalized,
        };
        let output = match &mut self.operator {
            ReplicatedOperator::Mamba(mamba) => mamba.forward(&normalized, state, context)?,
            ReplicatedOperator::Attention(attention) => attention.forward_instrumented(
                AttentionInput {
                    selected_positions: None,
                    hidden: &normalized,
                    mask,
                    cache: Some(state),
                    allow_sliding_prefill: true,
                    rotary_position: None,
                },
                None,
                context,
                instrumentation,
            )?,
            ReplicatedOperator::Dense(mlp) => {
                mlp.forward_instrumented(&normalized, None, context, instrumentation)?
            }
        };
        finish_component_residual::<B>(
            hidden,
            output,
            boundary,
            self.residual_in_fp32,
            context,
            instrumentation,
        )
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Block<B> {
    pub(crate) fn from_parameters(
        args: &ModelArgs,
        layer: usize,
        parameters: super::declarations::TargetBlockParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = *args
            .layer_schedule
            .get(layer)
            .ok_or_else(|| Error::backend("target scope is outside its schedule"))?;
        let mut parameters = parameters;
        parameters.norm = parameters.norm.with_groups(1)?;
        Self::from_scope(args, layer, policy, parameters, context)
    }

    pub(crate) fn from_scope(
        args: &ModelArgs,
        identity: usize,
        policy: LayerPolicy,
        parameters: super::declarations::TargetBlockParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        use super::declarations::TargetOperatorParameters as P;
        let operator = match parameters.operator {
            P::Mamba(spec) => Operator::Mamba(Mamba2::from_spec(spec, context)?),
            P::Attention(spec) => {
                let LayerPolicy::SelfAttention(attention) = policy else {
                    return Err(Error::backend(
                        "attention scope differs from invocation policy",
                    ));
                };
                Operator::Attention(spec.construct::<B>(attention, context)?)
            }
            P::Dense(spec) => Operator::Dense(spec.construct::<B>(context)?),
            P::Sparse(spec) => Operator::Sparse(SparseMoe::from_spec(identity, spec, context)?),
        };
        Ok(Self {
            operator,
            norm: B::normalization(parameters.norm, context)?,
            residual_in_fp32: args.residual_in_fp32,
        })
    }

    /// Executes one physical unit with resident experts.
    pub fn forward<S>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.forward_with_provider(hidden, mask, state, context, &mut ResidentExpertProvider)
    }

    /// Executes one physical unit through a runtime expert provider.
    pub fn forward_with_provider<S, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_instrumented_with_provider(
            "",
            0,
            hidden,
            mask,
            state,
            context,
            provider,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    /// Executes one physical unit and reports sparse routing through the neutral observer.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_observed_with_provider<S, O, P>(
        &mut self,
        path: &str,
        expert_count: i32,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_instrumented_with_provider(
            path,
            expert_count,
            hidden,
            mask,
            state,
            context,
            provider,
            &mut crate::decoder::ComponentInstrumentation::new(path, &mut borrowed),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_instrumented_with_provider<S, P>(
        &mut self,
        path: &str,
        expert_count: i32,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let boundary = match &self.operator {
            Operator::Attention(_) => Some("attention"),
            Operator::Dense(_) | Operator::Sparse(_) => Some("feed_forward"),
            Operator::Mamba(_) => Some("mixer"),
        };
        let normalized = self.norm.forward(hidden, context)?;
        let normalized = match boundary {
            Some("attention") => instrumentation.apply("attention.input", normalized)?,
            Some("mixer") => instrumentation.apply("mixer.input", normalized)?,
            Some(_) => instrumentation.apply("feed_forward.input", normalized)?,
            None => normalized,
        };
        let pass = if hidden.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let output = match &mut self.operator {
            Operator::Mamba(mamba) => mamba.forward(&normalized, state, context)?,
            Operator::Attention(attention) => attention.forward_instrumented(
                AttentionInput {
                    selected_positions: None,
                    hidden: &normalized,
                    mask,
                    cache: Some(state),
                    allow_sliding_prefill: true,
                    rotary_position: None,
                },
                None,
                context,
                instrumentation,
            )?,
            Operator::Dense(mlp) => {
                mlp.forward_instrumented(&normalized, None, context, instrumentation)?
            }
            Operator::Sparse(moe) => match instrumentation.observer() {
                Some(observer) => moe.forward_observed_with_provider(
                    &format!("{path}.routing"),
                    expert_count,
                    &normalized,
                    pass,
                    context,
                    observer,
                    provider,
                )?,
                None => moe.forward_with_provider(&normalized, pass, context, provider)?,
            },
        };
        finish_component_residual::<B>(
            hidden,
            output,
            boundary,
            self.residual_in_fp32,
            context,
            instrumentation,
        )
    }

    /// Emits the actual component and routed-unit boundaries of a TP unit.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_observed_with_provider<S, O, P>(
        &mut self,
        path: &str,
        expert_count: i32,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_parallel_instrumented_with_provider(
            path,
            expert_count,
            hidden,
            mask,
            state,
            parallel,
            context,
            provider,
            &mut crate::decoder::ComponentInstrumentation::new(path, &mut borrowed),
        )
    }

    /// Executes one placement-resolved unit with tensor collectives.
    pub fn forward_parallel<S, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_parallel_instrumented_with_provider(
            "",
            0,
            hidden,
            mask,
            state,
            parallel,
            context,
            provider,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_parallel_instrumented_with_provider<S, P>(
        &mut self,
        path: &str,
        expert_count: i32,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let boundary = match &self.operator {
            Operator::Attention(_) => Some("attention"),
            Operator::Dense(_) | Operator::Sparse(_) => Some("feed_forward"),
            Operator::Mamba(_) => Some("mixer"),
        };
        let normalized = self.norm.forward(hidden, context)?;
        let normalized = match boundary {
            Some("attention") => instrumentation.apply("attention.input", normalized)?,
            Some("mixer") => instrumentation.apply("mixer.input", normalized)?,
            Some(_) => instrumentation.apply("feed_forward.input", normalized)?,
            None => normalized,
        };
        let pass = if hidden.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let output = match &mut self.operator {
            Operator::Mamba(mamba) => {
                mamba.forward_parallel(&normalized, state, parallel, context)?
            }
            Operator::Attention(attention) => attention.forward_instrumented(
                AttentionInput {
                    selected_positions: None,
                    hidden: &normalized,
                    mask,
                    cache: Some(state),
                    allow_sliding_prefill: true,
                    rotary_position: None,
                },
                Some(parallel),
                context,
                instrumentation,
            )?,
            Operator::Dense(mlp) => {
                mlp.forward_instrumented(&normalized, Some(parallel), context, instrumentation)?
            }
            Operator::Sparse(moe) => match instrumentation.observer() {
                Some(observer) => moe.forward_parallel_observed_with_provider(
                    &format!("{path}.routing"),
                    expert_count,
                    &normalized,
                    pass,
                    parallel,
                    context,
                    observer,
                    provider,
                )?,
                None => moe.forward_parallel_with_provider(
                    &normalized,
                    pass,
                    parallel,
                    context,
                    provider,
                )?,
            },
        };
        finish_component_residual::<B>(
            hidden,
            output,
            boundary,
            self.residual_in_fp32,
            context,
            instrumentation,
        )
    }
}

fn finish_component_residual<B: NeuralBackend>(
    hidden: &B::Tensor,
    output: B::Tensor,
    boundary: Option<&str>,
    residual_in_fp32: bool,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error> {
    let output = match boundary {
        Some("attention") => {
            let output = instrumentation.apply("attention.write", output)?;
            instrumentation.apply("attention.output", output)?
        }
        Some("mixer") => {
            let output = instrumentation.apply("mixer.write", output)?;
            instrumentation.apply("mixer.output", output)?
        }
        Some(_) => {
            let output = instrumentation.apply("feed_forward.write", output)?;
            instrumentation.apply("feed_forward.output", output)?
        }
        None => output,
    };
    let residual = B::add_residual(hidden, &output, residual_in_fp32, context)?;
    match boundary {
        Some("attention") => instrumentation.apply("attention.residual", residual),
        Some("mixer") => instrumentation.apply("mixer.residual", residual),
        Some(_) => instrumentation.apply("feed_forward.residual", residual),
        None => Ok(residual),
    }
}
