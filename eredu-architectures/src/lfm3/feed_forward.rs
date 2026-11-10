//! Dense SwiGLU execution over the shared parameter-provider protocol.

use eredu_nn::{
    Error, GroupedNeuralBackend, LinearOperator, LinearSpec, NeuralBackend, ParameterSpec,
    Parameterized, Tensor,
};
use eredu_runtime::{ParameterProvider, TensorParallelParameterProvider};

use crate::decoder::{DecoderProjectionOperator, TensorParallelProjectionOperator};

use super::ModelArgs;

/// Dense LFM3 SwiGLU with checkpoint-compatible projection identities.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseSwiGlu<B: NeuralBackend> {
    /// Gate projection (`w1`).
    pub gate: B::Linear,
    /// Down projection (`w2`).
    pub down: B::Linear,
    /// Up projection (`w3`).
    pub up: B::Linear,
    #[parameter(skip)]
    policy: eredu_nn::GatedProductPolicy,
}

/// Declares one dense SwiGLU branch at global or rank-local intermediate width.
pub(crate) fn dense_parameters(
    args: &ModelArgs,
    root: &str,
    intermediate: i32,
) -> Result<eredu_nn::SplitGatedProductSpec, Error> {
    let prefix = format!("{root}.feed_forward");
    let linear = |field: &str, input, output| -> Result<LinearSpec, Error> {
        let name = format!("{prefix}.{field}.weight");
        Ok(LinearSpec {
            input,
            output,
            weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
            bias: None,
            format: crate::linear_format::standard_linear_format(
                &name,
                args.weight_quantization_for(&name).into(),
            )?,
        })
    };
    eredu_nn::SplitGatedProductSpec::new(
        linear("w1", args.hidden_size, intermediate)?,
        linear("w3", args.hidden_size, intermediate)?,
        linear("w2", intermediate, args.hidden_size)?,
        eredu_nn::GatedProductPolicy::default(),
    )
}

impl<B: NeuralBackend> DenseSwiGlu<B> {
    pub(crate) fn from_parameters(
        parameters: eredu_nn::SplitGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        eredu_nn::DeclareLogicalParameters::logical_parameters(&parameters)?;
        Ok(Self {
            gate: B::linear(parameters.gate, context)?,
            down: B::linear(parameters.down, context)?,
            up: B::linear(parameters.up, context)?,
            policy: parameters.policy,
        })
    }

    fn hidden(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(input, context)?;
        let up = self.up.forward(input, context)?;
        B::gated_product(gate, up, self.policy, context)
    }

    pub(crate) fn forward_instrumented(
        &mut self,
        input: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        let hidden = instrumentation.apply("feed_forward.units", hidden)?;
        instrumentation.project::<B>(
            "feed_forward.write_input",
            &mut self.down,
            &hidden,
            parallel,
            context,
        )
    }
}

/// Dense feed-forward branch.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Dense SwiGLU.
    Dense(DenseSwiGlu<B>),
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> FeedForward<B> {
    pub(crate) fn from_parameters(
        _layer: usize,
        parameters: super::declarations::FeedForwardParameters,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        match parameters {
            super::declarations::FeedForwardParameters::Dense(spec) => {
                DenseSwiGlu::from_parameters(spec, context).map(Self::Dense)
            }
        }
    }

    /// Executes the dense branch within a leased execution unit.
    pub fn forward_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        _pass: eredu_runtime::ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        _provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                dense.down.forward(&hidden, context)
            }
        }
    }

    /// Exposes the dense unit boundaries.
    pub fn forward_observed_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        _pass: eredu_runtime::ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        _provider: &mut P,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
        _points: Option<eredu_runtime::RoutedObservationPoints>,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => dense.forward_instrumented(input, None, context, instrumentation),
        }
    }

    /// Emits local dense components before the shared reduction.
    pub fn forward_parallel_observed_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        _pass: eredu_runtime::ExpertPass,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        _provider: &mut P,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
        _points: Option<eredu_runtime::RoutedObservationPoints>,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => {
                dense.forward_instrumented(input, Some(parallel), context, instrumentation)
            }
        }
    }

    /// Executes the tensor-partitioned dense branch.
    ///
    /// The down projection performs the selected row-parallel reduction.
    pub fn forward_parallel_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        _pass: eredu_runtime::ExpertPass,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        _provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                B::row_parallel_linear(&mut dense.down, &hidden, parallel, context)
            }
        }
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> DecoderProjectionOperator<B>
    for FeedForward<B>
{
    fn forward_feed_forward_observed(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Dense(dense) => dense.forward_instrumented(input, None, context, instrumentation),
        }
    }

    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                dense.down.forward(&hidden, context)
            }
        }
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    TensorParallelProjectionOperator<B> for FeedForward<B>
{
    fn forward_feed_forward_parallel_observed(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Dense(dense) => {
                dense.forward_instrumented(input, Some(parallel), context, instrumentation)
            }
        }
    }

    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                B::row_parallel_linear(&mut dense.down, &hidden, parallel, context)
            }
        }
    }
}
