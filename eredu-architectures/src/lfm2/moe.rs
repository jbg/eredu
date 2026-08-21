//! Dense and routed LFM2 feed-forward policies.

use eredu_nn::{
    Error, ExpertProjectionSpec, GatedProductExpertBankOperator, GatedProductExpertBankSpec,
    GatedProductExpertLayout, LinearOperator, LinearSpec, ParameterSpec, Parameterized,
    RoutedNeuralBackend, RoutingOperator, RoutingScoring, Tensor, TopKRouterSpec, TopKRoutingSpec,
};
use eredu_runtime::{ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest};

use crate::decoder::FeedForwardOperator;

use super::{FeedForwardPolicy, ModelArgs};

/// Dense LFM2 SwiGLU with checkpoint-compatible projection identities.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseSwiGlu<B: RoutedNeuralBackend> {
    /// Gate projection (`w1`).
    pub gate: B::Linear,
    /// Down projection (`w2`).
    pub down: B::Linear,
    /// Up projection (`w3`).
    pub up: B::Linear,
}

impl<B: RoutedNeuralBackend> DenseSwiGlu<B> {
    fn new(
        args: &ModelArgs,
        layer: usize,
        intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.layers.{layer}.feed_forward");
        let linear = |field: &str, input, output| {
            let name = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&name).map_err(Error::backend)?,
                    bias: None,
                    format: args.weight_quantization_for(&name).into(),
                },
                context,
            )
        };
        Ok(Self {
            gate: linear("w1", args.hidden_size, intermediate)?,
            down: linear("w2", intermediate, args.hidden_size)?,
            up: linear("w3", args.hidden_size, intermediate)?,
        })
    }

    fn hidden(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(input, context)?;
        let up = self.up.forward(input, context)?;
        B::gated_product(gate, up, eredu_nn::GatedProductPolicy::default(), context)
    }
}

/// Sigmoid router and routed gated-product expert bank used by LFM2-MoE.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct RoutedGatedProduct<B: RoutedNeuralBackend> {
    /// Global physical layer identity.
    #[parameter(skip)]
    pub layer: usize,
    /// Learned sigmoid top-k router.
    pub router: B::Router,
    /// Packed routed experts.
    pub experts: B::GatedProductExpertBank,
}

impl<B: RoutedNeuralBackend> RoutedGatedProduct<B> {
    fn new(
        args: &ModelArgs,
        layer: usize,
        intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.layers.{layer}.feed_forward");
        let gate_name = format!("{prefix}.gate.weight");
        let routing = TopKRoutingSpec::new(
            args.num_experts,
            args.num_experts_per_tok,
            RoutingScoring::Sigmoid,
            args.norm_topk_prob,
        )?
        .with_weight_policy(1e-6, args.routed_scaling_factor)?;
        let router = B::top_k_router(
            TopKRouterSpec {
                input_dimensions: args.hidden_size,
                weight: ParameterSpec::trainable(&gate_name).map_err(Error::backend)?,
                bias: None,
                correction_bias: args
                    .use_expert_bias
                    .then(|| ParameterSpec::trainable(format!("{prefix}.expert_bias")))
                    .transpose()
                    .map_err(Error::backend)?,
                input_transform: None,
                route_scale: None,
                quantization: args.weight_quantization_for(&gate_name),
                routing,
            },
            context,
        )?;
        let experts_prefix = format!("{prefix}.experts");
        let gate_up_name = format!("{experts_prefix}.gate_up_proj");
        let down_name = format!("{experts_prefix}.down_proj");
        let experts = B::gated_product_expert_bank(
            GatedProductExpertBankSpec {
                expert_count: args.num_experts,
                input_dimensions: args.hidden_size,
                intermediate_dimensions: intermediate,
                output_dimensions: args.hidden_size,
                policy: eredu_nn::GatedProductPolicy::ordinary_silu(),
                layout: GatedProductExpertLayout::Packed {
                    gate_up: ExpertProjectionSpec {
                        weight: ParameterSpec::trainable(&gate_up_name).map_err(Error::backend)?,
                        bias: None,
                        format: args.weight_quantization_for(&gate_up_name).into(),
                    },
                    down: ExpertProjectionSpec {
                        weight: ParameterSpec::trainable(&down_name).map_err(Error::backend)?,
                        bias: None,
                        format: args.weight_quantization_for(&down_name).into(),
                    },
                },
            },
            context,
        )?;
        Ok(Self {
            layer,
            router,
            experts,
        })
    }
}

/// Per-layer dense or routed feed-forward policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: RoutedNeuralBackend> {
    /// Dense SwiGLU.
    Dense(DenseSwiGlu<B>),
    /// Routed LFM2-MoE SwiGLU.
    Routed(RoutedGatedProduct<B>),
}

impl<B: RoutedNeuralBackend> FeedForward<B> {
    /// Builds the exact scheduled feed-forward operator.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_geometry(
            args,
            layer,
            args.dense_intermediate_size,
            args.moe_intermediate_size,
            context,
        )
    }

    /// Builds the scheduled operator from placement-resolved local widths.
    pub fn new_with_geometry(
        args: &ModelArgs,
        layer: usize,
        dense_intermediate: i32,
        expert_intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        match args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend(format!("LFM2 has no layer {layer}")))?
            .feed_forward
        {
            FeedForwardPolicy::Dense => {
                DenseSwiGlu::new(args, layer, dense_intermediate, context).map(Self::Dense)
            }
            FeedForwardPolicy::SparseMoe => {
                RoutedGatedProduct::new(args, layer, expert_intermediate, context).map(Self::Routed)
            }
        }
    }

    /// Executes dense locally or delegates routed experts to one runtime provider.
    pub fn forward_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                dense.down.forward(&hidden, context)
            }
            Self::Routed(routed) => {
                let routes = routed.router.route(input, context)?;
                RoutedExpertProvider::<B>::forward_routed(
                    provider,
                    &mut routed.experts,
                    RoutedExpertRequest {
                        layer: routed.layer,
                        input,
                        routes: &routes,
                        pass,
                    },
                    context,
                )
                .map_err(|error| Error::backend(error.to_string()))
            }
        }
    }

    /// Executes a tensor-partitioned dense path or delegates routed experts.
    ///
    /// Resident providers return rank-local contributions while external
    /// providers may return complete hidden-width contributions.
    pub fn forward_parallel_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                B::row_parallel_linear(&mut dense.down, &hidden, parallel, context)
            }
            Self::Routed(routed) => {
                let routes = routed.router.route(input, context)?;
                let output = RoutedExpertProvider::<B>::forward_routed_tensor_parallel(
                    provider,
                    &mut routed.experts,
                    RoutedExpertRequest {
                        layer: routed.layer,
                        input,
                        routes: &routes,
                        pass,
                    },
                    B::parallel_size(parallel),
                    context,
                )
                .map_err(|error| Error::backend(error.to_string()))?;
                eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(output, parallel, context)
            }
        }
    }

    /// Executes the selected policy and emits normalized routed-expert data.
    pub fn forward_observed<O>(
        &mut self,
        path: &str,
        expert_count: i32,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error>,
    {
        let mut provider = ResidentExpertProvider;
        self.forward_observed_with_provider(
            path,
            expert_count,
            input,
            pass,
            context,
            observer,
            &mut provider,
        )
    }

    /// Provider-backed execution with normalized route observation.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_observed_with_provider<O, P>(
        &mut self,
        path: &str,
        expert_count: i32,
        input: &B::Tensor,
        pass: eredu_runtime::ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => {
                let hidden = dense.hidden(input, context)?;
                dense.down.forward(&hidden, context)
            }
            Self::Routed(routed) => {
                let routes = routed.router.route(input, context)?;
                let output = provider
                    .forward_routed(
                        &mut routed.experts,
                        RoutedExpertRequest {
                            layer: routed.layer,
                            input,
                            routes: &routes,
                            pass,
                        },
                        context,
                    )
                    .map_err(|error| Error::backend(error.to_string()))?;
                observer.observe_routing(eredu_runtime::RoutingObservation {
                    path,
                    selected_experts: &routes.expert_ids,
                    selected_scores: &routes.selected_scores,
                    route_weights: &routes.route_weights,
                    routed_output: &output,
                    local_routed_output: None,
                    reduced_routed_output: None,
                    shared_output: None,
                    combined_output: None,
                    expert_count,
                })?;
                Ok(output)
            }
        }
    }
}

impl<B: RoutedNeuralBackend> FeedForwardOperator<B> for FeedForward<B> {
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
            Self::Routed(routed) => {
                let routes = routed.router.route(input, context)?;
                let mut provider = ResidentExpertProvider;
                RoutedExpertProvider::<B>::forward_routed(
                    &mut provider,
                    &mut routed.experts,
                    RoutedExpertRequest {
                        layer: routed.layer,
                        input,
                        routes: &routes,
                        pass: if input.shape()[input.shape().len() - 2] > 1 {
                            eredu_runtime::ExpertPass::Prefill
                        } else {
                            eredu_runtime::ExpertPass::Decode
                        },
                    },
                    context,
                )
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
            Self::Routed(routed) => {
                let routes = routed.router.route(input, context)?;
                let output = routed.experts.forward_routed_tensor_parallel(
                    input,
                    &routes,
                    B::parallel_size(parallel),
                    context,
                )?;
                eredu_runtime::reduce_tensor_parallel_expert_output::<B>(output, parallel, context)
            }
        }
    }
}
