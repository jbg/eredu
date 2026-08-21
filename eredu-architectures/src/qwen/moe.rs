//! Qwen dense and routed gated-product feed-forward policy.

use eredu_nn::{
    Error, ExpertProjectionSpec, GatedProductExpertBankSpec, GatedProductExpertLayout,
    ParameterSpec, Parameterized, RoutedNeuralBackend, RoutingOperator, RoutingScoring, Tensor,
    TopKRouterSpec, TopKRoutingSpec,
};
use eredu_runtime::{
    ExpertPass, ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest,
};

use crate::decoder::{FeedForwardOperator, Mlp};

use super::ModelArgs;

fn inferred_expert_pass<T: Tensor>(input: &T) -> ExpertPass {
    let sequence = input
        .shape()
        .get(input.shape().len().saturating_sub(2))
        .copied()
        .unwrap_or(1);
    if sequence > 1 {
        ExpertPass::Prefill
    } else {
        ExpertPass::Decode
    }
}

/// Qwen3 top-k router and packed expert bank.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct RoutedGatedProduct<B: RoutedNeuralBackend> {
    /// Global decoder layer used for runtime expert identity.
    #[parameter(skip)]
    pub layer: usize,
    /// Learned top-k router.
    pub router: B::Router,
    /// Packed or provider-materialized expert bank.
    pub experts: B::GatedProductExpertBank,
}

impl<B: RoutedNeuralBackend> RoutedGatedProduct<B> {
    /// Builds one unloaded Qwen3-MoE feed-forward block.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if !args.is_moe() {
            return Err(Error::backend("routed Qwen block requires Qwen3-MoE args"));
        }
        let prefix = format!("{}.layers.{layer}.mlp", args.parameter_root);
        let routing = TopKRoutingSpec::new(
            args.num_experts,
            args.num_experts_per_tok,
            RoutingScoring::Softmax,
            args.norm_topk_prob,
        )?;
        let router_name = format!("{prefix}.gate.weight");
        let router = B::top_k_router(
            TopKRouterSpec {
                input_dimensions: args.hidden_size,
                weight: ParameterSpec::trainable(&router_name).map_err(Error::backend)?,
                bias: None,
                correction_bias: None,
                input_transform: None,
                route_scale: None,
                quantization: args.weight_quantization_for(&router_name),
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
                intermediate_dimensions: args.moe_intermediate_size,
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

impl<B: RoutedNeuralBackend> FeedForwardOperator<B> for RoutedGatedProduct<B> {
    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let routes = self.router.route(input, context)?;
        let mut provider = ResidentExpertProvider;
        RoutedExpertProvider::<B>::forward_routed(
            &mut provider,
            &mut self.experts,
            RoutedExpertRequest {
                layer: self.layer,
                input,
                routes: &routes,
                pass: inferred_expert_pass(input),
            },
            context,
        )
    }

    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let routes = self.router.route(input, context)?;
        let mut provider = ResidentExpertProvider;
        let output = RoutedExpertProvider::<B>::forward_routed_tensor_parallel(
            &mut provider,
            &mut self.experts,
            RoutedExpertRequest {
                layer: self.layer,
                input,
                routes: &routes,
                pass: inferred_expert_pass(input),
            },
            B::parallel_size(parallel),
            context,
        )?;
        eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(output, parallel, context)
    }
}

/// One Qwen feed-forward policy used by the single decoder block type.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: RoutedNeuralBackend> {
    /// Dense SwiGLU used by Qwen2 and dense Qwen3.
    Dense(Mlp<B>),
    /// Top-k routed gated-product used by Qwen3-MoE.
    Routed(RoutedGatedProduct<B>),
}

impl<B: RoutedNeuralBackend> FeedForward<B> {
    /// Builds the validated dense or routed policy for one layer.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if args.is_moe() {
            RoutedGatedProduct::new(args, layer, context).map(Self::Routed)
        } else {
            Mlp::new(args, layer, context).map(Self::Dense)
        }
    }

    /// Executes the selected feed-forward policy and emits normalized routing
    /// data for routed blocks without changing the ordinary hot path.
    pub fn forward_observed<O>(
        &mut self,
        path: &str,
        expert_count: i32,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error>,
    {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward(input, context),
            Self::Routed(moe) => {
                let routes = moe.router.route(input, context)?;
                let mut provider = ResidentExpertProvider;
                let output = RoutedExpertProvider::<B>::forward_routed(
                    &mut provider,
                    &mut moe.experts,
                    RoutedExpertRequest {
                        layer: moe.layer,
                        input,
                        routes: &routes,
                        pass: inferred_expert_pass(input),
                    },
                    context,
                )?;
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

    /// Executes through a runtime-owned routed-expert provider.
    ///
    /// Dense blocks retain their ordinary static path. Routed blocks submit
    /// one typed route batch without exposing backend storage or residency to
    /// architecture code.
    pub fn forward_with_provider<P>(
        &mut self,
        layer: usize,
        pass: ExpertPass,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward(input, context),
            Self::Routed(moe) => {
                let routes = moe.router.route(input, context)?;
                provider
                    .forward_routed(
                        &mut moe.experts,
                        RoutedExpertRequest {
                            layer,
                            input,
                            routes: &routes,
                            pass,
                        },
                        context,
                    )
                    .map_err(Error::backend)
            }
        }
    }

    /// Provider-backed execution with normalized route observation.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_observed_with_provider<O, P>(
        &mut self,
        path: &str,
        layer: usize,
        pass: ExpertPass,
        expert_count: i32,
        input: &B::Tensor,
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
            Self::Dense(mlp) => mlp.forward_feed_forward(input, context),
            Self::Routed(moe) => {
                let routes = moe.router.route(input, context)?;
                let output = provider
                    .forward_routed(
                        &mut moe.experts,
                        RoutedExpertRequest {
                            layer,
                            input,
                            routes: &routes,
                            pass,
                        },
                        context,
                    )
                    .map_err(Error::backend)?;
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

    /// Executes the selected feed-forward policy with rank-local parameters
    /// and exactly one row reduction for tensor-parallel output.
    pub fn forward_with_provider_parallel<P>(
        &mut self,
        layer: usize,
        pass: ExpertPass,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward_parallel(input, parallel, context),
            Self::Routed(moe) => {
                let routes = moe.router.route(input, context)?;
                let output = provider
                    .forward_routed_tensor_parallel(
                        &mut moe.experts,
                        RoutedExpertRequest {
                            layer,
                            input,
                            routes: &routes,
                            pass,
                        },
                        B::parallel_size(parallel),
                        context,
                    )
                    .map_err(Error::backend)?;
                eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(output, parallel, context)
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
            Self::Dense(mlp) => mlp.forward_feed_forward(input, context),
            Self::Routed(moe) => moe.forward_feed_forward(input, context),
        }
    }

    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward_parallel(input, parallel, context),
            Self::Routed(moe) => moe.forward_feed_forward_parallel(input, parallel, context),
        }
    }
}
