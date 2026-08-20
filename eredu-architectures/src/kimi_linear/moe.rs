//! Kimi dense-prefix and grouped routed-expert feed-forward operators.

use eredu_nn::{
    Error, LinearOperator, LinearSpec, ParameterSpec, Parameterized, RoutedNeuralBackend,
    RoutingOperator, RoutingScoring, SwiGluExpertBankOperator, SwiGluExpertBankSpec,
    SwiGluExpertLayout, SwiGluExpertProjection, Tensor, TopKRouterSpec, TopKRoutingSpec,
};
use eredu_runtime::{ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest};

use crate::decoder::FeedForwardOperator;

use super::{FeedForwardPolicy, ModelArgs};

/// Checkpoint-compatible dense SwiGLU used by dense and shared paths.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseSwiGlu<B: RoutedNeuralBackend> {
    /// Gate projection.
    pub gate: B::Linear,
    /// Down projection.
    pub down: B::Linear,
    /// Up projection.
    pub up: B::Linear,
}

impl<B: RoutedNeuralBackend> DenseSwiGlu<B> {
    fn new(
        args: &ModelArgs,
        prefix: &str,
        intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
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
            gate: linear("gate_proj", args.hidden_size, intermediate)?,
            down: linear("down_proj", intermediate, args.hidden_size)?,
            up: linear("up_proj", args.hidden_size, intermediate)?,
        })
    }

    fn hidden(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(input, context)?;
        let up = self.up.forward(input, context)?;
        B::swiglu(gate, up, None, context)
    }

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        self.down.forward(&hidden, context)
    }
}

/// Grouped sigmoid router, routed bank, and one shared expert.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SparseMoe<B: RoutedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    /// Grouped sigmoid router with selection correction bias.
    pub router: B::Router,
    /// Packed routed SwiGLU experts.
    pub experts: B::SwiGluExpertBank,
    /// Always-executed shared SwiGLU expert.
    pub shared: DenseSwiGlu<B>,
}

impl<B: RoutedNeuralBackend> SparseMoe<B> {
    fn new(
        args: &ModelArgs,
        layer: usize,
        routed_intermediate: i32,
        shared_intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.layers.{layer}.mlp");
        let gate_name = format!("{prefix}.gate.weight");
        let mut routing = TopKRoutingSpec::new(
            args.num_experts,
            args.num_experts_per_token,
            RoutingScoring::Sigmoid,
            args.moe_renormalize,
        )?;
        if args.use_grouped_topk {
            routing = routing.with_groups(args.num_expert_group, args.topk_group)?;
        }
        routing = routing.with_weight_policy(1e-20, args.routed_scaling_factor)?;
        let router = B::top_k_router(
            TopKRouterSpec {
                input_dimensions: args.hidden_size,
                weight: ParameterSpec::trainable(&gate_name).map_err(Error::backend)?,
                correction_bias: Some(
                    ParameterSpec::trainable(format!("{prefix}.gate.e_score_correction_bias"))
                        .map_err(Error::backend)?,
                ),
                quantization: args.weight_quantization_for(&gate_name),
                routing,
            },
            context,
        )?;
        let experts_prefix = format!("{prefix}.experts");
        let gate_up_name = format!("{experts_prefix}.gate_up_proj");
        let down_name = format!("{experts_prefix}.down_proj");
        let experts = B::swiglu_expert_bank(
            SwiGluExpertBankSpec {
                expert_count: args.num_experts,
                input_dimensions: args.hidden_size,
                intermediate_dimensions: routed_intermediate,
                output_dimensions: args.hidden_size,
                limit: None,
                layout: SwiGluExpertLayout::Packed {
                    gate_up: SwiGluExpertProjection {
                        weight: ParameterSpec::trainable(&gate_up_name).map_err(Error::backend)?,
                        format: args.weight_quantization_for(&gate_up_name).into(),
                    },
                    down: SwiGluExpertProjection {
                        weight: ParameterSpec::trainable(&down_name).map_err(Error::backend)?,
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
            shared: DenseSwiGlu::new(
                args,
                &format!("{prefix}.shared_experts"),
                shared_intermediate,
                context,
            )?,
        })
    }
}

/// Per-layer dense-prefix or sparse Kimi feed-forward policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: RoutedNeuralBackend> {
    /// Dense prefix SwiGLU.
    Dense(DenseSwiGlu<B>),
    /// Routed plus shared experts.
    Sparse(SparseMoe<B>),
}

impl<B: RoutedNeuralBackend> FeedForward<B> {
    /// Builds one scheduled feed-forward operator at replicated geometry.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_geometry(
            args,
            layer,
            args.intermediate_size,
            args.moe_intermediate_size,
            args.moe_intermediate_size * args.num_shared_experts,
            context,
        )
    }

    /// Builds one scheduled operator from placement-resolved widths.
    pub fn new_with_geometry(
        args: &ModelArgs,
        layer: usize,
        dense_intermediate: i32,
        routed_intermediate: i32,
        shared_intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend(format!("Kimi Linear has no layer {layer}")))?;
        let prefix = format!("model.layers.{layer}.mlp");
        match policy.feed_forward {
            FeedForwardPolicy::Dense => {
                DenseSwiGlu::new(args, &prefix, dense_intermediate, context).map(Self::Dense)
            }
            FeedForwardPolicy::SparseMoe => SparseMoe::new(
                args,
                layer,
                routed_intermediate,
                shared_intermediate,
                context,
            )
            .map(Self::Sparse),
        }
    }

    /// Executes dense or provider-backed sparse computation.
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
            Self::Dense(dense) => dense.forward(input, context),
            Self::Sparse(sparse) => {
                let routes = sparse.router.route(input, context)?;
                let routed = provider
                    .forward_routed(
                        &mut sparse.experts,
                        RoutedExpertRequest {
                            layer: sparse.layer,
                            input,
                            routes: &routes,
                            pass,
                        },
                        context,
                    )
                    .map_err(|error| Error::backend(error.to_string()))?;
                routed.add(&sparse.shared.forward(input, context)?, context)
            }
        }
    }

    /// Executes tensor-partitioned dense/shared projections with provider-backed experts.
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
            Self::Sparse(sparse) => {
                let routes = sparse.router.route(input, context)?;
                let routed = provider
                    .forward_routed(
                        &mut sparse.experts,
                        RoutedExpertRequest {
                            layer: sparse.layer,
                            input,
                            routes: &routes,
                            pass,
                        },
                        context,
                    )
                    .map_err(|error| Error::backend(error.to_string()))?;
                let routed = if provider.output_is_tensor_parallel_partial() {
                    B::sum_parallel(routed, parallel, context)?
                } else {
                    routed
                };
                let shared_hidden = sparse.shared.hidden(input, context)?;
                let shared = B::row_parallel_linear(
                    &mut sparse.shared.down,
                    &shared_hidden,
                    parallel,
                    context,
                )?;
                routed.add(&shared, context)
            }
        }
    }

    /// Executes with stable activation and routing observations.
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
            Self::Dense(dense) => dense.forward(input, context),
            Self::Sparse(sparse) => {
                let routes = sparse.router.route(input, context)?;
                let routed = provider
                    .forward_routed(
                        &mut sparse.experts,
                        RoutedExpertRequest {
                            layer: sparse.layer,
                            input,
                            routes: &routes,
                            pass,
                        },
                        context,
                    )
                    .map_err(|error| Error::backend(error.to_string()))?;
                let shared = sparse.shared.forward(input, context)?;
                let combined = routed.add(&shared, context)?;
                observer.observe_routing(eredu_runtime::RoutingObservation {
                    path,
                    selected_experts: &routes.expert_ids,
                    selected_scores: &routes.selected_scores,
                    route_weights: &routes.route_weights,
                    routed_output: &routed,
                    local_routed_output: None,
                    reduced_routed_output: None,
                    shared_output: Some(&shared),
                    combined_output: Some(&combined),
                    expert_count,
                })?;
                Ok(combined)
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
        let mut provider = ResidentExpertProvider;
        self.forward_with_provider(
            input,
            if input.dim(input.shape().len() - 2) > 1 {
                eredu_runtime::ExpertPass::Prefill
            } else {
                eredu_runtime::ExpertPass::Decode
            },
            context,
            &mut provider,
        )
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
            Self::Sparse(sparse) => {
                let routes = sparse.router.route(input, context)?;
                let routed = sparse.experts.forward_routed(input, &routes, context)?;
                let routed = B::sum_parallel(routed, parallel, context)?;
                let shared_hidden = sparse.shared.hidden(input, context)?;
                let shared = B::row_parallel_linear(
                    &mut sparse.shared.down,
                    &shared_hidden,
                    parallel,
                    context,
                )?;
                routed.add(&shared, context)
            }
        }
    }
}
