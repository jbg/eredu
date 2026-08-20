//! Nemotron-H dense and routed ReLU-squared feed-forward operators.

use eredu_nn::{
    Error, LinearOperator, LinearSpec, ParameterSpec, Parameterized, Relu2ExpertBankSpec,
    RoutedNeuralBackend, RoutingOperator, RoutingScoring, SwiGluExpertProjection, Tensor,
    TopKRouterSpec, TopKRoutingSpec,
};
use eredu_runtime::{ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest};

use super::ModelArgs;

/// Dense up/ReLU²/down projection pair.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseMlp<B: RoutedNeuralBackend> {
    /// Up projection.
    pub up_proj: B::Linear,
    /// Down projection.
    pub down_proj: B::Linear,
}

impl<B: RoutedNeuralBackend> DenseMlp<B> {
    /// Builds one unloaded dense MLP.
    pub fn new(
        args: &ModelArgs,
        prefix: &str,
        intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let linear = |field: &str, input, output| {
            let weight = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: args
                        .mlp_bias
                        .then(|| ParameterSpec::trainable(format!("{prefix}.{field}.bias")))
                        .transpose()
                        .map_err(Error::backend)?,
                    format: args.weight_quantization_for(&weight).into(),
                },
                context,
            )
        };
        Ok(Self {
            up_proj: linear("up_proj", args.hidden_size, intermediate)?,
            down_proj: linear("down_proj", intermediate, args.hidden_size)?,
        })
    }

    fn hidden(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.up_proj
            .forward(input, context)?
            .maximum_scalar(0.0, context)?
            .square(context)
    }

    /// Executes replicated dense computation.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        self.down_proj.forward(&hidden, context)
    }

    /// Executes a row-parallel down projection.
    pub fn forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        B::row_parallel_linear(&mut self.down_proj, &hidden, parallel, context)
    }
}

/// Grouped sigmoid routing, packed ReLU² experts, and one shared expert.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SparseMoe<B: RoutedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    /// Grouped correction-bias router.
    pub gate: B::Router,
    /// Packed routed experts.
    pub experts: B::Relu2ExpertBank,
    /// Always-executed shared expert.
    pub shared_experts: DenseMlp<B>,
}

impl<B: RoutedNeuralBackend> SparseMoe<B> {
    /// Builds one unloaded sparse unit at placement-resolved widths.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        routed_intermediate: i32,
        shared_intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_at(
            args,
            layer,
            &format!("model.layers.{layer}.moe"),
            routed_intermediate,
            shared_intermediate,
            context,
        )
    }

    /// Builds one sparse unit at an explicit target or MTP parameter path.
    pub fn new_at(
        args: &ModelArgs,
        layer: usize,
        prefix: &str,
        routed_intermediate: i32,
        shared_intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let gate_weight = format!("{prefix}.gate.weight");
        let routing = TopKRoutingSpec::new(
            args.n_routed_experts,
            args.num_experts_per_tok,
            RoutingScoring::Sigmoid,
            args.norm_topk_prob,
        )?
        .with_groups(args.n_group, args.topk_group)?
        .with_weight_policy(1e-20, args.routed_scaling_factor)?;
        let gate = B::top_k_router(
            TopKRouterSpec {
                input_dimensions: args.hidden_size,
                weight: ParameterSpec::trainable(&gate_weight).map_err(Error::backend)?,
                correction_bias: Some(
                    ParameterSpec::trainable(format!("{prefix}.gate.e_score_correction_bias"))
                        .map_err(Error::backend)?,
                ),
                input_transform: None,
                route_scale: None,
                quantization: args.weight_quantization_for(&gate_weight),
                routing,
            },
            context,
        )?;
        let up_name = format!("{prefix}.experts.up_proj");
        let down_name = format!("{prefix}.experts.down_proj");
        let experts = B::relu2_expert_bank(
            Relu2ExpertBankSpec {
                expert_count: args.n_routed_experts,
                hidden_dimensions: args.hidden_size,
                intermediate_dimensions: routed_intermediate,
                up: SwiGluExpertProjection {
                    weight: ParameterSpec::trainable(&up_name).map_err(Error::backend)?,
                    format: args.weight_quantization_for(&up_name).into(),
                },
                down: SwiGluExpertProjection {
                    weight: ParameterSpec::trainable(&down_name).map_err(Error::backend)?,
                    format: args.weight_quantization_for(&down_name).into(),
                },
            },
            context,
        )?;
        Ok(Self {
            layer,
            gate,
            experts,
            shared_experts: DenseMlp::new(
                args,
                &format!("{prefix}.shared_experts"),
                shared_intermediate,
                context,
            )?,
        })
    }

    /// Executes resident routed and shared experts.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_with_provider(
            input,
            eredu_runtime::ExpertPass::Prefill,
            context,
            &mut ResidentExpertProvider,
        )
    }

    /// Executes routed experts through one runtime-owned provider.
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
        let routes = self.gate.route(input, context)?;
        let routed = provider
            .forward_relu2_routed(
                &mut self.experts,
                RoutedExpertRequest {
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass,
                },
                context,
            )
            .map_err(|error| Error::backend(error.to_string()))?;
        routed.add(&self.shared_experts.forward(input, context)?, context)
    }

    /// Executes routed and shared experts while emitting one normalized routing observation.
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
        let routes = self.gate.route(input, context)?;
        let routed = provider
            .forward_relu2_routed(
                &mut self.experts,
                RoutedExpertRequest {
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass,
                },
                context,
            )
            .map_err(|error| Error::backend(error.to_string()))?;
        let shared = self.shared_experts.forward(input, context)?;
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

    /// Executes TP shared projections while the provider owns routed experts.
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
        let routes = self.gate.route(input, context)?;
        let routed = provider
            .forward_relu2_routed(
                &mut self.experts,
                RoutedExpertRequest {
                    layer: self.layer,
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
        routed.add(
            &self
                .shared_experts
                .forward_parallel(input, parallel, context)?,
            context,
        )
    }
}
