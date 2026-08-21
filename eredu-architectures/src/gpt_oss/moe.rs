//! Backend-neutral GPT-OSS biased routing and bounded gated-product experts.

use eredu_checkpoint::WeightQuantization;
use eredu_nn::{
    Error, ExpertProjectionSpec, GatedProductExpertBankSpec, GatedProductExpertLayout,
    ParameterSpec, Parameterized, RoutedNeuralBackend, RoutingOperator, RoutingScoring, Tensor,
    TopKRouterSpec, TopKRoutingSpec,
};
use eredu_runtime::{
    ExpertPass, ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest,
};

use crate::decoder::FeedForwardOperator;

use super::config::ModelArgs;

#[cfg(test)]
fn expert_policy(limit: f32) -> Result<eredu_nn::GatedProductPolicy, Error> {
    eredu_nn::GatedProductPolicy::new(
        eredu_nn::GatedProductActivation::Silu,
        Some(limit),
        Some(limit),
        1.702,
        1.0,
    )
}

fn inferred_pass<T: Tensor>(input: &T) -> ExpertPass {
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

/// GPT-OSS SelectedSoftmax router and canonical biased MXFP4 expert bank.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct RoutedMlp<B: RoutedNeuralBackend> {
    /// Global layer identity used by runtime expert providers.
    #[parameter(skip)]
    pub layer: usize,
    /// Learned biased router.
    pub router: B::Router,
    /// Canonical component-major expert bank.
    pub experts: B::GatedProductExpertBank,
}

impl<B: RoutedNeuralBackend> RoutedMlp<B> {
    /// Builds one unloaded GPT-OSS routed feed-forward operator.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("{}.layers.{layer}.mlp", args.parameter_root);
        let router_weight = format!("{prefix}.router.weight");
        let router_bias = format!("{prefix}.router.bias");
        let routing = TopKRoutingSpec::new(
            args.num_local_experts,
            args.num_experts_per_tok,
            RoutingScoring::SelectedSoftmax,
            false,
        )?;
        let router = B::top_k_router(
            TopKRouterSpec {
                input_dimensions: args.hidden_size,
                weight: ParameterSpec::trainable(&router_weight).map_err(Error::backend)?,
                bias: Some(ParameterSpec::trainable(&router_bias).map_err(Error::backend)?),
                correction_bias: None,
                input_transform: None,
                route_scale: None,
                // Published routers remain dense unless the checkpoint itself
                // carries an exact native encoding for this identity.
                quantization: args.checkpoint_weight_quantization_for(&router_weight),
                routing,
            },
            context,
        )?;

        let expert_prefix = format!("{prefix}.experts");
        let gate_up_weight = format!("{expert_prefix}.gate_up_proj");
        let gate_up_bias = format!("{expert_prefix}.gate_up_proj_bias");
        let down_weight = format!("{expert_prefix}.down_proj");
        let down_bias = format!("{expert_prefix}.down_proj_bias");
        let policy = args.gated_product_policy;
        let experts = B::gated_product_expert_bank(
            GatedProductExpertBankSpec {
                expert_count: args.num_local_experts,
                input_dimensions: args.hidden_size,
                intermediate_dimensions: args.intermediate_size,
                output_dimensions: args.hidden_size,
                policy,
                layout: GatedProductExpertLayout::Packed {
                    gate_up: ExpertProjectionSpec {
                        weight: ParameterSpec::trainable(&gate_up_weight)
                            .map_err(Error::backend)?,
                        bias: Some(
                            ParameterSpec::trainable(&gate_up_bias).map_err(Error::backend)?,
                        ),
                        format: WeightQuantization::MxFp4.into(),
                    },
                    down: ExpertProjectionSpec {
                        weight: ParameterSpec::trainable(&down_weight).map_err(Error::backend)?,
                        bias: Some(ParameterSpec::trainable(&down_bias).map_err(Error::backend)?),
                        format: WeightQuantization::MxFp4.into(),
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

    /// Executes routed experts through a runtime-owned provider.
    pub fn forward_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.router.route(input, context)?;
        provider
            .forward_routed(
                &mut self.experts,
                RoutedExpertRequest {
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass,
                },
                context,
            )
            .map_err(Error::backend)
    }

    /// Executes provider-backed rank-local TP work and performs one reduction.
    pub fn forward_parallel_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        pass: ExpertPass,
        parallel: &B::ParallelContext,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.router.route(input, context)?;
        let output = provider
            .forward_routed_tensor_parallel(
                &mut self.experts,
                RoutedExpertRequest {
                    layer: self.layer,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt_oss_policy_preserves_published_gating_constants() {
        let policy = expert_policy(7.0).unwrap();
        assert_eq!(policy.activation(), eredu_nn::GatedProductActivation::Silu);
        assert_eq!(policy.gate_upper_bound(), Some(7.0));
        assert_eq!(policy.up_absolute_bound(), Some(7.0));
        assert_eq!(policy.sigmoid_multiplier(), 1.702);
        assert_eq!(policy.up_offset(), 1.0);
    }
}

impl<B: RoutedNeuralBackend> FeedForwardOperator<B> for RoutedMlp<B> {
    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_with_provider(
            input,
            inferred_pass(input),
            &mut ResidentExpertProvider,
            context,
        )
    }

    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_parallel_with_provider(
            input,
            inferred_pass(input),
            parallel,
            &mut ResidentExpertProvider,
            context,
        )
    }
}
