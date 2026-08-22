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

impl<B: RoutedNeuralBackend> crate::decoder::RoutedFeedForwardOperator<B> for RoutedMlp<B> {
    fn forward_with_provider<P>(
        &mut self,
        _layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        RoutedMlp::forward_with_provider(self, input, pass, provider, context)
    }

    fn forward_parallel_with_provider<P>(
        &mut self,
        _layer: usize,
        input: &B::Tensor,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        RoutedMlp::forward_parallel_with_provider(self, input, pass, parallel, provider, context)
    }
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

        let experts = B::gated_product_expert_bank(expert_bank_spec(args, layer)?, context)?;
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

/// Returns the architecture-owned routed expert specification for one layer.
pub fn expert_bank_spec(
    args: &ModelArgs,
    layer: usize,
) -> Result<GatedProductExpertBankSpec, Error> {
    let expert_prefix = format!("{}.layers.{layer}.mlp.experts", args.parameter_root);
    let gate_up_weight = format!("{expert_prefix}.gate_up_proj");
    let gate_up_bias = format!("{expert_prefix}.gate_up_proj_bias");
    let down_weight = format!("{expert_prefix}.down_proj");
    let down_bias = format!("{expert_prefix}.down_proj_bias");
    Ok(GatedProductExpertBankSpec {
        expert_count: args.num_local_experts,
        input_dimensions: args.hidden_size,
        intermediate_dimensions: args.intermediate_size,
        output_dimensions: args.hidden_size,
        policy: args.gated_product_policy,
        layout: GatedProductExpertLayout::Packed {
            gate_up: ExpertProjectionSpec {
                weight: ParameterSpec::trainable(&gate_up_weight).map_err(Error::backend)?,
                bias: Some(ParameterSpec::trainable(&gate_up_bias).map_err(Error::backend)?),
                format: WeightQuantization::MxFp4.into(),
            },
            down: ExpertProjectionSpec {
                weight: ParameterSpec::trainable(&down_weight).map_err(Error::backend)?,
                bias: Some(ParameterSpec::trainable(&down_bias).map_err(Error::backend)?),
                format: WeightQuantization::MxFp4.into(),
            },
        },
    })
}

/// Returns the architecture-owned routed expert specification at rank-local geometry.
///
/// Parameter identities, native MXFP4 formats, biases, and gating policy remain
/// identical to the global bank; only placement-resolved geometry is localized.
pub fn localized_expert_bank_spec(
    args: &ModelArgs,
    layer: usize,
    expert_count: i32,
    intermediate_dimensions: i32,
) -> Result<GatedProductExpertBankSpec, Error> {
    let mut spec = expert_bank_spec(args, layer)?;
    spec.expert_count = expert_count;
    spec.intermediate_dimensions = intermediate_dimensions;
    spec.validate()?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> ModelArgs {
        crate::gpt_oss::model_args_from_config_value(&serde_json::json!({
            "model_type": "gpt_oss",
            "hidden_size": 64,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "rms_norm_eps": 1e-5,
            "sliding_window": 128,
            "max_position_embeddings": 4096,
            "rope_theta": 150000.0,
            "quantization_config": { "quant_method": "mxfp4" },
            "swiglu_limit": 7.0
        }))
        .unwrap()
    }

    #[test]
    fn gpt_oss_policy_preserves_published_gating_constants() {
        let policy = expert_policy(7.0).unwrap();
        assert_eq!(policy.activation(), eredu_nn::GatedProductActivation::Silu);
        assert_eq!(policy.gate_upper_bound(), Some(7.0));
        assert_eq!(policy.up_absolute_bound(), Some(7.0));
        assert_eq!(policy.sigmoid_multiplier(), 1.702);
        assert_eq!(policy.up_offset(), 1.0);
    }

    #[test]
    fn localized_spec_preserves_native_format_biases_and_policy() {
        let args = args();
        let global = expert_bank_spec(&args, 0).unwrap();
        let local = localized_expert_bank_spec(&args, 0, 2, 32).unwrap();
        assert_eq!(local.expert_count, 2);
        assert_eq!(local.intermediate_dimensions, 32);
        assert_eq!(local.policy, global.policy);
        let GatedProductExpertLayout::Packed {
            gate_up: global_gate_up,
            down: global_down,
        } = global.layout
        else {
            panic!("GPT-OSS experts must be packed");
        };
        let GatedProductExpertLayout::Packed {
            gate_up: local_gate_up,
            down: local_down,
        } = local.layout
        else {
            panic!("GPT-OSS experts must be packed");
        };
        assert_eq!(local_gate_up.weight, global_gate_up.weight);
        assert_eq!(local_gate_up.bias, global_gate_up.bias);
        assert_eq!(local_gate_up.format, global_gate_up.format);
        assert_eq!(local_down.weight, global_down.weight);
        assert_eq!(local_down.bias, global_down.bias);
        assert_eq!(local_down.format, global_down.format);
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
