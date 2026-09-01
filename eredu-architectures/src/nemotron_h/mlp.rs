//! Nemotron-H dense and routed ReLU-squared feed-forward operators.

use eredu_nn::{
    Error, GroupScoring, GroupSelectionOperator, GroupedNeuralBackend, GroupedRelu2Spec,
    LinearOperator, LinearSpec, ParameterSpec, Parameterized, Tensor, TopKGroupSelectionSpec,
    TopKGroupSelectorSpec,
};
use eredu_runtime::{ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest};

use crate::linear_format::standard_expert_projection;

use super::ModelArgs;

/// Dense up/ReLU²/down projection pair.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseMlp<B: GroupedNeuralBackend> {
    /// Up projection.
    pub up_proj: B::Linear,
    /// Down projection.
    pub down_proj: B::Linear,
}

impl<B: GroupedNeuralBackend> DenseMlp<B> {
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
                    format: crate::linear_format::standard_linear_format(
                        &weight,
                        args.weight_quantization_for(&weight).into(),
                    )?,
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

/// Grouped sigmoid selection: routing, packed ReLU² experts, and one shared expert.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SparseMoe<B: GroupedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    /// Grouped correction-bias router.
    pub gate: B::Selector,
    /// Packed routed experts.
    pub experts: B::Relu2Groups,
    /// Always-executed shared expert.
    pub shared_experts: DenseMlp<B>,
}

impl<B: GroupedNeuralBackend> SparseMoe<B> {
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
        let routing = TopKGroupSelectionSpec::new(
            args.n_routed_experts,
            args.num_experts_per_tok,
            GroupScoring::Sigmoid,
            args.norm_topk_prob,
        )?
        .with_groups(args.n_group, args.topk_group)?
        .with_weight_policy(1e-20, args.routed_scaling_factor)?;
        let selector = TopKGroupSelectorSpec::new(
            args.hidden_size,
            ParameterSpec::trainable(&gate_weight).map_err(Error::backend)?,
            crate::linear_format::standard_linear_format(
                &gate_weight,
                args.weight_quantization_for(&gate_weight).into(),
            )?,
            routing,
        )?
        .with_correction_bias(
            ParameterSpec::trainable(format!("{prefix}.gate.e_score_correction_bias"))
                .map_err(Error::backend)?,
        )?;
        let gate = B::top_k_group_selector(selector, context)?;
        let experts = B::grouped_relu2(
            expert_bank_spec_at(
                args,
                &format!("{prefix}.experts"),
                args.n_routed_experts,
                routed_intermediate,
            )?,
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
        let routes = self.gate.select(input, context)?;
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
        let routes = self.gate.select(input, context)?;
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
            selected_experts: routes.group_indices(),
            selected_scores: routes.selected_scores(),
            coefficients: routes.coefficients(),
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
        let routes = self.gate.select(input, context)?;
        let routed = provider
            .forward_relu2_routed_tensor_parallel(
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
            .map_err(|error| Error::backend(error.to_string()))?;
        let routed =
            eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(routed, parallel, context)?;
        routed.add(
            &self
                .shared_experts
                .forward_parallel(input, parallel, context)?,
            context,
        )
    }
}

fn expert_bank_spec_at(
    args: &ModelArgs,
    experts_prefix: &str,
    expert_count: i32,
    intermediate_dimensions: i32,
) -> Result<GroupedRelu2Spec, Error> {
    let up_name = format!("{experts_prefix}.up_proj");
    let down_name = format!("{experts_prefix}.down_proj");
    GroupedRelu2Spec::new(
        expert_count,
        args.hidden_size,
        intermediate_dimensions,
        standard_expert_projection(
            &up_name,
            None,
            args.weight_quantization_for(&up_name).into(),
        )?,
        standard_expert_projection(
            &down_name,
            None,
            args.weight_quantization_for(&down_name).into(),
        )?,
    )
}

/// Returns the architecture-owned ReLU-squared expert specification for one
/// target or appended MTP layer.
pub fn expert_bank_spec(args: &ModelArgs, layer: usize) -> Result<GroupedRelu2Spec, Error> {
    localized_expert_bank_spec(
        args,
        layer,
        args.n_routed_experts,
        args.moe_intermediate_size,
    )
}

/// Returns the architecture-owned ReLU-squared expert specification at
/// placement-resolved local geometry.
pub fn localized_expert_bank_spec(
    args: &ModelArgs,
    layer: usize,
    expert_count: i32,
    intermediate_dimensions: i32,
) -> Result<GroupedRelu2Spec, Error> {
    let target_layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let (policy, experts_prefix) = if layer < target_layers {
        let policy = args
            .layer_schedule
            .get(layer)
            .copied()
            .ok_or_else(|| Error::backend(format!("Nemotron-H has no layer {layer}")))?;
        (policy, format!("model.layers.{layer}.moe.experts"))
    } else {
        let physical = layer - target_layers;
        let policies = args.mtp_policies().map_err(Error::backend)?;
        let policy = policies.get(physical).copied().ok_or_else(|| {
            Error::backend(format!("Nemotron-H has no appended MTP layer {physical}"))
        })?;
        (policy, format!("model.mtp.layers.{physical}.mixer.experts"))
    };
    if policy != super::LayerPolicy::SparseMoe {
        return Err(Error::backend(format!(
            "Nemotron-H layer {layer} is not a sparse expert unit"
        )));
    }
    expert_bank_spec_at(args, &experts_prefix, expert_count, intermediate_dimensions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> ModelArgs {
        crate::nemotron_h::model_args_from_config_value(&serde_json::json!({
            "model_type":"nemotron_h", "vocab_size":32, "hidden_size":16,
            "intermediate_size":24, "num_hidden_layers":4,
            "hybrid_override_pattern":"M*-E", "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
            "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
            "conv_kernel":3, "n_routed_experts":4, "n_shared_experts":1,
            "moe_intermediate_size":8, "moe_shared_expert_intermediate_size":8,
            "num_experts_per_tok":2, "n_group":2, "topk_group":1,
            "num_nextn_predict_layers":1, "mtp_hybrid_override_pattern":"*E"
        }))
        .unwrap()
    }

    #[test]
    fn localized_specs_resolve_target_and_mtp_parameter_identity() {
        let args = args();
        let target = localized_expert_bank_spec(&args, 3, 2, 4).unwrap();
        assert_eq!(target.group_count(), 2);
        assert_eq!(target.intermediate_dimensions(), 4);
        assert_eq!(
            target.up().weight().id.as_str(),
            "model.layers.3.moe.experts.up_proj"
        );
        assert_eq!(
            target.down().weight().id.as_str(),
            "model.layers.3.moe.experts.down_proj"
        );

        let mtp = expert_bank_spec(&args, 5).unwrap();
        assert_eq!(mtp.group_count(), 4);
        assert_eq!(mtp.intermediate_dimensions(), 8);
        assert_eq!(
            mtp.up().weight().id.as_str(),
            "model.mtp.layers.1.mixer.experts.up_proj"
        );
        assert_eq!(
            mtp.down().weight().id.as_str(),
            "model.mtp.layers.1.mixer.experts.down_proj"
        );
    }

    #[test]
    fn expert_spec_rejects_non_sparse_units() {
        assert!(expert_bank_spec(&args(), 0).is_err());
    }
}
