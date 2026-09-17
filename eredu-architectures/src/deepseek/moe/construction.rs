//! Immutable router and expert declarations, built by the ordinary policy worker.
use super::*;
use crate::{
    decoder::ModuleMetadata,
    deepseek::mtp::construction::{copy_grouped, copy_linear, copy_selector},
};
#[derive(Debug)]
pub(crate) struct RoutedPlusSharedSpec {
    layer: usize,
    expert_count: i32,
    router: TopKGroupSelectorSpec,
    experts: GroupedGatedProductSpec,
    shared_gate: LinearSpec,
    shared_up: LinearSpec,
    shared_down: LinearSpec,
    shared_limit: Option<eredu_nn::GatedProductPolicy>,
}
impl RoutedPlusSharedSpec {
    pub(crate) fn new(
        policy: &MoePolicy,
        expert_spec: GroupedGatedProductSpec,
    ) -> Result<Self, Error> {
        let routing = TopKGroupSelectionSpec::new(
            policy.expert_count,
            policy.routes_per_token,
            policy.scoring,
            policy.normalize_routes,
        )?
        .with_groups(policy.expert_groups, policy.selected_groups)?
        .with_weight_policy(policy.normalization_epsilon, policy.routed_scaling)?;
        let mut selector = TopKGroupSelectorSpec::new(
            policy.hidden,
            parameter(&policy.router_weight)?,
            crate::linear_format::standard_linear_format(
                &policy.router_weight,
                policy.router_format,
            )?,
            routing,
        )?;
        if let Some(correction) = policy
            .correction_bias
            .as_deref()
            .map(parameter)
            .transpose()?
        {
            selector = selector.with_correction_bias(correction)?;
        }
        let shared = |weight: &str, input, output, format| -> Result<_, Error> {
            Ok(LinearSpec {
                input,
                output,
                weight: parameter(weight)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(weight, format)?,
            })
        };
        Ok(Self {
            layer: policy.layer,
            expert_count: policy.expert_count,
            router: selector,
            experts: expert_spec,
            shared_gate: shared(
                &policy.shared_gate,
                policy.hidden,
                policy.shared_width,
                policy.shared_gate_format,
            )?,
            shared_up: shared(
                &policy.shared_up,
                policy.hidden,
                policy.shared_width,
                policy.shared_up_format,
            )?,
            shared_down: shared(
                &policy.shared_down,
                policy.shared_width,
                policy.hidden,
                policy.shared_down_format,
            )?,
            shared_limit: policy.shared_limit,
        })
    }
    pub(crate) fn instantiate<B>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<RoutedPlusShared<B>, Error>
    where
        B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        ModuleMetadata::new::<B>(context).controls::<(Self, RoutedPlusShared<B>, &Self)>()?;
        Ok(RoutedPlusShared {
            layer: self.layer,
            expert_count: self.expert_count,
            router: B::top_k_group_selector(copy_selector::<B>(&self.router, context)?, context)?,
            experts: B::grouped_gated_product(copy_grouped::<B>(&self.experts, context)?, context)?,
            shared_gate: B::linear(copy_linear::<B>(&self.shared_gate, context)?, context)?,
            shared_up: B::linear(copy_linear::<B>(&self.shared_up, context)?, context)?,
            shared_down: B::linear(copy_linear::<B>(&self.shared_down, context)?, context)?,
            shared_limit: self.shared_limit,
            resident_unit_coordinates: None,
        })
    }
}
