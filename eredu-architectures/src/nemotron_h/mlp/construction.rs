use super::super::attention::linear_spec_with_metadata;
use super::*;
use crate::decoder::{
    construction_specs::{copy_linear, copy_relu2, copy_selector},
    ModuleMetadata,
};
#[derive(Debug)]
pub(crate) struct DenseMlpSpec {
    up: LinearSpec,
    down: LinearSpec,
}
impl DenseMlpSpec {
    pub(crate) fn new(args: &ModelArgs, prefix: &str, intermediate: i32) -> Result<Self, Error> {
        Self::new_with_metadata(args, prefix, intermediate, ModuleMetadata::ordinary())
    }
    pub(crate) fn new_with_metadata(
        args: &ModelArgs, prefix: &str, intermediate: i32, metadata: ModuleMetadata<'_>,
    ) -> Result<Self, Error> {
        metadata.controls::<(Self, &ModelArgs, &str, i32)>()?;
        Ok(Self {
            up: linear_spec_with_metadata(
                args,
                &metadata.text(format_args!("{prefix}.up_proj"))?,
                args.hidden_size,
                intermediate,
                args.mlp_bias,
                metadata,
            )?,
            down: linear_spec_with_metadata(
                args,
                &metadata.text(format_args!("{prefix}.down_proj"))?,
                intermediate,
                args.hidden_size,
                args.mlp_bias,
                metadata,
            )?,
        })
    }
    pub(crate) fn instantiate<B: NeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DenseMlp<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(&Self, DenseMlp<B>)>()?;
        Ok(DenseMlp {
            up_proj: B::linear(copy_linear::<B>(&self.up, context)?, context)?,
            down_proj: B::linear(copy_linear::<B>(&self.down, context)?, context)?,
        })
    }
}
#[derive(Debug)]
pub(crate) struct SparseMoeSpec {
    layer: usize,
    selector: TopKGroupSelectorSpec,
    experts: GroupedRelu2Spec,
    shared: DenseMlpSpec,
}
impl SparseMoeSpec {
    pub(crate) fn new(
        args: &ModelArgs,
        layer: usize,
        prefix: &str,
        routed: i32,
        shared: i32,
    ) -> Result<Self, Error> {
        let experts = expert_bank_spec_at(
            args,
            &format!("{prefix}.experts"),
            args.n_routed_experts,
            routed,
        )?;
        Self::with_experts(args, layer, prefix, experts, shared)
    }
    pub(crate) fn with_experts(
        args: &ModelArgs,
        layer: usize,
        prefix: &str,
        spec: GroupedRelu2Spec,
        shared_intermediate: i32,
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

        Ok(Self {
            layer,
            selector,
            experts: spec,
            shared: DenseMlpSpec::new(
                args,
                &format!("{prefix}.shared_experts"),
                shared_intermediate,
            )?,
        })
    }
    pub(crate) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<SparseMoe<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(&Self, SparseMoe<B>)>()?;
        Ok(SparseMoe {
            layer: self.layer,
            resident_unit_coordinates: None,
            gate: B::top_k_group_selector(copy_selector::<B>(&self.selector, context)?, context)?,
            experts: B::grouped_relu2(copy_relu2::<B>(&self.experts, context)?, context)?,
            shared_experts: self.shared.instantiate::<B>(context)?,
        })
    }
}
