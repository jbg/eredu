//! Exact sparse worker, entered only after the shared layer policy selects it.
use super::*;
use eredu_nn::{GroupScoring, SelectorInputTransformSpec, TopKGroupSelectionSpec, TopKGroupSelectorSpec};

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> DenseBlock<B> {
    #[inline(never)]
    pub(super) fn construct_sparse(args: &ModelArgs, prefix: &str,
        routed_spec: Option<GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context)
        -> Result<(B::Selector, B::GatedProductGroups), Error> {
        let metadata = ModuleMetadata::new::<B>(context);
        metadata.controls::<(&ModelArgs, &str, Option<GroupedGatedProductSpec>,
            (B::Selector, B::GatedProductGroups), &<B::Tensor as Tensor>::Context)>()?;
            metadata.controls::<(
                TopKGroupSelectorSpec,
                TopKGroupSelectionSpec,
                SelectorInputTransformSpec,
                GroupedGatedProductSpec,
                String,
                String,
            )>()?;
            let expert_count = args.num_experts.ok_or_else(|| {
                metadata.error(format_args!("Gemma 4 sparse block has no expert count"))
            })?;
            let top_k = args.top_k_experts.ok_or_else(|| {
                metadata.error(format_args!("Gemma 4 sparse block has no top-k count"))
            })?;
            let router_prefix = metadata.text(format_args!("{prefix}.router"))?;
            let router_weight = metadata.text(format_args!("{router_prefix}.proj.weight"))?;
            let selector = TopKGroupSelectorSpec::new(
                args.hidden_size,
                metadata.plain_parameter(&router_weight)?,
                metadata.format(&router_weight, args.linear_format_for(&router_weight))?,
                TopKGroupSelectionSpec::new(
                    expert_count,
                    top_k,
                    GroupScoring::SelectedSoftmax,
                    false,
                )?,
            )?
            .with_input_transform(SelectorInputTransformSpec::new(
                args.rms_norm_eps,
                metadata.named_parameter(format_args!("{router_prefix}.scale"))?,
                true,
            )?)
            .with_coefficient_scale(
                metadata.named_parameter(format_args!("{router_prefix}.per_expert_scale"))?,
            );
            let router = B::top_k_group_selector(selector, context)?;
            let spec = match routed_spec {
                Some(spec) => spec,
                None => expert_bank_spec_at_with(
                    args,
                    &metadata.text(format_args!("{prefix}.experts.switch_glu"))?,
                    metadata,
                )?,
            };
            let experts = B::grouped_gated_product(spec, context)?;
        Ok((router, experts))
    }
}
