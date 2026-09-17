use super::super::{attention::AttentionSpec, mlp::SparseMoeSpec};
use super::*;
use crate::decoder::{construction_specs::copy_normalization, ModuleMetadata};
#[derive(Debug)]
enum PredictionOperatorSpec {
    Attention(AttentionSpec),
    Sparse(SparseMoeSpec),
}
#[derive(Debug)]
pub(crate) struct PredictionBlockSpec {
    operator: PredictionOperatorSpec,
    norm: NormalizationConstructionSpec,
    residual_in_fp32: bool,
}
impl PredictionBlockSpec {
    pub(crate) fn new(
        args: &ModelArgs,
        physical: usize,
        policy: LayerPolicy,
        geometry: LayerGeometry,
    ) -> Result<Self, Error> {
        let root = format!("model.mtp.layers.{physical}");
        let global = usize::try_from(args.num_hidden_layers).map_err(Error::backend)? + physical;
        let operator=match(policy,geometry){
            (LayerPolicy::SelfAttention(attention),LayerGeometry::Attention{query_heads,kv_heads})=>
                PredictionOperatorSpec::Attention(AttentionSpec::new(args,attention,&format!("{root}.mixer"),query_heads,kv_heads)?),
            (LayerPolicy::SparseMoe,LayerGeometry::SparseMoe{routed,shared})=>
                PredictionOperatorSpec::Sparse(SparseMoeSpec::new(args,global,&format!("{root}.mixer"),routed,shared)?),
            _=>return Err(Error::backend(format!("Nemotron-H MTP physical layer {physical} policy {policy:?} does not match {geometry:?}"))),
        };
        Ok(Self {
            operator,
            norm: NormalizationConstructionSpec::learned(
                args.hidden_size,
                args.layer_norm_epsilon,
                ParameterSpec::trainable(format!("{root}.norm.weight")).map_err(Error::backend)?,
            ),
            residual_in_fp32: args.residual_in_fp32,
        })
    }
    pub(crate) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Block<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(&Self, Block<B>)>()?;
        let operator = match &self.operator {
            PredictionOperatorSpec::Attention(spec) => {
                Operator::Attention(spec.instantiate::<B>(context)?)
            }
            PredictionOperatorSpec::Sparse(spec) => {
                Operator::Sparse(spec.instantiate::<B>(context)?)
            }
        };
        Ok(Block {
            operator,
            norm: B::normalization(copy_normalization::<B>(&self.norm, context)?, context)?,
            residual_in_fp32: self.residual_in_fp32,
        })
    }
}

#[derive(Debug)]
enum TargetOperatorSpec {
    Mamba(super::super::mamba::MambaSpec),
    Attention(AttentionSpec),
    Dense(super::super::mlp::DenseMlpSpec),
    Sparse(SparseMoeSpec),
}
#[derive(Debug)]
pub(crate) struct TargetBlockSpec {
    operator: TargetOperatorSpec,
    norm: NormalizationConstructionSpec,
    residual_in_fp32: bool,
}
impl TargetBlockSpec {
    pub(crate) fn global_geometry(args: &ModelArgs, layer: usize) -> Result<LayerGeometry, Error> {
        match args.layer_schedule.get(layer) {
            Some(LayerPolicy::Mamba) => Ok(LayerGeometry::Mamba { heads: args.mamba_num_heads, groups: args.n_groups }),
            Some(LayerPolicy::SelfAttention(_)) => Ok(LayerGeometry::Attention { query_heads: args.num_attention_heads, kv_heads: args.num_key_value_heads }),
            Some(LayerPolicy::DenseMlp) => Ok(LayerGeometry::DenseMlp { intermediate: args.intermediate_size }),
            Some(LayerPolicy::SparseMoe) => Ok(LayerGeometry::SparseMoe { routed: args.moe_intermediate_size, shared: args.moe_shared_expert_intermediate_size }),
            None => Err(Error::backend(format!("Nemotron-H has no layer {layer}"))),
        }
    }
    pub(crate) fn new(
        args: &ModelArgs, layer: usize, geometry: LayerGeometry, selected: Option<eredu_nn::GroupedRelu2Spec>,
    ) -> Result<Self, Error> {
        let policy = args.layer_schedule.get(layer).copied()
            .ok_or_else(|| Error::backend(format!("Nemotron-H has no layer {layer}")))?;
        if selected.is_some() && policy != LayerPolicy::SparseMoe {
            return Err(Error::backend(format!("Nemotron-H realization names non-sparse unit target.{layer}")));
        }
        let root = format!("model.layers.{layer}");
        let operator = match (policy, geometry) {
            (LayerPolicy::Mamba, LayerGeometry::Mamba { heads, groups }) =>
                TargetOperatorSpec::Mamba(super::super::mamba::MambaSpec::new(args, layer, heads, groups)?),
            (LayerPolicy::SelfAttention(attention), LayerGeometry::Attention { query_heads, kv_heads }) =>
                TargetOperatorSpec::Attention(AttentionSpec::new(args, attention, &format!("{root}.attention"), query_heads, kv_heads)?),
            (LayerPolicy::DenseMlp, LayerGeometry::DenseMlp { intermediate }) =>
                TargetOperatorSpec::Dense(super::super::mlp::DenseMlpSpec::new(args, &format!("{root}.mlp"), intermediate)?),
            (LayerPolicy::SparseMoe, LayerGeometry::SparseMoe { routed, shared }) =>
                TargetOperatorSpec::Sparse(match selected {
                    Some(spec) => SparseMoeSpec::with_experts(args, layer, &format!("{root}.moe"), spec, shared)?,
                    None => SparseMoeSpec::new(args, layer, &format!("{root}.moe"), routed, shared)?,
                }),
            _ => return Err(Error::backend(format!("Nemotron-H layer {layer} policy {policy:?} does not match {geometry:?}"))),
        };
        Ok(Self {
            operator,
            norm: NormalizationConstructionSpec::learned(args.hidden_size, args.layer_norm_epsilon,
                ParameterSpec::trainable(format!("{root}.norm.weight")).map_err(Error::backend)?),
            residual_in_fp32: args.residual_in_fp32,
        })
    }
    pub(crate) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self, context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Block<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(&Self, Block<B>)>()?;
        let operator = match &self.operator {
            TargetOperatorSpec::Mamba(spec) => Operator::Mamba(spec.instantiate::<B>(context)?),
            TargetOperatorSpec::Attention(spec) => Operator::Attention(spec.instantiate::<B>(context)?),
            TargetOperatorSpec::Dense(spec) => Operator::Dense(spec.instantiate::<B>(context)?),
            TargetOperatorSpec::Sparse(spec) => Operator::Sparse(spec.instantiate::<B>(context)?),
        };
        Ok(Block {
            operator,
            norm: B::normalization(copy_normalization::<B>(&self.norm, context)?, context)?,
            residual_in_fp32: self.residual_in_fp32,
        })
    }
}
