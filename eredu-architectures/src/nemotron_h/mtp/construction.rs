use super::super::{attention::linear_spec, block::PredictionBlockSpec};
use super::*;
use crate::decoder::{
    construction_specs::{copy_linear, copy_normalization},
    ModuleMetadata,
};
#[derive(Debug)]
pub(crate) struct PredictionUnitSpec {
    embedding_norm: Option<NormalizationConstructionSpec>,
    hidden_norm: Option<NormalizationConstructionSpec>,
    fusion: Option<LinearSpec>,
    block: PredictionBlockSpec,
    final_norm: Option<NormalizationConstructionSpec>,
    experts: i32,
}
impl PredictionUnitSpec {
    pub(crate) fn new(args: &ModelArgs, depth: usize, relative: usize) -> Result<Self, Error> {
        let steps = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?;
        let policies = args.mtp_policies().map_err(Error::backend)?;
        let pattern = policies
            .len()
            .checked_div(steps)
            .filter(|n| *n > 0)
            .ok_or_else(|| Error::backend("Nemotron-H MTP pattern is empty"))?;
        if depth >= steps || relative >= pattern {
            return Err(Error::backend(
                "Nemotron-H MTP unit is outside its schedule",
            ));
        }
        let physical = depth
            .checked_mul(pattern)
            .and_then(|n| n.checked_add(relative))
            .ok_or_else(|| Error::backend("Nemotron-H MTP physical index overflowed"))?;
        let policy = policies[physical];
        let geometry = match policy {
            LayerPolicy::SelfAttention(_) => LayerGeometry::Attention {
                query_heads: args.num_attention_heads,
                kv_heads: args.num_key_value_heads,
            },
            LayerPolicy::SparseMoe => LayerGeometry::SparseMoe {
                routed: args.moe_intermediate_size,
                shared: args.moe_shared_expert_intermediate_size,
            },
            _ => {
                return Err(Error::backend(format!(
                    "unsupported Nemotron-H MTP policy {policy:?}"
                )))
            }
        };
        Self::with_geometry(args, depth, relative, policy, geometry)
    }
    pub(crate) fn with_geometry(
        args: &ModelArgs,
        depth: usize,
        relative: usize,
        policy: LayerPolicy,
        geometry: LayerGeometry,
    ) -> Result<Self, Error> {
        let steps = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?;
        let policies = args.mtp_policies().map_err(Error::backend)?;
        let pattern_len = policies
            .len()
            .checked_div(steps)
            .filter(|length| *length > 0)
            .ok_or_else(|| Error::backend("Nemotron-H MTP pattern is empty"))?;
        if depth >= steps || relative >= pattern_len {
            return Err(Error::backend(
                "Nemotron-H MTP unit is outside its schedule",
            ));
        }
        let physical = depth
            .checked_mul(pattern_len)
            .and_then(|start| start.checked_add(relative))
            .ok_or_else(|| Error::backend("Nemotron-H MTP physical index overflowed"))?;
        if policies[physical] != policy {
            return Err(Error::backend(format!(
                "Nemotron-H MTP policy {policy:?} does not match schedule {:?}",
                policies[physical]
            )));
        }

        let root = format!("model.mtp.layers.{physical}");
        let norm = |name: String| {
            Ok::<_, Error>(NormalizationConstructionSpec::learned(
                args.hidden_size,
                args.layer_norm_epsilon,
                ParameterSpec::trainable(name).map_err(Error::backend)?,
            ))
        };
        let first = relative == 0;
        let last = relative + 1 == pattern_len;
        Ok(Self {
            embedding_norm: first
                .then(|| norm(format!("{root}.enorm.weight")))
                .transpose()?,
            hidden_norm: first
                .then(|| norm(format!("{root}.hnorm.weight")))
                .transpose()?,
            fusion: first
                .then(|| {
                    linear_spec(
                        args,
                        &format!("{root}.eh_proj"),
                        args.hidden_size * 2,
                        args.hidden_size,
                        false,
                    )
                })
                .transpose()?,
            block: PredictionBlockSpec::new(args, physical, policy, geometry)?,
            final_norm: last
                .then(|| norm(format!("{root}.final_layernorm.weight")))
                .transpose()?,
            experts: args.n_routed_experts,
        })
    }
    pub(crate) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PredictionUnit<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(&Self, PredictionUnit<B>)>()?;
        let norm = |spec: &NormalizationConstructionSpec| {
            B::normalization(copy_normalization::<B>(spec, context)?, context)
        };
        Ok(PredictionUnit {
            embedding_norm: self.embedding_norm.as_ref().map(norm).transpose()?,
            hidden_norm: self.hidden_norm.as_ref().map(norm).transpose()?,
            fusion: self
                .fusion
                .as_ref()
                .map(|spec| B::linear(copy_linear::<B>(spec, context)?, context))
                .transpose()?,
            block: self.block.instantiate::<B>(context)?,
            final_norm: self.final_norm.as_ref().map(norm).transpose()?,
            experts: self.experts,
        })
    }
}
#[cfg(test)]
mod tests;
