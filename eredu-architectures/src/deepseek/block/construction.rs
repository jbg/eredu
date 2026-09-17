//! One immutable source and instantiation worker for V3 target and prediction blocks.
use super::*;
use crate::{
    decoder::ModuleMetadata,
    deepseek::mtp::construction::{copy_linear, copy_normalization},
};
#[derive(Debug)]
pub(crate) struct DenseSwiGluSpec {
    gate: LinearSpec,
    up: LinearSpec,
    down: LinearSpec,
}
impl DenseSwiGluSpec {
    pub(super) fn new(args: &V3Args, layer: usize) -> Result<Self, Error> {
        let root = format!("model.layers.{layer}.mlp");
        let linear = |field: &str, input, output| {
            let name = format!("{root}.{field}.weight");
            Ok::<_, Error>(LinearSpec {
                input,
                output,
                weight: parameter(&name)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    &name,
                    args.linear_format_for(&name),
                )?,
            })
        };
        Ok(Self {
            gate: linear("gate_proj", args.hidden_size, args.intermediate_size)?,
            up: linear("up_proj", args.hidden_size, args.intermediate_size)?,
            down: linear("down_proj", args.intermediate_size, args.hidden_size)?,
        })
    }
    pub(super) fn instantiate<B: NeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DenseSwiGlu<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, DenseSwiGlu<B>, &Self)>()?;
        Ok(DenseSwiGlu {
            gate: B::linear(copy_linear::<B>(&self.gate, context)?, context)?,
            up: B::linear(copy_linear::<B>(&self.up, context)?, context)?,
            down: B::linear(copy_linear::<B>(&self.down, context)?, context)?,
        })
    }
}
#[derive(Debug)]
enum FeedForwardSpec {
    Dense(DenseSwiGluSpec),
    Routed(crate::deepseek::moe::RoutedPlusSharedSpec),
}
#[derive(Debug)]
pub(crate) struct V3BlockSpec {
    attention: crate::deepseek::attention::v3::AttentionSpec,
    feed_forward: FeedForwardSpec,
    input_norm: NormalizationConstructionSpec,
    post_attention_norm: NormalizationConstructionSpec,
}
impl V3BlockSpec {
    pub(crate) fn new(
        args: &V3Args,
        layer: usize,
        policy: LayerPolicy,
        expert_spec: Option<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        let root = format!("model.layers.{layer}");
        let norm = |field: &str| -> Result<_, Error> {
            Ok(NormalizationConstructionSpec::learned(
                args.hidden_size,
                args.rms_norm_eps,
                parameter(format!("{root}.{field}.weight"))?,
            ))
        };
        let attention = crate::deepseek::attention::v3::AttentionSpec::new(args, layer)?;
        let feed_forward = match policy {
            LayerPolicy::DenseMlp => FeedForwardSpec::Dense(DenseSwiGluSpec::new(args, layer)?),
            LayerPolicy::SparseMoe => {
                let policy = if layer < args.layer_schedule.len() {
                    super::super::v3::moe_policy(args, layer)?
                } else {
                    super::super::v3::prediction_moe_policy(args, layer)?
                };
                let experts = match expert_spec {
                    Some(spec) => spec,
                    None => crate::deepseek::moe::expert_bank_spec(&policy)?,
                };
                FeedForwardSpec::Routed(crate::deepseek::moe::RoutedPlusSharedSpec::new(
                    &policy, experts,
                )?)
            }
        };
        Ok(Self {
            attention,
            feed_forward,
            input_norm: norm("input_layernorm")?,
            post_attention_norm: norm("post_attention_layernorm")?,
        })
    }
    pub(crate) fn instantiate<B>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<V3Block<B>, Error>
    where
        B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
    {
        ModuleMetadata::new::<B>(context).controls::<(Self, V3Block<B>, &Self)>()?;
        Ok(V3Block {
            attention: self.attention.instantiate::<B>(context)?,
            feed_forward: match &self.feed_forward {
                FeedForwardSpec::Dense(spec) => {
                    V3FeedForward::Dense(spec.instantiate::<B>(context)?)
                }
                FeedForwardSpec::Routed(spec) => {
                    V3FeedForward::Routed(spec.instantiate::<B>(context)?)
                }
            },
            input_norm: B::normalization(
                copy_normalization::<B>(&self.input_norm, context)?,
                context,
            )?,
            post_attention_norm: B::normalization(
                copy_normalization::<B>(&self.post_attention_norm, context)?,
                context,
            )?,
        })
    }
}
/// Existing prediction source delegates to the same complete block declaration.
#[derive(Debug)]
pub(crate) struct V3PredictionBlockSpec(V3BlockSpec);
impl V3PredictionBlockSpec {
    pub(crate) fn new(args: &V3Args, layer: usize) -> Result<Self, Error> {
        V3BlockSpec::new(args, layer, LayerPolicy::SparseMoe, None).map(Self)
    }
    pub(crate) fn instantiate<B>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<V3Block<B>, Error>
    where
        B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
    {
        self.0.instantiate::<B>(context)
    }
}
