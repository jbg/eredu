//! The ordinary V4 block's complete immutable construction declaration.
use super::*;
use crate::decoder::{ModuleMetadata, construction_specs::*};
#[derive(Debug)]
pub(crate) struct V4BlockSpec {
    attention: super::super::attention::v4::V4AttentionSpec,
    feed_forward: super::super::moe::RoutedPlusSharedSpec,
    attention_norm: NormalizationConstructionSpec,
    feed_forward_norm: NormalizationConstructionSpec,
    attention_connection: HyperConnectionSpec,
    feed_forward_connection: HyperConnectionSpec,
    token_experts: Option<(ParameterSpec, [i32; 2])>,
    normalization_epsilon: f32,
}
impl V4BlockSpec {
    pub(crate) fn new(
        args: &V4Args,
        layer: usize,
        root: &str,
        expert_spec: Option<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
            .map_err(Error::backend)?;
        if layer >= total {
            return Err(Error::backend(format!("V4 layer {layer} is out of range")));
        }
        let norm = |name: String| -> Result<_, Error> {
            Ok(NormalizationConstructionSpec::learned(
                args.hidden_size,
                args.rms_norm_eps,
                parameter(name)?,
            ))
        };
        let connection = |kind: &str| -> Result<_, Error> {
            Ok(HyperConnectionSpec {
                streams: args.hc_mult,
                hidden_size: args.hidden_size,
                sinkhorn_iterations: usize::try_from(args.hc_sinkhorn_iters)
                    .map_err(Error::backend)?,
                epsilon: args.hc_eps,
                function: parameter(format!("{root}.hc_{kind}_fn"))?,
                base: parameter(format!("{root}.hc_{kind}_base"))?,
                scale: parameter(format!("{root}.hc_{kind}_scale"))?,
            })
        };
        let attention = super::super::attention::v4::V4AttentionSpec::new(
            args,
            layer,
            &format!("{root}.attn"),
        )?;
        let policy = super::super::v4::moe_policy_at(args, layer, &format!("{root}.ffn"))?;
        let experts = match expert_spec {
            Some(spec) => spec,
            None => super::super::moe::expert_bank_spec(&policy)?,
        };
        Ok(Self {
            attention,
            feed_forward: super::super::moe::RoutedPlusSharedSpec::new(&policy, experts)?,
            attention_norm: norm(format!("{root}.attn_norm.weight"))?,
            feed_forward_norm: norm(format!("{root}.ffn_norm.weight"))?,
            attention_connection: connection("attn")?,
            feed_forward_connection: connection("ffn")?,
            token_experts: (layer < args.num_hash_layers as usize)
                .then(|| {
                    Ok::<_, Error>((
                        parameter(format!("{root}.ffn.gate.tid2eid"))?,
                        [args.vocab_size, args.num_experts_per_tok],
                    ))
                })
                .transpose()?,
            normalization_epsilon: args.rms_norm_eps,
        })
    }
    pub(crate) fn instantiate<
        B: HyperNeuralBackend + eredu_nn::DistributedNeuralBackend + GroupedNeuralBackend,
    >(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<V4Block<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, V4Block<B>, &Self)>()?;
        Ok(V4Block {
            attention: self.attention.instantiate::<B>(context)?,
            feed_forward: self.feed_forward.instantiate::<B>(context)?,
            attention_norm: B::normalization(
                copy_normalization::<B>(&self.attention_norm, context)?,
                context,
            )?,
            feed_forward_norm: B::normalization(
                copy_normalization::<B>(&self.feed_forward_norm, context)?,
                context,
            )?,
            attention_connection: HyperConnection::new(
                copy_hyper_connection::<B>(&self.attention_connection, context)?,
                context,
            )?,
            feed_forward_connection: HyperConnection::new(
                copy_hyper_connection::<B>(&self.feed_forward_connection, context)?,
                context,
            )?,
            token_experts: self
                .token_experts
                .as_ref()
                .map(|(spec, shape)| {
                    Parameter::unloaded_i32(copy_parameter::<B>(spec, context)?, shape, context)
                })
                .transpose()?,
            normalization_epsilon: self.normalization_epsilon,
        })
    }
}
