//! Once-compiled declarations for each dense depth and optional shared norm.
use super::*;
use crate::decoder::{
    ModuleMetadata,
    construction_specs::{copy_linear, copy_normalization},
};
#[derive(Debug)]
pub(crate) struct MtpDepthSpec {
    hidden_norm: NormalizationConstructionSpec,
    embedding_norm: NormalizationConstructionSpec,
    input_projection: LinearSpec,
    transformer_block: super::super::text::DenseLayerSpec,
}
impl MtpDepthSpec {
    fn new(text: &TextArgs, attention: AttentionPolicy, depth: usize) -> Result<Self, Error> {
        let root = format!("model.mtp.layers.{depth}");
        let norm = |field: &str| {
            Ok::<_, Error>(NormalizationConstructionSpec::learned(
                text.hidden_size,
                text.rms_norm_eps,
                ParameterSpec::trainable(format!("{root}.{field}.weight"))
                    .map_err(Error::backend)?,
            ))
        };
        let input_weight = format!("{root}.input_proj.weight");
        Ok(Self {
            hidden_norm: norm("hidden_norm")?,
            embedding_norm: norm("embed_norm")?,
            input_projection: LinearSpec {
                input: text.hidden_size * 2,
                output: text.hidden_size,
                weight: ParameterSpec::trainable(&input_weight).map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    &input_weight,
                    text.linear_format_for(&input_weight),
                )?,
            },
            transformer_block: super::super::text::DenseLayerSpec::new(
                text,
                attention,
                &format!("{root}.transformer_block"),
                0,
                0,
            )?,
        })
    }
    pub(crate) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MtpDepth<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, &Self, MtpDepth<B>)>()?;
        Ok(MtpDepth {
            hidden_norm: B::normalization(
                copy_normalization::<B>(&self.hidden_norm, context)?,
                context,
            )?,
            embedding_norm: B::normalization(
                copy_normalization::<B>(&self.embedding_norm, context)?,
                context,
            )?,
            input_projection: B::linear(
                copy_linear::<B>(&self.input_projection, context)?,
                context,
            )?,
            transformer_block: self.transformer_block.instantiate::<B>(context)?,
        })
    }
}
#[derive(Debug)]
pub(crate) struct MtpModelSpec {
    layers: Vec<MtpDepthSpec>,
    chain_norm: Option<NormalizationConstructionSpec>,
    policies: Vec<AttentionPolicy>,
}
impl MtpModelSpec {
    pub(crate) fn new(args: &ModelArgs) -> Result<Option<Self>, Error> {
        let Some(config) = args.mtp_config.as_ref() else {
            return Ok(None);
        };
        let count = usize::try_from(config.num_nextn_predict_layers)
            .map_err(|_| Error::backend("Inkling MTP layer count is negative"))?;
        if count == 0 {
            return Ok(None);
        }
        let sliding = args
            .text_config
            .layer_schedule
            .iter()
            .find_map(|policy| policy.attention.window())
            .map(|window| AttentionPolicy::Sliding { window });
        if !config.local_layer_ids.is_empty() && sliding.is_none() {
            return Err(Error::backend(
                "Inkling MTP local layers require a backbone sliding window",
            ));
        }
        let policies = (0..count)
            .map(|depth| {
                if config.local_layer_ids.contains(&depth) {
                    sliding.expect("validated local MTP policy")
                } else {
                    AttentionPolicy::Full
                }
            })
            .collect::<Vec<_>>();
        let mut layers = Vec::with_capacity(count);
        for (depth, attention) in policies.iter().copied().enumerate() {
            let text = mtp_text_args(&args.text_config, config, attention)?;
            layers.push(MtpDepthSpec::new(&text, attention, depth)?);
        }
        let chain_norm = config
            .chain_hidden_post_norm
            .then(|| {
                Ok::<_, Error>(NormalizationConstructionSpec::learned(
                    args.text_config.hidden_size,
                    args.text_config.rms_norm_eps,
                    ParameterSpec::trainable("model.mtp.chain_norm.weight")
                        .map_err(Error::backend)?,
                ))
            })
            .transpose()?;
        Ok(Some(Self {
            layers,
            chain_norm,
            policies,
        }))
    }
    pub(crate) fn len(&self) -> usize {
        self.layers.len()
    }
    pub(crate) fn depth(&self, index: usize) -> Option<&MtpDepthSpec> {
        self.layers.get(index)
    }
    pub(crate) fn has_shared(&self) -> bool {
        self.chain_norm.is_some()
    }
    pub(crate) fn shared<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<MtpShared<B>>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(
            Option<&NormalizationConstructionSpec>,
            Option<MtpShared<B>>,
            &Self,
        )>()?;
        self.chain_norm
            .as_ref()
            .map(|spec| {
                B::normalization(copy_normalization::<B>(spec, context)?, context)
                    .map(|chain_norm| MtpShared { chain_norm })
            })
            .transpose()
    }
    pub(crate) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MtpModel<B>, Error> {
        ModuleMetadata::new::<B>(context)
            .controls::<(Self, &Self, MtpModel<B>, usize, AttentionPolicy)>()?;
        let mut layers = match B::construction_metadata(context) {
            Some(context) => context.metadata_vec(self.layers.len())?,
            None => Vec::with_capacity(self.layers.len()),
        };
        for spec in &self.layers {
            layers.push(spec.instantiate::<B>(context)?);
        }
        let chain_norm = self.shared::<B>(context)?.map(|shared| shared.chain_norm);
        let mut policies = match B::construction_metadata(context) {
            Some(context) => context.metadata_vec(self.policies.len())?,
            None => Vec::with_capacity(self.policies.len()),
        };
        policies.extend_from_slice(&self.policies);
        Ok(MtpModel {
            layers,
            chain_norm,
            policies,
        })
    }
}

#[cfg(test)]
mod tests;
