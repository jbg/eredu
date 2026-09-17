//! Immutable declarations consumed by the same typed prediction-layer builder.
use super::*;
use crate::decoder::ModuleMetadata;

#[derive(Debug)]
pub(crate) struct V3PredictionLayerSpec {
    embedding_norm: NormalizationConstructionSpec,
    hidden_norm: NormalizationConstructionSpec,
    fusion: LinearSpec,
    decoder: crate::deepseek::block::V3PredictionBlockSpec,
    output_norm: NormalizationConstructionSpec,
    output_head: LinearSpec,
}
impl V3PredictionLayerSpec {
    pub(crate) fn new(args: &V3Args, depth: usize) -> Result<Self, Error> {
        let count = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?;
        if depth >= count {
            return Err(Error::backend(format!(
                "V3 prediction depth {depth} is outside {count} layers"
            )));
        }
        let global = usize::try_from(args.num_hidden_layers).map_err(Error::backend)? + depth;
        let root = format!("model.layers.{global}");
        let norm = |name: String| -> Result<_, Error> {
            Ok(NormalizationConstructionSpec::learned(
                args.hidden_size,
                args.rms_norm_eps,
                parameter(name)?,
            ))
        };
        let linear = |name: String, input, output| -> Result<_, Error> {
            let format = args.linear_format_for(&name);
            Ok(LinearSpec {
                input,
                output,
                weight: parameter(&name)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(&name, format)?,
            })
        };
        Ok(Self {
            embedding_norm: norm(format!("{root}.enorm.weight"))?,
            hidden_norm: norm(format!("{root}.hnorm.weight"))?,
            fusion: linear(
                format!("{root}.eh_proj.weight"),
                2 * args.hidden_size,
                args.hidden_size,
            )?,
            decoder: crate::deepseek::block::V3PredictionBlockSpec::new(args, global)?,
            output_norm: norm(format!("{root}.shared_head.norm.weight"))?,
            output_head: linear(
                format!("{root}.shared_head.head.weight"),
                args.hidden_size,
                args.vocab_size,
            )?,
        })
    }
    pub(crate) fn instantiate<B>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<V3PredictionLayer<B>, Error>
    where
        B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
    {
        let metadata = ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, V3PredictionLayer<B>, &Self)>()?;
        Ok(V3PredictionLayer {
            embedding_norm: B::normalization(
                copy_normalization::<B>(&self.embedding_norm, context)?,
                context,
            )?,
            hidden_norm: B::normalization(
                copy_normalization::<B>(&self.hidden_norm, context)?,
                context,
            )?,
            fusion: B::linear(copy_linear::<B>(&self.fusion, context)?, context)?,
            decoder: self.decoder.instantiate::<B>(context)?,
            output_norm: B::normalization(
                copy_normalization::<B>(&self.output_norm, context)?,
                context,
            )?,
            output_head: B::linear(copy_linear::<B>(&self.output_head, context)?, context)?,
        })
    }
}
pub(crate) use crate::decoder::construction_specs::{
    copy_grouped, copy_linear, copy_low_rank, copy_normalization, copy_selector,
};

#[cfg(test)]
mod tests;
