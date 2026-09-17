//! Retained V4 sequential projection, block and shared-head declarations.
use super::*;
use crate::decoder::{ModuleMetadata, construction_specs::*};
#[derive(Debug)]
pub(crate) struct V4PredictionLayerSpec {
    embedding_projection: LinearSpec,
    hidden_projection: LinearSpec,
    embedding_norm: NormalizationConstructionSpec,
    hidden_norm: NormalizationConstructionSpec,
    decoder: super::super::block::V4BlockSpec,
    output_norm: NormalizationConstructionSpec,
    hyper_head: HyperHeadSpec,
}
impl V4PredictionLayerSpec {
    pub(crate) fn new(args: &V4Args, depth: usize) -> Result<Self, Error> {
        if args.dspark.is_some() {
            return Err(Error::backend(
                "fused DSpark checkpoints do not expose sequential V4 prediction layers",
            ));
        }
        let count = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?;
        if depth >= count {
            return Err(Error::backend(format!(
                "V4 prediction depth {depth} is outside {count} layers"
            )));
        }
        let global = usize::try_from(args.num_hidden_layers).map_err(Error::backend)? + depth;
        let root = format!("mtp.{depth}");
        let norm = |name: String| -> Result<_, Error> {
            Ok(NormalizationConstructionSpec::learned(
                args.hidden_size,
                args.rms_norm_eps,
                parameter(name)?,
            ))
        };
        let linear = |name: String| -> Result<_, Error> {
            Ok(LinearSpec {
                input: args.hidden_size,
                output: args.hidden_size,
                weight: parameter(&name)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    &name,
                    args.linear_format_for(&name),
                )?,
            })
        };
        Ok(Self {
            embedding_projection: linear(format!("{root}.e_proj.weight"))?,
            hidden_projection: linear(format!("{root}.h_proj.weight"))?,
            embedding_norm: norm(format!("{root}.enorm.weight"))?,
            hidden_norm: norm(format!("{root}.hnorm.weight"))?,
            decoder: super::super::block::V4BlockSpec::new(args, global, &root, None)?,
            output_norm: norm(format!("{root}.norm.weight"))?,
            hyper_head: HyperHeadSpec {
                streams: args.hc_mult,
                hidden_size: args.hidden_size,
                norm_epsilon: args.rms_norm_eps,
                epsilon: args.hc_eps,
                function: parameter(format!("{root}.hc_head_fn"))?,
                base: parameter(format!("{root}.hc_head_base"))?,
                scale: parameter(format!("{root}.hc_head_scale"))?,
            },
        })
    }
    pub(crate) fn instantiate<
        B: HyperNeuralBackend + eredu_nn::DistributedNeuralBackend + GroupedNeuralBackend,
    >(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<V4PredictionLayer<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, V4PredictionLayer<B>, &Self)>()?;
        Ok(V4PredictionLayer {
            embedding_projection: B::linear(
                copy_linear::<B>(&self.embedding_projection, context)?,
                context,
            )?,
            hidden_projection: B::linear(
                copy_linear::<B>(&self.hidden_projection, context)?,
                context,
            )?,
            embedding_norm: B::normalization(
                copy_normalization::<B>(&self.embedding_norm, context)?,
                context,
            )?,
            hidden_norm: B::normalization(
                copy_normalization::<B>(&self.hidden_norm, context)?,
                context,
            )?,
            decoder: self.decoder.instantiate::<B>(context)?,
            output_norm: B::normalization(
                copy_normalization::<B>(&self.output_norm, context)?,
                context,
            )?,
            hyper_head: HyperHead::new(copy_hyper_head::<B>(&self.hyper_head, context)?, context)?,
        })
    }
}
