//! Retained prediction factories from the model's actual local geometry.
use super::*;
use crate::decoder::{ModuleMetadata, construction_specs::*};
#[derive(Debug)]
enum PredictionKind {
    Sequential(crate::deepseek::mtp::V4PredictionLayerSpec),
    Dspark(crate::deepseek::block::V4BlockSpec),
}
#[derive(Debug)]
pub(crate) struct V4PredictionUnitSpec {
    kind: PredictionKind,
    experts: Option<eredu_nn::GroupedGatedProductSpec>,
}
impl V4PredictionUnitSpec {
    fn new(
        args: &V4Args,
        depth: usize,
        experts: Option<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        let kind = if args.dspark.is_some() {
            PredictionKind::Dspark(crate::deepseek::block::V4BlockSpec::new(
                args,
                usize::try_from(args.num_hidden_layers).map_err(Error::backend)? + depth,
                &format!("mtp.{depth}"),
                None,
            )?)
        } else {
            PredictionKind::Sequential(crate::deepseek::mtp::V4PredictionLayerSpec::new(
                args, depth,
            )?)
        };
        Ok(Self { kind, experts })
    }
    pub(crate) fn fused(&self) -> bool {
        matches!(&self.kind, PredictionKind::Dspark(_))
    }
    #[inline(never)]
    pub(crate) fn instantiate<
        B: HyperNeuralBackend + eredu_nn::DistributedNeuralBackend + GroupedNeuralBackend,
    >(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(&Self, &<B::Tensor as Tensor>::Context)>()?;
        // Select before constructing either concrete module. The unused
        // sibling's return/cleanup storage must not occupy the nested quote.
        match &self.kind {
            PredictionKind::Sequential(spec)=>self.instantiate_sequential(spec,context),
            PredictionKind::Dspark(spec)=>self.instantiate_dspark(spec,context),
        }
    }
    #[inline(never)]
    fn instantiate_sequential<B: HyperNeuralBackend + eredu_nn::DistributedNeuralBackend + GroupedNeuralBackend>(
        &self,spec:&crate::deepseek::mtp::V4PredictionLayerSpec,
        context:&<B::Tensor as Tensor>::Context,
    )->Result<Unit<B>,Error>{
        ModuleMetadata::new::<B>(context).controls::<(Unit<B>,
            &Self,&crate::deepseek::mtp::V4PredictionLayerSpec,&<B::Tensor as Tensor>::Context)>()?;
        let mut unit=Unit::Prediction(spec.instantiate::<B>(context)?);
        self.bind_experts(&mut unit,context)?;
        Ok(unit)
    }
    #[inline(never)]
    fn instantiate_dspark<B: HyperNeuralBackend + eredu_nn::DistributedNeuralBackend + GroupedNeuralBackend>(
        &self,spec:&crate::deepseek::block::V4BlockSpec,
        context:&<B::Tensor as Tensor>::Context,
    )->Result<Unit<B>,Error>{
        ModuleMetadata::new::<B>(context).controls::<(Unit<B>,
            &Self,&crate::deepseek::block::V4BlockSpec,&<B::Tensor as Tensor>::Context)>()?;
        let mut unit=Unit::Dspark(spec.instantiate::<B>(context)?);
        self.bind_experts(&mut unit,context)?;
        Ok(unit)
    }
    #[inline(never)]
    fn bind_experts<B: HyperNeuralBackend + eredu_nn::DistributedNeuralBackend + GroupedNeuralBackend>(
        &self,unit:&mut Unit<B>,context:&<B::Tensor as Tensor>::Context,
    )->Result<(),Error>{
        ModuleMetadata::new::<B>(context).controls::<(&Self,&mut Unit<B>,
            &<B::Tensor as Tensor>::Context,Result<(),Error>)>()?;
        if let Some(spec) = &self.experts {
            let ff = match unit {
                Unit::Prediction(p) => &mut p.decoder.feed_forward,
                Unit::Dspark(b) => &mut b.feed_forward,
                Unit::Target(_) => unreachable!(),
            };
            ff.experts = B::grouped_gated_product(copy_grouped::<B>(spec, context)?, context)?;
        }
        Ok(())
    }

}
impl<B: HyperNeuralBackend + eredu_nn::DistributedNeuralBackend + GroupedNeuralBackend> Model<B> {
    pub(crate) fn prediction_unit_spec(&self, depth: usize) -> Result<V4PredictionUnitSpec, Error> {
        self.groups.unit_count(depth + 1)?;
        let args = self
            .parallel_geometry
            .as_ref()
            .map_or(&*self.args, |geometry| geometry.args());
        let experts = match &self.expert_realization {
            Some(realization) => Some(
                realization
                    .unit_spec(&format!("mtp.{depth}"), 0)
                    .ok_or_else(|| {
                        Error::backend(format!(
                            "V4 expert realization has no bank for mtp.{depth}.0"
                        ))
                    })?
                    .clone(),
            ),
            None => None,
        };
        V4PredictionUnitSpec::new(args, depth, experts)
    }
    pub(crate) fn dspark_static_spec(&self) -> Result<Option<DsparkStaticSpec>, Error> {
        self.args
            .dspark
            .as_ref()
            .map(|config| DsparkStaticSpec::new(&self.args, config))
            .transpose()
    }
}
#[derive(Debug)]
pub(crate) struct DsparkStaticSpec {
    main_projection: LinearSpec,
    main_norm: NormalizationConstructionSpec,
    output_norm: NormalizationConstructionSpec,
    hyper_head: HyperHeadSpec,
    markov_embedding: EmbeddingSpec,
    markov_output: LinearSpec,
    confidence_head: LinearSpec,
}
impl DsparkStaticSpec {
    pub(super) fn new(args: &V4Args, config: &DsparkConfig) -> Result<Self, Error> {
        let last = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)? - 1;
        let norm = |name: String| -> Result<_, Error> {
            Ok(NormalizationConstructionSpec::learned(
                args.hidden_size,
                args.rms_norm_eps,
                parameter(name)?,
            ))
        };
        let projection = |name: String, input, output| -> Result<_, Error> {
            Ok(LinearSpec {
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
            main_projection: projection(
                "mtp.0.main_proj.weight".into(),
                args.hidden_size
                    * i32::try_from(
                        args.target_capture_policy
                            .as_ref()
                            .expect("validated DSpark target capture policy")
                            .len(),
                    )
                    .map_err(Error::backend)?,
                args.hidden_size,
            )?,
            main_norm: norm("mtp.0.main_norm.weight".into())?,
            output_norm: norm(format!("mtp.{last}.norm.weight"))?,
            hyper_head: HyperHeadSpec {
                streams: args.hc_mult,
                hidden_size: args.hidden_size,
                norm_epsilon: args.rms_norm_eps,
                epsilon: args.hc_eps,
                function: parameter(format!("mtp.{last}.hc_head_fn"))?,
                base: parameter(format!("mtp.{last}.hc_head_base"))?,
                scale: parameter(format!("mtp.{last}.hc_head_scale"))?,
            },
            markov_embedding: EmbeddingSpec {
                vocabulary: args.vocab_size,
                dimensions: config.markov_rank,
                weight: parameter(format!("mtp.{last}.markov_head.markov_w1.weight"))?,
                format: crate::linear_format::standard_linear_format(
                    &format!("mtp.{last}.markov_head.markov_w1.weight"),
                    LinearFormat::Dense,
                )?,
            },
            markov_output: projection(
                format!("mtp.{last}.markov_head.markov_w2.weight"),
                config.markov_rank,
                args.vocab_size,
            )?,
            confidence_head: projection(
                format!("mtp.{last}.confidence_head.proj.weight"),
                args.hidden_size + config.markov_rank,
                1,
            )?,
        })
    }
    pub(crate) fn instantiate<B: HyperNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<DsparkStatic<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(Self, DsparkStatic<B>, &Self)>()?;
        Ok(DsparkStatic {
            main_projection: B::linear(copy_linear::<B>(&self.main_projection, context)?, context)?,
            main_norm: B::normalization(
                copy_normalization::<B>(&self.main_norm, context)?,
                context,
            )?,
            output_norm: B::normalization(
                copy_normalization::<B>(&self.output_norm, context)?,
                context,
            )?,
            hyper_head: HyperHead::new(copy_hyper_head::<B>(&self.hyper_head, context)?, context)?,
            markov_embedding: B::embedding(
                copy_embedding::<B>(&self.markov_embedding, context)?,
                context,
            )?,
            markov_output: B::linear(copy_linear::<B>(&self.markov_output, context)?, context)?,
            confidence_head: B::linear(copy_linear::<B>(&self.confidence_head, context)?, context)?,
        })
    }
}

#[cfg(test)]
mod tests;
