//! The actual immutable config and static declarations retained after initial construction.
use super::*;
mod units;
use crate::decoder::{
    construction_specs::{copy_embedding, copy_linear, copy_normalization},
    ModuleMetadata,
};
use crate::replicated_text::SharedCompositeConfig;
pub(super) use units::PreparedUnits;

#[derive(Clone, Debug)]
pub(crate) struct RetainedModelSource {
    args: SharedCompositeConfig<ModelArgs>,
    modules: SharedCompositeConfig<StaticModulesSpec>,
    units: Option<SharedCompositeConfig<PreparedUnits>>,
}
impl std::ops::Deref for RetainedModelSource {
    type Target = ModelArgs;
    fn deref(&self) -> &ModelArgs {
        &self.args
    }
}
impl RetainedModelSource {
    pub(super) fn prepare<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        args: ModelArgs,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        // Compiling new declarations belongs to the initial constructor. Checked
        // reconstruction must borrow the actual retained result instead.
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        let modules = StaticModulesSpec::new(&args)?;
        Ok(Self {
            args: SharedCompositeConfig::new(args, B::construction_metadata(context))?,
            units: None,
            modules: SharedCompositeConfig::new(modules, B::construction_metadata(context))?,
        })
    }
    pub(crate) fn matches_admission(&self, admission: &ModelArgs) -> bool {
        std::ptr::eq(&*self.args, admission)
    }
    pub(crate) fn has_units(&self) -> bool {
        self.units.is_some()
    }
    pub(super) fn units(&self) -> Option<&PreparedUnits> {
        self.units.as_deref()
    }
    pub(super) fn set_units(&mut self, units: SharedCompositeConfig<PreparedUnits>) {
        self.units = Some(units);
    }
    pub(super) fn clear_units(&mut self) {
        self.units = None;
    }
    pub(super) fn args(&self) -> &SharedCompositeConfig<ModelArgs> {
        &self.args
    }
    pub(super) fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        geometry: Option<&LocalGeometry>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<StaticModules<B>, Error> {
        self.modules.instantiate::<B>(geometry, context)
    }
}

#[derive(Debug)]
struct StaticModulesSpec {
    embeddings: EmbeddingSpec,
    embedding_norm: NormalizationConstructionSpec,
    final_norm: NormalizationConstructionSpec,
    output: LinearSpec,
    mtp: Option<super::super::mtp::MtpModelSpec>,
    audio: Option<super::super::audio::AudioTowerSpec>,
    vision: Option<super::super::vision::VisionStaticSpec>,
}
impl StaticModulesSpec {
    fn new(args: &ModelArgs) -> Result<Self, Error> {
        let text = &args.text_config;
        let norm = |name: &str| {
            Ok::<_, Error>(NormalizationConstructionSpec::learned(
                text.hidden_size,
                text.rms_norm_eps,
                ParameterSpec::trainable(name).map_err(Error::backend)?,
            ))
        };
        Ok(Self {
            embeddings: EmbeddingSpec {
                vocabulary: text.vocab_size,
                dimensions: text.hidden_size,
                weight: ParameterSpec::trainable("model.embed_tokens.weight")
                    .map_err(Error::backend)?,
                format: crate::linear_format::standard_linear_format(
                    "model.embed_tokens.weight",
                    text.linear_format_for("model.embed_tokens.weight"),
                )?,
            },
            embedding_norm: norm("model.embed_norm.weight")?,
            final_norm: norm("model.norm.weight")?,
            output: LinearSpec {
                input: text.hidden_size,
                output: text.vocab_size,
                weight: ParameterSpec::trainable("lm_head.weight").map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(
                    "lm_head.weight",
                    text.linear_format_for("lm_head.weight"),
                )?,
            },
            mtp: super::super::mtp::MtpModelSpec::new(args)?,
            audio: args
                .audio_config
                .as_ref()
                .map(super::super::audio::AudioTowerSpec::new)
                .transpose()?,
            vision: args
                .vision_config
                .as_ref()
                .map(super::super::vision::VisionStaticSpec::new)
                .transpose()?,
        })
    }
    fn instantiate<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self,
        geometry: Option<&LocalGeometry>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<StaticModules<B>, Error> {
        ModuleMetadata::new::<B>(context).controls::<(
            &Self,
            Option<&LocalGeometry>,
            StaticModules<B>,
            eredu_nn::VocabularyParallelRange,
        )>()?;
        let embeddings = copy_embedding::<B>(&self.embeddings, context)?;
        let embeddings = match geometry {
            Some(geometry) => B::vocabulary_parallel_embedding(
                embeddings,
                geometry.embedding_range().clone(),
                context,
            )?,
            None => B::embedding(embeddings, context)?,
        };
        let embedding_norm = B::normalization(
            copy_normalization::<B>(&self.embedding_norm, context)?,
            context,
        )?;
        let final_norm =
            B::normalization(copy_normalization::<B>(&self.final_norm, context)?, context)?;
        let output = copy_linear::<B>(&self.output, context)?;
        let output = match geometry {
            Some(geometry) => {
                B::vocabulary_parallel_linear(output, geometry.output_range().clone(), context)?
            }
            None => B::linear(output, context)?,
        };
        Ok(StaticModules {
            embeddings,
            embedding_norm,
            final_norm,
            output,
            mtp: self
                .mtp
                .as_ref()
                .map(|spec| spec.instantiate::<B>(context))
                .transpose()?,
            audio: self
                .audio
                .as_ref()
                .map(|spec| spec.instantiate::<B>(context))
                .transpose()?,
            vision: self
                .vision
                .as_ref()
                .map(|spec| spec.instantiate::<B>(context))
                .transpose()?,
        })
    }
}

#[cfg(test)]
mod tests;
