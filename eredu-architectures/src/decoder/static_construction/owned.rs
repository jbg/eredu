//! Owning destinations for the borrowed static-module declaration.
use super::{ModuleMetadata, StaticModuleSpec, StaticModuleSpecView};
use crate::decoder::Config;
use eredu_nn::{Error, NeuralBackend, Tensor, workspace::WorkspaceContext};

impl StaticModuleSpecView<'_> {
    fn map_names<E>(
        self,
        mut own: impl FnMut(&str) -> Result<String, E>,
    ) -> Result<StaticModuleSpec, E> {
        Ok(StaticModuleSpec {
            normalization_groups: self.normalization_groups,
            embedding_weight: own(self.embedding_weight)?,
            normalization_weight: own(self.normalization_weight)?,
            head_weight: own(self.head_weight)?,
            vocabulary: self.vocabulary,
            hidden_size: self.hidden_size,
            normalization_epsilon: self.normalization_epsilon,
            normalization_offset: self.normalization_offset,
            embedding_quantization: self.embedding_quantization,
            head_format: self.head_format,
            tied_head: self.tied_head,
        })
    }

    /// Owns the three names using the ordinary destination.
    pub fn to_owned(self) -> StaticModuleSpec {
        match self.map_names(|name| Ok::<_, std::convert::Infallible>(name.to_owned())) {
            Ok(spec) => spec,
            Err(never) => match never {},
        }
    }

    /// Owns the same names after charging their actual constructor and buffers.
    /// The context's funding must remain in the enclosing prepared owner.
    pub fn to_owned_with_metadata(
        self,
        context: &WorkspaceContext,
    ) -> Result<StaticModuleSpec, Error> {
        context.charge_metadata(
            size_of::<Self>()
                + size_of::<StaticModuleSpec>()
                + size_of::<Result<StaticModuleSpec, Error>>()
                + size_of::<&WorkspaceContext>(),
        )?;
        self.map_names(|name| context.metadata_string(format_args!("{name}")))
    }
}

pub(in crate::decoder) fn from_config<B: NeuralBackend, C: Config>(
    config: &C,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<StaticModuleSpec, Error> {
    let metadata = ModuleMetadata::new::<B>(context);
    metadata.controls::<(
        StaticModuleSpec,
        String,
        String,
        &C,
        Option<eredu_checkpoint::WeightQuantization>,
        eredu_checkpoint::LinearFormat,
    )>()?;
    let embedding_name = metadata.text(format_args!(
        "{}.embed_tokens.weight",
        config.parameter_root()
    ))?;
    let norm_name = metadata.text(format_args!("{}.norm.weight", config.parameter_root()))?;
    Ok(StaticModuleSpec {
        normalization_groups: config.normalization_groups(),
        embedding_weight: metadata.text(format_args!("{embedding_name}"))?,
        normalization_weight: norm_name,
        head_weight: metadata.text(format_args!("lm_head.weight"))?,
        vocabulary: config.vocabulary_size(),
        hidden_size: config.hidden_size(),
        normalization_epsilon: config.rms_norm_epsilon(),
        normalization_offset: config.normalization_offset(),
        embedding_quantization: metadata.weight_quantization(config, &embedding_name)?,
        head_format: metadata.linear_format(config, "lm_head.weight")?,
        tied_head: config.tie_word_embeddings(),
    })
}
