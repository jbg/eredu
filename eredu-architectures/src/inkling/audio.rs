//! Neutral Inkling dMel audio projection.

use eredu_nn::{
    EmbeddingOperator, EmbeddingSpec, Error, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, ParameterSpec, Parameterized, Tensor,
};

use super::AudioConfig;

/// Prepared dMel code IDs and host-known valid-frame count.
pub struct AudioInput<'a, T> {
    /// Code IDs shaped `[1, frames, codebooks]`.
    pub code_ids: &'a T,
    /// Valid prefix length after processor padding.
    pub valid_frames: i32,
}

/// Codebook-offset embedding sum followed by learned RMS normalization.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct AudioTower<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    embedding: B::Embedding,
    final_norm: B::Normalization,
    #[parameter(skip, metadata)]
    num_codebooks: i32,
    #[parameter(skip, metadata)]
    codebook_size: i32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> AudioTower<B> {
    /// Builds the unloaded native dMel tower.
    pub fn new(
        config: &AudioConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        AudioTowerSpec::new(config)?.instantiate::<B>(context)
    }

    /// Embeds every codebook, sums the codebook axis, normalizes, and crops padding.
    pub fn forward(
        &mut self,
        input: AudioInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let frames = input.code_ids.shape().get(1).copied().unwrap_or(0);
        if input.code_ids.shape() != [1, frames, self.num_codebooks]
            || input.valid_frames < 0
            || input.valid_frames > frames
        {
            return Err(Error::backend("invalid Inkling prepared dMel geometry"));
        }
        let offsets = (0..self.num_codebooks)
            .map(|codebook| codebook * self.codebook_size)
            .collect::<Vec<_>>();
        let offsets = B::Tensor::from_i32_slice(&offsets, &[1, 1, self.num_codebooks], context)?;
        let indices = input.code_ids.add(&offsets, context)?;
        let embedded = self.embedding.forward(&indices, context)?;
        let embedded = B::Tensor::sum_axis(&embedded, -2, false, context)?;
        let embedded = self.final_norm.forward(&embedded, context)?;
        embedded.index(
            &[
                eredu_nn::Index::Full,
                eredu_nn::Index::Range(0, input.valid_frames),
                eredu_nn::Index::Full,
            ],
            context,
        )
    }
}

/// Immutable declarations produced once by the same ordinary audio constructor.
#[derive(Debug)]
pub(crate) struct AudioTowerSpec {
    embedding: EmbeddingSpec,
    final_norm: NormalizationConstructionSpec,
    num_codebooks: i32,
    codebook_size: i32,
}
impl AudioTowerSpec {
    pub(crate) fn new(config: &AudioConfig) -> Result<Self, Error> {
        let name = "audio.encoder.weight";
        Ok(Self {
            embedding: EmbeddingSpec {
                vocabulary: config.num_codebooks * config.codebook_size,
                dimensions: config.text_hidden_size,
                weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                format: crate::linear_format::standard_linear_format(name, config.linear_format_for(name))?,
            },
            final_norm: NormalizationConstructionSpec::learned(config.text_hidden_size, config.rms_norm_eps,
                ParameterSpec::trainable("audio.final_norm.weight").map_err(Error::backend)?),
            num_codebooks: config.num_codebooks,
            codebook_size: config.codebook_size,
        })
    }
    pub(crate) fn instantiate<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
        &self, context: &<B::Tensor as Tensor>::Context,
    ) -> Result<AudioTower<B>, Error> {
        use crate::decoder::construction_specs::{copy_embedding, copy_normalization};
        crate::decoder::ModuleMetadata::new::<B>(context).controls::<(&Self, AudioTower<B>)>()?;
        Ok(AudioTower {
            embedding: B::embedding(copy_embedding::<B>(&self.embedding, context)?, context)?,
            final_norm: B::normalization(copy_normalization::<B>(&self.final_norm, context)?, context)?,
            num_codebooks: self.num_codebooks,
            codebook_size: self.codebook_size,
        })
    }
}
