//! Neutral Inkling dMel audio projection.

use eredu_nn::{
    DeclareLogicalParameters, EmbeddingOperator, EmbeddingSpec, Error, LogicalParameterDeclaration,
    NeuralBackend, NormalizationConstructionSpec, NormalizationOperator, ParameterSpec,
    Parameterized, Tensor,
};

use super::{AudioConfig, AudioPolicy};

/// Construction-owned dMel parameter scope, independent of frames and code IDs.
#[derive(Debug, Clone)]
pub(crate) struct Parameters {
    pub embedding: EmbeddingSpec,
    pub final_norm: NormalizationConstructionSpec,
}
impl Parameters {
    pub(super) fn new(config: &AudioPolicy) -> Result<Self, Error> {
        if config.num_codebooks <= 0
            || config.codebook_size <= 0
            || config.bias
            || !config.use_audio_norm
            || config.audio_mode != "dmel"
        {
            return Err(Error::backend(
                "Inkling audio requires positive codebooks, unbiased dMel and output normalization",
            ));
        }
        let name = "audio.encoder.weight";
        let parameters = Self {
            embedding: EmbeddingSpec {
                vocabulary: config
                    .num_codebooks
                    .checked_mul(config.codebook_size)
                    .ok_or_else(|| Error::backend("Inkling audio vocabulary overflow"))?,
                dimensions: config.text_hidden_size,
                weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                format: crate::linear_format::standard_linear_format(
                    name,
                    config.linear_format_for(name),
                )?,
            },
            final_norm: NormalizationConstructionSpec::learned(
                config.text_hidden_size,
                config.rms_norm_eps,
                ParameterSpec::trainable("audio.final_norm.weight").map_err(Error::backend)?,
            ),
        };
        eredu_nn::validate_logical_parameter_topology(&parameters.logical_parameters()?)
            .map_err(Error::backend_source)?;
        parameters.validate_formats(config)?;
        Ok(parameters)
    }
    fn validate_formats(&self, config: &AudioPolicy) -> Result<(), Error> {
        crate::format_references::validate_matrices(
            self.logical_parameters()?,
            config
                .quantized_weight_configs
                .iter()
                .flat_map(|formats| formats.iter())
                .map(|(name, format)| (name.as_str(), Some((*format).into()))),
        )
    }
    pub(super) fn project_formats(&mut self, config: &AudioPolicy) -> Result<(), Error> {
        self.embedding.format = crate::linear_format::standard_linear_format(
            self.embedding.weight.id.as_str(),
            config.linear_format_for(self.embedding.weight.id.as_str()),
        )?;
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)?;
        self.validate_formats(config)
    }
}
impl DeclareLogicalParameters for Parameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.embedding.logical_parameters()?;
        parameters.extend(self.final_norm.logical_parameters()?);
        Ok(parameters)
    }
}

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
    #[parameter(skip)]
    num_codebooks: i32,
    #[parameter(skip)]
    codebook_size: i32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> AudioTower<B> {
    /// Builds the unloaded native dMel tower.
    pub fn new(
        config: &AudioConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let parameters = config.parameters();
        Ok(Self {
            embedding: B::embedding(parameters.embedding.clone(), context)?,
            final_norm: B::normalization(parameters.final_norm.clone(), context)?,
            num_codebooks: config.num_codebooks,
            codebook_size: config.codebook_size,
        })
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
