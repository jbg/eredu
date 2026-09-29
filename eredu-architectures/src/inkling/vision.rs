//! Neutral Inkling folded hMLP image tower.

use eredu_nn::{
    Error, LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, ParameterSpec, Parameterized, Tensor,
};

use super::VisionConfig;

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
/// One folded projection stage in the fixed hMLP image tower.
pub struct VisionLayer<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    projection: B::Linear,
    norm: Option<B::Normalization>,
    #[parameter(skip)]
    temporal_fold: i32,
    #[parameter(skip)]
    spatial_fold: i32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionLayer<B> {
    /// Builds one unloaded folded projection unit.
    pub fn new(
        config: &VisionConfig,
        layer: usize,
        spec: (i32, i32, i32, i32),
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let (input, output, temporal_fold, spatial_fold) = spec;
        let weight = format!("visual.layers.{layer}.projection.weight");
        Ok(Self {
            projection: B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &weight,
                        config.linear_format_for(&weight),
                    )?,
                },
                context,
            )?,
            norm: (layer + 1 != config.num_hidden_layers as usize)
                .then(|| {
                    B::normalization(
                        NormalizationConstructionSpec::learned(
                            output,
                            config.rms_norm_eps,
                            ParameterSpec::trainable(format!(
                                "visual.layers.{layer}.layer_norm.weight"
                            ))
                            .map_err(Error::backend)?,
                        ),
                        context,
                    )
                })
                .transpose()?,
            temporal_fold,
            spatial_fold,
        })
    }

    /// Folds and projects one hMLP stage.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let mut hidden = fold(input, self.temporal_fold, self.spatial_fold, context)?;
        hidden = self.projection.forward(&hidden, context)?;
        if let Some(norm) = self.norm.as_mut() {
            hidden = B::Tensor::gelu(&norm.forward(&hidden, context)?, context)?;
        }
        Ok(hidden)
    }
}

/// Pinned final normalization for the hMLP tower.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionStatic<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    final_norm: B::Normalization,
    #[parameter(skip)]
    hidden_size: i32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionStatic<B> {
    /// Builds the unloaded pinned image normalization.
    pub fn new(
        config: &VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            final_norm: B::normalization(
                NormalizationConstructionSpec::learned(
                    config.text_hidden_size,
                    config.rms_norm_eps,
                    ParameterSpec::trainable("visual.final_norm.weight").map_err(Error::backend)?,
                ),
                context,
            )?,
            hidden_size: config.text_hidden_size,
        })
    }

    /// Normalizes and flattens the final hMLP activation into decoder embeddings.
    pub fn finish(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.final_norm.forward(hidden, context)?;
        hidden.reshape(&[1, -1, self.hidden_size], context)
    }
}

/// Fixed four-layer Inkling hMLP image tower.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionTower<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Pinned final normalization.
    pub static_modules: VisionStatic<B>,
    /// Independently streamable folded projection stages.
    pub layers: Vec<VisionLayer<B>>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionTower<B> {
    /// Builds the released folded hMLP tower.
    pub fn new(
        config: &VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            static_modules: VisionStatic::new(config, context)?,
            layers: config
                .layer_specs()
                .into_iter()
                .enumerate()
                .map(|(layer, spec)| VisionLayer::new(config, layer, spec, context))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    /// Encodes patches shaped `[patches, 2, 40, 40, 3]` into decoder embeddings.
    pub fn forward(
        &mut self,
        patches: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if patches.shape().len() != 5 || patches.shape()[1..] != [2, 40, 40, 3] {
            return Err(Error::backend("invalid Inkling hMLP patch geometry"));
        }
        let mut hidden = patches.clone();
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, context)?;
        }
        self.static_modules.finish(&hidden, context)
    }
}

fn fold<T: Tensor>(
    input: &T,
    temporal: i32,
    spatial: i32,
    context: &T::Context,
) -> Result<T, Error> {
    if temporal == 1 && spatial == 1 {
        return Ok(input.clone());
    }
    let shape = input.shape();
    if shape.len() != 5
        || shape[1] % temporal != 0
        || shape[2] % spatial != 0
        || shape[3] % spatial != 0
    {
        return Err(Error::backend("invalid Inkling hMLP fold geometry"));
    }
    let (batch, time, height, width, channels) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    input
        .reshape(
            &[
                batch,
                time / temporal,
                temporal,
                height / spatial,
                spatial,
                width / spatial,
                spatial,
                channels,
            ],
            context,
        )?
        .transpose_axes(&[0, 1, 3, 5, 2, 4, 6, 7], context)?
        .reshape(
            &[
                batch,
                time / temporal,
                height / spatial,
                width / spatial,
                temporal * spatial * spatial * channels,
            ],
            context,
        )
}
