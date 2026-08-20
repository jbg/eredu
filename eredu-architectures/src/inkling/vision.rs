//! Neutral Inkling folded hMLP image tower.

use eredu_nn::{
    Error, LinearOperator, LinearSpec, NeuralBackend, NormalizationOperator, NormalizationSpec,
    ParameterSpec, Parameterized, Tensor,
};

use super::VisionConfig;

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
/// One folded projection stage in the fixed hMLP image tower.
pub struct VisionLayer<B: NeuralBackend> {
    projection: B::Linear,
    norm: Option<B::Normalization>,
    #[parameter(skip)]
    temporal_fold: i32,
    #[parameter(skip)]
    spatial_fold: i32,
}

impl<B: NeuralBackend> VisionLayer<B> {
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
                    format: config.linear_format_for(&weight),
                },
                context,
            )?,
            norm: (layer + 1 != config.num_hidden_layers as usize)
                .then(|| {
                    B::rms_norm(
                        NormalizationSpec {
                            dimensions: output,
                            epsilon: config.rms_norm_eps,
                            weight: ParameterSpec::trainable(format!(
                                "visual.layers.{layer}.layer_norm.weight"
                            ))
                            .map_err(Error::backend)?,
                        },
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
pub struct VisionStatic<B: NeuralBackend> {
    final_norm: B::Normalization,
    #[parameter(skip)]
    hidden_size: i32,
}

impl<B: NeuralBackend> VisionStatic<B> {
    /// Builds the unloaded pinned image normalization.
    pub fn new(
        config: &VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            final_norm: B::rms_norm(
                NormalizationSpec {
                    dimensions: config.text_hidden_size,
                    epsilon: config.rms_norm_eps,
                    weight: ParameterSpec::trainable("visual.final_norm.weight")
                        .map_err(Error::backend)?,
                },
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
        let hidden = self.final_norm.forward(&hidden, context)?;
        hidden.reshape(&[1, -1, self.hidden_size], context)
    }
}

/// Fixed four-layer Inkling hMLP image tower.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionTower<B: NeuralBackend> {
    /// Pinned final normalization.
    pub static_modules: VisionStatic<B>,
    /// Independently streamable folded projection stages.
    pub layers: Vec<VisionLayer<B>>,
}

impl<B: NeuralBackend> VisionTower<B> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct ShapeTensor(Vec<i32>);

    impl Tensor for ShapeTensor {
        type Context = ();
        fn shape(&self) -> &[i32] {
            &self.0
        }
        fn unloaded_f32(_: &[i32], _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn from_f32_slice(_: &[f32], shape: &[i32], _: &()) -> Result<Self, Error> {
            Ok(Self(shape.into()))
        }
        fn add(&self, _: &Self, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn subtract(&self, _: &Self, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn multiply(&self, _: &Self, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn multiply_scalar(&self, _: f32, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn divide(&self, _: &Self, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn square(&self, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn maximum_scalar(&self, _: f32, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn reshape(&self, shape: &[i32], _: &()) -> Result<Self, Error> {
            let mut shape = shape.to_vec();
            if let Some(index) = shape.iter().position(|value| *value == -1) {
                let total: i32 = self.0.iter().product();
                let known: i32 = shape.iter().filter(|value| **value != -1).product();
                shape[index] = total / known;
            }
            Ok(Self(shape))
        }
        fn transpose_axes(&self, axes: &[i32], _: &()) -> Result<Self, Error> {
            Ok(Self(
                axes.iter().map(|axis| self.0[*axis as usize]).collect(),
            ))
        }
        fn swap_axes(&self, _: i32, _: i32, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn transpose(&self, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn expand_dims(&self, _: i32, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn squeeze_axes(&self, _: &[i32], _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn index(&self, _: &[eredu_nn::Index], _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn take_axis(&self, _: &Self, _: i32, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn concatenate(_: &[Self], _: i32, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn stack(_: &[Self], _: i32, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn matmul(_: &Self, _: &Self, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn sum_axis(_: &Self, _: i32, _: bool, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn argmin_axis(_: &Self, _: i32, _: bool, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn pad(_: &Self, _: &[(i32, i32)], _: eredu_nn::PadMode, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn conv1d(
            _: &Self,
            _: &Self,
            _: i32,
            _: i32,
            _: i32,
            _: i32,
            _: &(),
        ) -> Result<Self, Error> {
            unreachable!()
        }
        fn conv_transpose1d(
            _: &Self,
            _: &Self,
            _: i32,
            _: i32,
            _: i32,
            _: i32,
            _: i32,
            _: &(),
        ) -> Result<Self, Error> {
            unreachable!()
        }
        fn linear(_: &Self, _: &Self, _: Option<&Self>, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn layer_norm(
            _: &Self,
            _: Option<&Self>,
            _: Option<&Self>,
            _: f32,
            _: &(),
        ) -> Result<Self, Error> {
            unreachable!()
        }
        fn gelu(_: &Self, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn elu(_: &Self, _: f32, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn rope(_: &Self, _: i32, _: bool, _: f32, _: f32, _: i32, _: &()) -> Result<Self, Error> {
            unreachable!()
        }
        fn scaled_dot_product_attention(
            _: &Self,
            _: &Self,
            _: &Self,
            _: f32,
            _: eredu_nn::AttentionMask<'_, Self>,
            _: &(),
        ) -> Result<Self, Error> {
            unreachable!()
        }
    }

    #[test]
    fn fold_matches_released_axis_order() {
        let folded = fold(&ShapeTensor(vec![2, 2, 40, 40, 3]), 1, 5, &()).unwrap();
        assert_eq!(folded.shape(), [2, 2, 8, 8, 75]);
        let folded = fold(&ShapeTensor(vec![2, 2, 1, 1, 4800]), 2, 1, &()).unwrap();
        assert_eq!(folded.shape(), [2, 1, 1, 1, 9600]);
    }
}
