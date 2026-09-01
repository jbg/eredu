//! Gemma 4 media-to-text projection shared by image and audio features.

use eredu_nn::{
    Error, LinearOperator, LinearSpec, NeuralBackend, ParameterSpec, Parameterized, Tensor,
};

use super::ModelArgs;

/// Weightless RMS normalization followed by a learned media-to-text projection.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct ModalityProjector<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    projection: B::Linear,
    #[parameter(skip)]
    epsilon: f32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> ModalityProjector<B> {
    /// Builds an unloaded image or audio projector under `model.<component>`.
    pub fn new(
        args: &ModelArgs,
        component: &str,
        input_size: i32,
        epsilon: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if input_size <= 0 || !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::backend(
                "invalid Gemma 4 modality projector geometry",
            ));
        }
        let weight = format!("model.{component}.embedding_projection.weight");
        Ok(Self {
            projection: B::linear(
                LinearSpec {
                    input: input_size,
                    output: args.hidden_size,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &weight,
                        args.linear_format_for(&weight),
                    )?,
                },
                context,
            )?,
            epsilon,
        })
    }

    /// Normalizes encoder features and projects them into decoder hidden space.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let normalized = B::rms_norm_without_weight(input, self.epsilon, context)?;
        self.projection.forward(&normalized, context)
    }
}
