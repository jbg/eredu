//! Shared normalized low-rank projection assembly for DeepSeek attention.

use eredu_checkpoint::LinearFormat;
use eredu_nn::{
    Error, LinearSpec, LowRankProjection, LowRankProjectionSpec, NeuralBackend, NormalizationSpec,
    ParameterSpec, Tensor,
};

/// Architecture-owned identities and geometry for one normalized low-rank
/// projection. V3 query/MLA and V4 query/output projections use this policy.
#[derive(Debug, Clone)]
pub struct ProjectionPolicy {
    /// Optional input-to-rank projection identity.
    pub first_weight: Option<String>,
    /// Rank-space normalization identity.
    pub normalization_weight: String,
    /// Rank-to-output projection identity.
    pub second_weight: String,
    /// Input hidden width.
    pub input_dimensions: i32,
    /// Normalized rank width.
    pub rank: i32,
    /// Final output width.
    pub output_dimensions: i32,
    /// RMS normalization epsilon.
    pub epsilon: f32,
    /// Physical format of the optional first projection.
    pub first_format: LinearFormat,
    /// Physical format of the second projection.
    pub second_format: LinearFormat,
}

impl ProjectionPolicy {
    /// Builds the shared statically dispatched neural layer.
    pub fn build<B: NeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LowRankProjection<B>, Error> {
        let linear = |weight: &str, input, output, format| -> Result<LinearSpec, Error> {
            Ok(LinearSpec {
                input,
                output,
                weight: ParameterSpec::trainable(weight).map_err(Error::backend)?,
                bias: None,
                format: crate::linear_format::standard_linear_format(weight, format)?,
            })
        };
        LowRankProjection::new(
            LowRankProjectionSpec {
                first: self
                    .first_weight
                    .as_deref()
                    .map(|weight| {
                        linear(weight, self.input_dimensions, self.rank, self.first_format)
                    })
                    .transpose()?,
                normalization: NormalizationSpec {
                    dimensions: self.rank,
                    epsilon: self.epsilon,
                    weight: ParameterSpec::trainable(&self.normalization_weight)
                        .map_err(Error::backend)?,
                },
                second: linear(
                    &self.second_weight,
                    self.rank,
                    self.output_dimensions,
                    self.second_format,
                )?,
            },
            context,
        )
    }
}
