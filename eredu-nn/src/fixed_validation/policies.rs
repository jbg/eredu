//! Fixed policy causes; ordinary diagnostics format the same borrowed source.
use crate::{
    EmbeddingLookupPolicy, Error, HyperConnectionSpec, HyperHeadSpec, LinearWeightValidationError,
    NormalizationConstructionSpec, NormalizationScale, RotaryAlgorithm, TopKGroupSelectorSpec,
};
/// A zero sentinel would alias an ordinary nonnegative token row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("embedding zero sentinel must be negative, got {0}")]
pub struct EmbeddingValidationError(pub i32);
/// RMS construction has invalid dimensions, grouping, epsilon or offset.
/// Scalar bit patterns retain the exact ordinary diagnostic, including NaNs,
/// without borrowing the construction or allocating a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizationValidationError {
    dimensions: i32,
    epsilon_bits: u32,
    offset_bits: Option<u32>,
}
impl std::fmt::Display for NormalizationValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let epsilon = f32::from_bits(self.epsilon_bits);
        let offset = self.offset_bits.map(f32::from_bits);
        write!(
            f,
            "invalid RMS normalization construction: dimensions={} epsilon={} offset={offset:?}",
            self.dimensions, epsilon
        )
    }
}
impl std::error::Error for NormalizationValidationError {}
impl NormalizationValidationError {
    pub(crate) fn into_ordinary(self, _: &NormalizationConstructionSpec) -> Error {
        Error::backend(self)
    }
}
/// A normalized rotary algorithm has invalid scalar policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("invalid normalized rotary algorithm")]
pub struct RotaryValidationError;
/// An ordered hyper-connection construction check failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum HyperConnectionValidationError {
    /// Nonpositive residual-stream count or hidden width.
    #[error("hyper-connection streams and hidden size must be positive")]
    Geometry,
    /// No Sinkhorn normalization passes were declared.
    #[error("hyper-connection Sinkhorn iteration count must be positive")]
    Iterations,
    /// Nonfinite or nonpositive numerical epsilon.
    #[error("hyper-connection epsilon must be finite and positive")]
    Epsilon,
}
/// An ordered final-stream-collapse check failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum HyperHeadValidationError {
    /// Nonpositive residual-stream count or hidden width.
    #[error("hyper-head streams and hidden size must be positive")]
    Geometry,
    /// A normalization or coefficient epsilon is nonfinite or nonpositive.
    #[error("hyper-head epsilons must be finite and positive")]
    Epsilon,
}
/// An ordered selector construction check failed without cloning its identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SelectorValidationError {
    /// Encoding or primary-weight identity failure.
    #[error(transparent)]
    Format(#[from] LinearWeightValidationError),
    /// Nonpositive input width.
    #[error("selector input dimensions must be positive, got {0}")]
    Input(i32),
    /// The optional input RMS epsilon is negative or nonfinite.
    #[error("selector input RMS epsilon must be finite and nonnegative")]
    Epsilon,
    /// Learned projection and correction bias identities coincide.
    #[error("selector projection bias and correction bias require distinct parameter identities")]
    BiasIdentity,
}
impl SelectorValidationError {
    pub(crate) fn into_ordinary(self, spec: &TopKGroupSelectorSpec) -> Error {
        match self {
            Self::Format(cause) => cause.into_ordinary(&spec.weight),
            other => Error::backend(other),
        }
    }
}
impl EmbeddingLookupPolicy {
    /// Validates the same source policy without allocating a diagnostic.
    pub fn validate_fixed(self) -> Result<(), EmbeddingValidationError> {
        if let Self::ZeroSentinel(sentinel) = self {
            if sentinel >= 0 {
                return Err(EmbeddingValidationError(sentinel));
            }
        }
        Ok(())
    }
}

impl NormalizationConstructionSpec {
    /// Validates the same source policy without allocating a diagnostic.
    pub fn validate_fixed(&self) -> Result<(), NormalizationValidationError> {
        let offset = match &self.scale {
            NormalizationScale::LearnedOffset { offset, .. } => Some(*offset),
            NormalizationScale::Learned(_) | NormalizationScale::Unit => None,
        };
        if self.dimensions <= 0
            || self
                .groups
                .is_some_and(|groups| groups <= 0 || self.dimensions % groups != 0)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || offset.is_some_and(|offset| !offset.is_finite())
        {
            return Err(NormalizationValidationError {
                dimensions: self.dimensions,
                epsilon_bits: self.epsilon.to_bits(),
                offset_bits: offset.map(f32::to_bits),
            });
        }
        Ok(())
    }
}

impl RotaryAlgorithm {
    /// Validates the same source policy without allocating a diagnostic.
    pub fn validate_fixed(self) -> Result<(), RotaryValidationError> {
        let positive = |value: f32| value.is_finite() && value > 0.0;
        let valid = match self {
            Self::Default => true,
            Self::Linear { factor } => positive(factor),
            Self::Llama3 {
                factor,
                low_frequency_factor,
                high_frequency_factor,
                original_max_positions,
            } => {
                positive(factor)
                    && positive(low_frequency_factor)
                    && positive(high_frequency_factor)
                    && high_frequency_factor > low_frequency_factor
                    && original_max_positions > 0
            }
            Self::Proportional {
                factor,
                rotary_fraction,
            } => positive(factor) && positive(rotary_fraction) && rotary_fraction <= 1.0,
            Self::Yarn {
                factor,
                original_max_positions,
                beta_fast,
                beta_slow,
                amplitude,
                ..
            } => {
                positive(factor)
                    && original_max_positions > 0
                    && positive(beta_fast)
                    && positive(beta_slow)
                    && beta_fast > beta_slow
                    && positive(amplitude)
            }
        };
        if valid {
            Ok(())
        } else {
            Err(RotaryValidationError)
        }
    }
}

impl HyperConnectionSpec {
    /// Validates the same source policy without allocating a diagnostic.
    pub fn validate_fixed(&self) -> Result<(), HyperConnectionValidationError> {
        if self.streams <= 0 || self.hidden_size <= 0 {
            return Err(HyperConnectionValidationError::Geometry);
        }
        if self.sinkhorn_iterations == 0 {
            return Err(HyperConnectionValidationError::Iterations);
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(HyperConnectionValidationError::Epsilon);
        }
        Ok(())
    }
}

impl HyperHeadSpec {
    /// Validates the same source policy without allocating a diagnostic.
    pub fn validate_fixed(&self) -> Result<(), HyperHeadValidationError> {
        if self.streams <= 0 || self.hidden_size <= 0 {
            return Err(HyperHeadValidationError::Geometry);
        }
        if !self.norm_epsilon.is_finite()
            || self.norm_epsilon <= 0.0
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
        {
            return Err(HyperHeadValidationError::Epsilon);
        }
        Ok(())
    }
}

impl TopKGroupSelectorSpec {
    /// Validates the same source policy without allocating a diagnostic.
    pub fn validate_fixed(&self) -> Result<(), SelectorValidationError> {
        self.format.validate_for_weight_fixed(&self.weight)?;
        if self.input_dimensions <= 0 {
            return Err(SelectorValidationError::Input(self.input_dimensions));
        }
        if self
            .input_transform
            .as_ref()
            .is_some_and(|transform| !transform.epsilon.is_finite() || transform.epsilon < 0.0)
        {
            return Err(SelectorValidationError::Epsilon);
        }
        if self
            .bias
            .as_ref()
            .zip(self.correction_bias.as_ref())
            .is_some_and(|(bias, correction_bias)| bias.id == correction_bias.id)
        {
            return Err(SelectorValidationError::BiasIdentity);
        }
        Ok(())
    }
}
