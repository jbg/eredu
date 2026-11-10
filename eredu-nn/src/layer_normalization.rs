//! Typed affine layer-normalization construction and logical parameter geometry.

use crate::{
    DeclareLogicalParameters, Error, LayerNormalizationArithmetic, LogicalParameterDeclaration,
    LogicalParameterRole, ParameterSpec,
};

/// Affine layer normalization over the final tensor axis.
#[derive(Debug, Clone)]
pub struct LayerNormalizationConstructionSpec {
    /// Width normalized by one invocation.
    pub dimensions: i32,
    /// Positive additive variance epsilon.
    pub epsilon: f32,
    /// Optional learned multiplicative gain.
    pub weight: Option<ParameterSpec>,
    /// Optional learned additive offset.
    pub bias: Option<ParameterSpec>,
    /// Required moment and affine arithmetic.
    pub arithmetic: LayerNormalizationArithmetic,
}
impl LayerNormalizationConstructionSpec {
    /// Validates dimensions, epsilon, and logical slot geometry before allocation.
    pub fn validate(&self) -> Result<(), Error> {
        if self.dimensions <= 0 || !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::backend(
                "invalid layer normalization geometry or epsilon",
            ));
        }
        self.logical_parameters().map(|_| ())
    }
}
impl DeclareLogicalParameters for LayerNormalizationConstructionSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::backend(
                "layer normalization epsilon must be positive",
            ));
        }
        let mut parameters = Vec::new();
        for (parameter, role) in [
            (&self.weight, LogicalParameterRole::Normalization),
            (&self.bias, LogicalParameterRole::NormalizationBias),
        ] {
            if let Some(parameter) = parameter {
                parameters.push(LogicalParameterDeclaration::new(
                    parameter,
                    &[self.dimensions],
                    role,
                )?);
            }
        }
        if self.dimensions <= 0 {
            return Err(Error::backend("layer normalization width must be positive"));
        }
        Ok(parameters)
    }
}
