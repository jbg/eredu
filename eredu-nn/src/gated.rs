//! Split and component-major fused affine declarations for gated products.

use crate::{
    DeclareLogicalParameters, Error, FusedProjectionLayout, FusedProjectionSegment,
    GatedProductPolicy, LinearSpec, LogicalParameterDeclaration, SplitGatedProductSpec,
};

/// One component-major gate/up affine followed by an output projection.
#[derive(Debug, Clone)]
pub struct FusedGatedProductSpec {
    /// Gate rows followed by value rows of equal width.
    pub gate_up: LinearSpec,
    /// Intermediate-to-output projection.
    pub down: LinearSpec,
    /// Activation, clipping and scaling before the output projection.
    pub policy: GatedProductPolicy,
}
impl FusedGatedProductSpec {
    /// Exact component geometry, derived from the output projection's input width.
    pub fn layout(&self) -> Result<FusedProjectionLayout, Error> {
        let layout = FusedProjectionLayout::new([
            FusedProjectionSegment::new("gate", self.down.input)?,
            FusedProjectionSegment::new("up", self.down.input)?,
        ])?;
        if self.gate_up.output != layout.output_width() {
            return Err(Error::backend("incompatible fused gated-product geometry"));
        }
        Ok(layout)
    }
}
impl DeclareLogicalParameters for FusedGatedProductSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.policy.validate()?;
        self.layout()?;
        let mut parameters = self.gate_up.logical_parameters()?;
        parameters.extend(self.down.logical_parameters()?);
        Ok(parameters)
    }
}

/// Complete affine slots and equation policy for a split or fused gated product.
/// Sharing resolves in the enclosing logical scope; invocations remain separate.
#[derive(Debug, Clone)]
pub enum GatedProductSpec {
    /// Independent gate and value projections.
    Split(SplitGatedProductSpec),
    /// Component-major gate/value rows in one physical projection.
    Fused(FusedGatedProductSpec),
}
impl GatedProductSpec {
    /// Validates a complete logical scope, including identities and sharing targets.
    pub fn validate(&self) -> Result<(), Error> {
        crate::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }

    /// Selects local intermediate channels while retaining identities, formats and arithmetic.
    pub fn with_intermediate_dimensions(&self, width: i32) -> Result<Self, Error> {
        crate::validate_logical_parameter_scope(&self.logical_parameters()?)
            .map_err(Error::backend_source)?;
        if width <= 0 || width > self.down().input {
            return Err(Error::backend("local gated width exceeds its declaration"));
        }
        let mut projected = self.clone();
        match &mut projected {
            Self::Split(spec) => {
                spec.gate.output = width;
                spec.up.output = width;
                spec.down.input = width;
            }
            Self::Fused(spec) => {
                spec.gate_up.output = width
                    .checked_mul(2)
                    .ok_or_else(|| Error::backend("local fused gated width overflows"))?;
                spec.down.input = width;
            }
        }
        crate::validate_logical_parameter_scope(&projected.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(projected)
    }

    /// Affine declarations in gate/up/down or fused/down allocation order.
    pub fn projections(&self) -> impl Iterator<Item = &LinearSpec> {
        match self {
            Self::Split(spec) => [Some(&spec.gate), Some(&spec.up), Some(&spec.down)],
            Self::Fused(spec) => [Some(&spec.gate_up), Some(&spec.down), None],
        }
        .into_iter()
        .flatten()
    }
    /// Intermediate-to-output affine declaration.
    pub fn down(&self) -> &LinearSpec {
        match self {
            Self::Split(spec) => &spec.down,
            Self::Fused(spec) => &spec.down,
        }
    }
    /// Resolved gated-product equation.
    pub fn policy(&self) -> GatedProductPolicy {
        match self {
            Self::Split(spec) => spec.policy,
            Self::Fused(spec) => spec.policy,
        }
    }
}
impl DeclareLogicalParameters for GatedProductSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Split(spec) => spec.logical_parameters(),
            Self::Fused(spec) => spec.logical_parameters(),
        }
    }
}
