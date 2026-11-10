//! Exact affine slots and equation policy for an ordinary split gated product.

use crate::{
    DeclareLogicalParameters, Error, GatedProductPolicy, LinearSpec, LogicalParameterDeclaration,
};

/// Independent gate/up reads followed by a gated product and an output projection.
/// Input and output widths may differ; enclosing residual equations validate their own width.
#[derive(Debug, Clone)]
pub struct SplitGatedProductSpec {
    /// Input-to-intermediate gate projection.
    pub gate: LinearSpec,
    /// Input-to-intermediate value projection.
    pub up: LinearSpec,
    /// Intermediate-to-output projection.
    pub down: LinearSpec,
    /// Pointwise activation, clipping and scaling before the output projection.
    pub policy: GatedProductPolicy,
}

impl SplitGatedProductSpec {
    /// Builds a complete scope and rejects incompatible geometry, identities or sharing.
    pub fn new(
        gate: LinearSpec,
        up: LinearSpec,
        down: LinearSpec,
        policy: GatedProductPolicy,
    ) -> Result<Self, Error> {
        let spec = Self {
            gate,
            up,
            down,
            policy,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Returns affine declarations in gate, up, down order.
    pub fn projections(&self) -> [&LinearSpec; 3] {
        [&self.gate, &self.up, &self.down]
    }

    /// Validates this complete logical scope before backend construction.
    pub fn validate(&self) -> Result<(), Error> {
        crate::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
}

impl DeclareLogicalParameters for SplitGatedProductSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        self.policy.validate()?;
        if self.gate.input != self.up.input
            || self.gate.output != self.up.output
            || self.down.input != self.gate.output
        {
            return Err(Error::backend("incompatible split gated-product geometry"));
        }
        let mut parameters = Vec::new();
        for projection in self.projections() {
            parameters.extend(projection.logical_parameters()?);
        }
        Ok(parameters)
    }
}
