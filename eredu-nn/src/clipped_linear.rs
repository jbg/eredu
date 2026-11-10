//! Input/projection/output clipping contracts without parameter naming policy.
use crate::{
    DeclareLogicalParameters, Error, LinearSpec, LogicalParameterDeclaration, LogicalParameterRole,
    ParameterSpec,
};
/// Atomic learned scalar bounds around a projection.
#[derive(Debug, Clone)]
pub struct LinearClippingSpec {
    /// Minimum input value.
    pub input_min: ParameterSpec,
    /// Maximum input value.
    pub input_max: ParameterSpec,
    /// Minimum output value.
    pub output_min: ParameterSpec,
    /// Maximum output value.
    pub output_max: ParameterSpec,
}
impl LinearClippingSpec {
    /// Exact scalar storage geometry, shared by declaration and unloaded construction.
    pub const SHAPE: &'static [i32] = &[];
}
impl DeclareLogicalParameters for LinearClippingSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        [
            (&self.input_min, LogicalParameterRole::ClipInputMinimum),
            (&self.input_max, LogicalParameterRole::ClipInputMaximum),
            (&self.output_min, LogicalParameterRole::ClipOutputMinimum),
            (&self.output_max, LogicalParameterRole::ClipOutputMaximum),
        ]
        .into_iter()
        .map(|(p, role)| LogicalParameterDeclaration::new(p, Self::SHAPE, role))
        .collect()
    }
}
/// Optional input clamp, ordinary projection, and optional output clamp, in equation order.
#[derive(Debug, Clone)]
pub struct ClippedLinearSpec {
    /// Ordinary projection with its exact encoding.
    pub linear: LinearSpec,
    /// Complete learned bounds; absent means neither clamp is applied.
    pub bounds: Option<LinearClippingSpec>,
}
impl ClippedLinearSpec {
    /// Rejects invalid identities, sharing, geometry and encoding references before allocation.
    pub fn validate(&self) -> Result<(), Error> {
        crate::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }
}
impl DeclareLogicalParameters for ClippedLinearSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.linear.logical_parameters()?;
        if let Some(bounds) = &self.bounds {
            parameters.extend(bounds.logical_parameters()?);
        }
        Ok(parameters)
    }
}
