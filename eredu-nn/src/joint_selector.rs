//! Parameter geometry for jointly normalized selectable and always-on groups.

use crate::{
    DeclareLogicalParameters, Error, JointGroupSelectionSpec, LogicalParameterDeclaration,
    LogicalParameterRole, ParameterSpec,
};

/// Joint router projection, selection-only correction and learned coefficient scale.
///
/// The projection retains the complete global routed/shared axis when either
/// parameter bank is localized. Invocation identity and bank ownership belong
/// to the enclosing architecture, independently of this neural parameter scope.
#[derive(Debug, Clone)]
pub struct JointGroupSelectorSpec {
    input_dimensions: i32,
    weight: ParameterSpec,
    correction_bias: ParameterSpec,
    coefficient_scale: ParameterSpec,
    selection: JointGroupSelectionSpec,
}

impl JointGroupSelectorSpec {
    /// Declares selector geometry; enclosing scopes resolve external sharing targets.
    pub fn new(
        input_dimensions: i32,
        weight: ParameterSpec,
        correction_bias: ParameterSpec,
        coefficient_scale: ParameterSpec,
        selection: JointGroupSelectionSpec,
    ) -> Result<Self, Error> {
        let spec = Self {
            input_dimensions,
            weight,
            correction_bias,
            coefficient_scale,
            selection,
        };
        crate::validate_logical_parameter_scope(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }

    /// Resolves identities, sharing and geometry before allocation.
    pub fn validate(&self) -> Result<(), Error> {
        crate::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)
    }

    /// Full input width, independent of local expert ownership.
    pub const fn input_dimensions(&self) -> i32 {
        self.input_dimensions
    }
    /// Joint routed/shared projection slot.
    pub const fn weight(&self) -> &ParameterSpec {
        &self.weight
    }
    /// Selectable-group correction, excluded from mixture coefficients.
    pub const fn correction_bias(&self) -> &ParameterSpec {
        &self.correction_bias
    }
    /// Learned scalar multiplying the joint mixture coefficients.
    pub const fn coefficient_scale(&self) -> &ParameterSpec {
        &self.coefficient_scale
    }
    /// Complete global selection and joint-normalization equation.
    pub const fn selection(&self) -> JointGroupSelectionSpec {
        self.selection
    }
    /// Exact joint projection geometry, with checked routed/shared cardinality.
    pub fn weight_shape(&self) -> Result<[i32; 2], Error> {
        if self.input_dimensions <= 0 {
            return Err(Error::backend(
                "joint selector input width must be positive",
            ));
        }
        Ok([self.selection.projected_groups(), self.input_dimensions])
    }
    /// Selection-only bias spans the selectable groups.
    pub const fn correction_bias_shape(&self) -> [i32; 1] {
        [self.selection.selectable_groups()]
    }
    /// One learned coefficient multiplier is shared by the complete mixture.
    pub const fn coefficient_scale_shape(&self) -> [i32; 1] {
        [1]
    }
}

impl DeclareLogicalParameters for JointGroupSelectorSpec {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![
            LogicalParameterDeclaration::new(
                &self.weight,
                &self.weight_shape()?,
                LogicalParameterRole::LinearWeight,
            )?,
            LogicalParameterDeclaration::new(
                &self.correction_bias,
                &self.correction_bias_shape(),
                LogicalParameterRole::RoutingCorrection,
            )?,
            LogicalParameterDeclaration::new(
                &self.coefficient_scale,
                &self.coefficient_scale_shape(),
                LogicalParameterRole::RoutingScale,
            )?,
        ])
    }
}
