//! Ordered borrowed identity checks for the existing grouped descriptions.
use crate::{
    Error, GatedProductGroupLayout, GatedProductPolicy, GroupedGatedProductSpec,
    GroupedProjectionSpec, LinearRowError, LinearWeightValidationError, ParameterSpec,
};

/// The scalar gated-product policy is invalid; its source stays borrowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("invalid gated-product policy")]
pub struct GatedProductValidationError;
impl GatedProductPolicy {
    /// Checks finite scalar policy and positive bounds without formatting it.
    pub fn validate_fixed(self) -> Result<(), GatedProductValidationError> {
        if self
            .gate_upper_bound
            .is_some_and(|v| !v.is_finite() || v <= 0.0)
            || self
                .up_absolute_bound
                .is_some_and(|v| !v.is_finite() || v <= 0.0)
            || !self.sigmoid_multiplier.is_finite()
            || self.sigmoid_multiplier <= 0.0
            || !self.up_offset.is_finite()
        {
            return Err(GatedProductValidationError);
        }
        Ok(())
    }
}

/// A grouped projection failed one ordered fixed check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GroupedProjectionValidationError {
    /// Physical format or primary-weight identity failed validation.
    #[error(transparent)]
    Format(#[from] LinearWeightValidationError),
    /// The earliest parameter with a later duplicate in binding order.
    #[error("grouped projection reuses parameter identity at ordinal {0}")]
    ParameterIdentity(usize),
}
impl GroupedProjectionValidationError {
    pub(crate) fn into_ordinary(self, spec: &GroupedProjectionSpec) -> Error {
        match self {
            Self::Format(cause) => cause.into_ordinary(spec.weight()),
            Self::ParameterIdentity(index) => {
                let parameter = spec
                    .parameters_borrowed()
                    .nth(index)
                    .expect("fixed validation produced an existing parameter ordinal");
                Error::backend(format!(
                    "grouped projection reuses parameter identity {:?}",
                    parameter.id
                ))
            }
        }
    }
}
impl GroupedProjectionSpec {
    /// Borrows weight, optional bias, scale and affine-bias in binding order.
    /// The iterator has at most four entries and allocates no destination.
    pub fn parameters_borrowed(&self) -> impl Clone + Iterator<Item = &ParameterSpec> {
        [
            Some(&self.weight),
            self.bias.as_ref(),
            self.format.scale(),
            self.format.affine_bias(),
        ]
        .into_iter()
        .flatten()
    }

    /// Checks the actual primary and companion identities without constructing
    /// a parameter vector. The first parameter with a later duplicate wins.
    pub fn validate_fixed(&self) -> Result<(), GroupedProjectionValidationError> {
        self.format.validate_for_weight_fixed(&self.weight)?;
        for (index, parameter) in self.parameters_borrowed().enumerate() {
            if self
                .parameters_borrowed()
                .skip(index + 1)
                .any(|candidate| candidate.id == parameter.id)
            {
                return Err(GroupedProjectionValidationError::ParameterIdentity(index));
            }
        }
        Ok(())
    }
}

/// A positive grouped-bank dimension, in the existing validation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupedBankDimension {
    /// Number of parameter groups.
    Groups,
    /// Input width.
    Input,
    /// Per-group hidden width.
    Intermediate,
    /// Output width.
    Output,
}
impl std::fmt::Display for GroupedBankDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Groups => "group_count",
            Self::Input => "input_dimensions",
            Self::Intermediate => "intermediate_dimensions",
            Self::Output => "output_dimensions",
        })
    }
}

/// Fixed failure of grouped-bank geometry and borrowed parameter validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GroupedBankValidationError {
    /// An ordered bank dimension is not positive.
    #[error("gated-product group-bank {field} must be positive, got {value}")]
    Dimension {
        /// Failing dimension.
        field: GroupedBankDimension,
        /// Supplied signed extent.
        value: i32,
    },
    /// Gating scalars do not describe a finite supported policy.
    #[error(transparent)]
    Policy(#[from] GatedProductValidationError),
    /// A positive group count cannot be represented in this target's usize.
    #[error("out of range integral type conversion attempted")]
    GroupCountConversion,
    /// The independent source contains the wrong number of groups.
    #[error("independent gated-product bank has {actual} groups, expected {expected}")]
    IndependentCount {
        /// Actual source group count.
        actual: usize,
        /// Declared group count.
        expected: usize,
    },
    /// Packed row geometry is invalid.
    #[error(transparent)]
    Rows(#[from] LinearRowError),
    /// One projection failed before its identities entered the bank.
    #[error("grouped projection {projection} is invalid: {cause}")]
    Projection {
        /// Projection ordinal in binding order.
        projection: usize,
        /// Actual fixed projection failure.
        #[source]
        cause: GroupedProjectionValidationError,
    },
    /// This parameter repeats an earlier bank identity.
    #[error("gated-product group parameter identity is duplicated at projection {projection}, parameter {parameter}")]
    ParameterIdentity {
        /// Projection containing the first repeated identity.
        projection: usize,
        /// Parameter ordinal within that projection.
        parameter: usize,
    },
}
impl GroupedBankValidationError {
    pub(crate) fn into_ordinary(self, spec: &GroupedGatedProductSpec) -> Error {
        match self {
            Self::Policy(_) => {
                Error::backend(format!("invalid gated-product policy: {:?}", spec.policy))
            }
            Self::Projection { projection, cause } => cause.into_ordinary(
                spec.projections_borrowed()
                    .nth(projection)
                    .expect("fixed validation produced an existing projection ordinal"),
            ),
            Self::ParameterIdentity {
                projection,
                parameter,
            } => {
                let source = spec
                    .projections_borrowed()
                    .nth(projection)
                    .and_then(|p| p.parameters_borrowed().nth(parameter))
                    .expect("fixed validation produced an existing bank parameter ordinal");
                Error::backend(format!(
                    "gated-product group parameter identity {} is duplicated",
                    source.id
                ))
            }
            other => Error::backend(other),
        }
    }
}
impl GroupedGatedProductSpec {
    /// Borrows packed projections or independent gate/up/down triples in their
    /// existing binding order, without a projection or identity-set allocation.
    pub fn projections_borrowed(&self) -> impl Clone + Iterator<Item = &GroupedProjectionSpec> {
        let (packed, independent) = match &self.layout {
            GatedProductGroupLayout::Packed { gate_up, down } => {
                ([Some(gate_up), Some(down)], &[][..])
            }
            GatedProductGroupLayout::Independent(groups) => ([None, None], groups.as_slice()),
        };
        packed.into_iter().flatten().chain(
            independent
                .iter()
                .flat_map(|group| [&group.gate, &group.up, &group.down]),
        )
    }

    /// Validates the source without temporary collections. Each projection is
    /// checked before its parameters; the first repeated bank identity wins.
    pub fn validate_fixed(&self) -> Result<(), GroupedBankValidationError> {
        for (field, value) in [
            (GroupedBankDimension::Groups, self.group_count),
            (GroupedBankDimension::Input, self.input_dimensions),
            (
                GroupedBankDimension::Intermediate,
                self.intermediate_dimensions,
            ),
            (GroupedBankDimension::Output, self.output_dimensions),
        ] {
            if value <= 0 {
                return Err(GroupedBankValidationError::Dimension { field, value });
            }
        }
        self.policy.validate_fixed()?;
        if let GatedProductGroupLayout::Independent(groups) = &self.layout {
            let expected = usize::try_from(self.group_count)
                .map_err(|_| GroupedBankValidationError::GroupCountConversion)?;
            if groups.len() != expected {
                return Err(GroupedBankValidationError::IndependentCount {
                    actual: groups.len(),
                    expected,
                });
            }
        }
        if let GatedProductGroupLayout::Packed { gate_up, down } = &self.layout {
            // Both positive i32 widths and the doubled fused width fit usize on
            // the supported 32/64-bit targets, matching ordinary construction.
            gate_up
                .format()
                .row_layout()
                .rows_per_partition_fixed(self.intermediate_dimensions as usize * 2)?;
            down.format()
                .row_layout()
                .rows_per_partition_fixed(self.output_dimensions as usize)?;
        }
        for (projection_index, projection) in self.projections_borrowed().enumerate() {
            projection.validate_fixed().map_err(|cause| {
                GroupedBankValidationError::Projection {
                    projection: projection_index,
                    cause,
                }
            })?;
            for (parameter_index, parameter) in projection.parameters_borrowed().enumerate() {
                let repeated = self
                    .projections_borrowed()
                    .take(projection_index)
                    .flat_map(GroupedProjectionSpec::parameters_borrowed)
                    .chain(projection.parameters_borrowed().take(parameter_index))
                    .any(|previous| previous.id == parameter.id);
                if repeated {
                    return Err(GroupedBankValidationError::ParameterIdentity {
                        projection: projection_index,
                        parameter: parameter_index,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Fixed failure of a selected-linear bank or its projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GroupedLinearValidationError {
    /// Bank cardinality, width or output ownership is invalid.
    #[error("invalid grouped-linear bank or output partition")]
    Geometry,
    /// The actual projection is invalid.
    #[error(transparent)]
    Projection(#[from] GroupedProjectionValidationError),
}
impl GroupedLinearValidationError {
    pub(crate) fn into_ordinary(self, projection: &GroupedProjectionSpec) -> Error {
        match self {
            Self::Projection(cause) => cause.into_ordinary(projection),
            other => Error::backend(other),
        }
    }
}

/// Fixed failure of a packed ReLU2 bank, preserving its validation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GroupedRelu2ValidationError {
    /// Nonpositive group or feature geometry.
    #[error("invalid ReLU2 group-bank geometry")]
    Geometry,
    /// One complete projection failed before bank-wide identity checks.
    #[error("ReLU2 projection {projection} is invalid: {cause}")]
    Projection {
        /// Zero for up, one for down.
        projection: usize,
        /// Actual projection failure.
        #[source]
        cause: GroupedProjectionValidationError,
    },
    /// A binding-order parameter repeats a previous bank identity.
    #[error("ReLU2 group parameter identity is duplicated at projection {projection}, parameter {parameter}")]
    ParameterIdentity {
        /// Zero for up, one for down.
        projection: usize,
        /// Parameter ordinal within that projection.
        parameter: usize,
    },
}
impl GroupedRelu2ValidationError {
    pub(crate) fn into_ordinary(self, spec: &crate::GroupedRelu2Spec) -> Error {
        let projections = [&spec.up, &spec.down];
        match self {
            Self::Projection { projection, cause } => cause.into_ordinary(projections[projection]),
            Self::ParameterIdentity {
                projection,
                parameter,
            } => {
                let source = projections[projection]
                    .parameters_borrowed()
                    .nth(parameter)
                    .expect("fixed validation produced an existing ReLU2 parameter ordinal");
                Error::backend(format!(
                    "ReLU2 group parameter identity {} is duplicated",
                    source.id
                ))
            }
            other => Error::backend(other),
        }
    }
}
impl crate::GroupedRelu2Spec {
    /// Checks both projections before checking bank-wide identity uniqueness.
    /// Repeated borrowed traversal replaces the ordinary temporary identity set.
    pub fn validate_fixed(&self) -> Result<(), GroupedRelu2ValidationError> {
        if self.group_count <= 0 || self.hidden_dimensions <= 0 || self.intermediate_dimensions <= 0
        {
            return Err(GroupedRelu2ValidationError::Geometry);
        }
        let projections = [&self.up, &self.down];
        for (projection, source) in projections.iter().enumerate() {
            source
                .validate_fixed()
                .map_err(|cause| GroupedRelu2ValidationError::Projection { projection, cause })?;
        }
        for (projection_index, source) in projections.iter().enumerate() {
            for (parameter_index, parameter) in source.parameters_borrowed().enumerate() {
                let repeated = projections[..projection_index]
                    .iter()
                    .flat_map(|p| p.parameters_borrowed())
                    .chain(source.parameters_borrowed().take(parameter_index))
                    .any(|prior| prior.id == parameter.id);
                if repeated {
                    return Err(GroupedRelu2ValidationError::ParameterIdentity {
                        projection: projection_index,
                        parameter: parameter_index,
                    });
                }
            }
        }
        Ok(())
    }
}
