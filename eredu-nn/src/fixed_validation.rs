//! Fixed construction checks shared by ordinary APIs and cold workspace facts.
use crate::{Error, LinearFormat, LinearFormatSpec, ParameterSpec};

/// A physical linear description is invalid. No source identity is cloned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LinearFormatValidationError {
    /// The checkpoint owner's encoding validation failed.
    #[error(transparent)]
    Encoding(#[from] eredu_checkpoint::EncodingValidationError),
    /// Independent row blocks were requested for an incompatible encoding.
    #[error("independent row blocks require a block-FP8 encoding")]
    RowLayout,
    /// Stored companions do not match the encoding's required cardinality.
    #[error(
        "linear format {format:?} requires scale/bias companions {expected:?}, got {actual:?}"
    )]
    CompanionCardinality {
        /// Exact scalar encoding declaration.
        format: LinearFormat,
        /// Required scale and affine-bias presence.
        expected: (bool, bool),
        /// Actual scale and affine-bias presence.
        actual: (bool, bool),
    },
    /// Scale and affine-bias use the same logical parameter identity.
    #[error("linear scale and affine-bias companions require distinct identities")]
    CompanionIdentity,
    /// A physical companion has the wrong semantic role.
    #[error("linear format companions have invalid semantic roles")]
    CompanionRole,
}

/// A format failed validation against its actual borrowed primary weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LinearWeightValidationError {
    /// The encoding or physical companions are invalid.
    #[error(transparent)]
    Format(#[from] LinearFormatValidationError),
    /// A companion reuses the primary weight's identity.
    #[error("linear format companion reuses primary weight identity")]
    PrimaryIdentity,
}
impl LinearWeightValidationError {
    pub(crate) fn into_ordinary(self, weight: &ParameterSpec) -> Error {
        match self {
            Self::Format(cause) => Error::backend(cause),
            Self::PrimaryIdentity => Error::backend(format!(
                "linear format companion reuses primary weight identity {}",
                weight.id,
            )),
        }
    }
}

impl LinearFormatSpec {
    /// Borrows exactly the identities and roles inspected by format validation.
    /// This is not a complete parameter-storage or source-ownership witness.
    pub fn validation_view(&self) -> crate::LinearFormatView<'_> {
        fn companion(parameter: &ParameterSpec) -> crate::LinearCompanionView<'_> {
            crate::LinearCompanionView {
                name: crate::ParameterNameView::new(parameter.id.as_str()),
                role: parameter.linear_companion,
            }
        }
        crate::LinearFormatView {
            format: self.format,
            row_layout: self.row_layout,
            scale: self.scale.as_ref().map(companion),
            affine_bias: self.affine_bias.as_ref().map(companion),
        }
    }

    /// Checks the actual encoding, companion cardinality, identities and roles
    /// without allocating a diagnostic or copying parameter descriptions.
    pub fn validate_fixed(&self) -> Result<(), LinearFormatValidationError> {
        self.validation_view().validate_fixed()
    }

    /// Validates the same description against the borrowed primary identity.
    /// The fixed cause never owns or clones that identity.
    pub fn validate_for_weight_fixed(
        &self,
        weight: &ParameterSpec,
    ) -> Result<(), LinearWeightValidationError> {
        self.validation_view()
            .validate_for_weight_fixed(crate::ParameterNameView::new(weight.id.as_str()))
    }
}

#[cfg(test)]
use crate::{LinearCompanionRole, LinearRowLayout};

mod grouped;
pub use grouped::{
    GatedProductValidationError, GroupedBankDimension, GroupedBankValidationError,
    GroupedLinearValidationError, GroupedProjectionValidationError, GroupedRelu2ValidationError,
};

mod policies;
pub use policies::{
    EmbeddingValidationError, HyperConnectionValidationError, HyperHeadValidationError,
    NormalizationValidationError, RotaryValidationError, SelectorValidationError,
};

#[cfg(test)]
mod tests;
