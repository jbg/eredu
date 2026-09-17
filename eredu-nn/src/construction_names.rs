//! Borrowed identity and physical-format checks for construction declarations.
use crate::{
    LinearCompanionRole, LinearFormat, LinearFormatValidationError, LinearRowLayout,
    LinearWeightValidationError,
};

/// A logical parameter name borrowed from at most two existing UTF-8 slices.
///
/// Joining slices creates no string. This is logical name transport, not source,
/// allocation or admission authority. Equal bytes compare equal across segment
/// boundaries, including empty segments.
#[derive(Debug, Clone, Copy)]
pub struct ParameterNameView<'a> {
    prefix: &'a str,
    suffix: &'a str,
}

/// Fixed failure when declaring a parameter name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ParameterNameError {
    /// The complete name is empty or consists only of Unicode whitespace.
    #[error("parameter identity must not be empty")]
    Empty,
}

impl<'a> ParameterNameView<'a> {
    /// Borrows one existing complete name.
    pub const fn new(name: &'a str) -> Self {
        Self::joined(name, "")
    }

    /// Borrows a complete name as an ordered pair of existing slices.
    pub const fn joined(prefix: &'a str, suffix: &'a str) -> Self {
        Self { prefix, suffix }
    }

    /// Returns the actual borrowed slices in byte order.
    pub const fn parts(self) -> [&'a str; 2] {
        [self.prefix, self.suffix]
    }

    /// Returns the checked complete UTF-8 byte length.
    pub fn byte_len(self) -> Option<usize> {
        self.prefix.len().checked_add(self.suffix.len())
    }

    /// Applies the same Unicode whitespace rule as ordinary ParameterId creation.
    pub fn validate_fixed(self) -> Result<(), ParameterNameError> {
        if self
            .prefix
            .chars()
            .chain(self.suffix.chars())
            .all(char::is_whitespace)
        {
            Err(ParameterNameError::Empty)
        } else {
            Ok(())
        }
    }
}

impl PartialEq for ParameterNameView<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.prefix
            .bytes()
            .chain(self.suffix.bytes())
            .eq(other.prefix.bytes().chain(other.suffix.bytes()))
    }
}
impl Eq for ParameterNameView<'_> {}
impl std::fmt::Display for ParameterNameView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.prefix)?;
        f.write_str(self.suffix)
    }
}

/// Identity and semantic role used by the shared physical-format validator.
///
/// This view is only the fields inspected by that validator. It is not a
/// complete parameter declaration or a parameter-storage size witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearCompanionView<'a> {
    /// Complete logical identity, possibly stored as two borrowed slices.
    pub name: ParameterNameView<'a>,
    /// Actual semantic role retained by the construction declaration.
    pub role: Option<LinearCompanionRole>,
}

/// Borrowed scalar and companion facts checked by owning and fixed construction.
///
/// Parameter-name validity is checked when declaring each parameter, before this
/// format check. Group/alias ownership and storage are separate obligations.
#[derive(Debug, Clone, Copy)]
pub struct LinearFormatView<'a> {
    /// Physical checkpoint encoding.
    pub format: LinearFormat,
    /// Independent row-block origins.
    pub row_layout: LinearRowLayout,
    /// Actual scale identity and role, if present.
    pub scale: Option<LinearCompanionView<'a>>,
    /// Actual affine-bias identity and role, if present.
    pub affine_bias: Option<LinearCompanionView<'a>>,
}

impl LinearFormatView<'_> {
    /// Checks encoding, rows, cardinality, identity and roles in ordinary order.
    pub fn validate_fixed(self) -> Result<(), LinearFormatValidationError> {
        self.format.validate_fixed()?;
        if self.row_layout != LinearRowLayout::Contiguous
            && !matches!(self.format, LinearFormat::E4M3BlockFp8(_))
        {
            return Err(LinearFormatValidationError::RowLayout);
        }
        let expected = match self.format {
            LinearFormat::Dense | LinearFormat::GgufIQuant { .. } => (false, false),
            LinearFormat::MxFp4 | LinearFormat::E4M3BlockFp8(_) => (true, false),
            LinearFormat::Affine(_) => (true, true),
        };
        let actual = (self.scale.is_some(), self.affine_bias.is_some());
        if actual != expected {
            return Err(LinearFormatValidationError::CompanionCardinality {
                format: self.format,
                expected,
                actual,
            });
        }
        if self
            .scale
            .zip(self.affine_bias)
            .is_some_and(|(scale, bias)| scale.name == bias.name)
        {
            return Err(LinearFormatValidationError::CompanionIdentity);
        }
        if self
            .scale
            .is_some_and(|scale| scale.role != Some(LinearCompanionRole::Scale))
            || self
                .affine_bias
                .is_some_and(|bias| bias.role != Some(LinearCompanionRole::AffineBias))
        {
            return Err(LinearFormatValidationError::CompanionRole);
        }
        Ok(())
    }

    /// Checks the format before comparing actual complete primary-name bytes.
    pub fn validate_for_weight_fixed(
        self,
        weight: ParameterNameView<'_>,
    ) -> Result<(), LinearWeightValidationError> {
        self.validate_fixed()?;
        if self
            .scale
            .into_iter()
            .chain(self.affine_bias)
            .any(|companion| companion.name == weight)
        {
            return Err(LinearWeightValidationError::PrimaryIdentity);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
