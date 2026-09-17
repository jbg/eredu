//! Cold filtering mechanisms and mask payload limits, without allocating bits.

use crate::{TextControllerContractError, TokenFilter};

/// Permitted filtering mechanisms and their maximum emitted mask payload.
///
/// This is numerical metadata, not shared-storage custody or execution
/// authority. Optional masks describe possible future decisions and do not
/// imply that a source mask currently exists.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TextFilterWorkspace<'a> {
    /// Preserves this filter's mechanism and allocation-capacity witness.
    /// Allowed values may change, but an All decision remains unfiltered and
    /// an Allowed decision remains filtered.
    Exact(&'a TokenFilter),
    /// Permits both unfiltered All and bounded Allowed decisions.
    ///
    /// The position bound applies only to an emitted mask. An unfiltered
    /// decision can select any executable token, including beyond that bound.
    OptionalMask {
        /// Maximum logical length of an emitted Allowed mask; must be nonzero.
        max_mask_positions: usize,
        /// Maximum owned allocation capacity, including spare mask elements.
        /// Must cover the position bound and fit a host Vec allocation.
        mask_capacity_bytes: u64,
    },
}

impl<'a> From<&'a TokenFilter> for TextFilterWorkspace<'a> {
    fn from(filter: &'a TokenFilter) -> Self {
        Self::Exact(filter)
    }
}

impl TextFilterWorkspace<'_> {
    /// Validates the declaration against a nonempty executable vocabulary.
    /// Exact masks must allow an executable token. Optional metadata has no
    /// membership bits; each eventual decision requires its own validation.
    pub fn validate_output_width(
        self,
        output_width: usize,
    ) -> Result<(), TextControllerContractError> {
        match self {
            Self::Exact(filter) => filter.validate_output_width(output_width)?,
            Self::OptionalMask { .. } => {
                TokenFilter::All.validate_output_width(output_width)?;
                self.mask_capacity_bytes()?;
            }
        }
        Ok(())
    }

    /// Maximum mask allocation capacity after validating its metadata.
    ///
    /// Exact All has no payload. An optional capacity is a possible emitted
    /// allocation, not evidence of an existing retained mask. This method does
    /// not validate token membership or allocate a mask.
    pub fn mask_capacity_bytes(self) -> Result<u64, TextControllerContractError> {
        match self {
            Self::Exact(TokenFilter::All) => Ok(0),
            Self::Exact(TokenFilter::Allowed(mask)) => u64::try_from(mask.capacity())
                .ok()
                .and_then(|capacity| capacity.checked_mul(std::mem::size_of::<bool>() as u64))
                .ok_or(TextControllerContractError::CapacityOverflow),
            Self::OptionalMask {
                max_mask_positions,
                mask_capacity_bytes,
            } => {
                let required = max_mask_positions
                    .checked_mul(std::mem::size_of::<bool>())
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or(TextControllerContractError::CapacityOverflow)?;
                let host_limit = u64::try_from(isize::MAX)
                    .map_err(|_| TextControllerContractError::CapacityOverflow)?;
                if required > host_limit || mask_capacity_bytes > host_limit {
                    return Err(TextControllerContractError::CapacityOverflow);
                }
                if max_mask_positions == 0 || mask_capacity_bytes < required {
                    return Err(TextControllerContractError::InvalidOptionalMask {
                        max_mask_positions,
                        mask_capacity_bytes,
                    });
                }
                Ok(mask_capacity_bytes)
            }
        }
    }
}
