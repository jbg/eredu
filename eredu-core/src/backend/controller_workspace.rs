//! Value-independent limits for decisions covered by a controller declaration.

use super::{TokenFilter, TokenFilterError, TokenSamplingDecision};
use crate::{TextControllerWorkspace, TextFilterWorkspace};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum FilterMechanism {
    All,
    Mask { capacity: u64 },
    OptionalMask { positions: usize, capacity: u64 },
}

/// Cold filter and visible decision-payload limits for one quoted controller.
///
/// This stores metadata, never the witness mask or controller. Native sampling
/// must separately price the selected mechanism at `output_width`. The original
/// declaration remains responsible for all controller state, temporary copies
/// and growth overlap: validating a decision cannot inspect those opaque owners.
/// An absent controller declaration remains unknown and cannot create a contract.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TextControllerContract {
    output_width: usize,
    filter: FilterMechanism,
    additional_host_bytes: u64,
}

/// A controller decision exceeds its declared sampling mechanism or payload.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum TextControllerContractError {
    /// The final filter has no valid token in the executable output vocabulary.
    #[error(transparent)]
    InvalidFilter(#[from] TokenFilterError),
    /// Filtered and unfiltered sampling require different native traces.
    #[error("token filter mechanism differs from the controller workspace declaration")]
    FilterMechanismMismatch,
    /// Optional mask metadata has no positions or cannot hold the stated extent.
    #[error("optional token mask extent {max_mask_positions} is incompatible with its {mask_capacity_bytes}-byte capacity")]
    InvalidOptionalMask {
        /// Declared maximum logical mask length.
        max_mask_positions: usize,
        /// Declared maximum mask allocation capacity.
        mask_capacity_bytes: u64,
    },
    /// An optional mask exceeds the declared logical extent even if bytes fit.
    #[error(
        "token mask has {required_positions} positions against a {bound_positions}-position bound"
    )]
    FilterPositionsExceeded {
        /// Actual logical mask length.
        required_positions: usize,
        /// Maximum optional mask length in the declaration.
        bound_positions: usize,
    },
    /// The final owned filter allocation exceeds the supplied witness capacity.
    #[error("token filter retains {required_bytes} bytes against a {bound_bytes}-byte bound")]
    FilterCapacityExceeded {
        /// Actual retained allocation, including spare capacity.
        required_bytes: u64,
        /// Supplied witness allocation capacity.
        bound_bytes: u64,
    },
    /// Visible additional decision payload alone exceeds the controller allowance.
    #[error("additional decision filters retain {required_bytes} bytes against a {bound_bytes}-byte controller allowance")]
    AdditionalPayloadExceeded {
        /// Actual additional capacity checked by the selected predicate. The full
        /// validator includes borrowed validity; the owned predicate checks only
        /// the pre-override allocation after independent source authentication.
        required_bytes: u64,
        /// Declared allowance for all additional controller payload.
        bound_bytes: u64,
    },
    /// Host capacities cannot be represented or summed without overflow.
    #[error("controller workspace payload capacity overflow")]
    CapacityOverflow,
}

impl TextControllerContract {
    /// Captures a declaration without cloning masks, advancing a controller or
    /// constructing native values. Exact witnesses must permit an executable
    /// token; optional declarations validate their logical and capacity bounds.
    pub fn from_workspace(
        workspace: TextControllerWorkspace<'_>,
        output_width: usize,
    ) -> Result<Self, TextControllerContractError> {
        workspace.filter.validate_output_width(output_width)?;
        let capacity = workspace.filter.mask_capacity_bytes()?;
        let filter = match workspace.filter {
            TextFilterWorkspace::Exact(TokenFilter::All) => FilterMechanism::All,
            TextFilterWorkspace::Exact(TokenFilter::Allowed(_)) => {
                FilterMechanism::Mask { capacity }
            }
            TextFilterWorkspace::OptionalMask {
                max_mask_positions, ..
            } => FilterMechanism::OptionalMask {
                positions: max_mask_positions,
                capacity,
            },
        };
        capacity
            .checked_add(workspace.additional_host_bytes)
            .ok_or(TextControllerContractError::CapacityOverflow)?;
        Ok(Self {
            output_width,
            filter,
            additional_host_bytes: workspace.additional_host_bytes,
        })
    }

    /// Executable vocabulary width at which the filtering mechanism was quoted.
    pub fn output_width(&self) -> usize {
        self.output_width
    }

    /// Whether the quote permits an explicit closed-mask filtering mechanism.
    /// Optional declarations may also produce unfiltered All decisions.
    pub fn uses_mask(&self) -> bool {
        !matches!(self.filter, FilterMechanism::All)
    }

    /// Whether every decision must use an explicit mask, rather than optionally.
    pub fn requires_mask(&self) -> bool {
        matches!(self.filter, FilterMechanism::Mask { .. })
    }

    /// Maximum final filter capacity included by sampling inspection. An optional
    /// declaration does not imply that this payload is already allocated.
    pub fn filter_capacity_bytes(&self) -> u64 {
        match self.filter {
            FilterMechanism::All => 0,
            FilterMechanism::Mask { capacity } | FilterMechanism::OptionalMask { capacity, .. } => {
                capacity
            }
        }
    }

    /// Enclosing controller allowance, beyond the final supplied filter payload.
    pub fn additional_host_bytes(&self) -> u64 {
        self.additional_host_bytes
    }

    /// Checks a decision without changing its filters, provenance or RNG behavior.
    /// Bit values may differ from the witness; executable membership and owned
    /// capacity must fit the declared native filtering mechanisms. Forced choices
    /// remain normal sampling decisions, with their original filter still live.
    ///
    /// Pre-override ownership counts even when observation provenance is unknown.
    /// Borrowed tokenizer validity also counts as visible additional payload;
    /// the declaration must already cover the controller storage that owns it.
    /// Passing is necessary but does not prove a bound for opaque controller state.
    pub fn validate_decision(
        &self,
        decision: &TokenSamplingDecision<'_>,
    ) -> Result<(), TextControllerContractError> {
        self.validate_final_filter(decision)?;
        let original = Self::owned_additional_bytes(decision)?;
        let validity = decision
            .tokenizer_validity
            .map(capacity_bytes)
            .transpose()?
            .unwrap_or(0);
        let required_bytes = original
            .checked_add(validity)
            .ok_or(TextControllerContractError::CapacityOverflow)?;
        self.validate_additional_bytes(required_bytes)
    }

    /// Checks only owned decision geometry and payload. This does not validate
    /// any borrowed source, establish its custody or admit execution. A runtime
    /// using a separately original domain must authenticate that domain before
    /// calling this predicate; ordinary callers use `validate_decision`.
    pub fn validate_owned_decision_payload(
        &self,
        decision: &TokenSamplingDecision<'_>,
    ) -> Result<(), TextControllerContractError> {
        self.validate_final_filter(decision)?;
        self.validate_additional_bytes(Self::owned_additional_bytes(decision)?)
    }

    fn validate_final_filter(
        &self,
        decision: &TokenSamplingDecision<'_>,
    ) -> Result<(), TextControllerContractError> {
        decision.filter.validate_output_width(self.output_width)?;
        match (self.filter, &decision.filter) {
            (FilterMechanism::All | FilterMechanism::OptionalMask { .. }, TokenFilter::All) => {}
            (
                FilterMechanism::Mask {
                    capacity: bound_bytes,
                }
                | FilterMechanism::OptionalMask {
                    capacity: bound_bytes,
                    ..
                },
                TokenFilter::Allowed(mask),
            ) => {
                if let FilterMechanism::OptionalMask { positions, .. } = self.filter {
                    if mask.len() > positions {
                        return Err(TextControllerContractError::FilterPositionsExceeded {
                            required_positions: mask.len(),
                            bound_positions: positions,
                        });
                    }
                }
                let required_bytes = capacity_bytes(&decision.filter)?;
                if required_bytes > bound_bytes {
                    return Err(TextControllerContractError::FilterCapacityExceeded {
                        required_bytes,
                        bound_bytes,
                    });
                }
            }
            _ => return Err(TextControllerContractError::FilterMechanismMismatch),
        }
        Ok(())
    }

    fn owned_additional_bytes(
        decision: &TokenSamplingDecision<'_>,
    ) -> Result<u64, TextControllerContractError> {
        let original = decision
            .pre_override_filter
            .as_ref()
            .map(capacity_bytes)
            .transpose()?
            .unwrap_or(0);
        Ok(original)
    }

    fn validate_additional_bytes(
        &self,
        required_bytes: u64,
    ) -> Result<(), TextControllerContractError> {
        if required_bytes > self.additional_host_bytes {
            return Err(TextControllerContractError::AdditionalPayloadExceeded {
                required_bytes,
                bound_bytes: self.additional_host_bytes,
            });
        }
        Ok(())
    }
}

fn capacity_bytes(filter: &TokenFilter) -> Result<u64, TextControllerContractError> {
    TextFilterWorkspace::Exact(filter).mask_capacity_bytes()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod original_predicate;
