//! Scalar geometry used by the existing canonical and optimistic proposal worker.
use super::SpeculativeConfig;

/// Exact proposal/output caps of one selected request. This stores no token
/// values, scheduler state, source owner, memory credit or execution authority.
/// Configuration validation remains at the existing request entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpeculativeRequestGeometry {
    output_positions: usize,
    proposal_capacity: usize,
}
impl SpeculativeRequestGeometry {
    /// Uses the same three caps as the canonical and optimistic driver. A zero
    /// selected capacity remains zero; this constructor does not admit it.
    pub fn new(config: &SpeculativeConfig, selected_capacity: usize) -> Self {
        Self::from_caps(config.max_tokens, config.max_draft_tokens, selected_capacity)
    }
    pub(super) fn from_caps(output_positions: usize, requested_proposals: usize, selected_capacity: usize) -> Self {
        Self {
            output_positions,
            proposal_capacity: requested_proposals.min(selected_capacity),
        }
    }
    /// Exact configured cumulative output ceiling, including terminal tokens.
    pub const fn output_positions(self) -> usize {
        self.output_positions
    }
    /// Requested/selected proposal ceiling before the remaining-output cap.
    pub const fn proposal_capacity(self) -> usize {
        self.proposal_capacity
    }
    /// Maximum new block length at the actual canonical or assumed prefix.
    /// Existing promoted proposals occupy part of this same block allowance.
    pub fn proposal_count(self, prefix_positions: usize) -> usize {
        self.proposal_capacity
            .min(self.output_positions.saturating_sub(prefix_positions))
    }
    /// A fresh successful request publishes one token at prefill, then every
    /// completed verification publishes at least one more. A terminal failed
    /// attempt can replace the final successful attempt, not add a new round.
    /// Restore/fork requires its own fresh cumulative occurrence owner.
    pub const fn verification_attempts(self) -> usize {
        self.output_positions.saturating_sub(1)
    }
}
