//! Request-local attention, cache mutation, adapter and score contracts.

use crate::refinement::PredictionScoreKind;
use serde::{Deserialize, Serialize};

mod control;
pub use control::{
    DecodingCapabilities, DecodingModeError, DecodingState, GenerationCommandOwner,
    ModeChangeAcknowledgement,
};
pub(crate) use control::{DecodingControl, DecodingControlError, DecodingRegistration};

/// Generation policy over the parameters retained by one loaded execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum DecodingMode {
    /// Sequential causal target sampling.
    #[default]
    Autoregressive,
    /// Iterative position sampling with architecture-owned attention equations.
    Diffusion {
        /// Absent selects the admitted architecture's numerical defaults.
        policy: Option<crate::refinement::RefinementPolicy>,
    },
    /// Draft and verify against the causal target using the retained draft source.
    SelfSpeculation(SpeculationOptions),
}

/// Validated proposal and scheduler controls, independent of output termination.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "SpeculationOptionsRequest",
    into = "SpeculationOptionsRequest"
)]
pub struct SpeculationOptions {
    request: SpeculationOptionsRequest,
}

/// Resource ceilings retained for every mode of one generation owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeculationLimits {
    /// Maximum proposals admitted in any current or prospective mode.
    pub max_draft_tokens: std::num::NonZeroUsize,
    /// Retained transaction, lookahead and completion bounds.
    pub scheduler: crate::generation::SpeculativeSchedulerOptions,
}
impl From<SpeculationOptions> for SpeculationLimits {
    fn from(options: SpeculationOptions) -> Self {
        let options = options.request();
        Self {
            max_draft_tokens: options.max_draft_tokens,
            scheduler: options.scheduler,
        }
    }
}
impl SpeculationLimits {
    /// Rejects a request which expands this owner's retained resource policy.
    pub fn validate(self, options: SpeculationOptions) -> Result<(), DecodingModeError> {
        let options = options.request();
        let requested = options.scheduler;
        let ceiling = self.scheduler;
        if options.max_draft_tokens > self.max_draft_tokens
            || requested.lookahead_blocks > ceiling.lookahead_blocks
            || requested.max_in_flight_verifications > ceiling.max_in_flight_verifications
            || requested.max_optimistic_branches > ceiling.max_optimistic_branches
            || requested.completion_timeout_milliseconds > ceiling.completion_timeout_milliseconds
            || ceiling.adaptive_lookahead && !requested.adaptive_lookahead
        {
            return Err(DecodingModeError::Unsupported(
                "request exceeds the retained drafting policy",
            ));
        }
        Ok(())
    }
}

/// Values accepted when configuring a speculative invocation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeculationOptionsRequest {
    /// Positive proposal capacity; shortened budgets retain full draft geometry.
    pub max_draft_tokens: std::num::NonZeroUsize,
    /// Bounded verification and optional lookahead controls.
    pub scheduler: crate::generation::SpeculativeSchedulerOptions,
    /// Numerical proposal policy for a masked shared-backbone source.
    pub refinement: Option<crate::refinement::RefinementPolicy>,
    /// Applies a separately retained low-rank adapter during drafting only.
    pub draft_adapter: bool,
}

impl Default for SpeculationOptionsRequest {
    fn default() -> Self {
        Self {
            max_draft_tokens: std::num::NonZeroUsize::new(4).unwrap(),
            scheduler: crate::generation::SpeculativeSchedulerOptions::default(),
            refinement: None,
            draft_adapter: false,
        }
    }
}

impl TryFrom<SpeculationOptionsRequest> for SpeculationOptions {
    type Error = crate::generation::GenerationError;
    fn try_from(request: SpeculationOptionsRequest) -> Result<Self, Self::Error> {
        request.scheduler.validate()?;
        Ok(Self { request })
    }
}

impl From<SpeculationOptions> for SpeculationOptionsRequest {
    fn from(options: SpeculationOptions) -> Self {
        options.request
    }
}

impl Default for SpeculationOptions {
    fn default() -> Self {
        SpeculationOptionsRequest::default()
            .try_into()
            .expect("valid default speculation")
    }
}

impl SpeculationOptions {
    /// Immutable, validated invocation controls.
    pub const fn request(self) -> SpeculationOptionsRequest {
        self.request
    }

    /// Selects a positive proposal allowance without changing scheduler policy.
    pub const fn with_proposal_capacity(mut self, capacity: std::num::NonZeroUsize) -> Self {
        self.request.max_draft_tokens = capacity;
        self
    }
}

impl DecodingMode {
    /// Applicable position-sampling policy, separate from causal target sampling.
    pub const fn refinement_policy(self) -> Option<crate::refinement::RefinementPolicy> {
        match self {
            Self::Autoregressive => None,
            Self::Diffusion { policy } => policy,
            Self::SelfSpeculation(options) => options.request.refinement,
        }
    }
}

/// Rank-local command facts agreed before the next physical generation action.
/// Completion ownership and mutable model state remain with their native owner.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DecodingCoordination {
    /// Already acknowledged execution policy.
    pub active: DecodingMode,
    /// One prospective command; an absent local command accepts a peer's command.
    pub pending: Option<DecodingMode>,
    /// Cancellation from any participant wins over a prospective command.
    pub cancelled: bool,
    /// Authoritative generated positions, independent of physical invocation count.
    pub committed_position: u64,
    /// Retained output allowance, excluding tentative proposals.
    pub remaining_tokens: Option<usize>,
}

/// Resolves exact rank-local controls without allocating native state.
pub fn agree_decoding_controls(
    participants: &[DecodingCoordination],
) -> Result<DecodingCoordination, DecodingModeError> {
    let Some(mut agreed) = participants.first().copied() else {
        return Err(DecodingModeError::Unsupported(
            "empty decoding control group",
        ));
    };
    agreed.cancelled = participants.iter().any(|peer| peer.cancelled);
    for peer in participants {
        if peer.active != agreed.active
            || peer.committed_position != agreed.committed_position
            || peer.remaining_tokens != agreed.remaining_tokens
        {
            return Err(DecodingModeError::Unsupported(
                "distributed decoding state disagrees",
            ));
        }
        if let Some(pending) = peer.pending.filter(|_| !agreed.cancelled) {
            if agreed.pending.is_some_and(|selected| selected != pending) {
                return Err(DecodingModeError::Unsupported(
                    "conflicting distributed decoding commands",
                ));
            }
            agreed.pending = Some(pending);
        }
    }
    if agreed.cancelled {
        agreed.pending = None;
    }
    Ok(agreed)
}

/// Visibility of a token span over an already retained causal prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionVisibility {
    /// Each supplied position sees only its causal predecessors.
    Causal,
    /// Each supplied position sees the retained prefix and complete supplied block.
    PrefixAndBlock,
}

/// Ownership of newly projected keys and values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheWritePolicy {
    /// Append supplied authoritative or transaction-local causal positions.
    Append,
    /// Keep projected block state transient and leave the canonical frontier fixed.
    Transient,
}

/// Low-rank parameter scope for one invocation over a shared base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterScope {
    /// Execute only the shared base projections.
    Disabled,
    /// Apply an admitted drafting adapter to its declared projections.
    Draft,
}

/// Complete immutable invocation semantics, independent of shared module state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionInvocation {
    /// Position visibility.
    pub visibility: AttentionVisibility,
    /// Projected state ownership.
    pub cache_write: CacheWritePolicy,
    /// Optional low-rank projection scope.
    pub adapters: AdapterScope,
    /// Meaning of returned token scores.
    pub scores: PredictionScoreKind,
}

impl AttentionInvocation {
    /// Authoritative context encoding and causal target verification.
    pub const CAUSAL: Self = Self {
        visibility: AttentionVisibility::Causal,
        cache_write: CacheWritePolicy::Append,
        adapters: AdapterScope::Disabled,
        scores: PredictionScoreKind::CausalConditional,
    };
    /// Ordinary diffusion refinement over a retained causal prefix.
    pub const DIFFUSION: Self = Self {
        visibility: AttentionVisibility::PrefixAndBlock,
        cache_write: CacheWritePolicy::Transient,
        adapters: AdapterScope::Disabled,
        scores: PredictionScoreKind::RefinementPosition,
    };
    /// Shared-backbone drafting with optional admitted low-rank projections.
    pub const DRAFT: Self = Self {
        adapters: AdapterScope::Draft,
        ..Self::DIFFUSION
    };
}
