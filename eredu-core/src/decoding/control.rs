//! Thread-safe commands over retained, source-free execution facts.

use super::DecodingMode;
use crate::refinement::RefinementPolicy;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, MutexGuard};

/// Mode mechanisms realized before generation starts; switching never loads sources.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodingCapabilities {
    /// Causal context and sequential target sampling.
    pub autoregressive: bool,
    /// Complete refinement commitment, distinct from a masked draft primitive.
    pub diffusion: bool,
    /// Entropy and soft-feedback refinement equations.
    pub entropy_feedback: bool,
    /// Mask initialization and confidence-based unmasking equations.
    pub masked_unmasking: bool,
    /// Exact admitted unresolved-token identity for masked refinement.
    pub mask_token_id: Option<u32>,
    /// Retained draft source and causal verification executor.
    pub speculation: bool,
    /// Separately bound draft-only low-rank parameters.
    pub draft_adapter: bool,
    /// Maximum proposal count of a fixed-geometry draft source, if applicable.
    pub max_draft_tokens: Option<std::num::NonZeroUsize>,
    /// Required safe disposition for live verification at its deadline.
    /// Absent means the retained executor supports every validated disposition.
    pub completion_cancellation: Option<crate::CompletionCancellationMode>,
    /// Settled handoff among every advertised mode and its admitted options.
    pub mode_changes: bool,
}

/// A command rejected before changing active or pending execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DecodingModeError {
    /// A second transition cannot replace an already admitted command.
    #[error("a decoding mode change is already pending")]
    ModeChangePending,
    /// No loaded generation has bound this handle.
    #[error("the control handle is not bound to a generation")]
    Unbound,
    /// A handle cannot redirect an active logical run to another command owner.
    #[error("the control handle is already bound to another active generation")]
    AlreadyBound,
    /// Cancellation or terminal completion fences prospective commands.
    #[error("generation is terminal or cancellation was requested")]
    Terminal,
    /// The current sampler requires causal conditional rather than position scores.
    #[error("the sampling strategy requires causal conditional scores")]
    RequiresCausalConditionalScores,
    /// The retained realization does not supply the requested mechanism.
    #[error("decoding mode is unsupported: {0}")]
    Unsupported(&'static str),
}

impl DecodingCapabilities {
    /// Causal execution and a retained draft source with an exact proposal geometry.
    pub fn causal_drafting(capacity: usize) -> Self {
        Self {
            autoregressive: true,
            speculation: capacity != 0,
            max_draft_tokens: std::num::NonZeroUsize::new(capacity),
            mode_changes: true,
            ..Default::default()
        }
    }
    /// Combines execution declarations with exact native completion mechanisms.
    pub fn with_completion_support(mut self, native: bool, quarantine: bool) -> Self {
        use crate::CompletionCancellationMode::{NativeCancel, QuarantineUntilComplete};
        self.completion_cancellation = match (native, quarantine) {
            (true, false) => Some(NativeCancel),
            (false, true) => Some(QuarantineUntilComplete),
            _ => None,
        };
        self.speculation &= native || quarantine;
        self
    }
    /// Checks options using immutable facts without execution or source access.
    pub fn validate(self, mode: DecodingMode) -> Result<(), DecodingModeError> {
        let unsupported = DecodingModeError::Unsupported;
        let check_policy = |policy: Option<RefinementPolicy>| match policy {
            None if self.entropy_feedback || self.masked_unmasking => Ok(()),
            Some(RefinementPolicy::EntropyFeedback(_)) if self.entropy_feedback => Ok(()),
            Some(RefinementPolicy::MaskedUnmasking { mask_token_id, .. })
                if self.masked_unmasking && self.mask_token_id == Some(mask_token_id) =>
            {
                Ok(())
            }
            _ => Err(unsupported("refinement numerical policy")),
        };
        match mode {
            DecodingMode::Autoregressive if self.autoregressive => Ok(()),
            DecodingMode::Autoregressive => Err(unsupported("causal target sampling")),
            DecodingMode::Diffusion { policy } if self.diffusion => check_policy(policy),
            DecodingMode::Diffusion { .. } => Err(unsupported("refinement commitment")),
            DecodingMode::SelfSpeculation(options) => {
                let options = options.request();
                if !self.speculation || !self.autoregressive {
                    return Err(unsupported("draft source and causal verification"));
                }
                if self
                    .max_draft_tokens
                    .is_some_and(|max| options.max_draft_tokens > max)
                {
                    return Err(unsupported(
                        "proposal capacity exceeds retained draft geometry",
                    ));
                }
                if options.draft_adapter && !self.draft_adapter {
                    return Err(unsupported("no draft adapter was retained at load time"));
                }
                if self
                    .completion_cancellation
                    .is_some_and(|required| required != options.scheduler.completion_cancellation)
                {
                    return Err(unsupported(
                        "verification completion cancellation disposition",
                    ));
                }
                if options.refinement.is_some() {
                    check_policy(options.refinement)?;
                }
                Ok(())
            }
        }
    }
}

/// A transition becomes effective only after canonical cache settlement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModeChangeAcknowledgement {
    /// Previous active execution policy.
    pub previous: DecodingMode,
    /// Newly active policy.
    pub active: DecodingMode,
    /// Generated tokens committed before this handoff, excluding prompt positions.
    pub committed_position: u64,
}

/// Rewindable logical mode state; parameters, RNG and budgets have separate owners.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DecodingState {
    /// Current policy used by model invocations.
    pub active: DecodingMode,
    /// One validated policy waiting for a settled commitment boundary.
    pub pending: Option<DecodingMode>,
    /// Most recent successfully settled transition.
    pub acknowledgement: Option<ModeChangeAcknowledgement>,
    /// Commands cannot enter a completed, cancelled or failed run.
    pub terminal: bool,
}

/// Shared command mailbox whose state belongs to one logical continuation.
#[derive(Debug, Clone)]
pub(crate) struct DecodingControl {
    capabilities: DecodingCapabilities,
    requires_causal_scores: bool,
    state: Arc<Mutex<DecodingState>>,
}

pub(crate) enum DecodingControlError<E> {
    Agreement(E),
    Mode(DecodingModeError),
}

/// Opaque command identity for one independently writable generation branch.
/// Construction and activation belong to the request's retained state owner.
#[derive(Debug, Clone)]
pub struct GenerationCommandOwner {
    pub(crate) identity: Arc<()>,
    pub(crate) control: DecodingControl,
}
impl GenerationCommandOwner {
    /// Read-only logical mode facts, including commands queued while inactive.
    pub fn state(&self) -> DecodingState {
        self.control.state()
    }
    /// Closes a discarded branch's mailbox without claiming native completion.
    pub fn close(&self) {
        self.control.finish();
    }
}

/// Fences a bound command owner if submission fails before a request takes it.
pub(crate) struct DecodingRegistration(Option<DecodingControl>);
impl DecodingRegistration {
    pub fn new(control: DecodingControl) -> Self {
        Self(Some(control))
    }
    pub fn control(&self) -> &DecodingControl {
        self.0.as_ref().expect("registered control owner")
    }
    pub fn install(mut self) -> DecodingControl {
        self.0.take().expect("registered control owner")
    }
}
impl Drop for DecodingRegistration {
    fn drop(&mut self) {
        if let Some(control) = &self.0 {
            control.finish();
        }
    }
}

impl DecodingControl {
    /// Binds immutable realized capabilities and validates the initial policy.
    pub fn new(
        capabilities: DecodingCapabilities,
        active: DecodingMode,
        requires_causal_scores: bool,
    ) -> Result<Self, DecodingModeError> {
        capabilities.validate(active)?;
        if requires_causal_scores && matches!(active, DecodingMode::Diffusion { .. }) {
            return Err(DecodingModeError::RequiresCausalConditionalScores);
        }
        Ok(Self {
            capabilities,
            requires_causal_scores,
            state: Arc::new(Mutex::new(DecodingState {
                active,
                pending: None,
                acknowledgement: None,
                terminal: false,
            })),
        })
    }
    fn lock(&self) -> MutexGuard<'_, DecodingState> {
        self.state.lock().unwrap_or_else(|error| error.into_inner())
    }
    /// Exact, source-free facts retained by this run.
    pub const fn capabilities(&self) -> DecodingCapabilities {
        self.capabilities
    }
    fn validate(&self, mode: DecodingMode) -> Result<(), DecodingModeError> {
        self.capabilities.validate(mode)?;
        if self.requires_causal_scores && matches!(mode, DecodingMode::Diffusion { .. }) {
            return Err(DecodingModeError::RequiresCausalConditionalScores);
        }
        Ok(())
    }
    pub(crate) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }
    /// Atomic view for observations and complete snapshot composition.
    pub fn state(&self) -> DecodingState {
        *self.lock()
    }

    /// Holds command ownership through bounded peer agreement, so a concurrent
    /// request cannot replace the command being installed at this boundary.
    pub(crate) fn coordinate<E>(
        &self,
        cancelled: bool,
        committed_position: u64,
        remaining_tokens: Option<usize>,
        agree: impl FnOnce(super::DecodingCoordination) -> Result<super::DecodingCoordination, E>,
    ) -> Result<bool, DecodingControlError<E>> {
        let mut state = self.lock();
        let local = super::DecodingCoordination {
            active: state.active,
            pending: state.pending,
            cancelled: cancelled || state.terminal,
            committed_position,
            remaining_tokens,
        };
        let agreed = agree(local).map_err(DecodingControlError::Agreement)?;
        if agreed.active != local.active
            || agreed.committed_position != committed_position
            || agreed.remaining_tokens != remaining_tokens
        {
            return Err(DecodingControlError::Mode(DecodingModeError::Unsupported(
                "decoding coordination changed authoritative state",
            )));
        }
        if agreed.cancelled {
            state.terminal = true;
            state.pending = None;
            return Ok(true);
        }
        if let Some(pending) = agreed.pending {
            self.validate(pending).map_err(DecodingControlError::Mode)?;
            if pending != state.active && !self.capabilities.mode_changes {
                return Err(DecodingControlError::Mode(DecodingModeError::Unsupported(
                    "settled execution handoff",
                )));
            }
        }
        state.pending = agreed.pending;
        Ok(false)
    }
    /// Admits at most one command without mutating the active execution.
    pub fn request_decoding_mode(&self, mode: DecodingMode) -> Result<(), DecodingModeError> {
        let mut state = self.lock();
        if state.terminal {
            return Err(DecodingModeError::Terminal);
        }
        if state.pending.is_some() {
            return Err(DecodingModeError::ModeChangePending);
        }
        self.validate(mode)?;
        if mode == state.active {
            return Ok(());
        }
        if mode != state.active && !self.capabilities.mode_changes {
            return Err(DecodingModeError::Unsupported("settled execution handoff"));
        }
        state.pending = Some(mode);
        Ok(())
    }
    /// Fences commands; pending work is discarded when cancellation wins.
    pub fn finish(&self) {
        let mut state = self.lock();
        state.terminal = true;
        state.pending = None;
    }
    /// Acknowledges after the executor appends any pending committed suffix once,
    /// removes tentative state and establishes exact native completion.
    pub(crate) fn acknowledge(
        &self,
        mode: DecodingMode,
        committed_position: u64,
        cancelled: impl FnOnce() -> bool,
    ) -> Option<ModeChangeAcknowledgement> {
        let mut state = self.lock();
        if state.terminal || cancelled() {
            state.terminal = true;
            state.pending = None;
            return None;
        }
        debug_assert_eq!(state.pending, Some(mode));
        state.acknowledgement = Some(ModeChangeAcknowledgement {
            previous: state.active,
            active: mode,
            committed_position,
        });
        state.active = mode;
        state.pending = None;
        state.acknowledgement
    }
    /// Installs logical snapshot state after its matching native state is installed.
    pub fn restore(&self, saved: DecodingState) {
        *self.lock() = saved;
    }
    /// Creates a separate command owner for an independently writable child.
    pub fn fork(&self, saved: DecodingState) -> Self {
        Self {
            capabilities: self.capabilities,
            requires_causal_scores: self.requires_causal_scores,
            state: Arc::new(Mutex::new(saved)),
        }
    }
}
