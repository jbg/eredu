//! Physical text advancement, independent of committed-token cadence.

use super::{ControlledToken, PendingTextInput};
use crate::refinement::{GenerationPhase, PredictionScoreKind};

/// Whether the next physical operation needs an evolving commitment decision.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TextOperationKind {
    /// Context encoding or tentative refinement without constraint advancement.
    Progress(GenerationPhase),
    /// Select one authoritative token under the current commitment constraints.
    Decision(PredictionScoreKind),
}

/// Completed physical progress that does not commit a canonical token.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TextProgress {
    /// Architecture-selected physical operation.
    pub phase: GenerationPhase,
    /// Block being encoded or refined.
    pub block: u64,
    /// Completed refinement invocations in that block.
    pub iteration: u32,
    /// Opt-in complete tentative IDs. Facade delivery assigns journal identity;
    /// these values never enter commitment constraints or the text parser.
    pub tentative_token_ids: Option<Vec<u32>>,
}

/// One shared-driver advancement, including work between token commitments.
pub enum TextAdvancement<T> {
    /// Completed context or refinement work; the controller is unchanged.
    Progress(TextProgress),
    /// One canonical token committed through the ordinary controller.
    Token(ControlledToken<T>),
}

pub(super) enum PhysicalTextOutput<T> {
    Progress(TextProgress),
    Token {
        token: T,
        score_kind: PredictionScoreKind,
    },
}

/// Submission output, with the pending input retained after physical progress.
pub enum TextOperationOutput<P, T> {
    /// Continue the same run without advancing the commitment controller.
    Progress {
        /// Input authority for the next operation.
        pending: PendingTextInput<P, T>,
        /// Completed physical progress.
        progress: TextProgress,
    },
    /// One selected canonical token, ready for ordinary commitment and decoding.
    Token(T),
    /// No further physical operation or token is available.
    Finished,
}

/// A backend violated the declared advancement protocol.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum TextOperationPolicyError {
    /// Invalid seeded/pinned canvas, rejected before native preparation.
    #[error(transparent)]
    InvalidRefinementCanvas(crate::refinement::RefinementPolicyError),
    /// Causal execution has no tentative block or refinement schedule.
    #[error("refinement policy requires a block-refinement architecture")]
    RequiresBlockRefinementArchitecture,
    /// Adaptive surprise requires scores from a causal conditional distribution.
    #[error("the selected sampling strategy requires causal conditional scores")]
    RequiresCausalConditionalScores,
    /// A selecting operation received no controller decision.
    #[error("token selection requires a commitment decision")]
    MissingDecision,
    /// This executable has no independently resumable physical operation.
    #[error("the executable does not accept a physical continuation")]
    UnsupportedContinuation,
    /// Submission output disagrees with the declared operation kind.
    #[error("text submission disagrees with its declared operation kind")]
    OutcomeMismatch,
    /// Execution or commitment coordinates cannot advance.
    #[error("text generation position overflow")]
    PositionOverflow,
}

/// Native execution or a neutral advancement-contract failure.
#[derive(Debug, thiserror::Error)]
pub enum TextOperationError<E> {
    /// Portable protocol validation rejected the operation.
    #[error(transparent)]
    Policy(#[from] TextOperationPolicyError),
    /// Native execution or exact completion failed.
    #[error("text operation failed: {0}")]
    Backend(#[source] E),
}
