//! Bounded all-rank preparation using retained session transport and identity.

use crate::CommunicationSessionIdentity;
use eredu_core::{
    consensus::BoundedConsensusTransport,
    run_preparation::{
        TextPreparationOutcome, TextPreparationStage, TextPreparationStatus, TextPreparationUsage,
    },
    BackendFailure, BoundedCompletion, BoundedCompletionWait, BoundedSubmissionOutcome, Completion,
    CompletionCancellationMode,
};
use std::sync::Mutex;

const MAGIC: u32 = 0x4552_5052;
const VERSION: u32 = 1;
mod speculative;
/// Fixed words in one preparation frame, independent of model and prompt size.
pub const TEXT_PREPARATION_WORDS: usize = 18;

/// Native transport facts and shared failure ownership for cold preparation.
pub trait TextPreparationTransport: BoundedConsensusTransport {
    /// Exact retained rank, independent of application-supplied labels.
    fn preparation_rank(&self) -> usize;
    /// Checks the actual selected communication authority before submission.
    fn ensure_preparation_active(&self) -> Result<(), BackendFailure>;
    /// Fences transport on unagreed failure without claiming native completion.
    fn fail_preparation(&self, error: &TextPreparationAgreementError);
}

/// Agreement failures preserve native sources behind a neutral public wrapper.
#[derive(Debug, thiserror::Error)]
pub enum TextPreparationAgreementError {
    /// No compatible finite native/host bound was admitted at setup.
    #[error("invalid text preparation admission: {0}")]
    Admission(&'static str),
    /// Checked cumulative accounting or attempt identity was exhausted.
    #[error("text preparation accounting overflow")]
    Overflow,
    /// Calls on the same local owner must not overlap or follow an unwind.
    #[error("text preparation owner is busy or poisoned")]
    Busy,
    /// A previous unagreed failure permanently fenced this owner.
    #[error("text preparation owner is fenced")]
    Fenced,
    /// Exact rank-major frame identity, shape, stage or status disagreed.
    #[error("text preparation protocol mismatch: {0}")]
    Protocol(&'static str),
    /// A participant rejected bounded scheduler metadata before any model action.
    #[error("speculative scheduling metadata was rejected by rank {rank}")]
    SchedulingRejected {
        /// First rejecting participant in the retained rank order.
        rank: usize,
    },
    /// Native submission, exact completion or host resolution failed.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    /// Completion applied the selected safe disposition at its deadline.
    #[error("text preparation deadline exceeded ({cancellation:?})")]
    Deadline {
        /// Actual safe cancellation/retention disposition.
        cancellation: CompletionCancellationMode,
    },
}

#[derive(Debug, Default)]
struct State {
    usage: TextPreparationUsage,
    fenced: bool,
}

/// Session-owned monotone preparation authority. It is not a model snapshot.
///
/// Construction admits one fixed exchange bound from exact native facts. Each
/// call reserves that bound before allocating protocol buffers or submitting
/// native work. The owner is shared with cloned native session adapters, never
/// duplicated by capture checkpoints or restored model branches.
#[derive(Debug)]
pub struct TextPreparationCoordinator {
    identity: CommunicationSessionIdentity,
    per_attempt: TextPreparationUsage,
    wait: BoundedCompletionWait,
    state: Mutex<State>,
}

impl TextPreparationCoordinator {
    /// Admits native retention/host resolution plus runtime framing at setup.
    /// `native` must conservatively price a `TEXT_PREPARATION_WORDS` gather;
    /// its `attempts` field is zero because runtime owns exchange counting.
    pub fn new(
        identity: CommunicationSessionIdentity,
        native: TextPreparationUsage,
        wait: BoundedCompletionWait,
    ) -> Result<Self, TextPreparationAgreementError> {
        let participants = identity.participant_count();
        if participants == 0 || u32::try_from(participants).is_err() || native.attempts != 0 {
            return Err(TextPreparationAgreementError::Admission(
                "topology or native estimate",
            ));
        }
        let gathered = (TEXT_PREPARATION_WORDS as u64)
            .checked_mul(participants as u64)
            .and_then(|n| n.checked_mul(4))
            .ok_or(TextPreparationAgreementError::Overflow)?;
        let framing = gathered
            .checked_add((TEXT_PREPARATION_WORDS * 4) as u64)
            .ok_or(TextPreparationAgreementError::Overflow)?;
        let per_attempt = TextPreparationUsage {
            attempts: 1,
            retained_bytes: native
                .retained_bytes
                .checked_add(framing)
                .and_then(|n| n.checked_mul(2))
                .ok_or(TextPreparationAgreementError::Overflow)?,
            host_bytes: native
                .host_bytes
                .checked_add(gathered)
                .and_then(|n| n.checked_mul(2))
                .ok_or(TextPreparationAgreementError::Overflow)?,
        };
        Ok(Self {
            identity,
            per_attempt,
            wait,
            state: Mutex::new(State::default()),
        })
    }

    /// Logical reservation for both exchanges in one preparation attempt.
    pub const fn per_attempt(&self) -> TextPreparationUsage {
        self.per_attempt
    }

    /// Charged session lifetime usage, including completed rejection and failure.
    pub fn usage(&self) -> Result<TextPreparationUsage, TextPreparationAgreementError> {
        self.state
            .try_lock()
            .map(|state| state.usage)
            .map_err(|_| TextPreparationAgreementError::Busy)
    }

    /// Agrees the same setup, attempt and stage before permitting advancement.
    /// Failed local preparation must call this with `Failed`, retaining its own
    /// error. Rejection/cancellation are completed decisions and permit retry;
    /// transport, protocol and accounting failures fence the shared owner.
    pub fn agree<T: TextPreparationTransport>(
        &self,
        transport: &T,
        stage: TextPreparationStage,
        local: TextPreparationStatus,
    ) -> Result<TextPreparationOutcome, TextPreparationAgreementError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(_) => {
                let error = TextPreparationAgreementError::Busy;
                transport.fail_preparation(&error);
                return Err(error);
            }
        };
        if state.fenced {
            return Err(TextPreparationAgreementError::Fenced);
        }
        let result = self.agree_inner(transport, stage, local, &mut state);
        if let Err(error) = &result {
            state.fenced = true;
            transport.fail_preparation(error);
        }
        result
    }

    fn agree_inner<T: TextPreparationTransport>(
        &self,
        transport: &T,
        stage: TextPreparationStage,
        local: TextPreparationStatus,
        state: &mut State,
    ) -> Result<TextPreparationOutcome, TextPreparationAgreementError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        transport.ensure_preparation_active()?;
        let participants = self.identity.participant_count();
        let rank = transport.preparation_rank();
        if transport.participant_count() != participants || rank >= participants {
            return Err(TextPreparationAgreementError::Protocol("retained topology"));
        }
        if !T::Completion::supports_cancellation(self.wait.cancellation()) {
            return Err(TextPreparationAgreementError::Admission(
                "bounded completion disposition",
            ));
        }
        let attempt = state.usage.attempts;
        state.usage = TextPreparationUsage {
            attempts: attempt
                .checked_add(1)
                .ok_or(TextPreparationAgreementError::Overflow)?,
            retained_bytes: state
                .usage
                .retained_bytes
                .checked_add(self.per_attempt.retained_bytes)
                .ok_or(TextPreparationAgreementError::Overflow)?,
            host_bytes: state
                .usage
                .host_bytes
                .checked_add(self.per_attempt.host_bytes)
                .ok_or(TextPreparationAgreementError::Overflow)?,
        };
        let mut frame = [0; TEXT_PREPARATION_WORDS];
        frame[..8].copy_from_slice(&[
            MAGIC,
            VERSION,
            attempt as u32,
            (attempt >> 32) as u32,
            stage_word(stage)?,
            participants as u32,
            rank as u32,
            status_word(local),
        ]);
        for (word, bytes) in frame[10..]
            .iter_mut()
            .zip(self.identity.bytes().chunks_exact(4))
        {
            *word = u32::from_le_bytes(bytes.try_into().expect("four-byte identity word"));
        }
        let prepared = self.exchange_frame(transport, &frame, participants, false);
        // No rank advances merely because its own gather/decoding succeeded.
        // Every participant confirms the same completed decision, or reports
        // local decoding/validation failure in this already prepaid round.
        frame[1] = VERSION | (1 << 16);
        frame[7] = if prepared.is_ok() { 0 } else { 2 };
        if let Ok(outcome) = &prepared {
            (frame[8], frame[9]) = match *outcome {
                TextPreparationOutcome::Ready => (0, 0),
                TextPreparationOutcome::Cancelled => (1, 0),
                TextPreparationOutcome::Rejected { rank } => (2, rank as u32),
            };
        }
        let confirmed = self.exchange_frame(transport, &frame, participants, true);
        let prepared = prepared?;
        match confirmed? {
            TextPreparationOutcome::Ready => Ok(prepared),
            _ => Err(TextPreparationAgreementError::Protocol(
                "peer rejected readiness validation",
            )),
        }
    }

    fn exchange_frame<T: TextPreparationTransport>(
        &self,
        transport: &T,
        frame: &[u32; TEXT_PREPARATION_WORDS],
        participants: usize,
        confirmation: bool,
    ) -> Result<TextPreparationOutcome, TextPreparationAgreementError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let gathered = self.gather_frame(transport, frame)?;
        if gathered.len()
            != participants
                .checked_mul(TEXT_PREPARATION_WORDS)
                .ok_or(TextPreparationAgreementError::Overflow)?
        {
            return Err(TextPreparationAgreementError::Protocol(
                "rank-major frame length",
            ));
        }
        let local_rank = transport.preparation_rank();
        if gathered[local_rank * TEXT_PREPARATION_WORDS..(local_rank + 1) * TEXT_PREPARATION_WORDS]
            != frame[..]
        {
            return Err(TextPreparationAgreementError::Protocol("local frame echo"));
        }
        let mut rejected = None;
        let mut cancelled = false;
        for (rank, peer) in gathered.chunks_exact(TEXT_PREPARATION_WORDS).enumerate() {
            if peer[..6] != frame[..6] || peer[6] != rank as u32 || peer[10..] != frame[10..] {
                return Err(TextPreparationAgreementError::Protocol(
                    "setup, attempt, stage or rank",
                ));
            }
            if !confirmation && peer[8..10] != [0, 0] {
                return Err(TextPreparationAgreementError::Protocol(
                    "reserved decision words",
                ));
            }
            if peer[7] == 0 && frame[7] == 0 && peer[8..10] != frame[8..10] {
                return Err(TextPreparationAgreementError::Protocol("prepared decision"));
            }
            if confirmation && peer[7] == 1 {
                return Err(TextPreparationAgreementError::Protocol(
                    "cancelled confirmation",
                ));
            }
            match peer[7] {
                0 => {}
                1 => cancelled = true,
                2 => {
                    rejected.get_or_insert(rank);
                }
                _ => return Err(TextPreparationAgreementError::Protocol("unknown status")),
            }
        }
        Ok(match rejected {
            Some(rank) => TextPreparationOutcome::Rejected { rank },
            None if cancelled => TextPreparationOutcome::Cancelled,
            None => TextPreparationOutcome::Ready,
        })
    }
    fn gather_frame<T: TextPreparationTransport>(
        &self,
        transport: &T,
        frame: &[u32; TEXT_PREPARATION_WORDS],
    ) -> Result<Vec<u32>, TextPreparationAgreementError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let submission = transport
            .submit_all_gather_words(frame)
            .map_err(BackendFailure::from_error)?;
        let output = match submission
            .wait_bounded(self.wait)
            .map_err(BackendFailure::from_error)?
        {
            BoundedSubmissionOutcome::Completed(output) => output,
            BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
                return Err(TextPreparationAgreementError::Deadline { cancellation });
            }
        };
        let gathered = transport
            .resolve_all_gather_words(output)
            .map_err(BackendFailure::from_error)?;
        Ok(gathered)
    }
}

#[cfg(test)]
mod tests;

fn stage_word(stage: TextPreparationStage) -> Result<u32, TextPreparationAgreementError> {
    Ok(match stage {
        TextPreparationStage::Request => 0,
        TextPreparationStage::Prompt => 1,
        TextPreparationStage::Sampling => 2,
        TextPreparationStage::Instrumentation => 3,
        TextPreparationStage::Delivery => 4,
        _ => {
            return Err(TextPreparationAgreementError::Admission(
                "unknown preparation stage",
            ))
        }
    })
}
fn status_word(status: TextPreparationStatus) -> u32 {
    match status {
        TextPreparationStatus::Ready => 0,
        TextPreparationStatus::Cancelled => 1,
        TextPreparationStatus::Failed => 2,
    }
}
