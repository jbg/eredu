//! Fixed-size, confirmed scheduler facts over the session's admitted transport.

use super::*;
use eredu_core::{SpeculativeRequestStatus, SpeculativeScheduleState};

const SCHEDULE_MAGIC: u32 = 0x4552_5350;
const CANCEL: u32 = 1 << 8;
const COMPLETE: u32 = 1 << 9;
const DEADLINE: u32 = 1 << 10;
const OPTIMISTIC: u32 = 1 << 11;
const ANY: u32 = CANCEL | DEADLINE;
const ALL: u32 = COMPLETE | OPTIMISTIC;

impl TextPreparationCoordinator {
    /// Coordinates one registered table without allocating an unbounded native
    /// packet. A count header and each request use the existing admitted fixed
    /// exchange bound; every exchange is charged and confirmed before returning.
    /// Returned facts preserve native completion ownership and request identity.
    pub fn coordinate_speculative_step<T: TextPreparationTransport>(
        &self,
        transport: &T,
        mut local: Vec<SpeculativeScheduleState>,
    ) -> Result<Vec<SpeculativeScheduleState>, TextPreparationAgreementError>
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
        let result = (|| {
            let count = u32::try_from(local.len()).ok();
            let header = self.schedule_frame(
                transport,
                &mut state,
                0,
                [count.unwrap_or(0), 0],
                count.is_none(),
            );
            let Some(_) = count else {
                return Err(TextPreparationAgreementError::Admission(
                    "scheduler request count",
                ));
            };
            header?;
            for (index, request) in local.iter_mut().enumerate() {
                let flags = encode(*request);
                let invalid = request.request.index() != index || flags.is_none();
                let resolved = self.schedule_frame(
                    transport,
                    &mut state,
                    1,
                    [index as u32, flags.unwrap_or(0)],
                    invalid,
                )?;
                request.cancellation_requested = resolved[1] & CANCEL != 0;
                request.verification_complete = resolved[1] & COMPLETE != 0;
                request.verification_deadline_expired = resolved[1] & DEADLINE != 0;
                request.optimistic_eligible = resolved[1] & OPTIMISTIC != 0;
            }
            Ok(local)
        })();
        if let Err(error) = &result {
            state.fenced = true;
            transport.fail_preparation(error);
        }
        result
    }

    fn schedule_frame<T: TextPreparationTransport>(
        &self,
        transport: &T,
        state: &mut State,
        kind: u32,
        payload: [u32; 2],
        rejected: bool,
    ) -> Result<[u32; 2], TextPreparationAgreementError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        transport.ensure_preparation_active()?;
        let participants = self.identity.participant_count();
        let rank = transport.preparation_rank();
        if transport.participant_count() != participants || rank >= participants {
            return Err(TextPreparationAgreementError::Protocol(
                "retained scheduler topology",
            ));
        }
        if !T::Completion::supports_cancellation(self.wait.cancellation()) {
            return Err(TextPreparationAgreementError::Admission(
                "scheduler completion disposition",
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
        frame[..10].copy_from_slice(&[
            SCHEDULE_MAGIC,
            VERSION,
            attempt as u32,
            (attempt >> 32) as u32,
            kind,
            participants as u32,
            rank as u32,
            if rejected { 2 } else { 0 },
            payload[0],
            payload[1],
        ]);
        for (word, bytes) in frame[10..]
            .iter_mut()
            .zip(self.identity.bytes().as_chunks::<4>().0.iter())
        {
            *word = u32::from_le_bytes(*bytes);
        }
        let prepared = self.resolve_schedule_frame(transport, &frame, false);
        frame[1] |= 1 << 16;
        frame[7] = if prepared.is_ok() { 0 } else { 2 };
        if let Ok(payload) = prepared.as_ref() {
            frame[8..10].copy_from_slice(payload);
        }
        let confirmed = self.resolve_schedule_frame(transport, &frame, true);
        let prepared = prepared?;
        confirmed?;
        Ok(prepared)
    }

    fn resolve_schedule_frame<T: TextPreparationTransport>(
        &self,
        transport: &T,
        frame: &[u32; TEXT_PREPARATION_WORDS],
        confirmation: bool,
    ) -> Result<[u32; 2], TextPreparationAgreementError>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let gathered = self.gather_frame(transport, frame)?;
        let participants = self.identity.participant_count();
        if gathered.len()
            != participants
                .checked_mul(TEXT_PREPARATION_WORDS)
                .ok_or(TextPreparationAgreementError::Overflow)?
        {
            return Err(TextPreparationAgreementError::Protocol(
                "scheduler frame length",
            ));
        }
        let rank = transport.preparation_rank();
        if gathered[rank * TEXT_PREPARATION_WORDS..(rank + 1) * TEXT_PREPARATION_WORDS] != frame[..]
        {
            return Err(TextPreparationAgreementError::Protocol(
                "scheduler local echo",
            ));
        }
        for (rank, peer) in gathered
            .as_chunks::<TEXT_PREPARATION_WORDS>()
            .0
            .iter()
            .enumerate()
        {
            if peer[..6] != frame[..6] || peer[6] != rank as u32 || peer[10..] != frame[10..] {
                return Err(TextPreparationAgreementError::Protocol(
                    "scheduler setup, attempt or rank",
                ));
            }
            match peer[7] {
                0 => {}
                2 => return Err(TextPreparationAgreementError::SchedulingRejected { rank }),
                _ => return Err(TextPreparationAgreementError::Protocol("scheduler status")),
            }
        }
        let mut resolved = [frame[8], frame[9]];
        for peer in gathered.as_chunks::<TEXT_PREPARATION_WORDS>().0 {
            if peer[8] != frame[8] {
                return Err(TextPreparationAgreementError::Protocol(
                    "scheduler count or request identity",
                ));
            }
            if confirmation || frame[4] == 0 {
                if peer[9] != frame[9] {
                    return Err(TextPreparationAgreementError::Protocol(
                        "scheduler confirmed decision",
                    ));
                }
            } else {
                let flags = peer[9];
                if flags & !(0xff | ANY | ALL) != 0
                    || flags & 0xff > 8
                    || flags & 0xff != frame[9] & 0xff
                {
                    return Err(TextPreparationAgreementError::Protocol(
                        "scheduler lifecycle or flags",
                    ));
                }
                resolved[1] =
                    (flags & 0xff) | ((resolved[1] & flags) & ALL) | ((resolved[1] | flags) & ANY);
            }
        }
        Ok(resolved)
    }
}

fn encode(state: SpeculativeScheduleState) -> Option<u32> {
    use SpeculativeRequestStatus::*;
    let status = match state.status {
        Prefill => 0,
        ReadyToDraft => 1,
        ReadyToSubmitVerification => 2,
        TargetVerificationInFlight => 3,
        OptimisticDraftRunning => 4,
        OptimisticDraftReady => 5,
        VerificationResolution => 6,
        Completed => 7,
        Cancelled => 8,
        _ => return None,
    };
    Some(
        status
            | if state.cancellation_requested {
                CANCEL
            } else {
                0
            }
            | if state.verification_complete {
                COMPLETE
            } else {
                0
            }
            | if state.verification_deadline_expired {
                DEADLINE
            } else {
                0
            }
            | if state.optimistic_eligible {
                OPTIMISTIC
            } else {
                0
            },
    )
}
