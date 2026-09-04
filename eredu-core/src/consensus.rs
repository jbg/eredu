//! Backend-neutral distributed scheduler consensus.
//!
//! Core defines the wire records and validates rank agreement. A backend
//! adapter supplies only a topology-scoped all-gather of portable words.

use crate::scheduler::{CancellationCause, RequestId, WorkId};
use crate::{BoundedCompletion, BoundedCompletionWait, BoundedSubmissionOutcome, Submission};

/// Topology-scoped transport for scheduler metadata.
///
/// Implementations must return rank-major concatenation of one equally sized
/// word frame from every participant. The scheduler never sends tensors,
/// caches, streams, or executable objects through this interface.
pub trait ConsensusTransport {
    /// Transport error.
    type Error: std::error::Error;

    /// Number of ranks in the consensus topology.
    fn participant_count(&self) -> usize;

    /// Gathers an equally sized word frame from every rank in rank order.
    fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error>;
}

/// Consensus transport that returns exact work ownership under a caller-selected bound.
pub trait BoundedConsensusTransport: ConsensusTransport {
    /// Completion retaining transport resources through completion or safe cancellation.
    type Completion: BoundedCompletion;
    /// Backend-owned gathered value that remains lazy until exact completion.
    type GatherOutput;

    /// Submits one equal-word rank-major gather without synchronizing the caller.
    fn submit_all_gather_words(
        &self,
        local: &[u32],
    ) -> Result<Submission<Self::GatherOutput, Self::Completion>, Self::Error>;

    /// Resolves completed backend output into rank-major portable words.
    fn resolve_all_gather_words(&self, output: Self::GatherOutput)
        -> Result<Vec<u32>, Self::Error>;
}

/// One planned transition and its stable semantic descriptor.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ScheduledWork<'a> {
    /// Scheduler transition identity.
    pub id: WorkId,
    /// Program-specific stable descriptor words.
    pub descriptor: &'a [u32],
}

/// Exact local completion observation before rank consensus.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CompletionObservation {
    /// Local backend work is incomplete.
    Incomplete,
    /// Local backend work completed successfully.
    Complete,
    /// Exact local completion observation failed.
    Failed,
}

impl CompletionObservation {
    const fn wire(self) -> u32 {
        match self {
            Self::Incomplete => 0,
            Self::Complete => 1,
            Self::Failed => 2,
        }
    }
}

/// Topology-wide resolution for one submitted transition.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CompletionResolution {
    /// At least one rank is still executing and no rank failed.
    Incomplete,
    /// Every rank completed successfully.
    Complete,
    /// At least one rank failed while another remains incomplete.
    FailedPending,
    /// At least one rank failed and every rank reached an exact terminal state.
    FailedComplete,
}

/// Structured consensus validation failure without backend error types.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ConsensusError {
    /// A consensus topology cannot be empty.
    #[error("distributed scheduler consensus topology has no participants")]
    EmptyTopology,
    /// Portable metadata exceeded its wire representation.
    #[error("distributed scheduler {0} exceeds u32")]
    MetadataOverflow(&'static str),
    /// The backend collective failed.
    #[error("distributed scheduler consensus failed: {0}")]
    Transport(String),
    /// The transport returned a malformed rank-major gather.
    #[error(
        "distributed scheduler consensus returned {actual} words; expected {expected} for {participants} ranks"
    )]
    MalformedGather {
        /// Expected gathered word count.
        expected: usize,
        /// Actual gathered word count.
        actual: usize,
        /// Expected participant count.
        participants: usize,
    },
    /// A schedule or disposition frame differed.
    #[error("{context} differs at rank {rank}")]
    Mismatch {
        /// Operation being validated.
        context: &'static str,
        /// First disagreeing rank.
        rank: usize,
    },
    /// Completion protocol, work count, or frame size differed.
    #[error("distributed completion header differs at rank {rank}")]
    CompletionHeader {
        /// First disagreeing rank.
        rank: usize,
    },
    /// Completion work ordering differed.
    #[error("distributed completion identity differs at rank {rank}")]
    CompletionIdentity {
        /// First disagreeing rank.
        rank: usize,
    },
    /// A rank emitted an unknown completion status.
    #[error("distributed completion status is invalid at rank {rank}")]
    CompletionStatus {
        /// First rank with an invalid value.
        rank: usize,
    },
}

/// Validates exact work ordering and descriptors across a topology.
pub fn validate_schedule<T: ConsensusTransport>(
    transport: &T,
    plan: &[ScheduledWork<'_>],
    drain_cycle: u64,
    protocol: u64,
) -> Result<(), ConsensusError> {
    let mut words = vec![
        u32::try_from(plan.len())
            .map_err(|_| ConsensusError::MetadataOverflow("schedule length"))?,
        drain_cycle as u32,
        (drain_cycle >> 32) as u32,
        protocol as u32,
        (protocol >> 32) as u32,
    ];
    for work in plan {
        push_u64(&mut words, work.id.request().value());
        push_u64(&mut words, work.id.sequence());
        words.push(
            u32::try_from(work.descriptor.len())
                .map_err(|_| ConsensusError::MetadataOverflow("work descriptor length"))?,
        );
        words.extend_from_slice(work.descriptor);
    }
    validate_equal_words(transport, &words, "distributed work descriptors")
}

/// Validates a cancellation or deadline disposition across a topology.
pub fn validate_disposition<T: ConsensusTransport>(
    transport: &T,
    protocol: u64,
    request: RequestId,
    cause: CancellationCause,
) -> Result<(), ConsensusError> {
    let mut words = vec![protocol as u32, (protocol >> 32) as u32];
    push_u64(&mut words, request.value());
    words.push(match cause {
        CancellationCause::Explicit => 1,
        CancellationCause::Deadline => 2,
    });
    validate_equal_words(transport, &words, "distributed cancellation disposition")
}

/// Resolves exact local completion observations into topology-wide outcomes.
pub fn resolve_completions<T: ConsensusTransport>(
    transport: &T,
    protocol: u64,
    local: &[(WorkId, CompletionObservation)],
) -> Result<Vec<CompletionResolution>, ConsensusError> {
    let participants = checked_participants(transport)?;
    if participants == 1 {
        return Ok(local
            .iter()
            .map(|(_, status)| match status {
                CompletionObservation::Incomplete => CompletionResolution::Incomplete,
                CompletionObservation::Complete => CompletionResolution::Complete,
                CompletionObservation::Failed => CompletionResolution::FailedComplete,
            })
            .collect());
    }

    let mut words = vec![
        protocol as u32,
        (protocol >> 32) as u32,
        u32::try_from(local.len())
            .map_err(|_| ConsensusError::MetadataOverflow("completion work count"))?,
    ];
    for (id, status) in local {
        push_u64(&mut words, id.request().value());
        push_u64(&mut words, id.sequence());
        words.push(status.wire());
    }
    let gathered = gather_words(transport, &words, participants)?;
    for rank in 0..participants {
        let candidate = &gathered[rank * words.len()..(rank + 1) * words.len()];
        if candidate[..3] != words[..3] {
            return Err(ConsensusError::CompletionHeader { rank });
        }
        for (index, (id, _)) in local.iter().enumerate() {
            let offset = 3 + index * 5;
            let expected = [
                id.request().value() as u32,
                (id.request().value() >> 32) as u32,
                id.sequence() as u32,
                (id.sequence() >> 32) as u32,
            ];
            if candidate[offset..offset + 4] != expected {
                return Err(ConsensusError::CompletionIdentity { rank });
            }
            if candidate[offset + 4] > CompletionObservation::Failed.wire() {
                return Err(ConsensusError::CompletionStatus { rank });
            }
        }
    }

    Ok((0..local.len())
        .map(|index| {
            let statuses =
                (0..participants).map(|rank| gathered[rank * words.len() + 3 + index * 5 + 4]);
            let statuses = statuses.collect::<Vec<_>>();
            let failed = statuses.contains(&CompletionObservation::Failed.wire());
            let incomplete = statuses.contains(&CompletionObservation::Incomplete.wire());
            match (failed, incomplete) {
                (true, true) => CompletionResolution::FailedPending,
                (true, false) => CompletionResolution::FailedComplete,
                (false, true) => CompletionResolution::Incomplete,
                (false, false) => CompletionResolution::Complete,
            }
        })
        .collect())
}

fn validate_equal_words<T: ConsensusTransport>(
    transport: &T,
    words: &[u32],
    context: &'static str,
) -> Result<(), ConsensusError> {
    let participants = checked_participants(transport)?;
    if participants == 1 {
        return Ok(());
    }
    let gathered = gather_words(transport, words, participants)?;
    for rank in 0..participants {
        let start = rank * words.len();
        let end = start + words.len();
        if gathered.get(start..end) != Some(words) {
            return Err(ConsensusError::Mismatch { context, rank });
        }
    }
    Ok(())
}

fn checked_participants<T: ConsensusTransport>(transport: &T) -> Result<usize, ConsensusError> {
    let participants = transport.participant_count();
    if participants == 0 {
        Err(ConsensusError::EmptyTopology)
    } else {
        Ok(participants)
    }
}

fn gather_words<T: ConsensusTransport>(
    transport: &T,
    words: &[u32],
    participants: usize,
) -> Result<Vec<u32>, ConsensusError> {
    let expected = words
        .len()
        .checked_mul(participants)
        .ok_or(ConsensusError::MetadataOverflow("gathered word count"))?;
    let gathered = transport
        .all_gather_words(words)
        .map_err(|error| ConsensusError::Transport(error.to_string()))?;
    if gathered.len() != expected {
        return Err(ConsensusError::MalformedGather {
            expected,
            actual: gathered.len(),
            participants,
        });
    }
    Ok(gathered)
}

fn gather_words_bounded<T: BoundedConsensusTransport>(
    transport: &T,
    words: &[u32],
    participants: usize,
    wait: BoundedCompletionWait,
) -> Result<Vec<u32>, ConsensusError>
where
    <T::Completion as crate::Completion>::Error: std::fmt::Display,
{
    let expected = words
        .len()
        .checked_mul(participants)
        .ok_or(ConsensusError::MetadataOverflow("gathered word count"))?;
    let gathered = transport
        .submit_all_gather_words(words)
        .map_err(|error| ConsensusError::Transport(error.to_string()))?
        .wait_bounded(wait)
        .map_err(|error| ConsensusError::Transport(error.to_string()))?;
    let gathered = match gathered {
        BoundedSubmissionOutcome::Completed(gathered) => transport
            .resolve_all_gather_words(gathered)
            .map_err(|error| ConsensusError::Transport(error.to_string()))?,
        BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
            return Err(ConsensusError::Transport(format!(
                "bounded consensus deadline exceeded ({cancellation:?})"
            )))
        }
    };
    if gathered.len() != expected {
        return Err(ConsensusError::MalformedGather {
            expected,
            actual: gathered.len(),
            participants,
        });
    }
    Ok(gathered)
}

/// Agrees one cancellation preparation or commit-authorization status under a bound.
pub fn agree_disposition_status_bounded<T: BoundedConsensusTransport>(
    transport: &T,
    protocol: u64,
    request: RequestId,
    cause: CancellationCause,
    phase: u32,
    local_ready: bool,
    wait: BoundedCompletionWait,
) -> Result<bool, ConsensusError>
where
    <T::Completion as crate::Completion>::Error: std::fmt::Display,
{
    let participants = checked_participants(transport)?;
    let mut words = vec![protocol as u32, (protocol >> 32) as u32];
    push_u64(&mut words, request.value());
    words.push(match cause {
        CancellationCause::Explicit => 1,
        CancellationCause::Deadline => 2,
    });
    words.push(phase);
    words.push(u32::from(local_ready));
    let gathered = gather_words_bounded(transport, &words, participants, wait)?;
    let semantic = &words[..words.len() - 1];
    let mut all_ready = true;
    for rank in 0..participants {
        let frame = &gathered[rank * words.len()..(rank + 1) * words.len()];
        if &frame[..semantic.len()] != semantic {
            return Err(ConsensusError::Mismatch {
                context: "distributed cancellation transaction",
                rank,
            });
        }
        match frame[semantic.len()] {
            0 => all_ready = false,
            1 => {}
            _ => {
                return Err(ConsensusError::Mismatch {
                    context: "distributed cancellation readiness",
                    rank,
                })
            }
        }
    }
    Ok(all_ready)
}

/// Agrees the exact active request set and returns every request whose deadline
/// has expired on at least one rank.
pub fn agree_deadline_candidates_bounded<T: BoundedConsensusTransport>(
    transport: &T,
    protocol: u64,
    local: &[(RequestId, bool)],
    max_requests: usize,
    wait: BoundedCompletionWait,
) -> Result<Vec<RequestId>, ConsensusError>
where
    <T::Completion as crate::Completion>::Error: std::fmt::Display,
{
    let participants = checked_participants(transport)?;
    if local.len() > max_requests {
        return Err(ConsensusError::MetadataOverflow(
            "active deadline request count",
        ));
    }
    let count = u32::try_from(local.len())
        .map_err(|_| ConsensusError::MetadataOverflow("active deadline request count"))?;
    let slots = u32::try_from(max_requests)
        .map_err(|_| ConsensusError::MetadataOverflow("deadline request slots"))?;
    let mut words = vec![protocol as u32, (protocol >> 32) as u32, count, slots];
    let mut previous = None;
    for &(request, expired) in local {
        if previous.is_some_and(|previous| previous >= request) {
            return Err(ConsensusError::Mismatch {
                context: "local deadline request ordering",
                rank: 0,
            });
        }
        previous = Some(request);
        push_u64(&mut words, request.value());
        words.push(u32::from(expired));
    }
    words.resize(4 + max_requests.saturating_mul(3), 0);
    let gathered = gather_words_bounded(transport, &words, participants, wait)?;
    let mut expired = vec![false; local.len()];
    for rank in 0..participants {
        let frame = &gathered[rank * words.len()..(rank + 1) * words.len()];
        if frame[..4] != words[..4] {
            return Err(ConsensusError::Mismatch {
                context: "distributed deadline request set header",
                rank,
            });
        }
        for (index, &(request, _)) in local.iter().enumerate() {
            let offset = 4 + index * 3;
            let expected = [request.value() as u32, (request.value() >> 32) as u32];
            if frame[offset..offset + 2] != expected {
                return Err(ConsensusError::Mismatch {
                    context: "distributed deadline request identity",
                    rank,
                });
            }
            match frame[offset + 2] {
                0 => {}
                1 => expired[index] = true,
                _ => {
                    return Err(ConsensusError::Mismatch {
                        context: "distributed deadline request status",
                        rank,
                    })
                }
            }
        }
        if frame[4 + local.len() * 3..].iter().any(|word| *word != 0) {
            return Err(ConsensusError::Mismatch {
                context: "distributed deadline request padding",
                rank,
            });
        }
    }
    Ok(local
        .iter()
        .zip(expired)
        .filter_map(|(&(request, _), expired)| expired.then_some(request))
        .collect())
}

fn push_u64(output: &mut Vec<u32>, value: u64) {
    output.extend_from_slice(&[value as u32, (value >> 32) as u32]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, convert::Infallible};

    type GatherMutation = dyn FnMut(&mut [u32], usize);

    struct MockTransport {
        participants: usize,
        mutate: RefCell<Option<Box<GatherMutation>>>,
    }

    impl MockTransport {
        fn agreeing(participants: usize) -> Self {
            Self {
                participants,
                mutate: RefCell::new(None),
            }
        }

        fn mutating(participants: usize, mutate: impl FnMut(&mut [u32], usize) + 'static) -> Self {
            Self {
                participants,
                mutate: RefCell::new(Some(Box::new(mutate))),
            }
        }
    }

    impl ConsensusTransport for MockTransport {
        type Error = Infallible;

        fn participant_count(&self) -> usize {
            self.participants
        }

        fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
            let mut gathered = Vec::with_capacity(local.len() * self.participants);
            for rank in 0..self.participants {
                let start = gathered.len();
                gathered.extend_from_slice(local);
                if let Some(mutate) = self.mutate.borrow_mut().as_mut() {
                    mutate(&mut gathered[start..], rank);
                }
            }
            Ok(gathered)
        }
    }

    #[test]
    fn schedule_and_disposition_agree_without_backend_types() {
        let transport = MockTransport::agreeing(3);
        let descriptor = [7, 8, 9];
        let work = [ScheduledWork {
            id: WorkId::new(RequestId::new(5), 2),
            descriptor: &descriptor,
        }];
        validate_schedule(&transport, &work, 11, 13).unwrap();
        validate_disposition(
            &transport,
            13,
            RequestId::new(5),
            CancellationCause::Explicit,
        )
        .unwrap();
    }

    #[test]
    fn schedule_mismatch_fails_closed() {
        let transport = MockTransport::mutating(2, |words, rank| {
            if rank == 1 {
                *words.last_mut().unwrap() ^= 1;
            }
        });
        let descriptor = [7];
        let error = validate_schedule(
            &transport,
            &[ScheduledWork {
                id: WorkId::new(RequestId::new(1), 0),
                descriptor: &descriptor,
            }],
            0,
            9,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ConsensusError::Mismatch {
                context: "distributed work descriptors",
                rank: 1,
            }
        );
    }

    #[test]
    fn completion_resolution_waits_for_failed_rank_peers() {
        let call = RefCell::new(0usize);
        let transport = MockTransport::mutating(3, move |words, rank| {
            if rank == 1 {
                words[7] = CompletionObservation::Failed.wire();
            } else if rank == 2 {
                words[7] = CompletionObservation::Incomplete.wire();
            }
            *call.borrow_mut() += 1;
        });
        let resolutions = resolve_completions(
            &transport,
            22,
            &[(
                WorkId::new(RequestId::new(4), 3),
                CompletionObservation::Complete,
            )],
        )
        .unwrap();
        assert_eq!(resolutions, vec![CompletionResolution::FailedPending]);
    }

    #[test]
    fn malformed_gather_and_empty_topology_fail_closed() {
        let empty = MockTransport::agreeing(0);
        assert_eq!(
            validate_disposition(&empty, 1, RequestId::new(1), CancellationCause::Deadline,)
                .unwrap_err(),
            ConsensusError::EmptyTopology
        );

        struct ShortGather;
        impl ConsensusTransport for ShortGather {
            type Error = Infallible;
            fn participant_count(&self) -> usize {
                2
            }
            fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
                Ok(local.to_vec())
            }
        }
        assert!(matches!(
            validate_disposition(
                &ShortGather,
                1,
                RequestId::new(1),
                CancellationCause::Explicit,
            ),
            Err(ConsensusError::MalformedGather { .. })
        ));
    }
}
