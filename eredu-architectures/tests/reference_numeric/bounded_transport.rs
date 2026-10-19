//! Bounded host word transport shared by prepared capture and generation conformance.
use super::*;
use eredu_core::consensus::{BoundedConsensusTransport, ConsensusTransport};
use eredu_core::{
    capture::*, BackendFailure, BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait,
    Submission,
};
use eredu_runtime::capture::partition::*;
use std::sync::atomic::AtomicBool;

type Round = (Vec<usize>, usize);
#[derive(Default)]
struct Frames {
    values: BTreeMap<usize, Vec<u32>>,
    readers: usize,
}
#[derive(Default)]
pub struct World {
    rounds: Mutex<BTreeMap<Round, Frames>>,
    ready: Condvar,
    failed: AtomicBool,
}
#[derive(Clone)]
pub struct Ticket {
    world: Arc<World>,
    round: Round,
}
impl Ticket {
    fn ready(&self, timeout: std::time::Duration) -> Result<(), std::io::Error> {
        let frames = self.world.rounds.lock().unwrap();
        let (_guard, result) = self
            .world
            .ready
            .wait_timeout_while(frames, timeout, |rounds| {
                !self.world.failed.load(Ordering::SeqCst)
                    && rounds.get(&self.round).unwrap().values.len() != self.round.0.len()
            })
            .unwrap();
        if self.world.failed.load(Ordering::SeqCst) {
            Err(std::io::Error::other("capture transport fenced"))
        } else if result.timed_out() {
            Err(std::io::Error::other("missing capture participant"))
        } else {
            Ok(())
        }
    }
}
impl Completion for Ticket {
    type Error = std::io::Error;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(self
            .world
            .rounds
            .lock()
            .unwrap()
            .get(&self.round)
            .unwrap()
            .values
            .len()
            == self.round.0.len())
    }
    fn wait(&self) -> Result<(), Self::Error> {
        self.ready(std::time::Duration::from_secs(10))
    }
}
impl BoundedCompletion for Ticket {
    fn wait_bounded(
        self,
        wait: BoundedCompletionWait,
    ) -> Result<BoundedCompletionOutcome, Self::Error> {
        self.ready(wait.timeout())?;
        Ok(BoundedCompletionOutcome::Completed)
    }
}
pub struct Transport {
    world: Arc<World>,
    rank: usize,
    size: usize,
    next: Mutex<BTreeMap<Vec<usize>, usize>>,
    pub(crate) hooks: AtomicUsize,
}
impl Transport {
    pub fn new(world: Arc<World>, rank: usize, size: usize) -> Self {
        Self {
            world,
            rank,
            size,
            next: Mutex::new(BTreeMap::new()),
            hooks: AtomicUsize::new(0),
        }
    }
    fn submit(
        &self,
        members: &[usize],
        words: &[u32],
    ) -> Result<Submission<Ticket, Ticket>, std::io::Error> {
        assert!(members.contains(&self.rank));
        let mut next = self.next.lock().unwrap();
        let sequence = next.entry(members.to_vec()).or_default();
        let round = (members.to_vec(), *sequence);
        *sequence += 1;
        let mut rounds = self.world.rounds.lock().unwrap();
        let entry = rounds.entry(round.clone()).or_default();
        assert!(entry
            .values
            .values()
            .all(|previous| previous.len() == words.len()));
        assert!(entry.values.insert(self.rank, words.to_vec()).is_none());
        self.world.ready.notify_all();
        let ticket = Ticket {
            world: self.world.clone(),
            round,
        };
        Ok(Submission {
            output: ticket.clone(),
            completion: ticket,
        })
    }
    fn resolve(&self, ticket: Ticket) -> Result<Vec<u32>, std::io::Error> {
        let mut rounds = self.world.rounds.lock().unwrap();
        let entry = rounds.get_mut(&ticket.round).unwrap();
        assert_eq!(
            entry.values.len(),
            ticket.round.0.len(),
            "completion precedes host resolution"
        );
        let output = entry.values.values().flatten().copied().collect();
        entry.readers += 1;
        if entry.readers == entry.values.len() {
            rounds.remove(&ticket.round);
        }
        Ok(output)
    }
}
impl ConsensusTransport for Transport {
    type Error = std::io::Error;
    fn participant_count(&self) -> usize {
        self.size
    }
    fn all_gather_words(&self, _: &[u32]) -> Result<Vec<u32>, Self::Error> {
        panic!("unbounded capture transport")
    }
}
impl BoundedConsensusTransport for Transport {
    type Completion = Ticket;
    type GatherOutput = Ticket;
    fn submit_all_gather_words(
        &self,
        words: &[u32],
    ) -> Result<Submission<Ticket, Ticket>, Self::Error> {
        self.submit(&(0..self.size).collect::<Vec<_>>(), words)
    }
    fn resolve_all_gather_words(&self, ticket: Ticket) -> Result<Vec<u32>, Self::Error> {
        self.resolve(ticket)
    }
}
impl PartitionCaptureTransport for Transport {
    fn capture_rank(&self) -> usize {
        self.rank
    }
    fn capture_wait(&self) -> Result<BoundedCompletionWait, CaptureError> {
        Ok(BoundedCompletionWait::new(
            std::time::Duration::from_secs(10),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap())
    }
    fn ensure_capture_active(&self) -> Result<(), BackendFailure> {
        if self.world.failed.load(Ordering::SeqCst) {
            Err(BackendFailure::from_error(std::io::Error::other(
                "capture transport fenced",
            )))
        } else {
            Ok(())
        }
    }
    fn estimate_capture_gather(&self, words: usize) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            retained_bytes: words as u64 * 4 * (self.size + 1) as u64,
            host_bytes: words as u64 * 4 * self.size as u64,
            ..Default::default()
        })
    }
    fn fail_capture_exchange(&self, _: &PartitionCaptureExchangeError) {
        let _guard = self.world.rounds.lock().unwrap();
        self.world.failed.store(true, Ordering::SeqCst);
        self.world.ready.notify_all();
    }
}
impl PartitionCaptureHookTransport for Transport {
    type HookOutput = Ticket;
    fn estimate_capture_hook(&self, members: &[usize]) -> Result<CaptureUsage, CaptureError> {
        assert!(
            !members.is_empty()
                && members.windows(2).all(|m| m[0] < m[1])
                && members.last().unwrap() < &self.size
        );
        Ok(CaptureUsage {
            retained_bytes: members.len() as u64 * 8,
            host_bytes: members.len() as u64 * 4,
            ..Default::default()
        })
    }
    fn submit_capture_hook(
        &self,
        members: &[usize],
        success: bool,
    ) -> Result<Submission<Ticket, Ticket>, Self::Error> {
        self.hooks.fetch_add(1, Ordering::SeqCst);
        self.submit(members, &[u32::from(success)])
    }
    fn resolve_capture_hook(&self, ticket: Ticket) -> Result<bool, Self::Error> {
        Ok(self.resolve(ticket)?.iter().all(|v| *v == 1))
    }
}

impl eredu_runtime::run_preparation::TextPreparationTransport for Transport {
    fn preparation_rank(&self) -> usize {
        self.rank
    }
    fn ensure_preparation_active(&self) -> Result<(), BackendFailure> {
        self.ensure_capture_active()
    }
    fn fail_preparation(&self, _: &eredu_runtime::run_preparation::TextPreparationAgreementError) {
        let _guard = self.world.rounds.lock().unwrap();
        self.world.failed.store(true, Ordering::SeqCst);
        self.world.ready.notify_all();
    }
}
impl Transport {
    /// Setup framing uses the same exact bounded tickets; runtime execution never
    /// invokes the unbounded transport entry point.
    pub(crate) fn establish_session(&self) -> eredu_runtime::CommunicationSessionIdentity {
        struct Setup<'a>(&'a Transport);
        impl ConsensusTransport for Setup<'_> {
            type Error = std::io::Error;
            fn participant_count(&self) -> usize {
                self.0.size
            }
            fn all_gather_words(&self, words: &[u32]) -> Result<Vec<u32>, Self::Error> {
                let submitted = self.0.submit_all_gather_words(words)?;
                let wait = BoundedCompletionWait::new(
                    std::time::Duration::from_secs(10),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap();
                assert_eq!(
                    submitted.completion.wait_bounded(wait)?,
                    BoundedCompletionOutcome::Completed
                );
                self.0.resolve_all_gather_words(submitted.output)
            }
        }
        let manifest =
            eredu_runtime::CommunicationManifest::new(self.size, self.rank, vec![], vec![])
                .unwrap();
        eredu_runtime::establish_communication_session(
            &Setup(self),
            &manifest,
            Some([self.rank as u8 + 1; 32]),
        )
        .unwrap()
        .identity()
    }
}
