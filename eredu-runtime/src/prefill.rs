//! One cancellable, completion-gated prefill scheduler for all execution paths.

use crate::working_memory::{InferenceExecutionIdentity, InferenceRequest, WorkingMemoryError};
use eredu_core::{
    Completion, GenerationCancellationToken, InferenceGeometry, OutputDemand, Submission,
};
use std::ops::Range;

mod controls;
pub use controls::{PrefillControlPlan, PrefillControlRole, PrefillSpanControlPhase};

/// Exact boundary reached by the shared driver; carries no allocation authority.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PrefillBoundary {
    /// No span is in flight; the next span starts at this prompt-relative position.
    Before { input_position: u64 },
    /// The span ending at this position has actually completed.
    After { input_position: u64 },
}

/// Default maximum decoder positions in ordinary prefill. Admission may choose
/// a smaller span when a working-memory budget is configured.
pub const DEFAULT_PREFILL_CHUNK_POSITIONS: u64 = 512;

/// A semantic decoder span. The architecture prepares this span from its
/// retained ingress; the runtime never slices media or re-encodes an input.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PrefillChunk {
    /// Prompt-relative positions supplied to this invocation.
    pub input: Range<u64>,
    /// Absolute decoder position, including a restored cached prefix.
    pub position: u64,
    /// Vocabulary rows required from this invocation.
    pub output: OutputDemand,
}

/// Native execution seam after request preparation. Implementations reuse
/// ordinary architecture execution, rollback and observation attribution.
pub trait PrefillExecutor<R = InferenceRequest> {
    /// Native scores or an architecture-owned sequence consumer result.
    type Output;
    /// Completion covering all mutable state and transient work in the chunk.
    type Completion: Completion;
    /// Concrete submission failure (translated at the public facade boundary).
    type Error;

    /// Agrees cancellation at the exact shared-driver boundary after retiring
    /// the preceding span. Local execution samples the current shared token.
    /// Parallel executors authenticate this boundary against the same request
    /// and source, retain its reservation through retirement and consensus, and
    /// sample the token immediately before the vote. Every rank participates.
    fn agree_cancellation_at(
        &mut self,
        _boundary: PrefillBoundary,
        cancellation: &GenerationCancellationToken,
        _reservation: R,
    ) -> Result<bool, Self::Error> {
        Ok(cancellation.is_cancelled())
    }

    /// Submits exactly one admitted span. No state-only vocabulary projection.
    /// The completion/recovery owner must retain `reservation`, including when
    /// submission partially succeeds and returns an error. This is the same
    /// lifetime rule as core's move-only `SubmissionLease`.
    fn submit_chunk(
        &mut self,
        chunk: &PrefillChunk,
        reservation: R,
    ) -> Result<Submission<Option<Self::Output>, Self::Completion>, Self::Error>;
}

/// One explicit advance, also used by uninterrupted execution.
#[derive(Debug)]
pub enum PrefillProgress<T> {
    /// An exact completion is still outstanding; no further work was submitted.
    Pending,
    /// One chunk settled. Scores are present only when demanded.
    Chunk {
        /// Completed semantic span.
        chunk: PrefillChunk,
        /// Scores delivered once, after exact completion.
        output: Option<T>,
    },
    /// All prompt chunks settled.
    Complete,
    /// Cancellation settled outstanding work and prevented further submission.
    Cancelled,
}

/// Final outcome of the uninterrupted wrapper over the same step driver.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PrefillOutcome {
    /// All input was processed.
    Complete,
    /// Further input was cancelled after settling submitted work.
    Cancelled,
}

/// A failed scheduler never submits another chunk. Observation failure does not
/// prove safe release of the outstanding completion.
#[derive(Debug, thiserror::Error)]
pub enum PrefillError<E, C> {
    /// Submitted native work or input preparation failed.
    #[error("prefill submission failed: {0}")]
    Submission(E),
    /// Exact native completion could not be observed.
    #[error("prefill completion failed: {0}")]
    Completion(C),
    /// Output presence disagrees with explicit demand.
    #[error("prefill output does not match admitted demand")]
    OutputContract,
    /// A previous error fenced this driver.
    #[error("prefill driver is terminal after failure")]
    Failed,
}

/// State ownership and cancellation shared by ordinary and controlled prefill.
/// Holds at most one native submission. A dropped completion must apply the
/// backend's existing safe retention/teardown contract.
/// The driver owns no executor loan: persistent generation state may retain it
/// between advances and borrow its session only for each `step` or `run` call.
/// Every executor must validate the same retained request and execution target.
pub struct PrefillDriver<T, C: Completion, R = InferenceRequest> {
    geometry: InferenceGeometry,
    reservation: R,
    cancellation: GenerationCancellationToken,
    next: u64,
    pending: Option<(PrefillChunk, Submission<Option<T>, C>)>,
    failed: bool,
    cancelled: bool,
}

impl<T, C: Completion> PrefillDriver<T, C> {
    /// Binds the admitted executable before any preparation or native work.
    pub fn new(
        execution: &InferenceExecutionIdentity,
        reservation: impl Into<InferenceRequest>,
        geometry: InferenceGeometry,
        cancellation: GenerationCancellationToken,
    ) -> Result<Self, WorkingMemoryError> {
        let reservation = reservation.into();
        reservation.begin_prefill(execution, geometry)?;
        Ok(Self {
            geometry,
            reservation,
            cancellation,
            next: 0,
            pending: None,
            failed: false,
            cancelled: false,
        })
    }
}

impl<T, C: Completion> PrefillDriver<T, C, crate::working_memory::OriginalSpeculativeRole> {
    /// Binds one accepted target/draft prefill to this same chunk driver. The
    /// role authenticates its actual report, execution and one-shot start; no
    /// ordinary text request or new native permission is manufactured.
    pub fn new_original_speculative(
        execution: &InferenceExecutionIdentity,
        reservation: crate::working_memory::OriginalSpeculativeRole,
        geometry: InferenceGeometry,
        cancellation: GenerationCancellationToken,
    ) -> Result<Self, WorkingMemoryError> {
        reservation.begin_prefill(execution, geometry)?;
        Ok(Self {
            geometry,
            reservation,
            cancellation,
            next: 0,
            pending: None,
            failed: false,
            cancelled: false,
        })
    }
}

impl<T, C: Completion, R: Clone> PrefillDriver<T, C, R> {
    /// Input positions whose native completion was actually observed in this
    /// invocation. Cancellation and caller rollback do not refund evaluated work.
    /// This is telemetry only, never allocation or global completion authority.
    pub fn completed_positions(&self) -> u64 {
        self.next
    }

    /// Advances by at most one native chunk. Repeated calls while pending only
    /// poll the same handle. Cancellation never releases unresolved authority.
    pub fn step<E: PrefillExecutor<R, Output = T, Completion = C>>(
        &mut self,
        executor: &mut E,
    ) -> Result<PrefillProgress<T>, PrefillError<E::Error, C::Error>> {
        if self.failed {
            return Err(PrefillError::Failed);
        }
        if self.pending.is_none() {
            if self.cancelled {
                return Ok(PrefillProgress::Cancelled);
            }
            if self.next == self.geometry.input_positions {
                return Ok(PrefillProgress::Complete);
            }
            self.agree_cancellation(
                executor,
                PrefillBoundary::Before {
                    input_position: self.next,
                },
            )?;
            if self.cancelled {
                return Ok(PrefillProgress::Cancelled);
            }
            let end = self
                .next
                .saturating_add(self.geometry.prefill_chunk_positions)
                .min(self.geometry.input_positions);
            let chunk = PrefillChunk {
                input: self.next..end,
                position: self.geometry.cached_positions + self.next,
                output: self
                    .geometry
                    .output
                    .for_chunk(end == self.geometry.input_positions),
            };
            match executor.submit_chunk(&chunk, self.reservation.clone()) {
                Ok(submission) => self.pending = Some((chunk, submission)),
                Err(error) => {
                    self.failed = true;
                    return Err(PrefillError::Submission(error));
                }
            }
        }
        let (_, submission) = self.pending.as_ref().expect("submitted above");
        match submission.completion.is_complete() {
            Ok(false) => return Ok(PrefillProgress::Pending),
            Err(error) => {
                self.failed = true;
                return Err(PrefillError::Completion(error));
            }
            Ok(true) => {}
        }
        let (chunk, submission) = self.pending.take().expect("completion was observed");
        self.next = chunk.input.end;
        self.agree_cancellation(
            executor,
            PrefillBoundary::After {
                input_position: self.next,
            },
        )?;
        if self.cancelled {
            return Ok(PrefillProgress::Cancelled);
        }
        if submission.output.is_some() != (chunk.output != OutputDemand::StateOnly) {
            self.failed = true;
            return Err(PrefillError::OutputContract);
        }
        Ok(PrefillProgress::Chunk {
            chunk,
            output: submission.output,
        })
    }

    fn agree_cancellation<E: PrefillExecutor<R, Output = T, Completion = C>>(
        &mut self,
        executor: &mut E,
        boundary: PrefillBoundary,
    ) -> Result<(), PrefillError<E::Error, C::Error>> {
        match executor.agree_cancellation_at(boundary, &self.cancellation, self.reservation.clone())
        {
            Ok(cancelled) => {
                self.cancelled = cancelled;
                if cancelled {
                    self.cancellation.cancel();
                }
                Ok(())
            }
            Err(error) => {
                self.failed = true;
                Err(PrefillError::Submission(error))
            }
        }
    }

    /// Rechecks release evidence after an observation error, without replaying
    /// work or making the failed driver resumable. Budgets remain charged while
    /// either this request or native recovery retains the reservation.
    pub fn release_settled_failure(&mut self) -> bool {
        if !self.failed {
            return false;
        }
        if self
            .pending
            .as_ref()
            .is_some_and(|(_, s)| !s.completion.resources_releasable())
        {
            return false;
        }
        self.pending = None;
        true
    }

    /// Runs ordinary final-output consumption through the same completion loop.
    /// Intermediate physical Sequence results retire at their settled boundary,
    /// before the next chunk's input preparation or submission. Observers which
    /// require every result continue to consume them through `run` or `step`.
    pub fn run_final<E: PrefillExecutor<R, Output = T, Completion = C>>(
        &mut self,
        executor: &mut E,
    ) -> Result<(PrefillOutcome, Option<T>), PrefillError<E::Error, C::Error>> {
        let end = self.geometry.input_positions;
        let mut final_output = None;
        let outcome = self.run(executor, |chunk, output| {
            if chunk.input.end == end {
                final_output = output;
            }
        })?;
        Ok((outcome, final_output))
    }

    /// Runs the same explicit advancement loop to completion. The consumer
    /// receives each demanded result once and owns any retention it requests.
    /// Waiting always refers to the one outstanding completion, never a new
    /// submission or a process-wide device synchronization.
    pub fn run<E: PrefillExecutor<R, Output = T, Completion = C>>(
        &mut self,
        executor: &mut E,
        mut consume: impl FnMut(PrefillChunk, Option<T>),
    ) -> Result<PrefillOutcome, PrefillError<E::Error, C::Error>> {
        loop {
            match self.step(executor)? {
                PrefillProgress::Pending => {
                    let (_, submission) = self
                        .pending
                        .as_ref()
                        .expect("pending progress has a completion");
                    if let Err(error) = submission.completion.wait() {
                        self.failed = true;
                        return Err(PrefillError::Completion(error));
                    }
                }
                PrefillProgress::Chunk { chunk, output } => consume(chunk, output),
                PrefillProgress::Complete => return Ok(PrefillOutcome::Complete),
                PrefillProgress::Cancelled => return Ok(PrefillOutcome::Cancelled),
            }
        }
    }
}
