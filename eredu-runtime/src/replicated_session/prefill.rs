//! Shared prefill scheduling over the ordinary session transaction.

mod lifecycle;
mod span;
pub use lifecycle::MediaPrefillSpan;
use lifecycle::SpanLifecycle;
pub use span::{OrdinaryPrefillSpan, PrefillScoreLayout, PrefillSpanOperation};

use super::*;
use crate::inspection::{
    PrefillChunkRetentionContext, PreparedPrefillChunkRetention, SettledPrefillChunkRetention,
};
use crate::layered::PreparedLayeredObservationPaths;
use crate::working_memory::InferenceStateRetention;
use crate::{
    prefill::{
        PrefillBoundary, PrefillChunk, PrefillControlPlan, PrefillControlRole, PrefillExecutor,
        PrefillSchedulingAuthority, PrefillSpanControlPhase,
    },
    working_memory::{InferenceExecutionIdentity, InferenceRequest, WorkingMemoryError},
};
use eredu_core::{
    Completion, GenerationCancellationToken, InferenceGeometry, OutputDemand, Submission,
};

/// Architecture-owned prepared ingress. Each chunk is an owned semantic view;
/// media implementations retain encoder results and align decoder coordinates.
/// Runtime only schedules spans and never interprets or slices these values.
pub trait PreparedPrefillSource<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    /// Values kept alive through a chunk's native completion.
    type Chunk;
    /// Exact geometry admitted before input preparation.
    fn geometry(&self) -> InferenceGeometry;
    /// Prepares one decoder span without repeating ingress work.
    fn prepare_chunk(
        &self,
        chunk: &PrefillChunk,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Chunk, A::Error>;
    /// Forms the architecture input from the prepared chunk.
    fn input<'a>(&'a self, chunk: &'a Self::Chunk) -> A::Input<'a>;
    /// Actual shared prompt identity installed after the final chunk commits.
    /// A source retains its existing owner; absence performs no allocation.
    fn shared_cache_identity(&self) -> Option<SharedPreparedInputCacheIdentity> {
        None
    }
}

/// Completion returned only after both session state and its native reservation
/// scope have settled. Retains the same request charge until the consumer drops it.
#[derive(Debug)]
pub struct SettledPrefillCompletion<R = InferenceRequest> {
    _reservation: R,
}
impl<R> Completion for SettledPrefillCompletion<R> {
    type Error = std::convert::Infallible;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn resources_releasable(&self) -> bool {
        true
    }
}

/// Adapts one session and prepared ingress to [`crate::prefill::PrefillDriver`].
/// The exclusive session loan prevents advancement, restore or mutation outside
/// the driver. `step` and `run` use this same executor and transaction engine.
pub struct SessionPrefill<
    'a,
    A,
    B,
    M,
    D,
    P,
    O: ?Sized,
    K = OrdinaryPrefillSpan,
    R = InferenceRequest,
> where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    session: &'a mut ReplicatedTextSession<A, B, M, D>,
    source: P,
    geometry: InferenceGeometry,
    controls: PrefillControlPlan,
    request: R,
    observer: &'a mut O,
    context: &'a <B::Tensor as Tensor>::Context,
    // Issued only after canonical model/guard success. Drop never releases pins.
    pending_retention: Option<SettledPrefillChunkRetention>,
    operation: K,
}

/// Selection and completion of a semantic prefill source.
pub enum PrefillSourceOutcome<T> {
    /// This source requires the existing whole-request ingress protocol.
    Unavailable,
    /// Every scheduled span completed and produced ordinary final scores.
    Complete(T),
    /// Agreed cancellation stopped at a safely completed boundary, without scores.
    Cancelled,
}

/// Selected-source outcome with the actual completed input-position count.
/// This telemetry survives cancellation and rollback; it grants no work or memory.
pub struct PrefillSourceProgress<T> {
    /// Completion/cancellation without a manufactured output.
    pub outcome: PrefillSourceOutcome<T>,
    /// Positions completed by this invocation, excluding the incoming cache.
    pub completed_positions: u64,
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    // One checked source-token boundary shared by opening and completed views.
    // Explicit field arguments keep the immutable execution/token loans disjoint
    // from the mutable mechanisms loan, with no second session lookup.
    fn validated_prefill_source_paths<'s>(
        required: bool,
        execution: &D::Runtime,
        paths: Option<&'s PreparedLayeredObservationPaths>,
    ) -> Result<
        Option<&'s PreparedLayeredObservationPaths>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        if !required {
            return Ok(None);
        }
        let paths = paths.ok_or(ReplicatedTextSessionError::PreparedObservation(
            PreparedSessionObservationError::Unavailable,
        ))?;
        D::validate_observation_paths(execution, paths).map_err(widen_infallible)?;
        Ok(Some(paths))
    }

    /// Stable target identity used to admit this exact session's inference work.
    pub fn inference_execution_identity(&self) -> &InferenceExecutionIdentity {
        &self.prefill_identity
    }
}

impl<'a, A, B, M, D, P, O, K, R: PrefillSchedulingAuthority>
    SessionPrefill<'a, A, B, M, D, P, O, K, R>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
    O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    K: SpanLifecycle<A, B, M, D, P, O>,
{
    /// Checks target, geometry and observer demand before preparing native input.
    /// Sequence observers must be included in the original workspace admission.
    fn new_with_operation(
        session: &'a mut ReplicatedTextSession<A, B, M, D>,
        source: P,
        reservation: impl Into<R>,
        context: &'a <B::Tensor as Tensor>::Context,
        observer: &'a mut O,
        operation: K,
    ) -> Result<Self, WorkingMemoryError> {
        let reservation = reservation.into();
        let geometry = operation.geometry(&source);
        reservation.validate(session.inference_execution_identity(), geometry)?;
        if observer.requires_sequence_readout() && geometry.output != OutputDemand::Sequence {
            return Err(WorkingMemoryError::OutputDemandMismatch {
                admitted: geometry.output,
                required: OutputDemand::Sequence,
            });
        }
        if !session.selected.exact_completion_available() {
            return Err(WorkingMemoryError::CompletionUnavailable);
        }
        if session.control_fence.is_some() {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        let controls = PrefillControlPlan::new(geometry, true)?;
        Ok(Self {
            session,
            source,
            geometry,
            controls,
            request: reservation,
            observer,
            context,
            pending_retention: None,
            operation,
        })
    }

    fn with_terminal_unwind<T>(
        &mut self,
        phase: crate::DistributedExecutionPhase,
        operation: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation(self)));
        self.session.finish_terminal_unwind(phase, result)
    }

    fn agree_cancellation_at_boundary(
        &mut self,
        reservation: R,
        boundary: PrefillBoundary,
        locally_cancelled: impl FnOnce() -> bool,
    ) -> Result<bool, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.with_terminal_unwind(
            crate::DistributedExecutionPhase::PrefillReservation,
            |executor| {
                executor.agree_cancellation_retained(reservation, boundary, locally_cancelled)
            },
        )
    }

    fn agree_cancellation_retained(
        &mut self,
        reservation: R,
        boundary: PrefillBoundary,
        locally_cancelled: impl FnOnce() -> bool,
    ) -> Result<bool, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.session.ensure_control_unfenced()?;
        self.request
            .validate_same_request(&reservation)
            .map_err(ReplicatedTextSessionError::WorkingMemory)?;
        reservation
            .validate(self.session.inference_execution_identity(), self.geometry)
            .map_err(ReplicatedTextSessionError::WorkingMemory)?;
        let role = self.controls.boundary_role(boundary).ok_or(
            ReplicatedTextSessionError::WorkingMemory(WorkingMemoryError::IdentityMismatch),
        )?;
        let retained = self
            .session
            .coordinate_scheduling_entry(&reservation, Some(role));
        let guard = match retained {
            Ok(guard) => guard,
            Err(error) => {
                self.session
                    .fence_terminal(crate::DistributedExecutionPhase::PrefillReservation);
                return Err(error);
            }
        };
        let retirement = match self.pending_retention.take() {
            Some(ticket) => ticket
                .validate_request(reservation.inference_request().ok_or(
                    ReplicatedTextSessionError::WorkingMemory(WorkingMemoryError::IdentityMismatch),
                )?)
                .map_err(ReplicatedTextSessionError::WorkingMemory)
                .and_then(|()| {
                    if self.session.mechanisms.requires_prefill_opening_sources() {
                        let paths =
                            ReplicatedTextSession::<A, B, M, D>::validated_prefill_source_paths(
                                self.session.mechanisms.requires_prepared_prefill_sources(),
                                &self.session.execution,
                                self.session.observation_paths.get(),
                            )?;
                        let execution = crate::inspection::RuntimeOpeningExecution::new(
                            &self.session.execution,
                            D::visit_retained_values,
                        );
                        self.session
                            .mechanisms
                            .prepare_prefill_retirement_sources(
                                &self.session.state,
                                &ticket,
                                &execution.borrow_prepared(paths),
                            )
                            .map_err(ReplicatedTextSessionError::Mechanism)?;
                    }
                    self.observer
                        .retire_prefill_chunk_retention(ticket)
                        .map_err(ReplicatedTextSessionError::Architecture)
                }),
            None => Ok(()),
        };
        let admitted = self.session.agree_execution_phase(
            crate::DistributedExecutionPhase::PrefillReservation,
            retirement.is_ok(),
            self.context,
        );
        if retirement.is_err() || !matches!(admitted, Ok(true)) {
            drop(guard);
            retirement?;
            admitted.map_err(widen_infallible)?;
            return Err(ReplicatedTextSessionError::Contract(
                "another rank rejected prefill cancellation retention".into(),
            ));
        }
        // The fresh token is sampled after retirement/readiness, under the same
        // retained guard. An agreed cancellation is a successful status vote.
        let continue_work = self.session.agree_execution_phase(
            crate::DistributedExecutionPhase::PrefillCancellation,
            !locally_cancelled(),
            self.context,
        );
        let all_continue = match continue_work {
            Ok(value) => value,
            Err(error) => {
                drop(guard);
                return Err(widen_infallible(error));
            }
        };
        if let Err(error) = self.session.mechanisms.finish_prefill_reservation(guard) {
            self.session
                .fence_terminal(crate::DistributedExecutionPhase::PrefillCancellation);
            return Err(ReplicatedTextSessionError::Mechanism(error));
        }
        Ok(!all_continue)
    }
}

impl<A, B, M, D, P, O, K, R: PrefillSchedulingAuthority> PrefillExecutor<R>
    for SessionPrefill<'_, A, B, M, D, P, O, K, R>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
    O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    K: SpanLifecycle<A, B, M, D, P, O>,
{
    type Output = B::Tensor;
    type Completion = SettledPrefillCompletion<R>;
    type Error = ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>;

    fn agree_cancellation_at(
        &mut self,
        boundary: PrefillBoundary,
        cancellation: &GenerationCancellationToken,
        reservation: R,
    ) -> Result<bool, Self::Error> {
        self.agree_cancellation_at_boundary(reservation, boundary, || cancellation.is_cancelled())
    }

    fn submit_chunk(
        &mut self,
        chunk: &PrefillChunk,
        reservation: R,
    ) -> Result<Submission<Option<Self::Output>, Self::Completion>, Self::Error> {
        self.with_terminal_unwind(
            crate::DistributedExecutionPhase::PrefillReservation,
            |executor| executor.submit_chunk_retained(chunk, reservation),
        )
    }
}

impl<A, B, M, D, P, O, K, R: PrefillSchedulingAuthority> SessionPrefill<'_, A, B, M, D, P, O, K, R>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
    O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    K: SpanLifecycle<A, B, M, D, P, O>,
{
    fn submit_chunk_retained(
        &mut self,
        chunk: &PrefillChunk,
        reservation: R,
    ) -> Result<
        Submission<Option<B::Tensor>, SettledPrefillCompletion<R>>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.request
            .validate_same_request(&reservation)
            .map_err(ReplicatedTextSessionError::WorkingMemory)?;
        reservation
            .validate(self.session.inference_execution_identity(), self.geometry)
            .map_err(ReplicatedTextSessionError::WorkingMemory)?;
        self.session.ensure_control_unfenced()?;
        if !self.session.selected.exact_completion_available() {
            return Err(ReplicatedTextSessionError::Contract(
                "bounded prefill requires exact state completion".into(),
            ));
        }
        let guard = match self.session.coordinate_scheduling_entry(
            &reservation,
            Some(PrefillControlRole::for_chunk(
                PrefillSpanControlPhase::SpanOuter,
                chunk,
            )),
        ) {
            Ok(guard) => guard,
            Err(error) => {
                self.session
                    .fence_terminal(crate::DistributedExecutionPhase::PrefillReservation);
                return Err(error);
            }
        };
        let frontier = self
            .session
            .mechanisms
            .prefill_state_frontier(&self.session.state)
            .map_err(ReplicatedTextSessionError::Mechanism)
            .and_then(|actual| match actual {
                Some(actual) if actual != chunk.position => {
                    Err(ReplicatedTextSessionError::WorkingMemory(
                        WorkingMemoryError::StateFrontierMismatch {
                            expected: chunk.position,
                            actual,
                        },
                    ))
                }
                _ => Ok(()),
            });
        // Read-only validation joins the initial vote. The real epoch increment
        // remains inside the ordinary input transaction, at its original point.
        let epoch = (|| {
            self.session.ensure_control_unfenced()?;
            let epoch = checked_upcoming_epoch(
                self.session.next_commit_epoch,
                self.session.active_commit_epoch,
                self.session.last_commit_outcome,
            )?;
            if self.pending_retention.is_some() {
                return Err(ReplicatedTextSessionError::WorkingMemory(
                    WorkingMemoryError::IdentityMismatch,
                ));
            }
            if let Some(request) = reservation.inference_request() {
                self.operation
                    .validate_admission(
                        &self.source,
                        self.session.state.inference_retention(),
                        request,
                    )
                    .map_err(ReplicatedTextSessionError::WorkingMemory)?;
            } else {
                self.operation
                    .validate_speculative_admission(&self.source)
                    .map_err(ReplicatedTextSessionError::WorkingMemory)?;
            }
            Ok(epoch)
        })();
        let agreement = self.session.agree_execution_phase(
            crate::DistributedExecutionPhase::PrefillReservation,
            frontier.is_ok() && epoch.is_ok(),
            self.context,
        );
        if frontier.is_err() || epoch.is_err() || !matches!(agreement, Ok(true)) {
            drop(guard);
            frontier?;
            epoch?;
            agreement.map_err(widen_infallible)?;
            return Err(ReplicatedTextSessionError::Contract(
                "another rank rejected prefill reservation or state frontier".into(),
            ));
        }
        frontier?;
        let epoch = epoch?;
        // The resulting cache can outlive the executor and its completion. Keep
        // its charge with state before preparation, including failed mutations
        // and rollback. Native guards independently cover unresolved work.
        if let Some(request) = reservation.inference_request() {
            self.operation.admit(
                &mut self.source,
                self.session.state.inference_retention_mut(),
                request,
            );
        }
        // Preparation belongs inside native retention, including failures before
        // the session receives a borrowed input. Keep owned views through settling.
        let identity = (chunk.input.end == self.geometry.input_positions)
            .then(|| self.operation.cache_identity(&self.source))
            .flatten();
        // The observer's span admission shares the original preparation guard.
        // Its error follows the existing input agreement on every rank before
        // any prepared chunk can enter the model transaction.
        let retention_context = reservation
            .inference_request()
            .map(|request| PrefillChunkRetentionContext::new(request, chunk, epoch));
        let mut registration = None;
        let source = &mut self.source;
        let operation = &mut self.operation;
        let observer = &mut *self.observer;
        let context = self.context;
        let prepared = self.session.with_terminal_unwind(
            crate::DistributedExecutionPhase::InputPreparation,
            |session| {
                observer
                    .begin_prefill_chunk(chunk)
                    .map_err(ReplicatedTextSessionError::Architecture)
                    .and_then(|()| {
                        if let Some(retention_context) = &retention_context {
                        if session.mechanisms.requires_prefill_opening_sources() {
                            let paths = ReplicatedTextSession::<A, B, M, D>::validated_prefill_source_paths(
                                session.mechanisms.requires_prepared_prefill_sources(),
                                &session.execution,
                                session.observation_paths.get(),
                            )?;
                            let execution = crate::inspection::RuntimeOpeningExecution::new(
                                &session.execution,
                                D::visit_retained_values,
                            );
                            session
                                .mechanisms
                                .prepare_prefill_opening_sources(
                                    &session.state,
                                    &retention_context,
                                    &execution.borrow_prepared(paths),
                                )
                                .map_err(ReplicatedTextSessionError::Mechanism)?;
                        }
                        }
                        // Both source loans end before observer retention or input work.
                        Ok(())
                    })
                    .and_then(|()| {
                        let Some(retention_context) = &retention_context else { return Ok(None); };
                        let result = if observer.requires_prefill_opening_state() {
                            let source =
                                crate::inspection::RuntimeOpeningState::<B, _>::new(&session.state);
                            observer.prepare_prefill_chunk_with_opening(
                                &retention_context,
                                &source.borrow(),
                            )
                        } else {
                            observer.prepare_prefill_chunk_retention(&retention_context)
                        };
                        result.map_err(ReplicatedTextSessionError::Architecture)
                    })
                    .and_then(|retained| {
                        registration = retained;
                        if let Some(retained) = &registration {
                            retained
                                .validate_context(retention_context.as_ref().ok_or(ReplicatedTextSessionError::WorkingMemory(WorkingMemoryError::IdentityMismatch))?)
                                .map_err(ReplicatedTextSessionError::WorkingMemory)?;
                        }
                        operation
                            .prepare(source, chunk, context)
                            .map_err(ReplicatedTextSessionError::Architecture)
                    })
            },
        );
        let input = match &prepared {
            Ok(prepared) => Ok(prepared),
            Err(_) => {
                // Move the original typed error without translating its source.
                let error = prepared.err().expect("preparation failed");
                return Self::finish_guarded_chunk(
                    self.session,
                    self.observer,
                    self.context,
                    &mut self.operation,
                    &mut self.source,
                    identity,
                    guard,
                    Err(error),
                    chunk,
                    reservation,
                    epoch,
                    registration,
                    &mut self.pending_retention,
                );
            }
        };
        Self::finish_guarded_chunk(
            self.session,
            self.observer,
            self.context,
            &mut self.operation,
            &mut self.source,
            identity,
            guard,
            input,
            chunk,
            reservation,
            epoch,
            registration,
            &mut self.pending_retention,
        )
    }

    fn finish_guarded_chunk<'s>(
        session: &mut ReplicatedTextSession<A, B, M, D>,
        observer: &mut O,
        context: &<B::Tensor as Tensor>::Context,
        operation: &mut K,
        source: &mut P,
        identity: Option<SharedPreparedInputCacheIdentity>,
        guard: M::PrefillReservationGuard,
        input: Result<&'s K::Chunk, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
        chunk: &PrefillChunk,
        reservation: R,
        epoch: DistributedCommitEpoch,
        registration: Option<PreparedPrefillChunkRetention>,
        pending: &mut Option<SettledPrefillChunkRetention>,
    ) -> Result<
        Submission<Option<B::Tensor>, SettledPrefillCompletion<R>>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        let role = PrefillControlRole::for_chunk(PrefillSpanControlPhase::InputTransaction, chunk);
        let previous = session.active_prefill_control.replace(role);
        // Restore the scalar operation context on every unwind before the outer
        // existing terminal fence runs. Native guards/owners remain independent.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            operation.execute(session, source, input, identity, chunk, context, observer)
        }));
        session.active_prefill_control = previous;
        let result = match result {
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        };
        let (output, target_commit) = match result {
            Ok(output) => output,
            Err(error) => {
                operation.failed(source);
                drop(guard);
                return Err(error);
            }
        };
        if let Err(error) = session.mechanisms.finish_prefill_reservation(guard) {
            session.fence_terminal(crate::DistributedExecutionPhase::PrefillReservationCompletion);
            return Err(ReplicatedTextSessionError::Mechanism(error));
        }
        let retention_ready = registration.as_ref().map_or(Ok(()), |retained| {
            retained.validate_commit(
                &PrefillChunkRetentionContext::new(
                    reservation
                        .inference_request()
                        .ok_or(WorkingMemoryError::IdentityMismatch)?,
                    chunk,
                    epoch,
                ),
                target_commit,
            )
        });
        let agreement_guard = match session.coordinate_scheduling_entry(
            &reservation,
            Some(PrefillControlRole::for_chunk(
                PrefillSpanControlPhase::SettlementAgreement,
                chunk,
            )),
        ) {
            Ok(guard) => guard,
            Err(error) => {
                session
                    .fence_terminal(crate::DistributedExecutionPhase::PrefillReservationCompletion);
                retention_ready.map_err(ReplicatedTextSessionError::WorkingMemory)?;
                return Err(error);
            }
        };
        let agreement = session.agree_execution_phase(
            crate::DistributedExecutionPhase::PrefillReservationCompletion,
            retention_ready.is_ok(),
            context,
        );
        if retention_ready.is_err() || !matches!(agreement, Ok(true)) {
            drop(agreement_guard);
            retention_ready.map_err(ReplicatedTextSessionError::WorkingMemory)?;
            agreement.map_err(widen_infallible)?;
            return Err(ReplicatedTextSessionError::Contract(
                "another rank failed reserved prefill completion".into(),
            ));
        }
        if let Err(error) = session
            .mechanisms
            .finish_prefill_reservation(agreement_guard)
        {
            session.fence_terminal(crate::DistributedExecutionPhase::PrefillReservationCompletion);
            return Err(ReplicatedTextSessionError::Mechanism(error));
        }
        debug_assert!(pending.is_none(), "checked before chunk work");
        *pending = registration.map(PreparedPrefillChunkRetention::into_settled);
        operation.settled(source, session)?;
        Ok(Submission {
            output,
            completion: SettledPrefillCompletion {
                _reservation: reservation,
            },
        })
    }
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    fn coordinate_scheduling_entry<R: PrefillSchedulingAuthority>(
        &mut self,
        authority: &R,
        role: Option<PrefillControlRole>,
    ) -> Result<
        M::PrefillReservationGuard,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        match (
            authority.inference_request(),
            authority.speculative_schedule(),
        ) {
            (Some(request), None) => self
                .mechanisms
                .coordinate_prefill_entry(request.clone(), role)
                .map_err(ReplicatedTextSessionError::Mechanism),
            (None, Some(external)) => self
                .mechanisms
                .coordinate_speculative_prefill_entry(external, role)
                .map_err(ReplicatedTextSessionError::Mechanism)?
                .ok_or(ReplicatedTextSessionError::WorkingMemory(
                    WorkingMemoryError::UnknownBound,
                )),
            _ => Err(ReplicatedTextSessionError::WorkingMemory(
                WorkingMemoryError::IdentityMismatch,
            )),
        }
    }

    fn validate_speculative_scheduling(
        &self,
        authority: &crate::working_memory::SpeculativePrefillScheduleAuthority,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        use crate::DistributedExecutionPhase as Phase;
        for phase in [
            Phase::PrefillSourcePreparation,
            Phase::PrefillSourceSelection,
            Phase::PrefillReservation,
            Phase::PrefillCancellation,
            Phase::PrefillReservationCompletion,
        ] {
            if D::parallel_control_operation(
                &self.execution,
                super::ParallelControlEvent::Phase(phase),
            )
            .is_some()
            {
                if !self
                    .mechanisms
                    .has_speculative_parallel_control(authority)
                    .map_err(ReplicatedTextSessionError::Mechanism)?
                {
                    return Err(ReplicatedTextSessionError::WorkingMemory(
                        WorkingMemoryError::UnknownBound,
                    ));
                }
            }
        }
        Ok(())
    }

    /// Selects semantic ingress under the caller's exact request authority.
    /// A supplied request retains its original authority and geometry. Its geometry and opening state are checked before
    /// source construction; the same charge remains with committed state and
    /// governs later decode. Missing authority is a typed admission failure.
    ///
    /// This gateway returns ordinary final-position scores. Physical Sequence
    /// capture also requires its original accepted span/path view and current
    /// prepared execution binding; all observed rows use the same capture bank.
    pub fn try_prefill_source_cancellable<P, O>(
        &mut self,
        admitted: Option<&InferenceRequest>,
        shape: Option<[u64; 2]>,
        max_chunk_positions: Option<std::num::NonZeroU64>,
        make_source: impl FnOnce(InferenceGeometry) -> Result<Option<P>, A::Error>,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        PrefillSourceOutcome<B::Tensor>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        P: PreparedPrefillSource<A, B, M::State>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.try_prefill_score_source_progress_cancellable(
            admitted,
            shape,
            max_chunk_positions,
            make_source,
            cancellation,
            context,
            observer,
        )
        .map(|progress| progress.outcome)
    }

    fn try_prefill_score_source_progress_cancellable<P, O>(
        &mut self,
        admitted: Option<&InferenceRequest>,
        shape: Option<[u64; 2]>,
        max_chunk_positions: Option<std::num::NonZeroU64>,
        make_source: impl FnOnce(InferenceGeometry) -> Result<Option<P>, A::Error>,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        PrefillSourceProgress<B::Tensor>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        P: PreparedPrefillSource<A, B, M::State>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let progress = self.try_prefill_source_output_cancellable(
            admitted,
            shape,
            max_chunk_positions,
            OutputDemand::LastPosition,
            make_source,
            cancellation,
            context,
            observer,
        )?;
        let outcome = match progress.outcome {
            PrefillSourceOutcome::Unavailable => Ok(PrefillSourceOutcome::Unavailable),
            PrefillSourceOutcome::Cancelled => Ok(PrefillSourceOutcome::Cancelled),
            PrefillSourceOutcome::Complete(Some(output)) => {
                Ok(PrefillSourceOutcome::Complete(output))
            }
            PrefillSourceOutcome::Complete(None) => Err(ReplicatedTextSessionError::Contract(
                "completed score prefill has no final scores".into(),
            )),
        }?;
        Ok(PrefillSourceProgress {
            outcome,
            completed_positions: progress.completed_positions,
        })
    }

    /// Runs externally admitted spans through the shared scheduling worker.
    /// The scheduling issuer funds host control; each operation must retain its
    /// independent native occurrence and completion custody.
    pub fn try_prefill_speculative_source_with_operation<P, O, K>(
        &mut self,
        authority: crate::working_memory::SpeculativePrefillScheduleAuthority,
        shape: Option<[u64; 2]>,
        max_chunk_positions: Option<std::num::NonZeroU64>,
        output_demand: OutputDemand,
        make_source: impl FnOnce(InferenceGeometry) -> Result<Option<P>, A::Error>,
        cancellation: &GenerationCancellationToken,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        operation: K,
    ) -> Result<
        PrefillSourceProgress<Option<B::Tensor>>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        P: PreparedPrefillSource<A, B, M::State>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
        K: PrefillSpanOperation<A, B, M, D, P, O>,
    {
        self.try_prefill_source_with_authority(
            authority,
            shape,
            max_chunk_positions,
            output_demand,
            |_, authority| {
                make_source(authority.geometry()).map_err(ReplicatedTextSessionError::Architecture)
            },
            cancellation,
            context,
            observer,
            lifecycle::ExistingSpan(operation),
            false,
            None,
        )
    }

    /// Executes a custom span operation under an already admitted request.
    /// Source selection, state advancement and completion use the shared driver.
    pub fn try_prefill_admitted_source_with_operation<P, O, K>(
        &mut self,
        admitted: &InferenceRequest,
        shape: Option<[u64; 2]>,
        max_chunk_positions: Option<std::num::NonZeroU64>,
        output_demand: OutputDemand,
        make_source: impl FnOnce(InferenceGeometry) -> Result<Option<P>, A::Error>,
        cancellation: &GenerationCancellationToken,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        operation: K,
    ) -> Result<
        PrefillSourceProgress<Option<B::Tensor>>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        P: PreparedPrefillSource<A, B, M::State>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
        K: PrefillSpanOperation<A, B, M, D, P, O>,
    {
        self.try_prefill_source_output_with_operation(
            Some(admitted),
            shape,
            max_chunk_positions,
            output_demand,
            make_source,
            cancellation,
            context,
            observer,
            operation,
        )
    }

    fn try_prefill_source_output_cancellable<P, O>(
        &mut self,
        admitted: Option<&InferenceRequest>,
        shape: Option<[u64; 2]>,
        max_chunk_positions: Option<std::num::NonZeroU64>,
        output_demand: OutputDemand,
        make_source: impl FnOnce(InferenceGeometry) -> Result<Option<P>, A::Error>,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        PrefillSourceProgress<Option<B::Tensor>>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        P: PreparedPrefillSource<A, B, M::State>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.try_prefill_source_output_with_operation(
            admitted,
            shape,
            max_chunk_positions,
            output_demand,
            make_source,
            cancellation,
            context,
            observer,
            OrdinaryPrefillSpan,
        )
    }

    fn try_prefill_source_output_with_operation<P, O, K>(
        &mut self,
        admitted: Option<&InferenceRequest>,
        shape: Option<[u64; 2]>,
        max_chunk_positions: Option<std::num::NonZeroU64>,
        output_demand: OutputDemand,
        make_source: impl FnOnce(InferenceGeometry) -> Result<Option<P>, A::Error>,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        operation: K,
    ) -> Result<
        PrefillSourceProgress<Option<B::Tensor>>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        P: PreparedPrefillSource<A, B, M::State>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
        K: PrefillSpanOperation<A, B, M, D, P, O>,
    {
        self.try_prefill_source_with_lifecycle(
            admitted,
            shape,
            max_chunk_positions,
            output_demand,
            |_, request| {
                make_source(request.geometry()).map_err(ReplicatedTextSessionError::Architecture)
            },
            cancellation,
            context,
            observer,
            lifecycle::ExistingSpan(operation),
            false,
            None,
        )
    }

    pub(super) fn try_prefill_source_with_lifecycle<P, O, K>(
        &mut self,
        admitted: Option<&InferenceRequest>,
        shape: Option<[u64; 2]>,
        max_chunk_positions: Option<std::num::NonZeroU64>,
        output_demand: OutputDemand,
        make_source: impl FnOnce(
            &Self,
            &InferenceRequest,
        ) -> Result<
            Option<P>,
            ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
        >,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        operation: K,
        retained_media: bool,
        media_metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<
        PrefillSourceProgress<Option<B::Tensor>>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
        K: SpanLifecycle<A, B, M, D, P, O>,
    {
        let request = admitted.ok_or_else(|| {
            ReplicatedTextSessionError::BeforeStateMutation(Box::new(
                ReplicatedTextSessionError::WorkingMemory(WorkingMemoryError::UnknownBound),
            ))
        })?;
        self.try_prefill_source_with_authority(
            request.clone(),
            shape,
            max_chunk_positions,
            output_demand,
            make_source,
            cancellation,
            context,
            observer,
            operation,
            retained_media,
            media_metadata,
        )
    }

    pub(super) fn try_prefill_source_with_authority<P, O, K, R: PrefillSchedulingAuthority>(
        &mut self,
        request: R,
        shape: Option<[u64; 2]>,
        max_chunk_positions: Option<std::num::NonZeroU64>,
        output_demand: OutputDemand,
        make_source: impl FnOnce(
            &Self,
            &R,
        ) -> Result<
            Option<P>,
            ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
        >,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        operation: K,
        retained_media: bool,
        media_metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<
        PrefillSourceProgress<Option<B::Tensor>>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
        K: SpanLifecycle<A, B, M, D, P, O>,
    {
        self.ensure_control_unfenced()?;
        let frontier = self.mechanisms.prefill_state_frontier(&self.state);
        // A failed original input still enters protocol selection with peers;
        // it returns to the ordinary result-bearing admission path below.
        let valid_shape = shape.filter(|shape| shape[0] > 0 && shape[1] > 0);
        let ordinary = request.inference_request().is_some();
        if !ordinary {
            self.validate_speculative_scheduling(request.speculative_schedule().ok_or(
                ReplicatedTextSessionError::WorkingMemory(WorkingMemoryError::IdentityMismatch),
            )?)?;
            if retained_media
                || observer.transactional()
                || observer.requires_prepared_traversal()
                || observer.requires_sequence_readout()
                || observer.requires_prefill_opening_state()
                || operation.score_layout() != PrefillScoreLayout::SelectedPositions
            {
                return Err(ReplicatedTextSessionError::WorkingMemory(
                    WorkingMemoryError::UnknownBound,
                ));
            }
        }
        let geometry = request.geometry();
        if geometry.output != output_demand && !ordinary {
            return Err(ReplicatedTextSessionError::WorkingMemory(
                WorkingMemoryError::OutputDemandMismatch {
                    admitted: geometry.output,
                    required: output_demand,
                },
            ));
        }
        let execution = self.inference_execution_identity().clone();
        let reject_request = |error| {
            ReplicatedTextSessionError::BeforeStateMutation(Box::new(
                ReplicatedTextSessionError::WorkingMemory(error),
            ))
        };
        request
            .validate(&execution, geometry)
            .map_err(reject_request)?;
        // This ordinary media path cannot reinterpret even a converted original
        // reservation as unbudgeted. Reject before its plan factory or driver claim.
        if retained_media
            && ordinary
            && media_metadata.is_none_or(|metadata| metadata.metadata_funding().is_none())
        {
            return Err(reject_request(WorkingMemoryError::UnknownBound));
        }
        // Authenticate the original cold path seal and current execution token
        // before the one-use driver claim or a potentially allocating factory.
        // These immutable loans end before any observer or source callback.
        let ordinary_capture = if let Some(capture) = observer.ordinary_prefill_capture() {
            if !retained_media
                || ordinary
                || !observer.requires_prepared_traversal()
                || !observer.transactional()
            {
                return Err(reject_request(WorkingMemoryError::IdentityMismatch));
            }
            let paths = Self::validated_prefill_source_paths(
                true,
                &self.execution,
                self.observation_paths.get(),
            )
            .map_err(|error| ReplicatedTextSessionError::BeforeStateMutation(Box::new(error)))?
            .ok_or_else(|| reject_request(WorkingMemoryError::IdentityMismatch))?;
            capture
                .validate(capture.source(), paths.source(), geometry)
                .map_err(|_| reject_request(WorkingMemoryError::IdentityMismatch))?;
            true
        } else {
            false
        };
        let accepted_capture = if let Some(capture) = observer.admitted_prefill_capture() {
            if !ordinary
                || !observer.requires_prepared_traversal()
                || !observer.transactional()
                || (retained_media && !capture.selection().selection().is_prepared_media())
            {
                return Err(reject_request(WorkingMemoryError::IdentityMismatch));
            }
            capture
                .validate_request(
                    request
                        .inference_request()
                        .ok_or_else(|| reject_request(WorkingMemoryError::IdentityMismatch))?,
                )
                .map_err(reject_request)?;
            let paths = Self::validated_prefill_source_paths(
                true,
                &self.execution,
                self.observation_paths.get(),
            )
            .map_err(|error| ReplicatedTextSessionError::BeforeStateMutation(Box::new(error)))?
            .expect("required current prepared paths");
            if !paths
                .source()
                .same_storage(capture.selection().selection().paths())
            {
                return Err(reject_request(WorkingMemoryError::IdentityMismatch));
            }
            true
        } else if let Some(capture) = observer.admitted_capture_continuation() {
            if !ordinary
                || !observer.requires_prepared_traversal()
                || !observer.transactional()
                || (retained_media && !capture.selection().is_prepared_media())
            {
                return Err(reject_request(WorkingMemoryError::IdentityMismatch));
            }
            capture
                .validate_request(
                    request
                        .inference_request()
                        .ok_or_else(|| reject_request(WorkingMemoryError::IdentityMismatch))?,
                )
                .map_err(reject_request)?;
            let paths = Self::validated_prefill_source_paths(
                true,
                &self.execution,
                self.observation_paths.get(),
            )
            .map_err(|error| ReplicatedTextSessionError::BeforeStateMutation(Box::new(error)))?
            .ok_or_else(|| reject_request(WorkingMemoryError::IdentityMismatch))?;
            if !paths.source().same_storage(capture.selection().paths()) {
                return Err(reject_request(WorkingMemoryError::IdentityMismatch));
            }
            true
        } else {
            ordinary_capture
        };
        let mut admitted_driver = {
            let check = (|| {
                if !self.selected.exact_completion_available() {
                    return Err(WorkingMemoryError::CompletionUnavailable);
                }
                let actual = valid_shape.ok_or(WorkingMemoryError::IdentityMismatch)?;
                let expected = [geometry.batch_size, geometry.input_positions];
                if actual != expected {
                    return Err(WorkingMemoryError::SpanShapeMismatch { actual, expected });
                }
                if max_chunk_positions
                    .is_some_and(|chunk| chunk.get() != geometry.prefill_chunk_positions)
                {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                let output_supported = match geometry.output {
                    OutputDemand::LastPosition => !observer.requires_sequence_readout(),
                    OutputDemand::Sequence => {
                        accepted_capture
                            || (!observer.requires_sequence_readout()
                                && output_demand == OutputDemand::Sequence)
                    }
                    OutputDemand::StateOnly => {
                        !observer.requires_sequence_readout()
                            && output_demand == OutputDemand::StateOnly
                    }
                };
                if !output_supported {
                    return Err(WorkingMemoryError::OutputDemandMismatch {
                        admitted: geometry.output,
                        required: if observer.requires_sequence_readout() {
                            OutputDemand::Sequence
                        } else {
                            OutputDemand::LastPosition
                        },
                    });
                }
                if let Ok(actual) = &frontier {
                    let actual = actual.ok_or(WorkingMemoryError::UnknownBound)?;
                    if actual != geometry.cached_positions {
                        return Err(WorkingMemoryError::StateFrontierMismatch {
                            expected: geometry.cached_positions,
                            actual,
                        });
                    }
                }
                Ok(())
            })();
            check.map_err(reject_request)?;
            // Claim one-use authority before invoking a potentially allocating
            // source factory. Retention clones cannot prepare a second request.
            Some(
                crate::prefill::PrefillDriver::new_scheduled(
                    &execution,
                    request.clone(),
                    geometry,
                    cancellation.clone(),
                )
                .map_err(reject_request)?,
            )
        };
        let guard = match self.with_terminal_unwind(
            crate::DistributedExecutionPhase::PrefillSourcePreparation,
            |session| {
                session.coordinate_scheduling_entry(
                    &request,
                    Some(PrefillControlRole::SourcePreparation),
                )
            },
        ) {
            Ok(guard) => guard,
            Err(error) => {
                self.fence_terminal(crate::DistributedExecutionPhase::PrefillSourcePreparation);
                return Err(match frontier {
                    Err(prior) => ReplicatedTextSessionError::Mechanism(prior),
                    Ok(_) => error,
                });
            }
        };
        let source = self.with_terminal_unwind(
            crate::DistributedExecutionPhase::PrefillSourcePreparation,
            |session| {
                frontier
                    .map_err(ReplicatedTextSessionError::Mechanism)
                    .and_then(|_| {
                        if retained_media
                            && !accepted_capture
                            && (observer.requires_prepared_traversal()
                                || observer.requires_sequence_readout()
                                || observer.transactional())
                        {
                            return Err(ReplicatedTextSessionError::Contract(
                                "media capture/transaction attribution is not admitted".into(),
                            ));
                        }
                        if !retained_media
                            && (valid_shape.is_none()
                                || (observer.requires_sequence_readout() && !accepted_capture))
                        {
                            Ok(None)
                        } else {
                            make_source(session, &request)
                        }
                    })
            },
        );
        let prepared = self.agree_execution_phase(
            crate::DistributedExecutionPhase::PrefillSourcePreparation,
            source.is_ok(),
            context,
        );
        let selected = if matches!(prepared, Ok(true)) {
            self.agree_execution_phase(
                crate::DistributedExecutionPhase::PrefillSourceSelection,
                source.as_ref().is_ok_and(Option::is_some),
                context,
            )
        } else {
            Ok(false)
        };
        if prepared.is_err() || selected.is_err() {
            drop(guard);
            // A terminal agreement failed. Preserve a preceding local cause,
            // but do not advertise retry-safe source rejection on this path.
            source?;
            prepared.map_err(widen_infallible)?;
            selected.map_err(widen_infallible)?;
            unreachable!("a checked communication error was returned");
        }
        if source.is_err() || matches!(prepared, Ok(false)) {
            drop(guard);
            // A completed negative vote preserves the ordinary classification:
            // no state mutation occurred. Abandonment is not native completion.
            return Err(ReplicatedTextSessionError::BeforeStateMutation(Box::new(
                match source {
                    Err(error) => error,
                    Ok(_) => ReplicatedTextSessionError::Contract(
                        "another rank rejected scheduled input preparation".into(),
                    ),
                },
            )));
        }
        if let Err(error) = self.with_terminal_unwind(
            crate::DistributedExecutionPhase::PrefillSourcePreparation,
            |session| session.mechanisms.finish_prefill_reservation(guard),
        ) {
            self.fence_terminal(crate::DistributedExecutionPhase::PrefillSourcePreparation);
            return Err(ReplicatedTextSessionError::Mechanism(error));
        }
        if !selected.map_err(widen_infallible)? {
            if ordinary || retained_media {
                return Err(reject_request(
                    WorkingMemoryError::PreparedSourceUnavailable,
                ));
            }
            return Ok(PrefillSourceProgress {
                outcome: PrefillSourceOutcome::Unavailable,
                completed_positions: 0,
            });
        }
        let source = source?.expect("every rank selected a prepared source");
        let score_layout = operation.score_layout();
        // Per-chunk state commits remain independent. A prediction-level host
        // observer stays provisional through final cancellation, output indexing
        // and the last reservation settlement; any early return aborts it.
        let observation =
            crate::inspection::PrefillObservationGuard::<B::Tensor, A::Error, _>::new(observer);
        let mut executor = SessionPrefill::new_with_operation(
            self,
            source,
            request.clone(),
            context,
            &mut *observation.observer,
            operation,
        )
        .map_err(ReplicatedTextSessionError::WorkingMemory)?;
        let mut driver = match admitted_driver.take() {
            Some(driver) => driver,
            None => crate::prefill::PrefillDriver::new_scheduled(
                &execution,
                request.clone(),
                geometry,
                cancellation.clone(),
            )
            .map_err(ReplicatedTextSessionError::WorkingMemory)?,
        };
        let outcome = driver.run_final(&mut executor);
        let completed_positions = driver.completed_positions();
        drop(executor);
        let output = match outcome {
            Ok((crate::prefill::PrefillOutcome::Complete, output)) => output,
            Ok((crate::prefill::PrefillOutcome::Cancelled, _)) => {
                return Ok(PrefillSourceProgress {
                    outcome: PrefillSourceOutcome::Cancelled,
                    completed_positions,
                });
            }
            Err(crate::prefill::PrefillError::Submission(error)) => return Err(error),
            Err(crate::prefill::PrefillError::Completion(never)) => match never {},
            Err(error) => return Err(ReplicatedTextSessionError::Contract(error.to_string())),
        };
        if geometry.output == OutputDemand::StateOnly {
            if output.is_some() {
                return Err(ReplicatedTextSessionError::Contract(
                    "completed state-only prefill retained scores".into(),
                ));
            }
            self.with_terminal_unwind(
                crate::DistributedExecutionPhase::PrefillReservationCompletion,
                |_| observation.finish(true),
            );
            return Ok(PrefillSourceProgress {
                outcome: PrefillSourceOutcome::Complete(None),
                completed_positions,
            });
        }
        let output = output.ok_or_else(|| {
            ReplicatedTextSessionError::Contract(
                "completed ordinary prefill has no final scores".into(),
            )
        })?;
        if score_layout == PrefillScoreLayout::SelectedPositions {
            self.with_terminal_unwind(
                crate::DistributedExecutionPhase::PrefillReservationCompletion,
                |_| observation.finish(true),
            );
            return Ok(PrefillSourceProgress {
                outcome: PrefillSourceOutcome::Complete(Some(output)),
                completed_positions,
            });
        }
        // Preserve the ordinary public score axes, with native indexing also
        // protected by the same request's completion/recovery retention.
        let guard = match self.with_terminal_unwind(
            crate::DistributedExecutionPhase::PrefillReservationCompletion,
            |session| {
                session.coordinate_scheduling_entry(&request, Some(PrefillControlRole::FinalIndex))
            },
        ) {
            Ok(guard) => guard,
            Err(error) => {
                self.fence_terminal(crate::DistributedExecutionPhase::PrefillReservationCompletion);
                return Err(error);
            }
        };
        let output = self.with_terminal_unwind(
            crate::DistributedExecutionPhase::PrefillReservationCompletion,
            |session| {
                session.mechanisms.index_text_output(
                    output,
                    session.output_selection.sequence_index(),
                    context,
                )
            },
        );
        let output = match output {
            Ok(output) => output,
            Err(error) => {
                drop(guard);
                return Err(ReplicatedTextSessionError::Mechanism(error));
            }
        };
        if let Err(error) = self.with_terminal_unwind(
            crate::DistributedExecutionPhase::PrefillReservationCompletion,
            |session| session.mechanisms.finish_prefill_reservation(guard),
        ) {
            self.fence_terminal(crate::DistributedExecutionPhase::PrefillReservationCompletion);
            return Err(ReplicatedTextSessionError::Mechanism(error));
        }
        self.with_terminal_unwind(
            crate::DistributedExecutionPhase::PrefillReservationCompletion,
            |_| observation.finish(true),
        );
        Ok(PrefillSourceProgress {
            outcome: PrefillSourceOutcome::Complete(Some(output)),
            completed_positions,
        })
    }
}

// No epoch authority is changed. The actual session fields enter this shared
// check before the existing initial reservation readiness vote.
fn checked_upcoming_epoch<A: std::fmt::Display, P: std::fmt::Display, M: std::fmt::Display>(
    next: DistributedCommitEpoch,
    active: Option<DistributedCommitEpoch>,
    outcome: Option<DistributedCommitOutcome>,
) -> Result<DistributedCommitEpoch, ReplicatedTextSessionError<A, P, M>> {
    if let Some(DistributedCommitOutcome::Indeterminate { epoch, phase }) = outcome {
        return Err(ReplicatedTextSessionError::CommitIndeterminate { epoch, phase });
    }
    if active.is_some() {
        return Err(ReplicatedTextSessionError::Contract(
            "distributed transaction already has an active commit epoch".into(),
        ));
    }
    next.next().ok_or_else(|| {
        ReplicatedTextSessionError::Contract("distributed commit epoch overflow".into())
    })?;
    Ok(next)
}
#[cfg(test)]
#[path = "prefill/retention_tests.rs"]
mod retention_tests;
