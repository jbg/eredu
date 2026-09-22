//! Funded record delivery composed over the canonical prepared chat session.
use super::{
    observed::TraceBudget, ConstraintError, GenerationSnapshotError, LoadedModel,
    ObservedGenerationEvent, PreparedChatOutputMode, PreparedChatRequest, PreparedChatSession,
    PreparedChatSessionError, TraceLimits,
};
use crate::runtime::generation::storage::SnapshotStorage;
use eredu_core::generation::SemanticEvent;
use eredu_core::{capture::*, execution_control::*, generation::*, *};
use eredu_runtime::execution_control::TextSnapshotBackend;
use eredu_runtime::execution_control::{
    SamplingOverride, SamplingOverrideError, SamplingStateFacts, SnapshotBudget,
    TextSamplingControlBackend, TokenChoiceError,
};
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, mem::size_of, ops::ControlFlow, time::Instant};

type ControlConstraintError = TokenChoiceError<ConstraintError>;
mod branch;
mod configuration_identity;
mod delivery;
mod failure;
mod prepared;
pub(crate) mod records;
mod snapshot;
pub use branch::{ControlledGenerationBranch, GenerationBranchOptions};
use delivery::Delivery;
pub use failure::ControlledSessionFailure;
pub use records::*;
use records::{BranchInfo, ControlEvent, PromptRecord, RecordContext, SnapshotInfo};
pub use snapshot::{
    ControlledGenerationSnapshot, GenerationOutputCheckpoint, GenerationOutputCheckpointData,
};

/// Original preparation, execution, record or copy failure.
#[derive(Debug, thiserror::Error)]
pub enum ControlledGenerationError {
    #[error(transparent)]
    Record(#[from] RecordConstructionError),
    #[error(transparent)]
    Session(#[from] PreparedChatSessionError),
    #[error(transparent)]
    Failed(ControlledSessionFailure),
    #[error("{record}")]
    RecordedFailure {
        #[source]
        record: RecordConstructionError,
        session: ControlledSessionFailure,
    },
    #[error("{capture}")]
    CaptureFailure {
        #[source]
        capture: CaptureError,
        session: ControlledSessionFailure,
    },
    #[error(transparent)]
    Snapshot(#[from] GenerationSnapshotError),
    #[error(transparent)]
    Control(#[from] ExecutionControlError),
    #[error(transparent)]
    Capture(#[from] CaptureError),
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    #[error("{0}")]
    Rejected(&'static str),
}
impl ControlledGenerationError {
    /// The canonical session failure, retaining its original source and prefix.
    pub fn session_failure(&self) -> Option<&PreparedChatSessionError> {
        match self {
            Self::Session(error) => Some(error),
            Self::Failed(error) => Some(error.error()),
            Self::RecordedFailure { session, .. } | Self::CaptureFailure { session, .. } => {
                Some(session.error())
            }
            _ => None,
        }
    }
    /// Borrow the provider's neutral failure without replacing its typed source.
    pub fn backend_failure(&self) -> Option<&BackendFailure> {
        match self {
            Self::Backend(error) => Some(error),
            Self::Snapshot(error) => match error.cause() {
                eredu_runtime::execution_control::TextSnapshotError::Backend(error)
                | eredu_runtime::execution_control::TextSnapshotError::HostPreparation(error) => {
                    Some(error)
                }
                error @ eredu_runtime::execution_control::TextSnapshotError::Resume(_) => {
                    error.resume_backend_failure()
                }
                eredu_runtime::execution_control::TextSnapshotError::RetainedBackend(error) => {
                    std::error::Error::source(error)?.downcast_ref()
                }
                _ => None,
            },
            _ => self.session_failure()?.backend_failure(),
        }
    }
}

/// The prepared session owns all model, cursor, decoder, sampling and lifecycle
/// state. This wrapper owns only synchronous record delivery and its journal.
pub struct ControlledGenerationSession<'a, B: TextGenerationBackend> {
    session: Option<PreparedChatSession<'a, B>>,
    failed: bool,
    failure: ControlledSessionFailure,
    tokenizer_identity: [u8; 32],
    snapshot_budget: Option<SnapshotBudget>,
    snapshot_host_capacity: eredu_core::MemoryLimitDeclarations,
    native_copy_limits: eredu_runtime::working_memory::WorkspaceCopyLimits,
    capabilities: ExecutionControlCapabilities,
    delivery: Delivery,
    // Copied record prefix destinations retire after all journal payloads.
    journal_destination: Option<HostPreparationAuthority>,
}
impl<'a, B: TextGenerationBackend> ControlledGenerationSession<'a, B> {
    fn active(&self) -> Result<&PreparedChatSession<'a, B>, ControlledGenerationError> {
        if self.failed {
            return Err(ControlledGenerationError::Rejected(
                "recorded generation has failed",
            ));
        }
        self.session
            .as_ref()
            .ok_or(ControlledGenerationError::Rejected(
                "recorded generation has failed",
            ))
    }
    fn active_mut(&mut self) -> Result<&mut PreparedChatSession<'a, B>, ControlledGenerationError> {
        if self.failed {
            return Err(ControlledGenerationError::Rejected(
                "recorded generation has failed",
            ));
        }
        self.session
            .as_mut()
            .ok_or(ControlledGenerationError::Rejected(
                "recorded generation has failed",
            ))
    }
    pub fn timing(&self) -> GenerationTiming {
        self.delivery.timing
    }
    pub fn status(&self) -> GenerationStatus {
        if self.failed {
            GenerationStatus::Failed
        } else {
            self.session
                .as_ref()
                .map_or(GenerationStatus::Failed, PreparedChatSession::status)
        }
    }
    pub fn next_prediction(&self) -> u64 {
        self.session.as_ref().map_or(
            self.delivery.prediction,
            PreparedChatSession::next_prediction,
        )
    }
    pub fn token_ids(&self) -> &[u32] {
        self.session
            .as_ref()
            .map_or_else(|| self.failure.prefix(), PreparedChatSession::token_ids)
    }
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.session
            .as_ref()
            .and_then(PreparedChatSession::finish_reason)
    }
    pub fn control_handle(&self) -> GenerationControlHandle {
        self.delivery.control.clone()
    }
    pub fn emitted_bytes(&self) -> u64 {
        self.delivery.budget.emitted_bytes()
    }
    pub fn prompt_attribution(&self) -> &PreparedPromptAttribution {
        self.delivery.prompt.prepared().attribution()
    }
    pub fn semantic_snapshot_bytes(&self) -> Option<u64> {
        self.session
            .as_ref()?
            .semantic_snapshot_bytes()?
            .checked_add(self.delivery.semantic_prefix.snapshot_bytes()?)
    }
    /// Borrowed report built under the original preparation account.
    pub fn capabilities(&self) -> &ExecutionControlCapabilities {
        &self.capabilities
    }
    pub fn force_next_token(&mut self, token: u32) -> Result<(), ControlledGenerationError> {
        self.require_live_control()?;
        self.active_mut()?.force_next_token(token)?;
        Ok(())
    }
    pub fn clear_forced_token(&mut self) -> Result<bool, ControlledGenerationError> {
        self.require_live_control()?;
        Ok(self.active_mut()?.clear_forced_token()?)
    }
    pub fn pending_forced_token(&self) -> Option<u32> {
        self.session
            .as_ref()
            .and_then(PreparedChatSession::pending_forced_token)
    }
    fn require_live_control(&self) -> Result<(), ControlledGenerationError> {
        self.active()?;
        if self.delivery.control.cancellation().is_cancelled() {
            return Err(ControlledGenerationError::Rejected(
                "generation cancellation was requested",
            ));
        }
        Ok(())
    }
    fn lifecycle_record(
        &mut self,
        emit: &mut impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) {
        self.delivery.send(
            ObservedGenerationEvent::Lifecycle {
                status: self.status(),
                next_prediction: self.next_prediction(),
            },
            emit,
        );
    }
    fn delivery_result(&mut self) -> Result<(), ControlledGenerationError> {
        let local = match self.delivery.record_failure.take() {
            Some(error) => Err(error.into()),
            None => match self.delivery.failure.take() {
                Some(error) => Err(error.into()),
                None => Ok((!self.delivery.control.cancellation().is_cancelled()).then_some(())),
            },
        };
        match self
            .active()?
            .finish_record_delivery(local, ControlledGenerationError::Backend)
        {
            Ok(Some(())) => Ok(()),
            Ok(None) => {
                self.delivery.control.cancel();
                Ok(())
            }
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }
    /// Advances the same prepared session once. Capture precedes semantic output;
    /// record refusal participates in the same completed-delivery agreement.
    pub fn step(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus, ControlledGenerationError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.step_inner(emit))) {
            Ok(result) => result,
            Err(payload) => {
                self.failed = true;
                std::panic::resume_unwind(payload)
            }
        }
    }
    fn step_inner(
        &mut self,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus, ControlledGenerationError> {
        self.active()?;
        if matches!(
            self.status(),
            GenerationStatus::Completed | GenerationStatus::Cancelled
        ) {
            return Err(ExecutionControlError::Transition {
                from: self.status(),
                to: GenerationStatus::Running,
            }
            .into());
        }
        let cancellation = self.delivery.control.cancellation().clone();
        let session = self.session.take().expect("checked live session");
        let result = {
            let delivery = RefCell::new((&mut self.delivery, &mut emit));
            let mut observer = |token, captures, facts: crate::api::request::TokenDeliveryFacts| {
                let mut cell = delivery.borrow_mut();
                let (delivery, emit) = &mut *cell;
                delivery.timing = facts.timing;
                delivery.token(token, facts.forced, captures, facts.step_seconds, emit);
            };
            let failure = || {
                let cell = delivery.borrow();
                (cell.0.record_failure.is_some() || cell.0.failure.is_some())
                    .then_some(CaptureError::Overflow)
            };
            let mut semantic = |event| {
                let mut cell = delivery.borrow_mut();
                let (delivery, emit) = &mut *cell;
                delivery.send(
                    ObservedGenerationEvent::Semantic {
                        prediction_index: delivery.prediction.checked_sub(1),
                        event,
                    },
                    emit,
                );
            };
            session.advance_with_delivery(
                &cancellation,
                Some(&mut observer),
                Some(&failure),
                &mut semantic,
            )
        };
        match result {
            Ok(session) => {
                self.delivery.timing = session.timing();
                self.session = Some(session);
            }
            Err(error) => {
                self.failed = true;
                let shared = self.failure.publish(error);
                // The typed source remains in `shared`; the diagnostic record
                // has fixed qualified text, never an opaque provider formatter.
                let diagnostic = records::construction::string(
                    "prepared chat advancement failed",
                    &self.delivery.funding,
                );
                match diagnostic {
                    Ok(message) => self.delivery.send(
                        ObservedGenerationEvent::Failed {
                            message,
                            elapsed_seconds: self.delivery.started.elapsed().as_secs_f64(),
                        },
                        &mut emit,
                    ),
                    Err(cause) if self.delivery.record_failure.is_none() => {
                        self.delivery.refuse_record(cause)
                    }
                    Err(_) => {}
                }
                return Err(match self.delivery.record_failure.take() {
                    Some(record) => ControlledGenerationError::RecordedFailure {
                        record,
                        session: shared,
                    },
                    None => match self.delivery.failure.take() {
                        Some(capture) => ControlledGenerationError::CaptureFailure {
                            capture,
                            session: shared,
                        },
                        None => ControlledGenerationError::Failed(shared),
                    },
                });
            }
        }
        if let Some(reason) = self.finish_reason() {
            self.delivery.send(
                ObservedGenerationEvent::Completed {
                    reason,
                    generated_tokens: self.token_ids().len() as u64,
                    elapsed_seconds: self.delivery.started.elapsed().as_secs_f64(),
                },
                &mut emit,
            );
        }
        self.lifecycle_record(&mut emit);
        self.delivery_result()?;
        Ok(self.status())
    }
    pub fn pause(
        &mut self,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.active_mut()?.pause()?;
            self.delivery.control.request_pause();
            self.lifecycle_record(&mut emit);
            self.delivery_result()
        })) {
            Ok(result) => result,
            Err(payload) => {
                self.failed = true;
                std::panic::resume_unwind(payload)
            }
        }
    }
    pub fn cancel(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus, ControlledGenerationError> {
        self.active()?;
        if !matches!(
            self.status(),
            GenerationStatus::Prepared | GenerationStatus::Paused
        ) {
            return Err(ExecutionControlError::Transition {
                from: self.status(),
                to: GenerationStatus::Cancelled,
            }
            .into());
        }
        self.delivery.control.cancel();
        self.step(emit)
    }
    pub fn run(
        &mut self,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus, ControlledGenerationError> {
        while matches!(
            self.status(),
            GenerationStatus::Prepared | GenerationStatus::Paused
        ) {
            if self.delivery.control.pause_requested()
                && !self.delivery.control.cancellation().is_cancelled()
            {
                self.pause(&mut emit)?;
                break;
            }
            self.step(&mut emit)?;
        }
        Ok(self.status())
    }
    pub fn resume(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus, ControlledGenerationError> {
        self.active_mut()?.pause()?;
        self.delivery.control.acknowledge_resume();
        self.run(emit)
    }
}
impl<B: TextSamplingControlBackend> ControlledGenerationSession<'_, B> {
    pub fn sampling_state(&mut self) -> Result<SamplingStateFacts, ControlledGenerationError> {
        self.require_live_control()?;
        Ok(self.active_mut()?.sampling_state()?)
    }
    pub fn override_sampling(
        &mut self,
        request: SamplingOverride,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError> {
        self.require_live_control()?;
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let before = self.active_mut()?.sampling_state()?;
            let after = self.active_mut()?.override_sampling(request)?;
            self.delivery.send(
                ObservedGenerationEvent::SamplingChanged {
                    next_prediction: self.next_prediction(),
                    request,
                    before,
                    after,
                },
                &mut emit,
            );
            self.delivery_result()
        })) {
            Ok(result) => result,
            Err(payload) => {
                self.failed = true;
                std::panic::resume_unwind(payload)
            }
        }
    }
}

pub(crate) fn map_continuation_failure<E: std::error::Error + 'static>(
    error: TextContinuationError<E, ControlConstraintError>,
    convert: impl FnOnce(E) -> BackendFailure,
) -> TextContinuationError<BackendFailure, ControlConstraintError> {
    match error {
        TextContinuationError::IncompatibleDriver => TextContinuationError::IncompatibleDriver,
        TextContinuationError::Failed => TextContinuationError::Failed,
        TextContinuationError::NotQuiescent => TextContinuationError::NotQuiescent,
        TextContinuationError::Generation(error) => {
            TextContinuationError::Generation(match error {
                ControlledTextGenerationError::Backend(error) => {
                    ControlledTextGenerationError::Backend(convert(error))
                }
                ControlledTextGenerationError::Controller(error) => {
                    ControlledTextGenerationError::Controller(error)
                }
                ControlledTextGenerationError::Preparation(error) => {
                    ControlledTextGenerationError::Preparation(error)
                }
            })
        }
    }
}

pub(crate) fn map_sampling_failure<E: std::error::Error + 'static>(
    error: SamplingOverrideError<E>,
    convert: impl FnOnce(E) -> BackendFailure,
) -> SamplingOverrideError<BackendFailure> {
    match error {
        SamplingOverrideError::Invalid(reason) => SamplingOverrideError::Invalid(reason),
        SamplingOverrideError::Backend(error) => SamplingOverrideError::Backend(convert(error)),
    }
}

pub(crate) fn map_snapshot_failure<E: std::error::Error + 'static>(
    error: eredu_runtime::execution_control::TextSnapshotError<E>,
    convert: impl FnOnce(E) -> BackendFailure,
) -> eredu_runtime::execution_control::TextSnapshotError<BackendFailure> {
    use eredu_runtime::execution_control::TextSnapshotError as S;
    match error {
        S::Host(reason) => S::Host(reason),
        S::HostPreparation(error) => S::HostPreparation(error),
        S::HostCopy(error) => S::HostCopy(error),
        S::Resume(error) => S::Resume(error),
        S::HostAdmission(error) => S::HostAdmission(error),
        S::Controller(reason) => S::Controller(reason),
        S::Backend(error) => S::Backend(convert(error)),
        S::RetainedBackend(error) => S::RetainedBackend(error.map_cause(convert)),
        S::Capture(error) => S::Capture(error),
        S::Control(error) => S::Control(error),
        S::IncompatibleRun => S::IncompatibleRun,
        S::InconsistentState => S::InconsistentState,
        S::Unsupported(reason) => S::Unsupported(reason),
    }
}

impl<B: TextGenerationBackend> LoadedModel<B> {
    /// Agrees a caller-owned preparation result before entering a distributed run.
    /// Use this around fallible media or opaque-prompt preparation. Every peer
    /// must call the same stage in the same order, including on local failure.
    /// Local errors are preserved; `map_backend` translates peer/transport failure.
    /// Charges are cumulative, and rejection alone never proves native completion.
    pub fn finish_text_preparation<T, E>(
        &self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        local: Result<T, E>,
        map_backend: impl FnOnce(BackendFailure) -> E,
    ) -> Result<T, E> {
        self.runtime
            .finish_text_preparation(stage, local, map_backend)
    }

    /// As [`Self::finish_text_preparation`], with `Ok(None)` representing
    /// cancellation. A peer failure takes precedence over cancellation; a
    /// completed cancellation returns `Ok(None)` to every successful peer.
    pub fn finish_text_preparation_cancellable<T, E>(
        &self,
        stage: eredu_core::run_preparation::TextPreparationStage,
        local: Result<Option<T>, E>,
        map_backend: impl FnOnce(BackendFailure) -> E,
    ) -> Result<Option<T>, E> {
        self.runtime
            .finish_text_preparation_cancellable(stage, local, map_backend)
    }

    /// Session lifetime logical preparation charges, including failed attempts.
    /// Model reset, snapshot restore and experimental branches do not refund them.
    pub fn text_preparation_usage(
        &self,
    ) -> Result<eredu_core::run_preparation::TextPreparationUsage, BackendFailure> {
        self.runtime.text_preparation_usage()
    }
}
