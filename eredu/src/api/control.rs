//! Stateful facade composition over the ordinary committed-token driver.

use super::{
    loaded::map_prepared_chat_setup_error,
    observed::{new_identity, TraceBudget},
    request::{prepared_chat_control_runtime, PreparedChatTokenDecoder},
    ConstraintError, LoadedModel, ObservedGenerationEvent, ObservedGenerationRecord,
    PreparedChatError, PreparedObservedGeneration,
};
use crate::runtime::{chat::constraints::ConstraintController, generation::streaming::*};
use eredu_core::{capture::*, execution_control::*, generation::*, *};
use eredu_runtime::execution_control::{
    GenerationLifecycle, ManagedTextContinuation, SamplingOverride, SamplingOverrideError,
    SamplingStateFacts, TextSamplingControlBackend, TokenChoiceController, TokenChoiceError,
};
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, ops::ControlFlow, time::Instant};

type ControlConstraints = TokenChoiceController<ConstraintController>;
type ControlConstraintError = TokenChoiceError<ConstraintError>;

mod branch;
mod snapshot;
pub use branch::{ControlledGenerationBranch, GenerationBranchMetadata, GenerationBranchOptions};
pub use snapshot::{
    ControlledGenerationSnapshot, GenerationOutputCheckpoint, GenerationSnapshotMetadata,
};

/// Ordered, bounded delivery from a controllable run. Sequence numbers count
/// delivered records monotonically; prediction positions are carried by the event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlledGenerationRecord {
    /// Control record wire version.
    pub schema_version: u32,
    /// Monotone per-run output sequence, starting at zero.
    pub sequence: u64,
    /// Monotone restoration epoch; zero before any restore.
    pub epoch: u64,
    /// Existing source/session/plan attribution and ordinary event semantics.
    pub generation: ObservedGenerationRecord,
}

/// Typed setup, execution, semantic, lifecycle or bounded delivery failure.
#[derive(Debug, thiserror::Error)]
pub enum ControlledGenerationError<E: std::error::Error + Send + Sync + 'static> {
    /// Existing prepared-chat generation failure.
    #[error(transparent)]
    Generation(#[from] PreparedChatError<E>),
    /// Completed-token ownership, execution or completion failure.
    #[error(transparent)]
    Continuation(#[from] TextContinuationError<E, ControlConstraintError>),
    /// Prospective forced token conflicts with canonical vocabulary or grammar.
    #[error(transparent)]
    Choice(#[from] ControlConstraintError),
    /// Invalid prospective sampling request or failure preparing a new native key.
    #[error(transparent)]
    Sampling(#[from] SamplingOverrideError<E>),
    /// Complete native or host snapshot preparation/compatibility failure.
    #[error(transparent)]
    Snapshot(#[from] eredu_runtime::execution_control::TextSnapshotError<E>),
    /// Invalid lifecycle transition or resource arithmetic.
    #[error(transparent)]
    Control(#[from] ExecutionControlError),
    /// Unsupported configuration, admission or transport budget failure.
    #[error(transparent)]
    Capture(#[from] CaptureError),
}

struct Delivery {
    configuration_identity: [u8; 32],
    template: ObservedGenerationRecord,
    budget: TraceBudget,
    control: GenerationControlHandle,
    sequence: u64,
    epoch: u64,
    prediction: u64,
    prompt_length: u64,
    prompt_token_ids: std::sync::Arc<[u32]>,
    started: Instant,
    closed: bool,
    failure: Option<CaptureError>,
    semantic_prefix: Vec<SemanticEvent>,
}
impl Delivery {
    fn send(
        &mut self,
        event: ObservedGenerationEvent,
        emit: &mut impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) {
        if self.closed || self.failure.is_some() {
            return;
        }
        let record = ControlledGenerationRecord {
            schema_version: EXECUTION_CONTROL_SCHEMA_VERSION,
            sequence: self.sequence,
            epoch: self.epoch,
            generation: ObservedGenerationRecord {
                event,
                ..self.template.clone()
            },
        };
        let Some(next) = self.sequence.checked_add(1) else {
            self.failure = Some(CaptureError::Invalid("output sequence overflow".into()));
            self.control.cancel();
            return;
        };
        if let Err(error) = self.budget.charge(&record) {
            self.failure = Some(error);
            self.control.cancel();
            return;
        }
        // Charge and advance before invoking user code, including a caught panic.
        self.sequence = next;
        if let ObservedGenerationEvent::Semantic { event, .. } = &record.generation.event {
            self.semantic_prefix.push(event.clone());
        }
        if emit(record).is_break() {
            self.closed = true;
            self.control.cancel();
        }
    }
    fn token(
        &mut self,
        token: Option<u32>,
        forced: bool,
        captures: Option<CapturedStep>,
        seconds: f64,
        emit: &mut impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) {
        let prediction_index = self.prediction;
        let input_range = if prediction_index == 0 {
            [0, self.prompt_length]
        } else {
            [
                self.prompt_length + prediction_index - 1,
                self.prompt_length + prediction_index,
            ]
        };
        match token {
            Some(token_id) => {
                self.prediction += 1;
                self.send(
                    ObservedGenerationEvent::Token {
                        token_id,
                        forced,
                        prediction_index,
                        input_range,
                        committed: true,
                        rank: 0,
                        captures,
                        step_seconds: seconds,
                    },
                    emit,
                );
            }
            None => {
                if let Some(captures) = captures {
                    self.send(
                        ObservedGenerationEvent::CaptureFailure {
                            prediction_index,
                            input_range,
                            captures,
                            step_seconds: seconds,
                        },
                        emit,
                    );
                }
            }
        }
    }
}

struct Source<'a, 'b, B: TextGenerationBackend, F> {
    driver: &'a mut TextGenerationDriver<'b, B>,
    state: &'a mut ManagedTextContinuation<B, ControlConstraints>,
    delivery: &'a RefCell<(&'a mut Delivery, &'a mut F)>,
}
impl<B: TextGenerationBackend, F: FnMut(ControlledGenerationRecord) -> ControlFlow<()>>
    CommittedTokenSource for Source<'_, '_, B, F>
{
    type Error = TextContinuationError<B::Error, ControlConstraintError>;
    fn next_token(&mut self) -> Result<Option<u32>, Self::Error> {
        let started = Instant::now();
        let result = self.state.advance(self.driver);
        let capture = self.state.take_completed_step(self.driver);
        let token = match result {
            Ok(token) => token.map(|token| token.token_id()),
            Err(error) => {
                if let Ok(capture) = capture {
                    let mut delivery = self.delivery.borrow_mut();
                    let (delivery, emit) = &mut *delivery;
                    delivery.token(None, false, capture, started.elapsed().as_secs_f64(), emit);
                }
                return Err(error);
            }
        };
        let capture = capture?;
        let mut delivery = self.delivery.borrow_mut();
        let (delivery, emit) = &mut *delivery;
        delivery.token(
            token,
            self.state.controller().last_committed_was_forced(),
            capture,
            started.elapsed().as_secs_f64(),
            emit,
        );
        Ok(token)
    }
    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
        self.state
            .controller_mut()
            .is_complete()
            .map_err(|error| ControlledTextGenerationError::Controller(error).into())
    }
}

/// One ordinary generation run with explicit, completed-token advancement.
/// It exclusively borrows its loaded model and retains incremental decoding,
/// constraints, pending input and the backend's existing completion owner.
/// Native objects remain thread-affine; `control_handle` may cross threads.
pub struct ControlledGenerationSession<'a, B: TextGenerationBackend> {
    driver: TextGenerationDriver<'a, B>,
    state: ManagedTextContinuation<B, ControlConstraints>,
    vocabulary: std::sync::Arc<tokenizers::Tokenizer>,
    tokenizer_identity: [u8; 32],
    snapshot_budget: Option<eredu_runtime::execution_control::SnapshotBudget>,
    branch_growth_known: bool,
    cursor: CommittedGenerationCursor,
    pipeline: CommittedTokenPipeline<PreparedChatTokenDecoder>,
    lifecycle: GenerationLifecycle,
    delivery: Delivery,
}

impl<B: TextGenerationBackend> ControlledGenerationSession<'_, B> {
    /// Current lifecycle, including native completion and record delivery.
    pub fn status(&self) -> GenerationStatus {
        self.lifecycle.status()
    }
    /// Next absolute model prediction (zero is prompt prefill).
    pub fn next_prediction(&self) -> u64 {
        self.lifecycle.next_prediction()
    }
    /// Canonical committed history, including EOS/special tokens.
    pub fn token_ids(&self) -> &[u32] {
        self.cursor.token_ids()
    }
    /// Retained ordinary terminal outcome.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.cursor.finish_reason()
    }
    /// Cross-thread pause and permanent cancellation requests.
    pub fn control_handle(&self) -> GenerationControlHandle {
        self.delivery.control.clone()
    }
    /// Cumulative compact-JSON transport bytes, including lifecycle records.
    pub fn emitted_bytes(&self) -> u64 {
        self.delivery.budget.emitted_bytes()
    }
    /// Known logical mutable semantic/decoding/history storage at this boundary.
    /// Excludes native state, constraints, and shared immutable tokenizer data.
    /// Unknown custom parser/decoder costs remain explicit.
    pub fn semantic_snapshot_bytes(&self) -> Option<u64> {
        use crate::runtime::generation::storage::SnapshotStorage;
        self.pipeline
            .snapshot_storage_bytes()?
            .checked_add(self.cursor.snapshot_storage_bytes()?)?
            .checked_add(self.delivery.semantic_prefix.snapshot_bytes()?)?
            .checked_add((self.delivery.prompt_token_ids.len() as u64).checked_mul(4)?)
    }
    /// Full facade support; native model-state primitives alone do not imply
    /// complete semantic/grammar snapshot support.
    pub fn capabilities(&self) -> ExecutionControlCapabilities {
        let mut result = ExecutionControlCapabilities::unsupported(
            "snapshot support requires successful enable_snapshots with complete native/grammar/semantic estimates and explicit limits",
        );
        result.step = ControlSupport::Supported;
        result.pause_resume = ControlSupport::Supported;
        result.force_next_token = ControlSupport::Supported;
        result.sampling_overrides = B::text_sampling_control_support(self.driver.runtime());
        if self.snapshot_budget.is_some() {
            result.snapshot = ControlSupport::Supported;
            result.restore = ControlSupport::Supported;
            result.isolation = Some(SnapshotIsolation::DeepCopy);
        }
        result.fork = if self.branch_growth_known {
            ControlSupport::Supported
        } else {
            ControlSupport::Unsupported { reason: "create a snapshot with known complete native, decoder and parser continuation growth before branching".into() }
        };
        result
            .conditions
            .push("ordinary single-sequence text; serial completed-token delivery".into());
        result.conditions.push("native in-process snapshots; exact exclusive driver and executable; same logical run for restore".into());
        result.conditions.push("unchanged continuations preserve RNG and state; equality requires deterministic native execution on the same device".into());
        result
    }

    fn validate_choice_boundary(&mut self) -> Result<(), ControlledGenerationError<B::Error>> {
        if self.delivery.control.cancellation().is_cancelled() {
            return Err(
                CaptureError::Invalid("generation cancellation was requested".into()).into(),
            );
        }
        if !matches!(
            self.status(),
            GenerationStatus::Prepared | GenerationStatus::Paused
        ) {
            return Err(ExecutionControlError::Transition {
                from: self.status(),
                to: GenerationStatus::Paused,
            }
            .into());
        }
        self.state.boundary(&mut self.driver)?;
        Ok(())
    }

    /// Restricts the next decision to this canonical tokenizer ID after validating
    /// active constraints. The ordinary sampler/RNG, adaptive update, penalties,
    /// commitment and decoder still execute once. No text is re-tokenized.
    /// This changes only the next decision; replacing a past token requires an
    /// earlier snapshot. Clear an existing choice before replacing it.
    pub fn force_next_token(
        &mut self,
        token: u32,
    ) -> Result<(), ControlledGenerationError<B::Error>> {
        self.validate_choice_boundary()?;
        if self.vocabulary.id_to_token(token).is_none() {
            return Err(TokenChoiceError::InvalidToken(token).into());
        }
        self.state.controller_mut().force_next(token)?;
        Ok(())
    }
    /// Removes a pending decision restriction without advancing generation.
    pub fn clear_forced_token(&mut self) -> Result<bool, ControlledGenerationError<B::Error>> {
        self.validate_choice_boundary()?;
        Ok(self.state.controller_mut().clear_forced())
    }
    /// Canonical choice waiting for commitment, if any.
    pub fn pending_forced_token(&self) -> Option<u32> {
        self.state.controller().pending_forced()
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
    fn delivery_result(&mut self) -> Result<(), ControlledGenerationError<B::Error>> {
        if let Some(error) = self.delivery.failure.take() {
            self.lifecycle.fail();
            return Err(error.into());
        }
        Ok(())
    }

    /// Advances at most one committed prediction and synchronously delivers its
    /// captures and semantic events. A paused handle remains sticky for `run`;
    /// this explicit single-step request may still advance once. Callback Break
    /// permanently cancels and closes delivery, preserving ordinary semantics.
    pub fn step(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus, ControlledGenerationError<B::Error>> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.step_inner(emit))) {
            Ok(result) => result,
            Err(payload) => {
                self.lifecycle.fail();
                std::panic::resume_unwind(payload)
            }
        }
    }

    fn step_inner(
        &mut self,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus, ControlledGenerationError<B::Error>> {
        let cancelled = self.delivery.control.cancellation().is_cancelled();
        // Validate before touching the cursor, including terminal/error states.
        if cancelled {
            self.lifecycle.cancel()?;
        } else {
            self.lifecycle.begin_prediction()?;
        }
        let cancellation = self.delivery.control.cancellation().clone();
        let committed_before = self.cursor.token_ids().len();
        let result = {
            let delivery = RefCell::new((&mut self.delivery, &mut emit));
            let mut source = Source {
                driver: &mut self.driver,
                state: &mut self.state,
                delivery: &delivery,
            };
            self.cursor.step(
                &mut source,
                &mut self.pipeline,
                &cancellation,
                &mut |event| {
                    let mut delivery = delivery.borrow_mut();
                    let (delivery, emit) = &mut *delivery;
                    delivery.send(
                        ObservedGenerationEvent::Semantic {
                            prediction_index: delivery.prediction.checked_sub(1),
                            event,
                        },
                        emit,
                    );
                },
            )
        };
        if let Err(error) = result {
            self.lifecycle.fail();
            let error = match error {
                CommittedGenerationError::Source(error) => {
                    ControlledGenerationError::Continuation(error)
                }
                CommittedGenerationError::Pipeline(CommittedTokenPipelineError::Decoder(error)) => {
                    PreparedChatError::Tokenizer(error).into()
                }
                CommittedGenerationError::Pipeline(CommittedTokenPipelineError::Semantic(
                    error,
                )) => PreparedChatError::Semantic(error).into(),
                CommittedGenerationError::Lifecycle(error) => {
                    PreparedChatError::Generation(error).into()
                }
                CommittedGenerationError::MissingTerminalToken => {
                    PreparedChatError::MissingTerminalToken.into()
                }
            };
            self.delivery.send(
                ObservedGenerationEvent::Failed {
                    message: error.to_string(),
                    elapsed_seconds: self.delivery.started.elapsed().as_secs_f64(),
                },
                &mut emit,
            );
            return Err(error);
        }
        self.delivery_result()?;
        if !cancelled {
            if self.cursor.token_ids().len() == committed_before
                && self.cursor.finish_reason() == Some(FinishReason::Cancelled)
            {
                self.state.boundary(&mut self.driver)?;
                self.lifecycle.cancel_without_prediction()?;
            } else {
                self.lifecycle
                    .complete_prediction(self.cursor.finish_reason())?;
            }
        }
        if let Some(reason) = self.cursor.finish_reason() {
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

    /// Pauses a quiescent session without consuming input, randomness or buffered
    /// semantic text. An in-flight worker observes remote requests through `run`.
    pub fn pause(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError<B::Error>> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.pause_inner(emit))) {
            Ok(result) => result,
            Err(payload) => {
                self.lifecycle.fail();
                std::panic::resume_unwind(payload)
            }
        }
    }

    fn pause_inner(
        &mut self,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError<B::Error>> {
        self.lifecycle.pause()?;
        self.delivery.control.request_pause();
        self.lifecycle_record(&mut emit);
        self.delivery_result()
    }

    /// Permanently cancels at the current completed boundary. Buffered decoder
    /// text is discarded under the ordinary cancellation policy, never flushed.
    pub fn cancel(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus, ControlledGenerationError<B::Error>> {
        // Validate before changing the persistent cancellation handle.
        self.lifecycle.checkpoint()?;
        if self.status() == GenerationStatus::Completed {
            return Err(ExecutionControlError::Transition {
                from: GenerationStatus::Completed,
                to: GenerationStatus::Cancelled,
            }
            .into());
        }
        self.delivery.control.cancel();
        self.step(emit)
    }
    /// Advances until the current pause request or a terminal outcome. Each token
    /// is delivered synchronously; no producer queue accumulates behind a consumer.
    pub fn run(
        &mut self,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus, ControlledGenerationError<B::Error>> {
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
    /// Explicitly acknowledges pause and continues the same ordinary run.
    pub fn resume(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus, ControlledGenerationError<B::Error>> {
        self.lifecycle.pause()?;
        self.delivery.control.acknowledge_resume();
        self.run(emit)
    }
}

impl<B: TextSamplingControlBackend> ControlledGenerationSession<'_, B> {
    /// Current temperature and RNG/adaptive compatibility facts at a quiescent
    /// boundary. No sampler or model work occurs.
    pub fn sampling_state(
        &mut self,
    ) -> Result<SamplingStateFacts, ControlledGenerationError<B::Error>> {
        self.validate_choice_boundary()?;
        Ok(B::sampling_control_facts(
            self.state.boundary(&mut self.driver)?.parts().1,
        ))
    }

    /// Applies a prospective temperature change or explicit reseed. Inherited RNG,
    /// penalties, committed history and adaptive counters remain unchanged unless
    /// the request explicitly reseeds randomness. Emits bounded provenance at the
    /// next absolute prediction; it cannot change earlier cache or semantic state.
    pub fn override_sampling(
        &mut self,
        request: SamplingOverride,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError<B::Error>> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.override_sampling_inner(request, emit)
        })) {
            Ok(Err(
                error @ ControlledGenerationError::Sampling(SamplingOverrideError::Backend(_)),
            )) => {
                // A native preparation error alone is not completion evidence.
                // Retain the backend recovery owner and fence further advancement.
                self.lifecycle.fail();
                Err(error)
            }
            Ok(result) => result,
            Err(payload) => {
                self.lifecycle.fail();
                std::panic::resume_unwind(payload)
            }
        }
    }

    fn override_sampling_inner(
        &mut self,
        request: SamplingOverride,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError<B::Error>> {
        self.validate_choice_boundary()?;
        let before = B::sampling_control_facts(self.state.boundary(&mut self.driver)?.parts().1);
        let after = eredu_runtime::execution_control::apply_sampling_override(
            &mut self.state.boundary(&mut self.driver)?,
            request,
        )?;
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
    }
}

impl<B: TextGenerationBackend> LoadedModel<B> {
    /// Starts a controllable ordinary/observed/intervened prepared request. No
    /// model prediction or RNG draw occurs here. Initial attribution is charged
    /// and delivered before returning the session. Unsupported execution fails
    /// before native prompt or sampler construction.
    pub fn start_controlled_chat<'a>(
        &'a mut self,
        prepared: PreparedObservedGeneration,
        caller_stop_sequences: &[String],
        control: GenerationControlHandle,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<ControlledGenerationSession<'a, B>, ControlledGenerationError<B::Error>> {
        if prepared.session_identity != self.session_identity {
            return Err(CaptureError::Invalid(
                "prepared request belongs to another loaded session".into(),
            )
            .into());
        }
        if let ControlSupport::Unsupported { reason } =
            B::text_execution_control_support(&self.runtime)
        {
            return Err(CaptureError::Unsupported(reason).into());
        }
        let (config, max_tokens) = self
            .resolve_text_generation_settings(prepared.settings)
            .map_err(PreparedChatError::Generation)?;
        let semantic = prepared_chat_control_runtime(
            &prepared.chat,
            caller_stop_sequences,
            self.token_validity.clone(),
        )
        .map_err(map_prepared_chat_setup_error)?;
        // Versioned initial policy identity; saved native sampler state retains
        // any later prospective changes. Compatibility also requires the opaque
        // exact driver/run identities, never this digest alone.
        use sha2::Digest;
        let generation_plan = prepared
            .chat
            .generation_runtime_plan()
            .expect("validated prepared chat runtime");
        let choice = match generation_plan.tool_choice() {
            crate::runtime::chat::ToolChoice::None => "none",
            crate::runtime::chat::ToolChoice::Auto => "auto",
            crate::runtime::chat::ToolChoice::Required => "required",
        };
        let configuration_identity: [u8; 32] = sha2::Sha256::digest(
            serde_json::to_vec(&(
                EXECUTION_CONTROL_SCHEMA_VERSION,
                self.tokenizer_fingerprint,
                prepared.resolved,
                prepared.settings.seed,
                prepared.chat.format_profile_identity(),
                generation_plan.generation_constraint().fingerprint,
                choice,
                prepared.chat.eos_token_ids(),
                prepared.chat.profile_stop_sequences(),
                caller_stop_sequences,
            ))
            .map_err(|error| CaptureError::Invalid(error.to_string()))?,
        )
        .into();
        let decoder = PreparedChatTokenDecoder {
            decoder: self.text_decoder(true),
        };
        let vocabulary = std::sync::Arc::clone(&decoder.decoder.tokenizer);
        let domain = eredu_runtime::TokenDomain::new(
            self.token_validity
                .allowed_mask()
                .expect("facade always supplies a closed token domain")
                .len(),
        );
        let pipeline = CommittedTokenPipeline::new(
            RawTokenDecoder::with_structural_tokens(decoder, semantic.structural_tokens),
            semantic.parser,
        );
        let cursor = CommittedGenerationCursor::new(prepared.chat.eos_token_ids(), max_tokens);
        let mut delivery = Delivery {
            configuration_identity,
            template: ObservedGenerationRecord {
                schema_version: CAPTURE_SCHEMA_VERSION,
                run_id: new_identity("run"),
                artifact_identity: prepared.artifact_identity,
                session_id: self.session_identity.clone(),
                capture_plan_id: prepared.plan.identity().into(),
                intervention_plan_id: prepared.intervention.as_ref().map(|p| p.identity().into()),
                event: ObservedGenerationEvent::Lifecycle {
                    status: GenerationStatus::Prepared,
                    next_prediction: 0,
                },
            },
            budget: TraceBudget::new(prepared.trace_limits),
            control,
            sequence: 0,
            epoch: 0,
            prediction: 0,
            prompt_length: prepared.prompt_token_ids.len() as u64,
            prompt_token_ids: prepared.prompt_token_ids.clone().into(),
            started: Instant::now(),
            closed: false,
            failure: None,
            semantic_prefix: Vec::new(),
        };
        let started = ObservedGenerationEvent::Started {
            prompt_token_ids: prepared.prompt_token_ids.clone(),
            generation: prepared.resolved,
            seed: prepared.settings.seed,
        };
        let prompt = B::prepare_text_prompt(self.runtime.backend(), prepared.prompt_token_ids)
            .map_err(PreparedChatError::Backend)?;
        let mut driver = TextGenerationDriver::new(&mut self.runtime);
        let mut state = driver
            .start(
                prompt,
                config,
                TokenChoiceController::new(semantic.controller, domain),
            )
            .map_err(TextContinuationError::Generation)?;
        match prepared.intervention {
            Some(plan) => driver.enable_interventions(&mut state, prepared.plan, plan)?,
            None => driver.enable_capture(&mut state, prepared.plan)?,
        }
        delivery.send(started, &mut emit);
        if let Some(error) = delivery.failure.take() {
            return Err(error.into());
        }
        Ok(ControlledGenerationSession {
            driver,
            state: ManagedTextContinuation::root(state),
            vocabulary,
            tokenizer_identity: self.tokenizer_fingerprint,
            snapshot_budget: None,
            branch_growth_known: false,
            cursor,
            pipeline,
            lifecycle: GenerationLifecycle::default(),
            delivery,
        })
    }
}
