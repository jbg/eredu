//! Attributed bounded capture composed with ordinary prepared-chat generation.

use super::{
    LoadedModel, PreparedChatError, PreparedChatGenerationOutput, PreparedChatGenerationRequest,
    PreparedChatGenerationSettings, PreparedChatInput, TextDecoderError,
};
use crate::runtime::chat::PreparedChat;
use eredu_core::{
    capture::*,
    generation::{
        FinishReason, GenerationCancellationToken, ResolvedGenerationConfig, SemanticEvent,
    },
    intervention::{AdmittedInterventionPlan, InterventionDiscovery, InterventionPlan},
    TextGenerationBackend,
};
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    ops::ControlFlow,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

/// Version of attributed generation events, independently of tensor-capture records.
pub const GENERATION_SCHEMA_VERSION: u32 = 2;

fn is_false(value: &bool) -> bool {
    !*value
}

pub(super) fn new_identity(kind: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "{kind}-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

pub(super) use eredu_runtime::execution_control::TraceBudget;
pub use eredu_runtime::execution_control::TraceLimits;

/// Prepared prompt and capture admission. Dropping this value submits no work.
/// Generation consumes it, preventing accidental reuse with different settings.
/// Both controlled entry points accept this value, including intervened requests:
/// [`LoadedModel::start_controlled_chat`] requires semantic support, while
/// [`LoadedModel::start_controlled_text`] explicitly selects ordinary text decoding.
pub struct PreparedObservedGeneration<B: TextGenerationBackend> {
    pub(super) prompt: Option<B::Prompt>,
    pub(super) chat: PreparedChat,
    pub(super) prompt_token_ids: Vec<u32>,
    pub(super) settings: PreparedChatGenerationSettings,
    pub(super) canvas: Option<std::sync::Arc<eredu_core::refinement::RefinementCanvas>>,
    pub(super) resolved: ResolvedGenerationConfig,
    pub(super) plan: AdmittedCapturePlan,
    pub(super) intervention: Option<AdmittedInterventionPlan>,
    pub(super) session_identity: String,
    pub(super) artifact_identity: Option<String>,
    pub(super) parameter_overlay_id: Option<String>,
    pub(super) trace_limits: TraceLimits,
}

impl<B: TextGenerationBackend> PreparedObservedGeneration<B> {
    /// Installs generic seed/pin controls. The loaded session validates them before execution.
    pub fn with_refinement_canvas(
        mut self,
        canvas: eredu_core::refinement::RefinementCanvas,
    ) -> Self {
        self.canvas = Some(std::sync::Arc::new(canvas));
        self
    }

    /// Prompt alignment in the checkpoint's canonical tokenizer vocabulary.
    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }
    /// Immutable admitted capture selections and request geometry.
    pub fn capture_plan(&self) -> &AdmittedCapturePlan {
        &self.plan
    }
    /// Immutable prospective intervention admission, absent for an ordinary run.
    pub fn intervention_plan(&self) -> Option<&AdmittedInterventionPlan> {
        self.intervention.as_ref()
    }
    /// Checkpoint defaults combined with explicit request settings.
    pub fn generation_config(&self) -> ResolvedGenerationConfig {
        self.resolved
    }
}

/// Versioned host record, serialized and size-checked before synchronous delivery.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservedGenerationRecord {
    /// Current capture/trace wire version.
    pub schema_version: u32,
    /// Unique identity assigned by the facade for this run.
    pub run_id: String,
    /// Content identity when explicitly requested or required by intervention admission.
    /// Session-bound observations and trace-only runs leave this absent.
    pub artifact_identity: Option<String>,
    /// Identity of the loaded facade session that admitted this request.
    pub session_id: String,
    /// Digest of the admitted plan, catalog point semantics, and request shape.
    pub capture_plan_id: String,
    /// Session/source-bound intervention identity, absent for an ordinary run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervention_plan_id: Option<String>,
    /// Active parameter transaction; absent for baseline model parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter_overlay_id: Option<String>,
    /// Ordered generation progress or terminal outcome.
    pub event: ObservedGenerationEvent,
}

/// Trace events use the existing semantic text protocol and committed-token policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObservedGenerationEvent {
    /// Independent child stream, including the delivered inherited semantic
    /// prefix. Later deltas may complete a tool call or Unicode/text boundary
    /// begun in this prefix; no parent journal lookup or replay is required.
    BranchStarted {
        /// Parent snapshot, absolute boundary, budgets and prospective changes.
        lineage: super::GenerationBranchMetadata,
        /// Exact source prompt for interpreting absolute input/capture positions.
        prompt_token_ids: Vec<u32>,
        /// Canonical inherited generated tokens, including special tokens.
        inherited_token_ids: Vec<u32>,
        /// Exact semantic events delivered before the snapshot boundary.
        inherited_semantics: Vec<SemanticEvent>,
    },
    /// An immutable complete in-process snapshot was retained at this boundary.
    SnapshotCreated {
        /// Serializable metadata only; native handles never enter records.
        metadata: super::GenerationSnapshotMetadata,
    },
    /// Reconcile visible history to this prior output prefix, then consume records
    /// in the new monotone epoch. Old token/semantic events are not emitted again.
    Restored {
        /// Reusable snapshot selected for this restore.
        snapshot_id: String,
        /// Prefix to retain in the consumer's output journal.
        output: super::GenerationOutputCheckpoint,
    },
    /// Prospective sampling change at a completed ordinary decision boundary.
    SamplingChanged {
        /// First absolute prediction using this change.
        next_prediction: u64,
        /// Explicit request, including any new seed.
        request: eredu_runtime::execution_control::SamplingOverride,
        /// Retained sampling compatibility before the change.
        before: eredu_runtime::execution_control::SamplingStateFacts,
        /// Sampling compatibility after the change.
        after: eredu_runtime::execution_control::SamplingStateFacts,
    },
    /// Quiescent or terminal state of a controllable run. Ordinary one-shot
    /// generation retains its existing event stream.
    Lifecycle {
        /// Completed-token execution state.
        status: eredu_core::execution_control::GenerationStatus,
        /// Absolute prediction that would execute next.
        next_prediction: u64,
    },
    /// One completed context or refinement operation without authoritative commitment.
    Progress {
        /// Physical phase and tentative block state.
        progress: eredu_core::TextProgress,
        /// Independent completed-work and authoritative-output coordinates.
        position: eredu_core::refinement::GenerationPosition,
        /// Physical chunk geometry and captures, if observation was requested.
        captures: Option<CapturedStep>,
        /// Wall time of this completed prefix and capture settlement.
        step_seconds: f64,
    },
    /// Replaceable tentative output, independently budgeted and never parsed
    /// as authoritative text or a tool call.
    Draft {
        /// Branch/epoch reconciliation of the active tentative block.
        event: eredu_core::refinement::DraftEvent,
        /// Completed physical and committed-output coordinates.
        position: eredu_core::refinement::GenerationPosition,
    },
    /// Captures from an operation that failed before a token was committed.
    CaptureFailure {
        /// Index of the unsuccessful prediction attempt.
        prediction_index: u64,
        /// Causal input span when applicable. Iterative captures carry their
        /// physical invocation geometry in the retained capture records.
        input_range: Option<[u64; 2]>,
        /// Completed host values and structured capture failures, with budgets.
        captures: CapturedStep,
        /// Elapsed attempt time before failure.
        step_seconds: f64,
    },
    /// Prepared token alignment and resolved ordinary sampling settings.
    Started {
        /// Exact prefill token sequence.
        prompt_token_ids: Vec<u32>,
        /// Effective checkpoint/request sampling configuration.
        generation: ResolvedGenerationConfig,
        /// Root sampler seed; observation does not draw from it.
        seed: u64,
    },
    /// One constraint-committed prediction with explicitly declared score semantics.
    Token {
        /// Canonical token identifier, including special/EOS tokens.
        token_id: u32,
        /// Meaning of the scores used for this commitment.
        score_kind: eredu_core::refinement::PredictionScoreKind,
        /// Independent completed-work and authoritative-output coordinates.
        position: eredu_core::refinement::GenerationPosition,
        /// This decision was restricted to one validated canonical token by
        /// execution control. Ordinary callers always emit false.
        #[serde(default, skip_serializing_if = "is_false")]
        forced: bool,
        /// Absolute authoritative commitment index, independent of physical steps.
        prediction_index: u64,
        /// Half-open causal input span. Refinement finalization has no causal
        /// forward span; its score kind and block coordinates describe attribution.
        input_range: Option<[u64; 2]>,
        /// All current observed-generation outputs are committed, single-rank values.
        committed: bool,
        /// Rank owning these complete values. Partitioned capture is rejected.
        rank: u32,
        /// Bounded records, absent when both capture and intervention plans are empty.
        captures: Option<CapturedStep>,
        /// Submission, capture, sampling, token read and exact completion elapsed time.
        step_seconds: f64,
    },
    /// Ordinary incremental decoding, semantic parsing, special-token and EOS behavior.
    Semantic {
        /// Associated prediction, absent before the first token.
        prediction_index: Option<u64>,
        /// Existing facade semantic event, unmodified.
        event: SemanticEvent,
    },
    /// Completed or cancelled at a committed token boundary.
    Completed {
        /// Ordinary termination reason.
        reason: FinishReason,
        /// Number of committed generated tokens.
        generated_tokens: u64,
        /// Wall time including synchronous consumer delivery.
        elapsed_seconds: f64,
    },
    /// Execution/decoding failed. Backend errors may leave the session poisoned;
    /// the backend's existing ownership and validation rules govern reuse.
    Failed {
        /// Failure description; typed error is also returned to the caller.
        message: String,
        /// Wall time including consumer delivery before the error.
        elapsed_seconds: f64,
    },
}

struct Delivery<F> {
    emit: F,
    template: ObservedGenerationRecord,
    budget: TraceBudget,
    cancellation: GenerationCancellationToken,
    failure: Option<CaptureError>,
    closed: bool,
    prediction: u64,
    drafts: super::draft::DraftJournal,
    position: eredu_core::refinement::GenerationPosition,
    iterative: bool,
}

impl<F: FnMut(ObservedGenerationRecord) -> ControlFlow<()>> Delivery<F> {
    fn send(&mut self, event: ObservedGenerationEvent) {
        if self.failure.is_some() || self.closed {
            return;
        }
        let record = ObservedGenerationRecord {
            event,
            ..self.template.clone()
        };
        if let Err(error) = self.budget.charge(&record) {
            self.failure = Some(error);
            self.cancellation.cancel();
            return;
        }
        if (self.emit)(record).is_break() {
            self.closed = true;
            self.cancellation.cancel();
        }
    }
}

impl<B: TextGenerationBackend> LoadedModel<B> {
    /// Returns genuine mutable points and actual loaded-session support.
    pub fn intervention_discovery(&self) -> Result<InterventionDiscovery, CaptureError> {
        B::intervention_discovery(&self.runtime)
    }

    /// Admits capture and intervention together before any native work. Plans apply
    /// prospectively; cached states are never retroactively recomputed. Reset the
    /// session before preparing an independent experiment. Controlled entry points
    /// support pausing, snapshots and prospective branch changes when the backend
    /// and complete continuation storage estimates support them.
    pub fn prepare_intervened_chat(
        &self,
        chat: &PreparedChat,
        settings: PreparedChatGenerationSettings,
        capture: CapturePlan,
        intervention: InterventionPlan,
        trace_limits: TraceLimits,
    ) -> Result<PreparedObservedGeneration<B>, PreparedChatError> {
        let prepared = self.prepare_observed_chat(chat, settings, capture, trace_limits)?;
        self.admit_prepared_interventions(prepared, intervention)
    }

    /// Admits an exact token-ID prefix with prospective interventions. The chat
    /// supplies the output/termination contract only; the prefix is never decoded
    /// or re-tokenized. Start from reset state or a prepared-boundary snapshot.
    pub fn prepare_intervened_token_ids(
        &self,
        chat: &PreparedChat,
        prefix: Vec<u32>,
        settings: PreparedChatGenerationSettings,
        capture: CapturePlan,
        intervention: InterventionPlan,
        trace_limits: TraceLimits,
    ) -> Result<PreparedObservedGeneration<B>, PreparedChatError> {
        let prepared =
            self.prepare_observed_token_ids(chat, prefix, settings, capture, trace_limits)?;
        self.admit_prepared_interventions(prepared, intervention)
    }

    fn admit_prepared_interventions(
        &self,
        mut prepared: PreparedObservedGeneration<B>,
        intervention: InterventionPlan,
    ) -> Result<PreparedObservedGeneration<B>, PreparedChatError> {
        if intervention.schema_version != eredu_core::intervention::INTERVENTION_SCHEMA_VERSION {
            return Err(CaptureError::Invalid("unsupported intervention schema".into()).into());
        }
        if intervention.operations.is_empty() {
            return Ok(prepared);
        }
        let discovery = self.intervention_discovery()?;
        if prepared
            .artifact_identity
            .as_ref()
            .is_some_and(|identity| identity != &discovery.artifact_identity)
        {
            return Err(CaptureError::Invalid(
                "capture/intervention source identities differ".into(),
            )
            .into());
        }
        let admitted =
            intervention.admit(&discovery, prepared.plan.request(), &self.session_identity)?;
        B::validate_text_interventions(&self.runtime, &prepared.plan, &admitted)?;
        prepared.artifact_identity = Some(discovery.artifact_identity);
        prepared.intervention = Some(admitted);
        Ok(prepared)
    }

    /// Returns loaded observation and collector facts without resolving source contents.
    pub fn capture_declarations(
        &self,
    ) -> Result<eredu_core::capture::CaptureDeclarations, CaptureError> {
        B::capture_declarations(&self.runtime)
    }

    /// Resolves content-exact provenance together with loaded observation facts.
    pub fn capture_discovery(&self) -> Result<CaptureDiscovery, CaptureError> {
        B::capture_discovery(&self.runtime)
    }

    pub(super) fn admit_capture(
        &self,
        plan: CapturePlan,
        request: CaptureRequestShape,
    ) -> Result<(AdmittedCapturePlan, Option<String>), PreparedChatError> {
        let bounds = B::text_capture_invocation_bounds(&self.runtime, request)?;
        let provenance = plan.provenance;
        let declarations = if plan.selections.is_empty() {
            eredu_core::capture::CaptureDeclarations {
                catalog: eredu_core::ObservationCatalog {
                    schema_version: eredu_core::DISCOVERY_SCHEMA_VERSION,
                    points: Vec::new(),
                    completeness: eredu_core::DescriptionCompleteness::Complete,
                },
                support: eredu_core::ObservationSupportReport {
                    schema_version: eredu_core::DISCOVERY_SCHEMA_VERSION,
                    points: Vec::new(),
                    capture: Default::default(),
                },
            }
        } else {
            self.capture_declarations()?
        };
        let admitted = match bounds {
            Some(bounds) => plan.admit_invocations(
                &declarations.catalog,
                &declarations.support,
                &declarations.support.capture,
                bounds,
            )?,
            None => plan.admit(
                &declarations.catalog,
                &declarations.support,
                &declarations.support.capture,
                request,
            )?,
        };
        B::validate_text_capture(&self.runtime, &admitted)?;
        let artifact_identity = match provenance {
            eredu_core::capture::CaptureProvenance::Session => None,
            eredu_core::capture::CaptureProvenance::ContentExact => {
                Some(self.capture_discovery()?.artifact_identity)
            }
        };
        Ok((admitted, artifact_identity))
    }

    /// Tokenizes with this model, resolves ordinary settings/EOS semantics, and
    /// admits capture against the loaded session before any execution is submitted.
    pub fn prepare_observed_chat(
        &self,
        chat: &PreparedChat,
        settings: PreparedChatGenerationSettings,
        plan: CapturePlan,
        trace_limits: TraceLimits,
    ) -> Result<PreparedObservedGeneration<B>, PreparedChatError> {
        let prompt_token_ids = self
            .tokenizer
            .encode(chat.rendered_prompt(), false)
            .map_err(TextDecoderError::Tokenizer)?
            .get_ids()
            .to_vec();
        self.prepare_observed_token_ids(chat, prompt_token_ids, settings, plan, trace_limits)
    }

    /// Admits a fixed prefix without decoding, chat-template rendering or tokenization.
    /// IDs must belong to this loaded tokenizer. `chat` retains the output contract
    /// (EOS, stop/semantic policy); its rendered prompt is not submitted. Captures
    /// are planned before replay. No output token is forced by this operation.
    pub fn prepare_observed_token_ids(
        &self,
        chat: &PreparedChat,
        prompt_token_ids: Vec<u32>,
        settings: PreparedChatGenerationSettings,
        plan: CapturePlan,
        trace_limits: TraceLimits,
    ) -> Result<PreparedObservedGeneration<B>, PreparedChatError> {
        if prompt_token_ids.is_empty()
            || prompt_token_ids
                .iter()
                .any(|id| self.tokenizer.id_to_token(*id).is_none())
        {
            return Err(CaptureError::Invalid(
                "token-ID prefix must be nonempty and use this model's tokenizer vocabulary".into(),
            )
            .into());
        }
        let (config, max_tokens) = self.resolve_text_generation_settings(settings)?;
        let request = CaptureRequestShape {
            batch: 1,
            prompt_tokens: prompt_token_ids.len() as u64,
            max_predictions: max_tokens.get() as u64,
        };
        let (admitted, artifact_identity) = self.admit_capture(plan, request)?;
        if trace_limits.per_record_bytes == 0 || trace_limits.total_bytes == 0 {
            return Err(CaptureError::Invalid("trace byte limits must be positive".into()).into());
        }
        Ok(PreparedObservedGeneration {
            prompt: None,
            chat: chat.clone(),
            prompt_token_ids,
            settings,
            canvas: None,
            resolved: config.sampling(),
            plan: admitted,
            intervention: None,
            session_identity: self.session_identity.clone(),
            artifact_identity,
            parameter_overlay_id: B::active_parameter_overlay(&self.runtime).map(str::to_owned),
            trace_limits,
        })
    }

    /// Admits an already prepared text/media prompt for observed or controlled execution.
    /// `original_token_ids` must be the complete ordered decoder IDs retained by
    /// `prompt`, including media placeholders. They provide capture and trace alignment;
    /// embeddings are never used to reconstruct IDs. The backend still validates the
    /// prepared input and its full decoder geometry before executing any chunk.
    pub fn prepare_observed_input(
        &self,
        chat: &PreparedChat,
        prompt: B::Prompt,
        original_token_ids: Vec<u32>,
        settings: PreparedChatGenerationSettings,
        plan: CapturePlan,
        trace_limits: TraceLimits,
    ) -> Result<PreparedObservedGeneration<B>, PreparedChatError> {
        let mut prepared = self.prepare_observed_token_ids(
            chat,
            original_token_ids,
            settings,
            plan,
            trace_limits,
        )?;
        prepared.prompt = Some(prompt);
        Ok(prepared)
    }

    /// Runs the ordinary prepared-chat generator with demand-driven host delivery.
    /// Return `Break(())` from the callback or cancel the shared token to stop at
    /// a token boundary. `Break` also stops further callback delivery; the returned
    /// result still reports the terminal outcome. There is no producer thread or queue. Before return (or
    /// unwinding on consumer panic), the existing generation owner resolves its
    /// outstanding submissions; native failures retain resources through recovery.
    pub fn generate_observed_chat<F>(
        &mut self,
        prepared: PreparedObservedGeneration<B>,
        caller_stop_sequences: &[String],
        cancellation: GenerationCancellationToken,
        on_record: F,
    ) -> Result<PreparedChatGenerationOutput, PreparedChatError>
    where
        F: FnMut(ObservedGenerationRecord) -> ControlFlow<()>,
    {
        self.generate_observed(
            prepared,
            caller_stop_sequences,
            cancellation,
            on_record,
            super::request::PreparedGenerationMode::Semantic,
        )
    }

    /// Generates observed literal text through the ordinary shared driver.
    /// Supports unrecognized templates under the same text/tool admission rules
    /// as `generate_prepared_text` and `start_controlled_text`.
    pub fn generate_observed_text<F>(
        &mut self,
        prepared: PreparedObservedGeneration<B>,
        caller_stop_sequences: &[String],
        cancellation: GenerationCancellationToken,
        on_record: F,
    ) -> Result<PreparedChatGenerationOutput, PreparedChatError>
    where
        F: FnMut(ObservedGenerationRecord) -> ControlFlow<()>,
    {
        self.generate_observed(
            prepared,
            caller_stop_sequences,
            cancellation,
            on_record,
            super::request::PreparedGenerationMode::Text,
        )
    }

    fn generate_observed<F>(
        &mut self,
        prepared: PreparedObservedGeneration<B>,
        caller_stop_sequences: &[String],
        cancellation: GenerationCancellationToken,
        on_record: F,
        mode: super::request::PreparedGenerationMode,
    ) -> Result<PreparedChatGenerationOutput, PreparedChatError>
    where
        F: FnMut(ObservedGenerationRecord) -> ControlFlow<()>,
    {
        let identity = if prepared.session_identity != self.session_identity {
            Err(CaptureError::Invalid(
                "observed request belongs to a different loaded session".into(),
            )
            .into())
        } else {
            Ok(())
        };
        self.runtime.finish_text_preparation(
            eredu_core::run_preparation::TextPreparationStage::Request,
            identity,
            PreparedChatError::Backend,
        )?;
        let started = Instant::now();
        let prompt_length = prepared.prompt_token_ids.len() as u64;
        let template = ObservedGenerationRecord {
            schema_version: GENERATION_SCHEMA_VERSION,
            run_id: new_identity("run"),
            artifact_identity: prepared.artifact_identity,
            parameter_overlay_id: prepared.parameter_overlay_id,
            session_id: self.session_identity.clone(),
            capture_plan_id: prepared.plan.identity().into(),
            intervention_plan_id: prepared
                .intervention
                .as_ref()
                .map(|plan| plan.identity().into()),
            event: ObservedGenerationEvent::Completed {
                reason: FinishReason::Cancelled,
                generated_tokens: 0,
                elapsed_seconds: 0.0,
            },
        };
        let delivery = RefCell::new(Delivery {
            emit: on_record,
            template,
            budget: TraceBudget::new(prepared.trace_limits),
            cancellation: cancellation.clone(),
            failure: None,
            closed: false,
            prediction: 0,
            drafts: super::draft::DraftJournal::new(prepared.settings.draft_events),
            position: Default::default(),
            iterative: B::supports_block_refinement(&self.runtime),
        });
        delivery
            .borrow_mut()
            .send(ObservedGenerationEvent::Started {
                prompt_token_ids: prepared.prompt_token_ids.clone(),
                generation: prepared.resolved,
                seed: prepared.settings.seed,
            });
        let mut on_token = |kind: super::request::GenerationStepKind, captures, step_seconds| {
            let mut delivery = delivery.borrow_mut();
            let index = delivery.prediction;
            let input_range = (!delivery.iterative).then(|| {
                if index == 0 {
                    [0, prompt_length]
                } else {
                    [prompt_length + index - 1, prompt_length + index]
                }
            });
            let (token_id, score_kind, position) = match kind {
                super::request::GenerationStepKind::Token {
                    token,
                    score_kind,
                    position,
                } => (token, score_kind, position),
                super::request::GenerationStepKind::Progress {
                    mut progress,
                    position,
                } => {
                    delivery.position = position;
                    if let Some(event) = delivery.drafts.progress(&mut progress, position, 0) {
                        delivery.send(ObservedGenerationEvent::Draft { event, position });
                    }
                    delivery.send(ObservedGenerationEvent::Progress {
                        progress,
                        position,
                        captures,
                        step_seconds,
                    });
                    return;
                }
                super::request::GenerationStepKind::Failed => {
                    if let Some(captures) = captures {
                        delivery.send(ObservedGenerationEvent::CaptureFailure {
                            prediction_index: index,
                            input_range,
                            captures,
                            step_seconds,
                        });
                    }
                    return;
                }
            };
            delivery.prediction += 1;
            delivery.position = position;
            if let Some(event) = delivery.drafts.clear() {
                delivery.send(ObservedGenerationEvent::Draft { event, position });
            }
            delivery.send(ObservedGenerationEvent::Token {
                token_id,
                score_kind,
                position,
                forced: false,
                prediction_index: index,
                input_range,
                committed: true,
                rank: 0,
                captures,
                step_seconds,
            });
        };
        let on_event = |event| {
            let mut delivery = delivery.borrow_mut();
            let prediction_index = delivery.prediction.checked_sub(1);
            delivery.send(ObservedGenerationEvent::Semantic {
                prediction_index,
                event,
            });
        };
        // The prepared prompt is fed into the existing generation route. No second
        // tokenizer, sampler, EOS loop, or semantic decoder is constructed here.
        let delivery_failure = || delivery.borrow().failure.clone();
        let result = self.generate_prepared(
            PreparedChatGenerationRequest {
                input: match prepared.prompt {
                    Some(prompt) => {
                        PreparedChatInput::prepared_backend_input(&prepared.chat, prompt)
                    }
                    None => PreparedChatInput::token_ids(&prepared.chat, prepared.prompt_token_ids),
                },
                settings: prepared.settings,
                caller_stop_sequences,
                cancellation,
                on_event,
            },
            Some((
                prepared.plan,
                prepared.intervention,
                &mut on_token,
                &delivery_failure,
            )),
            started,
            mode,
            prepared.canvas,
        );
        let mut delivery = delivery.into_inner();
        if let Some(event) = delivery.drafts.clear() {
            let position = delivery.position;
            delivery.send(ObservedGenerationEvent::Draft { event, position });
        }
        match &result {
            Ok(output) => delivery.send(ObservedGenerationEvent::Completed {
                reason: output.finish_reason,
                generated_tokens: output.token_ids.len() as u64,
                elapsed_seconds: started.elapsed().as_secs_f64(),
            }),
            Err(error) => delivery.send(ObservedGenerationEvent::Failed {
                message: error.to_string(),
                elapsed_seconds: started.elapsed().as_secs_f64(),
            }),
        }
        // The shared driver already agreed any failure. Preserve its cause;
        // a best-effort Failed record cannot replace it or start a new phase.
        let output = result?;
        self.runtime.finish_text_preparation_cancellable(
            eredu_core::run_preparation::TextPreparationStage::Delivery,
            match delivery.failure {
                Some(error) => Err(PreparedChatError::from(error)),
                None => Ok((!delivery.cancellation.is_cancelled()).then_some(())),
            },
            PreparedChatError::Backend,
        )?;
        // Terminal publication cannot change the already established stop/EOS
        // outcome. Cancellation is still agreed without turning it into an error.
        Ok(output)
    }
}
