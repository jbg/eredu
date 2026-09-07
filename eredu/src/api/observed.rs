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

/// Limits on the entire JSON trace, including prompt, decoded text, provenance,
/// capture data, and terminal records. Independent of capture storage/transfer bounds.
/// Measures compact JSON, excluding application framing or additional encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceLimits {
    /// Maximum UTF-8 JSON bytes in one record.
    pub per_record_bytes: u64,
    /// Maximum sum of UTF-8 JSON record lengths for the run.
    pub total_bytes: u64,
}

/// Prepared prompt and capture admission. Dropping this value submits no work.
/// Generation consumes it, preventing accidental reuse with different settings.
pub struct PreparedObservedGeneration {
    chat: PreparedChat,
    prompt_token_ids: Vec<u32>,
    settings: PreparedChatGenerationSettings,
    resolved: ResolvedGenerationConfig,
    plan: AdmittedCapturePlan,
    intervention: Option<AdmittedInterventionPlan>,
    session_identity: String,
    artifact_identity: Option<String>,
    trace_limits: TraceLimits,
}

impl PreparedObservedGeneration {
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
    /// Content identity of prepared sources, when the backend exposes it.
    pub artifact_identity: Option<String>,
    /// Identity of the loaded facade session that admitted this request.
    pub session_id: String,
    /// Digest of the admitted plan, catalog point semantics, and request shape.
    pub capture_plan_id: String,
    /// Session/source-bound intervention identity, absent for an ordinary run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervention_plan_id: Option<String>,
    /// Ordered generation progress or terminal outcome.
    pub event: ObservedGenerationEvent,
}

/// Trace events use the existing semantic text protocol and committed-token policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObservedGenerationEvent {
    /// Captures from an operation that failed before a token was committed.
    CaptureFailure {
        /// Index of the unsuccessful prediction attempt.
        prediction_index: u64,
        /// Input positions used by the failed forward operation.
        input_range: [u64; 2],
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
    /// One constraint-committed prediction and captures from its causal forward pass.
    Token {
        /// Canonical token identifier, including special/EOS tokens.
        token_id: u32,
        /// Zero is predicted by prefill; later values are decode predictions.
        prediction_index: u64,
        /// Half-open input positions covered by this forward pass.
        input_range: [u64; 2],
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
    limits: TraceLimits,
    emitted_bytes: u64,
    cancellation: GenerationCancellationToken,
    failure: Option<CaptureError>,
    closed: bool,
    prediction: u64,
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
        let remaining = self.limits.total_bytes.saturating_sub(self.emitted_bytes);
        let mut sink = TraceCounter {
            used: 0,
            limit: remaining.min(self.limits.per_record_bytes),
        };
        if serde_json::to_writer(&mut sink, &record).is_err() {
            self.failure = Some(CaptureError::Limit {
                budget: CaptureBudget::Encoded,
                cumulative: remaining < self.limits.per_record_bytes,
            });
            self.cancellation.cancel();
            return;
        }
        self.emitted_bytes += sink.used;
        if (self.emit)(record).is_break() {
            self.closed = true;
            self.cancellation.cancel();
        }
    }
}

struct TraceCounter {
    used: u64,
    limit: u64,
}
impl std::io::Write for TraceCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.used = self
            .used
            .checked_add(bytes.len() as u64)
            .filter(|n| *n <= self.limit)
            .ok_or_else(|| std::io::Error::other("trace byte limit"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<B: TextGenerationBackend> LoadedModel<B> {
    /// Returns genuine mutable points and actual loaded-session support.
    pub fn intervention_discovery(&self) -> Result<InterventionDiscovery, CaptureError> {
        B::intervention_discovery(&self.runtime)
    }

    /// Admits capture and intervention together before any native work. Plans apply
    /// prospectively; cached states are never retroactively recomputed. Reset the
    /// session before preparing an independent experiment. There is no hot plan
    /// replacement or resumable snapshot API.
    pub fn prepare_intervened_chat(
        &self,
        chat: &PreparedChat,
        settings: PreparedChatGenerationSettings,
        capture: CapturePlan,
        intervention: InterventionPlan,
        trace_limits: TraceLimits,
    ) -> Result<PreparedObservedGeneration, PreparedChatError<B::Error>> {
        let mut prepared = self.prepare_observed_chat(chat, settings, capture, trace_limits)?;
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

    /// Returns the exact loaded session's retained catalog and capture support.
    pub fn capture_discovery(&self) -> Result<CaptureDiscovery, CaptureError> {
        B::capture_discovery(&self.runtime)
    }

    /// Tokenizes with this model, resolves ordinary settings/EOS semantics, and
    /// admits capture against the loaded session before any execution is submitted.
    pub fn prepare_observed_chat(
        &self,
        chat: &PreparedChat,
        settings: PreparedChatGenerationSettings,
        plan: CapturePlan,
        trace_limits: TraceLimits,
    ) -> Result<PreparedObservedGeneration, PreparedChatError<B::Error>> {
        let (config, max_tokens) = self.resolve_text_generation_settings(settings)?;
        let prompt_token_ids = self
            .tokenizer
            .encode(chat.rendered_prompt(), false)
            .map_err(TextDecoderError::Tokenizer)?
            .get_ids()
            .to_vec();
        let request = CaptureRequestShape {
            batch: 1,
            prompt_tokens: prompt_token_ids.len() as u64,
            max_predictions: max_tokens.get() as u64,
        };
        let (admitted, artifact_identity) = if plan.selections.is_empty() {
            let catalog = eredu_core::ObservationCatalog {
                schema_version: eredu_core::DISCOVERY_SCHEMA_VERSION,
                points: Vec::new(),
                completeness: eredu_core::DescriptionCompleteness::Complete,
            };
            let support = eredu_core::ObservationSupportReport {
                schema_version: eredu_core::DISCOVERY_SCHEMA_VERSION,
                points: Vec::new(),
                capture: Default::default(),
            };
            let admitted = plan.admit(&catalog, &support, &support.capture, request)?;
            (
                admitted,
                B::capture_discovery(&self.runtime)
                    .ok()
                    .map(|d| d.artifact_identity),
            )
        } else {
            let discovery = self.capture_discovery()?;
            (
                plan.admit(
                    &discovery.catalog,
                    &discovery.support,
                    &discovery.support.capture,
                    request,
                )?,
                Some(discovery.artifact_identity),
            )
        };
        B::validate_text_capture(&self.runtime, &admitted)?;
        if trace_limits.per_record_bytes == 0 || trace_limits.total_bytes == 0 {
            return Err(CaptureError::Invalid("trace byte limits must be positive".into()).into());
        }
        Ok(PreparedObservedGeneration {
            chat: chat.clone(),
            prompt_token_ids,
            settings,
            resolved: config.sampling(),
            plan: admitted,
            intervention: None,
            session_identity: self.session_identity.clone(),
            artifact_identity,
            trace_limits,
        })
    }

    /// Runs the ordinary prepared-chat generator with demand-driven host delivery.
    /// Return `Break(())` from the callback or cancel the shared token to stop at
    /// a token boundary. `Break` also stops further callback delivery; the returned
    /// result still reports the terminal outcome. There is no producer thread or queue. Before return (or
    /// unwinding on consumer panic), the existing generation owner resolves its
    /// outstanding submissions; native failures retain resources through recovery.
    pub fn generate_observed_chat<F>(
        &mut self,
        prepared: PreparedObservedGeneration,
        caller_stop_sequences: &[String],
        cancellation: GenerationCancellationToken,
        on_record: F,
    ) -> Result<PreparedChatGenerationOutput, PreparedChatError<B::Error>>
    where
        F: FnMut(ObservedGenerationRecord) -> ControlFlow<()>,
    {
        if prepared.session_identity != self.session_identity {
            return Err(CaptureError::Invalid(
                "observed request belongs to a different loaded session".into(),
            )
            .into());
        }
        let started = Instant::now();
        let prompt_length = prepared.prompt_token_ids.len() as u64;
        let template = ObservedGenerationRecord {
            schema_version: CAPTURE_SCHEMA_VERSION,
            run_id: new_identity("run"),
            artifact_identity: prepared.artifact_identity,
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
            limits: prepared.trace_limits,
            emitted_bytes: 0,
            cancellation: cancellation.clone(),
            failure: None,
            closed: false,
            prediction: 0,
        });
        delivery
            .borrow_mut()
            .send(ObservedGenerationEvent::Started {
                prompt_token_ids: prepared.prompt_token_ids.clone(),
                generation: prepared.resolved,
                seed: prepared.settings.seed,
            });
        let mut on_token = |token_id: Option<u32>, captures, step_seconds| {
            let mut delivery = delivery.borrow_mut();
            let index = delivery.prediction;
            let input_range = if index == 0 {
                [0, prompt_length]
            } else {
                [prompt_length + index - 1, prompt_length + index]
            };
            let Some(token_id) = token_id else {
                if let Some(captures) = captures {
                    delivery.send(ObservedGenerationEvent::CaptureFailure {
                        prediction_index: index,
                        input_range,
                        captures,
                        step_seconds,
                    });
                }
                return;
            };
            delivery.prediction += 1;
            delivery.send(ObservedGenerationEvent::Token {
                token_id,
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
        let result = if cancellation.is_cancelled() {
            on_event(SemanticEvent::Finished {
                reason: FinishReason::Cancelled,
            });
            Ok(PreparedChatGenerationOutput {
                token_ids: Vec::new(),
                finish_reason: FinishReason::Cancelled,
            })
        } else {
            match B::prepare_text_prompt(self.runtime.backend(), prepared.prompt_token_ids) {
                Err(error) => Err(PreparedChatError::Backend(error)),
                Ok(prompt) => self.generate_prepared_chat_captured(
                    PreparedChatGenerationRequest {
                        input: PreparedChatInput::prepared_backend_input(&prepared.chat, prompt),
                        settings: prepared.settings,
                        caller_stop_sequences,
                        cancellation,
                        on_event,
                    },
                    Some((prepared.plan, prepared.intervention, &mut on_token)),
                ),
            }
        };
        let mut delivery = delivery.into_inner();
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
        if let Some(error) = delivery.failure {
            return Err(error.into());
        }
        result
    }
}
