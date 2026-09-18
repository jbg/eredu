//! Shared attributed progress events and explicit loaded discovery.

use super::LoadedModel;
use eredu_core::{capture::*, generation::{FinishReason, ResolvedGenerationConfig, SemanticEvent},
    intervention::InterventionDiscovery, TextGenerationBackend};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

mod delivery;

fn is_false(value: &bool) -> bool {
    !*value
}

/// Scalar identity selection, allowing destination text to wait for admission.
pub(super) struct PreparedIdentity<'a> {
    kind: &'a str,
    process: u32,
    nanos: u128,
    sequence: u64,
}
impl<'a> PreparedIdentity<'a> {
    pub(super) fn new(kind: &'a str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self {
            kind,
            process: std::process::id(),
            nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
            sequence: NEXT.fetch_add(1, Ordering::Relaxed),
        }
    }
    pub(super) fn bytes(&self) -> Option<u64> {
        fn digits(n: u128) -> u64 {
            n.checked_ilog10().map_or(1, |n| u64::from(n) + 1)
        }
        u64::try_from(self.kind.len()).ok()?
            .checked_add(3)?
            .checked_add(digits(u128::from(self.process)))?
            .checked_add(digits(self.nanos))?
            .checked_add(digits(u128::from(self.sequence)))
    }
    pub(super) fn write_into(&self, destination: &mut impl std::fmt::Write) -> std::fmt::Result {
        write!(destination, "{}-{}-{}-{}", self.kind, self.process, self.nanos, self.sequence)
    }
    pub(super) fn render(self) -> String {
        let mut destination = String::with_capacity(
            usize::try_from(self.bytes().expect("finite identity")).expect("host identity length"));
        self.write_into(&mut destination).expect("String formatting is infallible");
        destination
    }
}

pub(super) fn new_identity(kind: &str) -> String {
    PreparedIdentity::new(kind).render()
}

pub(super) use eredu_runtime::execution_control::TraceBudget;
pub use eredu_runtime::execution_control::TraceLimits;

/// Trace events use the existing semantic text protocol and committed-token policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObservedGenerationEvent {
    /// Reconcile visible history to this prior output prefix, then consume records
    /// in the new monotone epoch. Old token/semantic events are not emitted again.
    Restored {
        /// Reusable snapshot selected for this restore.
        snapshot_id: String,
        /// Prefix to retain in the consumer's output journal.
        output: super::GenerationOutputCheckpointData,
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
    /// Captures from an operation that failed before a token was committed.
    CaptureFailure {
        /// Index of the unsuccessful prediction attempt.
        prediction_index: u64,
        /// Input positions used by the failed forward operation.
        input_range: [u64; 2],
        /// Completed host values and structured capture failures, with budgets.
        captures: SharedCapturedStep,
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
        /// This decision was restricted to one validated canonical token by
        /// execution control. Ordinary callers always emit false.
        #[serde(default, skip_serializing_if = "is_false")]
        forced: bool,
        /// Zero is predicted by prefill; later values are decode predictions.
        prediction_index: u64,
        /// Half-open input positions covered by this forward pass.
        input_range: [u64; 2],
        /// All current observed-generation outputs are committed, single-rank values.
        committed: bool,
        /// Rank owning these complete values. Partitioned capture is rejected.
        rank: u32,
        /// Bounded records, absent when both capture and intervention plans are empty.
        captures: Option<SharedCapturedStep>,
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

impl<B: TextGenerationBackend> LoadedModel<B> {
    /// Returns genuine mutable points and actual loaded-session support.
    pub fn intervention_discovery(&self) -> Result<InterventionDiscovery, CaptureError> {
        B::intervention_discovery(&self.runtime)
    }

    /// Returns the exact loaded session's retained catalog and capture support.
    pub fn capture_discovery(&self) -> Result<CaptureDiscovery, CaptureError> {
        B::capture_discovery(&self.runtime)
    }

    pub(super) fn admit_capture(
        &self,
        plan: CapturePlan,
        request: CaptureRequestShape,
    ) -> Result<AdmittedCapturePlan, CaptureError> {
        let admitted = if plan.selections.is_empty() {
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
            admitted
        } else {
            let discovery = self.capture_discovery()?;
            plan.admit(
                    &discovery.catalog,
                    &discovery.support,
                    &discovery.support.capture,
                    request,
                )?
        };
        B::validate_text_capture(&self.runtime, &admitted)?;
        Ok(admitted)
    }

}
