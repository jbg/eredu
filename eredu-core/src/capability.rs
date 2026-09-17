//! Portable model capabilities, runtime-state accounting, and admission policy.

use crate::{
    AttentionPolicy, LayerSchedule, ObservationKind, Observed,
    cache::{
        LayerCachePolicy, StateTensorDimension, StateTensorDtype, StateTensorPolicy,
        StateTensorPresence, StateTensorRole,
    },
};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU8;

mod admission;
pub use admission::{
    AdmissionObservation, AdmissionPolicyDecision, AdmissionPolicyError, AdmissionRequirements,
    AdmissionStateRequirements, BorrowedAdmissionRejection, BorrowedAdmissionResult,
    ExecutionWorkspaceRequirements, SelectedStateRequirements, apply_admission_requirements,
    check_admission_context_borrowed,
};

mod state_facts;
pub use state_facts::{
    RuntimeStateFacts, StateWindowDestinationError, StateWindowPlan, estimate_runtime_state_facts,
};

mod workspace;
pub use workspace::{ExecutionWorkspaceEstimate, WorkspaceBound};

/// Model inputs accepted by a prepared model.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputModalities {
    /// Ordinary tokenizer IDs.
    pub text: bool,
    /// Prepared image inputs.
    pub image: bool,
    /// Prepared audio inputs.
    pub audio: bool,
    /// Prepared video inputs.
    pub video: bool,
}

impl InputModalities {
    /// Text-only input support.
    pub const TEXT: Self = Self {
        text: true,
        image: false,
        audio: false,
        video: false,
    };
}

/// Persistent decoder-state strategy used by a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum CacheStateStrategy {
    /// Ordinary full-context K/V attention.
    FullKv,
    /// Every attention cache is bounded by a sliding window.
    SlidingKv {
        /// Maximum retained positions per attention layer.
        window: u64,
    },
    /// Sliding-window key-only attention, optionally with pooled history.
    SlidingKey {
        /// Maximum retained key positions per attention layer.
        window: u64,
        /// Total layers retaining local keys.
        layers: u64,
        /// Layers that additionally retain append-only pooling state.
        pooling_layers: u64,
    },
    /// Full-context and sliding-window attention layers.
    MixedKv {
        /// Number of full-context layers.
        full_layers: u64,
        /// Bounded layer counts grouped by retained window.
        sliding: Vec<SlidingWindowLayerCount>,
    },
    /// Full-context KV backing with layers that reuse earlier K/V state.
    SharedFullKv {
        /// Layers that allocate their own K/V state.
        cached_layers: u64,
        /// Layers that reuse K/V produced by an earlier layer.
        shared_layers: u64,
        /// Total full-attention layer count.
        full_attention_layers: u64,
        /// Sliding-mask layers, which do not bound KV allocation.
        sliding_attention: Vec<SlidingWindowLayerCount>,
    },
    /// Multi-head latent attention compressed state.
    CompressedMla {
        /// Compressed latent width per layer and position.
        latent_width: u64,
        /// Shared rotary-key width per layer and position.
        rotary_width: u64,
    },
    /// Attention combined with bounded convolution or recurrent state.
    HybridRecurrent {
        /// Full-context attention layer count.
        full_attention_layers: u64,
        /// Bounded attention layers grouped by exact window.
        sliding_attention: Vec<SlidingWindowLayerCount>,
        /// Recurrent/linear-attention layer count.
        recurrent_layers: u64,
    },
    /// Multimodal preparation feeding positions into a decoder strategy.
    Multimodal {
        /// Underlying decoder state.
        decoder: Box<CacheStateStrategy>,
        /// Whether media embeddings consume persistent decoder positions.
        media_consumes_decoder_positions: bool,
    },
}

/// Sliding-attention layer count sharing one retained window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlidingWindowLayerCount {
    /// Exact positive retained positions, including the current token.
    pub window: u64,
    /// Number of layers using this window.
    pub layers: u64,
}

/// Coverage of a runtime-state estimate.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimationCompleteness {
    /// Persistent request state and execution workspace are modeled exactly.
    Complete,
    /// The estimate is a complete safe upper bound.
    Conservative,
    /// Persistent state is covered but execution transients are not.
    PersistentStateOnly,
}

/// Capabilities derived from validated model configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Parsed implementation or nested text-model type.
    pub effective_model_type: String,
    /// Original trained context before a supported extension.
    pub native_max_context: Observed<u64>,
    /// Maximum positions accepted by the prepared model.
    pub effective_max_context: Observed<u64>,
    /// Persistent cache or recurrent-state model.
    pub state_strategy: CacheStateStrategy,
    /// Accepted input modalities.
    pub modalities: InputModalities,
    /// Runtime-state estimator coverage.
    pub estimation: EstimationCompleteness,
}

/// Accounting for a tokenized or backend-prepared input.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputTokenCount {
    /// Ordinary tokenizer IDs present in the input.
    pub text_tokens: u64,
    /// Positions inserted by prepared media.
    pub media_positions: u64,
    /// Total decoder/model positions consumed by prefill.
    pub model_positions: u64,
    /// Semantics of the model-position count.
    pub kind: ObservationKind,
    media_execution_workspace_bytes: u64,
    media_execution_workspace_kind: ObservationKind,
}

impl InputTokenCount {
    /// Creates an exact count for tokenized text.
    pub const fn text(tokens: u64) -> Self {
        Self {
            text_tokens: tokens,
            media_positions: 0,
            model_positions: tokens,
            kind: ObservationKind::Exact,
            media_execution_workspace_bytes: 0,
            media_execution_workspace_kind: ObservationKind::Exact,
        }
    }

    /// Creates an exact position count for backend-prepared input.
    pub const fn prepared(
        text_tokens: u64,
        media_positions: u64,
        model_positions: u64,
        media_execution_workspace_bytes: u64,
        media_execution_workspace_kind: ObservationKind,
    ) -> Self {
        Self {
            text_tokens,
            media_positions,
            model_positions,
            kind: ObservationKind::Exact,
            media_execution_workspace_bytes,
            media_execution_workspace_kind,
        }
    }

    /// Conservative media-tower workspace attributed to this input.
    pub const fn media_execution_workspace_bytes(&self) -> u64 {
        self.media_execution_workspace_bytes
    }

    /// Measurement semantics of the media workspace.
    pub const fn media_execution_workspace_kind(&self) -> ObservationKind {
        self.media_execution_workspace_kind
    }
}

/// Floating-dtype and request assumptions used by state estimation.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct StateMemoryAssumptions {
    /// Bytes per architecture-declared generic floating-state scalar.
    ///
    /// Fixed-dtype tensors use their own widths from the exact state policy.
    pub floating_state_dtype_bytes: NonZeroU8,
    /// Logical request batch size.
    pub batch_size: u64,
    /// Total requested positions, including output allowance.
    pub requested_positions: u64,
    /// Distinct sliding-window bounds in ascending order.
    pub sliding_window_bounds: Vec<u64>,
    /// Backing-array growth granularity for unbounded caches.
    pub allocation_granularity: u64,
}

/// Complete selected decoder backing bound for one exact inference request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedStateBacking {
    /// Exact geometry used to project and advance the selected state.
    pub geometry: crate::InferenceGeometry,
    /// Complete retained allocation capacity, or its missing bound.
    pub bound: WorkspaceBound,
}

impl SelectedStateBacking {
    /// Complete capacity when every selected backing allocation is priced.
    pub const fn bytes(&self) -> Option<u64> {
        self.bound.bytes()
    }
}

/// Persistent and transient runtime-state estimate for one request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStateEstimate {
    /// Context-independent recurrent/convolution state.
    pub fixed_state_bytes: u64,
    /// Unbounded bytes added per position before multiplying by batch.
    pub bytes_per_position_per_batch: u64,
    /// Persistent context-dependent bytes at the requested length.
    pub context_state_bytes: u64,
    /// Complete selected decoder-state backing across the request, including
    /// capacity padding and distinct retained views. Logical fixed/context
    /// fields remain unchanged; the total uses the larger bound. Absence means
    /// only the architecture's original storage assumptions were supplied.
    #[serde(default)]
    pub selected_state_backing: Option<SelectedStateBacking>,
    /// Prepared-media embedding bytes retained during prefill.
    pub multimodal_embedding_bytes: u64,
    /// Conservative media-tower execution workspace.
    pub media_execution_workspace_bytes: u64,
    /// Modeled persistent state, retained media embeddings and media workspace
    /// for prompt plus output allowance. Text workspace is reported separately.
    pub requested_state_bytes: u64,
    /// Text execution transients beyond modeled state/media. Missing means
    /// unknown, independently of the persistent-state layout's coverage.
    #[serde(default)]
    pub execution_workspace: Option<ExecutionWorkspaceEstimate>,
    /// Coverage of state/media before adding text workspace. A complete state
    /// layout alone does not establish complete inference coverage.
    #[serde(default = "unknown_state_coverage")]
    pub persistent_state_completeness: EstimationCompleteness,
    /// Estimator assumptions.
    pub assumptions: StateMemoryAssumptions,
    /// Estimator coverage.
    pub completeness: EstimationCompleteness,
}

fn unknown_state_coverage() -> EstimationCompleteness {
    EstimationCompleteness::PersistentStateOnly
}

impl RuntimeStateEstimate {
    /// Refines logical state accounting with the selected mechanism's complete
    /// backing bound for this exact request. This never subtracts logical state,
    /// absorbs media into decoder state, or upgrades missing semantic coverage.
    pub fn with_selected_state_backing(
        self,
        geometry: crate::InferenceGeometry,
        backing: WorkspaceBound,
    ) -> Result<Self, CapabilityError> {
        self.with_selected_state_backing_fixed(geometry, backing)
            .map_err(Into::into)
    }

    /// The same backing refinement with allocation-free validation failures.
    pub fn with_selected_state_backing_fixed(
        mut self,
        geometry: crate::InferenceGeometry,
        backing: WorkspaceBound,
    ) -> Result<Self, AdmissionPolicyError> {
        geometry.validate_fixed()?;
        if geometry.batch_size != self.assumptions.batch_size
            || geometry.cached_positions + geometry.input_positions + geometry.max_output_tokens
                != self.assumptions.requested_positions
        {
            return Err(AdmissionPolicyError::InvalidConfiguration {
                field: "selected_state_backing",
                detail: "selected backing and persistent-state geometry differ",
            });
        }
        let logical = state_facts::checked_add(
            self.fixed_state_bytes,
            self.context_state_bytes,
            "logical decoder state",
        )?;
        self.requested_state_bytes = state_facts::checked_add(
            state_facts::checked_add(
                logical.max(backing.bytes().unwrap_or(logical)),
                self.multimodal_embedding_bytes,
                "selected state plus media embeddings",
            )?,
            self.media_execution_workspace_bytes,
            "selected state plus media workspace",
        )?;
        self.selected_state_backing = Some(SelectedStateBacking {
            geometry,
            bound: backing,
        });
        self.refresh_completeness()?;
        Ok(self)
    }

    fn refresh_completeness(&mut self) -> Result<(), AdmissionPolicyError> {
        self.validate_selected_geometry()?;
        let known_execution = self
            .execution_workspace
            .as_ref()
            .map(ExecutionWorkspaceEstimate::peak_bytes_fixed)
            .transpose()?
            .flatten()
            .is_some();
        self.completeness = if known_execution
            && self
                .selected_state_backing
                .as_ref()
                .is_none_or(|bound| bound.bytes().is_some())
            && self.persistent_state_completeness != EstimationCompleteness::PersistentStateOnly
        {
            EstimationCompleteness::Conservative
        } else {
            EstimationCompleteness::PersistentStateOnly
        };
        Ok(())
    }

    fn validate_selected_geometry(&self) -> Result<(), AdmissionPolicyError> {
        AdmissionStateRequirements::from(self)
            .validate_selected_geometry()
            .map_err(Into::into)
    }

    /// Attaches selected native workspace facts to the existing state report.
    /// The estimate must price the same batch and total context allowance.
    pub fn with_execution_workspace(
        self,
        workspace: ExecutionWorkspaceEstimate,
    ) -> Result<Self, CapabilityError> {
        self.with_execution_workspace_fixed(workspace)
            .map_err(Into::into)
    }

    /// The same attachment with allocation-free geometry/arithmetic failures.
    pub fn with_execution_workspace_fixed(
        mut self,
        workspace: ExecutionWorkspaceEstimate,
    ) -> Result<Self, AdmissionPolicyError> {
        workspace.geometry.validate_fixed()?;
        let geometry = workspace.geometry;
        if geometry.batch_size != self.assumptions.batch_size
            || geometry.cached_positions + geometry.input_positions + geometry.max_output_tokens
                != self.assumptions.requested_positions
        {
            return Err(AdmissionPolicyError::InvalidConfiguration {
                field: "execution_workspace",
                detail: "workspace and persistent-state geometry differ",
            });
        }
        self.execution_workspace = Some(workspace);
        self.refresh_completeness()?;
        Ok(self)
    }
}

/// Physical relationship between logical host and device tiers.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalMemorySemantics {
    /// Host and accelerator allocations share physical capacity.
    Unified,
    /// Host and accelerator memory are physically separate.
    SeparateTiers,
    /// The backend cannot determine the relationship.
    Unknown,
}

/// Static checkpoint and current residency observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticMemoryReport {
    /// Logical bytes in parameters or the complete residency plan.
    pub logical_parameter_bytes: Observed<u64>,
    /// Current logical host-resident bytes.
    pub current_host_resident_bytes: Observed<u64>,
    /// Current logical device-resident bytes.
    pub current_device_resident_bytes: Observed<u64>,
    /// Planned logical disk-backed bytes.
    pub planned_disk_backed_bytes: Observed<u64>,
    /// Process-global backend active allocation counter.
    pub backend_active_allocation_bytes: Observed<u64>,
    /// Process-global backend allocator-cache counter.
    pub backend_allocator_cache_bytes: Observed<u64>,
    /// Whether logical host/device tiers share physical capacity.
    pub physical_semantics: PhysicalMemorySemantics,
    /// Currently retained checkpoint shard buffers or readers.
    pub currently_cached_shards: Observed<u64>,
}

/// System memory usable as an admission signal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvailableMemory {
    /// Unified/host physical memory.
    pub physical_memory_bytes: Observed<u64>,
    /// Defensible point-in-time availability estimate.
    pub available_memory_bytes: Observed<u64>,
    /// Physical tier semantics.
    pub physical_semantics: PhysicalMemorySemantics,
}

/// One pre-generation admission request.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct AdmissionRequest {
    /// Authoritative prompt accounting.
    pub input: InputTokenCount,
    /// Maximum generated-token allowance.
    pub max_output_tokens: u64,
    /// Logical batch size.
    pub batch_size: u64,
    /// Caller-selected reserve added to modeled state.
    pub safety_reserve_bytes: u64,
    /// Optional enforceable application budget for incremental state, execution
    /// workspace and reserve. Supplying a budget requires complete bounds.
    pub application_memory_budget_bytes: Option<u64>,
    /// Reject estimates that omit execution transients even without a budget.
    pub require_complete_estimate: bool,
}

/// Detailed successful admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Admission {
    /// Prompt plus output allowance.
    pub requested_positions: u64,
    /// Runtime-state estimate.
    pub state: RuntimeStateEstimate,
    /// Required incremental bytes including the caller reserve. Ordinary policy
    /// uses full state plus workspace; separately proved incremental reporting
    /// may exclude physical storage already charged elsewhere.
    pub incremental_required_bytes: u64,
    /// Availability signal used, when supplied.
    pub available_memory_bytes: Option<u64>,
}

/// Structured reason a request was rejected before generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AdmissionRejection {
    /// Prompt alone exceeds configured context.
    PromptExceedsContext {
        /// Prompt model positions.
        prompt_positions: u64,
        /// Effective model limit.
        maximum_positions: u64,
    },
    /// Prompt fits but output allowance does not.
    OutputHeadroomExceedsContext {
        /// Prompt model positions.
        prompt_positions: u64,
        /// Requested maximum output tokens.
        output_tokens: u64,
        /// Effective model limit.
        maximum_positions: u64,
    },
    /// Application budget is smaller than modeled state, workspace and reserve.
    MemoryBudgetExceeded {
        /// Required incremental bytes.
        required_bytes: u64,
        /// Caller-supplied budget.
        budget_bytes: u64,
    },
    /// Current availability is smaller than modeled state, workspace and reserve.
    InsufficientAvailableMemory {
        /// Required incremental bytes.
        required_bytes: u64,
        /// Observed available bytes.
        available_bytes: u64,
    },
    /// A requested availability check could not be performed.
    AvailableMemoryUnavailable {
        /// Platform report detail.
        reason: String,
    },
    /// Policy requires estimator coverage the model cannot provide.
    EstimationUnsupported {
        /// Coverage detail.
        reason: String,
    },
}

/// Admission outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AdmissionResult {
    /// Request may proceed.
    Admitted(Admission),
    /// Request was rejected.
    Rejected(AdmissionRejection),
}

/// Structured capability and accounting failures.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum CapabilityError {
    /// A validated architecture exposed an invalid value.
    #[error("invalid model capability field {field}: {detail}")]
    InvalidConfiguration {
        /// Invalid field name.
        field: &'static str,
        /// Invalid-value detail.
        detail: String,
    },
    /// Checked byte or position arithmetic overflowed.
    #[error("capability arithmetic overflow while computing {operation}")]
    ArithmeticOverflow {
        /// Stable operation label.
        operation: &'static str,
    },
    /// Prepared input does not match the loaded architecture.
    #[error("unsupported prepared input for {architecture}: {reason}")]
    UnsupportedInput {
        /// Effective architecture name.
        architecture: String,
        /// Unsupported-input detail.
        reason: String,
    },
    /// A runtime observation could not be obtained.
    #[error("capability observation failed: {0}")]
    Observation(String),
}

/// Memory-accounting view of an executable runtime-state layout.
///
/// The ordered layer policies are copied directly from the architecture's
/// executable [`LayerSchedule`]. They are intentionally not summarized into a
/// second scalar geometry, so execution and admission share one semantic
/// source for every state-bearing layer and component.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct StateMemoryLayout {
    /// Exact ordered executable state policies.
    layer_layout: LayerSchedule<LayerCachePolicy>,
    /// Executable processed-token offset for every state layer.
    layer_prefix_offsets: Vec<i32>,
    /// Model hidden width used by retained media embeddings.
    pub hidden_size: u64,
    /// Allocation granularity for unbounded caches.
    pub allocation_granularity: u64,
    /// Coverage supplied by the layout.
    pub completeness: EstimationCompleteness,
}

impl StateMemoryLayout {
    /// Creates accounting metadata around an exact executable layer schedule.
    pub fn new(
        layer_layout: LayerSchedule<LayerCachePolicy>,
        layer_prefix_offsets: Vec<i32>,
        hidden_size: u64,
        allocation_granularity: u64,
        completeness: EstimationCompleteness,
    ) -> Result<Self, CapabilityError> {
        Self::new_with_diagnostic(
            layer_layout,
            layer_prefix_offsets,
            hidden_size,
            allocation_granularity,
            completeness,
            |field, detail| CapabilityError::InvalidConfiguration {
                field,
                detail: detail.to_string(),
            },
        )
    }

    /// Consumes the actual schedule and offsets using caller-owned diagnostics.
    /// Validation order is identical to [`Self::new`]; the callback can reserve
    /// diagnostic storage before constructing an owned error.
    pub fn new_with_diagnostic<E>(
        layer_layout: LayerSchedule<LayerCachePolicy>,
        layer_prefix_offsets: Vec<i32>,
        hidden_size: u64,
        allocation_granularity: u64,
        completeness: EstimationCompleteness,
        mut error: impl FnMut(&'static str, std::fmt::Arguments<'_>) -> E,
    ) -> Result<Self, E> {
        if layer_prefix_offsets.len() != layer_layout.len()
            || layer_prefix_offsets.iter().any(|offset| *offset > 0)
            || hidden_size == 0
            || allocation_granularity == 0
        {
            let (field, detail) = if layer_prefix_offsets.len() != layer_layout.len() {
                (
                    "layer_prefix_offsets",
                    "must contain one entry per executable state layer",
                )
            } else if layer_prefix_offsets.iter().any(|offset| *offset > 0) {
                (
                    "layer_prefix_offsets",
                    "must not advance beyond the request token frontier",
                )
            } else if hidden_size == 0 {
                ("hidden_size", "must be positive")
            } else {
                ("allocation_granularity", "must be positive")
            };
            return Err(error(field, format_args!("{detail}")));
        }
        for (layer, policy) in layer_layout.iter().enumerate() {
            policy
                .validate_with_diagnostic(|detail| {
                    error(
                        "layer_layout",
                        format_args!("invalid state policy at layer {layer}: {detail}"),
                    )
                })?;
        }
        Ok(Self {
            layer_layout,
            layer_prefix_offsets,
            hidden_size,
            allocation_granularity,
            completeness,
        })
    }

    /// Borrows the exact ordered executable state policies.
    pub const fn layer_layout(&self) -> &LayerSchedule<LayerCachePolicy> {
        &self.layer_layout
    }

    /// Returns executable processed-token offsets in state-layer order.
    pub fn layer_prefix_offsets(&self) -> &[i32] {
        &self.layer_prefix_offsets
    }
}

fn checked_add(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_add(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

fn checked_mul(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_mul(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

/// Estimates request state from exact executable layer policies.
pub fn estimate_runtime_state(
    layout: &StateMemoryLayout,
    input: InputTokenCount,
    max_output_tokens: u64,
    batch_size: u64,
    floating_state_dtype_bytes: NonZeroU8,
) -> Result<RuntimeStateEstimate, CapabilityError> {
    estimate_runtime_state_facts(
        layout,
        input,
        max_output_tokens,
        batch_size,
        floating_state_dtype_bytes,
    )
    .map(RuntimeStateFacts::into_estimate)
    .map_err(Into::into)
}

/// Checks context policy before potentially expensive workspace inspection.
/// A successful check is not a memory admission or submission authorization.
pub fn check_admission_context(
    capabilities: &ModelCapabilities,
    request: AdmissionRequest,
) -> Result<Option<AdmissionRejection>, CapabilityError> {
    admission::check_admission_context_borrowed(
        (&capabilities.effective_max_context).into(),
        request,
    )
    .map(|rejection| rejection.map(BorrowedAdmissionRejection::into_owned))
    .map_err(Into::into)
}

/// Applies context and memory policy to an already-computed state estimate.
pub fn apply_admission_policy(
    capabilities: &ModelCapabilities,
    request: AdmissionRequest,
    state: RuntimeStateEstimate,
    available: Option<&AvailableMemory>,
) -> Result<AdmissionResult, CapabilityError> {
    apply_admission_policy_impl(capabilities, request, state, None, available)
}

/// Applies ordinary admission policy using a separately proved complete
/// incremental bound, before adding the caller's safety reserve. The full state
/// and workspace estimate remains unchanged in the resulting diagnostics and
/// must be complete even when the request permits partial legacy estimates.
///
/// The provider must prove which existing physical allocations remain charged
/// elsewhere and exclude them by identity throughout the complete execution
/// trace. Subtracting an old state byte total from an unrelated full peak is not
/// such a proof. An unknown incremental bound rejects admission.
///
/// This performs reporting and policy checks only. It creates no reservation,
/// validates no allocation ownership and grants no execution permission. The
/// runtime must independently bind registered storage and the exact residual
/// quote before reserving less than the full estimate.
pub fn apply_admission_policy_with_incremental(
    capabilities: &ModelCapabilities,
    request: AdmissionRequest,
    state: RuntimeStateEstimate,
    incremental: &WorkspaceBound,
    available: Option<&AvailableMemory>,
) -> Result<AdmissionResult, CapabilityError> {
    apply_admission_policy_impl(capabilities, request, state, Some(incremental), available)
}

fn apply_admission_policy_impl(
    capabilities: &ModelCapabilities,
    request: AdmissionRequest,
    state: RuntimeStateEstimate,
    incremental: Option<&WorkspaceBound>,
    available: Option<&AvailableMemory>,
) -> Result<AdmissionResult, CapabilityError> {
    admission::owned(capabilities, request, state, incremental, available)
}

#[cfg(test)]
mod incremental_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn bounded_fixture_workspace(input: u64, output: u64) -> ExecutionWorkspaceEstimate {
        let zero = || WorkspaceBound::bounded(0, "fixture has no such allocation");
        ExecutionWorkspaceEstimate {
            geometry: crate::InferenceGeometry {
                batch_size: 1,
                cached_positions: 0,
                input_positions: input,
                max_output_tokens: output,
                prefill_chunk_positions: input,
                output: crate::OutputDemand::LastPosition,
            },
            activations: WorkspaceBound::bounded(64, "fixture activation bound"),
            attention: zero(),
            vocabulary: WorkspaceBound::bounded(32, "fixture score bound"),
            state_update: zero(),
            materialization: zero(),
            retained: zero(),
        }
    }

    #[test]
    fn selected_backing_and_workspace_require_exact_geometry_in_either_attachment_order() {
        let layout = StateMemoryLayout::new(
            LayerSchedule::empty(),
            vec![],
            1,
            1,
            EstimationCompleteness::Complete,
        )
        .unwrap();
        let base = estimate_runtime_state(
            &layout,
            InputTokenCount::text(3),
            2,
            1,
            NonZeroU8::new(4).unwrap(),
        )
        .unwrap();
        let workspace = bounded_fixture_workspace(3, 2);
        let g = workspace.geometry;
        for changed in [
            crate::InferenceGeometry {
                prefill_chunk_positions: 1,
                ..g
            },
            crate::InferenceGeometry {
                cached_positions: 1,
                input_positions: 2,
                prefill_chunk_positions: 2,
                ..g
            },
            crate::InferenceGeometry {
                input_positions: 2,
                max_output_tokens: 3,
                prefill_chunk_positions: 2,
                ..g
            },
            crate::InferenceGeometry {
                output: crate::OutputDemand::Sequence,
                ..g
            },
        ] {
            let mut changed_workspace = workspace.clone();
            changed_workspace.geometry = changed;
            let bound = || WorkspaceBound::bounded(64, "fixture bound for exact schedule");
            assert!(
                base.clone()
                    .with_selected_state_backing(g, bound())
                    .unwrap()
                    .with_execution_workspace(changed_workspace.clone())
                    .is_err()
            );
            assert!(
                base.clone()
                    .with_execution_workspace(changed_workspace)
                    .unwrap()
                    .with_selected_state_backing(g, bound())
                    .is_err()
            );
        }
        let mut estimate = base
            .with_selected_state_backing(g, WorkspaceBound::bounded(64, "fixture"))
            .unwrap()
            .with_execution_workspace(workspace)
            .unwrap();
        estimate
            .selected_state_backing
            .as_mut()
            .unwrap()
            .geometry
            .prefill_chunk_positions = 1;
        let decoded: RuntimeStateEstimate =
            serde_json::from_value(serde_json::to_value(estimate).unwrap()).unwrap();
        let capabilities = ModelCapabilities {
            effective_model_type: "geometry fixture".into(),
            native_max_context: Observed::exact(8, "fixture"),
            effective_max_context: Observed::exact(8, "fixture"),
            state_strategy: CacheStateStrategy::FullKv,
            modalities: InputModalities::TEXT,
            estimation: EstimationCompleteness::Complete,
        };
        assert!(
            apply_admission_policy(
                &capabilities,
                AdmissionRequest {
                    input: InputTokenCount::text(3),
                    max_output_tokens: 2,
                    batch_size: 1,
                    application_memory_budget_bytes: None,
                    safety_reserve_bytes: 0,
                    require_complete_estimate: true,
                },
                decoded,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn selected_state_backing_preserves_logical_and_media_categories_and_missing_coverage() {
        let layout = StateMemoryLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value(crate::AttentionPolicy::Full, 1, 4).unwrap()],
            )
            .unwrap(),
            vec![0],
            8,
            1,
            EstimationCompleteness::Complete,
        )
        .unwrap();
        let input = InputTokenCount::prepared(2, 1, 3, 64, ObservationKind::Exact);
        let base =
            estimate_runtime_state(&layout, input, 2, 1, NonZeroU8::new(4).unwrap()).unwrap();
        let workspace = bounded_fixture_workspace(3, 2);
        let geometry = workspace.geometry;
        let logical = base.fixed_state_bytes + base.context_state_bytes;
        let refined = base
            .clone()
            .with_selected_state_backing(
                geometry,
                WorkspaceBound::bounded(
                    logical + 4096,
                    "certified selected padding and retained views",
                ),
            )
            .unwrap()
            .with_execution_workspace(workspace.clone())
            .unwrap();
        assert_eq!(
            refined.requested_state_bytes,
            base.requested_state_bytes + 4096
        );
        assert_eq!(refined.context_state_bytes, base.context_state_bytes);
        assert_eq!(refined.fixed_state_bytes, base.fixed_state_bytes);
        assert_eq!(refined.multimodal_embedding_bytes, 32);
        assert_eq!(refined.media_execution_workspace_bytes, 64);
        assert_eq!(refined.completeness, EstimationCompleteness::Conservative);
        let restored: RuntimeStateEstimate =
            serde_json::from_value(serde_json::to_value(&refined).unwrap()).unwrap();
        assert_eq!(restored, refined);
        // Replacing a bound recalculates the total; neither repeated attachment
        // nor an undersized selected bound subtracts logical or media storage.
        let smaller = refined
            .with_selected_state_backing(
                geometry,
                WorkspaceBound::bounded(0, "fixture replacement"),
            )
            .unwrap();
        assert_eq!(smaller.requested_state_bytes, base.requested_state_bytes);
        let unknown = smaller
            .with_selected_state_backing(
                geometry,
                WorkspaceBound::Unknown {
                    reason: "missing native backing".into(),
                },
            )
            .unwrap()
            .with_execution_workspace(workspace.clone())
            .unwrap();
        assert_eq!(
            unknown.completeness,
            EstimationCompleteness::PersistentStateOnly
        );
        let mut incomplete = base.clone();
        incomplete.persistent_state_completeness = EstimationCompleteness::PersistentStateOnly;
        assert_eq!(
            incomplete
                .with_selected_state_backing(
                    geometry,
                    WorkspaceBound::bounded(logical, "fixture complete backing")
                )
                .unwrap()
                .with_execution_workspace(workspace)
                .unwrap()
                .completeness,
            EstimationCompleteness::PersistentStateOnly
        );
        assert!(
            base.clone()
                .with_selected_state_backing(
                    crate::InferenceGeometry {
                        batch_size: 2,
                        ..geometry
                    },
                    WorkspaceBound::bounded(logical, "wrong batch")
                )
                .is_err()
        );
        assert!(matches!(
            base.with_selected_state_backing(
                geometry,
                WorkspaceBound::bounded(u64::MAX, "overflow fixture")
            ),
            Err(CapabilityError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn observational_media_cannot_become_a_complete_bound_by_adding_text_workspace() {
        let layout = StateMemoryLayout::new(
            LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
            vec![0],
            8,
            1,
            EstimationCompleteness::Complete,
        )
        .unwrap();
        for kind in [
            ObservationKind::Exact,
            ObservationKind::Conservative,
            ObservationKind::Observational,
            ObservationKind::Estimated,
        ] {
            let input = InputTokenCount::prepared(0, 1, 1, 64, kind);
            let state = estimate_runtime_state(&layout, input, 0, 1, NonZeroU8::new(4).unwrap())
                .unwrap()
                .with_execution_workspace(bounded_fixture_workspace(1, 0))
                .unwrap();
            assert_eq!(
                state.completeness == EstimationCompleteness::Conservative,
                matches!(kind, ObservationKind::Exact | ObservationKind::Conservative)
            );
        }
    }

    #[test]
    fn workspace_geometry_and_arithmetic_fail_before_admission() {
        let mut workspace = bounded_fixture_workspace(5, 2);
        workspace.activations = WorkspaceBound::bounded(u64::MAX, "overflow fixture");
        assert!(matches!(
            workspace.peak_bytes(),
            Err(CapabilityError::ArithmeticOverflow { .. })
        ));
        workspace.geometry.cached_positions = u64::MAX;
        assert!(matches!(
            workspace.peak_bytes(),
            Err(CapabilityError::ArithmeticOverflow { .. })
        ));
        workspace.geometry.cached_positions = 0;
        workspace.geometry.prefill_chunk_positions = 0;
        assert!(matches!(
            workspace.peak_bytes(),
            Err(CapabilityError::InvalidConfiguration { .. })
        ));
    }

    #[test]
    fn state_estimation_and_admission_are_backend_independent() {
        let policies = (0..2)
            .map(|_| LayerCachePolicy::key_only(AttentionPolicy::Full, 1, 8).unwrap())
            .collect::<Vec<_>>();
        let layout = StateMemoryLayout::new(
            LayerSchedule::new(2, policies).unwrap(),
            vec![0; 2],
            32,
            8,
            EstimationCompleteness::Complete,
        )
        .unwrap();
        let input = InputTokenCount::text(5);
        let state =
            estimate_runtime_state(&layout, input, 2, 1, NonZeroU8::new(4).unwrap()).unwrap();
        assert_eq!(state.assumptions.requested_positions, 7);
        assert_eq!(state.context_state_bytes, 512);
        assert_eq!(
            state.completeness,
            EstimationCompleteness::PersistentStateOnly
        );
        let state = state
            .with_execution_workspace(bounded_fixture_workspace(5, 2))
            .unwrap();
        let capabilities = ModelCapabilities {
            effective_model_type: "mock".into(),
            native_max_context: Observed::exact(16, "mock"),
            effective_max_context: Observed::exact(16, "mock"),
            state_strategy: CacheStateStrategy::FullKv,
            modalities: InputModalities::TEXT,
            estimation: EstimationCompleteness::Complete,
        };
        let serialized = serde_json::to_value(&capabilities).unwrap();
        assert_eq!(serialized["effective_model_type"], "mock");
        assert!(serialized.get("model_type").is_none());
        assert!(matches!(
            apply_admission_policy(
                &capabilities,
                AdmissionRequest {
                    input,
                    max_output_tokens: 2,
                    batch_size: 1,
                    safety_reserve_bytes: 0,
                    application_memory_budget_bytes: Some(1024),
                    require_complete_estimate: true
                },
                state,
                None
            )
            .unwrap(),
            AdmissionResult::Admitted(_)
        ));
    }

    #[test]
    fn admission_rejections_are_portable_and_fail_closed() {
        let capabilities = ModelCapabilities {
            effective_model_type: "mock".into(),
            native_max_context: Observed::exact(8, "mock"),
            effective_max_context: Observed::exact(8, "mock"),
            state_strategy: CacheStateStrategy::FullKv,
            modalities: InputModalities::TEXT,
            estimation: EstimationCompleteness::Complete,
        };
        let state = RuntimeStateEstimate {
            fixed_state_bytes: 0,
            bytes_per_position_per_batch: 0,
            context_state_bytes: 0,
            selected_state_backing: None,
            multimodal_embedding_bytes: 0,
            media_execution_workspace_bytes: 0,
            requested_state_bytes: 0,
            execution_workspace: None,
            persistent_state_completeness: EstimationCompleteness::Complete,
            assumptions: StateMemoryAssumptions {
                floating_state_dtype_bytes: NonZeroU8::new(4).unwrap(),
                batch_size: 1,
                requested_positions: 9,
                sliding_window_bounds: Vec::new(),
                allocation_granularity: 1,
            },
            completeness: EstimationCompleteness::Complete,
        };
        let request = AdmissionRequest {
            input: InputTokenCount::text(7),
            max_output_tokens: 2,
            batch_size: 1,
            safety_reserve_bytes: 0,
            application_memory_budget_bytes: None,
            require_complete_estimate: true,
        };
        assert!(matches!(
            apply_admission_policy(&capabilities, request, state, None).unwrap(),
            AdmissionResult::Rejected(AdmissionRejection::OutputHeadroomExceedsContext { .. })
        ));

        let unavailable = AvailableMemory {
            physical_memory_bytes: Observed::unavailable("not reported"),
            available_memory_bytes: Observed::unavailable("not reported"),
            physical_semantics: PhysicalMemorySemantics::Unknown,
        };
        let request = AdmissionRequest {
            input: InputTokenCount::text(1),
            max_output_tokens: 0,
            batch_size: 1,
            safety_reserve_bytes: 0,
            application_memory_budget_bytes: None,
            require_complete_estimate: true,
        };
        let state = estimate_runtime_state(
            &StateMemoryLayout::new(
                LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
                vec![0],
                1,
                1,
                EstimationCompleteness::Complete,
            )
            .unwrap(),
            request.input,
            0,
            1,
            NonZeroU8::new(4).unwrap(),
        )
        .unwrap();
        let state = state
            .with_execution_workspace(bounded_fixture_workspace(1, 0))
            .unwrap();
        assert!(matches!(
            apply_admission_policy(&capabilities, request, state, Some(&unavailable)).unwrap(),
            AdmissionResult::Rejected(AdmissionRejection::AvailableMemoryUnavailable { .. })
        ));
    }

    #[test]
    fn capability_and_memory_schemas_round_trip_without_a_backend() {
        let report = StaticMemoryReport {
            logical_parameter_bytes: Observed::exact(1_024, "mock catalog"),
            current_host_resident_bytes: Observed::exact(512, "mock ledger"),
            current_device_resident_bytes: Observed::exact(512, "mock ledger"),
            planned_disk_backed_bytes: Observed::exact(0, "mock plan"),
            backend_active_allocation_bytes: Observed::unavailable("no allocator probe"),
            backend_allocator_cache_bytes: Observed::unsupported("no allocator cache"),
            physical_semantics: PhysicalMemorySemantics::SeparateTiers,
            currently_cached_shards: Observed::exact(1, "mock store"),
        };
        let encoded = serde_json::to_string(&report).unwrap();
        let decoded: StaticMemoryReport = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, report);
    }
}

#[cfg(test)]
#[path = "capability/admission_tests.rs"]
mod admission_tests;
