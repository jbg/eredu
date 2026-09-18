//! Output demand and exact geometry for bounded inference.

use serde::{Deserialize, Serialize};

mod filter_workspace;
pub use filter_workspace::TextFilterWorkspace;

/// Execution limits for one ordinary or controlled text request.
/// Memory capacity includes the shared managed domain's retained residency and
/// concurrent admitted work. It is not a total process-memory guarantee.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TextInferencePolicy {
    /// Maximum positions per prefill submission. Absence uses runtime policy;
    /// admission may select a smaller chunk to satisfy the memory capacity.
    pub prefill_chunk_positions: Option<std::num::NonZeroU64>,
    /// Enforced managed-domain capacity. Unknown required bounds must reject
    /// before prompt or sampling construction. A zero capacity is valid and
    /// rejects requests needing any managed storage.
    ///
    /// A larger capacity on a later managed request in the same session
    /// authorizes completed predecessor funding to adopt that ceiling when
    /// the session retains the original owner's succession capability. All
    /// native scopes must be certified and the funding run closed. Retained
    /// outputs stay charged; fixed pool limits and unrelated requests still
    /// constrain admission. The policy change commits with successful admission
    /// and survives a subsequent preparation failure.
    pub managed_memory_capacity_bytes: Option<u64>,
    /// Explicit component ceiling for storage tracking submitted work. Native
    /// mechanisms enforce it before tracked allocation; exhaustion may refuse
    /// execution even after successful original admission. It is separate from
    /// total managed capacity and is not an estimate that the graph will fit.
    /// Original token-input managed requests derive a complete required amount
    /// before admission when absent. Ordinary component diagnostics retain their
    /// existing behavior; no default may be introduced after a refusal.
    #[serde(default)]
    pub submission_tracking_capacity_bytes: Option<std::num::NonZeroU64>,
    /// Explicit ceiling for migrated native graph metadata: descriptors, their
    /// shared controls, shape/stride/array-edge buffers and primitive shells.
    /// This enforced component can refuse execution after admission; it neither
    /// predicts graph fit nor covers backing, tasks or other native allocations.
    /// Original token-input managed requests derive a complete required amount
    /// before admission when absent. Ordinary component diagnostics retain their
    /// existing behavior, and capacity cannot be refilled after refusal.
    #[serde(default)]
    pub graph_metadata_capacity_bytes: Option<std::num::NonZeroU64>,
}

impl TextInferencePolicy {
    /// Enforced requests need a finite output allowance to price all decode,
    /// sampling and retained-state work before the first native preparation.
    pub fn validate(self, max_output_tokens: Option<usize>) -> Result<(), crate::CapabilityError> {
        if self.submission_tracking_capacity_bytes.is_some()
            && self.managed_memory_capacity_bytes.is_none()
        {
            return Err(crate::CapabilityError::InvalidConfiguration {
                field: "submission_tracking_capacity_bytes",
                detail: "submission tracking requires an original managed-domain capacity".into(),
            });
        }
        if self.graph_metadata_capacity_bytes.is_some()
            && self.managed_memory_capacity_bytes.is_none()
        {
            return Err(crate::CapabilityError::InvalidConfiguration {
                field: "graph_metadata_capacity_bytes",
                detail: "graph metadata requires an original managed-domain capacity".into(),
            });
        }
        if self.managed_memory_capacity_bytes.is_some() && max_output_tokens.is_none() {
            return Err(crate::CapabilityError::InvalidConfiguration {
                field: "text_inference_policy",
                detail: "enforced inference memory requires a finite output-token allowance".into(),
            });
        }
        Ok(())
    }
}

/// Borrowed diagnostics of the actual accepted generation request.
///
/// The selected geometry may use a smaller prefill chunk than the requested
/// policy cap. The existing admission reports its historical incremental
/// requirement, including any proved existing-storage credit; it is neither a
/// measured high-water mark nor the managed domain's current total usage.
/// This loan clones no report or reservation and grants no execution authority.
#[derive(Debug, Clone, Copy)]
pub struct TextPreparationReport<'a> {
    /// Geometry selected by the accepted request, including its prefill chunk.
    pub geometry: InferenceGeometry,
    /// Existing successful admission retained by this request.
    pub admission: &'a crate::Admission,
}

/// Cold controller contribution for the complete remaining output allowance.
/// The filter is a geometry witness: every returned decision must fit its native
/// filtering mechanism and payload bound. Controller construction and mutation
/// must not occur while obtaining this description.
#[derive(Debug, Clone, Copy)]
pub struct TextControllerWorkspace<'a> {
    /// Filtering mechanisms and mask geometry permitted throughout the run.
    /// An optional mask is metadata and does not allocate a witness payload.
    pub filter: TextFilterWorkspace<'a>,
    /// All additional controller-owned numerical payload, including retained
    /// state, decision copies and growth overlap, beyond the final decision
    /// filter payload already priced by sampling. Unknown must return no description.
    pub additional_host_bytes: u64,
}

/// Vocabulary output required by one execution. This is selected before readout,
/// independently of hidden values retained for prediction or observation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputDemand {
    /// Update model state without executing vocabulary readout.
    StateOnly,
    /// Project the last position, retaining a sequence axis of length one.
    LastPosition,
    /// Project every position; direct scoring and verification keep this demand.
    Sequence,
}

impl OutputDemand {
    /// Number of projected positions for a nonempty input chunk.
    pub const fn positions(self, input_positions: u64) -> u64 {
        match self {
            Self::StateOnly => 0,
            Self::LastPosition => {
                if input_positions == 0 {
                    0
                } else {
                    1
                }
            }
            Self::Sequence => input_positions,
        }
    }

    /// Demand for a chunk of one logical prompt. A final-position consumer
    /// produces no intermediate scores; sequence consumers retain every row.
    pub const fn for_chunk(self, final_chunk: bool) -> Self {
        match self {
            Self::LastPosition if !final_chunk => Self::StateOnly,
            other => other,
        }
    }
}

/// Geometry priced by a selected inference mechanism. Bounds are invalid for
/// different geometry, even when a coincidental byte total is equal.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct InferenceGeometry {
    /// Logical batch size.
    pub batch_size: u64,
    /// Positions already committed to mutable state.
    pub cached_positions: u64,
    /// New decoder positions in the complete prepared prompt.
    pub input_positions: u64,
    /// Reserved generated positions, including the first prediction.
    pub max_output_tokens: u64,
    /// Maximum new decoder positions in any prefill chunk.
    pub prefill_chunk_positions: u64,
    /// Logical vocabulary output required for the complete input.
    pub output: OutputDemand,
}

impl InferenceGeometry {
    /// Checks sizes and frontier arithmetic before asking for native workspace.
    pub fn validate(self) -> Result<(), crate::CapabilityError> {
        self.validate_fixed().map_err(Into::into)
    }

    /// Checks geometry and frontier arithmetic without allocating diagnostics.
    ///
    /// This is the fixed-error companion to [`Self::validate`]. It preserves
    /// the same rejection order and establishes no admission or execution
    /// authority; callers still use the shared admission policy and grant.
    pub fn validate_fixed(self) -> Result<(), crate::AdmissionPolicyError> {
        // A completed saved-state placement has no next input or prediction.
        // Its copy work is separately admitted; this shape grants no forward.
        if self.batch_size > 0 && self.input_positions == 0
            && self.prefill_chunk_positions == 0 && self.max_output_tokens == 0
            && self.output == OutputDemand::StateOnly {
            return Ok(());
        }
        if self.batch_size == 0
            || self.input_positions == 0
            || self.prefill_chunk_positions == 0
            || self.prefill_chunk_positions > self.input_positions
        {
            return Err(
                crate::capability::AdmissionPolicyError::InvalidConfiguration {
                    field: "inference_geometry",
                    detail: "batch, input and chunk must be positive; chunk cannot exceed input",
                },
            );
        }
        self.cached_positions
            .checked_add(self.input_positions)
            .and_then(|n| n.checked_add(self.max_output_tokens))
            .ok_or(
                crate::capability::AdmissionPolicyError::ArithmeticOverflow {
                    operation: "inference frontier",
                },
            )?;
        Ok(())
    }
}
