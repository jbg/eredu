//! Scalar diagnostics for a rejected, fully constructed workspace candidate.
use eredu_core::InferenceGeometry;
use eredu_runtime::working_memory::PrefillPlanningError;
use std::fmt;

/// Descriptive components of the last rejected candidate. Components can
/// overlap; this report grants no capacity or authority and is not additive.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkspaceQuoteComponents {
    pub(crate) before_seal: Option<u64>,
    pub(crate) after_seal: Option<u64>,
    pub(crate) state: Option<u64>,
    pub(crate) activations: Option<u64>,
    pub(crate) vocabulary: Option<u64>,
    pub(crate) retained: Option<u64>,
    pub(crate) graph: Option<u64>,
    pub(crate) tracking: Option<u64>,
    pub(crate) pipeline: Option<u64>,
    pub(crate) recipe: Option<u64>,
    pub(crate) validation: Option<u64>,
    pub(crate) native_capacity: Option<u64>,
    pub(crate) native_controls: Option<u64>,
    pub(crate) direct_controls: Option<u64>,
    pub(crate) collector_controls: Option<u64>,
    pub(crate) publication_attempts: Option<usize>,
    pub(crate) publication_rows: Option<usize>,
    pub(crate) kernel_attempts: Option<usize>,
    pub(crate) equation_rows: usize,
    pub(crate) mixed_rows: usize,
    pub(crate) gpu_entries: Option<usize>,
    pub(crate) cpu_entries: Option<usize>,
    pub(crate) nested_frontiers: Option<usize>,
    pub(crate) sampling_gpu_entries: Option<usize>,
    pub(crate) sampling_cpu_entries: Option<usize>,
    pub(crate) sampling_nested_frontiers: Option<usize>,
}

impl WorkspaceQuoteComponents {
    /// Exact sealed incremental requirement, before policy safety reserves.
    pub fn incremental_bytes(&self) -> Option<u64> { self.after_seal }
    /// Original cumulative native graph constructor allowance.
    pub fn graph_bytes(&self) -> Option<u64> { self.graph }
    /// Request-owned precompiled pipeline and selector allowance.
    pub fn pipeline_bytes(&self) -> Option<u64> { self.pipeline }
    /// Separate native publication/control allowance; excludes native payload.
    pub fn native_control_bytes(&self) -> Option<u64> { self.native_controls }
    /// Selected native physical backing capacity.
    pub fn native_capacity_bytes(&self) -> Option<u64> { self.native_capacity }
    /// Certified shader lookup population, including selected completion policy.
    pub fn kernel_attempts(&self) -> Option<usize> { self.kernel_attempts }
}

/// The original typed refusal together with its already-funded scalar report.
#[derive(Debug)]
pub struct WorkspaceCandidateRefusal {
    pub(crate) geometry: InferenceGeometry,
    pub(crate) components: WorkspaceQuoteComponents,
    pub(crate) initial: Option<(InferenceGeometry, WorkspaceQuoteComponents)>,
    pub(crate) minimum: Option<(InferenceGeometry, WorkspaceQuoteComponents)>,
    pub(crate) cause: PrefillPlanningError,
    pub(crate) _funding: Option<eredu_core::HostMetadataFunding>,
}
impl WorkspaceCandidateRefusal {
    /// Geometry actually inspected for the last rejected candidate.
    pub fn geometry(&self) -> InferenceGeometry { self.geometry }
    /// Read-only component evidence from the original producer.
    pub fn components(&self) -> &WorkspaceQuoteComponents { &self.components }
    /// First complete candidate, before any smaller-chunk retry.
    pub fn initial(&self) -> Option<&(InferenceGeometry, WorkspaceQuoteComponents)> { self.initial.as_ref() }
    /// Smallest complete requirement actually observed by the shared planner.
    pub fn minimum(&self) -> Option<&(InferenceGeometry, WorkspaceQuoteComponents)> { self.minimum.as_ref() }
}
impl fmt::Display for WorkspaceCandidateRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}; candidate chunk={}, input={}, output={}: {:?}", self.cause,
            self.geometry.prefill_chunk_positions, self.geometry.input_positions,
            self.geometry.max_output_tokens, self.components)?;
        if let Some((geometry, components)) = &self.initial {
            if *geometry != self.geometry {
                write!(f, "; initial chunk={}: {:?}", geometry.prefill_chunk_positions, components)?;
            }
        }
        if let Some((geometry, components)) = &self.minimum {
            if *geometry != self.geometry && self.initial.as_ref().is_none_or(|(initial, _)| initial != geometry) {
                write!(f, "; minimum chunk={}: {:?}", geometry.prefill_chunk_positions, components)?;
            }
        }
        Ok(())
    }
}
impl std::error::Error for WorkspaceCandidateRefusal {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { Some(&self.cause) }
}
