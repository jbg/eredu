//! Typed executable contract for architecture-owned retained prefill sessions.

use eredu_core::{
    capture::CaptureRequestShape, BackendFailure, CapabilityError, SessionCapabilities,
};
use eredu_nn::Tensor;
use eredu_runtime::{
    execution_control::SessionState,
    prefill::{RetainedPrefillAdvance, RetainedPrefillCursor},
    PartitionOutputAuthority, PartitionOutputPublication, PreparedInputCacheIdentity,
    PreparedInputInspector, PreparedModelInput, ResidencyReport, SelectedProcessorExecution,
};

use crate::{capability::CapabilityEstimate, media_plan::PreparedInputPartPlan};

/// Fully constructed rank and the retained facts used by the outer loader.
pub struct RealizedRank<R> {
    /// Typed portable text session.
    pub session: R,
    /// Selected raw processor when applicable.
    pub raw_processor: Option<crate::processor_execution::PreparedProcessor>,
    /// Realized session capabilities.
    pub capabilities: SessionCapabilities,
    /// Width of the selected floating state.
    pub floating_state_bytes: std::num::NonZeroU8,
}

/// Backend-neutral invocation and state contract for a retained prefill session.
pub trait PreparedRetainedTextSession<T: Tensor, S> {
    /// Typed prefill continuation erased only at the backend API boundary.
    type Cursor: RetainedPrefillCursor<Tensor = T> + 'static;

    /// Retained processor admission.
    fn processor(&self) -> &SelectedProcessorExecution;
    /// Retained capability and memory facts.
    fn capabilities(&self) -> &CapabilityEstimate;
    /// Realized optional session capabilities.
    fn session_capabilities(&self) -> SessionCapabilities;
    /// Installed decoder state.
    fn state(&self) -> &SessionState<S>;
    /// Installed decoder state at a mutable session boundary.
    fn state_mut(&mut self) -> &mut SessionState<S>;
    /// Selected score publication and rank-local authority.
    fn publication(&self) -> Option<(PartitionOutputPublication, PartitionOutputAuthority)> {
        None
    }
    /// Weight residency of each bound component.
    fn component_residency(&self) -> Result<Vec<ResidencyReport>, BackendFailure> {
        Ok(Vec::new())
    }
    /// Overall residency when the components share a single policy.
    fn residency(&self) -> Result<Option<ResidencyReport>, BackendFailure> {
        Ok(None)
    }
    /// Dense-stream telemetry for a shared policy, when selected.
    fn dense_stream(&self) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, BackendFailure> {
        Ok(None)
    }
    /// Releases optional parameter conversion retention at an idle boundary.
    fn trim_parameter_conversions(
        &self,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        Err(eredu_core::residency::ParameterConversionTrimError::Unsupported)
    }
    /// Selected decoder parameter residency.
    fn decoder_residency(&self) -> eredu_runtime::LayerWeightResidency;
    /// Reports the admitted input parts without starting execution.
    fn input_plans(
        &self,
        input: &PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
    ) -> Result<Vec<PreparedInputPartPlan>, CapabilityError>;
    /// Reports the exact expanded decoder token IDs.
    fn input_tokens(
        &self,
        input: &PreparedModelInput<T>,
        inspector: &impl PreparedInputInspector<T>,
        identity: Option<&PreparedInputCacheIdentity>,
    ) -> Result<Vec<u32>, BackendFailure>;
    /// Admits prefill on all ranks, including rank-local conversion failures.
    fn start_prefill(
        &mut self,
        input: Result<PreparedModelInput<T>, BackendFailure>,
        inspector: &impl PreparedInputInspector<T>,
        identity: Option<&PreparedInputCacheIdentity>,
        max_tokens: usize,
        capture: Option<CaptureRequestShape>,
        context: &T::Context,
    ) -> Result<Self::Cursor, BackendFailure>;
    /// Advances one typed prefill step.
    fn advance_prefill(
        &mut self,
        cursor: &mut Self::Cursor,
        context: &T::Context,
        observer: &mut dyn eredu_runtime::ActivationObserver<T, eredu_nn::Error>,
    ) -> Result<Option<RetainedPrefillAdvance<T>>, BackendFailure>;
    /// Executes one cached decoder input, agreeing local input failures.
    fn decode(
        &mut self,
        tokens: Result<T, BackendFailure>,
        context: &T::Context,
        observer: &mut dyn eredu_runtime::ActivationObserver<T, eredu_nn::Error>,
    ) -> Result<T, BackendFailure>;
    /// Allocates the rank-local state for a local reset.
    fn fresh_state(&self) -> Result<S, BackendFailure>;
    /// Resets the installed state at a coordinated rank boundary.
    fn reset(&mut self, context: &T::Context) -> Result<(), BackendFailure>;
}
