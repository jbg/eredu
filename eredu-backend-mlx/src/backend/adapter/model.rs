use super::*;

/// Opaque MLX executable selected for one complete model session.
///
/// Replicated and axis-partitioned materializations share
/// this type. Architecture-specific rank-local executables are deliberately
/// not exposed through the public loading API.
pub struct MlxModel {
    executable: Executable,
    target: MlxPreparedTarget,
    floating_state_dtype_bytes: NonZeroU8,
    state_residency: eredu_runtime::CacheResidencyPolicy,
    distributed: Option<MlxDistributedSession>,
    capture_discovery: Option<eredu_core::capture::CaptureDiscovery>,
    intervention_discovery: Option<eredu_core::intervention::InterventionDiscovery>,
    #[cfg(any(feature = "image", feature = "audio"))]
    processor: Option<ModelProcessor>,
}

impl MlxModel {
    pub(crate) fn with_intervention_discovery(
        mut self,
        discovery: eredu_core::intervention::InterventionDiscovery,
    ) -> Self {
        self.intervention_discovery = Some(discovery);
        self
    }

    pub(crate) fn take_intervention_discovery(
        &mut self,
    ) -> Option<eredu_core::intervention::InterventionDiscovery> {
        self.intervention_discovery.take()
    }

    pub(crate) fn new(
        model: Executable,
        floating_state_dtype_bytes: NonZeroU8,
        state_residency: eredu_runtime::CacheResidencyPolicy,
        target: MlxPreparedTarget,
    ) -> Self {
        Self {
            executable: model,
            target,
            floating_state_dtype_bytes,
            state_residency,
            distributed: None,
            capture_discovery: None,
            intervention_discovery: None,
            #[cfg(any(feature = "image", feature = "audio"))]
            processor: None,
        }
    }

    pub(crate) const fn floating_state_dtype_bytes(&self) -> NonZeroU8 {
        self.floating_state_dtype_bytes
    }

    pub(crate) fn with_capture_discovery(
        mut self,
        discovery: eredu_core::capture::CaptureDiscovery,
    ) -> Self {
        self.capture_discovery = Some(discovery);
        self
    }

    pub(crate) fn take_capture_discovery(
        &mut self,
    ) -> Option<eredu_core::capture::CaptureDiscovery> {
        self.capture_discovery.take()
    }

    pub(crate) const fn state_residency(&self) -> &eredu_runtime::CacheResidencyPolicy {
        &self.state_residency
    }

    /// Reports speculative-weight readiness to backend integration tests.
    #[cfg(test)]
    pub fn speculative_capability_for_test(&self) -> eredu_core::SpeculativeCapability {
        self.executable.speculative_capability()
    }

    #[cfg(test)]
    pub(crate) fn into_executable(self) -> Executable {
        self.executable
    }

    pub(crate) const fn native_target(&self) -> &MlxPreparedTarget {
        &self.target
    }

    pub(crate) fn into_execution_parts(self) -> (Executable, MlxPreparedTarget) {
        (self.executable, self.target)
    }

    pub(crate) fn take_distributed(&mut self) -> Option<MlxDistributedSession> {
        self.distributed.take()
    }

    pub(crate) fn with_distributed(mut self, distributed: MlxDistributedSession) -> Self {
        self.distributed = Some(distributed);
        self
    }

    #[cfg(any(feature = "image", feature = "audio"))]
    pub(crate) fn take_processor(&mut self) -> Option<ModelProcessor> {
        self.processor.take()
    }

    #[cfg(any(feature = "image", feature = "audio"))]
    pub(crate) fn with_processor(mut self, processor: Option<ModelProcessor>) -> Self {
        self.processor = processor;
        self
    }

    #[cfg(test)]
    pub(crate) fn effective_model_type(&self) -> &str {
        self.executable.effective_model_type()
    }

    /// Returns bounded parameter-residency telemetry when available.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        self.executable.residency_report()
    }

    /// Returns dense checkpoint-streaming telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        self.executable.dense_stream_report()
    }

    /// Returns load-time weight transformation telemetry when present.
    pub fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport> {
        self.executable.materialization_report()
    }

    /// Returns addressable parameter-bank residency telemetry when enabled.
    pub fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        self.executable.parameter_bank_report()
    }
}
