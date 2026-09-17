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
    capture_discovery: Option<eredu_architectures::prepared_sources::PreparedModelDiscovery>,
    #[cfg(any(feature = "image", feature = "audio"))]
    processor: Option<ModelProcessor>,
    // Drop after native executable, processor and communication resources.
    memory_owner: Option<crate::backend::managed_memory::NativeMemoryOwner>,
}

impl MlxModel {
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
            #[cfg(any(feature = "image", feature = "audio"))]
            processor: None,
            memory_owner: None,
        }
    }

    pub(crate) fn executable(&self) -> &Executable {
        &self.executable
    }

    pub(crate) fn executable_mut(&mut self) -> &mut Executable {
        &mut self.executable
    }

    pub(crate) fn with_memory_owner(
        mut self,
        owner: crate::backend::managed_memory::NativeMemoryOwner,
    ) -> Result<Self, Error> {
        // The provider's detached loading scope retains this authority through
        // any unresolved native work. Realize only existing retained module
        // roots before rebuilding cold inventory for initial publication.
        self.executable.settle_loaded_numerical_values(&owner)?;
        self.executable.retain_memory_owner(owner.clone())?;
        self.memory_owner = Some(owner);
        Ok(self)
    }

    pub(crate) fn memory_owner(
        &self,
    ) -> Option<&crate::backend::managed_memory::NativeMemoryOwner> {
        self.memory_owner.as_ref()
    }

    pub(crate) const fn floating_state_dtype_bytes(&self) -> NonZeroU8 {
        self.floating_state_dtype_bytes
    }

    pub(crate) fn with_capture_discovery(
        mut self,
        discovery: eredu_architectures::prepared_sources::PreparedModelDiscovery,
    ) -> Result<Self, Error> {
        self.capture_discovery = Some(
            discovery
                .bind_partition_parameters(
                    self.executable
                        .erased()
                        .partition_parameter_description()
                        .cloned(),
                )
                .and_then(|discovery| {
                    discovery.bind_partition_observation_hooks(
                        self.executable.erased().partition_observation_hooks(),
                    )
                })
                .map_err(|error| Error::Other(Box::new(error)))?,
        );
        Ok(self)
    }

    pub(crate) fn take_capture_discovery(
        &mut self,
    ) -> Option<eredu_architectures::prepared_sources::PreparedModelDiscovery> {
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
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    > {
        self.executable.parameter_bank_report()
    }
}
