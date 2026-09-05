//! MLX adapter over the normalized portable preparation request.

use eredu_core::QuantizationRequest;

use crate::backend::error::Error;
use eredu_runtime::{
    NormalizedLoadRequest, NormalizedLoadRequestError, ParallelLoadRequest, PipelineWireContract,
    WeightResidency,
};

/// Caller request translated into an authoritative realization before materialization.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct MlxLoadRequest {
    /// Complete backend-neutral policy supplied to architecture selection.
    normalized: NormalizedLoadRequest,
    /// Optional native device paired with the normalized portable rank.
    parallel_device: Option<crate::backend::DeviceAssignment>,
}

impl MlxLoadRequest {
    /// Wraps one complete portable request without adding a native target.
    pub fn from_normalized(normalized: NormalizedLoadRequest) -> Self {
        Self {
            normalized,
            parallel_device: None,
        }
    }

    /// Returns the singular portable policy wrapped by this native adapter.
    pub const fn normalized(&self) -> &NormalizedLoadRequest {
        &self.normalized
    }

    #[cfg(test)]
    pub(crate) fn test_communication_completion_policy(
    ) -> eredu_runtime::CommunicationCompletionPolicy {
        eredu_runtime::CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(30),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .expect("test completion policy is positive")
    }

    /// Creates load options that quantize eligible dense weights on load.
    pub fn with_quantization(quantization: QuantizationRequest) -> Self {
        Self {
            normalized: NormalizedLoadRequest::with_quantization(quantization),
            parallel_device: None,
        }
    }

    /// Adds a validated MLX parallel topology and its activation wire contract.
    pub(crate) fn with_parallel_topology(
        mut self,
        topology: eredu_core::ParallelRankTopology,
        device: crate::backend::DeviceAssignment,
        pipeline_wire: PipelineWireContract,
        maximum_batch_size: i32,
        maximum_sequence_length: i32,
        completion_policy: eredu_runtime::CommunicationCompletionPolicy,
    ) -> Result<Self, Error> {
        let parallel = ParallelLoadRequest::new(
            topology,
            pipeline_wire,
            maximum_batch_size,
            maximum_sequence_length,
            completion_policy,
        )
        .map_err(normalized_request_error)?;
        self.normalized = self
            .normalized
            .with_parallel_execution(parallel)
            .map_err(normalized_request_error)?;
        self.parallel_device = Some(device);
        Ok(self)
    }

    /// Creates load options for a validated MLX parallel topology and
    /// activation wire contract.
    pub(crate) fn with_parallel(
        topology: eredu_core::ParallelRankTopology,
        device: crate::backend::DeviceAssignment,
        pipeline_wire: PipelineWireContract,
        maximum_batch_size: i32,
        maximum_sequence_length: i32,
        completion_policy: eredu_runtime::CommunicationCompletionPolicy,
    ) -> Result<Self, Error> {
        Self::default().with_parallel_topology(
            topology,
            device,
            pipeline_wire,
            maximum_batch_size,
            maximum_sequence_length,
            completion_policy,
        )
    }

    /// Selects the bounded completion policy for every native communication
    /// operation in the distributed session.
    pub const fn with_communication_completion_policy(
        mut self,
        policy: eredu_runtime::CommunicationCompletionPolicy,
    ) -> Self {
        self.normalized.set_communication_completion_policy(policy);
        self
    }

    /// Selects fully resident or bounded layer execution for checkpoint weights.
    pub fn with_weight_residency(mut self, residency: WeightResidency) -> Self {
        self.normalized = self.normalized.with_weight_residency(residency);
        self
    }

    /// Selects the exact mutable-state residency and paging controls.
    pub fn with_state_residency(mut self, residency: eredu_runtime::CacheResidencyPolicy) -> Self {
        self.normalized = self.normalized.with_state_residency(residency);
        self
    }

    /// Requires capabilities from the exact inspected and realized session.
    pub fn with_required_session_capabilities(
        mut self,
        capabilities: eredu_core::SessionCapabilities,
    ) -> Self {
        self.normalized = self
            .normalized
            .with_required_session_capabilities(capabilities);
        self
    }

    /// Applies the execution plan's drafting mode before target payload selection.
    pub(crate) fn with_drafting_plan(
        mut self,
        plan: &eredu_core::DraftingPlan,
    ) -> Result<Self, Error> {
        self.normalized = self
            .normalized
            .with_drafting_plan(plan)
            .map_err(normalized_request_error)?;
        Ok(self)
    }

    /// Returns the selected distributed topology, if any.
    pub(crate) const fn parallel_topology(&self) -> Option<eredu_core::ParallelRankTopology> {
        self.normalized.parallel_topology()
    }

    pub(crate) fn parallel_rank_context(
        &self,
    ) -> Result<Option<crate::backend::MlxRankContext>, Error> {
        self.checked_normalized().map(|(_, rank)| rank)
    }

    /// Returns the normalized request atomically paired with its validated MLX rank token.
    pub(crate) fn checked_normalized(
        &self,
    ) -> Result<
        (
            &NormalizedLoadRequest,
            Option<crate::backend::MlxRankContext>,
        ),
        Error,
    > {
        match (self.normalized.parallel_topology(), self.parallel_device) {
            (Some(topology), Some(device)) => crate::backend::MlxRankContext::new(
                topology.world_size(),
                topology.global_rank(),
                device,
            )
            .map(|rank| (&self.normalized, Some(rank))),
            (None, None) => Ok((&self.normalized, None)),
            (Some(_), None) => Err(Error::Parallel(
                "parallel execution has no MLX device assignment".into(),
            )),
            (None, Some(_)) => Err(Error::Parallel(
                "MLX device assignment requires a parallel topology".into(),
            )),
        }
    }

    /// Returns the activation wire contract paired with the distributed
    /// topology, if any.
    pub const fn pipeline_wire_contract(&self) -> Option<PipelineWireContract> {
        self.normalized.pipeline_wire_contract()
    }

    /// Reports whether composition attached a native parallel execution plan.
    pub const fn has_parallel_execution(&self) -> bool {
        self.normalized.has_parallel_execution()
    }

    /// Returns the requested dense-weight transformation, if any.
    pub const fn quantization(&self) -> Option<QuantizationRequest> {
        self.normalized.quantization()
    }

    /// Returns the selected immutable-weight residency policy.
    pub const fn weight_residency(&self) -> WeightResidency {
        self.normalized.weight_residency()
    }

    /// Returns the selected mutable-state residency policy.
    pub const fn state_residency(&self) -> &eredu_runtime::CacheResidencyPolicy {
        self.normalized.state_residency()
    }

    /// Returns the capabilities required from the realized session.
    pub const fn required_session_capabilities(&self) -> eredu_core::SessionCapabilities {
        self.normalized.required_session_capabilities()
    }

    /// Converts these MLX load options into the portable preparation policy.
    pub fn preparation_policy(&self) -> Result<eredu_core::PreparationPolicy, Error> {
        self.checked_normalized()?
            .0
            .preparation_policy()
            .map_err(normalized_request_error)
    }
}

fn normalized_request_error(error: NormalizedLoadRequestError) -> Error {
    let message = error.to_string();
    match error {
        NormalizedLoadRequestError::Quantization(message) => Error::Quantization(message),
        NormalizedLoadRequestError::ZeroEmbeddedDraftCapacity
        | NormalizedLoadRequestError::UnsupportedDraftingPlan => Error::AutomaticPlanning(message),
        _ => Error::Parallel(message),
    }
}

#[cfg(test)]
mod tests {
    use eredu_core::{DraftingPlan, ParallelRankTopology, ParallelTopology, QuantizationRequest};
    use eredu_runtime::LayerwiseLoadOptions;

    use super::MlxLoadRequest;
    use crate::backend::DeviceAssignment;
    use eredu_runtime::WeightResidency;

    fn topology(rank: usize, tp: usize, pp: usize, ep: usize) -> ParallelRankTopology {
        ParallelRankTopology::new(ParallelTopology::new(tp, pp, ep, 1).unwrap(), rank).unwrap()
    }

    #[test]
    fn preparation_policy_preserves_quantized_nonresident_request() {
        let options = MlxLoadRequest::with_quantization(QuantizationRequest::MxFp4)
            .with_weight_residency(WeightResidency::layerwise_host(
                LayerwiseLoadOptions::default(),
            ));
        let policy = options.preparation_policy().unwrap();
        assert_eq!(
            policy.quantization(),
            Some(eredu_core::QuantizationRequest::MxFp4)
        );
        assert_eq!(
            policy.residency(),
            eredu_core::ResidencyRequest::LayerwiseHost
        );
    }

    #[test]
    fn preparation_policy_rejects_invalid_affine_geometry() {
        let error = MlxLoadRequest::with_quantization(QuantizationRequest::Affine {
            group_size: 17,
            bits: 4,
        })
        .preparation_policy()
        .unwrap_err();

        assert!(matches!(
            error,
            crate::backend::error::Error::Quantization(message)
                if message.contains("group_size")
        ));
    }

    #[test]
    fn preparation_policy_preserves_exact_parallel_topology() {
        let topology = topology(5, 2, 3, 2);
        let device = DeviceAssignment::new(safemlx::DeviceType::Cpu, 0);
        let policy = MlxLoadRequest::with_parallel(
            topology,
            device,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            128,
            MlxLoadRequest::test_communication_completion_policy(),
        )
        .unwrap()
        .preparation_policy()
        .unwrap();
        assert_eq!(policy.topology(), Some(topology.topology()));
    }

    #[test]
    fn preparation_policies_distinguish_parallel_axes() {
        let device = DeviceAssignment::new(safemlx::DeviceType::Cpu, 0);
        let tensor_pipeline = topology(0, 2, 3, 1);
        let tensor_expert = topology(0, 2, 1, 3);
        let tensor_pipeline_policy = MlxLoadRequest::with_parallel(
            tensor_pipeline,
            device,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            128,
            MlxLoadRequest::test_communication_completion_policy(),
        )
        .unwrap()
        .preparation_policy()
        .unwrap();
        let tensor_expert_policy = MlxLoadRequest::with_parallel(
            tensor_expert,
            device,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            128,
            MlxLoadRequest::test_communication_completion_policy(),
        )
        .unwrap()
        .preparation_policy()
        .unwrap();

        assert_ne!(tensor_pipeline_policy, tensor_expert_policy);
    }

    #[test]
    fn parallel_policy_rejects_nonpositive_invocation_limits() {
        let topology = topology(0, 2, 1, 1);
        let device = DeviceAssignment::new(safemlx::DeviceType::Cpu, 0);
        let error = MlxLoadRequest::with_parallel(
            topology,
            device,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            0,
            128,
            MlxLoadRequest::test_communication_completion_policy(),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("partitioned invocation limits must be positive"));
    }

    #[test]
    fn portable_drafting_plan_is_fixed_before_payload_selection() {
        let disabled = MlxLoadRequest::default()
            .with_drafting_plan(&DraftingPlan::Disabled)
            .unwrap();
        assert_eq!(
            disabled.checked_normalized().unwrap().0.drafting(),
            eredu_runtime::DraftingLoadRequest::Disabled
        );

        let embedded = MlxLoadRequest::default()
            .with_drafting_plan(&DraftingPlan::Embedded {
                max_draft_tokens: 3,
                lookahead: false,
                adaptive_lookahead: false,
            })
            .unwrap();
        assert_eq!(
            embedded.checked_normalized().unwrap().0.drafting(),
            eredu_runtime::DraftingLoadRequest::Embedded {
                max_draft_tokens: std::num::NonZeroUsize::new(3).unwrap()
            }
        );

        assert!(MlxLoadRequest::default()
            .with_drafting_plan(&DraftingPlan::Embedded {
                max_draft_tokens: 0,
                lookahead: false,
                adaptive_lookahead: false,
            })
            .is_err());
    }

    #[test]
    fn adapter_translation_preserves_the_exact_normalized_request() {
        let capabilities = eredu_core::SessionCapabilities::new(true, false, true);
        let residency = WeightResidency::layerwise_host(LayerwiseLoadOptions::default());
        let options = MlxLoadRequest::with_quantization(QuantizationRequest::MxFp4)
            .with_weight_residency(residency)
            .with_required_session_capabilities(capabilities)
            .with_drafting_plan(&DraftingPlan::Disabled)
            .unwrap();
        let expected =
            eredu_runtime::NormalizedLoadRequest::with_quantization(QuantizationRequest::MxFp4)
                .with_weight_residency(residency)
                .with_required_session_capabilities(capabilities)
                .with_drafting(eredu_runtime::DraftingLoadRequest::Disabled);

        let (normalized, rank) = options.checked_normalized().unwrap();
        assert_eq!(normalized, &expected);
        assert!(rank.is_none());
    }

    #[test]
    fn native_device_is_not_part_of_normalized_request_equality() {
        let topology = topology(0, 2, 1, 1);
        let completion = MlxLoadRequest::test_communication_completion_policy();
        let wire = eredu_runtime::PipelineWireContract::new(
            eredu_runtime::PipelineActivationDtype::Float32,
        );
        let cpu = MlxLoadRequest::with_parallel(
            topology,
            DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            wire,
            1,
            128,
            completion,
        )
        .unwrap();
        let gpu = MlxLoadRequest::with_parallel(
            topology,
            DeviceAssignment::new(safemlx::DeviceType::Gpu, 0),
            wire,
            1,
            128,
            completion,
        )
        .unwrap();

        let (cpu_normalized, cpu_rank) = cpu.checked_normalized().unwrap();
        let (gpu_normalized, gpu_rank) = gpu.checked_normalized().unwrap();
        assert_eq!(cpu_normalized, gpu_normalized);
        assert_ne!(cpu_rank, gpu_rank);
        assert_ne!(cpu, gpu);
    }

    #[test]
    fn checked_translation_rejects_unpaired_parallel_halves() {
        let topology = topology(0, 2, 1, 1);
        let parallel = eredu_runtime::ParallelLoadRequest::new(
            topology,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            128,
            MlxLoadRequest::test_communication_completion_policy(),
        )
        .unwrap();
        let normalized = eredu_runtime::NormalizedLoadRequest::default()
            .with_parallel_execution(parallel)
            .unwrap();
        let missing_device = MlxLoadRequest {
            normalized,
            parallel_device: None,
        };
        assert!(missing_device.checked_normalized().is_err());

        let orphaned_device = MlxLoadRequest {
            normalized: eredu_runtime::NormalizedLoadRequest::default(),
            parallel_device: Some(DeviceAssignment::new(safemlx::DeviceType::Cpu, 0)),
        };
        assert!(orphaned_device.checked_normalized().is_err());
    }
}
