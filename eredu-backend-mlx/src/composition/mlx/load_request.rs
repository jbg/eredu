//! Composition-owned MLX preparation request.

use eredu_checkpoint::{AffineQuantization, WeightQuantization};
use eredu_core::QuantizationRequest;

use crate::backend::error::Error;
use eredu_runtime::{PipelineWireContract, WeightResidency};

/// Portable drafting intent applied before checkpoint payload selection.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub(crate) enum SpeculativeLoadRequest {
    /// Preserve direct-load behavior by installing an admitted embedded extension.
    #[default]
    ArchitectureDefault,
    /// Materialize only the ordinary target.
    Disabled,
    /// Require and install an embedded extension with this proposal bound.
    Embedded {
        max_draft_tokens: std::num::NonZeroUsize,
    },
    /// Prepare only the ordinary target for later external-assistant pairing.
    ExternalTarget,
}

/// Caller request translated into an authoritative realization before materialization.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct MlxLoadRequest {
    /// Optional weight transformation requested during dense checkpoint loading.
    pub(crate) quantization: Option<QuantizationRequest>,
    /// Validated runtime topology paired with its required wire contract.
    parallel: Option<(
        eredu_core::ParallelRankTopology,
        crate::backend::DeviceAssignment,
        PipelineWireContract,
    )>,
    /// Maximum invocation geometry used to resolve architecture-owned wires.
    partitioned_invocation_limits: Option<(i32, i32)>,
    /// Caller-selected bounded completion policy for native communication.
    communication_completion: Option<eredu_runtime::CommunicationCompletionPolicy>,
    /// Parameter placement and execution policy for cataloged checkpoint stores.
    pub(crate) weight_residency: WeightResidency,
    /// Exact mutable-state residency and paging controls.
    pub(crate) state_residency: eredu_runtime::CacheResidencyPolicy,
    /// Capabilities required from the exact realized model session.
    pub(crate) required_session_capabilities: eredu_core::SessionCapabilities,
    /// Drafting intent selected by portable planning before payload work.
    speculative: SpeculativeLoadRequest,
}

impl MlxLoadRequest {
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
            quantization: Some(quantization),
            parallel: None,
            partitioned_invocation_limits: None,
            communication_completion: None,
            weight_residency: WeightResidency::fully_resident(),
            state_residency: eredu_runtime::CacheResidencyPolicy::Device,
            required_session_capabilities: eredu_core::SessionCapabilities::default(),
            speculative: SpeculativeLoadRequest::ArchitectureDefault,
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
    ) -> Self {
        self.parallel = Some((topology, device, pipeline_wire));
        self.partitioned_invocation_limits = Some((maximum_batch_size, maximum_sequence_length));
        self.communication_completion = Some(completion_policy);
        self
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
    ) -> Self {
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
        self.communication_completion = Some(policy);
        self
    }

    /// Selects fully resident or bounded layer execution for checkpoint weights.
    pub fn with_weight_residency(mut self, residency: WeightResidency) -> Self {
        self.weight_residency = residency;
        self
    }

    /// Selects the exact mutable-state residency and paging controls.
    pub fn with_state_residency(mut self, residency: eredu_runtime::CacheResidencyPolicy) -> Self {
        self.state_residency = residency;
        self
    }

    /// Requires capabilities from the exact inspected and realized session.
    pub fn with_required_session_capabilities(
        mut self,
        capabilities: eredu_core::SessionCapabilities,
    ) -> Self {
        self.required_session_capabilities = capabilities;
        self
    }

    /// Applies the execution plan's drafting mode before target payload selection.
    pub(crate) fn with_drafting_plan(
        mut self,
        plan: &eredu_core::DraftingPlan,
    ) -> Result<Self, Error> {
        self.speculative = match plan {
            eredu_core::DraftingPlan::Disabled => SpeculativeLoadRequest::Disabled,
            eredu_core::DraftingPlan::Embedded {
                max_draft_tokens, ..
            } => SpeculativeLoadRequest::Embedded {
                max_draft_tokens: std::num::NonZeroUsize::new(*max_draft_tokens).ok_or_else(
                    || Error::AutomaticPlanning("embedded draft capacity must be positive".into()),
                )?,
            },
            eredu_core::DraftingPlan::External { .. } => SpeculativeLoadRequest::ExternalTarget,
            _ => {
                return Err(Error::AutomaticPlanning(
                    "unsupported speculative drafting plan".into(),
                ))
            }
        };
        Ok(self)
    }

    /// Returns the pre-payload drafting intent.
    pub(crate) const fn speculative_load_request(&self) -> SpeculativeLoadRequest {
        self.speculative
    }

    /// Returns the selected distributed topology, if any.
    pub(crate) const fn parallel_topology(&self) -> Option<eredu_core::ParallelRankTopology> {
        match self.parallel {
            Some((topology, _, _)) => Some(topology),
            None => None,
        }
    }

    pub(crate) fn parallel_rank_context(
        &self,
    ) -> Result<Option<crate::backend::MlxRankContext>, Error> {
        self.parallel
            .map(|(topology, device, _)| {
                crate::backend::MlxRankContext::new(
                    topology.world_size(),
                    topology.global_rank(),
                    device,
                )
            })
            .transpose()
    }

    /// Returns the activation wire contract paired with the distributed
    /// topology, if any.
    pub const fn pipeline_wire_contract(&self) -> Option<PipelineWireContract> {
        match self.parallel {
            Some((_, _, wire_contract)) => Some(wire_contract),
            None => None,
        }
    }

    /// Reports whether composition attached a native parallel execution plan.
    pub const fn has_parallel_execution(&self) -> bool {
        self.parallel.is_some()
    }

    /// Returns the requested dense-weight transformation, if any.
    pub const fn quantization(&self) -> Option<QuantizationRequest> {
        self.quantization
    }

    /// Returns the selected immutable-weight residency policy.
    pub const fn weight_residency(&self) -> WeightResidency {
        self.weight_residency
    }

    /// Returns the selected mutable-state residency policy.
    pub const fn state_residency(&self) -> &eredu_runtime::CacheResidencyPolicy {
        &self.state_residency
    }

    /// Returns the capabilities required from the realized session.
    pub const fn required_session_capabilities(&self) -> eredu_core::SessionCapabilities {
        self.required_session_capabilities
    }

    /// Returns validated invocation limits for partitioned selection.
    pub(crate) fn partitioned_invocation_limits(&self) -> Result<Option<(i32, i32)>, Error> {
        match (self.parallel, self.partitioned_invocation_limits) {
            (None, None) => Ok(None),
            (Some(_), Some((maximum_batch_size, maximum_sequence_length)))
                if maximum_batch_size > 0 && maximum_sequence_length > 0 =>
            {
                Ok(Some((maximum_batch_size, maximum_sequence_length)))
            }
            (Some(_), Some((maximum_batch_size, maximum_sequence_length))) => {
                Err(Error::Parallel(format!(
                    "partitioned invocation limits must be positive, got batch {maximum_batch_size} and sequence {maximum_sequence_length}"
                )))
            }
            (Some(_), None) => Err(Error::Parallel(
                "parallel execution requires explicit maximum batch and sequence limits".into(),
            )),
            (None, Some(_)) => Err(Error::Parallel(
                "partitioned invocation limits require a parallel topology".into(),
            )),
        }
    }

    pub(crate) fn communication_completion_policy(
        &self,
    ) -> Result<Option<eredu_runtime::CommunicationCompletionPolicy>, Error> {
        match (self.parallel.is_some(), self.communication_completion) {
            (true, Some(policy)) => Ok(Some(policy)),
            (true, None) => Err(Error::Parallel(
                "parallel execution requires an explicit bounded communication completion policy"
                    .into(),
            )),
            (false, None) => Ok(None),
            (false, Some(_)) => Err(Error::Parallel(
                "a communication completion policy requires a parallel topology".into(),
            )),
        }
    }

    /// Completion policy selected for realtime work, including local async evaluation.
    pub(crate) fn realtime_completion_policy(
        &self,
    ) -> Result<eredu_runtime::CommunicationCompletionPolicy, Error> {
        if self.parallel.is_some() {
            return self.communication_completion_policy()?.ok_or_else(|| {
                Error::Parallel("parallel realtime completion policy is missing".into())
            });
        }
        self.communication_completion.map_or_else(
            || {
                eredu_runtime::CommunicationCompletionPolicy::new(
                    std::time::Duration::from_secs(30),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .map_err(|error| Error::Parallel(error.to_string()))
            },
            Ok,
        )
    }

    pub(crate) fn weight_quantization(&self) -> Result<Option<WeightQuantization>, Error> {
        self.quantization
            .map(|request| match request {
                QuantizationRequest::Affine { group_size, bits } => {
                    let group_size = i32::try_from(group_size).map_err(|_| {
                        Error::Quantization(format!("group_size must fit in i32, got {group_size}"))
                    })?;
                    Ok(WeightQuantization::Affine(AffineQuantization::new(
                        group_size,
                        i32::from(bits),
                    )?))
                }
                QuantizationRequest::MxFp4 => Ok(WeightQuantization::MxFp4),
                _ => Err(Error::Quantization(
                    "unknown load-time transformation request".into(),
                )),
            })
            .transpose()
    }

    /// Converts these MLX load options into the portable preparation policy.
    pub fn preparation_policy(&self) -> Result<eredu_core::PreparationPolicy, Error> {
        use eredu_core::ResidencyRequest;
        use eredu_runtime::LayerWeightResidency;

        self.weight_quantization()?;
        self.partitioned_invocation_limits()?;
        self.communication_completion_policy()?;
        let residency = if self.weight_residency.parameter_bank_cache().is_some() {
            ResidencyRequest::AddressableParameterBanks
        } else {
            match self.weight_residency.layers() {
                LayerWeightResidency::FullyResident => ResidencyRequest::FullyResident,
                LayerWeightResidency::LayerwiseHost(_) => ResidencyRequest::LayerwiseHost,
                LayerWeightResidency::DenseDiskStream(_) => ResidencyRequest::DenseDiskStream,
                _ => {
                    return Err(Error::Parallel(
                        "unsupported additive layer residency policy".into(),
                    ));
                }
            }
        };
        let mut policy = eredu_core::PreparationPolicy::new(self.quantization, residency)
            .with_required_session_capabilities(self.required_session_capabilities);
        if let Some(topology) = self.parallel_topology() {
            policy = policy.with_topology(topology.topology());
        }
        Ok(policy)
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
        .preparation_policy()
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
            disabled.speculative_load_request(),
            super::SpeculativeLoadRequest::Disabled
        );

        let embedded = MlxLoadRequest::default()
            .with_drafting_plan(&DraftingPlan::Embedded {
                max_draft_tokens: 3,
                lookahead: false,
                adaptive_lookahead: false,
            })
            .unwrap();
        assert_eq!(
            embedded.speculative_load_request(),
            super::SpeculativeLoadRequest::Embedded {
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
}
