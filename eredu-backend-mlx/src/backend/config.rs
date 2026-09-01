//! MLX model materialization options.

use eredu_checkpoint::{AffineQuantization, WeightQuantization};
use eredu_core::QuantizationRequest;

use crate::backend::error::Error;
use eredu_runtime::{PipelineWireContract, WeightResidency};

use super::MlxParallelContext;

/// Options for materializing model weights with MLX.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ModelLoadOptions {
    /// Optional weight transformation requested during dense checkpoint loading.
    pub(crate) quantization: Option<QuantizationRequest>,
    /// Validated runtime topology paired with its required wire contract.
    parallel: Option<(MlxParallelContext, PipelineWireContract)>,
    /// Parameter placement and execution policy for cataloged checkpoint stores.
    pub(crate) weight_residency: WeightResidency,
    /// Capabilities required from the exact realized model session.
    pub(crate) required_session_capabilities: eredu_core::SessionCapabilities,
}

impl ModelLoadOptions {
    /// Creates load options that quantize eligible dense weights on load.
    pub fn with_quantization(quantization: QuantizationRequest) -> Self {
        Self {
            quantization: Some(quantization),
            parallel: None,
            weight_residency: WeightResidency::fully_resident(),
            required_session_capabilities: eredu_core::SessionCapabilities::default(),
        }
    }

    /// Adds a validated MLX parallel topology and its activation wire contract.
    pub(crate) fn with_parallel_topology(
        mut self,
        topology: MlxParallelContext,
        pipeline_wire: PipelineWireContract,
    ) -> Self {
        self.parallel = Some((topology, pipeline_wire));
        self
    }

    /// Creates load options for a validated MLX parallel topology and
    /// activation wire contract.
    pub(crate) fn with_parallel(
        topology: MlxParallelContext,
        pipeline_wire: PipelineWireContract,
    ) -> Self {
        Self::default().with_parallel_topology(topology, pipeline_wire)
    }

    /// Selects fully resident or bounded layer execution for checkpoint weights.
    pub fn with_weight_residency(mut self, residency: WeightResidency) -> Self {
        self.weight_residency = residency;
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

    /// Returns the selected distributed topology, if any.
    pub(crate) const fn parallel_topology(self) -> Option<MlxParallelContext> {
        match self.parallel {
            Some((topology, _)) => Some(topology),
            None => None,
        }
    }

    /// Returns the activation wire contract paired with the distributed
    /// topology, if any.
    pub const fn pipeline_wire_contract(self) -> Option<PipelineWireContract> {
        match self.parallel {
            Some((_, wire_contract)) => Some(wire_contract),
            None => None,
        }
    }

    /// Reports whether composition attached a native parallel execution plan.
    pub const fn has_parallel_execution(self) -> bool {
        self.parallel.is_some()
    }

    /// Returns the requested dense-weight transformation, if any.
    pub const fn quantization(self) -> Option<QuantizationRequest> {
        self.quantization
    }

    /// Returns the selected immutable-weight residency policy.
    pub const fn weight_residency(self) -> WeightResidency {
        self.weight_residency
    }

    /// Returns the capabilities required from the realized session.
    pub const fn required_session_capabilities(self) -> eredu_core::SessionCapabilities {
        self.required_session_capabilities
    }

    pub(crate) const fn parallel_execution(
        self,
    ) -> Option<(MlxParallelContext, PipelineWireContract)> {
        self.parallel
    }

    pub(crate) fn weight_quantization(self) -> Result<Option<WeightQuantization>, Error> {
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
            })
            .transpose()
    }

    pub(crate) fn validate_replicated(self) -> Result<(), Error> {
        if self
            .parallel_topology()
            .is_some_and(|topology| !topology.is_replicated())
        {
            return Err(Error::Parallel(
                "replicated MLX model loading requires a replicated topology; construct a distributed MlxBackend and MlxModelSession for partitioned execution"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Converts these MLX load options into the portable preparation policy.
    pub fn preparation_policy(self) -> Result<eredu_core::PreparationPolicy, Error> {
        use eredu_core::ResidencyRequest;
        use eredu_runtime::LayerWeightResidency;

        self.weight_quantization()?;
        let residency = if self.weight_residency.expert_cache().is_some() {
            ResidencyRequest::ExpertCache
        } else {
            match self.weight_residency.layers() {
                LayerWeightResidency::FullyResident => ResidencyRequest::FullyResident,
                LayerWeightResidency::LayerwiseHost(_) => ResidencyRequest::LayerwiseHost,
                LayerWeightResidency::DenseDiskStream(_) => ResidencyRequest::DenseDiskStream,
            }
        };
        Ok(eredu_core::PreparationPolicy {
            quantization: self.quantization,
            residency,
            topology: self.parallel_topology().map(MlxParallelContext::topology),
            required_session_capabilities: self.required_session_capabilities,
        })
    }
}

#[cfg(test)]
mod tests {
    use eredu_core::QuantizationRequest;
    use eredu_runtime::LayerwiseLoadOptions;

    use super::{MlxParallelContext, ModelLoadOptions};
    use crate::backend::DeviceAssignment;
    use eredu_runtime::WeightResidency;

    #[test]
    fn preparation_policy_preserves_quantized_nonresident_request() {
        let options = ModelLoadOptions::with_quantization(QuantizationRequest::MxFp4)
            .with_weight_residency(WeightResidency::layerwise_host(
                LayerwiseLoadOptions::default(),
            ));
        let policy = options.preparation_policy().unwrap();
        assert_eq!(
            policy.quantization,
            Some(eredu_core::QuantizationRequest::MxFp4)
        );
        assert_eq!(
            policy.residency,
            eredu_core::ResidencyRequest::LayerwiseHost
        );
    }

    #[test]
    fn preparation_policy_rejects_invalid_affine_geometry() {
        let error = ModelLoadOptions::with_quantization(QuantizationRequest::Affine {
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
        let topology = MlxParallelContext::for_rank(
            5,
            2,
            3,
            2,
            DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();
        let policy = ModelLoadOptions::with_parallel(
            topology,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
        )
        .preparation_policy()
        .unwrap();
        assert_eq!(policy.topology, Some(topology.topology()));
    }

    #[test]
    fn preparation_policies_distinguish_parallel_axes() {
        let device = DeviceAssignment::new(safemlx::DeviceType::Cpu, 0);
        let tensor_pipeline = MlxParallelContext::for_rank(0, 2, 3, 1, device).unwrap();
        let tensor_expert = MlxParallelContext::for_rank(0, 2, 1, 3, device).unwrap();
        let tensor_pipeline_policy = ModelLoadOptions::with_parallel(
            tensor_pipeline,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
        )
        .preparation_policy()
        .unwrap();
        let tensor_expert_policy = ModelLoadOptions::with_parallel(
            tensor_expert,
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
        )
        .preparation_policy()
        .unwrap();

        assert_ne!(tensor_pipeline_policy, tensor_expert_policy);
    }
}
