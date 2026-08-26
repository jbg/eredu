//! MLX model materialization options.

use eredu_checkpoint::WeightQuantization;

use crate::backend::error::Error;
use eredu_runtime::{PipelineWireContract, WeightResidency};

use super::MlxParallelContext;

/// Options for materializing model weights with MLX.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ModelLoadOptions {
    /// Optional MLX weight encoding requested during dense checkpoint loading.
    pub quantization: Option<WeightQuantization>,
    /// Validated runtime topology paired with its required wire contract.
    parallel: Option<(MlxParallelContext, PipelineWireContract)>,
    /// Parameter placement and execution policy for cataloged checkpoint stores.
    pub weight_residency: WeightResidency,
}

impl ModelLoadOptions {
    /// Creates load options that quantize eligible dense weights on load.
    pub fn with_quantization(quantization: impl Into<WeightQuantization>) -> Self {
        Self {
            quantization: Some(quantization.into()),
            parallel: None,
            weight_residency: WeightResidency::fully_resident(),
        }
    }

    /// Adds a validated MLX parallel topology and its activation wire contract.
    pub fn with_parallel_topology(
        mut self,
        topology: MlxParallelContext,
        pipeline_wire: PipelineWireContract,
    ) -> Self {
        self.parallel = Some((topology, pipeline_wire));
        self
    }

    /// Creates load options for a validated MLX parallel topology and
    /// activation wire contract.
    pub fn with_parallel(
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

    /// Returns the selected distributed topology, if any.
    pub const fn parallel_topology(self) -> Option<MlxParallelContext> {
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

    pub(crate) const fn parallel_execution(
        self,
    ) -> Option<(MlxParallelContext, PipelineWireContract)> {
        self.parallel
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

    pub fn preparation_policy(self) -> Result<eredu_core::PreparationPolicy, Error> {
        use eredu_core::{QuantizationRequest, ResidencyRequest};
        use eredu_runtime::LayerWeightResidency;

        if let Some(quantization) = self.quantization {
            quantization.validate()?;
        }
        let quantization = match self.quantization {
            Some(WeightQuantization::Affine(config)) => Some(QuantizationRequest::Affine {
                group_size: u32::try_from(config.group_size).map_err(|_| {
                    Error::Quantization(format!(
                        "group_size must be non-negative, got {}",
                        config.group_size
                    ))
                })?,
                bits: u8::try_from(config.bits).map_err(|_| {
                    Error::Quantization(format!("bits must fit in u8, got {}", config.bits))
                })?,
            }),
            Some(WeightQuantization::MxFp4) => Some(QuantizationRequest::MxFp4),
            Some(WeightQuantization::GgufIQuant { .. }) => {
                return Err(Error::Quantization(
                    "checkpoint-native GGML blocks describe GGUF storage and cannot be requested as a load-time quantization transform"
                        .into(),
                ));
            }
            None => None,
        };
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
            quantization,
            residency,
            topology: self.parallel_topology().map(MlxParallelContext::topology),
        })
    }
}

#[cfg(test)]
mod tests {
    use eredu_checkpoint::WeightQuantization;
    use eredu_gguf::{Endian, GgmlType};
    use eredu_runtime::LayerwiseLoadOptions;

    use super::{MlxParallelContext, ModelLoadOptions};
    use crate::backend::DeviceAssignment;
    use eredu_runtime::WeightResidency;

    #[test]
    fn preparation_policy_preserves_quantized_nonresident_request() {
        let options = ModelLoadOptions::with_quantization(WeightQuantization::MxFp4)
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
    fn preparation_policy_rejects_checkpoint_native_gguf_encoding_as_a_transform() {
        let error = ModelLoadOptions::with_quantization(WeightQuantization::GgufIQuant {
            ggml_type: GgmlType::Q4_0,
            endian: Endian::Little,
        })
        .preparation_policy()
        .unwrap_err();

        assert!(matches!(
            error,
            crate::backend::error::Error::Quantization(message)
                if message.contains("checkpoint-native GGML blocks")
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
