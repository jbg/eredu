//! MLX model materialization options.

use eredu_checkpoint::WeightQuantization;

use crate::backend::error::Error;
use eredu_runtime::WeightResidency;

use super::MlxParallelContext;

/// Options for materializing model weights with MLX.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ModelLoadOptions {
    /// Optional MLX weight encoding requested during dense checkpoint loading.
    pub quantization: Option<WeightQuantization>,
    /// Optional validated runtime topology and process-local device assignment.
    pub parallel: Option<MlxParallelContext>,
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

    /// Adds a validated MLX parallel topology to these options.
    pub fn with_parallel_topology(mut self, topology: MlxParallelContext) -> Self {
        self.parallel = Some(topology);
        self
    }

    /// Creates load options for a validated MLX parallel topology.
    pub fn with_parallel(topology: MlxParallelContext) -> Self {
        Self::default().with_parallel_topology(topology)
    }

    /// Selects fully resident or bounded layer execution for checkpoint weights.
    pub fn with_weight_residency(mut self, residency: WeightResidency) -> Self {
        self.weight_residency = residency;
        self
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
            Some(WeightQuantization::GgufIQuant { .. }) | None => None,
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
            topology: self.parallel.map(MlxParallelContext::topology),
        })
    }
}

pub fn ensure_replicated_load_options(options: ModelLoadOptions) -> Result<(), Error> {
    if options
        .parallel
        .is_some_and(|topology| !topology.is_replicated())
    {
        return Err(Error::Parallel(
            "this facade owns replicated runtime state; load through a distributed MlxBackend and create its MlxModelSession"
                .into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use eredu_checkpoint::WeightQuantization;
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
    fn preparation_policy_preserves_exact_parallel_topology() {
        let topology = MlxParallelContext::for_rank(
            5,
            2,
            3,
            2,
            DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
        )
        .unwrap();
        let policy = ModelLoadOptions::with_parallel(topology)
            .preparation_policy()
            .unwrap();
        assert_eq!(policy.topology, Some(topology.topology()));
    }

    #[test]
    fn preparation_policies_distinguish_parallel_axes() {
        let device = DeviceAssignment::new(safemlx::DeviceType::Cpu, 0);
        let tensor_pipeline = MlxParallelContext::for_rank(0, 2, 3, 1, device).unwrap();
        let tensor_expert = MlxParallelContext::for_rank(0, 2, 1, 3, device).unwrap();
        let tensor_pipeline_policy = ModelLoadOptions::with_parallel(tensor_pipeline)
            .preparation_policy()
            .unwrap();
        let tensor_expert_policy = ModelLoadOptions::with_parallel(tensor_expert)
            .preparation_policy()
            .unwrap();

        assert_ne!(tensor_pipeline_policy, tensor_expert_policy);
    }
}
