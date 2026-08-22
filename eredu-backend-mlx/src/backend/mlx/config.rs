//! MLX model materialization options.

use eredu_checkpoint::WeightQuantization;

use crate::backend::mlx::error::Error;
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
            distributed: self
                .parallel
                .is_some_and(|topology| !topology.is_replicated()),
        })
    }

    pub fn validate_preparation(
        self,
        kind: eredu_core::ModelKind,
        format: eredu_core::ArtifactFormat,
    ) -> Result<eredu_core::MaterializationRoute, Error> {
        Ok(eredu_core::validate_preparation_policy(
            kind,
            format,
            self.preparation_policy()?,
        )?)
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

    use super::ModelLoadOptions;
    use eredu_runtime::WeightResidency;

    #[test]
    fn quantization_composes_with_nonresident_layers() {
        let options = ModelLoadOptions::with_quantization(WeightQuantization::MxFp4)
            .with_weight_residency(WeightResidency::layerwise_host(
                LayerwiseLoadOptions::default(),
            ));
        options
            .validate_preparation(
                eredu_core::ModelKind::GptOss,
                eredu_core::ArtifactFormat::Gguf,
            )
            .unwrap();
    }
}
