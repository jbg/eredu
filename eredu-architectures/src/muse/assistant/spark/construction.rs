//! Exact physical encodings and target-bound construction for both Spark assistants.

use std::collections::HashMap;

use eredu_checkpoint::{schema::matrix_for_linear_format, LinearFormat, WeightQuantization};
use eredu_nn::{DistributedNeuralBackend, Error, GroupedNeuralBackend, Tensor};

use super::{DecoderConfig, SparkAssistantConfig, SparkAssistantError, SparkAssistantIdentity};
use crate::muse::assistant::{
    body::{DFlashBodyDefinition, DsparkBodyDefinition},
    candidate_selector::CandidateSelectorDefinition,
    dspark::DsparkHeadDefinition,
    execution::MuseAssistantDefinition,
    integration::MuseAssistantModule,
};

/// Admitted assistant equations and exact per-weight physical encodings.
/// Unlisted weights retain the released BF16 storage contract.
#[derive(Debug, Clone)]
pub struct SparkAssistantLoadConfig {
    released: SparkAssistantConfig,
    formats: HashMap<String, WeightQuantization>,
}

impl SparkAssistantLoadConfig {
    /// Retains the released dense source contract.
    pub fn new(released: SparkAssistantConfig) -> Self {
        Self {
            released,
            formats: HashMap::new(),
        }
    }

    /// Borrows the artifact's equations and identity independently of physical encoding.
    pub const fn released(&self) -> &SparkAssistantConfig {
        &self.released
    }

    /// Returns one primary matrix's selected physical encoding.
    pub fn linear_format(&self, name: &str) -> LinearFormat {
        self.formats
            .get(name)
            .copied()
            .map(Into::into)
            .unwrap_or(LinearFormat::Dense)
    }

    /// Replaces per-weight encodings for an explicit conversion of every matrix.
    pub fn load_time_quantization(
        &self,
        format: WeightQuantization,
    ) -> Result<Self, SparkAssistantError> {
        let formats = self
            .released
            .safetensors_plan()?
            .common_tensors
            .into_iter()
            .filter(|tensor| tensor.shape.len() == 2)
            .map(|tensor| (tensor.key, format))
            .collect();
        self.clone().with_checkpoint_formats(formats)
    }

    /// Selects exact matrix encodings, rejecting names outside the declared topology.
    /// Norms, signed mapping buffers and convolution taps cannot be encoded as matrices.
    pub fn with_checkpoint_formats(
        mut self,
        formats: HashMap<String, WeightQuantization>,
    ) -> Result<Self, SparkAssistantError> {
        let plan = self.released.safetensors_plan()?;
        for (name, format) in &formats {
            let tensor = plan
                .common_tensors
                .iter()
                .find(|tensor| tensor.key == *name)
                .filter(|tensor| tensor.shape.len() == 2)
                .ok_or_else(|| SparkAssistantError(format!("{name} is not an assistant matrix")))?;
            format
                .validate()
                .map_err(|e| SparkAssistantError(e.to_string()))?;
            let neural_format = crate::linear_format::matrix_linear_format(
                name,
                name.strip_suffix(".weight").unwrap_or(name),
                (*format).into(),
            )
            .map_err(|e| SparkAssistantError(e.to_string()))?;
            if name == "confidence_head.proj.weight" {
                neural_format
                    .with_compute_element(Some(eredu_nn::TensorElementType::F32))
                    .map_err(|e| SparkAssistantError(e.to_string()))?;
            }
            matrix_for_linear_format(
                name,
                tensor.aliases.clone(),
                tensor.shape.clone(),
                (*format).into(),
                None,
            )
            .map_err(|e| SparkAssistantError(e.to_string()))?;
        }
        self.formats = formats;
        Ok(self)
    }

    /// Builds the complete source catalog, including every selected quantization companion.
    pub fn safetensors_plan(
        &self,
    ) -> Result<eredu_checkpoint::schema::SafetensorsCheckpointPlan, SparkAssistantError> {
        let mut plan = self.released.safetensors_plan()?;
        let mut tensors = Vec::new();
        for tensor in plan.common_tensors {
            if let Some(format) = self.formats.get(&tensor.key) {
                tensors.extend(
                    matrix_for_linear_format(
                        tensor.key,
                        tensor.aliases,
                        tensor.shape,
                        (*format).into(),
                        None,
                    )
                    .map_err(|e| SparkAssistantError(e.to_string()))?,
                );
            } else {
                tensors.push(tensor);
            }
        }
        plan.common_tensors = tensors;
        Ok(plan)
    }

    /// Proves the target relationship and freezes complete neural geometry before binding.
    pub fn construction(
        &self,
        target: &DecoderConfig,
    ) -> Result<SparkAssistantConstruction, SparkAssistantError> {
        self.released.prove_compatibility(target)?;
        let decoder = match self.released.identity() {
            SparkAssistantIdentity::DSpark => {
                let mut decoder = self.released.dspark_decoder_config(target)?;
                decoder.quantized_weight_configs = Some(self.formats.clone());
                SparkAssistantDecoderConstruction::DSpark(decoder)
            }
            SparkAssistantIdentity::DFlash2 => {
                let mut decoder = self.released.dflash_decoder_config()?;
                decoder.quantized_weights = self.formats.clone();
                SparkAssistantDecoderConstruction::DFlash2(decoder)
            }
        };
        Ok(SparkAssistantConstruction {
            load: self.clone(),
            decoder,
        })
    }
}

#[derive(Debug, Clone)]
enum SparkAssistantDecoderConstruction {
    DSpark(DecoderConfig),
    DFlash2(crate::muse::assistant::DFlashDecoderConfig),
}

/// Complete target-proven geometry retained separately for source and selected encodings.
#[derive(Debug, Clone)]
pub struct SparkAssistantConstruction {
    load: SparkAssistantLoadConfig,
    decoder: SparkAssistantDecoderConstruction,
}

impl SparkAssistantConstruction {
    /// Freezes the complete body/head topology without native parameter construction.
    pub fn execution_definition(&self) -> Result<MuseAssistantDefinition, Error> {
        let config = &self.load.released;
        let formats = &self.load.formats;
        match &self.decoder {
            SparkAssistantDecoderConstruction::DSpark(decoder) => MuseAssistantDefinition::dspark(
                DsparkBodyDefinition::new(
                    decoder.clone(),
                    config.capture_layers().len(),
                    config.block_size(),
                )?,
                DsparkHeadDefinition::new(
                    config.dspark_head_geometry().map_err(Error::backend)?,
                    formats.clone(),
                )?,
            ),
            SparkAssistantDecoderConstruction::DFlash2(decoder) => {
                MuseAssistantDefinition::dflash2(
                    DFlashBodyDefinition::new(decoder.clone())?,
                    CandidateSelectorDefinition::new(
                        config
                            .candidate_selector_geometry()
                            .map_err(Error::backend)?,
                        formats.clone(),
                    )?,
                )
            }
        }
    }
    /// Constructs resident units through the same complete parameter-policy driver.
    pub fn module<B: GroupedNeuralBackend + DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MuseAssistantModule<B>, Error> {
        MuseAssistantModule::resident(self.execution_definition()?, context)
    }
}
