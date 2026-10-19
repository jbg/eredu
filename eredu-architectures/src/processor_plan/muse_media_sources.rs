//! Cold source selection and processor composition for shared Muse execution.

use super::*;
use crate::{
    artifact_preparation::{ArtifactSourceRole, PhysicalArtifactId},
    configuration::{GgufModelConfig, SafetensorsModelConfig},
    muse::media_companion::{Bf16MediaCompanion, MediaCompanionError},
};
use eredu_core::{ArtifactCompanionRole, MediaCompanionSources};
use std::sync::Arc;

impl ArtifactArchitecturePlan {
    fn muse_configuration(&self) -> Option<&crate::muse::DecoderConfig> {
        match &self.family {
            ArtifactFamilyPlan::Safetensors(plan) => match plan.model() {
                SafetensorsModelConfig::Muse(config) => Some(config),
                _ => None,
            },
            ArtifactFamilyPlan::Gguf(plan) => match (
                plan.model(),
                self.gguf_media_projector()
                    .map(|projector| projector.model()),
            ) {
                (_, Some(crate::gguf_companion::GgufMediaProjectorConfig::Muse(config))) => {
                    Some(config)
                }
                (GgufModelConfig::Muse(config), None) => Some(config),
                _ => None,
            },
        }
    }

    // A missing processor remains visible as a readiness problem; it does not
    // erase architecture-declared towers. A retained processor narrows framing.
    pub(crate) fn muse_input_modalities(&self) -> Option<eredu_core::InputModalities> {
        let config = self.muse_configuration()?;
        let vision = config.vision_config.is_some();
        Some(eredu_core::InputModalities {
            text: true,
            image: vision,
            audio: config.audio_config.is_some(),
            video: vision && self.muse().is_none_or(|processor| !processor.image_only),
        })
    }

    /// Optional source requirements do not prevent text-only preparation.
    pub(crate) fn missing_media_companions(&self) -> Vec<eredu_core::CapabilityError> {
        let Some(config) = self.muse_configuration() else {
            return Vec::new();
        };
        crate::capability::muse_missing_media_companions(config)
            .into_iter()
            .map(
                |(modality, component)| eredu_core::CapabilityError::MissingCompanion {
                    architecture: config.identity.text_model_type().into(),
                    modality,
                    component,
                },
            )
            .collect()
    }

    pub(crate) fn media_companions(&self) -> &[Arc<Bf16MediaCompanion>] {
        &self.media_companions
    }

    pub(crate) fn replaces_media_parameter(&self, key: &str) -> bool {
        crate::muse::checkpoint::media_source_role(key).is_some_and(|role| {
            self.media_companions
                .iter()
                .any(|source| source.roles().contains_key(&role))
        })
    }

    pub(crate) fn with_media_companions(
        mut self,
        sources: &MediaCompanionSources,
    ) -> Result<Self, MediaCompanionError> {
        if sources.is_empty() {
            return Ok(self);
        }
        let config = self
            .muse_configuration()
            .ok_or(MediaCompanionError::UnsupportedTarget)?;
        let admitted = Bf16MediaCompanion::admit_selected(config, sources)?
            .into_iter()
            .map(Arc::new)
            .collect::<Vec<_>>();
        let mut effective = config.clone();
        let mut processor = self.muse().cloned();
        for source in &admitted {
            let selected = source.configuration();
            if selected.vision_config.is_some() {
                effective.vision_config = selected.vision_config.clone();
                if let Some(current) = &mut processor {
                    current.image = source.processor().image.clone();
                    current.video = source.processor().video.clone();
                    current.image_only = source.processor().image_only;
                } else {
                    let mut selected_processor = source.processor().clone();
                    selected_processor.waveform = None;
                    processor = Some(selected_processor);
                }
            }
            if selected.audio_config.is_some() {
                effective.audio_config = selected.audio_config.clone();
                if let Some(current) = &mut processor {
                    current.waveform = source.processor().waveform;
                } else {
                    processor = Some(source.processor().clone());
                }
            }
        }
        // Sources provide component arithmetic and formats; the target retains
        // its decoder identity, encodings and exact original admission contract.
        effective.validate()?;
        if let Some(processor) = &processor {
            processor.validate_components(&effective)?;
        }
        if let Some(projector) = &mut self.media_projector {
            projector.replace_muse_execution_config(effective.clone());
        }
        match &mut self.family {
            ArtifactFamilyPlan::Safetensors(plan) => plan.replace_muse_execution_config(effective),
            ArtifactFamilyPlan::Gguf(plan) => plan.replace_muse_execution_config(effective),
        }
        self.processor = processor.map(NormalizedProcessorPlan::Muse);
        self.media_companions = admitted;
        self.validation = Default::default();
        Ok(self)
    }
}

pub(crate) fn physical_id(source: &Bf16MediaCompanion) -> PhysicalArtifactId {
    PhysicalArtifactId::Companion(
        if source.roles().contains_key(&ArtifactSourceRole::Vision) {
            ArtifactCompanionRole::Vision
        } else {
            ArtifactCompanionRole::Audio
        },
    )
}
