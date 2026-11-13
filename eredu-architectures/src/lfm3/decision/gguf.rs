//! Admission of published d1 GGUF weights and exact task sidecars.

use std::path::Path;

use eredu_checkpoint::{
    gguf_store::GgufCatalog,
    recipe::RecipeDtype,
    validation::{resolve_gguf_plan, ResolvedCheckpointPlan},
};
use eredu_gguf::{Checkpoint, MetadataValue};
use serde_json::Value;

use super::{
    is_d1_config, read, DecisionAudioProcessor, DecisionError, PreparedTextChoiceModel,
    TextChoicePreparation,
};
use crate::lfm3::vision::{
    gguf::{self, open_decoded},
    MediaTokens, VisionLanguageDeclaration,
};
use eredu_nn::Error;

const DECISION_TYPE: &str = "lfm3.decision.type";

pub(crate) fn declares_decision(checkpoint: &Checkpoint) -> bool {
    checkpoint.metadata().contains_key(DECISION_TYPE)
}

/// Physically validated d1 GGUF layouts without retained payload access.
pub struct DecisionGgufLayouts {
    /// Decoder tensor schema for the selected encoding.
    pub backbone: ResolvedCheckpointPlan,
    /// Optional input projector schema for the selected encoding.
    pub projector: Option<ResolvedCheckpointPlan>,
}

struct Sidecars {
    config: Value,
    preparation: TextChoicePreparation,
    processor: Option<Vec<u8>>,
    omni_tokens: Option<MediaTokens>,
    audio: Option<DecisionAudioProcessor>,
}

impl Sidecars {
    fn read(
        checkpoint: &Checkpoint,
        projector: bool,
        assets: &Path,
    ) -> Result<Self, DecisionError> {
        let config_bytes = read(assets, "config.json")?;
        let config: Value = serde_json::from_slice(&config_bytes)?;
        if !is_d1_config(&config) {
            return Err(DecisionError::Declaration("expected d1 GGUF sidecars"));
        }
        let preparation = TextChoicePreparation::from_sidecars(
            &config_bytes,
            &read(assets, "tokenizer.json")?,
            &read(assets, "tokenizer_config.json")?,
            &read(assets, "calibration.json")?,
        )?;
        let omni = config["model_type"] == "lfm3_omni";
        validate(checkpoint, &preparation, omni)?;
        let processor = (omni || projector)
            .then(|| read(assets, "processor_config.json"))
            .transpose()?;
        let (omni_tokens, audio) = if omni {
            let document: Value = serde_json::from_slice(processor.as_deref().ok_or(
                DecisionError::Declaration("d1 Omni processor metadata is required"),
            )?)?;
            let tokens =
                crate::lfm3::omni::vision_tokens(&config, &document, &preparation.tokenizer)?;
            let audio = projector
                .then(|| {
                    DecisionAudioProcessor::new(&config, &document, &preparation.tokenizer, tokens)
                })
                .transpose()?;
            (Some(tokens), audio)
        } else {
            (None, None)
        };
        Ok(Self {
            config,
            preparation,
            processor,
            omni_tokens,
            audio,
        })
    }
}

impl DecisionGgufLayouts {
    /// Validates published d1 layouts and media bindings from headers and sidecars.
    /// This does not retain weight access or validate tensor payload contents.
    pub fn inspect(
        backbone: &Checkpoint,
        projector: Option<&Checkpoint>,
        assets: &Path,
    ) -> Result<Self, DecisionError> {
        let sidecars = Sidecars::read(backbone, projector.is_some(), assets)?;
        let omni = sidecars.omni_tokens.is_some();
        let declaration = VisionLanguageDeclaration::from_checkpoint_config(
            &sidecars.config,
            "d1 GGUF",
            sidecars.omni_tokens,
        )?;
        let plan = gguf::decision_backbone_plan(backbone, &declaration.language, omni)?;
        let backbone = resolve_gguf_plan(backbone, &plan).map_err(|report| {
            Error::backend(format!("d1 GGUF backbone schema differs: {report:?}"))
        })?;
        let projector = projector
            .map(|checkpoint| {
                let processor = sidecars
                    .processor
                    .as_deref()
                    .ok_or(DecisionError::Declaration(
                        "d1 GGUF processor metadata is required",
                    ))?;
                declaration
                    .processor(processor)
                    .map_err(Error::backend_source)?;
                declaration.prompt_admission()?;
                let plan = gguf::decision_projector_plan(checkpoint, &declaration.vision, omni)?;
                let resolved = resolve_gguf_plan(checkpoint, &plan).map_err(|report| {
                    Error::backend(format!("d1 GGUF projector schema differs: {report:?}"))
                })?;
                let checkpoint = gguf::decoded_catalog(checkpoint).map_err(Error::backend)?;
                let mapping = checkpoint
                    .translated_outputs(str::to_owned)
                    .map_err(Error::backend_source)?;
                let catalog =
                    GgufCatalog::from_resolved_checkpoint(&checkpoint, &resolved, &mapping)
                        .map_err(Error::backend_source)?;
                gguf::projector_recipes(&catalog, &declaration.vision)?;
                if sidecars.audio.is_some() {
                    let encoder =
                        crate::lfm3::audio::Declaration::from_speech_config(&sidecars.config)?;
                    crate::lfm3::audio::gguf::input_encoder_recipes(&encoder, &catalog)
                        .map_err(Error::backend_source)?;
                }
                Ok::<_, DecisionError>(resolved)
            })
            .transpose()?;
        Ok(Self {
            backbone,
            projector,
        })
    }
}

impl PreparedTextChoiceModel {
    /// Opens a released d1 GGUF decoder and optional media projector.
    ///
    /// The assets directory contains the matching decision configuration, tokenizer,
    /// and calibration. The processor sidecar is required with a projector and for
    /// Omni marker admission. A supplied projector retains all input encoders.
    pub fn open_gguf(
        backbone: &Path,
        projector: Option<&Path>,
        assets: &Path,
        dtype: RecipeDtype,
        max_cached_readers: usize,
    ) -> Result<Self, DecisionError> {
        let checkpoint = open_decoded(backbone)?;
        let sidecars = Sidecars::read(&checkpoint, projector.is_some(), assets)?;
        let (decoder, vision, audio) = VisionLanguageDeclaration::prepare_decision_gguf(
            checkpoint,
            projector.zip(sidecars.processor.as_deref()),
            &sidecars.config,
            sidecars.omni_tokens,
            sidecars.audio,
            dtype,
            max_cached_readers,
        )?;
        Ok(Self {
            preparation: sidecars.preparation,
            decoder,
            vision,
            audio,
        })
    }
}

fn validate(
    checkpoint: &Checkpoint,
    preparation: &TextChoicePreparation,
    omni: bool,
) -> Result<(), DecisionError> {
    let metadata = checkpoint.metadata();
    let text = |key| metadata.get(key).and_then(MetadataValue::as_str);
    let (basename, task) = if omni {
        ("d1-omni", "lfm3-d1-omni")
    } else {
        ("d1", "lfm3-d1")
    };
    if text("general.architecture") != Some("lfm3")
        || text("general.type") != Some("model")
        || text("general.basename") != Some(basename)
        || text(DECISION_TYPE) != Some(task)
        || metadata
            .get("lfm3.attention.causal")
            .and_then(MetadataValue::as_bool)
            != Some(false)
    {
        return Err(DecisionError::Declaration(
            "expected released d1 GGUF for the supplied sidecars",
        ));
    }
    let tokens = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(MetadataValue::as_strings)
        .ok_or(DecisionError::Declaration("d1 GGUF tokenizer vocabulary"))?;
    let vocabulary = eredu_text::tokenizer::token_id_vocabulary(&preparation.tokenizer);
    if tokens.len() != preparation.vocab
        || vocabulary
            .iter()
            .any(|(&id, token)| tokens.get(id as usize) != Some(token))
        || metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(MetadataValue::as_i64)
            != preparation
                .tokenizer
                .token_to_id(&preparation.bos)
                .map(i64::from)
        || metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(MetadataValue::as_i64)
            != preparation
                .tokenizer
                .token_to_id(super::IM_END)
                .map(i64::from)
    {
        return Err(DecisionError::Declaration(
            "d1 GGUF tokenizer and sidecar disagree",
        ));
    }
    for bucket in ["noul", "choice", "score"] {
        let key = format!("lfm3.decision.temperature.{bucket}");
        if metadata.get(&key).and_then(MetadataValue::as_f32)
            != Some(preparation.calibration.temperature(bucket) as f32)
        {
            return Err(DecisionError::Declaration(
                "d1 GGUF calibration and sidecar disagree",
            ));
        }
    }
    Ok(())
}
