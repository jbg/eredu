//! Physical admission for the released Omni backbone and combined media projector.

use eredu_checkpoint::validation::ResolvedCheckpointPlan;
use eredu_gguf::Checkpoint;
use serde_json::Value;

use super::{vision_declaration, ArtifactError};
use crate::lfm3::{
    audio::{AdmittedAudioGguf, AudioGgufEncoding, Declaration, DepthDeclaration},
    vision,
};

/// An admitted Omni GGUF pair with the canonical component bindings checked.
pub struct AdmittedOmniGguf {
    pair: AdmittedAudioGguf,
}

impl AdmittedOmniGguf {
    /// Validates the released physical layouts and component recipes without
    /// reading payloads or granting new access to retained sources.
    pub fn from_checkpoints(
        backbone: Checkpoint,
        projector: Checkpoint,
        config: &Value,
        codec: &eredu_codec::liquid::Declaration,
    ) -> Result<Self, ArtifactError> {
        let vision = vision_declaration(config)?;
        let pair =
            AdmittedAudioGguf::from_multimodal_checkpoints(backbone, projector, Some(&vision))?;
        let speech = pair.speech_bindings(
            &Declaration::from_speech_config(config)?,
            &DepthDeclaration::from_speech_config(config)?,
        )?;
        speech.codec_decoder_bindings(codec)?;

        vision::gguf::projector_recipes(speech.projector_catalog(), &vision)?;
        Ok(Self { pair })
    }

    /// Resolved physical layouts in backbone and combined projector order.
    pub fn layouts(&self) -> (&ResolvedCheckpointPlan, &ResolvedCheckpointPlan) {
        self.pair.layouts()
    }

    /// Independently selected physical encodings.
    pub fn encodings(&self) -> (AudioGgufEncoding, AudioGgufEncoding) {
        self.pair.encodings()
    }
}
