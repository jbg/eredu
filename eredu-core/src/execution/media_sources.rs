//! Independently selected input-tower artifacts.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Explicit component sources, validated against the target during cold preparation.
/// Image and video share one vision tower; waveform input uses the audio tower.
#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct MediaCompanionSources {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    vision: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    audio: Option<PathBuf>,
}

impl MediaCompanionSources {
    /// Selects the image/video source directory.
    pub fn with_vision(mut self, path: impl Into<PathBuf>) -> Self {
        self.vision = Some(path.into());
        self
    }

    /// Selects the waveform source directory independently of vision.
    pub fn with_audio(mut self, path: impl Into<PathBuf>) -> Self {
        self.audio = Some(path.into());
        self
    }

    /// Explicit image/video source, if selected.
    pub fn vision(&self) -> Option<&Path> {
        self.vision.as_deref()
    }

    /// Explicit waveform source, if selected.
    pub fn audio(&self) -> Option<&Path> {
        self.audio.as_deref()
    }

    /// Whether both components use their primary artifact's sources.
    pub fn is_empty(&self) -> bool {
        self.vision.is_none() && self.audio.is_none()
    }
}
