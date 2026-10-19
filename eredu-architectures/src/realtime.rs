//! Architecture-owned discovery and exact artifact preparation for realtime models.
//!
//! Configuration, tokenizer assets, checkpoint admission and source retention finish
//! before a backend is selected. Prepared artifacts never require network access.

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use crate::voicechat::{
    artifact::{
        AdmissionError, AdmittedVoiceChatSource, RetainedVoiceChatArtifacts, VoiceChatArtifactPlan,
    },
    assets::{
        PreparedVoiceChatAssets, VoiceChatAssetError, VoiceChatAssetRole, VOICECHAT_ASSET_MODEL,
    },
    checkpoint::{Component, GgufLayout},
    config::VoiceChatConfig,
};
use crate::{configuration::resolve_model_identity, moshi::MoshiPreparationPlan, ModelKind};

/// Physical model sources. Split containers retain explicit component ownership.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(tag = "format", rename_all = "snake_case", deny_unknown_fields)]
pub enum RealtimeArtifactSource {
    /// A SafeTensors file, indexed directory, or model artifact directory.
    #[serde(rename = "safetensors")]
    SafeTensors {
        /// Exact file or directory.
        path: PathBuf,
    },
    /// One complete GGUF container.
    Gguf {
        /// Exact complete container.
        path: PathBuf,
    },
    /// Published component GGUF containers and an explicit complete companion.
    SplitGguf {
        /// Perception encoder container.
        perception: PathBuf,
        /// Language backbone container.
        backbone: PathBuf,
        /// Function channel container.
        function_head: PathBuf,
        /// TTS and codec decoder container.
        tts: PathBuf,
        /// Complete source supplying absent RNNT and codec encoder components.
        companion: PathBuf,
    },
}

/// Local immutable auxiliary assets, resolved and retained during preparation.
#[derive(Debug, Clone, Default)]
pub struct RealtimeAssetDirectories {
    /// Model configuration and model-specific tokenizers. SafeTensors defaults
    /// to its directory; GGUF and split sources require this explicit directory.
    pub model: Option<PathBuf>,
    /// Referenced text backbone configuration and tokenizer snapshot.
    pub text_backbone: Option<PathBuf>,
}

/// Backend-independent input to realtime discovery and preparation.
#[derive(Debug, Clone)]
pub struct RealtimeArtifactRequest {
    /// Physical checkpoint sources.
    pub source: RealtimeArtifactSource,
    /// Auxiliary asset roots.
    pub assets: RealtimeAssetDirectories,
}
impl RealtimeArtifactRequest {
    /// Prepares assets colocated with a SafeTensors model directory or file.
    pub fn safetensors(path: impl AsRef<Path>) -> Self {
        Self {
            source: RealtimeArtifactSource::SafeTensors {
                path: path.as_ref().to_owned(),
            },
            assets: RealtimeAssetDirectories::default(),
        }
    }
}

/// Exact architecture-owned artifacts accepted by realtime loading.
pub enum PreparedRealtimeArtifact {
    /// Moshi and PersonaPlex normalized checkpoint and binding recipes.
    Moshi(MoshiPreparationPlan),
    /// VoiceChat component stores, preprocessing and auxiliary assets.
    VoiceChat(RetainedVoiceChatArtifacts),
}
impl PreparedRealtimeArtifact {
    /// Discovered architecture identity.
    pub const fn family(&self) -> ModelKind {
        match self {
            Self::Moshi(_) => ModelKind::Moshi,
            Self::VoiceChat(_) => ModelKind::VoiceChat,
        }
    }
    /// Dispatches backend-independent preparation through an exhaustive visitor.
    pub fn visit<'a, V: RealtimeArtifactVisitor<'a>>(
        &'a self,
        visitor: V,
    ) -> Result<V::Output, V::Error> {
        match self {
            Self::Moshi(plan) => visitor.moshi(plan),
            Self::VoiceChat(artifacts) => visitor.voicechat(artifacts),
        }
    }
}

/// Typed realization of an already discovered realtime architecture.
pub trait RealtimeArtifactVisitor<'a>: Sized {
    /// Common output retained by the caller.
    type Output;
    /// Caller-owned portable error.
    type Error;
    /// Uses the exact Moshi-family preparation.
    fn moshi(self, preparation: &'a MoshiPreparationPlan) -> Result<Self::Output, Self::Error>;
    /// Uses the exact VoiceChat component stores and assets.
    fn voicechat(
        self,
        artifacts: &'a RetainedVoiceChatArtifacts,
    ) -> Result<Self::Output, Self::Error>;
}

/// A realtime request rejected before native resource construction.
#[derive(Debug, thiserror::Error)]
pub enum RealtimePreparationError {
    /// A required local auxiliary source is absent.
    #[error("realtime preparation requires the {0} asset directory")]
    MissingAssetDirectory(&'static str),
    /// The discovered model uses another execution protocol.
    #[error("model family {0:?} does not use realtime execution")]
    NotRealtime(ModelKind),
    /// The selected storage layout is not declared by the discovered family.
    #[error("model family {0:?} does not admit this realtime source layout")]
    SourceLayout(ModelKind),
    /// Generic model identity or configuration admission failed.
    #[error(transparent)]
    Configuration(#[from] eredu_core::artifact::ArtifactError),
    /// Reading a local configuration failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Local configuration is not JSON.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Moshi-family preparation failed.
    #[error(transparent)]
    Moshi(#[from] crate::moshi::MoshiPreparationError),
    /// VoiceChat component admission or retention failed.
    #[error(transparent)]
    VoiceChat(#[from] AdmissionError),
    /// Referenced VoiceChat auxiliary assets failed exact validation.
    #[error(transparent)]
    VoiceChatAssets(#[from] VoiceChatAssetError),
}

/// Discovers a realtime family, admits its exact sources and retains its assets.
/// Learned tensors remain lazy; bounded control and preprocessing buffers needed
/// for cold admission may be read. Construction consumes only this retained result.
pub fn prepare_realtime_model(
    request: &RealtimeArtifactRequest,
) -> Result<PreparedRealtimeArtifact, RealtimePreparationError> {
    let default_model = match &request.source {
        RealtimeArtifactSource::SafeTensors { path } => Some(if path.is_dir() {
            path.as_path()
        } else {
            path.parent().unwrap_or(Path::new("."))
        }),
        _ => None,
    };
    let model_root = request
        .assets
        .model
        .as_deref()
        .or(default_model)
        .ok_or(RealtimePreparationError::MissingAssetDirectory("model"))?;
    let config_bytes = match std::fs::read(model_root.join("config.json")) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if let (RealtimeArtifactSource::SafeTensors { path }, None) =
                (&request.source, &request.assets.model)
            {
                // Released native Moshi has no configuration document. Its complete
                // strict header schema is the identity contract, not the directory name.
                return Ok(PreparedRealtimeArtifact::Moshi(
                    crate::moshi::prepare_artifact(path, None)?,
                ));
            }
            return Err(error.into());
        }
        Err(error) => return Err(error.into()),
    };
    let config: serde_json::Value = serde_json::from_slice(&config_bytes)?;
    let family = resolve_model_identity(&config)?.kind;
    match family {
        ModelKind::Moshi => {
            let RealtimeArtifactSource::SafeTensors { path } = &request.source else {
                return Err(RealtimePreparationError::SourceLayout(family));
            };
            Ok(PreparedRealtimeArtifact::Moshi(
                crate::moshi::prepare_artifact(path, Some(&config))?,
            ))
        }
        ModelKind::VoiceChat => {
            let architecture =
                VoiceChatConfig::from_value(&config).map_err(AdmissionError::from)?;
            let text_root = request.assets.text_backbone.as_deref().ok_or(
                RealtimePreparationError::MissingAssetDirectory("text backbone"),
            )?;
            let mut config_bytes = Some(config_bytes);
            let assets = PreparedVoiceChatAssets::resolve(|asset| {
                if asset.role == VoiceChatAssetRole::Configuration {
                    return Ok(config_bytes.take().expect("configuration requested once"));
                }
                let root = if asset.model == VOICECHAT_ASSET_MODEL {
                    model_root
                } else {
                    text_root
                };
                std::fs::read(root.join(asset.filename))
            })?;
            let complete = |path: &Path| {
                if path
                    .extension()
                    .is_some_and(|extension| extension == "gguf")
                {
                    AdmittedVoiceChatSource::gguf(path, &architecture, GgufLayout::Complete)
                } else {
                    AdmittedVoiceChatSource::safetensors(path, &architecture)
                }
            };
            let (sources, companion) = match &request.source {
                RealtimeArtifactSource::SafeTensors { path } => (
                    vec![AdmittedVoiceChatSource::safetensors(path, &architecture)?],
                    None,
                ),
                RealtimeArtifactSource::Gguf { path } => (
                    vec![AdmittedVoiceChatSource::gguf(
                        path,
                        &architecture,
                        GgufLayout::Complete,
                    )?],
                    None,
                ),
                RealtimeArtifactSource::SplitGguf {
                    perception,
                    backbone,
                    function_head,
                    tts,
                    companion,
                } => (
                    [
                        (perception, GgufLayout::Perception),
                        (backbone, GgufLayout::Backbone),
                        (function_head, GgufLayout::FunctionHead),
                        (tts, GgufLayout::Tts),
                    ]
                    .into_iter()
                    .map(|(path, layout)| {
                        AdmittedVoiceChatSource::gguf(path, &architecture, layout)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                    Some(complete(companion)?),
                ),
            };
            let required = BTreeSet::from([
                Component::Perception,
                Component::Backbone,
                Component::FunctionHead,
                Component::Rnnt,
                Component::Tts,
                Component::CodecEncoder,
                Component::CodecDecoder,
                Component::CodecQuantizer,
                Component::Controls,
            ]);
            let artifacts =
                VoiceChatArtifactPlan::new(&architecture, sources, companion, required)?
                    .retain(assets, 2)?;
            Ok(PreparedRealtimeArtifact::VoiceChat(artifacts))
        }
        other => Err(RealtimePreparationError::NotRealtime(other)),
    }
}
