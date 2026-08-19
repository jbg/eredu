//! Stable model and artifact identity/configuration types.

use serde::{Deserialize, Serialize};

/// Stable identity of an immutable model artifact.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArtifactIdentity {
    /// Content-addressed artifact.
    ContentSha256 {
        /// Lowercase hexadecimal digest.
        digest: String,
    },
    /// Process-local model assembled without files.
    InMemory {
        /// Process-local unique identifier.
        id: u64,
    },
}

/// Portable model identity used by plans and sessions.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// Architecture family such as `llama`.
    pub family: String,
    /// Exact effective model type.
    pub model_type: String,
    /// Immutable checkpoint identity.
    pub artifact: ArtifactIdentity,
    /// Architecture/cache compatibility fingerprint.
    pub architecture_fingerprint: String,
}
