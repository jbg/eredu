use std::io;
use std::path::PathBuf;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Structured GGUF processing failures.
#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error at byte offset {offset}: {source}")]
    Io {
        offset: u64,
        #[source]
        source: io::Error,
    },
    #[error("failed to read GGUF shard {path:?}: {source}")]
    Shard {
        path: PathBuf,
        #[source]
        source: Box<Error>,
    },
    #[error("invalid GGUF header: {0}")]
    InvalidHeader(String),
    #[error("invalid GGUF metadata key {key:?}: {reason}")]
    InvalidMetadata { key: String, reason: String },
    #[error("invalid GGUF tensor {tensor:?}: {reason}")]
    InvalidTensor { tensor: String, reason: String },
    #[error("duplicate GGUF metadata key {0:?}")]
    DuplicateMetadata(String),
    #[error("duplicate GGUF tensor name {0:?}")]
    DuplicateTensor(String),
    #[error("invalid GGUF checkpoint: {0}")]
    InvalidShardSet(String),
    #[error(
        "GGUF logical tensor name {name:?} is produced by both {first_source:?} and {second_source:?}"
    )]
    DuplicateLogicalTensor {
        name: String,
        first_source: String,
        second_source: String,
    },
    #[error(
        "GGUF tensor names {first_source:?} and {second_source:?} collide after translation to {name:?}"
    )]
    TranslatedTensorCollision {
        name: String,
        first_source: String,
        second_source: String,
    },
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("unsupported GGUF metadata type {0}")]
    UnsupportedMetadataType(u32),
    #[error("unsupported GGML tensor type {0}")]
    UnsupportedTensorType(u32),
    #[error("resource limit exceeded for {resource}: {actual} > {limit}")]
    Limit {
        resource: &'static str,
        actual: u64,
        limit: u64,
    },
    #[error("integer overflow while computing {0}")]
    Overflow(&'static str),
}

impl Error {
    /// Returns the unsupported GGML tensor type carried by this failure.
    ///
    /// Checkpoint operations add [`Error::Shard`] context around reader
    /// failures. This accessor preserves semantic classification without
    /// requiring callers to inspect rendered diagnostics.
    pub fn unsupported_tensor_type_code(&self) -> Option<u32> {
        match self {
            Self::UnsupportedTensorType(code) => Some(*code),
            Self::Shard { source, .. } => source.unsupported_tensor_type_code(),
            _ => None,
        }
    }

    pub(crate) fn tensor(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidTensor {
            tensor: name.into(),
            reason: reason.into(),
        }
    }
}
