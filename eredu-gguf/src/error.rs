use std::io;
use std::path::PathBuf;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// A GGML encoding with no supported on-disk block geometry.
///
/// This fixed cause owns only the original numeric code; querying geometry
/// never constructs a general container, I/O or formatted metadata error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("unsupported GGML tensor type {code}")]
pub struct UnsupportedGgmlType {
    pub(crate) code: u32,
}

impl UnsupportedGgmlType {
    /// The exact unsupported code, including removed diagnostic encodings.
    pub const fn code(self) -> u32 {
        self.code
    }
}

/// Structured GGUF processing failures.
#[derive(Debug, Error)]
pub enum Error {
    /// The explicit prepared reader has no complete cold buffer available.
    #[error("prepared GGUF reader storage is unavailable: {resource}")]
    PreparedReaderStorage {
        /// The required immutable destination or available slot.
        resource: &'static str,
    },
    /// The retained prepared header cannot supply an exact parser destination.
    #[error("prepared GGUF header storage is unavailable: {resource}")]
    PreparedHeaderStorage {
        /// The exact retained destination that could not be supplied.
        resource: &'static str,
    },
    /// Bytes observed from an explicitly prepared header changed.
    #[error(transparent)]
    PreparedHeaderChanged(#[from] crate::PreparedHeaderChanged),
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
    /// Fixed prepared-reader storage refusal, including shard context.
    pub fn prepared_reader_storage(&self) -> Option<&'static str> {
        match self {
            Self::PreparedReaderStorage { resource } => Some(resource),
            Self::Shard { source, .. } => source.prepared_reader_storage(),
            _ => None,
        }
    }
    /// Fixed storage refusal from a prepared header, retaining any shard context.
    pub fn prepared_header_storage(&self) -> Option<&'static str> {
        match self {
            Self::PreparedHeaderStorage { resource } => Some(resource),
            Self::Shard { source, .. } => source.prepared_header_storage(),
            _ => None,
        }
    }
    /// The fixed source-change cause, including the original shard context.
    pub fn prepared_header_change(&self) -> Option<crate::PreparedHeaderChanged> {
        match self {
            Self::PreparedHeaderChanged(cause) => Some(*cause),
            Self::Shard { source, .. } => source.prepared_header_change(),
            _ => None,
        }
    }

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
