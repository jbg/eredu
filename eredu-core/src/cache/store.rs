//! Storage-independent publication and reading of the existing prompt-cache format.
use super::{PromptCacheError, PromptCacheManifest, PromptCacheOptions};
use std::{
    io::{Read, Seek},
    path::{Path, PathBuf},
};

/// An owned, seekable shard reader. A published generation must remain readable
/// after replacement or deletion of its store's current entry.
pub trait PromptCacheReader: Read + Seek + Send {}
impl<T: Read + Seek + Send> PromptCacheReader for T {}

/// One immutable published generation, with the same manifest for every store.
pub trait PromptCacheRead: Send + Sync {
    /// Validated cache identity and state geometry.
    fn manifest(&self) -> &PromptCacheManifest;
    /// Opens the exact shard admitted by this generation.
    fn open_shard(&self, name: &str) -> Result<Box<dyn PromptCacheReader>, PromptCacheStoreError>;
    /// Optional stable file location for lazy disk-backed restoration.
    fn shard_path(&self, _name: &str) -> Option<PathBuf> {
        None
    }
}

/// Private staging generation. Drop aborts; commit publishes atomically.
pub trait PromptCacheWrite {
    /// Publishes one encoded safetensors shard into this private generation.
    /// Returns its payload SHA-256 (excluding the safetensors header).
    fn put_shard(&mut self, name: &str, bytes: Vec<u8>) -> Result<String, PromptCacheStoreError>;
    /// Validates the complete manifest and publishes the generation.
    fn commit(self: Box<Self>, manifest: &PromptCacheManifest)
        -> Result<(), PromptCacheStoreError>;
}

/// Store for one reusable prompt-cache generation. Memory and directory stores
/// use identical shard encoding, identity checks and publication semantics.
pub trait PromptCacheStore: Send + Sync {
    /// Opens and pins the currently published generation, if any.
    fn open(&self) -> Result<Option<Box<dyn PromptCacheRead>>, PromptCacheStoreError>;
    /// Reserves a conservative complete capture allowance before copying state.
    fn begin(
        &self,
        reserved_bytes: u64,
        options: &PromptCacheOptions,
    ) -> Result<Box<dyn PromptCacheWrite + '_>, PromptCacheStoreError>;
    /// Evicts the current entry; existing readers remain valid.
    fn clear(&self) -> Result<(), PromptCacheStoreError>;
    /// Retained bytes reported by the store. Memory stores include active writers
    /// and generations pinned by readers; directory stores report current payload bytes.
    fn retained_bytes(&self) -> u64;
    /// Finite capture allowance (or an explicit directory-store allowance).
    fn max_bytes(&self) -> u64;
    /// Directory publication support used by existing all-rank coordination.
    fn directory(&self) -> Option<&Path> {
        None
    }
}

/// Portable storage, format and resource failures.
#[derive(Debug, thiserror::Error)]
pub enum PromptCacheStoreError {
    /// Invalid cache identity or state declarations.
    #[error(transparent)]
    Cache(#[from] PromptCacheError),
    /// Capture would exceed the store's explicit allowance.
    #[error("prompt cache needs {requested} bytes; store allowance is {limit}")]
    Budget {
        /// Conservative requested capture allowance.
        requested: u64,
        /// Configured store limit.
        limit: u64,
    },
    /// Storage implementation failure, retaining its original error source.
    #[error("prompt cache store: {0}")]
    Storage(#[source] crate::BackendFailure),
}
impl PromptCacheStoreError {
    /// Preserves an implementation error behind the neutral backend failure type.
    pub fn from_error(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Storage(crate::BackendFailure::from_error(error))
    }
}

impl<T: PromptCacheStore + ?Sized> PromptCacheStore for std::sync::Arc<T> {
    fn open(&self) -> Result<Option<Box<dyn PromptCacheRead>>, PromptCacheStoreError> {
        (**self).open()
    }
    fn begin(
        &self,
        bytes: u64,
        options: &PromptCacheOptions,
    ) -> Result<Box<dyn PromptCacheWrite + '_>, PromptCacheStoreError> {
        (**self).begin(bytes, options)
    }
    fn clear(&self) -> Result<(), PromptCacheStoreError> {
        (**self).clear()
    }
    fn retained_bytes(&self) -> u64 {
        (**self).retained_bytes()
    }
    fn max_bytes(&self) -> u64 {
        (**self).max_bytes()
    }
    fn directory(&self) -> Option<&Path> {
        (**self).directory()
    }
}
