//! The actual published file owner plus the existing positional read worker.
use super::*;
use eredu_checkpoint::artifact::{
    ArtifactFileReadError, ArtifactFileReadFailure, PreparedArtifactFileRead,
};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError, HostMetadataFunding};
use std::mem::{size_of, size_of_val};

/// Paid move-only read from an exact published cache source. Opening the supplied
/// handle, payload allocation, queue/task storage and native transfer remain the
/// caller's independent producers. This object owns no destination or retry grant.
#[derive(Debug)]
pub struct PreparedLiveCacheRead {
    read: PreparedArtifactFileRead,
    source: LiveCacheBlockSource,
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Metadata(WorkspaceMetadataError),
    #[error(transparent)]
    Binding(ArtifactFileReadError),
    #[error(transparent)]
    Transfer(ArtifactFileReadFailure),
}
/// Failed binding/read retaining the actual file/source prefix and its metadata
/// custody. The caller must separately retain or discard its destination buffer.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct LiveCacheReadFailure {
    #[source]
    cause: Cause,
    // A transfer cause owns the actual opened handle and drops before the file
    // publication. Every actual source alias then retires before read funding.
    source: LiveCacheBlockSource,
    funding: Option<HostMetadataFunding>,
}
impl LiveCacheReadFailure {
    /// Exact source publication retained through this failure.
    pub fn source_file(&self) -> &LiveCacheBlockSource {
        &self.source
    }
    /// Written prefix, excluding any caller-provided initialized tail.
    pub fn filled_bytes(&self) -> usize {
        match &self.cause {
            Cause::Transfer(cause) => cause.filled_bytes(),
            _ => 0,
        }
    }
    /// Unchanged binding/version failure before any byte transfer.
    pub fn binding_error(&self) -> Option<&ArtifactFileReadError> {
        match &self.cause {
            Cause::Binding(cause) => Some(cause),
            _ => None,
        }
    }
    /// Existing positional-read failure, with its actual handle and progress.
    pub fn read_error(&self) -> Option<&ArtifactFileReadFailure> {
        match &self.cause {
            Cause::Transfer(cause) => Some(cause),
            _ => None,
        }
    }
}
impl LiveCacheBlockSource {
    /// Actual complete published file extent for an ordinary destination reserve.
    pub fn file_bytes(&self) -> Option<usize> {
        self.inner.version.map(|version| version.byte_len())
    }
    /// Ordinary disk-worker read through the same retained-version positional
    /// worker. Its caller owns unquoted byte storage and I/O preparation.
    pub fn read_ordinary_into(
        &self,
        file: File,
        destination: &mut [u8],
    ) -> Result<(), LiveCacheReadFailure> {
        let failed = |cause| LiveCacheReadFailure {
            cause,
            source: self.clone(),
            funding: None,
        };
        let version = self
            .inner
            .version
            .ok_or_else(|| failed(Cause::Metadata(WorkspaceMetadataError::Unqualified)))?;
        version
            .bind(file)
            .map_err(|cause| failed(Cause::Binding(cause)))?
            .read_into(destination)
            .map_err(|cause| failed(Cause::Transfer(cause)))
    }

    /// Exact finite read/binding/return frames. File opening/path conversion and
    /// the byte destination are independently owned, not included as metadata.
    pub fn read_control_bytes() -> Option<usize> {
        let frames = [
            PreparedArtifactFileRead::control_bytes()?,
            size_of::<PreparedLiveCacheRead>(),
            size_of::<LiveCacheReadFailure>(),
            size_of::<LiveCacheBlockSource>(),
            size_of::<Cause>(),
            size_of::<Option<HostMetadataFunding>>(),
            size_of::<HostMetadataFunding>(),
            size_of::<(&Self, File, &WorkspaceContext)>(),
            size_of::<(PreparedLiveCacheRead, &mut [u8])>(),
            size_of::<Result<PreparedLiveCacheRead, LiveCacheReadFailure>>(),
            size_of::<Result<(), LiveCacheReadFailure>>(),
            size_of::<Result<PreparedArtifactFileRead, ArtifactFileReadError>>(),
            size_of::<Result<(), ArtifactFileReadFailure>>(),
            size_of::<Result<(), WorkspaceMetadataError>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Binds an already opened handle to this actual publication's version.
    /// Supplying matching dimensions or equal bytes cannot replace file identity.
    /// The caller owns opening and destination preparation; no reads occur here.
    pub fn prepare_read_from(
        &self,
        file: File,
        context: &WorkspaceContext,
    ) -> Result<PreparedLiveCacheRead, LiveCacheReadFailure> {
        let funding = context.metadata_funding();
        let failed = |cause| LiveCacheReadFailure {
            cause,
            source: self.clone(),
            funding: funding.clone(),
        };
        let accepted = funding
            .as_ref()
            .ok_or_else(|| failed(Cause::Metadata(WorkspaceMetadataError::Unqualified)))?;
        context
            .charge_metadata(
                Self::read_control_bytes()
                    .ok_or_else(|| failed(Cause::Metadata(WorkspaceMetadataError::Overflow)))?,
            )
            .map_err(|cause| failed(Cause::Metadata(cause)))?;
        let version = self
            .inner
            .version
            .ok_or_else(|| failed(Cause::Metadata(WorkspaceMetadataError::Unqualified)))?;
        let read = version
            .bind(file)
            .map_err(|cause| failed(Cause::Binding(cause)))?;
        Ok(PreparedLiveCacheRead {
            read,
            source: self.clone(),
            funding: accepted.clone(),
        })
    }
}
impl PreparedLiveCacheRead {
    /// Actual sealed source length required by the separately admitted output.
    pub fn byte_len(&self) -> usize {
        self.read.byte_len()
    }
    /// Consumes the existing positional worker into an exact caller-owned buffer.
    /// Read failures keep the same actual file and source owner until retirement.
    pub fn read_into(self, destination: &mut [u8]) -> Result<(), LiveCacheReadFailure> {
        let Self {
            read,
            source,
            funding,
        } = self;
        match read.read_into(destination) {
            Ok(()) => {
                drop(source);
                drop(funding);
                Ok(())
            }
            Err(cause) => Err(LiveCacheReadFailure {
                cause: Cause::Transfer(cause),
                source,
                funding: Some(funding),
            }),
        }
    }
}
#[cfg(all(test, unix))]
#[path = "live_read/tests.rs"]
mod tests;
