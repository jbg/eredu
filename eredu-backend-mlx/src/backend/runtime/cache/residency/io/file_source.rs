//! Closed, independently authenticated origins for the shared cache reader.
use super::*;
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::cache::{
    CacheShardLayout, LiveCacheReadFailure, PersistentCacheBlockSource, PersistentCacheReadFailure,
    PreparedLiveCacheRead, PreparedPersistentCacheRead,
};

#[derive(Clone, Debug)]
pub(crate) enum CacheFileSource {
    Live(LiveCacheBlockSource),
    Persistent(PersistentCacheBlockSource),
}
impl CacheFileSource {
    pub(crate) fn path(&self) -> &Path {
        match self {
            Self::Live(value) => value.path(),
            Self::Persistent(value) => value.path(),
        }
    }
    pub(crate) fn layout(&self) -> Option<&CacheShardLayout> {
        match self {
            Self::Live(value) => value.writer_layout(),
            Self::Persistent(value) => Some(value.layout()),
        }
    }
    pub(crate) fn file_bytes(&self) -> Option<usize> {
        match self {
            Self::Live(value) => value.file_bytes(),
            Self::Persistent(value) => Some(value.file_bytes()),
        }
    }
    pub(crate) fn same_source(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Live(left), Self::Live(right)) => left.same_source(right),
            (Self::Persistent(left), Self::Persistent(right)) => left.same_source(right),
            _ => false,
        }
    }
    pub(crate) fn same_live_source(&self, other: &LiveCacheBlockSource) -> bool {
        matches!(self, Self::Live(source) if source.same_source(other))
    }
    pub(crate) fn owns_declared_backing(&self) -> bool {
        match self {
            Self::Live(value) => value.owns_disk_reservation(),
            // Import authentication is a distinct file source, never a Disk token.
            Self::Persistent(_) => true,
        }
    }
    pub(crate) fn read_control_bytes(&self) -> Option<usize> {
        match self {
            Self::Live(_) => LiveCacheBlockSource::read_control_bytes(),
            Self::Persistent(_) => PersistentCacheBlockSource::read_control_bytes(),
        }
    }
    pub(crate) fn prepare_read_from(
        &self,
        file: File,
        context: &WorkspaceContext,
    ) -> Result<PreparedCacheFileRead, CacheFileReadFailure> {
        match self {
            Self::Live(value) => value
                .prepare_read_from(file, context)
                .map(PreparedCacheFileRead::Live)
                .map_err(CacheFileReadFailure::Live),
            Self::Persistent(value) => value
                .prepare_read_from(file, context)
                .map(PreparedCacheFileRead::Persistent)
                .map_err(CacheFileReadFailure::Persistent),
        }
    }
    pub(crate) fn read_ordinary_into(
        &self,
        file: File,
        destination: &mut [u8],
    ) -> Result<(), CacheFileReadFailure> {
        match self {
            Self::Live(value) => value
                .read_ordinary_into(file, destination)
                .map_err(CacheFileReadFailure::Live),
            Self::Persistent(value) => value
                .read_ordinary_into(file, destination)
                .map_err(CacheFileReadFailure::Persistent),
        }
    }
}

pub(crate) enum PreparedCacheFileRead {
    Live(PreparedLiveCacheRead),
    Persistent(PreparedPersistentCacheRead),
}
impl PreparedCacheFileRead {
    pub(crate) fn copy_to(self, destination: &mut File) -> Result<(), CacheFileReadFailure> {
        match self {
            Self::Live(value) => value
                .copy_to(destination)
                .map_err(CacheFileReadFailure::Live),
            Self::Persistent(value) => value
                .copy_to(destination)
                .map_err(CacheFileReadFailure::Persistent),
        }
    }
    pub(crate) fn read_into(self, destination: &mut [u8]) -> Result<(), CacheFileReadFailure> {
        match self {
            Self::Live(value) => value
                .read_into(destination)
                .map_err(CacheFileReadFailure::Live),
            Self::Persistent(value) => value
                .read_into(destination)
                .map_err(CacheFileReadFailure::Persistent),
        }
    }
}
#[derive(Debug, thiserror::Error)]
pub(crate) enum CacheFileReadFailure {
    #[error(transparent)]
    Live(#[from] LiveCacheReadFailure),
    #[error(transparent)]
    Persistent(#[from] PersistentCacheReadFailure),
}

impl DiskLocation {
    /// Only one authenticated source kind can back this canonical descriptor.
    pub(crate) fn file_source(&self) -> Option<CacheFileSource> {
        if self.buffered.is_some() {
            return None;
        }
        match (&self.live_source, &self.persistent_source, self.persistent) {
            (Some(source), None, false) => Some(CacheFileSource::Live(source.clone())),
            (None, Some(source), true) => Some(CacheFileSource::Persistent(source.clone())),
            _ => None,
        }
    }
}

impl From<CacheFileReadFailure> for CacheResidencyError {
    fn from(cause: CacheFileReadFailure) -> Self {
        match cause {
            CacheFileReadFailure::Live(cause) => Self::LiveRead(cause),
            CacheFileReadFailure::Persistent(cause) => Self::PersistentRead(cause),
        }
    }
}
