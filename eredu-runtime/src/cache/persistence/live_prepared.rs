//! Counted names and owning destinations for the shared live-file publication.
use super::*;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};
use std::{
    fmt::{self, Write as _},
    mem::{size_of, size_of_val},
};

#[derive(Clone, Copy, Debug)]
pub(super) enum PublicationAction {
    Publish,
    RemoveStaging,
    OpenSource,
}
impl PublicationAction {
    pub(super) fn description(self) -> &'static str {
        match self {
            Self::Publish => "publish uniquely named live cache block",
            Self::RemoveStaging => "remove published live cache temporary file",
            Self::OpenSource => "open completed live cache source",
        }
    }
}
#[derive(Debug)]
pub(super) struct PublicationIo {
    pub(super) action: PublicationAction,
    pub(super) source: std::io::Error,
}

struct Rank(Option<usize>);
impl fmt::Display for Rank {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(value) => write!(out, "{value}"),
            None => out.write_str("x"),
        }
    }
}
struct FileName<'a> {
    namespace: &'a str,
    publication: u64,
    id: &'a CacheBlockId,
}
impl fmt::Display for FileName<'_> {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        let representation = match self.id.representation {
            CacheRepresentation::KeyValue => "kv",
            CacheRepresentation::CompressedLatentRotary => "mla",
        };
        let rank = self.id.rank;
        write!(
            out,
            "live-{}-w{:016x}-s{:016x}-layer-{:05}-{representation}-rank-p{}-t{}-e{}-{}-{}",
            self.namespace,
            self.publication,
            self.id.session_id,
            self.id.global_layer,
            Rank(rank.and_then(|r| r.stage_rank())),
            Rank(rank.and_then(|r| r.shard_rank())),
            Rank(rank.and_then(|r| r.addressable_rank())),
            self.id.start,
            self.id.end
        )
    }
}
#[derive(Default)]
struct Count(usize);
impl fmt::Write for Count {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.0 = self.0.checked_add(value.len()).ok_or(fmt::Error)?;
        Ok(())
    }
}
fn name_bytes(name: &FileName<'_>) -> Option<usize> {
    let mut count = Count::default();
    write!(&mut count, "{name}").ok()?;
    Some(count.0)
}
fn name_string(name: &FileName<'_>, capacity: usize) -> String {
    let mut value = String::with_capacity(capacity);
    // Same formatter was counted first; no source or counter can change here.
    write!(&mut value, "{name}").expect("String formatting cannot fail");
    value
}
fn paths(
    directory: &Path,
    name: &str,
    destination_capacity: usize,
    staging_capacity: usize,
) -> LiveCacheBlockPublication {
    let mut destination = PathBuf::with_capacity(destination_capacity);
    destination.push(directory);
    // Extend the last component in place. Names are ASCII and contain no path separators.
    destination.push(name);
    destination.as_mut_os_string().push(".safetensors");
    let mut staging = PathBuf::with_capacity(staging_capacity);
    staging.push(directory);
    // Append the complete filename as one component, preserving Path::join.
    let mut component =
        std::ffi::OsString::with_capacity(1 + name.len() + ".tmp.safetensors".len());
    component.push(".");
    component.push(name);
    component.push(".tmp.safetensors");
    staging.push(component);
    LiveCacheBlockPublication {
        destination,
        staging,
        committed: false,
    }
}

pub(super) fn ordinary_paths(
    directory: &Path,
    id: &CacheBlockId,
    namespace: &str,
    publication: u64,
) -> LiveCacheBlockPublication {
    let name = FileName {
        namespace,
        publication,
        id,
    };
    let bytes = name_bytes(&name).expect("live filename fits address space");
    let name = name_string(&name, bytes);
    let destination = directory
        .as_os_str()
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_add(bytes))
        .and_then(|n| n.checked_add(".safetensors".len()))
        .expect("live path fits address space");
    let staging = directory
        .as_os_str()
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_add(1 + bytes + ".tmp.safetensors".len()))
        .expect("live path fits address space");
    paths(directory, &name, destination, staging)
}

/// Paid one-use path/file-owner destination for the existing live-cache writer.
/// Source selection, serializer buffers and OS I/O resources are independently
/// qualified by its caller. This value grants no cache or native execution.
#[derive(Debug)]
pub struct PreparedLiveCachePublication {
    file: Option<File>,
    publication: LiveCacheBlockPublication,
    source: LiveCacheBlockSource,
}
/// Failed atomic publication retaining both paths, unpublished file owner and
/// funding. No path/error string is cloned and no unrelated destination is removed.
#[derive(Debug)]
pub struct PreparedLiveCachePublicationFailure {
    cause: PublicationCause,
    retained: PreparedLiveCachePublication,
}
#[derive(Debug)]
enum PublicationCause {
    Io(PublicationIo),
    Version(eredu_checkpoint::artifact::ArtifactFileReadError),
    Storage(super::super::CachePoolError),
}
impl fmt::Display for PreparedLiveCachePublicationFailure {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            PublicationCause::Io(cause) => write!(
                out,
                "failed to {} at {}: {}",
                cause.action.description(),
                self.path().display(),
                cause.source
            ),
            PublicationCause::Storage(cause) => fmt::Display::fmt(cause, out),
            PublicationCause::Version(cause) => write!(
                out,
                "failed to inspect published live cache file at {}: {cause}",
                self.path().display()
            ),
        }
    }
}
impl std::error::Error for PreparedLiveCachePublicationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            PublicationCause::Io(cause) => &cause.source,
            PublicationCause::Version(cause) => cause,
            PublicationCause::Storage(cause) => cause,
        })
    }
}
impl PreparedLiveCachePublicationFailure {
    /// Exact path already owned by this failed publication.
    pub fn path(&self) -> &Path {
        match &self.cause {
            PublicationCause::Io(cause) => match cause.action {
                PublicationAction::Publish => self.retained.destination_path(),
                PublicationAction::RemoveStaging | PublicationAction::OpenSource => {
                    self.retained.staging_path()
                }
            },
            PublicationCause::Version(_) | PublicationCause::Storage(_) => {
                self.retained.destination_path()
            }
        }
    }
    /// Existing publication/open filesystem cause, without formatting.
    pub fn io_error(&self) -> Option<&std::io::Error> {
        match &self.cause {
            PublicationCause::Io(cause) => Some(&cause.source),
            _ => None,
        }
    }
    /// Exact post-publication file-version inspection cause, if that step failed.
    pub fn version_error(&self) -> Option<&eredu_checkpoint::artifact::ArtifactFileReadError> {
        match &self.cause {
            PublicationCause::Version(cause) => Some(cause),
            _ => None,
        }
    }
}

struct Layout {
    name: usize,
    destination: usize,
    staging: usize,
    fixed: usize,
}
impl Layout {
    fn inspect(directory: &Path, id: &CacheBlockId, namespace: &str) -> Option<Self> {
        // Every u64 publication ordinal occupies the same sixteen hex digits.
        let name = name_bytes(&FileName {
            namespace,
            publication: 0,
            id,
        })?;
        let component = 1usize
            .checked_add(name)?
            .checked_add(".tmp.safetensors".len())?;
        let prefix = directory.as_os_str().len().checked_add(1)?;
        let destination = prefix
            .checked_add(name)?
            .checked_add(".safetensors".len())?;
        let staging = prefix.checked_add(component)?;
        let frames = [
            name,
            destination,
            staging,
            destination,
            component,
            size_of::<Self>(),
            size_of::<FileName<'_>>(),
            size_of::<Count>(),
            size_of::<Rank>(),
            size_of::<String>(),
            size_of::<std::ffi::OsString>(),
            size_of::<PathBuf>(),
            size_of::<PreparedLiveCachePublication>(),
            size_of::<LiveCacheBlockPublication>(),
            size_of::<LiveCacheBlockSource>(),
            size_of::<PublicationIo>(),
            size_of::<PublicationCause>(),
            size_of::<Option<File>>(),
            size_of::<Option<CacheShardLayout>>(),
            size_of::<Option<super::super::CachePoolReservation>>(),
            super::super::CachePoolReservation::remaining_usage_control_bytes()?,
            size_of::<File>(),
            size_of::<Result<File, std::io::Error>>(),
            eredu_checkpoint::artifact::ArtifactFileVersion::control_bytes()?,
            size_of::<Option<&mut LiveCacheBlockFile>>(),
            size_of::<PreparedLiveCachePublicationFailure>(),
            size_of::<Result<PreparedLiveCachePublication, Error>>(),
            size_of::<Result<LiveCacheBlockSource, PreparedLiveCachePublicationFailure>>(),
            size_of::<Result<(), PublicationIo>>(),
            size_of::<Result<(), std::io::Error>>(),
            size_of::<Result<u64, u64>>(),
            size_of::<(&Path, &CacheBlockId, &WorkspaceContext)>(),
        ];
        let fixed = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)?;
        Some(Self {
            name,
            destination,
            staging,
            fixed,
        })
    }
}
impl PreparedLiveCachePublication {
    /// Complete actual Rust name/path and shared source-owner constructor census.
    /// OS and serialization allocations are outside this metadata contribution.
    pub fn control_bytes(directory: &Path, id: &CacheBlockId) -> Option<usize> {
        let namespace = LIVE_CACHE_PROCESS_NAMESPACE.get()?;
        Layout::inspect(directory, id, namespace)?
            .fixed
            .checked_add(WorkspaceContext::metadata_arc_bytes::<LiveCacheBlockFile>()?)
    }
    /// Prepares names from the ordinary manager's initialized process source.
    /// Counters are consumed once; abandoned/refused publications never refund
    /// or reuse an ordinal. No file is created or published by construction.
    pub fn prepare(
        directory: &Path,
        id: &CacheBlockId,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let funding = context
            .metadata_funding()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let namespace = LIVE_CACHE_PROCESS_NAMESPACE
            .get()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let layout =
            Layout::inspect(directory, id, namespace).ok_or(WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(layout.fixed)?;
        let publication = NEXT_LIVE_CACHE_PUBLICATION_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| WorkspaceMetadataError::Overflow)?;
        let name = name_string(
            &FileName {
                namespace,
                publication,
                id,
            },
            layout.name,
        );
        let publication = paths(directory, &name, layout.destination, layout.staging);
        let mut source_path = PathBuf::with_capacity(layout.destination);
        source_path.push(publication.destination_path());
        let inner = context.metadata_arc(LiveCacheBlockFile {
            path: source_path,
            published: AtomicBool::new(false),
            version: None,
            layout: None,
            storage: None,
        })?;
        Ok(Self {
            file: None,
            publication,
            source: LiveCacheBlockSource {
                inner,
                funding: Some(funding),
            },
        })
    }
    /// Unique staging destination for the caller's separately prepared writer.
    pub fn staging_path(&self) -> &Path {
        self.publication.staging_path()
    }
    /// Exact eventual immutable file path; it has not been published yet.
    pub fn destination_path(&self) -> &Path {
        self.publication.destination_path()
    }
    /// Runs the same hard-link/removal publication after the caller has completed
    /// its write/sync. The caller owns the OS operation's resource qualification.
    pub fn commit(self) -> Result<LiveCacheBlockSource, PreparedLiveCachePublicationFailure> {
        self.commit_inner(None, None)
    }
    /// Retains the actual immutable writer layout through publication and reads.
    /// Its original constructor funding remains independent of this path owner.
    pub fn commit_with_layout(
        self,
        layout: CacheShardLayout,
    ) -> Result<LiveCacheBlockSource, PreparedLiveCachePublicationFailure> {
        self.commit_inner(Some(layout), None)
    }
    /// Publishes this exact writer and moves its admitted Disk occupancy into
    /// the same shared file owner. Every file alias keeps that charge alive.
    /// Refusal retains both the uncommitted destination and reservation.
    pub fn commit_with_storage(
        self,
        layout: CacheShardLayout,
        storage: super::super::CachePoolReservation,
    ) -> Result<LiveCacheBlockSource, PreparedLiveCachePublicationFailure> {
        self.commit_inner(Some(layout), Some(storage))
    }
    fn commit_inner(
        mut self,
        layout: Option<CacheShardLayout>,
        storage: Option<super::super::CachePoolReservation>,
    ) -> Result<LiveCacheBlockSource, PreparedLiveCachePublicationFailure> {
        let owner = Arc::get_mut(&mut self.source.inner).expect("unpublished unique source");
        owner.layout = layout;
        owner.storage = storage;
        if let Some(storage) = &owner.storage {
            let valid = storage.remaining_usage().and_then(|usage| {
                let bytes = owner
                    .layout
                    .as_ref()
                    .and_then(|v| u64::try_from(v.file_bytes()).ok());
                if bytes == Some(usage.disk_bytes)
                    && usage.device_bytes == 0
                    && usage.host_bytes == 0
                    && usage.transfer_in_flight_bytes == 0
                {
                    Ok(())
                } else {
                    Err(super::super::CachePoolError::AccountingOverflow {
                        operation: "live file reservation differs from the exact writer extent",
                    })
                }
            });
            if let Err(cause) = valid {
                return Err(PreparedLiveCachePublicationFailure {
                    cause: PublicationCause::Storage(cause),
                    retained: self,
                });
            }
        }
        self.file = match File::open(self.staging_path()) {
            Ok(file) => Some(file),
            Err(source) => {
                return Err(PreparedLiveCachePublicationFailure {
                    cause: PublicationCause::Io(PublicationIo {
                        action: PublicationAction::OpenSource,
                        source,
                    }),
                    retained: self,
                });
            }
        };
        if let Err(cause) = self.publication.publish_raw() {
            return Err(PreparedLiveCachePublicationFailure {
                cause: PublicationCause::Io(cause),
                retained: self,
            });
        }
        self.source.inner.published.store(true, Ordering::Release);
        let version = match eredu_checkpoint::artifact::ArtifactFileVersion::capture(
            self.file.as_ref().expect("held publication handle"),
        ) {
            Ok(version) => version,
            Err(cause) => {
                return Err(PreparedLiveCachePublicationFailure {
                    cause: PublicationCause::Version(cause),
                    retained: self,
                });
            }
        };
        if self
            .source
            .inner
            .layout
            .as_ref()
            .is_some_and(|layout| layout.file_bytes() != version.byte_len())
        {
            return Err(PreparedLiveCachePublicationFailure {
                cause: PublicationCause::Version(
                    eredu_checkpoint::artifact::ArtifactFileReadError::DestinationLength,
                ),
                retained: self,
            });
        }
        // The source has never escaped or been cloned before publication.
        Arc::get_mut(&mut self.source.inner)
            .expect("unpublished unique source")
            .version = Some(version);
        drop(self.file.take());
        Ok(self.source)
    }
}

#[cfg(test)]
#[path = "live_prepared/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "live_prepared/storage_tests.rs"]
mod storage_tests;
