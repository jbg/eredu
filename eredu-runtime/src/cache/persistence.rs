//! Backend-neutral prompt-cache catalog validation and durable publication.

use eredu_core::cache::{
    CacheBlockId, CacheRepresentation, PROMPT_CACHE_SCHEMA_VERSION, PromptCacheBlock,
    PromptCacheError, PromptCacheManifest, PromptCacheStateTensor, PromptCacheTopology,
};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

mod funded;
pub use funded::{
    DEFAULT_PROMPT_CACHE_MANIFEST_BYTE_LIMIT, PROMPT_CACHE_DEPENDENCY_BASIS,
    PROMPT_CACHE_DEPENDENCY_SOURCE, PreparedPromptCachePublication,
    PreparedReversiblePromptCachePublication, PromptCachePersistenceFailure,
    PromptCachePersistenceFunding, hash_prompt_cache_shard_payload_from_funded,
    inspect_prompt_cache_funded, read_shard_metadata_from_funded,
};

static NEXT_LIVE_CACHE_PUBLICATION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_REVERSIBLE_CACHE_PUBLICATION_ID: AtomicU64 = AtomicU64::new(1);
static LIVE_CACHE_PROCESS_NAMESPACE: OnceLock<String> = OnceLock::new();

/// Maximum accepted safetensors metadata header size for a prompt-cache shard.
pub const MAX_PROMPT_CACHE_SHARD_HEADER_BYTES: u64 = 1024 * 1024;
/// Directory containing immutable replacement generations.
pub const PROMPT_CACHE_GENERATIONS_DIRECTORY: &str = ".generations";
/// Atomic pointer to the active immutable generation.
pub const PROMPT_CACHE_CURRENT_FILE: &str = "CURRENT";

/// Resolves the canonical rank-local prompt-cache storage directory.
///
/// Replicated cache state remains at `root`; distributed state is isolated by
/// all three semantic rank coordinates, using `x` for inactive axes.
pub fn prompt_cache_rank_path(root: &Path, topology: &PromptCacheTopology) -> PathBuf {
    funded::rank_path(root, topology, None).expect("prompt cache rank path fits address space")
}

/// Filesystem, catalog, and publication failures for a reusable prompt cache.
#[derive(Debug, thiserror::Error)]
pub enum PromptCachePersistenceError {
    /// Exact controlled metadata construction could not be funded.
    #[error(transparent)]
    Metadata(#[from] eredu_nn::workspace::WorkspaceMetadataError),
    /// The separately admitted dependency estimate was exhausted.
    #[error("prompt-cache dependency allowance: {0}")]
    Dependency(#[source] eredu_core::HostMetadataFundingError),
    /// An opened metadata file exceeds its admitted input extent.
    #[error("prompt-cache metadata input {bytes} exceeds limit {limit}")]
    MetadataInputLimit {
        /// Actual opened file length.
        bytes: usize,
        /// Selected metadata input limit.
        limit: usize,
    },
    /// Metadata grew after its exact opened-handle extent was admitted.
    #[error("prompt-cache metadata changed while reading its admitted extent")]
    MetadataChanged,
    /// Backend-neutral manifest geometry or identity is invalid.
    #[error(transparent)]
    PromptCache(#[from] PromptCacheError),
    /// A filesystem operation failed.
    #[error("failed to {action} at {path}: {source}")]
    Io {
        /// Filesystem action that failed.
        action: &'static str,
        /// Path involved in the failed action.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },
    /// A manifest could not be encoded or decoded.
    #[error("invalid prompt cache manifest JSON: {0}")]
    ManifestJson(#[source] serde_json::Error),
    /// Filesystem publication metadata is malformed.
    #[error("malformed prompt cache storage: {0}")]
    MalformedStorage(String),
    /// A shard path could escape the prompt-cache directory.
    #[error("unsafe prompt cache shard path {0:?}")]
    UnsafeShardPath(String),
    /// A manifest referenced a missing shard.
    #[error("missing prompt cache shard {0}")]
    MissingShard(PathBuf),
    /// A safetensors shard had missing, extra, or corrupt arrays.
    #[error("malformed prompt cache shard {path}: {reason}")]
    MalformedShard {
        /// Invalid shard path.
        path: PathBuf,
        /// Structural or data validation failure.
        reason: String,
    },
    /// The target path cannot be published atomically.
    #[error("invalid prompt cache path {0}")]
    InvalidPromptCachePath(PathBuf),
    /// The destination exists and explicit replacement was not requested.
    #[error("prompt cache destination already exists: {0}")]
    PromptCacheExists(PathBuf),
    /// A reversible publication method was invoked in the wrong lifecycle state.
    #[error("invalid reversible prompt cache publication state: {0}")]
    InvalidReversiblePublication(&'static str),
}

/// Filesystem failure while publishing one ephemeral live-cache block.
#[derive(Debug, thiserror::Error)]
pub enum LiveCachePublicationError {
    /// The completed published file could not establish its exact source version.
    #[error("failed to inspect live cache file {path}: {source}")]
    Source {
        /// Already owned publication path.
        path: PathBuf,
        /// Unchanged fixed or operating-system inspection cause.
        #[source]
        source: eredu_checkpoint::artifact::ArtifactFileReadError,
    },
    /// A filesystem operation failed.
    #[error("failed to {action} at {path}: {source}")]
    Io {
        /// Filesystem action that failed.
        action: &'static str,
        /// Path involved in the failed action.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },
}

/// Shared ownership of a file published by the live-cache writer.
///
/// Clones retain the same immutable file. The final owner removes only that
/// unique ephemeral path, after manager records, queued tasks and escaped
/// results have released it. This does not authorize a read or native transfer.
#[derive(Debug, Clone)]
pub struct LiveCacheBlockSource {
    inner: Arc<LiveCacheBlockFile>,
    // Each clone retires its actual file/Arc before the final funding alias.
    funding: Option<eredu_nn::workspace::HostMetadataFunding>,
}

#[derive(Debug)]
struct LiveCacheBlockFile {
    path: PathBuf,
    published: AtomicBool,
    version: Option<eredu_checkpoint::artifact::ArtifactFileVersion>,
    layout: Option<CacheShardLayout>,
    // File removal in Drop precedes release of its actual Disk reservation.
    storage: Option<super::CachePoolReservation>,
}

impl Drop for LiveCacheBlockFile {
    fn drop(&mut self) {
        if self.published.load(Ordering::Acquire) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

impl LiveCacheBlockSource {
    /// Exact uniquely published path, borrowed under this file owner.
    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    /// Immutable header/layout emitted by this publication's shared cache writer.
    /// Missing layout denotes an older/external caller requiring ordinary parsing.
    pub fn writer_layout(&self) -> Option<&CacheShardLayout> {
        self.inner.layout.as_ref()
    }

    /// Whether this actual shared file owns its independently admitted Disk
    /// occupancy. Manager reports must not charge that same reservation again.
    pub fn owns_disk_reservation(&self) -> bool {
        self.inner.storage.is_some()
    }

    /// Whether two source loans retain the same actual file lifetime.
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// Runtime-owned unique staging and atomic publication for one live-cache block.
#[derive(Debug)]
pub struct LiveCacheBlockPublication {
    destination: PathBuf,
    staging: PathBuf,
    committed: bool,
}

impl LiveCacheBlockPublication {
    /// Reserves unique paths derived from the complete block and rank identity.
    pub fn begin(directory: &Path, id: &CacheBlockId) -> Self {
        let namespace = Self::initialize_naming_source();
        let publication_id = NEXT_LIVE_CACHE_PUBLICATION_ID.fetch_add(1, Ordering::Relaxed);
        live_prepared::ordinary_paths(directory, id, namespace, publication_id)
    }

    /// Initializes the same immutable process naming source during ordinary
    /// manager setup. It creates no path, file, or read/write permission.
    pub fn initialize_naming_source() -> &'static str {
        LIVE_CACHE_PROCESS_NAMESPACE
            .get_or_init(|| {
                let started = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                format!("p{:08x}-t{started:032x}", std::process::id())
            })
            .as_str()
    }

    /// Unique temporary path into which the backend serializes native storage.
    pub fn staging_path(&self) -> &Path {
        &self.staging
    }

    /// Final unique path used by the live-cache catalog.
    pub fn destination_path(&self) -> &Path {
        &self.destination
    }

    /// Atomically publishes the staged file without replacing an existing path.
    pub fn commit(mut self) -> Result<PathBuf, LiveCachePublicationError> {
        self.publish()?;
        Ok(std::mem::take(&mut self.destination))
    }

    /// Publishes through the same writer and transfers file retirement to the
    /// actual shared source owner, rather than a particular manager record.
    pub fn commit_owned(self) -> Result<LiveCacheBlockSource, LiveCachePublicationError> {
        self.commit_owned_inner(None)
    }
    /// Publishes and retains the actual writer's immutable schema/extent.
    pub fn commit_owned_with_layout(
        self,
        layout: CacheShardLayout,
    ) -> Result<LiveCacheBlockSource, LiveCachePublicationError> {
        self.commit_owned_inner(Some(layout))
    }
    fn commit_owned_inner(
        mut self,
        layout: Option<CacheShardLayout>,
    ) -> Result<LiveCacheBlockSource, LiveCachePublicationError> {
        // Keep the actual staging handle across hard-link publication. A later
        // path replacement must not become this source's captured file version.
        let file = File::open(&self.staging).map_err(|source| LiveCachePublicationError::Io {
            action: "open completed live cache source",
            path: self.staging.clone(),
            source,
        })?;
        self.publish()?;
        let version = match eredu_checkpoint::artifact::ArtifactFileVersion::capture(&file) {
            Ok(version) => version,
            Err(source) => {
                let _ = fs::remove_file(&self.destination);
                return Err(LiveCachePublicationError::Source {
                    path: std::mem::take(&mut self.destination),
                    source,
                });
            }
        };
        if layout
            .as_ref()
            .is_some_and(|layout| layout.file_bytes() != version.byte_len())
        {
            let _ = fs::remove_file(&self.destination);
            return Err(LiveCachePublicationError::Source {
                path: std::mem::take(&mut self.destination),
                source: eredu_checkpoint::artifact::ArtifactFileReadError::DestinationLength,
            });
        }
        Ok(LiveCacheBlockSource {
            inner: Arc::new(LiveCacheBlockFile {
                path: std::mem::take(&mut self.destination),
                published: AtomicBool::new(true),
                version: Some(version),
                layout,
                storage: None,
            }),
            funding: None,
        })
    }

    fn publish(&mut self) -> Result<(), LiveCachePublicationError> {
        self.publish_raw()
            .map_err(|cause| LiveCachePublicationError::Io {
                action: cause.action.description(),
                path: match cause.action {
                    live_prepared::PublicationAction::Publish => self.destination.clone(),
                    live_prepared::PublicationAction::RemoveStaging
                    | live_prepared::PublicationAction::OpenSource => self.staging.clone(),
                },
                source: cause.source,
            })
    }
    fn publish_raw(&mut self) -> Result<(), live_prepared::PublicationIo> {
        use live_prepared::{PublicationAction, PublicationIo};
        fs::hard_link(&self.staging, &self.destination).map_err(|source| PublicationIo {
            action: PublicationAction::Publish,
            source,
        })?;
        if let Err(source) = fs::remove_file(&self.staging) {
            let _ = fs::remove_file(&self.destination);
            return Err(PublicationIo {
                action: PublicationAction::RemoveStaging,
                source,
            });
        }
        self.committed = true;
        Ok(())
    }
}

impl Drop for LiveCacheBlockPublication {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.staging);
        }
    }
}

/// A runtime-owned staging directory that publishes one immutable cache atomically.
#[derive(Debug)]
pub struct PromptCachePublication {
    destination: PathBuf,
    parent: PathBuf,
    generations: PathBuf,
    generation_name: String,
    publication_root: Option<PathBuf>,
    staging: PathBuf,
    replacing: bool,
    nonce: u128,
    committed: bool,
}

impl PromptCachePublication {
    /// Creates an isolated staging directory for a new cache or replacement generation.
    pub fn begin(
        destination: impl AsRef<Path>,
        replace_existing: bool,
    ) -> Result<Self, PromptCachePersistenceError> {
        Self::begin_with_funding(destination.as_ref(), replace_existing, None)
    }
    fn begin_with_funding(
        destination: &Path,
        replace_existing: bool,
        funding: Option<&funded::PersistenceAccounts>,
    ) -> Result<Self, PromptCachePersistenceError> {
        if let Some(funding) = funding {
            funding.path(destination)?;
        }
        let destination = funded::path_copy(destination, funding)?;
        let parent = funded::path_copy(
            destination
                .parent()
                .ok_or_else(|| funded::invalid_path(&destination, funding))?,
            funding,
        )?;
        let replacing = destination.exists();
        if replacing && !replace_existing {
            return Err(PromptCachePersistenceError::PromptCacheExists(destination));
        }
        if replacing && !destination.is_dir() {
            return Err(PromptCachePersistenceError::InvalidPromptCachePath(
                destination,
            ));
        }
        let file_name = destination
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| funded::invalid_path(&destination, funding))?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let generation_name = funded::text(format_args!("generation-{nonce}"), funding)?;
        // All path constructors finish before the first filesystem mutation.
        let publication_root = if replacing {
            None
        } else {
            Some(funded::path_join(
                &parent,
                funded::text(format_args!(".{file_name}.tmp-{nonce}"), funding)?,
                funding,
            )?)
        };
        let root = publication_root.as_deref().unwrap_or(&destination);
        let generations = funded::path_join(root, PROMPT_CACHE_GENERATIONS_DIRECTORY, funding)?;
        let staging = if replacing {
            funded::path_join(
                &generations,
                funded::text(format_args!(".tmp-{nonce}"), funding)?,
                funding,
            )?
        } else {
            funded::path_join(&generations, &generation_name, funding)?
        };
        fs::create_dir_all(&parent).map_err(|source| {
            funded::io_error("create prompt cache parent", &parent, source, funding)
        })?;
        if let Some(root) = &publication_root {
            fs::create_dir(root).map_err(|source| {
                funded::io_error("create temporary prompt cache root", root, source, funding)
            })?;
        }
        let generations_result = if replacing {
            fs::create_dir_all(&generations)
        } else {
            fs::create_dir(&generations)
        };
        if let Err(source) = generations_result {
            if let Some(root) = &publication_root {
                let _ = fs::remove_dir_all(root);
            }
            return Err(funded::io_error(
                "create prompt cache generation directory",
                &generations,
                source,
                funding,
            ));
        }
        if let Err(source) = fs::create_dir(&staging) {
            if let Some(root) = &publication_root {
                let _ = fs::remove_dir_all(root);
            }
            return Err(funded::io_error(
                "create temporary prompt cache",
                &staging,
                source,
                funding,
            ));
        }
        Ok(Self {
            destination,
            parent,
            generations,
            generation_name,
            publication_root,
            staging,
            replacing,
            nonce,
            committed: false,
        })
    }

    /// Directory into which the backend writes its native tensor shards.
    pub fn staging_directory(&self) -> &Path {
        &self.staging
    }

    /// Writes the manifest, validates every shard, and atomically publishes the cache.
    pub fn commit(self, manifest: &PromptCacheManifest) -> Result<(), PromptCachePersistenceError> {
        self.commit_with_funding(manifest, None)
    }
    fn commit_with_funding(
        mut self,
        manifest: &PromptCacheManifest,
        funding: Option<&funded::PersistenceAccounts>,
    ) -> Result<(), PromptCachePersistenceError> {
        if let Some(funding) = funding {
            funding.path(&self.staging)?;
            funding.estimate(funded::source_bytes(manifest)?)?;
            funding.buffer(8192)?;
        }
        let manifest_path = funded::path_join(&self.staging, "manifest.json", funding)?;
        let file = File::create(&manifest_path).map_err(|source| {
            funded::io_error(
                "create prompt cache manifest",
                &manifest_path,
                source,
                funding,
            )
        })?;
        let mut writer = BufWriter::with_capacity(8192, file);
        serde_json::to_writer_pretty(&mut writer, manifest)
            .map_err(PromptCachePersistenceError::ManifestJson)?;
        writer.write_all(b"\n").map_err(|source| {
            funded::io_error(
                "write prompt cache manifest",
                &manifest_path,
                source,
                funding,
            )
        })?;
        writer.flush().map_err(|source| {
            funded::io_error(
                "flush prompt cache manifest",
                &manifest_path,
                source,
                funding,
            )
        })?;
        sync_file_with_funding(&manifest_path, funding)?;
        validate_prompt_cache_manifest_with_funding(&self.staging, manifest, funding)?;
        sync_directory_with_funding(&self.staging, funding)?;

        if self.replacing {
            let generation = funded::path_join(&self.generations, &self.generation_name, funding)?;
            durable_rename(&self.staging, &generation, false).map_err(|source| {
                PromptCachePersistenceError::Io {
                    action: "publish prompt cache generation",
                    path: generation,
                    source,
                }
            })?;
            sync_directory_with_funding(&self.generations, funding)?;
            publish_generation_pointer(
                &self.destination,
                &self.generation_name,
                self.nonce,
                funding,
            )?;
            sync_directory_with_funding(&self.destination, funding)?;
        } else {
            sync_directory_with_funding(&self.generations, funding)?;
            let publication_root = self
                .publication_root
                .as_ref()
                .expect("new prompt-cache publication owns a staging root");
            publish_generation_pointer(
                publication_root,
                &self.generation_name,
                self.nonce,
                funding,
            )?;
            sync_directory_with_funding(publication_root, funding)?;
            durable_rename(publication_root, &self.destination, false).map_err(|source| {
                funded::io_error("publish prompt cache", &self.destination, source, funding)
            })?;
        }
        sync_directory_with_funding(&self.parent, funding)?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for PromptCachePublication {
    fn drop(&mut self) {
        if !self.committed {
            let staging = self.publication_root.as_ref().unwrap_or(&self.staging);
            if staging.exists() {
                let _ = fs::remove_dir_all(staging);
            }
        }
    }
}

/// A prepared cache directory whose visibility can be rolled back exactly.
///
/// The backend writes a complete ordinary cache to [`Self::staging_destination`].
/// Publication of a new destination is one directory rename. Replacement moves
/// only the immutable generation into the existing destination and atomically
/// switches `CURRENT`, retaining the previous generation identity until commit.
#[derive(Debug)]
pub struct ReversiblePromptCachePublication {
    destination: PathBuf,
    parent: PathBuf,
    staging: PathBuf,
    replace_existing: bool,
    previous_generation: Option<String>,
    moved_generation: Option<PathBuf>,
    published: bool,
    committed: bool,
    nonce: u128,
    funding: Option<funded::PersistenceAccounts>,
    rollback_funding: Option<funded::PersistenceAccounts>,
}

impl ReversiblePromptCachePublication {
    /// Reserves a unique sibling destination without creating or publishing it.
    pub fn begin(
        destination: impl AsRef<Path>,
        replace_existing: bool,
    ) -> Result<Self, PromptCachePersistenceError> {
        Self::begin_with_funding(destination.as_ref(), replace_existing, None)
    }
    fn begin_with_funding(
        destination: &Path,
        replace_existing: bool,
        funding: Option<&funded::PersistenceAccounts>,
    ) -> Result<Self, PromptCachePersistenceError> {
        if let Some(funding) = funding {
            funding.path(destination)?;
        }
        let destination = funded::path_copy(destination, funding)?;
        let parent = funded::path_copy(
            destination
                .parent()
                .ok_or_else(|| funded::invalid_path(&destination, funding))?,
            funding,
        )?;
        if destination.exists() && !replace_existing {
            return Err(PromptCachePersistenceError::PromptCacheExists(destination));
        }
        if destination.exists() && !destination.is_dir() {
            return Err(PromptCachePersistenceError::InvalidPromptCachePath(
                destination,
            ));
        }
        let file_name = destination
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| funded::invalid_path(&destination, funding))?;
        let publication_id = NEXT_REVERSIBLE_CACHE_PUBLICATION_ID.fetch_add(1, Ordering::Relaxed);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            ^ u128::from(publication_id);
        let staging = funded::path_join(
            &parent,
            funded::text(
                format_args!(
                    ".{file_name}.transaction-p{:08x}-{nonce:032x}",
                    std::process::id()
                ),
                funding,
            )?,
            funding,
        )?;
        if staging.exists() {
            return Err(PromptCachePersistenceError::InvalidPromptCachePath(staging));
        }
        let rollback_funding = funding
            .map(|funding| {
                funding.prepare_rollback(&destination, &parent, &staging, false, None, nonce)
            })
            .transpose()?;
        fs::create_dir_all(&parent).map_err(|source| {
            funded::io_error(
                "create reversible prompt cache parent",
                &parent,
                source,
                funding,
            )
        })?;
        Ok(Self {
            destination,
            parent,
            staging,
            replace_existing,
            previous_generation: None,
            moved_generation: None,
            published: false,
            committed: false,
            nonce,
            funding: funding.cloned(),
            rollback_funding,
        })
    }

    /// Hidden destination into which an ordinary cache save must be completed.
    pub fn staging_destination(&self) -> &Path {
        &self.staging
    }

    /// Makes the prepared cache visible while retaining an exact rollback path.
    pub fn publish(&mut self) -> Result<(), PromptCachePersistenceError> {
        self.publish_with_funding(None)
    }
    fn publish_with_funding(
        &mut self,
        funding: Option<&funded::PersistenceAccounts>,
    ) -> Result<(), PromptCachePersistenceError> {
        if let Some(funding) = funding {
            funding.path(&self.destination)?;
        }
        if self.published || self.moved_generation.is_some() {
            return Err(PromptCachePersistenceError::InvalidReversiblePublication(
                "publication was already attempted",
            ));
        }
        inspect_prompt_cache_with_funding(&self.staging, funding)?;
        if !self.destination.exists() {
            self.rollback_funding = funding
                .map(|funding| {
                    funding.prepare_rollback(
                        &self.destination,
                        &self.parent,
                        &self.staging,
                        false,
                        None,
                        self.nonce,
                    )
                })
                .transpose()?;
            durable_rename(&self.staging, &self.destination, false).map_err(|source| {
                funded::io_error(
                    "publish prepared prompt cache",
                    &self.destination,
                    source,
                    funding,
                )
            })?;
            self.published = true;
            sync_directory_with_funding(&self.parent, funding)?;
            return Ok(());
        }
        if !self.replace_existing {
            return Err(PromptCachePersistenceError::PromptCacheExists(
                funded::path_copy(&self.destination, funding)?,
            ));
        }

        let previous = resolve_prompt_cache_root_with_funding(&self.destination, funding)?;
        let previous_generation = previous
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                funded::storage_error(
                    format_args!("active prompt-cache generation has no safe name"),
                    funding,
                )
            })?;
        let previous_generation = funded::text(format_args!("{previous_generation}"), funding)?;
        let staged = resolve_prompt_cache_root_with_funding(&self.staging, funding)?;
        let generation_name = staged
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                funded::storage_error(
                    format_args!("prepared prompt-cache generation has no safe name"),
                    funding,
                )
            })?;
        let generation_name = funded::text(format_args!("{generation_name}"), funding)?;
        let generations = funded::path_join(
            &self.destination,
            PROMPT_CACHE_GENERATIONS_DIRECTORY,
            funding,
        )?;
        let target = funded::path_join(&generations, &generation_name, funding)?;
        if target.exists() {
            return Err(PromptCachePersistenceError::InvalidPromptCachePath(target));
        }
        self.rollback_funding = funding
            .map(|funding| {
                funding.prepare_rollback(
                    &self.destination,
                    &self.parent,
                    &self.staging,
                    true,
                    Some(&target),
                    self.nonce,
                )
            })
            .transpose()?;
        durable_rename(&staged, &target, false).map_err(|source| {
            funded::io_error(
                "install prepared prompt cache generation",
                &target,
                source,
                funding,
            )
        })?;
        self.previous_generation = Some(previous_generation);
        self.moved_generation = Some(target);
        sync_directory_with_funding(&generations, funding)?;
        publish_generation_pointer(&self.destination, &generation_name, self.nonce, funding)?;
        self.published = true;
        sync_directory_with_funding(&self.destination, funding)?;
        Ok(())
    }

    /// Accepts the visible cache and releases rollback metadata.
    pub fn commit(mut self) -> Result<(), PromptCachePersistenceError> {
        let retained = self
            .rollback_funding
            .as_ref()
            .or(self.funding.as_ref())
            .cloned();
        let funding = retained.as_ref();
        if let Some(funding) = funding {
            funding.path(&self.staging)?;
        }
        if !self.published {
            return Err(PromptCachePersistenceError::InvalidReversiblePublication(
                "an unpublished cache cannot commit",
            ));
        }
        if self.staging.exists() {
            fs::remove_dir_all(&self.staging).map_err(|source| {
                funded::io_error(
                    "remove committed prompt cache staging directory",
                    &self.staging,
                    source,
                    funding,
                )
            })?;
        }
        self.committed = true;
        Ok(())
    }

    /// Restores the prior visible cache, or removes a newly published cache.
    pub fn rollback(mut self) -> Result<(), PromptCachePersistenceError> {
        self.rollback_inner()?;
        self.committed = true;
        Ok(())
    }

    fn rollback_inner(&mut self) -> Result<(), PromptCachePersistenceError> {
        let retained = self
            .rollback_funding
            .as_ref()
            .or(self.funding.as_ref())
            .cloned();
        let funding = retained.as_ref();
        if let Some(funding) = funding {
            funding.path(&self.destination)?;
        }
        if self.published {
            match self.previous_generation.as_deref() {
                Some(previous) => {
                    publish_generation_pointer(
                        &self.destination,
                        previous,
                        self.nonce ^ 1,
                        funding,
                    )?;
                    sync_directory_with_funding(&self.destination, funding)?;
                }
                None if self.destination.exists() => {
                    fs::remove_dir_all(&self.destination).map_err(|source| {
                        funded::io_error(
                            "remove rolled-back prompt cache",
                            &self.destination,
                            source,
                            funding,
                        )
                    })?;
                    sync_directory_with_funding(&self.parent, funding)?;
                }
                None => {}
            }
        }
        if let Some(generation) = self.moved_generation.take() {
            if generation.exists() {
                fs::remove_dir_all(&generation).map_err(|source| {
                    funded::io_error(
                        "remove rolled-back prompt cache generation",
                        &generation,
                        source,
                        funding,
                    )
                })?;
                if let Some(parent) = generation.parent() {
                    sync_directory_with_funding(parent, funding)?;
                }
            }
        }
        if self.staging.exists() {
            fs::remove_dir_all(&self.staging).map_err(|source| {
                funded::io_error(
                    "remove rolled-back prompt cache staging directory",
                    &self.staging,
                    source,
                    funding,
                )
            })?;
        }
        Ok(())
    }
}

impl Drop for ReversiblePromptCachePublication {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.rollback_inner();
        }
    }
}

/// Reads and validates a prompt-cache manifest without loading tensor arrays.
pub fn inspect_prompt_cache(
    directory: impl AsRef<Path>,
) -> Result<PromptCacheManifest, PromptCachePersistenceError> {
    inspect_prompt_cache_with_funding(directory.as_ref(), None)
}
fn inspect_prompt_cache_with_funding(
    directory: &Path,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<PromptCacheManifest, PromptCachePersistenceError> {
    let directory = resolve_prompt_cache_root_with_funding(directory, funding)?;
    if let Some(funding) = funding {
        funding.path(&directory)?;
    }
    let manifest_path = funded::path_join(&directory, "manifest.json", funding)?;
    let input = funded::read_bounded(&manifest_path, funding.map(|f| f.manifest_limit), funding)?;
    let value: serde_json::Value =
        serde_json::from_slice(&input).map_err(PromptCachePersistenceError::ManifestJson)?;
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .ok_or_else(|| {
            match funded::text(
                format_args!("prompt-cache schema_version is missing or is not a u32"),
                funding,
            ) {
                Ok(message) => {
                    PromptCachePersistenceError::PromptCache(PromptCacheError::Malformed(message))
                }
                Err(error) => error,
            }
        })?;
    if schema_version != PROMPT_CACHE_SCHEMA_VERSION {
        return Err(PromptCacheError::UnsupportedSchema(schema_version).into());
    }
    let manifest =
        serde_json::from_value(value).map_err(PromptCachePersistenceError::ManifestJson)?;
    validate_prompt_cache_manifest_with_funding(&directory, &manifest, funding)?;
    Ok(manifest)
}

/// Resolves the active immutable generation selected by the durable pointer.
pub fn resolve_prompt_cache_root(directory: &Path) -> Result<PathBuf, PromptCachePersistenceError> {
    resolve_prompt_cache_root_with_funding(directory, None)
}
fn resolve_prompt_cache_root_with_funding(
    directory: &Path,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<PathBuf, PromptCachePersistenceError> {
    if let Some(funding) = funding {
        funding.path(directory)?;
    }
    let current_path = funded::path_join(&directory, PROMPT_CACHE_CURRENT_FILE, funding)?;
    let bytes =
        funded::read_bounded(&current_path, Some(256), funding).map_err(|error| match error {
            PromptCachePersistenceError::Io { source, .. }
                if source.kind() == std::io::ErrorKind::NotFound =>
            {
                funded::storage_error(
                    format_args!("prompt-cache generation pointer CURRENT is missing"),
                    funding,
                )
            }
            other => other,
        })?;
    if bytes.is_empty() {
        return Err(funded::storage_error(
            format_args!("prompt-cache generation pointer has an invalid length"),
            funding,
        ));
    }
    let generation = std::str::from_utf8(&bytes).map_err(|_| {
        funded::storage_error(
            format_args!("prompt-cache generation pointer is not UTF-8"),
            funding,
        )
    })?;
    let generation = generation.trim();
    let generation_path = Path::new(generation);
    if generation.is_empty()
        || generation_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || generation_path.components().count() != 1
    {
        return Err(funded::storage_error(
            format_args!("prompt-cache generation pointer is unsafe"),
            funding,
        ));
    }
    let root = funded::path_join(
        &funded::path_join(directory, PROMPT_CACHE_GENERATIONS_DIRECTORY, funding)?,
        generation_path,
        funding,
    )?;
    if !root.is_dir() {
        return Err(funded::storage_error(
            format_args!("prompt-cache generation {generation:?} is missing"),
            funding,
        ));
    }
    Ok(root)
}

/// Validates manifest structure and the bounded metadata of every referenced shard.
pub fn validate_prompt_cache_manifest(
    directory: &Path,
    manifest: &PromptCacheManifest,
) -> Result<(), PromptCachePersistenceError> {
    validate_prompt_cache_manifest_with_funding(directory, manifest, None)
}
fn validate_prompt_cache_manifest_with_funding(
    directory: &Path,
    manifest: &PromptCacheManifest,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<(), PromptCachePersistenceError> {
    if let Some(funding) = funding {
        funding.estimate(funded::source_bytes(manifest)?)?;
    }
    manifest.validate()?;
    for block in &manifest.blocks {
        if let Some(funding) = funding {
            funding.path(directory)?;
            funding.estimate(block.shard.len())?;
        }
        let shard = safe_prompt_cache_shard_path_with_funding(directory, &block.shard, funding)?;
        if !shard.is_file() {
            return Err(PromptCachePersistenceError::MissingShard(shard));
        }
        validate_block_shard(&shard, block, funding)?;
    }
    for state in &manifest.state_tensors {
        if let Some(funding) = funding {
            funding.path(directory)?;
            funding.estimate(state.shard.len())?;
        }
        let shard = safe_prompt_cache_shard_path_with_funding(directory, &state.shard, funding)?;
        if !shard.is_file() {
            return Err(PromptCachePersistenceError::MissingShard(shard));
        }
        validate_state_shard(&shard, state, funding)?;
    }
    Ok(())
}

/// Resolves a manifest shard path while rejecting traversal and symlink escapes.
pub fn safe_prompt_cache_shard_path(
    directory: &Path,
    relative: &str,
) -> Result<PathBuf, PromptCachePersistenceError> {
    safe_prompt_cache_shard_path_with_funding(directory, relative, None)
}
fn safe_prompt_cache_shard_path_with_funding(
    directory: &Path,
    relative: &str,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<PathBuf, PromptCachePersistenceError> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(PromptCachePersistenceError::UnsafeShardPath(funded::text(
            format_args!("{relative}"),
            funding,
        )?));
    }
    let joined = funded::path_join(directory, path, funding)?;
    if joined.exists() {
        if let Some(funding) = funding {
            funding.path(directory)?;
            funding.path(&joined)?;
        }
        let root = fs::canonicalize(directory).map_err(|source| {
            funded::io_error(
                "canonicalize prompt cache directory",
                &directory,
                source,
                funding,
            )
        })?;
        let canonical = fs::canonicalize(&joined).map_err(|source| {
            funded::io_error("canonicalize prompt cache shard", &joined, source, funding)
        })?;
        if !canonical.starts_with(&root) {
            return Err(PromptCachePersistenceError::UnsafeShardPath(funded::text(
                format_args!("{relative}"),
                funding,
            )?));
        }
    }
    Ok(joined)
}

/// Synchronizes a newly written shard and returns its exact payload SHA-256.
pub fn finalize_prompt_cache_shard(path: &Path) -> Result<String, PromptCachePersistenceError> {
    sync_file(path)?;
    hash_prompt_cache_shard_payload(path)
}

/// Hashes the safetensors payload bytes, excluding its bounded metadata header.
pub fn hash_prompt_cache_shard_payload(path: &Path) -> Result<String, PromptCachePersistenceError> {
    hash_prompt_cache_shard_payload_with_funding(path, None)
}
fn hash_prompt_cache_shard_payload_with_funding(
    path: &Path,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<String, PromptCachePersistenceError> {
    let (_, _, data_start) = read_shard_metadata_with_funding(path, funding)?;
    let mut file = File::open(path).map_err(|source| {
        funded::io_error("open prompt cache shard payload", &path, source, funding)
    })?;
    hash_shard_payload_from(&mut file, path, data_start, funding)
}
fn hash_shard_payload_from(
    file: &mut File,
    path: &Path,
    data_start: u64,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<String, PromptCachePersistenceError> {
    if let Some(funding) = funding {
        funding.buffer(64 * 1024)?;
        funding.buffer(64)?;
    }
    file.seek(SeekFrom::Start(data_start)).map_err(|source| {
        funded::io_error("seek prompt cache shard payload", &path, source, funding)
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|source| {
            funded::io_error("hash prompt cache shard payload", &path, source, funding)
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(hasher.finalize()))
}

fn validate_block_shard(
    path: &Path,
    block: &PromptCacheBlock,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<(), PromptCachePersistenceError> {
    let (metadata, file_len, data_start) = read_shard_metadata_with_funding(path, funding)?;
    let entries = metadata.tensors();
    if entries.len() != 2 {
        return Err(malformed_funded(
            path,
            format_args!("expected two arrays, found {}", entries.len()),
            funding,
        ));
    }
    let mut logical_bytes = 0u64;
    for (name, expected_shape, expected_dtype) in [
        (&block.first_array, &block.first_shape, &block.first_dtype),
        (
            &block.second_array,
            &block.second_shape,
            &block.second_dtype,
        ),
    ] {
        let tensor = metadata
            .info(name)
            .ok_or_else(|| malformed_funded(path, format_args!("missing array {name}"), funding))?;
        let shape_matches = tensor.shape.len() == expected_shape.len()
            && tensor
                .shape
                .iter()
                .zip(expected_shape)
                .all(|(actual, expected)| i32::try_from(*actual).ok() == Some(*expected));
        if !shape_matches || !stored_dtype_matches(tensor.dtype, expected_dtype) {
            return Err(malformed_funded(
                path,
                format_args!("array {name} shape or dtype does not match the manifest"),
                funding,
            ));
        }
        let bytes = tensor
            .data_offsets
            .1
            .checked_sub(tensor.data_offsets.0)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| {
                malformed_funded(path, format_args!("array byte span overflow"), funding)
            })?;
        logical_bytes = logical_bytes.checked_add(bytes).ok_or_else(|| {
            malformed_funded(path, format_args!("logical byte count overflow"), funding)
        })?;
    }
    if logical_bytes != block.logical_bytes {
        return Err(malformed_funded(
            path,
            format_args!(
                "logical byte count {logical_bytes} does not match manifest value {}",
                block.logical_bytes
            ),
            funding,
        ));
    }
    validate_file_boundary(path, &metadata, file_len, data_start, funding)
}

fn validate_state_shard(
    path: &Path,
    state: &PromptCacheStateTensor,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<(), PromptCachePersistenceError> {
    let (metadata, file_len, data_start) = read_shard_metadata_with_funding(path, funding)?;
    let entries = metadata.tensors();
    if entries.len() != 1 {
        return Err(malformed_funded(
            path,
            format_args!("expected one state array, found {}", entries.len()),
            funding,
        ));
    }
    let tensor = metadata.info(&state.array).ok_or_else(|| {
        malformed_funded(
            path,
            format_args!("missing state array {}", state.array),
            funding,
        )
    })?;
    let shape_matches = tensor.shape.len() == state.shape.len()
        && tensor
            .shape
            .iter()
            .zip(&state.shape)
            .all(|(actual, expected)| i32::try_from(*actual).ok() == Some(*expected));
    let logical_bytes = tensor
        .data_offsets
        .1
        .checked_sub(tensor.data_offsets.0)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| {
            malformed_funded(
                path,
                format_args!("state array byte span overflow"),
                funding,
            )
        })?;
    if !shape_matches
        || !stored_dtype_matches(tensor.dtype, &state.dtype)
        || logical_bytes != state.logical_bytes
    {
        return Err(malformed_funded(
            path,
            format_args!("state array shape, dtype, or byte count does not match the manifest"),
            funding,
        ));
    }
    validate_file_boundary(path, &metadata, file_len, data_start, funding)
}

fn validate_file_boundary(
    path: &Path,
    metadata: &safetensors::tensor::Metadata,
    file_len: u64,
    data_start: u64,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<(), PromptCachePersistenceError> {
    let expected_file_len = data_start
        .checked_add(u64::try_from(metadata.data_len()).map_err(|_| {
            malformed_funded(
                path,
                format_args!("safetensors data length overflow"),
                funding,
            )
        })?)
        .ok_or_else(|| {
            malformed_funded(
                path,
                format_args!("safetensors file length overflow"),
                funding,
            )
        })?;
    if expected_file_len != file_len {
        return Err(malformed_funded(
            path,
            format_args!(
                "safetensors payload boundary {expected_file_len} does not match file length {file_len}"
            ),
            funding,
        ));
    }
    Ok(())
}

fn read_shard_metadata(
    path: &Path,
) -> Result<(safetensors::tensor::Metadata, u64, u64), PromptCachePersistenceError> {
    read_shard_metadata_with_funding(path, None)
}
fn read_shard_metadata_with_funding(
    path: &Path,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<(safetensors::tensor::Metadata, u64, u64), PromptCachePersistenceError> {
    if let Some(funding) = funding {
        funding.path(path)?;
    }
    let mut file = File::open(path).map_err(|source| {
        funded::io_error("open prompt cache shard metadata", &path, source, funding)
    })?;
    let (metadata, header, file_len) = read_shard_metadata_from(&mut file, path, funding)?;
    Ok((
        metadata,
        u64::try_from(file_len).map_err(|_| {
            PromptCachePersistenceError::Metadata(
                eredu_nn::workspace::WorkspaceMetadataError::Overflow,
            )
        })?,
        u64::try_from(header.len()).map_err(|_| {
            PromptCachePersistenceError::Metadata(
                eredu_nn::workspace::WorkspaceMetadataError::Overflow,
            )
        })?,
    ))
}
fn read_shard_metadata_from(
    file: &mut File,
    path: &Path,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<(safetensors::tensor::Metadata, Vec<u8>, usize), PromptCachePersistenceError> {
    file.seek(SeekFrom::Start(0)).map_err(|source| {
        funded::io_error("seek prompt cache shard metadata", path, source, funding)
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| funded::io_error("stat prompt cache shard", &path, source, funding))?
        .len();
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes).map_err(|source| {
        funded::io_error(
            "read prompt cache shard header length",
            &path,
            source,
            funding,
        )
    })?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > MAX_PROMPT_CACHE_SHARD_HEADER_BYTES {
        return Err(malformed_funded(
            path,
            format_args!("safetensors header length {header_len} exceeds the prompt-cache bound"),
            funding,
        ));
    }
    let data_start = 8u64.checked_add(header_len).ok_or_else(|| {
        malformed_funded(
            path,
            format_args!("safetensors header length overflow"),
            funding,
        )
    })?;
    if data_start > file_len {
        return Err(malformed_funded(
            path,
            format_args!("safetensors header extends beyond the file"),
            funding,
        ));
    }
    let header_len = usize::try_from(header_len).map_err(|_| {
        PromptCachePersistenceError::Metadata(eredu_nn::workspace::WorkspaceMetadataError::Overflow)
    })?;
    if let Some(funding) = funding {
        funding.buffer(header_len.checked_add(8).ok_or(
            PromptCachePersistenceError::Metadata(
                eredu_nn::workspace::WorkspaceMetadataError::Overflow,
            ),
        )?)?;
        funding.estimate(header_len)?;
    }
    let total_header = header_len
        .checked_add(8)
        .ok_or(PromptCachePersistenceError::Metadata(
            eredu_nn::workspace::WorkspaceMetadataError::Overflow,
        ))?;
    let mut header = vec![0u8; total_header];
    header[..8].copy_from_slice(&length_bytes);
    file.read_exact(&mut header[8..]).map_err(|source| {
        funded::io_error("read prompt cache shard header", &path, source, funding)
    })?;
    let metadata = serde_json::from_slice(&header[8..])
        .map_err(|error| malformed_funded(path, format_args!("{error}"), funding))?;
    Ok((
        metadata,
        header,
        usize::try_from(file_len).map_err(|_| {
            PromptCachePersistenceError::Metadata(
                eredu_nn::workspace::WorkspaceMetadataError::Overflow,
            )
        })?,
    ))
}

fn malformed_funded(
    path: &Path,
    reason: std::fmt::Arguments<'_>,
    funding: Option<&funded::PersistenceAccounts>,
) -> PromptCachePersistenceError {
    let result = (|| {
        Ok::<_, PromptCachePersistenceError>(PromptCachePersistenceError::MalformedShard {
            path: funded::path_copy(path, funding)?,
            reason: funded::text(reason, funding)?,
        })
    })();
    result.unwrap_or_else(|error| error)
}

pub(super) fn stored_dtype_matches(dtype: safetensors::Dtype, expected: &str) -> bool {
    use safetensors::Dtype as Stored;
    let name = match dtype {
        Stored::BOOL => "Bool",
        Stored::U8 => "Uint8",
        Stored::U16 => "Uint16",
        Stored::U32 => "Uint32",
        Stored::U64 => "Uint64",
        Stored::I8 => "Int8",
        Stored::I16 => "Int16",
        Stored::I32 => "Int32",
        Stored::I64 => "Int64",
        Stored::F16 => "Float16",
        Stored::BF16 => "Bfloat16",
        Stored::F32 => "Float32",
        Stored::F64 => "Float64",
        dtype => {
            use std::fmt::Write as _;
            struct Equal<'a>(&'a str);
            impl std::fmt::Write for Equal<'_> {
                fn write_str(&mut self, value: &str) -> std::fmt::Result {
                    self.0 = self.0.strip_prefix(value).ok_or(std::fmt::Error)?;
                    Ok(())
                }
            }
            let mut compared = Equal(expected);
            return write!(&mut compared, "{dtype:?}").is_ok() && compared.0.is_empty();
        }
    };
    name == expected
}

fn publish_generation_pointer(
    destination: &Path,
    generation_name: &str,
    nonce: u128,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<(), PromptCachePersistenceError> {
    let temporary = funded::path_join(
        destination,
        funded::text(
            format_args!(".{PROMPT_CACHE_CURRENT_FILE}.tmp-{nonce}"),
            funding,
        )?,
        funding,
    )?;
    let current = funded::path_join(&destination, PROMPT_CACHE_CURRENT_FILE, funding)?;
    let mut file = File::create(&temporary).map_err(|source| {
        funded::io_error(
            "create prompt cache generation pointer",
            &temporary,
            source,
            funding,
        )
    })?;
    writeln!(file, "{generation_name}").map_err(|source| {
        funded::io_error(
            "write prompt cache generation pointer",
            &temporary,
            source,
            funding,
        )
    })?;
    file.sync_all().map_err(|source| {
        funded::io_error(
            "sync prompt cache generation pointer",
            &temporary,
            source,
            funding,
        )
    })?;
    durable_rename(&temporary, &current, true).map_err(|source| {
        PromptCachePersistenceError::Io {
            action: "switch prompt cache generation",
            path: current,
            source,
        }
    })?;
    Ok(())
}

fn sync_file(path: &Path) -> Result<(), PromptCachePersistenceError> {
    sync_file_with_funding(path, None)
}
fn sync_file_with_funding(
    path: &Path,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<(), PromptCachePersistenceError> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|source| funded::io_error("synchronize cache file", path, source, funding))
}

fn sync_directory(path: &Path) -> Result<(), PromptCachePersistenceError> {
    sync_directory_with_funding(path, None)
}
#[cfg(unix)]
fn sync_directory_with_funding(
    path: &Path,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<(), PromptCachePersistenceError> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|source| funded::io_error("synchronize cache directory", path, source, funding))
}
#[cfg(not(unix))]
fn sync_directory_with_funding(
    path: &Path,
    funding: Option<&funded::PersistenceAccounts>,
) -> Result<(), PromptCachePersistenceError> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(funded::io_error(
            "validate cache directory before publication",
            path,
            std::io::ErrorKind::NotADirectory.into(),
            funding,
        ))
    }
}

#[cfg(not(windows))]
fn durable_rename(source: &Path, destination: &Path, _replace: bool) -> std::io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn durable_rename(source: &Path, destination: &Path, _replace: bool) -> std::io::Result<()> {
    fs::rename(source, destination)
}

fn hex(digest: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = digest.as_ref();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for &byte in digest {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::cache::{
        CacheRepresentation, LayerCachePolicy, PromptCacheDescriptor, PromptCacheStateSegment,
        PromptCacheTopology,
    };
    use eredu_core::{AttentionPolicy, LayerSchedule};
    use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
    use std::collections::HashMap;

    pub(super) fn manifest(shard: &Path) -> PromptCacheManifest {
        let bytes = [0u8; 16];
        let tensor = TensorView::new(Dtype::F32, vec![1, 1, 2, 2], &bytes).unwrap();
        serialize_to_file(
            HashMap::from([("keys", tensor.clone()), ("values", tensor)]),
            None,
            shard,
        )
        .unwrap();
        let hash = finalize_prompt_cache_shard(shard).unwrap();
        let descriptor = PromptCacheDescriptor::new(
            "test",
            "test",
            "checkpoint",
            "content",
            "architecture",
            1,
            0,
            1,
            1,
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 2).unwrap()],
            )
            .unwrap(),
            vec![0],
            vec![PromptCacheStateSegment::new("state", 0..1).unwrap()],
            0,
            PromptCacheTopology::default(),
        )
        .unwrap();
        PromptCacheManifest {
            schema_version: PROMPT_CACHE_SCHEMA_VERSION,
            model_family: descriptor.model_family().into(),
            effective_model_type: descriptor.effective_model_type().into(),
            checkpoint_fingerprint: descriptor.checkpoint_fingerprint().into(),
            prefix_content_fingerprint: descriptor.prefix_content_fingerprint().into(),
            architecture_fingerprint: descriptor.architecture_fingerprint().into(),
            layer_count: 1,
            global_layer_start: 0,
            global_layer_end: 1,
            block_size_tokens: 2,
            batch_size: 1,
            total_prefix_tokens: 2,
            prefix_sha256: "00".repeat(32),
            layer_layout: descriptor.layer_layout().clone(),
            layer_prefix_offsets: vec![0],
            state_segments: descriptor.state_segments().to_vec(),
            sink_tokens: 0,
            topology: descriptor.topology().clone(),
            distributed_commit: None,
            application_namespace: None,
            blocks: vec![PromptCacheBlock {
                global_layer: 0,
                representation: CacheRepresentation::KeyValue,
                start: 0,
                end: 2,
                rank: None,
                shard: shard.file_name().unwrap().to_str().unwrap().into(),
                first_array: "keys".into(),
                second_array: "values".into(),
                first_shape: vec![1, 1, 2, 2],
                second_shape: vec![1, 1, 2, 2],
                first_dtype: "Float32".into(),
                second_dtype: "Float32".into(),
                logical_bytes: 32,
                payload_sha256: hash,
            }],
            state_tensors: vec![],
        }
    }

    #[test]
    fn publication_validates_and_atomically_replaces_generations() {
        let root = tempfile::tempdir().unwrap();
        let destination = root.path().join("cache");
        let publication = PromptCachePublication::begin(&destination, false).unwrap();
        let first = manifest(&publication.staging_directory().join("block.safetensors"));
        publication.commit(&first).unwrap();
        assert_eq!(inspect_prompt_cache(&destination).unwrap(), first);
        assert!(destination.join(PROMPT_CACHE_CURRENT_FILE).is_file());
        let first_root = resolve_prompt_cache_root(&destination).unwrap();
        assert_eq!(
            first_root.parent().unwrap(),
            destination.join(PROMPT_CACHE_GENERATIONS_DIRECTORY)
        );
        assert!(first_root.join("manifest.json").is_file());

        let publication = PromptCachePublication::begin(&destination, true).unwrap();
        let second = manifest(&publication.staging_directory().join("block.safetensors"));
        publication.commit(&second).unwrap();
        assert_eq!(inspect_prompt_cache(&destination).unwrap(), second);
        assert!(destination.join(PROMPT_CACHE_CURRENT_FILE).is_file());
    }

    #[test]
    fn reversible_publication_restores_replacement_and_removes_new_destination() {
        fn prepare_cache(destination: &Path, label: &str) -> PromptCacheManifest {
            let publication = PromptCachePublication::begin(destination, false).unwrap();
            let mut manifest = manifest(&publication.staging_directory().join("block.safetensors"));
            manifest.application_namespace = Some(label.into());
            publication.commit(&manifest).unwrap();
            manifest
        }

        let root = tempfile::tempdir().unwrap();
        let destination = root.path().join("cache");
        let mut fresh = ReversiblePromptCachePublication::begin(&destination, false).unwrap();
        let fresh_manifest = prepare_cache(fresh.staging_destination(), "fresh");
        fresh.publish().unwrap();
        assert_eq!(inspect_prompt_cache(&destination).unwrap(), fresh_manifest);
        fresh.rollback().unwrap();
        assert!(!destination.exists());

        let first = prepare_cache(&destination, "first");
        let first_root = resolve_prompt_cache_root(&destination).unwrap();
        let mut replacement = ReversiblePromptCachePublication::begin(&destination, true).unwrap();
        let second = prepare_cache(replacement.staging_destination(), "second");
        replacement.publish().unwrap();
        assert_eq!(inspect_prompt_cache(&destination).unwrap(), second);
        replacement.rollback().unwrap();
        assert_eq!(inspect_prompt_cache(&destination).unwrap(), first);
        assert_eq!(resolve_prompt_cache_root(&destination).unwrap(), first_root);

        let mut committed = ReversiblePromptCachePublication::begin(&destination, true).unwrap();
        let third = prepare_cache(committed.staging_destination(), "third");
        committed.publish().unwrap();
        committed.commit().unwrap();
        assert_eq!(inspect_prompt_cache(&destination).unwrap(), third);
    }

    #[test]
    fn pointerless_legacy_prompt_cache_layout_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let destination = root.path().join("cache");
        fs::create_dir(&destination).unwrap();
        let legacy = manifest(&destination.join("block.safetensors"));
        serde_json::to_writer(
            File::create(destination.join("manifest.json")).unwrap(),
            &legacy,
        )
        .unwrap();

        for result in [
            resolve_prompt_cache_root(&destination).map(|_| ()),
            inspect_prompt_cache(&destination).map(|_| ()),
        ] {
            assert!(matches!(
                result,
                Err(PromptCachePersistenceError::MalformedStorage(reason))
                    if reason == "prompt-cache generation pointer CURRENT is missing"
            ));
        }
    }

    #[test]
    fn failed_publication_removes_staging_directory() {
        let root = tempfile::tempdir().unwrap();
        let destination = root.path().join("cache");
        let staging = {
            let publication = PromptCachePublication::begin(&destination, false).unwrap();
            publication.staging_directory().to_path_buf()
        };
        assert!(!staging.exists());
    }

    #[test]
    fn shard_paths_reject_traversal() {
        let root = Path::new("/tmp/cache");
        assert_eq!(
            safe_prompt_cache_shard_path(root, "block.safetensors").unwrap(),
            root.join("block.safetensors")
        );
        assert!(matches!(
            safe_prompt_cache_shard_path(root, "../outside.safetensors"),
            Err(PromptCachePersistenceError::UnsafeShardPath(_))
        ));
        assert!(safe_prompt_cache_shard_path(root, "/outside.safetensors").is_err());
    }

    #[test]
    fn malformed_manifest_is_rejected_before_tensor_loading() {
        let directory = tempfile::tempdir().unwrap();
        let generation = directory
            .path()
            .join(PROMPT_CACHE_GENERATIONS_DIRECTORY)
            .join("generation-test");
        fs::create_dir_all(&generation).unwrap();
        fs::write(generation.join("manifest.json"), b"{not-json").unwrap();
        fs::write(
            directory.path().join(PROMPT_CACHE_CURRENT_FILE),
            b"generation-test\n",
        )
        .unwrap();
        assert!(matches!(
            inspect_prompt_cache(directory.path()),
            Err(PromptCachePersistenceError::ManifestJson(_))
        ));
    }

    #[test]
    fn shard_metadata_reads_are_bounded() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("oversized.safetensors");
        fs::write(
            &path,
            (MAX_PROMPT_CACHE_SHARD_HEADER_BYTES + 1).to_le_bytes(),
        )
        .unwrap();
        assert!(matches!(
            hash_prompt_cache_shard_payload(&path),
            Err(PromptCachePersistenceError::MalformedShard { .. })
        ));
    }

    #[test]
    fn live_cache_publication_is_unique_rank_aware_and_atomic() {
        let directory = tempfile::tempdir().unwrap();
        let id = CacheBlockId {
            session_id: 7,
            global_layer: 3,
            representation: CacheRepresentation::KeyValue,
            start: 4,
            end: 8,
            rank: Some(eredu_core::cache::CacheRankIdentity::new(
                Some(1),
                Some(2),
                None,
            )),
        };
        let first = LiveCacheBlockPublication::begin(directory.path(), &id);
        let second = LiveCacheBlockPublication::begin(directory.path(), &id);
        assert_ne!(first.destination_path(), second.destination_path());
        assert!(
            first
                .destination_path()
                .to_string_lossy()
                .contains("layer-00003-kv-rank-p1-t2-ex-4-8")
        );

        fs::write(first.staging_path(), b"block").unwrap();
        let destination = first.commit().unwrap();
        assert_eq!(fs::read(destination).unwrap(), b"block");
    }

    #[test]
    fn live_cache_publication_cleans_staging_and_never_replaces() {
        let directory = tempfile::tempdir().unwrap();
        let id = CacheBlockId {
            session_id: 1,
            global_layer: 0,
            representation: CacheRepresentation::CompressedLatentRotary,
            start: 0,
            end: 1,
            rank: None,
        };
        let abandoned = LiveCacheBlockPublication::begin(directory.path(), &id);
        let abandoned_path = abandoned.staging_path().to_path_buf();
        fs::write(&abandoned_path, b"temporary").unwrap();
        drop(abandoned);
        assert!(!abandoned_path.exists());

        let colliding = LiveCacheBlockPublication::begin(directory.path(), &id);
        fs::write(colliding.staging_path(), b"new").unwrap();
        fs::write(colliding.destination_path(), b"existing").unwrap();
        let destination = colliding.destination_path().to_path_buf();
        assert!(colliding.commit().is_err());
        assert_eq!(fs::read(destination).unwrap(), b"existing");
    }
}

#[path = "persistence/live_prepared.rs"]
mod live_prepared;
pub use live_prepared::{PreparedLiveCachePublication, PreparedLiveCachePublicationFailure};

#[path = "persistence/live_read.rs"]
mod live_read;
pub use live_read::{LiveCacheReadFailure, PreparedLiveCacheRead};

#[path = "persistence/shard.rs"]
mod shard;
pub use shard::{
    CacheShardError, CacheShardLayout, CacheShardMetadata, CacheShardTensor,
    cache_shard_tensor_names,
};

#[path = "persistence/persistent.rs"]
mod persistent;
pub use persistent::{
    PersistentCacheBlockSource, PersistentCacheReadFailure, PersistentCacheStateTensor,
    PreparedPersistentCacheRead,
};
