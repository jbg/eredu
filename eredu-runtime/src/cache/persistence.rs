//! Backend-neutral prompt-cache catalog validation and durable publication.

use eredu_core::cache::{
    CacheBlockId, CacheRepresentation, PromptCacheBlock, PromptCacheError, PromptCacheManifest,
    PromptCacheStateTensor, PROMPT_CACHE_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
    time::{SystemTime, UNIX_EPOCH},
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

/// Filesystem, catalog, and publication failures for a reusable prompt cache.
#[derive(Debug, thiserror::Error)]
pub enum PromptCachePersistenceError {
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
        let process_namespace = LIVE_CACHE_PROCESS_NAMESPACE.get_or_init(|| {
            let started = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            format!("p{:08x}-t{started:032x}", std::process::id())
        });
        let publication_id = NEXT_LIVE_CACHE_PUBLICATION_ID.fetch_add(1, Ordering::Relaxed);
        let representation = match id.representation {
            CacheRepresentation::KeyValue => "kv",
            CacheRepresentation::CompressedLatentRotary => "mla",
        };
        let rank_component =
            |rank: Option<usize>| rank.map_or_else(|| "x".to_string(), |rank| rank.to_string());
        let rank = id.rank.map_or_else(
            || "rank-px-tx-ex".to_string(),
            |rank| {
                format!(
                    "rank-p{}-t{}-e{}",
                    rank_component(rank.stage_rank()),
                    rank_component(rank.shard_rank()),
                    rank_component(rank.addressable_rank())
                )
            },
        );
        let base = format!(
            "live-{process_namespace}-w{publication_id:016x}-s{:016x}-layer-{:05}-{representation}-{rank}-{}-{}",
            id.session_id, id.global_layer, id.start, id.end
        );
        Self {
            destination: directory.join(format!("{base}.safetensors")),
            staging: directory.join(format!(".{base}.tmp.safetensors")),
            committed: false,
        }
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
        fs::hard_link(&self.staging, &self.destination).map_err(|source| {
            LiveCachePublicationError::Io {
                action: "publish uniquely named live cache block",
                path: self.destination.clone(),
                source,
            }
        })?;
        if let Err(source) = fs::remove_file(&self.staging) {
            let _ = fs::remove_file(&self.destination);
            return Err(LiveCachePublicationError::Io {
                action: "remove published live cache temporary file",
                path: self.staging.clone(),
                source,
            });
        }
        self.committed = true;
        Ok(self.destination.clone())
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
        let destination = destination.as_ref().to_path_buf();
        let parent = destination
            .parent()
            .ok_or_else(|| {
                PromptCachePersistenceError::InvalidPromptCachePath(destination.clone())
            })?
            .to_path_buf();
        fs::create_dir_all(&parent).map_err(|source| PromptCachePersistenceError::Io {
            action: "create prompt cache parent",
            path: parent.clone(),
            source,
        })?;
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
            .ok_or_else(|| {
                PromptCachePersistenceError::InvalidPromptCachePath(destination.clone())
            })?;
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let generation_name = format!("generation-{nonce}");
        let (generations, staging, publication_root) = if replacing {
            let generations = destination.join(PROMPT_CACHE_GENERATIONS_DIRECTORY);
            fs::create_dir_all(&generations).map_err(|source| PromptCachePersistenceError::Io {
                action: "create prompt cache generation directory",
                path: generations.clone(),
                source,
            })?;
            let staging = generations.join(format!(".tmp-{nonce}"));
            fs::create_dir(&staging).map_err(|source| PromptCachePersistenceError::Io {
                action: "create temporary prompt cache",
                path: staging.clone(),
                source,
            })?;
            (generations, staging, None)
        } else {
            let publication_root = parent.join(format!(".{file_name}.tmp-{nonce}"));
            fs::create_dir(&publication_root).map_err(|source| {
                PromptCachePersistenceError::Io {
                    action: "create temporary prompt cache root",
                    path: publication_root.clone(),
                    source,
                }
            })?;
            let generations = publication_root.join(PROMPT_CACHE_GENERATIONS_DIRECTORY);
            if let Err(source) = fs::create_dir(&generations) {
                let _ = fs::remove_dir_all(&publication_root);
                return Err(PromptCachePersistenceError::Io {
                    action: "create prompt cache generation directory",
                    path: generations,
                    source,
                });
            }
            let staging = generations.join(&generation_name);
            if let Err(source) = fs::create_dir(&staging) {
                let _ = fs::remove_dir_all(&publication_root);
                return Err(PromptCachePersistenceError::Io {
                    action: "create temporary prompt cache",
                    path: staging,
                    source,
                });
            }
            (generations, staging, Some(publication_root))
        };
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
    pub fn commit(
        mut self,
        manifest: &PromptCacheManifest,
    ) -> Result<(), PromptCachePersistenceError> {
        let manifest_path = self.staging.join("manifest.json");
        let file =
            File::create(&manifest_path).map_err(|source| PromptCachePersistenceError::Io {
                action: "create prompt cache manifest",
                path: manifest_path.clone(),
                source,
            })?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, manifest)
            .map_err(PromptCachePersistenceError::ManifestJson)?;
        writer
            .write_all(b"\n")
            .map_err(|source| PromptCachePersistenceError::Io {
                action: "write prompt cache manifest",
                path: manifest_path.clone(),
                source,
            })?;
        writer
            .flush()
            .map_err(|source| PromptCachePersistenceError::Io {
                action: "flush prompt cache manifest",
                path: manifest_path.clone(),
                source,
            })?;
        sync_file(&manifest_path)?;
        validate_prompt_cache_manifest(&self.staging, manifest)?;
        sync_directory(&self.staging)?;

        if self.replacing {
            let generation = self.generations.join(&self.generation_name);
            durable_rename(&self.staging, &generation, false).map_err(|source| {
                PromptCachePersistenceError::Io {
                    action: "publish prompt cache generation",
                    path: generation,
                    source,
                }
            })?;
            sync_directory(&self.generations)?;
            publish_generation_pointer(&self.destination, &self.generation_name, self.nonce)?;
        } else {
            sync_directory(&self.generations)?;
            let publication_root = self
                .publication_root
                .as_ref()
                .expect("new prompt-cache publication owns a staging root");
            publish_generation_pointer(publication_root, &self.generation_name, self.nonce)?;
            durable_rename(publication_root, &self.destination, false).map_err(|source| {
                PromptCachePersistenceError::Io {
                    action: "publish prompt cache",
                    path: self.destination.clone(),
                    source,
                }
            })?;
        }
        sync_directory(&self.parent)?;
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
}

impl ReversiblePromptCachePublication {
    /// Reserves a unique sibling destination without creating or publishing it.
    pub fn begin(
        destination: impl AsRef<Path>,
        replace_existing: bool,
    ) -> Result<Self, PromptCachePersistenceError> {
        let destination = destination.as_ref().to_path_buf();
        let parent = destination
            .parent()
            .ok_or_else(|| {
                PromptCachePersistenceError::InvalidPromptCachePath(destination.clone())
            })?
            .to_path_buf();
        fs::create_dir_all(&parent).map_err(|source| PromptCachePersistenceError::Io {
            action: "create reversible prompt cache parent",
            path: parent.clone(),
            source,
        })?;
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
            .ok_or_else(|| {
                PromptCachePersistenceError::InvalidPromptCachePath(destination.clone())
            })?;
        let publication_id = NEXT_REVERSIBLE_CACHE_PUBLICATION_ID.fetch_add(1, Ordering::Relaxed);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            ^ u128::from(publication_id);
        let staging = parent.join(format!(
            ".{file_name}.transaction-p{:08x}-{nonce:032x}",
            std::process::id()
        ));
        if staging.exists() {
            return Err(PromptCachePersistenceError::InvalidPromptCachePath(staging));
        }
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
        })
    }

    /// Hidden destination into which an ordinary cache save must be completed.
    pub fn staging_destination(&self) -> &Path {
        &self.staging
    }

    /// Makes the prepared cache visible while retaining an exact rollback path.
    pub fn publish(&mut self) -> Result<(), PromptCachePersistenceError> {
        if self.published || self.moved_generation.is_some() {
            return Err(PromptCachePersistenceError::InvalidReversiblePublication(
                "publication was already attempted",
            ));
        }
        inspect_prompt_cache(&self.staging)?;
        if !self.destination.exists() {
            durable_rename(&self.staging, &self.destination, false).map_err(|source| {
                PromptCachePersistenceError::Io {
                    action: "publish prepared prompt cache",
                    path: self.destination.clone(),
                    source,
                }
            })?;
            sync_directory(&self.parent)?;
            self.published = true;
            return Ok(());
        }
        if !self.replace_existing {
            return Err(PromptCachePersistenceError::PromptCacheExists(
                self.destination.clone(),
            ));
        }

        let previous = resolve_prompt_cache_root(&self.destination)?;
        let previous_generation = previous
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                PromptCachePersistenceError::MalformedStorage(
                    "active prompt-cache generation has no safe name".into(),
                )
            })?
            .to_owned();
        let staged = resolve_prompt_cache_root(&self.staging)?;
        let generation_name = staged
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                PromptCachePersistenceError::MalformedStorage(
                    "prepared prompt-cache generation has no safe name".into(),
                )
            })?
            .to_owned();
        let generations = self.destination.join(PROMPT_CACHE_GENERATIONS_DIRECTORY);
        let target = generations.join(&generation_name);
        if target.exists() {
            return Err(PromptCachePersistenceError::InvalidPromptCachePath(target));
        }
        durable_rename(&staged, &target, false).map_err(|source| {
            PromptCachePersistenceError::Io {
                action: "install prepared prompt cache generation",
                path: target.clone(),
                source,
            }
        })?;
        self.previous_generation = Some(previous_generation);
        self.moved_generation = Some(target);
        sync_directory(&generations)?;
        publish_generation_pointer(&self.destination, &generation_name, self.nonce)?;
        self.published = true;
        Ok(())
    }

    /// Accepts the visible cache and releases rollback metadata.
    pub fn commit(mut self) -> Result<(), PromptCachePersistenceError> {
        if !self.published {
            return Err(PromptCachePersistenceError::InvalidReversiblePublication(
                "an unpublished cache cannot commit",
            ));
        }
        if self.staging.exists() {
            fs::remove_dir_all(&self.staging).map_err(|source| {
                PromptCachePersistenceError::Io {
                    action: "remove committed prompt cache staging directory",
                    path: self.staging.clone(),
                    source,
                }
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
        if self.published {
            match self.previous_generation.as_deref() {
                Some(previous) => {
                    publish_generation_pointer(&self.destination, previous, self.nonce ^ 1)?;
                }
                None if self.destination.exists() => {
                    fs::remove_dir_all(&self.destination).map_err(|source| {
                        PromptCachePersistenceError::Io {
                            action: "remove rolled-back prompt cache",
                            path: self.destination.clone(),
                            source,
                        }
                    })?;
                    sync_directory(&self.parent)?;
                }
                None => {}
            }
        }
        if let Some(generation) = self.moved_generation.take() {
            if generation.exists() {
                fs::remove_dir_all(&generation).map_err(|source| {
                    PromptCachePersistenceError::Io {
                        action: "remove rolled-back prompt cache generation",
                        path: generation.clone(),
                        source,
                    }
                })?;
                if let Some(parent) = generation.parent() {
                    sync_directory(parent)?;
                }
            }
        }
        if self.staging.exists() {
            fs::remove_dir_all(&self.staging).map_err(|source| {
                PromptCachePersistenceError::Io {
                    action: "remove rolled-back prompt cache staging directory",
                    path: self.staging.clone(),
                    source,
                }
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
    let directory = resolve_prompt_cache_root(directory.as_ref())?;
    let manifest_path = directory.join("manifest.json");
    let reader = BufReader::new(File::open(&manifest_path).map_err(|source| {
        PromptCachePersistenceError::Io {
            action: "open prompt cache manifest",
            path: manifest_path.clone(),
            source,
        }
    })?);
    let value: serde_json::Value =
        serde_json::from_reader(reader).map_err(PromptCachePersistenceError::ManifestJson)?;
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .ok_or_else(|| {
            PromptCachePersistenceError::PromptCache(PromptCacheError::Malformed(
                "prompt-cache schema_version is missing or is not a u32".into(),
            ))
        })?;
    if schema_version != PROMPT_CACHE_SCHEMA_VERSION {
        return Err(PromptCacheError::UnsupportedSchema(schema_version).into());
    }
    let manifest =
        serde_json::from_value(value).map_err(PromptCachePersistenceError::ManifestJson)?;
    validate_prompt_cache_manifest(&directory, &manifest)?;
    Ok(manifest)
}

/// Resolves the active immutable generation selected by the durable pointer.
pub fn resolve_prompt_cache_root(directory: &Path) -> Result<PathBuf, PromptCachePersistenceError> {
    let current_path = directory.join(PROMPT_CACHE_CURRENT_FILE);
    let metadata = current_path.metadata().map_err(|source| {
        if source.kind() == std::io::ErrorKind::NotFound {
            PromptCachePersistenceError::MalformedStorage(
                "prompt-cache generation pointer CURRENT is missing".into(),
            )
        } else {
            PromptCachePersistenceError::Io {
                action: "stat prompt cache generation pointer",
                path: current_path.clone(),
                source,
            }
        }
    })?;
    let length = metadata.len();
    if length == 0 || length > 256 {
        return Err(PromptCachePersistenceError::MalformedStorage(
            "prompt-cache generation pointer has an invalid length".into(),
        ));
    }
    let generation =
        fs::read_to_string(&current_path).map_err(|source| PromptCachePersistenceError::Io {
            action: "read prompt cache generation pointer",
            path: current_path.clone(),
            source,
        })?;
    let generation = generation.trim();
    let generation_path = Path::new(generation);
    if generation.is_empty()
        || generation_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || generation_path.components().count() != 1
    {
        return Err(PromptCachePersistenceError::MalformedStorage(
            "prompt-cache generation pointer is unsafe".into(),
        ));
    }
    let root = directory
        .join(PROMPT_CACHE_GENERATIONS_DIRECTORY)
        .join(generation_path);
    if !root.is_dir() {
        return Err(PromptCachePersistenceError::MalformedStorage(format!(
            "prompt-cache generation {generation:?} is missing"
        )));
    }
    Ok(root)
}

/// Validates manifest structure and the bounded metadata of every referenced shard.
pub fn validate_prompt_cache_manifest(
    directory: &Path,
    manifest: &PromptCacheManifest,
) -> Result<(), PromptCachePersistenceError> {
    manifest.validate()?;
    for block in &manifest.blocks {
        let shard = safe_prompt_cache_shard_path(directory, &block.shard)?;
        if !shard.is_file() {
            return Err(PromptCachePersistenceError::MissingShard(shard));
        }
        validate_block_shard(&shard, block)?;
    }
    for state in &manifest.state_tensors {
        let shard = safe_prompt_cache_shard_path(directory, &state.shard)?;
        if !shard.is_file() {
            return Err(PromptCachePersistenceError::MissingShard(shard));
        }
        validate_state_shard(&shard, state)?;
    }
    Ok(())
}

/// Resolves a manifest shard path while rejecting traversal and symlink escapes.
pub fn safe_prompt_cache_shard_path(
    directory: &Path,
    relative: &str,
) -> Result<PathBuf, PromptCachePersistenceError> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(PromptCachePersistenceError::UnsafeShardPath(
            relative.into(),
        ));
    }
    let joined = directory.join(path);
    if joined.exists() {
        let root =
            fs::canonicalize(directory).map_err(|source| PromptCachePersistenceError::Io {
                action: "canonicalize prompt cache directory",
                path: directory.to_path_buf(),
                source,
            })?;
        let canonical =
            fs::canonicalize(&joined).map_err(|source| PromptCachePersistenceError::Io {
                action: "canonicalize prompt cache shard",
                path: joined.clone(),
                source,
            })?;
        if !canonical.starts_with(&root) {
            return Err(PromptCachePersistenceError::UnsafeShardPath(
                relative.into(),
            ));
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
    let (_, _, data_start) = read_shard_metadata(path)?;
    let mut file = File::open(path).map_err(|source| PromptCachePersistenceError::Io {
        action: "open prompt cache shard payload",
        path: path.to_path_buf(),
        source,
    })?;
    file.seek(SeekFrom::Start(data_start))
        .map_err(|source| PromptCachePersistenceError::Io {
            action: "seek prompt cache shard payload",
            path: path.to_path_buf(),
            source,
        })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| PromptCachePersistenceError::Io {
                action: "hash prompt cache shard payload",
                path: path.to_path_buf(),
                source,
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
) -> Result<(), PromptCachePersistenceError> {
    let (metadata, file_len, data_start) = read_shard_metadata(path)?;
    let entries = metadata.tensors();
    if entries.len() != 2 {
        return Err(malformed(
            path,
            format!("expected two arrays, found {}", entries.len()),
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
            .ok_or_else(|| malformed(path, format!("missing array {name}")))?;
        let shape = tensor
            .shape
            .iter()
            .map(|dimension| i32::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| malformed(path, "array dimension exceeds runtime range"))?;
        if &shape != expected_shape || stored_dtype_name(tensor.dtype) != *expected_dtype {
            return Err(malformed(
                path,
                format!("array {name} shape or dtype does not match the manifest"),
            ));
        }
        logical_bytes = logical_bytes.saturating_add(
            u64::try_from(tensor.data_offsets.1.saturating_sub(tensor.data_offsets.0))
                .unwrap_or(u64::MAX),
        );
    }
    if logical_bytes != block.logical_bytes {
        return Err(malformed(
            path,
            format!(
                "logical byte count {logical_bytes} does not match manifest value {}",
                block.logical_bytes
            ),
        ));
    }
    validate_file_boundary(path, &metadata, file_len, data_start)
}

fn validate_state_shard(
    path: &Path,
    state: &PromptCacheStateTensor,
) -> Result<(), PromptCachePersistenceError> {
    let (metadata, file_len, data_start) = read_shard_metadata(path)?;
    let entries = metadata.tensors();
    if entries.len() != 1 {
        return Err(malformed(
            path,
            format!("expected one state array, found {}", entries.len()),
        ));
    }
    let tensor = metadata
        .info(&state.array)
        .ok_or_else(|| malformed(path, format!("missing state array {}", state.array)))?;
    let shape = tensor
        .shape
        .iter()
        .map(|dimension| i32::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| malformed(path, "state array dimension exceeds runtime range"))?;
    let logical_bytes = u64::try_from(tensor.data_offsets.1.saturating_sub(tensor.data_offsets.0))
        .unwrap_or(u64::MAX);
    if shape != state.shape
        || stored_dtype_name(tensor.dtype) != state.dtype
        || logical_bytes != state.logical_bytes
    {
        return Err(malformed(
            path,
            "state array shape, dtype, or byte count does not match the manifest",
        ));
    }
    validate_file_boundary(path, &metadata, file_len, data_start)
}

fn validate_file_boundary(
    path: &Path,
    metadata: &safetensors::tensor::Metadata,
    file_len: u64,
    data_start: u64,
) -> Result<(), PromptCachePersistenceError> {
    let expected_file_len = data_start
        .checked_add(metadata.data_len() as u64)
        .ok_or_else(|| malformed(path, "safetensors file length overflow"))?;
    if expected_file_len != file_len {
        return Err(malformed(
            path,
            format!(
                "safetensors payload boundary {expected_file_len} does not match file length {file_len}"
            ),
        ));
    }
    Ok(())
}

fn read_shard_metadata(
    path: &Path,
) -> Result<(safetensors::tensor::Metadata, u64, u64), PromptCachePersistenceError> {
    let mut file = File::open(path).map_err(|source| PromptCachePersistenceError::Io {
        action: "open prompt cache shard metadata",
        path: path.to_path_buf(),
        source,
    })?;
    let file_len = file
        .metadata()
        .map_err(|source| PromptCachePersistenceError::Io {
            action: "stat prompt cache shard",
            path: path.to_path_buf(),
            source,
        })?
        .len();
    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|source| PromptCachePersistenceError::Io {
            action: "read prompt cache shard header length",
            path: path.to_path_buf(),
            source,
        })?;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > MAX_PROMPT_CACHE_SHARD_HEADER_BYTES {
        return Err(malformed(
            path,
            format!("safetensors header length {header_len} exceeds the prompt-cache bound"),
        ));
    }
    let data_start = 8u64
        .checked_add(header_len)
        .ok_or_else(|| malformed(path, "safetensors header length overflow"))?;
    if data_start > file_len {
        return Err(malformed(
            path,
            "safetensors header extends beyond the file",
        ));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|source| PromptCachePersistenceError::Io {
            action: "read prompt cache shard header",
            path: path.to_path_buf(),
            source,
        })?;
    let metadata =
        serde_json::from_slice(&header).map_err(|error| malformed(path, error.to_string()))?;
    Ok((metadata, file_len, data_start))
}

fn malformed(path: &Path, reason: impl Into<String>) -> PromptCachePersistenceError {
    PromptCachePersistenceError::MalformedShard {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

fn stored_dtype_name(dtype: safetensors::Dtype) -> String {
    use safetensors::Dtype as Stored;
    match dtype {
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
        dtype => return format!("{dtype:?}"),
    }
    .into()
}

fn publish_generation_pointer(
    destination: &Path,
    generation_name: &str,
    nonce: u128,
) -> Result<(), PromptCachePersistenceError> {
    let temporary = destination.join(format!(".{PROMPT_CACHE_CURRENT_FILE}.tmp-{nonce}"));
    let current = destination.join(PROMPT_CACHE_CURRENT_FILE);
    let mut file = File::create(&temporary).map_err(|source| PromptCachePersistenceError::Io {
        action: "create prompt cache generation pointer",
        path: temporary.clone(),
        source,
    })?;
    writeln!(file, "{generation_name}").map_err(|source| PromptCachePersistenceError::Io {
        action: "write prompt cache generation pointer",
        path: temporary.clone(),
        source,
    })?;
    file.sync_all()
        .map_err(|source| PromptCachePersistenceError::Io {
            action: "sync prompt cache generation pointer",
            path: temporary.clone(),
            source,
        })?;
    durable_rename(&temporary, &current, true).map_err(|source| {
        PromptCachePersistenceError::Io {
            action: "switch prompt cache generation",
            path: current,
            source,
        }
    })?;
    sync_directory(destination)
}

fn sync_file(path: &Path) -> Result<(), PromptCachePersistenceError> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|source| PromptCachePersistenceError::Io {
            action: "synchronize cache file",
            path: path.to_path_buf(),
            source,
        })
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), PromptCachePersistenceError> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|source| PromptCachePersistenceError::Io {
            action: "synchronize cache directory",
            path: path.to_path_buf(),
            source,
        })
}

#[cfg(windows)]
fn sync_directory(path: &Path) -> Result<(), PromptCachePersistenceError> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(PromptCachePersistenceError::Io {
            action: "validate cache directory before durable publication",
            path: path.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotADirectory,
                "cache publication path is not a directory",
            ),
        })
    }
}

#[cfg(not(any(unix, windows)))]
fn sync_directory(path: &Path) -> Result<(), PromptCachePersistenceError> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(PromptCachePersistenceError::Io {
            action: "validate cache directory before publication",
            path: path.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotADirectory,
                "cache publication path is not a directory",
            ),
        })
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
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use std::collections::HashMap;

    fn manifest(shard: &Path) -> PromptCacheManifest {
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
        assert!(first
            .destination_path()
            .to_string_lossy()
            .contains("layer-00003-kv-rank-p1-t2-ex-4-8"));

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
