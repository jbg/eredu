//! Separate accounted destinations and stock-dependency allowances for persistence.
use super::*;
use crate::working_memory::DependencyMemoryPolicy;
use eredu_core::HostMetadataFunding;
use eredu_core::cache::{PreparedPromptCacheManifest, SharedPromptCacheManifest};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataError};
use std::mem::{size_of, size_of_val};

/// Default admitted manifest input limit (64 MiB), independent of parser memory.
/// Low-level callers select another finite extent in `PromptCachePersistenceFunding::new`.
/// Caller-managed plain inspection is not restricted by this default.
pub const DEFAULT_PROMPT_CACHE_MANIFEST_BYTE_LIMIT: usize = 64 * 1024 * 1024;

/// Diagnostic source of the separately reserved dependency estimate.
pub const PROMPT_CACHE_DEPENDENCY_SOURCE: &str = "prompt-cache persistence dependency metadata";
/// Derivation of the finite allowance; it is not a measured or guaranteed bound.
pub const PROMPT_CACHE_DEPENDENCY_BASIS: &str = "stock serde_json and safetensors metadata, catalog validation containers and filesystem path scratch (serialized or retained diagnostic source bytes): fixed plus proportional allowance per bounded metadata input";

/// Two independently admitted payers for the same persistence worker.
/// The dependency account must come from an estimated-overhead reservation;
/// construction funding pays controlled byte buffers and retained output shells.
#[derive(Clone, Debug)]
pub struct PromptCachePersistenceFunding {
    context: WorkspaceContext,
    pub(super) accounts: PersistenceAccounts,
    storage: Option<crate::working_memory::StorageMetadataFunding>,
}
#[derive(Clone, Debug)]
pub(super) struct PersistenceAccounts {
    construction: HostMetadataFunding,
    dependency: HostMetadataFunding,
    policy: DependencyMemoryPolicy,
    pub(super) manifest_limit: usize,
    prepaid_rollback: bool,
}
impl PromptCachePersistenceFunding {
    /// Retains actual prepaid accounts without manufacturing an allowance.
    /// `manifest_limit` bounds each manifest input from the opened file handle.
    pub fn new(
        context: &WorkspaceContext,
        dependency: HostMetadataFunding,
        policy: DependencyMemoryPolicy,
        manifest_limit: usize,
    ) -> Result<Self, WorkspaceMetadataError> {
        let construction = context
            .metadata_funding()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        if construction.same_account(&dependency) {
            return Err(WorkspaceMetadataError::Unqualified);
        }
        context.charge_metadata(Self::control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        Ok(Self {
            context: context.clone(),
            accounts: PersistenceAccounts {
                construction,
                dependency,
                policy,
                manifest_limit,
                prepaid_rollback: false,
            },
            storage: None,
        })
    }
    /// Uses the same genuine ledger construction account for closed imported
    /// Host-table producers. Generic parser funding alone grants no publication.
    pub fn new_with_storage(
        context: &WorkspaceContext,
        storage: crate::working_memory::StorageMetadataFunding,
        dependency: HostMetadataFunding,
        policy: DependencyMemoryPolicy,
        manifest_limit: usize,
    ) -> Result<Self, WorkspaceMetadataError> {
        let construction = context
            .metadata_funding()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        if !construction.same_account(storage.funding()) {
            return Err(WorkspaceMetadataError::Unqualified);
        }
        context.charge_metadata(size_of::<(
            &WorkspaceContext,
            crate::working_memory::StorageMetadataFunding,
            HostMetadataFunding,
            DependencyMemoryPolicy,
            usize,
            Result<Self, WorkspaceMetadataError>,
        )>())?;
        let mut result = Self::new(context, dependency, policy, manifest_limit)?;
        result.storage = Some(storage);
        Ok(result)
    }
    /// Source-bound construction owner; absent for parser-only preparations.
    pub fn storage_source(&self) -> Option<&crate::working_memory::StorageMetadataFunding> {
        self.storage.as_ref()
    }
    /// Actual construction context, borrowed without creating another payer.
    pub fn context(&self) -> &WorkspaceContext {
        &self.context
    }
    /// Separately admitted dependency estimate account.
    pub fn dependency(&self) -> &HostMetadataFunding {
        &self.accounts.dependency
    }
    /// Selected source-based dependency estimate policy.
    pub fn policy(&self) -> DependencyMemoryPolicy {
        self.accounts.policy
    }
    /// Maximum accepted manifest byte input from its opened handle.
    pub fn manifest_limit(&self) -> usize {
        self.accounts.manifest_limit
    }
    /// Resolves the actual CURRENT generation through the funded pointer worker.
    pub fn resolve_root(&self, directory: &Path) -> Result<PathBuf, PromptCachePersistenceFailure> {
        resolve_prompt_cache_root_with_funding(directory, Some(&self.accounts))
            .map_err(|e| self.accounts.error(e))
    }
    /// Resolves a shard through the same traversal/symlink validation worker.
    pub fn shard_path(
        &self,
        directory: &Path,
        relative: &str,
    ) -> Result<PathBuf, PromptCachePersistenceFailure> {
        safe_prompt_cache_shard_path_with_funding(directory, relative, Some(&self.accounts))
            .map_err(|e| self.accounts.error(e))
    }
    /// Synchronizes and hashes a real shard through the same bounded reader.
    pub fn finalize_shard(&self, path: &Path) -> Result<String, PromptCachePersistenceFailure> {
        sync_file_with_funding(path, Some(&self.accounts))
            .and_then(|()| hash_prompt_cache_shard_payload_with_funding(path, Some(&self.accounts)))
            .map_err(|e| self.accounts.error(e))
    }
    /// Constructs a path with an admitted exact-capacity destination. The caller
    /// retains this funding alongside any path that escapes its current scope.
    pub fn join_path(
        &self,
        directory: &Path,
        component: impl AsRef<Path>,
    ) -> Result<PathBuf, PromptCachePersistenceFailure> {
        path_join(directory, component, Some(&self.accounts)).map_err(|e| self.accounts.error(e))
    }
    /// Constructs the same canonical rank-local path with funded destination bytes.
    pub fn rank_path(
        &self,
        root: &Path,
        topology: &PromptCacheTopology,
    ) -> Result<PathBuf, PromptCachePersistenceFailure> {
        rank_path(root, topology, Some(&self.accounts)).map_err(|e| self.accounts.error(e))
    }
    /// Admits stock catalog validation scratch from the actual manifest encoding.
    /// Uses the unchanged core schema validator, including identity and coverage.
    pub fn validate_manifest(
        &self,
        manifest: &PromptCacheManifest,
    ) -> Result<(), PromptCachePersistenceFailure> {
        source_bytes(manifest)
            .and_then(|bytes| self.accounts.estimate(bytes))
            .and_then(|()| manifest.validate().map_err(Into::into))
            .map_err(|e| self.accounts.error(e))
    }
    /// Admits core descriptor validation using its retained diagnostic source
    /// length as an explicit policy estimate, then invokes the same validator.
    pub fn validate_descriptor(
        &self,
        descriptor: &eredu_core::cache::PromptCacheDescriptor,
    ) -> Result<(), PromptCachePersistenceFailure> {
        text_bytes(format_args!("{descriptor:?}"))
            .and_then(|bytes| self.accounts.estimate(bytes))
            .and_then(|()| descriptor.validate().map_err(Into::into))
            .map_err(|e| self.accounts.error(e))
    }
    /// Validates exact retained model/rank identity through the shared core worker.
    pub fn validate_model_identity(
        &self,
        descriptor: &eredu_core::cache::PromptCacheDescriptor,
        identity: &eredu_core::cache::PromptCacheModelIdentity,
    ) -> Result<(), PromptCachePersistenceFailure> {
        text_bytes(format_args!("{descriptor:?}{identity:?}"))
            .and_then(|bytes| self.accounts.estimate(bytes))
            .and_then(|()| {
                eredu_core::cache::validate_prompt_cache_model_identity(descriptor, identity)
                    .map_err(Into::into)
            })
            .map_err(|e| self.accounts.error(e))
    }
    /// Runs the same schema, descriptor and token-prefix compatibility checks.
    pub fn validate_compatibility(
        &self,
        manifest: &PromptCacheManifest,
        descriptor: &eredu_core::cache::PromptCacheDescriptor,
        prefix: &[u32],
    ) -> Result<(), PromptCachePersistenceFailure> {
        let result = (|| {
            let bytes = source_bytes(manifest)?
                .checked_add(text_bytes(format_args!("{descriptor:?}"))?)
                .ok_or(PromptCachePersistenceError::Metadata(
                    WorkspaceMetadataError::Overflow,
                ))?;
            self.accounts.estimate(bytes)?;
            self.accounts.buffer(64)?;
            manifest
                .validate_compatibility(descriptor, prefix)
                .map_err(Into::into)
        })();
        result.map_err(|e| self.accounts.error(e))
    }
    /// Fixed wrapper and retained-error controls; payload sources are additional.
    pub fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<PersistenceAccounts>(),
            size_of::<PreparedPromptCachePublication>(),
            size_of::<PreparedReversiblePromptCachePublication>(),
            size_of::<PromptCachePublication>(),
            size_of::<ReversiblePromptCachePublication>(),
            size_of::<PromptCachePersistenceFailure>(),
            size_of::<PromptCachePersistenceError>(),
            size_of::<Result<Self, WorkspaceMetadataError>>(),
            HostMetadataFunding::reservation_control_bytes(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Checked allowance for one bounded dependency input, kept separate from
    /// controlled input capacity. The caller reserves this as estimated overhead.
    pub fn dependency_bytes(policy: DependencyMemoryPolicy, source_bytes: usize) -> Option<usize> {
        policy.estimate(source_bytes)
    }
}
impl PersistenceAccounts {
    fn error(&self, cause: PromptCachePersistenceError) -> PromptCachePersistenceFailure {
        PromptCachePersistenceFailure {
            cause,
            funding: self.clone(),
        }
    }
    pub(super) fn estimate(&self, input: usize) -> Result<(), PromptCachePersistenceError> {
        if self.prepaid_rollback {
            return Ok(());
        }
        let bytes = PromptCachePersistenceFunding::dependency_bytes(self.policy, input).ok_or(
            PromptCachePersistenceError::Metadata(WorkspaceMetadataError::Overflow),
        )?;
        self.dependency
            .reserve_metadata(bytes)
            .map_err(PromptCachePersistenceError::Dependency)
    }
    fn buffer_bytes(bytes: usize) -> Result<usize, PromptCachePersistenceError> {
        let controls = [
            bytes,
            size_of::<Vec<u8>>(),
            size_of::<PathBuf>(),
            size_of::<String>(),
            size_of::<std::fmt::Arguments<'_>>(),
            size_of::<File>(),
            size_of::<std::fs::Metadata>(),
            size_of::<Sha256>(),
            size_of::<Result<(), std::io::Error>>(),
            size_of::<Result<Vec<u8>, PromptCachePersistenceError>>(),
            HostMetadataFunding::reservation_control_bytes(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(PromptCachePersistenceError::Metadata(
                WorkspaceMetadataError::Overflow,
            ))?;
        Ok(bytes)
    }
    pub(super) fn buffer(&self, bytes: usize) -> Result<(), PromptCachePersistenceError> {
        if self.prepaid_rollback {
            return Ok(());
        }
        self.construction
            .reserve_metadata(Self::buffer_bytes(bytes)?)
            .map_err(|e| PromptCachePersistenceError::Metadata(e.into()))
    }
    pub(super) fn prepare_rollback(
        &self,
        destination: &Path,
        parent: &Path,
        staging: &Path,
        previous: bool,
        moved: Option<&Path>,
        nonce: u128,
    ) -> Result<Self, PromptCachePersistenceError> {
        let mut bytes = 0usize;
        let mut add = |capacity| -> Result<(), PromptCachePersistenceError> {
            bytes = bytes.checked_add(Self::buffer_bytes(capacity)?).ok_or(
                PromptCachePersistenceError::Metadata(WorkspaceMetadataError::Overflow),
            )?;
            Ok(())
        };
        // The rollback worker can return exactly one I/O failure. Reserve each
        // named possible diagnostic destination so no failure needs a new grant.
        for path in [destination, parent, staging] {
            add(path.as_os_str().len())?;
        }
        if let Some(path) = moved {
            add(path.as_os_str().len())?;
            if let Some(parent) = path.parent() {
                add(parent.as_os_str().len())?;
            }
        }
        if previous {
            let temporary_bytes = text_bytes(format_args!(
                ".{PROMPT_CACHE_CURRENT_FILE}.tmp-{}",
                nonce ^ 1
            ))?;
            add(temporary_bytes)?;
            let temporary = join_bytes(destination, temporary_bytes)?;
            let current = join_bytes(destination, PROMPT_CACHE_CURRENT_FILE.len())?;
            add(temporary)?;
            add(current)?;
            // Pointer create/write/sync use one cloned temporary on failure.
            add(temporary)?;
            // Directory sync can retain one destination path on failure.
            add(destination.as_os_str().len())?;
        }
        // Explicit rollback consumes the transaction; if it fails, Drop invokes
        // the same rollback once more. Both occurrences are prepared together.
        let bytes = bytes
            .checked_mul(2)
            .ok_or(PromptCachePersistenceError::Metadata(
                WorkspaceMetadataError::Overflow,
            ))?;
        self.construction
            .reserve_metadata(bytes)
            .map_err(|e| PromptCachePersistenceError::Metadata(e.into()))?;
        self.path(destination)?;
        self.path(destination)?;
        let mut paid = self.clone();
        paid.prepaid_rollback = true;
        Ok(paid)
    }
    pub(super) fn path(&self, path: &Path) -> Result<(), PromptCachePersistenceError> {
        // Platform filesystem path conversion/canonicalization scratch remains
        // a separate estimate. Owned PathBuf destinations use path_copy/join.
        self.estimate(path.as_os_str().len())
    }
}

/// A persistence refusal retaining both exact construction and estimate custody.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct PromptCachePersistenceFailure {
    #[source]
    cause: PromptCachePersistenceError,
    funding: PersistenceAccounts,
}
impl PromptCachePersistenceFailure {
    /// Original typed validation, I/O, construction, or dependency refusal.
    pub fn cause(&self) -> &PromptCachePersistenceError {
        &self.cause
    }
}

/// Funded ownership of the existing ordinary atomic publication worker.
#[derive(Debug)]
pub struct PreparedPromptCachePublication {
    publication: PromptCachePublication,
    funding: PersistenceAccounts,
}
impl PreparedPromptCachePublication {
    /// Reserves metadata before the shared path/publication constructor runs.
    pub fn begin(
        destination: &Path,
        replace: bool,
        funding: &PromptCachePersistenceFunding,
    ) -> Result<Self, PromptCachePersistenceFailure> {
        let funding = &funding.accounts;
        PromptCachePublication::begin_with_funding(destination, replace, Some(funding))
            .map(|publication| Self {
                publication,
                funding: funding.clone(),
            })
            .map_err(|e| funding.error(e))
    }
    /// Exact hidden shard destination retained by this publication owner.
    pub fn staging_directory(&self) -> &Path {
        self.publication.staging_directory()
    }
    /// Validates and publishes through the same atomic worker. Borrowed metadata
    /// remains under its caller's original custody for this whole call.
    pub fn commit(
        self,
        manifest: &PromptCacheManifest,
    ) -> Result<(), PromptCachePersistenceFailure> {
        let Self {
            publication,
            funding,
        } = self;
        publication
            .commit_with_funding(manifest, Some(&funding))
            .map_err(|e| funding.error(e))
    }
}

/// Funded ownership of the existing reversible distributed publication worker.
#[derive(Debug)]
pub struct PreparedReversiblePromptCachePublication {
    publication: ReversiblePromptCachePublication,
    funding: PersistenceAccounts,
}
impl PreparedReversiblePromptCachePublication {
    /// Reserves metadata before constructing the shared reversible transaction.
    pub fn begin(
        destination: &Path,
        replace: bool,
        funding: &PromptCachePersistenceFunding,
    ) -> Result<Self, PromptCachePersistenceFailure> {
        let funding = &funding.accounts;
        ReversiblePromptCachePublication::begin_with_funding(destination, replace, Some(funding))
            .map(|publication| Self {
                publication,
                funding: funding.clone(),
            })
            .map_err(|e| funding.error(e))
    }
    /// Hidden destination used by the ordinary shared save worker.
    pub fn staging_destination(&self) -> &Path {
        self.publication.staging_destination()
    }
    /// Shares validation and reversible publication, preserving every agreement.
    pub fn publish(&mut self) -> Result<(), PromptCachePersistenceFailure> {
        self.publication
            .publish_with_funding(Some(&self.funding))
            .map_err(|e| self.funding.error(e))
    }
    /// Commits the same reversible publication after distributed agreement.
    pub fn commit(self) -> Result<(), PromptCachePersistenceFailure> {
        let Self {
            publication,
            funding,
        } = self;
        publication.commit().map_err(|e| funding.error(e))
    }
    /// Rolls back the same publication and retains its metadata on failure.
    pub fn rollback(self) -> Result<(), PromptCachePersistenceFailure> {
        let Self {
            publication,
            funding,
        } = self;
        publication.rollback().map_err(|e| funding.error(e))
    }
}

/// Reads the bounded actual file through the ordinary parser/validator and
/// retains both payers with the immutable returned manifest and every failure.
pub fn inspect_prompt_cache_funded(
    directory: impl AsRef<Path>,
    funding: &PromptCachePersistenceFunding,
) -> Result<SharedPromptCacheManifest, PromptCachePersistenceFailure> {
    let funding = &funding.accounts;
    let shell = PreparedPromptCacheManifest::prepare_with_dependency(
        funding.construction.clone(),
        funding.dependency.clone(),
    )
    .map_err(|e| funding.error(PromptCachePersistenceError::Metadata(e.into())))?;
    inspect_prompt_cache_with_funding(directory.as_ref(), Some(funding))
        .map(|manifest| shell.publish(manifest))
        .map_err(|e| funding.error(e))
}

pub(super) fn source_bytes(
    value: &impl serde::Serialize,
) -> Result<usize, PromptCachePersistenceError> {
    struct Counter(usize);
    impl Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_add(bytes.len())
                .ok_or(std::io::ErrorKind::FileTooLarge)?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut count = Counter(0);
    serde_json::to_writer(&mut count, value).map_err(PromptCachePersistenceError::ManifestJson)?;
    Ok(count.0)
}

pub(super) fn read_bounded(
    path: &Path,
    maximum: Option<usize>,
    funding: Option<&PersistenceAccounts>,
) -> Result<Vec<u8>, PromptCachePersistenceError> {
    if let Some(funding) = funding {
        funding.path(path)?;
    }
    let error = |source| io_error("read bounded prompt cache metadata", path, source, funding);
    let mut file = File::open(path).map_err(error)?;
    let length = usize::try_from(file.metadata().map_err(error)?.len())
        .map_err(|_| PromptCachePersistenceError::Metadata(WorkspaceMetadataError::Overflow))?;
    if maximum.is_some_and(|maximum| length > maximum) {
        return Err(PromptCachePersistenceError::MetadataInputLimit {
            bytes: length,
            limit: maximum.unwrap(),
        });
    }
    if let Some(funding) = funding {
        funding.buffer(length)?;
        funding.estimate(length)?;
    }
    let mut input = Vec::new();
    input
        .try_reserve_exact(length)
        .map_err(|_| PromptCachePersistenceError::Metadata(WorkspaceMetadataError::Overflow))?;
    input.resize(length, 0);
    file.read_exact(&mut input).map_err(error)?;
    let mut extra = [0u8; 1];
    if file.read(&mut extra).map_err(error)? != 0 {
        return Err(PromptCachePersistenceError::MetadataChanged);
    }
    Ok(input)
}

pub(super) fn path_copy(
    path: &Path,
    funding: Option<&PersistenceAccounts>,
) -> Result<PathBuf, PromptCachePersistenceError> {
    let capacity = path.as_os_str().len();
    if let Some(funding) = funding {
        funding.buffer(capacity)?;
    }
    let mut value = PathBuf::with_capacity(capacity);
    value.push(path);
    Ok(value)
}
pub(super) fn path_join(
    path: &Path,
    component: impl AsRef<Path>,
    funding: Option<&PersistenceAccounts>,
) -> Result<PathBuf, PromptCachePersistenceError> {
    let component = component.as_ref();
    let capacity = join_bytes(path, component.as_os_str().len())?;
    if let Some(funding) = funding {
        funding.buffer(capacity)?;
    }
    let mut value = PathBuf::with_capacity(capacity);
    value.push(path);
    value.push(component);
    Ok(value)
}
fn text_bytes(arguments: std::fmt::Arguments<'_>) -> Result<usize, PromptCachePersistenceError> {
    use std::fmt::Write as _;
    struct Count(usize);
    impl std::fmt::Write for Count {
        fn write_str(&mut self, value: &str) -> std::fmt::Result {
            self.0 = self.0.checked_add(value.len()).ok_or(std::fmt::Error)?;
            Ok(())
        }
    }
    let mut count = Count(0);
    count
        .write_fmt(arguments)
        .map_err(|_| PromptCachePersistenceError::Metadata(WorkspaceMetadataError::Overflow))?;
    Ok(count.0)
}
pub(super) fn text(
    arguments: std::fmt::Arguments<'_>,
    funding: Option<&PersistenceAccounts>,
) -> Result<String, PromptCachePersistenceError> {
    use std::fmt::Write as _;
    let count = text_bytes(arguments)?;
    if let Some(funding) = funding {
        funding.buffer(count)?;
    }
    let mut value = String::with_capacity(count);
    value
        .write_fmt(arguments)
        .expect("counted String formatter");
    Ok(value)
}
fn join_bytes(path: &Path, component_bytes: usize) -> Result<usize, PromptCachePersistenceError> {
    path.as_os_str()
        .len()
        .checked_add(component_bytes)
        .and_then(|n| n.checked_add(1))
        .ok_or(PromptCachePersistenceError::Metadata(
            WorkspaceMetadataError::Overflow,
        ))
}
pub(super) fn io_error(
    action: &'static str,
    path: &Path,
    source: std::io::Error,
    funding: Option<&PersistenceAccounts>,
) -> PromptCachePersistenceError {
    match path_copy(path, funding) {
        Ok(path) => PromptCachePersistenceError::Io {
            action,
            path,
            source,
        },
        Err(error) => error,
    }
}
pub(super) fn storage_error(
    message: std::fmt::Arguments<'_>,
    funding: Option<&PersistenceAccounts>,
) -> PromptCachePersistenceError {
    match text(message, funding) {
        Ok(message) => PromptCachePersistenceError::MalformedStorage(message),
        Err(error) => error,
    }
}
pub(super) fn invalid_path(
    path: &Path,
    funding: Option<&PersistenceAccounts>,
) -> PromptCachePersistenceError {
    match path_copy(path, funding) {
        Ok(path) => PromptCachePersistenceError::InvalidPromptCachePath(path),
        Err(error) => error,
    }
}

/// Reads the actual header from an already opened file. The caller retains and
/// verifies its file-version witness; no path is reopened by this worker.
pub fn read_shard_metadata_from_funded(
    file: &mut File,
    path: &Path,
    funding: &PromptCachePersistenceFunding,
) -> Result<(safetensors::tensor::Metadata, Vec<u8>, usize), PromptCachePersistenceFailure> {
    funding
        .accounts
        .path(path)
        .and_then(|()| read_shard_metadata_from(file, path, Some(&funding.accounts)))
        .map_err(|e| funding.accounts.error(e))
}
/// Hashes the payload on the same opened file under its admitted fixed buffer.
/// File-version and expected digest comparison remain the calling source's proof.
pub fn hash_prompt_cache_shard_payload_from_funded(
    file: &mut File,
    path: &Path,
    data_start: usize,
    funding: &PromptCachePersistenceFunding,
) -> Result<String, PromptCachePersistenceFailure> {
    let data_start = u64::try_from(data_start).map_err(|_| {
        funding
            .accounts
            .error(PromptCachePersistenceError::Metadata(
                WorkspaceMetadataError::Overflow,
            ))
    })?;
    hash_shard_payload_from(file, path, data_start, Some(&funding.accounts))
        .map_err(|e| funding.accounts.error(e))
}

pub(super) fn rank_path(
    root: &Path,
    topology: &PromptCacheTopology,
    funding: Option<&PersistenceAccounts>,
) -> Result<PathBuf, PromptCachePersistenceError> {
    let result = if topology.cache_rank_identity().is_none() {
        path_copy(root, funding)
    } else {
        struct Coordinate(Option<(usize, usize)>);
        impl std::fmt::Display for Coordinate {
            fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    Some((_, rank)) => write!(out, "{rank}"),
                    None => out.write_str("x"),
                }
            }
        }
        text(
            format_args!(
                "rank-p{}-t{}-e{}",
                Coordinate(topology.stage()),
                Coordinate(topology.shard()),
                Coordinate(topology.addressable())
            ),
            funding,
        )
        .and_then(|name| path_join(root, name, funding))
    };
    result
}

#[cfg(test)]
mod tests;
