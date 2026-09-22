//! Authenticated persistent cache shards, independent of live-writer ownership.
use super::*;
use eredu_checkpoint::artifact::{
    ArtifactFileReadError, ArtifactFileReadFailure, ArtifactFileVersion, PreparedArtifactFileRead,
};
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceContext, WorkspaceMetadataError};
use std::mem::{size_of, size_of_val};

#[derive(Debug)]
struct PersistentFile {
    path: PathBuf,
    // Keeping the imported handle prevents its inode from being recycled while
    // source aliases exist. Reads still authenticate their independently opened handle.
    _file: File,
    version: ArtifactFileVersion,
    layout: CacheShardLayout,
    digest: [u8; 32],
}

/// One verified persistent file, exact header and observable version. Aliases
/// retain its opened handle and paid metadata; retirement never unlinks the file
/// and owns no live-cache Disk reservation or native transfer authority.
#[derive(Clone, Debug)]
pub struct PersistentCacheBlockSource {
    inner: Arc<PersistentFile>,
    construction: HostMetadataFunding,
    dependency: HostMetadataFunding,
}

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Metadata(#[from] WorkspaceMetadataError),
    #[error(transparent)]
    Source(#[from] eredu_nn::Error),
    #[error(transparent)]
    Persistence(#[from] PromptCachePersistenceFailure),
    #[error(transparent)]
    Version(#[from] ArtifactFileReadError),
    #[error(transparent)]
    Read(#[from] ArtifactFileReadFailure),
    #[error(transparent)]
    Layout(#[from] CacheShardError),
    #[error("persistent cache shard declarations differ from the manifest")]
    Declaration,
    #[error("persistent cache payload digest differs from the authenticated source")]
    Digest,
}

/// Typed import or read failure retaining the actual source and its accepted
/// construction/dependency accounts. Failed positional reads retain their file.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct PersistentCacheReadFailure {
    #[source]
    cause: Cause,
    source: Option<PersistentCacheBlockSource>,
    construction: Option<HostMetadataFunding>,
    dependency: HostMetadataFunding,
}
impl PersistentCacheReadFailure {
    /// Existing file-version refusal, before transfer to a caller destination.
    pub fn binding_error(&self) -> Option<&ArtifactFileReadError> {
        match &self.cause {
            Cause::Version(cause) => Some(cause),
            _ => None,
        }
    }
    /// Existing positional-read failure and its exact written prefix.
    pub fn read_error(&self) -> Option<&ArtifactFileReadFailure> {
        match &self.cause {
            Cause::Read(cause) => Some(cause),
            _ => None,
        }
    }
    /// Whether authentication rejected the payload bytes.
    pub fn is_digest_mismatch(&self) -> bool {
        matches!(self.cause, Cause::Digest)
    }
    /// Exact source retained by a read failure; import failures have no source yet.
    pub fn source_file(&self) -> Option<&PersistentCacheBlockSource> {
        self.source.as_ref()
    }
}

impl PersistentCacheBlockSource {
    /// Authenticates the same opened file through bounded parsing, exact
    /// manifest declarations, payload SHA-256 and before/after version checks.
    /// The caller owns path selection and canonical cache-row publication.
    pub fn prepare(
        mut file: File,
        path: &Path,
        block: &PromptCacheBlock,
        funding: &PromptCachePersistenceFunding,
    ) -> Result<Self, PersistentCacheReadFailure> {
        let construction = funding.context().metadata_funding();
        let failure = |cause| PersistentCacheReadFailure {
            cause,
            source: None,
            construction: construction.clone(),
            dependency: funding.dependency().clone(),
        };
        let construct =
            || -> Result<Self, Cause> {
                let context = funding.context();
                let account = construction
                    .as_ref()
                    .ok_or(WorkspaceMetadataError::Unqualified)?;
                context.charge_metadata(
                    Self::preparation_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
                )?;
                let version = ArtifactFileVersion::capture(&file)?;
                let (metadata, header, file_bytes) =
                    read_shard_metadata_from_funded(&mut file, path, funding)?;
                if version.byte_len() != file_bytes {
                    return Err(ArtifactFileReadError::Changed.into());
                }
                let names = cache_shard_tensor_names(block.representation);
                if names != [block.first_array.as_str(), block.second_array.as_str()] {
                    return Err(Cause::Declaration);
                }
                let expected = [
                    (&block.first_shape, &block.first_dtype),
                    (&block.second_shape, &block.second_dtype),
                ];
                let mut payload = 0u64;
                for (name, (shape, dtype)) in names.into_iter().zip(expected) {
                    let tensor = metadata.info(name).ok_or(Cause::Declaration)?;
                    if tensor.shape.len() != shape.len()
                        || tensor.shape.iter().zip(shape).any(|(actual, expected)| {
                            usize::try_from(*expected).ok() != Some(*actual)
                        })
                        || !stored_dtype_matches(tensor.dtype, dtype)
                    {
                        return Err(Cause::Declaration);
                    }
                    let bytes = tensor
                        .data_offsets
                        .1
                        .checked_sub(tensor.data_offsets.0)
                        .ok_or(Cause::Declaration)?;
                    payload = payload
                        .checked_add(
                            u64::try_from(bytes).map_err(|_| WorkspaceMetadataError::Overflow)?,
                        )
                        .ok_or(WorkspaceMetadataError::Overflow)?;
                }
                if payload != block.logical_bytes {
                    return Err(Cause::Declaration);
                }
                // Upstream exposes this population through an allocating map. Its
                // allowance remains a separately labelled dependency estimate.
                let estimate =
                    PromptCachePersistenceFunding::dependency_bytes(funding.policy(), header.len())
                        .ok_or(WorkspaceMetadataError::Overflow)?;
                funding
                    .dependency()
                    .reserve_metadata(estimate)
                    .map_err(WorkspaceMetadataError::from)?;
                let tensor_count = metadata.tensors().len();
                let hash = hash_prompt_cache_shard_payload_from_funded(
                    &mut file,
                    path,
                    header.len(),
                    funding,
                )?;
                if hash != block.payload_sha256 {
                    return Err(Cause::Digest);
                }
                let digest = digest_bytes(&hash).ok_or(Cause::Digest)?;
                version.validate(&file)?;
                let layout = CacheShardLayout::imported(
                    block.representation,
                    &metadata,
                    tensor_count,
                    header,
                    file_bytes,
                    context,
                )?;
                context.charge_metadata(
                    path.as_os_str()
                        .len()
                        .checked_add(size_of::<PathBuf>())
                        .ok_or(WorkspaceMetadataError::Overflow)?,
                )?;
                let mut owned_path = PathBuf::with_capacity(path.as_os_str().len());
                owned_path.push(path);
                Ok(Self {
                    inner: context.metadata_arc(PersistentFile {
                        path: owned_path,
                        _file: file,
                        version,
                        layout,
                        digest,
                    })?,
                    construction: account.clone(),
                    dependency: funding.dependency().clone(),
                })
            };
        construct().map_err(failure)
    }
    /// Exact path whose opened file was authenticated during import.
    pub fn path(&self) -> &Path {
        &self.inner.path
    }
    /// Retained actual header and tensor offsets, without re-encoding.
    pub fn layout(&self) -> &CacheShardLayout {
        &self.inner.layout
    }
    /// Complete authenticated file extent required by the read destination.
    pub fn file_bytes(&self) -> usize {
        self.inner.version.byte_len()
    }
    /// Shared identity of one actual validated import, independent of labels.
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Binds a caller-opened handle to the original imported file version.
    /// Destination capacity and I/O task admission remain with the caller.
    pub fn prepare_read_from(
        &self,
        file: File,
        context: &WorkspaceContext,
    ) -> Result<PreparedPersistentCacheRead, PersistentCacheReadFailure> {
        let construction = context.metadata_funding();
        let failed = |cause| PersistentCacheReadFailure {
            cause,
            source: Some(self.clone()),
            construction: construction.clone(),
            dependency: self.dependency.clone(),
        };
        let funding = construction
            .as_ref()
            .ok_or_else(|| failed(WorkspaceMetadataError::Unqualified.into()))?;
        context
            .charge_metadata(
                Self::read_control_bytes()
                    .ok_or_else(|| failed(WorkspaceMetadataError::Overflow.into()))?,
            )
            .map_err(|cause| failed(Cause::Metadata(cause)))?;
        let read = self
            .inner
            .version
            .bind(file)
            .map_err(|cause| failed(Cause::Version(cause)))?;
        Ok(PreparedPersistentCacheRead {
            read,
            source: self.clone(),
            funding: funding.clone(),
        })
    }
    /// Reads into ordinary caller-owned storage through the same version and
    /// digest worker. The caller owns its destination and I/O preparation.
    pub fn read_ordinary_into(
        &self,
        file: File,
        destination: &mut [u8],
    ) -> Result<(), PersistentCacheReadFailure> {
        let read = self
            .inner
            .version
            .bind(file)
            .map_err(|cause| PersistentCacheReadFailure {
                cause: Cause::Version(cause),
                source: Some(self.clone()),
                construction: Some(self.construction.clone()),
                dependency: self.dependency.clone(),
            })?;
        PreparedPersistentCacheRead {
            read,
            source: self.clone(),
            funding: self.construction.clone(),
        }
        .read_into(destination)
    }
    /// Authenticated payload checksum, without formatting or allocating.
    pub fn payload_digest(&self) -> &[u8; 32] {
        &self.inner.digest
    }

    /// Fixed preparation controls; parser buffers and dependency estimates are
    /// independently charged by the same-handle persistence workers.
    pub fn preparation_control_bytes() -> Option<usize> {
        let frames = [
            ArtifactFileVersion::control_bytes()?,
            size_of::<Self>(),
            size_of::<PersistentFile>(),
            size_of::<PersistentCacheReadFailure>(),
            size_of::<Cause>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<Self, PersistentCacheReadFailure>>(),
            size_of::<(
                File,
                &Path,
                &PromptCacheBlock,
                &PromptCachePersistenceFunding,
            )>(),
            size_of::<[u8; 32]>(),
            size_of::<String>(),
            size_of::<[(&Vec<i32>, &String); 2]>(),
            size_of::<(usize, usize, u64)>(),
            size_of::<Result<(), ArtifactFileReadError>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Fixed binding, digest validation and positional-read transport controls.
    /// The actual whole-file destination is funded separately by the reader.
    pub fn read_control_bytes() -> Option<usize> {
        let frames = [
            PreparedArtifactFileRead::control_bytes()?,
            size_of::<PreparedPersistentCacheRead>(),
            size_of::<PersistentCacheReadFailure>(),
            size_of::<Cause>(),
            size_of::<Self>(),
            size_of::<(&Self, File, &WorkspaceContext)>(),
            size_of::<&mut [u8]>(),
            size_of::<Result<PreparedPersistentCacheRead, PersistentCacheReadFailure>>(),
            size_of::<Result<(), PersistentCacheReadFailure>>(),
            size_of::<Result<(), ArtifactFileReadFailure>>(),
            size_of::<[CacheShardTensor<'_>; 2]>(),
            size_of::<Result<[CacheShardTensor<'_>; 2], CacheShardError>>(),
            size_of::<Sha256>(),
            size_of::<sha2::digest::Output<Sha256>>(),
            size_of::<[u8; 32]>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}

/// One paid read retaining the persistent import and its authenticated version.
#[derive(Debug)]
pub struct PreparedPersistentCacheRead {
    read: PreparedArtifactFileRead,
    source: PersistentCacheBlockSource,
    funding: HostMetadataFunding,
}
impl PreparedPersistentCacheRead {
    /// Copies this authenticated source into a caller-owned provisional file
    /// with the shared bounded reader, checking the actual streamed payload.
    pub fn copy_to(self, destination: &mut File) -> Result<(), PersistentCacheReadFailure> {
        let Self {
            read,
            source,
            funding,
        } = self;
        let result = (|| -> Result<(), Cause> {
            funding
                .reserve_metadata(
                    Self::copy_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
                )
                .map_err(WorkspaceMetadataError::from)?;
            let mut hash = Sha256::new();
            let mut header_matches = true;
            read.copy_to_with(
                destination,
                copy_observer(source.layout().header(), &mut hash, &mut header_matches),
            )?;
            if !header_matches {
                return Err(CacheShardError::Header.into());
            }
            let actual: [u8; 32] = hash.finalize().into();
            if actual != source.inner.digest {
                return Err(Cause::Digest);
            }
            Ok(())
        })();
        result.map_err(|cause| PersistentCacheReadFailure {
            cause,
            dependency: source.dependency.clone(),
            source: Some(source),
            construction: Some(funding),
        })
    }
    /// The same concrete observer factory used by streaming copies, including
    /// its hash and header-validation state and fixed byte buffer.
    pub fn copy_control_bytes() -> Option<usize> {
        // A type witness visits the actual closure without invoking it.
        fn with_observer<F: FnMut(usize, &[u8])>(_: &F) -> Option<usize> {
            PreparedArtifactFileRead::copy_to_control_bytes::<F>()
        }
        let mut hash = Sha256::new();
        let mut matches = true;
        let observer = copy_observer(&[], &mut hash, &mut matches);
        let frames = [
            with_observer(&observer)?,
            size_of::<Self>(),
            size_of::<PersistentCacheReadFailure>(),
            size_of::<Cause>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), PersistentCacheReadFailure>>(),
            size_of::<Sha256>(),
            size_of::<sha2::digest::Output<Sha256>>(),
            size_of::<[u8; 32]>(),
            size_of::<(&mut File, usize, bool)>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Exact whole-file destination extent.
    pub fn byte_len(&self) -> usize {
        self.read.byte_len()
    }
    /// Uses the existing positional reader, then checks the retained actual
    /// header and payload digest before the bytes can become a cache source.
    pub fn read_into(self, destination: &mut [u8]) -> Result<(), PersistentCacheReadFailure> {
        let Self {
            read,
            source,
            funding,
        } = self;
        let result = (|| -> Result<(), Cause> {
            read.read_into(destination)?;
            source.layout().tensors(destination)?;
            let mut hash = Sha256::new();
            hash.update(&destination[source.layout().header().len()..]);
            let actual: [u8; 32] = hash.finalize().into();
            if actual != source.inner.digest {
                return Err(Cause::Digest);
            }
            Ok(())
        })();
        result.map_err(|cause| PersistentCacheReadFailure {
            cause,
            dependency: source.dependency.clone(),
            source: Some(source),
            construction: Some(funding),
        })
    }
}

fn digest_bytes(hash: &str) -> Option<[u8; 32]> {
    if hash.len() != 64 || !hash.is_ascii() {
        return None;
    }
    let mut output = [0; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hash[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(output)
}

fn copy_observer<'a>(
    header: &'a [u8],
    hash: &'a mut Sha256,
    header_matches: &'a mut bool,
) -> impl FnMut(usize, &[u8]) + 'a {
    move |offset, bytes| {
        let prefix = if offset < header.len() {
            (header.len() - offset).min(bytes.len())
        } else {
            0
        };
        if prefix != 0 && bytes[..prefix] != header[offset..offset + prefix] {
            *header_matches = false;
        }
        hash.update(&bytes[prefix..]);
    }
}

#[cfg(all(test, unix))]
#[path = "persistent/tests.rs"]
mod tests;

#[path = "persistent/state.rs"]
mod state;
pub use state::PersistentCacheStateTensor;
