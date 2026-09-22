//! Completed immutable fixed-state payload from one authenticated opened file.
use super::*;

/// Exact initialized bytes and scalar layout authenticated against one retained
/// manifest declaration. Both metadata accounts follow every byte and shape.
#[derive(Debug)]
pub struct PersistentCacheStateTensor<B> {
    bytes: B,
    header_bytes: usize,
    shape: Vec<i32>,
    dtype: safetensors::tensor::Dtype,
    construction: HostMetadataFunding,
    dependency: HostMetadataFunding,
}
impl<B: AsRef<[u8]> + AsMut<[u8]>> PersistentCacheStateTensor<B> {
    /// Parses and fills through the same bounded-header and positional workers.
    /// The actual file version and digest are checked before payload publication.
    pub fn read_with<F>(
        mut file: File,
        path: &Path,
        declaration: &PromptCacheStateTensor,
        funding: &PromptCachePersistenceFunding,
        allocate: F,
    ) -> Result<Self, PersistentCacheReadFailure>
    where
        F: FnOnce(usize) -> Result<B, eredu_nn::Error>,
    {
        let context = funding.context();
        let construction = context.metadata_funding();
        let read = || -> Result<Self, Cause> {
            let account = construction
                .as_ref()
                .ok_or(WorkspaceMetadataError::Unqualified)?;
            let frames = [
                ArtifactFileVersion::control_bytes().ok_or(WorkspaceMetadataError::Overflow)?,
                PreparedArtifactFileRead::control_bytes()
                    .ok_or(WorkspaceMetadataError::Overflow)?,
                size_of::<Self>(),
                size_of::<F>(),
                size_of::<B>(),
                size_of::<Result<B, eredu_nn::Error>>(),
                size_of::<Cause>(),
                size_of::<PersistentCacheReadFailure>(),
                size_of::<Result<Self, Cause>>(),
                size_of::<Result<Self, PersistentCacheReadFailure>>(),
                size_of::<(
                    File,
                    &Path,
                    &PromptCacheStateTensor,
                    &PromptCachePersistenceFunding,
                )>(),
                size_of::<(usize, usize, [u8; 32], Sha256)>(),
                size_of::<sha2::digest::Output<Sha256>>(),
            ];
            context.charge_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(WorkspaceMetadataError::Overflow)?,
            )?;
            let version = ArtifactFileVersion::capture(&file)?;
            let (metadata, header, file_bytes) =
                read_shard_metadata_from_funded(&mut file, path, funding)?;
            funding
                .dependency()
                .reserve_metadata(
                    PromptCachePersistenceFunding::dependency_bytes(funding.policy(), header.len())
                        .ok_or(WorkspaceMetadataError::Overflow)?,
                )
                .map_err(WorkspaceMetadataError::from)?;
            let tensor = metadata
                .info(&declaration.array)
                .ok_or(Cause::Declaration)?;
            let bits = tensor
                .shape
                .iter()
                .try_fold(tensor.dtype.bitsize(), |n, d| n.checked_mul(*d))
                .ok_or(WorkspaceMetadataError::Overflow)?;
            if metadata.tensors().len() != 1
                || tensor.shape.len() != declaration.shape.len()
                || tensor
                    .shape
                    .iter()
                    .zip(&declaration.shape)
                    .any(|(a, b)| usize::try_from(*b).ok() != Some(*a))
                || !stored_dtype_matches(tensor.dtype, &declaration.dtype)
                || bits % 8 != 0
                || tensor.data_offsets != (0, bits / 8)
                || u64::try_from(bits / 8).ok() != Some(declaration.logical_bytes)
                || header.len().checked_add(bits / 8) != Some(file_bytes)
                || version.byte_len() != file_bytes
            {
                return Err(Cause::Declaration);
            }
            let dtype = tensor.dtype;
            let mut bytes = allocate(file_bytes).map_err(Cause::Source)?;
            if bytes.as_ref().len() != file_bytes || bytes.as_mut().len() != file_bytes {
                return Err(Cause::Declaration);
            }
            version.bind(file)?.read_into(bytes.as_mut())?;
            if bytes.as_ref()[..header.len()] != header {
                return Err(CacheShardError::Header.into());
            }
            let mut hash = Sha256::new();
            hash.update(&bytes.as_ref()[header.len()..]);
            let actual: [u8; 32] = hash.finalize().into();
            if Some(actual) != digest_bytes(&declaration.payload_sha256) {
                return Err(Cause::Digest);
            }
            let mut shape = context
                .metadata_vec(declaration.shape.len())
                .map_err(Cause::Source)?;
            shape.extend_from_slice(&declaration.shape);
            Ok(Self {
                bytes,
                header_bytes: header.len(),
                shape,
                dtype,
                construction: account.clone(),
                dependency: funding.dependency().clone(),
            })
        };
        read().map_err(|cause| PersistentCacheReadFailure {
            cause,
            source: None,
            construction,
            dependency: funding.dependency().clone(),
        })
    }
    /// Authenticated logical shape, without copying its paid metadata.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
    /// Stored scalar encoding of the initialized payload.
    pub fn dtype(&self) -> safetensors::tensor::Dtype {
        self.dtype
    }
    /// Only the authenticated tensor bytes, excluding the retained file header.
    pub fn payload(&self) -> &[u8] {
        &self.bytes.as_ref()[self.header_bytes..]
    }
}
