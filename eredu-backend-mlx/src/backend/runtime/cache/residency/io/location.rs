//! Paid immutable publication descriptor shared by canonical and queued owners.
use super::*;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};
use eredu_runtime::cache::{PreparedLiveCachePublication, cache_shard_tensor_names};
use std::mem::{size_of, size_of_val};

impl DiskLocation {
    /// Publishes only the immutable descriptor of an authenticated persistent
    /// import. Its file has no live-writer reservation or unlink owner.
    pub(crate) fn prepare_persistent(
        source: eredu_runtime::cache::PersistentCacheBlockSource,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let funding = context
            .metadata_funding()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let names = source.layout().names();
        let frames = [
            size_of::<Self>(),
            size_of::<DiskLocationData>(),
            size_of::<eredu_runtime::cache::PersistentCacheBlockSource>(),
            size_of::<Result<Self, Error>>(),
            size_of::<[u8; 64]>(),
            size_of::<(&WorkspaceContext, [&str; 2])>(),
            source.path().as_os_str().len(),
            names[0].len(),
            names[1].len(),
            64,
        ];
        context.charge_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let mut digest = String::with_capacity(64);
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for byte in source.payload_digest() {
            digest.push(char::from(HEX[usize::from(byte >> 4)]));
            digest.push(char::from(HEX[usize::from(byte & 15)]));
        }
        let data = DiskLocationData {
            path: source.path().to_path_buf(),
            first_name: names[0].to_owned(),
            second_name: names[1].to_owned(),
            persistent: true,
            buffered: None,
            payload_sha256: Some(digest),
            payload_verification: context.metadata_arc(OnceLock::new())?,
            live_source: None,
            persistent_source: Some(source),
        };
        Ok(Self {
            inner: context.metadata_arc(data)?,
            funding: Some(funding),
        })
    }
    pub(super) fn original_control_bytes(
        path: &Path,
        representation: CacheRepresentation,
    ) -> Option<usize> {
        let names = cache_shard_tensor_names(representation);
        let parts = [
            path.as_os_str().len(),
            names[0].len(),
            names[1].len(),
            WorkspaceContext::metadata_arc_bytes::<DiskLocationData>()?,
            WorkspaceContext::metadata_arc_bytes::<OnceLock<Result<(), String>>>()?,
            size_of::<Self>(),
            size_of::<DiskLocationData>(),
            size_of::<[&str; 2]>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Option<LiveCacheBlockSource>>(),
            size_of::<(
                &PreparedLiveCachePublication,
                CacheRepresentation,
                &WorkspaceContext,
            )>(),
            size_of::<(&mut Self, LiveCacheBlockSource)>(),
            size_of::<Result<(), LiveCacheBlockSource>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Qualifies the actual path/names/verification/descriptor allocations.
    /// This unbound private value neither opens a file nor grants read authority.
    pub(super) fn prepare_live(
        publication: &PreparedLiveCachePublication,
        representation: CacheRepresentation,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        let funding = context
            .metadata_funding()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let total = Self::original_control_bytes(publication.destination_path(), representation)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let arcs = WorkspaceContext::metadata_arc_bytes::<DiskLocationData>()
            .and_then(|n| {
                n.checked_add(WorkspaceContext::metadata_arc_bytes::<
                    OnceLock<Result<(), String>>,
                >()?)
            })
            .ok_or(WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(
            total
                .checked_sub(arcs)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let names = cache_shard_tensor_names(representation);
        let verification = context.metadata_arc(OnceLock::new())?;
        let data = DiskLocationData {
            path: publication.destination_path().to_path_buf(),
            first_name: names[0].to_owned(),
            second_name: names[1].to_owned(),
            persistent: false,
            buffered: None,
            payload_sha256: None,
            payload_verification: verification,
            live_source: None,
            persistent_source: None,
        };
        Ok(Self {
            inner: context.metadata_arc(data)?,
            funding: Some(funding),
        })
    }
    /// Consumes only this uniquely prepared publication slot. Path, layout,
    /// version and file ownership all come from the actual writer result.
    pub(super) fn bind_live(
        &mut self,
        source: LiveCacheBlockSource,
    ) -> Result<(), LiveCacheBlockSource> {
        if self.funding.is_none()
            || source.path() != self.path.as_path()
            || source.writer_layout().is_none_or(|layout| {
                layout.names() != [self.first_name.as_str(), self.second_name.as_str()]
            })
        {
            return Err(source);
        }
        let Some(data) = Arc::get_mut(&mut self.inner) else {
            return Err(source);
        };
        if data.live_source.is_some() {
            return Err(source);
        }
        data.live_source = Some(source);
        Ok(())
    }
}
