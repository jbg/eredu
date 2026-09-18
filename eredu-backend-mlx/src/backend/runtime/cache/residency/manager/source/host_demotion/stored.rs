//! Closed immutable Host publication, after both actual copies completed.
use super::*;
use crate::backend::nn::workspace::OriginalPagedScanSource;

/// Actual registered Host backing and original source account. This can only
/// be formed from this worker's positively completed canonical publication.
pub(crate) struct StoredCacheHostSource {
    host: HostCacheBlock,
    id: CacheBlockId,
    manager: CacheResidencyManager,
    generation: u64,
    custody: eredu_runtime::working_memory::OriginalHostSourceCustody,
    context: WorkspaceContext,
    _funding: HostMetadataFunding,
}
impl PreparedCacheHostDemotion {
    pub(crate) fn stored_source(&self) -> Option<StoredCacheHostSource> {
        if !self.completed || !self.published {
            return None;
        }
        let first = Arc::clone(self.host[0].as_ref()?);
        let second = Arc::clone(self.host[1].as_ref()?);
        let host = match self.id.representation {
            CacheRepresentation::KeyValue => HostCacheBlock::KeyValue {
                keys: first,
                values: second,
            },
            CacheRepresentation::CompressedLatentRotary => HostCacheBlock::CompressedLatentRotary {
                latent: first,
                rotary_key: second,
            },
        };
        Some(StoredCacheHostSource {
            host,
            id: self.id.clone(),
            manager: self.manager.clone(),
            generation: self.generation,
            custody: self.source_custody.as_ref()?.clone(),
            context: self.context.clone(),
            _funding: self._funding.clone(),
        })
    }
}
impl StoredCacheHostSource {
    pub(crate) fn control_bytes() -> Option<usize> {
        Some(size_of::<(
            Self,
            Option<Self>,
            &PreparedCacheHostDemotion,
            HostCacheBlock,
            [Arc<ImmutableHostTransferBuffer>; 2],
            Result<(), Exception>,
        )>())
    }
    pub(crate) fn validate(
        &self,
        proof: &OriginalPagedScanSource<'_>,
        id: &CacheBlockId,
        actual: [&ImmutableHostTransferBuffer; 2],
    ) -> Result<(), Exception> {
        proof.validate_manager(&self.manager, self.generation)?;
        if id != &self.id
            || !self.custody.same_source(proof.host_source_custody())
            || !self.context.shares_trace(proof.context())
        {
            return Err(proof.error(CacheSourceError::Identity));
        }
        if self
            .host
            .buffers()
            .into_iter()
            .zip(actual)
            .any(|(expected, actual)| !std::ptr::eq(expected, actual))
        {
            return Err(proof.error(CacheSourceError::Identity));
        }
        Ok(())
    }
}
