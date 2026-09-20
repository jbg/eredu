//! Immutable materialized tensors over their retained checkpoint source.
use super::*;

/// Completed tensors take precedence over the original catalog. Both physical
/// stores remain retained, including original tensors hidden by replacement.
/// The constructor consumes validated tensors and materialization provenance.
/// Payload reservations remain with the tensors; catalog and sharing overhead
/// follow the caller's host-memory policy.
#[derive(Debug)]
pub struct MaterializedCheckpointSource {
    source: RetainedCheckpointSource,
    transformed: RetainedCheckpointSource,
    materialized_source_keys: BTreeSet<String>,
    materialized_source_shards: BTreeSet<PathBuf>,
}
impl MaterializedCheckpointSource {
    pub(super) fn materialization_input_bytes(&self) -> Option<usize> {
        let mut bytes = self.source.materialization_input_bytes()?
            .checked_add(self.transformed.materialization_input_bytes()?)?;
        for key in &self.materialized_source_keys {
            bytes = bytes.checked_add(std::mem::size_of::<String>())?.checked_add(key.len())?;
        }
        for path in &self.materialized_source_shards {
            bytes = bytes.checked_add(std::mem::size_of::<PathBuf>())?
                .checked_add(path.as_os_str().as_encoded_bytes().len())?;
        }
        Some(bytes)
    }
    /// Publish completed tensors without copying payloads. Input keys and shards
    /// record the original tensors consumed by the materialization plan.
    pub fn new(
        source: RetainedCheckpointSource,
        transformed: MemoryWeightStore,
        materialized_source_keys: BTreeSet<String>,
        materialized_source_shards: BTreeSet<PathBuf>,
    ) -> Self {
        Self {
            source,
            transformed: Arc::new(transformed).into(),
            materialized_source_keys,
            materialized_source_shards,
        }
    }
    /// Whether this exact key is supplied by the completed tensor store.
    pub fn is_materialized(&self, key: &str) -> bool {
        self.transformed.source_metadata_borrowed(key).is_ok()
    }
    pub(super) fn source_for(&self, key: &str) -> &RetainedCheckpointSource {
        if self.is_materialized(key) {
            &self.transformed
        } else {
            &self.source
        }
    }
    pub(super) fn encoded_read_owner(&self, keys: &[String]) -> Option<&RetainedCheckpointSource> {
        if keys.iter().all(|key| self.is_materialized(key)) {
            Some(&self.transformed)
        } else if keys.iter().all(|key| !self.is_materialized(key)) {
            Some(&self.source)
        } else {
            None
        }
    }
}

impl CheckpointSource for MaterializedCheckpointSource {
    fn prepared_acquisition_owner(self: Arc<Self>) -> Option<PreparedAcquisitionOwner> {
        Some(acquisition::retained_route::owner_materialized(self))
    }
    fn prepared_acquisition_source(&self) -> PreparedAcquisitionSource<'_> {
        PreparedAcquisitionSource(acquisition::Route::Materialized(self))
    }

    fn prepare_encoded_read(
        &self,
        keys: &[String],
    ) -> Result<Option<EncodedReadBatch>, StoreError> {
        // A batch has one exact retained source, matching the existing composed
        // source contract. Per-binding materialization reaches one of these.
        match self.encoded_read_owner(keys) {
            Some(source) => source.prepare_encoded_read(keys),
            None => Ok(None),
        }
    }

    fn source_lease_controls<'a>(
        &'a self,
        key: &'a str,
    ) -> Result<SourceLeaseControls<'a>, LeaseControlBorrowError<'a>> {
        self.source_for(key).source_lease_controls(key)
    }

    fn source_metadata_borrowed(&self, key: &str) -> SourceMetadataLoan<'_> {
        self.source_for(key).source_metadata_borrowed(key)
    }
    fn source_key_authority_borrowed(
        &self,
        key: &str,
    ) -> Result<SourceKeyAuthority, SourceMetadataBorrowError<'_>> {
        Ok(if self.is_materialized(key) {
            SourceKeyAuthority::Materialized
        } else {
            SourceKeyAuthority::Ordinary
        })
    }

    fn source_storage_slot_bound(&self) -> Result<Option<usize>, StoreError> {
        match (
            self.source.source_storage_slot_bound()?,
            self.transformed.source_storage_slot_bound()?,
        ) {
            (Some(source), Some(transformed)) => source
                .checked_add(transformed)
                .map(Some)
                .ok_or_else(|| StoreError::Overflow {
                    context: "transformed source storage slots".into(),
                }),
            _ => Ok(None),
        }
    }

    fn visit_source_storage(
        &self,
        visitor: &mut dyn FnMut(SourceStorageRef<'_>),
    ) -> Result<bool, StoreError> {
        // Hidden original owners remain live beneath the overlay. Visit both
        // physical branches even when one reports incomplete coverage.
        let source = self.source.visit_source_storage(visitor)?;
        let transformed = self.transformed.visit_source_storage(visitor)?;
        Ok(source && transformed)
    }

    fn source_storage(&self) -> Result<Option<SourceStorage>, StoreError> {
        SourceStorage::collect([self.source.as_ref(), self.transformed.as_ref()])
    }

    fn source_keys(&self) -> Vec<String> {
        self.source
            .source_keys()
            .into_iter()
            .chain(self.transformed.source_keys())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn materialized_source_keys(&self) -> Vec<String> {
        self.source
            .materialized_source_keys()
            .into_iter()
            .chain(self.materialized_source_keys.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn materialized_source_shards(&self) -> Vec<PathBuf> {
        self.materialized_source_shards.iter().cloned().collect()
    }

    fn unclaimed_checkpoint_keys(&self) -> Vec<String> {
        self.source.unclaimed_checkpoint_keys()
    }

    fn is_authoritative_materialized_key(&self, key: &str) -> bool {
        self.is_materialized(key)
    }

    fn is_checkpoint_contract_resolved(&self) -> bool {
        self.source.is_checkpoint_contract_resolved()
    }

    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.source_for(key).source_metadata(key)
    }

    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        self.source_for(key).source_provenance(key)
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.source_for(&request.key).acquire_lease(request)
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        let source = self.source.source_diagnostics()?;
        let transformed = self.transformed.source_diagnostics()?;
        let mut touched = source.touched_shard_paths;
        touched.extend(transformed.touched_shard_paths);
        touched.sort();
        touched.dedup();
        let mut payloads = source.payload_shard_paths;
        payloads.extend(transformed.payload_shard_paths);
        payloads.sort();
        payloads.dedup();
        Ok(WeightStoreDiagnostics {
            backend: source.backend,
            cache_hits: source.cache_hits.saturating_add(transformed.cache_hits),
            cache_misses: source.cache_misses.saturating_add(transformed.cache_misses),
            evictions: source.evictions.saturating_add(transformed.evictions),
            currently_cached_shards: source
                .currently_cached_shards
                .saturating_add(transformed.currently_cached_shards),
            touched_shard_paths: touched,
            payload_shard_paths: payloads,
            physical_reads: source
                .physical_reads
                .saturating_add(transformed.physical_reads),
            physical_read_bytes: source
                .physical_read_bytes
                .saturating_add(transformed.physical_read_bytes),
            coalesced_group_hits: source
                .coalesced_group_hits
                .saturating_add(transformed.coalesced_group_hits),
        })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod prepared_tests;
