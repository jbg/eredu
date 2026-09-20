//! Retained SafeTensors metadata/provenance views over an originally admitted leaf.
use super::{
    DependencyMemoryPolicy, OriginalRetainedSourceError, OriginalSafetensorsSourceError,
    WorkingMemoryError, WorkingMemoryPool, gguf_source::SourceAccount, qualified_storage,
};
use eredu_checkpoint::{
    safetensors::SafetensorsShards,
    store::{
        PreparedCheckpointSource, PreparedTensorSource, ResolvedCheckpointSource,
        RetainedCheckpointSource, SourceErasureStorageRequest, TensorMetadata,
    },
    validation::ResolvedCheckpointPlan,
};
use eredu_core::{artifact::ArtifactError, checkpoint::TensorCatalog};
use std::{fmt, mem::size_of};

#[derive(Debug)]
struct ViewCustody(SourceAccount);

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Source(#[from] OriginalSafetensorsSourceError),
    #[error(transparent)]
    Erasure(#[from] OriginalRetainedSourceError),
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
}

/// Prepared-source refusal retaining its original input and accepted view custody.
#[derive(Debug)]
pub struct OriginalPreparedSafetensorsError {
    cause: Cause,
    source: Option<RetainedCheckpointSource>,
    completed: Option<RetainedCheckpointSource>,
    account: Option<SourceAccount>,
    completion: Option<WorkingMemoryError>,
}
impl OriginalPreparedSafetensorsError {
    /// Completion failure accompanying the primary construction failure.
    pub fn completion_failure(&self) -> Option<&WorkingMemoryError> {
        self.completion.as_ref()
    }
    /// View admission or completion refusal; earlier source-stage failures retain
    /// their typed cause in the error source chain.
    pub fn memory_failure(&self) -> Option<&WorkingMemoryError> {
        match &self.cause {
            Cause::Memory(e) => Some(e),
            _ => None,
        }
    }
    /// Metadata/provenance construction failure.
    pub fn construction_failure(&self) -> Option<&ArtifactError> {
        match &self.cause {
            Cause::Artifact(e) => Some(e),
            _ => None,
        }
    }
    /// Original leaf retained when view construction was refused or failed.
    pub fn rejected_source(&self) -> Option<&RetainedCheckpointSource> {
        self.source.as_ref()
    }
}
impl fmt::Display for OriginalPreparedSafetensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for OriginalPreparedSafetensorsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl OriginalPreparedSafetensorsError {
    fn earlier(cause: impl Into<Cause>) -> Self {
        Self {
            cause: cause.into(),
            source: None,
            completed: None,
            account: None,
            completion: None,
        }
    }
}
fn outer_bytes(request: Option<SourceErasureStorageRequest>) -> Result<u64, WorkingMemoryError> {
    let request = request.ok_or(WorkingMemoryError::Overflow)?;
    qualified_storage::shared_layout_bytes(request.source_body())?
        .checked_add(qualified_storage::shared_layout_bytes(
            request.custody_body(),
        )?)
        .and_then(|n| n.checked_add(u64::try_from(request.control_bytes()).ok()?))
        .ok_or(WorkingMemoryError::Overflow)
}

impl WorkingMemoryPool {
    /// Input-derived metadata/provenance/contract-map estimate plus the actual
    /// outer wrapper and custody requests. Maps and transient copies use the
    /// configurable estimate; this is not an allocator or process-wide ceiling.
    /// Existing leaf/header storage and future recipe entries remain separate.
    pub fn prepared_safetensors_view_bytes(
        tensors: &TensorCatalog,
        resolution: &ResolvedCheckpointPlan,
        metadata: DependencyMemoryPolicy,
    ) -> Result<u64, WorkingMemoryError> {
        let mut input = Some(resolution.identity().len());
        for key in resolution
            .source_keys()
            .iter()
            .chain(resolution.unclaimed_keys())
        {
            input = input
                .and_then(|n| n.checked_add(size_of::<String>()))
                .and_then(|n| n.checked_add(key.len()));
        }
        for tensor in tensors.descriptors() {
            let path = tensor.storage.as_ref().map_or(0, |s| s.member.len());
            let dtype = match &tensor.dtype {
                eredu_core::checkpoint::TensorDtype::Encoded(name) => name.len(),
                _ => 0,
            };
            let fields = tensor
                .name
                .len()
                .checked_mul(6)
                .and_then(|n| n.checked_add(path.checked_mul(3)?))
                .and_then(|n| {
                    n.checked_add(tensor.shape.len().checked_mul(4 * size_of::<usize>())?)
                })
                .and_then(|n| n.checked_add(dtype.checked_mul(3)?))
                .and_then(|n| {
                    n.checked_add(
                        size_of::<TensorMetadata>()
                            + size_of::<PreparedTensorSource>()
                            + 2 * size_of::<String>(),
                    )
                });
            input = input.and_then(|n| n.checked_add(fields?));
        }
        let estimated = input
            .and_then(|n| metadata.estimate(n))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        [
            estimated,
            SourceAccount::storage_bytes()?,
            outer_bytes(RetainedCheckpointSource::prepared_storage_request::<
                ViewCustody,
            >())?,
            outer_bytes(RetainedCheckpointSource::resolved_storage_request::<
                ViewCustody,
            >())?,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or(WorkingMemoryError::Overflow)
    }

    /// Constructs a fresh admitted store, retains its typed leaf and pins the
    /// inspected catalog and selected contract without rediscovering artifacts.
    /// This admits source/view metadata, not graph snapshots or payload buffers.
    pub fn prepare_safetensors_artifact(
        &self,
        tensors: &TensorCatalog,
        shards: SafetensorsShards,
        resolution: &ResolvedCheckpointPlan,
        max_cached_shards: usize,
        metadata: DependencyMemoryPolicy,
    ) -> Result<RetainedCheckpointSource, OriginalPreparedSafetensorsError> {
        let source = self
            .open_admitted_safetensors_source(shards, max_cached_shards, metadata)
            .map_err(OriginalPreparedSafetensorsError::earlier)?;
        let source = self
            .retain_safetensors_source(source)
            .map_err(OriginalPreparedSafetensorsError::earlier)?;
        self.prepare_safetensors_views(source, tensors, resolution, metadata)
    }

    /// Pins this pool's original typed leaf using the same metadata comparison
    /// and provenance worker as ordinary preparation. Admission precedes every
    /// copied map and retained wrapper. All accepted custody survives refusal.
    pub fn prepare_safetensors_views(
        &self,
        source: RetainedCheckpointSource,
        tensors: &TensorCatalog,
        resolution: &ResolvedCheckpointPlan,
        metadata: DependencyMemoryPolicy,
    ) -> Result<RetainedCheckpointSource, OriginalPreparedSafetensorsError> {
        let admit = || {
            self.validate_retained_source_controls(&source)?;
            let bytes = Self::prepared_safetensors_view_bytes(tensors, resolution, metadata)?;
            Ok::<_, WorkingMemoryError>(self.admit_source_compiler(bytes)?.into_source_account())
        };
        let account = match admit() {
            Ok(account) => account,
            Err(error) => {
                return Err(OriginalPreparedSafetensorsError {
                    cause: Cause::Memory(error),
                    source: Some(source),
                    completed: None,
                    account: None,
                    completion: None,
                });
            }
        };
        let result = (|| {
            let catalog = eredu_core::artifact::safetensors_artifact_catalog(tensors)?;
            let prepared = PreparedCheckpointSource::pin_safetensors(source.clone(), catalog)?;
            let prepared = RetainedCheckpointSource::from_prepared_with_custody(
                prepared,
                ViewCustody(account.share()),
            );
            let resolved = ResolvedCheckpointSource::new(prepared, resolution.clone());
            Ok::<_, ArtifactError>(RetainedCheckpointSource::from_resolved_with_custody(
                resolved,
                ViewCustody(account.share()),
            ))
        })();
        let completion = account.finish();
        match (result, completion) {
            (Ok(completed), Ok(())) => Ok(completed),
            (Ok(completed), Err(error)) => Err(OriginalPreparedSafetensorsError {
                cause: Cause::Memory(error),
                source: Some(source),
                completed: Some(completed),
                account: Some(account),
                completion: None,
            }),
            (Err(error), completion) => Err(OriginalPreparedSafetensorsError {
                cause: Cause::Artifact(error),
                source: Some(source),
                completed: None,
                account: Some(account),
                completion: completion.err(),
            }),
        }
    }
    pub(super) fn validate_prepared_safetensors_controls(
        &self,
        source: &RetainedCheckpointSource,
    ) -> Result<(), WorkingMemoryError> {
        let custody = source
            .constructor_control_owner::<ViewCustody>()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if custody.0.matches_pool(self) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}

#[cfg(test)]
mod tests;
