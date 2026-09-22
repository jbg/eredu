//! Original SafeTensors source opening under one retained pool contribution.
use super::{
    DependencyMemoryPolicy, MemoryLedger, WorkingMemoryError, gguf_source::SourceAccount,
    qualified_storage,
};
use eredu_checkpoint::{
    safetensors::{
        SafetensorsDiscoveryLimits, SafetensorsHeaderAdmission, SafetensorsIndexRequest,
        SafetensorsShards, SafetensorsSourceAdmission,
    },
    store::{SafetensorsWeightStore, StoreError},
};
use std::{
    error::Error,
    fmt,
    path::Path,
    sync::{Arc, Weak},
};

#[derive(Debug)]
struct SourcePolicy {
    pool: Weak<super::Pool>,
    metadata: DependencyMemoryPolicy,
    account: SourceAccount,
}
impl SafetensorsSourceAdmission for SourcePolicy {
    fn reserve_index(
        &self,
        request: SafetensorsIndexRequest,
    ) -> Result<(), Arc<dyn Error + Send + Sync>> {
        MemoryLedger::safetensors_source_index_bytes(request, self.metadata)
            .and_then(|bytes| self.account.reserve_more(bytes))
            .map_err(|error| self.refusal(error))
    }
    fn headers(
        &self,
        count: usize,
    ) -> Result<Arc<dyn SafetensorsHeaderAdmission>, Arc<dyn Error + Send + Sync>> {
        self.pool
            .upgrade()
            .ok_or(WorkingMemoryError::IdentityMismatch)
            .and_then(|pool| MemoryLedger(pool).safetensors_header_admission(count, self.metadata))
            .map_err(|error| self.refusal(error))
    }
    fn reserve_store(
        &self,
        metadata_input_bytes: usize,
    ) -> Result<(), Arc<dyn Error + Send + Sync>> {
        estimate(self.metadata, metadata_input_bytes)
            .and_then(|bytes| self.account.reserve_more(bytes))
            .map_err(|error| self.refusal(error))
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct SourceRefusal {
    #[source]
    cause: WorkingMemoryError,
    _account: SourceAccount,
}
impl SourcePolicy {
    fn refusal(&self, cause: WorkingMemoryError) -> Arc<dyn Error + Send + Sync> {
        Arc::new(SourceRefusal {
            cause,
            _account: self.account.share(),
        })
    }
}
fn estimate(policy: DependencyMemoryPolicy, bytes: usize) -> Result<u64, WorkingMemoryError> {
    policy
        .estimate(bytes)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)
}

/// Source opening failure retaining its accepted contribution and completed prefix.
#[derive(Debug)]
pub struct OriginalSafetensorsSourceError {
    memory: Option<WorkingMemoryError>,
    construction: Option<StoreError>,
    _completed: Option<SafetensorsWeightStore>,
    _shards: Option<SafetensorsShards>,
    _policy: Option<Arc<SourcePolicy>>,
}
impl OriginalSafetensorsSourceError {
    /// Exact retained shards when a fresh-store constructor was refused.
    pub fn rejected_shards(&self) -> Option<&SafetensorsShards> {
        self._shards.as_ref()
    }
    /// Initial pool refusal or construction-completion failure.
    /// Later phase refusals remain in the construction error's source chain.
    pub fn memory_failure(&self) -> Option<&WorkingMemoryError> {
        self.memory.as_ref()
    }
    /// Typed discovery, parsing, identity or incremental admission error.
    pub fn construction_failure(&self) -> Option<&StoreError> {
        self.construction.as_ref()
    }
}
impl fmt::Display for OriginalSafetensorsSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(error) = &self.construction {
            error.fmt(f)
        } else {
            self.memory.as_ref().expect("source failure").fmt(f)
        }
    }
}
impl Error for OriginalSafetensorsSourceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.construction
            .as_ref()
            .map(|error| error as _)
            .or_else(|| self.memory.as_ref().map(|error| error as _))
    }
}
/// Inspection failure retaining accepted source and tensor-catalog metadata.
#[derive(Debug)]
pub struct OriginalArtifactInspectionError<P: fmt::Debug> {
    memory: Option<WorkingMemoryError>,
    construction: Option<eredu_core::artifact::ArtifactError>,
    _completed: Option<eredu_core::ArtifactInspection<P>>,
    _policy: Option<Arc<SourcePolicy>>,
}
impl<P: fmt::Debug> OriginalArtifactInspectionError<P> {
    /// Initial admission or completion refusal.
    pub fn memory_failure(&self) -> Option<&WorkingMemoryError> {
        self.memory.as_ref()
    }
    /// Actual inspection failure, including typed incremental pool refusals.
    pub fn construction_failure(&self) -> Option<&eredu_core::artifact::ArtifactError> {
        self.construction.as_ref()
    }
}
impl<P: fmt::Debug> fmt::Display for OriginalArtifactInspectionError<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(error) = &self.construction {
            error.fmt(f)
        } else {
            self.memory.as_ref().expect("inspection failure").fmt(f)
        }
    }
}
impl<P: fmt::Debug> Error for OriginalArtifactInspectionError<P> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.construction
            .as_ref()
            .map(|e| e as _)
            .or_else(|| self.memory.as_ref().map(|e| e as _))
    }
}

impl MemoryLedger {
    /// Loading entry that installs source admission before inspection when the
    /// pool can establish it. Unknown admission before any construction retains
    /// ordinary inspection behavior. Once any contribution is accepted, every
    /// failure propagates without reopening or retrying the artifact.
    pub fn inspect_artifact_for_loading<R: eredu_core::ModelConfigurationResolver>(
        &self,
        path: impl AsRef<Path>,
        resolver: &R,
        limits: SafetensorsDiscoveryLimits,
        metadata: DependencyMemoryPolicy,
    ) -> Result<
        eredu_core::ArtifactInspection<R::ArtifactPlan>,
        OriginalArtifactInspectionError<R::ArtifactPlan>,
    > {
        let path = path.as_ref();
        match self.inspect_artifact_with_safetensors_pool(path, resolver, limits, metadata) {
            Err(error)
                if matches!(error.memory, Some(WorkingMemoryError::UnknownBound))
                    && error._policy.is_none()
                    && error.construction.is_none() =>
            {
                eredu_core::artifact::inspect_artifact_with_safetensors_limits(
                    path, resolver, limits,
                )
                .map_err(|construction| OriginalArtifactInspectionError {
                    memory: None,
                    construction: Some(construction),
                    _completed: None,
                    _policy: None,
                })
            }
            result => result,
        }
    }

    /// Reports whether the exact shards have this pool's original discovery
    /// policy. Ordinary or caller-defined policies are not promoted; an original
    /// policy from another pool is an identity error rather than an ordinary input.
    pub fn safetensors_shards_have_source_admission(
        &self,
        shards: &SafetensorsShards,
    ) -> Result<bool, WorkingMemoryError> {
        match shards.source_admission_owner::<SourcePolicy>() {
            None => Ok(false),
            Some(policy) if policy.account.matches_pool(self) => Ok(true),
            Some(_) => Err(WorkingMemoryError::IdentityMismatch),
        }
    }

    /// Builds independent store/cache metadata over this pool's already admitted
    /// shards. No directory, index or header is rediscovered. The input and any
    /// fresh accepted contribution remain owned on failure.
    pub fn open_admitted_safetensors_source(
        &self,
        shards: SafetensorsShards,
        max_cached_shards: usize,
        metadata: DependencyMemoryPolicy,
    ) -> Result<SafetensorsWeightStore, OriginalSafetensorsSourceError> {
        let refused = |shards, memory| OriginalSafetensorsSourceError {
            memory: Some(memory),
            construction: None,
            _completed: None,
            _shards: Some(shards),
            _policy: None,
        };
        let Some(origin) = shards.source_admission_owner::<SourcePolicy>() else {
            return Err(refused(shards, WorkingMemoryError::UnknownBound));
        };
        if !origin.account.matches_pool(self) {
            return Err(refused(shards, WorkingMemoryError::IdentityMismatch));
        }
        let policy = match self.new_safetensors_policy(Path::new(""), metadata) {
            Ok(policy) => policy,
            Err(error) => return Err(refused(shards, error)),
        };
        let result = SafetensorsWeightStore::open_admitted_with_source_admission(
            shards.clone(),
            max_cached_shards,
            Some(policy.clone()),
        );
        let completion = policy.account.finish().err();
        match (result, completion) {
            (Ok(source), None) => Ok(source),
            (Ok(source), Some(memory)) => Err(OriginalSafetensorsSourceError {
                memory: Some(memory),
                construction: None,
                _completed: Some(source),
                _shards: Some(shards),
                _policy: Some(policy),
            }),
            (Err(construction), memory) => Err(OriginalSafetensorsSourceError {
                memory,
                construction: Some(construction),
                _completed: None,
                _shards: Some(shards),
                _policy: Some(policy),
            }),
        }
    }
    fn new_safetensors_policy(
        &self,
        path: &Path,
        metadata: DependencyMemoryPolicy,
    ) -> Result<Arc<SourcePolicy>, WorkingMemoryError> {
        let bytes = Self::safetensors_source_initial_bytes(path, metadata)?;
        let account = self.admit_source_compiler(bytes)?.into_source_account();
        Ok(Arc::new(SourcePolicy {
            pool: Arc::downgrade(&self.0),
            metadata,
            account,
        }))
    }

    /// Performs portable inspection with source admission installed before
    /// SafeTensors discovery. Tensor-catalog clones share their map and custody.
    /// Configuration and architecture-plan allocations remain separate policy;
    /// GGUF uses captured headers without claiming SafeTensors source admission.
    pub fn inspect_artifact_with_safetensors_pool<R: eredu_core::ModelConfigurationResolver>(
        &self,
        path: impl AsRef<Path>,
        resolver: &R,
        limits: SafetensorsDiscoveryLimits,
        metadata: DependencyMemoryPolicy,
    ) -> Result<
        eredu_core::ArtifactInspection<R::ArtifactPlan>,
        OriginalArtifactInspectionError<R::ArtifactPlan>,
    > {
        let policy = self
            .new_safetensors_policy(path.as_ref(), metadata)
            .map_err(|memory| OriginalArtifactInspectionError {
                memory: Some(memory),
                construction: None,
                _completed: None,
                _policy: None,
            })?;
        let result = eredu_core::artifact::inspect_artifact_with_safetensors_admission(
            path,
            resolver,
            limits,
            policy.clone(),
        );
        let completion = policy.account.finish().err();
        match (result, completion) {
            (Ok(inspection), None) => Ok(inspection),
            (Ok(inspection), Some(memory)) => Err(OriginalArtifactInspectionError {
                memory: Some(memory),
                construction: None,
                _completed: Some(inspection),
                _policy: Some(policy),
            }),
            (Err(construction), memory) => Err(OriginalArtifactInspectionError {
                memory,
                construction: Some(construction),
                _completed: None,
                _policy: Some(policy),
            }),
        }
    }

    /// Validate the actual private source policy installed by this pool's
    /// original constructor. Public policy callbacks cannot supply this origin.
    /// This covers discovery/header policy and initial store metadata, not later
    /// payload/cache/read storage.
    pub fn validate_safetensors_source_controls(
        &self,
        source: &SafetensorsWeightStore,
    ) -> Result<(), WorkingMemoryError> {
        let policy = source
            .source_admission_owner::<SourcePolicy>()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if policy.account.matches_pool(self) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }

    /// Initial policy/account controls, potential admission error Arcs and
    /// configurable path/discovery headroom, before any filesystem discovery.
    /// Path and map estimates are not enforceable allocation ceilings.
    pub fn safetensors_source_initial_bytes(
        path: &Path,
        metadata: DependencyMemoryPolicy,
    ) -> Result<u64, WorkingMemoryError> {
        [
            estimate(metadata, path.as_os_str().len())?,
            SourceAccount::storage_bytes()?,
            qualified_storage::shared_bytes::<SourcePolicy>()?,
            qualified_storage::shared_bytes::<SourceRefusal>()?
                .checked_mul(3)
                .ok_or(WorkingMemoryError::Overflow)?,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or(WorkingMemoryError::Overflow)
    }
    /// Exact encoded index buffer plus configurable decoded-index/catalog
    /// headroom. The reader rejects growth rather than expanding this buffer.
    pub fn safetensors_source_index_bytes(
        request: SafetensorsIndexRequest,
        metadata: DependencyMemoryPolicy,
    ) -> Result<u64, WorkingMemoryError> {
        let input = request
            .encoded_bytes
            .checked_add(request.path_bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        u64::try_from(request.encoded_bytes)
            .ok()
            .and_then(|n| n.checked_add(estimate(metadata, input).ok()?))
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Opens a source in this pool before discovery allocates its paths/index.
    /// The index is decoded once; its distinct shard count sizes the separate
    /// header policy. Headers remain lazy for indexed checkpoints. Source maps,
    /// file identities and fixed diagnostics retain the source contribution.
    /// Metadata is estimated; payload/read buffers, independently exported
    /// metadata and subsequent cache contents require separate admission.
    pub fn open_safetensors_source(
        &self,
        path: impl AsRef<Path>,
        max_cached_shards: usize,
        limits: SafetensorsDiscoveryLimits,
        metadata: DependencyMemoryPolicy,
    ) -> Result<SafetensorsWeightStore, OriginalSafetensorsSourceError> {
        let rejected = |memory| OriginalSafetensorsSourceError {
            memory: Some(memory),
            construction: None,
            _completed: None,
            _shards: None,
            _policy: None,
        };
        let path = path.as_ref();
        let policy = self
            .new_safetensors_policy(path, metadata)
            .map_err(rejected)?;
        let result = SafetensorsWeightStore::open_with_source_admission(
            path,
            max_cached_shards,
            limits,
            policy.clone(),
        );
        let completion = policy.account.finish().err();
        match (result, completion) {
            (Ok(source), None) => Ok(source),
            (Ok(source), Some(memory)) => Err(OriginalSafetensorsSourceError {
                memory: Some(memory),
                construction: None,
                _completed: Some(source),
                _shards: None,
                _policy: Some(policy),
            }),
            (Err(construction), memory) => Err(OriginalSafetensorsSourceError {
                memory,
                construction: Some(construction),
                _completed: None,
                _shards: None,
                _policy: Some(policy),
            }),
        }
    }
}

#[cfg(test)]
mod tests;
