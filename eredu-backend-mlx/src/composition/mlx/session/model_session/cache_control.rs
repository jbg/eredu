//! Retained source preparation for the existing distributed cache protocol.
use super::*;
use crate::backend::runtime::distributed::topology::original_source::control::cache::CacheControlOwner;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataAllocation};
use eredu_runtime::replicated_session::{SessionCacheControlOperation, SessionCacheControlPlan};

impl MlxModelSession {
    pub(super) fn prepare_cache_materialization(
        &self,
        backend: &MlxBackend<'_>,
        funding: &eredu_runtime::cache::PromptCachePersistenceFunding,
    ) -> Result<crate::backend::runtime::cache::residency::PromptCacheMaterialization, Error> {
        use crate::backend::OriginalCopyEnvironment;
        funding
            .context()
            .charge_metadata(OriginalCopyEnvironment::control_bytes().ok_or(
                Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::Overflow),
            )?)
            .map_err(|cause| Error::Neural(cause.into()))?;
        let host = funding
            .context()
            .metadata_funding()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let result = (|| {
            let environment = backend
                .original_copy_environment()
                .map_err(|cause| Error::Neural(host.metadata_source(cause)))?;
            if !environment.pool().same_ledger(&self.payload.memory_ledger) {
                return Err(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ));
            }
            let runtime = environment
                .input_runtime()
                .map_err(|cause| Error::Neural(host.metadata_source(cause)))?;
            crate::backend::runtime::cache::residency::PromptCacheMaterialization::new(
                environment.pool(),
                self.payload.model.erased().inference_execution_identity(),
                runtime,
                funding.context(),
            )
            .map_err(Error::Neural)
        })();
        result.map_err(|cause| crate::composition::mlx::model::retain_planning_error(cause, host))
    }

    pub(super) fn prepare_cache_persistence(
        &self,
    ) -> Result<eredu_runtime::cache::PromptCachePersistenceFunding, Error> {
        let funding = self
            .original_model_source()
            .map_err(Error::PrefillControl)?
            .prepare_prompt_cache_persistence(&self.payload.memory_ledger)?;
        funding
            .context()
            .charge_metadata(std::mem::size_of::<(
                &mut Self,
                &MlxBackend<'_>,
                &Path,
                PromptCacheDescriptor,
                &[u32],
                &PromptCacheOptions,
                Option<CacheControlOwner>,
                eredu_runtime::SharedPreparedInputCacheIdentity,
                eredu_core::cache::SharedPromptCacheManifest,
                Result<eredu_core::cache::SharedPromptCacheManifest, Error>,
            )>())
            .map_err(|cause| Error::Neural(cause.into()))?;
        Ok(funding)
    }

    pub(super) fn prepare_cache_control(
        &self,
        operation: SessionCacheControlOperation,
        context: &WorkspaceContext,
    ) -> Result<Option<CacheControlOwner>, Error> {
        context
            .charge_metadata(std::mem::size_of::<(
                &Self,
                SessionCacheControlOperation,
                SessionCacheControlPlan,
                &WorkspaceContext,
                Option<CacheControlOwner>,
                Result<Option<CacheControlOwner>, Error>,
            )>())
            .map_err(|cause| Error::Neural(cause.into()))?;
        if !self.payload.model.has_neutral_partitioned_control() {
            return Ok(None);
        }
        let funding = context
            .metadata_funding()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let result = (|| {
            let model = self.payload.model.erased();
            let plan = model.cache_control_plan(operation)?;
            let distributed = self
                .payload
                .distributed
                .as_ref()
                .ok_or(Error::PrefillScopeUnavailable)?;
            let selected = self
                .payload
                .model
                .inference_blueprint()
                .ok_or(Error::PrefillScopeUnavailable)?
                .selected();
            let manifest = selected
                .communication_manifest()
                .ok_or(Error::PrefillScopeUnavailable)?;
            let group = selected
                .partitioned_session_group()
                .ok_or(Error::PrefillScopeUnavailable)?;
            let source = distributed
                .original_communication_owner(manifest, distributed.native_world(), &funding)?
                .prepare_control_source(group, &self.payload.memory_ledger)?;
            CacheControlOwner::new(
                source,
                plan,
                &self.payload.memory_ledger,
                model.inference_execution_identity(),
                context,
            )
            .map(Some)
        })();
        result
            .map_err(|cause| crate::composition::mlx::model::retain_planning_error(cause, funding))
    }
}
