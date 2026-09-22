//! Metadata and finite dependency admission for the existing persistence worker.
use super::*;
use eredu_nn::workspace::WorkspaceContext;

impl Executable {
    pub(crate) fn prepare_prompt_cache_persistence(
        &self,
        ledger: &eredu_runtime::working_memory::MemoryLedger,
    ) -> Result<eredu_runtime::cache::PromptCachePersistenceFunding, Error> {
        use eredu_runtime::{
            cache::{
                DEFAULT_PROMPT_CACHE_MANIFEST_BYTE_LIMIT, PROMPT_CACHE_DEPENDENCY_BASIS,
                PROMPT_CACHE_DEPENDENCY_SOURCE, PromptCachePersistenceFunding,
            },
            working_memory::DependencyMemoryPolicy,
        };
        let model = self;
        let facts = model
            .workspace_mechanisms()
            .ok_or(Error::PrefillScopeUnavailable)?;
        let execution = model.erased().inference_execution_identity();
        // This host construction owner uses configured and all live ceilings.
        // No limit vector is copied before its bookkeeping has been admitted.
        let prepared = ledger
            .prepare_construction_metadata()
            .map_err(Error::WorkspacePlanning)?;
        let construction = prepared.funding().clone();
        let result = (|| {
            construction
                .reserve_metadata(std::mem::size_of::<(
                    &Self,
                    &eredu_runtime::working_memory::MemoryLedger,
                    &eredu_runtime::working_memory::InferenceExecutionIdentity,
                    WorkspaceContext,
                    eredu_core::HostMetadataFunding,
                    PromptCachePersistenceFunding,
                    Result<PromptCachePersistenceFunding, Error>,
                )>())
                .map_err(Error::WorkspacePlanning)?;
            let context = WorkspaceContext::new_with_metadata_funding(facts, construction.clone())
                .map_err(|cause| Error::Neural(cause.into()))?;
            let dependency = ledger
                .prepare_dependency_estimates(
                    execution,
                    ledger.configured_limits(),
                    PROMPT_CACHE_DEPENDENCY_SOURCE,
                    PROMPT_CACHE_DEPENDENCY_BASIS,
                    construction.clone(),
                )
                .map_err(Error::WorkspacePlanning)?;
            let storage = prepared
                .storage_funding()
                .map_err(Error::WorkspacePlanning)?;
            PromptCachePersistenceFunding::new_with_storage(
                &context,
                storage,
                dependency,
                DependencyMemoryPolicy::default(),
                DEFAULT_PROMPT_CACHE_MANIFEST_BYTE_LIMIT,
            )
            .map_err(|cause| Error::Neural(cause.into()))
        })();
        result.map_err(|cause| {
            crate::composition::mlx::model::retain_planning_error(cause, construction)
        })
    }
}
