//! Joins each recorded layerwise visit with its actual foreground producer.
use super::*;
use crate::backend::{
    error::Error as BackendError, runtime::execution::generic::LayerwiseWorkspace,
};
use eredu_core::DomainMemoryRequirements;
use eredu_runtime::working_memory::{MemoryLedger, WorkingMemoryError, WorkspaceReportMetadata};

impl ResidentNativeRecipe {
    pub(crate) fn bind_ordinary_materialization(
        &mut self,
        source: &LayerwiseWorkspace,
        runtime: &safemlx::PreparedInputRuntime,
        allocation: super::super::super::NativeAllocationFacts,
        pool: &MemoryLedger,
        context: &WorkspaceContext,
    ) -> Result<Option<DomainMemoryRequirements>, BackendError> {
        let reports = WorkspaceReportMetadata::new(context);
        let mut requirements = None;
        context
            .charge_metadata(size_of::<(
                &mut Self,
                &LayerwiseWorkspace,
                &safemlx::PreparedInputRuntime,
                super::super::super::NativeAllocationFacts,
                &MemoryLedger,
                &WorkspaceContext,
                Option<DomainMemoryRequirements>,
                Result<Option<DomainMemoryRequirements>, BackendError>,
            )>())
            .map_err(|cause| BackendError::Neural(cause.into()))?;
        for row in &mut self.records {
            let ordinals =
                row.layerwise_ordinals
                    .as_deref()
                    .ok_or(BackendError::PrefillControl(
                        WorkingMemoryError::UnknownBound,
                    ))?;
            let quote = source.ordinary_materialization_requirements(
                ordinals, pool, runtime, allocation, context,
            )?;
            if let Some(native) = quote.requirements {
                requirements = Some(match requirements.take() {
                    None => native,
                    Some(previous) => reports
                        .combine_domain_requirements(&previous, &native, true)
                        .map_err(|cause| BackendError::Neural(reports.error(cause)))?,
                });
            }
            let calls = row
                .ordinary_calls
                .as_mut()
                .ok_or(BackendError::PrefillControl(
                    WorkingMemoryError::UnknownBound,
                ))?;
            calls.metadata_bytes = calls
                .metadata_bytes
                .checked_add(quote.metadata)
                .ok_or(BackendError::PrefillControl(WorkingMemoryError::Overflow))?;
        }
        Ok(requirements)
    }
}
