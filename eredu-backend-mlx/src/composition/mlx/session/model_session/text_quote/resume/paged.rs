//! Actual installed paged manager joins the saved run after copy completion.
use super::*;
use crate::backend::nn::workspace::{ProjectedPagedSources, ProjectedResidentState};
use std::{
    mem::{size_of, size_of_val},
    num::NonZeroU32,
};

impl PendingSavedTextAdmission {
    pub(super) fn install_paged_sources(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
    ) -> Result<(), Error> {
        let Some(saved) = self.source_native() else {
            return Ok(());
        };
        if saved.paged_sources().is_empty() {
            return Ok(());
        }
        let quote = self.quote();
        let context = self
            .paged_source_metadata
            .as_ref()
            .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?;
        let fail = |cause| Error::Neural(context.metadata_source(cause));
        let controls = [
            size_of::<(&Self, &ModelRuntime<MlxBackend<'_>>)>(),
            size_of::<ProjectedResidentState>(),
            size_of::<Result<ProjectedResidentState, Error>>(),
            size_of::<Option<ProjectedPagedSources>>(),
            size_of::<Option<eredu_runtime::working_memory::HostSourceConstructionFacts>>(),
            size_of::<Result<(), Error>>(),
        ];
        context
            .charge_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or_else(|| fail(WorkingMemoryError::Overflow))?,
            )
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        if quote
            .paged_sources
            .try_borrow()
            .map_err(|_| fail(WorkingMemoryError::ExecutionFenced))?
            .is_some()
        {
            return Err(fail(WorkingMemoryError::IdentityMismatch));
        }
        let model = &runtime.session().payload.model;
        let recipe = quote
            .native_recipe
            .as_ref()
            .ok_or_else(|| fail(WorkingMemoryError::UnknownBound))?;
        let batch = u32::try_from(quote.request.geometry().batch_size)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or_else(|| fail(WorkingMemoryError::IdentityMismatch))?;
        let mut actual = model
            .erased()
            .project_resident_workspace_with_storage(batch, context)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        actual
            .storage
            .inherit_copied_paged_host_traces(saved, context)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let mut source = actual
            .storage
            .take_paged_sources(context)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?
            .ok_or_else(|| fail(WorkingMemoryError::IdentityMismatch))?;
        // No native work was submitted during inspection. Drop temporary Array
        // witnesses before another fallible step could retire their source pins.
        drop(actual);
        source
            .prepare_catalogs(recipe.plan(), context)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        source
            .prepare_host_program(runtime.backend().memory_pool(), context)
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        let actual_facts = source
            .host_source_facts()
            .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
        if actual_facts != self.paged_host_facts {
            return Err(fail(WorkingMemoryError::IdentityMismatch));
        }
        let mut destination = quote
            .paged_sources
            .try_borrow_mut()
            .map_err(|_| fail(WorkingMemoryError::ExecutionFenced))?;
        if destination.is_some() {
            return Err(fail(WorkingMemoryError::IdentityMismatch));
        }
        *destination = Some(source);
        Ok(())
    }
}
