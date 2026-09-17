//! Original selected bank source preparation before ordinary native loading.
use super::*;
use crate::backend::runtime::residency::parameter_bank::{
    entries_from_selected_members, AddressableParameterBank,
};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound,
};

pub(crate) fn prepare_addressable_source(
    sources: &PreparedModelSources,
    pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    source_stream: &Stream,
    execution_stream: &Stream,
) -> Result<Option<crate::backend::runtime::residency::parameter_bank::PreparedAddressableSource>, Error> {
    let context = WorkspaceContext::new(DestinationFacts);
    let Some(projected) =
        eredu_architectures::prepared_execution::project_addressable_binding_destinations(
            sources, &context,
        )
        .map_err(|cause| Error::ArchitectureModel(cause.to_string()))?
    else {
        return Ok(None);
    };
    let local;
    let members = if let Some(layout) = projected.layout() {
        local = super::super::replicated_text::shard_addressable_members(
            projected.members(),
            sources.target().as_ref(),
            layout,
        )?;
        local.as_slice()
    } else {
        projected.members()
    };
    let selected = entries_from_selected_members(members, sources.target().as_ref())?;
    AddressableParameterBank::prepare_selected_manager(
        sources.target().clone(),
        selected,
        projected.options(),
        source_stream,
        execution_stream,
        pool,
    )
    .map_err(Into::into)
}
#[derive(Debug)]
struct DestinationFacts;
impl WorkspaceMechanisms for DestinationFacts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        Ok(None)
    }
}
