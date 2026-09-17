//! Cold inventory and the admitted clone of actual non-target manager rows.
use super::*;
use crate::backend::nn::workspace::MetalAllocationFacts;
use std::num::NonZeroUsize;

pub(super) struct SupplementarySourcePlan {
    ids: Vec<OffloadUnitId>,
    definitions: Vec<OffloadUnit>,
    layout: ExecutionUnitLayout,
}
impl SupplementarySourcePlan {
    pub(super) fn prepare(
        units: &[OffloadUnit],
        selected: &[OffloadUnitId],
    ) -> Result<Option<Self>, WorkingMemoryError> {
        let definitions: Vec<_> = units
            .iter()
            .filter(|unit| !selected.contains(unit.id()))
            .cloned()
            .collect();
        if definitions.is_empty() {
            return Ok(None);
        }
        let ids: Vec<_> = definitions.iter().map(|unit| unit.id().clone()).collect();
        // This group describes the actual singleton source-acquire protocol,
        // not an architecture layer schedule or target execution identity.
        let graph = eredu_runtime::ArchitectureExecutionGraph::single("supplementary-source")
            .and_then(|graph| graph.into_owned())
            .map_err(|_| WorkingMemoryError::UnknownBound)?;
        let layout = ExecutionUnitLayout::new(&graph, [ids.len()])
            .map_err(|_| WorkingMemoryError::UnknownBound)?;
        Ok(Some(Self {
            ids,
            definitions,
            layout,
        }))
    }
    pub(super) fn storage_bytes(
        &self,
        plan: &OriginalManagerPlan,
    ) -> Result<usize, WorkingMemoryError> {
        let overflow = || WorkingMemoryError::Overflow;
        let unknown = || WorkingMemoryError::UnknownBound;
        let mut bytes =
            SupplementaryResidencySource::constructor_bytes(&self.ids).ok_or_else(overflow)?;
        let mut add = |n: usize| {
            bytes = bytes.checked_add(n).ok_or_else(overflow)?;
            Ok::<_, WorkingMemoryError>(())
        };
        if plan.foreground.is_some() {
            add(usize::try_from(
                ResidencyManager::original_foreground_operation_source_bytes(
                    &plan.pool,
                    &self.layout,
                    &plan.units,
                    self.ids.len(),
                )?,
            )
            .map_err(|_| overflow())?)?;
        } else {
            add(usize::try_from(
                HostCopyWorkspace::constructor_storage_bytes(&self.definitions, |id, binding| {
                    plan.shape_for(id, binding)
                })
                .map_err(|_| unknown())?,
            )
            .map_err(|_| overflow())?)?;
            add(
                HostCopyIdentity::requested_bytes(&self.layout, self.definitions.iter())
                    .ok_or_else(unknown)?,
            )?;
            add(usize::try_from(
                OriginalResidencySource::constructor_storage_bytes(
                    plan.units.len(),
                    self.ids.len(),
                )
                .ok_or_else(overflow)?,
            )
            .map_err(|_| overflow())?)?;
        }
        for control in [
            size_of::<(&Self, &mut ResidencyManager, &WorkingMemoryPool)>(),
            size_of::<Result<(), ConstructionCause>>(),
            size_of::<Option<HostCopyWorkspace>>(),
            size_of::<Result<HostCopyWorkspace, HostCopyWorkspaceError>>(),
            size_of::<SupplementaryResidencySource>(),
            size_of::<MetalAllocationFacts>(),
            size_of::<ManagerCustody>(),
        ] {
            add(control)?;
        }
        Ok(bytes)
    }
    pub(super) fn initialize(
        &self,
        manager: &mut ResidencyManager,
        pool: &WorkingMemoryPool,
    ) -> Result<(), ConstructionCause> {
        let custody = manager
            .original_source_custody()
            .ok_or(OperationSourceFailure::Layout)?;
        let depth = NonZeroUsize::new(1).expect("singleton source windows");
        let (source, host) = if manager.original_foreground_disk_descriptors().is_some() {
            (
                manager.build_original_foreground_operation_source(
                    pool,
                    &self.ids,
                    &self.layout,
                    depth,
                )?,
                None,
            )
        } else {
            let host = manager
                .build_original_host_workspace(
                    &self.ids,
                    &self.layout,
                    depth.get(),
                    original_allocation_facts()?,
                )
                .map_err(ConstructionCause::Snapshot)?;
            let source = manager.build_original_host_operation_source(
                &self.ids,
                &self.layout,
                depth,
                &host,
            )?;
            (source, Some(host))
        };
        let value = SupplementaryResidencySource::construct(&self.ids, source, host, custody)?;
        manager
            .inner
            .supplementary_source
            .set(value)
            .map_err(|_| OperationSourceFailure::Layout)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
