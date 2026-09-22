//! Actual parameter backing joins for the same cold workspace trace.
use super::*;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_runtime::working_memory::RegisteredWorkspaceStorageRow;

mod completed;
pub(crate) use completed::{CompletedParameterSource, CompletedParameterSources};

#[derive(Debug, Default)]
pub(crate) struct ParameterWorkspaceBackings {
    rows: Vec<(safemlx::AllocationIdentity, WorkspaceExistingStorage)>,
    completed: Vec<(
        safemlx::AllocationIdentity,
        eredu_runtime::working_memory::CompletedWorkspaceSourceAccount,
    )>,
}
impl ParameterWorkspaceBackings {
    pub(crate) fn clear(&mut self) {
        self.rows.clear();
        self.completed.clear();
    }
    pub(crate) fn len(&self) -> usize {
        self.rows.len() - self.completed.len()
    }
    pub(crate) fn roots(
        &self,
    ) -> impl Iterator<Item = RegisteredWorkspaceStorageRow<StorageIdentity>> + '_ {
        self.rows
            .iter()
            .filter(|(identity, _)| !self.completed.iter().any(|(other, _)| identity == other))
            .map(|(identity, root)| registered_storage_row(*identity, root))
    }
    pub(crate) fn classify_completed(
        &mut self,
        sources: &CompletedParameterSources,
        context: &WorkspaceContext,
    ) -> Result<(), Error> {
        self.completed = context.metadata_vec(self.rows.len())?;
        for (identity, root) in &self.rows {
            if let Some(account) = sources.account(*identity, root)? {
                self.completed.push((*identity, account));
            }
        }
        Ok(())
    }
    pub(crate) fn completed_source(
        &self,
        context: &WorkspaceContext,
    ) -> Result<Option<eredu_runtime::working_memory::OriginalCompletedWorkspaceSource>, Error>
    {
        if self.completed.is_empty() {
            return Ok(None);
        }
        let layout = eredu_runtime::working_memory::CompletedWorkspaceSourceLayout::new_accounts(
            self.completed.len(),
        )
        .map_err(|cause| context.metadata_source(cause))?;
        context.charge_metadata(
            layout
                .requested_bytes()
                .checked_add(
                    eredu_core::HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
                        .ok_or(WorkspaceMetadataError::Overflow)?,
                )
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let funding = context
            .metadata_funding()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let host = eredu_core::HostPreparationAuthority::retain(funding);
        let mut entries = context.metadata_vec(self.completed.len())?;
        for (identity, account) in &self.completed {
            let root = self
                .rows
                .iter()
                .find(|(other, _)| identity == other)
                .ok_or(WorkspaceMetadataError::Unqualified)?;
            entries.push((root.1.clone(), account.clone()));
        }
        layout
            .construct_accounts(context, entries, &host)
            .map(Some)
            .map_err(|cause| context.metadata_source(cause))
    }
    pub(crate) fn import(
        &mut self,
        array: &safemlx::Array,
        context: &WorkspaceContext,
    ) -> Result<Option<WorkspaceExistingStorage>, Error> {
        context.charge_metadata(std::mem::size_of::<(
            &mut Self,
            &safemlx::Array,
            Option<safemlx::ArrayAllocationInfo>,
            Result<Option<WorkspaceExistingStorage>, Error>,
        )>())?;
        let Some(info) = array
            .try_allocation_info()
            .map_err(|cause| context.metadata_source(cause))?
        else {
            return Ok(None);
        };
        self.import_observed(info, context).map(Some)
    }
    /// Imports the immutable descriptor snapshot while its original array owner
    /// remains retained. Sharing is decided only by the native generation key.
    pub(crate) fn import_observed(
        &mut self,
        info: safemlx::ArrayAllocationInfo,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceExistingStorage, Error> {
        context.charge_metadata(std::mem::size_of::<(
            &mut Self,
            safemlx::ArrayAllocationInfo,
            &WorkspaceContext,
            Result<WorkspaceExistingStorage, Error>,
        )>())?;
        let placement = crate::backend::managed_memory::cold_allocation_placement(&info)
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        let capacity = u64::try_from(info.bytes()).map_err(|_| WorkspaceMetadataError::Overflow)?;
        let controls = u64::try_from(info.host_control_bytes())
            .map_err(|_| WorkspaceMetadataError::Overflow)?;
        if let Some((_, root)) = self
            .rows
            .iter()
            .find(|(identity, _)| *identity == info.identity())
        {
            if root.capacity_bytes() != Some(capacity)
                || root.host_control_bytes() != Some(controls)
                || root.placement() != Some(placement)
            {
                return Err(WorkspaceMetadataError::Report(WorkspaceReportError::Source).into());
            }
            return Ok(root.clone());
        }
        context.reserve_metadata_vec(&mut self.rows, 1)?;
        let root = WorkspaceExistingStorage::try_new_placed_with_host_controls(
            Some(capacity),
            placement,
            Some(controls),
            context,
        )?;
        self.rows.push((info.identity(), root.clone()));
        Ok(root)
    }
}
