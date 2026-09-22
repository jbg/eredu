//! Completed parameter roots retain their actual allocation proof and payer.
use super::*;
use crate::backend::error::Error as NativeError;
use crate::backend::runtime::residency::storage::{
    RetainedAllocationReceipt, RetainedAllocationSource,
};
use crate::backend::submission_recovery::native_role::physical::CompletedNumericalSource;
use crate::MlxTensor;
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::parameter_operations::ParameterReplacementValues;
use eredu_runtime::working_memory::{CompletedWorkspaceSourceAccount, WorkingMemoryError};
use std::{mem::size_of, sync::Arc};

#[derive(Debug)]
struct Row {
    allocation: safemlx::ArrayAllocationInfo,
    source: CompletedNumericalSource,
}
#[derive(Debug)]
struct Owner {
    sources: Vec<CompletedNumericalSource>,
    host_sources: Vec<RetainedAllocationSource>,
    rows: Vec<Row>,
    _funding: HostMetadataFunding,
}
#[derive(Clone, Debug)]
pub(crate) enum CompletedParameterSource {
    Numerical(CompletedNumericalSource),
    Host(RetainedAllocationSource),
}
impl From<CompletedNumericalSource> for CompletedParameterSource {
    fn from(source: CompletedNumericalSource) -> Self {
        Self::Numerical(source)
    }
}
/// Immutable completed source facts. It issues no numerical or text authority.
#[derive(Debug, Default)]
pub(crate) struct CompletedParameterSources(Option<Arc<Owner>>);
impl Clone for CompletedParameterSources {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for CompletedParameterSources {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
fn memory(cause: WorkingMemoryError) -> NativeError {
    NativeError::PrefillControl(cause)
}
impl CompletedParameterSources {
    pub(crate) fn observation_control_bytes() -> Option<usize> {
        type Witness<'a> = (
            safemlx::OriginalBufferWitness<'a>,
            &'a eredu_runtime::working_memory::OriginalNumericalBudgetCustody,
        );
        safemlx::OriginalBufferInspection::inspection_control_bytes()?
            .checked_add(Self::host_receipt_control_bytes()?)?
            .checked_add(size_of::<(
                &Self,
                &safemlx::Array,
                Option<&Arc<Owner>>,
                &Arc<Owner>,
                std::slice::Iter<'_, Row>,
                &Row,
                safemlx::ArrayAllocationInfo,
                Option<Witness<'_>>,
                Result<Option<Witness<'_>>, safemlx::OriginalBufferCause>,
            )>())
    }
    fn host_receipt_control_bytes() -> Option<usize> {
        fn iterator_bytes<T>(_: impl FnOnce(&'static CompletedParameterSources) -> T) -> usize {
            size_of::<T>()
        }
        size_of::<(
            &Self,
            safemlx::AllocationInfo,
            Option<&RetainedAllocationSource>,
            std::slice::Iter<'_, RetainedAllocationSource>,
            Option<RetainedAllocationReceipt<'_>>,
            Result<Option<RetainedAllocationReceipt<'_>>, safemlx::OriginalBufferCause>,
            bool,
        )>()
        .checked_add(iterator_bytes(Self::host_sources))?
        .checked_add(RetainedAllocationReceipt::control_bytes()?)
    }
    fn host_sources(&self) -> impl Iterator<Item = &RetainedAllocationSource> {
        self.0.iter().flat_map(|owner| &owner.host_sources)
    }
    pub(crate) fn sources(&self) -> impl Iterator<Item = &CompletedNumericalSource> {
        self.0.iter().flat_map(|owner| &owner.sources)
    }
    pub(crate) fn prepared_sources(&self) -> impl Iterator<Item = CompletedParameterSource> + '_ {
        self.sources()
            .cloned()
            .map(CompletedParameterSource::Numerical)
            .chain(
                self.0
                    .iter()
                    .flat_map(|owner| &owner.host_sources)
                    .cloned()
                    .map(CompletedParameterSource::Host),
            )
    }
    pub(crate) fn from_prepared(
        prepared: Vec<CompletedParameterSource>,
        context: &WorkspaceContext,
        funding: HostMetadataFunding,
    ) -> Result<Self, NativeError> {
        context
            .charge_metadata(size_of::<(
                Vec<CompletedParameterSource>,
                std::vec::IntoIter<CompletedParameterSource>,
                CompletedParameterSource,
                Vec<CompletedNumericalSource>,
                Vec<RetainedAllocationSource>,
                std::slice::Iter<'_, RetainedAllocationSource>,
                Option<&RetainedAllocationSource>,
                &RetainedAllocationSource,
                bool,
                Result<Self, NativeError>,
            )>())
            .map_err(|cause| NativeError::Neural(cause.into()))?;
        let mut sources = context
            .metadata_vec(prepared.len())
            .map_err(NativeError::Neural)?;
        let mut host_sources = context
            .metadata_vec(prepared.len())
            .map_err(NativeError::Neural)?;
        for source in prepared {
            match source {
                CompletedParameterSource::Numerical(source) => sources.push(source),
                CompletedParameterSource::Host(source) => {
                    if let Some(previous) =
                        host_sources.iter().find(|row: &&RetainedAllocationSource| {
                            row.allocation().identity() == source.allocation().identity()
                        })
                    {
                        if !previous.same_source(&source) {
                            return Err(memory(WorkingMemoryError::IdentityMismatch));
                        }
                    } else {
                        host_sources.push(source);
                    }
                }
            }
        }
        Self::new(sources, host_sources, Vec::new(), context, funding)
    }
    fn new(
        sources: Vec<CompletedNumericalSource>,
        host_sources: Vec<RetainedAllocationSource>,
        rows: Vec<Row>,
        context: &WorkspaceContext,
        funding: HostMetadataFunding,
    ) -> Result<Self, NativeError> {
        if !context
            .metadata_funding()
            .is_some_and(|actual| actual.same_account(&funding))
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        context
            .charge_metadata(size_of::<(Self, Owner, Result<Self, NativeError>)>())
            .map_err(|cause| NativeError::Neural(cause.into()))?;
        Ok(Self(Some(
            context
                .metadata_arc(Owner {
                    sources,
                    host_sources,
                    rows,
                    _funding: funding,
                })
                .map_err(|cause| NativeError::Neural(cause.into()))?,
        )))
    }
    pub(crate) fn host_receipt(
        &self,
        allocation: safemlx::AllocationInfo,
    ) -> Result<Option<RetainedAllocationReceipt<'_>>, safemlx::OriginalBufferCause> {
        let Some(source) = self
            .host_sources()
            .find(|source| source.allocation().identity() == allocation.identity())
        else {
            return Ok(None);
        };
        if source.allocation() != allocation {
            return Err(safemlx::OriginalBufferCause::UncertifiedBacking);
        }
        Ok(Some(source.borrow()))
    }
    /// Borrows a settled witness only for a retained, selected physical root.
    /// Sharing an arena with that root does not authorize other arena contents.
    pub(crate) fn observe<'a>(
        &'a self,
        array: &'a safemlx::Array,
    ) -> Result<
        Option<(
            safemlx::OriginalBufferWitness<'a>,
            &'a eredu_runtime::working_memory::OriginalNumericalBudgetCustody,
        )>,
        safemlx::OriginalBufferCause,
    > {
        let Some(owner) = &self.0 else {
            return Ok(None);
        };
        for row in &owner.rows {
            let witness = match row.source.budget().inspect_array(array) {
                Ok(Some(witness)) => witness,
                Ok(None) | Err(safemlx::OriginalBufferCause::ForeignDomain) => continue,
                Err(cause) => return Err(cause),
            };
            let actual = witness.allocation();
            if actual.identity() != row.allocation.identity() {
                continue;
            }
            if actual != row.allocation {
                return Err(safemlx::OriginalBufferCause::UncertifiedBacking);
            }
            return Ok(Some((witness, row.source.account())));
        }
        Ok(None)
    }
    /// Reauthenticates each actual root after completion and retains only arenas
    /// used by these immutable values. Registered ordinary roots remain separate.
    pub(crate) fn select(
        &self,
        values: &ParameterReplacementValues<MlxTensor>,
        context: &WorkspaceContext,
        funding: HostMetadataFunding,
    ) -> Result<Self, NativeError> {
        let mut sources: Vec<CompletedNumericalSource> = context
            .metadata_vec(values.len())
            .map_err(NativeError::Neural)?;
        let mut rows: Vec<Row> = context
            .metadata_vec(values.len())
            .map_err(NativeError::Neural)?;
        let mut host_sources: Vec<RetainedAllocationSource> = context
            .metadata_vec(values.len())
            .map_err(NativeError::Neural)?;
        context
            .charge_metadata(
                size_of::<(
                    Self,
                    Option<safemlx::ArrayAllocationInfo>,
                    Result<Self, NativeError>,
                )>()
                .checked_add(
                    Self::host_receipt_control_bytes()
                        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
                )
                .and_then(|n| {
                    n.checked_add(
                        safemlx::HostTransferArrayAliasWitness::inspection_control_bytes()?,
                    )
                })
                .and_then(|n| {
                    n.checked_add(size_of::<(
                        Vec<RetainedAllocationSource>,
                        Option<safemlx::HostTransferArrayAliasWitness<'_>>,
                        &RetainedAllocationSource,
                        Option<&RetainedAllocationSource>,
                    )>())
                })
                .and_then(|n| {
                    n.checked_add(safemlx::OriginalBufferInspection::inspection_control_bytes()?)
                })
                .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
            )
            .map_err(|cause| NativeError::Neural(cause.into()))?;
        for value in values.values() {
            let array = value.as_array();
            let mut selected: Option<Row> = None;
            for source in self.0.iter().flat_map(|owner| &owner.sources) {
                let witness = match source.budget().inspect_array(array) {
                    Ok(witness) => witness,
                    Err(safemlx::OriginalBufferCause::ForeignDomain) => continue,
                    Err(cause) => return Err(NativeError::Neural(context.metadata_source(cause))),
                };
                let Some(witness) = witness else {
                    continue;
                };
                let allocation = witness.allocation();
                if selected.as_ref().is_some_and(|previous| {
                    previous.allocation != allocation
                        || !previous.source.account().same_account(source.account())
                }) {
                    return Err(memory(WorkingMemoryError::IdentityMismatch));
                }
                selected = Some(Row {
                    allocation,
                    source: source.clone(),
                });
            }
            let Some(row) = selected else {
                if array
                    .inspect_original_buffer_alias()
                    .map_err(|cause| NativeError::Neural(context.metadata_source(cause)))?
                    .is_some()
                {
                    return Err(memory(WorkingMemoryError::IdentityMismatch));
                }
                if let Some(witness) = array
                    .inspect_host_transfer_alias()
                    .map_err(|cause| NativeError::Neural(context.metadata_source(cause)))?
                {
                    if let Some(source) =
                        self.0
                            .iter()
                            .flat_map(|owner| &owner.host_sources)
                            .find(|source| {
                                source.allocation().identity() == witness.allocation().identity()
                            })
                    {
                        if source.allocation() != witness.allocation() {
                            return Err(memory(WorkingMemoryError::IdentityMismatch));
                        }
                        if let Some(previous) = host_sources.iter().find(|row| {
                            row.allocation().identity() == source.allocation().identity()
                        }) {
                            if !previous.same_source(source) {
                                return Err(memory(WorkingMemoryError::IdentityMismatch));
                            }
                        } else {
                            host_sources.push(source.clone());
                        }
                    }
                }
                continue;
            };
            if let Some(previous) = rows
                .iter()
                .find(|old| old.allocation.identity() == row.allocation.identity())
            {
                if previous.allocation != row.allocation
                    || !previous.source.account().same_account(row.source.account())
                {
                    return Err(memory(WorkingMemoryError::IdentityMismatch));
                }
                continue;
            }
            if !sources.iter().any(|source| {
                source.budget().same_budget(row.source.budget())
                    && source.account().same_account(row.source.account())
            }) {
                sources.push(row.source.clone());
            }
            rows.push(row);
        }
        Self::new(sources, host_sources, rows, context, funding)
    }
    pub(super) fn account(
        &self,
        identity: safemlx::AllocationIdentity,
        root: &WorkspaceExistingStorage,
    ) -> Result<Option<CompletedWorkspaceSourceAccount>, Error> {
        let Some(row) = self
            .0
            .iter()
            .flat_map(|owner| &owner.rows)
            .find(|row| row.allocation.identity() == identity)
        else {
            return Ok(None);
        };
        let placement = crate::backend::managed_memory::cold_allocation_placement(&row.allocation)
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        if root.capacity_bytes() != u64::try_from(row.allocation.bytes()).ok()
            || root.host_control_bytes() != u64::try_from(row.allocation.host_control_bytes()).ok()
            || root.placement() != Some(placement)
        {
            return Err(WorkspaceMetadataError::Report(WorkspaceReportError::Source).into());
        }
        Ok(Some(CompletedWorkspaceSourceAccount::Standalone(
            row.source.account().clone(),
        )))
    }
}
