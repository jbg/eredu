//! The constructor's temporary native allowance and actual one-shot observer.
use super::*;
use eredu_core::DomainMemoryRequirements;
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::working_memory::{WorkingMemoryFundingScope, WorkspaceReportMetadata};

pub(in crate::backend::runtime::residency::manager::construction) struct ConstructionPlan {
    pub(in crate::backend::runtime::residency::manager::construction) requirements:
        DomainMemoryRequirements,
    metadata: usize,
    _funding: HostMetadataFunding,
}
impl ConstructionPlan {
    pub(in crate::backend::runtime::residency::manager::construction) fn prepare(
        plan: &OriginalManagerPlan,
    ) -> Result<Option<Self>, PreparationFailure> {
        let mut prepared: Option<Self> = None;
        for row in &plan.reads {
            let read_source_plan::ReadValue::Materialized(read) = &row.read else {
                continue;
            };
            let runtime =
                crate::backend::managed_memory::input_allocator::borrow_admitted(&plan.pool)?;
            let allocation =
                original_allocation_facts().map_err(|_| WorkingMemoryError::UnknownBound)?;
            let quote = quotation::quote(
                read,
                &plan.pool,
                &runtime,
                allocation,
                plan.streams.source_stream(),
                read.funding(),
            )?;
            match &mut prepared {
                None => {
                    prepared = Some(Self {
                        requirements: quote.requirements,
                        metadata: quote.metadata,
                        _funding: read.funding().clone(),
                    })
                }
                Some(previous) => {
                    let reports = WorkspaceReportMetadata::with_funding(&previous._funding);
                    previous.requirements = reports
                        .combine_domain_requirements(
                            &previous.requirements,
                            &quote.requirements,
                            true,
                        )
                        .map_err(|cause| WeightRecipeError::Workspace(reports.error(cause)))?;
                    previous.metadata = previous
                        .metadata
                        .checked_add(quote.metadata)
                        .ok_or(WorkingMemoryError::Overflow)?;
                }
            }
        }
        if let Some(prepared) = &mut prepared {
            prepared.metadata =
                quotation::with_owner(&mut prepared.requirements, &plan.pool, prepared.metadata)?;
        }
        Ok(prepared)
    }
}

pub(in crate::backend::runtime::residency::manager::construction) struct Session {
    pub(in crate::backend::runtime::residency::manager::construction) observer:
        ScopedPhysicalBackingObserver,
    pub(in crate::backend::runtime::residency::manager::construction) host:
        HostPreparationAuthority,
    scope: WorkingMemoryFundingScope,
}
impl Session {
    pub(in crate::backend::runtime::residency::manager::construction) fn new(
        plan: &ConstructionPlan,
        mut scope: WorkingMemoryFundingScope,
    ) -> Result<Self, Failure> {
        let metadata = scope.prepare_storage_metadata().map_err(Failure::Funding)?;
        let host = metadata
            .prepare_host_owner(plan.metadata)
            .map_err(Failure::Funding)?;
        let observer = crate::backend::managed_memory::prepare_temporary_scoped_observer(
            &mut scope,
            host.clone(),
        )
        .map_err(Failure::Backend)?;
        Ok(Self {
            observer,
            host,
            scope,
        })
    }
    pub(in crate::backend::runtime::residency::manager::construction) fn finish(
        self,
    ) -> Result<(), Failure> {
        let Self {
            observer,
            host,
            scope,
        } = self;
        drop((observer, host));
        scope.certify().map_err(Failure::Accounting)
    }
}
