//! One constructor admission for fixed host owners and temporary native work.
use super::*;
use crate::working_memory::{
    StorageMetadataFunding, WorkingMemoryFundingRun, WorkingMemoryFundingScope,
    funding::{
        CopyHostHolds, PreparedCopyAccount, copy_account_control_bytes, copy_domain_controls,
    },
    reservation_metadata::funding_error,
    transaction_buffers::RequirementProjection,
};
use eredu_core::{DomainMemoryRequirements, MemoryHeadroomDeclarations, MemoryLimitDeclarations};

impl MemoryLedger {
    fn shared_allocation_controls(
        &self,
        fixed_bytes: u64,
        native: &DomainMemoryRequirements,
    ) -> Result<(u64, u64), WorkingMemoryError> {
        native.validate(self.topology())?;
        let fixed = usize::try_from(fixed_bytes).map_err(|_| WorkingMemoryError::Overflow)?;
        let host =
            u64::try_from(StorageMetadataFunding::host_owner_bytes(fixed).map_err(funding_error)?)
                .map_err(|_| WorkingMemoryError::Overflow)?
                .checked_add(Self::storage_metadata_control_bytes()?)
                .ok_or(WorkingMemoryError::Overflow)?;
        let projection = RequirementProjection {
            parts: &[native],
            headroom: &MemoryHeadroomDeclarations::none(),
            host_bytes: host,
        };
        let frames = [
            size_of::<DomainMemoryRequirements>(),
            size_of::<WorkingMemoryFundingRun>(),
            size_of::<WorkingMemoryFundingScope>(),
            size_of::<StorageMetadataFunding>(),
            size_of::<Result<StorageMetadataFunding, eredu_core::HostMetadataFundingError>>(),
            size_of::<eredu_core::HostPreparationAuthority>(),
            size_of::<
                Result<eredu_core::HostPreparationAuthority, eredu_core::HostMetadataFundingError>,
            >(),
            size_of::<(Account, WorkingMemoryFundingScope)>(),
            size_of::<Result<(Account, WorkingMemoryFundingScope), WorkingMemoryError>>(),
            size_of::<Result<(u64, u64), WorkingMemoryError>>(),
            size_of::<(&Self, u64, &DomainMemoryRequirements)>(),
        ];
        let fixed_controls = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .and_then(|n| n.checked_add(copy_account_control_bytes(false, 0, false).ok()?))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let controls = fixed_controls
            .checked_add(copy_domain_controls(self, &projection)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok((host, controls))
    }

    /// Complete constructor requirements, including the ordinary account and
    /// its domain bookkeeping. This is descriptive and grants no allocation
    /// or execution permission. The source owner funds this report's storage.
    pub fn shared_native_initialization_requirements<P: SharedNativeInitializer>(
        &self,
        plan: &P,
    ) -> Result<DomainMemoryRequirements, WorkingMemoryError> {
        let fixed = Self::shared_native_initialization_required_bytes(plan)?;
        match plan.temporary_allocation_requirements() {
            Some(native) => {
                let (host, controls) = self.shared_allocation_controls(fixed, native)?;
                RequirementProjection {
                    parts: &[native],
                    headroom: &MemoryHeadroomDeclarations::none(),
                    host_bytes: host
                        .checked_add(controls)
                        .ok_or(WorkingMemoryError::Overflow)?,
                }
                .materialize(self.topology())
            }
            None => RequirementProjection {
                parts: &[],
                headroom: &MemoryHeadroomDeclarations::none(),
                host_bytes: fixed,
            }
            .materialize(self.topology()),
        }
    }

    pub(super) fn admit_shared_native_allocations(
        &self,
        fixed_bytes: u64,
        native: &DomainMemoryRequirements,
    ) -> Result<(Account, WorkingMemoryFundingScope), WorkingMemoryError> {
        let (host, controls) = self.shared_allocation_controls(fixed_bytes, native)?;
        let execution = &self.0.construction_execution;
        let prepared = PreparedCopyAccount::accept(
            self,
            execution,
            RequirementProjection {
                parts: &[native],
                headroom: &MemoryHeadroomDeclarations::none(),
                host_bytes: host,
            },
            &MemoryLimitDeclarations::default(),
            controls,
            CopyHostHolds::None,
            |usage| {
                if usage.unquoted_owners != 0 {
                    Err(WorkingMemoryError::UnknownBound)
                } else {
                    Ok(())
                }
            },
        )?;
        let (requirements, run, scope) = prepared.workspace(execution, None)?;
        let fixed = (|| {
            let metadata = StorageMetadataFunding::from_shared_constructor_account(self, scope.id)
                .map_err(funding_error)?;
            metadata
                .prepare_host_owner(
                    usize::try_from(fixed_bytes).map_err(|_| WorkingMemoryError::Overflow)?,
                )
                .map_err(funding_error)
        })();
        let fixed = match fixed {
            Ok(fixed) => fixed,
            Err(cause) => {
                // Only host constructors have run. No native observer or
                // allocation handle has been handed to the producer yet.
                scope.certify()?;
                return Err(cause);
            }
        };
        let account = Account::from_funded_host(self.clone(), fixed_bytes, fixed);
        drop(requirements);
        drop(run);
        Ok((account, scope))
    }
}

#[cfg(test)]
mod tests;
