//! Named controls of the existing account commit/retire and scope constructors.
use super::*;
use std::mem::size_of;

/// Closed, checked account geometry prepared before physical admission.
pub(in crate::working_memory) struct CopyAccountState {
    pub(super) control_floor: u64,
    pub(super) host_held: u64,
    pub(super) scopes: usize,
    pub(super) native_scopes: usize,
    pub(super) run_open: bool,
}
impl CopyAccountState {
    pub(super) fn prepare(
        controls: u64,
        holds: CopyHostHolds,
        host_requirement: u64,
    ) -> Result<Self, WorkingMemoryError> {
        let native_scopes = usize::from(!matches!(holds, CopyHostHolds::HostOnly(_)));
        let (held, scopes) = holds.total_and_scopes()?;
        let host_held = held
            .checked_add(controls)
            .ok_or(WorkingMemoryError::Overflow)?;
        if host_held > host_requirement {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(Self {
            control_floor: controls,
            host_held,
            scopes,
            native_scopes,
            run_open: native_scopes != 0,
        })
    }
}

pub(in crate::working_memory) fn copy_account_control_bytes(
    sampler: bool,
    tables: usize,
    group: bool,
) -> Result<usize, WorkingMemoryError> {
    let scope_count = tables
        .checked_add(usize::from(sampler))
        .and_then(|n| n.checked_add(1))
        .ok_or(WorkingMemoryError::Overflow)?;
    let scopes = size_of::<WorkingMemoryFundingScope>()
        .checked_mul(scope_count)
        .ok_or(WorkingMemoryError::Overflow)?;
    // The group producer owns the destination scope Vec. This is the actual
    // per-table host_scope return/construction frame, not a second Vec charge.
    let table_frames = size_of::<WorkingMemoryDecoderHostScope>()
        .checked_mul(tables)
        .ok_or(WorkingMemoryError::Overflow)?;
    [
        size_of::<AccountNode>(), // actual one Box allocation
        size_of::<AccountNode>(), // empty value passed to the actual Box constructor
        size_of::<Box<AccountNode>>(),
        size_of::<Option<Box<AccountNode>>>(),
        size_of::<FundingState>(), // prepared committed state
        size_of::<FundingState>(), // empty placeholder state
        size_of::<PreparedAccountCommit<'_>>(),
        size_of::<Result<PreparedAccountCommit<'_>, WorkingMemoryError>>(),
        size_of::<CopyHostHolds>(),
        size_of::<Result<(u64, usize), WorkingMemoryError>>(),
        size_of::<Result<u64, WorkingMemoryError>>(),
        match (sampler, tables, group) {
            (false, 0, false) => size_of::<
                Result<(WorkingMemoryFundingRun, WorkingMemoryFundingScope), WorkingMemoryError>,
            >(),
            (true, 0, false) => size_of::<
                Result<
                    (
                        WorkingMemoryFundingRun,
                        WorkingMemorySamplerScope,
                        WorkingMemoryFundingScope,
                    ),
                    WorkingMemoryError,
                >,
            >(),
            (true, 1, false) => size_of::<
                Result<
                    (
                        WorkingMemoryFundingRun,
                        WorkingMemorySamplerScope,
                        WorkingMemoryDecoderHostScope,
                        WorkingMemoryFundingScope,
                    ),
                    WorkingMemoryError,
                >,
            >(),
            _ => size_of::<
                Result<
                    (
                        WorkingMemoryFundingRun,
                        WorkingMemorySamplerScope,
                        Vec<WorkingMemoryDecoderHostScope>,
                        WorkingMemoryFundingScope,
                    ),
                    WorkingMemoryError,
                >,
            >(),
        },
        size_of::<accounts::RetirementGuard<'_>>(),
        size_of::<std::sync::MutexGuard<'_, Usage>>(),
        size_of::<std::sync::LockResult<std::sync::MutexGuard<'_, Usage>>>(),
        size_of::<WorkingMemoryFundingRun>(),
        size_of::<WorkingMemoryFundingScope>(), // final native return transport
        scopes,
        table_frames,
        if sampler {
            size_of::<WorkingMemorySamplerScope>()
        } else {
            0
        },
        size_of::<QuarantinedStoragePins>(), // one abandoned native scope pin node
        size_of::<Box<QuarantinedStoragePins>>(),
        size_of::<Option<Box<QuarantinedStoragePins>>>(),
        size_of::<Option<RegisteredStoragePin>>(),
        size_of::<Option<eredu_core::HostPreparationAuthority>>(), // terminal custody extraction
        size_of::<accounts::RetiringAccount>(),
        size_of::<Option<accounts::RetiringAccount>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or(WorkingMemoryError::Overflow)
}
