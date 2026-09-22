//! Fresh immutable intervention declarations under the shared original source compiler.
use super::original_declaration_source::Account;
use super::{MemoryLedger, WorkingMemoryError, loaded_decode_source::Allowance};
use eredu_core::{
    HostPreparationAuthority,
    capture::CapturePlanCopyError,
    intervention::{PreparedInterventionPlanCopy, SharedInterventionPlan},
};
use std::mem::{size_of, size_of_val};

/// The freshly copied immutable declaration and its actual original C account.
/// Every shared-plan alias retains that account internally; neither the caller's
/// source buffers nor a caller-supplied host token can construct this proof.
#[derive(Debug, Clone)]
pub struct OriginalInterventionSource {
    source: SharedInterventionPlan,
    account: Account,
}
impl OriginalInterventionSource {
    /// Borrow the exact freshly constructed shared plan, with its real custody.
    pub fn plan(&self) -> &SharedInterventionPlan {
        &self.source
    }
    /// Fixed borrowed source/account validation controls; no allocation is made.
    pub fn validation_control_bytes() -> Option<usize> {
        [
            size_of::<&Self>(),
            size_of::<&MemoryLedger>(),
            size_of::<std::sync::MutexGuard<'_, Allowance>>(),
            size_of::<WorkingMemoryError>(),
            size_of::<Result<(), WorkingMemoryError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    /// Check the exact source domain and its private owner mutex. Request/pool
    /// admission health is validated separately by the consuming request. This
    /// grants no publication, quote or execution.
    pub fn validate_pool(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
        self.account.validate(pool)
    }
    /// Identity of the physical immutable source, never a semantic digest.
    pub fn same_source(&self, other: &Self) -> bool {
        self.source.same_storage(&other.source)
    }
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("{0}")]
    Memory(#[from] WorkingMemoryError),
    #[error("{0}")]
    Copy(#[from] CapturePlanCopyError),
}
/// A typed refusal or failed copy retaining its full original compiler charge.
/// No partial source or account can be extracted or reused to retry a copy.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalInterventionSourceError {
    #[source]
    cause: Cause,
    settlement: Option<WorkingMemoryError>,
    completed: Option<SharedInterventionPlan>,
    account: Option<Account>,
}
impl OriginalInterventionSourceError {
    fn refused(cause: impl Into<Cause>) -> Self {
        Self {
            cause: cause.into(),
            settlement: None,
            completed: None,
            account: None,
        }
    }
}
impl MemoryLedger {
    /// Exact source destination and closed-account controls. The borrowed plan
    /// determines every copied String/Vec; no caller amount is accepted.
    pub fn intervention_source_required_bytes(
        plan: &PreparedInterventionPlanCopy<'_>,
    ) -> Result<u64, WorkingMemoryError> {
        let controls = [
            Account::control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            OriginalInterventionSource::validation_control_bytes()
                .ok_or(WorkingMemoryError::Overflow)?,
            size_of::<OriginalInterventionSource>(),
            size_of::<OriginalInterventionSourceError>(),
            size_of::<Result<OriginalInterventionSource, OriginalInterventionSourceError>>(),
            size_of::<Result<SharedInterventionPlan, CapturePlanCopyError>>(),
        ];
        controls
            .into_iter()
            .try_fold(
                plan.required_bytes()
                    .checked_add(size_of_val(&controls))
                    .ok_or(WorkingMemoryError::Overflow)?,
                usize::checked_add,
            )
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Reserve before the first destination/account allocation, then consume
    /// the same closed copy worker. Existing active request ceilings apply.
    /// This source compiler creates no native value and no numerical occurrence.
    pub fn compile_intervention_source(
        &self,
        plan: PreparedInterventionPlanCopy<'_>,
    ) -> Result<OriginalInterventionSource, OriginalInterventionSourceError> {
        let bytes = Self::intervention_source_required_bytes(&plan)
            .map_err(OriginalInterventionSourceError::refused)?;
        let account =
            Account::admit(self, bytes).map_err(OriginalInterventionSourceError::refused)?;
        let host = HostPreparationAuthority::retain(account.clone());
        match plan.copy(host) {
            Err(cause) => {
                let settlement = account.finish().err();
                Err(OriginalInterventionSourceError {
                    cause: cause.into(),
                    settlement,
                    completed: None,
                    account: Some(account),
                })
            }
            Ok(source) => match account.finish() {
                Ok(()) => Ok(OriginalInterventionSource { source, account }),
                Err(cause) => Err(OriginalInterventionSourceError {
                    cause: cause.into(),
                    settlement: None,
                    completed: Some(source),
                    account: Some(account),
                }),
            },
        }
    }
}
