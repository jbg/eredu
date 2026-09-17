//! One shared native constructor through the existing cold source allowance.
use super::{original_prepared_native_input::Account, WorkingMemoryError, WorkingMemoryPool};
use std::{
    fmt,
    mem::{size_of, size_of_val},
};

/// Closed raw accounting custody for a shared native constructor. The producer
/// retains it through its actual object and queued-control retirement. No public
/// clone, amount constructor, request authority or physical publication proof.
#[derive(Debug)]
pub struct SharedNativeInitializationCustody(Account);
impl SharedNativeInitializationCustody {
    /// Read-only original pool identity. This grants no new constructor,
    /// amount, alias or reservation authority.
    pub fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        if self.0.matches_pool(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    /// Compare only the existing reservation's pool with the constructor's
    /// original domain. The caller must still validate reservation/source health.
    pub fn validate_reservation(
        &self,
        reservation: &super::WorkingMemoryReservation,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_pool(&reservation.0.pool)
    }
}

/// A concrete shared constructor. Its owner selects a single winner before
/// invoking this mechanism. Construction occurs once, after the existing Pool
/// comparison, outside Usage. The output/error must preserve all actual prefixes;
/// native aliases must independently retain the supplied custody.
///
/// A successful allocator constructor is not proof for borrowed Device, thread,
/// stream, scheduler or preexisting ordinary owners. Those remain prerequisites.
pub trait SharedNativeInitializer: Sized {
    /// Actual successful shared resource identity/borrower.
    type Output;
    /// Fixed cause and every unretired constructor prefix.
    type Error: std::error::Error;
    /// Complete dynamic producer storage/controls. Fixed static storage already
    /// charged by the domain is excluded. Unknown qualification must refuse.
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError>;
    /// Exactly one constructor attempt. No fallback, polling or second grant.
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error>;
}

/// Shared completed resource plus its original accounting. It cannot yield an
/// owned output or mint a different domain. The native resource retains another
/// raw account alias, so dropping this wrapper cannot refund surviving storage.
#[derive(Debug)]
pub struct InitializedSharedNative<T> {
    output: T,
    account: Account,
}
impl<T> InitializedSharedNative<T> {
    /// Borrow the concrete completed identity. No native call occurs here.
    pub fn output(&self) -> &T {
        &self.output
    }
    /// Preserve the original accounting domain across later borrowers.
    pub fn validate_pool(&self, pool: &WorkingMemoryPool) -> Result<(), WorkingMemoryError> {
        if self.account.matches_pool(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    /// The actual original constructor hold, never a new allowance.
    pub fn original_bytes(&self) -> u64 {
        self.account.held_bytes()
    }
}

/// Rejected plan or actual completed/failed constructor. All value fields retire
/// before the raw account. Source errors remain typed and no native callback is
/// invoked with Usage borrowed.
pub struct SharedNativeInitializationError<P: SharedNativeInitializer> {
    accounting: Option<WorkingMemoryError>,
    construction: Option<P::Error>,
    plan: Option<P>,
    output: Option<P::Output>,
    account: Option<Account>,
}
impl<P: SharedNativeInitializer> SharedNativeInitializationError<P> {
    fn rejected(plan: P, error: WorkingMemoryError) -> Self {
        Self {
            accounting: Some(error),
            construction: None,
            plan: Some(plan),
            output: None,
            account: None,
        }
    }
    /// Exact comparison/settlement refusal.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.accounting.as_ref()
    }
    /// Actual fixed constructor cause with its preserved failed owner/prefix.
    pub fn constructor_failure(&self) -> Option<&P::Error> {
        self.construction.as_ref()
    }
    /// An admission rejection still owns the actual uncalled plan.
    pub fn rejected_plan(&self) -> Option<&P> {
        self.plan.as_ref()
    }
    /// A settlement refusal may retain a genuinely completed native output.
    pub fn completed_output(&self) -> Option<&P::Output> {
        self.output.as_ref()
    }
}
impl<P: SharedNativeInitializer> fmt::Debug for SharedNativeInitializationError<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedNativeInitializationError")
            .field("accounting", &self.accounting)
            .field("construction", &self.construction)
            .finish_non_exhaustive()
    }
}
impl<P: SharedNativeInitializer> fmt::Display for SharedNativeInitializationError<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(cause) = &self.construction {
            cause.fmt(f)
        } else {
            self.accounting
                .as_ref()
                .expect("initialization failure")
                .fmt(f)
        }
    }
}
impl<P: SharedNativeInitializer> std::error::Error for SharedNativeInitializationError<P>
where
    P::Error: 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.construction
            .as_ref()
            .map(|e| e as &(dyn std::error::Error + 'static))
            .or_else(|| {
                self.accounting
                    .as_ref()
                    .map(|e| e as &(dyn std::error::Error + 'static))
            })
    }
}

impl WorkingMemoryPool {
    /// Complete checked producer contribution plus actual cold-account and
    /// constructor/result controls; no allocation or admission occurs here.
    pub fn shared_native_initialization_required_bytes<P: SharedNativeInitializer>(
        plan: &P,
    ) -> Result<u64, WorkingMemoryError> {
        let controls = [
            Account::storage_bytes().ok_or(WorkingMemoryError::UnknownBound)?,
            size_of::<super::loaded_decode_source::Allowance>(),
            size_of::<Account>(),
            size_of::<SharedNativeInitializationCustody>(),
            size_of::<P>(),
            size_of::<Result<P::Output, P::Error>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<InitializedSharedNative<P::Output>>(),
            size_of::<SharedNativeInitializationError<P>>(),
            size_of::<Result<InitializedSharedNative<P::Output>, SharedNativeInitializationError<P>>>(
            ),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(plan.required_storage_bytes()?, usize::checked_add)
            .and_then(|n| n.checked_add(size_of_val(&controls)))
            .ok_or(WorkingMemoryError::Overflow)?;
        u64::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)
    }
    /// One comparison before the account/node/native constructor. This reuses
    /// the source compiler's exact active/reserved ledger and unquoted exclusion.
    /// A sharing coordinator must choose its winner before invoking this method.
    pub fn initialize_shared_native<P: SharedNativeInitializer>(
        &self,
        plan: P,
    ) -> Result<InitializedSharedNative<P::Output>, SharedNativeInitializationError<P>> {
        let bytes = match Self::shared_native_initialization_required_bytes(&plan) {
            Ok(bytes) => bytes,
            Err(error) => return Err(SharedNativeInitializationError::rejected(plan, error)),
        };
        let allowance = match self.admit_source_compiler(bytes) {
            Ok(allowance) => allowance,
            Err(error) => return Err(SharedNativeInitializationError::rejected(plan, error)),
        };
        // The old allowance stays armed through creation of the account. Its
        // existing handoff disarms only after the new shared owner exists.
        let account = allowance.into_prepared_native_account();
        let result = plan.initialize(SharedNativeInitializationCustody(account.share()));
        let accounting = account.finish().err();
        match result {
            Ok(output) if accounting.is_none() => Ok(InitializedSharedNative { output, account }),
            Ok(output) => Err(SharedNativeInitializationError {
                accounting,
                construction: None,
                plan: None,
                output: Some(output),
                account: Some(account),
            }),
            Err(error) => Err(SharedNativeInitializationError {
                accounting,
                construction: Some(error),
                plan: None,
                output: None,
                account: Some(account),
            }),
        }
    }
}

#[cfg(test)]
mod tests;
