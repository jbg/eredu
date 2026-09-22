//! One shared native constructor through the existing cold source allowance.
use super::{MemoryLedger, WorkingMemoryError, original_prepared_native_input::Account};
use std::{
    fmt,
    mem::{size_of, size_of_val},
};

mod allocations;
mod file_read;
mod memory_read;
mod read_catalog;
mod read_projection;
mod recipe_compilation;
mod recipe_inference;
mod recipe_keys;
mod recipe_mapping;
mod recipe_read;
mod recipe_source;
mod selection_ranges;
pub use recipe_compilation::{AdmittedRecipeConstruction, EncodedRecipeConstructionError};
pub use recipe_read::{CompiledRecipeCustody, EncodedRecipeReadPreparationError};
pub use recipe_source::EncodedRecipeSourceError;

/// Closed raw accounting custody for a shared native constructor. The producer
/// retains it through its actual object and queued-control retirement. No public
/// clone, amount constructor, request authority or physical publication proof.
#[derive(Debug)]
pub struct SharedNativeInitializationCustody(Account, Option<super::WorkingMemoryFundingScope>);
impl SharedNativeInitializationCustody {
    /// Takes the constructor's admitted temporary allocation scope once. Its
    /// allowance belongs to the same transaction as the final shared resource.
    /// The producer must retain it through native completion and certify only
    /// after every surviving allocation has independent registered custody.
    /// A constructor without temporary requirements has no such scope.
    pub fn take_allocation_scope(
        &mut self,
    ) -> Result<super::WorkingMemoryFundingScope, WorkingMemoryError> {
        self.1.take().ok_or(WorkingMemoryError::AlreadyStarted)
    }
    /// Read-only original pool identity. This grants no new constructor,
    /// amount, alias or reservation authority.
    pub fn validate_pool(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
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
    /// Producer storage and controls, excluding fixed storage already charged
    /// by the domain. Native resources and fixed buffers require their actual
    /// qualified contribution; dependency host overhead may use an explicitly
    /// documented configurable input-derived estimate. Neither an estimate nor
    /// this constructor qualifies preexisting source/native owners.
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError>;
    /// Native allocations and their host bookkeeping created during this
    /// constructor, separately from its fixed host storage above. These facts
    /// are prepared and funded by the source owner before admission. Supplying
    /// them selects the ordinary allocation-account lifecycle; finite and
    /// unlimited limits use the same transaction and completion requirements.
    fn temporary_allocation_requirements(&self) -> Option<&eredu_core::DomainMemoryRequirements> {
        None
    }
    /// Exactly one constructor attempt. No fallback, polling or second grant.
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error>;
}

/// Completed resource plus its original accounting. It cannot yield an owned
/// output or mint a different domain. A constructor publishing native aliases
/// retains another raw account share with those aliases; purely owned host
/// output needs only this owner's account until its storage is destroyed.
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
    pub fn validate_pool(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
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

impl<T> std::borrow::Borrow<T> for InitializedSharedNative<T> {
    fn borrow(&self) -> &T {
        &self.output
    }
}

/// Constructor failure and its actual output/prefix custody, independent of
/// the input plan's type or lifetime. All value fields retire before the raw
/// account. A completed output remains borrowed; it cannot escape its account.
pub struct SharedNativeInitializationFailure<T, E> {
    accounting: Option<WorkingMemoryError>,
    construction: Option<E>,
    output: Option<T>,
    account: Option<Account>,
}
impl<T, E> SharedNativeInitializationFailure<T, E> {
    /// Disposes any completed output locally, then maps
    /// the owned constructor error. The original account remains in the returned
    /// failure until its diagnostics retire. The mapper must dispose or retain
    /// every constructor prefix; this operation grants no completion authority.
    pub fn retire_output_and_map_error<F>(
        self,
        map: impl FnOnce(E) -> F,
    ) -> SharedNativeInitializationFailure<(), F> {
        let SharedNativeInitializationFailure {
            accounting,
            construction,
            output,
            account,
        } = self;
        drop(output);
        SharedNativeInitializationFailure {
            accounting,
            construction: construction.map(map),
            output: None,
            account,
        }
    }
    /// Exact comparison/settlement refusal.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.accounting.as_ref()
    }
    /// Actual constructor cause with its preserved failed owner/prefix.
    pub fn constructor_failure(&self) -> Option<&E> {
        self.construction.as_ref()
    }
    /// A settlement refusal may retain a genuinely completed native output.
    pub fn completed_output(&self) -> Option<&T> {
        self.output.as_ref()
    }
}
impl<T, E: fmt::Debug> fmt::Debug for SharedNativeInitializationFailure<T, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedNativeInitializationFailure")
            .field("accounting", &self.accounting)
            .field("construction", &self.construction)
            .finish_non_exhaustive()
    }
}
impl<T, E: fmt::Display> fmt::Display for SharedNativeInitializationFailure<T, E> {
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
impl<T, E: std::error::Error + 'static> std::error::Error
    for SharedNativeInitializationFailure<T, E>
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

/// Rejected plan or actual completed/failed constructor. Source errors remain
/// typed and no native callback is invoked with Usage borrowed.
pub struct SharedNativeInitializationError<P: SharedNativeInitializer> {
    plan: Option<P>,
    failure: SharedNativeInitializationFailure<P::Output, P::Error>,
}
impl<P: SharedNativeInitializer> SharedNativeInitializationError<P> {
    fn rejected(plan: P, error: WorkingMemoryError) -> Self {
        Self {
            plan: Some(plan),
            failure: SharedNativeInitializationFailure {
                accounting: Some(error),
                construction: None,
                output: None,
                account: None,
            },
        }
    }
    /// Exact comparison/settlement refusal.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.failure.accounting_failure()
    }
    /// Actual fixed constructor cause with its preserved failed owner/prefix.
    pub fn constructor_failure(&self) -> Option<&P::Error> {
        self.failure.constructor_failure()
    }
    /// An admission rejection still owns the actual uncalled plan.
    pub fn rejected_plan(&self) -> Option<&P> {
        self.plan.as_ref()
    }
    /// A settlement refusal may retain a genuinely completed native output.
    pub fn completed_output(&self) -> Option<&P::Output> {
        self.failure.completed_output()
    }
    /// Separate an uncalled plan from the failure without copying or releasing
    /// any constructor prefix, completed output or original account. This lets
    /// callers handle a borrowed plan locally while retaining an owned failure
    /// when its output and cause types do not borrow those prerequisites.
    /// Thread-safety remains determined by the actual retained output and cause.
    pub fn into_parts(
        self,
    ) -> (
        Option<P>,
        SharedNativeInitializationFailure<P::Output, P::Error>,
    ) {
        (self.plan, self.failure)
    }
}
impl<P: SharedNativeInitializer> fmt::Debug for SharedNativeInitializationError<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedNativeInitializationError")
            .field("accounting", &self.failure.accounting)
            .field("construction", &self.failure.construction)
            .finish_non_exhaustive()
    }
}
impl<P: SharedNativeInitializer> fmt::Display for SharedNativeInitializationError<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.failure.fmt(f)
    }
}
impl<P: SharedNativeInitializer> std::error::Error for SharedNativeInitializationError<P>
where
    P::Error: 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.failure.source()
    }
}

impl MemoryLedger {
    /// Fixed host contribution plus its cold-account and constructor/result
    /// controls. Temporary domain requirements are separate; the complete
    /// description is `shared_native_initialization_requirements`.
    /// No allocation or admission occurs here.
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
        let (account, allocations) = match plan.temporary_allocation_requirements() {
            Some(requirements) => match self.admit_shared_native_allocations(bytes, requirements) {
                Ok((account, scope)) => (account, Some(scope)),
                Err(error) => return Err(SharedNativeInitializationError::rejected(plan, error)),
            },
            None => {
                let allowance = match self.admit_source_compiler(bytes) {
                    Ok(allowance) => allowance,
                    Err(error) => {
                        return Err(SharedNativeInitializationError::rejected(plan, error));
                    }
                };
                (allowance.into_prepared_native_account(), None)
            }
        };
        let result = plan.initialize(SharedNativeInitializationCustody(
            account.share(),
            allocations,
        ));
        let accounting = account.finish().err();
        match result {
            Ok(output) if accounting.is_none() => Ok(InitializedSharedNative { output, account }),
            Ok(output) => Err(SharedNativeInitializationError {
                plan: None,
                failure: SharedNativeInitializationFailure {
                    accounting,
                    construction: None,
                    output: Some(output),
                    account: Some(account),
                },
            }),
            Err(error) => Err(SharedNativeInitializationError {
                plan: None,
                failure: SharedNativeInitializationFailure {
                    accounting,
                    construction: Some(error),
                    output: None,
                    account: Some(account),
                },
            }),
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod read_sequence_tests;
