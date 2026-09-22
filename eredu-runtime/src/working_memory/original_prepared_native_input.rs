//! One original materialization of the immutable host source's native leaves.
//! Concrete native recipes implement the compiler contract. This is not request,
//! encoder, prepared-input control or managed generation authority.
use super::{MemoryLedger, OriginalPreparedHostInput, WorkingMemoryError};
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

#[derive(Debug)]
struct AccountInner {
    pool: MemoryLedger,
    bytes: u64,
    requirements: Option<eredu_core::DomainMemoryRequirements>,
    armed: AtomicBool,
    compiling: AtomicBool,
    // Converted host storage spends an ordinary account exactly once. This
    // custody retires after the shared shell, without a second scalar refund.
    _funded_host: Option<eredu_core::HostPreparationAuthority>,
}
impl Drop for AccountInner {
    fn drop(&mut self) {
        if !*self.armed.get_mut() {
            return;
        }
        // All strong wrappers use into_inner, so this allocation is already
        // gone. No callback/destructor runs with Usage borrowed.
        if let Ok(mut usage) = self.pool.0.usage.lock() {
            if *self.compiling.get_mut() {
                usage.reservations -= 1;
            }
            if let Some(requirements) = &self.requirements {
                for (slot, (_, charge)) in requirements.iter().enumerate() {
                    let domain = &mut usage.domains[slot];
                    domain.reserved -= charge.total().expect("validated compiler charge");
                    domain.placement_allowances -= charge.placement_allowance_bytes;
                    domain.estimates -= charge.estimated_overhead_bytes;
                    domain.headroom -= charge.headroom_bytes;
                }
            } else {
                usage.reserved -= self.bytes;
            }
        }
    }
}
#[derive(Debug)]
pub(super) struct Account(Option<Arc<AccountInner>>);
impl Account {
    pub(super) fn new_unarmed(pool: MemoryLedger, bytes: u64) -> Self {
        Self(Some(Arc::new(AccountInner {
            pool,
            bytes,
            requirements: None,
            armed: AtomicBool::new(false),
            compiling: AtomicBool::new(true),
            _funded_host: None,
        })))
    }
    fn new_domains(pool: MemoryLedger, requirements: eredu_core::DomainMemoryRequirements) -> Self {
        Self(Some(Arc::new(AccountInner {
            pool,
            bytes: 0,
            requirements: Some(requirements),
            armed: AtomicBool::new(false),
            compiling: AtomicBool::new(true),
            _funded_host: None,
        })))
    }
    pub(super) fn from_funded_host(
        pool: MemoryLedger,
        bytes: u64,
        host: eredu_core::HostPreparationAuthority,
    ) -> Self {
        Self(Some(Arc::new(AccountInner {
            pool,
            bytes,
            requirements: None,
            armed: AtomicBool::new(false),
            compiling: AtomicBool::new(false),
            _funded_host: Some(host),
        })))
    }
    pub(super) fn share(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live cold account"),
        )))
    }
    pub(super) fn storage_bytes() -> Option<usize> {
        super::qualified_storage::shared_bytes::<AccountInner>()
            .ok()
            .and_then(|n| usize::try_from(n).ok())
    }
    pub(super) fn matches_pool(&self, pool: &MemoryLedger) -> bool {
        self.inner().pool.same_ledger(pool)
    }
    pub(super) fn held_bytes(&self) -> u64 {
        self.inner().bytes
    }
    pub(super) fn activate(&self) {
        self.inner().armed.store(true, Ordering::Relaxed);
    }
    fn inner(&self) -> &AccountInner {
        self.0.as_deref().expect("live input account")
    }
    fn native_owner(&self) -> OriginalPreparedInputCustody {
        OriginalPreparedInputCustody(Account(Some(Arc::clone(
            self.0.as_ref().expect("live account"),
        ))))
    }
    pub(super) fn finish(&self) -> Result<(), WorkingMemoryError> {
        let inner = self.inner();
        let mut usage = inner
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if inner.compiling.swap(false, Ordering::Relaxed) {
            usage.reservations -= 1;
        }
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}

/// Closed custody for the one concrete native source resource. The compiler
/// moves it into its allocation retirement callback; it cannot clone, refill,
/// extract a guard, attach arbitrary bytes or expose Weak ownership.
///
/// ```compile_fail
/// use eredu_runtime::working_memory::OriginalPreparedInputCustody;
/// fn duplicate(owner: OriginalPreparedInputCustody) { let _ = owner.clone(); }
/// ```
#[derive(Debug)]
pub struct OriginalPreparedInputCustody(Account);

/// Internal host half of the same accepted B account. Only the concrete
/// source-storage builder can split the incoming custody; no public token Clone
/// or arbitrary payload attachment exists.
#[derive(Debug)]
pub(crate) struct PreparedInputHostCustody(Account);
impl OriginalPreparedInputCustody {
    pub(crate) fn into_model_input_parts(self) -> (Self, PreparedInputHostCustody) {
        let native = self.0.native_owner();
        (native, PreparedInputHostCustody(self.0))
    }
}
impl Clone for PreparedInputHostCustody {
    fn clone(&self) -> Self {
        self.share()
    }
}
impl PreparedInputHostCustody {
    pub(crate) fn same_account(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.0.as_ref().expect("live prepared input account"),
            other.0.0.as_ref().expect("live prepared input account"),
        )
    }
    pub(crate) fn share(&self) -> Self {
        let OriginalPreparedInputCustody(account) = self.0.native_owner();
        Self(account)
    }
    pub(crate) fn pool(&self) -> &MemoryLedger {
        &self.0.inner().pool
    }
    pub(crate) fn bytes(&self) -> Option<u64> {
        self.requirements()
            .iter()
            .try_fold(0u64, |sum, (_, charge)| {
                sum.checked_add(charge.total().ok()?)
            })
    }
    pub(crate) fn requirements(&self) -> &eredu_core::DomainMemoryRequirements {
        self.0
            .inner()
            .requirements
            .as_ref()
            .expect("prepared native compiler domain account")
    }
}

/// Native mechanism contract for one source-derived leaf materialization.
/// Implementations must report every reached native/backing/host allocation and
/// control, or require its genuine preinitialization. `compile` receives custody
/// only after comparison, and returns every allocated prefix on failure. It must
/// retain custody in native aliases independently of its returned output.
pub trait PreparedNativeInputCompiler: Sized {
    /// Closed completed or partial native source representation.
    type Output;
    /// Fixed constructor/refusal error. Formatting is outside compilation.
    type Error: std::error::Error;
    /// The genuine immutable source, not its content digest.
    fn source(&self) -> &OriginalPreparedHostInput;
    /// Checked concrete allocation and control contribution, without allocation.
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError>;
    /// Actual backend-supplied placement of every native allocation and host control.
    /// This descriptor is descriptive; compilation starts only after admission.
    fn required_storage_requirements(
        &self,
        topology: &eredu_core::MemoryTopology,
    ) -> Result<eredu_core::DomainMemoryRequirements, WorkingMemoryError>;
    /// Performs the one actual materialization, preserving a failed prefix.
    fn compile(
        self,
        custody: OriginalPreparedInputCustody,
    ) -> Result<Self::Output, (Self::Output, Self::Error)>;
}

struct Payload<T> {
    storage: T,
    source: OriginalPreparedHostInput,
    account: Account,
}
fn take_payload<T>(payload: Box<Payload<T>>) -> Payload<T> {
    // Returning moves the value out and deallocates the Box in this frame,
    // before the caller can drop storage/accounting contained in that value.
    *payload
}
/// Closed native source. Only shared borrows escape; owning aliases are the
/// concrete native mechanism's responsibility. Its Box retires before storage,
/// original host source and the final Rust account reference.
///
/// ```compile_fail
/// use eredu_runtime::working_memory::OriginalPreparedInputMaterialization;
/// fn raw<T>(source: OriginalPreparedInputMaterialization<T>) { source.into_inner(); }
/// ```
pub struct OriginalPreparedInputMaterialization<T>(Option<Box<Payload<T>>>);
impl<T> OriginalPreparedInputMaterialization<T> {
    fn payload(&self) -> &Payload<T> {
        self.0.as_deref().expect("live native input")
    }
    /// Borrows completed immutable mechanism storage without transferring it.
    pub fn storage(&self) -> &T {
        &self.payload().storage
    }
    /// The exact original source used by the compiler.
    pub fn source(&self) -> &OriginalPreparedHostInput {
        &self.payload().source
    }
    /// Total original B1 allowance, independent of the host source's allowance.
    pub fn original_bytes(&self) -> Option<u64> {
        self.requirements()
            .iter()
            .try_fold(0u64, |sum, (_, charge)| {
                sum.checked_add(charge.total().ok()?)
            })
    }
    /// Complete physical-domain allowance retained by all native aliases.
    pub fn requirements(&self) -> &eredu_core::DomainMemoryRequirements {
        self.payload()
            .account
            .inner()
            .requirements
            .as_ref()
            .expect("native compiler account")
    }
    /// Converts a rejected completed result into the same closed retirement
    /// failure used by compiler refusals. Concrete storage retires before the
    /// immutable source and accounting-only B hold; no alias or credit escapes.
    pub fn retire_rejected<E>(self, cause: E) -> RetiredPreparedInputMaterializationError<E> {
        let source = self.source().clone();
        OriginalPreparedInputMaterializationError {
            accounting: None,
            compilation: Some(cause),
            completed: Some(self),
            source,
        }
        .retire_storage()
    }

    /// Validates the actual domain without changing any accounting state.
    pub fn validate_pool(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
        if self.payload().account.inner().pool.same_ledger(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}
impl<T> Drop for OriginalPreparedInputMaterialization<T> {
    fn drop(&mut self) {
        if let Some(payload) = self.0.take() {
            drop(take_payload(payload));
        }
    }
}
impl<T> fmt::Debug for OriginalPreparedInputMaterialization<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalPreparedInputMaterialization")
            .field("original_bytes", &self.original_bytes())
            .finish_non_exhaustive()
    }
}

/// Admission rejection or owning terminal failure. The actual source and every
/// constructed leaf/control remain retained. There is no retry or prefix take.
pub struct OriginalPreparedInputMaterializationError<T, E> {
    accounting: Option<WorkingMemoryError>,
    compilation: Option<E>,
    completed: Option<OriginalPreparedInputMaterialization<T>>,
    source: OriginalPreparedHostInput,
}
impl<T, E> OriginalPreparedInputMaterializationError<T, E> {
    fn rejected(source: OriginalPreparedHostInput, error: WorkingMemoryError) -> Self {
        Self {
            accounting: Some(error),
            compilation: None,
            completed: None,
            source,
        }
    }
    /// Fixed moves for retiring this native prefix without allocating another
    /// shell. An enclosing funded adapter pays these before materialization.
    pub fn retirement_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Option<OriginalPreparedInputMaterialization<T>>>(),
            size_of::<RetiredPreparedInputMaterializationError<E>>(),
            size_of::<Option<Account>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Retires concrete storage through its existing Drop path and preserves
    /// the exact cause, immutable I and accounting-only B owner. Native aliases
    /// still retain their own B custody; this creates no completion evidence,
    /// retry, source credit or allocation. No concrete prefix can escape.
    pub fn retire_storage(self) -> RetiredPreparedInputMaterializationError<E> {
        let Self {
            accounting,
            compilation,
            completed,
            source,
        } = self;
        let account = completed
            .as_ref()
            .map(|value| value.payload().account.share());
        let retired = RetiredPreparedInputMaterializationError {
            accounting,
            compilation,
            source,
            account,
        };
        // Establish the closed result first so unwinding native retirement also
        // keeps the original account until the concrete prefix has unwound.
        drop(completed);
        retired
    }
    /// The exact comparison or settlement error.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.accounting.as_ref()
    }
    /// The actual materialization error, retaining every preceding leaf.
    pub fn compiler_failure(&self) -> Option<&E> {
        self.compilation.as_ref()
    }
    /// Original bytes still held by the failed materialization.
    pub fn retained_bytes(&self) -> Option<u64> {
        self.completed
            .as_ref()
            .map_or(Some(0), |v| v.original_bytes())
    }
    /// Exact immutable input, including on a foreign-ledger rejection.
    pub fn source(&self) -> &OriginalPreparedHostInput {
        &self.source
    }
}
/// Exact materialization failure after the mechanism's concrete storage has
/// entered its existing retirement path. This exposes no native owner, account
/// or retry; Send/Sync depend only on the original compiler error E.
#[derive(Debug)]
pub struct RetiredPreparedInputMaterializationError<E> {
    accounting: Option<WorkingMemoryError>,
    compilation: Option<E>,
    source: OriginalPreparedHostInput,
    // Both the exact cause and original host source retire before B.
    account: Option<Account>,
}
impl<E> RetiredPreparedInputMaterializationError<E> {
    /// Exact original comparison/settlement diagnostic.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.accounting.as_ref()
    }
    /// Exact original compiler diagnostic; no formatted replacement.
    pub fn compiler_failure(&self) -> Option<&E> {
        self.compilation.as_ref()
    }
    /// Original B still retained by this closed failure.
    pub fn retained_bytes(&self) -> Option<u64> {
        self.account.as_ref().map_or(Some(0), |account| {
            account.inner().requirements.as_ref().map_or(
                Some(account.held_bytes()),
                |requirements| {
                    requirements.iter().try_fold(0u64, |sum, (_, charge)| {
                        sum.checked_add(charge.total().ok()?)
                    })
                },
            )
        })
    }
    /// Immutable I consumed by the failed materialization.
    pub fn source(&self) -> &OriginalPreparedHostInput {
        &self.source
    }
}
impl<E: fmt::Display> fmt::Display for RetiredPreparedInputMaterializationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.compilation {
            Some(cause) => cause.fmt(f),
            None => self
                .accounting
                .as_ref()
                .expect("materialization failure")
                .fmt(f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error
    for RetiredPreparedInputMaterializationError<E>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.compilation
            .as_ref()
            .map(|cause| cause as &dyn std::error::Error)
            .or_else(|| {
                self.accounting
                    .as_ref()
                    .map(|cause| cause as &dyn std::error::Error)
            })
    }
}

impl<T, E: fmt::Debug> fmt::Debug for OriginalPreparedInputMaterializationError<T, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalPreparedInputMaterializationError")
            .field("accounting", &self.accounting)
            .field("compilation", &self.compilation)
            .field("retained_bytes", &self.retained_bytes())
            .finish_non_exhaustive()
    }
}
impl<T, E: fmt::Display> fmt::Display for OriginalPreparedInputMaterializationError<T, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(e) = &self.compilation {
            e.fmt(f)
        } else {
            self.accounting
                .as_ref()
                .expect("materialization failure")
                .fmt(f)
        }
    }
}
impl<T, E: std::error::Error + 'static> std::error::Error
    for OriginalPreparedInputMaterializationError<T, E>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.compilation
            .as_ref()
            .map(|e| e as &dyn std::error::Error)
            .or_else(|| {
                self.accounting
                    .as_ref()
                    .map(|e| e as &dyn std::error::Error)
            })
    }
}

impl MemoryLedger {
    /// Source-derived native recipe plus actual generic host/accounting controls.
    pub fn prepared_native_input_required_bytes<P: PreparedNativeInputCompiler>(
        plan: &P,
    ) -> Result<u64, WorkingMemoryError> {
        let requirements =
            plan.required_storage_requirements(plan.source().memory_ledger().topology())?;
        Self::prepared_native_input_control_bytes::<P>()?
            .checked_add(
                u64::try_from(plan.required_storage_bytes()?)
                    .map_err(|_| WorkingMemoryError::Overflow)?,
            )
            .and_then(|bytes| bytes.checked_add(requirements.backing_bytes().ok()?))
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn prepared_native_input_control_bytes<P: PreparedNativeInputCompiler>()
    -> Result<u64, WorkingMemoryError> {
        let account_arc = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<AccountInner>())
            .map_err(|_| WorkingMemoryError::Overflow)?
            .0
            .pad_to_align()
            .size();
        let controls = [
            account_arc,
            size_of::<Payload<P::Output>>(),
            size_of::<Payload<P::Output>>(), // unboxed retirement value
            size_of::<OriginalPreparedInputMaterialization<P::Output>>(),
            size_of::<OriginalPreparedInputMaterializationError<P::Output, P::Error>>(),
            size_of::<
                Result<
                    OriginalPreparedInputMaterialization<P::Output>,
                    OriginalPreparedInputMaterializationError<P::Output, P::Error>,
                >,
            >(),
            size_of::<Result<P::Output, (P::Output, P::Error)>>(),
            size_of::<super::loaded_decode_source::Allowance>(),
            size_of::<Account>(),
            size_of::<Option<AccountInner>>(),
            size_of::<OriginalPreparedInputCustody>(),
            size_of::<P>(),
            size_of::<Result<(), WorkingMemoryError>>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        u64::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)
    }
    /// Compares once before every compiler allocation. Completed source/error
    /// storage and independent native aliases retain the original full charge.
    pub fn compile_prepared_native_input<P: PreparedNativeInputCompiler>(
        &self,
        plan: P,
    ) -> Result<
        OriginalPreparedInputMaterialization<P::Output>,
        OriginalPreparedInputMaterializationError<P::Output, P::Error>,
    > {
        let source = plan.source().clone();
        let reject = |e| OriginalPreparedInputMaterializationError::rejected(source.clone(), e);
        source.validate_pool(self).map_err(reject)?;
        let mut requirements = plan
            .required_storage_requirements(self.topology())
            .map_err(reject)?;
        requirements
            .validate(self.topology())
            .map_err(WorkingMemoryError::from)
            .map_err(reject)?;
        let generic = Self::prepared_native_input_control_bytes::<P>().map_err(reject)?;
        let host = generic
            .checked_add(
                requirements
                    .backing_bytes()
                    .map_err(WorkingMemoryError::from)
                    .map_err(reject)?,
            )
            .ok_or_else(|| reject(WorkingMemoryError::Overflow))?;
        requirements
            .add_allocation(host, self.host_placement())
            .map_err(WorkingMemoryError::from)
            .map_err(reject)?;
        let account = Account::new_domains(self.clone(), requirements);
        {
            let requirements = account
                .inner()
                .requirements
                .as_ref()
                .expect("prepared requirements");
            let mut usage = self
                .0
                .usage
                .lock()
                .map_err(|_| reject(WorkingMemoryError::Poisoned))?;
            if usage.unquoted_owners != 0 {
                return Err(reject(WorkingMemoryError::UnknownBound));
            }
            let reservations = usage
                .reservations
                .checked_add(1)
                .ok_or_else(|| reject(WorkingMemoryError::Overflow))?;
            for (slot, (domain, charge)) in requirements.iter().enumerate() {
                let increment = charge
                    .total()
                    .map_err(WorkingMemoryError::from)
                    .map_err(reject)?;
                let current = &usage.domains[slot];
                let existing = self.0.domains[slot]
                    .existing
                    .checked_add(current.registered)
                    .and_then(|bytes| bytes.checked_add(current.reserved))
                    .ok_or_else(|| reject(WorkingMemoryError::Overflow))?;
                self.0
                    .domain_capacity(&usage, domain, None)
                    .map_err(reject)?
                    .check(domain, existing, increment)
                    .map_err(WorkingMemoryError::from)
                    .map_err(reject)?;
                current
                    .reserved
                    .checked_add(increment)
                    .ok_or_else(|| reject(WorkingMemoryError::Overflow))?;
                current
                    .placement_allowances
                    .checked_add(charge.placement_allowance_bytes)
                    .ok_or_else(|| reject(WorkingMemoryError::Overflow))?;
                current
                    .estimates
                    .checked_add(charge.estimated_overhead_bytes)
                    .ok_or_else(|| reject(WorkingMemoryError::Overflow))?;
                current
                    .headroom
                    .checked_add(charge.headroom_bytes)
                    .ok_or_else(|| reject(WorkingMemoryError::Overflow))?;
            }
            // Fixed scalar writes only. All domains and counters were validated.
            for (slot, (_, charge)) in requirements.iter().enumerate() {
                let current = &mut usage.domains[slot];
                current.reserved += charge.total().expect("validated domain charge");
                current.placement_allowances += charge.placement_allowance_bytes;
                current.estimates += charge.estimated_overhead_bytes;
                current.headroom += charge.headroom_bytes;
                current.peak = current
                    .peak
                    .max(self.0.domains[slot].existing + current.registered + current.reserved);
            }
            usage.reservations = reservations;
            account.activate();
        }
        let result = plan.compile(account.native_owner());
        let (storage, compilation) = match result {
            Ok(v) => (v, None),
            Err((v, e)) => (v, Some(e)),
        };
        let materialized = OriginalPreparedInputMaterialization(Some(Box::new(Payload {
            storage,
            source: source.clone(),
            account,
        })));
        let accounting = materialized.payload().account.finish().err();
        if accounting.is_some() || compilation.is_some() {
            Err(OriginalPreparedInputMaterializationError {
                accounting,
                compilation,
                completed: Some(materialized),
                source,
            })
        } else {
            Ok(materialized)
        }
    }
}

#[cfg(test)]
mod tests;
