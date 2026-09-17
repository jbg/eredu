//! Closed host-metadata accounting; no execution or native allocation authority.
mod prepaid;
use std::{
    alloc::Layout,
    fmt,
    mem::{size_of, size_of_val},
    sync::{Arc, atomic::AtomicUsize},
};

/// Fixed host-funding refusal shared with the neutral public error boundary.
use crate::HostMetadataFundingError;

/// An actual retained host-metadata account supplied by the caller.
///
/// A successful reservation must retain the requested amount until this
/// account retires. Refusal must leave the account unchanged. Spans and retired
/// temporary values do not refund these cumulative constructor reservations.
/// Implementations own their host admission and final refund policy; this
/// contract supplies no execution, tensor, or native allocation permission.
pub trait HostMetadataAccount: fmt::Debug + Send + Sync + 'static {
    /// Reserves the next actual constructor before it allocates or mutates its
    /// destination. Failure must use the fixed, allocation-free refusal above.
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError>;
}

trait ErasedAccount: Send + Sync {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError>;
    fn retire(self: Box<Self>);
}

struct Account<A>(Option<A>);

// The caller receives the moved value only after the concrete Box allocation
// is gone. Dropping A inside Box's ordinary destructor would refund too early.
fn unbox<T>(value: Box<T>) -> T {
    *value
}

impl<A: HostMetadataAccount> ErasedAccount for Account<A> {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        self.0
            .as_ref()
            .expect("live workspace metadata account")
            .reserve_metadata(bytes)
    }

    fn retire(self: Box<Self>) {
        let Account(account) = unbox(self);
        drop(account);
    }
}

struct Owner(Option<Box<dyn ErasedAccount>>);

impl Drop for Owner {
    fn drop(&mut self) {
        if let Some(account) = self.0.take() {
            account.retire();
        }
    }
}

/// A closed shared host-metadata account. Cloning retains the same account and
/// its cumulative reservations; it creates neither a new allowance nor a grant.
///
/// The last alias frees this shared control and the concrete account Box before
/// the account's destructor can refund its admission. No raw or weak owner can
/// escape that retirement order. Callers must retain an independent alias with
/// every output or error that can outlive the Context which constructed it.
#[derive(Clone)]
pub struct HostMetadataFunding(Option<Arc<Owner>>);

impl HostMetadataFunding {
    /// Fixed transports for reserving through this same erased account.
    /// This describes the call frame and grants no amount or execution authority.
    pub const fn reservation_control_bytes() -> usize {
        size_of::<Option<&HostMetadataFunding>>()
            + size_of::<&dyn ErasedAccount>()
            + size_of::<usize>()
            + size_of::<HostMetadataFundingError>()
            + size_of::<Result<(), HostMetadataFundingError>>()
    }

    /// Exact requested account Box, shared control and constructor transports
    /// for the actual A. Storage already owned by A and host allocator overhead
    /// require the caller's separate qualification.
    pub fn constructor_bytes<A: HostMetadataAccount>() -> Option<usize> {
        let shared = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Owner>())
            .ok()?
            .0
            .pad_to_align();
        let parts = [
            Layout::new::<Account<A>>().size(),
            shared.size(),
            size_of::<A>(),
            size_of::<Account<A>>(),
            size_of::<Owner>(),
            size_of::<Box<Account<A>>>(),
            size_of::<Box<dyn ErasedAccount>>(),
            size_of::<Arc<Owner>>(),
            size_of::<Self>(),
            size_of::<Result<Self, HostMetadataFundingError>>(),
            size_of::<Option<usize>>(),
            size_of::<Layout>(),
            Self::reservation_control_bytes(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }

    /// Reserves its complete constructor on the supplied account before either
    /// allocation. On refusal the unboxed account retires without a new shell.
    pub fn new<A: HostMetadataAccount>(
        account: A,
    ) -> Result<Self, HostMetadataFundingError> {
        let bytes =
            Self::constructor_bytes::<A>().ok_or(HostMetadataFundingError::Overflow)?;
        account.reserve_metadata(bytes)?;
        Ok(Self(Some(Arc::new(Owner(Some(Box::new(Account(Some(
            account,
        )))))))))
    }

    /// Whether both handles retain this exact cumulative metadata account.
    /// Equality is descriptive and does not reserve bytes or grant execution.
    pub fn same_account(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.0.as_ref().expect("live workspace funding"),
            other.0.as_ref().expect("live workspace funding"))
    }

    /// Reserves bytes on this same retained account before the caller's actual
    /// metadata producer. No reservation is refunded until final account retirement.
    pub fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        self.0
            .as_ref()
            .expect("live workspace metadata funding")
            .0
            .as_ref()
            .expect("live workspace metadata owner")
            .reserve_metadata(bytes)
    }
}

impl fmt::Debug for HostMetadataFunding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostMetadataFunding")
            .finish_non_exhaustive()
    }
}

impl Drop for HostMetadataFunding {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // Every strong alias uses into_inner and no Weak exists. Exactly
            // one final alias receives Owner after the Arc shell has retired.
            if let Some(owner) = Arc::into_inner(owner) {
                drop(owner);
            }
        }
    }
}


