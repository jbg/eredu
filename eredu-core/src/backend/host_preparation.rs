use std::{any::Any, fmt, sync::Arc};

type ErasedOwner = Arc<dyn Any + Send + Sync>;

#[derive(Clone)]
struct Retention {
    owner: Option<ErasedOwner>,
    retire: fn(ErasedOwner),
}
impl Drop for Retention {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            (self.retire)(owner);
        }
    }
}
fn retire<T: Send + Sync + 'static>(owner: ErasedOwner) {
    let typed = match owner.downcast::<T>() {
        Ok(owner) => owner,
        Err(_) => unreachable!("closed host authority retains its exact type"),
    };
    // No Weak or raw Arc is exported. The final strong clone releases the
    // allocation/header before T can return the account that paid for it.
    drop(Arc::into_inner(typed));
}

/// Shared lifetime custody for backend-authorized host preparation.
///
/// A managed backend retains an exclusion token before host constructors or
/// copies allocate. Every escaping owner must retain custody until its payload
/// retires, including shared aliases and error paths. Enclosing owners must
/// destroy payloads before their authority; this value does not wrap or inspect
/// those payloads and cannot enforce their ownership or drop order.
///
/// This contract concerns exclusion and lifetime only. It proves no finite byte
/// bound, complete storage inventory, inference permission, or completion. An
/// unmanaged value grants no managed-domain evidence. Clones share the same
/// token; they do not acquire fresh authority for another domain or operation.
#[derive(Clone, Default)]
#[must_use = "retain host preparation authority until all covered payloads retire"]
pub struct HostPreparationAuthority {
    owner: Option<Retention>,
}

impl HostPreparationAuthority {
    /// Creates the portable default without managed resource authority.
    pub const fn unmanaged() -> Self {
        Self { owner: None }
    }

    /// Whether this is the ordinary default with no retained host owner.
    /// The opposite reports custody presence only; it supplies no byte bound,
    /// constructor qualification, native permission or completion evidence.
    pub const fn is_unmanaged(&self) -> bool {
        self.owner.is_none()
    }

    /// Checks the actual retained metadata payer without exposing its authority.
    /// This establishes account identity only, never an allocation permission.
    pub(crate) fn is_funded_by(&self, funding: &super::HostMetadataFunding) -> bool {
        self.owner.as_ref().and_then(|retention| retention.owner.as_ref())
            .and_then(|owner| owner.downcast_ref::<super::HostMetadataFunding>())
            .is_some_and(|actual| actual.same_account(funding))
    }

    /// Exact requested shared-shell layout and named constructor/retirement
    /// controls for the same closed authority producer. This reports storage;
    /// it grants neither bytes nor permission to retain an unqualified payload.
    #[doc(hidden)]
    pub fn retention_bytes<T: Send + Sync + 'static>() -> Option<usize> {
        use std::{alloc::Layout, mem::size_of, sync::atomic::AtomicUsize};
        let shared = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<T>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        [
            shared,
            size_of::<Self>(),
            size_of::<Retention>(),
            size_of::<Option<Retention>>(),
            size_of::<Arc<T>>(),
            size_of::<ErasedOwner>(),
            size_of::<Option<ErasedOwner>>(),
            size_of::<Result<Arc<T>, ErasedOwner>>(),
            size_of::<Option<T>>(),
            size_of::<fn(ErasedOwner)>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    /// Backend construction after acquiring the exact domain's exclusion token.
    ///
    /// The supplied owner must be a payload-free lifetime token. It must not own
    /// the host parser, native work, or an enclosing object that retains this
    /// authority. Erasing an arbitrary value does not certify its storage or
    /// establish permission. No lock or callback is invoked here; final token
    /// destruction follows the final authority clone's retirement. The private
    /// typed retirement path releases its shared shell before dropping the token.
    #[doc(hidden)]
    pub fn retain<T: Send + Sync + 'static>(owner: T) -> Self {
        Self {
            owner: Some(Retention {
                owner: Some(Arc::new(owner)),
                retire: retire::<T>,
            }),
        }
    }
}

impl fmt::Debug for HostPreparationAuthority {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never invoke an erased owner's formatting or expose its contents.
        formatter
            .debug_struct("HostPreparationAuthority")
            .field("retains_owner", &self.owner.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
