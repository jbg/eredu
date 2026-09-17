//! Finite metadata root selection, shared with ordinary borrowed inspection.
use super::*;
use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

/// Fixed selection validation or the original failed Vec reservation.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceBorrowedStorageError {
    #[error("borrowed storage belongs to another workspace context")]
    /// A root belongs to another workspace.
    Context,
    #[error("borrowed storage requires an exact known capacity")]
    /// A supplied root has no declared physical capacity.
    UnknownCapacity,
    #[error("borrowed storage capacity sum overflow")]
    /// The capacity sum or constructor layout overflows.
    Overflow,
    #[error("borrowed selection belongs to another workspace context")]
    /// An installed selection belongs to another workspace.
    SelectionContext,
    #[error("borrowed storage must be selected once before workspace tracing")]
    /// Selection was already installed or tracing began.
    SelectionStarted,
    #[error("borrowed storage exceeds its prepared root slots")]
    /// Unique retained roots exceed the supplied constructor slots.
    Slots,
    #[error("borrowed storage reservation failed")]
    /// The exact root vector failed to reserve.
    Reserve(#[source] TryReserveError),
}
impl WorkspaceBorrowedStorageError {
    pub(super) fn into_ordinary(self) -> Error {
        match self {
            Self::Context => {
                Error::backend("borrowed storage belongs to another workspace context")
            }
            Self::UnknownCapacity => {
                Error::backend("borrowed storage requires an exact known capacity")
            }
            Self::Overflow => workspace_overflow("borrowed storage capacity sum overflow"),
            Self::SelectionContext => {
                Error::backend("borrowed selection belongs to another workspace context")
            }
            Self::SelectionStarted => {
                Error::backend("borrowed storage must be selected once before workspace tracing")
            }
            Self::Slots => Error::backend("borrowed storage exceeds its prepared root slots"),
            Self::Reserve(cause) => Error::backend(cause.to_string()),
        }
    }
}

/// Immutable selection of exact existing roots excluded by a residual trace.
/// This retains metadata/context, not native storage, registered charges or
/// allocation permission. Provider physical ownership remains independent.
#[derive(Clone, Debug)]
pub struct WorkspaceBorrowedStorage(Rc<BorrowedStorage>);
#[derive(Debug)]
struct BorrowedStorage {
    context: Rc<WorkspaceIdentity>,
    roots: Vec<WorkspaceExistingStorage>,
    total_bytes: u64,
}
impl WorkspaceBorrowedStorage {
    /// Ordinary selection uses the same validation and first-occurrence order.
    /// Aliases occur once; an empty selection remains an explicit no-exclusion proof.
    pub fn new<'a>(
        context: &WorkspaceContext,
        roots: impl IntoIterator<Item = &'a WorkspaceExistingStorage>,
    ) -> Result<Self, Error> {
        Self::build(context, roots, None).map_err(WorkspaceBorrowedStorageError::into_ordinary)
    }

    /// Constructor/storage bound for at most this many unique metadata roots.
    /// Includes one requested root Vec reserve, its Rc shell and fixed constructor,
    /// selection and failure transports. Source/iterator payloads are separate.
    /// The caller qualifies its host allocator and funds construction before use.
    pub fn construction_bytes(slots: usize) -> Option<usize> {
        let rows = Layout::array::<WorkspaceExistingStorage>(slots)
            .ok()?
            .size();
        let shared = Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<BorrowedStorage>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        [
            rows,
            shared,
            size_of::<BorrowedStorage>(),
            size_of::<Self>(),
            size_of::<Result<Self, WorkspaceBorrowedStorageError>>(),
            size_of::<WorkspaceBorrowedStorageError>(),
            size_of::<Vec<WorkspaceExistingStorage>>(),
            size_of::<(
                &WorkspaceContext,
                &WorkspaceExistingStorage,
                Option<usize>,
                u64,
                u64,
                bool,
            )>(),
            size_of::<std::slice::Iter<'static, WorkspaceExistingStorage>>(),
            size_of::<std::cell::RefMut<'static, Option<Self>>>(),
            size_of::<Result<(), WorkspaceBorrowedStorageError>>(),
            size_of::<Result<(), TryReserveError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    /// Builds the same selection using one pre-sized root vector and no set.
    /// Overflow/identity/slot failures are fixed causes; reserve failure keeps
    /// the original allocator cause. No input root or context is mutated.
    pub fn new_finite<'a>(
        context: &WorkspaceContext,
        roots: impl IntoIterator<Item = &'a WorkspaceExistingStorage>,
        slots: usize,
    ) -> Result<Self, WorkspaceBorrowedStorageError> {
        Self::construction_bytes(slots).ok_or(WorkspaceBorrowedStorageError::Overflow)?;
        Self::build(context, roots, Some(slots))
    }

    fn build<'a>(
        context: &WorkspaceContext,
        roots: impl IntoIterator<Item = &'a WorkspaceExistingStorage>,
        slots: Option<usize>,
    ) -> Result<Self, WorkspaceBorrowedStorageError> {
        let mut retained: Vec<WorkspaceExistingStorage> = Vec::new();
        if let Some(slots) = slots {
            retained
                .try_reserve_exact(slots)
                .map_err(WorkspaceBorrowedStorageError::Reserve)?;
        }
        let mut total_bytes = 0u64;
        for root in roots {
            if !Rc::ptr_eq(&root.context, &context.identity) {
                return Err(WorkspaceBorrowedStorageError::Context);
            }
            let bytes = root
                .capacity_bytes()
                .ok_or(WorkspaceBorrowedStorageError::UnknownCapacity)?;
            if retained.iter().any(|prior| prior.same_storage(root)) {
                continue;
            }
            total_bytes = total_bytes
                .checked_add(bytes)
                .ok_or(WorkspaceBorrowedStorageError::Overflow)?;
            if slots.is_some_and(|n| retained.len() == n) {
                return Err(WorkspaceBorrowedStorageError::Slots);
            }
            retained.push(root.clone());
        }
        Ok(Self(Rc::new(BorrowedStorage {
            context: context.identity.clone(),
            roots: retained,
            total_bytes,
        })))
    }
    /// Exact roots in the same first-occurrence order as ordinary construction.
    pub fn roots(&self) -> &[WorkspaceExistingStorage] {
        &self.0.roots
    }
    /// Sum of unique declared capacities; never a scalar admission discount.
    pub fn total_bytes(&self) -> u64 {
        self.0.total_bytes
    }
    /// Same immutable selection token, irrespective of equal capacities.
    pub fn same_identity(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
    pub(super) fn belongs_to(&self, context: &WorkspaceContext) -> bool {
        Rc::ptr_eq(&self.0.context, &context.identity)
    }
    pub(super) fn storage_identities(&self) -> BTreeSet<*const Storage> {
        self.0
            .roots
            .iter()
            .map(|root| Rc::as_ptr(&root.storage))
            .collect()
    }
}
