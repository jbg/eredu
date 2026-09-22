//! Known collector storage for the existing funded saved-copy worker.
//!
//! Native clone handles, primitive/record construction and publication have
//! separate owners. This contribution does not certify complete original fit.

use super::{
    memory, text_funding, Array, Error, WorkingMemoryError, WorkspaceCopyCustody,
    WorkspaceCopyLimits,
};
use eredu_core::{BackendFailure, BackendFailureKind};
use eredu_runtime::working_memory::WorkingMemoryFundingScope;
use std::{
    alloc::Layout,
    cell::{Cell, RefCell, RefMut},
    collections::TryReserveError,
    mem::size_of,
    rc::Rc,
};

type Roots = Rc<RefCell<Vec<Array>>>;

/// Same plan supplies admission bytes and the post-admission constructor.
#[derive(Clone, Copy)]
pub(super) struct RootCollectorPlan {
    roots: usize,
    known_bytes: u64,
}

impl RootCollectorPlan {
    pub(super) fn from_root_count(roots: usize) -> Result<Self, Error> {
        Self::from_root_count_fixed(roots).map_err(memory)
    }

    /// Same owning constructor layout, callable before host admission without
    /// allocating a boxed backend diagnostic on a failed capacity calculation.
    pub(super) fn from_root_count_fixed(roots: usize) -> Result<Self, WorkingMemoryError> {
        let overflow = || WorkingMemoryError::Overflow;
        let backing = Layout::array::<Array>(roots)
            .map_err(|_| overflow())?
            .size();
        // Same supported Rust RcInner layout used by the existing funded-work
        // owner: two Cell<usize> counters, then the actual payload and padding.
        let block = Layout::new::<[Cell<usize>; 2]>()
            .align_to(2)
            .map_err(|_| overflow())?
            .pad_to_align()
            .extend(Layout::new::<RefCell<Vec<Array>>>())
            .map_err(|_| overflow())?
            .0
            .pad_to_align()
            .size();
        let controls = [
            size_of::<Self>(),
            size_of::<Vec<Array>>(),
            size_of::<RefCell<Vec<Array>>>(),
            size_of::<RefCell<Vec<Array>>>(),
            size_of::<Roots>(),
            size_of::<Roots>(),
            size_of::<Roots>(),
            size_of::<RefMut<'static, Vec<Array>>>(),
            size_of::<TryReserveError>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Roots, TryReserveError>>(),
            // Core owns the concrete source Box and retires that allocation
            // before the original reserve cause and its copy custody.
            BackendFailure::source_retention_peak_bytes::<CollectorAllocationFailure>()
                .ok_or_else(overflow)?,
            size_of::<BackendFailure>(),
            size_of::<WorkingMemoryFundingScope>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Error>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(overflow)?;
        let known_bytes = backing
            .checked_add(block)
            .and_then(|n| n.checked_add(controls))
            .ok_or_else(overflow)?;
        Ok(Self {
            roots,
            known_bytes: u64::try_from(known_bytes).map_err(|_| overflow())?,
        })
    }

    pub(super) fn retention_control_bytes(self) -> u64 {
        self.known_bytes
    }

    pub(super) fn publication_control_bytes(self) -> Result<u64, WorkingMemoryError> {
        crate::backend::runtime::residency::storage::generic_storage_publication_layout(
            self.roots
                .checked_mul(2)
                .ok_or(WorkingMemoryError::Overflow)?,
        )?
        .requested_bytes()
        .checked_add(eredu_runtime::working_memory::MemoryLedger::storage_metadata_control_bytes()?)
        .ok_or(WorkingMemoryError::Overflow)
    }

    pub(super) fn limits(
        self,
        limits: WorkspaceCopyLimits,
        generic_publication: bool,
    ) -> Result<WorkspaceCopyLimits, Error> {
        // Existing transport field, with a measured contribution; the caller's
        // reserve stays additive and the same account owns the entire amount.
        let publication = if generic_publication {
            self.publication_control_bytes().map_err(memory)?
        } else {
            0
        };
        let mut limits = text_funding::copy_limits_with_work_controls(limits)?;
        limits.additional_host_metadata_bytes = limits
            .additional_host_metadata_bytes
            .checked_add(self.known_bytes)
            .and_then(|bytes| bytes.checked_add(publication))
            .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
        Ok(limits)
    }

    /// Called only after the original copy account accepted this same plan.
    pub(super) fn construct(self) -> Result<Roots, TryReserveError> {
        let mut roots = Vec::new();
        roots.try_reserve_exact(self.roots)?;
        Ok(Rc::new(RefCell::new(roots)))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct CollectorAllocationFailure {
    #[source]
    cause: TryReserveError,
    // The original allocator cause retires before the same destination account.
    _custody: WorkspaceCopyCustody,
}

/// No source clone, native constructor or recovery scope has been entered.
/// Any host-only destination prefix retains its independent host scope; the
/// decoder caller retires its empty tables before entering this helper.
/// A failed certification still follows the existing quarantine behavior; it
/// never replaces the original allocation refusal or drops its copy custody.
pub(super) fn allocation_failure(
    cause: TryReserveError,
    custody: WorkspaceCopyCustody,
    scope: WorkingMemoryFundingScope,
) -> Error {
    let _ = scope.certify();
    Error::SavedCopyConstructor(BackendFailure::new(
        BackendFailureKind::ResourceExhausted,
        CollectorAllocationFailure {
            cause,
            _custody: custody,
        },
    ))
}
