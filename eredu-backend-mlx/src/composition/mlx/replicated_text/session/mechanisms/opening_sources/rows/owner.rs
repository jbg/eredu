//! One closed strong population and one once-minted installed weak per bank.
use super::{NativeOpeningRows, OpeningError, OriginalTextControlGuard, RowError};
use crate::backend::error::Error;
use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    mem::{size_of, ManuallyDrop},
    ops::Deref,
    rc::{Rc, Weak},
};

#[derive(Debug)]
pub(crate) struct NativeOpeningRowsOwner(Option<Rc<NativeOpeningRows>>);
impl NativeOpeningRowsOwner {
    pub(super) fn new(rows: NativeOpeningRows) -> Self {
        Self(Some(Rc::new(rows)))
    }

    #[cfg(test)]
    pub(crate) fn strong_count_for_test(&self) -> usize {
        Rc::strong_count(self.0.as_ref().expect("live row owner"))
    }

    // Only the actual bank can mint its one installation. Cloning a strong
    // owner cannot reset the flag or construct a second weak custody owner.
    pub(in crate::composition::mlx::replicated_text::session::mechanisms::opening_sources) fn claim_installation(
        &self,
    ) -> Result<InstalledOpeningRows, RowError> {
        if self.installed.replace(true) {
            return Err(OpeningError::Used.into());
        }
        Ok(InstalledOpeningRows {
            weak: Some(Rc::downgrade(self.0.as_ref().expect("live row owner"))),
            _custody: self.custody.clone(),
        })
    }
}
impl Clone for NativeOpeningRowsOwner {
    fn clone(&self) -> Self {
        Self(Some(Rc::clone(self.0.as_ref().expect("live row owner"))))
    }
}
impl Deref for NativeOpeningRowsOwner {
    type Target = NativeOpeningRows;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live row owner")
    }
}
impl AsRef<NativeOpeningRows> for NativeOpeningRowsOwner {
    fn as_ref(&self) -> &NativeOpeningRows {
        self
    }
}
impl Drop for NativeOpeningRowsOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            let rows = extract_rows(owner);
            // Preserve semantic last-strong destruction. With an installed weak,
            // its separate original guard keeps the Rc allocation charged; if
            // it was removed first, extraction already freed that allocation.
            drop(rows);
        }
    }
}
fn extract_rows(owner: Rc<NativeOpeningRows>) -> Option<NativeOpeningRows> {
    Rc::into_inner(owner)
}

/// No Clone, raw Weak export or independently supplied guard. This accounting
/// owner never retains semantic rows or refreshes their prepared authority.
#[derive(Debug)]
pub(in crate::composition::mlx::replicated_text) struct InstalledOpeningRows {
    weak: Option<Weak<NativeOpeningRows>>,
    // Last: the actual original aggregate and registered source witnesses
    // survive final Rc block retirement, even after the row value is gone.
    _custody: OriginalTextControlGuard,
}
impl InstalledOpeningRows {
    pub(in crate::composition::mlx::replicated_text) fn is_live(&self) -> bool {
        self.weak
            .as_ref()
            .is_some_and(|weak| weak.strong_count() != 0)
    }
    pub(in crate::composition::mlx::replicated_text) fn upgrade(
        &self,
    ) -> Option<NativeOpeningRowsOwner> {
        self.weak
            .as_ref()
            .and_then(Weak::upgrade)
            .map(|owner| NativeOpeningRowsOwner(Some(owner)))
    }
}
impl Drop for InstalledOpeningRows {
    fn drop(&mut self) {
        // Drop can release the allocation. It must complete before the field
        // cleanup can release original accounting/source custody below.
        drop(self.weak.take());
    }
}

/// Actual fixed controls for this one closed allocation/installation. Existing
/// A and W count their own stored strong fields and call/return movements.
/// Supported Rust1.98 RcInner is repr(C, align(2)): two Cell<usize>, then T.
/// Includes allocation padding, not allocator usable size or arbitrary stacks.
pub(super) fn control_bytes() -> Option<u64> {
    type RuntimeError = eredu_runtime::ReplicatedTextSessionError<eredu_nn::Error, Error, Error>;
    let header = Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align();
    let block = header
        .extend(Layout::new::<NativeOpeningRows>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let controls = [
        // Caller aggregate, closed new argument, and Rc::new argument.
        size_of::<NativeOpeningRows>(),
        size_of::<NativeOpeningRows>(),
        size_of::<NativeOpeningRows>(),
        // Strong return/caller and upgrade outputs; no new heap per clone.
        size_of::<NativeOpeningRowsOwner>(),
        size_of::<Option<NativeOpeningRowsOwner>>(),
        size_of::<Result<NativeOpeningRowsOwner, super::RowAllocationFailure>>(),
        size_of::<Result<NativeOpeningRowsOwner, Error>>(),
        size_of::<Result<NativeOpeningRowsOwner, eredu_core::BackendFailure>>(),
        // Strong Drop, helper argument and std consuming extraction controls.
        size_of::<Option<Rc<NativeOpeningRows>>>(),
        size_of::<Rc<NativeOpeningRows>>(),
        size_of::<Rc<NativeOpeningRows>>(),
        size_of::<ManuallyDrop<Rc<NativeOpeningRows>>>(),
        size_of::<Weak<NativeOpeningRows>>(),
        size_of::<Result<NativeOpeningRows, Rc<NativeOpeningRows>>>(),
        size_of::<Option<NativeOpeningRows>>(),
        size_of::<Option<NativeOpeningRows>>(),
        // One installed slot, newly minted return/move, and a displaced/taken
        // Option outside all loans. A prior bank funds its own displaced value.
        size_of::<RefCell<Option<InstalledOpeningRows>>>(),
        size_of::<InstalledOpeningRows>(),
        size_of::<Result<InstalledOpeningRows, RowError>>(),
        size_of::<Option<InstalledOpeningRows>>(),
        size_of::<Result<Option<InstalledOpeningRows>, Error>>(),
        size_of::<Result<Option<InstalledOpeningRows>, RuntimeError>>(),
        size_of::<Result<(), Error>>(),
        // Weak downgrade/Drop and guard-clone return/move controls.
        size_of::<Weak<NativeOpeningRows>>(),
        size_of::<Option<Weak<NativeOpeningRows>>>(),
        size_of::<OriginalTextControlGuard>(),
    ]
    .into_iter()
    .try_fold(block, usize::checked_add)?;
    u64::try_from(controls).ok()
}

// The field is after all semantic row/source controls and before final custody.
// This observes actual value destruction, not Weak expiration or an Rc counter.
#[cfg(test)]
pub(super) struct RetirementProbe(pub(super) Rc<Cell<usize>>);
#[cfg(test)]
impl Drop for RetirementProbe {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}
