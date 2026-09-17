//! Sealed rows shared only with the historical quote and native work owner.
use super::PreparedTextInterventions;
use eredu_runtime::working_memory::WorkingMemoryError;
use std::{
    alloc::Layout,
    cell::Cell,
    fmt,
    mem::{size_of, ManuallyDrop},
    ops::Deref,
    rc::{Rc, Weak},
};

pub(crate) struct PreparedTextInterventionsOwner(Option<Rc<PreparedTextInterventions>>);
impl PreparedTextInterventionsOwner {
    pub(super) fn new(rows: PreparedTextInterventions) -> Result<Self, WorkingMemoryError> {
        if rows.active.is_some() || rows.next_prediction != rows.end_prediction {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // The actual constructor allocation was reserved by row preparation.
        Ok(Self(Some(Rc::new(rows))))
    }
}
impl Clone for PreparedTextInterventionsOwner {
    fn clone(&self) -> Self {
        Self(Some(self.0.as_ref().expect("live rows").clone()))
    }
}
impl Deref for PreparedTextInterventionsOwner {
    type Target = PreparedTextInterventions;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live rows")
    }
}
impl fmt::Debug for PreparedTextInterventionsOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PreparedTextInterventionsOwner")
            .field(&**self)
            .finish()
    }
}
impl Drop for PreparedTextInterventionsOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            let rows = Rc::into_inner(owner);
            drop(rows);
        }
    }
}
// Same pinned Rust RcInner layout and closed no-Weak population as the existing
// text quote owner. The final block retires before the rows' preparation account.
pub(super) fn control_bytes() -> Option<usize> {
    let block = Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align()
        .extend(Layout::new::<PreparedTextInterventions>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    [
        size_of::<[PreparedTextInterventions; 3]>(),
        size_of::<PreparedTextInterventionsOwner>(),
        size_of::<Option<PreparedTextInterventionsOwner>>(),
        size_of::<Option<Rc<PreparedTextInterventions>>>(),
        size_of::<Rc<PreparedTextInterventions>>(),
        size_of::<ManuallyDrop<Rc<PreparedTextInterventions>>>(),
        size_of::<Weak<PreparedTextInterventions>>(),
        size_of::<Result<PreparedTextInterventions, Rc<PreparedTextInterventions>>>(),
        size_of::<Option<PreparedTextInterventions>>(),
        size_of::<Result<PreparedTextInterventionsOwner, WorkingMemoryError>>(),
    ]
    .into_iter()
    .try_fold(block, usize::checked_add)
}
