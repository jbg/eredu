//! Closed concrete Rc population; payload retirement follows allocation retirement.
use super::FundedWork;
use std::{
    alloc::Layout,
    cell::Cell,
    fmt,
    mem::{size_of, ManuallyDrop},
    ops::Deref,
    rc::{Rc, Weak},
};

/// No Rc/Weak export. Every strong exit uses the same consuming retirement.
/// No native completion, scope certification or new host hold is implied.
pub(in crate::composition::mlx::session::model_session) struct FundedWorkOwner(
    Option<Rc<FundedWork>>,
);
impl FundedWorkOwner {
    pub(super) fn new(value: FundedWork) -> Self {
        Self(Some(Rc::new(value)))
    }
}
impl Clone for FundedWorkOwner {
    fn clone(&self) -> Self {
        Self(Some(self.0.as_ref().expect("live closed owner").clone()))
    }
}
impl Deref for FundedWorkOwner {
    type Target = FundedWork;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live closed owner")
    }
}
impl AsRef<FundedWork> for FundedWorkOwner {
    fn as_ref(&self) -> &FundedWork {
        self
    }
}
impl fmt::Debug for FundedWorkOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FundedWorkOwner").finish_non_exhaustive()
    }
}
impl Drop for FundedWorkOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // The closed population has no Weak. With the final strong owner,
            // the Rc block is gone before these source/native/custody fields.
            // Existing guarded native recovery destruction remains guarded.
            let payload = Rc::into_inner(owner);
            drop(payload);
        }
    }
}

/// Original diagnostics only. Supported Rust1.98 RcInner is repr(C, align(2))
/// with two Cell<usize> counters followed by T; official source is pinned in
/// this change. Layout includes trailing padding; this is requested allocation
/// storage, not allocator usable size/RSS. Reaudit on a toolchain layout change.
/// No separately owned payload buffer or arbitrary alias population is added.
pub(super) fn control_bytes() -> Option<u64> {
    let header = Layout::new::<[Cell<usize>; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align();
    let block = header
        .extend(Layout::new::<FundedWork>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let bytes = block
        .checked_add(size_of::<[Option<crate::composition::mlx::session::intervention::PreparedTextInterventionsOwner>; 4]>())?
        .checked_add(super::publication_scope::control_bytes()?)?
        // Caller aggregate, owner constructor parameter, Rc::new parameter.
        .checked_add(size_of::<FundedWork>())?
        .checked_add(size_of::<FundedWork>())?
        .checked_add(size_of::<FundedWork>())?
        // Closed return/caller Option and the exact consuming-retirement locals.
        .checked_add(size_of::<FundedWorkOwner>())?
        .checked_add(size_of::<Option<FundedWorkOwner>>())?
        .checked_add(size_of::<Option<Rc<FundedWork>>>())?
        .checked_add(size_of::<Rc<FundedWork>>())?
        .checked_add(size_of::<ManuallyDrop<Rc<FundedWork>>>())?
        .checked_add(size_of::<Weak<FundedWork>>())?
        .checked_add(size_of::<Result<FundedWork, Rc<FundedWork>>>())?
        .checked_add(size_of::<Option<FundedWork>>())?;
    u64::try_from(bytes).ok()
}
