//! Closed concrete Rc population; payload retirement follows allocation retirement.
use super::TextExecutionQuote;
use std::{
    alloc::Layout,
    cell::Cell,
    fmt,
    mem::{ManuallyDrop, size_of},
    ops::Deref,
    rc::{Rc, Weak},
};

/// No Rc/Weak export. Every strong exit uses the same consuming retirement.
/// No native completion, scope certification or new host hold is implied.
pub(in crate::composition::mlx::session) struct TextExecutionQuoteOwner(
    Option<Rc<TextExecutionQuote>>,
);
impl TextExecutionQuoteOwner {
    pub(super) fn new(value: TextExecutionQuote) -> Self {
        Self(Some(Rc::new(value)))
    }

    pub(in crate::composition::mlx::session) fn same_owner(&self, other: &Self) -> bool {
        Rc::ptr_eq(
            self.0.as_ref().expect("live quote"),
            other.0.as_ref().expect("live quote"),
        )
    }

    // Existing opening-seal rejection fixture only; no raw owning escape.
    #[cfg(test)]
    pub(super) fn unique_for_test(&mut self) -> Option<&mut TextExecutionQuote> {
        Rc::get_mut(self.0.as_mut()?)
    }
}
impl Clone for TextExecutionQuoteOwner {
    fn clone(&self) -> Self {
        Self(Some(self.0.as_ref().expect("live closed owner").clone()))
    }
}
impl Deref for TextExecutionQuoteOwner {
    type Target = TextExecutionQuote;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live closed owner")
    }
}
impl AsRef<TextExecutionQuote> for TextExecutionQuoteOwner {
    fn as_ref(&self) -> &TextExecutionQuote {
        self
    }
}
impl fmt::Debug for TextExecutionQuoteOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TextExecutionQuoteOwner")
            .field(&**self)
            .finish()
    }
}
impl Drop for TextExecutionQuoteOwner {
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
        .extend(Layout::new::<TextExecutionQuote>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let bytes = block
        // Caller aggregate, owner constructor parameter, Rc::new parameter.
        .checked_add(size_of::<TextExecutionQuote>())?
        .checked_add(size_of::<TextExecutionQuote>())?
        .checked_add(size_of::<TextExecutionQuote>())?
        // Closed return/caller Option and the exact consuming-retirement locals.
        .checked_add(size_of::<TextExecutionQuoteOwner>())?
        .checked_add(size_of::<Option<TextExecutionQuoteOwner>>())?
        .checked_add(size_of::<Option<Rc<TextExecutionQuote>>>())?
        .checked_add(size_of::<Rc<TextExecutionQuote>>())?
        .checked_add(size_of::<ManuallyDrop<Rc<TextExecutionQuote>>>())?
        .checked_add(size_of::<Weak<TextExecutionQuote>>())?
        .checked_add(size_of::<Result<TextExecutionQuote, Rc<TextExecutionQuote>>>())?
        .checked_add(size_of::<Option<TextExecutionQuote>>())?;
    u64::try_from(bytes).ok()
}
