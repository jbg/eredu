//! One closed shared owner for capture payloads and their final custody.
use std::sync::Arc;

trait CaptureCustody: Send + Sync {
    fn retire(self: Box<Self>);
}
fn unbox_custody<T>(owner: Box<T>) -> T {
    *owner
}
impl<T: Send + Sync> CaptureCustody for T {
    fn retire(self: Box<Self>) {
        let value = unbox_custody(self);
        drop(value);
    }
}
struct CaptureCustodyOwner(Option<Box<dyn CaptureCustody>>);
impl Drop for CaptureCustodyOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.retire();
        }
    }
}
pub(super) struct RetainedCapturePayload<T> {
    // Every payload field retires before its custody.
    value: T,
    _custody: CaptureCustodyOwner,
}

/// Crate-private storage, never a source, funding or completion grant.
/// No Weak/raw export exists. The unique mutable loan is used only by closed
/// unpublished wrappers; published wrappers expose immutable loans only.
pub(crate) struct RetainedCaptureOwner<T>(pub(super) Option<Arc<RetainedCapturePayload<T>>>);
impl<T> Clone for RetainedCaptureOwner<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> RetainedCaptureOwner<T> {
    pub(crate) fn get(&self) -> &T {
        &self.0.as_ref().expect("live capture payload").value
    }
    pub(crate) fn get_mut(&mut self) -> &mut T {
        &mut Arc::get_mut(self.0.as_mut().expect("live capture payload"))
            .expect("unpublished capture payload has no aliases")
            .value
    }
    pub(crate) fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live capture payload"),
            other.0.as_ref().expect("live capture payload"),
        )
    }
    pub(crate) fn retained_control_bytes<C: Send + Sync + 'static>() -> Option<u64> {
        // The same pinned ArcInner representation used by capture frames.
        let arc = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<RetainedCapturePayload<T>>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        u64::try_from(arc.checked_add(std::mem::size_of::<C>())?).ok()
    }
    pub(crate) fn retain(value: T, custody: impl Send + Sync + 'static) -> Self {
        Self(Some(Arc::new(RetainedCapturePayload {
            value,
            _custody: CaptureCustodyOwner(Some(Box::new(custody))),
        })))
    }
}
impl<T> Drop for RetainedCaptureOwner<T> {
    fn drop(&mut self) {
        // Every strong owner follows this consuming path. Exactly one final
        // concurrent drop obtains the payload after Arc's allocation is gone;
        // its fields retire before the concrete custody Box and its value.
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
