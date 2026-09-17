//! Retained sources with closed concrete retirement; no core/native dependency.
use super::{Error, ErrorStorage};
use std::{fmt, sync::Arc};
type ErrorObject = dyn std::error::Error + Send + Sync + 'static;
trait Source: Send + Sync {
    fn error(&self) -> &ErrorObject;
    fn retire(self: Box<Self>);
}
fn unbox<T>(value: Box<T>) -> T {
    *value
}
impl<E: std::error::Error + Send + Sync + 'static> Source for E {
    fn error(&self) -> &ErrorObject {
        self
    }
    fn retire(self: Box<Self>) {
        drop(unbox(self));
    }
}
struct SourceOwner(Option<Box<dyn Source>>);
impl Drop for SourceOwner {
    fn drop(&mut self) {
        if let Some(source) = self.0.take() {
            source.retire();
        }
    }
}
struct Inner {
    source: SourceOwner,
    #[cfg(test)]
    retired_control: Option<Arc<std::sync::atomic::AtomicBool>>,
}
#[derive(Clone)]
pub(super) struct RetainedSource(Option<Arc<Inner>>);
impl RetainedSource {
    pub(super) fn original(&self) -> &ErrorObject {
        self.0
            .as_ref()
            .expect("live neural source")
            .source
            .0
            .as_ref()
            .expect("live source")
            .error()
    }
}
impl Drop for RetainedSource {
    fn drop(&mut self) {
        if let Some(inner) = self.0.take() {
            // Remove this control allocation before the concrete source Box,
            // then its Error/custody. No raw/Weak exit bypasses this discipline.
            if let Some(inner) = Arc::into_inner(inner) {
                #[cfg(test)]
                if let Some(retired) = &inner.retired_control {
                    retired.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                drop(inner);
            }
        }
    }
}
impl fmt::Debug for RetainedSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.original(), f)
    }
}
impl fmt::Display for RetainedSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.original(), f)
    }
}
impl Error {
    /// Shares a complete existing source without eagerly formatting a String.
    /// The caller already owns all required custody before this constructor;
    /// this owner grants no allocation authority, quota or native completion.
    /// Clone shares both control and source; source retirement follows control
    /// and concrete Box deallocation. Legacy constructors remain unchanged.
    pub fn backend_retained_source(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self {
            storage: ErrorStorage::Retained(RetainedSource(Some(Arc::new(Inner {
                source: SourceOwner(Some(Box::new(error))),
                #[cfg(test)]
                retired_control: None,
            })))),
        }
    }
    /// Exact retained source/control allocations for the actual concrete E.
    /// Dynamic source contents, allocator overhead and stack moves are separate.
    pub fn retained_source_control_bytes<E: std::error::Error + Send + Sync + 'static>(
    ) -> Option<usize> {
        std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<Inner>())
            .ok()?
            .0
            .pad_to_align()
            .size()
            .checked_add(std::mem::size_of::<E>())
    }
}
#[cfg(test)]
mod tests;
