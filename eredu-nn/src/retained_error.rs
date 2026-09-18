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
    /// and concrete Box deallocation. This is the canonical typed-source owner;
    /// caller-owned diagnostic text uses `backend_message` instead.
    pub fn backend_retained_source(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        let source: Box<dyn Source> = Box::new(error);
        let owner = SourceOwner(Some(source));
        let inner = Inner {
            source: owner,
            #[cfg(test)]
            retired_control: None,
        };
        let shared = Arc::new(inner);
        let retained = RetainedSource(Some(shared));
        Self {
            storage: ErrorStorage::Retained(retained),
        }
    }
    /// Complete source Box, shared control allocation and concrete constructor
    /// transports for the canonical typed-source worker. Nested source contents,
    /// allocator overhead and the caller's own controls remain separate.
    pub fn retained_source_construction_bytes<E: std::error::Error + Send + Sync + 'static>()
    -> Option<usize> {
        let shared = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<Inner>())
            .ok()?
            .0
            .pad_to_align();
        let allocations = shared.size().checked_add(std::mem::size_of::<E>())?;
        // These are the actual input, coercion, nested constructor and return
        // values above. Caller-specific source/result locals are not included.
        let frames = [
            std::mem::size_of::<E>(),
            std::mem::size_of::<Box<E>>(),
            std::mem::size_of::<Box<dyn Source>>(),
            std::mem::size_of::<Option<Box<dyn Source>>>(),
            std::mem::size_of::<SourceOwner>(),
            std::mem::size_of::<Inner>(),
            std::mem::size_of::<Arc<Inner>>(),
            std::mem::size_of::<Option<Arc<Inner>>>(),
            std::mem::size_of::<RetainedSource>(),
            std::mem::size_of::<ErrorStorage>(),
            std::mem::size_of::<Error>(),
        ];
        frames.into_iter().try_fold(allocations, usize::checked_add)
    }
}
#[cfg(test)]
mod tests;
