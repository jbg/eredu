//! Closed typed source erasure for already-funded native error transports.
use super::Exception;
use std::{
    error::Error,
    mem::{size_of, size_of_val},
    sync::Arc,
};

trait Erased: std::fmt::Debug + Send + Sync {
    fn source(&self) -> &(dyn Error + Send + Sync + 'static);
    fn retire(self: Box<Self>);
}
#[derive(Debug)]
struct Source<E>(Option<E>);
impl<E: Error + Send + Sync + 'static> Erased for Source<E> {
    fn source(&self) -> &(dyn Error + Send + Sync + 'static) {
        self.0.as_ref().expect("live typed source")
    }
    fn retire(mut self: Box<Self>) {
        // Explicitly free the actual erased Box shell before E can release its
        // last account/custody owner. A move-out alone does not order deallocation.
        let cause = self.0.take().expect("live typed source");
        drop(self);
        drop(cause);
    }
}
#[derive(Debug)]
pub(crate) struct ClosedSource(Option<Box<dyn Erased>>);
impl Drop for ClosedSource {
    fn drop(&mut self) {
        if let Some(source) = self.0.take() {
            source.retire();
        }
    }
}
#[derive(Debug)]
pub(crate) enum ExceptionSource {
    Ordinary(Arc<dyn Error + Send + Sync>),
    Retained(ClosedSource),
}
impl ExceptionSource {
    pub(super) fn source(&self) -> &(dyn Error + Send + Sync + 'static) {
        match self {
            Self::Ordinary(source) => source.as_ref(),
            Self::Retained(source) => source.0.as_ref().expect("live retained error").source(),
        }
    }
    pub(super) fn retained(&self) -> bool {
        matches!(self, Self::Retained(_))
    }
}
impl PartialEq for ExceptionSource {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Ordinary(a), Self::Ordinary(b)) => Arc::ptr_eq(a, b),
            (Self::Retained(a), Self::Retained(b)) => std::ptr::eq(a, b),
            _ => false,
        }
    }
}
impl Exception {
    /// One real erased Box and the concrete source/transport controls. The
    /// caller must reserve this before construction and supply E with custody
    /// for its own payload. This query creates neither funding nor authority.
    pub fn retained_source_control_bytes<E: Error + Send + Sync + 'static>() -> Option<usize> {
        let parts = [
            size_of::<Source<E>>(),
            size_of::<E>(),
            size_of::<Self>(),
            size_of::<ClosedSource>(),
            size_of::<ExceptionSource>(),
            size_of::<Box<Source<E>>>(),
            size_of::<Box<dyn Erased>>(),
            size_of::<Option<Box<dyn Erased>>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Retain an already-funded typed source without formatting diagnostic text.
    /// E must itself keep the required source/account custody. Final destruction
    /// frees the erased shell before E; this does not grant storage or settle work.
    #[track_caller]
    pub fn from_retained_source<E: Error + Send + Sync + 'static>(cause: E) -> Self {
        Self {
            what: String::new(),
            location: std::panic::Location::caller(),
            source: Some(ExceptionSource::Retained(ClosedSource(Some(Box::new(
                Source(Some(cause)),
            ))))),
            tracking: None,
            graph: None,
            scoped: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[derive(Debug)]
    struct Counted {
        formats: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    }
    impl std::fmt::Display for Counted {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.formats.fetch_add(1, Ordering::SeqCst);
            f.write_str("typed retained sentinel")
        }
    }
    impl Error for Counted {}
    impl Drop for Counted {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn retained_rust_exception_keeps_source_without_formatting() {
        let formats = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        assert!(
            Exception::retained_source_control_bytes::<Counted>().unwrap() >= size_of::<Counted>()
        );
        let error = Exception::from_retained_source(Counted {
            formats: formats.clone(),
            drops: drops.clone(),
        });
        assert_eq!(formats.load(Ordering::SeqCst), 0);
        assert!(error.source().unwrap().downcast_ref::<Counted>().is_some());
        assert_eq!(error, error);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        std::thread::spawn(move || drop(error)).join().unwrap();
        assert_eq!(formats.load(Ordering::SeqCst), 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
