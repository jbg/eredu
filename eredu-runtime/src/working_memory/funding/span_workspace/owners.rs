//! Closed strong-owner populations: retire the Arc allocation before payloads
//! can release their original accounting hold. No Arc or Weak handle escapes.
use super::{RawSpanHostCustody, SpanHostCustody};
use std::{mem::size_of, ops::Deref, sync::Arc};

#[derive(Debug)]
struct Owner<T>(Option<Arc<T>>);
impl<T> Owner<T> {
    fn new(value: T) -> Self {
        Self(Some(Arc::new(value)))
    }
    fn inner(&self) -> &Arc<T> {
        self.0.as_ref().expect("live custody owner")
    }
}
impl<T> Clone for Owner<T> {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.inner())))
    }
}
impl<T> Deref for Owner<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.inner()
    }
}
impl<T> Drop for Owner<T> {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // Every strong owner uses into_inner, including concurrent exits.
            // Its unique winner receives the payload after the implicit Weak
            // allocation owner retires. No external Weak exists in this module.
            // Payload destruction remains outside the existing Usage loan.
            drop(Arc::into_inner(owner));
        }
    }
}

#[derive(Clone, Debug)]
pub(in crate::working_memory) struct RawSpanHostOwner(Owner<RawSpanHostCustody>);
impl RawSpanHostOwner {
    #[cfg(any(test, feature = "original-custody-test-support"))]
    pub(in crate::working_memory) fn is_sole_owner(&self) -> bool {
        Arc::strong_count(self.0.inner()) == 1
    }
    pub(in crate::working_memory) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.0.inner(), other.0.inner())
    }
    pub(super) fn new(value: RawSpanHostCustody) -> Self {
        Self(Owner::new(value))
    }
}
impl Deref for RawSpanHostOwner {
    type Target = RawSpanHostCustody;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub(in crate::working_memory) struct SpanHostOwner(Owner<SpanHostCustody>);
impl SpanHostOwner {
    pub(super) fn new(value: SpanHostCustody) -> Self {
        Self(Owner::new(value))
    }
    pub(in crate::working_memory) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.0.inner(), other.0.inner())
    }
}
impl Deref for SpanHostOwner {
    type Target = SpanHostCustody;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// The final full payload can retire its raw owner while its own extracted
// Option remains live. Include both named return values and both moved Arc
// arguments. Existing payload and construction terms remain in original P.
pub(in crate::working_memory) fn retirement_control_bytes() -> Option<usize> {
    size_of::<Option<SpanHostCustody>>()
        .checked_add(size_of::<Option<RawSpanHostCustody>>())?
        .checked_add(size_of::<Arc<SpanHostCustody>>())?
        .checked_add(size_of::<Arc<RawSpanHostCustody>>())
}
