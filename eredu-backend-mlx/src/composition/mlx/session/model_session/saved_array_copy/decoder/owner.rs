//! Closed active aliases; semantic payload destruction stays in ordinary retirement.
use super::{CopiedTextComponents, FrozenDecoder};
use crate::backend::ordinary_retirement::OrdinaryRetirement;
use std::{ops::Deref, rc::Rc};

// No raw Rc/Weak or owning OrdinaryRetirement escapes this module.
pub(super) struct FrozenDecoderOwner {
    inner: Option<Rc<OrdinaryRetirement<FrozenDecoder>>>,
}
impl FrozenDecoderOwner {
    pub(super) fn new(value: FrozenDecoder) -> Self {
        Self {
            inner: Some(Rc::new(OrdinaryRetirement::new(value))),
        }
    }
}
impl Clone for FrozenDecoderOwner {
    fn clone(&self) -> Self {
        Self {
            inner: Some(self.inner.as_ref().expect("live payload owner").clone()),
        }
    }
}
impl Deref for FrozenDecoderOwner {
    type Target = FrozenDecoder;
    fn deref(&self) -> &FrozenDecoder {
        self.inner.as_deref().expect("live payload owner")
    }
}

// With no exported Weak, Rc::into_inner retires the last Rc block before this
// helper returns the unchanged ordinary owner. Do not extract its semantic T.
fn retire_active(
    owner: Rc<OrdinaryRetirement<FrozenDecoder>>,
) -> Option<OrdinaryRetirement<FrozenDecoder>> {
    Rc::into_inner(owner)
}
impl Drop for FrozenDecoderOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.inner.take() {
            let pending = retire_active(owner);
            // Only queues the existing Box. No callback, wait or eager reclaim.
            drop(pending);
        }
    }
}

/// Closed aliases of the returned decoder/sampler pair. No raw Rc or Weak is
/// exported, so the final shell retires before payloads can release preparation.
pub(in crate::composition::mlx::session) struct CopiedTextComponentsOwner {
    inner: Option<Rc<CopiedTextComponents>>,
}
impl CopiedTextComponentsOwner {
    pub(in crate::composition::mlx::session) fn new(value: CopiedTextComponents) -> Self {
        Self {
            inner: Some(Rc::new(value)),
        }
    }
}
impl Clone for CopiedTextComponentsOwner {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
impl Deref for CopiedTextComponentsOwner {
    type Target = CopiedTextComponents;
    fn deref(&self) -> &Self::Target {
        self.inner.as_deref().expect("live saved pair")
    }
}
impl Drop for CopiedTextComponentsOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.inner.take() {
            let value = Rc::into_inner(owner);
            drop(value);
        }
    }
}

fn rc_bytes<T>() -> Option<usize> {
    std::alloc::Layout::new::<[std::cell::Cell<usize>; 2]>()
        .extend(std::alloc::Layout::new::<T>())
        .ok()
        .map(|(layout, _)| layout.pad_to_align().size())
}

/// The two actual returned owner blocks and the existing one-node deferred Drop.
/// Source/table/sampler/array payloads remain in their separately admitted owners.
pub(super) fn returned_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        rc_bytes::<OrdinaryRetirement<FrozenDecoder>>()?,
        usize::try_from(OrdinaryRetirement::<FrozenDecoder>::control_bytes()?).ok()?,
        rc_bytes::<CopiedTextComponents>()?,
        size_of::<FrozenDecoder>(),
        size_of::<FrozenDecoderOwner>(),
        size_of::<Option<OrdinaryRetirement<FrozenDecoder>>>(),
        size_of::<Option<Rc<OrdinaryRetirement<FrozenDecoder>>>>(),
        size_of::<CopiedTextComponents>(),
        size_of::<CopiedTextComponentsOwner>(),
        size_of::<Option<CopiedTextComponents>>(),
        size_of::<Option<Rc<CopiedTextComponents>>>(),
        size_of::<super::super::TextArraySource>(), // actual provenance return
        size_of::<super::super::CopiedTextSampling>(),
        size_of::<super::super::CopiedTextArrays>(),
        size_of::<eredu_runtime::replicated_session::ReplicatedTextControlOrigin>(),
        size_of::<Option<eredu_runtime::SharedPreparedInputCacheIdentity>>(),
        size_of::<Option<eredu_core::HostPreparationAuthority>>(),
        crate::composition::mlx::session::text_snapshot::MlxSavedTextComponents::funded_control_bytes()?,
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)?;
    Some(bytes)
}
