//! Closed active aliases; semantic payload destruction stays in ordinary retirement.
use super::{CopiedTextComponents, FrozenDecoder};
use crate::backend::ordinary_retirement::OrdinaryRetirement;
#[cfg(test)]
use std::cell::Cell;
use std::{ops::Deref, rc::Rc};

#[cfg(test)]
struct RetirementProbe {
    retired: Rc<Cell<bool>>,
    _host: Option<eredu_core::HostPreparationAuthority>,
}
#[cfg(test)]
impl RetirementProbe {
    fn retired(&self) -> bool {
        self.retired.get()
    }
}

// No raw Rc/Weak or owning OrdinaryRetirement escapes this module.
pub(super) struct FrozenDecoderOwner {
    inner: Option<Rc<OrdinaryRetirement<FrozenDecoder>>>,
    // Separate from semantic Drop. All clones observe the same final active alias.
    #[cfg(test)]
    active_retired: Option<Rc<Cell<bool>>>,
}
impl FrozenDecoderOwner {
    pub(super) fn new(value: FrozenDecoder) -> Self {
        Self {
            inner: Some(Rc::new(OrdinaryRetirement::new(value))),
            #[cfg(test)]
            active_retired: Some(Rc::new(Cell::new(false))),
        }
    }

    #[cfg(test)]
    pub(super) fn retirement_probe(&self) -> impl Fn() -> bool + 'static {
        let probe = RetirementProbe {
            retired: self.active_retired.as_ref().expect("live probe").clone(),
            _host: self._host_preparation.clone(),
        };
        move || probe.retired()
    }
}
impl Clone for FrozenDecoderOwner {
    fn clone(&self) -> Self {
        Self {
            inner: Some(self.inner.as_ref().expect("live payload owner").clone()),
            #[cfg(test)]
            active_retired: self.active_retired.clone(),
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
            #[cfg(test)]
            {
                let probe = self.active_retired.take().expect("live probe");
                if pending.is_some() {
                    probe.set(true);
                }
                drop(probe);
            }
            // Only queues the existing Box. No callback, wait or eager reclaim.
            drop(pending);
        }
    }
}

/// Closed aliases of the returned decoder/sampler pair. No raw Rc or Weak is
/// exported, so the final shell retires before payloads can release preparation.
pub(in crate::composition::mlx::session) struct CopiedTextComponentsOwner {
    inner: Option<Rc<CopiedTextComponents>>,
    #[cfg(test)]
    active_retired: Option<Rc<Cell<bool>>>,
}
impl CopiedTextComponentsOwner {
    pub(in crate::composition::mlx::session) fn new(value: CopiedTextComponents) -> Self {
        Self {
            inner: Some(Rc::new(value)),
            #[cfg(test)]
            active_retired: Some(Rc::new(Cell::new(false))),
        }
    }
    #[cfg(test)]
    pub(in crate::composition::mlx::session) fn retirement_probe(
        &self,
    ) -> impl Fn() -> bool + 'static {
        let probe = RetirementProbe {
            retired: self.active_retired.as_ref().expect("live probe").clone(),
            _host: self.host_preparation.clone(),
        };
        move || probe.retired()
    }
}
impl Clone for CopiedTextComponentsOwner {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            #[cfg(test)]
            active_retired: self.active_retired.clone(),
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
            #[cfg(test)]
            {
                // This test-only allocation is included in host preparation.
                // Retire the local probe alias before the pair can release it;
                // exported probes retain their own preparation authority.
                let probe = self.active_retired.take().expect("live probe");
                if value.is_some() {
                    probe.set(true);
                }
                drop(probe);
            }
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
    #[cfg(test)]
    let bytes = bytes
        .checked_add(rc_bytes::<Cell<bool>>()?)?
        .checked_add(rc_bytes::<Cell<bool>>()?)?;
    Some(bytes)
}
