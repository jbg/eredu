//! Per-constructor weak request channel, independent of physical bank ownership.
use super::*;
use std::{cell::RefCell,rc::Weak};
use eredu_nn::TensorParallelGroupedOutput;
use eredu_runtime::expert::IndexedInvocationRequest;

/// Actual accepted request source. The implementation owns native parent
/// admission/completion and occurrence order; this trait grants none of them.
pub(crate) trait IndexedRequestSource {
    fn funding(&self)->&HostMetadataFunding;
    fn enter_local(&self, bank: &IndexedBankSource,
        declaration: eredu_nn::workspace::WorkspaceAddressableRegionView<'_>, rows: usize) -> Result<(), Error>;
    fn complete_local(&self, completed: bool) -> Result<(), Error>;
    fn abort_local(&self);
    fn with_region(&self,bank:&IndexedBankSource,request:IndexedInvocationRequest<'_,MlxTensor>,stream:&Stream,
        run:&mut dyn FnMut(OriginalIndexedResidencyFactory)->Result<TensorParallelGroupedOutput<MlxTensor>,Error>)
        ->Result<TensorParallelGroupedOutput<MlxTensor>,Error>;
}
struct Activation {active:Cell<bool>}
struct Binding {
    owner:Weak<dyn IndexedRequestSource>,
    activation:Weak<Activation>,
    funding:HostMetadataFunding,
}
#[derive(Default)]
pub(super) struct Channel {binding:RefCell<Option<Binding>>}
impl Channel {
    pub(super) fn control_bytes()->Option<usize> {
        Layout::new::<[usize;2]>().extend(Layout::new::<Self>()).ok().map(|(layout,_)|layout.pad_to_align().size())
    }
}
/// The exact lexical request installation. No strong request owner is retained.
pub(crate) struct IndexedRequestInstallation {
    source:IndexedBankSource,
    activation:Rc<Activation>,
    funding:HostMetadataFunding,
}
impl Drop for IndexedRequestInstallation {
    fn drop(&mut self) {
        self.activation.active.set(false);
        if let Ok(mut slot)=self.source.request.binding.try_borrow_mut() {
            let same=slot.as_ref().and_then(|binding|binding.activation.upgrade())
                .is_some_and(|active|Rc::ptr_eq(&active,&self.activation));
            if same {slot.take();}
        }
    }
}
impl IndexedRequestInstallation {
    pub(crate) fn control_bytes()->Option<usize> {
        let frames=[size_of::<Self>(),size_of::<Binding>(),size_of::<Activation>(),
            size_of::<Option<Binding>>(),size_of::<Weak<dyn IndexedRequestSource>>(),
            size_of::<std::cell::RefMut<'_,Option<Binding>>>(),size_of::<Result<Self,Error>>(),
            Layout::new::<[usize;2]>().extend(Layout::new::<Activation>()).ok()?.0.pad_to_align().size()];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
}
/// The existing channel remains borrowed for the entire actual local callback.
/// A failed callback or unwind fences its accepted source without refunding it.
pub(crate) struct IndexedLocalRequest<'a> {
    _slot: std::cell::Ref<'a, Option<Binding>>,
    owner: Rc<dyn IndexedRequestSource>,
    activation: Rc<Activation>,
    finished: bool,
}
impl IndexedLocalRequest<'_> {
    pub(crate) fn finish(mut self, completed: bool) -> Result<(), Error> {
        if !self.activation.active.get() {
            self.owner.abort_local();
            self.finished = true;
            return Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));
        }
        let result = self.owner.complete_local(completed);
        self.finished = true;
        result
    }
}
impl Drop for IndexedLocalRequest<'_> {
    fn drop(&mut self) {
        if !self.finished { self.owner.abort_local(); }
    }
}
impl IndexedBankSource {
    /// Descriptive identity includes the actual scoped movement channel.
    pub(crate) fn same_binding(&self,other:&Self)->bool {
        Rc::ptr_eq(&self.request,&other.request)
            && Arc::ptr_eq(&self.bank.inner,&other.bank.inner)&&self.bank.scope==other.bank.scope
    }
    /// Installs an explicitly supplied accepted request for this actual movement.
    /// The caller retains the provider through native completion or recovery.
    pub(crate) fn install_request(&self,owner:&Rc<dyn IndexedRequestSource>)
        ->Result<IndexedRequestInstallation,Error> {
        let funding=owner.funding();let fail=|cause|failed(cause,&self.bank,funding,None);
        funding.reserve_metadata(IndexedRequestInstallation::control_bytes().ok_or_else(||fail(Cause::Overflow))?)
            .map_err(|cause|fail(Cause::Funding(cause)))?;
        let mut slot=self.request.binding.try_borrow_mut().map_err(|_|fail(Cause::Spent))?;
        // A dead weak owner is still a spent installation until its lexical
        // guard closes; it cannot silently become another request epoch.
        if slot.is_some(){return Err(fail(Cause::Spent));}
        let activation=Rc::new(Activation{active:Cell::new(true)});
        *slot=Some(Binding{owner:Rc::downgrade(owner),activation:Rc::downgrade(&activation),funding:funding.clone()});
        Ok(IndexedRequestInstallation{source:self.clone(),activation,funding:funding.clone()})
    }
    /// The actual borrowed request-channel worker, including the explicitly
    /// supplied callback object layout. This describes controls, never a grant.
    pub(crate) fn request_region_control_bytes(callback_bytes:usize)->Option<usize> {
        let frames=[callback_bytes,size_of::<std::cell::Ref<'_,Option<Binding>>>(),size_of::<Rc<dyn IndexedRequestSource>>(),
            size_of::<Rc<Activation>>(),size_of::<IndexedInvocationRequest<'_,MlxTensor>>(),
            size_of::<Result<TensorParallelGroupedOutput<MlxTensor>,Error>>(),
            size_of::<(&Self,&Stream,&mut dyn FnMut(OriginalIndexedResidencyFactory)
                ->Result<TensorParallelGroupedOutput<MlxTensor>,Error>)>()];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
    pub(crate) fn local_request_control_bytes() -> Option<usize> {
        let frames = [size_of::<std::cell::Ref<'_, Option<Binding>>>(),
            size_of::<Rc<dyn IndexedRequestSource>>(), size_of::<Rc<Activation>>(),
            size_of::<eredu_nn::workspace::WorkspaceAddressableRegionView<'_>>(),
            size_of::<(&Self, usize)>(), size_of::<Result<(), Error>>(),
            size_of::<IndexedLocalRequest<'_>>(), size_of::<Option<IndexedLocalRequest<'_>>>(),
            size_of::<Result<IndexedLocalRequest<'_>, Error>>(),
            size_of::<(IndexedLocalRequest<'_>, bool, Result<(), Error>)>(),
            size_of::<&mut IndexedLocalRequest<'_>>()];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Completed expert rows select the authentic installed occurrence. The
    /// returned lexical loan spans the caller's existing typed callback/result;
    /// the channel neither owns nor erases that generic storage.
    pub(crate) fn enter_local_request(&self,
        declaration: eredu_nn::workspace::WorkspaceAddressableRegionView<'_>, rows: usize)
        -> Result<IndexedLocalRequest<'_>, Error> {
        let slot = self.request.binding.try_borrow().map_err(|_| Error::OriginalSourceContract {
            stage: "expert local request channel is mutably borrowed",
            cause: eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        })?;
        let binding = slot.as_ref().ok_or(Error::OriginalSourceContract {
            stage: "expert local request channel is absent",
            cause: eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        })?;
        let fail = |cause| failed(cause, &self.bank, &binding.funding, None);
        binding.funding.reserve_metadata(Self::local_request_control_bytes()
            .ok_or_else(|| fail(Cause::Overflow))?).map_err(|cause| fail(Cause::Funding(cause)))?;
        let active = binding.activation.upgrade().ok_or_else(|| fail(Cause::Spent))?;
        let owner = binding.owner.upgrade().ok_or_else(|| fail(Cause::Spent))?;
        if !active.active.get() || !owner.funding().same_account(&binding.funding) { return Err(fail(Cause::Identity)); }
        owner.enter_local(self, declaration, rows)?;
        Ok(IndexedLocalRequest { _slot: slot, owner, activation: active, finished: false })
    }
    /// None means no installed request; an expired installed request refuses.
    /// The immutable channel borrow remains held through the lexical callback,
    /// so its installation cannot be replaced during native work.
    pub(crate) fn with_request_region(&self,request:IndexedInvocationRequest<'_,MlxTensor>,stream:&Stream,
        run:&mut dyn FnMut(OriginalIndexedResidencyFactory)->Result<TensorParallelGroupedOutput<MlxTensor>,Error>)
        ->Option<Result<TensorParallelGroupedOutput<MlxTensor>,Error>> {
        let slot=match self.request.binding.try_borrow() {
            Ok(slot)=>slot,
            Err(_)=>return Some(Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))),
        };
        let binding=slot.as_ref()?;
        let fail=|cause|failed(cause,&self.bank,&binding.funding,None);
        Some((|| {
            binding.funding.reserve_metadata(Self::request_region_control_bytes(size_of_val(run))
                .ok_or_else(||fail(Cause::Overflow))?).map_err(|cause|fail(Cause::Funding(cause)))?;
            let active=binding.activation.upgrade().ok_or_else(||fail(Cause::Spent))?;
            let owner=binding.owner.upgrade().ok_or_else(||fail(Cause::Spent))?;
            if !active.active.get()||!owner.funding().same_account(&binding.funding){return Err(fail(Cause::Identity));}
            owner.with_region(self,request,stream,run)
        })())
    }
}
