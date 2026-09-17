//! The shared callback registry with source-paid, one-use registration nodes.
use std::{alloc::Layout,cell::{Cell,RefCell},mem::{size_of,size_of_val},rc::Rc};

/// Fixed refusal before registering a prepared callback.
#[derive(Clone,Copy,Debug,Eq,PartialEq,thiserror::Error)]
pub enum HousekeepingRegistrationCause {
    /// No registry mutation occurred.
    #[error("runtime housekeeping registry is currently borrowed")]
    Busy,
    /// The submitting thread has begun registry teardown.
    #[error("runtime housekeeping registry is no longer available on this thread")]
    ThreadUnavailable,
    /// The finite generation counter cannot admit another registration.
    #[error("runtime housekeeping generation overflow")]
    GenerationOverflow,
}
/// Refusal with the exact unchanged preparation and its source custody.
#[derive(Debug)]
pub struct HousekeepingRegistrationFailure<P> { cause:HousekeepingRegistrationCause, prepared:P }
impl<P> HousekeepingRegistrationFailure<P> {
    /// Return the fixed cause and the original owned preparation.
    pub fn into_parts(self)->(HousekeepingRegistrationCause,P) { (self.cause,self.prepared) }
}
struct Fields {
    callback:fn(), ordinary:bool,
    born:Cell<Option<u64>>, retired:Cell<Option<u64>>,
    next:RefCell<Option<Reference>>,
}
trait ErasedNode {
    fn fields(&self)->&Fields;
    fn release(self:Rc<Self>);
}
struct Node<C> { fields:Fields, _custody:C }
impl<C:'static> ErasedNode for Node<C> {
    fn fields(&self)->&Fields { &self.fields }
    fn release(self:Rc<Self>) {
        // All aliases take this route. The final actual Rc allocation retires
        // before the moved value, and therefore before account custody.
        if let Some(value)=Rc::into_inner(self) { drop(value); }
    }
}
struct Reference(Option<Rc<dyn ErasedNode>>);
impl Reference {
    fn new<C:'static>(callback:fn(),ordinary:bool,custody:C)->Self {
        Self(Some(Rc::new(Node { fields:Fields { callback,ordinary,
            born:Cell::new(None),retired:Cell::new(None),next:RefCell::new(None) },_custody:custody })))
    }
    fn fields(&self)->&Fields { self.0.as_ref().expect("live callback node").fields() }
    fn active(&self,epoch:u64)->bool {
        self.fields().born.get().is_some_and(|born|born<=epoch)
            && !self.fields().retired.get().is_some_and(|retired|retired<=epoch)
    }
    fn same_node(&self,other:&Self)->bool { Rc::ptr_eq(self.0.as_ref().unwrap(),other.0.as_ref().unwrap()) }
    fn same_callback(&self,callback:fn())->bool { std::ptr::fn_addr_eq(self.fields().callback,callback) }
    fn next(&self)->Option<Self> { self.fields().next.borrow().clone() }
}
impl Clone for Reference { fn clone(&self)->Self { Self(self.0.clone()) } }
impl Drop for Reference { fn drop(&mut self) { if let Some(owner)=self.0.take() { owner.release(); } } }
impl std::fmt::Debug for Reference {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result { f.debug_struct("HousekeepingNode").finish_non_exhaustive() }
}
#[derive(Default)]
struct Chain(Option<Reference>);
impl Drop for Chain {
    fn drop(&mut self) {
        while let Some(node)=self.0.take() {
            self.0=node.fields().next.borrow_mut().take();
            drop(node);
        }
    }
}
#[derive(Default)]
struct Registry { head:Chain, epoch:u64 }
impl Registry {
    fn advance(&mut self)->Result<u64,HousekeepingRegistrationCause> {
        self.epoch=self.epoch.checked_add(1).ok_or(HousekeepingRegistrationCause::GenerationOverflow)?;
        Ok(self.epoch)
    }
    fn install(&mut self,node:&Reference)->Result<(),HousekeepingRegistrationCause> {
        let epoch=self.advance()?;
        node.fields().born.set(Some(epoch));
        let mut tail=self.head.0.clone();
        while let Some(current)=tail {
            if let Some(next)=current.next() { tail=Some(next); }
            else { *current.fields().next.borrow_mut()=Some(node.clone()); return Ok(()); }
        }
        self.head.0=Some(node.clone()); Ok(())
    }
    /// Call only outside an active callback snapshot. Retired accounts drop
    /// after the caller releases the registry RefMut.
    fn prune(&mut self)->Chain {
        let mut current=self.head.0.take();
        let mut kept=None;
        let mut tail:Option<Reference>=None;
        let mut removed=Chain::default();
        while let Some(node)=current {
            current=node.fields().next.borrow_mut().take();
            if node.active(self.epoch) {
                if let Some(previous)=&tail { *previous.fields().next.borrow_mut()=Some(node.clone()); }
                else { kept=Some(node.clone()); }
                tail=Some(node);
            } else {
                *node.fields().next.borrow_mut()=removed.0.take();
                removed.0=Some(node);
            }
        }
        self.head.0=kept; removed
    }
}
thread_local! { static REGISTRY:RefCell<Registry> = RefCell::new(Registry::default()); }
fn with_registry<R>(body:impl FnOnce(&mut Registry)->R)->Result<R,HousekeepingRegistrationCause> {
    REGISTRY.try_with(|registry|registry.try_borrow_mut().map(|mut value|body(&mut value))
        .map_err(|_|HousekeepingRegistrationCause::Busy)).map_err(|_|HousekeepingRegistrationCause::ThreadUnavailable)?
}
fn prune_if_idle() {
    if !super::RUNNING_HOUSEKEEPING.try_with(Cell::get).unwrap_or(true) {
        if let Ok(removed)=with_registry(Registry::prune) { drop(removed); }
    }
}

fn node_allocation_bytes<C>() -> Option<usize> {
    // RcInner places its strong/weak Cell<usize> header before the actual T.
    // The payload may require stricter alignment than either counter.
    let (layout, _) = Layout::new::<[usize; 2]>().extend(Layout::new::<Node<C>>()).ok()?;
    Some(layout.pad_to_align().size())
}

/// A final callback node allocated before native submission. `C` is retained
/// accounting/source custody; it must not contain unresolved native resources.
/// The caller must reserve control_bytes on that same custody before new().
pub struct PreparedThreadRuntimeHousekeeping<C:'static> {
    node:Reference,
    _custody:std::marker::PhantomData<C>,
}
impl<C:'static> std::fmt::Debug for PreparedThreadRuntimeHousekeeping<C> {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result { f.debug_struct("PreparedThreadRuntimeHousekeeping").finish_non_exhaustive() }
}
impl<C:'static> PreparedThreadRuntimeHousekeeping<C> {
    /// Actual node allocation and named constructor/registration/retirement controls.
    pub fn control_bytes()->Option<usize> {
        let parts=[node_allocation_bytes::<C>()?,size_of::<Node<C>>(),
            size_of::<Self>(),size_of::<RegisteredThreadRuntimeHousekeeping>(),size_of::<Reference>(),
            size_of::<Option<Reference>>(),size_of::<Chain>(),size_of::<Registry>(),size_of::<HousekeepingRegistrationCause>(),
            size_of::<HousekeepingRegistrationFailure<Self>>(),
            size_of::<Result<RegisteredThreadRuntimeHousekeeping,HousekeepingRegistrationFailure<Self>>>(),
            size_of::<std::cell::RefMut<'static,Registry>>(),size_of::<std::cell::BorrowMutError>(),
            size_of::<Result<Chain,HousekeepingRegistrationCause>>(),size_of::<std::thread::AccessError>(),
            size_of::<(&dyn ErasedNode,fn(),u64,bool)>(),size_of::<(Option<Reference>,Option<Reference>)>(),
            size_of::<(Option<Reference>,Option<Reference>)>(),size_of::<(Option<Reference>,u64)>(),
        ];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    /// Allocate the final registration node after its provider has paid the layout.
    pub fn new(callback:fn(),custody:C)->Self {
        Self { node:Reference::new(callback,false,custody),_custody:std::marker::PhantomData }
    }
    /// No allocation, native entry, callback or ordinary-registry fallback.
    /// Failure returns the same still-unregistered prepared node.
    pub fn try_register(self)->Result<RegisteredThreadRuntimeHousekeeping,HousekeepingRegistrationFailure<Self>> {
        match with_registry(|registry|registry.install(&self.node)).and_then(|result|result) {
            Ok(())=>Ok(RegisteredThreadRuntimeHousekeeping { node:self.node }),
            Err(cause)=>Err(HousekeepingRegistrationFailure { cause,prepared:self }),
        }
    }
}
/// Holds exactly one prepared registration until completion or its orphan is
/// retired. Dropping another/ordinary registration cannot revoke this one.
#[derive(Debug)]
pub struct RegisteredThreadRuntimeHousekeeping { node:Reference }
impl Drop for RegisteredThreadRuntimeHousekeeping {
    fn drop(&mut self) {
        let retired=with_registry(|registry|registry.advance().unwrap_or(u64::MAX));
        self.node.fields().retired.set(Some(retired.unwrap_or(0)));
        prune_if_idle();
    }
}

pub(super) fn register(callback:fn()) {
    let exists=with_registry(|registry| {
        let mut at=registry.head.0.clone();
        while let Some(node)=at {
            if node.fields().ordinary && node.active(registry.epoch) && node.same_callback(callback) { return true; }
            at=node.next();
        }
        false
    }).unwrap_or(true);
    if !exists {
        let node=Reference::new(callback,true,());
        let _=with_registry(|registry|registry.install(&node));
    }
    prune_if_idle();
}
pub(super) fn unregister(callback:fn()) {
    let _=with_registry(|registry| {
        let epoch=registry.advance().unwrap_or(u64::MAX);
        let mut at=registry.head.0.clone();
        while let Some(node)=at {
            if node.fields().ordinary && node.same_callback(callback) && node.fields().retired.get().is_none() { node.fields().retired.set(Some(epoch)); }
            at=node.next();
        }
    });
    prune_if_idle();
}
pub(super) fn run() {
    let Ok((head,epoch))=with_registry(|registry|(registry.head.0.clone(),registry.epoch)) else { return; };
    let mut at=head.clone();
    while let Some(node)=at {
        if node.active(epoch) {
            // Multiple live prepared owners of the same idempotent callback
            // preserve one invocation per snapshot, without a scratch set.
            let mut prior=head.clone();
            let mut duplicate=false;
            while let Some(previous)=prior {
                if previous.same_node(&node) { break; }
                if previous.active(epoch) && previous.same_callback(node.fields().callback) { duplicate=true; break; }
                prior=previous.next();
            }
            if !duplicate { (node.fields().callback)(); }
        }
        at=node.next();
    }
    drop(head);
    // The callback walk is over. Unlink tombstones before account destructors,
    // which may themselves register another future callback.
    if let Ok(removed)=with_registry(Registry::prune) { drop(removed); }
}

#[cfg(test)]
mod tests;
