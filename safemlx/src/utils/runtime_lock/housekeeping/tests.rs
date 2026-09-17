use super::*;
thread_local! {
    static CALLS:Cell<usize> = const { Cell::new(0) };
    static TRACE:RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static DURING_CALLBACK:RefCell<Option<RegisteredThreadRuntimeHousekeeping>> = const { RefCell::new(None) };
}
struct Custody(Rc<Cell<usize>>);
impl Drop for Custody { fn drop(&mut self) { self.0.set(self.0.get()+1); } }
fn count() { CALLS.with(|calls|calls.set(calls.get()+1)); drop(DURING_CALLBACK.with(|held|held.borrow_mut().take())); }
fn first() { TRACE.with(|trace|trace.borrow_mut().push(1)); unregister(first); unregister(second); register(third); }
fn second() { TRACE.with(|trace|trace.borrow_mut().push(2)); }
fn third() { TRACE.with(|trace|trace.borrow_mut().push(3)); }
fn run() { super::super::run_housekeeping_hooks(); }

#[test]
fn prepared_housekeeping_returns_busy_owner_and_keeps_independent_custody_until_retirement() {
    let drops=Rc::new(Cell::new(0));
    assert!(PreparedThreadRuntimeHousekeeping::<Custody>::control_bytes().unwrap()>size_of::<Custody>());
    let ready=PreparedThreadRuntimeHousekeeping::new(count,Custody(drops.clone()));
    let ready=REGISTRY.with(|registry| {
        let _borrow=registry.borrow_mut();
        let (cause,ready)=ready.try_register().unwrap_err().into_parts();
        assert_eq!(cause,HousekeepingRegistrationCause::Busy);
        assert!(ready.node.fields().born.get().is_none());
        ready
    });
    assert_eq!(drops.get(),0);
    let first=ready.try_register().unwrap();
    let second=PreparedThreadRuntimeHousekeeping::new(count,Custody(drops.clone())).try_register().unwrap();
    register(count); unregister(count);
    CALLS.with(|calls|calls.set(0));
    DURING_CALLBACK.with(|held|*held.borrow_mut()=Some(first));
    run();
    assert_eq!(CALLS.with(Cell::get),1,"live registrations share one callback per stable snapshot");
    assert_eq!(drops.get(),1,"retired callback custody follows snapshot unlinking");
    run(); assert_eq!(CALLS.with(Cell::get),2,"ordinary unregister cannot revoke a prepared registration");
    drop(second); assert_eq!(drops.get(),2);
    run(); assert_eq!(CALLS.with(Cell::get),2);
    #[repr(align(64))]
    struct Aligned(Custody);
    let ready=PreparedThreadRuntimeHousekeeping::new(count,Aligned(Custody(drops.clone())));
    let alignment=std::mem::align_of::<Node<Aligned>>();
    let address=Rc::as_ptr(ready.node.0.as_ref().unwrap()) as *const () as usize;
    assert_eq!(address % alignment,0);
    let header=2*size_of::<usize>();
    let padded_header=header.div_ceil(alignment)*alignment;
    let allocation=(padded_header+size_of::<Node<Aligned>>()).div_ceil(alignment)*alignment;
    assert_eq!(node_allocation_bytes::<Aligned>(),Some(allocation));
    assert!(allocation>header+size_of::<Node<Aligned>>(),"aligned custody really requires header padding");
    assert!(PreparedThreadRuntimeHousekeeping::<Aligned>::control_bytes().unwrap()>=allocation);
    drop(ready); assert_eq!(drops.get(),3);
}

#[test]
fn shared_housekeeping_snapshot_keeps_removed_callbacks_and_defers_new_callbacks() {
    TRACE.with(|trace|*trace.borrow_mut()=Vec::with_capacity(3));
    register(first); register(second);
    run();
    TRACE.with(|trace|assert_eq!(&*trace.borrow(),&[1,2]));
    run();
    TRACE.with(|trace|assert_eq!(&*trace.borrow(),&[1,2,3]));
    unregister(third);
}
