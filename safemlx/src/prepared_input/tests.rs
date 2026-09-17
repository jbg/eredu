use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
struct Owner(Arc<AtomicUsize>);
impl Drop for Owner {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
thread_local! {static HOOKS:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};}
fn hook() {
    HOOKS.set(HOOKS.get() + 1);
}
#[test]
fn prepared_leaf_is_completed_and_never_invokes_entry_housekeeping() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    crate::register_thread_runtime_housekeeping(hook);
    let count = Arc::new(AtomicUsize::new(0));
    let values = [0.25f32, -2.0, 7.5, 9.];
    let plan = runtime.f32(&values, &[2, 2]).unwrap();
    let prepared =
        PreparedSubmissionGraphQuota::try_new(plan.metadata_bytes(), Owner(count.clone())).unwrap();
    let arena = PreparedInputArena::try_allocate(prepared).unwrap();
    HOOKS.set(0);
    let leaf = plan.construct(&arena).unwrap();
    assert_eq!(HOOKS.get(), 0);
    let identity = leaf.allocation_info().unwrap();
    assert_eq!(HOOKS.get(), 0);
    assert!(identity.bytes() > 0);
    let alias = leaf.try_clone_array().unwrap();
    assert!(HOOKS.get() > 0);
    crate::unregister_thread_runtime_housekeeping(hook);
    assert_eq!(alias.evaluated().unwrap().as_slice::<f32>(), &values);
    assert_eq!(alias.allocation_info().unwrap(), Some(identity));
    drop(leaf);
    drop(arena);
    crate::reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    drop(alias);
    crate::reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}
#[test]
fn prepared_source_busy_returns_same_original_node_and_leaf_refusal_has_no_fallback() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let values = [13u32, 23];
    let plan = runtime.u32(&values, &[1, 2]).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let prepared =
        PreparedSubmissionGraphQuota::try_new(plan.metadata_bytes(), Owner(count.clone())).unwrap();
    let (entered, received) = std::sync::mpsc::channel::<()>();
    let (release, wait) = std::sync::mpsc::channel::<()>();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        let _ = entered.send(());
        let _ = wait.recv();
    });
    received.recv().unwrap();
    let result = PreparedInputArena::try_allocate(prepared);
    drop(release);
    worker.join().unwrap();
    let error = result.unwrap_err();
    assert_eq!(error.cause(), SubmissionGraphQuotaCause::RuntimeBusy);
    assert_eq!(count.load(Ordering::SeqCst), 0);
    let (_, prepared) = error.into_parts();
    let arena = PreparedInputArena::try_allocate(prepared).unwrap();
    let leaf = plan.construct(&arena).unwrap();
    // Same arena has no second leaf capacity. Its existing values stay live.
    let second = runtime.u32(&values, &[1, 2]).unwrap().construct(&arena);
    assert!(matches!(second, Err(PreparedInputCause::Capacity)));
    let alias = leaf.try_clone_array().unwrap();
    assert_eq!(alias.evaluated().unwrap().as_slice::<u32>(), &values);
    drop(alias);
    drop(leaf);
    drop(arena);
    crate::reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}
#[test]
fn prepared_source_native_retirement_queue_can_be_reclaimed_on_another_rust_thread() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let (ready, arrivals) = std::sync::mpsc::channel::<bool>();
    let mut releases = Vec::new();
    let mut workers = Vec::new();
    for _ in 0..4 {
        let ready = ready.clone();
        let (release, wait) = std::sync::mpsc::channel::<()>();
        releases.push(release);
        workers.push(std::thread::spawn(move || {
            // Explicit ordinary worker initialization precedes the source below.
            let initialized = std::panic::catch_unwind(|| drop(runtime_lock::enter())).is_ok();
            let _ = ready.send(initialized);
            drop(ready);
            let _ = wait.recv();
            if initialized {
                crate::reclaim_allocation_owners();
            }
        }));
    }
    drop(ready);
    let mut initialized = true;
    for _ in 0..4 {
        initialized &= arrivals.recv().unwrap_or(false);
    }
    if !initialized {
        drop(releases);
        for worker in workers {
            let _ = worker.join();
        }
        panic!("ordinary reclaimer initialization failed");
    }
    let values = [-3i32, 17];
    let plan = runtime.i32(&values, &[1, 2]).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let prepared =
        PreparedSubmissionGraphQuota::try_new(plan.metadata_bytes(), Owner(count.clone())).unwrap();
    let arena = PreparedInputArena::try_allocate(prepared).unwrap();
    let leaf = plan.construct(&arena).unwrap();
    let alias = leaf.try_clone_array().unwrap();
    drop(leaf);
    drop(arena);
    drop(alias);
    drop(releases);
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn scalar_sources_preserve_rank_values_and_custody_without_entry_housekeeping() {
    let runtime=PreparedInputRuntime::prepare().unwrap();
    assert!(matches!(runtime.f32(&[],&[]),Err(PreparedInputCause::Invalid)));
    assert!(matches!(runtime.f32(&[1.0,2.0],&[]),Err(PreparedInputCause::Invalid)));
    assert!(matches!(runtime.u32(&[37],&[0]),Err(PreparedInputCause::Invalid)));
    let floating=[-2.25_f32];let token=[37_u32];
    let f=runtime.f32(&floating,&[]).unwrap();
    let u=runtime.u32(&token,&[]).unwrap();
    let array=PreparedInputLeaf::array_layout().unwrap();
    let metadata=f.metadata_bytes()+u.metadata_bytes()+2*array.metadata_bytes();
    let count=Arc::new(AtomicUsize::new(0));
    let arena=PreparedInputArena::try_allocate(
        PreparedSubmissionGraphQuota::try_new(metadata,Owner(count.clone())).unwrap()).unwrap();
    crate::register_thread_runtime_housekeeping(hook);HOOKS.set(0);
    let f=f.construct(&arena).unwrap();let u=u.construct(&arena).unwrap();
    let f_id=f.allocation_info().unwrap();let u_id=u.allocation_info().unwrap();
    let f_alias=f.try_source_array().unwrap();let u_alias=u.try_source_array().unwrap();
    assert_eq!(HOOKS.get(),0);
    crate::unregister_thread_runtime_housekeeping(hook);
    assert_ne!(f_id,u_id);
    assert!(f_alias.shape().is_empty());assert!(u_alias.shape().is_empty());
    assert_eq!(f_alias.size(),1);assert_eq!(u_alias.size(),1);
    drop((f,u,arena,runtime));crate::reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst),0);
    assert_eq!(f_alias.evaluated().unwrap().try_as_slice::<f32>().unwrap(),&floating);
    assert_eq!(u_alias.evaluated().unwrap().try_as_slice::<u32>().unwrap(),&token);
    assert_eq!(f_alias.allocation_info().unwrap(),Some(f_id));
    assert_eq!(u_alias.allocation_info().unwrap(),Some(u_id));
    drop(f_alias);crate::reclaim_allocation_owners();assert_eq!(count.load(Ordering::SeqCst),0);
    drop(u_alias);crate::reclaim_allocation_owners();assert_eq!(count.load(Ordering::SeqCst),1);
}

#[test]
fn boolean_prepared_mask_has_exact_dtype_and_survives_source_handle_retirement() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let flags = [true, false, true, true, false, false];
    assert!(matches!(runtime.boolean(&flags, &[1,5]), Err(PreparedInputCause::Invalid)));
    let plan = runtime.boolean(&flags, &[2,3]).unwrap();
    let wrapper = PreparedInputLeaf::array_layout().unwrap();
    let metadata = plan.metadata_bytes().checked_add(wrapper.metadata_bytes()).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let arena = PreparedInputArena::try_allocate(
        PreparedSubmissionGraphQuota::try_new(metadata, Owner(count.clone())).unwrap()).unwrap();
    crate::register_thread_runtime_housekeeping(hook);
    HOOKS.set(0);
    let leaf = plan.construct(&arena).unwrap();
    let id = leaf.allocation_info().unwrap();
    let value = leaf.try_source_array().unwrap();
    assert_eq!(HOOKS.get(), 0);
    crate::unregister_thread_runtime_housekeeping(hook);
    drop((leaf, arena, runtime));
    crate::reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert_eq!(value.dtype(), crate::Dtype::Bool);
    assert_eq!(value.shape(), &[2,3]);
    assert_eq!(value.allocation_info().unwrap(), Some(id));
    assert_eq!(value.evaluated().unwrap().try_as_slice::<bool>().unwrap(), &flags);
    drop(value);
    crate::reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}
