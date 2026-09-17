use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc,
};
#[derive(Debug)]
struct Owner(Arc<AtomicUsize>);
impl Drop for Owner {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn cpu_stream_registration_busy_keeps_actual_node_and_recovers_same_custody() {
    let layout = PreparedCpuStream::<Owner>::layout();
    if std::env::var_os("EREDU_REQUIRE_STREAM_REGISTRATION_QUALIFICATION").is_some() {
        assert!(layout.is_ok(), "{layout:?}");
    }
    let layout = match layout {
        Ok(value) => value,
        Err(StreamRegistrationCause::UnknownLayout) => return,
        Err(error) => panic!("unexpected qualification: {error}"),
    };
    let drops = Arc::new(AtomicUsize::new(0));
    let prepared = PreparedCpuStream::with_layout(layout, Owner(drops.clone())).unwrap();
    let pointer = &**prepared.node.as_ref().unwrap() as *const OwnedNode<Owner>;
    let (entered, wait_entered) = mpsc::channel();
    let (release, wait_release) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _loan = runtime_lock::enter();
        entered.send(()).unwrap();
        // Channel disconnection also releases during assertion unwind.
        let _ = wait_release.recv();
    });
    wait_entered.recv().unwrap();
    let error = prepared.try_initialize().unwrap_err();
    assert_eq!(error.cause(), StreamRegistrationCause::Busy);
    let (_, prepared) = error.into_parts();
    assert_eq!(
        &**prepared.node.as_ref().unwrap() as *const OwnedNode<Owner>,
        pointer
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    release.send(()).unwrap();
    worker.join().unwrap();
    let owner = prepared.into_owner();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(owner);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
