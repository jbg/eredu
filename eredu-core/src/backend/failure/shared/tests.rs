use super::*;
use crate::backend::failure::source::shared_retirement_count;
use std::sync::{
    atomic::{AtomicUsize, Ordering::SeqCst},
    Arc, Barrier,
};

#[derive(Debug)]
struct Leaf {
    seen: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    cause: std::io::Error,
}
impl fmt::Display for Leaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("retained leaf")
    }
}
impl Error for Leaf {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.cause)
    }
}
impl Drop for Leaf {
    fn drop(&mut self) {
        self.seen.store(shared_retirement_count(), SeqCst);
        self.drops.fetch_add(1, SeqCst);
    }
}
fn leaf() -> (Leaf, Arc<AtomicUsize>, Arc<AtomicUsize>) {
    let seen = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    (
        Leaf {
            seen: seen.clone(),
            drops: drops.clone(),
            cause: std::io::ErrorKind::InvalidData.into(),
        },
        seen,
        drops,
    )
}

#[test]
fn retained_public_failures_share_one_concrete_source_across_reerasure() {
    let baseline = shared_retirement_count();
    let (leaf, seen, drops) = leaf();
    let shared = SharedBackendFailure::new(BackendFailureKind::Busy, leaf);
    let output = crate::SpeculativeOutputError::Retained(shared.clone());
    let alias = output.clone();
    assert_eq!(output, alias);
    drop((output, alias));
    let address = std::ptr::from_ref(shared.source_error().downcast_ref::<Leaf>().unwrap());
    let errors = std::array::from_fn::<_, 128, _>(|_| {
        BackendFailure::from_error(shared.retained().into_failure().with_operation("poll"))
    });
    drop(shared);
    for error in &errors {
        assert_eq!(error.kind(), BackendFailureKind::Busy);
        assert_eq!(error.operation(), "poll");
        assert!(std::ptr::eq(
            error.source().unwrap().downcast_ref::<Leaf>().unwrap(),
            address
        ));
        assert_eq!(
            error
                .source()
                .unwrap()
                .source()
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .kind(),
            std::io::ErrorKind::InvalidData
        );
    }
    assert_eq!(drops.load(SeqCst), 0);
    drop(errors);
    assert_eq!(drops.load(SeqCst), 1);
    assert_eq!(seen.load(SeqCst), baseline + 1);
}

#[test]
fn concurrent_final_shared_exits_destroy_payload_once_after_allocation_retirement() {
    let (leaf, seen, drops) = leaf();
    let shared = SharedBackendFailure::new(BackendFailureKind::Other, leaf);
    let aliases = std::array::from_fn::<_, 8, _>(|_| shared.retained().into_failure());
    drop(shared);
    let barrier = Arc::new(Barrier::new(8));
    let threads = aliases.map(|error| {
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            assert_eq!(shared_retirement_count(), 0);
            barrier.wait();
            drop(error);
        })
    });
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(drops.load(SeqCst), 1);
    assert_eq!(seen.load(SeqCst), 1);
}

#[test]
fn shared_source_panic_keeps_field_cleanup_after_final_owner_retirement() {
    #[derive(Debug)]
    struct Guard(Arc<AtomicUsize>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.fetch_add(1, SeqCst);
        }
    }
    #[derive(Debug)]
    struct Panicking(Guard);
    impl fmt::Display for Panicking {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("panic")
        }
    }
    impl Error for Panicking {}
    impl Drop for Panicking {
        fn drop(&mut self) {
            assert!(shared_retirement_count() > 0);
            panic!("source drop");
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let shared =
        SharedBackendFailure::new(BackendFailureKind::Other, Panicking(Guard(drops.clone())));
    let error = shared.retained().into_failure();
    drop(shared);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(error))).is_err());
    assert_eq!(drops.load(SeqCst), 1);
}
