use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Barrier,
};

#[derive(Debug)]
struct Cause {
    drops: Arc<AtomicUsize>,
    retained: Arc<Vec<u8>>,
}
impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("original retained native cause")
    }
}
impl std::error::Error for Cause {}
impl Drop for Cause {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn closed_observation_errors_preserve_source_custody_and_preservation_across_final_threads() {
    let drops = Arc::new(AtomicUsize::new(0));
    let retained = Arc::new(vec![7, 19, 31]);
    let original = Error::before_model_mutation(Cause {
        drops: drops.clone(),
        retained: retained.clone(),
    });
    let first = OutputObservationFailure::new(original);
    let second = first.retained();
    assert!(first.state_preserved());
    assert!(std::ptr::eq(first.original(), second.original()));
    let nested = std::error::Error::source(first.original())
        .unwrap()
        .downcast_ref::<Cause>()
        .unwrap();
    assert!(Arc::ptr_eq(&nested.retained, &retained));
    drop(retained);
    let wrapped = Error::OutputObservation(first);
    assert!(wrapped.model_state_preserved());
    let first = OutputObservationFailure::new(wrapped); // Moves the same source.
    assert!(std::ptr::eq(first.original(), second.original()));
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        let barrier = &barrier;
        scope.spawn(move || {
            barrier.wait();
            drop(first);
        });
        barrier.wait();
        drop(second);
    });
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
