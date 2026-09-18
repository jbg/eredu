use super::*;
use crate::backend::managed_memory::NativeMemoryOwner;
use eredu_runtime::working_memory::WorkingMemoryPool;
fn exception<'a>(
    mut error: &'a (dyn std::error::Error + 'static),
) -> &'a safemlx::error::Exception {
    loop {
        if let Some(cause) = error.downcast_ref() {
            return cause;
        }
        error = error.source().expect("native source preserved");
    }
}
#[test]
fn native_capture_and_escaped_neural_aliases_preserve_actual_unquoted_source_and_state_fact() {
    for preserved in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        let host = HostPreparationAuthority::retain(owner.unquoted_lease().unwrap());
        let cause = Error::Exception(safemlx::error::Exception::custom(
            "actual native diagnostic",
        ));
        let cause = if preserved {
            Error::before_model_mutation(cause)
        } else {
            cause
        };
        let error = cause.retain_ordinary_capture(Some(host));
        assert_eq!(error.model_state_preserved(), preserved);
        assert_eq!(exception(&error).what(), "actual native diagnostic");
        let neural = crate::composition::neural_observer_error(error);
        let alias = neural.clone();
        drop(owner);
        drop(neural);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        assert_eq!(exception(&alias).what(), "actual native diagnostic");
        let barrier = std::sync::Barrier::new(2);
        let other = alias.clone();
        std::thread::scope(|scope| {
            let barrier = &barrier;
            scope.spawn(move || {
                barrier.wait();
                drop(other);
            });
            barrier.wait();
            drop(alias);
        });
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
    assert!(control_peak_bytes().unwrap() >= std::mem::size_of::<OrdinaryCaptureFailure>());
    assert!(
        eredu_nn::Error::retained_source_construction_bytes::<Error>().unwrap()
            >= std::mem::size_of::<Error>()
    );
}
