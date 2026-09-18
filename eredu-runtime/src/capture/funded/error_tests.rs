use super::FundedCaptureError;
use std::{error::Error, sync::{Arc, atomic::{AtomicUsize, Ordering}}};

#[derive(Debug, thiserror::Error)]
#[error("native capture fault")]
struct NativeFault(Arc<AtomicUsize>);
impl Drop for NativeFault {
    fn drop(&mut self) { self.0.fetch_add(1, Ordering::SeqCst); }
}

#[test]
fn public_backend_failure_exposes_and_retains_original_capture_leaf() {
    let retired = Arc::new(AtomicUsize::new(0));
    let captured = FundedCaptureError::Backend(NativeFault(retired.clone()));
    assert_eq!(captured.to_string(), "native capture fault");
    let failure = eredu_core::BackendFailure::from_error(captured);
    let mut source: Option<&(dyn Error + 'static)> = Some(&failure);
    let mut leaves = 0;
    while let Some(error) = source {
        if let Some(leaf) = error.downcast_ref::<NativeFault>() {
            assert!(Arc::ptr_eq(&leaf.0, &retired));
            leaves += 1;
        }
        source = error.source();
    }
    assert_eq!(leaves, 1, "the actual native leaf must survive neutral erasure");
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(failure);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}
