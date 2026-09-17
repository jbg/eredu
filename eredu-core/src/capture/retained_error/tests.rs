use super::*;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Barrier,
};
struct Host {
    drops: Arc<AtomicUsize>,
    control: Arc<AtomicBool>,
}
impl Drop for Host {
    fn drop(&mut self) {
        assert!(self.control.load(Ordering::SeqCst));
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
fn owned(cause: CaptureError) -> (CaptureError, Arc<AtomicUsize>) {
    let drops = Arc::new(AtomicUsize::new(0));
    let control = Arc::new(AtomicBool::new(false));
    let host = HostPreparationAuthority::retain(Host {
        drops: drops.clone(),
        control: control.clone(),
    });
    let value = CaptureError::Retained(RetainedCaptureError(Some(Arc::new(Inner {
        cause,
        _host: host,
        retired_control: Some(control),
    }))));
    (value, drops)
}
#[test]
fn retained_capture_error_preserves_typed_equality_display_and_exact_control() {
    use crate::capture::CaptureBudget;
    for raw in [
        CaptureError::Invalid("owned diagnostic".into()),
        CaptureError::Unsupported("operation".into()),
        CaptureError::MissingPath("layer.output".into()),
        CaptureError::Overflow,
        CaptureError::Limit {
            budget: CaptureBudget::Host,
            cumulative: true,
        },
    ] {
        let (retained, drops) = owned(raw.clone());
        assert_eq!(retained, raw);
        assert_eq!(retained.to_string(), raw.to_string());
        let alias = retained
            .clone()
            .retain_ordinary(HostPreparationAuthority::unmanaged());
        assert!(std::ptr::eq(retained.cause(), alias.cause()));
        drop(retained);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(alias);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
    let block = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
        .extend(std::alloc::Layout::new::<Inner>())
        .unwrap()
        .0
        .pad_to_align();
    assert_eq!(
        CaptureError::ordinary_retained_control_bytes(),
        Some(block.size() as u64)
    );
}
#[test]
fn concurrent_capture_error_aliases_retire_control_before_actual_host() {
    let (error, drops) = owned(CaptureError::Invalid("persistent failed prefix".into()));
    let gate = Arc::new(Barrier::new(5));
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let error = error.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                gate.wait();
                assert!(matches!(error.cause(), CaptureError::Invalid(_)));
                drop(error);
            })
        })
        .collect();
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    gate.wait();
    for t in threads {
        t.join().unwrap();
    }
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
