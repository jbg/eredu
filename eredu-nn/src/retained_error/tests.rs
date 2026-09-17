use super::*;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Barrier,
};
#[derive(Debug)]
struct Cause {
    displays: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    data: Vec<u8>,
    retired: Option<Arc<AtomicBool>>,
}
impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.displays.fetch_add(1, Ordering::SeqCst);
        f.write_str("actual retained source")
    }
}
impl std::error::Error for Cause {}
impl Drop for Cause {
    fn drop(&mut self) {
        if let Some(retired) = &self.retired {
            assert!(retired.load(Ordering::SeqCst));
        }
        assert_eq!(self.data, [7, 11, 19]);
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn retained_neural_clone_shares_complete_source_without_formatting_or_string_copy() {
    let displays = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let error = Error::backend_retained_source(Cause {
        displays: displays.clone(),
        drops: drops.clone(),
        data: vec![7, 11, 19],
        retired: None,
    });
    let alias = error.clone();
    assert_eq!(displays.load(Ordering::SeqCst), 0);
    let a = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<Cause>()
        .unwrap();
    let b = std::error::Error::source(&alias)
        .unwrap()
        .downcast_ref::<Cause>()
        .unwrap();
    assert!(std::ptr::eq(a, b));
    assert_eq!(alias.to_string(), "actual retained source");
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(alias);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let block = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
        .extend(std::alloc::Layout::new::<Inner>())
        .unwrap()
        .0
        .pad_to_align();
    assert_eq!(
        Error::retained_source_control_bytes::<Cause>(),
        Some(block.size() + std::mem::size_of::<Cause>())
    );
    let legacy = Error::backend_source(Cause {
        displays: displays.clone(),
        drops: drops.clone(),
        data: vec![7, 11, 19],
        retired: None,
    });
    assert_eq!(displays.load(Ordering::SeqCst), 2);
    let copy = legacy.clone();
    assert_eq!(displays.load(Ordering::SeqCst), 2);
    assert_eq!(copy.to_string(), legacy.to_string());
    drop((legacy, copy));
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}
#[test]
fn concurrent_neural_aliases_remove_control_before_source_destructor() {
    let drops = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicBool::new(false));
    let cause = Cause {
        displays: Arc::new(AtomicUsize::new(0)),
        drops: drops.clone(),
        data: vec![7, 11, 19],
        retired: Some(retired.clone()),
    };
    let error = Error {
        storage: ErrorStorage::Retained(RetainedSource(Some(Arc::new(Inner {
            source: SourceOwner(Some(Box::new(cause))),
            retired_control: Some(retired),
        })))),
    };
    let gate = Arc::new(Barrier::new(3));
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let error = error.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                gate.wait();
                drop(error);
            })
        })
        .collect();
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    gate.wait();
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn unwinding_source_drop_releases_custody_after_control_without_a_state_loan() {
    #[derive(Debug)]
    struct Host {
        retired: Arc<AtomicBool>,
        drops: Arc<AtomicUsize>,
        inspection: Arc<std::sync::Mutex<()>>,
    }
    impl Drop for Host {
        fn drop(&mut self) {
            assert!(self.retired.load(Ordering::SeqCst));
            let _inspection = self
                .inspection
                .try_lock()
                .expect("no state loan at retirement");
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[derive(Debug)]
    struct PanickingSource {
        _host: Host,
    }
    impl fmt::Display for PanickingSource {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("source destructor")
        }
    }
    impl std::error::Error for PanickingSource {}
    impl Drop for PanickingSource {
        fn drop(&mut self) {
            panic!("injected source retirement panic");
        }
    }
    let retired = Arc::new(AtomicBool::new(false));
    let drops = Arc::new(AtomicUsize::new(0));
    let source = PanickingSource {
        _host: Host {
            retired: retired.clone(),
            drops: drops.clone(),
            inspection: Arc::new(std::sync::Mutex::new(())),
        },
    };
    let error = Error {
        storage: ErrorStorage::Retained(RetainedSource(Some(Arc::new(Inner {
            source: SourceOwner(Some(Box::new(source))),
            retired_control: Some(retired),
        })))),
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(error)));
    assert!(result.is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
