use super::*;
use std::sync::{
    Barrier,
    atomic::{AtomicBool, AtomicUsize, Ordering},
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
        Error::retained_source_construction_bytes::<Cause>(),
        Some(
            block.size()
                + std::mem::size_of::<Cause>()
                + std::mem::size_of::<(
                    Cause,
                    Box<Cause>,
                    Box<dyn Source>,
                    Option<Box<dyn Source>>,
                    SourceOwner,
                    Inner,
                    Arc<Inner>,
                    Option<Arc<Inner>>,
                    RetainedSource,
                    ErrorStorage,
                    Error
                )>()
        )
    );
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

#[test]
fn caller_owned_message_moves_its_buffer_without_a_synthetic_source() {
    let message = String::from("already produced diagnostic");
    let original = message.as_ptr();
    let error = Error::backend_message(message);
    let ErrorStorage::Message(message) = &error.storage else {
        panic!("owned diagnostic must remain a message");
    };
    assert_eq!(message.as_ptr(), original);
    assert!(std::error::Error::source(&error).is_none());
    assert_eq!(error.to_string(), "already produced diagnostic");
    let alias = error.clone();
    drop(error);
    assert_eq!(alias.to_string(), "already produced diagnostic");
    assert!(std::error::Error::source(&alias).is_none());
}

#[test]
fn canonical_typed_source_reserves_exact_owner_and_keeps_original_payer_until_last_alias() {
    use crate::workspace::{
        HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError, WorkspaceContext,
        WorkspaceMetadataError, WorkspaceMetadataAllocation,
    };
    use std::sync::Mutex;
    #[derive(Debug)]
    struct State {
        remaining: Mutex<usize>,
        requests: Mutex<Vec<usize>>,
        retired: AtomicBool,
        source_drops: AtomicUsize,
    }
    #[derive(Debug)]
    struct Account(Arc<State>);
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
            self.0.requests.lock().unwrap().push(bytes);
            let mut remaining = self.0.remaining.lock().unwrap();
            *remaining =
                remaining
                    .checked_sub(bytes)
                    .ok_or(HostMetadataFundingError::Capacity {
                        required: bytes as u64,
                        available: *remaining as u64,
                    })?;
            Ok(())
        }
    }
    impl Drop for Account {
        fn drop(&mut self) {
            self.0.retired.store(true, Ordering::SeqCst);
        }
    }
    #[derive(Debug)]
    struct FundedCause {
        state: Arc<State>,
        _payer: HostMetadataFunding,
    }
    impl fmt::Display for FundedCause {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("original funded cause")
        }
    }
    impl std::error::Error for FundedCause {}
    impl Drop for FundedCause {
        fn drop(&mut self) {
            assert!(!self.state.retired.load(Ordering::SeqCst));
            self.state.source_drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    let required = WorkspaceContext::metadata_source_bytes::<FundedCause>().unwrap();
    for available in [required - 1, required] {
        let state = Arc::new(State {
            remaining: Mutex::new(usize::MAX),
            requests: Mutex::new(Vec::new()),
            retired: AtomicBool::new(false),
            source_drops: AtomicUsize::new(0),
        });
        let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
        state.requests.lock().unwrap().clear();
        *state.remaining.lock().unwrap() = available;
        let error = funding.metadata_source(FundedCause {
            state: state.clone(),
            _payer: funding.clone(),
        });
        assert_eq!(*state.requests.lock().unwrap(), [required]);
        if available < required {
            assert!(matches!(error.storage, ErrorStorage::WorkspaceMetadata(
                WorkspaceMetadataError::Funding(HostMetadataFundingError::Capacity {
                    required: needed, available: left,
                })) if needed == required as u64 && left == available as u64));
            assert_eq!(state.source_drops.load(Ordering::SeqCst), 1);
            drop(error);
            drop(funding);
        } else {
            assert!(
                std::error::Error::source(&error)
                    .unwrap()
                    .is::<FundedCause>()
            );
            let alias = error.clone();
            assert_eq!(*state.requests.lock().unwrap(), [required]);
            drop(funding);
            drop(error);
            assert!(!state.retired.load(Ordering::SeqCst));
            assert_eq!(state.source_drops.load(Ordering::SeqCst), 0);
            assert_eq!(alias.to_string(), "original funded cause");
            drop(alias);
        }
        assert_eq!(state.source_drops.load(Ordering::SeqCst), 1);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}
