use super::*;
use crate::BackendFailureKind;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering::SeqCst},
};

#[derive(Debug)]
struct Leaf {
    seen: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    cause: std::io::Error,
}
impl fmt::Display for Leaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("concrete leaf")
    }
}
impl Error for Leaf {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.cause)
    }
}
impl Drop for Leaf {
    fn drop(&mut self) {
        self.seen.store(retirement_count(), SeqCst);
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
fn closed_source_preserves_original_allocation_and_typed_chain() {
    let baseline = retirement_count();
    let (leaf, seen, drops) = leaf();
    let boxed = Box::new(leaf);
    let address = std::ptr::from_ref(boxed.as_ref());
    let owner = SourceOwner::from_box(boxed);
    assert!(std::ptr::eq(
        owner.error().downcast_ref::<Leaf>().unwrap(),
        address
    ));
    assert_eq!(
        owner
            .error()
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .kind(),
        std::io::ErrorKind::InvalidData
    );
    assert_eq!(format!("{owner}"), "concrete leaf");
    assert_eq!(retirement_count(), baseline);
    drop(owner);
    assert_eq!(seen.load(SeqCst), baseline + 1);
    assert_eq!(drops.load(SeqCst), 1);
}

#[test]
fn public_error_flattening_preserves_leaf_context_and_one_retirement() {
    let baseline = retirement_count();
    let (leaf, seen, drops) = leaf();
    let error =
        BackendFailure::new(BackendFailureKind::InvalidSession, leaf).with_operation("admission");
    let address = std::ptr::from_ref(error.source().unwrap().downcast_ref::<Leaf>().unwrap());
    let error = BackendFailure::from_error(error);
    assert_eq!(error.kind(), BackendFailureKind::InvalidSession);
    assert_eq!(error.operation(), "admission");
    assert_eq!(
        format!("{error}"),
        "admission failed (InvalidSession): concrete leaf"
    );
    assert!(std::ptr::eq(
        error.source().unwrap().downcast_ref::<Leaf>().unwrap(),
        address
    ));
    assert_eq!(
        retirement_count(),
        baseline,
        "flattening returns the existing source owner"
    );
    drop(error);
    assert_eq!(seen.load(SeqCst), baseline + 1);
    assert_eq!(drops.load(SeqCst), 1);
}

#[test]
fn public_error_classification_and_concrete_sources_stay_compatible() {
    for (kind, expected) in [
        (
            std::io::ErrorKind::OutOfMemory,
            BackendFailureKind::ResourceExhausted,
        ),
        (
            std::io::ErrorKind::InvalidInput,
            BackendFailureKind::InvalidInput,
        ),
        (
            std::io::ErrorKind::Unsupported,
            BackendFailureKind::Unsupported,
        ),
        (std::io::ErrorKind::NotFound, BackendFailureKind::Io),
    ] {
        let error = BackendFailure::from_error(std::io::Error::from(kind));
        assert_eq!(error.kind(), expected);
        assert_eq!(
            error
                .source()
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .kind(),
            kind
        );
    }
    for (cause, expected) in [
        (crate::SessionAuthorityError::Busy, BackendFailureKind::Busy),
        (
            crate::SessionAuthorityError::TicketExhausted,
            BackendFailureKind::ResourceExhausted,
        ),
    ] {
        let error = BackendFailure::from_error(cause);
        assert_eq!(error.kind(), expected);
        assert_eq!(
            error
                .source()
                .unwrap()
                .downcast_ref::<crate::SessionAuthorityError>(),
            Some(&cause)
        );
    }
    let (leaf, _, _) = leaf();
    let error = BackendFailure::from_error(leaf);
    assert_eq!(error.kind(), BackendFailureKind::Other);
    assert!(error.source().unwrap().is::<Leaf>());
}

#[test]
fn concrete_error_disposal_remains_send_sync_and_runs_after_unboxing() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<BackendFailure>();
    let (leaf, seen, drops) = leaf();
    let error = BackendFailure::from_error(leaf);
    let count = std::thread::spawn(move || {
        let baseline = retirement_count();
        assert!(error.source().unwrap().is::<Leaf>());
        drop(error);
        assert_eq!(retirement_count(), baseline + 1);
        baseline + 1
    })
    .join()
    .unwrap();
    assert_eq!(seen.load(SeqCst), count);
    assert_eq!(drops.load(SeqCst), 1);
}

#[test]
fn concrete_error_drop_panic_preserves_payload_after_source_boundary() {
    #[derive(Debug)]
    struct Panicking(Arc<AtomicUsize>);
    impl fmt::Display for Panicking {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("panicking source")
        }
    }
    impl Error for Panicking {}
    impl Drop for Panicking {
        fn drop(&mut self) {
            self.0.store(retirement_count(), SeqCst);
            std::panic::panic_any(73_u64);
        }
    }
    let baseline = retirement_count();
    let seen = Arc::new(AtomicUsize::new(0));
    let error = BackendFailure::from_error(Panicking(seen.clone()));
    let panic =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(error))).unwrap_err();
    assert_eq!(panic.downcast_ref::<u64>(), Some(&73));
    assert_eq!(seen.load(SeqCst), baseline + 1);
    assert_eq!(retirement_count(), baseline + 1);
}

#[test]
fn concrete_error_diagnostics_include_actual_source_padding_without_constructing_it() {
    #[derive(Debug)]
    struct Empty;
    impl fmt::Display for Empty {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("empty")
        }
    }
    impl Error for Empty {}
    #[derive(Debug)]
    #[repr(align(64))]
    struct Padded([u8; 65]);
    impl fmt::Display for Padded {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{} bytes", self.0.len())
        }
    }
    impl Error for Padded {}
    let baseline = retirement_count();
    let empty = BackendFailure::source_retention_peak_bytes::<Empty>().unwrap();
    let padded = BackendFailure::source_retention_peak_bytes::<Padded>().unwrap();
    assert!(empty >= size_of::<BackendFailure>());
    assert_eq!(
        padded - empty,
        3 * (size_of::<Padded>() - size_of::<Empty>()) + size_of::<Option<Padded>>()
            - size_of::<Option<Empty>>()
    );
    assert_eq!(retirement_count(), baseline);
}

#[test]
fn sequence_bank_rejections_reuse_only_fixed_typed_sources() {
    let baseline = retirement_count();
    for (cause, kind) in [
        (
            GenerationSequenceBankRejection::Unavailable,
            BackendFailureKind::InvalidSession,
        ),
        (
            GenerationSequenceBankRejection::Busy,
            BackendFailureKind::Busy,
        ),
        (
            GenerationSequenceBankRejection::IdentityMismatch,
            BackendFailureKind::InvalidSession,
        ),
    ] {
        let errors = std::array::from_fn::<_, 128, _>(|_| cause.into_backend_failure());
        let address = std::ptr::from_ref(
            errors[0]
                .source()
                .unwrap()
                .downcast_ref::<GenerationSequenceBankRejection>()
                .unwrap(),
        );
        for error in &errors {
            assert_eq!(error.kind(), kind);
            assert_eq!(error.operation(), "generation sequence bank");
            let source = error
                .source()
                .unwrap()
                .downcast_ref::<GenerationSequenceBankRejection>()
                .unwrap();
            assert_eq!(*source, cause);
            assert!(std::ptr::eq(source, address));
            assert!(source.source().is_none());
            assert!(matches!(&error.source.0, OwnerKind::SequenceBank(_)));
        }
        drop(errors);
    }
    assert_eq!(
        retirement_count(),
        baseline,
        "no dynamic source owner was created or retired"
    );
}

#[test]
fn fixed_rejection_context_and_dynamic_wrapping_keep_source_semantics() {
    let baseline = retirement_count();
    let fixed = GenerationSequenceBankRejection::Busy
        .into_backend_failure()
        .with_operation("first delivery");
    let source = std::ptr::from_ref(
        fixed
            .source()
            .unwrap()
            .downcast_ref::<GenerationSequenceBankRejection>()
            .unwrap(),
    );
    // This deliberately uses the existing allocating flattening API. The fixed
    // conversion itself does not call it or claim to fund arbitrary wrapping.
    let flattened = BackendFailure::from_error(fixed);
    assert_eq!(flattened.kind(), BackendFailureKind::Busy);
    assert_eq!(flattened.operation(), "first delivery");
    assert!(std::ptr::eq(
        flattened
            .source()
            .unwrap()
            .downcast_ref::<GenerationSequenceBankRejection>()
            .unwrap(),
        source
    ));
    drop(flattened);
    assert_eq!(retirement_count(), baseline);
    let dynamic = BackendFailure::new(
        BackendFailureKind::Other,
        GenerationSequenceBankRejection::Busy,
    );
    assert!(matches!(&dynamic.source.0, OwnerKind::Owned(_)));
    assert_eq!(
        dynamic
            .source()
            .unwrap()
            .downcast_ref::<GenerationSequenceBankRejection>(),
        Some(&GenerationSequenceBankRejection::Busy)
    );
    drop(dynamic);
    assert_eq!(retirement_count(), baseline + 1);
}

#[test]
fn exhausted_host_metadata_retains_exact_inline_source_through_public_conversion() {
    use crate::HostMetadataFundingError as Funding;
    let baseline = retirement_count();
    for cause in [
        Funding::Capacity {
            required: 4097,
            available: 17,
        },
        Funding::Overflow,
        Funding::Unavailable,
    ] {
        let failure = BackendFailure::from_error(cause).with_operation("snapshot");
        assert_eq!(failure.operation(), "snapshot");
        assert_eq!(
            failure.source().unwrap().downcast_ref::<Funding>(),
            Some(&cause)
        );
        let failure = BackendFailure::from_error(failure);
        assert_eq!(
            failure.source().unwrap().downcast_ref::<Funding>(),
            Some(&cause)
        );
        assert_eq!(
            failure.kind(),
            match cause {
                Funding::Unavailable => BackendFailureKind::Other,
                _ => BackendFailureKind::ResourceExhausted,
            }
        );
        drop(failure);
    }
    assert_eq!(
        retirement_count(),
        baseline,
        "inline refusals create no owned source shell"
    );
}
