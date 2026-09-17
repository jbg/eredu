use super::*;
use crate::gguf_store::{
    GgufRawStorage, GgufRawStorageProvider, GgufRawStorageRequest, PreparedGgufSuppliedFailure,
};

#[derive(Debug)]
struct Storage {
    bytes: Vec<u8>,
    retired: Arc<std::sync::atomic::AtomicUsize>,
}
impl AsRef<[u8]> for Storage {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}
impl AsMut<[u8]> for Storage {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}
impl GgufRawStorage for Storage {
    fn capacity(&self) -> usize {
        self.bytes.capacity()
    }
}
impl Drop for Storage {
    fn drop(&mut self) {
        // Independent test custody observes the actual final byte allocation
        // release, rather than treating a provider return as retirement.
        drop(std::mem::take(&mut self.bytes));
        self.retired.fetch_add(1, Ordering::SeqCst);
    }
}
#[derive(Debug, thiserror::Error)]
#[error("test provider retains supplied destination")]
struct Refusal(Storage);
struct Provider {
    pointer: *const GgufLease,
    calls: Arc<std::sync::atomic::AtomicUsize>,
    retired: Arc<std::sync::atomic::AtomicUsize>,
    refuse: bool,
    short: bool,
}
impl GgufRawStorageProvider for Provider {
    type Storage = Storage;
    type Error = Refusal;
    fn prepare(self, request: GgufRawStorageRequest<'_>) -> Result<Storage, Refusal> {
        assert_eq!(std::ptr::from_ref(request.lease()), self.pointer);
        assert_eq!(request.layout().size(), 24);
        self.calls.fetch_add(1, Ordering::SeqCst);
        let storage = Storage {
            bytes: vec![0; 24 - usize::from(self.short)],
            retired: self.retired,
        };
        if self.refuse {
            Err(Refusal(storage))
        } else {
            Ok(storage)
        }
    }
}

#[test]
fn supplied_g1_refusal_precedes_io_and_retains_actual_box_and_destination() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("supplied.gguf");
    dense_file(&path, "matrix.weight");
    let source = test_store(&path);
    for (refuse, short) in [(true, false), (false, true)] {
        let lease = Box::new(
            source
                .acquire(request("matrix.weight", TensorSelection::Full))
                .unwrap(),
        );
        let pointer = std::ptr::from_ref(lease.as_ref());
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let retired = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let failure = GgufLease::materialize_prepared_boxed_with_storage(
            lease,
            Provider {
                pointer,
                calls: calls.clone(),
                retired: retired.clone(),
                refuse,
                short,
            },
        )
        .unwrap_err();
        assert_eq!(std::ptr::from_ref(failure.lease()), pointer);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        assert!(if refuse {
            matches!(failure, PreparedGgufSuppliedFailure::Provider { .. })
        } else {
            matches!(failure, PreparedGgufSuppliedFailure::Extent { .. })
        });
        drop(failure);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn supplied_g1_layout_and_later_conversion_errors_preserve_existing_order_and_success() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("supplied-order.gguf");
    dense_file(&path, "matrix.weight");
    let source = test_store(&path);
    for stage in 0..3 {
        let mut lease = Box::new(
            source
                .acquire(request("matrix.weight", TensorSelection::Full))
                .unwrap(),
        );
        if stage == 0 {
            lease.proof.length_bytes = u64::MAX;
        }
        if stage == 1 {
            lease.entry.physical_descriptor.byte_len = u64::MAX;
        }
        let pointer = std::ptr::from_ref(lease.as_ref());
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let retired = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let result = GgufLease::materialize_prepared_boxed_with_storage(
            lease,
            Provider {
                pointer,
                calls: calls.clone(),
                retired: retired.clone(),
                refuse: false,
                short: false,
            },
        );
        if stage < 2 {
            let failure = result.unwrap_err();
            assert_eq!(std::ptr::from_ref(failure.lease()), pointer);
            assert_eq!(calls.load(Ordering::SeqCst), usize::from(stage == 1));
            assert_eq!(retired.load(Ordering::SeqCst), 0);
            assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
            if stage == 1 {
                let PreparedGgufSuppliedFailure::Prepared(PreparedGgufBoxedFailure::Conversion(
                    ref partial,
                )) = failure
                else {
                    panic!("actual conversion failure");
                };
                assert_eq!(partial.raw_bytes(), &[0; 24]);
            }
            drop(failure);
            assert_eq!(retired.load(Ordering::SeqCst), usize::from(stage == 1));
        } else {
            let (output, lease) = result.unwrap();
            assert_eq!(std::ptr::from_ref(lease.as_ref()), pointer);
            assert_eq!(decoded(&output), [1.5, -2.5, 3.5, 4.5, -5.5, 6.5]);
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(retired.load(Ordering::SeqCst), 1);
            assert_eq!(source.diagnostics().unwrap().physical_reads, 1);
        }
    }
}

#[test]
fn supplied_raw_partial_read_keeps_real_prefix_and_owner_until_error_retirement() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("supplied-partial.gguf");
    dense_file(&path, "matrix.weight");
    let source = test_store(&path);
    let weak = source.inner.ordinary_weak();
    let lease = Box::new(
        source
            .acquire(request("matrix.weight", TensorSelection::Full))
            .unwrap(),
    );
    lease.materialize_portable().unwrap();
    let pointer = std::ptr::from_ref(lease.as_ref());
    let reads = source.diagnostics().unwrap().physical_reads;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(lease.entry.physical_descriptor.data_offset + 3)
        .unwrap();
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let retired = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let failure = GgufLease::materialize_prepared_boxed_with_storage(
        lease,
        Provider {
            pointer,
            calls: calls.clone(),
            retired: retired.clone(),
            refuse: false,
            short: false,
        },
    )
    .unwrap_err();
    assert_eq!(std::ptr::from_ref(failure.lease()), pointer);
    let PreparedGgufSuppliedFailure::Prepared(PreparedGgufBoxedFailure::Tensor(ref partial)) =
        failure
    else {
        panic!("actual reader failure");
    };
    assert!(partial.store_error().is_some());
    assert_eq!(&partial.raw_bytes()[..3], &1.5_f32.to_le_bytes()[..3]);
    assert_eq!(&partial.raw_bytes()[3..], &[0; 21]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(source.diagnostics().unwrap().physical_reads, reads);
    drop(source);
    assert!(weak.upgrade().is_some());
    drop(failure);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert!(weak.upgrade().is_none());
}
