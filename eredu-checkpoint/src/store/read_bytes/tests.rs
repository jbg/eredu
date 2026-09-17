use super::*;
use crate::store::{
    fs_error, invalid_selection, io_error, CheckpointSource, EncodedTensorLease, ReadPolicy,
    SafetensorsWeightStore, TensorReadRequest, TensorSelection, WeightStore,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::{cell::RefCell, rc::Rc, sync::Arc};

fn old_read_safetensors_range_pass(
    path: &Path,
    file: &mut File,
    tensor_payload_start: usize,
    ranges: &[Range<usize>],
    capacity: usize,
    telemetry: &SafetensorsReadTelemetry,
) -> Result<Vec<u8>, StoreError> {
    let mut output = Vec::with_capacity(capacity);
    for range in ranges {
        let absolute = tensor_payload_start
            .checked_add(range.start)
            .ok_or_else(|| StoreError::Overflow {
                context: format!("selected payload offset for {}", path.display()),
            })?;
        file.seek(SeekFrom::Start(u64::try_from(absolute).map_err(|_| {
            StoreError::Overflow {
                context: format!("selected payload offset for {}", path.display()),
            }
        })?))
        .map_err(|error| io_error(path, error))?;
        let start = output.len();
        output.resize(start + range.len(), 0);
        file.read_exact(&mut output[start..])
            .map_err(|error| io_error(path, error))?;
        telemetry.physical_reads.fetch_add(1, Ordering::Relaxed);
        telemetry
            .physical_read_bytes
            .fetch_add(range.len() as u64, Ordering::Relaxed);
    }
    Ok(output)
}

fn old_copy_safetensors_ranges(
    key: &str,
    payload: &[u8],
    ranges: &[Range<usize>],
) -> Result<Vec<u8>, StoreError> {
    let capacity = ranges.iter().try_fold(0usize, |total, range| {
        total
            .checked_add(range.len())
            .ok_or_else(|| StoreError::Overflow {
                context: format!("cached selected payload length for {key:?}"),
            })
    })?;
    let mut output = Vec::with_capacity(capacity);
    for range in ranges {
        output.extend_from_slice(
            payload
                .get(range.clone())
                .ok_or_else(|| invalid_selection(key, "cached selection exceeds payload"))?,
        );
    }
    Ok(output)
}

impl AdmittedFile {
    fn old_validate_file(&self, path: &Path, file: &File) -> Result<(), StoreError> {
        let current = AdmittedFileIdentity::from_metadata(
            path,
            &file.metadata().map_err(|error| fs_error(path, error))?,
        )?;
        if current != self.identity {
            return Err(StoreError::AdmittedFileChanged {
                path: path.to_path_buf(),
            });
        }
        Ok(())
    }
}

fn actual_store() -> (
    tempfile::TempDir,
    std::path::PathBuf,
    SafetensorsWeightStore,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("weights.safetensors");
    serialize_to_file(
        [
            (
                "weight",
                TensorView::new(Dtype::U8, vec![2, 4], &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap(),
            ),
            (
                "other",
                TensorView::new(Dtype::U8, vec![2, 4], &[11, 12, 13, 14, 15, 16, 17, 18]).unwrap(),
            ),
        ],
        None,
        &path,
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(&path).unwrap();
    (directory, path, store)
}
fn gathered() -> TensorSelection {
    TensorSelection::Indices {
        axis: 1,
        indices: vec![0, 3],
    }
}

#[test]
fn actual_byte_destinations_match_nonzero_ordinary_reads_and_independent_workers() {
    let (_directory, _path, store) = actual_store();
    for selection in [
        TensorSelection::Full,
        gathered(),
        TensorSelection::Indices {
            axis: 1,
            indices: vec![1, 1],
        },
    ] {
        let controls = store.source_lease_controls("weight").unwrap();
        let source = controls.safetensors_reads().unwrap();
        let mut shape = [0; 2];
        let mut replacement = [];
        let read = source
            .plan(
                &selection,
                ReadPolicy::RequireBounded,
                &mut shape,
                &mut replacement,
            )
            .unwrap();
        let mut ranges = vec![0..0; read.range_count()];
        let plan = read.byte_plan(&mut ranges).unwrap();
        let old_telemetry = SafetensorsReadTelemetry::default();
        let mut file = plan
            .source
            .admitted
            .open_validated(plan.source.path)
            .unwrap();
        let old = old_read_safetensors_range_pass(
            plan.source.path,
            &mut file,
            plan.tensor_payload_start,
            plan.ranges,
            plan.destination_layout().size(),
            &old_telemetry,
        )
        .unwrap();
        let copied =
            old_copy_safetensors_ranges("weight", &[1, 2, 3, 4, 5, 6, 7, 8], plan.ranges).unwrap();
        let mut destination = vec![0xcc; plan.destination_layout().size()];
        let pointer = destination.as_ptr();
        let bytes = plan.read_into(&mut destination).unwrap();
        assert_eq!(bytes.as_slice(), old);
        assert_eq!(bytes.as_slice(), copied);
        assert_eq!(bytes.as_slice().as_ptr(), pointer);
        assert!(bytes.physically_bounded());
        assert_eq!(
            bytes.physical_reads(),
            old_telemetry.physical_reads.load(Ordering::Relaxed) as usize
        );
        let ordinary = store
            .acquire(TensorReadRequest {
                key: "weight".into(),
                selection: selection.clone(),
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        assert_eq!(bytes.as_slice(), ordinary.encoded_bytes().unwrap());
    }
}

#[test]
fn exact_byte_destination_refuses_short_and_long_before_open_or_write() {
    let (_directory, path, store) = actual_store();
    let controls = store.source_lease_controls("weight").unwrap();
    let source = controls.safetensors_reads().unwrap();
    let selection = gathered();
    let mut shape = [0; 2];
    let mut replacement = [];
    let read = source
        .plan(
            &selection,
            ReadPolicy::RequireBounded,
            &mut shape,
            &mut replacement,
        )
        .unwrap();
    std::fs::remove_file(path).unwrap();
    for length in [3, 5] {
        let mut ranges = vec![0..0; read.range_count()];
        let plan = read.byte_plan(&mut ranges).unwrap();
        let mut destination = vec![0xa7; length];
        let failure = plan.read_into(&mut destination).unwrap_err();
        assert!(matches!(
            failure.cause(),
            SafetensorsByteError::Destination { expected: 4, .. }
        ));
        assert_eq!(failure.destination(), vec![0xa7; length]);
        assert_eq!(failure.initialized_bytes(), 0);
        assert!(!failure.retains_file());
    }
    let mut ranges = vec![0..0; read.range_count()];
    let mut destination = [0xa7; 4];
    let failure = read
        .byte_plan(&mut ranges)
        .unwrap()
        .read_into(&mut destination)
        .unwrap_err();
    assert!(
        matches!(failure.cause(), SafetensorsByteError::Filesystem { cause, .. } if cause.kind() == io::ErrorKind::NotFound)
    );
    assert_eq!(failure.destination(), &[0xa7; 4]);
    assert!(std::error::Error::source(&failure)
        .unwrap()
        .is::<io::Error>());
    assert_eq!(store.diagnostics().unwrap().physical_reads, 0);
}

#[test]
fn cached_copy_requires_actual_same_full_source_and_validates_before_copy() {
    let (_directory, path, store) = actual_store();
    let full = store
        .acquire(TensorReadRequest {
            key: "weight".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    let other = store
        .acquire(TensorReadRequest {
            key: "other".into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    let selected = store
        .acquire(TensorReadRequest {
            key: "weight".into(),
            selection: gathered(),
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    let reordered = store
        .acquire(TensorReadRequest {
            key: "weight".into(),
            selection: TensorSelection::Indices {
                axis: 1,
                indices: vec![0, 2, 3, 1],
            },
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    let duplicate = store
        .acquire(TensorReadRequest {
            key: "weight".into(),
            selection: TensorSelection::Indices {
                axis: 1,
                indices: vec![0, 1, 1, 2],
            },
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    // Both defeat a merely numeric offset/length check: they start at zero,
    // have the full extent, yet their retained bytes have different meaning.
    for wrong in [&reordered, &duplicate] {
        assert_eq!(wrong.proof.offset_bytes, 0);
        assert_eq!(wrong.proof.length_bytes, full.proof.length_bytes);
        assert_eq!(wrong.bytes.len(), full.bytes.len());
        assert_ne!(wrong.encoded_bytes(), full.encoded_bytes());
        assert_ne!(
            old_copy_safetensors_ranges("weight", &wrong.bytes, &[0..1, 3..5, 7..8]).unwrap(),
            vec![1, 4, 5, 8]
        );
    }
    let controls = store.source_lease_controls("weight").unwrap();
    let source = controls.safetensors_reads().unwrap();
    let selection = gathered();
    let mut shape = [0; 2];
    let mut replacement = [];
    let read = source
        .plan(
            &selection,
            ReadPolicy::RequireBounded,
            &mut shape,
            &mut replacement,
        )
        .unwrap();
    let before = store.diagnostics().unwrap();
    for wrong in [&other, &selected, &reordered, &duplicate] {
        let mut ranges = vec![0..0; read.range_count()];
        let mut output = [0xcc; 4];
        let error = read
            .byte_plan(&mut ranges)
            .unwrap()
            .copy_from(wrong, &mut output)
            .unwrap_err();
        assert!(matches!(
            error.cause(),
            SafetensorsByteError::SourceMismatch
        ));
        assert_eq!(error.destination(), &[0xcc; 4]);
    }
    let mut ranges = vec![0..0; read.range_count()];
    let mut output = [0xcc; 4];
    let copied = read
        .byte_plan(&mut ranges)
        .unwrap()
        .copy_from(&full, &mut output)
        .unwrap();
    assert_eq!(copied.as_slice(), &[1, 4, 5, 8]);
    assert_eq!(copied.physical_reads(), 0);
    drop(copied);
    let after = store.diagnostics().unwrap();
    assert_eq!(before.physical_reads, after.physical_reads);
    assert_eq!(before.cache_hits, after.cache_hits);
    assert_eq!(before.cache_misses, after.cache_misses);
    assert_eq!(before.payload_shard_paths, after.payload_shard_paths);
    std::fs::remove_file(path).unwrap();
    let failure = read
        .byte_plan(&mut ranges)
        .unwrap()
        .copy_from(&full, &mut output)
        .unwrap_err();
    assert!(matches!(
        failure.cause(),
        SafetensorsByteError::Filesystem { .. }
    ));
    assert_eq!(failure.destination(), &[1, 4, 5, 8]);
    assert_eq!(failure.initialized_bytes(), 0);
}

#[test]
fn completed_read_retains_source_file_and_failed_output_on_final_version_change() {
    let (_directory, path, store) = actual_store();
    let controls = store.source_lease_controls("weight").unwrap();
    let source = controls.safetensors_reads().unwrap();
    let selection = gathered();
    let mut shape = [0; 2];
    let mut replacement = [];
    let read = source
        .plan(
            &selection,
            ReadPolicy::RequireBounded,
            &mut shape,
            &mut replacement,
        )
        .unwrap();
    let mut ranges = vec![0..0; read.range_count()];
    let mut output = [0xcc; 4];
    let plan = read.byte_plan(&mut ranges).unwrap();
    let error = plan
        .read_into_with(&mut output, || {
            File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(1)
                .unwrap();
        })
        .unwrap_err();
    assert!(matches!(
        error.cause(),
        SafetensorsByteError::Changed { .. }
    ));
    assert_eq!(error.completed_ranges(), 3);
    assert_eq!(error.completed_bytes(), 4);
    assert_eq!(error.initialized_bytes(), 4);
    assert_eq!(error.destination(), &[1, 4, 5, 8]);
    assert!(error.retains_file());
    assert_eq!(store.diagnostics().unwrap().physical_reads, 3);
    let opened = error.file.as_ref().unwrap();
    let old = error
        .plan
        .source
        .admitted
        .old_validate_file(error.source_path(), opened)
        .unwrap_err();
    assert_eq!(old.to_string(), error.to_string());
    // Exact source and destination loans stay in the error until this drop.
    drop(error);
    output.fill(0);
    assert_eq!(output, [0; 4]);
}

#[derive(Debug)]
struct OwnedIoCause(Arc<std::sync::atomic::AtomicUsize>);
impl std::fmt::Display for OwnedIoCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("injected read cause")
    }
}
impl std::error::Error for OwnedIoCause {}
impl Drop for OwnedIoCause {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}
struct FailingReader {
    log: Rc<RefCell<Vec<&'static str>>>,
    reads: usize,
    dropped: Arc<std::sync::atomic::AtomicUsize>,
}
impl Read for FailingReader {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.log.borrow_mut().push("read");
        self.reads += 1;
        match self.reads {
            1 => {
                bytes[0] = 9;
                Ok(1)
            }
            2 => {
                bytes[0] = 7;
                Ok(1)
            }
            _ => Err(io::Error::other(OwnedIoCause(self.dropped.clone()))),
        }
    }
}
impl Seek for FailingReader {
    fn seek(&mut self, _: SeekFrom) -> io::Result<u64> {
        self.log.borrow_mut().push("seek");
        Ok(0)
    }
}
struct LoggedOutput<'a> {
    inner: FixedOutput<'a>,
    log: Rc<RefCell<Vec<&'static str>>>,
}
impl ByteOutput for LoggedOutput<'_> {
    fn zeroed(&mut self, count: usize) -> &mut [u8] {
        self.log.borrow_mut().push("zero");
        self.inner.zeroed(count)
    }
    fn append(&mut self, bytes: &[u8]) {
        self.log.borrow_mut().push("copy");
        self.inner.append(bytes);
    }
}
#[test]
fn shared_byte_worker_preserves_seek_zero_read_order_and_owned_partial_failure() {
    let log = Rc::new(RefCell::new(vec![]));
    let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut reader = FailingReader {
        log: log.clone(),
        reads: 0,
        dropped: drops.clone(),
    };
    let mut bytes = [0xcc; 5];
    let mut output = LoggedOutput {
        inner: FixedOutput {
            bytes: &mut bytes,
            used: 0,
        },
        log: log.clone(),
    };
    let telemetry = SafetensorsReadTelemetry::default();
    let mut progress = Progress::default();
    let error = read_pass(
        &mut reader,
        0,
        &[0..1, 3..6, 9..10],
        &telemetry,
        &mut output,
        &mut progress,
        &FixedErrors {
            path: Path::new("actual-test-reader"),
            key: "weight",
        },
    )
    .unwrap_err();
    assert_eq!(
        *log.borrow(),
        ["seek", "zero", "read", "seek", "zero", "read", "read"]
    );
    assert_eq!(progress.completed_ranges, 1);
    assert_eq!(progress.completed_bytes, 1);
    assert_eq!(progress.initialized_bytes, 4);
    assert_eq!(output.inner.bytes, &[9, 7, 0, 0, 0xcc]);
    assert_eq!(telemetry.physical_reads.load(Ordering::Relaxed), 1);
    assert_eq!(telemetry.physical_read_bytes.load(Ordering::Relaxed), 1);
    let SafetensorsByteError::Read { cause, .. } = &error else {
        panic!("real IO cause");
    };
    assert!(cause.get_ref().unwrap().is::<OwnedIoCause>());
    assert_eq!(drops.load(Ordering::Relaxed), 0);
    drop(error);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}

#[test]
fn ordinary_byte_adapters_preserve_capacity_errors_and_cache_copy_prefix_order() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bytes");
    std::fs::write(&path, [1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    for ranges in [
        vec![0..8],
        vec![0..1, 3..5, 7..8],
        vec![1..2; 17],
        vec![0..1, 7..10],
    ] {
        let capacity = ranges.iter().map(|range| range.len()).sum();
        let old_t = SafetensorsReadTelemetry::default();
        let new_t = SafetensorsReadTelemetry::default();
        let old = old_read_safetensors_range_pass(
            &path,
            &mut File::open(&path).unwrap(),
            0,
            &ranges,
            capacity,
            &old_t,
        );
        let new = ordinary_read_pass(
            &path,
            &mut File::open(&path).unwrap(),
            0,
            &ranges,
            capacity,
            &new_t,
        );
        match (old, new) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b);
                assert_eq!(a.capacity(), b.capacity());
            }
            (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
            pair => panic!("different read outcomes: {pair:?}"),
        }
        assert_eq!(
            old_t.physical_reads.load(Ordering::Relaxed),
            new_t.physical_reads.load(Ordering::Relaxed)
        );
        assert_eq!(
            old_t.physical_read_bytes.load(Ordering::Relaxed),
            new_t.physical_read_bytes.load(Ordering::Relaxed)
        );
        match (
            old_copy_safetensors_ranges("weight", &[1, 2, 3, 4, 5, 6, 7, 8], &ranges),
            ordinary_copy("weight", &[1, 2, 3, 4, 5, 6, 7, 8], &ranges),
        ) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b);
                assert_eq!(a.capacity(), b.capacity());
            }
            (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
            pair => panic!("different copy outcomes: {pair:?}"),
        }
    }
    let mut bytes = [0xcc; 4];
    let mut output = FixedOutput {
        bytes: &mut bytes,
        used: 0,
    };
    let mut progress = Progress::default();
    let error = copy_pass(
        &[1, 2, 3],
        &[0..1, 2..5],
        &mut output,
        &mut progress,
        &FixedErrors {
            path: &path,
            key: "weight",
        },
    )
    .unwrap_err();
    assert!(matches!(error, SafetensorsByteError::CachedRange { .. }));
    assert_eq!(output.bytes, &[1, 0xcc, 0xcc, 0xcc]);
    assert_eq!(progress.completed_ranges, 1);
}
