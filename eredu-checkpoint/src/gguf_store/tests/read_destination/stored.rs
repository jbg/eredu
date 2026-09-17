use super::*;
use crate::gguf_store::{GgufRawStorageRequest, GgufStorageProvider};
use eredu_gguf::{InitializedStorage, StorageFamily, StorageProvider, StoredConvertedTensor};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
#[derive(Debug)]
struct Family;
#[derive(Debug)]
struct Buffer<T> {
    values: Vec<T>,
    retired: Rc<Cell<usize>>,
}
impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        drop(std::mem::take(&mut self.values));
        self.retired.set(self.retired.get() + 1)
    }
}
impl<T> AsRef<[T]> for Buffer<T> {
    fn as_ref(&self) -> &[T] {
        &self.values
    }
}
impl<T> AsMut<[T]> for Buffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}
impl<T: std::fmt::Debug + 'static> InitializedStorage<T> for Buffer<T> {
    fn capacity(&self) -> usize {
        self.values.capacity()
    }
}
#[derive(Debug, thiserror::Error)]
#[error("reached supplied destination refused at {0}")]
struct Refused(usize);
impl StorageFamily for Family {
    type Buffer<T: Copy + std::fmt::Debug + 'static> = Buffer<T>;
    type Error = Refused;
}
struct Provider {
    pointer: usize,
    calls: Rc<RefCell<Vec<(usize, usize)>>>,
    retired: Rc<Cell<usize>>,
    fail: Option<usize>,
    inspect: Option<GgufWeightStore>,
}
impl StorageProvider for Provider {
    type Family = Family;
    fn prepare<T: Copy + std::fmt::Debug + 'static>(
        &mut self,
        n: usize,
        initial: T,
    ) -> Result<Buffer<T>, (Refused, Option<Buffer<T>>)> {
        if let Some(source) = &self.inspect {
            // Fail deterministically rather than hang if a callback regresses
            // under this mutex. Then execute the real public reentrant API.
            let lock = source
                .inner
                .readers
                .try_lock()
                .expect("provider outside reader lock");
            drop(lock);
            assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
        }
        let index = self.calls.borrow().len();
        self.calls.borrow_mut().push((n, std::mem::size_of::<T>()));
        let out = Buffer {
            values: vec![initial; n],
            retired: self.retired.clone(),
        };
        if self.fail == Some(index) {
            Err((Refused(index), Some(out)))
        } else {
            Ok(out)
        }
    }
}
impl GgufStorageProvider for Provider {
    fn prepare_raw(
        &mut self,
        request: GgufRawStorageRequest<'_>,
    ) -> Result<Buffer<u8>, (Refused, Option<Buffer<u8>>)> {
        if std::ptr::from_ref(request.lease()) as usize != self.pointer {
            return Err((Refused(usize::MAX), None));
        }
        self.prepare(request.layout().size(), 0)
    }
}
fn supplied(
    lease: Box<GgufLease>,
    fail: Option<usize>,
) -> (
    Result<
        (eredu_gguf::StoredCheckpointTensor<Family>, Box<GgufLease>),
        crate::gguf_store::StoredGgufFailure<Family>,
    >,
    Rc<RefCell<Vec<(usize, usize)>>>,
    Rc<Cell<usize>>,
) {
    let pointer = std::ptr::from_ref(lease.as_ref()) as usize;
    let calls = Rc::new(RefCell::new(Vec::new()));
    let retired = Rc::new(Cell::new(0));
    (
        GgufLease::materialize_prepared_boxed_with_destinations(
            lease,
            Provider {
                pointer,
                calls: calls.clone(),
                retired: retired.clone(),
                fail,
                inspect: None,
            },
        ),
        calls,
        retired,
    )
}
#[test]
fn supplied_g1_g2_g3_preserves_actual_box_cache_order_and_owning_names() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("stored.gguf");
    dense_file(&path, "matrix.weight");
    let source = test_store(&path);
    let ordinary = test_store(&path);
    for selection in [
        TensorSelection::Full,
        TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0, 1],
        },
        TensorSelection::Range {
            axis: 0,
            start: 1,
            end: 2,
        },
        TensorSelection::Contiguous {
            offset_elements: 1,
            shape: vec![1, 2],
        },
    ] {
        let expected = ordinary
            .acquire(request("matrix.weight", selection.clone()))
            .unwrap()
            .materialize_portable()
            .unwrap();
        let query = request("matrix.weight", selection.clone());
        let before = stats(&source);
        let bound = source
            .conversion_plan(&query)
            .unwrap()
            .supplied_storage_bound()
            .unwrap();
        assert_eq!(stats(&source), before);
        let lease = Box::new(source.acquire(query).unwrap());
        let pointer = std::ptr::from_ref(lease.as_ref());
        let (result, calls, retired) = supplied(lease, None);
        let (output, lease) = result.unwrap();
        let used: usize = calls.borrow().iter().map(|(n, size)| n * size).sum();
        assert!(bound.bytes() >= used);
        assert!(bound.calls() >= calls.borrow().len());
        if matches!(selection, TensorSelection::Full) {
            assert_eq!((bound.bytes(), bound.calls()), (used, calls.borrow().len()));
        }
        assert_eq!(std::ptr::from_ref(lease.as_ref()), pointer);
        assert_eq!(output.descriptor.view(), expected.descriptor().view());
        assert_eq!(
            output.output_names.get(0),
            Some(expected.output_names()[0].as_str())
        );
        let StoredConvertedTensor::Dense { data, .. } = &output.converted else {
            panic!("dense")
        };
        let ConvertedTensor::Dense(expected) = expected.converted() else {
            panic!("dense")
        };
        assert_eq!(data.as_slice(), expected.data);
        assert!(retired.get() < calls.borrow().len());
        assert_eq!(stats(&source), stats(&ordinary));
        drop(output);
        assert_eq!(retired.get(), calls.borrow().len());
        drop(lease);
    }
}
#[test]
fn every_reached_supplied_refusal_retains_same_source_and_exact_failed_prefix() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("prefix.gguf");
    dense_file(&path, "matrix.weight");
    let baseline = test_store(&path);
    let (result, calls, _) = supplied(
        Box::new(
            baseline
                .acquire(request("matrix.weight", TensorSelection::Full))
                .unwrap(),
        ),
        None,
    );
    drop(result.unwrap());
    let total = calls.borrow().len();
    assert!(total > 5);
    for fail in 0..total {
        let source = test_store(&path);
        let weak = source.inner.ordinary_weak();
        let lease = Box::new(
            source
                .acquire(request("matrix.weight", TensorSelection::Full))
                .unwrap(),
        );
        let pointer = std::ptr::from_ref(lease.as_ref());
        let (result, calls, retired) = supplied(lease, Some(fail));
        let failure = result.unwrap_err();
        assert_eq!(std::ptr::from_ref(failure.lease()), pointer);
        assert_eq!(calls.borrow().len(), fail + 1);
        assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
        assert!(retired.get() < fail + 1);
        drop(source);
        assert!(weak.upgrade().is_some());
        drop(failure);
        assert_eq!(retired.get(), fail + 1);
        assert!(weak.upgrade().is_none());
    }
}
#[test]
fn supplied_foreign_source_refuses_before_payload_or_cache_and_keeps_genuine_box() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("foreign.gguf");
    dense_file(&path, "matrix.weight");
    let source = test_store(&path);
    let lease = Box::new(
        source
            .acquire(request("matrix.weight", TensorSelection::Full))
            .unwrap(),
    );
    let pointer = std::ptr::from_ref(lease.as_ref());
    let calls = Rc::new(RefCell::new(Vec::new()));
    let retired = Rc::new(Cell::new(0));
    let failure = GgufLease::materialize_prepared_boxed_with_destinations(
        lease,
        Provider {
            pointer: 0,
            calls: calls.clone(),
            retired: retired.clone(),
            fail: None,
            inspect: None,
        },
    )
    .unwrap_err();
    assert_eq!(std::ptr::from_ref(failure.lease()), pointer);
    assert!(calls.borrow().is_empty());
    assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
    assert!(failure.raw_bytes().is_none());
    drop(failure);
    assert_eq!(retired.get(), 0);
}

#[test]
fn supplied_provider_reenters_actual_source_diagnostics_outside_reader_lock() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reentrant.gguf");
    dense_file(&path, "matrix.weight");
    let source = test_store(&path);
    let ordinary = test_store(&path);
    let (baseline, expected_calls, _) = supplied(
        Box::new(
            ordinary
                .acquire(request("matrix.weight", TensorSelection::Full))
                .unwrap(),
        ),
        None,
    );
    let (expected, _) = baseline.unwrap();
    let lease = Box::new(
        source
            .acquire(request("matrix.weight", TensorSelection::Full))
            .unwrap(),
    );
    let pointer = std::ptr::from_ref(lease.as_ref()) as usize;
    let calls = Rc::new(RefCell::new(Vec::new()));
    let retired = Rc::new(Cell::new(0));
    let (actual, returned) = GgufLease::materialize_prepared_boxed_with_destinations(
        lease,
        Provider {
            pointer,
            calls: calls.clone(),
            retired: retired.clone(),
            fail: None,
            inspect: Some(source.clone()),
        },
    )
    .unwrap();
    assert_eq!(std::ptr::from_ref(returned.as_ref()) as usize, pointer);
    assert_eq!(*calls.borrow(), *expected_calls.borrow());
    assert_eq!(actual.descriptor.view(), expected.descriptor.view());
    assert_eq!(
        actual.output_names.iter().collect::<Vec<_>>(),
        expected.output_names.iter().collect::<Vec<_>>()
    );
    let StoredConvertedTensor::Dense {
        data: actual_data, ..
    } = &actual.converted
    else {
        panic!("dense")
    };
    let StoredConvertedTensor::Dense {
        data: expected_data,
        ..
    } = &expected.converted
    else {
        panic!("dense")
    };
    assert_eq!(actual_data.as_slice(), expected_data.as_slice());
    assert!(actual_data.as_slice().iter().any(|b| *b != 0));
    assert_eq!(stats(&source), stats(&ordinary));
    drop(actual);
    assert_eq!(retired.get(), calls.borrow().len());
}

#[test]
fn supplied_request_bounds_cover_zero_calls_and_every_failed_prefix_without_io() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bounds.gguf");
    dense_file(&path, "matrix.weight");
    for selection in [
        TensorSelection::Full,
        TensorSelection::Range {
            axis: 0,
            start: 0,
            end: 1,
        },
        TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0, 1],
        },
        TensorSelection::Contiguous {
            offset_elements: 1,
            shape: vec![1, 2],
        },
    ] {
        let source = test_store(&path);
        let query = request("matrix.weight", selection);
        let bound = source
            .conversion_plan(&query)
            .unwrap()
            .supplied_storage_bound()
            .unwrap();
        let (baseline, calls, _) = supplied(Box::new(source.acquire(query.clone()).unwrap()), None);
        let calls = calls.borrow().clone();
        drop(baseline); // invalid empty selections retain their actual failure
        assert!(calls.len() <= bound.calls());
        assert!(calls.iter().map(|(n, s)| n * s).sum::<usize>() <= bound.bytes());
        for fail in 0..calls.len() {
            let source = test_store(&path);
            let (result, actual, retired) =
                supplied(Box::new(source.acquire(query.clone()).unwrap()), Some(fail));
            assert!(result.is_err());
            assert_eq!(*actual.borrow(), calls[..=fail]);
            assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
            drop(result);
            assert_eq!(retired.get(), fail + 1);
        }
    }
}

#[test]
fn supplied_request_query_keeps_invalid_selection_unavailable_without_reading() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("invalid-bounds.gguf");
    dense_file(&path, "matrix.weight");
    let source = test_store(&path);
    for selection in [
        TensorSelection::Range {
            axis: 0,
            start: 0,
            end: 0,
        },
        TensorSelection::Indices {
            axis: 0,
            indices: vec![],
        },
    ] {
        let query = request("matrix.weight", selection);
        let before = stats(&source);
        assert!(source.conversion_plan(&query).is_err());
        assert!(source.acquire(query).is_err());
        assert_eq!(stats(&source), before);
    }
}

#[test]
fn supplied_request_bounds_match_all_full_physical_groups_and_endian_dispatch() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("all-bounds.gguf");
    for endian in [Endian::Little, Endian::Big] {
        for ty in [
            GgmlType::F32,
            GgmlType::Q4_0,
            GgmlType::IQ4NL,
            GgmlType::MxFp4,
        ] {
            let (values, bytes) = ty.block_and_bytes().unwrap();
            let raw = if ty == GgmlType::F32 {
                (0..128)
                    .flat_map(|i| {
                        let v = (i as f32 - 31.0) * 0.25;
                        match endian {
                            Endian::Little => v.to_le_bytes(),
                            Endian::Big => v.to_be_bytes(),
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                (0..128 / values * bytes)
                    .map(|i| (i.wrapping_mul(13).wrapping_add(37) % 256) as u8)
                    .collect()
            };
            Writer::new(WriterOptions {
                version: 3,
                endian,
                alignment: 32,
            })
            .unwrap()
            .write(
                File::create(&path).unwrap(),
                &BTreeMap::new(),
                &[TensorInput {
                    name: "matrix.weight",
                    dimensions: &[64, 2],
                    ggml_type: ty,
                    data: &raw,
                }],
            )
            .unwrap();
            let source = test_store(&path);
            for key in source.keys() {
                let query = request(&key, TensorSelection::Full);
                let before = stats(&source);
                let bound = source
                    .conversion_plan(&query)
                    .unwrap()
                    .supplied_storage_bound()
                    .unwrap();
                assert_eq!(stats(&source), before);
                let (result, calls, retired) =
                    supplied(Box::new(source.acquire(query).unwrap()), None);
                let calls = calls.borrow().clone();
                assert_eq!(bound.calls(), calls.len(), "{ty:?} {endian:?} {key}");
                assert_eq!(
                    bound.bytes(),
                    calls.iter().map(|(n, s)| n * s).sum::<usize>()
                );
                let (output, lease) = result.unwrap();
                assert_eq!(
                    output.output_names.len(),
                    source
                        .conversion_plan(&request(&key, TensorSelection::Full))
                        .unwrap()
                        .conversion()
                        .outputs()
                        .len()
                );
                drop((output, lease));
                assert_eq!(retired.get(), calls.len());
            }
        }
    }
}
