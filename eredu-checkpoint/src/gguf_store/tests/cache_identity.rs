use super::*;
use eredu_gguf::{InitializedStorage, StorageFamily, StorageProvider};
use std::sync::atomic::AtomicUsize;

#[derive(Debug)]
struct Family;
#[derive(Debug)]
struct Buffer<T> {
    values: Vec<T>,
    drops: Arc<AtomicUsize>,
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
impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
#[derive(Debug)]
struct Failure;
impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("key allocation refusal")
    }
}
impl std::error::Error for Failure {}
impl StorageFamily for Family {
    type Buffer<T: Copy + std::fmt::Debug + 'static> = Buffer<T>;
    type Error = Failure;
}
struct Provider {
    requests: Vec<(usize, usize)>,
    drops: Arc<AtomicUsize>,
    fail: Option<usize>,
    wrong_extent: bool,
}
impl StorageProvider for Provider {
    type Family = Family;
    fn prepare<T: Copy + std::fmt::Debug + 'static>(
        &mut self,
        elements: usize,
        initializer: T,
    ) -> Result<Buffer<T>, (Failure, Option<Buffer<T>>)> {
        self.requests.push((std::mem::size_of::<T>(), elements));
        let length = if self.wrong_extent {
            elements + 1
        } else {
            elements
        };
        let buffer = Buffer {
            values: vec![initializer; length],
            drops: self.drops.clone(),
        };
        if self.fail == Some(self.requests.len()) {
            Err((Failure, Some(buffer)))
        } else {
            Ok(buffer)
        }
    }
}
fn source() -> (tempfile::TempDir, GgufWeightStore) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.gguf");
    let bytes = (1..=8)
        .flat_map(|n| (n as f32).to_le_bytes())
        .collect::<Vec<_>>();
    write_tensor(&path, "matrix.weight", &[4, 2], GgmlType::F32, &bytes);
    (dir, test_store(&path))
}
#[test]
fn supplied_cache_key_uses_actual_ordered_metadata_requests_without_payload_reads() {
    let (_dir, source) = source();
    let drops = Arc::new(AtomicUsize::new(0));
    for selection in [
        TensorSelection::Full,
        TensorSelection::Range {
            axis: 0,
            start: 1,
            end: 2,
        },
        TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0, 1],
        },
        TensorSelection::Contiguous {
            offset_elements: 4,
            shape: vec![1, 2],
        },
    ] {
        let plan = source
            .conversion_plan(&TensorReadRequest {
                key: "matrix.weight".into(),
                selection,
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        let mut provider = Provider {
            requests: Vec::new(),
            drops: drops.clone(),
            fail: None,
            wrong_extent: false,
        };
        let stored = StoredGgufCacheIdentity::prepare(plan.identity(), &mut provider).unwrap();
        let bound = plan.identity().cache_storage_requests().unwrap();
        assert_eq!(stored.cache_view(), plan.identity().cache_view());
        assert_eq!(provider.requests.len(), bound.calls());
        assert_eq!(
            provider
                .requests
                .iter()
                .map(|(width, n)| width * n)
                .sum::<usize>(),
            bound.bytes()
        );
        let before = drops.load(Ordering::SeqCst);
        drop(stored);
        assert_eq!(drops.load(Ordering::SeqCst) - before, bound.calls());
    }
    assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
}
#[test]
fn supplied_cache_key_refusal_keeps_successful_and_failed_buffers_until_retirement() {
    let (_dir, source) = source();
    let plan = source
        .conversion_plan(&TensorReadRequest {
            key: "matrix.weight".into(),
            selection: TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 0, 1],
            },
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    for wrong_extent in [false, true] {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut provider = Provider {
            requests: Vec::new(),
            drops: drops.clone(),
            fail: (!wrong_extent).then_some(2),
            wrong_extent,
        };
        let error = StoredGgufCacheIdentity::prepare(plan.identity(), &mut provider).unwrap_err();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        let (cause, prefix) = error.into_parts();
        if wrong_extent {
            assert!(matches!(
                cause,
                eredu_gguf::SuppliedStorageError::Extent { .. }
            ));
            assert_eq!(provider.requests.len(), 1);
        } else {
            assert!(matches!(
                cause,
                eredu_gguf::SuppliedStorageError::Provider(Failure)
            ));
            assert_eq!(provider.requests.len(), 2);
        }
        drop(prefix);
        assert_eq!(drops.load(Ordering::SeqCst), provider.requests.len());
    }
    assert_eq!(source.diagnostics().unwrap().physical_reads, 0);
}
