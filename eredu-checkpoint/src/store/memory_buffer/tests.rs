use super::*;
use crate::recipe::{DerivedWeightRecipe, EncodedRecipeRead};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct Custody(Arc<AtomicUsize>);
impl Drop for Custody {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn buffer(name: &str, drops: &Arc<AtomicUsize>) -> MemoryTensorBuffer {
    let mut buffer = MemoryTensorBuffer::allocate_with_custody(
        name,
        Dtype::U32,
        &[3, 2],
        24,
        Custody(drops.clone()),
    )
    .unwrap();
    for (bytes, value) in buffer.bytes_mut().chunks_exact_mut(4).zip(1u32..=6) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
    buffer
}

fn request(selection: TensorSelection) -> TensorReadRequest {
    TensorReadRequest {
        key: "packed".into(),
        selection,
        policy: ReadPolicy::RequireBounded,
    }
}

#[test]
fn publication_moves_the_original_bytes_and_preserves_source_metadata() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut buffer = buffer("packed", &drops);
    let original = buffer.bytes_mut().as_ptr();
    let store = MemoryWeightStore::from_buffers([buffer]).unwrap();
    let lease = store.acquire(request(TensorSelection::Full)).unwrap();
    assert_eq!(lease.encoded_bytes().unwrap().as_ptr(), original);
    assert_eq!(lease.metadata().logical_shape, [3, 2]);
    assert_eq!(lease.metadata().encoded_byte_len, 24);
    assert_eq!(lease.metadata().stored_dtype, StoredDtype::U32);
    assert_eq!(lease.metadata().backing_shard, None);
    assert!(matches!(
        store.source_key_authority_borrowed("packed"),
        Ok(SourceKeyAuthority::Ordinary)
    ));
    drop(store);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(
        lease.encoded_bytes().unwrap(),
        (1u32..=6).flat_map(u32::to_le_bytes).collect::<Vec<_>>()
    );
    drop(lease);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn each_independent_reader_retains_constructor_custody() {
    for route in 0..4 {
        let drops = Arc::new(AtomicUsize::new(0));
        let store = MemoryWeightStore::from_buffers([buffer("packed", &drops)]).unwrap();
        let alias: Box<dyn Any> = match route {
            0 => Box::new(store.acquire(request(TensorSelection::Full)).unwrap()),
            1 => Box::new(
                store
                    .acquire(request(TensorSelection::Indices {
                        axis: 0,
                        indices: vec![2, 0, 2],
                    }))
                    .unwrap(),
            ),
            2 => Box::new(
                store
                    .acquire(request(TensorSelection::Full))
                    .unwrap()
                    .pin_encoded_bytes(),
            ),
            3 => {
                let mut owner = None;
                assert!(store
                    .visit_source_storage(&mut |row| owner = Some(row.retain()))
                    .unwrap());
                Box::new(owner.unwrap())
            }
            _ => unreachable!(),
        };
        drop(store);
        assert_eq!(drops.load(Ordering::SeqCst), 0, "route {route}");
        drop(alias);
        assert_eq!(drops.load(Ordering::SeqCst), 1, "route {route}");
    }
}

#[test]
fn detached_reader_keeps_payload_custody_after_all_source_wrappers_retire() {
    let drops = Arc::new(AtomicUsize::new(0));
    let store = MemoryWeightStore::from_buffers([buffer("packed", &drops)]).unwrap();
    let recipe = DerivedWeightRecipe::source(
        "packed",
        TensorSelection::Indices {
            axis: 0,
            indices: vec![2, 0, 2],
        },
    );
    let read = recipe.prepare_encoded_read(&store).unwrap().unwrap();
    let detached = EncodedRecipeRead::prepare_detached(std::iter::once(&read))
        .unwrap()
        .construct(())
        .unwrap();
    drop((read, store));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let mut actual = vec![0; 24];
    detached.read_many_into(&mut [&mut actual]).unwrap();
    assert_eq!(
        actual,
        [5u32, 6, 1, 2, 5, 6]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    drop(detached);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn weak_storage_identity_retains_the_same_original_constructor_control() {
    let drops = Arc::new(AtomicUsize::new(0));
    let store = MemoryWeightStore::from_buffers([buffer("packed", &drops)]).unwrap();
    let mut identity = None;
    store
        .visit_source_storage(&mut |row| identity = Some(row.identity()))
        .unwrap();
    let identity = identity.unwrap();
    assert!(identity.constructor_control_owner::<Custody>().is_some());
    let cloned = identity.clone();
    drop(store);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(identity);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(cloned);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn construction_failures_keep_their_actual_custody_and_publication_prefix() {
    let drops = Arc::new(AtomicUsize::new(0));
    let error = MemoryTensorBuffer::allocate_with_custody(
        "invalid",
        Dtype::U32,
        &[3, 2],
        23,
        Custody(drops.clone()),
    )
    .unwrap_err();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(std::error::Error::source(&error)
        .unwrap()
        .is::<StoreError>());
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 1);

    let drops = Arc::new(AtomicUsize::new(0));
    let error = MemoryWeightStore::from_buffers([
        buffer("duplicate", &drops),
        buffer("duplicate", &drops),
        buffer("unconsumed", &drops),
    ])
    .unwrap_err();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(error.to_string().contains("duplicate in-memory tensor"));
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 3);
}
