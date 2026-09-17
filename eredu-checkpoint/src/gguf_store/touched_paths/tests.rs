use super::*;
use eredu_gguf::{GgmlType, TensorInput, Writer};
use std::fs::File;

fn file(path: &Path) {
    Writer::default()
        .write(
            File::create(path).unwrap(),
            &BTreeMap::new(),
            &[TensorInput {
                name: "matrix.weight",
                dimensions: &[2],
                ggml_type: GgmlType::F32,
                data: &[1.25_f32.to_le_bytes(), (-3.5_f32).to_le_bytes()].concat(),
            }],
        )
        .unwrap();
}

#[test]
fn actual_shard_loans_preserve_catalog_addresses_and_lookup_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.gguf");
    file(&path);
    let checkpoint = Checkpoint::open(&path).unwrap();
    let address = checkpoint.shards().as_ptr();
    let materializer = checkpoint.into_materializer();
    assert_eq!(materializer.shards().as_ptr(), address);
    let (index, actual) = materializer
        .shard_source_for_tensor("matrix.weight")
        .unwrap();
    assert_eq!(index, 0);
    assert!(std::ptr::eq(actual, materializer.shards()[index].path()));
    assert!(std::ptr::eq(
        actual,
        materializer.shard_path_for_tensor("matrix.weight").unwrap()
    ));
    assert!(materializer.open_shard_path().is_none());
    let expected = eredu_gguf::Error::InvalidTensor {
        tensor: "missing".into(),
        reason: "tensor is not present in the checkpoint".into(),
    };
    assert_eq!(
        materializer
            .shard_source_for_tensor("missing")
            .unwrap_err()
            .to_string(),
        expected.to_string()
    );
    assert_eq!(
        materializer
            .shard_path_for_tensor("missing")
            .unwrap_err()
            .to_string(),
        expected.to_string()
    );
}

#[test]
fn finite_coordinates_preserve_sparse_order_exact_lengths_and_first_spelling() {
    let dir = tempfile::tempdir().unwrap();
    let z = dir.path().join("z.gguf");
    let a = dir.path().join("a.gguf");
    let alias = dir.path().join(".").join("a.gguf");
    file(&z);
    file(&a);
    let materializers: Vec<_> = [&z, &a, &alias]
        .into_iter()
        .map(|p| Checkpoint::open(p).unwrap().into_materializer())
        .collect();
    let mut table = TouchedPaths::new(&materializers).unwrap();
    assert_eq!(table.requested, Layout::array::<TouchedEntry>(3).unwrap());
    assert_eq!(
        table.capacity_layout().unwrap(),
        Layout::array::<TouchedEntry>(table.entries.capacity()).unwrap()
    );
    assert_eq!(table.entries.len(), 2);
    assert!(table.entries.capacity() >= 3);
    let storage = table.entries.as_ptr();
    let mut expected = BTreeSet::new();
    for coordinate in [
        ShardCoordinate {
            checkpoint: 0,
            shard: 0,
        },
        ShardCoordinate {
            checkpoint: 2,
            shard: 0,
        },
        ShardCoordinate {
            checkpoint: 1,
            shard: 0,
        },
    ] {
        let path = coordinate.path(&materializers);
        let index = table.find(&materializers, path);
        table.mark(index, coordinate);
        expected.insert(path.to_path_buf());
        assert_eq!(table.entries.as_ptr(), storage);
        let mut actual = table.paths(&materializers);
        assert_eq!(actual.len(), expected.len());
        for (remaining, expected) in expected.iter().enumerate() {
            assert_eq!(actual.next().unwrap().as_os_str(), expected.as_os_str());
            assert_eq!(actual.len(), table.touched - remaining - 1);
        }
        assert!(actual.next().is_none());
        assert!(actual.next().is_none());
    }
    assert_eq!(
        table.paths(&materializers).next().unwrap().as_os_str(),
        alias.as_os_str()
    );
}

#[test]
fn shared_source_prices_actual_capacity_and_keeps_table_through_failed_and_prepared_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("actual.gguf");
    file(&path);
    let source = super::super::tests::test_store(&path);
    let request = || TensorReadRequest {
        key: "matrix.weight".into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    };
    let (storage, payload) = {
        let cache = source.inner.readers.lock().unwrap();
        (
            cache.touched.entries.as_ptr(),
            cache.touched.capacity_layout().unwrap().size() as u64,
        )
    };
    assert!(payload > 0);
    assert_eq!(source.inner.touched_storage_bytes, payload);
    assert_eq!(
        source.source_storage().unwrap().unwrap().bytes().unwrap(),
        source.reader_storage_bytes().unwrap() + payload
    );
    let ordinary = source.acquire(request()).unwrap();
    let failed = source
        .acquire(request())
        .unwrap()
        .prepare_portable_read()
        .unwrap()
        .prepare_conversion()
        .unwrap()
        .prepare_result_metadata()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    let failure = failed.materialize().unwrap_err();
    assert!(failure.store_error().is_some());
    assert!(failure.lease().store.same(&source.inner));
    {
        let cache = source.inner.readers.lock().unwrap();
        assert_eq!(cache.touched.entries.as_ptr(), storage);
        assert_eq!(cache.touched.paths(&cache.materializers).len(), 0);
    }
    file(&path);
    let expected = ordinary.materialize_portable().unwrap();
    let actual = source
        .clone()
        .acquire(request())
        .unwrap()
        .prepare_portable_read()
        .unwrap()
        .prepare_conversion()
        .unwrap()
        .prepare_result_metadata()
        .unwrap()
        .materialize()
        .unwrap();
    assert_eq!(actual, expected);
    let cache = source.inner.readers.lock().unwrap();
    assert_eq!(cache.touched.entries.as_ptr(), storage);
    assert_eq!(cache.touched.paths(&cache.materializers).len(), 1);
    assert_eq!(
        cache
            .touched
            .paths(&cache.materializers)
            .next()
            .unwrap()
            .as_os_str(),
        path.as_os_str()
    );
}
