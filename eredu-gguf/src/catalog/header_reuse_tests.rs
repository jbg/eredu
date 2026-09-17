use super::*;
use crate::{GgmlType, TensorInput, Writer};

#[test]
fn cloned_prepared_materializers_and_iterators_own_independent_scratch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source.gguf");
    let metadata = BTreeMap::from([(
        "long-key".into(),
        MetadataValue::String("retained source".into()),
    )]);
    let data = [1.25_f32, -3.5, 7.0, 9.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    Writer::default()
        .write(
            File::create(&path).unwrap(),
            &metadata,
            &[TensorInput {
                name: "weight",
                dimensions: &[4],
                ggml_type: GgmlType::F32,
                data: &data,
            }],
        )
        .unwrap();
    let checkpoint = Checkpoint::open_with_prepared_headers(&path).unwrap();
    let mut a = checkpoint.materializer();
    let mut b = checkpoint.clone().into_materializer();
    let mut iter = checkpoint.converted_tensors();
    let header = checkpoint.shards()[0].prepared_header().unwrap();
    assert!(std::ptr::eq(
        header,
        a.shards()[0].prepared_header().unwrap()
    ));
    assert!(std::ptr::eq(
        header,
        b.shards()[0].prepared_header().unwrap()
    ));
    assert!(a.header_scratch.len() > 0);
    assert_ne!(a.header_scratch.as_ptr(), b.header_scratch.as_ptr());
    assert_ne!(a.header_scratch.as_ptr(), iter.header_scratch.as_ptr());
    let addresses = (
        a.header_scratch.as_ptr(),
        b.header_scratch.as_ptr(),
        iter.header_scratch.as_ptr(),
    );
    let ordinary = Checkpoint::open(&path)
        .unwrap()
        .into_materializer()
        .converted_tensor("weight")
        .unwrap();
    assert_eq!(a.converted_tensor("weight").unwrap(), ordinary);
    assert_eq!(b.converted_tensor("weight").unwrap(), ordinary);
    assert_eq!(iter.next().unwrap().unwrap(), ordinary);
    assert!(a.close_reader_without_path());
    assert!(b.close_reader_without_path());
    assert_eq!(a.converted_tensor("weight").unwrap(), ordinary);
    assert_eq!(b.converted_tensor("weight").unwrap(), ordinary);
    assert_eq!(
        addresses,
        (
            a.header_scratch.as_ptr(),
            b.header_scratch.as_ptr(),
            iter.header_scratch.as_ptr()
        )
    );
    assert!(iter.next().is_none());
    // Header source ownership survives independent checkpoint/iterator retirement.
    drop(iter);
    drop(checkpoint);
    drop(a);
    assert_eq!(b.converted_tensor("weight").unwrap(), ordinary);
}
