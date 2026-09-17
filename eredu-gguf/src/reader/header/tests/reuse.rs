use super::*;
use crate::reader::CatalogReader;

fn final_header(bytes: Vec<u8>) -> Arc<PreparedHeader> {
    let (source, _) = Trace::new(bytes);
    let reader =
        Reader::with_header_policy(source, Limits::default(), HeaderPolicy::capture()).unwrap();
    CatalogReader::captured(reader)
        .unwrap()
        .captured_header()
        .unwrap()
        .clone()
}

#[test]
fn prepared_final_headers_borrow_original_owners_and_match_reference_reads() {
    for version in [1, 2, 3] {
        for big in [false, true] {
            let (bytes, _) = fixture(version, big);
            let header = final_header(bytes.clone());
            let weak = Arc::downgrade(&header);
            let metadata = &header.parsed.as_ref().unwrap().metadata;
            let descriptors = &header.parsed.as_ref().unwrap().tensors;
            let mut scratch = vec![0xa5; header.scratch_len()];
            let address = scratch.as_ptr();
            let capacity = scratch.capacity();
            let (source, log) = Trace::new(bytes.clone());
            let mut actual =
                CatalogReader::reopened(source, Limits::default(), &header, &mut scratch).unwrap();
            let (source, reference_log) = Trace::new(bytes);
            let mut old = reference::read(source, Limits::default()).unwrap();
            assert_eq!(*log.borrow(), *reference_log.borrow());
            assert!(std::ptr::eq(actual.metadata(), metadata));
            assert_eq!(actual.tensors().as_ptr(), descriptors.as_ptr());
            assert_eq!(
                actual.tensors()[0].name.as_ptr(),
                descriptors[0].name.as_ptr()
            );
            assert_eq!(
                actual.tensors()[0].dimensions.as_ptr(),
                descriptors[0].dimensions.as_ptr()
            );
            assert_eq!(actual.metadata(), old.metadata());
            let descriptor = old.tensors()[0].clone();
            assert_eq!(
                actual.read_tensor(&descriptor).unwrap(),
                old.read_tensor(&descriptor).unwrap()
            );
            assert_eq!(*log.borrow(), *reference_log.borrow());
            assert_eq!(scratch.as_ptr(), address);
            assert_eq!(scratch.capacity(), capacity);
            drop(header);
            assert!(weak.upgrade().is_some());
            drop(actual);
            assert!(weak.upgrade().is_none());
        }
    }
}

#[test]
fn prepared_final_header_rejects_changed_bytes_and_partial_io_in_original_order() {
    let (bytes, _) = fixture(3, false);
    let header = final_header(bytes.clone());
    let mut scratch = vec![0; header.scratch_len()];
    for (change, fail) in [(24, None), (32, None), (32, Some(34))] {
        let mut changed = bytes.clone();
        changed[change] ^= 1;
        let (mut source, actual_log) = Trace::new(changed.clone());
        source.fail_at = fail;
        let error = CatalogReader::reopened(source, Limits::default(), &header, &mut scratch)
            .err()
            .unwrap();
        let (mut source, old_log) = Trace::new(changed);
        source.fail_at = fail;
        let reference =
            Reader::with_header_policy(source, Limits::default(), HeaderPolicy::compare(&header))
                .err()
                .unwrap();
        assert_eq!(error.to_string(), reference.to_string());
        assert_eq!(*actual_log.borrow(), *old_log.borrow());
        if fail.is_some() {
            assert!(matches!(error, Error::Io { offset: 32, .. }));
        } else {
            assert!(error.prepared_header_change().is_some());
        }
    }
    let (source, _) = Trace::new(bytes);
    assert!(CatalogReader::reopened(source, Limits::default(), &header, &mut scratch).is_ok());
}

#[test]
fn prepared_final_header_scratch_refuses_without_replacement_and_keeps_source() {
    let (bytes, _) = fixture(3, false);
    let header = final_header(bytes.clone());
    let mut short = vec![0xcc; 4];
    let ptr = short.as_ptr();
    let cap = short.capacity();
    let (source, log) = Trace::new(bytes.clone());
    let error = CatalogReader::reopened(source, Limits::default(), &header, &mut short)
        .err()
        .unwrap();
    assert_eq!(
        error.prepared_header_storage(),
        Some("string scratch extent")
    );
    assert_eq!(short, [0xcc; 4]);
    assert_eq!(ptr, short.as_ptr());
    assert_eq!(cap, short.capacity());
    assert_eq!(log.borrow().last().unwrap(), "read 24 8");
    let mut exact = vec![0; header.scratch_len()];
    let (source, _) = Trace::new(bytes);
    assert!(CatalogReader::reopened(source, Limits::default(), &header, &mut exact).is_ok());
}

#[test]
fn prepared_final_header_rechecks_current_extent_and_reads_current_payload() {
    let (bytes, end) = fixture(3, false);
    let header = final_header(bytes.clone());
    let mut scratch = vec![0; header.scratch_len()];
    for length in [end, bytes.len() - 1] {
        let changed = bytes[..length].to_vec();
        let (source, log) = Trace::new(changed.clone());
        let actual = CatalogReader::reopened(source, Limits::default(), &header, &mut scratch)
            .err()
            .unwrap();
        let (source, old_log) = Trace::new(changed);
        let old = reference::read(source, Limits::default()).err().unwrap();
        assert_eq!(actual.to_string(), old.to_string());
        assert_eq!(*log.borrow(), *old_log.borrow());
        assert!(actual.prepared_header_change().is_none());
    }
    let mut changed = bytes;
    let len = changed.len();
    changed[len - 4..].copy_from_slice(&(-9.25f32).to_le_bytes());
    let (source, _) = Trace::new(changed.clone());
    let mut actual =
        CatalogReader::reopened(source, Limits::default(), &header, &mut scratch).unwrap();
    let (source, _) = Trace::new(changed);
    let mut old = reference::read(source, Limits::default()).unwrap();
    let descriptor = old.tensors()[0].clone();
    assert_eq!(
        actual.read_tensor(&descriptor).unwrap(),
        old.read_tensor(&descriptor).unwrap()
    );
}

fn same_value(actual: &MetadataValue, expected: &MetadataValue) {
    match (actual, expected) {
        (MetadataValue::Float32(a), MetadataValue::Float32(b)) => {
            assert_eq!(a.to_bits(), b.to_bits())
        }
        (MetadataValue::Float64(a), MetadataValue::Float64(b)) => {
            assert_eq!(a.to_bits(), b.to_bits())
        }
        (MetadataValue::Array(a), MetadataValue::Array(b)) => same_array(a, b),
        _ => assert_eq!(actual, expected),
    }
}
fn same_array(actual: &MetadataArray, expected: &MetadataArray) {
    match (actual, expected) {
        (MetadataArray::Float32(a), MetadataArray::Float32(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
        (MetadataArray::Float64(a), MetadataArray::Float64(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
        (MetadataArray::Array(a), MetadataArray::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                same_array(a, b);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

#[test]
fn prepared_final_header_walks_all_array_variants_without_float_equality() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("arrays.gguf");
    let arrays = vec![
        MetadataArray::Uint8(vec![1, 2]),
        MetadataArray::Int8(vec![-1, 2]),
        MetadataArray::Uint16(vec![1, 2]),
        MetadataArray::Int16(vec![-1, 2]),
        MetadataArray::Uint32(vec![1, 2]),
        MetadataArray::Int32(vec![-1, 2]),
        MetadataArray::Float32(vec![f32::from_bits(0x7fc00042), -0.0]),
        MetadataArray::Bool(vec![false, true]),
        MetadataArray::String(vec!["".into(), "é".into()]),
        MetadataArray::Array(vec![
            MetadataArray::String(vec!["nested".into()]),
            MetadataArray::Float64(vec![f64::from_bits(0x7ff8000000000042), -0.0]),
        ]),
        MetadataArray::Uint64(vec![1, 2]),
        MetadataArray::Int64(vec![-1, 2]),
        MetadataArray::Float64(vec![1.25, -0.0]),
    ];
    let mut metadata: BTreeMap<_, _> = arrays
        .into_iter()
        .enumerate()
        .map(|(i, a)| (format!("array.{i}"), MetadataValue::Array(a)))
        .collect();
    metadata.insert(
        "nan".into(),
        MetadataValue::Float32(f32::from_bits(0x7fc00042)),
    );
    crate::Writer::default()
        .write(File::create(&path).unwrap(), &metadata, &[])
        .unwrap();
    let bytes = std::fs::read(path).unwrap();
    let header = final_header(bytes.clone());
    let mut scratch = vec![0; header.scratch_len()];
    let (source, actual_log) = Trace::new(bytes.clone());
    let actual = CatalogReader::reopened(source, Limits::default(), &header, &mut scratch).unwrap();
    let (source, old_log) = Trace::new(bytes);
    let old = reference::read(source, Limits::default()).unwrap();
    assert_eq!(*actual_log.borrow(), *old_log.borrow());
    assert_eq!(actual.metadata().len(), old.metadata().len());
    for ((key, actual), (old_key, expected)) in actual.metadata().iter().zip(old.metadata()) {
        assert_eq!(key, old_key);
        same_value(actual, expected);
    }
    assert!(std::ptr::eq(
        actual.metadata(),
        &header.parsed.as_ref().unwrap().metadata
    ));
    match actual.metadata().get("nan").unwrap() {
        MetadataValue::Float32(v) => assert_eq!(v.to_bits(), 0x7fc00042),
        _ => panic!("actual scalar"),
    }
    assert!(actual.tensors().is_empty());
}
