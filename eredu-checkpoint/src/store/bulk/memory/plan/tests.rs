use super::*;

fn source() -> MemoryWeightStore {
    MemoryWeightStore::from_safetensors([
        ("z".into(), Dtype::U8, vec![2], vec![13, 29]),
        ("é".into(), Dtype::U8, vec![3], vec![3, 7, 11]),
        ("empty".into(), Dtype::U8, vec![0], vec![]),
    ])
    .unwrap()
}

#[test]
fn original_constructor_retains_sources_and_custody_in_occurrence_order() {
    let source = source();
    let alive = source.tensors["é"].ordinary_weak();
    let keys = ["é".into(), "empty".into(), "z".into(), "é".into()];
    let custody = Arc::new(());
    let retained = Arc::downgrade(&custody);
    let read = MemoryEncodedReadPlan::new(&source, &keys)
        .unwrap()
        .construct(custody)
        .unwrap();
    assert_eq!(
        read.tensors()
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["é", "empty", "z", "é"]
    );
    assert_eq!(read.byte_len(), 8);
    drop((source, keys));
    assert!(alive.upgrade().is_some());
    assert!(retained.upgrade().is_some());
    let mut wrong = [99; 7];
    let error = read.read_into(&mut wrong).unwrap_err();
    assert!(matches!(
        error.cause,
        EncodedReadFailureCause::DestinationLengths
    ));
    assert_eq!(wrong, [99; 7]);
    let mut output = [0; 8];
    read.read_into(&mut output).unwrap();
    assert_eq!(output, [3, 7, 11, 13, 29, 3, 7, 11]);
    drop(read);
    assert!(alive.upgrade().is_none());
    assert!(retained.upgrade().is_none());
}

#[test]
fn constructor_inspection_checks_real_source_and_preserves_ordinary_missing_key() {
    let source = source();
    let keys = ["z".into(), "absent".into(), "é".into()];
    assert!(matches!(
        MemoryEncodedReadPlan::new(&source, &keys),
        Err(MemoryEncodedReadPlanError::UnknownTensor { index: 1 })
    ));
    assert!(
        matches!(prepare(&source, &keys), Err(StoreError::UnknownTensor { key }) if key == "absent")
    );
    let empty = MemoryEncodedReadPlan::new(&source, &[])
        .unwrap()
        .construct(())
        .unwrap();
    assert_eq!(empty.byte_len(), 0);
    empty.read_into(&mut []).unwrap();
}

#[test]
fn metadata_admission_does_not_reserve_another_copy_of_payload_bytes() {
    let small =
        MemoryWeightStore::from_safetensors([("x".into(), Dtype::U8, vec![1], vec![3])]).unwrap();
    let large = MemoryWeightStore::from_safetensors([(
        "x".into(),
        Dtype::U8,
        vec![100_000],
        vec![7; 100_000],
    )])
    .unwrap();
    let keys = ["x".into()];
    let small = MemoryEncodedReadPlan::new(&small, &keys).unwrap();
    let large = MemoryEncodedReadPlan::new(&large, &keys).unwrap();
    assert_eq!(small.required_bytes::<()>(), large.required_bytes::<()>());
    assert!(large.required_bytes::<()>().unwrap() < large.byte_len);
}

#[test]
fn ordinary_memory_batch_uses_the_same_original_constructor() {
    let source = source();
    let keys = ["z".into(), "é".into(), "z".into()];
    let read = prepare(&source, &keys).unwrap();
    let mut output = [0; 7];
    read.read_into(&mut output).unwrap();
    assert_eq!(output, [13, 29, 3, 7, 11, 13, 29]);
}
