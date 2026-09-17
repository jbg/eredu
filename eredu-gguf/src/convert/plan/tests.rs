use super::*;
#[allow(dead_code)]
#[path = "../destination/old.rs"]
mod old;

fn descriptor(ty: GgmlType, blocks: u64) -> TensorDescriptor {
    let (values, bytes) = ty.block_and_bytes().unwrap();
    TensorDescriptor {
        name: "actual.plan.weight".into(),
        dimensions: vec![values, blocks],
        ggml_type: ty,
        relative_offset: 32,
        data_offset: 128,
        byte_len: bytes * blocks,
    }
}
fn payload(d: &TensorDescriptor) -> Vec<u8> {
    (0..d.byte_len as usize)
        .map(|i| (i.wrapping_mul(73).wrapping_add(29) % 256) as u8)
        .collect()
}
fn geometry(tensor: &ConvertedTensor) -> Vec<(Vec<u64>, LogicalDtype)> {
    match tensor {
        ConvertedTensor::Dense(t) => vec![(t.shape.clone(), t.dtype.into())],
        ConvertedTensor::IQuant(t) => vec![(t.packed_shape().unwrap(), LogicalDtype::U8)],
        ConvertedTensor::Affine(t) => vec![
            (t.weight_shape.clone(), LogicalDtype::U32),
            (t.scale_shape.clone(), LogicalDtype::F16),
            (t.scale_shape.clone(), LogicalDtype::F16),
        ],
        ConvertedTensor::MxFp4(t) => vec![
            (t.weight_shape.clone(), LogicalDtype::U32),
            (t.scale_shape.clone(), LogicalDtype::U8),
        ],
    }
}
#[test]
fn cold_plans_match_actual_requested_storage_and_all_independent_conversion_branches() {
    for ty in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Bf16,
        GgmlType::I8,
        GgmlType::I16,
        GgmlType::I32,
        GgmlType::I64,
        GgmlType::F64,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q2K,
        GgmlType::Q3K,
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
        GgmlType::IQ2XXS,
        GgmlType::IQ2XS,
        GgmlType::IQ3XXS,
        GgmlType::IQ1S,
        GgmlType::IQ4NL,
        GgmlType::IQ3S,
        GgmlType::IQ2S,
        GgmlType::IQ4XS,
        GgmlType::IQ1M,
        GgmlType::MxFp4,
    ] {
        for endian in [Endian::Little, Endian::Big] {
            let d = descriptor(ty, 3);
            let plan = ConversionPlan::new(&d, endian).unwrap();
            assert_eq!(plan.descriptor(), &d);
            assert_eq!(plan.endian(), endian);
            assert!(plan.metadata_bytes().unwrap() > std::mem::size_of::<ConversionPlan>());
            let destination = PreparedConversion::prepare(d.clone(), endian).unwrap();
            let request = destination.layouts().unwrap();
            let cold = plan.requested_layouts();
            assert_eq!(cold.vectors, request.vectors);
            assert_eq!(cold.owner, request.owner);
            assert_eq!(cold.descriptor_name, request.descriptor_name);
            assert_eq!(cold.descriptor_dimensions, request.descriptor_dimensions);
            for (request, actual) in plan
                .requested_elements()
                .into_iter()
                .zip(destination.capacities())
            {
                assert!(actual >= request); // actual capacity is not an assumed equality
            }
            let raw = payload(&d);
            let output = destination.convert(&raw).unwrap();
            assert_eq!(
                format!("{output:?}"),
                format!("{:?}", old::convert(&d, &raw, endian).unwrap())
            );
            let planned: Vec<_> = plan
                .outputs()
                .iter()
                .map(|p| (p.shape().to_vec(), p.dtype()))
                .collect();
            assert_eq!(planned, geometry(&output), "{ty:?} {endian:?}");
            if ty == GgmlType::Q8_0 {
                assert_eq!(planned.len(), if endian == Endian::Little { 1 } else { 3 });
            }
            if ty == GgmlType::MxFp4 {
                assert_eq!(
                    planned.iter().map(|p| p.1).collect::<Vec<_>>(),
                    [LogicalDtype::U32, LogicalDtype::U8]
                );
            }
        }
    }
}
#[test]
fn malformed_emission_keeps_distinct_shape_and_reservation_without_changing_converter_failure() {
    for ty in [GgmlType::Q4_0, GgmlType::Q3K, GgmlType::Q8_0] {
        let mut d = descriptor(ty, 3);
        let raw = payload(&d);
        d.dimensions[1] = 1;
        let plan = ConversionPlan::affine(&d, Endian::Big).unwrap();
        assert!(plan.requested_elements()[3] as u64 > plan.outputs()[0].shape_elements().unwrap());
        let prepared = PreparedConversion::prepare_affine(d.clone(), Endian::Big).unwrap();
        assert_eq!(
            plan.requested_layouts().vectors,
            prepared.layouts().unwrap().vectors
        );
        let actual = prepared.convert(&raw).unwrap_err();
        assert_eq!(
            actual.to_string(),
            old::convert_affine(&d, &raw, Endian::Big)
                .unwrap_err()
                .to_string()
        );
        assert!(matches!(
            actual.cause(),
            ConversionDestinationError::Gguf(_)
        ));
    }
    let mut dense = descriptor(GgmlType::F32, 3);
    dense.byte_len = 17;
    let plan = ConversionPlan::new(&dense, Endian::Big).unwrap();
    assert_eq!(plan.requested_elements()[2], 17);
    assert_eq!(plan.outputs()[0].shape_elements(), Some(3));
    let raw = payload(&dense);
    let actual = PreparedConversion::prepare(dense.clone(), Endian::Big)
        .unwrap()
        .convert(&raw)
        .unwrap();
    assert_eq!(
        format!("{actual:?}"),
        format!("{:?}", old::convert(&dense, &raw, Endian::Big).unwrap())
    );
}
#[test]
fn zero_extent_and_overflow_are_metadata_only_and_do_not_claim_capacity() {
    let mut d = descriptor(GgmlType::F32, 0);
    let plan = ConversionPlan::new(&d, Endian::Little).unwrap();
    assert_eq!(plan.outputs()[0].shape_elements(), Some(0));
    assert_eq!(plan.requested_layouts().vectors[2].size(), 0);
    d.byte_len = u64::MAX;
    assert!(matches!(
        ConversionPlan::new(&d, Endian::Little),
        Err(ConversionDestinationError::Layout)
    ));
}
