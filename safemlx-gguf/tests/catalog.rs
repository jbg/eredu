use safemlx_gguf::{
    Checkpoint, ConvertedTensor, DenseTensorSpan, DenseTensorSpanPlan, Endian, Error, GgmlType,
    LogicalDtype, MetadataValue, TensorDescriptor, TensorInput, TensorSelection,
    TensorSelectionPlan, Writer, WriterOptions,
};
use std::collections::BTreeMap;
use std::path::Path;

struct FixtureTensor<'a> {
    name: &'a str,
    dimensions: &'a [u64],
    ty: GgmlType,
    data: &'a [u8],
}

#[test]
fn selects_dense_outer_ranges_and_reordered_indices_without_full_conversion() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("selected.gguf");
    let values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    write_file(
        &path,
        None,
        "selected",
        &[FixtureTensor {
            name: "matrix.weight",
            dimensions: &[3, 4],
            ty: GgmlType::F32,
            data: &bytes,
        }],
    );

    let checkpoint = Checkpoint::open(path).unwrap();
    let mut materializer = checkpoint.materializer();
    let range = materializer
        .converted_tensor_selected(
            "matrix.weight",
            &TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 3,
            },
        )
        .unwrap();
    let ConvertedTensor::Dense(range) = range.converted() else {
        panic!("expected dense selection");
    };
    assert_eq!(range.shape, [2, 3]);
    assert_eq!(range.data, bytes[12..36]);

    let reordered = materializer
        .converted_tensor_selected(
            "matrix.weight",
            &TensorSelection::Indices {
                axis: 0,
                indices: vec![3, 0, 2],
            },
        )
        .unwrap();
    let ConvertedTensor::Dense(reordered) = reordered.converted() else {
        panic!("expected dense selection");
    };
    assert_eq!(reordered.shape, [3, 3]);
    let expected = [
        bytes[36..48].to_vec(),
        bytes[0..12].to_vec(),
        bytes[24..36].to_vec(),
    ]
    .concat();
    assert_eq!(reordered.data, expected);
}

#[test]
fn reads_one_reshaped_contiguous_span_from_a_dense_bank() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dense-span.gguf");
    let values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    write_file(
        &path,
        None,
        "dense contiguous span",
        &[FixtureTensor {
            name: "bank.weight",
            dimensions: &[4, 3, 2],
            ty: GgmlType::F32,
            data: &bytes,
        }],
    );

    let checkpoint = Checkpoint::open(path).unwrap();
    let descriptor = checkpoint.shards()[0].tensors()[0].descriptor();
    let selection = DenseTensorSpan::new(8, vec![1, 2, 4]).unwrap();
    let plan = DenseTensorSpanPlan::new(descriptor, selection.clone()).unwrap();
    assert_eq!(plan.selected_descriptor().mlx_shape(), [1, 2, 4]);
    assert_eq!(plan.encoded_byte_len(), 32);
    assert_eq!(plan.encoded_span().offset(), descriptor.data_offset + 8 * 4);

    let selected = checkpoint
        .materializer()
        .converted_dense_tensor_span("bank.weight", &selection)
        .unwrap();
    let ConvertedTensor::Dense(selected) = selected.converted() else {
        panic!("expected dense selection");
    };
    assert_eq!(selected.shape, [1, 2, 4]);
    let selected = selected
        .data
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(selected, values[8..16]);
}

#[test]
fn dense_span_type_rejects_packed_and_out_of_bounds_sources() {
    let dense = TensorDescriptor {
        name: "dense.weight".into(),
        dimensions: vec![4, 3, 2],
        ggml_type: GgmlType::Bf16,
        relative_offset: 0,
        data_offset: 128,
        byte_len: 48,
    };
    assert!(
        DenseTensorSpanPlan::new(&dense, DenseTensorSpan::new(20, vec![1, 2, 4]).unwrap()).is_err()
    );
    let mut malformed = dense.clone();
    malformed.byte_len -= 1;
    assert!(
        DenseTensorSpanPlan::new(&malformed, DenseTensorSpan::new(0, vec![1, 2, 4]).unwrap())
            .is_err()
    );

    let packed = TensorDescriptor {
        name: "packed.weight".into(),
        dimensions: vec![32, 2],
        ggml_type: GgmlType::Q4_0,
        relative_offset: 0,
        data_offset: 128,
        byte_len: 36,
    };
    let error = DenseTensorSpanPlan::new(&packed, DenseTensorSpan::new(0, vec![1, 32]).unwrap())
        .unwrap_err();
    assert!(error.to_string().contains("unquantized F32, F16, or BF16"));
    assert!(DenseTensorSpan::new(0, vec![]).is_err());
    assert!(DenseTensorSpan::new(0, vec![1, 0, 4]).is_err());
}

#[test]
fn selects_affine_outer_rows_equal_to_slicing_the_converted_group() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("selected-affine.gguf");
    let first = [0u8; 18];
    let second = [0xffu8; 18];
    let bytes = [first.as_slice(), second.as_slice()].concat();
    write_file(
        &path,
        None,
        "selected affine",
        &[FixtureTensor {
            name: "experts.weight",
            dimensions: &[32, 2],
            ty: GgmlType::Q4_0,
            data: &bytes,
        }],
    );

    let checkpoint = Checkpoint::open(path).unwrap();
    let mut materializer = checkpoint.materializer();
    let full = materializer.converted_tensor("experts.weight").unwrap();
    let selected = materializer
        .converted_tensor_selected(
            "experts.weight",
            &TensorSelection::Indices {
                axis: 0,
                indices: vec![1],
            },
        )
        .unwrap();
    let ConvertedTensor::Affine(full) = full.converted() else {
        panic!("expected affine tensor");
    };
    let ConvertedTensor::Affine(selected) = selected.converted() else {
        panic!("expected affine tensor");
    };
    assert_eq!(selected.weight_shape, [1, 4]);
    assert_eq!(selected.scale_shape, [1, 1]);
    assert_eq!(selected.weights, full.weights[4..8]);
    assert_eq!(selected.scales, full.scales[1..2]);
    assert_eq!(selected.biases, full.biases[1..2]);
}

#[test]
fn plans_and_reads_dense_inner_axis_ranges_as_strided_spans() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("selected-inner.gguf");
    let values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    write_file(
        &path,
        None,
        "selected inner",
        &[FixtureTensor {
            name: "matrix.weight",
            dimensions: &[3, 4],
            ty: GgmlType::F32,
            data: &bytes,
        }],
    );

    let checkpoint = Checkpoint::open(path).unwrap();
    let descriptor = checkpoint.shards()[0].tensors()[0].descriptor();
    let selection = TensorSelection::Range {
        axis: 1,
        start: 1,
        end: 3,
    };
    let plan = TensorSelectionPlan::new(descriptor, selection.clone()).unwrap();
    assert_eq!(plan.logical_axis(), 1);
    assert_eq!(plan.gguf_dimension(), 0);
    assert_eq!(plan.alignment().block_values(), 1);
    assert_eq!(plan.alignment().block_bytes(), 4);
    assert_eq!(plan.alignment().selected_axis_multiple(), 1);
    assert_eq!(plan.selected_descriptor().mlx_shape(), [4, 2]);
    assert_eq!(plan.encoded_byte_len(), 32);
    let spans = plan.encoded_spans().collect::<Vec<_>>();
    assert_eq!(spans.len(), 4);
    assert_eq!(spans[0].offset(), descriptor.data_offset + 4);
    assert_eq!(spans[0].byte_len(), 8);
    assert_eq!(spans[3].offset(), descriptor.data_offset + 40);

    let selected = checkpoint
        .materializer()
        .converted_tensor_selected("matrix.weight", &selection)
        .unwrap();
    let ConvertedTensor::Dense(selected) = selected.converted() else {
        panic!("expected dense selection");
    };
    assert_eq!(selected.shape, [4, 2]);
    let selected = selected
        .data
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(selected, [1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0]);
}

#[test]
fn reads_reordered_dense_indices_on_an_intermediate_axis() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("selected-intermediate.gguf");
    let values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    write_file(
        &path,
        None,
        "selected intermediate",
        &[FixtureTensor {
            name: "bank.weight",
            dimensions: &[2, 3, 2],
            ty: GgmlType::F32,
            data: &bytes,
        }],
    );

    let checkpoint = Checkpoint::open(path).unwrap();
    let selection = TensorSelection::Indices {
        axis: 1,
        indices: vec![2, 0],
    };
    let descriptor = checkpoint.shards()[0].tensors()[0].descriptor();
    let plan = TensorSelectionPlan::new(descriptor, selection.clone()).unwrap();
    assert_eq!(plan.gguf_dimension(), 1);
    assert_eq!(plan.selected_descriptor().mlx_shape(), [2, 2, 2]);
    assert_eq!(plan.encoded_spans().count(), 4);

    let selected = checkpoint
        .materializer()
        .converted_tensor_selected("bank.weight", &selection)
        .unwrap();
    let ConvertedTensor::Dense(selected) = selected.converted() else {
        panic!("expected dense selection");
    };
    assert_eq!(selected.shape, [2, 2, 2]);
    let selected = selected
        .data
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(selected, [4.0, 5.0, 0.0, 1.0, 10.0, 11.0, 6.0, 7.0]);
}

#[test]
fn reads_block_aligned_affine_inner_ranges_without_other_blocks() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("selected-affine-inner.gguf");
    let blocks = [[0x00u8; 18], [0x11u8; 18], [0x22u8; 18], [0x33u8; 18]];
    let bytes = blocks.into_iter().flatten().collect::<Vec<_>>();
    write_file(
        &path,
        None,
        "selected affine inner",
        &[FixtureTensor {
            name: "matrix.weight",
            dimensions: &[64, 2],
            ty: GgmlType::Q4_0,
            data: &bytes,
        }],
    );

    let checkpoint = Checkpoint::open(path).unwrap();
    let mut materializer = checkpoint.materializer();
    let full = materializer.converted_tensor("matrix.weight").unwrap();
    let selection = TensorSelection::Range {
        axis: 1,
        start: 32,
        end: 64,
    };
    let descriptor = checkpoint.shards()[0].tensors()[0].descriptor();
    let plan = TensorSelectionPlan::new(descriptor, selection.clone()).unwrap();
    assert_eq!(plan.alignment().selected_axis_multiple(), 32);
    assert_eq!(plan.encoded_byte_len(), 36);
    assert_eq!(plan.encoded_spans().count(), 2);

    let selected = materializer
        .converted_tensor_selected("matrix.weight", &selection)
        .unwrap();
    let ConvertedTensor::Affine(full) = full.converted() else {
        panic!("expected affine tensor");
    };
    let ConvertedTensor::Affine(selected) = selected.converted() else {
        panic!("expected affine tensor");
    };
    assert_eq!(selected.weight_shape, [2, 4]);
    assert_eq!(selected.scale_shape, [2, 1]);
    assert_eq!(
        selected.weights,
        [full.weights[4..8].to_vec(), full.weights[12..16].to_vec()].concat()
    );
    assert_eq!(selected.scales, [full.scales[1], full.scales[3]]);
    assert_eq!(selected.biases, [full.biases[1], full.biases[3]]);
}

#[test]
fn reads_reordered_complete_quantization_blocks() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("selected-affine-blocks.gguf");
    let blocks = [[0x00u8; 18], [0x11u8; 18], [0x22u8; 18], [0x33u8; 18]];
    let bytes = blocks.into_iter().flatten().collect::<Vec<_>>();
    write_file(
        &path,
        None,
        "selected affine blocks",
        &[FixtureTensor {
            name: "matrix.weight",
            dimensions: &[64, 2],
            ty: GgmlType::Q4_0,
            data: &bytes,
        }],
    );

    let checkpoint = Checkpoint::open(path).unwrap();
    let mut materializer = checkpoint.materializer();
    let full = materializer.converted_tensor("matrix.weight").unwrap();
    let indices = (32..64).chain(0..32).collect::<Vec<_>>();
    let selected = materializer
        .converted_tensor_selected(
            "matrix.weight",
            &TensorSelection::Indices { axis: 1, indices },
        )
        .unwrap();
    let ConvertedTensor::Affine(full) = full.converted() else {
        panic!("expected affine tensor");
    };
    let ConvertedTensor::Affine(selected) = selected.converted() else {
        panic!("expected affine tensor");
    };
    assert_eq!(selected.weight_shape, [2, 8]);
    assert_eq!(selected.scale_shape, [2, 2]);
    assert_eq!(
        selected.weights,
        [
            full.weights[4..8].to_vec(),
            full.weights[0..4].to_vec(),
            full.weights[12..16].to_vec(),
            full.weights[8..12].to_vec(),
        ]
        .concat()
    );
    assert_eq!(
        selected.scales,
        [
            full.scales[1],
            full.scales[0],
            full.scales[3],
            full.scales[2],
        ]
    );
    assert_eq!(
        selected.biases,
        [
            full.biases[1],
            full.biases[0],
            full.biases[3],
            full.biases[2],
        ]
    );
}

#[test]
fn every_block_encoding_exposes_its_inner_axis_alignment() {
    let encodings = [
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
    ];
    for encoding in encodings {
        let (block_values, block_bytes) = encoding.block_and_bytes().unwrap();
        let descriptor = TensorDescriptor {
            name: format!("{encoding:?}"),
            dimensions: vec![block_values * 2, 3],
            ggml_type: encoding,
            relative_offset: 0,
            data_offset: 4096,
            byte_len: block_bytes * 6,
        };
        let plan = TensorSelectionPlan::new(
            &descriptor,
            TensorSelection::Range {
                axis: 1,
                start: block_values as usize,
                end: (block_values * 2) as usize,
            },
        )
        .unwrap();
        assert_eq!(plan.alignment().block_values(), block_values);
        assert_eq!(plan.alignment().block_bytes(), block_bytes);
        assert_eq!(plan.alignment().selected_axis_multiple(), block_values);
        assert_eq!(plan.encoded_byte_len(), block_bytes * 3);
        assert_eq!(plan.encoded_spans().count(), 3);
    }
}

#[test]
fn rejects_partial_or_scrambled_native_blocks() {
    let descriptor = TensorDescriptor {
        name: "quant.weight".into(),
        dimensions: vec![64, 2],
        ggml_type: GgmlType::Q4_0,
        relative_offset: 0,
        data_offset: 1024,
        byte_len: 72,
    };
    assert!(TensorSelectionPlan::new(
        &descriptor,
        TensorSelection::Range {
            axis: 1,
            start: 1,
            end: 33,
        },
    )
    .is_err());
    assert!(TensorSelectionPlan::new(
        &descriptor,
        TensorSelection::Indices {
            axis: 1,
            indices: (1..33).collect(),
        },
    )
    .is_err());
    let mut scrambled = (0..32).collect::<Vec<_>>();
    scrambled.swap(0, 1);
    assert!(TensorSelectionPlan::new(
        &descriptor,
        TensorSelection::Indices {
            axis: 1,
            indices: scrambled,
        },
    )
    .is_err());
    let mut malformed = descriptor.clone();
    malformed.byte_len -= 1;
    assert!(TensorSelectionPlan::new(
        &malformed,
        TensorSelection::Range {
            axis: 0,
            start: 0,
            end: 1,
        },
    )
    .is_err());
    assert!(TensorSelectionPlan::new(
        &descriptor,
        TensorSelection::Range {
            axis: 2,
            start: 0,
            end: 1,
        },
    )
    .is_err());
}

#[test]
fn rejects_invalid_selections() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("invalid-selection.gguf");
    let bytes = [0u8; 16];
    write_file(
        &path,
        None,
        "invalid selection",
        &[FixtureTensor {
            name: "matrix.weight",
            dimensions: &[2, 2],
            ty: GgmlType::F32,
            data: &bytes,
        }],
    );
    let checkpoint = Checkpoint::open(path).unwrap();
    let mut materializer = checkpoint.materializer();
    assert!(materializer
        .converted_tensor_selected(
            "matrix.weight",
            &TensorSelection::Range {
                axis: 0,
                start: 2,
                end: 1,
            },
        )
        .is_err());
    assert!(materializer
        .converted_tensor_selected(
            "matrix.weight",
            &TensorSelection::Indices {
                axis: 0,
                indices: vec![],
            },
        )
        .is_err());
    assert!(materializer
        .converted_tensor_selected(
            "matrix.weight",
            &TensorSelection::Indices {
                axis: 0,
                indices: vec![2],
            },
        )
        .is_err());
}

#[test]
fn selects_from_a_big_endian_tensor_in_a_noninitial_shard() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("model-00001-of-00002.gguf");
    let second = directory.path().join("model-00002-of-00002.gguf");
    write_file(
        &first,
        Some((0, 2, 2)),
        "first",
        &[FixtureTensor {
            name: "first.weight",
            dimensions: &[1],
            ty: GgmlType::F32,
            data: &0f32.to_le_bytes(),
        }],
    );
    let metadata = BTreeMap::from([
        (
            "general.name".into(),
            MetadataValue::String("second".into()),
        ),
        ("split.no".into(), MetadataValue::Uint16(1)),
        ("split.count".into(), MetadataValue::Uint16(2)),
        ("split.tensors.count".into(), MetadataValue::Uint16(2)),
    ]);
    let values = [1f32, 2.0, 3.0, 4.0];
    let bytes = values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect::<Vec<_>>();
    Writer::new(WriterOptions {
        endian: Endian::Big,
        ..WriterOptions::default()
    })
    .unwrap()
    .write(
        std::fs::File::create(second).unwrap(),
        &metadata,
        &[TensorInput {
            name: "second.weight",
            dimensions: &[2, 2],
            ggml_type: GgmlType::F32,
            data: &bytes,
        }],
    )
    .unwrap();

    // Shards must agree on endianness, so rewrite the first shard as big-endian too.
    let first_metadata = BTreeMap::from([
        ("general.name".into(), MetadataValue::String("first".into())),
        ("split.no".into(), MetadataValue::Uint16(0)),
        ("split.count".into(), MetadataValue::Uint16(2)),
        ("split.tensors.count".into(), MetadataValue::Uint16(2)),
    ]);
    Writer::new(WriterOptions {
        endian: Endian::Big,
        ..WriterOptions::default()
    })
    .unwrap()
    .write(
        std::fs::File::create(&first).unwrap(),
        &first_metadata,
        &[TensorInput {
            name: "first.weight",
            dimensions: &[1],
            ggml_type: GgmlType::F32,
            data: &0f32.to_be_bytes(),
        }],
    )
    .unwrap();

    let checkpoint = Checkpoint::open(first).unwrap();
    let selected = checkpoint
        .materializer()
        .converted_tensor_selected(
            "second.weight",
            &TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 2,
            },
        )
        .unwrap();
    let ConvertedTensor::Dense(selected) = selected.converted() else {
        panic!("expected dense selection");
    };
    let selected = selected
        .data
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(selected, [3.0, 4.0]);
}

fn write_file(
    path: &Path,
    split: Option<(u16, u16, u16)>,
    general_name: &str,
    tensors: &[FixtureTensor<'_>],
) {
    let mut metadata = BTreeMap::from([(
        "general.name".to_string(),
        MetadataValue::String(general_name.to_string()),
    )]);
    if let Some((split_no, split_count, tensor_count)) = split {
        metadata.extend([
            ("split.no".to_string(), MetadataValue::Uint16(split_no)),
            (
                "split.count".to_string(),
                MetadataValue::Uint16(split_count),
            ),
            (
                "split.tensors.count".to_string(),
                MetadataValue::Uint16(tensor_count),
            ),
        ]);
    }
    let inputs = tensors
        .iter()
        .map(|tensor| TensorInput {
            name: tensor.name,
            dimensions: tensor.dimensions,
            ggml_type: tensor.ty,
            data: tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(std::fs::File::create(path).unwrap(), &metadata, &inputs)
        .unwrap();
}

#[test]
fn catalogs_all_shards_and_plans_logical_outputs() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("model-00001-of-00002.gguf");
    let second = directory.path().join("model-00002-of-00002.gguf");
    let quantized = [0u8; 36];
    let dense = [0u8; 24];
    write_file(
        &first,
        Some((0, 2, 2)),
        "first metadata",
        &[FixtureTensor {
            name: "blk.0.attn_q.weight",
            dimensions: &[64],
            ty: GgmlType::Q4_0,
            data: &quantized,
        }],
    );
    write_file(
        &second,
        Some((1, 2, 2)),
        "ignored metadata",
        &[FixtureTensor {
            name: "output.weight",
            dimensions: &[3, 2],
            ty: GgmlType::F32,
            data: &dense,
        }],
    );

    let catalog = Checkpoint::open(&first).unwrap();
    assert_eq!(catalog.shards().len(), 2);
    assert_eq!(catalog.physical_tensor_count(), 2);
    assert_eq!(catalog.shards()[0].split_no(), 0);
    assert_eq!(catalog.shards()[1].split_no(), 1);
    assert_eq!(catalog.shards()[1].path(), second.as_path());
    assert_eq!(catalog.shards()[0].version(), 3);
    assert_eq!(catalog.shards()[0].alignment(), 32);
    assert_eq!(
        catalog.metadata()["general.name"].as_str(),
        Some("first metadata")
    );

    let quantized = &catalog.shards()[0].tensors()[0];
    assert_eq!(quantized.affine(), Some((4, 32)));
    assert_eq!(quantized.outputs().len(), 3);
    assert_eq!(quantized.outputs()[0].shape, [8]);
    assert_eq!(quantized.outputs()[0].dtype, LogicalDtype::U32);
    assert_eq!(quantized.outputs()[1].name, "blk.0.attn_q.scales");
    assert_eq!(quantized.outputs()[1].shape, [2]);
    assert_eq!(quantized.outputs()[2].name, "blk.0.attn_q.biases");

    let dense = &catalog.shards()[1].tensors()[0];
    assert_eq!(dense.affine(), None);
    assert_eq!(dense.outputs()[0].shape, [2, 3]);
    assert_eq!(dense.outputs()[0].dtype, LogicalDtype::F32);
    assert_eq!(catalog.logical_outputs().count(), 4);

    let translated = catalog
        .translated_outputs(|name| format!("model.{name}"))
        .unwrap();
    assert_eq!(translated[1].physical_name, "blk.0.attn_q.weight");
    assert_eq!(translated[1].original_name, "blk.0.attn_q.scales");
    assert_eq!(translated[1].layout.name, "model.blk.0.attn_q.scales");
}

#[test]
fn streams_one_physical_tensor_group_at_a_time_across_shards() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("model-00001-of-00002.gguf");
    let second = directory.path().join("model-00002-of-00002.gguf");
    let quantized = [0u8; 18];
    let dense = 7.0f32.to_le_bytes();
    write_file(
        &first,
        Some((0, 2, 2)),
        "first",
        &[FixtureTensor {
            name: "packed.weight",
            dimensions: &[32],
            ty: GgmlType::Q4_0,
            data: &quantized,
        }],
    );
    write_file(
        &second,
        Some((1, 2, 2)),
        "second",
        &[FixtureTensor {
            name: "dense.weight",
            dimensions: &[1],
            ty: GgmlType::F32,
            data: &dense,
        }],
    );

    let checkpoint = Checkpoint::open(first).unwrap();
    let mut tensors = checkpoint.converted_tensors();
    let packed = tensors.next().unwrap().unwrap();
    assert_eq!(packed.shard_index(), 0);
    assert_eq!(packed.tensor_index(), 0);
    assert_eq!(packed.descriptor().name, "packed.weight");
    assert!(matches!(packed.converted(), ConvertedTensor::Affine(_)));

    let dense = tensors.next().unwrap().unwrap();
    assert_eq!(dense.shard_index(), 1);
    assert_eq!(dense.tensor_index(), 0);
    assert_eq!(dense.descriptor().name, "dense.weight");
    assert!(matches!(dense.converted(), ConvertedTensor::Dense(_)));
    assert!(tensors.next().is_none());
}

#[test]
fn callback_receives_affine_outputs_as_one_atomic_group() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.gguf");
    let quantized = [0u8; 18];
    write_file(
        &path,
        None,
        "packed",
        &[FixtureTensor {
            name: "packed.weight",
            dimensions: &[32],
            ty: GgmlType::Q4_0,
            data: &quantized,
        }],
    );

    let checkpoint = Checkpoint::open(path).unwrap();
    let mut visits = 0;
    checkpoint
        .for_each_converted_tensor(|tensor| {
            visits += 1;
            let ConvertedTensor::Affine(affine) = tensor.converted() else {
                panic!("expected affine group");
            };
            assert_eq!(affine.bits, 4);
            assert_eq!(affine.group_size, 32);
            assert_eq!(affine.scales.len(), affine.biases.len());
            Ok(())
        })
        .unwrap();
    assert_eq!(visits, 1);
}

#[test]
#[cfg(unix)]
fn indexed_materializer_reuses_the_open_shard_reader() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.gguf");
    let first = 3.0f32.to_le_bytes();
    let second = 7.0f32.to_le_bytes();
    write_file(
        &path,
        None,
        "named lookup",
        &[
            FixtureTensor {
                name: "first.weight",
                dimensions: &[1],
                ty: GgmlType::F32,
                data: &first,
            },
            FixtureTensor {
                name: "second.weight",
                dimensions: &[1],
                ty: GgmlType::F32,
                data: &second,
            },
        ],
    );

    let checkpoint = Checkpoint::open(&path).unwrap();
    let mut materializer = checkpoint.materializer();
    let first_tensor = materializer.raw_tensor("first.weight").unwrap();
    assert_eq!(first_tensor.tensor_index(), 0);
    assert_eq!(first_tensor.descriptor().name, "first.weight");
    assert_eq!(first_tensor.data(), first);

    // An open file remains readable after unlink on supported Unix targets.
    // This proves the second lookup reuses the reader instead of reopening and
    // reparsing the shard.
    std::fs::remove_file(&path).unwrap();
    let tensor = materializer.converted_tensor("second.weight").unwrap();
    assert_eq!(tensor.tensor_index(), 1);
    assert_eq!(tensor.descriptor().name, "second.weight");
    let ConvertedTensor::Dense(dense) = tensor.converted() else {
        panic!("expected a dense tensor");
    };
    assert_eq!(dense.data, second);

    let error = materializer.converted_tensor("missing.weight").unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidTensor { ref tensor, .. } if tensor == "missing.weight"
    ));

    // A materialized raw tensor owns its bytes independently of the reader and
    // checkpoint, which is the safe lifetime boundary used by native backends.
    drop(materializer);
    drop(checkpoint);
    assert_eq!(first_tensor.data(), first);
}

#[test]
fn streaming_rejects_a_shard_changed_after_open() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.gguf");
    let payload = 1.0f32.to_le_bytes();
    write_file(
        &path,
        None,
        "before",
        &[FixtureTensor {
            name: "before.weight",
            dimensions: &[1],
            ty: GgmlType::F32,
            data: &payload,
        }],
    );
    let checkpoint = Checkpoint::open(&path).unwrap();
    write_file(
        &path,
        None,
        "after",
        &[FixtureTensor {
            name: "after.weight",
            dimensions: &[1],
            ty: GgmlType::F32,
            data: &payload,
        }],
    );

    let error = checkpoint.converted_tensors().next().unwrap().unwrap_err();
    assert!(error
        .to_string()
        .contains("changed after the checkpoint was opened"));
}

#[test]
fn rejects_generated_logical_name_collisions_without_converting_payloads() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.gguf");
    let quantized = [0u8; 18];
    let dense = [0u8; 2];
    write_file(
        &path,
        None,
        "collision",
        &[
            FixtureTensor {
                name: "foo.weight",
                dimensions: &[32],
                ty: GgmlType::Q4_0,
                data: &quantized,
            },
            FixtureTensor {
                name: "foo.scales",
                dimensions: &[1],
                ty: GgmlType::F16,
                data: &dense,
            },
        ],
    );

    let error = Checkpoint::open(path).unwrap_err();
    assert!(matches!(
        error,
        Error::DuplicateLogicalTensor { ref name, .. } if name == "foo.scales"
    ));
}

#[test]
fn plans_legacy_q5_as_native_affine() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("legacy.gguf");
    let payload = [0u8; 22];
    write_file(
        &path,
        None,
        "legacy",
        &[FixtureTensor {
            name: "legacy.weight",
            dimensions: &[32],
            ty: GgmlType::Q5_0,
            data: &payload,
        }],
    );

    let catalog = Checkpoint::open(path).unwrap();
    let tensor = &catalog.shards()[0].tensors()[0];
    assert_eq!(tensor.affine(), Some((5, 32)));
    assert_eq!(tensor.outputs().len(), 3);
    assert_eq!(tensor.outputs()[0].shape, [5]);
    assert_eq!(tensor.outputs()[0].dtype, LogicalDtype::U32);
    assert_eq!(tensor.outputs()[1].shape, [1]);
    assert_eq!(tensor.outputs()[1].dtype, LogicalDtype::F16);
    assert_eq!(tensor.outputs()[2].shape, [1]);
    assert_eq!(tensor.outputs()[2].dtype, LogicalDtype::F16);
}

#[test]
fn rejects_translated_name_collisions_from_catalog_only() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.gguf");
    let first = [0u8; 4];
    let second = [0u8; 4];
    write_file(
        &path,
        None,
        "translation",
        &[
            FixtureTensor {
                name: "source.a",
                dimensions: &[1],
                ty: GgmlType::F32,
                data: &first,
            },
            FixtureTensor {
                name: "source.b",
                dimensions: &[1],
                ty: GgmlType::F32,
                data: &second,
            },
        ],
    );

    let catalog = Checkpoint::open(path).unwrap();
    let error = catalog
        .translated_outputs(|_| "target.weight".to_string())
        .unwrap_err();
    assert!(matches!(
        error,
        Error::TranslatedTensorCollision { ref name, .. } if name == "target.weight"
    ));
}

#[test]
fn validates_every_shard_header_before_materialization() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("model-00001-of-00002.gguf");
    let second = directory.path().join("model-00002-of-00002.gguf");
    let payload = [0u8; 4];
    write_file(
        &first,
        Some((0, 2, 2)),
        "first",
        &[FixtureTensor {
            name: "duplicate",
            dimensions: &[1],
            ty: GgmlType::F32,
            data: &payload,
        }],
    );
    write_file(
        &second,
        Some((1, 2, 2)),
        "second",
        &[FixtureTensor {
            name: "duplicate",
            dimensions: &[1],
            ty: GgmlType::F32,
            data: &payload,
        }],
    );

    let error = Checkpoint::open(first).unwrap_err().to_string();
    assert!(error.contains("duplicated across GGUF shards"), "{error}");
}

#[test]
fn reports_missing_canonical_shards() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("model-00001-of-00002.gguf");
    let payload = [0u8; 4];
    write_file(
        &first,
        Some((0, 2, 2)),
        "first",
        &[FixtureTensor {
            name: "only",
            dimensions: &[1],
            ty: GgmlType::F32,
            data: &payload,
        }],
    );

    let error = Checkpoint::open(first).unwrap_err().to_string();
    assert!(error.contains("missing GGUF shard"), "{error}");
    assert!(error.contains("model-00002-of-00002.gguf"), "{error}");
}

#[test]
fn requires_canonical_name_for_multi_shard_catalogs() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.gguf");
    let payload = [0u8; 4];
    write_file(
        &path,
        Some((0, 2, 1)),
        "first",
        &[FixtureTensor {
            name: "only",
            dimensions: &[1],
            ty: GgmlType::F32,
            data: &payload,
        }],
    );

    let error = Checkpoint::open(path).unwrap_err().to_string();
    assert!(
        error.contains("must end in -00001-of-NNNNN.gguf"),
        "{error}"
    );
}

#[test]
fn rejects_inconsistent_shard_counts() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("model-00001-of-00002.gguf");
    let second = directory.path().join("model-00002-of-00002.gguf");
    let payload = [0u8; 4];
    write_file(
        &first,
        Some((0, 2, 2)),
        "first",
        &[FixtureTensor {
            name: "first",
            dimensions: &[1],
            ty: GgmlType::F32,
            data: &payload,
        }],
    );
    write_file(
        &second,
        Some((1, 3, 2)),
        "second",
        &[FixtureTensor {
            name: "second",
            dimensions: &[1],
            ty: GgmlType::F32,
            data: &payload,
        }],
    );

    let error = Checkpoint::open(first).unwrap_err().to_string();
    assert!(error.contains("split.count=3, expected 2"), "{error}");
}
