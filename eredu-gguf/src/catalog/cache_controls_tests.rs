use super::*;
use crate::{GgmlType, TensorInput, Writer};

mod materializer_index;
mod original;
mod reader_buffers;

fn sharded() -> (tempfile::TempDir, Checkpoint) {
    let dir = tempfile::tempdir().unwrap();
    for index in 0..2 {
        let metadata = BTreeMap::from([
            (
                "general.name".into(),
                MetadataValue::String("moved catalog α".into()),
            ),
            (SPLIT_NO.into(), MetadataValue::Uint16(index)),
            (SPLIT_COUNT.into(), MetadataValue::Uint16(2)),
            (SPLIT_TENSORS_COUNT.into(), MetadataValue::Uint16(2)),
        ]);
        let data = if index == 0 {
            [1.25_f32.to_le_bytes(), (-3.5_f32).to_le_bytes()].concat()
        } else {
            let mut data = vec![0x21; 17];
            data[0] = 130;
            data
        };
        Writer::default()
            .write(
                File::create(
                    dir.path()
                        .join(format!("moved-{:05}-of-00002.gguf", index + 1)),
                )
                .unwrap(),
                &metadata,
                &[TensorInput {
                    name: if index == 0 {
                        "dense.weight"
                    } else {
                        "packed.weight"
                    },
                    dimensions: if index == 0 { &[2] } else { &[32, 1] },
                    ggml_type: if index == 0 {
                        GgmlType::F32
                    } else {
                        GgmlType::MxFp4
                    },
                    data: &data,
                }],
            )
            .unwrap();
    }
    let checkpoint = Checkpoint::open(dir.path().join("moved-00001-of-00002.gguf")).unwrap();
    (dir, checkpoint)
}

fn allocations(checkpoint: &Checkpoint) -> Vec<usize> {
    let mut result = vec![checkpoint.shards.as_ptr() as usize];
    for (key, value) in &checkpoint.metadata {
        result.push(key.as_ptr() as usize);
        if let MetadataValue::String(value) = value {
            result.push(value.as_ptr() as usize);
        }
    }
    for shard in &checkpoint.shards {
        result.push(shard.path.as_os_str().as_encoded_bytes().as_ptr() as usize);
        result.push(shard.tensors.as_ptr() as usize);
        for tensor in &shard.tensors {
            result.push(tensor.descriptor.name.as_ptr() as usize);
            result.push(tensor.descriptor.dimensions.as_ptr() as usize);
            result.push(tensor.outputs.as_ptr() as usize);
            for output in &tensor.outputs {
                result.push(output.name.as_ptr() as usize);
                result.push(output.shape.as_ptr() as usize);
            }
        }
    }
    result
}

#[test]
fn consuming_catalog_preserves_owned_buffers_and_original_nonzero_outputs() {
    let (_dir, checkpoint) = sharded();
    let addresses = allocations(&checkpoint);
    let mut expected = checkpoint.old_materializer();
    let mut borrowed = checkpoint.materializer();
    assert_eq!(borrowed.checkpoint.shards, expected.checkpoint.shards);
    assert_eq!(borrowed.checkpoint.metadata, expected.checkpoint.metadata);
    assert!(addresses
        .iter()
        .zip(allocations(&borrowed.checkpoint))
        .all(|(a, b)| *a != b));
    let mut moved = checkpoint.into_materializer();
    assert_eq!(allocations(&moved.checkpoint), addresses);
    assert!(moved.open_shard_path().is_none());
    for name in ["packed.weight", "dense.weight", "packed.weight"] {
        let old = expected.converted_tensor(name).unwrap();
        assert_eq!(borrowed.converted_tensor(name).unwrap(), old);
        let actual = moved.converted_tensor(name).unwrap();
        assert_eq!(actual, old);
        if name == "dense.weight" {
            let ConvertedTensor::Dense(dense) = actual.converted() else {
                panic!("dense")
            };
            assert_eq!(
                dense.data,
                [1.25_f32.to_ne_bytes(), (-3.5_f32).to_ne_bytes()].concat()
            );
        } else {
            assert_eq!(actual.output_names(), ["packed.weight", "packed.scales"]);
        }
    }
    let old = expected.converted_tensor("missing").unwrap_err();
    assert_eq!(
        moved.converted_tensor("missing").unwrap_err().to_string(),
        old.to_string()
    );
    assert_eq!(
        borrowed
            .converted_tensor("missing")
            .unwrap_err()
            .to_string(),
        old.to_string()
    );
}

#[test]
fn no_path_close_retires_reader_and_preserves_owned_close_results() {
    let (_dir, checkpoint) = sharded();
    let path = checkpoint.shards[0].path.clone();
    let mut old = checkpoint.old_materializer();
    let mut owned = checkpoint.materializer();
    let mut quiet = checkpoint.into_materializer();
    assert_eq!(old.old_close_reader(), None);
    assert_eq!(owned.close_reader(), None);
    assert!(!quiet.close_reader_without_path());
    for materializer in [&mut old, &mut owned, &mut quiet] {
        materializer.converted_tensor("dense.weight").unwrap();
        assert_eq!(materializer.open_shard_path(), Some(path.as_path()));
    }
    assert_eq!(old.old_close_reader(), Some(path.clone()));
    assert_eq!(owned.close_reader(), Some(path.clone()));
    assert!(quiet.close_reader_without_path());
    assert!(!quiet.close_reader_without_path());
    assert!(quiet.reader.is_none());
    std::fs::remove_file(&path).unwrap();
    let expected = old.converted_tensor("dense.weight").unwrap_err();
    assert!(matches!(expected, Error::Shard { .. }));
    assert_eq!(
        owned
            .converted_tensor("dense.weight")
            .unwrap_err()
            .to_string(),
        expected.to_string()
    );
    assert_eq!(
        quiet
            .converted_tensor("dense.weight")
            .unwrap_err()
            .to_string(),
        expected.to_string()
    );
}
