use super::*;
use crate::ReaderBuffer;

fn addresses(materializer: &TensorMaterializer) -> Vec<usize> {
    let mut result = match &materializer.reader_storage {
        ReaderStorage::Local(buffers) => buffers
            .iter()
            .flatten()
            .map(ReaderBuffer::address)
            .collect::<Vec<_>>(),
        _ => panic!("local storage"),
    };
    if let Some((_, reader)) = &materializer.reader {
        result.push(reader.buffer_address().unwrap());
    }
    result.sort_unstable();
    result
}
#[test]
fn direct_reader_buffers_preserve_two_owners_and_original_replacement_errors() {
    for captured in [false, true] {
        let (_dir, mut checkpoint) = sharded();
        if captured {
            checkpoint =
                Checkpoint::open_with_prepared_headers(&checkpoint.shards[0].path).unwrap();
        }
        let source_path = checkpoint.shards[1].path.clone();
        let original_file = std::fs::read(&source_path).unwrap();
        let mut ordinary = checkpoint.materializer();
        let buffers = [
            ReaderBuffer::prepare().unwrap(),
            ReaderBuffer::prepare().unwrap(),
        ];
        let mut expected = buffers
            .iter()
            .map(ReaderBuffer::address)
            .collect::<Vec<_>>();
        expected.sort_unstable();
        let mut actual = checkpoint
            .into_materializer_with_reader_buffers(buffers)
            .unwrap();
        assert_eq!(addresses(&actual), expected);
        let dense = ordinary.converted_tensor("dense.weight").unwrap();
        assert_eq!(actual.converted_tensor("dense.weight").unwrap(), dense);
        let current = actual.reader.as_ref().unwrap().1.buffer_address();
        std::fs::remove_file(&source_path).unwrap();
        let error = actual.converted_tensor("packed.weight").unwrap_err();
        assert_eq!(
            error.to_string(),
            ordinary
                .converted_tensor("packed.weight")
                .unwrap_err()
                .to_string()
        );
        assert_eq!(actual.reader.as_ref().unwrap().1.buffer_address(), current);
        assert_eq!(addresses(&actual), expected);
        assert_eq!(actual.converted_tensor("dense.weight").unwrap(), dense);
        std::fs::write(&source_path, &original_file).unwrap();
        // Captured-header and ordinary parse failures return the second buffer
        // while the original File/buffer remains readable.
        std::fs::write(&source_path, &original_file[..3]).unwrap();
        assert!(actual.converted_tensor("packed.weight").is_err());
        assert_eq!(actual.reader.as_ref().unwrap().1.buffer_address(), current);
        assert_eq!(addresses(&actual), expected);
        std::fs::write(&source_path, &original_file).unwrap();
        for name in [
            "packed.weight",
            "dense.weight",
            "packed.weight",
            "dense.weight",
        ] {
            assert_eq!(
                actual.converted_tensor(name).unwrap(),
                ordinary.converted_tensor(name).unwrap()
            );
            assert_eq!(addresses(&actual), expected);
        }
        assert!(actual.close_reader_without_path());
        assert_eq!(addresses(&actual), expected);
    }
}
#[test]
fn iterator_reuses_one_buffer_across_real_shards_and_pooled_refusal_has_no_fallback() {
    let (_dir, checkpoint) = sharded();
    let expected = checkpoint
        .converted_tensors()
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let buffer = ReaderBuffer::prepare().unwrap();
    let address = buffer.address();
    let mut actual = checkpoint
        .converted_tensors_with_reader_buffer(buffer)
        .unwrap();
    let first = actual.next().unwrap().unwrap();
    assert_eq!(
        actual.reader.as_ref().unwrap().buffer_address(),
        Some(address)
    );
    let second = actual.next().unwrap().unwrap();
    assert_eq!(
        actual.reader.as_ref().unwrap().buffer_address(),
        Some(address)
    );
    assert_eq!(vec![first, second], expected);
    assert!(actual.next().is_none());
    let ReaderStorage::Local(buffers) = &actual.reader_storage else {
        panic!()
    };
    assert_eq!(
        buffers
            .iter()
            .flatten()
            .map(ReaderBuffer::address)
            .collect::<Vec<_>>(),
        [address]
    );
    drop(actual);
    let mut pooled = checkpoint
        .into_materializer()
        .with_pooled_reader_storage()
        .unwrap();
    assert!(pooled
        .converted_tensor("dense.weight")
        .unwrap_err()
        .prepared_reader_storage()
        .is_some());
    let buffer = ReaderBuffer::prepare().unwrap();
    let address = buffer.address();
    pooled.supply_reader_buffer(buffer).unwrap();
    pooled.converted_tensor("dense.weight").unwrap();
    assert!(pooled.take_idle_reader_buffer().is_none());
    let refused = ReaderBuffer::prepare().unwrap();
    let refused_address = refused.address();
    assert_eq!(
        pooled.supply_reader_buffer(refused).unwrap_err().address(),
        refused_address
    );
    assert!(pooled.close_reader_without_path());
    assert_eq!(pooled.take_idle_reader_buffer().unwrap().address(), address);
}
