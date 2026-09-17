use super::*;
use std::{convert::Infallible, sync::Arc};

#[test]
fn streaming_preserves_canonical_multimodal_metadata_and_extent_words() {
    // Independent wire fixture: video F32[2,6] + grid I32[1,3] + host grid,
    // then audio F16[4,8] + Boolean mask[4] + three valid frames.
    let expected = vec![
        2, 2, 1, 10, 2, 2, 6, 1, 0, 7, 2, 1, 3, 1, 0, 2, 3, 1, 3, 1, 9, 2, 4, 8, 1, 2, 0, 1, 4, 1,
        1, 3,
    ];
    let descriptor = PreparedInputIdentity::decode_words(&expected).unwrap();
    let mut streamed = Vec::new();
    descriptor
        .visit_encoded_words(|word| {
            streamed.push(word);
            Ok::<_, Infallible>(())
        })
        .unwrap();
    assert_eq!(streamed, expected);
    assert_eq!(descriptor.encoded_word_count().unwrap(), expected.len());
    assert_eq!(descriptor.encode_words().unwrap(), expected);
}

#[derive(Debug)]
struct Stop(Arc<()>);
impl std::fmt::Display for Stop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("stop at exact prefix")
    }
}
impl std::error::Error for Stop {}

#[test]
fn sink_failure_preserves_its_source_and_stops_before_later_words() {
    let descriptor = TextTokenInputDescriptorPlan::new(2, 5).unwrap().construct();
    let identity = Arc::new(());
    let mut calls = 0;
    let error = descriptor
        .visit_encoded_words(|_| {
            calls += 1;
            if calls == 4 {
                Err(Stop(identity.clone()))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert_eq!(calls, 4);
    let InputWordWriteError::Sink(error) = error else {
        panic!("original sink cause")
    };
    assert!(Arc::ptr_eq(&error.0, &identity));
}

#[test]
fn encoded_runtime_dtype_remains_a_typed_encoding_failure() {
    let mut descriptor = TextTokenInputDescriptorPlan::new(1, 5).unwrap().construct();
    descriptor.parts[0].payload.dtype = TensorDtype::Encoded("q4-source".into());
    let mut prefix = Vec::new();
    let error = descriptor
        .visit_encoded_words(|word| {
            prefix.push(word);
            Ok::<_, Infallible>(())
        })
        .unwrap_err();
    assert_eq!(prefix, [1, 0, 0]);
    assert!(
        matches!(error, InputWordWriteError::Encoding(PreparedInputError::EncodedRuntimeDtype(name)) if name == "q4-source")
    );
    assert!(
        matches!(descriptor.encode_words(), Err(PreparedInputError::EncodedRuntimeDtype(name)) if name == "q4-source")
    );
}
