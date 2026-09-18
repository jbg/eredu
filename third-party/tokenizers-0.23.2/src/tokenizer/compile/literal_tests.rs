use super::*;
use crate::{EncodeIdsError, EncodeIdsPlan, NormalizationBuffer};
use serde_json::json;
fn source(pattern: &str, content: &str, scheme: &str) -> String {
    json!({"version":"1.0", "added_tokens":[
        {"id":6,"content":"a ","normalized":true,"single_word":false,"lstrip":false,"rstrip":false,"special":false},
        {"id":7,"content":"<R>","normalized":false,"single_word":false,"lstrip":false,"rstrip":false,"special":true}],
        "normalizer":{"type":"Replace","pattern":{"String":pattern},"content":content},
        "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":scheme,"split":true},
        "post_processor":null,"decoder":{"type":"Sequence","decoders":[
            {"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},
        "model":{"type":"Unigram","unk_id":0,"byte_fallback":false,"vocab":[["<unk>",-9.0],["a",-1.0],["ab",-1.0],["é",-1.0],["▁",-1.0],["b",-1.0]]}}).to_string()
}
#[test]
fn original_literal_replacement_preserves_added_matches_and_first_origin() {
    for pattern in [" ", "", "a", "ab", "é", "🙂"] {
        for content in ["▁", "", "xé"] {
            for scheme in ["always", "first", "never"] {
                let bytes = source(pattern, content, scheme);
                let actual = TokenizerCompilePlan::prepare_json(bytes.as_bytes())
                    .unwrap()
                    .compile()
                    .unwrap();
                let ordinary = Tokenizer::from_bytes(bytes.as_bytes()).unwrap();
                assert!(actual.matches_compiled_configuration(&ordinary));
                for text in [
                    "",
                    "a b",
                    "ab",
                    "é a",
                    "🙂ab",
                    "aa ",
                    "<R>a <R>ab",
                    " a\né ",
                    "é\u{301}",
                ] {
                    let expected = ordinary.encode(text, false).unwrap();
                    let output = EncodeIdsPlan::prepare(&actual, text, false)
                        .unwrap()
                        .encode()
                        .unwrap();
                    assert_eq!(
                        output.ids(),
                        expected.get_ids(),
                        "{pattern:?} {content:?} {scheme} {text:?}"
                    );
                    assert_eq!(
                        actual.decode(output.ids(), false).unwrap(),
                        ordinary.decode(expected.get_ids(), false).unwrap()
                    );
                }
            }
        }
    }
}
#[test]
fn literal_constructor_refuses_each_real_string_destination_with_prefix_custody() {
    let bytes = source(" ", "▁", "always");
    for stage in 0..2 {
        let failure = TokenizerCompilePlan::prepare_json(bytes.as_bytes())
            .unwrap()
            .fail_normalizer_reservation(stage)
            .compile()
            .unwrap_err();
        assert!(failure.completed_model());
        assert!(failure.allocation_error().is_some());
        assert_eq!(
            failure.literal_normalizer_capacities(),
            if stage == 0 { [0, 0] } else { [1, 0] }
        );
    }
}
#[cfg(feature = "tokenizer-compiler-test-support")]
#[test]
fn literal_operation_refusal_does_not_publish_partial_ids() {
    let bytes = source(" ", "▁", "always");
    let source = TokenizerCompilePlan::prepare_json(bytes.as_bytes())
        .unwrap()
        .compile()
        .unwrap();
    let failure = EncodeIdsPlan::prepare(&source, "a é", false)
        .unwrap()
        .fail_normalization_reservation(NormalizationBuffer::Text)
        .unwrap()
        .encode()
        .unwrap_err();
    assert!(matches!(failure.cause(), EncodeIdsError::Reserve(_)));
    assert_eq!(failure.partial_id_count(), 0);
    assert!(failure.capacities()[2] > 0);
    assert!(failure.unigram_capacities()[0] > 0);
}
