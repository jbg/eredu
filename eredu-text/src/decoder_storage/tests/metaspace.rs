use super::*;
use serde_json::json;

#[test]
fn metaspace_source_preserves_first_retained_piece_and_stream_frontiers() {
    for replacement in ['▁', '_', '🦀'] {
        for scheme in ["always", "first", "never"] {
            for split in [false, true] {
                for sequence in [false, true] {
                    let decoder = json!({"type":"Metaspace", "replacement":replacement,
                        "prepend_scheme":scheme, "split":split});
                    let decoder = if sequence { json!({"type":"Sequence", "decoders":[decoder]}) } else { decoder };
                    let pieces = ["[UNK]".into(), "".into(), replacement.to_string(),
                        format!("{replacement}a{replacement}b{replacement}"),
                        "é".into(), format!("{replacement}c"), "<stop>".into()];
                    let mut hf = tokenizers::Tokenizer::new(model(
                        pieces.into_iter().enumerate().map(|(id, piece)| (piece, id as u32)),
                    ));
                    hf.with_decoder(Some(serde_json::from_value::<DecoderWrapper>(decoder).unwrap()));
                    hf.add_special_tokens([AddedToken::from("<stop>", true)]).unwrap();
                    let snapshot = Tokenizer::from_tokenizer(hf).snapshot();
                    let source = DecodeCompilePlan::prepare(&snapshot).unwrap().compile().unwrap();
                    for skip in [false, true] {
                        for ids in [&[3, 5, 4][..], &[6, 999, 3, 2, 5], &[1, 3, 5], &[2, 3, 5]] {
                            pipelines::trace_source(&snapshot, &source, ids, skip);
                        }
                    }
                    let mut raw = [];
                    let mut output = [0; 64];
                    let n = source.decode_into(&[999, 6, 3, 5], true, &mut raw, &mut output);
                    let expected = if scheme == "never" { " a b  c" } else { "ab c" };
                    assert_eq!(std::str::from_utf8(&output[..n]).unwrap(), expected);
                    for stage in 0..3 {
                        let error = DecodeCompilePlan::prepare(&snapshot).unwrap()
                            .fail_reservation(stage).compile().unwrap_err();
                        assert!(matches!(error.cause(), DecodeSourceError::Allocation(_)));
                        assert_eq!(error.retained_buffer_bytes() == 0, stage == 0);
                    }
                }
            }
        }
    }
}
