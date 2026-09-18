//! Independent executable: local tokenizers path alias plus pristine registry
//! tokenizers =0.23.2 alias `reference`, both with fancy-regex enabled.
fn main() {
    let mut compared = 0;
    for replacement in ['▁', '_', '🦀'] {
        for scheme in ["always", "first", "never"] {
            for split in [false, true] {
                for sequence in [false, true] {
                    let decoder = serde_json::json!({"type":"Metaspace", "replacement":replacement,
                        "prepend_scheme":scheme, "split":split});
                    let decoder = if sequence { serde_json::json!({"type":"Sequence", "decoders":[decoder]}) } else { decoder };
                    let pieces = ["[UNK]".into(), "".into(), replacement.to_string(),
                        format!("{replacement}a{replacement}b{replacement}"), "é".into(),
                        format!("{replacement}c"), "<stop>".into()];
                    let vocab = pieces.into_iter().enumerate().map(|(id, text)| (text, serde_json::json!(id)))
                        .collect::<serde_json::Map<_, _>>();
                    let source = serde_json::json!({"version":"1.0", "truncation":null, "padding":null,
                        "added_tokens":[{"id":6,"content":"<stop>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
                        "normalizer":null, "pre_tokenizer":null, "post_processor":null, "decoder":decoder,
                        "model":{"type":"WordLevel", "vocab":vocab, "unk_token":"[UNK]"}}).to_string();
                    let upstream = reference::Tokenizer::from_bytes(source.as_bytes()).unwrap();
                    let selected = local::tokenizer::TokenizerCompilePlan::prepare_json(source.as_bytes()).unwrap().compile().unwrap();
                    let ids = [0, 1, 2, 3, 4, 5, 6, 999];
                    for skip in [false, true] {
                        for a in ids { for b in ids { for c in ids {
                            let input = [a, b, c];
                            assert_eq!(selected.decode(&input, skip).unwrap(), upstream.decode(&input, skip).unwrap(),
                                "{replacement:?}/{scheme}/{split}/{sequence}/{input:?}/{skip}");
                            compared += 1;
                        } } }
                    }
                }
            }
        }
    }
    println!("{compared} pristine Metaspace decoder comparisons passed");
}
