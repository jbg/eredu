//! Local source/ID workers compared with pristine tokenizers =0.23.2.
use serde_json::json;
fn main() {
    let steps = [
        json!({"type":"Whitespace"}),
        json!({"type":"Digits","individual_digits":true}),
        json!({"type":"Digits","individual_digits":false}),
        json!({"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true}),
        json!({"type":"Metaspace","replacement":"🦀","prepend_scheme":"always","split":false}),
        json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}),
        json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true}),
        json!({"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}),
        json!({"type":"Split","pattern":{"Regex":"\\w+|[^\\w\\s]+"},"behavior":"Isolated","invert":false}),
    ];
    let inputs = [
        "",
        "ab12",
        " ab 123 ",
        "éab🙂",
        "<R>ab 12",
        "ab<R>123",
        "12é34",
        "a\nb\t12",
        "¼12٣",
        "▁ab🦀12",
        "İAB",
        "E\u{301}a",
        "ab...12",
        "ab\0ab",
    ];
    let mut alphabet = std::collections::BTreeSet::new();
    alphabet.extend(reference::pre_tokenizers::byte_level::ByteLevel::alphabet());
    for text in inputs {
        alphabet.extend(text.chars());
    }
    alphabet.extend(['🦀', '▁', 'é']);
    let mut vocab = serde_json::Map::new();
    for c in alphabet {
        vocab.insert(c.to_string(), json!(vocab.len()));
    }
    let merges = [
        ("a", "b"),
        ("1", "2"),
        ("12", "3"),
        ("▁", "ab"),
        ("Ġ", "ab"),
        ("é", "a"),
        (".", "."),
        ("..", "."),
        ("Ä", "ł"),
    ];
    for (a, b) in merges {
        let merged = format!("{a}{b}");
        if !vocab.contains_key(&merged) {
            vocab.insert(merged, json!(vocab.len()));
        }
    }
    let next = vocab.len();
    let mut comparisons = 0;
    for normalizer in [
        json!(null),
        json!({"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"É"},{"type":"Lowercase"},{"type":"NFC"}]}),
        json!({"type":"Sequence","normalizers":[{"type":"Replace","pattern":{"String":"a"},"content":" b"},{"type":"Lowercase"}]}),
    ] {
        for a in &steps {
            for b in &steps {
                for c in &steps {
                    let pre = json!({"type":"Sequence","pretokenizers":[a,{"type":"Sequence","pretokenizers":[b,c]}]});
                    let source = json!({"version":"1.0","truncation":null,"padding":null,"normalizer":normalizer,
            "pre_tokenizer":pre,"post_processor":null,"decoder":{"type":"Metaspace","replacement":"▁","prepend_scheme":"never","split":true},
            "added_tokens":[{"id":next,"content":"<R>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
            "model":{"type":"BPE","vocab":vocab,"merges":merges,"dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false}}).to_string();
                    let expected = reference::Tokenizer::from_bytes(source.as_bytes()).unwrap();
                    let actual =
                        local::tokenizer::TokenizerCompilePlan::prepare_json(source.as_bytes())
                            .unwrap_or_else(|e| panic!("{e:?} {pre}"))
                            .compile()
                            .unwrap();
                    let ordinary = local::Tokenizer::from_bytes(source.as_bytes()).unwrap();
                    assert!(
                        actual.matches_compiled_configuration(&ordinary),
                        "source {pre}"
                    );
                    for input in inputs {
                        let got = local::EncodeIdsPlan::prepare(&actual, input, false)
                            .unwrap_or_else(|e| panic!("{e:?} {pre}"))
                            .encode()
                            .unwrap();
                        assert_eq!(
                            got.ids(),
                            expected.encode(input, false).unwrap().get_ids(),
                            "{pre} {input:?}"
                        );
                        comparisons += 1;
                    }
                }
            }
        }
    }
    println!("{comparisons} pristine ordered pre-tokenizer comparisons passed");
}
