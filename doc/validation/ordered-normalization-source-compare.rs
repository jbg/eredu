//! Independent executable: local tokenizers path plus pristine registry 0.23.2.
use reference::Normalizer;
use serde_json::json;
fn main() {
    let steps = [json!({"type":"Lowercase"}), json!({"type":"NFC"}),
        json!({"type":"Prepend","prepend":"É"}),
        json!({"type":"Replace","pattern":{"String":"É"},"content":"e\u{301}"}),
        json!({"type":"Replace","pattern":{"String":""},"content":"X"}),
        json!({"type":"Replace","pattern":{"String":"A"},"content":""}),
        json!({"type":"Replace","pattern":{"String":" "},"content":"▁"})];
    let inputs = ["", "A", "É", "E\u{301}", "İΣẞ", "\u{301}\u{300}A", " A É ",
        "<R>A É<R>İ", "ÉAé", "한글", "A\0É", "𐐀İ", "🙂a", "AA", "É\u{301}"];
    let mut comparisons = 0;
    // Pristine 0.23.2's empty Replace followed by another transforming
    // normalizer has an upstream alignment panic (e.g. Lowercase, Replace("",
    // "X"), Lowercase on "A"). Keep empty replacement in the final stage so
    // every comparison below has a successful independent reference result.
    for a in steps.iter().enumerate().filter(|(i, _)| *i != 4).map(|(_, s)| s) {
      for b in steps.iter().enumerate().filter(|(i, _)| *i != 4).map(|(_, s)| s) { for c in &steps {
        let normalization = json!({"type":"Sequence","normalizers":[a,{"type":"Sequence","normalizers":[b,c]}]});
        let probe: reference::normalizers::NormalizerWrapper = serde_json::from_value(normalization.clone()).unwrap();
        let mut characters = std::collections::BTreeSet::new();
        for text in inputs.into_iter().chain(["É", "A"]) {
            let mut normalized = reference::NormalizedString::from(text);
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| probe.normalize(&mut normalized).unwrap())).unwrap_or_else(|_| panic!("pristine normalization panicked for {normalization} {text:?}"));
            characters.extend(normalized.get().chars());
            characters.extend(text.chars());
        }
        characters.insert('▁');
        let mut vocab = serde_json::Map::new();
        for (id, c) in characters.into_iter().enumerate() { vocab.insert(c.to_string(), json!(id)); }
        let next = vocab.len();
        for scheme in ["always", "first", "never"] {
            let source = json!({"version":"1.0","normalizer":normalization,
                "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":scheme,"split":true},
                "post_processor":null,"decoder":{"type":"Metaspace","replacement":"▁","prepend_scheme":scheme,"split":true},
                "added_tokens":[
                    {"id":next,"content":"<R>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                    {"id":vocab["É"],"content":"É","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}],
                "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":vocab,"merges":[]}}).to_string();
            let reference = reference::Tokenizer::from_bytes(source.as_bytes()).unwrap();
            let local = local::tokenizer::TokenizerCompilePlan::prepare_json(source.as_bytes()).unwrap_or_else(|error| panic!("{error:?} {source}")).compile().unwrap();
            let ordinary_local = local::Tokenizer::from_bytes(source.as_bytes()).unwrap();
            assert!(local.matches_compiled_configuration(&ordinary_local), "configuration {normalization}");
            for input in inputs {
                let expected = reference.encode(input, false).unwrap();
                let got = local::EncodeIdsPlan::prepare(&local, input, false).unwrap().encode().unwrap();
                assert_eq!(got.ids(), expected.get_ids(), "{normalization} / {scheme} / {input:?}");
                comparisons += 1;
            }
        }
    } } }
    println!("{comparisons} pristine ordered-normalization comparisons passed");
}
