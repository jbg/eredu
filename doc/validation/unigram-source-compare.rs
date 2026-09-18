fn main() {
    let mut cases = 0;
    for unk in [None, Some(0)] { for fallback in 0..3 { for scheme in ["always", "first", "never"] {
        let mut vocab = vec![("<unk>".to_string(), 0.0), ("a".into(), -1.0), ("b".into(), -1.0), ("ab".into(), -2.0), ("é".into(), -0.25), ("中文".into(), -0.4), ("▁".into(), -0.1), ("a".into(), -0.75)];
        if fallback != 0 {for b in 0..=255 {if fallback == 1 || b != 0x99 {vocab.push((format!("<0x{b:02X}>"), -10.0));}}}
        let source = serde_json::json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"post_processor":null,"decoder":null,
            "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":scheme,"split":true},
            "model":{"type":"Unigram","vocab":vocab,"unk_id":unk,"byte_fallback":fallback != 0}}).to_string();
        let old = reference::Tokenizer::from_bytes(source.as_bytes()).unwrap();
        let new = local::Tokenizer::from_bytes(source.as_bytes()).unwrap();
        let compiled = local::TokenizerCompilePlan::prepare_json(source.as_bytes()).unwrap().compile().unwrap();
        let pieces = ["", "a", "ab", "é", "🙂", "中文", " ", "\n", "??", "é\u{301}"];
        for a in pieces {for b in pieces {for c in pieces {
            let text = format!("{a}{b}{c}");
            let expected = old.encode(text.as_str(), false);
            for actual in [new.encode(text.as_str(),false), compiled.encode(text.as_str(),false)] {
                match (&expected, actual) {
                    (Ok(e), Ok(a)) => { assert_eq!(a.get_ids(),e.get_ids(),"{scheme} {fallback} {unk:?} {text:?}"); assert_eq!(a.get_tokens(),e.get_tokens()); assert_eq!(a.get_offsets(),e.get_offsets()); },
                    (Err(_), Err(_)) => (), _ => panic!("result parity {scheme} {fallback} {unk:?} {text:?}"),
                }
            }
            match (expected, local::EncodeIdsPlan::prepare(&compiled,&text,false).unwrap().encode()) {
                (Ok(e),Ok(a)) => assert_eq!(a.ids(),e.get_ids()), (Err(_),Err(_))=>(), _=>panic!("ID parity {text:?}"),
            }
            cases += 1;
        }}}
    }}}
    println!("pristine Unigram ordinary/source tokens, IDs, offsets and admitted IDs: {cases} cases");
}
