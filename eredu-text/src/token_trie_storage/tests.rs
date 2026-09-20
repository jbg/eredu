use super::*;
use crate::tokenizer_storage::TokenizerPlan;
use serde_json::json;

fn tokenizer(vocab: serde_json::Value, added: serde_json::Value) -> PreparedTokenizer {
    let source = json!({"version":"1.0", "truncation":null, "padding":null,
        "normalizer":null,"pre_tokenizer":null,"post_processor":null,
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "added_tokens":added,"model":{"type":"BPE","vocab":vocab,"merges":[]}}).to_string();
    TokenizerPlan::prepare_json(source.as_bytes())
        .unwrap()
        .compile()
        .unwrap()
}

#[test]
fn sparse_vocabulary_marker_bytes_and_ordered_eos_use_stock_trie() {
    let source = tokenizer(
        json!({"h":0,"i":1,"hi":2,"Ġ":4}),
        json!([
            {"id":5,"content":"<S>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
        ]),
    );
    // Stock HF assigns the added token after its vocabulary entry count.
    let count = source.token_trie_vocabulary().unwrap().token_count() as u32;
    let special = source.token_id("<S>").unwrap();
    let eos = [special, 2, special];
    let info = TokRxInfo::new(count, special);
    let source = TokenTriePlan::prepare(&source, &info, &eos)
        .unwrap()
        .compile()
        .unwrap();
    assert_eq!(source.trie().token(0), b"h");
    assert_eq!(source.trie().token(2), b"hi");
    assert_eq!(source.trie().token(3), b"");
    assert_eq!(source.trie().token(special), b"\xff<S>");
    assert_eq!(source.trie().eos_tokens(), eos);
    assert_eq!(source.trie().token_id(b"hi"), Some(2));
}

#[test]
fn absent_eos_is_a_nonmatching_upstream_sentinel() {
    let source = tokenizer(json!({"a":0,"b":1}), json!([]));
    let info = TokRxInfo::new(2, toktrie::INVALID_TOKEN);
    let trie = TokenTriePlan::prepare(&source, &info, &[])
        .unwrap()
        .compile()
        .unwrap();
    assert_eq!(trie.trie().eos_tokens(), [toktrie::INVALID_TOKEN]);
    let set = trie.trie().eos_token_set();
    assert!(!set.is_allowed(0));
    assert!(!set.is_allowed(1));
    assert!(matches!(
        TokenTriePlan::prepare(&source, &TokRxInfo::new(2, 0), &[]),
        Err(TokenTrieSourceError::Eos)
    ));
    assert!(matches!(
        TokenTriePlan::prepare(&source, &TokRxInfo::new(2, 0), &[0, 2]),
        Err(TokenTrieSourceError::Eos)
    ));
}

#[test]
fn prefix_summary_handles_empty_duplicate_prefixes_and_representable_long_branches() {
    for words in [
        vec![
            b"".to_vec(),
            b"a".to_vec(),
            b"a".to_vec(),
            b"ab".to_vec(),
            b"ac".to_vec(),
        ],
        vec![vec![b'x'; 1024]],
        vec![vec![b'a'; 1025], [vec![b'a'; 500], vec![b'z']].concat()],
    ] {
        validate_geometry(&words).unwrap();
        let trie = TokTrie::from(&TokRxInfo::new(words.len() as u32, 0), &words);
        trie.check_against(&words);
    }
    assert!(matches!(
        validate_geometry(&[vec![b'x'; 1025]]),
        Err(Cause::Source(TokenTrieSourceError::NodeEncoding))
    ));
}

#[test]
fn policy_estimates_are_configurable_and_geometry_failures_retain_owned_inputs() {
    let word = "x".repeat(1025);
    let source = tokenizer(json!({word:0}), json!([]));
    let info = TokRxInfo::new(1, 0);
    let failure = TokenTriePlan::prepare(&source, &info, &[0])
        .unwrap()
        .compile()
        .unwrap_err();
    assert!(matches!(
        failure.source_error(),
        Some(TokenTrieSourceError::TokenLength {
            actual: 1025,
            limit: 1024
        })
    ));
    let policy = TokenTrieMemoryPolicy {
        max_token_bytes: 2048,
        ..Default::default()
    };
    let plan = TokenTriePlan::prepare(&source, &info, &[0])
        .unwrap()
        .with_memory_policy(policy)
        .unwrap();
    let estimate = plan.requirements().required_bytes();
    let failure = plan.compile().unwrap_err();
    drop(source);
    assert!(matches!(
        failure.source_error(),
        Some(TokenTrieSourceError::NodeEncoding)
    ));
    assert!(failure.packed_capacity() > 1025);
    assert!(failure.retained_buffer_bytes() < estimate);
}
