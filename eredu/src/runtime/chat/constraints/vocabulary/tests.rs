use super::*;
use llguidance::toktrie::TokRxInfo;

#[test]
fn token_bytes_preserve_sparse_multibyte_and_special_token_slices() {
    let words = vec![
        b"hello".to_vec(),
        Vec::new(),
        "é🙂".as_bytes().to_vec(),
        b"\xff<|tool|>".to_vec(),
        Vec::new(),
        b"\xff".to_vec(),
    ];
    let trie = TokTrie::from(&TokRxInfo::new(words.len() as u32, 0), &words);
    let expected: &[&[u8]] = &[b"hello", b"", "é🙂".as_bytes(), b"<|tool|>", b"", b""];
    for (token, expected) in expected.iter().enumerate() {
        assert_eq!(token_bytes(&trie, token as u32), *expected);
    }
}
