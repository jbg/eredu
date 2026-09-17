use super::*;
use tokenizers::ModelCachePolicy::{Legacy, NoModelCaches};
const JSON: &[u8] = br#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}}"#;
#[test]
fn absent_cache_survives_snapshot_cow_and_source_drop_with_same_decode() {
    let mut source = Tokenizer::from_bytes_with_cache_policy(JSON, NoModelCaches).unwrap();
    let snapshot = source.snapshot();
    let ids = source.encode("abab", false).unwrap().get_ids().to_vec();
    assert_eq!(ids, [2, 2]);
    source
        .add_tokens([tokenizers::AddedToken::from("extra", false)])
        .unwrap();
    assert!(source.token_to_id("extra").is_some());
    assert!(snapshot.token_to_id("extra").is_none());
    assert_eq!(source.model_cache_policy(), NoModelCaches);
    assert_eq!(snapshot.model_cache_policy(), NoModelCaches);
    drop(source);
    let alias = snapshot.clone();
    drop(snapshot);
    assert_eq!(alias.decode(&ids, false).unwrap(), "abab");
    assert_eq!(alias.encode("abab", false).unwrap().get_ids(), ids);
    assert_eq!(
        Tokenizer::from_bytes(JSON).unwrap().model_cache_policy(),
        Legacy
    );
    // Imported HF is still an ordinary unmanaged wrapper, not a paid residence.
    let imported = Tokenizer::from_tokenizer(tokenizers::Tokenizer::from_bytes(JSON).unwrap());
    assert_eq!(imported.model_cache_policy(), Legacy);
}
