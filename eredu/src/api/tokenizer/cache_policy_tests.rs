use super::*;
use tokenizers::ModelCachePolicy::{Legacy, NoModelCaches};
const JSON: &str = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}}"#;
#[test]
fn selected_policy_reaches_json_tiktoken_and_gguf_fallback_without_changing_defaults() {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "eredu-facade-cache-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(dir.clone());
    std::fs::write(dir.join("tiktoken.model"), "YQ== 0\nYg== 1\nYWI= 2\n").unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"added_tokens_decoder":{}}"#,
    )
    .unwrap();
    let rank_import =
        load_tokenizer_for_kind_with_cache_policy(ModelKind::KimiLinear, &dir, NoModelCaches)
            .unwrap();
    assert_eq!(rank_import.model_cache_policy(), NoModelCaches);
    assert_eq!(
        rank_import.encode("abab", false).unwrap().get_ids(),
        &[2, 2]
    );
    std::fs::write(dir.join("tokenizer.json"), JSON).unwrap();
    for kind in [ModelKind::Llama, ModelKind::KimiLinear] {
        let default = load_tokenizer_for_kind(kind, &dir).unwrap();
        let absent = load_tokenizer_for_kind_with_cache_policy(kind, &dir, NoModelCaches).unwrap();
        assert_eq!(default.model_cache_policy(), Legacy);
        assert_eq!(absent.model_cache_policy(), NoModelCaches);
        assert_eq!(
            default.encode("abab", false).unwrap().get_ids(),
            absent.encode("abab", false).unwrap().get_ids()
        );
    }
    let artifact = dir.join("model.gguf");
    let mut metadata = std::collections::HashMap::new();
    let fallback =
        load_gguf_tokenizer_from_metadata_with_cache_policy(&artifact, &metadata, NoModelCaches)
            .unwrap();
    assert_eq!(fallback.tokenizer.model_cache_policy(), NoModelCaches);
    metadata.insert(
        "tokenizer.huggingface.json".into(),
        GgufMetadataValue::String(JSON.into()),
    );
    let embedded =
        load_gguf_tokenizer_from_metadata_with_cache_policy(&artifact, &metadata, NoModelCaches)
            .unwrap();
    assert_eq!(
        embedded.tokenizer.encode("abab", false).unwrap().get_ids(),
        &[2, 2]
    );
    assert_eq!(embedded.tokenizer.model_cache_policy(), NoModelCaches);
    std::fs::write(dir.join("tokenizer.json"), "{").unwrap();
    assert!(
        load_tokenizer_for_kind_with_cache_policy(ModelKind::Llama, &dir, NoModelCaches).is_err()
    );
    metadata.insert(
        "tokenizer.huggingface.json".into(),
        GgufMetadataValue::String("{".into()),
    );
    assert!(load_gguf_tokenizer_from_metadata_with_cache_policy(
        &artifact,
        &metadata,
        NoModelCaches
    )
    .is_err());
}
