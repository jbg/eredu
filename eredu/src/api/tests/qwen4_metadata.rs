//! Released Flash-Next sidecars keep lexical reset and facade termination distinct.
use super::*;

const CONFIG: &str =
    include_str!("../../../../eredu-architectures/src/qwen4_exp/config/released.json");
const GENERATION: &str = include_str!(
    "../../../tests/fixtures/chat_templates/qwen3.8-flash-next-de4b8e4d-generation-config.json"
);
const TOKENIZER: &str = include_str!(
    "../../../tests/fixtures/chat_templates/qwen3.8-flash-next-de4b8e4d-tokenizer-config.json"
);

fn sidecars() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    for (name, contents) in [
        ("config.json", CONFIG),
        ("generation_config.json", GENERATION),
        ("tokenizer_config.json", TOKENIZER),
    ] {
        fs::write(directory.path().join(name), contents).unwrap();
    }
    directory
}

#[test]
fn flash_next_metadata_preserves_declared_reset_and_both_generation_terminators() {
    let directory = sidecars();
    let configuration: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    let family = eredu_architectures::qwen4_exp::config::Config::from_json(&configuration).unwrap();
    // The model equation uses endoftext; the tokenizer's chat EOS is im_end.
    assert_eq!(family.eos, [248044]);
    let kwargs = load_tokenizer_template_kwargs(directory.path()).unwrap();
    assert_eq!(kwargs["pad_token"], "<|endoftext|>");
    assert_eq!(kwargs["eos_token"], "<|im_end|>");
    let tokenizer: serde_json::Value = serde_json::from_str(TOKENIZER).unwrap();
    assert_eq!(
        tokenizer["added_tokens_decoder"]["248044"]["content"],
        kwargs["pad_token"]
    );
    assert_eq!(
        tokenizer["added_tokens_decoder"]["248046"]["content"],
        kwargs["eos_token"]
    );

    let eos = eos_token_ids_from_sidecar_dir(directory.path()).unwrap();
    assert_eq!(eos, [248044, 248046]);
    let generation: serde_json::Value = serde_json::from_str(GENERATION).unwrap();
    assert_eq!(generation["eos_token_id"], json!([248046, 248044]));
    for terminal in [248044, 248046] {
        let mut sequence = eredu_core::generation::GenerationSequence::new(8, eos.iter().copied());
        assert_eq!(
            sequence
                .commit(248045, Default::default())
                .unwrap()
                .finish_reason,
            None
        );
        assert_eq!(
            sequence
                .commit(terminal, Default::default())
                .unwrap()
                .finish_reason,
            Some(FinishReason::Eos)
        );
    }

    // A GGUF reset declaration is not a generation EOS source. The published
    // tokenizer EOS combines with sidecar policy without duplicate stops.
    let metadata = std::collections::HashMap::from([
        (
            "qwen4exp.ple.eos_token_id".to_owned(),
            eredu_gguf::MetadataValue::Uint32(248044),
        ),
        (
            "tokenizer.ggml.eos_token_id".to_owned(),
            eredu_gguf::MetadataValue::Uint32(248046),
        ),
    ]);
    assert_eq!(gguf_eos_token_ids(&metadata).unwrap(), [248046]);
    assert_eq!(
        merge_eos_token_id_sources([eos, gguf_eos_token_ids(&metadata).unwrap()]),
        [248044, 248046]
    );
}

#[test]
fn flash_next_metadata_resolves_pinned_sampling_defaults_and_request_overrides() {
    let directory = sidecars();
    let checkpoint = read_checkpoint_generation_config(directory.path())
        .unwrap()
        .unwrap();
    let selected =
        resolve_generation_config(Some(&checkpoint), GenerationConfigOverrides::default()).unwrap();
    assert!(selected.do_sample);
    assert_eq!(selected.temperature, 1.0);
    assert_eq!(selected.top_k, 20);
    assert_eq!(selected.top_p, 0.95);
    assert_eq!(selected.max_new_tokens, None);
    let overridden = resolve_generation_config(
        Some(&checkpoint),
        GenerationConfigOverrides {
            temperature: Some(0.0),
            top_k: Some(12),
            max_new_tokens: Some(32),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!overridden.do_sample);
    assert_eq!(overridden.temperature, 0.0);
    assert_eq!(overridden.top_k, 12);
    assert_eq!(overridden.top_p, 0.95);
    assert_eq!(overridden.max_new_tokens, Some(32));
}

#[test]
fn flash_next_metadata_does_not_infer_termination_from_padding_or_hash_reset() {
    let directory = sidecars();
    let mut config: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    config["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("eos_token_id");
    let mut generation: serde_json::Value = serde_json::from_str(GENERATION).unwrap();
    generation.as_object_mut().unwrap().remove("eos_token_id");
    fs::write(
        directory.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    fs::write(
        directory.path().join("generation_config.json"),
        serde_json::to_vec(&generation).unwrap(),
    )
    .unwrap();
    assert_eq!(generation["pad_token_id"], 248044);
    assert_eq!(generation["bos_token_id"], 248044);
    assert!(eos_token_ids_from_sidecar_dir(directory.path())
        .unwrap()
        .is_empty());
    let metadata = std::collections::HashMap::from([(
        "qwen4exp.ple.eos_token_id".to_owned(),
        eredu_gguf::MetadataValue::Uint32(248044),
    )]);
    assert!(gguf_eos_token_ids(&metadata).unwrap().is_empty());
}
