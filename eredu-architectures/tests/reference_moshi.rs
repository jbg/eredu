//! Dedicated backend-independent Moshi-family reference target.
//!
//! The shared scalar backend also exercises the existing decoder families, so
//! keep one implementation of those backend primitives while publishing a
//! separately runnable Moshi integration target.

#[path = "reference_llama.rs"]
mod shared_reference_backend;

#[test]
fn released_native_and_personaplex_profiles_share_one_portable_model_contract() {
    let native = eredu_architectures::moshi::MoshiConfig::native_v0_1().unwrap();
    let persona = eredu_architectures::moshi::MoshiConfig::from_json(
        r#"{"model_type":"personaplex","version":"7b-v1"}"#,
    )
    .unwrap();
    for config in [&native, &persona] {
        assert_eq!(config.family(), "moshi");
        assert_eq!(config.temporal().parameter_root(), "transformer");
        assert_eq!(
            config.temporal().attention_window(),
            config.temporal().context() + 1
        );
        assert_eq!(
            config.depth_template().attention_window(),
            config.depth_template().context()
        );
        assert_eq!(
            config.depth_transformer(0).unwrap().parameter_root(),
            "depformer.slices.0.transformer"
        );
        let layout = eredu_architectures::moshi::state_layout(config).unwrap();
        assert_eq!(layout.segments()[0].id().as_str(), "temporal");
        assert_eq!(layout.segments()[1].id().as_str(), "depth");
    }
    assert_ne!(
        native.architecture_fingerprint(),
        persona.architecture_fingerprint()
    );
}

#[test]
fn decision_domains_include_exact_released_padding_rows() {
    for config in [
        eredu_architectures::moshi::MoshiConfig::native_v0_1().unwrap(),
        eredu_architectures::moshi::MoshiConfig::from_json(
            r#"{"model_type":"personaplex","version":"7b-v1"}"#,
        )
        .unwrap(),
    ] {
        let boundary = eredu_architectures::moshi::DecisionBoundary::new(&config).unwrap();
        assert_eq!(
            boundary.text_token_domain().cardinality(),
            config.text_vocabulary_size() as usize + 1
        );
        assert_eq!(
            boundary.audio_token_domain().cardinality(),
            config.audio_vocabulary_size() as usize + 1
        );
    }
}
