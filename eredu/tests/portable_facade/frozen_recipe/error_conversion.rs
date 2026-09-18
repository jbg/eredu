use super::*;
use std::error::Error as _;
fn converted(error: &(dyn std::error::Error + 'static)) {
    let neutral = cause::<eredu_core::BackendFailure>(error)
        .unwrap_or_else(|| panic!("neutral backend cause missing: {error:?}"));
    assert_eq!(neutral.kind(), eredu_core::BackendFailureKind::Busy);
    assert_eq!(neutral.operation(), "portable-provider-hook");
    assert!(
        cause::<MockError>(neutral).is_some(),
        "the retained cursor must preserve the original provider error"
    );
}
#[test]
fn generated_token_public_observation_uses_provider_hook() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut model, _, _) = fixture(&pool, true);
    let config = model
        .resolve_generation_config(GenerationConfigOverrides {
            max_new_tokens: Some(1),
            ..Default::default()
        })
        .unwrap();
    let token = model
        .generate_tokens(vec![0].into(), TextGenerationConfig::new(config))
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            TOKEN_READ_FAILURE.with(|flag| flag.set(self.0));
        }
    }
    let restore = Restore(TOKEN_READ_FAILURE.with(|flag| flag.replace(true)));
    let error = token.token_id().unwrap_err();
    converted(&error);
    drop(restore);
    assert!(token.token_id().is_ok());
}
#[test]
fn controlled_and_uninterrupted_facade_errors_use_provider_hook() {
    for controlled in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut model, calls, _) = fixture(&pool, true);
        let cancel = eredu_core::GenerationCancellationToken::new();
        let source = model.chat_source(false, &cancel).unwrap().unwrap();
        let chat = model
            .prepare_chat(&source, &request(), original_sources::CAPACITY, &cancel)
            .unwrap()
            .unwrap();
        calls.borrow_mut().reject_submission = true;
        let run = model
            .start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancel)
            .unwrap()
            .unwrap();
        let error = if controlled {
            run.advance(&cancel, &mut |_| {})
                .err()
                .expect("submission failure")
        } else {
            run.run(&cancel, &mut |_| {}).unwrap_err()
        };
        assert_eq!(
            error.backend_failure().unwrap().kind(),
            eredu_core::BackendFailureKind::Busy
        );
        converted(&error);
        assert!(
            calls.borrow().filters.is_empty(),
            "submission failure precedes sampling"
        );
    }
}

#[test]
fn prepared_chat_startup_preserves_provider_classification_and_source() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut model, calls, _) = fixture(&pool, true);
    let cancel = eredu_core::GenerationCancellationToken::new();
    let source = model.chat_source(false, &cancel).unwrap().unwrap();
    let chat = model
        .prepare_chat(&source, &request(), original_sources::CAPACITY, &cancel)
        .unwrap()
        .unwrap();
    calls.borrow_mut().reject_sampling = true;
    let error = model
        .start_prepared_chat(PreparedChatRequest::new(&chat, settings()), &cancel)
        .err()
        .expect("sampling preparation must fail before submission");
    converted(&error);
    assert_eq!(
        error.backend_failure().unwrap().kind(),
        eredu_core::BackendFailureKind::Busy
    );
    assert!(calls.borrow().filters.is_empty());
}
