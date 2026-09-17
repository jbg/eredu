use super::*;
use std::error::Error as _;
fn converted(error: &(dyn std::error::Error + 'static)) {
    let neutral = cause::<eredu_core::BackendFailure>(error).unwrap();
    assert_eq!(neutral.kind(), eredu_core::BackendFailureKind::Busy);
    assert_eq!(neutral.operation(), "portable-provider-hook");
    assert!(neutral.source().unwrap().is::<MockError>());
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
        .generate_tokens(vec![0], TextGenerationConfig::new(config))
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
        let chat = model.prepare_chat(request()).unwrap();
        let prepared = model
            .prepare_observed_token_ids(&chat, vec![0], settings(), CapturePlan::none(), limits())
            .unwrap();
        calls.borrow_mut().reject_submission = true;
        if controlled {
            let mut run = model
                .start_controlled_text(prepared, &[], Default::default(), |_| {
                    ControlFlow::Continue(())
                })
                .unwrap();
            let error = run.step(|_| ControlFlow::Continue(())).unwrap_err();
            converted(&error);
            assert!(run.token_ids().is_empty());
        } else {
            let error = model
                .generate_observed_text(prepared, &[], Default::default(), |_| {
                    ControlFlow::Continue(())
                })
                .unwrap_err();
            converted(&error);
        }
        assert!(
            calls.borrow().filters.is_empty(),
            "submission failure precedes sampling"
        );
    }
}
