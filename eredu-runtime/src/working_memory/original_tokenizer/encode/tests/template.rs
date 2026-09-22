use super::*;
pub(super) fn json() -> String {
    super::regex::json(false).replace("\"post_processor\":null",r#""post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"start","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"SpecialToken":{"id":"start","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"start","type_id":1}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"start":{"id":"start","ids":[5],"tokens":["<S>"]}}}"#)
}
#[test]
fn template_original_c_and_e_exact_short_flags_empty_and_escaped_result_source_tail() {
    exact_case(&json(), TEXT, &[2, 5, 2, 3, 4]);
}
pub(super) fn exact_case(input: &str, text: &str, expected_ids: &[u32]) {
    let p = TokenizerPlan::prepare_json(input.as_bytes()).unwrap();
    let c = MemoryLedger::tokenizer_required_bytes(&p).unwrap();
    let short = crate::working_memory::memory_fixture::host_ledger(c - 1, 0).unwrap();
    assert!(
        short
            .compile_tokenizer_with(p, || panic!("short C entered compiler"))
            .is_err()
    );
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    for text in [text, ""] {
        for special in [false, true] {
            let sizing = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
            let original = source(&sizing, &input);
            let e =
                MemoryLedger::tokenizer_encode_required_bytes(&original, text, special).unwrap();
            drop(original);
            assert_eq!(sizing.payload_used_bytes().unwrap(), 0);
            for one_short in [true, false] {
                let pool = crate::working_memory::memory_fixture::host_ledger(
                    c + e - u64::from(one_short),
                    0,
                )
                .unwrap();
                let original = source(&pool, &input);
                let result = pool.encode_tokenizer_ids_with(
                    &original,
                    text,
                    special,
                    |p| p,
                    || {
                        assert!(!one_short);
                        assert_eq!(pool.payload_used_bytes().unwrap(), c + e);
                        assert!(matches!(
                            pool.acquire_unquoted(),
                            Err(WorkingMemoryError::ReservedWorkActive)
                        ));
                    },
                    || {},
                );
                if one_short {
                    let error = result.unwrap_err();
                    assert_eq!(error.retained_bytes(), 0);
                    assert!(!error.matches_source(&original));
                    drop(original);
                    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
                    drop(error);
                } else {
                    let output = result.unwrap();
                    let mut expected = if special { vec![5] } else { vec![] };
                    if !text.is_empty() {
                        expected.extend_from_slice(expected_ids);
                    }
                    assert_eq!(output.ids(), expected);
                    assert!(output.matches_source(&original));
                    let alias = original.clone();
                    drop(original);
                    drop(alias);
                    assert_eq!(pool.payload_used_bytes().unwrap(), c + e);
                    drop(output);
                    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
                }
            }
        }
    }
}
#[test]
fn template_source_decoder_failure_and_core_error_retirement() {
    for stage in 0..3 {
        let input = json();
        let plan = TokenizerPlan::prepare_json(input.as_bytes()).unwrap();
        let plan = plan.fail_decode_reservation(stage);
        let c = MemoryLedger::tokenizer_required_bytes(&plan).unwrap();
        let pool = crate::working_memory::memory_fixture::host_ledger(c, 0).unwrap();
        let admitted = std::cell::Cell::new(0usize);
        let error = pool
            .compile_tokenizer_with(plan, || {
                admitted.set(admitted.get() + 1);
                assert_eq!(pool.payload_used_bytes().unwrap(), c);
                assert!(matches!(
                    pool.acquire_unquoted(),
                    Err(WorkingMemoryError::ReservedWorkActive)
                ));
            })
            .unwrap_err();
        assert_eq!(admitted.get(), 1);
        drop(input);
        assert_eq!(error.retained_bytes(), c);
        let failed = error.compiler_failure().unwrap();
        assert!(failed.completed_tokenizer());
        assert!(matches!(
            failed.decode_failure().unwrap().cause(),
            eredu_text::decoder_storage::DecodeSourceError::Allocation(_)
        ));
        let erased = BackendFailure::new(BackendFailureKind::ResourceExhausted, error);
        crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
        assert_eq!(pool.payload_used_bytes().unwrap(), c);
        drop(erased);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}
#[test]
fn template_foreign_pool_refusal_preserves_original_source() {
    errors_case(&json(), TEXT);
}
pub(super) fn errors_case(input: &str, text: &str) {
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let original = source(&pool, &input);
    let c = original.original_bytes();
    let foreign = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let error = foreign
        .encode_tokenizer_ids(&original, text, true)
        .unwrap_err();
    assert_eq!(error.retained_bytes(), 0);
    assert!(!error.matches_source(&original));
    assert_eq!(foreign.payload_used_bytes().unwrap(), 0);
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), c);
    drop(original);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn template_concurrent_e_keeps_two_results_under_one_original_source() {
    super::concurrent_case(&json(), TEXT, true);
}
#[test]
fn template_unwind_and_terminal_poison_preserve_original_retirement_order() {
    super::unwind_case(&json(), TEXT, true);
}
#[test]
fn template_prefix_is_private_on_actual_missing_unknown_failure_and_retains_c_and_e() {
    let input = json().replace("\"unk_token\":\"?\"", "\"unk_token\":\"missing\"");
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let original = source(&pool, &input);
    let c = original.original_bytes();
    let e = MemoryLedger::tokenizer_encode_required_bytes(&original, "z", true).unwrap();
    let error = pool.encode_tokenizer_ids(&original, "z", true).unwrap_err();
    assert_eq!(error.retained_bytes(), e);
    assert!(error.matches_source(&original));
    let failure = error.encoding_failure().unwrap();
    assert!(matches!(failure.cause(), EncodeIdsError::Upstream(_)));
    assert!(failure.source().is_some());
    drop(original);
    assert_eq!(pool.payload_used_bytes().unwrap(), c + e);
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
