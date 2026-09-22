use super::*;
use crate::working_memory::MemoryLedger;
use eredu_text::{stop_storage::StopCompilePlan, tokenizer_storage::TokenizerPlan};

#[test]
fn source_preparation_ceiling_constrains_concurrent_compilers_until_retirement() {
    let json = br#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}}"#;
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let source = pool
        .compile_tokenizer(TokenizerPlan::prepare_json(json).unwrap())
        .unwrap();
    let execution = InferenceExecutionIdentity::default();
    let source_bytes = pool.payload_used_bytes().unwrap();
    let probe = source
        .prepare_text_source_budget(
            &execution,
            crate::working_memory::memory_fixture::host_limits(u64::MAX)
                .resolve(&crate::working_memory::memory_fixture::host_topology())
                .unwrap(),
        )
        .unwrap();
    let controls = pool.payload_used_bytes().unwrap() - source_bytes;
    assert!(controls > 0);
    drop(probe);
    assert_eq!(pool.payload_used_bytes().unwrap(), source_bytes);
    let refusal = source
        .prepare_text_source_budget(
            &execution,
            crate::working_memory::memory_fixture::host_limits(source_bytes)
                .resolve(&crate::working_memory::memory_fixture::host_topology())
                .unwrap(),
        )
        .unwrap_err();
    assert!(matches!(
        refusal.cause,
        WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), source_bytes);
    let encoding = MemoryLedger::tokenizer_encode_required_bytes(&source, "abba", false).unwrap();
    let capacity = source_bytes + controls + encoding;
    let budget = source
        .prepare_text_source_budget(
            &execution,
            crate::working_memory::memory_fixture::host_limits(capacity)
                .resolve(&crate::working_memory::memory_fixture::host_topology())
                .unwrap(),
        )
        .unwrap();
    let encoded = pool.encode_tokenizer_ids(&source, "abba", false).unwrap();
    assert_eq!(encoded.ids(), [0, 1, 1, 0]);
    assert_eq!(pool.payload_used_bytes().unwrap(), capacity);
    std::thread::scope(|scope| {
        let other = &pool;
        scope
            .spawn(move || {
                let error = other
                    .compile_stop_source(StopCompilePlan::prepare_refs(&[]).unwrap())
                    .unwrap_err();
                assert!(matches!(
                    error.accounting_failure(),
                    Some(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { limit_bytes, existing_bytes, .. }))
                 if limit_bytes - existing_bytes == 0));
            })
            .join()
            .unwrap();
    });
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    // A later larger request cannot override the still-live smaller ceiling.
    let too_large = source
        .prepare_text_source_budget(
            &execution,
            crate::working_memory::memory_fixture::host_limits(u64::MAX)
                .resolve(&crate::working_memory::memory_fixture::host_topology())
                .unwrap(),
        )
        .unwrap_err();
    assert!(matches!(
        too_large.cause,
        WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { limit_bytes, existing_bytes, .. })
     if limit_bytes - existing_bytes == 0));
    drop(too_large);
    drop(encoded);
    assert_eq!(pool.payload_used_bytes().unwrap(), source_bytes + controls);
    drop(budget);
    assert_eq!(pool.payload_used_bytes().unwrap(), source_bytes);
    drop(
        pool.compile_stop_source(StopCompilePlan::prepare_refs(&[]).unwrap())
            .unwrap(),
    );
    drop(source);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
