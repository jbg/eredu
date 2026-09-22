use super::*;
use std::{fs::File, io::Write};
const SOURCE: &str = include_str!("../tests/template.jinja");
fn config() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({"chat_template":SOURCE})).unwrap()
}
fn read(bytes: &[u8]) -> PreparedArtifactFileRead {
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(bytes).unwrap();
    PreparedArtifactFileRead::new(file).unwrap()
}
fn sizes(bytes: &[u8]) -> (u64, u64) {
    (
        MemoryLedger::chat_template_file_required_bytes(&read(bytes)).unwrap(),
        MemoryLedger::chat_template_required_bytes(
            &ChatTemplatePlan::prepare_config(bytes, "chat", false).unwrap(),
        )
        .unwrap(),
    )
}
#[test]
fn chat_file_real_i_j_overlap_and_last_source_retirement() {
    let bytes = config();
    let (i, j) = sizes(&bytes);
    let pool = crate::working_memory::memory_fixture::host_ledger(i + j, 0).unwrap();
    let source = pool
        .compile_chat_template_file_with(
            read(&bytes),
            "chat",
            false,
            |_| {
                assert_eq!(pool.payload_used_bytes().unwrap(), i);
                assert!(matches!(
                    pool.acquire_unquoted(),
                    Err(WorkingMemoryError::ReservedWorkActive)
                ));
            },
            || {
                assert_eq!(pool.payload_used_bytes().unwrap(), i);
                crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
            },
            |pool, plan| {
                assert_eq!(
                    MemoryLedger::chat_template_required_bytes(&plan).unwrap(),
                    j
                );
                pool.compile_chat_template_with(plan, || {
                    assert_eq!(pool.payload_used_bytes().unwrap(), i + j);
                    assert!(matches!(
                        pool.acquire_unquoted(),
                        Err(WorkingMemoryError::ReservedWorkActive)
                    ));
                })
            },
        )
        .unwrap();
    assert_eq!(source.name(), "chat");
    assert_eq!(source.original_bytes(), j);
    assert_eq!(pool.payload_used_bytes().unwrap(), j);
    assert_eq!(pool.payload_peak_bytes().unwrap(), i + j);
    let alias = source.clone();
    drop(source);
    assert_eq!(pool.payload_used_bytes().unwrap(), j);
    drop(alias);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn chat_file_exact_minus_one_at_i_and_j_keeps_actual_prefixes() {
    let bytes = config();
    let (i, j) = sizes(&bytes);
    let short = crate::working_memory::memory_fixture::host_ledger(i - 1, 0).unwrap();
    let error = short
        .compile_chat_template_file_with(
            read(&bytes),
            "chat",
            false,
            |_| panic!("short I entered reserve"),
            || panic!("short I read"),
            |_, _| panic!("short I reached J"),
        )
        .unwrap_err();
    assert_eq!(error.input_bytes(), 0);
    assert_eq!(error.input_capacity(), 0);
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
    assert!(
        matches!(error.accounting_failure(),Some(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. })) if *required_bytes==i && (limit_bytes - existing_bytes)==i-1)
    );
    let short = crate::working_memory::memory_fixture::host_ledger(i + j - 1, 0).unwrap();
    let error = short
        .compile_chat_template_file(read(&bytes), "chat", false)
        .unwrap_err();
    assert_eq!(error.input_bytes(), i);
    assert_eq!(error.input_capacity(), bytes.len());
    assert_eq!(error.filled_bytes(), bytes.len());
    assert_eq!(error.compiler_failure().unwrap().retained_bytes(), 0);
    assert!(
        matches!(error.accounting_failure(),Some(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. })) if *required_bytes==j && (limit_bytes - existing_bytes)==j-1)
    );
    crate::working_memory::memory_fixture::assert_unquoted_idle(&short);
    assert_eq!(short.payload_used_bytes().unwrap(), i);
    drop(error);
    assert_eq!(short.payload_used_bytes().unwrap(), 0);
}
#[test]
fn chat_file_real_reserve_changed_file_and_late_j_errors_are_retained() {
    let bytes = config();
    let (i, j) = sizes(&bytes);
    let pool = crate::working_memory::memory_fixture::host_ledger(i + j, 0).unwrap();
    let error = pool
        .compile_chat_template_file_with(
            read(&bytes),
            "chat",
            false,
            |capacity| *capacity = usize::MAX,
            || panic!("reserve failure reached read"),
            |_, _| panic!("reserve failure reached J"),
        )
        .unwrap_err();
    assert_eq!(error.input_bytes(), i);
    assert_eq!(error.input_capacity(), 0);
    assert_eq!(error.filled_bytes(), 0);
    assert!(matches!(
        error.cause,
        Cause::Input(original_file::Cause::Reserve(_))
    ));
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let mut path = tempfile::NamedTempFile::new().unwrap();
    path.write_all(&bytes).unwrap();
    let prepared = PreparedArtifactFileRead::new(File::open(path.path()).unwrap()).unwrap();
    path.as_file().set_len(0).unwrap();
    let error = pool
        .compile_chat_template_file(prepared, "chat", false)
        .unwrap_err();
    assert!(error.read_failure().is_some());
    assert_eq!(error.input_bytes(), i);
    assert_eq!(error.input_capacity(), bytes.len());
    assert_eq!(error.filled_bytes(), 0);
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    let invalid = br#"{"chat_template":"{% if %}"}"#;
    let i = MemoryLedger::chat_template_file_required_bytes(&read(invalid)).unwrap();
    let plan = ChatTemplatePlan::prepare_config(invalid, "chat", false).unwrap();
    let j = MemoryLedger::chat_template_required_bytes(&plan).unwrap();
    let error = pool
        .compile_chat_template_file(read(invalid), "chat", false)
        .unwrap_err();
    assert_eq!(error.input_bytes(), i);
    let failed = error.compiler_failure().unwrap();
    assert_eq!(failed.retained_bytes(), j);
    assert!(failed.compiler_failure().is_some());
    assert_eq!(pool.payload_used_bytes().unwrap(), i + j);
    crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn named_file_selection_retains_exact_source_and_original_overlap() {
    let bytes = serde_json::to_vec(&serde_json::json!({"chat_template": [
        {"name":"default", "template":"plain:{{ messages[0].content }}"},
        {"name":"tool_use", "template":"tools:{{ messages[0].content }}|{{ tools|tojson }}"},
    ]}))
    .unwrap();
    let selected = eredu_text::tokenizer::load_model_chat_template_from_str(
        std::str::from_utf8(&bytes).unwrap(),
    )
    .unwrap()
    .unwrap();
    for has_tools in [false, true] {
        let input = MemoryLedger::chat_template_file_required_bytes(&read(&bytes)).unwrap();
        let source_bytes = MemoryLedger::chat_template_required_bytes(
            &ChatTemplatePlan::prepare_config(&bytes, "named", has_tools).unwrap(),
        )
        .unwrap();
        let pool =
            crate::working_memory::memory_fixture::host_ledger(input + source_bytes, 0).unwrap();
        let source = pool
            .compile_chat_template_file(read(&bytes), "named", has_tools)
            .unwrap();
        assert!(source.matches_selection(&selected, "named", has_tools));
        assert!(!source.matches_selection(&selected, "named", !has_tools));
        assert_eq!(pool.payload_peak_bytes().unwrap(), input + source_bytes);
        assert_eq!(pool.payload_used_bytes().unwrap(), source_bytes);
        let alias = source.clone();
        drop(source);
        assert_eq!(pool.payload_used_bytes().unwrap(), source_bytes);
        drop(alias);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        let short = crate::working_memory::memory_fixture::host_ledger(input + source_bytes - 1, 0)
            .unwrap();
        let failure = short
            .compile_chat_template_file(read(&bytes), "named", has_tools)
            .unwrap_err();
        assert!(matches!(
            failure.accounting_failure(),
            Some(WorkingMemoryError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded { .. }
            ))
        ));
        assert_eq!(short.payload_used_bytes().unwrap(), input);
        drop(failure);
        assert_eq!(short.payload_used_bytes().unwrap(), 0);
    }
}
