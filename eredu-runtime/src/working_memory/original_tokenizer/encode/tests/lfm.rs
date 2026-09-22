use super::*;
fn json() -> String {
    let mut value: serde_json::Value = serde_json::from_str(&super::template::json()).unwrap();
    value["pre_tokenizer"]=serde_json::from_str(r###"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}"###).unwrap();
    let added = value["added_tokens"].as_array_mut().unwrap();
    for (id, content) in [(6, "Mathias"), (7, "python")] {
        added.push(serde_json::json!({"id":id,"content":content,"single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}));
    }
    value.to_string()
}
const INPUT: &str = "Mathias<S>python hi";
#[test]
fn two_phase_original_exact_one_short_empty_and_both_special_flags_keep_c_e_tails() {
    super::template::exact_case(&json(), INPUT, &[6, 5, 7, 3, 2]);
}
#[test]
fn two_phase_foreign_pool_and_upstream_encoding_failure_keep_custody() {
    super::template::errors_case(&json(), INPUT);
    let input = json().replace("\"unk_token\":\"?\"", "\"unk_token\":\"missing\"");
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let original = source(&pool, &input);
    let c = original.original_bytes();
    let e = MemoryLedger::tokenizer_encode_required_bytes(&original, "Mathiasz", true).unwrap();
    let error = pool
        .encode_tokenizer_ids(&original, "Mathiasz", true)
        .unwrap_err();
    assert!(matches!(
        error.encoding_failure().unwrap().cause(),
        EncodeIdsError::Upstream(_)
    ));
    assert!(error.matches_source(&original));
    drop(original);
    assert_eq!(pool.payload_used_bytes().unwrap(), c + e);
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
#[test]
fn two_phase_concurrent_original_operations_share_source_without_destination_or_workspace_aliases()
{
    super::concurrent_case(&json(), INPUT, true);
}
#[test]
fn two_phase_unwind_and_poison_use_existing_original_retirement_order() {
    super::unwind_case(&json(), INPUT, true);
}
