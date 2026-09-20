use super::*;
use eredu_core::HostMetadataAccount;
use eredu_runtime::working_memory::DependencyMemoryPolicy;
use serde_json::json;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug, Default)]
struct State {
    calls: AtomicUsize,
    bytes: AtomicUsize,
    stop: AtomicUsize,
    retired: AtomicBool,
}
#[derive(Debug)]
struct Account(Arc<State>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let call = self.0.calls.fetch_add(1, Ordering::SeqCst);
        if call >= self.0.stop.load(Ordering::SeqCst) {
            return Err(HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: 0,
            });
        }
        self.0.bytes.fetch_add(bytes, Ordering::SeqCst);
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
fn account(stop: usize) -> (HostMetadataFunding, Arc<State>) {
    let state = Arc::new(State {
        stop: AtomicUsize::new(usize::MAX),
        ..State::default()
    });
    let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
    state.calls.store(0, Ordering::SeqCst);
    state.bytes.store(0, Ordering::SeqCst);
    state.stop.store(stop, Ordering::SeqCst);
    (funding, state)
}
fn policy(fixed_bytes: usize, bytes_per_input_byte: usize) -> DependencyMemoryPolicy {
    DependencyMemoryPolicy {
        fixed_bytes,
        bytes_per_input_byte,
    }
}

#[test]
fn stock_schema_keeps_full_references_patterns_arrays_and_draft_semantics() {
    for draft in [
        "http://json-schema.org/draft-07/schema#",
        "https://json-schema.org/draft/2020-12/schema",
    ] {
        let schema = json!({
            "$schema": draft, "type":"object", "definitions":{"code":{"type":"string","pattern":"^[A-Z]{2}[0-9]{2}$"}},
            "properties":{"code":{"$ref":"#/definitions/code"},"values":{"type":"array","minItems":2,
                "uniqueItems":true,"items":{"type":"number","minimum":0.25,"maximum":9.5}}},
            "required":["code","values"],"additionalProperties":false
        });
        let source = Source::compile(
            &schema,
            &HostPreparationAuthority::unmanaged(),
            &PreparationFunding::unmanaged(),
        )
        .unwrap();
        let (funding, _) = account(usize::MAX);
        source
            .validate(r#"{"code":"AB17","values":[0.5,9.5]}"#, &funding)
            .unwrap();
        for invalid in [
            r#"{"code":"ab17","values":[0.5,9.5]}"#,
            r#"{"code":"AB17","values":[0.5,0.5]}"#,
            r#"{"code":"AB17","values":[0.1,9.5]}"#,
            r#"{"code":"AB17","values":[0.5,9.6]}"#,
            r#"{"code":"AB17","values":[0.5]}"#,
            r#"{"code":"AB17"}"#,
            r#"{"code":"AB17","values":[0.5,9.5],"extra":1}"#,
            "[]",
        ] {
            assert!(
                source.validate(invalid, &funding).is_err(),
                "{draft}: {invalid}"
            );
        }
    }
}

#[test]
fn compilation_estimate_is_configurable_stable_and_keeps_alias_account() {
    let schema = json!({"type":"object","properties":{"n":{"type":"integer"}}});
    let (account, state) = account(usize::MAX);
    let funding = PreparationFunding::from_metadata(&account).with_memory_policy(policy(123, 7));
    let source =
        Source::compile(&schema, &HostPreparationAuthority::unmanaged(), &funding).unwrap();
    let shell = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<Payload>())
        .unwrap()
        .0
        .pad_to_align()
        .size();
    assert_eq!(
        source.admission_bytes(),
        shell + 123 + 7 * serde_json::to_vec(&schema).unwrap().len()
    );
    let calls = state.calls.load(Ordering::SeqCst);
    let alias = source.clone();
    assert_eq!(source.admission_bytes(), alias.admission_bytes());
    assert_eq!(state.calls.load(Ordering::SeqCst), calls);
    drop((source, funding, account));
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(alias);
    assert!(state.retired.load(Ordering::SeqCst));
}

#[test]
fn compilation_refusals_keep_the_original_account_and_typed_error() {
    for cut in [0, 1] {
        let (account, state) = account(cut);
        let funding = PreparationFunding::from_metadata(&account);
        let error = Source::compile(
            &json!({"type":"object"}),
            &HostPreparationAuthority::unmanaged(),
            &funding,
        )
        .unwrap_err();
        assert_eq!(state.calls.load(Ordering::SeqCst), cut + 1);
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        while cause.downcast_ref::<HostMetadataFundingError>().is_none() {
            cause = cause.source().expect("original funding cause");
        }
        drop((account, funding));
        assert!(!state.retired.load(Ordering::SeqCst));
        drop(error);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}

#[test]
fn validation_failures_keep_input_and_source_until_error_retirement() {
    let (compile_account, compile_state) = account(usize::MAX);
    let source = Source::compile(
        &json!({"type":"object","required":["n"]}),
        &HostPreparationAuthority::unmanaged(),
        &PreparationFunding::from_metadata(&compile_account),
    )
    .unwrap();
    let (invocation, state) = account(usize::MAX);
    let failure = source.validate("{}", &invocation).unwrap_err();
    assert_eq!(failure.tree.as_ref().unwrap().value(), &json!({}));
    drop((source, compile_account, invocation));
    assert!(!compile_state.retired.load(Ordering::SeqCst));
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(failure);
    assert!(compile_state.retired.load(Ordering::SeqCst));
    assert!(state.retired.load(Ordering::SeqCst));
}

#[test]
fn validation_reserves_before_parsing_and_preserves_trailing_input_errors() {
    let source = Source::compile(
        &json!({"type":"object"}),
        &HostPreparationAuthority::unmanaged(),
        &PreparationFunding::unmanaged(),
    )
    .unwrap();
    for cut in [0, 1] {
        let (funding, state) = account(cut);
        let failure = source.validate("{invalid", &funding).unwrap_err();
        assert_eq!(state.calls.load(Ordering::SeqCst), cut + 1);
        let mut cause: &(dyn std::error::Error + 'static) = &failure;
        while cause.downcast_ref::<HostMetadataFundingError>().is_none() {
            cause = cause.source().expect("funding precedes syntax");
        }
    }
    let (funding, _) = account(usize::MAX);
    assert!(
        source
            .validate("{} trailing", &funding)
            .unwrap_err()
            .parse_failure
            .is_some()
    );
}

#[test]
fn local_property_reference_failure_never_accepts_an_invalid_complete_schema() {
    let schema = json!({"type":"object","properties":{"n":{"$ref":"#/$defs/count"}},"$defs":{"count":{"type":"integer","minimum":2}}});
    let funding = PreparationFunding::unmanaged();
    let authority = HostPreparationAuthority::unmanaged();
    let property = Source::compile(&schema["properties"]["n"], &authority, &funding).unwrap_err();
    assert!(property.is_local_property_error());
    let source = Source::compile(&schema, &authority, &funding).unwrap();
    let (invocation, _) = account(usize::MAX);
    assert!(source.validate(r#"{"n":1}"#, &invocation).is_err());
    source.validate(r#"{"n":2}"#, &invocation).unwrap();
    assert!(Source::compile(&json!({"$ref":"#/$defs/absent"}), &authority, &funding).is_err());
}
