use super::*;
use eredu_core::HostMetadataAccount;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Debug)]
struct State {
    remaining: AtomicUsize,
    retired: AtomicBool,
}
#[derive(Debug)]
struct Account(Arc<State>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        self.0
            .remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(bytes))
            .map(|_| ())
            .map_err(|available| HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: available as u64,
            })
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
fn account(limit: usize) -> (HostMetadataFunding, Arc<State>) {
    let state = Arc::new(State {
        remaining: AtomicUsize::new(usize::MAX),
        retired: AtomicBool::new(false),
    });
    let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
    state.remaining.store(limit, Ordering::SeqCst);
    (funding, state)
}

#[test]
fn root_kinds_and_unicode_strings_retain_their_original_source() {
    let (funding, state) = account(usize::MAX);
    for (input, kind) in [
        ("{\"a\":[7,11]}", OriginalJsonValueKind::Object),
        ("[false,null]", OriginalJsonValueKind::Array),
        ("18446744073709551615", OriginalJsonValueKind::Number),
        ("-0.0", OriginalJsonValueKind::Number),
        ("true", OriginalJsonValueKind::Bool),
        ("null", OriginalJsonValueKind::Null),
    ] {
        let parsed = OriginalJsonValue::parse(input, &funding).unwrap();
        assert_eq!(parsed.source(), input);
        assert_eq!(parsed.kind(), kind);
        assert_eq!(parsed.string(), None);
    }
    let parsed = OriginalJsonValue::parse(r#""É\u754c\ud83d\ude42\u0000""#, &funding).unwrap();
    assert_eq!(parsed.kind(), OriginalJsonValueKind::String);
    assert_eq!(parsed.string(), Some("É界🙂\0"));
    let text = parsed.into_string().unwrap();
    drop(funding);
    assert!(!state.retired.load(Ordering::SeqCst));
    assert_eq!(text.as_str(), "É界🙂\0");
    drop(text);
    assert!(state.retired.load(Ordering::SeqCst));
}

#[test]
fn complete_syntax_validation_preserves_a_root_string_on_trailing_failure() {
    let (funding, state) = account(usize::MAX);
    for input in ["1e400", "[1,]", r#"{"x":"\ud800"}"#, "true false"] {
        assert!(
            OriginalJsonValue::parse(input, &funding).is_err(),
            "{input}"
        );
    }
    let error = OriginalJsonValue::parse(r#""kept 界" trailing"#, &funding).unwrap_err();
    assert_eq!(error.partial.as_ref().unwrap().as_str(), "kept 界");
    drop(funding);
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(error);
    assert!(state.retired.load(Ordering::SeqCst));
}

#[test]
fn headroom_refusal_precedes_parsing_and_string_copy_keeps_separate_admission() {
    let (funding, _) = account(0);
    let error = OriginalJsonValue::parse("invalid", &funding).unwrap_err();
    assert!(matches!(error.cause, Cause::Funding(_)));
    assert!(
        std::error::Error::source(&error)
            .unwrap()
            .is::<HostMetadataFundingError>()
    );
    assert!(error.partial.is_none());

    let policy = DependencyMemoryPolicy {
        fixed_bytes: 17,
        bytes_per_input_byte: 3,
    };
    let input = "\"kept\"";
    let estimate = policy.estimate(input.len()).unwrap()
        + size_of::<OriginalJsonValue<'_>>()
        + size_of::<OriginalJsonValueError>();
    let (funding, state) = account(estimate);
    let error = OriginalJsonValue::parse_with_memory_policy(input, &funding, policy).unwrap_err();
    assert!(matches!(error.cause, Cause::Funding(_)));
    assert!(error.partial.is_none());
    assert_eq!(state.remaining.load(Ordering::SeqCst), 0);
    drop(funding);
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(error);
    assert!(state.retired.load(Ordering::SeqCst));
}
