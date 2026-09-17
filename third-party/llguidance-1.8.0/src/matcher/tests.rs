use super::*;
use std::{
    fmt,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

#[test]
fn parser_construction_causes_keep_their_stop_classification() {
    for (cause, expected) in [
        (
            ParserError::LexerError("finite state table exhausted".into()),
            StopReason::LexerTooComplex,
        ),
        (
            ParserError::ParserError("finite row exhausted".into()),
            StopReason::ParserTooComplex,
        ),
    ] {
        let message = cause.message();
        let matcher = Matcher::new(Err(anyhow::Error::new(cause).context("parser construction")));
        assert_eq!(matcher.stop_reason(), expected);
        let mut branch = matcher.deep_clone();
        let error = branch.compute_mask_or_eos().unwrap_err();
        assert_eq!(
            error.downcast_ref::<ParserError>().unwrap().message(),
            message
        );
        assert_eq!(branch.stop_reason(), expected);
        assert!(branch.consume_token(17).is_err());
    }
}

#[derive(Debug)]
struct ExternalFailure {
    retired: Arc<AtomicUsize>,
}

impl fmt::Display for ExternalFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("external factory failure")
    }
}

impl std::error::Error for ExternalFailure {}

impl Drop for ExternalFailure {
    fn drop(&mut self) {
        self.retired.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn unrelated_constructor_error_payload_retires_before_matcher_publication() {
    let retired = Arc::new(AtomicUsize::new(0));
    let matcher = Matcher::new(Err(anyhow::Error::new(ExternalFailure {
        retired: retired.clone(),
    })));
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    let mut branch = matcher.deep_clone();
    let diagnostic = branch.compute_mask().unwrap_err();
    assert_eq!(diagnostic.to_string(), "external factory failure");
    assert!(diagnostic.downcast_ref::<ExternalFailure>().is_none());
    assert_eq!(matcher.stop_reason(), StopReason::InternalError);
    drop(matcher);
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert!(branch.compute_mask_or_eos().is_err());
}

#[test]
fn caught_parser_panic_does_not_panic_again_while_probing_the_cause() {
    use crate::{api::TopLevelGrammar, toktrie::ApproximateTokEnv, ParserFactory};

    let mut factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap();
    factory.quiet();
    let mut matcher = Matcher::new(factory.create_parser(TopLevelGrammar::from_json_schema(
        serde_json::json!({"type": "string", "pattern": "^[a-z]+$"}),
    )));
    let failure = matcher.with_inner::<()>(|inner| {
        inner
            .parser
            .parser
            .with_recognizer(|_| panic!("recognizer callback failed"))
    });
    let error = failure.unwrap_err();
    assert!(error.to_string().contains("recognizer callback failed"));
    assert!(matcher.is_error());
    assert!(matcher.is_stopped());
    assert!(error.downcast_ref::<ParserError>().is_none());
    let mut copy = matcher.deep_clone();
    assert!(copy
        .compute_mask_or_eos()
        .unwrap_err()
        .to_string()
        .contains("recognizer callback failed"));
}
