//! Public construction and runtime coverage for bounded lexer-state insertion.

use llguidance::{
    api::{InvalidLexerStateLimit, StopReason, TopLevelGrammar},
    earley::ParserError,
    toktrie::ApproximateTokEnv,
    Matcher, ParserFactory, TokenParser,
};

fn grammar() -> TopLevelGrammar {
    TopLevelGrammar::from_lark(r#"start: /[a-z]{3,8}/ ":" ("17" | "23" | "41") "\n""#.to_owned())
}

fn factory(max_states: usize) -> ParserFactory {
    let mut factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env())
        .expect("the byte tokenizer supports the parser factory");
    factory.quiet();
    let limits = factory.limits_mut();
    limits.max_lexer_states = max_states;
    // Exercise ordinary mask construction, rather than optional large-lexeme
    // precomputation. Required constructor first-byte warming still runs.
    limits.precompute_large_lexemes = false;
    factory
}

fn assert_invalid_limit(error: &anyhow::Error, configured: usize) {
    let cause = error
        .downcast_ref::<InvalidLexerStateLimit>()
        .unwrap_or_else(|| panic!("missing configuration cause: {error:#}"));
    assert_eq!(cause.configured, configured);
    assert_eq!(cause.minimum, 2);
    assert!(error.downcast_ref::<ParserError>().is_none());
}

fn assert_lexer_limit(error: &anyhow::Error) -> &str {
    let Some(ParserError::LexerError(reason)) = error.downcast_ref::<ParserError>() else {
        panic!("missing typed lexer cause: {error:#}");
    };
    assert!(reason.contains("too many states"), "{reason}");
    assert!(error.downcast_ref::<InvalidLexerStateLimit>().is_none());
    reason
}

#[test]
fn zero_and_one_state_limits_reject_before_grammar_compilation_and_survive_matcher_copies() {
    for configured in [0, 1] {
        let factory = factory(configured);
        for grammar in [grammar(), TopLevelGrammar::from_lark("start: [".to_owned())] {
            let error = factory
                .create_parser(grammar)
                .err()
                .expect("invalid scalar limits must reject even malformed grammars");
            assert_invalid_limit(&error, configured);

            let matcher = Matcher::new(Err(error));
            let mut ordinary_copy = matcher.clone();
            let mut deep_copy = matcher.deep_clone();
            let mut second_deep_copy = deep_copy.deep_clone();
            let mut matcher = matcher;
            for failed in [
                &mut matcher,
                &mut ordinary_copy,
                &mut deep_copy,
                &mut second_deep_copy,
            ] {
                assert!(failed.is_error());
                assert!(failed.is_stopped());
                for _ in 0..2 {
                    assert_invalid_limit(&failed.compute_mask_or_eos().unwrap_err(), configured);
                    assert_invalid_limit(
                        &failed.consume_token(u32::from(b'm')).unwrap_err(),
                        configured,
                    );
                }
                assert_invalid_limit(&failed.tok_env().err().unwrap(), configured);
                assert_invalid_limit(&failed.last_step_stats().err().unwrap(), configured);
            }
        }
    }
}

fn assert_failed_matcher_preserves_lexer_cause(matcher: &mut Matcher, reason: &str) {
    assert!(matcher.is_error());
    assert!(matcher.is_stopped());
    assert_eq!(matcher.stop_reason(), StopReason::LexerTooComplex);
    for _ in 0..2 {
        assert_eq!(
            assert_lexer_limit(&matcher.compute_mask_or_eos().unwrap_err()),
            reason
        );
        assert_eq!(
            assert_lexer_limit(&matcher.consume_token(u32::from(b'a')).unwrap_err()),
            reason,
        );
    }
    assert_eq!(
        assert_lexer_limit(&matcher.tok_env().err().unwrap()),
        reason
    );
    assert_eq!(
        assert_lexer_limit(&matcher.last_step_stats().err().unwrap()),
        reason
    );
}

#[test]
fn exact_observed_construction_limit_succeeds_and_one_short_preserves_typed_failure() {
    let generous = factory(4096).create_parser(grammar()).unwrap();
    let required = generous.parser.lexer_stats().num_states;
    assert!(
        required > 2,
        "the grammar must need more than sentinel states"
    );
    assert!(!generous.parser.lexer_stats().error);
    assert!(generous.parser.get_error().is_none());

    let exact = factory(required).create_parser(grammar()).unwrap();
    assert_eq!(exact.parser.lexer_stats().num_states, required);
    assert!(!exact.parser.lexer_stats().error);
    assert!(exact.parser.get_error().is_none());

    let error = factory(required - 1)
        .create_parser(grammar())
        .err()
        .expect("one fewer state must reject during construction");
    let reason = assert_lexer_limit(&error).to_owned();
    let mut matcher = Matcher::new(Err(error));
    let mut ordinary_copy = matcher.clone();
    let mut deep_copy = matcher.deep_clone();
    for failed in [&mut matcher, &mut ordinary_copy, &mut deep_copy] {
        assert_failed_matcher_preserves_lexer_cause(failed, &reason);
    }
}

fn consume_parser_prefix(parser: &mut TokenParser, bytes: &[u8]) {
    for &byte in bytes {
        assert!(parser.compute_mask().unwrap().is_allowed(u32::from(byte)));
        assert_eq!(parser.consume_token(u32::from(byte)).unwrap(), 0);
        assert!(!parser.check_stop().unwrap());
    }
}

fn runtime_limited_factory() -> ParserFactory {
    let mut probe = factory(4096).create_parser(grammar()).unwrap();
    probe.start_without_prompt();
    consume_parser_prefix(&mut probe, b"mar");
    // Use a real post-prefix count: construction and these masks fit, while a
    // longer identifier needs further derivatives. TokenParser.limits is a
    // separate public copy, so changing it would not exercise the lexer cap.
    factory(probe.parser.lexer_stats().num_states)
}

#[test]
fn lexer_state_limit_preserves_parser_error_and_stops_matcher() {
    let factory = runtime_limited_factory();
    let mut parser = factory.create_parser(grammar()).unwrap();
    let cap = parser.limits.max_lexer_states;
    assert!(parser.parser.get_error().is_none());
    assert!(parser.parser.lexer_stats().num_states <= cap);
    parser.start_without_prompt();

    // Bootstrap and first-byte states can already be warm. Existing hits must
    // remain usable below the limit; follow real valid input until a mask needs
    // a previously unseen derivative instead of assuming the first mask grows.
    let sample = b"marigold:23\n";
    let mut committed = Vec::new();
    let error = sample
        .iter()
        .find_map(|&byte| match parser.compute_mask() {
            Err(error) => Some(error),
            Ok(mask) => {
                assert!(mask.is_allowed(u32::from(byte)));
                assert_eq!(parser.consume_token(u32::from(byte)).unwrap(), 0);
                parser.check_stop().unwrap();
                committed.push(byte);
                None
            }
        })
        .expect("this valid sample must require another lexer state");
    assert!(
        committed.starts_with(b"mar"),
        "the measured prefix must fit"
    );
    assert!(sample.starts_with(&committed));
    assert_eq!(parser.stop_reason(), StopReason::LexerTooComplex);
    let Some(ParserError::LexerError(reason)) = parser.parser.get_error() else {
        panic!("expected the original typed lexer error: {error:#}");
    };
    assert!(reason.contains("too many states"), "{reason}");
    assert!(error.to_string().contains(&reason), "{error:#}");
    assert_eq!(parser.parser.get_bytes(), committed.as_slice());
    assert_eq!(parser.parser.lexer_stats().num_states, cap);

    // Matcher is the facade's entry point. It must preserve the same cause and
    // remain failed on later requests or copies, rather than emit a fallback ID.
    let mut matcher = Matcher::new(factory.create_parser(grammar()));
    for &byte in &committed {
        assert!(matcher
            .compute_mask_or_eos()
            .unwrap()
            .is_allowed(u32::from(byte)));
        matcher.consume_token(u32::from(byte)).unwrap();
    }
    let error = matcher.compute_mask_or_eos().unwrap_err();
    assert_eq!(assert_lexer_limit(&error), reason);
    let mut ordinary_copy = matcher.clone();
    let mut copied_failure = matcher.deep_clone();
    for failed in [&mut matcher, &mut ordinary_copy, &mut copied_failure] {
        assert_failed_matcher_preserves_lexer_cause(failed, &reason);
    }
}

#[test]
fn finite_state_caps_survive_deep_clone_and_fail_independently_after_nonzero_progress() {
    let mut original = runtime_limited_factory().create_parser(grammar()).unwrap();
    original.start_without_prompt();
    consume_parser_prefix(&mut original, b"m");
    let source_count = original.parser.lexer_stats().num_states;
    let cap = original.limits.max_lexer_states;
    let mut branch = original.deep_clone();
    assert_eq!(branch.limits.max_lexer_states, cap);

    consume_parser_prefix(&mut branch, b"ar");
    let branch_error = branch.compute_mask().unwrap_err();
    let Some(ParserError::LexerError(branch_reason)) = branch.parser.get_error() else {
        panic!("missing branch lexer failure: {branch_error:#}");
    };
    assert!(branch_reason.contains("too many states"), "{branch_reason}");
    assert_eq!(branch.parser.lexer_stats().num_states, cap);
    assert_eq!(branch.parser.get_bytes(), b"mar");
    assert_eq!(original.parser.get_bytes(), b"m");
    assert_eq!(original.parser.lexer_stats().num_states, source_count);
    assert!(original.parser.get_error().is_none());
    assert_eq!(original.stop_reason(), StopReason::NotStopped);

    // The source still advances through its own valid nonzero prefix. Its
    // independently retained cap then rejects the same next missing state.
    consume_parser_prefix(&mut original, b"ar");
    let original_error = original.compute_mask().unwrap_err();
    let Some(ParserError::LexerError(original_reason)) = original.parser.get_error() else {
        panic!("missing source lexer failure: {original_error:#}");
    };
    assert_eq!(original_reason, branch_reason);
    assert_eq!(original.parser.lexer_stats().num_states, cap);
    assert_eq!(original.parser.get_bytes(), b"mar");

    let mut failed_copy = branch.deep_clone();
    assert!(failed_copy.compute_mask().is_err());
    let Some(ParserError::LexerError(copied_reason)) = failed_copy.parser.get_error() else {
        panic!("a failed deep clone lost the retained lexer failure");
    };
    assert_eq!(copied_reason, branch_reason);
    assert_eq!(failed_copy.parser.lexer_stats().num_states, cap);
    assert_eq!(failed_copy.parser.get_bytes(), b"mar");
}

fn consume_matching_masks(matcher: &mut Matcher, reference: &mut Matcher, bytes: &[u8]) {
    for &byte in bytes {
        let mask = matcher.compute_mask_or_eos().unwrap();
        let ordinary_mask = reference.compute_mask().unwrap();
        assert_eq!(
            mask.iter().collect::<Vec<_>>(),
            ordinary_mask.iter().collect::<Vec<_>>(),
            "different masks before byte {byte:?}",
        );
        assert!(mask.is_allowed(u32::from(byte)), "rejected byte {byte:?}");
        matcher.consume_token(u32::from(byte)).unwrap();
        reference.consume_token(u32::from(byte)).unwrap();
    }
}

#[test]
fn adequate_state_limit_preserves_nonzero_mask_commit_and_independent_clone_parity() {
    let factory = factory(4096);
    let eos = factory.tok_env().tok_trie().eos_tokens()[0];
    let mut original = Matcher::new(factory.create_parser(grammar()));
    let mut reference = Matcher::new(factory.create_parser(grammar()));
    let initial = original.compute_mask().unwrap();
    assert!(initial.is_allowed(u32::from(b'm')));
    assert!(!initial.is_allowed(u32::from(b'9')));
    assert!(!initial.is_allowed(eos));
    consume_matching_masks(&mut original, &mut reference, b"mar");

    let mut branch = original.deep_clone();
    let mut branch_reference = Matcher::new(factory.create_parser(grammar()));
    for &byte in b"mar" {
        assert!(branch_reference
            .compute_mask()
            .unwrap()
            .is_allowed(u32::from(byte)));
        branch_reference.consume_token(u32::from(byte)).unwrap();
    }
    drop(factory);

    consume_matching_masks(&mut original, &mut reference, b"ket:17\n");
    assert!(original.is_accepting().unwrap());
    assert!(original.compute_mask_or_eos().unwrap().is_allowed(eos));
    assert!(!branch.is_accepting().unwrap());

    consume_matching_masks(&mut branch, &mut branch_reference, b"ble:23\n");
    assert!(branch.is_accepting().unwrap());
    assert!(branch.compute_mask_or_eos().unwrap().is_allowed(eos));
    assert!(original.is_accepting().unwrap());
    assert!(!original.is_error());
    assert!(!branch.is_error());
}
