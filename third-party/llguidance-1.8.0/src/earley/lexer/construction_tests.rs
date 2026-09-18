use super::*;
use crate::api::InvalidLexerStateLimit;
use derivre::RegexAst;

fn literal_spec() -> LexerSpec {
    let mut spec = LexerSpec::new(derivre::ParserAllocationFunding::unenforced()).unwrap();
    spec.setup_lexeme_class(RegexAst::NoMatch).unwrap();
    spec.add_simple_literal("word".to_string(), "abc", false)
        .unwrap();
    spec
}

fn limits(max_lexer_states: usize) -> ParserLimits {
    ParserLimits {
        max_lexer_states,
        precompute_large_lexemes: false,
        ..ParserLimits::default()
    }
}

fn require_error(result: Result<Lexer>) -> anyhow::Error {
    match result {
        Ok(_) => panic!("construction unexpectedly succeeded"),
        Err(error) => error,
    }
}

fn assert_lexer_error(error: &anyhow::Error, expected: &str) {
    match error.downcast_ref::<ParserError>().unwrap() {
        ParserError::LexerError(message) => assert_eq!(message, expected),
        ParserError::ParserError(message) => panic!("unexpected parser error: {message}"),
    }
}

#[test]
fn direct_spec_and_lexer_entries_reject_impossible_limits_before_fuel_use() {
    let spec = literal_spec();
    for configured in [0, 1] {
        let mut limits = limits(configured);
        limits.initial_lexer_fuel = 0;
        let spec_error = spec.to_regex_vec(&mut limits).unwrap_err();
        let lexer_error = require_error(Lexer::from(&spec, &mut limits, false));
        for error in [spec_error, lexer_error] {
            let cause = error.downcast_ref::<InvalidLexerStateLimit>().unwrap();
            assert_eq!(cause.configured, configured);
            assert_eq!(cause.minimum, 2);
        }
        assert_eq!(limits.initial_lexer_fuel, 0);
    }
}

#[test]
fn initial_state_and_first_byte_warming_keep_typed_state_limit_causes() {
    let spec = literal_spec();
    let initial = require_error(Lexer::from(&spec, &mut limits(2), false));
    assert_eq!(initial.to_string(), "initial lexer state");
    assert_lexer_error(&initial, "too many states: 2 >= 2");

    let warming = require_error(Lexer::from(&spec, &mut limits(3), false));
    assert_eq!(warming.to_string(), "warming lexer first bytes");
    assert_lexer_error(&warming, "too many states: 3 >= 3");

    let lex = Lexer::from(&spec, &mut limits(4), false).unwrap();
    assert_eq!(lex.dfa.stats().num_states, 4);
    assert!(!lex.dfa.has_error());
    assert!(lex.allowed_first_byte.is_allowed(b'a' as u32));
    assert!(!lex.allowed_first_byte.is_allowed(b'b' as u32));
}

#[test]
fn warmed_clone_keeps_finite_limit_but_independent_error_state() {
    let spec = literal_spec();
    let mut original = Lexer::from(&spec, &mut limits(4), false).unwrap();
    let mut copied = original.clone();
    let start = copied.start_state(&spec.all_lexemes());
    let after_a = copied.dfa.transition(start, b'a');
    assert!(!after_a.is_dead());
    assert!(!copied.dfa.has_error());
    let before = copied.dfa.stats();
    assert_eq!(copied.dfa.transition(after_a, b'b'), StateID::DEAD);
    assert_eq!(copied.dfa.stats().num_states, before.num_states);
    assert_lexer_error(
        &copied.check_error().unwrap_err(),
        "too many states: 4 >= 4",
    );
    assert!(!original.dfa.has_error());
    assert_eq!(original.start_state(&spec.all_lexemes()), start);
    assert_eq!(original.dfa.transition(start, b'a'), after_a);
}

#[test]
fn prepared_warmup_and_advance_use_the_same_lexical_decisions_and_retain_failure() {
    use super::prepared::PreparedLexer;
    use crate::earley::regexvec::prepared::PreparedRegexVector;
    use std::cell::Cell;
    let mut spec = literal_spec();
    for word in [
        "if", "while", "else", "return", "loop", "match", "break", "const",
    ] {
        spec.add_simple_literal(word.to_string(), word, false)
            .unwrap();
    }
    let reserve = |_bytes| Ok::<(), &'static str>(());
    let vector = || {
        let input = spec
            .root_source_plan()
            .unwrap()
            .compile()
            .unwrap()
            .ordinary(spec.regex_builder.exprset().clone())
            .unwrap();
        PreparedRegexVector::prepare_with_backing(
            input,
            &mut ParserLimits::default(),
            Some(derivre::ParserAllocationFunding::unenforced()),
            &reserve,
        )
        .unwrap()
    };
    let calls = Cell::new(0usize);
    let spending = Cell::new(0usize);
    let mut actual = PreparedLexer::prepare(vector(), &|bytes| {
        calls.set(calls.get() + 1);
        spending.set(spending.get().checked_add(bytes).unwrap());
        Ok::<(), &'static str>(())
    })
    .unwrap();
    assert!(spending.get() > 32);
    let mut ordinary = Lexer::from(&spec, &mut ParserLimits::default(), false).unwrap();
    for byte in 0..=255 {
        assert_eq!(
            actual.allows_first_byte(byte),
            ordinary.allowed_first_byte.is_allowed(byte as u32)
        );
    }
    let selected = spec.all_lexemes();
    let tokens = [
        b"a".to_vec(),
        b"ab".to_vec(),
        b"abc".to_vec(),
        b"i".to_vec(),
        b"if".to_vec(),
        Vec::new(),
        b"z".to_vec(),
    ];
    let trie = TokTrie::from(
        &crate::toktrie::TokRxInfo::new(tokens.len() as u32, 5),
        &tokens,
    );
    actual.precompute_for(&trie, &selected, &reserve).unwrap();
    ordinary.precompute_for(&trie, &selected).unwrap();
    let startup = ParserLimits::default();
    actual
        .prepare_large_lexemes(&trie, &startup, &reserve)
        .unwrap();
    ordinary.prepare_large_lexemes(&trie, &startup).unwrap();
    assert_eq!(actual.vector().fuel(), ordinary.dfa.get_fuel());
    let mut a = actual.start_state(&selected, &reserve).unwrap();
    let mut b = ordinary.start_state(&selected);
    for &byte in b"abci" {
        assert_eq!(
            actual.next_byte(a, &reserve).unwrap(),
            ordinary.next_byte(b)
        );
        let lhs = actual.advance(a, byte, &reserve).unwrap();
        let rhs = ordinary.advance(b, byte, false);
        assert_eq!(format!("{lhs:?}"), format!("{rhs:?}"));
        if let LexerResult::State(state, _) = lhs {
            a = state;
        }
        if let LexerResult::State(state, _) = rhs {
            b = state;
        }
    }
    let warm_calls = calls.get();
    for failed_call in [1, warm_calls] {
        let call = Cell::new(0usize);
        let error = PreparedLexer::prepare(vector(), &|_| {
            call.set(call.get() + 1);
            if call.get() == failed_call {
                Err("funding")
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert_eq!(call.get(), failed_call);
        assert_eq!(error.to_string(), "funding");
        assert!(error.retained_vector().is_some());
    }
    let failure = actual
        .advance(a, b'a', &|_| Err("runtime funding"))
        .unwrap_err();
    assert_eq!(failure.to_string(), "runtime funding");
    let untouched = Cell::new(0usize);
    assert!(actual
        .start_state(&selected, &|_| {
            untouched.set(1);
            Ok::<(), &'static str>(())
        })
        .is_err());
    assert_eq!(untouched.get(), 0);
}

#[test]
fn cached_mutation_uses_borrowed_controls_and_stops_at_each_refusal() {
    use super::prepared::PreparedLexer;
    use crate::earley::regexvec::prepared::PreparedRegexVector;
    use std::cell::Cell;

    let spec = literal_spec();
    let reserve = |_| Ok::<(), &'static str>(());
    let prepared = || {
        let input = spec
            .root_source_plan()
            .unwrap()
            .compile()
            .unwrap()
            .ordinary(spec.regex_builder.exprset().clone())
            .unwrap();
        let vector = PreparedRegexVector::prepare_with_backing(
            input,
            &mut ParserLimits::default(),
            Some(derivre::ParserAllocationFunding::unenforced()),
            &reserve,
        )
        .unwrap();
        let mut lexer = PreparedLexer::prepare(vector, &reserve).unwrap();
        let state = lexer.start_state(&spec.all_lexemes(), &reserve).unwrap();
        (lexer, state)
    };
    let mut ordinary = Lexer::from(&spec, &mut ParserLimits::default(), false).unwrap();
    let ordinary_state = ordinary.start_state(&spec.all_lexemes());
    let expected = format!("{:?}", ordinary.advance(ordinary_state, b'a', false));
    let (mut lexer, state) = prepared();
    let transitions = lexer.vector().transitions_attempted();
    let calls = Cell::new(0);
    for _ in 0..64 {
        // A warmed transition has an 8 KiB operation allowance. Its existing
        // lexer/DFA owners remain in place; constructing either again would
        // exceed this allowance even though this traversal allocates no owner.
        let remaining = Cell::new(8 * 1024usize);
        calls.set(0);
        let result = lexer
            .advance(state, b'a', &|bytes| {
                calls.set(calls.get() + 1);
                remaining.set(
                    remaining
                        .get()
                        .checked_sub(bytes)
                        .ok_or("operation allowance")?,
                );
                Ok::<(), &'static str>(())
            })
            .unwrap();
        assert_eq!(format!("{result:?}"), expected);
    }
    assert_eq!(lexer.vector().transitions_attempted(), transitions);
    for cut in 1..=calls.get() {
        let (mut lexer, state) = prepared();
        let reached = Cell::new(0);
        let error = lexer
            .advance(state, b'a', &|_| {
                reached.set(reached.get() + 1);
                if reached.get() == cut {
                    Err("exact mutation refusal")
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(error.to_string(), "exact mutation refusal");
        assert_eq!(reached.get(), cut);
        assert!(lexer
            .advance(state, b'a', &|_| -> Result<(), &'static str> {
                panic!("failed mutable owner must not request later funding")
            })
            .is_err());
    }
}
