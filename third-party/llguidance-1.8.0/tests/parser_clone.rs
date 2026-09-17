//! Public clone conformance. State-copy lifetime/count coverage lives in private tests;
//! these cases do not claim a complete parser or snapshot byte bound.

use llguidance::{
    api::{GrammarInit, StopReason, TopLevelGrammar},
    derivre::RegexAst,
    earley::{lexerspec::LexerSpec, Grammar, SymbolProps},
    toktrie::{ApproximateTokEnv, InferenceCapabilities, TokEnv, TokEnvWithTrie},
    Matcher, ParserFactory, TokenParser,
};
use std::{sync::Arc, time::Duration};

fn captured_grammar(no_forcing: bool, open_word: bool) -> GrammarInit {
    let mut lexer = LexerSpec::new().unwrap();
    lexer.setup_lexeme_class(RegexAst::NoMatch).unwrap();
    lexer.no_forcing = no_forcing;
    let mut grammar = Grammar::new(Some("captured clone fixture".to_owned()));
    let start = grammar.fresh_symbol_ext(
        "start",
        SymbolProps {
            is_start: true,
            ..SymbolProps::default()
        },
    );
    let mut sequence = Vec::new();
    let parts = if open_word {
        vec![("word", RegexAst::Regex("[a-z]+".to_owned()), true)]
    } else {
        vec![
            ("prefix", RegexAst::Literal("record:".to_owned()), false),
            ("word", RegexAst::Regex("[a-z]{3,12}".to_owned()), true),
            ("separator", RegexAst::Literal(":".to_owned()), false),
            ("answer", RegexAst::Regex("17|23".to_owned()), true),
            ("end", RegexAst::Literal("\n".to_owned()), false),
        ]
    };
    for (name, expression, capture) in parts {
        let lexeme = lexer
            .add_greedy_lexeme(name.to_owned(), expression, false, None, usize::MAX)
            .unwrap();
        let symbol = grammar.fresh_symbol_ext(
            name,
            SymbolProps {
                capture_name: capture.then(|| name.to_owned()),
                ..SymbolProps::default()
            },
        );
        grammar.make_terminal(symbol, lexeme, &lexer).unwrap();
        sequence.push(symbol);
    }
    grammar.add_rule(start, sequence).unwrap();
    GrammarInit::Internal(grammar, lexer)
}

fn factory(env: &TokEnv) -> ParserFactory {
    let mut factory = ParserFactory::new(
        env,
        InferenceCapabilities {
            ff_tokens: true,
            conditional_ff_tokens: true,
            fork: true,
            backtrack: false,
        },
        &[],
    )
    .unwrap();
    factory.quiet();
    let limits = factory.limits_mut();
    limits.max_lexer_states = 4096;
    limits.step_lexer_fuel = 123_457;
    limits.step_max_items = 12_347;
    limits.precompute_large_lexemes = false;
    factory
}

fn consume(parser: &mut TokenParser, bytes: &[u8]) {
    for &byte in bytes {
        let mask = parser.compute_mask().unwrap();
        assert!(
            mask.is_allowed(u32::from(byte)),
            "byte {byte}: {}",
            parser.dump_state()
        );
        assert_eq!(parser.consume_token(u32::from(byte)).unwrap(), 0);
        parser.check_stop().unwrap();
    }
}

fn assert_clone_snapshot(source: &TokenParser, copied: &TokenParser) {
    assert_eq!(copied.is_fresh(), source.is_fresh());
    assert_eq!(copied.num_tokens(), source.num_tokens());
    assert_eq!(copied.final_bytes(), source.final_bytes());
    assert_eq!(copied.bytes_since(0), source.bytes_since(0));
    assert_eq!(copied.parser.get_bytes(), source.parser.get_bytes());
    assert_eq!(copied.captures(), source.captures());
    assert_eq!(copied.stop_reason(), source.stop_reason());
    assert_eq!(copied.error_message(), source.error_message());
    assert_eq!(copied.dbg_grammar, source.dbg_grammar);
    assert_eq!(
        copied.compute_mask_start_time,
        source.compute_mask_start_time
    );
    assert_eq!(copied.last_bias_time, source.last_bias_time);
    assert_eq!(
        serde_json::to_value(&copied.limits).unwrap(),
        serde_json::to_value(&source.limits).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&copied.inference_caps).unwrap(),
        serde_json::to_value(&source.inference_caps).unwrap()
    );
    assert_eq!(
        serde_json::to_value(copied.parser_stats()).unwrap(),
        serde_json::to_value(source.parser_stats()).unwrap()
    );
    assert_eq!(
        serde_json::to_value(copied.last_step_stats()).unwrap(),
        serde_json::to_value(source.last_step_stats()).unwrap()
    );
    assert_eq!(
        serde_json::to_value(copied.max_step_stats()).unwrap(),
        serde_json::to_value(source.max_step_stats()).unwrap()
    );
    assert!(Arc::ptr_eq(&copied.token_env, &source.token_env));
    assert!(Arc::ptr_eq(&copied.bias_computer, &source.bias_computer));
    assert!(std::ptr::eq(
        copied.parser.grammar(),
        source.parser.grammar()
    ));
    assert_eq!(copied.logger.buffer_level(), source.logger.buffer_level());
    assert_eq!(copied.logger.stderr_level(), source.logger.stderr_level());
    // Logger's existing Clone deliberately starts a fresh log buffer.
    assert!(copied.logger.get_buffer().is_empty());
}

#[test]
fn clones_preserve_committed_captures_configuration_and_cached_forcing_after_source_drop() {
    let env = ApproximateTokEnv::single_byte_env();
    let factory = factory(&env);
    let mut source = factory
        .create_parser_from_init_default(captured_grammar(false, false))
        .unwrap();
    source.start_without_prompt();
    consume(&mut source, b"record:marigold:1");
    assert_eq!(source.get_capture("word"), Some(&b"marigold"[..]));
    assert_eq!(source.final_bytes(), b"record:marigold:1");
    assert_eq!(source.force_bytes(), b"7\n");
    let forced = source.compute_ff_tokens();
    assert_eq!(forced, vec![u32::from(b'7'), u32::from(b'\n')]);
    assert!(!source.is_accepting());
    source.dbg_grammar = "frozen diagnostic grammar".to_owned();
    source.last_bias_time = Duration::from_micros(23);
    source.logger.set_buffer_level(2);
    source.logger.write_buffer("source-only log\n");
    assert!(!source.logger.get_buffer().is_empty());
    let counters = source.parser.perf_counters().tokenize_ff.get().2;
    let shared_counters = factory.perf_counters();

    let ordinary = source.clone();
    let independent = source.deep_clone();
    assert_clone_snapshot(&source, &ordinary);
    assert_clone_snapshot(&source, &independent);
    drop((source, factory, env));
    assert_eq!(shared_counters.tokenize_ff.get().2, counters);

    for mut copied in [ordinary, independent] {
        // A copied cache supplies the first forced token without repeating
        // tokenization. Both copies share the existing perf counter owner.
        let before_cached_mask = shared_counters.tokenize_ff.get().2;
        let mask = copied.compute_mask().unwrap();
        assert_eq!(mask.iter().collect::<Vec<_>>(), vec![u32::from(b'7')]);
        assert_eq!(shared_counters.tokenize_ff.get().2, before_cached_mask);
        assert_eq!(copied.consume_token(u32::from(b'7')).unwrap(), 0);
        assert!(!copied.check_stop().unwrap());
        consume(&mut copied, b"\n");
        assert_eq!(copied.final_bytes(), b"record:marigold:17\n");
        assert_eq!(copied.get_capture("word"), Some(&b"marigold"[..]));
        assert_eq!(copied.get_capture("answer"), Some(&b"17"[..]));
        assert_eq!(copied.stop_reason(), StopReason::NoExtension);
        let terminal = copied.deep_clone();
        assert_eq!(terminal.final_bytes(), copied.final_bytes());
        assert_eq!(terminal.captures(), copied.captures());
        assert_eq!(terminal.stop_reason(), StopReason::NoExtension);
    }
}

#[test]
fn ordinary_clones_share_lexer_growth_while_deep_clones_keep_independent_captured_histories() {
    let env = ApproximateTokEnv::single_byte_env();
    let factory = factory(&env);
    let mut source = factory
        .create_parser_from_init_default(captured_grammar(true, false))
        .unwrap();
    source.start_without_prompt();
    consume(&mut source, b"record:mar");
    let baseline = source.parser.lexer_stats().num_states;
    let mut ordinary = source.clone();
    let mut independent = source.deep_clone();
    assert_clone_snapshot(&source, &ordinary);
    assert_clone_snapshot(&source, &independent);

    consume(&mut ordinary, b"igold:");
    assert!(ordinary.parser.lexer_stats().num_states > baseline);
    assert_eq!(
        source.parser.lexer_stats().num_states,
        ordinary.parser.lexer_stats().num_states
    );
    assert_eq!(independent.parser.lexer_stats().num_states, baseline);
    assert_eq!(source.final_bytes(), b"record:mar");
    assert_eq!(independent.final_bytes(), b"record:mar");
    assert_eq!(ordinary.get_capture("word"), Some(&b"marigold"[..]));
    drop((factory, env));

    consume(&mut independent, b"ble:23\n");
    consume(&mut source, b"ket:17\n");
    consume(&mut ordinary, b"17\n");
    for (parser, text, word, answer) in [
        (
            &independent,
            &b"record:marble:23\n"[..],
            &b"marble"[..],
            &b"23"[..],
        ),
        (
            &source,
            &b"record:market:17\n"[..],
            &b"market"[..],
            &b"17"[..],
        ),
        (
            &ordinary,
            &b"record:marigold:17\n"[..],
            &b"marigold"[..],
            &b"17"[..],
        ),
    ] {
        assert_eq!(parser.final_bytes(), text);
        assert_eq!(parser.get_capture("word"), Some(word));
        assert_eq!(parser.get_capture("answer"), Some(answer));
        assert_eq!(parser.stop_reason(), StopReason::NoExtension);
    }
}

#[test]
fn accepting_clones_preserve_secondary_eos_and_terminal_state_after_source_retirement() {
    let base = ApproximateTokEnv::single_byte_env();
    let primary = base.tok_trie().eos_token();
    let secondary = primary - 1;
    let trie = base.tok_trie().with_eos_tokens(&[primary, secondary]);
    let env: TokEnv = Arc::new(TokEnvWithTrie::new(base, trie));
    let factory = factory(&env);
    let mut source = factory
        .create_parser_from_init_default(captured_grammar(true, true))
        .unwrap();
    source.start_without_prompt();
    consume(&mut source, b"marigold");
    assert!(source.is_accepting());
    assert_eq!(source.stop_reason(), StopReason::NotStopped);
    let ordinary = source.clone();
    let independent = source.deep_clone();
    drop((source, factory, env));

    for (mut copied, eos) in [(ordinary, primary), (independent, secondary)] {
        assert!(copied.is_accepting());
        let mask = copied.compute_mask().unwrap();
        assert!(mask.is_allowed(primary));
        assert!(mask.is_allowed(secondary));
        assert_eq!(copied.consume_token(eos).unwrap(), 0);
        assert!(copied.check_stop().unwrap());
        assert_eq!(copied.stop_reason(), StopReason::EndOfSentence);
        assert_eq!(copied.final_bytes(), b"marigold");
        assert_eq!(copied.num_tokens(), 9);
        let terminal = copied.deep_clone();
        drop(copied);
        let mut matcher = Matcher::new(Ok(terminal));
        assert!(matcher.is_stopped());
        assert_eq!(matcher.stop_reason(), StopReason::EndOfSentence);
        let eos_mask = matcher.compute_mask_or_eos().unwrap();
        let mut expected = vec![primary, secondary];
        expected.sort_unstable();
        assert_eq!(eos_mask.iter().collect::<Vec<_>>(), expected);
    }
}

#[test]
fn clones_preserve_freshness_remaining_token_allowance_and_terminal_error_payload() {
    let env = ApproximateTokEnv::single_byte_env();
    let factory = factory(&env);
    let mut grammar = TopLevelGrammar::from_json_schema(serde_json::json!({
        "type": "string",
        "pattern": "^[a-z]{3,12}$",
    }));
    grammar.max_tokens = Some(5);
    let fresh = factory.create_parser(grammar).unwrap();
    assert!(fresh.is_fresh());
    let mut source = fresh.deep_clone();
    assert!(source.is_fresh());
    drop(fresh);
    source.start_without_prompt();
    consume(&mut source, b"\"mar");
    assert_eq!(source.num_tokens(), 4);
    let ordinary = source.clone();
    let independent = source.deep_clone();
    drop((source, factory, env));

    for mut copied in [ordinary, independent] {
        assert!(!copied.is_fresh());
        assert_eq!(copied.consume_token(u32::from(b'a')).unwrap(), 0);
        assert_eq!(copied.num_tokens(), 5);
        let error = copied.consume_token(u32::from(b'r')).unwrap_err();
        assert!(error.to_string().contains("max_tokens_total reached"));
        assert_eq!(copied.stop_reason(), StopReason::MaxTokensTotal);
        assert_eq!(copied.final_bytes(), b"\"mara");
        let message = copied.error_message().unwrap();
        let mut terminal = copied.deep_clone();
        drop(copied);
        assert_eq!(terminal.error_message(), Some(message));
        assert_eq!(terminal.stop_reason(), StopReason::MaxTokensTotal);
        assert_eq!(terminal.num_tokens(), 5);
        assert!(terminal.compute_mask().is_err());
        assert!(terminal.consume_token(u32::from(b'r')).is_err());
        assert_eq!(terminal.final_bytes(), b"\"mara");
    }
}
