//! Construction-phase regressions; these do not claim a complete parser byte bound.

use super::*;
use crate::{api::LLGuidanceOptions, grammar_builder::GrammarBuilder};
use toktrie::ApproximateTokEnv;

fn limits(precompute: bool) -> ParserLimits {
    ParserLimits {
        precompute_large_lexemes: precompute,
        ..ParserLimits::default()
    }
}

fn grammar(first: &str, allow_initial_skip: bool) -> SharedGrammar {
    let limits = limits(false);
    let mut builder = GrammarBuilder::new(
        None,
        limits.clone(),
        derivre::ParserAllocationFunding::unenforced(),
    )
    .unwrap();
    builder
        .add_grammar(
            LLGuidanceOptions {
                allow_initial_skip,
                ..LLGuidanceOptions::default()
            },
            RegexAst::Literal(" ".to_owned()),
        )
        .unwrap();
    let first = builder.string(first).unwrap();
    let suffix = builder.string(":23\n").unwrap();
    let sequence = builder.join(&[first, suffix]).unwrap();
    builder.set_start_node(sequence).unwrap();
    SharedGrammar::new(
        builder
            .grammar
            .compile(
                builder.regex.spec,
                &limits,
                derivre::ParserAllocationFunding::unenforced(),
            )
            .unwrap(),
    ).unwrap()
}

fn new_state(
    grammar: &SharedGrammar,
    env: TokEnv,
    limits: ParserLimits,
) -> Result<(ParserState, Lexer)> {
    ParserState::new(
        env,
        grammar.clone(),
        limits,
        Arc::new(ParserPerfCounters::new()),
    )
}

fn construction_error(grammar: &SharedGrammar, env: TokEnv, limits: ParserLimits) -> anyhow::Error {
    match new_state(grammar, env, limits) {
        Ok(_) => panic!("construction unexpectedly published a parser"),
        Err(error) => error,
    }
}

fn assert_lexer_error(error: &anyhow::Error, phase: &str, original: &str) {
    let cause = error
        .downcast_ref::<ParserError>()
        .unwrap_or_else(|| panic!("missing original ParserError: {error:#}"));
    match cause {
        ParserError::LexerError(message) => assert_eq!(message, original),
        ParserError::ParserError(message) => panic!("lexer cause relabelled: {message}"),
    }
    assert_eq!(cause.stop_reason(), StopReason::LexerTooComplex);
    assert!(
        format!("{error:#}").contains(phase),
        "wrong construction phase: {error:#}"
    );
}

#[test]
fn initial_row_exhaustion_preserves_the_lexer_cause_after_successful_warming() {
    let grammar = grammar("marigold", true);
    let env = ApproximateTokEnv::single_byte_env();
    let mut warm_limits = limits(false);
    let warmed = Lexer::from(grammar.lexer_spec(), &mut warm_limits, false).unwrap();
    let warm_states = warmed.dfa.stats().num_states;
    assert!(!warmed.dfa.has_error());

    let (state, healthy) = new_state(&grammar, env.clone(), limits(false)).unwrap();
    assert_eq!(state.rows.len(), 1);
    assert_eq!(healthy.dfa.stats().num_states, warm_states + 1);
    let initial = state.rows[0].lexer_start_state;
    let skip = grammar.lexer_spec().skip_id(LexemeClass::ROOT);
    assert!(healthy.possible_lexemes(initial).contains(skip));

    // The exact warm-state capacity admits the entire Lexer::from path. Only
    // the initial Earley row's smaller lexeme selection needs another state.
    let mut capped = limits(false);
    capped.max_lexer_states = warm_states;
    let mut direct_limits = capped.clone();
    let mut direct = Lexer::from(grammar.lexer_spec(), &mut direct_limits, false).unwrap();
    let next = direct.start_state(healthy.possible_lexemes(initial));
    assert!(next.is_dead());
    assert_eq!(direct.dfa.stats().num_states, warm_states);
    let original = direct.dfa.get_error().unwrap();
    assert!(original.contains("too many states"));

    let error = construction_error(&grammar, env, capped);
    assert_lexer_error(&error, "initial parser row", &original);
}

#[test]
fn initial_skip_exhaustion_preserves_the_lexer_cause_after_a_valid_first_row() {
    let with_skip = grammar("marigold", true);
    let without_skip = grammar("marigold", false);
    let env = ApproximateTokEnv::single_byte_env();
    let (row_state, row_lexer) = new_state(&with_skip, env.clone(), limits(false)).unwrap();
    let row_states = row_lexer.dfa.stats().num_states;
    let (final_state, final_lexer) = new_state(&without_skip, env.clone(), limits(false)).unwrap();
    let skip = without_skip.lexer_spec().skip_id(LexemeClass::ROOT);
    assert!(row_lexer
        .possible_lexemes(row_state.rows[0].lexer_start_state)
        .contains(skip));
    assert!(!final_lexer
        .possible_lexemes(final_state.rows[0].lexer_start_state)
        .contains(skip));
    assert_eq!(final_lexer.dfa.stats().num_states, row_states + 1);

    let mut capped = limits(false);
    capped.max_lexer_states = row_states;
    let (row_state, mut row_lexer) = new_state(&with_skip, env.clone(), capped.clone()).unwrap();
    assert_eq!(row_state.rows.len(), 1);
    let mut possible = row_lexer
        .possible_lexemes(row_state.rows[0].lexer_start_state)
        .clone();
    possible.remove(skip);
    assert!(row_lexer.start_state(&possible).is_dead());
    assert_eq!(row_lexer.dfa.stats().num_states, row_states);
    let original = row_lexer.dfa.get_error().unwrap();

    let error = construction_error(&without_skip, env, capped);
    assert_lexer_error(&error, "initial skip selection", &original);
}

fn large_lexeme_fixture() -> (SharedGrammar, TokEnv) {
    let first = "marigold".repeat(150);
    let grammar = grammar(&first, true);
    let base = ApproximateTokEnv::single_byte_env();
    let mut words = base.tok_trie().all_tokens();
    // A real multi-byte token makes precomputation visit beyond the first-byte
    // transitions warmed by Lexer::from. The grammar itself remains nonempty.
    words.push(b"marigold".to_vec());
    let mut info = *base.tok_trie().info();
    info.vocab_size = words.len().try_into().unwrap();
    let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)));
    (grammar, env)
}

fn large_selection(grammar: &CGrammar, lexer: &mut Lexer) -> LexemeSet {
    let large: Vec<_> = grammar
        .lexer_spec()
        .lexemes
        .iter()
        .filter_map(|lexeme| (lexer.dfa.lexeme_weight(lexeme.idx).unwrap() > 1000).then_some(lexeme.idx))
        .collect();
    assert_eq!(
        large.len(),
        1,
        "fixture must enter actual optional precompute"
    );
    let mut selected = grammar.lexer_spec().alloc_lexeme_set();
    selected.add(large[0]);
    selected
}

#[test]
fn high_weight_precompute_exhaustion_preserves_the_original_state_limit() {
    let (grammar, env) = large_lexeme_fixture();
    let mut warm_limits = limits(true);
    let mut warm = Lexer::from(grammar.lexer_spec(), &mut warm_limits, false).unwrap();
    let selected = large_selection(&grammar, &mut warm);
    let warm_states = warm.dfa.stats().num_states;
    warm.dfa.set_fuel(warm_limits.initial_lexer_fuel);
    warm.precompute_for(env.tok_trie(), &selected).unwrap();
    assert!(!warm.dfa.has_error());
    assert!(warm.dfa.stats().num_states > warm_states);

    let (uncomputed, uncomputed_lexer) = new_state(&grammar, env.clone(), limits(false)).unwrap();
    let (computed, computed_lexer) = new_state(&grammar, env.clone(), limits(true)).unwrap();
    assert_eq!(uncomputed.perf_counters.precompute.get().2, 0);
    assert_eq!(computed.perf_counters.precompute.get().2, 1);
    assert!(computed_lexer.dfa.stats().num_states > uncomputed_lexer.dfa.stats().num_states);

    let mut capped = limits(true);
    capped.max_lexer_states = warm_states;
    let mut direct_limits = capped.clone();
    let mut direct = Lexer::from(grammar.lexer_spec(), &mut direct_limits, false).unwrap();
    direct.dfa.set_fuel(direct_limits.initial_lexer_fuel);
    let direct_error = direct
        .precompute_for(env.tok_trie(), &selected)
        .unwrap_err();
    assert_eq!(direct.dfa.stats().num_states, warm_states);
    let original = direct.dfa.get_error().unwrap();
    assert!(original.contains("too many states"));
    assert_lexer_error(&direct_error, "lexer error", &original);

    let error = construction_error(&grammar, env, capped);
    assert_lexer_error(&error, "precomputing large lexeme", &original);
}

#[test]
fn high_weight_precompute_fuel_exhaustion_is_not_relabelled_as_a_state_limit() {
    let (grammar, env) = large_lexeme_fixture();
    let mut generous = limits(true);
    let initial_fuel = generous.initial_lexer_fuel;
    let mut probe = Lexer::from(grammar.lexer_spec(), &mut generous, false).unwrap();
    let selected = large_selection(&grammar, &mut probe);

    // Pay the actual relevance pass exactly. Lexer warming retains its existing
    // fuel policy; the optional precompute then receives no remaining fuel.
    let mut exhausted = limits(true);
    exhausted.initial_lexer_fuel = initial_fuel - generous.initial_lexer_fuel;
    let mut direct_limits = exhausted.clone();
    let mut direct = Lexer::from(grammar.lexer_spec(), &mut direct_limits, false).unwrap();
    assert!(!direct.dfa.has_error());
    assert_eq!(direct_limits.initial_lexer_fuel, 0);
    direct.dfa.set_fuel(direct_limits.initial_lexer_fuel);
    let direct_error = direct
        .precompute_for(env.tok_trie(), &selected)
        .unwrap_err();
    let original = direct.dfa.get_error().unwrap();
    assert_eq!(original, "too many expressions constructed");
    assert_lexer_error(&direct_error, "lexer error", &original);
    assert!(direct.dfa.stats().num_states < exhausted.max_lexer_states);

    // Disabling the optional phase still constructs a healthy initial row with
    // this same initial-fuel configuration and the same nonzero grammar.
    let mut disabled = exhausted.clone();
    disabled.precompute_large_lexemes = false;
    let (_, healthy) = new_state(&grammar, env.clone(), disabled).unwrap();
    assert!(!healthy.dfa.has_error());

    let error = construction_error(&grammar, env, exhausted);
    assert_lexer_error(&error, "precomputing large lexeme", &original);
}

#[test]
fn prepared_earley_seed_uses_actual_scratch_predictions_and_retains_failed_destinations() {
    use std::cell::Cell;
    let grammar = grammar("marigold", true);
    let mut ordinary = Scratch::new(grammar.clone());
    ordinary.grammar_stack.push(GrammarStackNode::root());
    seed_predictions(&grammar, |item, param| {
        ordinary.add_unique_arg(item, "init", param);
        Ok::<(), std::convert::Infallible>(())
    })
    .unwrap();
    let calls = Cell::new(0usize);
    let spent = Cell::new(0usize);
    let seed = PreparedEarleySeed::prepare(grammar.clone(), &|bytes| {
        calls.set(calls.get() + 1);
        spent.set(spent.get() + bytes);
        Ok::<(), &'static str>(())
    })
    .unwrap();
    let actual = seed.scratch.as_ref().unwrap();
    assert!(spent.get() > 0);
    assert_eq!(actual.items, ordinary.items);
    assert_eq!(actual.item_args, ordinary.item_args);
    assert_eq!(actual.row_start, ordinary.row_start);
    assert_eq!(actual.row_end, ordinary.row_end);
    assert_eq!(
        actual.push_allowed_lexemes.iter().collect::<Vec<_>>(),
        ordinary.push_allowed_lexemes.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        actual.push_allowed_grammar_ids.as_slice(),
        ordinary.push_allowed_grammar_ids.as_slice()
    );
    assert_eq!(actual.grammar_stack.len(), 1);
    let total = calls.get();
    let mut escaped = None;
    for stop in 1..=total {
        let seen = Cell::new(0usize);
        let failure = PreparedEarleySeed::prepare(grammar.clone(), &|_| {
            seen.set(seen.get() + 1);
            if seen.get() == stop {
                Err("seed funding")
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert_eq!(seen.get(), stop);
        assert_eq!(failure.to_string(), "seed funding");
        escaped = Some(failure);
    }
    let retained = grammar.clone();
    drop((seed, ordinary, grammar));
    let failed = escaped.unwrap();
    assert!(retained.strong_count() > 1);
    let scratch = failed.prefix.scratch.as_ref().unwrap();
    assert_eq!(scratch.grammar_stack.len(), 1);
    assert!(!scratch.push_allowed_grammar_ids.as_slice().is_empty());
    drop(failed);
    assert_eq!(retained.strong_count(), 1);
}

#[test]
fn prepared_initial_agenda_preserves_nullable_capture_order_and_partial_chart_custody() {
    use crate::api::NodeProps;
    use std::cell::Cell;
    let config = limits(false);
    let mut builder = GrammarBuilder::new(
        None,
        config.clone(),
        derivre::ParserAllocationFunding::unenforced(),
    )
    .unwrap();
    builder
        .add_grammar(LLGuidanceOptions::default(), RegexAst::Literal(" ".into()))
        .unwrap();
    let word = builder.string("ivy").unwrap();
    let optional = builder.optional(word).unwrap();
    let first = builder.join_props(
        &[optional],
        NodeProps {
            capture_name: Some("nullable".into()),
            ..NodeProps::default()
        },
    ).unwrap();
    let second = builder.join_props(
        &[optional],
        NodeProps {
            capture_name: Some("nullable".into()),
            ..NodeProps::default()
        },
    ).unwrap();
    let append = builder.join_props(
        &[optional],
        NodeProps {
            capture_name: Some("__LIST_APPEND:nullable".into()),
            ..NodeProps::default()
        },
    ).unwrap();
    let suffix = builder.string("marigold").unwrap();
    let sequence = builder.join(&[first, second, append, append, suffix]).unwrap();
    builder.set_start_node(sequence).unwrap();
    let grammar = SharedGrammar::new(
        builder
            .grammar
            .compile(
                builder.regex.spec,
                &config,
                derivre::ParserAllocationFunding::unenforced(),
            )
            .unwrap(),
    ).unwrap();
    let (ordinary, _) = new_state(&grammar, ApproximateTokEnv::single_byte_env(), config).unwrap();
    let seed =
        PreparedEarleySeed::prepare(grammar.clone(), &|_| Ok::<_, &'static str>(())).unwrap();
    let calls = Cell::new(0);
    let owner = seed
        .close_initial_agenda(&|bytes| {
            assert!(bytes > 0);
            calls.set(calls.get() + 1);
            Ok::<_, &'static str>(())
        })
        .unwrap();
    assert!(owner.initial_agenda_closed());
    let actual = owner.scratch.as_ref().unwrap();
    assert_eq!(actual.items, ordinary.scratch.items);
    assert_eq!(actual.item_args, ordinary.scratch.item_args);
    assert_eq!(
        actual.grammar_stack.len(),
        ordinary.scratch.grammar_stack.len()
    );
    assert_eq!(actual.push_grm_top, ordinary.scratch.push_grm_top);
    let mut expected = ordinary.scratch.push_allowed_lexemes.clone();
    expected.remove(grammar.lexer_spec().skip_id(LexemeClass::ROOT));
    assert_eq!(
        owner.initial_lexemes().unwrap().iter().collect::<Vec<_>>(),
        expected.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        owner.initial_captures().collect::<Vec<_>>(),
        ordinary
            .captures
            .capture_list
            .iter()
            .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        owner
            .initial_captures()
            .filter(|(name, _)| *name == "nullable")
            .count(),
        1
    );
    assert!(
        owner
            .initial_captures()
            .filter(|(name, _)| name.starts_with("__LIST_APPEND:"))
            .count()
            >= 2
    );
    let mut escaped = None;
    for stop in 1..=calls.get() {
        let seed =
            PreparedEarleySeed::prepare(grammar.clone(), &|_| Ok::<_, &'static str>(()))
                .unwrap();
        let seen = Cell::new(0);
        let error = seed
            .close_initial_agenda(&|_| {
                seen.set(seen.get() + 1);
                if seen.get() == stop {
                    Err("agenda destination")
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(seen.get(), stop);
        assert_eq!(error.to_string(), "agenda destination");
        assert!(!error.prefix.initial_agenda_closed());
        assert!(error.prefix.scratch.as_ref().is_some());
        escaped = Some(error);
    }
    let retained = grammar.clone();
    drop((owner, ordinary, grammar));
    assert!(retained.strong_count() > 1);
    let error = escaped.unwrap();
    assert!(error.prefix.initial_item_count() > 0);
    drop(error);
    assert_eq!(retained.strong_count(), 1);
}

#[test]
fn prepared_initial_row_uses_actual_skip_states_and_retains_partial_publication() {
    use crate::earley::{regexvec::prepared::PreparedRegexVector, PreparedLexer};
    use derivre::raw::ParserAllocationFunding;
    use std::cell::Cell;
    fn lexer(grammar: &CGrammar, mut config: ParserLimits) -> PreparedLexer {
        let source = grammar.lexer_spec();
        let input = source
            .root_source_plan()
            .unwrap()
            .compile()
            .unwrap()
            .ordinary(source.regex_builder.exprset().clone())
            .unwrap();
        let funding = ParserAllocationFunding::prepare(|_| Ok::<_, std::io::Error>(())).unwrap();
        let vector =
            PreparedRegexVector::prepare_with_backing(input, &mut config, Some(funding), &|_| {
                Ok::<_, std::io::Error>(())
            })
            .unwrap();
        PreparedLexer::prepare(vector, &|_| Ok::<_, std::io::Error>(())).unwrap()
    }
    for skip in [true, false] {
        let source = grammar("marigold", skip);
        let (ordinary, expected_lexer) =
            new_state(&source, ApproximateTokEnv::single_byte_env(), limits(false)).unwrap();
        let mut actual_lexer = lexer(&source, limits(false));
        let seed =
            PreparedEarleySeed::prepare(source.clone(), &|_| Ok::<_, std::io::Error>(()))
                .unwrap()
                .close_initial_agenda(&|_| Ok::<_, std::io::Error>(()))
                .unwrap();
        let calls = Cell::new(0);
        let owner = seed
            .publish_initial_row(&mut actual_lexer, &|_| {
                calls.set(calls.get() + 1);
                Ok::<_, std::io::Error>(())
            })
            .unwrap();
        let actual = owner.initial_lexer_state().unwrap();
        let expected = ordinary.rows[0].lexer_start_state;
        assert_eq!(actual, expected);
        assert_eq!(owner.rows.len(), 1);
        assert_eq!(owner.row_infos.len(), 1);
        assert_eq!(owner.lexer_stack.len(), 1);
        assert_eq!(owner.rows[0].first_item, ordinary.rows[0].first_item);
        assert_eq!(owner.rows[0].last_item, ordinary.rows[0].last_item);
        assert_eq!(
            owner.rows[0].grammar_stack_ptr,
            ordinary.rows[0].grammar_stack_ptr
        );
        assert_eq!(
            owner.lexer_stack[0].lexer_state,
            ordinary.lexer_stack[0].lexer_state
        );
        assert_eq!(
            owner.row_infos[0].token_idx_start,
            ordinary.row_infos[0].token_idx_start
        );
        assert_eq!(
            owner.row_infos[0].start_byte_idx,
            ordinary.row_infos[0].start_byte_idx
        );
        assert_eq!(
            actual_lexer
                .vector()
                .state_desc(actual)
                .unwrap()
                .possible
                .iter()
                .collect::<Vec<_>>(),
            expected_lexer
                .possible_lexemes(expected)
                .iter()
                .collect::<Vec<_>>()
        );
        let skip_id = source.lexer_spec().skip_id(LexemeClass::ROOT);
        assert_eq!(
            actual_lexer
                .vector()
                .state_desc(actual)
                .unwrap()
                .possible
                .contains(skip_id),
            skip
        );
        if !skip {
            // Every reached row/mask/lexer allocation refuses before publishing
            // a completed initial row. Escaping failures retain the exact chart.
            let mut final_error = None;
            for stop in 1..=calls.get() {
                let mut lexical = lexer(&source, limits(false));
                let seed = PreparedEarleySeed::prepare(source.clone(), &|_| {
                    Ok::<_, std::io::Error>(())
                })
                .unwrap()
                .close_initial_agenda(&|_| Ok::<_, std::io::Error>(()))
                .unwrap();
                let seen = Cell::new(0);
                let error = seed
                    .publish_initial_row(&mut lexical, &|_| {
                        seen.set(seen.get() + 1);
                        if seen.get() == stop {
                            Err(std::io::Error::other("initial row destination"))
                        } else {
                            Ok(())
                        }
                    })
                    .unwrap_err();
                assert_eq!(seen.get(), stop);
                assert!(error.prefix.initial_lexer_state().is_none());
                assert!(error.to_string().contains("initial row destination"));
                final_error = Some(error);
            }
            let error = final_error.unwrap();
            assert_eq!(error.prefix.rows.len(), 1);
            assert_eq!(error.prefix.row_infos.len(), 1);
            assert!(error.prefix.initial_selection.is_some());
            let retained = source.clone();
            drop((owner, ordinary, source));
            assert!(retained.strong_count() > 1);
            drop(error);
            assert_eq!(retained.strong_count(), 1);
        }
    }
}

fn prepared_chart(source: &SharedGrammar) -> (PreparedEarleySeed, crate::earley::PreparedLexer) {
    use crate::earley::{regexvec::prepared::PreparedRegexVector, PreparedLexer};
    use derivre::raw::ParserAllocationFunding;
    let reserve = |_| Ok::<_, std::io::Error>(());
    let spec = source.lexer_spec();
    let input = spec
        .root_source_plan()
        .unwrap()
        .compile()
        .unwrap()
        .ordinary(spec.regex_builder.exprset().clone())
        .unwrap();
    let backing = ParserAllocationFunding::prepare(reserve).unwrap();
    let vector = PreparedRegexVector::prepare_with_backing(
        input,
        &mut limits(false),
        Some(backing),
        &reserve,
    )
    .unwrap();
    let mut lexer = PreparedLexer::prepare(vector, &reserve).unwrap();
    let seed = PreparedEarleySeed::prepare(source.clone(), &reserve)
        .unwrap()
        .close_initial_agenda(&reserve)
        .unwrap()
        .publish_initial_row(&mut lexer, &reserve)
        .unwrap();
    (seed, lexer)
}

#[test]
fn prepared_definitive_bytes_share_scan_skip_captures_and_failed_history_custody() {
    use crate::api::NodeProps;
    use std::cell::Cell;
    let config = limits(false);
    let mut builder = GrammarBuilder::new(
        None,
        config.clone(),
        derivre::ParserAllocationFunding::unenforced(),
    )
    .unwrap();
    builder
        .add_grammar(LLGuidanceOptions::default(), RegexAst::Literal(" ".into()))
        .unwrap();
    let word = builder.string("ivy").unwrap();
    let word = builder.join_props(
        &[word],
        NodeProps {
            capture_name: Some("word".into()),
            ..NodeProps::default()
        },
    ).unwrap();
    let suffix = builder.string(":23\n").unwrap();
    let sequence = builder.join(&[word, suffix]).unwrap();
    builder.set_start_node(sequence).unwrap();
    let grammar = SharedGrammar::new(
        builder
            .grammar
            .compile(
                builder.regex.spec,
                &config,
                derivre::ParserAllocationFunding::unenforced(),
            )
            .unwrap(),
    ).unwrap();
    let env = ApproximateTokEnv::single_byte_env();
    let (mut ordinary, ordinary_lexer) = new_state(&grammar, env.clone(), config).unwrap();
    ordinary.shared_box.lexer_opt = Some(ordinary_lexer);
    let (mut owner, mut lexer) = prepared_chart(&grammar);
    let mut scan_calls = 0;
    for &byte in b"ivy :23\n" {
        let calls = Cell::new(0);
        let expected = ordinary.try_push_byte_definitive(Some(byte));
        let (next, accepted, backtrack) = owner
            .push_byte(&mut lexer, env.tok_trie(), Some(byte), &|_| {
                calls.set(calls.get() + 1);
                Ok::<_, std::io::Error>(())
            })
            .unwrap();
        owner = next;
        assert_eq!((accepted, backtrack), expected);
        assert!(accepted);
        assert_eq!(owner.bytes(), ordinary.get_bytes());
        assert_eq!(
            owner.initial_captures().collect::<Vec<_>>(),
            ordinary
                .captures
                .capture_list
                .iter()
                .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            lexer
                .vector()
                .state_desc(owner.lexer_state().unwrap())
                .unwrap()
                .possible
                .iter()
                .collect::<Vec<_>>(),
            ordinary
                .lexer()
                .possible_lexemes(ordinary.lexer_state().lexer_state)
                .iter()
                .collect::<Vec<_>>()
        );
        if byte == b' ' {
            scan_calls = calls.get();
        }
    }
    let expected = ordinary.try_push_byte_definitive(None);
    let (next, accepted, backtrack) = owner
        .push_byte(&mut lexer, env.tok_trie(), None, &|_| {
            Ok::<_, std::io::Error>(())
        })
        .unwrap();
    owner = next;
    assert_eq!((accepted, backtrack), expected);
    assert_eq!(owner.bytes(), b"ivy :23\n");
    assert!(owner
        .initial_captures()
        .any(|(name, bytes)| name == "word" && bytes == b"ivy"));
    assert!(scan_calls > 4);
    let mut escaped = None;
    for stop in [1, scan_calls / 2, scan_calls] {
        let (mut partial, mut lexical) = prepared_chart(&grammar);
        for &byte in b"ivy" {
            let (next, accepted, backtrack) = partial
                .push_byte(&mut lexical, env.tok_trie(), Some(byte), &|_| {
                    Ok::<_, std::io::Error>(())
                })
                .unwrap();
            assert!(accepted);
            assert_eq!(backtrack, 0);
            partial = next;
        }
        let calls = Cell::new(0);
        let failure = partial
            .push_byte(&mut lexical, env.tok_trie(), Some(b' '), &|_| {
                calls.set(calls.get() + 1);
                if calls.get() == stop {
                    Err(std::io::Error::other("byte destination"))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(calls.get(), stop);
        assert!(failure.to_string().contains("byte destination"));
        assert_eq!(failure.prefix.bytes(), b"ivy");
        assert!(failure.prefix.initial_item_count() > 0);
        escaped = Some(failure);
    }
    let retained = grammar.clone();
    drop((owner, ordinary, grammar));
    assert!(retained.strong_count() > 1);
    drop(escaped);
    assert_eq!(retained.strong_count(), 1);
}

#[test]
fn prepared_trie_walk_restores_committed_history_and_retains_failed_mask_source() {
    use std::cell::Cell;
    let source = grammar("ivy", true);
    let words: Vec<Vec<u8>> = [
        b"".as_slice(),
        b"iv",
        b"ivy :23\n",
        b"ivy :24\n",
        b"ivy",
        b"ivy:",
        b":23\n",
        b"x",
        b" ",
    ]
    .into_iter()
    .map(|word| word.to_vec())
    .collect();
    let info = toktrie::TokRxInfo {
        vocab_size: words.len() as u32,
        tok_eos: 0,
        tok_bos: None,
        tok_pad: None,
        tok_unk: None,
        tok_end_of_turn: None,
    };
    let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)));
    let (mut ordinary, ordinary_lexer) = new_state(&source, env.clone(), limits(false)).unwrap();
    ordinary.shared_box.lexer_opt = Some(ordinary_lexer);
    let (mut owner, mut lexer) = prepared_chart(&source);
    let reserve = |_| Ok::<_, std::io::Error>(());
    for (bytes, start) in [
        (b"".as_slice(), b"".as_slice()),
        (b"iv", b"iv"),
        (b"y ", b""),
    ] {
        for &byte in bytes {
            assert!(ordinary.try_push_byte_definitive(Some(byte)).0);
            let (next, accepted, backtrack) = owner
                .push_byte(&mut lexer, env.tok_trie(), Some(byte), &reserve)
                .unwrap();
            assert!(accepted);
            assert_eq!(backtrack, 0);
            owner = next;
        }
        let expected_bytes = owner.bytes().to_vec();
        let expected_state = owner.lexer_state();
        let expected_rows = owner.row_infos.len();
        let expected_stack = owner.lexer_stack.len();
        let expected_grammar = owner.scratch.as_ref().unwrap().grammar_stack.len();
        let mut expected = env.tok_trie().alloc_token_set();
        env.tok_trie().add_bias(
            &mut ParserRecognizer {
                state: &mut ordinary,
            },
            &mut expected,
            start,
        );
        owner = owner
            .scan_token_mask(&mut lexer, env.tok_trie(), start, &reserve)
            .unwrap();
        assert_eq!(owner.scanned_token_mask().unwrap(), &expected);
        assert_eq!(owner.bytes(), expected_bytes);
        assert_eq!(owner.lexer_state(), expected_state);
        assert_eq!(owner.row_infos.len(), expected_rows);
        assert_eq!(owner.lexer_stack.len(), expected_stack);
        assert_eq!(
            owner.scratch.as_ref().unwrap().grammar_stack.len(),
            expected_grammar
        );
        assert!(owner.scratch.as_ref().unwrap().definitive);
        assert_eq!(owner.initial_captures().count(), 0);
        if expected_bytes.is_empty() {
            assert!(expected.is_allowed(2));
            assert!(!expected.is_allowed(3));
        }
    }
    let calls = Cell::new(0);
    owner = owner
        .scan_token_mask(&mut lexer, env.tok_trie(), &[], &|_| {
            calls.set(calls.get() + 1);
            Ok::<_, std::io::Error>(())
        })
        .unwrap();
    assert!(calls.get() > 4);
    let mut escaped = None;
    for stop in [1, 2, calls.get() / 2] {
        let (mut partial, mut lexical) = prepared_chart(&source);
        for &byte in b"ivy " {
            partial = partial
                .push_byte(&mut lexical, env.tok_trie(), Some(byte), &reserve)
                .unwrap()
                .0;
        }
        let seen = Cell::new(0);
        let failure = partial
            .scan_token_mask(&mut lexical, env.tok_trie(), &[], &|_| {
                seen.set(seen.get() + 1);
                if seen.get() == stop {
                    Err(std::io::Error::other("trie destination"))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(seen.get(), stop);
        assert!(failure.to_string().contains("trie destination"));
        assert_eq!(failure.prefix.bytes(), b"ivy ");
        if stop > 2 {
            assert!(failure.prefix.scanned_token_mask().is_some());
        }
        escaped = Some(failure);
    }
    let retained = source.clone();
    drop((owner, ordinary, source));
    assert!(retained.strong_count() > 1);
    drop(escaped);
    assert_eq!(retained.strong_count(), 1);
}

#[test]
fn prepared_tokens_share_forced_numeric_limits_stop_eos_and_failed_history() {
    use crate::api::{GenOptions, NodeProps};
    use std::cell::Cell;
    let words: Vec<Vec<u8>> = [
        b"".as_slice(),
        b"iv",
        b"y ",
        b":23",
        b"\n",
        b"hi",
        b"<X>",
        b"!",
        b"ivy",
        b"x",
        b"i",
        b"iiENDspill",
        b"\xff",
        b"ivy :23\n",
    ]
    .into_iter()
    .map(|word| word.to_vec())
    .collect();
    let info = toktrie::TokRxInfo {
        vocab_size: words.len() as u32,
        tok_eos: 0,
        tok_bos: None,
        tok_pad: None,
        tok_unk: None,
        tok_end_of_turn: None,
    };
    let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)));
    struct ActualTrie(TokEnv);
    impl BiasComputer for ActualTrie {
        fn compute_bias(&self, recognizer: &mut ParserRecognizer<'_>, start: &[u8]) -> SimpleVob {
            let trie = self.0.tok_trie();
            let mut mask = trie.alloc_token_set();
            trie.add_bias(recognizer, &mut mask, start);
            mask
        }
        fn trie(&self) -> &TokTrie {
            self.0.tok_trie()
        }
    }
    let raw_ids = [0, 6, u32::MAX];
    let mut expected_raw = vec![TokTrie::SPECIAL_TOKEN_MARKER];
    expected_raw.extend_from_slice(b"[0]<X>");
    expected_raw.push(TokTrie::SPECIAL_TOKEN_MARKER);
    expected_raw.extend_from_slice(b"[4294967295]");
    assert_eq!(
        env.tok_trie().raw_token_bytes(&raw_ids).collect::<Vec<_>>(),
        expected_raw
    );
    assert_eq!(
        env.tok_trie().raw_token_bytes_len(&raw_ids),
        Some(expected_raw.len())
    );
    assert_eq!(env.tok_trie().decode_raw(&raw_ids), expected_raw);
    let computer = ActualTrie(env.clone());
    let reserve = |_| Ok::<_, std::io::Error>(());
    let mut observed_calls = 0;
    for case in 0..4 {
        let config = limits(false);
        let source = if case == 0 {
            grammar("ivy", false)
        } else {
            let mut builder = GrammarBuilder::new(
                Some(env.tok_trie()),
                config.clone(),
                derivre::ParserAllocationFunding::unenforced(),
            )
            .unwrap();
            builder
                .add_grammar(LLGuidanceOptions::default(), RegexAst::Literal(" ".into()))
                .unwrap();
            let start = if case == 1 {
                let prefix = builder.string("hi").unwrap();
                let numeric = builder.token_ranges(vec![6..=6]).unwrap();
                let suffix = builder.string("!").unwrap();
                builder.join(&[prefix, numeric, suffix]).unwrap()
            } else {
                let body = builder
                    .gen(
                        GenOptions {
                            body_rx: RegexAst::Regex("i+".into()),
                            stop_rx: if case == 3 {
                                RegexAst::Literal("END".into())
                            } else {
                                RegexAst::EmptyString
                            },
                            stop_capture_name: (case == 3).then(|| "stop".into()),
                            lazy: None,
                            is_suffix: None,
                            temperature: None,
                        },
                        NodeProps {
                            max_tokens: (case == 2).then_some(2),
                            capture_name: Some("body".into()),
                            ..NodeProps::default()
                        },
                    )
                    .unwrap();
                if case == 2 {
                    let suffix = builder.string("!").unwrap();
                    builder.join(&[body, suffix]).unwrap()
                } else {
                    body
                }
            };
            builder.set_start_node(start).unwrap();
            SharedGrammar::new(
                builder
                    .grammar
                    .compile(
                        builder.regex.spec,
                        &config,
                        derivre::ParserAllocationFunding::unenforced(),
                    )
                    .unwrap(),
            ).unwrap()
        };
        let (mut ordinary, ordinary_lexer) =
            new_state(&source, env.clone(), config.clone()).unwrap();
        ordinary.shared_box.lexer_opt = Some(ordinary_lexer);
        let (mut owner, mut lexer) = prepared_chart(&source);
        if case == 0 {
            assert!(ordinary.try_push_byte_definitive(Some(b'i')).0);
            owner = owner
                .push_byte(&mut lexer, env.tok_trie(), Some(b'i'), &reserve)
                .unwrap()
                .0;
        }
        if case == 0 {
            let expected = env.tok_trie().chop_tokens(
                &mut ParserRecognizer {
                    state: &mut ordinary,
                },
                &[10],
            );
            assert_eq!(expected, (1, 1));
            let before = owner.bytes().to_vec();
            let (next, tokens, bytes) = owner
                .chop_tokens(&mut lexer, env.tok_trie(), &[10], &reserve)
                .unwrap();
            owner = next;
            assert_eq!((tokens, bytes), expected);
            assert_eq!(owner.bytes(), before);
            assert_eq!(owner.byte_token_indices(), ordinary.byte_to_token_idx);
        }
        let tokens: &[u32] = match case {
            0 => &[1, 2, 3, 4],
            1 => &[5, 6, 7],
            2 => &[10, 10, 7],
            _ => &[11],
        };
        for &token in tokens {
            for candidate in [token, 9, 0] {
                let bytes_before = owner.bytes().to_vec();
                let ordinals_before = owner.byte_token_indices().to_vec();
                let expected = ordinary.validate_tokens(&[candidate]);
                let (next, actual) = owner
                    .validate_tokens(&mut lexer, env.tok_trie(), &config, &[candidate], &reserve)
                    .unwrap();
                owner = next;
                assert_eq!(
                    actual, expected,
                    "validation case {case}, token {candidate}"
                );
                assert_eq!(owner.bytes(), bytes_before);
                assert_eq!(owner.byte_token_indices(), ordinals_before);
                assert_eq!(owner.can_advance(), Some(ordinary.can_advance()));
            }

            let expected_mask = ordinary.compute_bias(&computer, &[]);
            let fresh_calls = Cell::new(0);
            owner = owner
                .compute_token_mask(&mut lexer, env.tok_trie(), &config, &[], &|_| {
                    fresh_calls.set(fresh_calls.get() + 1);
                    Ok::<_, std::io::Error>(())
                })
                .unwrap();
            assert_eq!(
                owner.scanned_token_mask().unwrap(),
                &expected_mask,
                "mask case {case}, token {token}"
            );
            assert!(!owner.scanned_token_mask().unwrap().is_allowed(12));
            if case == 1 && token == 6 {
                assert!(expected_mask.is_allowed(6));
            }
            let pointer = owner.scanned_token_mask().unwrap().as_slice().as_ptr();
            let cached_calls = Cell::new(0);
            owner = owner
                .compute_token_mask(&mut lexer, env.tok_trie(), &config, &[], &|_| {
                    cached_calls.set(cached_calls.get() + 1);
                    Ok::<_, std::io::Error>(())
                })
                .unwrap();
            assert_eq!(
                owner.scanned_token_mask().unwrap().as_slice().as_ptr(),
                pointer
            );
            assert!(cached_calls.get() <= fresh_calls.get());
            let bytes = env.tok_trie().token(token);
            let expected = ordinary.apply_token(bytes, token).unwrap();
            ordinary.token_idx += 1;
            let calls = Cell::new(0);
            let (next, actual) = owner
                .apply_token(&mut lexer, env.tok_trie(), &config, bytes, token, &|_| {
                    calls.set(calls.get() + 1);
                    Ok::<_, std::io::Error>(())
                })
                .unwrap();
            owner = next;
            if case == 0 && token == 1 {
                observed_calls = calls.get();
            }
            assert_eq!(actual, expected, "case {case}, token {token}");
            if case == 3 {
                assert!(actual > 0);
            }
            assert_eq!(owner.bytes(), ordinary.bytes);
            assert_eq!(owner.byte_token_indices(), ordinary.byte_to_token_idx);
            assert_eq!(owner.row_infos.len(), ordinary.row_infos.len());
            assert_eq!(
                owner.initial_captures().collect::<Vec<_>>(),
                ordinary
                    .captures
                    .capture_list
                    .iter()
                    .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
                    .collect::<Vec<_>>()
            );
        }
        let before = owner.bytes().to_vec();
        let expected = ordinary.is_accepting();
        let (next, actual) = owner
            .is_accepting(&mut lexer, env.tok_trie(), &config, &reserve)
            .unwrap();
        owner = next;
        assert_eq!(actual, expected, "case {case} acceptance");
        assert!(actual);
        assert_eq!(owner.bytes(), before);
        let expected = ordinary.scan_eos();
        let (owner, actual) = owner
            .scan_eos(&mut lexer, env.tok_trie(), &config, &reserve)
            .unwrap();
        assert_eq!(actual, expected, "case {case} EOS");
        assert_eq!(owner.bytes(), ordinary.bytes);
        assert_eq!(owner.byte_token_indices(), ordinary.byte_to_token_idx);
    }
    let source = grammar("ivy", false);
    let (owner, mut lexer) = prepared_chart(&source);
    let calls = Cell::new(0);
    let (owner, accepted) = owner
        .validate_tokens(
            &mut lexer,
            env.tok_trie(),
            &limits(false),
            &[1, 2, 3, 4],
            &|_| {
                calls.set(calls.get() + 1);
                Ok::<_, std::io::Error>(())
            },
        )
        .unwrap();
    assert_eq!(accepted, 4);
    assert!(owner.bytes().is_empty());
    drop((owner, lexer, source));
    assert!(calls.get() > 4);
    for stop in [1, 2, calls.get() / 2] {
        let source = grammar("ivy", false);
        let retained = source.clone();
        let (owner, mut lexer) = prepared_chart(&source);
        let attempted = Cell::new(0);
        let error = owner
            .validate_tokens(
                &mut lexer,
                env.tok_trie(),
                &limits(false),
                &[1, 2, 3, 4],
                &|_| {
                    attempted.set(attempted.get() + 1);
                    if attempted.get() == stop {
                        Err(std::io::Error::other("token validation refused"))
                    } else {
                        Ok(())
                    }
                },
            )
            .unwrap_err();
        assert_eq!(attempted.get(), stop);
        assert!(error.to_string().contains("token validation refused"));
        assert!(error.prefix.bytes().is_empty());
        drop((source, lexer));
        assert!(retained.strong_count() > 1);
        drop(error);
        assert_eq!(retained.strong_count(), 1);
    }
    for stop in [1, 2] {
        let source = grammar("ivy", false);
        let retained = source.clone();
        let (chart, mut lexer) = prepared_chart(&source);
        let chart = chart
            .push_byte(&mut lexer, env.tok_trie(), Some(b'i'), &reserve)
            .unwrap()
            .0;
        let calls = Cell::new(0);
        let failure = chart
            .chop_tokens(&mut lexer, env.tok_trie(), &[10], &|_| {
                calls.set(calls.get() + 1);
                if calls.get() == stop {
                    Err(std::io::Error::other("token chop refused"))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(calls.get(), stop);
        assert!(failure.to_string().contains("token chop refused"));
        assert_eq!(failure.prefix.bytes(), b"i");
        drop((source, lexer));
        assert!(retained.strong_count() > 1);
        drop(failure);
        assert_eq!(retained.strong_count(), 1);
    }
    assert!(observed_calls > 4);
    for stop in [1, 2, observed_calls / 2] {
        let source = grammar("ivy", false);
        let retained = source.clone();
        let (owner, mut lexer) = prepared_chart(&source);
        let owner = owner
            .push_byte(&mut lexer, env.tok_trie(), Some(b'i'), &reserve)
            .unwrap()
            .0;
        let calls = Cell::new(0);
        let error = owner
            .apply_token(
                &mut lexer,
                env.tok_trie(),
                &limits(false),
                b"iv",
                1,
                &|_| {
                    calls.set(calls.get() + 1);
                    if calls.get() == stop {
                        Err(std::io::Error::other("token history destination"))
                    } else {
                        Ok(())
                    }
                },
            )
            .unwrap_err();
        assert_eq!(calls.get(), stop);
        assert!(error.to_string().contains("token history destination"));
        assert!(error.prefix.bytes().starts_with(b"i"));
        drop((source, lexer));
        assert!(retained.strong_count() > 1);
        drop(error);
        assert_eq!(retained.strong_count(), 1);
    }
    for token in [0, 9, 10, u32::MAX] {
        let mut expected = format!("X[{token}]").into_bytes();
        expected[0] = TokTrie::SPECIAL_TOKEN_MARKER;
        assert_eq!(force::Numeric::new(token).bytes(), expected);
    }
    let mut force_calls = 0;
    for numeric in [false, true] {
        let config = limits(false);
        let source = if numeric {
            let mut builder = GrammarBuilder::new(
                Some(env.tok_trie()),
                config.clone(),
                derivre::ParserAllocationFunding::unenforced(),
            )
            .unwrap();
            builder
                .add_grammar(LLGuidanceOptions::default(), RegexAst::NoMatch)
                .unwrap();
            let prefix = builder.string("hi").unwrap();
            let special = builder.token_ranges(vec![6..=6]).unwrap();
            let suffix = builder.string("!").unwrap();
            let root = builder.join(&[prefix, special, suffix]).unwrap();
            builder.set_start_node(root).unwrap();
            SharedGrammar::new(
                builder
                    .grammar
                    .compile(
                        builder.regex.spec,
                        &config,
                        derivre::ParserAllocationFunding::unenforced(),
                    )
                    .unwrap(),
            ).unwrap()
        } else {
            grammar("ivy", false)
        };
        let (mut ordinary, original_lexer) =
            new_state(&source, env.clone(), config.clone()).unwrap();
        ordinary.shared_box.lexer_opt = Some(original_lexer);
        let (mut owner, mut lexer) = prepared_chart(&source);
        let tokens: &[u32] = if numeric { &[5, 6, 7] } else { &[1, 2, 3, 4] };
        for (position, &token) in tokens.iter().enumerate() {
            ordinary.force_bytes();
            let seen = Cell::new(0);
            owner = owner
                .force_bytes(&mut lexer, env.tok_trie(), &config, &|_| {
                    seen.set(seen.get() + 1);
                    Ok::<_, std::io::Error>(())
                })
                .unwrap();
            if !numeric && position == 0 {
                force_calls = seen.get();
            }
            assert_eq!(owner.bytes(), ordinary.bytes);
            assert_eq!(
                owner.pending_token_bytes(),
                &ordinary.bytes[ordinary.byte_to_token_idx.len()..]
            );
            if numeric && position == 0 {
                assert!(owner
                    .pending_token_bytes()
                    .contains(&TokTrie::SPECIAL_TOKEN_MARKER));
            }
            let bytes = env.tok_trie().token(token);
            let expected = ordinary.apply_token(bytes, token).unwrap();
            ordinary.token_idx += 1;
            let (next, actual) = owner
                .apply_token(&mut lexer, env.tok_trie(), &config, bytes, token, &reserve)
                .unwrap();
            owner = next;
            assert_eq!(actual, expected);
            assert_eq!(owner.bytes(), ordinary.bytes);
            assert_eq!(owner.byte_token_indices(), ordinary.byte_to_token_idx);
        }
        assert!(owner.pending_token_bytes().is_empty());
    }
    assert!(force_calls > 4);
    for stop in [1, 2, force_calls / 2] {
        let source = grammar("ivy", false);
        let retained = source.clone();
        let (owner, mut lexer) = prepared_chart(&source);
        let seen = Cell::new(0);
        let failure = owner
            .force_bytes(&mut lexer, env.tok_trie(), &limits(false), &|_| {
                seen.set(seen.get() + 1);
                if seen.get() == stop {
                    Err(std::io::Error::other("forced-byte destination"))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(seen.get(), stop);
        assert!(failure.to_string().contains("forced-byte destination"));
        drop((source, lexer));
        assert!(retained.strong_count() > 1);
        drop(failure);
        assert_eq!(retained.strong_count(), 1);
    }
    let source = grammar("ivy", false);
    let (owner, mut lexer) = prepared_chart(&source);
    let mut limited = limits(false);
    limited.step_max_items = 0;
    let failure = owner
        .compute_token_mask(&mut lexer, env.tok_trie(), &limited, &[], &reserve)
        .unwrap_err();
    assert!(failure.to_string().contains("Earley step created"));
    assert!(failure.prefix.bytes().is_empty());
    assert!(failure.prefix.scanned_token_mask().is_some());
}

#[test]
fn prepared_token_session_shares_matcher_commit_stop_limits_and_failed_owner() {
    use crate::{api::TopLevelGrammar, ParserFactory};
    use std::cell::Cell;
    let words: Vec<Vec<u8>> = [b"".as_slice(), b"iv", b"y", b"!", b"x"]
        .into_iter()
        .map(|word| word.to_vec())
        .collect();
    let info = toktrie::TokRxInfo {
        vocab_size: words.len() as u32,
        tok_eos: 0,
        tok_bos: None,
        tok_pad: None,
        tok_unk: None,
        tok_end_of_turn: None,
    };
    let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)));
    let reserve = |_| Ok::<_, std::io::Error>(());
    let mut factory = ParserFactory::new_simple(&env).unwrap();
    factory.quiet();
    factory.limits_mut().precompute_large_lexemes = false;
    let request = || TopLevelGrammar::from_lark("start: \"iv\" \"y\" [\"!\"]".into());
    for eos in [false, true] {
        let mut ordinary = factory.create_parser(request()).unwrap();
        ordinary.start_without_prompt();
        let source = SharedGrammar::new(
            ordinary
                .parser
                .grammar()
                .source_copy_plan(&derivre::ParserAllocationFunding::unenforced())
                .unwrap()
                .compile()
                .unwrap(),
        ).unwrap();
        let (chart, mut lexer) = prepared_chart(&source);
        let config = factory.limits().clone();
        let mut owner = PreparedTokenParser::prepare(
            chart,
            &mut lexer,
            env.tok_trie(),
            &config,
            None,
            &reserve,
        )
        .unwrap();
        for token in [4, 1, 2, if eos { 0 } else { 3 }, 4] {
            let expected_mask = if ordinary.stopped() {
                env.tok_trie().eos_token_set()
            } else {
                ordinary.compute_mask().unwrap()
            };
            owner = owner
                .compute_mask(&mut lexer, env.tok_trie(), &config, None, &[], &reserve)
                .unwrap();
            assert_eq!(owner.token_mask().unwrap(), &expected_mask);
            assert_ne!(
                owner.token_mask().unwrap().as_slice().as_ptr(),
                owner
                    .chart()
                    .scanned_token_mask()
                    .unwrap()
                    .as_slice()
                    .as_ptr()
            );

            let expected =
                crate::tokenparser::progress::try_consume(&mut ordinary, &[token]).unwrap();
            let (next, actual) = owner
                .try_consume_tokens(&mut lexer, env.tok_trie(), &config, &[token], &reserve)
                .unwrap();
            owner = next;
            assert_eq!(actual, expected);
            assert_eq!(owner.token_mask().is_none(), actual != 0);
            assert_eq!(owner.tokens().len(), ordinary.num_tokens());
            assert_eq!(owner.bytes(), ordinary.final_bytes());
            assert_eq!(owner.stop_reason(), ordinary.stop_reason());
            let expected = ordinary.is_accepting();
            let (next, actual) = owner
                .is_accepting(&mut lexer, env.tok_trie(), &config, &reserve)
                .unwrap();
            owner = next;
            assert_eq!(actual, expected);
        }
        assert_eq!(
            owner.stop_reason(),
            if eos {
                StopReason::EndOfSentence
            } else {
                StopReason::NoExtension
            }
        );
        let lexer_copy_calls = Cell::new(0usize);
        let lexer_required = lexer.copy_required_bytes::<std::io::Error>().unwrap();
        let lexer_spent = Cell::new(0usize);
        let mut copied_lexer = lexer
            .try_copy(None, &|bytes| {
                lexer_copy_calls.set(lexer_copy_calls.get() + 1);
                let next = lexer_spent.get().checked_add(bytes).unwrap();
                if next > lexer_required {
                    return Err(std::io::Error::other("copy bound exceeded"));
                }
                lexer_spent.set(next);
                Ok::<_, std::io::Error>(())
            })
            .unwrap();
        assert_eq!(lexer_spent.get(), lexer_required);
        let short = Cell::new(0usize);
        let error = lexer
            .try_copy(None, &|bytes| {
                let next = short.get().checked_add(bytes).unwrap();
                if next >= lexer_required {
                    return Err(std::io::Error::other("one byte short"));
                }
                short.set(next);
                Ok(())
            })
            .unwrap_err();
        assert!(error.to_string().contains("one byte short"));
        drop(error);
        assert_eq!(
            copied_lexer.vector().transitions_attempted(),
            lexer.vector().transitions_attempted()
        );
        assert_eq!(copied_lexer.vector().fuel(), lexer.vector().fuel());
        assert_eq!(copied_lexer.vector().roots(), lexer.vector().roots());
        let copy_calls = Cell::new(0usize);
        let required = owner.copy_required_bytes::<std::io::Error>().unwrap();
        let spent = Cell::new(0usize);
        let copied = owner
            .try_copy(&|bytes| {
                copy_calls.set(copy_calls.get() + 1);
                let next = spent.get().checked_add(bytes).unwrap();
                if next > required {
                    return Err(std::io::Error::other("copy bound exceeded"));
                }
                spent.set(next);
                Ok::<_, std::io::Error>(())
            })
            .unwrap();
        assert_eq!(spent.get(), required);
        assert_eq!(
            owner.copy_required_bytes::<std::io::Error>(),
            Some(required)
        );
        let short = Cell::new(0usize);
        let error = owner
            .try_copy(&|bytes| {
                let next = short.get().checked_add(bytes).unwrap();
                if next >= required {
                    return Err(std::io::Error::other("one byte short"));
                }
                short.set(next);
                Ok(())
            })
            .unwrap_err();
        assert!(error.to_string().contains("one byte short"));
        drop(error);
        assert_eq!(copied.tokens(), owner.tokens());
        assert_eq!(copied.bytes(), owner.bytes());
        assert_ne!(copied.bytes().as_ptr(), owner.bytes().as_ptr());
        assert_ne!(copied.tokens().as_ptr(), owner.tokens().as_ptr());
        assert_eq!(copied.token_mask(), owner.token_mask());
        assert_ne!(
            copied.token_mask().unwrap().as_slice().as_ptr(),
            owner.token_mask().unwrap().as_slice().as_ptr()
        );
        assert!(std::ptr::eq(
            copied.chart().grammar(),
            owner.chart().grammar()
        ));
        assert_eq!(
            copied.chart().initial_captures().collect::<Vec<_>>(),
            owner.chart().initial_captures().collect::<Vec<_>>()
        );
        let original_tokens = owner.tokens().len();
        let copied = copied
            .rollback(&mut copied_lexer, env.tok_trie(), &config, 1, &reserve)
            .unwrap();
        assert_eq!(owner.tokens().len(), original_tokens);
        assert_eq!(copied.tokens().len(), original_tokens - 1);
        let copied = copied
            .compute_mask(
                &mut copied_lexer,
                env.tok_trie(),
                &config,
                None,
                &[],
                &reserve,
            )
            .unwrap();
        assert!(copied.token_mask().is_some());
        assert_eq!(owner.tokens().len(), original_tokens);
        drop((copied, copied_lexer));
        for cutoff in [1, lexer_copy_calls.get() / 2, lexer_copy_calls.get()] {
            let attempted = Cell::new(0usize);
            let error = lexer
                .try_copy(None, &|_| {
                    attempted.set(attempted.get() + 1);
                    if attempted.get() == cutoff {
                        Err(std::io::Error::other("lexer copy refused"))
                    } else {
                        Ok(())
                    }
                })
                .unwrap_err();
            assert_eq!(attempted.get(), cutoff);
            assert!(error.to_string().contains("lexer copy refused"));
        }
        for cutoff in [1, copy_calls.get() / 2, copy_calls.get()] {
            let attempted = Cell::new(0);
            let error = owner
                .try_copy(&|_| {
                    attempted.set(attempted.get() + 1);
                    if attempted.get() == cutoff {
                        Err(std::io::Error::other("token copy refused"))
                    } else {
                        Ok(())
                    }
                })
                .unwrap_err();
            assert_eq!(attempted.get(), cutoff);
            assert!(error.to_string().contains("token copy refused"));
            assert_eq!(owner.tokens().len(), original_tokens);
        }
        ordinary.rollback(1).unwrap();
        owner = owner
            .rollback(&mut lexer, env.tok_trie(), &config, 1, &reserve)
            .unwrap();
        assert!(owner.token_mask().is_none());
        assert_eq!(owner.bytes(), ordinary.final_bytes());
        assert_eq!(owner.tokens().len(), ordinary.num_tokens());
        assert_eq!(owner.stop_reason(), ordinary.stop_reason());
        let expected_mask = ordinary.compute_mask().unwrap();
        owner = owner
            .compute_mask(&mut lexer, env.tok_trie(), &config, None, &[], &reserve)
            .unwrap();
        assert_eq!(owner.token_mask().unwrap(), &expected_mask);
        ordinary.reset().unwrap();
        owner = owner
            .reset(&mut lexer, env.tok_trie(), &config, &reserve)
            .unwrap();
        assert!(owner.bytes().is_empty());
        assert!(owner.tokens().is_empty());
        assert_eq!(owner.bytes(), ordinary.final_bytes());
        let expected_mask = ordinary.compute_mask().unwrap();
        owner = owner
            .compute_mask(&mut lexer, env.tok_trie(), &config, None, &[], &reserve)
            .unwrap();
        assert_eq!(owner.token_mask().unwrap(), &expected_mask);
        let before_tokens = owner.tokens().len();
        let ordinary_error = ordinary
            .rollback(before_tokens + 1)
            .unwrap_err()
            .to_string();
        let error = owner
            .rollback(
                &mut lexer,
                env.tok_trie(),
                &config,
                before_tokens + 1,
                &reserve,
            )
            .unwrap_err();
        assert_eq!(error.to_string(), ordinary_error);
    }
    let mut capped = request();
    capped.max_tokens = Some(1);
    let mut ordinary = factory.create_parser(capped).unwrap();
    ordinary.start_without_prompt();
    let source = SharedGrammar::new(
        ordinary
            .parser
            .grammar()
            .source_copy_plan(&derivre::ParserAllocationFunding::unenforced())
            .unwrap()
            .compile()
            .unwrap(),
    ).unwrap();
    let (chart, mut lexer) = prepared_chart(&source);
    let config = factory.limits().clone();
    let owner = PreparedTokenParser::prepare(
        chart,
        &mut lexer,
        env.tok_trie(),
        &config,
        Some(1),
        &reserve,
    )
    .unwrap();
    let owner = owner
        .try_consume_tokens(&mut lexer, env.tok_trie(), &config, &[1], &reserve)
        .unwrap()
        .0;
    crate::tokenparser::progress::try_consume(&mut ordinary, &[1]).unwrap();
    assert!(
        crate::tokenparser::progress::try_consume(&mut ordinary, &[2])
            .unwrap_err()
            .to_string()
            .contains("max_tokens_total reached")
    );
    assert!(owner
        .try_consume_tokens(&mut lexer, env.tok_trie(), &config, &[2], &reserve)
        .unwrap_err()
        .to_string()
        .contains("max_tokens_total reached"));
    drop((ordinary, lexer, source));
    let ordinary = factory.create_parser(request()).unwrap();
    let source = SharedGrammar::new(
        ordinary
            .parser
            .grammar()
            .source_copy_plan(&derivre::ParserAllocationFunding::unenforced())
            .unwrap()
            .compile()
            .unwrap(),
    ).unwrap();
    let (chart, mut lexer) = prepared_chart(&source);
    let owner =
        PreparedTokenParser::prepare(chart, &mut lexer, env.tok_trie(), &config, None, &reserve)
            .unwrap();
    let calls = Cell::new(0);
    let owner = owner
        .try_consume_tokens(&mut lexer, env.tok_trie(), &config, &[1, 2], &|_| {
            calls.set(calls.get() + 1);
            Ok::<_, std::io::Error>(())
        })
        .unwrap()
        .0;
    assert_eq!(owner.bytes(), b"ivy");
    drop((ordinary, source, owner, lexer));
    assert!(calls.get() > 4);
    for stop in [1, 2, calls.get() / 2] {
        let ordinary = factory.create_parser(request()).unwrap();
        let source = SharedGrammar::new(
            ordinary
                .parser
                .grammar()
                .source_copy_plan(&derivre::ParserAllocationFunding::unenforced())
                .unwrap()
                .compile()
                .unwrap(),
        ).unwrap();
        let retained = source.clone();
        let (chart, mut lexer) = prepared_chart(&source);
        let owner = PreparedTokenParser::prepare(
            chart,
            &mut lexer,
            env.tok_trie(),
            &config,
            None,
            &reserve,
        )
        .unwrap();
        let attempted = Cell::new(0);
        let failure = owner
            .try_consume_tokens(&mut lexer, env.tok_trie(), &config, &[1, 2], &|_| {
                attempted.set(attempted.get() + 1);
                if attempted.get() == stop {
                    Err(std::io::Error::other("token session refused"))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(attempted.get(), stop);
        assert!(failure.to_string().contains("token session refused"));
        drop((source, lexer, ordinary));
        assert!(retained.strong_count() > 1);
        drop(failure);
        assert_eq!(retained.strong_count(), 1);
    }
}
