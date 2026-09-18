use super::*;
use derivre::{ParserAllocationFunding, RegexAst};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Debug)]
struct Refused;
impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("construction refused")
    }
}
impl std::error::Error for Refused {}

fn source() -> (Grammar, LexerSpec) {
    let mut lexer = LexerSpec::new(derivre::ParserAllocationFunding::unenforced()).unwrap();
    lexer.setup_lexeme_class(RegexAst::NoMatch).unwrap();
    let terminal = lexer
        .add_simple_literal("word".into(), "ready", false)
        .unwrap();
    let mut grammar = Grammar::new(None, derivre::ParserAllocationFunding::unenforced());
    let start = grammar
        .fresh_symbol_ext(
            "start",
            SymbolProps {
                is_start: true,
                parametric: true,
                ..Default::default()
            },
        )
        .unwrap();
    let word = grammar
        .fresh_symbol_ext(
            "word",
            SymbolProps {
                capture_name: Some("captured".into()),
                ..Default::default()
            },
        )
        .unwrap();
    grammar.make_terminal(word, terminal, &lexer).unwrap();
    grammar
        .add_rule_ext(start, ParamCond::True, vec![(word, ParamExpr::Null)])
        .unwrap();
    grammar
        .add_rule_ext(
            start,
            ParamCond::And(
                Box::new(ParamCond::GE(ParamRef::full(), ParamValue(2))),
                Box::new(ParamCond::NE(ParamRef::full(), ParamValue(4))),
            ),
            Vec::new(),
        )
        .unwrap();
    (grammar, lexer)
}

#[test]
fn compiled_tables_refuse_every_reached_allocation_and_keep_original_cause() {
    let (grammar, lexer) = source();
    let calls = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicUsize::new(usize::MAX));
    let funding = ParserAllocationFunding::prepare({
        let calls = calls.clone();
        let stop = stop.clone();
        move |_| {
            if calls.fetch_add(1, Ordering::SeqCst) == stop.load(Ordering::SeqCst) {
                Err(Refused)
            } else {
                Ok(())
            }
        }
    })
    .unwrap();
    calls.store(0, Ordering::SeqCst);
    let compiled = grammar
        .compile(lexer.clone(), &ParserLimits::default(), funding)
        .unwrap();
    let count = calls.load(Ordering::SeqCst);
    eprintln!("grammar construction reached {count} allocation requests");
    assert!(count > 0);
    let start = compiled.sym_data(compiled.start());
    assert_eq!(start.rules.len(), 1);
    for value in [0, 1, 2, 3, 4, 5] {
        assert_eq!(
            start
                .cond_nullable
                .iter()
                .any(|condition| condition.eval(ParamValue(value))),
            value >= 2 && value != 4
        );
    }
    drop(compiled);

    for limit in 0..count {
        let calls = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicUsize::new(usize::MAX));
        let custody = Arc::new(());
        let weak = Arc::downgrade(&custody);
        let funding = ParserAllocationFunding::prepare({
            let calls = calls.clone();
            let stop = stop.clone();
            move |_| {
                let _custody = &custody;
                if calls.fetch_add(1, Ordering::SeqCst) == stop.load(Ordering::SeqCst) {
                    Err(Refused)
                } else {
                    Ok(())
                }
            }
        })
        .unwrap();
        calls.store(0, Ordering::SeqCst);
        stop.store(limit, Ordering::SeqCst);
        let error = grammar
            .compile(lexer.clone(), &ParserLimits::default(), funding)
            .unwrap_err();
        assert!(
            error.chain().any(|cause| cause.is::<Refused>()),
            "allocation {limit}: {error}"
        );
        assert!(weak.upgrade().is_some());
        drop(error);
        assert!(weak.upgrade().is_none());
    }
}

#[cfg(feature = "lark")]
#[test]
fn grammar_construction_keeps_each_funding_refusal_distinct_from_syntax() {
    fn compile(
        funding: ParserAllocationFunding,
    ) -> Result<CGrammar, crate::earley::GrammarCompilationError> {
        crate::api::GrammarInit::Serialized(crate::api::TopLevelGrammar::from_lark(
            "start: word word\nword: \"a\" | \"b\"".into(),
        ))
        .to_cgrammar(
            None,
            &mut crate::Logger::new(0, 0),
            ParserLimits::default(),
            &[],
            funding,
        )
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let funding = ParserAllocationFunding::prepare({
        let calls = calls.clone();
        move |_| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok::<_, Refused>(())
        }
    })
    .unwrap();
    calls.store(0, Ordering::SeqCst);
    let grammar = compile(funding).unwrap();
    let count = calls.load(Ordering::SeqCst);
    eprintln!("grammar construction reached {count} allocation requests");
    assert!(!grammar.rules_of(grammar.start()).is_empty());
    drop(grammar);

    for limit in 0..count {
        let calls = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicUsize::new(usize::MAX));
        let custody = Arc::new(());
        let weak = Arc::downgrade(&custody);
        let funding = ParserAllocationFunding::prepare({
            let calls = calls.clone();
            let stop = stop.clone();
            move |_| {
                let _custody = &custody;
                if calls.fetch_add(1, Ordering::SeqCst) == stop.load(Ordering::SeqCst) {
                    Err(Refused)
                } else {
                    Ok(())
                }
            }
        })
        .unwrap();
        calls.store(0, Ordering::SeqCst);
        stop.store(limit, Ordering::SeqCst);
        let error = compile(funding).unwrap_err();
        assert!(error.is_storage_failure(), "allocation {limit}: {error}");
        assert!(error.funding_failure().is_some());
        assert!(weak.upgrade().is_some());
        drop(error);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn json_schema_construction_never_swallows_a_reached_funding_refusal() {
    fn compile(
        funding: ParserAllocationFunding,
    ) -> Result<CGrammar, crate::earley::GrammarCompilationError> {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "a": { "type": "integer", "minimum": 2, "maximum": 8 } },
            "patternProperties": { "^a$": { "type": "integer", "minimum": 1 } },
            "required": ["a"],
            "additionalProperties": false,
        });
        crate::api::GrammarInit::Serialized(crate::api::TopLevelGrammar::from_json_schema(schema))
            .to_cgrammar(
                None,
                &mut crate::Logger::new(0, 0),
                ParserLimits::default(),
                &[],
                funding,
            )
    }
    fn account(calls: &Arc<AtomicUsize>, stop: usize, custody: Arc<()>) -> ParserAllocationFunding {
        let callback_calls = calls.clone();
        let funding = ParserAllocationFunding::prepare(move |_| {
            let _custody = &custody;
            if callback_calls.fetch_add(1, Ordering::SeqCst) == stop {
                Err(Refused)
            } else {
                Ok(())
            }
        })
        .unwrap();
        calls.store(0, Ordering::SeqCst);
        funding
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let grammar = compile(account(&calls, usize::MAX, Arc::new(()))).unwrap();
    let count = calls.load(Ordering::SeqCst);
    eprintln!("JSON schema construction reached {count} allocation requests");
    assert!(!grammar.rules_of(grammar.start()).is_empty());
    drop(grammar);
    for stop in 0..count {
        let custody = Arc::new(());
        let weak = Arc::downgrade(&custody);
        // The callback owner construction is always funded; refusal starts at
        // the selected reached worker request, after resetting the counter.
        let calls = Arc::new(AtomicUsize::new(usize::MAX));
        let funding = account(&calls, stop, custody);
        let error = compile(funding).unwrap_err();
        assert!(error.is_storage_failure(), "allocation {stop}: {error}");
        assert!(
            error.funding_failure().is_some(),
            "allocation {stop}: {error}"
        );
        assert!(weak.upgrade().is_some());
        drop(error);
        assert!(weak.upgrade().is_none(), "allocation {stop}");
    }
}

#[test]
fn invalid_local_reference_remains_a_typed_semantic_diagnostic() {
    let schema = serde_json::json!({ "$ref": "#/$defs/missing" });
    let error = crate::api::GrammarInit::Serialized(crate::api::TopLevelGrammar::from_json_schema(schema))
        .to_cgrammar(None, &mut crate::Logger::new(0, 0), ParserLimits::default(), &[], ParserAllocationFunding::unenforced())
        .unwrap_err();
    assert!(!error.is_storage_failure());
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    let mut found = false;
    while let Some(error) = cause {
        found |= matches!(error.downcast_ref::<referencing::Error>(), Some(referencing::Error::PointerToNowhere { .. }));
        cause = error.source();
    }
    assert!(found);
}
