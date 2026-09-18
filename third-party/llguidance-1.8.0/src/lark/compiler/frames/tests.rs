use super::*;
use crate::api::ParserLimits;
use std::{error::Error, sync::{Arc, atomic::{AtomicUsize, Ordering}}};

#[derive(Debug)]
struct Refused(usize);
impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "Lark emitter cut {}", self.0) }
}
impl Error for Refused {}

const NESTED: &str = "start: outer \"!\"\nouter: inner [inner]\ninner: WORD\nWORD: ((\"a\" | \"b\")+ & /[ab]+/)\n%ignore \" \"\n";

fn emit(cut: usize) -> (Result<GrammarResult<'static>>, usize, std::sync::Weak<()>) {
    // This filter starts at the emitter's actual parsed-source boundary; the
    // public entry test below also runs the tokenizer/parser under the payer.
    let parsed = parse_lark(NESTED, ParserAllocationFunding::unenforced()).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let refusal = Arc::new(AtomicUsize::new(usize::MAX));
    let owner = Arc::new(());
    let weak = Arc::downgrade(&owner);
    let (c, r) = (calls.clone(), refusal.clone());
    let funding = ParserAllocationFunding::prepare(move |_| {
        let _owner = &owner;
        let n = c.fetch_add(1, Ordering::SeqCst);
        if n == r.load(Ordering::SeqCst) { Err(Refused(n)) } else { Ok(()) }
    }).unwrap();
    let builder = GrammarBuilder::new(None, ParserLimits::default(), funding.clone()).unwrap();
    let scope = Scope::new(&funding).unwrap();
    calls.store(0, Ordering::SeqCst);
    refusal.store(cut, Ordering::SeqCst);
    let result = compile_lark(builder, parsed, &scope);
    drop(scope);
    drop(funding);
    (result, calls.load(Ordering::SeqCst), weak)
}

#[test]
fn recursive_lark_emitter_refuses_every_reached_frame_and_heap_before_publishing() {
    let (accepted, reached, weak) = emit(usize::MAX);
    let accepted = accepted.unwrap();
    assert!(accepted.builder.grammar.num_symbols() > 2);
    assert!(reached > 20);
    assert!(weak.upgrade().is_some());
    drop(accepted);
    assert!(weak.upgrade().is_none());
    for cut in 0..reached {
        let (result, calls, weak) = emit(cut);
        let error = match result { Err(error) => error, Ok(_) => panic!("reached cut {cut} was ignored") };
        assert_eq!(calls, cut + 1, "stop at first refused emitter operation");
        let mut cause: &dyn Error = &error;
        loop {
            if let Some(original) = cause.downcast_ref::<Refused>() { assert_eq!(original.0, cut); break; }
            cause = cause.source().expect("original typed compiler refusal");
        }
        assert!(weak.upgrade().is_some());
        drop(error);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn public_funded_lark_emission_preserves_recursive_rules_tokens_and_nested_grammars() {
    use crate::{api::{GrammarInit, TopLevelGrammar}, earley::SharedGrammar, ParserFactory, Logger};
    use toktrie::{ApproximateTokEnv, InferenceCapabilities};
    let env = ApproximateTokEnv::single_byte_env();
    let factory = ParserFactory::new(&env, InferenceCapabilities {
        ff_tokens: false, backtrack: false, conditional_ff_tokens: false, fork: false,
    }, &[]).unwrap();
    let cases: &[(&str, &[(&str, bool)])] = &[
        (NESTED, &[("ab!", true), ("a b!", true), ("c!", false), ("ab", false)]),
        ("start: \"x\" nested \"!\"\nnested: %lark {\nstart: (\"b\" | \"c\")\n}\n", &[("xb!", true), ("xc!", true), ("xd!", false), ("xb", false)]),
    ];
    for (source, words) in cases {
        for enforced in [false, true] {
            let calls = Arc::new(AtomicUsize::new(0));
            let count = calls.clone();
            let funding = if enforced { ParserAllocationFunding::prepare(move |_| {
                count.fetch_add(1, Ordering::SeqCst); Ok::<_, Refused>(())
            }).unwrap() } else { ParserAllocationFunding::unenforced() };
            let grammar = GrammarInit::Serialized(TopLevelGrammar::from_lark((*source).into()))
                .to_cgrammar(Some(env.tok_trie()), &mut Logger::new(0, 0), ParserLimits::default(), &[], funding).unwrap();
            if enforced { assert!(calls.load(Ordering::SeqCst) > 20); }
            let grammar = SharedGrammar::new(grammar).unwrap();
            for (word, expected) in *words {
                let mut parser = factory.create_parser_from_compiled(grammar.clone(), None).unwrap();
                parser.start_without_prompt();
                let valid = word.bytes().all(|byte| parser.consume_token(byte as u32).is_ok());
                assert_eq!(valid && parser.is_accepting(), *expected, "source={source:?} input={word:?} enforced={enforced}");
            }
        }
    }
}
